"""PyTorch CPU/CUDA backend for the fixed paired SAF fits; never imports MLX.

The registered procedure is shared with MLX through ``train``: the same
model shape, seeds, batch/physics order, loss and Adam settings. Primary
Torch training is float64 on CPU. Neutral ``layers.i`` NPZ weights retain
the trained float64 values; CPU64 scoring never downcasts them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .models import make_model
from .registration import write_once
from .thermo import TorchOps
from .train import export_member, fit_loop, peak_rss_bytes, registered_loss

LAYER_SHAPES = {f"layers.{i}.{kind}": shape for i, (incoming, outgoing) in
                enumerate(((12, 128), (128, 128), (128, 128), (128, 128), (128, 494)))
                for kind, shape in (("weight", (outgoing, incoming)), ("bias", (outgoing,)))}


def resolve_device(device="auto"):
    """Primary auto is CPU; an explicitly requested CUDA device must exist."""
    import torch
    if device == "auto":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device cuda was requested, but torch {torch.__version__} "
                               f"(CUDA build {torch.version.cuda}) reports no usable CUDA device; "
                               "install a CUDA-enabled torch build or use --device cpu")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unknown torch device {device!r}; use auto, cpu or cuda")


def safetensors_bytes(tensors):
    """Canonical safetensors preserving each native F32/F64 tensor dtype."""
    import numpy as np
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        values = np.asarray(tensors[name])
        if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Checkpoint tensors must be float32 or float64")
        dtype = "<f8" if values.dtype.itemsize == 8 else "<f4"
        data = np.ascontiguousarray(values, dtype=dtype).tobytes()
        header[name] = {"dtype": "F64" if values.dtype.itemsize == 8 else "F32", "shape": list(values.shape),
                        "data_offsets": [offset, offset+len(data)]}
        blobs.append(data); offset += len(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" "*(-len(text) % 8)
    return len(text).to_bytes(8, "little")+text+b"".join(blobs)


def read_safetensors(data):
    import numpy as np
    length = int.from_bytes(data[:8], "little")
    header, body = json.loads(data[8:8+length]), data[8+length:]
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = {"F32":"<f4", "F64":"<f8"}.get(entry["dtype"])
        if dtype is None:
            raise ValueError("Checkpoint tensors must be float32 or float64")
        start, end = entry["data_offsets"]
        tensors[name] = np.frombuffer(body[start:end], dtype=dtype).reshape(entry["shape"]).copy()
    return tensors


class TorchBackend:
    """Same registered fit as MLXBackend, on an explicit torch device."""

    def __init__(self, device="auto"):
        import torch
        self.torch = torch
        self.device = resolve_device(device)
        self.requested_device = device
        if self.device == "cuda":
            # Deterministic cuBLAS needs this before the first CUDA matmul.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
        self.ops = TorchOps(self.device)
        from simulation.ml_backend import get_backend
        self.backend = get_backend("torch", device=self.device, dtype="float64")

    def start(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def info(self):
        torch = self.torch
        record = {"backend": "torch", "requested_device": self.requested_device, "device": self.device_label(),
                  "dtype": "float64", "score_backend":"torch", "score_device":"cpu", "score_dtype":"float64", "version": torch.__version__, "cuda_build": torch.version.cuda,
                  "cpu_threads": torch.get_num_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "initialization": "shared backend seeded U[-1/sqrt(fan_in),+1/sqrt(fan_in)]",
                  "optimizer": "torch.optim.Adam lr=0.001 betas=(0.9,0.999) eps=1e-8 weight_decay=0 amsgrad=False"}
        if self.device == "cuda":
            record.update(device_name=torch.cuda.get_device_name(torch.cuda.current_device()),
                          cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                          matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        return record

    def device_label(self):
        # model.to("cuda") places weights on torch.cuda.current_device(); the
        # label must name that same actual device, not a hardcoded index.
        return str(self.torch.device(self.device, self.torch.cuda.current_device())
                   if self.device == "cuda" else self.torch.device("cpu"))

    def memory(self):
        cuda = self.device == "cuda"
        return {"peak_torch_cuda_bytes": int(self.torch.cuda.max_memory_allocated()) if cuda else None,
                "process_peak_rss_bytes": peak_rss_bytes()}

    def fit(self, arm, seed, arrays, states, widths, thermo, check, log, *, epochs=2000, physics_rows=2048):
        import numpy as np
        torch = self.torch
        model = make_model(seed, "torch", self.device)
        optimizer = self.backend.adam(model, lr=.001, betas=(.9, .999), eps=1e-8,
                                      weight_decay=0.0)
        X, target, Y, P, Pminus, Pplus = arrays
        def constant(value):
            return torch.as_tensor(value, dtype=torch.float64, device=self.device)
        def batch(index, pindex):
            values = [constant(value) for value in
                      (X[index], target[index], Y[index], P[pindex], Pminus[pindex], Pplus[pindex])]
            state = {key: constant(value[pindex]) for key, value in states.items()}
            value = self.backend.step(model, optimizer,
                lambda current: registered_loss(current, arm, thermo, self.ops,
                                                *values, state, constant(widths[pindex])))
            return float(value.detach())
        return model, fit_loop(len(X), seed, batch, check, log, epochs=epochs, physics_rows=physics_rows)

    def save(self, model, output, arm, N, seed, metadata):
        import numpy as np
        output = Path(output)
        base = output / f"models/{arm}/N{N}/seed{seed}"
        tensors = {key: value.detach().to("cpu").numpy().copy()
                   for key, value in model.state_dict().items()}
        if {key: value.shape for key, value in tensors.items()} != LAYER_SHAPES:
            raise ValueError("Torch model is not the registered canonical layer layout")
        checkpoint = safetensors_bytes(tensors)
        restored = read_safetensors(checkpoint)
        if restored.keys() != tensors.keys() or not all(np.array_equal(restored[key], tensors[key], equal_nan=True) for key in tensors):
            raise ValueError("Checkpoint does not round-trip the trained float64 weights")
        target = base.with_suffix(".safetensors")
        write_once(target, checkpoint)
        if any(value.dtype != np.float64 for value in tensors.values()):
            raise ValueError("Primary Torch checkpoint unexpectedly lost float64 precision")
        params = {key: tensors[key] for key in LAYER_SHAPES}
        return export_member(output, base, target, params, metadata)
