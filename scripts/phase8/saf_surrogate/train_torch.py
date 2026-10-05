"""PyTorch CPU/CUDA backend for the fixed paired SAF fits; never imports MLX.

The registered procedure is shared with MLX through ``train``: the same
model shape, seeds, batch/physics order, loss, Adam settings and float32
training dtype. Weights are exported as the canonical ``layers.i`` float32
safetensors checkpoint plus the CPU64 NPZ that scoring reads.
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
    """auto = CUDA when available, else CPU. Requested CUDA must exist."""
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
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
    """Canonical little-endian F32 safetensors, the layout MLX save_weights uses."""
    import numpy as np
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        data = np.ascontiguousarray(tensors[name], dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(np.shape(tensors[name])),
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
        if entry["dtype"] != "F32":
            raise ValueError("Checkpoint tensors must be float32")
        start, end = entry["data_offsets"]
        tensors[name] = np.frombuffer(body[start:end], dtype="<f4").reshape(entry["shape"]).copy()
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

    def start(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def info(self):
        torch = self.torch
        record = {"backend": "torch", "requested_device": self.requested_device, "device": self.device_label(),
                  "dtype": "float32", "version": torch.__version__, "cuda_build": torch.version.cuda,
                  "cpu_threads": torch.get_num_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "initialization": "torch.Generator(cpu).manual_seed(seed) U[-1/sqrt(fan_in),+1/sqrt(fan_in)]",
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
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0.0, amsgrad=False)
        X, target, Y, P, Pminus, Pplus = arrays
        def constant(value):
            # Normalize on CPU64 before the explicit float32 cast, as in MLX.
            return torch.as_tensor(value.astype(np.float32), device=self.device)
        def batch(index, pindex):
            values = [constant(value) for value in
                      (X[index], target[index], Y[index], P[pindex], Pminus[pindex], Pplus[pindex])]
            state = {key: constant(value[pindex]) for key, value in states.items()}
            optimizer.zero_grad(set_to_none=True)
            value = registered_loss(model, arm, thermo, self.ops, *values, state, constant(widths[pindex]))
            value.backward()
            optimizer.step()
            return float(value.detach())
        return model, fit_loop(len(X), seed, batch, check, log, epochs=epochs, physics_rows=physics_rows)

    def save(self, model, output, arm, N, seed, metadata):
        import numpy as np
        output = Path(output)
        base = output / f"models/{arm}/N{N}/seed{seed}"
        tensors = {key: value.detach().to("cpu").numpy().astype(np.float32)
                   for key, value in model.state_dict().items()}
        if {key: value.shape for key, value in tensors.items()} != LAYER_SHAPES:
            raise ValueError("Torch model is not the registered canonical layer layout")
        checkpoint = safetensors_bytes(tensors)
        restored = read_safetensors(checkpoint)
        if restored.keys() != tensors.keys() or not all(np.array_equal(restored[key], tensors[key], equal_nan=True) for key in tensors):
            raise ValueError("Checkpoint does not round-trip the trained float32 weights")
        target = base.with_suffix(".safetensors")
        write_once(target, checkpoint)
        params = {key: tensors[key].astype(np.float64) for key in LAYER_SHAPES}
        return export_member(output, base, target, params, metadata)
