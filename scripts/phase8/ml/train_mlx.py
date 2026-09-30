"""Generic MLX training loop (Track D3 scaffolding). Import only inside catjet-mlx.

- Adam (mlx.optimizers.Adam with bias_correction=True, i.e. standard Adam;
  MLX's default is bias_correction=False).
- One mx.compile'd step; model, optimizer and RNG state are declared as
  compile inputs/outputs so parameter updates persist.
- Deterministic: mx.random.seed(seed) plus a numpy Generator(seed) for batch
  order; init comes from the model's own keyed init.
- Early stopping is off by default (patience=None).
- Checkpoints: safetensors weights + JSON sidecar holding the sha256 of the
  weight file; load verifies it.

Exercised only on synthetic toy data. No model is trained on project data
before gate G2.
"""
from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .spec import MLPSpec


def mse_loss(model, x, y):
    return mx.mean((model(x) - y) ** 2)


def train(model: nn.Module, X, Y, *, loss_fn: Callable = mse_loss, epochs: int = 100,
          lr: float = 1e-3, batch_size: Optional[int] = None, seed: int = 0,
          patience: Optional[int] = None, X_val=None, Y_val=None,
          compile: bool = True) -> Dict:
    """Train in place. X, Y: non-dimensional float32-castable arrays.

    batch_size None = full batch. patience None = no early stopping (the
    default); with patience and validation data, the best-validation weights
    are restored at the end.
    """
    mx.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))
    X = mx.array(np.asarray(X, dtype=np.float32))
    Y = mx.array(np.asarray(Y, dtype=np.float32))
    has_val = X_val is not None and Y_val is not None
    if has_val:
        X_val = mx.array(np.asarray(X_val, dtype=np.float32))
        Y_val = mx.array(np.asarray(Y_val, dtype=np.float32))
    if patience is not None and not has_val:
        raise ValueError("early stopping needs validation data")

    opt = optim.Adam(learning_rate=lr, bias_correction=True)
    opt.init(model.trainable_parameters())
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        return loss

    if compile:
        state = [model.state, opt.state, mx.random.state]
        step = partial(mx.compile, inputs=state, outputs=state)(step)

    n = X.shape[0]
    bs = n if batch_size is None else int(batch_size)
    hist = {"train_loss": [], "val_loss": [], "best_epoch": None, "stopped_early": False}
    best, best_params, wait = np.inf, None, 0
    for ep in range(int(epochs)):
        order = rng.permutation(n) if bs < n else np.arange(n)
        tot = 0.0
        for s in range(0, n, bs):
            idx = mx.array(order[s:s + bs])
            loss = step(X[idx], Y[idx])
            mx.eval(model.parameters(), opt.state, loss)
            tot += float(loss) * int(idx.shape[0])
        hist["train_loss"].append(tot / n)
        if has_val:
            v = float(loss_fn(model, X_val, Y_val))
            hist["val_loss"].append(v)
            if v < best:
                best, wait, hist["best_epoch"] = v, 0, ep
                if patience is not None:
                    best_params = tree_map(lambda a: mx.array(a), model.parameters())
            elif patience is not None:
                wait += 1
                if wait >= patience:
                    hist["stopped_early"] = True
                    break
    if best_params is not None:
        model.update(best_params)
        mx.eval(model.parameters())
    return hist


def max_sample_sd(metrics) -> float:
    """Registered-style seed spread: metrics shape (n_seeds, n_metrics); the
    sample SD (ddof=1) across seeds of each metric, then the max over metrics."""
    m = np.asarray(metrics, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    if m.shape[0] < 2:
        raise ValueError("need at least 2 seeds")
    return float(np.max(np.std(m, axis=0, ddof=1)))


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(model: nn.Module, path, meta: Optional[dict] = None) -> str:
    """Write <path>.safetensors and <path>.json (sha256, spec, meta); return sha256."""
    path = Path(path)
    wpath = path.with_suffix(".safetensors")
    model.save_weights(str(wpath))
    sha = sha256_file(wpath)
    spec = getattr(model, "spec", None)
    side = {
        "weights": wpath.name,
        "sha256": sha,
        "spec": spec.to_dict() if isinstance(spec, MLPSpec) else None,
        "mlx_version": mx.__version__,
        "meta": meta or {},
    }
    path.with_suffix(".json").write_text(json.dumps(side, indent=2, sort_keys=True))
    return sha


def load_checkpoint(model: nn.Module, path, expected_sha256: Optional[str] = None) -> dict:
    """Verify sha256 (sidecar and, if given, expected) then load weights."""
    path = Path(path)
    side = json.loads(path.with_suffix(".json").read_text())
    wpath = path.with_suffix(".safetensors")
    sha = sha256_file(wpath)
    if sha != side["sha256"]:
        raise ValueError(f"checkpoint sha256 mismatch vs sidecar: {sha} != {side['sha256']}")
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"checkpoint sha256 mismatch vs expected: {sha} != {expected_sha256}")
    spec = getattr(model, "spec", None)
    if side.get("spec") is not None and isinstance(spec, MLPSpec) and spec.to_dict() != side["spec"]:
        raise ValueError("checkpoint spec does not match the model spec")
    model.load_weights(str(wpath), strict=True)
    mx.eval(model.parameters())
    return side


def param_count(model: nn.Module) -> int:
    return int(sum(v.size for _, v in tree_flatten(model.parameters())))
