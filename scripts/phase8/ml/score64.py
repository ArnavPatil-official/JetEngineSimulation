"""Float64 CPU scoring path.

MLX has no float64 on the Metal GPU, so final scores never come from the
float32 training forward pass. A trained model's weights are exported to
numpy float64 and the forward pass defined by its ``MLPSpec`` is re-run in
numpy float64; predictions are mapped back to physical units with the
float64 ``nondim.Scaler``. mlx is imported lazily (export only), so this
module and saved ``.npz`` weight files can be scored without MLX.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .spec import MLPSpec


def export_params64(model) -> Dict[str, np.ndarray]:
    """MLX model -> {'layers.i.weight'|'layers.i.bias': float64 ndarray}."""
    from mlx.utils import tree_flatten  # lazy: only needed when exporting

    return {k: np.asarray(v, dtype=np.float32).astype(np.float64)
            for k, v in tree_flatten(model.parameters())}


def _silu(z):
    return z / (1.0 + np.exp(-z))


_ACT = {"silu": _silu, "tanh": np.tanh}


def forward64(params: Dict[str, np.ndarray], spec: MLPSpec, X) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64)
    n = len(spec.layer_dims)
    act = _ACT[spec.activation]
    for i in range(n):
        W = params[f"layers.{i}.weight"]
        b = params[f"layers.{i}.bias"]
        x = x @ W.T + b
        if i < n - 1:
            x = act(x)
    if spec.output == "tanh_bounded":
        return spec.bound * np.tanh(x)
    if spec.output == "nozzle_fields":
        r = spec.out_ref
        return np.stack([r[0] * np.exp(x[..., 0]), r[1] * x[..., 1], r[2] * np.exp(x[..., 2])], axis=-1)
    return x


def predict64(params, spec: MLPSpec, X_phys, x_scaler=None, y_scaler=None, residual: bool = False):
    """Physical-unit float64 prediction.

    residual=True: the output is a difference (M1 residual), so only the
    y scale is applied (no mean offset).
    """
    Z = x_scaler.transform(X_phys) if x_scaler is not None else np.asarray(X_phys, dtype=np.float64)
    Y = forward64(params, spec, Z)
    if y_scaler is None:
        return Y
    return y_scaler.inverse_transform_delta(Y) if residual else y_scaler.inverse_transform(Y)


def ensemble_predict64(param_list: Sequence[Dict[str, np.ndarray]], spec: MLPSpec, X_phys,
                       x_scaler=None, y_scaler=None, residual: bool = False):
    """(mean, sample SD ddof=1, all members) in float64."""
    ys = np.stack([predict64(p, spec, X_phys, x_scaler, y_scaler, residual) for p in param_list])
    sd = ys.std(axis=0, ddof=1) if len(param_list) > 1 else np.zeros_like(ys[0])
    return ys.mean(axis=0), sd, ys


def score64(params, spec: MLPSpec, X_phys, Y_phys, x_scaler=None, y_scaler=None,
            groups: Optional[Sequence] = None) -> Dict[str, float]:
    """Float64 metrics on physical units: RMSE, MAE, MAPE (%) and, when
    groups are given, group-weighted MAPE (mean of per-group MAPEs)."""
    Y_phys = np.asarray(Y_phys, dtype=np.float64)
    P = predict64(params, spec, X_phys, x_scaler, y_scaler).reshape(Y_phys.shape)
    err = P - Y_phys
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(100.0 * np.mean(np.abs(err / Y_phys))),
    }
    if groups is not None:
        g = np.asarray(groups)
        ape = np.abs(err / Y_phys).reshape(len(g), -1).mean(axis=1)
        out["group_mape_pct"] = float(100.0 * np.mean([ape[g == k].mean() for k in np.unique(g)]))
    return out


def save_params64(path, params: Dict[str, np.ndarray], spec: MLPSpec) -> None:
    np.savez(path, __spec__=np.array(json.dumps(spec.to_dict())), **params)


def load_params64(path):
    with np.load(path, allow_pickle=False) as z:
        spec = MLPSpec.from_dict(json.loads(str(z["__spec__"])))
        params = {k: z[k].astype(np.float64) for k in z.files if k != "__spec__"}
    return params, spec
