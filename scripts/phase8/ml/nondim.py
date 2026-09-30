"""Registered-style non-dimensionalisation (numpy float64 only).

Precision rule (envs/mlx/README.md): physical inputs/outputs are mapped to
O(1) before any float32 training; scoring maps back in float64.

The Scaler is fit once, on a training array only, and is frozen afterwards:
a second ``fit`` raises. The fit record stores the row count and the sha256
of the float64 training array so a saved scaler can be tied to the exact
rows it was fit on.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _as_2d64(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {X.shape}")
    return X


class Scaler:
    """z = (x - mean) / scale, column-wise; scale = population SD.

    A column whose SD is below ``min_rel_scale * max(1, |mean|)`` (constant
    in the training rows) gets scale 1 so the map stays invertible; it is
    listed in ``constant_columns``.
    """

    def __init__(self, names: Optional[Sequence[str]] = None, min_rel_scale: float = 1e-12):
        self.names = list(names) if names is not None else None
        self.min_rel_scale = float(min_rel_scale)
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.n_fit: Optional[int] = None
        self.fit_sha256: Optional[str] = None
        self.constant_columns: list = []

    @property
    def fitted(self) -> bool:
        return self.mean is not None

    def fit(self, X_train) -> "Scaler":
        if self.fitted:
            raise RuntimeError("Scaler is frozen: it was already fit (fit on the training array once)")
        X = _as_2d64(X_train)
        if X.shape[0] < 2:
            raise ValueError("need at least 2 training rows to fit a scale")
        if not np.all(np.isfinite(X)):
            raise ValueError("training array has non-finite values")
        if self.names is not None and len(self.names) != X.shape[1]:
            raise ValueError("names length does not match the number of columns")
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        tiny = self.min_rel_scale * np.maximum(1.0, np.abs(mean))
        const = scale < tiny
        scale = np.where(const, 1.0, scale)
        self.mean, self.scale = mean, scale
        self.constant_columns = [int(i) for i in np.flatnonzero(const)]
        self.n_fit = int(X.shape[0])
        self.fit_sha256 = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
        return self

    def _check(self, X):
        if not self.fitted:
            raise RuntimeError("Scaler is not fit")
        X = _as_2d64(X)
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} columns, got {X.shape[1]}")
        return X

    def transform(self, X) -> np.ndarray:
        X = self._check(X)
        return (X - self.mean) / self.scale

    def inverse_transform(self, Z) -> np.ndarray:
        Z = self._check(Z)
        return Z * self.scale + self.mean

    def inverse_transform_delta(self, dZ) -> np.ndarray:
        """Map a difference (no offset), e.g. a residual or an SD, to physical units."""
        dZ = self._check(dZ)
        return dZ * self.scale

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Scaler is not fit")
        return {
            "kind": "Scaler/v1",
            "names": self.names,
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
            "n_fit": self.n_fit,
            "fit_sha256": self.fit_sha256,
            "constant_columns": self.constant_columns,
            "min_rel_scale": self.min_rel_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        if d.get("kind") != "Scaler/v1":
            raise ValueError("not a Scaler/v1 record")
        s = cls(names=d.get("names"), min_rel_scale=d.get("min_rel_scale", 1e-12))
        s.mean = np.asarray(d["mean"], dtype=np.float64)
        s.scale = np.asarray(d["scale"], dtype=np.float64)
        s.n_fit = d["n_fit"]
        s.fit_sha256 = d["fit_sha256"]
        s.constant_columns = list(d.get("constant_columns", []))
        return s

    def save(self, path) -> None:
        # repr-exact floats: json writes shortest round-trip repr of float64
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "Scaler":
        return cls.from_dict(json.loads(Path(path).read_text()))
