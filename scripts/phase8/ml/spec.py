"""Framework-neutral architecture and problem specifications (numpy only).

The same frozen dataclasses drive the MLX models (models_mlx), the PyTorch
twins (models_torch) and the float64 numpy scoring path (score64), so the
three forward passes are defined by one record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

ACTIVATIONS = ("silu", "tanh")
OUTPUTS = ("linear", "tanh_bounded", "nozzle_fields")


@dataclass(frozen=True)
class MLPSpec:
    """Plain MLP: Linear -> act -> ... -> Linear -> output map.

    output:
      linear         y = z
      tanh_bounded   y = bound * tanh(z)            (M1 residual, |y| <= bound)
      nozzle_fields  (rho, u, p) = (rho_ref e^z0, u_ref z1, p_ref e^z2)  (M3)
    """

    in_dim: int
    hidden: Tuple[int, ...]
    out_dim: int
    activation: str = "silu"
    output: str = "linear"
    bound: Optional[float] = None
    out_ref: Optional[Tuple[float, ...]] = None  # nozzle_fields references

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation {self.activation!r} not in {ACTIVATIONS}")
        if self.output not in OUTPUTS:
            raise ValueError(f"output {self.output!r} not in {OUTPUTS}")
        if self.output == "tanh_bounded" and not (self.bound and self.bound > 0):
            raise ValueError("tanh_bounded output needs bound > 0")
        if self.output == "nozzle_fields":
            if self.out_dim != 3 or self.out_ref is None or len(self.out_ref) != 3:
                raise ValueError("nozzle_fields needs out_dim=3 and out_ref=(rho,u,p) refs")

    @property
    def layer_dims(self):
        dims = (self.in_dim,) + tuple(self.hidden) + (self.out_dim,)
        return list(zip(dims[:-1], dims[1:]))

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        if self.out_ref is not None:
            d["out_ref"] = list(self.out_ref)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["hidden"] = tuple(d["hidden"])
        if d.get("out_ref") is not None:
            d["out_ref"] = tuple(d["out_ref"])
        return cls(**d)


def m1_spec(in_dim: int, out_dim: int = 1, bound: float = 0.05) -> MLPSpec:
    """M1 residual MLP: 3 x 64, SiLU, tanh-bounded output (|dy| <= bound)."""
    return MLPSpec(in_dim, (64, 64, 64), out_dim, "silu", "tanh_bounded", bound=bound)


def m4_spec(in_dim: int, out_dim: int) -> MLPSpec:
    """M4 emulator MLP: 4 x 128, SiLU, linear output."""
    return MLPSpec(in_dim, (128, 128, 128, 128), out_dim, "silu", "linear")


def m3_spec(hidden=(64, 64, 64), out_ref=(1.0, 1.0, 1.0)) -> MLPSpec:
    """M3 nozzle field network x -> (rho, u, p), tanh hidden activations.

    rho and p pass through exp() so T = p/(rho R) and ln() in the entropy
    term are always defined; u is linear.
    """
    return MLPSpec(1, tuple(hidden), 3, "tanh", "nozzle_fields",
                   out_ref=tuple(float(v) for v in out_ref))


@dataclass(frozen=True)
class Q1DProblem:
    """Quasi-1D nozzle problem in any consistent unit system.

    x is the (non-dimensional) axial coordinate the network sees.
    A(x) = sum_k area_coeffs[k] * x**k.
    Calorically perfect gas: h = cp T, T = p/(rho R); entropy is carried as
    s/R = (cp/R) ln T - ln p (dimensionless, datum T = p = 1 unit).
    s_in is the inlet entropy on the same datum (entropy_over_R).
    mom_ref normalises the momentum residual d(pA + rho u^2 A)/dx - p dA/dx.
    """

    area_coeffs: Tuple[float, ...]
    mdot: float
    h0: float
    cp: float
    R: float
    s_in: float
    mom_ref: float = 1.0

    def area_np(self, x):
        x = np.asarray(x, dtype=np.float64)
        return sum(c * x ** k for k, c in enumerate(self.area_coeffs))

    def darea_np(self, x):
        x = np.asarray(x, dtype=np.float64)
        return sum(k * c * x ** (k - 1) for k, c in enumerate(self.area_coeffs) if k > 0)


def entropy_over_R(p, rho, cp, R):
    """s/R for a calorically perfect gas (numpy), same datum as the losses."""
    T = np.asarray(p, dtype=np.float64) / (np.asarray(rho, dtype=np.float64) * R)
    return (cp / R) * np.log(T) - np.log(p)


def max_rel_diff(a, b, floor: float = 1e-30) -> float:
    """max|a-b| / max|b|: the parity metric (max-normalised relative diff)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), floor))
