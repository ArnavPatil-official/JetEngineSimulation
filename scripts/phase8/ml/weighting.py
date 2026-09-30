"""Loss-weighting schemes for the M3 composite PINN loss (numpy only).

Every scheme returns plain Python floats computed from loss *values* (or
gradient values), so the weights are constants to the autodiff graph:
detached by construction. The in-graph variants of the Ma et al. weights
(models_mlx.ma_weights_mx, models_torch.ma_weights_torch) apply
stop_gradient / detach explicitly.

Schemes
-------
fixed        constant weights.
gradnorm     learning-rate annealing, Wang, Teng & Perdikaris (2021),
             SIAM J. Sci. Comput. 43(5) A3055: for each non-reference term i
               lam_hat_i = max_theta |grad L_ref| / mean_theta |grad L_i|
               lam_i <- (1 - alpha) lam_i + alpha lam_hat_i   (alpha = 0.9)
             The reference term (the PDE residual, 'phys') keeps weight 1.
             Following the authors' reference code, grad L_i is the gradient
             of the unweighted term.
relobralo    Relative Loss Balancing with Random Lookback, Bischof & Kraus
             (2021, arXiv:2110.09813):
               bal_i(t, t') = m exp(L_i(t) / (T L_i(t'))) / sum_j exp(L_j(t) / (T L_j(t')))
               lam_i(t) = alpha [rho lam_i(t-1) + (1 - rho) bal_i(t, 0)]
                          + (1 - alpha) bal_i(t, t-1)
             rho ~ Bernoulli(E[rho]) (the "saudade"), seeded RNG.
ma_sigmoid   current-loss sigmoid weights (Ma et al.):
               lam_data = .1 + .9 sigmoid((L_phys + L_bc - L_data) / (L_data + eps))
               lam_phys = .1 + .9 sigmoid((L_data - L_phys) / (L_phys + eps))
               lam_bc   = .1 + .9 sigmoid((L_data - L_bc) / (L_bc + eps))
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

GROUPS = ("data", "phys", "bc")


def _sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = np.exp(z)
    return float(e / (1.0 + e))


def ma_weights(L_data: float, L_phys: float, L_bc: float, eps: float = 1e-8) -> Dict[str, float]:
    """Ma et al. sigmoid current-loss weights; each lies in (0.1, 1.0)."""
    L_data, L_phys, L_bc = float(L_data), float(L_phys), float(L_bc)
    return {
        "data": 0.1 + 0.9 * _sigmoid((L_phys + L_bc - L_data) / (L_data + eps)),
        "phys": 0.1 + 0.9 * _sigmoid((L_data - L_phys) / (L_phys + eps)),
        "bc": 0.1 + 0.9 * _sigmoid((L_data - L_bc) / (L_bc + eps)),
    }


class FixedWeights:
    name = "fixed"

    def __init__(self, weights: Mapping[str, float]):
        if any(float(v) <= 0 for v in weights.values()):
            raise ValueError("fixed weights must be positive")
        self.weights = {k: float(v) for k, v in weights.items()}

    def update(self, *_, **__) -> Dict[str, float]:
        return dict(self.weights)


class MaSigmoidWeights:
    name = "ma_sigmoid"

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def update(self, losses: Mapping[str, float]) -> Dict[str, float]:
        return ma_weights(losses["data"], losses["phys"], losses["bc"], self.eps)


class GradNormWeights:
    """Wang et al. (2021) gradient-statistics balancing (see module doc)."""

    name = "gradnorm"

    def __init__(self, names: Sequence[str] = GROUPS, ref: str = "phys",
                 alpha: float = 0.9, eps: float = 1e-12):
        if ref not in names:
            raise ValueError("reference term must be one of names")
        self.names, self.ref, self.alpha, self.eps = tuple(names), ref, float(alpha), eps
        self.weights = {k: 1.0 for k in self.names}

    def target(self, grads: Mapping[str, np.ndarray]) -> Dict[str, float]:
        gmax = float(np.max(np.abs(np.asarray(grads[self.ref], dtype=np.float64))))
        out = {self.ref: 1.0}
        for k in self.names:
            if k == self.ref:
                continue
            gmean = float(np.mean(np.abs(np.asarray(grads[k], dtype=np.float64))))
            out[k] = gmax / (gmean + self.eps)
        return out

    def update(self, grads: Mapping[str, np.ndarray]) -> Dict[str, float]:
        """grads: term name -> flat parameter-gradient vector of the unweighted term."""
        lam_hat = self.target(grads)
        for k in self.names:
            if k == self.ref:
                continue
            self.weights[k] = (1.0 - self.alpha) * self.weights[k] + self.alpha * lam_hat[k]
        return dict(self.weights)


class ReLoBRaLoWeights:
    """Bischof & Kraus (2021) ReLoBRaLo (see module doc)."""

    name = "relobralo"

    def __init__(self, names: Sequence[str] = GROUPS, alpha: float = 0.999,
                 temperature: float = 0.1, expected_rho: float = 0.999,
                 seed: int = 0, eps: float = 1e-12):
        self.names = tuple(names)
        self.alpha, self.T, self.expected_rho, self.eps = float(alpha), float(temperature), float(expected_rho), eps
        self.rng = np.random.default_rng(seed)
        self.L0: Optional[np.ndarray] = None
        self.L_prev: Optional[np.ndarray] = None
        self.lam = np.ones(len(self.names))
        self.last_rho: Optional[int] = None

    def balance(self, L_t: np.ndarray, L_ref: np.ndarray) -> np.ndarray:
        z = L_t / (self.T * L_ref + self.eps)
        z = z - np.max(z)  # softmax shift; the ratio is unchanged
        e = np.exp(z)
        return len(self.names) * e / np.sum(e)

    def update(self, losses: Mapping[str, float]) -> Dict[str, float]:
        L = np.array([float(losses[k]) for k in self.names], dtype=np.float64)
        if self.L0 is None:  # t = 0: all weights 1
            self.L0 = L.copy()
            self.L_prev = L.copy()
            self.lam = np.ones(len(self.names))
            return dict(zip(self.names, self.lam.tolist()))
        rho = int(self.rng.random() < self.expected_rho)
        self.last_rho = rho
        hist = rho * self.lam + (1 - rho) * self.balance(L, self.L0)
        self.lam = self.alpha * hist + (1.0 - self.alpha) * self.balance(L, self.L_prev)
        self.L_prev = L.copy()
        return dict(zip(self.names, self.lam.tolist()))


def make_scheme(name: str, **kw):
    return {
        "fixed": FixedWeights,
        "gradnorm": GradNormWeights,
        "relobralo": ReLoBRaLoWeights,
        "ma_sigmoid": MaSigmoidWeights,
    }[name](**kw)
