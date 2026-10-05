"""MLX models for Phase 8 (Track D3 scaffolding). Import only inside catjet-mlx.

M1  residual MLP, 3 x 64, SiLU, tanh-bounded output (|dy| <= bound).
    Ensemble wrapper: N members with independent init seeds and a
    bootstrap-over-groups row-index generator.
M4  emulator MLP, 4 x 128, SiLU, linear output.
M3  nozzle PINN skeleton: field network x -> (rho, u, p) (tanh hidden) and a
    quasi-1D composite physics loss (mass, energy, momentum, entropy hinge,
    boundary conditions, optional data) with the weighting schemes in
    ``weighting``.

Architectures are defined by ``spec.MLPSpec`` so models_torch and score64
reproduce the forward pass exactly. Initialisation is explicit and keyed
(``mx.random.key(seed)``), independent of the global MLX RNG state.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .spec import MLPSpec, Q1DProblem, m1_spec, m3_spec, m4_spec
from .weighting import GROUPS

_ACT = {"silu": nn.silu, "tanh": mx.tanh}


class MLP(nn.Module):
    def __init__(self, spec: MLPSpec, seed: int = 0):
        super().__init__()
        self.spec = spec  # frozen dataclass: plain attribute, not a parameter
        self.layers = [nn.Linear(i, o) for i, o in spec.layer_dims]
        init_mlp(self, seed)

    def __call__(self, x):
        act = _ACT[self.spec.activation]
        for layer in self.layers[:-1]:
            x = act(layer(x))
        z = self.layers[-1](x)
        s = self.spec
        if s.output == "tanh_bounded":
            return s.bound * mx.tanh(z)
        if s.output == "nozzle_fields":
            r = s.out_ref
            return mx.stack([r[0] * mx.exp(z[..., 0]), r[1] * z[..., 1], r[2] * mx.exp(z[..., 2])], axis=-1)
        return z


def init_mlp(model: MLP, seed: int) -> None:
    """U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias (the MLX and
    PyTorch nn.Linear default family), drawn from keys split off seed."""
    key = mx.random.key(int(seed))
    for layer in model.layers:
        key, kw, kb = mx.random.split(key, 3)
        out_d, in_d = layer.weight.shape
        s = 1.0 / math.sqrt(in_d)
        layer.weight = mx.random.uniform(-s, s, (out_d, in_d), key=kw)
        layer.bias = mx.random.uniform(-s, s, (out_d,), key=kb)
    mx.eval(model.parameters())


def make_m1(in_dim: int, out_dim: int = 1, bound: float = 0.05, seed: int = 0) -> MLP:
    return MLP(m1_spec(in_dim, out_dim, bound), seed)


def make_m4(in_dim: int, out_dim: int, seed: int = 0) -> MLP:
    return MLP(m4_spec(in_dim, out_dim), seed)


def make_m3(hidden=(64, 64, 64), out_ref=(1.0, 1.0, 1.0), seed: int = 0) -> MLP:
    return MLP(m3_spec(hidden, out_ref), seed)


def flat_params(model: nn.Module) -> Dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def flat_grad_vector(grads) -> np.ndarray:
    """Concatenate a gradient tree into one float64 numpy vector (sorted keys)."""
    items = sorted(tree_flatten(grads), key=lambda kv: kv[0])
    return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for _, v in items])


# --------------------------------------------------------------------------
# Ensemble (M1)
# --------------------------------------------------------------------------
def member_seeds(base_seed: int, n_members: int) -> List[int]:
    """Independent init seeds via numpy SeedSequence.spawn (reproducible)."""
    ss = np.random.SeedSequence(int(base_seed))
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in ss.spawn(n_members)]


def bootstrap_group_indices(groups: Sequence, n_members: int, seed: int):
    """Bootstrap over groups (e.g. engines): for each member draw n_groups
    groups with replacement and return the row indices of the drawn groups
    (a group drawn k times contributes its rows k times), plus the
    out-of-bag group labels.

    Returns a list of (row_idx: np.ndarray[int], oob_groups: list).
    """
    groups = np.asarray(groups)
    labels = np.unique(groups)  # sorted: order does not depend on row order
    rows_of = {g: np.flatnonzero(groups == g) for g in labels}
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(n_members):
        drawn = rng.choice(len(labels), size=len(labels), replace=True)
        idx = np.concatenate([rows_of[labels[j]] for j in drawn])
        oob = [labels[j].item() for j in sorted(set(range(len(labels))) - set(drawn.tolist()))]
        out.append((idx, oob))
    return out


class Ensemble:
    """N independently initialised members of one spec."""

    def __init__(self, spec: MLPSpec, n_members: int, base_seed: int = 0):
        self.spec = spec
        self.seeds = member_seeds(base_seed, n_members)
        self.members = [MLP(spec, s) for s in self.seeds]

    def __len__(self):
        return len(self.members)

    def predict_all(self, x) -> mx.array:
        return mx.stack([m(x) for m in self.members], axis=0)

    def predict(self, x):
        """(mean, sample SD over members, ddof=1)."""
        ys = self.predict_all(x)
        n = ys.shape[0]
        mean = ys.mean(axis=0)
        sd = mx.sqrt(((ys - mean) ** 2).sum(axis=0) / max(n - 1, 1))
        return mean, sd


# --------------------------------------------------------------------------
# M3 quasi-1D physics loss
# --------------------------------------------------------------------------
def area_mx(prob: Q1DProblem, x):
    return sum(c * x ** k for k, c in enumerate(prob.area_coeffs))


def darea_mx(prob: Q1DProblem, x):
    return sum(k * c * x ** (k - 1) for k, c in enumerate(prob.area_coeffs) if k > 0)


def fields(net: MLP, x):
    """x: (N,) -> rho, u, p each (N,)."""
    q = net(x[:, None])
    return q[:, 0], q[:, 1], q[:, 2]


def momentum_flux(net: MLP, prob: Q1DProblem, x_scalar):
    """F(x) = p A + rho u^2 A at one scalar x (for grad w.r.t. x)."""
    q = net(x_scalar.reshape(1, 1))[0]
    A = area_mx(prob, x_scalar)
    return q[2] * A + q[0] * q[1] ** 2 * A


def dflux_dx(net: MLP, prob: Q1DProblem, x):
    """dF/dx at each collocation point via mx.vmap(mx.grad(.)) on the input."""
    return mx.vmap(mx.grad(lambda xs: momentum_flux(net, prob, xs)))(x)


def d2flux_dx2(net: MLP, prob: Q1DProblem, x):
    """d2F/dx2 (second derivative w.r.t. the network input)."""
    g = mx.grad(lambda xs: momentum_flux(net, prob, xs))
    return mx.vmap(mx.grad(g))(x)


def entropy_over_R_mx(p, rho, prob: Q1DProblem):
    T = p / (rho * prob.R)
    return (prob.cp / prob.R) * mx.log(T) - mx.log(p)


def physics_terms(net: MLP, prob: Q1DProblem, x_col, bc=None, data=None) -> Dict[str, mx.array]:
    """All loss terms as scalars.

    x_col: (N,) collocation points.
    bc:    (x_bc (k,), idx (k,) int field index 0=rho 1=u 2=p, val (k,))
    data:  (x_d (n,), q_d (n, 3)); relative to spec.out_ref.
    Group sums: phys = mass + energy + momentum + entropy.
    """
    rho, u, p = fields(net, x_col)
    A = area_mx(prob, x_col)
    mass = mx.mean(((rho * u * A - prob.mdot) / prob.mdot) ** 2)
    T = p / (rho * prob.R)
    h = prob.cp * T
    energy = mx.mean(((h + 0.5 * u ** 2 - prob.h0) / prob.h0) ** 2)
    mom_res = (dflux_dx(net, prob, x_col) - p * darea_mx(prob, x_col)) / prob.mom_ref
    momentum = mx.mean(mom_res ** 2)
    s = entropy_over_R_mx(p, rho, prob)
    entropy = mx.mean(mx.maximum(prob.s_in - s, 0.0) ** 2)
    terms = {"mass": mass, "energy": energy, "momentum": momentum, "entropy": entropy}
    terms["phys"] = mass + energy + momentum + entropy
    if bc is not None:
        x_bc, idx, val = bc
        q = net(x_bc[:, None])
        qb = mx.take_along_axis(q, idx[:, None], axis=1)[:, 0]
        terms["bc"] = mx.mean(((qb - val) / val) ** 2)
    else:
        terms["bc"] = mx.array(0.0)
    if data is not None:
        x_d, q_d = data
        ref = mx.array(net.spec.out_ref)
        terms["data"] = mx.mean(((net(x_d[:, None]) - q_d) / ref) ** 2)
    else:
        terms["data"] = mx.array(0.0)
    return terms


def ma_weights_mx(L_data, L_phys, L_bc, eps: float = 1e-8):
    """Ma et al. sigmoid weights in-graph, detached with stop_gradient."""
    Ld, Lp, Lb = (mx.stop_gradient(v) for v in (L_data, L_phys, L_bc))
    return {
        "data": 0.1 + 0.9 * mx.sigmoid((Lp + Lb - Ld) / (Ld + eps)),
        "phys": 0.1 + 0.9 * mx.sigmoid((Ld - Lp) / (Lp + eps)),
        "bc": 0.1 + 0.9 * mx.sigmoid((Ld - Lb) / (Lb + eps)),
    }


def composite_loss(terms: Dict[str, mx.array], weights) -> mx.array:
    """sum_g lam_g L_g over groups (data, phys, bc).

    weights: dict of floats (fixed / gradnorm / relobralo; constants to the
    graph) or the string "ma_sigmoid" (computed in-graph, detached).
    """
    if isinstance(weights, str):
        if weights != "ma_sigmoid":
            raise ValueError("only 'ma_sigmoid' is computed in-graph")
        weights = ma_weights_mx(terms["data"], terms["phys"], terms["bc"])
    # plain floats (never numpy scalars: np.float64 * mx.array leaves the graph)
    w = {g: v if isinstance(v, mx.array) else float(v) for g, v in weights.items()}
    return sum(w[g] * terms[g] for g in GROUPS)


def per_term_grads(net: MLP, terms_fn, names: Sequence[str] = GROUPS) -> Dict[str, np.ndarray]:
    """Flat float64 parameter gradient of each unweighted term (for gradnorm).

    terms_fn(model) -> dict of scalar terms.
    """
    out = {}
    for k in names:
        _, g = nn.value_and_grad(net, lambda m, k=k: terms_fn(m)[k])(net)
        out[k] = flat_grad_vector(g)
    return out
