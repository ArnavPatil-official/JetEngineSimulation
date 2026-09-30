"""MLX-vs-PyTorch parity measurements (gate-G2 infrastructure check).

Same weights in MLX and PyTorch, float32 on CPU for both
(mx.set_default_device(mx.cpu) during the comparison): forward outputs,
loss values and parameter gradients for M1, M4 and the M3 physics-loss
terms (including d2F/dx2 and the mixed x/theta second derivatives in the
momentum residual's parameter gradient). The MLX forward is also run on
the GPU and its difference reported. Inputs are synthetic (seeded random
numbers, a toy nozzle); no project data.

Metric: max_rel_diff(a, b) = max|a - b| / max|b| (spec.max_rel_diff).

    ~/miniforge3/envs/catjet-mlx/bin/python scripts/phase8/ml/parity.py
prints the numbers as JSON.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import torch
from mlx.utils import tree_flatten

if __package__ in (None, ""):  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "ml"

from .models_mlx import (composite_loss, d2flux_dx2, dflux_dx, make_m1, make_m3, make_m4,
                         physics_terms)
from .models_torch import (TorchMLP, composite_loss_t, copy_mlx_to_torch, dflux_dx_t,
                           grads_by_mlx_name, physics_terms_t)
from .spec import Q1DProblem, entropy_over_R, max_rel_diff

M3_TERMS = ("mass", "energy", "momentum", "entropy", "bc", "data", "phys")


@contextmanager
def default_device(dev):
    old = mx.default_device()
    mx.set_default_device(dev)
    try:
        yield
    finally:
        mx.set_default_device(old)


def _np(a):
    return np.array(a, dtype=np.float64)


def _grad_diffs(g_mlx: dict, g_t: dict) -> float:
    assert set(g_mlx) == set(g_t), (sorted(g_mlx), sorted(g_t))
    return max(max_rel_diff(g_mlx[k], g_t[k]) for k in g_mlx)


def mlp_parity(kind: str, seed: int = 0, n: int = 257) -> dict:
    """M1 or M4: forward, MSE loss and parameter gradients, CPU float32."""
    rng = np.random.default_rng(1000 + seed)
    d_in, d_out = (6, 1) if kind == "M1" else (8, 5)
    X = rng.standard_normal((n, d_in)).astype(np.float32)
    Y = (0.03 if kind == "M1" else 1.0) * rng.standard_normal((n, d_out)).astype(np.float32)
    out = {}
    with default_device(mx.cpu):
        m = make_m1(d_in, d_out, bound=0.05, seed=seed) if kind == "M1" else make_m4(d_in, d_out, seed=seed)
        t = copy_mlx_to_torch(m, TorchMLP(m.spec))
        # layout check: identical shapes, identical values (no transpose)
        for (k, v), (kt, vt) in zip(sorted(tree_flatten(m.parameters())), sorted(t.named_parameters())):
            assert k == kt and tuple(v.shape) == tuple(vt.shape)
            assert np.array_equal(np.array(v), vt.detach().numpy())
        xm, ym = mx.array(X), mx.array(Y)
        y_mlx = m(xm)
        loss_fn = lambda mod, x, y: mx.mean((mod(x) - y) ** 2)
        L_mlx, g_mlx = nn.value_and_grad(m, loss_fn)(m, xm, ym)
        g_mlx = {k: _np(v) for k, v in tree_flatten(g_mlx)}
        xt, yt = torch.from_numpy(X), torch.from_numpy(Y)
        y_t = t(xt)
        L_t = torch.mean((y_t - yt) ** 2)
        t.zero_grad()
        L_t.backward()
        out["forward_cpu"] = max_rel_diff(_np(y_mlx), y_t.detach().numpy())
        out["loss_cpu"] = max_rel_diff(float(L_mlx), float(L_t.detach()))
        out["grad_cpu"] = _grad_diffs(g_mlx, grads_by_mlx_name(t))
        ref = y_t.detach().numpy()
    with default_device(mx.gpu):
        y_gpu = m(mx.array(X))
        out["forward_gpu_vs_torch_cpu"] = max_rel_diff(_np(y_gpu), ref)
        out["forward_gpu_vs_mlx_cpu"] = max_rel_diff(_np(y_gpu), _np(y_mlx))
    return out


def toy_nozzle(seed: int = 0):
    """Synthetic toy nozzle (nondimensional, gamma = 1.4, R = 1): not project data."""
    gamma, R = 1.4, 1.0
    cp = gamma * R / (gamma - 1.0)
    coeffs = (1.5, -2.0, 2.0)  # A(x) = 1 + 2 (x - 1/2)^2
    rho_in, u_in, p_in = 1.0, 0.3, 1.0
    A_in = coeffs[0]
    mdot = rho_in * u_in * A_in
    h0 = cp * p_in / (rho_in * R) + 0.5 * u_in ** 2
    s_in = float(entropy_over_R(p_in, rho_in, cp, R))
    prob = Q1DProblem(area_coeffs=coeffs, mdot=mdot, h0=h0, cp=cp, R=R, s_in=s_in, mom_ref=1.0)
    rng = np.random.default_rng(2000 + seed)
    x_col = np.sort(rng.uniform(0, 1, 64)).astype(np.float32)
    bc = (np.array([0.0, 0.0, 0.0, 1.0], np.float32), np.array([0, 1, 2, 2], np.int32),
          np.array([rho_in, u_in, p_in, 0.8], np.float32))
    x_d = rng.uniform(0, 1, 16).astype(np.float32)
    q_d = np.stack([1 + 0.1 * rng.standard_normal(16), 0.3 + 0.05 * rng.standard_normal(16),
                    1 + 0.1 * rng.standard_normal(16)], axis=1).astype(np.float32)
    return prob, x_col, bc, (x_d, q_d)


def m3_parity(seed: int = 0) -> dict:
    prob, x_col, bc, data = toy_nozzle(seed)
    out = {}
    with default_device(mx.cpu):
        net = make_m3(hidden=(64, 64, 64), out_ref=(1.0, 0.3, 1.0), seed=seed)
        t = copy_mlx_to_torch(net, TorchMLP(net.spec))
        xm = mx.array(x_col)
        bc_m = tuple(mx.array(a) for a in bc)
        data_m = tuple(mx.array(a) for a in data)
        xt = torch.from_numpy(x_col)
        bc_t = tuple(torch.from_numpy(a) for a in bc)
        data_t = tuple(torch.from_numpy(a) for a in data)

        # fields and x-derivatives
        q_m = _np(net(xm[:, None]))
        # put s_in at the median collocation entropy so the second-law hinge
        # is active at about half the points (otherwise its parity is 0 == 0)
        s_m = entropy_over_R(q_m[:, 2], q_m[:, 0], prob.cp, prob.R)
        prob = replace(prob, s_in=float(np.median(s_m)))
        d1_m = _np(dflux_dx(net, prob, xm))
        d2_m = _np(d2flux_dx2(net, prob, xm))
        d1_t, q_t = dflux_dx_t(t, prob, xt, order=1)
        d2_t, _ = dflux_dx_t(t, prob, xt, order=2)
        out["fields_cpu"] = max_rel_diff(q_m, q_t.detach().numpy())
        out["dF_dx_cpu"] = max_rel_diff(d1_m, d1_t.detach().numpy())
        out["d2F_dx2_cpu"] = max_rel_diff(d2_m, d2_t.detach().numpy())

        # term values and per-term parameter gradients
        tm = physics_terms(net, prob, xm, bc_m, data_m)
        tt = physics_terms_t(t, prob, xt, bc_t, data_t)
        out["entropy_hinge_active_points"] = f"{int(np.sum(prob.s_in - s_m > 0))}/{len(x_col)}"
        loss_d, grad_d, g_m32, g_t32 = {}, {}, {}, {}
        for k in M3_TERMS:
            loss_d[k] = max_rel_diff(float(tm[k]), float(tt[k].detach()))
            _, g = nn.value_and_grad(net, lambda mod, k=k: physics_terms(mod, prob, xm, bc_m, data_m)[k])(net)
            t.zero_grad()
            physics_terms_t(t, prob, xt, bc_t, data_t)[k].backward()
            g_m32[k] = {kk: _np(v) for kk, v in tree_flatten(g)}
            g_t32[k] = grads_by_mlx_name(t)
            grad_d[k] = _grad_diffs(g_m32[k], g_t32[k])

        # float64 reference: the same float32 weights cast exactly to a
        # float64 torch twin. If MLX-f32 and torch-f32 are each about as far
        # from float64 as from each other, a difference is float32
        # conditioning (e.g. cancellation in dF/dx - p dA/dx or s_in - s),
        # not an implementation mismatch.
        t64 = copy_mlx_to_torch(net, TorchMLP(net.spec, dtype=torch.float64))
        c64 = lambda a: torch.from_numpy(a.astype(np.float64) if a.dtype.kind == "f" else a)
        x64, bc64, data64 = c64(x_col), tuple(c64(a) for a in bc), tuple(c64(a) for a in data)
        vs64 = {}
        for k in M3_TERMS:
            t64.zero_grad()
            physics_terms_t(t64, prob, x64, bc64, data64)[k].backward()
            g64 = grads_by_mlx_name(t64)
            vs64[k] = {"mlx32": _grad_diffs(g_m32[k], g64), "torch32": _grad_diffs(g_t32[k], g64)}
        out["grad_vs_float64"] = vs64
        for scheme in ("fixed", "ma_sigmoid"):
            w = {"data": 1.0, "phys": 0.5, "bc": 2.0} if scheme == "fixed" else "ma_sigmoid"
            f = lambda mod: composite_loss(physics_terms(mod, prob, xm, bc_m, data_m), w)
            L_m, g = nn.value_and_grad(net, f)(net)
            t.zero_grad()
            L_t = composite_loss_t(physics_terms_t(t, prob, xt, bc_t, data_t), w)
            L_t.backward()
            loss_d[f"composite_{scheme}"] = max_rel_diff(float(L_m), float(L_t.detach()))
            grad_d[f"composite_{scheme}"] = _grad_diffs({kk: _np(v) for kk, v in tree_flatten(g)},
                                                        grads_by_mlx_name(t))
        out["loss_cpu"] = loss_d
        out["grad_cpu"] = grad_d
        out["loss_cpu_max"] = max(loss_d.values())
        out["grad_cpu_max"] = max(grad_d.values())
        tm_cpu = {k: float(v) for k, v in tm.items()}
        tt_ref = {k: float(tt[k].detach()) for k in M3_TERMS}
    with default_device(mx.gpu):
        xg = mx.array(x_col)
        out["fields_gpu_vs_torch_cpu"] = max_rel_diff(_np(net(xg[:, None])), q_t.detach().numpy())
        out["dF_dx_gpu_vs_torch_cpu"] = max_rel_diff(_np(dflux_dx(net, prob, xg)), d1_t.detach().numpy())
        tg = physics_terms(net, prob, xg, tuple(mx.array(a) for a in bc), tuple(mx.array(a) for a in data))
        out["loss_gpu_vs_torch_cpu_max"] = max(max_rel_diff(float(tg[k]), tt_ref[k]) for k in M3_TERMS)
        out["loss_gpu_vs_mlx_cpu_max"] = max(max_rel_diff(float(tg[k]), tm_cpu[k]) for k in M3_TERMS)
    return out


def run_all(seed: int = 0) -> dict:
    return {
        "versions": {"mlx": mx.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "M1": mlp_parity("M1", seed),
        "M4": mlp_parity("M4", seed),
        "M3": m3_parity(seed),
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(run_all(), indent=2, default=float))
