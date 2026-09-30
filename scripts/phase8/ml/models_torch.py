"""PyTorch twins of the MLX models (parity reference only; never imports mlx
at module level).

Weight layout: MLX ``nn.Linear.weight`` has shape (out, in) and computes
``x @ W.T + b``, exactly like ``torch.nn.Linear``; ``copy_mlx_to_torch``
checks the shapes and copies the arrays unchanged (no transpose).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

from .spec import MLPSpec, Q1DProblem
from .weighting import GROUPS

_ACT = {"silu": F.silu, "tanh": torch.tanh}


class TorchMLP(tnn.Module):
    def __init__(self, spec: MLPSpec, dtype=torch.float32):
        super().__init__()
        self.spec = spec
        self.layers = tnn.ModuleList([tnn.Linear(i, o, dtype=dtype) for i, o in spec.layer_dims])

    def forward(self, x):
        act = _ACT[self.spec.activation]
        for layer in self.layers[:-1]:
            x = act(layer(x))
        z = self.layers[-1](x)
        s = self.spec
        if s.output == "tanh_bounded":
            return s.bound * torch.tanh(z)
        if s.output == "nozzle_fields":
            r = s.out_ref
            return torch.stack([r[0] * torch.exp(z[..., 0]), r[1] * z[..., 1], r[2] * torch.exp(z[..., 2])], dim=-1)
        return z


def copy_mlx_to_torch(mlx_model, torch_model: TorchMLP) -> TorchMLP:
    """Copy every Linear weight/bias bit-for-bit (float32) from MLX to torch."""
    if len(mlx_model.layers) != len(torch_model.layers):
        raise ValueError("layer count mismatch")
    with torch.no_grad():
        for lm, lt in zip(mlx_model.layers, torch_model.layers):
            w = np.array(lm.weight, copy=True)
            b = np.array(lm.bias, copy=True)
            if w.shape != tuple(lt.weight.shape) or b.shape != tuple(lt.bias.shape):
                raise ValueError(f"layout mismatch: mlx {w.shape} vs torch {tuple(lt.weight.shape)}")
            lt.weight.copy_(torch.from_numpy(w).to(lt.weight.dtype))
            lt.bias.copy_(torch.from_numpy(b).to(lt.bias.dtype))
    return torch_model


def grads_by_mlx_name(torch_model: TorchMLP) -> Dict[str, np.ndarray]:
    """Parameter gradients keyed like mlx.utils.tree_flatten ('layers.0.weight')."""
    return {name: p.grad.detach().cpu().numpy().copy() for name, p in torch_model.named_parameters()}


# --------------------------------------------------------------------------
# M3 quasi-1D physics loss (same formulas as models_mlx.physics_terms)
# --------------------------------------------------------------------------
def area_t(prob: Q1DProblem, x):
    return sum(c * x ** k for k, c in enumerate(prob.area_coeffs))


def darea_t(prob: Q1DProblem, x):
    return sum(k * c * x ** (k - 1) for k, c in enumerate(prob.area_coeffs) if k > 0)


def dflux_dx_t(net: TorchMLP, prob: Q1DProblem, x, order: int = 1):
    """dF/dx (order 1) or d2F/dx2 (order 2) of F = pA + rho u^2 A, with the
    graph kept (create_graph=True) so parameter gradients flow through."""
    x = x.detach().requires_grad_(True)
    q = net(x[:, None])
    A = area_t(prob, x)
    Fx = q[:, 2] * A + q[:, 0] * q[:, 1] ** 2 * A  # F_i depends only on x_i
    d = torch.autograd.grad(Fx.sum(), x, create_graph=True)[0]
    if order == 2:
        d = torch.autograd.grad(d.sum(), x, create_graph=True)[0]
    return d, q


def physics_terms_t(net: TorchMLP, prob: Q1DProblem, x_col, bc=None, data=None) -> Dict[str, torch.Tensor]:
    dF, q = dflux_dx_t(net, prob, x_col)
    x = x_col
    rho, u, p = q[:, 0], q[:, 1], q[:, 2]
    A = area_t(prob, x)
    mass = torch.mean(((rho * u * A - prob.mdot) / prob.mdot) ** 2)
    T = p / (rho * prob.R)
    h = prob.cp * T
    energy = torch.mean(((h + 0.5 * u ** 2 - prob.h0) / prob.h0) ** 2)
    momentum = torch.mean(((dF - p * darea_t(prob, x)) / prob.mom_ref) ** 2)
    s = (prob.cp / prob.R) * torch.log(T) - torch.log(p)
    entropy = torch.mean(torch.clamp(prob.s_in - s, min=0.0) ** 2)
    terms = {"mass": mass, "energy": energy, "momentum": momentum, "entropy": entropy}
    terms["phys"] = mass + energy + momentum + entropy
    zero = torch.zeros((), dtype=q.dtype)
    if bc is not None:
        x_bc, idx, val = bc
        qb = net(x_bc[:, None]).gather(1, idx[:, None].long())[:, 0]
        terms["bc"] = torch.mean(((qb - val) / val) ** 2)
    else:
        terms["bc"] = zero
    if data is not None:
        x_d, q_d = data
        ref = torch.tensor(net.spec.out_ref, dtype=q.dtype)
        terms["data"] = torch.mean(((net(x_d[:, None]) - q_d) / ref) ** 2)
    else:
        terms["data"] = zero
    return terms


def ma_weights_torch(L_data, L_phys, L_bc, eps: float = 1e-8):
    Ld, Lp, Lb = (v.detach() for v in (L_data, L_phys, L_bc))
    return {
        "data": 0.1 + 0.9 * torch.sigmoid((Lp + Lb - Ld) / (Ld + eps)),
        "phys": 0.1 + 0.9 * torch.sigmoid((Ld - Lp) / (Lp + eps)),
        "bc": 0.1 + 0.9 * torch.sigmoid((Ld - Lb) / (Lb + eps)),
    }


def composite_loss_t(terms, weights):
    if isinstance(weights, str):
        if weights != "ma_sigmoid":
            raise ValueError("only 'ma_sigmoid' is computed in-graph")
        weights = ma_weights_torch(terms["data"], terms["phys"], terms["bc"])
    w = {g: v if torch.is_tensor(v) else float(v) for g, v in weights.items()}
    return sum(w[g] * terms[g] for g in GROUPS)
