"""
P7.4 — new LE-PINN study on the Sajben weak-shock case: Ma-form RANS residual,
current-loss weights, and data efficiency. Registered in
``outputs/phase7/p74_registration.json`` before any training.

This is a NEW study, not P4.3 attempt 4. The legacy module
``simulation/nozzle/le_pinn.py``, its training entry point and its scorer are
not changed. Nothing here is promoted to the production cycle.

Physics (planar, steady, compressible, Favre-averaged RANS; conservative form)
----------------------------------------------------------------------------
With velocity (u, v), density rho, pressure p, temperature T:

    continuity  d(rho u)/dx + d(rho v)/dy = 0
    x-momentum  d(rho u u + p - tau_xx)/dx + d(rho u v - tau_xy)/dy = 0
    y-momentum  d(rho u v - tau_xy)/dx + d(rho v v + p - tau_yy)/dy = 0
    energy      d(rho u H - u tau_xx - v tau_xy + q_x)/dx
              + d(rho v H - u tau_xy - v tau_yy + q_y)/dy = 0
    EOS         p - rho R T = 0          (ideal gas, calorically perfect)

    H = cv T + (u^2 + v^2)/2 + p/rho     (total enthalpy; k is not included)
    tau_ij = mu_eff (du_i/dx_j + du_j/dx_i - 2/3 delta_ij div u),
    mu_eff = mu(T) + mu_t
    q_i = -cp (mu/Pr + mu_t/Pr_t) dT/dx_i

Turbulence is counted ONCE. The molecular viscous stress and the Boussinesq
Reynolds stress -rho<u_i'u_j'> = mu_t (2 S_ij - 2/3 delta_ij div u) - 2/3 rho k
delta_ij are combined into one mu_eff flux. There is no separate Reynolds-stress
output. The isotropic 2/3 rho k term is NOT modelled: the Spalart-Allmaras
solution supplies no turbulent kinetic energy and no measured R_ij. That is an
approximation (equivalently, k is absorbed into a modified pressure), not a
replication of Ma et al. Viscous dissipation enters through the work terms
u_j tau_ij in the conservative energy flux. The heat flux uses Pr = 0.72 and
Pr_t = 0.9. The molecular viscosity mu(T) is Sutherland's law (air: C1 =
1.458e-6, S = 110.4 K), evaluated on the predicted T. It is a closure, not a
WIND label. mu_t is a network output, trained on WIND mu_t only at the sampled
training points (both arms).

Derivatives are taken by autograd with respect to PHYSICAL coordinates (the
network normalises its inputs internally), so every chain-rule factor is
exact. Residuals are divided by the reference scales rho_r a_r / L (mass),
rho_r a_r^2 / L (momentum), rho_r a_r^3 / L (energy) and rho_r a_r^2 (EOS). The
references are the WIND file's freestream reference (rho_ref, a_ref, T_ref),
and L is the throat height from the geometry.

Wall conditions (lower wall y = 0; upper wall curved): no-slip u = v = 0 and
adiabatic dT/dn = 0, with n the unit normal of the actual wall computed from
the wall coordinates.

Loss weights (Ma et al. 2026, Eqs. 30-33, p. 5, in the forms fixed by the
Phase 7 plan), detached from the gradient:
    lambda_data = 0.1 + 0.9 sigmoid((L_phys + L_bc - L_data)/(L_data + eps))
    lambda_phys = 0.1 + 0.9 sigmoid((L_data - L_phys)/(L_phys + eps))
    lambda_bc   = 0.1 + 0.9 sigmoid((L_data - L_bc)/(L_bc + eps))
    L = lambda_data L_data + lambda_phys L_phys + lambda_bc L_bc
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
WIND_DIR = ROOT / "data" / "raw" / "cfd_datasets" / "nasa" / "transdif01"
WIND_CGD = WIND_DIR / "sajben.cgd"
WIND_CFL = WIND_DIR / "sajben.cfl"

VARS = ("rho", "u", "v", "p", "T", "mu_t")      # network outputs (physical meaning)
FLOW_VARS = ("rho", "u", "v", "p", "T")
SUTH_C1, SUTH_S = 1.458e-6, 110.4
PR, PR_T = 0.72, 0.9
FUSION_DELTA = 5e-4                              # m, as the legacy LE-PINN
EXCLUDED_I = (0,)                                # inflow column (legacy convention, P4.1)


# --------------------------------------------------------------------------
# Data: WIND rows, split, collocation and wall points
# --------------------------------------------------------------------------
@dataclass
class Geometry:
    x_nodes: np.ndarray        # (ni,) x of the grid columns (vertical grid lines)
    y_lower: np.ndarray        # (ni,) lower wall y (0)
    y_upper: np.ndarray        # (ni,) upper wall y
    grid_x: np.ndarray         # (nj, ni) grid coordinates (geometry only)
    grid_y: np.ndarray
    x_min: float
    x_max: float
    y_max: float
    L_ref: float               # throat height [m]
    ref: dict                  # WIND freestream reference: rho_ref, a_ref, T_ref, mu_ref, gamma, R


def load_wind():
    from simulation.nozzle.wind_cff import load_wind_solution
    return load_wind_solution(WIND_CGD, WIND_CFL)


def geometry_from(sol) -> Geometry:
    """Geometry and reference constants only (no flow field)."""
    if not np.allclose(sol.x, sol.x[0:1]):
        raise ValueError("expected vertical grid lines")
    ref = {k: float(sol.reference[k]) for k in ("rho_ref", "a_ref", "T_ref", "mu_ref", "gamma", "R", "p_ref")}
    return Geometry(x_nodes=sol.x[0].copy(), y_lower=sol.y[0].copy(), y_upper=sol.y[-1].copy(),
                    grid_x=sol.x.copy(), grid_y=sol.y.copy(),
                    x_min=float(sol.x.min()), x_max=float(sol.x.max()), y_max=float(sol.y.max()),
                    L_ref=float(sol.h_throat), ref=ref)


def all_row_ids(nj: int, ni: int) -> np.ndarray:
    """Flat C-order grid index g = j*ni + i of every WIND node outside the inflow column."""
    return np.array([j * ni + i for j in range(nj) for i in range(ni) if i not in EXCLUDED_I])


def label_table(sol, ids: np.ndarray) -> np.ndarray:
    """(len(ids), 8) = [x, y, rho, u, v, p, T, mu_t] at the given grid indices ONLY."""
    cols = [sol.x, sol.y, sol.rho, sol.u, sol.v, sol.p, sol.T, sol.mu_t]
    return np.column_stack([c.ravel()[ids] for c in cols]).astype(np.float64)


def make_split(nj: int, ni: int, test_fraction: float, split_seed: int) -> dict:
    ids = all_row_ids(nj, ni)
    perm = np.random.default_rng(split_seed).permutation(ids)
    n_test = int(round(test_fraction * len(ids)))
    return {"test": np.sort(perm[:n_test]), "pool": np.sort(perm[n_test:])}


def nested_subsets(pool: np.ndarray, fractions: list[float], subset_seed: int) -> dict:
    """First round(f * N_pool) rows of one seeded permutation: nested by construction."""
    perm = np.random.default_rng(subset_seed).permutation(pool)
    return {f: perm[: int(round(f * len(pool)))] for f in fractions}


def collocation_points(geom: Geometry, n: int, seed: int) -> np.ndarray:
    """Interior points from uniform draws in continuous grid-index space
    (xi in [1, ni-1], eta in [0, nj-1]) mapped by bilinear interpolation of the
    grid COORDINATES (geometry, no flow labels). Follows the grid's wall
    clustering; excludes the inflow column."""
    nj, ni = geom.grid_x.shape
    rng = np.random.default_rng(seed)
    xi = rng.uniform(1.0, ni - 1.0, n)
    eta = rng.uniform(0.0, nj - 1.0, n)
    i0 = np.clip(np.floor(xi).astype(int), 0, ni - 2)
    j0 = np.clip(np.floor(eta).astype(int), 0, nj - 2)
    a, b = xi - i0, eta - j0

    def bil(F):
        return ((1 - a) * (1 - b) * F[j0, i0] + a * (1 - b) * F[j0, i0 + 1]
                + (1 - a) * b * F[j0 + 1, i0] + a * b * F[j0 + 1, i0 + 1])
    return np.column_stack([bil(geom.grid_x), bil(geom.grid_y)])


def wall_normals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit normals (pointing +y side) of a wall y(x) from its coordinates."""
    dydx = np.gradient(y, x)
    n = np.column_stack([-dydx, np.ones_like(dydx)])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def wall_points(geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Wall nodes i >= 1 of both walls and their unit normals (geometry only)."""
    keep = np.array([i for i in range(len(geom.x_nodes)) if i not in EXCLUDED_I])
    xl, yl = geom.x_nodes[keep], geom.y_lower[keep]
    xu, yu = geom.x_nodes[keep], geom.y_upper[keep]
    nl = wall_normals(geom.x_nodes, geom.y_lower)[keep]
    nu = -wall_normals(geom.x_nodes, geom.y_upper)[keep]       # into the fluid (sign irrelevant for dT/dn = 0)
    return (np.vstack([np.column_stack([xl, yl]), np.column_stack([xu, yu])]),
            np.vstack([nl, nu]))


def wall_distance(xy: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """Vertical distance to the nearer wall (fusion mask only; no gradient)."""
    with torch.no_grad():
        xs = torch.as_tensor(geom.x_nodes, dtype=xy.dtype)
        yu = torch.as_tensor(geom.y_upper, dtype=xy.dtype)
        x = xy[:, 0].clamp(float(xs[0]), float(xs[-1]))
        k = torch.searchsorted(xs, x).clamp(1, len(xs) - 1)
        w = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
        y_up = yu[k - 1] + w * (yu[k] - yu[k - 1])
        return torch.minimum(xy[:, 1] - 0.0, y_up - xy[:, 1]).clamp(min=0.0)


# --------------------------------------------------------------------------
# Training-only scalers
# --------------------------------------------------------------------------
def mu_t_transform(mu_t, mu_ref: float):
    return np.log1p(np.maximum(mu_t, 0.0) / mu_ref) if isinstance(mu_t, np.ndarray) else torch.log1p(mu_t / mu_ref)


def fit_output_scalers(labels: np.ndarray, mu_ref: float) -> dict:
    """Mean/std of [rho, u, v, p, T, log1p(mu_t/mu_ref)] over the TRAINING rows only."""
    t = np.column_stack([labels[:, 2:7], mu_t_transform(labels[:, 7], mu_ref)])
    return {"mean": t.mean(axis=0).tolist(), "std": np.maximum(t.std(axis=0), 1e-8).tolist()}


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
def _mlp(n_in: int, width: int, n_hidden: int, n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(n_in, width), nn.Tanh()]
    for _ in range(n_hidden):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, n_out))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class MaLEPINN(nn.Module):
    """Dual network (legacy LE-PINN widths/depths: global 400 x (1 + 6), boundary
    100 x (1 + 6)), tanh, Xavier. Inputs: physical (x, y), normalised inside by
    geometry bounds. Global outputs: standardised [rho, u, v, p, T, log1p(mu_t/mu_ref)];
    the boundary network's (p, T) replace the global ones within FUSION_DELTA of a wall."""

    def __init__(self, geom: Geometry, scalers: dict, width: int = 400, n_hidden: int = 6,
                 b_width: int = 100, b_hidden: int = 6, delta: float = FUSION_DELTA):
        super().__init__()
        self.geom = geom
        self.delta = delta
        self.global_net = _mlp(2, width, n_hidden, len(VARS))
        self.boundary_net = _mlp(2, b_width, b_hidden, 2)
        self.register_buffer("in_lo", torch.tensor([geom.x_min, 0.0]))
        self.register_buffer("in_hi", torch.tensor([geom.x_max, geom.y_max]))
        self.register_buffer("out_mean", torch.tensor(scalers["mean"], dtype=torch.float32))
        self.register_buffer("out_std", torch.tensor(scalers["std"], dtype=torch.float32))
        self.mu_ref = geom.ref["mu_ref"]

    def normalised(self, xy: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (xy - self.in_lo) / (self.in_hi - self.in_lo) - 1.0
        g = self.global_net(z)
        mask = wall_distance(xy, self.geom) < self.delta
        if bool(mask.any()):
            b = self.boundary_net(z)
            m = mask.unsqueeze(1).to(g.dtype)
            sel = torch.zeros_like(g)
            sel[:, 3:5] = 1.0
            g = g * (1 - m * sel) + torch.cat([g[:, :3], b, g[:, 5:]], dim=1) * (m * sel)
        return g

    def fields(self, xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Physical fields: rho [kg/m3], u, v [m/s], p [Pa], T [K], mu_t [Pa s] (>= 0)."""
        q = self.out_mean + self.out_std * self.normalised(xy)
        out = {k: q[:, i] for i, k in enumerate(FLOW_VARS)}
        out["mu_t"] = (self.mu_ref * torch.expm1(q[:, 5])).clamp(min=0.0)
        return out


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
def sutherland(T: torch.Tensor) -> torch.Tensor:
    return SUTH_C1 * T.clamp(min=1.0) ** 1.5 / (T.clamp(min=1.0) + SUTH_S)


def _d(f: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """(N, 2) gradient of a pointwise field w.r.t. physical coordinates."""
    return torch.autograd.grad(f.sum(), xy, create_graph=True)[0]


def rans_residuals(xy: torch.Tensor, fields_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
                   ref: dict, L_ref: float, mu_fn: Callable = sutherland,
                   pr: float = PR, pr_t: float = PR_T) -> Dict[str, torch.Tensor]:
    """Nondimensional residuals (continuity, x_mom, y_mom, energy, eos) at xy.
    ``xy`` must require grad; ``fields_fn`` maps physical (x, y) to physical fields."""
    g, R = ref["gamma"], ref["R"]
    cp = g * R / (g - 1.0)
    cv = cp - R
    f = fields_fn(xy)
    rho, u, v, p, T, mut = f["rho"], f["u"], f["v"], f["p"], f["T"], f["mu_t"]
    mu = mu_fn(T)
    mu_eff = mu + mut                                   # ONE stress: molecular + Boussinesq
    du, dv, dT = _d(u, xy), _d(v, xy), _d(T, xy)
    ux, uy, vx, vy = du[:, 0], du[:, 1], dv[:, 0], dv[:, 1]
    div = ux + vy
    txx = mu_eff * (2.0 * ux - 2.0 / 3.0 * div)
    tyy = mu_eff * (2.0 * vy - 2.0 / 3.0 * div)
    txy = mu_eff * (uy + vx)
    kappa = cp * (mu / pr + mut / pr_t)
    qx, qy = -kappa * dT[:, 0], -kappa * dT[:, 1]
    H = cv * T + 0.5 * (u * u + v * v) + p / rho
    fluxes = {
        "continuity": (rho * u, rho * v),
        "x_mom": (rho * u * u + p - txx, rho * u * v - txy),
        "y_mom": (rho * u * v - txy, rho * v * v + p - tyy),
        "energy": (rho * u * H - u * txx - v * txy + qx, rho * v * H - u * txy - v * tyy + qy),
    }
    rr, ar = ref["rho_ref"], ref["a_ref"]
    scale = {"continuity": rr * ar / L_ref, "x_mom": rr * ar ** 2 / L_ref,
             "y_mom": rr * ar ** 2 / L_ref, "energy": rr * ar ** 3 / L_ref}
    out = {}
    for k, (Fx, Fy) in fluxes.items():
        out[k] = (_d(Fx, xy)[:, 0] + _d(Fy, xy)[:, 1]) / scale[k]
    out["eos"] = (p - rho * R * T) / (rr * ar ** 2)
    return out


def physics_loss(res: Dict[str, torch.Tensor]) -> torch.Tensor:
    """L_phys = sum over the five residuals of their mean square."""
    return sum(torch.mean(r ** 2) for r in res.values())


def wall_bc_loss(xy_w: torch.Tensor, normals: torch.Tensor,
                 fields_fn: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
                 ref: dict, L_ref: float) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """No-slip and adiabatic wall: mean (u/a_r)^2 + mean (v/a_r)^2 + mean ((L/T_r) dT/dn)^2."""
    f = fields_fn(xy_w)
    dTdn = (_d(f["T"], xy_w) * normals).sum(dim=1)
    ar = ref["a_ref"]
    terms = {"u": torch.mean((f["u"] / ar) ** 2), "v": torch.mean((f["v"] / ar) ** 2),
             "dTdn": torch.mean((dTdn * L_ref / ref["T_ref"]) ** 2)}
    return sum(terms.values()), terms


def ma_weights(L_data: torch.Tensor, L_phys: torch.Tensor, L_bc: torch.Tensor,
               eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Current-loss weights (plan forms of Ma Eqs. 31-33), DETACHED."""
    d, p, b = L_data.detach(), L_phys.detach(), L_bc.detach()
    s = torch.sigmoid
    return {"data": 0.1 + 0.9 * s((p + b - d) / (d + eps)),
            "phys": 0.1 + 0.9 * s((d - p) / (p + eps)),
            "bc": 0.1 + 0.9 * s((d - b) / (b + eps))}


def data_loss(model: MaLEPINN, xy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the six standardised outputs at the training rows.
    ``labels`` = (N, 6) [rho, u, v, p, T, mu_t] physical."""
    t = torch.cat([labels[:, :5], mu_t_transform(labels[:, 5:6], model.mu_ref)], dim=1)
    target = (t - model.out_mean) / model.out_std
    return torch.mean((model.normalised(xy) - target) ** 2)


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------
def ids_hash(ids) -> str:
    return hashlib.sha256(np.asarray(sorted(int(i) for i in ids), dtype=np.int64).tobytes()).hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
