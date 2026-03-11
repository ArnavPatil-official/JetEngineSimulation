"""
Locally Enhanced Physics-Informed Neural Network (LE-PINN) Surrogate Model.

Implements the dual-network architecture from Ma et al. for 2D RANS nozzle
flow field prediction.

Architecture
------------
- **GlobalNetwork** : 6 hidden layers × 400 neurons, ReLU.
  Inputs (6): [x, y, A5, A6, P_in, T_in]
  Outputs (9): [ρ, u, v, P, T, UU, VV, UV, μ_eff]

- **BoundaryNetwork** : 6 hidden layers × 100 neurons, ReLU.
  Inputs (6): [x_b, y_b, A5, A6, P_in, T_in]
  Outputs (2): [P_b, T_b]

- **Fusion** : If min wall distance d(x) < δ (δ = 5e-4), replace the global
  network's pressure and temperature with the boundary network's predictions.

Training
--------
- Loss: L = λ_data·L_data + λ_physics·L_physics + λ_BC·L_BC
- Physics: 2D RANS residuals (mass, x-mom, y-mom, energy) + ideal gas EOS
- BC: Wall no-slip (u=v=0) + adiabatic wall (∂T/∂n=0)
- Adaptive sigmoid-based loss weighting
- Optimizer: AdamW (lr=1e-4, wd=1e-5)
- Scheduler: ReduceLROnPlateau (patience 10, factor 0.5, min_lr 1e-8)
- Init: Xavier uniform weights, zero biases

Wall geometry is sourced from ``scripts/visualization/nozzle_2d_geometry.py``.
Training data is synthetic, generated from isentropic + ideal-gas relations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Default fluid properties
R_GAS: float = 287.0       # J/(kg·K) — specific gas constant for air
GAMMA: float = 1.4         # heat capacity ratio
CP: float = R_GAS * GAMMA / (GAMMA - 1.0)  # J/(kg·K)
PR_T: float = 0.9          # turbulent Prandtl number
SUTHERLAND_C1: float = 1.458e-6  # Sutherland constant C1
SUTHERLAND_S: float = 110.4      # Sutherland constant S (K)
FUSION_DELTA: float = 5e-4       # wall-distance fusion threshold


# ============================================================================
# 1. NETWORK ARCHITECTURE
# ============================================================================

class GlobalNetwork(nn.Module):
    """Network 1: Global flow field predictor (6 hidden layers, 400 neurons)."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 6   # x, y, A5, A6, P_in, T_in

        # Input layer
        layers.append(nn.Linear(input_dim, 400))
        layers.append(nn.ReLU())

        # 6 Hidden layers
        for _ in range(6):
            layers.append(nn.Linear(400, 400))
            layers.append(nn.ReLU())

        # Output layer: 9 variables
        # [ρ, u, v, P, T, UU, VV, UV, μ_eff]
        layers.append(nn.Linear(400, 9))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier uniform init (weights), zero init (biases)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoundaryNetwork(nn.Module):
    """Network 2: Near-wall P/T predictor (6 hidden layers, 100 neurons)."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 6   # x_b, y_b, A5, A6, P_in, T_in

        # Input layer
        layers.append(nn.Linear(input_dim, 100))
        layers.append(nn.ReLU())

        # 6 Hidden layers
        for _ in range(6):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.ReLU())

        # Output layer: 2 variables [P_b, T_b]
        layers.append(nn.Linear(100, 2))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier uniform init (weights), zero init (biases)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_b: torch.Tensor) -> torch.Tensor:
        return self.net(x_b)


class LE_PINN(nn.Module):
    """Locally Enhanced Physics-Informed Neural Network with fusion."""

    def __init__(self, threshold_delta: float = FUSION_DELTA) -> None:
        super().__init__()
        self.global_net = GlobalNetwork()
        self.boundary_net = BoundaryNetwork()
        self.delta = threshold_delta

    def forward(
        self,
        inputs: torch.Tensor,
        wall_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fused LE-PINN forward pass.

        Args:
            inputs: ``(N, 6)`` — ``[x, y, A5, A6, P_in, T_in]``
            wall_distances: ``(N, 1)`` — minimum distance to wall for each point

        Returns:
            ``(N, 9)`` — ``[ρ, u, v, P, T, UU, VV, UV, μ_eff]``
        """
        global_preds = self.global_net(inputs)
        boundary_preds = self.boundary_net(inputs)

        near_wall_mask = (wall_distances < self.delta).squeeze(-1)
        fused_outputs = global_preds.clone()

        if near_wall_mask.any():
            fused_outputs[near_wall_mask, 3] = boundary_preds[near_wall_mask, 0]  # P
            fused_outputs[near_wall_mask, 4] = boundary_preds[near_wall_mask, 1]  # T

        return fused_outputs


# ============================================================================
# 2. NORMALIZER
# ============================================================================

class MinMaxNormalizer:
    """Min-max normalizer with ε guard against division by zero."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.data_min: Optional[torch.Tensor] = None
        self.data_max: Optional[torch.Tensor] = None

    def fit(self, data: torch.Tensor) -> "MinMaxNormalizer":
        self.data_min = data.min(dim=0).values
        self.data_max = data.max(dim=0).values
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        assert self.data_min is not None and self.data_max is not None, "Call fit() first"
        return (data - self.data_min) / (self.data_max - self.data_min + self.epsilon)

    def inverse_transform(self, data_norm: torch.Tensor) -> torch.Tensor:
        assert self.data_min is not None and self.data_max is not None, "Call fit() first"
        return data_norm * (self.data_max - self.data_min + self.epsilon) + self.data_min

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)


# ============================================================================
# 3. PHYSICS LOSS — 2D RANS RESIDUALS
# ============================================================================

def compute_rans_residuals(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    R: float = R_GAS,
    cp: float = CP,
    Pr_t: float = PR_T,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D RANS PDE residuals + ideal gas EOS via automatic differentiation.

    All spatial derivatives are computed w.r.t. the *full* ``inputs`` tensor and
    the x / y components are extracted from the resulting Jacobian columns.
    This avoids second-order autograd failures that occur when differentiating
    w.r.t. sliced sub-tensors.

    Args:
        inputs: ``(N, 6)`` with ``requires_grad=True`` — ``[x, y, A5, A6, P_in, T_in]``
        outputs: ``(N, 9)`` — predicted ``[ρ, u, v, P, T, UU, VV, UV, μ_eff]``
        R: specific gas constant [J/(kg·K)]
        cp: specific heat [J/(kg·K)]
        Pr_t: turbulent Prandtl number

    Returns:
        Tuple of 5 residual tensors:
        ``(res_mass, res_xmom, res_ymom, res_energy, res_eos)``
    """
    rho    = outputs[:, 0:1]
    u      = outputs[:, 1:2]
    v      = outputs[:, 2:3]
    P      = outputs[:, 3:4]
    T      = outputs[:, 4:5]
    UU     = outputs[:, 5:6]
    VV     = outputs[:, 6:7]
    UV     = outputs[:, 7:8]
    mu_eff = outputs[:, 8:9]

    ones = torch.ones_like(rho)

    def _grad_wrt_inputs(y: torch.Tensor) -> torch.Tensor:
        """Gradient of scalar field y w.r.t. full inputs → (N, 6)."""
        g = torch.autograd.grad(
            y, inputs, grad_outputs=ones, create_graph=True, retain_graph=True,
            allow_unused=True,
        )[0]
        return g if g is not None else torch.zeros_like(inputs)

    # --- First derivatives (extract x=col0, y=col1 from full Jacobian) ---
    grad_rho  = _grad_wrt_inputs(rho)
    drho_dx, drho_dy = grad_rho[:, 0:1], grad_rho[:, 1:2]

    grad_u = _grad_wrt_inputs(u)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]

    grad_v = _grad_wrt_inputs(v)
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    grad_P = _grad_wrt_inputs(P)
    dP_dx, dP_dy = grad_P[:, 0:1], grad_P[:, 1:2]

    grad_T = _grad_wrt_inputs(T)
    dT_dx, dT_dy = grad_T[:, 0:1], grad_T[:, 1:2]

    grad_UU = _grad_wrt_inputs(UU)
    dUU_dx = grad_UU[:, 0:1]

    grad_UV = _grad_wrt_inputs(UV)
    dUV_dx, dUV_dy = grad_UV[:, 0:1], grad_UV[:, 1:2]

    grad_VV = _grad_wrt_inputs(VV)
    dVV_dy = grad_VV[:, 1:2]

    # --- Second derivatives (viscous terms) ---
    # Differentiate the first derivative scalars again w.r.t. full inputs.
    grad_du_dx = _grad_wrt_inputs(du_dx)
    d2u_dx2 = grad_du_dx[:, 0:1]

    grad_du_dy = _grad_wrt_inputs(du_dy)
    d2u_dy2 = grad_du_dy[:, 1:2]

    grad_dv_dx = _grad_wrt_inputs(dv_dx)
    d2v_dx2 = grad_dv_dx[:, 0:1]

    grad_dv_dy = _grad_wrt_inputs(dv_dy)
    d2v_dy2 = grad_dv_dy[:, 1:2]

    grad_dT_dx = _grad_wrt_inputs(dT_dx)
    d2T_dx2 = grad_dT_dx[:, 0:1]

    grad_dT_dy = _grad_wrt_inputs(dT_dy)
    d2T_dy2 = grad_dT_dy[:, 1:2]

    # ---- 1. Mass continuity: ∂(ρu)/∂x + ∂(ρv)/∂y = 0 ----
    res_mass = (rho * du_dx + u * drho_dx) + (rho * dv_dy + v * drho_dy)

    # ---- 2. X-momentum ----
    res_xmom = (
        rho * (u * du_dx + v * du_dy) + dP_dx
        - mu_eff * (d2u_dx2 + d2u_dy2)
        + rho * (dUU_dx + dUV_dy)
    )

    # ---- 3. Y-momentum ----
    res_ymom = (
        rho * (u * dv_dx + v * dv_dy) + dP_dy
        - mu_eff * (d2v_dx2 + d2v_dy2)
        + rho * (dUV_dx + dVV_dy)
    )

    # ---- 4. Energy ----
    k_eff = mu_eff * cp / Pr_t
    res_energy = rho * cp * (u * dT_dx + v * dT_dy) - k_eff * (d2T_dx2 + d2T_dy2)

    # ---- 5. Ideal gas EOS: P − ρ R T = 0 ----
    res_eos = P - rho * R * T

    return res_mass, res_xmom, res_ymom, res_energy, res_eos


# ============================================================================
# 4. BOUNDARY CONDITION LOSS
# ============================================================================

def compute_wall_bc_loss(
    model: LE_PINN,
    wall_inputs: torch.Tensor,
    wall_normals: torch.Tensor,
) -> torch.Tensor:
    """
    Wall boundary condition loss: no-slip + adiabatic wall.

    Args:
        model: LE_PINN instance
        wall_inputs: ``(M, 6)`` with ``requires_grad=True``
        wall_normals: ``(M, 2)`` unit outward normals ``[n_x, n_y]``

    Returns:
        Scalar BC loss
    """
    wall_distances = torch.zeros(wall_inputs.size(0), 1, device=wall_inputs.device)
    preds = model(wall_inputs, wall_distances)

    u_wall = preds[:, 1:2]
    v_wall = preds[:, 2:3]
    T_wall = preds[:, 4:5]

    L_u = (u_wall ** 2).mean()
    L_v = (v_wall ** 2).mean()

    # Neumann BC for temperature: ∂T/∂n = 0
    ones = torch.ones_like(T_wall)
    grad_T_result = torch.autograd.grad(
        T_wall, wall_inputs, grad_outputs=ones,
        create_graph=True, retain_graph=True, allow_unused=True,
    )[0]
    if grad_T_result is not None:
        dT_dx = grad_T_result[:, 0:1]
        dT_dy = grad_T_result[:, 1:2]
    else:
        dT_dx = torch.zeros(wall_inputs.size(0), 1, device=wall_inputs.device)
        dT_dy = torch.zeros(wall_inputs.size(0), 1, device=wall_inputs.device)

    dT_dn = dT_dx * wall_normals[:, 0:1] + dT_dy * wall_normals[:, 1:2]
    L_T = (dT_dn ** 2).mean()

    return L_u + L_v + L_T


# ============================================================================
# 5. ADAPTIVE LOSS WEIGHTING
# ============================================================================

class AdaptiveLossWeighting:
    """Sigmoid-based dynamic weighting for data / physics / BC losses."""

    def __init__(
        self,
        max_epochs: int = 5000,
        alpha_data: float = 5.0,
        alpha_bc: float = 3.0,
        base_physics: float = 0.1,
    ) -> None:
        self.max_epochs = max_epochs
        self.alpha_data = alpha_data
        self.alpha_bc = alpha_bc
        self.base_physics = base_physics

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def compute_weights(self, epoch: int) -> Tuple[float, float, float]:
        """Return ``(λ_data, λ_physics, λ_bc)`` for the current epoch."""
        t = epoch / max(self.max_epochs, 1)
        # Data weight decays over training (starts high, decreases)
        lambda_data = 1.0 - self._sigmoid(self.alpha_data * (t - 0.5))
        # Physics weight grows as data weight shrinks
        lambda_physics = self._sigmoid(self.alpha_data * (t - 0.5)) + self.base_physics
        # BC weight grows steadily
        lambda_bc = self._sigmoid(self.alpha_bc * (t - 0.3))
        return lambda_data, lambda_physics, lambda_bc


# ============================================================================
# 6. GEOMETRY HELPERS
# ============================================================================

def generate_wall_geometry(
    NPR: float = 6.5,
    AR: float = 1.53,
    Throat_Radius: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Generate upper and lower wall coordinates from the 2D nozzle profile.

    Returns:
        ``(upper_wall, lower_wall, geometry_dict)``
            - upper_wall: ``(W, 2)`` array of ``(x, y)`` wall points
            - lower_wall: ``(W, 2)`` array of ``(x, -y)`` wall points (mirror)
            - geometry_dict: contains ``A5``, ``A6``, radii, etc.
    """
    from scripts.visualization.nozzle_2d_geometry import generate_nozzle_profile

    profile = generate_nozzle_profile(NPR=NPR, AR=AR, Throat_Radius=Throat_Radius)
    upper_wall = np.stack([profile.x, profile.y], axis=1)   # (W, 2)
    lower_wall = np.stack([profile.x, -profile.y], axis=1)  # (W, 2) mirror

    return upper_wall, lower_wall, profile.geometry


def compute_wall_distances(
    query_points: torch.Tensor,
    wall_points: torch.Tensor,
) -> torch.Tensor:
    """
    Minimum Euclidean distance from each query point to any wall point.

    Args:
        query_points: ``(N, 2)`` — ``(x, y)``
        wall_points: ``(W, 2)`` — all wall coordinates

    Returns:
        ``(N, 1)`` — min distance to wall
    """
    dists = torch.cdist(query_points, wall_points)  # (N, W)
    return dists.min(dim=1).values.unsqueeze(1)      # (N, 1)


def compute_wall_normals(wall_x: np.ndarray, wall_y: np.ndarray) -> np.ndarray:
    """
    Compute outward-pointing unit normals along a wall contour.

    Uses central finite differences.  For the upper wall the outward normal
    points toward the centreline (negative y for a convex-outward wall) so we
    choose ``(-dy, dx)`` and flip the sign to ensure it points **into** the
    flow domain (away from the solid wall).

    Returns:
        ``(M, 2)`` array of unit normals at segment midpoints
        (M = len(wall_x) - 1).
    """
    dx = np.diff(wall_x)
    dy = np.diff(wall_y)

    # Tangent vector is (dx, dy); normal is (-dy, dx) — rotated 90° CCW.
    normals = np.stack([-dy, dx], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = normals / norms

    # Ensure normals point inward (toward centreline y=0) for upper wall.
    # For the upper wall (y > 0), inward means n_y < 0.
    # If average n_y is positive, flip.
    if normals[:, 1].mean() > 0:
        normals = -normals

    return normals


# ============================================================================
# 7. SYNTHETIC DATA GENERATION
# ============================================================================

def _sutherland_viscosity(T: np.ndarray) -> np.ndarray:
    """Dynamic viscosity via Sutherland's law [Pa·s]."""
    return SUTHERLAND_C1 * T ** 1.5 / (T + SUTHERLAND_S)


def generate_synthetic_training_data(
    n_axial: int = 100,
    n_radial: int = 40,
    NPR: float = 6.5,
    AR: float = 1.53,
    Throat_Radius: float = 0.05,
    P_in: float = 658_612.5,   # Pa (NPR=6.5 × 101325)
    T_in: float = 1700.0,      # K
    gamma: float = GAMMA,
    R: float = R_GAS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic 2D training data from quasi-1D isentropic relations
    extruded across the radial direction.

    Returns:
        ``(inputs, targets, wall_distances)``
            - inputs: ``(N, 6)`` — ``[x, y, A5, A6, P_in, T_in]``
            - targets: ``(N, 9)`` — ``[ρ, u, v, P, T, UU, VV, UV, μ_eff]``
            - wall_distances: ``(N, 1)``
    """
    upper_wall, lower_wall, geom = generate_wall_geometry(NPR, AR, Throat_Radius)
    A5 = geom["A5"]
    A6 = geom["A6"]

    # Build interior grid
    x_wall = upper_wall[:, 0]
    y_upper = upper_wall[:, 1]

    x_range = np.linspace(x_wall.min(), x_wall.max(), n_axial)
    # Interpolate upper wall radius at each x
    r_upper = np.interp(x_range, x_wall, y_upper)

    # Build 2D grid of interior points
    x_list, y_list = [], []
    for i, (xi, ri) in enumerate(zip(x_range, r_upper)):
        # Radial points from centreline to wall (exclude wall itself)
        y_pts = np.linspace(0.0, ri * 0.98, n_radial)
        x_list.append(np.full(n_radial, xi))
        y_list.append(y_pts)

    x_pts = np.concatenate(x_list)
    y_pts = np.concatenate(y_list)
    N = len(x_pts)

    # --- Quasi-1D isentropic flow ---
    # Local cross-section area (axisymmetric): A(x) = π r(x)²
    r_local = np.interp(x_pts, x_wall, y_upper)
    A_local = np.pi * r_local ** 2
    A_throat = A5  # already π r_t²
    area_ratio = A_local / A_throat

    # Mach from area-Mach relation (subsonic branch for converging section,
    # supersonic for diverging) — use Newton iteration.
    mach = np.ones(N) * 0.5  # initial guess (subsonic)
    # Identify throat location
    x_throat = x_wall[np.argmin(y_upper)]

    for _ in range(50):  # Newton iterations
        g = gamma
        gp1 = g + 1.0
        gm1 = g - 1.0
        M2 = mach ** 2

        # f(M) = (A/A*)² − [ (2/(γ+1) (1 + (γ-1)/2 M²))^((γ+1)/(γ-1)) ] / M²
        term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2)
        ar_func = (1.0 / mach) * term ** (0.5 * gp1 / gm1)

        f = ar_func - area_ratio
        # Derivative df/dM via numerical perturbation
        dM = 1e-6
        M_pert = mach + dM
        M2_pert = M_pert ** 2
        term_pert = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2_pert)
        ar_pert = (1.0 / M_pert) * term_pert ** (0.5 * gp1 / gm1)
        df = (ar_pert - area_ratio - f) / dM

        df = np.where(np.abs(df) < 1e-12, 1e-12, df)
        mach = mach - f / df
        mach = np.clip(mach, 0.01, 3.0)

    # For points downstream of throat and where AR > 1, use supersonic branch
    supersonic_mask = (x_pts > x_throat) & (area_ratio > 1.01)
    # Re-solve supersonic branch for those points
    mach_sup = np.ones(N) * 1.5
    for _ in range(50):
        M2 = mach_sup ** 2
        term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M2)
        ar_func = (1.0 / mach_sup) * term ** (0.5 * (gamma + 1.0) / (gamma - 1.0))
        f = ar_func - area_ratio
        dM = 1e-6
        M_pert = mach_sup + dM
        M2_pert = M_pert ** 2
        term_pert = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M2_pert)
        ar_pert = (1.0 / M_pert) * term_pert ** (0.5 * (gamma + 1.0) / (gamma - 1.0))
        df = (ar_pert - area_ratio - f) / dM
        df = np.where(np.abs(df) < 1e-12, 1e-12, df)
        mach_sup = mach_sup - f / df
        mach_sup = np.clip(mach_sup, 1.0, 5.0)

    mach[supersonic_mask] = mach_sup[supersonic_mask]

    # Isentropic relations
    T_local = T_in / (1.0 + 0.5 * (gamma - 1.0) * mach ** 2)
    P_local = P_in * (T_local / T_in) ** (gamma / (gamma - 1.0))
    rho_local = P_local / (R * T_local)
    a_local = np.sqrt(gamma * R * T_local)
    u_local = mach * a_local   # axial velocity
    v_local = np.zeros(N)      # radial velocity ≈ 0 for quasi-1D

    # Reynolds stresses (zero for laminar / synthetic baseline)
    UU = np.zeros(N)
    VV = np.zeros(N)
    UV = np.zeros(N)

    # Effective viscosity from Sutherland
    mu_eff = _sutherland_viscosity(T_local)

    # --- Pack tensors ---
    inputs = np.stack([
        x_pts, y_pts,
        np.full(N, A5), np.full(N, A6),
        np.full(N, P_in), np.full(N, T_in),
    ], axis=1).astype(np.float32)

    targets = np.stack([
        rho_local, u_local, v_local, P_local, T_local,
        UU, VV, UV, mu_eff,
    ], axis=1).astype(np.float32)

    inputs_t = torch.from_numpy(inputs)
    targets_t = torch.from_numpy(targets)

    # Wall distances
    all_walls = np.concatenate([upper_wall, lower_wall], axis=0)
    wall_tensor = torch.from_numpy(all_walls.astype(np.float32))
    query_xy = inputs_t[:, :2]
    wall_dists = compute_wall_distances(query_xy, wall_tensor)

    return inputs_t, targets_t, wall_dists


# ============================================================================
# 8. TRAINING SETUP
# ============================================================================

def setup_training(
    model: LE_PINN,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Create AdamW optimizer and ReduceLROnPlateau scheduler per paper spec.

    Returns:
        ``(optimizer, scheduler)``
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-8,
    )
    return optimizer, scheduler


# ============================================================================
# 9. TRAINING LOOP
# ============================================================================

def train_le_pinn(
    n_epochs: int = 5000,
    NPR: float = 6.5,
    AR: float = 1.53,
    Throat_Radius: float = 0.05,
    P_in: float = 658_612.5,
    T_in: float = 1700.0,
    save_path: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[LE_PINN, Dict[str, list]]:
    """
    Full training loop for the LE-PINN.

    Returns:
        ``(trained_model, history_dict)``
    """
    dev = torch.device(device)

    # ---- Data ----
    inputs, targets, wall_dists = generate_synthetic_training_data(
        NPR=NPR, AR=AR, Throat_Radius=Throat_Radius, P_in=P_in, T_in=T_in,
    )

    # Normalizers
    input_norm = MinMaxNormalizer().fit(inputs)
    output_norm = MinMaxNormalizer().fit(targets)

    inputs_n = input_norm.transform(inputs).to(dev)
    targets_n = output_norm.transform(targets).to(dev)
    wall_dists = wall_dists.to(dev)

    # ---- Wall BC data ----
    upper_wall, lower_wall, geom = generate_wall_geometry(NPR, AR, Throat_Radius)
    wall_normals_upper = compute_wall_normals(upper_wall[:, 0], upper_wall[:, 1])
    wall_normals_lower = compute_wall_normals(lower_wall[:, 0], lower_wall[:, 1])
    # Lower wall normals should point upward (away from wall into domain)
    wall_normals_lower[:, 1] = -wall_normals_lower[:, 1]

    # Build wall input tensors (use midpoints of segments)
    def _midpoints(wall: np.ndarray) -> np.ndarray:
        return 0.5 * (wall[:-1] + wall[1:])

    wall_mid_upper = _midpoints(upper_wall)
    wall_mid_lower = _midpoints(lower_wall)

    n_wall = len(wall_mid_upper) + len(wall_mid_lower)
    wall_xy = np.concatenate([wall_mid_upper, wall_mid_lower], axis=0)
    wall_normals_all = np.concatenate([wall_normals_upper, wall_normals_lower], axis=0)

    wall_inputs_raw = np.column_stack([
        wall_xy[:, 0], wall_xy[:, 1],
        np.full(n_wall, geom["A5"]),
        np.full(n_wall, geom["A6"]),
        np.full(n_wall, P_in),
        np.full(n_wall, T_in),
    ]).astype(np.float32)

    wall_inputs_t = input_norm.transform(torch.from_numpy(wall_inputs_raw)).to(dev).requires_grad_(True)
    wall_normals_t = torch.from_numpy(wall_normals_all.astype(np.float32)).to(dev)

    # ---- Model, optimiser, scheduler ----
    model = LE_PINN().to(dev)
    optimizer, scheduler = setup_training(model)
    weighting = AdaptiveLossWeighting(max_epochs=n_epochs)

    history: Dict[str, list] = {
        "loss_total": [],
        "loss_data": [],
        "loss_physics": [],
        "loss_bc": [],
        "lr": [],
    }

    if verbose:
        print("=" * 70)
        print("LE-PINN TRAINING")
        print("=" * 70)
        print(f"  Epochs:   {n_epochs}")
        print(f"  NPR:      {NPR},  AR: {AR}")
        print(f"  P_in:     {P_in:.0f} Pa,  T_in: {T_in:.0f} K")
        print(f"  Points:   {len(inputs)} interior,  {n_wall} wall BC")
        print("=" * 70)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # ---- 1. Data loss ----
        preds = model(inputs_n, wall_dists)
        loss_data = nn.functional.mse_loss(preds, targets_n)

        # ---- 2. Physics loss ----
        # Need gradients w.r.t. spatial inputs
        inputs_phys = inputs_n.detach().clone().requires_grad_(True)
        preds_phys = model(inputs_phys, wall_dists)

        res_mass, res_xmom, res_ymom, res_energy, res_eos = compute_rans_residuals(
            inputs_phys, preds_phys,
        )
        loss_physics = (
            (res_mass ** 2).mean()
            + (res_xmom ** 2).mean()
            + (res_ymom ** 2).mean()
            + (res_energy ** 2).mean()
            + (res_eos ** 2).mean()
        )

        # ---- 3. BC loss ----
        loss_bc = compute_wall_bc_loss(model, wall_inputs_t, wall_normals_t)

        # ---- 4. Adaptive weighting ----
        lam_d, lam_p, lam_bc = weighting.compute_weights(epoch)
        loss_total = lam_d * loss_data + lam_p * loss_physics + lam_bc * loss_bc

        # ---- Backprop ----
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_total.item())

        # ---- Logging ----
        history["loss_total"].append(loss_total.item())
        history["loss_data"].append(loss_data.item())
        history["loss_physics"].append(loss_physics.item())
        history["loss_bc"].append(loss_bc.item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose and epoch % 100 == 0:
            print(
                f"Ep {epoch:5d} | Total {loss_total.item():.3e} | "
                f"Data {loss_data.item():.3e} | Physics {loss_physics.item():.3e} | "
                f"BC {loss_bc.item():.3e} | λ=({lam_d:.2f},{lam_p:.2f},{lam_bc:.2f}) | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

    # ---- Save ----
    if save_path is not None:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_norm_min": input_norm.data_min,
            "input_norm_max": input_norm.data_max,
            "output_norm_min": output_norm.data_min,
            "output_norm_max": output_norm.data_max,
            "config": {
                "NPR": NPR, "AR": AR, "Throat_Radius": Throat_Radius,
                "P_in": P_in, "T_in": T_in, "n_epochs": n_epochs,
            },
            "seed": RANDOM_SEED,
        }
        torch.save(ckpt, save_path)
        if verbose:
            print(f"\n💾 Checkpoint saved: {save_path}")

    if verbose:
        print("=" * 70)
        print("✅ LE-PINN training complete!")
        print("=" * 70)

    return model, history


# ============================================================================
# 10. CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    model, hist = train_le_pinn(
        n_epochs=2000,
        save_path=str(_REPO_ROOT / "models" / "le_pinn.pt"),
        verbose=True,
    )
