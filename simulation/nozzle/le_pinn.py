"""
Locally Enhanced Physics-Informed Neural Network (LE-PINN) Surrogate Model.

Implements the dual-network architecture from Ma et al. for axisymmetric RANS
nozzle flow field prediction.

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

Physics
-------
- **Axisymmetric RANS**: The y coordinate is treated as the radial distance r.
  Continuity, radial momentum, and energy equations include geometric source
  terms (ρv/r, hoop stress ρv²/r, 1/r diffusion correction).
- **Turbulence closure**: Reynolds stress outputs (UU, VV, UV) are trained
  purely via the data loss (L_data). They are NOT included in the physics
  residuals, which follow an Euler/laminar Navier-Stokes formulation.

Boundary Conditions
-------------------
- Wall no-slip: u = v = 0
- **Adiabatic wall**: ∂T/∂n = 0. The nozzle wall is modeled as perfectly
  insulated. This is acceptable for a proof-of-concept where convective heat
  transfer through the wall is negligible compared to the enthalpy flux.

Training
--------
- Loss: L = λ_data·L_data + λ_physics·L_physics + λ_BC·L_BC
- Adaptive sigmoid-based loss weighting
- Optimizer: AdamW (lr=1e-4, wd=1e-5)
- Scheduler: ReduceLROnPlateau (patience 10, factor 0.5, min_lr 1e-8)
- Init: Xavier uniform weights, zero biases

Wall geometry is sourced from ``scripts/visualization/nozzle_2d_geometry.py``.
Training data is synthetic, generated from isentropic + ideal-gas relations.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
_PHYSICS_FAIL_COUNTER: Dict[str, int] = {"count": 0}


def _safe_torch_load(path: str, **kwargs) -> dict:
    """Load a PyTorch checkpoint with weights_only fallback for older versions."""
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)  # type: ignore[call-overload]


# ============================================================================
# 1. WEIGHT INITIALIZATION
# ============================================================================

def _xavier_init(module: nn.Module) -> None:
    """Xavier uniform init (weights), zero init (biases)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ============================================================================
# 2. NETWORK ARCHITECTURE
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
        _xavier_init(self)

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
        _xavier_init(self)

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
        delta = self.data_max - self.data_min
        const_mask = delta.abs() < self.epsilon
        safe_delta = torch.where(const_mask, torch.ones_like(delta), delta)
        data_norm = (data - self.data_min) / safe_delta
        # If a feature was constant during fit, keep it at 0 in normalized space.
        # This avoids exploding values during inference when callers provide
        # different inputs for that dimension.
        if const_mask.any():
            data_norm = data_norm.clone()
            data_norm[..., const_mask] = 0.0
        return data_norm

    def inverse_transform(self, data_norm: torch.Tensor) -> torch.Tensor:
        assert self.data_min is not None and self.data_max is not None, "Call fit() first"
        delta = self.data_max - self.data_min
        const_mask = delta.abs() < self.epsilon
        safe_delta = torch.where(const_mask, torch.zeros_like(delta), delta)
        data = data_norm * safe_delta + self.data_min
        if const_mask.any():
            data = data.clone()
            data[..., const_mask] = self.data_min[..., const_mask]
        return data

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.fit(data).transform(data)


# ============================================================================
# 3. PHYSICS LOSS — 2D RANS RESIDUALS
# ============================================================================

def _compute_reference_scales(
    outputs: torch.Tensor,
    geometry: str = "axisymmetric",
) -> dict:
    """
    Compute physically meaningful reference scales for residual normalization.

    Args:
        outputs: ``(N, 9)`` — predicted ``[ρ, u, v, P, T, UU, VV, UV, μ_eff]``
        geometry: ``"axisymmetric"`` or ``"planar"``

    Returns:
        Dictionary with reference scales for mass, momentum, energy, and EOS residuals.
    """
    rho = outputs[:, 0:1]
    u = outputs[:, 1:2]
    v = outputs[:, 2:3]
    P = outputs[:, 3:4]
    T = outputs[:, 4:5]

    # Reference scales (using mean + std to avoid outliers)
    rho_ref = rho.abs().mean() + rho.std() + 1e-8
    u_ref = u.abs().mean() + u.std() + 1e-8
    v_ref = v.abs().mean() + v.std() + 1e-8
    P_ref = P.abs().mean() + P.std() + 1e-8
    T_ref = T.abs().mean() + T.std() + 1e-8

    # Mass residual scale: rho * u (convective flux magnitude)
    mass_scale = rho_ref * u_ref

    # Momentum residual scale: rho * u^2 (inertia term magnitude)
    momentum_scale = rho_ref * u_ref ** 2

    # Energy residual scale: rho * cp * u * T (convective enthalpy flux)
    energy_scale = rho_ref * CP * u_ref * T_ref

    # EOS residual scale: P (pressure magnitude)
    eos_scale = P_ref

    return {
        "mass": mass_scale,
        "momentum": momentum_scale,
        "energy": energy_scale,
        "eos": eos_scale,
    }


def compute_rans_residuals(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    R: float = R_GAS,
    cp: float = CP,
    Pr_t: float = PR_T,
    geometry: str = "axisymmetric",
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute RANS PDE residuals + ideal gas EOS via automatic differentiation.

    Supports two geometry modes:
    - ``"axisymmetric"``: y is treated as radial distance r; geometric source
      terms (ρv/r, hoop stress ρv²/r, 1/r diffusion) are included.
    - ``"planar"``: Cartesian 2D; axisymmetric source terms are dropped.

    Reynolds stress outputs (UU, VV, UV) are NOT included in the physics
    residuals. They are trained purely via the data loss (L_data), avoiding
    the need for a turbulence closure model in the physics loss.

    All spatial derivatives are computed w.r.t. the *full* ``inputs`` tensor and
    the x / y components are extracted from the resulting Jacobian columns.

    Args:
        inputs: ``(N, 6)`` with ``requires_grad=True`` — ``[x, y, A5, A6, P_in, T_in]``
        outputs: ``(N, 9)`` — predicted ``[ρ, u, v, P, T, UU, VV, UV, μ_eff]``
        R: specific gas constant [J/(kg·K)]
        cp: specific heat [J/(kg·K)]
        Pr_t: turbulent Prandtl number
        geometry: ``"axisymmetric"`` (default, engine nozzle) or ``"planar"``
            (Sajben 2D diffuser). Controls whether axisymmetric source terms
            are included in continuity, y-momentum, and energy residuals.
        normalize: If ``True`` (default), scale residuals by physically meaningful
            reference values to prevent magnitude imbalance during training.

    Returns:
        Tuple of 5 residual tensors:
        ``(res_mass, res_xmom, res_ymom, res_energy, res_eos)``
        Each tensor is dimensionless-scaled if ``normalize=True``.
    """
    rho    = outputs[:, 0:1]
    u      = outputs[:, 1:2]
    v      = outputs[:, 2:3]
    P      = outputs[:, 3:4]
    T      = outputs[:, 4:5]
    # UU, VV, UV (indices 5-7) are NOT used in physics loss — trained via data loss only
    mu_eff = outputs[:, 8:9]

    # Radial coordinate (y) with epsilon guard for centreline
    y = inputs[:, 1:2]
    eps = 1e-8
    y_safe = y + eps  # prevent division by zero at r = 0

    ones = torch.ones_like(rho)

    def _grad_wrt_inputs(y_field: torch.Tensor) -> torch.Tensor:
        """Gradient of scalar field y w.r.t. full inputs → (N, 6)."""
        g = torch.autograd.grad(
            y_field, inputs, grad_outputs=ones, create_graph=True, retain_graph=True,
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

    # --- Second derivatives (viscous terms) ---
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

    # ---- 1. Continuity ----
    # Axisymmetric: ∂(ρu)/∂x + ∂(ρv)/∂y + ρv/r = 0
    # Planar:       ∂(ρu)/∂x + ∂(ρv)/∂y         = 0
    res_mass = (
        (rho * du_dx + u * drho_dx)
        + (rho * dv_dy + v * drho_dy)
    )
    if geometry == "axisymmetric":
        res_mass = res_mass + rho * v / y_safe

    # ---- 2. X-momentum (no Reynolds stress terms — Euler/laminar N-S) ----
    res_xmom = (
        rho * (u * du_dx + v * du_dy) + dP_dx
        - mu_eff * (d2u_dx2 + d2u_dy2)
    )

    # ---- 3. Y-momentum (no Reynolds stress terms) ----
    # Axisymmetric: includes hoop stress -ρv²/r
    # Planar: no hoop stress term
    res_ymom = (
        rho * (u * dv_dx + v * dv_dy) + dP_dy
        - mu_eff * (d2v_dx2 + d2v_dy2)
    )
    if geometry == "axisymmetric":
        res_ymom = res_ymom - rho * v ** 2 / y_safe  # hoop stress: -ρv²/r

    # ---- 4. Energy ----
    # Axisymmetric: includes 1/r diffusion correction (+dT_dy/r)
    # Planar: standard 2D Laplacian only
    k_eff = mu_eff * cp / Pr_t
    res_energy = (
        rho * cp * (u * dT_dx + v * dT_dy)
        - k_eff * (d2T_dx2 + d2T_dy2)
    )
    if geometry == "axisymmetric":
        res_energy = res_energy - k_eff * dT_dy / y_safe

    # ---- 5. Ideal gas EOS: P − ρ R T = 0 ----
    res_eos = P - rho * R * T

    # ---- Phase 2: Apply dimensionless scaling ----
    if normalize:
        scales = _compute_reference_scales(outputs, geometry=geometry)
        res_mass = res_mass / scales["mass"]
        res_xmom = res_xmom / scales["momentum"]
        res_ymom = res_ymom / scales["momentum"]
        res_energy = res_energy / scales["energy"]
        res_eos = res_eos / scales["eos"]

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

    NOTE: The nozzle wall is modeled as ADIABATIC (∂T/∂n = 0).
    This is acceptable for a proof-of-concept where heat transfer through
    the nozzle wall is negligible compared to the convective enthalpy flux.
    For production fidelity, replace the Neumann BC with a convective BC:
        k·∂T/∂n = h·(T_wall − T_ambient)
    where h is a convective heat transfer coefficient.

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

    def _newton_mach(area_ratio, gamma, m_init, m_lo, m_hi, n_iter=50):
        """Solve area-Mach relation via Newton iteration."""
        mach = np.full_like(area_ratio, m_init)
        gp1, gm1 = gamma + 1.0, gamma - 1.0
        for _ in range(n_iter):
            M2 = mach ** 2
            term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2)
            ar_func = (1.0 / mach) * term ** (0.5 * gp1 / gm1)
            f = ar_func - area_ratio
            dM = 1e-6
            M_pert = mach + dM
            term_pert = (2.0 / gp1) * (1.0 + 0.5 * gm1 * (M_pert ** 2))
            ar_pert = (1.0 / M_pert) * term_pert ** (0.5 * gp1 / gm1)
            df = (ar_pert - area_ratio - f) / dM
            df = np.where(np.abs(df) < 1e-12, 1e-12, df)
            mach = mach - f / df
            mach = np.clip(mach, m_lo, m_hi)
        return mach

    # Subsonic branch (converging section)
    mach = _newton_mach(area_ratio, gamma, m_init=0.5, m_lo=0.01, m_hi=3.0)

    # Supersonic branch (diverging section downstream of throat)
    x_throat = x_wall[np.argmin(y_upper)]
    supersonic_mask = (x_pts > x_throat) & (area_ratio > 1.01)
    mach_sup = _newton_mach(area_ratio, gamma, m_init=1.5, m_lo=1.0, m_hi=5.0)
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
# 7b. SAJBEN EXPERIMENTAL DATA PARSING
# ============================================================================

def _expand_fortran_tokens(text: str) -> list:
    """
    Expand Fortran list-directed repeat notation in a whitespace/comma token stream.

    E.g. ``'6*2.59650254'`` → ``[2.59650254] * 6``.
    Plain numeric strings are returned as single-element lists.
    Non-numeric tokens (e.g. column headers) are silently skipped.
    """
    import re
    values: list = []
    for token in re.split(r"[,\s]+", text.strip()):
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            parts = token.split("*", 1)
            try:
                count = int(parts[0])
                val = float(parts[1])
                values.extend([val] * count)
            except (ValueError, IndexError):
                pass
        else:
            try:
                values.append(float(token))
            except ValueError:
                pass
    return values


def parse_sajben_experimental_data(filepath: str) -> Dict[str, object]:
    """
    Parse the Sajben / Hseih et al. (1987) Mach-0.46 experimental data file.

    File structure (``data.Mach46.txt``):

    * Lines 1-15  : Header.  Throat height H = 0.14435 ft = 4.407 cm.
      All distances normalised by H.
    * Lines 17-53 : Surface static pressure table.  Two side-by-side columns:

      .. code-block::

         TOP WALL (X/H*, P/P0)  |  BOTTOM WALL (X/H*, P/P0)

      The top-wall column has 37 rows; the bottom-wall column has 36 rows
      (the final top-wall line has no matching bottom-wall entry).

    * Lines 61-207 : Axial velocity profiles at four X/H stations:
      1.729, 2.882, 4.611, 6.340.  Each section lists rows of
      ``(Y/H, X-VELOCITY [m/s])``.

    Parameters
    ----------
    filepath : str
        Path to ``data.Mach46.txt``.

    Returns
    -------
    dict with keys:

    ``top_wall``
        ``{'xh': ndarray, 'pp0': ndarray}`` — X/H* and P/P₀ on the top wall.
    ``bot_wall``
        ``{'xh': ndarray, 'pp0': ndarray}`` — same for the bottom wall.
    ``vel_profiles``
        ``{station_label: {'yh': ndarray, 'u_ms': ndarray}}``
        where station labels are ``'1.729'``, ``'2.882'``, ``'4.611'``, ``'6.340'``.
    ``H_m``
        Throat height in metres (0.044014).
    """
    fpath = Path(filepath)
    if not fpath.exists():
        raise FileNotFoundError(f"Sajben data file not found: {fpath}")

    lines = fpath.read_text().splitlines()

    # ------------------------------------------------------------------ #
    # 1. Surface static pressure                                           #
    # ------------------------------------------------------------------ #
    top_xh: list = []
    top_pp0: list = []
    bot_xh: list = []
    bot_pp0: list = []

    # Locate header line containing both "X/H*" and "P/P0"
    pressure_start: Optional[int] = None
    for idx, line in enumerate(lines):
        if "X/H*" in line and "P/P0" in line:
            pressure_start = idx + 1
            break
    if pressure_start is None:
        raise ValueError("Could not find pressure-table header in data file.")

    i = pressure_start
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        # Stop at the velocity profiles section
        if "VELOCITY" in line.upper() or ("X/H" in line and "=" in line):
            break
        # Collect numeric tokens from this line (up to 4 floats)
        nums = []
        for tok in line.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        if len(nums) >= 2:
            top_xh.append(nums[0])
            top_pp0.append(nums[1])
        if len(nums) >= 4:
            bot_xh.append(nums[2])
            bot_pp0.append(nums[3])

    # ------------------------------------------------------------------ #
    # 2. Velocity profiles                                                 #
    # ------------------------------------------------------------------ #
    # Nominal X/H labels (as they appear in file headers)
    _STATION_CANON = {
        "1.729": "1.729", "1.73": "1.729",
        "2.882": "2.882", "2.88": "2.882",
        "4.611": "4.611", "4.61": "4.611",
        "6.340": "6.340", "6.34": "6.340",
    }

    vel_profiles: Dict[str, dict] = {}
    current_station: Optional[str] = None
    yh_buf: list = []
    u_buf: list = []

    for line in lines:
        line_s = line.strip()
        # Detect station header: "X/H = 1.729" or "X/H= 2.882" etc.
        if "X/H" in line_s and "=" in line_s:
            # Flush previous station buffer
            if current_station is not None and yh_buf:
                vel_profiles[current_station] = {
                    "yh":  np.array(yh_buf, dtype=np.float32),
                    "u_ms": np.array(u_buf, dtype=np.float32),
                }
            yh_buf, u_buf = [], []
            current_station = None
            # Extract numeric value after '='
            try:
                val_str = line_s.split("=", 1)[1].strip().split()[0]
                # Strip trailing non-numeric chars
                val_str_clean = val_str.rstrip(",.:;")
                # Map to canonical label
                # Try exact match first, then first-3-decimal match
                if val_str_clean in _STATION_CANON:
                    current_station = _STATION_CANON[val_str_clean]
                else:
                    # Try matching first 4 characters
                    for k, v in _STATION_CANON.items():
                        if val_str_clean.startswith(k[:4]):
                            current_station = v
                            break
            except (IndexError, ValueError):
                pass
            continue

        if current_station is None:
            continue
        # Skip blank lines and column header lines
        if not line_s or "Y/H" in line_s or "M/S" in line_s or "VELOCITY" in line_s.upper():
            continue

        nums = []
        for tok in line_s.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        if len(nums) >= 2:
            yh_buf.append(nums[0])
            u_buf.append(nums[1])

    # Flush final station
    if current_station is not None and yh_buf:
        vel_profiles[current_station] = {
            "yh":  np.array(yh_buf, dtype=np.float32),
            "u_ms": np.array(u_buf, dtype=np.float32),
        }

    H_m = 0.14435 * 0.3048  # 0.14435 ft → metres

    return {
        "top_wall": {
            "xh":  np.array(top_xh,  dtype=np.float32),
            "pp0": np.array(top_pp0, dtype=np.float32),
        },
        "bot_wall": {
            "xh":  np.array(bot_xh,  dtype=np.float32),
            "pp0": np.array(bot_pp0, dtype=np.float32),
        },
        "vel_profiles": vel_profiles,
        "H_m": float(H_m),
    }


def parse_sajben_geometry(filepath: str) -> Dict[str, object]:
    """
    Parse a Plot3D formatted 2D geometry file (``sajben.x.fmt``).

    Format:

    * Line 1  : number of blocks (= 1 for Sajben).
    * Line 2  : ``ni, nj, nk`` grid dimensions (81 × 51 × 1 for Sajben).
    * Remaining lines: ``ni*nj`` x-values then ``ni*nj`` y-values stored in
      Fortran i-major order (i varies fastest).  Fortran list-directed
      repeat notation ``N*val`` is expanded transparently.  A z-coordinate
      array (all zeros for a 2D problem) is ignored if present.

    The Sajben coordinates are in **inches**.  Confirmation: the x-range
    (−6.998 to +14.984 in) equals the experimental X/H range (−4.04 to +8.65)
    scaled by H = 1.7322 in (0.14435 ft).  The maximum upper-wall y (≈ 2.598 in)
    equals the exit channel height = 1.5 H ✓.

    Parameters
    ----------
    filepath : str
        Path to ``sajben.x.fmt``.

    Returns
    -------
    dict with keys:

    ``x_in``, ``y_in`` : (ni, nj) arrays — coordinates in inches.
    ``x_m``,  ``y_m``  : (ni, nj) arrays — coordinates in metres.
    ``ni``, ``nj``     : grid dimensions.
    ``H_in``, ``H_m``  : throat height in inches / metres (= min upper-wall y).
    ``x_vec``          : (ni,) 1-D x-vector (same for all j, in inches).
    ``upper_wall_y_m`` : (ni,) upper wall y-coordinates in metres.
    ``lower_wall_y_m`` : (ni,) lower wall y-coordinates in metres (≈ 0).
    ``AR_entrance``    : entrance-to-throat area ratio (2D: height ratio).
    ``AR_exit``        : exit-to-throat area ratio.
    """
    fpath = Path(filepath)
    if not fpath.exists():
        raise FileNotFoundError(f"Plot3D geometry file not found: {fpath}")

    text = fpath.read_text()
    lines = text.splitlines()

    # Header
    n_blocks = int(lines[0].strip())
    if n_blocks != 1:
        raise ValueError(f"Expected 1 block, got {n_blocks}")

    dim_parts = lines[1].replace(",", " ").split()
    ni, nj, nk = int(dim_parts[0]), int(dim_parts[1]), int(dim_parts[2])
    n_pts = ni * nj  # points per coordinate array

    # Parse all numeric values from remaining lines
    all_values: list = []
    for line in lines[2:]:
        all_values.extend(_expand_fortran_tokens(line))

    # Determine how many coordinate arrays are present
    if len(all_values) >= 3 * n_pts:
        n_arrays = 3
    elif len(all_values) >= 2 * n_pts:
        n_arrays = 2
    else:
        raise ValueError(
            f"Insufficient values in Plot3D file: got {len(all_values)}, "
            f"expected ≥ {2 * n_pts} for ni={ni}, nj={nj}."
        )

    vals = np.array(all_values[: n_arrays * n_pts], dtype=np.float64)
    x_flat = vals[:n_pts]
    y_flat = vals[n_pts : 2 * n_pts]

    # Reshape: Fortran i-major order → (nj, ni) → transpose to (ni, nj)
    x_2d = x_flat.reshape((nj, ni)).T  # (ni, nj)
    y_2d = y_flat.reshape((nj, ni)).T  # (ni, nj)

    INCH_TO_M = 0.0254
    x_m = x_2d * INCH_TO_M
    y_m = y_2d * INCH_TO_M

    lower_wall_y_m = y_m[:, 0]
    upper_wall_y_m = y_m[:, -1]

    # Throat height = minimum upper-wall y (narrowest cross-section)
    H_in = float(y_2d[:, -1].min())
    H_m  = H_in * INCH_TO_M

    x_vec = x_2d[:, 0]  # (ni,) same for all j

    # Area ratios (2D channel: area ∝ height = upper_y − lower_y)
    h_entrance = float(y_2d[0, -1]  - y_2d[0,  0])
    h_exit      = float(y_2d[-1, -1] - y_2d[-1, 0])
    h_throat    = H_in
    AR_entrance = h_entrance / h_throat if h_throat > 0 else float("nan")
    AR_exit     = h_exit     / h_throat if h_throat > 0 else float("nan")

    return {
        "x_in": x_2d,
        "y_in": y_2d,
        "x_m":  x_m,
        "y_m":  y_m,
        "ni":   ni,
        "nj":   nj,
        "H_in": H_in,
        "H_m":  H_m,
        "x_vec": x_vec,
        "upper_wall_y_m": upper_wall_y_m,
        "lower_wall_y_m": lower_wall_y_m,
        "AR_entrance": AR_entrance,
        "AR_exit":     AR_exit,
    }


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
    geometry: str = "axisymmetric",
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

        # ---- 2. Physics loss (de-normalized for correct chain-rule scaling) ----
        # De-normalize to physical space so PDE coefficients (cp, R, Pr_t) are
        # applied to physical quantities, not min-max-scaled surrogates.
        inputs_phys_raw = input_norm.inverse_transform(
            inputs_n.detach().clone()
        ).requires_grad_(True)
        # Re-run forward pass through normalizer→model→denormalizer so autograd
        # traces the full chain: physical inputs → normalized → model → denorm
        preds_phys_raw = model(input_norm.transform(inputs_phys_raw), wall_dists)
        preds_phys_denorm = output_norm.inverse_transform(preds_phys_raw)

        res_mass, res_xmom, res_ymom, res_energy, res_eos = compute_rans_residuals(
            inputs_phys_raw, preds_phys_denorm, geometry=geometry, normalize=True,
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
        # Phase 3: Step scheduler on stable monitor metric (data loss) instead of
        # adaptively weighted total loss to prevent premature LR collapse
        scheduler.step(loss_data.item())

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


# ============================================================================
# 11. CFD DATA FINE-TUNING HELPERS
# ============================================================================

def _estimate_wall_distances(
    query_xy: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y_abs: torch.Tensor,
    n_bins: int = 60,
) -> torch.Tensor:
    """
    Estimate minimum wall distance for each query point.

    Approximates the wall envelope as the maximum ``|y|`` observed in
    ``ref_x / ref_y_abs`` within the same x-bin as the query point.
    Returns ``(N, 1)`` non-negative distances.
    """
    x_min = float(ref_x.min().item())
    x_max = float(ref_x.max().item())
    bin_width = max((x_max - x_min) / n_bins, 1e-12)

    bin_r = ((ref_x - x_min) / bin_width).long().clamp(0, n_bins - 1)
    bin_q = ((query_xy[:, 0] - x_min) / bin_width).long().clamp(0, n_bins - 1)

    max_y_per_bin = torch.zeros(n_bins, dtype=ref_y_abs.dtype)
    for b in range(n_bins):
        mask = bin_r == b
        if mask.any():
            max_y_per_bin[b] = ref_y_abs[mask].max()
        elif b > 0:
            max_y_per_bin[b] = max_y_per_bin[b - 1]

    wall_y = max_y_per_bin[bin_q]
    dist = (wall_y - query_xy[:, 1].abs()).clamp(min=0.0).unsqueeze(1)
    return dist


def _safe_physics_loss(
    model: "LE_PINN",
    inputs_n: torch.Tensor,
    wall_dists: torch.Tensor,
    input_norm: Optional["MinMaxNormalizer"] = None,
    output_norm: Optional["MinMaxNormalizer"] = None,
    geometry: str = "axisymmetric",
    max_failures: int = 10,
) -> torch.Tensor:
    """
    Compute the RANS physics loss in physical (de-normalized) space.

    Failures (e.g. autograd graph issues) are reported via ``warnings.warn``.
    After ``max_failures`` consecutive failures a ``RuntimeError`` is raised to
    prevent silent training drift.

    Args:
        model: The LE_PINN model.
        inputs_n: Normalized input tensor ``(N, 6)``.
        wall_dists: Wall distance tensor ``(N, 1)``.
        input_norm: Input normalizer. When provided, inputs are de-normalized
            before PDE evaluation so residuals are physically meaningful.
        output_norm: Output normalizer for de-normalizing predictions.
        geometry: ``"axisymmetric"`` or ``"planar"`` — passed to
            :func:`compute_rans_residuals`.
        max_failures: Raise ``RuntimeError`` after this many consecutive
            failures (default 10). Guards against silent zero-loss drift.
    """
    try:
        if input_norm is not None and output_norm is not None:
            # De-normalize to physical space for correct PDE scaling.
            # Handles partial normalizers (e.g. output_norm fitted on first 5
            # columns only): denormalize covered columns, keep rest as-is.
            inputs_phys_raw = input_norm.inverse_transform(
                inputs_n.detach().clone()
            ).requires_grad_(True)
            preds_phys_raw = model(input_norm.transform(inputs_phys_raw), wall_dists)
            n_norm_cols = output_norm.data_min.shape[0]  # type: ignore[union-attr]
            preds_denorm_part = output_norm.inverse_transform(preds_phys_raw[:, :n_norm_cols])
            if n_norm_cols < preds_phys_raw.shape[1]:
                preds_denorm = torch.cat(
                    [preds_denorm_part, preds_phys_raw[:, n_norm_cols:]], dim=1
                )
            else:
                preds_denorm = preds_denorm_part
            if n_norm_cols < 9 and preds_denorm.shape[1] > 8:
                T_phys = preds_denorm[:, 4:5].clamp(min=100.0, max=5000.0)
                mu_eff = SUTHERLAND_C1 * T_phys.pow(1.5) / (T_phys + SUTHERLAND_S)
                preds_denorm = preds_denorm.clone()
                preds_denorm[:, 8:9] = mu_eff.clamp(min=1e-6, max=1e-3)
            res_mass, res_xmom, res_ymom, res_energy, res_eos = compute_rans_residuals(
                inputs_phys_raw, preds_denorm, geometry=geometry, normalize=True,
            )
        else:
            # Fallback: operate on normalized space (legacy behaviour)
            inputs_phys = inputs_n.detach().clone().requires_grad_(True)
            preds_phys = model(inputs_phys, wall_dists)
            res_mass, res_xmom, res_ymom, res_energy, res_eos = compute_rans_residuals(
                inputs_phys, preds_phys, geometry=geometry, normalize=True,
            )
        loss = (
            (res_mass ** 2).mean()
            + (res_xmom ** 2).mean()
            + (res_ymom ** 2).mean()
            + (res_energy ** 2).mean()
            + (res_eos ** 2).mean()
        )
        _PHYSICS_FAIL_COUNTER["count"] = 0
        return loss
    except Exception as exc:
        _PHYSICS_FAIL_COUNTER["count"] += 1
        count = _PHYSICS_FAIL_COUNTER["count"]
        warnings.warn(
            f"Physics loss failed ({count}/{max_failures}): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        if count >= max_failures:
            raise RuntimeError(
                f"Physics loss failed {max_failures} consecutive times. "
                "Aborting training to prevent silent drift."
            ) from exc
        return torch.tensor(0.0, device=inputs_n.device)


# ============================================================================
# 12. CFD DATA FINE-TUNING
# ============================================================================

def finetune_on_cfd_data(
    dataset_path: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    save_path: Optional[str] = None,
    n_epochs: int = 500,
    lr: float = 1e-5,
    val_fraction: float = 0.2,
    physics_loss_weight: float = 0.05,
    physics_max_points: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
    geometry: str = "planar",
) -> Tuple["LE_PINN", Dict[str, list]]:
    """
    Fine-tune the LE-PINN on real CFD data from ``master_shock_dataset.pt``.

    The dataset provides inputs ``(N, 6) = [x, y, A5, A6, P_in, T_in]`` and
    targets ``(N, 9) = [rho, u, v, P, T, 0, 0, 0, 0]``.  Only the first 5
    target columns are used for the data loss; columns 5–8 are zero-padded and
    ignored.

    Warnings from physics / BC loss computation are **non-fatal** — they are
    emitted via :func:`warnings.warn` and the failed term is set to zero for
    that step so training can continue.

    Args:
        dataset_path: Path to ``master_shock_dataset.pt``.  Defaults to
            ``<repo>/data/processed/master_shock_dataset.pt``.
        pretrained_path: Optional path to a pre-trained LE-PINN checkpoint.
            If ``None`` a freshly initialised model is fine-tuned.
        save_path: Where to write the fine-tuned checkpoint.  Defaults to
            ``<repo>/models/le_pinn_cfd.pt``.
        n_epochs: Fine-tuning epochs.
        lr: Learning rate (should be lower than initial training, e.g. 1e-5).
        val_fraction: Fraction of data held out for validation.
        physics_loss_weight: Scaling factor for the RANS physics loss term.
            Set to 0 to disable physics loss entirely during fine-tuning.
        physics_max_points: Optional cap on the number of training points used
            for physics loss per epoch. If ``None`` and device is ``"mps"``,
            a default cap is applied to reduce Apple GPU memory pressure.
        device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        verbose: Print epoch progress every 50 steps.

    Returns:
        ``(fine_tuned_model, history_dict)``
    """
    dev = torch.device(device)
    if physics_max_points is None and dev.type == "mps":
        physics_max_points = 16384

    # ---- Resolve default paths ----
    if dataset_path is None:
        dataset_path = str(_REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt")
    if save_path is None:
        save_path = str(_REPO_ROOT / "models" / "le_pinn_cfd.pt")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"CFD dataset not found: {dataset_path}. "
            "Run fetch_and_build_cfd_data.py first."
        )

    # ---- Load dataset (weights_only with fallback for older PyTorch) ----
    cfd = _safe_torch_load(dataset_path)

    inputs_raw: torch.Tensor = cfd["inputs"].float()   # (N, 6)
    targets_raw: torch.Tensor = cfd["targets"].float() # (N, 9)
    weights_raw: Optional[torch.Tensor] = (
        cfd["sample_weights"].float() if "sample_weights" in cfd else None
    )  # (N,) or None

    if inputs_raw.shape[1] != 6:
        raise ValueError(f"Expected inputs with 6 columns, got {inputs_raw.shape[1]}")
    if targets_raw.shape[1] != 9:
        raise ValueError(f"Expected targets with 9 columns, got {targets_raw.shape[1]}")

    # ---- Drop rows with NaN / Inf ----
    valid = (
        torch.isfinite(inputs_raw).all(dim=1)
        & torch.isfinite(targets_raw[:, :5]).all(dim=1)
    )
    n_bad = int((~valid).sum().item())
    if n_bad > 0:
        warnings.warn(
            f"Dropping {n_bad} rows with NaN/Inf values from CFD dataset.",
            RuntimeWarning,
            stacklevel=1,
        )
    inputs_raw = inputs_raw[valid]
    targets_raw = targets_raw[valid]
    if weights_raw is not None:
        weights_raw = weights_raw[valid]

    if len(inputs_raw) == 0:
        raise RuntimeError("No valid rows in CFD dataset after NaN filtering.")

    N = len(inputs_raw)

    # ---- Train / val split ----
    torch.manual_seed(RANDOM_SEED)
    perm = torch.randperm(N)
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    inputs_train = inputs_raw[train_idx]
    targets_train = targets_raw[train_idx]
    inputs_val = inputs_raw[val_idx]
    targets_val = targets_raw[val_idx]
    weights_train = weights_raw[train_idx] if weights_raw is not None else None

    # ---- Normalizers fitted on training split only ----
    input_norm = MinMaxNormalizer().fit(inputs_train)
    # Fit output normalizer on first 5 columns only; skip cols with zero range
    output_norm_5 = MinMaxNormalizer().fit(targets_train[:, :5])

    inputs_train_n = input_norm.transform(inputs_train).to(dev)
    targets_train_5n = output_norm_5.transform(targets_train[:, :5]).to(dev)
    inputs_val_n = input_norm.transform(inputs_val).to(dev)
    targets_val_5n = output_norm_5.transform(targets_val[:, :5]).to(dev)

    # ---- Wall distance approximation from dataset x/y envelope ----
    ref_x = inputs_train[:, 0]
    ref_y_abs = inputs_train[:, 1].abs()
    wall_dists_train = _estimate_wall_distances(inputs_train[:, :2], ref_x, ref_y_abs).to(dev)
    wall_dists_val = _estimate_wall_distances(inputs_val[:, :2], ref_x, ref_y_abs).to(dev)

    # ---- Load or init model ----
    model = LE_PINN().to(dev)
    if pretrained_path is not None:
        if not Path(pretrained_path).exists():
            warnings.warn(
                f"Pretrained checkpoint not found: {pretrained_path}. "
                "Starting fine-tuning from random init.",
                RuntimeWarning,
                stacklevel=1,
            )
        else:
            try:
                ckpt = _safe_torch_load(pretrained_path, map_location=dev)
                model.load_state_dict(ckpt["model_state_dict"])
                if verbose:
                    print(f"Loaded pretrained weights from {pretrained_path}")
            except Exception as exc:
                warnings.warn(
                    f"Failed to load pretrained checkpoint ({exc}). "
                    "Starting fine-tuning from random init.",
                    RuntimeWarning,
                    stacklevel=1,
                )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-9
    )
    weighting = AdaptiveLossWeighting(max_epochs=n_epochs)

    history: Dict[str, list] = {
        "loss_total": [],
        "loss_data": [],
        "loss_physics": [],
        "val_loss": [],
        "lr": [],
    }

    if verbose:
        print("=" * 70)
        print("LE-PINN FINE-TUNING ON CFD DATA")
        print("=" * 70)
        print(f"  Dataset : {dataset_path}")
        print(f"  Train/Val: {n_train} / {n_val}   Epochs: {n_epochs}   LR: {lr}")
        print(f"  Physics weight: {physics_loss_weight}")
        if physics_max_points is not None and physics_max_points > 0:
            print(f"  Physics points/epoch cap: {physics_max_points}")
        print("=" * 70)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Data loss on first 5 output columns (weighted if sample_weights present)
        preds = model(inputs_train_n, wall_dists_train)
        if weights_train is not None:
            w = weights_train.to(dev)
            w = w / (w.mean() + 1e-12)  # normalize so mean weight ≈ 1
            sq_err = (preds[:, :5] - targets_train_5n) ** 2  # (N, 5)
            loss_data = (sq_err.mean(dim=1) * w).mean()
        else:
            loss_data = nn.functional.mse_loss(preds[:, :5], targets_train_5n)

        # Physics loss (non-fatal; returns zero tensor on failure)
        if physics_loss_weight > 0:
            phys_inputs = inputs_train_n
            phys_walls = wall_dists_train
            if (
                physics_max_points is not None
                and physics_max_points > 0
                and inputs_train_n.shape[0] > physics_max_points
            ):
                phys_idx = torch.randperm(
                    inputs_train_n.shape[0], device=inputs_train_n.device
                )[:physics_max_points]
                phys_inputs = inputs_train_n[phys_idx]
                phys_walls = wall_dists_train[phys_idx]
            loss_physics = _safe_physics_loss(
                model, phys_inputs, phys_walls,
                input_norm=input_norm,
                output_norm=output_norm_5,
                geometry=geometry,
            )
        else:
            loss_physics = torch.tensor(0.0, device=dev)

        lam_d, lam_p, _ = weighting.compute_weights(epoch)
        loss_total = lam_d * loss_data + physics_loss_weight * lam_p * loss_physics

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Phase 3: Step scheduler on stable monitor metric (data loss)
        scheduler.step(loss_data.item())

        history["loss_total"].append(loss_total.item())
        history["loss_data"].append(loss_data.item())
        history["loss_physics"].append(float(loss_physics.item()))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(inputs_val_n, wall_dists_val)
                val_loss = nn.functional.mse_loss(
                    val_preds[:, :5], targets_val_5n
                ).item()
            history["val_loss"].append(val_loss)
            model.train()
            if verbose:
                print(
                    f"Ep {epoch:4d} | Total {loss_total.item():.3e} | "
                    f"Data {loss_data.item():.3e} | "
                    f"Physics {float(loss_physics.item()):.3e} | "
                    f"Val {val_loss:.3e} | "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}"
                )

    # ---- Save checkpoint ----
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_norm_min": input_norm.data_min,
            "input_norm_max": input_norm.data_max,
            "output_norm_min": output_norm_5.data_min,
            "output_norm_max": output_norm_5.data_max,
            "config": {
                "n_epochs": n_epochs,
                "lr": lr,
                "physics_loss_weight": physics_loss_weight,
                "dataset": dataset_path,
            },
            "seed": RANDOM_SEED,
        },
        save_path,
    )
    if verbose:
        print(f"\nFine-tuned checkpoint saved: {save_path}")
        print("=" * 70)

    return model, history


# ============================================================================
# 13. VALIDATION
# ============================================================================

_CFD_VAR_NAMES = ["rho", "u", "v", "P", "T"]


def validate_le_pinn(
    model: "LE_PINN",
    dataset_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the LE-PINN against the master CFD dataset.

    Loads normaliser stats from ``checkpoint_path`` when provided.  Skips
    target columns whose range is zero (e.g. constant ``T=900 K`` or ``v=0``).
    All failure modes that are non-critical (missing checkpoint, NaN rows,
    zero-range columns) are handled with :func:`warnings.warn` instead of
    raising, so the caller always receives a result dict (possibly partial).

    Args:
        model: ``LE_PINN`` instance.
        dataset_path: Path to ``master_shock_dataset.pt``.
        checkpoint_path: Optional checkpoint to reload model weights / norms.
        device: Compute device.
        verbose: Print a results table.

    Returns:
        Dict with keys ``rmse_<var>`` and ``r2_<var>`` for each non-trivial
        target column.
    """
    dev = torch.device(device)

    if dataset_path is None:
        dataset_path = str(_REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"CFD dataset not found: {dataset_path}")

    cfd = _safe_torch_load(dataset_path)

    inputs_raw: torch.Tensor = cfd["inputs"].float()
    targets_raw: torch.Tensor = cfd["targets"].float()

    # Drop bad rows
    valid = (
        torch.isfinite(inputs_raw).all(dim=1)
        & torch.isfinite(targets_raw[:, :5]).all(dim=1)
    )
    n_bad = int((~valid).sum().item())
    if n_bad > 0:
        warnings.warn(
            f"Dropping {n_bad} invalid rows during validation.",
            RuntimeWarning,
            stacklevel=1,
        )
    inputs_raw = inputs_raw[valid]
    targets_raw = targets_raw[valid]

    # ---- Load checkpoint if provided ----
    input_norm = MinMaxNormalizer().fit(inputs_raw)
    output_norm_5: Optional[MinMaxNormalizer] = None  # set from checkpoint when available

    if checkpoint_path is not None:
        if not Path(checkpoint_path).exists():
            warnings.warn(
                f"Checkpoint not found: {checkpoint_path}. "
                "Using current model weights and fitting normaliser on full dataset.",
                RuntimeWarning,
                stacklevel=1,
            )
        else:
            try:
                ckpt = _safe_torch_load(checkpoint_path, map_location=dev)
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(dev)
                # Input normalizer
                in_min = ckpt.get("input_norm_min")
                in_max = ckpt.get("input_norm_max")
                if in_min is not None and in_max is not None:
                    input_norm.data_min = in_min.to(inputs_raw.device)
                    input_norm.data_max = in_max.to(inputs_raw.device)
                # Output normalizer (needed to de-normalize predictions)
                out_min = ckpt.get("output_norm_min")
                out_max = ckpt.get("output_norm_max")
                if out_min is not None and out_max is not None:
                    # Some older checkpoints store 9-column output normalizers
                    # while validation compares only first 5 targets.
                    if out_min.shape[0] >= 5 and out_max.shape[0] >= 5:
                        out_min = out_min[:5]
                        out_max = out_max[:5]
                    else:
                        warnings.warn(
                            "Checkpoint output normalizer has fewer than 5 columns; "
                            "validation de-normalization may be unavailable.",
                            RuntimeWarning,
                            stacklevel=1,
                        )
                    output_norm_5 = MinMaxNormalizer()
                    output_norm_5.data_min = out_min.to("cpu")
                    output_norm_5.data_max = out_max.to("cpu")
            except Exception as exc:
                warnings.warn(
                    f"Failed to load checkpoint for validation ({exc}). "
                    "Using current model weights.",
                    RuntimeWarning,
                    stacklevel=1,
                )

    inputs_n = input_norm.transform(inputs_raw).to(dev)
    wall_dists = _estimate_wall_distances(
        inputs_raw[:, :2], inputs_raw[:, 0], inputs_raw[:, 1].abs()
    ).to(dev)

    model.eval()
    model.to(dev)
    with torch.no_grad():
        preds = model(inputs_n, wall_dists).cpu()

    targets_5 = targets_raw[:, :5].cpu()

    # De-normalize predictions if we have an output normalizer from the checkpoint
    if output_norm_5 is not None:
        preds_5 = output_norm_5.inverse_transform(preds[:, :5])
    else:
        warnings.warn(
            "No output normaliser available; comparing raw model outputs against "
            "un-normalised targets — RMSE/R² may be misleading.",
            RuntimeWarning,
            stacklevel=1,
        )
        preds_5 = preds[:, :5]

    metrics: Dict[str, float] = {}
    rows = []
    for i, name in enumerate(_CFD_VAR_NAMES):
        t = targets_5[:, i]
        p = preds_5[:, i]

        t_range = float((t.max() - t.min()).item())
        if t_range < 1e-10:
            warnings.warn(
                f"Target column '{name}' has zero range — skipping metric.",
                RuntimeWarning,
                stacklevel=1,
            )
            continue

        rmse = float(torch.sqrt(((p - t) ** 2).mean()).item())
        ss_res = float(((p - t) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        metrics[f"rmse_{name}"] = rmse
        metrics[f"r2_{name}"] = r2
        rows.append((name, rmse, r2))

    if verbose:
        print("\n" + "=" * 50)
        print("LE-PINN VALIDATION RESULTS (CFD DATA)")
        print("=" * 50)
        print(f"{'Variable':<10} {'RMSE':>14} {'R²':>10}")
        print("-" * 36)
        for name, rmse, r2 in rows:
            print(f"{name:<10} {rmse:>14.4e} {r2:>10.4f}")
        print("=" * 50)

    return metrics


def _restore_normalizer(
    data_min: torch.Tensor,
    data_max: torch.Tensor,
) -> "MinMaxNormalizer":
    """Rebuild a ``MinMaxNormalizer`` from checkpoint statistics."""
    normalizer = MinMaxNormalizer()
    normalizer.data_min = data_min.detach().clone().to("cpu").float()
    normalizer.data_max = data_max.detach().clone().to("cpu").float()
    return normalizer


def _safe_rel_error(
    predicted: float,
    actual: float,
    floor: float = 1e-12,
) -> float:
    """Relative error with a small denominator floor."""
    denom = max(abs(actual), floor)
    return float(abs(predicted - actual) / denom)


def _converging_area_profile(
    x: np.ndarray,
    A_in: float,
    A_exit: float,
    length: float,
) -> np.ndarray:
    """Cosine-shaped converging nozzle area profile."""
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length:.6f}")
    return A_in + (A_exit - A_in) * (1.0 - np.cos(np.pi * x / (2.0 * length)))


def _build_scalar_consistent_profile(
    inlet_state: Dict[str, float],
    exit_state: Dict[str, float],
    A_in: float,
    A_exit: float,
    length: float,
    thermo_props: Dict[str, float],
    n_points: int,
) -> Dict[str, np.ndarray]:
    """
    Build a 1D analytical profile consistent with a prescribed scalar exit state.

    This is used for fallback rows so the returned profile matches the scalar
    exit state reported by the wrapper.
    """
    x = np.linspace(0.0, length, n_points, dtype=np.float64)
    A_x = _converging_area_profile(x, A_in, A_exit, length)
    gamma = thermo_props["gamma"]
    cp = thermo_props["cp"]
    gas_constant = thermo_props["R"]

    s = (A_in - A_x) / max(A_in - A_exit, 1e-12)
    s = np.clip(s, 0.0, 1.0)

    p_in = float(inlet_state["p"])
    T_in = float(inlet_state["T"])
    u_in = float(inlet_state["u"])
    p_exit = float(exit_state["p"])

    p = p_in - (p_in - p_exit) * s
    p_ratio = np.clip(p / max(p_in, 1e-12), 1e-12, None)
    T = T_in * p_ratio ** ((gamma - 1.0) / gamma)
    u_sq = u_in ** 2 + 2.0 * cp * (T_in - T)
    u = np.sqrt(np.maximum(u_sq, 0.0))
    rho = p / np.maximum(gas_constant * T, 1e-12)

    return {
        "x": x.astype(np.float32),
        "rho": rho.astype(np.float32),
        "u": u.astype(np.float32),
        "p": p.astype(np.float32),
        "T": T.astype(np.float32),
    }


def _resolve_le_checkpoint_path(model_path: str) -> Path:
    """Resolve the requested LE checkpoint, including the engine-unified fallback."""
    requested = Path(model_path)
    if requested.exists():
        return requested

    if requested.name == "le_pinn_unified.pt":
        fallback = requested.with_name("le_pinn_engine_unified.pt")
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"LE-PINN checkpoint not found: {model_path}")


def _adapt_nozzle_result_to_le_schema(
    nozzle_result: Dict[str, Any],
    inlet_state: Dict[str, float],
    m_dot: float,
    force_reason: str,
) -> Dict[str, Any]:
    """Adapt ``run_nozzle_pinn`` output to the LE wrapper schema."""
    inlet_verification = dict(nozzle_result.get("inlet_verification", {}))
    relative_errors = dict(inlet_verification.get("relative_errors", {}))
    mass_conservation = dict(nozzle_result.get("mass_conservation", {}))

    rho_match = float(relative_errors.get("rho", 1.0))
    u_match = float(relative_errors.get("u", 1.0))
    p_match = float(relative_errors.get("p", 1.0))
    T_match = float(relative_errors.get("T", 1.0))

    max_error_fraction = float(mass_conservation.get("error_pct", 100.0)) / 100.0
    adapted = {
        "exit_state": dict(nozzle_result["exit_state"]),
        "thrust_total": float(nozzle_result["thrust_total"]),
        "thrust_momentum": float(nozzle_result["thrust_momentum"]),
        "thrust_pressure": float(nozzle_result["thrust_pressure"]),
        "thrust_model": nozzle_result["thrust_model"],
        "used_fallback": True,
        "fallback_reason": force_reason,
        "inlet_verification": {
            "rho_match": rho_match,
            "u_match": u_match,
            "p_match": p_match,
            "T_match": T_match,
            "inlet_actual": inlet_verification.get(
                "inlet_actual",
                {k: float(inlet_state[k]) for k in ("rho", "u", "p", "T")},
            ),
            "inlet_predicted": inlet_verification.get("inlet_predicted", {}),
            "relative_errors": relative_errors,
            "max_error": float(inlet_verification.get("max_error", max(rho_match, u_match, p_match, T_match))),
            "check_passed": bool(inlet_verification.get("check_passed", False)),
        },
        "mass_conservation": {
            "inlet_mdot": float(mass_conservation.get("m_dot_input", m_dot)),
            "exit_mdot": float(mass_conservation.get("m_dot_exit_predicted", 0.0)),
            "max_error": max_error_fraction,
            "m_dot_input": float(mass_conservation.get("m_dot_input", m_dot)),
            "m_dot_inlet_predicted": float(mass_conservation.get("m_dot_inlet_predicted", 0.0)),
            "m_dot_exit_predicted": float(mass_conservation.get("m_dot_exit_predicted", 0.0)),
            "inlet_error_pct": float(mass_conservation.get("inlet_error_pct", 100.0)),
            "exit_error_pct": float(mass_conservation.get("exit_error_pct", 100.0)),
            "error_pct": float(mass_conservation.get("error_pct", 100.0)),
            "check_passed": bool(mass_conservation.get("check_passed", False)),
        },
    }
    if "profiles" in nozzle_result:
        adapted["profiles"] = nozzle_result["profiles"]
    return adapted


def run_le_pinn(
    model_path: str,
    inlet_state: Dict[str, float],
    ambient_p: float,
    A_in: float,
    A_exit: float,
    length: float,
    thermo_props: Dict[str, float],
    m_dot: float,
    n_axial: int = 50,
    n_radial: int = 20,
    device: str = "cpu",
    return_profile: bool = False,
    thrust_model: str = "static_test_stand",
) -> Dict[str, Any]:
    """
    Run the LE-PINN with a ``run_nozzle_pinn``-compatible interface.

    The LE-PINN predicts a 2D flowfield that is reduced to a 1D centerline
    profile for thrust and diagnostics. When the LE checkpoint is unavailable
    or the predicted exit state is non-physical, the wrapper falls back to an
    analytical nozzle model.
    """
    from simulation.nozzle.nozzle import analytical_isentropic_nozzle, run_nozzle_pinn

    if m_dot <= 0.0:
        raise ValueError(f"Mass flow must be positive, got {m_dot:.3f} kg/s")
    if inlet_state["u"] <= 0.0:
        raise ValueError(f"Inlet velocity must be positive, got {inlet_state['u']:.3f} m/s")
    if inlet_state["p"] <= 0.0:
        raise ValueError(f"Inlet pressure must be positive, got {inlet_state['p']:.3f} Pa")
    if inlet_state["T"] <= 0.0:
        raise ValueError(f"Inlet temperature must be positive, got {inlet_state['T']:.3f} K")
    if thermo_props["cp"] <= 0.0:
        raise ValueError(f"cp must be positive, got {thermo_props['cp']:.3f}")
    if thermo_props["R"] <= 0.0:
        raise ValueError(f"R must be positive, got {thermo_props['R']:.3f}")
    if thermo_props["gamma"] <= 1.0:
        raise ValueError(f"gamma must be > 1.0, got {thermo_props['gamma']:.3f}")
    if A_in <= 0.0 or A_exit <= 0.0:
        raise ValueError(f"Nozzle areas must be positive, got A_in={A_in:.6f}, A_exit={A_exit:.6f}")
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length:.6f} m")
    if n_axial < 2:
        raise ValueError(f"n_axial must be >= 2, got {n_axial}")
    if n_radial < 1:
        raise ValueError(f"n_radial must be >= 1, got {n_radial}")

    if inlet_state["p"] <= ambient_p:
        warnings.warn(
            f"Nozzle inlet pressure ({inlet_state['p']/1e3:.2f} kPa) <= ambient "
            f"({ambient_p/1e3:.2f} kPa). Over-expanded turbine. Thrust may be low/negative."
        )

    try:
        resolved_model_path = _resolve_le_checkpoint_path(model_path)
    except FileNotFoundError:
        regular_model_path = _REPO_ROOT / "models" / "nozzle_pinn.pt"
        if not regular_model_path.exists():
            raise FileNotFoundError(
                f"LE-PINN checkpoint not found ({model_path}) and regular PINN fallback "
                f"checkpoint is also missing: {regular_model_path}"
            ) from None

        delegated = run_nozzle_pinn(
            model_path=str(regular_model_path),
            inlet_state=inlet_state,
            ambient_p=ambient_p,
            A_in=A_in,
            A_exit=A_exit,
            length=length,
            thermo_props=thermo_props,
            m_dot=m_dot,
            device=device,
            return_profile=return_profile,
            thrust_model=thrust_model,
        )
        delegated_reason = "LE checkpoint unavailable; delegated to run_nozzle_pinn"
        if delegated.get("fallback_reason"):
            delegated_reason = f"{delegated_reason}; {delegated['fallback_reason']}"
        adapted = _adapt_nozzle_result_to_le_schema(
            delegated,
            inlet_state=inlet_state,
            m_dot=m_dot,
            force_reason=delegated_reason,
        )
        if return_profile and delegated.get("used_fallback"):
            adapted["profiles"] = _build_scalar_consistent_profile(
                inlet_state=inlet_state,
                exit_state=adapted["exit_state"],
                A_in=A_in,
                A_exit=A_exit,
                length=length,
                thermo_props=thermo_props,
                n_points=n_axial,
            )
        return adapted

    ckpt = _safe_torch_load(str(resolved_model_path), map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state_dict: {resolved_model_path}")
    if "input_norm_min" not in ckpt or "input_norm_max" not in ckpt:
        raise KeyError(f"Checkpoint missing input normalizer statistics: {resolved_model_path}")
    if "output_norm_min" not in ckpt or "output_norm_max" not in ckpt:
        raise KeyError(f"Checkpoint missing output normalizer statistics: {resolved_model_path}")

    input_norm = _restore_normalizer(ckpt["input_norm_min"], ckpt["input_norm_max"])
    output_norm = _restore_normalizer(ckpt["output_norm_min"][:5], ckpt["output_norm_max"][:5])

    dev = torch.device(device)
    model = LE_PINN().to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_stations = np.linspace(0.0, length, n_axial, dtype=np.float32)
    A_x = _converging_area_profile(x_stations.astype(np.float64), A_in, A_exit, length).astype(np.float32)
    radii = np.sqrt(np.maximum(A_x, 1e-12) / np.pi).astype(np.float32)

    inputs_rows = []
    y_rows = []
    for x_i, radius_i in zip(x_stations, radii):
        y_i = np.linspace(0.0, 0.98 * radius_i, n_radial, dtype=np.float32)
        y_rows.append(y_i)
        inputs_rows.append(
            np.column_stack(
                [
                    np.full(n_radial, x_i, dtype=np.float32),
                    y_i,
                    np.full(n_radial, A_in, dtype=np.float32),
                    np.full(n_radial, A_exit, dtype=np.float32),
                    np.full(n_radial, inlet_state["p"], dtype=np.float32),
                    np.full(n_radial, inlet_state["T"], dtype=np.float32),
                ]
            )
        )

    inputs_phys = torch.from_numpy(np.concatenate(inputs_rows, axis=0)).float()
    y_matrix = np.stack(y_rows, axis=0)
    x_ref = torch.from_numpy(x_stations).float()
    r_ref = torch.from_numpy(radii).float()
    wall_dists = _estimate_wall_distances(inputs_phys[:, :2], x_ref, r_ref).to(dev)
    inputs_norm = input_norm.transform(inputs_phys).to(dev)

    with torch.no_grad():
        preds_raw = model(inputs_norm, wall_dists).cpu()

    preds_denorm_5 = output_norm.inverse_transform(preds_raw[:, :5])
    preds_denorm = torch.cat([preds_denorm_5, preds_raw[:, 5:]], dim=1)
    T_safe = preds_denorm[:, 4:5].clamp(min=100.0, max=5000.0)
    mu_eff = SUTHERLAND_C1 * T_safe.pow(1.5) / (T_safe + SUTHERLAND_S)
    preds_denorm = preds_denorm.clone()
    preds_denorm[:, 8:9] = mu_eff.clamp(min=1e-6, max=1e-3)

    preds_grid = preds_denorm.reshape(n_axial, n_radial, preds_denorm.shape[1])
    centerline_idx = torch.from_numpy(np.argmin(y_matrix, axis=1)).long()
    centerline = preds_grid[torch.arange(n_axial), centerline_idx]

    rho_1d = centerline[:, 0].numpy().astype(np.float64)
    u_raw_1d = centerline[:, 1].numpy().astype(np.float64)
    p_1d = centerline[:, 3].numpy().astype(np.float64)
    T_1d = centerline[:, 4].numpy().astype(np.float64)

    raw_centerline_finite = (
        np.isfinite(rho_1d).all()
        and np.isfinite(u_raw_1d).all()
        and np.isfinite(p_1d).all()
        and np.isfinite(T_1d).all()
    )
    rho_positive = raw_centerline_finite and np.all(rho_1d > 0.0)

    if raw_centerline_finite and rho_positive:
        u_corrected = m_dot / np.maximum(rho_1d * A_x.astype(np.float64), 1e-12)
        raw_mdot_profile = rho_1d * u_raw_1d * A_x.astype(np.float64)
        max_raw_mdot_error = float(
            np.max(np.abs(raw_mdot_profile - m_dot) / max(abs(m_dot), 1e-12))
        )
        exit_mdot_raw = float(raw_mdot_profile[-1])
        rho_match = _safe_rel_error(float(rho_1d[0]), float(inlet_state["rho"]))
        u_match = _safe_rel_error(float(u_raw_1d[0]), float(inlet_state["u"]), floor=1.0)
        p_match = _safe_rel_error(float(p_1d[0]), float(inlet_state["p"]))
        T_match = _safe_rel_error(float(T_1d[0]), float(inlet_state["T"]))
        inlet_predicted = {
            "rho": float(rho_1d[0]),
            "u": float(u_raw_1d[0]),
            "p": float(p_1d[0]),
            "T": float(T_1d[0]),
        }
    else:
        u_corrected = np.full(n_axial, np.nan, dtype=np.float64)
        raw_mdot_profile = np.full(n_axial, np.nan, dtype=np.float64)
        max_raw_mdot_error = 1.0
        exit_mdot_raw = 0.0
        rho_match = 1.0
        u_match = 1.0
        p_match = 1.0
        T_match = 1.0
        inlet_predicted = {"rho": 0.0, "u": 0.0, "p": 0.0, "T": 0.0}

    relative_errors = {
        "rho": rho_match,
        "u": u_match,
        "p": p_match,
        "T": T_match,
    }

    used_fallback = False
    fallback_reason: Optional[str] = None
    exit_state = {
        "rho": float(rho_1d[-1]) if raw_centerline_finite else np.nan,
        "u": float(u_corrected[-1]) if np.isfinite(u_corrected[-1]) else np.nan,
        "p": float(p_1d[-1]) if raw_centerline_finite else np.nan,
        "T": float(T_1d[-1]) if raw_centerline_finite else np.nan,
    }

    invalid_reasons = []
    if not raw_centerline_finite:
        invalid_reasons.append("LE centerline contains NaN/Inf")
    elif not rho_positive:
        invalid_reasons.append("LE centerline density is non-positive")
    elif max(relative_errors.values()) > 0.50:
        invalid_reasons.append(
            f"inlet mismatch ({max(relative_errors.values()) * 100.0:.2f}%)"
        )
    elif max_raw_mdot_error > 0.50:
        invalid_reasons.append(
            f"mass conservation error ({max_raw_mdot_error * 100.0:.2f}%)"
        )

    for name in ("rho", "u", "p", "T"):
        value = exit_state[name]
        if not np.isfinite(value) or value <= 0.0:
            invalid_reasons.append(f"non-physical exit {name}")

    if raw_centerline_finite and rho_positive:
        if exit_state["T"] < max(80.0, 0.18 * float(inlet_state["T"])) or exit_state["T"] > 1.2 * float(inlet_state["T"]):
            invalid_reasons.append("implausible exit temperature")
        if exit_state["p"] < max(5_000.0, 0.03 * float(inlet_state["p"])) or exit_state["p"] > 1.2 * float(inlet_state["p"]):
            invalid_reasons.append("implausible exit pressure")
        if exit_state["u"] < 0.0 or exit_state["u"] > 2_500.0:
            invalid_reasons.append("implausible exit velocity")
        if exit_state["rho"] <= 1e-8 or exit_state["rho"] > 20.0:
            invalid_reasons.append("implausible exit density")

    profile_payload = {
        "x": x_stations.astype(np.float32),
        "rho": rho_1d.astype(np.float32),
        "u": u_corrected.astype(np.float32),
        "p": p_1d.astype(np.float32),
        "T": T_1d.astype(np.float32),
    }

    if invalid_reasons:
        fallback_reason = "; ".join(dict.fromkeys(invalid_reasons))
        fallback_exit = analytical_isentropic_nozzle(
            inlet_state=inlet_state,
            ambient_p=ambient_p,
            A_exit=A_exit,
            thermo_props=thermo_props,
            m_dot=m_dot,
        )
        exit_state = {
            "rho": float(fallback_exit["rho"]),
            "u": float(fallback_exit["u"]),
            "p": float(fallback_exit["p"]),
            "T": float(fallback_exit["T"]),
        }
        profile_payload = _build_scalar_consistent_profile(
            inlet_state=inlet_state,
            exit_state=exit_state,
            A_in=A_in,
            A_exit=A_exit,
            length=length,
            thermo_props=thermo_props,
            n_points=n_axial,
        )
        used_fallback = True

    if thrust_model == "static_test_stand":
        thrust_momentum = m_dot * exit_state["u"]
    elif thrust_model == "incremental_nozzle":
        thrust_momentum = m_dot * (exit_state["u"] - inlet_state["u"])
    else:
        raise ValueError(
            f"Invalid thrust_model: {thrust_model}. "
            "Use 'static_test_stand' or 'incremental_nozzle'"
        )

    thrust_pressure = (exit_state["p"] - ambient_p) * A_exit
    thrust_total = thrust_momentum + thrust_pressure

    inlet_verification = {
        "rho_match": float(rho_match),
        "u_match": float(u_match),
        "p_match": float(p_match),
        "T_match": float(T_match),
        "inlet_actual": {k: float(inlet_state[k]) for k in ("rho", "u", "p", "T")},
        "inlet_predicted": inlet_predicted,
        "relative_errors": relative_errors,
        "max_error": float(max(relative_errors.values())),
        "check_passed": bool(max(relative_errors.values()) < 0.05),
    }
    mass_conservation = {
        "inlet_mdot": float(m_dot),
        "exit_mdot": float(exit_mdot_raw),
        "max_error": float(max_raw_mdot_error),
        "m_dot_input": float(m_dot),
        "m_dot_inlet_predicted": float(raw_mdot_profile[0]) if np.isfinite(raw_mdot_profile[0]) else 0.0,
        "m_dot_exit_predicted": float(exit_mdot_raw),
        "inlet_error_pct": float(
            abs(raw_mdot_profile[0] - m_dot) / max(abs(m_dot), 1e-12) * 100.0
        ) if np.isfinite(raw_mdot_profile[0]) else 100.0,
        "exit_error_pct": float(
            abs(exit_mdot_raw - m_dot) / max(abs(m_dot), 1e-12) * 100.0
        ) if np.isfinite(exit_mdot_raw) else 100.0,
        "error_pct": float(max_raw_mdot_error * 100.0),
        "check_passed": bool(max_raw_mdot_error < 0.05),
    }

    result: Dict[str, Any] = {
        "exit_state": exit_state,
        "thrust_total": float(thrust_total),
        "thrust_momentum": float(thrust_momentum),
        "thrust_pressure": float(thrust_pressure),
        "thrust_model": thrust_model,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "inlet_verification": inlet_verification,
        "mass_conservation": mass_conservation,
    }
    if return_profile:
        result["profiles"] = profile_payload
    return result
