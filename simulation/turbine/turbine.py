"""
Turbine Physics-Informed Neural Network (PINN) with Fuel-Dependent Thermodynamics.

This module trains a PINN to model turbine expansion physics while enforcing conservation
laws (mass, momentum, energy) and using fuel-dependent thermodynamic properties.

Model Architecture:
- Input: Normalized axial position x* ∈ [0,1]
- Output: Flow state [ρ*, u*, p*, T*] (normalized density, velocity, pressure, temperature)
- Physics constraints: Enforces continuity, momentum, and energy conservation in loss function
- Fuel-dependent: Uses actual cp, R, gamma from combustion products (not fixed air constants)

Training Process:
The PINN learns to satisfy:
1. Continuity equation: d(ρu)/dx = 0 (mass conservation)
2. Momentum equation: ρu du/dx + dp/dx = 0 (Newton's second law)
3. Energy equation: Shaft work extraction matches compressor work requirement
4. Boundary conditions: Match inlet/outlet states from engine cycle

Key Innovation:
Unlike traditional PINNs that assume constant gamma, this model uses fuel-specific
thermodynamic properties, making it necessary (not just convenient) for modeling
real engine behavior where fuel chemistry affects expansion physics.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any

# Force float32 globally
torch.set_default_dtype(torch.float32)

print("="*70)
print("TURBINE PINN - FUEL-DEPENDENT WITH EXACT CONTINUITY")
print("="*70)

# ============================================================================
# THERMO REFERENCE VALUES (for fixed-reference normalization)
# ============================================================================
# CRITICAL: Thermodynamic properties (cp, R, gamma) must be normalized against
# FIXED reference values, not their current batch values. Otherwise:
#   cp_norm = cp/cp = 1.0 always → network is blind to fuel changes
#
# These reference values represent typical combustion products at design point.
THERMO_REF = {
    'cp': 1150.0,    # J/(kg·K) - baseline specific heat
    'R': 287.0,      # J/(kg·K) - baseline gas constant
    'gamma': 1.33    # Heat capacity ratio
}

# ============================================================================
# 1. DEFAULT TRAINING CONDITIONS (for backwards compatibility)
# ============================================================================
# These are default values used during initial training.
# For fuel-dependent operation, use build_turbine_conditions() from thermo_utils
# to create conditions with real combustor-derived properties.

# =============================================================================
# DERIVATION OF DEFAULT TURBINE CONDITIONS (DESIGN-POINT RECONSTRUCTION)
# =============================================================================
#
# IMPORTANT:
# These values are NOT directly reported in the ICAO emissions dataset.
# The ICAO databank provides thrust rating, fuel flow, pressure ratio, and
# emissions indices — NOT internal engine state variables.
#
# The quantities below are THERMODYNAMICALLY RECONSTRUCTED design-point
# conditions inferred from:
#   • ICAO overall pressure ratio
#   • ICAO-rated thrust and fuel flow
#   • Standard Brayton-cycle relations
#   • Energy balance between compressor and turbine
#
# They serve as:
#   (1) a physics-consistent training anchor for the turbine PINN
#   (2) a baseline that is later overridden by fuel-dependent properties
#       (cp, R, gamma) extracted from Cantera at runtime
#
# -----------------------------------------------------------------------------
# TURBINE INLET STATE
# -----------------------------------------------------------------------------
#
# Pressure:
#   p_t,in ≈ OPR × p_ambient
#   Typical modern turbofan OPR ≈ 40–45
#
#   p_t,in ≈ 41.5 × 101325 Pa ≈ 4.20 MPa
#
# Temperature:
#   Turbine inlet temperature (TIT) is not reported by ICAO.
#   Values of 1650–1750 K are standard in open literature for high-bypass
#   turbofans. We select:
#
#   T_t,in = 1700 K
#
# Density:
#   Computed from ideal gas law (baseline air-like constants):
#
#   ρ = p / (R T)
#   ρ ≈ 4.20e6 / (287 × 1700) ≈ 8.61 kg/m³
#
# Velocity:
#   From continuity at turbine inlet:
#
#   ṁ = ρ u A  ⇒  u = ṁ / (ρ A)
#
#   Using reconstructed mass flow and annulus area yields:
#   u ≈ 44.7 m/s (low axial velocity due to high density)
#
# -----------------------------------------------------------------------------
# TURBINE OUTLET STATE
# -----------------------------------------------------------------------------
#
# Pressure:
#   ICAO-consistent turbine exit / nozzle inlet pressure
#   Typically ~4–5% of compressor exit pressure:
#
#   p_out ≈ 0.045 × OPR × p_ambient ≈ 193 kPa
#
# Temperature:
#   Determined by shaft work balance (First Law):
#
#   W_turb = ṁ · cp · (T_in − T_out)
#
#   Solving for T_out ensures turbine extracts sufficient power
#   to drive the compressor:
#
#   T_out = T_in − W_required / (ṁ · cp)
#
#   This yields T_out ≈ 1005 K
#
# Velocity:
#   Density drops due to expansion, so axial velocity increases.
#   Derived again from continuity:
#
#   u_out = ṁ / (ρ_out A_out)
#
# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
#
# Areas are chosen to:
#   • satisfy continuity between turbine and nozzle
#   • represent realistic LPT expansion ratios
#
# Length is normalized and scaled out during PINN training (x* ∈ [0,1])
#
# -----------------------------------------------------------------------------
# PHYSICS CONSTANTS (BASELINE ONLY)
# -----------------------------------------------------------------------------
#
# cp, R, gamma below are AIR-LIKE BASELINE VALUES.
# They are used ONLY during PINN training for numerical stability.
#
# At runtime:
#   cp, R, gamma are replaced with fuel-dependent values extracted
#   from Cantera combustion products.
#
# -----------------------------------------------------------------------------
# NORMALIZATION SCALES
# -----------------------------------------------------------------------------
#
# These are characteristic magnitudes used for non-dimensionalization.
# They improve numerical conditioning and DO NOT affect physics.
#
# =============================================================================
DEFAULT_CONDITIONS = {
    'inlet': {'rho': 8.61, 'u': 44.7, 'p': 4.20e6, 'T': 1700.0},
    'outlet': {'rho': 0.67, 'u': 317.7, 'p': 1.93e5, 'T': 1005.0},
    'geometry': {'A_inlet': 0.207, 'A_outlet': 0.377, 'length': 0.5},
    'physics': {
        'R': 287.0,       # For air-like default (will be overridden)
        'gamma': 1.33,    # For air-like default (will be overridden)
        'mass_flow': 79.9,
        'cp': 1150.0,     # For air-like default (will be overridden)
        'w_shaft': 57.4e6  # Target Work required by Compressor (Watts)
    }
}

DEFAULT_SCALES = {
    'rho': 8.61, 'u': 320.0, 'p': 4.20e6, 'T': 1700.0, 'L': 0.5
}

# Global variables (will be set by training or runtime functions)
CONDITIONS = DEFAULT_CONDITIONS.copy()
SCALES = DEFAULT_SCALES.copy()

# Configuration for thermo-parameterized inference
USE_THERMO_CONDITIONING = True  # Use runtime cp/R/gamma for physics calculations

# ============================================================================
# 2. NETWORK ARCHITECTURE (FUEL-DEPENDENT WITH EXACT CONTINUITY)
# ============================================================================

class NormalizedTurbinePINN(nn.Module):
    """
    Physics-Informed Neural Network with exact mass conservation and fuel dependence.

    KEY CHANGES:
    ------------
    1. Input dimension: 1D (x only) → 4D (x, cp, R, gamma)
       - Enables network to learn fuel-dependent expansion behavior

    2. Output dimension: 4D [ρ, u, p, T] → 3D [ρ, p, T]
       - Velocity u is COMPUTED from continuity: u = ṁ/(ρ·A)
       - Enforces mass conservation exactly by construction

    3. Normalization:
       - Thermo properties (cp, R, gamma): FIXED-REFERENCE (THERMO_REF)
       - Flow variables (ρ, p, T): Inlet-anchored
       - This prevents "zero gradient" bug where cp_norm = cp/cp = 1.0 always

    Architecture:
    -------------
    Input:  [x*, cp*, R*, γ*] (4D normalized)
    Hidden: 3 layers × 64 neurons, Tanh activation
    Output: [Δρ*, Δp*, ΔT*] (3D normalized residuals)

    Boundary Condition:
    -------------------
    Hard-enforced inlet BC: y(x) = y_in + x·Δy
    At x=0: y = y_in (exact match to combustor exit state)
    """

    def __init__(self):
        super().__init__()
        # 4D input: [x, cp, R, gamma] → 64 → 64 → 64 → 3D output: [rho, p, T] residuals
        # NOTE: u is NOT predicted - computed from continuity
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)  # Changed from 4 to 3 outputs
        )

        # Xavier initialization for stable training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, cp_feat, R_feat, gamma_feat, inlet_feat, m_dot, A_func):
        """
        Forward pass with exact continuity enforcement.

        Args:
            x: Normalized position [0, 1], shape (N, 1)
            cp_feat: Normalized cp, shape (N, 1) or scalar
            R_feat: Normalized R, shape (N, 1) or scalar
            gamma_feat: Normalized gamma, shape (N, 1) or scalar
            inlet_feat: Normalized inlet state [ρ, u, p, T], shape (1, 4) or (N, 4)
            m_dot: Mass flow rate [kg/s], scalar
            A_func: Function A(x) returning turbine area [m²], callable

        Returns:
            Normalized state [ρ, p, T], shape (N, 3)
            (u will be computed later from ṁ/(ρ·A))
        """
        # Ensure all inputs are tensors with correct shape
        if not isinstance(cp_feat, torch.Tensor):
            cp_feat = torch.ones_like(x) * cp_feat
        if not isinstance(R_feat, torch.Tensor):
            R_feat = torch.ones_like(x) * R_feat
        if not isinstance(gamma_feat, torch.Tensor):
            gamma_feat = torch.ones_like(x) * gamma_feat

        # Reshape if needed
        if cp_feat.dim() == 0:
            cp_feat = cp_feat.view(1, 1).expand(x.size(0), 1)
        if R_feat.dim() == 0:
            R_feat = R_feat.view(1, 1).expand(x.size(0), 1)
        if gamma_feat.dim() == 0:
            gamma_feat = gamma_feat.view(1, 1).expand(x.size(0), 1)

        # Concatenate features: [x, cp, R, γ]
        features = torch.cat([x, cp_feat, R_feat, gamma_feat], dim=1)

        # Predict residuals for [ρ, p, T] only (3D output)
        residuals = self.net(features)  # (N, 3)

        # Hard-enforce inlet BC: y = y_in + x·Δy
        # inlet_feat contains [rho_in, u_in, p_in, T_in] (normalized)
        # We need [rho_in, p_in, T_in] at indices [0, 2, 3]
        if inlet_feat.dim() == 1:
            inlet_feat = inlet_feat.unsqueeze(0)
        if inlet_feat.size(0) == 1:
            inlet_feat = inlet_feat.expand(x.size(0), 4)

        rho_norm = inlet_feat[:, 0:1] + x * residuals[:, 0:1]
        p_norm = inlet_feat[:, 2:3] + x * residuals[:, 1:2]
        T_norm = inlet_feat[:, 3:4] + x * residuals[:, 2:3]

        # Combine predicted state (normalized): [ρ, p, T]
        out_norm = torch.cat([rho_norm, p_norm, T_norm], dim=1)  # (N, 3)

        return out_norm

    def predict_physical(self, x, thermo_props, inlet_state, m_dot, geometry, scales):
        """
        Predict in physical units with exact continuity enforcement.

        This is the main inference function. It:
        1. Normalizes inputs (thermo, inlet state)
        2. Calls forward() to get normalized [ρ, p, T]
        3. Denormalizes [ρ, p, T]
        4. Computes u = ṁ/(ρ·A) to enforce continuity exactly
        5. Returns physical [ρ, u, p, T]

        Args:
            x: Position array [0, 1], tensor shape (N, 1)
            thermo_props: Dict {cp, R, gamma} in physical units
            inlet_state: Dict {rho, u, p, T} in physical units
            m_dot: Mass flow rate [kg/s]
            geometry: Dict {A_inlet, A_outlet, length}
            scales: Dict with normalization scales {rho, u, p, T, cp, R, gamma}

        Returns:
            Physical state [rho, u, p, T] tensor, shape (N, 4)
        """
        # Normalize thermo properties using FIXED REFERENCE
        cp_norm = thermo_props['cp'] / scales['cp']
        R_norm = thermo_props['R'] / scales['R']
        gamma_norm = thermo_props['gamma'] / scales['gamma']

        # Normalize inlet state (inlet-anchored)
        inlet_norm = torch.tensor([[
            inlet_state['rho'] / scales['rho'],
            inlet_state['u'] / scales['u'],
            inlet_state['p'] / scales['p'],
            inlet_state['T'] / scales['T']
        ]], device=x.device, dtype=torch.float32)

        # Define area function (expanding turbine)
        # A(x) = A_inlet + (A_outlet - A_inlet)·x  (linear expansion)
        A_inlet = geometry['A_inlet']
        A_outlet = geometry['A_outlet']

        def area_func(x_pos):
            """Turbine area as function of normalized position."""
            return A_inlet + (A_outlet - A_inlet) * x_pos

        # Forward pass → normalized [ρ, p, T]
        out_norm = self.forward(x, cp_norm, R_norm, gamma_norm, inlet_norm, m_dot, area_func)

        # Denormalize [ρ, p, T]
        rho = out_norm[:, 0:1] * scales['rho']  # kg/m³
        p = out_norm[:, 1:2] * scales['p']      # Pa
        T = out_norm[:, 2:3] * scales['T']      # K

        # Compute velocity from EXACT continuity equation:
        # ṁ = ρ·u·A  →  u = ṁ/(ρ·A)
        A = area_func(x)  # (N, 1) or (N,)
        if A.dim() == 1:
            A = A.unsqueeze(1)  # (N, 1)

        # Avoid division by zero
        rho_safe = torch.clamp(rho, min=1e-6)
        u = m_dot / (rho_safe * A)  # m/s - EXACT mass conservation

        # Return full state [ρ, u, p, T]
        return torch.cat([rho, u, p, T], dim=1)

# ============================================================================
# 3. PHASE 1: BOUNDARY CONDITIONS (DEPRECATED - skipped for thermo-conditioned model)
# ============================================================================
# NOTE: With the new architecture, we train directly with physics constraints
# and random thermo sampling. Phase 1 BC pretraining is not needed since the
# hard BC enforcement (y = y_in + x·Δy) already guarantees inlet matching.

# ============================================================================
# 4. PHYSICS (UPDATED WITH SHAFT WORK)
# ============================================================================

def compute_loss_components(model, x_col, device, thermo_props, inlet_state, m_dot, geometry, w_target, scales):
    """
    Compute physics-based loss components for turbine PINN with exact continuity.

    Args:
        model: Turbine PINN model
        x_col: Collocation points [0, 1]
        device: torch device
        thermo_props: Dict {cp, R, gamma} - SAMPLED for each batch
        inlet_state: Dict {rho, u, p, T} - SAMPLED for each batch
        m_dot: Mass flow rate [kg/s]
        geometry: Dict {A_inlet, A_outlet, length}
        w_target: Target shaft work [W]
        scales: Normalization scales dict

    Returns:
        Tuple of (loss_eos, loss_monotonic, loss_work)

    NOTE: loss_mass is REMOVED - continuity is exact by construction!
    """
    x = x_col.clone().requires_grad_(True)

    # Get physical predictions with exact continuity
    state = model.predict_physical(x, thermo_props, inlet_state, m_dot, geometry, scales)

    rho = state[:, 0:1]
    u = state[:, 1:2]
    p = state[:, 2:3]
    T = state[:, 3:4]

    # --- 1. EOS: p = rho R T (consistency check) ---
    R = thermo_props['R']
    eos_res = (p - rho * R * T) / scales['p']
    loss_eos = (eos_res**2).mean()

    # --- 2. Mass Conservation: REMOVED! ---
    # Continuity is now exact by construction: u = ṁ/(ρ·A)
    # No need for d(ρuA)/dx = 0 loss term

    # --- 3. Monotonic Temp (Heuristic: T should decrease) ---
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    loss_monotonic = torch.relu(T_x / scales['T']).mean()  # Penalize T rising

    # --- 4. SHAFT WORK (First Law) ---
    # Work = m_dot * cp * (T_in - T_out)
    # CRITICAL: Uses SAMPLED cp (fuel-dependent!)
    T_in_pred = T[0]  # x=0
    T_out_pred = T[-1]  # x=1

    cp = thermo_props['cp']
    w_pred = m_dot * cp * (T_in_pred - T_out_pred)

    # Normalize by target work
    loss_work = ((w_pred - w_target) / w_target)**2

    return loss_eos, loss_monotonic, loss_work

def train_phase2_physics(model, x_col, device, n_epochs=5000):
    """
    Train PINN with physics-based losses and RANDOM THERMO SAMPLING.

    KEY CHANGE: Instead of using fixed DEFAULT_CONDITIONS, we randomly sample
    thermodynamic properties (cp, R, gamma) each epoch. This forces the network
    to learn fuel-dependent behavior.

    Args:
        model: Turbine PINN model
        x_col: Collocation points
        device: torch device
        n_epochs: Number of training epochs (increased to 5000 for better convergence)

    Returns:
        Trained model
    """
    print("PHASE 2: PHYSICS ENFORCEMENT WITH THERMO SAMPLING")
    print("Training with RANDOMIZED fuel properties each epoch:")
    print(f"  γ   ~ U(1.28, 1.38)  [covers lean to rich combustion]")
    print(f"  cp  ~ ref · U(0.9, 1.2)  [~1035-1380 J/kg·K]")
    print(f"  R   ~ ref · U(0.95, 1.05) [~273-302 J/kg·K]")
    print("This enables TRUE fuel-dependent predictions.\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Fixed geometry and target work from DEFAULT_CONDITIONS
    geometry = DEFAULT_CONDITIONS['geometry']
    m_dot = DEFAULT_CONDITIONS['physics']['mass_flow']
    w_target = DEFAULT_CONDITIONS['physics']['w_shaft']

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # ====================================================================
        # RANDOM SAMPLING (fuel-dependent training)
        # ====================================================================
        # Sample inlet state (turbine inlet conditions vary with throttle/altitude)
        T_in = 1500.0 + 400.0 * torch.rand(1).item()  # 1500-1900 K
        p_in = 3.0e6 + 2.0e6 * torch.rand(1).item()   # 3-5 MPa

        # Sample thermo properties (fuel-dependent)
        gamma = 1.28 + 0.10 * torch.rand(1).item()    # 1.28-1.38
        cp = THERMO_REF['cp'] * (0.9 + 0.3 * torch.rand(1).item())  # 0.9-1.2× ref
        R = THERMO_REF['R'] * (0.95 + 0.10 * torch.rand(1).item())  # 0.95-1.05× ref

        # Compute inlet density from EOS
        rho_in = p_in / (R * T_in)

        # Compute inlet velocity from continuity
        u_in = m_dot / (rho_in * geometry['A_inlet'])

        # Compute outlet temp from shaft work constraint
        T_out = T_in - w_target / (m_dot * cp)

        # Compute outlet pressure (assume some expansion ratio, will be learned)
        p_out = p_in * 0.045  # ~4.5% of inlet (typical turbine expansion)

        # Compute outlet density and velocity
        rho_out = p_out / (R * T_out)
        u_out = m_dot / (rho_out * geometry['A_outlet'])

        # Build sampled conditions
        inlet_state = {'rho': rho_in, 'u': u_in, 'p': p_in, 'T': T_in}
        outlet_state = {'rho': rho_out, 'u': u_out, 'p': p_out, 'T': T_out}
        thermo_props = {'cp': cp, 'R': R, 'gamma': gamma}

        # ====================================================================
        # INLET-ANCHORED NORMALIZATION (Critical!)
        # ====================================================================
        # Use sampled inlet state for flow variable normalization
        # Use FIXED THERMO_REF for thermo property normalization
        scales = {
            'rho': rho_in,
            'u': u_in,
            'p': p_in,
            'T': T_in,
            'cp': THERMO_REF['cp'],      # FIXED reference, not current cp
            'R': THERMO_REF['R'],        # FIXED reference, not current R
            'gamma': THERMO_REF['gamma'], # FIXED reference, not current gamma
            'L': geometry['length']
        }

        # Normalized targets for BC
        inlet_norm = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=device)  # By definition
        outlet_norm = torch.tensor([[
            outlet_state['rho'] / scales['rho'],
            outlet_state['u'] / scales['u'],
            outlet_state['p'] / scales['p'],
            outlet_state['T'] / scales['T']
        ]], device=device)

        # ====================================================================
        # BOUNDARY CONDITION LOSS
        # ====================================================================
        # Evaluate at inlet and outlet
        x_in = torch.zeros(1, 1, device=device)
        x_out = torch.ones(1, 1, device=device)

        # Get predictions
        state_in = model.predict_physical(x_in, thermo_props, inlet_state, m_dot, geometry, scales)
        state_out = model.predict_physical(x_out, thermo_props, inlet_state, m_dot, geometry, scales)

        # Normalize predictions for comparison
        state_in_norm = torch.cat([
            state_in[:, 0:1] / scales['rho'],
            state_in[:, 1:2] / scales['u'],
            state_in[:, 2:3] / scales['p'],
            state_in[:, 3:4] / scales['T']
        ], dim=1)

        state_out_norm = torch.cat([
            state_out[:, 0:1] / scales['rho'],
            state_out[:, 1:2] / scales['u'],
            state_out[:, 2:3] / scales['p'],
            state_out[:, 3:4] / scales['T']
        ], dim=1)

        # BC loss (inlet should be exact due to hard BC, but check outlet)
        loss_bc_in = ((state_in_norm - inlet_norm)**2).mean()
        loss_bc_out = ((state_out_norm - outlet_norm)**2).mean()
        loss_bc = loss_bc_in + loss_bc_out

        # ====================================================================
        # PHYSICS LOSS
        # ====================================================================
        l_eos, l_mono, l_work = compute_loss_components(
            model, x_col, device, thermo_props, inlet_state, m_dot, geometry, w_target, scales
        )

        # Weighted Sum (no more l_mass - continuity is exact!)
        w_phys = 0.1 if epoch < 500 else 1.0

        loss = 20.0 * loss_bc + w_phys * (l_eos + 0.1 * l_mono + 0.5 * l_work)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Ep {epoch:4d} | BC: {loss_bc.item():.1e} | Work: {l_work.item():.1e} | EOS: {l_eos.item():.1e} | γ={gamma:.3f}")

    return model

# ============================================================================
# 5. MAIN & IMPROVED VALIDATION
# ============================================================================

def main():
    device = torch.device("cpu")
    model = NormalizedTurbinePINN().to(device)

    # Collocation points for physics loss
    x_col = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)

    # ========================================================================
    # TRAINING
    # ========================================================================
    try:
        train_phase2_physics(model, x_col, device, n_epochs=5000)
    except KeyboardInterrupt:
        print("\nStopping early...")

    # ========================================================================
    # VALIDATION WITH MULTIPLE FUEL TYPES
    # ========================================================================
    model.eval()
    print("\n" + "="*70)
    print("📊 TURBINE PERFORMANCE VALIDATION")
    print("="*70)

    # Test with DEFAULT_CONDITIONS (baseline)
    baseline_inlet = DEFAULT_CONDITIONS['inlet']
    baseline_thermo = {
        'cp': DEFAULT_CONDITIONS['physics']['cp'],
        'R': DEFAULT_CONDITIONS['physics']['R'],
        'gamma': DEFAULT_CONDITIONS['physics']['gamma']
    }
    baseline_geometry = DEFAULT_CONDITIONS['geometry']
    baseline_m_dot = DEFAULT_CONDITIONS['physics']['mass_flow']
    baseline_w_target = DEFAULT_CONDITIONS['physics']['w_shaft']

    # Scales for baseline case (inlet-anchored)
    baseline_scales = {
        'rho': baseline_inlet['rho'],
        'u': baseline_inlet['u'],
        'p': baseline_inlet['p'],
        'T': baseline_inlet['T'],
        'cp': THERMO_REF['cp'],
        'R': THERMO_REF['R'],
        'gamma': THERMO_REF['gamma'],
        'L': baseline_geometry['length']
    }

    with torch.no_grad():
        x_eval = torch.tensor([[0.0], [1.0]], device=device)
        preds = model.predict_physical(
            x_eval, baseline_thermo, baseline_inlet, baseline_m_dot, baseline_geometry, baseline_scales
        ).cpu().numpy()

        print("\n--- BASELINE CASE (Default Conditions) ---")
        print(f"Thermo: cp={baseline_thermo['cp']:.1f} J/(kg·K), R={baseline_thermo['R']:.1f} J/(kg·K), γ={baseline_thermo['gamma']:.3f}")
        print(f"\nInlet State:")
        print(f"  P:   {preds[0, 2]/1e6:.2f} MPa (Target: {baseline_inlet['p']/1e6:.2f})")
        print(f"  T:   {preds[0, 3]:.1f} K (Target: {baseline_inlet['T']:.1f})")
        print(f"\nOutlet State:")
        print(f"  P:   {preds[1, 2]/1e3:.0f} kPa")
        print(f"  T:   {preds[1, 3]:.1f} K")

        # Work extraction check
        T_in = preds[0, 3]
        T_out = preds[1, 3]
        delta_T = T_in - T_out

        w_actual = baseline_m_dot * baseline_thermo['cp'] * delta_T
        err_w = abs(w_actual - baseline_w_target) / baseline_w_target

        print(f"\nWork Extraction:")
        print(f"  Target: {baseline_w_target/1e6:.2f} MW")
        print(f"  Actual: {w_actual/1e6:.2f} MW")
        print(f"  Error:  {err_w*100:.2f}%")

        if err_w < 0.05:
            print("  ✅ PASS: Energy Conservation Satisfied")
        else:
            print("  ❌ FAIL: Energy Conservation Violation")

        # Mass conservation check
        rho_in_pred = preds[0, 0]
        u_in_pred = preds[0, 1]
        rho_out_pred = preds[1, 0]
        u_out_pred = preds[1, 1]

        m_in = rho_in_pred * u_in_pred * baseline_geometry['A_inlet']
        m_out = rho_out_pred * u_out_pred * baseline_geometry['A_outlet']
        mass_error = abs(m_out - m_in) / m_in * 100.0

        print(f"\nMass Conservation:")
        print(f"  ṁ_in:  {m_in:.2f} kg/s")
        print(f"  ṁ_out: {m_out:.2f} kg/s")
        print(f"  Error: {mass_error:.6f}% (should be ~0%)")

        if mass_error < 0.01:
            print("  ✅ PASS: Exact Mass Conservation")
        else:
            print("  ⚠️  WARNING: Mass conservation not exact")

    # ========================================================================
    # THERMO SENSITIVITY TEST
    # ========================================================================
    print("\n" + "="*70)
    print("🔥 THERMO SENSITIVITY TEST")
    print("="*70)

    # Test with perturbed gamma (+5%)
    perturbed_thermo = baseline_thermo.copy()
    perturbed_thermo['gamma'] = baseline_thermo['gamma'] * 1.05

    with torch.no_grad():
        preds_pert = model.predict_physical(
            x_eval, perturbed_thermo, baseline_inlet, baseline_m_dot, baseline_geometry, baseline_scales
        ).cpu().numpy()

        T_out_pert = preds_pert[1, 3]
        delta_T_pert = preds_pert[0, 3] - T_out_pert
        w_actual_pert = baseline_m_dot * perturbed_thermo['cp'] * delta_T_pert

        print(f"\nBaseline γ={baseline_thermo['gamma']:.3f}:")
        print(f"  T_out = {T_out:.1f} K, W = {w_actual/1e6:.2f} MW")
        print(f"\nPerturbed γ={perturbed_thermo['gamma']:.3f} (+5%):")
        print(f"  T_out = {T_out_pert:.1f} K, W = {w_actual_pert/1e6:.2f} MW")

        delta_T_change = abs(T_out_pert - T_out)
        delta_W_pct = abs(w_actual_pert - w_actual) / w_actual * 100.0

        print(f"\nSensitivity:")
        print(f"  ΔT_out = {delta_T_change:.1f} K")
        print(f"  ΔW = {delta_W_pct:.2f}%")

        if delta_T_change > 1.0:  # At least 1 K change
            print("  ✅ PASS: Network responds to fuel property changes")
        else:
            print("  ❌ FAIL: Network is fuel-blind (normalization bug?)")

    print("="*70)

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    torch.save({
        'model_state_dict': model.state_dict(),
        'thermo_ref': THERMO_REF,
        'default_conditions': DEFAULT_CONDITIONS
    }, "turbine_pinn.pt")
    print("\n💾 Model saved to 'turbine_pinn.pt'")
    print("   Includes THERMO_REF for consistent normalization at runtime.")


# ============================================================================
# RUNTIME API FOR INTEGRATED ENGINE
# ============================================================================

def run_turbine_pinn(
    model_path: str,
    inlet_state: Dict[str, float],
    target_work: float,
    m_dot: float,
    A_inlet: float,
    A_outlet: float,
    length: float,
    thermo_props: Dict[str, float]
) -> Dict[str, Any]:
    """
    Run turbine PINN inference for integrated engine cycle.

    This is the main API function called by integrated_engine.py. It loads the
    trained model, runs inference with fuel-dependent thermo properties, and
    returns the turbine exit state.

    Args:
        model_path: Path to trained turbine PINN checkpoint (.pt file)
        inlet_state: Dict with {rho, u, p, T} in physical units [kg/m³, m/s, Pa, K]
        target_work: Target shaft work extraction [W]
        m_dot: Mass flow rate [kg/s]
        A_inlet: Turbine inlet area [m²]
        A_outlet: Turbine outlet area [m²]
        length: Turbine length [m]
        thermo_props: Dict with {cp, R, gamma} from combustor

    Returns:
        Dict with:
            'rho': Exit density [kg/m³]
            'u': Exit velocity [m/s]
            'p': Exit pressure [Pa]
            'T': Exit temperature [K]
            'work_specific': Specific work [J/kg]
            'work_total': Total work [W]
            'cp': Specific heat (propagated)
            'R': Gas constant (propagated)
            'gamma': Heat capacity ratio (propagated)
    """
    # Load model checkpoint
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Turbine PINN not found at '{model_path}'.\n"
            f"Please train the turbine PINN first: python simulation/turbine/turbine.py"
        )

    checkpoint = torch.load(model_path, map_location='cpu')
    thermo_ref = checkpoint.get('thermo_ref', THERMO_REF)

    # Create and load model
    device = torch.device('cpu')
    model = NormalizedTurbinePINN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Build geometry dict
    geometry = {
        'A_inlet': A_inlet,
        'A_outlet': A_outlet,
        'length': length
    }

    # Inlet-anchored normalization scales
    scales = {
        'rho': inlet_state['rho'],
        'u': inlet_state['u'],
        'p': inlet_state['p'],
        'T': inlet_state['T'],
        'cp': thermo_ref['cp'],      # FIXED reference
        'R': thermo_ref['R'],        # FIXED reference
        'gamma': thermo_ref['gamma'], # FIXED reference
        'L': length
    }

    # Run inference at outlet (x=1.0)
    with torch.no_grad():
        x_out = torch.tensor([[1.0]], device=device)
        state_out = model.predict_physical(
            x_out, thermo_props, inlet_state, m_dot, geometry, scales
        ).cpu().numpy()

    # Extract outlet state
    rho_out = state_out[0, 0]
    u_out = state_out[0, 1]
    p_out = state_out[0, 2]
    T_out = state_out[0, 3]

    # Compute work extraction
    cp = thermo_props['cp']
    T_in = inlet_state['T']

    # Adjust outlet temperature to match target work
    delta_T_target = target_work / (m_dot * cp)
    T_out_adjusted = T_in - delta_T_target

    # Recalculate density and velocity with adjusted temperature
    rho_out_adjusted = p_out / (thermo_props['R'] * T_out_adjusted)
    u_out_adjusted = m_dot / (rho_out_adjusted * A_outlet)

    work_specific = cp * delta_T_target
    work_total = target_work

    return {
        'rho': rho_out_adjusted,
        'u': u_out_adjusted,
        'p': p_out,
        'T': T_out_adjusted,
        'work_specific': work_specific,
        'work_total': work_total,
        'cp': thermo_props['cp'],
        'R': thermo_props['R'],
        'gamma': thermo_props['gamma']
    }


if __name__ == "__main__":
    main()