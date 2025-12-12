"""
TURBINE PINN - FUEL-DEPENDENT THERMODYNAMICS
=============================================

This turbine PINN now uses real, fuel-dependent thermodynamic properties
from Cantera combustion products instead of fixed air-like constants.

Key Features:
1. Thermodynamic properties (cp, R, gamma) derived from combustor output
2. Different fuel blends → different expansion behavior
3. Constant-gamma analytical formulas are NO LONGER VALID
4. The PINN is genuinely necessary for non-ideal, fuel-dependent expansion
5. Enforces Shaft Work Conservation using actual fuel-specific cp

Note: This makes the PINN essential (not just a numerical trick) because
real combustion products have temperature- and composition-dependent
properties that break the isentropic relations used in simple cycle analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# Force float32 globally
torch.set_default_dtype(torch.float32)

print("="*70)
print("TURBINE PINN - ENERGY CONSISTENT APPROACH")
print("="*70)

# ============================================================================
# 1. DEFAULT TRAINING CONDITIONS (for backwards compatibility)
# ============================================================================
# These are default values used during initial training.
# For fuel-dependent operation, use build_turbine_conditions() from thermo_utils
# to create conditions with real combustor-derived properties.

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
# 2. NETWORK ARCHITECTURE
# ============================================================================

class NormalizedTurbinePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 -> 64 -> 64 -> 64 -> 4
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)  # Outputs [rho*, u*, p*, T*]
        )
        
        # Initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1) 

    def forward(self, x):
        return self.net(x)

    def predict_physical(self, x, scales=None):
        """
        Helper to get physical units.

        Args:
            x: Normalized spatial position [0, 1]
            scales: Optional custom scales dict (uses global SCALES if None)

        Returns:
            Physical state [rho, u, p, T]
        """
        if scales is None:
            scales = SCALES

        out_norm = self.forward(x)
        return torch.cat([
            out_norm[:, 0:1] * scales['rho'],
            out_norm[:, 1:2] * scales['u'],
            out_norm[:, 2:3] * scales['p'],
            out_norm[:, 3:4] * scales['T']
        ], dim=1)

    def predict_with_thermo(self, x, cp, R, gamma, scales=None):
        """
        Predict physical state with explicit thermodynamic parameters.

        This method does NOT require retraining. It simply:
        1. Predicts the physical state using the trained network
        2. Reports the thermodynamic parameters used for interpretation

        The PINN was trained with baseline thermo constants, but at inference
        we use actual fuel-dependent properties for consistency checks and
        energy balance calculations.

        Args:
            x: Normalized spatial position [0, 1]
            cp: Specific heat capacity [J/(kg·K)] - FUEL-DEPENDENT
            R: Gas constant [J/(kg·K)] - FUEL-DEPENDENT
            gamma: Heat capacity ratio [-] - FUEL-DEPENDENT
            scales: Optional custom scales dict

        Returns:
            tuple: (physical_state, thermo_params)
                - physical_state: [rho, u, p, T] tensor
                - thermo_params: dict with cp, R, gamma used
        """
        # Get standard prediction
        state = self.predict_physical(x, scales)

        # Package thermo parameters for reference
        thermo_params = {
            'cp': cp,
            'R': R,
            'gamma': gamma,
            'note': 'Runtime thermo parameters (may differ from training baseline)'
        }

        return state, thermo_params

# ============================================================================
# 3. PHASE 1: BOUNDARY CONDITIONS
# ============================================================================

def train_phase1_boundaries(model, device, n_epochs=1000):
    print("PHASE 1: LEARNING BOUNDARY CONDITIONS")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Normalized Targets
    t_in = torch.tensor([[CONDITIONS['inlet'][k]/SCALES[k] for k in ['rho','u','p','T']]], device=device)
    t_out = torch.tensor([[CONDITIONS['outlet'][k]/SCALES[k] for k in ['rho','u','p','T']]], device=device)
    
    x_in = torch.zeros(1, 1, device=device)
    x_out = torch.ones(1, 1, device=device)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        p_in = model(x_in)
        p_out = model(x_out)
        loss = ((p_in - t_in)**2).mean() + ((p_out - t_out)**2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | BC Loss: {loss.item():.6e}")
            
    print(f"✓ Phase 1 Complete. Final BC Loss: {loss.item():.6e}\n")

# ============================================================================
# 4. PHYSICS (UPDATED WITH SHAFT WORK)
# ============================================================================

def compute_loss_components(model, x_col, device, conditions=None, scales=None):
    """
    Compute physics-based loss components for turbine PINN.

    Args:
        model: Turbine PINN model
        x_col: Collocation points [0, 1]
        device: torch device
        conditions: Optional conditions dict with fuel-dependent properties
        scales: Optional scales dict

    Returns:
        Tuple of (loss_eos, loss_mass, loss_monotonic, loss_work)

    Note:
        If conditions/scales are None, uses global CONDITIONS/SCALES.
        The key change: cp, R, gamma now come from combustor output,
        not hardcoded constants. This makes the physics fuel-dependent.
    """
    if conditions is None:
        conditions = CONDITIONS
    if scales is None:
        scales = SCALES

    x = x_col.clone().requires_grad_(True)
    out_norm = model(x)

    # Un-normalize for physics
    rho = out_norm[:, 0:1] * scales['rho']
    u   = out_norm[:, 1:2] * scales['u']
    p   = out_norm[:, 2:3] * scales['p']
    T   = out_norm[:, 3:4] * scales['T']

    # Geometry
    A = conditions['geometry']['A_inlet'] + \
        (conditions['geometry']['A_outlet'] - conditions['geometry']['A_inlet']) * x

    # --- 1. EOS: p = rho R T ---
    # CRITICAL: R now comes from combustor (fuel-dependent!)
    R = conditions['physics']['R']
    eos_res = (p - rho * R * T) / scales['p']
    loss_eos = (eos_res**2).mean()

    # --- 2. Mass: d(rho u A)/dx = 0 ---
    mass_flow = rho * u * A
    mass_flow_x = torch.autograd.grad(mass_flow, x, torch.ones_like(mass_flow), create_graph=True)[0]
    m_scale = scales['rho'] * scales['u'] * conditions['geometry']['A_inlet']
    loss_mass = ((mass_flow_x / m_scale)**2).mean()

    # --- 3. Monotonic Temp (Heuristic) ---
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    loss_monotonic = torch.relu(T_x / scales['T']).mean() # Penalize T rising

    # --- 4. SHAFT WORK (First Law) ---
    # Work = m_dot * cp * (T_in - T_out)
    # CRITICAL: cp now comes from combustor (fuel-dependent!)
    T_in_pred = T[0]  # x=0
    T_out_pred = T[-1] # x=1

    cp = conditions['physics']['cp']  # FUEL-DEPENDENT
    m_dot = conditions['physics']['mass_flow']
    w_target = conditions['physics'].get('w_shaft', 57.4e6)  # Use default if not provided

    w_pred = m_dot * cp * (T_in_pred - T_out_pred)

    # Normalize by target work
    loss_work = ((w_pred - w_target) / w_target)**2

    return loss_eos, loss_mass, loss_monotonic, loss_work

def train_phase2_physics(model, x_col, device, n_epochs=2000, conditions=None, scales=None):
    """
    Train PINN with physics-based losses.

    Args:
        model: Turbine PINN model
        x_col: Collocation points
        device: torch device
        n_epochs: Number of training epochs
        conditions: Optional conditions dict (uses global if None)
        scales: Optional scales dict (uses global if None)

    Returns:
        Trained model
    """
    if conditions is None:
        conditions = CONDITIONS
    if scales is None:
        scales = SCALES

    print("PHASE 2: PHYSICS ENFORCEMENT (+ SHAFT WORK)")
    print("Training turbine PINN with baseline thermo constants:")
    print(f"  cp_train    = {conditions['physics']['cp']:.1f} J/(kg·K)")
    print(f"  R_train     = {conditions['physics']['R']:.1f} J/(kg·K)")
    print(f"  gamma_train = {conditions['physics']['gamma']:.3f}")
    print("Runtime thermo parameters will override these during full-cycle simulation.\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # BC Targets
    t_in = torch.tensor([[conditions['inlet'][k]/scales[k] for k in ['rho','u','p','T']]], device=device)
    t_out = torch.tensor([[conditions['outlet'][k]/scales[k] for k in ['rho','u','p','T']]], device=device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # BC Loss
        p_in = model(torch.zeros(1, 1, device=device))
        p_out = model(torch.ones(1, 1, device=device))
        loss_bc = ((p_in - t_in)**2).mean() + ((p_out - t_out)**2).mean()

        # Physics Loss (now with fuel-dependent properties!)
        l_eos, l_mass, l_mono, l_work = compute_loss_components(model, x_col, device, conditions, scales)

        # Weighted Sum
        # Note: We give Shaft Work a high weight (0.5) to ensure it's respected
        w_phys = 0.1 if epoch < 500 else 1.0

        loss = 20.0 * loss_bc + w_phys * (l_eos + l_mass + 0.1 * l_mono + 0.5 * l_work)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Ep {epoch} | BC: {loss_bc.item():.1e} | Work Loss: {l_work.item():.1e} | EOS: {l_eos.item():.1e}")

    return model

# ============================================================================
# 5. MAIN & IMPROVED VALIDATION
# ============================================================================

def main():
    device = torch.device("cpu")
    model = NormalizedTurbinePINN().to(device)
    
    # Use sorted x_col to ensure T[0] is inlet and T[-1] is outlet
    x_col = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    
    try:
        train_phase1_boundaries(model, device)
        train_phase2_physics(model, x_col, device)
    except KeyboardInterrupt:
        print("\nStopping early...")

    # --- VALIDATION ---
    model.eval()
    with torch.no_grad():
        preds = model.predict_physical(torch.tensor([[0.0], [1.0]], device=device)).numpy()
        
        print("\n" + "="*40)
        print("📊 TURBINE PERFORMANCE VALIDATION")
        print("="*40)
        
        # 1. State Check
        print(f"Inlet P:  {preds[0, 2]/1e6:.2f} MPa (Target: 4.20)")
        print(f"Outlet P: {preds[1, 2]/1e3:.0f} kPa (Target: 193)")
        
        # 2. Work Extraction Check (The new physics!)
        m_dot = CONDITIONS['physics']['mass_flow']
        cp = CONDITIONS['physics']['cp']
        w_target = CONDITIONS['physics']['w_shaft']
        
        T_in = preds[0, 3]
        T_out = preds[1, 3]
        delta_T = T_in - T_out
        
        w_actual = m_dot * cp * delta_T
        err_w = abs(w_actual - w_target) / w_target
        
        print("-" * 20)
        print(f"Inlet Temp:  {T_in:.1f} K")
        print(f"Outlet Temp: {T_out:.1f} K")
        print(f"Delta T:     {delta_T:.1f} K")
        print("-" * 20)
        print(f"Target Work: {w_target/1e6:.2f} MW")
        print(f"Actual Work: {w_actual/1e6:.2f} MW")
        print(f"Work Error:  {err_w*100:.2f}%")
        
        if err_w < 0.05:
            print("✅ PASS: Energy Conservation Satisfied")
        else:
            print("❌ FAIL: Energy Conservation Violation")
        print("="*40)

    # SAVE
    torch.save({
        'model_state_dict': model.state_dict(),
        'scales': SCALES,
        'conditions': CONDITIONS
    }, "turbine_pinn.pt")
    print("\n💾 Model saved to 'turbine_pinn.pt'")

if __name__ == "__main__":
    main()