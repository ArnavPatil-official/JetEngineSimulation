"""
TURBINE PINN - NORMALIZED CURRICULUM LEARNING
=============================================
Fixed:
1. Syntax errors resolved.
2. Inputs/Outputs normalized to range [0,1] or ~O(1).
3. Loss functions scaled to balance gradients.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# Force float32 globally
torch.set_default_dtype(torch.float32)

print("="*70)
print("TURBINE PINN - NORMALIZED APPROACH")
print("="*70)

# ============================================================================
# 1. SCALES & CONFIGURATION
# ============================================================================
# Hardcoded from your data extraction for reproducibility
CONDITIONS = {
    'inlet': {'rho': 8.61, 'u': 44.7, 'p': 4.20e6, 'T': 1700.0},
    'outlet': {'rho': 0.67, 'u': 317.7, 'p': 1.93e5, 'T': 1005.0},
    'geometry': {'A_inlet': 0.207, 'A_outlet': 0.377, 'length': 0.5},
    'physics': {'R': 287.0, 'gamma': 1.33, 'mass_flow': 79.9, 'cp': 1150.0, 'w_shaft': 57.4e6}
}

# Characteristic Scales (Key to convergence!)
SCALES = {
    'rho': 8.61,    # Inlet density
    'u': 320.0,     # Max velocity (approx outlet)
    'p': 4.20e6,    # Inlet pressure
    'T': 1700.0,    # Inlet temperature
    'L': 0.5        # Length
}

# ============================================================================
# 2. NETWORK ARCHITECTURE (NORMALIZED)
# ============================================================================

class NormalizedTurbinePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 -> 64 -> 64 -> 64 -> 4
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)  # Outputs [rho*, u*, p*, T*] (Normalized)
        )
        
        # Initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1) 

    def forward(self, x):
        return self.net(x)

    def predict_physical(self, x):
        """Helper to get physical units for printing/validation"""
        out_norm = self.forward(x)
        return torch.cat([
            out_norm[:, 0:1] * SCALES['rho'],
            out_norm[:, 1:2] * SCALES['u'],
            out_norm[:, 2:3] * SCALES['p'],
            out_norm[:, 3:4] * SCALES['T']
        ], dim=1)

# ============================================================================
# 3. PHASE 1: BOUNDARY CONDITIONS
# ============================================================================

def train_phase1_boundaries(model, device, n_epochs=1000):
    print("PHASE 1: LEARNING BOUNDARY CONDITIONS (NORMALIZED)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Pre-calculate NORMALIZED targets
    t_in = torch.tensor([[
        CONDITIONS['inlet']['rho'] / SCALES['rho'],
        CONDITIONS['inlet']['u']   / SCALES['u'],
        CONDITIONS['inlet']['p']   / SCALES['p'],
        CONDITIONS['inlet']['T']   / SCALES['T']
    ]], device=device)
    
    t_out = torch.tensor([[
        CONDITIONS['outlet']['rho'] / SCALES['rho'],
        CONDITIONS['outlet']['u']   / SCALES['u'],
        CONDITIONS['outlet']['p']   / SCALES['p'],
        CONDITIONS['outlet']['T']   / SCALES['T']
    ]], device=device)
    
    # Input points (Normalized geometry 0 and 1)
    x_in = torch.zeros(1, 1, device=device)
    x_out = torch.ones(1, 1, device=device)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        p_in = model(x_in)
        p_out = model(x_out)
        
        # MSE on normalized values
        loss = ((p_in - t_in)**2).mean() + ((p_out - t_out)**2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | BC Loss: {loss.item():.6e}")
            
    print(f"✓ Phase 1 Complete. Final BC Loss: {loss.item():.6e}\n")

# ============================================================================
# 4. PHYSICS & PHASE 2/3
# ============================================================================

def compute_loss_components(model, x_col, device):
    x = x_col.clone().requires_grad_(True)
    
    # 1. Forward Pass (Normalized)
    out_norm = model(x)
    rho_s = out_norm[:, 0:1]
    u_s   = out_norm[:, 1:2]
    p_s   = out_norm[:, 2:3]
    T_s   = out_norm[:, 3:4]
    
    # 2. Re-dimensionalize for Physics
    # It is safer to calculate residuals in physical units, then normalize the residual
    rho = rho_s * SCALES['rho']
    u   = u_s   * SCALES['u']
    p   = p_s   * SCALES['p']
    T   = T_s   * SCALES['T']
    
    # Geometry
    A = CONDITIONS['geometry']['A_inlet'] + \
        (CONDITIONS['geometry']['A_outlet'] - CONDITIONS['geometry']['A_inlet']) * x
    
    # --- EOS: p = rho R T ---
    R = CONDITIONS['physics']['R']
    # Residual = (p - rhoRT) / P_scale
    eos_res = (p - rho * R * T) / SCALES['p']
    loss_eos = (eos_res**2).mean()
    
    # --- Mass: d(rho u A)/dx = 0 ---
    mass_flow = rho * u * A
    mass_flow_x = torch.autograd.grad(mass_flow, x, torch.ones_like(mass_flow), create_graph=True)[0]
    # Normalize by characteristic mass flow
    m_scale = SCALES['rho'] * SCALES['u'] * CONDITIONS['geometry']['A_inlet']
    loss_mass = ((mass_flow_x / m_scale)**2).mean()  # <--- FIXED SYNTAX HERE
    
    # --- Energy (Optional/Phase 3): Temperature decreases ---
    # d(T)/dx < 0. We penalize positive gradients.
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    # Normalize by T_scale
    loss_energy = torch.relu(T_x / SCALES['T']).mean()
    
    return loss_eos, loss_mass, loss_energy

def train_phase2_physics(model, x_col, device, n_epochs=2000):
    print("PHASE 2 & 3: PHYSICS ENFORCEMENT")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Boundary Targets (same as Phase 1)
    t_in = torch.tensor([[CONDITIONS['inlet'][k]/SCALES[k] for k in ['rho','u','p','T']]], device=device)
    t_out = torch.tensor([[CONDITIONS['outlet'][k]/SCALES[k] for k in ['rho','u','p','T']]], device=device)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # BC Loss
        p_in = model(torch.zeros(1, 1, device=device))
        p_out = model(torch.ones(1, 1, device=device))
        loss_bc = ((p_in - t_in)**2).mean() + ((p_out - t_out)**2).mean()
        
        # Physics Loss
        l_eos, l_mass, l_energy = compute_loss_components(model, x_col, device)
        
        # Weighted Sum (Weights are easier to tune now that losses are all ~O(1) or O(1e-3))
        # We increase physics weight in later epochs (Curriculum)
        w_phys = 0.1 if epoch < 500 else 1.0
        
        loss = 10.0 * loss_bc + w_phys * (l_eos + l_mass + 0.1 * l_energy)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Ep {epoch} | Total: {loss.item():.4f} | BC: {loss_bc.item():.5f} | EOS: {l_eos.item():.5f} | Mass: {l_mass.item():.5f}")

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    device = torch.device("cpu")
    model = NormalizedTurbinePINN().to(device)
    
    x_col = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    
    # Train
    train_phase1_boundaries(model, device)
    train_phase2_physics(model, x_col, device)
    
    # Validate
    model.eval()
    with torch.no_grad():
        preds = model.predict_physical(torch.tensor([[0.0], [1.0]], device=device)).numpy()
        
        print("\n" + "="*30)
        print("FINAL VALIDATION")
        print("="*30)
        
        print("Inlet Target (P):  4.20 MPa")
        print(f"Inlet Pred (P):    {preds[0, 2]/1e6:.2f} MPa")
        
        print("\nOutlet Target (P): 193 kPa")
        print(f"Outlet Pred (P):   {preds[1, 2]/1e3:.0f} kPa")
        
        p_err = abs(preds[0,2] - 4.2e6) / 4.2e6
        print(f"\nInlet Pressure Error: {p_err*100:.2f}%")
        
        if p_err < 0.05:
            print("SUCCESS: Model converged correctly.")
        else:
            print("WARNING: Convergence issues remain.")

        torch.save({
            'model_state_dict': model.state_dict(),
            'scales': SCALES,
            'conditions': CONDITIONS
        }, "turbine_pinn.pt")
        
        print(f"\nModel and metadata saved to 'turbine_pinn.pt'")

if __name__ == "__main__":
    main()