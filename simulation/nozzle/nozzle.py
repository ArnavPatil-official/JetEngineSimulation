import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Import pandas for data loading
from pathlib import Path

# ============================================================================
# 1. CONFIGURATION & DATA LOADING
# ============================================================================

def load_engine_conditions(filename='icao_engine_data.csv', engine_id='Trent 1000-AE3', mode='TAKE-OFF'):
    """
    Reads ICAO data and returns the CONDITIONS dictionary.
    (Simplified cycle analysis is still hardcoded for temperature/pressures,
     but mass flow is now dynamic.)
    """
    try:
        # Build a robust path to the data file
        script_dir = Path(__file__).resolve().parent
        data_path = script_dir.parent.parent / 'data' / filename
        
        df = pd.read_csv(data_path)
        # Filter for specific engine and mode
        row = df[(df['Engine ID'].str.contains(engine_id, regex=False)) & 
                 (df['Mode'] == mode)].iloc[0]
        
        fuel_flow = float(row['Fuel Flow (kg/s)'])
        pr_overall = float(row['Pressure Ratio'])
        thrust_total = float(row['Rated Thrust (kN)']) * 1000.0
        
        # Simplified Cycle Calculations
        FAR = 0.030 # Fuel-Air Ratio
        core_air = fuel_flow / FAR
        mass_flow = core_air + fuel_flow # Total core mass flow
        
        p_amb = 101325.0
        # p_inlet_nozzle: Approx 4.5% of overall pressure ratio * ambient pressure
        p_inlet_nozzle = p_amb * pr_overall * 0.045
        T_inlet_nozzle = 1005.0 # K (From previous turbine analysis)
        
        # NOTE: We keep rho/u inlet fixed for stability, as they are derived from turbine output
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load data from CSV. Using hardcoded defaults. Error: {e}")
        mass_flow = 79.9
        p_inlet_nozzle = 193000.0
        thrust_total = 310.9e3
    
    # Inlet Area calculated from mass flow: A_in = m_dot / (rho * u)
    # A_in = 79.9 / (0.67 * 317.7) ≈ 0.375 m^2 (Using previous successful values)
    
    return {
        'inlet': {'rho': 0.67, 'u': 317.7, 'p': p_inlet_nozzle, 'T': 1005.0},
        'ambient': {'p': 101325.0}, 
        'geometry': {
            'A_inlet': 0.375, 
            'A_exit': 0.340,   # Corrected area for realistic expansion
            'length': 1.0
        },
        'physics': {
            'R': 287.0, 'gamma': 1.33, 'cp': 1150.0, 
            'mass_flow': mass_flow,
            'target_thrust': thrust_total * 0.15 # 15% of total thrust
        }
    }

CONDITIONS = load_engine_conditions('icao_engine_data.csv', mode='TAKE-OFF')

# Normalization Scales
SCALES = {
    'rho': CONDITIONS['inlet']['rho'], 
    'u': 650.0, 
    'p': CONDITIONS['inlet']['p'], 
    'T': CONDITIONS['inlet']['T'], 
    'L': 1.0
}

# ============================================================================
# 2. NETWORK ARCHITECTURE (Same as before)
# ============================================================================
class NozzlePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.net(x)

    def predict_physical(self, x):
        out_norm = self.forward(x)
        return torch.cat([
            out_norm[:, 0:1] * SCALES['rho'],
            out_norm[:, 1:2] * SCALES['u'],
            out_norm[:, 2:3] * SCALES['p'],
            out_norm[:, 3:4] * SCALES['T']
        ], dim=1)

# ============================================================================
# 3. PHYSICS & LOSS (Same as before)
# ============================================================================
def get_area(x, conditions):
    A_in = conditions['geometry']['A_inlet']
    A_out = conditions['geometry']['A_exit']
    return A_in + (A_out - A_in) * (1 - torch.cos(x * np.pi / 2))

def compute_loss(model, x_col, device):
    x = x_col.clone().requires_grad_(True)
    out_norm = model(x)
    rho = out_norm[:, 0:1] * SCALES['rho']
    u   = out_norm[:, 1:2] * SCALES['u']
    p   = out_norm[:, 2:3] * SCALES['p']
    T   = out_norm[:, 3:4] * SCALES['T']
    
    A = get_area(x, CONDITIONS)
    
    # 1. EOS Loss
    eos_res = (p - rho * CONDITIONS['physics']['R'] * T) / SCALES['p']
    
    # 2. Mass Flow Loss
    m_flow = rho * u * A
    m_flow_x = torch.autograd.grad(m_flow, x, torch.ones_like(m_flow), create_graph=True)[0]
    mass_res = m_flow_x / CONDITIONS['physics']['mass_flow']
    
    # 3. Energy Loss
    cp = CONDITIONS['physics']['cp']
    H0_target = cp * CONDITIONS['inlet']['T'] + 0.5 * CONDITIONS['inlet']['u']**2
    H0_current = cp * T + 0.5 * u**2
    energy_res = (H0_current - H0_target) / H0_target
    
    return (eos_res**2).mean(), (mass_res**2).mean(), (energy_res**2).mean()

# ============================================================================
# 4. TRAINING & SAVE
# ============================================================================
def save_model(model, filename="nozzle_pinn.pt"):
    """Saves the model state, scales, and conditions."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'scales': SCALES,
        'conditions': CONDITIONS
    }, filename)
    print(f"\n💾 Model and metadata successfully saved to '{filename}'")

def train_nozzle():
    device = torch.device("cpu")
    model = NozzlePINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Targets
    t_in = torch.tensor([[CONDITIONS['inlet'][k]/SCALES[k] for k in ['rho','u','p','T']]], device=device)
    p_amb_norm = CONDITIONS['ambient']['p'] / SCALES['p']
    t_p_out = torch.tensor([[p_amb_norm]], device=device)
    
    x_col = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    
    print("🚀 Starting Nozzle Training (Saving Enabled)...")
    
    try:
        for epoch in range(3001):
            optimizer.zero_grad()
            
            # BC Loss: Inlet (Strict)
            pred_in = model(torch.zeros(1, 1, device=device))
            loss_inlet = ((pred_in - t_in)**2).mean()
            
            # BC Loss: Outlet Pressure (Soft)
            pred_out = model(torch.ones(1, 1, device=device))
            p_exit_norm = pred_out[:, 2:3]
            loss_outlet = ((p_exit_norm - t_p_out)**2).mean()
            
            # Physics
            l_eos, l_mass, l_energy = compute_loss(model, x_col, device)
            
            # Weighted Total Loss
            loss = 100.0 * loss_inlet + 1.0 * loss_outlet + 1.0 * (l_eos + l_mass + l_energy)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Ep {epoch} | Inlet: {loss_inlet:.1e} | OutletP: {loss_outlet:.1e} | Phys: {l_eos+l_mass+l_energy:.1e}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
    
    # ALWAYS SAVE THE MODEL HERE
    save_model(model, "nozzle_pinn.pt")
    
    return model, device

# ============================================================================
# 5. VALIDATION
# ============================================================================
def validate_nozzle(model, device):
    model.eval()
    x_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    
    with torch.no_grad():
        preds = model.predict_physical(x_test).cpu().numpy()
        x_plot = x_test.cpu().numpy()
        
    u_exit = preds[-1, 1]
    p_exit = preds[-1, 2]
    T_exit = preds[-1, 3]
    rho_exit = preds[-1, 0]

    # Mass Flow Check (at exit, using PINN prediction)
    m_dot_exit = rho_exit * u_exit * CONDITIONS['geometry']['A_exit']
    
    # Thrust Calculation (using target mass flow for calculation)
    m_dot_target = CONDITIONS['physics']['mass_flow']
    p_amb = CONDITIONS['ambient']['p']
    A_exit = CONDITIONS['geometry']['A_exit']
    
    F_mom = m_dot_target * u_exit
    F_pres = (p_exit - p_amb) * A_exit
    F_total = F_mom + F_pres
    
    print("\n" + "="*40)
    print("📊 NOZZLE PERFORMANCE RESULTS")
    print("="*40)
    print(f"Exit Velocity:    {u_exit:.1f} m/s  (Target: ~580)")
    print(f"Exit Temperature: {T_exit:.1f} K")
    print(f"Exit Pressure:    {p_exit/1000:.1f} kPa (Ambient: {p_amb/1000:.1f} kPa)")
    print(f"Mass Flow Check:  {m_dot_exit:.2f} kg/s (Target: {m_dot_target:.2f})")
    print("-" * 20)
    print(f"Momentum Thrust:  {F_mom/1000:.1f} kN")
    print(f"Pressure Thrust:  {F_pres/1000:.1f} kN")
    print(f"TOTAL THRUST:     {F_total/1000:.2f} kN (Target: {CONDITIONS['physics']['target_thrust']/1000:.1f} kN)")
    print("="*40)
    
    # --- PLOTS --- 
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Nozzle Expansion Profiles (PINN)', fontsize=16)

    # Pressure
    axs[0, 0].plot(x_plot, preds[:, 2]/1e3, 'b-', linewidth=2)
    axs[0, 0].axhline(y=p_amb/1e3, color='r', linestyle='--', label='Ambient')
    axs[0, 0].set_title('Pressure (kPa)')
    axs[0, 0].set_ylabel('kPa')
    axs[0, 0].grid(True, alpha=0.3)

    # Velocity
    axs[0, 1].plot(x_plot, preds[:, 1], 'g-', linewidth=2)
    axs[0, 1].set_title('Velocity (m/s)')
    axs[0, 1].set_ylabel('m/s')
    axs[0, 1].grid(True, alpha=0.3)

    # Mach Number
    gamma = CONDITIONS['physics']['gamma']
    R = CONDITIONS['physics']['R']
    a = np.sqrt(gamma * R * preds[:, 3]) # Speed of sound
    M = preds[:, 1] / a
    axs[1, 0].plot(x_plot, M, 'k-', linewidth=2)
    axs[1, 0].axhline(y=1.0, color='r', linestyle='--', label='Sonic')
    axs[1, 0].set_title('Mach Number')
    axs[1, 0].set_ylabel('M')
    axs[1, 0].grid(True, alpha=0.3)

    # Temperature
    axs[1, 1].plot(x_plot, preds[:, 3], 'r-', linewidth=2)
    axs[1, 1].set_title('Temperature (K)')
    axs[1, 1].set_ylabel('K')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trained_model, device = train_nozzle()
    validate_nozzle(trained_model, device)