import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. SETUP ---
sys.path.append(os.getcwd())
DEVICE = torch.device('cpu') 

# Import models
try:
    from simulation.nozzle.nozzle import NozzlePINN
    from simulation.turbine.turbine import NormalizedTurbinePINN
    print("✅ Imported models from simulation package.")
except ImportError:
    from nozzle import NozzlePINN
    from turbine import NormalizedTurbinePINN
    print("✅ Imported models from root folder.")

# --- 2. CONSTANTS ---
THERMO_REF = {'cp': 1150.0, 'R': 287.0, 'gamma': 1.33}

# --- 3. THE PLOTTER ---
def visualize(model_class, pt_file, title, filename, inlet_p, inlet_T, A_in, m_dot):
    print(f"\n🎨 Visualizing {title}...")
    
    # 1. Initialize & Load
    model = model_class()
    checkpoint = torch.load(pt_file, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Physics Prep
    # Calculate Density (Ideal Gas)
    rho_val = inlet_p / (THERMO_REF['R'] * inlet_T)
    u_val = m_dot / (rho_val * A_in)
    
    # Inputs
    inlet_state = {'p': inlet_p, 'T': inlet_T, 'rho': rho_val}
    geometry = {'A_in': A_in}

    # --- THE FIX: ADD ALL POSSIBLE KEY NAMES ---
    # The model might look for 'rho' OR 'rho_in', so we provide BOTH.
    scales = {
        'L': 1.0,
        # Standard keys
        'p_in': inlet_p, 'T_in': inlet_T, 'rho_in': rho_val, 'u_in': u_val,
        # "Just in case" keys (The one causing your error)
        'p': inlet_p, 'T': inlet_T, 'rho': rho_val, 'u': u_val,
        # Thermo keys
        'cp': THERMO_REF['cp'], 'R': THERMO_REF['R'], 'gamma': THERMO_REF['gamma']
    }

    # 3. Predict
    x = torch.linspace(0, 1, 100).view(-1, 1)
    with torch.no_grad():
        out = model.predict_physical(x, THERMO_REF, inlet_state, m_dot, geometry, scales)
    
    # 4. Plot
    data = out.numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} Physics Profile", fontsize=16)

    # Velocity
    ax[0].plot(x, data[:, 1], 'b-', lw=3)
    ax[0].set_title("Velocity (m/s)")
    ax[0].grid(True, alpha=0.3)

    # Pressure
    ax[1].plot(x, data[:, 2]/1e5, 'g-', lw=3)
    ax[1].set_title("Pressure (bar)")
    ax[1].grid(True, alpha=0.3)
    
    # Temperature
    ax[2].plot(x, data[:, 3], 'r-', lw=3)
    ax[2].set_title("Temperature (K)")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Saved {filename}")

# --- 4. RUN ---
if __name__ == "__main__":
    # Turbine
    visualize(NormalizedTurbinePINN, "models/turbine_pinn.pt", "Turbine Expansion", "outputs/plots/viz_turbine.png",
              inlet_p=30e5, inlet_T=1600.0, A_in=0.1, m_dot=50.0)

    # Nozzle
    visualize(NozzlePINN, "models/nozzle_pinn.pt", "Nozzle Acceleration", "outputs/plots/viz_nozzle.png",
              inlet_p=3e5, inlet_T=900.0, A_in=0.2, m_dot=50.0)