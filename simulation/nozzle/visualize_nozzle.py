import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. DEFINE NETWORK ARCHITECTURE
#    (Must match the training script exactly to load weights)
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

    def forward(self, x):
        return self.net(x)

    # Helper to un-normalize using loaded scales
    def predict_physical(self, x, scales):
        out_norm = self.forward(x)
        return torch.cat([
            out_norm[:, 0:1] * scales['rho'],
            out_norm[:, 1:2] * scales['u'],
            out_norm[:, 2:3] * scales['p'],
            out_norm[:, 3:4] * scales['T']
        ], dim=1)

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================
def get_area(x, conditions):
    """Reconstructs the nozzle geometry for plotting."""
    A_in = conditions['geometry']['A_inlet']
    A_out = conditions['geometry']['A_exit']
    return A_in + (A_out - A_in) * (1 - torch.cos(x * np.pi / 2))

def visualize():
    device = torch.device("cpu")
    print("📂 Loading 'nozzle_pinn.pt'...")

    # --- LOAD CHECKPOINT ---
    try:
        checkpoint = torch.load("/Users/arnavpatil/Desktop/JetEngineSimulation/nozzle_pinn.pt", map_location=device)
        
        # 1. Load Metadata
        SCALES = checkpoint['scales']
        CONDITIONS = checkpoint['conditions']
        print(f"   ✓ Metadata loaded (Engine Mode: {CONDITIONS.get('mode', 'Unknown')})")
        
        # 2. Initialize Model & Load Weights
        model = NozzlePINN().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✓ Weights loaded")
        
    except FileNotFoundError:
        print("❌ Error: 'nozzle_pinn.pt' not found. Run the training script first.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- GENERATE PREDICTIONS ---
    model.eval()
    x_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    
    with torch.no_grad():
        preds = model.predict_physical(x_test, SCALES).cpu().numpy()
        # Calculate Area profile for visualization
        area_profile = get_area(x_test, CONDITIONS).cpu().numpy()

    # Extract variables
    x_plot = x_test.cpu().numpy()
    rho = preds[:, 0]
    u   = preds[:, 1]
    p   = preds[:, 2]
    T   = preds[:, 3]

    # Derived Physics
    gamma = CONDITIONS['physics']['gamma']
    R = CONDITIONS['physics']['R']
    a = np.sqrt(gamma * R * T)  # Speed of sound
    M = u / a                   # Mach number
    m_dot_profile = rho * u * area_profile.flatten()

    # --- CALCULATE THRUST METRICS ---
    u_exit = u[-1]
    p_exit = p[-1]
    A_exit = CONDITIONS['geometry']['A_exit']
    p_amb = CONDITIONS['ambient']['p']
    m_dot_target = CONDITIONS['physics']['mass_flow']
    
    F_mom = m_dot_target * u_exit
    F_pres = (p_exit - p_amb) * A_exit
    F_total = F_mom + F_pres

    print("\n" + "="*40)
    print("📊 RECONSTRUCTED RESULTS")
    print("="*40)
    print(f"Exit Velocity:    {u_exit:.1f} m/s")
    print(f"Exit Mach:        {M[-1]:.2f}")
    print(f"Exit Pressure:    {p_exit/1000:.1f} kPa (Ambient: {p_amb/1000:.1f} kPa)")
    print(f"Exit Temperature: {T[-1]:.1f} K")
    print("-" * 20)
    print(f"Calculated Thrust: {F_total/1000:.2f} kN")
    print("="*40)

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Nozzle Physics Visualization (Loaded Model)', fontsize=16)

    # 1. Pressure
    axs[0, 0].plot(x_plot, p/1e3, 'b-', linewidth=2)
    axs[0, 0].axhline(y=p_amb/1e3, color='r', linestyle='--', alpha=0.5, label='Ambient')
    axs[0, 0].fill_between(x_plot.flatten(), p_amb/1e3, p/1e3, where=(p>p_amb), color='blue', alpha=0.1)
    axs[0, 0].set_title('Pressure Distribution')
    axs[0, 0].set_ylabel('Pressure (kPa)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Velocity & Mach
    ax2 = axs[0, 1]
    ax2.plot(x_plot, u, 'g-', linewidth=2, label='Velocity')
    ax2.set_ylabel('Velocity (m/s)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_title('Velocity & Mach Number')
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(x_plot, M, 'k--', linewidth=1.5, label='Mach')
    ax2b.set_ylabel('Mach Number', color='k')
    ax2b.axhline(1.0, color='red', linestyle=':', alpha=0.5)

    # 3. Temperature
    axs[1, 0].plot(x_plot, T, 'r-', linewidth=2)
    axs[1, 0].set_title('Static Temperature')
    axs[1, 0].set_ylabel('Temperature (K)')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Mass Flow Conservation Check
    axs[1, 1].plot(x_plot, m_dot_profile, 'purple', linewidth=2)
    axs[1, 1].axhline(y=m_dot_target, color='k', linestyle='--', label='Target')
    axs[1, 1].set_title('Mass Flow Conservation')
    axs[1, 1].set_ylabel('Mass Flow (kg/s)')
    axs[1, 1].set_ylim(m_dot_target*0.9, m_dot_target*1.1) # Zoom in to show stability
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("📈 Plot generated.")

if __name__ == "__main__":
    visualize()