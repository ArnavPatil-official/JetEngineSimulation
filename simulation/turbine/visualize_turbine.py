import torch
import matplotlib.pyplot as plt
from turbine import NormalizedTurbinePINN, SCALES 

def plot_results(model, device):
    """Visualizes the internal profiles of the turbine."""
    model.eval()
    
    # 1. Generate spatial points (0 to 1)
    x_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    
    # 2. Predict (and un-normalize)
    with torch.no_grad():
        preds = model.predict_physical(x_test).cpu().numpy()
        x_plot = x_test.cpu().numpy()
        
    rho = preds[:, 0]
    u   = preds[:, 1]
    p   = preds[:, 2] / 1e6  # Convert Pa to MPa
    T   = preds[:, 3]
    
    # 3. Create Plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Turbine Expansion Profiles (PINN)', fontsize=16)
    
    # Pressure
    axs[0, 0].plot(x_plot, p, 'b-', linewidth=2)
    axs[0, 0].set_title('Pressure (MPa)')
    axs[0, 0].set_ylabel('MPa')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Temperature
    axs[0, 1].plot(x_plot, T, 'r-', linewidth=2)
    axs[0, 1].set_title('Temperature (K)')
    axs[0, 1].set_ylabel('Kelvin')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Velocity
    axs[1, 0].plot(x_plot, u, 'g-', linewidth=2)
    axs[1, 0].set_title('Velocity (m/s)')
    axs[1, 0].set_ylabel('m/s')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Density
    axs[1, 1].plot(x_plot, rho, 'k-', linewidth=2)
    axs[1, 1].set_title('Density (kg/m³)')
    axs[1, 1].set_ylabel('kg/m³')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# CORRECTED LOADING SEQUENCE
# ==========================================
# 1. Initialize the empty model structure
device = torch.device("cpu")
model = NormalizedTurbinePINN().to(device)

# 2. Load the file into a temporary variable
path = "/Users/arnavpatil/Desktop/JetEngineSimulation/turbine_pinn.pt"
checkpoint = torch.load(path, map_location=device)

# 3. Extract ONLY the weights and load them into the model
model.load_state_dict(checkpoint['model_state_dict'])

print("✅ Model weights loaded successfully.")

# 4. Plot
plot_results(model, device)