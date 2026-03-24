import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on path so imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.le_pinn import LE_PINN, MinMaxNormalizer
from scripts.visualization.nozzle_2d_geometry import generate_nozzle_profile

# Thermodynamic constants matching your engine setup
GAMMA = 1.33
R_GAS = 287.0
DEVICE = torch.device("cpu")

def run_zero_shot_mach_test(ckpt_path="models/le_pinn.pt"):
    print(f"Loading actual LE-PINN weights from: {ckpt_path}...")
    
    # 1. Load Model and Checkpoint
    model = LE_PINN().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Handle state dict and normalizers
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Reconstruct normalizers from the checkpoint
    input_norm = MinMaxNormalizer()
    input_norm.data_min = ckpt["input_norm_min"].to(DEVICE)
    input_norm.data_max = ckpt["input_norm_max"].to(DEVICE)
    
    output_norm = MinMaxNormalizer()
    output_norm.data_min = ckpt["output_norm_min"].to(DEVICE)
    output_norm.data_max = ckpt["output_norm_max"].to(DEVICE)
    
    # 2. Get Geometry Bounds
    # Using typical values from your visualization script (NPR=8.0)
    profile = generate_nozzle_profile(NPR=8.0, AR=1.60, Throat_Radius=0.05)
    x_min, x_max = float(np.min(profile.x)), float(np.max(profile.x))
    
    # Create a dense 2D mesh grid for the evaluation
    n_x, n_y = 200, 100
    x_lin = np.linspace(x_min, x_max, n_x)
    
    # Arrays to store the coordinates and predicted Mach number
    X_mesh = np.zeros((n_y, n_x))
    Y_mesh = np.zeros((n_y, n_x))
    Mach_mesh = np.zeros((n_y, n_x))
    
    print("Running 2D forward pass to evaluate learned physics...")
    
    # Constant inputs for the test state (A5, A6, P_in, T_in)
    # Using defaults from your architecture/visualization
    A5, A6 = 0.20, 0.32
    P_in, T_in = 300000.0, 900.0  
    
    with torch.no_grad():
        for i, x_val in enumerate(x_lin):
            # Find the actual wall height at this x coordinate
            y_wall = float(np.interp(x_val, profile.x, profile.y))
            y_lin = np.linspace(0.0, y_wall, n_y)
            
            # Prepare the 6-dimensional input tensor: [x, y, A5, A6, P_in, T_in]
            inputs = torch.zeros((n_y, 6), device=DEVICE)
            inputs[:, 0] = x_val
            inputs[:, 1] = torch.tensor(y_lin, dtype=torch.float32)
            inputs[:, 2] = A5
            inputs[:, 3] = A6
            inputs[:, 4] = P_in
            inputs[:, 5] = T_in

            # Distance from each point to the nearest wall (top wall at y_wall)
            wall_dists = torch.tensor(
                (y_wall - y_lin), dtype=torch.float32, device=DEVICE
            ).unsqueeze(1)  # (N, 1)

            # Normalize inputs, run inference, denormalize outputs
            inputs_norm = input_norm.transform(inputs)
            preds_norm = model(inputs_norm, wall_dists)
            preds = output_norm.inverse_transform(preds_norm)
            
            # Extract state variables (Outputs: [ρ, u, v, P, T, UU, VV, UV, μ_eff])
            u = preds[:, 1].numpy()
            v = preds[:, 2].numpy()
            T = preds[:, 4].numpy()
            
            # Physics calculation: True Mach Number
            V_mag = np.sqrt(u**2 + v**2)
            speed_of_sound = np.sqrt(GAMMA * R_GAS * np.maximum(T, 1e-6))
            mach = V_mag / speed_of_sound
            
            # Store in mesh
            X_mesh[:, i] = x_val
            Y_mesh[:, i] = y_lin
            Mach_mesh[:, i] = mach

    # 3. Plot the Actual Neural Network Output
    print("Generating validation contour plot...")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the heatmap using real PINN predictions
    cf = ax.contourf(X_mesh, Y_mesh, Mach_mesh, levels=60, cmap="turbo", vmin=0.1, vmax=2.2)
    
    # Mirror the bottom half so it looks like a full nozzle (y vs -y)
    ax.contourf(X_mesh, -Y_mesh, Mach_mesh, levels=60, cmap="turbo", vmin=0.1, vmax=2.2)
    
    # Draw the rigid nozzle walls
    ax.plot(profile.x, profile.y, color="black", lw=2)
    ax.plot(profile.x, -profile.y, color="black", lw=2)
    
    # Aesthetics
    ax.set_title("Actual LE-PINN Learned Physics (Zero-Shot Mach Contour)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Axial Position (m)")
    ax.set_ylabel("Radial Position (m)")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("True Mach Number ($M = V/a$)", fontsize=12)
    
    # Save the output
    out_path = REPO_ROOT / "outputs" / "plots" / "pinn_physics_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Done! Open {out_path} to see the results.")

if __name__ == "__main__":
    run_zero_shot_mach_test()