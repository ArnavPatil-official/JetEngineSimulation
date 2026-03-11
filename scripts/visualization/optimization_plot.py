import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import re
import matplotlib.pyplot as plt
import pandas as pd

# 1. Paste your FULL terminal output into a file named 'outputs/results/full_logs.txt'
# 2. Run this script from the project root

def parse_and_plot(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        log_data = f.read()

    # Split by trials to keep data organized
    chunks = log_data.split("RUNNING FULL ENGINE CYCLE: ")
    parsed_data = []

    for i, chunk in enumerate(chunks):
        # Find Trial Number
        if i == 0: t_match = re.search(r"Fuel Blend:\s+Trial_(\d+)_Blend", chunk)
        else:      t_match = re.search(r"Trial_(\d+)_Blend", chunk)
        
        if not t_match: continue
        trial_num = int(t_match.group(1))
        
        # Extract Metrics
        tsfc_match = re.search(r"TSFC:\s+([\d\.]+)\s+mg", chunk)
        thrust_match = re.search(r"Thrust:\s+([\d\.]+)\s+kN", chunk)
        turb_u_match = re.search(r"Turbine Exit State.*?u\s+=\s+([\d\.]+)\s+m/s", chunk, re.DOTALL)
        nozz_u_match = re.search(r"Nozzle PINN Results.*?Exit State:.*?u=([\d\.]+)\s+m/s", chunk, re.DOTALL)

        if tsfc_match and thrust_match and turb_u_match and nozz_u_match:
            parsed_data.append({
                "Trial": trial_num,
                "TSFC": float(tsfc_match.group(1)),
                "Thrust": float(thrust_match.group(1)),
                "Turbine_U": float(turb_u_match.group(1)),
                "Nozzle_U": float(nozz_u_match.group(1))
            })

    df = pd.DataFrame(parsed_data)
    
    # --- PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pareto Plot
    sc = axes[0].scatter(df['TSFC'], df['Thrust'], c=df['Trial'], cmap='viridis', s=100, edgecolors='k')
    axes[0].set_xlabel("TSFC (mg/N·s) [Minimize]")
    axes[0].set_ylabel("Thrust (kN) [Maximize]")
    axes[0].set_title(f"Pareto Front ({len(df)} Points Rescued)")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label="Trial Number")

    # PINN Validation Plot
    axes[1].scatter(df['Turbine_U'], df['Nozzle_U'], color='orange', edgecolors='k', s=80)
    axes[1].plot([300, 800], [300, 800], 'k--', alpha=0.5, label="No Acceleration")
    axes[1].set_xlabel("Turbine Exit Velocity (m/s)")
    axes[1].set_ylabel("Nozzle Exit Velocity (m/s)")
    axes[1].set_title("PINN Acceleration Validation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Run it
try:
    parse_and_plot('outputs/results/full_logs.txt')
except FileNotFoundError:
    print("❌ Please save your terminal output to 'outputs/results/full_logs.txt' first!")