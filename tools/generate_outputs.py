#!/usr/bin/env python3
"""
Generate curated outputs (text + figures) for the Full_Report.

This script avoids any absolute paths and uses only local mechanisms:
- data/creck_c1c16_full.yaml

It prints:
- Combustor outputs (T_out, cp, R, gamma) for Jet-A1 and a sample SAF blend
- Compressor example (isentropic with efficiency)
And saves a simple comparison plot to outputs/ directory.

Run standalone:
  python tools/generate_outputs.py
"""

import os
import sys
from pathlib import Path
import json
import numpy as np

# Use non-interactive backend for plot saving
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

try:
    import cantera as ct
except Exception as e:
    print(f"[WARN] Cantera import failed: {e}")
    print("       Install Cantera (e.g., conda install -c conda-forge cantera).")
    sys.exit(0)

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Local imports
from simulation.fuels import JET_A1, make_saf_blend
from simulation.combustor.combustor import Combustor
from simulation.compressor.compressor import Compressor

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_combustor_examples(mech_path: Path):
    print("\n=== Combustor Demo ===")
    combustor = Combustor(mechanism_file=str(mech_path))

    # Conditions (typical compressor exit ballpark)
    T_in = 900.0
    p_in = 15.0 * ct.one_atm
    phi = 0.35
    eff = 0.98

    cases = [
        ("Jet-A1", JET_A1),
        ("SAF 70/30 Jet-A1/HEFA", make_saf_blend(p_j=0.7, p_h=0.3, p_f=0.0, p_a=0.0)),
    ]

    results = []
    for label, fuel in cases:
        out = combustor.run(T_in=T_in, p_in=p_in, fuel_blend=fuel, phi=phi, efficiency=eff)
        entry = {
            "label": label,
            "T_out": float(out["T_out"]),
            "p_out": float(out["p_out"]),
            "cp_out": float(out["cp_out"]),
            "R_out": float(out["R_out"]),
            "gamma_out": float(out["gamma_out"]),
        }
        results.append(entry)
        print(f"- {label}: T_out={entry['T_out']:.2f} K, cp={entry['cp_out']:.1f} J/kg-K, "
              f"R={entry['R_out']:.2f}, gamma={entry['gamma_out']:.4f}")

    # Plot comparison
    ensure_dir(REPO_ROOT / "outputs")
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r["T_out"] for r in results], width=0.4, label="T_out [K]")
    ax2 = ax.twinx()
    ax2.plot(x, [r["gamma_out"] for r in results], "ro--", label="gamma")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in results], rotation=15)
    ax.set_ylabel("Temperature [K]")
    ax2.set_ylabel("gamma [-]")
    ax.set_title("Combustor: T_out and gamma vs Fuel")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPO_ROOT / "outputs" / "combustor_comparison.png"
    plt.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")

    return results

def run_compressor_example(mech_path: Path):
    print("\n=== Compressor Demo ===")
    gas = ct.Solution(str(mech_path))
    comp = Compressor(gas=gas, eta_c=0.85, pi_c=10.0)
    out = comp.compute_outlet_state(T_in=288.15, p_in=101325.0)
    print(f"- T_out={out['T_out']:.2f} K, p_out={out['p_out']:.1f} Pa, "
          f"work_specific={out['work_specific']:.1f} J/kg")
    return out

def main():
    mech = REPO_ROOT / "data" / "creck_c1c16_full.yaml"
    if not mech.exists():
        print(f"[WARN] Mechanism missing: {mech}")
        print("       Place 'creck_c1c16_full.yaml' in data/ to run the demos.")
        return

    comb_res = run_combustor_examples(mech)
    comp_res = run_compressor_example(mech)

    # Save a machine-readable copy of numeric outputs
    ensure_dir(REPO_ROOT / "outputs")
    with open(REPO_ROOT / "outputs" / "demo_outputs.json", "w") as f:
        json.dump({"combustor": comb_res, "compressor": comp_res}, f, indent=2)

if __name__ == "__main__":
    main()
