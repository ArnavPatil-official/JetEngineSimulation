"""
Combustor airflow-split diagnostic for the take-off fuel-flow bias (Phase 3.4, F3).

Background: the v3 model overpredicts held-out fuel flow in ALL modes with a
rising-with-power shape (idle +6.1%, approach +12.0%, take-off +19.8% signed
mean), and phi_app converged onto its lower bound — the calibration wanted
less fuel than the phi bounds allowed. The model burns ALL core air at the
calibrated phi, whereas real combustors pass only ~70-80% of core flow
through the burner (Lefebvre & Ballal, Gas Turbine Combustion: ~20-30% of
combustor air is liner cooling + dilution). Fuel flow scales exactly with
beta at fixed phi (m_fuel = FAR(phi) * beta * m_core), so beta provides the
headroom the phi bounds blocked — and makes T4 a diluted (more realistic)
turbine inlet temperature.

This script:
1. Sweeps beta in {1.0, 0.9, 0.8, 0.7} at the frozen v3 parameters and
   reports per-mode signed fuel-flow error against the Trent 1000-AE3
   certification targets (in-sample quick look; at fixed phi the scaling is
   analytically linear in beta — the cycle runs confirm it and give the T4
   effect).
2. The adoption decision (recalibration at a sourced beta -> v4 + holdout)
   is run separately via calibrate_lto.py --beta <value> --tag v4 and
   holdout_icao_validation.py; see the plan's decision rule.

Output: outputs/airflow_split_sensitivity.csv (+ plot)
"""

import sys
import json
import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY, part_power_state

CALIBRATION_JSON = PROJECT_ROOT / "outputs" / "calibration_trent1000_ae3_v3.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "airflow_split_sensitivity.csv"
OUT_PLOT = PROJECT_ROOT / "outputs" / "plots" / "airflow_split_sensitivity.png"

BETAS = [1.0, 0.9, 0.8, 0.7]
MODE_PHI_KEY = {"Idle": "phi_idle", "Approach": "phi_app", "Takeoff": "phi_to"}

# dataviz reference palette, slots 1-3 per mode
MODE_COLORS = {"Takeoff": "#2a78d6", "Approach": "#1baf7a", "Idle": "#eda100"}


def main():
    with open(CALIBRATION_JSON) as fh:
        calib = json.load(fh)
    best = calib["best_params"]
    targets = calib["icao_targets"]
    fixed = calib["fixed_parameters"]

    engine = IntegratedTurbofanEngine()
    engine.design_point["combustor_pressure_loss"] = best["pressure_loss"]

    rows = []
    for beta in BETAS:
        engine.design_point["combustor_air_fraction"] = beta
        for mode, tgt in targets.items():
            x = tgt["power_fraction"]
            pi_c, m_dot = part_power_state(
                x, fixed["base_pi_c"], fixed["base_airflow_kg_s"],
                best["k_pi"], best["k_mdot"])
            engine.design_point["pi_c"] = pi_c
            engine.design_point["mass_flow_core"] = m_dot
            engine.design_point["fpr"] = 1.0 + 0.45 * x ** best["k_pi"]
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                res = engine.run_full_cycle(
                    fuel_blend=FUEL_LIBRARY["Jet-A1"],
                    phi=best[MODE_PHI_KEY[mode]],
                    combustor_efficiency=best["eta_combustor"],
                )
            pred = res["performance"]["fuel_mass_flow"]
            icao = tgt["fuel_flow_kg_s"]
            rows.append({
                "beta": beta, "Mode": mode,
                "Predicted_FF_kg_s": pred, "ICAO_FF_kg_s": icao,
                "Signed_Error_pct": (pred / icao - 1) * 100,
                "T4_K": res["combustor"]["T_out"],
            })
            print(f"beta={beta:.1f}  {mode:<9} ff={pred:.4f} vs {icao:.3f} "
                  f"({rows[-1]['Signed_Error_pct']:+6.2f}%)  "
                  f"T4={rows[-1]['T4_K']:7.1f} K")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\nMean |error| by beta (in-sample AE3, v3 params frozen):")
    for beta, grp in df.groupby("beta"):
        print(f"  beta={beta:.1f}: MAPE {grp['Signed_Error_pct'].abs().mean():6.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for mode in ("Takeoff", "Approach", "Idle"):
        sub = df[df.Mode == mode]
        ax.plot(sub["beta"], sub["Signed_Error_pct"], marker="o", ms=5,
                lw=1.6, color=MODE_COLORS[mode], label=mode)
    ax.axhline(0, color="#9a9a94", lw=1, ls="--")
    ax.set_xlabel("Combustor air fraction β")
    ax.set_ylabel("Signed fuel-flow error vs ICAO [%]")
    ax.set_title("Airflow-split diagnostic (v3 parameters frozen)\n"
                 "fuel flow ∝ β at fixed φ; β<1 supplies the headroom the "
                 "φ bounds blocked")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {OUT_CSV}\n       {OUT_PLOT}")


if __name__ == "__main__":
    main()
