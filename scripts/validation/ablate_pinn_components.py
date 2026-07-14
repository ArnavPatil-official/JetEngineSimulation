"""
PINN-vs-analytic component ablation (Phase 2.5, V10).

Runs the full cycle for four fuels x four model combinations
{turbine, nozzle} x {pinn, analytic} at the calibrated take-off design point
(frozen v3 calibration) and reports the deviation each PINN introduces
relative to the all-analytic cycle.

Decision rule (plan AB1/AB2): if the max deviation is smaller than the
held-out calibration error (12.71% MAPE, v3), the manuscript claims the PINNs
as physics-consistency surrogate layers (architecture contribution), NOT as
accuracy contributors.

Output: outputs/ablation_pinn_components.csv
"""

import sys
import json
import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY, LocalFuelBlend
from simulation.fuels import HEFA_SPK, FT_SPK, ATJ_SPK

CALIBRATION_JSON = PROJECT_ROOT / "outputs" / "calibration_trent1000_ae3_v3.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "ablation_pinn_components.csv"

COMBOS = [
    ("analytic", "analytic"),
    ("pinn", "analytic"),
    ("analytic", "pinn"),
    ("pinn", "pinn"),
]


def main():
    with open(CALIBRATION_JSON) as fh:
        calib = json.load(fh)
    best = calib["best_params"]
    phi = best["phi_to"]
    eta_b = best["eta_combustor"]

    fuels = [
        FUEL_LIBRARY["Jet-A1"],
        LocalFuelBlend(HEFA_SPK.name, HEFA_SPK.species),
        LocalFuelBlend(FT_SPK.name, FT_SPK.species),
        LocalFuelBlend(ATJ_SPK.name, ATJ_SPK.species),
    ]

    engine = IntegratedTurbofanEngine()
    engine.design_point["combustor_pressure_loss"] = best["pressure_loss"]

    rows = []
    for fuel in fuels:
        for turb, nozz in COMBOS:
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                res = engine.run_full_cycle(
                    fuel_blend=fuel, phi=phi, combustor_efficiency=eta_b,
                    turbine_model=turb, nozzle_model=nozz,
                )
            p = res["performance"]
            rows.append({
                "Fuel": fuel.name,
                "Turbine": turb,
                "Nozzle": nozz,
                "T5_K": res["turbine"]["T"],
                "p5_bar": res["turbine"]["p"] / 1e5,
                "Thrust_kN": p["thrust_kN"],
                "Core_Thrust_kN": p["thrust_core_kN"],
                "TSFC_mg_per_Ns": p["tsfc_mg_per_Ns"],
                "Fuel_Flow_kg_s": p["fuel_mass_flow"],
            })
            print(f"{fuel.name:<10} turbine={turb:<8} nozzle={nozz:<8} "
                  f"T5={rows[-1]['T5_K']:7.1f} K  thrust={rows[-1]['Thrust_kN']:6.1f} kN  "
                  f"TSFC={rows[-1]['TSFC_mg_per_Ns']:6.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 78)
    print("DEVIATION vs ALL-ANALYTIC BASELINE (per fuel, worst combo)")
    print("=" * 78)
    summary = []
    for fuel, grp in df.groupby("Fuel", sort=False):
        base = grp[(grp.Turbine == "analytic") & (grp.Nozzle == "analytic")].iloc[0]
        others = grp[(grp.Turbine != "analytic") | (grp.Nozzle != "analytic")]
        dev = {
            "Fuel": fuel,
            "max_dT5_K": (others["T5_K"] - base["T5_K"]).abs().max(),
            "max_dThrust_pct": ((others["Thrust_kN"] / base["Thrust_kN"] - 1) * 100).abs().max(),
            "max_dCoreThrust_pct": ((others["Core_Thrust_kN"] / base["Core_Thrust_kN"] - 1) * 100).abs().max(),
            "max_dTSFC_pct": ((others["TSFC_mg_per_Ns"] / base["TSFC_mg_per_Ns"] - 1) * 100).abs().max(),
            "max_dFF_pct": ((others["Fuel_Flow_kg_s"] / base["Fuel_Flow_kg_s"] - 1) * 100).abs().max(),
        }
        summary.append(dev)
        print(f"{fuel:<10} dT5={dev['max_dT5_K']:6.1f} K  "
              f"dThrust={dev['max_dThrust_pct']:5.2f}% (core {dev['max_dCoreThrust_pct']:5.2f}%)  "
              f"dTSFC={dev['max_dTSFC_pct']:5.2f}%  dFF={dev['max_dFF_pct']:5.3f}%")

    pd.DataFrame(summary).to_csv(
        OUT_CSV.with_name("ablation_pinn_components_summary.csv"), index=False)
    print(f"\nHeld-out fuel-flow MAPE (v3) for comparison: 12.71%")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
