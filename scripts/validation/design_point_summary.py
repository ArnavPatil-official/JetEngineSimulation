"""
Take-off design-point summary on the production configuration (Phase 3).

Runs one full cycle at the adopted calibration (default v4) with the
adjudicated analytic components and writes every manuscript-bound
design-point number to a CSV — so none of them exists only in a printout
(plan F2 / ARTIFACT_MANIFEST rule).

Output: outputs/design_point_summary_v4.csv
"""

import sys
import json
import argparse
import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration",
                    default="outputs/calibration_trent1000_ae3_v4.json")
    ap.add_argument("--tag", default="_v4")
    args = ap.parse_args()

    with open(PROJECT_ROOT / args.calibration) as fh:
        calib = json.load(fh)
    best = calib["best_params"]
    fixed = calib["fixed_parameters"]

    engine = IntegratedTurbofanEngine()
    engine.design_point["combustor_pressure_loss"] = best["pressure_loss"]
    engine.design_point["combustor_air_fraction"] = fixed.get(
        "combustor_air_fraction", 1.0)

    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        res = engine.run_full_cycle(
            fuel_blend=FUEL_LIBRARY["Jet-A1"], phi=best["phi_to"],
            combustor_efficiency=best["eta_combustor"],
            turbine_model="analytic", nozzle_model="analytic",
        )
    p = res["performance"]
    row = {
        "calibration": Path(args.calibration).name,
        "component_config": "turbine=analytic, nozzle=analytic (P3.1 adjudicated)",
        "phi_to": best["phi_to"],
        "eta_comb": best["eta_combustor"],
        "combustor_air_fraction": fixed.get("combustor_air_fraction", 1.0),
        "T3_K": res["compressor"]["T_out"],
        "T4_K": res["combustor"]["T_out"],
        "T5_K": res["turbine"]["T"],
        "p5_bar": res["turbine"]["p"] / 1e5,
        "fuel_flow_kg_s": p["fuel_mass_flow"],
        "icao_takeoff_fuel_flow_kg_s": calib["icao_targets"]["Takeoff"]["fuel_flow_kg_s"],
        "thrust_kN": p["thrust_kN"],
        "thrust_core_kN": p["thrust_core_kN"],
        "thrust_bypass_kN": p["thrust_bypass_kN"],
        "TSFC_mg_per_Ns": p["tsfc_mg_per_Ns"],
        "specific_thrust_Ns_kg": p["specific_thrust_Ns_kg"],
        "total_air_mass_flow_kg_s": p["total_air_mass_flow"],
    }
    out = PROJECT_ROOT / "outputs" / f"design_point_summary{args.tag}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    for k, v in row.items():
        print(f"  {k:<30} {v if isinstance(v, str) else round(v, 4)}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
