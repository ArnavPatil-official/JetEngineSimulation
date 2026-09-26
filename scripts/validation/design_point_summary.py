"""
Take-off design-point summary on the production configuration (Phase 3).

Runs one full cycle at the adopted calibration (default v4) with the
adjudicated analytic components and writes every manuscript-bound
design-point number to a CSV — so none of them exists only in a printout
(plan F2 / ARTIFACT_MANIFEST rule).

Output: outputs/design_point_summary_v4.csv

--v5 (Phase 6, P6.6): Trent 1000-AE3 at the ICAO thrust of each LTO mode
(take-off is the design point), v5 calibration outputs/calibration_v5_A2.json,
A1 central fixed values, single-zone equilibrium combustor, phi solved by
run_at_thrust. Output: outputs/design_point_summary_v5.csv (write-once).
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


def main_v5():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimization"))
    import lto_v5 as v5
    from integrated_engine import EmissionsEstimator
    out = PROJECT_ROOT / "outputs" / "design_point_summary_v5.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    reg, split = v5.load_registration(), v5.load_split()
    fit = json.loads(v5.V5_FIT.read_text())
    fixed = reg["fixed_central"]
    ae3 = v5.load_rows(["02P23RR126"], with_targets=True)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        engine = IntegratedTurbofanEngine()
        engine.emissions = EmissionsEstimator(nox_fit_exclude_models=set(split["heldout_models"]))
    engine.compressor.eta_c = fixed["eta_compressor"]
    engine.turbine_design["eta_polytropic"] = fixed["eta_turbine_polytropic"]
    rows = []
    for _, r in ae3.iterrows():
        x = v5.MODE_X[r["Mode"]]
        engine.design_point.update(v5.mode_state(fit["params"], fixed, r["Pressure Ratio"],
                                                 r["Bypass Ratio"], r["Rated Thrust (kN)"], x))
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res = engine.run_at_thrust(x * r["Rated Thrust (kN)"], FUEL_LIBRARY["Jet-A1"],
                                       combustor_efficiency=fixed["eta_b"][r["Mode"]])
        p = res["performance"]
        rows.append({
            "calibration": v5.V5_FIT.name, "mode": r["Mode"],
            "component_config": "turbine=analytic, nozzle=analytic; single-zone equilibrium combustor",
            "target_thrust_kN": x * r["Rated Thrust (kN)"], "thrust_kN": p["thrust_kN"],
            "phi_solved": res["thrust_match"]["phi"], "eta_b": fixed["eta_b"][r["Mode"]],
            "pi_c": engine.design_point["pi_c"], "fpr": engine.design_point["fpr"],
            "mass_flow_core_kg_s": engine.design_point["mass_flow_core"],
            "T3_K": res["compressor"]["T_out"], "T4_K": res["combustor"]["T_out"],
            "T5_K": res["turbine"]["T"], "p5_bar": res["turbine"]["p"] / 1e5,
            "fuel_flow_kg_s": p["fuel_mass_flow"],
            "icao_fuel_flow_kg_s": float(r["Fuel Flow (kg/s)"]),
            "fuel_flow_rel_err_pct": 100 * (p["fuel_mass_flow"] / r["Fuel Flow (kg/s)"] - 1),
            "thrust_core_kN": p["thrust_core_kN"], "thrust_bypass_kN": p["thrust_bypass_kN"],
            "TSFC_mg_per_Ns": p["tsfc_mg_per_Ns"],
            "specific_thrust_Ns_kg": p["specific_thrust_Ns_kg"],
            "total_air_mass_flow_kg_s": p["total_air_mass_flow"],
            "NOx_corr_g_s": res["emissions"]["NOx_g_s"],
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(f"\nSaved: {out} (AE3 is a calibration-group engine: in-sample)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration",
                    default="outputs/calibration_trent1000_ae3_v4.json")
    ap.add_argument("--tag", default="_v4")
    ap.add_argument("--v5", action="store_true",
                    help="Phase 6 thrust-matched v5 design point (all LTO modes)")
    args = ap.parse_args()
    if args.v5:
        main_v5()
        return

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
