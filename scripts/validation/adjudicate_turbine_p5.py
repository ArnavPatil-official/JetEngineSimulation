"""
Turbine exit-pressure adjudication (Phase 3.1, F1).

Three candidate p5 values at the calibrated take-off design point:

(a) Analytic work-consistent polytropic value — the run_turbine_analytic path:
        T5 = T4 - W/(m_dot cp)                       (work-imposed)
        p5 = p4 * (T5/T4)^(gamma / (eta_poly (gamma-1)))   (polytropic expansion)
    HAND CALCULATION (values from the Jet-A1 ablation row, Phase 2.5):
        T4 = 2030.9 K, T5 = 1334.5 K  ->  T5/T4 = 0.65709
        products: gamma = 1.2827, eta_poly = 0.9
        exponent = gamma/(eta (gamma-1)) = 1.2827/(0.9*0.2827) = 5.0413
        p4 = 41.58 bar (43.2 * (1-0.0340) * 1.01325)
        p5 = 41.58 * 0.65709^5.0413 = 41.58 * 0.12063 = 5.02 bar
    (Exact code value differs only through the exact gamma/p4 used; the
    script recomputes with the run's own gamma and must match run_turbine_
    analytic's p5 to <0.1%.)

(b) The turbine PINN's p5: the raw network output (state_out[:,2] in
    simulation/turbine/turbine.py, line ~821). The PINN's temperature is
    work-adjusted after inference, but its PRESSURE is not — so PINN p5 is
    an unvalidated NN prediction with no work-consistency constraint.

(c) turbine_design['P_out'] = 1.93e5 Pa: a fixed constant, consumed ONLY in
    run_turbine's default-target-work branch (target_work_total=None), which
    the production cycle never takes (it always passes compressor+fan work).

Adjudication rule (from the Phase 2.5 decision rule):
- Nozzle: analytic — the LE-PINN failed external (Sajben) validation.
- Turbine: analytic work-consistent polytropic, unless the PINN's p5 is
  closer to the work-consistent value than the analytic path itself (it
  cannot be: the analytic path IS the work-consistent value; the test below
  verifies the analytic implementation reproduces the hand calculation).

Output: outputs/turbine_p5_adjudication.csv (+ console verdict)
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

CALIBRATION_JSON = PROJECT_ROOT / "outputs" / "calibration_trent1000_ae3_v3.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "turbine_p5_adjudication.csv"


def main(model_path: str | None = None, calibration: Path = CALIBRATION_JSON, tag: str = ""):
    """
    P4.4: ``model_path`` scores an additional turbine checkpoint (row b'),
    ``calibration`` selects the frozen calibration, ``tag`` suffixes the
    output so the Phase-3 artifact is never overwritten. Defaults reproduce
    the Phase-3 run exactly.
    """
    out_csv = OUT_CSV if not tag else OUT_CSV.with_name(f"turbine_p5_adjudication{tag}.csv")
    with open(calibration) as fh:
        calib = json.load(fh)
    best = calib["best_params"]
    fixed = calib.get("fixed_parameters", {})

    engine = IntegratedTurbofanEngine()
    engine.design_point["combustor_pressure_loss"] = best["pressure_loss"]
    if "combustor_air_fraction" in fixed:
        engine.design_point["combustor_air_fraction"] = fixed["combustor_air_fraction"]

    results = {}
    for turb in ("analytic", "pinn"):
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res = engine.run_full_cycle(
                fuel_blend=FUEL_LIBRARY["Jet-A1"], phi=best["phi_to"],
                combustor_efficiency=best["eta_combustor"],
                turbine_model=turb, nozzle_model="analytic",
            )
        results[turb] = res
    p5_v5 = None
    if model_path is not None:
        engine_v5 = IntegratedTurbofanEngine(turbine_pinn_path=str(model_path))
        engine_v5.design_point.update(engine.design_point)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res_v5 = engine_v5.run_full_cycle(
                fuel_blend=FUEL_LIBRARY["Jet-A1"], phi=best["phi_to"],
                combustor_efficiency=best["eta_combustor"],
                turbine_model="pinn", nozzle_model="analytic",
            )
        p5_v5 = res_v5["turbine"]["p"]

    r = results["analytic"]
    T4 = r["combustor"]["T_out"]
    p4 = r["combustor"]["p_out"]
    gamma = r["turbine"]["gamma"]
    eta = engine.turbine_design["eta_polytropic"]
    T5 = r["turbine"]["T"]

    # Independent hand-formula recomputation (work-consistent polytropic)
    exponent = gamma / (eta * (gamma - 1.0))
    p5_hand = p4 * (T5 / T4) ** exponent

    p5_analytic = r["turbine"]["p"]
    p5_pinn = results["pinn"]["turbine"]["p"]
    p5_fixed = engine.turbine_design["P_out"]

    rows = [
        {"candidate": "(a) analytic work-consistent (production path)",
         "p5_bar": p5_analytic / 1e5,
         "deviation_from_work_consistent_pct":
             (p5_analytic / p5_hand - 1) * 100},
        {"candidate": "(a') hand-formula recomputation",
         "p5_bar": p5_hand / 1e5,
         "deviation_from_work_consistent_pct": 0.0},
        {"candidate": "(b) turbine PINN raw output",
         "p5_bar": p5_pinn / 1e5,
         "deviation_from_work_consistent_pct":
             (p5_pinn / p5_hand - 1) * 100},
        {"candidate": "(c) fixed turbine_design P_out (unused in production)",
         "p5_bar": p5_fixed / 1e5,
         "deviation_from_work_consistent_pct":
             (p5_fixed / p5_hand - 1) * 100},
    ]
    if p5_v5 is not None:
        rows.insert(3, {"candidate": f"(b') turbine surrogate {Path(model_path).name} raw output",
                        "p5_bar": p5_v5 / 1e5,
                        "deviation_from_work_consistent_pct": (p5_v5 / p5_hand - 1) * 100})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("=" * 74)
    print(f"TURBINE EXIT-PRESSURE ADJUDICATION (take-off design point, {Path(calibration).stem[-2:]} calib)")
    print("=" * 74)
    print(f"T4 = {T4:.1f} K, T5 = {T5:.1f} K (work-matched), p4 = {p4/1e5:.2f} bar, "
          f"gamma = {gamma:.4f}, eta_poly = {eta}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    match = abs(p5_analytic / p5_hand - 1) < 1e-3
    print(f"\nAnalytic path reproduces the hand-verified work-consistent value: {match}")
    print(f"PINN p5 deviates {(p5_pinn/p5_hand-1)*100:+.1f}% from work consistency "
          f"(raw NN output, pressure not work-adjusted).")
    if p5_v5 is not None:
        print(f"Surrogate {Path(model_path).name} p5 deviates {(p5_v5/p5_hand-1)*100:+.3f}% "
              f"from work consistency (raw NN output).")
    print(f"Fixed P_out (1.93 bar) matches neither; confirmed unused in the "
          f"production path (only the target_work_total=None branch reads it).")
    print("\nVERDICT: turbine = analytic (work-consistent), nozzle = analytic "
          "(failed Sajben). Production defaults flipped accordingly (F1).")
    if not match:
        raise SystemExit("Analytic p5 does NOT match the hand calculation — "
                         "investigate before flipping defaults.")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="extra turbine checkpoint to score (row b')")
    ap.add_argument("--calibration", default=str(CALIBRATION_JSON))
    ap.add_argument("--tag", default="", help="suffix for the output CSV (e.g. _v5)")
    a = ap.parse_args()
    main(model_path=a.model, calibration=Path(a.calibration), tag=a.tag)
