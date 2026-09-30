#!/usr/bin/env python3
"""P8-A1 products-only approximation gate on 180 frozen v6 rows and AE3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import AE3, ROOT, ae3_task, inputs, protected_check
from v6_optimized import OptimizedModel, PRODUCT_SPECIES
import lto_v5 as v5


def check(pred: pd.DataFrame, reference: pd.DataFrame, ff_col: str,
          t4_col: str, status_col: str) -> dict:
    ff_a, ff_b = pred["ff"].to_numpy(float), reference[ff_col].to_numpy(float)
    t4_a, t4_b = pred["T4"].to_numpy(float), reference[t4_col].to_numpy(float)
    status_equal = pred["status"].to_numpy() == reference[status_col].to_numpy()
    ff_rel = np.abs(ff_a - ff_b) / np.abs(ff_b)
    t4_abs = np.abs(t4_a - t4_b)
    ff_ok = (ff_rel <= 1e-4) | (np.isnan(ff_a) & np.isnan(ff_b))
    t4_ok = (t4_abs <= 0.1) | (np.isnan(t4_a) & np.isnan(t4_b))
    bad = np.flatnonzero(~(status_equal & ff_ok & t4_ok))
    return {"match": len(pred) == len(reference) and len(bad) == 0,
            "n_rows": len(pred), "max_ff_relative": float(np.nanmax(ff_rel)),
            "max_T4_abs_K": float(np.nanmax(t4_abs)),
            "bad_rows": bad[:30].tolist()}


def source_sha256() -> str:
    """Hash of the variant code this gate result certifies."""
    return hashlib.sha256((ROOT / "scripts/phase8/v6_optimized.py").read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/phase8/benchmark/arm2c_gate_rev2.json")
    a = ap.parse_args()
    if a.output.exists():
        ap.error(f"{a.output} exists; output is write-once")
    protected = protected_check()
    if protected["mismatches"]:
        ap.error(f"protected hashes changed: {protected['mismatches'][:5]}")
    data = inputs(1)
    model = OptimizedModel(data["reg"]["fixed_central"],
                           data["split"]["heldout_models"], n_workers=1,
                           fuel=data["fuel"], variant="c")
    try:
        cal = model.predict(data["params"], data["cal"])
        model.guess.clear()
        held = model.predict(data["params"], data["held"])
        model.guess.clear()
        ae3 = list(model.pool.map(v5.solve_task, [ae3_task(data, data["params"])]))[0]
    finally:
        model.close()
    cal_ref = pd.read_csv(ROOT / "outputs/phase8/g0/calibration_v6_rows_python.csv")
    held_ref = pd.read_csv(ROOT / "outputs/phase8/g0/holdout_icao_validation_v6_python.csv")
    checks = {"calibration": check(cal, cal_ref, "ff", "T4", "status"),
              "heldout": check(held, held_ref, "Predicted Fuel Flow (kg/s)",
                               "model_T4", "Status")}
    ae3_ref = cal_ref.loc[(cal_ref["Unique ID"] == AE3) &
                          (cal_ref["Mode"] == "TAKE-OFF")].iloc[0]
    ae3_ff = abs(ae3["ff"] - ae3_ref["ff"]) / ae3_ref["ff"]
    ae3_t4 = abs(ae3["T4"] - ae3_ref["T4"])
    checks["ae3"] = {"match": bool(ae3["status"] == ae3_ref["status"]
                      and ae3_ff <= 1e-4 and ae3_t4 <= 0.1),
                     "ff_relative": ae3_ff, "T4_abs_K": ae3_t4}
    baseline_ff = held_ref["Predicted Fuel Flow (kg/s)"]
    baseline_status = held_ref["Status"]
    mape_ref = v5.weighted_mape(baseline_ff, data["held"], baseline_status)
    mape_c = v5.weighted_mape(held["ff"], data["held"], held["status"])
    mape_delta = abs(mape_c - mape_ref)
    checks["heldout_mape"] = {"match": bool(mape_delta <= 0.01),
                              "reference_pct": mape_ref, "variant_c_pct": mape_c,
                              "absolute_change_percentage_points": mape_delta}
    verdict = "PASS" if all(c["match"] for c in checks.values()) else "FAIL"
    doc = {"gate": "P8-A1 variant c approximation", "verdict": verdict,
           "product_species": PRODUCT_SPECIES,
           "rule": "all 180 + AE3: |delta FF|/FF <= 1e-4, |delta T4| <= 0.1 K; "
                   "held-out group-weighted MAPE change <= 0.01 percentage points",
           "reference": "outputs/phase8/g0/*_python.csv", "checks": checks,
           "source_sha256": source_sha256(),
           "protected": protected}
    payload = json.dumps(doc, indent=2) + "\n"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(payload)
    print(f"variant c gate: {verdict}; max FF rel "
          f"{max(checks[k]['max_ff_relative'] for k in ('calibration', 'heldout')):.3e}; "
          f"max T4 delta {max(checks[k]['max_T4_abs_K'] for k in ('calibration', 'heldout')):.3e} K")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
