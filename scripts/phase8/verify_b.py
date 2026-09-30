#!/usr/bin/env python3
"""P8-A1 variant-b G0 check on the 93 + 87 frozen v6 rows and AE3 W1.

This is run before counting any variant-b benchmark timing.  It uses the
protected Phase 8 G0 Python CSVs as the per-row reference and writes once.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import AE3, ROOT, ae3_task, inputs, protected_check
from v6_optimized import OptimizedModel
import lto_v5 as v5

CAL_FIELDS = {"ff": "ff", "phi": "phi", "thrust_kN": "thrust_kN",
              "tsfc_mg_Ns": "tsfc_mg_Ns", "T3": "T3", "T4": "T4", "T5": "T5",
              "p3_bar": "p3_bar", "thrust_core_kN": "thrust_core_kN",
              "thrust_bypass_kN": "thrust_bypass_kN", "m_core": "m_core",
              "pi_c": "pi_c", "nox_corr_g_s": "nox_corr_g_s"}
HELD_FIELDS = {"ff": "Predicted Fuel Flow (kg/s)", "phi": "model_phi",
               "T3": "model_T3", "T4": "model_T4", "T5": "model_T5",
               "m_core": "model_m_core", "pi_c": "model_pi_c",
               "nox_corr_g_s": "model_nox_corr_g_s", "thrust_kN": "model_thrust_kN"}


def check(pred: pd.DataFrame, frozen: pd.DataFrame, fields: dict) -> dict:
    if len(pred) != len(frozen):
        return {"match": False, "actual_rows": len(pred), "reference_rows": len(frozen)}
    bad = []
    worst = {}
    for current, expected in (("status", "status") if "status" in frozen else ("status", "Status"),):
        mismatch = pred[current].to_numpy() != frozen[expected].to_numpy()
        bad.extend({"row": int(i), "field": current} for i in np.flatnonzero(mismatch)[:10])
    for actual, reference in fields.items():
        a = pred[actual].to_numpy(dtype=float)
        b = frozen[reference].to_numpy(dtype=float)
        ok = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        finite = rel[np.isfinite(rel)]
        worst[actual] = float(finite.max()) if len(finite) else 0.0
        bad.extend({"row": int(i), "field": actual, "actual": float(a[i]),
                    "reference": float(b[i])} for i in np.flatnonzero(~ok)[:10])
    return {"match": not bad, "n_rows": len(pred), "worst_rel_by_field": worst,
            "failures": bad[:30]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/phase8/benchmark/arm2b_g0.json")
    a = ap.parse_args()
    if a.output.exists():
        ap.error(f"{a.output} exists; output is write-once")
    protected = protected_check()
    if protected["mismatches"]:
        ap.error(f"protected hashes changed: {protected['mismatches'][:5]}")
    data = inputs(1)
    model = OptimizedModel(data["reg"]["fixed_central"],
                           data["split"]["heldout_models"], n_workers=1,
                           fuel=data["fuel"], variant="b")
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
    checks = {"calibration": check(cal, cal_ref, CAL_FIELDS),
              "heldout": check(held, held_ref, HELD_FIELDS)}
    ae3_ref = cal_ref.loc[(cal_ref["Unique ID"] == AE3) &
                          (cal_ref["Mode"] == "TAKE-OFF")].iloc[0]
    ae3_fields = {k: (math.isnan(float(ae3.get(k, math.nan))) and math.isnan(float(ae3_ref[v])))
                  or math.isclose(float(ae3.get(k, math.nan)), float(ae3_ref[v]),
                                  rel_tol=1e-9, abs_tol=1e-12)
                  for k, v in CAL_FIELDS.items()}
    checks["ae3"] = {"match": ae3["status"] == ae3_ref["status"] and all(ae3_fields.values()),
                     "fields": ae3_fields}
    verdict = "PASS" if all(c["match"] for c in checks.values()) else "FAIL"
    doc = {"gate": "P8-A1 variant b G0", "verdict": verdict,
           "reference": "outputs/phase8/g0/*_python.csv; frozen v6 source artifacts",
           "rule": "rtol=1e-9, atol=1e-12 per available numeric output; status exact",
           "checks": checks, "protected": protected}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
    print(f"variant b G0: {verdict}; 93 calibration, 87 held-out, AE3 W1")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
