#!/usr/bin/env python3
"""P8-A1.2 gates for the native (arms 3/4) benchmark variants.

On all 93 calibration rows, 87 held-out rows and AE3 W1, against the
protected Phase 8 G0 Python CSVs:
- "full" (the benchmark-only copied engine with full equilibrium) and "b"
  (V6Engine + probe bracket): G0 rule, rtol 1e-9 / atol 1e-12, status exact.
  "full" passing shows the copy that carries variant c is faithful.
- "c" (products-only equilibrium + probe): the registered approximation rule,
  |dFF|/FF <= 1e-4, |dT4| <= 0.1 K, held-out MAPE change <= 0.01 pp.
Untimed. Writes outputs/phase8/benchmark/native_variants_gate.json once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from benchmark import AE3, ROOT, NativeModel, inputs, protected_check
from verify_b import CAL_FIELDS, HELD_FIELDS, check as check_g0
from verify_c import check as check_c
import lto_v5 as v5

SOURCES = ("cpp/catjet_core/v6_variant.cpp", "cpp/catjet_core/v6_variant.hpp",
           "cpp/bindings/catjet_benchmark.cpp", "cpp/catjet_core/v6_engine.cpp",
           "cpp/catjet_core/v6_engine.hpp", "cpp/catjet_core/brentq.hpp")


def source_sha256() -> str:
    """One hash over the native benchmark sources this gate certifies."""
    h = hashlib.sha256()
    for name in SOURCES:
        h.update(name.encode() + b"\0" + (ROOT / name).read_bytes() + b"\0")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/phase8/benchmark/native_variants_gate.json")
    a = ap.parse_args()
    if a.output.exists():
        ap.error(f"{a.output} exists; output is write-once")
    protected = protected_check()
    if protected["mismatches"]:
        ap.error(f"protected hashes changed: {protected['mismatches'][:5]}")
    data = inputs(1)
    cal_ref = pd.read_csv(ROOT / "outputs/phase8/g0/calibration_v6_rows_python.csv")
    held_ref = pd.read_csv(ROOT / "outputs/phase8/g0/holdout_icao_validation_v6_python.csv")
    ae3_ref = cal_ref.loc[(cal_ref["Unique ID"] == AE3) & (cal_ref["Mode"] == "TAKE-OFF")].iloc[0]

    results, counts = {}, {}
    for variant in ("full", "b", "c"):
        model = NativeModel(data, n_threads=11, variant=variant)
        try:
            held = pd.DataFrame(model.evaluate("W2", data)[0], index=data["held"].index)
            cal = pd.DataFrame(model.evaluate("W3", data)[0], index=data["cal"].index)
            ae3 = model.evaluate("W1", data)[0][0]
            counts[variant] = model.cycle_counts("W3")
        finally:
            model.close()
        held, cal = held.reset_index(drop=True), cal.reset_index(drop=True)
        if variant in ("full", "b"):
            checks = {"calibration": check_g0(cal, cal_ref, CAL_FIELDS),
                      "heldout": check_g0(held, held_ref, HELD_FIELDS)}
            fields = {k: (math.isnan(float(ae3.get(k, math.nan))) and math.isnan(float(ae3_ref[v])))
                      or math.isclose(float(ae3.get(k, math.nan)), float(ae3_ref[v]),
                                      rel_tol=1e-9, abs_tol=1e-12)
                      for k, v in CAL_FIELDS.items()}
            checks["ae3"] = {"match": ae3["status"] == ae3_ref["status"] and all(fields.values()),
                             "fields": fields}
        else:
            checks = {"calibration": check_c(cal, cal_ref, "ff", "T4", "status"),
                      "heldout": check_c(held, held_ref, "Predicted Fuel Flow (kg/s)",
                                         "model_T4", "Status")}
            ff = abs(ae3["ff"] - ae3_ref["ff"]) / ae3_ref["ff"]
            t4 = abs(ae3["T4"] - ae3_ref["T4"])
            checks["ae3"] = {"match": bool(ae3["status"] == ae3_ref["status"] and ff <= 1e-4 and t4 <= 0.1),
                             "ff_relative": ff, "T4_abs_K": t4}
            mape_ref = v5.weighted_mape(held_ref["Predicted Fuel Flow (kg/s)"], data["held"],
                                        held_ref["Status"])
            mape_c = v5.weighted_mape(held["ff"].set_axis(data["held"].index), data["held"],
                                      held["status"].set_axis(data["held"].index))
            checks["heldout_mape"] = {"match": bool(abs(mape_c - mape_ref) <= 0.01),
                                      "reference_pct": mape_ref, "variant_c_pct": mape_c,
                                      "absolute_change_percentage_points": abs(mape_c - mape_ref)}
        results[variant] = {"verdict": "PASS" if all(c["match"] for c in checks.values()) else "FAIL",
                            "checks": checks}
        print(variant, results[variant]["verdict"], flush=True)

    doc = {"gate": "P8-A1.2 native variants (arms 3/4)",
           "verdict_full": results["full"]["verdict"],
           # b and c timings count only if the copy and their own rule pass
           "verdict_b": results["b"]["verdict"],
           "verdict_c": ("PASS" if results["c"]["verdict"] == "PASS" and
                         results["full"]["verdict"] == "PASS" else "FAIL"),
           "rules": {"full_and_b": "rtol=1e-9, atol=1e-12 per numeric output; status exact",
                     "c": "|dFF|/FF <= 1e-4, |dT4| <= 0.1 K, held-out MAPE change <= 0.01 pp"},
           "reference": "outputs/phase8/g0/*_python.csv",
           "results": results,
           "untimed_W3_cycle_calls": {v: [c["actual_cycle_calls"] for c in counts[v]] for v in counts},
           "source_sha256": source_sha256(), "sources": SOURCES,
           "protected": protected}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(doc, indent=2, default=float) + "\n")
    print({k: v for k, v in doc.items() if k.startswith("verdict")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
