#!/usr/bin/env python3
"""
P6.8: regenerate manifest rows in THIS checkout and compare with the committed artifacts.

Pipeline scripts refuse to overwrite their outputs, so each committed artifact
is moved to a scratch directory, the row's command is re-run, and the new file
is compared with the committed one: numbers to a relative tolerance, strings
exactly; JSON keys that record wall-clock time are ignored. Run it in a
disposable clone (see REPRODUCE.md), never in the working repository.

Usage: .venv/bin/python scripts/reproduce_check.py --in-clone [--rows V3 E4 ...] [--long] [--rtol 1e-9]
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TIME_KEYS = {"started", "finished"}

# (id, command, artifacts compared) — in dependency order
ROWS = [
    ("V1-pilot", ["scripts/optimization/calibrate_lto.py", "--tag", "v5", "--pilot"],
     ["outputs/phase6/p61_pilot_fit.json", "outputs/phase6/p61_pilot_fit_rows.csv"]),
    ("V1-A2", ["scripts/optimization/calibrate_lto.py", "--tag", "v5", "--a2-select"],
     ["outputs/calibration_v5_A2.json", "outputs/calibration_v5_A2_rows.csv"]),
    ("V3", ["scripts/validation/holdout_icao_validation.py", "--tag", "_v5"],
     ["outputs/holdout_icao_validation_v5.csv", "outputs/holdout_icao_validation_summary_v5.csv",
      "outputs/holdout_icao_validation_v5.json"]),
    ("V5", ["scripts/validation/design_point_summary.py", "--v5"], ["outputs/design_point_summary_v5.csv"]),
    ("V6", ["scripts/validation/nox_holdout_validation.py", "--split", "outputs/phase6/split_p61.json"],
     ["outputs/nox_holdout_validation_summary_p61.csv", "outputs/nox_holdout_validation_p61.csv"]),
    ("E3", ["scripts/validation/nox_dual_path.py", "--v5"], ["outputs/nox_dual_path_v5.csv"]),
    ("E4", ["scripts/validation/heat_loss_sensitivity.py", "--v5"], ["outputs/heat_loss_sensitivity_v5.csv"]),
    ("E9", ["scripts/validation/mechanism_sensitivity.py"],
     ["outputs/mechanism_sensitivity_v5.csv", "outputs/mechanism_sensitivity_v5.json"]),
    ("B1", ["scripts/optimization/blend_matched_thrust_v5.py"],
     ["outputs/results/blend_matched_thrust_v5.csv", "outputs/results/blend_matched_thrust_v5_rankings.csv",
      "outputs/results/blend_matched_thrust_v5_refs_p62.csv", "outputs/results/blend_matched_thrust_v5.json"]),
    ("B2-B3", ["scripts/analysis/variance_decomposition.py", "--v5", "--seed", "42", "--n-mc", "1000"],
     ["outputs/results/variance_decomposition_v5.csv", "outputs/results/objective_correlation_v5.csv",
      "outputs/results/lca_rank_stability_v5.csv"]),
]
# Not in the default set (long): full fit ~17 min, profiles ~30 + ~90 min, P6.2 bands ~60 min.
LONG = [
    ("V1-full", ["scripts/optimization/calibrate_lto.py", "--tag", "v5", "--free", "W_ref", "a_thrust",
                 "k_pi", "k_mdot"], ["outputs/calibration_v5.json", "outputs/calibration_v5_rows.csv"]),
    ("V2", ["scripts/validation/identifiability_profile.py", "--v5", "full"],
     ["outputs/identifiability_profile_v5.json", "outputs/identifiability_profile_v5.csv"]),
    ("V8", ["scripts/validation/p62_parameter_bands.py"], ["outputs/p62_bands_v5.csv", "outputs/p62_bands_v5.json"]),
]
# Side outputs the scripts also write (moved aside, not compared)
SIDE = {
    "V1-pilot": ["outputs/phase6/p61_pilot_fit_evaluations.csv"],
    "V1-A2": ["outputs/calibration_v5_A2_evaluations.csv"],
    "V1-full": ["outputs/calibration_v5_evaluations.csv"],
    "V2": ["outputs/identifiability_profile_v5_progress.log"],
    "E3": ["outputs/plots/nox_path_comparison_v5.png"],
    "E4": ["outputs/plots/heat_loss_sensitivity_v5.png"],
}


def close(a, b, rtol) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - TIME_KEYS
        return all(k in a and k in b and close(a[k], b[k], rtol) for k in keys)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12)
    return a == b


def compare(old: Path, new: Path, rtol: float) -> str:
    if old.suffix == ".json":
        return "match" if close(json.loads(old.read_text()), json.loads(new.read_text()), rtol) else "DIFFER"
    a, b = pd.read_csv(old), pd.read_csv(new)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return "DIFFER (shape/columns)"
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not all(close(float(x), float(y), rtol) for x, y in zip(a[c], b[c])):
                return f"DIFFER (column {c})"
        elif not a[c].fillna("").astype(str).equals(b[c].fillna("").astype(str)):
            return f"DIFFER (column {c})"
    return "match"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--long", action="store_true", help="include the long rows")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--in-clone", action="store_true",
                    help="required: confirms this checkout is a disposable clone (artifacts are moved)")
    args = ap.parse_args()
    if not args.in_clone:
        print("refusing to move artifacts without --in-clone; run in a disposable clone (REPRODUCE.md)")
        return 2
    rows = ROWS + (LONG if args.long else [])
    if args.rows:
        rows = [r for r in rows + LONG if r[0] in args.rows]
    scratch = ROOT / "reproduce_committed"
    bad = 0
    for rid, cmd, arts in rows:
        for a in arts + SIDE.get(rid, []):
            dst = scratch / a
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ROOT / a), dst)
        t0 = time.time()
        proc = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"{rid}: command FAILED ({dt:.0f} s)\n{proc.stderr[-2000:]}")
            bad += 1
            continue
        for a in arts:
            res = compare(scratch / a, ROOT / a, args.rtol)
            bad += res != "match"
            print(f"{rid}: {a}: {res} ({dt:.0f} s)")
    print(f"rows checked: {len(rows)}; mismatches/failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
