#!/usr/bin/env python3
"""
P4.2 — Re-score the two existing Sajben checkpoints under the corrected
scorer and record old vs new numbers side by side.

The pre-P4.2 scorer mapped experimental x/H with the throat *radius* (H/2)
instead of the throat height; ``outputs/sajben_validation_errors.csv`` holds
those numbers (frozen v4 artifact, left untouched until the P4.6 archive
step).  This script writes ``outputs/sajben_validation_errors_rescored.csv``
with both, plus the predicted wall-pressure span and the degeneracy flag
that explains why neither number measures anything (both checkpoints are
collapsed networks; see the P4.2 commit message).

Usage::

    python scripts/validation/sajben_rescore_p42.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation import sajben_validation as sv  # noqa: E402

OLD_CSV = REPO_ROOT / "outputs" / "sajben_validation_errors.csv"
OUT_CSV = REPO_ROOT / "outputs" / "sajben_validation_errors_rescored.csv"
MODELS = ["le_pinn_sajben.pt", "le_pinn_sajben_finetuned.pt"]


def main() -> pd.DataFrame:
    old = pd.read_csv(OLD_CSV).set_index("Model") if OLD_CSV.exists() else None
    rows = []
    for name in MODELS:
        with contextlib.redirect_stdout(io.StringIO()):
            r = sv.main(model_file=REPO_ROOT / "models" / name)
        row = {
            "Model": name,
            "L2_Cp_upper_old_scorer": float(old.loc[name, "L2_Cp_upper"]) if old is not None else None,
            "L2_Cp_lower_old_scorer": float(old.loc[name, "L2_Cp_lower"]) if old is not None else None,
            "L2_Cp_upper": r["l2_cp_upper"],
            "L2_Cp_lower": r["l2_cp_lower"],
            "pred_Cp_span_upper": r["cp_span_upper"],
            "pred_Cp_span_lower": r["cp_span_lower"],
            "degenerate": bool(r["degenerate_upper"] or r["degenerate_lower"]),
        }
        for lbl, info in sorted(r["vel_profile_errors"].items(), key=lambda kv: float(kv[0])):
            row[f"L2_u_XH_{lbl}"] = info["l2_error"]
        rows.append(row)
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    print(f"\nWritten: {OUT_CSV.relative_to(REPO_ROOT)}")
    return df


if __name__ == "__main__":
    main()
