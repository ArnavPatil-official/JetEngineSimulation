#!/usr/bin/env python3
"""
P4.3 — Score every registered Sajben attempt and apply the pre-registered
bands, without exception and without re-tuning.

Reads each checkpoint's ``attempt`` record, scores it with
``sajben_validation.py`` (corrected throat-height mapping, split guard), and
writes ``outputs/sajben_retrain_v5.md`` + ``outputs/sajben_retrain_v5.csv``.
The band is applied to the WORSE of the two walls (max of upper/lower
shape-L2), which is the conservative reading of "held-out wall-Cp shape-L2".

Bands (docs/plan.md P4.3, fixed before any run):
    < 0.10        pass     LE-PINN becomes production; PINN framing earned
    0.10 – 0.25   partial  quantified near-miss; PINN stays non-production
    > 0.25        fail     retired to an appendix

Usage::

    python scripts/validation/sajben_report_p43.py \
        models/le_pinn_sajben_v5.pt models/le_pinn_sajben_v5_dataonly_ref.pt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation import sajben_validation as sv  # noqa: E402

OUT_MD = REPO_ROOT / "outputs" / "sajben_retrain_v5.md"
OUT_CSV = REPO_ROOT / "outputs" / "sajben_retrain_v5.csv"
GATE_PASS, GATE_PARTIAL = 0.10, 0.25
WIND_SELF_SCORE = (0.089, 0.084)   # P4.1 §3: the training data's own score


def band(v: float) -> str:
    if v < GATE_PASS:
        return "pass"
    if v < GATE_PARTIAL:
        return "partial"
    return "fail"


def score(path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        r = sv.main(model_file=path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    att = ck.get("attempt") or {}
    worst = max(r["l2_cp_upper"], r["l2_cp_lower"])
    row = {
        "checkpoint": path.name,
        "attempt": att.get("id", "(none)"),
        "gate_attempt": not str(att.get("id", "")).startswith("P4.3-reference"),
        "physics_weight": att.get("physics_loss_weight"),
        "epochs": ck.get("epochs_run"), "collapsed_at_epoch": ck.get("collapsed_at_epoch"),
        "best_epoch": ck.get("best_epoch"), "val_loss_best": ck.get("val_loss_best"),
        "L2_Cp_upper": r["l2_cp_upper"], "L2_Cp_lower": r["l2_cp_lower"], "L2_Cp_worst": worst,
        "degenerate": bool(r["degenerate_upper"] or r["degenerate_lower"]),
        "band": band(worst),
        "split_ok": r["split"] is not None and r["split"].get("eval_rows_in_train") == 0,
        "git_sha": (ck.get("git_sha") or "")[:8], "seed": ck.get("seed"),
    }
    for lbl, info in sorted(r["vel_profile_errors"].items(), key=lambda kv: float(kv[0])):
        row[f"L2_u_XH_{lbl}"] = info["l2_error"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()
    rows = [score(Path(p)) for p in args.checkpoints]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    gate_rows = df[df["gate_attempt"]]
    L = [f"# Nozzle LE-PINN retrain on the Sajben weak-shock case (P4.3) — {date.today().isoformat()}\n",
         "Scored by `sajben_validation.py` (throat-height mapping, split guard, corrected in P4.2). "
         f"Pre-registered bands on the worse wall: < {GATE_PASS} pass / {GATE_PASS}–{GATE_PARTIAL} partial / > {GATE_PARTIAL} fail. "
         f"The training data (WIND RANS) itself scores {WIND_SELF_SCORE[0]} / {WIND_SELF_SCORE[1]} on this metric (P4.1 §3), "
         "so the pass band is reachable only by a near-perfect surrogate of the CFD.\n",
         "## Every attempt, as registered\n",
         "| Checkpoint | Attempt | Gate attempt? | Physics w | Epochs (best val at) | Internal val MSE | Collapsed | Upper L2 | Lower L2 | Worst | Degenerate | Split verified | Band |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['checkpoint']}` | {r['attempt']} | {'yes' if r['gate_attempt'] else 'no (ablation)'} | {r['physics_weight']} | "
                 f"{r['epochs']} ({r['best_epoch']}) | {r['val_loss_best']:.2e} | "
                 f"{r['collapsed_at_epoch'] if r['collapsed_at_epoch'] is not None else '—'} | "
                 f"{r['L2_Cp_upper']:.3f} | {r['L2_Cp_lower']:.3f} | {r['L2_Cp_worst']:.3f} | {'YES' if r['degenerate'] else 'no'} | "
                 f"{'yes' if r['split_ok'] else 'NO'} | **{r['band'] if r['gate_attempt'] else r['band'] + ' (n/a)'}** |")
    L.append("\nVelocity-profile L2 (x/H = 1.729 / 2.882 / 4.611 / 6.340):\n")
    for r in rows:
        v = [r.get(k) for k in sorted(k for k in r if k.startswith("L2_u_XH_"))]
        L.append(f"- `{r['checkpoint']}`: " + " / ".join(f"{x:.3f}" for x in v))

    L.append("\n## Outcome applied to the gate attempt(s)\n")
    for r in gate_rows.itertuples(index=False):
        if r.band == "pass":
            L.append(f"- **{r.attempt}: PASS ({r.L2_Cp_worst:.3f}).** The LE-PINN becomes the production nozzle model "
                     "for the Sajben case and the physics-informed framing is earned for a single-condition surrogate "
                     "trained on RANS and validated on an independent experiment. P4.6 re-runs the component ablation with it.")
        elif r.band == "partial":
            L.append(f"- **{r.attempt}: PARTIAL ({r.L2_Cp_worst:.3f}).** The PINN stays non-production. The preprint reports "
                     f"a quantified near-miss against the Sajben benchmark: worse wall {r.L2_Cp_worst:.3f} vs the 0.10 gate, "
                     f"with the training data itself at {max(WIND_SELF_SCORE):.3f}. This is a publishable negative result.")
        else:
            L.append(f"- **{r.attempt}: FAIL ({r.L2_Cp_worst:.3f}).** The nozzle PINN is retired to an appendix; the title and "
                     "framing drop the accuracy claim; production stays analytic.")
    ref = df[~df["gate_attempt"]]
    if len(ref):
        L.append("\n## Ablation reference (not a gate attempt)\n")
        for r in ref.itertuples(index=False):
            L.append(f"- `{r.checkpoint}` (physics weight {r.physics_weight}): worse wall {r.L2_Cp_worst:.3f} "
                     f"(would fall in the *{r.band}* band). Reported so the effect of the physics term is visible: "
                     "the comparison between this row and the gate attempt is the ablation.")
    L.append("\n## Observations recorded for any future attempt (no re-tuning was done)\n")
    for r in rows:
        if r["best_epoch"] is not None and r["epochs"] and r["best_epoch"] < 0.2 * r["epochs"]:
            L.append(f"- `{r['checkpoint']}`: best internal validation at epoch {r['best_epoch']} of {r['epochs']}; "
                     "the run made no further progress after that point (see the training log for the learning-rate "
                     "trajectory). Whether the registered scheduler interacts badly with the physics warm-up is a "
                     "question for a separately registered attempt, not a reason to adjust this one.")
    L.append(f"\nArtifacts: `{OUT_CSV.relative_to(REPO_ROOT)}`, checkpoints listed above, "
             "`outputs/logs/train_sajben_v5_*.log`.\n")
    OUT_MD.write_text("\n".join(L))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[["checkpoint", "attempt", "physics_weight", "epochs", "collapsed_at_epoch",
                  "L2_Cp_upper", "L2_Cp_lower", "L2_Cp_worst", "degenerate", "band"]].to_string(index=False))
    print(f"\nWritten: {OUT_MD.relative_to(REPO_ROOT)}, {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
