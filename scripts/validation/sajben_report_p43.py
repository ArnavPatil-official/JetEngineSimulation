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

Terminal-attempt guard (P5.1): a checkpoint of a ``terminal`` attempt is
only reported when ALL of that attempt's registered runs (every seed,
physics-on and matched data-only) have a checkpoint and completion evidence
with exit code 0. Evidence is, in order of preference:

* ``<log>.done`` written by ``train_sajben.py``'s launch guard — observed
  exit code, bound to the checkpoint by SHA-256;
* for runs launched before the guard existed (the 2026-09-19 data-only
  runs), the ``exit=N`` trailer their launcher appended to the log, accepted
  only if the log also names the checkpoint as saved and its "best val"
  matches the checkpoint. The report labels which kind each run has.

Anything else — missing checkpoint, missing evidence, non-zero exit, or a
checkpoint whose recorded attempt/seed is not the one registered at its path —
makes the report refuse (non-zero exit) before either output is written.

The scored selection is also closed over terminal attempts: naming any one run
of a terminal attempt selects all of its registered runs (every seed, physics-on
and data-only, in registration order), so a subset can never be scored as the
terminal outcome. Duplicate paths, and checkpoints that claim a terminal attempt
from outside its registered paths, are refused. Earlier attempts are reported
exactly as selected. A half-run terminal attempt cannot be reported.

Usage::

    python scripts/validation/sajben_report_p43.py            # every registered attempt
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
from scripts.validation.train_sajben import (  # noqa: E402
    ATTEMPTS, default_log_path, done_marker_path,
)

OUT_MD = REPO_ROOT / "outputs" / "sajben_retrain_v5.md"
OUT_CSV = REPO_ROOT / "outputs" / "sajben_retrain_v5.csv"
GATE_PASS, GATE_PARTIAL = 0.10, 0.25
WIND_SELF_SCORE = (0.089, 0.084)   # P4.1 §3: the training data's own score
CEILING = max(WIND_SELF_SCORE)     # supplementary reference, not a gate change
NEAR_BOUNDARY = 0.01               # a single-seed score this close to a band edge is not called


def band(v: float) -> str:
    if v < GATE_PASS:
        return "pass"
    if v < GATE_PARTIAL:
        return "partial"
    return "fail"


# Every registered run, in registration order (the no-argument default).
REGISTERED = [
    "models/le_pinn_sajben_v5.pt",                 # attempt 1
    "models/le_pinn_sajben_v5_dataonly_ref.pt",    # reference 1 (data-only)
    "models/le_pinn_sajben_v5_a2.pt",              # attempt 2
] + [ATTEMPTS[3][k].format(seed=s) for s in ATTEMPTS[3]["seeds"] for k in ("out", "out_dataonly")]


def completion_evidence(checkpoint: Path, log: Path) -> dict:
    """Exit-code evidence for one run; ``ok`` only if it shows exit 0 for THIS checkpoint."""
    import hashlib
    import json
    import re

    ev = {"checkpoint": str(checkpoint.relative_to(REPO_ROOT)), "log": str(log.relative_to(REPO_ROOT)),
          "ok": False, "kind": None, "exit_code": None, "detail": ""}
    if not checkpoint.exists():
        ev["detail"] = "checkpoint missing"
        return ev
    marker = done_marker_path(log)
    if marker.exists():
        rec = json.loads(marker.read_text())
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ev.update(kind="launch-guard .done marker", exit_code=rec.get("exit_code"))
        if rec.get("exit_code") != 0:
            ev["detail"] = f"exit code {rec.get('exit_code')}"
        elif rec.get("checkpoint") != ev["checkpoint"]:
            ev["detail"] = f"marker names checkpoint {rec.get('checkpoint')!r} (mismatched identity)"
        elif rec.get("checkpoint_sha256") != sha:
            ev["detail"] = "marker SHA-256 does not match the checkpoint"
        else:
            ev.update(ok=True, detail=f"sha256 {sha[:12]} matches; finished {rec.get('finished')}")
        return ev
    if not log.exists() or log.stat().st_size == 0:
        ev["detail"] = "no .done marker and no (non-empty) log"
        return ev
    text = log.read_text()
    m = re.search(r"^exit=(\d+)\s*\Z", text, re.M)
    ev["kind"] = "legacy log trailer (exit=N appended by the pre-guard launcher)"
    if m is None:
        ev["detail"] = "log has no exit=N trailer"
        return ev
    ev["exit_code"] = int(m.group(1))
    saved = f"Fine-tuned checkpoint saved: {checkpoint.resolve()}" in text
    bv = re.search(r"best val ([0-9.eE+-]+)", text)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vb = ck.get("val_loss_best")
    val_match = bv is not None and vb is not None and f"{vb:.3e}" == bv.group(1)
    if ev["exit_code"] != 0:
        ev["detail"] = f"exit code {ev['exit_code']}"
    elif not saved:
        ev["detail"] = "log does not record saving this checkpoint"
    elif not val_match:
        ev["detail"] = f"log best val {bv.group(1) if bv else None} != checkpoint {vb}"
    else:
        ev.update(ok=True, detail=f"log names this checkpoint as saved; best val {bv.group(1)} matches the checkpoint")
    return ev


def _terminal_runs(a: dict) -> list[tuple[int, str, Path]]:
    """Every registered run of a terminal attempt, in registration order."""
    return [(s, k, (REPO_ROOT / a[k].format(seed=s)).resolve())
            for s in a["seeds"] for k in ("out", "out_dataonly")]


def resolve_selection(paths: list[Path]) -> list[Path]:
    """
    The scored selection, with every terminal attempt expanded to its complete
    registered set (inserted where its first run appears, in registration
    order). Refuses (SystemExit) duplicates and checkpoints that claim a
    terminal attempt without being one of its registered paths.
    """
    resolved = [p.resolve() for p in paths]
    dups = sorted({str(p) for p in resolved if resolved.count(p) > 1})
    if dups:
        raise SystemExit("REFUSING: checkpoint(s) selected more than once:\n" + "\n".join(f"  {d}" for d in dups))
    owner = {}   # registered terminal run path -> attempt number
    for n, a in ATTEMPTS.items():
        if a.get("terminal"):
            for _s, _k, rp in _terminal_runs(a):
                owner[rp] = n
    out, expanded = [], set()
    for p in resolved:
        n = owner.get(p)
        if n is None:
            att = (torch.load(p, map_location="cpu", weights_only=False).get("attempt") or {}) if p.exists() else {}
            if att.get("terminal"):
                raise SystemExit(f"REFUSING: {p} records terminal attempt {att.get('id')!r} but is not one of "
                                 "that attempt's registered checkpoint paths (mismatched identity)")
            out.append(p)
        elif n not in expanded:
            expanded.add(n)
            out.extend(rp for _s, _k, rp in _terminal_runs(ATTEMPTS[n]))
    return out


def terminal_guard(paths: list[Path]) -> list[dict]:
    """Refuse (SystemExit 2) unless every registered run of each terminal attempt completed with exit 0."""
    families = set()
    for p in paths:
        att = (torch.load(p, map_location="cpu", weights_only=False).get("attempt") or {}) if p.exists() else {}
        if att.get("terminal"):
            families.add(str(att.get("id")).removesuffix("-dataonly"))
    # a terminal attempt whose runs are absent from `paths` must still be complete
    for p in paths:
        for n, a in ATTEMPTS.items():
            if a.get("terminal") and any(p.name == Path(a[k].format(seed=s)).name
                                         for s in a["seeds"] for k in ("out", "out_dataonly")):
                families.add(a["id"])
    evidence = []
    for n, a in ATTEMPTS.items():
        if not a.get("terminal") or a["id"] not in families:
            continue
        for s in a["seeds"]:
            for k in ("out", "out_dataonly"):
                ev = completion_evidence(REPO_ROOT / a[k].format(seed=s),
                                         default_log_path(n, s, k == "out_dataonly"))
                ev.update(attempt=a["id"] + ("-dataonly" if k == "out_dataonly" else ""), seed=s)
                if ev["ok"]:
                    ck = torch.load(REPO_ROOT / a[k].format(seed=s), map_location="cpu", weights_only=False)
                    got = ((ck.get("attempt") or {}).get("id"), ck.get("seed"))
                    if got != (ev["attempt"], s):
                        ev.update(ok=False, detail=f"checkpoint records attempt/seed {got}, registered "
                                                   f"{(ev['attempt'], s)} (mismatched identity)")
                evidence.append(ev)
    bad = [e for e in evidence if not e["ok"]]
    if bad:
        msg = "\n".join(f"  {e['attempt']} seed {e['seed']}: {e['checkpoint']} — {e['detail']}" for e in bad)
        raise SystemExit(f"REFUSING to report a terminal attempt with incomplete runs:\n{msg}")
    return evidence


def score(path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        r = sv.main(model_file=path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    att = ck.get("attempt") or {}
    worst = max(r["l2_cp_upper"], r["l2_cp_lower"])
    row = {
        "checkpoint": path.name,
        "attempt": att.get("id", "(none)"),
        "gate_attempt": not (str(att.get("id", "")).startswith("P4.3-reference")
                             or str(att.get("id", "")).endswith("-dataonly")),
        "physics_weight": att.get("physics_loss_weight"),
        "epochs": ck.get("epochs_run"), "collapsed_at_epoch": ck.get("collapsed_at_epoch"),
        "best_epoch": ck.get("best_epoch"), "val_loss_best": ck.get("val_loss_best"),
        "L2_Cp_upper": r["l2_cp_upper"], "L2_Cp_lower": r["l2_cp_lower"], "L2_Cp_worst": worst,
        "degenerate": bool(r["degenerate_upper"] or r["degenerate_lower"]),
        "band": band(worst),
        "ceiling_rel": worst - CEILING,
        "activation": ck.get("activation", "relu"),
        "mu_source": (ck.get("config") or {}).get("physics_mu_source", "sutherland"),
        "split_ok": r["split"] is not None and r["split"].get("eval_rows_in_train") == 0,
        "git_sha": (ck.get("git_sha") or "")[:8], "seed": ck.get("seed"),
    }
    for lbl, info in sorted(r["vel_profile_errors"].items(), key=lambda kv: float(kv[0])):
        row[f"L2_u_XH_{lbl}"] = info["l2_error"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", help="default: every registered run (REGISTERED)")
    args = ap.parse_args()
    paths = [(REPO_ROOT / p) if not Path(p).is_absolute() else Path(p)
             for p in (args.checkpoints or REGISTERED)]
    paths = resolve_selection(paths)
    evidence = terminal_guard(paths)
    rows = [score(p) for p in paths]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    df["family"] = df["attempt"].str.replace(r"-dataonly$", "", regex=True)
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
    seen_multi = set()
    for r in gate_rows.itertuples(index=False):
        n_seeds = int(df.loc[df["family"] == r.family, "seed"].nunique())
        if n_seeds > 1:
            if r.family in seen_multi:
                continue
            seen_multi.add(r.family)
            vals = gate_rows.loc[gate_rows["family"] == r.family, "L2_Cp_worst"].to_numpy()
            bands = sorted(set(band(v) for v in vals))
            if len(bands) > 1:
                L.append(f"- **{r.family}: seeds straddle {'/'.join(bands)} (range {vals.min():.3f}–{vals.max():.3f}, "
                         f"mean {vals.mean():.3f}); no band claimed.** Reported as straddling per the registered rule.")
                continue
            r = r._replace(L2_Cp_worst=float(vals.mean()), band=bands[0], attempt=f"{r.family} (mean of {len(vals)} seeds)")
        near = min(abs(r.L2_Cp_worst - GATE_PASS), abs(r.L2_Cp_worst - GATE_PARTIAL))
        if n_seeds == 1 and near < NEAR_BOUNDARY:
            L.append(f"- **{r.attempt}: {r.L2_Cp_worst:.3f} on a single seed, within {near:.3f} of a band boundary — "
                     f"band not claimed.** A single draw this close to the line does not support a band call; "
                     "see the multi-seed section for the attempt that carries the outcome.")
            continue
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
    # ---- multi-seed attempts: band on the distribution, matched ablation ----
    multi = [f for f in df["family"].unique() if (df["family"] == f).sum() > 1 and df.loc[df["family"] == f, "seed"].nunique() > 1]
    if multi:
        L.append("\n## Multi-seed attempts — band called on the distribution\n")
        L.append("| Attempt | Config | Seeds | Worse-wall L2 per seed | Mean ± sd | Range | Ceiling-relative (mean − 0.089) | Band (mean) | Seeds agree? |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        fam_stats = {}
        for f in multi:
            for is_ref in (False, True):
                part = df[(df["family"] == f) & (df["attempt"].str.endswith("-dataonly") == is_ref)].sort_values("seed")
                if part.empty:
                    continue
                vals = part["L2_Cp_worst"].to_numpy()
                bands = sorted(set(band(v) for v in vals))
                agree = len(bands) == 1
                label = part["attempt"].iloc[0]
                fam_stats[label] = vals
                cfg = f"{part['activation'].iloc[0]}, μ={part['mu_source'].iloc[0]}, physics w {part['physics_weight'].iloc[0]}"
                L.append(f"| {label} | {cfg} | {', '.join(str(int(x)) for x in part['seed'])} | "
                         f"{' / '.join(f'{v:.3f}' for v in vals)} | {vals.mean():.3f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.3f} | "
                         f"{vals.min():.3f}–{vals.max():.3f} | {vals.mean() - CEILING:+.3f} | "
                         f"**{band(vals.mean())}**{'' if agree else ' (not claimed)'} | {'yes' if agree else 'NO — straddles ' + '/'.join(bands)} |")
        L.append("")
        for f in multi:
            on, off = fam_stats.get(f), fam_stats.get(f + "-dataonly")
            if on is None or off is None:
                continue
            spread = max(on.std(ddof=1) if len(on) > 1 else 0.0, off.std(ddof=1) if len(off) > 1 else 0.0)
            d = on.mean() - off.mean()
            if abs(d) <= spread:
                verdict = ("physics-on is within the seed spread of data-only: **physics consistency at no "
                           "accuracy cost** (pre-registered reading 1)")
            elif d < 0:
                verdict = ("physics-on is below data-only beyond the seed spread: **the physics term improves "
                           "agreement with the experiment** (pre-registered reading 2)")
            else:
                verdict = ("physics-on is above data-only beyond the seed spread: **a negative result about this "
                           "residual formulation**, reported as such (pre-registered reading 3)")
            L.append(f"**{f} vs its matched data-only ablation:** physics-on mean {on.mean():.3f}, data-only mean "
                     f"{off.mean():.3f}, difference {d:+.3f} against a seed spread (larger sd) of {spread:.3f} → {verdict}.")
        L.append("\nThe pass band (< 0.10) sits 0.011 above the training data's own score; the ceiling-relative column "
                 "shows how much of each score is the surrogate's error rather than the CFD's. The gate itself is unchanged.")
    if evidence:
        L.append("\n## Completion evidence for the terminal attempt (P5.1 guard)\n")
        L.append("Every registered run must have completed with exit code 0 before the terminal attempt is reported.\n")
        L.append("| Run | Seed | Checkpoint | Evidence | Exit | Check |")
        L.append("|---|---|---|---|---|---|")
        for e in evidence:
            L.append(f"| {e['attempt']} | {e['seed']} | `{e['checkpoint']}` | {e['kind']} | {e['exit_code']} | {e['detail']} |")
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
