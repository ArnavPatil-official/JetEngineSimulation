#!/usr/bin/env python3
"""
P7.4 report: score all 30 registered runs of the new LE-PINN study and apply
the registered claim rule (``outputs/phase7/p74_registration.json``).

Refuses (non-zero exit, nothing written) unless EVERY registered run has a
record and a checkpoint whose SHA-256 matches the record, the record's
configuration/split hashes match the registration, and (inside the Phase 7
runner) a COMPLETE.json. An incomplete batch cannot be reported as a
terminal result.

Scores per run:
* experiment: upper/lower wall P/P_in shape-L2 with the existing scorer
  (sajben_validation.build_sajben_grid + compute_wall_cp_errors); primary
  scalar = the worse wall; existing bands; degenerate flags;
* WIND held-out rows (816): relative L2 and RMSE of rho, u, v, p, T; aggregate
  = mean of the five relative L2; mu_t and mu_eff separately.
Labels: WIND held-out = in-case spatial interpolation; experiment = external
comparison on the same case; no cross-case claim.

Outputs (write-once): outputs/phase7/p74_scores.csv, p74_report.json, p74_report.md
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from simulation.nozzle import le_pinn_ma as lm  # noqa: E402

REGISTRATION = ROOT / "outputs" / "phase7" / "p74_registration.json"
GATE_PASS, GATE_PARTIAL = 0.10, 0.25
CLAIM_FRACTIONS = ("f002", "f005", "f010")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def band(v: float) -> str:
    return "pass" if v < GATE_PASS else ("partial" if v < GATE_PARTIAL else "fail")


def verify_complete(reg: dict, root: Path = ROOT, require_runner: bool = True) -> list[dict]:
    """Every registered run complete and hash-bound, or SystemExit."""
    problems, recs = [], []
    for r in reg["runs"]:
        rec_p, ck_p = root / r["record"], root / r["checkpoint"]
        if not rec_p.exists() or not ck_p.exists():
            problems.append(f"{r['run_id']}: missing record or checkpoint")
            continue
        rec = json.loads(rec_p.read_text())
        if rec.get("final_checkpoint_sha256") != sha256(ck_p):
            problems.append(f"{r['run_id']}: checkpoint hash differs from its record")
        if rec.get("train_ids_sha256") != r["train_ids_sha256"] or rec.get("smoke"):
            problems.append(f"{r['run_id']}: record is not the registered run (subset or smoke)")
        if rec.get("epochs") != reg["training"]["epochs"]:
            problems.append(f"{r['run_id']}: epochs {rec.get('epochs')} != registered")
        if require_runner:
            done = root / "outputs" / "phase7" / "runs" / "pinn" / r["run_id"] / "COMPLETE.json"
            if not done.exists():
                problems.append(f"{r['run_id']}: no runner COMPLETE.json")
        recs.append(rec)
    if problems:
        raise SystemExit("P7.4 report refused (incomplete or inconsistent batch):\n  " + "\n  ".join(problems))
    return recs


def load_model(path: Path, geom: lm.Geometry) -> lm.MaLEPINN:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["model_config"]
    model = lm.MaLEPINN(geom, ck["scalers"], width=m["width"], n_hidden=m["n_hidden"],
                        b_width=m["b_width"], b_hidden=m["b_hidden"], delta=m["fusion_delta_m"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def experimental_scores(model: lm.MaLEPINN) -> dict:
    from scripts.validation import sajben_validation as sv
    from simulation.nozzle.le_pinn import parse_sajben_experimental_data, parse_sajben_geometry
    geom = parse_sajben_geometry(str(sv.GEOM_FILE))
    exp = parse_sajben_experimental_data(str(sv.DATA_FILE))
    n_ax, n_no = 60, 25
    inputs_raw, x_vec, upper_y = sv.build_sajben_grid(geom, n_axial=n_ax, n_normal=n_no)
    with torch.no_grad():
        f = model.fields(inputs_raw[:, :2].float())
    preds = torch.stack([f[k] for k in lm.FLOW_VARS], dim=1)
    r = sv.compute_wall_cp_errors(preds, inputs_raw, x_vec, upper_y, exp, n_no)
    return {"l2_upper": r["l2_upper"], "l2_lower": r["l2_bot"],
            "primary_worse_wall": max(r["l2_upper"], r["l2_bot"]),
            "degenerate_upper": r["degenerate_upper"], "degenerate_lower": r["degenerate_bot"],
            "span_upper": r["span_upper"], "span_lower": r["span_bot"],
            "n_top": r["n_top_pts"], "n_bot": r["n_bot_pts"]}


def wind_scores(model: lm.MaLEPINN, tab: np.ndarray) -> dict:
    xy = torch.tensor(tab[:, :2], dtype=torch.float32)
    with torch.no_grad():
        f = model.fields(xy)
        mu_eff_pred = (lm.sutherland(f["T"]) + f["mu_t"]).double().numpy()
    out = {}
    rels = []
    for k, col in zip(lm.FLOW_VARS, range(2, 7)):
        ref, pred = tab[:, col], f[k].double().numpy()
        rel = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
        out[f"wind_rel_l2_{k}"] = rel
        out[f"wind_rmse_{k}"] = float(np.sqrt(np.mean((pred - ref) ** 2)))
        rels.append(rel)
    out["wind_rel_l2_aggregate"] = float(np.mean(rels))
    mut_ref = tab[:, 7]
    mut_pred = f["mu_t"].double().numpy()
    out["wind_rel_l2_mu_t"] = float(np.linalg.norm(mut_pred - mut_ref) / np.linalg.norm(mut_ref))
    out["wind_rmse_mu_t"] = float(np.sqrt(np.mean((mut_pred - mut_ref) ** 2)))
    mueff_ref = tab[:, 8] + tab[:, 7]
    out["wind_rel_l2_mu_eff"] = float(np.linalg.norm(mu_eff_pred - mueff_ref) / np.linalg.norm(mueff_ref))
    return out


def claim_table(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for fk, g in scores.groupby("fraction_key", sort=False):
        ph = g[g["arm"] == "physics"].set_index("seed")["primary_worse_wall"]
        do = g[g["arm"] == "dataonly"].set_index("seed")["primary_worse_wall"]
        paired = (do - ph).sort_index()
        spread = max(float(ph.std(ddof=1)), float(do.std(ddof=1)))
        diff = float(do.mean() - ph.mean())
        rows.append({"fraction_key": fk, "fraction": float(g["fraction"].iloc[0]),
                     "physics_mean": float(ph.mean()), "physics_sd": float(ph.std(ddof=1)),
                     "dataonly_mean": float(do.mean()), "dataonly_sd": float(do.std(ddof=1)),
                     "mean_dataonly_minus_physics": diff, "seed_spread": spread,
                     "benefit": bool(diff > spread),
                     **{f"paired_diff_s{s}": float(v) for s, v in paired.items()},
                     "paired_diff_mean": float(paired.mean()), "paired_diff_sd": float(paired.std(ddof=1)),
                     "physics_band_of_mean": band(float(ph.mean())),
                     "dataonly_band_of_mean": band(float(do.mean()))})
    t = pd.DataFrame(rows)
    n_benefit = int(t[t["fraction_key"].isin(CLAIM_FRACTIONS)]["benefit"].sum())
    verdict = {"n_claim_fractions_with_benefit": n_benefit,
               "physics_benefit_claimed": bool(n_benefit >= 2),
               "rule": "mean(data-only primary) - mean(physics primary) > max(sampleSD(physics), "
                       "sampleSD(data-only)) (ddof 1) at >= 2 of {2, 5, 10} %"}
    return t, verdict


def main(root: Path = ROOT, require_runner: bool = True) -> dict:
    reg = json.loads((root / "outputs" / "phase7" / "p74_registration.json").read_text())
    outs = [root / p for p in reg["report_outputs"]]
    for p in outs:
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    recs = verify_complete(reg, root, require_runner)
    split = json.loads((root / reg["split"]["file"]).read_text())
    sol = lm.load_wind()
    geom = lm.geometry_from(sol)
    test_ids = np.asarray(split["test"], dtype=np.int64)
    tab = np.column_stack([lm.label_table(sol, test_ids), sol.mu_l.ravel()[test_ids]])
    rows = []
    for r, rec in zip(reg["runs"], recs):
        model = load_model(root / r["checkpoint"], geom)
        with contextlib.redirect_stdout(io.StringIO()):
            e = experimental_scores(model)
        w = wind_scores(model, tab)
        rows.append({"run_id": r["run_id"], "seed": r["seed"], "fraction": r["fraction"],
                     "fraction_key": r["fraction_key"], "arm": r["arm"], "n_train_rows": r["n_train_rows"],
                     **e, "band_primary": band(e["primary_worse_wall"]), **w,
                     "final_L_data": rec["final_training_side"]["L_data"],
                     "final_L_phys": rec["final_training_side"]["L_phys"],
                     "final_L_bc": rec["final_training_side"]["L_bc"],
                     "checkpoint_sha256": rec["final_checkpoint_sha256"]})
        print(f"{r['run_id']}: worse wall {e['primary_worse_wall']:.4f}  WIND agg {w['wind_rel_l2_aggregate']:.4f}",
              flush=True)
    scores = pd.DataFrame(rows)
    claims, verdict = claim_table(scores)
    wind = scores.groupby(["fraction_key", "arm"], sort=False)[
        ["wind_rel_l2_aggregate", "wind_rel_l2_mu_t", "wind_rel_l2_mu_eff"]].agg(["mean", "std"])
    report = {"registration": "outputs/phase7/p74_registration.json", "n_runs": len(scores),
              "verdict": verdict, "per_fraction": claims.to_dict("records"),
              "wind_heldout_by_fraction_arm": {f"{a}|{b}": {f"{c[0]}_{c[1]}": float(v) for c, v in row.items()}
                                               for (a, b), row in wind.iterrows()},
              "labels": reg["scoring"]["labels"]}
    scores.to_csv(outs[0], index=False)
    with open(outs[1], "x") as fh:
        fh.write(json.dumps(report, indent=2) + "\n")
    with open(outs[2], "x") as fh:
        fh.write(report_md(scores, claims, verdict, wind))
    print(json.dumps(verdict, indent=2))
    return report


def report_md(scores, claims, verdict, wind) -> str:
    L = ["# P7.4 new LE-PINN study (Ma-form residual, current-loss weights): results", "",
         "Registered in `outputs/phase7/p74_registration.json` before training; all 30 runs complete "
         "(hash-verified). New study, not P4.3 attempt 4; nothing promoted to the production cycle.", "",
         f"**Physics benefit claimed: {'YES' if verdict['physics_benefit_claimed'] else 'NO'}** "
         f"({verdict['n_claim_fractions_with_benefit']} of the 3 small fractions meet the rule; needs >= 2). "
         f"Rule: {verdict['rule']}.", "",
         "## Experimental worse-wall shape-L2 (primary; external comparison, same case)", "",
         "| Fraction | physics mean ± SD | data-only mean ± SD | Δ (data − physics) | seed spread | benefit | "
         "paired Δ s42 / s43 / s44 |", "|---|---|---|---|---|---|---|"]
    for _, r in claims.iterrows():
        L.append(f"| {r['fraction']*100:g} % | {r['physics_mean']:.4f} ± {r['physics_sd']:.4f} | "
                 f"{r['dataonly_mean']:.4f} ± {r['dataonly_sd']:.4f} | {r['mean_dataonly_minus_physics']:+.4f} | "
                 f"{r['seed_spread']:.4f} | {'yes' if r['benefit'] else 'no'} | "
                 f"{r['paired_diff_s42']:+.4f} / {r['paired_diff_s43']:+.4f} / {r['paired_diff_s44']:+.4f} |")
    L += ["", "## Per run", "",
          "| Run | upper | lower | worse | band | degenerate | WIND agg rel-L2 | ρ | u | v | p | T | μ_t | μ_eff |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in scores.iterrows():
        L.append(f"| {r['run_id']} | {r['l2_upper']:.4f} | {r['l2_lower']:.4f} | {r['primary_worse_wall']:.4f} | "
                 f"{r['band_primary']} | {bool(r['degenerate_upper']) or bool(r['degenerate_lower'])} | "
                 f"{r['wind_rel_l2_aggregate']:.4f} | {r['wind_rel_l2_rho']:.4f} | {r['wind_rel_l2_u']:.4f} | "
                 f"{r['wind_rel_l2_v']:.4f} | {r['wind_rel_l2_p']:.4f} | {r['wind_rel_l2_T']:.4f} | "
                 f"{r['wind_rel_l2_mu_t']:.4f} | {r['wind_rel_l2_mu_eff']:.4f} |")
    L += ["", "## WIND held-out (in-case spatial interpolation), mean ± SD over seeds", "",
          "| Fraction | Arm | aggregate rel-L2 | μ_t rel-L2 | μ_eff rel-L2 |", "|---|---|---|---|---|"]
    for (fk, arm), row in wind.iterrows():
        L.append(f"| {fk} | {arm} | {row[('wind_rel_l2_aggregate','mean')]:.4f} ± "
                 f"{row[('wind_rel_l2_aggregate','std')]:.4f} | {row[('wind_rel_l2_mu_t','mean')]:.4f} | "
                 f"{row[('wind_rel_l2_mu_eff','mean')]:.4f} |")
    L += ["", "WIND held-out rows are in-case spatial interpolation on the training flow case; the experiment is "
          "an external comparison on the same geometry and condition. No cross-case claim. The WIND solution "
          "itself scores 0.089 / 0.084 (upper/lower) on the experimental metric (P4.1).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
