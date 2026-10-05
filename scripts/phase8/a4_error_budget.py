#!/usr/bin/env python3
"""A4 error budget on the AE3 calibration engine (docs/phase8_p84c_registration.md section 5).

AE3 (Trent 1000-AE3, 02P23RR126) only, take-off/approach/idle. Cases:
  old_A4_central             trent_p84b.solve_engine with trent_p84b.CENTRAL (A4 itself),
                             compared with outputs/phase8/p84b_a1_g1.json (old files untouched)
  old_A4_central_new_builder A4c builder at the A4 inputs and the single A4 guess (regression)
  new_central_prior          A4c central prior with the ordered design starts
  one-at-a-time endpoints from the new central (registered list, incl. Cv core/bypass/joint)
Each case re-solves the design, freezes its geometry and solves its modes.

The 'share' is the change of AE3 calibration error (pp) divided by the known
A4 held-out aggregate of that mode (+15.4/+10.3/+8.3 %). It is ILLUSTRATIVE:
not an additive causal allocation of held-out errors, not claimed to sum to
100 %, never used to tune a central value. No held-out row is read.

    .venv/bin/python scripts/phase8/a4_error_budget.py [--workers 6]

Write-once output: outputs/phase8/a4_error_budget/20261003_attempt1/.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import trent_p84c as A  # noqa: E402

v5, IC = A.v5, A.IC
EB = A.REG["error_budget"]
OUT = ROOT / EB["output_dir"].split()[0].rstrip("/")
AE3 = "02P23RR126"
MODES = ("TAKE-OFF", "APPROACH", "IDLE")
# OAT name -> engine-input keys it sets (joint = both nozzles)
TARGETS = {"Cv_joint": ("Cv_core", "Cv_bypass"), "Cd_joint": ("Cd_core", "Cd_bypass")}


def oat_cases(reg: dict = A.REG) -> list[dict]:
    """Registered one-at-a-time endpoint cases, in registration order (low, high)."""
    cases = []
    for name, ends in reg["error_budget"]["oat_from_new_central"].items():
        keys = TARGETS.get(name, (name,))
        for end, v in zip(("low", "high"), ends):
            cases.append({"name": f"{name}={end}:{v:g}", "kind": "oat", "parameter": name, "end": end,
                          "value": float(v), "overrides": {k: float(v) for k in keys}})
    return cases


def all_cases() -> list[dict]:
    return ([{"name": "old_A4_central", "kind": "old_a4_p84b", "overrides": {}},
             {"name": "old_A4_central_new_builder", "kind": "old_a4_builder", "overrides": {}},
             {"name": "new_central_prior", "kind": "new_central", "overrides": {}}] + oat_cases())


def ae3_inputs(rows: pd.DataFrame) -> dict:
    g = rows[rows["Unique ID"] == AE3]
    if sorted(g["Mode"]) != sorted(MODES):
        raise RuntimeError("AE3 calibration rows incomplete")
    v = g[["Pressure Ratio", "Bypass Ratio", "Rated Thrust (kN)"]].drop_duplicates()
    if len(v) != 1:
        raise RuntimeError("AE3 design inputs differ across modes")
    return {"opr": float(v.iloc[0, 0]), "bpr": float(v.iloc[0, 1]), "rated_kN": float(v.iloc[0, 2]),
            "obs_ff": {m: float(g.loc[g["Mode"] == m, "Fuel Flow (kg/s)"].iloc[0]) for m in MODES}}


def solve_case(args) -> dict:
    case, ae3 = args
    A._init_worker()
    key = (ae3["opr"], ae3["bpr"], ae3["rated_kN"], list(MODES))
    if case["kind"] == "old_a4_p84b":
        B = A._p84b()
        return B.solve_engine((*key, dict(B.CENTRAL)))
    if case["kind"] == "old_a4_builder":
        return A.solve_engine((*key, A.old_a4_inputs(), True))
    return A.solve_engine((*key, A.engine_inputs(A.central(), case["overrides"]), False))


def error_pct(ff, obs) -> float:
    return 100.0 * (ff - obs) / obs if ff is not None and np.isfinite(ff) else float("nan")


def shares(delta_pp: dict, reg: dict = A.REG) -> dict:
    """Illustrative normalisation by the known A4 aggregates (not an additive allocation)."""
    agg = reg["known_A4_facts"]["A4_converged_rows_mean_signed_error_pct"]
    return {m: (delta_pp[m] / agg[m] if np.isfinite(delta_pp[m]) else float("nan")) for m in MODES}


def summarise(name: str, res: dict, ae3: dict, ref: dict | None) -> dict:
    d = res.get("design") or {}
    sc = d.get("scalars", {})
    row = {"case": name, "design_status": "converged" if d.get("converged") else "unreachable",
           "design_start_index": d.get("start_index"), "design_tries": len(res.get("design_tries", [])),
           "W_kg_s": sc.get("W"), "FAR": sc.get("FAR"), "A_core_m2": sc.get("A_core_m2"),
           "A_byp_m2": sc.get("A_byp_m2"), "design_iterations": d.get("iterations"),
           "design_residual_inf": max(map(abs, d["residuals"])) if d.get("residuals") else None,
           "design_closure": [d.get("mass_closure"), d.get("energy_closure"), d.get("element_closure")]}
    for m in MODES:
        r = res["modes"][m]
        e = error_pct(r.get("ff"), ae3["obs_ff"][m])
        row[f"{m}_status"] = r["status"]
        row[f"{m}_reason"] = r.get("reason", "")
        row[f"{m}_ff"] = r.get("ff")
        row[f"{m}_cal_error_pct"] = e
        row[f"{m}_extrapolated"] = r.get("extrapolated")
        row[f"{m}_closure"] = [r.get("mass_closure"), r.get("energy_closure"), r.get("element_closure")]
        row[f"{m}_residual_inf"] = max(map(abs, r["residuals"])) if r.get("residuals") else None
        if ref is not None:
            row[f"{m}_dff_vs_new_central"] = (r.get("ff") - ref[f"{m}_ff"]
                                              if r.get("ff") is not None and ref[f"{m}_ff"] is not None else None)
            row[f"{m}_dcal_error_pp"] = e - ref[f"{m}_cal_error_pct"]
    if ref is not None:
        row["share_illustrative"] = shares({m: row[f"{m}_dcal_error_pp"] for m in MODES})
    return row


def report_md(doc: dict) -> str:
    lines = [f"# A4 error budget, AE3 calibration engine ({doc['status']})", "",
             "Illustrative one-at-a-time diagnostic. 'Share' = change of AE3 calibration error (pp) / known "
             "A4 held-out aggregate (+15.4/+10.3/+8.3 %). Not an additive causal allocation; shares are not "
             "claimed to sum to 100 %; no central value is tuned by them; no held-out row was read.", "",
             f"Old A4 reproduction vs p84b_a1_g1.json: {doc.get('old_A4_reproduction')}", "",
             f"Builder regression (A4c builder at A4 inputs vs trent_p84b): {doc.get('builder_regression')}", "",
             "| case | TO err % | APP err % | IDLE err % | dTO pp | dAPP pp | dIDLE pp | share TO/APP/IDLE |",
             "|---|---|---|---|---|---|---|---|"]
    for r in doc.get("cases", []):
        def f(v):
            return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.2f}"
        sh = r.get("share_illustrative")
        lines.append(f"| {r['case']} | " + " | ".join(f(r[f'{m}_cal_error_pct']) for m in MODES) + " | "
                     + " | ".join(f(r.get(f'{m}_dcal_error_pp')) for m in MODES) + " | "
                     + ("/".join(f(sh[m]) for m in MODES) if sh else "") + " |")
    if doc.get("failures"):
        lines += ["", "Failures:"] + [f"- {x}" for x in doc["failures"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if OUT.exists():
        print(f"{OUT.relative_to(ROOT)} exists; write-once")
        return 2
    blockers = IC.live_gate(A.REG)
    if blockers:
        print("BLOCKED:\n- " + "\n- ".join(blockers))
        return 3
    IC.check_registered_hashes(A.REG)
    ae3 = ae3_inputs(A.read_calibration_rows())
    started, ident = IC.utc(), IC.identity(A.REG)
    OUT.mkdir(parents=True, exist_ok=False)
    log = OUT / "run.log"
    log.write_text(f"{started} START a4_error_budget head={ident['git_head']}\n")
    doc = {"stage": "A4 error budget (AE3 calibration engine, illustrative)",
           "registration": "docs/phase8_p84c_registration.json", "ae3": {"uid": AE3, **ae3},
           "start_identity": ident, "started_utc": started}
    try:
        cases = all_cases()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            results = list(pool.map(solve_case, [(c, ae3) for c in cases]))
        by = dict(zip([c["name"] for c in cases], results))
        ref = summarise("new_central_prior", by["new_central_prior"], ae3, None)
        rows = [summarise(c["name"], by[c["name"]], ae3, None if c["kind"] != "oat" else ref)
                for c in cases]
        old = json.loads((ROOT / "outputs/phase8/p84b_a1_g1.json").read_text())["ae3_central"]["modes"]
        doc["old_A4_reproduction"] = {m: {"recorded_ff": old[m].get("ff"),
                                          "reproduced_ff": by["old_A4_central"]["modes"][m].get("ff")}
                                      for m in MODES}
        doc["builder_regression"] = {m: {"p84b_ff": by["old_A4_central"]["modes"][m].get("ff"),
                                         "a4c_builder_ff": by["old_A4_central_new_builder"]["modes"][m].get("ff")}
                                     for m in MODES}
        doc["old_to_new_central_dcal_error_pp"] = {
            m: rows[2][f"{m}_cal_error_pct"] - rows[0][f"{m}_cal_error_pct"] for m in MODES}
        doc["old_to_new_central_share_illustrative"] = shares(doc["old_to_new_central_dcal_error_pp"])
        doc["cases"] = rows
        doc["raw_results"] = by
        doc["failures"] = [f"{r['case']}: {m} {r[f'{m}_reason']}" for r in rows for m in MODES
                           if r[f"{m}_status"] != "converged"]
        doc["status"] = "COMPLETE" if not doc["failures"] else "COMPLETE_WITH_FAILURES"
        rc = 0
    except Exception:
        doc["status"] = "ERROR"
        doc["error"] = traceback.format_exc()
        rc = 1
    end_ident, drift = IC.finish_identity(A.REG, ident)
    doc["finished_utc"], doc["end_identity"] = IC.utc(), end_ident
    if drift:
        doc["status"], doc["error"], rc = "ERROR", drift, 1
    doc["share_disclaimer"] = EB["share_diagnostic"]
    A._write_once(OUT / "error_budget.json", v5._json(doc))
    if doc.get("cases"):
        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in doc["cases"]]
                     ).to_csv(OUT / "cases.csv", index=False, mode="x")
    A._write_once(OUT / "report.md", report_md(doc))
    with log.open("a") as fh:
        fh.write(f"{doc['finished_utc']} EXIT {doc['status']} rc={rc}\n")
    print(f"A4 error budget: {doc['status']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
