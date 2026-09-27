"""
Phase 7 calibration v6: the thrust-matched LTO formulation of v5
(``lto_v5.py``) with the fuel replaced by the registered Dooley et al. (2012)
Jet A surrogate. Registered in ``outputs/phase7/p72_registration.json``
(``docs/phase7_registration.md`` section P7.2) before any v6 computation.

Only the fuel changes. The split, fitted set and box, objective, weighting,
fixed central values (including the per-mode eta_b proxy with its n-dodecane
heating value), optimizer settings and budgets, identifiability rule,
baselines, metrics and thresholds are the Phase 6 ones (amendment A1) and are
read from the frozen Phase 6 registration, whose hash is checked.

Phase number 7 and calibration version 6 are distinct: every artifact here is
named ``*_v6`` and lives under an explicit output directory (default
``outputs/phase7``). Nothing in this module writes to a v5 path; files are
created write-once.

Entry points (``scripts/phase7_supervisor.py`` runs them in a frozen source
snapshot):
    python scripts/optimization/lto_v6.py calibrate [--out-dir DIR]
    python scripts/optimization/lto_v6.py profile   [--out-dir DIR]
    python scripts/optimization/lto_v6.py holdout   [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))

import lto_v5 as v5  # noqa: E402

REGISTRATION_V6 = ROOT / "outputs" / "phase7" / "p72_registration.json"
DEFAULT_OUT = ROOT / "outputs" / "phase7"

# write-once artifact names inside the output directory
FIT_JSON = "calibration_v6.json"
FIT_EVALS = "calibration_v6_evaluations.csv"
FIT_ROWS = "calibration_v6_rows.csv"
PROFILE_STEM = "identifiability_profile_v6"
HOLDOUT_CSV = "holdout_icao_validation_v6.csv"
HOLDOUT_SUMMARY = "holdout_icao_validation_summary_v6.csv"
HOLDOUT_JSON = "holdout_icao_validation_v6.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Registration and fuel
# --------------------------------------------------------------------------
def load_registration_v6(path: Path = REGISTRATION_V6) -> dict:
    """The P7.2 registration, with the hashes of every frozen input verified."""
    reg = json.loads(Path(path).read_text())
    if reg.get("phase") != "P7.2" or reg.get("calibration_version") != 6:
        raise ValueError(f"{path} is not the P7.2 (calibration v6) registration")
    for rel, digest in reg["inputs_sha256"].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(f"registered input {rel} changed since registration")
    return reg


def base_registration(reg6: dict) -> dict:
    """The Phase 6 A1 registration the v6 procedure reuses unchanged."""
    base = v5.load_registration()
    if base["fixed_central"] != reg6["fixed_central"]:
        raise RuntimeError("v6 fixed central values differ from the Phase 6 registration")
    return base


def fuel_composition(reg6: dict) -> dict:
    """Mole fractions of the registered production Jet A surrogate (frozen YAML)."""
    from simulation import fuels_v7
    name = reg6["fuel"]["surrogate"]
    x = fuels_v7.surrogate(name)
    if set(x) != set(reg6["fuel"]["mole_fractions"]):
        raise RuntimeError("fuel species differ from the registration")
    for k, v in reg6["fuel"]["mole_fractions"].items():
        if abs(x[k] - v) > 1e-12:
            raise RuntimeError(f"fuel mole fraction {k} differs from the registration")
    return x


def make_model(reg6: dict, split: dict, n_workers: int) -> v5.V5Model:
    """V5Model with the v6 fuel; worker NOx refit without the held-out models."""
    return v5.V5Model(reg6["fixed_central"], nox_fit_exclude_models=split["heldout_models"],
                      n_workers=n_workers, fuel=fuel_composition(reg6))


def _out(out_dir: Path, name: str) -> Path:
    p = Path(out_dir) / name
    if p.exists():
        raise SystemExit(f"{p} exists; refusing to overwrite")
    return p


def _refuse_v5_paths(out_dir: Path) -> None:
    """Output isolation: a v6 output directory can never be a v5/Phase 6 location."""
    out_dir = Path(out_dir).resolve()
    forbidden = {(ROOT / "outputs").resolve(), (ROOT / "outputs" / "phase6").resolve(),
                 (ROOT / "outputs" / "results").resolve()}
    if out_dir in forbidden:
        raise SystemExit(f"{out_dir} holds v5/Phase 6 evidence; choose a Phase 7 directory")


# --------------------------------------------------------------------------
# Calibration: two registered candidates, lower SSE kept (ties -> candidate 1)
# --------------------------------------------------------------------------
def select_candidate(candidates: list[dict]) -> dict:
    """Lowest calibration SSE; exact ties go to the earlier (TPE) candidate."""
    return v5.select_optimum(candidates)


def run_calibration(out_dir: Path = DEFAULT_OUT, n_workers: int = 6) -> dict:
    _refuse_v5_paths(out_dir)
    reg6 = load_registration_v6()
    base = base_registration(reg6)
    split = v5.load_split()
    out_json, out_evals, out_rows = (_out(out_dir, n) for n in (FIT_JSON, FIT_EVALS, FIT_ROWS))
    budget = reg6["fit"]
    free = [k for k in v5.FIT_ORDER if k in budget["free"]]
    v5_fit = json.loads((ROOT / reg6["v5_selected_calibration"]).read_text())
    rows = v5.calibration_rows(split)
    model = make_model(reg6, split, n_workers)
    obj = v5.Objective(model, rows)
    t0 = dt.datetime.now()
    try:
        c1 = v5.fit(obj, free, budget["n_trials"], budget["polish_max_nfev"], seed=budget["tpe_seed"])
        n1 = len(obj.log)
        res2 = v5.polish(obj, free, [v5_fit["params"][k] for k in free], budget["polish_max_nfev"])
        lo, hi = v5._box(free)
        js = res2.jac * (hi - lo)
        candidates = [
            {"name": "tpe_then_polish", "start": f"TPE({budget['n_trials']}, seed {budget['tpe_seed']}) best trial",
             "best_trial": c1["best_trial"], "params": c1["params"], "sse": c1["sse"],
             "polish": c1["polish"], "jtj_condition_box_scaled": c1["jtj_condition_box_scaled"],
             "n_evaluations": n1},
            {"name": "polish_from_v5_A2", "start": f"v5 A2 optimum ({reg6['v5_selected_calibration']})",
             "params": obj.params(free, res2.x), "sse": float(res2.fun @ res2.fun),
             "polish": {"status": int(res2.status), "message": res2.message, "nfev": int(res2.nfev)},
             "jtj_condition_box_scaled": float(np.linalg.cond(js.T @ js)),
             "n_evaluations": len(obj.log) - n1},
        ]
        best = select_candidate(candidates)
        pred = model.predict(best["params"], rows)
    finally:
        model.close()
    rows_out = rows.drop(columns=["CO (g/kg)", "HC (g/kg)", "NOx (g/kg)"]).join(pred)
    out = {
        "stage": "v6 calibration (P7.2)", "registration": str(REGISTRATION_V6.relative_to(ROOT)),
        "calibration_version": 6, "fuel": reg6["fuel"], "free": free,
        "selected": best["name"], "params": best["params"], "sse": best["sse"],
        "jtj_condition_box_scaled": best["jtj_condition_box_scaled"], "candidates": candidates,
        "started": t0.isoformat(timespec="seconds"),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "fixed_central": reg6["fixed_central"], "fit_bounds": v5.FIT_BOUNDS,
        "calibration_weighted_mape_pct": v5.weighted_mape(pred["ff"], rows, pred["status"]),
        "n_rows": len(rows), "n_unreachable": int((pred["status"] == "unreachable").sum()),
        "unreachable_rows": rows_out.loc[pred["status"] == "unreachable",
                                         ["Unique ID", "Model", "Mode", "Target Thrust (kN)", "reason"]
                                         ].to_dict("records"),
        "calibration_airflow": {"W_ref_rated_core_airflow_at_F_REF_kg_s": best["params"].get("W_ref"),
                                "F_REF_kN": v5.F_REF_KN},
        "base_registration_fit": base["fit"]["full"],
        "note": "in-sample calibration fit (calibration group only); not validation",
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    v5._write_new(out_json, v5._json(out))
    pd.DataFrame(obj.log).to_csv(out_evals, index=False)
    rows_out.to_csv(out_rows, index=False)
    return out


# --------------------------------------------------------------------------
# Identifiability profile (A1): existing rule + registered penalty guard
# --------------------------------------------------------------------------
def _lookup_unreachable(log: list[dict], row: dict, free: list[str]) -> int | None:
    """n_unreachable of the evaluation at a profile point's returned optimum
    (least_squares returns an evaluated iterate; searched newest first)."""
    target = {row["param"]: row["value"], **{k: row[f"fit_{k}"] for k in free if k != row["param"]}}
    for e in reversed(log):
        if all(e.get(k) == v for k, v in target.items()):
            return int(e["n_unreachable"])
    return None


def verdict_points(v: np.ndarray, d: np.ndarray) -> list[int]:
    """Grid indices on which the registered verdict depends: both box edges and
    the grid points that bracket the 95 % interval ends (or the grid minimum's
    neighbours when no grid point lies inside the interval)."""
    n = len(v)
    idx = {0, n - 1}
    inside = np.where(d < v5.CHI2_1_95)[0]
    if len(inside):
        i0, i1 = int(inside.min()), int(inside.max())
        idx |= {i0, max(i0 - 1, 0), i1, min(i1 + 1, n - 1)}
    else:
        j = int(np.argmin(d))
        idx |= {max(j - 1, 0), j, min(j + 1, n - 1)}
    return sorted(idx)


def apply_penalty_guard(table: pd.DataFrame, verdicts: dict, max_nfev: int) -> dict:
    """Registered guard (P7.2): a parameter is IDENTIFIED only if the existing
    rule holds AND no verdict-deciding grid point carries a penalised
    (unreachable) calibration row or an unknown unreachable count. Inner fits
    stopped at the evaluation cap are reported (not gating), as in Phase 6."""
    out = {}
    for name, vd in verdicts.items():
        g = table[table["param"] == name].sort_values("value").reset_index(drop=True)
        pts = verdict_points(g["value"].to_numpy(), g["D"].to_numpy())
        sub = g.loc[pts]
        penalised = sub[(sub["n_unreachable"].isna()) | (sub["n_unreachable"] > 0)]
        capped = g[(g["status"] == 0) | (g["nfev"] >= max_nfev)]
        rule = bool(vd["IDENTIFIED"])
        out[name] = {
            **vd,
            "IDENTIFIED_existing_rule": rule,
            "verdict_grid_values": [float(x) for x in sub["value"]],
            "verdict_points_penalised": [float(x) for x in penalised["value"]],
            "IDENTIFIED": bool(rule and penalised.empty),
            "penalty_dependent": bool(rule and not penalised.empty),
            "n_grid_points_with_unreachable_rows": int((g["n_unreachable"].fillna(1) > 0).sum()),
            "n_inner_fits_at_cap": int(len(capped)),
            "inner_fits_at_cap_values": [float(x) for x in capped["value"]],
            "verdict_points_at_cap": [float(x) for x in sub[(sub["status"] == 0)
                                                            | (sub["nfev"] >= max_nfev)]["value"]],
            "grid_step": float(g["value"].diff().median()),
        }
    return out


def run_profile(out_dir: Path = DEFAULT_OUT, n_workers: int = 6) -> dict:
    _refuse_v5_paths(out_dir)
    reg6 = load_registration_v6()
    base_registration(reg6)
    split = v5.load_split()
    stem = Path(out_dir) / PROFILE_STEM
    for p in (stem.with_suffix(".json"), stem.with_suffix(".csv")):
        _out(out_dir, p.name)
    opt = json.loads((Path(out_dir) / FIT_JSON).read_text())
    free = opt["free"]
    ident = reg6["identifiability"]
    rows = v5.calibration_rows(split)
    model = make_model(reg6, split, n_workers)
    obj = v5.Objective(model, rows)
    log_path = Path(f"{stem}_progress.log")
    extra: dict = {}

    def progress(r):
        nu = _lookup_unreachable(obj.log, r, free)
        if nu is None:      # not found in the log: evaluate the returned point once more
            p = {r["param"]: r["value"], **{k: r[f"fit_{k}"] for k in free if k != r["param"]}}
            nu = int((model.predict(p, rows)["status"] == "unreachable").sum())
        extra[(r["param"], r["i"])] = nu
        with open(log_path, "a") as fh:
            fh.write(f"{r['param']} i={r['i']} value={r['value']:.6g} sse={r['sse']:.6g} "
                     f"nfev={r['nfev']} status={r['status']} n_unreachable={nu}\n")
    t0 = dt.datetime.now()
    try:
        prof = v5.profile(obj, free, opt, ident["grid_points"], ident["inner_max_nfev"],
                          ident["n_eff"], progress=progress)
    finally:
        model.close()
    table = prof.pop("table")
    table["n_unreachable"] = [extra.get((p, i)) for p, i in zip(table["param"], table["i"])]
    verdicts = apply_penalty_guard(table, prof["verdicts"], ident["inner_max_nfev"])
    identified = [k for k in free if verdicts[k]["IDENTIFIED"]]
    prof.update(
        verdicts=verdicts, stage="v6 full profile (P7.2 A1)", fit=FIT_JSON, free=free,
        rule=ident["rule"], penalty_guard=ident["penalty_guard"], threshold=v5.CHI2_1_95,
        grid_points=ident["grid_points"], inner_max_nfev=ident["inner_max_nfev"],
        identified=identified, not_identified=[k for k in free if k not in identified],
        A1="PASS" if len(identified) == len(free) else "FAIL",
        n_inner_fits=int(len(table)), n_inner_fits_at_cap=int(((table["status"] == 0)
                                                               | (table["nfev"] >= ident["inner_max_nfev"])).sum()),
        n_points_with_unreachable_rows=int((table["n_unreachable"].fillna(1) > 0).sum()),
        jtj_condition_box_scaled_at_fit=opt["jtj_condition_box_scaled"],
        started=t0.isoformat(timespec="seconds"),
        finished=dt.datetime.now().isoformat(timespec="seconds"),
    )
    table.to_csv(stem.with_suffix(".csv"), index=False)
    v5._write_new(stem.with_suffix(".json"), v5._json(prof))
    return prof


# --------------------------------------------------------------------------
# Held-out test (A2-A4), unchanged definitions
# --------------------------------------------------------------------------
def a4_informativeness(df: pd.DataFrame) -> dict:
    """The assertions of tests/test_holdout_informativeness.py::
    test_v5_holdout_prediction_depends_on_the_cycle, as a function of the table."""
    checks = {}
    cols = ("Model", "Mode", "OPR", "BPR", "Rated Thrust (kN)", "Target Thrust (kN)",
            "Predicted Fuel Flow (kg/s)", "B0 Fuel Flow (kg/s)", "B1 Fuel Flow (kg/s)")
    checks["columns_present"] = all(c in df.columns for c in cols)
    checks["every_row_predicted"] = bool(df["Predicted Fuel Flow (kg/s)"].notna().all())
    tsfc = df["Predicted Fuel Flow (kg/s)"] / df["Target Thrust (kN)"]
    checks["tsfc_varies_per_mode"] = bool(all(g.std() / g.mean() > 1e-4 for _, g in tsfc.groupby(df["Mode"])))
    n_pairs, ok = 0, True
    for _, g in df.assign(tsfc=tsfc).groupby(["Mode", "Rated Thrust (kN)"]):
        if g["OPR"].nunique() > 1:
            n_pairs += 1
            ok &= bool(g["tsfc"].nunique() > 1)
    checks["equal_thrust_different_opr_differ"] = bool(ok and n_pairs > 0)
    checks["n_equal_thrust_groups"] = n_pairs
    return {"checks": checks,
            "verdict": "PASS" if all(v for k, v in checks.items() if k != "n_equal_thrust_groups") else "FAIL"}


def run_holdout(out_dir: Path = DEFAULT_OUT, n_workers: int = 6) -> dict:
    _refuse_v5_paths(out_dir)
    reg6 = load_registration_v6()
    base = base_registration(reg6)
    split = v5.load_split()
    paths = [_out(out_dir, n) for n in (HOLDOUT_CSV, HOLDOUT_SUMMARY, HOLDOUT_JSON)]
    fitted = json.loads((Path(out_dir) / FIT_JSON).read_text())
    prof = json.loads((Path(out_dir) / f"{PROFILE_STEM}.json").read_text())
    cal = v5.calibration_rows(split)
    held = v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True),
                            split["heldout_groups"])
    model = make_model(reg6, split, n_workers)
    try:
        pred = model.predict(fitted["params"], held)
    finally:
        model.close()
    df, summary, fields = v5.holdout_tables(base, cal, held, pred)
    a4 = a4_informativeness(df)
    v5_hold = json.loads(v5.HOLDOUT_JSON.read_text())
    v5_fit = json.loads((ROOT / reg6["v5_selected_calibration"]).read_text())
    per_mode = summary.set_index("Scope")
    result = {
        "registration": str(REGISTRATION_V6.relative_to(ROOT)),
        "calibration": FIT_JSON, "calibration_version": 6, "fuel": reg6["fuel"]["surrogate"],
        "fitted_params": fitted["params"],
        "calibration_airflow": {"W_ref_kg_s": fitted["params"].get("W_ref"),
                                "v5_W_ref_kg_s": v5_fit["params"].get("W_ref"),
                                "handset_v4_kg_s": 79.9},
        **fields,
        "per_mode_group_weighted_mape_pct": {
            s: {t: float(per_mode.loc[s, f"{t} group-weighted MAPE (%)"]) for t in ("Model", "B0", "B1")}
            for s in per_mode.index},
        "A1_profile": prof["A1"],
        "A4": a4,
        "v5_to_v6": {
            "params_v5": v5_fit["params"], "params_v6": fitted["params"],
            "calibration_sse_v5": v5_fit["sse"], "calibration_sse_v6": fitted["sse"],
            "calibration_mape_v5_pct": v5_fit["calibration_weighted_mape_pct"],
            "calibration_mape_v6_pct": fitted["calibration_weighted_mape_pct"],
            "heldout_primary_v5": v5_hold["primary_group_weighted_mape_pct"],
            "heldout_primary_v6": fields["primary_group_weighted_mape_pct"],
            "A2_v5": v5_hold["A2"]["verdict"], "A2_v6": fields["A2"]["verdict"],
            "A3_v5": v5_hold["A3"]["verdict"], "A3_v6": fields["A3"]["verdict"],
        },
        "nox_note": "model_nox_corr_g_s from a NOx correlation refit without held-out models "
                    "(worker exclusion asserted at initialisation)",
        "blend_gate": blend_gate(prof["A1"], fields["A2"]["verdict"]),
    }
    df.to_csv(paths[0], index=False)
    summary.to_csv(paths[1], index=False)
    v5._write_new(paths[2], v5._json(result))
    return result


def blend_gate(a1: str, a2_verdict: str) -> dict:
    """Registered dependency: P7.3 production blends run only if A1 passes and
    the existing B0 escalation is not triggered."""
    esc = a2_verdict.startswith("ESCALATE")
    ok = (a1 == "PASS") and not esc
    reason = [] if ok else ([f"A1 {a1}"] if a1 != "PASS" else []) + (["A2 ESCALATE"] if esc else [])
    return {"open": bool(ok), "reason": "; ".join(reason) or "A1 PASS and no B0 escalation"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["calibrate", "profile", "holdout"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    fn = {"calibrate": run_calibration, "profile": run_profile, "holdout": run_holdout}[a.step]
    res = fn(Path(a.out_dir), a.workers)
    keep = ("selected", "params", "sse", "calibration_weighted_mape_pct", "n_unreachable",
            "A1", "identified", "not_identified", "primary_group_weighted_mape_pct", "A2", "A3", "A4",
            "blend_gate")
    print(json.dumps({k: res[k] for k in keep if k in res}, indent=2, default=str))


if __name__ == "__main__":
    main()
