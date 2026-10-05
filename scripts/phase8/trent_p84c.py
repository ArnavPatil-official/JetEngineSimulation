#!/usr/bin/env python3
"""Ladder A4c: shared-parameter Trent 1000 model (docs/phase8_p84c_registration.md).

A new prospective model, separate from A4. Physics is the P8.4b three-shaft
cycle with the P8.4b-A1 handling bleed: ``trent_p84b.make_engine`` and
``trent_p84b.offdesign_point`` are reused unchanged (trent_p84b.py is not
edited). A4c changes only the nozzle prior (P8.3-A2: Cd 0.96, Cv 0.985), the
five shared calibrated parameters, ordered input-only design starts and
C1 opt-in numerical design-flow box.
Every number comes from ``docs/phase8_p84c_registration.json``.

  calibrate --stage primary    TPE(42, 150) + polish, central and midpoint starts
  calibrate --stage fallback   only if the committed primary profile fails A1
  frozen                       print the frozen model from the committed records
  score                        the 87 Trent held-out rows ONCE, behind an atomic
                               reservation created before any target is read

Calibration reads only whitelisted columns of outputs/phase7/calibration_v6_rows.csv.
The C++ module is imported lazily (CATJET_BUILD or cpp/build), so the pure
functions and tests need no build. Heavy commands refuse without AC power,
while another heavy job runs, or when registration, sources or required
records are not committed and clean.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (ROOT, ROOT / "scripts" / "optimization", ROOT / "scripts" / "phase8"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lto_v5 as v5  # noqa: E402
import lto_v6  # noqa: E402
import p84_input_convergence as IC  # noqa: E402

REG_PATH = ROOT / "docs" / "phase8_p84c_registration.json"
A2_PATH = ROOT / "docs" / "phase8_p83_a2_registration.json"
REG = IC.load_registration(REG_PATH)
A2 = json.loads(A2_PATH.read_text())
SP = REG["shared_parameters"]
ORDER = tuple(SP["order"])
BOUNDS = {k: (float(SP[k]["bounds"][0]), float(SP[k]["bounds"][1])) for k in ORDER}
CENTRAL = {k: float(SP[k]["central"]) for k in ORDER}
CANDIDATE_ORDER = tuple(c["name"] for c in REG["fit"]["candidates"])
OUT_DIR = ROOT / "outputs" / "phase8" / "ladder" / "A4c"
CAL_CSV = ROOT / REG["calibration_data"]["file"]
SPLIT_JSON = ROOT / REG["calibration_data"]["split"]["file"]
INPUT_CONVERGENCE = "outputs/phase8/p84_input_convergence.json"
RESERVATION = "heldout_score_reservation.json"
SCORE_FILES = ("holdout_p8_A4c.csv", "holdout_p8_A4c_summary.csv", "holdout_p8_A4c.json")
OLD_A4_NOZZLE = {"Cd": 0.96, "Cv": 0.95}         # P8.3 priors in trent_p84b.make_engine (A4)
PYCYCLE_LPC_PR, PYCYCLE_HPC_PR = 1.935, 9.369   # pyCycle HBTF example (compare_hbtf.make_spec): generic LPC share


def central() -> dict:
    return dict(CENTRAL)


def midpoint() -> dict:
    return {k: 0.5 * (lo + hi) for k, (lo, hi) in BOUNDS.items()}


def check_shared(params: dict) -> None:
    if set(params) != set(ORDER):
        raise ValueError(f"shared parameters must be exactly {ORDER}, got {sorted(params)}")
    for k, v in params.items():
        lo, hi = BOUNDS[k]
        if not (math.isfinite(v) and lo <= v <= hi):
            raise ValueError(f"{k} = {v} outside the registered bounds [{lo}, {hi}]")


def engine_inputs(shared: dict, overrides: dict | None = None) -> dict:
    """Full engine input dictionary from the five shared parameters (and OAT overrides)."""
    check_shared(shared)
    fx = REG["model"]["fixed_inputs"]
    num, den = fx["ipc_share"].split()[0].split("/")
    eff, cool, noz = fx["efficiency_centers"], fx["cooling_cited"], fx["nozzle"]
    inp = {"T4_K": shared["T4_K"], "FPR": shared["FPR"], "ipc_share": float(num) / float(den),
           **{k: eff[k] + shared["eff_comp_offset"] for k in ("eff_fan", "eff_ipc", "eff_hpc")},
           **{k: eff[k] + shared["eff_turb_offset"] for k in ("eff_hpt", "eff_ipt", "eff_lpt")},
           "duct_scale": float(fx["duct_scale"]), "ram_recovery": float(fx["ram_recovery"]),
           "sm_floor": float(fx["sm_floor_pct"]),
           "ngv_cooling_frac": cool["ngv_frac_of_hpc_exit"] * shared["cooling_scale"],
           "rotor_cooling_frac": cool["rotor_frac_of_hpc_exit"] * shared["cooling_scale"],
           **{k: float(noz[k]) for k in ("Cd_core", "Cd_bypass", "Cv_core", "Cv_bypass")}}
    unknown = set(overrides or {}) - set(inp)
    if unknown:
        raise ValueError(f"unknown engine-input overrides {sorted(unknown)}")
    inp.update(overrides or {})
    return inp


def old_a4_inputs() -> dict:
    """The A4 central inputs expressed in the A4c builder (Cv 0.95, cited centrals)."""
    return engine_inputs(central(), {"Cd_core": OLD_A4_NOZZLE["Cd"], "Cd_bypass": OLD_A4_NOZZLE["Cd"],
                                     "Cv_core": OLD_A4_NOZZLE["Cv"], "Cv_bypass": OLD_A4_NOZZLE["Cv"]})


# ---------------------------------------------------------------------------
# Engine builders (lazy C++ import through the unchanged trent_p84b module)
# ---------------------------------------------------------------------------

def _p84b():
    import trent_p84b   # imports catjet_core from CATJET_BUILD or cpp/build
    return trent_p84b


def core_module_path() -> Path:
    return Path(_p84b().core.__file__)


def design_flow_bound(rated_kN: float) -> float:
    if not math.isfinite(rated_kN) or rated_kN <= 0:
        raise ValueError("rated thrust must be finite and positive")
    return max(3000 * 0.45359237, 4 * rated_kN * 1000 / 220)


def make_three_shaft(opr: float, bpr: float, rated_kN: float, inp: dict, *, use_flow_override: bool = True):
    """trent_p84b.make_engine unchanged, then the A4c nozzle and per-row cooling fractions."""
    B = _p84b()
    p = {k: inp[k] for k in B.CENTRAL if k != "cooling_scale"}
    p["cooling_scale"] = 1.0
    eng = B.make_engine(opr, bpr, rated_kN, p)
    s = eng.spec
    if use_flow_override:
        s.design_W_max_kg_s = design_flow_bound(rated_kN)
    s.Cd_core, s.Cd_byp = inp["Cd_core"], inp["Cd_bypass"]
    s.Cv_core, s.Cv_byp = inp["Cv_core"], inp["Cv_bypass"]
    s.cool3_frac_W, s.cool4_frac_W = inp["ngv_cooling_frac"], inp["rotor_cooling_frac"]
    eng.spec = s
    return eng


def make_two_shaft(opr: float, bpr: float, rated_kN: float, inp: dict):
    """Existing C++ two-shaft path (input-convergence check only; declared proxy limits:
    pyCycle Cv nozzle without Cd, no eta_b scaling, US 1976 table at 0 m). Geared
    families use it as a lossless-gearbox design-power proxy."""
    B = _p84b()
    core, C = B.core, B.C
    reg6 = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
    fuel = ", ".join(f"{k}:{v}" for k, v in reg6["fuel"]["mole_fractions"].items())
    eng = core.Hbtf(str(B.CRECK), "production", "O2:1, N2:3.76", fuel)
    s = core.HbtfSpec()
    s.alt_m, s.MN, s.dTs_K = 0.0, 0.0, 0.0
    s.Fn_des_N, s.T4_max_K = rated_kN * 1000.0, inp["T4_K"]
    s.design_W_max_kg_s = design_flow_bound(rated_kN)
    s.N_lp_des = s.N_hp_des = 1.0
    s.BPR_des, s.ram_recovery = bpr, inp["ram_recovery"]
    for k in ("dPqP_duct4", "dPqP_duct6", "dPqP_duct11", "dPqP_duct13", "dPqP_duct15"):
        setattr(s, k, B.DUCTS[k] * inp["duct_scale"])
    s.dPqP_burner = B.fixed_v6()["combustor_pressure_loss"]
    s.Cv_core, s.Cv_byp = inp["Cv_core"], inp["Cv_bypass"]
    s.frac_byp_bleed, s.HPX_W = 0.0, 0.0
    s.cool3_frac_W, s.cool4_frac_W = inp["ngv_cooling_frac"], inp["rotor_cooling_frac"]
    s.cool3_frac_P, s.cool4_frac_P, s.cool1_frac_P_lpt, s.cool2_frac_P_lpt = 1.0, 0.0, 1.0, 0.0
    core_pr = opr / (inp["FPR"] * (1 - s.dPqP_duct4) * (1 - s.dPqP_duct6))
    share = math.log(PYCYCLE_LPC_PR) / math.log(PYCYCLE_LPC_PR * PYCYCLE_HPC_PR)
    pr_lpc = math.exp(share * math.log(core_pr))
    hpc_bleeds = [C.bleed("cool1", 0.0, 0.5, 0.5), C.bleed("cool2", 0.0, 0.55, 0.5), C.bleed("cust", 0.0, 0.5, 0.5)]
    for attr, mapname, PR, eff, bleeds in (("fan", "FanMap", inp["FPR"], inp["eff_fan"], []),
                                           ("lpc", "LPCMap", pr_lpc, inp["eff_ipc"], []),
                                           ("hpc", "HPCMap", core_pr / pr_lpc, inp["eff_hpc"], hpc_bleeds)):
        c = core.CompressorSpec()
        c.name, c.map, c.PR_des, c.eff_des, c.bleeds = attr, C.load_map(mapname), PR, eff, bleeds
        setattr(s, attr, c)
    for attr, mapname, eff in (("hpt", "HPTMap", inp["eff_hpt"]), ("lpt", "LPTMap", inp["eff_lpt"])):
        t = core.TurbineSpec()
        t.name, t.map, t.eff_des = attr, C.load_map(mapname), eff
        setattr(s, attr, t)
    atm = json.loads(C.US1976.read_text())
    s.atm_alt_ft, s.atm_T_R, s.atm_P_psi = atm["alt_ft"], atm["T_degR"], atm["P_psi"]
    eng.spec = s
    return eng


def engine_for(opr: float, bpr: float, rated_kN: float, shafts: int, inp: dict):
    if shafts == 3:
        return make_three_shaft(opr, bpr, rated_kN, inp)
    if shafts == 2:
        return make_two_shaft(opr, bpr, rated_kN, inp)
    raise ValueError(f"unsupported shaft count {shafts}")


def solve_engine(args) -> dict:
    """Design (ordered starts) + the unchanged P8.4b continuation to each mode.
    args = (opr, bpr, rated_kN, modes, inp, single_start)."""
    opr, bpr, rated, modes, inp, single = args
    B = _p84b()
    eng = make_three_shaft(opr, bpr, rated, inp, use_flow_override=not single)
    out = {"design": None, "design_tries": [], "modes": {}}
    d, tries = IC.solve_design_ordered(eng, rated, 3, REG, single=single)
    out["design_tries"] = tries
    if d is None:
        why = "; ".join(t["reason"] or ", ".join(t["problems"]) for t in tries)
        for m in modes:
            out["modes"][m] = {"status": "unreachable", "reason": f"design point: no closure-valid start ({why})"}
        return out
    out["design"] = {k: d[k] for k in ("converged", "iterations", "reason", "x", "scalars", "residuals",
                                       "mass_closure", "energy_closure", "element_closure", "extrapolated_maps")}
    out["design"]["start_index"] = len(tries) - 1
    sx = d["x"]
    x = [sx[0], sx[1], bpr, 1.0, 1.0, 1.0,
         eng.spec.fan.map.defaults["RlineMap"], eng.spec.ipc.map.defaults["RlineMap"],
         eng.spec.hpc.map.defaults["RlineMap"], sx[2], sx[3], sx[4]]
    eta = B.fixed_v6()["eta_b"]
    targets = sorted(((v5.MODE_X[m], m) for m in modes), reverse=True)
    level, failed, beta = 1.0, None, 0.0
    for xt, mode in targets:
        spec = eng.spec
        spec.eta_b = eta[mode]
        eng.spec = spec
        path = []
        while level - B.STEP > xt + 1e-12:
            level = round(level - B.STEP, 10)
            path.append(level)
        path.append(xt)
        for lv in path:
            if failed:
                break
            r, beta = B.offdesign_point(eng, x, beta, lv * rated * 1000.0)
            if not r["converged"]:
                failed = f"off-design at {lv:.2f} rated: {r['reason']}"
                break
            x = r["x"][:12]
            level = lv
        if failed:
            out["modes"][mode] = {"status": "unreachable", "reason": failed}
            continue
        out["modes"][mode] = {"status": "converged", "reason": "", "ff": r["scalars"]["Wfuel"],
                              "scalars": r["scalars"], "x": r["x"], "residuals": r["residuals"],
                              "iterations": r["iterations"], "extrapolated": r["extrapolated_maps"],
                              "mass_closure": r["mass_closure"], "energy_closure": r["energy_closure"],
                              "element_closure": r["element_closure"]}
    return out


# ---------------------------------------------------------------------------
# Calibration rows (whitelisted columns of the frozen v6 calibration file only)
# ---------------------------------------------------------------------------

def read_calibration_rows(path: Path = CAL_CSV, split_path: Path = SPLIT_JSON,
                          check_hash: bool = True) -> pd.DataFrame:
    """The 93 frozen calibration rows. Never touches data/icao_engine_data.csv;
    old prediction columns are never parsed."""
    cd = REG["calibration_data"]
    raw, split_raw = Path(path).read_bytes(), Path(split_path).read_bytes()
    if check_hash and (hashlib.sha256(raw).hexdigest() != cd["sha256"]
                       or hashlib.sha256(split_raw).hexdigest() != cd["split"]["sha256"]):
        raise RuntimeError("calibration rows or split differ from the registered files")
    split = json.loads(split_raw)
    cols = cd["whitelist_columns"]
    df = pd.read_csv(io.BytesIO(raw), usecols=cols, float_precision="round_trip")[cols]
    ids = set(df["Unique ID"])
    gid = {m: i for i, g in enumerate(split["calibration_groups"]) for m in g}
    problems = []
    if len(df) != 93:
        problems.append(f"{len(df)} rows, not 93")
    if ids != set(split["calibration_records"]):
        problems.append("record set differs from split_p61 calibration_records")
    if ids & set(split["heldout_records"]) or set(df["Model"]) & set(split["heldout_models"]):
        problems.append("held-out record or model present")
    if not set(df["Model"]) <= set(split["calibration_models"]):
        problems.append("model outside the calibration models")
    if set(df["Group"]) != set(range(len(split["calibration_groups"]))) or len(split["calibration_groups"]) != 9:
        problems.append("groups differ from the 9 calibration groups")
    elif not (df["Model"].map(gid) == df["Group"]).all():
        problems.append("group labels differ from split_p61 calibration_groups")
    if abs(float(df["w"].sum()) - 1.0) > 1e-12:
        problems.append("weights do not sum to 1")
    if not (df.groupby("Unique ID")["Mode"].apply(lambda m: sorted(m) == sorted(v5.MODES))).all():
        problems.append("a record does not have exactly the three LTO modes")
    ff = df["Fuel Flow (kg/s)"].to_numpy(float)
    if not (np.isfinite(ff).all() and (ff > 0).all()):
        problems.append("non-finite or non-positive observed fuel flow")
    if problems:
        raise ValueError("calibration rows rejected: " + "; ".join(problems))
    return df


# ---------------------------------------------------------------------------
# Model, objective and the registered fit
# ---------------------------------------------------------------------------

def _init_worker():
    logging.getLogger("cantera").setLevel(logging.ERROR)
    os.chdir(ROOT)


class A4cModel:
    """predict(shared params, rows) -> status/reason/ff per row; parallel over record keys."""

    def __init__(self, n_workers: int = 6):
        self.pool = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
        self.n_evals = 0

    def close(self):
        self.pool.shutdown()

    def predict(self, params: dict, rows: pd.DataFrame) -> pd.DataFrame:
        IC.ensure_ac()
        inp = engine_inputs(params)
        keys = list(dict.fromkeys(zip(rows["Pressure Ratio"], rows["Bypass Ratio"], rows["Rated Thrust (kN)"])))
        modes = {k: sorted(set(rows.loc[(rows["Pressure Ratio"] == k[0]) & (rows["Bypass Ratio"] == k[1])
                                        & (rows["Rated Thrust (kN)"] == k[2]), "Mode"])) for k in keys}
        tasks = [(k[0], k[1], k[2], modes[k], inp, False) for k in keys]
        results = dict(zip(keys, self.pool.map(solve_engine, tasks)))
        self.n_evals += 1
        return predictions_frame(rows, results)


def predictions_frame(rows: pd.DataFrame, results: dict) -> pd.DataFrame:
    recs = []
    for _, r in rows.iterrows():
        res = results[(r["Pressure Ratio"], r["Bypass Ratio"], r["Rated Thrust (kN)"])]
        m = res["modes"][r["Mode"]]
        sc = m.get("scalars", {})
        recs.append({"status": m["status"], "reason": m["reason"], "ff": m.get("ff", np.nan),
                     "T3": sc.get("Tt3", np.nan), "T4": sc.get("Tt4", np.nan),
                     "thrust_kN": sc.get("Fn_N", np.nan) / 1000.0 if m["status"] == "converged" else np.nan,
                     "design_start_index": (res["design"] or {}).get("start_index"),
                     "design_tries": len(res["design_tries"])})
    return pd.DataFrame(recs, index=rows.index)


def box(free: list[str]):
    return np.array([BOUNDS[k][0] for k in free]), np.array([BOUNDS[k][1] for k in free])


def polish(obj, free: list[str], x0, max_nfev: int):
    """Registered local step: least_squares trf, bounds, x_scale = box widths, diff_step 1e-4."""
    from scipy.optimize import least_squares
    pol = REG["fit"]["polish"]
    lo, hi = box(free)
    x0 = np.clip(np.asarray(x0, float), lo, hi)
    return least_squares(lambda x: obj.residuals(free, x), x0, bounds=(lo, hi), method="trf",
                         x_scale=hi - lo, diff_step=pol["diff_step"], max_nfev=max_nfev)


def _cond(res, free) -> float:
    lo, hi = box(free)
    js = res.jac * (hi - lo)
    return float(np.linalg.cond(js.T @ js))


def tpe_then_polish(obj, free: list[str]) -> dict:
    """Candidate 1, as lto_v5.fit with the A4c box and order."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    fit = REG["fit"]
    n0 = len(obj.log)

    def objective(trial):
        x = [trial.suggest_float(k, *BOUNDS[k]) for k in free]
        r = obj.residuals(free, x)
        return float(r @ r)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=fit["tpe_seed"]))
    study.optimize(objective, n_trials=fit["n_trials"])
    x_best = [study.best_params[k] for k in free]
    res = polish(obj, free, x_best, fit["polish"]["max_nfev"])
    sse_pol = float(res.fun @ res.fun)
    use_pol = sse_pol <= study.best_value
    return {"name": "tpe_then_polish", "start": f"TPE({fit['n_trials']}, seed {fit['tpe_seed']}) best trial",
            "best_trial": {"params": study.best_params, "sse": study.best_value, "number": study.best_trial.number},
            "polish": {"status": int(res.status), "message": res.message, "nfev": int(res.nfev),
                       "sse": sse_pol, "accepted": bool(use_pol)},
            "params": obj.params(free, res.x if use_pol else np.array(x_best)),
            "sse": min(sse_pol, study.best_value), "jtj_condition_box_scaled": _cond(res, free),
            "n_evaluations": len(obj.log) - n0}


def polish_candidate(obj, free: list[str], name: str, start: dict) -> dict:
    n0 = len(obj.log)
    res = polish(obj, free, [start[k] for k in free], REG["fit"]["polish"]["max_nfev"])
    return {"name": name, "start": {k: start[k] for k in free}, "params": obj.params(free, res.x),
            "sse": float(res.fun @ res.fun),
            "polish": {"status": int(res.status), "message": res.message, "nfev": int(res.nfev)},
            "jtj_condition_box_scaled": _cond(res, free), "n_evaluations": len(obj.log) - n0}


def select_candidate(candidates: list[dict]) -> dict:
    """Lowest SSE; exact ties: TPE, then central, then midpoint."""
    return min(candidates, key=lambda c: (c["sse"], CANDIDATE_ORDER.index(c["name"])))


def run_candidates(obj, free: list[str]) -> list[dict]:
    free = [k for k in ORDER if k in free]
    return [tpe_then_polish(obj, free),
            polish_candidate(obj, free, "polish_from_central", central()),
            polish_candidate(obj, free, "polish_from_midpoint", midpoint())]


def fallback_plan(primary_profile: dict) -> dict:
    """Registered single fallback: fix every unidentified parameter at its cited central."""
    free = primary_profile["free"]
    if list(free) != list(ORDER):
        raise ValueError("the fallback plan is defined on the primary (all five parameters) profile")
    unident = [k for k in ORDER if not primary_profile["verdicts"][k]["IDENTIFIED"]]
    return {"triggered": bool(unident), "unidentified": unident,
            "fixed": {k: CENTRAL[k] for k in unident}, "remaining": [k for k in ORDER if k not in unident]}


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def _rel(name: str) -> str:
    return str((OUT_DIR / name).relative_to(ROOT))


def artifact_paths(kind: str, stage: str) -> list[str]:
    names = REG["pre_run_identity_validation"]["artifact_sets"][
        f"{kind}_{stage}" if stage == "primary" else
        ("fit_fallback_if_triggered" if kind == "fit" else "profile_fallback_if_free_parameters_remain")]
    return [_rel(n) for n in names]


def validate_record(rel: str, current: dict, dependencies: list[str]) -> dict:
    """Validate completed scientific identity, complete evidence and exact dependency hashes."""
    doc = _load(rel)
    if doc.get("status") != "COMPLETE":
        raise RuntimeError(f"{rel}: no COMPLETE scientific record")
    IC.assert_same_identity(doc["start_identity"], doc["end_identity"])
    if IC.scientific_identity(doc["start_identity"]) != IC.scientific_identity(current):
        raise RuntimeError(f"{rel}: scientific identity differs from current committed model")
    expected_deps = {p: IC.sha256(ROOT / p) for p in dependencies}
    if doc["start_identity"].get("dependencies_sha256") != expected_deps:
        raise RuntimeError(f"{rel}: dependency artifact identity differs")
    stage = "fallback" if "fallback" in Path(rel).name else "primary"
    kind = "profile" if Path(rel).name.startswith("profile") else "fit"
    paths = artifact_paths(kind, stage)
    want_artifacts = set(paths) - {rel}
    if set(doc.get("artifacts_sha256", {})) != want_artifacts:
        raise RuntimeError(f"{rel}: incomplete artifact hash set")
    for p, digest in doc["artifacts_sha256"].items():
        if not isinstance(digest, str) or IC.sha256(ROOT / p) != digest:
            raise RuntimeError(f"{rel}: evidence changed: {p}")
    if kind == "fit":
        check_shared(doc["params"])
        if doc["free"] != [k for k in ORDER if k not in doc["fixed"]]:
            raise RuntimeError("fit fixed/free partition invalid")
        if not math.isfinite(doc["sse"]) or doc["sse"] < 0 or doc["n_rows"] != 93:
            raise RuntimeError("fit SSE/row coverage invalid")
        evals = pd.read_csv(ROOT / paths[1], float_precision="round_trip")
        rows = pd.read_csv(ROOT / paths[2], float_precision="round_trip")
        if not set((*ORDER, "sse", "n_unreachable")) <= set(evals) or len(evals) != doc["n_evaluations"]:
            raise RuntimeError("fit evaluation CSV schema/coverage invalid")
        if len(rows) != 93 or not set(REG["calibration_data"]["whitelist_columns"] + ["status", "ff"]) <= set(rows):
            raise RuntimeError("fit calibration-row CSV schema/coverage invalid")
        # Validate against the frozen calibration whitelist; no empirical held-out reader.
        expected_rows = read_calibration_rows()
        cols = REG["calibration_data"]["whitelist_columns"]
        pd.testing.assert_frame_equal(rows[cols], expected_rows[cols], check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=0)
    else:
        expected_fit = _rel(f"fit_{stage}.json")
        if doc.get("fit") != expected_fit or expected_fit not in expected_deps:
            raise RuntimeError("profile does not identify its exact registered fit dependency")
        fit = _load(expected_fit)
        if doc["free"] != fit["free"] or doc["fixed"] != fit["fixed"]:
            raise RuntimeError("profile fixed/free partition differs from exact fit")
        table = pd.read_csv(ROOT / paths[1], float_precision="round_trip")
        needed = {"param", "i", "value", "sse", "D", "nfev", "status", "n_unreachable"}
        if not needed <= set(table) or len(table) != len(doc["free"]) * REG["profile"]["grid_points"]:
            raise RuntimeError("profile CSV schema/coverage invalid")
        counts = table["n_unreachable"].dropna().to_numpy(float)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts > 93).any() \
                or not np.equal(counts, np.floor(counts)).all():
            raise RuntimeError("profile unreachable counts must be known integers 0..93 or explicitly unknown")
        seen = []
        for name in doc["free"]:
            g = table[table["param"] == name].sort_values("i")
            grid = np.linspace(*BOUNDS[name], REG["profile"]["grid_points"])
            if list(g["i"]) != list(range(len(grid))) or not np.allclose(g["value"], grid, rtol=1e-14, atol=1e-14):
                raise RuntimeError("profile grid coverage differs from registration")
            seen.extend((name, int(i)) for i in g["i"])
        logs = (ROOT / paths[2]).read_text().splitlines()
        logkeys = []
        for line in logs:
            parts = line.split()
            if len(parts) < 2 or not parts[1].startswith("i="):
                raise RuntimeError("profile progress evidence malformed")
            logkeys.append((parts[0], int(parts[1][2:])))
        if sorted(logkeys) != sorted(seen):
            raise RuntimeError("profile progress coverage differs from table")
        import a4c_profile as P
        pr = REG["profile"]
        for key in ("n_eff", "threshold", "grid_points", "inner_max_nfev"):
            if doc.get(key) != pr[key]:
                raise RuntimeError(f"profile configuration differs from registration: {key}")
        calculated = P.profile_statistics(table, doc["free"], fit["sse"], BOUNDS,
                                          pr["n_eff"], pr["threshold"])
        canonical_table = calculated.pop("table")
        if not np.array_equal(table["D"].to_numpy(float), canonical_table["D"].to_numpy(float)):
            raise RuntimeError("profile D evidence differs from the registered SSE statistic")
        for key in ("sse_min", "sse_min_source", "n_eff"):
            if doc.get(key) != calculated[key]:
                raise RuntimeError(f"profile SSE summary differs from exact fit/table evidence: {key}")
        if doc.get("raw_verdicts") != calculated["verdicts"]:
            raise RuntimeError("profile unguarded verdict differs from its registered interval rule")
        reproduced = P.guarded(canonical_table, calculated["verdicts"], doc["free"])
        for key in ("verdicts", "identified", "not_identified", "failed_profiles", "A1"):
            if reproduced[key] != doc[key]:
                raise RuntimeError(f"profile guard/verdict evidence mismatch: {key}")
    return doc


def frozen_model() -> dict:
    """Full evidence validation precedes a frozen model and any scoring reservation."""
    current = IC.identity(REG)
    fp, pp = _rel("fit_primary.json"), _rel("profile_primary.json")
    prim = validate_record(fp, current, [])
    primary_fit_paths = artifact_paths("fit", "primary")
    prof = validate_record(pp, current, primary_fit_paths)
    req = primary_fit_paths + artifact_paths("profile", "primary")
    if prim["free"] != list(ORDER) or prim["fixed"]:
        raise RuntimeError("primary fit must use exactly the five shared parameters")
    if prof["A1"] == "PASS":
        return {"source": "primary", "params": prim["params"], "A1_primary": "PASS", "A1_fallback": None,
                "fitted": prim["free"], "fixed_at_central": {}, "required_records": req}
    plan = fallback_plan(prof)
    ffp = _rel("fit_fallback.json")
    fb = validate_record(ffp, current, req)
    if fb["fixed"] != plan["fixed"] or fb["free"] != plan["remaining"]:
        raise RuntimeError("fallback fit does not follow the registered fallback plan")
    req += artifact_paths("fit", "fallback")
    if fb["free"]:
        pfp = _rel("profile_fallback.json")
        a1_fb = validate_record(pfp, current, req)["A1"]
        req += artifact_paths("profile", "fallback")
    else:
        a1_fb = "n/a (all fixed at cited centrals; not a fitted identifiability success)"
    return {"source": "fallback", "params": fb["params"], "A1_primary": prof["A1"], "A1_fallback": a1_fb,
            "fitted": fb["free"], "fixed_at_central": plan["fixed"], "required_records": req}


def _write_once(path: Path, text: str) -> None:
    with open(path, "x") as fh:
        fh.write(text)


def calibrate(stage: str, n_workers: int) -> int:
    paths = {k: OUT_DIR / f"fit_{stage}{s}" for k, s in
             (("json", ".json"), ("evals", "_evaluations.csv"), ("rows", "_calibration_rows.csv"))}
    if any(p.exists() for p in paths.values()):
        print(f"A4c {stage} fit exists; write-once")
        return 2
    extra = []
    if stage == "fallback":
        extra = artifact_paths("fit", "primary") + artifact_paths("profile", "primary")
        current = IC.identity(REG)
        validate_record(_rel("fit_primary.json"), current, [])
        profile_record = validate_record(_rel("profile_primary.json"), current, artifact_paths("fit", "primary"))
        plan = fallback_plan(profile_record)
        if not plan["triggered"]:
            print("primary A1 PASS: the registered fallback is not triggered")
            return 2
        free, fixed = plan["remaining"], plan["fixed"]
    else:
        free, fixed = list(ORDER), {}
    blockers = IC.live_gate(REG, extra)
    if blockers:
        print("BLOCKED:\n- " + "\n- ".join(blockers))
        return 3
    IC.check_registered_hashes(REG)
    rows = read_calibration_rows()
    started, ident = IC.utc(), IC.identity(REG, extra)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IC.ensure_ac()
    model = A4cModel(n_workers)
    obj = v5.Objective(model, rows, fixed)
    try:
        if free:
            cands = run_candidates(obj, free)
            best = select_candidate(cands)
            params = best["params"]
        else:
            cands, best, params = [], None, dict(fixed)
        pred = model.predict(params, rows)
    except Exception as exc:
        end_ident, drift = IC.finish_identity(REG, ident, extra)
        pd.DataFrame(obj.log, columns=[*ORDER, "sse", "n_unreachable"]).to_csv(paths["evals"], index=False, mode="x")
        rows.to_csv(paths["rows"], index=False, mode="x")
        _write_once(paths["json"], v5._json({"status": "ERROR", "stage": f"A4c {stage} fit",
                    "error": f"{type(exc).__name__}: {exc}", "drift": drift,
                    "start_identity": ident, "end_identity": end_ident,
                    "started_utc": started, "finished_utc": IC.utc()}))
        return 1
    finally:
        model.close()
    r = v5.residual_vector(pred["ff"], rows, pred["status"])
    end_ident, drift = IC.finish_identity(REG, ident, extra)
    doc = {"status": "ERROR" if drift else "COMPLETE", "error": drift,
           "stage": f"A4c {stage} fit", "registration": "docs/phase8_p84c_registration.json",
           "free": [k for k in ORDER if k in free], "fixed": fixed, "bounds": BOUNDS,
           "selected": best["name"] if best else None, "params": params,
           "sse": best["sse"] if best else float(r @ r),
           "jtj_condition_box_scaled": best["jtj_condition_box_scaled"] if best else None,
           "candidates": cands, "n_evaluations": len(obj.log),
           "calibration_weighted_mape_pct_in_sample": v5.weighted_mape(pred["ff"], rows, pred["status"]),
           "n_rows": len(rows), "n_unreachable": int((pred["status"] == "unreachable").sum()),
           "unreachable_rows": rows.join(pred).loc[pred["status"] == "unreachable",
                                                   ["Unique ID", "Model", "Mode", "reason"]].to_dict("records"),
           "note": ("all parameters fixed at cited centrals (none remain); no refit" if not free
                    else "in-sample calibration fit on the frozen 93-row Trent calibration split; not validation"),
           "started_utc": started, "finished_utc": IC.utc(), "start_identity": ident,
           "end_identity": end_ident, "core_module_sha256": IC.sha256(core_module_path())}
    pd.DataFrame(obj.log, columns=[*ORDER, "sse", "n_unreachable"]).to_csv(paths["evals"], index=False, mode="x")
    rows.join(pred).to_csv(paths["rows"], index=False, mode="x")
    doc["artifacts_sha256"] = {str(p.relative_to(ROOT)): IC.sha256(p) for p in (paths["evals"], paths["rows"])}
    _write_once(paths["json"], v5._json(doc))
    print(f"A4c {stage}: SSE {doc['sse']:.6g}, in-sample MAPE {doc['calibration_weighted_mape_pct_in_sample']:.3f} %, "
          f"params {params}, unreachable {doc['n_unreachable']}")
    return 1 if drift else 0


# ---------------------------------------------------------------------------
# One-shot held-out score
# ---------------------------------------------------------------------------

def reserve(path: Path, doc: dict) -> None:
    """Atomic exclusive creation; raises FileExistsError if a reservation exists."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(v5._json(doc))
        fh.flush()
        os.fsync(fh.fileno())


def read_heldout_targets() -> pd.DataFrame:
    split = v5.load_split()
    return v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True), split["heldout_groups"])


def run_score(n_workers: int, *, out_dir: Path = OUT_DIR, gate=IC.live_gate, frozen=frozen_model,
              read_targets=read_heldout_targets, make_model=A4cModel) -> int:
    res_path = out_dir / RESERVATION
    if res_path.exists() or any((out_dir / n).exists() for n in SCORE_FILES):
        print("A4c held-out reservation or score exists; held-out rows are scored once")
        return 2
    try:
        fm = frozen()
    except Exception as exc:
        print(f"BLOCKED: frozen evidence invalid ({type(exc).__name__}: {exc})")
        return 3
    extra = fm["required_records"] + [INPUT_CONVERGENCE]
    blockers = gate(REG, extra)
    if blockers:
        print("BLOCKED:\n- " + "\n- ".join(blockers))
        return 3
    try:
        IC.check_registered_hashes(REG)
        check_shared(fm["params"])
        conv = _load(INPUT_CONVERGENCE)
        start_ident = IC.identity(REG, extra)
        IC.assert_same_identity(conv["start_identity"], conv["end_identity"])
        if IC.scientific_identity(conv["start_identity"]) != IC.scientific_identity(start_ident):
            raise RuntimeError("input-convergence scientific identity differs from frozen model")
        IC.validate_convergence_record(conv, REG)
        reservation = {"step": "A4c one-shot Trent held-out score", "created_utc": IC.utc(),
                       "frozen_model": fm, "input_convergence_verdict": conv.get("verdict"),
                       "identity": start_ident,
                       "note": "created before any held-out target is read; retained even if scoring fails"}
    except Exception as exc:
        print(f"BLOCKED: scoring evidence invalid ({type(exc).__name__}: {exc})")
        return 3
    try:
        reserve(res_path, reservation)
    except FileExistsError:
        print("A4c held-out reservation exists; held-out rows are scored once")
        return 2
    try:
        held = read_targets()
        cal = read_calibration_rows()
        reg6 = lto_v6.load_registration_v6()
        base = lto_v6.base_registration(reg6)
        model = make_model(n_workers)
        try:
            pred_cal = model.predict(fm["params"], cal)
            pred = model.predict(fm["params"], held)
        finally:
            model.close()
        df, summary, fields = v5.holdout_tables(base, cal, held, pred)
        per_mode = summary.set_index("Scope")
        diagnostic = fm["A1_primary"] != "PASS"
        end_ident, drift = IC.finish_identity(REG, start_ident, extra)
        if drift:
            raise RuntimeError(drift)
        result = {"status": "COMPLETE", "step": "A4c (P8.4c, shared parameters)", "registration": "docs/phase8_p84c_registration.json",
                  "frozen_model": fm, **fields,
                  "per_mode_group_weighted_mape_pct": {
                      sc: {t: float(per_mode.loc[sc, f"{t} group-weighted MAPE (%)"]) for t in ("Model", "B0", "B1")}
                      for sc in per_mode.index},
                  "A4_informativeness": lto_v6.a4_informativeness(df),
                  "calibration_rows_weighted_mape_pct_in_sample": v5.weighted_mape(pred_cal["ff"], cal,
                                                                                   pred_cal["status"]),
                  "n_unreachable_heldout": int((pred["status"] == "unreachable").sum()),
                  "n_unreachable_calibration": int((pred_cal["status"] == "unreachable").sum()),
                  "input_convergence_verdict": conv.get("verdict"),
                  "references_pct": REG["known_A4_facts"]["references_pct"],
                  "A4_reference_pct": REG["known_A4_facts"]["A4_heldout_group_weighted_mape_pct"],
                  "diagnostic_only": diagnostic,
                  "G2": "closed" if diagnostic else "not opened by this score alone",
                  "reservation": str(res_path.relative_to(ROOT)) if res_path.is_relative_to(ROOT) else str(res_path),
                  "end_identity": end_ident,
                  "note": "scored once behind the reservation; no parameter was chosen with held-out data"}
        df.to_csv(out_dir / SCORE_FILES[0], index=False, mode="x")
        summary.to_csv(out_dir / SCORE_FILES[1], index=False, mode="x")
        _write_once(out_dir / SCORE_FILES[2], v5._json(result))
    except Exception as exc:
        traceback.print_exc()
        error_path = out_dir / SCORE_FILES[2]
        if not error_path.exists():
            _write_once(error_path, v5._json({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                                            "reservation": str(res_path), "finished_utc": IC.utc()}))
        print("A4c held-out score FAILED after the reservation; the reservation is retained")
        return 1
    print(f"A4c held-out: {fields['primary_group_weighted_mape_pct']['model']:.3f} % "
          f"(A4 21.24, v6 1.830); unreachable {result['n_unreachable_heldout']}; "
          f"{'diagnostic only (A1 FAIL)' if diagnostic else 'A1 PASS'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=("calibrate", "frozen", "score", "build-provenance"))
    ap.add_argument("--stage", choices=("primary", "fallback"), default="primary")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if a.phase == "build-provenance":
        IC.ensure_ac()
        context = IC.main_context()
        IC.ensure_build_provenance(context, core_module_path(), export=True)
        print("verified build provenance exported; commit it before standalone numerical work")
        return 0
    if a.phase == "calibrate":
        return calibrate(a.stage, a.workers)
    if a.phase == "frozen":
        print(json.dumps(frozen_model(), indent=1))
        return 0
    return run_score(a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
