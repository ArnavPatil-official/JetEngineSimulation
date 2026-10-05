#!/usr/bin/env python3
"""P8.4b Trent 1000 three-shaft knob-free model = ladder A4 (docs/phase8_p84b_registration.md).

  g1     AE3 checks (closure, x = 1.00 reproduces the design point, two-shaft
         regression, one-at-a-time sensitivities). Writes outputs/phase8/p84b_g1.json once.
  score  Predicts all 93 calibration rows (in-sample, no fit) and scores the 87
         Trent held-out rows ONCE with the frozen P7.2 tables. Refuses unless
         p84b_g1.json is committed with verdict PASS.
  smoke  Development: AE3 only, printed, nothing written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(os.environ.get("CATJET_BUILD", ROOT / "cpp" / "build"))))
for p in (ROOT, ROOT / "scripts" / "optimization", ROOT / "scripts" / "phase8", ROOT / "scripts" / "phase8" / "pycycle"):
    sys.path.insert(0, str(p))
import catjet_core as core  # noqa: E402
import lto_v5 as v5  # noqa: E402
import lto_v6  # noqa: E402
import compare_hbtf as C  # noqa: E402

AE3 = "02P23RR126"
CRECK = ROOT / "data" / "creck_c1c16_full.yaml"
OUT_DIR = ROOT / "outputs" / "phase8" / "ladder" / "A4"
G1_PATH = ROOT / "outputs" / "phase8" / "p84b_a1_g1.json"   # P8.4b-A1 re-run (p84b_g1.json: failed original)

# registered inputs (section 3); keys are also the sensitivity knobs
CENTRAL = {"T4_K": 1800.0, "FPR": 1.45, "ipc_share": 8.0 / 14.0,
           "eff_fan": 0.8948, "eff_ipc": 0.9243, "eff_hpc": 0.8707,
           "eff_hpt": 0.8888, "eff_ipt": 0.8888, "eff_lpt": 0.8996,
           "duct_scale": 1.0, "cooling_scale": 1.0, "ram_recovery": 0.999, "sm_floor": 10.0}
RANGES = {"T4_K": (1700.0, 1900.0), "FPR": (1.40, 1.55),
          "ipc_share": (0.8 * 8 / 14, min(1.2 * 8 / 14, 0.95)),
          "eff_fan": (0.8748, 0.9148), "eff_ipc": (0.9043, 0.9443), "eff_hpc": (0.8507, 0.8907),
          "eff_hpt": (0.8688, 0.9088), "eff_ipt": (0.8688, 0.9088), "eff_lpt": (0.8796, 0.9196),
          "duct_scale": (0.0, 2.0), "cooling_scale": (0.0, 1.0), "ram_recovery": (0.99, 1.0),
          "sm_floor": (5.0, 15.0)}
DUCTS = {"dPqP_duct4": 0.0048, "dPqP_duct6": 0.0101, "dPqP_duct11": 0.0051,
         "dPqP_duct_ipt_lpt": 0.0051, "dPqP_duct13": 0.0107, "dPqP_duct15": 0.0149}
COOLING = {"cool3_frac_W": 0.0641, "cool4_frac_W": 0.0275}
STEP = 0.10


def fixed_v6() -> dict:
    return json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())["fixed_central"]


def make_engine(opr: float, bpr: float, rated_kN: float, p: dict):
    reg = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
    fuel = ", ".join(f"{k}:{v}" for k, v in reg["fuel"]["mole_fractions"].items())
    eng = core.Hbtf(str(CRECK), "production", "O2:1, N2:3.76", fuel)
    s = core.HbtfSpec()
    s.three_shaft, s.nozzle_p83 = True, True
    s.alt_m, s.MN, s.dTs_K = 0.0, 0.0, 0.0
    s.Ts_override_K, s.Ps_override_Pa = 288.15, 101325.0
    s.Fn_des_N, s.T4_max_K = rated_kN * 1000.0, p["T4_K"]
    s.N_lp_des = s.N_ip_des = s.N_hp_des = 1.0          # normalisation constants
    s.BPR_des, s.ram_recovery = bpr, p["ram_recovery"]
    for k, v in DUCTS.items():
        setattr(s, k, v * p["duct_scale"])
    s.dPqP_burner = fixed_v6()["combustor_pressure_loss"]
    s.Cd_core = s.Cd_byp = 0.96
    s.Cv_core = s.Cv_byp = 0.95
    s.frac_byp_bleed, s.HPX_W = 0.0, 0.0
    s.cool3_frac_W = COOLING["cool3_frac_W"] * p["cooling_scale"]
    s.cool4_frac_W = COOLING["cool4_frac_W"] * p["cooling_scale"]
    s.cool3_frac_P, s.cool4_frac_P = 1.0, 0.0
    core_pr = opr / (p["FPR"] * (1 - s.dPqP_duct4) * (1 - s.dPqP_duct6))
    pr_ipc = math.exp(p["ipc_share"] * math.log(core_pr))
    pr_hpc = core_pr / pr_ipc
    for attr, mapname, PR, eff in (("fan", "FanMap", p["FPR"], p["eff_fan"]),
                                   ("ipc", "LPCMap", pr_ipc, p["eff_ipc"]),
                                   ("hpc", "HPCMap", pr_hpc, p["eff_hpc"])):
        c = core.CompressorSpec()
        c.name, c.map, c.PR_des, c.eff_des, c.bleeds = attr, C.load_map(mapname), PR, eff, []
        setattr(s, attr, c)
    for attr, mapname, eff in (("hpt", "HPTMap", p["eff_hpt"]), ("ipt", "HPTMap", p["eff_ipt"]),
                               ("lpt", "LPTMap", p["eff_lpt"])):
        t = core.TurbineSpec()
        t.name, t.map, t.eff_des = attr, C.load_map(mapname), eff
        setattr(s, attr, t)
    s.eta_b = fixed_v6()["eta_b"]["TAKE-OFF"]
    s.sm_floor_pct = p["sm_floor"]
    eng.spec = s
    return eng


def offdesign_point(eng, x, beta, target_N):
    """P8.4b-A1: beta = 0 first; bleed active (13th unknown) if needed."""
    spec = eng.spec
    spec.ipc_bleed_active = False
    eng.spec = spec
    r = eng.solve_offdesign(x[:12], 0.0, 0.0, 0.0, "Fn", target_N)
    if r["converged"] and r["scalars"]["SMN_ipc"] >= spec.sm_floor_pct:
        return r, 0.0
    spec.ipc_bleed_active = True
    eng.spec = spec
    rb = eng.solve_offdesign(list(x[:12]) + [max(beta, 0.02)], 0.0, 0.0, 0.0, "Fn", target_N)
    spec.ipc_bleed_active = False
    eng.spec = spec
    if rb["converged"] and rb["x"][12] >= 0.0:
        return rb, rb["x"][12]
    if r["converged"]:
        r["reason"] = "bleed solve did not converge; beta = 0 point has SMN below the floor"
        r["converged"] = False
        return r, 0.0
    return rb, beta


def solve_engine(args) -> dict:
    """Design point + continuation to each requested mode. Returns per-mode results."""
    opr, bpr, rated, modes, p = args
    eng = make_engine(opr, bpr, rated, p)
    out = {"design": None, "modes": {}}
    d = eng.solve_design([rated * 1000.0 / 280.0, 0.03, 3.5, 2.0, 5.0])
    out["design"] = {k: d[k] for k in ("converged", "iterations", "reason", "x", "scalars",
                                       "mass_closure", "energy_closure", "element_closure")}
    if not d["converged"]:
        for m in modes:
            out["modes"][m] = {"status": "unreachable", "reason": f"design point: {d['reason']}"}
        return out
    sx = d["x"]
    x = [sx[0], sx[1], bpr, 1.0, 1.0, 1.0,
         eng.spec.fan.map.defaults["RlineMap"], eng.spec.ipc.map.defaults["RlineMap"],
         eng.spec.hpc.map.defaults["RlineMap"], sx[2], sx[3], sx[4]]
    eta = fixed_v6()["eta_b"]
    targets = sorted(((v5.MODE_X[m], m) for m in modes), reverse=True)
    level, failed, beta = 1.0, None, 0.0
    for xt, mode in targets:
        spec = eng.spec
        spec.eta_b = eta[mode]
        eng.spec = spec
        path = []
        while level - STEP > xt + 1e-12:
            level = round(level - STEP, 10)
            path.append(level)
        path.append(xt)
        for lv in path:
            if failed:
                break
            r, beta = offdesign_point(eng, x, beta, lv * rated * 1000.0)
            if not r["converged"]:
                failed = f"off-design at {lv:.2f} rated: {r['reason']}"
                break
            x = r["x"][:12]
            level = lv
        if failed:
            out["modes"][mode] = {"status": "unreachable", "reason": failed}
            continue
        out["modes"][mode] = {"status": "converged", "reason": "", "ff": r["scalars"]["Wfuel"],
                              "scalars": r["scalars"], "x": r["x"], "extrapolated": r["extrapolated_maps"],
                              "mass_closure": r["mass_closure"], "energy_closure": r["energy_closure"],
                              "element_closure": r["element_closure"]}
    return out


def ae3_inputs():
    row = v5.load_rows([AE3], with_targets=False)
    row = row.loc[row["Mode"] == "TAKE-OFF"].iloc[0]
    return float(row["Pressure Ratio"]), float(row["Bypass Ratio"]), float(row["Rated Thrust (kN)"])


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run_g1() -> int:
    if G1_PATH.exists():
        sys.exit(f"{G1_PATH} exists; write-once")
    opr, bpr, rated = ae3_inputs()
    modes = ["TAKE-OFF", "APPROACH", "IDLE"]
    central = solve_engine((opr, bpr, rated, modes, CENTRAL))
    checks = {}
    ok_modes = [m for m in modes if central["modes"][m]["status"] == "converged"]
    clos = {m: {k: central["modes"][m][k] for k in ("mass_closure", "energy_closure", "element_closure")}
            for m in ok_modes}
    checks["G1.1_closure_all_modes"] = (len(ok_modes) == 3 and all(
        c["mass_closure"] < 1e-8 and c["energy_closure"] < 1e-8 and c["element_closure"] < 1e-10
        for c in clos.values()) and central["design"]["converged"])
    repro = {}
    if "TAKE-OFF" in ok_modes:
        dx, ox = central["design"]["x"], central["modes"]["TAKE-OFF"]["x"][:12]
        pairs = {"W": (dx[0], ox[0]), "FAR": (dx[1], ox[1]), "PR_hpt": (dx[2], ox[9]),
                 "PR_ipt": (dx[3], ox[10]), "PR_lpt": (dx[4], ox[11])}
        repro = {k: abs(a - b) / abs(a) for k, (a, b) in pairs.items()}
    checks["G1.2_x1_reproduces_design"] = bool(repro) and max(repro.values()) < 1e-8
    # two-shaft regression against the committed P8.4 G1 solutions
    rec = json.loads((ROOT / "outputs/phase8/p84_g1.json").read_text())["solutions"]
    worst = 0.0
    for mode in ("matched", "production"):
        pts = C.run_points(C.make_engine(mode))
        for name, pt in pts.items():
            old = rec[mode][name]["scalars"]
            for k, v in pt["scalars"].items():
                worst = max(worst, abs(v - old[k]) / max(abs(old[k]), 1e-300))
    checks["G1.3_two_shaft_regression_bit_identical"] = worst == 0.0
    sens = {}
    for key, (lo, hi) in RANGES.items():
        for end, val in (("low", lo), ("high", hi)):
            p = dict(CENTRAL)
            p[key] = val
            res = solve_engine((opr, bpr, rated, modes, p))
            sens[f"{key}={end}:{val:g}"] = {m: (res["modes"][m].get("ff"), res["modes"][m]["status"]) for m in modes}
        print("sensitivity", key, flush=True)
    verdict = "PASS" if all(checks.values()) else "FAIL"
    doc = {"gate": "P8.4b checks with P8.4b-A1 handling bleed (ladder A4 pre-score)", "verdict": verdict,
           "checks": checks, "amendment": "docs/phase8_p84b_amendment_a1.md",
           "previous_record": "outputs/phase8/p84b_g1.json (FAIL, IPC stall at 0.5 rated)",
           "ae3_inputs": {"OPR": opr, "BPR": bpr, "rated_kN": rated}, "central_inputs": CENTRAL,
           "ae3_central": central, "closures": clos, "x1_vs_design_rel": repro,
           "two_shaft_regression_max_rel": worst,
           "sensitivity_ff_kg_s_reported": sens, "ranges": RANGES,
           "provenance": {"git_sha": git("rev-parse", "HEAD"),
                          "dirty_source": git("status", "--porcelain", "--", "cpp", "scripts"),
                          "module_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()}}
    with G1_PATH.open("x") as f:
        f.write(json.dumps(doc, indent=1, default=float) + "\n")
    print("P8.4b checks:", verdict, checks)
    return 0 if verdict == "PASS" else 1


def predict(rows: pd.DataFrame, workers: int) -> pd.DataFrame:
    keys = list(dict.fromkeys(zip(rows["Pressure Ratio"], rows["Bypass Ratio"], rows["Rated Thrust (kN)"])))
    modes = {k: sorted(set(rows.loc[(rows["Pressure Ratio"] == k[0]) & (rows["Bypass Ratio"] == k[1]) &
                                    (rows["Rated Thrust (kN)"] == k[2]), "Mode"])) for k in keys}
    tasks = [(k[0], k[1], k[2], modes[k], CENTRAL) for k in keys]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(keys, pool.map(solve_engine, tasks)))
    recs = []
    for _, r in rows.iterrows():
        res = results[(r["Pressure Ratio"], r["Bypass Ratio"], r["Rated Thrust (kN)"])]["modes"][r["Mode"]]
        recs.append({"status": res["status"], "reason": res["reason"],
                     "ff": res.get("ff", np.nan),
                     "T3": res.get("scalars", {}).get("Tt3"), "T4": res.get("scalars", {}).get("Tt4"),
                     "thrust_kN": res.get("scalars", {}).get("Fn_N", np.nan) / 1000.0
                     if res["status"] == "converged" else np.nan})
    return pd.DataFrame(recs, index=rows.index)


def run_score(workers: int) -> int:
    rel = str(G1_PATH.relative_to(ROOT))
    if subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode:
        sys.exit("commit p84b_g1.json first")
    if json.loads(G1_PATH.read_text())["verdict"] != "PASS":
        sys.exit("P8.4b checks did not pass; A4 is not scored")
    paths = [OUT_DIR / n for n in ("calibration_p8_A4_rows.csv", "holdout_p8_A4.csv",
                                   "holdout_p8_A4_summary.csv", "holdout_p8_A4.json")]
    if any(p.exists() for p in paths):
        sys.exit("A4 outputs exist; held-out rows are scored once")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reg6 = lto_v6.load_registration_v6()
    base = lto_v6.base_registration(reg6)
    split = v5.load_split()
    cal = v5.calibration_rows(split)
    held = v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True), split["heldout_groups"])
    pred_cal = predict(cal, workers)
    cal_out = cal.drop(columns=["CO (g/kg)", "HC (g/kg)", "NOx (g/kg)"], errors="ignore").join(pred_cal)
    cal_mape = v5.weighted_mape(pred_cal["ff"], cal, pred_cal["status"])
    pred = predict(held, workers)
    df, summary, fields = v5.holdout_tables(base, cal, held, pred)
    per_mode = summary.set_index("Scope")
    result = {"step": "A4 (P8.4b, no knobs)", "registration": "docs/phase8_p84b_registration.md",
              **fields, "calibration_rows_weighted_mape_pct_in_sample_no_fit": cal_mape,
              "per_mode_group_weighted_mape_pct": {
                  sc: {t: float(per_mode.loc[sc, f"{t} group-weighted MAPE (%)"]) for t in ("Model", "B0", "B1")}
                  for sc in per_mode.index},
              "A4_informativeness": lto_v6.a4_informativeness(df),
              "n_unreachable_heldout": int((pred["status"] == "unreachable").sum()),
              "n_unreachable_calibration": int((pred_cal["status"] == "unreachable").sum()),
              "v6_reference_pct": 1.830, "A1_pct": 1.835,
              "provenance": {"git_sha": git("rev-parse", "HEAD"),
                             "module_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()},
              "note": "scored once after the committed P8.4b checks; no parameter was fitted"}
    cal_out.to_csv(paths[0], index=False)
    df.to_csv(paths[1], index=False)
    summary.to_csv(paths[2], index=False)
    v5._write_new(paths[3], v5._json(result))
    print(f"A4 held-out: {fields['primary_group_weighted_mape_pct']['model']:.3f} %; calibration rows "
          f"(no fit) {cal_mape:.3f} %; unreachable held-out {result['n_unreachable_heldout']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("smoke", "g1", "score"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.phase == "g1":
        return run_g1()
    if a.phase == "score":
        return run_score(a.workers)
    opr, bpr, rated = ae3_inputs()
    res = solve_engine((opr, bpr, rated, ["TAKE-OFF", "APPROACH", "IDLE"], CENTRAL))
    d = res["design"]
    print("design", d["converged"], d["iterations"], d["reason"],
          {k: round(d["scalars"][k], 4) for k in ("W", "FAR", "OPR", "Tt4", "PR_hpt", "PR_ipt", "PR_lpt",
                                                 "Wfuel")} if d["converged"] else "")
    for m, r in res["modes"].items():
        print(m, r["status"], r["reason"], r.get("ff"), r.get("extrapolated"),
              {k: round(r["scalars"][k], 4) for k in ("SMN_ipc", "handling_bleed_frac", "SMN_hpc", "SMN_fan")}
              if "scalars" in r else "",
              {k: f"{r[k]:.0e}" for k in ("mass_closure", "energy_closure", "element_closure")} if "ff" in r else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
