#!/usr/bin/env python3
"""P8.4 code-to-code check of the C++ HBTF against the pinned pyCycle 4.4.0 example.

docs/phase8_p84_registration.md and amendment P8.4-A1. Points: DESIGN,
OD_full_pwr (T4 throttle) and OD_part_pwr (Fn = 0.8 Fn_full), M 0.8, 35 kft.
Thermo-matched mode (pyCycle JANAF in Cantera, shifting equilibrium) is
compared at the registered tolerance; production mode (CRECK, frozen
outside the burner) is reported as deltas only.

Default: print a development summary (no file). --g1 writes the write-once
outputs/phase8/p84_g1.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(os.environ.get("CATJET_BUILD", ROOT / "cpp" / "build"))))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import catjet_core as core  # noqa: E402

LBM, PSI, FT, LBF, HP = 0.45359237, 6894.757293168361, 0.3048, 4.4482216152605, 745.6998715822702
R2K = 5.0 / 9.0
REF = ROOT / "outputs" / "phase8" / "pycycle_hbtf_reference.json"
JANAF = ROOT / "data" / "thermo" / "pycycle_janaf.yaml"
US1976 = ROOT / "data" / "thermo" / "pycycle_us1976.json"
MAPS = ROOT / "data" / "maps" / "pycycle_4.4.0"
CRECK = ROOT / "data" / "creck_c1c16_full.yaml"
TOL_REL, TOL_T_ABS = 1e-3, 0.5


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_map(name: str):
    f = json.loads((MAPS / f"{name}.json").read_text())["fields"]
    m = core.ComponentMap()
    m.name = name
    grids = [list(map(float, p["values"]["data"] if isinstance(p["values"], dict) else p["values"]))
             for p in f["param_data"]]
    m.params = [p["name"] for p in f["param_data"]]
    outs = {}
    for o in f["output_data"]:
        v = o["values"]
        flat = [float(x) for x in _flatten(v["data"] if isinstance(v, dict) else v)]
        outs[o["name"]] = core.GridTable(grids, flat)
    m.outputs = outs
    m.defaults = {k: float(v) for k, v in f["defaults"].items()}
    m.rline_stall = float(f.get("RlineStall", 0.0) or 0.0)
    return m


def _flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from _flatten(y)
    else:
        yield x


def bleed(name, W, P, work):
    b = core.Bleed()
    b.name, b.frac_W, b.frac_P, b.frac_work = name, W, P, work
    return b


def make_spec():
    """Every value from envs/pycycle/upstream/example_cycles/high_bypass_turbofan.py (4.4.0)."""
    s = core.HbtfSpec()
    s.alt_m, s.MN, s.dTs_K = 35000.0 * FT, 0.8, 0.0
    s.Fn_des_N, s.T4_max_K = 5900.0 * LBF, 2857.0 * R2K
    s.N_lp_des, s.N_hp_des, s.BPR_des = 4666.1, 14705.7, 5.105
    s.ram_recovery = 0.9990
    s.dPqP_duct4, s.dPqP_duct6, s.dPqP_burner = 0.0048, 0.0101, 0.0540
    s.dPqP_duct11, s.dPqP_duct13, s.dPqP_duct15 = 0.0051, 0.0107, 0.0149
    s.Cv_core, s.Cv_byp, s.frac_byp_bleed = 0.9933, 0.9939, 0.005
    s.cool3_frac_W, s.cool4_frac_W = 0.067214, 0.101256
    s.cool3_frac_P, s.cool4_frac_P, s.cool1_frac_P_lpt, s.cool2_frac_P_lpt = 1.0, 0.0, 1.0, 0.0
    s.HPX_W = 250.0 * HP
    for attr, mapname, PR, eff, bleeds in (
            ("fan", "FanMap", 1.685, 0.8948, []),
            ("lpc", "LPCMap", 1.935, 0.9243, []),
            ("hpc", "HPCMap", 9.369, 0.8707, [bleed("cool1", 0.050708, 0.5, 0.5),
                                              bleed("cool2", 0.020274, 0.55, 0.5),
                                              bleed("cust", 0.0445, 0.5, 0.5)])):
        c = core.CompressorSpec()
        c.name, c.map, c.PR_des, c.eff_des, c.bleeds = attr, load_map(mapname), PR, eff, bleeds
        setattr(s, attr, c)
    for attr, mapname, eff in (("hpt", "HPTMap", 0.8888), ("lpt", "LPTMap", 0.8996)):
        t = core.TurbineSpec()
        t.name, t.map, t.eff_des = attr, load_map(mapname), eff
        setattr(s, attr, t)
    atm = json.loads(US1976.read_text())
    s.atm_alt_ft, s.atm_T_R, s.atm_P_psi = atm["alt_ft"], atm["T_degR"], atm["P_psi"]
    return s


def make_engine(mode: str):
    if mode == "matched":
        meta = yaml.safe_load(JANAF.read_text())["pycycle_metadata"]
        air = ",".join(f"{e}x:{v}" for e, v in meta["CEA_AIR_COMPOSITION"].items())
        fuel = ",".join(f"{e}x:{v}" for e, v in meta["jet_a_g_elements"].items())
        weights = {f"{e}x": float(w) for e, w in meta["element_wts"].items()}
        eng = core.Hbtf(str(JANAF), "matched", air, fuel, weights)
    else:
        reg = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
        fuel = ", ".join(f"{k}:{v}" for k, v in reg["fuel"]["mole_fractions"].items())
        eng = core.Hbtf(str(CRECK), "production", "O2:1, N2:3.76", fuel)
    eng.spec = make_spec()
    return eng


def run_points(eng) -> dict:
    out = {}
    d = eng.solve_design([100.0 * LBM, 0.025, 3.0, 4.0])   # pyCycle example initial guesses
    out["DESIGN"] = d
    if not d["converged"]:
        return out
    s = d["scalars"]
    guess = [300.0 * LBM, 0.02467, 5.105, 5000.0, 15000.0, 2.0, 2.0, 2.0, 3.0, 4.0]  # example guesses
    full = eng.solve_offdesign(guess, 35000.0 * FT, 0.8, 0.0, "T4", 2857.0 * R2K)
    out["OD_full_pwr"] = full
    if full["converged"]:
        part = eng.solve_offdesign(full["x"], 35000.0 * FT, 0.8, 0.0, "Fn", 0.8 * full["scalars"]["Fn_N"])
        out["OD_part_pwr"] = part
    return out


def mine(point: dict) -> dict:
    s = point["scalars"]
    st = point["stations"]
    v = {"W_lbm_s": s["W"] / LBM, "Fn_lbf": s["Fn_N"] / LBF, "Fg_lbf": s["Fg_N"] / LBF,
         "Fram_lbf": s["F_ram_N"] / LBF, "OPR": s["OPR"], "BPR": s["BPR"], "FAR": s["FAR"],
         "TSFC_lbm_h_lbf": s["TSFC_kg_per_N_s"] * 3600.0 * LBF / LBM, "Tt4_degR": s["Tt4"] / R2K,
         "Wfuel_lbm_s": s["Wfuel"] / LBM, "LP_Nmech_rpm": s["N_lp"], "HP_Nmech_rpm": s["N_hp"],
         "fan_PR": s["PR_fan"], "lpc_PR": s["PR_lpc"], "hpc_PR": s["PR_hpc"], "hpt_PR": s["PR_hpt"],
         "lpt_PR": s["PR_lpt"], "Tt3_degR": s["Tt3"] / R2K}
    rename = {"splitter1": "splitter.Fl_O1", "splitter2": "splitter.Fl_O2"}
    for name, f in st.items():
        name = rename.get(name, f"{name}.Fl_O")
        v[f"{name}:tot:T"] = f["Tt"] / R2K
        v[f"{name}:tot:P"] = f["Pt"] / PSI
        v[f"{name}:stat:W"] = f["W"] / LBM
    return v


def reference(name: str) -> dict:
    p = json.loads(REF.read_text())["reference_case"]["points"][name]
    v = {k: x for k, x in p["summary"].items() if isinstance(x, (int, float)) and "RlineMap" not in k}
    v.pop("MN", None)
    v.pop("alt_ft", None)
    for st, fields in p["data"]["station"].items():
        for f in ("tot:T", "tot:P", "stat:W"):
            if f in fields:
                v[f"{st}:{f}"] = fields[f]["value"]
    v["Tt3_degR"] = p["data"]["station"]["hpc.Fl_O"]["tot:T"]["value"]
    return v


def compare(m: dict, r: dict) -> dict:
    rows, worst, fails = {}, 0.0, []
    for k, ref in r.items():
        if k not in m or ref == 0:
            continue
        rel = abs(m[k] - ref) / abs(ref)
        if "tot:T" in k or k.startswith("Tt"):
            ok = abs(m[k] - ref) * R2K <= min(TOL_REL * abs(ref) * R2K, TOL_T_ABS)
        else:
            ok = rel <= TOL_REL
        rows[k] = {"catjet": m[k], "pycycle": ref, "rel": rel, "pass": ok}
        worst = max(worst, rel)
        if not ok:
            fails.append(k)
    missing = sorted(set(r) - set(rows))
    return {"n_compared": len(rows), "worst_rel": worst, "failures": fails, "rows": rows,
            "not_compared": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g1", action="store_true", help="write outputs/phase8/p84_g1.json (once)")
    a = ap.parse_args()
    out_path = ROOT / "outputs" / "phase8" / "p84_g1.json"
    if a.g1 and out_path.exists():
        ap.error(f"{out_path} exists; write-once")
    results = {}
    for mode in ("matched", "production"):
        pts = run_points(make_engine(mode))
        results[mode] = pts
        for name, p in pts.items():
            print(mode, name, "converged" if p["converged"] else f"FAILED ({p['reason']})",
                  "it", p["iterations"], "|r|", f"{p['norm_history'][-1] if p['norm_history'] else float('nan'):.1e}",
                  flush=True)
    cmp_ = {}
    for name in ("DESIGN", "OD_full_pwr", "OD_part_pwr"):
        p = results["matched"].get(name)
        if p and p["converged"]:
            cmp_[name] = compare(mine(p), reference(name))
            print(name, "worst rel", f"{cmp_[name]['worst_rel']:.2e}", "fails", cmp_[name]["failures"][:8])
    if not a.g1:
        return 0
    names = ("DESIGN", "OD_full_pwr", "OD_part_pwr")
    matched, prod = results["matched"], results["production"]
    all_conv = all(n in matched and matched[n]["converged"] for n in names)
    g1_1 = all_conv and all(not cmp_[n]["failures"] and not cmp_[n]["not_compared"] for n in names)
    g1_2 = all_conv and all(matched[n]["mass_closure"] < 1e-8 and matched[n]["energy_closure"] < 1e-8
                            for n in names) and \
        all(n in prod and prod[n]["converged"] and prod[n]["element_closure"] < 1e-10 and
            prod[n]["mass_closure"] < 1e-8 and prod[n]["energy_closure"] < 1e-8 for n in names)
    g1_3 = all_conv and all(not matched[n]["extrapolated"] for n in names)
    checks = {"G1.1_matched_within_0.10pct": g1_1, "G1.2_closure_audit": g1_2,
              "G1.3_bounded_no_extrapolation": g1_3}
    deltas = {}
    for n in names:
        if n in prod and prod[n]["converged"]:
            m, r = mine(prod[n]), reference(n)
            deltas[n] = {k: {"catjet_production": m[k], "pycycle": r[k], "rel": (m[k] - r[k]) / r[k]}
                         for k in r if k in m and r[k] != 0}
    strip = lambda p: {k: v for k, v in p.items() if k != "stations"}  # noqa: E731
    doc = {"gate": "P8.4 G1", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
           "tolerances": {"relative": TOL_REL, "temperature_abs_K": TOL_T_ABS,
                          "temperature_rule": "min(0.10 % of T, 0.5 K)", "closure_mass_energy": 1e-8,
                          "element_closure_production": 1e-10, "newton_max_scaled_residual": 1e-10},
           "comparison_matched": cmp_,
           "solutions": {mode: {n: strip(p) for n, p in pts.items()} for mode, pts in results.items()},
           "production_deltas_reported_only": deltas,
           "notes": ["Matched-mode element closure (~1e-5) reflects pyCycle's own inconsistency: species "
                     "'wt' imply C = 12.0107 while element_wts (used for the fuel formula) has 12.0170; it "
                     "is not gated (registration: element closure with production thermo).",
                     "Production deltas are differences of thermochemistry (CRECK, Dooley 2012 surrogate, "
                     "liquid-basis fuel, air without Ar), reported, never compared or tuned."],
           "provenance": {"git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                   text=True).stdout.strip(),
                          "dirty_source": subprocess.run(["git", "status", "--porcelain", "--", "cpp", "scripts"],
                                                         cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                          "module_sha256": sha256(Path(core.__file__)), "reference_sha256": sha256(REF),
                          "janaf_sha256": sha256(JANAF), "us1976_sha256": sha256(US1976),
                          "maps_sha256": {p.name: sha256(p) for p in sorted(MAPS.glob("*.json"))},
                          "creck_sha256": sha256(CRECK), "machine": platform.platform()}}
    with out_path.open("x") as f:
        f.write(json.dumps(doc, indent=1, default=float) + "\n")
    print("P8.4 G1:", doc["verdict"], checks)
    return 0 if doc["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
