#!/usr/bin/env python3
"""P8.5 G1 verdict (docs/phase8_p85_registration.md, amendments A1, A2).

Gating: (1) long-residence limit vs HP equilibrium, (2) element and energy
closure < 1e-10, (3) sigma_phi = 0 gives identical primary PSRs (1e-12).
Reported only: (4) K = 7 vs 9, (5) A2NOx/CRECK thermo consistency,
(6) mechanism spread with CRECK. Points: AE3 TAKE-OFF, APPROACH, IDLE
combustor inlets of the frozen v6 rows. Test parameter values only; no
prediction, calibration or held-out row. Writes outputs/phase8/p85_g1.json once.

Non-gating diagnostic (P8.5-A2): the element imbalance of the protected
A2NOx mechanism's lumped fuel reactions, integrated over every PSR at its
outlet state, compared with the observed network element error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BUILD = Path(os.environ.get("CATJET_BUILD", ROOT / "cpp" / "build"))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import catjet_core as core  # noqa: E402
import cantera as ct  # noqa: E402

from benchmark import protected_check  # noqa: E402

A2 = ROOT / "data" / "A2NOx.yaml"
CRECK = ROOT / "data" / "creck_c1c16_full.yaml"
AE3 = "02P23RR126"
MODES = ("TAKE-OFF", "APPROACH", "IDLE")
DP = 0.045
TOL_T, TOL_Y, Y_FLOOR = 0.1, 1e-5, 1e-6
TOL_CLOSURE, TOL_SIGMA0 = 1e-10, 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def params(**kw):
    p = core.NetworkParams()  # registered test values are the defaults
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def inlets() -> dict:
    rows = pd.read_csv(ROOT / "outputs/phase7/calibration_v6_rows.csv")
    rows = rows[rows["Unique ID"] == AE3].set_index("Mode")
    # v6 burner air = core air (single zone, beta = 1)
    return {m: (float(rows.loc[m, "T3"]), float(rows.loc[m, "p3_bar"]) * 1e5,
                float(rows.loc[m, "m_core"]), float(rows.loc[m, "ff"])) for m in MODES}


def imbalance_diagnostic(result: dict) -> dict:
    """Element creation by imbalanced reactions, summed over all PSRs."""
    g = ct.Solution(str(A2))
    A = np.array([[g.n_atoms(k, e) for k in range(g.n_species)] for e in g.element_names])
    imb = A @ (g.product_stoich_coeffs - g.reactant_stoich_coeffs)   # atoms per reaction
    w = np.array([g.atomic_weight(e) for e in g.element_names])
    created = np.zeros(len(g.element_names))
    for z in result["primary"] + [result["quench"]] + result["lean"]:
        out = z["outlet"]
        g.TPY = out["T"], out["P"], out["Y"]
        created += imb @ g.net_rates_of_progress * z["volume"] * w   # kg/s per element
    g.TPY = result["exit"]["T"], result["exit"]["P"], result["exit"]["Y"]
    reactions = [int(j) for j in np.where(np.abs(imb).max(axis=0) > 1e-12)[0]]
    return {"elements": g.element_names, "predicted_creation_kg_s": created.tolist(),
            "imbalanced_reactions": [{"index": j, "equation": g.reaction(j).equation,
                                      "atoms": {e: float(imb[i, j]) for i, e in enumerate(g.element_names)
                                                if abs(imb[i, j]) > 1e-12}} for j in reactions]}


def observed_element_error(result: dict, T3, P, m_air, m_fuel, fuel) -> dict:
    g = ct.Solution(str(A2))
    el = lambda: np.array([g.elemental_mass_fraction(e) for e in g.element_names])  # noqa: E731
    g.TPX = T3, P, "O2:1, N2:3.76"
    za = el()
    g.TPX = T3, P, fuel
    zf = el()
    g.TPY = result["exit"]["T"], result["exit"]["P"], result["exit"]["Y"]
    z_in = m_air * za + m_fuel * zf
    z_out = result["mass_flow_exit"] * el()
    return {"elements": g.element_names, "observed_out_minus_in_kg_s": (z_out - z_in).tolist(),
            "relative": (np.abs(z_out - z_in) / np.maximum(np.abs(z_in), 1e-12)).tolist(),
            "exit_sum_Y_minus_1": float(np.sum(result["exit"]["Y"]) - 1.0)}


def thermo_consistency() -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a2 = ct.Solution(str(A2))
    disc = sorted({str(w.message).split("For species ")[1].split(",")[0]
                   for w in caught if "discontinuity" in str(w.message)})
    cr = ct.Solution(str(CRECK))
    shared = sorted(set(a2.species_names) & set(cr.species_names))
    grid = np.arange(300.0, 2501.0, 50.0)
    worst = {}
    for name in shared:
        dcp, dh = 0.0, 0.0
        for T in grid:
            sa, sc = a2.species(name).thermo, cr.species(name).thermo
            cpa, cpc = sa.cp(T), sc.cp(T)
            ha, hc = sa.h(T), sc.h(T)
            dcp = max(dcp, abs(cpa - cpc) / abs(cpc))
            dh = max(dh, abs(ha - hc) / max(abs(hc), ct.gas_constant * T))
        worst[name] = {"cp_rel": dcp, "h_rel_vs_max(|h|,RT)": dh}
    top = sorted(worst.items(), key=lambda kv: -max(kv[1].values()))
    return {"n_shared": len(shared), "a2_polynomial_discontinuities": disc,
            "worst_10_shared": dict(top[:10]),
            "max_cp_rel": max(v["cp_rel"] for v in worst.values()),
            "max_h_rel": max(v["h_rel_vs_max(|h|,RT)"] for v in worst.values()),
            "h_relative_note": "h difference divided by max(|h|, R T) since h crosses zero"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/phase8/p85_g1_rev1.json")
    a = ap.parse_args()
    if a.output.exists():
        ap.error(f"{a.output} exists; output is write-once")
    protected = protected_check()
    if protected["mismatches"]:
        ap.error(f"protected hashes changed: {protected['mismatches'][:5]}")
    reg = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
    creck_fuel = ", ".join(f"{k}:{v}" for k, v in reg["fuel"]["mole_fractions"].items())
    t0 = time.time()
    net = core.ReactorNetwork(str(A2), "POSF10325", 9)
    pts = inlets()
    out: dict = {"inputs": {m: dict(zip(("T3", "P3", "m_air", "m_fuel"), v)) for m, v in pts.items()}}
    g1, g2, g3, g4, diag = {}, {}, {}, {}, {}
    for mode, (T3, P3, ma, mf) in pts.items():
        P = P3 * (1 - DP)
        base = params()
        d = net.design(T3, P3, ma, mf, base)
        # (1) long residence, no dilution
        long = net.run(T3, P3, ma, mf, params(no_dilution=True, volume_scale=1e4), d)
        eq = net.equilibrium(T3, P, ma, mf)
        Y_eq, Y_net = np.array(eq.Y), np.array(long["lean_exit"]["Y"])
        mask = Y_eq > Y_FLOOR
        dT = abs(long["lean_exit"]["T"] - eq.T)
        dY = float(np.max(np.abs(Y_net - Y_eq)[mask]))
        g1[mode] = {"T_network": long["lean_exit"]["T"], "T_equilibrium": eq.T, "dT_K": dT,
                    "max_dY_where_Yeq_gt_1e-6": dY, "converged": long["all_converged"],
                    "psr_errors": [z["error"][:300] for z in long["primary"] + [long["quench"]] + long["lean"]
                                   if z["error"]],
                    "pass": bool(dT <= TOL_T and dY <= TOL_Y and long["all_converged"])}
        # (2) closure at test values, and in the long-residence run
        res = net.run(T3, P3, ma, mf, base, d)
        cases = {"test_values": res, "long_residence": long}
        g2[mode] = {k: {"energy_relative": r["energy_relative"], "element_relative": r["element_relative"],
                        "max_mixer_energy_relative": r["max_mixer_energy_relative"],
                        "max_mixer_element_relative": r["max_mixer_element_relative"]}
                    for k, r in cases.items()}
        worst = max(max(v.values()) for v in g2[mode].values())
        g2[mode]["worst"] = worst
        g2[mode]["pass"] = bool(worst < TOL_CLOSURE)
        # (3) sigma = 0
        s0 = net.run(T3, P3, ma, mf, params(sigma_rel=0.0), d)
        Ts = [z["outlet"]["T"] for z in s0["primary"]]
        Ys = np.array([z["outlet"]["Y"] for z in s0["primary"]])
        t_rel = (max(Ts) - min(Ts)) / max(Ts)
        y_rel = float(np.max((Ys.max(0) - Ys.min(0)) / np.maximum(np.abs(Ys).max(0), 1e-300)))
        y_rel_major = float(np.max(((Ys.max(0) - Ys.min(0)) / np.maximum(Ys.max(0), 1e-300))[Ys.max(0) > 1e-6]))
        g3[mode] = {"T_relative_spread": t_rel, "Y_relative_spread_all": y_rel,
                    "Y_relative_spread_Y_gt_1e-6": y_rel_major,
                    "pass": bool(t_rel <= TOL_SIGMA0 and y_rel <= TOL_SIGMA0)}
        # (4) K = 7 vs 9 (reported)
        k9 = net.run(T3, P3, ma, mf, params(K=9), d)
        g4[mode] = {f: {"K7": res[f], "K9": k9[f], "delta": k9[f] - res[f]}
                    for f in ("eta_b", "EI_NOx_g_kg", "EI_CO_g_kg")}
        g4[mode]["exit_T"] = {"K7": res["exit"]["T"], "K9": k9["exit"]["T"],
                              "delta": k9["exit"]["T"] - res["exit"]["T"]}
        out.setdefault("test_value_outputs", {})[mode] = {
            "eta_b": res["eta_b"], "EI_NOx_g_kg": res["EI_NOx_g_kg"], "EI_CO_g_kg": res["EI_CO_g_kg"],
            "EI_UHC_g_kg": res["EI_UHC_g_kg"], "exit_T": res["exit"]["T"], "phi_pz": res["phi_pz"],
            "alpha_pz": res["alpha_pz"], "alpha_dil": res["alpha_dil"], "all_converged": res["all_converged"],
            "any_extinguished": res["any_extinguished"],
            "primary_T": [z["outlet"]["T"] for z in res["primary"]],
            "primary_tau_ms": [1e3 * z["residence_time"] for z in res["primary"]],
            "note": "registered G1 test values; not a prediction"}
        diag[mode] = {"observed": observed_element_error(res, T3, P, ma, mf, "POSF10325:1"),
                      "predicted_from_mechanism_imbalance": imbalance_diagnostic(res)}
        print(mode, "G1.1", g1[mode]["pass"], "G1.2", g2[mode]["pass"], "%.2e" % worst,
              "G1.3", g3[mode]["pass"], "%.1e" % t_rel, flush=True)

    # (5) thermo consistency and HP-equilibrium T with each mechanism at AE3 take-off FAR
    T3, P3, ma, mf = pts["TAKE-OFF"]
    P = P3 * (1 - DP)
    t5 = thermo_consistency()
    eq_a2 = net.equilibrium(T3, P, ma, mf).T
    creck = core.ReactorNetwork(str(CRECK), creck_fuel, 9)
    eq_cr = creck.equilibrium(T3, P, ma, mf).T
    t5["HP_equilibrium_T_AE3_takeoff"] = {"A2NOx_POSF10325": eq_a2, "CRECK_Dooley2012": eq_cr,
                                          "delta_K": eq_cr - eq_a2,
                                          "basis": "same fuel/air mass ratio, liquid basis 360 kJ/kg"}
    # (6) mechanism spread at test values (CRECK has no N chemistry)
    spread = {}
    for mode, (T3, P3, ma, mf) in pts.items():
        dcr = creck.design(T3, P3, ma, mf, params())
        rc = creck.run(T3, P3, ma, mf, params(), dcr)
        ra = out["test_value_outputs"][mode]
        spread[mode] = {"eta_b": {"A2NOx": ra["eta_b"], "CRECK": rc["eta_b"]},
                        "EI_CO_g_kg": {"A2NOx": ra["EI_CO_g_kg"], "CRECK": rc["EI_CO_g_kg"]},
                        "exit_T": {"A2NOx": ra["exit_T"], "CRECK": rc["exit"]["T"]},
                        "CRECK_converged": rc["all_converged"]}
        print(mode, "spread done", flush=True)

    checks = {"G1.1_long_residence_equilibrium": all(v["pass"] for v in g1.values()),
              "G1.2_element_energy_closure": all(v["pass"] for v in g2.values()),
              "G1.3_sigma_zero_identical": all(v["pass"] for v in g3.values())}
    verdict = "PASS" if all(checks.values()) else "FAIL"
    doc = {"gate": "P8.5 G1", "verdict": verdict, "checks": checks,
           "tolerances": {"dT_K": TOL_T, "dY": TOL_Y, "Y_floor": Y_FLOOR,
                          "closure": TOL_CLOSURE, "sigma0": TOL_SIGMA0},
           "G1.1": g1, "G1.2": g2, "G1.3": g3, "G1.4_quadrature_reported": g4,
           "G1.5_thermo_consistency_reported": t5, "G1.6_mechanism_spread_reported": spread,
           "diagnostic_mechanism_element_imbalance": diag, **out,
           "previous_attempts": [
               "attempt 1 (no record written): TAKE-OFF printed G1.1 True, G1.2 False (4.59e-07), "
               "G1.3 False (T spread 1.2e-15); the APPROACH long-residence run then raised an "
               "uncaught CVODES error-test failure (t = 2.31 s). Fix before this attempt: the PSR "
               "solver now records integrator failure as non-convergence (registered reporting rule); "
               "no tolerance or check was changed.",
               "attempt 2 = outputs/phase8/p85_g1.json (FAIL). This rev1 run follows P8.5-A3: non-cloned "
               "file-loaded Solutions (cloning rounded the HyChem stoichiometry, 15x larger imbalance with "
               "the opposite sign) and the temperature-state PSR; checks unchanged."],
           "wall_s": time.time() - t0,
           "provenance": {"git_sha": run("git", "rev-parse", "HEAD"),
                          "dirty_source": run("git", "status", "--porcelain", "--", "cpp", "scripts"),
                          "module": str(BUILD), "module_sha256": sha256(Path(core.__file__)),
                          "a2nox_sha256": sha256(A2), "creck_sha256": sha256(CRECK),
                          "cantera_python": ct.__version__, "machine": platform.platform()},
           "protected": protected}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(doc, indent=2, default=float) + "\n")
    print(f"P8.5 G1: {verdict} {checks}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
