#!/usr/bin/env python3
"""P8.3 G1 verdict (docs/phase8_p83_registration.md).

Component checks of the fixed-area convergent nozzle. States and synthetic
areas are declared below, before the first run; no held-out row is read and
no full-cycle score is computed. Real-gas states come from the P8.2 A2 cycle
at the AE3 calibration engine (frozen v6 phi), TAKE-OFF and IDLE.

Writes outputs/phase8/p83_g1.json once.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
for path in (ROOT, ROOT / "scripts" / "optimization", ROOT / "scripts" / "phase8"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lto_v5 as v5  # noqa: E402
from integrated_engine import LocalFuelBlend  # noqa: E402
from simulation.catjet_backend import CppEngine, load_core  # noqa: E402
from benchmark import protected_check  # noqa: E402
from p82_g1 import AE3, setup  # noqa: E402

MECH = ROOT / "data" / "creck_c1c16_full.yaml"
FIXTURE = ROOT / "cpp" / "tests" / "fixtures" / "perfect_gas.yaml"
TOL_LIMIT = 1e-10   # G1.1, G1.3, energy closure
TOL_JUMP = 1e-8     # G1.2
EPS = 1e-9          # p_ambient = p*(1 +/- EPS)
CD, CV = 0.96, 0.95  # registered fixed analog priors
P_AMB = 101325.0
# Declared constant-cp fixture states (T0 K, P0 Pa): choked and unchoked at P_AMB.
FIXTURE_STATES = {"hot_choked": (1100.0, 400000.0), "hot_unchoked": (1100.0, 160000.0),
                  "cold_unchoked": (330.0, 150000.0)}
# Declared synthetic areas for real-gas component tests only (m^2).
CORE_AREA, BYPASS_AREA = 0.34, 1.80
MODES = ("TAKE-OFF", "IDLE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def continuity(nozzle, state, area, cd, cv) -> dict:
    pstar = nozzle.critical_pressure(state)
    lo = nozzle.run(state, pstar * (1.0 - EPS), area, cd, cv)
    hi = nozzle.run(state, pstar * (1.0 + EPS), area, cd, cv)
    out = {"p_star": pstar, "p_star_over_p0": pstar / state.P,
           "choked_below": lo["choked"], "choked_above": hi["choked"],
           "mass_flow_jump": rel(lo["mass_flow"], hi["mass_flow"]),
           "force_jump": rel(lo["thrust_total"], hi["thrust_total"]),
           "energy_relative": max(lo["energy_relative"], hi["energy_relative"]),
           "Y_unchanged": lo["exit"]["Y"] == list(state.Y) and hi["exit"]["Y"] == list(state.Y)}
    # max(rho*u) check on both sides of p*
    peak = nozzle.mass_flux(state, pstar)
    out["flux_is_local_max"] = (nozzle.mass_flux(state, pstar * (1 - 1e-4)) < peak and
                                nozzle.mass_flux(state, pstar * (1 + 1e-4)) < peak)
    out["pass"] = (lo["choked"] and not hi["choked"] and out["mass_flow_jump"] < TOL_JUMP and
                   out["force_jump"] < TOL_JUMP and out["energy_relative"] < TOL_LIMIT and
                   out["Y_unchanged"] and out["flux_is_local_max"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/phase8/p83_g1.json")
    a = ap.parse_args()
    if a.output.exists():
        ap.error(f"{a.output} exists; output is write-once")
    p82_record = json.loads((ROOT / "outputs/phase8/p82_g1.json").read_text())
    if p82_record.get("verdict") != "PASS":
        ap.error("P8.3 runs only after a committed P8.2 G1 PASS")
    protected = protected_check()
    core = load_core()
    perfect = core.ChokingNozzle(str(FIXTURE))
    fixture_thermo = core.GasThermo(str(FIXTURE))
    real = core.ChokingNozzle(str(MECH))

    # G1.1 constant-cp critical ratio; G1.2 continuity; G1.3 v6 fully expanded limit
    g11, g12, g13 = {}, {}, {}
    for name, (T0, P0) in FIXTURE_STATES.items():
        state = core.GasState(T0, P0, [1.0])
        props = fixture_thermo.properties(state)
        gamma = props["gamma"]
        analytic = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        pstar = perfect.critical_pressure(state)
        g11[name] = {"gamma": gamma, "analytic": analytic, "computed": pstar / P0,
                     "relative": rel(pstar / P0, analytic)}
        g11[name]["pass"] = g11[name]["relative"] <= TOL_LIMIT
        g12[name] = {f"Cd{cd}_Cv{cv}": continuity(perfect, state, 0.08, cd, cv)
                     for cd, cv in ((1.0, 1.0), (CD, CV))}
        if pstar < P_AMB < P0:
            m_dot = 50.0
            area = m_dot / perfect.mass_flux(state, P_AMB)
            new = perfect.run(state, P_AMB, area, 1.0, 1.0)
            with contextlib.redirect_stdout(io.StringIO()):
                v6 = core.V6Engine(str(MECH))
            v6_noz = v6.run_nozzle({"T": T0, "p": P0, "cp": props["cp"], "R": props["R"],
                                    "gamma": gamma}, m_dot)
            g13[name] = {"area": area, "mass_flow": new["mass_flow"],
                         "force_new": new["thrust_total"], "force_v6": v6_noz["thrust_total"],
                         "relative": rel(new["thrust_total"], v6_noz["thrust_total"]),
                         "mass_flow_relative": rel(new["mass_flow"], m_dot),
                         "choked": new["choked"]}
            g13[name]["pass"] = (not new["choked"] and g13[name]["relative"] <= TOL_LIMIT and
                                 g13[name]["mass_flow_relative"] <= TOL_LIMIT)

    # G1.4 separate real-gas core/bypass nozzles at P8.2 A2 AE3 states
    reg = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
    fit = json.loads((ROOT / "outputs/phase7/calibration_v6.json").read_text())
    rows = v5.load_rows([AE3], with_targets=True).set_index("Mode")
    frozen = pd.read_csv(ROOT / "outputs/phase7/calibration_v6_rows.csv")
    frozen = frozen[frozen["Unique ID"] == AE3].set_index("Mode")
    fuel = LocalFuelBlend("JetA_dooley2012", dict(reg["fuel"]["mole_fractions"]))
    real_thermo = core.GasThermo(str(MECH))
    g14 = {}
    for mode in MODES:
        v6, p82 = setup(core, reg, fit, rows, mode)
        cyc = p82.run_full_cycle(*v6.fuel_args(fuel), float(frozen.loc[mode, "phi"]),
                                 float(reg["fixed_central"]["eta_b"][mode]), 2)
        st = cyc["stations"]
        core_in = core.GasState(st["lp_exit"]["T"], st["lp_exit"]["P"], st["lp_exit"]["Y"])
        byp_in = core.GasState(st["fan_exit"]["T"], st["fan_exit"]["P"], st["fan_exit"]["Y"])
        dual = real.run_dual(core_in, byp_in, P_AMB, CORE_AREA, BYPASS_AREA, CD, CV, CD, CV)
        elem = {}
        for label, inlet, res in (("core", core_in, dual["core"]), ("bypass", byp_in, dual["bypass"])):
            e_in = real_thermo.properties(inlet)["elements"]
            ex = res["exit"]
            e_out = real_thermo.properties(core.GasState(ex["T"], ex["P"], ex["Y"]))["elements"]
            elem[label] = {"Y_unchanged": ex["Y"] == list(inlet.Y),
                           "element_max_abs": max(abs(x - y) for x, y in zip(e_in, e_out)),
                           "energy_relative": res["energy_relative"], "choked": res["choked"],
                           "p0_over_pamb": inlet.P / P_AMB, "p_star_over_p0":
                           res["critical_pressure"] / inlet.P,
                           "mass_flow_capacity": res["mass_flow"],
                           "thrust_pressure": res["thrust_pressure"]}
        g14[mode] = {
            "states_distinct": (core_in.T != byp_in.T and core_in.P != byp_in.P),
            "areas": [CORE_AREA, BYPASS_AREA], "nozzles": elem,
            "continuity_core": continuity(real, core_in, CORE_AREA, CD, CV),
            "continuity_bypass": continuity(real, byp_in, BYPASS_AREA, CD, CV),
            "sum_consistent": (abs(dual["thrust_total"] - dual["core"]["thrust_total"] -
                                   dual["bypass"]["thrust_total"]) <= 1e-12 * abs(dual["thrust_total"])),
        }
        g14[mode]["pass"] = (g14[mode]["states_distinct"] and g14[mode]["sum_consistent"] and
                             all(n["Y_unchanged"] and n["element_max_abs"] == 0.0 and
                                 n["energy_relative"] < TOL_LIMIT for n in elem.values()) and
                             g14[mode]["continuity_core"]["pass"] and
                             g14[mode]["continuity_bypass"]["pass"])

    checks = {
        "G1.1_constant_cp_critical_ratio": all(v["pass"] for v in g11.values()),
        "G1.2_continuity_at_critical_ratio": all(c["pass"] for v in g12.values() for c in v.values()),
        "G1.3_unchoked_v6_limit": bool(g13) and all(v["pass"] for v in g13.values()),
        "G1.4_separate_real_gas_nozzles_closure": all(v["pass"] for v in g14.values()),
        "G1.4_protected_hashes": not protected["mismatches"],
        "G1.4_g0_sources_unchanged_since_G0": run("git", "diff", "--stat", "74f53c8", "--",
                                                  "cpp/catjet_core/v6_engine.cpp",
                                                  "cpp/catjet_core/v6_engine.hpp",
                                                  "cpp/catjet_core/brentq.hpp") == "",
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    doc = {
        "gate": "P8.3 G1", "verdict": verdict, "checks": checks,
        "tolerances": {"limit_and_energy": TOL_LIMIT, "jump": TOL_JUMP, "eps": EPS},
        "coefficients": {"Cd": CD, "Cv": CV, "status": "fixed analog priors (TN-1757, RP-1235)"},
        "G1.1": g11, "G1.2": g12, "G1.3": g13, "G1.4": g14,
        "notes": [
            "Real-gas areas are declared synthetic values for component tests; nozzle mass-flow "
            "capacity is reported, not matched to the imposed v6 flow (P8.4 matching).",
            "A fresh 180-row g0_parity run is not in this record; the v6 translation units are "
            "checked byte-identical to the G0 PASS commit and the C++ parity tests run in pytest.",
        ],
        "provenance": {
            "git_sha": run("git", "rev-parse", "HEAD"),
            "dirty_source": run("git", "status", "--porcelain", "--", "cpp", "scripts", "tests"),
            "p82_g1_sha256": sha256(ROOT / "outputs/phase8/p82_g1.json"),
            "mechanism_sha256": sha256(MECH), "fixture_sha256": sha256(FIXTURE),
            "conda_lock_sha256": sha256(ROOT / "cpp/conda-lock-osx-arm64.txt"),
            "module_sha256": sha256(Path(core.__file__)),
            "compiler": run("/Library/Developer/CommandLineTools/usr/bin/clang++", "--version")
                        .splitlines()[0],
            "machine": platform.platform(),
        },
        "protected": protected,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(doc, indent=2, default=float) + "\n")
    print(f"P8.3 G1: {verdict} {checks}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
