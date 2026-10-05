#!/usr/bin/env python3
"""P8.2 G1 verdict (docs/phase8_p82_registration.md, amendments A1 and A2).

Numerical checks only, at the Trent 1000-AE3 calibration engine with the
frozen v6 parameters. No held-out row is read and no ablation step is
calibrated here. States: AE3 TAKE-OFF plus the lower-work AE3 APPROACH and
IDLE rows (all three enter the verdict; fixed before the first run). Each
state uses the frozen v6 converged phi of that row.

Writes outputs/phase8/p82_g1.json once.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
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

AE3 = "02P23RR126"
MODES = ("TAKE-OFF", "APPROACH", "IDLE")
TOL_LIMIT = 1e-10      # G1.1 constant-cp and A1/A2 interface
TOL_CLOSURE = 1e-10    # G1.2 mass, element and energy closure
TOL_STEPS = 1e-8       # G1.3 50 -> 100 pressure steps
# G0 translation units: unchanged since the G0 PASS commit is part of G1.4.
G0_COMMIT = "74f53c8"
G0_SOURCES = ("cpp/catjet_core/v6_engine.cpp", "cpp/catjet_core/v6_engine.hpp",
              "cpp/catjet_core/brentq.hpp")
# P8.2-A2: development probe results kept in the record (not G1 results).
DEVELOPMENT_PROBE = {
    "note": "uncommitted smoke probe before the 1e-13 HP-inversion correction",
    "ae3_takeoff_A1_energy_relative": 1.14e-16,
    "ae3_takeoff_A2_max_energy_relative": 1.25e-9,
    "two_stream_mix_energy_relative": 4.5e-10,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def setup(core, reg: dict, fit: dict, rows: pd.DataFrame, mode: str):
    row = rows.loc[mode]
    state = v5.mode_state(fit["params"], reg["fixed_central"], row["Pressure Ratio"],
                          row["Bypass Ratio"], row["Rated Thrust (kN)"], v5.MODE_X[mode])
    with contextlib.redirect_stdout(io.StringIO()):
        v6 = CppEngine()
    v6.design_point.update(state)
    v6.compressor.eta_c = reg["fixed_central"]["eta_compressor"]
    v6.turbine_design["eta_polytropic"] = reg["fixed_central"]["eta_turbine_polytropic"]
    v6.push_config()
    p82 = core.P82Engine(v6.mechanism)
    c = p82.config
    c.base = v6.core.config
    p82.config = c
    return v6, p82


def set_config(engine, **kwargs):
    c = engine.config
    for key, value in kwargs.items():
        setattr(c, key, value)
    engine.config = c


def closure(result: dict) -> dict:
    stages = {name: {"energy_relative": s["energy_relative"],
                     "element_relative": s["element_relative"],
                     "requested_work_W": s["requested_work"],
                     "actual_work_W": s["actual_work"]}
              for name, s in result.get("stages", {}).items()}
    return {"max_mass_relative": result["max_mass_relative"],
            "max_energy_relative": result["max_energy_relative"],
            "max_element_relative": result["max_element_relative"],
            "burner_heat_rejection_W": result["burner_heat_rejection_W"],
            "stages": stages}


def mode_checks(core, thermo, v6, p82, fuel_args, phi: float, eta_b: float) -> dict:
    out: dict = {"phi_frozen_v6": phi, "eta_b": eta_b}

    # G1.1a: constant-cp stage fed the frozen v6 turbine inlet reproduces v6 T5, p5.
    legacy = v6.core.run_full_cycle(*fuel_args, phi, eta_b)
    comb, turb = legacy["combustor"], legacy["turbine"]
    m_turb = legacy["performance"]["total_mass_flow"] if "total_mass_flow" in legacy.get(
        "performance", {}) else None
    if m_turb is None:
        m_turb = legacy["total_mass_flow"]
    work = turb["work_total"]
    inlet = core.GasState(comb["T_out"], comb["p_out"], list(comb["Y_out"]))
    const = thermo.expand_for_work(core.MassStream(inlet, m_turb), work,
                                   p82.config.base.eta_polytropic, 50,
                                   comb["cp_out"], comb["R_out"])
    out["constant_cp_limit"] = {
        "v6_T5": turb["T"], "v6_p5": turb["p"], "stage_T5": const["T"], "stage_p5": const["P"],
        "T_relative": rel(const["T"], turb["T"]), "p_relative": rel(const["P"], turb["p"]),
        "energy_relative": const["energy_relative"],
    }
    out["constant_cp_limit"]["pass"] = (out["constant_cp_limit"]["T_relative"] <= TOL_LIMIT and
                                        out["constant_cp_limit"]["p_relative"] <= TOL_LIMIT and
                                        const["energy_relative"] <= TOL_CLOSURE)

    # G1.1b: A2 with zero cooling equals A1 at the turbine inlet interface.
    set_config(p82, ngv_fraction=0.0641, rotor_fraction=0.0275, pressure_steps=50)
    a1 = p82.run_full_cycle(*fuel_args, phi, eta_b, 1)
    set_config(p82, ngv_fraction=0.0, rotor_fraction=0.0)
    zero = p82.run_full_cycle(*fuel_args, phi, eta_b, 2)
    s1, s0 = a1["stations"]["dilution_exit"], zero["stations"]["ngv_exit_hp_in"]
    y_abs = max(abs(a - b) for a, b in zip(s1["Y"], s0["Y"]))
    iface = {"T_relative": rel(s1["T"], s0["T"]), "P_relative": rel(s1["P"], s0["P"]),
             "mass_flow_relative": rel(s1["mass_flow"], s0["mass_flow"]),
             "Y_max_abs": y_abs}
    iface["pass"] = all(v <= TOL_LIMIT for v in iface.values())
    out["zero_cooling_interface"] = iface

    # G1.2: closure of every mix and stage, A1, A2 (cited cooling) and A2 zero cooling.
    set_config(p82, ngv_fraction=0.0641, rotor_fraction=0.0275, pressure_steps=50)
    a2 = p82.run_full_cycle(*fuel_args, phi, eta_b, 2)
    clos = {"A1": closure(a1), "A2": closure(a2), "A2_zero_cooling": closure(zero)}
    worst = max(max(c["max_mass_relative"], c["max_energy_relative"], c["max_element_relative"])
                for c in clos.values())
    out["closure"] = {"cases": clos, "worst_relative": worst, "pass": worst <= TOL_CLOSURE}
    out["performance"] = {name: {k: r["performance"][k] for k in ("thrust_kN", "tsfc_mg_per_Ns")
                                 if k in r.get("performance", {})}
                          for name, r in (("v6", legacy), ("A1", a1), ("A2", a2))}

    # G1.3: 50 -> 100 steps, full cycle at cited cooling. Record both values.
    set_config(p82, pressure_steps=100)
    a2_100 = p82.run_full_cycle(*fuel_args, phi, eta_b, 2)
    set_config(p82, pressure_steps=50)
    steps: dict = {}
    for name in ("HP", "IP", "LP"):
        s50, s100 = a2["stages"][name], a2_100["stages"][name]
        steps[name] = {"T_50": s50["T"], "T_100": s100["T"], "P_50": s50["P"], "P_100": s100["P"],
                       "T_relative": rel(s50["T"], s100["T"]),
                       "P_relative": rel(s50["P"], s100["P"])}
    worst_steps = max(max(s["T_relative"], s["P_relative"]) for s in steps.values())
    out["step_doubling"] = {"stages": steps, "worst_relative": worst_steps,
                            "pass": worst_steps < TOL_STEPS}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/phase8/p82_g1.json")
    a = ap.parse_args()
    if a.output.exists():
        ap.error(f"{a.output} exists; output is write-once")
    protected = protected_check()
    core = load_core()
    reg_path = ROOT / "outputs/phase7/p72_registration.json"
    fit_path = ROOT / "outputs/phase7/calibration_v6.json"
    rows_path = ROOT / "outputs/phase7/calibration_v6_rows.csv"
    reg = json.loads(reg_path.read_text())
    fit = json.loads(fit_path.read_text())
    rows = v5.load_rows([AE3], with_targets=True).set_index("Mode")
    frozen = pd.read_csv(rows_path)
    frozen = frozen[frozen["Unique ID"] == AE3].set_index("Mode")
    fuel = LocalFuelBlend("JetA_dooley2012", dict(reg["fuel"]["mole_fractions"]))

    modes = {}
    for mode in MODES:
        v6, p82 = setup(core, reg, fit, rows, mode)
        thermo = core.GasThermo(v6.mechanism)
        modes[mode] = mode_checks(core, thermo, v6, p82, v6.fuel_args(fuel),
                                  float(frozen.loc[mode, "phi"]),
                                  float(reg["fixed_central"]["eta_b"][mode]))
        print(mode, {k: v["pass"] for k, v in modes[mode].items() if isinstance(v, dict) and "pass" in v},
              flush=True)

    g0_diff = run("git", "diff", "--stat", G0_COMMIT, "--", *G0_SOURCES)
    checks = {
        "G1.1_constant_cp_and_interface": all(m["constant_cp_limit"]["pass"] and
                                              m["zero_cooling_interface"]["pass"]
                                              for m in modes.values()),
        "G1.2_closure": all(m["closure"]["pass"] for m in modes.values()),
        "G1.3_step_doubling": all(m["step_doubling"]["pass"] for m in modes.values()),
        "G1.4_protected_hashes": not protected["mismatches"],
        "G1.4_g0_sources_unchanged_since_G0": g0_diff == "",
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    mech = ROOT / "data" / "creck_c1c16_full.yaml"
    doc = {
        "gate": "P8.2 G1", "verdict": verdict, "checks": checks,
        "tolerances": {"limit": TOL_LIMIT, "closure": TOL_CLOSURE, "steps": TOL_STEPS},
        "modes": modes,
        "notes": [
            "Burner energy closure is definitional: heat rejection is the residual of "
            "reactant minus product enthalpy flux under the v6 eta_b convention, so it is "
            "reported, not counted as an independent closure check.",
            "G1.4 180-row recomputation is not in this record: the v6 translation units are "
            "checked byte-identical to the G0 PASS commit, the C++ parity tests run in pytest, "
            "and the protected hashes are checked. A fresh g0_parity run was not performed "
            "in this session.",
        ],
        "development_probe_P8.2-A2": DEVELOPMENT_PROBE,
        "provenance": {
            "git_sha": run("git", "rev-parse", "HEAD"),
            "dirty_source": run("git", "status", "--porcelain", "--", "cpp", "scripts", "tests"),
            "g0_commit": G0_COMMIT, "g0_source_diff": g0_diff,
            "mechanism_sha256": sha256(mech),
            "registration_sha256": sha256(reg_path), "calibration_v6_sha256": sha256(fit_path),
            "frozen_rows_sha256": sha256(rows_path),
            "cantera_cpp": "libcantera-devel 3.2.0 (catjet-cpp, cpp/conda-lock-osx-arm64.txt)",
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
    print(f"P8.2 G1: {verdict} {checks}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
