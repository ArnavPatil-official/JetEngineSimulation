#!/usr/bin/env python3
"""
P8.1 step 2: C++ Cantera (conda libcantera 3.2.0) vs Python Cantera (.venv 3.2.0)
HP equilibrium parity at 1e-12 (docs/plan.md P8.1).

Both sides mirror simulation/combustor/combustor.py: a fresh Solution of the
CRECK mechanism, TP, set_equivalence_ratio with oxidizer "O2:1.0, N2:3.76",
equilibrate("HP"). Fuel: the registered v6 Dooley 2012 Jet A surrogate
(outputs/phase7/p72_registration.json). Cases: the AE3 take-off combustor
inlet at the v6 solution, plus approach- and idle-like inlets.

Pass rule (plan: "match Python to 1e-12"): |dX| <= 1e-12 |X| for T, P,
h_mass, cp_mass, mean_mw, and |dY_k| <= 1e-12 |Y_k| + 1e-18 for every species.
Attempt 1 (outputs/phase8/p81_hello_equilibrium_attempt1_exactP.json) also
required P to match exactly; equilibrate("HP") returns P as solver output
(4377239.99999957 vs 4377239.999999568 at AE3, 5e-16 relative), so that
stricter check failed while every quantity was within 1e-13. Kept as evidence.

Usage: .venv/bin/python scripts/phase8/hello_equilibrium_check.py
Output (write-once): outputs/phase8/p81_hello_equilibrium.json
"""

import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import cantera as ct

ROOT = Path(__file__).resolve().parent.parent.parent
BIN = ROOT / "cpp" / "build" / "hello_equilibrium"
MECH = "data/creck_c1c16_full.yaml"
OUT = ROOT / "outputs" / "phase8" / "p81_hello_equilibrium.json"
RTOL, YATOL = 1e-12, 1e-18
CASES = [   # (label, T_in K, p_in Pa, phi)
    ("AE3 take-off (v6 solution)", 901.5451512272988, 43.7724e5, 0.33879271456691074),
    ("approach-like", 700.0, 20.0e5, 0.25),
    ("idle-like", 500.0, 5.0e5, 0.15),
]


def fuel_string() -> str:
    reg = json.loads((ROOT / "outputs" / "phase7" / "p72_registration.json").read_text())
    return ", ".join(f"{k}:{v!r}" for k, v in reg["fuel"]["mole_fractions"].items())


def python_side(T, p, phi, fuel) -> dict:
    gas = ct.Solution(str(ROOT / MECH))
    gas.TP = T, p
    gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer="O2:1.0, N2:3.76")
    gas.equilibrate("HP")
    return {"T": gas.T, "P": gas.P, "h_mass": gas.enthalpy_mass, "cp_mass": gas.cp_mass,
            "mean_mw": gas.mean_molecular_weight, "Y": dict(zip(gas.species_names, gas.Y))}


def cpp_side(T, p, phi, fuel) -> dict:
    out = subprocess.run([str(BIN), str(ROOT / MECH), repr(T), repr(p), repr(phi), fuel],
                         capture_output=True, text=True, check=True).stdout
    res = {"Y": {}}
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "Y":
            res["Y"][parts[1]] = float(parts[2])
        else:
            res[parts[0]] = float(parts[1])
    return res


def compare(py: dict, cc: dict) -> dict:
    scal = {}
    ok = list(py["Y"]) == list(cc["Y"])
    for k in ("T", "P", "h_mass", "cp_mass", "mean_mw"):
        rel = abs(cc[k] - py[k]) / abs(py[k])
        scal[k] = {"python": py[k], "cpp": cc[k], "rel_diff": rel}
        ok &= rel <= RTOL
    worst, worst_k = 0.0, None
    for k, y in py["Y"].items():
        excess = abs(cc["Y"][k] - y) / (RTOL * abs(y) + YATOL)
        if excess > worst:
            worst, worst_k = excess, k
    ok &= worst <= 1.0
    return {"pass": bool(ok), "scalars": scal,
            "Y_worst_ratio_to_tolerance": worst, "Y_worst_species": worst_k,
            "Y_max_abs_diff": max(abs(cc["Y"][k] - y) for k, y in py["Y"].items())}


def main() -> int:
    if OUT.exists():
        print(f"{OUT.relative_to(ROOT)} exists; refusing to overwrite")
        return 1
    logging.getLogger("cantera").setLevel(logging.ERROR)
    fuel = fuel_string()
    results = []
    for label, T, p, phi in CASES:
        r = compare(python_side(T, p, phi, fuel), cpp_side(T, p, phi, fuel))
        results.append({"case": label, "T_in_K": T, "p_in_Pa": p, "phi": phi} | r)
        print(f"{label}: {'PASS' if r['pass'] else 'FAIL'}  T rel {r['scalars']['T']['rel_diff']:.1e}, "
              f"worst Y {r['Y_worst_species']} at {r['Y_worst_ratio_to_tolerance']:.2e} x tol")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    clang = subprocess.run(["/Library/Developer/CommandLineTools/usr/bin/clang++", "--version"],
                           capture_output=True, text=True).stdout.splitlines()[0]
    doc = {
        "stage": "P8.1 step 2: C++ vs Python HP equilibrium parity",
        "commit": sha, "mechanism": MECH, "fuel": fuel,
        "rule": f"|dX| <= {RTOL} |X| for T, P, h, cp, mean_mw; |dY| <= {RTOL} |Y| + {YATOL}",
        "attempt_1": "outputs/phase8/p81_hello_equilibrium_attempt1_exactP.json (failed only the "
                     "exact-P check; P rel diff 5e-16)",
        "python_cantera": ct.__version__, "python": platform.python_version(),
        "cpp": {"compiler": clang, "sdk": "MacOSX26.5.sdk", "libcantera": "3.2.0 (conda-forge, "
                "cpp/conda-lock-osx-arm64.txt)"},
        "all_pass": all(r["pass"] for r in results),
        "cases": results,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print("ALL PASS" if doc["all_pass"] else "FAILED")
    return 0 if doc["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
