#!/usr/bin/env python3
"""
P8.0: cProfile of the Python v6 matched-thrust solve (diagnostic, not a benchmark arm).

One AE3 take-off solve (record 02P23RR126, 310.9 kN) at the frozen v6
parameters, fixed central values and Dooley 2012 Jet A fuel, through the same
worker path the v6 runs use (lto_v5._init_worker + solve_task, phi_guess None).
Two profiles:
  first  - the first solve in a freshly built worker engine (includes the
           one-time Cantera Solution parsing in _fresh_solutions);
  steady - a second identical solve after it (what every later row costs).
The solved fuel flow is checked against the frozen calibration_v6_rows.csv row
(rtol 1e-9) so the profile is of the reference path. Outputs are write-once.

Usage: .venv/bin/python scripts/phase8/profile_v6.py
"""

import contextlib
import cProfile
import io
import json
import platform
import pstats
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))

import lto_v5 as v5  # noqa: E402
import lto_v6  # noqa: E402

OUT = ROOT / "outputs" / "phase8"
AE3 = "02P23RR126"
MODE = "TAKE-OFF"


def bucket(filename: str, func: str) -> str:
    """cProfile cannot see Cantera's compiled (Cython) methods: their time is
    counted as tottime of the Python function that calls them. The buckets say so."""
    if filename.endswith("combustor.py") and func == "_fresh_solutions":
        return "combustor._fresh_solutions: Cantera Solution parsing (one-time per worker)"
    if filename.endswith("combustor.py") and func == "run":
        return "combustor.run: Cantera equilibrate/property calls (compiled, not profiled separately)"
    if filename.endswith("integrated_engine.py") and func == "_calculate_fuel_air_ratio":
        return "integrated_engine._calculate_fuel_air_ratio (first call builds a Solution)"
    if filename.endswith("integrated_engine.py"):
        return "integrated_engine.py other Python"
    if "/simulation/" in filename:
        return "simulation/ other Python"
    if "scipy" in filename or "_brentq" in func:
        return "scipy"
    if "numpy" in filename:
        return "numpy"
    if "builtins.print" in func:
        return "print"
    return "other"


def time_equilibrate(combustor, n: int = 25) -> float:
    """Median wall time of one HP equilibrate on the combustor's own Solution at
    the AE3 take-off inlet (after the profiled solves), to split combustor.run."""
    gas, pristine = combustor._solutions[0]
    ts = []
    for _ in range(n):
        gas.state = pristine
        gas.TP = 901.545, 43.7724e5
        gas.set_equivalence_ratio(0.3388, "NC12H26", "O2:1.0, N2:3.76")
        t0 = time.perf_counter()
        gas.equilibrate("HP")
        ts.append(time.perf_counter() - t0)
    return float(sorted(ts)[n // 2])


def summarize(prof: cProfile.Profile, wall_s: float) -> dict:
    st = pstats.Stats(prof)
    total = st.total_tt
    by = {}
    for (fn, _line, func), (_cc, _nc, tt, _ct, _callers) in st.stats.items():
        b = bucket(fn, func)
        by[b] = by.get(b, 0.0) + tt
    top = sorted(st.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:15]
    return {
        "wall_s": wall_s,
        "profiled_total_s": total,
        "share_by_bucket": {k: v / total for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
        "top15_tottime": [{"func": f"{Path(fn).name}:{line}({func})", "ncalls": nc,
                           "tottime_s": tt, "cumtime_s": ct}
                          for (fn, line, func), (_cc, nc, tt, ct, _c) in top],
    }


def report(prof: cProfile.Profile) -> str:
    s = io.StringIO()
    st = pstats.Stats(prof, stream=s)
    st.sort_stats("cumulative").print_stats(40)
    st.sort_stats("tottime").print_stats(40)
    return s.getvalue()


def main() -> int:
    names = {k: OUT / f"profile_v6_ae3_takeoff_{k}.txt" for k in ("first", "steady")}
    summary_path = OUT / "profile_v6_ae3_takeoff.json"
    for p in [*names.values(), summary_path]:
        if p.exists():
            print(f"{p.relative_to(ROOT)} exists; refusing to overwrite")
            return 1

    reg6 = lto_v6.load_registration_v6()
    split = v5.load_split()
    fixed = reg6["fixed_central"]
    fit = json.loads((ROOT / "outputs" / "phase7" / "calibration_v6.json").read_text())
    params = fit["params"]
    fuel = lto_v6.fuel_composition(reg6)
    row = v5.load_rows([AE3], with_targets=True).set_index("Mode").loc[MODE]
    x = v5.MODE_X[MODE]
    state = v5.mode_state(params, fixed, row["Pressure Ratio"], row["Bypass Ratio"],
                          row["Rated Thrust (kN)"], x)
    task = (state, fixed["eta_compressor"], fixed["eta_turbine_polytropic"], fixed["eta_b"][MODE],
            x * row["Rated Thrust (kN)"], None, fuel)
    frozen = __import__("pandas").read_csv(ROOT / "outputs" / "phase7" / "calibration_v6_rows.csv")
    ff_frozen = float(frozen[(frozen["Unique ID"] == AE3) & (frozen["Mode"] == MODE)]["ff"].iloc[0])

    v5._init_worker(split["heldout_models"])
    results, summaries = {}, {}
    for kind in ("first", "steady"):
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            prof.enable()
            res = v5.solve_task(task)
            prof.disable()
        wall = time.perf_counter() - t0
        if res["status"] != "converged":
            raise RuntimeError(f"AE3 take-off did not converge: {res}")
        rel = abs(res["ff"] / ff_frozen - 1.0)
        if rel > 1e-9:
            raise RuntimeError(f"ff {res['ff']!r} differs from frozen {ff_frozen!r} (rel {rel:.2e})")
        results[kind] = res
        summaries[kind] = summarize(prof, wall) | {"ff_kg_s": res["ff"], "ff_rel_diff_vs_frozen": rel}
        names[kind].write_text(report(prof))

    eq_s = time_equilibrate(v5._ENGINE.combustor_creck)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    import cantera
    import numpy
    import scipy
    out = {
        "stage": "P8.0 cProfile (diagnostic; not a benchmark arm)",
        "registration": "docs/phase8_registration.md",
        "commit": sha,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version(), "cantera": cantera.__version__,
                    "numpy": numpy.__version__, "scipy": scipy.__version__},
        "workload": f"AE3 ({AE3}) {MODE}, target {x * row['Rated Thrust (kN)']} kN, frozen v6 "
                    "parameters and fuel, phi_guess None, in-process worker path",
        "ff_frozen_kg_s": ff_frozen,
        "profiles": summaries,
        "equilibrate_HP_median_s": eq_s,
        "note": "cProfile does not see Cantera's compiled methods; their time is inside "
                "combustor.run and _fresh_solutions. equilibrate_HP_median_s times one HP "
                "equilibrate (n-dodecane stand-in, CRECK, at the AE3 take-off inlet, phi 0.3388) on the "
                "combustor's own Solution, so combustor.run time / cycles can be compared with it.",
        "files": {k: str(p.relative_to(ROOT)) for k, p in names.items()},
    }
    summary_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"one HP equilibrate: {1e3 * eq_s:.2f} ms")
    for kind, s in summaries.items():
        print(f"{kind}: wall {s['wall_s']:.3f} s; " + ", ".join(
            f"{k} {100 * v:.1f} %" for k, v in s["share_by_bucket"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
