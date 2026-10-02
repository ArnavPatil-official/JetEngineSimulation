#!/usr/bin/env python3
"""Registered P8.1 benchmark (docs/phase8_registration.md section 4/P8-A1).

Each invocation measures one arm/variant/workload/worker setting.  The output
directory is created once and never reused.  Run arm 1 for a workload first;
other arms compare their warm-up and each timed result against its saved arm-1
records before a timing is counted.  Timed repeats always clear warm starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for path in (ROOT, ROOT / "scripts" / "optimization", ROOT / "scripts" / "phase8"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc  # noqa: E402

import lto_v5 as v5  # noqa: E402
import lto_v6  # noqa: E402
from v6_backend import make_model_v6  # noqa: E402
from v6_optimized import OptimizedModel  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "phase8" / "benchmark"
RTOL, ATOL = 1e-9, 1e-12
AE3 = "02P23RR126"


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(args, cwd=ROOT, check=True, text=True,
                              capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_check() -> dict:
    manifest = ROOT / "outputs/phase8/protected_sha256_phase8.json"
    entries = json.loads(manifest.read_text())
    bad = [name for name, digest in entries.items()
           if not (ROOT / name).is_file() or _sha256(ROOT / name) != digest]
    return {"manifest_sha256": _sha256(manifest), "n_protected": len(entries),
            "mismatches": bad}


def machine_info() -> dict:
    import cantera
    import scipy

    return {
        "host": platform.node(), "platform": platform.platform(),
        "processor": _command("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "logical_cores": os.cpu_count(),
        "memory_bytes": _command("sysctl", "-n", "hw.memsize"),
        "power": _command("pmset", "-g", "batt"),
        "python": platform.python_version(), "cantera": cantera.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        # the /usr/bin shim follows xcode-select (a broken Xcode.app here)
        "compiler": _command("/Library/Developer/CommandLineTools/usr/bin/clang++", "--version"),
        "developer_dir": "/Library/Developer/CommandLineTools",
        "sdk": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk",
        "cpp_compile_flags": "C++17; -ffp-contract=off; hidden symbols; static Cantera (cpp/CMakeLists.txt)",
    }


def inputs(points: int) -> dict:
    reg = lto_v6.load_registration_v6()
    split = v5.load_split()
    params = json.loads((ROOT / "outputs/phase7/calibration_v6.json").read_text())["params"]
    cal = v5.calibration_rows(split)
    held = v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True),
                            split["heldout_groups"])
    ae3 = v5.load_rows([AE3], with_targets=True)
    ae3 = ae3.loc[ae3["Mode"] == "TAKE-OFF"].iloc[0]
    if len(cal) != 93 or len(held) != 87:
        raise RuntimeError(f"registered row counts changed: {len(cal)} calibration, {len(held)} held-out")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
        unit = qmc.Sobol(d=4, scramble=True, seed=20260929).random(points)
    lo = np.array([v5.FIT_BOUNDS[k][0] for k in v5.FIT_ORDER])
    hi = np.array([v5.FIT_BOUNDS[k][1] for k in v5.FIT_ORDER])
    sobol = lo + unit * (hi - lo)
    return {"reg": reg, "split": split, "params": params, "cal": cal,
            "held": held, "ae3": ae3, "sobol": sobol,
            "fuel": lto_v6.fuel_composition(reg)}


def model_for(arm: int, variant: str, workers: int, data: dict, native: bool = False):
    if arm == 1 and variant == "a":
        return make_model_v6("python", n_workers=workers)
    if arm == 2 and variant in ("a", "b", "c"):
        return OptimizedModel(data["reg"]["fixed_central"],
                              data["split"]["heldout_models"], n_workers=workers,
                              fuel=data["fuel"], variant=variant)
    if arm == 3 and variant == "a" and not native:
        return make_model_v6("cpp", n_workers=1)
    # Arm 3 b/c (and the labelled arm-3a native run) use the one-thread native
    # pool, so their difference from each other is not Python dispatch.
    if arm == 3:
        return NativeModel(data, n_threads=1, variant=variant)
    if arm == 4:
        return NativeModel(data, n_threads=11, variant=variant)
    raise ValueError(f"arm {arm} variant {variant} is not implemented; no timing will be labelled as it")


def ae3_task(data: dict, params: dict) -> tuple:
    row, fixed = data["ae3"], data["reg"]["fixed_central"]
    state = v5.mode_state(params, fixed, row["Pressure Ratio"],
                          row["Bypass Ratio"], row["Rated Thrust (kN)"], 1.0)
    return (state, fixed["eta_compressor"], fixed["eta_turbine_polytropic"],
            fixed["eta_b"]["TAKE-OFF"], row["Rated Thrust (kN)"], None, data["fuel"])


def row_tasks(rows: pd.DataFrame, data: dict) -> tuple[list[tuple], list[int]]:
    """Mirror V5Model.predict's unique-key task order and row expansion."""
    keys = list(zip(rows["Pressure Ratio"], rows["Bypass Ratio"],
                    rows["Rated Thrust (kN)"], rows["Mode"]))
    unique = list(dict.fromkeys(keys))
    positions = {key: i for i, key in enumerate(unique)}
    fixed = data["reg"]["fixed_central"]
    tasks = []
    for opr, bpr, rated, mode in unique:
        x = v5.MODE_X[mode]
        state = v5.mode_state(data["params"], fixed, opr, bpr, rated, x)
        tasks.append((state, fixed["eta_compressor"], fixed["eta_turbine_polytropic"],
                      fixed["eta_b"][mode], x * rated, None, data["fuel"]))
    return tasks, [positions[key] for key in keys]


def count_task(task: tuple) -> dict:
    """Untimed per-solve cycle count on the unchanged worker engine."""
    from integrated_engine import FUEL_LIBRARY, LocalFuelBlend, ThrustTargetUnreachable

    state, eta_c, eta_poly, eta_b, target, _guess, fuel = task
    blend = FUEL_LIBRARY[fuel] if isinstance(fuel, str) else LocalFuelBlend("blend", dict(fuel))
    engine = v5._ENGINE
    engine.design_point.update(state)
    engine.compressor.eta_c = eta_c
    engine.turbine_design["eta_polytropic"] = eta_poly
    n_actual = 0
    original = engine.run_full_cycle

    def counted(*args, **kwargs):
        nonlocal n_actual
        n_actual += 1
        return original(*args, **kwargs)

    engine.run_full_cycle = counted
    try:
        try:
            result = engine.run_at_thrust(target, blend, combustor_efficiency=eta_b, phi_guess=None)
            n_reported = result["thrust_match"]["n_cycle_evaluations"]
            return {"status": "converged", "actual_cycle_calls": n_actual or n_reported,
                    "reported_unique_evaluations": n_reported}
        except ThrustTargetUnreachable:
            return {"status": "unreachable", "actual_cycle_calls": n_actual or None,
                    "reported_unique_evaluations": None}
    finally:
        engine.run_full_cycle = original


def summarize_counts(counts: list[dict]) -> dict:
    observed = [c["actual_cycle_calls"] for c in counts if c["actual_cycle_calls"] is not None]
    return {"n_solves": len(counts), "n_counted": len(observed),
            "n_unreachable_without_count": sum(c["status"] == "unreachable" and
                                                c["actual_cycle_calls"] is None for c in counts),
            "min": min(observed) if observed else None,
            "median": statistics.median(observed) if observed else None,
            "max": max(observed) if observed else None,
            "mean": statistics.mean(observed) if observed else None,
            "counts": counts}


class NativeModel:
    """Benchmark-only reusable C++ std::thread pool with prebuilt cold tasks."""

    CONFIG_FIELDS = ("mass_flow_core", "bypass_ratio", "fpr", "eta_fan", "pi_c",
                     "combustor_pressure_loss", "combustor_heat_loss_fraction",
                     "combustor_air_fraction", "A_combustor_exit", "A_nozzle_exit",
                     "P_ambient", "T_ambient", "eta_c", "eta_polytropic",
                     "nox_A", "nox_B", "nox_C")

    def __init__(self, data: dict, n_threads: int, variant: str = "a"):
        from integrated_engine import LocalFuelBlend
        from simulation.catjet_backend import CppEngine

        build = ROOT / "cpp/build"
        if str(build) not in sys.path:
            sys.path.insert(0, str(build))
        import catjet_benchmark

        adapter = CppEngine(nox_fit_exclude_models=data["split"]["heldout_models"])
        fuel = LocalFuelBlend("blend", data["fuel"])
        fuel_string, fuel_species = adapter.fuel_args(fuel)
        self.pool = catjet_benchmark.NativePool(adapter.mechanism, fuel_string,
                                               fuel_species, n_threads, variant)
        self.n_threads = n_threads
        self.jobs = {}
        self.row_positions = {}
        tasks = {"W1": [ae3_task(data, data["params"])]}
        for workload, rows in (("W2", data["held"]), ("W3", data["cal"])):
            tasks[workload], self.row_positions[workload] = row_tasks(rows, data)
        tasks["W4"] = [ae3_task(data, dict(zip(v5.FIT_ORDER, point)))
                       for point in data["sobol"]]
        for workload, seq in tasks.items():
            jobs = []
            for state, eta_c, eta_poly, eta_b, target, _guess, _fuel in seq:
                adapter.design_point.update(state)
                adapter.compressor.eta_c = eta_c
                adapter.turbine_design["eta_polytropic"] = eta_poly
                adapter.push_config()
                config = adapter.core.config
                jobs.append({"config": {key: getattr(config, key) for key in self.CONFIG_FIELDS},
                             "target_kN": target, "eta_b": eta_b})
            self.jobs[workload] = jobs
        del adapter

    def evaluate(self, workload: str, data: dict) -> tuple[list[dict], dict]:
        unique = list(self.pool.run_many(self.jobs[workload]))
        if workload in ("W2", "W3"):
            records = [unique[i] for i in self.row_positions[workload]]
            rows = data["held"] if workload == "W2" else data["cal"]
            pred = pd.DataFrame(records, index=rows.index)
            if workload == "W2":
                return records, {"heldout_group_weighted_mape_pct": v5.weighted_mape(
                    pred["ff"], rows, pred["status"])}
            r = v5.residual_vector(pred["ff"], rows, pred["status"])
            return records, {"calibration_objective_sse": float(r @ r)}
        if workload == "W4":
            return unique, {"unreachable": sum(r["status"] == "unreachable" for r in unique)}
        return unique, {}

    def close(self):
        del self.pool

    def cycle_counts(self, workload: str) -> list[dict]:
        unique = list(self.pool.run_many(self.jobs[workload]))
        # "cycle_calls" (copied engine) counts failed probes too; for V6Engine
        # variant b a failed probe's evaluations are not observable here.
        return [{"status": row["status"],
                 "actual_cycle_calls": row.get("cycle_calls",
                                               None if row.get("probe_failed") or row["status"] != "converged"
                                               else row.get("n_cycle_evaluations")),
                 "reported_unique_evaluations": row.get("n_cycle_evaluations"),
                 "probe_failed": row.get("probe_failed")}
                for row in unique]


def run_workload(model, workload: str, data: dict) -> tuple[list[dict], dict]:
    if isinstance(model, NativeModel):
        return model.evaluate(workload, data)
    model.guess.clear()  # every timed repeat is cold
    if workload == "W1":
        records = list(model.pool.map(v5.solve_task, [ae3_task(data, data["params"])]))
        return records, {}
    if workload in ("W2", "W3"):
        rows = data["held"] if workload == "W2" else data["cal"]
        pred = model.predict(data["params"], rows)
        records = pred.to_dict("records")
        if workload == "W2":
            score = {"heldout_group_weighted_mape_pct": v5.weighted_mape(
                pred["ff"], rows, pred["status"])}
        else:
            r = v5.residual_vector(pred["ff"], rows, pred["status"])
            score = {"calibration_objective_sse": float(r @ r)}
        return records, score
    if workload == "W4":
        tasks = [ae3_task(data, dict(zip(v5.FIT_ORDER, point))) for point in data["sobol"]]
        records = list(model.pool.map(v5.solve_task, tasks))
        return records, {"unreachable": sum(r["status"] == "unreachable" for r in records)}
    raise ValueError(workload)


def _numeric_equal(a, b, rtol=RTOL, atol=ATOL) -> bool:
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return (math.isnan(af) and math.isnan(bf)) or math.isclose(af, bf, rel_tol=rtol, abs_tol=atol)


def compare(records: list[dict], baseline: list[dict], variant: str) -> dict:
    if len(records) != len(baseline):
        return {"match": False, "detail": "row count", "n": len(records), "reference_n": len(baseline)}
    differences = []
    max_rel = 0.0
    for i, (row, ref) in enumerate(zip(records, baseline)):
        if row.get("status") != ref.get("status"):
            differences.append({"row": i, "field": "status", "actual": row.get("status"),
                                "reference": ref.get("status")})
            continue
        # Scientific prediction fields are compared; the bracket's wording is
        # diagnostic and is allowed to differ in variant b.
        fields = (("ff", "T4") if variant == "c" else
                  ("ff", "phi", "thrust_kN", "tsfc_mg_Ns", "T3", "T4", "T5",
                   "p3_bar", "thrust_core_kN", "thrust_bypass_kN", "m_core",
                   "pi_c", "nox_corr_g_s"))
        for field in fields:
            a, b = row.get(field), ref.get(field)
            if a is None and b is None:
                continue
            if variant == "c" and field == "ff":
                equal = _numeric_equal(a, b, rtol=1e-4, atol=0.0)
            elif variant == "c" and field == "T4":
                equal = _numeric_equal(a, b, rtol=0.0, atol=0.1)
            else:
                equal = _numeric_equal(a, b)
            if not equal:
                if len(differences) < 30:
                    differences.append({"row": i, "field": field, "actual": a, "reference": b})
            elif a is not None and b is not None and not math.isnan(float(a)):
                max_rel = max(max_rel, abs(float(a) - float(b)) / max(abs(float(b)), ATOL))
        if variant == "a" and row.get("reason") != ref.get("reason") and len(differences) < 30:
            differences.append({"row": i, "field": "reason", "actual": row.get("reason"),
                                "reference": ref.get("reason")})
    return {"match": not differences, "n": len(records), "max_relative_difference": max_rel,
            "differences": differences}


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _power_source() -> str | None:
    out = _command("pmset", "-g", "ps")
    return out.splitlines()[0] if out else None


def _write_line(path: Path, obj: dict) -> None:
    obj = {"t_unix": time.time(), **obj}
    with path.open("a") as f:
        f.write(json.dumps(obj, default=_json_safe, allow_nan=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _reference(out: Path, workload: str, workers: int) -> dict | None:
    for n in (workers, 1, 11):
        path = out / f"arm1a_{workload}_{n}w" / "result.json"
        if path.is_file():
            doc = json.loads(path.read_text())
            if doc.get("verdict") == "PASS":
                return doc
    return None


def _rss_bytes(usage) -> int:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    return int(usage.ru_maxrss) * (1024 if sys.platform.startswith("linux") else 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=int, choices=(1, 2, 3, 4), required=True)
    ap.add_argument("--variant", choices=("a", "b", "c"), default="a")
    ap.add_argument("--workload", choices=("W1", "W2", "W3", "W4"), required=True)
    ap.add_argument("--workers", type=int, choices=(1, 11), default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--native", action="store_true",
                    help="arm 3 variant a on the one-thread native pool (labelled supplementary run)")
    a = ap.parse_args()
    if a.arm in (3, 4) and a.workers != 1 and a.arm == 3:
        ap.error("arm 3 is the one-thread C++ core")
    if a.workload == "W1" and a.workers != 1:
        ap.error("W1 is a single-solve latency workload")
    if a.repeats < 1 or a.points < 1:
        ap.error("repeats and points must be positive")
    if a.arm == 1 and a.variant != "a":
        ap.error("arm 1 has variant a only")
    if a.native and not (a.arm == 3 and a.variant == "a"):
        ap.error("--native labels only the arm-3 variant-a supplementary run")
    if a.arm in (3, 4) and a.variant in ("b", "c"):
        gate = ROOT / "outputs/phase8/benchmark/native_variants_gate.json"
        record = json.loads(gate.read_text()) if gate.is_file() else {}
        from verify_native_variants import source_sha256 as native_sha
        if record.get("verdict_" + a.variant) != "PASS" or record.get("source_sha256") != native_sha():
            ap.error(f"native variant {a.variant} must pass its registered 180-row/AE3 gate "
                     "on the current native sources first")
    if a.variant == "c" and a.arm == 2:
        # arm2c_gate.json predates a later uncommitted edit of v6_optimized.py;
        # only a gate recording the current source hash certifies variant c.
        from verify_c import source_sha256
        gate = ROOT / "outputs/phase8/benchmark/arm2c_gate_rev2.json"
        record = json.loads(gate.read_text()) if gate.is_file() else {}
        if record.get("verdict") != "PASS" or record.get("source_sha256") != source_sha256():
            ap.error("variant c's registered 180-row/AE3 approximation gate must pass "
                     "on the current v6_optimized.py first")
    if a.arm == 4:
        try:
            build = ROOT / "cpp/build"
            if str(build) not in sys.path:
                sys.path.insert(0, str(build))
            __import__("catjet_benchmark")
        except ImportError as exc:
            ap.error(f"build the registered std::thread batch module before arm 4: {exc}")

    run_name = f"arm{a.arm}{a.variant}{'-native' if a.native else ''}_{a.workload}_{a.workers}w"
    run_dir = a.out_dir / run_name
    if run_dir.exists():
        ap.error(f"{run_dir} exists; output is write-once")
    protected = protected_check()
    if protected["mismatches"]:
        ap.error(f"protected hashes changed: {protected['mismatches'][:5]}")
    data = inputs(a.points)
    ref = None if a.arm == 1 else _reference(a.out_dir, a.workload, a.workers)
    if a.arm != 1 and ref is None:
        ap.error("run arm 1 for this workload first; no validated reference result found")

    run_dir.mkdir(parents=True)
    manifest = {
        "registration": "docs/phase8_registration.md section 4, amendment P8-A1",
        "arm": a.arm, "variant": a.variant, "workload": a.workload,
        "workers": a.workers, "repeats": a.repeats, "warmups": 1,
        "implementation": ("python v6" if a.arm == 1 else "optimized python" if a.arm == 2
                           else "C++ via python worker" if a.arm == 3 and a.variant == "a" and not a.native
                           else f"C++ native std::thread pool, {1 if a.arm == 3 else 11} thread(s)"),
        "sobol_points": a.points if a.workload == "W4" else None,
        "registered_protocol": a.repeats == 5 and a.points == 1000,
        "git_sha": _command("git", "rev-parse", "HEAD"),
        "dirty_source": _command("git", "status", "--porcelain", "--", "cpp", "simulation", "scripts"),
        "machine": machine_info(), "protected": protected,
        "reference": None if ref is None else f"arm1a_{a.workload}",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_safe) + "\n")
    progress = run_dir / "progress.jsonl"
    samples = []
    first_records = None
    comparisons = []
    count_summary = None
    verdict = "PASS"
    _write_line(progress, {"phase": "model_create_start", "power": _power_source()})
    model = model_for(a.arm, a.variant, a.workers, data, native=a.native)
    _write_line(progress, {"phase": "model_created"})
    try:
        for rep in range(a.repeats + 1):
            load_before = os.getloadavg()
            t0 = time.perf_counter()
            records, score = run_workload(model, a.workload, data)
            elapsed = time.perf_counter() - t0
            load_after = os.getloadavg()
            if first_records is None:
                first_records = records
            internal = compare(records, first_records, "a")
            external = None if ref is None else compare(records, ref["records"], a.variant)
            ok = internal["match"] and (external is None or external["match"])
            if not ok:
                verdict = "FAIL"
            comparisons.append({"repeat": rep, "internal": internal, "reference": external})
            if rep:
                samples.append(elapsed)
            _write_line(progress, {"repeat": rep, "warmup": rep == 0, "wall_s": elapsed,
                                   "loadavg_before": load_before, "loadavg_after": load_after,
                                   "power": _power_source(),
                                   "score": score, "output_match": ok,
                                   "n_unreachable": sum(r["status"] == "unreachable" for r in records)})
            print(f"{run_name} {'warmup' if rep == 0 else f'repeat {rep}'}: "
                  f"{elapsed:.3f}s, output {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                break
        if verdict == "PASS":
            _write_line(progress, {"phase": "untimed_cycle_counts_start"})
            if isinstance(model, NativeModel):
                counts = model.cycle_counts(a.workload)
            else:
                if a.workload == "W1":
                    tasks = [ae3_task(data, data["params"])]
                elif a.workload in ("W2", "W3"):
                    tasks, _ = row_tasks(data["held" if a.workload == "W2" else "cal"], data)
                else:
                    tasks = [ae3_task(data, dict(zip(v5.FIT_ORDER, point)))
                             for point in data["sobol"]]
                counts = list(model.pool.map(count_task, tasks))
            count_summary = summarize_counts(counts)
            _write_line(progress, {"phase": "untimed_cycle_counts", "n_solves": len(counts),
                                   "median": count_summary["median"]})
    finally:
        model.close()
    parent_rss = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF))
    child_rss = _rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN))
    memory = {"parent_peak_bytes": parent_rss, "max_child_peak_bytes": child_rss,
              "upper_bound_bytes": parent_rss + a.workers * child_rss,
              "formula": "parent ru_maxrss + workers * RUSAGE_CHILDREN ru_maxrss; upper bound"}
    n_rows = len(first_records)
    n_solves = (len(row_tasks(data["held" if a.workload == "W2" else "cal"], data)[0])
                if a.workload in ("W2", "W3") else n_rows)
    final_verdict = "FAIL" if verdict == "FAIL" else ("PASS" if len(samples) == a.repeats else "INCOMPLETE")
    # registered protocol: mains power for every timed repeat (the queue only checks at start)
    powers = [json.loads(line).get("power") for line in progress.read_text().splitlines()]
    power_ok = all(p is None or "AC Power" in p for p in powers)
    if not power_ok and final_verdict == "PASS":
        final_verdict = "INVALID_POWER"
    result = {**manifest, "verdict": final_verdict,
              "records": first_records, "comparisons": comparisons,
              "timings_s": samples, "median_s": statistics.median(samples) if samples else None,
              "range_s": [min(samples), max(samples)] if samples else None,
              "solves_per_s": n_solves / statistics.median(samples) if samples else None,
              "rows_per_s": n_rows / statistics.median(samples) if samples else None,
              "n_solves": n_solves, "n_rows": n_rows, "memory": memory,
              "cycle_evaluations_per_solve": count_summary,
              "power_ok": power_ok,
              "cprofile": "outputs/phase8/profile_v6_ae3_takeoff.json"}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=_json_safe,
                                                    allow_nan=True) + "\n")
    print(f"{run_name}: {result['verdict']}, median {result['median_s']} s, "
          f"{result['solves_per_s']} solves/s", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
