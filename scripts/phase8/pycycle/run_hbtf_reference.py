#!/usr/bin/env python3
"""
Phase 8, track D2: pyCycle high-bypass turbofan (HBTF) reference run.

Runs pyCycle's published HBTF example UNMODIFIED (om-pycycle 4.4.0, GitHub tag
4.4.0, files in envs/pycycle/upstream/, sha256 in SOURCES.json) and records it
as the future code-to-code check target for the project's C++ off-design
matching solver (P8.4). Nothing here feeds a calibration or validation set.

What runs
  A. "reference" case = the upstream regression test configuration: the
     upstream HBTFTestCase.setUp() and HBTFTestCase.benchmark_case1() are
     called in-process (one run_model of the MPhbtf multipoint model:
     DESIGN at MN 0.8 / 35 kft / Fn 5900 lbf / T4 2857 degR, OD_full_pwr at
     the same flight condition with T4 throttle, OD_part_pwr at PC = 0.8).
     A wrapper around openmdao Problem.run_model snapshots every station and
     element quantity right after the solve, before the upstream asserts.
  B. "example_main_sweep" = the example file itself executed as __main__
     (runpy; its flight-envelope sweep of 21 (MN, alt) pairs x PC 1/0.9/0.8/0.7
     plus the 1/0.85 throttle-back runs). The same wrapper records a
     performance summary and solver status after every run_model. Upstream
     skips this script in its own test suite ("Runtime is more than 25 min").
  C. The upstream test file itself, in subprocesses from a temporary directory:
     pytest through scripts/phase8/pycycle/upstream_benchmark_pytest_shim.py,
     and testflo -b (the runner upstream names its benchmark_* files for).

Every value the upstream test asserts is read from the upstream file by AST
(reference number, promoted path, tolerance), never retyped, and compared with
openmdao assert_near_equal semantics (relative error when the reference is
nonzero) -> JSON "upstream_test_comparison".

Must run in the catjet-pycycle env (versions are checked against the pins):
    ~/miniforge3/envs/catjet-pycycle/bin/python scripts/phase8/pycycle/run_hbtf_reference.py
Outputs (write-once; the script refuses to overwrite):
    outputs/phase8/pycycle_hbtf_reference.json
    outputs/phase8/pycycle_hbtf_reference.csv   (case A, long format, native + SI units)
Options: --out-prefix PATH (dry runs elsewhere), --no-sweep (skip case B; recorded).
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.metadata as md
import io
import json
import os
import platform
import re
import runpy
import subprocess
import sys
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "envs" / "pycycle" / "upstream"
SOURCES = UPSTREAM / "SOURCES.json"
EXAMPLE = UPSTREAM / "example_cycles" / "high_bypass_turbofan.py"
BENCH = UPSTREAM / "example_cycles" / "tests" / "benchmark_hbtf.py"
SHIM = Path(__file__).resolve().parent / "upstream_benchmark_pytest_shim.py"
DEFAULT_PREFIX = ROOT / "outputs" / "phase8" / "pycycle_hbtf_reference"

PINS = {"om-pycycle": "4.4.0", "openmdao": "3.41.0", "numpy": "1.26.4", "scipy": "1.14.1"}
REPORT_VERSIONS = ["om-pycycle", "openmdao", "numpy", "scipy", "pytest", "testflo", "parameterized",
                   "networkx", "requests", "packaging"]

POINTS = ["DESIGN", "OD_full_pwr", "OD_part_pwr"]
# flow stations exactly as the upstream viewer() lists them
STATIONS = ["fc.Fl_O", "inlet.Fl_O", "fan.Fl_O", "splitter.Fl_O1", "splitter.Fl_O2", "duct4.Fl_O",
            "lpc.Fl_O", "duct6.Fl_O", "hpc.Fl_O", "bld3.Fl_O", "burner.Fl_O", "hpt.Fl_O", "duct11.Fl_O",
            "lpt.Fl_O", "duct13.Fl_O", "core_nozz.Fl_O", "byp_bld.Fl_O", "duct15.Fl_O", "byp_nozz.Fl_O"]
STATION_Q = ["stat:W", "tot:P", "tot:T", "tot:h", "tot:S", "stat:P", "stat:T", "stat:MN", "stat:V", "stat:area"]
BLEEDS = ["hpc.cool1", "hpc.cool2", "hpc.cust", "bld3.cool3", "bld3.cool4", "byp_bld.bypBld"]
BLEED_Q = ["stat:W", "tot:P", "tot:T", "tot:h"]
ELEMENTS = {
    "compressor": (["fan", "lpc", "hpc"],
                   ["PR", "eff", "eff_poly", "Wc", "Nc", "power", "trq", "SMN", "SMW",
                    "map.RlineMap", "map.NcMap", "map.PRmap", "map.WcMap", "map.effMap", "map.map.alphaMap",
                    "s_Wc", "s_PR", "s_eff", "s_Nc"]),
    "turbine": (["hpt", "lpt"],
                ["PR", "eff", "eff_poly", "Wp", "Np", "power", "trq",
                 "map.NpMap", "map.PRmap", "map.alphaMap", "map.WpMap", "map.effMap",
                 "s_Wp", "s_PR", "s_eff", "s_Np"]),
    "burner": (["burner"], ["Wfuel", "dPqP", "Fl_I:FAR", "Fl_O:tot:T"]),
    "nozzle": (["core_nozz", "byp_nozz"],
               ["PR", "Cv", "Fg", "Ps_exhaust", "Throat:stat:area", "Throat:stat:MN", "Throat:stat:W",
                "Throat:stat:V", "Fl_O:stat:MN", "Fl_O:stat:V"]),
    "duct": (["duct4", "duct6", "duct11", "duct13", "duct15"], ["dPqP"]),
    "inlet": (["inlet"], ["ram_recovery", "F_ram"]),
    "splitter": (["splitter"], ["BPR"]),
    "shaft": (["lp_shaft", "hp_shaft"], ["Nmech", "trq_in", "trq_out", "trq_net", "pwr_in", "pwr_out",
                                         "pwr_net"]),
    "shaft_offtake": (["hp_shaft"], ["HPX"]),
    "flight_conditions": (["fc"], ["alt", "MN", "dTs", "W", "Fl_O:stat:P", "Fl_O:stat:T"]),
    "performance": (["perf"], ["Fn", "Fg", "TSFC", "OPR", "Wfuel", "ram_drag"]),
}
BLEED_FRAC_Q = {  # (element, bleed) -> fraction names
    ("hpc", "cool1"): ["frac_W", "frac_P", "frac_work"], ("hpc", "cool2"): ["frac_W", "frac_P", "frac_work"],
    ("hpc", "cust"): ["frac_W", "frac_P", "frac_work"], ("bld3", "cool3"): ["frac_W"],
    ("bld3", "cool4"): ["frac_W"], ("byp_bld", "bypBld"): ["frac_W"],
    ("hpt", "cool3"): ["frac_P"], ("hpt", "cool4"): ["frac_P"], ("lpt", "cool1"): ["frac_P"],
    ("lpt", "cool2"): ["frac_P"],
}
BALANCE = {"design": ["W", "FAR", "lpt_PR", "hpt_PR"],
           "off_design": ["FAR", "W", "BPR", "lp_Nmech", "hp_Nmech"]}

SI = {"lbm/s": "kg/s", "lbf/inch**2": "Pa", "psi": "Pa", "degR": "degK", "Btu/lbm": "J/kg",
      "Btu/lbm/degR": "J/(kg*degK)", "ft/s": "m/s", "inch**2": "m**2", "hp": "W", "ft*lbf": "N*m",
      "lbf": "N", "lbm/h/lbf": "g/(kN*s)", "lbm/hr/lbf": "g/(kN*s)", "ft": "m", "bar": "Pa",
      "lbm/ft**3": "kg/m**3"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail(msg):
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- preconditions
def check_versions():
    found = {}
    for name in REPORT_VERSIONS:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            found[name] = None
    bad = {k: (found.get(k), v) for k, v in PINS.items() if found.get(k) != v}
    if bad:
        fail(f"version pins not met (found, pinned): {bad}; run in the catjet-pycycle env")
    return found


def check_upstream():
    src = json.loads(SOURCES.read_text())
    out = []
    for f in src["files"]:
        got = sha256(UPSTREAM / f["path"])
        if got != f["sha256"]:
            fail(f"upstream file {f['path']} sha256 {got} != SOURCES.json {f['sha256']}")
        out.append({"path": f"envs/pycycle/upstream/{f['path']}", "url": f["url"], "sha256": got})
    return src, out


def parse_upstream_asserts():
    """(label, path, reference, tol, section) for every assert_near_equal in benchmark_case1."""
    tree = ast.parse(BENCH.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "benchmark_case1")
    body = fn.body
    try_node = next(n for n in body if isinstance(n, ast.Try))
    stmts = try_node.body
    tol = ref = path = None
    section = "DESIGN"
    out = []
    for s in stmts:
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
            name = s.targets[0].id
            if name == "tol":
                tol = ast.literal_eval(s.value)
            elif name == "reg_data":
                ref = ast.literal_eval(s.value)
            elif name == "pyc":
                # self.prob['<path>'][0]
                path = s.value.value.slice.value
        elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            call = s.value
            fname = getattr(call.func, "id", getattr(call.func, "attr", ""))
            if fname == "print" and call.args and isinstance(call.args[0], ast.Constant):
                txt = call.args[0].value
                if isinstance(txt, str) and txt.startswith("# "):
                    section = txt[2:].strip()
                elif isinstance(txt, str) and txt.endswith(":") and path is not None:
                    label = txt[:-1]
            if fname == "assert_near_equal":
                args = [a.id for a in call.args]
                if args != ["pyc", "reg_data", "tol"]:
                    fail(f"unexpected assert form {args}")
                out.append({"label": label, "path": path, "reference": ref, "tol": tol,
                            "point": path.split(".")[0], "upstream_section": section})
                path = ref = None
    if len(out) == 0:
        fail("no asserts parsed from upstream benchmark")
    return out


# ---------------------------------------------------------------- capture helpers
class Units:
    def __init__(self, prob):
        self.by_prom = {}
        meta_out = prob.model.get_io_metadata(iotypes=("output",), metadata_keys=["units"],
                                              get_remote=True)
        meta_in = prob.model.get_io_metadata(iotypes=("input",), metadata_keys=["units"],
                                             get_remote=True)
        for meta in (meta_in, meta_out):  # outputs win
            for absn, m in meta.items():
                self.by_prom[m["prom_name"]] = m.get("units")
                self.by_prom.setdefault(absn, m.get("units"))

    def get(self, name):
        return self.by_prom.get(name)


def getval(prob, units, name):
    from openmdao.utils.units import convert_units
    u = units.get(name)
    try:
        v = prob.get_val(name)
    except Exception:
        v = prob.get_val(name, units=u)
    v = float(v.ravel()[0])
    usi = SI.get(u) if u else None
    vsi = convert_units(v, u, usi) if usi else v
    return v, u, vsi, (usi if usi else u)


def point_mode(prob, pt):
    return "design" if prob.model._get_subsystem(pt).options["design"] else "off_design"


def snapshot_point(prob, units, pt):
    """Return (records, missing). records: list of dict(group, element, quantity, path, value, units, value_si, units_si).

    A name not found under the point is looked up as an MPCycle cycle parameter
    (pyc_add_cycle_param promotes e.g. 'hpc.cool1:frac_W' at model level, shared by all points),
    exactly as pycycle.viewers.get_val does."""
    recs, missing = [], []

    def add(group, element, quantity, path):
        if path not in units.by_prom:
            alt = path.split(".", 1)[1]
            if alt in units.by_prom:
                path = alt
            else:
                missing.append({"path": path, "error": "not in model"})
                return
        v, u, vsi, usi = getval(prob, units, path)
        recs.append({"group": group, "element": element, "quantity": quantity, "path": path,
                     "value": v, "units": u, "value_si": vsi, "units_si": usi})

    mode = point_mode(prob, pt)
    for st in STATIONS:
        for q in STATION_Q:
            add("station", st, q, f"{pt}.{st}:{q}")
    for b in BLEEDS:
        for q in BLEED_Q:
            add("bleed_flow", b, q, f"{pt}.{b}:{q}")
    for (el, bn), qs in BLEED_FRAC_Q.items():
        for q in qs:
            add("bleed_fraction", f"{el}.{bn}", q, f"{pt}.{el}.{bn}:{q}")
    for group, (els, qs) in ELEMENTS.items():
        for el in els:
            for q in qs:
                add(group, el, q, f"{pt}.{el}.{q}")
    for q in BALANCE[mode]:
        add("balance", "balance", q, f"{pt}.balance.{q}")
    for q in ["LP_Nmech", "HP_Nmech"]:
        add("spool", pt, q, f"{pt}.{q}")
    if mode == "design":
        for q in ["T4_MAX", "Fn_DES"]:
            add("target", pt, q, f"{pt}.{q}")
    else:
        if pt == "OD_full_pwr":
            add("target", pt, "T4_MAX", f"{pt}.T4_MAX")
        else:
            add("target", pt, "PC", f"{pt}.PC")
            add("target", pt, "Fn_max", f"{pt}.Fn_max")
    return recs, missing


NEWTON_HISTORY = {}  # system pathname -> residual norms seen by its Newton solver in the last solve


def solver_status(prob, pt):
    """Convergence of the point's Newton solve from the norms the solver itself evaluated
    (recorded by a pass-through wrapper on NewtonSolver._iter_get_norm)."""
    sysm = prob.model._get_subsystem(pt)
    nl = sysm.nonlinear_solver
    hist = list(NEWTON_HISTORY.get(sysm.pathname, []))
    it = int(nl._iter_count)
    atol, rtol = float(nl.options["atol"]), float(nl.options["rtol"])
    maxiter = int(nl.options["maxiter"])
    final = hist[-1] if hist else float("nan")
    init = hist[0] if hist else float("nan")
    return {"solver": type(nl).__name__, "linesearch": type(nl.linesearch).__name__ if nl.linesearch else None,
            "linear_solver": type(sysm.linear_solver).__name__,
            "atol": atol, "rtol": rtol, "maxiter": maxiter, "solve_subsystems": bool(nl.options["solve_subsystems"]),
            "iterations": it, "initial_residual_norm": init, "final_residual_norm": final,
            "norm_history": hist,
            "converged": bool(hist and len(hist) - 1 == it and (final <= atol or final / (init or 1.0) <= rtol))}


def summary_row(prob, units, pt):
    def g(path):
        return float(prob.get_val(path).ravel()[0])
    row = {
        "MN": g(f"{pt}.fc.Fl_O:stat:MN"), "alt_ft": g(f"{pt}.fc.alt"),
        "W_lbm_s": g(f"{pt}.inlet.Fl_O:stat:W"), "Fn_lbf": g(f"{pt}.perf.Fn"), "Fg_lbf": g(f"{pt}.perf.Fg"),
        "Fram_lbf": g(f"{pt}.inlet.F_ram"), "OPR": g(f"{pt}.perf.OPR"), "TSFC_lbm_h_lbf": g(f"{pt}.perf.TSFC"),
        "BPR": g(f"{pt}.splitter.BPR"), "FAR": g(f"{pt}.balance.FAR"), "Tt4_degR": g(f"{pt}.burner.Fl_O:tot:T"),
        "Wfuel_lbm_s": g(f"{pt}.perf.Wfuel"), "LP_Nmech_rpm": g(f"{pt}.LP_Nmech"),
        "HP_Nmech_rpm": g(f"{pt}.HP_Nmech"), "fan_PR": g(f"{pt}.fan.PR"), "lpc_PR": g(f"{pt}.lpc.PR"),
        "hpc_PR": g(f"{pt}.hpc.PR"), "hpt_PR": g(f"{pt}.hpt.PR"), "lpt_PR": g(f"{pt}.lpt.PR"),
        "fan_RlineMap": g(f"{pt}.fan.map.RlineMap"), "lpc_RlineMap": g(f"{pt}.lpc.map.RlineMap"),
        "hpc_RlineMap": g(f"{pt}.hpc.map.RlineMap"),
    }
    return row


class WarningTally:
    def __init__(self):
        self.counts = {}

    def showwarning(self, message, category, filename, lineno, file=None, line=None):
        key = (category.__name__, os.path.relpath(filename, sys.prefix) if filename.startswith(sys.prefix)
               else filename, lineno, str(message)[:160])
        self.counts[key] = self.counts.get(key, 0) + 1

    def table(self):
        rows = [{"category": k[0], "file": k[1], "line": k[2], "message": k[3], "count": v}
                for k, v in self.counts.items()]
        rows.sort(key=lambda r: (-r["count"], r["category"]))
        by_cat = {}
        for r in rows:
            by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["count"]
        return {"total_by_category": by_cat, "distinct": rows}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-prefix", default=str(DEFAULT_PREFIX))
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()
    out_json = Path(args.out_prefix + ".json")
    out_csv = Path(args.out_prefix + ".csv")
    for p in (out_json, out_csv):
        if p.exists():
            fail(f"{p} exists; outputs are write-once")

    versions = check_versions()
    sources, upstream_files = check_upstream()
    asserts = parse_upstream_asserts()
    t_start = time.time()

    workdir = Path(tempfile.mkdtemp(prefix="pycycle_hbtf_"))
    os.chdir(workdir)
    sys.path.insert(0, str(UPSTREAM))

    import numpy as np
    import openmdao
    import openmdao.api as om
    from openmdao.core.problem import Problem
    import pycycle

    tally = WarningTally()
    warnings.simplefilter("always")
    warnings.showwarning = tally.showwarning

    import example_cycles.tests.benchmark_hbtf as bench

    from openmdao.solvers.nonlinear.newton import NewtonSolver

    orig_norm = NewtonSolver._iter_get_norm

    def recording_norm(self):  # pass-through: returns the solver's own value unchanged
        n = orig_norm(self)
        path = self._system().pathname
        if path in POINTS:
            if self._iter_count == 0:
                NEWTON_HISTORY[path] = []
            NEWTON_HISTORY.setdefault(path, []).append(float(n))
        return n

    NewtonSolver._iter_get_norm = recording_norm

    orig_run_model = Problem.run_model
    state = {"mode": None, "snap": None, "sweep": [], "t": None}

    def hooked_run_model(self, *a, **k):
        t0 = time.perf_counter()
        r = orig_run_model(self, *a, **k)
        dt = time.perf_counter() - t0
        units = Units(self) if state.get("units_for") is not self else state["units"]
        state["units_for"], state["units"] = self, units
        if state["mode"] == "reference":
            snap = {}
            for pt in POINTS:
                recs, missing = snapshot_point(self, units, pt)
                snap[pt] = {"records": recs, "missing": missing, "solver": solver_status(self, pt),
                            "summary": summary_row(self, units, pt)}
            snap["_asserted"] = {a_["path"]: float(self.get_val(a_["path"]).ravel()[0]) for a_ in asserts}
            snap["_run_model_seconds"] = dt
            state["snap"] = snap
        elif state["mode"] == "sweep":
            n = len(state["sweep"])
            ent = {"run_index": n, "run_model_seconds": dt,
                   "OD_part_pwr_PC": float(self.get_val("OD_part_pwr.PC").ravel()[0])}
            for pt in POINTS:
                ent[pt] = {"summary": summary_row(self, units, pt), "solver": solver_status(self, pt)}
            state["sweep"].append(ent)
        return r

    Problem.run_model = hooked_run_model

    # ---------------- A. reference (upstream regression configuration)
    state["mode"] = "reference"
    tc = bench.HBTFTestCase("benchmark_case1")
    tc.setUp()
    buf = io.StringIO()
    upstream_method = {"method": "example_cycles.tests.benchmark_hbtf.HBTFTestCase.benchmark_case1 (in-process)"}
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            tc.benchmark_case1()
        upstream_method["result"] = "passed"
    except AssertionError as e:
        upstream_method["result"] = "failed"
        upstream_method["assertion"] = str(e)
    upstream_method["seconds"] = time.perf_counter() - t0
    stdout_a = buf.getvalue()
    upstream_method["stdout"] = stdout_a.strip().splitlines()
    snap = state["snap"]
    if snap is None:
        fail("reference run did not reach run_model")
    ref_prob = tc.prob

    comparison = []
    for a_ in asserts:
        actual = snap["_asserted"][a_["path"]]
        ref = a_["reference"]
        rel = abs(actual - ref) / abs(ref) if ref != 0 else abs(actual - ref)
        comparison.append({**a_, "actual": actual, "error_kind": "relative" if ref != 0 else "absolute",
                           "error": rel, "pass": bool(rel <= a_["tol"])})
    n_pass = sum(c["pass"] for c in comparison)

    # ---------------- B. example __main__ sweep (unmodified file run as a script)
    sweep_info = {"run": not args.no_sweep}
    if not args.no_sweep:
        state["mode"] = "sweep"
        sweep_dir = workdir / "sweep"
        sweep_dir.mkdir()
        os.chdir(sweep_dir)
        log_path = sweep_dir / "stdout.log"
        t0 = time.perf_counter()
        err = None
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            try:
                runpy.run_path(str(EXAMPLE), run_name="__main__")
            except Exception as e:  # noqa: BLE001 - recorded
                err = f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        os.chdir(workdir)
        log = log_path.read_text()
        view = sweep_dir / "hbtf_view.out"
        runs = state["sweep"]
        # the example's loop order: per (MN, alt): PC 1, .9, .8, .7 (viewer printed), then 1, .85
        flight_env = [(0.8, 35000), (0.7, 35000), (0.55, 35000), (0.46, 35000), (0.4, 35000),
                      (0.4, 20000), (0.6, 20000), (0.8, 20000),
                      (0.8, 10000), (0.6, 10000), (0.4, 10000), (0.2, 10000), (0.001, 10000),
                      (.001, 1000), (0.2, 1000), (0.4, 1000), (0.6, 1000),
                      (0.6, 0), (0.4, 0), (0.2, 0), (0.001, 0)]
        expected = [(mn, alt, pc, i < 4) for mn, alt in flight_env
                    for i, pc in enumerate([1, 0.9, 0.8, .7, 1, 0.85])]
        cols = ["run_index", "MN_set", "alt_ft_set", "PC", "viewer_printed", "point", "converged", "iterations",
                "final_residual_norm"] + list(runs[0]["DESIGN"]["summary"].keys()) if runs else []
        rows = []
        design_ref = snap["DESIGN"]["summary"]
        design_dev = 0.0
        for ent, exp in zip(runs, expected):
            for pt in ("OD_full_pwr", "OD_part_pwr"):
                s, sv = ent[pt]["summary"], ent[pt]["solver"]
                rows.append([ent["run_index"], exp[0], exp[1], exp[2] if pt == "OD_part_pwr" else None, exp[3], pt,
                             sv["converged"], sv["iterations"], sv["final_residual_norm"]] + list(s.values()))
            for k, v in ent["DESIGN"]["summary"].items():
                if design_ref[k] != 0:
                    design_dev = max(design_dev, abs(v - design_ref[k]) / abs(design_ref[k]))
        nonconv = [{"run_index": e["run_index"], "point": pt, **e[pt]["solver"]}
                   for e in runs for pt in POINTS if not e[pt]["solver"]["converged"]]
        # the benchmark configuration also occurs in the sweep: (0.8, 35000), PC 0.8 -> run index 2
        cross = None
        if len(runs) > 2:
            ref_pp = snap["OD_part_pwr"]["summary"]
            sw_pp = runs[2]["OD_part_pwr"]["summary"]
            cross = {"sweep_run_index": 2, "note": "sweep reaches (MN 0.8, 35 kft, PC 0.8) after PC 1 and 0.9 "
                     "warm starts; the reference case starts from the upstream initial guesses",
                     "max_rel_diff_OD_part_pwr_summary": max(abs(sw_pp[k] - ref_pp[k]) / abs(ref_pp[k])
                                                               for k in ref_pp if ref_pp[k] != 0)}
        sweep_info.update({
            "source": "envs/pycycle/upstream/example_cycles/high_bypass_turbofan.py executed with runpy as __main__",
            "seconds": dt, "error": err, "n_run_model": len(runs), "n_expected": len(expected),
            "order_matches_example_loop": len(runs) == len(expected) and all(
                abs(e["OD_part_pwr_PC"] - x[2]) < 1e-12 and
                abs(e["OD_part_pwr"]["summary"]["alt_ft"] - x[1]) < 1e-6 for e, x in zip(runs, expected)),
            "non_converged": nonconv,
            "design_point_max_rel_drift_vs_reference": design_dev,
            "reference_cross_check": cross,
            "stdout_lines": len(log.splitlines()),
            "stdout_failure_lines": [ln.strip() for ln in log.splitlines()
                                     if re.search(r"fail|stall|nan|inf\b", ln, re.I)][:50],
            "hbtf_view_out_sha256": sha256(view) if view.exists() else None,
            "hbtf_view_out_note": "the example writes hbtf_view.out (upstream viewer() text for DESIGN once and "
                                  "OD_part_pwr at every printed point) in its working directory; it was written "
                                  "to a temporary directory and only its hash is kept",
            "columns": cols, "rows": rows,
            "row_units": {"W_lbm_s": "lbm/s", "Fn_lbf": "lbf", "Fg_lbf": "lbf", "Fram_lbf": "lbf",
                          "TSFC_lbm_h_lbf": "lbm/h/lbf", "Tt4_degR": "degR", "Wfuel_lbm_s": "lbm/s",
                          "LP_Nmech_rpm": "rpm", "HP_Nmech_rpm": "rpm", "alt_ft": "ft"},
        })
    Problem.run_model = orig_run_model

    # ---------------- C. upstream test file in subprocesses
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(UPSTREAM))
    sub = {}
    pdir = workdir / "pytest"
    pdir.mkdir()
    junit = pdir / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", str(SHIM), "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=pdir, env=env, capture_output=True, text=True)
    last = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
    jx = ET.parse(junit).getroot() if junit.exists() else None
    ts = jx.find("testsuite") if jx is not None and jx.tag == "testsuites" else jx
    sub["pytest"] = {"command": "python -m pytest scripts/phase8/pycycle/upstream_benchmark_pytest_shim.py -q "
                                "-p no:cacheprovider (cwd = temporary directory)",
                     "returncode": p.returncode, "summary": last[0], "seconds": time.perf_counter() - t0,
                     "junit": {k: ts.get(k) for k in ("tests", "failures", "errors", "skipped")} if ts is not None
                     else None}
    tdir = workdir / "testflo"
    tdir.mkdir()
    testflo = Path(sys.executable).parent / "testflo"
    cmd = [str(testflo), "-b", "-n", "1", str(BENCH)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=tdir, env=env, capture_output=True, text=True)
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-6:]
    sub["testflo"] = {"command": "testflo -b -n 1 envs/pycycle/upstream/example_cycles/tests/benchmark_hbtf.py "
                                 "(PYTHONPATH = envs/pycycle/upstream, cwd = temporary directory)",
                      "returncode": p.returncode, "tail": tail, "seconds": time.perf_counter() - t0}

    # ---------------- write outputs
    csv_rows = []
    for pt in POINTS:
        for r in snap[pt]["records"]:
            csv_rows.append({"case": "reference", "point": pt, **{k: r[k] for k in
                            ("group", "element", "quantity", "path", "value", "units", "value_si", "units_si")}})
    point_block = {}
    for pt in POINTS:
        nested = {}
        for r in snap[pt]["records"]:
            nested.setdefault(r["group"], {}).setdefault(r["element"], {})[r["quantity"]] = {
                "value": r["value"], "units": r["units"], "value_si": r["value_si"], "units_si": r["units_si"]}
        point_block[pt] = {"mode": point_mode(ref_prob, pt), "summary": snap[pt]["summary"],
                           "solver": snap[pt]["solver"], "data": nested, "missing_paths": snap[pt]["missing"]}

    result = {
        "record": "Phase 8 D2 pyCycle HBTF reference (code-to-code target for P8.4); not calibration or validation data",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/phase8/pycycle/run_hbtf_reference.py",
        "script_sha256": sha256(__file__),
        "pytest_shim_sha256": sha256(SHIM),
        "environment": {
            "conda_env": sys.prefix, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "versions": versions, "pins": PINS,
            "pycycle___version__": pycycle.__version__, "openmdao___version__": openmdao.__version__,
            "numpy___version__": np.__version__,
            "lockfiles": ["envs/pycycle/conda-lock-osx-arm64.txt", "envs/pycycle/pip-freeze.txt",
                          "envs/pycycle/pip-requirements-lock.txt"],
        },
        "upstream": {"repository": sources["repository"], "tag": sources["tag"], "tag_commit": sources["tag_commit"],
                     "files": upstream_files, "sources_json": "envs/pycycle/upstream/SOURCES.json"},
        "units_note": "native pyCycle units (English) in 'value'/'units'; 'value_si'/'units_si' via "
                      "openmdao.utils.units.convert_units; TSFC SI in g/(kN*s); degR->degK",
        "reference_case": {
            "description": "upstream HBTFTestCase.setUp() + benchmark_case1(): one run_model of MPhbtf "
                           "(DESIGN, OD_full_pwr, OD_part_pwr at PC 0.8), np.seterr(divide='raise') as upstream",
            "run_model_seconds": snap["_run_model_seconds"],
            "upstream_method": upstream_method,
            "points": point_block,
        },
        "upstream_test_comparison": {
            "source": "envs/pycycle/upstream/example_cycles/tests/benchmark_hbtf.py (parsed by AST; numbers not retyped)",
            "semantics": "openmdao assert_near_equal: |actual-ref|/|ref| <= tol (absolute if ref == 0)",
            "n_values": len(comparison), "n_pass": n_pass, "all_pass": n_pass == len(comparison),
            "values": comparison,
        },
        "upstream_test_runs": {"in_process": upstream_method["result"], **sub},
        "example_main_sweep": sweep_info,
        "warnings": tally.table(),
        "total_seconds": time.time() - t_start,
        "workdir_temporary": str(workdir),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "point", "group", "element", "quantity", "path", "value", "units", "value_si", "units_si"]
    with open(out_csv, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})
    result["csv"] = {"path": str(out_csv.relative_to(ROOT)) if out_csv.is_relative_to(ROOT) else str(out_csv),
                     "rows": len(csv_rows), "sha256": sha256(out_csv)}
    with open(out_json, "x") as fh:
        json.dump(result, fh, indent=1, allow_nan=True)
        fh.write("\n")

    print(f"upstream asserts: {n_pass}/{len(comparison)} pass; in-process {upstream_method['result']}; "
          f"pytest rc={sub['pytest']['returncode']} ({sub['pytest']['summary']}); "
          f"testflo rc={sub['testflo']['returncode']}")
    if not args.no_sweep:
        print(f"sweep: {sweep_info['n_run_model']}/{sweep_info['n_expected']} run_model calls, "
              f"{len(sweep_info['non_converged'])} non-converged point solves, {sweep_info['seconds']:.0f} s")
    print(f"wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
