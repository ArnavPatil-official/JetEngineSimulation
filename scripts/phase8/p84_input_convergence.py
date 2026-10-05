#!/usr/bin/env python3
"""A4c input-only design initialization and the 20-family convergence gate
(docs/phase8_p84c_registration.md sections 1 and 6; JSON
``design_initialization`` and ``input_convergence``).

Ordered starts: W = rated_N / d for d in (280, 220, 340), inside that FAR in
(0.03, 0.02, 0.04), with the existing turbine pressure-ratio guesses of the
right dimension. The FIRST converged, closure-valid candidate is chosen;
every try is retained. Fuel flow plays no part in the choice. The C++ solver
(scaled residual < 1e-10, 50 iterations and every other bound) is unchanged;
C1 widens only the opt-in numerical design-flow upper box on the new paths.

    .venv/bin/python scripts/phase8/p84_input_convergence.py [--xlsx PATH]

writes outputs/phase8/p84_input_convergence.json once, after the runtime gate
(AC power, no other heavy job, registration and sources committed). The
verdict is PASS only if all 20 families are covered and every case, including
both Trent 1000-E records, converges; exit status 1 otherwise.

This module also holds the A4c runtime gate shared by the other A4c scripts.
The C++ module is imported lazily, so the pure functions need no build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
from pinn_diagnostics.run_diagnostics import on_mains, python_target  # noqa: E402

REGISTRATION = ROOT / "docs" / "phase8_p84c_registration.json"
OUT = ROOT / "outputs" / "phase8" / "p84_input_convergence.json"
POSITIVE_SCALARS = ("W", "FAR", "Wfuel", "Fn_N", "Tt3", "Tt4", "A_core_m2", "A_byp_m2")
# Python jobs that must not overlap heavy A4c work (script paths; module form is the dotted path)
HEAVY_SCRIPTS = ("scripts/phase8/benchmark.py", "scripts/phase8/ablation_ladder.py",
                 "scripts/optimization/lto_v6.py", "scripts/optimization/calibrate_lto.py",
                 "scripts/phase8/trent_p84b.py", "scripts/phase8/trent_p84c.py",
                 "scripts/phase8/a4c_profile.py", "scripts/phase8/a4_error_budget.py",
                 "scripts/phase8/p84_input_convergence.py",
                 "scripts/phase8/pinn_diagnostics/run_diagnostics.py")


def load_registration(path: Path = REGISTRATION) -> dict:
    return json.loads(Path(path).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Ordered design starts and the closure-valid selection
# ---------------------------------------------------------------------------

def start_grid(reg: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(divisors, FARs) parsed from the registered start order text."""
    text = reg["design_initialization"]["starts_order"]
    m = re.fullmatch(r"for d in \(([^)]*)\): for FAR in \(([^)]*)\): W = rated_thrust_N / d", text)
    if not m:
        raise ValueError(f"unrecognised registered start order: {text!r}")
    return tuple(float(v) for v in m.group(1).split(",")), tuple(float(v) for v in m.group(2).split(","))


def design_starts(rated_kN: float, shafts: int, reg: dict) -> list[list[float]]:
    """The nine ordered design guesses [W, FAR, *turbine PRs]."""
    guess = reg["design_initialization"]["turbine_PR_guess"]
    prs = {3: guess["three_shaft"], 2: guess["two_shaft"]}[shafts]
    divisors, fars = start_grid(reg)
    return [[rated_kN * 1000.0 / d, far, *prs] for d in divisors for far in fars]


def _finite_pos(v) -> bool:
    try:
        return math.isfinite(float(v)) and float(v) > 0.0
    except (TypeError, ValueError):
        return False


def closure_problems(r: dict) -> list[str]:
    """Registered closure-validity criteria of one design result; empty = valid."""
    if not r.get("converged"):
        return [f"not converged: {r.get('reason', '')}"]
    bad = []
    x, residuals = r.get("x", []), r.get("residuals", [])
    if len(x) not in (4, 5) or len(residuals) != len(x):
        bad.append("design unknown/residual dimensions invalid")
    if not residuals or not all(math.isfinite(float(v)) for v in residuals) \
            or max(abs(float(v)) for v in residuals) >= 1e-10:
        bad.append("dispatched design scaled residual not finite and < 1e-10")
    iterations = r.get("iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 0 <= iterations <= 50:
        bad.append("design iteration evidence outside the registered 50-iteration limit")
    if not all(_finite_pos(v) for v in r.get("x", [])) or not r.get("x"):
        bad.append("unknowns not finite and positive")
    sc = r.get("scalars", {})
    bad += [f"scalar {k} not finite and positive" for k in POSITIVE_SCALARS if not _finite_pos(sc.get(k))]
    for name, st in (r.get("stations") or {}).items():
        if not all(_finite_pos(st.get(k)) for k in ("W", "Tt", "Pt")):
            bad.append(f"station {name} state not finite and positive")
    if not r.get("stations"):
        bad.append("no station states reported")
    for key, tol in (("mass_closure", 1e-8), ("energy_closure", 1e-8), ("element_closure", 1e-10)):
        v = r.get(key)
        if v is None or not math.isfinite(v) or not 0 <= v < tol:
            bad.append(f"{key} {v} not < {tol:g}")
    return bad


def try_record(start: list[float], r: dict) -> dict:
    probs = closure_problems(r)
    return {"start": [float(v) for v in start], "converged": bool(r.get("converged")),
            "iterations": int(r.get("iterations", 0)), "reason": r.get("reason", ""),
            "closure_valid": not probs, "problems": probs, "x": [float(v) for v in r.get("x", [])],
            "final_scaled_residual_inf": (max(abs(v) for v in r["residuals"]) if r.get("residuals") else None),
            **{k: r.get(k) for k in ("mass_closure", "energy_closure", "element_closure")}}


def solve_design_ordered(eng, rated_kN: float, shafts: int, reg: dict, single: bool = False):
    """Try the ordered starts on ``eng`` (its spec already set); return
    (selected design result or None, list of try records). ``single`` runs only
    the first start (the A4 guess), for reproducing the old A4 path."""
    starts = design_starts(rated_kN, shafts, reg)
    tries = []
    for start in starts[:1] if single else starts:
        r = eng.solve_design(list(start))
        tries.append(try_record(start, r))
        if tries[-1]["closure_valid"]:
            return r, tries
    return None, tries


# ---------------------------------------------------------------------------
# Runtime gate (fail closed; pure parts take command output so they are testable)
# ---------------------------------------------------------------------------

def is_heavy_target(target: str) -> bool:
    if not target:
        return False
    for s in HEAVY_SCRIPTS:
        if s in target or s[:-3].replace("/", ".") in target:
            return True
    name = os.path.basename(target.split()[0])
    return name in {os.path.basename(s) for s in HEAVY_SCRIPTS} or "calibrat" in name


def heavy_processes(ps_output: str, own_pid: int) -> list[str]:
    """Python interpreters running a heavy job (``ps -Ao pid=,args=``); parked shells ignored."""
    hits = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == own_pid:
            continue
        target = python_target(parts[1])
        if target is not None and is_heavy_target(target):
            hits.append(line.strip())
    return hits


def runtime_blockers(pmset_output: str | None, ps_output: str | None, own_pid: int,
                     head: str | None, porcelain: str | None) -> list[str]:
    out = []
    if not on_mains(pmset_output):
        out.append("not on mains (AC) power, or power source unknown")
    if ps_output is None or not ps_output.strip():
        out.append("process list unreadable; refusing (fail closed)")
    else:
        heavy = heavy_processes(ps_output, own_pid)
        if heavy:
            out.append("another heavy Python job is running: " + " | ".join(heavy))
    if head is None or not re.fullmatch(r"[0-9a-f]{40}", head):
        out.append("git HEAD unreadable; refusing (fail closed)")
    if porcelain is None:
        out.append("git status unreadable; refusing (fail closed)")
    elif porcelain.strip():
        out.append("registration/sources/records not committed and clean:\n" + porcelain.rstrip())
    return out


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def git_head() -> str | None:
    out = _cmd(["git", "rev-parse", "--verify", "HEAD"])
    return out.strip() if out else None


def tracked_clean_status(rels: list[str]) -> str | None:
    """``git status --porcelain`` for the paths, plus a line for any path that is
    not tracked at HEAD; None if git cannot be read."""
    st = _cmd(["git", "status", "--porcelain", "--", *rels])
    if st is None:
        return None
    lines = [st.rstrip()] if st.strip() else []
    for rel in rels:
        if subprocess.run(["git", "cat-file", "-e", f"HEAD:{rel}"], cwd=ROOT,
                          capture_output=True).returncode != 0:
            lines.append(f"?? {rel} (not committed at HEAD)")
    return "\n".join(lines)


def registered_files(reg: dict) -> list[str]:
    return ["docs/phase8_p84c_registration.json", "docs/phase8_p84c_registration.md",
            "docs/phase8_p83_a2_registration.json", "docs/phase8_p83_amendment_a2.md",
            *reg["implementation_files"], *reg["verification_files"],
            "docs/phase8_parameter_ledger.md", "scripts/phase8/ac_workflow.py",
            "docs/phase8_queue_recovery_registration.json",
            "scripts/phase8/pinn_diagnostics/run_diagnostics.py"]


BUILD_RECORD = "outputs/phase8/p84c_build_next.json"
WORKFLOW_REG = "docs/phase8_queue_recovery_registration.json"


def main_context(*, allow_full_pytest_child: bool = False) -> dict:
    """Only the strict record validator can authorize idle or serialized work."""
    import ac_workflow
    context = ac_workflow.validate_terminal_context(
        ROOT, registration=WORKFLOW_REG, require_idle=not allow_full_pytest_child,
        allow_active_stage="full_pytest" if allow_full_pytest_child else None)
    for stage in ("a2_calibration", "validation_build"):
        if context.get("stages", {}).get(stage, {}).get("state") != "PASS":
            raise RuntimeError(f"validated main {stage} PASS evidence required")
    return context


def cpp_sources() -> list[str]:
    out = _cmd(["git", "ls-files", "cpp"])
    if out is None:
        raise RuntimeError("tracked C++ sources unreadable")
    return [r for r in out.splitlines() if Path(r).suffix in (".cpp", ".hpp", ".h", ".sh", ".txt")]


def build_provenance(context: dict, core_path: Path) -> dict:
    """Derive provenance from validated build evidence; never launch a compiler."""
    rec = context["stages"]["validation_build"]
    if rec.get("state") != "PASS" or rec.get("exit_code") != 0:
        raise RuntimeError("validation build did not pass")
    launch = rec.get("launch", {})
    if not (launch.get("went") is True and launch.get("waited") is True
            and launch.get("exit_code") == 0 and launch.get("log_sha256")):
        raise RuntimeError("build launch/exit/log evidence missing")
    source = rec["identity"]["git_head"]
    if rec["identity"]["tracked_tree_sha256"] != context["identity"]["tracked_tree_sha256"]:
        raise RuntimeError("build scientific sources differ from current workflow identity")
    core_path = Path(core_path).resolve()
    if core_path.parent != (ROOT / "cpp/build_next").resolve():
        raise RuntimeError("A4c must load the separately built cpp/build_next module")
    binary_rel = str(core_path.relative_to(ROOT))
    digest = sha256(core_path)
    if rec.get("outputs", {}).get(binary_rel) != digest:
        raise RuntimeError("loaded core binary differs from validated build output")
    sources = {r: sha256(ROOT / r) for r in cpp_sources()}
    for r, value in sources.items():
        blob = subprocess.run(["git", "show", f"{source}:{r}"], cwd=ROOT, capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != value:
            raise RuntimeError(f"current C++ source differs from validated build commit: {r}")
    return {"source_commit": source, "actual_cpp_and_build_script_sha256": sources,
            "registration_sha256": sha256(REGISTRATION), "command": rec["command"],
            "build_directory": "cpp/build_next", "exit_evidence": launch,
            "loaded_core_path": binary_rel, "loaded_core_sha256": digest,
            "workflow_registration_sha256": context["registration_sha256"],
            "validation_build_record_sha256": hashlib.sha256(
                json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def ensure_build_provenance(context: dict, core_path: Path, *, export: bool = False) -> dict:
    want = build_provenance(context, core_path)
    path = ROOT / BUILD_RECORD
    if not path.exists():
        if not export:
            raise RuntimeError(f"{BUILD_RECORD} absent; export verified build evidence, then commit it")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(want, f, indent=1)
            f.write("\n")
    if json.loads(path.read_text()) != want:
        raise RuntimeError("write-once A4c build provenance disagrees with current verified build")
    return want


def ensure_ac() -> None:
    if not on_mains(_cmd(["pmset", "-g", "batt"])):
        raise RuntimeError("AC power required immediately before heavy work")


def live_gate(reg: dict, extra: list[str] = (), *, allow_full_pytest_child: bool = False) -> list[str]:
    rels = registered_files(reg) + cpp_sources() + list(extra) + [
        k for k in {**reg["inputs_sha256"], **reg["sources_sha256"], **reg["config_sha256"]}
        if "(untracked)" not in k]
    out = runtime_blockers(_cmd(["pmset", "-g", "batt"]), _cmd(["ps", "-Ao", "pid=,args="]),
                           os.getpid(), git_head(), tracked_clean_status(rels))
    try:
        context = main_context(allow_full_pytest_child=allow_full_pytest_child)
        import trent_p84c as A
        ensure_build_provenance(context, A.core_module_path(), export=allow_full_pytest_child)
        if not allow_full_pytest_child:
            dirty = tracked_clean_status([BUILD_RECORD])
            if dirty is None or dirty:
                raise RuntimeError("build provenance must be committed and clean before standalone work")
    except Exception as exc:
        out.append(f"main-record/build gate: {type(exc).__name__}: {exc}")
    return out


def identity(reg: dict, extra: list[str] = (), *, core_path: Path | None = None) -> dict:
    """Freeze scientific bytes separately from artifact dependency hashes."""
    if core_path is None:
        import trent_p84c as A
        core_path = A.core_module_path()
    files = registered_files(reg) + cpp_sources() + [k for k in {
        **reg["inputs_sha256"], **reg["sources_sha256"], **reg["config_sha256"]}]
    return {"git_head": git_head(), "sha256": {f: sha256(ROOT / f.removesuffix(" (untracked)")) for f in dict.fromkeys(files)},
            "core_module_sha256": sha256(core_path), "build_provenance_sha256": sha256(ROOT / BUILD_RECORD),
            "dependencies_sha256": {f: sha256(ROOT / f) for f in dict.fromkeys(extra)}}


def scientific_identity(ident: dict) -> dict:
    return {k: ident[k] for k in ("sha256", "core_module_sha256", "build_provenance_sha256")}


def assert_same_identity(start: dict, end: dict) -> None:
    if scientific_identity(start) != scientific_identity(end) \
            or start.get("dependencies_sha256") != end.get("dependencies_sha256"):
        raise RuntimeError("scientific source/input/binary/dependency drift; run is ERROR")


def finish_identity(reg: dict, start: dict, extra: list[str] = ()) -> tuple[dict | None, str | None]:
    end = None
    try:
        end = identity(reg, extra)
        assert_same_identity(start, end)
        ensure_ac()
        return end, None
    except Exception as exc:
        return end, f"{type(exc).__name__}: {exc}"


def check_registered_hashes(reg: dict) -> None:
    for rel, want in {**reg["inputs_sha256"], **reg["sources_sha256"], **reg["config_sha256"]}.items():
        path = rel.removesuffix(" (untracked)")
        if sha256(ROOT / path) != want:
            raise RuntimeError(f"registered file {rel} changed since registration")


# ---------------------------------------------------------------------------
# 20-family gate
# ---------------------------------------------------------------------------

def gate_verdict(records: list[dict], n_families: int) -> dict:
    """PASS iff every registered family is covered and every case converged with a
    closure-valid design (missing binary/databank or errors are never PASS)."""
    families = {r["family"] for r in records}
    failed = [r["case"] for r in records if r.get("status") != "converged"]
    coverage = len(families)
    return {"coverage": coverage, "n_registered_families": n_families, "n_cases": len(records),
            "failed_cases": failed,
            "verdict": "PASS" if coverage == n_families and records and not failed else "FAIL"}


def validate_convergence_record(doc: dict, reg: dict) -> None:
    cases = doc.get("cases", [])
    policy = reg["input_convergence"]["architecture_policy"]
    families = {f for v in policy.values() if isinstance(v, dict) for f in v.get("families", [])}
    if {c.get("family") for c in cases} != families:
        raise RuntimeError("convergence evidence does not cover exactly the registered 20 families")
    ids = [c.get("uid") for c in cases]
    if len(ids) != len(set(ids)) or not {"11RR052", "12RR058"} <= set(ids):
        raise RuntimeError("convergence evidence misses/duplicates a required Trent 1000-E case")
    if any(c.get("status") not in ("converged", "unreachable", "error") for c in cases):
        raise RuntimeError("unknown convergence case status")
    for c in cases:
        tries = c.get("tries", [])
        if c["status"] == "converged":
            idx = c.get("selected_start_index")
            if not isinstance(idx, int) or idx != len(tries) - 1 or idx < 0 \
                    or not tries[idx].get("closure_valid") or any(t.get("closure_valid") for t in tries[:idx]):
                raise RuntimeError("convergence selected-start evidence invalid")
            if len(c.get("design", {}).get("x", [])) != c.get("shafts", 0) + 2:
                raise RuntimeError("design dimension differs from registered two/three-shaft path")
            if closure_problems(c.get("design", {})):
                raise RuntimeError("convergence design closure evidence invalid")
    import public_engine_inputs as P
    expected_cases = {c["uid"]: c for c in P.load_cases(reg=reg)}
    if set(ids) != set(expected_cases):
        raise RuntimeError("convergence cases differ from deterministic representatives plus both Trent 1000-E")
    for c in cases:
        want = expected_cases[c["uid"]]
        for k, value in want.items():
            if c.get(k) != value:
                raise RuntimeError(f"convergence public input/policy differs for {c['uid']}: {k}")
        tries = c["tries"]
        starts = design_starts(c["rated_kN"], c["shafts"], reg)
        if len(tries) > len(starts) or any(t["start"] != start for t, start in zip(tries, starts)):
            raise RuntimeError("convergence ordered-start evidence differs from registered input-only ladder")
        if c["status"] == "converged":
            chosen, design = tries[-1], c["design"]
            for k in ("x", "iterations", "mass_closure", "energy_closure", "element_closure"):
                if chosen[k] != design[k]:
                    raise RuntimeError(f"selected-start/design evidence differs: {k}")
            if chosen["final_scaled_residual_inf"] != max(abs(v) for v in design["residuals"]):
                raise RuntimeError("selected residual evidence differs from dispatched design result")
        elif c["status"] == "unreachable" and len(tries) != len(starts):
            raise RuntimeError("unreachable case lacks all registered start attempts")
    expected = gate_verdict(cases, len(families))
    if doc.get("verdict") not in ("PASS", "FAIL") or any(doc.get(k) != v for k, v in expected.items()):
        raise RuntimeError("convergence aggregate does not match retained cases")


def run_case(case: dict, reg: dict) -> dict:
    """Design solve of one case at the A4c central prior; an exception is that
    case's recorded ERROR (status 'error'), never a skip."""
    import trent_p84c as A

    inp = A.engine_inputs(A.central())
    try:
        ensure_ac()
        eng = A.engine_for(case["opr"], case["bpr"], case["rated_kN"], case["shafts"], inp)
        d, tries = solve_design_ordered(eng, case["rated_kN"], case["shafts"], reg)
    except Exception as exc:
        return {**case, "status": "error", "reason": f"{type(exc).__name__}: {exc}", "tries": []}
    rec = {**case, "status": "converged" if d is not None else "unreachable", "tries": tries,
           "selected_start_index": (len(tries) - 1) if d is not None else None}
    if d is not None:
        rec["design"] = {k: d[k] for k in ("converged", "iterations", "x", "scalars", "stations", "residuals",
                                           "mass_closure", "energy_closure", "element_closure", "extrapolated_maps")}
    return rec


def main(argv=None, *, allow_full_pytest_child: bool = False) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xlsx", type=Path, default=None, help="ICAO databank (untracked raw file)")
    a = ap.parse_args(argv)
    reg = load_registration()
    if OUT.exists():
        print(f"{OUT.relative_to(ROOT)} exists; write-once")
        return 2
    blockers = live_gate(reg, allow_full_pytest_child=allow_full_pytest_child)
    if blockers:
        print("BLOCKED:\n- " + "\n- ".join(blockers))
        return 3
    check_registered_hashes(reg)
    import public_engine_inputs as P
    import trent_p84c as A

    try:                                 # prerequisites: refuse without writing anything
        cases = P.load_cases(a.xlsx or P.XLSX, reg)
        core_path = A.core_module_path()
    except Exception as exc:
        print(f"BLOCKED: prerequisite unavailable ({type(exc).__name__}: {exc}); nothing written")
        return 3
    started, ident = utc(), identity(reg)
    records, error = [], None
    try:
        for case in cases:
            records.append(run_case(case, reg))
            print(f"{records[-1]['case']:<28} {records[-1]['status']} after {len(records[-1]['tries'])} tries",
                  flush=True)
    except Exception as exc:            # recorded as an ERROR; never a PASS
        error = f"{type(exc).__name__}: {exc}"
    verdict = gate_verdict(records, P.reg_n_families(reg))
    end_ident, drift = finish_identity(reg, ident)
    error = error or drift
    if error:
        verdict["verdict"] = "ERROR"
    doc = {"stage": "A4c input-only design initialization, 20-family convergence gate",
           "registration": "docs/phase8_p84c_registration.json", **verdict, "error": error,
           "parameters": "A4c central prior", "scope": reg["input_convergence"]["scope"],
           "proxy_limits": reg["input_convergence"]["architecture_policy"]["two_shaft_proxy_limits"],
           "cases": records, "started_utc": started, "finished_utc": utc(), "start_identity": ident,
           "end_identity": end_ident, "xlsx_sha256": reg["inputs_sha256"][P.XLSX_KEY],
           "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                           "catjet_core_sha256": sha256(core_path)}}
    if doc["verdict"] != "ERROR":
        try:
            validate_convergence_record(doc, reg)
        except Exception as exc:
            doc["verdict"], doc["error"] = "ERROR", f"{type(exc).__name__}: {exc}"
    with OUT.open("x") as f:
        f.write(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"input convergence: {doc['verdict']} (coverage {verdict['coverage']}/{verdict['n_registered_families']}, "
          f"failed {verdict['failed_cases']}{', error ' + error if error else ''})")
    return 0 if doc["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
