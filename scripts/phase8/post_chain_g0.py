"""Fresh original G0 procedure with actual C++ worker provenance.

This wrapper requires committed post-chain evidence, idle AC and exclusive
ownership. It preserves the original tasks, comparison rules and tables.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase8"))
import scientific_workflow_gate as gate

SPEC_ENV = "CATJET_G0_PROVENANCE_SPEC"
_POOL_COUNTER = 0


class TrackedPool(ProcessPoolExecutor):
    """Original executor with write-once spawn ownership, never task changes."""
    def __init__(self, *args, **kwargs):
        global _POOL_COUNTER
        self._provenance_pool = _POOL_COUNTER
        _POOL_COUNTER += 1
        self._provenance_spawn = 0
        super().__init__(*args, **kwargs)

    def _spawn_process(self):
        spec = gate.read(Path(os.environ[SPEC_ENV]))
        root = Path(spec["root"])
        folder = root / spec["attempt"] / "spawns"
        name = f"pool_{self._provenance_pool}_spawn_{self._provenance_spawn}"
        self._provenance_spawn += 1
        gate.write_once(folder / f"{name}.starting.json", {"state": "STARTING", "parent_pid": os.getpid(),
                        "spec_sha256": gate.digest(Path(os.environ[SPEC_ENV]))})
        before = set(self._processes)
        super()._spawn_process()
        added = set(self._processes)-before
        if len(added) != 1:
            raise gate.GateError("G0 pool spawn ownership is ambiguous")
        pid = added.pop()
        ac = gate._ac(root)
        birth = ac.process_birth(pid)
        if not birth or birth == ac.DEAD:
            raise gate.GateError("G0 pool birth unreadable; spawn remains ambiguous")
        gate.write_once(folder / f"{name}.complete.json", {"state": "CREATED", "parent_pid": os.getpid(),
                        "pid": pid, "birth": birth, "argv": ["G0", "pool_worker", name],
                        "spec_sha256": gate.digest(Path(os.environ[SPEC_ENV]))})


def preload_core(root, core):
    path = gate.relative(root, core["path"]).resolve()
    if gate.digest(path) != core["sha256"]:
        raise gate.GateError("Selected G0 core differs from registered proof")
    existing = sys.modules.get("catjet_core")
    if existing is None:
        spec = importlib.util.spec_from_file_location("catjet_core", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["catjet_core"] = module
    else:
        module = existing
    actual = Path(module.__file__).resolve()
    if actual != path or gate.digest(actual) != core["sha256"]:
        raise gate.GateError("G0 loaded a foreign C++ core; no fallback")
    return module


def _proof(root, spec, role):
    ac = gate._ac(root)
    preload_core(root, spec["core"])
    birth = ac.process_birth(os.getpid())
    if not birth or birth == ac.DEAD:
        raise gate.GateError("G0 process birth unreadable")
    proof = {"role": role, "pid": os.getpid(), "birth": birth,
             "core": spec["core"], "spec_sha256": gate.digest(Path(os.environ[SPEC_ENV])),
             "main_dependency_sha256": spec["identity"]["main_dependency_sha256"],
             "extension_sha256": spec["identity"]["extension_sha256"],
             "created_utc": gate.utc()}
    gate.write_once(root / spec["attempt"] / "workers" / f"{role}_{os.getpid()}.json", proof)


def _authorize_worker(spec):
    root = Path(spec["root"])
    ctx = gate.prepare_context(root, gate.OPERATIONS, require_g0=False)
    if ctx.identity != spec["identity"]:
        raise gate.GateError("G0 worker source identity differs")
    ready = root / spec["attempt"] / "worker_go" / f"{os.getpid()}.json"
    deadline = time.monotonic()+60
    while not ready.exists():
        if time.monotonic() > deadline:
            raise gate.GateError("G0 worker never received its recorded owner GO")
        time.sleep(.05)
    go = gate.read(ready)
    lease = gate.read(root / ctx.op["paths"]["owner_lease"])
    ac = gate._ac(root)
    birth = ac.process_birth(os.getpid())
    if go.get("pid") != os.getpid() or go.get("birth") != birth \
            or go.get("spec_sha256") != gate.digest(Path(os.environ[SPEC_ENV])) \
            or go.get("identity") != ctx.identity or lease.get("identity") != ctx.identity \
            or lease.get("owner_pid") != spec["owner_pid"] \
            or ac.liveness(lease.get("owner_pid"), lease.get("owner_birth")) != "alive" \
            or not any(c.get("pid") == os.getpid() and c.get("birth") == birth for c in lease.get("children", [])) \
            or not ac.on_ac(ac.read_power()):
        raise gate.GateError("G0 worker lacks actual recorded ownership or AC")


def g0_worker_init(excluded):
    """Spawned initializer delegates unchanged initializer after explicit preload."""
    path = Path(os.environ[SPEC_ENV])
    spec = gate.read(path)
    root = Path(spec["root"]).resolve()
    _authorize_worker(spec)
    for p in (root, root / "scripts/phase8", root / "scripts/optimization"):
        sys.path.insert(0, str(p))
    _proof(root, spec, "cpp_worker")
    import v6_backend
    # Spawn imports the original backend afresh; no parent monkeypatch survives.
    if v6_backend.init_worker_cpp is g0_worker_init:
        raise gate.GateError("G0 initializer requires spawn, never fork")
    v6_backend.init_worker_cpp(excluded)
    preload_core(root, spec["core"])


def g0_python_worker_init(excluded):
    spec = gate.read(Path(os.environ[SPEC_ENV]))
    root = Path(spec["root"])
    _authorize_worker(spec)
    ac = gate._ac(root)
    proof = {"role": "python_worker", "pid": os.getpid(), "birth": ac.process_birth(os.getpid()),
             "spec_sha256": gate.digest(Path(os.environ[SPEC_ENV])),
             "main_dependency_sha256": spec["identity"]["main_dependency_sha256"],
             "extension_sha256": spec["identity"]["extension_sha256"], "created_utc": gate.utc()}
    gate.write_once(root / spec["attempt"] / "workers" / f"python_worker_{os.getpid()}.json", proof)
    import lto_v5
    if lto_v5._init_worker is g0_python_worker_init:
        raise gate.GateError("G0 Python initializer requires spawn")
    lto_v5._init_worker(excluded)


def _execute(spec_path):
    root = ROOT.resolve()
    op = gate.operations(root)
    spec_path = Path(spec_path).resolve()
    attempt = root / op["paths"]["g0_attempt"]
    if spec_path != attempt / "command_spec.json":
        raise gate.GateError("Foreign private G0 command spec")
    spec = gate.read(spec_path)
    if spec.get("root") != str(root) or spec.get("attempt") != op["paths"]["g0_attempt"]:
        raise gate.GateError("G0 spec root or attempt mismatch")
    ctx = gate.prepare_context(root, gate.OPERATIONS, require_g0=False)
    if spec["identity"] != ctx.identity:
        raise gate.GateError("Private G0 identity changed")
    ready = attempt / "launch_ready.json"
    # Parent publishes the actual child ownership before allowing heavy imports.
    deadline = time.monotonic() + 30
    while not ready.exists():
        if time.monotonic() > deadline:
            raise gate.GateError("G0 parent did not publish ownership")
        time.sleep(0.05)
    lease = gate.read(root / op["paths"]["owner_lease"])
    ac = gate._ac(root)
    if lease["identity"] != ctx.identity or lease["owner_pid"] != spec["owner_pid"] \
            or ac.liveness(lease["owner_pid"], lease["owner_birth"]) != "alive" \
            or not any(c["pid"] == os.getpid() and ac.liveness(c["pid"], c["birth"]) == "alive" for c in lease["children"]):
        raise gate.GateError("G0 child is outside actual owner lease")
    power = ac.read_power()
    if not ac.on_ac(power):
        raise gate.GateError("G0 child has no AC power")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ[SPEC_ENV] = str(spec_path)
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    for p in (root, root / "scripts/optimization", root / "scripts/phase8"):
        sys.path.insert(0, str(p))
    preload_core(root, spec["core"])
    _proof(root, spec, "ae3_parent")
    import v6_backend
    import lto_v5
    original = v6_backend.init_worker_cpp
    original_py = lto_v5._init_worker
    original_pool = v6_backend.ProcessPoolExecutor
    original_py_pool = lto_v5.ProcessPoolExecutor
    v6_backend.init_worker_cpp = g0_worker_init
    lto_v5._init_worker = g0_python_worker_init
    v6_backend.ProcessPoolExecutor = lto_v5.ProcessPoolExecutor = TrackedPool
    try:
        import g0_parity
        return g0_parity.main(["--workers", "6", "--out-dir", spec["target"]])
    finally:
        v6_backend.init_worker_cpp = original
        lto_v5._init_worker = original_py
        v6_backend.ProcessPoolExecutor = original_pool
        lto_v5.ProcessPoolExecutor = original_py_pool
        preload_core(root, spec["core"])


def run(registration, out_dir):
    root = ROOT.resolve()
    op = gate.operations(root)
    if registration != gate.OPERATIONS or out_dir != op["paths"]["g0_target"]:
        raise gate.GateError("G0 command must match its committed registration")
    if (root / out_dir).exists() or (root / op["paths"]["g0_evidence"]).exists():
        raise gate.GateError("Fresh G0 target or receipt exists; no overwrite")
    ctx = gate.prepare_context(root, registration, require_g0=False)
    owned = ctx.acquire_run(op["paths"]["g0_attempt"], ctx.identity["registration_sha256"])
    attempt = owned.out
    start = time.perf_counter()
    child = None
    errors = []
    artifact_hashes = {}
    expected_outputs = []
    receipt = None
    ambiguous = False
    children = []
    status = "ERROR"
    try:
        spec_path = attempt / "command_spec.json"
        argv = [sys.executable, str(root / "scripts/phase8/post_chain_g0.py"), "_execute", "--spec", str(spec_path)]
        spec = {"root": str(root), "attempt": op["paths"]["g0_attempt"],
                "target": out_dir, "owner_pid": os.getpid(), "identity": ctx.identity,
                "core": ctx.identity["core"], "workers": 6, "child_argv": argv}
        gate.write_once(spec_path, spec)
        owned.assert_current()
        with (attempt / "command.log").open("x") as log:
            child = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            ac = gate._ac(root)
            birth = ac.process_birth(child.pid)
            if not birth or birth == ac.DEAD:
                raise gate.GateError("G0 child birth unreadable")
            children = [{"pid": child.pid, "birth": birth, "argv": argv}]
            owned.record_children(children)
            gate.write_once(attempt / "launch_ready.json", {"child": children[0], "created_utc": gate.utc()})
            while child.poll() is None:
                # Retain worker ownership as soon as its provenance is available.
                workers = [gate.read(p) for p in sorted((attempt / "spawns").glob("*.complete.json"))]
                now = children + [{k:p[k] for k in ("pid", "birth", "argv")} for p in workers]
                owned.record_children(now)
                owned.assert_current()
                for worker in workers:
                    go = attempt / "worker_go" / f"{worker['pid']}.json"
                    if not go.exists():
                        gate.write_once(go, {**worker, "identity": ctx.identity})
                time.sleep(2)
            exit_code = child.wait()
        exit_path = attempt / "command.exit.json"
        gate.write_once(exit_path, {"argv": argv, "pid": child.pid, "birth": birth,
                                   "exit_code": exit_code, "waited": True,
                                   "wall_s": time.perf_counter()-start, "ended_utc": gate.utc()})
        owned.assert_current()
        verdict = gate.read(root / out_dir / "g0_parity.json")
        proofs = []
        for p in sorted((attempt / "workers").glob("*.json")):
            proofs.append({**gate.read(p), "evidence_path": str(p.relative_to(root))})
        cpp_proofs = [p for p in proofs if p["role"] != "python_worker"]
        py_proofs = [p for p in proofs if p["role"] == "python_worker"]
        paths = [p for p in sorted((root / out_dir).rglob("*")) if p.is_file()]
        paths += [p for p in sorted(attempt.rglob("*")) if p.is_file()]
        for backend in verdict.get("backends", {}).values():
            paths += [root / c["frozen"] for c in backend.get("checks", {}).values()]
        artifact_hashes = {str(p.relative_to(root)): gate.digest(p) for p in paths}
        gate.recompare_g0(root, op, verdict, artifact_hashes)
        status = "PASS" if exit_code == 0 and verdict.get("verdict") == "PASS" else "FAIL"
        if status == "PASS":
            receipt = {"id": "P8-SCREENING-G0-EVIDENCE", "created_utc": gate.utc(),
                       "verdict": "PASS", "main_dependency_sha256": ctx.identity["main_dependency_sha256"],
                       "core": ctx.identity["core"], "worker_proofs": cpp_proofs, "python_worker_proofs": py_proofs,
                       "command_exit": str(exit_path.relative_to(root)), "files": artifact_hashes,
                       "identity": ctx.identity}
            # Validate complete raw provenance before publishing a receipt.
            if len(cpp_proofs) != 7 or len(py_proofs) != 6:
                raise gate.GateError("Incomplete G0 actual worker coverage")
        expected_outputs = list(artifact_hashes)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        status = "ERROR"
    finally:
        if child is not None and child.poll() is None:
            # No implicit child termination or ownership recovery.
            raise gate.GateError("G0 child remains live; owner lease retained")
        # Sync every spawn after the parent's waited exit, including crashes.
        # A STARTING slot without its complete receipt is never inferred ended.
        starts = sorted((attempt / "spawns").glob("*.starting.json"))
        completed = []
        for p in starts:
            done = p.with_name(p.name.replace(".starting.json", ".complete.json"))
            if not done.exists():
                ambiguous = True
            else:
                completed.append(gate.read(done))
        if children:
            owned.record_children(children + [{k:p[k] for k in ("pid", "birth", "argv")} for p in completed])
        terminal = owned.release({"status": status, "errors": errors,
                                  "exit_code": 0 if status == "PASS" else 1,
                                  "artifact_hashes": artifact_hashes,
                                  "expected_outputs": expected_outputs,
                                  "outputs_complete": status == "PASS", "ambiguous_child": ambiguous})
    if terminal["status"] == "PASS" and receipt is not None:
        ctx.require_idle_ac()
        for p in (attempt / "terminal.json", attempt / "released_lease.json"):
            receipt["files"][str(p.relative_to(root))] = gate.digest(p)
        receipt["terminal"] = str((attempt / "terminal.json").relative_to(root))
        gate.write_once(root / op["paths"]["g0_evidence"], receipt)
    print(json.dumps({"status": terminal["status"], "out_dir": out_dir, "errors": errors}))
    return 0 if terminal["status"] == "PASS" else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--registration", required=True)
    r.add_argument("--out-dir", required=True)
    private = sub.add_parser("_execute", help=argparse.SUPPRESS)
    private.add_argument("--spec", required=True)
    args = p.parse_args(argv)
    return _execute(args.spec) if args.command == "_execute" else run(args.registration, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
