"""One owned, write-once SAF study pipeline; phase shortcuts are refused."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.phase8.saf_surrogate"

from .registration import (REGISTRATION, append_progress, artifact_hashes, contained,
                           json_bytes, load_registration, read_json, sha256_file,
                           verify_artifacts, verify_named_prerequisite, write_once)

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def utc():
    return datetime.now(timezone.utc).isoformat()


def birth(pid):
    if os.name == "nt" or os.environ.get("CATJET_SIMULATOR_BACKEND") == "python":
        from simulation.runtime import process_birth
        value = process_birth(pid)
        if not value or value == "<dead>":
            raise RuntimeError("Recorded process is absent or unreadable")
        return value
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], check=True, capture_output=True, text=True)
    value = " ".join(result.stdout.split())
    if not value:
        raise RuntimeError("Recorded process is absent")
    return value


def native_command(pid):
    if os.name == "nt":
        # Spawn handshakes bind exact Python argv; Windows has no procps ps.
        return f"Windows process {pid}; command bound by child sys.orig_argv"
    return subprocess.check_output(["ps","-ww","-p",str(pid),"-o","command="],text=True).strip()


def require_ac():
    result = subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True)
    if "AC Power" not in result.stdout:
        raise RuntimeError("Scientific work requires AC power")


def check_child_authorization(spec):
    """Validate an exact registered child authorization, never ancestry/name."""
    root = Path(spec["root"])
    if birth(spec["owner_pid"]) != spec["owner_birth"]:
        raise RuntimeError("Authorized owner process identity is gone")
    lease = spec["owner_lease"]
    owner = read_json(root / lease["path"])
    if owner["state"]!="RUNNING":
        raise RuntimeError("Owned lease is not authorized for heavy child work")
    if owner["owner_pid"] != spec["owner_pid"] or owner["owner_birth"] != spec["owner_birth"]:
        raise RuntimeError("Raw lease owner differs from child authorization")
    snapshot = read_json(root / lease["snapshot_path"])
    if sha256_file(root / lease["snapshot_path"]) != lease["sha256"]:
        raise RuntimeError("Frozen lease snapshot changed")
    immutable = set(snapshot) - {"children", "state", "terminal_sha256"}
    if any(owner.get(key) != snapshot[key] for key in immutable):
        raise RuntimeError("Live lease differs from the frozen owner reservation")
    if os.getpid() != spec["owner_pid"] and not any(c["pid"] == os.getpid() and c["birth"] == birth(os.getpid()) for c in owner["children"]):
        raise RuntimeError("Heavy child is absent from exact live owner lease")
    if sha256_file(root / spec["registration_path"]) != spec["registration_sha256"]:
        raise RuntimeError("Child registration drift")
    verify_artifacts(root, spec["source_hashes"])
    verify_artifacts(root,spec["input_source_hashes"])
    if spec.get("simulator", {}).get("name") == "python-v6":
        from scripts.phase8.pc_python_runtime import fingerprint
        verify_artifacts(root, spec["simulator"]["source_hashes"])
        if fingerprint(spec["simulator"]) != spec["simulator_identity_sha256"]:
            raise RuntimeError("Child Python simulator identity drift")
    elif sha256_file(root / spec["binary"]["path"]) != spec["binary"]["sha256"]:
        raise RuntimeError("Child core drift")
    if spec.get("simulator", {}).get("name") == "python-v6":
        from simulation.runtime import require_ac
        require_ac()
    elif spec.get("execution_profile") == "pc":
        from scripts.phase8.pc_runtime import require_idle_ac_linux
        require_idle_ac_linux()
    else:
        require_ac()


def csv_once(path, rows, fieldnames=None):
    rows = list(rows)
    fieldnames = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (json.dumps(value, sort_keys=True, allow_nan=False)
                        if isinstance(value, (dict, list, bool)) else value) for key, value in row.items()})
    return write_once(path, stream.getvalue().encode())


def _dependency(root, path):
    return {"path": str(Path(path)), "sha256": sha256_file(Path(root) / path)}


def mac_platform_info():
    """Default historical platform/hardware metadata collector (macOS only)."""
    return {"physical_cores": int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)),
            "platform": subprocess.check_output(["sw_vers"], text=True),
            "hardware": subprocess.check_output(["sysctl", "hw.model", "hw.memsize"], text=True)}


def freeze(root, output, reg, reg_sha, context, run, backend=None, platform_info=None,
           source_hashes=None, provenance_dependencies=None):
    from .inputs import load_fixed_draws, load_public_inputs, named_queries, query_designs
    from .thermo import freeze_properties
    import importlib.metadata
    from simulation.ml_backend import resolve_backend
    backend = resolve_backend(backend)
    run.assert_current()
    public, draws = load_public_inputs(root, reg), load_fixed_draws(root, reg)
    queries, named = query_designs(reg, draws, public), named_queries(draws, public)
    known = {query["input_sha256"] for rows in queries.values() for query in rows}
    if any(row["input_sha256"] in known for row in named) or len({q["input_sha256"] for q in named}) != 68:
        raise ValueError("Named/cross-split duplicate; no seed replacement permitted")
    write_once(output / "public_inputs.json", public)
    write_once(output / "fixed_draws.json", draws)
    for name, rows in queries.items():
        write_once(output / f"splits/{name}.json", rows)
    write_once(output / "splits/named_central.json", named)
    write_once(output / "frozen_properties.json", freeze_properties(root, reg))
    paths = reg["relevant_files"]["implementation_create"] + reg["relevant_files"]["registration_create"]
    sources = {path: sha256_file(root / path) for path in paths}
    if source_hashes is not None:
        verify_artifacts(root, source_hashes)
        sources.update(source_hashes)
    python_simulator = getattr(context, "simulator_backend", None) == "python"
    environment = {"registration_id": reg["id"], "registration_sha256": reg_sha,
        "identity": context.identity, "binary_path": None if python_simulator else str(context.binary_path.relative_to(root)),
        "binary_sha256": context.binary_sha256, "executable": sys.executable,
        "python": sys.version, "thread_environment": {key: os.environ[key] for key in THREAD_ENV},
        "training_backend": backend,
        "training_dtype":"float64" if backend == "torch" else "float32",
        "score_backend":"torch" if backend == "torch" else "numpy", "score_dtype":"float64", "score_device":"cpu",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "PyYAML", "Cantera", backend)},
        **(platform_info or mac_platform_info)()}
    if python_simulator:
        environment.update(simulator=context.simulator_identity,
            simulator_identity_sha256=context.simulator_identity_sha256, workers=context.workers)
    write_once(output / "environment.json", environment)
    manifest = {"registration_id": reg["id"], "registration_sha256": reg_sha,
        "consumer_identity": context.identity, "scientific_sources": sources,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "inputs": artifact_hashes(output, ["public_inputs.json", "fixed_draws.json", "frozen_properties.json"]
                                    + [f"splits/{name}.json" for name in (*queries, "named_central")])}
    write_once(output / "manifest.json", manifest)
    extension = "outputs/phase8/screening_operations/source_extension_manifest.json"
    dependency_paths = {
        "main_dependency": "outputs/phase8/screening_operations/main_dependency.json",
        "g0_evidence": "outputs/phase8/screening_operations/g0_evidence.json",
        "source_extension_manifest": extension}
    if provenance_dependencies is not None:
        dependency_paths = dict(provenance_dependencies)
    dependency_paths.update({
        "frozen_properties": str((output / "frozen_properties.json").relative_to(root)),
        "train_queries": str((output / "splits/train.json").relative_to(root)),
        "named_queries": str((output / "splits/named_central.json").relative_to(root))})
    dependencies = {key: _dependency(root, path) for key, path in dependency_paths.items()}
    pre = {"schema_version": 1, "registration_id": reg["id"], "registration_sha256": reg_sha,
        "source_commit": manifest["source_commit"], "consumer_identity": context.identity,
        "binary": {"path": environment["binary_path"], "sha256": context.binary_sha256},
        "dependencies": dependencies, "counts": {"train": 4096, "named_central": 68},
        "cases": {"train": [{key: q[key] for key in ("design_id", "prefix_index", "draw_id", "input_sha256")}
                            for q in queries["train"]],
                  "named_central": [{key: q[key] for key in ("named_case_id", "fuel", "op", "input_sha256", "in_product_API", "fuel_parts")}
                                    for q in named]}}
    if python_simulator:
        pre.update(simulator=context.simulator_identity, simulator_identity_sha256=context.simulator_identity_sha256)
    write_once(output / "property_inputs_manifest.json", pre)
    run.assert_current()


def command_spec(root, output, context, reg_sha, stage, argv):
    lease = "outputs/phase8/screening_operations/owner.lease.json"
    pc = getattr(context, "execution_profile", None) == "pc"
    python_simulator = getattr(context, "simulator_backend", None) == "python"
    if pc:
        lease = str(Path(context.pc_lease_path).relative_to(root))
    manifest = read_json(output / "manifest.json")
    properties=read_json(output/"frozen_properties.json")
    prefix="generation" if stage=="generate" else stage
    spec = {"schema_version": 1, "stage": stage, "root": str(root), "output": str(output),
        "registration_path": REGISTRATION, "registration_sha256": reg_sha,
        "consumer_identity": context.identity, "start_identity": context.identity,
        "binary": None if python_simulator else {"path": str(context.binary_path.relative_to(root)), "sha256": context.binary_sha256},
        "property_inputs_manifest_sha256": sha256_file(output / "property_inputs_manifest.json"),
        "owner_lease": dict(_dependency(root, lease), snapshot_path=str((output / f"proofs/{prefix}_owner_lease.json").relative_to(root))),
        "owner_pid": os.getpid(), "owner_birth": birth(os.getpid()),
        "source_hashes": manifest["scientific_sources"], "argv": argv, "started_utc": utc(),
        "input_source_hashes":{"data/creck_c1c16_full.yaml":properties["mechanism_sha256"],
            "data/fuel_properties_v7.yaml":properties["fuel_yaml_sha256"],
            "data/corsia_lca_values.yaml":properties["lca_yaml_sha256"]},
        "command_spec_path": str((output / f"proofs/{prefix}_command_spec.json").relative_to(root))}
    if pc:
        spec["execution_profile"] = "pc"
    if python_simulator:
        spec.update(simulator=context.simulator_identity, simulator_identity_sha256=context.simulator_identity_sha256,
                    workers=context.workers)
    return spec


def light_training_check(run):
    """Per-epoch live ownership/AC/frozen registration+extension fingerprints.

    Full original byte/record proof is revalidated at fit boundaries and every
    100 epochs. This lightweight check is never accepted as the final proof.
    """
    if getattr(run.context, "execution_profile", None) == "pc":
        run.light_training_check()
        return
    run._assert_owned();require_ac()
    lease=read_json(run.lease_path)
    if lease["owner_pid"]!=os.getpid() or lease["owner_birth"]!=run.owner_birth:
        raise RuntimeError("Training owner identity changed")
    root=run.context.root
    if sha256_file(root/REGISTRATION)!=run.identity["registration_sha256"]:
        raise RuntimeError("Training registration fingerprint changed")
    path=run.context.op["paths"]["source_extension_manifest"]
    if sha256_file(root/path)!=run.identity["extension_sha256"]:
        raise RuntimeError("Training source-extension fingerprint changed")


def wait_generation(root, output, context, run, reg_sha):
    spec_path = output / "proofs/generation_command_spec.json"
    argv = [sys.executable, "-m", "scripts.phase8.saf_surrogate.run", "_generate",
            "--spec", str(spec_path)]
    write_once(output / "proofs/generation_starting.json", {"state":"STARTING", "owner_pid":os.getpid(), "argv":argv})
    log = output / "proofs/generation.log"
    with log.open("xb", buffering=0) as stream:
        child = subprocess.Popen(argv, cwd=root, env=os.environ.copy(), stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        child_birth = birth(child.pid)
        run._saf_children = [{"pid":child.pid,"birth":child_birth,"argv":argv}]
        run.record_children(run._saf_children)
        run.assert_current()
        spec = command_spec(root, output, context, reg_sha, "generate", argv)
        write_once(root / spec["owner_lease"]["snapshot_path"], (root / spec["owner_lease"]["path"]).read_bytes())
        write_once(spec_path, spec)
        seen=set()
        try:
            while child.poll() is None:
                for receipt_path in sorted((output / "proofs/spawns").glob("generate_*.complete.json")):
                    if receipt_path.name in seen:
                        continue
                    receipt=read_json(receipt_path)
                    if receipt["spec_sha256"] != sha256_file(spec_path) or receipt["parent_pid"] != child.pid or birth(receipt["pid"]) != receipt["birth"]:
                        raise RuntimeError("Worker spawn receipt differs from exact waited child")
                    if receipt["native_command"] != native_command(receipt["pid"]):
                        raise RuntimeError("Worker receipt command differs from native process command")
                    run._saf_children.append({key:receipt[key] for key in ("pid","birth","argv")})
                    run.record_children(run._saf_children); run.assert_current()
                    write_once(output / f"proofs/go/worker_{receipt['pid']}.json", {"receipt_sha256":sha256_file(receipt_path), "spec_sha256":sha256_file(spec_path)})
                    seen.add(receipt_path.name)
                run.assert_current()
                time.sleep(.5)
            exit_code = child.wait()
            run.assert_current()
        except BaseException:
            if child.poll() is None:
                child.terminate() if os.name == "nt" else os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill() if os.name == "nt" else os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    starting=list((output/"proofs/spawns").glob("generate_*.starting.json"))
    completed=list((output/"proofs/spawns").glob("generate_*.complete.json"))
    if len(starting)!=len(completed) or len(completed)!=spec.get("workers", 6):
        raise RuntimeError("Unresolved or wrong-count generation pool spawn; lease retained")
    worker_proofs=[]
    for receipt_path in sorted(completed):
        receipt=read_json(receipt_path)
        expected_handshake=output/f"proofs/workers/generate_{receipt['pid']}.json"
        if receipt["handshake_path"]!=str(expected_handshake.relative_to(root)) or receipt["handshake_sha256"]!=sha256_file(expected_handshake):
            raise RuntimeError("Generation worker handshake path/hash mismatch")
        handshake=read_json(expected_handshake)
        if any(handshake[key]!=receipt[key] for key in ("pid","birth","argv","native_command","parent_pid","spec_sha256")) or receipt["parent_pid"]!=child.pid or receipt["spec_sha256"]!=sha256_file(spec_path):
            raise RuntimeError("Final generation worker receipt differs from waited command")
        core_path=output/f"proofs/workers/generate_{receipt['pid']}_core.json"
        exit_path_worker=output/f"proofs/workers/generate_{receipt['pid']}_exit.json"
        loaded,ended=read_json(core_path),read_json(exit_path_worker)
        actual_identity = loaded.get("actual_simulator") if spec.get("simulator") else loaded.get("actual_core")
        expected_identity = spec.get("simulator") or spec["binary"]
        if any(loaded[key]!=receipt[key] for key in ("pid","birth","parent_pid")) or loaded["input_source_hashes"]!=spec["input_source_hashes"] or actual_identity!=expected_identity or loaded["spec_sha256"]!=sha256_file(spec_path) or loaded["handshake_sha256"]!=sha256_file(expected_handshake):
            raise RuntimeError("Generation worker did not prove the selected loaded core")
        go=output/f"proofs/go/worker_{receipt['pid']}.json"
        if loaded["go_sha256"]!=sha256_file(go) or read_json(go)!={"receipt_sha256":sha256_file(receipt_path),"spec_sha256":sha256_file(spec_path)}:
            raise RuntimeError("Generation worker GO does not bind parent-recorded spawn/spec")
        if any(ended[key]!=receipt[key] for key in ("pid","birth","argv","parent_pid","spec_sha256")) or ended["waited"] is not True or ended["exit_code"]!=0:
            raise RuntimeError("Generation worker exit was not successfully waited")
        if receipt_path.name not in seen:
            # After shutdown native PID may be dead/reused. Only the already
            # recorded exact handshake+loaded core+actual waited-exit proof
            # can bind this late receipt; no live-process guess is accepted.
            run._saf_children.append({key:receipt[key] for key in ("pid","birth","argv")})
            run.record_children(run._saf_children);seen.add(receipt_path.name)
        worker_proofs.append({key:_dependency(root,str(path.relative_to(root))) for key,path in
            (("spawn",receipt_path),("handshake",expected_handshake),("core",core_path),("exit",exit_path_worker))})
    handshake_path = output / "proofs/generation_handshake.json"
    handshake = read_json(handshake_path)
    if handshake["pid"] != child.pid or handshake["birth"] != child_birth or handshake["argv"] != argv:
        raise RuntimeError("Waited generation child differs from its handshake")
    exit_record = dict(spec, pid=child.pid, birth=child_birth, waited=True, exit_code=exit_code,
                       end_identity=context.identity, finished_utc=utc(), identity_problems=[])
    exit_path = output / "proofs/generation_exit.json"
    write_once(exit_path, exit_record)
    if exit_code:
        raise RuntimeError(f"Generation child exited {exit_code}; no retry")
    summary = read_json(output / "proofs/generation_summary.json")
    raw = {key: _dependency(root, str(path.relative_to(root))) for key, path in
           (("spec", spec_path), ("handshake", handshake_path), ("exit", exit_path), ("log", log))}
    terminal = dict(summary, schema_version=1, registration_id="P8-S-20261004",
        registration_sha256=reg_sha, stage="generate", state="COMPLETE",
        start_identity=context.identity, end_identity=context.identity, identity_problems=[],
        binary_path=None if spec.get("simulator") else spec["binary"]["path"], binary_sha256=context.binary_sha256,
        property_inputs_manifest_sha256=spec["property_inputs_manifest_sha256"], raw_command=raw,workers=worker_proofs,
        launch={"pid": child.pid, "birth": child_birth, "argv": argv, "started_utc": spec["started_utc"],
                "finished_utc": exit_record["finished_utc"], "exit_code": 0, "waited": True,
                "log_path": str(log.relative_to(root)), "log_sha256": sha256_file(log)})
    if spec.get("simulator"):
        terminal.update(simulator=spec["simulator"], simulator_identity_sha256=spec["simulator_identity_sha256"])
    write_once(output / "generation_terminal.json", terminal)
    pre = read_json(output / "property_inputs_manifest.json")
    allowed = ("teacher_rows.csv", "teacher_species.npz", "named_central_properties.csv", "frozen_properties.json")
    write_once(output / "property_manifest.json", {"schema_version": 1, "registration_id": "P8-S-20261004",
        "registration_sha256": reg_sha, "source_commit": pre["source_commit"],
        "consumer_identity": context.identity, "binary": pre["binary"], "counts": pre["counts"],
        "property_inputs_manifest": {"path": "property_inputs_manifest.json", "sha256": sha256_file(output / "property_inputs_manifest.json")},
        "producer_terminal": {"path": "generation_terminal.json", "sha256": sha256_file(output / "generation_terminal.json")},
        "case_identity": pre["cases"], "artifacts": artifact_hashes(output, allowed)})


def _capture_dataset(output, name, queries, states, properties, proof):
    import numpy as np
    from .train import DATASETS
    rows, compositions, identities = [], [], []
    csv_name, species_name = DATASETS[name]
    for query, state in zip(queries, states):
        row = {key: query[key] for key in ("split", "design_id", "prefix_index", "draw_id", "input_sha256",
                                          "f_JetA", "f_HEFA", "f_FT", "f_ATJ", "thrust_fraction")}
        row.update(proof, status=state["status"], reason=state["reason"], species_row_index=None)
        if state["status"] == "converged":
            row.update({key: state[key] for key in state if key not in ("Y4", "input_state", "status", "reason")})
            row["species_row_index"] = len(compositions)
            compositions.append(state["Y4"]); identities.append(query)
        rows.append(row)
    csv_once(output / csv_name, rows)
    stream = io.BytesIO()
    np.savez(stream, Y4=np.asarray(compositions, dtype=np.float64).reshape(-1, 492),
        species_order=np.asarray(properties["species_order"]),
        **{key: np.asarray([q[key] for q in identities]) for key in ("design_id", "draw_id", "prefix_index", "input_sha256")})
    write_once(output / species_name, stream.getvalue())
    return [csv_name, species_name], sum(row["status"] == "converged" for row in rows)


def generate_child(spec_path):
    deadline=time.monotonic()+60
    while not spec_path.exists():
        if time.monotonic()>deadline:
            raise RuntimeError("Parent never published exact child authorization")
        time.sleep(.05)
    spec = read_json(spec_path)
    check_child_authorization(spec)
    from .teacher import ParallelTeacher
    import numpy as np
    actual_argv = list(sys.orig_argv)
    if actual_argv != spec["argv"]:
        raise RuntimeError("Child command differs from registered exact argv")
    root, output = Path(spec["root"]), Path(spec["output"])
    write_once(output / "proofs/generation_handshake.json", dict(spec, pid=os.getpid(), birth=birth(os.getpid()), argv=actual_argv))
    properties, public, draws = (read_json(output / name) for name in ("frozen_properties.json", "public_inputs.json", "fixed_draws.json"))
    prerequisite = read_json(output / "prerequisite.json")
    proof = {"source_registration_sha256": spec["registration_sha256"], "binary_sha256": None if spec.get("simulator") else spec["binary"]["sha256"],
        "source_commit": read_json(output / "manifest.json")["source_commit"],
        "property_manifest_sha256": spec["property_inputs_manifest_sha256"]}
    if spec.get("simulator"):
        proof["simulator_identity_sha256"] = spec["simulator_identity_sha256"]
    pool = ParallelTeacher(spec, properties, public, draws, spec.get("workers", 6))
    paths, converged = [], {}
    try:
        for name in ("train", "validation", "test", "ranking_test"):
            queries = read_json(output / f"splits/{name}.json")
            states = pool.full_states(queries)
            new_paths, count = _capture_dataset(output, name, queries, states, properties, proof)
            paths += new_paths; converged[name] = count
            print(json.dumps({"split": name, "requested": len(queries), "converged": count}), flush=True)
        named = read_json(output / "splits/named_central.json")
        states = pool.full_states(named)
    finally:
        pool.close()
    named_rows, full_rows, compositions, keys = [], [], [], []
    for query, state in zip(named, states):
        row = dict(query, **proof, prerequisite_registration_sha256=prerequisite["registration_sha256"],
                   status=state["status"], reason=state["reason"])
        row["full_state_sha256"] = __import__("hashlib").sha256(json_bytes(state)).hexdigest()
        if state["status"] == "converged":
            row.update({key: state[key] for key in ("cp4_J_kg_K", "R4_J_kg_K", "gamma4")})
        named_rows.append(row)
        full = dict(row, species_row_index=None)
        if state["status"] == "converged":
            full.update({key: state[key] for key in state if key not in ("Y4", "input_state", "status", "reason")})
            full["species_row_index"] = len(compositions)
            compositions.append(state["Y4"]); keys.append(query)
        full_rows.append(full)
    csv_once(output / "named_central_properties.csv", named_rows)
    csv_once(output / "sealed/named_central_full_state.csv", full_rows)
    stream = io.BytesIO(); np.savez(stream, Y4=np.asarray(compositions, dtype=np.float64).reshape(-1, 492),
        species_order=np.asarray(properties["species_order"]),
        **{key: np.asarray([q[key] for q in keys]) for key in ("named_case_id", "draw_id", "input_sha256")})
    write_once(output / "sealed/named_central_species.npz", stream.getvalue())
    paths += ["named_central_properties.csv", "sealed/named_central_full_state.csv", "sealed/named_central_species.npz", "frozen_properties.json"]
    converged["named_central"] = len(compositions)
    requested = {"train": 4096, "validation": 1024, "test": 2048, "ranking_test": 4096, "named_central": 68}
    check_child_authorization(spec)
    write_once(output / "proofs/generation_summary.json", {
        "scientific_coverage": "PASS" if converged == requested else "WITH_INVALID_REFERENCES",
        "counts": {"train_requested": 4096, "validation_requested": 1024, "test_requested": 2048,
                   "ranking_requested": 4096, "named_central_requested": 68, "total_requested": 11332,
                   "converged_by_split": converged, "failed_by_split": {k: v-converged[k] for k, v in requested.items()}},
        "outputs": {str((output / path).relative_to(root)): sha256_file(output / path) for path in paths}})


def pipeline(root, output, registration, backend=None, device="auto"):
    from scripts.phase8 import scientific_workflow_gate as gate
    from .score import seal_predictions, score_all, deployment_receipt
    from .timing import measure, source_diagnostics, finalize_timing
    from .study import screen
    from .train import train_all
    from simulation.ml_backend import resolve_backend
    backend = resolve_backend(backend)
    # The one-thread-per-worker policy must be enforced before any Torch
    # import: OMP/OpenBLAS/MKL/Accelerate read these env vars at library
    # init, so setting them after import is a no-op for that process.
    for key in THREAD_ENV:
        if os.environ.get(key, "1") != "1":
            raise ValueError("Registered one-thread-per-worker policy is required")
        os.environ[key] = "1"
    # Reject a bad backend/device or unavailable CUDA before any gate or lease;
    # MLX itself is still imported only after authorization.
    if backend == "torch":
        from .train_torch import resolve_device
        resolved_device = resolve_device(device)
        import torch
        # A Torch already imported earlier in this process missed the env
        # setup above; reapply the one-thread policy through the API too.
        torch.set_num_threads(1)
    elif backend != "mlx" or device != "auto":
        raise ValueError("MLX uses its default device; --device applies to --backend torch")
    else:
        resolved_device = device
    root = Path(root).resolve()
    reg, reg_sha = load_registration(root, registration)
    output = contained(root, output)
    if str(output.relative_to(root)) != reg["artifact_root"]:
        raise ValueError("Only the one registered attempt is permitted")
    if output.exists():
        raise FileExistsError("Attempt already exists; no fresh retry, overwrite or score reopening")
    context = gate.prepare_context(root, registration, require_g0=True)
    context.require_idle_ac()
    prerequisite = verify_named_prerequisite(root, reg, context.binary_sha256)
    run = context.acquire_run(output, reg_sha)
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=True)
        write_once(output / "execution.log", b"P8-S owned pipeline started\n")
        command_started=utc()
        write_once(output/"command.start.json",{"argv":list(sys.orig_argv),"native_command":native_command(os.getpid()),
            "pid":os.getpid(),"birth":birth(os.getpid()),"registration_sha256":reg_sha,"identity":context.identity,
            "binary_path":str(context.binary_path.relative_to(root)),"binary_sha256":context.binary_sha256,
            "started":command_started})
        write_once(output / "prerequisite.json", prerequisite)
        freeze(root, output, reg, reg_sha, context, run, backend)
        source_diagnostics(root,output,reg,context,run,backend)
        wait_generation(root, output, context, run, reg_sha)
        train_all(output, reg, reg_sha, run, backend=backend, device=device)
        seal_predictions(output, reg, reg_sha, context, run, backend=backend)
        scored = score_all(root, output, reg, reg_sha, context, run, backend=backend)
        timing = measure(root, output, reg, reg_sha, context, run, backend, resolved_device)
        study = screen(root, output, reg, reg_sha, context, run, backend=backend)
        timing = finalize_timing(root,output,started,timing,study,scored)
        report = {"registration_id": reg["id"], "registration_sha256": reg_sha,
            "metrics": scored, "timing": timing, "study": study,
            "total_pipeline_seconds": time.perf_counter()-started,
            "teacher_full_cycle_requests": 49421, "diagnostic_unsafe": not scored["fidelity_pass"]}
        write_once(output / "report.json", report)
        deployment_receipt(output, reg, reg_sha, scored, timing)
        write_once(output / "README.md", b"P8-S quantitative artifacts\n\nSee the registration for units, schemas and limitations. Product prediction requires a matching deployment receipt. Fixed-draw bands describe registered sensitivity draws, not confidence intervals. GPU32 timing does not substitute CPU64 scoring.\n")
        append_progress(output / "progress.jsonl", {"stage": "complete", "utc": utc()})
        run.assert_current()
        expected = reg["provenance"]["successful_release"]["expected_outputs"]
        verdict="PASS" if read_json(output/"deployment_receipt.json")["state"]=="PASS" else "FAIL"
        write_once(output/"command.exit.json",{"argv":list(sys.orig_argv),"pid":os.getpid(),"birth":birth(os.getpid()),
            "started":command_started,"finished":utc(),"identity":context.identity,"exit_code":0 if verdict=="PASS" else 1,
            "in_process_completed":True,"scientific_verdict":verdict,"completed_phases":["freeze","source_checks","generate","train","seal","score","timing","study"]})
        hashes = {str(path.relative_to(root)):sha256_file(path) for path in output.rglob("*") if path.is_file()}
        if not set(expected)<=set(hashes):raise RuntimeError("Registered artifact coverage incomplete")
        terminal=run.release({"state": verdict, "status": verdict, "exit_code": 0 if verdict=="PASS" else 1, "identity": context.identity,
            "core_sha256": context.binary_sha256, "outputs_complete": True, "expected_outputs": expected,
            "execution_complete":True,"scientific_verdict":verdict,"scientific_only_failure":verdict=="FAIL",
            "artifact_hashes": hashes, "fidelity_pass": scored["fidelity_pass"],
            "operational_pass": timing["operational_pass"], "deployment_pass": read_json(output / "deployment_receipt.json")["state"] == "PASS"})
        return 0 if terminal["status"]=="PASS" else 1
    except BaseException as error:
        unresolved = list((output/"proofs/spawns").glob("*.starting.json"))
        ambiguous = any(not path.with_name(path.name.replace(".starting.json",".complete.json")).exists() for path in unresolved)
        ambiguous |= (output/"proofs/generation_starting.json").exists() and not (output/"proofs/generation_exit.json").exists()
        if ambiguous:
            # Never release an owner after a spawn gap. Only this exact Run may
            # retain its lease; no foreign lease cleanup or process inference.
            run._assert_owned()
            lease=read_json(run.lease_path);lease["state"]="AMBIGUOUS_CHILD";run._write_lease(lease)
            write_once(output/"ambiguous_child_error.json",{"execution_complete":False,"scientific_verdict":"ERROR",
                "reason":str(error),"unresolved_starting_paths":[str(p.relative_to(root)) for p in unresolved]})
            raise
        run.release({"state": "ERROR", "status": "ERROR", "exit_code": 1, "identity": context.identity,
            "core_sha256": context.binary_sha256, "outputs_complete": False,
            "artifact_hashes": {str(path.relative_to(root)): sha256_file(path) for path in output.rglob("*") if path.is_file()},
            "execution_complete":False,"scientific_verdict":"ERROR", "ambiguous_child":ambiguous, "exception_type": type(error).__name__, "reason": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "_generate", "freeze", "generate", "train", "seal", "score", "timing", "study"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--registration", default=REGISTRATION)
    parser.add_argument("--out", default="outputs/phase8/saf_surrogate/attempt_001")
    parser.add_argument("--spec")
    parser.add_argument("--backend", choices=("mlx", "torch"), default=os.environ.get("CATJET_ML_BACKEND", "torch"),
                        help="training/scoring backend (default: CATJET_ML_BACKEND or torch)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="torch training device; auto uses CPU. All primary scoring uses CPU float64")
    args = parser.parse_args(argv)
    if args.backend == "mlx" and args.device != "auto":
        parser.error("--device applies to --backend torch; MLX uses its default device")
    if args.command == "_generate":
        if not args.spec:
            parser.error("Internal generation requires exact parent authorization")
        generate_child(Path(args.spec).resolve())
    elif args.command == "run":
        return pipeline(args.root, args.out, args.registration, args.backend, args.device)
    else:
        parser.error("Standalone phases are refused; use the one owned run pipeline")


if __name__ == "__main__":
    sys.exit(main())
