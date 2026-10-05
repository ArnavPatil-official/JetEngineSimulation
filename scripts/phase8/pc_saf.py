"""Run the fixed SAF study on a locally built PC core, with the same sole score.

The output directory and scientific procedures remain those of P8-S. Local
source/core/G0 records identify the PC computation; historical Mac evidence is
neither required nor manufactured. Importing this module performs no studies.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

from .saf_surrogate.registration import (REGISTRATION, contained, load_registration,
    read_json, sha256_file, verify_artifacts, verify_named_prerequisite, write_once)

PREREQUISITE = "outputs/phase7/p73_a1_cpp_20261004"
PC_SOURCES = ("scripts/phase8/pc_saf.py", "scripts/phase8/pc_runtime.py",
              "scripts/phase8/saf_surrogate/train_torch.py",
              "scripts/phase8/ml/models_torch.py", "scripts/phase8/ml/spec.py")


def _inside(root, path):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (root/path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Producer path leaves this repository")
    return path


def validate_local_terminal(root, registration_path, output_dir, *, expected_binary_sha256=None,
                            artifact_paths=None, allow_scientific_fail=False):
    """Verify an actual released PC producer without decoding sealed targets.

    The metadata-only prerequisite projection does not open or hash any target
    outside that projection. Complete artifact coverage is checked against the
    producer's bound hash manifest; requested artifacts are then byte-verified.
    """
    root = Path(root).resolve()
    out = _inside(root, output_dir)
    reg_path = _inside(root, registration_path)
    reg = read_json(reg_path)
    expected_out = reg.get("artifact_root") or reg.get("outputs", {}).get("directory")
    if expected_out is not None and out != (root/expected_out).resolve():
        raise ValueError("Producer output differs from its canonical registered path")
    terminal = read_json(out/"terminal.json")
    reservation = read_json(out/"reservation.json")
    released = read_json(out/"released_lease.json")
    identity = terminal.get("identity", {})
    if identity.get("schema") != "pc-local-identity-v1":
        raise ValueError("Producer is not a local PC computation")
    if any(record.get("identity") != identity for record in (reservation, released)):
        raise ValueError("Producer reservation/release source identity differs")
    if identity.get("registration_sha256") != sha256_file(reg_path):
        raise ValueError("Producer registration drift")
    verify_artifacts(root, identity["source_hashes"])
    binary = _inside(root, identity["binary_path"])
    core_hash = sha256_file(binary)
    if core_hash != identity.get("binary_sha256") or (expected_binary_sha256 is not None
            and core_hash != expected_binary_sha256):
        raise ValueError("Producer/consumer local core mismatch")
    g0_path = _inside(root, identity["g0_record_path"])
    if sha256_file(g0_path) != identity.get("g0_record_sha256"):
        raise ValueError("Producer local G0 record drift")
    g0 = read_json(g0_path)
    if g0.get("verdict") != "PASS" or g0.get("core") != {
            "path": identity["binary_path"], "sha256": core_hash}:
        raise ValueError("Producer has no matching local G0 PASS")
    complete = (terminal.get("status") in ("PASS", "COMPLETE")
                and terminal.get("exit_code") == 0 and terminal.get("outputs_complete") is True)
    failed = (allow_scientific_fail and terminal.get("status") == "FAIL"
              and terminal.get("execution_complete") is True
              and terminal.get("scientific_verdict") == "FAIL"
              and isinstance(terminal.get("exit_code"), int) and terminal["exit_code"] != 0
              and terminal.get("outputs_complete") is True)
    if not (complete or failed) or terminal.get("errors") or terminal.get("identity_problems"):
        raise ValueError("Producer has no complete local execution proof")
    reservation_hash = sha256_file(out/"reservation.json")
    if (terminal.get("reservation_sha256") != reservation_hash
            or released.get("reservation_sha256") != reservation_hash
            or released.get("terminal_sha256") != sha256_file(out/"terminal.json")
            or released.get("state") != "RELEASED"
            or released.get("owner_pid") != reservation.get("owner_pid")
            or released.get("owner_birth") != reservation.get("owner_birth")):
        raise ValueError("Producer reservation/terminal/released lease does not agree")
    hashes = terminal.get("artifact_hashes", {})
    expected = terminal.get("expected_outputs", [])
    if not expected or not set(expected).issubset(hashes):
        raise ValueError("Producer artifact coverage is incomplete")
    # Check existence and opaque hashes without consuming held-out numeric data.
    for name in expected:
        path = _inside(root, name)
        if not path.is_file() or path.is_symlink() or not isinstance(hashes[name], str) or len(hashes[name]) != 64:
            raise ValueError("Producer expected artifact is missing or unbound")
    selected = list(hashes) if artifact_paths is None else list(artifact_paths)
    for name in selected:
        if name not in hashes or sha256_file(_inside(root, name)) != hashes[name]:
            raise ValueError(f"Producer artifact drift: {name}")
    command = read_json(out/"command.exit.json")
    if (command.get("identity") != identity or command.get("in_process_completed") is not True
            or command.get("exit_code") != terminal.get("exit_code")):
        raise ValueError("Producer command exit differs from terminal")
    return {"status": terminal["status"], "scientific_verdict": terminal.get("scientific_verdict"),
            "identity": identity, "core_sha256": core_hash,
            "terminal_sha256": sha256_file(out/"terminal.json"), "artifact_hashes": hashes}


def platform_info():
    """Actual Linux/WSL metadata for the existing SAF environment artifact."""
    from .pc_runtime import read_power_linux
    topology = Path("/sys/devices/system/cpu")
    physical = set()
    for cpu in topology.glob("cpu[0-9]*"):
        try:
            if (cpu/"online").exists() and (cpu/"online").read_text().strip() == "0":
                continue
            package = int((cpu/"topology/physical_package_id").read_text().strip())
            core = int((cpu/"topology/core_id").read_text().strip())
            if package < 0 or core < 0:
                raise ValueError("CPU topology is unknown")
            physical.add((package, core))
        except (OSError, ValueError):
            raise RuntimeError(f"Cannot read physical CPU topology: {cpu}") from None
    if not physical:
        raise RuntimeError("Cannot determine the physical CPU count")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    memory_bytes = int(meminfo["MemTotal"].split()[0])*1024
    return {"physical_cores": len(physical), "platform": platform.platform(),
            "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                         "logical_cores": os.cpu_count(), "memory_bytes": memory_bytes,
                         "physical_core_count_source": "Linux sysfs visible online CPU topology"},
            "power": read_power_linux(), "execution_profile": "pc"}


def execute(root, context_factory, *, device="auto", backend="torch"):
    """Run freeze → source checks → generate → train → seal → one score → measure/study.

    ``context_factory(root, registration_path)`` returns the PCContext tied to
    this run's actual local G0 and built core. A completed scientific FAIL is
    returned honestly; execution errors retain their artifacts and raise.
    """
    from .saf_surrogate import run as saf_run
    # Establish native-library thread policy before importing NumPy/Torch.
    for name in saf_run.THREAD_ENV:
        if os.environ.get(name, "1") != "1":
            raise ValueError("Registered one-thread-per-worker policy is required")
        os.environ[name] = "1"
    if backend != "torch":
        raise ValueError("The PC SAF route uses the PyTorch backend")
    from .saf_surrogate.train_torch import resolve_device
    resolved_device = resolve_device(device)
    import torch
    torch.set_num_threads(1)
    from .saf_surrogate.score import seal_predictions, score_all, deployment_receipt
    from .saf_surrogate.train import train_all
    from .saf_surrogate.timing import source_diagnostics, measure, finalize_timing
    from .saf_surrogate.study import screen

    root = Path(root).resolve()
    reg, reg_sha = load_registration(root, REGISTRATION)
    output = contained(root, reg["artifact_root"])
    if output.exists():
        raise FileExistsError("Registered SAF attempt already exists; no retry, overwrite or score reopening")
    context = context_factory(root, REGISTRATION)
    if getattr(context, "execution_profile", None) != "pc":
        raise ValueError("The portable SAF route requires a real PCContext")
    context.require_idle_ac()
    prerequisite = verify_named_prerequisite(root, reg, context.binary_sha256,
        validate=validate_local_terminal)
    # Bind the backend and wrappers as well as the registered shared sources.
    sources = dict(context.identity["source_hashes"])
    for name in PC_SOURCES:
        if sources.get(name) != sha256_file(root/name):
            raise ValueError(f"PC context did not capture the actual SAF backend source: {name}")
    sources.update({name: sha256_file(root/name) for name in PC_SOURCES})
    sources.update({name: sha256_file(root/name) for name in reg["relevant_files"]["read_only"]
                    if (root/name).is_file()})
    run = context.acquire_run(output, reg_sha)
    run._saf_children = []
    started = time.perf_counter()
    command_started = saf_run.utc()
    try:
        write_once(output/"execution.log", b"P8-S local PC pipeline started\n")
        write_once(output/"command.start.json", {"argv": list(sys.orig_argv), "pid": os.getpid(),
            "birth": saf_run.birth(os.getpid()), "identity": context.identity,
            "registration_sha256": reg_sha, "binary_sha256": context.binary_sha256,
            "started": command_started, "execution_profile": "pc"})
        write_once(output/"prerequisite.json", prerequisite)
        write_once(output/"pc_origin.json", {"execution_profile": "pc", "identity": context.identity,
            "g0": context.original_context["g0"], "binary_sha256": context.binary_sha256,
            "source_hashes": sources, "historical_mac_chain_completion_claimed": False})
        dependencies = {"local_pc_origin": str((output/"pc_origin.json").relative_to(root)),
                        "named_prerequisite_terminal": str((root/PREREQUISITE/"terminal.json").relative_to(root))}
        saf_run.freeze(root, output, reg, reg_sha, context, run, backend,
            platform_info=platform_info, source_hashes=sources, provenance_dependencies=dependencies)
        source_diagnostics(root, output, reg, context, run, backend)
        saf_run.wait_generation(root, output, context, run, reg_sha)
        train_all(output, reg, reg_sha, run, backend=backend, device=resolved_device)
        seal_predictions(output, reg, reg_sha, context, run)
        scored = score_all(root, output, reg, reg_sha, context, run)
        timing = measure(root, output, reg, reg_sha, context, run, backend, resolved_device)
        study = screen(root, output, reg, reg_sha, context, run)
        timing = finalize_timing(root, output, started, timing, study, scored)
        report = {"registration_id": reg["id"], "registration_sha256": reg_sha,
            "execution_profile": "pc", "metrics": scored, "timing": timing, "study": study,
            "total_pipeline_seconds": time.perf_counter()-started,
            "teacher_full_cycle_requests": 49421, "diagnostic_unsafe": not scored["fidelity_pass"],
            "historical_mac_chain_completion_claimed": False}
        write_once(output/"report.json", report)
        deployment_receipt(output, reg, reg_sha, scored, timing)
        write_once(output/"README.md", b"P8-S local PC quantitative artifacts.\n\nNumPy CPU64 is the scoring/inference path. Torch CUDA measurements, when available, use float32 and include synchronized transfers and CPU64 postprocessing. A measured scientific failure remains diagnostic; historical Mac chain completion is not claimed.\n")
        verdict = "PASS" if read_json(output/"deployment_receipt.json")["state"] == "PASS" else "FAIL"
        write_once(output/"command.exit.json", {"argv": list(sys.orig_argv), "pid": os.getpid(),
            "birth": saf_run.birth(os.getpid()), "started": command_started, "finished": saf_run.utc(),
            "identity": context.identity, "exit_code": 0 if verdict == "PASS" else 1,
            "in_process_completed": True, "scientific_verdict": verdict,
            "completed_phases": ["freeze", "source_checks", "generate", "train", "seal", "score", "timing", "study"]})
        run.assert_current()
        expected = reg["provenance"]["successful_release"]["expected_outputs"]
        hashes = {str(path.relative_to(root)): sha256_file(path) for path in output.rglob("*") if path.is_file()}
        if not set(expected).issubset(hashes):
            raise RuntimeError("Registered SAF artifact coverage is incomplete")
        return run.release({"status": verdict, "state": verdict, "exit_code": 0 if verdict == "PASS" else 1,
            "registration_id": reg["id"], "registration_sha256": reg_sha,
            "execution_profile": "pc", "core_sha256": context.binary_sha256,
            "binary_sha256": context.binary_sha256, "outputs_complete": True,
            "expected_outputs": expected, "artifact_hashes": hashes,
            "execution_complete": True, "scientific_verdict": verdict, "scientific_only_failure": verdict == "FAIL",
            "fidelity_pass": scored["fidelity_pass"], "operational_pass": timing["operational_pass"],
            "deployment_pass": verdict == "PASS", "errors": []})
    except BaseException as error:
        if not getattr(run, "_released", False):
            # An unresolved spawn remains an owned ambiguous lease. Otherwise
            # PCRun.release checks actual child liveness before publishing ERROR.
            starts = list((output/"proofs/spawns").glob("*.starting.json"))
            ambiguous = any(not path.with_name(path.name.replace(".starting.json", ".complete.json")).exists()
                            for path in starts)
            ambiguous |= ((output/"proofs/generation_starting.json").exists()
                          and not (output/"proofs/generation_exit.json").exists())
            hashes = {str(path.relative_to(root)): sha256_file(path) for path in output.rglob("*") if path.is_file()}
            run.release({"status": "ERROR", "state": "ERROR", "exit_code": 1,
                "execution_profile": "pc", "outputs_complete": False, "execution_complete": False,
                "artifact_hashes": hashes, "ambiguous_child": ambiguous,
                "errors": [f"{type(error).__name__}: {error}"]})
        raise
