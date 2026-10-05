"""PC provenance/ownership fixtures; no core import, build, or project solve."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase8 import pc_runtime as pc


@pytest.fixture
def local_context(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout="fixture-head\n")))
    monkeypatch.setattr(pc, "process_birth", lambda pid: "fixture-birth")
    monkeypatch.setattr(pc, "require_idle_ac_linux", lambda: {"battery_present": False, "on_ac": True})
    source = tmp_path / "scripts/helper.py"
    registration = tmp_path / "docs/registration.json"
    core = tmp_path / "cpp/build_pc/catjet_core.so"
    for path, data in ((source, b"original-source"), (registration, b"{}"), (core, b"original-core")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    args = {"binary_path": core, "binary_sha256": pc.sha256_file(core),
            "source_hashes": {"scripts/helper.py": pc.sha256_file(source)},
            "g0_record": {"verdict": "PASS", "core": {
                "path": "cpp/build_pc/catjet_core.so", "sha256": pc.sha256_file(core)}}}
    context = pc.PCContext(tmp_path, "docs/registration.json", **args)
    return context, args, {"source": source, "registration": registration, "core": core}


@pytest.mark.parametrize("verdict", ["FAIL", "ERROR", None])
def test_scientific_context_refuses_nonpassing_g0(local_context, verdict):
    context, args, _ = local_context
    args = copy.deepcopy(args)
    args["g0_record"]["verdict"] = verdict
    with pytest.raises(pc.PCGateError, match="G0 parity PASS"):
        pc.PCContext(context.root, context.registration, **args)


def test_g0_evidence_must_use_selected_core(local_context):
    context, args, _ = local_context
    args = copy.deepcopy(args)
    args["g0_record"]["core"]["sha256"] = "foreign"
    with pytest.raises(pc.PCGateError, match="different core"):
        pc.PCContext(context.root, context.registration, **args)


def test_recorded_g0_bytes_are_rechecked_at_boundary_and_release(local_context):
    context, args, _ = local_context
    evidence = context.root / "outputs/metadata/g0.json"
    pc.write_once(evidence, args["g0_record"])
    bound = pc.PCContext(context.root, context.registration, **args, identity_extra={
        "g0_record_path": "outputs/metadata/g0.json", "g0_record_sha256": pc.sha256_file(evidence)})
    run = bound.acquire_run("outputs/fixture", bound.identity["registration_sha256"])
    evidence.write_text('{"verdict": "FAIL"}')
    with pytest.raises(pc.PCGateError, match="G0 evidence hash drift"):
        run.assert_current()
    assert run.release({"status": "PASS"})["status"] == "ERROR"


def test_recorded_g0_document_must_equal_actual_pass_context(local_context):
    context, args, _ = local_context
    evidence = context.root / "outputs/metadata/g0.json"
    pc.write_once(evidence, {**args["g0_record"], "verdict": "FAIL"})
    with pytest.raises(pc.PCGateError, match="selected PASS/core"):
        pc.PCContext(context.root, context.registration, **args, identity_extra={
            "g0_record_path": "outputs/metadata/g0.json", "g0_record_sha256": pc.sha256_file(evidence)})


def test_light_epoch_guard_does_not_reread_source_or_core(local_context, monkeypatch):
    context, args, paths = local_context
    evidence = context.root / "outputs/metadata/g0.json"
    pc.write_once(evidence, args["g0_record"])
    bound = pc.PCContext(context.root, context.registration, **args, identity_extra={
        "g0_record_path": "outputs/metadata/g0.json", "g0_record_sha256": pc.sha256_file(evidence)})
    run = bound.acquire_run("outputs/fixture", bound.identity["registration_sha256"])
    original_hash = pc.sha256_file
    reads, power_checks = [], []
    def small_hash(path):
        path = Path(path)
        assert path not in (paths["core"], paths["source"]), "Epoch guard read full provenance bytes"
        reads.append(path)
        return original_hash(path)
    monkeypatch.setattr(pc, "sha256_file", small_hash)
    monkeypatch.setattr(pc, "require_idle_ac_linux", lambda: power_checks.append("power"))
    paths["source"].write_bytes(b"changed")
    run.light_training_check()
    assert set(reads) == {run.out / "reservation.json", paths["registration"], evidence}
    assert power_checks == ["power"]
    monkeypatch.setattr(pc, "sha256_file", original_hash)
    with pytest.raises(pc.PCGateError, match="source hash drift"):
        run.assert_current()
    assert run.release({"status": "PASS"})["status"] == "ERROR"


def test_light_epoch_guard_still_detects_registration_drift(local_context):
    context, _, paths = local_context
    run = context.acquire_run("outputs/fixture", context.identity["registration_sha256"])
    paths["registration"].write_bytes(b"changed")
    with pytest.raises(pc.PCGateError, match="registration hash drift"):
        run.light_training_check()
    assert run.release({"status": "PASS"})["status"] == "ERROR"


def test_light_epoch_guard_still_detects_g0_drift(local_context):
    context, args, _ = local_context
    evidence = context.root / "outputs/metadata/g0.json"
    pc.write_once(evidence, args["g0_record"])
    bound = pc.PCContext(context.root, context.registration, **args, identity_extra={
        "g0_record_path": "outputs/metadata/g0.json", "g0_record_sha256": pc.sha256_file(evidence)})
    run = bound.acquire_run("outputs/fixture", bound.identity["registration_sha256"])
    evidence.write_text('{"verdict": "FAIL"}')
    with pytest.raises(pc.PCGateError, match="G0 evidence hash drift"):
        run.light_training_check()
    assert run.release({"status": "PASS"})["status"] == "ERROR"


@pytest.mark.parametrize("changed", ["source", "registration", "core"])
def test_boundary_detects_drift_and_release_cannot_claim_pass(local_context, changed):
    context, _, paths = local_context
    run = context.acquire_run("outputs/fixture", context.identity["registration_sha256"])
    captured = copy.deepcopy(run.identity)
    paths[changed].write_bytes(b"changed")
    with pytest.raises(pc.PCGateError, match="drift"):
        run.assert_current()
    terminal = run.release({"status": "PASS", "state": "PASS", "exit_code": 0,
                            "outputs_complete": True})
    assert terminal["status"] == "ERROR" and terminal["exit_code"] != 0
    assert terminal["outputs_complete"] is False and terminal["identity_problems"]
    assert terminal["identity"] == captured
    assert not run.lease_path.exists()
    assert (run.out / "released_lease.json").is_file()


def test_missing_source_is_drift(local_context):
    context, _, paths = local_context
    run = context.acquire_run("outputs/fixture", context.identity["registration_sha256"])
    paths["source"].unlink()
    with pytest.raises(pc.PCGateError, match="unreadable"):
        run.assert_current()
    assert run.release({"status": "PASS"})["status"] == "ERROR"


def test_extra_metadata_cannot_forge_core_or_source_identity(local_context):
    context, args, _ = local_context
    with pytest.raises(pc.PCGateError, match="cannot replace"):
        pc.PCContext(context.root, context.registration, **args,
                     identity_extra={"binary_sha256": "foreign"})


def test_identity_mutation_is_not_a_new_authoritative_snapshot(local_context):
    context, _, _ = local_context
    run = context.acquire_run("outputs/fixture", context.identity["registration_sha256"])
    context.identity["source_hashes"].clear()
    with pytest.raises(pc.PCGateError, match="identity was changed"):
        run.assert_current()
    assert run.release({"status": "PASS"})["status"] == "ERROR"


def test_existing_lease_refuses_second_run_without_creating_output(local_context):
    context, _, _ = local_context
    run = context.acquire_run("outputs/first", context.identity["registration_sha256"])
    lease_before = run.lease_path.read_bytes()
    with pytest.raises(FileExistsError):
        context.acquire_run("outputs/second", context.identity["registration_sha256"])
    assert run.lease_path.read_bytes() == lease_before
    assert not (context.root / "outputs/second").exists()
    run.release({"status": "ERROR"})


def test_changed_reservation_cannot_be_released_as_owned(local_context):
    context, _, _ = local_context
    run = context.acquire_run("outputs/fixture", context.identity["registration_sha256"])
    (run.out / "reservation.json").write_text("{}")
    with pytest.raises(pc.PCGateError, match="reservation bytes"):
        run.assert_current()
    with pytest.raises(pc.PCGateError, match="reservation bytes"):
        run.release({"status": "PASS"})
    assert run.lease_path.exists() and not (run.out / "terminal.json").exists()


def test_unknown_child_exit_keeps_lease(local_context, monkeypatch):
    context, _, _ = local_context
    run = context.acquire_run("outputs/fixture", context.identity["registration_sha256"])
    run.record_children([{"pid": os.getpid() + 1, "birth": "fixture-birth", "argv": ["fixture"]}])
    monkeypatch.setattr(pc, "liveness", lambda pid, birth: "unknown")
    with pytest.raises(pc.PCGateError, match="Child exit unproven"):
        run.release({"status": "PASS"})
    assert pc.read_json(run.lease_path)["state"] == "AMBIGUOUS_CHILD"
    assert not (run.out / "terminal.json").exists()


def test_released_run_cannot_touch_subsequent_lease(local_context):
    context, _, _ = local_context
    first = context.acquire_run("outputs/first", context.identity["registration_sha256"])
    first.release({"status": "ERROR"})
    second = context.acquire_run("outputs/second", context.identity["registration_sha256"])
    before = second.lease_path.read_bytes()
    with pytest.raises(pc.PCGateError, match="Released run"):
        first.release({"status": "PASS"})
    assert second.lease_path.read_bytes() == before
    second.release({"status": "ERROR"})


def test_cached_module_does_not_adopt_changed_binary_bytes(tmp_path, monkeypatch):
    path = tmp_path / "fixture_core.so"
    path.write_bytes(b"original")
    module = SimpleNamespace(__file__=str(path), _pc_binary_sha256=pc.sha256_file(path))
    monkeypatch.setattr(pc.importlib.machinery, "EXTENSION_SUFFIXES", [".so"])
    monkeypatch.setitem(sys.modules, "fixture_core", module)
    assert pc.select_core(tmp_path, "fixture_core")[0] is module
    path.write_bytes(b"changed")
    with pytest.raises(pc.PCGateError, match="selected binary bytes"):
        pc.select_core(tmp_path, "fixture_core")


def test_multiple_compatible_binary_candidates_are_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(pc.importlib.machinery, "EXTENSION_SUFFIXES", [".abi.so", ".so"])
    for suffix in pc.importlib.machinery.EXTENSION_SUFFIXES:
        (tmp_path / f"fixture_core{suffix}").write_bytes(b"fixture")
    with pytest.raises(pc.PCGateError, match="Ambiguous"):
        pc.select_core(tmp_path, "fixture_core")


@pytest.mark.parametrize("kind", ["Mains", "USB", "USB_C", "USB_PD", "USB_PD_DRP"])
def test_connected_linux_usb_chargers_are_external_power(tmp_path, kind):
    battery, supply = tmp_path / "BAT0", tmp_path / "charger"
    battery.mkdir()
    supply.mkdir()
    (battery / "type").write_text("Battery\n")
    (supply / "type").write_text(kind + "\n")
    (supply / "online").write_text("1\n")
    assert pc.read_power_linux(tmp_path)["on_ac"] is True
    (supply / "online").write_text("0\n")
    assert pc.read_power_linux(tmp_path)["on_ac"] is False
