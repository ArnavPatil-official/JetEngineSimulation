"""Independent mocked PC receipt checks; no project labels or numerical runs."""
import json
import os

import pytest

from scripts.phase8 import pc_python_runtime as runtime
from scripts.phase8 import python_pc
from scripts.phase8.pc_runtime import PCGateError, read_json, sha256_file


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def pc_fixture(tmp_path, monkeypatch):
    from simulation import runtime as portable
    from scripts.phase8 import pc_runtime
    registrations = {
        "p73": {"outputs": {"directory": "outputs/phase8/mock_p73"}},
        "saf": {"artifact_root": "outputs/phase8/mock_saf"},
        "nozzle": {"outputs": {"root": "outputs/phase8/mock_nozzle"}},
    }
    for key, value in registrations.items():
        put_json(tmp_path/python_pc.REGISTRATIONS[key], value)
    put_json(tmp_path/runtime.AMENDMENT, {"fixture": "prospective backend amendment"})
    stub = tmp_path/"simulation/stub.py"
    stub.parent.mkdir()
    stub.write_text("# Small scientific source fixture\n")
    sources = {"simulation/stub.py": sha256_file(stub)}
    metadata = tmp_path/"outputs/phase8/mock_pipeline"
    metadata.mkdir(parents=True)
    parity = {"status": "PASS", "rows": 20, "source_hashes": sources}
    power = {"on_ac": True, "battery_present": False}
    monkeypatch.setattr(portable.platform, "system", lambda: "Linux")
    monkeypatch.setattr(portable.platform, "platform", lambda: "Linux fixture")
    monkeypatch.setattr(pc_runtime, "read_power_linux", lambda: dict(power))
    monkeypatch.setattr(runtime, "process_birth", lambda pid: "fixture-owner-birth" if pid == os.getpid() else "fixture-child-birth")
    monkeypatch.setattr(runtime, "host_identity", lambda: "fixture-host")
    monkeypatch.setattr(runtime, "liveness", lambda *args, **kwargs: "dead")
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: "a"*40)
    # A Linux route must never invoke pmset/sysctl/caffeinate or a native shell.
    def no_native_command(*args, **kwargs):
        raise AssertionError(f"Unexpected native command: {args}")
    monkeypatch.setattr(runtime.subprocess, "run", no_native_command)
    return {"root": tmp_path, "metadata": metadata, "sources": sources,
            "parity": parity, "power": power}


def context(fixture):
    return runtime.PythonContext(fixture["root"], python_pc.REGISTRATIONS["saf"],
        sources=fixture["sources"], metadata_dir=fixture["metadata"], parity=fixture["parity"])


def complete_stage(fixture, letter, *, artifact_paths=()):
    ctx = context(fixture)
    folder = fixture["metadata"]/"runs"/letter
    run = ctx.acquire_run(folder, ctx.identity["registration_sha256"])
    run.record_children([])
    hashes = {p.relative_to(fixture["root"]).as_posix(): sha256_file(p) for p in artifact_paths}
    terminal = run.release({"status": "COMPLETE", "state": "COMPLETE", "exit_code": 0,
        "execution_complete": True, "outputs_complete": True, "scientific_verdict": "PASS",
        "artifact_hashes": hashes, "expected_outputs": list(hashes)})
    return ctx, folder, terminal


def test_linux_python_context_binds_real_simulator_without_mac_calls(pc_fixture):
    ctx = context(pc_fixture)
    assert ctx.simulator_backend == "python" and ctx.binary_path is None and ctx.binary_sha256 is None
    assert ctx.identity["simulator"]["name"] == "python-v6"
    assert runtime.fingerprint(ctx.simulator_identity) == ctx.simulator_identity_sha256
    assert ctx.require_idle_ac()["source"] == "Linux power_supply sysfs"
    run = ctx.acquire_run(pc_fixture["metadata"]/"runs/ownership", ctx.identity["registration_sha256"])
    run.record_children([])
    run.light_training_check()
    terminal = run.release({"status": "COMPLETE", "exit_code": 0, "artifact_hashes": {}})
    assert terminal["identity"] == ctx.identity
    assert not ctx.pc_lease_path.exists()


@pytest.mark.parametrize("field,value", [("host", "foreign-host"), ("owner_pid", -1),
                                           ("owner_birth", "reused-process")])
def test_python_lease_rejects_host_pid_or_birth_change(pc_fixture, field, value):
    ctx = context(pc_fixture)
    run = ctx.acquire_run(pc_fixture["metadata"]/"runs/ownership", ctx.identity["registration_sha256"])
    lease = read_json(run.lease_path)
    lease[field] = value
    put_json(run.lease_path, lease)
    with pytest.raises(PCGateError, match="ownership"):
        run.light_training_check()
    assert not (run.out/"terminal.json").exists()


def test_python_light_guard_requires_power_and_release_downgrades_drift(pc_fixture):
    ctx = context(pc_fixture)
    run = ctx.acquire_run(pc_fixture["metadata"]/"runs/power", ctx.identity["registration_sha256"])
    pc_fixture["power"]["on_ac"] = False
    with pytest.raises(RuntimeError, match="external power"):
        run.light_training_check()
    pc_fixture["power"]["on_ac"] = True
    (pc_fixture["root"]/"simulation/stub.py").write_text("# Drift after reservation\n")
    terminal = run.release({"status": "COMPLETE", "exit_code": 0, "outputs_complete": True,
                            "artifact_hashes": {}})
    assert terminal["status"] == "ERROR" and terminal["outputs_complete"] is False
    assert "source/dependency drift" in terminal["errors"][0]
    assert not run.lease_path.exists()


def test_python_child_liveness_prevents_false_release(pc_fixture, monkeypatch):
    ctx = context(pc_fixture)
    run = ctx.acquire_run(pc_fixture["metadata"]/"runs/child", ctx.identity["registration_sha256"])
    run.record_children([{"pid": 12345, "birth": "fixture-child-birth"}])
    monkeypatch.setattr(runtime, "liveness", lambda *args, **kwargs: "alive")
    with pytest.raises(PCGateError, match="Child exit unproven"):
        run.release({"status": "COMPLETE", "exit_code": 0, "artifact_hashes": {}})
    assert read_json(run.lease_path)["state"] == "AMBIGUOUS_CHILD"
    assert not (run.out/"terminal.json").exists()


def test_python_run_identity_tamper_refuses_boundary(pc_fixture):
    ctx = context(pc_fixture)
    run = ctx.acquire_run(pc_fixture["metadata"]/"runs/identity", ctx.identity["registration_sha256"])
    run.identity["simulator_identity_sha256"] = "b"*64
    with pytest.raises(PCGateError, match="identity|Identity"):
        run.assert_current()
    terminal = run.release({"status": "COMPLETE", "exit_code": 0, "outputs_complete": True,
                            "artifact_hashes": {}})
    assert terminal["status"] == "ERROR" and terminal["outputs_complete"] is False
    assert terminal["identity"] == ctx._captured_identity


def test_producer_proof_binds_python_identity_and_artifact_bytes(pc_fixture):
    path = pc_fixture["root"]/"outputs/phase8/mock_saf/weights.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"neutral fixture weights; no numerical model")
    ctx, folder, terminal = complete_stage(pc_fixture, "e", artifact_paths=[path])
    proof = python_pc.verify_stage_producer(pc_fixture["root"], folder,
        expected_simulator=ctx.simulator_identity_sha256)
    assert proof == terminal
    with pytest.raises(ValueError, match="another Python simulator"):
        python_pc.verify_stage_producer(pc_fixture["root"], folder, expected_simulator="b"*64)
    path.write_bytes(b"weights changed after authenticated release")
    with pytest.raises(ValueError, match="artifact drift"):
        python_pc.verify_stage_producer(pc_fixture["root"], folder)


def test_product_history_cannot_replace_authenticated_source_hashes(pc_fixture):
    output = pc_fixture["root"]/"outputs/phase8/mock_saf"
    output.mkdir(parents=True)
    artifact = output/"product.json"
    artifact.write_text("{}")
    proofs = {}
    for letter in "defh":
        ctx, folder, _ = complete_stage(pc_fixture, letter, artifact_paths=[artifact] if letter == "h" else [])
        proofs[letter] = {"path": folder.relative_to(pc_fixture["root"]).as_posix(),
                          "terminal_sha256": sha256_file(folder/"terminal.json")}
    history = {"schema": "pc-python-stage-history-v1", "simulator": ctx.simulator_identity,
        "simulator_identity_sha256": ctx.simulator_identity_sha256,
        "source_hashes": pc_fixture["sources"], "stages": proofs}
    history_path = output/"pc_pipeline_provenance.json"
    put_json(history_path, history)
    verified = python_pc.validate_product_provenance(pc_fixture["root"], output,
        expected_simulator_identity_sha256=ctx.simulator_identity_sha256)
    assert verified["status"] == "COMPLETE" and verified["core_sha256"] is None
    history["source_hashes"] = {}
    put_json(history_path, history)
    with pytest.raises(ValueError, match="authenticated stage identity"):
        python_pc.validate_product_provenance(pc_fixture["root"], output)


def workflow_fixture(fixture):
    workflow = python_pc.ScientificWorkflow(fixture["root"], fixture["metadata"],
                                           backend="mlx", device="auto")
    workflow.sources, workflow.parity = fixture["sources"], fixture["parity"]
    workflow.saf.mkdir(parents=True)
    return workflow


def mocked_training(members, calls):
    def train(output, registration, registration_sha, run, *, backend, device):
        run.assert_current()
        calls.append((backend, device, registration_sha))
        for member in members:
            relative = f"models/{member['arm']}/N{member['N']}/seed{member['seed']}.npz"
            path = output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(member, sort_keys=True).encode())
        log = output/"training/tiny_mock.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text('{"fixture":"no actual training"}\n')
        put_json(output/"validation.json", {"members": members})
        put_json(output/"selection.json", {"Mdata": 64, "Mphys": 64})
        (output/"progress.jsonl").write_text('{"fixture":"completed mocked fits"}\n')
    return train


def registered_members():
    return [{"arm": arm, "N": size, "seed": seed} for arm in ("Mdata", "Mphys")
            for size in (64, 256, 1024, 4096) for seed in (42, 43, 44)]


def test_stage_e_seals_exact24_fits_and_every_consumed_training_artifact(pc_fixture, monkeypatch):
    from scripts.phase8.saf_surrogate import train
    workflow, calls = workflow_fixture(pc_fixture), []
    monkeypatch.setattr(train, "train_all", mocked_training(registered_members(), calls))
    result = workflow.e()
    assert result["fit_count"] == 24 and result["status"] == "COMPLETE"
    assert calls[0][:2] == ("mlx", "auto")
    folder = workflow.metadata/"runs/e"
    proof = python_pc.verify_stage_producer(workflow.root, folder)
    expected = {p.relative_to(workflow.root).as_posix() for p in workflow.saf.rglob("*") if p.is_file()}
    assert set(proof["artifact_hashes"]) == expected
    assert expected <= set(result["artifacts"])
    assert sum(name.endswith(".npz") for name in expected) == 24
    selection = workflow.saf/"selection.json"
    selection.write_text('{"Mphys":4096}')
    with pytest.raises(ValueError, match="artifact drift"):
        python_pc.verify_stage_producer(workflow.root, folder,
            selected=[selection.relative_to(workflow.root).as_posix()])


@pytest.mark.parametrize("bad_members", [registered_members()[:-1], registered_members()[:-1]+[registered_members()[0]]])
def test_stage_e_refuses_missing_or_duplicate_fits_and_retains_partial_files(pc_fixture, monkeypatch, bad_members):
    from scripts.phase8.saf_surrogate import train
    workflow = workflow_fixture(pc_fixture)
    monkeypatch.setattr(train, "train_all", mocked_training(bad_members, []))
    with pytest.raises(RuntimeError, match="fixed24 fits"):
        workflow.e()
    terminal = read_json(workflow.metadata/"runs/e/terminal.json")
    assert terminal["status"] == "ERROR" and terminal["outputs_complete"] is False
    assert (workflow.saf/"validation.json").exists()
    assert not (workflow.metadata/"stage_owner.json").exists()
