"""Portable SAF orchestration/proof tests; no scientific data or study runs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.phase8 import pc_saf
from scripts.phase8.saf_surrogate import run as saf_run


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def producer(tmp_path, *, status="COMPLETE"):
    root = tmp_path
    out = root/"outputs/toy"
    reg = root/"docs/toy_registration.json"
    dump(reg, {"outputs": {"directory": "outputs/toy"}})
    (root/"core.so").write_bytes(b"manufactured core bytes")
    (root/"source.py").write_text("manufactured source\n")
    core_hash = pc_saf.sha256_file(root/"core.so")
    dump(root/"outputs/g0.json", {"verdict": "PASS", "core": {"path": "core.so", "sha256": core_hash}})
    identity = {"schema": "pc-local-identity-v1", "registration_sha256": pc_saf.sha256_file(reg),
                "binary_path": "core.so", "binary_sha256": core_hash,
                "source_hashes": {"source.py": pc_saf.sha256_file(root/"source.py")},
                "g0_record_path": "outputs/g0.json", "g0_record_sha256": pc_saf.sha256_file(root/"outputs/g0.json")}
    reservation = {"identity": identity, "owner_pid": 123, "owner_birth": "manufactured birth"}
    dump(out/"reservation.json", reservation)
    code = 1 if status == "FAIL" else 0
    dump(out/"command.exit.json", {"identity": identity, "in_process_completed": True, "exit_code": code})
    dump(out/"metadata.json", {"opaque metadata": True})
    (out/"sealed.csv").write_text("manufactured locked target\n")
    names = ["outputs/toy/metadata.json", "outputs/toy/sealed.csv", "outputs/toy/command.exit.json"]
    terminal = {"identity": identity, "status": status, "exit_code": code, "outputs_complete": True,
                "execution_complete": True, "scientific_verdict": "FAIL" if code else "PASS",
                "reservation_sha256": pc_saf.sha256_file(out/"reservation.json"), "errors": [],
                "expected_outputs": names,
                "artifact_hashes": {name: pc_saf.sha256_file(root/name) for name in names}}
    dump(out/"terminal.json", terminal)
    dump(out/"released_lease.json", {**reservation, "state": "RELEASED",
        "reservation_sha256": terminal["reservation_sha256"], "terminal_sha256": pc_saf.sha256_file(out/"terminal.json")})
    return root, reg, out, core_hash


def test_local_prerequisite_projection_does_not_open_or_hash_locked_targets(tmp_path, monkeypatch):
    root, reg, out, core_hash = producer(tmp_path)
    original = pc_saf.sha256_file
    def hash_allowed(path):
        if Path(path) == out/"sealed.csv":
            pytest.fail("prerequisite metadata projection consumed locked target bytes")
        return original(path)
    monkeypatch.setattr(pc_saf, "sha256_file", hash_allowed)
    result = pc_saf.validate_local_terminal(root, reg, out, expected_binary_sha256=core_hash,
        artifact_paths=["outputs/toy/metadata.json", "outputs/toy/command.exit.json"])
    assert result["status"] == "COMPLETE" and result["core_sha256"] == core_hash


@pytest.mark.parametrize("name", ["source.py", "core.so", "outputs/g0.json", "outputs/toy/metadata.json",
                                  "outputs/toy/reservation.json", "outputs/toy/terminal.json"])
def test_local_producer_rejects_changed_source_core_g0_artifacts_and_proofs(tmp_path, name):
    root, reg, out, core_hash = producer(tmp_path)
    path = root/name
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        pc_saf.validate_local_terminal(root, reg, out, expected_binary_sha256=core_hash)


def test_completed_scientific_fail_is_explicit_and_default_loader_evidence_refuses_it(tmp_path):
    root, reg, out, core_hash = producer(tmp_path, status="FAIL")
    with pytest.raises(ValueError, match="complete local execution"):
        pc_saf.validate_local_terminal(root, reg, out)
    result = pc_saf.validate_local_terminal(root, reg, out, expected_binary_sha256=core_hash,
        allow_scientific_fail=True)
    assert result["status"] == "FAIL"


def test_public_product_dispatches_to_pc_proof_and_refuses_failed_deployment(tmp_path, monkeypatch):
    from scripts.phase8.saf_surrogate import models
    root = tmp_path
    reg = root/"docs/phase8_saf_surrogate_registration.json"
    dump(reg, {"manufactured registration": True})
    out = root/"outputs/toy"
    bundle = {"registration_id": "P8-S-20261004", "registration_sha256": pc_saf.sha256_file(reg),
              "model": {"seeds": [42, 43, 44], "output_dimension": 494},
              "artifacts": {}, "scientific_sources": {}, "execution_profile": "pc", "binary_sha256": "a"*64}
    dump(out/"product.json", bundle)
    receipt = {"state": "PASS", "product_sha256": pc_saf.sha256_file(out/"product.json"),
               "registration_sha256": bundle["registration_sha256"], "evidence_sha256": {},
               "gates": {name: "PASS" for name in ("fidelity", "ranking", "precision", "provenance", "operational")}}
    dump(out/"deployment_receipt.json", receipt)
    monkeypatch.setattr(models, "__file__", str(root/"scripts/phase8/saf_surrogate/models.py"))
    class CalledLocalProof(Exception): pass
    calls = []
    def validate(*args, **kwargs):
        calls.append((args, kwargs)); raise CalledLocalProof()
    monkeypatch.setattr(pc_saf, "validate_local_terminal", validate)
    with pytest.raises(CalledLocalProof):
        models.load_product(out/"product.json")
    assert calls[0][0] == (root, "docs/phase8_saf_surrogate_registration.json", out)
    assert calls[0][1]["expected_binary_sha256"] == "a"*64
    receipt["state"] = "FAIL"; dump(out/"deployment_receipt.json", receipt)
    with pytest.raises(ValueError, match="successful deployment receipt"):
        models.load_product(out/"product.json")
    assert len(calls) == 1


def test_pc_light_training_check_uses_local_owned_context_not_mac(monkeypatch):
    checks = []
    run = SimpleNamespace(context=SimpleNamespace(execution_profile="pc"),
                          light_training_check=lambda: checks.append("local"),
                          assert_current=lambda: pytest.fail("full source scan on lightweight epoch"))
    monkeypatch.setattr(saf_run, "require_ac", lambda: pytest.fail("pmset invoked"))
    saf_run.light_training_check(run)
    assert checks == ["local"]


def test_pc_child_authorization_proves_same_lease_sources_core_and_linux_power(tmp_path, monkeypatch):
    from scripts.phase8 import pc_runtime
    root = tmp_path
    (root/"registration.json").write_text("manufactured registration")
    (root/"core.so").write_text("manufactured core")
    owner = {"state": "RUNNING", "owner_pid": saf_run.os.getpid(), "owner_birth": "birth", "children": []}
    dump(root/"lease.json", owner); dump(root/"snapshot.json", owner)
    spec = {"root": str(root), "owner_pid": owner["owner_pid"], "owner_birth": "birth",
        "owner_lease": {"path": "lease.json", "snapshot_path": "snapshot.json",
                        "sha256": pc_saf.sha256_file(root/"snapshot.json")},
        "registration_path": "registration.json", "registration_sha256": pc_saf.sha256_file(root/"registration.json"),
        "source_hashes": {}, "input_source_hashes": {},
        "binary": {"path": "core.so", "sha256": pc_saf.sha256_file(root/"core.so")}, "execution_profile": "pc"}
    monkeypatch.setattr(saf_run, "birth", lambda pid: "birth")
    monkeypatch.setattr(saf_run, "require_ac", lambda: pytest.fail("Mac child power check invoked"))
    calls = []
    monkeypatch.setattr(pc_runtime, "require_idle_ac_linux", lambda: calls.append("linux"))
    saf_run.check_child_authorization(spec)
    assert calls == ["linux"]
    (root/"core.so").write_text("changed core")
    with pytest.raises(RuntimeError, match="core drift"):
        saf_run.check_child_authorization(spec)


def test_pc_full_route_preserves_order_fixed_attempt_and_sole_score(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from scripts.phase8.saf_surrogate import score, train, train_torch, timing, study
    root = tmp_path
    reg = {"id": "P8-S-20261004", "artifact_root": "outputs/phase8/saf_surrogate/attempt_001",
           "relevant_files": {"read_only": []},
           "provenance": {"successful_release": {"expected_outputs": []}}}
    for name in pc_saf.PC_SOURCES:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fixture source\n")
    sources = {name: pc_saf.sha256_file(root/name) for name in pc_saf.PC_SOURCES}
    output = root/reg["artifact_root"]
    events = []
    class Run:
        _released = False
        def assert_current(self): pass
        def release(self, terminal): self._released = True; return terminal
    class Context:
        execution_profile = "pc"
        identity = {"source_hashes": sources}
        binary_sha256 = "a"*64
        original_context = {"g0": {"verdict": "PASS"}}
        def require_idle_ac(self): pass
        def acquire_run(self, out, registration_hash): out.mkdir(parents=True); return Run()
    for name in saf_run.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(pc_saf, "load_registration", lambda *args: (reg, "b"*64))
    monkeypatch.setattr(pc_saf, "verify_named_prerequisite", lambda *a, **k: {"registration_sha256": "c"*64})
    monkeypatch.setattr(train_torch, "resolve_device", lambda value: "cuda" if value == "auto" else value)
    def freeze(*args, **kwargs):
        events.append("freeze")
        assert set(pc_saf.PC_SOURCES).issubset(kwargs["source_hashes"])
        assert set(kwargs["provenance_dependencies"]) == {"local_pc_origin", "named_prerequisite_terminal"}
    monkeypatch.setattr(saf_run, "freeze", freeze)
    monkeypatch.setattr(timing, "source_diagnostics", lambda *a: events.append("source_checks"))
    monkeypatch.setattr(saf_run, "wait_generation", lambda *a: events.append("generate"))
    def train_all(*args, **kwargs):
        events.append("train"); assert kwargs == {"backend": "torch", "device": "cuda"}
    monkeypatch.setattr(train, "train_all", train_all)
    monkeypatch.setattr(score, "seal_predictions", lambda *a: events.append("seal"))
    def score_once(*args):
        assert events[-1] == "seal"
        pc_saf.write_once(output/"score_reservation.json", {"sole": True})
        events.append("score")
        return {"fidelity_pass": False}
    monkeypatch.setattr(score, "score_all", score_once)
    def measure(*args):
        assert args[-1] == "cuda"; events.append("timing"); return {}
    monkeypatch.setattr(timing, "measure", measure)
    monkeypatch.setattr(study, "screen", lambda *a: events.append("study") or {})
    monkeypatch.setattr(timing, "finalize_timing", lambda *a: events.append("finalize") or {"operational_pass": False})
    monkeypatch.setattr(score, "deployment_receipt", lambda *a: pc_saf.write_once(output/"deployment_receipt.json", {"state": "FAIL"}))
    result = pc_saf.execute(root, lambda *a: Context(), device="auto")
    assert events == ["freeze", "source_checks", "generate", "train", "seal", "score", "timing", "study", "finalize"]
    assert result["status"] == "FAIL" and result["execution_complete"] is True
    with pytest.raises(FileExistsError, match="already exists"):
        pc_saf.execute(root, lambda *a: pytest.fail("repeat acquired a new context"))
    assert events.count("score") == 1
