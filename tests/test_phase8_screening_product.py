"""Synthetic API checks: no trained model, C++ core or held-out data."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scripts.phase8 import screening_product as api


def prediction(query, scale=1.0):
    value = scale * (1 + int(query["draw_id"][-2:]) / 100)
    return {**query, "ff_kg_s": value, "T4_K": value*1000,
            "status": "predicted", "diagnostic_unsafe": False,
            "EI_CO2_kg_kg": 3.0, "CO2_g_s": value*3000,
            "lifecycle_g_s": value*100, "nvpm_dEI_number_pct": None,
            "nvpm_status": "outside_Brem_domain", "seed_sd_ff_kg_s": .01,
            "seed_sd_T4_K": 1.0}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    envelope = {"fractions": {key: [0, 1] for key in api.FRACTIONS},
                "thrust_fraction": [.07, 1], "draw_ids": list(api.DRAW_IDS)}
    path = tmp_path / "product.json"
    path.write_text(json.dumps({"training_envelope": envelope, "selected_training_envelope": envelope}))
    calls = []
    class Product:
        def predict(self, queries):
            calls.extend(queries)
            return [prediction(q) for q in queries]
    monkeypatch.setattr(api, "_load_product", lambda model: Product())
    return path, calls


def test_all64_draws_and_sensitivity_bands_are_not_seed_bands(bundle):
    path, calls = bundle
    result = api.screen_blends([{"JetA": .5, "HEFA": .5}], model=path)
    assert [q["draw_id"] for q in calls] == list(api.DRAW_IDS)
    row = result["candidates"][0]
    ff = row["predictions"]["ff_kg_s"]
    assert ff["available_draws"] == 64
    assert ff["mean"] == pytest.approx(1.315)
    assert ff["min"] == 1
    assert ff["max"] == 1.63
    assert ff["q025"] == pytest.approx(1.01575)
    assert row["predictions"]["seed_sd_ff_kg_s"]["mean"] == pytest.approx(.01)
    assert "not empirical confidence" in result["bands"]


def test_invalid_candidate_is_flagged_without_physical_query(bundle):
    path, calls = bundle
    result = api.screen_blends([{"id": "negative", "JetA": 1.1, "HEFA": -.1},
                               {"id": "idle", "JetA": 1, "thrust_fraction": .05}], model=path)
    assert not calls
    assert all(r["status"] == "INVALID_INPUT" and r["predictions"] is None for r in result["candidates"])


def test_selected_training_envelope_flags_without_clipping(bundle):
    path, calls = bundle
    data = json.loads(path.read_text())
    data["selected_training_envelope"]["fractions"]["f_JetA"] = [.1, .9]
    path.write_text(json.dumps(data))
    result = api.screen_blends([{"JetA": 1}], model=path)
    row = result["candidates"][0]
    assert row["outside_training_envelope"]
    assert any(flag.startswith("selected_N:") for flag in row["flags"])
    assert all(q["f_JetA"] == 1 for q in calls)


def test_deterministic_simplex_grid_exact_coverage_and_no_duplicates():
    rows = api.simplex_grid(.5)
    assert len(rows) == math.comb(5, 3)
    assert rows == api.simplex_grid(.5)
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(sum(r[f] for f in api.FRACTIONS) == 1 for r in rows)
    with pytest.raises(ValueError, match="divide"):
        api.simplex_grid(.3)


def test_missing_or_duplicate_draw_denies_summary():
    rows = [prediction({"draw_id": d}) for d in api.DRAW_IDS]
    with pytest.raises(ValueError, match="Exactly"):
        api.summarize_draws(rows[:-1])
    rows[-1]["draw_id"] = rows[0]["draw_id"]
    with pytest.raises(ValueError, match="IDs"):
        api.summarize_draws(rows)


def test_nonfinite_predictions_are_not_successful_bands():
    rows = [prediction({"draw_id": d}) for d in api.DRAW_IDS]
    rows[20]["ff_kg_s"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        api.summarize_draws(rows)


def test_top_k_locked_from_predictions_before_teacher_labels(bundle, monkeypatch, tmp_path):
    path, calls = bundle
    events = []
    def verifier(selected, queries, predictions, model, model_bytes, output):
        events.append(([r["id"] for r in selected], len(predictions), len(calls)))
        return {"selected_ids": [r["id"] for r in selected]}
    monkeypatch.setattr(api, "_verify", verifier)
    result = api.screen_blends([{"id": "z", "JetA": 1}, {"id": "a", "HEFA": 1}], model=path,
                              verify_top_k=1, verification_out=tmp_path / "fresh")
    assert result["ranking"] == ["a", "z"]
    assert events == [(["a"], 128, 128)]


def test_default_loader_requires_deployment_receipt(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    module = ModuleType("scripts.phase8.saf_surrogate.models")
    def load(model, *, require_deployment):
        assert require_deployment is True
        raise RuntimeError("Failed deployment receipt")
    module.load_product = load
    monkeypatch.setitem(sys.modules, "scripts.phase8.saf_surrogate.models", module)
    with pytest.raises(RuntimeError, match="receipt"):
        api._load_product(tmp_path / "product.json")


def test_duplicate_ids_are_refused_before_any_prediction(bundle):
    path, calls = bundle
    with pytest.raises(ValueError, match="unique"):
        api.screen_blends([{"id": "same", "JetA": 1}, {"id": "same", "HEFA": 1}], model=path)
    assert not calls


@pytest.mark.parametrize("fault", ["invalid_thermo", "diagnostic", "missing_ff", "nonfinite"])
def test_invalid_model_group_is_not_ranked_or_selected_for_verification(bundle, monkeypatch, tmp_path, fault):
    path, calls = bundle
    class Product:
        def predict(self, queries):
            calls.extend(queries)
            rows = [prediction(query) for query in queries]
            bad = next(row for row in rows if row["candidate_id"] == "bad" and row["draw_id"] == "draw_17")
            if fault == "invalid_thermo":
                bad["status"] = "invalid_thermo"
            elif fault == "diagnostic":
                bad["diagnostic_unsafe"] = True
            elif fault == "missing_ff":
                bad["ff_kg_s"] = None
            else:
                bad["T4_K"] = float("inf")
            return rows
    monkeypatch.setattr(api, "_load_product", lambda model: Product())
    verified = []
    def verify(selected, *args):
        verified.extend(row["id"] for row in selected)
        return {"selected_ids": list(verified)}
    monkeypatch.setattr(api, "_verify", verify)
    result = api.screen_blends([{"id": "bad", "JetA": 1}, {"id": "good", "HEFA": 1}],
        model=path, verify_top_k=1, verification_out=tmp_path / "fresh")
    bad, good = result["candidates"]
    assert bad["status"] == "MODEL_INVALID" and bad["predictions"] is None and bad["flags"]
    assert good["status"] == "PREDICTED" and good["predictions"]["ff_kg_s"]["available_draws"] == 64
    assert result["ranking"] == ["good"] and verified == ["good"]


@pytest.fixture
def owned_verification(bundle, monkeypatch, tmp_path):
    """Real record/lease code with a fake synchronous verifier, never a core import."""
    import os
    import sys
    from types import ModuleType, SimpleNamespace
    from scripts.phase8 import scientific_workflow_gate as gate
    path, calls = bundle
    root, events, behaviour = tmp_path, [], {"fault": None}
    monkeypatch.setattr(api, "ROOT", root)
    binary = root / "cpp/build/mock_core.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"mock module bytes; never imported")
    lease_path = root / "outputs/operations/owner.json"
    ac = SimpleNamespace(DEAD="dead", process_birth=lambda pid: "owner birth",
                         liveness=lambda pid, birth: "alive" if pid == os.getpid() else "dead")
    monkeypatch.setattr(gate, "_ac", lambda root: ac)
    class Context:
        _owned = None
        op = {"paths": {"owner_lease": "outputs/operations/owner.json"}}
        acquire_run = gate.Context.acquire_run
        def require_idle_ac(self):
            events.append("guard")
            if lease_path.exists():
                lease = gate.read(lease_path)
                if lease.get("reservation_sha256") != self._owned or lease.get("owner_pid") != os.getpid():
                    raise gate.GateError("foreign mock lease")
    context = Context()
    context.root, context.binary_path = root, binary
    context.binary_sha256 = gate.digest(binary)
    context.identity = {"registration_sha256": "registration", "core": {
        "path": str(binary.relative_to(root)), "sha256": context.binary_sha256}}
    def prepare(actual_root, registration, **kwargs):
        assert actual_root == root and registration == api.REGISTRATION
        events.append("prepare")
        return context
    monkeypatch.setattr(gate, "prepare_context", prepare)
    package = ModuleType("scripts.phase8.saf_surrogate")
    package.__path__ = []
    teacher = ModuleType("scripts.phase8.saf_surrogate.teacher")
    out = root / "outputs/verification"
    class Verifier:
        def __call__(self, queries):
            events.append("teacher")
            assert [query["draw_id"] for query in queries] == list(api.DRAW_IDS)
            assert gate.read(out / "selected.json")["ids"] == ["chosen"]
            if behaviour["fault"] == "exception":
                raise RuntimeError("mock teacher interrupted")
            actual = [prediction(query, .99) | {"status": "converged"} for query in queries]
            if behaviour["fault"] == "incomplete":
                actual.pop()
            elif behaviour["fault"] == "unreachable":
                actual[17].update(status="unreachable", ff_kg_s=None)
            elif behaviour["fault"] == "misordered":
                actual[0]["query_id"], actual[1]["query_id"] = actual[1]["query_id"], actual[0]["query_id"]
            return actual
    def load(actual_context, bundle_path, *, assert_current):
        assert actual_context is context and bundle_path == path
        assert_current()
        events.append("load")
        assert (out / "command_spec.json").exists() and (out / "selected.json").exists()
        module = ModuleType("catjet_core")
        if behaviour["fault"] == "foreign_core":
            foreign = root / "cpp/foreign_core.so"
            foreign.write_bytes(b"foreign bytes; never imported")
            module.__file__ = str(foreign)
        else:
            module.__file__ = str(binary)
        monkeypatch.setitem(sys.modules, "catjet_core", module)
        return Verifier()
    teacher.load_verifier = load
    monkeypatch.setitem(sys.modules, "scripts.phase8.saf_surrogate", package)
    monkeypatch.setitem(sys.modules, "scripts.phase8.saf_surrogate.teacher", teacher)
    return path, context, out, lease_path, behaviour, events, gate


def _verify_one(path, out):
    return api.screen_blends([{"id": "chosen", "JetA": 1}], model=path,
                             verify_top_k=1, verification_out=out)


def test_verification_binds_completed_owner_core_and_numeric_differences(owned_verification):
    import os
    path, context, out, lease_path, _, events, gate = owned_verification
    result = _verify_one(path, out)
    reservation = gate.read(out / "reservation.json")
    spec = gate.read(out / "command_spec.json")
    proof = gate.read(out / "core_proof.json")
    command = gate.read(out / "command.exit.json")
    terminal = gate.read(out / "terminal.json")
    summary = gate.read(out / "verification.json")
    assert events.index("prepare") < events.index("load") < events.index("teacher")
    for record in (spec, proof, command):
        assert record["identity"] == context.identity
        assert record["owner_pid"] == reservation["owner_pid"] == os.getpid()
        assert record["owner_birth"] == reservation["owner_birth"]
    assert spec["argv"] == command["argv"] == reservation["argv"]
    assert proof["path"] == str(context.binary_path.resolve()) and proof["sha256"] == context.binary_sha256
    assert command["in_process_completed"] is True and command["exit_code"] == 0
    assert command["status"] == terminal["status"] == "COMPLETE"
    assert command["waited_child_exit"] == "not_applicable_synchronous_owner"
    assert result["verification"]["simulator_requests"] == 64
    assert summary["actual_bands"]["chosen"]["ff_kg_s"]["available_draws"] == 64
    assert summary["differences"]["ff_kg_s"]["max_abs"] == pytest.approx(.0163)
    assert summary["differences"]["T4_K"]["max_abs"] == pytest.approx(16.3)
    assert summary["differences"]["lifecycle_g_s"]["max_abs"] == pytest.approx(1.63)
    assert summary["global_overlap"].startswith("unavailable")
    required = {str((out / name).relative_to(context.root)) for name in (
        "command_spec.json", "core_proof.json", "command.exit.json", "selected.json", "raw_verification.json", "verification.json")}
    assert set(terminal["expected_outputs"]) == required <= set(terminal["artifact_hashes"])
    assert terminal["outputs_complete"] is True and terminal["errors"] == []
    assert not lease_path.exists()


@pytest.mark.parametrize("fault", ["exception", "incomplete", "unreachable", "misordered", "foreign_core"])
def test_verification_failure_retains_truthful_raw_error_provenance(owned_verification, fault):
    path, context, out, lease_path, behaviour, events, gate = owned_verification
    behaviour["fault"] = fault
    with pytest.raises(RuntimeError, match="verification failed"):
        _verify_one(path, out)
    terminal = gate.read(out / "terminal.json")
    command = gate.read(out / "command.exit.json")
    reservation = gate.read(out / "reservation.json")
    assert terminal["status"] == command["status"] == "ERROR"
    assert terminal["exit_code"] != 0 and terminal["outputs_complete"] is False
    assert command["exit_code"] != 0 and command["in_process_completed"] is True and command["errors"]
    assert command["owner_pid"] == reservation["owner_pid"] and command["owner_birth"] == reservation["owner_birth"]
    assert command["identity"] == context.identity and command["argv"] == reservation["argv"]
    assert (out / "selected.json").exists() and not (out / "verification.json").exists()
    if fault in {"incomplete", "unreachable", "misordered"}:
        raw = gate.read(out / "raw_verification.json")
        assert raw["identity"] == context.identity and raw["actual"]
        raw_name = str((out / "raw_verification.json").relative_to(context.root))
        assert terminal["artifact_hashes"][raw_name] == gate.digest(out / "raw_verification.json")
    else:
        assert not (out / "raw_verification.json").exists()
    if fault == "foreign_core":
        assert "teacher" not in events and not (out / "core_proof.json").exists()
    assert not lease_path.exists()
