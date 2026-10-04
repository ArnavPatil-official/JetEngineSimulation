"""Synthetic safety/algorithm checks; never import a scientific core or targets.

These tests were written while the main benchmark was active and must not be
executed until the separately authorized validation window.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p73_a1_synthetic", REPO / "scripts/phase8/p73_a1_cpp.py")
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)


def dependency_objects():
    free = ["W_ref", "a_thrust", "k_pi", "k_mdot"]
    fit = {"free": free, "params": {name: 1.0 for name in free}}
    profile = {"A1": "FAIL", "free": free, "identified": [], "not_identified": free,
               "verdicts": {name: {"IDENTIFIED_existing_rule": True, "IDENTIFIED": False,
                                  "penalty_dependent": True, "verdict_points_penalised": [0.0]} for name in free}}
    gate = {"open": False, "reason": "A1 FAIL"}
    return profile, fit, {"A2": {"verdict": "FAIL (no demonstrated skill)"}, "blend_gate": gate}, {"gate": gate}


def test_penalty_guard_only_exception_preserves_closed_gate():
    values = dependency_objects()
    before = copy.deepcopy(values)
    consumer.validate_exception(*values)
    assert values == before
    assert values[2]["blend_gate"]["open"] is False


@pytest.mark.parametrize("mutation", [
    lambda p, f, h, g: p.update(A1="PASS"),
    lambda p, f, h, g: p["verdicts"]["W_ref"].update(IDENTIFIED_existing_rule=False),
    lambda p, f, h, g: p["verdicts"]["W_ref"].update(penalty_dependent=False),
    lambda p, f, h, g: p["verdicts"]["W_ref"].update(IDENTIFIED=True),
    lambda p, f, h, g: p["verdicts"]["W_ref"].update(verdict_points_penalised=[]),
    lambda p, f, h, g: p["verdicts"].pop("W_ref"),
    lambda p, f, h, g: f.update(free=["W_ref"]),
    lambda p, f, h, g: h["A2"].update(verdict="ESCALATE (B0 margin)"),
    lambda p, f, h, g: h.pop("A2"),
    lambda p, f, h, g: g.update(gate={"open": True, "reason": "A1 PASS"}),
])
def test_exception_rejects_wrong_failure_or_escalation(mutation):
    values = copy.deepcopy(dependency_objects())
    mutation(*values)
    with pytest.raises(consumer.Blocked):
        consumer.validate_exception(*values)


def test_registered_parity_tolerance_and_exact_strings():
    a = [{"status": "converged", "reason": "", "ff": 1.0}]
    assert consumer.compare_records(a, [{"status": "converged", "reason": "", "ff": 1.0 + 5e-10}])["match"]
    assert not consumer.compare_records(a, [{"status": "converged", "reason": "", "ff": 1.0 + 2e-9}])["match"]
    assert not consumer.compare_records(a, [{"status": "converged", "reason": "changed", "ff": 1.0}])["match"]
    assert not consumer.compare_records(a, [{"ff": 1.0, "reason": "", "status": "converged"}])["match"]
    assert not consumer.compare_records(a, [])["match"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_converged_rows_cannot_pass_parity(value):
    rows = [{"status": "converged", "reason": "", "ff": value}]
    assert not consumer.compare_records(rows, rows)["match"]


def test_unreachable_missing_values_require_matching_reason_and_pattern():
    a = [{"status": "unreachable", "reason": "target outside bracket", "ff": float("nan")}]
    assert consumer.compare_records(a, copy.deepcopy(a))["match"]
    assert not consumer.compare_records(a, [{"status": "unreachable", "reason": "other", "ff": float("nan")}])["match"]
    assert not consumer.compare_records(a, [{"status": "unreachable", "reason": "target outside bracket", "ff": 1.0}])["match"]


def test_write_once_never_replaces_evidence(tmp_path):
    path = tmp_path / "record.json"
    consumer.write_new(path, "first\n")
    with pytest.raises(FileExistsError):
        consumer.write_new(path, "second\n")
    assert path.read_text() == "first\n"


def test_selected_core_drift_refuses_before_extension_import(tmp_path, monkeypatch):
    path = tmp_path / "core.so"
    path.write_bytes(b"synthetic binary, never imported")
    monkeypatch.setattr(consumer, "load_module", lambda *args: pytest.fail("must refuse before core import"))
    with pytest.raises(consumer.Blocked, match="hash changed"):
        consumer.load_selected_core(path, "wrong")


def test_selected_core_rejects_preloaded_other_module(tmp_path, monkeypatch):
    path = tmp_path / "selected.so"
    path.write_bytes(b"selected")
    monkeypatch.setitem(consumer.sys.modules, "catjet_core", SimpleNamespace(__file__=str(tmp_path / "other.so")))
    with pytest.raises(consumer.Blocked, match="another core"):
        consumer.load_selected_core(path, consumer.sha256(path))


def test_every_cpp_worker_preloads_before_protected_initializer(tmp_path, monkeypatch):
    root = tmp_path / "main"
    workers = tmp_path / "workers"
    workers.mkdir()
    selected = SimpleNamespace()
    events = []
    backend = SimpleNamespace(init_worker_cpp=lambda excluded: events.append(("protected-init", excluded)))
    monkeypatch.setattr(consumer, "import_protocol", lambda root: (None, backend))
    monkeypatch.setattr(consumer, "load_selected_core", lambda path, sha: events.append(("preload", str(path), sha)) or selected)
    adapter = SimpleNamespace(__file__=str(root / "simulation/catjet_backend.py"), load_core=lambda: selected)
    monkeypatch.setattr(consumer.importlib, "import_module", lambda name: adapter)
    monkeypatch.setattr(consumer.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="Sun  Oct  4 12:00:00 2026\n"))
    consumer.init_cpp_worker(root, tmp_path / "selected.so", "frozen-sha", ["heldout-family"], workers, "study_cpp")
    assert events[0][0] == "preload" and events[1] == ("protected-init", ["heldout-family"])
    proof = consumer.read_json(next(workers.glob("*.json")))
    assert proof["binary_sha256"] == "frozen-sha" and proof["birth"] == "Sun Oct 4 12:00:00 2026"
    assert proof["pool"] == "study_cpp"


def test_worker_refuses_adapter_using_a_different_core(tmp_path, monkeypatch):
    root = tmp_path / "main"
    workers = tmp_path / "workers"
    workers.mkdir()
    monkeypatch.setattr(consumer, "import_protocol", lambda root: (None, SimpleNamespace(init_worker_cpp=lambda excluded: None)))
    monkeypatch.setattr(consumer, "load_selected_core", lambda *args: object())
    adapter = SimpleNamespace(__file__=str(root / "simulation/catjet_backend.py"), load_core=lambda: object())
    monkeypatch.setattr(consumer.importlib, "import_module", lambda name: adapter)
    with pytest.raises(consumer.Blocked, match="did not use the selected"):
        consumer.init_cpp_worker(root, tmp_path / "selected.so", "sha", [], workers, "study_cpp")
    assert not list(workers.iterdir())


class FakeRun:
    def __init__(self, context, out):
        self.context, self.out, self.calls = context, out, 0
        self.children = []

    def assert_current(self):
        self.calls += 1
        if self.context.fail_assert_at == self.calls:
            raise consumer.Blocked("identity or ownership drift")

    def release(self, terminal):
        if self.context.release_drift:
            terminal = {**terminal, "status": "ERROR", "exit_code": 1,
                        "errors": [*terminal["errors"], "shared gate end drift"]}
        consumer.write_new(self.out / "terminal.json", consumer.json_text(terminal))
        self.context.released = terminal
        return terminal

    def record_children(self, children):
        self.children = list(children)


class FakeContext:
    def __init__(self):
        self.identity = {"original": "attested-original", "consumer": "new-distinct", "core": "synthetic"}
        self.original_context = {"identity": {"original": "attested-original"}}
        self.binary_path = Path("synthetic-not-imported.so")
        self.binary_sha256 = "synthetic-core"
        self.fail_assert_at, self.release_drift, self.idle = None, False, 0
        self.acquired, self.released = None, None

    def require_idle_ac(self):
        self.idle += 1

    def acquire_run(self, out, registration_sha256, identity=None):
        assert identity == self.identity
        out.mkdir(parents=True)
        consumer.write_new(out / "reservation.json", consumer.json_text({"registration_sha256": registration_sha256,
            "owner_pid": consumer.os.getpid(), "owner_birth": "synthetic owner birth", "argv": ["synthetic consumer"]}))
        self.acquired = FakeRun(self, out)
        return self.acquired


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    root = tmp_path / "main"
    root.mkdir()
    contract = {"registration_id": "P7.3-A1", "implementation_contract": {
        "fit_path": "fit.json", "profile_path": "profile.json", "gate_path": "hold.json",
        "historical_closed_gate_path": "closed.json"}, "outputs": {"directory": "outputs/p73"}}
    registration = root / "registration.json"
    registration.write_text(json.dumps(contract))
    profile, fit, hold, gate = dependency_objects()
    for name, value in (("fit", fit), ("profile", profile), ("hold", hold), ("closed", gate)):
        (root / f"{name}.json").write_text(json.dumps(value))
    identity = {"files": {"synthetic": {"sha256": "fixture", "mode": "100644"}},
                "registration_sha256": consumer.sha256(registration)}
    monkeypatch.setattr(consumer, "consumer_identity", lambda *args: identity)
    monkeypatch.setattr(consumer, "validate_contract", lambda *args: None)
    monkeypatch.setattr(consumer, "environment", lambda context, selected: {
        "identity": context.identity, "consumer_identity": selected,
        "original_main_context": context.original_context, "conditional_label": consumer.LABEL})
    context, prepared = FakeContext(), []

    def factory(main_root, registration_path, **kwargs):
        prepared.append((main_root, registration_path, kwargs))
        return context

    return root, registration, context, factory, prepared


def fake_quantitative_runner(root, registration, fit, context, run, out):
    for rel in consumer.EXPECTED_OUTPUTS:
        if rel in {"command.log", "command.exit.json", "environment.json", "partial_rows.jsonl"}:
            continue
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "parity/parity.json":
            value = {"status": "PASS", "checks": {name: {"match": True} for name in ("central", "draw_00", "draw_31", "draw_63")},
                     "binary_path": str(context.binary_path), "binary_sha256": context.binary_sha256}
        elif rel == "p73_blends_v6.json":
            value = {"n_draws": 64, "n_comparisons": 720, "n_claimed": 0,
                     "conditional_label": consumer.LABEL, "original_blend_gate_open": False}
        else:
            value = None
        consumer.write_new(path, consumer.json_text(value) if value else "synthetic quantitative fixture\n")
    for index, pool in enumerate(("parity_python", "parity_cpp", "study_cpp")):
        consumer.write_new(out / "workers" / f"{pool}-123.json", consumer.json_text({
            "pid": 123 + index, "birth": "synthetic birth", "pool": pool, "argv": ["synthetic worker"],
            "backend": "python-v6" if pool == "parity_python" else "cpp-full-equilibrium-v6",
            "binary_path": None if pool == "parity_python" else str(context.binary_path.resolve()),
            "binary_sha256": None if pool == "parity_python" else context.binary_sha256}))
    consumer.sync_worker_children(run, out)
    result = {"n_draws": 64, "n_comparisons": 720, "n_claimed": 0}
    result["_sealed_outputs"] = {name: consumer.sha256(out / name)
                                  for name in (*consumer.TABLE_NAMES, "p73_blends_v6.json", "README.md")}
    result["_sealed_schemas"] = {name: {"columns": ["synthetic quantitative fixture"], "rows": 0}
                                  for name in consumer.TABLE_NAMES}
    return result


def test_agreed_gate_separates_identities_and_requires_fresh_g0(orchestration):
    root, registration, context, factory, prepared = orchestration
    terminal = consumer.execute(root, registration, gate_factory=factory, scientific_runner=fake_quantitative_runner)
    assert terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
    assert prepared[0][2]["require_g0"] is True
    assert prepared[0][1] == consumer.REGISTRATION
    assert prepared[0][2]["expected_consumer_identity"] != context.original_context["identity"]
    assert context.idle == 1 and context.acquired.calls >= 3
    assert terminal["conditional_label"] == consumer.LABEL
    assert terminal["artifacts_sha256"]["artifact_hashes.json"]
    assert terminal["outputs_complete"] is True
    assert set(terminal["expected_outputs"]) <= set(terminal["artifact_hashes"])
    assert all(name.startswith("outputs/p73/") for name in terminal["artifact_hashes"])
    assert (root / "outputs/p73/reservation.json").exists()
    assert context.released == terminal
    exit_record = consumer.read_json(root / "outputs/p73/command.exit.json")
    assert exit_record["identity"] == context.identity and exit_record["in_process_completed"] is True
    assert exit_record["owner_pid"] == consumer.os.getpid() and exit_record["argv"] == ["synthetic consumer"]
    assert len(context.acquired.children) == 3


def test_ac_or_active_owner_gate_blocks_before_reservation(orchestration, monkeypatch):
    root, registration, context, factory, _ = orchestration
    def refuse():
        raise consumer.Blocked("AC absent or active lease")
    monkeypatch.setattr(context, "require_idle_ac", refuse)
    with pytest.raises(consumer.Blocked, match="AC absent"):
        consumer.execute(root, registration, gate_factory=factory,
                         scientific_runner=lambda *args: pytest.fail("must not run"))
    assert not (root / "outputs").exists()


def test_missing_required_output_is_error_with_nonzero_exit(orchestration):
    root, registration, context, factory, _ = orchestration

    def partial(*args):
        value = fake_quantitative_runner(*args)
        (args[-1] / "p73_blends_v6_claims.csv").unlink()
        return value

    terminal = consumer.execute(root, registration, gate_factory=factory, scientific_runner=partial)
    assert terminal["status"] == "ERROR" and terminal["exit_code"] != 0
    assert any("missing expected outputs" in error for error in terminal["errors"])
    assert consumer.read_json(root / "outputs/p73/command.exit.json")["exit_code"] != 0
    assert (root / "outputs/p73/reservation.json").exists()


def test_missing_worker_binary_proof_cannot_be_complete(orchestration):
    root, registration, context, factory, _ = orchestration

    def incomplete(*args):
        value = fake_quantitative_runner(*args)
        (args[-1] / "workers/study_cpp-123.json").unlink()
        return value

    terminal = consumer.execute(root, registration, gate_factory=factory, scientific_runner=incomplete)
    assert terminal["status"] == "ERROR" and terminal["exit_code"] != 0
    assert any("worker provenance" in error for error in terminal["errors"])


def test_fresh_quantitative_file_drift_cannot_be_complete(orchestration):
    root, registration, context, factory, _ = orchestration

    def altered(*args):
        result = fake_quantitative_runner(*args)
        (args[-1] / "p73_blends_v6_draws.csv").write_text("changed after sealed write\n")
        return result

    terminal = consumer.execute(root, registration, gate_factory=factory, scientific_runner=altered)
    assert terminal["status"] == "ERROR" and terminal["exit_code"] != 0
    assert any("quantitative output drift" in error for error in terminal["errors"])


def test_parity_failure_prevents_full_stage_and_retains_truthful_failure(orchestration):
    root, registration, context, factory, _ = orchestration

    def failed(*args):
        raise consumer.ParityFailure("synthetic mismatch")

    terminal = consumer.execute(root, registration, gate_factory=factory, scientific_runner=failed)
    assert terminal["status"] == "FAIL" and terminal["exit_code"] != 0
    assert not (root / "outputs/p73/p73_blends_v6_claims.csv").exists()
    assert (root / "outputs/p73/command.log").exists()


def test_prelaunch_drift_prevents_scientific_runner(orchestration):
    root, registration, context, factory, _ = orchestration
    context.fail_assert_at = 1
    terminal = consumer.execute(root, registration, gate_factory=factory,
                                scientific_runner=lambda *args: pytest.fail("drift must block science"))
    assert terminal["status"] == "ERROR"
    assert (root / "outputs/p73/reservation.json").exists()


def test_postcompute_drift_downgrades_completion(orchestration):
    root, registration, context, factory, _ = orchestration
    context.release_drift = True
    terminal = consumer.execute(root, registration, gate_factory=factory, scientific_runner=fake_quantitative_runner)
    assert terminal["status"] == "ERROR" and terminal["exit_code"] != 0
    assert "shared gate end drift" in terminal["errors"]


def test_existing_output_refuses_second_run_without_overwrite(orchestration):
    root, registration, context, factory, _ = orchestration
    consumer.execute(root, registration, gate_factory=factory, scientific_runner=fake_quantitative_runner)
    before = (root / "outputs/p73/terminal.json").read_bytes()
    with pytest.raises(FileExistsError):
        consumer.execute(root, registration, gate_factory=factory, scientific_runner=fake_quantitative_runner)
    assert (root / "outputs/p73/terminal.json").read_bytes() == before


def synthetic_protocol():
    # Extract only pure registered functions without executing protected imports
    # or loading chemistry/engine/empirical targets. All cycle rows below are toys.
    import numpy as np
    import pandas as pd
    import math
    source = ast.parse((REPO / "scripts/optimization/blend_matched_thrust_v6.py").read_text())
    names = {"claim", "nvpm_brem", "saf_fraction", "corsia_central", "corsia_common_draws"}
    body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"np": np, "math": math, "JETA": "JetA", "JETA_ALT": "JetA_dooley2010",
             "N_DRAWS": 64, "SIGN_AGREEMENT_MIN": 0.95}
    fuel_source = ast.parse((REPO / "simulation/fuels_v7.py").read_text())
    brem_body = [node for node in fuel_source.body if isinstance(node, ast.FunctionDef) and node.name == "nvpm_brem_dEIn_pct"]
    brem_scope = {"math": math, "load_properties": lambda: {"nvpm": {"brem2015_via_teoh2022_S1": {
        "validity": {"F_pct_gt": 30., "dH_pct_lt": .6}, "alpha0": -30., "alpha1": .1}}}}
    exec(compile(ast.Module(body=brem_body, type_ignores=[]), "pure_registered_brem", "exec"), brem_scope)
    fuel_helpers = SimpleNamespace(nvpm_brem_dEIn_pct=brem_scope["nvpm_brem_dEIn_pct"],
                                  mass_blend=lambda parts: parts,
                                  properties=lambda parts: {"lhv_gas_MJ_kg": 44.})
    scope["fuels_v7"] = fuel_helpers
    exec(compile(ast.Module(body=body, type_ignores=[]), "pure_registered_functions", "exec"), scope)
    modes = {"TAKE-OFF": (1.0, "TAKE-OFF", True), "APPROACH": (.3, "APPROACH", True),
             "IDLE": (.07, "IDLE", True), "CLIMB85": (.85, "TAKE-OFF", False)}
    protocol = SimpleNamespace(np=np, pd=pd, N_DRAWS=64, JETA="JetA", JETA_ALT="JetA_dooley2010",
        OPERATING_POINTS=modes, QUANTITIES=("ff", "tsfc_mg_Ns", "T4", "phi", "nox_corr_g_s", "lifecycle_g_s"),
        pairs=lambda: [{"a": "HEFA-10", "b": "JetA", "family": "SAF_vs_JetA"}],
        corsia=lambda: {"base": 89., "tri": {"HEFA": {"min": 20., "mode": 30., "max": 40.}}},
        lifecycle_factor=lambda fuel, draw: sum(weight * draw["fossil" if part.startswith("JetA") else "HEFA"] for part, weight in fuel.items()),
        lhv_liquid=lambda name: 43.,
        fuels_v7=fuel_helpers)
    for name in names:
        setattr(protocol, name, scope[name])
    return protocol


def toy_frames(protocol):
    fuels = {"JetA": {"JetA2012": 1.}, "HEFA-10": {"JetA2012": .9, "SAF": .1}, "JetA_dooley2010": {"JetA2010": 1.}}
    rows = []
    for fuel, factor in (("JetA", 1.), ("HEFA-10", .8), ("JetA_dooley2010", 1.05)):
        for mode in protocol.OPERATING_POINTS:
            rows.append({"fuel": fuel, "op": mode, "status": "converged", "reason": "", "ff": factor,
                         "phi": factor, "T4": 1000. * factor, "tsfc_mg_Ns": 10. * factor, "nox_corr_g_s": factor})
    central = protocol.pd.DataFrame(rows)
    frames = []
    for index in range(64):
        frame = central[central.fuel != protocol.JETA_ALT].copy()
        frame.insert(0, "draw", f"draw_{index:02d}")
        frames.append(frame)
    return fuels, central, protocol.pd.concat(frames, ignore_index=True)


def test_exact_draw_fuel_mode_coverage_refuses_missing_or_duplicate():
    protocol = synthetic_protocol()
    fuels, central, draws = toy_frames(protocol)
    consumer.validate_coverage(protocol, central, draws, fuels)
    with pytest.raises(RuntimeError, match="missing, duplicated or reordered"):
        consumer.validate_coverage(protocol, central, draws.iloc[:-1], fuels)
    doubled = protocol.pd.concat([draws, draws.iloc[-1:]], ignore_index=True)
    with pytest.raises(RuntimeError):
        consumer.validate_coverage(protocol, central, doubled, fuels)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_full_study_cannot_claim_with_nonfinite_converged_draw(value):
    protocol = synthetic_protocol()
    fuels, central, draws = toy_frames(protocol)
    draws.loc[0, "ff"] = value
    with pytest.raises(RuntimeError, match="nonfinite or missing converged"):
        consumer.validate_coverage(protocol, central, draws, fuels)


def test_raw_checkpoint_retains_missing_values_without_invalid_json():
    protocol = synthetic_protocol()
    frame = protocol.pd.DataFrame([{"status": "unreachable", "reason": "bracket", "ff": float("nan")}])
    text = consumer.checkpoint_record("study-cpp", "draw_00", frame, {"source": "frozen"})
    record = json.loads(text)
    assert text.count("\n") == 1
    assert record["rows"][0]["status"] == "unreachable"
    assert record["rows"][0]["ff"] == {"nonfinite_raw_value": "nan"}
    assert record["identity"] == {"source": "frozen"}


def test_unproven_shutdown_is_explicit_ambiguous_child_error(tmp_path):
    (tmp_path / "workers").mkdir()
    children = []
    run = SimpleNamespace(record_children=lambda value: children.extend(value))
    def interrupted_close():
        raise KeyboardInterrupt("synthetic interrupted join")
    with pytest.raises(consumer.ChildLifecycleError, match="preserve owner lease"):
        consumer.close_model(SimpleNamespace(close=interrupted_close), run, tmp_path)


def test_python_parity_worker_reuses_protected_initializer_only(monkeypatch):
    events = []
    protocol = SimpleNamespace(v5=SimpleNamespace(_init_worker=lambda excluded: events.append(("protected-python", excluded))))
    monkeypatch.setattr(consumer, "import_protocol", lambda root: (protocol, None))
    monkeypatch.setattr(consumer, "worker_record", lambda *args: events.append(("proof", args[2])))
    consumer.init_python_worker(Path("synthetic-main"), ["heldout"], Path("workers"), "parity_python")
    assert events == [("protected-python", ["heldout"]), ("proof", "python-v6")]


def test_protected_claim_lifecycle_and_brem_rules_used_with_parent_csv_schema():
    protocol = synthetic_protocol()
    fuels, central, draws = toy_frames(protocol)
    registration = {"unchanged_protocol": {"lifecycle": {"n_corsia_common": 1000, "seed": 42},
                    "nvpm": {"h_ref_pct": 13.8, "h_saf_pct": 15.3}}}
    tables, summary = consumer.postprocess(protocol, registration, {"frozen": 1.}, central, draws, fuels)
    claims, nvpm = tables[2], tables[3]
    assert len(claims) == 24
    assert claims[claims.op == "CLIMB85"].claimed.eq(False).all()
    assert claims[claims.op == "TAKE-OFF"].claimed.eq(True).all()
    assert claims.paired_n.eq(64).all()
    assert claims.spread_S_dooley2012_vs_2010.gt(0).all()
    assert "conditional_label" not in claims.columns
    assert list(claims.columns) == ["family", "a", "b", "op", "quantity", "central_a", "central_b", "delta_central",
        "delta_rel_pct_of_b", "spread_S_dooley2012_vs_2010", "paired_sign_agreement", "paired_n", "paired_delta_p5",
        "paired_delta_p95", "paired_delta_min", "paired_delta_max", "corsia_sign_agreement", "in_domain", "all_converged", "claimed", "rejection_reasons"]
    unavailable = nvpm[nvpm.status == "unavailable"]
    assert unavailable.dEIn_pct.isna().all()
    assert nvpm[(nvpm.fuel == "HEFA-10") & (nvpm.op == "APPROACH")].status.iloc[0] == "unavailable"
    assert summary["conditional_label"] == consumer.LABEL
    assert summary["original_A1"] == "FAIL" and summary["original_blend_gate_open"] is False
    assert summary["v6_params_fixed_for_all_fuels_and_draws"] == {"frozen": 1.}


def test_failed_paired_row_rejects_every_quantity_at_that_mode():
    protocol = synthetic_protocol()
    fuels, central, draws = toy_frames(protocol)
    mask = (draws.draw == "draw_63") & (draws.fuel == "HEFA-10") & (draws.op == "TAKE-OFF")
    draws.loc[mask, "status"] = "unreachable"
    registration = {"unchanged_protocol": {"lifecycle": {"n_corsia_common": 1000, "seed": 42},
                    "nvpm": {"h_ref_pct": 13.8, "h_saf_pct": 15.3}}}
    tables, _ = consumer.postprocess(protocol, registration, {}, central, draws, fuels)
    assert not tables[2][tables[2].op == "TAKE-OFF"].claimed.any()
    assert not tables[2][tables[2].op == "TAKE-OFF"].all_converged.any()
