"""Post-chain negative fixtures; no C++/MLX import or scientific solve."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase8 import scientific_workflow_gate as gate


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def ownership(tmp_path, monkeypatch):
    root = tmp_path
    lease_path = root / "outputs/operations/owner.json"
    fake = SimpleNamespace(process_birth=lambda pid: "born", DEAD="dead",
                           liveness=lambda pid, birth: "alive" if pid == os.getpid() else "dead",
                           read_power=lambda: "AC", on_ac=lambda power: power == "AC")
    monkeypatch.setattr(gate, "_ac", lambda root: fake)
    class Context:
        identity = {"registration_sha256": "r"}
        op = {"paths": {"owner_lease": "outputs/operations/owner.json"}}
        _owned = None
        def require_idle_ac(self):
            if lease_path.exists():
                lease = gate.read(lease_path)
                if lease.get("owner_pid") != os.getpid() or lease.get("reservation_sha256") != self._owned:
                    raise gate.GateError("foreign lease")
        acquire_run = gate.Context.acquire_run
    ctx = Context()
    ctx.root = root
    return ctx, fake, lease_path


def artifact(ctx, owned):
    path = owned.out / "rows.csv"
    path.write_text("value\n1\n")
    name = str(path.relative_to(ctx.root))
    return {"status": "COMPLETE", "exit_code": 0, "artifact_hashes": {name: gate.digest(path)},
            "expected_outputs": [name], "outputs_complete": True}


def test_existing_lease_refuses_new_reservation(ownership):
    ctx, _, path = ownership
    put(path, {"owner_pid": -1, "reservation_sha256": "foreign"})
    with pytest.raises(gate.GateError):
        ctx.acquire_run("outputs/new", "r")
    assert not (ctx.root / "outputs/new").exists()
    assert gate.read(path)["reservation_sha256"] == "foreign"


def test_drift_is_error_not_complete_and_reservation_survives(ownership):
    ctx, _, path = ownership
    owned = ctx.acquire_run("outputs/new", "r")
    success = artifact(ctx, owned)
    ctx.identity = {"registration_sha256": "changed"}
    terminal = owned.release(success)
    assert terminal["status"] == "ERROR"
    assert terminal["exit_code"] != 0 and terminal["outputs_complete"] is False
    assert not path.exists()
    assert (owned.out / "reservation.json").exists()
    with pytest.raises(gate.GateError):
        ctx.acquire_run("outputs/new", "changed")


@pytest.mark.parametrize("fault", ["deleted", "hash", "coverage", "complete"])
def test_missing_or_changed_success_coverage_denies_complete(ownership, fault):
    ctx, _, _ = ownership
    owned = ctx.acquire_run("outputs/new", "r")
    success = artifact(ctx, owned)
    name = success["expected_outputs"][0]
    if fault == "deleted":
        (ctx.root / name).unlink()
    elif fault == "hash":
        (ctx.root / name).write_text("value\n2\n")
    elif fault == "coverage":
        success["expected_outputs"].append("outputs/new/missing.csv")
    else:
        success["outputs_complete"] = False
    assert owned.release(success)["status"] == "ERROR"


def test_ambiguous_child_keeps_lease_and_no_terminal(ownership):
    ctx, fake, path = ownership
    owned = ctx.acquire_run("outputs/new", "r")
    owned.record_children([{"pid": 9900, "birth": "born", "argv": ["fixture"]}])
    fake.liveness = lambda pid, birth: "unknown"
    with pytest.raises(gate.GateError, match="Child exit"):
        owned.release(artifact(ctx, owned))
    assert gate.read(path)["state"] == "AMBIGUOUS_CHILD"
    assert not (owned.out / "terminal.json").exists()


@pytest.mark.parametrize("operation", ["assert_current", "record_children", "release"])
def test_released_run_cannot_mutate_new_lease(ownership, operation):
    ctx, _, path = ownership
    old = ctx.acquire_run("outputs/old", "r")
    old.release(artifact(ctx, old))
    new = ctx.acquire_run("outputs/new", "r")
    before = path.read_bytes()
    with pytest.raises(gate.GateError):
        if operation == "assert_current":
            old.assert_current()
        elif operation == "record_children":
            old.record_children([])
        else:
            old.release({"status": "ERROR", "exit_code": 1, "errors": ["old run"]})
    assert path.read_bytes() == before
    new.release(artifact(ctx, new))


@pytest.mark.parametrize("field,value", [("identity", {"registration_sha256": "forged"}),
                                          ("owner_birth", "another birth"), ("argv", ["foreign"])])
def test_reservation_mutation_is_error(ownership, field, value):
    ctx, _, _ = ownership
    owned = ctx.acquire_run("outputs/new", "r")
    success = artifact(ctx, owned)
    reservation = gate.read(owned.out / "reservation.json")
    reservation[field] = value
    put(owned.out / "reservation.json", reservation)
    with pytest.raises(gate.GateError, match="reservation"):
        owned.assert_current()
    terminal = owned.release(success)
    assert terminal["status"] == "ERROR"
    assert terminal["exit_code"] != 0 and terminal["outputs_complete"] is False


@pytest.mark.parametrize("field,value", [
    ("identity", {"registration_sha256": "forged"}), ("registration_sha256", "forged"),
    ("owner_birth", "another birth"), ("argv", ["foreign"]), ("created_utc", "changed"),
    ("output_dir", "outputs/other"), ("owner_pid", -100),
])
def test_immutable_lease_mutation_never_releases_success(ownership, field, value):
    ctx, _, path = ownership
    owned = ctx.acquire_run("outputs/new", "r")
    success = artifact(ctx, owned)
    reservation_bytes = (owned.out / "reservation.json").read_bytes()
    lease = gate.read(path)
    lease[field] = value
    put(path, lease)
    with pytest.raises(gate.GateError):
        owned.assert_current()
    if field in {"owner_pid", "owner_birth"}:
        with pytest.raises(gate.GateError, match="ownership"):
            owned.release(success)
        assert path.exists() and not (owned.out / "terminal.json").exists()
    else:
        terminal = owned.release(success)
        assert terminal["status"] == "ERROR"
        assert terminal["exit_code"] != 0 and terminal["outputs_complete"] is False
    assert (owned.out / "reservation.json").read_bytes() == reservation_bytes


@pytest.mark.parametrize("entry", [{}, {"mode": "100644"}, {"sha256": "a" * 64},
                                   {"mode": "644", "sha256": "a" * 64},
                                   {"mode": "100644", "sha256": "z" * 64}])
def test_forged_expected_file_entries_fail_before_source_acceptance(tmp_path, entry):
    put(tmp_path / "docs/reg.json", {"id": "fixture"})
    source = tmp_path / "scripts/consumer.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")
    expected = {"registration_sha256": gate.digest(tmp_path / "docs/reg.json"),
                "files": {"scripts/consumer.py": entry}}
    with pytest.raises(gate.GateError, match="Malformed expected consumer file identity"):
        gate._scientific_expected(tmp_path, "docs/reg.json", expected)
    expected["files"]["scripts/consumer.py"] = gate.file_identity(source)
    gate._scientific_expected(tmp_path, "docs/reg.json", expected)


@pytest.mark.parametrize("registration", ["docs/phase7_p73_a1_registration.json",
    "docs/phase8_saf_surrogate_registration.json", "docs/phase8_nozzle_ode_registration.json",
    "docs/phase8_screening_tool_registration.json"])
def test_fresh_g0_bypass_refused_before_context_refresh(tmp_path, monkeypatch, registration):
    refreshed = []
    monkeypatch.setattr(gate.Context, "_refresh", lambda self: refreshed.append(self.registration))
    with pytest.raises(gate.GateError, match="Only the registered G0 wrapper"):
        gate.prepare_context(tmp_path, registration, require_g0=False)
    assert refreshed == []
    gate.prepare_context(tmp_path, gate.OPERATIONS, require_g0=False)
    assert refreshed == [gate.OPERATIONS]


def test_write_once_and_unsafe_paths(tmp_path):
    p = tmp_path / "record.json"
    gate.write_once(p, {"x": 1})
    with pytest.raises(FileExistsError):
        gate.write_once(p, {"x": 2})
    for path in ("../escape", "/absolute", "."):
        with pytest.raises(gate.GateError):
            gate.relative(tmp_path, path)
    assert gate.read(p) == {"x": 1}


@pytest.mark.parametrize("fault", ["none", "truncated", "partial_fail", "valid_fail",
                                   "foreign_owner", "foreign_argv", "wrong_phases", "false_exit", "raw_hash"])
def test_surrogate_terminal_requires_registered_coverage(tmp_path, monkeypatch, fault):
    root = tmp_path
    out = root / "outputs/study"
    reg = "docs/study.json"
    put(root / reg, {"id": "P8-S-20261004", "artifact_root": "outputs/study",
                    "provenance": {"successful_release": {"expected_outputs": ["rows.csv", "metrics.json"]}}})
    ctx = SimpleNamespace(root=root, registration=reg, identity={"source": "fixture", "registration_sha256": "reg"},
                          binary_sha256="core", binary_path=root / "cpp/core.so")
    monkeypatch.setattr(gate, "prepare_context", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(gate, "require_committed", lambda *args: None)
    monkeypatch.setattr(gate, "_ac", lambda *args: SimpleNamespace(liveness=lambda *args: "dead"))
    put(out / "reservation.json", {"identity": ctx.identity, "owner_pid": 10, "owner_birth": "fixture",
                                   "orig_argv": ["fixture", "run"]})
    body = {"identity": ctx.identity, "pid": 10, "birth": "fixture", "argv": ["fixture", "run"], "started": "before"}
    put(out / "command.start.json", {**body, "native_command": "fixture run", "registration_sha256": "reg",
                                      "binary_path": "cpp/core.so", "binary_sha256": "core"})
    put(out / "command.exit.json", {**body, "finished": "after", "in_process_completed": True,
        "completed_phases": ["freeze", "source_checks", "generate", "train", "seal", "score", "timing", "study"],
        "exit_code": 1 if fault == "valid_fail" else 0, "scientific_verdict": "FAIL" if fault == "valid_fail" else "PASS"})
    (out / "rows.csv").write_text("value\n1\n")
    put(out / "metrics.json", {"verdict": "fixture"})
    paths = ["outputs/study/rows.csv", "outputs/study/metrics.json"]
    terminal = {"identity": ctx.identity, "status": "PASS", "exit_code": 0, "outputs_complete": True,
                "reservation_sha256": gate.digest(out / "reservation.json"), "expected_outputs": paths,
                "scientific_verdict": "PASS",
                "artifact_hashes": {name: gate.digest(root / name) for name in [*paths,
                    "outputs/study/command.start.json", "outputs/study/command.exit.json"]}}
    if fault == "truncated":
        terminal["expected_outputs"] = paths[:1]
        terminal["artifact_hashes"].pop(paths[1])
    if fault in {"partial_fail", "valid_fail"}:
        terminal.update(status="FAIL", exit_code=1, outputs_complete=False, scientific_verdict="FAIL",
                        execution_complete=fault == "valid_fail")
    if fault in {"foreign_owner", "foreign_argv", "wrong_phases", "false_exit", "raw_hash"}:
        path = out / "command.exit.json"
        command = gate.read(path)
        if fault == "foreign_owner": command["pid"] = 11
        elif fault == "foreign_argv": command["argv"] = ["foreign"]
        elif fault == "wrong_phases": command["completed_phases"] = ["score"]
        elif fault == "false_exit": command["exit_code"] = 1
        else: command["finished"] = "changed after freezing"
        put(path, command)
        if fault != "raw_hash": terminal["artifact_hashes"]["outputs/study/command.exit.json"] = gate.digest(path)
    put(out / "terminal.json", terminal)
    put(out / "released_lease.json", {"identity": ctx.identity, "owner_pid": 10, "owner_birth": "fixture",
        "state": "RELEASED", "reservation_sha256": terminal["reservation_sha256"],
        "terminal_sha256": gate.digest(out / "terminal.json")})
    if fault not in {"none", "valid_fail"}:
        with pytest.raises(gate.GateError):
            gate.validate_consumer_terminal(root, reg, out, allow_scientific_fail=True)
    else:
        proof = gate.validate_consumer_terminal(root, reg, out, allow_scientific_fail=True)
        assert proof["status"] == terminal["status"]


def test_consumer_registration_change_refused(tmp_path):
    reg = tmp_path / "docs/reg.json"
    put(reg, {"version": 1})
    expected = {"registration_sha256": gate.digest(reg), "files": {}}
    gate._scientific_expected(tmp_path, "docs/reg.json", expected)
    put(reg, {"version": 2})
    with pytest.raises(gate.GateError, match="registration"):
        gate._scientific_expected(tmp_path, "docs/reg.json", expected)


def test_executable_mode_and_symlink_target_are_scientific_identity(tmp_path):
    p = tmp_path / "source.py"
    p.write_text("value = 1\n")
    before = gate.file_identity(p)
    p.chmod(0o755)
    assert gate.file_identity(p)["mode"] != before["mode"]
    link = tmp_path / "link.py"
    link.symlink_to("source.py")
    linked = gate.file_identity(link)
    p.write_text("value = 2\n")
    assert gate.file_identity(link)["sha256"] == linked["sha256"]
    assert gate.file_identity(link)["target_sha256"] != linked["target_sha256"]


def test_original_raw_evidence_mutation_denied_before_validator(tmp_path, monkeypatch):
    root = tmp_path
    original = root / "scripts/source.py"
    original.parent.mkdir(parents=True)
    original.write_text("x=1\n")
    entry = gate.file_identity(original)
    raw = f"{entry['mode']} blob {entry['blob']}\tscripts/source.py\n"
    reg_path = "docs/main.json"
    put(root / reg_path, {"lease_path": "outputs/main/owner.json", "identity": {
        "tracked_paths": ["scripts"], "built_modules_glob": "cpp/build/*.so"}})
    put(root / gate.OPERATIONS, {"id": "fixture"})
    evidence = root / "outputs/main/result.json"
    put(evidence, {"status": "FAIL"})
    identity = {"git_head": "old", "tracked_tree_sha256": hashlib.sha256(raw.encode()).hexdigest(), "built_modules": {}}
    dep = {"id": "P8-SCREENING-MAIN-DEPENDENCY", "operations_registration_sha256": gate.digest(root / gate.OPERATIONS),
           "main_registration_sha256": gate.digest(root / reg_path), "identity": identity,
           "original_files": {"scripts/source.py": entry}, "raw_evidence": {"outputs/main/result.json": gate.digest(evidence)},
           "context": {"identity": identity, "status": "FAIL"}}
    monkeypatch.setattr(gate, "_tree", lambda *args: (raw, {"scripts/source.py": {"mode": entry["mode"], "blob": entry["blob"]}}))
    called = []
    monkeypatch.setattr(gate, "_original_validate", lambda *args: called.append(True) or dep["context"])
    op = {"paths": {"main_registration": reg_path}}
    gate._prove_original(root, op, dep)
    put(evidence, {"status": "PASS"})
    with pytest.raises(gate.GateError, match="Raw main evidence"):
        gate._prove_original(root, op, dep)
    assert len(called) == 1


@pytest.fixture
def g0_tables(tmp_path):
    # Use the exact pinned original comparison function on synthetic CSV only.
    original = ROOT / "scripts/phase8/g0_parity.py"
    source = tmp_path / "scripts/phase8/g0_parity.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(original.read_bytes())
    names = {"cal": "outputs/phase7/calibration_v6_rows.csv",
             "hold": "outputs/phase7/holdout_icao_validation_v6.csv",
             "hold_summary": "outputs/phase7/holdout_icao_validation_summary_v6.csv",
             "ae3": "outputs/design_point_summary_v5.csv"}
    files, checks = {}, {}
    stems = {"cal": "calibration_v6_rows", "hold": "holdout_icao_validation_v6", "hold_summary": "holdout_icao_validation_summary_v6", "ae3": "design_point_summary_v5"}
    for key, name in names.items():
        frozen, new = tmp_path / name, tmp_path / f"outputs/new/{stems[key]}_cpp.csv"
        for p in (frozen, new):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("value,label\n1,same\n")
            files[str(p.relative_to(tmp_path))] = gate.digest(p)
        checks[key] = {"frozen": name, "new": str(new.relative_to(tmp_path)), "rows": 1,
                       "columns": {"value": {"numeric": True, "max_rel_diff": 0.0, "match": True},
                                   "label": {"numeric": False, "match": True}}, "match": True}
    py_checks = copy.deepcopy({k:v for k,v in checks.items() if k != "ae3"})
    for c in py_checks.values():
        cpp = tmp_path / c["new"]
        c["new"] = c["new"].replace("_cpp.csv", "_python.csv")
        (tmp_path / c["new"]).write_bytes(cpp.read_bytes())
        files[c["new"]] = gate.digest(tmp_path / c["new"])
    verdict = {"backends": {"cpp": {"checks": checks}, "python": {"checks": py_checks}}}
    op = {"original_pins": {"scripts/phase8/g0_parity.py": gate.digest(source)}, "paths": {"g0_target": "outputs/new"}}
    return tmp_path, op, verdict, files


def test_g0_comparison_recomputed_from_raw_tables(g0_tables):
    tmp_path, op, verdict, files = g0_tables
    checks = verdict["backends"]["cpp"]["checks"]
    gate.recompare_g0(tmp_path, op, verdict, files)
    (tmp_path / checks["cal"]["new"]).write_text("value,label\n1.001,same\n")
    # A claimed PASS and even an updated hash cannot disguise a numeric mismatch.
    files[checks["cal"]["new"]] = gate.digest(tmp_path / checks["cal"]["new"])
    with pytest.raises(gate.GateError, match="Recomputed"):
        gate.recompare_g0(tmp_path, op, verdict, files)


@pytest.mark.parametrize("fault", ["frozen_copy", "foreign_target", "wrong_table_name"])
def test_g0_foreign_or_copied_regenerated_table_cannot_claim_parity(g0_tables, fault):
    root, op, verdict, files = g0_tables
    check = verdict["backends"]["cpp"]["checks"]["cal"]
    if fault == "frozen_copy":
        check["new"] = check["frozen"]
    else:
        check["new"] = "outputs/foreign/calibration_v6_rows_cpp.csv" if fault == "foreign_target" else "outputs/new/copied_cpp.csv"
        foreign = root / check["new"]
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_bytes((root / check["frozen"]).read_bytes())
        files[check["new"]] = gate.digest(foreign)
    with pytest.raises(gate.GateError, match="Foreign G0 regenerated table"):
        gate.recompare_g0(root, op, verdict, files)


@pytest.mark.parametrize("fault", ["spawn_raises", "no_process", "two_processes", "birth_unknown"])
def test_tracked_pool_ambiguous_spawn_preserves_starting_slot(tmp_path, monkeypatch, fault):
    from scripts.phase8 import post_chain_g0 as g0
    monkeypatch.setattr(g0, "gate", gate)
    spec_path = tmp_path / "outputs/attempt/command_spec.json"
    put(spec_path, {"root": str(tmp_path), "attempt": "outputs/attempt"})
    monkeypatch.setenv(g0.SPEC_ENV, str(spec_path))
    pool = g0.TrackedPool.__new__(g0.TrackedPool)
    pool._processes, pool._provenance_pool, pool._provenance_spawn = {}, 0, 0
    def spawn(self):
        if fault == "spawn_raises":
            self._processes[9901] = object()
            raise RuntimeError("mock interrupted spawn")
        if fault != "no_process":
            self._processes[9901] = object()
        if fault == "two_processes":
            self._processes[9902] = object()
    monkeypatch.setattr(g0.ProcessPoolExecutor, "_spawn_process", spawn)
    monkeypatch.setattr(gate, "_ac", lambda root: SimpleNamespace(DEAD="dead", process_birth=lambda pid: None if fault == "birth_unknown" else "born"))
    with pytest.raises((gate.GateError, RuntimeError)):
        pool._spawn_process()
    slot = spec_path.parent / "spawns/pool_0_spawn_0.starting.json"
    assert gate.read(slot) == {"state": "STARTING", "parent_pid": os.getpid(), "spec_sha256": gate.digest(spec_path)}
    assert not slot.with_name("pool_0_spawn_0.complete.json").exists()


def test_tracked_pool_records_actual_created_pid_and_birth(tmp_path, monkeypatch):
    from scripts.phase8 import post_chain_g0 as g0
    monkeypatch.setattr(g0, "gate", gate)
    spec_path = tmp_path / "outputs/attempt/command_spec.json"
    put(spec_path, {"root": str(tmp_path), "attempt": "outputs/attempt"})
    monkeypatch.setenv(g0.SPEC_ENV, str(spec_path))
    pool = g0.TrackedPool.__new__(g0.TrackedPool)
    pool._processes, pool._provenance_pool, pool._provenance_spawn = {}, 3, 0
    monkeypatch.setattr(g0.ProcessPoolExecutor, "_spawn_process", lambda self: self._processes.update({9901: object()}))
    monkeypatch.setattr(gate, "_ac", lambda root: SimpleNamespace(DEAD="dead", process_birth=lambda pid: "worker birth"))
    pool._spawn_process()
    proof = gate.read(spec_path.parent / "spawns/pool_3_spawn_0.complete.json")
    assert proof["pid"] == 9901 and proof["birth"] == "worker birth"
    assert proof["parent_pid"] == os.getpid() and proof["spec_sha256"] == gate.digest(spec_path)


@pytest.mark.parametrize("ending", ["starting_only", "worker_unknown", "all_dead"])
def test_g0_final_sync_retains_unproven_spawn_or_records_ended_worker(ownership, monkeypatch, ending):
    from scripts.phase8 import post_chain_g0 as g0
    ctx, fake, lease_path = ownership
    op = {"paths": {"owner_lease": "outputs/operations/owner.json", "g0_attempt": "outputs/g0_attempt",
                    "g0_target": "outputs/g0_target", "g0_evidence": "outputs/g0_receipt.json"}}
    ctx.op = op
    ctx.identity = {"registration_sha256": "r", "core": {"path": "cpp/mock_core.so", "sha256": "a" * 64}}
    monkeypatch.setattr(g0, "gate", gate)
    monkeypatch.setattr(g0, "ROOT", ctx.root)
    monkeypatch.setattr(gate, "operations", lambda root: op)
    monkeypatch.setattr(gate, "prepare_context", lambda *args, **kwargs: ctx)
    fake.liveness = lambda pid, birth: "alive" if pid == os.getpid() else "unknown" if ending == "worker_unknown" and pid == 9901 else "dead"
    class Child:
        pid = 9900
        def poll(self): return 1
        def wait(self): return 1
    def launch(argv, **kwargs):
        attempt = ctx.root / op["paths"]["g0_attempt"]
        spec_sha = gate.digest(attempt / "command_spec.json")
        put(attempt / "spawns/pool_0_spawn_0.starting.json", {"state": "STARTING", "parent_pid": 9900, "spec_sha256": spec_sha})
        if ending != "starting_only":
            put(attempt / "spawns/pool_0_spawn_0.complete.json", {"state": "CREATED", "parent_pid": 9900,
                "pid": 9901, "birth": "worker birth", "argv": ["mock worker"], "spec_sha256": spec_sha})
        return Child()
    monkeypatch.setattr(g0.subprocess, "Popen", launch)
    if ending == "all_dead":
        assert g0.run(gate.OPERATIONS, op["paths"]["g0_target"]) == 1
        released = gate.read(ctx.root / op["paths"]["g0_attempt"] / "released_lease.json")
        assert {(c["pid"], c["birth"]) for c in released["children"]} == {(9900, "born"), (9901, "worker birth")}
        terminal = gate.read(ctx.root / op["paths"]["g0_attempt"] / "terminal.json")
        assert terminal["status"] == "ERROR" and terminal["exit_code"] != 0
        assert not lease_path.exists()
    else:
        with pytest.raises(gate.GateError, match="Child exit unproven"):
            g0.run(gate.OPERATIONS, op["paths"]["g0_target"])
        assert gate.read(lease_path)["state"] == "AMBIGUOUS_CHILD"
        assert not (ctx.root / op["paths"]["g0_attempt"] / "terminal.json").exists()
