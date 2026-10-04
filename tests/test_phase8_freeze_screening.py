"""Pure fixtures for published-only quantitative packaging; no simulator imports."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.phase8 import freeze_screening as freeze


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def file_identity(path):
    return {"sha256": freeze.sha256(path), "mode": "100644"}


@pytest.mark.parametrize("name", ["/tmp/labels.csv", "outputs/../labels.csv",
                                "data/labels.csv", "outputs/sealed_test/labels.csv"])
def test_published_paths_refuse_private_or_foreign_locations(tmp_path, name):
    with pytest.raises(freeze.Incomplete):
        freeze.safe_path(tmp_path, name)


def test_published_paths_refuse_parent_symlink(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/redirect").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(freeze.Incomplete):
        freeze.safe_path(tmp_path, "outputs/redirect/scored.csv")


@pytest.mark.parametrize("value", [True, None, "", "nan", "inf", "-inf"])
def test_missing_or_nonfinite_numbers_are_never_placeholders(value):
    with pytest.raises(freeze.Incomplete):
        freeze.number(value)


@pytest.mark.parametrize("mutation", ["sealed", "foreign_filename", "wrong_format", "hash_bypass", "duplicate", "missing"])
def test_published_descriptor_cannot_bypass_owned_source_or_read_private_labels(tmp_path, mutation):
    entry = {"role": "surrogate_simulator_parity", "path": "outputs/scored/test_predictions.csv", "format": "csv",
             "producer_registration": "docs/scientific.json", "producer_output_dir": "outputs/scored"}
    entries = [entry]
    if mutation == "sealed": entry["path"] = "outputs/sealed/test_predictions.csv"
    elif mutation == "foreign_filename": entry["path"] = "outputs/scored/private_teacher.csv"
    elif mutation == "wrong_format": entry["format"] = "json"
    elif mutation == "hash_bypass": entry["expected_sha256"] = "0" * 64
    elif mutation == "duplicate": entries.append(dict(entry))
    elif mutation == "missing": entries.clear()
    context = SimpleNamespace(root=tmp_path)
    # No read/hash hooks exist: rejection must occur before artifact bytes.
    with pytest.raises(freeze.Incomplete):
        freeze.Published(context, {"published_inputs": entries}, SimpleNamespace())


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    frozen_source = root / "scripts/phase8/freeze_screening.py"
    frozen_source.parent.mkdir(parents=True)
    frozen_source.write_text("# synthetic committed source\n")
    tests = ["tests/test_phase8_scientific_workflow_gate.py", "tests/test_phase8_screening_product.py",
             "tests/test_phase7_p73_a1_cpp.py", "tests/test_phase8_saf_surrogate.py",
             "tests/test_phase8_saf_surrogate_numerical.py", "tests/test_phase8_nozzle_ode.py",
             "tests/test_phase8_freeze_screening.py"]
    for name in tests:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# synthetic test identity\n")
    test_files = {name: file_identity(root / name) for name in tests}
    original_name = "outputs/phase8/screening_operations/main_dependency.json"
    write(root / original_name, {"original_files": test_files})
    verification = {"path": "outputs/phase8/screening_operations/product_verification.json",
        "command_spec": "outputs/phase8/screening_operations/product_verification.command_spec.json",
        "command_log": "outputs/phase8/screening_operations/product_verification.command.log",
        "command_exit": "outputs/phase8/screening_operations/product_verification.command.exit.json",
        "junit_xml": "outputs/phase8/screening_operations/product_verification_run/junit.xml",
        "registered_tests": tests}
    reg = {"id": "P8-SCREENING-TOOL-20261004", "outputs": {"root": "outputs/freeze", "local_tag": "freeze-2026-10-18"},
           "quantitative_freeze_2026_10_04": {"required": ["NUMBERS.md", *freeze.PREFIXES],
                                             "verification_receipt": verification}, "published_inputs": []}
    write(root / freeze.REGISTRATION, reg)
    published = "outputs/phase8/saf_surrogate/attempt_001/timing.json"
    write(root / published, {"synthetic_published_metric": 7})
    identity = {"registration_sha256": freeze.sha256(root / freeze.REGISTRATION), "new_files": {},
                "core": {"path": "cpp/synthetic.so", "sha256": "1" * 64}, "g0_sha256": "2" * 64}
    state = SimpleNamespace(statuses={role: "PASS" for role in freeze.REQUIRED_ROLES}, terminal_override=None,
                            committed=[], guard_calls=0, run=None)

    class Owned:
        def __init__(self, context):
            self.out = root / "outputs/freeze"
            self.out.mkdir(parents=True, exist_ok=False)
            self.context = context
            self.reservation = {"identity": identity, "owner_pid": 81, "owner_birth": "owner-birth",
                                "argv": ["synthetic-freeze"], "children": []}
            write(self.out / "reservation.json", self.reservation)
            state.run = self

        def assert_current(self):
            state.guard_calls += 1

        def release(self, terminal):
            terminal = copy.deepcopy(terminal)
            terminal.update(identity=identity, reservation_sha256=freeze.sha256(self.out / "reservation.json"))
            if state.terminal_override:
                terminal.update(status=state.terminal_override, exit_code=1, execution_complete=False)
            write(self.out / "terminal.json", terminal)
            write(self.out / "released_lease.json", {**self.reservation, "state": "RELEASED",
                "reservation_sha256": terminal["reservation_sha256"], "terminal_sha256": freeze.sha256(self.out / "terminal.json")})
            return terminal

    context = SimpleNamespace(root=root, identity=identity, binary_sha256="1" * 64,
        op={"paths": {"main_dependency": original_name}}, require_idle_ac=lambda: None)
    context.acquire_run = lambda *args, **kwargs: Owned(context)

    def write_once(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, allow_nan=False)

    def terminal_proof(repo, registration, out, **kwargs):
        terminal = json.loads((root / out / "terminal.json").read_text())
        if terminal["status"] not in {"COMPLETE", "FAIL"}:
            raise freeze.Incomplete("terminal is not complete")
        for name in kwargs["artifact_paths"]:
            if terminal["artifact_hashes"].get(name) != freeze.sha256(root / name):
                raise freeze.Incomplete("terminal artifact coverage or hash changed")
        return {"status": terminal["status"]}

    gate = SimpleNamespace(read=lambda path: json.loads(path.read_text()), write_once=write_once,
        file_identity=file_identity, prepare_context=lambda *args, **kwargs: context,
        validate_consumer_terminal=terminal_proof,
        require_committed=lambda repo, paths: state.committed.extend(paths))

    class Published:
        def __init__(self, *args):
            self.hashes = {published: freeze.sha256(root / published)}
            self.statuses = dict(state.statuses)
            self.proofs = {role: {"status": verdict} for role, verdict in self.statuses.items()}

        def assert_current(self):
            if freeze.sha256(root / published) != self.hashes[published]:
                raise freeze.Incomplete("published drift")

    class Figure:
        def suptitle(self, text, **kwargs):
            assert "conditional on v6 calibration" in text

        def tight_layout(self, **kwargs):
            pass

        def savefig(self, stream, **kwargs):
            stream.write(b"synthetic scientific figure bytes\n")

    def draw(data, plt, rows):
        freeze.metric(rows, "synthetic", "case", "existing_published_metric", 7, "ratio", "PASS")
        return Figure()

    matplotlib = SimpleNamespace(use=lambda *args: None, __version__="synthetic", rcParams={})
    pyplot = SimpleNamespace(close=lambda *args: None)
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setattr(freeze.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(freeze, "ROOT", root)
    monkeypatch.setattr(freeze, "workflow_gate", lambda: gate)
    monkeypatch.setattr(freeze, "Published", Published)
    monkeypatch.setattr(freeze, "DRAW", {prefix: draw for prefix in freeze.PREFIXES})
    return SimpleNamespace(root=root, reg=reg, state=state, context=context, gate=gate,
        tests=test_files, descriptor=verification, published=published)


def add_verification(fixture):
    root, descriptor = fixture.root, fixture.descriptor
    interpreter = root / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("synthetic interpreter bytes\n")
    command = {"argv": [str(interpreter), "-m", "pytest", *descriptor["registered_tests"], "-v", "--junitxml=" + descriptor["junit_xml"]],
        "workdir": str(root), "interpreter": str(interpreter), "interpreter_sha256": freeze.sha256(interpreter),
        "owner_pid": 90, "owner_birth": "verification-owner", "child_pid": 91, "child_birth": "verification-child"}
    shared = {"identity": fixture.context.identity, "test_files": fixture.tests, "command": command}
    write(root / descriptor["command_spec"], shared)
    log = root / descriptor["command_log"]
    log.write_text("collected 17 items\n================ 17 passed in 0.02s ================\n")
    junit = root / descriptor["junit_xml"]
    junit.parent.mkdir(parents=True)
    junit.write_text('<testsuites><testsuite tests="17" errors="0" failures="0" skipped="0">'
        + ''.join('<testcase name="synthetic_' + str(index) + '"/>' for index in range(17)) + '</testsuite></testsuites>')
    end = {**shared, "status": "PASS", "exit_code": 0, "waited": True, "passed": 17, "skipped": 0,
           "ended_utc": "2026-10-04T01:00:00+00:00", "command_spec_sha256": freeze.sha256(root / descriptor["command_spec"]),
           "log_sha256": freeze.sha256(log), "junit_sha256": freeze.sha256(junit)}
    write(root / descriptor["command_exit"], end)
    receipt = {**shared, **{key: end[key] for key in ("status", "exit_code", "waited", "passed", "skipped", "ended_utc")},
        "raw_sha256": {descriptor[key]: freeze.sha256(root / descriptor[key]) for key in ("command_spec", "command_log", "command_exit", "junit_xml")}}
    write(root / descriptor["path"], receipt)
    return receipt


def test_complete_packaging_binds_all_figures_and_owner_body(packaged):
    assert freeze.run(packaged.root) == 0
    out = packaged.root / "outputs/freeze"
    terminal = json.loads((out / "terminal.json").read_text())
    assert terminal["status"] == "COMPLETE" and terminal["execution_complete"] is True
    assert terminal["expected_outputs"] == freeze.terminal_outputs()
    assert all((packaged.root / name).is_file() for name in terminal["expected_outputs"])
    command = json.loads((out / "command.exit.json").read_text())
    assert command["owner_pid"] == 81 and command["owner_birth"] == "owner-birth"
    assert command["argv"] == ["synthetic-freeze"] and command["in_process_completed"] is True
    assert command["waited_child_exit"] == "not_applicable_synchronous_owner"
    assert command["log_sha256"] == freeze.sha256(out / "execution.log")
    assert packaged.state.guard_calls >= 26


def test_completed_scientific_fail_is_not_promoted(packaged):
    packaged.state.statuses["nozzle_pinn_exact"] = "FAIL"
    assert freeze.run(packaged.root) == 1
    out = packaged.root / "outputs/freeze"
    terminal = json.loads((out / "terminal.json").read_text())
    receipt = json.loads((out / "freeze_receipt.json").read_text())
    assert terminal["status"] == receipt["status"] == "FAIL"
    assert terminal["scientific_verdict"] == "FAIL" and terminal["exit_code"] != 0
    assert terminal["execution_complete"] is True and terminal["outputs_complete"] is False
    assert (out / "NUMBERS.md").is_file() and receipt["tag_created"] is False


def test_missing_component_keeps_partial_evidence_and_continues(packaged, monkeypatch):
    def missing(*args):
        raise freeze.Incomplete("missing scored artifact")
    draws = dict(freeze.DRAW)
    draws["learning_curve"] = missing
    monkeypatch.setattr(freeze, "DRAW", draws)
    assert freeze.run(packaged.root) == 1
    out = packaged.root / "outputs/freeze"
    receipt = json.loads((out / "freeze_receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE" and receipt["execution_complete"] is False
    assert "g0_parity" in receipt["completed_figures"] and len(receipt["completed_figures"]) == 7
    assert not (out / "NUMBERS.md").exists()
    assert (out / "speed_break_even.png").is_file() and (out / "g0_parity.pdf").is_file()


def test_release_drift_cannot_authorize_tag(packaged):
    packaged.state.terminal_override = "ERROR"
    assert freeze.run(packaged.root) == 1
    add_verification(packaged)
    with pytest.raises(freeze.Incomplete, match="terminal"):
        freeze.tag_guard(packaged.root, "outputs/freeze/freeze_receipt.json",
                         verification_receipt=packaged.descriptor["path"])


@pytest.mark.parametrize("failed", [False, True])
def test_guard_accepts_only_review_ready_quantitative_completion(packaged, failed):
    if failed:
        packaged.state.statuses["nozzle_pinn_exact"] = "FAIL"
    assert freeze.run(packaged.root) == (1 if failed else 0)
    add_verification(packaged)
    ready = freeze.tag_guard(packaged.root, "outputs/freeze/freeze_receipt.json",
                             verification_receipt=packaged.descriptor["path"])
    assert ready["status"] == "READY_FOR_ROOT_REVIEW" and ready["tag_created"] is False
    assert ready["scientific_verdict"] == ("FAIL" if failed else "PASS")
    assert set(packaged.descriptor["registered_tests"]) <= set(packaged.state.committed)


@pytest.mark.parametrize("mutation", ["pending", "no_wait", "wrong_count", "foreign_owner", "changed_log", "missing_test", "skipped"])
def test_verification_refuses_summary_only_or_changed_raw_proof(packaged, mutation):
    freeze.run(packaged.root)
    receipt = add_verification(packaged)
    if mutation == "pending": receipt["status"] = "PENDING"
    elif mutation == "no_wait": receipt["waited"] = False
    elif mutation == "wrong_count": receipt["passed"] = 18
    elif mutation == "foreign_owner": receipt["command"]["owner_birth"] = "foreign"
    elif mutation == "changed_log": (packaged.root / packaged.descriptor["command_log"]).write_text("17 passed\nforged\n")
    elif mutation == "missing_test": receipt["test_files"].pop("tests/test_phase8_freeze_screening.py")
    elif mutation == "skipped": receipt["skipped"] = 1
    write(packaged.root / packaged.descriptor["path"], receipt)
    with pytest.raises(freeze.Incomplete):
        freeze.tag_guard(packaged.root, "outputs/freeze/freeze_receipt.json",
                         verification_receipt=packaged.descriptor["path"])


def test_tag_guard_refuses_changed_published_data(packaged):
    freeze.run(packaged.root)
    add_verification(packaged)
    write(packaged.root / packaged.published, {"synthetic_published_metric": 8})
    with pytest.raises(freeze.Incomplete, match="producers"):
        freeze.tag_guard(packaged.root, "outputs/freeze/freeze_receipt.json",
                         verification_receipt=packaged.descriptor["path"])


def test_output_reservation_never_reuses_old_freeze_directory(packaged):
    freeze.run(packaged.root)
    with pytest.raises(FileExistsError):
        freeze.run(packaged.root)


@pytest.mark.parametrize("mutation", ["malformed", "count", "failure", "skipped", "missing_cases"])
def test_actual_junit_required_even_when_all_summary_hashes_are_updated(packaged, mutation):
    freeze.run(packaged.root)
    receipt = add_verification(packaged)
    junit_path = packaged.root / packaged.descriptor["junit_xml"]
    text = junit_path.read_text()
    if mutation == "malformed": text = "not XML"
    elif mutation == "count": text = text.replace('tests="17"', 'tests="16"')
    elif mutation == "failure": text = text.replace('failures="0"', 'failures="1"')
    elif mutation == "skipped": text = text.replace('skipped="0"', 'skipped="1"')
    elif mutation == "missing_cases": text = text.replace('<testcase name="synthetic_0"/>', '')
    junit_path.write_text(text)
    end_path = packaged.root / packaged.descriptor["command_exit"]
    end = json.loads(end_path.read_text())
    end["junit_sha256"] = freeze.sha256(junit_path)
    write(end_path, end)
    receipt["raw_sha256"][packaged.descriptor["junit_xml"]] = freeze.sha256(junit_path)
    receipt["raw_sha256"][packaged.descriptor["command_exit"]] = freeze.sha256(end_path)
    write(packaged.root / packaged.descriptor["path"], receipt)
    with pytest.raises(freeze.Incomplete, match="JUnit"):
        freeze.tag_guard(packaged.root, "outputs/freeze/freeze_receipt.json",
                         verification_receipt=packaged.descriptor["path"])
