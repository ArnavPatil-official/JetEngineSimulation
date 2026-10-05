"""Manufactured PC stage/figure fixtures; no project labels or C++ imports."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.phase8 import pc_finish as finish
from scripts.phase8 import pc_runtime as runtime
from scripts.phase8 import freeze_screening as figures
from scripts.phase8.nozzle_ode import run as nozzle


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_nozzle_missing_properties_is_incomplete_before_ownership(tmp_path):
    put(tmp_path / finish.NOZZLE_REGISTRATION, {"dependencies":{
        "product_rows":"outputs/train.csv", "product_species":"outputs/species.npz",
        "named_properties":"outputs/named.csv", "property_output_manifest":"outputs/properties.json"}})
    result = finish.run_nozzle(tmp_path, lambda *a, **k: pytest.fail("missing properties cannot acquire ownership"))
    assert result["status"] == "INCOMPLETE" and len(result["missing"]) == 4
    assert not (tmp_path / "outputs").exists()


def test_nozzle_dispatches_existing_study_and_never_a2(tmp_path, monkeypatch):
    names = {key:"outputs/"+key for key in
             ("product_rows", "product_species", "named_properties", "property_output_manifest")}
    for name in names.values():
        put(tmp_path / name, {})
    output = "outputs/nozzle"
    put(tmp_path / finish.NOZZLE_REGISTRATION, {"dependencies":names, "outputs":{"root":output}})
    calls = []
    factory = object()
    def existing_study(root, **kwargs):
        calls.append((root, kwargs))
        put(root / output / "terminal.json", {"status":"FAIL", "execution_complete":True,
                                             "scientific_verdict":"FAIL"})
        put(root / output / "report.json", {"registered_status":"INCOMPLETE",
                                             "original_shock_oracle_status":"INCOMPLETE"})
    monkeypatch.setattr(nozzle, "run", existing_study)
    result = finish.run_nozzle(tmp_path, factory)
    assert len(calls) == 1
    assert calls[0][1] == {"context_factory":factory, "source_loader":finish._source_properties_pc, "portable":True}
    assert result["status"] == "FAIL" and result["execution_complete"] is True
    assert result["registered_status"] == "INCOMPLETE"


def test_missing_original_track4_is_never_recreated_or_promoted(tmp_path):
    reg = {"dependencies":{"old_registration":"docs/track4.json"}}
    put(tmp_path / "docs/track4.json", {"output_dir":"outputs/old_attempt"})
    historical, hashes = nozzle.inherited_evidence(tmp_path, reg, portable=True)
    assert historical["status"] == "INCOMPLETE" and historical["rungs"] == {} and hashes == {}
    assert not (tmp_path / "outputs/old_attempt").exists()
    with pytest.raises(FileNotFoundError):
        nozzle.inherited_evidence(tmp_path, reg)


def test_invalid_original_track4_is_not_hidden_as_missing(tmp_path, monkeypatch):
    def invalid(*args):
        raise nozzle.Blocked("changed historical hash")
    monkeypatch.setattr(nozzle, "old_evidence", invalid)
    with pytest.raises(nozzle.Blocked, match="changed historical hash"):
        nozzle.inherited_evidence(tmp_path, {}, portable=True)


def test_junit_requires_actual_complete_cases(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"><testcase/></testsuite></testsuites>')
    with pytest.raises(finish.FinishError, match="coverage"):
        finish._junit(path)
    path.write_text('<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0"><testcase><failure/></testcase></testsuite></testsuites>')
    assert finish._junit(path)["failures"] == 1


@pytest.fixture
def local(tmp_path, monkeypatch):
    root = tmp_path
    put(root / finish.PRODUCT_REGISTRATION, {"outputs":{"root":"outputs/freeze", "local_tag":"freeze-2026-10-18"}})
    context = SimpleNamespace(root=root, identity={"registration_sha256":"registered", "binary_sha256":"core"},
                              binary_sha256="core", original_context={"g0":{}})
    state = SimpleNamespace(released=[], runs=[], checks=0)
    class Run:
        def __init__(self, output):
            self.out = root / output
            self.out.mkdir(parents=True, exist_ok=False)
            put(self.out / "reservation.json", {"identity":context.identity})
            state.runs.append(self)
        def assert_current(self):
            state.checks += 1
        def record_children(self, children):
            self.children = children
        def release(self, terminal):
            terminal = copy.deepcopy(terminal)
            state.released.append(terminal)
            put(self.out / "terminal.json", terminal)
            return terminal
    context.acquire_run = lambda output, *a, **k: Run(output)
    factory = lambda *a, **k: context
    return root, context, state, factory


def fake_pytest(root, monkeypatch, *, code=0, fail=0):
    for name in ("tests/test_phase8_screening_product.py", "tests/test_phase8_pc_finish.py"):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# manufactured test identity\n")
    calls = []
    class Child:
        pid = 999991
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            self.path = Path(argv[-1].split("=",1)[1])
        def wait(self):
            self.path.write_text(f'<testsuites><testsuite tests="1" failures="{fail}" errors="0" skipped="0"><testcase/></testsuite></testsuites>')
            return code
        def poll(self):
            return code
    monkeypatch.setattr(finish.subprocess, "Popen", Child)
    monkeypatch.setattr(runtime, "process_birth", lambda pid:"manufactured birth")
    return calls


def test_product_waits_for_actual_check_child_and_keeps_scientific_fail_refusal(local, monkeypatch):
    root, context, state, factory = local
    put(root/finish.SAF_OUTPUT/"product.json", {})
    monkeypatch.setattr(finish, "_validate", lambda *a, **k:{"status":"FAIL"})
    from scripts.phase8 import screening_product
    monkeypatch.setattr(screening_product, "screen_blends", lambda *a, **k:(_ for _ in ()).throw(ValueError("no deployment receipt")))
    calls = fake_pytest(root, monkeypatch)
    result = finish.check_product(root, factory)
    assert result["status"] == "PASS" and result["execution_complete"] is True
    proof = runtime.read_json(root/finish.CHECK_OUTPUT/"command.exit.json")
    assert proof["waited"] is True and proof["exit_code"] == 0
    smoke = runtime.read_json(root/finish.CHECK_OUTPUT/"product_checks.json")["smoke"]
    assert smoke["deployment_ready"] is False
    assert calls[0][1:3] == ["-m", "pytest"] and not any("A2" in item for item in calls[0])


def test_product_cannot_accept_failed_study_by_default(local, monkeypatch):
    root, _, state, factory = local
    put(root/finish.SAF_OUTPUT/"product.json", {})
    monkeypatch.setattr(finish, "_validate", lambda *a, **k:{"status":"FAIL"})
    from scripts.phase8 import screening_product
    monkeypatch.setattr(screening_product, "screen_blends", lambda *a, **k:{})
    with pytest.raises(finish.FinishError, match="accepted"):
        finish.check_product(root, factory)
    assert state.released[-1]["status"] == "ERROR"


def test_product_test_failure_stays_failure(local, monkeypatch):
    root, _, _, factory = local
    put(root/finish.SAF_OUTPUT/"product.json", {})
    monkeypatch.setattr(finish, "_validate", lambda *a, **k:{"status":"FAIL"})
    from scripts.phase8 import screening_product
    monkeypatch.setattr(screening_product, "screen_blends", lambda *a, **k:(_ for _ in ()).throw(ValueError("unavailable")))
    fake_pytest(root, monkeypatch, code=1, fail=1)
    result = finish.check_product(root, factory)
    assert result["status"] == "FAIL" and result["exit_code"] == 1


def test_missing_product_returns_real_incomplete(local, monkeypatch):
    root, _, _, factory = local
    monkeypatch.setattr(finish.subprocess, "Popen", lambda *a, **k:pytest.fail("no Product to check"))
    result = finish.check_product(root, factory)
    assert result["status"] == "INCOMPLETE" and result["execution_complete"] is False


@pytest.fixture
def plotted(local, monkeypatch):
    root, context, state, factory = local
    rt = runtime
    context.identity["g0_record_path"] = "outputs/pc/g0/g0_parity.json"
    context.original_context["g0"] = {"verdict":"PASS", "backends":{}}
    put(root/context.identity["g0_record_path"], context.original_context["g0"])
    roles = []
    for role in figures.REQUIRED_ROLES:
        name = figures.INPUT_NAMES[role]
        path = "outputs/scored/"+name
        put(root/path, {})
        roles.append({"role":role, "path":path, "format":Path(name).suffix[1:],
                      "producer_registration":finish.SAF_REGISTRATION, "producer_output_dir":"outputs/scored"})
    reg = rt.read_json(root/finish.PRODUCT_REGISTRATION)
    reg["published_inputs"] = roles
    put(root/finish.PRODUCT_REGISTRATION, reg)
    def validate(repo, registration, output, **kwargs):
        return {"status":"PASS", "artifact_hashes":{name:rt.sha256_file(repo/name) for name in kwargs["artifact_paths"]}}
    monkeypatch.setattr(finish, "_validate", validate)
    return root, context, state, factory, reg


def test_published_adapter_uses_actual_pc_g0_and_detects_drift(plotted):
    root, context, _, _, reg = plotted
    data = finish._PublishedPC(context, reg)
    assert data.descriptors["G0"]["path"] == context.identity["g0_record_path"]
    assert "outputs/phase8/g0_rerun_20261003/g0_parity.json" not in data.hashes
    path = root/data.descriptors["surrogate_metrics"]["path"]
    put(path, {"changed":True})
    with pytest.raises(finish.FinishError, match="changed"):
        data.assert_current()


def test_published_adapter_never_opens_sealed_labels(plotted):
    _, context, _, _, reg = plotted
    next(entry for entry in reg["published_inputs"] if entry["role"]=="surrogate_metrics")["path"] = "outputs/sealed/test_metrics.json"
    with pytest.raises(figures.Incomplete, match="declared"):
        finish._PublishedPC(context, reg)


def fake_draws(monkeypatch):
    class Figure:
        def suptitle(self, *a, **k): pass
        def tight_layout(self, *a, **k): pass
        def savefig(self, stream, **kwargs): stream.write(b"manufactured figure\n")
    pyplot = SimpleNamespace(close=lambda *a:None)
    matplotlib = SimpleNamespace(use=lambda *a:None, rcParams={}, pyplot=pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    def draw(data, plt, rows):
        figures.metric(rows, "fixture", "case", "published", 7.0, "ratio", "PASS")
        return Figure()
    monkeypatch.setattr(figures, "DRAW", {prefix:draw for prefix in figures.PREFIXES})


def test_freeze_exports_actual_values_and_figures_but_missing_track4_stays_incomplete(plotted, monkeypatch):
    root, context, _, factory, reg = plotted
    report = next(entry for entry in reg["published_inputs"] if entry["role"]=="nozzle_report")
    put(root/report["path"], {"original_shock_oracle_status":"INCOMPLETE"})
    put(root/finish.CHECK_OUTPUT/"terminal.json", {"status":"PASS"})
    for name in ("product_checks.json", "command_spec.json", "command.exit.json", "pytest.log", "junit.xml"):
        put(root/finish.CHECK_OUTPUT/name, {})
    fake_draws(monkeypatch)
    result = finish.freeze(root, factory)
    assert result["status"] == "INCOMPLETE" and result["tag_created"] is False
    out = root/"outputs/freeze"
    assert (out/"NUMBERS.md").is_file()
    assert all((out/(prefix+"."+ext)).is_file() for prefix in figures.PREFIXES for ext in ("png", "pdf"))
    receipt = runtime.read_json(out/"freeze_receipt.json")
    assert receipt["inherited_prerequisites"]["original_track4"] == "INCOMPLETE"
    assert receipt["numeric_rows"] == 8


def test_freeze_missing_published_inputs_never_invents_numbers(local):
    root, _, _, factory = local
    reg = runtime.read_json(root/finish.PRODUCT_REGISTRATION)
    reg["published_inputs"] = []
    put(root/finish.PRODUCT_REGISTRATION, reg)
    result = finish.freeze(root, factory)
    assert result["status"] == "INCOMPLETE"
    assert not (root/"outputs/freeze/NUMBERS.md").exists()


def test_freeze_source_error_raises_and_records_error(local, monkeypatch):
    root, _, state, factory = local
    monkeypatch.setattr(finish, "_PublishedPC", lambda *a:(_ for _ in ()).throw(finish.FinishError("core drift")))
    with pytest.raises(finish.FinishError, match="core drift"):
        finish.freeze(root, factory)
    assert state.released[-1]["status"] == "ERROR"
