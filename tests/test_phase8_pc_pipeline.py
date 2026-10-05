"""Manufactured PC dispatch checks; never train or read project labels."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import pc_pipeline as pc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    registrations = {
        "p73_a1": {"outputs": {"directory": "outputs/p73"}},
        "saf": {"artifact_root": "outputs/saf"},
        "nozzle": {"outputs": {"root": "outputs/nozzle"}},
        "freeze": {"outputs": {"root": "outputs/freeze"}},
    }
    for key, value in registrations.items():
        (tmp_path / pc.REGISTRATIONS[key]).write_text(json.dumps(value))
    return pc.Config(root=tmp_path)


def handlers(calls, replacements=None):
    overrides = replacements or {}
    def step(name):
        calls.append(name)
        value = overrides.get(name, {"state": "PASS"})
        if isinstance(value, BaseException):
            raise value
        return value
    return {name: lambda name=name: step(name) for name in pc.STAGES}


def test_help_and_dry_run_import_no_scientific_libraries():
    code = """import sys,runpy
sys.argv=['pc_pipeline.py','--dry-run']
try: runpy.run_path('pc_pipeline.py',run_name='__main__')
except SystemExit as e: assert e.code==0
assert not any(x in sys.modules for x in ['torch','numpy','cantera','mlx.core'])
"""
    before = (ROOT / "outputs/phase8/pc_pipeline").exists()
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (ROOT / "outputs/phase8/pc_pipeline").exists() == before
    help_result = subprocess.run([sys.executable, "pc_pipeline.py", "--help"], cwd=ROOT,
                                 capture_output=True, text=True)
    assert help_result.returncode == 0 and "--preflight" in help_result.stdout


def test_dry_run_has_exact_sequence_and_no_mutation(config):
    before = sorted(p.relative_to(config.root) for p in config.root.rglob("*"))
    plan = pc.describe(config)
    assert plan["stages"] == ["preflight", "build", "focused_checks", "g0", "p73_a1",
                              "saf", "nozzle", "product", "freeze"]
    assert plan["a2"] == "DEFERRED"
    assert before == sorted(p.relative_to(config.root) for p in config.root.rglob("*"))


def test_dispatches_every_stage_synchronously(config):
    calls = []
    result = pc.execute(config, handlers=handlers(calls))
    assert calls == list(pc.STAGES)
    assert result["state"] == "PASS"
    terminal = pc.read_json(config.paths()[3] / "terminal.json")
    assert [r["stage"] for r in terminal["stages"]] == calls


def test_completed_scientific_failure_keeps_later_stages(config):
    calls = []
    result = pc.execute(config, handlers=handlers(calls, {
        "saf": {"state": "FAIL", "execution_complete": True, "fidelity_pass": False},
        "nozzle": {"state": "INCOMPLETE", "missing": ["historical_track4"]}}))
    assert calls == list(pc.STAGES)
    assert result["state"] == "INCOMPLETE"
    assert result["stages"][5]["fidelity_pass"] is False


@pytest.mark.parametrize("stage", ["g0", "p73_a1"])
def test_failed_prerequisite_stops_later_science(config, stage):
    calls = []
    result = pc.execute(config, handlers=handlers(calls, {stage: {"state": "FAIL"}}))
    assert calls == list(pc.STAGES[:pc.STAGES.index(stage)+1])
    assert result["state"] == "FAIL" and result["stopped_after"] == stage


@pytest.mark.parametrize("failure", [RuntimeError("child exited 7"), {"state": "ERROR", "exit_code": 7}])
def test_execution_error_stops_and_retains_actual_stage(config, failure):
    calls = []
    result = pc.execute(config, handlers=handlers(calls, {"saf": failure}))
    assert calls[-1] == "saf" and "nozzle" not in calls
    assert result["state"] == "ERROR" and result["stopped_at"] == "saf"
    assert "7" in result["error"]


def test_preflight_failure_creates_no_attempt(config):
    result = pc.execute(config, handlers=handlers([], {"preflight": RuntimeError("missing CUDA")}))
    assert result["state"] == "ERROR" and not config.paths()[3].exists()


def test_repeated_metadata_attempt_refuses_overwrite(config):
    first = pc.execute(config, handlers=handlers([]))
    terminal = config.paths()[3] / "terminal.json"
    frozen = terminal.read_bytes()
    calls = []
    second = pc.execute(config, handlers=handlers(calls))
    assert first["state"] == "PASS" and second["state"] == "ERROR"
    assert calls == ["preflight"] and terminal.read_bytes() == frozen


@pytest.mark.parametrize("name", [".", "build", "build_next", "build/subdir", "build_next/subdir"])
def test_original_build_tree_is_refused(config, name):
    with pytest.raises(ValueError, match="separate"):
        pc.describe(pc.Config(root=config.root, build_dir=config.root / "cpp" / name))


def test_metadata_cannot_overlap_producer(config):
    (config.root / pc.REGISTRATIONS["saf"]).write_text(json.dumps({"artifact_root": "outputs/phase8/saf"}))
    with pytest.raises(ValueError, match="overlaps"):
        pc.describe(pc.Config(root=config.root, metadata_dir=config.root / "outputs/phase8/saf/logs"))


def test_checked_command_preserves_failure_log(tmp_path):
    log = tmp_path / "child.log"
    with pytest.raises(RuntimeError, match="exited 7"):
        pc.checked_command([sys.executable, "-c", "print('actual failure');raise SystemExit(7)"],
                           log, cwd=tmp_path)
    assert "actual failure" in log.read_text()


def test_stage_methods_call_actual_helpers(monkeypatch, config):
    from scripts.phase8 import p73_a1_cpp, pc_saf
    workflow = pc.Workflow(config)
    called = []
    monkeypatch.setattr(p73_a1_cpp, "execute", lambda *a, **k: called.append(("p73", a, k)) or {"status": "COMPLETE"})
    monkeypatch.setattr(pc_saf, "execute", lambda *a, **k: called.append(("saf", a, k)) or {"state": "PASS"})
    workflow.p73_a1()
    workflow.saf()
    assert [r[0] for r in called] == ["p73", "saf"]
    assert called[0][2]["gate_factory"].__self__ is workflow
    assert called[1][2] == {"device": "auto", "backend": "torch"}
