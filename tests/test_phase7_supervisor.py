"""
P7.5 runner contracts (scripts/phase7_supervisor.py), on a throw-away git repo:
jobs run from a frozen snapshot of a commit; completion requires exit 0, every
registered output and a passing check, and is recorded with hashes; failures
block dependants; interrupted attempts are kept as evidence; promotion never
overwrites a different file. No study job is run here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import phase7_supervisor as ps  # noqa: E402


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "outputs" / "phase7").mkdir(parents=True)
    (r / "outputs" / "phase7" / ".keep").write_text("")
    (r / ".venv" / "bin").mkdir(parents=True)
    (r / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (r / "job.py").write_text(
        "import sys, pathlib\n"
        "mode = sys.argv[1]\n"
        "out = pathlib.Path('outputs/phase7/result.txt')\n"
        "if mode == 'fail': sys.exit(3)\n"
        "if mode != 'nooutput': out.write_text('VERSION-1\\n')\n")
    (r / ".gitignore").write_text(".venv\n")
    _git(r, "init", "-q")
    _git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return r


def _head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def _sup(repo, jobs, queue="q"):
    commit = _head(repo)
    snap = ps.make_snapshot(repo, commit)
    s = ps.Supervisor.__new__(ps.Supervisor)
    s.repo, s.queue, s.commit, s.snap = repo, queue, commit, snap
    s.spec = {"max_threads": 2, "jobs": jobs}
    s.running, s.stop = {}, False
    return s


def test_snapshot_is_frozen_against_working_tree_edits(repo):
    s = _sup(repo, [])
    (repo / "job.py").write_text("raise SystemExit('edited after launch')\n")
    assert "VERSION-1" in (s.snap / "job.py").read_text()
    rec = json.loads((s.snap / ".snapshot.json").read_text())
    assert rec["commit"] == _head(repo)


def test_complete_requires_exit_zero_outputs_and_records_hashes(repo):
    job = ps.Job("ok", ["job.py", "ok"], ["outputs/phase7/result.txt"])
    s = _sup(repo, [job])
    assert s.run() == 0
    d = ps.job_dir(repo, "q", "ok")
    rec = json.loads((d / "COMPLETE.json").read_text())
    assert rec["exit_code"] == 0
    assert rec["outputs_sha256"]["outputs/phase7/result.txt"] == ps.sha256(repo / "outputs/phase7/result.txt")
    assert json.loads((d / "attempt_1.json").read_text())["commit"] == _head(repo)
    assert json.loads((d / "attempt_1.exit.json").read_text())["exit_code"] == 0
    assert ps.verify_complete(repo, "q", "ok")
    (repo / "outputs/phase7/result.txt").write_text("tampered\n")
    assert not ps.verify_complete(repo, "q", "ok")


def test_failure_blocks_dependants_and_is_not_promoted(repo):
    a = ps.Job("a", ["job.py", "fail"], ["outputs/phase7/result.txt"])
    b = ps.Job("b", ["job.py", "ok"], ["outputs/phase7/result.txt"], requires=[("q", "a")])
    s = _sup(repo, [a, b])
    s.run()
    assert ps.job_state(repo, "q", "a") == "FAILED"
    assert ps.job_state(repo, "q", "b") == "GATE_CLOSED"
    assert not (repo / "outputs/phase7/result.txt").exists()


def test_zero_exit_without_outputs_is_not_complete(repo):
    job = ps.Job("noout", ["job.py", "nooutput"], ["outputs/phase7/result.txt"])
    s = _sup(repo, [job])
    s.run()
    assert ps.job_state(repo, "q", "noout") == "FAILED"
    assert "missing outputs" in json.loads((ps.job_dir(repo, "q", "noout") / "FAILED.json").read_text())["reason"]


def test_completion_check_is_enforced(repo):
    job = ps.Job("chk", ["job.py", "ok"], ["outputs/phase7/result.txt"],
                 check=lambda root: {"ok": False, "detail": "registered check"})
    s = _sup(repo, [job])
    s.run()
    assert ps.job_state(repo, "q", "chk") == "FAILED"


def test_gate_closed_is_recorded_and_job_not_run(repo):
    job = ps.Job("g", ["job.py", "ok"], ["outputs/phase7/result.txt"],
                 gate=lambda r: {"open": False, "reason": "A1 FAIL"})
    s = _sup(repo, [job])
    s.run()
    rec = json.loads((ps.job_dir(repo, "q", "g") / "GATE_CLOSED.json").read_text())
    assert rec["gate"]["reason"] == "A1 FAIL"
    assert not ps.attempts(ps.job_dir(repo, "q", "g"))


def test_interrupted_attempt_is_kept_and_rerun(repo):
    job = ps.Job("int", ["job.py", "ok"], ["outputs/phase7/result.txt"])
    s = _sup(repo, [job])
    d = ps.job_dir(repo, "q", "int")
    d.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (d / "attempt_1.json").write_text(json.dumps({"pid": dead.pid, "supervisor_pid": -1}))
    (d / "attempt_1.log").write_text("partial log\n")
    (d / "lock").write_text("stale\n")
    (s.snap / "outputs/phase7/result.txt").write_text("PARTIAL\n")
    assert s.run() == 0
    assert (d / "attempt_1.interrupted.json").exists()
    assert (d / "attempt_1.log").read_text() == "partial log\n"
    kept = d / "attempt_1_partial" / "outputs__phase7__result.txt"
    assert kept.read_text() == "PARTIAL\n"
    assert json.loads((d / "COMPLETE.json").read_text())["attempt"] == 2
    assert (repo / "outputs/phase7/result.txt").read_text() == "VERSION-1\n"


def test_promotion_never_overwrites_a_different_file(repo):
    (repo / "outputs/phase7/result.txt").write_text("OLD EVIDENCE\n")
    job = ps.Job("p", ["job.py", "ok"], ["outputs/phase7/result.txt"])
    s = _sup(repo, [job])
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        s.run()
    assert (repo / "outputs/phase7/result.txt").read_text() == "OLD EVIDENCE\n"


def test_logs_and_records_are_write_once(tmp_path):
    p = tmp_path / "x.json"
    ps.write_json_new(p, {"a": 1})
    with pytest.raises(FileExistsError):
        ps.write_json_new(p, {"a": 2})


def test_status_file_reports_states_from_records(repo):
    job = ps.Job("ok", ["job.py", "ok"], ["outputs/phase7/result.txt"])
    _sup(repo, [job]).run()
    ps.write_status_md(repo)
    text = (repo / "outputs/phase7/runs/STATUS.md").read_text()
    assert "| q | ok | COMPLETE |" in text
