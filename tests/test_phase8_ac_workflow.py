"""Synthetic record/process tests: no model imports, builds or scientific inputs."""
from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase8 import ac_workflow as ac


AC = "Now drawing from 'AC Power'\n fixture"
BATTERY = "Now drawing from 'Battery Power'\n fixture"


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


def result(root, queue, spec, verdict="PASS", power=AC):
    out = root / queue["out_dir"] / ac.run_name(spec)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {k: spec[k] for k in ("arm", "variant", "workload", "workers")}
    manifest.update(repeats=5, warmups=1, registered_protocol=True)
    put(out / "manifest.json", manifest)
    put(out / "result.json", {**manifest, "verdict": verdict, "timings_s": [1.] * 5,
                              "median_s": 1., "comparisons": [
                                  {"repeat": n, "internal": {"match": True}, "reference": None}
                                  for n in range(6)]})
    (out / "progress.jsonl").write_text("".join(json.dumps({"repeat": n, "power": power}) + "\n"
                                               for n in range(6)))
    return out


def report(root, path, status="PASS"):
    out = (root / path).parent
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "environment.json", "run_log.txt", "report.md", "turbine_score.json",
                 "manufactured_scores.json", "nozzle_scores.json", "turbine_checkpoint.pt"):
        (out / name).write_text("synthetic artifact\n")
    put(root / path, {"status": status, "turbine": {"checkpoint": "turbine_checkpoint.pt"}})
    put(out / "hashes.json", {p.name: ac.sha256_file(p) for p in out.iterdir() if p.name != "hashes.json"})


class FakeProcess:
    def __init__(self, owner, argv, log, fd, handshake):
        self.owner, self.argv, self.log = owner, argv, log
        self.pid = 800000 + len(owner.spawned)
        self.fd = os.dup(fd)
        normalized = [str(owner.root / a) if i == 0 and "/" in a and not os.path.isabs(a) else a
                      for i, a in enumerate(argv)]
        ac.write_once(handshake, {"pid": self.pid, "birth": "fixture birth", "command": normalized})
        owner.spawned.append(self)

    def wait(self):
        msg = os.read(self.fd, 16)
        os.close(self.fd)
        self.owner.messages.append(msg)
        if msg == b"go":
            code = self.owner.execute(self.argv)
            self.log.write("synthetic command exit %s\n" % code)
            self.log.flush()
        else:
            code = ac.ABORTED_EXIT
        self.owner.dead.add(self.pid)
        return code


class Harness:
    def __init__(self, root):
        self.root, self.dead, self.spawned, self.messages = root, set(), [], []
        (root / "cpp/build").mkdir(parents=True)
        (root / "cpp/build/core.so").write_bytes(b"old benchmark binary")
        (root / "bench.py").write_text("# synthetic benchmark\n")
        self.identity = {"git_head": "fixture-head", "tracked_tree_sha256": "a" * 64,
                         "built_modules": {"cpp/build/core.so": ac.sha256_file(root / "cpp/build/core.so")}}
        self.reg = {
            "id": ac.REGISTRATION_ID, "output_root": "operations", "lease_path": "operations/owner.json",
            "power": {"ac_marker": "'AC Power'", "poll_s": 300},
            "identity": {"tracked_paths": ["src"], "built_modules_glob": "cpp/build/*.so"},
            "benchmark": {"script": "bench.py", "script_sha256": ac.sha256_file(root / "bench.py"),
                          "command": ["fake", "benchmark", "{arm}", "{variant}", "{workload}",
                                      "{workers}", "{out_dir}"], "native_flag": "--native",
                          "validation": {"terminal_verdicts": list(ac.ELIGIBLE), "repeats": 5,
                                         "warmups": 1, "timing_samples": 5}},
            "historical_records": {}, "flags": [], "queues": {},
        }
        for name, text in (("run2_ac", "1 a W1 1"), ("run2_ac_rerun", "1 a W2 11")):
            plan = root / f"{name}.txt"; plan.write_text(text + "\n")
            self.reg["queues"][name] = {"plan": plan.name, "plan_sha256": ac.sha256_file(plan),
                                        "n_specs": 1, "out_dir": f"results/{name}"}
        self.reg["stages"] = [
            {"id": "benchmark_run2", "kind": "benchmark_queue", "queue": "run2_ac"},
            {"id": "benchmark_rerun", "kind": "benchmark_queue", "queue": "run2_ac_rerun"},
            {"id": "a2_calibration", "kind": "command", "command": ["fake", "a2"],
             "requires_queue_completions": list(self.reg["queues"]), "requires_benchmark_owner_released": True,
             "must_not_exist": ["a2/fit.json"], "expected_outputs": ["a2/fit.json"],
             "expected_json_keys": ["params", "sse"]},
            {"id": "validation_build", "kind": "command", "command": ["fake", "build"],
             "expected_outputs": ["cpp/build_next/core.so"], "retain_unchanged": ["cpp/build/core.so"]},
            {"id": "track4_focused_pytest", "kind": "command", "command": ["fake", "focused"],
             "requires_stage_pass": "validation_build"},
            {"id": "track4_diagnostic", "kind": "command", "command": ["fake", "diagnostic"],
             "requires_stage_pass": "track4_focused_pytest", "must_not_exist": ["diagnostic"],
             "expected_outputs": ["diagnostic/report.json"], "report": "diagnostic/report.json"},
            {"id": "full_pytest", "kind": "command", "command": ["fake", "full"],
             "requires_stage_pass": "validation_build"},
            {"id": "protected_hashes", "kind": "command", "command": ["fake", "hashes"], "heavy": False},
        ]
        self.write_registration()
        self.wf = self.workflow()
        self.wf.start("fixture")

    def write_registration(self):
        put(self.root / "docs/reg.json", self.reg)

    def workflow(self):
        return ac.Workflow(self.root, "docs/reg.json", power=lambda: AC, birth=self.birth,
                           spawn=lambda argv, log, fd, hs: FakeProcess(self, argv, log, fd, hs),
                           identity=lambda: copy.deepcopy(self.identity), sleep=lambda _: None)

    def birth(self, pid):
        return ac.DEAD if pid in self.dead else "fixture birth"

    def execute(self, argv):
        if argv[1] == "benchmark":
            sp = ac.parse_specs([" ".join(argv[2:6])])[0]
            q = next(q for q in self.reg["queues"].values() if q["out_dir"] == argv[6])
            result(self.root, q, sp)
        elif argv[1] == "a2":
            put(self.root / "a2/fit.json", {"params": {}, "sse": 1})
        elif argv[1] == "build":
            p = self.root / "cpp/build_next/core.so"; p.parent.mkdir(); p.write_bytes(b"new validation binary")
        elif argv[1] == "diagnostic":
            report(self.root, "diagnostic/report.json")
        return 0

    def stage(self, sid):
        return next(s for s in self.reg["stages"] if s["id"] == sid)

    def queues(self):
        for q in self.reg["queues"]:
            self.wf.run_queue(q, q)
        assert self.wf.release_benchmark_owner()["state"] == "RELEASED"


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def test_complete_synthetic_chain_and_terminal_api(h, monkeypatch):
    outcome = h.wf.run_chain()
    assert outcome["status"] == "COMPLETE"
    h.wf.write_session_record(outcome)
    assert h.wf.release("COMPLETE")
    monkeypatch.setattr(ac, "git_identity", lambda *args: copy.deepcopy(h.identity))
    context = ac.validate_terminal_context(h.root, "docs/reg.json", expected_identity=h.identity)
    assert context["lease"] is None and context["chain"]["status"] == "COMPLETE"
    assert context["stages"]["validation_build"]["outputs"]["cpp/build_next/core.so"]
    assert [p.argv[1] for p in h.spawned] == ["benchmark", "benchmark", "a2", "build", "focused", "diagnostic", "full", "hashes"]


@pytest.mark.parametrize("what", ["result", "progress", "log", "handshake", "record", "source", "names", "summary"])
def test_cached_completion_rechecks_all_evidence(h, what):
    h.queues()
    q = "run2_ac"; completion = h.wf.records / f"{q}.completion.json"
    cp = ac.read_json(completion); name = next(iter(cp["spec_records"]))
    rp = h.wf.records / q / f"{name}.json"; rec = ac.read_json(rp)
    if what in ("result", "progress"):
        path = next(p for p in rec["evidence"] if p.endswith(f"{what}.json" if what == "result" else "progress.jsonl"))
        (h.root / path).unlink()
    elif what in ("log", "handshake"):
        (h.root / rec["launch"][what]).unlink()
    elif what == "record": rp.unlink()
    elif what == "source": cp["identity"]["tracked_tree_sha256"] = "b" * 64; put(completion, cp)
    elif what == "names": cp["spec_records"] = {}; put(completion, cp)
    else: cp["states"] = {"FAIL": 1}; put(completion, cp)
    with pytest.raises(ac.Blocked): h.wf.check_queue_completion(q)
    with pytest.raises(ac.Blocked): h.wf.check_owner_release()
    with pytest.raises(ac.Blocked): h.wf.run_command_stage(h.stage("a2_calibration"), {})


@pytest.mark.parametrize("what", ["launch", "source", "malformed_result", "flags", "extra_evidence"])
def test_cached_spec_cannot_forge_terminal_content_or_provenance(h, what):
    h.queues(); q = "run2_ac"; sp = h.wf._plan_specs(q)[0]
    rec = ac.read_json(h.wf.records / q / f"{ac.run_name(sp)}.json")
    if what == "launch": rec["launch"] = None
    elif what == "source": rec["identity"]["tracked_tree_sha256"] = "b" * 64
    elif what == "malformed_result":
        path = next(p for p in rec["evidence"] if p.endswith("result.json"))
        put(h.root / path, {}); rec["evidence"][path] = ac.sha256_file(h.root / path)
    elif what == "flags": rec["provenance_flags"] = ["invented"]
    else: rec["evidence"]["bench.py"] = ac.sha256_file(h.root / "bench.py")
    assert h.wf.spec_record_problems(q, sp, rec)


def test_registered_history_skip_preserves_invalid_power_and_log(h):
    q = "run2_ac"; sp = h.wf._plan_specs(q)[0]; name = ac.run_name(sp)
    out = result(h.root, h.reg["queues"][q], sp, power=BATTERY)
    log = h.root / h.reg["queues"][q]["out_dir"] / "queue_logs/old.log"
    log.parent.mkdir(); log.write_text("old immutable log")
    hist = {n: ac.sha256_file(out / n) for n in ac.RUN_FILES}
    hist["queue_logs/old.log"] = ac.sha256_file(log)
    h.reg["historical_records"] = {q: {name: hist}}
    h.reg["flags"] = [{"record": f"{q}/{name}", "flag": "TIMING_INVALID_POWER"}]
    h.write_registration(); h.wf.reg = copy.deepcopy(h.reg); h.wf.reg_sha = ac.sha256_file(h.root / "docs/reg.json")
    before = {p: ac.sha256_file(p) for p in (*out.iterdir(), log)}
    res = h.wf.run_queue(q, q)
    assert res["state"] == "TERMINAL_COMPLETE_WITH_FLAGS" and not h.spawned
    assert before == {p: ac.sha256_file(p) for p in before}
    rp = h.wf.records / q / f"{name}.json"; rec = ac.read_json(rp)
    assert rec["numerical_reference_valid"] and not rec["timing_valid"]
    rec["evidence"].pop(str(log.relative_to(h.root)))
    assert h.wf.spec_record_problems(q, sp, rec)


@pytest.mark.parametrize("kind", ["no_output", "partial", "malformed_progress"])
def test_refused_partial_runs_never_complete_or_retry(h, kind):
    def execute(argv):
        if kind != "no_output":
            sp = ac.parse_specs([" ".join(argv[2:6])])[0]
            out = result(h.root, h.reg["queues"]["run2_ac"], sp)
            if kind == "partial": (out / "result.json").unlink()
            else: (out / "progress.jsonl").write_text("not JSON")
        return 2
    h.execute = execute
    assert h.wf.run_queue("run2_ac", "run2")["state"] == "BLOCKED"
    assert not (h.wf.records / "run2_ac.completion.json").exists()
    count = len(h.spawned)
    assert h.wf.run_queue("run2_ac", "run2")["state"] == "BLOCKED"
    assert len(h.spawned) == count


def test_occupied_partial_directory_blocks_without_launch(h):
    q = h.reg["queues"]["run2_ac"]; sp = h.wf._plan_specs("run2_ac")[0]
    out = result(h.root, q, sp); (out / "result.json").unlink()
    assert h.wf.run_queue("run2_ac", "run2")["state"] == "BLOCKED"
    assert not h.spawned


@pytest.mark.parametrize("what", ["missing_output", "missing_log", "report_fail", "report_missing_hash", "retained_missing"])
def test_fresh_stage_missing_evidence_cannot_pass(h, what):
    if what == "retained_missing":
        (h.root / "cpp/build/core.so").unlink()
        with pytest.raises(ac.Blocked): h.wf.run_command_stage(h.stage("validation_build"), {})
        assert not h.spawned; return
    st = {"id": "probe", "kind": "command", "command": ["fake", "probe"]}
    if what == "missing_output": st["expected_outputs"] = ["missing.json"]
    if what.startswith("report"):
        st.update(expected_outputs=["probe/report.json"], report="probe/report.json")
        report(h.root, "probe/report.json", "FAIL" if what == "report_fail" else "PASS")
        if what == "report_missing_hash":
            (h.root / "probe/hashes.json").unlink()
    if what == "missing_log":
        original = h.wf.launch
        def launch(*args, **kwargs):
            rec = original(*args, **kwargs); (h.root / rec["log"]).unlink(); rec["log_sha256"] = None; return rec
        h.wf.launch = launch
    try: res = h.wf.run_command_stage(st, {})
    except ac.Blocked: res = {"state": "BLOCKED"}
    assert res["state"] != "PASS"


@pytest.mark.parametrize("what", ["null_output", "null_log", "failed_report", "missing_checkpoint", "json_keys"])
def test_cached_pass_requires_nonnull_current_evidence_and_content(h, what):
    h.queues(); h.wf.run_command_stage(h.stage("a2_calibration"), {})
    h.wf.run_command_stage(h.stage("validation_build"), {})
    h.wf.run_command_stage(h.stage("track4_focused_pytest"), {})
    sid = "track4_diagnostic" if what in ("failed_report", "missing_checkpoint") else "a2_calibration"
    if sid == "track4_diagnostic": h.wf.run_command_stage(h.stage(sid), {})
    st = h.stage(sid); path = h.wf.records / "stages" / f"{sid}.json"; rec = ac.read_json(path)
    if what == "null_output":
        p = next(iter(rec["outputs"])); (h.root / p).unlink(); rec["outputs"][p] = None
    elif what == "null_log":
        (h.root / rec["launch"]["log"]).unlink(); rec["launch"]["log_sha256"] = None
    elif what == "json_keys":
        p = next(iter(rec["outputs"])); put(h.root / p, {}); rec["outputs"][p] = ac.sha256_file(h.root / p)
    else:
        if what == "failed_report":
            put(h.root / st["report"], {"status": "FAIL"})
        else: (h.root / "diagnostic/turbine_checkpoint.pt").unlink()
        out = (h.root / st["report"]).parent
        put(out / "hashes.json", {p.name: ac.sha256_file(p) for p in out.iterdir() if p.name != "hashes.json"})
        rec["outputs"][st["report"]] = ac.sha256_file(h.root / st["report"])
        rec["report"] = h.wf._report_evidence(st)
    assert h.wf.stage_record_problems(st, rec)


def test_cached_diagnostic_dependency_rechecks_build_and_focused(h):
    h.wf.run_command_stage(h.stage("validation_build"), {})
    h.wf.run_command_stage(h.stage("track4_focused_pytest"), {})
    h.wf.run_command_stage(h.stage("track4_diagnostic"), {})
    (h.root / "cpp/build_next/core.so").unlink()
    count = len(h.spawned)
    with pytest.raises(ac.Blocked): h.wf.run_command_stage(h.stage("track4_diagnostic"), {})
    assert len(h.spawned) == count


@pytest.mark.parametrize("point", ["after_wait", "before_go", "during_command"])
def test_validation_binary_is_rechecked_after_wait_and_before_go(h, point):
    h.wf.run_command_stage(h.stage("validation_build"), {})
    binary = h.root / "cpp/build_next/core.so"
    original = h.execute
    if point == "after_wait":
        readings = iter([BATTERY, AC])
        h.wf.power = lambda: next(readings)
        h.wf.sleep = lambda _: binary.write_bytes(b"changed during AC wait")
    elif point == "before_go":
        original_spawn = h.wf.spawn
        def spawn(*args):
            proc = original_spawn(*args); binary.write_bytes(b"changed after preflight"); return proc
        h.wf.spawn = spawn
    else:
        def execute(argv):
            binary.write_bytes(b"changed during child execution"); return original(argv)
        h.execute = execute
    try: state = h.wf.run_command_stage(h.stage("full_pytest"), {})["state"]
    except ac.Blocked: state = "BLOCKED"
    assert state != "PASS"
    if point == "after_wait": assert len(h.spawned) == 1
    if point == "before_go": assert h.messages[-1] == b"abort"


@pytest.mark.parametrize("end", ["drift", "unreadable"])
def test_end_identity_failure_is_recorded_without_retry(h, end):
    def execute(argv):
        if end == "drift": h.identity["tracked_tree_sha256"] = "b" * 64
        else: h.wf.identity = lambda: (_ for _ in ()).throw(ac.Blocked("unreadable"))
        return 0
    h.execute = execute
    st = {"id": "probe", "kind": "command", "command": ["fake", "probe"]}
    assert h.wf.run_command_stage(st, {})["state"] == "FAIL"
    rec = ac.read_json(h.wf.records / "stages/probe.json")
    assert rec["identity"] == h.wf.start_identity and rec["launch"]["identity_problems"]
    assert (rec["launch"]["end_identity"] is None) == (end == "unreadable")


def test_ac_recheck_after_wait_and_before_go(h):
    readings = iter([BATTERY, AC, AC, BATTERY, AC, AC, AC])
    h.wf.power = lambda: next(readings); sleeps = []; h.wf.sleep = sleeps.append
    launch = h.wf.launch("probe", "probe", ["fake", "probe"])
    assert h.messages == [b"abort", b"go"] and sleeps == [300]
    assert launch["waited"] and launch["went"]


def test_unreadable_power_waits_without_spawning(h):
    h.wf.power = lambda: None
    h.wf.sleep = lambda _: (_ for _ in ()).throw(ac.Blocked("stop synthetic wait"))
    with pytest.raises(ac.Blocked): h.wf.launch("probe", "probe", ["fake", "probe"])
    assert not h.spawned and h.wf.lease["state"] == "WAITING_FOR_POWER_READING"


def test_spawn_exception_keeps_ambiguous_lease(h):
    h.wf.spawn = lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic spawn boundary"))
    with pytest.raises(RuntimeError): h.wf.launch("probe", "probe", ["fake", "probe"])
    assert not h.wf.release("ABORTED")
    assert ac.read_json(h.wf.lease_path)["state"] == "AMBIGUOUS_CHILD"
    with pytest.raises(ac.Refused): h.wf.recover_stale("live owner")


def test_signal_between_spawn_and_record_keeps_launcher_owned(h):
    original = h.wf.spawn
    def spawn(*a):
        original(*a); raise SystemExit(143)
    h.wf.spawn = spawn
    with pytest.raises(SystemExit): h.wf.launch("probe", "probe", ["fake", "probe"])
    assert not h.wf.release("ABORTED") and h.wf._child_state(h.wf.lease) == "alive"
    # fake child receives EOF, exactly as a real launcher would after the driver's exit
    assert h.spawned[0].wait() == ac.ABORTED_EXIT


def test_duplicate_stopped_owner_and_serialized_stale_recovery(h):
    with pytest.raises(ac.Refused): h.workflow().start("duplicate")
    with pytest.raises(ac.Refused): h.wf.recover_stale("stopped owner is alive")
    other = h.workflow(); other.birth = lambda _: ac.DEAD
    fd = os.open(other.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ac.Refused): other.recover_stale("concurrent")
        assert other.lease_path.exists()
    finally: os.close(fd)
    path = other.recover_stale("provably stale")
    assert not other.lease_path.exists()
    assert ac.read_json(h.root / path)["inferred_completion"] is None


def test_reused_pid_is_stale_but_never_completion():
    assert ac.liveness(12, "old", lambda _: "new") == "reused"
    assert ac.liveness(12, None, lambda _: ac.DEAD) == "unknown"


def test_scientific_callers_can_catch_runtime_gate_errors():
    assert issubclass(ac.Blocked, RuntimeError)
    assert issubclass(ac.Refused, RuntimeError)


def test_script_hash_blocks_before_any_queue_work(h):
    (h.root / "bench.py").write_text("changed")
    with pytest.raises(ac.Blocked): h.wf.run_queue("run2_ac", "probe")
    assert not h.spawned


def test_focused_failure_blocks_diagnostic_and_independent_stages_continue(h):
    original = h.execute
    h.execute = lambda argv: 1 if argv[1] == "focused" else original(argv)
    outcome = h.wf.run_chain()
    assert outcome["status"] == "STOPPED_WITH_BLOCKERS"
    assert outcome["stages"]["track4_focused_pytest"]["state"] == "FAIL"
    assert outcome["stages"]["track4_diagnostic"]["state"] == "BLOCKED"
    assert outcome["stages"]["full_pytest"]["state"] == "PASS"
    assert outcome["stages"]["protected_hashes"]["state"] == "PASS"
    assert not any(p.argv[1] == "diagnostic" for p in h.spawned)


def test_truthful_failed_stage_with_missing_outputs_is_terminal_not_pass(h, monkeypatch):
    original = h.execute
    h.execute = lambda argv: 1 if argv[1] == "a2" else original(argv)
    outcome = h.wf.run_chain()
    assert outcome["status"] == "FINISHED_WITH_FLAGS_OR_FAILURES"
    h.wf.write_session_record(outcome); h.wf.release(outcome["status"])
    monkeypatch.setattr(ac, "git_identity", lambda *args: copy.deepcopy(h.identity))
    context = ac.validate_terminal_context(h.root, "docs/reg.json")
    assert context["stages"]["a2_calibration"]["state"] == "REFUSED"
    assert context["stages"]["a2_calibration"]["outputs"] == {"a2/fit.json": None}


def test_active_full_pytest_requires_exact_live_child_and_build_evidence(h, monkeypatch):
    h.queues()
    for sid in ("a2_calibration", "validation_build", "track4_focused_pytest", "track4_diagnostic"):
        h.wf.run_command_stage(h.stage(sid), {})
    hs = h.wf.session_dir / "handshakes/full.json"; log = h.wf.session_dir / "logs/full.log"
    put(hs, {"pid": os.getpid(), "birth": "fixture birth", "command": h.stage("full_pytest")["command"]})
    log.write_text("synthetic active log")
    child = {"pid": os.getpid(), "birth": "fixture birth", "argv": h.stage("full_pytest")["command"],
             "handshake": h.wf.rel(hs), "log": h.wf.rel(log)}
    h.wf.update_lease(stage="full_pytest", state="RUNNING", child=child)
    lease = ac.read_json(h.wf.lease_path); lease["owner_pid"] = 900000; put(h.wf.lease_path, lease)
    monkeypatch.setattr(ac, "git_identity", lambda *a: copy.deepcopy(h.identity))
    monkeypatch.setattr(ac, "process_birth", lambda _: "fixture birth")
    # Workflow's default argument retains the original function; inject birth at the constructor boundary.
    original = ac.Workflow
    def workflow(*args, **kwargs):
        kwargs["birth"] = lambda _: "fixture birth"; return original(*args, **kwargs)
    monkeypatch.setattr(ac, "Workflow", workflow)
    with pytest.raises(ac.Blocked, match="require_idle"):
        ac.validate_terminal_context(h.root, "docs/reg.json", require_idle=True, allow_active_stage="full_pytest")
    context = ac.validate_terminal_context(h.root, "docs/reg.json", require_idle=False, allow_active_stage="full_pytest")
    assert context["chain"] is None and context["lease"]["child"]["pid"] == os.getpid()
    child["pid"] += 1; lease["child"] = child; put(h.wf.lease_path, lease)
    with pytest.raises(ac.Blocked):
        ac.validate_terminal_context(h.root, "docs/reg.json", require_idle=False, allow_active_stage="full_pytest")


def test_terminal_context_rejects_forged_chain_and_expected_sources(h, monkeypatch):
    outcome = h.wf.run_chain(); path = h.root / h.wf.write_session_record(outcome)
    h.wf.release("COMPLETE")
    monkeypatch.setattr(ac, "git_identity", lambda *a: copy.deepcopy(h.identity))
    wrong = copy.deepcopy(h.identity); wrong["built_modules"] = {}
    with pytest.raises(ac.Blocked): ac.validate_terminal_context(h.root, "docs/reg.json", expected_identity=wrong)
    chain = ac.read_json(path); chain["stages"]["full_pytest"]["record_sha256"] = "0" * 64; put(path, chain)
    with pytest.raises(ac.Blocked): ac.validate_terminal_context(h.root, "docs/reg.json")


def test_finalization_failure_returns_nonzero(monkeypatch):
    class Workflow:
        def __init__(self, **kw): pass
        def start(self, *a): pass
        def run_chain(self): return {"status": "COMPLETE"}
        def write_session_record(self, *a): raise OSError("synthetic record failure")
    monkeypatch.setattr(ac, "Workflow", Workflow)
    assert ac.main(["chain"]) == 3


def test_wrapper_preserves_arguments_without_starting_python(tmp_path):
    root = tmp_path; (root / "scripts/phase8").mkdir(parents=True); (root / ".venv/bin").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/phase8/run_benchmark_queue.sh"
    wrapper = root / "scripts/phase8/run_benchmark_queue.sh"; wrapper.write_text(source.read_text())
    python = root / ".venv/bin/python"; python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n"); python.chmod(0o755)
    p = subprocess.run(["bash", str(wrapper), "results/path with space", "1 a W1 1", "3 a W2 11 native"],
                       cwd=root, capture_output=True, text=True, check=True)
    assert p.stdout.splitlines() == [str(root / "scripts/phase8/ac_workflow.py"), "queue", "--registration",
                                    ac.REGISTRATION, "--out-dir", "results/path with space", "--",
                                    "1 a W1 1", "3 a W2 11 native"]
