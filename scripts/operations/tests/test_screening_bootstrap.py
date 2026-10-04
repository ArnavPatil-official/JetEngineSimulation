"""Pure operational controls. No scientific imports, labels or native jobs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]/"screening_bootstrap.py"
spec = importlib.util.spec_from_file_location("bootstrap_fixture_subject", SOURCE)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def config():
    # The production config will become ARMED. Fixtures remain deterministic.
    return {"schema_version":1,"registration_id":"P8-SCREENING-BOOTSTRAP-20261004","state":"PENDING_ROOT_IDENTITIES",
        "main_root":"/Users/arnavpatil/Documents/JetEngineSimulation",
        "output_dir":"outputs/phase8/screening_operations/bootstrap_20261004",
        "scientific_lease_path":"outputs/phase8/screening_operations/owner.lease.json","wait_poll_seconds":15,
        "stage_order":["focused_tests","g0","p73","surrogate_checks","surrogate","nozzle_checks","nozzle","product_tests","freeze"],
        "tag":{"name":"freeze-2026-10-18","authorized":False},
        "original":{"registration":"docs/phase8_queue_recovery_registration.json","operations_registration":"docs/phase8_screening_operations_registration.json",
            "session":"20261004T022331Z-22543","chain_path":"outputs/phase8/operations/20261003_recovery/records/chain.20261004T022331Z-22543.json",
            "lease_path":"outputs/phase8/operations/20261003_recovery/owner.lease.json","owner_pid":22543,
            "owner_birth":"Sat Oct 3 22:23:31 2026","launch_head":"8afb0e0f5c225396f2f486b6083bdf09c59a353b"},
        "provider":None,"sources":[],"stages":{},"driver_sha256":None,"governing_files":{},"tag_guard":None}


def fake_terminal(root, *, failed=False):
    out = root/"outputs/example"
    out.mkdir(parents=True)
    identity = {"source":"fixed","core":"fixed","g0":"fixed"}
    reservation = {"identity":identity,"owner_pid":123,"owner_birth":"fixed birth"}
    bootstrap.write_once(out/"reservation.json",reservation)
    bootstrap.write_once(out/"metrics.json",{"value":.1})
    name = "outputs/example/metrics.json"
    terminal = {"identity":identity,"status":"FAIL" if failed else "PASS","exit_code":1 if failed else 0,
        "outputs_complete":not failed,"execution_complete":True,"scientific_verdict":"FAIL" if failed else "PASS",
        "errors":[],"reservation_sha256":bootstrap.digest(out/"reservation.json"),
        "expected_outputs":[name],"artifact_hashes":{name:bootstrap.digest(root/name)}}
    bootstrap.write_once(out/"terminal.json",terminal)
    released = {**reservation,"state":"RELEASED","reservation_sha256":terminal["reservation_sha256"],
        "terminal_sha256":bootstrap.digest(out/"terminal.json")}
    bootstrap.write_once(out/"released_lease.json",released)
    stage = {"output_dir":"outputs/example","artifact_roots":["outputs/example"],"allow_scientific_fail":failed}
    return stage, identity, out


def replace(path, doc):
    path.write_text(json.dumps(doc))


def repair_release(out):
    release=bootstrap.read(out/"released_lease.json")
    release["terminal_sha256"]=bootstrap.digest(out/"terminal.json")
    replace(out/"released_lease.json",release)


def fake_bootstrap_owner(root,birth):
    bootstrap.write_once(root/"outputs/phase8/screening_operations/bootstrap_20261004/owner.lease.json",
        {"owner_pid":bootstrap.os.getpid(),"owner_birth":birth,"driver_sha256":bootstrap.digest(SOURCE),"config_sha256":"0"*64})


def test_unarmed_validation_has_no_execution_authorization(tmp_path):
    result=bootstrap.validate_config(tmp_path,config())
    assert result["execution_authorized"] is False and "provider" in result["pending"]


@pytest.mark.parametrize("key,value",[("session","foreign"),("owner_pid",1),("owner_birth","foreign"),("launch_head","0"*40)])
def test_foreign_original_identity_is_rejected(tmp_path,key,value):
    value_config=config();value_config["original"][key]=value
    with pytest.raises(bootstrap.Blocked):bootstrap.validate_config(tmp_path,value_config)


def test_unarmed_cannot_run_even_with_requested_arm(tmp_path):
    with pytest.raises(bootstrap.Blocked):bootstrap.validate_config(tmp_path,config(),armed=True)


@pytest.mark.parametrize("name",["../outside","/outside","."])
def test_path_traversal_is_rejected(tmp_path,name):
    with pytest.raises(bootstrap.Blocked):bootstrap.relative(tmp_path,name)


def test_symlink_parent_is_rejected(tmp_path):
    (tmp_path/"real").mkdir();(tmp_path/"alias").symlink_to(tmp_path/"real",target_is_directory=True)
    with pytest.raises(bootstrap.Blocked):bootstrap.relative(tmp_path,"alias/file.json")


def test_write_once_is_complete_immutable_publication(tmp_path):
    path=tmp_path/"nested/record.json"
    bootstrap.write_once(path,{"value":1})
    assert bootstrap.read(path)=={"value":1}
    with pytest.raises(FileExistsError):bootstrap.write_once(path,{"value":2})
    assert bootstrap.read(path)=={"value":1} and not list(path.parent.glob("*.tmp"))


def test_reviewed_blob_checks_mode_sha_and_exact_tree(monkeypatch,tmp_path):
    data=b"reviewed bytes\n"; blob="a"*40
    entry={"commit":"b"*40,"source_path":"scripts/new.py","mode":"100644","blob":blob,"sha256":hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(bootstrap,"git",lambda root,*args,binary=False:data if args[0]=="cat-file" else f"100644 blob {blob}\tscripts/new.py\n")
    assert bootstrap.committed_entry(tmp_path,entry)==data
    with pytest.raises(bootstrap.Blocked):bootstrap.committed_entry(tmp_path,{**entry,"sha256":"0"*64})
    with pytest.raises(bootstrap.Blocked):bootstrap.committed_entry(tmp_path,{**entry,"mode":"100755"})


def test_completed_pass_requires_exact_release_and_hashes(tmp_path):
    stage,identity,out=fake_terminal(tmp_path)
    paths,terminal=bootstrap.completed_paths(tmp_path,stage,identity,0)
    assert terminal["status"]=="PASS" and set(paths)=={"outputs/example/metrics.json","outputs/example/terminal.json","outputs/example/reservation.json","outputs/example/released_lease.json"}


def test_completed_scientific_fail_retains_verdict_and_nonzero(tmp_path):
    stage,identity,out=fake_terminal(tmp_path,failed=True)
    _,terminal=bootstrap.completed_paths(tmp_path,stage,identity,1)
    assert terminal["status"]=="FAIL" and terminal["exit_code"]==1
    with pytest.raises(bootstrap.Blocked):bootstrap.completed_paths(tmp_path,{**stage,"allow_scientific_fail":False},identity,1)


@pytest.mark.parametrize("change",[{"execution_complete":False},{"errors":["source drift"]},{"status":"ERROR"},{"scientific_verdict":"PASS"}])
def test_partial_or_provenance_failure_never_continues(tmp_path,change):
    stage,identity,out=fake_terminal(tmp_path,failed=True)
    replace(out/"terminal.json",{**bootstrap.read(out/"terminal.json"),**change});repair_release(out)
    with pytest.raises(bootstrap.Blocked):bootstrap.completed_paths(tmp_path,stage,identity,1)


def test_actual_waited_exit_cannot_be_replaced_by_summary(tmp_path):
    stage,identity,out=fake_terminal(tmp_path)
    with pytest.raises(bootstrap.Blocked):bootstrap.completed_paths(tmp_path,stage,identity,1)


def test_artifact_drift_is_detected_before_commit(tmp_path):
    stage,identity,out=fake_terminal(tmp_path)
    (out/"metrics.json").write_text("changed")
    with pytest.raises(bootstrap.Blocked):bootstrap.completed_paths(tmp_path,stage,identity,0)


def test_forged_release_is_rejected(tmp_path):
    stage,identity,out=fake_terminal(tmp_path)
    replace(out/"released_lease.json",{**bootstrap.read(out/"released_lease.json"),"owner_pid":999})
    with pytest.raises(bootstrap.Blocked):bootstrap.completed_paths(tmp_path,stage,identity,0)


def test_missing_coverage_is_rejected(tmp_path):
    stage,identity,out=fake_terminal(tmp_path)
    replace(out/"terminal.json",{**bootstrap.read(out/"terminal.json"),"expected_outputs":["outputs/example/missing.json"]});repair_release(out)
    with pytest.raises(bootstrap.Blocked):bootstrap.completed_paths(tmp_path,stage,identity,0)


def test_external_frozen_artifact_is_checked_never_restaged(monkeypatch,tmp_path):
    stage,identity,out=fake_terminal(tmp_path)
    reference=tmp_path/"outputs/frozen/reference.csv";reference.parent.mkdir();reference.write_text("frozen")
    terminal=bootstrap.read(out/"terminal.json");terminal["artifact_hashes"]["outputs/frozen/reference.csv"]=bootstrap.digest(reference)
    replace(out/"terminal.json",terminal);repair_release(out)
    checked=[];monkeypatch.setattr(bootstrap,"prove_committed",lambda root,paths:checked.extend(paths))
    paths,_=bootstrap.completed_paths(tmp_path,stage,identity,0)
    assert checked==["outputs/frozen/reference.csv"] and "outputs/frozen/reference.csv" not in paths


@pytest.mark.parametrize("tag",["failure","error","skipped"])
def test_junit_rejects_every_nonpass_case(tmp_path,tag):
    path=tmp_path/"junit.xml";path.write_text(f'<testsuite><testcase name="x"><{tag}/></testcase></testsuite>')
    with pytest.raises(bootstrap.Blocked):bootstrap.pytest_counts(path)


def test_junit_and_raw_log_independently_agree(tmp_path):
    junit=tmp_path/"junit.xml";junit.write_text('<testsuites><testsuite tests="2" errors="0" failures="0" skipped="0"><testcase name="x"/><testcase name="y"/></testsuite></testsuites>')
    log=tmp_path/"command.log";log.write_text("==== 2 passed in 0.1s ====\n")
    assert bootstrap.pytest_counts(junit)==2
    bootstrap.prove_pytest_log(log,2)
    log.write_text("==== 2 passed, 1 deselected in 0.1s ====\n")
    with pytest.raises(bootstrap.Blocked):bootstrap.prove_pytest_log(log,2)


def test_empty_junit_is_incomplete(tmp_path):
    path=tmp_path/"junit.xml";path.write_text("<testsuite/>")
    with pytest.raises(bootstrap.Blocked):bootstrap.pytest_counts(path)


def test_raw_log_cannot_forge_or_duplicate_pass_count(tmp_path):
    path=tmp_path/"command.log"
    for text in ("1 passed\n","2 passed\n2 passed\n","2 passed, 1 skipped\n"):
        path.write_text(text)
        with pytest.raises(bootstrap.Blocked):bootstrap.prove_pytest_log(path,2)


@pytest.mark.parametrize("attrs",['tests="3" errors="0" failures="0" skipped="0"','tests="2" errors="1" failures="0" skipped="0"','tests="2" errors="0" failures="1" skipped="0"','tests="2" errors="0" failures="0" skipped="1"','tests="2"'])
def test_junit_declared_totals_and_failure_counts_are_enforced(tmp_path,attrs):
    path=tmp_path/"junit.xml";path.write_text(f'<testsuites><testsuite {attrs}><testcase name="x"/><testcase name="y"/></testsuite></testsuites>')
    with pytest.raises(bootstrap.Blocked):bootstrap.pytest_counts(path)


def test_selective_commit_preserves_unrelated_index(monkeypatch,tmp_path):
    path=tmp_path/"owned.txt";path.write_text("owned")
    calls=[]
    monkeypatch.setattr(bootstrap,"git_preflight",lambda root:calls.append(("preflight",)))
    def fake_git(root,*args,binary=False):
        calls.append(args)
        if args[:2]==("ls-files","--stage"):
            return "100644 old 0\tunrelated.txt\0"
        if args[0]=="rev-parse":return "a"*40+"\n"
        return ""
    monkeypatch.setattr(bootstrap,"git",fake_git)
    assert bootstrap.selective_commit(tmp_path,["owned.txt"],"Preserve exact owned evidence")=="a"*40
    assert ("commit","--only","-m","Preserve exact owned evidence","--","owned.txt") in calls
    assert calls.count(("preflight",))==2


def test_selective_commit_detects_unrelated_index_drift(monkeypatch,tmp_path):
    (tmp_path/"owned.txt").write_text("owned")
    monkeypatch.setattr(bootstrap,"git_preflight",lambda root:None)
    projections=iter(["100644 old 0\tunrelated.txt\0","100644 changed 0\tunrelated.txt\0"])
    monkeypatch.setattr(bootstrap,"git",lambda root,*args,binary=False:next(projections) if args[0]=="ls-files" else "")
    with pytest.raises(bootstrap.Blocked):bootstrap.selective_commit(tmp_path,["owned.txt"],"Preserve evidence")


def test_pre_go_guard_failure_waits_exact_new_child(monkeypatch,tmp_path):
    class FakeChild:
        pid=98765
        returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):assert self.returncode is not None;return self.returncode
    child=FakeChild()
    monkeypatch.setattr(bootstrap.subprocess,"Popen",lambda *args,**kwargs:child)
    monkeypatch.setattr(bootstrap,"birth",lambda pid:"known birth")
    fake_bootstrap_owner(tmp_path,"known birth")
    monkeypatch.setattr(bootstrap.os,"getpgid",lambda pid:child.pid)
    signals=[]
    def stop(pid,sig):signals.append(pid);child.returncode=-15
    monkeypatch.setattr(bootstrap.os,"killpg",stop)
    calls=[]
    def guard():
        calls.append(1)
        if len(calls)>1:raise bootstrap.Blocked("source drift before GO")
    stage={"argv":["/fake/python","job.py"],"executable_sha256":"0"*64}
    with pytest.raises(bootstrap.Blocked,match="source drift"):
        bootstrap.spawn(tmp_path,stage,tmp_path/"raw",guard)
    assert signals==[child.pid] and not (tmp_path/"raw/go.json").exists()
    assert bootstrap.read(tmp_path/"raw/command_exit.json")["waited"] is True
    assert child.pid not in bootstrap.ACTIVE_CHILDREN


def test_unknown_child_birth_retains_ownership(monkeypatch,tmp_path):
    class FakeChild:
        pid=87654
        def poll(self):return None
    child=FakeChild()
    monkeypatch.setattr(bootstrap.subprocess,"Popen",lambda *args,**kwargs:child)
    def unknown(pid):raise bootstrap.Blocked("birth unreadable")
    monkeypatch.setattr(bootstrap,"birth",lambda pid:"owner birth" if pid==bootstrap.os.getpid() else unknown(pid))
    fake_bootstrap_owner(tmp_path,"owner birth")
    signals=[];monkeypatch.setattr(bootstrap.os,"killpg",lambda pid,sig:signals.append(pid))
    stage={"argv":["/fake/python","job.py"],"executable_sha256":"0"*64}
    try:
        with pytest.raises(bootstrap.AmbiguousChild):bootstrap.spawn(tmp_path,stage,tmp_path/"raw",lambda:None)
        assert child.pid in bootstrap.ACTIVE_CHILDREN and not signals
        assert not (tmp_path/"raw/command_exit.json").exists()
    finally:
        bootstrap.ACTIVE_CHILDREN.pop(child.pid,None)


def test_cli_preserves_actual_scientific_failure_exit(monkeypatch,tmp_path):
    observed=[]
    def failed(root,path,sha):observed.append((root,path,sha));return 1
    monkeypatch.setattr(bootstrap,"execute",failed)
    result=bootstrap.main(["run","--main-root",str(tmp_path),"--config",str(tmp_path/"config.json"),"--config-sha256","a"*64])
    assert result==1 and observed==[(tmp_path,tmp_path/"config.json","a"*64)]


def test_even_complete_static_validation_never_launches(monkeypatch,tmp_path,capsys):
    path=tmp_path/"config.json";path.write_text('{"state":"ARMED"}')
    checks=[]
    def validated(root,config,armed=False):checks.append(armed);return {"state":"ARMED","pending":[],"execution_authorized":True}
    monkeypatch.setattr(bootstrap,"validate_config",validated)
    monkeypatch.setattr(bootstrap,"execute",lambda *args:pytest.fail("static validation must never execute"))
    assert bootstrap.main(["validate","--main-root",str(tmp_path),"--config",str(path)])==0
    result=json.loads(capsys.readouterr().out)
    assert checks==[True] and result["execution_authorized"] is False and result["config_identity_complete"] is True
