"""Manufactured PC orchestration fixtures: no project solves or label generation."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import pc_pipeline as pc


@pytest.fixture
def config(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, value in {
        "phase7_p73_a1_registration.json": {"outputs": {"directory": "outputs/phase8/p73"}},
        "phase8_saf_surrogate_registration.json": {"artifact_root": "outputs/phase8/saf"},
        "phase8_nozzle_ode_registration.json": {"outputs": {"root": "outputs/phase8/nozzle"}},
    }.items():
        (docs/name).write_text(json.dumps(value))
    from simulation import runtime
    monkeypatch.setattr(runtime,"require_ac",lambda:{"on_ac":True,"source":"manufactured"})
    class Inhibitor:
        metadata={"fixture":True}
        def __enter__(self): return self
        def __exit__(self,*args): return False
    monkeypatch.setattr(runtime,"SleepInhibitor",Inhibitor)
    for key in pc.THREAD_ENV:
        monkeypatch.setenv(key,"1")
    return pc.Config(root=tmp_path)


def handlers(config,calls,overrides=None):
    def dispatch(letter):
        calls.append(letter)
        value=(overrides or {}).get(letter)
        if isinstance(value,BaseException): raise value
        path=config.paths()[1]/"fixture_artifacts"/(letter+".txt")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("manufactured "+letter)
        return {"status":"COMPLETE","execution_complete":True,"scientific_verdict":"PASS",
                "artifacts":{path.relative_to(config.root).as_posix():pc.sha256(path)}, **(value or {})}
    return {letter:lambda letter=letter:dispatch(letter) for letter in pc.STAGES}


def test_direct_cli_dry_run_imports_no_scientific_library():
    root=Path(__file__).resolve().parents[1]
    code="""import runpy,sys
sys.argv=['scripts/pc_pipeline.py','--dry-run']
try: runpy.run_path('scripts/pc_pipeline.py',run_name='__main__')
except SystemExit as exc: assert exc.code==0
assert not any(name in sys.modules for name in ('numpy','torch','cantera','mlx.core'))
"""
    result=subprocess.run([sys.executable,"-c",code],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert [s["id"] for s in plan["stages"]]==list("abcdefghi")
    assert plan["workers"]==10 and plan["backend"]=="torch" and plan["device"]=="cpu"
    assert plan["a2"]=="DEFERRED" and plan["cpp_required"] is False


def test_workflow_backend_precedence_and_environment_normalization(config,monkeypatch):
    from scripts.phase8.python_pc import ScientificWorkflow
    monkeypatch.setenv("CATJET_ML_BACKEND"," MLX ")
    assert ScientificWorkflow(config.root,config.paths()[1]).backend=="mlx"
    assert ScientificWorkflow(config.root,config.paths()[1],backend="torch").backend=="torch"


def test_stop_resume_skips_byte_verified_stages_and_never_reopens_score(config):
    calls=[]
    first=pc.execute(config,stop_after="f",handlers=handlers(config,calls))
    assert first["status"]=="PAUSED" and calls==list("abcdef")
    resumed=pc.execute(config,resume=True,handlers=handlers(config,calls))
    assert resumed["status"]=="COMPLETE" and calls==list("abcdefghi")
    assert calls.count("f")==1
    assert all(s["skipped_verified_complete"] for s in resumed["stages"][:6])


def test_resume_refuses_changed_completed_artifact_before_dispatch(config):
    calls=[]
    pc.execute(config,stop_after="a",handlers=handlers(config,calls))
    (config.paths()[1]/"fixture_artifacts/a.txt").write_text("changed")
    with pytest.raises(RuntimeError,match="artifact changed"):
        pc.execute(config,resume=True,handlers=handlers(config,calls))
    assert calls==["a"]


def test_resume_refuses_same_bytes_replaced_with_symlink(config):
    calls=[]
    pc.execute(config,stop_after="a",handlers=handlers(config,calls))
    artifact=config.paths()[1]/"fixture_artifacts/a.txt"
    copy=config.root/"same-bytes.txt";copy.write_bytes(artifact.read_bytes())
    artifact.unlink();artifact.symlink_to(copy)
    with pytest.raises(RuntimeError,match="artifact changed"):
        pc.execute(config,resume=True,handlers=handlers(config,calls))
    assert calls==["a"]


@pytest.mark.parametrize("letter",["d","e","f","g"])
def test_consumed_partial_science_fails_closed(config,letter):
    calls=[]
    first=pc.execute(config,handlers=handlers(config,calls,{letter:RuntimeError("interrupted fixture")}))
    assert first["status"]=="ERROR" and first["stopped_at"]==letter
    before=list(calls)
    with pytest.raises(RuntimeError,match="consumed"):
        pc.execute(config,resume=True,handlers=handlers(config,calls))
    assert calls==before


def test_parity_failure_stops_before_science_even_on_resume(config):
    calls=[]
    run_handlers=handlers(config,calls,{"b":{"scientific_verdict":"FAIL"}})
    first=pc.execute(config,handlers=run_handlers)
    assert first["status"]=="FAIL" and calls==["a","b"]
    resumed=pc.execute(config,resume=True,handlers=run_handlers)
    assert resumed["status"]=="FAIL" and calls==["a","b"]


def test_actual_scientific_failure_completes_diagnostic_sequence(config):
    calls=[]
    result=pc.execute(config,handlers=handlers(config,calls,{"f":{"scientific_verdict":"FAIL"}}))
    assert result["status"]=="COMPLETE" and result["scientific_verdict"]=="FAIL"
    assert calls==list("abcdefghi")


def test_resume_rejects_foreign_host_lease(config):
    calls=[]
    pc.execute(config,stop_after="a",handlers=handlers(config,calls))
    lease=config.paths()[1]/"owner.json"
    lease.write_text(json.dumps({"owner_pid":os.getpid(),"owner_birth":"manufactured","host":"another-host"}))
    with pytest.raises(RuntimeError,match="foreign-host"):
        pc.execute(config,resume=True,handlers=handlers(config,calls))
    assert calls==["a"] and lease.exists()


@pytest.mark.parametrize("delta,verdict",[(0.0,"PASS"),(2e-9,"FAIL")])
def test_fresh_parity_dispatches_exact20_python_rows_at_registered_tolerance(config,monkeypatch,delta,verdict):
    import pandas as pd
    from scripts.phase8 import python_pc
    rows=pd.DataFrame({"UID":[f"fixture{i}" for i in range(21)],"Mode":["fixture"]*21,
                       "CO (g/kg)":[0.0]*21,"HC (g/kg)":[0.0]*21,"NOx (g/kg)":[0.0]*21})
    predictions=pd.DataFrame({"fixture_prediction":[1.0]*20})
    phase7=config.root/"outputs/phase7"
    phase7.mkdir(parents=True)
    (phase7/"calibration_v6.json").write_text(json.dumps({"params":{"manufactured":True}}))
    rows.iloc[:20].drop(columns=["CO (g/kg)","HC (g/kg)","NOx (g/kg)"]).join(predictions).to_csv(
        phase7/"calibration_v6_rows.csv",index=False)
    observed=[]
    class Model:
        def predict(self,params,actual):
            observed.append((params,len(actual)))
            return predictions+delta
        def close(self): observed.append("closed")
    monkeypatch.setitem(sys.modules,"lto_v5",SimpleNamespace(load_split=lambda:{},calibration_rows=lambda split:rows))
    monkeypatch.setitem(sys.modules,"lto_v6",SimpleNamespace(load_registration_v6=lambda:{}))
    monkeypatch.setitem(sys.modules,"v6_backend",SimpleNamespace(make_model_v6=lambda name,workers:
        observed.append((name,workers)) or Model()))
    import importlib
    compare_module=importlib.import_module("scripts.phase8.g0_parity")
    monkeypatch.setattr(compare_module,"ROOT",config.root)
    monkeypatch.setitem(sys.modules,"g0_parity",compare_module)
    metadata=config.paths()[1]; metadata.mkdir(parents=True)
    workflow=python_pc.ScientificWorkflow(config.root,metadata)
    workflow.sources={"fixture":"manufactured"}
    monkeypatch.setattr(workflow,"context",lambda *a,**kw:SimpleNamespace(assert_current=lambda:None))
    result=workflow.b()
    proof=pc.read_json(metadata/"parity20.json")
    assert result["scientific_verdict"]==verdict and proof["rows"]==20 and proof["rtol"]==1e-9
    assert observed==[("python",10),({"manufactured":True},20),"closed"]


def fixture_git(config):
    root=config.root
    pc.checked_git(root,"init","-b","fixture-base")
    pc.checked_git(root,"config","user.email","fixture@example.invalid")
    pc.checked_git(root,"config","user.name","Manufactured fixture")
    pc.checked_git(root,"add","docs")
    pc.checked_git(root,"commit","-m","Manufactured baseline")


def test_publication_network_retry_pushes_same_owned_commit_once(config,monkeypatch):
    fixture_git(config)
    monkeypatch.setattr(pc,"DIRECT_FILE_LIMIT",1024)
    monkeypatch.setattr(pc,"TRANSPORT_CHUNK_LIMIT",128)
    large=config.root/"outputs/phase8/saf/manufactured_large.bin"
    large.parent.mkdir(parents=True);original=os.urandom(4096);large.write_bytes(original)
    calls=[]; git_calls=[]; remote={"commit":None}; failed={"once":False}
    def git(root,*args):
        git_calls.append(args)
        if args[:2]==("ls-remote","--heads"):
            return "" if remote["commit"] is None else remote["commit"]+"\trefs/heads/"+args[-1]
        if args[0]=="push":
            assert "--force" not in args and "-f" not in args
            if not failed["once"]:
                failed["once"]=True
                raise RuntimeError("manufactured network unavailable")
            remote["commit"]=pc.checked_git(root,"rev-parse","HEAD")
            return ""
        return pc.checked_git(root,*args)
    run_handlers=handlers(config,calls)
    run_handlers["i"]=lambda:pc.publish_outputs(config,git=git)
    first=pc.execute(config,handlers=run_handlers)
    assert first["status"]=="ERROR" and first["stopped_at"]=="i"
    committed=pc.checked_git(config.root,"rev-parse","HEAD")
    intent=pc.read_json(config.paths()[1]/"publication_intent.json")
    assert intent["state"]=="COMMITTED" and intent["commit"]==committed
    resumed=pc.execute(config,resume=True,handlers=run_handlers)
    assert resumed["status"]=="COMPLETE" and calls==list("abcdefgh")
    assert remote["commit"]==committed
    assert sum(args[0]=="commit" for args in git_calls)==1
    receipt=pc.read_json(config.paths()[1]/"checkpoints/i.json")
    assert receipt["retried_owned_commit"] is True and receipt["force"] is False
    assert pc.checked_git(config.root,"rev-list","--count","HEAD")=="2"
    assert large.read_bytes()==original
    large_name=large.relative_to(config.root).as_posix()
    assert intent["large_original_hashes"][large_name]==pc.sha256(large)
    assert large_name not in intent["artifact_hashes"]
    assert large_name not in pc.checked_git(config.root,"ls-tree","-r","--name-only","HEAD").splitlines()
    large.unlink()
    restored=pc.restore_artifacts(config)
    assert restored["status"]=="PASS" and large.read_bytes()==original
    manifest=pc.read_json(config.paths()[1]/"artifact_transport.json")
    assert all(chunk["size"]<=128 for entry in manifest["files"].values() for chunk in entry["chunks"])


def test_publication_retry_refuses_mutated_large_original_not_in_commit(config,monkeypatch):
    fixture_git(config)
    monkeypatch.setattr(pc,"DIRECT_FILE_LIMIT",1024)
    monkeypatch.setattr(pc,"TRANSPORT_CHUNK_LIMIT",128)
    large=config.root/"outputs/phase8/saf/manufactured_large.bin"
    large.parent.mkdir(parents=True);large.write_bytes(os.urandom(4096))
    def git(root,*args):
        if args[0]=="ls-remote":return ""
        if args[0]=="push":raise RuntimeError("manufactured network unavailable")
        return pc.checked_git(root,*args)
    calls=[];run_handlers=handlers(config,calls);run_handlers["i"]=lambda:pc.publish_outputs(config,git=git)
    assert pc.execute(config,handlers=run_handlers)["stopped_at"]=="i"
    large.write_bytes(b"changed original")
    result=pc.execute(config,resume=True,handlers=run_handlers)
    assert result["status"]=="ERROR" and "Original large scientific artifact changed" in result["error"]
    assert pc.checked_git(config.root,"rev-list","--count","HEAD")=="2"


def test_publication_retry_refuses_changed_committed_bytes(config):
    fixture_git(config)
    calls=[]
    def git(root,*args):
        if args[0]=="ls-remote": return ""
        if args[0]=="push": raise RuntimeError("manufactured network unavailable")
        return pc.checked_git(root,*args)
    run_handlers=handlers(config,calls);run_handlers["i"]=lambda:pc.publish_outputs(config,git=git)
    assert pc.execute(config,handlers=run_handlers)["stopped_at"]=="i"
    (config.paths()[1]/"fixture_artifacts/a.txt").write_text("changed after commit")
    with pytest.raises(RuntimeError,match="artifact changed"):
        pc.execute(config,resume=True,handlers=run_handlers)
    assert pc.checked_git(config.root,"rev-list","--count","HEAD")=="2"


def test_publication_keeps_crlf_artifact_bytes_without_changing_git_config(config):
    fixture_git(config)
    pc.checked_git(config.root,"config","core.autocrlf","true")
    metadata=config.paths()[1];metadata.mkdir(parents=True)
    artifact=metadata/"manufactured.csv";artifact.write_bytes(b"fixture,value\r\na,1\r\n")
    remote={"sha":None}
    def git(root,*args):
        if args[0]=="ls-remote":return "" if remote["sha"] is None else remote["sha"]+"\trefs/heads/"+args[-1]
        if args[0]=="push":remote["sha"]=pc.checked_git(root,"rev-parse","HEAD");return ""
        return pc.checked_git(root,*args)
    result=pc.publish_outputs(config,git=git)
    name=artifact.relative_to(config.root).as_posix()
    assert result["status"]=="COMPLETE"
    assert pc.checked_git(config.root,"hash-object","--no-filters",str(artifact))==pc.checked_git(config.root,"rev-parse","HEAD:"+name)
    assert artifact.read_bytes()==b"fixture,value\r\na,1\r\n"
    assert pc.checked_git(config.root,"config","core.autocrlf")=="true"
