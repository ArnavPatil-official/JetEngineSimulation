"""Pure contract and negative-control tests; no scientific runtime imports."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from scripts.phase8.saf_surrogate.inputs import canonical_query,named_queries,state_for,simplex
from scripts.phase8.saf_surrogate.postprocess import derived_outputs
from scripts.phase8.saf_surrogate.registration import contained,write_once
from scripts.phase8.saf_surrogate.score import effect_floor
from scripts.phase8.saf_surrogate.timing import break_even
from scripts.phase8.saf_surrogate.train import monotonic_endpoints,validation_pass

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    reg=json.loads((ROOT/"docs/phase8_saf_surrogate_registration.json").read_text())
    fixed=reg["scope"]["fixed_central"]
    row={key:fixed[key] for key in ("combustor_pressure_loss","eta_compressor","eta_turbine_polytropic","fpr_rated","eta_fan")}
    row.update({"eta_b_"+mode:fixed["eta_b"][mode] for mode in ("IDLE","APPROACH","TAKE-OFF")})
    draws={key:dict(row) for key in ("central",*[f"draw_{i:02d}" for i in range(64)])}
    public={"opr":18.65,"bpr":5.6,"rated_kN":42.6,"fit":reg["scope"]["fit_parameters"]}
    return reg,draws,public


def query(**changes):
    return {"f_JetA":.7,"f_HEFA":.1,"f_FT":.1,"f_ATJ":.1,"thrust_fraction":.55,"draw_id":"draw_00",**changes}


def test_registered_total_requests_and_exact_outputs(inputs):
    reg,_,_=inputs
    budget=reg["sampling"]["main_teacher_budget"]
    assert budget["total_new_P8S_full_cycle_requests_excluding_separate_A1"]==11264+68+37449+640==49421
    assert reg["sampling"]["train_sizes"]==[64,128,256,512,1024,2048,4096]
    assert reg["model"]["seeds"]==[42,43,44]
    assert reg["teacher"]["workers"]==6
    expected=reg["provenance"]["successful_release"]["expected_outputs"]
    assert len(expected)==len(set(expected))
    assert all(path.startswith(reg["artifact_root"]+"/") for path in expected)
    assert not any("report.md" in path for path in expected)


def test_write_once_and_containment_negative_controls(tmp_path):
    path=tmp_path/"proof.json";first=write_once(path,{"identity":1})
    with pytest.raises(FileExistsError):write_once(path,{"identity":2})
    assert len(first)==64 and json.loads(path.read_text())=={"identity":1}
    with pytest.raises(ValueError):contained(tmp_path,"../foreign.json")
    with pytest.raises(ValueError):contained(tmp_path,"/foreign.json")


def test_canonical_query_hash_binds_draw_and_public_inputs(inputs):
    _,draws,public=inputs
    original=canonical_query(query(),draws,public)
    altered=copy.deepcopy(draws);altered["draw_00"]["eta_compressor"]-=.001
    assert original["input_sha256"]!=canonical_query(query(),altered,public)["input_sha256"]
    assert original["input_sha256"]!=canonical_query(query(thrust_fraction=.56),draws,public)["input_sha256"]
    for bad in (query(f_JetA=-.1),query(f_HEFA=float("nan")),query(thrust_fraction=.069),query(draw_id="foreign"),query(f_JetA=.8)):
        with pytest.raises(ValueError):canonical_query(bad,draws,public)


def test_named_manifest_preserves_alternative_context(inputs):
    _,draws,public=inputs;rows=named_queries(draws,public)
    assert len(rows)==68 and len({r["named_case_id"] for r in rows})==68
    outside=[r for r in rows if not r["in_product_API"]]
    assert len(outside)==4
    assert all(r["fuel_parts"]=={"JetA_dooley2010":1.0} and r["f_JetA"] is None for r in outside)
    assert [r["op"] for r in rows[:4]]==["TAKE-OFF","APPROACH","IDLE","CLIMB85"]


def test_eta_interpolation_preserves_four_anchors(inputs):
    _,draws,public=inputs;fixed=draws["draw_00"]
    for x,key in ((.07,"IDLE"),(.30,"APPROACH"),(.85,"TAKE-OFF"),(1,"TAKE-OFF")):
        assert state_for(query(thrust_fraction=x),public,draws)["eta_b"]==pytest.approx(fixed["eta_b_"+key])
    assert state_for(query(thrust_fraction=.925),public,draws)["eta_b"]==pytest.approx(fixed["eta_b_TAKE-OFF"])
    minus,plus,width=monotonic_endpoints([query(thrust_fraction=x) for x in (.07,.30,.85,1)])
    assert [v for q,p in zip(minus,plus) for v in (q["thrust_fraction"],p["thrust_fraction"])]==pytest.approx([.07,.0705,.30,.3005,.85,.8505,.9995,1])
    assert all(v>0 for v in width)


def test_effect_floor_zero_excluded_but_undefined_denies_pass():
    modes=("TAKE-OFF","APPROACH","IDLE","CLIMB85")
    rows=[{"fuel":"JetA","op":op,"status":"converged","ff":1} for op in modes]
    rows += [{"fuel":f"{fuel}-{p}","op":op,"status":"converged","ff":1+p/1000}
             for fuel in ("HEFA","FT","ATJ") for p in (10,20,30,50) for op in modes]
    assert effect_floor(rows)["threshold_kg_s"]==pytest.approx(.001)
    rows[-1]["ff"]=1
    assert len(effect_floor(rows)["excluded_zero"])==1
    rows[-2]["status"]="unreachable"
    assert effect_floor(rows)["state"]=="UNDEFINED" and effect_floor(rows)["threshold_kg_s"] is None


def test_brem_strict_bounds_and_dimensional_postprocessing():
    fuels={name:{"carbon_mass_fraction":.8,"lhv_liquid_MJ_kg":42} for name in ("JetA","HEFA","FT","ATJ")}
    properties={"surrogates":{name:name for name in fuels},"fuels":fuels,
        "lifecycle":{"baseline_fossil_gCO2e_MJ":89,"pathways":{name:{"triangular":{"mode":10}} for name in ("HEFA","FT","ATJ")}}}
    row=derived_outputs(query(),.5,properties)
    assert row["CO2_g_s"]==pytest.approx(1000*(44.01/12.011)*.8*.5)
    assert row["lifecycle_g_s"]==pytest.approx(.5*42*(.7*89+.3*10))
    assert derived_outputs(query(thrust_fraction=.30),.5,properties)["nvpm_dEI_number_pct"] is None
    assert derived_outputs(query(f_JetA=.6,f_HEFA=.4,f_FT=0,f_ATJ=0),.5,properties)["nvpm_dEI_number_pct"] is None


def test_break_even_undefined_and_setup_charged():
    assert break_even(10,.003,.001)==5000
    assert break_even(10,.001,.001) is None
    assert break_even(10,.0005,.001) is None


def test_new_modules_import_no_scientific_runtime_at_module_scope():
    forbidden={"numpy","scipy","mlx","cantera","catjet_core","integrated_engine"}
    for path in (ROOT/"scripts/phase8/saf_surrogate").glob("*.py"):
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.Import):names=[n.name.split('.')[0] for n in node.names]
            elif isinstance(node,ast.ImportFrom):names=[(node.module or '').split('.')[0]]
            else:continue
            assert not forbidden.intersection(names),(path,names)


def test_phase_shortcuts_refuse_before_any_gate_or_runtime_import(capsys):
    from scripts.phase8.saf_surrogate.run import main
    with pytest.raises(SystemExit) as error:main(["train"])
    assert error.value.code==2 and "Standalone phases are refused" in capsys.readouterr().err


def test_draw_order_and_identity_negative_controls_before_shared_math():
    from scripts.phase8.saf_surrogate.models import summarize_draws
    rows=[{**query(draw_id=f"draw_{i:02d}"),"design_id":"paired"} for i in range(64)]
    predictions=[{"draw_id":q["draw_id"],"status":"predicted"} for q in rows]
    with pytest.raises(ValueError,match="canonical ascending"):
        summarize_draws(list(reversed(rows)),list(reversed(predictions)))
    predictions[0]["draw_id"]="draw_63"
    with pytest.raises(ValueError,match="identity mismatch"):summarize_draws(rows,predictions)


@pytest.mark.parametrize("exit_code",[1,None])
def test_worker_shutdown_retains_failed_exit_and_blocks_completion(tmp_path,exit_code):
    from types import SimpleNamespace
    from scripts.phase8.saf_surrogate.teacher import ParallelTeacher
    worker=SimpleNamespace(exitcode=exit_code)
    class Pool:
        _processes={23:worker}
        def shutdown(self,*,wait,cancel_futures):
            assert wait and cancel_futures
    teacher=ParallelTeacher.__new__(ParallelTeacher)
    teacher.pool=Pool();teacher.spec={"output":str(tmp_path),"stage":"timing"}
    teacher.receipts={23:{"birth":"fixed birth","argv":["python","worker"],"parent_pid":17,"spec_sha256":"a"*64}}
    with pytest.raises(RuntimeError,match="worker shutdown failed"):
        teacher.close()
    receipt=json.loads((tmp_path/"proofs/workers/timing_23_exit.json").read_text())
    assert receipt["waited"] is True and receipt["exit_code"]==exit_code


def test_named_prerequisite_verifies_only_metadata_before_training(monkeypatch,tmp_path,inputs):
    import sys
    from types import SimpleNamespace
    import scripts.phase8
    from scripts.phase8.saf_surrogate import registration as subject
    reg,_,_=inputs
    prefix="outputs/phase7/p73_a1_cpp_20261004/"
    expected=[prefix+name for name in ("environment.json","artifact_hashes.json","command.exit.json","parity/parity.json")]
    calls=[]
    def validator(root,path,output,*,expected_binary_sha256,artifact_paths):
        assert artifact_paths==expected
        assert not set(artifact_paths)&set(reg["sole_score"]["named_paths"])
        calls.append(artifact_paths)
        return {"core_sha256":expected_binary_sha256}
    fake=SimpleNamespace(validate_consumer_terminal=validator)
    monkeypatch.setitem(sys.modules,"scripts.phase8.scientific_workflow_gate",fake)
    monkeypatch.setattr(scripts.phase8,"scientific_workflow_gate",fake,raising=False)
    monkeypatch.setattr(subject,"read_json",lambda path:{"registration_id":"P7.3-A1"})
    def metadata_hash(path):
        assert str(path).endswith(("phase7_p73_a1_registration.json","terminal.json","environment.json"))
        return "a"*64
    monkeypatch.setattr(subject,"sha256_file",metadata_hash)
    assert subject.verify_named_prerequisite(tmp_path,reg,"core")["id"]=="P7.3-A1"
    assert calls==[expected]
    changed=copy.deepcopy(reg);changed["sole_score"]["named_metadata_projection"].append(reg["sole_score"]["named_paths"][0])
    with pytest.raises(ValueError,match="metadata projection"):
        subject.verify_named_prerequisite(tmp_path,changed,"core")
