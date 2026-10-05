"""Synthetic scope/provenance/physics controls; run only after original ownership releases."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.phase8.nozzle_ode import inputs

ROOT = Path(__file__).resolve().parents[1]


def registration():
    return json.loads((ROOT / "docs/phase8_nozzle_ode_registration.json").read_text())


@pytest.fixture(autouse=True)
def numerical_authorization():
    from scripts.phase8.scientific_workflow_gate import prepare_context, authorize_fixture_context
    context = prepare_context(ROOT,"docs/phase8_nozzle_ode_registration.json",require_g0=True)
    authorize_fixture_context(context)
    return context


def test_execution_coverage_denies_missing_groups_and_rows():
    from scripts.phase8.nozzle_ode.run import validate_score_coverage
    reg = registration()
    splits = {"validation": [{"regime":regime} for regime in ("smooth_subcritical", "smooth_choked")]}
    values = {("validation", regime, seed, arm): {"cases":1, "points":161*4}
              for regime in ("smooth_subcritical", "smooth_choked")
              for seed in reg["models"]["paired_seeds"] for arm in reg["models"]["arms"]}
    validate_score_coverage(reg, splits, values, ("validation",))
    first = next(iter(values))
    with pytest.raises(inputs.Blocked, match="scoring groups"):
        validate_score_coverage(reg, splits, {k:v for k,v in values.items() if k != first}, ("validation",))
    values[first] = {"cases":1, "points":161*4-1}
    with pytest.raises(inputs.Blocked, match="row coverage"):
        validate_score_coverage(reg, splits, values, ("validation",))


def test_fixed_source_identity_and_draw_balance_refuse_omission():
    reg = registration()
    train_ids, named_ids = inputs.expected_source_ids(reg)
    train = [{"design_id":name, "prefix_index":str(i), "split":"train", "draw_id":f"draw_{i%64:02d}"}
             for i,name in enumerate(train_ids)]
    named = [{"named_case_id":name, "fuel":name.split("|")[0], "op":name.split("|")[1]} for name in named_ids]
    inputs.validate_source_ids(train,named,reg)
    with pytest.raises(inputs.Blocked,match="ID/order/count"):
        inputs.validate_source_ids(train[:-1],named,reg)
    train[0]["draw_id"] = "draw_01"
    with pytest.raises(inputs.Blocked,match="balance"):
        inputs.validate_source_ids(train,named,reg)


def test_alternate_context_keeps_separate_fuel_identity():
    alternate = {"in_product_API":"False", "fuel_parts":'{"JetA_dooley2010":1}',
                 **{f"f_{name}":"" for name in ("JetA","HEFA","FT","ATJ")}}
    assert inputs.fraction_fields(alternate,alternate=True) is None
    alternate["f_JetA"] = "1"
    with pytest.raises(inputs.Blocked,match="separate non-product identity"):
        inputs.fraction_fields(alternate,alternate=True)


def test_repeated_product_coefficients_retain_both_ids_and_scope_failures():
    reg = registration()
    gamma,gas_R = 1.3,287.0
    proof = {"status":"converged", "source_registration_sha256":"producer", "binary_sha256":"core",
             "property_manifest_sha256":"inputs", "source_commit":"commit", "input_sha256":"query",
             "gamma4":str(gamma),"R4_J_kg_K":str(gas_R),"cp4_J_kg_K":str(gamma*gas_R/(gamma-1)),
             "f_JetA":"1","f_HEFA":"0","f_FT":"0","f_ATJ":"0"}
    rows = [{**proof,"design_id":"first"},{**proof,"design_id":"second"}]
    cases,failures = inputs.property_cases(rows,[],reg,SimpleNamespace(binary_sha256="core"),"producer","inputs")
    assert [row["source_id"] for row in cases] == ["first","second"] and not failures
    rows[1]["R4_J_kg_K"] = "400"
    rows[1]["cp4_J_kg_K"] = str(gamma*400/(gamma-1))
    cases,failures = inputs.property_cases(rows,[],reg,SimpleNamespace(binary_sha256="core"),"producer","inputs")
    assert len(cases) == 1 and failures == [{"source":"train","source_id":"second","status":"OUT_OF_ENVELOPE"}]


def test_source_evidence_cannot_route_to_sealed_targets(tmp_path):
    with pytest.raises(inputs.Blocked,match="sealed"):
        inputs.safe_path(tmp_path,"sealed/test_teacher_rows.csv")
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}")
    (tmp_path/"escape.json").symlink_to(outside)
    with pytest.raises(inputs.Blocked,match="escaped"):
        inputs.safe_path(tmp_path,"escape.json")


def test_whole_condition_overlap_is_refused_but_product_duplicates_remain():
    def case(npr):
        return {"regime":"smooth_choked","NPR":npr,"gamma":1.3,"R":287.0}
    splits = {"train":[case(8.5)],"validation":[case(9.5)],"synthetic_test":[case(10.5)],
              "product_test":[case(12),case(12)]}
    inputs.assert_disjoint(splits)
    splits["product_test"].append(case(8.5))
    with pytest.raises(inputs.Blocked,match="overlap"):
        inputs.assert_disjoint(splits)


def test_analytic_residual_has_physical_chain_and_area_term():
    torch = pytest.importorskip("torch")
    from scripts.phase8.nozzle_ode.model import residuals
    raw = torch.tensor([[-.5,1.05,1.3,287.0],[.5,1.05,1.3,287.0]],dtype=torch.float64,requires_grad=True)
    x = raw[:,0]
    rho,u,T,p = 1+.1*x, .5+.2*x, 1-.03*x, .8-.1*x
    actual = residuals(torch.stack((rho,u,T,p),dim=1),raw)
    area = 1+.5*x*x
    expected = torch.stack(((.1*u+.2*rho)*area+rho*u*x, .2*rho*u-.1,
                            -1.3/(1.3-1)*.03+.2*u, p-rho*T),dim=1)
    assert torch.allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_choked_interior_oracle_keeps_sonic_gate_and_refuses_gap():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scripts.phase8.nozzle_ode.oracle import ExactReference, load_original
    exact = ExactReference(load_original(ROOT),registration())
    case = {"regime":"smooth_choked","NPR":10.0,"gamma":1.3,"R":287.0}
    values,errors = exact.profile(case,[-.875,-.125,.125,.875])
    assert values.shape == (4,4) and errors["throat_mach_abs"] <= 1e-10
    with pytest.raises(ValueError,match="outside registered"):
        exact.profile({**case,"NPR":4.0},[-1.,0.,1.])


def test_paired_comparison_cannot_replace_failed_seed_or_zero_control():
    from scripts.phase8.nozzle_ode.score import paired_decisions
    reg = registration()
    aggregates = {(panel,regime,seed,arm):{"accuracy_pass":True,"relative_rmse":.0005 if arm=="physics_on" else .001}
        for panel in ("synthetic_test","product_test") for regime in ("smooth_subcritical","smooth_choked")
        for seed in reg["models"]["paired_seeds"] for arm in reg["models"]["arms"]}
    assert paired_decisions(aggregates,reg)["registered_comparison_pass"]
    key = ("product_test","smooth_choked",reg["models"]["paired_seeds"][0],"physics_on")
    aggregates[key]["accuracy_pass"] = False
    assert not paired_decisions(aggregates,reg)["physics_on_accuracy_pass"]
    aggregates[key[:-1]+("data_only",)]["relative_rmse"] = 0
    assert not paired_decisions(aggregates,reg)["physics_benefit_pass"]


def test_raw_generation_rejects_unwaited_child_despite_inline_success(tmp_path):
    output = tmp_path/"outputs/source"
    proofs = output/"proofs"
    proofs.mkdir(parents=True)
    (tmp_path/"source.py").write_text("# frozen synthetic source\n")
    dep = {"source_registration":"source.json"}
    producer = {"relevant_files":{"implementation_create":["source.py"],"registration_create":[]}}
    identity,binary = {"registration_sha256":"registered"},{"path":"core.so","sha256":"core"}
    spec_path = proofs/"generation_command_spec.json"
    common = {"schema_version":1,"stage":"generate","root":str(tmp_path),"output":str(output),
        "registration_path":"source.json","registration_sha256":"registered","consumer_identity":identity,
        "start_identity":identity,"binary":binary,"property_inputs_manifest_sha256":"input",
        "owner_pid":100,"owner_birth":"owner", "owner_lease":{"path":"owner.json","sha256":"lease"},
        "source_hashes":{"source.py":inputs.sha256(tmp_path/"source.py")},
        "argv":["python","-m","scripts.phase8.saf_surrogate.run","_generate","--spec",str(spec_path)],
        "started_utc":"start"}
    records = {"spec":common,"handshake":{**common,"pid":101,"birth":"child"},
        "exit":{**common,"pid":101,"birth":"child","waited":False,"exit_code":0,
                "end_identity":identity,"identity_problems":[],"finished_utc":"end"}}
    paths = {"spec":spec_path,"handshake":proofs/"generation_handshake.json",
             "exit":proofs/"generation_exit.json","log":proofs/"generation.log"}
    for name,path in paths.items():
        path.write_text(json.dumps(records[name]) if name != "log" else "done\n")
    generation = {"launch":{"exit_code":0,"waited":True},"raw_command":{
        name:{"path":str(path.relative_to(tmp_path)),"sha256":inputs.sha256(path)} for name,path in paths.items()}}
    with pytest.raises(inputs.Blocked,match="actual successful joined wait"):
        inputs.generation_command_proof(tmp_path,output,dep,producer,"registered",identity,binary,"input",generation,None)
