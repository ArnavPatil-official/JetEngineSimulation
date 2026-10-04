"""Small manufactured numeric checks; real source diagnostics run once in pipeline."""
from __future__ import annotations

import math
from pathlib import Path

import pytest


@pytest.fixture
def runtime():
    # This test file is never run while the original main chain is active.
    from scripts.phase8.scientific_workflow_gate import authorize_fixture_context, prepare_context
    root=Path(__file__).resolve().parents[1]
    context=prepare_context(root,"docs/phase8_saf_surrogate_registration.json",require_g0=True)
    authorize_fixture_context(context)
    import numpy as np
    return np


def manufactured_properties():
    return {"species_order":["A","B"],"molecular_weights_kg_kmol":[20,40],
        "gas_constant_J_kmol_K":8000,"atomic_weights_kg_kmol":[20],"elements":["E"],
        "element_mass_matrix":[[1],[1]],
        "coefficients":[[[3.0,0,0,0,0,-1000,1],[5.0,0,0,0,0,-1000,1]],
                        [[4.0,0,0,0,0,-2000,2],[6.0,0,0,0,0,-2000,2]]],
        "temperature_bounds_K":[[300,1000,3000],[300,1000,3000]],
        "burner_air_Y":[1,0],"compressor_air_Y":[1,0]}


def test_nasa_units_reference_and_equal_break(runtime):
    np=runtime
    from scripts.phase8.saf_surrogate.thermo import Thermo
    thermo=Thermo(manufactured_properties())
    assert np.allclose(thermo.species_cp(1000),[1200,800])
    assert np.allclose(thermo.species_cp(1000.001),[2000,1200])
    assert np.allclose(thermo.species_h(298.15),[400*(3*298.15-1000),200*(4*298.15-2000)])
    with pytest.raises(ValueError):thermo.species_cp(298.15)
    mass=np.asarray([.25,.75]);mix=thermo.mixture(900,mass)
    assert mix["R"]==pytest.approx(250) and mix["cp"]==pytest.approx(900)
    assert mix["gamma"]==pytest.approx(900/650)


def test_independent_energy_manufactured_flow(runtime):
    np=runtime
    from scripts.phase8.saf_surrogate.thermo import Thermo
    thermo=Thermo(manufactured_properties())
    ma,mf,T3,eta=2,.5,600,.8
    air=np.array([1,0]);fuel=np.array([[0,1]]);products=np.array([[.6,.4]])
    # Independent constant-cp/formation derivation, not residual autodiff.
    cp=np.array([1200,800]);href=np.array([400*(3*298.15-1000),200*(4*298.15-2000)])
    qchem=ma*(air@href)+mf*(fuel[0]@href)-(ma+mf)*(products[0]@href)
    incoming=ma*(air@cp)*(T3-298.15)+mf*(fuel[0]@cp)*(T3-298.15)
    T4=298.15+(incoming+eta*qchem)/((ma+mf)*(products[0]@cp))
    states={"ma":np.array([ma]),"eta_b":np.array([eta]),"T3":np.array([T3]),"fuel_Y":fuel}
    energy,elements=thermo.residuals(np.array([mf]),np.array([T4]),products,states)
    assert abs(energy[0])<1e-14 and np.max(np.abs(elements))<1e-14
    wrong,_=thermo.residuals(np.array([mf]),np.array([T4+1]),products,states)
    assert abs(wrong[0])>1e-4


def test_sobol_total_requests_and_balance(runtime):
    import json
    from scripts.phase8.saf_surrogate.inputs import query_designs
    root=Path(__file__).resolve().parents[1]
    reg=json.loads((root/"docs/phase8_saf_surrogate_registration.json").read_text())
    fixed=reg["scope"]["fixed_central"]
    row={key:fixed[key] for key in ("combustor_pressure_loss","eta_compressor","eta_turbine_polytropic","fpr_rated","eta_fan")}
    row.update({"eta_b_"+mode:fixed["eta_b"][mode] for mode in ("IDLE","APPROACH","TAKE-OFF")})
    draws={f"draw_{i:02d}":row for i in range(64)}
    public={"opr":18.65,"bpr":5.6,"rated_kN":42.6,"fit":reg["scope"]["fit_parameters"]}
    designed=query_designs(reg,draws,public)
    assert {key:len(value) for key,value in designed.items()}=={"train":4096,"validation":1024,"test":2048,"physics":2048,"ranking_test":4096,"study":640000}
    assert len(designed["train"][:64])==64
    assert len({q["draw_id"] for q in designed["train"][:64]})==64


def test_shared_public_draw_summary_contract(runtime):
    np=runtime
    from scripts.phase8.saf_surrogate.models import summarize_draws
    queries=[{"design_id":"paired","draw_id":f"draw_{i:02d}","f_JetA":.7,"f_HEFA":.1,"f_FT":.1,"f_ATJ":.1,"thrust_fraction":1} for i in range(64)]
    predictions=[{"draw_id":q["draw_id"],"status":"predicted","ff_kg_s":1+i/64,"T4_K":1500+i,
        "EI_CO2_kg_kg":3,"CO2_g_s":3000*(1+i/64),"lifecycle_g_s":1000*(1+i/64),
        "nvpm_dEI_number_pct":-5,"nvpm_status":"screening","seed_sd_ff_kg_s":.001,"seed_sd_T4_K":.1}
        for i,q in enumerate(queries)]
    summary=summarize_draws(queries,predictions)[0]
    assert summary["ff_kg_s"]["available_draws"]==64
    assert summary["ranking_q95_lifecycle_g_s"]==pytest.approx(np.quantile([p["lifecycle_g_s"] for p in predictions],.95))


def test_failed_rows_keep_conditional_metrics_without_full_domain_pass(runtime):
    np=runtime
    from scripts.phase8.saf_surrogate.thermo import Thermo
    from scripts.phase8.saf_surrogate.train import prediction_metrics,validation_pass
    thermo=Thermo(manufactured_properties())
    Y=np.asarray([[.25,.75],[.25,.75]])
    metrics=prediction_metrics(np.asarray([1.0,np.nan]),np.asarray([900,900]),Y,
        np.asarray([1.0,1.0]),np.asarray([900,900]),Y,thermo)
    assert metrics["complete"] is False and metrics["conditional_valid_rows"]==1 and metrics["invalid_rows"]==1
    assert metrics["ff_MAE_kg_s"]==0 and metrics["T4_MAE_K"]==0
    assert validation_pass(metrics,{"validation_selection":{}}) is False


def test_ensemble_renormalizes_mean_species_for_every_backend(runtime):
    np=runtime
    from scripts.phase8.saf_surrogate.models import ensemble_outputs
    members=[(np.asarray([1.]),np.asarray([900.]),np.asarray([[.25,.75+epsilon]])) for epsilon in (1e-15,2e-15,3e-15)]
    outputs=ensemble_outputs(members)
    assert outputs["mean_Y4"].sum()==pytest.approx(1.0,abs=1e-16)
    assert outputs["ff"].shape==(3,1) and outputs["Y4"].shape==(3,1,2)
