"""
P7.2 (calibration v6) contracts: the registration reuses the Phase 6 procedure
unchanged except for the fuel; the fuel is threaded into the cycle; v6 outputs
cannot land on v5 evidence; the candidate rule, the penalty guard and the A4
check behave as registered; the extracted v5 held-out helper reproduces the
committed v5 tables.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))

import lto_v5 as v5  # noqa: E402
import lto_v6 as v6  # noqa: E402


@pytest.fixture(scope="module")
def reg6():
    return v6.load_registration_v6()


def test_registration_reuses_phase6_procedure(reg6):
    base = v5.load_registration()
    assert reg6["fixed_central"] == base["fixed_central"]           # incl. eta_b proxy values
    assert reg6["fit"]["bounds"] == base["fitted"]
    assert reg6["fit"]["n_trials"] == base["fit"]["full"]["n_trials"] == 150
    assert reg6["fit"]["polish_max_nfev"] == base["fit"]["full"]["polish_max_nfev"] == 100
    assert reg6["fit"]["tpe_seed"] == 42
    assert reg6["identifiability"]["grid_points"] == base["identifiability"]["full"]["grid_points"] == 17
    assert reg6["identifiability"]["inner_max_nfev"] == base["identifiability"]["full"]["inner_max_nfev"] == 40
    assert reg6["identifiability"]["rule"] == base["identifiability"]["rule"]
    assert reg6["heldout"]["acceptance"] == base["acceptance"]
    assert reg6["eta_b_proxy"]["Q_fuel_MJ_kg"] == base["eta_b_proxy"]["Q_fuel_MJ_kg"] == 44.462
    assert list(v5.FIT_BOUNDS) == reg6["fit"]["free"]


def test_fuel_is_the_frozen_dooley2012_surrogate(reg6):
    from simulation import fuels_v7
    x = v6.fuel_composition(reg6)
    assert x == fuels_v7.surrogate("JetA_dooley2012")
    assert set(x) == {"NC12H26", "IC8H18", "NPBENZ", "TMBENZ"}
    assert sum(x.values()) == pytest.approx(1.0, abs=1e-12)


def test_fuel_is_threaded_into_the_cycle(reg6):
    """Same cycle state, Jet-A1 (n-dodecane) vs Dooley 2012: the lower heating
    value of the new fuel must raise matched-thrust fuel flow."""
    split = v5.load_split()
    v5._init_worker(split["heldout_models"])
    fit = json.loads((ROOT / reg6["v5_selected_calibration"]).read_text())
    fx = reg6["fixed_central"]
    st = v5.mode_state(fit["params"], fx, 43.2, 10.0, 310.9, 1.0)
    base = (st, fx["eta_compressor"], fx["eta_turbine_polytropic"], fx["eta_b"]["TAKE-OFF"], 310.9, None)
    r_old = v5.solve_task(base + ("Jet-A1",))
    r_new = v5.solve_task(base + (v6.fuel_composition(reg6),))
    assert r_old["status"] == r_new["status"] == "converged"
    assert r_new["thrust_kN"] == pytest.approx(310.9, rel=1e-6)
    assert r_new["ff"] > r_old["ff"] * 1.005       # ~1.6 % lower LHV


@pytest.mark.parametrize("rel", ["outputs", "outputs/phase6", "outputs/results"])
def test_v6_refuses_v5_output_locations(rel):
    with pytest.raises(SystemExit):
        v6._refuse_v5_paths(ROOT / rel)


def test_v6_writes_are_write_once(tmp_path):
    (tmp_path / v6.FIT_JSON).write_text("{}")
    with pytest.raises(SystemExit):
        v6._out(tmp_path, v6.FIT_JSON)
    assert v6._out(tmp_path, "new.json") == tmp_path / "new.json"


def test_candidate_selection_ties_go_to_tpe():
    c = [{"name": "tpe_then_polish", "sse": 1e-3}, {"name": "polish_from_v5_A2", "sse": 1e-3}]
    assert v6.select_candidate(c)["name"] == "tpe_then_polish"
    c[1]["sse"] = 9e-4
    assert v6.select_candidate(c)["name"] == "polish_from_v5_A2"


def _profile_table(n_unreach_edge: int):
    rows = []
    grid = np.linspace(0.0, 1.0, 17)
    d = 400.0 * (grid - 0.5) ** 2                   # interval well inside the box
    for i, (g, dd) in enumerate(zip(grid, d)):
        rows.append(dict(param="p", i=i, value=g, sse=1.0, D=dd, nfev=5, status=2,
                         n_unreachable=n_unreach_edge if i == 16 else 0))
    return pd.DataFrame(rows)


def test_penalty_guard_blocks_identification_from_penalised_rows():
    verdict = {"p": {"IDENTIFIED": True, "D_at_lower_edge": 100.0, "D_at_upper_edge": 100.0}}
    clean = v6.apply_penalty_guard(_profile_table(0), verdict, 40)["p"]
    assert clean["IDENTIFIED"] and not clean["penalty_dependent"]
    bad = v6.apply_penalty_guard(_profile_table(3), verdict, 40)["p"]
    assert not bad["IDENTIFIED"] and bad["penalty_dependent"]
    assert bad["verdict_points_penalised"] == [1.0]


def test_verdict_points_include_edges_and_interval_brackets():
    v = np.linspace(0, 1, 17)
    d = 400.0 * (v - 0.5) ** 2
    pts = v6.verdict_points(v, d)
    inside = np.where(d < v5.CHI2_1_95)[0]
    assert {0, 16, inside.min(), inside.min() - 1, inside.max(), inside.max() + 1} <= set(pts)


def test_a4_check_matches_the_v5_test_on_the_v5_csv():
    df = pd.read_csv(v5.HOLDOUT_CSV)
    assert v6.a4_informativeness(df)["verdict"] == "PASS"
    flat = df.copy()
    flat["Predicted Fuel Flow (kg/s)"] = flat["Target Thrust (kN)"] * 0.0077
    assert v6.a4_informativeness(flat)["verdict"] == "FAIL"


def test_extracted_holdout_helper_reproduces_committed_v5_tables():
    reg = v5.load_registration()
    split = v5.load_split()
    committed = pd.read_csv(v5.HOLDOUT_CSV)
    cal = v5.calibration_rows(split)
    held = v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True),
                            split["heldout_groups"])
    pred = pd.DataFrame({"ff": committed["Predicted Fuel Flow (kg/s)"].to_numpy(),
                         "status": committed["Status"].to_numpy(),
                         "reason": committed["Reason"].to_numpy(),
                         **{c: committed[f"model_{c}"].to_numpy()
                            for c in ("phi", "T3", "T4", "T5", "m_core", "pi_c", "nox_corr_g_s", "thrust_kN")}},
                        index=held.index)
    df, summary, fields = v5.holdout_tables(reg, cal, held, pred)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), committed, check_dtype=False, rtol=1e-12)
    pd.testing.assert_frame_equal(summary, pd.read_csv(v5.HOLDOUT_SUMMARY), check_dtype=False, rtol=1e-12)
    j = json.loads(v5.HOLDOUT_JSON.read_text())
    assert fields["A2"] == j["A2"]
    assert fields["A3"]["verdict"] == j["A3"]["verdict"]
    assert fields["primary_group_weighted_mape_pct"] == pytest.approx(j["primary_group_weighted_mape_pct"])


def test_blend_gate():
    assert v6.blend_gate("PASS", "FAIL: no demonstrated skill")["open"]
    assert not v6.blend_gate("FAIL", "PASS")["open"]
    assert not v6.blend_gate("PASS", "ESCALATE (plan section 9 ...)")["open"]
