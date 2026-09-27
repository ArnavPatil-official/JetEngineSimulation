"""
P7.3 contracts (scripts/optimization/blend_matched_thrust_v6.py): mass and
energy accounting of mass-basis blends, the single liquid-basis LHV correction
in the lifecycle, nvPM validity exclusions, the registered pair list and claim
rule, and the frozen-v6 fixed-parameter draw convention. No cycle is run.
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

import blend_matched_thrust_v6 as b6  # noqa: E402
from simulation import fuels_v7  # noqa: E402

REG = json.loads((ROOT / "outputs" / "phase7" / "p73_registration.json").read_text())


def test_fuel_set_and_mass_fractions():
    f = b6.fuel_parts()
    assert len(f) == 1 + 12 + 3 + 1
    assert f["HEFA-30"] == {"JetA_dooley2012": 0.7, "HEFA_liang2025": 0.3}
    assert f["ATJ-100"] == {"ATJ_C1matched": 1.0}
    assert all(sum(p.values()) == pytest.approx(1.0) for p in f.values())
    assert b6.saf_fraction("FT-50") == 0.5 and b6.saf_fraction("JetA") == 0.0


@pytest.mark.parametrize("name", ["HEFA-10", "FT-30", "ATJ-50"])
def test_mass_blend_conserves_mass_and_energy(name):
    """For a mass blend, H mass % and LHV per kg are exactly mass-weighted."""
    parts = b6.fuel_parts()[name]
    mix = fuels_v7.properties(fuels_v7.mass_blend(parts))
    comp = {s: fuels_v7.properties(fuels_v7.surrogate(s)) for s in parts}
    for key in ("h_mass_pct", "lhv_gas_MJ_kg", "lhv_liquid_basis_MJ_kg"):
        assert mix[key] == pytest.approx(sum(w * comp[s][key] for s, w in parts.items()), rel=1e-10)


def test_lifecycle_uses_liquid_basis_once():
    lc = b6.corsia_central(b6.corsia())
    ja = fuels_v7.properties(fuels_v7.surrogate("JetA_dooley2012"))
    hvap = fuels_v7.load_properties()["heat_of_vaporization_MJ_kg"]["value"]
    assert hvap == 0.360
    assert b6.lifecycle_factor({"JetA_dooley2012": 1.0}, lc) == pytest.approx((ja["lhv_gas_MJ_kg"] - hvap) * 89.0)
    parts = b6.fuel_parts()["HEFA-50"]
    expect = sum(w * (fuels_v7.properties(fuels_v7.surrogate(s))["lhv_gas_MJ_kg"] - hvap)
                 * lc[b6.component_class(s)] for s, w in parts.items())
    assert b6.lifecycle_factor(parts, lc) == pytest.approx(expect, rel=1e-12)
    assert b6.component_class("JetA_dooley2010") == "fossil"


def test_corsia_draws_match_p63_generator():
    sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
    import blend_matched_thrust_v5 as b5
    _pp, common, _m = b5.corsia(42)
    old = common(20, 42)
    new = b6.corsia_common_draws(b6.corsia(), 20, 42)
    for o, n in zip(old, new):
        assert {k: o[k] for k in ("HEFA", "FT", "ATJ")} == {k: n[k] for k in ("HEFA", "FT", "ATJ")}


def test_nvpm_domain_exclusions():
    cfg = REG["nvpm"]
    to = {f: b6.nvpm_brem(f, 100.0, cfg) for f in ("HEFA-10", "FT-20", "ATJ-30", "HEFA-50", "FT-100")}
    for f in ("HEFA-10", "FT-20", "ATJ-30"):
        assert to[f]["status"] == "screening"
        dh = b6.saf_fraction(f) * 1.5
        assert to[f]["dEIn_pct"] == pytest.approx((-114.21 + 1.06 * 100.0) * dh)
    for f in ("HEFA-50", "FT-100"):
        assert to[f]["status"] == "unavailable" and to[f]["dEIn_pct"] is None
    assert b6.nvpm_brem("HEFA-10", 30.0, cfg)["status"] == "unavailable"      # F > 30 is strict
    assert b6.nvpm_brem("HEFA-10", 7.0, cfg)["status"] == "unavailable"
    assert b6.nvpm_brem("HEFA-30", 85.0, cfg)["status"] == "screening"


def test_pair_list_is_the_registered_one():
    p = b6.pairs()
    fam = pd.Series([x["family"] for x in p]).value_counts().to_dict()
    assert fam == {"SAF_vs_JetA": 12, "pathway_like_fraction": 12,
                   "context_neat_vs_JetA": 3, "context_neat_pathway": 3}
    assert {"a": "FT-20", "b": "ATJ-20", "family": "pathway_like_fraction"} in p


def _paired(n_agree, delta=1.0):
    return np.array([delta] * n_agree + [-delta] * (64 - n_agree))


def test_claim_rule():
    ok, _ = b6.claim(1.0, _paired(61), 0.5, True, True)
    assert ok
    assert not b6.claim(1.0, _paired(60), 0.5, True, True)[0]            # 60/64 < 95 %
    assert not b6.claim(1.0, _paired(64), 1.0, True, True)[0]            # |d| == S
    assert not b6.claim(0.0, np.zeros(64), 0.0, True, True)[0]           # zero difference
    assert not b6.claim(1.0, _paired(64), 0.5, False, True)[0]           # extrapolated point
    assert not b6.claim(1.0, _paired(64), 0.5, True, False)[0]           # unconverged draw
    assert not b6.claim(1.0, _paired(64)[:63], 0.5, True, True)[0]       # not all 64 draws
    assert not b6.claim(1.0, _paired(64), 0.5, True, True, extra_sign_frac=0.94)[0]
    zero_counts = np.array([1.0] * 61 + [0.0] * 3)                        # zero paired diff disagrees
    assert b6.claim(1.0, zero_counts, 0.5, True, True)[0]
    assert not b6.claim(1.0, np.array([1.0] * 60 + [0.0] * 4), 0.5, True, True)[0]


def test_draws_are_the_64_fixed_parameter_draws_without_refit_parameters():
    draws = b6.load_draws()
    assert [c for c, _ in draws] == [f"draw_{i:02d}" for i in range(64)]
    central = json.loads((ROOT / "outputs" / "phase7" / "p72_registration.json").read_text())["fixed_central"]
    fx = b6.draw_fixed(draws[0][1], central)
    assert set(fx) == set(central)
    assert not any(k.startswith("fit_") or k in ("W_ref", "a_thrust", "k_pi", "k_mdot") for k in fx)
    assert fx["eta_compressor"] == pytest.approx(draws[0][1]["fixed_eta_compressor"])


def test_tasks_hold_v6_parameters_fixed_and_climb_uses_takeoff_eta_b():
    import lto_v5 as v5
    central = json.loads((ROOT / "outputs" / "phase7" / "p72_registration.json").read_text())["fixed_central"]
    params = {"W_ref": 100.0, "a_thrust": 1.1, "k_pi": 1.3, "k_mdot": 0.4}
    ae3 = v5.load_rows([b6.AE3_UID], with_targets=False).iloc[0]
    fx = b6.draw_fixed(b6.load_draws()[5][1], central)
    keys, tasks = b6.tasks_for(params, fx, ae3, {"JetA": {"JetA_dooley2012": 1.0}})
    by = dict(zip([k[1] for k in keys], tasks))
    assert by["CLIMB85"][3] == fx["eta_b"]["TAKE-OFF"]
    assert by["IDLE"][3] == fx["eta_b"]["IDLE"]
    assert by["TAKE-OFF"][0]["mass_flow_core"] == pytest.approx(
        100.0 * (ae3["Rated Thrust (kN)"] / v5.F_REF_KN) ** 1.1)
    assert by["CLIMB85"][4] == pytest.approx(0.85 * ae3["Rated Thrust (kN)"])
    assert b6.OPERATING_POINTS["CLIMB85"][2] is False
