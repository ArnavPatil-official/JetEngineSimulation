"""
Phase 6 P6.1 registration amendment A1 (docs/plan_phase6_review.md R6-A, R6-C).

The active registration must not fix or fit beta, must keep R0's pre-registered
decisions verbatim, and its hardcoded constants must reproduce from their
stated derivations.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
import lto_v5 as v5  # noqa: E402

REG = ROOT / "outputs" / "phase6" / "p61_registration.json"
R0 = ROOT / "outputs" / "phase6" / "superseded" / "p61_registration_R0_REJECTED.json"
PRESERVED = ("split", "weighting", "fitted", "fit", "identifiability", "baselines",
             "heldout_metrics", "acceptance", "t4_guard_K", "fuel", "components")


@pytest.fixture(scope="module")
def reg():
    return json.loads(REG.read_text())


def test_r0_archived_unmodified_and_cross_referenced(reg):
    sha = hashlib.sha256(R0.read_bytes()).hexdigest()
    assert sha == "89e7d11630f2aa0bd1cbfb1e2a8fccf8e57c98b69132388ffb2e7ddb35158dc8"
    assert reg["amendment"] == "A1"
    assert reg["supersedes"]["sha256"] == sha


def test_a1_preserves_r0_decisions(reg):
    r0 = json.loads(R0.read_text())
    for key in PRESERVED:
        assert reg[key] == r0[key], key


def test_beta_is_neither_fixed_nor_fitted(reg):
    assert "combustor_air_fraction" not in reg["fixed_central"]
    assert "combustor_air_fraction" not in reg["fixed_ranges"]
    assert "combustor_air_fraction" not in reg["fitted"]
    assert "combustor_air_fraction" not in v5.FIXED
    assert v5.SINGLE_ZONE_AIR_FRACTION == 1.0
    state = v5.mode_state({"W_ref": 90.0, "a_thrust": 1.0, "k_pi": 0.6, "k_mdot": 0.6},
                          v5.FIXED, 43.2, 9.1, 310.9, 1.0)
    assert state["combustor_air_fraction"] == 1.0
    for rng in reg["fixed_ranges"].values():
        assert "ILLUSTRATIVE" not in rng["basis"]


def test_fixed_ranges_have_sources_and_contain_central(reg):
    for name, rng in reg["fixed_ranges"].items():
        assert rng["sources"], name
        central = reg["fixed_central"][name]
        if name == "eta_b":
            for mode, (lo, hi) in rng["range"].items():
                assert lo <= central[mode] <= hi
        else:
            lo, hi = rng["range"]
            assert lo <= central <= hi, name


def test_heating_values_reproduce_from_creck():
    hv = v5.creck_heating_values()
    assert hv["Q_CO_MJ_KG"] == pytest.approx(v5.Q_CO_MJ_KG, abs=5e-5)
    assert hv["Q_FUEL_MJ_KG"] == pytest.approx(v5.Q_FUEL_MJ_KG, abs=5e-5)


def test_fan_conversion_is_exact_for_the_fan_model():
    from simulation.fan import Fan
    e_poly, fpr, T0 = 0.93, 1.5, 288.15
    k = 0.4 / 1.4
    # polytropic temperature ratio vs the fan model's isentropic-efficiency rise
    T_poly = T0 * fpr ** (k / e_poly)
    fan = Fan(fpr=fpr, eta_fan=v5.fan_isentropic(e_poly, fpr)).run(T0, 101325.0, 1.0)
    assert fan["T_exit"] == pytest.approx(T_poly, rel=1e-12)
    lo, hi = v5.fan_isentropic_envelope()
    assert lo < 0.90 < hi


def test_compressor_variable_cp_conversion_brackets_central():
    lo, hi = (v5.compressor_isentropic_variable_cp(e, n_steps=500)
              for e in v5.FIXED_RANGES["eta_compressor"]["polytropic_range"])
    assert lo < v5.FIXED["eta_compressor"] < hi


def test_v5_model_requires_heldout_nox_exclusion():
    fixed = dict(v5.FIXED, eta_b={m: 0.9999 for m in v5.MODES})
    with pytest.raises(ValueError, match="nox_fit_exclude_models"):
        v5.V5Model(fixed, nox_fit_exclude_models=[])
    with pytest.raises(ValueError, match="combustor_air_fraction"):
        v5.V5Model(dict(fixed, combustor_air_fraction=0.8), nox_fit_exclude_models=["x"])


def test_surrogate_lhv_reproduces_from_creck_thermo():
    """P6.2: stored surrogate LHVs = complete combustion to H2O(g) at 298.15 K, CRECK thermo."""
    import cantera as ct
    from simulation.fuels import ATJ_SPK, FT_SPK, HEFA_SPK, JET_A1
    g = ct.Solution(str(ROOT / "data" / "creck_c1c16_full.yaml"))

    def h(sp):
        g.TPX = 298.15, ct.one_atm, f"{sp}:1"
        return g.enthalpy_mole

    ch = {"NC12H26": (12, 26), "NC10H22": (10, 22), "IC8H18": (8, 18)}
    for fuel in (JET_A1, HEFA_SPK, FT_SPK, ATJ_SPK):
        dh = mw = 0.0
        for sp, x in fuel.normalized_species().items():
            c, hh = ch[sp]
            dh += x * (h(sp) + (c + hh / 4) * h("O2") - c * h("CO2") - hh / 2 * h("H2O"))
            mw += x * g.molecular_weights[g.species_index(sp)]
        assert fuel.LHV_MJ_per_kg == pytest.approx(dh / mw / 1e6, abs=6e-4), fuel.name


def test_a2_selection_rule_and_consumers():
    """Amendment A2: lowest calibration SSE wins, ties go to the registered fit;
    every downstream v5 consumer reads the selected calibration."""
    a2 = json.loads(v5.AMENDMENT_A2.read_text())
    assert a2["amendment"] == "A2" and "no held-out target" in a2["information_used"]
    reg_fit, pilot = {"name": "registered_full_fit", "sse": 0.5}, {"name": "pilot_start_polish", "sse": 0.1}
    assert v5.select_optimum([reg_fit, pilot])["name"] == "pilot_start_polish"
    assert v5.select_optimum([reg_fit, dict(pilot, sse=0.5)])["name"] == "registered_full_fit"
    assert v5.V5_FIT != v5.FULL_FIT
    for script in ("scripts/validation/p62_parameter_bands.py",
                   "scripts/validation/mechanism_sensitivity.py"):
        text = (ROOT / script).read_text()
        assert "v5.V5_FIT" in text and "FULL_FIT" not in text, script
