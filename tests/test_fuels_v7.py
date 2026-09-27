"""Phase 7 fuel representation (simulation/fuels_v7.py)."""
import math

import pytest

from simulation import fuels_v7 as f7


def test_lhv_method_reproduces_v5_n_dodecane():
    assert f7.properties({"NC12H26": 1.0})["lhv_gas_MJ_kg"] == pytest.approx(44.462, abs=1e-3)


def test_surrogates_are_normalised_and_use_mechanism_species():
    for name in f7.load_properties()["surrogates"]:
        x = f7.surrogate(name)
        assert sum(x.values()) == pytest.approx(1.0)
        for sp in x:
            f7._creck().species(sp)       # raises if absent from CRECK


def test_mass_blend_hydrogen_is_mass_weighted():
    a, b = f7.PRODUCTION["JetA"], f7.PRODUCTION["HEFA"]
    ha = f7.properties(f7.surrogate(a))["h_mass_pct"]
    hb = f7.properties(f7.surrogate(b))["h_mass_pct"]
    hm = f7.properties(f7.mass_blend({a: 0.7, b: 0.3}))["h_mass_pct"]
    assert hm == pytest.approx(0.7 * ha + 0.3 * hb, rel=1e-9)


def test_jet_a_is_no_longer_saf_like():
    """The v5 defect: Jet A and SAF heating values were within 0.25 %."""
    p = {k: f7.properties(f7.surrogate(v))["lhv_gas_MJ_kg"] for k, v in f7.PRODUCTION.items()}
    for saf in ("HEFA", "FT", "ATJ"):
        assert (p[saf] - p["JetA"]) / p["JetA"] > 0.01


def test_registered_p71_outcomes():
    """Outcomes of the P7.1 check against tolerances fixed before computation."""
    rows = {r["surrogate"]: r for r in f7.check_surrogates()}
    prod = rows[f7.PRODUCTION["JetA"]]
    assert prod["lhv_pass"] is True
    assert prod["h_pass"] is False and prod["aromatics_pass"] is False
    for saf in ("HEFA", "FT", "ATJ"):
        assert rows[f7.PRODUCTION[saf]]["h_pass"] is True
    assert rows[f7.PRODUCTION["ATJ"]]["lhv_pass"] is True


def test_nvpm_relations_values_and_validity():
    assert f7.nvpm_brem_dEIn_pct(0.5, 100.0) == pytest.approx((-114.21 + 106.0) * 0.5)
    assert math.isnan(f7.nvpm_brem_dEIn_pct(0.5, 30.0))      # F must exceed 30 %
    assert math.isnan(f7.nvpm_brem_dEIn_pct(0.7, 100.0))     # dH must be below 0.6
    # more hydrogen -> fewer particles, most strongly at low thrust
    assert f7.nvpm_brem_dEIn_pct(0.5, 40.0) < f7.nvpm_brem_dEIn_pct(0.5, 100.0) < 0.0


def test_unverified_caep11_relation_is_not_implemented():
    assert not hasattr(f7, "nvpm_caep11_k_fuel")
    assert "EXCLUDED" in f7.load_properties()["nvpm"]["caep11_via_teoh2022_S2"]["status"]
