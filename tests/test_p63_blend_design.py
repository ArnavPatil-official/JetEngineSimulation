"""P6.3 matched-thrust blend study: mass-basis blending and the registered design."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
import blend_matched_thrust_v5 as b  # noqa: E402

REG = json.loads((ROOT / "outputs" / "phase6" / "p63_registration.json").read_text())


def _mass_fractions(x: dict) -> dict:
    mw = {sp: b.SPECIES_ATOMS[sp][0] * b.M_C + b.SPECIES_ATOMS[sp][1] * b.M_H for sp in x}
    m = {sp: v * mw[sp] for sp, v in x.items()}
    tot = sum(m.values())
    return {sp: v / tot for sp, v in m.items()}


@pytest.mark.parametrize("name", list(b.REFERENCES))
def test_mass_basis_blend_preserves_component_mass_fractions(name):
    """ATJ-50 must be 50 % ATJ surrogate by MASS (make_saf_blend would give 42.5 %)."""
    p = b.REFERENCES[name]
    y = _mass_fractions(b.mass_blend_mole_fractions(p))
    # species mass per component, summed: reconstruct each component's mass share
    expect = {}
    for (cname, fuel), pi in zip(b.COMPONENTS.items(), p):
        for sp, ys in _mass_fractions(fuel.normalized_species()).items():
            expect[sp] = expect.get(sp, 0.0) + pi * ys
    assert y.keys() == {k for k, v in expect.items() if v > 0}
    for sp in y:
        assert y[sp] == pytest.approx(expect[sp], rel=1e-12)


def test_sobol_design_respects_the_registered_space():
    d = REG["design"]
    pts = b.sobol_design(d["n_sobol"], d["seed"])
    assert len(pts) == d["n_sobol"] == 256
    for pj, ph, pf, pa in pts:
        assert pj + ph + pf + pa == pytest.approx(1.0)
        assert 0.5 - 1e-12 <= pj <= 1.0 and min(ph, pf, pa) >= 0.0
    assert pts == b.sobol_design(d["n_sobol"], d["seed"])      # seeded, reproducible


def test_registration_matches_the_script():
    assert REG["design"]["references"] == list(b.REFERENCES)
    assert REG["design"]["seed"] == 42 and REG["design"]["n_corsia_common"] == 1000
