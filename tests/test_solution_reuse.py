"""
R6-D (docs/plan_phase6_review.md) — reused Cantera Solutions are order-independent.

Commit ba333a8 made Combustor and the fuel-air-ratio helper reuse one Solution
per role, reset to its as-constructed state before every call. This test
evaluates interleaved states in A/B/A order on the reused objects and compares
every result with fresh Solutions (a new Combustor / an emptied FAR cache),
for every shipped mechanism file. Equality is exact: the reset is only
acceptable if it reproduces fresh-Solution results bit for bit.
"""

import contextlib
import io

import numpy as np
import pytest

from integrated_engine import FUEL_LIBRARY, IntegratedTurbofanEngine
from simulation.combustor.combustor import Combustor

# (mechanism, fuel string for state A, fuel string for state B)
MECHANISMS = [
    ("data/creck_c1c16_full.yaml", "NC12H26:1.0", "NC12H26:0.85, IC8H18:0.15"),
    ("data/A1highT.yaml", "POSF10264:1.0", "POSF10264:1.0"),
    ("data/A2NOx.yaml", "POSF10325:1.0", "POSF10325:1.0"),
    ("data/n_dodecane_hychem.yaml", "NC12H26:1.0", "NC12H26:1.0"),
    ("data/isooctane.yaml", "IC8H18:1.0", "IC8H18:1.0"),
]
STATE_A = dict(T_in=850.0, p_in=42.0e5, phi=0.45, efficiency=0.9999)
STATE_B = dict(T_in=560.0, p_in=9.0e5, phi=0.30, efficiency=0.998)
SCALARS = ("T_out", "p_out", "h_out", "cp_out", "R_out", "gamma_out")


def _run(comb, state, fuel):
    return comb.run(fuel_blend=fuel, **state)


def _assert_identical(got, ref):
    for key in SCALARS:
        assert got[key] == ref[key], key
    np.testing.assert_array_equal(np.asarray(got["Y_out"]), np.asarray(ref["Y_out"]))


@pytest.mark.parametrize("mech,fuel_a,fuel_b", MECHANISMS,
                         ids=[m[0].split("/")[-1] for m in MECHANISMS])
def test_combustor_reuse_matches_fresh_solutions_in_aba_order(mech, fuel_a, fuel_b):
    reused = Combustor(mech)
    seq = [(STATE_A, fuel_a), (STATE_B, fuel_b), (STATE_A, fuel_a)]
    got = [_run(reused, s, f) for s, f in seq]
    for (s, f), g in zip(seq, got):
        _assert_identical(g, _run(Combustor(mech), s, f))   # fresh Solutions
    _assert_identical(got[2], got[0])


@pytest.fixture(scope="module")
def engine():
    with contextlib.redirect_stdout(io.StringIO()):
        e = IntegratedTurbofanEngine()
    e.design_point.update(pi_c=43.2, mass_flow_core=79.9, combustor_pressure_loss=0.045,
                          combustor_air_fraction=0.8, fpr=1.45)
    return e


def _fresh(engine):
    engine.combustor_creck._solutions = None
    engine.__dict__.pop("_far_gas_cache", None)


def test_fuel_air_ratio_reuse_matches_fresh_in_aba_order(engine):
    seq = [(FUEL_LIBRARY["Jet-A1"], 0.45), (FUEL_LIBRARY["Bio-SPK"], 0.62),
           (FUEL_LIBRARY["Jet-A1"], 0.45)]
    got = [engine._calculate_fuel_air_ratio(f, p) for f, p in seq]
    for (f, p), g in zip(seq, got):
        _fresh(engine)
        assert engine._calculate_fuel_air_ratio(f, p) == g
    assert got[2] == got[0]


def _cycle(engine, fuel, phi, eta_b):
    with contextlib.redirect_stdout(io.StringIO()):
        r = engine.run_full_cycle(FUEL_LIBRARY[fuel], phi=phi, combustor_efficiency=eta_b)
    return {
        "fuel_mass_flow": r["performance"]["fuel_mass_flow"],
        "fuel_air_ratio": r["performance"]["fuel_air_ratio"],
        "thrust_N": r["performance"]["thrust_N"],
        "tsfc_SI": r["performance"]["tsfc_SI"],
        "T4": r["combustor"]["T_out"],
        "cp4": r["combustor"]["cp_out"],
        "T5": r["turbine"]["T"],
        "u9": r["nozzle"]["u"],
    }


def test_full_cycle_reuse_matches_fresh_in_aba_order(engine):
    seq = [("Jet-A1", 0.50, 0.9999), ("HEFA-50", 0.62, 0.998), ("Jet-A1", 0.50, 0.9999)]
    got = [_cycle(engine, *c) for c in seq]
    for c, g in zip(seq, got):
        _fresh(engine)
        assert _cycle(engine, *c) == g
    assert got[2] == got[0]
