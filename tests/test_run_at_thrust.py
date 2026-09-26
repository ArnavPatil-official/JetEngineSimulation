"""
Phase 6 P6.1 Step 2 — thrust-matched cycle (IntegratedTurbofanEngine.run_at_thrust).

Round trip (phi -> thrust -> recovered phi), monotonicity of thrust in phi over
the working range, and explicit failure when the target is outside the
closure / T4-guard bracket.
"""

import contextlib
import io

import numpy as np
import pytest

from integrated_engine import (FUEL_LIBRARY, IntegratedTurbofanEngine, T4_GUARD_K,
                               ThrustTargetUnreachable)

JET_A1 = FUEL_LIBRARY["Jet-A1"]
ETA_B = 0.9963


@pytest.fixture(scope="module")
def engine():
    with contextlib.redirect_stdout(io.StringIO()):
        e = IntegratedTurbofanEngine()
    e.design_point.update(pi_c=43.2, mass_flow_core=79.9, combustor_pressure_loss=0.0442,
                          combustor_air_fraction=0.8, fpr=1.45)
    return e


def thrust_at(engine, phi):
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run_full_cycle(JET_A1, phi=phi, combustor_efficiency=ETA_B)


@pytest.mark.parametrize("phi0", [0.35, 0.5409, 0.65])
def test_round_trip_recovers_phi(engine, phi0):
    ref = thrust_at(engine, phi0)
    res = engine.run_at_thrust(ref["performance"]["thrust_kN"], JET_A1,
                               combustor_efficiency=ETA_B)
    tm = res["thrust_match"]
    assert tm["status"] == "converged"
    assert abs(tm["phi"] - phi0) < 1e-6
    assert abs(tm["residual_kN"]) < 1e-6
    assert res["performance"]["fuel_mass_flow"] == pytest.approx(
        ref["performance"]["fuel_mass_flow"], rel=1e-6)


def test_thrust_monotone_in_phi_over_working_range(engine):
    res = engine.run_at_thrust(250.0, JET_A1, combustor_efficiency=ETA_B)
    lo = res["thrust_match"].get("phi_lower_cycle_closure", 0.05)
    hi = res["thrust_match"]["phi_upper"]
    phis = np.linspace(lo * 1.0001, hi, 12)
    thrust = [thrust_at(engine, p)["performance"]["thrust_kN"] for p in phis]
    assert np.all(np.diff(thrust) > 0)


def test_unreachable_above_t4_guard_fails_cleanly(engine):
    with pytest.raises(ThrustTargetUnreachable) as exc:
        engine.run_at_thrust(400.0, JET_A1, combustor_efficiency=ETA_B)
    assert "T4 guard" in exc.value.reason
    assert exc.value.info["t4_guard_active"]
    assert exc.value.info["thrust_at_upper_kN"] < 400.0
    assert T4_GUARD_K == pytest.approx(2111.111, abs=1e-3)


def test_unreachable_below_minimum_fails_cleanly(engine):
    with pytest.raises(ThrustTargetUnreachable) as exc:
        engine.run_at_thrust(50.0, JET_A1, combustor_efficiency=ETA_B)
    assert "below the minimum thrust" in exc.value.reason


def test_solved_state_respects_t4_guard(engine):
    res = engine.run_at_thrust(270.0, JET_A1, combustor_efficiency=ETA_B)
    assert res["combustor"]["T_out"] <= T4_GUARD_K + 1e-6
