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


def test_thrust_and_t4_monotone_in_phi_over_working_range(engine):
    res = engine.run_at_thrust(250.0, JET_A1, combustor_efficiency=ETA_B)
    lo = res["thrust_match"]["phi_lower_cycle_closure"]
    phis = np.linspace(lo * 1.0001, 1.0, 16)
    runs = [thrust_at(engine, p) for p in phis]
    thrust = [r["performance"]["thrust_kN"] for r in runs]
    t4 = [r["combustor"]["T_out"] for r in runs]
    assert np.all(np.diff(thrust) > 0)
    assert np.all(np.diff(t4) > 0)


def test_warm_start_gives_the_same_root(engine):
    cold = engine.run_at_thrust(240.0, JET_A1, combustor_efficiency=ETA_B)
    for guess in (0.3, 0.5, 0.6):
        warm = engine.run_at_thrust(240.0, JET_A1, combustor_efficiency=ETA_B, phi_guess=guess)
        assert abs(warm["thrust_match"]["phi"] - cold["thrust_match"]["phi"]) < 1e-10
    near = engine.run_at_thrust(240.0, JET_A1, combustor_efficiency=ETA_B,
                                phi_guess=cold["thrust_match"]["phi"] * 1.01)
    assert near["thrust_match"]["n_cycle_evaluations"] < cold["thrust_match"]["n_cycle_evaluations"]


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


# ---- R6-B (docs/plan_phase6_review.md): guarded failures from a warm start ----
# At these v4 design values phi = 0.9 gives 291.726 kN at T4 = 2366.8 K, so the
# target is reachable below phi = 1 but only above the T4 guard. The warm path
# used to diagnose the guard from the original phi = 0.05 endpoint (where the
# cycle does not close) and raised a generic nozzle ValueError instead.

def _guarded_target(engine):
    ref = thrust_at(engine, 0.9)
    assert ref["combustor"]["T_out"] > T4_GUARD_K
    return ref["performance"]["thrust_kN"]


def test_warm_and_cold_agree_on_guarded_unreachable_target(engine):
    target = _guarded_target(engine)
    failures = []
    for guess in (None, 0.89, 0.5, 0.2):
        with pytest.raises(ThrustTargetUnreachable) as exc:
            engine.run_at_thrust(target, JET_A1, combustor_efficiency=ETA_B, phi_guess=guess)
        failures.append(exc.value)
    cold = failures[0]
    assert cold.info["t4_guard_active"]
    assert "T4 guard" in cold.reason
    for warm in failures[1:]:
        assert warm.reason == cold.reason
        assert warm.info["t4_guard_active"]
        assert warm.info["phi_upper"] == pytest.approx(cold.info["phi_upper"], abs=1e-10)
        assert warm.info["thrust_at_upper_kN"] == pytest.approx(
            cold.info["thrust_at_upper_kN"], abs=1e-8)


@pytest.mark.parametrize("kwargs", [
    dict(target_kN=float("nan")),
    dict(target_kN=float("inf")),
    dict(target_kN=-10.0),
    dict(target_kN=250.0, phi_guess=float("nan")),
    dict(target_kN=250.0, combustor_efficiency=float("nan")),
    dict(target_kN=250.0, t4_max_K=float("nan")),
])
def test_non_finite_inputs_are_rejected_not_reported_unreachable(engine, kwargs):
    kwargs.setdefault("combustor_efficiency", ETA_B)
    with pytest.raises(ValueError) as exc:
        engine.run_at_thrust(fuel_blend=JET_A1, **kwargs)
    assert not isinstance(exc.value, ThrustTargetUnreachable)


def test_configuration_errors_propagate_not_reported_unreachable(engine):
    saved = engine.design_point["combustor_pressure_loss"]
    engine.design_point["combustor_pressure_loss"] = 1.5
    try:
        with pytest.raises(ValueError, match="combustor_pressure_loss") as exc:
            engine.run_at_thrust(250.0, JET_A1, combustor_efficiency=ETA_B)
        assert not isinstance(exc.value, ThrustTargetUnreachable)
    finally:
        engine.design_point["combustor_pressure_loss"] = saved
