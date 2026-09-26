"""
Static engine-level thrust accounting (docs/plan_phase5_nozzle_repair.md).

On a static test stand the engine-level momentum balance is

    F = m_e u_e - m_0 u_0 + (p_e - p_0) A_e,   u_0 = 0

(NASA general thrust equation). The subtracted momentum is the *freestream*
inflow, not the velocity at an internal station such as the turbine exit.
The bypass stream (simulation/fan.py) and the PINN nozzle paths
(thrust_model='static_test_stand') already follow this; these tests hold the
analytic core nozzle to the same equation.

Pre-repair values at the frozen v4 take-off point are preserved in
outputs/takeoff_thrust_gap.json (P5.2 Step 2, commit f1bd920).
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "optimization"))

import calibrate_lto as cal  # noqa: E402

V4 = REPO_ROOT / "outputs" / "calibration_trent1000_ae3_v4.json"
PRE_REPAIR = REPO_ROOT / "outputs" / "takeoff_thrust_gap.json"

# Frozen v4 take-off point, captured from the cycle before the repair.
PRE_CORE_KN = 55.3914545
PRE_TOTAL_KN = 241.6099946
POST_CORE_KN = 71.8918355
POST_TOTAL_KN = 258.1103756
FUEL_FLOW_KG_S = 2.3182068896
PRE_STATE = {
    "T4_K": 1894.8836005359406,
    "turbine_exit_T_K": 1174.3793059470293,
    "turbine_exit_p_Pa": 369180.7343580888,
    "turbine_exit_u_m_s": 200.69011021595028,
    "core_jet_u_m_s": 874.4028631755976,
    "NOx_g_s": 110.39747208322001,
    "CO_g_s": 0.08857962935761912,
    "CO2_combustion_g_s": 7187.339571039451,
}


@pytest.fixture(scope="module")
def engine():
    with contextlib.redirect_stdout(io.StringIO()):
        return cal.IntegratedTurbofanEngine()


@pytest.fixture(scope="module")
def takeoff(engine):
    """Production take-off cycle at the frozen v4 calibration (analytic turbine and nozzle)."""
    rec = json.loads(V4.read_text())
    fixed = rec["fixed_parameters"]
    params = rec["best_params"]
    cal.set_mode_state(engine, params, fixed.get("combustor_air_fraction", 1.0),
                       fixed.get("combustor_heat_loss_fraction", 0.0), "Takeoff")
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run_full_cycle(fuel_blend=cal.FUEL_LIBRARY["Jet-A1"],
                                     phi=params["phi_to"],
                                     combustor_efficiency=params["eta_combustor"])


def _nozzle_state(u: float) -> dict:
    """Synthetic nozzle inlet: the analytic model treats T and p as the nozzle total state."""
    return {"T": 1174.0, "p": 369000.0, "u": u, "rho": 1.09,
            "cp": 1320.0, "R": 289.0, "gamma": 1320.0 / (1320.0 - 289.0)}


# ---- equation-based checks ---------------------------------------------------

def test_static_momentum_equation(engine):
    m_dot = 82.0
    with contextlib.redirect_stdout(io.StringIO()):
        out = engine.run_nozzle(_nozzle_state(200.0), m_dot)
    assert out["thrust_momentum"] == pytest.approx(m_dot * out["u"], rel=1e-12)
    assert out["thrust_pressure"] == 0.0            # ideal full expansion: p_e = p_amb
    assert out["thrust_total"] == pytest.approx(out["thrust_momentum"] + out["thrust_pressure"], rel=1e-12)


def test_thrust_independent_of_internal_inlet_velocity(engine):
    """At a fixed nozzle total state, the internal inlet velocity must not enter static thrust."""
    with contextlib.redirect_stdout(io.StringIO()):
        outs = [engine.run_nozzle(_nozzle_state(u), 82.0) for u in (0.0, 150.0, 300.0)]
    for o in outs[1:]:
        for key in ("u", "T", "p", "rho", "thrust_total", "thrust_momentum", "thrust_pressure"):
            assert o[key] == outs[0][key], key


def test_effective_exit_area_is_continuity_diagnostic(engine):
    m_dot = 82.0
    with contextlib.redirect_stdout(io.StringIO()):
        out = engine.run_nozzle(_nozzle_state(200.0), m_dot)
    assert out["A_exit_effective"] == pytest.approx(m_dot / (out["rho"] * out["u"]), rel=1e-12)
    assert out["thrust_model"] == "static_test_stand"


# ---- frozen v4 integration regression ---------------------------------------

def test_core_plus_bypass_use_freestream_momentum(takeoff):
    perf, nozz, fan = takeoff["performance"], takeoff["nozzle"], takeoff["fan"]
    m_core = perf["total_mass_flow"]
    assert nozz["thrust_total"] == pytest.approx(m_core * nozz["u"] + nozz["thrust_pressure"], rel=1e-12)
    assert fan["thrust_bypass"] == pytest.approx(perf["bypass_mass_flow"] * fan["u_bypass_exit"], rel=1e-12)
    assert perf["thrust_N"] == pytest.approx(nozz["thrust_total"] + fan["thrust_bypass"], rel=1e-12)


def test_frozen_v4_takeoff_thrust(takeoff):
    perf = takeoff["performance"]
    assert perf["thrust_core_kN"] == pytest.approx(POST_CORE_KN, abs=1e-3)
    assert perf["thrust_kN"] == pytest.approx(POST_TOTAL_KN, abs=1e-3)
    assert perf["fuel_mass_flow"] == pytest.approx(FUEL_FLOW_KG_S, rel=1e-9)


def test_thrust_change_is_exactly_the_removed_internal_momentum(takeoff):
    pre = json.loads(PRE_REPAIR.read_text())["base"]
    perf = takeoff["performance"]
    removed = perf["total_mass_flow"] * takeoff["turbine"]["u"] / 1e3
    assert removed == pytest.approx(pre["subtracted_inlet_momentum_kN"], rel=1e-12)
    assert perf["thrust_kN"] - pre["thrust_kN"] == pytest.approx(removed, rel=1e-9)
    assert perf["thrust_core_kN"] - pre["core_kN"] == pytest.approx(removed, rel=1e-9)
    assert perf["thrust_bypass_kN"] == pytest.approx(pre["bypass_kN"], rel=1e-12)


def test_thermal_fuel_and_emissions_inputs_unchanged(takeoff):
    perf, emis = takeoff["performance"], takeoff["emissions"]
    got = {
        "T4_K": takeoff["combustor"]["T_out"],
        "turbine_exit_T_K": takeoff["turbine"]["T"],
        "turbine_exit_p_Pa": takeoff["turbine"]["p"],
        "turbine_exit_u_m_s": takeoff["turbine"]["u"],   # still available for diagnostics
        "core_jet_u_m_s": takeoff["nozzle"]["u"],
        "NOx_g_s": emis["NOx_g_s"],
        "CO_g_s": emis["CO_g_s"],
        "CO2_combustion_g_s": emis["CO2_combustion_g_s"],
    }
    for key, want in PRE_STATE.items():
        assert got[key] == pytest.approx(want, rel=1e-9), key
    assert np.isfinite(perf["tsfc_mg_per_Ns"]) and perf["tsfc_mg_per_Ns"] > 0
