"""P8.1 unit parity: each C++ v6 component vs its Python original at 1e-12 relative.

Skipped when the C++ module has not been built (cpp/build.sh)."""

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))

backend = pytest.importorskip("simulation.catjet_backend")
try:
    backend.load_core()
except ImportError:
    pytest.skip("cpp/build/catjet_core not built", allow_module_level=True)

from integrated_engine import IntegratedTurbofanEngine, LocalFuelBlend, ThrustTargetUnreachable  # noqa: E402
from simulation.fan import Fan  # noqa: E402

RTOL = 1e-12
REG = json.loads((ROOT / "outputs" / "phase7" / "p72_registration.json").read_text())
FUEL = LocalFuelBlend("JetA_dooley2012", dict(REG["fuel"]["mole_fractions"]))


def close(a, b, rtol=RTOL, atol=0.0):
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


@pytest.fixture(scope="module")
def engines():
    with contextlib.redirect_stdout(io.StringIO()):
        py = IntegratedTurbofanEngine()
        cpp = backend.CppEngine()
    return py, cpp


STATES = [(288.15, 101325.0, 43.2, 0.86), (288.15, 101325.0, 30.1, 0.84), (250.0, 60000.0, 12.0, 0.88),
          (300.0, 101325.0, 3.5, 0.90)]


@pytest.mark.parametrize("T,p,pi,eta", STATES)
def test_compressor(engines, T, p, pi, eta):
    py, cpp = engines
    py.design_point["pi_c"] = pi
    py.compressor.eta_c = eta
    cpp.design_point["pi_c"] = pi
    cpp.compressor.eta_c = eta
    cpp.push_config()
    with contextlib.redirect_stdout(io.StringIO()):
        a = py.run_compressor(T, p)
    b = cpp.core.run_compressor(T, p)
    for k in ("T_out", "p_out", "h_in", "h_out", "work_specific"):
        assert close(a[k], b[k]), (k, a[k], b[k])


@pytest.mark.parametrize("fpr,eta,m", [(1.45, 0.90, 700.0), (1.1, 0.88, 300.0), (1.7, 0.92, 50.0), (1.0, 1.0, 10.0)])
def test_fan(engines, fpr, eta, m):
    _, cpp = engines
    cpp.design_point.update(fpr=fpr, eta_fan=eta)
    cpp.push_config()
    a = Fan(fpr=fpr, eta_fan=eta).run(288.15, 101325.0, m)
    b = cpp.core.run_fan(288.15, 101325.0, m)
    for k in a:
        assert close(a[k], b[k], atol=1e-300), (k, a[k], b[k])


@pytest.mark.parametrize("phi", [0.08, 0.2, 0.33879271456691074, 0.6, 1.0])
@pytest.mark.parametrize("fuel", [FUEL, LocalFuelBlend("Jet-A1", {"NC12H26": 1.0}),
                                  LocalFuelBlend("HEFA-50", {"NC12H26": 0.5, "NC10H22": 0.5})])
def test_fuel_air_ratio(engines, phi, fuel):
    py, cpp = engines
    a = py._calculate_fuel_air_ratio(fuel, phi)
    b = cpp.core.fuel_air_ratio(*cpp.fuel_args(fuel), phi)
    assert close(a, b), (a, b)


@pytest.mark.parametrize("T,p,phi,eff", [(901.5451512272988, 43.7724e5 * 0.955, 0.33879271456691074, 0.999893720535389),
                                          (700.0, 20.0e5, 0.25, 0.9998), (480.0, 4.0e5, 0.12, 0.998),
                                          (800.0, 30.0e5, 0.9, 1.0)])
def test_combustor(engines, T, p, phi, eff):
    py, cpp = engines
    with contextlib.redirect_stdout(io.StringIO()):
        a = py.combustor_creck.run(T_in=T, p_in=p, fuel_blend=FUEL, phi=phi, efficiency=eff, heat_loss_fraction=0.0)
    b = cpp.core.combustor_run(T, p, cpp.fuel_args(FUEL)[0], phi, eff, 0.0)
    for k in ("T_out", "p_out", "h_out", "cp_out", "R_out", "gamma_out"):
        assert close(a[k], b[k]), (k, a[k], b[k])
    Ya, Yb = np.asarray(a["Y_out"]), np.asarray(b["Y_out"])
    assert np.all(np.abs(Ya - Yb) <= RTOL * np.abs(Ya) + 1e-18)


@pytest.mark.parametrize("W,eta", [(80e6, 0.90), (40e6, 0.88), (5e6, 0.92)])
def test_turbine_and_nozzle(engines, W, eta):
    py, cpp = engines
    state = {"T": 1650.0, "p": 40e5, "cp": 1290.0, "R": 288.3, "gamma": 1290.0 / (1290.0 - 288.3),
             "rho": 1.0, "u": 1.0}
    py.turbine_design["eta_polytropic"] = eta
    cpp.turbine_design["eta_polytropic"] = eta
    cpp.push_config()
    with contextlib.redirect_stdout(io.StringIO()):
        a = py.run_turbine_analytic(state, 120.0, W)
        na = py.run_nozzle(a, 120.0)
    b = cpp.core.run_turbine_analytic(state, 120.0, W)
    nb = cpp.core.run_nozzle(b, 120.0)
    for k in ("rho", "u", "p", "T", "work_specific", "work_total"):
        assert close(a[k], b[k]), (k, a[k], b[k])
    for k in ("rho", "u", "p", "T", "thrust_total", "thrust_momentum", "thrust_pressure", "A_exit_effective"):
        assert close(na[k], nb[k], atol=1e-300), (k, na[k], nb[k])


def test_nox_correlation(engines):
    py, cpp = engines
    cpp.push_config()
    for opr, mf in ((43.8, 2.39), (20.0, 0.8), (8.0, 0.25), (1.0, 1.0), (30.0, 0.0)):
        assert close(py.emissions.estimate_nox(opr, mf), cpp.core.estimate_nox(opr, mf), atol=1e-300)


def _v6_state(mode):
    import lto_v5 as v5
    fit = json.loads((ROOT / "outputs" / "phase7" / "calibration_v6.json").read_text())
    row = v5.load_rows(["02P23RR126"], with_targets=True).set_index("Mode").loc[mode]
    x = v5.MODE_X[mode]
    st = v5.mode_state(fit["params"], REG["fixed_central"], row["Pressure Ratio"], row["Bypass Ratio"],
                       row["Rated Thrust (kN)"], x)
    return st, x * row["Rated Thrust (kN)"], REG["fixed_central"]["eta_b"][mode]


def _set_both(py, cpp, st):
    for e in (py, cpp):
        e.design_point.update(st)
        e.compressor.eta_c = REG["fixed_central"]["eta_compressor"]
        e.turbine_design["eta_polytropic"] = REG["fixed_central"]["eta_turbine_polytropic"]


@pytest.mark.parametrize("mode", ["TAKE-OFF", "APPROACH", "IDLE"])
def test_full_cycle_and_thrust_match_ae3(engines, mode):
    py, cpp = engines
    st, target, eta_b = _v6_state(mode)
    _set_both(py, cpp, st)
    with contextlib.redirect_stdout(io.StringIO()):
        a = py.run_at_thrust(target, FUEL, combustor_efficiency=eta_b)
    b = cpp.run_at_thrust(target, FUEL, combustor_efficiency=eta_b)
    assert close(a["thrust_match"]["phi"], b["thrust_match"]["phi"])
    assert a["thrust_match"]["n_cycle_evaluations"] == b["thrust_match"]["n_cycle_evaluations"]
    for k in ("fuel_mass_flow", "thrust_kN", "thrust_core_kN", "thrust_bypass_kN", "tsfc_mg_per_Ns",
              "specific_thrust_Ns_kg", "thermal_efficiency"):
        assert close(a["performance"][k], b["performance"][k]), k
    assert close(a["combustor"]["T_out"], b["combustor"]["T_out"])
    assert close(a["turbine"]["T"], b["turbine"]["T"]) and close(a["turbine"]["p"], b["turbine"]["p"])
    assert close(a["emissions"]["NOx_g_s"], b["emissions"]["NOx_g_s"])
    # warm start from a guess reaches the same root to the solver tolerance
    with contextlib.redirect_stdout(io.StringIO()):
        aw = py.run_at_thrust(target, FUEL, combustor_efficiency=eta_b, phi_guess=0.9 * a["thrust_match"]["phi"])
    bw = cpp.run_at_thrust(target, FUEL, combustor_efficiency=eta_b, phi_guess=0.9 * a["thrust_match"]["phi"])
    assert close(aw["thrust_match"]["phi"], bw["thrust_match"]["phi"])


def test_unreachable_reason_matches(engines):
    py, cpp = engines
    st, target, eta_b = _v6_state("TAKE-OFF")
    _set_both(py, cpp, st)
    for tgt in (5.0 * target, 0.001):
        with pytest.raises(ThrustTargetUnreachable) as ea, contextlib.redirect_stdout(io.StringIO()):
            py.run_at_thrust(tgt, FUEL, combustor_efficiency=eta_b)
        with pytest.raises(ThrustTargetUnreachable) as eb:
            cpp.run_at_thrust(tgt, FUEL, combustor_efficiency=eta_b)
        assert ea.value.reason == eb.value.reason
        assert str(ea.value) == str(eb.value)
