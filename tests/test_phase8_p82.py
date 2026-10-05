"""P8.2 numerical contracts registered before the first result.

These are local component and AE3 calibration-point checks, never held-out
scoring or ablation-ladder calibration.
"""

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))

from simulation.catjet_backend import CppEngine, load_core  # noqa: E402

try:
    core = load_core()
except ImportError:
    pytest.skip("cpp/build/catjet_core not built", allow_module_level=True)

MECHANISM = str(ROOT / "data" / "creck_c1c16_full.yaml")


@pytest.fixture(scope="module")
def thermo():
    return core.GasThermo(MECHANISM)


def test_enthalpy_mixing_closes_mass_energy_and_elements(thermo):
    products = thermo.from_moles(1700.0, 4.0e6, "CO2:1,H2O:1,N2:8")
    air = thermo.from_moles(900.0, 4.0e6, "O2:0.21,N2:0.79")
    for m_air in (1.0, 10.0, 30.0):
        mixed = thermo.mix(core.MassStream(products, 100.0),
                           core.MassStream(air, m_air), 4.0e6)
        assert math.isclose(mixed["mass_flow"], 100.0 + m_air, rel_tol=1e-14)
        assert mixed["energy_relative"] < 1e-10
        assert mixed["element_relative"] < 1e-12
        assert math.isclose(sum(mixed["Y"]), 1.0, rel_tol=1e-12)


def test_liquid_fuel_enthalpy_is_applied_once(thermo):
    gas = thermo.from_moles(1700.0, 4.0e6, "CO2:1,H2O:1,N2:8")
    h_gas = thermo.properties(gas)["h"]
    mass_product, mass_fuel = 105.0, 2.0
    liquid = thermo.at_enthalpy(h_gas - mass_fuel / mass_product * 360000.0,
                                gas.P, gas.Y)
    drop = mass_product * (h_gas - thermo.properties(liquid)["h"])
    assert math.isclose(drop, mass_fuel * 360000.0, rel_tol=1e-10)


def test_constant_cp_limit_is_v6_exact(thermo):
    state = thermo.from_moles(1650.0, 4.0e6, "CO2:1,H2O:1,N2:8")
    cp, R, work, mass = 1290.0, 288.3, 80.0e6, 120.0
    eta = 0.90
    upgraded = thermo.expand_for_work(core.MassStream(state, mass), work,
                                       eta, 50, cp, R)
    legacy = core.V6Engine(MECHANISM).run_turbine_analytic(
        {"T": state.T, "p": state.P, "cp": cp, "R": R,
         "gamma": cp / (cp - R)}, mass, work)
    assert math.isclose(upgraded["T"], legacy["T"], rel_tol=1e-10)
    assert math.isclose(upgraded["P"], legacy["p"], rel_tol=1e-10)
    assert upgraded["energy_relative"] < 1e-10


def test_polytropic_steps_double_converges(thermo):
    state = thermo.from_moles(1600.0, 4.0e6, "CO2:1,H2O:1,N2:8")
    stream = core.MassStream(state, 100.0)
    a = thermo.expand_for_work(stream, 40.0e6, 0.9, 50)
    b = thermo.expand_for_work(stream, 40.0e6, 0.9, 100)
    assert abs(a["T"] - b["T"]) / b["T"] < 1e-8
    assert abs(a["P"] - b["P"]) / b["P"] < 1e-8
    assert a["energy_relative"] < 1e-10
    assert a["element_relative"] < 1e-10
    assert a["T"] != b["T"] or a["P"] != b["P"]


@pytest.fixture(scope="module")
def ae3():
    import lto_v5 as v5
    from integrated_engine import LocalFuelBlend

    reg = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
    fit = json.loads((ROOT / "outputs/phase7/calibration_v6.json").read_text())
    fuel = LocalFuelBlend("JetA_dooley2012", dict(reg["fuel"]["mole_fractions"]))
    row = v5.load_rows(["02P23RR126"], with_targets=True).set_index("Mode").loc["TAKE-OFF"]
    state = v5.mode_state(fit["params"], reg["fixed_central"],
                          row["Pressure Ratio"], row["Bypass Ratio"],
                          row["Rated Thrust (kN)"], v5.MODE_X["TAKE-OFF"])
    with contextlib.redirect_stdout(io.StringIO()):
        v6 = CppEngine()
    v6.design_point.update(state)
    v6.compressor.eta_c = reg["fixed_central"]["eta_compressor"]
    v6.turbine_design["eta_polytropic"] = reg["fixed_central"]["eta_turbine_polytropic"]
    v6.push_config()
    p82 = core.P82Engine(v6.mechanism)
    c = p82.config
    c.base = v6.core.config
    p82.config = c
    return p82, v6.fuel_args(fuel), reg["fixed_central"]["eta_b"]["TAKE-OFF"]


def test_ae3_a1_a2_conservation_and_zero_cooling_limit(ae3):
    engine, fuel_args, eta = ae3
    phi = 0.33879271456691074
    a1 = engine.run_full_cycle(*fuel_args, phi, eta, 1)
    a2 = engine.run_full_cycle(*fuel_args, phi, eta, 2)
    for result in (a1, a2):
        assert result["max_mass_relative"] < 1e-10
        assert result["max_energy_relative"] < 1e-10
        assert result["max_element_relative"] < 1e-10
        assert result["burner_heat_rejection_W"] > 0.0
        assert result["performance"]["thrust_kN"] > 0.0
    assert set(a2["stages"]) == {"HP", "IP", "LP"}
    c = engine.config
    c.ngv_fraction = 0.0
    c.rotor_fraction = 0.0
    engine.config = c
    zero = engine.run_full_cycle(*fuel_args, phi, eta, 2)
    for key in ("T", "P"):
        assert math.isclose(zero["stations"]["ngv_exit_hp_in"][key],
                            a1["stations"]["dilution_exit"][key], rel_tol=1e-10)
    assert max(zero["max_energy_relative"], zero["max_element_relative"]) < 1e-10


def test_ae3_50_to_100_step_cycle(ae3):
    engine, fuel_args, eta = ae3
    phi = 0.33879271456691074
    c = engine.config
    c.pressure_steps = 50
    engine.config = c
    a = engine.run_full_cycle(*fuel_args, phi, eta, 2)
    c.pressure_steps = 100
    engine.config = c
    b = engine.run_full_cycle(*fuel_args, phi, eta, 2)
    for key in ("T", "p"):
        assert abs(a["turbine"][key] - b["turbine"][key]) / abs(b["turbine"][key]) < 1e-8
    assert a["max_energy_relative"] < 1e-10
    assert a["max_element_relative"] < 1e-10


@pytest.mark.parametrize("mode", ["TAKE-OFF", "APPROACH", "IDLE"])
def test_thrust_match_template_reproduces_v6_solver(mode):
    """P8-A2 ladder: the solver template driven by the v6 cycle is the v6 solver."""
    import lto_v5 as v5
    from integrated_engine import LocalFuelBlend
    sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
    from p82_g1 import AE3, setup

    reg = json.loads((ROOT / "outputs/phase7/p72_registration.json").read_text())
    fit = json.loads((ROOT / "outputs/phase7/calibration_v6.json").read_text())
    rows = v5.load_rows([AE3], with_targets=True).set_index("Mode")
    v6, _ = setup(core, reg, fit, rows, mode)
    fuel = LocalFuelBlend("JetA_dooley2012", dict(reg["fuel"]["mole_fractions"]))
    args = v6.fuel_args(fuel)
    eta = reg["fixed_central"]["eta_b"][mode]
    target = v5.MODE_X[mode] * rows.loc[mode, "Rated Thrust (kN)"]
    cases = [(target, None), (target, 0.9 * 0.3), (target, 0.31), (5000.0, None), (0.5, None)]
    for t, guess in cases:
        ref = v6.core.run_at_thrust(t, *args, eta, phi_guess=guess)
        new = core.v6_template_run_at_thrust(v6.core, t, *args, eta, phi_guess=guess)
        assert new["status"] == ref["status"], (t, guess)
        if ref["status"] == "converged":
            assert new["phi"] == ref["thrust_match"]["phi"]
            assert new["n_cycle_evaluations"] == ref["thrust_match"]["n_cycle_evaluations"]
            assert new["fuel_mass_flow"] == ref["performance"]["fuel_mass_flow"]
        else:
            assert new["reason"] == ref["reason"]
