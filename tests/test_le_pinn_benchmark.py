"""
Physics-grounded benchmark tests for the LE-PINN nozzle wrapper.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.nozzle.le_pinn import LE_PINN, MinMaxNormalizer, run_le_pinn
from simulation.nozzle.nozzle import run_nozzle_pinn


ROOT = Path(__file__).resolve().parent.parent
NOZZLE_PATH = ROOT / "models" / "nozzle_pinn.pt"
LE_PINN_PATH = ROOT / "models" / "le_pinn_unified.pt"
LE_PINN_ENGINE_PATH = ROOT / "models" / "le_pinn_engine_unified.pt"
BOTH_MODELS_PRESENT = NOZZLE_PATH.exists() and (
    LE_PINN_PATH.exists() or LE_PINN_ENGINE_PATH.exists()
)

pytestmark = pytest.mark.skipif(
    not BOTH_MODELS_PRESENT,
    reason="Required PINN checkpoints are not available for benchmark tests.",
)

INLET_CONDITIONS = [
    ("nominal", 6.5, 1700.0, 658612.0, 500.0, 0.25, 0.10),
    ("low_NPR", 4.0, 1500.0, 405300.0, 400.0, 0.25, 0.12),
    ("high_NPR", 8.0, 1900.0, 810600.0, 650.0, 0.25, 0.09),
    ("ood_hot", 6.5, 2200.0, 658612.0, 600.0, 0.25, 0.10),
    ("ood_cool", 6.5, 1200.0, 658612.0, 450.0, 0.25, 0.10),
]

THERMO_CONFIGS = {
    "JetA1": {"cp": 1150.0, "R": 287.0, "gamma": 1.33},
    "HEFA50": {"cp": 1200.0, "R": 287.0, "gamma": 1.30},
    "BioSPK": {"cp": 1250.0, "R": 287.0, "gamma": 1.28},
}

CONDITION_MAP = {name: values for name, *values in INLET_CONDITIONS}


def make_inlet(T_in: float, P_in: float, u_in: float, R: float = 287.0) -> dict[str, float]:
    return {"rho": P_in / (R * T_in), "u": u_in, "p": P_in, "T": T_in}


def _iter_condition_fuel_pairs():
    for name, npr, T_in, P_in, u_in, A_in, A_exit in INLET_CONDITIONS:
        for fuel_name in THERMO_CONFIGS:
            yield name, npr, T_in, P_in, u_in, A_in, A_exit, fuel_name


@lru_cache(maxsize=None)
def _run_le_cached(
    condition_name: str,
    fuel_name: str,
    return_profile: bool,
):
    npr, T_in, P_in, u_in, A_in, A_exit = CONDITION_MAP[condition_name]
    thermo = THERMO_CONFIGS[fuel_name]
    inlet_state = make_inlet(T_in, P_in, u_in, R=thermo["R"])
    m_dot = inlet_state["rho"] * inlet_state["u"] * A_in
    return run_le_pinn(
        model_path=str(LE_PINN_PATH),
        inlet_state=inlet_state,
        ambient_p=101325.0,
        A_in=A_in,
        A_exit=A_exit,
        length=1.0,
        thermo_props=thermo,
        m_dot=m_dot,
        n_axial=50,
        n_radial=20,
        device="cpu",
        return_profile=return_profile,
        thrust_model="static_test_stand",
    )


@lru_cache(maxsize=None)
def _run_regular_nominal_cached():
    npr, T_in, P_in, u_in, A_in, A_exit = CONDITION_MAP["nominal"]
    thermo = THERMO_CONFIGS["JetA1"]
    inlet_state = make_inlet(T_in, P_in, u_in, R=thermo["R"])
    m_dot = inlet_state["rho"] * inlet_state["u"] * A_in
    return run_nozzle_pinn(
        model_path=str(NOZZLE_PATH),
        inlet_state=inlet_state,
        ambient_p=101325.0,
        A_in=A_in,
        A_exit=A_exit,
        length=1.0,
        thermo_props=thermo,
        m_dot=m_dot,
        device="cpu",
        return_profile=False,
        thrust_model="static_test_stand",
    )


@lru_cache(maxsize=None)
def _regular_fallback_rate_jet_a1() -> float:
    fallback_count = 0
    for name, npr, T_in, P_in, u_in, A_in, A_exit in INLET_CONDITIONS:
        thermo = THERMO_CONFIGS["JetA1"]
        inlet_state = make_inlet(T_in, P_in, u_in, R=thermo["R"])
        m_dot = inlet_state["rho"] * inlet_state["u"] * A_in
        result = run_nozzle_pinn(
            model_path=str(NOZZLE_PATH),
            inlet_state=inlet_state,
            ambient_p=101325.0,
            A_in=A_in,
            A_exit=A_exit,
            length=1.0,
            thermo_props=thermo,
            m_dot=m_dot,
            device="cpu",
            return_profile=False,
            thrust_model="static_test_stand",
        )
        fallback_count += int(result["used_fallback"])
    return fallback_count / len(INLET_CONDITIONS)


def _count_upward_steps(values: np.ndarray) -> int:
    return int(np.sum(np.diff(values) > 0.0))


class TestPhysicalBounds:
    def test_le_pinn_exit_state_positive(self) -> None:
        for name, npr, T_in, P_in, u_in, A_in, A_exit, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, False)
            exit_state = result["exit_state"]
            assert np.isfinite(list(exit_state.values())).all()
            assert exit_state["rho"] > 0.0
            assert exit_state["u"] > 0.0
            assert exit_state["p"] > 0.0
            assert exit_state["T"] > 0.0
            assert exit_state["p"] < P_in, f"{name}/{fuel_name}: exit pressure did not drop"

    def test_le_pinn_exit_velocity_subsonic_or_sonic(self) -> None:
        for name, *_rest, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, False)
            if result["used_fallback"]:
                assert result["fallback_reason"]
                continue
            gamma = THERMO_CONFIGS[fuel_name]["gamma"]
            gas_constant = THERMO_CONFIGS[fuel_name]["R"]
            exit_state = result["exit_state"]
            mach = exit_state["u"] / math.sqrt(gamma * gas_constant * exit_state["T"])
            assert mach > 0.05, f"{name}/{fuel_name}: exit Mach is trivially small"
            assert mach <= 1.05, f"{name}/{fuel_name}: exit Mach {mach:.3f} exceeds sonic tolerance"

    def test_le_pinn_temperature_drop(self) -> None:
        for name, _npr, T_in, _P_in, _u_in, _A_in, _A_exit, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, True)
            if result["used_fallback"]:
                assert result["fallback_reason"]
                continue
            profile_T = np.asarray(result["profiles"]["T"], dtype=float)
            assert _count_upward_steps(profile_T) <= 3, f"{name}/{fuel_name}: temperature profile is too non-monotone"
            assert profile_T[-1] < T_in * 0.99, f"{name}/{fuel_name}: exit temperature did not fall by 1%"
            assert abs(profile_T[0] - T_in) / T_in < 0.05, f"{name}/{fuel_name}: inlet temperature recovery is poor"

    def test_le_pinn_pressure_drop(self) -> None:
        for name, _npr, _T_in, P_in, _u_in, _A_in, _A_exit, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, True)
            if result["used_fallback"]:
                assert result["fallback_reason"]
                continue
            profile_p = np.asarray(result["profiles"]["p"], dtype=float)
            assert _count_upward_steps(profile_p) <= 3, f"{name}/{fuel_name}: pressure profile is too non-monotone"
            assert profile_p[-1] < P_in * 0.70, f"{name}/{fuel_name}: exit pressure drop is too small"
            assert abs(profile_p[0] - P_in) / P_in < 0.10, f"{name}/{fuel_name}: inlet pressure recovery is poor"


class TestConservationLaws:
    def test_mass_conservation_le_pinn(self) -> None:
        for name, *_rest, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, True)
            max_error = float(result["mass_conservation"]["max_error"])
            if result["used_fallback"]:
                assert max_error >= 0.0
                assert result["fallback_reason"]
            else:
                assert 0.0 < max_error < 0.50, f"{name}/{fuel_name}: raw mass error {max_error:.3f} out of range"

    def test_energy_conservation_le_pinn(self) -> None:
        for name, *_rest, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, True)
            if result["used_fallback"]:
                continue
            thermo = THERMO_CONFIGS[fuel_name]
            profile = result["profiles"]
            H0 = thermo["cp"] * np.asarray(profile["T"], dtype=float) + 0.5 * np.asarray(profile["u"], dtype=float) ** 2
            drift = np.max(np.abs(H0 - H0[0]) / max(abs(H0[0]), 1e-12))
            assert drift < 0.05, f"{name}/{fuel_name}: total enthalpy drift {drift:.3f} exceeds tolerance"

    def test_eos_consistency_le_pinn(self) -> None:
        for name, *_rest, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, True)
            if result["used_fallback"]:
                continue
            thermo = THERMO_CONFIGS[fuel_name]
            profile = result["profiles"]
            p = np.asarray(profile["p"], dtype=float)
            rho = np.asarray(profile["rho"], dtype=float)
            T = np.asarray(profile["T"], dtype=float)
            eos_residual = np.abs(p - rho * thermo["R"] * T) / np.maximum(np.abs(p), 1e-12)
            assert float(eos_residual.mean()) < 0.10, f"{name}/{fuel_name}: mean EOS residual too large"
            assert float(eos_residual.max()) < 0.30, f"{name}/{fuel_name}: max EOS residual too large"


class TestNonFalsePosGuards:
    def test_fallback_rate_acceptable(self) -> None:
        fallback_count = 0
        for name, *_ in INLET_CONDITIONS:
            result = _run_le_cached(name, "JetA1", False)
            fallback_count += int(result["used_fallback"])
        fallback_rate = fallback_count / len(INLET_CONDITIONS)
        regular_rate = _regular_fallback_rate_jet_a1()
        assert fallback_rate <= regular_rate + 0.20 + 1e-12, (
            f"LE fallback rate {fallback_rate:.2f} is materially worse than "
            f"regular PINN fallback rate {regular_rate:.2f}"
        )
        assert fallback_rate > 0.0, "Fallback path was never exercised"

    def test_ood_does_not_silently_pass(self) -> None:
        for condition_name in ("ood_hot", "ood_cool"):
            for fuel_name in THERMO_CONFIGS:
                result = _run_le_cached(condition_name, fuel_name, False)
                exit_state = result["exit_state"]
                assert np.isfinite(list(exit_state.values())).all(), f"{condition_name}/{fuel_name}: exit state contains NaN/Inf"
                if result["used_fallback"]:
                    assert result["fallback_reason"], f"{condition_name}/{fuel_name}: fallback reason missing"
                else:
                    assert all(value > 0.0 for value in exit_state.values())

    def test_thrust_greater_than_zero(self) -> None:
        for name, *_rest, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, False)
            assert result["thrust_total"] > 500.0, f"{name}/{fuel_name}: total thrust is too small"
            assert result["thrust_momentum"] > 0.0, f"{name}/{fuel_name}: momentum thrust is not positive"

    def test_inlet_not_identically_reproduced(self) -> None:
        for name, _npr, T_in, P_in, u_in, _A_in, _A_exit, fuel_name in _iter_condition_fuel_pairs():
            result = _run_le_cached(name, fuel_name, False)
            if result["used_fallback"]:
                continue
            inlet = make_inlet(T_in, P_in, u_in, R=THERMO_CONFIGS[fuel_name]["R"])
            exit_state = result["exit_state"]
            changed_variables = 0
            for key in ("rho", "u", "p", "T"):
                rel_delta = abs(exit_state[key] - inlet[key]) / max(abs(inlet[key]), 1e-12)
                if rel_delta >= 0.05:
                    changed_variables += 1
            assert changed_variables >= 2, f"{name}/{fuel_name}: exit state is too close to inlet"

    def test_wrapper_matches_interface(self) -> None:
        result = _run_le_cached("nominal", "JetA1", False)
        assert set(result.keys()) == {
            "exit_state",
            "thrust_total",
            "thrust_momentum",
            "thrust_pressure",
            "thrust_model",
            "used_fallback",
            "fallback_reason",
            "inlet_verification",
            "mass_conservation",
        }
        assert set(result["exit_state"].keys()) == {"rho", "u", "p", "T"}


class TestComparisonWithRegularPINN:
    def test_thrust_within_order_of_magnitude(self) -> None:
        le_result = _run_le_cached("nominal", "JetA1", False)
        reg_result = _run_regular_nominal_cached()
        relative_error = abs(le_result["thrust_total"] - reg_result["thrust_total"]) / max(abs(reg_result["thrust_total"]), 1e-12)
        assert relative_error < 0.50, f"Thrust disagreement {relative_error:.3f} exceeds tolerance"

    def test_exit_temperature_within_bounds(self) -> None:
        le_result = _run_le_cached("nominal", "JetA1", False)
        reg_result = _run_regular_nominal_cached()
        relative_error = abs(le_result["exit_state"]["T"] - reg_result["exit_state"]["T"]) / max(abs(reg_result["exit_state"]["T"]), 1e-12)
        assert relative_error < 0.30, f"Exit temperature disagreement {relative_error:.3f} exceeds tolerance"

    def test_models_disagree_at_least_slightly(self) -> None:
        le_result = _run_le_cached("nominal", "JetA1", False)
        reg_result = _run_regular_nominal_cached()
        if le_result["used_fallback"] and reg_result["used_fallback"]:
            pytest.skip("Both models used fallback on the nominal case.")
        relative_difference = abs(le_result["thrust_total"] - reg_result["thrust_total"]) / max(abs(reg_result["thrust_total"]), 1e-12)
        assert relative_difference > 1e-4, "Models are numerically identical, which is suspicious"
