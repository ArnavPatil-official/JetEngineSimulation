"""
Phase 8 (P8.1): C++ backend for the v6 thrust-matched cycle.

``CppEngine`` stands in for ``integrated_engine.IntegratedTurbofanEngine`` in
``scripts/optimization/lto_v5.solve_task``: it exposes the attributes that
function sets (``design_point``, ``compressor.eta_c``,
``turbine_design['eta_polytropic']``) and ``run_at_thrust``, and forwards the
solve to the C++ core (``cpp/build/catjet_core``, built by ``cpp/build.sh``).
Design-point defaults, the fuel composition string and the NOx correlation
coefficients are taken from the Python objects, so both backends start from
identical inputs. The Python v6 path itself is unchanged (protected).

Failure semantics match the Python engine: an unreachable target raises
``integrated_engine.ThrustTargetUnreachable`` with the same reason string;
any other error propagates.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "cpp" / "build"


def load_core():
    """Import the compiled module from cpp/build (raises ImportError if not built)."""
    if str(BUILD) not in sys.path:
        sys.path.insert(0, str(BUILD))
    import catjet_core
    return catjet_core


class CppEngine:
    """Drop-in for IntegratedTurbofanEngine on the v6 path (analytic turbine and nozzle)."""

    def __init__(self, nox_fit_exclude_models=None, mechanism: str = "data/creck_c1c16_full.yaml"):
        from integrated_engine import EmissionsEstimator, IntegratedTurbofanEngine
        core = load_core()
        with contextlib.redirect_stdout(io.StringIO()):
            ref = IntegratedTurbofanEngine(creck_mechanism_path=mechanism)
            self.emissions = EmissionsEstimator(
                nox_fit_exclude_models=set(nox_fit_exclude_models) if nox_fit_exclude_models else None)
        self.design_point = dict(ref.design_point)
        self.compressor = SimpleNamespace(eta_c=ref.compressor.eta_c)
        self.turbine_design = dict(ref.turbine_design)
        self.mechanism = str((ROOT / mechanism).resolve())
        self.core = core.V6Engine(self.mechanism)

    def push_config(self) -> None:
        """Copy the live Python-side settings into the C++ config (every call)."""
        dp, c = self.design_point, self.core.config
        c.mass_flow_core = float(dp["mass_flow_core"])
        c.bypass_ratio = float(dp.get("bypass_ratio", 0.0))
        c.fpr = float(dp.get("fpr", 1.45))
        c.eta_fan = float(dp.get("eta_fan", 0.90))
        c.pi_c = float(dp["pi_c"])
        c.combustor_pressure_loss = float(dp.get("combustor_pressure_loss", 0.0))
        c.combustor_heat_loss_fraction = float(dp.get("combustor_heat_loss_fraction", 0.0))
        c.combustor_air_fraction = float(dp.get("combustor_air_fraction", 1.0))
        c.A_combustor_exit = float(dp["A_combustor_exit"])
        c.A_nozzle_exit = float(dp["A_nozzle_exit"])
        c.P_ambient = float(dp["P_ambient"])
        c.T_ambient = float(dp["T_ambient"])
        c.eta_c = float(self.compressor.eta_c)
        c.eta_polytropic = float(self.turbine_design["eta_polytropic"])
        c.nox_A = float(self.emissions.nox_A)
        c.nox_B = float(self.emissions.nox_B)
        c.nox_C = float(self.emissions.nox_C)
        self.core.config = c

    @staticmethod
    def fuel_args(fuel_blend) -> tuple[str, list[str]]:
        return fuel_blend.as_composition_string(), list(fuel_blend.composition.keys())

    def run_full_cycle(self, fuel_blend, phi: float, combustor_efficiency: float) -> dict:
        self.push_config()
        fuel, species = self.fuel_args(fuel_blend)
        return self.core.run_full_cycle(fuel, species, float(phi), float(combustor_efficiency))

    def run_at_thrust(self, target_kN, fuel_blend, combustor_efficiency=None, phi_bounds=None,
                      t4_max_K=None, phi_xtol: float = 1e-12, phi_guess=None) -> dict:
        from integrated_engine import PHI_SOLVE_BOUNDS, T4_GUARD_K, ThrustTargetUnreachable
        if combustor_efficiency is None:
            raise NotImplementedError("the C++ v6 path needs an explicit combustor_efficiency (eta_b)")
        lo, hi = phi_bounds if phi_bounds is not None else PHI_SOLVE_BOUNDS
        t4 = T4_GUARD_K if t4_max_K is None else t4_max_K
        self.push_config()
        fuel, species = self.fuel_args(fuel_blend)
        r = self.core.run_at_thrust(float(target_kN), fuel, species, float(combustor_efficiency),
                                    float(lo), float(hi), float(t4), float(phi_xtol), phi_guess)
        if r["status"] == "unreachable":
            info = dict(r["info"])
            info["phi_bounds"] = (float(lo), float(hi))
            raise ThrustTargetUnreachable(r["reason"], target_kN, info)
        return r
