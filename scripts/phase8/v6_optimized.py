"""Isolated optimized-Python v6 benchmark arm.

The protected v6 sources are untouched.  The P8.0 profile identified
``_calculate_fuel_air_ratio`` as repeated work on the steady path.  For a
fixed fuel and oxidizer, Cantera's fuel/air mass ratio is linear in phi; this
arm asks the original Cantera routine for the stoichiometric ratio once per
fuel and uses ``phi * FAR_stoich`` thereafter.  The equilibrium state is still
reset and solved exactly as in v6 on every cycle evaluation.

Variant b uses the registered cold-solve bracket improvement: for a mode
whose cycle closes above a conservative probe phi, try a narrower bracket.
If the probe excludes the solution or a feasibility bound, the original
solver supplies the result.  The cycle function and Brent tolerances remain
those of v6.  This is deliberately separate from the protected engine.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cantera as ct
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
for path in (ROOT, ROOT / "scripts" / "optimization"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lto_v5 as v5  # noqa: E402


PRODUCT_SPECIES = ("N2", "O2", "AR", "CO2", "H2O", "CO", "H2", "OH", "H", "O", "HO2", "H2O2")


class ProductsOnlyCombustor:
    """CRECK-thermo equilibrium constrained to P8-A1's 12 products.

    The reactant mixture remains on the full mechanism.  Its enthalpy and
    elemental mole fractions initialise a product-only ideal-gas phase at the
    same pressure; HP equilibrium then uses just the registered product set.
    Thermodynamic outlet properties are extracted on the original full phase.
    """

    def __init__(self, mechanism_file: str):
        from simulation.combustor.combustor import Combustor

        self.full = Combustor(mechanism_file)
        full = ct.Solution(mechanism_file)
        self.products = ct.Solution(thermo="ideal-gas",
                                    species=[full.species(name) for name in PRODUCT_SPECIES])
        self.products_pristine = self.products.state.copy()
        self.full_indices = [full.species_index(name) for name in PRODUCT_SPECIES]
        self.full_species_count = full.n_species

    def run(self, T_in, p_in, fuel_blend, phi, efficiency=1.0,
            heat_loss_fraction=0.0):
        if not 0.0 <= heat_loss_fraction < 1.0:
            raise ValueError(f"heat_loss_fraction must be in [0, 1), got {heat_loss_fraction}")
        fuel_string = fuel_blend.as_composition_string()
        gas_in, gas_out = self.full._fresh_solutions()
        gas_in.TP = T_in, p_in
        gas_in.set_equivalence_ratio(phi, fuel=fuel_string, oxidizer="O2:1.0, N2:3.76")
        h_in = gas_in.enthalpy_mass
        z = {element: gas_in.elemental_mole_fraction(element)
             for element in ("C", "H", "O", "N", "Ar")}
        co2 = z["C"]
        h2o = z["H"] / 2.0
        o2 = (z["O"] - 2.0 * co2 - h2o) / 2.0
        if o2 < -1e-12:
            raise ValueError(f"products-only initial mixture has negative O2 at phi={phi}")
        initial = {"CO2": co2, "H2O": h2o, "O2": max(o2, 0.0),
                   "N2": z["N"] / 2.0, "AR": z["Ar"]}
        self.products.state = self.products_pristine
        self.products.TPX = T_in, p_in, initial
        self.products.HP = h_in, p_in
        self.products.equilibrate("HP")
        t_ideal = self.products.T
        t_out = T_in + efficiency * (1.0 - heat_loss_fraction) * (t_ideal - T_in)
        full_y = np.zeros(self.full_species_count)
        full_y[self.full_indices] = self.products.Y
        gas_out.TPY = t_out, p_in, full_y
        cp = gas_out.cp_mass
        r = ct.gas_constant / gas_out.mean_molecular_weight
        return {"T_out": gas_out.T, "p_out": gas_out.P,
                "h_out": gas_out.enthalpy_mass, "Y_out": gas_out.Y,
                "cp_out": cp, "R_out": r, "gamma_out": cp / (cp - r)}


def _optimized_engine(variant: str):
    from integrated_engine import IntegratedTurbofanEngine, ThrustTargetUnreachable

    class OptimizedEngine(IntegratedTurbofanEngine):
        def __init__(self):
            super().__init__()
            self._stoich_far: dict[tuple[tuple[str, float], ...], float] = {}
            if variant == "c":
                self.combustor_creck = ProductsOnlyCombustor(self.creck_mech)

        def _calculate_fuel_air_ratio(self, fuel_blend, phi):
            key = tuple(sorted(fuel_blend.composition.items()))
            if key not in self._stoich_far:
                self._stoich_far[key] = super()._calculate_fuel_air_ratio(fuel_blend, 1.0)
            return phi * self._stoich_far[key]

        def run_at_thrust(self, target_kN, fuel_blend, combustor_efficiency=None,
                          phi_bounds=(0.05, 1.0), t4_max_K=None, phi_xtol=1e-12,
                          phi_guess=None, **cycle_kwargs):
            if variant not in ("b", "c") or phi_guess is not None or tuple(phi_bounds) != (0.05, 1.0):
                return super().run_at_thrust(
                    target_kN, fuel_blend, combustor_efficiency, phi_bounds,
                    t4_max_K=t4_max_K if t4_max_K is not None else 3800.0 * 5.0 / 9.0,
                    phi_xtol=phi_xtol, phi_guess=phi_guess, **cycle_kwargs)
            # Rated/approach points spend many v6 evaluations locating cycle
            # closure below the root.  A failed narrower solve is only a probe;
            # the original cold bracket then determines the result and reason.
            pi_c = float(self.design_point["pi_c"])
            probe = 0.20 if pi_c >= 20.0 else (0.10 if pi_c >= 5.0 else None)
            if probe is not None:
                try:
                    result = super().run_at_thrust(
                        target_kN, fuel_blend, combustor_efficiency,
                        (probe, phi_bounds[1]),
                        t4_max_K=t4_max_K if t4_max_K is not None else 3800.0 * 5.0 / 9.0,
                        phi_xtol=phi_xtol, phi_guess=None, **cycle_kwargs)
                    result["thrust_match"]["phi_bounds"] = tuple(phi_bounds)
                    return result
                except ThrustTargetUnreachable:
                    pass
            return super().run_at_thrust(
                target_kN, fuel_blend, combustor_efficiency, phi_bounds,
                t4_max_K=t4_max_K if t4_max_K is not None else 3800.0 * 5.0 / 9.0,
                phi_xtol=phi_xtol, phi_guess=None, **cycle_kwargs)

    return OptimizedEngine()


def init_worker_optimized(nox_fit_exclude_models, variant="a"):
    """Install this arm's engine in the unchanged ``lto_v5.solve_task`` path."""
    from integrated_engine import EmissionsEstimator

    if variant not in ("a", "b", "c"):
        raise ValueError(f"unimplemented optimized-Python variant {variant!r}")
    logging.getLogger("cantera").setLevel(logging.ERROR)
    os.chdir(ROOT)
    with contextlib.redirect_stdout(io.StringIO()):
        v5._ENGINE = _optimized_engine(variant)
        v5._ENGINE.emissions = EmissionsEstimator(
            nox_fit_exclude_models=set(nox_fit_exclude_models))
    if v5._ENGINE.emissions.nox_fit_exclude_models != set(nox_fit_exclude_models):
        raise RuntimeError("NOx held-out exclusion not applied")


class OptimizedModel(v5.V5Model):
    """V5Model's task creation/scoring with an isolated optimized worker pool."""

    def __init__(self, fixed, nox_fit_exclude_models, n_workers=1, fuel="Jet-A1", variant="a"):
        if fixed.get("eta_b") is None or not nox_fit_exclude_models:
            raise ValueError("fixed eta_b and held-out NOx exclusions are required")
        self.fixed = fixed
        self.fuel = fuel
        self.guess = {}
        self.n_evals = 0
        self.nox_fit_exclude_models = sorted(nox_fit_exclude_models)
        self.pool = ProcessPoolExecutor(
            max_workers=n_workers, initializer=init_worker_optimized,
            initargs=(self.nox_fit_exclude_models, variant))
        self.variant = variant
