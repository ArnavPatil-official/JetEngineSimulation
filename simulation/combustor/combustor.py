"""
Combustor Model Using Chemical Equilibrium.

This module implements a constant-pressure combustor using Cantera's chemical equilibrium
solver. The model calculates adiabatic flame temperature and equilibrium product composition
for arbitrary fuel-air mixtures, then extracts fuel-dependent thermodynamic properties.

Model Capabilities:
- Handles any fuel compatible with the loaded chemical mechanism
- Computes equilibrium combustion products (CO2, H2O, N2, etc.)
- Extracts fuel-dependent thermodynamic properties (cp, R, gamma) from products
- Applies efficiency factor to account for incomplete combustion

The fuel-dependent properties are critical: different fuels produce different combustion
products with different heat capacities, which directly affect downstream expansion processes.
"""

import cantera as ct
import numpy as np
from typing import Dict, Any


class Combustor:
    """
    Constant-pressure combustor using Cantera chemical equilibrium.

    The combustor solves for chemical equilibrium at constant enthalpy and pressure (HP)
    to find the adiabatic flame temperature and product composition. Real combustors have
    heat losses and incomplete combustion, captured by the efficiency parameter.

    Key Feature:
        Extracts fuel-dependent thermodynamic properties (cp, R, gamma) from the actual
        combustion product mixture. These properties vary significantly between fuels:
        - Jet-A1 combustion products: different cp, gamma than SAF combustion products
        - This variation propagates through the engine and affects performance
    """

    def __init__(self, mechanism_file: str):
        """
        Initialize combustor with a chemical kinetics mechanism.

        Args:
            mechanism_file: Path to Cantera-compatible mechanism YAML file
                           Examples:
                           - 'data/creck_c1c16_full.yaml': For multi-fuel comparisons
                           - 'data/A1highT.yaml': For high-fidelity Jet-A1 validation
        """
        self.mechanism_file = mechanism_file
        # Validate mechanism file exists and can be loaded
        try:
            ct.Solution(mechanism_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load Cantera mechanism '{mechanism_file}': {e}")

    @staticmethod
    def estimate_efficiency(phi: float, fuel_blend=None) -> float:
        """
        Estimate combustor efficiency based on equivalence ratio and fuel type.

        Efficiency peaks near stoichiometric (φ = 1.0) and drops as φ deviates:
            η = η_max − k_phi × (φ − 1.0)²

        An additional 1–1.5% penalty is applied for heavier SAF blends based
        on literature values for alternative fuel combustion characteristics.

        Args:
            phi: Equivalence ratio (lean < 1.0, stoichiometric = 1.0, rich > 1.0)
            fuel_blend: Optional fuel blend object with a `name` attribute.
                        Used to apply fuel-specific efficiency penalties.

        Returns:
            Estimated combustion efficiency in [0.90, 0.999]
        """
        eta_max = 0.995
        k_phi = 0.04  # quadratic penalty coefficient

        eta = eta_max - k_phi * (phi - 1.0) ** 2

        # SAF blend penalty (heavier / alternative molecules = slightly lower efficiency)
        if fuel_blend is not None:
            blend_name = getattr(fuel_blend, 'name', '')
            if 'Bio-SPK' in blend_name:
                eta -= 0.015  # 1.5% penalty for pure synthetic paraffinic kerosene
            elif 'HEFA' in blend_name:
                eta -= 0.01   # 1.0% penalty for blended HEFA

        return float(max(min(eta, 0.999), 0.90))

    def run(
        self,
        T_in: float,
        p_in: float,
        fuel_blend,
        phi: float,
        efficiency: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Simulate combustion and extract fuel-dependent thermodynamic properties.

        The combustion calculation proceeds in several steps:
        1. Set up fuel-air mixture at inlet conditions
        2. Solve for chemical equilibrium (adiabatic flame temperature)
        3. Apply efficiency factor to account for real combustor losses
        4. Extract thermodynamic properties from equilibrium product mixture

        Args:
            T_in: Inlet temperature from compressor [K]
            p_in: Inlet pressure from compressor [Pa]
            fuel_blend: FuelSurrogate object or composition string (e.g., "NC12H26:1.0")
            phi: Equivalence ratio (phi < 1 is lean, phi = 1 is stoichiometric)
            efficiency: Combustion efficiency accounting for incomplete combustion (0-1)
                       Typical value: 0.98 for modern combustors

        Returns:
            Dictionary containing outlet state and fuel-dependent properties:
                - T_out: Outlet temperature [K]
                - p_out: Outlet pressure [Pa]
                - h_out: Outlet specific enthalpy [J/kg]
                - Y_out: Mass fractions of product species
                - cp_out: Specific heat at constant pressure [J/(kg·K)] - FUEL-DEPENDENT
                - R_out: Specific gas constant [J/(kg·K)] - FUEL-DEPENDENT
                - gamma_out: Heat capacity ratio cp/cv - FUEL-DEPENDENT
        """
        # Convert fuel blend object to Cantera composition string
        try:
            fuel_string = fuel_blend.as_composition_string()
        except AttributeError:
            if isinstance(fuel_blend, str):
                fuel_string = fuel_blend
            else:
                raise ValueError(
                    "fuel_blend must be a composition string or implement .as_composition_string()."
                )

        # Set up fuel-air mixture at inlet conditions
        gas_eq = ct.Solution(self.mechanism_file)
        gas_eq.TP = T_in, p_in
        gas_eq.set_equivalence_ratio(phi, fuel=fuel_string, oxidizer="O2:1.0, N2:3.76")

        # Solve for chemical equilibrium at constant enthalpy and pressure
        # This finds the adiabatic flame temperature and product composition
        gas_eq.equilibrate("HP")
        T_ideal = gas_eq.T
        Y_ideal = gas_eq.Y  # Product mass fractions (CO2, H2O, N2, etc.)

        # Apply combustion efficiency to temperature rise
        # Real combustors have heat losses and incomplete combustion
        T_out = T_in + efficiency * (T_ideal - T_in)

        # Set outlet state with equilibrium composition at efficiency-corrected temperature
        gas_out = ct.Solution(self.mechanism_file)
        gas_out.TPY = T_out, p_in, Y_ideal

        # Extract fuel-dependent thermodynamic properties from product mixture
        # These properties vary with fuel because different fuels produce different
        # combustion products (e.g., different H2O/CO2 ratios)
        cp_out = gas_out.cp_mass  # Specific heat [J/(kg·K)]
        R_out = ct.gas_constant / gas_out.mean_molecular_weight  # Gas constant [J/(kg·K)]
        cv_out = cp_out - R_out  # Specific heat at constant volume [J/(kg·K)]
        gamma_out = cp_out / cv_out  # Heat capacity ratio (typically 1.3-1.35 for combustion products)

        return {
            "T_out": gas_out.T,
            "p_out": gas_out.P,
            "h_out": gas_out.enthalpy_mass,
            "Y_out": gas_out.Y,
            "cp_out": cp_out,
            "R_out": R_out,
            "gamma_out": gamma_out,
        }
