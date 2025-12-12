import cantera as ct
import numpy as np
from typing import Dict, Any


class Combustor:
    """
    Constant-pressure combustor model using Cantera.

    - Uses HP equilibrium to compute the ideal adiabatic flame state.
    - Applies a combustion efficiency factor N_cmb to scale the ideal
      temperature rise (models incomplete combustion / losses).

    The combustor is mechanism-agnostic: it can use any Cantera-compatible
    mechanism file (CRECK, HyChem, GRI-Mech, etc.). The choice of mechanism
    should be made by the calling code based on the simulation purpose.
    """

    def __init__(self, mechanism_file: str):
        """
        Initialize the Combustor with a Cantera reaction mechanism file path.

        Args:
            mechanism_file: Path to the Cantera mechanism file.
                           Examples: 'data/creck_c1c16_full.yaml' (for blend studies),
                                    'data/A1highT.yaml' (for HyChem Jet-A1 validation),
                                    'gri30.yaml' (for simple methane combustion)
        """
        self.mechanism_file = mechanism_file
        try:
            ct.Solution(mechanism_file)  # Validate mechanism file on initialization
        except Exception as e:
            raise RuntimeError(f"Failed to load Cantera mechanism '{mechanism_file}': {e}")

    def run(
        self,
        T_in: float,
        p_in: float,
        fuel_blend,
        phi: float,
        efficiency: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run the combustor model for a given inlet state, fuel blend, and equivalence ratio.

        Args:
            T_in: Inlet temperature [K] (from compressor).
            p_in: Inlet pressure [Pa] (from compressor).
            fuel_blend: FuelSurrogate or string; if object, must implement .as_composition_string().
            phi: Equivalence ratio.
            efficiency: Combustion efficiency N_cmb, scaling the ideal temperature rise (0–1).

        Returns:
            dict with:
                T_out, p_out, h_out, Y_out, cp_out, R_out, gamma_out
        """
        # 1. Fuel composition string
        try:
            fuel_string = fuel_blend.as_composition_string()
        except AttributeError:
            if isinstance(fuel_blend, str):
                fuel_string = fuel_blend
            else:
                raise ValueError(
                    "fuel_blend must be a composition string or implement .as_composition_string()."
                )

        # 2. Set up equilibrium gas (ideal adiabatic flame)
        gas_eq = ct.Solution(self.mechanism_file)
        gas_eq.TP = T_in, p_in
        gas_eq.set_equivalence_ratio(phi, fuel=fuel_string, oxidizer="O2:1.0, N2:3.76")

        # Ideal adiabatic equilibrium (HP)
        gas_eq.equilibrate("HP")
        T_ideal = gas_eq.T
        Y_ideal = gas_eq.Y

        # 3. Apply combustion efficiency to temperature rise
        #    T_out = T_in + N_cmb * (T_ideal - T_in)
        T_out = T_in + efficiency * (T_ideal - T_in)

        # 4. Set final state with same composition but reduced temperature
        gas_out = ct.Solution(self.mechanism_file)
        gas_out.TPY = T_out, p_in, Y_ideal

        # 5. Package outputs including gamma
        cp_out = gas_out.cp_mass
        R_out = ct.gas_constant / gas_out.mean_molecular_weight
        cv_out = cp_out - R_out
        gamma_out = cp_out / cv_out

        return {
            "T_out": gas_out.T,
            "p_out": gas_out.P,
            "h_out": gas_out.enthalpy_mass,
            "Y_out": gas_out.Y,
            "cp_out": cp_out,
            "R_out": R_out,
            "gamma_out": gamma_out,
        }
