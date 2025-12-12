"""
Compressor Model for Jet Engine Simulation.

This module implements a thermodynamic model of an axial compressor using Cantera
for accurate thermodynamic property evaluation. The model uses isentropic relations
with an efficiency correction to account for real-world losses.

Model Approach:
1. Calculate ideal isentropic compression (constant entropy)
2. Apply efficiency factor to account for irreversibilities
3. Calculate actual work requirement from enthalpy change

This provides realistic compression behavior without requiring detailed blade geometry
or CFD simulation.
"""

import cantera as ct

class Compressor:
    """
    Axial compressor model using isentropic compression with efficiency losses.

    The compressor raises air pressure through a series of rotating and stationary
    blades. Real compressors have losses due to friction, turbulence, and blade
    inefficiencies, captured here through the efficiency parameter eta_c.

    Attributes:
        eta_c: Compressor isentropic efficiency (0 < eta_c ≤ 1)
               Typical values: 0.80-0.90 for modern compressors
        pi_c: Overall pressure ratio (p_out / p_in)
              Typical values: 15-50 for high-bypass turbofans
        gas: Cantera Solution object for thermodynamic property evaluation
    """

    def __init__(self, gas: ct.Solution, eta_c: float = 0.85, pi_c: float = 10.0):
        """
        Initialize compressor model with thermodynamic properties and performance parameters.

        Args:
            gas: Cantera Solution object for working fluid property evaluation
            eta_c: Isentropic efficiency (default 0.85)
            pi_c: Overall pressure ratio (default 10.0)
        """
        self.gas = gas
        self.eta_c = eta_c
        self.pi_c = pi_c

    def compute_outlet_state(self, T_in: float, p_in: float):
        """
        Calculate compressor outlet conditions using isentropic relations with efficiency correction.

        The calculation proceeds in three steps:
        1. Find ideal isentropic outlet state (constant entropy to target pressure)
        2. Apply efficiency correction to get real outlet temperature
        3. Calculate work requirement from enthalpy rise

        Args:
            T_in: Inlet temperature [K]
            p_in: Inlet pressure [Pa]

        Returns:
            Dictionary containing:
                - T_out: Outlet temperature [K]
                - p_out: Outlet pressure [Pa]
                - h_in: Inlet specific enthalpy [J/kg]
                - h_out: Outlet specific enthalpy [J/kg]
                - work_specific: Specific work input [J/kg]
        """
        # Set inlet state and store entropy
        self.gas.TP = T_in, p_in
        s_in = self.gas.entropy_mass

        # Calculate ideal isentropic compression to target pressure
        # (constant entropy, s_out = s_in)
        p_out = p_in * self.pi_c
        self.gas.SP = s_in, p_out
        T_out_ideal = self.gas.T

        # Apply efficiency correction to account for real losses
        # Real temperature rise is larger than ideal due to irreversibilities
        T_out = T_in + (T_out_ideal - T_in) / self.eta_c

        # Calculate work requirement from enthalpy change
        self.gas.TP = T_in, p_in
        h_in = self.gas.enthalpy_mass
        self.gas.TP = T_out, p_out
        h_out = self.gas.enthalpy_mass

        return {
            "T_out": T_out,
            "p_out": p_out,
            "h_in": h_in,
            "h_out": h_out,
            "work_specific": h_out - h_in,  # Specific work input [J/kg]
        }


    def summary(self, T_in: float, p_in: float):
        """
        Print formatted summary of compressor performance for diagnostics.

        Args:
            T_in: Inlet temperature [K]
            p_in: Inlet pressure [Pa]

        Returns:
            Dictionary with outlet state and performance metrics
        """
        result = self.compute_outlet_state(T_in, p_in)
        print(f"[Compressor]")
        print(f"  Inlet : T={T_in:.1f} K, p={p_in/1e5:.3f} bar")
        print(f"  Outlet: T={result['T_out']:.1f} K, p={result['p_out']/1e5:.3f} bar")
        print(f"  Work  : {result['work_specific']/1e3:.2f} kJ/kg")
        print(f"  pi_c={self.pi_c:.2f}, eta_c={self.eta_c:.2f}")

        return result


if __name__ == "__main__":
    # Example: Compress ambient air with typical high-bypass turbofan parameters
    gas = ct.Solution('air.yaml')
    comp = Compressor(gas, eta_c=0.86, pi_c=15.0)
    comp.summary(T_in=288.0, p_in=101325.0)  # ISA sea level conditions