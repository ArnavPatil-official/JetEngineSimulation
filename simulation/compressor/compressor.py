# Subsetting Data in Train/Test Sets

import cantera as ct
from pathlib import Path
import os
# Directory containing the YAML file (absolute path)
# def_path = Path("/Users/arnavpatil/Desktop/JetEngineSimulation/data/processed")


# # Ensure CANTERA_DATA points to the directory containing the file
# os.environ["CANTERA_DATA"] = str(def_path)

# def fuel_used(sol: str) -> str:
#     return str(def_path / sol)

# mixture = fuel_used("n_dodecane_hychem.yaml")

class Compressor:
    """
    Compressor model (isentropic with efficiency correction).

    Attributes
    ----------
    eta_c : float
        Compressor efficiency (0 < eta_c ≤ 1)
    pi_c : float
        Pressure ratio (p_out / p_in)
    gas : ct.Solution
        Cantera gas object representing working fluid
    """

    def __init__(self, gas: ct.Solution, eta_c: float = 0.85, pi_c: float = 10.0):
        """
        Initialize the compressor with a Cantera gas object and parameters.
        """
        self.gas = gas
        self.eta_c = eta_c
        self.pi_c = pi_c

    # Core methods

    def compute_outlet_state(self, T_in: float, p_in: float):
      # Inlet state
      self.gas.TP = T_in, p_in
      s_in = self.gas.entropy_mass

      # Ideal isentropic outlet at target pressure
      p_out = p_in * self.pi_c
      self.gas.SP = s_in, p_out
      T_out_ideal = self.gas.T

      # Efficiency-corrected outlet temperature
      T_out = T_in + (T_out_ideal - T_in) / self.eta_c

      # Enthalpy change (J/kg)
      self.gas.TP = T_in, p_in
      h_in = self.gas.enthalpy_mass
      self.gas.TP = T_out, p_out
      h_out = self.gas.enthalpy_mass

      return {
          "T_out": T_out,
          "p_out": p_out,
          "h_in": h_in,
          "h_out": h_out,
          "work_specific": h_out - h_in,  # ~ +400 kJ/kg for PR~15, eta~0.86
    }


    # Utility methods

    def summary(self, T_in: float, p_in: float):
        """
        Print a formatted summary of compressor performance.
        """
        result = self.compute_outlet_state(T_in, p_in)
        print(f"[Compressor]")
        print(f"  Inlet : T={T_in:.1f} K, p={p_in/1e5:.3f} bar")
        print(f"  Outlet: T={result['T_out']:.1f} K, p={result['p_out']/1e5:.3f} bar")
        print(f"  Work  : {result['work_specific']/1e3:.2f} kJ/kg")
        print(f"  pi_c={self.pi_c:.2f}, eta_c={self.eta_c:.2f}")

        return result


# Example
if __name__ == "__main__":
    gas = ct.Solution('air.yaml')
    comp = Compressor(gas, eta_c=0.86, pi_c=15.0)
    comp.summary(T_in=288.0, p_in=101325.0)