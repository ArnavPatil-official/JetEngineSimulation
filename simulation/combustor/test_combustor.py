import sys
import os
from pathlib import Path

import cantera as ct
import numpy as np

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from simulation.combustor.combustor import Combustor
from simulation.fuels import JET_A1, make_saf_blend


def test_combustor_saf_blends():
    print("\n=== Combustor Test: Jet-A1 and SAF Blends ===")

    # Locate the CRECK mechanism (adjust path if your layout is different)
    project_root = Path(__file__).resolve().parents[2]
    mech_path = project_root / "data" / "creck_c1c16_full.yaml"
    mech_file = str(mech_path)

    if not mech_path.exists():
        print(f"Mechanism file not found at: {mech_path}")
        print("Please check the path to 'creck_c1c16_full.yaml'.")
        return

    print(f"Using mechanism: {mech_file}")
    combustor = Combustor(mechanism_file=mech_file)

    # Test inlet conditions (typical compressor outlet ballpark)
    T_in = 900.0            # K
    p_in = 15.0 * 101325.0  # 15 atm
    phi = 0.35              # lean, realistic for modern turbofans
    efficiency = 0.98       # high, but less than ideal

    # Define test fuels: Jet-A1 and several SAF blends
    test_cases = [
        ("Jet-A1 only", JET_A1),
        ("Jet-A1 + HEFA", make_saf_blend(p_j=0.7, p_h=0.3, p_f=0.0, p_a=0.0)),
        ("Jet-A1 + FT",   make_saf_blend(p_j=0.7, p_h=0.0, p_f=0.3, p_a=0.0)),
        ("Jet-A1 + ATJ",  make_saf_blend(p_j=0.7, p_h=0.0, p_f=0.0, p_a=0.3)),
        ("Mixed SAF",     make_saf_blend(p_j=0.6, p_h=0.2, p_f=0.1, p_a=0.1)),
    ]

    for label, fuel_obj in test_cases:
        print(f"\n--- Case: {label} ---")
        print(f"  T_in={T_in} K, p_in={p_in/101325:.1f} atm, phi={phi}, eff={efficiency}")

        outputs = combustor.run(
            T_in=T_in,
            p_in=p_in,
            fuel_blend=fuel_obj,
            phi=phi,
            efficiency=efficiency,
        )

        T_out = outputs["T_out"]
        p_out = outputs["p_out"]
        h_out = outputs["h_out"]

        print(f"  T_out = {T_out:.2f} K")
        print(f"  p_out = {p_out:.2f} Pa")
        print(f"  h_out = {h_out:.2f} J/kg")

        # ---- Verification checks ----
        # 1. Flame temperature must exceed inlet temperature
        assert T_out > T_in, f"FAIL ({label}): Flame temperature did not rise!"

        # 2. Pressure should remain (numerically) constant
        assert np.isclose(p_out, p_in, rtol=1e-6), f"FAIL ({label}): Pressure not constant!"

        # 3. Reasonable upper bound on flame temperature
        assert T_out < 3000, f"FAIL ({label}): Temperature too high (unphysical)!"

    print("\n✓ All combustor SAF blend tests passed!")


if __name__ == "__main__":
    test_combustor_saf_blends()
