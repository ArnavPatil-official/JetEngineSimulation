"""
Minimal test script for nozzle PINN v3.0 integration fix.

This script validates that the nozzle PINN produces positive thrust
with representative turbine exit conditions, without requiring the
ICAO CSV database.

Tests:
1. Load nozzle_pinn.pt checkpoint
2. Call run_nozzle_pinn() with realistic turbine exit state
3. Assert thrust_total > 0
4. Verify exit velocity > inlet velocity
5. Check TSFC and efficiency are finite

Run with: python test_nozzle_pinn_fix.py
"""

import sys
from pathlib import Path
import numpy as np

# Add simulation modules to path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.nozzle.nozzle import run_nozzle_pinn


def test_nozzle_pinn_positive_thrust():
    """
    Test that nozzle PINN produces positive thrust for realistic conditions.

    This test uses representative turbine exit conditions similar to those
    encountered in the integrated engine simulation:
    - High temperature combustion products (~2000 K)
    - Moderate pressure (~2 bar)
    - Subsonic inlet velocity (~470 m/s)
    - Fuel-dependent thermodynamic properties
    """
    print("="*70)
    print("TEST: Nozzle PINN v3.0 - Positive Thrust Validation")
    print("="*70)

    # Representative turbine exit state (from integrated engine logs)
    # These values are typical for a high-bypass turbofan after turbine expansion
    turbine_exit = {
        'rho': 0.67,      # kg/m³ - density from ideal gas law at exit conditions
        'u': 467.5,       # m/s - subsonic velocity entering nozzle
        'p': 200000.0,    # Pa (2.0 bar) - turbine exit pressure
        'T': 2062.0,      # K - high temperature combustion products
    }

    # Fuel-dependent thermodynamic properties (from combustion products)
    # These are extracted from Cantera equilibrium calculations for Jet-A1
    thermo_props = {
        'cp': 1384.0,     # J/(kg·K) - specific heat at constant pressure
        'R': 289.8,       # J/(kg·K) - specific gas constant for products
        'gamma': 1.265    # Heat capacity ratio (cp/cv)
    }

    # Operating conditions
    ambient_p = 101325.0  # Pa - sea level ambient pressure
    m_dot = 82.6          # kg/s - total mass flow (air + fuel)

    # Nozzle geometry (matches PINN training conditions)
    A_in = 0.375      # m² - inlet area
    A_exit = 0.340    # m² - exit area (converging nozzle)
    length = 1.0      # m - nozzle length

    print("\nInput Conditions:")
    print(f"  Turbine Exit: T={turbine_exit['T']:.1f} K, "
          f"p={turbine_exit['p']/1e3:.2f} kPa, u={turbine_exit['u']:.1f} m/s")
    print(f"  Thermo Props: cp={thermo_props['cp']:.1f}, "
          f"R={thermo_props['R']:.1f}, γ={thermo_props['gamma']:.3f}")
    print(f"  Mass Flow:    {m_dot:.2f} kg/s")
    print(f"  Ambient:      {ambient_p/1e3:.2f} kPa")

    # Run nozzle PINN
    print("\nRunning nozzle PINN...")
    try:
        result = run_nozzle_pinn(
            model_path='nozzle_pinn.pt',
            inlet_state=turbine_exit,
            ambient_p=ambient_p,
            A_in=A_in,
            A_exit=A_exit,
            length=length,
            thermo_props=thermo_props,
            m_dot=m_dot,
            device='cpu'
        )
    except Exception as e:
        print(f"\n❌ TEST FAILED: Nozzle PINN raised exception: {e}")
        return False

    # Extract results
    thrust_total = result['thrust_total']
    thrust_momentum = result['thrust_momentum']
    thrust_pressure = result['thrust_pressure']
    u_exit = result['exit_state']['u']
    T_exit = result['exit_state']['T']
    p_exit = result['exit_state']['p']

    print("\nResults:")
    print(f"  Exit State:   T={T_exit:.1f} K, p={p_exit/1e3:.2f} kPa, u={u_exit:.1f} m/s")
    print(f"  Thrust (momentum): {thrust_momentum/1e3:.2f} kN")
    print(f"  Thrust (pressure): {thrust_pressure/1e3:.2f} kN")
    print(f"  Thrust (total):    {thrust_total/1e3:.2f} kN")

    # === ASSERTIONS ===
    print("\nValidation Checks:")

    # Check 1: Positive total thrust
    if thrust_total <= 0:
        print(f"  ❌ FAILED: Thrust is non-positive ({thrust_total/1e3:.2f} kN)")
        return False
    else:
        print(f"  ✓ Thrust is positive: {thrust_total/1e3:.2f} kN")

    # Check 2: Exit velocity exceeds inlet velocity (nozzle accelerates flow)
    if u_exit <= turbine_exit['u']:
        print(f"  ❌ FAILED: Exit velocity ({u_exit:.1f} m/s) <= inlet ({turbine_exit['u']:.1f} m/s)")
        return False
    else:
        delta_u = u_exit - turbine_exit['u']
        print(f"  ✓ Flow accelerated: Δu = {delta_u:.1f} m/s")

    # Check 3: Momentum thrust should be positive (main thrust component)
    if thrust_momentum <= 0:
        print(f"  ❌ FAILED: Momentum thrust is non-positive ({thrust_momentum/1e3:.2f} kN)")
        return False
    else:
        print(f"  ✓ Momentum thrust positive: {thrust_momentum/1e3:.2f} kN")

    # Check 4: Exit temperature should be lower than inlet (expansion cools)
    if T_exit >= turbine_exit['T']:
        print(f"  ⚠️  WARNING: Exit temp ({T_exit:.1f} K) >= inlet ({turbine_exit['T']:.1f} K)")
    else:
        print(f"  ✓ Temperature dropped: {turbine_exit['T']:.1f} K → {T_exit:.1f} K")

    # Check 5: Calculate TSFC (should be finite)
    # Assuming fuel flow ~ 2.5% of total mass flow (typical FAR ~ 0.025)
    fuel_flow = 0.025 * m_dot  # kg/s
    tsfc = (fuel_flow * 3600) / thrust_total  # kg/(N·hr)
    tsfc_mg_per_Ns = tsfc * 1000  # mg/(N·s)

    if not np.isfinite(tsfc_mg_per_Ns):
        print(f"  ❌ FAILED: TSFC is not finite")
        return False
    else:
        print(f"  ✓ TSFC is finite: {tsfc_mg_per_Ns:.2f} mg/(N·s)")

    # Check 6: Calculate thermal efficiency (should be finite and reasonable)
    LHV = 43e6  # J/kg - lower heating value of jet fuel
    fuel_power = fuel_flow * LHV
    thrust_power = thrust_total * u_exit  # Approximate propulsive power
    eta_thermal = thrust_power / fuel_power

    if not np.isfinite(eta_thermal) or eta_thermal <= 0:
        print(f"  ❌ FAILED: Thermal efficiency is not finite or positive")
        return False
    else:
        print(f"  ✓ Thermal efficiency is finite: {eta_thermal*100:.2f}%")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Nozzle PINN produces positive thrust")
    print("="*70)

    return True


def test_nozzle_scaling_robustness():
    """
    Test that nozzle PINN handles different inlet conditions correctly.

    This verifies that the runtime scaling fix works across a range of
    operating conditions.
    """
    print("\n" + "="*70)
    print("TEST: Nozzle PINN v3.0 - Scaling Robustness")
    print("="*70)

    # Test cases with different inlet velocities and temperatures
    test_cases = [
        {
            'name': 'Low velocity',
            'rho': 0.8, 'u': 300.0, 'p': 180000.0, 'T': 1800.0,
            'cp': 1350.0, 'R': 288.0, 'gamma': 1.27
        },
        {
            'name': 'High velocity',
            'rho': 0.6, 'u': 550.0, 'p': 220000.0, 'T': 2200.0,
            'cp': 1400.0, 'R': 291.0, 'gamma': 1.26
        },
        {
            'name': 'Nominal',
            'rho': 0.67, 'u': 467.5, 'p': 200000.0, 'T': 2062.0,
            'cp': 1384.0, 'R': 289.8, 'gamma': 1.265
        }
    ]

    ambient_p = 101325.0
    m_dot = 82.6

    all_passed = True

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print(f"  Inlet: u={case['u']:.1f} m/s, T={case['T']:.1f} K")

        inlet_state = {k: case[k] for k in ['rho', 'u', 'p', 'T']}
        thermo_props = {k: case[k] for k in ['cp', 'R', 'gamma']}

        try:
            result = run_nozzle_pinn(
                model_path='nozzle_pinn.pt',
                inlet_state=inlet_state,
                ambient_p=ambient_p,
                A_in=0.375,
                A_exit=0.340,
                length=1.0,
                thermo_props=thermo_props,
                m_dot=m_dot,
                device='cpu'
            )

            thrust = result['thrust_total']
            u_exit = result['exit_state']['u']

            if thrust > 0 and u_exit > inlet_state['u']:
                print(f"  ✓ PASS: Thrust={thrust/1e3:.2f} kN, u_exit={u_exit:.1f} m/s")
            else:
                print(f"  ❌ FAIL: Thrust={thrust/1e3:.2f} kN, u_exit={u_exit:.1f} m/s")
                all_passed = False

        except Exception as e:
            print(f"  ❌ FAIL: Exception raised: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ ALL SCALING TESTS PASSED")
    else:
        print("\n❌ SOME SCALING TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NOZZLE PINN v3.0 INTEGRATION FIX - TEST SUITE")
    print("="*70)
    print("\nThis test validates the fix for negative thrust bug.")
    print("Root cause: Missing velocity scale update in runtime normalization")
    print("Fix: Added runtime_scales['u'] = max(checkpoint_scale, 1.5 * inlet_u)")
    print("="*70)

    # Run tests
    test1_passed = test_nozzle_pinn_positive_thrust()
    test2_passed = test_nozzle_scaling_robustness()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Test 1 (Positive Thrust):    {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Test 2 (Scaling Robustness): {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✅ ALL TESTS PASSED - Fix verified!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Investigation needed")
        sys.exit(1)
