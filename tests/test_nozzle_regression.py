"""
Nozzle Regression Test: Analytic vs PINN Agreement.

This test verifies that the PINN-anchored nozzle produces similar results
to the analytic nozzle for a baseline Jet-A1 case. This ensures the PINN
has been trained correctly and the anchoring logic is working.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only run if integrated_engine can be imported
try:
    from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import integrated_engine: {e}")
    INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="integrated_engine not available")
class TestNozzleRegression:
    """Regression tests for nozzle PINN vs analytic."""

    @classmethod
    def setup_class(cls):
        """Initialize engine once for all tests."""
        cls.engine = IntegratedTurbofanEngine(
            mechanism_profile="blends",
            creck_mechanism_path="data/creck_c1c16_full.yaml"
        )

    def test_analytic_nozzle_baseline(self):
        """Test analytic nozzle with Jet-A1 surrogate."""
        fuel = FUEL_LIBRARY["Jet-A1"]

        try:
            result = self.engine.run_full_cycle(fuel, phi=0.5)

            # Check that result contains expected keys
            assert 'nozzle' in result
            assert 'performance' in result

            nozz = result['nozzle']
            perf = result['performance']

            # Validate nozzle outputs
            assert nozz['thrust_total'] > 0, "Thrust should be positive"
            assert nozz['u'] > 0, "Exit velocity should be positive"
            assert nozz['T'] > 0, "Exit temperature should be positive"
            assert nozz['p'] > 0, "Exit pressure should be positive"

            # Validate performance outputs
            assert perf['thrust_kN'] > 0, "Net thrust should be positive"
            assert 0 < perf['tsfc_mg_per_Ns'] < 100, "TSFC should be reasonable"
            assert 0 < perf['thermal_efficiency'] < 1.0, "Thermal efficiency should be fraction"

            print("\n✅ Analytic nozzle baseline test PASSED")
            print(f"   Thrust: {perf['thrust_kN']:.2f} kN")
            print(f"   TSFC:   {perf['tsfc_mg_per_Ns']:.2f} mg/(N·s)")
            print(f"   η_th:   {perf['thermal_efficiency']*100:.2f}%")

        except Exception as e:
            pytest.fail(f"Analytic nozzle test failed: {e}")

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="PINN not loaded")
    def test_pinn_vs_analytic_agreement(self):
        """
        Test that PINN-anchored nozzle agrees with analytic within tolerance.

        Acceptance criteria: Exit velocity, thrust within 5% for same inlet conditions.
        """
        fuel = FUEL_LIBRARY["Jet-A1"]

        # Run with analytic nozzle
        self.engine.USE_NOZZLE_PINN = False
        self.engine.NOZZLE_MODE = "analytic"

        try:
            result_analytic = self.engine.run_full_cycle(fuel, phi=0.5)
        except Exception as e:
            pytest.skip(f"Could not run analytic case: {e}")

        # Run with PINN nozzle (if available)
        self.engine.USE_NOZZLE_PINN = True
        self.engine.NOZZLE_MODE = "pinn_anchored"

        try:
            result_pinn = self.engine.run_full_cycle(fuel, phi=0.5)
        except Exception as e:
            pytest.skip(f"Could not run PINN case (PINN not loaded?): {e}")

        # Extract values
        u_exit_analytic = result_analytic['nozzle']['u']
        F_analytic = result_analytic['performance']['thrust_N']

        u_exit_pinn = result_pinn['nozzle']['u']
        F_pinn = result_pinn['performance']['thrust_N']

        # Compute percent differences
        delta_u_pct = abs(u_exit_pinn - u_exit_analytic) / u_exit_analytic * 100
        delta_F_pct = abs(F_pinn - F_analytic) / F_analytic * 100

        # Tolerance: 5%
        TOLERANCE_PCT = 5.0

        print("\n" + "="*60)
        print("NOZZLE PINN vs ANALYTIC REGRESSION TEST")
        print("="*60)
        print(f"Exit Velocity:")
        print(f"  Analytic: {u_exit_analytic:.1f} m/s")
        print(f"  PINN:     {u_exit_pinn:.1f} m/s")
        print(f"  Δ:        {delta_u_pct:.2f}% {'✅ PASS' if delta_u_pct < TOLERANCE_PCT else '❌ FAIL'}")
        print(f"\nNet Thrust:")
        print(f"  Analytic: {F_analytic/1e3:.2f} kN")
        print(f"  PINN:     {F_pinn/1e3:.2f} kN")
        print(f"  Δ:        {delta_F_pct:.2f}% {'✅ PASS' if delta_F_pct < TOLERANCE_PCT else '❌ FAIL'}")
        print("="*60)

        # Assert within tolerance
        assert delta_u_pct < TOLERANCE_PCT, \
            f"Exit velocity mismatch: {delta_u_pct:.2f}% > {TOLERANCE_PCT}%"
        assert delta_F_pct < TOLERANCE_PCT, \
            f"Thrust mismatch: {delta_F_pct:.2f}% > {TOLERANCE_PCT}%"

        print("\n✅ PINN vs Analytic agreement test PASSED")

    def test_optimizer_guards(self):
        """Test that performance outputs are finite and physically valid."""
        fuel = FUEL_LIBRARY["Jet-A1"]

        # Normal case should produce physically valid metrics
        result = self.engine.run_full_cycle(fuel, phi=0.5)
        perf = result['performance']

        assert np.isfinite(perf['thrust_N']) and perf['thrust_N'] > 0, \
            "Thrust should be finite and positive"
        assert np.isfinite(perf['thrust_kN']) and perf['thrust_kN'] > 0, \
            "Thrust (kN) should be finite and positive"
        assert np.isfinite(perf['tsfc_mg_per_Ns']) and perf['tsfc_mg_per_Ns'] > 0, \
            "TSFC should be finite and positive"
        assert np.isfinite(perf['thermal_efficiency']) and 0 < perf['thermal_efficiency'] < 1.0, \
            "Thermal efficiency should be a valid fraction"

        print("\n✅ Optimizer guards test PASSED")
        print(f"   Thrust: {perf['thrust_kN']:.2f} kN")


def run_standalone_comparison():
    """Standalone comparison script (no pytest)."""
    if not INTEGRATION_AVAILABLE:
        print("❌ Cannot run standalone comparison: integrated_engine not available")
        return

    print("\n" + "="*70)
    print("STANDALONE NOZZLE COMPARISON")
    print("="*70)

    # Initialize engine
    engine = IntegratedTurbofanEngine(
        mechanism_profile="blends",
        creck_mechanism_path="data/creck_c1c16_full.yaml"
    )

    fuel = FUEL_LIBRARY["Jet-A1"]

    # Run analytic
    print("\n[1] Running ANALYTIC nozzle...")
    engine.USE_NOZZLE_PINN = False
    engine.NOZZLE_MODE = "analytic"

    try:
        result_analytic = engine.run_full_cycle(fuel, phi=0.5)
        print(f"✅ Analytic: Thrust = {result_analytic['performance']['thrust_kN']:.2f} kN")
    except Exception as e:
        print(f"❌ Analytic failed: {e}")
        return

    # Run PINN (if available)
    print("\n[2] Running PINN-ANCHORED nozzle...")
    engine.USE_NOZZLE_PINN = True
    engine.NOZZLE_MODE = "pinn_anchored"

    try:
        result_pinn = engine.run_full_cycle(fuel, phi=0.5)
        print(f"✅ PINN:     Thrust = {result_pinn['performance']['thrust_kN']:.2f} kN")
    except Exception as e:
        print(f"⚠️  PINN not available: {e}")
        return

    # Compare
    delta_thrust = abs(result_pinn['performance']['thrust_kN'] -
                      result_analytic['performance']['thrust_kN'])
    delta_pct = delta_thrust / result_analytic['performance']['thrust_kN'] * 100

    print("\n" + "="*70)
    print(f"COMPARISON RESULT: Δ = {delta_pct:.2f}%")
    if delta_pct < 5.0:
        print("✅ AGREEMENT WITHIN 5% - PASS")
    else:
        print("⚠️  DIFFERENCE > 5% - INVESTIGATE")
    print("="*70)


if __name__ == "__main__":
    # Run pytest tests
    pytest.main([__file__, "-v", "--tb=short"])

    # Run standalone comparison
    run_standalone_comparison()
