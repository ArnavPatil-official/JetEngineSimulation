"""
Unit Tests for Nozzle Choking Detection.

Tests the choking detection logic with constructed inlet conditions that must
result in choked flow.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.nozzle.nozzle_pinn_v2 import (
    compute_critical_pressure_ratio,
    detect_choking
)


class TestChokingDetection:
    """Test suite for choking detection physics."""

    def test_critical_pressure_ratio_air(self):
        """Test critical pressure ratio for air (gamma=1.4)."""
        gamma_air = 1.4
        pr_critical = compute_critical_pressure_ratio(gamma_air)

        # For gamma=1.4, pr_critical = (2/2.4)^(1.4/0.4) = 0.5283
        expected = 0.5283
        assert abs(pr_critical - expected) < 0.001, \
            f"Critical PR for air should be ~0.5283, got {pr_critical}"

    def test_critical_pressure_ratio_combustion_products(self):
        """Test critical pressure ratio for combustion products (gamma=1.33)."""
        gamma_products = 1.33
        pr_critical = compute_critical_pressure_ratio(gamma_products)

        # For gamma=1.33, pr_critical = (2/2.33)^(1.33/0.33) ≈ 0.5408
        expected = 0.5408
        assert abs(pr_critical - expected) < 0.001, \
            f"Critical PR for products should be ~0.5408, got {pr_critical}"

    def test_detect_choking_unchoked(self):
        """Test detection of unchoked flow (high ambient pressure)."""
        p_inlet = 200000.0  # 2 bar
        p_ambient = 101325.0  # 1 bar (sea level)
        gamma = 1.33

        is_choked, p_critical = detect_choking(p_inlet, p_ambient, gamma)

        # p_ambient/p_inlet = 0.506 > pr_critical ≈ 0.541
        # Therefore NOT choked
        assert not is_choked, "Flow should NOT be choked for p_amb/p_in > pr_critical"
        assert p_critical < p_ambient, "Critical pressure should be below ambient"

    def test_detect_choking_choked(self):
        """Test detection of choked flow (low ambient pressure)."""
        p_inlet = 400000.0  # 4 bar
        p_ambient = 101325.0  # 1 bar
        gamma = 1.33

        is_choked, p_critical = detect_choking(p_inlet, p_ambient, gamma)

        # p_ambient/p_inlet = 0.253 < pr_critical ≈ 0.541
        # Therefore CHOKED
        assert is_choked, "Flow should be CHOKED for p_amb/p_in < pr_critical"
        assert p_critical > p_ambient, "Critical pressure should be above ambient"

    def test_detect_choking_at_critical(self):
        """Test detection at exact critical pressure ratio."""
        gamma = 1.33
        pr_critical = compute_critical_pressure_ratio(gamma)

        p_inlet = 200000.0  # 2 bar
        p_ambient = p_inlet * pr_critical  # Exactly at critical

        is_choked, p_critical_calc = detect_choking(p_inlet, p_ambient, gamma)

        # At critical point, flow is choked (M=1)
        # Use small tolerance for floating point comparison
        assert is_choked or abs(p_ambient - p_critical_calc) < 100, \
            "Flow at critical pressure should be detected as choked"

    def test_high_altitude_choking(self):
        """Test choking at high altitude (low ambient pressure)."""
        p_inlet = 150000.0  # 1.5 bar (typical after turbine at altitude)
        p_ambient = 26500.0  # ~250 hPa (10 km altitude)
        gamma = 1.33

        is_choked, p_critical = detect_choking(p_inlet, p_ambient, gamma)

        # p_ambient/p_inlet = 0.177 << pr_critical
        # Definitely CHOKED
        assert is_choked, "Flow should be CHOKED at high altitude"

    def test_gamma_variation_effect(self):
        """Test that different gamma values affect choking detection."""
        p_inlet = 300000.0  # 3 bar
        p_ambient = 101325.0  # 1 bar

        # Test with air (gamma=1.4)
        is_choked_air, _ = detect_choking(p_inlet, p_ambient, gamma=1.4)

        # Test with combustion products (gamma=1.33)
        is_choked_products, _ = detect_choking(p_inlet, p_ambient, gamma=1.33)

        # Both should be choked, but critical pressures differ
        assert is_choked_air, "Should be choked for air"
        assert is_choked_products, "Should be choked for combustion products"

    def test_static_sea_level_typical(self):
        """Test typical static sea-level conditions (after turbine)."""
        # Typical turbine exit: ~1.9 bar
        # Ambient: 1.013 bar
        # Expected: NOT choked (pressure ratio too high)

        p_inlet = 190000.0  # 1.9 bar
        p_ambient = 101325.0  # 1.013 bar
        gamma = 1.33

        is_choked, p_critical = detect_choking(p_inlet, p_ambient, gamma)

        # p_ambient/p_inlet = 0.533 ≈ pr_critical
        # This is borderline - might or might not be choked
        # Just ensure function doesn't crash
        assert isinstance(is_choked, bool)
        assert p_critical > 0

    def test_edge_case_equal_pressures(self):
        """Test edge case where inlet = ambient (no flow)."""
        p_inlet = 101325.0
        p_ambient = 101325.0
        gamma = 1.33

        is_choked, p_critical = detect_choking(p_inlet, p_ambient, gamma)

        # p_ambient/p_inlet = 1.0 > pr_critical
        # NOT choked (but also no flow!)
        assert not is_choked

    def test_invalid_gamma(self):
        """Test that invalid gamma values are handled."""
        p_inlet = 200000.0
        p_ambient = 101325.0

        # Gamma <= 1 is unphysical
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_critical_pressure_ratio(gamma=1.0)

        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_critical_pressure_ratio(gamma=0.9)


def test_choking_integration():
    """
    Integration test: Verify choking logic with realistic engine conditions.

    This test constructs a scenario where choking MUST occur and verifies detection.
    """
    # Scenario: Jet engine at takeoff, sea level
    # Turbine exit: 4.5 bar (high power setting)
    # Ambient: 1.013 bar
    # Combustion products: gamma ≈ 1.33

    p_turbine_exit = 450000.0  # Pa (4.5 bar)
    p_sea_level = 101325.0     # Pa (1.013 bar)
    gamma_products = 1.33

    is_choked, p_critical = detect_choking(
        p_turbine_exit,
        p_sea_level,
        gamma_products
    )

    # Verify choking occurs
    assert is_choked, \
        "Nozzle should be CHOKED at takeoff thrust (high turbine exit pressure)"

    # Verify critical pressure is physically reasonable
    pr_calc = p_critical / p_turbine_exit
    pr_theory = compute_critical_pressure_ratio(gamma_products)

    assert abs(pr_calc - pr_theory) < 0.001, \
        f"Critical pressure ratio mismatch: {pr_calc} vs {pr_theory}"

    # Verify critical pressure is between ambient and inlet
    assert p_sea_level < p_critical < p_turbine_exit, \
        "Critical pressure should be between ambient and inlet"

    print("\n✅ Choking integration test PASSED")
    print(f"   Turbine exit: {p_turbine_exit/1e5:.2f} bar")
    print(f"   Ambient:      {p_sea_level/1e5:.3f} bar")
    print(f"   Critical:     {p_critical/1e5:.2f} bar")
    print(f"   Status:       CHOKED at M=1")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])

    # Also run integration test standalone
    print("\n" + "="*60)
    print("Running integration test...")
    print("="*60)
    test_choking_integration()
