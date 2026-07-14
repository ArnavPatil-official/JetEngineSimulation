"""
Unit tests for the extended-Zeldovich thermal-NO post-processor (Phase 2.4).

The rate constants are the standard evaluation tabulated in Turns,
"An Introduction to Combustion" (3rd ed.); tests pin them at reference
temperatures and check the physical behavior of the integrator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from simulation.nox_chemistry import (
    k1f, k2f, k3f, thermal_no_rate, integrate_no, combustor_residence_time,
)


class TestRateConstants:
    """Pin the cited Arrhenius expressions at reference temperatures."""

    def test_k1f_at_2000K(self):
        # k1f = 1.8e11 * exp(-38370/2000) = 1.8e11 * exp(-19.185)
        expected = 1.8e11 * np.exp(-19.185)
        assert k1f(2000.0) == pytest.approx(expected, rel=1e-12)

    def test_k2f_at_2000K(self):
        expected = 1.8e7 * 2000.0 * np.exp(-2.34)
        assert k2f(2000.0) == pytest.approx(expected, rel=1e-12)

    def test_k3f_at_2000K(self):
        expected = 7.1e10 * np.exp(-0.225)
        assert k3f(2000.0) == pytest.approx(expected, rel=1e-12)

    def test_strong_temperature_sensitivity(self):
        # Thermal NO signature: ~38370 K activation temperature means the
        # initiation rate roughly doubles every ~90 K near 2200 K.
        r_2200 = thermal_no_rate(2200.0, 1e-4, 1e-2)
        r_2400 = thermal_no_rate(2400.0, 1e-4, 1e-2)
        assert r_2400 / r_2200 > 4.0  # e^(38370*(1/2200-1/2400)) ~ 4.3


class TestIntegrator:
    def test_linear_growth_far_from_equilibrium(self):
        # With NO_eq huge, NO(t) ~ rate0 * t
        conc = {'O': 1e-5, 'N2': 1e-2}
        rate0 = thermal_no_rate(2300.0, conc['O'], conc['N2'])
        c_no = integrate_no(2300.0, conc, tau=1e-3, no_eq=1e9)
        assert c_no == pytest.approx(rate0 * 1e-3, rel=1e-2)

    def test_never_exceeds_equilibrium(self):
        conc = {'O': 1e-3, 'N2': 1e-2}
        no_eq = 1e-6
        c_no = integrate_no(2600.0, conc, tau=10.0, no_eq=no_eq)
        assert c_no <= no_eq * (1 + 1e-9)

    def test_zero_tau_zero_no(self):
        assert integrate_no(2300.0, {'O': 1e-5, 'N2': 1e-2}, tau=0.0,
                            no_eq=1e-6) == 0.0


class TestResidenceTime:
    def test_magnitude_at_takeoff_conditions(self):
        # V ~ 0.104 m^3, m_dot ~ 82 kg/s, rho ~ 5.5 kg/m^3 (40 bar, 2500 K)
        tau = combustor_residence_time(0.207 * 0.5, 82.0, 5.5)
        assert 0.002 < tau < 0.02  # aero-combustor ms range

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            combustor_residence_time(0.1, 0.0, 5.0)
