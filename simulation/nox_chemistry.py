"""
Extended-Zeldovich thermal-NO post-processor (Phase 2.4, V5).

The CRECK C1-C16 mechanism used for blend combustion contains no nitrogen
chemistry (zero N-species), so NO cannot come out of the combustor solution
directly. This module post-processes the equilibrium combustion products with
the extended Zeldovich (thermal-NO) mechanism:

    (1)  O + N2  ->  NO + N       k1f = 1.8e11 * exp(-38370/T)
    (2)  N + O2  ->  NO + O       k2f = 1.8e7  * T * exp(-4680/T)
    (3)  N + OH  ->  NO + H       k3f = 7.1e10 * exp(-450/T)

Rate constants in m^3/(kmol s), T in K — the standard evaluation tabulated in
Turns, "An Introduction to Combustion" (3rd ed., Table on thermal NO, after
Dean & Bozzelli / Baulch et al. evaluations; consistent with GRI-Mech 3.0
within tens of percent over 1800-2800 K).

Assumptions (each is a documented limitation, not a hidden choice):
- O, O2, OH, N2 at their EQUILIBRIUM concentrations in the combustor products
  (classical Zeldovich decoupling: thermal NO is slow vs. fuel oxidation).
- N-atom in quasi-steady state.
- NO growth integrated over a residence time tau with the finite-rate
  approach-to-equilibrium factor (1 - alpha^2), alpha = [NO]/[NO]_eq, so NO
  can never exceed its equilibrium value.
- Single homogeneous zone at the equilibrium flame temperature: no primary-
  zone/dilution-zone split. This makes the estimate an UPPER-BOUND-flavored
  proxy (a real combustor quenches NO formation by dilution).

EI convention: emission index reported as g NO2-equivalent per kg fuel
(ICAO databank convention), i.e. moles of NO x M_NO2.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import cantera as ct

M_NO2 = 46.0055  # g/mol, ICAO EI-NOx convention (NOx as NO2)

# Extended Zeldovich forward-rate constants, m^3/(kmol s)
def k1f(T: float) -> float:
    return 1.8e11 * np.exp(-38370.0 / T)

def k2f(T: float) -> float:
    return 1.8e7 * T * np.exp(-4680.0 / T)

def k3f(T: float) -> float:
    return 7.1e10 * np.exp(-450.0 / T)


def thermal_no_rate(T: float, c_o: float, c_n2: float) -> float:
    """
    Initial (NO-free) thermal-NO formation rate, d[NO]/dt = 2 k1f [O][N2].

    Args:
        T: Temperature [K]
        c_o: Equilibrium O-atom concentration [kmol/m^3]
        c_n2: N2 concentration [kmol/m^3]

    Returns:
        NO formation rate [kmol/(m^3 s)]
    """
    return 2.0 * k1f(T) * c_o * c_n2


def integrate_no(
    T: float,
    concentrations: Dict[str, float],
    tau: float,
    no_eq: Optional[float] = None,
    n_steps: int = 200,
) -> float:
    """
    Integrate quasi-steady extended-Zeldovich NO over a residence time.

        d[NO]/dt = 2 k1f [O][N2] * (1 - alpha^2),  alpha = [NO]/[NO]_eq

    (Turns eq. 5.9-5.11 form; the (1 - alpha^2) factor enforces approach to
    equilibrium and keeps the result bounded.)

    Args:
        T: Flame temperature [K]
        concentrations: dict with 'O', 'N2' (and optionally 'NO' initial),
                        [kmol/m^3]
        tau: Residence time [s]
        no_eq: Equilibrium NO concentration [kmol/m^3]; if None the rate is
               integrated without the approach-to-equilibrium factor.
        n_steps: Explicit-Euler substeps (rate is smooth; 200 is plenty)

    Returns:
        NO concentration after tau [kmol/m^3]
    """
    c_no = concentrations.get('NO', 0.0)
    rate0 = thermal_no_rate(T, concentrations['O'], concentrations['N2'])
    dt = tau / n_steps
    for _ in range(n_steps):
        if no_eq is not None and no_eq > 0:
            alpha = min(c_no / no_eq, 1.0)
            rate = rate0 * (1.0 - alpha * alpha)
        else:
            rate = rate0
        c_no += rate * dt
        if no_eq is not None and c_no >= no_eq:
            return no_eq
    return c_no


def zeldovich_ei_nox(
    gas: ct.Solution,
    tau: float,
    fuel_mass_per_kg_mix: float,
) -> Dict[str, float]:
    """
    EI-NOx from equilibrium combustor products via extended Zeldovich.

    Args:
        gas: Cantera Solution ALREADY equilibrated at combustor exit
             (HP-equilibrium products; must contain O, O2, OH, N2).
        tau: Combustor residence time [s]
        fuel_mass_per_kg_mix: Fuel mass fraction of the burned mixture
             (m_dot_fuel / m_dot_total), used to convert to per-kg-fuel EI.

    Returns:
        Dict with EI_NOx_g_per_kg_fuel, X_NO (mole fraction after tau),
        T_flame, tau, and the initial rate.
    """
    T = gas.T
    conc = gas.concentrations  # kmol/m^3
    def c(sp: str) -> float:
        return float(conc[gas.species_index(sp)]) if sp in gas.species_names else 0.0

    c_o, c_n2 = c('O'), c('N2')

    # Equilibrium NO estimate from O2/N2 equilibrium: NO_eq via the reaction
    # N2 + O2 <-> 2 NO with Kp(T) (JANAF fit adequate for a bound):
    #   Kp = 21.9 * exp(-21590/T)   (Turns eq. for NO equilibrium)
    c_o2 = c('O2')
    kp = 21.9 * np.exp(-21590.0 / T)
    no_eq = float(np.sqrt(max(kp * c_n2 * c_o2, 0.0)))

    c_no = integrate_no(T, {'O': c_o, 'N2': c_n2}, tau, no_eq=no_eq)

    # Convert concentration to mass basis: g NO2-equiv per kg of mixture
    rho = gas.density  # kg/m^3
    g_no2_per_kg_mix = c_no * M_NO2 / rho * 1000.0  # kmol/m3 * g/mol -> g/m3 /(kg/m3)

    ei = g_no2_per_kg_mix / max(fuel_mass_per_kg_mix, 1e-12)

    return {
        'EI_NOx_g_per_kg_fuel': float(ei),
        'X_NO': float(c_no * ct.gas_constant * T / gas.P
                      if gas.P > 0 else 0.0),  # ideal gas: X = c*R_u*T/p
        'c_NO_kmol_m3': float(c_no),
        'c_NO_eq_kmol_m3': no_eq,
        'T_flame_K': float(T),
        'tau_s': float(tau),
        'initial_rate_kmol_m3_s': thermal_no_rate(T, c_o, c_n2),
    }


def combustor_residence_time(
    volume_m3: float,
    m_dot: float,
    rho: float,
) -> float:
    """
    Residence time tau = V / (m_dot / rho).

    The combustor volume is not part of the 0-D design point; the default used
    by callers is V = A_combustor_exit (0.207 m^2) x L = 0.5 m ~= 0.104 m^3,
    documented in outputs/parameter_provenance.md. At take-off-like conditions
    this gives tau ~= 5-8 ms, the standard aero-combustor magnitude.
    """
    if m_dot <= 0 or rho <= 0:
        raise ValueError("m_dot and rho must be positive")
    return volume_m3 * rho / m_dot
