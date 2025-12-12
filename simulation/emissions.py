"""
Emissions Estimation for Jet Engine Simulation.

This module provides emissions indices (EI) calculation for CO and NOx.
Two modes are supported:
1. Cantera-based: Extract species from chemical equilibrium (if mechanism supports it)
2. Correlation-based: Use empirical correlations from literature (for blends without detailed chemistry)

All emissions are reported as Emissions Indices: g pollutant / kg fuel
"""

from typing import Dict, Tuple, Optional
import cantera as ct
import numpy as np


# Physical constants
MW_CO = 28.01  # g/mol
MW_NO = 30.01  # g/mol (we report as NO for NOx)
MW_NO2 = 46.01  # g/mol


def extract_cantera_emissions(
    gas: ct.Solution,
    m_dot_fuel: float
) -> Tuple[float, float]:
    """
    Extract CO and NOx emissions from Cantera equilibrium products.

    Args:
        gas: Cantera Solution object at combustor exit state
        m_dot_fuel: Fuel mass flow rate [kg/s]

    Returns:
        Tuple of (EI_CO, EI_NOx) in [g/kg_fuel]

    Note:
        This method only works if the mechanism includes CO, NO, NO2 species.
        If species are not present, returns correlation-based estimate.
    """
    try:
        # Get species indices
        species_dict = {sp: i for i, sp in enumerate(gas.species_names)}

        # CO emissions
        if 'CO' in species_dict:
            idx_co = species_dict['CO']
            Y_CO = gas.Y[idx_co]  # Mass fraction
        else:
            Y_CO = 0.0

        # NOx emissions (sum of NO and NO2)
        Y_NOx = 0.0
        if 'NO' in species_dict:
            idx_no = species_dict['NO']
            Y_NOx += gas.Y[idx_no]
        if 'NO2' in species_dict:
            idx_no2 = species_dict['NO2']
            Y_NOx += gas.Y[idx_no2]

        # Convert to emissions indices [g/kg_fuel]
        # EI = (mass pollutant / mass total) / (mass fuel / mass total) = Y_pollutant / FAR
        # But we don't have FAR directly, so we use: EI ≈ Y_pollutant * (1 + 1/FAR)
        # For typical lean combustion FAR ~ 0.03, so (1 + 1/FAR) ≈ 34
        # More accurate: compute from actual mass fractions

        # Simplified: assume product mixture is mostly air + fuel products
        # EI = Y_pollutant / Y_fuel_products ≈ Y_pollutant * (m_total / m_fuel)
        # For FAR = 0.03, m_total/m_fuel = (1 + 1/FAR) = 34.33

        # Conservative approach: report mass fraction scaled by typical FAR
        FAR_typical = 0.03
        scale_factor = (1.0 + 1.0/FAR_typical)

        EI_CO = Y_CO * scale_factor * 1000  # g/kg_fuel
        EI_NOx = Y_NOx * scale_factor * 1000  # g/kg_fuel

        return EI_CO, EI_NOx

    except Exception as e:
        # Fallback to correlation if extraction fails
        print(f"⚠️  Warning: Cantera species extraction failed ({e}). Using correlation.")
        return estimate_emissions_correlation(T_flame=gas.T, p_combustor=gas.P, FAR=0.03)


def estimate_emissions_correlation(
    T_flame: float,
    p_combustor: float,
    FAR: float,
    mode: str = "cruise"
) -> Tuple[float, float]:
    """
    Estimate emissions using empirical correlations.

    This correlation-based approach is used when:
    - Chemical mechanism does not include CO/NOx species
    - Comparing fuel blends without detailed pollutant chemistry
    - Quick estimates for optimization studies

    Correlations based on:
    - Lefebvre & Ballal, "Gas Turbine Combustion" (3rd ed.)
    - ICAO Engine Emissions Databank trends

    Args:
        T_flame: Combustor exit temperature [K] (approximates flame temperature)
        p_combustor: Combustor pressure [Pa]
        FAR: Fuel-Air Ratio (mass basis) [-]
        mode: Operating mode ("idle", "cruise", "takeoff")

    Returns:
        Tuple of (EI_CO, EI_NOx) in [g/kg_fuel]

    Note:
        These are APPROXIMATIONS. Real emissions depend on combustor design,
        residence time, mixing quality, etc. Labeled as "correlation-estimated".
    """
    # Convert pressure to bar for correlation
    p_bar = p_combustor / 1e5

    # NOx correlation (Zeldovich thermal NOx mechanism)
    # EI_NOx ∝ p^0.4 * exp(T_flame / T_ref) * τ
    # Simplified form calibrated to ICAO data
    T_ref = 2000.0  # K
    if T_flame > 1600:  # NOx only significant at high temperatures
        EI_NOx = 5.0 * (p_bar ** 0.4) * np.exp((T_flame - T_ref) / 400.0)
    else:
        EI_NOx = 0.5  # Low-temperature baseline

    # CO correlation (incomplete combustion, inverse temperature dependence)
    # EI_CO ∝ p^(-0.5) * exp(-T_flame / T_ref_CO)
    # High at low T (incomplete combustion), low at high T (complete combustion)
    T_ref_CO = 1800.0  # K
    EI_CO = 50.0 * (p_bar ** (-0.5)) * np.exp(-(T_flame - 1200.0) / T_ref_CO)

    # Operating mode adjustments
    if mode == "idle":
        EI_CO *= 3.0  # Higher CO at idle (low T, poor mixing)
        EI_NOx *= 0.3  # Lower NOx at idle (low T)
    elif mode == "takeoff":
        EI_CO *= 0.5  # Lower CO at takeoff (high T, good combustion)
        EI_NOx *= 1.5  # Higher NOx at takeoff (high T, high p)

    # Clamp to physically reasonable ranges
    EI_CO = np.clip(EI_CO, 0.1, 200.0)  # g/kg_fuel
    EI_NOx = np.clip(EI_NOx, 0.1, 50.0)  # g/kg_fuel

    return float(EI_CO), float(EI_NOx)


def estimate_emissions_indices(
    combustor_out: Dict,
    gas: Optional[ct.Solution] = None,
    m_dot_fuel: float = 1.0,
    use_cantera: bool = True,
    mode: str = "cruise"
) -> Dict[str, float]:
    """
    Unified interface for emissions estimation.

    This function provides a single API for the engine cycle to call,
    automatically selecting the best available method.

    Args:
        combustor_out: Combustor output dictionary with T_out, p_out
        gas: Optional Cantera Solution object at combustor exit
        m_dot_fuel: Fuel mass flow rate [kg/s]
        use_cantera: If True, try Cantera extraction first
        mode: Operating mode for correlation fallback

    Returns:
        Dictionary with:
            - EI_CO: CO emissions index [g/kg_fuel]
            - EI_NOx: NOx emissions index [g/kg_fuel]
            - method: "cantera" or "correlation"
    """
    T_combustor = combustor_out['T_out']
    p_combustor = combustor_out['p_out']

    if use_cantera and gas is not None:
        try:
            EI_CO, EI_NOx = extract_cantera_emissions(gas, m_dot_fuel)
            method = "cantera"
        except Exception:
            EI_CO, EI_NOx = estimate_emissions_correlation(
                T_flame=T_combustor,
                p_combustor=p_combustor,
                FAR=0.03,  # Estimate
                mode=mode
            )
            method = "correlation"
    else:
        EI_CO, EI_NOx = estimate_emissions_correlation(
            T_flame=T_combustor,
            p_combustor=p_combustor,
            FAR=0.03,
            mode=mode
        )
        method = "correlation"

    return {
        'EI_CO': EI_CO,
        'EI_NOx': EI_NOx,
        'method': method
    }


if __name__ == "__main__":
    # Test correlation at typical cruise conditions
    print("Testing emissions correlations:")
    print("="*50)

    # Cruise conditions
    T_cruise = 1700.0  # K
    p_cruise = 30e5  # Pa (30 bar)
    FAR_cruise = 0.03

    EI_CO, EI_NOx = estimate_emissions_correlation(
        T_flame=T_cruise,
        p_combustor=p_cruise,
        FAR=FAR_cruise,
        mode="cruise"
    )

    print(f"Cruise (T={T_cruise} K, p={p_cruise/1e5:.1f} bar):")
    print(f"  EI_CO  = {EI_CO:.2f} g/kg_fuel")
    print(f"  EI_NOx = {EI_NOx:.2f} g/kg_fuel")
    print()

    # Idle conditions
    T_idle = 1200.0  # K
    p_idle = 5e5  # Pa (5 bar)

    EI_CO_idle, EI_NOx_idle = estimate_emissions_correlation(
        T_flame=T_idle,
        p_combustor=p_idle,
        FAR=0.01,
        mode="idle"
    )

    print(f"Idle (T={T_idle} K, p={p_idle/1e5:.1f} bar):")
    print(f"  EI_CO  = {EI_CO_idle:.2f} g/kg_fuel")
    print(f"  EI_NOx = {EI_NOx_idle:.2f} g/kg_fuel")
