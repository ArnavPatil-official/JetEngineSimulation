"""
Thermodynamic Utilities for PINN Integration
=============================================

Helper functions to bridge Cantera combustion chemistry with PINNs,
enabling fuel-dependent, non-ideal thermodynamic modeling in turbine
and nozzle components.

Key Features:
- Extracts real thermodynamic properties (cp, R, gamma) from combustor output
- Builds PINN-compatible condition dictionaries
- Ensures thermodynamic consistency across engine components
"""

from typing import Dict, Any


def build_turbine_conditions(
    combustor_out: Dict[str, Any],
    mass_flow_core: float,
    geometry: Dict[str, float] = None,
    target_work: float = None
) -> Dict[str, Any]:
    """
    Build turbine PINN conditions dictionary from combustor output.

    This function extracts fuel-dependent thermodynamic properties from
    the combustor and packages them for use in the turbine PINN physics
    losses. Unlike the old fixed-gamma approach, this makes the PINN
    genuinely necessary for modeling non-ideal expansion.

    Args:
        combustor_out: Dictionary from Combustor.run() containing:
            - T_out: Temperature [K]
            - p_out: Pressure [Pa]
            - cp_out: Specific heat at constant pressure [J/(kg·K)]
            - R_out: Specific gas constant [J/(kg·K)]
            - gamma_out: Heat capacity ratio [-]
        mass_flow_core: Core mass flow rate [kg/s]
        geometry: Optional dict with A_inlet, A_outlet, length [m, m, m]
        target_work: Optional target shaft work [W]

    Returns:
        Dictionary with structure:
        {
            'inlet': {'T': ..., 'p': ..., 'rho': ..., 'u': ...},
            'physics': {
                'cp': ...,      # From combustor (fuel-dependent!)
                'R': ...,       # From combustor (fuel-dependent!)
                'gamma': ...,   # From combustor (fuel-dependent!)
                'mass_flow': ...,
                'w_shaft': ...  # Optional
            },
            'geometry': {'A_inlet': ..., 'A_outlet': ..., 'length': ...}
        }

    Note:
        The thermodynamic properties (cp, R, gamma) are derived from the
        actual combustion products for the current fuel blend, NOT from
        fixed air-like constants. This breaks the validity of constant-gamma
        isentropic relations and justifies the PINN approach.
    """
    # Extract thermodynamic properties from combustor
    cp = combustor_out["cp_out"]
    R = combustor_out["R_out"]
    gamma = combustor_out["gamma_out"]

    T_in = combustor_out["T_out"]
    p_in = combustor_out["p_out"]

    # Calculate density at turbine inlet using ideal gas law
    # (this is still valid even for non-ideal cp, gamma)
    rho_in = p_in / (R * T_in)

    # Default geometry (can be overridden)
    if geometry is None:
        geometry = {
            'A_inlet': 0.207,    # m²
            'A_outlet': 0.377,   # m²
            'length': 0.5        # m
        }

    # Estimate inlet velocity from mass flow and density
    u_in = mass_flow_core / (rho_in * geometry['A_inlet'])

    # Build conditions dictionary
    conditions = {
        'inlet': {
            'T': T_in,
            'p': p_in,
            'rho': rho_in,
            'u': u_in,
        },
        'physics': {
            'cp': cp,           # FUEL-DEPENDENT (from Cantera)
            'R': R,             # FUEL-DEPENDENT (from Cantera)
            'gamma': gamma,     # FUEL-DEPENDENT (from Cantera)
            'mass_flow': mass_flow_core,
        },
        'geometry': geometry,
    }

    # Optionally add target shaft work
    if target_work is not None:
        conditions['physics']['w_shaft'] = target_work

    return conditions


def build_nozzle_conditions(
    turbine_exit_state: Dict[str, Any],
    thermo_props: Dict[str, float],
    mass_flow: float,
    p_ambient: float = 101325.0,
    geometry: Dict[str, float] = None,
    target_thrust: float = None
) -> Dict[str, Any]:
    """
    Build nozzle PINN conditions dictionary from turbine exit state.

    Args:
        turbine_exit_state: Dictionary with turbine exit state:
            - T: Temperature [K]
            - p: Pressure [Pa]
            - rho: Density [kg/m³]
            - u: Velocity [m/s]
        thermo_props: Thermodynamic properties (cp, R, gamma) from combustor
        mass_flow: Total mass flow rate [kg/s]
        p_ambient: Ambient pressure [Pa]
        geometry: Optional dict with A_inlet, A_exit, length
        target_thrust: Optional target thrust [N]

    Returns:
        Dictionary with structure:
        {
            'inlet': {'T': ..., 'p': ..., 'rho': ..., 'u': ...},
            'ambient': {'p': ...},
            'physics': {
                'cp': ...,      # From combustor (fuel-dependent!)
                'R': ...,       # From combustor (fuel-dependent!)
                'gamma': ...,   # From combustor (fuel-dependent!)
                'mass_flow': ...,
                'target_thrust': ...  # Optional
            },
            'geometry': {'A_inlet': ..., 'A_exit': ..., 'length': ...}
        }
    """
    # Default geometry
    if geometry is None:
        geometry = {
            'A_inlet': 0.375,
            'A_exit': 0.340,
            'length': 1.0
        }

    # Build conditions dictionary
    conditions = {
        'inlet': {
            'T': turbine_exit_state['T'],
            'p': turbine_exit_state['p'],
            'rho': turbine_exit_state['rho'],
            'u': turbine_exit_state['u'],
        },
        'ambient': {
            'p': p_ambient
        },
        'physics': {
            'cp': thermo_props['cp'],       # FUEL-DEPENDENT
            'R': thermo_props['R'],         # FUEL-DEPENDENT
            'gamma': thermo_props['gamma'], # FUEL-DEPENDENT
            'mass_flow': mass_flow,
        },
        'geometry': geometry,
    }

    # Optionally add target thrust
    if target_thrust is not None:
        conditions['physics']['target_thrust'] = target_thrust

    return conditions


def extract_thermo_props(combustor_out: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract thermodynamic properties from combustor output.

    Convenience function to get just the cp, R, gamma values.

    Args:
        combustor_out: Output from Combustor.run()

    Returns:
        Dictionary with keys: cp, R, gamma
    """
    return {
        'cp': combustor_out['cp_out'],
        'R': combustor_out['R_out'],
        'gamma': combustor_out['gamma_out'],
    }
