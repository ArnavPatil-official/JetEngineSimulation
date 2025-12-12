"""
Nozzle PINN Conditions Builder.

This module provides utilities to construct the CONDITIONS dictionary for nozzle
PINN training and inference with fuel-dependent thermodynamic properties.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def load_engine_conditions_from_icao(
    filename: str = 'icao_engine_data.csv',
    engine_id: str = 'Trent 1000-AE3',
    mode: str = 'TAKE-OFF',
    thermo_props: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Load nozzle inlet conditions from ICAO engine database.

    This function constructs the CONDITIONS dictionary used for PINN training
    and inference. Thermodynamic properties (cp, R, gamma) can be overridden
    to support fuel-dependent modeling.

    Args:
        filename: ICAO data CSV filename
        engine_id: Engine identifier from database
        mode: Operating mode ('TAKE-OFF', 'CLIMB', 'CRUISE', 'IDLE')
        thermo_props: Optional fuel-dependent properties dict with keys:
                     - cp: Specific heat [J/(kg·K)]
                     - R: Gas constant [J/(kg·K)]
                     - gamma: Heat capacity ratio [-]

    Returns:
        CONDITIONS dictionary with structure:
        {
            'inlet': {'rho': ..., 'u': ..., 'p': ..., 'T': ...},
            'ambient': {'p': ...},
            'geometry': {'A_inlet': ..., 'A_exit': ..., 'length': ...},
            'physics': {'R': ..., 'gamma': ..., 'cp': ..., 'mass_flow': ..., 'target_thrust': ...}
        }
    """
    try:
        # Build robust path to data file
        script_dir = Path(__file__).resolve().parent
        data_path = script_dir.parent.parent / 'data' / filename

        df = pd.read_csv(data_path)
        row = df[(df['Engine ID'].str.contains(engine_id, regex=False)) &
                 (df['Mode'] == mode)].iloc[0]

        fuel_flow = float(row['Fuel Flow (kg/s)'])
        pr_overall = float(row['Pressure Ratio'])
        thrust_total = float(row['Rated Thrust (kN)']) * 1000.0

        # Simplified cycle calculations
        FAR = 0.030  # Fuel-Air Ratio
        core_air = fuel_flow / FAR
        mass_flow = core_air + fuel_flow  # Total core mass flow

        p_amb = 101325.0
        # Nozzle inlet pressure: ~4.5% of overall PR * ambient
        p_inlet_nozzle = p_amb * pr_overall * 0.045
        T_inlet_nozzle = 1005.0  # K (from turbine analysis)

    except Exception as e:
        print(f"⚠️  Warning: Could not load ICAO data. Using defaults. Error: {e}")
        mass_flow = 79.9
        p_inlet_nozzle = 193000.0
        thrust_total = 310.9e3

    # Use fuel-dependent properties if provided, else defaults
    if thermo_props is not None:
        cp = thermo_props['cp']
        R = thermo_props['R']
        gamma = thermo_props['gamma']
    else:
        # Default air-like values
        cp = 1150.0
        R = 287.0
        gamma = 1.33

    # Fixed inlet flow state (from previous turbine output)
    # These are kept constant for training stability
    rho_inlet = 0.67  # kg/m³
    u_inlet = 317.7   # m/s

    return {
        'inlet': {
            'rho': rho_inlet,
            'u': u_inlet,
            'p': p_inlet_nozzle,
            'T': 1005.0
        },
        'ambient': {
            'p': 101325.0
        },
        'geometry': {
            'A_inlet': 0.375,  # m²
            'A_exit': 0.340,   # m²
            'length': 1.0      # m
        },
        'physics': {
            'R': R,             # FUEL-DEPENDENT
            'gamma': gamma,     # FUEL-DEPENDENT
            'cp': cp,           # FUEL-DEPENDENT
            'mass_flow': mass_flow,
            'target_thrust': thrust_total * 0.15  # 15% of total engine thrust
        }
    }


def build_nozzle_conditions_from_turbine_exit(
    turbine_exit_state: Dict[str, float],
    thermo_props: Dict[str, float],
    mass_flow: float,
    p_ambient: float = 101325.0,
    target_thrust: Optional[float] = None
) -> Dict[str, Any]:
    """
    Build nozzle CONDITIONS from actual turbine exit state (for engine cycle integration).

    This function is used during engine cycle simulation to construct the nozzle
    inlet conditions from the upstream turbine exit state.

    Args:
        turbine_exit_state: Dict with keys: rho, u, p, T, cp, R, gamma
        thermo_props: Fuel-dependent properties (cp, R, gamma)
        mass_flow: Total mass flow [kg/s]
        p_ambient: Ambient pressure [Pa]
        target_thrust: Optional target thrust [N] (for training only)

    Returns:
        CONDITIONS dictionary compatible with nozzle PINN
    """
    return {
        'inlet': {
            'rho': turbine_exit_state['rho'],
            'u': turbine_exit_state['u'],
            'p': turbine_exit_state['p'],
            'T': turbine_exit_state['T']
        },
        'ambient': {
            'p': p_ambient
        },
        'geometry': {
            'A_inlet': 0.375,
            'A_exit': 0.340,
            'length': 1.0
        },
        'physics': {
            'R': thermo_props['R'],
            'gamma': thermo_props['gamma'],
            'cp': thermo_props['cp'],
            'mass_flow': mass_flow,
            'target_thrust': target_thrust if target_thrust is not None else 0.0
        }
    }
