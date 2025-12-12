"""
Engine Simulation Data Structures.

This module defines dataclasses for type-safe, optimizer-ready engine simulation.
All performance metrics, flow states, and fuel properties are structured to ensure
consistent access patterns and clear documentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class FlowState:
    """
    Represents the thermodynamic and flow field state at a point in the engine.

    Attributes:
        rho: Density [kg/m³]
        u: Velocity [m/s]
        p: Pressure [Pa]
        T: Temperature [K]
        cp: Specific heat at constant pressure [J/(kg·K)] - FUEL-DEPENDENT
        R: Specific gas constant [J/(kg·K)] - FUEL-DEPENDENT
        gamma: Heat capacity ratio cp/cv [-] - FUEL-DEPENDENT
    """
    rho: float
    u: float
    p: float
    T: float
    cp: float
    R: float
    gamma: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backward compatibility."""
        return {
            'rho': self.rho,
            'u': self.u,
            'p': self.p,
            'T': self.T,
            'cp': self.cp,
            'R': self.R,
            'gamma': self.gamma
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'FlowState':
        """Create FlowState from dictionary."""
        return cls(
            rho=d['rho'],
            u=d['u'],
            p=d['p'],
            T=d['T'],
            cp=d['cp'],
            R=d['R'],
            gamma=d['gamma']
        )


@dataclass
class FuelProperties:
    """
    Fuel-specific thermochemical properties.

    Attributes:
        name: Fuel identifier (e.g., "Jet-A1", "HEFA-50")
        LHV_J_per_kg: Lower Heating Value [J/kg]
        w_C: Carbon mass fraction (mass carbon / mass fuel) [-]
        composition: Cantera composition string (e.g., "NC12H26:1.0")
    """
    name: str
    LHV_J_per_kg: float
    w_C: float
    composition: str


@dataclass
class PerformanceMetrics:
    """
    Complete engine performance metrics for optimization and reporting.

    All metrics are optimizer-safe: NaN/Inf indicate invalid solutions.

    Thrust Metrics:
        net_thrust_N: Net thrust force [N]
        net_thrust_kN: Net thrust force [kN]

    Fuel Consumption:
        tsfc_kg_per_Ns: Thrust Specific Fuel Consumption [kg/(N·s)]
        tsfc_mg_per_Ns: TSFC in [mg/(N·s)] (industry standard unit)

    Emissions (all per unit thrust basis):
        co2_g_per_kN_s: CO₂ emissions [g/(kN·s)]
        EI_NOx: NOx Emissions Index [g/kg_fuel]
        EI_CO: CO Emissions Index [g/kg_fuel]

    Efficiency:
        eta_thermal: Thermal efficiency (jet power / fuel power) [-]
        eta_exergy: Exergy efficiency (jet power / fuel exergy) [-]

    Mass Flows:
        m_dot_fuel: Fuel mass flow rate [kg/s]
        m_dot_total: Total mass flow (air + fuel) [kg/s]
        fuel_air_ratio: f = m_fuel / m_air [-]

    Validity:
        is_valid: True if all physics constraints satisfied
        error_msg: Description of constraint violation (if any)
    """
    # Thrust
    net_thrust_N: float
    net_thrust_kN: float

    # Fuel consumption
    tsfc_kg_per_Ns: float
    tsfc_mg_per_Ns: float

    # Emissions
    co2_g_per_kN_s: float
    EI_NOx: float
    EI_CO: float

    # Efficiency
    eta_thermal: float
    eta_exergy: float

    # Mass flows
    m_dot_fuel: float
    m_dot_total: float
    fuel_air_ratio: float

    # Validity flags
    is_valid: bool = True
    error_msg: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility and serialization."""
        return {
            'net_thrust_N': self.net_thrust_N,
            'net_thrust_kN': self.net_thrust_kN,
            'tsfc_kg_per_Ns': self.tsfc_kg_per_Ns,
            'tsfc_mg_per_Ns': self.tsfc_mg_per_Ns,
            'co2_g_per_kN_s': self.co2_g_per_kN_s,
            'EI_NOx': self.EI_NOx,
            'EI_CO': self.EI_CO,
            'eta_thermal': self.eta_thermal,
            'eta_exergy': self.eta_exergy,
            'm_dot_fuel': self.m_dot_fuel,
            'm_dot_total': self.m_dot_total,
            'fuel_air_ratio': self.fuel_air_ratio,
            'is_valid': self.is_valid,
            'error_msg': self.error_msg
        }


@dataclass
class EngineCycleResult:
    """
    Complete engine cycle simulation result.

    Attributes:
        compressor: Compressor stage output dictionary
        combustor: Combustor stage output dictionary
        turbine: Turbine stage output dictionary
        nozzle: Nozzle stage output dictionary
        performance: Structured performance metrics
        fuel_properties: Fuel properties used in this run
    """
    compressor: Dict[str, Any]
    combustor: Dict[str, Any]
    turbine: Dict[str, Any]
    nozzle: Dict[str, Any]
    performance: PerformanceMetrics
    fuel_properties: FuelProperties

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary for backward compatibility."""
        return {
            'compressor': self.compressor,
            'combustor': self.combustor,
            'turbine': self.turbine,
            'nozzle': self.nozzle,
            'performance': self.performance.to_dict(),
            'fuel_properties': {
                'name': self.fuel_properties.name,
                'LHV_J_per_kg': self.fuel_properties.LHV_J_per_kg,
                'w_C': self.fuel_properties.w_C,
                'composition': self.fuel_properties.composition
            }
        }
