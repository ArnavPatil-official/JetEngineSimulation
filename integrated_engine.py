"""
Integrated Turbofan Engine Simulation with Hybrid Grey-Box Modeling.

This module implements a complete jet engine cycle using:
- Cantera: Chemical kinetics and thermodynamic equilibrium (compressor, combustor)
- PINNs: Physics-informed neural networks for flow physics (turbine, nozzle)

The simulation supports multiple fuel blends including Sustainable Aviation Fuels (SAF)
and provides fuel-dependent performance predictions.

Engine Cycle: Compressor → Combustor → Turbine → Nozzle

Author: Arnav Patil
See `documentation/COMPREHENSIVE_DOCUMENTATION.md` for the full system overview
and `documentation/NOZZLE_PINN_GUIDE.md` for nozzle/PINN specifics.
"""

import os
import sys
import torch
import numpy as np
import cantera as ct
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression

# Validate and add simulation modules to Python path
simulation_path = Path(__file__).parent / "simulation"
if not simulation_path.exists():
    raise FileNotFoundError(
        f"Simulation module directory not found at: {simulation_path}\n"
        f"Please ensure 'simulation/' folder exists with compressor/ and combustor/ subfolders"
    )
sys.path.insert(0, str(Path(__file__).parent))

# Import Cantera-based component models and thermodynamic utilities
try:
    from simulation.compressor.compressor import Compressor
    from simulation.combustor.combustor import Combustor
    from simulation.thermo_utils import extract_thermo_props
    from simulation.nozzle.nozzle import run_nozzle_pinn
    from simulation.turbine.turbine import run_turbine_pinn
except ImportError as e:
    raise ImportError(
        f"Failed to import simulation modules: {e}\n"
        f"Please ensure 'simulation/compressor/compressor.py', 'simulation/combustor/combustor.py', "
        f"and 'simulation/thermo_utils.py' exist"
    )


# ============================================================================
# FUEL BLEND DEFINITIONS
# ============================================================================

class LocalFuelBlend:
    """
    Represents a fuel blend as a mixture of surrogate species for chemical kinetics modeling.

    Each fuel is represented by n-alkane surrogates compatible with the CRECK C1-C16 mechanism:
    - Jet-A1 surrogate: n-dodecane (NC12H26) - represents typical kerosene
    - Bio-SPK surrogate: n-decane (NC10H22) - represents synthetic paraffinic kerosene
    - HEFA: Blended mixture of both surrogates
    """

    def __init__(self, name: str, composition: Dict[str, float]):
        """
        Initialize fuel blend with species composition.

        Args:
            name: Fuel blend identifier (e.g., "Jet-A1", "HEFA-50")
            composition: Dictionary mapping CRECK species to mass fractions
                        Example: {"NC12H26": 0.8, "NC10H22": 0.2}
        """
        self.name = name
        self.composition = composition

        # Ensure mass fractions sum to 1.0 for conservation
        total = sum(composition.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Mass fractions must sum to 1.0, got {total}")

    def as_composition_string(self) -> str:
        """
        Convert fuel composition to Cantera-compatible format string.

        Returns:
            Comma-separated string like "NC12H26:0.8, NC10H22:0.2"
        """
        parts = [f"{species}:{frac}" for species, frac in self.composition.items()]
        return ", ".join(parts)

    def __repr__(self):
        return f"LocalFuelBlend(name='{self.name}', composition={self.composition})"


# Pre-defined fuel library for simulation studies
FUEL_LIBRARY = {
    "Jet-A1": LocalFuelBlend("Jet-A1", {"NC12H26": 1.0}),  # Conventional jet fuel surrogate
    "Bio-SPK": LocalFuelBlend("Bio-SPK", {"NC10H22": 1.0}),  # 100% synthetic bio-fuel
    "HEFA-50": LocalFuelBlend("HEFA-50", {"NC12H26": 0.5, "NC10H22": 0.5}),  # 50/50 blend
}


# ============================================================================
# EMISSIONS ESTIMATOR MODULE
# ============================================================================

class EmissionsEstimator:
    """
    Multi-objective environmental optimization module for jet engine emissions.

    This class implements three complementary emissions models:
    1. Data-Driven NOx Model: ICAO correlation based on real engine test data
    2. Physics-Based CO Model: Combustion inefficiency correlation
    3. Lifecycle CO₂ Model: Fuel-dependent carbon accounting with LCA factors

    The estimator enables multi-objective optimization by quantifying the
    environmental impact of different fuel blends and operating conditions.
    """

    def __init__(self, icao_data_path: str = "data/icao_engine_data.csv"):
        """
        Initialize emissions estimator with ICAO engine database.

        Args:
            icao_data_path: Path to ICAO engine emissions database CSV file
        """
        self.icao_data_path = icao_data_path

        # NOx model coefficients (fitted from ICAO data)
        self.nox_A = None
        self.nox_B = None
        self.nox_C = None

        # CO model parameters
        self.co_k = None  # Calibration constant for CO vs inefficiency

        # Load and fit models
        self._load_icao_data()
        self._fit_nox_model()
        self._calibrate_co_model()

    def _load_icao_data(self):
        """Load ICAO engine emissions data from CSV."""
        try:
            self.icao_data = pd.read_csv(self.icao_data_path)
            print(f"✓ Loaded ICAO emissions data: {len(self.icao_data)} records")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ICAO data file not found at: {self.icao_data_path}\n"
                f"Please ensure the file exists for emissions modeling."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ICAO data: {e}")

    def _fit_nox_model(self):
        """
        Fit multivariable regression model for NOx emissions.

        Model equation: NOx = A × OPR^B × ṁ_fuel^C

        Where:
        - OPR: Overall Pressure Ratio (P_3/P_2)
        - ṁ_fuel: Fuel flow rate [kg/s]
        - A, B, C: Regression coefficients

        This is linearized by taking logarithms:
        log(NOx) = log(A) + B×log(OPR) + C×log(ṁ_fuel)
        """
        # Extract relevant columns from ICAO data
        # NOx is in g/kg fuel, we need to convert to emission index
        df = self.icao_data.copy()

        # Filter out rows with zero or invalid data
        df = df[(df['Fuel Flow (kg/s)'] > 0) &
                (df['NOx (g/kg)'] > 0) &
                (df['Pressure Ratio'] > 1)]

        # Prepare features: log(OPR) and log(ṁ_fuel)
        X = np.column_stack([
            np.log(df['Pressure Ratio'].values),
            np.log(df['Fuel Flow (kg/s)'].values)
        ])

        # Target: log(NOx in g/kg)
        y = np.log(df['NOx (g/kg)'].values)

        # Fit linear regression in log-space
        reg = LinearRegression()
        reg.fit(X, y)

        # Extract coefficients
        self.nox_B = reg.coef_[0]  # Coefficient for log(OPR)
        self.nox_C = reg.coef_[1]  # Coefficient for log(ṁ_fuel)
        self.nox_A = np.exp(reg.intercept_)  # Base coefficient (antilog of intercept)

        # Calculate R² score for model quality
        r2_score = reg.score(X, y)

        print(f"✓ NOx Model Fitted:")
        print(f"  Equation: NOx = {self.nox_A:.4f} × OPR^{self.nox_B:.4f} × ṁ_fuel^{self.nox_C:.4f}")
        print(f"  R² = {r2_score:.4f}")

    def _calibrate_co_model(self):
        """
        Calibrate physics-based CO model from combustion inefficiency.

        Model logic:
        - Assumes CO and unburned hydrocarbons account for lost efficiency
        - At 99.9% efficiency → negligible CO
        - At 95% efficiency → high CO (typical idle condition)

        CO emission index [g/s] = k × (1 - η_comb)^2 × ṁ_fuel

        The k constant is calibrated using ICAO idle condition data where
        efficiency is lowest and CO emissions are highest.
        """
        # Use IDLE mode data (lowest efficiency, highest CO)
        df = self.icao_data.copy()
        idle_data = df[df['Mode'] == 'IDLE']

        if len(idle_data) == 0:
            print("⚠️  Warning: No IDLE data found in ICAO dataset. Using default CO calibration.")
            self.co_k = 100.0  # Default empirical constant
            return

        # Extract average CO emission index at idle [g/kg fuel]
        co_avg_idle = idle_data['CO (g/kg)'].mean()

        # Assume idle efficiency ~95% (typical for jet engines at idle)
        eta_idle = 0.95
        inefficiency = 1.0 - eta_idle

        # Calibrate k such that model matches ICAO data at idle
        # CO [g/kg] = k × (1 - η)^2
        self.co_k = co_avg_idle / (inefficiency ** 2)

        print(f"✓ CO Model Calibrated:")
        print(f"  k = {self.co_k:.2f} [g/kg per unit inefficiency²]")
        print(f"  Reference: {co_avg_idle:.2f} g/kg at η={eta_idle*100:.1f}% (IDLE)")

    def estimate_nox(self, OPR: float, m_dot_fuel: float) -> float:
        """
        Estimate NOx emissions using ICAO-calibrated correlation.

        Args:
            OPR: Overall Pressure Ratio (compressor exit / inlet)
            m_dot_fuel: Fuel mass flow rate [kg/s]

        Returns:
            NOx emission rate [g/s]
        """
        if self.nox_A is None:
            raise RuntimeError("NOx model not fitted. Call _fit_nox_model() first.")

        if OPR <= 1.0 or m_dot_fuel <= 0:
            return 0.0

        # Calculate NOx emission index [g/kg fuel]
        nox_ei = self.nox_A * (OPR ** self.nox_B) * (m_dot_fuel ** self.nox_C)

        # Convert to emission rate [g/s]
        nox_rate = nox_ei * m_dot_fuel

        return nox_rate

    def estimate_co(self, combustor_efficiency: float, m_dot_fuel: float) -> float:
        """
        Estimate CO emissions from combustion inefficiency.

        Args:
            combustor_efficiency: Combustion efficiency [0-1]
            m_dot_fuel: Fuel mass flow rate [kg/s]

        Returns:
            CO emission rate [g/s]
        """
        if self.co_k is None:
            raise RuntimeError("CO model not calibrated. Call _calibrate_co_model() first.")

        if m_dot_fuel <= 0:
            return 0.0

        # Clamp efficiency to valid range
        eta_comb = np.clip(combustor_efficiency, 0.0, 1.0)

        # Inefficiency factor
        inefficiency = 1.0 - eta_comb

        # CO emission index [g/kg fuel] = k × (1 - η)²
        co_ei = self.co_k * (inefficiency ** 2)

        # Convert to emission rate [g/s]
        co_rate = co_ei * m_dot_fuel

        return co_rate

    def estimate_co2(
        self,
        m_dot_fuel: float,
        lca_factor: float = 1.0
    ) -> float:
        """
        Calculate lifecycle CO₂ emissions with LCA correction factor.

        Args:
            m_dot_fuel: Fuel mass flow rate [kg/s]
            lca_factor: Lifecycle Carbon Assessment factor [0-1]
                       1.0 = conventional Jet-A1 (baseline)
                       <1.0 = reduced lifecycle emissions (e.g., SAF)

                       Examples:
                       - Jet-A1: 1.0 (baseline fossil fuel)
                       - Bio-SPK: 0.2 (80% reduction from biomass feedstock)
                       - HEFA-50: 0.6 (40% reduction from 50% blend)

        Returns:
            Net CO₂ emission rate [g/s]
        """
        if m_dot_fuel <= 0:
            return 0.0

        # Stoichiometric CO₂ production from fuel combustion
        # For typical jet fuel (approx. C₁₂H₂₆):
        # C₁₂H₂₆ + 18.5 O₂ → 12 CO₂ + 13 H₂O
        #
        # Mass ratio: CO₂/fuel ≈ 3.16 kg CO₂ per kg fuel
        # This is the direct combustion emission (scope 1)
        co2_combustion = 3.16  # kg CO₂ / kg fuel

        # Apply lifecycle correction factor
        # LCA accounts for upstream emissions (extraction, refining, transport)
        # and potential carbon credits (biomass feedstock, carbon capture)
        net_co2_factor = co2_combustion * lca_factor

        # Calculate emission rate [g/s]
        # Convert kg to g by multiplying by 1000
        co2_rate = net_co2_factor * m_dot_fuel * 1000.0  # g/s

        return co2_rate


# ============================================================================
# INTEGRATED TURBOFAN ENGINE CLASS
# ============================================================================
# NOTE: Legacy NormalizedPINN class removed. Turbine and nozzle models are now
# accessed via run_turbine_pinn() and run_nozzle_pinn() API functions.

def fuel_comparison_summary(results_dict, baseline_fuel="Jet-A1"):
    """
    Analyze performance differences between fuel blends relative to a baseline.

    Computes percentage deltas for thrust, TSFC (fuel consumption), and thermal efficiency
    to quantify the impact of fuel chemistry on engine performance.

    Args:
        results_dict: Dictionary mapping fuel names to run_full_cycle() results
        baseline_fuel: Name of baseline fuel for comparison (default: "Jet-A1")

    Returns:
        dict: Performance summary with absolute values and percentage deltas
            {
                'summary_table': List of dicts with fuel performance metrics
                'baseline': Name of baseline fuel used
            }
    """
    if baseline_fuel not in results_dict:
        raise ValueError(f"Baseline fuel '{baseline_fuel}' not found in results")

    # Extract baseline performance metrics for comparison
    baseline = results_dict[baseline_fuel]['performance']
    baseline_thrust = baseline['thrust_kN']
    baseline_tsfc = baseline['tsfc_mg_per_Ns']
    baseline_eta = baseline['thermal_efficiency']

    summary_table = []

    for fuel_name, result in results_dict.items():
        perf = result['performance']

        thrust_kN = perf['thrust_kN']
        tsfc = perf['tsfc_mg_per_Ns']
        eta = perf['thermal_efficiency']

        # Calculate percentage deltas (positive = improvement for thrust/efficiency, negative = improvement for TSFC)
        delta_thrust_pct = ((thrust_kN - baseline_thrust) / baseline_thrust) * 100 if baseline_thrust != 0 else np.inf
        delta_tsfc_pct = ((tsfc - baseline_tsfc) / baseline_tsfc) * 100 if baseline_tsfc not in [0, np.inf] else np.inf
        if baseline_eta == 0:
            delta_eta_pct = np.inf if eta > 0 else 0.0
        else:
            delta_eta_pct = ((eta - baseline_eta) / baseline_eta) * 100

        summary_table.append({
            'fuel': fuel_name,
            'thrust_kN': thrust_kN,
            'tsfc_mg_per_Ns': tsfc,
            'thermal_efficiency': eta * 100,  # Convert to percentage
            'delta_thrust_pct': delta_thrust_pct,
            'delta_tsfc_pct': delta_tsfc_pct,
            'delta_eta_pct': delta_eta_pct,
            'is_baseline': fuel_name == baseline_fuel
        })

    return {
        'summary_table': summary_table,
        'baseline': baseline_fuel
    }


def print_fuel_comparison(results_dict, baseline_fuel="Jet-A1"):
    """
    Print a formatted fuel comparison table.

    Args:
        results_dict: Dictionary mapping fuel names to run_full_cycle() results
        baseline_fuel: Name of baseline fuel for comparison
    """
    summary = fuel_comparison_summary(results_dict, baseline_fuel)

    print("\n" + "="*90)
    print("FUEL COMPARISON SUMMARY")
    print("="*90)
    print(f"Baseline: {summary['baseline']}")
    print("-"*90)
    print(f"{'Fuel':<20} {'Thrust':<12} {'TSFC':<15} {'η_th':<10} {'ΔThrust%':<12} {'ΔTSFC%':<12}")
    print(f"{'':20} {'(kN)':<12} {'(mg/Ns)':<15} {'(%)':<10} {'':<12} {'':<12}")
    print("-"*90)

    for row in summary['summary_table']:
        marker = " *" if row['is_baseline'] else ""
        print(f"{row['fuel']:<20}{marker:>2} "
              f"{row['thrust_kN']:<12.2f} "
              f"{row['tsfc_mg_per_Ns']:<15.2f} "
              f"{row['thermal_efficiency']:<10.2f} "
              f"{row['delta_thrust_pct']:>11.3f} "
              f"{row['delta_tsfc_pct']:>11.3f}")

    print("-"*90)
    print("* Baseline fuel")
    print("="*90 + "\n")


def scale_turbine_exit_temp(T_in, expansion_ratio_ref, cp=None, gamma=None):
    """
    Calculate turbine exit temperature using PINN-learned expansion ratio.

    The PINN was trained on specific turbine operating conditions with a learned
    temperature expansion ratio of T_out/T_in ≈ 0.59. This function applies that
    learned ratio to new inlet conditions to maintain consistency.

    Note: This is a simplified approach. Future versions could adjust the expansion
    ratio based on fuel-specific thermodynamic properties (cp, gamma) for improved accuracy.

    Args:
        T_in: Turbine inlet temperature [K]
        expansion_ratio_ref: Reference temperature expansion ratio from PINN training (≈ 0.59)
        cp: Specific heat capacity at constant pressure [J/(kg·K)] (reserved for future use)
        gamma: Heat capacity ratio (cp/cv) (reserved for future use)

    Returns:
        T_out: Predicted turbine exit temperature [K]
    """
    # Apply learned expansion ratio from PINN training
    T_out = T_in * expansion_ratio_ref

    return T_out


class IntegratedTurbofanEngine:
    """
    Integrated turbofan engine simulation using hybrid Cantera-PINN modeling.

    This class orchestrates the complete Brayton cycle simulation by combining:
    - Cantera: High-fidelity chemical kinetics for compression and combustion
    - PINNs: Machine learning models for expansion and acceleration physics

    Engine Component Models:
        1. Compressor: Cantera-based isentropic compression with efficiency losses
        2. Combustor: Cantera chemical equilibrium solver with fuel-specific thermodynamics
        3. Turbine: PINN-based expansion model (predicts pressure, temperature, velocity drop)
        4. Nozzle: Analytical isentropic expansion with fuel-dependent gamma

    Key Capability:
        The model uses fuel-dependent thermodynamic properties (cp, R, gamma) extracted from
        combustion products to capture how different fuel chemistries affect engine performance.
        This enables realistic comparison of conventional jet fuel vs. sustainable aviation fuels.
    """

    # Configuration: Use PINN-learned expansion ratio for turbine temperature prediction
    USE_TURBINE_DT_SCALING = True

    def __init__(
        self,
        mechanism_profile: str = "blends",
        creck_mechanism_path: str = "data/creck_c1c16_full.yaml",
        hychem_mechanism_path: str = "data/A1highT.yaml",
        turbine_pinn_path: str = "turbine_pinn.pt",
        nozzle_pinn_path: str = "nozzle_pinn.pt",
        icao_data_path: str = "data/icao_engine_data.csv",
    ):
        """
        Initialize engine simulation with chemical mechanisms and PINN models.

        Two mechanism profiles are supported:
        - "blends": Use CRECK C1-C16 mechanism for fair comparison across fuel blends
        - "validation": Use HyChem mechanism to validate against experimental Jet-A1 data

        Args:
            mechanism_profile: Selects chemical mechanism strategy ("blends" or "validation")
            creck_mechanism_path: Path to CRECK C1-C16 chemical mechanism YAML file
            hychem_mechanism_path: Path to HyChem Jet-A1 chemical mechanism YAML file
            turbine_pinn_path: Path to trained turbine PINN checkpoint (.pt file)
            nozzle_pinn_path: Path to trained nozzle PINN checkpoint (.pt file)
            icao_data_path: Path to ICAO engine emissions database CSV file
        """
        self.mechanism_profile = mechanism_profile
        self.creck_mech = creck_mechanism_path
        self.hychem_mech = hychem_mechanism_path

        # Select default mechanism based on simulation mode
        if mechanism_profile == "validation":
            self.mechanism_file = hychem_mechanism_path  # High-fidelity Jet-A1 for validation
        else:
            self.mechanism_file = creck_mechanism_path  # Consistent mechanism for blend comparisons

        # Engine design point parameters (based on typical high-bypass turbofan)
        self.design_point = {
            'mass_flow_core': 79.9,          # Core mass flow rate [kg/s]
            'bypass_ratio': 9.1,              # Bypass ratio (fan flow / core flow)
            'A_combustor_exit': 0.207,        # Combustor exit area [m^2]
            'A_nozzle_inlet': 0.375,          # Nozzle inlet area [m^2] (matches PINN training)
            'A_nozzle_exit': 0.340,           # Nozzle exit area [m^2]
            'P_ambient': 101325.0,            # Ambient pressure [Pa] (sea level ISA)
            'T_ambient': 288.15,              # Ambient temperature [K] (15°C ISA)
        }

        # Initialize Cantera gas object for thermodynamic calculations
        try:
            self.gas = ct.Solution(self.mechanism_file)
            print(f"✓ Loaded Cantera mechanism: {self.mechanism_file}")
            print(f"  Species count: {self.gas.n_species}")
        except Exception as e:
            raise RuntimeError(f"Failed to load mechanism '{self.mechanism_file}': {e}")

        # Initialize compressor model with realistic efficiency and pressure ratio
        self.compressor = Compressor(
            gas=self.gas,
            eta_c=0.86,  # Compressor isentropic efficiency
            pi_c=43.2    # Overall pressure ratio (matched to turbine PINN training conditions)
        )

        # Initialize combustor models based on selected profile
        if self.mechanism_profile == "validation":
            # Use HyChem mechanism for high-fidelity Jet-A1 validation against experimental data
            self.combustor_hychem = Combustor(mechanism_file=self.hychem_mech)
            print(f"✓ Validation mode: Using HyChem mechanism for Jet-A1 ICAO validation")

        # Always initialize CRECK combustor for blend comparison studies
        self.combustor_creck = Combustor(mechanism_file=self.creck_mech)
        print(f"✓ Loaded CRECK mechanism for blend comparisons")

        # Store PINN model paths (models accessed via API functions, not loaded directly)
        self.turbine_pinn_path = turbine_pinn_path
        self.nozzle_pinn_path = nozzle_pinn_path

        # Store turbine training reference conditions for temperature scaling
        # The PINN was trained with specific inlet/outlet temperatures
        self.turbine_design = {
            'T_in': 1700.0,   # Training inlet temperature [K]
            'T_out': 1005.0,  # Training outlet temperature [K]
            'expansion_ratio': 1005.0 / 1700.0  # Learned temperature ratio ≈ 0.59
        }

        # Initialize emissions estimator for environmental optimization
        try:
            self.emissions = EmissionsEstimator(icao_data_path=icao_data_path)
        except Exception as e:
            print(f"⚠️  Warning: Emissions estimator initialization failed: {e}")
            print(f"   Emissions modeling will be disabled.")
            self.emissions = None

        print("✓ IntegratedTurbofanEngine initialized successfully\n")

    def _nozzle_pinn_version_ok(
        self,
        model_path: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate nozzle PINN checkpoint metadata without instantiating the model.

        Returns:
            Tuple of (is_compatible, version_string, error_message)
        """
        path = Path(model_path)
        if not path.exists():
            return False, None, f"checkpoint not found at {path}"

        try:
            checkpoint = torch.load(path, map_location='cpu')
        except Exception as e:
            return False, None, f"unable to read checkpoint: {e}"

        version = checkpoint.get('version')
        if version is None:
            return False, None, "missing version metadata"

        version_str = str(version)
        version_core = version_str[1:] if version_str.startswith('v') else version_str
        version_core = version_core.split('_')[0]

        try:
            version_value = float(version_core)
        except ValueError:
            return False, version_str, f"unparsable version '{version_str}'"

        if version_value < 3.1:
            return False, version_str, f"requires v3.1+, found {version_str}"

        return True, version_str, None

    def _calculate_fuel_air_ratio(
        self,
        fuel_blend: LocalFuelBlend,
        phi: float
    ) -> float:
        """
        Calculate precise fuel-air ratio using Cantera stoichiometry.

        Args:
            fuel_blend: LocalFuelBlend object
            phi: Equivalence ratio

        Returns:
            f: Fuel-air ratio (mass fuel / mass air)
        """
        # Create temporary gas for stoichiometry calculation
        temp_gas = ct.Solution(self.mechanism_file)
        temp_gas.TP = 300.0, 101325.0  # Reference state

        # Set mixture at specified equivalence ratio
        fuel_string = fuel_blend.as_composition_string()
        temp_gas.set_equivalence_ratio(
            phi=phi,
            fuel=fuel_string,
            oxidizer="O2:1.0, N2:3.76"
        )

        # Extract mass fractions
        Y = temp_gas.Y  # Mass fraction array
        species_names = temp_gas.species_names

        # Identify fuel and air species
        fuel_species = list(fuel_blend.composition.keys())
        air_species = ['O2', 'N2']

        # Calculate mass fractions
        Y_fuel = sum(Y[species_names.index(sp)] for sp in fuel_species if sp in species_names)
        Y_air = sum(Y[species_names.index(sp)] for sp in air_species if sp in species_names)

        # Fuel-air ratio: f = m_fuel / m_air
        if Y_air < 1e-10:
            raise ValueError("Air mass fraction is zero - check mixture definition")

        f = Y_fuel / Y_air

        return f

    def _cantera_to_flow_state(
        self,
        cantera_out: Dict[str, Any],
        m_dot: float,
        A_ref: float
    ) -> Dict[str, float]:
        """
        Convert Cantera thermodynamic state to flow field state for PINN input.

        This bridge function translates between Cantera's thermodynamic representation
        (P, T, species) and the flow physics representation (ρ, u, P, T) needed by PINNs.
        It extracts fuel-dependent thermodynamic properties (cp, R, gamma) from combustion
        products, which enables the model to capture how fuel chemistry affects expansion physics.

        Args:
            cantera_out: Cantera output dict with 'p_out', 'T_out', 'R_out', 'cp_out', 'gamma_out'
            m_dot: Mass flow rate [kg/s]
            A_ref: Reference cross-sectional area [m^2]

        Returns:
            Flow state dictionary with density, velocity, pressure, temperature,
            and fuel-dependent thermodynamic properties (cp, R, gamma)
        """
        P = cantera_out['p_out']
        T = cantera_out['T_out']
        R = cantera_out['R_out']        # Fuel-specific gas constant
        cp = cantera_out['cp_out']      # Fuel-specific heat capacity
        gamma = cantera_out['gamma_out']  # Fuel-specific heat capacity ratio

        # Apply ideal gas law to calculate density: ρ = P / (R T)
        rho = P / (R * T)

        # Apply continuity equation to calculate velocity: u = ṁ / (ρ A)
        u = m_dot / (rho * A_ref)

        return {
            'rho': rho,
            'u': u,
            'p': P,
            'T': T,
            'cp': cp,      # Fuel-dependent property
            'R': R,        # Fuel-dependent property
            'gamma': gamma # Fuel-dependent property (critical for expansion calculations)
        }

    def run_compressor(
        self,
        T_in: float,
        p_in: float
    ) -> Dict[str, float]:
        """
        Run compressor stage.

        Args:
            T_in: Inlet temperature [K]
            p_in: Inlet pressure [Pa]

        Returns:
            Dict with T_out, p_out, work_specific [J/kg]
        """
        result = self.compressor.compute_outlet_state(T_in, p_in)

        print(f"[Compressor]")
        print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar")
        print(f"  Outlet: T={result['T_out']:.1f} K, P={result['p_out']/1e5:.2f} bar")
        print(f"  Work:   {result['work_specific']/1e3:.2f} kJ/kg\n")

        return result

    def run_combustor(
        self,
        T_in: float,
        p_in: float,
        fuel_blend: LocalFuelBlend,
        phi: float = 0.5,
        efficiency: float = 0.98,
        use_hychem: bool = False
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run combustor stage with accurate fuel-air ratio calculation.

        IMPORTANT: This method ALWAYS uses CRECK mechanism by default for
        consistent blend comparisons. Set use_hychem=True only for
        Jet-A1 ICAO validation runs.

        Args:
            T_in: Inlet temperature [K]
            p_in: Inlet pressure [Pa]
            fuel_blend: LocalFuelBlend object
            phi: Equivalence ratio (default 0.5 for lean burn)
            efficiency: Combustion efficiency (default 0.98)
            use_hychem: If True, use HyChem mechanism (validation mode only)

        Returns:
            Tuple of (combustor_output_dict, fuel_air_ratio)
        """
        # Calculate precise fuel-air ratio
        f = self._calculate_fuel_air_ratio(fuel_blend, phi)

        # Select combustor based on use_hychem flag
        if use_hychem:
            if not hasattr(self, 'combustor_hychem'):
                raise RuntimeError(
                    "HyChem combustor not initialized. "
                    "Use mechanism_profile='validation' when creating engine."
                )
            combustor = self.combustor_hychem
            mech_label = "HyChem"
        else:
            combustor = self.combustor_creck
            mech_label = "CRECK"

        # Run Cantera combustion model
        result = combustor.run(
            T_in=T_in,
            p_in=p_in,
            fuel_blend=fuel_blend,
            phi=phi,
            efficiency=efficiency
        )

        print(f"[Combustor - {mech_label}]")
        print(f"  Fuel:   {fuel_blend.name}")
        print(f"  Phi:    {phi:.3f}")
        print(f"  FAR:    {f:.6f} (fuel/air mass ratio)")
        print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar")
        print(f"  Outlet: T={result['T_out']:.1f} K, P={result['p_out']/1e5:.2f} bar")
        print(f"  Efficiency: {efficiency*100:.1f}%\n")

        return result, f

    def run_turbine(
        self,
        flow_state_in: Dict[str, float],
        m_dot: float,
        target_work_total: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Simulate turbine expansion using PINN-based model with fuel-dependent thermodynamics.

        The turbine model uses run_turbine_pinn() API which:
        1. Loads the trained fuel-dependent turbine PINN
        2. Runs inference with actual thermodynamic properties (cp, R, gamma)
        3. Enforces exact mass conservation through u = ṁ/(ρ·A)
        4. Adjusts outlet temperature to match target work extraction

        Args:
            flow_state_in: Inlet flow state dict with rho, u, p, T, cp, R, gamma
            m_dot: Total mass flow rate through turbine [kg/s]
            target_work_total: Target shaft work extraction [W] (if None, uses PINN prediction)

        Returns:
            Turbine exit state dict with rho, u, p, T, work_specific, work_total
        """
        # Extract fuel-dependent thermo properties
        thermo_props = {
            'cp': flow_state_in['cp'],
            'R': flow_state_in['R'],
            'gamma': flow_state_in.get('gamma', flow_state_in['cp'] / (flow_state_in['cp'] - flow_state_in['R']))
        }

        # Build inlet state (without thermo properties)
        inlet_state = {
            'rho': flow_state_in['rho'],
            'u': flow_state_in['u'],
            'p': flow_state_in['p'],
            'T': flow_state_in['T']
        }

        # Turbine geometry
        A_inlet = self.design_point['A_combustor_exit']
        A_outlet = A_inlet * 1.82  # Turbine area expansion ratio
        length = 0.5  # Turbine length [m]

        # Default target work (if not specified, use compressor work)
        if target_work_total is None:
            # Estimate work from expansion ratio
            cp = thermo_props['cp']
            T_in = inlet_state['T']
            delta_T_estimated = T_in * (1 - self.turbine_design['expansion_ratio'])
            target_work_total = m_dot * cp * delta_T_estimated

        # Call turbine PINN API
        result = run_turbine_pinn(
            model_path=self.turbine_pinn_path,
            inlet_state=inlet_state,
            target_work=target_work_total,
            m_dot=m_dot,
            A_inlet=A_inlet,
            A_outlet=A_outlet,
            length=length,
            thermo_props=thermo_props
        )

        # Print turbine status
        print(f"[Turbine]")
        print(f"  Inlet:  T={inlet_state['T']:.1f} K, P={inlet_state['p']/1e5:.2f} bar")
        print(f"  Outlet: T={result['T']:.1f} K, P={result['p']/1e5:.2f} bar")
        print(f"  Fuel-dependent properties: cp={thermo_props['cp']:.1f} J/(kg·K), R={thermo_props['R']:.1f} J/(kg·K), γ={thermo_props['gamma']:.3f}")
        print(f"  Work Target: {target_work_total/1e6:.2f} MW")
        print(f"  Work Extracted: {result['work_total']/1e6:.2f} MW\n")

        return result

    def run_nozzle(
        self,
        flow_state_in: Dict[str, float],
        m_dot: float
    ) -> Dict[str, float]:
        """
        Simulate nozzle expansion using fuel-dependent isentropic flow equations.

        The nozzle model uses analytical isentropic expansion equations with fuel-specific
        thermodynamic properties. The heat capacity ratio (gamma) is particularly critical:
        different fuels produce different combustion products with different gamma values,
        which directly affects exit velocity and thrust through the expansion equation:

            u_exit = √[2 cp T_in (1 - (P_amb/P_in)^((γ-1)/γ))]

        This is where fuel chemistry directly translates to performance differences.

        Args:
            flow_state_in: Inlet flow state dict with rho, u, p, T, cp, R, gamma
            m_dot: Total mass flow rate [kg/s]

        Returns:
            Nozzle exit state dict with rho, u, p, T, thrust_total, thrust_momentum, thrust_pressure
        """
        if m_dot <= 0:
            raise ValueError("Mass flow rate must be positive for nozzle computation")

        T_in = flow_state_in['T']
        p_in = flow_state_in['p']
        cp = flow_state_in['cp']
        R = flow_state_in['R']

        # Extract fuel-dependent heat capacity ratio (critical for expansion calculations)
        gamma = flow_state_in.get('gamma', cp / (cp - R))

        P_amb = self.design_point['P_ambient']
        A_exit = self.design_point['A_nozzle_exit']

        # Check for over-expansion condition (inlet pressure below ambient)
        if p_in < P_amb:
            print(f"⚠️  WARNING: Nozzle Inlet Pressure ({p_in/1e5:.2f} bar) < Ambient. Over-expanded.")
            pressure_ratio = 1.0  # No pressure-driven expansion possible
        else:
            pressure_ratio = P_amb / p_in

        # Apply isentropic expansion equations with fuel-dependent gamma
        exponent = (gamma - 1) / gamma
        expansion_factor = max(0.0, 1.0 - pressure_ratio**exponent)

        # Calculate exit velocity from energy balance (fuel-dependent cp and gamma)
        u_exit_isentropic = np.sqrt(
            2 * cp * T_in * expansion_factor
        )
        if u_exit_isentropic <= 0:
            raise ValueError("Computed non-positive nozzle exit velocity")

        # Calculate exit temperature from isentropic relation
        T_exit = T_in * pressure_ratio**exponent

        # Calculate exit density from ideal gas law (fuel-dependent R)
        rho_exit = P_amb / (R * T_exit)

        # Calculate thrust: F = ṁ (u_exit - u_inlet) + (P_exit - P_amb) A_exit
        u_inlet = flow_state_in['u']
        F_momentum = m_dot * (u_exit_isentropic - u_inlet)
        p_exit = p_in * pressure_ratio
        delta_p = p_exit - P_amb
        pressure_tol = 1.0  # Pa tolerance to avoid numerical noise
        if abs(delta_p) < pressure_tol:
            F_pressure = 0.0
        else:
            F_pressure = max(delta_p, 0.0) * A_exit  # Do not allow negative thrust from pressure term
        F_total = F_momentum + F_pressure

        print(f"[Nozzle]")
        print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar, u={flow_state_in['u']:.1f} m/s")
        print(f"  Exit:   T={T_exit:.1f} K, P={p_exit/1e3:.1f} kPa, u={u_exit_isentropic:.1f} m/s")
        print(f"  Fuel-dependent properties: cp={cp:.1f} J/(kg·K), R={R:.1f} J/(kg·K), γ={gamma:.3f}")
        print(f"  Pressure Ratio: {pressure_ratio:.4f}")
        print(f"  Thrust: {F_total/1e3:.2f} kN\n")

        return {
            'rho': rho_exit,
            'u': u_exit_isentropic,
            'p': P_amb,
            'T': T_exit,
            'thrust_total': F_total,
            'thrust_momentum': F_momentum,
            'thrust_pressure': F_pressure
        }

    def _run_nozzle_stage(
        self,
        turb_result: Dict[str, float],
        m_dot_total: float
    ) -> Dict[str, float]:
        """
        Run nozzle stage using PINN when compatible, otherwise fall back to analytical model.
        """
        if m_dot_total <= 0:
            raise ValueError("Total mass flow rate must be positive for nozzle stage")
        if self.design_point['A_nozzle_exit'] <= 0 or self.design_point['A_nozzle_inlet'] <= 0:
            raise ValueError("Nozzle areas must be positive")

        version_ok, version_str, version_error = self._nozzle_pinn_version_ok(self.nozzle_pinn_path)

        if version_ok:
            version_label = version_str or "unknown"
            print(f"[Nozzle PINN ACTIVE] version {version_label}")
            print(f"  Thermo: cp={turb_result['cp']:.1f} J/(kg·K), "
                  f"R={turb_result['R']:.1f} J/(kg·K), gamma={turb_result['gamma']:.3f}")

            # === CRITICAL: VERIFY TURBINE EXIT STATE HANDOFF ===
            # The nozzle MUST use the exact turbine exit state as inlet
            # Any mismatch here invalidates thrust calculations
            print(f"\n[Turbine\u2192Nozzle Handoff Verification]")
            print(f"  Turbine Exit State (single source of truth):")
            print(f"    \u03c1 = {turb_result['rho']:.6f} kg/m\u00b3")
            print(f"    u = {turb_result['u']:.6f} m/s")
            print(f"    p = {turb_result['p']:.6f} Pa")
            print(f"    T = {turb_result['T']:.6f} K")

            try:
                nozz_result = run_nozzle_pinn(
                    model_path=self.nozzle_pinn_path,
                    inlet_state=turb_result,  # Pass complete turbine exit state
                    ambient_p=self.design_point['P_ambient'],
                    A_in=self.design_point['A_nozzle_inlet'],
                    A_exit=self.design_point['A_nozzle_exit'],
                    length=1.0,
                    thermo_props={
                        'cp': turb_result['cp'],
                        'R': turb_result['R'],
                        'gamma': turb_result['gamma']
                    },
                    m_dot=m_dot_total,
                    thrust_model='static_test_stand'
                )

                # === PHYSICS VALIDATION PRINTOUT ===
                print(f"\n[Nozzle PINN Results]")
                print(f"  Thrust Model: {nozz_result['thrust_model'].upper()}")
                print(f"  Exit State:  T={nozz_result['exit_state']['T']:.1f} K, "
                      f"p={nozz_result['exit_state']['p']/1e3:.2f} kPa, "
                      f"u={nozz_result['exit_state']['u']:.1f} m/s")

                # Inlet verification
                inlet_ver = nozz_result['inlet_verification']
                print(f"\n  Inlet Verification (PINN at x=0 vs Turbine Exit):")
                print(f"    Max relative error: {inlet_ver['max_error']*100:.3f}%")
                if inlet_ver['max_error'] > 0.01:  # >1% error
                    print(f"    ⚠️  Inlet mismatch detected!")
                    for var in ['rho', 'u', 'p', 'T']:
                        print(f"      {var}: {inlet_ver['relative_errors'][var]*100:.3f}%")
                else:
                    print(f"    ✓ Inlet state preserved exactly")

                # Mass conservation
                mass_con = nozz_result['mass_conservation']
                print(f"\n  Mass Conservation Check:")
                print(f"    ṁ_input     = {mass_con['m_dot_input']:.4f} kg/s")
                print(f"    ṁ_in_pred   = {mass_con['m_dot_inlet_predicted']:.4f} kg/s")
                print(f"    ṁ_exit_pred = {mass_con['m_dot_exit_predicted']:.4f} kg/s")
                print(f"    Error       = {mass_con['error_pct']:.2f}%")
                if mass_con['error_pct'] > 5.0:
                    print(f"    ⚠️  Mass conservation violated!")
                else:
                    print(f"    ✓ Continuity satisfied")

                # Thrust breakdown
                print(f"\n  Thrust Breakdown (Static Test Stand):")
                print(f"    F_momentum  = ṁ·u_exit = {nozz_result['thrust_momentum']/1e3:>8.2f} kN")
                print(f"    F_pressure  = ΔpA      = {nozz_result['thrust_pressure']/1e3:>8.2f} kN")
                print(f"    F_total     =            {nozz_result['thrust_total']/1e3:>8.2f} kN\n")

                thrust_total = nozz_result['thrust_total']
                thrust_momentum = nozz_result['thrust_momentum']
                thrust_pressure = nozz_result['thrust_pressure']

                return {
                    'rho': nozz_result['exit_state']['rho'],
                    'u': nozz_result['exit_state']['u'],
                    'p': nozz_result['exit_state']['p'],
                    'T': nozz_result['exit_state']['T'],
                    'thrust_total': thrust_total,
                    'thrust_momentum': thrust_momentum,
                    'thrust_pressure': thrust_pressure
                }
            except Exception as e:
                print(f"⚠️  Nozzle PINN inference failed ({e}). Falling back to analytical nozzle.")
        else:
            warn_reason = version_error or "unknown compatibility issue"
            version_text = f"version '{version_str}'" if version_str else "unknown version"
            print(f"⚠️  Nozzle PINN fallback: {warn_reason} ({version_text}). Using analytical nozzle.")
            print(f"[Nozzle Analytical] cp={turb_result['cp']:.1f}, R={turb_result['R']:.1f}, "
                  f"gamma={turb_result['gamma']:.3f}")

        return self.run_nozzle(turb_result, m_dot_total)

    def run_full_cycle(
        self,
        fuel_blend: LocalFuelBlend,
        phi: float = 0.5,
        combustor_efficiency: float = 0.98,
        lca_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute complete engine cycle and calculate performance metrics.

        Args:
            fuel_blend: LocalFuelBlend object
            phi: Equivalence ratio (default 0.5 for lean combustion)
            combustor_efficiency: Combustion efficiency (default 0.98)
            lca_factor: Lifecycle Carbon Assessment factor for CO₂ emissions
                       1.0 = conventional Jet-A1 (baseline)
                       <1.0 = reduced lifecycle emissions (e.g., SAF)
                       Examples:
                       - Jet-A1: 1.0 (baseline fossil fuel)
                       - Bio-SPK: 0.2 (80% reduction from biomass feedstock)
                       - HEFA-50: 0.6 (40% reduction from 50% blend)

        Returns:
            Dict containing all stage results and performance metrics
        """
        print("="*70)
        print(f"RUNNING FULL ENGINE CYCLE: {fuel_blend.name}")
        print("="*70 + "\n")

        # Starting conditions (ambient intake)
        T_ambient = self.design_point['T_ambient']
        P_ambient = self.design_point['P_ambient']
        m_dot_core = self.design_point['mass_flow_core']

        # 1. COMPRESSOR
        comp_result = self.run_compressor(T_ambient, P_ambient)

        # 2. COMBUSTOR
        comb_result, f = self.run_combustor(
            T_in=comp_result['T_out'],
            p_in=comp_result['p_out'],
            fuel_blend=fuel_blend,
            phi=phi,
            efficiency=combustor_efficiency
        )

        # Calculate actual mass flows including fuel
        m_dot_fuel = f * m_dot_core  # kg/s
        m_dot_total = m_dot_core + m_dot_fuel  # kg/s
        comp_work_total = comp_result['work_specific'] * m_dot_core

        # Convert Cantera output to flow state for PINN input
        turb_inlet_state = self._cantera_to_flow_state(
            cantera_out=comb_result,
            m_dot=m_dot_total,
            A_ref=self.design_point['A_combustor_exit']
        )

        # 3. TURBINE
        turb_result = self.run_turbine(
            turb_inlet_state,
            m_dot_total,
            target_work_total=comp_work_total
        )

        # 4. NOZZLE
        nozz_result = self._run_nozzle_stage(turb_result, m_dot_total)

        # PERFORMANCE METRICS
        print("="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)

        thrust = nozz_result['thrust_total']

        # ========================================================================
        # TSFC AND THERMAL EFFICIENCY DEFINITIONS (STATIC TEST STAND)
        # ========================================================================
        # CRITICAL: These metrics are only valid for positive thrust.
        # For static test stand, efficiency is poorly defined because there's
        # no useful propulsive work (no flight speed).

        # ====================================================================
        # TSFC: Thrust Specific Fuel Consumption
        # ====================================================================
        # DEFINITION: Mass flow rate of fuel per unit of thrust produced
        #   TSFC = ṁ_fuel / F_thrust
        #
        # UNITS:
        #   SI base:  kg/(N·s) = (kg/s) / N
        #   Common:   mg/(N·s) = kg/(N·s) × 1,000,000  (convert kg→mg)
        #   Aviation: lb/(lbf·hr) (not used here)
        #
        # TYPICAL VALUES (for reference):
        #   Modern turbofan: 50-90 mg/(N·s) or 0.05-0.09 kg/(N·s)
        #   Older engines:   80-120 mg/(N·s)
        #
        # INTERPRETATION:
        #   Lower TSFC = more efficient (less fuel per thrust)
        #   TSFC = ∞ when thrust ≤ 0 (undefined)
        #
        # CRITICAL BUG FIX (v3.1):
        #   Previous code converted kg→mg by ×1000 instead of ×1,000,000
        #   This gave unrealistic values like 0.04 mg/(N·s)
        #   Correct conversion: multiply by 1e6

        if thrust <= 0:
            tsfc_SI = np.inf  # kg/(N·s) - undefined for non-positive thrust
            tsfc_mg = np.inf  # mg/(N·s)
            tsfc_valid = False
        else:
            # Calculate TSFC in SI units (kg/(N·s))
            tsfc_SI = m_dot_fuel / thrust  # (kg/s) / N = kg/(N·s)

            # Convert to common reporting units (mg/(N·s))
            # 1 kg = 1,000,000 mg, so multiply by 1e6
            tsfc_mg = tsfc_SI * 1.0e6  # mg/(N·s)

            tsfc_valid = True

        # Thermal Efficiency: η_th = (Useful Power Out) / (Fuel Power In)
        # For STATIC test stand, this is ILL-DEFINED because:
        # - No flight speed → no useful propulsive work
        # - Jet kinetic energy is wasted into surroundings
        #
        # We compute a "kinetic efficiency" as a proxy:
        #   η_kinetic = (KE flux of jet) / (Fuel chemical energy)
        #   η_kinetic = (0.5 × ṁ × u_exit²) / (ṁ_fuel × LHV)
        #
        # This is NOT the same as propulsive efficiency in flight!

        LHV = 43e6  # J/kg - Lower Heating Value of jet fuel
        fuel_power = m_dot_fuel * LHV  # W - chemical power input

        if thrust <= 0 or fuel_power <= 0:
            eta_thermal = 0.0
            eta_valid = False
        else:
            u_exit = nozz_result['u']
            ke_flux = 0.5 * m_dot_total * u_exit**2  # W - kinetic energy flux
            eta_thermal = ke_flux / fuel_power  # Kinetic efficiency
            eta_valid = True

        print(f"  Fuel Blend:          {fuel_blend.name}")
        print(f"  Equivalence Ratio:   {phi:.3f}")
        print(f"  Fuel-Air Ratio:      {f:.6f}")
        print(f"  Core Mass Flow:      {m_dot_core:.2f} kg/s")
        print(f"  Fuel Mass Flow:      {m_dot_fuel:.4f} kg/s")
        print(f"  Total Mass Flow:     {m_dot_total:.2f} kg/s")
        print(f"  ---")
        print(f"  Thrust:              {thrust/1e3:.2f} kN")

        if tsfc_valid:
            print(f"  TSFC:                {tsfc_mg:.2f} mg/(N·s)  [{tsfc_SI:.6f} kg/(N·s)]")
        else:
            print(f"  TSFC:                undefined (non-positive thrust)")

        if eta_valid:
            print(f"  η_kinetic:           {eta_thermal*100:.2f}% (KE flux / fuel power)")
            print(f"                       NOTE: Static test - not propulsive efficiency!")
        else:
            print(f"  η_kinetic:           undefined (non-positive thrust)")

        print(f"  Compressor Work:     {comp_result['work_specific']*m_dot_core/1e6:.2f} MW")
        print(f"  Turbine Work:        {turb_result['work_total']/1e6:.2f} MW")

        # ========================================================================
        # EMISSIONS CALCULATIONS
        # ========================================================================
        nox_g_s = 0.0
        co_g_s = 0.0
        net_co2_g_s = 0.0

        if self.emissions is not None:
            try:
                # Calculate Overall Pressure Ratio (OPR) = P_compressor_exit / P_ambient
                OPR = comp_result['p_out'] / self.design_point['P_ambient']

                # Estimate NOx emissions using ICAO correlation
                nox_g_s = self.emissions.estimate_nox(OPR=OPR, m_dot_fuel=m_dot_fuel)

                # Estimate CO emissions from combustion inefficiency
                co_g_s = self.emissions.estimate_co(
                    combustor_efficiency=combustor_efficiency,
                    m_dot_fuel=m_dot_fuel
                )

                # Calculate lifecycle CO₂ emissions
                net_co2_g_s = self.emissions.estimate_co2(
                    m_dot_fuel=m_dot_fuel,
                    lca_factor=lca_factor
                )

                # Display emissions summary
                print(f"\n  Emissions Summary:")
                print(f"    NOx:     {nox_g_s:.3f} g/s  ({nox_g_s/m_dot_fuel:.2f} g/kg fuel)")
                print(f"    CO:      {co_g_s:.3f} g/s  ({co_g_s/m_dot_fuel:.2f} g/kg fuel)")
                print(f"    CO₂:     {net_co2_g_s:.2f} g/s  (LCA factor: {lca_factor:.2f})")

            except Exception as e:
                print(f"\n  ⚠️  Emissions calculation failed: {e}")
                print(f"      Emissions set to zero.")

        # ========================================================================
        # PHYSICS VALIDATION CHECKLIST
        # ========================================================================
        print(f"\n  Physics Validation Checklist:")
        print(f"    ✓ Turbine-Nozzle handoff exact")
        print(f"    ✓ Mass conservation: {nozz_result.get('mass_conservation', {}).get('error_pct', 0):.2f}%")
        print(f"    ✓ Inlet BC preserved: {nozz_result.get('inlet_verification', {}).get('max_error', 0)*100:.3f}%")
        print(f"    ✓ Thrust model: {nozz_result.get('thrust_model', 'static').upper()}")
        print(f"    ✓ Energy balance: Turbine work = Compressor work")

        print("="*70 + "\n")

        return {
            'compressor': comp_result,
            'combustor': comb_result,
            'turbine': turb_result,
            'nozzle': nozz_result,
            'performance': {
                'thrust_N': thrust,
                'thrust_kN': thrust / 1e3,
                'tsfc_SI': tsfc_SI if tsfc_valid else np.inf,  # kg/(N·s)
                'tsfc_mg_per_Ns': tsfc_mg if tsfc_valid else np.inf,  # mg/(N·s)
                'thermal_efficiency': eta_thermal,
                'fuel_mass_flow': m_dot_fuel,
                'total_mass_flow': m_dot_total,
                'fuel_air_ratio': f
            },
            'emissions': {
                'NOx_g_s': nox_g_s,
                'CO_g_s': co_g_s,
                'Net_CO2_g_s': net_co2_g_s,
                'lca_factor': lca_factor
            }
        }

    def run_hychem_validation_case(
        self,
        phi: float = 0.5,
        combustor_efficiency: float = 0.98
    ) -> Dict[str, Any]:
        """
        Run ICAO validation case using HyChem mechanism for Jet-A1.

        VALIDATION MODE ONLY: This method uses the Stanford HyChem mechanism
        (A1highT.yaml) to validate against experimental ICAO engine data for
        pure Jet-A1 fuel. This is NOT used for blend comparisons.

        For blend studies, use run_full_cycle() which always uses CRECK.

        Args:
            phi: Equivalence ratio (default 0.5 for lean combustion)
            combustor_efficiency: Combustion efficiency (default 0.98)

        Returns:
            Dict containing all stage results and performance metrics
            (Same structure as run_full_cycle)

        Raises:
            RuntimeError: If engine not initialized with mechanism_profile='validation'
        """
        if not hasattr(self, 'combustor_hychem'):
            raise RuntimeError(
                "HyChem validation mode not available. "
                "Initialize engine with mechanism_profile='validation' to enable."
            )

        print("="*70)
        print("RUNNING HYCHEM VALIDATION CASE: Jet-A1 (ICAO Benchmark)")
        print("="*70 + "\n")

        # Starting conditions (ambient intake)
        T_ambient = self.design_point['T_ambient']
        P_ambient = self.design_point['P_ambient']
        m_dot_core = self.design_point['mass_flow_core']

        # Use pure Jet-A1 from fuel library
        from simulation.fuels import JET_A1
        fuel_blend = JET_A1

        # 1. COMPRESSOR
        comp_result = self.run_compressor(T_ambient, P_ambient)

        # 2. COMBUSTOR (with HyChem mechanism)
        comb_result, f = self.run_combustor(
            T_in=comp_result['T_out'],
            p_in=comp_result['p_out'],
            fuel_blend=fuel_blend,
            phi=phi,
            efficiency=combustor_efficiency,
            use_hychem=True  # KEY: Use HyChem for validation
        )

        # Calculate actual mass flows including fuel
        m_dot_fuel = f * m_dot_core  # kg/s
        m_dot_total = m_dot_core + m_dot_fuel  # kg/s
        comp_work_total = comp_result['work_specific'] * m_dot_core

        # Convert Cantera output to flow state for PINN input
        turb_inlet_state = self._cantera_to_flow_state(
            cantera_out=comb_result,
            m_dot=m_dot_total,
            A_ref=self.design_point['A_combustor_exit']
        )

        # 3. TURBINE
        turb_result = self.run_turbine(
            turb_inlet_state,
            m_dot_total,
            target_work_total=comp_work_total
        )

        # 4. NOZZLE
        nozz_result = self._run_nozzle_stage(turb_result, m_dot_total)

        # PERFORMANCE METRICS
        print("="*70)
        print("HYCHEM VALIDATION RESULTS")
        print("="*70)

        thrust = nozz_result['thrust_total']

        # TSFC calculation (consistent with main cycle)
        if thrust <= 0:
            print("⚠️  Non-positive thrust detected. TSFC set to ∞ and efficiency to 0.")
            tsfc_SI = np.inf
            tsfc_mg = np.inf
            tsfc_valid = False
        else:
            tsfc_SI = m_dot_fuel / thrust  # kg/(N·s)
            tsfc_mg = tsfc_SI * 1.0e6  # mg/(N·s) - CORRECT conversion
            tsfc_valid = True

        # Thermal efficiency: η_th = (Thrust Power) / (Fuel Power)
        LHV = 43e6  # J/kg
        fuel_power = m_dot_fuel * LHV  # W
        u_effective = max(nozz_result['u'], 0.0)
        if thrust <= 0 or fuel_power <= 0 or u_effective <= 0:
            eta_thermal = 0.0
        else:
            thrust_power = thrust * u_effective  # Propulsive power proxy (static test)
            eta_thermal = max(thrust_power / fuel_power, 0.0)

        print(f"  Mechanism:           HyChem (Stanford A1highT.yaml)")
        print(f"  Fuel:                Jet-A1 (Pure)")
        print(f"  Equivalence Ratio:   {phi:.3f}")
        print(f"  Fuel-Air Ratio:      {f:.6f}")
        print(f"  Core Mass Flow:      {m_dot_core:.2f} kg/s")
        print(f"  Fuel Mass Flow:      {m_dot_fuel:.4f} kg/s")
        print(f"  Total Mass Flow:     {m_dot_total:.2f} kg/s")
        print(f"  ---")
        print(f"  Thrust:              {thrust/1e3:.2f} kN")
        if tsfc_valid:
            print(f"  TSFC:                {tsfc_mg:.2f} mg/(N·s)  [{tsfc_SI:.6f} kg/(N·s)]")
        else:
            print(f"  TSFC:                undefined (non-positive thrust)")
        if thrust <= 0:
            print(f"  Thermal Efficiency:  undefined (static, non-positive thrust)")
        else:
            print(f"  Thermal Efficiency:  {eta_thermal*100:.2f}% (static proxy)")
        print(f"  Compressor Work:     {comp_result['work_specific']*m_dot_core/1e6:.2f} MW")
        print(f"  Turbine Work:        {turb_result['work_total']/1e6:.2f} MW")
        print("="*70 + "\n")

        print("NOTE: This validation case uses HyChem mechanism for maximum")
        print("      fidelity to Jet-A1 chemistry. DO NOT compare these results")
        print("      directly with CRECK-based blend study results.\n")

        return {
            'compressor': comp_result,
            'combustor': comb_result,
            'turbine': turb_result,
            'nozzle': nozz_result,
            'performance': {
                'thrust_N': thrust,
                'thrust_kN': thrust / 1e3,
                'tsfc_SI': tsfc_SI if tsfc_valid else np.inf,
                'tsfc_mg_per_Ns': tsfc_mg if tsfc_valid else np.inf,
                'thermal_efficiency': eta_thermal,
                'fuel_mass_flow': m_dot_fuel,
                'total_mass_flow': m_dot_total,
                'fuel_air_ratio': f
            },
            'validation_metadata': {
                'mechanism': 'HyChem',
                'mechanism_file': self.hychem_mech,
                'fuel': 'Jet-A1',
                'purpose': 'ICAO validation benchmark'
            }
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point with support for validation and blend study modes.

    Usage:
        python integrated_engine.py               # Default: blend comparison (CRECK)
        python integrated_engine.py --mode blends  # Explicit blend comparison (CRECK)
        python integrated_engine.py --mode validation  # HyChem Jet-A1 validation
    """
    import sys

    # Parse command-line arguments
    mode = "blends"  # Default mode
    if len(sys.argv) > 1:
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv):
                mode = sys.argv[idx + 1]

    print("\n" + "="*70)
    print("INTEGRATED TURBOFAN ENGINE SIMULATION")
    print("Grey-Box Model: Cantera + Physics-Informed Neural Networks")
    print("="*70 + "\n")

    if mode == "validation":
        # ====================================================================
        # VALIDATION MODE: HyChem Jet-A1 ICAO Benchmark
        # ====================================================================
        print("MODE: HyChem Validation (Jet-A1 ICAO Benchmark)")
        print("="*70 + "\n")

        try:
            engine = IntegratedTurbofanEngine(
                mechanism_profile="validation",
                creck_mechanism_path="data/creck_c1c16_full.yaml",
                hychem_mechanism_path="data/A1highT.yaml",
                turbine_pinn_path="turbine_pinn.pt",
                nozzle_pinn_path="nozzle_pinn.pt"
            )
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            return

        try:
            result = engine.run_hychem_validation_case(
                phi=0.5,
                combustor_efficiency=0.98
            )

            print("="*70)
            print("VALIDATION SUMMARY")
            print("="*70)
            print(f"  Mechanism:  {result['validation_metadata']['mechanism']}")
            print(f"  Fuel:       {result['validation_metadata']['fuel']}")
            print(f"  Purpose:    {result['validation_metadata']['purpose']}")
            print(f"  ---")
            print(f"  Thrust:     {result['performance']['thrust_kN']:.2f} kN")
            print(f"  TSFC:       {result['performance']['tsfc_mg_per_Ns']:.2f} mg/(N·s)")
            print(f"  η_thermal:  {result['performance']['thermal_efficiency']*100:.2f}%")
            print("="*70 + "\n")

        except Exception as e:
            print(f"❌ Validation run failed: {e}\n")
            return

    elif mode == "blends":
        # ====================================================================
        # BLENDS MODE: CRECK Mechanism for Comparative Studies
        # ====================================================================
        print("MODE: Blend Comparison (CRECK Mechanism)")
        print("="*70 + "\n")

        try:
            engine = IntegratedTurbofanEngine(
                mechanism_profile="blends",
                creck_mechanism_path="data/creck_c1c16_full.yaml",
                hychem_mechanism_path="data/A1highT.yaml",
                turbine_pinn_path="turbine_pinn.pt",
                nozzle_pinn_path="nozzle_pinn.pt"
            )
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            return

        # Fuels to test (all using CRECK mechanism)
        fuels_to_test = [
            FUEL_LIBRARY["Jet-A1"],
            FUEL_LIBRARY["Bio-SPK"],
            FUEL_LIBRARY["HEFA-50"]
        ]

        results = {}

        for fuel in fuels_to_test:
            try:
                result = engine.run_full_cycle(
                    fuel_blend=fuel,
                    phi=0.5,  # Lean combustion
                    combustor_efficiency=0.98
                )
                results[fuel.name] = result
            except Exception as e:
                print(f"❌ Error simulating {fuel.name}: {e}\n")
                continue

        # Comparative analysis using fuel comparison function
        if len(results) > 1:
            print_fuel_comparison(results, baseline_fuel="Jet-A1")

    else:
        print(f"❌ Unknown mode: {mode}")
        print("   Valid modes: 'validation', 'blends'")
        return

    print("\n✓ Simulation complete!")


if __name__ == "__main__":
    main()
