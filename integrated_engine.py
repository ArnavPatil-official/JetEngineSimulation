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
    from simulation.fan import Fan
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

    def __init__(self,
                 icao_data_path: str = "data/icao_engine_data.csv",
                 nox_fit_exclude_models=None):
        """
        Initialize emissions estimator with ICAO engine database.

        Args:
            icao_data_path: Path to ICAO engine emissions database CSV file
            nox_fit_exclude_models: Optional iterable of engine MODEL names
                (e.g. {"Trent 1000-AE3"}) to hold out of the NOx correlation
                fit. Default None reproduces the production fit over the whole
                databank. Used by scripts/validation/nox_holdout_validation.py
                to run leave-one-engine-out cross-validation; the fit is
                otherwise in-sample and must not be reported as validated.
        """
        self.icao_data_path = icao_data_path
        self.nox_fit_exclude_models = (
            set(nox_fit_exclude_models) if nox_fit_exclude_models else set()
        )

        # NOx model coefficients (fitted from ICAO data)
        self.nox_A = None
        self.nox_B = None
        self.nox_C = None
        # Fit diagnostics -- IN-SAMPLE, never a validation statistic
        self.nox_fit_n = None
        self.nox_fit_n_models = None
        self.nox_fit_r2_in_sample = None

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
            print(f"[OK] Loaded ICAO emissions data: {len(self.icao_data)} records")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ICAO data file not found at: {self.icao_data_path}\n"
                f"Please ensure the file exists for emissions modeling."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ICAO data: {e}")

    @staticmethod
    def engine_model_name(engine_id: str) -> str:
        """Strip the ' BYPASS RATIO: x.x' suffix from an ICAO 'Engine ID' cell."""
        return str(engine_id).split(' BYPASS')[0].strip()

    def _fit_nox_model(self):
        """
        Fit the ICAO-derived NOx correlation.

        Model equation: EI_NOx = A × OPR^B × ṁ_fuel^C

        Where:
        - OPR: Overall Pressure Ratio (P_3/P_2)
        - ṁ_fuel: Fuel flow rate [kg/s]
        - A, B, C: Regression coefficients

        This is linearized by taking logarithms:
        log(EI_NOx) = log(A) + B×log(OPR) + C×log(ṁ_fuel)

        SCOPE AND LIMITS (read before quoting any NOx number):
        - This is a CORRELATION over certificated engines, not chemistry. It has
          no fuel-composition dependence whatsoever, so it cannot rank fuel
          blends on NOx; any blend-to-blend NOx difference it produces comes
          from the blend's effect on ṁ_fuel alone.
        - The R² reported below is IN-SAMPLE. It is a training diagnostic, not
          evidence of predictive skill. (An in-sample R² of this fit, printed at
          startup, was the origin of the withdrawn "R² = 0.9969" Highlight.)
          Held-out skill is measured by leave-one-engine-out cross-validation in
          scripts/validation/nox_holdout_validation.py.
        - A chemistry comparison path (Zeldovich thermal NO) lives in
          simulation/nox_chemistry.py; the three-path divergence is recorded in
          outputs/nox_dual_path.csv.
        """
        # Extract relevant columns from ICAO data
        # NOx is in g/kg fuel, we need to convert to emission index
        df = self.icao_data.copy()

        # Filter out rows with zero or invalid data
        df = df[(df['Fuel Flow (kg/s)'] > 0) &
                (df['NOx (g/kg)'] > 0) &
                (df['Pressure Ratio'] > 1)]

        # Optional held-out split for cross-validation (default: fit on all)
        df['_model'] = df['Engine ID'].map(self.engine_model_name)
        if self.nox_fit_exclude_models:
            df = df[~df['_model'].isin(self.nox_fit_exclude_models)]
        if len(df) < 3:
            raise ValueError(
                "NOx fit needs at least 3 usable ICAO records after filtering; "
                f"got {len(df)} (excluded models: "
                f"{sorted(self.nox_fit_exclude_models)})"
            )

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

        # In-sample R² -- a training diagnostic ONLY (see docstring)
        r2_score = reg.score(X, y)
        self.nox_fit_n = int(len(df))
        self.nox_fit_n_models = int(df['_model'].nunique())
        self.nox_fit_r2_in_sample = float(r2_score)

        print(f"[OK] NOx correlation fitted (ICAO databank):")
        print(f"  EI_NOx = {self.nox_A:.4f} × OPR^{self.nox_B:.4f} × ṁ_fuel^{self.nox_C:.4f}")
        print(f"  fit set: {self.nox_fit_n} records / {self.nox_fit_n_models} engine models"
              + (f" (excluded: {sorted(self.nox_fit_exclude_models)})"
                 if self.nox_fit_exclude_models else ""))
        print(f"  in-sample R² = {r2_score:.4f}  <- TRAINING DIAGNOSTIC, NOT VALIDATION")
        print(f"  no fuel-composition dependence: cannot rank blends on NOx")

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

    # Default carbon mass fraction (pure n-dodecane surrogate) used when a
    # blend's composition is unavailable
    DEFAULT_CARBON_FRACTION = 0.8461

    def estimate_co2(
        self,
        m_dot_fuel: float,
        lca_factor: float = 1.0,
        carbon_fraction: Optional[float] = None
    ) -> float:
        """
        Calculate CO₂ emissions from combustion stoichiometry.

        Phase 2.3 (V7): the emission index is computed from the blend's carbon
        mass fraction instead of the former flat 3.16 kg/kg constant:

            EI_CO2 = (M_CO2 / M_C) × w_C = 3.664 × w_C   [kg CO₂ / kg fuel]

        For the n-dodecane Jet-A1 surrogate (w_C = 0.846) this gives
        3.10 kg/kg — about 1.9% below the old 3.16 constant.

        Args:
            m_dot_fuel: Fuel mass flow rate [kg/s]
            lca_factor: LEGACY multiplier retained for backward compatibility
                        (1.0 = combustion only). New code should report
                        combustion CO₂ (this method, lca_factor=1) and
                        lifecycle CO₂e (estimate_lifecycle_co2e) as separate
                        quantities — never mixed on one axis.
            carbon_fraction: Carbon mass fraction w_C of the fuel. If None,
                             DEFAULT_CARBON_FRACTION (n-dodecane) is used.

        Returns:
            CO₂ emission rate [g/s]
        """
        if m_dot_fuel <= 0:
            return 0.0

        w_c = carbon_fraction if carbon_fraction is not None else self.DEFAULT_CARBON_FRACTION
        ei_co2 = (44.01 / 12.011) * w_c  # kg CO₂ / kg fuel (stoichiometric)

        return ei_co2 * lca_factor * m_dot_fuel * 1000.0  # g/s

    def estimate_lifecycle_co2e(
        self,
        m_dot_fuel: float,
        lcef_gCO2e_per_MJ: float,
        lhv_MJ_per_kg: float = 43.2
    ) -> float:
        """
        Calculate lifecycle CO₂-equivalent emissions on the CORSIA basis.

        CORSIA life-cycle emissions values (L_CEF, ICAO Document 06) are
        energy-specific [gCO₂e/MJ], so:

            lifecycle CO₂e [g/s] = L_CEF × LHV × ṁ_fuel

        This is a DIFFERENT quantity from combustion CO₂ (estimate_co2):
        it includes upstream/feedstock/ILUC terms and uses the fossil
        baseline 89 gCO₂e/MJ for conventional jet fuel. Report the two
        separately; never mix them on one axis.

        Args:
            m_dot_fuel: Fuel mass flow rate [kg/s]
            lcef_gCO2e_per_MJ: Blend life-cycle emissions value [gCO₂e/MJ]
                               (energy-weighted over pathway components)
            lhv_MJ_per_kg: Blend lower heating value [MJ/kg]

        Returns:
            Lifecycle CO₂e emission rate [g/s]
        """
        if m_dot_fuel <= 0:
            return 0.0
        return lcef_gCO2e_per_MJ * lhv_MJ_per_kg * m_dot_fuel


# ============================================================================
# INTEGRATED TURBOFAN ENGINE CLASS
# ============================================================================

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


def scale_turbine_exit_temp(
    T_in: float,
    p_in: float,
    p_out: float,
    gamma: float,
    eta_polytropic: float = 0.9,
) -> float:
    """
    Calculate turbine exit temperature using the isentropic expansion relation
    with a polytropic efficiency correction.

    Formula:
        T_out = T_in × [1 − η_t × (1 − (P_out / P_in)^((γ−1)/γ))]

    Where:
        η_t: Polytropic turbine efficiency (default 0.9)
        γ: Heat capacity ratio (fuel-dependent, from combustion products)
        P_out/P_in: Turbine pressure ratio

    This replaces the old hardcoded expansion_ratio = 1005/1700 ≈ 0.59 approach,
    which was fuel-blind. The new formula uses the fuel-specific γ from Cantera,
    so different fuels now produce different turbine work extraction.

    Args:
        T_in: Turbine inlet temperature [K]
        p_in: Turbine inlet pressure [Pa]
        p_out: Turbine exit pressure [Pa]
        gamma: Heat capacity ratio (cp/cv) from combustion products
        eta_polytropic: Polytropic turbine efficiency (default 0.9)

    Returns:
        T_out: Predicted turbine exit temperature [K]
    """
    pressure_ratio = p_out / p_in
    exponent = (gamma - 1.0) / gamma
    T_out = T_in * (1.0 - eta_polytropic * (1.0 - pressure_ratio ** exponent))

    return T_out


def part_power_state(
    power_fraction: float,
    pi_rated: float,
    m_dot_rated: float,
    k_pi: float,
    k_mdot: float,
) -> Tuple[float, float]:
    """
    Low-fidelity throttle model mapping an ICAO LTO power setting to
    compressor pressure ratio and core mass flow.

        pi_c(x)  = 1 + (pi_rated - 1) * x^k_pi
        m_dot(x) = m_dot_rated * x^k_mdot

    where x = F/F00 (ICAO 'Power (%)' / 100, using thrust fraction as the
    corrected-speed proxy). At x = 1 both quantities equal their rated values;
    as x -> 0, pi_c -> 1 (no compression) and m_dot -> 0. The two exponents
    are calibrated once against LTO fuel-flow data (see
    scripts/optimization/calibrate_lto.py) and replace the four hand-set
    per-mode scales that Phase 1 found to be partly inert.

    Args:
        power_fraction: LTO power setting as a fraction of rated thrust (0, 1]
        pi_rated: Rated overall pressure ratio
        m_dot_rated: Rated core mass flow [kg/s]
        k_pi: Throttle exponent for the pressure ratio
        k_mdot: Throttle exponent for the core mass flow

    Returns:
        Tuple (pi_c, m_dot_core) at the requested power setting
    """
    if not 0.0 < power_fraction <= 1.0:
        raise ValueError(f"power_fraction must be in (0, 1], got {power_fraction}")
    pi_c = 1.0 + (pi_rated - 1.0) * power_fraction ** k_pi
    m_dot = m_dot_rated * power_fraction ** k_mdot
    return pi_c, m_dot


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
        turbine_pinn_path: str = "models/turbine_pinn.pt",
        nozzle_pinn_path: str = "models/nozzle_pinn.pt",
        nozzle_variant: str = "pinn",
        le_pinn_path: str = "models/le_pinn_unified.pt",
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
            nozzle_variant: Nozzle model selection ("pinn" or "le_pinn")
            le_pinn_path: Path to trained LE-PINN nozzle checkpoint (.pt file)
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
            'bypass_ratio': 9.1,              # Bypass ratio (fan flow / core flow);
                                              # set to 0 to disable the bypass stream
            'fpr': 1.45,                      # Fan pressure ratio (live value; part-power
                                              # scripts may scale it per mode)
            'pi_c': 43.2,                     # Overall pressure ratio (live value read
                                              # by run_compressor at call time)
            'combustor_pressure_loss': 0.0,   # Fractional total-pressure loss between
                                              # compressor exit and combustor
                                              # (p_comb = p3 * (1 - loss)); 0 = legacy
            'combustor_heat_loss_fraction': 0.0,  # Cycle heat loss xi (Phase 2.6 hook).
                                                  # Production 0.0 by argument, not by a
                                                  # sourced value: liner heat is recovered
                                                  # by the annulus air upstream of the
                                                  # turbine, only casing loss is a cycle
                                                  # term and no engine-class source exists.
                                                  # scripts/validation/heat_loss_provenance.md
            'combustor_air_fraction': 1.0,    # beta (Phase 3.4): fraction of core air
                                              # burned at phi; (1-beta) bypasses the
                                              # burner as cooling/dilution air and
                                              # remixes before the turbine. beta ~
                                              # 0.7-0.8 for conventional/RQL-era
                                              # combustors (Lefebvre & Ballal, Gas
                                              # Turbine Combustion: ~20-30% of
                                              # combustor air is liner cooling +
                                              # dilution). 1.0 = legacy behavior.
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
        self.nozzle_variant = nozzle_variant
        self.le_pinn_path = le_pinn_path

        # Turbine design-point parameters for isentropic expansion calculation
        self.turbine_design = {
            'T_in_ref': 1700.0,        # Reference inlet temperature [K]
            'T_out_ref': 1005.0,       # Reference outlet temperature [K]
            'P_out': 1.93e5,           # Turbine exit pressure [Pa]
            'eta_polytropic': 0.9,     # Polytropic turbine efficiency
        }

        # Initialize emissions estimator for environmental optimization
        self.emissions = EmissionsEstimator(icao_data_path=icao_data_path)

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
        # design_point['pi_c'] is the single live OPR source: part-power scripts
        # (calibration/holdout) write it per mode and it must take effect here.
        # (Phase 1 found the old write path was never read — all LTO modes ran
        # at rated OPR.)
        self.compressor.pi_c = self.design_point['pi_c']
        # DEFECT FIX (Phase 2, 2026-07-14): the shared Cantera Solution
        # initializes to the mechanism's first species (pure argon for CRECK),
        # and no prior code ever set an air composition — the compressor was
        # compressing monatomic argon (gamma 1.67), inflating T3 by ~500 K at
        # rated OPR and propagating into every downstream temperature. Fuel
        # flow was unaffected (FAR uses its own gas), so calibrations remain
        # valid; temperatures/thrust/TSFC before this fix were not.
        self.gas.TPX = T_in, p_in, "O2:0.21, N2:0.79"
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

        # Run Cantera combustion model (heat_loss_fraction: Phase 2.6 hook,
        # 0.0 by default via design_point)
        result = combustor.run(
            T_in=T_in,
            p_in=p_in,
            fuel_blend=fuel_blend,
            phi=phi,
            efficiency=efficiency,
            heat_loss_fraction=self.design_point.get('combustor_heat_loss_fraction', 0.0)
        )

        print(f"[Combustor - {mech_label}]")
        print(f"  Fuel:   {fuel_blend.name}")
        print(f"  Phi:    {phi:.3f}")
        print(f"  FAR:    {f:.6f} (fuel/air mass ratio)")
        print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar")
        print(f"  Outlet: T={result['T_out']:.1f} K, P={result['p_out']/1e5:.2f} bar")
        print(f"  Efficiency: {efficiency*100:.1f}%\n")

        return result, f

    def run_turbine_analytic(
        self,
        flow_state_in: Dict[str, float],
        m_dot: float,
        target_work_total: float
    ) -> Dict[str, float]:
        """
        Analytic turbine counterpart for PINN ablations (Phase 2.5).

        Work-matched polytropic expansion using the same fuel-dependent
        thermodynamic properties as the PINN path:
            T5 = T4 - W / (m_dot cp)
            p5 = p4 (T5/T4)^(gamma / (eta_poly (gamma - 1)))
            u5 = m_dot / (rho5 A_out)   (exact continuity, same as the PINN)

        Returns the same dict shape as run_turbine (rho, u, p, T,
        work_specific, work_total, cp, R, gamma).
        """
        cp = flow_state_in['cp']
        R = flow_state_in['R']
        gamma = flow_state_in.get('gamma', cp / (cp - R))
        T_in = flow_state_in['T']
        p_in = flow_state_in['p']
        eta_t = self.turbine_design['eta_polytropic']

        T_out = T_in - target_work_total / (m_dot * cp)
        if T_out <= 0:
            raise ValueError(
                f"Analytic turbine: target work {target_work_total/1e6:.1f} MW "
                f"exceeds available enthalpy flux"
            )
        # Polytropic expansion: T5/T4 = (p5/p4)^(eta (gamma-1)/gamma)
        p_out = p_in * (T_out / T_in) ** (gamma / (eta_t * (gamma - 1.0)))
        A_outlet = self.design_point['A_combustor_exit'] * 1.82
        rho_out = p_out / (R * T_out)
        u_out = m_dot / (rho_out * A_outlet)

        result = {
            'rho': rho_out,
            'u': u_out,
            'p': p_out,
            'T': T_out,
            'work_specific': target_work_total / m_dot,
            'work_total': target_work_total,
            'cp': cp,
            'R': R,
            'gamma': gamma,
        }
        print(f"[Turbine - ANALYTIC (polytropic, eta={eta_t})]")
        print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar")
        print(f"  Outlet: T={T_out:.1f} K, P={p_out/1e5:.2f} bar")
        print(f"  Work Extracted: {target_work_total/1e6:.2f} MW\n")
        return result

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
            # Estimate target work from fuel-dependent isentropic expansion
            cp = thermo_props['cp']
            gamma = thermo_props['gamma']
            T_in = inlet_state['T']
            p_in = inlet_state['p']
            p_out = self.turbine_design['P_out']
            eta_t = self.turbine_design['eta_polytropic']

            T_out_est = scale_turbine_exit_temp(T_in, p_in, p_out, gamma, eta_t)
            delta_T = T_in - T_out_est
            target_work_total = m_dot * cp * delta_T

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

        Thrust model (static test stand, same as the bypass stream and the PINN
        paths' thrust_model='static_test_stand'): engine-level momentum balance
        F = ṁ·u_exit − ṁ_0·u_0 + (p_exit − p_amb)·A_exit with freestream u_0 = 0
        (NASA general thrust equation). The inlet velocity flow_state_in['u'] is an
        internal station (turbine exit) and is not subtracted; it does not enter
        this model and remains available on the turbine state for diagnostics.

        Modeling boundary: an ideal nozzle that always expands fully to p_amb
        (p_exit = p_amb, so the pressure term is zero whenever p_in ≥ p_amb). T and
        p of flow_state_in are used as the nozzle total state. The exit area is not
        a constraint here: it follows from continuity, A_exit_effective =
        ṁ/(ρ_exit·u_exit), and is returned as diagnostic metadata only — it is not
        a measured engine dimension. The configured design_point['A_nozzle_exit']
        (the PINN geometry) does not constrain this flow and is used only in the
        signed pressure term.

        Args:
            flow_state_in: Inlet flow state dict with rho, u, p, T, cp, R, gamma
            m_dot: Total mass flow rate [kg/s]

        Returns:
            Nozzle exit state dict with rho, u, p, T, thrust_total, thrust_momentum,
            thrust_pressure, thrust_model, A_exit_effective
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

        # Static-stand thrust: F = ṁ u_exit + (P_exit - P_amb) A_exit (freestream u_0 = 0;
        # the internal nozzle-inlet momentum is not subtracted)
        F_momentum = m_dot * u_exit_isentropic
        p_exit = p_in * pressure_ratio
        delta_p = p_exit - P_amb
        pressure_tol = 1.0  # Pa tolerance to avoid numerical noise
        if abs(delta_p) < pressure_tol:
            F_pressure = 0.0
        else:
            F_pressure = delta_p * A_exit  # Signed: negative for over-expanded jets
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
            'thrust_pressure': F_pressure,
            'thrust_model': 'static_test_stand',
            # continuity-implied exit area of the ideal fully expanded jet (diagnostic only)
            'A_exit_effective': m_dot / (rho_exit * u_exit_isentropic)
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

        if self.nozzle_variant == "le_pinn":
            from simulation.nozzle.le_pinn import run_le_pinn

            le_path = self.le_pinn_path
            if not Path(le_path).exists():
                le_path = str(Path(le_path).parent / "le_pinn_engine_unified.pt")

            try:
                le_result = run_le_pinn(
                    model_path=le_path,
                    inlet_state=turb_result,
                    ambient_p=self.design_point['P_ambient'],
                    A_in=self.design_point['A_nozzle_inlet'],
                    A_exit=self.design_point['A_nozzle_exit'],
                    length=1.0,
                    thermo_props={
                        'cp': turb_result['cp'],
                        'R': turb_result['R'],
                        'gamma': turb_result['gamma'],
                    },
                    m_dot=m_dot_total,
                    n_axial=50,
                    n_radial=20,
                    device="cpu",
                    return_profile=False,
                    thrust_model="static_test_stand",
                )
                le_values = [
                    le_result['exit_state']['rho'],
                    le_result['exit_state']['u'],
                    le_result['exit_state']['p'],
                    le_result['exit_state']['T'],
                    le_result['thrust_total'],
                    le_result['thrust_momentum'],
                    le_result['thrust_pressure'],
                ]
                if not all(np.isfinite(value) for value in le_values):
                    raise ValueError("LE-PINN returned non-finite nozzle outputs")
                return {
                    'rho': le_result['exit_state']['rho'],
                    'u': le_result['exit_state']['u'],
                    'p': le_result['exit_state']['p'],
                    'T': le_result['exit_state']['T'],
                    'thrust_total': le_result['thrust_total'],
                    'thrust_momentum': le_result['thrust_momentum'],
                    'thrust_pressure': le_result['thrust_pressure'],
                }
            except Exception as exc:
                print(f"⚠️  LE-PINN nozzle failed ({exc}). Falling back to regular PINN.")

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
        combustor_efficiency: Optional[float] = None,
        lca_factor: float = 1.0,
        lcef_gCO2e_per_MJ: Optional[float] = None,
        turbine_model: str = "analytic",
        nozzle_model: str = "analytic"
    ) -> Dict[str, Any]:
        """
        Execute complete engine cycle and calculate performance metrics.

        Args:
            fuel_blend: LocalFuelBlend object
            phi: Equivalence ratio (default 0.5 for lean combustion)
            combustor_efficiency: Combustion efficiency. If None (default), computed
                                  dynamically from phi and fuel_blend via
                                  ``Combustor.estimate_efficiency()``.
            lca_factor: LEGACY lifecycle multiplier (kept for backward
                       compatibility; scales the combustion CO₂ into the
                       'Net_CO2_g_s' field). New analyses should pass
                       lcef_gCO2e_per_MJ instead and read the separate
                       combustion / lifecycle fields.
            lcef_gCO2e_per_MJ: Blend CORSIA life-cycle emissions value
                       [gCO₂e/MJ] (see data/corsia_lca_values.yaml). When
                       given, the result includes 'Lifecycle_CO2e_g_s'.
            turbine_model: "analytic" (default) or "pinn". Defaults were
                       adjudicated in Phase 3.1: the analytic path is the
                       hand-verified work-consistent polytropic expansion;
                       the PINN's exit pressure is a raw NN output deviating
                       −41.5% from work consistency
                       (outputs/turbine_p5_adjudication.csv).
            nozzle_model: "analytic" (default) or "pinn". The LE-PINN nozzle
                       failed external (Sajben) validation, so analytic is
                       the production configuration; the flags remain for
                       ablation studies.

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

        # 1a. FAN / BYPASS STREAM (Phase 2.2)
        # 0-D fan on the bypass stream only; core fan-root compression is part
        # of pi_c (OPR) by definition, so it is not modeled separately.
        # eta_fan = 0.90: standard modern civil-fan isentropic efficiency
        # (see simulation/fan.py docstring for sourcing).
        bpr = self.design_point.get('bypass_ratio', 0.0)
        if bpr > 0:
            fan = Fan(fpr=self.design_point.get('fpr', 1.45), eta_fan=0.90)
            m_dot_bypass = bpr * m_dot_core
            fan_result = fan.run(T_ambient, P_ambient, m_dot_bypass)
            fan_work_total = fan_result['work_total']
            print(f"[Fan]")
            print(f"  FPR:    {fan.fpr:.3f} (BPR {bpr:.1f}, eta {fan.eta_fan:.2f})")
            print(f"  Bypass: {m_dot_bypass:.1f} kg/s, dT={fan_result['dT']:.1f} K")
            print(f"  Work:   {fan_work_total/1e6:.2f} MW")
            print(f"  Bypass thrust: {fan_result['thrust_bypass']/1e3:.2f} kN "
                  f"(u_exit={fan_result['u_bypass_exit']:.1f} m/s)\n")
        else:
            fan_result = None
            m_dot_bypass = 0.0
            fan_work_total = 0.0

        # 1b. COMPRESSOR
        comp_result = self.run_compressor(T_ambient, P_ambient)

        # 2. COMBUSTOR
        # If no explicit efficiency provided, estimate dynamically
        if combustor_efficiency is None:
            combustor_efficiency = Combustor.estimate_efficiency(phi, fuel_blend)

        # Combustor total-pressure loss between compressor exit and combustor
        # (0.0 by default; calibration sets design_point['combustor_pressure_loss'])
        p_loss = self.design_point.get('combustor_pressure_loss', 0.0)
        if not 0.0 <= p_loss < 1.0:
            raise ValueError(f"combustor_pressure_loss must be in [0, 1), got {p_loss}")
        p_comb_in = comp_result['p_out'] * (1.0 - p_loss)

        # Combustor airflow split (Phase 3.4): only beta * m_dot_core is
        # burned at phi; the rest bypasses as liner-cooling/dilution air.
        beta = self.design_point.get('combustor_air_fraction', 1.0)
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"combustor_air_fraction must be in (0, 1], got {beta}")
        m_dot_burn = beta * m_dot_core

        comb_result, f = self.run_combustor(
            T_in=comp_result['T_out'],
            p_in=p_comb_in,
            fuel_blend=fuel_blend,
            phi=phi,
            efficiency=combustor_efficiency
        )

        # Calculate actual mass flows including fuel (phi applies to burner air)
        m_dot_fuel = f * m_dot_burn  # kg/s
        m_dot_total = m_dot_core + m_dot_fuel  # kg/s (core stream incl. fuel)
        comp_work_total = comp_result['work_specific'] * m_dot_core

        if beta < 1.0:
            # Remix the unburned (1-beta) core air with the combustion
            # products before the turbine: enthalpy balance with constant-cp
            # mixing sets the diluted turbine inlet temperature. Mixture
            # transport properties are mass-weighted (0-D fidelity).
            m_prod = m_dot_burn + m_dot_fuel
            m_byp_air = (1.0 - beta) * m_dot_core
            cp_p = comb_result['cp_out']
            R_p = comb_result['R_out']
            self.gas.TPX = comp_result['T_out'], p_comb_in, "O2:0.21, N2:0.79"
            cp_air = self.gas.cp_mass
            R_air = ct.gas_constant / self.gas.mean_molecular_weight
            T4_mix = ((m_prod * cp_p * comb_result['T_out'] +
                       m_byp_air * cp_air * comp_result['T_out']) /
                      (m_prod * cp_p + m_byp_air * cp_air))
            cp_mix = (m_prod * cp_p + m_byp_air * cp_air) / m_dot_total
            R_mix = (m_prod * R_p + m_byp_air * R_air) / m_dot_total
            print(f"[Combustor dilution mix] beta={beta:.2f}: "
                  f"T_burner={comb_result['T_out']:.1f} K -> "
                  f"T4_mixed={T4_mix:.1f} K "
                  f"({m_byp_air:.1f} kg/s dilution air at {comp_result['T_out']:.1f} K)\n")
            comb_result = dict(
                comb_result,
                T_out=T4_mix,
                cp_out=cp_mix,
                R_out=R_mix,
                gamma_out=cp_mix / (cp_mix - R_mix),
            )

        # Convert Cantera output to flow state for PINN input
        turb_inlet_state = self._cantera_to_flow_state(
            cantera_out=comb_result,
            m_dot=m_dot_total,
            A_ref=self.design_point['A_combustor_exit']
        )

        # 3. TURBINE — must supply compressor AND fan shaft work (Phase 2.2)
        # turbine_model/nozzle_model: PINN-vs-analytic ablation flags (Phase 2.5);
        # components are swapped at inference only, never retrained.
        if turbine_model not in ("pinn", "analytic"):
            raise ValueError(f"turbine_model must be 'pinn' or 'analytic', got {turbine_model!r}")
        if nozzle_model not in ("pinn", "analytic"):
            raise ValueError(f"nozzle_model must be 'pinn' or 'analytic', got {nozzle_model!r}")

        if turbine_model == "analytic":
            turb_result = self.run_turbine_analytic(
                turb_inlet_state,
                m_dot_total,
                target_work_total=comp_work_total + fan_work_total
            )
        else:
            turb_result = self.run_turbine(
                turb_inlet_state,
                m_dot_total,
                target_work_total=comp_work_total + fan_work_total
            )

        # 4. NOZZLE
        if nozzle_model == "analytic":
            nozz_result = self.run_nozzle(turb_result, m_dot_total)
        else:
            nozz_result = self._run_nozzle_stage(turb_result, m_dot_total)

        # PERFORMANCE METRICS
        print("="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)

        # Two-stream thrust (Phase 2.2): core nozzle + bypass stream
        thrust_core = nozz_result['thrust_total']
        thrust_bypass = fan_result['thrust_bypass'] if fan_result else 0.0
        thrust = thrust_core + thrust_bypass
        m_dot_air_total = m_dot_core + m_dot_bypass

        # TSFC: ṁ_fuel / F_thrust [kg/(N·s)], reported in mg/(N·s)
        if thrust <= 0:
            tsfc_SI = np.inf
            tsfc_mg = np.inf
            tsfc_valid = False
        else:
            tsfc_SI = m_dot_fuel / thrust
            tsfc_mg = tsfc_SI * 1.0e6
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
            ke_flux = 0.5 * m_dot_total * u_exit**2  # W - core jet KE flux
            if fan_result:
                ke_flux += 0.5 * m_dot_bypass * fan_result['u_bypass_exit']**2
            eta_thermal = ke_flux / fuel_power  # Kinetic efficiency
            eta_valid = True

        print(f"  Fuel Blend:          {fuel_blend.name}")
        print(f"  Equivalence Ratio:   {phi:.3f}")
        print(f"  Fuel-Air Ratio:      {f:.6f}")
        print(f"  Core Mass Flow:      {m_dot_core:.2f} kg/s")
        print(f"  Bypass Mass Flow:    {m_dot_bypass:.2f} kg/s (BPR {bpr:.1f})")
        print(f"  Fuel Mass Flow:      {m_dot_fuel:.4f} kg/s")
        print(f"  Total Mass Flow:     {m_dot_total:.2f} kg/s (core stream)")
        print(f"  ---")
        print(f"  Thrust:              {thrust/1e3:.2f} kN "
              f"(core {thrust_core/1e3:.2f} + bypass {thrust_bypass/1e3:.2f})")
        if m_dot_air_total > 0:
            print(f"  Specific Thrust:     {thrust/m_dot_air_total:.1f} N·s/kg (total air)")
        print(f"  Fan Work:            {fan_work_total/1e6:.2f} MW")

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

        # Emissions calculations
        OPR = comp_result['p_out'] / self.design_point['P_ambient']
        nox_g_s = self.emissions.estimate_nox(OPR=OPR, m_dot_fuel=m_dot_fuel)
        co_g_s = self.emissions.estimate_co(
            combustor_efficiency=combustor_efficiency,
            m_dot_fuel=m_dot_fuel
        )

        # Combustion CO₂ from blend stoichiometry (Phase 2.3): w_C computed
        # from the composition; falls back to the n-dodecane default if a
        # species is unknown to the surrogate atom table.
        try:
            from simulation.fuels import carbon_fraction_of_composition
            w_c = carbon_fraction_of_composition(fuel_blend.composition)
        except (KeyError, AttributeError, ValueError):
            w_c = self.emissions.DEFAULT_CARBON_FRACTION
        co2_combustion_g_s = self.emissions.estimate_co2(
            m_dot_fuel=m_dot_fuel,
            carbon_fraction=w_c
        )
        # Legacy mixed quantity (combustion × LCA multiplier), retained so
        # existing consumers keep working; new analyses use the two separate
        # axes (CO2_combustion_g_s + Lifecycle_CO2e_g_s).
        net_co2_g_s = co2_combustion_g_s * lca_factor
        lifecycle_co2e_g_s = (
            self.emissions.estimate_lifecycle_co2e(
                m_dot_fuel=m_dot_fuel,
                lcef_gCO2e_per_MJ=lcef_gCO2e_per_MJ,
            ) if lcef_gCO2e_per_MJ is not None else None
        )

        print(f"\n  Emissions Summary:")
        print(f"    NOx:     {nox_g_s:.3f} g/s  ({nox_g_s/m_dot_fuel:.2f} g/kg fuel)")
        print(f"    CO:      {co_g_s:.3f} g/s  ({co_g_s/m_dot_fuel:.2f} g/kg fuel)")
        print(f"    CO₂ (combustion): {co2_combustion_g_s:.2f} g/s "
              f"(EI {co2_combustion_g_s/(m_dot_fuel*1000):.3f} kg/kg, w_C={w_c:.4f})")
        if lifecycle_co2e_g_s is not None:
            print(f"    CO₂e (lifecycle): {lifecycle_co2e_g_s:.2f} g/s "
                  f"(L_CEF {lcef_gCO2e_per_MJ:.1f} gCO₂e/MJ)")
        else:
            print(f"    CO₂ (legacy net): {net_co2_g_s:.2f} g/s  (LCA factor: {lca_factor:.2f})")

        # ========================================================================
        # INTERNAL CONSISTENCY CHECKS
        #
        # These are SELF-CONSISTENCY checks on the solver, NOT validation
        # against data. Each holds by construction:
        #   - continuity: velocity is set as u = mdot / (rho * A), so the mass
        #     residual is a floating-point artefact and can only ever read ~0%;
        #   - the inlet state is imposed as a hard boundary condition;
        #   - turbine work is solved to equal compressor + fan work.
        # A near-zero number here says the code solved the equations it was
        # given. It says nothing about whether those equations describe a real
        # engine. (The withdrawn "mass continuity error = 0.00%" Highlight came
        # from this printout.) Validation against data lives in
        # scripts/validation/holdout_icao_validation.py (fuel flow) and
        # scripts/validation/nox_holdout_validation.py (NOx correlation).
        # ========================================================================
        print(f"\n  Internal consistency (by construction -- NOT validation):")
        print(f"    - Turbine-Nozzle handoff exact (states passed directly)")
        print(f"    - Continuity residual: {nozz_result.get('mass_conservation', {}).get('error_pct', 0):.2f}%  (u = mdot/rho*A: zero by construction)")
        print(f"    - Inlet BC residual: {nozz_result.get('inlet_verification', {}).get('max_error', 0)*100:.3f}%  (hard boundary condition)")
        print(f"    - Thrust model: {nozz_result.get('thrust_model', 'static').upper()}")
        print(f"    - Energy balance: turbine work = compressor + fan work (solved, not checked)")

        print("="*70 + "\n")

        return {
            'compressor': comp_result,
            'combustor': comb_result,
            'turbine': turb_result,
            'nozzle': nozz_result,
            'fan': fan_result,
            'performance': {
                'thrust_N': thrust,
                'thrust_kN': thrust / 1e3,
                'thrust_core_kN': thrust_core / 1e3,
                'thrust_bypass_kN': thrust_bypass / 1e3,
                'tsfc_SI': tsfc_SI if tsfc_valid else np.inf,  # kg/(N·s)
                'tsfc_mg_per_Ns': tsfc_mg if tsfc_valid else np.inf,  # mg/(N·s)
                'thermal_efficiency': eta_thermal,
                'fuel_mass_flow': m_dot_fuel,
                'total_mass_flow': m_dot_total,       # core stream (air + fuel)
                'bypass_mass_flow': m_dot_bypass,
                'total_air_mass_flow': m_dot_air_total,  # core + bypass air
                'specific_thrust_Ns_kg': (thrust / m_dot_air_total
                                          if m_dot_air_total > 0 else np.inf),
                'fan_work_W': fan_work_total,
                'fuel_air_ratio': f
            },
            'emissions': {
                'NOx_g_s': nox_g_s,
                'CO_g_s': co_g_s,
                'CO2_combustion_g_s': co2_combustion_g_s,
                'carbon_fraction': w_c,
                'Lifecycle_CO2e_g_s': lifecycle_co2e_g_s,
                'lcef_gCO2e_per_MJ': lcef_gCO2e_per_MJ,
                'Net_CO2_g_s': net_co2_g_s,      # legacy: combustion x lca_factor
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
        print("MODE: HyChem Validation (Jet-A1 ICAO Benchmark)")
        print("="*70 + "\n")

        engine = IntegratedTurbofanEngine(
            mechanism_profile="validation",
            creck_mechanism_path="data/creck_c1c16_full.yaml",
            hychem_mechanism_path="data/A1highT.yaml",
            turbine_pinn_path="turbine_pinn.pt",
            nozzle_pinn_path="nozzle_pinn.pt"
        )

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

    elif mode == "blends":
        print("MODE: Blend Comparison (CRECK Mechanism)")
        print("="*70 + "\n")

        engine = IntegratedTurbofanEngine(
            mechanism_profile="blends",
            creck_mechanism_path="data/creck_c1c16_full.yaml",
            hychem_mechanism_path="data/A1highT.yaml",
            turbine_pinn_path="turbine_pinn.pt",
            nozzle_pinn_path="nozzle_pinn.pt"
        )

        fuels_to_test = [
            FUEL_LIBRARY["Jet-A1"],
            FUEL_LIBRARY["Bio-SPK"],
            FUEL_LIBRARY["HEFA-50"]
        ]

        results = {}
        for fuel in fuels_to_test:
            result = engine.run_full_cycle(
                fuel_blend=fuel,
                phi=0.5,
                combustor_efficiency=0.98
            )
            results[fuel.name] = result

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
