"""
Integrated Turbofan Engine Simulation with Hybrid Grey-Box Modeling.

This module implements a complete jet engine cycle using:
- Cantera: Chemical kinetics and thermodynamic equilibrium (compressor, combustor)
- PINNs: Physics-informed neural networks for flow physics (turbine, nozzle)

The simulation supports multiple fuel blends including Sustainable Aviation Fuels (SAF)
and provides fuel-dependent performance predictions.

Engine Cycle: Compressor → Combustor → Turbine → Nozzle

Author: Arnav Patil
Note: See LIMITATIONS_SUMMARY.md and MECHANISM_STRATEGY.md for detailed technical discussion.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cantera as ct
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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
# PHYSICS-INFORMED NEURAL NETWORK ARCHITECTURE
# ============================================================================

class NormalizedPINN(nn.Module):
    """
    Physics-Informed Neural Network for modeling turbine and nozzle flow physics.

    Architecture: 3-layer feedforward network with hyperbolic tangent activation
    Input: Normalized axial position x* ∈ [0,1] along component length
    Output: Normalized flow state [ρ*, u*, p*, T*] (density, velocity, pressure, temperature)

    The PINN learns to satisfy conservation laws (mass, momentum, energy) through
    physics-based loss functions during training, enabling it to predict realistic
    expansion behavior without requiring CFD simulation at runtime.
    """

    def __init__(self):
        super().__init__()
        # Three hidden layers with 64 neurons each
        # Tanh activation enforces smooth, differentiable solutions required for physics constraints
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)  # Output: [rho, u, p, T]
        )

    def forward(self, x):
        """Forward pass through network to predict flow state at position x."""
        return self.net(x)


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
        delta_thrust_pct = ((thrust_kN - baseline_thrust) / baseline_thrust) * 100
        delta_tsfc_pct = ((tsfc - baseline_tsfc) / baseline_tsfc) * 100
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

        # Load pre-trained PINN models for turbine and nozzle
        # PINNs predict flow evolution (ρ, u, p, T) as function of axial position
        self.turbine_pinn, self.turbine_scales, self.turbine_conditions = \
            self._load_pinn(turbine_pinn_path, "Turbine")

        self.nozzle_pinn, self.nozzle_scales, self.nozzle_conditions = \
            self._load_pinn(nozzle_pinn_path, "Nozzle")

        # Store turbine training reference conditions for temperature scaling
        # The PINN was trained with specific inlet/outlet temperatures
        self.turbine_design = {
            'T_in': 1700.0,   # Training inlet temperature [K]
            'T_out': 1005.0,  # Training outlet temperature [K]
            'expansion_ratio': 1005.0 / 1700.0  # Learned temperature ratio ≈ 0.59
        }

        print("✓ IntegratedTurbofanEngine initialized successfully\n")

    def _load_pinn(
        self,
        checkpoint_path: str,
        component_name: str
    ) -> Tuple[nn.Module, Dict, Dict]:
        """
        Load a trained PINN model with its normalization scales and conditions.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            component_name: Name for error messages (e.g., "Turbine")

        Returns:
            Tuple of (model, scales_dict, conditions_dict)
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"{component_name} PINN not found at '{checkpoint_path}'.\n"
                f"Please train the {component_name.lower()} PINN first by running:\n"
                f"  python simulation/{component_name.lower()}/{component_name.lower()}.py"
            )

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract components (handle different checkpoint formats)
            model_state = checkpoint.get('model_state_dict', checkpoint)
            scales = checkpoint.get('scales', {})
            conditions = checkpoint.get('conditions', {})

            # Create model and load weights
            model = NormalizedPINN()
            model.load_state_dict(model_state)
            model.eval()  # Set to evaluation mode

            print(f"✓ Loaded {component_name} PINN from {checkpoint_path}")

            return model, scales, conditions

        except Exception as e:
            raise RuntimeError(
                f"Failed to load {component_name} PINN from '{checkpoint_path}': {e}"
            )

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
        m_dot: float
    ) -> Dict[str, float]:
        """
        Simulate turbine expansion using PINN-predicted flow field and fuel-dependent thermodynamics.

        The turbine model combines:
        1. PINN predictions for pressure drop (learned from training data)
        2. Temperature scaling based on learned expansion ratio
        3. Fuel-dependent work extraction using actual cp from combustion products

        This hybrid approach allows the model to capture realistic expansion behavior
        while accounting for how different fuel properties (cp, R, gamma) affect turbine work.

        Args:
            flow_state_in: Inlet flow state dict with rho, u, p, T, cp, R, gamma
            m_dot: Total mass flow rate through turbine [kg/s]

        Returns:
            Turbine exit state dict with rho, u, p, T, work_specific, work_total
        """
        T_in = flow_state_in['T']
        cp = flow_state_in['cp']
        gamma = flow_state_in.get('gamma', cp / (cp - flow_state_in['R']))

        # Calculate outlet temperature using PINN-learned expansion ratio
        if self.USE_TURBINE_DT_SCALING:
            T_out_predicted = scale_turbine_exit_temp(
                T_in,
                self.turbine_design['expansion_ratio'],
                cp=cp,
                gamma=gamma
            )
        else:
            T_out_predicted = T_in * self.turbine_design['expansion_ratio']

        # Query PINN for pressure and velocity at turbine exit (x=1.0)
        with torch.no_grad():
            x_exit = torch.tensor([[1.0]], dtype=torch.float32)
            out_norm = self.turbine_pinn(x_exit)

            # Denormalize PINN outputs to physical units
            p_out_pinn = out_norm[0, 2].item() * self.turbine_scales['p']
            u_out_pinn = out_norm[0, 1].item() * self.turbine_scales['u']
            rho_out_pinn = out_norm[0, 0].item() * self.turbine_scales['rho']

        # Scale PINN pressure to match actual inlet conditions
        p_scale = flow_state_in['p'] / self.turbine_conditions['inlet']['p']
        p_out = p_out_pinn * p_scale

        # Recalculate density using ideal gas law with fuel-specific gas constant
        R = flow_state_in['R']
        rho_out = p_out / (R * T_out_predicted)

        # Calculate exit velocity from mass continuity: ṁ = ρ u A
        A_exit = self.design_point['A_combustor_exit'] * 1.82  # Turbine area expansion ratio
        u_out = m_dot / (rho_out * A_exit)

        # Calculate turbine work extraction using fuel-dependent specific heat
        # Work equation: W = ṁ cp ΔT (fuel-dependent cp is critical here!)
        cp = flow_state_in['cp']
        R = flow_state_in['R']
        gamma = flow_state_in.get('gamma', cp / (cp - R))

        work_specific = cp * (T_in - T_out_predicted)  # Specific work [J/kg]
        work_total = m_dot * work_specific  # Total power [W]

        print(f"[Turbine]")
        print(f"  Inlet:  T={T_in:.1f} K, P={flow_state_in['p']/1e5:.2f} bar")
        print(f"  Outlet: T={T_out_predicted:.1f} K, P={p_out/1e5:.2f} bar")
        print(f"  Fuel-dependent properties: cp={cp:.1f} J/(kg·K), R={R:.1f} J/(kg·K), γ={gamma:.3f}")
        print(f"  Expansion Ratio: {self.turbine_design['expansion_ratio']:.3f}")
        print(f"  Work Extracted: {work_total/1e6:.2f} MW\n")

        return {
            'rho': rho_out,
            'u': u_out,
            'p': p_out,
            'T': T_out_predicted,
            'cp': cp,      # Propagate fuel-dependent properties
            'R': R,
            'gamma': gamma,
            'work_specific': work_specific,
            'work_total': work_total
        }

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

        # Calculate exit temperature from isentropic relation
        T_exit = T_in * pressure_ratio**exponent

        # Calculate exit density from ideal gas law (fuel-dependent R)
        rho_exit = P_amb / (R * T_exit)

        # Calculate thrust: F = ṁ Δu + (P_exit - P_amb) A_exit
        # For perfectly expanded nozzle: P_exit = P_amb, so only momentum thrust contributes
        F_momentum = m_dot * u_exit_isentropic
        F_pressure = (P_amb - P_amb) * A_exit  # Zero for ideal expansion
        F_total = F_momentum + F_pressure

        print(f"[Nozzle]")
        print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar, u={flow_state_in['u']:.1f} m/s")
        print(f"  Exit:   T={T_exit:.1f} K, P={P_amb/1e3:.1f} kPa, u={u_exit_isentropic:.1f} m/s")
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

    def run_full_cycle(
        self,
        fuel_blend: LocalFuelBlend,
        phi: float = 0.5,
        combustor_efficiency: float = 0.98
    ) -> Dict[str, Any]:
        """
        Execute complete engine cycle and calculate performance metrics.

        Args:
            fuel_blend: LocalFuelBlend object
            phi: Equivalence ratio (default 0.5 for lean combustion)
            combustor_efficiency: Combustion efficiency (default 0.98)

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

        # Convert Cantera output to flow state for PINN input
        turb_inlet_state = self._cantera_to_flow_state(
            cantera_out=comb_result,
            m_dot=m_dot_total,
            A_ref=self.design_point['A_combustor_exit']
        )

        # 3. TURBINE
        turb_result = self.run_turbine(turb_inlet_state, m_dot_total)

        # 4. NOZZLE
        nozz_result = self.run_nozzle(turb_result, m_dot_total)

        # PERFORMANCE METRICS
        print("="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)

        thrust = nozz_result['thrust_total']

        # TSFC: Thrust Specific Fuel Consumption [kg fuel / N thrust / hour]
        tsfc = (m_dot_fuel * 3600) / thrust  # (kg/s * s/hr) / N = kg/(N·hr)

        # Thermal efficiency: η_th = (Thrust Power) / (Fuel Power)
        # Assuming LHV ≈ 43 MJ/kg for jet fuel
        LHV = 43e6  # J/kg
        fuel_power = m_dot_fuel * LHV  # W
        thrust_power = thrust * nozz_result['u']  # W (simplified)
        eta_thermal = thrust_power / fuel_power if fuel_power > 0 else 0

        print(f"  Fuel Blend:          {fuel_blend.name}")
        print(f"  Equivalence Ratio:   {phi:.3f}")
        print(f"  Fuel-Air Ratio:      {f:.6f}")
        print(f"  Core Mass Flow:      {m_dot_core:.2f} kg/s")
        print(f"  Fuel Mass Flow:      {m_dot_fuel:.4f} kg/s")
        print(f"  Total Mass Flow:     {m_dot_total:.2f} kg/s")
        print(f"  ---")
        print(f"  Thrust:              {thrust/1e3:.2f} kN")
        print(f"  TSFC:                {tsfc*1000:.2f} mg/(N·s)")
        print(f"  Thermal Efficiency:  {eta_thermal*100:.2f}%")
        print(f"  Compressor Work:     {comp_result['work_specific']*m_dot_core/1e6:.2f} MW")
        print(f"  Turbine Work:        {turb_result['work_total']/1e6:.2f} MW")
        print("="*70 + "\n")

        return {
            'compressor': comp_result,
            'combustor': comb_result,
            'turbine': turb_result,
            'nozzle': nozz_result,
            'performance': {
                'thrust_N': thrust,
                'thrust_kN': thrust / 1e3,
                'tsfc_mg_per_Ns': tsfc * 1000,
                'thermal_efficiency': eta_thermal,
                'fuel_mass_flow': m_dot_fuel,
                'total_mass_flow': m_dot_total,
                'fuel_air_ratio': f
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

        # Convert Cantera output to flow state for PINN input
        turb_inlet_state = self._cantera_to_flow_state(
            cantera_out=comb_result,
            m_dot=m_dot_total,
            A_ref=self.design_point['A_combustor_exit']
        )

        # 3. TURBINE
        turb_result = self.run_turbine(turb_inlet_state, m_dot_total)

        # 4. NOZZLE
        nozz_result = self.run_nozzle(turb_result, m_dot_total)

        # PERFORMANCE METRICS
        print("="*70)
        print("HYCHEM VALIDATION RESULTS")
        print("="*70)

        thrust = nozz_result['thrust_total']

        # TSFC: Thrust Specific Fuel Consumption [kg fuel / N thrust / hour]
        tsfc = (m_dot_fuel * 3600) / thrust  # (kg/s * s/hr) / N = kg/(N·hr)

        # Thermal efficiency: η_th = (Thrust Power) / (Fuel Power)
        LHV = 43e6  # J/kg
        fuel_power = m_dot_fuel * LHV  # W
        thrust_power = thrust * nozz_result['u']  # W (simplified)
        eta_thermal = thrust_power / fuel_power if fuel_power > 0 else 0

        print(f"  Mechanism:           HyChem (Stanford A1highT.yaml)")
        print(f"  Fuel:                Jet-A1 (Pure)")
        print(f"  Equivalence Ratio:   {phi:.3f}")
        print(f"  Fuel-Air Ratio:      {f:.6f}")
        print(f"  Core Mass Flow:      {m_dot_core:.2f} kg/s")
        print(f"  Fuel Mass Flow:      {m_dot_fuel:.4f} kg/s")
        print(f"  Total Mass Flow:     {m_dot_total:.2f} kg/s")
        print(f"  ---")
        print(f"  Thrust:              {thrust/1e3:.2f} kN")
        print(f"  TSFC:                {tsfc*1000:.2f} mg/(N·s)")
        print(f"  Thermal Efficiency:  {eta_thermal*100:.2f}%")
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
                'tsfc_mg_per_Ns': tsfc * 1000,
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