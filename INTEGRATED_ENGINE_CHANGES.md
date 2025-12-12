# Required Changes to integrated_engine.py

## 1. Add New Imports (After line 45)

```python
# Import new data structures and utilities
from simulation.engine_types import FlowState, PerformanceMetrics, EngineCycleResult, FuelProperties
from simulation.emissions import estimate_emissions_indices
from simulation.nozzle.nozzle_pinn_v2 import (
    NozzlePINN, load_model as load_nozzle_model,
    detect_choking, compute_critical_pressure_ratio
)
import math
```

## 2. Add Constants (After import block, ~line 48)

```python
# Physical and efficiency constants
BETA_EXERGY = 1.06  # Chemical exergy factor for hydrocarbon fuels
MW_CO2 = 44.01  # g/mol
MW_C = 12.01    # g/mol
```

## 3. Update LocalFuelBlend Class (line 52)

Add LHV and carbon fraction fields:

```python
class LocalFuelBlend:
    def __init__(self, name: str, composition: Dict[str, float],
                 LHV_MJ_per_kg: float = 43.2, carbon_fraction: float = 0.847):
        self.name = name
        self.composition = composition
        self.LHV_MJ_per_kg = LHV_MJ_per_kg
        self.carbon_fraction = carbon_fraction

        total = sum(composition.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Mass fractions must sum to 1.0, got {total}")
```

## 4. Update FUEL_LIBRARY (line ~98)

```python
FUEL_LIBRARY = {
    "Jet-A1": LocalFuelBlend("Jet-A1", {"NC12H26": 1.0},
                             LHV_MJ_per_kg=44.1, carbon_fraction=0.847),
    "Bio-SPK": LocalFuelBlend("Bio-SPK", {"NC10H22": 1.0},
                              LHV_MJ_per_kg=43.8, carbon_fraction=0.845),
    "HEFA-50": LocalFuelBlend("HEFA-50", {"NC12H26": 0.5, "NC10H22": 0.5},
                              LHV_MJ_per_kg=43.95, carbon_fraction=0.846),
}
```

## 5. Update IntegratedTurbofanEngine.__init__ (line ~370)

Add nozzle PINN mode config and u0:

```python
class IntegratedTurbofanEngine:
    # Configuration flags
    USE_TURBINE_DT_SCALING = True
    USE_NOZZLE_PINN = False  # Set True to use PINN nozzle
    NOZZLE_MODE = "analytic"  # Options: "analytic", "pinn_anchored"

    def __init__(self, ...):
        # ... existing code ...

        # Update design_point with inlet velocity
        self.design_point = {
            'mass_flow_core': 79.9,
            'bypass_ratio': 9.1,
            'A_combustor_exit': 0.207,
            'A_nozzle_exit': 0.340,
            'P_ambient': 101325.0,
            'T_ambient': 288.15,
            'u0': 0.0  # Inlet flight speed [m/s] (0 for static test)
        }

        # Load nozzle PINN if enabled
        if self.USE_NOZZLE_PINN:
            try:
                self.nozzle_pinn_model, self.nozzle_pinn_scales, \
                self.nozzle_pinn_conditions, _ = load_nozzle_model("nozzle_pinn.pt")
                print("✓ Loaded nozzle PINN for anchored mode")
            except Exception as e:
                print(f"⚠️  Warning: Could not load nozzle PINN: {e}")
                self.USE_NOZZLE_PINN = False
```

## 6. Add New Method: run_nozzle_pinn_anchored()

Insert after `run_nozzle()` method (~line 766):

```python
def run_nozzle_pinn_anchored(
    self,
    flow_state_in: Dict[str, float],
    m_dot: float
) -> Dict[str, float]:
    """
    Simulate nozzle using PINN with anchored boundary conditions.

    This method:
    1. Queries PINN for flow profile
    2. Applies anchoring to match inlet and enforce physics
    3. Detects choking and adjusts accordingly
    4. Computes thrust with fuel-dependent properties

    Args:
        flow_state_in: Inlet state from turbine
        m_dot: Mass flow rate [kg/s]

    Returns:
        Nozzle exit state with thrust
    """
    import torch

    T_in = flow_state_in['T']
    p_in = flow_state_in['p']
    cp = flow_state_in['cp']
    R = flow_state_in['R']
    gamma = flow_state_in.get('gamma', cp / (cp - R))

    P_amb = self.design_point['P_ambient']
    A_exit = self.design_point['A_nozzle_exit']
    u0 = self.design_point['u0']

    # Detect choking
    is_choked, p_critical = detect_choking(p_in, P_amb, gamma)

    if is_choked:
        # Use isentropic relations for choked flow
        pr = p_critical / p_in
        exponent = (gamma - 1) / gamma
        T_exit = T_in * (pr ** exponent)
        p_exit = p_critical

        # For M=1 at throat (simplified converging nozzle model)
        rho_exit = p_exit / (R * T_exit)
        u_exit = np.sqrt(gamma * R * T_exit)  # Mach 1
    else:
        # Query PINN for exit state
        with torch.no_grad():
            x_exit = torch.tensor([[1.0]], dtype=torch.float32)
            out_norm = self.nozzle_pinn_model(x_exit)

            # Denormalize
            p_exit_pinn = out_norm[0, 2].item() * self.nozzle_pinn_scales['p']
            u_exit_pinn = out_norm[0, 1].item() * self.nozzle_pinn_scales['u']
            T_exit_pinn = out_norm[0, 3].item() * self.nozzle_pinn_scales['T']

        # Anchor to actual inlet conditions
        p_scale = p_in / self.nozzle_pinn_conditions['inlet']['p']
        p_exit = p_exit_pinn * p_scale

        # Enforce ambient pressure (soft anchoring)
        p_exit = 0.7 * p_exit + 0.3 * P_amb  # Blend with ambient

        # Recalculate with fuel-dependent isentropic relations
        pr = p_exit / p_in
        exponent = (gamma - 1) / gamma
        T_exit = T_in * (pr ** exponent)

        rho_exit = p_exit / (R * T_exit)

        # Mass continuity: u = m_dot / (rho * A)
        u_exit = m_dot / (rho_exit * A_exit)

    # Compute net thrust
    F_momentum = m_dot * (u_exit - u0)
    F_pressure = (p_exit - P_amb) * A_exit
    F_total = F_momentum + F_pressure

    print(f"[Nozzle PINN Anchored]")
    print(f"  Inlet:  T={T_in:.1f} K, P={p_in/1e5:.2f} bar")
    print(f"  Exit:   T={T_exit:.1f} K, P={p_exit/1e3:.1f} kPa, u={u_exit:.1f} m/s")
    print(f"  Choking: {is_choked}")
    print(f"  Thrust: {F_total/1e3:.2f} kN\n")

    return {
        'rho': rho_exit,
        'u': u_exit,
        'p': p_exit,
        'T': T_exit,
        'cp': cp,
        'R': R,
        'gamma': gamma,
        'thrust_total': F_total,
        'thrust_momentum': F_momentum,
        'thrust_pressure': F_pressure,
        'is_choked': is_choked
    }
```

## 7. Update run_full_cycle() (line ~753)

Replace performance metrics computation section (lines ~808-853) with:

```python
# PERFORMANCE METRICS (New comprehensive calculation)
print("="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

# Extract values
thrust_N = nozz_result['thrust_total']
u_exit = nozz_result['u']
u0 = self.design_point['u0']

# Fuel properties
LHV_J_per_kg = fuel_blend.LHV_MJ_per_kg * 1e6  # Convert MJ to J
w_C = fuel_blend.carbon_fraction

# 1. TSFC: Thrust Specific Fuel Consumption
tsfc_kg_per_Ns = m_dot_fuel / thrust_N  # kg/(N·s)
tsfc_mg_per_Ns = tsfc_kg_per_Ns * 1e6   # mg/(N·s)

# 2. Thermal Efficiency: P_jet / (m_fuel * LHV)
P_jet = 0.5 * m_dot_total * (u_exit**2 - u0**2)  # Jet kinetic power [W]
fuel_power = m_dot_fuel * LHV_J_per_kg  # Fuel chemical power [W]
eta_thermal = P_jet / fuel_power if fuel_power > 0 else 0.0

# 3. Exergy Efficiency: P_jet / (m_fuel * beta * LHV)
eta_exergy = P_jet / (fuel_power * BETA_EXERGY) if fuel_power > 0 else 0.0

# 4. CO₂ Emissions (from carbon balance, NOT from ICAO directly)
m_dot_CO2 = m_dot_fuel * w_C * (MW_CO2 / MW_C)  # kg/s
co2_g_per_kN_s = (m_dot_CO2 / (thrust_N / 1e3)) * 1e3  # g/(kN·s)

# 5. NOx and CO Emissions (correlation-based for now)
emissions_result = estimate_emissions_indices(
    combustor_out=comb_result,
    gas=self.gas if hasattr(self, 'gas') else None,
    m_dot_fuel=m_dot_fuel,
    use_cantera=False,  # Use correlation for blend comparisons
    mode='cruise'
)
EI_NOx = emissions_result['EI_NOx']  # g/kg_fuel
EI_CO = emissions_result['EI_CO']    # g/kg_fuel

# 6. Validity checks for optimizer
is_valid = True
error_msg = ""

# Check physics constraints
if thrust_N <= 0 or not math.isfinite(thrust_N):
    is_valid = False
    error_msg = f"Invalid thrust: {thrust_N}"
elif nozz_result['p'] < self.design_point['P_ambient']:
    is_valid = False
    error_msg = f"Nozzle inlet pressure below ambient"
elif comb_result['gamma_out'] <= 1.0:
    is_valid = False
    error_msg = f"Invalid gamma: {comb_result['gamma_out']}"
elif comb_result['cp_out'] <= 0:
    is_valid = False
    error_msg = f"Invalid cp: {comb_result['cp_out']}"
elif comb_result['T_out'] <= 0:
    is_valid = False
    error_msg = f"Invalid temperature: {comb_result['T_out']}"

# Print summary
print(f"  Fuel Blend:          {fuel_blend.name}")
print(f"  LHV:                 {fuel_blend.LHV_MJ_per_kg:.2f} MJ/kg")
print(f"  Carbon Fraction:     {fuel_blend.carbon_fraction:.3f}")
print(f"  Equivalence Ratio:   {phi:.3f}")
print(f"  Fuel-Air Ratio:      {f:.6f}")
print(f"  ---")
print(f"  Net Thrust:          {thrust_N/1e3:.2f} kN")
print(f"  TSFC:                {tsfc_mg_per_Ns:.2f} mg/(N·s)")
print(f"  Thermal Efficiency:  {eta_thermal*100:.2f}%")
print(f"  Exergy Efficiency:   {eta_exergy*100:.2f}%")
print(f"  ---")
print(f"  CO₂ per Thrust:      {co2_g_per_kN_s:.2f} g/(kN·s)")
print(f"  EI_NOx:              {EI_NOx:.2f} g/kg_fuel ({emissions_result['method']})")
print(f"  EI_CO:               {EI_CO:.2f} g/kg_fuel ({emissions_result['method']})")
print(f"  ---")
print(f"  Valid Solution:      {is_valid}")
if not is_valid:
    print(f"  Error: {error_msg}")
print("="*70 + "\n")

# Create structured performance metrics
performance_metrics = PerformanceMetrics(
    net_thrust_N=thrust_N,
    net_thrust_kN=thrust_N / 1e3,
    tsfc_kg_per_Ns=tsfc_kg_per_Ns,
    tsfc_mg_per_Ns=tsfc_mg_per_Ns,
    co2_g_per_kN_s=co2_g_per_kN_s,
    EI_NOx=EI_NOx,
    EI_CO=EI_CO,
    eta_thermal=eta_thermal,
    eta_exergy=eta_exergy,
    m_dot_fuel=m_dot_fuel,
    m_dot_total=m_dot_total,
    fuel_air_ratio=f,
    is_valid=is_valid,
    error_msg=error_msg
)

fuel_properties = FuelProperties(
    name=fuel_blend.name,
    LHV_J_per_kg=LHV_J_per_kg,
    w_C=w_C,
    composition=fuel_blend.as_composition_string()
)

# Return structured result
return EngineCycleResult(
    compressor=comp_result,
    combustor=comb_result,
    turbine=turb_result,
    nozzle=nozz_result,
    performance=performance_metrics,
    fuel_properties=fuel_properties
).to_dict()  # Convert to dict for backward compatibility
```

## 8. Update Nozzle Call in run_full_cycle()

Replace line 806:

```python
# OLD:
# nozz_result = self.run_nozzle(turb_result, m_dot_total)

# NEW:
if self.USE_NOZZLE_PINN and self.NOZZLE_MODE == "pinn_anchored":
    nozz_result = self.run_nozzle_pinn_anchored(turb_result, m_dot_total)
else:
    nozz_result = self.run_nozzle(turb_result, m_dot_total)
```

## 9. Add Optimizer Wrapper Method (After run_full_cycle)

```python
def evaluate_for_optimizer(
    self,
    fuel_blend: LocalFuelBlend,
    phi: float,
    combustor_efficiency: float = 0.98,
    return_dict: bool = True
) -> Dict[str, Any]:
    """
    Optimizer-safe engine evaluation wrapper.

    This method wraps run_full_cycle() with error handling and returns
    a consistent structure for optimization algorithms.

    Args:
        fuel_blend: Fuel blend to evaluate
        phi: Equivalence ratio
        combustor_efficiency: Combustion efficiency
        return_dict: If True, return dict; else return PerformanceMetrics

    Returns:
        Performance metrics dict or PerformanceMetrics object
        If execution fails, returns invalid metrics with NaN values
    """
    try:
        result = self.run_full_cycle(fuel_blend, phi, combustor_efficiency)

        if return_dict:
            return result['performance']
        else:
            return PerformanceMetrics(**result['performance'])

    except Exception as e:
        print(f"⚠️  Error in evaluate_for_optimizer: {e}")
        # Return invalid metrics
        invalid_metrics = PerformanceMetrics(
            net_thrust_N=np.nan,
            net_thrust_kN=np.nan,
            tsfc_kg_per_Ns=np.nan,
            tsfc_mg_per_Ns=np.nan,
            co2_g_per_kN_s=np.nan,
            EI_NOx=np.nan,
            EI_CO=np.nan,
            eta_thermal=np.nan,
            eta_exergy=np.nan,
            m_dot_fuel=np.nan,
            m_dot_total=np.nan,
            fuel_air_ratio=np.nan,
            is_valid=False,
            error_msg=str(e)
        )

        if return_dict:
            return invalid_metrics.to_dict()
        else:
            return invalid_metrics
```

## Summary of Changes

### Files Modified:
- `integrated_engine.py`: Added imports, constants, updated methods, new metrics

### New Functionality:
1. ✅ Comprehensive performance metrics (TSFC, CO₂, NOx, CO, exergy efficiency)
2. ✅ CO₂ computed from fuel carbon balance (not ICAO directly)
3. ✅ Emissions estimation (Cantera + correlation modes)
4. ✅ Net thrust with inlet velocity u0
5. ✅ Fixed thermal/exergy efficiency definitions
6. ✅ Optimizer-safe validity flags
7. ✅ Nozzle PINN anchored mode with choking detection
8. ✅ Fuel properties (LHV, carbon fraction) in all blends
9. ✅ Structured dataclass outputs

### Maintained:
- ✅ HyChem/CRECK separation logic intact
- ✅ Existing CLI modes (`--mode blends`, `--mode validation`)
- ✅ Backward compatibility (returns dicts via `.to_dict()`)
