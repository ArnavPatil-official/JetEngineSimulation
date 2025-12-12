# Comprehensive Engine Upgrade Implementation Summary

## Overview

This document summarizes the judge-proof, optimizer-ready upgrades to the jet engine simulation codebase.

## Completed Work

### Part A: New Metrics & Physics Definitions (IN PROGRESS)

#### Created Files:
1. **`simulation/engine_types.py`** - Data structures for type safety
   - `FlowState`: Flow field state with fuel-dependent thermo props
   - `FuelProperties`: Fuel thermochemical properties (LHV, carbon fraction)
   - `PerformanceMetrics`: Complete optimizer-ready metrics
   - `EngineCycleResult`: Structured cycle output

2. **`simulation/emissions.py`** - Emissions estimation module
   - `extract_cantera_emissions()`: Extract CO/NOx from Cantera if available
   - `estimate_emissions_correlation()`: Empirical correlations (Lefebvre & Ballal)
   - `estimate_emissions_indices()`: Unified interface for engine cycle
   - Modes: Cantera-based or correlation-based (labeled)

3. **`simulation/fuels.py`** - Updated with LHV and carbon fractions
   - Added `LHV_MJ_per_kg` and `carbon_fraction` fields to `FuelSurrogate`
   - All fuels updated with accurate values:
     - JET_A1: LHV=44.1 MJ/kg, w_C=0.847
     - HEFA_SPK: LHV=44.0, w_C=0.846
     - FT_SPK: LHV=43.9, w_C=0.846
     - ATJ_SPK: LHV=43.5, w_C=0.843

#### New Performance Metrics (to be integrated in `integrated_engine.py`):
- `net_thrust_N` and `net_thrust_kN`
- `tsfc_kg_per_Ns` and `tsfc_mg_per_Ns`
- `co2_g_per_kN_s` - CO₂ per thrust (computed from fuel carbon, NOT ICAO)
- `EI_NOx` and `EI_CO` - Emissions indices [g/kg_fuel]
- `eta_thermal` - Fixed definition: P_jet / (m_fuel * LHV)
- `eta_exergy` - P_jet / (m_fuel * beta * LHV), beta=1.06
- `is_valid` - Validity flag for optimizer
- `error_msg` - Constraint violation description

#### Physics Improvements:
- Added `u0` (inlet flight speed) to design_point (default 0 for static)
- Net thrust: `F = m_dot*(u_exit - u0) + (p_exit - p_amb)*A_exit`
- Jet power: `P_jet = 0.5*m_dot*(u_exit² - u0²)`
- CO₂ calculation: `m_dot_CO2 = m_dot_fuel * w_C * (44/12)`
- CO₂ per thrust: `co2_g_per_kN_s = (m_dot_CO2 / F) * 1e3` [g/(kN·s)]

#### Optimizer Guards (to be added):
- Check for: negative thrust, non-finite numbers, p_nozzle_inlet < p_amb
- Check for: gamma ≤ 1, cp ≤ 0, T ≤ 0
- Return `PerformanceMetrics` with `is_valid=False` if violated

### Part C: Nozzle PINN Upgrades (COMPLETED)

#### Created Files:
1. **`simulation/nozzle/nozzle_conditions.py`**
   - `load_engine_conditions_from_icao()`: ICAO data loader with fuel-dependent thermo
   - `build_nozzle_conditions_from_turbine_exit()`: Runtime conditions builder
   - Supports runtime thermo props injection

2. **`simulation/nozzle/nozzle_pinn_v2.py`** - Complete rewrite with:
   - **Reproducibility**: Fixed random seeds (RANDOM_SEED=42)
   - **Robust paths**: All paths relative to REPO_ROOT
   - **Enhanced checkpoints**: Saves model + scales + conditions + thermo baseline + version tag
   - **Choking detection**: `detect_choking()` and `compute_critical_pressure_ratio()`
   - **Mach-aware physics**: Added Mach number penalty in loss (prevents unphysical M>1 in converging section)
   - **Improved boundary conditions**:
     - Strict inlet enforcement (100x weight)
     - Soft outlet pressure with penalty for p_exit < p_amb
   - **Version tracking**: VERSION_TAG = "v2.0_fuel_dependent_choking"
   - **Better validation**: Prints net thrust, choking status, Mach profile
   - **Saved plots**: `nozzle_validation.png`

### Part B: Nozzle PINN Integration (PENDING)

**To be implemented in `integrated_engine.py`:**

1. Add config flags:
   ```python
   USE_NOZZLE_PINN = True  # Enable nozzle PINN usage
   NOZZLE_MODE = "pinn_anchored"  # Options: "analytic", "pinn_anchored"
   ```

2. Implement `run_nozzle_pinn_anchored()` method:
   - Load nozzle PINN checkpoint
   - Query PINN for x-profile from turbine exit state
   - Apply anchoring/rescaling:
     - Enforce inlet = turbine exit (rho, u, p, T)
     - Enforce exit pressure = ambient (or critical if choked)
     - Enforce mass continuity at exit
     - Enforce stagnation enthalpy consistency with fuel cp/gamma
   - Compute thrust with anchored exit state
   - Detect choking and handle appropriately

3. Add choking logic:
   ```python
   gamma = flow_state_in['gamma']
   pr_critical = (2/(gamma+1))**(gamma/(gamma-1))
   is_choked = (p_amb/p_inlet <= pr_critical)
   if is_choked:
       # Set exit M=1, compute exit state from isentropic relations
   ```

4. Add regression test utility:
   - Compare `run_nozzle()` (analytic) vs `run_nozzle_pinn_anchored()` on baseline
   - Should agree within ~5% on thrust for Jet-A1 surrogate

### Part D: Optimization Readiness (PENDING)

**To be implemented:**

1. Refactor `run_full_cycle()` to return `EngineCycleResult` dataclass
2. Create `evaluate_for_optimizer()` wrapper:
   ```python
   def evaluate_for_optimizer(fuel_blend, phi, mode='cruise'):
       try:
           result = run_full_cycle(fuel_blend, phi)
           if not result.performance.is_valid:
               return penalty_vector()
           return result.performance.to_dict()
       except Exception as e:
           return penalty_vector()
   ```

3. Update `LocalFuelBlend` to use `FuelSurrogate` from `simulation/fuels.py` OR
   add LHV/carbon_fraction fields to `LocalFuelBlend`

## File Status

| File | Status | Notes |
|------|--------|-------|
| `simulation/engine_types.py` | ✅ Created | Dataclasses ready |
| `simulation/emissions.py` | ✅ Created | Dual-mode (Cantera + correlation) |
| `simulation/fuels.py` | ✅ Updated | Added LHV & carbon fraction |
| `simulation/nozzle/nozzle_conditions.py` | ✅ Created | Conditions builder |
| `simulation/nozzle/nozzle_pinn_v2.py` | ✅ Created | Improved PINN with choking |
| `integrated_engine.py` | ⏳ PENDING | Needs Part A, B, D integration |
| Unit tests | ⏳ PENDING | Choking test, nozzle regression test |

## Next Steps

1. **Update `integrated_engine.py`**:
   - Import new modules (`engine_types`, `emissions`)
   - Add u0 to design_point
   - Update `run_full_cycle()` to compute new metrics
   - Implement `run_nozzle_pinn_anchored()`
   - Add validity checks and guard clauses
   - Return `EngineCycleResult` dataclass

2. **Create unit tests**:
   - `test_choking_detection.py`: Test choking logic with constructed inlet conditions
   - `test_nozzle_regression.py`: Analytic vs PINN agreement test
   - `test_optimizer_guards.py`: Test validity flag for edge cases

3. **Update CLI** (`integrated_engine.py` main block):
   - Ensure `--mode blends` and `--mode validation` still work
   - Add `--nozzle-mode` flag for PINN selection

## Key Physics Constants

```python
BETA_EXERGY = 1.06  # Chemical exergy factor for hydrocarbons
MW_CO2 = 44.01  # g/mol
MW_C = 12.01  # g/mol
CO2_per_C = 44.01 / 12.01  # Mass ratio: 3.664
```

## Mechanism Separation (Maintained)

- **HyChem (A1highT.yaml)**: ONLY for Jet-A1 validation against ICAO
- **CRECK (creck_c1c16_full.yaml)**: For ALL blend comparisons and optimization
- **Never mix** HyChem and CRECK results in same comparative table

## Critical Formulas

### CO₂ Emissions:
```
m_dot_CO2 [kg/s] = m_dot_fuel [kg/s] × w_C [-] × (44/12) [-]
CO2_per_thrust [g/(kN·s)] = (m_dot_CO2 / F_net) × 1e3
```

### Thermal Efficiency:
```
P_jet [W] = 0.5 × m_dot × (u_exit² - u0²)
eta_th [-] = P_jet / (m_dot_fuel × LHV)
```

### Exergy Efficiency:
```
eta_ex [-] = P_jet / (m_dot_fuel × beta × LHV)
where beta = 1.06 (common approximation for kerosene)
```

### Net Thrust:
```
F_net [N] = m_dot × (u_exit - u0) + (p_exit - p_amb) × A_exit
```

### Choking Criterion:
```
pr_critical = (2/(gamma+1))^(gamma/(gamma-1))
is_choked = (p_amb / p_inlet) ≤ pr_critical
```

## Optimizer-Safe Output

All `PerformanceMetrics` include:
- `is_valid: bool` - False if any physics constraint violated
- `error_msg: str` - Description of violation
- All metrics initialized to `np.nan` if invalid
- Optimizer can penalize based on `is_valid` flag

## Version Control

- Nozzle PINN version: `v2.0_fuel_dependent_choking`
- All checkpoints include version tag for reproducibility
- Random seed: 42 (fixed for all training)
