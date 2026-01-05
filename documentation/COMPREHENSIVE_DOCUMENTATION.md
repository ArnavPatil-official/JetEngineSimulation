# Jet Engine Simulation - Comprehensive Documentation

## Overview
- Hybrid grey-box turbofan model: Cantera for chemistry/thermodynamics and PINNs for turbine/nozzle flow physics.
- Fuel-dependent throughout: cp, R, and gamma are extracted from Cantera and passed into the turbine and nozzle PINNs (no constant-air shortcuts).
- Default thrust model is a static test stand: `F = m_dot * u_exit + (p_exit - p_amb) * A_exit`, with automatic fallback to an analytical nozzle if the PINN fails physics checks.

## Quick Start
- CLI (blends mode, CRECK mechanism): `python integrated_engine.py --mode blends`
- CLI (HyChem validation, Jet-A1 only): `python integrated_engine.py --mode validation`
- Python snippet for blends:
```python
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    turbine_pinn_path="turbine_pinn.pt",
    nozzle_pinn_path="nozzle_pinn.pt",  # v3.1+ checkpoints; falls back to analytic if older
)

result = engine.run_full_cycle(
    fuel_blend=FUEL_LIBRARY["HEFA-50"],
    phi=0.5,
    combustor_efficiency=0.98,
)
print(f"Thrust: {result['performance']['thrust_kN']:.2f} kN")
print(f"TSFC:   {result['performance']['tsfc_mg_per_Ns']:.2f} mg/(N·s)")
```

## Architecture & Key Files
- `integrated_engine.py`: Orchestrates the Brayton cycle, selects HyChem vs. CRECK, and routes to turbine/nozzle PINN APIs with safety gates and fallbacks.
- `simulation/`: Core modules.
  - `compressor/`, `combustor/`: Cantera-based compression/combustion.
  - `turbine/`: Fuel-dependent PINN with exact continuity; `run_turbine_pinn()` API.
  - `nozzle/`: Fuel-dependent PINN v4.0 (`VERSION_TAG = "v4.0_exact_continuity"`) plus analytical fallback.
  - `fuels.py`: Fuel surrogate definitions and blending utilities.
  - `engine_types.py`, `thermo_utils.py`, `emissions.py`: Helper dataclasses, thermo bridges, and emissions utilities.
- `data/`: Mechanisms (`creck_c1c16_full.yaml`, `A1highT.yaml`, etc.) and ICAO reference data.
- `tests/`: Regression and physics checks (`test_nozzle_pinn_fix.py`, `test_nozzle_regression.py`, `test_choking_detection.py`, `verify_*`).
- `optimize_blend.py`: Optuna demo for blend exploration.

## Component Notes
- **Compressor/Combustor**: Cantera equilibrium with user-selectable mechanism; returns cp/R/gamma for downstream stages. Fuel-air ratio is computed with Cantera stoichiometry.
- **Turbine PINN**: Inputs include thermo props; velocity computed from `u = m_dot / (rho * A)` for exact continuity. Target work defaults to compressor work if not provided.
- **Nozzle PINN**: 8D input (`x*, cp*, R*, gamma*, rho_in*, u_in*, p_in*, T_in*`), three hidden layers × 64 (Tanh), predicts residuals anchored to the inlet. Physics gates enforce inlet reproduction, mass conservation, and sane exit states before accepting predictions. Automatic analytic fallback retains fuel-dependent gamma if the checkpoint is missing or fails validation.
- **Fuel models**: `FUEL_LIBRARY` in `integrated_engine.py` holds LocalFuelBlend surrogates (Jet-A1, Bio-SPK, HEFA-50). Richer surrogates and blending helpers live in `simulation/fuels.py`.

## Operation Modes
- **Blends mode (default)**: Uses CRECK (`data/creck_c1c16_full.yaml`) for all fuels; call `run_full_cycle`.
- **Validation mode**: Uses HyChem (`data/A1highT.yaml`) for pure Jet-A1; call `run_hychem_validation_case`. Do not mix HyChem results with CRECK comparisons.

## Performance Outputs
- `thrust_kN`, `tsfc_mg_per_Ns`, `thermal_efficiency` (kinetic proxy for static stand), fuel/total mass flow, and fuel-air ratio.
- TSFC uses the corrected kg→mg conversion (×1e6) and is set to infinity if thrust ≤ 0.
- Thrust model (static test stand): `F = m_dot * u_exit + (p_exit - p_amb) * A_exit`; pressure and momentum components are printed separately when the PINN is used.
- Nozzle compatibility check requires checkpoint version ≥ v3.1; otherwise the analytic nozzle with fuel-dependent gamma is used automatically.

## Validation & Tests
- `tests/test_nozzle_pinn_fix.py`: Ensures positive thrust and scaling robustness for the nozzle PINN.
- `tests/test_nozzle_regression.py`: Compares PINN vs. analytical nozzle behavior.
- `tests/test_choking_detection.py`: Validates choking logic.
- `tests/verify_physics_corrections.py`, `tests/verify_thermo_fix.py`: Guard against regressions in physics and thermo normalization.

For HyChem-specific history see `CHANGELOG_HYCHEM.md`. For nozzle details see `NOZZLE_PINN_GUIDE.md`; quick usage examples live in `QUICKSTART_FUEL_DEPENDENT.md`.
