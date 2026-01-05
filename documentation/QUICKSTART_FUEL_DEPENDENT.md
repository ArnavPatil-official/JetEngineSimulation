# Quick Start: Fuel-Dependent Simulation

## Prerequisites
- Python 3.10+, Cantera, PyTorch, NumPy/Matplotlib (for plotting in examples)
- `nozzle_pinn.pt` checkpoint version ≥ v3.1 (otherwise the analytic nozzle will be used automatically)

## 1) Run a Blend Comparison (CRECK)
```python
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    turbine_pinn_path="turbine_pinn.pt",
    nozzle_pinn_path="nozzle_pinn.pt",
)

results = {}
for name in ["Jet-A1", "Bio-SPK", "HEFA-50"]:
    results[name] = engine.run_full_cycle(
        fuel_blend=FUEL_LIBRARY[name],
        phi=0.5,
        combustor_efficiency=0.98,
    )

from integrated_engine import print_fuel_comparison
print_fuel_comparison(results, baseline_fuel="Jet-A1")
```

## 2) Run HyChem Validation (Jet-A1 only)
```bash
python integrated_engine.py --mode validation
```
Uses `data/A1highT.yaml` and returns TSFC/efficiency for the ICAO-style Jet-A1 benchmark. Do not mix these results with CRECK-based comparisons.

## 3) Inspect Combustor Thermodynamics
```python
from simulation.combustor.combustor import Combustor
from simulation.fuels import JET_A1, make_saf_blend
from simulation.thermo_utils import extract_thermo_props

combustor = Combustor(mechanism_file="data/creck_c1c16_full.yaml")
hefa_blend = make_saf_blend(p_j=0.5, p_h=0.5, p_f=0.0, p_a=0.0)

for fuel in [JET_A1, hefa_blend]:
    out = combustor.run(T_in=850.0, p_in=4.37e6, fuel_blend=fuel, phi=0.5, efficiency=0.98)
    props = extract_thermo_props(out)
    print(f"\n{fuel.name}")
    print(f"  cp    = {props['cp']:.2f} J/(kg·K)")
    print(f"  R     = {props['R']:.2f} J/(kg·K)")
    print(f"  gamma = {props['gamma']:.4f}")
    print(f"  T_out = {out['T_out']:.1f} K, p_out = {out['p_out']/1e5:.2f} bar")
```

## 4) Reuse Thermo Props in PINNs
```python
from simulation.thermo_utils import build_turbine_conditions, build_nozzle_conditions
from simulation.nozzle.nozzle import run_nozzle_pinn

# Assume combustor_out is produced as above
turbine_conditions = build_turbine_conditions(combustor_out, mass_flow_core=79.9)
nozzle_conditions = build_nozzle_conditions(
    turbine_exit_state=turbine_conditions['inlet'],
    thermo_props=turbine_conditions['physics'],
    mass_flow=79.9,
    p_ambient=101325.0,
)

result = run_nozzle_pinn(
    model_path="nozzle_pinn.pt",
    inlet_state=nozzle_conditions['inlet'],
    ambient_p=nozzle_conditions['ambient']['p'],
    A_in=nozzle_conditions['geometry']['A_inlet'],
    A_exit=nozzle_conditions['geometry']['A_exit'],
    length=nozzle_conditions['geometry']['length'],
    thermo_props=nozzle_conditions['physics'],
    m_dot=nozzle_conditions['physics']['mass_flow'],
)
print(f"Thrust: {result['thrust_total']/1e3:.2f} kN (fallback={result['used_fallback']})")
```

## 5) Quick Test Hooks
- `python -m pytest tests/test_nozzle_pinn_fix.py` — positive thrust and scaling checks.
- `python -m pytest tests/test_nozzle_regression.py` — PINN vs. analytic regression.
- `python -m pytest tests/test_choking_detection.py` — choking criteria.
