# Quick Reference: HyChem Validation Mode

## TL;DR

```bash
# Run HyChem validation (Jet-A1 ICAO benchmark)
python integrated_engine.py --mode validation

# Run blend comparison (default, uses CRECK)
python integrated_engine.py --mode blends
# OR
python integrated_engine.py
```

---

## When to Use Which Mode?

### Use **Validation Mode** when:
- ✅ Validating engine model against ICAO experimental data
- ✅ Testing pure Jet-A1 performance with highest fidelity
- ✅ Benchmarking the simulation framework
- ✅ Demonstrating model accuracy to judges/reviewers

### Use **Blends Mode** when:
- ✅ Comparing Jet-A1 vs. SAF blends
- ✅ Testing multiple fuel compositions
- ✅ Analyzing fuel-dependent performance differences
- ✅ Generating comparative performance tables

---

## Command-Line Usage

### Validation Mode
```bash
python integrated_engine.py --mode validation
```

**What it does:**
- Initializes engine with both HyChem and CRECK mechanisms
- Runs `run_hychem_validation_case()` with pure Jet-A1
- Uses HyChem mechanism (A1highT.yaml) for combustion
- Outputs validation results with metadata

**Output includes:**
- Thrust, TSFC, thermal efficiency
- Validation metadata (mechanism, fuel, purpose)
- Clear warning not to compare with CRECK results

---

### Blends Mode (Default)
```bash
python integrated_engine.py --mode blends
```

**What it does:**
- Initializes engine with CRECK mechanism
- Runs `run_full_cycle()` for multiple fuels
- Uses CRECK mechanism for ALL fuels
- Outputs comparative performance table

**Output includes:**
- Individual fuel results
- Comparative table with percentage deltas
- Baseline fuel marked with asterisk

---

## Programmatic API

### Validation Mode Example
```python
from integrated_engine import IntegratedTurbofanEngine

# Initialize in validation mode
engine = IntegratedTurbofanEngine(
    mechanism_profile="validation",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    hychem_mechanism_path="data/A1highT.yaml",
    turbine_pinn_path="turbine_pinn.pt",
    nozzle_pinn_path="nozzle_pinn.pt"
)

# Run validation
result = engine.run_hychem_validation_case(
    phi=0.5,
    combustor_efficiency=0.98
)

# Access results
print(f"Thrust: {result['performance']['thrust_kN']:.2f} kN")
print(f"TSFC: {result['performance']['tsfc_mg_per_Ns']:.2f} mg/(N·s)")
print(f"Mechanism: {result['validation_metadata']['mechanism']}")
```

---

### Blends Mode Example
```python
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

# Initialize in blends mode
engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",  # or omit (default)
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    turbine_pinn_path="turbine_pinn.pt",
    nozzle_pinn_path="nozzle_pinn.pt"
)

# Test multiple fuels
results = {}
for fuel_name in ["Jet-A1", "Bio-SPK", "HEFA-50"]:
    fuel = FUEL_LIBRARY[fuel_name]
    results[fuel_name] = engine.run_full_cycle(
        fuel_blend=fuel,
        phi=0.5,
        combustor_efficiency=0.98
    )

# Print comparison table
from integrated_engine import print_fuel_comparison
print_fuel_comparison(results, baseline_fuel="Jet-A1")
```

---

## Custom Fuel Testing (Advanced)

### Test Custom SAF Blend
```python
from integrated_engine import IntegratedTurbofanEngine
from simulation.fuels import make_saf_blend

# Initialize engine
engine = IntegratedTurbofanEngine(mechanism_profile="blends")

# Create custom blend: 70% Jet-A1, 30% HEFA
custom_fuel = make_saf_blend(
    p_j=0.7,  # 70% Jet-A1 surrogate
    p_h=0.3,  # 30% HEFA surrogate
    p_f=0.0,  # 0% Fischer-Tropsch
    p_a=0.0   # 0% Alcohol-to-Jet
)
custom_fuel.name = "Custom-HEFA30"

# Run simulation
result = engine.run_full_cycle(
    fuel_blend=custom_fuel,
    phi=0.5,
    combustor_efficiency=0.98
)

print(f"Custom blend thrust: {result['performance']['thrust_kN']:.2f} kN")
```

---

## Key Methods Reference

### `IntegratedTurbofanEngine.__init__()`
```python
def __init__(
    self,
    mechanism_profile: str = "blends",  # "validation" or "blends"
    creck_mechanism_path: str = "data/creck_c1c16_full.yaml",
    hychem_mechanism_path: str = "data/A1highT.yaml",
    turbine_pinn_path: str = "turbine_pinn.pt",
    nozzle_pinn_path: str = "nozzle_pinn.pt"
)
```

**Parameters:**
- `mechanism_profile`: Operating mode ("validation" or "blends")
- `creck_mechanism_path`: Path to CRECK mechanism file
- `hychem_mechanism_path`: Path to HyChem mechanism file
- `turbine_pinn_path`: Path to turbine PINN weights
- `nozzle_pinn_path`: Path to nozzle PINN weights

---

### `run_full_cycle()`
```python
def run_full_cycle(
    self,
    fuel_blend: LocalFuelBlend,
    phi: float = 0.5,
    combustor_efficiency: float = 0.98
) -> Dict[str, Any]
```

**Use for:** Blend comparisons (always uses CRECK)

**Returns:** Dict with `compressor`, `combustor`, `turbine`, `nozzle`, `performance`

---

### `run_hychem_validation_case()`
```python
def run_hychem_validation_case(
    self,
    phi: float = 0.5,
    combustor_efficiency: float = 0.98
) -> Dict[str, Any]
```

**Use for:** ICAO validation (always uses HyChem with pure Jet-A1)

**Returns:** Dict with same structure as `run_full_cycle()` plus `validation_metadata`

---

### `run_combustor()`
```python
def run_combustor(
    self,
    T_in: float,
    p_in: float,
    fuel_blend: LocalFuelBlend,
    phi: float = 0.5,
    efficiency: float = 0.98,
    use_hychem: bool = False  # Advanced: override mechanism
) -> Tuple[Dict[str, Any], float]
```

**Normally:** Called internally by `run_full_cycle()` and `run_hychem_validation_case()`

**Advanced usage:** Can override mechanism with `use_hychem=True`

---

## Safety Checks

The implementation includes several safety checks to prevent errors:

1. **HyChem Not Available:**
   ```python
   # This will raise RuntimeError if mechanism_profile != "validation"
   engine = IntegratedTurbofanEngine(mechanism_profile="blends")
   engine.run_hychem_validation_case()  # ERROR!
   ```

2. **Explicit Mechanism Override:**
   ```python
   # This will raise RuntimeError if combustor_hychem not initialized
   engine.run_combustor(..., use_hychem=True)  # ERROR if not in validation mode
   ```

3. **Clear Output Labels:**
   - All output clearly shows which mechanism was used
   - `[Combustor - CRECK]` vs. `[Combustor - HyChem]`

---

## Best Practices

### ✅ DO:
- Use validation mode for ICAO benchmarking
- Use blends mode for all comparative studies
- Keep mechanisms consistent within a single comparison
- Document which mechanism was used in your results

### ❌ DON'T:
- Mix HyChem and CRECK results in the same comparison table
- Compare validation mode results with blends mode results
- Use `use_hychem=True` unless you have a specific reason

---

## Troubleshooting

### Error: "HyChem validation mode not available"
**Solution:** Initialize engine with `mechanism_profile="validation"`

### Error: "mechanism_file is not defined"
**Cause:** Old initialization code
**Solution:** Use new parameter names:
```python
# Old (deprecated)
engine = IntegratedTurbofanEngine(mechanism_file="...")

# New (correct)
engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="..."
)
```

### Question: "Which mode should I use for my study?"
**Answer:**
- Pure Jet-A1 ICAO validation → `--mode validation`
- Any fuel blend comparison → `--mode blends` (default)

---

## Example Workflows

### Workflow 1: ICAO Validation
```bash
# 1. Run HyChem validation
python integrated_engine.py --mode validation

# 2. Compare results with ICAO experimental data
# (Manual comparison in spreadsheet or analysis script)
```

### Workflow 2: SAF Performance Study
```bash
# 1. Run blend comparison
python integrated_engine.py --mode blends

# 2. Analyze comparative table output
# 3. Document fuel-dependent performance differences
```

### Workflow 3: Custom Blend Optimization
```python
from integrated_engine import IntegratedTurbofanEngine
from simulation.fuels import make_saf_blend

engine = IntegratedTurbofanEngine(mechanism_profile="blends")

# Test range of blend ratios
blend_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
results = {}

for ratio in blend_ratios:
    fuel = make_saf_blend(p_j=1-ratio, p_h=ratio, p_f=0.0, p_a=0.0)
    fuel.name = f"HEFA-{int(ratio*100)}"

    result = engine.run_full_cycle(fuel_blend=fuel, phi=0.5)
    results[fuel.name] = result

# Find optimal blend
max_thrust = max(results.items(), key=lambda x: x[1]['performance']['thrust_kN'])
print(f"Optimal blend: {max_thrust[0]}")
```

---

## Summary

| Mode | Mechanism | Fuel(s) | Purpose |
|------|-----------|---------|---------|
| **Validation** | HyChem | Jet-A1 only | ICAO benchmark |
| **Blends** | CRECK | Any fuel(s) | Comparative studies |

**Golden Rule:** Never mix mechanisms in comparative analysis!

---

For detailed implementation information, see [HYCHEM_VALIDATION_SUMMARY.md](HYCHEM_VALIDATION_SUMMARY.md)
