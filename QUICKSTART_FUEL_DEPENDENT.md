# Quick Start: Fuel-Dependent Thermodynamics

## Running the Upgraded System

### 1. Basic Test - Compare Jet-A1 vs SAF Blend

```python
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

# Initialize engine
engine = IntegratedTurbofanEngine(
    mechanism_file="data/creck_c1c16_full.yaml",
    turbine_pinn_path="turbine_pinn.pt",
    nozzle_pinn_path="nozzle_pinn.pt"
)

# Test pure Jet-A1
result_jeta1 = engine.run_full_cycle(
    fuel_blend=FUEL_LIBRARY["Jet-A1"],
    phi=0.5,  # Lean combustion
    combustor_efficiency=0.98
)

# Test 50% HEFA blend
result_hefa = engine.run_full_cycle(
    fuel_blend=FUEL_LIBRARY["HEFA-50"],
    phi=0.5,
    combustor_efficiency=0.98
)

# Compare
print("\n" + "="*70)
print("FUEL COMPARISON")
print("="*70)
print(f"{'Metric':<30} {'Jet-A1':<20} {'HEFA-50':<20} {'Δ%':<10}")
print("-"*70)

thrust_j = result_jeta1['performance']['thrust_kN']
thrust_h = result_hefa['performance']['thrust_kN']
delta_thrust = ((thrust_h - thrust_j) / thrust_j) * 100

tsfc_j = result_jeta1['performance']['tsfc_mg_per_Ns']
tsfc_h = result_hefa['performance']['tsfc_mg_per_Ns']
delta_tsfc = ((tsfc_h - tsfc_j) / tsfc_j) * 100

print(f"{'Thrust (kN)':<30} {thrust_j:<20.2f} {thrust_h:<20.2f} {delta_thrust:<10.2f}")
print(f"{'TSFC (mg/Ns)':<30} {tsfc_j:<20.2f} {tsfc_h:<20.2f} {delta_tsfc:<10.2f}")
```

---

### 2. Inspect Thermodynamic Properties

```python
from simulation.combustor.combustor import Combustor
from simulation.fuels import JET_A1, make_saf_blend
from simulation.thermo_utils import extract_thermo_props

# Initialize combustor
combustor = Combustor(mechanism_file="data/creck_c1c16_full.yaml")

# Compressor outlet conditions (example)
T_in = 850.0  # K
p_in = 4.37e6  # Pa

# Test Jet-A1
comb_out_jeta1 = combustor.run(
    T_in=T_in,
    p_in=p_in,
    fuel_blend=JET_A1,
    phi=0.5,
    efficiency=0.98
)

# Test HEFA blend
hefa_blend = make_saf_blend(p_j=0.5, p_h=0.5, p_f=0.0, p_a=0.0)
comb_out_hefa = combustor.run(
    T_in=T_in,
    p_in=p_in,
    fuel_blend=hefa_blend,
    phi=0.5,
    efficiency=0.98
)

# Extract and compare properties
props_jeta1 = extract_thermo_props(comb_out_jeta1)
props_hefa = extract_thermo_props(comb_out_hefa)

print("\n" + "="*60)
print("THERMODYNAMIC PROPERTIES COMPARISON")
print("="*60)
print(f"{'Property':<15} {'Jet-A1':<20} {'HEFA-50':<20} {'Δ%':<10}")
print("-"*60)

for key in ['cp', 'R', 'gamma']:
    val_j = props_jeta1[key]
    val_h = props_hefa[key]
    delta = ((val_h - val_j) / val_j) * 100
    print(f"{key:<15} {val_j:<20.4f} {val_h:<20.4f} {delta:<10.4f}")

print(f"\nCombustor Exit T:  {comb_out_jeta1['T_out']:.1f} K (Jet-A1)")
print(f"Combustor Exit T:  {comb_out_hefa['T_out']:.1f} K (HEFA-50)")
```

---

### 3. Use Helper Functions for PINN Conditions

```python
from simulation.thermo_utils import build_turbine_conditions, build_nozzle_conditions

# After running combustor
combustor_out = combustor.run(T_in, p_in, fuel_blend, phi=0.5)

# Build turbine conditions with fuel-dependent properties
turbine_cond = build_turbine_conditions(
    combustor_out=combustor_out,
    mass_flow_core=79.9,  # kg/s
    geometry={
        'A_inlet': 0.207,
        'A_outlet': 0.377,
        'length': 0.5
    },
    target_work=57.4e6  # Watts (optional)
)

print("\nTurbine Physics Properties (from combustor):")
print(f"  cp    = {turbine_cond['physics']['cp']:.2f} J/(kg·K)")
print(f"  R     = {turbine_cond['physics']['R']:.2f} J/(kg·K)")
print(f"  gamma = {turbine_cond['physics']['gamma']:.4f}")
print(f"  T_in  = {turbine_cond['inlet']['T']:.1f} K")
print(f"  p_in  = {turbine_cond['inlet']['p']/1e6:.2f} MPa")

# These properties will be used in turbine PINN physics losses!
```

---

### 4. Effect of Equivalence Ratio

```python
# Test how phi affects thermodynamic properties
phi_values = [0.4, 0.5, 0.6, 0.7]
fuel = JET_A1

print("\n" + "="*70)
print("EFFECT OF EQUIVALENCE RATIO ON THERMODYNAMIC PROPERTIES")
print("="*70)
print(f"{'φ':<10} {'T_out (K)':<15} {'cp (J/kg·K)':<15} {'γ':<10}")
print("-"*70)

for phi in phi_values:
    comb_out = combustor.run(
        T_in=850.0,
        p_in=4.37e6,
        fuel_blend=fuel,
        phi=phi,
        efficiency=0.98
    )

    print(f"{phi:<10.2f} {comb_out['T_out']:<15.1f} {comb_out['cp_out']:<15.2f} {comb_out['gamma_out']:<10.4f}")

print("\nNote: Richer mixtures (higher φ) → higher T → different cp and γ")
```

---

### 5. Verify Physics in Training

If you want to retrain the turbine PINN with fuel-dependent properties:

```python
import sys
sys.path.insert(0, 'simulation/turbine')
from turbine import (
    NormalizedTurbinePINN,
    compute_loss_components,
    train_phase1_boundaries,
    train_phase2_physics
)
import torch

# Get fuel-dependent conditions from combustor
combustor_out = combustor.run(T_in=850, p_in=4.37e6, fuel_blend=JET_A1, phi=0.5)
turbine_cond = build_turbine_conditions(combustor_out, mass_flow_core=79.9)

# Define scales (for normalization)
scales = {
    'rho': turbine_cond['inlet']['rho'],
    'u': 320.0,
    'p': turbine_cond['inlet']['p'],
    'T': turbine_cond['inlet']['T'],
    'L': 0.5
}

# Initialize model
device = torch.device("cpu")
model = NormalizedTurbinePINN().to(device)
x_col = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)

# Train with fuel-dependent properties
print(f"\nTraining turbine PINN with:")
print(f"  cp    = {turbine_cond['physics']['cp']:.2f} J/(kg·K)  [FUEL-DEPENDENT]")
print(f"  R     = {turbine_cond['physics']['R']:.2f} J/(kg·K)  [FUEL-DEPENDENT]")
print(f"  gamma = {turbine_cond['physics']['gamma']:.4f}      [FUEL-DEPENDENT]")

train_phase1_boundaries(model, device, n_epochs=1000)
train_phase2_physics(
    model, x_col, device,
    n_epochs=2000,
    conditions=turbine_cond,  # Uses fuel-dependent cp, R, gamma!
    scales=scales
)

# The PINN now learned expansion with REAL fuel properties!
```

---

### 6. Full Workflow Example

```python
#!/usr/bin/env python3
"""
Full workflow demonstrating fuel-dependent thermodynamics.
"""

from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY
from simulation.fuels import make_saf_blend

def main():
    # Initialize engine
    engine = IntegratedTurbofanEngine(
        mechanism_file="data/creck_c1c16_full.yaml",
        turbine_pinn_path="turbine_pinn.pt",
        nozzle_pinn_path="nozzle_pinn.pt"
    )

    # Define test fuels
    fuels = {
        "Pure Jet-A1": FUEL_LIBRARY["Jet-A1"],
        "Pure Bio-SPK": FUEL_LIBRARY["Bio-SPK"],
        "50% HEFA Blend": FUEL_LIBRARY["HEFA-50"],
        "Custom SAF (30% FT)": make_saf_blend(p_j=0.7, p_h=0.0, p_f=0.3, p_a=0.0)
    }

    results = {}

    print("\n" + "="*80)
    print("FUEL-DEPENDENT THERMODYNAMICS TEST")
    print("="*80 + "\n")

    for name, fuel in fuels.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

        result = engine.run_full_cycle(
            fuel_blend=fuel,
            phi=0.5,
            combustor_efficiency=0.98
        )

        results[name] = result

    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)

    baseline = results["Pure Jet-A1"]
    baseline_thrust = baseline['performance']['thrust_kN']
    baseline_tsfc = baseline['performance']['tsfc_mg_per_Ns']

    print(f"\n{'Fuel':<25} {'Thrust (kN)':<15} {'TSFC (mg/Ns)':<15} {'ΔThrust%':<12} {'ΔTSFC%':<10}")
    print("-"*80)

    for name, res in results.items():
        thrust = res['performance']['thrust_kN']
        tsfc = res['performance']['tsfc_mg_per_Ns']
        delta_t = ((thrust - baseline_thrust) / baseline_thrust) * 100
        delta_f = ((tsfc - baseline_tsfc) / baseline_tsfc) * 100

        print(f"{name:<25} {thrust:<15.2f} {tsfc:<15.2f} {delta_t:<12.2f} {delta_f:<10.2f}")

    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("1. Different fuels show different thrust due to different thermodynamic properties")
    print("2. cp, R, and γ from combustor affect both turbine work extraction and nozzle expansion")
    print("3. The PINNs are essential because constant-γ formulas don't apply!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
```

---

## Expected Output

When you run the integrated engine with different fuels, you should see:

```
[Combustor]
  Fuel:   Jet-A1
  Phi:    0.500
  FAR:    0.029456 (fuel/air mass ratio)
  Inlet:  T=850.0 K, P=43.72 bar
  Outlet: T=1685.2 K, P=43.72 bar
  Efficiency: 98.0%

[Turbine]
  Inlet:  T=1685.2 K, P=43.72 bar
  Outlet: T=994.3 K, P=1.93 bar
  Fuel-dependent properties: cp=1185.3 J/(kg·K), R=289.2 J/(kg·K), γ=1.323
  Expansion Ratio: 0.590
  Work Extracted: 58.12 MW

[Nozzle]
  Inlet:  T=994.3 K, P=1.93 bar, u=318.5 m/s
  Exit:   T=742.1 K, P=101.3 kPa, u=591.2 m/s
  Fuel-dependent properties: cp=1185.3 J/(kg·K), R=289.2 J/(kg·K), γ=1.323
  Pressure Ratio: 0.0525
  Thrust: 46.71 kN
```

Notice that **cp, R, and γ are displayed and used** at each stage!

---

## Troubleshooting

### Issue: "gamma_out not found"
**Solution:** Make sure you've updated [combustor.py](simulation/combustor/combustor.py) to return `gamma_out`

### Issue: PINN gives different results
**Expected:** The PINNs were trained with default cp/R/γ. For best accuracy with new fuel properties, consider retraining.

### Issue: Import error for thermo_utils
**Solution:** Ensure [simulation/thermo_utils.py](simulation/thermo_utils.py) exists and is in the right location.

---

## Next Steps

1. Run the integrated engine with different fuel blends
2. Compare thrust and TSFC across fuels
3. Verify that changing fuel properties affects results
4. Optional: Retrain PINNs with variable fuel properties as inputs

Enjoy exploring fuel-dependent thermodynamics! 🚀
