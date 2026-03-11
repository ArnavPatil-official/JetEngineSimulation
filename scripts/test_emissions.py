"""
Test script for the Emissions Estimator module.

This script demonstrates multi-objective environmental optimization by comparing
emissions across different fuel blends with varying lifecycle carbon factors.
"""

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

# Initialize engine with emissions estimator
engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    hychem_mechanism_path="data/A1highT.yaml",
    turbine_pinn_path="models/turbine_pinn.pt",
    nozzle_pinn_path="models/nozzle_pinn.pt"
)

# Define LCA factors for each fuel
# Based on typical lifecycle assessments:
# - Jet-A1: 1.0 (baseline fossil fuel - full lifecycle emissions)
# - Bio-SPK: 0.2 (80% reduction from sustainable biomass feedstock)
# - HEFA-50: 0.6 (40% reduction from 50% bio-blend)
lca_factors = {
    "Jet-A1": 1.0,
    "Bio-SPK": 0.2,
    "HEFA-50": 0.6
}

print("\n" + "="*90)
print("MULTI-OBJECTIVE ENVIRONMENTAL OPTIMIZATION TEST")
print("="*90)
print("\nComparing emissions across fuel blends with lifecycle carbon accounting\n")

results = {}

for fuel_name, fuel_blend in FUEL_LIBRARY.items():
    lca = lca_factors.get(fuel_name, 1.0)

    print(f"\n{'='*90}")
    print(f"Testing: {fuel_name} (LCA Factor: {lca})")
    print(f"{'='*90}\n")

    result = engine.run_full_cycle(
        fuel_blend=fuel_blend,
        phi=0.5,
        combustor_efficiency=0.98,
        lca_factor=lca
    )

    results[fuel_name] = result

# Print emissions comparison table
print("\n" + "="*90)
print("EMISSIONS COMPARISON SUMMARY")
print("="*90)
print(f"{'Fuel':<15} {'NOx (g/s)':<12} {'CO (g/s)':<12} {'CO₂ (g/s)':<12} {'LCA Factor':<12} {'Thrust (kN)':<12}")
print("-"*90)

for fuel_name, result in results.items():
    perf = result['performance']
    emis = result['emissions']
    print(f"{fuel_name:<15} "
          f"{emis['NOx_g_s']:<12.2f} "
          f"{emis['CO_g_s']:<12.2f} "
          f"{emis['Net_CO2_g_s']:<12.2f} "
          f"{emis['lca_factor']:<12.2f} "
          f"{perf['thrust_kN']:<12.2f}")

print("-"*90)

# Calculate environmental benefit of SAF
jet_a1_co2 = results['Jet-A1']['emissions']['Net_CO2_g_s']
bio_spk_co2 = results['Bio-SPK']['emissions']['Net_CO2_g_s']
hefa_co2 = results['HEFA-50']['emissions']['Net_CO2_g_s']

print(f"\nCO₂ Reduction Analysis:")
print(f"  Bio-SPK vs Jet-A1:  {((jet_a1_co2 - bio_spk_co2)/jet_a1_co2)*100:.1f}% reduction")
print(f"  HEFA-50 vs Jet-A1:  {((jet_a1_co2 - hefa_co2)/jet_a1_co2)*100:.1f}% reduction")
print("="*90 + "\n")

print("✓ Emissions estimator test complete!")
