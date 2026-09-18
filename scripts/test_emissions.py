"""
Test script for the Emissions Estimator module.

This script demonstrates multi-objective environmental optimization by comparing
emissions across different fuel blends with varying lifecycle carbon factors.
"""

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

# Initialize engine with emissions estimator
engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    hychem_mechanism_path="data/A1highT.yaml",
    turbine_pinn_path="models/turbine_pinn.pt",
    nozzle_pinn_path="models/nozzle_pinn.pt"
)

# Lifecycle carbon factors, derived from CORSIA default life-cycle emissions
# values (data/corsia_lca_values.yaml; ICAO Doc 06, 8th ed., Nov 2025) rather
# than asserted.
#
# The retired hard-coded factors {Jet-A1 1.0, Bio-SPK 0.2, HEFA-50 0.6} had no
# source and sat near each pathway's BEST case; the 0.2 is what produced the
# withdrawn "80% CO2 cut" claim, which was an echo of the input, not a result.
#
# A CORSIA L_CEF is a RANGE per pathway, not a point. The `mode` value is used
# below so the script runs deterministically, and the min/max band is printed
# alongside every number so no single reduction figure can be quoted from it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "data" / "corsia_lca_values.yaml") as _fh:
    CORSIA = yaml.safe_load(_fh)

LCEF_BASELINE = CORSIA["baseline_fossil_gCO2e_MJ"]          # 89 gCO2e/MJ
HEFA_TRI = CORSIA["pathways"]["HEFA"]["triangular"]         # gCO2e/MJ


def _factor(lcef_gco2e_mj: float) -> float:
    """CORSIA L_CEF -> carbon factor relative to the fossil-jet baseline."""
    return lcef_gco2e_mj / LCEF_BASELINE


# Bio-SPK is modelled here as a neat HEFA-pathway SPK; HEFA-50 as a 50/50 blend
# of that pathway with fossil Jet-A1.
LCA_BANDS = {
    "Jet-A1":  (1.0, 1.0, 1.0),
    "Bio-SPK": tuple(_factor(HEFA_TRI[k]) for k in ("min", "mode", "max")),
    "HEFA-50": tuple(0.5 + 0.5 * _factor(HEFA_TRI[k]) for k in ("min", "mode", "max")),
}
lca_factors = {name: band[1] for name, band in LCA_BANDS.items()}

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

# Lifecycle carbon factor ranges.
#
# Net_CO2_g_s is affine in the assumed lifecycle factor, so any "X% reduction"
# read off this script is a restatement of the CORSIA input, NOT a model output.
# The band is printed to make that explicit; a single reduction figure is
# deliberately not produced here. The manuscript-bound lifecycle numbers come
# from the seeded Monte Carlo in scripts/optimization/optimize_blend.py
# (manifest rows S5/S6), not from this script.
print("\nLifecycle carbon factor ranges (CORSIA L_CEF / fossil baseline):")
print(f"  {'Fuel':<12} {'min':>8} {'mode':>8} {'max':>8}   (used: mode)")
for _name, (_lo, _mid, _hi) in LCA_BANDS.items():
    print(f"  {_name:<12} {_lo:>8.3f} {_mid:>8.3f} {_hi:>8.3f}")
print("\n  Net CO2 scales linearly with these factors -- treat the spread, not")
print("  the point value, as the lifecycle result of this screening model.")
print("="*90 + "\n")

print("✓ Emissions estimator test complete!")
