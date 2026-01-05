#!/usr/bin/env python3
"""
Verification script to demonstrate the thermodynamic normalization fix.

This script shows the difference between the broken normalization (always 1.0)
and the fixed normalization (actual variation).
"""

# Reference values from THERMO_REF
THERMO_REF = {
    'cp': 1150.0,
    'R': 287.0,
    'gamma': 1.33
}

print("="*70)
print("THERMODYNAMIC NORMALIZATION FIX VERIFICATION")
print("="*70)

# Test Case 1: Baseline combustion products
thermo_1 = {'cp': 1150.0, 'R': 287.0, 'gamma': 1.33}

# Test Case 2: High-temperature combustion (e.g., afterburner)
thermo_2 = {'cp': 1384.0, 'R': 289.8, 'gamma': 1.265}

print("\nTest Case 1: Baseline (typical combustion)")
print(f"  cp = {thermo_1['cp']:.1f} J/(kg·K)")
print(f"  R  = {thermo_1['R']:.1f} J/(kg·K)")
print(f"  γ  = {thermo_1['gamma']:.3f}")

print("\nTest Case 2: High-temp (afterburner-like)")
print(f"  cp = {thermo_2['cp']:.1f} J/(kg·K)")
print(f"  R  = {thermo_2['R']:.1f} J/(kg·K)")
print(f"  γ  = {thermo_2['gamma']:.3f}")

print("\n" + "="*70)
print("BROKEN NORMALIZATION (cp_scale = cp_current)")
print("="*70)

# Broken: normalize by current value
cp_norm_1_broken = thermo_1['cp'] / thermo_1['cp']
gamma_norm_1_broken = thermo_1['gamma'] / thermo_1['gamma']

cp_norm_2_broken = thermo_2['cp'] / thermo_2['cp']
gamma_norm_2_broken = thermo_2['gamma'] / thermo_2['gamma']

print(f"\nCase 1: cp_norm = {cp_norm_1_broken:.6f}, γ_norm = {gamma_norm_1_broken:.6f}")
print(f"Case 2: cp_norm = {cp_norm_2_broken:.6f}, γ_norm = {gamma_norm_2_broken:.6f}")
print(f"\nDifference: Δcp_norm = {abs(cp_norm_2_broken - cp_norm_1_broken):.6f}")
print(f"            Δγ_norm  = {abs(gamma_norm_2_broken - gamma_norm_1_broken):.6f}")
print("\n❌ Network sees ZERO variation → thermo-blind!")

print("\n" + "="*70)
print("FIXED NORMALIZATION (cp_scale = THERMO_REF['cp'])")
print("="*70)

# Fixed: normalize by reference value
cp_norm_1_fixed = thermo_1['cp'] / THERMO_REF['cp']
gamma_norm_1_fixed = thermo_1['gamma'] / THERMO_REF['gamma']

cp_norm_2_fixed = thermo_2['cp'] / THERMO_REF['cp']
gamma_norm_2_fixed = thermo_2['gamma'] / THERMO_REF['gamma']

print(f"\nCase 1: cp_norm = {cp_norm_1_fixed:.6f}, γ_norm = {gamma_norm_1_fixed:.6f}")
print(f"Case 2: cp_norm = {cp_norm_2_fixed:.6f}, γ_norm = {gamma_norm_2_fixed:.6f}")
print(f"\nDifference: Δcp_norm = {abs(cp_norm_2_fixed - cp_norm_1_fixed):.6f}")
print(f"            Δγ_norm  = {abs(gamma_norm_2_fixed - gamma_norm_1_fixed):.6f}")
print(f"\nRelative change:")
print(f"  cp: {(cp_norm_2_fixed / cp_norm_1_fixed - 1.0)*100:+.2f}%")
print(f"  γ:  {(gamma_norm_2_fixed / gamma_norm_1_fixed - 1.0)*100:+.2f}%")
print("\n✅ Network sees REAL variation → thermo-sensitive!")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nThe fix ensures that when gamma changes from 1.33 to 1.265:")
print(f"  • Network input changes from {gamma_norm_1_fixed:.3f} to {gamma_norm_2_fixed:.3f}")
print(f"  • This {abs(gamma_norm_2_fixed - gamma_norm_1_fixed)*100:.1f}% variation enables fuel-dependent predictions")
print(f"\nPreviously, the network saw {gamma_norm_1_broken:.3f} in both cases (blind).")
print("="*70)
