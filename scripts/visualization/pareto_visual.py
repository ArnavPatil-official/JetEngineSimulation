"""
Pareto Front Visualization for Multi-Objective Fuel Blend Optimization

This script visualizes the optimization results showing trade-offs between:
- Thrust Specific Fuel Consumption (TSFC) - minimize
- CO2 emissions - minimize
- NOx emissions - minimize
- Specific Thrust - maximize
"""

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 10)
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Load optimization results
df = pd.read_csv('outputs/results/optimization_results.csv')

print(f"Loaded {len(df)} optimization results")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData summary:")
print(df.describe())


def identify_pareto_front(df, objectives, minimize):
    """
    Identify Pareto-optimal solutions.
    
    Parameters:
    - df: DataFrame with optimization results
    - objectives: list of objective column names
    - minimize: list of booleans indicating whether to minimize each objective
    
    Returns:
    - Boolean array indicating Pareto-optimal points
    """
    is_pareto = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        
        # For each point, check if any other point dominates it
        for j in range(len(df)):
            if i == j:
                continue
            
            # Check if j dominates i
            dominates = True
            strictly_better = False
            
            for obj_idx, obj in enumerate(objectives):
                if minimize[obj_idx]:
                    # Minimization objective
                    if df.iloc[j][obj] > df.iloc[i][obj]:
                        dominates = False
                        break
                    elif df.iloc[j][obj] < df.iloc[i][obj]:
                        strictly_better = True
                else:
                    # Maximization objective
                    if df.iloc[j][obj] < df.iloc[i][obj]:
                        dominates = False
                        break
                    elif df.iloc[j][obj] > df.iloc[i][obj]:
                        strictly_better = True
            
            if dominates and strictly_better:
                is_pareto[i] = False
                break
    
    return is_pareto


# Define optimization objectives
objectives = ['TSFC', 'CO2', 'NOx', 'SpecThrust']
minimize_flags = [True, True, True, False]  # TSFC, CO2, NOx minimize; SpecThrust maximize

# Identify Pareto front
pareto_mask = identify_pareto_front(df, objectives, minimize_flags)
pareto_df = df[pareto_mask]
non_pareto_df = df[~pareto_mask]

print(f"\nPareto-optimal solutions: {sum(pareto_mask)}")
print(f"Non-Pareto solutions: {sum(~pareto_mask)}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. TSFC vs CO2 emissions
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(non_pareto_df['TSFC'], non_pareto_df['CO2'], 
           c='lightblue', alpha=0.3, s=30, label='Non-Pareto')
ax1.scatter(pareto_df['TSFC'], pareto_df['CO2'], 
           c='red', alpha=0.8, s=80, marker='*', label='Pareto Front')
ax1.set_xlabel('TSFC (mg/N·s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('CO₂ Emissions (g/kg-fuel)', fontsize=11, fontweight='bold')
ax1.set_title('TSFC vs CO₂ Trade-off', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. TSFC vs NOx emissions
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(non_pareto_df['TSFC'], non_pareto_df['NOx'], 
           c='lightblue', alpha=0.3, s=30, label='Non-Pareto')
ax2.scatter(pareto_df['TSFC'], pareto_df['NOx'], 
           c='red', alpha=0.8, s=80, marker='*', label='Pareto Front')
ax2.set_xlabel('TSFC (mg/N·s)', fontsize=11, fontweight='bold')
ax2.set_ylabel('NOₓ Emissions (g/kg-fuel)', fontsize=11, fontweight='bold')
ax2.set_title('TSFC vs NOₓ Trade-off', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. CO2 vs NOx emissions
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(non_pareto_df['CO2'], non_pareto_df['NOx'], 
           c='lightblue', alpha=0.3, s=30, label='Non-Pareto')
ax3.scatter(pareto_df['CO2'], pareto_df['NOx'], 
           c='red', alpha=0.8, s=80, marker='*', label='Pareto Front')
ax3.set_xlabel('CO₂ Emissions (g/kg-fuel)', fontsize=11, fontweight='bold')
ax3.set_ylabel('NOₓ Emissions (g/kg-fuel)', fontsize=11, fontweight='bold')
ax3.set_title('CO₂ vs NOₓ Trade-off', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. SAF Total vs LCA (Lifecycle Assessment)
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(non_pareto_df['SAF_Total'], non_pareto_df['LCA'], 
           c='lightblue', alpha=0.3, s=30, label='Non-Pareto')
ax4.scatter(pareto_df['SAF_Total'], pareto_df['LCA'], 
           c='red', alpha=0.8, s=80, marker='*', label='Pareto Front')
ax4.set_xlabel('SAF Total Fraction', fontsize=11, fontweight='bold')
ax4.set_ylabel('LCA Score', fontsize=11, fontweight='bold')
ax4.set_title('SAF Content vs Lifecycle Assessment', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. SpecThrust vs TSFC
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(non_pareto_df['SpecThrust'], non_pareto_df['TSFC'], 
           c='lightblue', alpha=0.3, s=30, label='Non-Pareto')
ax5.scatter(pareto_df['SpecThrust'], pareto_df['TSFC'], 
           c='red', alpha=0.8, s=80, marker='*', label='Pareto Front')
ax5.set_xlabel('Specific Thrust (N·s/kg)', fontsize=11, fontweight='bold')
ax5.set_ylabel('TSFC (mg/N·s)', fontsize=11, fontweight='bold')
ax5.set_title('Specific Thrust vs TSFC Trade-off', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Equivalence Ratio vs NOx
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(non_pareto_df['Phi'], non_pareto_df['NOx'], 
           c='lightblue', alpha=0.3, s=30, label='Non-Pareto')
ax6.scatter(pareto_df['Phi'], pareto_df['NOx'], 
           c='red', alpha=0.8, s=80, marker='*', label='Pareto Front')
ax6.set_xlabel('Equivalence Ratio (Φ)', fontsize=11, fontweight='bold')
ax6.set_ylabel('NOₓ Emissions (g/kg-fuel)', fontsize=11, fontweight='bold')
ax6.set_title('Equivalence Ratio vs NOₓ', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/pareto_front_2d.png', dpi=300, bbox_inches='tight')
print("\nSaved 2D Pareto front visualization to 'outputs/plots/pareto_front_2d.png'")
plt.show()

# Create 3D Pareto front visualization
fig2 = plt.figure(figsize=(16, 12))

# 3D plot: TSFC vs CO2 vs NOx
ax_3d1 = fig2.add_subplot(2, 2, 1, projection='3d')
ax_3d1.scatter(non_pareto_df['TSFC'], non_pareto_df['CO2'], non_pareto_df['NOx'],
              c='lightblue', alpha=0.2, s=20, label='Non-Pareto')
ax_3d1.scatter(pareto_df['TSFC'], pareto_df['CO2'], pareto_df['NOx'],
              c='red', alpha=0.8, s=100, marker='*', label='Pareto Front')
ax_3d1.set_xlabel('TSFC (mg/N·s)', fontweight='bold')
ax_3d1.set_ylabel('CO₂ (g/kg-fuel)', fontweight='bold')
ax_3d1.set_zlabel('NOₓ (g/kg-fuel)', fontweight='bold')
ax_3d1.set_title('3D Pareto Front: TSFC-CO₂-NOₓ', fontweight='bold')
ax_3d1.legend()

# 3D plot: TSFC vs CO2 vs SpecThrust
ax_3d2 = fig2.add_subplot(2, 2, 2, projection='3d')
ax_3d2.scatter(non_pareto_df['TSFC'], non_pareto_df['CO2'], non_pareto_df['SpecThrust'],
              c='lightblue', alpha=0.2, s=20, label='Non-Pareto')
ax_3d2.scatter(pareto_df['TSFC'], pareto_df['CO2'], pareto_df['SpecThrust'],
              c='red', alpha=0.8, s=100, marker='*', label='Pareto Front')
ax_3d2.set_xlabel('TSFC (mg/N·s)', fontweight='bold')
ax_3d2.set_ylabel('CO₂ (g/kg-fuel)', fontweight='bold')
ax_3d2.set_zlabel('Specific Thrust (N·s/kg)', fontweight='bold')
ax_3d2.set_title('3D Pareto Front: TSFC-CO₂-Thrust', fontweight='bold')
ax_3d2.legend()

# 3D plot: SAF blend composition
ax_3d3 = fig2.add_subplot(2, 2, 3, projection='3d')
ax_3d3.scatter(non_pareto_df['HEFA_Frac'], non_pareto_df['FT_Frac'], non_pareto_df['ATJ_Frac'],
              c=non_pareto_df['NOx'], cmap='viridis', alpha=0.3, s=20)
sc = ax_3d3.scatter(pareto_df['HEFA_Frac'], pareto_df['FT_Frac'], pareto_df['ATJ_Frac'],
                   c=pareto_df['NOx'], cmap='plasma', alpha=0.8, s=100, marker='*', 
                   edgecolors='black', linewidths=1)
ax_3d3.set_xlabel('HEFA Fraction', fontweight='bold')
ax_3d3.set_ylabel('FT Fraction', fontweight='bold')
ax_3d3.set_zlabel('ATJ Fraction', fontweight='bold')
ax_3d3.set_title('SAF Blend Composition (colored by NOₓ)', fontweight='bold')
plt.colorbar(sc, ax=ax_3d3, label='NOₓ (g/kg-fuel)', shrink=0.5)

# 3D plot: Performance metrics
ax_3d4 = fig2.add_subplot(2, 2, 4, projection='3d')
ax_3d4.scatter(non_pareto_df['SAF_Total'], non_pareto_df['LCA'], non_pareto_df['TSFC'],
              c='lightblue', alpha=0.2, s=20, label='Non-Pareto')
ax_3d4.scatter(pareto_df['SAF_Total'], pareto_df['LCA'], pareto_df['TSFC'],
              c='red', alpha=0.8, s=100, marker='*', label='Pareto Front')
ax_3d4.set_xlabel('SAF Total Fraction', fontweight='bold')
ax_3d4.set_ylabel('LCA Score', fontweight='bold')
ax_3d4.set_zlabel('TSFC (mg/N·s)', fontweight='bold')
ax_3d4.set_title('3D Pareto Front: SAF-LCA-TSFC', fontweight='bold')
ax_3d4.legend()

plt.tight_layout()
plt.savefig('outputs/plots/pareto_front_3d.png', dpi=300, bbox_inches='tight')
print("Saved 3D Pareto front visualization to 'outputs/plots/pareto_front_3d.png'")
plt.show()

# Create heatmap of correlations for Pareto-optimal solutions
fig3, ax = plt.subplots(figsize=(12, 10))
correlation_cols = ['TSFC', 'SpecThrust', 'CO2', 'NOx', 'SAF_Total', 'Phi', 'LCA']
correlation_matrix = pareto_df[correlation_cols].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Pareto-Optimal Solutions', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/plots/pareto_correlations.png', dpi=300, bbox_inches='tight')
print("Saved correlation heatmap to 'outputs/plots/pareto_correlations.png'")
plt.show()

# Summary statistics
print("\n" + "="*80)
print("PARETO-OPTIMAL SOLUTIONS SUMMARY")
print("="*80)
print("\nPareto Front Statistics:")
print(pareto_df[objectives].describe())

print("\n" + "="*80)
print("Best individual objectives in Pareto front:")
print("="*80)
print(f"Minimum TSFC: {pareto_df['TSFC'].min():.2f} mg/N·s")
print(f"Maximum Specific Thrust: {pareto_df['SpecThrust'].max():.2f} N·s/kg")
print(f"Minimum CO₂: {pareto_df['CO2'].min():.2f} g/kg-fuel")
print(f"Minimum NOₓ: {pareto_df['NOx'].min():.2f} g/kg-fuel")
print(f"Maximum SAF Total: {pareto_df['SAF_Total'].max():.4f}")
print(f"Minimum LCA: {pareto_df['LCA'].min():.4f}")

# Save Pareto-optimal solutions to CSV
pareto_df.to_csv('outputs/results/pareto_optimal_solutions.csv', index=False)
print("\n" + "="*80)
print(f"Saved {len(pareto_df)} Pareto-optimal solutions to 'outputs/results/pareto_optimal_solutions.csv'")
print("="*80)
