"""
Generate a parallel-coordinates style plot of all optimization runs.
Pareto-optimal blends are highlighted with thicker, darker strokes.
"""

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_style("whitegrid")

# Load results
df = pd.read_csv('outputs/results/optimization_results.csv')
if df.empty:
    raise SystemExit("No data found in optimization_results.csv")

# Columns to visualize (blend + performance)
cols = ['HEFA_Frac', 'FT_Frac', 'ATJ_Frac', 'SAF_Total', 'Phi', 'TSFC', 'SpecThrust', 'CO2', 'NOx']
missing = [c for c in cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in optimization_results.csv: {missing}")


def identify_pareto_front(df, objectives, minimize):
    """Identify Pareto-optimal solutions."""
    is_pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        for j in range(len(df)):
            if i == j:
                continue
            dominates = True
            strictly_better = False
            for obj_idx, obj in enumerate(objectives):
                if minimize[obj_idx]:
                    if df.iloc[j][obj] > df.iloc[i][obj]:
                        dominates = False
                        break
                    elif df.iloc[j][obj] < df.iloc[i][obj]:
                        strictly_better = True
                else:
                    if df.iloc[j][obj] < df.iloc[i][obj]:
                        dominates = False
                        break
                    elif df.iloc[j][obj] > df.iloc[i][obj]:
                        strictly_better = True
            if dominates and strictly_better:
                is_pareto[i] = False
                break
    return is_pareto


# Compute ParetoOptimal if not present
if 'ParetoOptimal' not in df.columns:
    objectives = ['TSFC', 'CO2', 'NOx', 'SpecThrust']
    minimize = [True, True, True, False]  # minimize TSFC, CO2, NOx; maximize SpecThrust
    df['ParetoOptimal'] = identify_pareto_front(df, objectives, minimize)
    print(f"Computed Pareto front: {df['ParetoOptimal'].sum()} optimal solutions out of {len(df)}")

# Normalize each column to [0,1] for plotting
norm = df[cols].copy()
for c in cols:
    span = norm[c].max() - norm[c].min()
    norm[c] = 0.5 if span == 0 else (norm[c] - norm[c].min()) / (span + 1e-9)

colors = plt.cm.viridis(df['SAF_Total'].clip(0, 1))
fig, ax = plt.subplots(figsize=(13, 7))

for i, row in norm.iterrows():
    pareto = bool(df.loc[i, 'ParetoOptimal'])
    lw = 2.8 if pareto else 0.9
    alpha = 0.9 if pareto else 0.25
    z = 3 if pareto else 1
    ax.plot(range(len(cols)), row.values, color=colors[i], linewidth=lw, alpha=alpha, zorder=z)

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=20, ha='right')
ax.set_ylabel('Normalized scale')
ax.set_title('All Optimization Runs with Pareto-Optimal Blends Highlighted')
ax.set_xlim(0, len(cols) - 1)
ax.set_ylim(-0.05, 1.05)

# Legend
legend_lines = [
    Line2D([0], [0], color='k', lw=2.8, label='Pareto-optimal'),
    Line2D([0], [0], color='k', lw=0.9, alpha=0.4, label='Non-Pareto')
]
ax.legend(handles=legend_lines, loc='upper right')

plt.tight_layout()
plt.savefig('outputs/plots/marked_parallel_coordinates.png', dpi=300, bbox_inches='tight')
print("Saved marked parallel coordinates to 'outputs/plots/marked_parallel_coordinates.png'")

