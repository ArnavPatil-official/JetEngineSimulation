"""
Multi-objective optimization for jet engine fuel blends with emissions.
"""

import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import io
import sys
import re
from mpl_toolkits.mplot3d import Axes3D
from integrated_engine import IntegratedTurbofanEngine, LocalFuelBlend
from simulation.fuels import make_saf_blend

# --- CONFIGURATION ---
N_TRIALS = 1000           # Number of optimization trials
TIT_HARD_LIMIT = 2800.0   # Physics failure point [K]
TIT_SOFT_LIMIT = 1850.0   # Cooling penalty threshold [K]
LCA_FACTORS = {
    'JetA': 1.0,
    'HEFA': 0.2,
    'FT': 0.1,
    'ATJ': 0.3
}

optuna.logging.set_verbosity(optuna.logging.ERROR)

print("="*80)
print("🚀 4-OBJECTIVE OPTIMIZATION: Performance + Environment")
print(f"Targeting: {N_TRIALS} Trials")
print("Output Format: Single-line summary per trial")
print("="*80 + "\n")

# --- 1. INITIALIZE ENGINE ---
engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    hychem_mechanism_path="data/A1highT.yaml",
    turbine_pinn_path="turbine_pinn.pt",
    nozzle_pinn_path="nozzle_pinn.pt",
    icao_data_path="data/icao_engine_data.csv"
)

# --- 2. FUEL WRAPPER ---
class SafeFuelWrapper:
    def __init__(self, name, species_dict):
        self.name = name; self.composition = species_dict
    def as_composition_string(self):
        return ", ".join([f"{k}:{v}" for k, v in self.composition.items()])
    def __repr__(self): return f"SafeFuelWrapper({self.name})"

# --- 3. SCRAPER ---
def scrape_log_data(log_text):
    try:
        tsfc = float(re.search(r"TSFC:\s+([\d\.]+)", log_text).group(1))
        thrust = float(re.search(r"Thrust:\s+([\d\.]+)", log_text).group(1))
        t4_match = re.search(r"Combustor.*?Outlet:\s*T=([\d\.]+)", log_text, re.DOTALL) or \
                   re.search(r"Turbine.*?Inlet:\s*T=([\d\.]+)", log_text, re.DOTALL)
        t4 = float(t4_match.group(1)) if t4_match else 2000.0
        nox = float(re.search(r"NOx:\s+([\d\.]+)", log_text).group(1))
        co2 = float(re.search(r"CO₂:\s+([\d\.]+)", log_text).group(1))
        return tsfc, thrust, t4, nox, co2
    except: return None, None, None, None, None

# --- 4. OBJECTIVE FUNCTION ---
def objective(trial):
    # --- A. Design Variables ---
    saf_total = trial.suggest_float("saf_total", 0.0, 0.5)
    jet_a = 1.0 - saf_total

    w_h = trial.suggest_float("w_hefa", 0.0, 1.0)
    w_f = trial.suggest_float("w_ft", 0.0, 1.0)
    w_a = trial.suggest_float("w_atj", 0.0, 1.0)

    total_w = w_h + w_f + w_a + 1e-6
    p_h = saf_total * (w_h / total_w)
    p_f = saf_total * (w_f / total_w)
    p_a = saf_total * (w_a / total_w)

    phi = trial.suggest_float("phi", 0.35, 0.65)

    lca_factor = (jet_a * LCA_FACTORS['JetA'] + p_h * LCA_FACTORS['HEFA'] +
                  p_f * LCA_FACTORS['FT'] + p_a * LCA_FACTORS['ATJ'])

    # --- B. Simulation (Captured) ---
    capture_buffer = io.StringIO()
    
    try:
        raw_blend = make_saf_blend(jet_a, p_h, p_f, p_a, enforce_astm=True)
        fuel = SafeFuelWrapper(f"Trial_{trial.number}", raw_blend.species)

        # CAPTURE ALL OUTPUT (Suppress engine noise)
        with contextlib.redirect_stdout(capture_buffer):
            result = engine.run_full_cycle(
                fuel_blend=fuel, phi=phi, combustor_efficiency=0.98, lca_factor=lca_factor
            )

        # --- C. Data Extraction ---
        if 'performance' in result and 'emissions' in result:
            tsfc = result['performance']['tsfc_mg_per_Ns']
            thrust = result['performance']['thrust_kN']
            nox = result['emissions']['NOx_g_s']
            co2 = result['emissions']['Net_CO2_g_s']
            m_dot = result['performance']['total_mass_flow'] - result['performance']['fuel_mass_flow']
            t4 = result['combustor']['T_out']
        else:
            tsfc, thrust, t4, nox, co2 = scrape_log_data(capture_buffer.getvalue())
            if tsfc is None: raise optuna.TrialPruned()
            m_dot = 79.9

        # --- D. Constraints & Penalties ---
        if t4 > TIT_HARD_LIMIT:
            print(f"Trial {trial.number:03d}: ❌ PRUNED (T4={t4:.0f}K > Limit)")
            raise optuna.TrialPruned()

        penalty = 1.0 + max(0.0, (t4 - TIT_SOFT_LIMIT) * 0.0005)

        final_tsfc = tsfc * penalty
        spec_thrust = (thrust * 1000.0 / m_dot) / penalty
        final_co2 = co2 * penalty
        final_nox = nox * penalty

        # --- E. ONE-LINE SUMMARY ---
        blend_summary = f"[H:{p_h:.2f} F:{p_f:.2f} A:{p_a:.2f}]"
        print(f"Trial {trial.number:03d}: SAF={saf_total*100:4.1f}% {blend_summary} | "
              f"TSFC={final_tsfc:5.2f} | Thrust={spec_thrust:5.1f} | "
              f"CO2={final_co2:6.0f} | NOx={final_nox:5.1f}")

        return final_tsfc, spec_thrust, final_co2, final_nox

    except optuna.TrialPruned: raise
    except Exception: return 1000.0, 0.0, 10000.0, 1000.0

# --- 5. RUNNER ---
study = optuna.create_study(directions=["minimize", "maximize", "minimize", "minimize"])
study.optimize(objective, n_trials=N_TRIALS)

# --- 6. EXTRACT & SAVE (UPDATED) ---
print("\n📊 Saving detailed results...")
trials = [t for t in study.best_trials if t.values[0] < 500]

# Initialize dictionary with ALL columns
results_data = {
    'TSFC': [], 'SpecThrust': [], 'CO2': [], 'NOx': [],
    'SAF_Total': [], 'Phi': [], 'LCA': [],
    'HEFA_Frac': [], 'FT_Frac': [], 'ATJ_Frac': []  # <--- ADDED COLUMNS
}

for t in trials:
    results_data['TSFC'].append(t.values[0])
    results_data['SpecThrust'].append(t.values[1])
    results_data['CO2'].append(t.values[2])
    results_data['NOx'].append(t.values[3])
    
    saf = t.params.get('saf_total', 0.0)
    results_data['SAF_Total'].append(saf)
    results_data['Phi'].append(t.params.get('phi', 0.5))

    # Reconstruct Blend Fractions
    w_h = t.params.get('w_hefa', 0); w_f = t.params.get('w_ft', 0); w_a = t.params.get('w_atj', 0)
    total_w = w_h + w_f + w_a + 1e-6
    
    p_h = saf * (w_h / total_w)
    p_f = saf * (w_f / total_w)
    p_a = saf * (w_a / total_w)
    
    # Save Blend Components
    results_data['HEFA_Frac'].append(p_h)
    results_data['FT_Frac'].append(p_f)
    results_data['ATJ_Frac'].append(p_a)

    jet_a = 1.0 - saf
    lca = jet_a*LCA_FACTORS['JetA'] + p_h*LCA_FACTORS['HEFA'] + p_f*LCA_FACTORS['FT'] + p_a*LCA_FACTORS['ATJ']
    results_data['LCA'].append(lca)

df_results = pd.DataFrame(results_data)
df_results.to_csv('optimization_results.csv', index=False)
print(f"✅ Saved 'optimization_results.csv' with full blend composition.")

# --- 7. VISUALIZATION (Standard) ---
print("📈 Generating plots...")
import os
os.makedirs('optimization_plots', exist_ok=True)

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_results['TSFC'], df_results['SpecThrust'], df_results['CO2'],
                c=df_results['NOx'], cmap='RdYlGn_r', s=60, edgecolors='k')
ax.set_xlabel('TSFC (mg/N·s)'); ax.set_ylabel('Spec Thrust'); ax.set_zlabel('CO2 (g/s)')
plt.colorbar(sc, label='NOx (g/s)')
plt.savefig('optimization_plots/pareto_3d.png', dpi=300)

# Parallel Coordinates
plt.figure(figsize=(12, 6))
norm_df = df_results[['TSFC','SpecThrust','CO2','NOx']].copy()
norm_df['SpecThrust'] = -norm_df['SpecThrust'] # Invert so lower is better for plot consistency
norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())
for i, r in norm_df.iterrows():
    plt.plot(range(4), r, color=plt.cm.viridis(df_results.loc[i, 'SAF_Total']*2), alpha=0.3)
plt.xticks(range(4), ['TSFC', 'Thrust(Inv)', 'CO2', 'NOx'])
plt.savefig('optimization_plots/parallel_coordinates.png', dpi=300)

print("✅ Optimization Complete.")