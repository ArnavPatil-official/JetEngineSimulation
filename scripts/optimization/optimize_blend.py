"""
Multi-objective optimization for jet engine fuel blends with emissions.

Phase 2.7 objective set (upgraded two-stream engine, v3 calibration):
  1. TSFC [mg/(N s)]                 minimize   (total thrust, both streams)
  2. Specific thrust [N s/kg]        maximize   (total thrust / total airflow)
  3. Lifecycle CO2e [g/s]            minimize   (CORSIA L_CEF triangular draws
                                                 from data/corsia_lca_values.yaml,
                                                 draw recorded per trial)
  4. NOx [g/s]                       minimize   (ICAO-derived correlation —
                                                 fuel-flow/OPR proxy, NOT
                                                 chemistry; see P2.4)

Combustion CO2 (chemistry EI, P2.3) is recorded per trial but is not an
objective: at fixed phi it tracks fuel flow, and mixing it with the lifecycle
axis is exactly the conflation V6 flagged.
"""

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import yaml
from mpl_toolkits.mplot3d import Axes3D
from integrated_engine import IntegratedTurbofanEngine, LocalFuelBlend
from simulation.fuels import make_saf_blend, JET_A1, HEFA_SPK, FT_SPK, ATJ_SPK
from optuna.trial import TrialState

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="4-objective SAF blend optimization")
parser.add_argument("--n-trials", type=int, default=1000,
                    help="Number of Optuna trials (default 1000)")
parser.add_argument("--seed", type=int, default=42,
                    help="Sampler seed for reproducibility (default 42)")
parser.add_argument("--output-csv", default="outputs/results/optimization_results.csv",
                    help="Where to write the per-trial results CSV")
parser.add_argument("--plots-dir", default="outputs/plots",
                    help="Directory for generated plots")
parser.add_argument("--freeze-phi", action="store_true",
                    help="AB8 variant: fix phi at the calibrated take-off value "
                         "so only blend fractions vary (separates operating-"
                         "point effects from fuel effects)")
parser.add_argument("--calibration",
                    default="outputs/calibration_trent1000_ae3_v3.json",
                    help="Frozen calibration JSON supplying eta_comb/p_loss/phi_to")
parser.add_argument("--turbine-model", choices=["analytic", "pinn"],
                    default="analytic",
                    help="Turbine component (P3.1 adjudicated production "
                         "default: analytic work-consistent polytropic)")
parser.add_argument("--nozzle-model", choices=["analytic", "pinn"],
                    default="analytic",
                    help="Nozzle component (P3.1 adjudicated production "
                         "default: analytic; LE-PINN failed Sajben)")
args = parser.parse_args()

N_TRIALS = args.n_trials
SEED = args.seed
TIT_HARD_LIMIT = 2800.0   # Solver-failure guard [K] (not a design constraint)
TIT_SOFT_LIMIT = 1850.0   # Cooling penalty threshold [K]

# Frozen calibration (Phase 2.1/2.2): eta_comb, pressure loss, phi_to
with open(PROJECT_ROOT / args.calibration) as _fh:
    CALIB = json.load(_fh)
ETA_COMB = CALIB["best_params"]["eta_combustor"]
P_LOSS = CALIB["best_params"]["pressure_loss"]
PHI_FROZEN = CALIB["best_params"]["phi_to"]

# CORSIA lifecycle values (Phase 2.3): triangular(min, mode, max) per pathway,
# gCO2e/MJ, cited to ICAO Document 06 (Nov 2025) table rows in the YAML.
with open(PROJECT_ROOT / "data" / "corsia_lca_values.yaml") as _fh:
    CORSIA = yaml.safe_load(_fh)
LCEF_BASELINE = CORSIA["baseline_fossil_gCO2e_MJ"]  # 89 gCO2e/MJ fossil jet
LCEF_TRI = {p: CORSIA["pathways"][p]["triangular"] for p in ("HEFA", "FT", "ATJ")}

# Component LHVs [MJ/kg] for energy-weighted lifecycle emissions
LHV = {"JetA": JET_A1.LHV_MJ_per_kg, "HEFA": HEFA_SPK.LHV_MJ_per_kg,
       "FT": FT_SPK.LHV_MJ_per_kg, "ATJ": ATJ_SPK.LHV_MJ_per_kg}

optuna.logging.set_verbosity(optuna.logging.ERROR)

print("="*80)
print("🚀 4-OBJECTIVE OPTIMIZATION: Performance + Environment (Phase 2.7)")
print(f"Targeting: {N_TRIALS} Trials (seed={SEED}, "
      f"phi {'FROZEN at %.4f' % PHI_FROZEN if args.freeze_phi else 'free'})")
print(f"Calibration: {args.calibration} (eta_comb={ETA_COMB:.4f}, p_loss={P_LOSS:.4f})")
print(f"Components: turbine={args.turbine_model}, nozzle={args.nozzle_model} "
      f"(P3.1 adjudicated)")
print("Output Format: Single-line summary per trial")
print("="*80 + "\n")


def draw_lcef(trial_number: int) -> dict:
    """Seeded triangular CORSIA draw per pathway; reproducible per trial."""
    rng = np.random.default_rng(SEED * 1_000_003 + trial_number)
    draws = {"JetA": LCEF_BASELINE}
    for p, tri in LCEF_TRI.items():
        draws[p] = float(rng.triangular(tri["min"], tri["mode"], tri["max"]))
    return draws

# --- 1. INITIALIZE ENGINE ---
engine = IntegratedTurbofanEngine(
    mechanism_profile="blends",
    creck_mechanism_path="data/creck_c1c16_full.yaml",
    hychem_mechanism_path="data/A1highT.yaml",
    turbine_pinn_path="models/turbine_pinn.pt",
    nozzle_pinn_path="models/nozzle_pinn.pt",
    icao_data_path="data/icao_engine_data.csv"
)
# Rated design point (take-off, x=1): pi_c 43.2 / FPR 1.45 defaults apply;
# combustor pressure loss and airflow split from the frozen calibration.
engine.design_point['combustor_pressure_loss'] = P_LOSS
engine.design_point['combustor_air_fraction'] = (
    CALIB['fixed_parameters'].get('combustor_air_fraction', 1.0))

# --- 2. FUEL WRAPPER ---
class SafeFuelWrapper:
    def __init__(self, name, species_dict):
        self.name = name; self.composition = species_dict
    def as_composition_string(self):
        return ", ".join([f"{k}:{v}" for k, v in self.composition.items()])
    def __repr__(self): return f"SafeFuelWrapper({self.name})"

def compute_blend_components(params):
    saf = params.get('saf_total', 0.0)
    jet_a = 1.0 - saf

    w_h = params.get('w_hefa', 0.0)
    w_f = params.get('w_ft', 0.0)
    w_a = params.get('w_atj', 0.0)

    total_w = w_h + w_f + w_a + 1e-6
    p_h = saf * (w_h / total_w)
    p_f = saf * (w_f / total_w)
    p_a = saf * (w_a / total_w)

    return saf, jet_a, p_h, p_f, p_a


def blend_lifecycle_g_s(m_dot_fuel, fractions, lcef_draws):
    """
    Lifecycle CO2e rate [g/s] for a blend under one CORSIA draw:
        sum_i p_i * LHV_i [MJ/kg] * L_CEF_i [gCO2e/MJ] * m_dot_fuel [kg/s]
    (energy-specific L_CEF applied per component; Doc 06 eq. 1 weighting)
    """
    jet_a, p_h, p_f, p_a = fractions
    per_kg = (jet_a * LHV["JetA"] * lcef_draws["JetA"] +
              p_h * LHV["HEFA"] * lcef_draws["HEFA"] +
              p_f * LHV["FT"] * lcef_draws["FT"] +
              p_a * LHV["ATJ"] * lcef_draws["ATJ"])
    return per_kg * m_dot_fuel


def identify_pareto_front(df, objectives, minimize_flags):
    """Return boolean mask for Pareto-optimal rows."""
    is_pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        for j in range(len(df)):
            if i == j:
                continue
            dominates = True
            strictly_better = False
            for k, obj in enumerate(objectives):
                a, b = df.iloc[i][obj], df.iloc[j][obj]
                if minimize_flags[k]:
                    if b > a:
                        dominates = False; break
                    elif b < a:
                        strictly_better = True
                else:
                    if b < a:
                        dominates = False; break
                    elif b > a:
                        strictly_better = True
            if dominates and strictly_better:
                is_pareto[i] = False
                break
    return is_pareto

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

    if args.freeze_phi:
        phi = PHI_FROZEN  # AB8 variant: operating point fixed, blends only
    else:
        phi = trial.suggest_float("phi", 0.35, 0.65)

    # CORSIA lifecycle draw for this trial (seeded, reproducible)
    lcef_draws = draw_lcef(trial.number)
    lcef_blend_energy = (
        (jet_a * LHV["JetA"] * lcef_draws["JetA"] + p_h * LHV["HEFA"] * lcef_draws["HEFA"] +
         p_f * LHV["FT"] * lcef_draws["FT"] + p_a * LHV["ATJ"] * lcef_draws["ATJ"]) /
        (jet_a * LHV["JetA"] + p_h * LHV["HEFA"] + p_f * LHV["FT"] + p_a * LHV["ATJ"])
    )

    # --- B. Simulation ---
    try:
        raw_blend = make_saf_blend(jet_a, p_h, p_f, p_a, enforce_astm=True)
        fuel = SafeFuelWrapper(f"Trial_{trial.number}", raw_blend.species)

        # Suppress engine console noise only — all data is read from the
        # structured result dict below; nothing is parsed from printed output.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = engine.run_full_cycle(
                fuel_blend=fuel, phi=phi, combustor_efficiency=ETA_COMB,
                turbine_model=args.turbine_model,
                nozzle_model=args.nozzle_model,
            )

        # --- C. Data Extraction (structured results only; KeyError = failed trial) ---
        tsfc = result['performance']['tsfc_mg_per_Ns']
        thrust = result['performance']['thrust_kN']
        spec_thrust_raw = result['performance']['specific_thrust_Ns_kg']
        m_dot_fuel = result['performance']['fuel_mass_flow']
        nox = result['emissions']['NOx_g_s']  # ICAO-derived correlation path
        co2_combustion = result['emissions']['CO2_combustion_g_s']
        lifecycle = blend_lifecycle_g_s(m_dot_fuel, (jet_a, p_h, p_f, p_a), lcef_draws)
        t4 = result['combustor']['T_out']

        # --- D. Constraints & Penalties ---
        if t4 > TIT_HARD_LIMIT:
            print(f"Trial {trial.number:03d}: ❌ PRUNED (T4={t4:.0f}K > Limit)")
            raise optuna.TrialPruned()

        penalty = 1.0 + max(0.0, (t4 - TIT_SOFT_LIMIT) * 0.0005)

        final_tsfc = tsfc * penalty
        spec_thrust = spec_thrust_raw / penalty
        final_lifecycle = lifecycle * penalty
        final_nox = nox * penalty

        # Record raw (pre-penalty) values so reported objectives are traceable
        # to structured engine outputs, plus the CORSIA draw itself.
        trial.set_user_attr("raw_tsfc_mg_per_Ns", tsfc)
        trial.set_user_attr("raw_thrust_kN", thrust)
        trial.set_user_attr("raw_spec_thrust_Ns_kg", spec_thrust_raw)
        trial.set_user_attr("raw_T4_K", t4)
        trial.set_user_attr("raw_NOx_g_s", nox)
        trial.set_user_attr("raw_CO2_combustion_g_s", co2_combustion)
        trial.set_user_attr("raw_Lifecycle_CO2e_g_s", lifecycle)
        trial.set_user_attr("fuel_mass_flow_kg_s", m_dot_fuel)
        trial.set_user_attr("phi_used", phi)
        trial.set_user_attr("lcef_blend_gCO2e_MJ", lcef_blend_energy)
        for pathway, v in lcef_draws.items():
            trial.set_user_attr(f"lcef_draw_{pathway}", v)
        trial.set_user_attr("tit_penalty_factor", penalty)

        # --- E. ONE-LINE SUMMARY ---
        blend_summary = f"[H:{p_h:.2f} F:{p_f:.2f} A:{p_a:.2f}]"
        print(f"Trial {trial.number:03d}: SAF={saf_total*100:4.1f}% {blend_summary} | "
              f"TSFC={final_tsfc:5.2f} | SpThr={spec_thrust:5.1f} | "
              f"LC-CO2e={final_lifecycle:7.0f} | NOx={final_nox:5.1f}")

        return final_tsfc, spec_thrust, final_lifecycle, final_nox

    except optuna.TrialPruned:
        raise
    except Exception as e:
        # No default values are ever substituted: the trial fails loudly and is
        # recorded as FAIL by Optuna (see catch= in study.optimize below).
        print(f"Trial {trial.number:03d}: ❌ FAILED ({type(e).__name__}: {e})")
        raise

# --- 5. RUNNER ---
study = optuna.create_study(
    directions=["minimize", "maximize", "minimize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=SEED),
)
# catch=(Exception,): crashed trials are recorded as FAIL and excluded from
# results — they never contribute default/penalty objective values.
study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

n_failed = len([t for t in study.trials if t.state == TrialState.FAIL])
n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
print(f"\nTrial states: {len(study.trials)} total | "
      f"{n_failed} failed | {n_pruned} pruned")

# --- 6. EXTRACT & SAVE (ALL TRIALS + PARETO FLAG) ---
print("\n📊 Saving detailed results for all completed trials...")
completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values]

rows = []
for t in completed_trials:
    tsfc, spec_thrust, lifecycle, nox = t.values
    saf, jet_a, p_h, p_f, p_a = compute_blend_components(t.params)

    rows.append({
        'Trial': t.number,
        'TSFC': tsfc,
        'SpecThrust': spec_thrust,
        'Lifecycle_CO2e': lifecycle,
        'NOx_correlation': nox,
        'SAF_Total': saf,
        'JetA_Frac': jet_a,
        'Phi': t.user_attrs.get('phi_used'),
        'HEFA_Frac': p_h,
        'FT_Frac': p_f,
        'ATJ_Frac': p_a,
        'State': t.state.name,
        # Raw structured engine outputs (pre-TIT-penalty) for traceability
        'Raw_TSFC': t.user_attrs.get('raw_tsfc_mg_per_Ns'),
        'Raw_Thrust_kN': t.user_attrs.get('raw_thrust_kN'),
        'Raw_SpecThrust_Ns_kg': t.user_attrs.get('raw_spec_thrust_Ns_kg'),
        'Raw_T4_K': t.user_attrs.get('raw_T4_K'),
        'Raw_NOx_g_s': t.user_attrs.get('raw_NOx_g_s'),
        'Raw_CO2_combustion_g_s': t.user_attrs.get('raw_CO2_combustion_g_s'),
        'Raw_Lifecycle_CO2e_g_s': t.user_attrs.get('raw_Lifecycle_CO2e_g_s'),
        'Fuel_Flow_kg_s': t.user_attrs.get('fuel_mass_flow_kg_s'),
        'LCEF_blend_gCO2e_MJ': t.user_attrs.get('lcef_blend_gCO2e_MJ'),
        'LCEF_draw_HEFA': t.user_attrs.get('lcef_draw_HEFA'),
        'LCEF_draw_FT': t.user_attrs.get('lcef_draw_FT'),
        'LCEF_draw_ATJ': t.user_attrs.get('lcef_draw_ATJ'),
        'TIT_Penalty': t.user_attrs.get('tit_penalty_factor'),
    })

df_results = pd.DataFrame(rows)

# Pareto analysis across all completed trials
objective_cols = ['TSFC', 'SpecThrust', 'Lifecycle_CO2e', 'NOx_correlation']
pareto_mask = identify_pareto_front(df_results, objective_cols, [True, False, True, True])
df_results['ParetoOptimal'] = pareto_mask

os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
df_results.to_csv(args.output_csv, index=False)
print(f"✅ Saved {len(df_results)} rows to '{args.output_csv}' with ParetoOptimal flag ({pareto_mask.sum()} Pareto points).")

# --- 7. VISUALIZATION (Standard) ---
print("📈 Generating plots...")
os.makedirs(args.plots_dir, exist_ok=True)

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_results['TSFC'], df_results['SpecThrust'], df_results['Lifecycle_CO2e'],
                c=df_results['NOx_correlation'], cmap='RdYlGn_r', s=60, edgecolors='k')
ax.set_xlabel('TSFC (mg/N·s)'); ax.set_ylabel('Spec Thrust (N·s/kg)')
ax.set_zlabel('Lifecycle CO2e (g/s, CORSIA draw)')
plt.colorbar(sc, label='NOx (g/s, ICAO-derived correlation)')
plt.savefig(os.path.join(args.plots_dir, 'pareto_3d.png'), dpi=300)

# Parallel Coordinates
plt.figure(figsize=(12, 6))
norm_df = df_results[['TSFC','SpecThrust','Lifecycle_CO2e','NOx_correlation']].copy()
norm_df['SpecThrust'] = -norm_df['SpecThrust'] # Invert so lower is better for plot consistency
norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())
for i, r in norm_df.iterrows():
    plt.plot(range(4), r, color=plt.cm.viridis(df_results.loc[i, 'SAF_Total']*2), alpha=0.3)
plt.xticks(range(4), ['TSFC', 'Thrust(Inv)', 'Lifecycle CO2e', 'NOx (corr.)'])
plt.savefig(os.path.join(args.plots_dir, 'parallel_coordinates.png'), dpi=300)

print("✅ Optimization Complete.")