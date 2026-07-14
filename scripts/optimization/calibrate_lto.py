"""
One-time seeded LTO fuel-flow calibration against the Trent 1000-AE3 (Phase 2.1).

Phase 2 replaces the Phase 1 hand-set per-mode scales with a part-power
throttle law (see integrated_engine.part_power_state):

    pi_c(x)  = 1 + (pi_rated - 1) * x^k_pi
    m_dot(x) = m_dot_rated * x^k_mdot,   x = ICAO power setting (F/F00)

and wires the combustor pressure loss for real
(design_point['combustor_pressure_loss'] -> p_comb = p3 * (1 - p_loss)).

The untraceable Climb target (2.050 kg/s) is gone: data/icao_engine_data.csv
contains no CLIMB rows (Phase 1 finding), so calibration uses the three modes
present in the certification record — Idle (7%), Approach (30%), Take-off
(100%).

Output: outputs/calibration_trent1000_ae3_<tag>.json (default tag "v2";
Phase 1's calibration_trent1000_ae3.json is never overwritten).
"""

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import optuna
import numpy as np
import logging
from datetime import date
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY, part_power_state

# Suppress messy logs, only show our custom prints
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("cantera").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description="Seeded LTO calibration (part-power law)")
parser.add_argument("--tag", default="v2",
                    help="Version tag for the output JSON (default 'v2')")
parser.add_argument("--n-trials", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print("======================================================================")
print("🛠️  LTO CALIBRATION: PART-POWER THROTTLE MODEL")
print("======================================================================")

engine = IntegratedTurbofanEngine()
N_TRIALS = args.n_trials
SEED = args.seed

BASE_PI_C = 43.2      # Trent 1000-AE3 rated OPR (ICAO CSV, UID 02P23RR126)
BASE_AIRFLOW = 79.9   # Rated core mass flow assumption [kg/s] (hand-set, unsourced)
FPR_RATED = 1.45      # Rated fan pressure ratio (see simulation/fan.py sourcing)

# Targets: Trent 1000-AE3 (UID 02P23RR126), all traceable to data/icao_engine_data.csv.
# Power fractions are the CSV 'Power (%)' column. No CLIMB rows exist in the CSV.
ICAO_TARGETS = {
    'Idle':     {'power_fraction': 0.07, 'fuel_flow_kg_s': 0.244},
    'Approach': {'power_fraction': 0.30, 'fuel_flow_kg_s': 0.643},
    'Takeoff':  {'power_fraction': 1.00, 'fuel_flow_kg_s': 2.327},
}
error_history = []

def objective(trial):
    # --- 1. PHYSICAL EFFICIENCY PARAMETERS ---
    eta_b = trial.suggest_float("eta_combustor", 0.96, 0.999)
    p_loss = trial.suggest_float("pressure_loss", 0.03, 0.06)

    # --- 2. PART-POWER THROTTLE EXPONENTS (replace 4 hand-set scales) ---
    k_pi = trial.suggest_float("k_pi", 0.5, 1.5)
    k_mdot = trial.suggest_float("k_mdot", 0.3, 1.0)

    # --- 3. THROTTLE SETTINGS (equivalence ratio per mode) ---
    phi_idle = trial.suggest_float("phi_idle", 0.22, 0.30)
    phi_app  = trial.suggest_float("phi_app",  0.30, 0.40)
    phi_to   = trial.suggest_float("phi_to",   0.50, 0.60)

    phis = {'Idle': phi_idle, 'Approach': phi_app, 'Takeoff': phi_to}

    error_sum = 0.0

    try:
        for mode, target in ICAO_TARGETS.items():
            pi_c, m_dot = part_power_state(
                power_fraction=target['power_fraction'],
                pi_rated=BASE_PI_C,
                m_dot_rated=BASE_AIRFLOW,
                k_pi=k_pi,
                k_mdot=k_mdot,
            )
            engine.design_point['pi_c'] = pi_c
            engine.design_point['mass_flow_core'] = m_dot
            engine.design_point['combustor_pressure_loss'] = p_loss
            # Fan pressure ratio follows the same throttle law as pi_c
            # (rated FPR at full power would demand rated fan work at idle)
            x = target['power_fraction']
            engine.design_point['fpr'] = 1.0 + (FPR_RATED - 1.0) * x ** k_pi

            # Run Cycle
            res = engine.run_full_cycle(
                fuel_blend=FUEL_LIBRARY["Jet-A1"],
                phi=phis[mode],
                combustor_efficiency=eta_b
            )

            sim_ff = res['performance']['fuel_mass_flow']
            error_sum += abs(sim_ff - target['fuel_flow_kg_s']) / target['fuel_flow_kg_s']

        # Success! Return average error
        avg_error = error_sum / len(ICAO_TARGETS)

        # --- CUSTOM LOGGING ---
        tn = trial.number
        if tn == 0 or tn == N_TRIALS - 1 or tn % 10 == 0:
            print(f"Trial {tn:02d}: Mean Error = {avg_error*100:.2f}% | "
                  f"k_pi={k_pi:.3f} k_mdot={k_mdot:.3f}")

        error_history.append(avg_error)

        return avg_error

    except Exception as e:
        print(f"  [Crash] Trial {trial.number} failed: {e}")
        return 1.0 # Heavy penalty (100% error) if it crashes

# Run Optimization (seeded sampler for reproducible calibration)
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
print(f"\n🧠 Calibration started ({N_TRIALS} trials)...")
study.optimize(objective, n_trials=N_TRIALS)
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(np.array(error_history) * 100, marker='o', linewidth=1)
plt.xlabel("Trial")
plt.ylabel("Mean LTO Fuel Flow Error (%)")
plt.title("ICAO LTO Calibration Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output Results
print("\n✅ CALIBRATED SETTINGS (COPY THESE TO CRUISE OPTIMIZER):")
print(f"  Best Error: {study.best_value*100:.2f}%")
for k, v in study.best_params.items():
    print(f"  {k}: {v:.4f}")

# Report the part-power states implied by the best trial (traceability)
best = study.best_params
per_mode_states = {}
for mode, target in ICAO_TARGETS.items():
    pi_c, m_dot = part_power_state(
        target['power_fraction'], BASE_PI_C, BASE_AIRFLOW,
        best['k_pi'], best['k_mdot'],
    )
    per_mode_states[mode] = {
        'power_fraction': target['power_fraction'],
        'pi_c': pi_c,
        'mass_flow_core_kg_s': m_dot,
    }
    print(f"  {mode:<9} x={target['power_fraction']:.2f} -> "
          f"pi_c={pi_c:.2f}, m_dot={m_dot:.1f} kg/s")

# Persist calibrated parameters for held-out validation
# (scripts/validation/holdout_icao_validation.py loads this frozen JSON)
calibration_record = {
    "description": "One-time Optuna calibration of LTO fuel flow against ICAO "
                   "certification data for the Trent 1000-AE3 (in-sample fit, "
                   "not validation). Part-power throttle law version.",
    "schema": "part_power_v2",
    "calibration_engine": "Trent 1000-AE3",
    "icao_uid": "02P23RR126",
    "date": date.today().isoformat(),
    "seed": SEED,
    "n_trials": N_TRIALS,
    "fuel": "Jet-A1",
    "icao_targets": ICAO_TARGETS,
    "best_mean_abs_pct_error": study.best_value,
    "best_params": study.best_params,
    "per_mode_states": per_mode_states,
    "fixed_parameters": {
        "base_pi_c": BASE_PI_C,
        "base_airflow_kg_s": BASE_AIRFLOW,
        "fpr_rated": FPR_RATED,
        "eta_fan": 0.90,
        "eta_compressor": 0.86,
        "eta_turbine_polytropic": 0.9,
    },
    "notes": [
        "Part-power law: pi_c = 1 + (pi_rated - 1) * x^k_pi; "
        "m_dot = m_dot_rated * x^k_mdot; x = ICAO power setting.",
        "pressure_loss is now consumed: "
        "design_point['combustor_pressure_loss'] applies "
        "p_comb = p3 * (1 - p_loss) in run_full_cycle.",
        "Climb is excluded: data/icao_engine_data.csv has no CLIMB rows; the "
        "former 2.050 kg/s climb target was untraceable and has been removed.",
    ],
}
out_path = (Path(__file__).resolve().parent.parent.parent / "outputs" /
            f"calibration_trent1000_ae3_{args.tag}.json")
with open(out_path, "w") as fh:
    json.dump(calibration_record, fh, indent=2)
print(f"\n💾 Calibration record written to {out_path}")
