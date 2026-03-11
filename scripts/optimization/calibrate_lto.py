import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import optuna
import numpy as np
import logging
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

# Suppress messy logs, only show our custom prints
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("cantera").setLevel(logging.ERROR)

print("======================================================================")
print("🛠️  LTO CALIBRATION: ENGINE TUNING (SCALED AIRFLOW)")
print("======================================================================")

engine = IntegratedTurbofanEngine()
N_TRIALS = 50

# Targets: Trent 1000-AE3 (Fuel Flow kg/s)
ICAO_TARGETS = {
    'Idle': 0.244, 
    'Approach': 0.643, 
    'Climb': 2.050, 
    'Takeoff': 2.327
}
error_history = []

def objective(trial):
    # --- 1. PHYSICAL EFFICIENCY PARAMETERS ---
    eta_b = trial.suggest_float("eta_combustor", 0.96, 0.999)
    p_loss = trial.suggest_float("pressure_loss", 0.03, 0.06)
    
    # --- 2. THROTTLE SETTINGS ---
    phi_idle = trial.suggest_float("phi_idle", 0.22, 0.30)
    phi_app  = trial.suggest_float("phi_app",  0.30, 0.40)
    phi_climb= trial.suggest_float("phi_climb",0.42, 0.50)
    phi_to   = trial.suggest_float("phi_to",   0.50, 0.60)

    phis = {'Idle': phi_idle, 'Approach': phi_app, 'Climb': phi_climb, 'Takeoff': phi_to}
    
    error_sum = 0.0
    
    try:
        for mode, target_ff in ICAO_TARGETS.items():
            # 1. Reset Pressure Ratio base
            base_pi_c = 43.2
            
            # 2. Scale Airflow & Pressure for each mode
            if mode == 'Idle':
                scale = 0.15
                pi_scale = 0.15
            elif mode == 'Approach':
                scale = 0.35
                pi_scale = 0.40
            elif mode == 'Climb':
                scale = 0.85
                pi_scale = 0.90
            else: # Takeoff
                scale = 1.0
                pi_scale = 1.0

            # Apply Scales
            engine.design_point['pi_c'] = base_pi_c * pi_scale
            engine.design_point['mass_flow_core'] = 79.9 * scale 

            # Run Cycle
            res = engine.run_full_cycle(
                fuel_blend=FUEL_LIBRARY["Jet-A1"], 
                phi=phis[mode], 
                combustor_efficiency=eta_b
            )
            
            # 3. Calculate Error (The part needed to make it work!)
            sim_ff = res['performance'].get('fuel_flow', res['performance'].get('fuel_mass_flow'))
            error_sum += abs(sim_ff - target_ff) / target_ff

        # Success! Return average error
        avg_error = error_sum / 4.0

        # --- CUSTOM LOGGING ---
        tn = trial.number
        if tn == 0 or tn == N_TRIALS - 1 or tn % 10 == 0:
            print(f"Trial {tn:02d}: Mean Error = {avg_error*100:.2f}% | Idle Phi={phi_idle:.3f}")

        error_history.append(avg_error)

        return avg_error

    except Exception as e:
        print(f"  [Crash] Trial {trial.number} failed: {e}")
        return 1.0 # Heavy penalty (100% error) if it crashes

# Run Optimization
study = optuna.create_study(direction="minimize")
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