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

BASE_PI_C = 43.2      # Trent 1000-AE3 rated OPR (ICAO CSV, UID 02P23RR126)
BASE_AIRFLOW = 79.9   # Rated core mass flow assumption [kg/s] (hand-set, unsourced)
FPR_RATED = 1.45      # Rated fan pressure ratio (see simulation/fan.py sourcing)
# Cycle heat loss fraction: 0 by the structural argument in
# scripts/validation/heat_loss_provenance.md (liner heat recovered upstream of
# the turbine; casing loss has no engine-class source). Fixed, never fitted:
# fitting it would be a learnable factor absorbing model error (Reviewer 2, 4c).
# Note (same record, section 5): fuel flow is independent of eta_b and xi by
# construction; eta_combustor, pressure_loss and k_pi are inert in the fuel-flow
# objective, and k_mdot / phi_idle / phi_app lie on a ridge (two equations, three
# unknowns). It identifies phi_to alone (docs/plan.md Phase 5, finding F2).
HEAT_LOSS_XI = 0.0

# Targets: Trent 1000-AE3 (UID 02P23RR126), all traceable to data/icao_engine_data.csv.
# Power fractions are the CSV 'Power (%)' column. No CLIMB rows exist in the CSV.
ICAO_TARGETS = {
    'Idle':     {'power_fraction': 0.07, 'fuel_flow_kg_s': 0.244},
    'Approach': {'power_fraction': 0.30, 'fuel_flow_kg_s': 0.643},
    'Takeoff':  {'power_fraction': 1.00, 'fuel_flow_kg_s': 2.327},
}

# Search box of every sampled parameter, in Optuna suggest order (the order is
# part of the seeded calibration's reproducibility — do not reorder).
PARAM_BOUNDS = {
    # --- 1. PHYSICAL EFFICIENCY PARAMETERS ---
    "eta_combustor": (0.96, 0.999),
    "pressure_loss": (0.03, 0.06),
    # --- 2. PART-POWER THROTTLE EXPONENTS (replace 4 hand-set scales) ---
    "k_pi": (0.5, 1.5),
    "k_mdot": (0.3, 1.0),
    # --- 3. THROTTLE SETTINGS (equivalence ratio per mode) ---
    "phi_idle": (0.22, 0.30),
    "phi_app": (0.30, 0.40),
    "phi_to": (0.50, 0.60),
}
PHI_KEYS = {'Idle': 'phi_idle', 'Approach': 'phi_app', 'Takeoff': 'phi_to'}
CRASH_PENALTY = 1.0   # objective value (100 % error) of a crashed cycle evaluation


def set_mode_state(engine, params: dict, beta: float, xi: float, mode: str) -> None:
    """Set ``engine.design_point`` for one ICAO mode and parameter set (no cycle run)."""
    target = ICAO_TARGETS[mode]
    pi_c, m_dot = part_power_state(
        power_fraction=target['power_fraction'],
        pi_rated=BASE_PI_C,
        m_dot_rated=BASE_AIRFLOW,
        k_pi=params['k_pi'],
        k_mdot=params['k_mdot'],
    )
    engine.design_point['pi_c'] = pi_c
    engine.design_point['mass_flow_core'] = m_dot
    engine.design_point['combustor_pressure_loss'] = params['pressure_loss']
    engine.design_point['combustor_air_fraction'] = beta
    engine.design_point['combustor_heat_loss_fraction'] = xi
    # Fan pressure ratio follows the same throttle law as pi_c
    # (rated FPR at full power would demand rated fan work at idle)
    x = target['power_fraction']
    engine.design_point['fpr'] = 1.0 + (FPR_RATED - 1.0) * x ** params['k_pi']


def run_lto_modes(engine, params: dict, beta: float, xi: float = HEAT_LOSS_XI,
                  modes=None) -> dict:
    """Run the cycle at each ICAO mode (or only ``modes``) for one parameter set.

    Returns ``{mode: performance dict}`` (``run_full_cycle()['performance']``).
    """
    out = {}
    for mode in ICAO_TARGETS:
        if modes is not None and mode not in modes:
            continue
        set_mode_state(engine, params, beta, xi, mode)

        # Run Cycle
        res = engine.run_full_cycle(
            fuel_blend=FUEL_LIBRARY["Jet-A1"],
            phi=params[PHI_KEYS[mode]],
            combustor_efficiency=params['eta_combustor']
        )
        out[mode] = res['performance']
    return out


def fuel_flow_error(perf: dict) -> float:
    """v2-v4 objective: mean |relative fuel-flow error| over the three modes."""
    error_sum = 0.0
    for mode, target in ICAO_TARGETS.items():
        sim_ff = perf[mode]['fuel_mass_flow']
        error_sum += abs(sim_ff - target['fuel_flow_kg_s']) / target['fuel_flow_kg_s']
    return error_sum / len(ICAO_TARGETS)


OBJECTIVES = {"fuel_flow_v4": fuel_flow_error}


def objective_value(engine, params: dict, beta: float, objective: str = "fuel_flow_v4",
                    xi: float = HEAT_LOSS_XI) -> float:
    """Objective for one full parameter set; a crashed cycle scores CRASH_PENALTY."""
    try:
        return OBJECTIVES[objective](run_lto_modes(engine, params, beta, xi))
    except Exception:
        return CRASH_PENALTY


def main():
    parser = argparse.ArgumentParser(description="Seeded LTO calibration (part-power law)")
    parser.add_argument("--tag", default="v2",
                        help="Version tag for the output JSON (default 'v2')")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true",
                        help="--tag v5 only: registered pilot budget (P6.1 Step 4)")
    parser.add_argument("--free", nargs="*", default=None,
                        help="--tag v5 only: fitted parameters kept after the pilot profile")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Combustor air fraction (Phase 3.4): fraction of core "
                             "air burned at phi; sourced range 0.7-0.8 "
                             "(Lefebvre & Ballal). Default 1.0 = legacy")
    args = parser.parse_args()

    if args.tag == "v5":
        # Phase 6 thrust-matched calibration over the registered calibration
        # group (scripts/optimization/lto_v5.py; outputs/phase6/p61_registration.json)
        import lto_v5
        res = lto_v5.run_calibration(pilot=args.pilot, free=args.free)
        print(json.dumps({k: res[k] for k in ("stage", "free", "params", "sse",
                                               "calibration_weighted_mape_pct",
                                               "n_unreachable", "jtj_condition_box_scaled")},
                         indent=2, default=str))
        return

    # Frozen calibration records (v1-v4 and any later tag) are never overwritten.
    out_path = (Path(__file__).resolve().parent.parent.parent / "outputs" /
                f"calibration_trent1000_ae3_{args.tag}.json")
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing calibration record {out_path}; "
                         "choose a new --tag")

    print("======================================================================")
    print("🛠️  LTO CALIBRATION: PART-POWER THROTTLE MODEL")
    print("======================================================================")

    engine = IntegratedTurbofanEngine()
    N_TRIALS = args.n_trials
    SEED = args.seed
    error_history = []

    def objective(trial):
        params = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in PARAM_BOUNDS.items()}
        try:
            avg_error = fuel_flow_error(run_lto_modes(engine, params, args.beta))

            # --- CUSTOM LOGGING ---
            tn = trial.number
            if tn == 0 or tn == N_TRIALS - 1 or tn % 10 == 0:
                print(f"Trial {tn:02d}: Mean Error = {avg_error*100:.2f}% | "
                      f"k_pi={params['k_pi']:.3f} k_mdot={params['k_mdot']:.3f}")

            error_history.append(avg_error)

            return avg_error

        except Exception as e:
            print(f"  [Crash] Trial {trial.number} failed: {e}")
            return CRASH_PENALTY  # Heavy penalty (100% error) if it crashes

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
            "combustor_air_fraction": args.beta,
            "combustor_heat_loss_fraction": HEAT_LOSS_XI,   # fixed, not fitted (P4.5)
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
    with open(out_path, "x") as fh:
        json.dump(calibration_record, fh, indent=2)
    print(f"\n💾 Calibration record written to {out_path}")


if __name__ == "__main__":
    main()
