# Actual-Data Additions Required

## Purpose
This document lists additions needed to make visualization outputs fully data-backed when actual datasets were not used and proxies/heuristics were used instead.

Scope reviewed: `scripts/visualization/visualize_results.py`.

## Summary of Gaps
The current visualization overhaul runs end-to-end, but several figures use one or more of:
- Proxy losses derived from cycle metrics
- Fixed +/- percent uncertainty bands
- Synthetic benchmark targets
- Simplified combustor/species surrogates
- Hard-coded validation simulation values

## Additions Needed (By Figure)

### Figure 01: PINN Curriculum Loss Dashboard
Current fallback: proxy losses inferred from cycle log fields when explicit training history is missing.

Additions needed:
- Persist true training history (epoch, BC/EOS/work/monotonic losses, LR, phase tags) for nozzle/turbine/LE-PINN training.
- Export history to structured files at train time and load those files in plotting.

Suggested files:
- `simulation/turbine/turbine.py` (persist training diagnostics)
- `simulation/nozzle/nozzle.py` (persist training diagnostics)
- `simulation/nozzle/le_pinn.py` (persist training diagnostics)
- `outputs/results/visualization/pinn_loss_history_{model}.csv` (new)

### Figure 02: Flow Profiles with Uncertainty Bands
Current fallback: fixed +/-5% envelope around single-model predictions.

Additions needed:
- Generate uncertainty from repeated runs (multi-seed retraining and/or ensemble checkpoints).
- Store per-x mean/std or quantiles for each state variable.

Suggested files:
- `scripts/visualization/build_visualization_data.py` (new export builder)
- `outputs/results/visualization/profile_uncertainty_{model}.csv` (new)

### Figure 03: LE-PINN Benchmark Comparison
Current fallback: uses generic guide targets (5%, 5%, 2%).

Additions needed:
- Create benchmark dataset from actual evaluation runs (single-network PINN vs LE-PINN on same test set).
- Include metric definitions and sample counts.

Suggested files:
- `scripts/visualization/build_visualization_data.py`
- `outputs/results/visualization/lepinn_benchmark_metrics.csv` (new)

### Figure 04: Nozzle Centerline vs Isentropic
Current fallback: simplified pressure and area trends are hard-coded.

Additions needed:
- Compute analytic isentropic reference from actual geometry, area distribution, inlet/exit constraints, and mass flow.
- Store centerline comparison data (model, reference, residual).

Suggested files:
- `simulation/nozzle/nozzle_conditions.py` (or new helper for analytic reference)
- `outputs/results/visualization/nozzle_centerline_reference.csv` (new)

### Figure 06: Combustor Temperature vs Time
Current fallback: synthetic exponential response curves by fuel group.

Additions needed:
- Add diagnostics-only reactor transient path (Cantera constant-pressure reactor) to produce real `T(t)` traces.
- Export traces across fuel groups and ICAO power settings.

Suggested files:
- `simulation/combustor/combustor.py` or `simulation/combustor/diagnostics.py` (new helper)
- `outputs/results/visualization/combustor_transients.csv` (new)

### Figure 07: Species Heatmap
Current fallback: composite intensity proxy from CO2/NOx/TSFC and mode scaling factors.

Additions needed:
- Export actual species mass fractions (for selected species) by fuel group and ICAO mode from combustor outputs/reactor results.
- Define explicit species list and normalization approach.

Suggested files:
- `outputs/results/visualization/species_matrix.csv` (new)

### Figure 09: BO Convergence with Uncertainty
Current fallback: uncertainty from rolling std of a single run history.

Additions needed:
- Run optimization with repeated seeds and aggregate trial-best trajectories across runs.
- Export per-trial mean/std or confidence intervals.

Suggested files:
- `scripts/optimization/optimize_blend.py` (seed-aware export)
- `outputs/results/visualization/bo_convergence_aggregate.csv` (new)

### Figure 12: Engine State Waterfall
Current fallback: error bars approximated with a global heuristic scale.

Additions needed:
- Use uncertainty from repeated full-cycle runs or sensitivity sweeps (phi, efficiency, fuel property perturbation).
- Export component-level uncertainty per state variable.

Suggested files:
- `outputs/results/visualization/engine_state_summary.csv` (new)

### Figure 13: ICAO Validation Subplots
Current fallback: simulation fuel flow vector is hard-coded.

Additions needed:
- Generate simulation values directly from validation runs at IDLE/APPROACH/CLIMB/TAKEOFF.
- Store paired reference vs simulation values plus tolerance bands and run metadata.

Suggested files:
- `scripts/visualization/build_visualization_data.py`
- `outputs/results/visualization/icao_validation_dataset.csv` (new)

## Cross-Cutting Additions
- Add one canonical data builder script:
  - `scripts/visualization/build_visualization_data.py`
- Add a stable output contract folder:
  - `outputs/results/visualization/`
- Require plot functions to prefer structured exports over log parsing and synthetic fallbacks.
- Add provenance metadata (`generated_at`, git commit hash if available, model/checkpoint IDs, seed, mechanism profile).

## Minimal Data Contract (Recommended)
Each generated CSV should include:
- `source` (module/function that produced row)
- `trial` or `run_id`
- `seed`
- `fuel_group` and/or blend fractions
- `mode` (for ICAO/LTO data)
- measured fields for plotting
- uncertainty fields (`std`, `ci_low`, `ci_high`) only when computed from repeated data

## Priority Order
1. Persist true training history (Figure 01 blocker)
2. Build validation dataset (Figure 13 blocker)
3. Add combustor transients + species export (Figures 06-07 blocker)
4. Add uncertainty aggregation from repeated runs (Figures 02, 09, 12 quality blocker)
5. Replace benchmark and centerline synthetic references with computed/validated datasets (Figures 03-04 quality blocker)

## Acceptance Check
The visuals are "actual-data-backed" when all of the following are true:
- No hard-coded simulation vectors are used in plots.
- No fixed uncertainty percentage bands are used unless explicitly specified as a requirement band.
- Every plotted series traces to a saved structured dataset in `outputs/results/visualization/`.
- Figure captions can cite the exact data file and generation method.
