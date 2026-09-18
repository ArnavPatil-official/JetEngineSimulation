# LE-PINN Stabilization + Sajben Data Path Plan — May 6, 2026

## Objective
Diagnose and fix LE-PINN training instability (physics-loss domination, LR collapse, data-loss stagnation) and guarantee that Sajben training uses the correct processed Sajben dataset (`data/processed/master_shock_dataset.pt`) rather than synthetic-only data.

## Constraints
- Preserve existing trained weights under `models/*.pt`; do not overwrite without explicit save targets.
- Keep chemical YAML files (`data/*.yaml`) unchanged.
- Maintain reproducibility (fixed seeds, deterministic data paths, explicit config metadata in checkpoints).
- Keep axisymmetric nozzle workflow intact while improving Sajben (planar) workflow.
- Pass repo tests: `python3 -m pytest tests/ -v`.

## Repo Context
- `simulation/nozzle/le_pinn.py`: core LE-PINN architecture, physics residuals, synthetic training, CFD fine-tuning.
- `scripts/parse_sajben_cfd.py`: builds Sajben-derived master dataset.
- `scripts/validation/train_sajben.py`: intended Sajben training entrypoint.
- `scripts/validation/finetune_sajben.py`: Sajben fine-tuning workflow on `master_shock_dataset.pt`.

## Relevant Files
- Modify: `simulation/nozzle/le_pinn.py`
- Modify: `scripts/validation/train_sajben.py`
- Modify: `scripts/validation/finetune_sajben.py`
- Modify: `tests/test_le_pinn.py`
- Create: `tests/test_sajben_training_path.py`
- Read/verify only: `scripts/parse_sajben_cfd.py`, `data/processed/master_shock_dataset.pt`

## Implementation Phases

### Phase 1 — Lock Correct Sajben Data Path
1. Ensure Sajben training entrypoint uses processed Sajben dataset (not synthetic-only training loop).
2. Add explicit dataset existence and schema checks before training starts.
3. Print resolved dataset path and sample stats in startup logs for traceability.

### Phase 2 — Stabilize Physics Loss Magnitude
1. Add residual scaling (dimensionless normalization) before physics-loss aggregation:
   - mass, x-momentum, y-momentum, energy, EOS scaled by physically meaningful reference scales.
2. Keep geometry mode explicit (`axisymmetric` vs `planar`) and enforce `planar` for Sajben fine-tuning paths.
3. Add per-term debug telemetry (optional flag) so training prints residual term magnitudes.

### Phase 3 — Fix Scheduler Signal
1. Stop stepping `ReduceLROnPlateau` on adaptively weighted total loss (which drifts by design as λ changes).
2. Step scheduler on a stable monitor metric (unweighted composite or data-only metric) to prevent premature LR collapse.
3. Retain min LR floor but avoid immediate descent to `1e-8` in first few hundred epochs.

### Phase 4 — Verification + Guardrails
1. Add tests that assert Sajben scripts resolve `master_shock_dataset.pt` and fail fast with clear errors when missing/invalid.
2. Add tests for physics-loss scaling utility (finite outputs, reasonable order of magnitude).
3. Run smoke training checks (few epochs) to verify:
   - data loss decreases from initial value,
   - LR does not collapse to min immediately,
   - physics loss remains finite.

## File-Level Edits
- `simulation/nozzle/le_pinn.py`
  - Add helper for reference-scale computation and normalized physics loss aggregation.
  - Update `train_le_pinn` and `finetune_on_cfd_data` to use normalized physics loss.
  - Change scheduler monitor from weighted total loss to stable monitor.
  - Add optional verbose diagnostics for residual components and monitor metric.
- `scripts/validation/train_sajben.py`
  - Repoint workflow to dataset-backed training path (using `finetune_on_cfd_data` with `geometry="planar"`) or explicitly label synthetic pretraining mode.
  - Add startup checks/logs proving dataset source.
- `scripts/validation/finetune_sajben.py`
  - Enforce dataset key checks (`inputs`, `targets`, optional `sample_weights`) and print summary.
- `tests/test_le_pinn.py`
  - Extend with scheduler-monitor behavior and scaled-physics-loss sanity checks.
- `tests/test_sajben_training_path.py` (new)
  - Validate Sajben script resolves processed dataset and uses planar geometry mode.

## Commands to Run
1. `python3 scripts/parse_sajben_cfd.py`
2. `python3 -m pytest tests/test_le_pinn.py -v`
3. `python3 -m pytest tests/test_sajben_training_path.py -v`
4. `python3 -m pytest tests/ -v`
5. `python3 scripts/validation/finetune_sajben.py --epochs 20 --device cpu --physics-weight 0.05 --physics-max-points 2048`

## Tests
- Unit tests:
  - physics-loss scaling returns finite scalar and bounded per-term magnitudes,
  - scheduler monitor path works and does not error with adaptive λ weighting,
  - Sajben dataset path validation catches missing keys/files.
- Integration smoke:
  - short Sajben fine-tune run completes,
  - checkpoint saves with config and seed,
  - losses logged with expected trends.

## Acceptance Criteria
- Training no longer shows immediate LR collapse to `~1e-8` by ~epoch 200 under default settings.
- Data loss decreases measurably during short smoke runs (`final_data_loss < initial_data_loss`).
- Physics loss remains finite and dimensionless-scaled (no `~1e12` to `~1e15` raw dominance in final aggregated physics term).
- Sajben training scripts explicitly use `data/processed/master_shock_dataset.pt` and `geometry="planar"`.
- `python3 -m pytest tests/ -v` passes.
- No unplanned file changes beyond scoped files.

## Rollback Notes
- Revert modified files with:
  - `git checkout -- simulation/nozzle/le_pinn.py scripts/validation/train_sajben.py scripts/validation/finetune_sajben.py tests/test_le_pinn.py tests/test_sajben_training_path.py`
- If dataset regeneration needs rollback:
  - restore `data/processed/master_shock_dataset.pt` from git-tracked baseline or backup artifact.

## Escalation Guidance
- Complexity: **High** (multi-file training behavior changes + numerical stability + tests).
- Risk: **Moderate** (affects optimization dynamics, requires smoke validation).
- Recommended executor model: **Claude Opus** via dispatcher for robust multi-step code+test iteration.
