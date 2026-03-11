# Skill: Implement ML Change

Use this skill when the plan involves modifying PINN architecture, loss functions, training configs, or surrogate model behavior.

## Pre-flight

1. Read `docs/plan.md` fully.
2. Identify which PINN is affected: nozzle (`models/nozzle_pinn.pt`) or turbine (`models/turbine_pinn.pt`).
3. Check the current model definition in `simulation/nozzle/nozzle.py` or `simulation/turbine/turbine.py`.

## Steps

1. **Back up existing weights** before any training change:
   ```bash
   cp models/<name>_pinn.pt models/<name>_pinn.pt.bak
   ```

2. **Modify architecture/loss** as specified in the plan. Keep changes localized to the PINN class.

3. **Preserve reproducibility**:
   - Keep `torch.manual_seed()` calls.
   - Don't change batch sizes or learning rates unless the plan says to.
   - Log any config changes to stdout.

4. **Validate**:
   ```bash
   python -m pytest tests/ -v
   python scripts/test_emissions.py  # if combustor/emissions touched
   ```

5. **Check physics constraints** still hold — boundary conditions, conservation laws, monotonicity in nozzle profiles.

## Pitfalls

- Changing hidden layer sizes without retraining will produce garbage.
- `nozzle.py` has both analytical and PINN paths — make sure you're editing the right one.
- Loss weighting between data loss and physics loss is sensitive. Small changes can break convergence.
