# Skill: Debug Training Failure

Use this skill when a PINN training run produces NaN losses, diverges, or fails to converge.

## Diagnostic Checklist

1. **Check loss values**: Look for NaN/Inf in training logs (`outputs/results/full_logs.txt`).
2. **Check input data**: Verify Cantera mechanism files (`data/*.yaml`) load without errors.
3. **Check normalization**: PINN inputs must be normalized. Verify scaling constants haven't drifted.
4. **Check learning rate**: If recently changed, try reverting.
5. **Check physics constraints**: Overly strict physics losses can fight data losses → NaN.

## Common Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| NaN after epoch 1 | Learning rate too high | Reduce by 10x |
| Loss plateaus high | Physics loss too dominant | Reduce physics weight |
| Oscillating loss | Batch size too small | Increase or add gradient clipping |
| Monotonicity violated | Missing boundary constraint | Re-check `nozzle_conditions.py` |

## Steps

1. Reproduce the failure: run the training script with current config.
2. Add `torch.autograd.set_detect_anomaly(True)` temporarily.
3. Identify the first NaN source.
4. Apply fix from table above or as specified in plan.
5. Retrain and verify convergence.
6. Remove debug flags before committing.

## Validation

```bash
python -m pytest tests/test_nozzle_pinn_fix.py -v
python -m pytest tests/test_nozzle_regression.py -v
```
