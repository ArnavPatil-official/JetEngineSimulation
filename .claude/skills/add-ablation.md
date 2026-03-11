# Skill: Add Ablation Study

Use this skill when the plan requires systematic ablation of PINN components to measure their contribution.

## Setup

1. Identify the component to ablate (e.g., physics loss term, network branch, input feature).
2. Create a config matrix: baseline + one variant per ablated component.
3. Ensure each run uses the same seed, data split, and training epochs.

## Steps

1. **Create ablation script** (or modify existing training script):
   - Accept a `--ablate` flag specifying which component to disable.
   - Log: ablation name, final loss, physics metrics, wall time.

2. **Run all variants**:
   ```bash
   python scripts/run_ablation.py --ablate none        # baseline
   python scripts/run_ablation.py --ablate physics_loss
   python scripts/run_ablation.py --ablate boundary_bc
   ```

3. **Collect results** into `outputs/results/ablation_results.csv`:
   ```
   variant,data_loss,physics_loss,total_loss,wall_seconds
   ```

4. **Generate comparison plot** using matplotlib → `outputs/plots/ablation_comparison.png`.

5. **Validate**: Baseline must match previous best within 5% tolerance.

## Output Expectations

- CSV with one row per variant.
- Bar chart or table comparing metrics.
- Brief text summary of which components are essential vs. disposable.
