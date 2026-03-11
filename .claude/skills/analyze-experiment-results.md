# Skill: Analyze Experiment Results

Use this skill when the plan involves comparing simulation outputs against baselines, ICAO data, or across parameter sweeps.

## Data Sources

- **ICAO reference data**: `data/icao_engine_data.csv`
- **Simulation results**: `outputs/results/*.csv`
- **Existing plots**: `outputs/plots/`
- **EDA scripts**: `evaluation/cantera_eda.py`, `evaluation/icao_eda.py`

## Steps

1. **Load results** from the CSV files specified in the plan.
2. **Compare against baseline**:
   - If ICAO data: compute relative error per engine parameter.
   - If previous run: compute delta and % change.
3. **Generate validation plots**:
   - Use existing `scripts/visualization/plot_validation.py` as a starting point.
   - Save to `outputs/plots/` with descriptive filenames.
4. **Produce summary table** in markdown or CSV:
   ```
   parameter,baseline,current,delta,pct_change,pass
   ```

## Validation Thresholds

| Metric | Acceptable | Warning | Fail |
|--------|-----------|---------|------|
| Thrust (kN) | ±2% | ±5% | >5% |
| TSFC | ±3% | ±5% | >5% |
| NOx (g/kN) | ±10% | ±20% | >20% |
| CO2 index | ±5% | ±10% | >10% |

## Output

- Updated plots in `outputs/plots/`.
- Summary CSV in `outputs/results/`.
- Print pass/warn/fail status for each metric.
