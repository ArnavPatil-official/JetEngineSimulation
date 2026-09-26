# Integrated Cycle Comparison + Research-Paper Graph Suite — May 2026

## Objective

Three tasks in one plan:

1. **Integrated cycle bar chart** — add LE-PINN nozzle support to the engine runner and produce a
   grouped bar chart (ICAO / Regular PINN / LE-PINN) across the four ICAO LTO power settings,
   matching the style of the existing `validation_chart.png`.

2. **Research-paper graph audit** — replace, fix, and remove the 16 numbered plots in
   `scripts/visualization/visualize_results.py` so that only publication-quality, data-backed panels
   remain.

3. **Statistical validation** — two formal statistical tests embedded in the comparison script:
   - Test A: Integrated cycle (Regular PINN) vs ICAO ground-truth fuel flow
   - Test B: Integrated cycle (Regular PINN) vs Integrated cycle (LE-PINN)

## Complexity / Recommended Model
High — Claude Opus

## Constraints
- Do NOT overwrite `models/*.pt`. All model paths are read-only.
- Do NOT change random seeds or PINN training code in `le_pinn.py` or `nozzle.py`.
- Do NOT modify YAML mechanism files in `data/`.
- Run `python3 -m pytest tests/ -v` after any change to `integrated_engine.py` or `le_pinn.py`.
- All new plots must use `matplotlib.use("Agg")` and save to `outputs/plots/`.
- The `integrated_engine.py` change must be backward-compatible (existing callers with no
  `nozzle_variant` argument must behave identically to today).

---

## Relevant Files

### Modify
- `integrated_engine.py` — add `nozzle_variant` parameter
- `scripts/validation/integrated_cycle_comparison.py` — CREATE (does not exist yet)
- `scripts/visualization/visualize_results.py` — graph audit / redesign

### Read-only
- `simulation/nozzle/le_pinn.py` — `run_le_pinn()` public API
- `simulation/nozzle/nozzle.py` — `run_nozzle_pinn()` public API
- `scripts/optimization/calibrate_lto.py` — LTO phi values and mass-flow scaling
- `scripts/visualization/plot_validation.py` — reference ICAO chart style
- `data/icao_engine_data.csv` — ICAO ground-truth
- `models/` — all checkpoints (read-only)

---

## Part 1 — Add LE-PINN Support to `integrated_engine.py`

### 1-A. Add `nozzle_variant` to `__init__`

In `IntegratedTurbofanEngine.__init__`, add one new parameter after `nozzle_pinn_path`:

```python
nozzle_variant: str = "pinn",   # "pinn" | "le_pinn"
le_pinn_path: str = "models/le_pinn_unified.pt",
```

Store as instance attributes:
```python
self.nozzle_variant = nozzle_variant          # "pinn" | "le_pinn"
self.le_pinn_path   = le_pinn_path
```

### 1-B. Modify `_run_nozzle_stage`

At the top of `_run_nozzle_stage` (line ~977), add the LE-PINN branch BEFORE the existing
`version_ok` check:

```python
if self.nozzle_variant == "le_pinn":
    from simulation.nozzle.le_pinn import run_le_pinn
    le_path = self.le_pinn_path
    # Resolve fallback checkpoint path if primary missing
    from pathlib import Path as _Path
    if not _Path(le_path).exists():
        le_path = str(_Path(le_path).parent / "le_pinn_engine_unified.pt")
    try:
        le_result = run_le_pinn(
            model_path=le_path,
            inlet_state=turb_result,
            ambient_p=self.design_point['P_ambient'],
            A_in=self.design_point['A_nozzle_inlet'],
            A_exit=self.design_point['A_nozzle_exit'],
            length=1.0,
            thermo_props={
                'cp': turb_result['cp'],
                'R':  turb_result['R'],
                'gamma': turb_result['gamma'],
            },
            m_dot=m_dot_total,
            n_axial=50,
            n_radial=20,
            device="cpu",
            return_profile=False,
            thrust_model="static_test_stand",
        )
        return {
            'rho':             le_result['exit_state']['rho'],
            'u':               le_result['exit_state']['u'],
            'p':               le_result['exit_state']['p'],
            'T':               le_result['exit_state']['T'],
            'thrust_total':    le_result['thrust_total'],
            'thrust_momentum': le_result['thrust_momentum'],
            'thrust_pressure': le_result['thrust_pressure'],
        }
    except Exception as exc:
        print(f"⚠️  LE-PINN nozzle failed ({exc}). Falling back to regular PINN.")
        # fall through to existing run_nozzle_pinn code below
```

The existing `version_ok` / `run_nozzle_pinn` code path continues unchanged as the fallback.

### 1-C. Verify backward compatibility

After the change, the following must produce identical output to pre-change:

```bash
python3 -c "
from integrated_engine import IntegratedTurbofanEngine
# Default: nozzle_variant='pinn' — must work without error
e = IntegratedTurbofanEngine.__new__(IntegratedTurbofanEngine)
import inspect
sig = inspect.signature(IntegratedTurbofanEngine.__init__)
assert 'nozzle_variant' in sig.parameters
assert sig.parameters['nozzle_variant'].default == 'pinn'
assert 'le_pinn_path' in sig.parameters
print('Signature OK')
"
```

---

## Part 2 — Create `scripts/validation/integrated_cycle_comparison.py`

This script is the main deliverable. It must be fully standalone (no external state required beyond
the model checkpoints and ICAO CSV).

### 2-A. LTO Operating Points

Use the same scaling from `calibrate_lto.py`. Hard-code the best-calibrated phi values and
mass-flow scales. If the calibrated values are not stored anywhere, use the midpoints of the
search ranges as defaults (which match the existing `sim_data` in `plot_validation.py`):

```python
LTO_MODES = ["Idle", "Approach", "Climb", "Takeoff"]

# Per-mode scaling relative to design point (79.9 kg/s core, pi_c=43.2)
LTO_SCALES = {
    "Idle":     {"mass_scale": 0.15, "pi_scale": 0.15, "phi": 0.26},
    "Approach": {"mass_scale": 0.35, "pi_scale": 0.40, "phi": 0.35},
    "Climb":    {"mass_scale": 0.85, "pi_scale": 0.90, "phi": 0.46},
    "Takeoff":  {"mass_scale": 1.00, "pi_scale": 1.00, "phi": 0.55},
}

# ICAO ground-truth fuel flow [kg/s] (Trent 1000-AE3)
ICAO_FUEL_FLOW = {
    "Idle":     0.244,
    "Approach": 0.643,
    "Climb":    2.050,
    "Takeoff":  2.327,
}
ICAO_STD = {   # ±1σ across engines in icao_engine_data.csv — compute from CSV at runtime
    "Idle":     None,   # filled from CSV
    "Approach": None,
    "Climb":    None,
    "Takeoff":  None,
}
```

The ICAO standard deviations must be read from `data/icao_engine_data.csv` at runtime (do not
hard-code them). Use the same groupby logic as `scripts/visualization/plot_validation.py`:

```python
icao_df = pd.read_csv(REPO_ROOT / "data" / "icao_engine_data.csv")
stats = icao_df.groupby("Mode")["Fuel Flow (kg/s)"].agg(["mean", "std"])
# Map ICAO mode labels to LTO_MODES keys:
#   "IDLE" -> "Idle", "APPROACH" -> "Approach",
#   "CLIMB" -> "Climb" (may be missing — use 0.08 fallback std),
#   "TAKE-OFF" -> "Takeoff"
```

### 2-B. Engine runner function

```python
def _run_lto_sweep(
    fuel_blend,
    nozzle_variant: str,       # "pinn" | "le_pinn"
    design_base: dict,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run the integrated cycle at all four LTO points.
    Returns dict mapping mode name -> fuel_flow (kg/s).
    """
    fuel_flows = {}
    for mode, cfg in LTO_SCALES.items():
        engine = IntegratedTurbofanEngine(
            nozzle_variant=nozzle_variant,
        )
        # Apply per-mode scaling
        engine.design_point["mass_flow_core"] = design_base["mass_flow_core"] * cfg["mass_scale"]
        engine.design_point["pi_c"] = design_base["pi_c"] * cfg["pi_scale"]  # note: pi_c key
        result = engine.run_full_cycle(
            fuel_blend=fuel_blend,
            phi=cfg["phi"],
        )
        perf = result["performance"]
        fuel_flows[mode] = float(
            perf.get("fuel_flow", perf.get("fuel_mass_flow", float("nan")))
        )
    return fuel_flows
```

**IMPORTANT**: `IntegratedTurbofanEngine.design_point` uses `'pi_c'` as the key only if that was
added during calibration. Inspect the dict at runtime: if `'pi_c'` is absent, skip the pi_c
scaling (it may be internal to the Compressor). Do NOT crash — fall back to leaving pi_c at its
default.

### 2-C. Grouped bar chart

Style must match the reference `validation_chart.png` exactly:
- Gray bars: ICAO (mean ± 1σ)
- Blue bars: Regular PINN integrated cycle (± half the ICAO σ as propagated uncertainty)
- Orange bars: LE-PINN integrated cycle (same uncertainty convention)
- Δ annotations above each model bar: `Δ=sim − ICAO` (signed, 3 decimal places)
- Y-axis label: `"Fuel Flow Rate (kg/s)"`
- Title: `"Integrated Cycle Validation: Regular PINN vs LE-PINN vs ICAO"`
- Grid: `axis='y', linestyle='--', alpha=0.3`
- DPI: 300
- Save to: `outputs/plots/integrated_cycle_comparison.png`

Bar layout (4 mode groups × 3 bars):

```python
x = np.arange(len(LTO_MODES))
width = 0.25
ax.bar(x - width,     icao_vals,  width, yerr=icao_stds, ...)   # ICAO
ax.bar(x,             pinn_vals,  width, yerr=pinn_errs,  ...)   # Regular PINN
ax.bar(x + width,     le_vals,    width, yerr=le_errs,    ...)   # LE-PINN
```

### 2-D. Statistical Tests

Append a second figure (or second panel) with the results of two statistical tests:

#### Test A — Regular PINN vs ICAO (Physical validity)

Use a **paired t-test** (scipy.stats.ttest_rel) on the 4-point vectors:
- `pinn_vals` (Regular PINN fuel flow per LTO mode)
- `icao_vals` (ICAO mean fuel flow per LTO mode)

Also report:
- Mean signed bias: `mean(pinn_vals - icao_vals)`
- MAPE: `mean(|pinn_vals - icao_vals| / icao_vals) * 100`
- 95% CI on the bias (from t-distribution with df=3)

#### Test B — Regular PINN vs LE-PINN (Model comparison)

Use a **paired t-test** (scipy.stats.ttest_rel) on:
- `pinn_vals`
- `le_vals`

Also report:
- Mean absolute difference: `mean(|pinn_vals - le_vals|)`
- Max absolute difference
- Whether LE-PINN is statistically distinguishable from PINN (p-value interpretation)

**Note on n=4**: With only 4 paired observations, the minimum achievable two-sided p-value
for the paired t-test is ~0.058 (df=3, t≈3.18). Report the exact p-value without claiming
significance at α=0.05 unless p < 0.05. Include a note: *"n=4; interpret with caution."*

#### Statistical results figure

Create `outputs/plots/statistical_tests.png` — a single figure with 2 panels side by side:

**Panel left — Test A (PINN vs ICAO)**:
- Scatter plot: x = ICAO values, y = PINN values, identity line, ±10% band
- Annotate: `t={t:.3f}, p={p:.4f}, MAPE={mape:.1f}%`

**Panel right — Test B (PINN vs LE-PINN)**:
- Scatter plot: x = PINN values, y = LE-PINN values, identity line
- Annotate: `t={t:.3f}, p={p:.4f}, max |Δ|={max_diff:.4f} kg/s`

Print all test results to stdout with clear headers.

### 2-E. Script entry point

```python
if __name__ == "__main__":
    main()
```

`main()` should:
1. Load ICAO data from CSV
2. Run Regular PINN LTO sweep (Jet-A1 fuel)
3. Run LE-PINN LTO sweep (Jet-A1 fuel)
4. Print comparison table
5. Run both statistical tests
6. Save both figures
7. Exit with code 0 on success, 1 on any caught exception

---

## Part 3 — Research-Paper Graph Audit (`scripts/visualization/visualize_results.py`)

### 3-A. Plots to REMOVE entirely

Delete the following functions and remove their calls from `main()`:

| Function | File saves to | Reason for removal |
|---|---|---|
| `plot_05_fuel_radar` | `05_fuel_radar.png` | Non-standard chart type; information duplicated in heatmap |
| `plot_06_combustor_temperature_time` | `06_combustor_temperature_time.png` | Supplementary material only |
| `plot_07_species_heatmap` | `07_species_heatmap.png` | Uses `sim_nox * arbitrary_scale` — not real simulation output; misleading |
| `plot_09_bo_convergence_dual_axis` | `09_bo_convergence_dual_axis.png` | Methodology plot; move to supplementary if needed |
| `plot_12_engine_state_waterfall` | `12_engine_state_waterfall.png` | Replaces a table; design illustration not data |
| `plot_14_fuel_delta_heatmap` | `14_fuel_delta_heatmap.png` | Redundant with `11_lca_vs_netco2_scatter.png` |
| `plot_15_engine_cross_section_state_map` | `15_engine_cross_section_state_map.png` | Illustration, not scientific data |
| `plot_16_lepinn_nozzle_cross_section` | `16_lepinn_nozzle_cross_section.png` | Illustration, not scientific data |

Do not delete the function bodies yet — comment them out with a `# REMOVED:` marker so they
can be recovered. Remove their calls from the `main()` block.

### 3-B. Plots to KEEP unchanged

| Function | Output | Reason to keep |
|---|---|---|
| `plot_01_pinn_loss_curriculum` | `01_pinn_loss_curriculum.png` | Training methodology — required |
| `plot_08_pareto_3d_enhanced` | `08_pareto_3d_enhanced.png` | Core optimization result |
| `plot_10_parallel_coordinates` | `10_parallel_coordinates_highlighted.png` | Multi-objective trade-space |
| `plot_11_lca_vs_co2` | `11_lca_vs_netco2_scatter.png` | Sustainability contribution |

### 3-C. Plots to FIX

#### Fix `plot_02_flow_profiles_with_bands` → `02_flow_profiles_uncertainty.png`

**Problem**: Only shows regular Nozzle PINN. LE-PINN predictions are absent.

**Fix**: After generating the regular PINN profile for each of the 3 test cases
(`(inlet_p, inlet_t, a_in, a_out, m_dot)` triples), also call `run_le_pinn()` and overlay
its centerline predictions. Use dashed lines for LE-PINN, solid for regular PINN.
If `run_le_pinn` raises an exception (missing checkpoint, etc.), skip the LE-PINN overlay
gracefully and add a legend note: `"LE-PINN: unavailable"`.

The function signature `_predict_profile(model_class, ...)` is for the regular PINN only.
Add a parallel `_predict_le_profile(inlet_p, inlet_t, a_in, a_out, m_dot, length=1.0)`
that calls `run_le_pinn()` and returns `(x, profile_dict)` where `profile_dict` contains
`"T"`, `"p"`, `"u"`, `"rho"` arrays. Return `None, None` on failure.

#### Fix `plot_03_lepinn_benchmark_comparison` → `03_lepinn_benchmark_bars.png`

**Problem**: Plots internal PINN diagnostic metrics (inlet BC error %, mass error %, work
balance %) from `cycle_df` columns that are often NaN or unavailable. This is not a
scientifically meaningful bar chart.

**Redesign**: Replace entirely with the integrated cycle comparison bar chart.
- This function should now call the same `_run_lto_sweep` logic from
  `scripts/validation/integrated_cycle_comparison.py`, or simply load
  `outputs/results/integrated_cycle_results.csv` if it exists.
- If neither the CSV nor the models are available, fall back to the hardcoded sim_data
  (`[0.2316, 0.6620, 2.1066, 2.6809]`) and LE-PINN data as `None` with a "not available"
  overlay.
- Style: exactly match the reference `validation_chart.png` (gray ICAO bars, blue PINN bars,
  orange LE-PINN bars if available).
- The old LE-PINN diagnostic content (inlet BC error, mass error, work balance) is better
  served by the metrics table in `pinn_le_pinn_comparison.png` — do not recreate it here.

#### Fix `plot_04_nozzle_centerline_vs_isentropic` → `04_nozzle_centerline_vs_isentropic.png`

**Problem**: Only shows regular PINN centerline. Three test NPR cases but no LE-PINN overlay.

**Fix**: For each of the 3 test cases, call `run_le_pinn()` and add an orange dashed line
for T, p, u profiles. Wrap in try/except — if LE-PINN fails, skip gracefully.
Add "LE-PINN" entry to legend in each subplot.

#### Fix `plot_13_icao_validation_subplots` → `13_icao_validation_subplots.png`

**Problem**: Two-panel layout (fuel flow bar chart + NOx line chart). The NOx panel uses
`sim_nox = stats["nox_mean"].ffill().values * np.array([0.93, 0.96, 1.04, 1.08])` — this is
a scaled version of ICAO NOx, not actual simulation output. This is scientifically invalid and
must be removed.

**Redesign**: Replace with a SINGLE panel that exactly matches the reference chart style
(`validation_chart.png`):
- Three bars per LTO mode: ICAO (gray), Regular PINN (blue), LE-PINN (orange)
- Load LE-PINN values from `outputs/results/integrated_cycle_results.csv` if available,
  otherwise show only ICAO and Regular PINN bars
- Same Δ annotations as the reference chart
- ICAO ±1σ error bars, model bars use ±(0.5 × icao_std) propagated spread
- Save to `13_icao_validation_subplots.png` (keep filename for compatibility with any callers)

### 3-D. Final paper-ready graph set after audit

After the changes, `visualize_results.py` produces exactly these 8 plots:

| # | Filename | Content |
|---|---|---|
| 01 | `01_pinn_loss_curriculum.png` | PINN training loss + curriculum schedule |
| 02 | `02_flow_profiles_uncertainty.png` | Nozzle/turbine flow profiles: PINN + LE-PINN overlay |
| 03 | `03_lepinn_benchmark_bars.png` | Integrated LTO fuel flow: ICAO vs PINN vs LE-PINN |
| 04 | `04_nozzle_centerline_vs_isentropic.png` | Centerline profiles: PINN + LE-PINN vs isentropic |
| 08 | `08_pareto_3d_enhanced.png` | 3-D Pareto front (unchanged) |
| 10 | `10_parallel_coordinates_highlighted.png` | Parallel coordinates (unchanged) |
| 11 | `11_lca_vs_netco2_scatter.png` | LCA vs net CO₂ scatter (unchanged) |
| 13 | `13_icao_validation_subplots.png` | ICAO validation 3-bar chart (redesigned) |

Plus two outputs from the separate comparison scripts:
- `pinn_le_pinn_comparison.png` (from `compare_pinn_le_pinn.py` — already done)
- `integrated_cycle_comparison.png` + `statistical_tests.png` (from new script)

---

## Part 4 — Validation Commands

Run these in order after implementation:

```bash
# 1. Syntax check all modified files
python3 -c "
import ast
for f in [
    'integrated_engine.py',
    'scripts/validation/integrated_cycle_comparison.py',
    'scripts/visualization/visualize_results.py',
]:
    with open(f) as fh: ast.parse(fh.read())
    print(f'  [OK] {f}')
"

# 2. Backward-compat check for integrated_engine.py
python3 -c "
from integrated_engine import IntegratedTurbofanEngine
import inspect
sig = inspect.signature(IntegratedTurbofanEngine.__init__)
assert 'nozzle_variant' in sig.parameters, 'nozzle_variant param missing'
assert sig.parameters['nozzle_variant'].default == 'pinn', 'default must be pinn'
assert 'le_pinn_path' in sig.parameters, 'le_pinn_path param missing'
print('[OK] IntegratedTurbofanEngine backward-compatible')
"

# 3. Run test suite (must all pass)
python3 -m pytest tests/ -v --tb=short

# 4. Run the integrated cycle comparison script (dry-run with --help)
python3 scripts/validation/integrated_cycle_comparison.py --help

# 5. Verify plot outputs exist
python3 -c "
from pathlib import Path
expected = [
    'outputs/plots/integrated_cycle_comparison.png',
    'outputs/plots/statistical_tests.png',
]
for p in expected:
    assert Path(p).exists(), f'Missing: {p}'
    print(f'  [OK] {p}')
"
```

---

## Acceptance Criteria

- [ ] `integrated_engine.py` accepts `nozzle_variant="le_pinn"` without breaking existing callers
- [ ] `integrated_cycle_comparison.py` runs end-to-end and produces both PNG outputs
- [ ] Grouped bar chart has 3 bars per LTO mode with Δ annotations in the reference style
- [ ] Statistical test outputs include t-statistic, p-value, MAPE/bias for both test pairs
- [ ] `visualize_results.py` produces exactly 8 plots (not 16); removed functions are commented out
- [ ] All 8 surviving plots have LE-PINN overlays where applicable (plots 02, 03, 04, 13)
- [ ] `python3 -m pytest tests/ -v` passes with 0 failures

---

## Notes for the Executor

### `performance` dict key for fuel flow
In `run_full_cycle`, the returned dict contains a `"performance"` sub-dict. The fuel flow key
may be `"fuel_flow"` or `"fuel_mass_flow"` depending on the engine version. Always do:
```python
perf = result["performance"]
ff = perf.get("fuel_flow", perf.get("fuel_mass_flow", float("nan")))
```

### `pi_c` vs `P_ratio`
`design_point` does NOT have a `'pi_c'` key — the pressure ratio is passed to `Compressor`
during `__init__` as `pi_c=43.2`. To scale per-mode pressure ratio, the engine must be
re-instantiated per mode OR the Compressor must be re-parameterized. The safest approach
is **re-instantiating `IntegratedTurbofanEngine` per mode** (each mode creates a fresh engine
instance). This avoids mutating shared state. The Cantera load overhead is acceptable
for 4 × 2 = 8 engine runs.

### LE-PINN checkpoint resolution
`run_le_pinn()` in `le_pinn.py` already handles fallback internally:
it tries `le_pinn_unified.pt` then `le_pinn_engine_unified.pt`. Do not replicate this logic —
just pass `model_path=str(REPO_ROOT / "models" / "le_pinn_unified.pt")` and let `run_le_pinn`
resolve.

### Statistical test with n=4
```python
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(pinn_vals, icao_vals)
```
The paired t-test is appropriate here because each observation pair (mode → fuel flow)
shares the same operating condition. However, with df=3, p < 0.05 requires |t| > 3.18.
If p ≥ 0.05, the correct interpretation is "insufficient evidence to reject H₀ that the
models agree" — NOT that they definitely agree. Report this distinction explicitly.
