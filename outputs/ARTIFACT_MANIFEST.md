# ARTIFACT MANIFEST — canonical sources for every manuscript-bound number

Phase 3.3 (2026-07-14). **The Google-Docs manuscript edit works ONLY from
this manifest.** Every manuscript-bound number or figure maps to exactly one
generating script and output file below. If a file under `outputs/` is not
listed here (or in the WIP/archive sections), it must not be cited.

Production configuration (frozen): calibration **v4**
(`calibration_trent1000_ae3_v4.json`, seed 42, β=0.8), components
**turbine=analytic, nozzle=analytic** (P3.1 adjudicated), heat loss ξ=0,
corrected air cycle. All stochastic artifacts record their seed internally.

## Model + validation

| # | Manuscript-bound content | Generating script (invocation) | Output file(s) | Seed |
|---|---|---|---|---|
| M1 | Calibrated parameters: η_comb 0.9963, p_loss 0.0442, k_π 0.5618, k_ṁ 0.6218, φ = {0.2953, 0.3053, 0.5409}; in-sample 1.62% | `scripts/optimization/calibrate_lto.py --tag v4 --n-trials 100 --beta 0.8` | `outputs/calibration_trent1000_ae3_v4.json` | 42 |
| M2 | Held-out validation MAPE **2.50%** (take-off 1.81 / approach 1.09 / idle 4.60); 57 records excl. AE3 | `scripts/validation/holdout_icao_validation.py --calibration outputs/calibration_trent1000_ae3_v4.json --tag _v4` | `outputs/holdout_icao_validation_summary_v4.csv`, scatter + boxplot `_v4.png` | via M1 |
| M3 | Validation-history table (v1 7.27% / v2 14.64% / v3 12.71% / v4 2.50%) | same script, `--calibration` per version | `holdout_icao_validation_summary*.csv` (4 files) | via each JSON |
| M4 | Take-off design point: thrust 241.6 kN (core 55.4 + bypass 186.2), TSFC 9.59 mg/(N·s), specific thrust 299 N·s/kg, T3 901.5 K, T4 1894.9 K, T5 1174.4 K, fuel flow 2.318 vs ICAO 2.327 kg/s | `scripts/validation/design_point_summary.py` | `outputs/design_point_summary_v4.csv` | deterministic |
| M5 | Airflow-split diagnostic (β sweep; bias resolution evidence) | `scripts/validation/airflow_split_sensitivity.py` | `outputs/airflow_split_sensitivity.csv` + plot | deterministic |
| M6 | Turbine p5 adjudication (analytic = work-consistent 4.728 bar; PINN −41.5%) | `scripts/validation/adjudicate_turbine_p5.py` | `outputs/turbine_p5_adjudication.csv` | deterministic |
| M7 | Held-out NOx-correlation validation: leave-one-engine-out MAPE **4.17%** (take-off 3.17 / approach 5.67 / idle 3.69), 180 records / 28 engine models; in-sample reference 4.15%. **Within-family only** — all 28 models are Trent 1000 variants — and the correlation has no fuel-composition term, so this is NOT evidence for blend NOx ranking | `scripts/validation/nox_holdout_validation.py` | `outputs/nox_holdout_validation.csv`, `outputs/nox_holdout_validation_summary.csv`, `plots/nox_holdout_pred_vs_icao.png`, `plots/nox_holdout_error_boxplot.png` | deterministic |

## Optimization studies + analysis (Phase 3.2 re-runs, adjudicated config)

| # | Content | Generating script (invocation) | Output file(s) | Seed |
|---|---|---|---|---|
| S1 | Free-φ study, N=1000, 0 failed/pruned; objective spreads; Pareto flag (642 members) | `scripts/optimization/optimize_blend.py --n-trials 1000 --seed 42 --calibration outputs/calibration_trent1000_ae3_v4.json --turbine-model analytic --nozzle-model analytic` | `outputs/results/optimization_results.csv`; `plots/pareto_3d.png`, `plots/parallel_coordinates.png` | 42 |
| S2 | φ-frozen study (AB8): blends move TSFC 0.25%, spec thrust 0.03%, NOx 0.48%; lifecycle CO₂e spans 89.8%; 15-member Pareto front | same + `--freeze-phi --output-csv outputs/results/optimization_results_phi_frozen.csv` | `outputs/results/optimization_results_phi_frozen.csv` | 42 |
| S3 | Variance decomposition (φ ≈ 100% of TSFC/thrust/NOx variance; blends ≈ 0%; lifecycle: φ 68.5% / CORSIA draw 29.3%) | `scripts/analysis/variance_decomposition.py --seed 42 --n-mc 1000` | `outputs/results/variance_decomposition.csv` + `plots/variance_decomposition.png` | 42 |
| S4 | Objective correlations | same | `outputs/results/objective_correlation_*.csv` + `plots/objective_correlation_heatmap.png` | 42 |
| S5 | CORSIA MC rank stability **96.7%** (1000 common-scenario draws) + lifecycle P5–P95 bands | same | `outputs/results/lca_rank_stability.csv` + `plots/pareto_lca_bands.png` | 42 |
| S6 | Representative balanced solution: **Trial 900** — TSFC 8.14 mg/(N·s), spec thrust 270.9 N·s/kg, lifecycle CO₂e 4177 g/s, NOx(corr) 65.9 g/s, SAF 49.4% (HEFA 8.6/FT 36.3/ATJ 4.5), φ 0.4155, CORSIA draw in-row | same (min-normalized-distance-to-ideal rule, encoded in script) | `outputs/results/representative_solution.csv` | 42 |
| S7 | Pareto-membership diff vs Phase 2 PINN-config study: 490 stable / 152 gained / 174 lost (40% churn of union) → Phase 2's 99.4% stability figure retired; 96.7% (S5) is the reported value | reproduce: compare `outputs/archive/pre_phase3_results/optimization_results_phase2_pinncfg.csv` vs S1 on the `ParetoOptimal` column | archived snapshot + S1 | 42/42 |

## Emissions & sensitivity evidence (Phase 2, still-current evidence)

| # | Content | Generating script | Output file(s) |
|---|---|---|---|
| E1 | CORSIA L_CEF values (every value cited to ICAO Doc 06 Nov-2025 table rows) | hand-built data file, values verified against the PDF | `data/corsia_lca_values.yaml` |
| E2 | EI-CO₂ = 3.664·w_C (3.100 kg/kg Jet-A1; Cantera cross-check 3.098) | `simulation/fuels.py::carbon_fraction_of_composition` + `EmissionsEstimator.estimate_co2`; derivation in provenance doc | `outputs/parameter_provenance.md` (Phase 2 section) |
| E3 | NOx three-path comparison (paths differ by orders of magnitude; no blend NOx ranking) | `scripts/validation/nox_dual_path.py` (v3-state conditions; evidence artifact) | `outputs/nox_dual_path.csv` + `plots/nox_path_comparison.png` |
| E4 | Heat-loss sensitivity (ξ=4% > blend signal) | `scripts/validation/heat_loss_sensitivity.py` (v3-state; evidence artifact) | `outputs/heat_loss_sensitivity.csv` + plot |
| E5 | PINN component ablation (±10% thrust; basis of P3.1 adjudication) | `scripts/validation/ablate_pinn_components.py` (v3-state; evidence artifact) | `outputs/ablation_pinn_components*.csv` |
| E6 | Sajben negative result (Cp shape-L2 0.71–1.09 vs <0.10) | `scripts/validation/sajben_figure.py` (+ read-only `sajben_validation.py`) | `outputs/sajben_validation_errors.csv` + `plots/sajben_wall_cp_validation.png` |
| E7 | SAF-penalty ablation (removed penalties flipped TSFC ranking) | `scripts/validation/ablate_saf_penalty.py` (Phase 1) | `outputs/ablation_saf_penalty.csv` |
| E8 | Parameter provenance table (supplementary material) | maintained by hand, updated each phase | `outputs/parameter_provenance.md` |

Note on E3–E5: these were generated at the v3/PINN-default state as the
*evidence that drove the Phase 3 decisions*; they are cited as such
(decision evidence), never as production-configuration results.

## Provenance note on the retracted R² = 0.9969 Highlight

Confirmed by `scripts/validation/nox_holdout_validation.py`: the withdrawn
Highlight figure is the **in-sample** R² of the three-parameter log-log NOx fit
(`EI_NOx = 9.8214 · OPR^0.2070 · ṁ_f^0.9506`), which `EmissionsEstimator`
printed at construction. It was never a validation statistic. The startup print
now labels it a training diagnostic, and M7 supplies the held-out figure that
the Highlight implied but never had.

## Not manuscript sources

- `outputs/results/pinn_comparison_results.csv`, `outputs/results/integrated_cycle_results.csv`, `outputs/plots/integrated_cycle_comparison.png` — user work-in-progress (LE-PINN comparison stream, uncommitted local script edits); not part of the canonical pipeline.
- `outputs/pinn_architecture_diagram.png` (+ `.py`) — static architecture illustration; no numbers.
- `outputs/archive/pre_phase2/` — the old manuscript's artifacts, moved here in Phase 3.3 (includes `pareto_optimal_solutions.csv`, whose first row was the old paper's representative solution 29.464/800.58/0.7616). **Dead numbers. Never cite.**
- `outputs/archive/pre_phase3_results/` — Phase 2 PINN-config study snapshot (kept for the S7 diff).
- `outputs/archive/le_pinn/` — user's own LE-PINN plot archive.

## Visualization-script status (P3.3 resolution)

`scripts/visualization/pareto_visual.py` and `visualize_results.py` carry
uncommitted user modifications and were NOT edited or executed by Phases
2–3. They are **superseded** for all manuscript figures by the scripts in
this manifest (S1–S6 figures come from `optimize_blend.py` and
`variance_decomposition.py`). Their historical output
`pareto_optimal_solutions.csv` is archived (dead). Committing or discarding
the local modifications to those two files is left to the user — flagged,
not clobbered.
