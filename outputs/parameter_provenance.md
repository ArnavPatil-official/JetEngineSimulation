# Parameter Provenance — Integrated Turbofan Model

Generated as part of the Phase 1 reviewer-response repair (2026-07-13; see
`docs/plan.md`). Every model parameter that influences reported results is
listed with its value, status (fixed / calibrated / fitted / removed / inert),
code location, and sourcing status. "Unsourced" means no literature citation
exists for the value; the manuscript must not describe such values as
literature-based or learnable.

Calibrated values below are the converged parameters of the seeded one-time
Optuna calibration (`scripts/optimization/calibrate_lto.py`, seed 42,
50 trials), frozen in `outputs/calibration_trent1000_ae3.json`.

## Cycle parameters

| Parameter | Value | Status | Bounds searched | Code location | Source / citation status |
|---|---|---|---|---|---|
| Compressor isentropic efficiency η_c | 0.86 | **Fixed** (never learnable) | — | `integrated_engine.py:565` | Unsourced; within the 0.80–0.90 "typical modern compressor" range noted in `simulation/compressor/compressor.py` docstring. No engine-specific citation. |
| Overall pressure ratio π_c | 43.2 | **Fixed** at design | — | `integrated_engine.py:566` | Trent 1000-AE3 rated OPR, `data/icao_engine_data.csv` (UID 02P23RR126). |
| Turbine polytropic efficiency η_poly | 0.9 | **Fixed** | — | `integrated_engine.py:590` | Unsourced round number. |
| Core mass flow (design) | 79.9 kg/s | **Fixed**, hand-set | — | `integrated_engine.py:545` | Hand-set during calibration to bring take-off fuel flow into range; not a published Trent 1000 core flow. |
| Bypass ratio | 9.1 | **Fixed** (bookkeeping only; no fan/bypass thrust model in Phase 1) | — | `integrated_engine.py:546` | Trent 1000-AE3, ICAO CSV. |
| Ambient T, P | 288.15 K, 101325 Pa | Fixed | — | `integrated_engine.py:550-551` | Sea-level ISA (standard). |
| Combustor efficiency η_comb (LTO calibration) | **0.9765** (converged) | **Calibrated** once vs 3 Trent 1000-AE3 LTO points + 1 untraceable climb point | 0.96–0.999 | `scripts/optimization/calibrate_lto.py`; frozen in `outputs/calibration_trent1000_ae3.json` | Calibration artifact, not a measurement. In-sample mean abs. error of seeded run: **5.46%**. |
| Combustor efficiency (blend optimization) | 0.98 | **Fixed** during all Optuna blend trials | — | `scripts/optimization/optimize_blend.py` (objective) | Assumed constant; NOT learnable, contrary to manuscript §2.1–2.2 wording. |
| η_comb φ-parabola (η = 0.995 − 0.04(φ−1)², clip [0.90, 0.999]) | η_max=0.995, k_φ=0.04 | Fixed functional form | — | `simulation/combustor/combustor.py::estimate_efficiency` | Unsourced heuristic shape; only used when no explicit efficiency is passed. |
| SAF name-triggered efficiency penalties (−1.5% 'Bio-SPK', −1.0% 'HEFA') | — | **REMOVED 2026-07-13** | — | formerly `simulation/combustor/combustor.py::estimate_efficiency` | Unsourced. Ablation (`scripts/validation/ablate_saf_penalty.py`, `outputs/ablation_saf_penalty.csv`, φ=0.5): penalties shifted TSFC +0.333% (HEFA blends) / +0.501% (Bio-SPK), thrust −0.33%/−0.50%, T4 −10.5/−15.7 K — and **flipped the TSFC fuel ranking** (with penalties Jet-A1 ranked above HEFA/Bio-SPK; without them Jet-A1 ranks last). Any blend-sensitivity claim built on these penalties must be withdrawn. |
| Combustor pressure loss p_loss | 0.0497 (converged) | **INERT** — sampled by Optuna but never consumed by any cycle equation | 0.03–0.06 | `scripts/optimization/calibrate_lto.py` | No pressure-loss model exists in the cycle path; the parameter has zero effect on the objective. Must not be reported as a model parameter. |
| φ per LTO mode | Idle **0.2950**, Approach **0.3401**, Climb **0.4573**, Take-off **0.5065** | **Calibrated** (converged values) | 0.22–0.30 / 0.30–0.40 / 0.42–0.50 / 0.50–0.60 | `scripts/optimization/calibrate_lto.py`; JSON | Throttle-setting surrogates, not measured equivalence ratios. NOTE: the Climb target (2.050 kg/s) does **not** appear in `data/icao_engine_data.csv` (the CSV has no CLIMB rows); only Idle/Approach/Take-off targets are traceable to the dataset. |
| Per-mode airflow scales | 0.15 / 0.35 / 0.85 / 1.00 | Fixed, hand-set | — | `scripts/optimization/calibrate_lto.py::MODE_SCALES` | Unsourced hand-tuning. |
| Per-mode π scales | 0.15 / 0.40 / 0.90 / 1.00 | **INERT** — assigned to `design_point['pi_c']`, which `run_compressor()` never reads (`Compressor.pi_c` stays 43.2) | — | `scripts/optimization/calibrate_lto.py::MODE_SCALES` | Dead code path: every LTO mode was simulated at full rated OPR. Disclose in manuscript; a real part-power OPR model is Phase 2 scope. |
| TIT hard limit / soft limit / penalty slope | 2800 K / 1850 K / 5×10⁻⁴ K⁻¹ | Fixed | — | `scripts/optimization/optimize_blend.py` | Unsourced optimization heuristics (hard limit is far above real TIT capability; it is a solver-failure guard, not a design constraint). |

## Emissions parameters

| Parameter | Value | Status | Code location | Source / citation status |
|---|---|---|---|---|
| NOx model NOx = a·OPR^b·ṁ_fuel^c | a=9.8214, b=0.2070, c=0.9506 | **Fitted in-sample** to the full ICAO CSV at startup; R²=0.9969 is the in-sample fit statistic (this is the number that leaked into the Highlights) | `integrated_engine.py::_fit_nox_model` | Functional form is a P3-style correlation choice (unsourced as applied); coefficients are data-fit artifacts, refit on every run. NOT validated out-of-sample. |
| CO model k = 2844.33 (g/kg per unit inefficiency²) | anchored to 7.11 g/kg at η=95% (IDLE) | Calibrated to one anchor point | `integrated_engine.py::estimate_co` | Single-point anchor; unsourced functional form. |
| CO₂ emission index | 3.16 kg CO₂ / kg fuel | Fixed | `integrated_engine.py:330` | Standard kerosene stoichiometric EI (e.g., ICAO carbon-calculator convention); citable. Applied to ALL blends despite differing carbon fractions — surrogate-specific w_C is **not** used in CO₂ (see below). |
| LCA factors (JetA 1.0, HEFA 0.2, FT 0.1, ATJ 0.3) | — | Fixed | `scripts/optimization/optimize_blend.py::LCA_FACTORS` | **Unsourced point values** — no CORSIA/feedstock basis, no uncertainty. The "80% CO₂ cut" Highlights claim is the HEFA input assumption (1 − 0.2) echoed back, not a result. Phase 2: replace with CORSIA ranges + Monte Carlo. |

## Fuel surrogate properties (`simulation/fuels.py`)

Species dictionaries are **mole fractions** (normalized before use; consumed by
Cantera `set_equivalence_ratio`). This is the single source of truth — the
manuscript composition table must match these values and state the mole basis.
H/C and w_C below are computed programmatically
(`FuelSurrogate.h_over_c_ratio()` / `.computed_carbon_fraction()`).

| Surrogate | Composition (mole fractions) | LHV [MJ/kg] | H/C (computed) | w_C (computed) |
|---|---|---|---|---|
| Jet-A1 | NC12H26 1.00 | 44.1 | 2.167 | 0.846 |
| HEFA-SPK | NC12H26 0.85, IC8H18 0.15 | 44.0 | 2.175 | 0.846 |
| FT-SPK | NC12H26 0.50, NC10H22 0.35, IC8H18 0.15 | 43.9 | 2.187 | 0.845 |
| ATJ-SPK | IC8H18 0.80, NC12H26 0.20 | 43.5 | 2.227 | 0.843 |

Notes:

1. **Which set produced the results.** Git history is conclusive: these
   compositions are unchanged since `simulation/fuels.py` was created
   (commit `12c9852`, 2025-12-10), which predates all reported runs (e.g.,
   `outputs/results/optimization_results.csv`, 2025-12-20). The manuscript
   table (HEFA 60/40, FT 45/35/20, ATJ 75/25 "by mass") did **not** generate
   any reported number and must be corrected to the values above.
2. **Effect of the discrepancy** (design point φ=0.5, η=0.985, manuscript
   set minus code set): HEFA ΔTSFC −0.124%, ΔFF −0.153%; FT ΔTSFC −0.026%;
   ATJ ΔTSFC +0.037%. The blend-to-blend TSFC spread in the code set is
   0.380%, and the HEFA composition shift (0.124%) **exceeds** the
   HEFA-vs-FT separation (0.092%) — i.e., the composition ambiguity is the
   same order as some blend distinctions the manuscript draws. Flagged for
   the Phase 2 sensitivity reframing.
3. **Inert attributes.** `LHV_MJ_per_kg` and `carbon_fraction` are not
   consumed anywhere in the simulation (CO₂ uses the flat 3.16 factor;
   heat release comes from Cantera equilibrium). They are documentation
   metadata. The `FuelSurrogate` default `carbon_fraction` was corrected
   from 0.857 to 0.847 (144.132/170.34 ≈ 0.846–0.847) on 2026-07-13; no
   result changes because nothing reads it.
4. **DCN values** in the manuscript table are neither computed by this code
   nor cited; drop them unless a citation is added (plan P1.4.5).

## Validation status summary

| Output | Status |
|---|---|
| LTO fuel flow | Calibrated on Trent 1000-AE3 (UID 02P23RR126); **held-out cross-engine validation: overall MAPE 7.27%** across 57 held-out certification records (Approach 2.06%, Idle 3.21%, Take-off 16.53% — the take-off error is a systematic ~+16% overprediction, not scatter). Script: `scripts/validation/holdout_icao_validation.py`; results: `outputs/holdout_icao_validation_summary.csv`. CSV contains no Climb mode — climb is uncalibratable/unvalidatable from this dataset. |
| NOx | NOT validated (in-sample correlation fit only). |
| CO | NOT validated (single-anchor calibration). |
| Lifecycle CO₂ | NOT validated (unsourced point factors). |
| Blend-discriminating outputs (TSFC/thrust deltas between fuels) | NOT validated; historically inflated by the removed SAF penalties (see ablation above). |
