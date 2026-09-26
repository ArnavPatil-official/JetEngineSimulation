# Parameter Provenance — Integrated Turbofan Model

Generated as part of the Phase 1 reviewer-response repair (2026-07-13; see
`docs/plan.md`) and updated for Phase 2 (2026-07-14; see the "Phase 2
updates" section at the end — it supersedes rows above where noted). Every
model parameter that influences reported results is listed with its value,
status (fixed / calibrated / fitted / removed / inert), code location, and
sourcing status. "Unsourced" means no literature citation exists for the
value; the manuscript must not describe such values as literature-based or
learnable.

Values marked "calibrated" in the Phase 1–3 sections are the values the
seeded one-time Optuna calibration (`scripts/optimization/calibrate_lto.py`,
seed 42) returned, frozen in `outputs/calibration_trent1000_ae3*.json`.
**Returned is not identified:** the Phase 5 section at the end (2026-09-25)
supersedes the LTO-calibration statuses of the part-power formulation
(v2–v4) — of its seven sampled parameters the fuel-flow objective
identifies **φ_to alone**; η_b, pressure_loss and k_pi are unidentified,
and k_mdot / φ_idle / φ_app are jointly unidentified (a ridge). The Phase 1
(v1) formulation is different (fixed per-mode airflow scales, no airflow
exponent); its rows below are scoped separately. In every version η_b and
pressure_loss are inert in the fuel-flow objective. No returned value —
identified or not — is claimed to be the exact optimum of its objective.

## Cycle parameters

| Parameter | Value | Status | Bounds searched | Code location | Source / citation status |
|---|---|---|---|---|---|
| Compressor isentropic efficiency η_c | 0.86 | **Fixed** (never learnable) | — | `integrated_engine.py:565` | Unsourced; within the 0.80–0.90 "typical modern compressor" range noted in `simulation/compressor/compressor.py` docstring. No engine-specific citation. |
| Overall pressure ratio π_c | 43.2 | **Fixed** at design | — | `integrated_engine.py:566` | Trent 1000-AE3 rated OPR, `data/icao_engine_data.csv` (UID 02P23RR126). |
| Turbine polytropic efficiency η_poly | 0.9 | **Fixed** | — | `integrated_engine.py:590` | Unsourced round number. |
| Core mass flow (design) | 79.9 kg/s | **Fixed**, hand-set | — | `integrated_engine.py:545` | Hand-set during calibration to bring take-off fuel flow into range; not a published Trent 1000 core flow. |
| Bypass ratio | 9.1 | **Fixed** (bookkeeping only; no fan/bypass thrust model in Phase 1) | — | `integrated_engine.py:546` | Trent 1000-AE3, ICAO CSV. |
| Ambient T, P | 288.15 K, 101325 Pa | Fixed | — | `integrated_engine.py:550-551` | Sea-level ISA (standard). |
| Combustor efficiency η_comb (LTO calibration) | 0.9765 (sampler's value; not an estimate) | **Unidentified by the objective** (Phase 5) — sampled once vs 3 Trent 1000-AE3 LTO points + 1 untraceable climb point; fuel flow has zero sensitivity to it | 0.96–0.999 | `scripts/optimization/calibrate_lto.py`; frozen in `outputs/calibration_trent1000_ae3.json` | Calibration artifact, not a measurement. In-sample mean abs. error of seeded run: **5.46%**. |
| Combustor efficiency (blend optimization) | 0.98 | **Fixed** during all Optuna blend trials | — | `scripts/optimization/optimize_blend.py` (objective) | Assumed constant; NOT learnable, contrary to manuscript §2.1–2.2 wording. |
| η_comb φ-parabola (η = 0.995 − 0.04(φ−1)², clip [0.90, 0.999]) | η_max=0.995, k_φ=0.04 | Fixed functional form | — | `simulation/combustor/combustor.py::estimate_efficiency` | Unsourced heuristic shape; only used when no explicit efficiency is passed. |
| SAF name-triggered efficiency penalties (−1.5% 'Bio-SPK', −1.0% 'HEFA') | — | **REMOVED 2026-07-13** | — | formerly `simulation/combustor/combustor.py::estimate_efficiency` | Unsourced. Ablation (`scripts/validation/ablate_saf_penalty.py`, `outputs/ablation_saf_penalty.csv`, φ=0.5): penalties shifted TSFC +0.333% (HEFA blends) / +0.501% (Bio-SPK), thrust −0.33%/−0.50%, T4 −10.5/−15.7 K — and **flipped the TSFC fuel ranking** (with penalties Jet-A1 ranked above HEFA/Bio-SPK; without them Jet-A1 ranks last). Any blend-sensitivity claim built on these penalties must be withdrawn. |
| Combustor pressure loss p_loss | 0.0497 (sampler's value) | **INERT** — sampled by Optuna but never consumed by any cycle equation | 0.03–0.06 | `scripts/optimization/calibrate_lto.py` | No pressure-loss model exists in the cycle path; the parameter has zero effect on the objective. Must not be reported as a model parameter. |
| φ per LTO mode | Idle **0.2950**, Approach **0.3401**, Climb **0.4573**, Take-off **0.5065** (sampler's values) | Phase 1 (v1) formulation: airflow per mode was a **fixed** hand-set scale, so each φ is structurally identifiable by its own mode's fuel-flow target (Climb's target is untraceable, below); the Phase 5 F2 k_mdot ridge is a property of the v2–v4 part-power law and does not apply here. The returned values are a 50-trial sampler's, not verified optima: φ_to = 0.5065 sits at the 0.50 lower bound while take-off fuel flow is over-predicted (+16.5 %), so there the box, not the data, limited it. | 0.22–0.30 / 0.30–0.40 / 0.42–0.50 / 0.50–0.60 | `scripts/optimization/calibrate_lto.py`; JSON | Throttle-setting surrogates, not measured equivalence ratios. NOTE: the Climb target (2.050 kg/s) does **not** appear in `data/icao_engine_data.csv` (the CSV has no CLIMB rows); only Idle/Approach/Take-off targets are traceable to the dataset. |
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

---

# Phase 2 updates (2026-07-14)

## Discovered defect: argon compression (fixed)

The shared Cantera `Solution` initialized to the CRECK mechanism's first
species — **pure argon** — and no code ever set an air composition before the
compressor state calculations. Every previously reported temperature was
computed for monatomic argon (γ = 1.67): T3 at rated OPR was 1464 K instead
of 901 K, inflating T4 and all downstream temperatures by hundreds of kelvin.
`run_compressor` now sets air (O2:0.21, N2:0.79). Fuel flow was unaffected
(FAR is computed on a separate, correctly initialized gas), so all fuel-flow
calibrations and holdout MAPEs remain valid; every pre-fix temperature,
thrust, and TSFC number was not. Fixed in commit 9588d31.

## Cycle parameters (new / superseding)

| Parameter | Value | Status | Code location | Source / notes |
|---|---|---|---|---|
| Part-power throttle law | pi_c = 1+(pi_rated−1)x^k_pi; m_dot = m_rated·x^k_mdot; x = ICAO power setting | Functional form, hand-chosen | `integrated_engine.part_power_state` | Low-fidelity throttle model; replaces the 4 hand-set scales (2 of which were inert). |
| k_pi | 1.342 (v3 sampler's value) | **Unidentified by the v4 objective** (and v2/v3): FAR depends on phi only, so pi_c does not enter the calibration objective except via crashes; v2 landed at 0.61, v3 at 1.34 on the same data. It DOES affect NOx (OPR) and T3. Report as an assumption, not a fitted constant. | `outputs/calibration_trent1000_ae3_v3.json` | — |
| k_mdot | 0.697 (v3 sampler's value) | **Jointly unidentified (ridge)** with φ_idle / φ_app — Phase 5 F2. The earlier reading "identifiable; fuel flow ∝ m_dot" was wrong: fuel flow ∝ φ·x^k_mdot, so idle and approach give two equations in three unknowns. | same | — |
| p_loss (combustor pressure loss) | 0.0340 (v3 sampler's value) | Consumed by the cycle (p_comb = p3(1−p_loss)) but **unidentified by the fuel-flow objective** (zero fuel-flow sensitivity; `scripts/validation/heat_loss_provenance.md` §5). | `run_full_cycle` | — |
| Per-mode pi-scales | — | **DELETED** (replaced by part-power law) | — | — |
| Climb calibration target (2.050 kg/s) | — | **DELETED** (untraceable; CSV has no CLIMB rows) | — | — |
| FPR (fan pressure ratio) | 1.45 rated; part power 1+(0.45)x^k_pi | Fixed, design-class value | `simulation/fan.py`, `design_point['fpr']` | Standard civil high-BPR fan magnitude (Mattingly-class textbook value); not measured Trent 1000 data. |
| eta_fan | 0.90 | Fixed | `simulation/fan.py` | Standard fan isentropic efficiency magnitude; not engine-specific. |
| BPR | 9.1 | Fixed, **now consumed** (bypass stream + fan work + two-stream thrust) | `run_full_cycle` | Trent 1000-AE3, ICAO CSV. |
| Heat-loss fraction xi | 0.0 (production default) | Optional hook; sweep in `outputs/heat_loss_sensitivity.csv` | `Combustor.run(heat_loss_fraction=...)` | xi=4% shifts TSFC by 0.059 mg/(N·s) — MORE than the full blend-to-blend spread (0.052) — and T4 by 45 K: heat-loss treatment bounds the resolvable blend effect size (decision rule AB6). eta_comb is a **lumped heat-delivery efficiency**. |

## Calibration / validation status (supersedes Phase 1 numbers)

| Version | Model | In-sample error | Held-out MAPE (excl. AE3) | Take-off mode |
|---|---|---|---|---|
| v1 (Phase 1) | free per-mode scales, rated OPR everywhere, no fan, argon-T3 | 5.46% | 7.27% | 16.5% (systematic over) |
| v2 | part-power law + p_loss, no fan | 12.98% | 14.64% | 18.8% |
| v3 (production) | + fan/bypass | 10.11% | **12.71%** | 19.7% |

The Phase 1 hypothesis that the take-off bias stemmed from the missing
part-power OPR is **falsified**: constraining the throttle physics made
fuel-flow generalization worse, not better (a single exponent cannot match
both the idle and approach airflow ratios that the free scales fit). All
versions remain within the ≤20% genuine-validation rule. Fuel-flow results
are insensitive to the argon fix and to the fan (FAR is phi-only).

## Emissions (supersedes Phase 1 rows)

| Parameter | Value | Status | Source / notes |
|---|---|---|---|
| EI-CO2 (combustion) | 3.664 × w_C kg/kg (3.100 for Jet-A1 surrogate) | Computed from composition | Replaces flat 3.16 (−1.9%); Cantera equilibrium cross-check 3.098 kg/kg. |
| Lifecycle CO2e | L_CEF × LHV_i × m_dot_f, per component | CORSIA basis | Every L_CEF cited to an ICAO Doc 06 (8th ed., Nov 2025) table row in `data/corsia_lca_values.yaml`; fossil baseline 89 gCO2e/MJ. Old point factors {0.2, 0.1, 0.3} sat near best-case and are retired. Triangular ranges: HEFA 13.9/28.6/74.0, FT −3.4/7.7/21.1, ATJ 18.2/29.3/81.4 gCO2e/MJ. |
| NOx | ICAO-derived correlation (labeled proxy) + Zeldovich post-processor (`simulation/nox_chemistry.py`, Turns rate constants) + HyChem-A2 kinetic anchor (`data/A2NOx.yaml`) | Three-path comparison | Path spread: orders of magnitude at idle/approach, ~7x at take-off (`outputs/nox_dual_path.csv`). **Blend NOx ranking is not supportable.** Combustor volume for residence time: A_exit(0.207 m²)×0.5 m assumed length → tau 5–9 ms. |

## PINN components (Phase 2.5)

Ablation at the calibrated take-off point (`outputs/ablation_pinn_components.csv`):
turbine-PINN −10% total thrust, nozzle-PINN +10% (core-stream deviations to
37%), partial cancellation in the production pinn/pinn combo (−1.6% vs
all-analytic). T5 and fuel flow identical by construction (work-matched).
Sajben external validation FAILS for both checkpoints (wall-Cp shape-L2
0.71–1.09 vs <0.10 threshold; `outputs/sajben_validation_errors.csv`) — the
PINNs are physics-consistency surrogate layers, not validated accuracy
contributors; the ±10% model-choice sensitivity must be disclosed.

---

# Phase 3 updates (2026-07-14) — production freeze

## Component configuration (adjudicated, P3.1)

Production: **turbine = analytic, nozzle = analytic** (now the code
defaults). Evidence: the analytic turbine p5 reproduces the hand-verified
work-consistent polytropic value exactly (4.728 bar at the take-off point);
the turbine PINN's p5 is a raw network output at −41.5% from work
consistency; the LE-PINN nozzle failed Sajben. The ±10% thrust spread
between configurations is disclosed as model-choice sensitivity
(`outputs/turbine_p5_adjudication.csv`, `outputs/ablation_pinn_components.csv`).

## New parameter (P3.4)

| Parameter | Value | Status | Source |
|---|---|---|---|
| Combustor air fraction β | **0.8** | Fixed, sourced | Lefebvre & Ballal, *Gas Turbine Combustion*: ~20–30% of combustor air is liner cooling + dilution. Only β·ṁ_core burns at φ; the rest remixes before the turbine (enthalpy balance). Resolves the take-off fuel-flow bias (+19.7% → +1.8%) and drops take-off T4 to 1895 K (realistic TIT). |

## Calibration/validation table (final)

| Version | Model | In-sample | Held-out MAPE (excl. AE3) | Take-off |
|---|---|---|---|---|
| v1 | free scales, inert params, no fan | 5.46% | 7.27% | +16.5% |
| v2 | part-power + p_loss | 12.98% | 14.64% | +18.8% |
| v3 | + fan/bypass | 10.11% | 12.71% | +19.7% |
| **v4 (adopted)** | + airflow split β=0.8 | **1.62%** | **2.50%** | **1.81%** |

v4 values returned by the sampler: η_comb 0.9963, p_loss 0.0442, k_π 0.5618,
k_ṁ 0.6218, φ = {0.2953, 0.3053, 0.5409}. Only φ_to = 0.5409 is identified by
the objective; η_comb, p_loss and k_π are unidentified and k_ṁ / φ_idle /
φ_app lie on a ridge (Phase 5 section). φ_app sits off its search bound, but
that is the box selecting a point on the ridge, not the data. Bias-hypothesis
history: part-power OPR falsified (P2.1); airflow split confirmed (P3.4).

Canonical number sources: `outputs/ARTIFACT_MANIFEST.md`. Old→new mapping:
`docs/number_crosswalk.md`.

---

# Phase 5 updates (2026-09-25) — identifiability of the LTO calibration

Supersedes every "calibrated" / "converged" status above for the LTO
calibration parameters of the part-power formulation (v2–v4; the v1 rows
are scoped in the Phase 1 table). Source: `docs/plan.md` (Phase 5), finding F2, and
`scripts/validation/heat_loss_provenance.md` §5.

Fuel flow per mode is φ·f_st·β·ṁ_rated·x^k_mdot. Take-off (x = 1) pins φ_to;
idle and approach give two equations in three unknowns (φ_idle, φ_app,
k_mdot). Moving along that ridge with every other parameter at v4
(measured 2026-09-26, F2):

| k_mdot | φ_idle | φ_app | fuel-flow MAPE | idle thrust | approach thrust |
|---|---|---|---|---|---|
| 0.5800 | 0.2643 | 0.2903 | 1.619 % | 28.56 kN | 84.44 kN |
| 0.6000 | 0.2787 | 0.2974 | 1.619 % | 27.58 kN | 83.30 kN |
| 0.6218 (v4) | 0.2953 | 0.3053 | 1.619 % | 26.53 kN | 82.04 kN |

Identical objective, 8 % spread in idle thrust; the only thing selecting v4's
point is the φ search box.

| Parameter (v4 value) | Status under the v4 fuel-flow objective |
|---|---|
| φ_to (0.5409) | **Identified** (structurally: take-off, x = 1, pins it through its own fuel-flow target). 0.5409 is the sampler's returned value; identifiability does not make it the exact optimum — see the P5.2 profile |
| η_b / η_comb (0.9963) | **Unidentified by the v4 objective** — zero fuel-flow sensitivity |
| pressure_loss (0.0442) | **Unidentified by the v4 objective** — zero fuel-flow sensitivity |
| k_pi (0.5618) | **Unidentified by the v4 objective** — zero fuel-flow sensitivity; swings approach thrust 57 → 84 kN across its range |
| k_mdot (0.6218) | **Jointly unidentified (ridge)** with φ_idle, φ_app |
| φ_idle (0.2953) | **Jointly unidentified (ridge)** with k_mdot, φ_app |
| φ_app (0.3053) | **Jointly unidentified (ridge)** with k_mdot, φ_idle |

Every T4, thrust and TSFC derived from v4 (and v2–v3) depends on the six
unidentified values; v1's depend on its unidentified η_b and its hand-set
airflow scales. The repair is Phase 5 P5.2 (calibration v5).

---

# Phase 6 updates (2026-09-26)

## Finding F-A — the v2–v4 held-out fuel-flow validation is a rescaling rule

`scripts/validation/holdout_icao_validation.py` sets each held-out engine's
core airflow to `base_airflow × thrust_ratio`, where `thrust_ratio` is that
engine's ICAO rated thrust over AE3's. With φ fixed per mode, fuel flow is
φ·f_st·β·ṁ_core·x^k_mdot, so OPR and BPR never enter the prediction. On the
frozen `outputs/holdout_icao_validation_v4.csv` (unchanged; archived in P6.6 to `outputs/archive/pre_phase6/`):

- Predicted fuel flow ÷ thrust ratio is constant to ~1e-16 within each mode
  (take-off 2.318207, approach 0.618863, idle 0.242230 kg/s; 59 rows each).
- A model-free baseline — AE3's ICAO fuel flow × thrust ratio — scores
  **3.21571920 %** MAPE against the model's **2.46016229 %** on all 177 rows,
  and **3.24190221 %** vs **2.50278929 %** on the 171-row subset that excludes
  every Trent 1000-AE3 record. At take-off the baseline is better
  (1.78 % vs 1.81 %, all rows).

The v4 held-out MAPE (manifest M2) therefore measures whether ICAO fuel flow is
proportional to rated thrust within the Trent 1000 family — a property of the
data, not of the cycle model. It is **superseded** as validation evidence by
the Phase 6 P6.1 thrust-matched held-out test. Pinned by
`tests/test_holdout_informativeness.py` (the two v4 tests keep the historical
diagnosis; the v5 test requires the new predictions to depend on the cycle).


## Phase 6 P6.2 — fixed parameters under the range rule (v5)

Decision 2 (approved 2026-09-26): a fixed engine parameter needs a cited range,
a central value inside it, and its effect reported as a band. Registration A1
(`docs/phase6_p61_registration.md` §3, `outputs/phase6/p61_registration.json`)
holds the sources with page/table references and PDF hashes. β (burner air
fraction) had no citable range for the implemented quantity and was **dropped**
(single-zone combustor); ξ = 0 by the structural argument. Surrogate
compositions stay **H/C-matched illustrative binaries**; their LHVs are computed
from CRECK thermo (`simulation/fuels.py` LHV_METHOD, tested).

One-at-a-time endpoints (other parameters central, calibration **re-fit**),
from `outputs/p62_bands_v5.json` (`scripts/validation/p62_parameter_bands.py`).
Columns give the value at the low / high end of each range.

| Parameter | Central | Range | Source | Held-out MAPE % | TO TSFC mg/(N·s) | TO T4 K | fitted W_ref kg/s |
|---|---|---|---|---|---|---|---|
| Combustor pressure loss | 0.045 | 0.04–0.05 | NASA/TM-2017-219501 p. 4; NASA/CR-2005-213657 Table 1; NASA/TM-2007-214690 p. 12 | 1.83 / 1.82 | 7.704 / 7.701 | 1719 / 1718 | 102.3 / 102.5 |
| η_c (isentropic) | 0.86 | 0.8225–0.8865 | polytropic 0.89–0.93 (NASA/TM-2017-219501 p. 3; NASA/CR-2005-213657 Table 1), constant-γ conversion at OPR 43.2 | 1.88 / 1.81 | 7.684 / 7.714 | 1734 / 1708 | 103.6 / 101.7 |
| Turbine η_poly | 0.90 | 0.90–0.92 | NASA/TM-2017-219501 pp. 3–4; NASA/CR-2005-213657 Table 1 | 1.82 / 1.82 | 7.703 / 7.723 | 1718 / 1728 | 102.4 / 101.3 |
| FPR rated | 1.45 | 1.3–1.7 | NASA/TM-2017-219501 Table 3 p. 12 | 1.89 / 1.91 | 7.732 / 7.592 | 1640 / 1780 | 115.1 / 93.0 |
| Fan e_poly (→ η_fan per draw) | 0.905 (η_fan 0.90) | 0.8961–0.97 | NASA/CR-2005-213657 Table 1; NASA/TM-2017-219501 p. 3 | 1.82 / 1.87 | 7.697 / 7.738 | 1716 / 1732 | 102.6 / 101.0 |
| η_b per mode (CO/HC proxy) | TAK 0.999894, APP 0.999834, IDL 0.998152 | calibration-group min–max | ICAO CO/HC of the calibration group (data-derived) | 1.82 / 1.82 | 7.703 / 7.703 | 1718 / 1718 | 102.4 / 102.4 |

Central: held-out MAPE 1.82 %, TSFC 7.703, T4 1718 K, W_ref 102.4 kg/s.
Joint Monte Carlo (64 seeded draws, uniform over every range, re-fit per draw),
P5–P95: held-out MAPE 1.82–1.91 %, TSFC 7.643–7.752 mg/(N·s),
T4 1667–1795 K, W_ref 91.8–111.7 kg/s. These are **range bands under
assumed uniform ranges, not confidence intervals**.

**Limiting assumption: rated FPR.** It moves fitted W_ref by 22 kg/s and T4 by
140 K across its range; no other parameter moves T4 by more than 26 K. It does
not change the P6.1 conclusion: held-out MAPE stays between B1 (1.08 %) and B0
(2.19 %) at every draw and endpoint.
