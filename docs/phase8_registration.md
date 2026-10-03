# Phase 8 registration (P8.0)

Registered 2026-09-29 on branch `phase8`, parent `7524a7a` (Phase 7 freeze),
**before** any Phase 8 computation (benchmark, parity run, calibration or
validation). The P8.0 cProfile runs after this commit. Plan: `docs/plan.md`. Changes to anything below need a numbered
amendment (P8-A1, P8-A2, …) committed before the computation it affects,
stating what was already known when it was written.

## 1. v6 baseline (frozen, from `outputs/phase7/`, hashes in the protected list)

| Quantity | Value | Artifact |
|---|---|---|
| Calibration fit (in-sample, 93 rows, 0 unreachable) | group-weighted MAPE 1.785 %, SSE 4.5708e-4 | `calibration_v6.json` |
| v6 parameters | W_ref 103.2852, a_thrust 1.10982, k_pi 1.34644, k_mdot 0.413812 | `calibration_v6.json` |
| Held-out fuel flow (87 rows, 9 groups, 0 unreachable), group-weighted MAPE | model 1.830 %, B0 2.189 %, B1 1.076 % | `holdout_icao_validation_summary_v6.csv` |
| Held-out by mode, model / B0 / B1 | take-off 1.720 / 2.045 / 0.579; approach 1.927 / 1.123 / 0.855; idle 1.843 / 3.397 / 1.795 | same |
| Gates | A1 FAIL (all four knobs penalty-dependent), A2 FAIL (no demonstrated skill), A3 FAIL (approach sign), A4 PASS | `identifiability_profile_v6.json`, `holdout_icao_validation_v6.json` |
| ICAO engines in repo | Trent 1000 family only, 28 variants, 180 rows | `data/icao_engine_data.csv` |
| AE3 take-off (design point, calibration engine) | FF 2.39211 kg/s, φ 0.338793, T4 1700.72 K at 310.9 kN | `calibration_v6_rows.csv` |

The pre-Phase-1 "~11 % MAPE" is an in-sample residual of the old manuscript
and is not a Phase 8 baseline. No P7.3 blend result exists (gate closed); the
preliminary −0.9 % blend fuel-flow value is a diagnostic only.

## 2. Parameter ledger (template; living copy `docs/phase8_parameter_ledger.md`)

One row per parameter, added in the same commit as the code that introduces it.

| Field | Content |
|---|---|
| name, symbol, unit | |
| component / ablation step | e.g. turbine / A2 |
| status | `fixed-cited` (range rule), `calibrated:<stage>:<dataset>`, `retired` |
| central value and range | with citation for each end |
| constraining dataset | required for `calibrated`; none → must be `fixed-cited` |
| identifiability verdict | A1 profile rule with penalty guard, artifact path, or `n/a (fixed)` |
| introduced / changed in | commit |

Seeded rows: W_ref, a_thrust, k_pi, k_mdot = `calibrated:v6:ICAO calibration
group`, verdict A1 FAIL (penalty-dependent); scheduled `retired` at A4.

## 3. Gates

A gate that fails stops the steps below it. The failure is recorded in
`outputs/phase8_execution_status.md` and is not tuned away.

**G0 — C++ parity (guards P8.2).** The C++ v6 path, called through the
Python backend flag with the frozen v6 parameters, fuel and `fixed_central`,
reproduces:
- every numeric column of the 93 rows of `calibration_v6_rows.csv` and the
  model columns of the 87 rows of `holdout_icao_validation_v6.csv`
  (ff, φ, thrust, TSFC, T3, T4, T5, p3, core/bypass thrust, NOx correlation);
- the AE3 take-off design point;
- status and reason strings exactly;

at relative tolerance 1e-9 (absolute 1e-12 for values that are exactly zero),
using the `scripts/reproduce_check.py` comparison rule. Per-component unit
tests (compressor/fan, burner, analytic turbine, nozzle, HP equilibrium) pass
at 1e-12 relative. If Brent's iterates differ, SciPy's `brentq` is ported
exactly; the tolerance is never loosened.

**G1 — each physics upgrade alone (guards the next upgrade).**
- P8.2 state + turbine: constant-cp limit reproduces v6 T5 and p5 to 1e-10
  relative; energy closure |Σṁh_in − Σṁh_out − W|/(ṁh) < 1e-10; the
  polytropic integral changes by < 1e-8 relative when the step count doubles
  from the registered count; element conservation across mixing < 1e-12.
- P8.3 nozzles: with p_amb ≥ p* and Cd = Cv = 1, reproduces the v6
  fully-expanded thrust to 1e-10 relative; ṁ and F are continuous at the
  critical pressure ratio (jump < 1e-8 relative across ±1e-9 in NPR); p*
  from max ρu agrees with the constant-γ value 2/(γ+1)^{γ/(γ−1)} to 1e-10 for
  a calorically perfect test gas.
- P8.4 and P8.5 checks as in the plan (pyCycle code-to-code; long-τ network
  returns HP equilibrium; element/energy balance < 1e-10). Their numerical
  tolerances are fixed by amendment before those steps start.

**G2 — calibrated model before synthetic data (guards P8.9).** All must hold:
1. Cross-family held-out set: engine fuel-flow group-weighted MAPE ≤ B1 MAPE
   − 0.25 pp (the Phase 6 A2 margin). B1 on this set is the registered F–A
   rule unchanged (`lto_v5.baselines`): nearest-rated-thrust group of the
   **whole calibration pool** (all calibration families), so it is a
   cross-family transfer. B0 is reported alongside.
2. v6 Trent held-out split: model MAPE ≤ 1.830 % (no worse than v6).
3. Each locked component source: χ²/N ≤ 2, and the share of observations
   inside the 95 % predictive interval is between 85 % and 99 %.
4. Combustor NOx: the claimed path follows D4.

**G3 — emulator before the optimizer (guards P8.13–P8.14).** The M4
emulator's error against the simulator is below 10 % of the validation
uncertainty σ_y at every scored output, on held-out simulator points and on
optimizer-visited points (95th percentile of |error|/σ_y ≤ 0.10).

## 4. Benchmark protocol (P8.1)

Machine: Apple M3 Pro (5 performance + 6 efficiency cores, 11 total), 18 GB,
macOS Darwin 25.6, on mains power, `caffeinate`, no other heavy jobs. Record
the git SHA, compiler and flags, Cantera/Python/NumPy versions.

Arms:
1. Python v6 as is (`lto_v5.V5Model` / `run_at_thrust`, unmodified).
2. Optimized Python: new module `scripts/phase8/v6_optimized.py`; changes
   chosen from the P8.0 cProfile before arm 2 is timed (for example reuse
   `Solution` objects instead of `_fresh_solutions()`, no printing in hot
   loops, cached cycle evaluations). Protected sources are not edited.
3. C++ core, 1 thread.
4. C++ core, all 11 cores (`std::thread` pool).
5. Reactor network, Python vs C++ (added once P8.5 exists; own amendment).

Workloads:
- W1: one AE3 take-off matched-thrust solve, `phi_guess=None`.
- W2: the 87-row held-out scoring at the frozen v6 parameters.
- W3: one calibration-objective evaluation (93 rows) at the frozen v6 parameters.
- W4: 1,000 scrambled Sobol points (SciPy `qmc.Sobol`, d = 4, seed 20260929)
  over the registered v6 box of (W_ref, a_thrust, k_pi, k_mdot), each solved
  at AE3 take-off. Unreachable points are timed and counted separately.

Python arms run W2–W4 both serial (1 worker) and parallel (11 workers).
Pool creation and one warm-up evaluation are excluded from timing; warm-start
guesses are cleared before each timed repeat (every repeat is a cold solve).

Measurement: median and range of 5 timed repeats after 1 warm-up
(`time.perf_counter`); solves per second; peak memory = parent `ru_maxrss` +
workers × `RUSAGE_CHILDREN` `ru_maxrss` (an upper bound, labelled as such).
A timing counts only if that arm's outputs match arm 1 at the G0 tolerance.
cProfile of arm 1 (P8.0) is published with the report. If arm 2 closes most of
the gap to arm 3, the report and the paper say so.

## 5. Protected list

`outputs/phase8/protected_sha256_phase8.json`, generated once by
`scripts/phase8/freeze_protected_phase8.py` in this commit and never
regenerated. Contents: the Phase 7 list; every committed `outputs/phase7/`
artifact (v6 outputs, P7.2/P7.3 registrations, run records); the Python v6
source path (`integrated_engine.py`, all tracked `simulation/**/*.py`,
`scripts/optimization/lto_v5.py`, `lto_v6.py`). Verify with
`verify_protected_hashes.py --phase8`.

## Amendment P8-A1 (2026-09-29, before any cross-family data or benchmark)

Requested by Arnav after reviewing the registration. Known when written: the
v6 baselines of section 1 and the P8.0 profile (97 % of a steady solve is
Cantera HP equilibrium; 25 cycle evaluations per solve). No ICAO data beyond
`data/icao_engine_data.csv` has been downloaded, and no benchmark arm has run.

### A1.1 G2 criterion 1 replaced (cross-family engine fuel flow)

Baselines, all fitted on the calibration pool only:
- B0 constant TSFC and B1 F–A rule as registered (B1 over the whole
  calibration pool, confirmed by Arnav).
- **B2 (new)**: per mode m, weighted least squares
  TSFC_i = β0,m + β1,m·OPR_i + β2,m·BPR_i over the calibration-pool rows of
  that mode, weights = the registered group-balanced row weights w_i
  (`lto_v5.attach_groups`), TSFC = ICAO fuel flow / target thrust. Prediction:
  FF = TSFC_hat × target thrust. Linear terms only, no transformation.

Rule, for each b in {B1, B2}: let Δ_b = MAPE_b − MAPE_model (group-weighted,
percentage points) on the cross-family held-out set. G2.1 passes only if, for
both B1 and B2:
1. Δ_b ≥ 0.25 pp (point estimate; the margin is a minimum), and
2. the 95 % percentile interval of Δ_b from a **paired cluster bootstrap**
   excludes 0 (lower bound > 0). Bootstrap: 10,000 replicates, NumPy
   `default_rng(20260929)`; each replicate resamples the held-out groups with
   replacement (the same draw for the model and every baseline); within a
   replicate each drawn group has equal weight and records keep their
   registered within-group weights.

B0, B1 and B2 MAPE and all Δ intervals are reported for the cross-family set
and, for information, the v6 Trent held-out set (already open since Phase 6).
G2 criteria 2–4 are unchanged.

### A1.2 Benchmark variants added (section 4)

Arms 2, 3 and 4 are each timed in three variants:
- **a — identical evaluation sequence.** Same cycle evaluations as arm 1. G0.
- **b — fewer cycle evaluations per solve.** Any change to how the matched-
  thrust solve brackets and iterates φ (for example a tighter initial bracket,
  a secant/Newton-started Brent, removing repeated evaluations at the same φ,
  reusing the T4-guard evaluations) with the cycle function unchanged and the
  same xtol/rtol. Warm starts across rows are not used (cold protocol of
  section 4). **Must pass G0** on all 180 rows and the AE3 point; a variant
  that fails G0 is reported and its timing is not counted. Cycle evaluations
  per solve are reported for every arm.
- **c — b + products-only equilibrium.** The HP equilibrium is solved on a
  products-only species set taken from the same CRECK thermo
  (`data/creck_c1c16_full.yaml`): N2, O2, AR, CO2, H2O, CO, H2, OH, H, O,
  HO2, H2O2 (12 species; CRECK has no other nitrogen species). The inlet
  mixture enthalpy and element composition come from the full mechanism at
  (T_in, p); the products-only mixture is set to that enthalpy, pressure and
  element composition and equilibrated at HP. Everything downstream is
  unchanged. **Outside G0**, with its own tolerance vs the frozen v6 rows: on
  all 180 rows and the AE3 point, |ΔFF|/FF ≤ 1e-4 and |ΔT4| ≤ 0.1 K, and the
  held-out group-weighted MAPE changes by ≤ 0.01 pp. A variant c result is
  reported as a separate approximation, never as the v6 path.

### A1.3 Toolchain facts recorded (no scientific change)

`xcode-select` points to a broken `/Applications/Xcode.app`, and the default
Command Line Tools SDK (MacOSX27.0) cannot be linked by the installed linker.
Builds use `DEVELOPER_DIR=/Library/Developer/CommandLineTools` and
`CMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`
(Apple clang 21.0.0). Record the compiler and SDK in every benchmark report.

## Registration P8-R1: empirical database vocabulary v1 (2026-09-29, before any source is entered)

- Schema `data/empirical/schema.sql`, sha256 `de9e1e780c660e64afb3d8f667dd3f1545bf7cd80483293ef72602d0f63b9418`.
- Vocabulary `data/empirical/vocabulary.yaml` (v1), sha256 `f9edeb8d0c29034cfb36e72f95f27f5176ddfc0d5031bca7977afa368df6ef51`:
  quantities with SI units, unit conversions and plausibility ranges;
  fidelity tiers 0–4; quality classes A/B/C; sigma kinds; the component
  stopping rule (≥ 3 independent calibration sources and ≥ 1 locked
  validation source, independence = distinct `independence_key`).
- Enforced in the schema: an `experiment` row is tier 4; validation and
  locked-test splits accept tier 4 only. The database refuses to open if the
  schema or vocabulary hash differs from the one it was built with.
- Not yet set, each by amendment before it is used: σ_class(tier, quality)
  values (before P8.7), the acquisition close date, and every split.
- QA tolerances (checks for hand inspection, not gates): corrected flow/speed
  and PR/NPR consistency 0.5 %; cold-air energy balance 2 % against dry-air
  h(T) from Cantera `air.yaml`.

### Amendment P8-R1a: vocabulary v2 (2026-09-29, before any source is entered)

Reading the first source (NASA CR-168189, Leach 1983) showed quantities that
v1 cannot hold without mislabelling them: the P&W speed parameter N/√T
(rpm/√°R), the flow function W√T/P (lbm·√°R/(s·psia)), an expansion ratio
whose total/static basis the report does not state, static-pressure reaction,
a cooled-turbine thermodynamic efficiency (Σṁ_iΔh_i / Σṁ_iΔh_i′ over primary,
coolant and leakage streams), and tip clearance. v2 adds `speed_param`,
`flow_function`, `PR_reported`, `reaction_p`, `eta_thermo_cooled`,
`tip_clearance` and their unit conversions. No v1 entry changed; no data had
been entered. `vocabulary.yaml` v2 sha256 `ba9a9258ebffbd821b11729bc7438369d8ed5588b7f65ce5622cc81e45d79a12`.

## Amendment P8-A2: ablation ladder (before the first P8.2 result)

Known at registration: G0 passed on all frozen v6 rows (`74f53c8`); the
Python v6 profile shows that equilibrium dominates solve time. No P8.2/P8.3
result, recalibration or upgraded-model held-out score has been run. This
amendment fixes the sequence and scoring rule; a failed G1 step remains
reported and stops dependent steps.

| Ladder step | Component change from the preceding step |
|---|---|
| A0 | Frozen v6 C++ parity path (G0 reference). |
| A1 | Composition-carrying GasState, liquid-fuel enthalpy and enthalpy dilution mixing (P8.2). |
| A2 | Enthalpy polytropic HP/IP/LP turbine and enthalpy cooling-air re-entry (P8.2). |
| A3 | Fixed-area separate core/bypass convergent nozzles with real-gas choking (P8.3). |
| A4 | Off-design shaft/map matching (P8.4, after its plan revision and G1). |

At **each** A1–A4 step, refit exactly the four v6 knobs (`W_ref`,
`a_thrust`, `k_pi`, `k_mdot`) with the frozen P7.2/v6 calibration procedure in
`scripts/optimization/lto_v6.py`: same initial values, bounds, seed, penalty,
mode/group weights, fuel, fixed settings and 93 calibration rows. The single
change is the named C++ physics step. Save the calibration run and parameters
as new, write-once Phase 8 artifacts with the source commit, config and seed.
The 87 Trent held-out rows remain outside fitting, warm-start choice and
component selection. After a step's fitted parameters and G1 result are
committed, score those rows **once** using the frozen P7.2 held-out procedure
and report group-weighted MAPE, mode-wise MAPE, B0/B1 comparisons, signs and
unreachable count. Always publish A2 and A3 scores even if either worsens the
v6 result. Do not select a ladder step using the held-out score.

The G1 limit/closure checks in section 3 are independent of calibration.
Where a new parameter lacks a constraining source, use a fixed cited central
value with a cited range under the parameter-ledger range rule; do not fit it
to these held-out rows.

## Amendment P8-R1b: empirical vocabulary v3 (before new source entry)

Known when written: the first source, NASA CR-168189, is in the database;
NASA CR-168289 (GE E3 HPT), CR-168290 (GE E3 LPT), CR-168069 (GE E3 core)
and NASA TN D-6967 have been inspected for their table headings and axis
definitions. None of their measured rows has been entered or used in a fit.
The schema is unchanged. Vocabulary v3 adds four quantities needed to record
the source definitions without treating unlike measurements as equivalent:

- `FAR`: measured fuel/air mass ratio in the core report's Table XIX;
- `eta_chemical`: efficiency derived from gas chemical analysis in that table,
  distinct from thermal cycle or turbine efficiency;
- `PR_ts`: turbine inlet total to exit static pressure ratio in the LPT report
  and the TN D-6967 plots, distinct from total-to-total `PR`;
- `torque_corr`: the TN D-6967 plotted torque corrected by its report's
  temperature/pressure factors, distinct from uncorrected shaft torque.

The first three use existing ratio units; `torque_corr` uses the existing
torque units and requires its correction basis in the experiment notes.
Their SI plausibility intervals are in `data/empirical/vocabulary.yaml`.
The v3 file SHA-256 is
`126a49c71aab45e80f62ad3d1c554dd99eef03287188852ab199d2ba0d14f7aa`.
The SQLite file is derived from source-entry modules: rebuild it after this
amendment, recheck the source-1 row count and QA, and record the new database
hash in the source-entry commit. No locked split or calibration has been run.

## P8-A2 label correction (2026-10-02)

Only the label of amendment P8-A2 changed to "ablation ladder". Its registered
content, procedures, scoring rules, gates and acceptance criteria are unchanged.
