# P8.4c prospective registration: ladder A4c (shared-parameter Trent model)

Date: 2026-10-03. User decisions: 2026-10-02. Executable protocol:
`docs/phase8_p84c_registration.json` (authoritative for every number, path
and hash). Parents: P8.4b and P8.4b-A1, P8.3-A2 (`ca3f8d6`), the P8.0 A1
rule and ledger, P8-R2 and the frozen P7.2 objective and held-out rules.

This file is committed **before** any A4c code exists and before any fit,
profile, fallback, error-budget, input-convergence solve or held-out
score. A4c is a new prospective model. It is not a corrected A4 score.
The A4 record, `scripts/phase8/trent_p84b.py`, the P8.3 and P8.4b
registrations and every A4 output stay exactly as committed.

## Known when written

- A4 was scored once (`2dd705f`): **21.24 %** held-out group-weighted MAPE
  (v6 1.830, A1 1.835, B0 2.189, B1 1.076). Six held-out rows were
  unreachable because both Trent 1000-E records (`11RR052`, `12RR058`)
  failed at the design point from the single registered guess. Converged
  rows over-predict by **+15.4 / +10.3 / +8.3 %** (take-off, approach,
  idle).
- These are committed aggregates. They are never optimization, selection
  or stopping targets. No A4 held-out row was opened for this registration.
- AE3 (calibration engine) A4 central fuel flows: 2.700 / 0.693 / 0.255 kg/s
  (`p84b_a1_g1.json`).

## 1. Model

Physics is the P8.4b three-shaft cycle with the P8.4b-A1 handling-bleed
rule, unchanged. `trent_p84b.make_engine`, `trent_p84b.offdesign_point`
and the C++ solver are reused without edits. A4c differs from A4 in three
ways only:

1. The nozzles use P8.3-A2: Cd = 0.96 and Cv = 0.985 for core and bypass.
2. Five **shared** parameters are calibrated:

| Parameter | Bounds | Initial (cited central) | Applies to | Central source | Bounds source |
|---|---|---|---|---|---|
| `T4_K` | 1700–1900 K | 1800 | design T4 | Martinez, *Aerospace engine data*: Trent XWB max TET (sister engine) | P8.4b engineering range |
| `FPR` | 1.40–1.55 | 1.45 | design FPR | frozen v6 `fpr_rated` | 1.55 Martinez table; 1.40 engineering choice |
| `eff_comp_offset` | −0.02–0.02 | 0 | added to fan 0.8948, IPC 0.9243, HPC 0.8707 | pyCycle 4.4.0 HBTF generic values | chosen ±0.02 (P8.4b) |
| `eff_turb_offset` | −0.02–0.02 | 0 | added to HPT 0.8888, IPT 0.8888, LPT 0.8996 | pyCycle 4.4.0 HBTF generic values | chosen ±0.02 (P8.4b) |
| `cooling_scale` | 0–1 | 1 | multiplies NGV 0.0641 and rotor 0.0275 | NASA CR-168189 §3.2.3 design (other machine) | 0 = no-cooling limit, 1 = cited |

   The cited centrals are generic or sister-engine values. The bounds are
   declared engineering endpoints, not measured intervals. No per-engine
   knob exists, so no certification-year surrogate is needed. Certification
   year may enter only if cited metadata is registered prospectively later.
   ICAO test dates are never substituted for it. Public OPR, BPR and rated
   thrust stay physical inputs.
3. **Input-only design initialization.** Nine ordered starts are tried:
   `W = rated_N / d` for d in (280, 220, 340) and, inside that,
   FAR in (0.03, 0.02, 0.04). The existing turbine pressure-ratio guesses
   are used: (3.5, 2.0, 5.0) for three shafts and (3.0, 4.0) for two
   shafts. The first start that converges and is closure-valid is chosen,
   and every try is retained. The choice is **never** made on observed
   fuel flow or on the lowest fuel-flow error. The solver is unchanged:
   scaled residual < 1e-10 within 50 iterations, existing bounds.
   Closure-valid means: finite, positive unknowns, station states and
   geometry; mass and energy closure < 1e-8; element closure < 1e-10.
   The first start equals the A4 guess.

Every other input stays as registered for P8.4b: IPC share 8/14, duct and
burner losses, ram recovery, SMN floor 10 %, per-mode eta_b, fuel and maps.

## 2. Calibration data (Trent v6 calibration split only)

Without steering from the main session, the pool is the frozen 93-row Trent
calibration split (9 groups, effective N = 27), not the 12-family pool. The
reader opens only `outputs/phase7/calibration_v6_rows.csv` (hash
registered) and parses only these whitelisted columns: ID, model, mode,
power, x, OPR, BPR, rated and target thrust, group, weight, and ICAO fuel
flow. Old predictions are never parsed. Before any fit it asserts:

- exactly 93 rows;
- the record set equals `split_p61.json` `calibration_records`;
- 9 groups;
- the weights sum to 1;
- the file hash matches.

`lto_v5.load_rows(..., with_targets=True)` and `calibration_rows()` are
not used, because they decode held-out targets before filtering.

## 3. Fit (registered starts and budgets)

The objective is the existing group-balanced relative fuel-flow SSE.
Unreachable rows score e = 1. Any other exception aborts the run. There
are three candidates, evaluated in this order in one process:

1. Optuna TPE (seed 42, 150 trials, parameter order as in the table), then
   a bounded `least_squares` trf polish (max_nfev 100, x_scale = box
   widths, diff_step 1e-4). The value is min(polished, best trial), as in
   `lto_v5.fit`.
2. Polish from the cited centrals (100 evaluations).
3. Polish from the exact box midpoint (1800, 1.475, 0, 0, 0.5;
   100 evaluations).

The lowest calibration SSE is selected. Exact ties go to candidate 1, then
2, then 3. Every candidate, log and status is preserved. There is no extra
budget and no held-out-based choice.

## 4. A1 profile, guard and one fallback

The generic A1 profile uses 17 uniform grid points per parameter, a
40-evaluation inner cap, and the `lto_v5.profile` warm-start direction.
It computes `D = 27 ln(SSE_profile/SSE_min)` with threshold 3.841. A
parameter is IDENTIFIED only if the interval is interior at both ends
and its width is ≤ 50 % of the box. The guard is `lto_v6.apply_penalty_guard`
reused exactly, including unknown or unreachable counts at the deciding
points. The profile also reports capped inner fits, grid resolution,
conditioning and every failed profile.

**One prospective fallback pass.** If any parameter is not identified,
all unidentified parameters are fixed jointly at their cited centrals.
The remaining parameters are refitted with the same three starts and
budgets, then profiled once. There is no further adaptive search. If no
parameter remains, the result is reported as "all fixed, A1 n/a", not as a
fitted identifiability success. The primary A1 verdict is the A4c verdict.
An A1 FAIL stays a failure and G2 stays closed. The frozen model is the
primary fit if A1 passes. Otherwise it is the fallback fit (or the
all-central model).

## 5. AE3 error budget (calibration engine only)

The error budget covers AE3 `02P23RR126` at all three modes. It first
reproduces the old A4 central with `trent_p84b` itself and compares against
`p84b_a1_g1.json`, without touching old files. It then runs the A4c
builder at the old A4 inputs as a regression, and then the new central
prior. From the new central, one-at-a-time range endpoints are run for:

- Cv core, Cv bypass and joint Cv;
- Cd core, Cd bypass and joint Cd;
- T4 and FPR;
- each of the six efficiencies;
- NGV cooling and rotor cooling, separately.

Each case re-solves the design, freezes its geometry and solves its modes.
The report gives FF deltas, calibration error (pp), areas, W/FAR,
residuals, closure, extrapolation and every failure. Each response is also
normalized by the known +15.4/+10.3/+8.3 % aggregates as an illustrative
"share". This is **not** an additive causal allocation of held-out
errors. The shares are not claimed to sum to 100 %, and centrals are not
tuned by them. Output (write-once):
`outputs/phase8/a4_error_budget/20261003_attempt1/`.

## 6. Input convergence (20 families, design only)

This check covers a representative design scope, not all 89 records: the
lowest eligible UID of each of the 20 families in the frozen cross-family
split, plus both original Trent 1000-E input records (UIDs deduplicated).
It reuses the whitelisted databank reader with the xlsx hash and split hash
checked, and reads only input columns of `data/icao_engine_data.csv`.
Targets are never decoded. Eligibility does not change because of solves.

Architecture policy (public sources; details in the JSON):

- **Three shafts:** Trent 1000 (TCDS E.036), Trent 7000 (certified under
  E.036 as a Trent 1000 variant), Trent XWB (TCDS E.111), D-36 and D-436
  (secondary public descriptions). The Trent 8/14 IPC share is a declared
  proxy for D-36 and D-436.
- **Two shafts:** CF34-3/-8/-10, CF6-80, GE90, GEnx-1B, LEAP-1A/1B/1C and
  PW4000. The LPC share is the generic pyCycle example share.
- **Geared** (PW1100G/1200G/1400G/1500G/1900G): a two-shaft solve with a
  declared **lossless gearbox design-power proxy**. Hbtf has no gearbox
  field, and no geared off-design claim is made.
- **Limits of the existing two-shaft C++ path** (no C++ edit is
  authorized): pyCycle Cv nozzle without Cd or P8.3 choking, no eta_b
  scaling, US 1976 sea-level table. These are declared limits of a
  convergence check, not a model claim.

**Gate:** coverage = 20 families and **every** case, including both Trent
1000-E records, converges with a closure-valid design. No family is
skipped, a missing binary or databank is not a PASS, and convergence is
never inferred from initialization. **Known static risk (source reading,
not a result):** the C++ design bounds cap W at 453.6 kg/s (two shafts) and
1360.8 kg/s (three shafts). Large two-shaft families such as GE90,
GEnx-1B, CF6-80 and PW4000, and possibly Trent XWB, may need more flow. A
failure is recorded and not tuned. Write-once output:
`outputs/phase8/p84_input_convergence.json`, after the AC gates.

## 7. One-shot Trent held-out score (by 2026-10-15)

Before any held-out target is read, all of these must hold:

- this registration and P8.3-A2 are committed and clean;
- the A4c implementation is committed and clean;
- the fit, profile and any fallback records are committed and clean;
- the input-convergence record is committed and clean (its verdict is
  copied into the score);
- frozen parameter, source and input identities exist;
- the machine is on AC power and idle.

`outputs/phase8/ladder/A4c/heldout_score_reservation.json` is then created
atomically and write-once. It is retained even if scoring fails, and its
existence refuses any later score. The original 87 Trent held-out rows are
scored **once**. Metrics, baselines, unreachable handling and the scoring
rule are unchanged (`lto_v5.holdout_tables`). If scoring is not done by
Oct 15 it is reported as in progress, with no deadline-driven retuning.
The profile is reported even if it FAILs. A score of a frozen failed model
is diagnostic and never opens G2. No new cross-family held-out targets
are read. All A4c outputs go under `outputs/phase8/ladder/A4c/`.

## Scope notes

The P8-R2 P8.7 stages, including stage 4 (combustor fitting), stay intact.
A4c is an additive engine-level calibration that the user expressly
authorized. Runtime commands refuse without AC power, while another heavy
job runs, or when sources and registrations are uncommitted. The launch
order comes from the main workflow record, after the benchmark, A2 and
Track 4 AC chain.

## Relevant files

Create: `scripts/phase8/trent_p84c.py`, `a4c_profile.py`,
`a4_error_budget.py`, `public_engine_inputs.py`, `p84_input_convergence.py`,
`tests/test_phase8_p84c.py` and `tests/test_phase8_p84_input_convergence.py`.
Modify: `docs/phase8_parameter_ledger.md` (new dated rows only).
