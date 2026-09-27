# Phase 7 — fuel representation and new LE-PINN study

Date: 2026-09-27. Branch `phase7`, baseline `a44c131` (from main `c847f5a`).
This implements the user's Phase 7 continuation instructions after the device
link dropped. P7.0 and P7.1 are already committed; preserve their outcomes.
Completed Phase 6 plan: `docs/plan_phase6_completed.md`. This plan authorizes
repo work and Mac-local study execution only; no manuscript edits, outreach,
remote publishing, merge or release tag. The turbine PINN remains retired.

## Objective

Recalibrate the thrust-matched cycle with the registered Dooley 2012 Jet A
surrogate (calibration **v6**), evaluate the requested mass-basis blends with
paired sensitivity draws, and run a separately registered Sajben LE-PINN
study of the corrected RANS formulation and data efficiency. Report every
registered outcome, including failures. Phase number 7 and calibration version
6 are distinct; keep artifact names unambiguous.

## Constraints

- Register and commit each procedure before its first computation/training run.
  The user supplied preliminary blend values as diagnostics, not registered
  results. State that these were known when the P7.3 claim rule was specified.
- Preserve v1-v5 outputs, all old model weights and the P7.0/P7.1 registration
  and results. Chemical mechanism YAML is read-only. Never overwrite an old
  checkpoint. Use new write-once names and checkpoint/completion hashes.
- Work through the Mac executor, never rely on a suspended device VM. All long
  jobs must survive link loss: detached supervisor, caffeinate, durable logs,
  PID/start-time records, exit codes and verified completion markers.
- Do not automatically promote a new PINN to the production cycle. This is a
  new study, not P4.3 attempt 4. Historical negative results remain in the record.
- Keep the Jet A hydrogen/aromatics failures visible. The published surrogate
  targets POSF4658; the P7.1 comparison target is POSF10325. A published
  composition is not proof of matching this other fuel. The two-surrogate
  difference is a sensitivity measure, not a complete uncertainty bound.
- Mass blend percentages are not volumetric certification limits. Neat SAF is
  contextual. Do not imply operational approval of any blend from this study.
- No held-out target, experimental wall-Cp or preliminary PINN score may choose
  hyperparameters, splits, stopping rules or winning runs after registration.
- No incomplete batch can be reported as a terminal study result.

## Repo Context

P7.0 (`4b8eeda`) is `data/fuel_properties_v7.yaml`: property targets, surrogates,
tolerances and nvPM validity. P7.1 (`a44c131`) added `simulation/fuels_v7.py`,
7 tests, and `outputs/phase7/p71_surrogate_check.{csv,md}`. Dooley2012 passes
LHV but fails H/aromatics; preserve those results. CAEP/11 nvPM correction is
excluded by P7-A1 and stays excluded. V5Model accepts composition dictionaries.

The Phase6 selected calibration is `outputs/calibration_v5_A2.json`. Its split,
A1 registration and A2 selection rule are in `outputs/phase6/`. V5 A1 passed,
A2/A3 failed (no demonstrated skill over both baselines), A4 passed. Do not
silently restate v5 as a predictive success.

WIND supplies one weak-shock Sajben S-A RANS field (rho,u,v,p,T,mu_l,mu_t), not
a multi-condition dataset and not measured Reynolds stresses or turbulent k.
`outputs/sajben_data_audit.md` and `docs/le_pinn_vs_ma2025.md` give the prior
limitations. The experiment remains an external scoring set only.

## Relevant Files

READ: `data/fuel_properties_v7.yaml`, `simulation/fuels_v7.py`,
`outputs/phase7/p71_surrogate_check.*`, `outputs/phase6/p61_registration.json`,
`outputs/phase6/p61_amendment_A2.json`, `outputs/calibration_v5_A2.json`,
`outputs/p62_bands_v5.csv`, `scripts/optimization/lto_v5.py`,
`scripts/optimization/blend_matched_thrust_v5.py`,
`simulation/nozzle/le_pinn.py`, `simulation/nozzle/wind_cff.py`,
`scripts/validation/sajben_split.py`, `scripts/validation/sajben_validation.py`,
`scripts/validation/train_sajben.py`, `docs/le_pinn_vs_ma2025.md`.

CREATE (supporting paths may be enumerated in each committed registration):
- `docs/phase7_registration.md`, `outputs/phase7/p72_registration.json`,
  `outputs/phase7/p73_registration.json`, `outputs/phase7/p74_registration.json`.
- `scripts/optimization/lto_v6.py`, `scripts/optimization/blend_matched_thrust_v6.py`.
- `simulation/nozzle/le_pinn_ma.py`, `scripts/validation/train_sajben_ma.py`,
  `scripts/validation/report_sajben_ma.py` and focused tests.
- `scripts/run_phase7.sh` (or a small Python supervisor with shell entry point),
  `outputs/phase7_execution_status.md`, per-job logs under `outputs/logs/phase7/`.
- New v6 outputs and new-study checkpoints with explicit version/study/seed/fraction.

MODIFY only as necessary while preserving old behavior:
`scripts/optimization/lto_v5.py` reusable helpers;
`scripts/build_manifest.py`, `docs/model_map.md`, `outputs/ARTIFACT_MANIFEST.md`,
`REPRODUCE.md`, new v5-to-v6 number crosswalk and integrity/provenance tests.
No old training entry point may change its historical default residual/weights.

## Implementation Phases

### P7.2 — register and launch v6 calibration first

Commit machine-readable procedure, inputs/hashes, outputs and commands before
running. Reuse the Phase6 split, four free parameters and bounds, objective,
weighting, fixed central values, optimizer settings and full budget. Fuel is
Dooley2012 composition from the frozen YAML. Preserve the fixed eta_b proxy
convention explicitly; do not silently change other physics while replacing fuel.

Candidates: (1) seeded TPE150 trials then least_squares polish at max_nfev100;
(2) polish from v5 A2 optimum with the same max_nfev100. Keep lower calibration
SSE; exact ties go to (1). Record both. Do not label candidate2 as a new pilot.
Then run the full A1 profile (17 points, nuisance max_nfev40, existing rule),
and report nuisance convergence/caps and grid resolution. A parameter cannot
be called identified because a numerical error was converted to a penalty.

Run held-out B0/B1 and A2-A4 with unchanged definitions/thresholds. Report all
three per mode, calibration airflow and v5-to-v6 changes. No held-out tuning.
If A1 fails or the existing B0 escalation triggers, record the result and stop
its dependent production study; independent LE-PINN work may continue.
The existing engine-model split must also protect any newly reported NOx fit.

### P7.3 — register and run matched-thrust blends

Jet A plus HEFA/FT/ATJ at 10,20,30,50 mass%, neat SAF as context. Compare at
AE3 ICAO take-off (primary), approach and idle; any climb calculation uses
85% rated thrust and is explicitly extrapolated (no climb data in the CSV).
For each point report fuel flow, TSFC, T4, phi, correlation NOx and lifecycle
CO2e with each method's limits.

Reuse the **64 fixed-parameter draws** from P6.2, with the **selected v6 fitted
parameters held fixed** for all draws/fuels. Do not carry over the per-draw v5
refitted parameters. Use common draws for paired differences. Label these as
conditional fixed-calibration sensitivity bands, not refit-conditioned P6.2
bands or statistical confidence intervals.

Before results, specify exact pair list and per-quantity denominators/spread:
SAF fractions vs Jet A, and like-fraction pathways against each other. Apply
user rule: central difference must be nonzero, at least **95%** of the 64
paired draw differences must agree in sign, and its absolute magnitude must
exceed the same-mode, same-quantity central Dooley2012-versus-Dooley2010
spread at unchanged v6 calibration. No claim for zero/equal differences or
out-of-domain quantities. Publish all pass/fail comparisons, including neat
context separately. Do not assume the diagnostic 0.44% spread survives v6.

Lifecycle: same CORSIA sources/scenario draws as P6.3, correct energy-weighted
mixture LCEF and component contributions `ff * sum(mass_i * LHV_i * LCEF_i)`.
Register the LHV basis (liquid-basis targets versus gas-phase cycle) explicitly;
never count the vaporization correction twice or hide the common dodecane
vaporization approximation for other species.

nvPM number: Brem relation only; H_ref13.8 and H_SAF15.30 mass%, linearly
mass-blended. Enforce F>30% and deltaH<0.6 strictly; 50% and neat SAF exceed
that hydrogen interval and get an explicit unavailable status, not a number.
Report take-off and optional85% climb as extrapolated-engine-condition
screening, not validated engine nvPM. No approach/idle use, no fabricated
absolute nvPM EI without a measured baseline, no CO2-to-nvPM equivalence.

### P7.4 — new LE-PINN study: register and test before training

Design: fractions {2,5,10,25,100}% x {physics,data-only} x seeds{42,43,44}:
30 runs. Declare exactly what the denominator of '100%' is while retaining a
separate held-out WIND set. Commit row IDs, split/fraction RNG seeds and
hashes, nested subsets per seed, shared subsets/init/architecture/budget per
paired arm, training-only scalers and all hyperparameters before training.
Use a fixed held-out WIND test set, independent validation for selection only
if registered, and never use experimental taps to select/checkpoint/stop.
Register epochs, optimizer, LR schedule, collocation rule, boundary-point rule,
normalization, checkpoint selection and restart/resume policy before running.
Prefer established repo budgets unless a compute-only benchmark justifies a
change; record it before any performance comparison. No optional early stopping
chosen after observing which arm wins.

**Data-access decision:** a concise question has been sent to the user about
viscosity access. Record the actual answer before the P7.4 registration. If no
answer arrives after a reasonable opportunity, use the stated recommended
assumption: every WIND field label, including viscosity, is available only at
the sampled training points. Do not provide full held-out viscosity/derivatives
to physics alone and call it a sparse-data comparison. Training-only learned
or differentiable interpolated viscosity is permissible if registered with its
information budget and tested; use identical information access in both arms.
Geometry and physical wall conditions are not held-out flow measurements.

Physics must be written as explicit equations in the registration:
- Planar, compressible conservative momentum and energy fluxes with variable
  viscosity and correct chain-rule derivatives in physical coordinates.
- Boussinesq deviatoric Reynolds stress using the WIND/learned eddy viscosity.
  Explain that S-A supplies no turbulent k or independent R_ij measurements.
  Do not count turbulent stress twice: either molecular viscous flux plus
  explicit Boussinesq stress, or equivalent combined mu_eff flux.
- State heat conduction/turbulent Prandtl closure, dissipation, EOS and any
  unavailable isotropic-k approximation. Do not call those exact Ma replication.
- No-slip and adiabatic wall loss using the curved wall's actual normal.
- tanh; Ma Eqs30-33 current-loss weights, detached from loss gradients unless
  another convention is explicitly registered and justified. Exact forms:
  lambda_data=.1+.9*sigmoid((Lphys+Lbc-Ldata)/(Ldata+epsilon));
  lambda_phys=.1+.9*sigmoid((Ldata-Lphys)/(Lphys+epsilon));
  lambda_bc=.1+.9*sigmoid((Ldata-Lbc)/(Lbc+epsilon)).
  The existing audit's shorthand 'sum of other two' is not correct for all
  three equations; correct that wording with a precise citation.

Verify implementation on manufactured differentiable fields BEFORE training:
variable-mu gradient terms survive, constant/uniform limits hold, turbulence
is counted once, unit/coordinate scaling is correct, wall normal flux and
weight equations are correct, and withheld labels cannot enter training.
Full WIND-derived stress gradients cannot be covertly supplied as targets.

Scoring: experimental upper/lower wall-Cp shape-L2 with existing bands;
primary scalar = worse of the two walls. WIND held-out relative L2/RMSE per
rho,u,v,p,T plus declared aggregate, viscosity separately. Label in-case spatial
interpolation versus external experimental comparison; no cross-case claim.

Claim: for each fraction use three seed scores per arm; define seed spread
before training as max(sampleSD(physics),sampleSD(data-only)), ddof1. Physics
benefit requires mean(data-only primary score)-mean(physics primary score) to
exceed that spread at >=2 of {2,5,10}%. Report per-seed paired differences,
mean/SD, all five fractions and both kinds of evaluation even on failure.
Do not select a favorable wall, metric, seed or checkpoint after outcomes.

### P7.5 — runner, reporting and provenance

Build/test the durable runner early enough to launch independent calibration
and ML queues without oversubscribing the Mac. Commit numerical registrations
before their jobs start. Persist a frozen source revision/config hash per run;
active training must not import changing source. Use locked run directories,
atomic checkpoint saves, explicit completion markers with hashes, and a report
that rejects missing/failed runs. Preserve partial launch logs as evidence.

After prerequisites/tests pass, actually launch the Mac jobs; do not stop at
writing a script. Record PIDs, start times, source SHA, config/input hashes,
queue order, logs and commands to inspect/resume. Automatic continuation from
calibration to profile/holdout/blends and from 30 training runs to reporting
must enforce scientific and completeness gates. Infrastructure recovery may
resume exact registered jobs; never silently change seeds/budgets or rerun a
bad scientific outcome as if it had not happened.

Update manifest/model map/crosswalk with completed artifacts only. Pending
jobs must be visibly pending; never replace old completed evidence with empty
new files. Retain P7.1 failed checks and all old negative surrogate results.
If long jobs are still running at handoff, the runner must finish downstream
reports autonomously and keep status truthful; do not call Phase7 complete.

## File-Level Edits

- New v6 wrapper reuses tested v5 fit/profile/holdout primitives with explicit
  fuel/output/registration arguments; no global path monkeypatch that could
  overwrite v5 evidence. Add tests for fuel threading and output isolation.
- New blend wrapper uses `fuels_v7.mass_blend` and the specified frozen-v6 draw
  convention; test mass/energy accounting, domain exclusions and claim rule.
- New LE-PINN module/entry points preserve legacy defaults and scorers; tests
  enforce flux mathematics, leak-free splits and complete-run provenance.
- Runner and status own durable execution/completion. Manifest builder consumes
  completed summaries rather than stdout. New crosswalk states version/basis.

## Commands to Run

Executor: `bash scripts/run_claude_from_plan.sh --force-model opus` on the Mac.
Use `.venv/bin/python`. Before launch, write the exact numeric/training/report
commands in each registration and `outputs/phase7_execution_status.md`.
Run `.venv/bin/python -m pytest tests/ -v`,
`.venv/bin/python scripts/test_emissions.py`,
`.venv/bin/python scripts/validation/verify_protected_hashes.py` plus extended
v5/P7.0/P7.1 hashes, and relevant Cantera validation after physical-path edits.

## Tests

Existing suite floor: Phase6 190 passed/1 skipped + P7.1 seven tests.
Add only meaningful tests for new numerical contracts, reproducibility and
failure guards as specified above. Training smoke tests must be labeled
non-study implementation checks, use temporary outputs and cannot influence
registered hyperparameters via test-set scores. Do not run long studies in pytest.

## Acceptance Criteria

1. Registrations precede computations, v1-v5/P7.0/P7.1 preserved.
2. V6 uses the exact fuel, split, full budgets and candidate-selection rule;
   A1-A4 and model/B0/B1 reported without tuning against holdout.
3. P7.3 uses paired fixed-parameter draws and frozen v6 parameters; all claims
   pass the specified sign/spread rule or are explicitly rejected. nvPM domain
   exclusions and LHV bases are visible and tested.
4. New-study physics passes manufactured-field tests and has no privileged
   held-out labels; all30 registered runs are required for a terminal claim.
5. Long jobs execute durably on the Mac, with inspectable source/config/input
   provenance and incomplete-run guards. Completion is never inferred from PID
   disappearance or a wrapper's zero exit alone.
6. Tests/emissions/mechanisms/hashes pass; completed artifacts have manifest,
   crosswalk and model-map provenance. Running work is stated as running.

## Rollback Notes

Retain separate new artifact names and existing main/v5.0. Revert named Phase7
commits or reverse archive moves if requested. No destructive reset, no deleting
failed-run evidence, no old-model overwrite. Leave unrelated `.DS_Store` untouched.

## Escalation Guidance

High complexity across calibration and differentiable RANS: Claude Opus executor.
Resolve ordinary defects directly within scope. Report a real calibration gate,
unavailable required data, or unresolved physical closure precisely. Continue
independent authorized work. New-study PINN failure is a result, not permission
for another unregistered attempt. Do not request blanket plan approval again;
the user's continuation specifies the work, and routine registration details
are made concrete here before execution.
