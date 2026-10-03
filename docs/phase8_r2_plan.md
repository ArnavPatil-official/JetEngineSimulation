# Phase 8 revision P8-R2 — off-design, validation, emulation and optimization

Registered 2026-09-29 on `phase8`. This is an additive revision to
`docs/plan.md`; the P8.0 registration and P8-A1/P8-A2 in
`docs/phase8_registration.md` remain authoritative. The user's 2026-09-29
continuation instruction authorizes repo work and Mac-local computation in
the gate order below, without an approval stop between unblocked steps. This
document must be committed before the first P8.4 or later computation. Each
run-specific procedure and its input hashes must also be committed before
the first result of that procedure is computed.

Known when written: G0 passed on 93 calibration rows, 87 Trent held-out rows
and the AE3 design point (`outputs/phase8/g0_parity.json`, commit `74f53c8`);
the Python v6 profile attributes about 97% of a steady solve to the
combustor/Cantera calls. P8.1 benchmarking, P8.2 and P8.3 had not completed.
One NASA E3 HPT table is in the empirical database. The ICAO 03/2026
databank's identifier/design fields were inventoried; target columns have
not been decoded. A local pyCycle 4.4.0 HBTF reference is being prepared but
was uncommitted at registration.

The Claude design artifact linked from `docs/plan.md` was inaccessible to
this executor. This revision uses the user-supplied continuation instruction,
`docs/plan.md`, `docs/phase8_registration.md`, and the frozen Phase 6–7
registrations. Details absent from those sources are explicitly assigned to
pre-run registrations below; they must not be inferred after seeing results.

## Objective

Replace the four penalty-dependent v6 matching knobs with a design-point
and off-design, map-matched C++ cycle; test a chemical reactor-network route
for emissions; calibrate against independent component and engine evidence;
then, only after G2, create simulator data and test MLX residual/emulator
models before robust SAF optimization. Retain negative results and compare
against the frozen v6 and registered baselines.

## Constraints

- Gate order is G0 → G1 separately for P8.2, P8.3, P8.4 and P8.5 →
  data-qualified P8.7 calibration and P8.8 locked validation → G2 →
  P8.9–P8.12 → G3 → P8.13–P8.14 → P8.15. A failed gate is recorded and
  closes only dependent work. Independent data, infrastructure and model
  scaffolding may continue.
- Do not edit protected v6 sources, Phase 6–7 outputs, chemical mechanism
  YAML, `data/icao_engine_data.csv` or `models/*.pt`. Never overwrite a
  checkpoint or a completed Phase 8 output. Keep `.DS_Store` untouched.
- No simulated/CFD record enters empirical validation. Only Tier 4
  experiments may enter locked component validation. Use rigs from other
  machines to constrain dimensionless relations, never Trent-specific values.
- Commit exact splits, priors, bounds, stopping rules, seeds, training
  budgets, envelopes, metrics and claim rules before opening their target
  data or starting their first numerical run. Source, config, input and
  environment hashes travel with each write-once result. Durable Mac jobs
  have start/exit records, logs and completion markers.
- A new fitted parameter requires a constraining calibration dataset and
  the existing A1 profile/penalty guard. Otherwise use a cited central value
  and range. The four v6 knobs are retired by P8.4 rather than relabelled.
- No push, merge, tag, manuscript edit, outreach, remote publication, or
  system-setting change is authorized here.

## Repo Context

`cpp/catjet_core/v6_engine.*` is the G0 C++ reference; the original Python
v6 path is protected. P8.2 and P8.3 build the composition-carrying turbine
and fixed-area nozzle on the C++ core. `data/empirical/` has the schema,
vocabulary and source provenance; `scripts/phase8/icao_families.py` inventories
ICAO family identifiers without opening targets. The pyCycle HBTF example
is a code-to-code check target, not fitted data. `scripts/phase8/ml/` contains
early MLX/PyTorch scaffolding; it is not evidence that G2 has passed.

## Relevant Files

Read-only references: `integrated_engine.py`, `simulation/**` existing v6
files, `scripts/optimization/lto_v5.py`, `lto_v6.py`,
`outputs/phase7/calibration_v6.json`, `outputs/phase7/calibration_v6_rows.csv`,
`outputs/phase7/holdout_icao_validation_v6.csv`,
`outputs/phase6/split_p61.json`, `data/icao_engine_data.csv`,
`data/A2NOx.yaml`, `data/creck_c1c16_full.yaml`,
`data/fuel_properties_v7.yaml`, and the protected list.

Planned new paths (create a numbered amendment before changing the output
schema or scientific meaning of a path):

| Step | Source and registration files | Write-once evidence |
|---|---|---|
| P8.4 | `cpp/catjet_core/offdesign.hpp`, `offdesign.cpp`, `maps.hpp`, `maps.cpp`; `cpp/tests/test_offdesign.cpp`; `scripts/phase8/pycycle/compare_hbtf.py`; `docs/phase8_p84_registration.md` | `outputs/phase8/p84_design_point.json`, `p84_pycycle_comparison.json` |
| P8.5 | `cpp/catjet_core/reactor_network.hpp`, `reactor_network.cpp`; `cpp/tests/test_reactor_network.cpp`; `scripts/phase8/reactor_validation.py`; `docs/phase8_p85_registration.md` | `outputs/phase8/p85_mechanism_check.json`, `p85_g1.json` |
| P8.6/D3 | `scripts/phase8/empirical_db.py`, `wpd_import.py`, `icao_families.py`; `scripts/phase8/cross_family_split.py`; `data/empirical/DIGITISING.md`, `acquisition_log.md`; `docs/phase8_family_split.md` | `outputs/phase8/cross_family_split.json` and source QA records |
| P8.7–8 | `scripts/phase8/calibration.py`, `component_validation.py`, `engine_validation.py`, `baselines.py`, `family_bootstrap.py`; `docs/phase8_calibration_registration.md`; `docs/phase8_parameter_ledger.md`; `tests/test_phase8_calibration.py` | `outputs/phase8/p87_*`, `p88_*`, `g2_verdict.json` |
| P8.9–12 | `scripts/phase8/synthetic.py`, `active_learning.py`; `scripts/phase8/ml/spec.py`, `models_mlx.py`, `models_torch.py`, `train_mlx.py`, `parity.py`, `score64.py`; `docs/phase8_ml_registration.md`; `tests/test_phase8_ml.py` | `outputs/phase8/p89_*` through `p812_*`, `g3_verdict.json`; new checkpoints under `models/phase8/` |
| P8.13–15 | `scripts/phase8/uq.py`, `robust_optimize.py`, `freeze.py`; `docs/phase8_optimization_registration.md`; `tests/test_phase8_optimization.py`; `scripts/build_manifest.py` | `outputs/phase8/p813_*` through `p815_*`, `outputs/phase8_execution_status.md`, regenerated `outputs/ARTIFACT_MANIFEST.md` and `docs/model_map.md` |

The table names ownership and output families; run registrations enumerate
the exact write-once filenames and input hashes before computation.

## Implementation Phases

### P8.4 — design point and off-design matching (after P8.2 and P8.3 G1)

1. Register the architecture and station graph, two-shaft and three-shaft
   residual vectors, unknowns, design-point inputs, generic scaled-map
   sources/interpolation/extrapolation rules, solver bounds and initialisation,
   numerical tolerances, and matched pyCycle outputs before a solve. The
   registration must identify which quantities are genuinely comparable when
   the two programs use different thermochemistry.
2. Compute the design point with declared area, mass-flow, speed and map
   scaling. At off-design, use a bounded/damped Newton solve for shaft power,
   corrected flow/map compatibility, nozzle area/flow and target thrust.
   Publish residuals and unreachable reasons. Do not reintroduce W_ref,
   a_thrust, k_pi or k_mdot as free fit parameters.
3. Run the published pyCycle high-bypass turbofan design and off-design
   example in the pinned separate environment. Compare to the frozen
   upstream example outputs after explicit unit and closure translation;
   include design, full-power and part-power cases, station/shaft quantities,
   residuals and all discrepancies. The pyCycle reproduction itself is
   infrastructure evidence, not validation or calibration data.
4. G1 requires a converged mass, element and energy accounting audit, a
   finite and bounded solution over the registered checks, and the
   code-to-code comparison within the tolerance registered before the first
   P8.4 solve. A failed comparison is reported, not recalibrated against
   pyCycle. Run the previously registered ablation ladder, including A2 and
   A3 even where they worsen performance, under its own order and data guard.

### P8.5 — reactor network (may start after G0, independently of P8.4)

1. Register the network before its first numerical run: parallel primary
   PSRs with a Gaussian equivalence-ratio distribution (mean phi_pz and width
   sigma_phi), a quick-quench PSR with air injection, a lean PFR represented
   by a PSR chain, and final dilution. Use HyChem A2 + NOx
   (`data/A2NOx.yaml`). Specify quadrature, residence times, air splits,
   pressure loss, volume scale, thermal boundary, solver tolerance and
   admissible parameter ranges in `docs/phase8_p85_registration.md`.
2. Implement with C++ thread isolation and the Phase 8 static-Cantera,
   hidden-symbol pybind11 recipe. Compute eta_b from unburned CO/H2/UHC
   using a declared energy basis; report NOx and CO without deriving their
   values from the ICAO correlation.
3. G1 checks the registered long-residence-time HP-equilibrium limit and
   element/energy closure below 1e-10, plus a documented CRECK thermo
   consistency check and mechanism spread. A cross-mechanism difference is
   reported, not used to select the favorable mechanism. Do not calibrate
   until the relevant P8.7 registration and source stopping rule are met.

### P8.6 and D3 — database and family split (independent track)

Add the three-repeat WebPlotDigitizer import, exact figure instructions and
tabulated sources through the committed vocabulary and QA. Record dead ends.
Before reading any cross-family fuel-flow or emission target, commit a
design-only, seeded, thrust-class-stratified split containing at least eight
eligible held-out families and at least eight calibration families including
Trent 1000. Eligibility requires a supported turbofan architecture, OPR,
BPR, rated thrust, and in-production or recent certification status. Freeze
family IDs, rows, design-field hashes, class boundaries, seed and split hash.
Do not use the Trent v6 held-out rows to choose the cross-family split.
The design-only inventory currently suggests only about 15 eligible
direct-drive families, fewer than the required 16. Keep the split gate open
until at least 16 eligible supported families exist. A geared family must not
be silently counted as supported by the two-/three-shaft direct-drive solver;
including one requires a prospective architecture registration or a genuine
scientific scope decision. Do not read targets to solve the count problem.

### P8.7 — four-stage calibration (after P8.4 and P8.5 G1)

Register every stage's dataset/source IDs, split, parameter list, fixed
ranges, priors, likelihood, uncertainty classes, optimizer/MCMC settings,
stopping rule and output hash before fitting. The stages are ordered:

1. **Component relations:** fit only nondimensional map loss and nozzle
   discharge/velocity relations using independent rig calibration sources.
   A component without at least three independent calibration sources and
   one locked validation source remains fixed to cited ranges and is marked
   data-limited; rig values from other machines never become Trent constants.
2. **Design point:** infer only identifiable engine-specific geometric/map
   scale variables from calibration-pool design fields and non-held-out
   station evidence. Retire the v6 four-knob parameterization. Record the
   parameter ledger and A1 penalty guard.
3. **LTO/off-design:** fit remaining identifiable parameters on the
   registered engine calibration pool only, using the group-balanced
   objective. Preserve mode and family labels. No held-out family target is
   inspected for selection or stopping.
4. **Combustor:** fit phi_pz, sigma_phi, air splits and volume scale only where
   independent experimental data constrain each parameter. Run posterior
   sampling for estimable parameters and predictive bands; cite ranges for
   data-limited ones. D4: the network becomes the claimed NOx/CO path only if
   it beats the v6 ICAO-derived correlation on held-out engines under the
   registered paired rule. Otherwise report the reactor result as an
   unvalidated alternative and keep the correlation as the reference.

MCMC chains are reproducible from committed seeds/configs. Report
convergence, effective samples and prior sensitivity; a nonconverged chain
is not a posterior band. The exact numerical convergence threshold and
budget are committed in the stage registration before sampling.

### P8.8 — locked-source and engine validation; G2

Open each locked Tier 4 component source only after its split hash is
committed; score all predeclared outputs. Apply the existing G2 component
rule: chi-square per observation no greater than 2 and 95% predictive
interval coverage between 85% and 99%. Score the frozen Trent v6 split with
MAPE no worse than 1.830% and the cross-family held-out set against B0, B1
and B2. B1 uses the whole calibration pool; B2 is the registered per-mode
weighted TSFC regression on OPR and BPR, fitted on calibration data only.
G2.1 requires at least a 0.25 percentage-point improvement over **both** B1
and B2 and a positive lower bound in each paired 95% percentile bootstrap
interval. In this revision, the bootstrap clusters are **families** (user's
later explicit answer), resampled with replacement in 10,000 replicates
using NumPy `default_rng(20260929)`; the same family draw is used for model
and baselines, with existing within-family/group row weighting. Report the
Trent baselines for information. D4 and all other existing G2 clauses remain.
No G2 failure is hidden by the synthetic/emulator tracks.

### P8.9 — synthetic-data pyramid (after G2)

Commit a generation registration containing physical-driver variables,
joint sampling distribution, admissible envelope, exclusion rules, fidelity
levels, seeds and all split IDs/hashes. Sample physical drivers and feed the
simulator; never sample raw species mass fractions Y. Carry composition and
station-state consistency through each level. Separate training, validation
and locked simulator test points by design point/driver group so near
duplicates cannot leak. Keep simulated rows out of empirical validation.
Record each run's solver status, fidelity, source revision and rejection
reason. Stop generation outside the registered envelope instead of
extrapolating silently.

### P8.10 — M0–M4 models and MLX precision

M0 is the C++ physics solver, the reference and fallback. M1 is a residual
model trained on simulator-minus-M0 targets. M3 is the optional nozzle PINN
under a new Phase 8 registration; the Phase 7 P7.4 work-in-progress branch
is neither registration nor validation. M4 is the composed emulator tested
by G3. The precise M2 architecture and its comparison role are not in the
accessible plan/user text; define and commit them, with M1/M3/M4 exact input
and output contracts, losses, architectures, seeds, split and budget, before
training. No turbine PINN is restored.

The separate pinned `catjet-mlx` environment may build scaffolding before
G2. Its parity test loads the same weights in MLX and PyTorch and checks
outputs, loss and gradients within 1e-5 relative. Report precision and
device for every run. Scoring that determines a gate or scientific claim uses
the float64 CPU path. No simulator or empirical target trains any model
before G2. Empirical locked rows never choose a neural checkpoint.

### P8.11–P8.12 — active learning, emulator selection and G3

Register a fixed acquisition rule, candidate pool, batch size, seed, budget,
retraining schedule, uncertainty score, stop criterion and locked test set
before the first acquisition. Every simulator call joins the pyramid with
its fidelity and provenance. Compare M0–M4 on the same locked simulator
points and report all ablations, not only the winner. On held-out simulator
points and optimizer-visited points, G3 requires the M4 95th percentile of
absolute emulator error divided by validation uncertainty sigma_y to be at
most 0.10 for every scored output, as already registered. A failed G3 keeps
optimization on M0 only; it does not authorize emulator-guided claims.

### P8.13 — uncertainty table

Before uncertainty propagation, register the table's quantities, sources
and dependence assumptions. Separate experimental measurement,
digitisation, parameter/posterior, mechanism, fuel-surrogate, numerical and
emulator terms. Propagate shared draws across paired fuels and modes where
correlations are known; avoid double counting. Mark unsupported terms as
unbounded or data-limited instead of assigning artificial precision. Report
conditional bands distinctly from statistical confidence intervals.

### P8.14 — robust SAF optimization

Register objectives, constraints, candidate fuels/blend fractions,
certification-domain exclusions, uncertainty draws, dominance rule,
optimizer budget/seeds and holdout scoring before search. Use M4 only if G3
passes; otherwise use M0. Re-evaluate optimizer-selected points with M0 and
the registered uncertainty table. The existing Phase 7 claim rule governs
paired fuel comparisons: nonzero central difference, at least 61 of 64
paired fixed-parameter draw differences agreeing in sign, and magnitude
exceeding the same-mode, same-quantity central Dooley2012-versus-Dooley2010
spread, with every quantity in its declared domain. Report failed claims and
out-of-domain results. This is a screening study, not certification.

### P8.15 — freeze

Freeze-package outputs go under `outputs/freeze/`; the local freeze tag is
`freeze-2026-10-18`.

Freeze registrations, source/config/input and output hashes, gate verdicts,
negative results, model weights and run manifests. Regenerate the artifact
manifest and model map through `scripts/build_manifest.py`; never hand-edit
them. Publish a complete local execution-status table and reproducibility
commands. A failed gate is a recorded outcome, not a reason to rewrite the
gate after seeing data. No push, merge or release tag is part of this step.

## File-Level Edits

- `docs/plan.md`: link this revision; retain the original Slice 1 plan and its
  registration priority. `docs/phase8_r2_plan.md`: this committed plan.
- `cpp/catjet_core/{maps,offdesign,reactor_network}.*` and `cpp/CMakeLists.txt`:
  implement only their registered C++ contracts; preserve the hidden-symbol
  static-Cantera pybind11 recipe and G0 path.
- `scripts/phase8/pycycle/compare_hbtf.py`: read frozen upstream output,
  translate units and report station/solver differences. No fitting.
- `scripts/phase8/{calibration,component_validation,engine_validation,baselines,family_bootstrap}.py`:
  stage-controlled estimation and locked scoring, with refusal to open an
  unregistered target. `docs/phase8_parameter_ledger.md`: add one row per
  introduced/fitted/fixed/retired parameter.
- `scripts/phase8/{synthetic,active_learning}.py` and
  `scripts/phase8/ml/*`: generation, MLX training/parity, float64 scoring,
  checkpoint and test isolation. `models/phase8/`: unique run directories.
- `scripts/phase8/{uq,robust_optimize,freeze}.py`: reproducible uncertainty,
  search and freeze; `outputs/phase8_execution_status.md`: update after each
  step and gate; `scripts/build_manifest.py`: register completed artifacts.
- Each `tests/test_phase8_*.py`/C++ test verifies a new numerical or leakage
  contract. No long calibration, benchmark, generation or training in pytest.

## Commands to Run

Run each command from the repository root with a frozen source/config snapshot:

```sh
.venv/bin/python scripts/validation/verify_protected_hashes.py --phase7
.venv/bin/python scripts/validation/verify_protected_hashes.py --phase8
.venv/bin/python -m pytest tests/ -v
.venv/bin/python scripts/test_emissions.py
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
.venv/bin/python scripts/phase8/g0_parity.py
```

Run-specific P8.4–P8.15 commands are committed with each registration and
copied verbatim into durable run records. Do not put long studies in pytest.

## Tests

Floor before P8-R2: 245 passed, 1 skipped per the user's continuation note;
verify the live count and explain any difference from the earlier Phase 8
status floor (231 passed, 1 skipped at the Phase 7 freeze). Tests cover map
scaling/interpolation, Newton residual/limit behavior, pyCycle translation,
reactor long-time and closure limits, database QA and split leakage, B2 and
paired family bootstrap, calibration stage guards, MLX parity, deterministic
sampling, float64 scoring, G3 and domain exclusions. Test protected hashes
after each implementation phase and Cantera mechanisms after combustor or
emissions changes.

## Acceptance Criteria

| Step | Pass condition |
|---|---|
| Registration | This revision and every run-specific rule committed before its first computation; exact split hash committed before any held-out family target is opened. |
| P8.4 | Design/off-design solver satisfies its pre-registered residual/physics tolerances, retires four v6 knobs and passes the matched pyCycle code-to-code rule, or records G1 failure. |
| P8.5 | Registered HP limit, element/energy closure below 1e-10, CRECK consistency and mechanism spread reported; NOx/CO claim follows D4. |
| P8.6 | At least eight eligible supported families in each cross-family pool; if the design-only inventory is too small, keep the split/G2 cross-family branch open without reading targets. Tier 4/QA/source independence and digitisation provenance are enforced. |
| P8.7–8/G2 | Four stages data-qualified or explicitly data-limited; A1 guard applied; every pre-existing G2 condition and this revision's family bootstrap rule passes, or failure is recorded and synthetic training remains closed. |
| P8.9–12/G3 | Drivers and envelopes registered; no empirical leakage; MLX parity passes at 1e-5 relative; M4 passes every registered G3 output on both required sets, or emulator-guided optimization stays closed. |
| P8.13–15 | Uncertainty sources and conditional/statistical meaning labelled; optimization obeys domain and existing paired claim rule; all final artifacts write-once, hashed, manifest-registered, and status truthful. |
| Repository | Full pytest suite, C++ tests, emissions test after chemistry changes, and both protected-hash checks pass; no unplanned protected/config/model edits. |

## Rollback Notes

Revert named Phase 8 commits, never reset branch history or replace
write-once evidence. Preserve failed run records and gate verdicts. Restart
only an incomplete exact registered job from its snapshot; a scientific
change needs a numbered prospective amendment and a new run ID.

## Escalation Guidance

Complexity is high; use the dispatcher-selected highest-capability executor
for physics/calibration gates and parallel bounded tasks for independent
data and infrastructure. Stop for a genuine scientific scope change, a
user-only digitisation/credential task, or an irreversible external action.
Also report an unavailable source, unmet data stopping rule, non-importable
Cantera module, or failed gate. Continue every track that does not depend on
the blocker. Where this revision calls for a pre-run registration, complete
and commit it prospectively; do not choose a method based on validation or
held-out outcomes.
