# Phase 8 — physics-first revision

**Active dispatch (2026-10-02):** execute only the Track 4 addendum at the end
of this file. Earlier Phase 8 sections are preserved baseline context, not
instructions to rerun prior work. The user instructed this diagnostic work
to run now and to flag blockers while continuing independent work.

Date: 2026-09-29. Branch `phase8`, from the Phase 7 freeze `7524a7a`.
Completed Phase 7 plan: `docs/plan_phase7_completed.md`; closure in
`outputs/phase7_execution_status.md`. Full design document (the source of this
plan; read it for equations and rationale): Claude doc "CAT-JET Phase 8 —
Physics-First Revision Plan",
https://claude.ai/code/artifact/53b56f20-5553-4edb-96d3-8c36b12513a5.
Where the two differ, this file and `docs/phase8_registration.md` govern.

This plan authorizes repo work and Mac-local computation for **Slice 1
(D1)** only: P8.0, P8.1, P8.2, P8.3 and the first P8.6 database sources.
P8.4 onward are listed for order and gates; they need a plan revision before
execution. No manuscript edits, outreach, remote publishing, merge or tag.

**Revision P8-R2 (2026-09-29):** `docs/phase8_r2_plan.md` extends local work
authorization to P8.4–P8.15 in their registered gate order. Its prospective
run registrations and the existing Phase 8 gate rules remain required. This
paragraph supersedes only the original Slice 1 scope limit above; the
original phase descriptions remain as the Phase 8 baseline.

## Objective

Turn the v6 thrust-matched cycle into a component-level model in this order:
C++ port with identical results (G0), then physics upgrades each verified
alone (G1), then staged calibration and source-level validation (G2), and
only then synthetic data, MLX residual models and robust SAF optimization (G3).

The target is honest: v6 held-out fuel flow (87 rows, 9 groups) is 1.830 %
group-weighted MAPE, B0 2.189 %, B1 1.076 %. The model loses to the B1
thrust-ratio rule. The old "~11 % MAPE" is the pre-Phase-1 in-sample figure and
must not be quoted as a baseline. v6 A1 failed: all four knobs (W_ref,
a_thrust, k_pi, k_mdot) are penalty-dependent. P8.4 exists to retire them.

## Decisions (answered by Arnav, 2026-09-29)

- **D1** Slice 1:
  C++ parity port, benchmark, composition-carrying state, enthalpy turbine,
  choking nozzle, first database sources. Everything else is reported as work
  in progress.
- **D2** Write the off-design matching solver in C++; verify it code-to-code
  against pyCycle's high-bypass turbofan example.
- **D3** Add other ICAO engine families. Keep the registered v6 Trent split;
  add a cross-family held-out set (whole families held out).
- **D4** The reactor network becomes the claimed NOx/CO path only if it beats
  the V6 ICAO-derived correlation on held-out engines.
- **D5** Rig data from other machines calibrates non-dimensional loss and
  discharge relations only, never Trent-specific values.
- **D6 (changed)** Phase 7 froze without P7.4. A nozzle PINN returns only as
  model M3 under a Phase 8 registration, if time allows. The turbine PINN stays
  retired. `phase7-p74-wip` is unreviewed code, not a registration.

## Constraints

- Register and commit each procedure before its first computation (benchmark,
  gates, splits, calibration stages). A failed gate stops the steps below it
  and is recorded, never tuned away.
- Protected: everything in `outputs/phase8/protected_sha256_phase8.json`
  (Phase 7 list + v6 evidence + the Python v6 source path). The Python v6 path
  (`integrated_engine.py`, `simulation/`, `scripts/optimization/lto_v5.py`,
  `lto_v6.py`) is the A0 reference and is **not edited** in Phase 8. New
  physics lives in the C++ core; the optimized-Python benchmark arm lives in a
  new module under `scripts/phase8/`.
- Chemical mechanism YAML, `data/icao_engine_data.csv` and `models/*.pt` are
  read-only. Never reuse a v1–v6 artifact name; new outputs go under
  `outputs/phase8/`, write-once.
- Each physics upgrade reduces to the v6 answer in its simplifying limit and
  is added alone, so every change in results is attributable.
- New parameters need a dataset that constrains them and must pass the A1
  profile rule including the penalty guard; otherwise they return to a cited
  fixed value (range rule).
- Simulated/CFD rows never enter a validation set. Validation uses Tier 4
  experimental rows only. No held-out row is opened before its split is
  registered with a hash.
- Keep all Phase 7 negative results visible (A1/A2/A3 FAIL, P7.3 gate
  closed, P7.4 not run, Dooley 2012 H/aromatics failures).
- `.DS_Store` belongs to the user. Ask before any push to origin.
- Use "Slice 1" for the original scope and "ablation ladder" for P8-A2 in
  future files, executor prompts and commit messages. Freeze-package outputs
  go under `outputs/freeze/`; the local freeze tag is `freeze-2026-10-18`.
- Phase 7 detached jobs are finished; there is nothing to leave running. Check
  `ps` for other executors in this tree before editing.

## Repo Context

The v6 production path: `scripts/optimization/lto_v6.py` builds
`lto_v5.V5Model` with the Dooley 2012 Jet A surrogate; each row calls
`integrated_engine.IntegratedTurbofanEngine.run_at_thrust` (single-zone HP
equilibrium combustor, η_b scales the temperature rise, constant-cp analytic
turbine, fully expanded nozzle, φ solved by bracketed `brentq` with
xtol 1e-12, rtol 4ε; the warm-start guess only narrows the bracket).
Knobs: `lto_v5.mode_state` (W_ref, a_thrust, k_pi, k_mdot). Frozen v6
parameters: `outputs/phase7/calibration_v6.json`. Frozen v6 rows:
`calibration_v6_rows.csv` (93) and `holdout_icao_validation_v6.csv` (87).
AE3 = Trent 1000-AE3, record `02P23RR126`, a calibration engine.
Reproduction harness: `scripts/reproduce_check.py` (P6.8, rtol compare).

Environment: Apple M3 Pro, 11 cores, 18 GB. `.venv` Python with
cantera 3.2.0, torch 2.9.1 (CPU), optuna 4.6.0. **No cmake or conda is
installed**; Homebrew is.

## Relevant Files

Existing (read; edit only where stated):
- `integrated_engine.py`, `simulation/**`, `scripts/optimization/lto_v5.py`,
  `lto_v6.py` — A0 reference, read-only in Phase 8.
- `scripts/reproduce_check.py` — pattern for the G0 comparison (not edited).
- `scripts/validation/verify_protected_hashes.py` — add a `--phase8` flag.
- `scripts/build_manifest.py` — add a Phase 8 records entry so new
  `outputs/phase8/` files are registered (orphan test); regenerate, never
  hand-edit, `outputs/ARTIFACT_MANIFEST.md` and `docs/model_map.md`.
- `requirements.txt` — add Python dependencies as each step needs them
  (pybind11, pyarrow, emcee; mlx only at P8.10).
- `outputs/phase7/*_v6.*`, `outputs/phase6/split_p61.json`,
  `data/icao_engine_data.csv`, `data/A2NOx.yaml`, `data/fuel_properties_v7.yaml`.

New:
- `docs/phase8_registration.md` — P8.0 registration (baseline, ledger
  template, gates G0–G3, benchmark protocol).
- `docs/phase8_parameter_ledger.md` — the living parameter ledger.
- `outputs/phase8/protected_sha256_phase8.json`, `scripts/phase8/freeze_protected_phase8.py`.
- `scripts/phase8/profile_v6.py`, `outputs/phase8/profile_v6_*.{txt,json}`.
- `scripts/phase8/benchmark.py`, `scripts/phase8/v6_optimized.py` (arm 2).
- `scripts/phase8/g0_parity.py`, `outputs/phase8/g0_parity_*.{csv,json}`.
- `cpp/environment.yml`, `cpp/conda-lock-osx-arm64.txt`.
- `cpp/catjet_core/` (CMakeLists, `thermo`, `combustor`, `components`,
  `nozzle`, `engine`, `batch`, `bindings`, Catch2 tests).
- `simulation/catjet_backend.py` — Python wrapper keeping the `V5Model`
  call signature, backend selected by one flag. New file; `lto_v5.py` is not
  edited (a new driver passes the backend).
- `data/empirical/` (schema SQL, vocabulary, SQLite file, CSV exports),
  `scripts/phase8/empirical_db.py`.
- `tests/test_phase8_*.py`.
- `outputs/phase8_execution_status.md`.

## Implementation Phases

### P8.0 — registration (no computation except the profile)

1. Freeze `outputs/phase8/protected_sha256_phase8.json`: the Phase 7 list,
   all committed `outputs/phase7/` v6 and registration artifacts, and the
   Python v6 source path. Add `--phase8` to `verify_protected_hashes.py`.
2. Commit `docs/phase8_registration.md` (v6 baseline numbers, parameter-ledger
   template, gates G0–G3, benchmark protocol, workloads) **before** any
   benchmark or parity run.
3. cProfile the Python v6 `run_at_thrust` at the AE3 take-off point with the
   frozen v6 parameters and fuel. Commit the profile and a short summary of
   where the time goes. This is diagnostic, not a benchmark arm.

### P8.1 — C++ core at parity (G0) and benchmark

1. Toolchain (approved 2026-09-29): Miniforge in `~/miniforge3`,
   `auto_activate_base false`, env `catjet-cpp` from `cpp/environment.yml`
   pinning libcantera-devel 3.2.0 with cmake, eigen and catch2; commit the
   explicit lockfile `cpp/conda-lock-osx-arm64.txt`. Compiler: Command Line
   Tools with the MacOSX26.5 SDK (registration amendment P8-A1.3). pybind11
   and the Python interpreter come from `.venv`. Confirm the built module
   imports in `.venv`; if the ABI does not allow it, stop and report.
2. Hello-world: HP equilibrium of the v6 fuel/air at the AE3 combustor inlet
   in C++ matches Python Cantera to 1e-12.
3. Port compressor/fan, burner (v6 HP equilibrium + η_b scaling + dilution
   mixing as coded), analytic turbine, nozzle; unit-test each at 1e-12 against
   the Python functions.
4. Port the thrust-matched φ solve, reproducing SciPy `brentq` iterates
   exactly (port the algorithm; never loosen the tolerance).
5. G0 (registered): C++ reproduces every numeric column of the 93 calibration
   rows and 87 held-out rows and the AE3 design point at rtol 1e-9.
6. Benchmark arms 1–4 on W1–W4 as registered, arms 2–4 in variants a/b/c
   (P8-A1.2: b = fewer evaluations per solve, must pass G0; c = products-only
   equilibrium, own tolerance, outside G0); publish the report.

### P8.2 — composition-carrying state and enthalpy turbine (G1)

GasState (T, P, Y) at every station, properties from Cantera; enthalpy
mixing; liquid fuel vaporisation enthalpy (v7 value). HP/IP/LP turbines on
h(T,Y) with a polytropic path (~50 pressure steps), cooling-air re-entry
(NGV before the rotor, rotor after), cooling fractions by the range rule.
G1 checks: constant-cp limit reproduces v6 T5, p5 to 1e-10; energy closure
< 1e-10 of ṁh; polytropic integral converges as steps double. Scaled maps
come with P8.4.

### P8.3 — choking nozzles at fixed area (G1)

Separate core and bypass convergent nozzles, quasi-1D, frozen composition,
real-gas critical point (max ρu on the isentrope), Cd and Cv, pressure thrust
when choked. G1 checks: reduces to v6 fully expanded when unchoked with
Cd = Cv = 1; ṁ and F continuous at the critical pressure ratio.

### P8.6 — empirical database (starts in parallel with P8.1)

SQLite long format per the design document (tables source, experiment,
operating_point, observation, digitisation, split), controlled vocabulary and
quality classes registered first. QA tests: SI round-trips, corrected-flow
recomputation, duplicate detection, plausibility. First three sources: NASA
CR-168189 (E3 HPT rig), one Langley nozzle report (TP-2991, TP-3411 or NTRS
19870014999), one NASA cold-air turbine report (e.g. TN D-6967). Confirm each
report number and table on acquisition. Download the full ICAO databank and
list candidate families for D3 **without opening any held-out rows**; the
cross-family split is registered before any family's rows are scored.

### Later (need a plan revision before execution)

P8.4 off-design matching (retires the knobs), P8.5 reactor network,
P8.7–P8.8 staged calibration and locked-source validation (G2),
P8.9–P8.12 synthetic data, MLX residual models, active learning (G3),
P8.13–P8.15 UQ, robust SAF optimization, freeze.

## Commands to Run

- `.venv/bin/python -m pytest tests/ -v`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase7`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase8`
- `.venv/bin/python scripts/test_emissions.py` after any combustor/emissions change
- `.venv/bin/python scripts/phase8/profile_v6.py`
- From P8.1: `ctest` in `cpp/build`, `.venv/bin/python scripts/phase8/g0_parity.py`,
  `.venv/bin/python scripts/phase8/benchmark.py --arm N --workload WN`

## Tests

Floor at the freeze commit: record the count in
`outputs/phase8_execution_status.md`. Add tests only for new numerical
contracts: C++ vs Python component parity, G0 comparator, G1 limit and closure
checks, nozzle continuity at choking, database QA. No long benchmark or study
inside pytest.

## Acceptance Criteria (Slice 1)

1. Registration and protected list committed before any benchmark/parity run.
2. G0 passes at rtol 1e-9 on all 180 rows and the AE3 point, or its failure is
   recorded and P8.2 does not start.
3. Benchmark report covers arms 1–4 and W1–W4 with medians of 5 after 1
   warm-up, solves/s, peak memory, named machine, and the cProfile breakdown;
   if arm 2 closes most of the gap, the report says so.
4. P8.2 and P8.3 each pass their G1 checks alone.
5. Database schema, vocabulary and QA tests committed, with ≥ 3 sources
   entered and the D3 family list recorded without held-out rows opened.
6. Tests and protected hashes pass; `outputs/phase8_execution_status.md`
   states truthfully what is done, running and not started.

## Rollback Notes

Phase 8 work is additive on `phase8`. Revert named commits; never reset.
`main` = v5.0, `phase7` = freeze `7524a7a` and `phase7-p74-wip` stay as they
are. No overwrite of protected files or old models.

## Escalation Guidance

Stop and report: G0 cannot be met without loosening tolerance; the Cantera C++
build cannot be imported from `.venv`; a planned source is unavailable or has
no usable tables; any step needs a dependency not yet approved; a gate fails.


# Track 4 addendum — PINN diagnosis (2026-10-02)

The user's attached CAT-JET PINN Repair Guide and instruction to do this now
approve the scope below. Register before computing. This addendum is the
only active task for the current dispatcher; previous registrations, gates,
results and production implementations remain unchanged.

## Objective

Run the isolated turbine 0-D analytic reformulation diagnostic and build the
nozzle verification ladder with known exact answers. Correct unsupported
statements in the guide using source evidence. Report completed work,
failures and blockers honestly. This is diagnostic work, not engine or PINN
validation, and does not reopen G2/G3 or the retired production turbine PINN.

## Constraints

- The committed `docs/phase8_track4_registration.json` is the complete
  prospective protocol: inputs, envelope, coefficients, budgets, seed,
  precision, tolerance and score grid. Do not tune after scoring or change
  thresholds. A failed turbine score ends that attempt; independent nozzle
  verification may continue. Flag a blocked rung and move to independent work.
- The user authorizes a narrow pre-G2 exception for this analytic diagnostic
  only. No empirical residual fitting, production ML, new empirical holdout
  scoring or Sajben training. Do not open WIND/experimental targets.
- Preserve all existing `simulation/**`, `integrated_engine.py`, historical
  sources/results/registrations, chemical YAML, requirements and model weights.
  New checkpoint goes under the new Track 4 output directory, not models/.
- CPU float64, one thread, fixed seed, explicit configs and hashes. Do not
  import the legacy turbine module (it forces a global float32 default).
  Use 4 requested features; eta is fixed and cp/R is derived from gamma.
  State that this envelope has only two varying independent dimensions.
- Track 1 has priority. Before any numerical run inspect actual Python
  benchmark/calibration processes and AC power; if either heavy job is
  actively running or mains is absent, record BLOCKED and continue static
  work. Parked shells are not active computations. Do not change or terminate
  the existing benchmark queue or calibration jobs. Numerical commands use
  nice 15. Do not launch durable training queues or new supervisors.
- Write-once output creation, start/exit logs, final config/weight/source/input
  hashes, library/platform versions and exact reproduction commands. A
  simple synchronous runner is sufficient. Refuse overwriting a prior result.
- Retain the historical 6.390% failure beside the new analytic score. Do not
  attribute that attempt to a global 4.2MPa pressure scale: its actual source
  has analytic path supervision and per-point relative errors. Missing pressure
  equations apply to the legacy physics-only loss; describe that distinction.
- Use Slice 1, ablation ladder, outputs/freeze/ and freeze-2026-10-18 naming.
  No push, merge, tag, manuscript changes, system changes or new dependencies.

## Repo Context

The v5 synthetic envelope is already available and may define this analytic
box. Production stays analytic. Ma's numbered equations differ from the
Boussinesq closure variant parked on phase7-p74-wip; use isolated new code,
not that branch's runner. The local primary Ma PDF was inspected on printed
pages 4–5; its SHA256 is recorded in the protocol. Phase 8 G1/G2 blockers and
all prior negative results remain authoritative. The benchmark queue currently
waits because its broad process predicate also matches parked shells; report
this pre-existing blocker without expanding this task into queue changes.

## Relevant Files

| Action | Exact path | Purpose |
|---|---|---|
| READ | `AGENTS.md`, `CLAUDE.md`, `docs/phase8_r2_plan.md` | workflow and preserved gates |
| READ | `scripts/validation/train_turbine_surrogate.py`, `simulation/turbine/turbine.py`, `integrated_engine.py`, `simulation/combustor/combustor.py` | diagnose historical code; no imports or edits |
| READ | `outputs/turbine_envelope_v5.csv`, `outputs/turbine_surrogate_v5_a2.md`, `outputs/physics_residual_defect.md` | existing synthetic envelope and published negative results |
| READ | `outputs/phase8/protected_sha256_phase8.json`, `docs/phase7_review_followup.md` if present | protected inputs and deferred review |
| MODIFY | `docs/plan.md` | this prospective addendum only, already prepared |
| CREATE | `docs/phase8_track4_registration.json` | protocol, already prepared |
| CREATE | `scripts/phase8/pinn_diagnostics/__init__.py` | isolated package |
| CREATE | `scripts/phase8/pinn_diagnostics/turbine_map.py` | four-input CPU float64 MLP diagnostic |
| CREATE | `scripts/phase8/pinn_diagnostics/nozzle_verification.py` | literal Ma residual, independent forcing, exact nozzle helpers |
| CREATE | `scripts/phase8/pinn_diagnostics/run_diagnostics.py` | minimal write-once runner, resource checks and reports |
| CREATE | `tests/test_phase8_pinn_diagnostics.py` | synthetic numerical contracts only |
| MODIFY | `docs/le_pinn_vs_ma2025.md` | correct actual asymmetric weights and note thermal-unit ambiguity |
| CREATE | `docs/phase8_pinn_repair_notes.md` | E1–E7 worked explanations, proposed write-up paragraph, source corrections, verified reading list and blockers |
| MODIFY | `outputs/phase8_execution_status.md` | append accurate Track 4 status; retain old results |
| MODIFY | `scripts/build_manifest.py` | register diagnostic freeze note as a record, not a claim |
| MODIFY | `outputs/ARTIFACT_MANIFEST.md`, `docs/model_map.md` | generated through build_manifest only |
| CREATE | `outputs/phase8/track4/20261002_attempt1/**` | config, checkpoint, scores, report, environment, start/exit log and hashes |
| CREATE | `outputs/freeze/NUMBERS.md` | only completed diagnostic entries before Oct16, final freeze pending |

No references that are optional/missing block independent tasks. Add no other
files; use temporary logs outside the repository. Do not stage the user's
.DS_Store or the pre-existing evolving benchmark queue log.

## Implementation Phases

### Phase 1 — registered isolation and meaningful tests

The planner commits this plan/protocol before dispatch. Read them completely.
Implement the isolated diagnostic package and tests. The exact Ma equations,
verified from the primary PDF, are:

- Eq22: dx(rho*u)+dy(rho*v).
- Eq23: rho*(u*ux+v*uy)+px-dx(mu*ux)-dy(mu*uy)
  +dx(rho*UU)+dy(rho*UV).
- Eq24: rho*(u*vx+v*vy)+py-dx(mu*vx)-dy(mu*vy)
  +dx(rho*UV)+dy(rho*VV).
- Eq25: rho*cp*(u*Tx+v*Ty)-dx(kappa*Tx)-dy(kappa*Ty)-Phi,
  kappa=conductivity+mu_t/Pr, exactly as printed.
- Eq26: p-rho*R*T.
- Phi=mu*(4*(ux**2+vy**2-ux*vy)/3+(uy+vx)**2), from Eqs7–10.

Keep molecular mu and independent Reynolds-stress covariances distinct.
The printed thermal coefficient adds conductivity and turbulent viscosity;
its dimensional interpretation is unresolved. Implement and label a literal
**dimensionless algebra verifier**, not a physically dimensional replication.
Do not silently multiply the mu_t term by cp.

For each activation tanh/SiLU, the protocol defines smooth fixed fields,
positive coefficients and EOS pressure. Implement independent expanded
analytic forcing from closed-form activation derivatives, without calling
residual/autograd helpers. For SiLU with s=sigmoid(z), h'=s+z*s*(1-s),
h''=2*s*(1-s)+z*s*(1-s)*(1-2*s); for tanh, h'=1-h*h,
h''=-2*h*(1-h*h). Verify forced residual~=0, not the unforced arbitrary field.
Negative controls omit viscosity gradients, stress divergence, conductivity
 gradients or dissipation and must produce detectable error. The manufactured
checks use both activations; exact quasi-1D references involve no network.

Implement actual Ma loss weights, detached from gradients:
  data=.1+.9*sigmoid((Lphys+Lbc-Ldata)/(Ldata+eps));
  phys=.1+.9*sigmoid((Ldata-Lphys)/(Lphys+eps));
  bc=.1+.9*sigmoid((Ldata-Lbc)/(Lbc+eps)).
Correct the audit's all-three-symmetric wording using these equations.

### Phase 2 — run independent diagnostics, fixed budgets

Check AC and actual heavy Python processes. If clear, run the fixed turbine
MLP training and score the 65x65 analytic grid exactly once. Use training-only
stopping; no intermediate score-grid checks or extra optimizer runs. Log max
relative pressure error, worst coordinates, train loss and the fixed historical
6.390% result. Failure ends this attempt without changing its configuration.

Run the Ma manufactured verifier for tanh and SiLU. Advance the sequential
nozzle ladder only when its preceding rung passes; if a rung fails report it
and keep independent Track 4a/report work going.

Exact nozzle rungs: smooth subsonic isentropic, choked isentropic sub/supersonic
branches, then normal shock determined from back pressure. Use the fixed
geometry/constants/grids in the registration. Include back pressure in each
complete shock-case input. Derive two reference back pressures from the fixed
manufactured shock locations before inversion; score recovered location,
exit pressure, mass, momentum and total-enthalpy jumps independently. Check
increasing back pressure moves the shock upstream. Use the independent gamma
1.4, M1=2 rational shock oracle in the protocol. Reject out-of-range internal
shock pressures, invalid gamma/tau and inconsistent duplicate full input rows.
Never differentiate through the shock or call these trained-PINN results.

### Phase 3 — evidence, freeze notes and review

Produce a concise JSON and Markdown report with each rung PASS/FAIL/BLOCKED,
raw errors, tolerances, config/source/input/output hashes and reproduction
commands. Record known blockers and scope limits. Add completed diagnostics to
outputs/freeze/NUMBERS.md only by Oct16 in America/New_York, explicitly marked
diagnostic; do not create the local freeze tag now. Use the manifest registry
for that file and regenerate both generated documents. Document actual E1–E7
mathematics and a proposed explanatory paragraph without claiming the user
completed their handwritten exercises or independently validated the engine.
Retain R7-1..R7-3 and post-freeze Sajben study as deferred, not silently fixed.

## File-Level Edits

New implementation remains in the four listed package files and one test
file. The protocol is authoritative, numeric constants are read from it,
and no new dependency is required. Keep report writes localized to the new
attempt directory. Modify the audit only for verified equation corrections,
append the Track 4 execution status, and update the registry for the new freeze
record. Preserve all old numerical results and registered content. Include
verified DOI links in the repair notes, supplied below; no broad literature
review or external research job is needed.

## Commands to Run

- `nice -n 15 .venv/bin/python -m pytest tests/test_phase8_pinn_diagnostics.py -v`
- `nice -n 15 .venv/bin/python -m scripts.phase8.pinn_diagnostics.run_diagnostics --registration docs/phase8_track4_registration.json`
- `.venv/bin/python scripts/build_manifest.py`
- `.venv/bin/python scripts/build_manifest.py --check`
- `nice -n 15 .venv/bin/python -m pytest tests/ -v`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase7 --phase8`
- `git diff --check`; a word-boundary scan must find no retired event label or sponsor name.

Check resources before numerical tests as well. Static implementation may
proceed while timing is active; flag numerical execution as pending instead
of slowing Track 1. Do not rerun the entire suite repeatedly after it passes.
Commit scoped implementation before producing its first diagnostic result,
then commit reviewed results/logs/status; no unplanned files or partial job
can be described as complete. Check no Git process and no index lock before
index writes/commits; never rewrite history.

## Tests

Known-value polytropic pressure; invalid domain rejection; strict max rather
than mean gate; exact relative-log scoring and pressure-scale invariance;
constant eta feature normalization; explicit float64 gradients/checkpoint
roundtrip; disjoint training/grid inputs; repeatability of generated fixtures;
independent activation first/second derivatives; MMS forcing and all four
negative controls; uniform-flow/EOS limits; coordinate chain rule; actual
asymmetric Ma weights. Exact nozzle branch inversion, choking limit, mass and
enthalpy conservation, rational normal-shock jump oracle, back-pressure
inversion, monotonic shock movement, invalid backpressure and complete-input
identity. Resource refusal/write-once behavior needs meaningful focused tests.
No production simulation, empirical target training or long optimization in
pytest; only synthetic small fixtures. Existing full regression suite is the
final integrity check, with protected hashes and before/after config/model
checksums.

## Acceptance Criteria

1. Plan/protocol and implementation are committed before first diagnostic run;
   all outputs have source/config/input identities, fixed seed and budgets.
2. Turbine gate is max relative pressure error strictly below 0.001 on the
   once-scored 4225-point analytic grid, or a recorded FAIL ends the attempt.
3. MMS for both activations: max absolute forced residual <=1e-10 and relative
   forcing disagreement <=1e-9; each omission control discrepancy >1e-7.
4. Smooth/shock invariants and jump errors <=1e-10; manufactured shock location
   error <=1e-9; exit matches supplied back pressure; all raw errors reported.
5. No nozzle/Sajben training, new empirical holdout use, G2/G3 bypass or old model
   changes. Explicitly flag literal Ma thermal-unit ambiguity and guide's
   inaccurate attempt2 root-cause attribution.
6. Independent tasks completed even if another rung is blocked. Full tests,
   manifest check and protected hashes pass; no unexpected files change.
7. Freeze notes contain only completed diagnostics, historical failures stay
   visible, and final report lists remaining user/data/post-freeze work.

## Rollback Notes

All production and protected sources remain unchanged. Revert only the named
Track 4 commits if needed; preserve any diagnostic negative evidence and the
unrelated user changes. Never reset branches, overwrite weights or remove old
registrations. New write-once outputs may be retained as superseded evidence.

## Escalation Guidance

Scientific verification plus independent derivative tests spans turbine,
nozzle and PINN code: high complexity (about 8/10); dispatcher-selected Opus
is appropriate. Do not grow supervisors or training infrastructure. On an
unknown equation or missing datum, flag that specific rung and proceed with
independent exact cases. On fixed-budget turbine failure, stop that attempt,
report its measured error and do not retune. Repair implementation bugs only
before scored computation; after scoring any rerun needs a prospective new
attempt. Production empirical residuals, balancing cross-validation and the
Sajben low-label study are explicitly deferred.

Verified primary reference links for the repair notes:
- Ma: https://doi.org/10.1016/j.ast.2025.111002
- Krishnapriyan: https://arxiv.org/abs/2109.01050 (NeurIPS 2021)
- Mao/Jagtap/Karniadakis: https://doi.org/10.1016/j.cma.2019.112789 (CMAME360,2020)
- Jagtap/Kharazmi/Karniadakis: https://doi.org/10.1016/j.cma.2020.113028
- Wang/Yu/Perdikaris: https://doi.org/10.1016/j.jcp.2021.110768 (JCP449,2022)
- Kennedy/O'Hagan: https://doi.org/10.1111/1467-9868.00294
- Wang/Teng/Perdikaris gradient balancing: https://doi.org/10.1137/20M1318043


# Track 4 pre-run review corrections — active continuation (2026-10-02)

The reviewer paused the first executor pass before any diagnostic computation
to correct the issues below. Its uncommitted package/tests/audit/notes are in
the worktree: continue from them, do not rebuild from scratch. There is no
completed diagnostic result to preserve or rescore. The protocol and all
numeric budgets/tolerances remain unchanged. This section is authoritative
where it refines the addendum above. Do not execute old Phase8 tasks.

## Objective

Finish the same isolated diagnostics with trustworthy refusal, failure,
provenance and boundary behavior. Numerical work remains contingent on AC and
absence of actual benchmark/calibration jobs. On a resource blocker, complete
all static implementation/documentation, run non-numerical integrity checks,
record the exact pending numerical commands, and report BLOCKED honestly.

## Constraints

Same protected sources, fixed prospective protocol, no production integration,
no empirical targets, one CPU thread and no post-score tuning. Never treat
CLI exit0 or dispatcher's DONE line as proof of successful work: the interrupted
first pass printed Execution error but returned0. Inspect files/logs/statuses.
Do not create placeholder diagnostic scores or freeze notes on battery.

## Repo Context

The implementation is still uncommitted. The Mac switched to battery after
planning. Registered numerical diagnostics and numerical pytest were not run.
Review findings below are static, and the actual formula implementations are
otherwise consistent with the registration. The two prior test-rounding/
small-corruption defects have already been corrected in current tests; do not
reintroduce them. Old benchmark queue remains parked by its false process match.

## Relevant Files

Use exactly the implementation/documentation paths from the Track4 table
above; no extra package or supervisor. The source guide IS present, read-only,
at the user's supplied attachment path:
`/Users/arnavpatil/.codex/attachments/8da7fbac-4294-4608-9c1f-f4f824ae6400/Pasted text.txt`.
This external source path is a read-only planning reference, not a path to
embed in runtime code. Read its E1–E7 and final Track4 instructions; the first
pass's repair-notes assertion that it is absent from the machine is false.

## Implementation Phases

### Phase 4 — correct reviewed defects before any scored execution

1. Runner gates must fail closed when ps, Git status or HEAD cannot be read.
   Avoid Mac ps comm-column truncation: the real dispatch Python appears as
   `/Library/Framewo` in `ps -Ao pid=,comm=,args=`. Parse actual executable from
   `ps -Ao pid=,args=` with shlex or an equivalent reliable approach; recognize
   Python absolute paths and both script/module invocation forms. Ignore parked
   bash/caffeinate shells. Test command failures, actual Mac Python argv,
   module-form jobs and parked shells using mocked text, not real jobs.
2. Freeze a start identity (HEAD, exact registration bytes/hash, sources,
   envelope identity) and use it consistently in config/checkpoint/report.
   Recheck at end; flag drift without replacing the start identity. Missing
   HEAD is BLOCKED. No checkpoint/report may claim a newly re-read config or
   different source state generated its earlier predictions. Do not add a
   supervisor or snapshots beyond these small identity dictionaries.
3. Include all post-output-directory setup in finalizing error handling so
   import/thread/environment/write failures get an ERROR/exit log and hashes.
   Return nonzero for any FAIL, ERROR or required BLOCKED rung. Give report an
   explicit aggregate status; completed failed diagnostics remain evidence.
4. Read the synthetic envelope with round-trip float parsing before exact
   extent checks (e.g. pandas float_precision='round_trip'). Confine an envelope
   verification error to Track4a; continue independent MMS activations. Capture
   errors separately for each activation. If either MMS activation fails/errors,
   mark downstream exact-nozzle rungs BLOCKED, as required by the ladder.
5. Use the registered Torch Sobol draw_base2 implementation for powers-of-two
   samples. Reject training/grid overlap before any scoring; a contaminated
   grid must never yield PASS. Add a mocked collision regression that performs
   no training or score. Existing actual split-disjointness test remains.
6. Ma loss-weight helper must compute CPU float64 even from Python scalar
   inputs; add dtype and asymmetric-value regressions for that caller form.
7. Backpressure inversion accepts both internal-shock-range endpoints, so
   shock_profile must show the downstream exit pressure for a shock at the
   exit. Use a consistent jump convention and test both endpoints in addition
   to registered interior cases. Do not change the registered positions or
   thresholds. Verify postshock p0 loss and actual side conservation.
8. Fix docs/phase8_pinn_repair_notes.md source availability and numbering to
   match the supplied guide exactly: E1 polytropic derivation/sensitivity;
   E2 pressure-loss nonidentifiability and its missing log residual;
   E3 global-normalization example (label legacy, not attempt2);
   E4 ReLU/product derivatives; E5 duplicate-input MSE mean;
   E6 area-Mach/choking/normal shock; E7 actual asymmetric Ma weights and
   gradient-statistics comparison. Keep the four-feature/two-dimension insight
   as an additional note. Do not claim the user completed exercises. Also the
   exact pressure gate is log(1-threshold)<Delta<log(1+threshold), rather than
   symmetric |Delta| iff log(1+threshold). Explain that distinction accurately.
9. If numerical work remains BLOCKED, no outputs/freeze/NUMBERS.md exists and
   no diagnostic score entries are eligible yet. Remove the first pass's
   uncommitted mandatory Freeze record from build_manifest (it refers to an
   absent file), keep generated manifests unchanged, and record the conditional
   freeze step in execution status. Do not create an empty placeholder to make
   manifest tests pass. If diagnostics actually complete before Oct16, follow
   the original completed-diagnostic freeze instructions and register the file.

## File-Level Edits

Local corrections in turbine_map.py, nozzle_verification.py, run_diagnostics.py
and tests/test_phase8_pinn_diagnostics.py; corrected repair notes, accurate status,
conditional registry changes only. The already-written audit corrections are
valid. No model/config/production/registered content changes. The attachment is
read-only. Preserve pre-existing .DS_Store and benchmark queue log changes.

## Commands to Run

If AC/resource checks clear, run the originally registered focused tests,
commit implementation, run the once-scored diagnostics, then full regression,
manifest and protected-hash checks. Do not run the fixed diagnostic twice.
If blocked, use AST parsing without importing/executing diagnostic functions,
`git diff --check`, manifest --check (no new science), protected-file hashes,
and byte-check unchanged protected sources/models/config. Run the resource
refusal command to demonstrate BLOCKED without creating the attempt directory.
Explicitly say focused/full numerical pytest remains pending; do not call it
passed based on static checks. Commit only reviewed scope, with honest pending
status. The user requested flagging blockers and moving on, not waiting forever.

## Tests

Add only the meaningful mocked/endpoint/scalar-dtype regressions described
above. Keep previous synthetic mathematical tests. No registered score-grid
predictions during pytest and no long training. Tests of refused resources or
identity failures must create no real attempt outputs or launch workloads.

## Acceptance Criteria

All listed static defects fixed; syntax/manifest/protected checks pass. When
numerical resources permit, all original numerical criteria and full regression
pass. Otherwise report BLOCKED for each unrun numerical check and create no
freeze score entries or production integration. Guide notes match E1–E7 and
contain no false availability statement. No unexpected worktree files changed.
A non-PASS diagnostic cannot return success. Logged identities remain coherent.

## Rollback Notes

Same additive-only rollback. Preserve user changes and historical records.
Remove/revert only this task's own uncommitted or named changes; no reset,
history rewrite, old model overwrite or killing unrelated jobs.

## Escalation Guidance

This is a bounded reviewer fix pass, not a new experiment. If AC stays absent,
finish static work, clearly mark run/test/freeze steps pending, and report back.
Do not create new training budgets, relax tolerances or add infrastructure.
