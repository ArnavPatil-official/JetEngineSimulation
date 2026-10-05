# Phase 7 executor review and continuation (2026-09-27)

## Objective

Finish the approved `docs/plan.md` Phase 7 implementation, fix the concrete
review findings below before launching P7.4, and launch all registered work
durably. This supplements, rather than replaces, that plan. These are routine
correctness fixes within its authorized scope; do not ask for another approval.

## Constraints

- The prior executor was deliberately interrupted by the reviewer at about
  11:44 while testing the new PINN implementation. Its wrapper prints DONE
  after Execution error; that is not successful completion. Preserve its log.
- Calibration is COMPLETE at frozen commit `271656e`: selected
  polish_from_v5_A2, SSE 0.0004570833141686023, calibration MAPE 1.7848700777%,
  W_ref 103.2852112459. The full profile is RUNNING under its own durable
  supervisor. The blend supervisor at `a242c7b` waits on that pipeline.
  Do not stop, duplicate, refit, or change those numerical jobs.
- No registered P7.4 scientific training has launched. Timing-only temporary
  smoke runs exist. Preserve the 5000-epoch budget and existing split,
  architecture, loss, threshold and benchmark choices; no score-based tuning.
- The user EXPLICITLY ANSWERED: "Use WIND labels, including viscosity, only at
  sampled training points (recommended)". This is recorded in `docs/plan.md`
  at commit `0e1f4e4`, while the prior headless executor was running. Its
  `register_p74.py` and generated registration incorrectly say no answer was
  present. Correct this provenance before registration commit/training.
- All protected artifacts, old models and chemical YAML remain untouched.
  `.DS_Store` belongs to the user. No merge, push, tag or manuscript work.

## Repo Context

P7.2 and P7.3 are committed and independently reviewed. New PINN files and
tests are uncommitted and partly tested, NOT a completed implementation.
Read the working tree and finish it; don't start over. The residual's variable
viscosity flux, Boussinesq single counting, current-loss weights and training
row extraction look consistent so far. Molecular viscosity uses Sutherland(T)
instead of WIND mu_l labels; state this closure explicitly and validate it.

## Relevant Files

`scripts/phase7_supervisor.py`, `scripts/run_phase7.sh`,
`scripts/validation/{register_p74,train_sajben_ma,report_sajben_ma}.py`,
`simulation/nozzle/le_pinn_ma.py`, `tests/test_{phase7_supervisor,le_pinn_ma}.py`,
`outputs/phase7/p74_{registration,split,compute_benchmark}.json`,
`docs/{plan,phase7_registration,le_pinn_vs_ma2025}.md`,
`outputs/phase7_execution_status.md`, and the manifest/crosswalk/model-map
files named in the main plan.

## Implementation Phases

1. Read the main plan, this review and current diffs. Confirm active Mac jobs.
2. Repair the issues below and complete meaningful PINN tests. Commit the
   numerical registration and code BEFORE any scientific training.
3. Launch the P7.4 queue durably from that commit, inspect real progress, and
   finish the main plan's reporting/manifest/model-map work with honest pending
   states. Automate the known downstream work where results are pending.
4. Run the required complete review checks and report exact commits, jobs,
   gates, outputs, test counts and remaining runtime. Do not wait idly for a
   20+ hour study or call it finished.

## File-Level Edits / Review Findings

### R7-1: report cannot currently run in its frozen snapshot (blocking)

`pinn_jobs()` imports checkpoints and run records only. The report defaults
`require_runner=True` and demands a per-run COMPLETE.json under its snapshot,
but those completion files exist only in the live repo and are never imported.
After all 30 runs it will therefore refuse. Pass/import a VERIFIED completion
bundle into the report snapshot, or an equivalent sound design. Do not fix by
silently dropping completion verification. Add an integration test using tiny
synthetic jobs that exercises the actual snapshot -> completion -> report path.

### R7-2: claimed report verification is not implemented (blocking)

`report_sajben_ma.verify_complete` currently checks checkpoint hash, train IDs,
smoke flag and epoch count, but not config hash, registration hash, run/arm/seed,
or completion-record contents. Its docstring promises configuration/split
checks. Validate those against the registration and the shared run_config
definition, verify completion output hashes, and fail closed on tampering or
nonfinite scores. Validate scoring inputs and split hashes too. Tests must
reject an altered config, wrong arm/seed, changed record, and partial batch.

### R7-3: resume may silently change source/config (blocking)

`cmd_launch` defaults to current HEAD even for an existing queue. A later HEAD
can therefore restart an interrupted attempt with different code, despite
promising an exact registered rerun. Pin existing queues to their original
source snapshot/config automatically, or refuse incompatible resumes with a
clear message. Verify completed jobs when skipping them, not only when another
job depends on them. Include source/inputs in resume identity as appropriate.
Preserve existing active calibration and blend snapshots; changes here apply
to new launches. Test HEAD changes and tampered completed outputs.

### R7-4: shared status temporary filename can race

`write_status_md` is called by multiple supervisors, while
`write_json_atomic_text` uses the same `STATUS.tmp` filename for all writers.
Use a unique temporary name per writer and test concurrent updates. A status
refresh must not kill a numerical supervisor or leave work unobserved.

### R7-5: provenance / limits / final documentation

Correct the answer-recorded text in generator and uncommitted registration to
the user's explicit answer (not a timed-out assumption). Document the learned
mu_t + Sutherland(T) closure and no isotropic k approximation accurately; don't
equate ignoring k with an exact pressure modification. Document the hard
near-wall fusion switch limitation (Ma itself discusses it). The supplied Ma
paper text is `/tmp/phase6_review/ma2025.txt` and PDF is the attached file in
the conversation; exact current-loss equations are already in the main plan.
Fix the old audit's weight shorthand and finish the main plan's pending
manifest/crosswalk/model map/runner documentation. Never insert numerical
results for jobs that are pending or incomplete. Include a reviewable durable
path to regenerate these after study completion; don't leave a manual-only
dependency unnecessarily.

## Commands to Run

- `bash scripts/run_phase7.sh status`
- `.venv/bin/python -m pytest tests/test_le_pinn_ma.py tests/test_phase7_supervisor.py -q`
- `.venv/bin/python -m pytest tests/ -v`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase7`
- Relevant Cantera/emissions validations from the main plan.
- `bash scripts/run_phase7.sh launch pinn` after registration/code commit.

## Tests

Complete manufactured-field tests (variable viscosity gradients, flux/energy,
uniform limit, turbulence once), curved wall normals, physical-coordinate
chain rule, true Ma weights, train-only scalers/viscosity, paired initialization
and nested subsets. Poison held-out labels and verify training inputs/losses
are unaffected. Exercise a short reproducible restart and record config
identity. Add the runner/report integration and tampering tests above.

Independent reviewer checks already passed: fuel tests 7; new calibration and
supervisor tests 23; blend tests 11; shared holdout scorer reproduces frozen v5
metrics to 1e-12. Do not rerun old studies. Reviewer protected-file baseline is
`/tmp/phase7_protected_review.json` (81 files); compare, do not regenerate it.

## Acceptance Criteria

All main-plan criteria applicable before numerical completion, plus R7-1 to
R7-5 resolved, clean meaningful tests, preserved historical evidence, actual
durable jobs launched, and a report path that can consume their verified
outputs automatically. Phase 7 remains pending while long jobs run.

## Rollback Notes

Do not reset or discard uncommitted work. Keep interrupted dispatcher evidence.
Use corrective commits. Preserve running snapshots and all registered settings.
If an actual scientific failure occurs, record it under the registered rule;
do not change the objective or budget to obtain a favorable outcome.

## Escalation Guidance

High complexity scientific/runner integration review: Claude Opus. Continue
autonomously on routine correctness fixes; surface a genuine scientific scope
change rather than silently changing the registration.
