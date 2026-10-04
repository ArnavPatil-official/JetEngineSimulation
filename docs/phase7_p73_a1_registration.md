# P7.3-A1: conditional matched-thrust blends through the C++ v6 backend

Registered prospectively on 2026-10-04, based on main commit
`59eab6c2ac52865a470884e55f2c7f3e0539631a`. Machine-readable contract:
`docs/phase7_p73_a1_registration.json`. This additive record does not alter
`outputs/phase7/p73_registration.json`, `docs/phase7_registration.md`, the
protected runner, or any historical gate/result record. It registers a future
separate consumer; it neither implements nor launches that consumer.

## Objective

Execute the originally registered P7.3 comparisons with the full-equilibrium
C++ v6 backend, keeping the selected Phase 7 v6 calibration frozen. The user
authorizes a narrowly defined exception to the original A1 dependency after
disclosure of its penalty-guard-only failure. Preserve the original claim rule
and conditional ranking decisions. Every artifact and technical README must
say **conditional on the frozen v6 calibration; A1 FAIL (penalty-dependent)**.
Those decisions describe this calibrated model, not demonstrated real-world
SAF performance, an identifiable calibration, or a passed original gate.

Known at registration: the frozen profile reports `IDENTIFIED_existing_rule`
true, `IDENTIFIED` false and `penalty_dependent` true for all four fitted
parameters. The original A1 remains FAIL; A2 and A3 also remain FAIL, A4 PASS,
as already recorded in Phase 8 registration section 1. A2 is not ESCALATE.
The original `GATE_CLOSED.json` says `open=false`, reason `A1 FAIL`. No P7.3
production blend result exists. No new blend, parity, calibration, nozzle or
PINN calculation was performed while preparing this amendment.

## Constraints

- The exception applies only when every fitted parameter satisfies the
  existing profile rule and fails identification solely through the registered
  penalty guard. Require the frozen profile, fit, dependency and closed-gate
  hashes. A different A1 failure or B0 escalation blocks this exception.
- Keep all original scientific definitions below and in the JSON's
  `unchanged_protocol`. Do not refit, replace selected v6 parameters, substitute
  Phase 8 fitted parameters, change tolerances, change draws, remove rejected
  comparisons, or convert failures to successful solves.
- Use the full-equilibrium `core.V6Engine` through `CppEngine` and
  `V6Model(..., backend="cpp")`; no optimized approximation, newer physics,
  reactor network, PINN substitution or Python fallback.
- Retain conditional claims/rankings with pass/fail reasons. Publish quantitative
  CSV/JSON artifacts and a technical README only; no narrative results draft.
- Preserve the currently active main AC chain and its source identity. This
  registration adds no stage to it and changes no order. Numerical execution
  waits for validated terminal records, ownership release and a separate
  exclusive reservation, on AC under `caffeinate` and `nice -n 15`.
- Read only AE3 engine inputs (`with_targets=False`) for this comparison. Do
  not read empirical nozzle targets or make a new held-out score. Existing
  frozen gate/profile JSON metadata is dependency evidence, not a new score.
- New output directory and logs are write-once. An occupied, failed or partial
  reservation is retained and blocks an automatic retry. Never overwrite
  original Phase 7 outputs, protected manifests, model weights or YAML files.

## Repo Context

The protected Python runner already defines the fuel mixtures, task order,
fixed draws and postprocessing. Its `main(out_dir)` couples fit/gate inputs to
the output directory and enforces the original closed gate. A new consumer
must supply explicit frozen fit/gate paths rather than call or edit that main
function, copy its inputs into a new output directory, or change its gate.

The adapter replaces the worker engine while keeping `lto_v5.solve_task` and
its held-out NOx exclusion. Construct `V6Model` with the original fixed/split
context and call the protected `tasks_for`/`run` task path. Each task supplies
its fuel composition and fixed draw, `phi_guess=None`, original task order and
`chunksize=4`. `model.predict()` uses central defaults and is not this path.

## Relevant Files

Read unchanged: `outputs/phase7/p73_registration.json`,
`docs/phase7_registration.md`, `outputs/phase7/p72_registration.json`,
`outputs/phase7/calibration_v6.json`,
`outputs/phase7/identifiability_profile_v6.json`,
`outputs/phase7/holdout_icao_validation_v6.json`,
`outputs/phase7/runs/blends/p73_blends/GATE_CLOSED.json`,
`outputs/p62_bands_v5.csv`, `outputs/phase6/split_p61.json`,
`scripts/optimization/blend_matched_thrust_v6.py`,
`scripts/optimization/lto_v5.py`, `scripts/optimization/lto_v6.py`,
`scripts/phase8/v6_backend.py`, `simulation/catjet_backend.py`,
`simulation/fuels_v7.py`, `data/fuel_properties_v7.yaml`,
`data/corsia_lca_values.yaml`, `data/creck_c1c16_full.yaml`,
`outputs/phase8/protected_sha256_phase8.json`,
`docs/phase8_queue_recovery_registration.json`,
`scripts/phase8/ac_workflow.py`, and `scripts/phase8/g0_parity.py`.

This commit adds only this document and
`docs/phase7_p73_a1_registration.json`. A future implementation requires a new
consumer `scripts/phase8/p73_a1_cpp.py`, a separate synthetic test file and a
committed implementation identity before execution. Their creation is not
part of this docs-only commit.

## Implementation Phases

1. Commit this prospective registration and review the additive implementation
   in isolation. Freeze its final committed source/config identity before
   numerical work. No main scientific source integration while the AC chain
   is active.
   The main validator compares its entire registered source set: adding a
   consumer/test afterward can also invalidate that terminal identity. Run
   the additive code from a separately frozen isolated worktree while keeping
   the validated main root unchanged, or prospectively register exact identity
   extension rules before execution. Do not waive checks or reuse stale records.
   Supply the main root explicitly for protected imports, inputs, records and
   binary; verify loaded modules originate there and freeze additive code
   separately. Do not mix scientific sources from different checkouts.
2. Validate the main terminal context using
   `validate_terminal_context(root, expected_identity=..., require_idle=True)`.
   Require strict benchmark completion/owner-release evidence, referenced
   stage hashes, no lease, and a PASS `validation_build` with its actual binary
   hash. Terminal command failures retain their truthful status; they cannot
   supply a PASS prerequisite. Repeat validation after waits and before launch.
3. Obtain a fresh write-once G0 PASS with the registered comparison rule and
   the same actually loaded full-equilibrium binary to be used by this study.
   Preserve legacy `outputs/phase8/g0/` and `g0_parity.json`. At registration,
   the documented `outputs/phase8/g0_rerun_20261003` directory and a matching
   reservation were not found; their absence is pending, not PASS evidence.
4. Reserve `outputs/phase7/p73_a1_cpp_20261004` atomically with the source,
   parent/input, dependency, registration and binary hashes, environment,
   command, AC evidence and PID/birth ownership. Verify the exact loaded module
   path/hash equals the validated build and fresh G0 evidence. The benchmark
   module in `cpp/build` is never rebuilt or overwritten. Explicitly preload
   and verify the selected `cpp/build_next` module in the parent and every
   worker; an environment variable or parent-only preload does not select it
   reliably in spawned workers.
5. Execute registered backend parity first (coverage/tolerances in JSON). A
   failure records FAIL/ERROR and blocks the conditional blend run; retain all
   parity artifacts. This checks numerical backend equivalence, not empirical
   blend validity or every untested draw.
6. Run all original central tasks and exactly 64 common fixed draws through
   the C++ task path. Keep all fuel/mode/status rows, including unconverged
   rows; compute the original postprocessing and conditional decisions.
7. Revalidate dependencies, source/input/binary identity and resource/ownership
   state before publishing terminal artifacts. Any unreadable evidence, drift,
   exception or missing expected output prevents COMPLETE. Verify protected
   hashes and retain all quantitative/rejected outputs and technical README.

## File-Level Edits

The future new consumer only adapts orchestration, explicit frozen input paths,
backend selection, additive dependency exception, provenance and output routing.
Reuse protected fuel/pair/task/draw/lifecycle/nvPM/claim functions. Do not edit
the protected implementation. Export the same quantitative columns, plus
registration/backend/conditional-label provenance; no change to their meaning.

The future synthetic tests cover exact inheritance, penalty-guard-only
exception refusal cases, draw/fuel/mode completeness, conditional labels,
unavailable nvPM values, backend selection, dependency/hash drift,
reservation/write-once failures and missing-output terminal failure.

## Commands to Run

For this docs-only change: parse the new JSON with the standard library, check
that `unchanged_protocol` equals the named sections of the protected parent
registration, and run `git diff --check`. These checks import no model/core and
run no scientific computation. Check Git processes and index locks before the
additive commit.

Future execution, only after the prior phases and an implemented preflight:
`caffeinate -i nice -n 15 .venv/bin/python scripts/phase8/p73_a1_cpp.py --registration docs/phase7_p73_a1_registration.json --workers 6`.
This command is prospective; the consumer does not yet exist. Verify protected
hashes using the existing checker after the permitted future execution. Never
use process-name matches as completion or ownership evidence.

## Tests

The registration itself receives JSON/inheritance and whitespace checks only.
No model import, build, full pytest, target read or numerical test is authorized
in this worktree. Future pure synthetic tests must pass before arming. Future
numerical parity uses `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` per numeric
cell, exact status/reason strings, identical keys/columns/shapes and matching
missing-value patterns. No tolerance may be loosened after results.

## Acceptance Criteria

- Parent registration, profile, fit and historical gate hashes remain intact;
  the original A1 remains FAIL and gate remains closed.
- Exactly the original 64 paired fixed draws, fuels, fractions, four points,
  quantities, comparisons, spread S and original claim rule are preserved.
  CLIMB85 remains reported and never claimed. Context families stay separate.
- Every claim/ranking artifact visibly carries the calibration-conditional
  disclosure. No empirical validity or operational approval is asserted.
- Original CORSIA draws, liquid-LHV correction and Brem-only validity rules
  remain unchanged; unavailable nvPM outputs contain no number.
- Fresh G0 and registered blend parity PASS, actual binary/source/input hashes,
  main terminal-context evidence, AC evidence and exclusive ownership exist
  before the full blend stage. Legacy G0 alone does not authorize this launch.
- All expected quantitative outputs, technical README and terminal evidence
  exist with recorded hashes; any drift, partial output or prerequisite failure
  is recorded without a false COMPLETE/PASS or automatic rerun.
- This registration commit adds only the two new docs files and starts no job.

## Rollback Notes

An additive revert can withdraw this amendment before computation. Keep the
original gate and records intact. After reservation or computation, retain the
reservation, logs, partial outputs and failure records; annotate an invalid run
in a new dated record. Do not delete or overwrite evidence or rewrite history.

## Escalation Guidance

The implementation is a medium-complexity reproducibility/orchestration change.
Use a reviewed executor implementation in isolation; seek review for changed
scientific definitions, additional physics, empirical target access or any
new calibration. A required input/hash, fresh G0, binary, AC or main-context
failure blocks that numerical stage and is reported; independent docs work
can continue. This commit remains registration-only until its prerequisites
and implementation have been separately completed.
