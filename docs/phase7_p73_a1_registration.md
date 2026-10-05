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
and conditional ranking decisions. The quantitative artifacts' JSON metadata
and technical README must say **conditional on the frozen v6 calibration;
A1 FAIL (penalty-dependent)**. Preserve the parent CSV schemas unchanged.
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
   consumer/test afterward can invalidate that terminal identity. The
   prospective source-extension protocol below preserves that original
   identity and freezes the additive consumer identity separately. Supply the
   main root explicitly for imports, inputs, records and binary, and verify
   module origins/hashes. Do not mix unverified sources from different checkouts.
2. After the main chain terminates and before integrating any new scientific
   source path, validate the original main terminal context strictly and export
   a write-once hashed dependency attestation. Include raw registration,
   queue/stage/completion/owner-release/terminal evidence, command/handshake/log/
   output hashes, original launch-tree blobs/modes and original module hashes.
   A new additive helper revalidates all of that evidence and proves every
   original source/module unchanged, permitting only prospectively registered
   new paths. Only then may it inject the **proven original main identity**
   into read-only Workflow checks. `expected_identity` is that original
   workflow identity, never the new consumer identity. Do not edit the old
   helper or old records. Terminal failures remain truthful and cannot supply
   a required PASS prerequisite. Repeat proof after waits and before launch.
3. Obtain a fresh write-once G0 PASS with the registered comparison rule and
   the same actually loaded full-equilibrium binary to be used by this study.
   Preserve legacy `outputs/phase8/g0/` and `g0_parity.json`. The existing G0
   script emits no reservation or actual loaded binary hash. Do not claim
   those records exist. A supplied rerun lacking that evidence remains pending
   provenance verification, rather than discarded or accepted with fabricated
   metadata. A separately authorized fresh run may use a registered wrapper
   that reserves a new namespace and captures actual source/core provenance
   around the unchanged G0 script. The documented
   `outputs/phase8/g0_rerun_20261003` directory was not found at registration;
   fresh evidence remains pending.
4. Reserve `outputs/phase7/p73_a1_cpp_20261004` atomically with the source,
   parent/input, dependency, registration and binary hashes, environment,
   command, AC evidence and PID/birth ownership. Verify the exact loaded module
   path/hash equals the selected core in fresh G0 and blend parity evidence.
   Select the old `cpp/build` core explicitly, with attested original binary
   provenance, or select `cpp/build_next` through an explicitly registered
   preload wrapper with matching PASS build provenance. Disclose that choice;
   do not require the old core to equal a different validation-build artifact.
   The benchmark module is never rebuilt or overwritten. Verify the selected
   module in the parent and every worker; an environment variable or
   parent-only preload does not select it reliably in spawned workers.
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

Prospective clarification dated 2026-10-04, before implementation execution:
the source-extension attestation is exported to
`outputs/phase8/screening_operations/main_dependency.json` only
after strict original-main terminal validation and before new code integration.
It is not asserted to exist now. The new consumer and
`tests/test_phase7_p73_a1_cpp.py` are explicitly allowed new paths; the additive
shared gate is separately registered. Recheck original blobs/modes/modules
and every referenced raw dependency hash against this immutable attestation.
Freeze the consumer/helper/test/config identity separately. Any changed or
deleted original path, undeclared new source, altered raw record or old module
drift blocks execution. No scientific rule is changed by this clarification.

The separately registered `scripts.phase8.scientific_workflow_gate` exports
the original attestation through `export_main_context(root)`. The consumer uses
`prepare_context(root, registration, expected_consumer_identity=...,
require_g0=True)`, `Context.require_idle_ac()`, and
`Context.acquire_run(output_dir, registration_sha256, identity=...)`.
`Run.assert_current()` revalidates dependencies/resource/ownership and
`Run.release(terminal)` publishes terminal evidence and releases ownership.
The central lease is `outputs/phase8/screening_operations/owner.lease.json`.
The consumer never disables the G0 prerequisite.

The future new consumer only adapts orchestration, explicit frozen input paths,
backend selection, additive dependency exception, provenance and output routing.
Reuse protected fuel/pair/task/draw/lifecycle/nvPM/claim functions. Do not edit
the protected implementation. Export the same quantitative columns unchanged;
put registration/backend/conditional-label provenance in JSON and technical
README, with no change to quantitative meaning.

The future synthetic tests cover exact inheritance, penalty-guard-only
exception refusal cases, draw/fuel/mode completeness, conditional labels,
unavailable nvPM values, backend selection, dependency/hash drift,
reservation/write-once failures and missing-output terminal failure.

Additional prospective clarification dated 2026-10-04, before execution:
`new_source_paths` names exactly `scripts/phase8/p73_a1_cpp.py` and
`tests/test_phase7_p73_a1_cpp.py`. Approved `partial_rows.jsonl` in the new
output namespace records each finished parity/study batch append-only with
stage, case and frozen identity, retaining raw rows after a later failure.
It adds no simulator requests and changes no final CSV schema. Nonfinite raw
values are encoded explicitly. A row labeled converged must have finite
required solve quantities before scoring; malformed evidence is ERROR.
The protected claim function and its thresholds remain unchanged.

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
- Claim/ranking CSVs retain their parent schema and accompany JSON/technical
  README with the calibration-conditional disclosure. No empirical validity or
  operational approval is asserted.
- Original CORSIA draws, liquid-LHV correction and Brem-only validity rules
  remain unchanged; unavailable nvPM outputs contain no number.
- Fresh G0 and registered blend parity PASS, verified actual selected core
  provenance, distinct original-main/additive identities and attestation,
  source/input hashes, main terminal evidence, AC and exclusive ownership exist
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
