# Stage f serialization recovery — user-approved operational plan

## Objective

Implement the exact recovery block below for the existing CPU-only WSL run, preserving all scientific bytes and the consumed score reservation. The user approved this scope on 2026-10-05. Implement and validate the fix and recovery script for root review; root then performs the authorized push, dry-run, and sole real run in tmux.

## Constraints

Only this PC agent works; no Codex subagents. Follow the repository planner/Claude executor workflow. All repository files, tests, training and scoring stay in the Linux filesystem at /home/arnav_patil/projects/JetEngineSimulation, using catjet-pc Python 3.12.14, torch 2.9.1+cpu, Cantera 3.2.0, CPU float64, ten workers, and the four thread environment variables set to 1. Never execute scientific commands in /mnt/c or the Windows checkout.

No metric, threshold, model, selection, data, registration, amendment or protected artifact changes. No retraining, reselection, prediction regeneration, deletion, overwrite, target-driven changes or automatic retries. Old metadata directory stays byte-identical. A scientific FAIL remains a result; an execution ERROR stops.

Do not read authentication credentials or print secrets. Do not install anything. Do not push, call --run, or perform actual scoring/training during implementation: root reviews first and then executes the remaining already-authorized operations. Fixture tests and the read-only recovery dry-run are allowed.

## Repo Context

Existing run on main commit 0d16bcc245e93172aa4b3749bccc877727ce950c has stages a-e COMPLETE. Stage f consumed its identical hashed score reservation, froze 72 prediction archives and the 24 trained weight bundles, and wrote test_metrics.json and ranking_metrics.json before serialization failed. Original f metadata contains an ERROR and no completed checkpoint.

Read-only diagnosis confirmed claim_concordance[*].central_sign_concordant is numpy.bool, from comparison of np.sign values. The failing output is named_metrics.json in score_all, using registration.write_once -> json_bytes -> json.dumps. This is a representation/serialization defect; the arithmetic and metric/selection logic are unchanged. A synthetic example reproduces exactly TypeError: Object of type bool is not JSON serializable. The launcher did not persist an original traceback and no test metric was printed in catjet-study.log or tmux capture. Stages a-e have been verified against all recorded artifact hashes.

The registration sole_score.crash_recovery demands a reviewed operational clarification for exact resume. This user-supplied plan is that explicitly reviewed authorization; do not amend the registration or claim an automatic retry was allowed.

## Relevant Files

| Action | Path |
| --- | --- |
| READ | scripts/pc_pipeline.py |
| READ | scripts/phase8/python_pc.py |
| READ | scripts/phase8/pc_python_runtime.py |
| READ | scripts/phase8/saf_surrogate/score.py |
| READ | scripts/phase8/saf_surrogate/registration.py |
| READ | scripts/phase8/saf_surrogate/run.py and train.py |
| READ | scripts/phase8/nozzle_ode/run.py and score.py |
| READ | docs/phase8_saf_surrogate_registration.json |
| MODIFY | either score.py at the failing serialization boundary OR registration.py JSON writer, smallest possible serialization-only change |
| CREATE | scripts/phase8/pc_recover_f_to_i.py |
| CREATE | tests/test_phase8_pc_recovery.py |
| MODIFY | docs/FIXES.md |
| MODIFY | docs/plan.md (already written by planner; do not broaden) |

Old metadata: outputs/phase8/pc_python_pipeline.
SAF attempt: outputs/phase8/saf_surrogate/attempt_001.
New recovery metadata suggestion: outputs/phase8/pc_python_recovery_20261005.
Do not create the new metadata directory in --dry-run.

## Implementation Phases

### Phase 1: Serialization-only fix and fixture regression

Smallest possible conversion of NumPy scalar(s) to native Python values at serialization. Preserve exact existing JSON bytes for ordinary supported values and rejection of invalid/nonfinite values. Do not change claim_concordance arithmetic or gate decisions. Add a regression using entirely manufactured queries/claims/predictions to reproduce the old error and confirm the fixed serialized boolean. Document the cause, fix and exact-run recovery constraints in FIXES.md.

### Phase 2: Narrow integrity-first recovery script

Dry-run is the default; mutually exclusive --dry-run and --run. Verify the known source commit and exact authorized change contents, not merely a filename allowlist. Compare current source_hashes against old scientific_sources.json, allowing only the reviewed fix and additive script. Do not rebaseline arbitrary current files. Verify old completed checkpoints using validate_checkpoint and their actual original configuration SHA.

Use these independently captured immutable hashes:
score_reservation.json d03bfdc00d6383ff32787420ef64413a73b38c5f0a32b565340077f0ca937924
predictions_freeze.json f8973dd677d667cc087274221beba43b14e7b2c450869805b39a775122a2dbbb
selection.json 6c76f8457fc588fd5da68715b5a391e27cf520a782bb48b903cc80466f1abb51
test_metrics.json df550d78e0e781ae5a1b82f66e37326e94b10c1113f93789ee83c83ddc7422b6
ranking_metrics.json 26ef11f4af2554f6cfd3a458f9f34838563819de25b8148ac85e2611f3a83564

Verify frozen predictions/weights/inputs and every target in the existing reservation. Reject symlinks, escaped paths, changed files, foreign identities, ambiguous active owners, unexpected scientific outputs and any prior recovery run. Snapshot the entire old metadata tree and all preexisting SAF files; preserve their bytes and names. Keep all original sealed and source identity provenance.

Run the identical existing score_all logic once, using the original identity/reservation as bound, without writing/replacing a reservation, predictions or weights. A narrowly scoped writer adapter may compare existing writes against exactly regenerated bytes and return only if byte-identical; it must reject any mismatch before mutation and allow no generic overwrite. Seal artifacts and targets remain original. Record the old vs new source mappings plus the explicitly authorized operational mapping, so unchanged producer proofs can be consumed honestly by g and h with current source integrity still enforced. Do not silently skip identity checks or manufacture a new producer receipt.

Use actual existing ScientificWorkflow g, h and publish_outputs implementations. New metadata holds the new source snapshot and authentic stage records and recovery link to unchanged original a-e/f metadata. Reuse their ownership, power, provenance, threads, budgets, timing accounting and scientific verdict behavior. Explicit, verified bridge of old generated data identity to new serialization-only code is required; no modifications to python_pc.py or pc_python_runtime.py. Ensure stage h includes the originally incurred stages c-e cost plus f and g appropriately; new metadata must not omit setup cost by dropping old checkpoints.

Create a durable consumed marker exclusively before real work. Any second --run refuses, whether successful, interrupted or failed. Preserve outputs and traceback on failure. New metadata only; old metadata never modified. No scientific execution in dry-run. Show verification and intended actions without printing numerical test results.

### Phase 3: Verification and review handoff

Tests must cover serialization, unchanged old bytes, target/seal drift rejection, unauthorized source drift, byte mismatch rejection, reservation reuse, dry-run no writes, one-attempt refusal, and operational source bridge without bypassing checks. Use manufactured fixtures; never run the real score in tests. Run exactly the requested tests and git diff --check. Run the real read-only recovery --dry-run only after these pass. Stop for root review. Do not run --run or push yourself.

## File-Level Edits

Serialization fix: one narrow boundary conversion, with no scientific logic changes.
Recovery script: additive operational entrypoint, integrity checks and exact score resume only.
Test file: manufactured regression and negative controls.
FIXES.md: serialization cause and reviewed exact recovery; disclose source snapshot transition.
plan.md: planner's approved instructions, includes the exact user block below.

## Commands to Run

In WSL Ubuntu Linux home, activate catjet-pc. Export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CATJET_ML_BACKEND=torch PYTHONDONTWRITEBYTECODE=1.
python -m pytest tests/test_phase8_pc_recovery.py tests/test_phase8_pc_pipeline.py tests/test_ml_backend.py <existing saf_surrogate test filenames discovered in tests/> -q
git diff --check
python scripts/phase8/pc_recover_f_to_i.py --dry-run

Root after review: commit only authorized source/test/docs files on pc-recovery-20261005, push that branch without force, report dry-run output, then run python scripts/phase8/pc_recover_f_to_i.py --run exactly once inside tmux. If push fails, stop and report per user; do not start real recovery until the requested prior branch push succeeds.

## Tests

The new unit regression proves the unconverted synthetic NumPy bool raises the old TypeError, while fixed output preserves boolean meaning and native JSON formatting. Integrity tests deny changed checkpoints, seals/targets, source logic or arbitrary fix-file edits, old-file byte mismatches, active owners, and second attempts. Verify no dry-run writes or scientific calls. Required existing tests include PC pipeline, ML backend and SAF surrogate fixtures. The known unrelated test_all20_eligible_families is excluded by the user's exact test scope; do not repair it.

## Acceptance Criteria

The requested branch and only authorized modifications. No changes to any metric, model, data or frozen file. All requested fixtures pass (expected optional MLX skips). Dry-run verifies immutable original receipts/seals and source allowlist without writes. Exact score reuse and existing partial-byte equality have negative controls. Real recovery is armed once, preserves old metadata and original reservation/predictions/weights, runs actual g/h/i, and pushes a pc-run branch, or stops with exact failure evidence. No interpretation beyond registered criteria and no freeze before separate user go-ahead.

## Rollback Notes

Never reset, clean, delete, replace or rerun consumed files. Before any real recovery only revert task source commits if directed. Once real recovery is consumed, preserve all evidence and report; do not retry scoring. Original models and metadata must survive every failure.

## Escalation Guidance

This is a high-integrity operational recovery with source identity bridging and one-shot scoring. The approved Claude dispatcher selects the model. If a bridge cannot honestly preserve every invariant without metric or registration edits, stop and explain. No alternative training/scoring procedures.

## Exact user-approved recovery block

```text
STAGE F RECOVERY. Stage f crashed with "TypeError: Object of type bool is not JSON serializable" after score_reservation.json and predictions_freeze.json were written. The SAF registration allows crash recovery that resumes the identical hashed reservation. The pipeline refuses (f.started exists, no receipt), and any code fix changes the source hashes captured at stage a, which stages g and h check. Do exactly the following, and nothing else.

1. DIAGNOSE (no writes). Report:
   - the full traceback, which value is a numpy bool, and where it is serialized
   - every file in the SAF output dir and the pipeline metadata dir written at or after score_reservation.json (name, size, time)
   - whether any test metric was printed to the terminal or logs
   Continue to step 2 unless the bug is inside metric/selection logic. In that case stop and report.

2. FIX on a new branch pc-recovery-20261005 from main:
   - Smallest possible fix: convert numpy scalars to plain Python types at the failing write (or in the shared JSON writer). Change no metric, threshold, selection, model or data code.
   - Add a unit test that raises the old TypeError without the fix and passes with it.
   - Add a docs/FIXES.md entry.

3. RECOVERY SCRIPT scripts/phase8/pc_recover_f_to_i.py (dry-run by default; --run executes):
   - Verify stages a-e by their recorded checkpoint artifact hashes (reuse validate_checkpoint).
   - Verify score_reservation.json, predictions_freeze.json and selection.json are unchanged, and every target hash in the reservation still matches.
   - Write a recovery record: old vs new source hashes. The only differences allowed are the fix file(s) and this script; anything else -> stop.
   - Recompute stage f with the same score_all logic, REUSING the existing reservation (do not write a new one). If the crash left partial outputs (e.g. CSVs), the recomputed ones must match them byte for byte; any mismatch -> stop.
   - Then run stages g (nozzle ODE), h (timing + break-even) and i (commit + push pc-run branch) exactly as the pipeline would. Record the new source snapshot in a NEW metadata dir; leave the old attempt's metadata untouched.
   - One attempt only. If anything fails after the score is written, keep the outputs and report. No re-scoring.

4. Run: the new test, tests/test_phase8_pc_pipeline.py, tests/test_ml_backend.py and the saf_surrogate tests. Push branch pc-recovery-20261005. Run the script with --dry-run and show me the output, then run it with --run.

5. When done, give the STEP 4 final report (pc-run branch + hash, learning curve, every registered bar with value and PASS/FAIL, nozzle PINN, speed and break-even, warnings).
```


## User-approved continuation (2026-10-05)

The user reviewed the six numerical-fixture errors and their reproduction on unchanged main,
authorized logging them in FIXES.md without fixing or skipping them, and authorized committing
and pushing the existing fix/tests/recovery script, repeating the dry-run once on the committed
tree, and executing the single real recovery. Pre-existing test failures reproduced on unchanged
main are recorded and do not stop execution. New failures, recovery failures and anything affecting
the sealed score stop execution. Recomputed test_metrics.json and ranking_metrics.json must match
their crash-time bytes. Product deployment validation versus changed source provenance is explicitly
deferred to the freeze step and must not be changed during this continuation.
