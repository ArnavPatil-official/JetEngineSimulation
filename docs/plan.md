# Timing and publication after recorded nozzle FAIL — user-authorized continuation

## Objective

Execute the user's 2026-10-05 NEXT request: diagnose the existing nozzle results read-only, implement an explicit h/i-only continuation in scripts/phase8/pc_recover_f_to_i.py, test and review it, commit and push, dry-run on the committed tree, then run timing and publication once in tmux. The current code/execution scope is this plan; the prior plan retained below documents the already consumed f/g recovery and is not permission to repeat those stages.

## Constraints

Single PC agent only; no spawned agents. Use the repository Claude dispatcher to implement this approved scope. All repository edits, scientific execution and tests remain in /home/arnav_patil/projects/JetEngineSimulation on the WSL Linux filesystem, using catjet-pc Python 3.12, torch 2.9.1+cpu, Cantera 3.2.0, CPU float64, workers 10 and all four thread environment variables set to 1.

Do not rerun f or g, invoke score_all, regenerate predictions, retrain, change selection, modify metrics/thresholds/science/data/models/registrations/amendments/protected files, or delete/reset any consumed attempt. Preserve every existing nozzle byte, terminal, receipt and artifact, including the absent source_property_coverage.json: do not fill it in. Preserve both old metadata directories and every existing SAF file byte-identically. Leave product deployment validation versus changed source provenance for the freeze step.

Do not modify h or i logic: scripts/phase8/python_pc.py::run_timing and scripts/pc_pipeline.py::publish_outputs remain byte-identical. The existing recovery h source bridge is retained. Scientific FAIL/INCOMPLETE is recorded as such, never fabricated COMPLETE. Stop on new test failures or execution/recovery errors. The six known numerical-test fixture errors (Required evidence is absent from committed tree) reproduce on unchanged main and remain logged, not fixed or skipped.

During implementation: do not commit, push, run the actual continuation, run timing/scoring/training, or install anything. Root reviews then performs those authorized actions. Fixture tests and the read-only new-mode dry-run are allowed.

## Repo Context

Known original source main: 0d16bcc245e93172aa4b3749bccc877727ce950c. First serialization recovery commit and branch: 3af5cc879b7916bbe9b21074030d065b2704f534 on pc-recovery-20261005. Stage f completed once using its original reservation: recovered test_metrics.json and ranking_metrics.json match crash-time bytes. Its scientific verdict is FAIL; ranking and precision PASS.

First recovery metadata outputs/phase8/pc_python_recovery_20261005 ended ERROR at g: RuntimeError: Nozzle execution did not complete: FAIL. The nozzle attempt has six frozen fits and completed one scoring pass with full reports/predictions; its terminal errors list is empty, scientific_verdict FAIL, outputs_complete false, execution_complete false. Only the required source_property_coverage.json is absent. The source loader validates 4164 properties and writes source_manifest.json, but writes coverage JSON only in its failure branch. Nozzle input manifest/case CSV/report confirm all 4096 TRAIN plus 68 named properties, yielding 24984 product conditions. Original Track 4 hashes/report/nozzle_scores are absent from both HEAD and original committed source tree; CPU float64 exact oracle code is present and its fresh smooth-reference gate PASS. Do not infer historical Mac proof completion.

The ordinary outer pipeline requires COMPLETE checkpoints. Direct unchanged run_timing only reads the g elapsed_seconds entry to charge incurred cost; it does not require g completion. Implement an honest cost-only g entry if needed, explicitly marked INCOMPLETE, execution_complete false, outputs_complete false, scientific_verdict FAIL, with immutable terminal/failed recovery references and actual elapsed time. Never label it a completed g receipt or include it among validated completed checkpoints. Record that the standard pipeline completion gate is bypassed only under this explicit user instruction, while the unchanged h runner is called directly through the existing h adapter. The user permits this deviation and requires it in the recovery record.

## Relevant Files

MODIFY only scripts/phase8/pc_recover_f_to_i.py, tests/test_phase8_pc_recovery.py, docs/FIXES.md. docs/plan.md is already updated by the planner. READ scripts/pc_pipeline.py, scripts/phase8/python_pc.py, pc_python_runtime.py, pc_runtime.py, saf_surrogate/run.py, score.py, timing.py, study.py, nozzle_ode/run.py, score.py, source manifests, original and first recovery metadata. No other file-level edits authorized.

## Implementation Phases

1. Add explicit --timing-publish-only mode, mutually compatible with --dry-run (default) or --run, that never constructs or calls f/g handlers. Retain the original f/g recovery mode and its one-shot refusal behavior unchanged. The new mode uses a fresh dedicated metadata directory outputs/phase8/pc_python_timing_publish_20261005 with its own durable exclusive consumed marker. It may be run exactly once; after any actual failure preserve evidence and refuse another actual attempt.
2. Read-only verification: validate all original a-e checkpoints using recorded config hashes; validate first recovery completed a/f and any carried checkpoints with their actual recorded config hashes. Verify all original metadata, preexisting SAF manifest entries, completed f artifacts, sealed reservation targets/freeze/selection and exact pinned partial metrics. Verify all first recovery metadata and all nozzle artifact hashes, six checkpoints and score reservation checkpoint/input bindings. Require exactly the known first recovery ERROR at g and no h/i start, receipt, reservation, output or live owner anywhere. Require only the known missing coverage artifact; reject any other incomplete/drift/error condition. Verify source provenance against original source and first recovery snapshot, allowing only the previously approved literal serialization fix and current recovery-script edit; reject any other scientific source change. Current changes since original commit must remain within the previously authorized five files.
3. Record read-only manifest snapshots of original metadata, first recovery metadata, current pre-timing SAF and nozzle trees; the known missing output and truthful nozzle terminal/report/completion states; old versus current source hashes and the current script hash; exact source commit and first recovery commit; current branch/head; user authorization; carried completed stages; h/i-only plan and deviation; authentic per-stage setup costs including consumed original f cost already charged in f receipt and the actual failed g attempt. Do not mutate previous records or rebaseline any changed protected artifact. Root may supply independently captured hashes if needed; do not silently accept arbitrary current nozzle bytes without checking its recorded hashes.
4. Prepare only NEW metadata for current sources/config/environment/parity. Carry authentic prior b-e and f checkpoints/proof records byte-for-byte, including producer d/e/f raw reservation/terminal/released-lease files needed by h history. Preserve original checkpoint config hashes and source identities. No fake g completion. A cost-only g reference must remain explicitly incomplete and only be used by unchanged h setup arithmetic, not validated as a completed stage. Call only existing RecoveryWorkflow.h (unchanged bridge + unchanged ScientificWorkflow.h/run_timing) and publish_outputs(config), validating actual h/i outputs and writing authentic new h/i receipts. Recheck all retained trees before h, after h, and after i; SAF may add only legitimate h outputs. g completion, scientific verdict, and missing file remain unchanged. Final summary includes the carried f/nozzle FAIL/INCOMPLETE, independent of publication success.
5. Add meaningful manufactured regression fixtures proving no f/g calls or score imports/reruns, dry-run no writes, output-drift and source-drift refusals, existing/pending h/i refusal, one-shot consumption, no fabricated COMPLETE g receipt, incomplete g recorded and actual h/i adapter called, honest cost accounting, original/nozzle tree byte preservation, and failure retention/no automatic retries. Document this authorized deviation and existing deployment provenance freeze deferral in FIXES.md.

## File-Level Edits

scripts/phase8/pc_recover_f_to_i.py: additive h/i-only verifier, description, metadata preparation and one-attempt dispatcher; CLI flag. Keep f scoring logic, g scientific code, existing h source adapter and publication logic unchanged. Do not make g silently complete; classify the saved diagnostic explicitly. Prefer reusable existing integrity helpers to duplicating or broadening them.

tests/test_phase8_pc_recovery.py: add manufactured recovery-continuation fixtures covering boundaries above. No real study paths/metric targets used for training/scoring tests.

docs/FIXES.md: explain missing nozzle coverage receipt file, evidence that all input properties loaded, no science fix performed, authentic FAIL/INCOMPLETE retention, explicit timing-only continuation and setup-cost accounting, inherited numerical fixture errors unchanged, freeze issue deferred.

## Commands to Run

source ~/miniforge3/etc/profile.d/conda.sh; conda activate catjet-pc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CATJET_ML_BACKEND=torch PYTHONDONTWRITEBYTECODE=1
python -m pytest tests/test_phase8_pc_recovery.py tests/test_phase8_pc_pipeline.py tests/test_ml_backend.py tests/test_phase8_saf_surrogate.py tests/test_phase8_saf_surrogate_numerical.py tests/test_phase8_saf_surrogate_torch.py -q
git diff --check
python scripts/phase8/pc_recover_f_to_i.py --timing-publish-only --dry-run

Root only, after review: commit exactly the four authorized changed files on pc-recovery-20261005, git push origin pc-recovery-20261005, committed-tree --timing-publish-only --dry-run, then --timing-publish-only --run exactly once inside tmux. No ordinary pipeline --resume, no f/g recovery --run, no force push.

## Tests

All new fixtures and existing recovery/backend/pipeline/SAF fixtures pass. Six numerical fixture errors may persist exactly as recorded on unchanged main; do not fix/skip them. Any new failure is an execution stop. Verify h/i runner source file hashes match first recovery commit and original main; registration serialization fix remains byte-identical to its reviewed patch. Verify all preexisting scientific and metadata bytes after executor completion; no unit fixture may touch real attempt files.

## Acceptance Criteria

Read-only nozzle diagnostic reports best physics_on seed by recorded panel RMSE, per-field worst condition max/RMS versus 0.5%/0.1%, full property coverage proof, and missing historical evidence's actual provenance. Only requested four files changed. Committed-tree dry-run VERIFIED, run_ready true, stages h/i only, no write. Actual run consumes a new marker once, invokes no f/g/model training or score_all, preserves every existing score/nozzle byte, records honest g FAIL/INCOMPLETE and missing coverage, leaves h/i logic unchanged, and publishes via existing pc-run-date behavior. Final user report gives publication branch/hash, actual per-query timings at both worker counts/all requested batches, 640000 run, speedup, registered break-even with cost scope, operational verdict and nozzle diagnostic. If actual recovery fails, stop and report exact exception with all outputs retained.

## Rollback Notes

Before actual h/i only, source changes can be reverted with ordinary reviewed commits. After consumption, never delete marker/reservations, mutate retained scientific outputs, re-score or retry automatically. Preserve a failed timing/publishing attempt and report exact failure. A later attempt requires explicit new user scope; existing f/g outputs are permanent.

## Escalation Guidance

Complexity high: provenance-sensitive resume around consumed scoring, sealed failed nozzle evidence and long-running timing. Let existing dispatcher select its configured Claude model. Do not use parallel agents. Stop rather than relax verification or fabricate completion; root reviews diffs, fixture tests and integrity before real execution.

---

The previous approved f/g recovery plan below is retained as historical authorization; it has already been consumed and must not be re-executed by this NEXT task.

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
