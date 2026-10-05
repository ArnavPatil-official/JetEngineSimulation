# Authorized push / PyTorch / PC pipeline — 2026-10-04

The user directed immediate work: push first (done at a906ea7), PyTorch backend
(implemented and toy-checked), PC_SETUP.md and pc_pipeline.py, then final push.
A2 is DEFERRED. Its parked original chain was stopped before any A2 child started.
Existing registrations are sufficient. Do not add registration refinements,
clarifications or handoff rules. User authorization covers the PC/PyTorch host
adaptation; do not request a new registration-level decision for that adaptation.
The earlier detailed Phase 8 plan is retained in Git history at a906ea7.

## Objective

Finish review fixes in the actual PyTorch backend, then deliver a runnable
Linux/WSL2 PC screening pipeline and setup guide. Keep all fixed scientific
procedures, losses, budgets, locked scoring and NumPy CPU64 scoring unchanged.

## Constraints

Do not change scientific registrations, chemical YAML, requirements.txt,
protected Python v6 sources or existing model/results files. Do not modify the
historical Mac chain, its source pins or historical evidence. New PC computation
records its actual local sources/core/backend/hardware and does not claim Mac
chain completion. No numerical study, training, scoring or rebuild on this Mac.
Use existing scientific helpers and existing ownership primitives; do not create
additional approval/arming/config/handoff policies. Preserve unrelated dirt.
Only concrete defects go in docs/FIXES.md, and existing unrelated defects remain
for later repair. The two new-backend implementation defects below need correction
before shipping. Do not push or commit; root reviews and performs final push.

## Repo Context

SAF and shared product implementation sources are now integrated as uncommitted
additions. The backend toy suite passes in .venv and catjet-mlx. The old gate pins
a Mac build and rejects new backend paths; PC must use an explicit local execution
profile with real provenance, not change existing registrations or imitate Mac
receipts. The nozzle model already uses Torch CPU64 and must retain that procedure.

## Relevant Files

READ AGENTS.md, existing SAF/nozzle/P7.3 registrations, cpp/CMakeLists.txt,
scripts/phase8/scientific_workflow_gate.py and existing implementations.
MODIFY scripts/phase8/saf_surrogate/train_torch.py, run.py, models.py, timing.py,
registration.py, teacher.py, study.py only for concrete fixes/portable injection.
CREATE PC_SETUP.md and pc_pipeline.py at repository root.
CREATE scripts/phase8/pc_runtime.py (only if useful to keep CLI maintainable).
CREATE scripts/phase8/pc_saf.py and pc_finish.py as portable stage adapters.
CREATE tests/test_phase8_pc_runtime.py, test_phase8_pc_saf.py and
test_phase8_pc_finish.py for manufactured integration checks.
CREATE cpp/pc/CMakeLists.txt and cpp/build_pc.sh; MODIFY .gitignore for build_pc.
COPY only scripts/phase8/p73_a1_cpp.py and tests/test_phase7_p73_a1_cpp.py from
/Users/arnavpatil/Documents/JetEngineSimulation-worktrees/p73-a1-registration;
COPY scripts/phase8/nozzle_ode/*.py and tests/test_phase8_nozzle_ode.py from
/Users/arnavpatil/Documents/JetEngineSimulation-worktrees/nozzle-ode-registration.
Do not copy bootstrap or registration changes. Minimal injected context/output
parameters in these new consumers are permitted; preserve default Mac behavior.
CREATE tests/test_phase8_pc_pipeline.py; extend Torch tests meaningfully.
MODIFY README.md, docs/FIXES.md, outputs/phase8_execution_status.md for actual state.

## Implementation Phases

### Phase 1 — Finish the backend before PC authoring

Fix run.pipeline's thread environment validation/setup before Torch import and
apply torch.set_num_threads(1) for pre-imported Torch. GPU metadata must describe
the actual selected CUDA device rather than hardcode index0 while model.to(cuda)
uses current_device. Test both defects and rerun the Torch suite. Then proceed.

### Phase 2 — PC setup and build

Support Linux and Windows via WSL2; native Windows need not be supported. Use
Python3.12 and pinned requirements. Document fresh clone/checkout of phase8,
conda C++ dependencies including libcantera-devel3.2.0, CUDA2.9.1 or CPU Torch,
setup/build/preflight/dry-run/run, output locations and resuming limitations.
Official verified links: https://pytorch.org/get-started/previous-versions/ and
https://www.cantera.org/3.2/install/conda.html . Torch2.9.1 official CUDA wheel
indices include cu126/cu128/cu130; CPU index cpu. Keep requirements.txt intact.
Use cpp/build_pc separate from original Mac build. New CMake entry compiles the
same C++ core/bindings/tests with GNU linker symbol isolation and appropriate
OpenMP runtime. Never edit original C++ scientific source or original CMake.

### Phase 3 — Actual sequential PC pipeline, excluding A2

Implement root pc_pipeline.py with --help, --dry-run (no mutation/heavy imports),
--preflight and --run, --device auto|cpu|cuda, configurable compiler prefix/build
location and metadata output location. Default Torch, fixed scientific budgets.
No extra approval/armed-config step. Real stage order: portable preflight/build,
focused checks, local fresh G0 parity, P7.3-A1, complete SAF generation/training/
frozen one-shot scoring/timing/study, nozzle study, product checks, quantitative
freeze. All stages execute synchronously, stop on execution ERROR, retain measured
scientific FAIL honestly, and never run A2 or retry/overwrite consumed outputs.

Reuse existing gate.Run ownership behavior with a small PC context if practical.
The context identifies the real local binary/source hashes and local G0; it never
requires or manufactures a completed Mac main chain. Existing historical source
and registration bytes stay unchanged. Mac-specific operations in NEW consumers
get minimal platform/context hooks. Linux desktop power status can record no
battery; laptops must verify mains. Record actual metadata, never sw_vers/sysctl
on Linux. Build/core selection must be explicit and verified before import.

P7.3 execute already has gate_factory injection: preserve contract/penalty-only
exception/parity/64 fixed draws. Fresh PC clones can use canonical registered
artifact directories if absent, minimizing path changes. For configurable output
paths use explicitly recorded effective path mapping, never rewrite registrations.
SAF reuse input helpers, Thermo export, teacher full_state/parallel generation,
_capture_dataset, source_diagnostics, train_all(backend=torch), seal_predictions,
score_all, measure, study.screen, finalize_timing and deployment_receipt. Keep
train/validation readers separated from sealed test/ranking/named targets.
Torch CUDA timing measures actual f32 GPU batches with synchronization and
CPU64 comparison/postprocess. Label it Torch CUDA, rather than invent Metal
measurements. CPU mode records GPU timing unavailable and retains that limitation.
Product provenance validation should handle PC artifacts using actual local
terminal/source/core hashes, leaving default historical Mac validation unchanged.

Nozzle reuse property_cases/make_splits/oracle/model.train_pair/score_panels and
paired_decisions, preserving CPU64, exact model, fixed seeds and one test opening.
If historical original Track4 rung evidence is unavailable, record that fact;
a fresh local run_ladder self-check in a new directory is local evidence and must
not become a historical PASS. Keep official inherited gate status incomplete
while any diagnostic PC computations/results are explicitly labeled as such.

Freeze reads actual published numeric artifacts, records measured PASS/FAIL and
missing evidence accurately, and produces numbers/figures when supported. Never
invent receipts, scientific success, source identity or tags. Missing inherited
Mac evidence is a concrete existing defect for FIXES.md, not an invitation to
create more rules or amend registrations. Make every exposed command functional;
if an unavoidable inherited gate blocks a scientific claim, report it clearly in
artifacts while delivering runnable diagnostic computation and the PC pipeline.

## File-Level Edits

Backend fixes address import-time thread capture and current-CUDA metadata.
PC_SETUP.md provides copy/paste commands matching implemented CLI/build flags.
pc_pipeline.py is the single PC entry point and explicitly omits A2. The portable
runtime adapts execution environment/provenance and reuses actual scientific
methods. New portable CMake/build files compile existing sources. Existing new
consumer wrappers gain optional injected context/path/profile support only where
needed. Tests verify functionality/failures without project labels or studies.
README links PC_SETUP and states actual availability. Status/FIXES retain defects.

## Commands to Run

.venv/bin/python -m pytest tests/test_phase8_saf_surrogate_torch.py -v
.venv/bin/python pc_pipeline.py --help
.venv/bin/python pc_pipeline.py --dry-run
.venv/bin/python -m pytest tests/test_phase8_pc_pipeline.py -v
Run relevant pure P7.3/nozzle/product tests. Root runs full pytest and protected
hash/config checks afterward. Do not run --run, G0, teacher, full fits, test scoring
or C++ builds here. Do not stage/commit/push; root performs final review and push.

## Tests

Test import-time thread setup with pre-imported Torch, actual CUDA metadata with
mocked selected-device APIs, CPU toy gradients/export/determinism. PC tests must
cover dry-run no imports/no writes, exact sequential stage order without A2,
actual helper dispatch, unavailable CUDA, missing build/dependencies, subprocess
failures, completed FAIL vs ERROR, local source/core drift and fresh-output refusal.
Use short manufactured data, not project label reads or reduced scientific runs.

## Acceptance Criteria

Backend checks pass before PC files are authored. PC commands/build recipe agree,
Linux/WSL2 CPU/CUDA execution is implemented, no A2 step exists in active PC chain,
CPU64 scoring and fixed scientific procedures remain intact, and no registrations
or protected sources/data/models are changed. New scientific claims require real
results and retain inherited limitations. Dry-run/help/pure tests pass. Task files
are reviewable for final commit/push. Actual Linux build/CUDA run remains untested
on this Mac and is stated accurately; do not invent success.

## Rollback Notes

Revert only these source/docs additions and task edits. Never reset unrelated
work or overwrite/delete old evidence, existing weights or score reservations.

## Escalation Guidance

This is a substantive portability implementation; dispatcher chooses the model.
The user already ordered immediate work and final pushes. Resolve routine code
choices without further permission or registration amendments.

The second Claude dispatcher stopped at its session quota after the backend,
portable context and build entry were written. The existing explicit Codex
executor override recorded in docs/FIXES.md applies to finishing this work.
Concrete review defects are logged there before implementation corrections.
