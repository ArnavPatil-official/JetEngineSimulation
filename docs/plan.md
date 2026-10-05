# Shared ML backend and resumable Python v6 PC study

## Objective

Push all branches and tags, implement the requested shared ML backend and PC pipeline,
commit the changes, run fixture tests, then push all branches and tags again.

## Constraints

Work only in the isolated pc-backend worktree. Preserve the original checkout,
running chain, lease, operational files and untracked files. Do not run the actual
study here. Commit the prospective amendment before any training fixture. Preserve
chemical data, existing model weights, frozen results and consumed score locks.
Use the existing user authorization for Codex execution.

## Repo Context

The active Phase 8 SAF and nozzle studies need a shared Torch/MLX backend. The PC
uses the full Python v6 simulator with actual local source, host and PID identity.
The Mac C++ path and historical chain remain optional and untouched.

## Relevant Files

Create simulation/ml_backend/{__init__,interface,numpy_reference,torch_backend,
mlx_backend}.py and tests/test_ml_backend.py. Create the prospective backend/PC
amendment JSON and Markdown; update the SAF/nozzle registration JSON only for the
requested backend, dtype and four-size schedule.

Update scripts/phase8/saf_surrogate/{models,train,train_torch,thermo,registration,
score,run,timing,teacher}.py, nozzle_ode/{model,run,score}.py and p73_a1_cpp.py.
Create scripts/pc_pipeline.py, scripts/phase8/pc_python_runtime.py,
simulation/runtime.py, docs/PC_SETUP.md, docs/pc_backend_execution.md and manufactured backend/pipeline fixtures.
Update the root PC_SETUP.md and pc_pipeline.py compatibility entries, README.md,
docs/FIXES.md and execution status documentation. New PC fixtures are
tests/test_pc_backend_review.py, tests/test_pc_python_pipeline.py and
tests/test_pc_scientific_dispatch.py. The oversized-output correction adds
scripts/phase8/artifact_transport.py and tests/test_artifact_transport.py.

## Implementation Phases

1. Push origin --all, then origin --tags without force. Commit the prospective
   amendment before fits and record that no new study results exist.
2. Implement shared arrays/dtype, MLP, losses, Adam, optional Torch L-BFGS,
   parameter/input gradients, second derivatives, seeding, devices and neutral
   NPZ checkpoints. Integrate active study training and scoring entry points.
3. Implement the nine PC stages, checkpoints and platform-aware runtime. Review
   code, commit changes, run fixture tests and repeat the two pushes.

## File-Level Edits

Torch defaults to CPU float64 training and scoring; MLX uses float32 training and
NumPy CPU float64 scoring. Selection uses --backend, then CATJET_ML_BACKEND, then
Torch. Preserve float64 exports and scientific losses, labels, seeds and Sobol
samples. The SAF schedule has four sizes, two arms and three seeds. The primary
nozzle retains its Adam/L-BFGS budgets; MLX records its Adam-only procedure.

The PC command records hardware, verifies 20 frozen v6 rows at rtol=1e-9, runs
P7.3-A1, generates SAF data, trains 24 fits, scores once, runs the nozzle PINN,
measures simulator timing with one and ten workers plus surrogate timing and
break-even, then commits its outputs to pc-run-date and pushes. It omits A2.
Checkpoint validation protects completed stages and consumed scoring. Provenance
identifies Python sources without manufacturing a C++ identity. Stage generation
receipts authenticate actual generated data without claiming a completed study.
Document WSL2, Miniforge, pip Cantera 3.2.0, Torch CPU wheels and thread settings.

## Commands to Run

Use the original checkout's .venv Python with this worktree as cwd for fixture
pytest and syntax checks. Use the installed catjet-mlx Python for tiny actual
backend parity fixtures after the amendment commit. Run git diff --check and CLI
help/dry-run checks. Do not run PC --run or project training here. Finally run
git push origin --all, then git push origin --tags and verify remote refs.

## Tests

Compare identical weights, outputs, losses, parameter gradients and first/second
input derivatives: float32 relative tolerance 1e-5; Torch float64 versus an
independent analytic NumPy reference 1e-10. Skip MLX cleanly when unavailable.
Check neutral weight transfer, dtype preservation, CLI/env selection and optimizer
options. Check Linux desktop mains, Mac behavior, host/PID ownership, parity
failure, checkpoint drift, one-shot score protection and scoped Git publication.
Use manufactured fixtures without project training or locked test evaluation.

## Acceptance Criteria

Both pushes follow the requested order and use no force. Original operational and
untracked files stay untouched. The amendment precedes training fixtures. Backend
parity passes and selected frameworks actually execute. The documented PC command
runs stages a-i without C++ or Mac prerequisites and resumes verified checkpoints.
Report actual test results and inherited defects; do not invent PC study results.

## Rollback Notes

Revert isolated task commits before PC computation. Retain consumed attempts and
locks. Never reset the original checkout, chain, weights or historical evidence.

## Escalation Guidance

This is a complex ML/PINN/platform integration. Independent agents own shared
backend, consumer integration and PC dispatch files. Root reviews and commits.
The existing explicit Codex executor authorization applies. No further
registration refinements or handoff rules are introduced.
