# Backend and Python PC delivery record

Work is isolated on phase8-pc-backend-20261004. The original phase8 checkout
stays at 7c19b5e; its chain, untracked operational files and local settings are
untouched. A2 is deferred and absent from the new PC sequence.

## Push and commit order

The first git push origin --all succeeded without force or rejected branches.
It published these previously local branches:

- phase7
- phase7-p73-a1-registration-20261004
- phase7-p74-wip
- phase8-evening-reports-20261004
- phase8-g0-autorun-20261004
- phase8-nozzle-ode-registration-20261004
- phase8-p83-a2-a4c-20261003
- phase8-p85-a5-20261003
- phase8-p85-safety-20261003
- phase8-queue-recovery-20261003
- phase8-saf-surrogate-amendment-a1-20261004
- phase8-saf-surrogate-registration-20261004
- phase8-screening-operations-20261004

Existing main, phase4 and phase8 were current. The following tag push reported
Everything up-to-date; tags are 1.0.0 and v5.0. GitHub phase8 was verified at
7c19b5e250f5dd964a92c4d249444b0447c5936b.

The prospective amendment commit is d3767dc, before every training fixture.
The backend implementation commit is 6dd62d2, followed by the PC source/docs
commit. The final all-branches push publishes the isolated implementation branch;
the original checkout's phase8 HEAD is preserved for chain identity.

## Manufactured validation

Before the final PC commit, the combined backend/consumer/runtime/pipeline/public
product fixture suite reported 281 passed and 7 optional MLX skips in 8.38 s.
A final scoped Git text-filter fixture was added afterward and is included in
the post-commit repeat. CLI help/dry-run perform no scientific imports or writes.
The Python source capture binds the frozen 20-row reference and frozen split;
no native binary is required. Syntax, diff whitespace, prospective amendment
hashes, chemical data, requirements and existing model-weight checks pass.

Actual installed MLX backend and consumer fixtures together: 43 passed in
4.18 s (17 backend and 26 consumer fixtures). Maximum relative family errors:

| Comparison | Outputs | Loss | Parameter gradients | First input derivative | Second input derivative | Jacobian | Hessian |
|---|---:|---:|---:|---:|---:|---:|---:|
| Torch64 / analytic NumPy64 | 1.31e-16 | 1.60e-16 | 2.75e-16 | 2.37e-16 | 2.87e-16 | 2.01e-16 | 2.54e-16 |
| Torch32 / actual MLX32 | 6.35e-8 | 9.54e-8 | 1.67e-7 | 9.95e-8 | 1.85e-7 | 9.35e-8 | 1.92e-7 |

Neutral NPZ preserves native parameter precision, cross-loads, and includes
activation metadata. Primary scoring executes native Torch float64 on CPU.
Optional MLX scoring executes NumPy float64 on CPU.

Pipeline fixtures execute actual stage helpers with mocked computations. They
check exact 20-row parity and failure stop, 24-fit artifact sealing, one-shot
score protection, completed-checkpoint resume, host/PID/birth ownership, Linux
power, property projection without sealed target reads, genuine Python producer
proofs, actual 1/10-worker timing dispatch, CPU operation without a GPU gate,
scoped output publication, and retry of the same commit after a push failure.
Oversized files publish as lossless gzip chunks with SHA manifests; originals
stay intact locally. Restore a cloned run using --restore-artifacts.

The historical Mac-gated numerical suites and the full inherited suite were
not repeated. Their missing main-dependency/source-extension receipts and
historical chain defect are already recorded in FIXES.md. No replacement
historical receipts were invented.

## PC commands

Use docs/PC_SETUP.md for WSL2 and Miniforge installation. In an Ubuntu terminal:

```bash
git clone --branch phase8-pc-backend-20261004 https://github.com/ArnavPatil-official/JetEngineSimulation.git
cd JetEngineSimulation
conda create -n catjet-pc -c conda-forge python=3.12 pip
conda activate catjet-pc
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CATJET_ML_BACKEND=torch
python scripts/pc_pipeline.py --run --backend torch --device cpu --workers 10
```

Resume with the same command plus --resume. Only byte-verified completed
scientific stages are skipped; consumed partial scientific stages are retained.
Publication retry uses the exact recorded owned commit without force.

No actual Ryzen/WSL2 study, new scientific fit, locked test score or PC speed
measurement was run during this delivery. Those results will be produced on the
PC by the nine-stage command and committed to pc-run-date.
