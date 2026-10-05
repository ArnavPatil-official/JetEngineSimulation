# Phase 8 on a PC

Use Linux or a Linux terminal inside Windows WSL2. The PC pipeline uses PyTorch
for SAF training, optional CUDA, and NumPy CPU64 for scientific scoring and
product inference. A2 is deferred and absent from its stage sequence.
Native Windows Python is not supported by this entry point.

The source and manufactured checks have been reviewed on macOS. A real Linux
C++ build and CUDA study have not yet been run; their results must come from
PC execution. No trained SAF product is shipped or claimed here.

## Clone and create the environment

Install [Linux Miniforge](https://github.com/conda-forge/miniforge) inside Linux.
WSL users should use its Linux installer from a Linux terminal. Keep the clone
in the Linux filesystem, for example under `~/projects`.

```bash
git clone --branch phase8 https://github.com/ArnavPatil-official/JetEngineSimulation.git
cd JetEngineSimulation

conda create -n catjet-cpp-pc -c conda-forge \
  python=3.12 libcantera-devel=3.2.0 cantera=3.2.0 \
  cmake ninja cxx-compiler eigen catch2 fmt yaml-cpp sundials hdf5 \
  libblas liblapack libgomp
conda activate catjet-cpp-pc
export CATJET_CPP_PREFIX="$CONDA_PREFIX"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The conda environment supplies the Linux compiler and C++ Cantera libraries;
the repository `.venv` supplies Python packages and pybind11. The C++ development
package is `libcantera-devel`, described in the
[Cantera 3.2 conda guide](https://www.cantera.org/3.2/install/conda.html).
Keep `requirements.txt` unchanged.

Choose one Torch installation before the remaining requirements. For CPU:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA GPU with a driver supporting CUDA 12.8:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Use the wheel index matching your driver. Official Torch 2.9.1 instructions
also list `cu126` and `cu130`, alongside CPU, in
[PyTorch's previous-version instructions](https://pytorch.org/get-started/previous-versions/).
`--device cuda` refuses unavailable CUDA; `auto` chooses CUDA when available
and otherwise CPU. CPU execution records GPU timing unavailable and cannot
claim the GPU-dependent operational gate.

## Inspect, check and run

Run from the repository root with the conda environment and `.venv` active:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python pc_pipeline.py --help
python pc_pipeline.py --dry-run --device auto --cpp-prefix "$CATJET_CPP_PREFIX"
python pc_pipeline.py --preflight --device auto --cpp-prefix "$CATJET_CPP_PREFIX"
python pc_pipeline.py --run --device auto --cpp-prefix "$CATJET_CPP_PREFIX"
```

Dry-run prints stages and effective paths without scientific imports or writes.
Preflight checks Python/packages, the C++ prefix, power and fresh attempt paths.
Run executes synchronously:

1. Preflight, separate C++ build and C++ tests.
2. Focused regression checks.
3. Fresh local G0 parity using that PC's core.
4. P7.3-A1 conditional v6 screening with all 64 fixed draws.
5. Complete SAF generation, fixed paired fits, sealing, sole scoring, timing and study.
6. CPU64 nozzle study, product checks and quantitative freeze.

Scientific sizes, seeds, epochs, losses, teacher budgets and score rules come
from the existing registrations. Torch initialization and Adam use their
distributions/hyperparameters; they do not produce bitwise MLX weights. Fit
records disclose backend/device/settings. CUDA timing is labeled `torch_cuda`
and includes synchronized transfers and CPU64 postprocessing. CPU64 remains
the scoring path.

Execution ERROR stops the pipeline. Completed scientific FAIL is retained and
does not enable a failed product. Failed G0 or P7.3 parity prevents downstream
science. Missing historical Track 4 proof stays INCOMPLETE; a local nozzle
diagnostic does not claim historical completion. No freeze tag is created.

## Paths and retained attempts

Default build: `cpp/build_pc`; historical `cpp/build` and `cpp/build_next` stay
separate. To build manually:

```bash
bash cpp/build_pc.sh --build-dir cpp/build_pc
"$CATJET_CPP_PREFIX/bin/ctest" --test-dir cpp/build_pc --output-on-failure
```

`--build-dir` selects a separate directory under `cpp/`. `--metadata-dir`
selects a fresh directory under `outputs/phase8/`, for example:

```bash
python pc_pipeline.py --run --device cuda --cpp-prefix "$CATJET_CPP_PREFIX" \
  --build-dir cpp/build_pc --metadata-dir outputs/phase8/pc_pipeline
```

Logs, per-stage records and the pipeline terminal go in that metadata directory.
Fresh G0 tables/result go in its `g0/` subdirectory. Scientific outputs retain
their canonical paths:

| Stage | Output |
|---|---|
| P7.3-A1 | `outputs/phase7/p73_a1_cpp_20261004/` |
| SAF | `outputs/phase8/saf_surrogate/attempt_001/` |
| Nozzle | `outputs/phase8/nozzle_ode/attempt_001/` |
| Product / freeze | Paths printed by dry-run; freeze in `outputs/freeze/` |

These are retained, write-once attempts. There is no resume/retry shortcut that
reopens scoring or overwrites weights. Changing metadata paths does not permit
reuse of consumed scientific outputs. Preserve interrupted outputs/logs.
Concrete inherited defects are in [docs/FIXES.md](docs/FIXES.md) for later repair;
registrations are unchanged.

After a real passing deployment receipt, use the [screening CLI/API](README.md).
Optional top-K simulator verification still uses the historical Mac gate; its
PC portability is deferred in `docs/FIXES.md`.
