# Python-v6 study on Ryzen 5 3600 / WSL2 Ubuntu

The PC simulator is the full Python v6 path. Existing frozen G0 is PASS; the
PC command verifies 20 frozen v6 rows at `rtol=1e-9` before new computation. The
Mac C++ core stays optional. No C++ compiler/build/conda Cantera development
package is required on this PC. This branch has no new scientific study results.

Torch CPU float64 is primary for training/scoring. The RX 5700 XT is outside the
official ROCm support list, so use the CPU wheel. MLX float32 training and CPU64
scoring are optional on a supported Mac. See the
[AMD compatibility matrix](https://rocm.docs.amd.com/en/docs-7.0.2/compatibility/compatibility-matrix.html).

## Install in WSL2 Ubuntu

Follow [Microsoft's WSL setup](https://learn.microsoft.com/en-us/windows/wsl/install).
If WSL2 Ubuntu is not installed, run this in an Administrator PowerShell and
restart Windows when prompted:

```powershell
wsl --install -d Ubuntu
```

In the Ubuntu terminal, install the Linux
[Miniforge installer](https://github.com/conda-forge/miniforge):

```bash
sudo apt-get update
sudo apt-get install -y git curl
curl -L -o Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
mkdir -p ~/projects
cd ~/projects
```

Keep the clone in the Linux filesystem. Then clone and create the environment:

```bash
git clone --branch phase8-pc-backend-20261004 https://github.com/ArnavPatil-official/JetEngineSimulation.git
cd JetEngineSimulation
conda create -n catjet-pc -c conda-forge python=3.12 pip
conda activate catjet-pc
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c 'import torch, cantera; print(torch.__version__, torch.cuda.is_available(), cantera.__version__)'
```

requirements.txt pins Cantera 3.2.0 installed by pip. The CPU wheel command is
from [PyTorch 2.9.1 installation instructions](https://pytorch.org/get-started/previous-versions/).
Do not copy Mac binaries or environment folders to the PC.

## Inspect and run

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CATJET_ML_BACKEND=torch
python scripts/pc_pipeline.py --help
python scripts/pc_pipeline.py --dry-run --backend torch --device cpu --workers 10
python scripts/pc_pipeline.py --run --backend torch --device cpu --workers 10
```

The command records environment/hardware, checks 20-row parity, runs P7.3-A1,
generates SAF data, trains 24 learning-curve models (N 64/256/1024/4096, two arms,
three fixed seeds), seals/opens the registered sole test score, runs the nozzle
ODE PINN, measures actual simulator 1/10-worker and surrogate timings with
break-even, then commits its own outputs to pc-run-date and pushes without force.
Configure Git identity/authentication before the final output publication stage.
Existing tracked/untracked unrelated files are never swept into that commit.

Resume uses verified stage checkpoints:

```bash
python scripts/pc_pipeline.py --run --resume --backend torch --device cpu --workers 10
```

Completed checkpoints must match source, registration and artifact hashes.
Interrupted or consumed partial scientific stages are retained and fail closed;
resume cannot reopen a score or overwrite earlier trained weights. The command
reports the exact retained checkpoint/attempt if intervention is required.
Publication can retry the same recorded commit after a push failure.
See --help for output/checkpoint paths and stop-after options.

Linux desktops with no battery are recorded as mains; laptops read sysfs power.
No pmset/sysctl/caffeinate is required on Linux. Mac execution keeps its native
power and sleep-inhibition behavior. Ownership records bind host, PID and birth.
Ten simulator workers each use one OpenMP/OpenBLAS thread; Torch CPU kernels also
use one thread, preserving the recorded worker/thread configuration.

## Backend and precision

--backend overrides CATJET_ML_BACKEND; the default is torch. Torch float64
weights remain float64 in framework-neutral NPZ. Optional MLX float32 weights are
promoted only for CPU64 scoring; promotion cannot recover float32 precision.
Identical weights are used for the parity comparison. The shared
interface supports parameter gradients and first/second input derivatives.
Torch supports Adam and optional L-BFGS polish; the primary nozzle retains its
registered polish, optional MLX records Adam-only behavior.

PC operational checks use the actual CPU timing and break-even result. An
unavailable GPU is recorded separately and does not block the primary PC path.

## Published output files

The 640,000-row study JSON exceeds
[GitHub's regular file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Publication stores oversized outputs as lossless gzip chunks with SHA256
manifests. Each chunk is at most 50 MiB. Original outputs stay intact on the PC;
Git LFS is not required.

After cloning a completed `pc-run-<date>` branch, restore the original files
before loading or reviewing that run's artifacts:

```bash
python scripts/pc_pipeline.py --restore-artifacts
```

Restoration verifies all recorded bytes and refuses to replace a differing
existing file. This command performs no training or scoring.

The prospective amendment is
[phase8_backend_pc_amendment.json](phase8_backend_pc_amendment.json), committed
before fixtures/training. No actual Ryzen/WSL2 study was run while preparing
these files. Scientific PASS/FAIL and measured speeds must come from PC outputs.
