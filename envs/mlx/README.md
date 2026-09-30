# `catjet-mlx`: MLX environment and parity gate (Phase 8, Track D3)

## Purpose

This environment holds the Apple-silicon (MLX) scaffolding for the Phase 8
learned models, plus a PyTorch twin of each model for the MLX-vs-PyTorch
parity test:

- **M1**: residual MLP (3 × 64, SiLU, output bounded by tanh, |Δ| ≤ bound),
  with an ensemble wrapper (independent init seeds, bootstrap over groups).
- **M4**: emulator MLP (4 × 128, SiLU, linear output).
- **M3**: nozzle PINN skeleton. A field network maps x to (ρ, u, p), and a
  quasi-1D composite physics loss covers mass, energy, momentum
  (d(pA + ρu²A)/dx − p dA/dx via `mx.vmap(mx.grad(.))` on the input), an
  entropy hinge max(0, s_in − s)² and the boundary conditions. It supports
  four weightings: fixed, gradient-norm (Wang et al. 2021), ReLoBRaLo, and
  the Ma et al. sigmoid weights (detached).

**No model is trained on project data (simulator output or empirical
targets) before gate G2 passes.** Everything here is infrastructure. The
tests use only seeded random numbers, a toy nozzle and a toy analytic
function.

Code: `scripts/phase8/ml/`. Tests: `tests/test_phase8_mlx_parity.py`.

| module | imports | role |
|---|---|---|
| `spec.py` | numpy | `MLPSpec` (M1/M3/M4 architectures), `Q1DProblem`, parity metric |
| `nondim.py` | numpy | `Scaler`: fit once on training rows, frozen, float64, sha256 of fit rows |
| `weighting.py` | numpy | fixed / gradnorm / ReLoBRaLo / Ma-sigmoid weights (Python floats) |
| `score64.py` | numpy (mlx lazily) | float64 export and forward pass, float64 scoring |
| `models_torch.py` | torch | PyTorch twins, `copy_mlx_to_torch`, torch physics loss |
| `models_mlx.py` | mlx | MLX models, ensemble, bootstrap, M3 physics loss |
| `train_mlx.py` | mlx | compiled Adam loop, seeding, sha256 checkpoints, `max_sample_sd` |
| `parity.py` | mlx + torch | parity measurements (prints JSON when run as a script) |

The main `.venv` has no mlx. There, the parity test module is skipped by
`pytest.importorskip("mlx.core")`, and `nondim`, `spec`, `weighting`,
`score64` and `models_torch` all import cleanly.

## Recreate / run

```bash
# from the spec
~/miniforge3/bin/conda env create -f envs/mlx/environment.yml
# exact builds
~/miniforge3/bin/conda create -n catjet-mlx --file envs/mlx/conda-lock-osx-arm64.txt
~/miniforge3/envs/catjet-mlx/bin/python -m pip install -r envs/mlx/pip-freeze.txt

# parity gate (all must pass)
~/miniforge3/envs/catjet-mlx/bin/python -m pytest tests/test_phase8_mlx_parity.py -v
# numbers as JSON
~/miniforge3/envs/catjet-mlx/bin/python scripts/phase8/ml/parity.py
# main venv: must SKIP, not error
.venv/bin/python -m pytest tests/test_phase8_mlx_parity.py -q
```

Pins: python 3.12.14 (conda-forge), mlx 0.32.3 (+ mlx-metal 0.32.3),
torch 2.9.1 (the project pin, used as the parity reference only),
numpy 2.5.3, pytest 9.1.1. numpy is installed from PyPI (Accelerate BLAS)
rather than conda-forge. The conda-forge numpy links OpenBLAS built with
llvm-openmp, and the torch wheel bundles its own `libomp.dylib`, so
importing both aborted with `OMP: Error #15`.

## Precision rule

1. **Non-dimensionalise**: map every physical input and output to O(1)
   with `nondim.Scaler`, fit once on the training rows only.
2. **Train in float32 on the GPU** (MLX, Metal).
3. **Score in float64 on the CPU**: export the weights to numpy float64
   (`score64.export_params64`), run the float64 forward pass
   (`score64.forward64`/`predict64`) and map back with the float64 scaler.
   MLX has no float64 on the Metal GPU, so final metrics never come from the
   float32 pass.

## Parity results (measured 2026-09-29, M3 Pro, seed 0)

Metric: max|a − b| / max|b| per tensor (max over tensors for gradients).
Both frameworks run float32 on the CPU with the same weights, and MLX is set
with `mx.set_default_device(mx.cpu)`. Registered tolerance: 1e-5.

| quantity | M1 | M4 | M3 |
|---|---|---|---|
| forward, CPU | 3.6e-7 | 4.1e-7 | 9.3e-8 (fields) |
| loss, CPU | 6.3e-8 | 1.2e-7 | ≤ 7.2e-6 (max over terms; entropy hinge) |
| parameter gradients, CPU | 7.1e-7 | 6.2e-7 | ≤ 4.7e-6 (max over terms; entropy hinge) |
| MLX GPU forward vs torch CPU | 4.9e-7 | 6.6e-7 | 9.3e-8 (fields), 3.8e-6 (loss terms) |
| MLX GPU vs MLX CPU | 3.4e-7 | 6.1e-7 | 3.4e-6 (loss terms) |

M3 details (seed 0): dF/dx 2.5e-7 and d²F/dx² 1.8e-7 (second derivative
with respect to the network input). The entropy hinge is active at 32 of 64
collocation points. Loss values: mass 0, energy 1.4e-7, momentum 4.6e-7,
entropy 7.2e-6, bc 1.7e-7, data 0, composite (fixed) 7.5e-8, composite
(Ma) 6.1e-8. Gradients: mass 2.0e-7, energy 3.7e-7, momentum 2.2e-6 (the
mixed x/θ second derivative), entropy 4.7e-6, bc 4.9e-7, data 1.8e-7,
composite (fixed) 5.0e-7, composite (Ma) 2.3e-7. The float64 scoring path
agrees with the MLX float32 forward pass to 1.2e-7 to 4.0e-7 (CPU and GPU;
M1, M4, M3).

**Seed sweep (M3, seeds 1–5):** the worst MLX-vs-torch difference is 1.5e-5,
at seed 3 on the momentum-residual parameter gradient. That is above 1e-5.
Against a float64 reference built from the same weights, MLX-f32 is off by
1.4e-5 on that gradient and torch-f32 by 7.9e-7. The other seeds' worst
values are 1.0e-6 to 8.4e-6. The momentum residual dF/dx − p dA/dx and the
hinge argument s_in − s are both small differences of O(1) float32 numbers,
so these two terms are limited by float32 cancellation. MLX's
higher-order backward pass has 0.9–18× (median about 4×) the rounding error of torch's on the momentum
gradient. Every other term agrees at about 1e-7. The sweep is checked
against 5e-5. The registered 1e-5 check is seed 0. M1 and M4 pass 1e-5 on
seeds 0–3.

## MLX API notes

- `nn.Linear.weight` is (out, in), and the layer computes `x @ W.T + b`,
  the same as torch, so weights copy over with no transpose.
- `mlx.optimizers.Adam` defaults to `bias_correction=False`. `train_mlx`
  passes `True`, which gives standard Adam.
- **A numpy scalar times an `mx.array` returns a numpy scalar.** The
  expression `np.float64(2.0) * mx.array(3.0)` evaluates to a
  `numpy.float64`, which silently leaves the autodiff graph. Loss weights
  are therefore cast to Python `float` (`composite_loss`).
- Higher-order autodiff works: `mx.vmap(mx.grad(mx.grad(f)))` gives
  d²F/dx², and `nn.value_and_grad` around a loss containing
  `mx.vmap(mx.grad(.))` gives the mixed second derivative. Both work under
  `mx.compile`, with model, optimizer and RNG state as compile
  inputs/outputs. Compiled and eager runs agree to about 1e-6 after one
  step, not bitwise: fusion changes the rounding. Tracing a second-order
  graph for the first time takes seconds, and on tiny problems eager is
  faster.
- Training is bitwise reproducible for a fixed seed on the GPU:
  `mx.random.seed`, keyed init and a numpy batch-order RNG.

## Toy 3-seed training (synthetic function only)

M4 on y = 1000 + 50 sin(πa) + 30 b² (512 training rows, 150 epochs),
scored in float64 on 256 held-out toy rows. Test RMSE was 0.516, 0.420
and 0.463, and MAPE was 0.039 %, 0.034 % and 0.034 %. The registered-style
spread (the maximum over metrics of the sample SD across seeds, ddof=1) is
0.048.
