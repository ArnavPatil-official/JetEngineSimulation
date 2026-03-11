# LE-PINN Implementation Plan

## Objective

Implement the LE-PINN (Locally Enhanced Physics-Informed Neural Network) architecture from Ma et al. as a **new standalone module** in `simulation/nozzle/le_pinn.py`. The existing 1D PINN in `nozzle.py` is preserved unchanged.

The module implements a dual-network architecture for 2D RANS nozzle flow field prediction with 9 output variables, physics-informed loss (RANS equations + ideal gas EOS), wall boundary conditions, distance-based fusion, and adaptive loss weighting.

Wall geometry comes from `scripts/visualization/nozzle_2d_geometry.py` (`generate_nozzle_profile()`).
Training data is synthetic, generated analytically from isentropic + ideal-gas relations on the 2D nozzle grid.

## Constraints

- **Do NOT modify** any existing files — this is a pure additive change
- Existing `nozzle_pinn.pt` and `turbine_pinn.pt` must be preserved untouched
- Chemical data files (`data/*.yaml`) are read-only
- All new code must use type hints and docstrings
- Use fixed random seed (42) for reproducibility
- Must be compatible with PyTorch (already in requirements.txt)

## Repo Context

Subsystems involved: nozzle (PINN surrogate model), geometry (2D wall profile generation).

The geometry module `scripts/visualization/nozzle_2d_geometry.py` exports:
- `generate_nozzle_profile(NPR, AR, Throat_Radius)` → `NozzleProfile` dataclass
- `NozzleProfile` has fields: `x` (ndarray), `y` (ndarray), `key_points` (dict), `geometry` (dict)
- `geometry` dict contains `A5` (throat area = π·r_t²), `A6` (exit area = π·r_exit²), `r_throat`, `r_exit`, `r_inlet`

## Relevant Files

| Action | Path | Purpose |
|--------|------|---------|
| READ | `scripts/visualization/nozzle_2d_geometry.py` | Wall geometry API |
| READ | `simulation/nozzle/nozzle.py` | Reference for existing patterns |
| CREATE | `simulation/nozzle/le_pinn.py` | New LE-PINN module |
| CREATE | `tests/test_le_pinn.py` | Unit tests |

## Implementation Phases

### Phase 1: Core Architecture (`le_pinn.py` — Network Classes)

Create `simulation/nozzle/le_pinn.py` with these classes:

**1a. `GlobalNetwork(nn.Module)`**
- Architecture: Input(6) → Linear(6, 400) → ReLU → [Linear(400, 400) → ReLU] × 6 → Linear(400, 9)
- That is: 1 input layer + 6 hidden layers + 1 output layer
- Xavier uniform initialization on all weights, zero initialization on all biases
- Input: `(N, 6)` tensor `[x, y, A5, A6, P_in, T_in]`
- Output: `(N, 9)` tensor `[ρ̂, û, v̂, P̂, T̂, ÛÛ, V̂V̂, ÛV̂, μ̂_eff]`

**1b. `BoundaryNetwork(nn.Module)`**
- Architecture: Input(6) → Linear(6, 100) → ReLU → [Linear(100, 100) → ReLU] × 6 → Linear(100, 2)
- That is: 1 input layer + 6 hidden layers + 1 output layer
- Xavier uniform initialization on all weights, zero initialization on all biases
- Input: `(N, 6)` tensor `[x_b, y_b, A5, A6, P_in, T_in]`
- Output: `(N, 2)` tensor `[P̂_b, T̂_b]`

**1c. `LE_PINN(nn.Module)`**
- Contains `GlobalNetwork` and `BoundaryNetwork`
- `self.delta = 5e-4` (fusion threshold)
- `forward(inputs, wall_distances)`:
  1. Get global predictions: `global_preds = self.global_net(inputs)` → `(N, 9)`
  2. Get boundary predictions: `boundary_preds = self.boundary_net(inputs)` → `(N, 2)`
  3. Create mask: `near_wall = (wall_distances < self.delta).squeeze()`
  4. Clone global: `fused = global_preds.clone()`
  5. Replace pressure: `fused[near_wall, 3] = boundary_preds[near_wall, 0]`
  6. Replace temperature: `fused[near_wall, 4] = boundary_preds[near_wall, 1]`
  7. Return `fused` → `(N, 9)`

**1d. `MinMaxNormalizer` class**
- `__init__(self, epsilon=1e-8)`: stores epsilon
- `fit(self, data)`: compute and store `data_min`, `data_max`
- `transform(self, data)`: `(data - data_min) / (data_max - data_min + epsilon)`
- `inverse_transform(self, data_norm)`: reverse operation
- Applies to both inputs and outputs

### Phase 2: Physics & Loss Functions

**2a. `compute_rans_residuals(inputs, outputs)` function**
- `inputs` is `(N, 6)` with `requires_grad=True` for x, y columns
- `outputs` is `(N, 9)`: `[ρ, u, v, P, T, UU, VV, UV, μ_eff]`
- Use `torch.autograd.grad` to compute spatial derivatives:
  - `∂ρ/∂x, ∂ρ/∂y, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂P/∂x, ∂P/∂y, ∂T/∂x, ∂T/∂y`
  - Second derivatives for viscous terms: `∂²u/∂x², ∂²u/∂y², ∂²v/∂x², ∂²v/∂y², ∂²T/∂x², ∂²T/∂y²`
- Compute RANS equation residuals (using the Reynolds stress components from network output):
  - Mass: `∂(ρu)/∂x + ∂(ρv)/∂y = 0`
  - X-momentum: `ρ(u·∂u/∂x + v·∂u/∂y) + ∂P/∂x - μ_eff·(∂²u/∂x² + ∂²u/∂y²) + ρ·(∂UU/∂x + ∂UV/∂y) = 0`
  - Y-momentum: `ρ(u·∂v/∂x + v·∂v/∂y) + ∂P/∂y - μ_eff·(∂²v/∂x² + ∂²v/∂y²) + ρ·(∂UV/∂x + ∂VV/∂y) = 0`
  - Energy: `ρ·cp·(u·∂T/∂x + v·∂T/∂y) - k_eff·(∂²T/∂x² + ∂²T/∂y²) = 0` (with k_eff = μ_eff·cp/Pr_t, use Pr_t = 0.9)
- Ideal gas EOS: `P - ρ·R·T = 0` (R = 287.0 J/(kg·K) default, or parameterizable)
- Return: tuple of residual tensors (mass, xmom, ymom, energy, eos)
- `L_physics = mean(mass² + xmom² + ymom² + energy² + eos²)`

**2b. `compute_wall_bc_loss(model, wall_points, wall_normals)` function**
- `wall_points`: `(M, 6)` tensor of points on the wall boundary
- `wall_normals`: `(M, 2)` tensor of outward normal vectors at wall points
- Evaluate model at wall points: `preds = model(wall_points, wall_distances=zeros)`
- Enforce:
  - `u_wall = 0`: `L_u = mean(preds[:, 1]²)`
  - `v_wall = 0`: `L_v = mean(preds[:, 2]²)`
  - `∂T/∂n = 0` (Neumann): compute `∂T/∂x` and `∂T/∂y` via autograd, then `∂T/∂n = ∂T/∂x·n_x + ∂T/∂y·n_y`, `L_T = mean((∂T/∂n)²)`
- Return: `L_bc = L_u + L_v + L_T`

**2c. `AdaptiveLossWeighting` class**
- Implements sigmoid-based exponential dynamic weighting from the paper
- `__init__(self)`: initialize `self.epoch = 0`
- `compute_weights(self, epoch, loss_data, loss_physics, loss_bc)`:
  - `λ_data = sigmoid(α_d · (epoch / max_epoch))` where α_d is a scaling parameter (use α_d = 5.0)
  - `λ_physics = 1.0 - λ_data + base_weight` (grows as data weight decreases)
  - `λ_bc = sigmoid(α_bc · (epoch / max_epoch))` where α_bc = 3.0
  - Return `(λ_data, λ_physics, λ_bc)`
- The exact sigmoid formulation: `sigmoid(x) = 1 / (1 + exp(-x))`

### Phase 3: Synthetic Data Generation & Geometry Integration

**3a. `generate_wall_geometry(NPR, AR, Throat_Radius)` function**
- Import `generate_nozzle_profile` from `scripts.visualization.nozzle_2d_geometry`
- Call `generate_nozzle_profile(NPR, AR, Throat_Radius)` to get wall x, y arrays
- Build upper wall coordinates array (already provided by the function)
- Mirror for lower wall: `(x, -y)` for axisymmetric nozzle
- Return upper and lower wall arrays, and geometry dict with A5, A6

**3b. `compute_wall_distances(query_points, wall_points)` function**
- `query_points`: `(N, 2)` tensor of (x, y) coordinates
- `wall_points`: `(W, 2)` tensor of all wall coordinates (upper + lower + centerline if needed)
- Compute minimum Euclidean distance from each query point to any wall point
- Return: `(N, 1)` tensor of minimum distances
- Use efficient broadcasting: `distances = torch.cdist(query_points, wall_points).min(dim=1).values`

**3c. `compute_wall_normals(wall_x, wall_y)` function**
- Given wall x, y arrays in order, compute outward normal vectors
- Use finite differences: `dx = wall_x[1:] - wall_x[:-1]`, `dy = wall_y[1:] - wall_y[:-1]`
- Normal = `(-dy, dx)` normalized, with sign chosen to point outward (away from flow domain)
- Return `(M, 2)` tensor of unit normals at wall midpoints

**3d. `generate_synthetic_training_data(n_samples, NPR, AR, Throat_Radius, P_in, T_in)` function**
- Generate 2D grid of (x, y) points inside the nozzle domain
- Use wall geometry to determine domain bounds at each x
- Compute isentropic flow field analytically:
  - Local area A(x) from nozzle profile
  - Mach number from area-Mach relation with NPR
  - Temperature: `T = T_in * (1 + (γ-1)/2 * M²)^(-1)`
  - Pressure: `P = P_in * (T/T_in)^(γ/(γ-1))`
  - Density: `ρ = P / (R·T)`
  - Axial velocity: `u = M * sqrt(γ·R·T)`
  - Radial velocity: `v = 0` (1st order approximation for axisymmetric)
  - Reynolds stresses: `UU = VV = UV = 0` (laminar baseline, small perturbation for training)
  - Effective viscosity: `μ_eff = μ_lam` (estimate from Sutherland's law)
- Package as input tensor `(N, 6)` and output tensor `(N, 9)`
- Return: `(inputs, outputs, wall_distances)`

### Phase 4: Training Infrastructure

**4a. `setup_training(model)` function**
- Optimizer: `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)`
- Scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-8)`
- Return `(optimizer, scheduler)`

**4b. `train_le_pinn(config)` function**
- Config dict with: `n_epochs`, `NPR`, `AR`, `Throat_Radius`, `P_in`, `T_in`, `batch_size`, `save_path`
- Training loop:
  1. Generate synthetic data (Phase 3d)
  2. Generate wall geometry (Phase 3a)
  3. Compute wall distances (Phase 3b)
  4. Compute wall normals (Phase 3c)
  5. For each epoch:
     a. Forward pass through LE_PINN
     b. Compute data loss: `L_data = MSE(predictions, synthetic_targets)`
     c. Compute physics loss: `L_physics` from Phase 2a
     d. Compute BC loss: `L_bc` from Phase 2b
     e. Compute adaptive weights: `(λ_d, λ_p, λ_bc)` from Phase 2c
     f. Total: `L = λ_d·L_data + λ_p·L_physics + λ_bc·L_bc`
     g. Backward + optimizer step
     h. Scheduler step with total loss
     i. Log every 100 epochs
  6. Save checkpoint with metadata

### Phase 5: Unit Tests (`tests/test_le_pinn.py`)

Create `tests/test_le_pinn.py` with these test functions using pytest:

1. `test_global_network_shape`: Create `GlobalNetwork()`, pass `torch.randn(32, 6)`, assert output shape `(32, 9)`
2. `test_boundary_network_shape`: Create `BoundaryNetwork()`, pass `torch.randn(32, 6)`, assert output shape `(32, 2)`
3. `test_le_pinn_fusion_near_wall`: Create `LE_PINN()`, pass inputs with `wall_distances < 5e-4` for some points. Verify those points have P, T replaced by boundary network outputs.
4. `test_le_pinn_no_fusion_far_wall`: Same but with `wall_distances > 5e-4`. Verify output equals global network output.
5. `test_xavier_init`: Verify all Linear layers have Xavier uniform weights (check weight std is reasonable, not default init).
6. `test_zero_bias_init`: Verify all Linear biases are initialized to zero.
7. `test_min_max_normalizer`: Fit normalizer on known data, verify transform/inverse_transform round-trips correctly.
8. `test_normalizer_epsilon`: Verify normalizer handles constant columns (max == min) without division by zero.
9. `test_optimizer_config`: Call `setup_training()`, verify optimizer is AdamW with lr=1e-4, weight_decay=1e-5.
10. `test_scheduler_config`: Call `setup_training()`, verify scheduler is ReduceLROnPlateau with patience=10, factor=0.5.
11. `test_wall_distance_computation`: Create known wall points and query points, verify distances are correct.
12. `test_rans_residuals_shape`: Verify `compute_rans_residuals` returns 5 tensors of correct shape.
13. `test_existing_nozzle_pinn_import`: Import `NozzlePINN` from `simulation.nozzle.nozzle` and verify it still instantiates correctly — ensures no breakage to existing code.

## File-Level Edits

### `simulation/nozzle/le_pinn.py` [NEW]
- Complete LE-PINN implementation as described in Phases 1-4
- ~500-700 lines of code
- Imports: torch, torch.nn, numpy, typing, sys, pathlib
- Imports `generate_nozzle_profile` from `scripts.visualization.nozzle_2d_geometry`

### `tests/test_le_pinn.py` [NEW]
- 13 unit tests as described in Phase 5
- ~200-300 lines of code
- Uses pytest

## Commands to Run

```bash
# After implementation, run the new tests:
python -m pytest tests/test_le_pinn.py -v

# Verify existing tests still pass:
python -m pytest tests/ -v --ignore=tests/test_le_pinn.py

# Quick smoke test that import works:
python -c "from simulation.nozzle.le_pinn import LE_PINN, GlobalNetwork, BoundaryNetwork; print('Import OK')"
```

## Tests

### Automated Tests
```bash
python -m pytest tests/test_le_pinn.py -v
```

Expected: All 13 tests pass. Key validations:
- Network shapes: GlobalNetwork (N,6)→(N,9), BoundaryNetwork (N,6)→(N,2)
- Fusion: P and T replaced only for d < 5e-4
- Initialization: Xavier uniform weights, zero biases
- Normalization: epsilon-safe min-max
- Optimizer: AdamW lr=1e-4, wd=1e-5
- Scheduler: ReduceLROnPlateau patience=10, factor=0.5

### Existing Tests (Regression)
```bash
python -m pytest tests/test_nozzle_pinn_fix.py tests/test_nozzle_regression.py -v
```

Expected: All existing tests pass unchanged.

## Acceptance Criteria

- [ ] `GlobalNetwork` accepts `(N, 6)` and returns `(N, 9)` with 6 hidden layers × 400 neurons, ReLU
- [ ] `BoundaryNetwork` accepts `(N, 6)` and returns `(N, 2)` with 6 hidden layers × 100 neurons, ReLU
- [ ] `LE_PINN` fuses outputs: replaces P (idx 3) and T (idx 4) when `wall_distance < 5e-4`
- [ ] RANS physics loss computes mass, x-mom, y-mom, energy, EOS residuals via autograd
- [ ] Wall BC loss enforces u=0, v=0, ∂T/∂n=0 (Neumann)
- [ ] Adaptive loss weighting uses sigmoid-based dynamic λ_data, λ_physics, λ_bc
- [ ] AdamW optimizer with lr=1e-4, weight_decay=1e-5
- [ ] ReduceLROnPlateau scheduler with patience=10, factor=0.5, min_lr=1e-8
- [ ] Xavier uniform initialization on all Linear weights, zero biases
- [ ] Min-max normalization with epsilon=1e-8
- [ ] Wall geometry from `generate_nozzle_profile()` in `nozzle_2d_geometry.py`
- [ ] Synthetic data generator produces valid (N,6) inputs and (N,9) outputs
- [ ] All 13 new tests pass: `python -m pytest tests/test_le_pinn.py -v`
- [ ] Existing tests unaffected: `python -m pytest tests/ -v` passes
- [ ] No existing files modified

## Rollback Notes

Pure additive change. Rollback by deleting:
```bash
rm simulation/nozzle/le_pinn.py tests/test_le_pinn.py
```

## Escalation Guidance

**Complexity: 8/10** — Full neural network architecture from paper spec with 2D autograd physics, multi-component adaptive loss, dual-network fusion, and synthetic data pipeline.

**Recommended model: claude-opus** (high complexity, precise mathematical formulation required).
