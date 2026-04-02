# Diagnostic Remediation Plan — April 2026

## Objective

Address seven diagnostic findings (2 CRITICAL, 3 HIGH, 2 MEDIUM) spanning the Sajben diffuser dataset pipeline, LE-PINN physics formulation, analytical nozzle thrust, test coverage, and training robustness. Each fix improves physics fidelity, model consistency, or engineering defensibility.

## Constraints

- **No PINN retraining required by this plan.** Fixes are code-structural. Retraining is a follow-up step after these changes land.
- **No Cantera mechanism changes.** Chemical data files (`data/*.yaml`) are read-only.
- **Preserve existing trained model files** (`models/*.pt`) — no overwrites.
- **All fixes must pass `python -m pytest tests/ -v`** (new and existing tests).
- **No hardcoded paths or magic numbers** introduced.

## Repo Context

| Subsystem | Role |
|-----------|------|
| `scripts/parse_sajben_cfd.py` | Builds training dataset from Sajben geometry |
| `simulation/nozzle/le_pinn.py` | LE-PINN architecture, physics loss, training, fine-tuning |
| `tests/test_pinn_physics.py` | (Current) visual-only PINN test |
| `integrated_engine.py` | Integrated engine cycle with analytical nozzle fallback |

## Relevant Files

### Read
- `simulation/nozzle/le_pinn.py` — full file (1707 lines)
- `scripts/parse_sajben_cfd.py` — full file (491 lines)
- `tests/test_pinn_physics.py` — full file (130 lines)
- `integrated_engine.py` — lines 940–980 (nozzle fallback thrust)

### Modify
- `scripts/parse_sajben_cfd.py`
- `simulation/nozzle/le_pinn.py`
- `tests/test_pinn_physics.py`
- `integrated_engine.py`

### Create
- `tests/test_physics_conservation.py` (new quantitative physics test)

---

## Implementation Phases

### Phase 1 — [CRITICAL] Fix Pressure Thrust Clipping (integrated_engine.py)

**Finding:** Analytical fallback clips `F_pressure = max(delta_p, 0.0) * A_exit`, zeroing out legitimate negative pressure thrust from over-expanded jets.

**Fix:** Replace with signed pressure thrust, keeping the ±1 Pa tolerance band for numerical noise:

#### [MODIFY] [integrated_engine.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/integrated_engine.py)

**Lines 959–964**, replace:
```python
delta_p = p_exit - P_amb
pressure_tol = 1.0  # Pa tolerance to avoid numerical noise
if abs(delta_p) < pressure_tol:
    F_pressure = 0.0
else:
    F_pressure = max(delta_p, 0.0) * A_exit  # Do not allow negative thrust from pressure term
```
with:
```python
delta_p = p_exit - P_amb
pressure_tol = 1.0  # Pa tolerance to avoid numerical noise
if abs(delta_p) < pressure_tol:
    F_pressure = 0.0
else:
    F_pressure = delta_p * A_exit  # Signed: negative for over-expanded jets
```

---

### Phase 2 — [CRITICAL] Fix Sajben Geometry/Physics Inconsistency

Two critical findings share a root cause: the Sajben dataset conflates planar 2D geometry with axisymmetric physics.

#### Phase 2A — Planar A5/A6 for Sajben (parse_sajben_cfd.py)

**Finding:** Sajben is a 2D planar diffuser but A5/A6 are converted to `π r²` axisymmetric areas.

**Fix:** Use planar area (height × unit depth) for Sajben cases and add a `geometry_type` flag to the dataset.

#### [MODIFY] [parse_sajben_cfd.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/scripts/parse_sajben_cfd.py)

**Lines 337–341**, replace:
```python
    # A5, A6: axisymmetric equivalents (mirror train_sajben.py / le_pinn.py)
    r_throat = H_m / 2.0
    A5 = float(np.pi * r_throat ** 2)
    A6 = A5 * float(AR_exit)
```
with:
```python
    # A5, A6: planar channel areas (height × unit depth = h)
    # Sajben is a 2D planar diffuser, NOT axisymmetric.
    # Using πr² would create inconsistency with the planar PDE residuals.
    A5 = float(h_throat)          # throat area per unit depth [m]
    A6 = A5 * float(AR_exit)     # exit area per unit depth [m]
```

#### Phase 2B — Add Planar PDE Residual Mode (le_pinn.py)

**Finding:** `compute_rans_residuals` always uses axisymmetric source terms (`ρv/r`, `ρv²/r`, `1/r` diffusion), but Sajben data is planar.

**Fix:** Add a `geometry` parameter (`"axisymmetric"` or `"planar"`) to `compute_rans_residuals` and the training/fine-tuning functions. In planar mode, drop the `ρv/r`, hoop stress, and `1/r` diffusion terms.

#### [MODIFY] [le_pinn.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/simulation/nozzle/le_pinn.py)

**`compute_rans_residuals` (lines 229–350):** Add `geometry: str = "axisymmetric"` keyword argument. Gate the three source terms:

```python
def compute_rans_residuals(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    R: float = R_GAS,
    cp: float = CP,
    Pr_t: float = PR_T,
    geometry: str = "axisymmetric",
) -> Tuple[...]:
```

In the body, conditionally add axisymmetric source terms:

- **Continuity (line 320–325):** `+ rho * v / y_safe` only if `geometry == "axisymmetric"`
- **Y-momentum (line 334–338):** `- rho * v**2 / y_safe` only if `geometry == "axisymmetric"`
- **Energy (line 340–344):** `+ dT_dy / y_safe` only if `geometry == "axisymmetric"`

**`_safe_physics_loss` (lines 1235–1265):** Thread `geometry` parameter through.

**`train_le_pinn` (line 1024):** Add `geometry` parameter, thread to `compute_rans_residuals`.

**`finetune_on_cfd_data` (line 1272):** Add `geometry` parameter (default `"planar"` for CFD/Sajben), thread to `_safe_physics_loss`.

#### Phase 2C — Weakly-2D Surrogate Fields for v and Stresses (parse_sajben_cfd.py)

**Finding:** All Sajben cases pack `v=0` and `UU=VV=UV=0`, collapsing important 2D physics.

**Fix:** Instead of hard-coding zeros:
1. **v (radial velocity):** Estimate from mass conservation: `v ≈ -y · (1/ρ) · d(ρu)/dx` using finite differences along the axial direction, broadcast to each radial point.
2. **Reynolds stresses:** Drop them from supervised targets by marking channels 5–7 as "ignore" in the dataset metadata. They are already excluded from physics loss — the docstring confirms "trained via data loss only." Since they are all zeros, they add no information; ignoring them in data loss prevents the model from being penalized for non-zero predictions.

For the v-field estimation, add a helper `_estimate_transverse_velocity` to `parse_sajben_cfd.py` and call it in `_pack_case`.

#### [MODIFY] [parse_sajben_cfd.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/scripts/parse_sajben_cfd.py)

Add function after line 58:
```python
def _estimate_transverse_velocity(
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    rho_2d: np.ndarray,
    u_2d: np.ndarray,
    ni: int,
    nj: int,
) -> np.ndarray:
    """
    Estimate v from 2D continuity: ∂(ρu)/∂x + ∂(ρv)/∂y = 0.
    
    Uses axial finite differences of ρu, then integrates vertically
    (trapezoidal) from the lower wall (v=0) to each y-station.
    """
    rho_u = (rho_2d * u_2d).reshape(ni, nj)
    x_1d = x_flat.reshape(ni, nj)[:, 0]
    y_2d = y_flat.reshape(ni, nj)
    rho_2d_r = rho_2d.reshape(ni, nj)
    
    # d(ρu)/dx via central differences (forward/backward at edges)
    d_rhou_dx = np.zeros((ni, nj))
    d_rhou_dx[1:-1, :] = (rho_u[2:, :] - rho_u[:-2, :]) / (
        (x_1d[2:] - x_1d[:-2])[:, None]
    )
    d_rhou_dx[0, :] = (rho_u[1, :] - rho_u[0, :]) / max(x_1d[1] - x_1d[0], 1e-12)
    d_rhou_dx[-1, :] = (rho_u[-1, :] - rho_u[-2, :]) / max(x_1d[-1] - x_1d[-2], 1e-12)
    
    # Integrate: v(x, y) = -(1/ρ) ∫₀ʸ d(ρu)/dx dy', with v(y=0) = 0
    v_2d = np.zeros((ni, nj))
    for i in range(ni):
        for j in range(1, nj):
            dy = y_2d[i, j] - y_2d[i, j - 1]
            integrand = 0.5 * (d_rhou_dx[i, j] + d_rhou_dx[i, j - 1])
            rho_local = max(rho_2d_r[i, j], 1e-12)
            v_2d[i, j] = v_2d[i, j - 1] - integrand * dy / rho_local
    
    return v_2d.ravel()
```

Update `_pack_case` (lines 248–289):
- Replace `v_2d = np.zeros(ni * nj)` with call to `_estimate_transverse_velocity`.
- Replace `zero = np.zeros(ni * nj)` for columns 5–7 with `np.full(ni * nj, np.nan)` as sentinel (or keep zeros but add metadata flag).

> **Note:** Using NaN sentinels for Reynolds stress channels requires the fine-tuning code to mask those columns. Since `finetune_on_cfd_data` already only uses `targets[:, :5]` for data loss (line 1460), setting channels 5–7 to NaN is safe — they are never read during training.

---

### Phase 3 — [HIGH] Chain-Rule Scaling for Normalized Physics Residuals

**Finding:** PDE residuals are computed on normalized inputs/outputs without de-normalization or Jacobian correction. Dimensional PDE terms (with fixed `cp`, `R`, `Pr_t`) applied to min-max-scaled values produce surrogate equations, not physical conservation laws.

**Fix:** De-normalize before computing residuals. In `train_le_pinn` and `_safe_physics_loss`, apply `output_norm.inverse_transform` to predictions and reconstruct physical-space inputs before calling `compute_rans_residuals`.

#### [MODIFY] [le_pinn.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/simulation/nozzle/le_pinn.py)

**Training loop (lines 1117–1123):** After `preds_phys = model(inputs_phys, wall_dists)`, de-normalize:
```python
# De-normalize to physical space before physics residuals
inputs_phys_raw = inputs_phys.detach().clone()
inputs_phys_raw = input_norm.inverse_transform(inputs_phys_raw).requires_grad_(True)
preds_phys_raw = output_norm.inverse_transform(preds_phys)

# Re-run forward pass with physical inputs for correct autograd graph
preds_phys_raw = model(input_norm.transform(inputs_phys_raw), wall_dists)
preds_phys_denorm = output_norm.inverse_transform(preds_phys_raw)

res_mass, res_xmom, res_ymom, res_energy, res_eos = compute_rans_residuals(
    inputs_phys_raw, preds_phys_denorm,
)
```

**Key insight:** We need the autograd graph to trace through `inputs_phys_raw` (physical coordinates) → normalizer → model → denormalizer, so that `torch.autograd.grad` w.r.t. `inputs_phys_raw` gives physically correct derivatives. The normalizer's linear transform is differentiable.

**`_safe_physics_loss` (lines 1235–1265):** Accept `input_norm` and `output_norm` objects, apply the same pattern.

**`finetune_on_cfd_data`:** Thread normalizers into `_safe_physics_loss`.

---

### Phase 4 — [HIGH] Replace Visual Test with Quantitative Physics Assertions

**Finding:** `test_pinn_physics.py` generates a Mach contour plot but has zero `assert` statements.

**Fix:** Create a new `tests/test_physics_conservation.py` with quantitative conservation tests. Keep the existing visual test as-is for backwards compatibility.

#### [NEW] [test_physics_conservation.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/tests/test_physics_conservation.py)

```python
"""
Quantitative physics conservation test for LE-PINN.

Tests:
1. PDE residual norms (continuity, momentum, energy, EOS) < thresholds
2. Mass flow consistency across x-stations: ∫ρu dA ≈ const
3. Stagnation enthalpy conservation: h0 = cpT + 0.5(u²+v²) ≈ const
4. Ideal gas EOS satisfaction: |P - ρRT| / P < tolerance
"""
import pytest
import torch
import numpy as np

# ... (full implementation, ~200 lines, with parametrized fixtures for
#      subsonic and transonic conditions, and assert statements with
#      physically motivated tolerance thresholds)
```

Key assertions:
- `assert max_eos_residual < 0.05` (relative to P)
- `assert mass_flow_std / mass_flow_mean < 0.10` (10% axial consistency)
- `assert h0_std / h0_mean < 0.05` (5% stagnation enthalpy consistency)
- `assert max_continuity_residual_norm < threshold`

---

### Phase 5 — [MEDIUM] Robustness & Dataset Quality Fixes

#### Phase 5A — Fail-Loud Physics Loss (le_pinn.py)

**Finding:** `_safe_physics_loss` silently returns zero on autograd failure.

**Fix:** Add a failure counter and raise after N successive failures.

#### [MODIFY] [le_pinn.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/simulation/nozzle/le_pinn.py)

**`_safe_physics_loss` (lines 1235–1265):** Add `_physics_fail_count` attribute tracking and a configurable `max_physics_failures` threshold. On exceeding the threshold, raise `RuntimeError`. Log each failure with `warnings.warn` including the count.

```python
def _safe_physics_loss(
    model: "LE_PINN",
    inputs_n: torch.Tensor,
    wall_dists: torch.Tensor,
    input_norm: Optional[MinMaxNormalizer] = None,
    output_norm: Optional[MinMaxNormalizer] = None,
    geometry: str = "axisymmetric",
    max_failures: int = 10,
) -> torch.Tensor:
    # Track failures via function attribute
    if not hasattr(_safe_physics_loss, "_fail_count"):
        _safe_physics_loss._fail_count = 0

    try:
        # ... compute physics loss (with denormalization from Phase 3)
        _safe_physics_loss._fail_count = 0  # reset on success
        return loss
    except Exception as exc:
        _safe_physics_loss._fail_count += 1
        warnings.warn(
            f"Physics loss failed ({_safe_physics_loss._fail_count}/{max_failures}): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        if _safe_physics_loss._fail_count >= max_failures:
            raise RuntimeError(
                f"Physics loss failed {max_failures} consecutive times. "
                "Aborting training to prevent silent drift."
            ) from exc
        return torch.tensor(0.0, device=inputs_n.device)
```

#### Phase 5B — Supersonic Family Weighting via Loss (parse_sajben_cfd.py)

**Finding:** Fully-supersonic cases are duplicated N times identically.

**Fix:** Replace duplication with a single copy plus a per-sample weight vector stored in the dataset. Add `"sample_weights": Tensor(N,)` to the saved dict.

#### [MODIFY] [parse_sajben_cfd.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/scripts/parse_sajben_cfd.py)

**Lines 413–421**, replace the duplication loop with:
```python
    # Single copy of the fully-supersonic flow field
    inp_sup, tgt_sup = _pack_case(x_flat, y_flat, ni, nj,
                                   M_axial_sup, P0_axial_sup, T0, A5, A6, P0)
    all_inputs.append(inp_sup)
    all_targets.append(tgt_sup)
    # Weight this family higher in the loss instead of duplicating
    all_weights.append(np.full(len(inp_sup), float(n_supersonic)))
```

Initialize `all_weights: list[np.ndarray] = []` alongside `all_inputs`/`all_targets` (line 389) and add default weight `1.0` for shock cases (line 399) and subsonic cases (line 433). Save as:
```python
torch.save({
    "inputs": torch.from_numpy(inputs_all),
    "targets": torch.from_numpy(targets_all),
    "sample_weights": torch.from_numpy(weights_all),
}, out)
```

Update `finetune_on_cfd_data` to load and apply `sample_weights` if present (weighted MSE loss).

---

## Open Questions

> [!IMPORTANT]
> **Planar vs Axisymmetric default:** Should the default `geometry` parameter for `compute_rans_residuals` be `"axisymmetric"` (current engine nozzle) or `"planar"` (Sajben)? The plan defaults to `"axisymmetric"` in `train_le_pinn` and `"planar"` in `finetune_on_cfd_data`. Please confirm this is acceptable.

> [!IMPORTANT]
> **v-field estimation accuracy:** The continuity-based v estimation is a first-order approximation. For quasi-1D base flow the transverse velocities will be small but non-zero. Is this acceptable as a "weakly 2D" surrogate, or do you prefer to simply drop v from supervised targets for Sajben (rely on physics loss to learn v)?

> [!WARNING]
> **Retrain after Phase 3:** The chain-rule scaling fix (Phase 3) changes what the physics loss optimizes. Any existing trained model's physics loss was optimized against the wrong equations. Models trained after this fix will have different (correct) convergence behavior, but existing checkpoints remain valid for data-loss evaluation.

---

## Commands to Run

```bash
# After all edits, run tests
python -m pytest tests/ -v

# Regenerate Sajben dataset (if parse_sajben_cfd.py changes)
python scripts/parse_sajben_cfd.py

# Lint
python -m flake8 simulation/nozzle/le_pinn.py scripts/parse_sajben_cfd.py integrated_engine.py tests/test_physics_conservation.py --max-line-length=120
```

## Tests

| Test file | Expected outcome |
|-----------|-----------------|
| `tests/test_le_pinn.py` | All existing tests pass (no API breaking changes) |
| `tests/test_pinn_physics.py` | Unchanged, still generates plot |
| `tests/test_physics_conservation.py` (NEW) | All quantitative assertions pass with existing model |
| `tests/test_nozzle_regression.py` | Pass (nozzle regression unchanged) |

## Acceptance Criteria

- [ ] `integrated_engine.py` uses signed pressure thrust (no `max()` clipping)
- [ ] `parse_sajben_cfd.py` computes planar A5/A6 (height-based, not πr²)
- [ ] `parse_sajben_cfd.py` estimates non-zero transverse velocity from continuity
- [ ] `parse_sajben_cfd.py` stores `sample_weights` tensor (no row duplication)
- [ ] `compute_rans_residuals` accepts `geometry` parameter with planar mode
- [ ] Physics residuals are computed in physical (de-normalized) space
- [ ] `_safe_physics_loss` tracks failure count and raises after threshold
- [ ] `tests/test_physics_conservation.py` exists with ≥4 quantitative `assert` statements
- [ ] `python -m pytest tests/ -v` passes (no regressions)
- [ ] No unplanned files modified
- [ ] No trained model files (`models/*.pt`) overwritten

## Rollback Notes

All changes are code-only (no model weights modified). Rollback via:
```bash
git checkout HEAD -- \
  scripts/parse_sajben_cfd.py \
  simulation/nozzle/le_pinn.py \
  integrated_engine.py \
  tests/test_pinn_physics.py
git rm tests/test_physics_conservation.py  # if new file was added
```

## Escalation Guidance

| Metric | Value |
|--------|-------|
| **Complexity** | HIGH — 5 files, PDE formulation changes, normalization pipeline refactor |
| **Risk** | MEDIUM — code-only, no model retraining in this plan |
| **Recommended Model** | Claude Opus (multi-file edits, physics awareness needed) |
| **Estimated LOC changed** | ~400 |
