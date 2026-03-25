# Physics Fixes Plan: Turbine, Combustor, and LE-PINN

## Objective

Fix five physics/engineering issues across the simulation codebase to make results scientifically defensible:

1. **Issue B** — Turbine expansion uses a hardcoded `expansion_ratio = 1005.0 / 1700.0` instead of a fuel-specific isentropic relation with polytropic efficiency.
2. **Issue D** — Combustor efficiency is a static `0.98` constant; it should vary with equivalence ratio (φ) and fuel blend.
3. **Issue A** — LE-PINN RANS residuals use 2D Cartesian PDEs instead of axisymmetric (adds ρv/r source terms).
4. **Issue C** — Reynolds stress outputs (UU, VV, UV) appear in the physics loss without a turbulence closure; they should be dropped from the physics loss and trained only via data loss.
5. **Issue E** — Wall BC assumes adiabatic (∂T/∂n = 0); this must be explicitly documented, or replaced with a convective BC.

## Constraints

- Existing trained model weights (`models/*.pt`) are NOT overwritten; a retrain will be needed separately (Phase 3 from user plan, out of scope for this code change).
- Chemical data files (`data/*.yaml`) remain read-only.
- All changes must preserve random seeds and reproducibility.
- Type hints and docstrings must be maintained.
- Existing tests must continue to pass (with test updates where physics equations change).

## Repo Context

| Subsystem | Involvement |
|-----------|-------------|
| Turbine (integrated_engine.py) | Issue B: Fix `run_turbine` work target calculation |
| Combustor (combustor.py) | Issue D: Add dynamic efficiency heuristic |
| LE-PINN (le_pinn.py) | Issues A, C, E: PDE residuals, turbulence closure, wall BC |
| Tests | Update `test_le_pinn.py` for changed RANS residuals |

## Relevant Files

| Action | Path | Purpose |
|--------|------|---------|
| MODIFY | `integrated_engine.py` | Fix turbine expansion (Issue B), wire dynamic combustor efficiency (Issue D) |
| MODIFY | `simulation/combustor/combustor.py` | Add `estimate_efficiency()` heuristic function (Issue D) |
| MODIFY | `simulation/nozzle/le_pinn.py` | Axisymmetric PDEs (Issue A), drop Reynolds stresses from physics loss (Issue C), document/update wall BC (Issue E) |
| MODIFY | `tests/test_le_pinn.py` | Update RANS residuals test for new equation structure |

---

## Implementation Phases

### Phase 1: Turbine Expansion Fix (Issue B)

**File: `integrated_engine.py`**

**1a. Replace `scale_turbine_exit_temp` function (L444-467)**

Replace the hardcoded expansion ratio approach with the isentropic expansion formula:

```python
T_out = T_in * [1 - η_t * (1 - (P_out / P_in)^((γ-1)/γ))]
```

Where η_t ≈ 0.9 (polytropic turbine efficiency). The function now requires `gamma` and `P_out/P_in` ratio.

**1b. Fix `run_turbine` method (L843-849)**

Currently at L846-849:
```python
# Estimate work from expansion ratio
cp = thermo_props['cp']
T_in = inlet_state['T']
delta_T_estimated = T_in * (1 - self.turbine_design['expansion_ratio'])
target_work_total = m_dot * cp * delta_T_estimated
```

Replace with dynamic isentropic calculation using gamma from `thermo_props`:
```python
cp = thermo_props['cp']
gamma = thermo_props['gamma']
T_in = inlet_state['T']
p_in = inlet_state['p']
p_out_ratio = 1.93e5 / p_in  # turbine exit pressure ratio
eta_t = 0.9  # polytropic turbine efficiency
T_out = T_in * (1.0 - eta_t * (1.0 - p_out_ratio ** ((gamma - 1.0) / gamma)))
delta_T = T_in - T_out
target_work_total = m_dot * cp * delta_T
```

**1c. Update `turbine_design` dict (L569-573)**

Remove the hardcoded `expansion_ratio` and add documentation:
```python
self.turbine_design = {
    'T_in_ref': 1700.0,
    'T_out_ref': 1005.0,
    'P_out': 1.93e5,     # Turbine exit pressure [Pa]
    'eta_polytropic': 0.9  # Polytropic turbine efficiency
}
```

---

### Phase 2: Dynamic Combustor Efficiency (Issue D)

**File: `simulation/combustor/combustor.py`**

**2a. Add `estimate_efficiency()` static method to `Combustor` class**

Add a heuristic function that penalizes efficiency as φ deviates from stoichiometric and applies a small penalty for heavier SAF blends:

```python
@staticmethod
def estimate_efficiency(phi: float, fuel_blend=None) -> float:
    """
    Estimate combustor efficiency based on equivalence ratio and fuel type.
    
    Efficiency drops as phi deviates from stoichiometric (1.0):
      η = η_max - k_phi * (phi - 1.0)^2
    
    Additional penalty of 1-2% for heavier SAF blends (literature-based).
    """
    eta_max = 0.995
    k_phi = 0.04  # penalty coefficient
    eta = eta_max - k_phi * (phi - 1.0) ** 2
    
    # SAF blend penalty (heavier molecules = slightly lower efficiency)
    if fuel_blend is not None:
        blend_name = getattr(fuel_blend, 'name', '')
        if 'Bio-SPK' in blend_name:
            eta -= 0.015  # 1.5% penalty
        elif 'HEFA' in blend_name:
            eta -= 0.01   # 1.0% penalty
    
    return max(min(eta, 0.999), 0.90)  # clamp to [0.90, 0.999]
```

**File: `integrated_engine.py`**

**2b. Wire dynamic efficiency into `run_full_cycle` (L1072-1116)**

Replace the static `combustor_efficiency=0.98` parameter default with dynamic calculation. In `run_full_cycle`, before calling `run_combustor`, compute the dynamic efficiency:

```python
# If no explicit efficiency provided, use dynamic heuristic
if combustor_efficiency is None:
    combustor_efficiency = Combustor.estimate_efficiency(phi, fuel_blend)
```

Change the default parameter from `combustor_efficiency: float = 0.98` to `combustor_efficiency: Optional[float] = None`.

---

### Phase 3: Axisymmetric PDEs (Issue A)

**File: `simulation/nozzle/le_pinn.py`**

**3a. Update `compute_rans_residuals` function (L215-330)**

Add the axisymmetric source terms. The `y` coordinate represents the radial distance `r`.

- **Continuity** — add `ρv/y` term:
  ```python
  eps = 1e-8  # avoid division by zero at centerline
  y = inputs[:, 1:2]
  res_mass = (rho * du_dx + u * drho_dx) + (rho * dv_dy + v * drho_dy) + rho * v / (y + eps)
  ```

- **Y-momentum** — add hoop stress term `−P/r` (centrifugal pressure balance):
  ```python
  res_ymom = (
      rho * (u * dv_dx + v * dv_dy) + dP_dy
      - mu_eff * (d2v_dx2 + d2v_dy2)
      + rho * (dUV_dx + dVV_dy)
      - rho * v**2 / (y + eps)   # hoop stress
  )
  ```

- **Energy** — add `ρv/r` geometric source:
  ```python
  # No additional energy source for axisymmetric, but the conduction term gets a 1/r d/dr(r dT/dr) form
  # For the physics loss approximation, we add the 1/r term to the diffusion:
  k_eff = mu_eff * cp / Pr_t
  res_energy = rho * cp * (u * dT_dx + v * dT_dy) - k_eff * (d2T_dx2 + d2T_dy2 + dT_dy / (y + eps))
  ```

---

### Phase 4: Turbulence Closure Fix (Issue C)

**File: `simulation/nozzle/le_pinn.py`**

**4a. Drop Reynolds stress terms from physics residuals**

In `compute_rans_residuals`, remove the Reynolds stress gradient terms from x-momentum and y-momentum. The network still outputs UU, VV, UV (indices 5-7) so they can be trained via data loss, but the physics loss treats the flow as Euler/laminar Navier-Stokes:

- **X-momentum** (remove `+ rho * (dUU_dx + dUV_dy)`):
  ```python
  res_xmom = (
      rho * (u * du_dx + v * du_dy) + dP_dx
      - mu_eff * (d2u_dx2 + d2u_dy2)
  )
  ```

- **Y-momentum** (remove `+ rho * (dUV_dx + dVV_dy)`):
  ```python
  res_ymom = (
      rho * (u * dv_dx + v * dv_dy) + dP_dy
      - mu_eff * (d2v_dx2 + d2v_dy2)
      - rho * v**2 / (y + eps)  # hoop stress (from Issue A)
  )
  ```

- **Remove corresponding autograd computations** for `dUU_dx`, `dUV_dx`, `dUV_dy`, `dVV_dy` (L277-284) to avoid unnecessary computation.

---

### Phase 5: Heat Transfer BC Documentation (Issue E)

**File: `simulation/nozzle/le_pinn.py`**

**5a. Document the adiabatic assumption explicitly**

Update the docstring for `compute_wall_bc_loss` (L337-380) to explicitly state:

```
NOTE: The nozzle wall is modeled as ADIABATIC (∂T/∂n = 0).
This is acceptable for a proof-of-concept where heat transfer through
the nozzle wall is negligible compared to the convective enthalpy flux.
For production fidelity, replace with a convective BC:
    k·∂T/∂n = h·(T_wall − T_ambient)
```

Also update the module-level docstring (L1-32) to mention the adiabatic wall assumption.

---

## File-Level Edits

### [MODIFY] [integrated_engine.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/integrated_engine.py)

- L444-467: Rewrite `scale_turbine_exit_temp()` to use isentropic relation with polytropic efficiency
- L528-573: Update `turbine_design` dict to remove `expansion_ratio`, add `P_out` and `eta_polytropic`
- L843-849: Replace hardcoded expansion ratio delta-T calculation with dynamic isentropic formula
- L1072-1077: Change `combustor_efficiency` default to `None`, add dynamic calculation before combustor call

### [MODIFY] [combustor.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/simulation/combustor/combustor.py)

- Add `estimate_efficiency()` static method to `Combustor` class (~25 lines)

### [MODIFY] [le_pinn.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/simulation/nozzle/le_pinn.py)

- L1-32: Update module docstring to mention adiabatic wall assumption
- L215-330: Rewrite `compute_rans_residuals()`: add axisymmetric terms, remove Reynolds stress terms
- L337-380: Update `compute_wall_bc_loss()` docstring with adiabatic documentation

### [MODIFY] [test_le_pinn.py](file:///Users/arnavpatil/Desktop/JetEngineSimulation/tests/test_le_pinn.py)

- Update `TestRANSResiduals.test_output_shapes` — the function signature and return shapes are the same (5 residuals), but verify the test still passes with the new axisymmetric equations.

## Commands to Run

```bash
# Run all tests after implementation
python -m pytest tests/ -v

# Specifically run the LE-PINN tests
python -m pytest tests/test_le_pinn.py -v

# Verify imports still work
python -c "from simulation.nozzle.le_pinn import LE_PINN, compute_rans_residuals; print('LE-PINN import OK')"
python -c "from simulation.combustor.combustor import Combustor; print(Combustor.estimate_efficiency(0.5)); print('Combustor import OK')"
python -c "from integrated_engine import IntegratedTurbofanEngine; print('Engine import OK')"
```

## Tests

### Automated Tests -- Existing
```bash
# All existing tests must still pass
python -m pytest tests/ -v
```

Specific expectations:
- `tests/test_le_pinn.py::TestRANSResiduals::test_output_shapes` — must still pass (5 residuals, same shapes)
- `tests/test_le_pinn.py::TestLE_PINN_Fusion` — unchanged, must pass
- `tests/test_le_pinn.py::TestCFDFinetune` — unchanged, must pass (if dataset present)
- `tests/test_nozzle_pinn_fix.py` — unchanged, must pass
- `tests/test_nozzle_regression.py` — unchanged, must pass

### Manual Verification
After code changes, the user should retrain the LE-PINN and run the blends mode:
```bash
# Retrain LE-PINN (Phase 3 from user's plan — separate step)
# python simulation/nozzle/le_pinn.py

# Run blends comparison
# python integrated_engine.py --mode blends
```

## Acceptance Criteria

- [x] `scale_turbine_exit_temp` uses isentropic expansion with η_t = 0.9 and fuel-specific gamma
- [x] `run_turbine` default work target uses dynamic `delta_T` from isentropic formula, not hardcoded ratio
- [x] `Combustor.estimate_efficiency(phi, fuel_blend)` returns dynamic efficiency based on φ and fuel type
- [x] `run_full_cycle` uses dynamic combustor efficiency by default
- [x] `compute_rans_residuals` uses axisymmetric continuity (`+ ρv/(y+ε)`) and momentum (hoop stress)
- [x] Reynolds stress terms (UU, VV, UV gradients) removed from physics loss residuals
- [x] Adiabatic wall BC explicitly documented in `compute_wall_bc_loss` and module docstring
- [x] All existing tests pass: `python -m pytest tests/ -v`
- [x] No model weights (`models/*.pt`) overwritten
- [x] No chemical data files (`data/*.yaml`) modified

## Rollback Notes

Each change is isolated to its own function/method:
```bash
# Git revert is the cleanest rollback
git diff HEAD~1 --stat   # review changes
git revert HEAD          # undo if needed
```

## Escalation Guidance

**Complexity: 6/10** — Five localized fixes across three files. The math is precise but each change is self-contained. The LE-PINN PDE changes (axisymmetric terms) require the most care.

**Recommended model: claude-sonnet** — Sufficient for localized, well-specified edits.
