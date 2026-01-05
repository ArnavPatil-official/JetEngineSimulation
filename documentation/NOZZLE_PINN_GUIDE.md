# Nozzle PINN Guide (v4.0_exact_continuity)

## What This Model Does
- Fuel-dependent converging-nozzle PINN with exact mass conservation (`u = m_dot / (rho * A)`).
- Inlet-anchored normalization for flow variables and fixed-reference normalization for thermo props to preserve fuel sensitivity.
- Physics gates guard inference; automatic analytical fallback retains fuel-dependent gamma if the checkpoint is missing or fails validation.

## Checkpoints & Compatibility
- `VERSION_TAG` = `v4.0_exact_continuity`; `_nozzle_pinn_version_ok()` accepts checkpoints with `version >= 3.1`.
- If the checkpoint is absent or incompatible, `IntegratedTurbofanEngine` falls back to the analytical nozzle and logs the reason.

## Inputs, Outputs, and Normalization
- Input features (8D, normalized): `x*`, `cp*`, `R*`, `gamma*`, `rho_in*`, `u_in*`, `p_in*`, `T_in*`.
- Architecture: 3 hidden layers × 64, Tanh activation; outputs residuals that are anchored to the inlet state.
- Continuity is exact by construction; velocity is derived from mass flow, not predicted.
- Normalization: flow variables anchored to the inlet; thermo properties normalized by `THERMO_REF = {cp: 1150, R: 287, gamma: 1.33}`.

## Runtime API (primary interface)
```python
from simulation.nozzle.nozzle import run_nozzle_pinn

result = run_nozzle_pinn(
    model_path="nozzle_pinn.pt",
    inlet_state={"rho": 0.67, "u": 470.0, "p": 200000.0, "T": 2062.0},
    ambient_p=101325.0,
    A_in=0.375,
    A_exit=0.340,
    length=1.0,
    thermo_props={"cp": 1384.0, "R": 289.8, "gamma": 1.265},
    m_dot=82.6,
    thrust_model="static_test_stand",  # or "incremental_nozzle"
)
print(f"Thrust: {result['thrust_total']/1e3:.2f} kN (fallback={result['used_fallback']})")
```
- Returns exit state, thrust components, inlet verification metrics, mass-conservation check, and optional profiles (`return_profile=True`).

## Physics Gates and Fallback
- Inference is accepted only if:
  - Inlet reproduction error < 5%
  - Mass conservation error < 5%
  - Exit state has positive p/ρ/T and Mach < 2
- Otherwise, the solver switches to the analytical nozzle and reports `fallback_reason`.
- Thrust models:
  - `static_test_stand` (default): `F = m_dot * u_exit + (p_exit - p_amb) * A_exit`
  - `incremental_nozzle`: `F = m_dot * (u_exit - u_in) + (p_exit - p_amb) * A_exit`

## Training & Validation Hooks
- Train/validate: `python simulation/nozzle/nozzle.py` (uses THERMO_REF and inlet-anchored scaling).
- Regression/health checks:
  - `tests/test_nozzle_pinn_fix.py` — positive thrust and scaling robustness.
  - `tests/test_nozzle_regression.py` — PINN vs. analytic consistency.
  - `tests/test_choking_detection.py` — choking logic guardrails.

## Troubleshooting
- `used_fallback=True`: Check checkpoint version and inlet pressures; PINN gates may have rejected the prediction.
- Non-positive thrust: Verify inlet pressure > ambient and that `m_dot`/areas are positive. Analytical fallback uses fuel-dependent gamma but still requires sensible inlet states.
