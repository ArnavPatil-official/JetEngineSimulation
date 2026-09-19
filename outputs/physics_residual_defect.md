# The physics loss has been enforcing inviscid Euler, not RANS — diagnostic record

**Date:** 2026-09-19. **Status:** verified by direct measurement, not inferred.
**Bears on:** P4.3 (nozzle gate outcome), and the interpretation of every
"physics-informed" claim in the project.

## The measurement

`LE_PINN`'s `GlobalNetwork` and `BoundaryNetwork` are ReLU throughout
(`simulation/nozzle/le_pinn.py` lines 286, 291, 314, 319). A ReLU MLP is
piecewise linear, so its second derivative with respect to the inputs is zero
almost everywhere. Measured on a fresh model, 64 random collocation points:

```
d(u)/dy   abs-mean        : 6.1e-03      <- first derivatives are fine
d2(u)/dy2 abs-max         : 0.0
d2(u)/dy2 exactly zero    : 100% of points
```

Every second derivative is **exactly** zero — not small, zero.

## What that does to the residual

`compute_rans_residuals` (lines 574–595) builds the momentum and energy
residuals as

```
res_xmom  = rho*(u*du_dx + v*du_dy) + dP_dx - mu_eff*(d2u_dx2 + d2u_dy2)
res_ymom  = rho*(u*dv_dx + v*dv_dy) + dP_dy - mu_eff*(d2v_dx2 + d2v_dy2)
res_energy= rho*cp*(u*dT_dx + v*dT_dy) - k_eff*(d2T_dx2 + d2T_dy2) ...
```

The viscous and thermal-diffusion terms are the only place the second
derivatives appear. They evaluate to exactly zero and contribute exactly zero
gradient. **The physics loss has been enforcing the inviscid Euler equations**
on a benchmark whose entire interest is shock/boundary-layer interaction.

The code comment on line 574 already says what the residual is —
`# ---- 2. X-momentum (no Reynolds stress terms — Euler/laminar N-S) ----`.
The ReLU finding means it is not even laminar Navier–Stokes; it is Euler.

## The turbulence closure is absent at all three levels

1. **Architecture** — the network emits 9 outputs, columns 5–7 being the
   Reynolds stresses `UU, VV, UV`. Nothing consumes them: the residual has no
   Reynolds-stress terms.
2. **Data** — those same columns are NaN on all 128,061 rows of
   `master_shock_dataset.pt`, correctly masked out by the finiteness filter in
   `finetune_on_cfd_data`.
3. **Residual** — `mu_eff` is not the WIND field's effective viscosity. It is
   overwritten with Sutherland *molecular* viscosity at lines 1558–1560 and
   2549–2551. In a separating transonic diffuser boundary layer mu_t/mu_l is
   O(1e2–1e3), so even with a C2 activation the viscous term would be wrong by
   two to three orders of magnitude exactly where the wall-Cp metric is most
   sensitive.

## The measured cost

From `outputs/sajben_retrain_v5.md`, worse-wall Cp shape-L2:

| Configuration | Upper | Lower | Worse |
|---|---|---|---|
| WIND RANS training data itself (P4.1 ceiling) | 0.089 | 0.084 | **0.089** |
| Data-only, physics weight 0.0 (ablation) | 0.105 | 0.072 | **0.105** |
| P4.3-attempt-1, physics weight 0.05 | 0.258 | 0.187 | **0.258** |
| P4.3-attempt-2, physics weight 0.05, cosine LR | 0.245 | 0.166 | **0.245** |

Velocity profiles show the same ordering at every station. Turning the physics
term on costs a factor of 2.3 on the gate metric, and the training logs record
the mechanism: best validation at epoch ~300 in both runs, then the data loss
*rises* as the physics weight ramps in. A residual describing different physics
from the data will do exactly that.

## Consequence

The nozzle result is not "a PINN nearly reproduced the Sajben benchmark". It is
"a physics loss encoding the wrong equations degraded a data surrogate that was
otherwise within 0.016 of the pass gate". Those are different findings, and only
the second one is supported by the evidence above.

This was diagnosed from the code and confirmed by measurement, independently of
any attempt's score. It is a formulation defect, in the same class as the
collapsed-initialisation defect P4.2 found — not a hyperparameter.
