# LE-PINN reimplementation audit against Ma et al. (P6.5 item 1)

**Date:** 2026-09-26. **Scope:** the repo's LE-PINN nozzle surrogate
(`simulation/nozzle/le_pinn.py`) compared with the *published, numbered
equations and text* of Ma et al. No new training was run (P6.5; review R6-E).
This is a comparison with the paper. It does **not** verify the authors' own
implementation, which was not available and was not inspected.

**Reference.** T. Ma et al., "LE-PINNs …", *Aerospace Science and Technology*
**168 (2026) 111002**. Available online **23 September 2025** (journal header
and article history, p. 1). The issue year is 2026 and the online date is 2025.
Cite the volume as 2026. "ma2025" survives only as a legacy file label.
Pages below are the article's printed page numbers. The supplied PDF was read
at pp. 4–6 (text and rendered pp. 5–6) and p. 8.

## 1. What matches

| Item | Ma et al. (page) | Repo |
|---|---|---|
| Inputs | X = [x, y, A5, A6, P_in, T_in], Eq. 14 (p. 4) | same 6 inputs |
| Outputs | ρ, u, v, P, T, u′u′, v′v′, u′v′, μ_eff (Eqs. 14–15, p. 4) | same 9 outputs |
| Architecture | dual network: global + near-wall boundary network, fused within δ of the wall (Eqs. 34–37, p. 6) | `GlobalNetwork` + `BoundaryNetwork` |
| Activation / init | ReLU, Xavier initialisation (p. 4) | ReLU in attempts 1–2; attempt 3 used tanh (a repo deviation, §3) |
| Optimiser / scheduler | AdamW (weight decay 1e-5); ReduceLROnPlateau, patience 10, factor 0.5, min LR 1e-8 (p. 4) | same. The attempt-1 diagnosis traced a collapse to this scheduler |

## 2. The three differences

### (a) Momentum/energy residual — numbered Eqs. 22–26 (p. 5)

Ma's numbered x-momentum residual (Eq. 23) is

ε_u = ρ(u ∂u/∂x + v ∂u/∂y) + ∂P/∂x − ∂/∂x(μ ∂u/∂x) − ∂/∂y(μ ∂u/∂y) + ∂(ρu′u′)/∂x + ∂(ρu′v′)/∂y,

and Eq. 24 has the same structure for v. So the numbered equations keep:

1. the **Reynolds-stress divergence**, which needs only first derivatives of network outputs; and
2. the viscous term in **product (divergence) form**, ∂/∂x(μ ∂u/∂x) = μ_x u_x + μ u_xx.

The energy residual (Eq. 25) likewise uses ∂/∂x((λ + μ_t/Pr) ∂T/∂x) plus the
dissipation Φ.

Under ReLU a network is piecewise linear, so every second derivative is zero
almost everywhere. With the numbered equations, the μ u_xx part vanishes, but
μ_x u_x and the Reynolds-stress divergence survive. The repo's
`compute_rans_residuals` uses μ_eff·(u_xx + u_yy) and has no Reynolds-stress
terms. Under ReLU **all** of its viscous and turbulent content therefore
vanishes, and the residual enforces the Euler equations. This was measured
directly (`outputs/physics_residual_defect.md`).

**Figure 2 (p. 5) is a sketch, not the numbered equations.** Its box writes
the momentum residual in simplified incompressible-looking form,
u_j ∂u_i/∂x_j + (1/ρ) ∂P/∂x_i − ν ∂²u_i/∂x_j² + ∂(u_i′u_j′)/∂x_j. That is a
Laplacian viscous term, as in the repo, but it *still* has the Reynolds-stress
divergence, which the repo lacks. Statements about "Ma's residual" in this repo
refer to numbered Eqs. 23–25. Which form the authors' code implements is
**unknown**.

### (b) Loss weights — Eqs. 30–33 (p. 5)

Eq. 30 is the weighted total. Eqs. 31–33 set each weight as
λ = 0.1 + 0.9·sigmoid((sum of the other two losses − own loss)/(own loss + ε)).
Every weight is a function of the **current loss values**. The repo's
`AdaptiveLossWeighting.compute_weights(epoch)` is a sigmoid of **training
progress (epoch)**, independent of the losses. The registered P4.3 gate
attempts used a fixed physics weight of 0.05 (attempt 1 with a linear
physics warm-up, `PhysicsWarmupWeighting`). No repo attempt used
current-loss weights.

### (c) What the evaluation tests (pp. 6, 8)

Ma's CFD database covers NPR 4.0–14.5 in steps of 0.5: 22 NPR groups, with
the area ratio tied to NPR (p. 6). Table 2 (p. 8) splits it into 64 training,
14 validation and 2 test cases. The test cases are **held-out CFD conditions
inside that range**: NPR 6.5/AR 1.53 and NPR 12/AR 2.14 (p. 8). The one
comparison with experiment (Fig. 8, p. 8; NPR 4.2, AR 1.79, Ref. [38])
validates the **CFD method** that generated the database, not the surrogate.

The repo's gate instead scores the surrogate on a **different geometry** (the
Sajben transonic diffuser) against **experimental** wall pressure. Ma's
in-range CFD hold-out does not establish accuracy on an external geometry or
against experiment, so the two results are not in contradiction.

## 3. Which negative result each difference bears on

| Repo result | (a) residual form | (b) weights | (c) evaluation |
|---|---|---|---|
| P4.3 attempts 1–2 (ReLU, Laplacian, no Reynolds stress; worse wall 0.258 / 0.245, single seeds, band not claimed) | **Yes**: residual was Euler in effect (second derivatives exactly zero) | Yes: epoch schedule / fixed weight, not current-loss | Yes: external geometry, experiment |
| P4.3 attempt 3 (tanh, μ from data, physics w 0.05; 0.159 ± 0.016, PARTIAL; physics-on worse than matched data-only by 0.044 > seed spread 0.038) | **Yes**: tanh restores u_xx, but there is still no Reynolds-stress divergence and no μ-gradient term, so this is a finding about *this* residual formulation | Yes: fixed weight | Yes |
| Turbine surrogate attempts 1–2 (P4.4, retired) | No: Ma covers nozzles only | No | No |

The repo's negative results are therefore findings about **transfer to an
external experiment** and about **the repo's residual formulation**. They are
not a failed replication of Ma et al.

## 4. Other supplied papers: claims as checked (for P6.8 citations)

- **Wang 2024 (NNICE)**, p. 3, states that a hyperbolic-tangent (tanh)
  activation is used in the hidden layers. The rationale that N–S residuals
  need non-zero second derivatives is **our mathematical interpretation**. It
  is not a quoted justification from the paper.
- **Uy & San Juan 2024 (J. Clean. Prod. 482, 144241)**, p. 2, states "up to
  80 %" SAF carbon reduction, citing Shell (2020). This is **context only**. It
  does not prove that this was the origin of the repo's retired HEFA factor
  0.2 or of the "80 % CO₂ cut" Highlight. CORSIA Doc 06 stays the LCA input.
- **Kuzhagaliyeva et al. 2022 (Commun. Chem. 5, 111)**, p. 3, Table 2,
  compares predictions for **69 mixtures in an independent test set** against
  the **linear-by-mole** mixing rule. This is the naive-baseline discipline
  adopted in P6.1.
- **Nath et al. 2023 (Sci. Rep. 13, 13683)** covers **diesel** engines. A
  PINN estimates unknown parameters and states from data (inverse problem,
  pp. 1–2), with flowcharts of the inverse problem and the physics-loss
  calculation (pp. 3, 7). It is an analogue for identifiability and for the
  R1.12 flowchart, not a turbofan source.
- **Gal et al. 2024 (Energies 17, 5543)** use Cantera 0D/1D reactors
  (products, laminar flame speed, ignition delay; p. 3) for hydrogen
  **reciprocating** engines, with engine simulations validated against
  measurements (p. 1). This supports the "right tool for the quantity"
  argument. It is not an aero-engine source.
- **Şahin 2023 (Heliyon 9, e21365)** reports R², RMSE and MAE (p. 5), with a
  70 %/30 % train/test split (pp. 4, 9). Before citing any score, confirm
  which split it was computed on.

## 5. Future work (not a registered attempt; no training authorised)

A Ma-form residual (Eqs. 23–25 with Reynolds-stress divergence and product-form
diffusion), current-loss weights (Eqs. 31–33), and an evaluation that reports
**both** a Ma-style in-range CFD hold-out **and** the Sajben experiment test,
separately. Any such attempt would need its own pre-registration.
