# Phase 8 Track 4 — PINN repair notes (2026-10-02)

Registration `docs/phase8_track4_registration.json` (P8-TRACK4-20261002-attempt1),
plan `docs/plan.md` (Track 4 addendum and pre-run review corrections). Code
`scripts/phase8/pinn_diagnostics/`, tests `tests/test_phase8_pinn_diagnostics.py`,
outputs (when run) `outputs/phase8/track4/20261002_attempt1/`.

**Scope.** These are diagnostics with known exact answers. They do not
validate the engine, the turbine, or any PINN. No empirical data, held-out
rows or Sajben/WIND data were opened. G2/G3 are not reopened. The retired
production turbine PINN stays retired.

**Source of the E-numbering.** E1–E7 below follow the user's CAT-JET PINN
Repair Guide ("Equations to work through by hand"). The guide was supplied as
a planning attachment. It is read-only and outside the repository, and this
pass read it. The sections below are worked explanations of the guide's
exercises, checked against repository sources where possible. They do not
claim that the user has completed any handwritten exercise. The user's
checklist items (E1–E7 on paper, the readings, the paragraph in their own
words) remain theirs.

## Results

**Not run.** Every numerical diagnostic is BLOCKED (2026-10-02): the Mac is
on battery, and the registration requires mains power. The turbine map has
not been trained or scored. The manufactured-solution checks and the nozzle
ladder have not been evaluated. The focused and full pytest runs are pending.
No score exists, and none is quoted here or in any freeze note. The exact
pending commands are listed in `outputs/phase8_execution_status.md`
(Track 4 section).

## Source corrections

1. **Attempt-2 turbine failure (6.390 %) is not caused by a global 4.2 MPa
   pressure scale.** In `scripts/validation/train_turbine_surrogate.py`,
   P4.4 attempt 2 trains in inlet-anchored variables: every condition's state
   is divided by its own inlet state (`inlet_norm = ones`). It has analytic
   path supervision: `losses()` fits ρ, p and T against the analytic
   expansion at 33 collocation points per condition. Its path error is
   per-point relative (`path_error: "relative"`). The 4.20 MPa value is
   `DEFAULT_SCALES['p']` in `simulation/turbine/turbine.py`. The *legacy*
   losses there (`compute_loss_components`, `surrogate_path_loss`) divide by
   it. Attempt 2 does not. The root cause of the 6.390 % is **not
   established**. The measured facts are a held-out max |Δp5|/p5 of 6.390 %
   with a raw-T5 max error of 0.262 % (`outputs/turbine_surrogate_v5_a2.md`),
   in float32 with Adam only. The guide's Track 4 text attributes the failure
   to the 4.2 MPa scale and to the missing pressure equation. For attempt 2
   that attribution is inaccurate. Both mechanisms are real for the legacy
   loss (E2, E3).
2. **"Missing pressure equations" applies to the legacy physics-only loss,
   not to attempt 2.** `compute_loss_components` (legacy) has only the
   equation of state p = ρRT, a monotonic-T penalty and the shaft-work
   endpoint. No momentum or isentropic relation fixes p(x) (E2). Attempt 2
   added the path term, which supervises p directly.
3. **Ma loss weights are asymmetric** (Eqs. 31–33, printed p. 5). Only
   λ_data compares the sum of the other two losses with its own loss.
   λ_physics and λ_BC each compare with L_data alone (E7). The earlier
   "sum of the other two" shorthand in `docs/le_pinn_vs_ma2025.md`, repeated
   in the guide's E7 prompt, is corrected there.
4. **Ma Eq. 25's thermal coefficient is dimensionally ambiguous** (λ + μ_t/Pr
   adds W m⁻¹ K⁻¹ to Pa s). This is flagged, not repaired. The verifier uses
   it literally, as a dimensionless algebra check (see "Ma Eq. 25" below).
5. *Observation, not acted on:* Eq. 25 is a temperature-form energy
   residual without the pressure-work term u·∇p. The compressible
   temperature equation ρc_p DT/Dt = Dp/Dt + ∇·(λ∇T) + Φ contains it. Ma's
   CFD equation (Eq. 4, total-enthalpy form) includes it implicitly. The
   verifier implements Eq. 25 as printed.
6. **Pass-gate algebra.** The turbine gate is max |expm1(Δ)| < t with
   Δ = pred_log − exact_log and t = 10⁻³. That is equivalent to
   log(1 − t) < Δ < log(1 + t), i.e. −1.0005003 × 10⁻³ < Δ < 9.995003 × 10⁻⁴.
   It is **not** |Δ| < log(1 + t). That symmetric form is sufficient but
   stricter on the negative side: it would reject an under-prediction
   Δ = −0.99990 × 10⁻³, whose true relative pressure error is 0.9994 × 10⁻³ < t.
   An earlier draft of these notes stated the symmetric form. The code
   scores expm1(Δ) directly, so it implements the exact two-sided interval.

## Worked mathematics (guide E1–E7)

### E1 — Polytropic turbine expansion and its sensitivity

Polytropic expansion: dh = η_p dp/ρ, with h = c_p T and p = ρRT, so
c_p dT = η_p RT dp/p and dp/p = (c_p/(η_p R)) dT/T = γ/(η_p(γ − 1)) · dT/T.
Integrating from 4 to 5 with T5/T4 = 1 − τ, τ = W/(ṁ c_p T4):

  p5/p4 = (1 − τ)^k,  k = γ/(η_p(γ − 1)),  log(p5/p4) = k · log1p(−τ).

Guide check: γ = 1.30, η_p = 0.90 gives k = 4.815. With τ = 0.30,
p5/p4 = 0.7^4.815 = 0.180. Sensitivity: d ln p5 / d ln T5 = k, so a 1 % error
in T5 becomes about 4.8 % in p5. Pressure is hypersensitive. That is why the
diagnostic predicts ln(p5/p4).

Over the registered box (τ 0.2593–0.3985, γ 1.2795–1.3152, η_p 0.9), k runs
4.64–5.09 and p5/p4 runs 0.075–0.249. In terms of τ,
∂ log(p5/p4)/∂τ = −k/(1 − τ). A 0.1 % pressure error therefore equals a τ error
of only (1 − τ) · 10⁻³/k ≈ 1.2–1.6 × 10⁻⁴.

Error metric: p_pred/p_exact − 1 = e^Δ − 1 = expm1(Δ). This is exact, and p4
cancels, so the score does not depend on the pressure scale (tested with
p4 = 1, 101325 and 4.2 × 10⁶). `expm1` avoids cancellation when Δ is tiny.
The gate interval is two-sided and asymmetric (source correction 6).

### E2 — Why the legacy turbine loss could not pin down pressure

The legacy physics loss (`compute_loss_components`) has three terms:
p − ρRT (gas law), relu(dT/dx) (temperature falls) and
ṁ c_p (T[0] − T[−1]) against the target work (endpoint work; endpoints are
read by array position). Velocity is u = ṁ/(ρA) by construction and enters
no loss. Take any smooth g(x) > 0 with g(0) = 1 and map
(ρ, p, T) → (ρg, pg, T). Then pg − (ρg)RT = g(p − ρRT), which is zero whenever
the original is. The temperature terms do not change, and the inlet state does
not change. So every loss value is unchanged, and p(x) is undetermined
up to an arbitrary positive profile. The missing equation is the polytropic
path as a residual:

  r(x) = d ln p/dx − k · d ln T/dx = 0.

With r, p is fixed by T and the inlet value. Without it, the network is free
to put any pressure path through the gas law. Lesson: count unknowns against
independent equations before training. P4.4 attempt 2 did not have this
defect, because it supervised p against the analytic path (source
correction 2).

### E3 — The normalisation trap (legacy loss, not attempt 2)

The legacy losses divide pressure errors by one global scale,
`DEFAULT_SCALES['p']` = 4.20 MPa (take-off p4). Guide example: a p5 of
0.197 MPa (the Trent 1000-E approach worst case, analytic 1.973 bar) with a
6.39 % error is 12.6 kPa. Its normalised squared error is
(12.6 × 10³/4.2 × 10⁶)² = (3.0 × 10⁻³)² ≈ 9.0 × 10⁻⁶. A 1 % error at 4.2 MPa
gives (10⁻²)² = 10⁻⁴, which is 11 times larger. Equivalently, the same
*relative* error at p5 = 0.197 MPa is down-weighted by 4.2/0.197 ≈ 21 times in
the residual and by about 450 times in the squared loss. That is the guide's
"~20x". Lesson: use ln p, or divide each error by its own true value.

This example borrows attempt 2's measured worst case only as a number. It
illustrates the legacy loss. It does not explain attempt 2, which used
inlet-anchored, per-point relative path errors (source correction 1).

### E4 — ReLU and second derivatives

For f(x) = Σ aᵢ max(0, wᵢx + bᵢ), each term is linear on either side of its
kink at x = −bᵢ/wᵢ. So f′ is piecewise constant and f″ = 0 everywhere except at
the kinks, where it is undefined (a Dirac mass in the distributional sense,
which autograd does not see). Expanding the divergence form:

  ∂/∂x(μ ∂u/∂x) = μ_x u_x + μ u_xx.

Under ReLU only μ_x u_x survives. The repository's Laplacian form
μ(u_xx + u_yy) vanishes entirely (`outputs/physics_residual_defect.md`).
Lesson: match the activation's smoothness to the highest derivative in the
residual, and use the divergence form the paper writes. The verifier uses
tanh and SiLU, with closed-form derivatives

- tanh: h′ = 1 − h², h″ = −2h(1 − h²);
- SiLU: with s = σ(z), h = z s, h′ = s + z s(1 − s),
  h″ = 2s(1 − s) + z s(1 − s)(1 − 2s);

and the chain rule for Ma Eq. 16 min–max scaling, x_n = (x − x_min)/Δx:
∂q/∂x = (1/Δx) ∂q/∂x_n and ∂²q/∂x² = (1/Δx²) ∂²q/∂x_n² (tested).

### E5 — What mean-squared error learns from one-to-many data

d/dc Σ(yᵢ − c)² = −2 Σ(yᵢ − c) = 0 gives c = (1/n) Σ yᵢ, and the second
derivative 2n > 0 makes it the minimum. For inputs that repeat with different
targets, the best any regression can do is predict the mean. In the old
Sajben set, A5, A6, P0 and T0 are identical for all 31 cases, so the same
(x, y) carries 31 different pressures (`outputs/sajben_data_audit.md`,
Finding 1b). A network fits the average over shock positions: a smeared ramp
instead of a shock. Before any training, count input rows that repeat with
different targets. The nozzle ladder's `validate_cases` applies the same
rule to complete case inputs. Exact duplicates collapse. The same case
with a different input is rejected. Back pressure is part of every complete
shock-case input.

### E6 — Quasi-1D nozzle, choking and the normal shock

- Critical pressure ratio: p*/p0 = (2/(γ + 1))^{γ/(γ−1)}. For γ = 1.33 this
  is 0.540. The repository's real-gas value at AE3 take-off is 0.537, so the
  core chokes once p0/pa exceeds 1/0.537 ≈ 1.86. It runs at 1.87, just over
  (P8.3 G1 record, `outputs/phase8_execution_status.md`).
- Area–Mach relation:
  A/A* = (1/M)[(2/(γ+1))(1 + (γ−1)M²/2)]^{(γ+1)/(2(γ−1))}, with one subsonic
  and one supersonic root for every A/A* > 1. With inlet M = 0.2 at A = 1.5
  and γ = 1.33, A* = 0.5026 < A_throat = 1, so the registered smooth case is
  subsonic throughout. When choked, A* = A_throat and
  ṁ* = A* p0 √(γ/(R T0)) (2/(γ+1))^{(γ+1)/(2(γ−1))}.
- Normal shock: M2² = (1 + (γ−1)M1²/2)/(γM1² − (γ−1)/2),
  p2/p1 = 1 + 2γ(M1² − 1)/(γ+1), ρ2/ρ1 = (γ+1)M1²/((γ−1)M1² + 2). For
  γ = 1.4 and M1 = 2: M2 = 1/√3, p2/p1 = 9/2, ρ2/ρ1 = 8/3, T2/T1 = 27/16,
  p02/p01 = (8/3)^{3.5}(2/9)^{2.5} = 0.72087. At M1 = 1 there is no jump. The
  code returns the exact identity there, so a shock at the throat loses no
  total pressure.
- Back pressure alone sets the shock position in the diverging section.
  Constant ṁ and T0 give A*₂ = A*₁ p01/p02. Behind the shock the flow is
  subsonic to the exit, where p_e = p_b. As the shock moves downstream it
  strengthens, so p02 falls and p_e falls. Higher back pressure therefore
  moves the shock upstream. An internal normal shock exists only for
  p_e(shock at exit) ≤ p_b ≤ p_e(shock at throat). Both endpoints are valid
  inputs. Anything outside is rejected. Jump convention: grid points with
  x < x_s are upstream and a point at x = x_s is downstream, so a shock at the
  exit shows the post-shock exit pressure p_b. The downstream state is
  computed from the area inversion with A*₂, independently of the
  Rankine–Hugoniot M2, and the scores compare the two, together with the
  mass, momentum, total-enthalpy and total-pressure-loss jumps.
- Lesson: back pressure must be a network input. The shock is a
  discontinuity. The references are evaluated algebraically on each side, and
  nothing is differentiated across it. These are reference solutions, not
  trained-PINN results.

### E7 — Ma loss-balancing weights versus gradient-norm balancing

The guide's E7 prompt paraphrases Eqs. 31–33 as
λᵢ = 0.1 + 0.9 σ((sum of the other losses − own loss)/(own loss + ε)).
Checked against the printed p. 5, only λ_data has that form:

  λ_data = 0.1 + 0.9 σ((L_p + L_b − L_d)/(L_d + ε)),
  λ_phys = 0.1 + 0.9 σ((L_d − L_p)/(L_p + ε)),
  λ_BC  = 0.1 + 0.9 σ((L_d − L_b)/(L_b + ε)),

with ε = 10⁻¹² here and the weights detached from the gradient. For the
guide's losses (data, physics, BC) = (10⁻², 10⁻⁴, 10⁻³): the data argument
is (10⁻⁴ + 10⁻³ − 10⁻²)/10⁻² = −0.89, giving λ_data = 0.362. The physics
argument is 99, giving λ_phys = 1.000. The BC argument is 9, giving
λ_BC = 0.99989. These match the guide's "about 0.36, 1.00, 1.00" (tested with
plain-float inputs in CPU float64). A second example, L = (1, 2, 3), gives
λ_data = 0.9838, λ_phys = 0.4398 and λ_BC = 0.4053, where the symmetric
shorthand would give λ_phys = 0.7580.

Which term it favours: a residual or BC loss that is *already small relative
to the data loss* gets weight near 1. The data term is damped when it
dominates. So the scheme pushes the optimiser to keep satisfying physics and
boundary conditions once they are nearly met, rather than chasing the data
misfit. That suits the authors' setting, where the data are sparse CFD
samples and the physics should regularise between them. Because losses are
non-negative, every argument is ≥ −1 (as ε → 0), so every weight lies in
[0.1 + 0.9 σ(−1), 1) = [0.342, 1), never down to the nominal 0.1.

Wang, Teng and Perdikaris instead balance gradient statistics:
λ̂ᵢ = max_θ |∇_θ L_residual| / mean_θ |∇_θ Lᵢ|, smoothed by a moving average.
That rule uses the gradients' sizes, not the loss values. It responds to
stiffness (a residual whose gradients dwarf the data gradients) even when
the loss values are comparable, and it is scale-free in a different sense.
Ma's rule is cheap (no extra backward passes) but blind to gradient
imbalance. Choosing between them must use training cases only, by
cross-validation. That comparison is deferred, and no held-out score may
choose it.

### Additional note — four inputs, two dimensions

The Track 4a features are [τ, γ, η_p, c_p/R]. τ and γ are mapped to [−1, 1] by
the envelope bounds. η_p is mapped as η_p/0.9 − 1, which is identically 0
because η_p = 0.9 everywhere. c_p/R = γ/(γ − 1) decreases with γ, so its
bounds are [4.1727, 4.5776] (from γ_max, γ_min), and it is a fixed function of
the γ feature. The inputs therefore lie on a two-dimensional surface in a
four-dimensional space. **This envelope has only two varying independent
dimensions.** A passing score would show that an MLP can represent this
two-variable map. It would say nothing about representing variable η_p or
independent c_p/R.

### Ma Eq. 25 — dissipation and the thermal coefficient

From Eqs. 7–10, with τ_xx = 2μu_x − (2/3)μ(u_x + v_y),
τ_yy = 2μv_y − (2/3)μ(u_x + v_y) and τ_xy = μ(u_y + v_x):

  Φ = τ_xx u_x + τ_xy(u_y + v_x) + τ_yy v_y
    = 2μ(u_x² + v_y²) − (2/3)μ(u_x + v_y)² + μ(u_y + v_x)²
    = μ[(4/3)(u_x² + v_y² − u_x v_y) + (u_y + v_x)²].

Eq. 25's coefficient λ + μ_t/Pr is not dimensionally homogeneous. The usual
turbulent-heat-flux closure gives λ + c_p μ_t/Pr_t. Choosing between them
needs the authors' code or a statement from them. The verifier therefore
labels itself a dimensionless algebra check and keeps the printed form. Its
manufactured-solution method expands each printed term by the product rule
into closed-form forcing (NumPy, no autograd). It requires the autograd
residual to equal that forcing to rounding. Four omission controls (viscosity
gradient, Reynolds-stress divergence, conductivity gradient, dissipation)
must each leave a discrepancy above 10⁻⁷. Agreement shows that the residual
code implements the printed equations. It does not show that they are
physically right.

## Proposed explanatory paragraph (for the write-up; not yet used anywhere)

*Draft for the user to rewrite in their own words. It makes no claim about
the Track 4a result, which has not been run.*

> The retired turbine PINN treated a zero-dimensional component as a field
> along an invented coordinate. Its original physics loss could not determine
> pressure: the gas law, a falling-temperature penalty and an endpoint work
> balance are all unchanged when density and pressure are multiplied by any
> positive profile g(x) with g(0) = 1, because no equation tied the pressure
> path to the temperature path. That loss also divided pressure errors by a
> single 4.2 MPa scale, so low-pressure operating points were down-weighted
> about twentyfold. The registered second attempt removed both defects by
> supervising the analytic path with per-point relative errors in
> inlet-anchored units, yet it still missed the gate (6.39 % worst-case exit
> pressure error) for reasons that were not isolated. The pressure ratio is
> five times more sensitive than temperature (d ln p5/d ln T5 = k ≈ 5), so
> small representation errors are amplified. The repair is to model the
> turbine as what it is, a map from (τ, γ, η_p, c_p/R) to ln(p5/p4), or
> better, to compute p5 from the polytropic relation and learn only a
> bounded correction from rig data. Without measured turbine data, the only
> honest claim for any network copy is speed and differentiability, not
> accuracy.

## Verified reading list

Links as supplied in the plan (checked by the planner):

- Ma et al., LE-PINNs, *Aerospace Science and Technology* 168 (2026) 111002 —
  https://doi.org/10.1016/j.ast.2025.111002
- Krishnapriyan et al., characterizing possible failure modes in PINNs
  (NeurIPS 2021) — https://arxiv.org/abs/2109.01050
- Mao, Jagtap, Karniadakis, PINNs for high-speed flows, CMAME 360 (2020) —
  https://doi.org/10.1016/j.cma.2019.112789
- Jagtap, Kharazmi, Karniadakis, conservative PINNs on discrete domains —
  https://doi.org/10.1016/j.cma.2020.113028
- Wang, Yu, Perdikaris, when and why PINNs fail to train: an NTK perspective,
  JCP 449 (2022) — https://doi.org/10.1016/j.jcp.2021.110768
- Kennedy, O'Hagan, Bayesian calibration of computer models —
  https://doi.org/10.1111/1467-9868.00294
- Wang, Teng, Perdikaris, gradient-pathology mitigation —
  https://doi.org/10.1137/20M1318043

The guide also lists Raissi et al. 2019, McClenny & Braga-Neto 2023 and
Willard et al. 2023, plus textbook chapters. Those links were not
re-verified in this pass.

## Blockers, limits and deferred work

- **Numerical execution BLOCKED** (battery power). Turbine training and
  scoring, MMS, the nozzle ladder, and focused and full pytest are all
  pending. No freeze-note entry exists, and `outputs/freeze/NUMBERS.md` is
  not created.
- Ma Eq. 25 thermal units are unresolved. Do not silently repair them.
- Loss balancing must not be chosen by held-out scores. Balancing
  cross-validation is deferred.
- M1 empirical residual fitting remains data/G2 gated. Production empirical
  residuals are deferred.
- Phase 7 review items R7-1 (report snapshot completion bundle), R7-2 (report
  verification) and R7-3 (resume pinning) remain **deferred, not fixed**.
- The Sajben low-label study and runner repairs are deferred until after the
  18 October freeze (`freeze-2026-10-18`, `outputs/freeze/`).
- The benchmark queue's wait predicate (`pgrep -if "python.*scripts/phase8/"`)
  matches the parked A2-calibration chain shell (bash PID 48047 and its
  caffeinate 48049, still present on 2026-10-03 UTC). That shell's command
  line names `.venv/bin/python scripts/phase8/ablation_ladder.py`, but it is
  only waiting for `run2_ac_rerun` to log "queue done". Each therefore waits
  for the other, and the queue has logged "waiting … before arm1a_W1_1w"
  every 2 min. This is a pre-existing blocker. It is flagged here, and the
  queue and shells were not altered. The Track 4 runner parses the actual
  executable of each process instead, so parked shells do not block it.
