# Combustor heat-loss fraction ξ — provenance record (P4.5 step 1)

**Status: DECIDED 2026-09-19 — ξ = 0 in production, by the structural argument in §2, not as a sourced value.**
The user accepted option A of §6 with two corrections, both applied below: (i) no source says ξ = 0
either — what is sourced is the *structure* (liner heat is recovered by the annulus air upstream of the
turbine; only casing loss is a cycle term), and the preprint claims the argument, not a value; (ii) a
bound from the calibration residual was proposed and was **tested and found not to hold in this model**
(§5). Date of search: 2026-09-18; sources in §4.

## 1. What ξ is in this code

`Combustor.run(heat_loss_fraction=ξ)` computes

    T_out = T_in + η_b · (1 − ξ) · (T_ideal − T_in)

so ξ is the fraction of the heat release **removed from the working fluid** between compressor exit
and turbine inlet. It is a cycle heat loss, not a liner heat flux. `integrated_engine.py` reads it from
`design_point['combustor_heat_loss_fraction']` (production default 0.0), and the Phase-2.6 sweep
`outputs/heat_loss_sensitivity.csv` shows ξ = 4 % moves T4 by 45 K and TSFC by 0.059 mg/(N·s).

The combustion efficiency η_b is a sampled calibration parameter (`calibrate_lto.py`, search range
0.96–0.999; v4 sampler's value 0.9963, not an estimate — §5). Only the product η_b(1 − ξ) enters the temperature rise. An earlier draft of this record
argued that η_b's headroom to its 0.999 bound (≈ 0.27 %) limits an absorbable ξ; §5 shows that η_b is
not determined by the calibration at all, so that argument is withdrawn.

## 2. What the reviewer asked and what would answer it

Reviewer 1 (lines 303–310, as recorded in `docs/plan_phase2_completed.md` §2.6 and
`docs/response_letter_draft.md`): *case/liner heat loss can exceed the incomplete-combustion loss*.

Physically, for an annular aero-engine combustor:

- Heat transferred from the flame to the **liner** (radiation + convection, Lefebvre & Ballal,
  *Gas Turbine Combustion*, 3rd ed., ch. 8 "Heat Transfer") is removed from the liner by the
  **annulus air**, which then enters the liner as film-cooling and dilution air. That heat re-joins
  the working fluid upstream of the turbine; it is not a loss to the cycle. Lefebvre & Ballal treat
  liner heat transfer for liner-temperature and cooling design, not as an energy loss term. The
  reviewer's statement is true of liner heat *transfer* (several per cent of heat release) and does
  not by itself imply a cycle loss of that size.
- The cycle loss is the heat that leaves through the **outer casing** to the engine bay. No accessible
  source quantifies it for a Trent-class engine; every performance-modelling reference located
  treats the combustor as adiabatic apart from η_b (§4).

So "turning on ξ" at a liner-heat-transfer magnitude (the 2–6 % range of the Phase-2.6 sweep) would
model energy leaving the cycle that, in the real engine, does not. That is the substantive finding of
this step.

**Caveat the preprint must state (user, 2026-09-19).** The recovery argument assumes the cooling air
re-enters *upstream of the turbine*. In the code, `combustor_air_fraction = 0.8` sends 20 % of the core
air past the burner and remixes it with the products at compressor-exit temperature *at the combustor
exit* (`integrated_engine.py`, "Combustor dilution mix", T4_mix), i.e. upstream of the turbine — in
energy terms exactly full recovery. Turbine cooling air that re-enters **downstream of the NGV throat**
bypasses part of the expansion and is a genuine cycle effect; the model does not represent it. The
preprint should say: the split captures liner-cooling and dilution air (recovered), not turbine cooling
air (not modelled).

## 3. Values considered and why none is adoptable under the plan's rule

| Candidate | Basis | Verdict |
|---|---|---|
| ξ from Lefebvre & Ballal ch. 8 | The plan's named source. Not accessible in this session (Google Books API quota exhausted; publisher excerpt timed out). From the chapter's structure the liner heat balance (R1, C1 vs R2, C2 to the annulus air) supports the recovery argument in §2 but does not give a cycle-loss fraction. | **no number to adopt**; user may supply a page-cited value |
| ξ ≈ 0.1 % (order of magnitude) | Casing ≈ 2 m² at ≈ T3 (≈ 900 K) losing to bay air by convection + radiation ≈ 0.05–0.1 MW against ≈ 100 MW take-off heat release. | an estimate this document made, **not a source**; prohibited by P4.5(1) |
| ξ ≤ 0.27 % | Headroom of the fitted η_b to its 0.999 bound. | withdrawn: η_b is not fitted in any meaningful sense (§5) |
| ξ ≲ a few 0.1 % from the held-out fuel-flow residual | Proposed 2026-09-19: v4 fits held-out fuel flow to 2.50 % with η_b off its bound, so a large cycle loss would have shown as a systematic bias. | **does not hold**: fuel flow is independent of η_b(1 − ξ) by construction (§5) |
| "conventional combustors lose ~2 orders of magnitude less than micro-combustors" | Micro-combustion literature (surface-to-volume argument; micro-combustors lose tens of per cent). | supports "O(0.1–1 %)" qualitatively; no engine-class number |
| "total radiative heat flux to the walls < 25 % of the energy supplied … justifying the adiabatic assumption" | Bahador, Nilsson & Sundén, *On heat load calculations in gas turbine combustors*, WIT Trans. Eng. Sci. 46 (2004), citing a CFD study of a laboratory combustor. | wall **flux**, not cycle loss; lab geometry; **not applicable** |
| ξ = 0 | Adiabatic-combustor treatment used by the cycle-analysis references located (§4); η_b lumps heat delivery. | the only value with literature support, **but it is the status quo** the user asked to move away from — a decision, not an executor call |

## 4. Sources examined

1. Lefebvre, A. H. & Ballal, D. R., *Gas Turbine Combustion: Alternative Fuels and Emissions*, 3rd ed.,
   CRC Press 2010, ch. 8 (Heat Transfer) — text not accessible in session; chapter scope from the
   publisher's table of contents.
2. Bahador, M., Nilsson, T. K. & Sundén, B., "On heat load calculations in gas turbine combustors",
   WIT Transactions on Engineering Sciences 46 (2004) — full text read; one quantitative statement (table above).
3. Walsh, P. P. & Fletcher, P., *Gas Turbine Performance*, 2nd ed., Blackwell 2004 — paywalled; no
   heat-loss guideline surfaced in searchable metadata.
4. Kurzke, J., *GasTurb 15 User Manual* (gasturb.com) — downloaded; rasterised PDF, 7 of 369 pages
   carry text; burner section not extractable. Kurzke & Halliwell, *Propulsion and Power* (Springer 2018) — not accessible.
5. NETL *Gas Turbine Handbook* §3.2.1.1 (Conventional Type Combustion) — full text read; no heat-loss quantification.
6. Seitzman, J. M., Georgia Tech AE4451 "Aircraft Combustors" notes (2020) — full text read; energy
   equation only, no heat-loss term.
7. Aerostudents "The combustion chamber" notes (after Saravanamuttoo) — full text read; heat loss named
   qualitatively ("heating the combustor itself"), no number.
8. NASA NTRS searches ("combustor heat loss", "combustor heat balance casing", "liner heat transfer
   percent heat release") — 38 hits, none quantifying casing heat loss for a large engine. The one
   external-heat-balance report found (Meng & Wulf, NASA TM X-3223, 1975) is for a 150 hp automotive
   gas turbine and reports whole-engine, not combustor, losses.
9. Micro-combustion literature (Wikipedia "Micro-combustion" and its references) — surface-to-volume
   argument only.
10. `docs/response_letter_draft.md`, `docs/plan_phase2_completed.md` §2.6, `outputs/parameter_provenance.md`
    (ξ and η_b rows), `outputs/heat_loss_sensitivity.csv` — repo record of the reviewer point and the sweep.

## 5. What the calibration can and cannot say about ξ — measured 2026-09-19

The proposed bound was tested directly. At the v4 take-off point (`IntegratedTurbofanEngine`, Jet-A1,
φ_to from the v4 record):

| η_b | ξ | fuel flow (kg/s) | T4 (K) | thrust (kN) | TSFC (mg/N·s) |
|---|---|---|---|---|---|
| 0.9963 | 0.00 | 2.31821 | 1894.8 | 241.61 | 9.595 |
| 0.9600 | 0.00 | 2.31821 | 1858.3 | 238.40 | 9.724 |
| 0.9990 | 0.00 | 2.31821 | 1897.6 | 241.84 | 9.586 |
| 0.9963 | 0.04 | 2.31821 | 1854.7 | 238.08 | 9.737 |

Fuel flow is identical to five decimals. The fuel–air ratio is set by φ, so fuel flow =
φ·f_stoich·β·ṁ_core, and neither η_b nor ξ enters it. The calibration objective
(`calibrate_lto.py`, mean |Δfuel flow| over three modes) therefore has **zero sensitivity to η_b**, and
the held-out 2.50 % residual carries no information about η_b(1 − ξ). Neither the headroom argument nor
the residual argument bounds ξ. What ξ does move is T4 (−40 K at 4 %), thrust (−1.5 %) and TSFC (+1.5 %),
none of which is fitted or validated against data at the take-off point (the model's 241.6 kN static
thrust vs the ICAO rated 310.9 kN is a separate, larger discrepancy discussed in the manuscript).

The same test on the other sampled parameters at the approach point:

| Parameter (range) | fuel flow at range ends | Sensitivity of the objective |
|---|---|---|
| η_b (0.96 → 0.999) | 1.09655 → 1.09655 | **none** |
| pressure_loss (0.03 → 0.06) | 1.09655 → 1.09655 | **none** |
| k_pi (0.5 → 1.5) | 1.09655 → 1.09655 | **none** — yet approach thrust 83.7 → 57.1 kN, T4 1373 → 1184 K, TSFC 7.40 → 10.85 |
| k_mdot (0.3 → 1.0) | 1.615 → 0.695 | strong |
| φ_app (0.30 → 0.40) | 1.014 → 1.216 | strong (by construction) |

**Finding.** Of the seven parameters Optuna samples, the objective identifies **φ_to alone**.
[Corrected 2026-09-25: this paragraph originally said "identifies four (k_mdot and the three φ)". The
table above shows k_mdot and φ_app each *move* fuel flow, but not that they are separately identified:
fuel flow is φ·f_st·β·ṁ_rated·x^k_mdot, so take-off (x = 1) pins φ_to while idle and approach give two
equations in three unknowns — k_mdot, φ_idle and φ_app lie on a ridge of identical objective. Measured
table: `docs/plan.md` (Phase 5), finding F2; also `outputs/parameter_provenance.md`, Phase 5 section.]
η_b = 0.9963, pressure_loss = 0.0442 and k_pi = 0.562 are wherever the seeded sampler landed
and carry no information. Every manuscript-bound T4, thrust and TSFC depends on these three; across
their search ranges that dependence is ±20 K / ±0.7 % at take-off from η_b alone and far larger at part
power from k_pi. This is a Phase-4 finding outside P4.5's scope, recorded here because it was found here.
[Superseded 2026-09-25 — the following were Phase-4 recommendations, not current facts:
"`parameter_provenance.md` must now say the same of η_b and k_pi" (done in Phase 5 P5.0, Phase 5
section of that file), and "how to fix it … is a user decision for P4.6/P4.8" (the repair is now
`docs/plan.md` Phase 5 P5.2: thrust targets added to the objective, η_b and pressure_loss fixed at
page-cited values or escalated).]

**Consequence for ξ.** The decision rests on §2 alone. ξ = 0 is adopted because the cycle-relevant
casing loss has no engine-class source and every performance reference located treats it as negligible;
η_b is *declared* the lumped heat-delivery efficiency and its value is *unidentified* by the data. The
Phase-2.6 sweep remains the reported sensitivity.

## 6. Decision needed from the user — RESOLVED (option A, with the corrections above)

Options, in the order this record recommends them:

**A. ξ = 0, sourced.** Production stays at ξ = 0, now *justified* rather than defaulted: the combustor is
adiabatic to the cycle because liner heat is recovered by the annulus air (Lefebvre & Ballal ch. 8
heat-balance structure), external casing loss has no engine-class source and every performance reference
located treats it as negligible (the fuel-flow data provide **no** bound on it — §5), and η_b is
declared the lumped heat-delivery efficiency (already so labelled). The Phase-2.6 sweep is reported as a
bound on the blend effect size, as it is now. Reviewer 1's point is answered in the model's *structure*
(a lumped η_b) and in the physics, not by a limitations paragraph. [Superseded 2026-09-25: "Consequence
for Phase 4: v5 differs from v4 only through the P4.3/P4.4 component adjudication; no combustor
recalibration." Phase 5 (F2, P5.2) recalibrates v5 on a new objective; ξ = 0 is unchanged.]

**B. The user supplies a page-cited ξ.** If the user has Lefebvre & Ballal (or Walsh & Fletcher) to hand
and can cite a casing/cycle heat-loss fraction with page and engine class, this document is updated
with the citation and P4.5 steps 2–5 run exactly as planned (ξ fixed, not fitted; η_b re-fitted with
ξ recorded under `fixed_parameters`). Note that any ξ above ≈ 0.27 % will move v5 numbers.

**C. ξ fixed at an estimate.** Not recommended: it is the "learnable factor absorbing model error"
problem in different clothes, and P4.5(1) forbids it.

[Superseded: option A was chosen 2026-09-19. "Until one of these is chosen, P4.5(2)–(5) and the
combustor side of P4.6 do not run" described the state before that decision; P4.6 is replaced by
Phase 5 P5.3.]
