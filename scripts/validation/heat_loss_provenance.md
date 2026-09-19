# Combustor heat-loss fraction ξ — provenance record (P4.5 step 1)

**Status: BLOCKED — no defensible sourced value found; escalated per `docs/plan.md` P4.5(1).**
Nothing in the code or calibration was changed on the basis of this document.
Date of search: 2026-09-18. Sources are listed in §4 so the search can be extended rather than repeated.

## 1. What ξ is in this code

`Combustor.run(heat_loss_fraction=ξ)` computes

    T_out = T_in + η_b · (1 − ξ) · (T_ideal − T_in)

so ξ is the fraction of the heat release **removed from the working fluid** between compressor exit
and turbine inlet. It is a cycle heat loss, not a liner heat flux. `integrated_engine.py` reads it from
`design_point['combustor_heat_loss_fraction']` (production default 0.0), and the Phase-2.6 sweep
`outputs/heat_loss_sensitivity.csv` shows ξ = 4 % moves T4 by 45 K and TSFC by 0.059 mg/(N·s).

The combustion efficiency η_b is a **fitted** calibration parameter (`calibrate_lto.py`, search range
0.96–0.999; v4 optimum 0.9963, `outputs/calibration_trent1000_ae3_v4.json`). Only the product
η_b(1 − ξ) enters the temperature rise, so a fixed ξ is identifiable only through η_b's bound: the v4
fit can absorb at most ξ ≈ 0.27 % (0.9963/0.999 − 1) without changing the calibrated cycle. Any larger
fixed ξ changes every manuscript-bound number by the choice of that number. `outputs/parameter_provenance.md`
already labels η_b a "lumped heat-delivery efficiency" for this reason.

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

## 3. Values considered and why none is adoptable under the plan's rule

| Candidate | Basis | Verdict |
|---|---|---|
| ξ from Lefebvre & Ballal ch. 8 | The plan's named source. Not accessible in this session (Google Books API quota exhausted; publisher excerpt timed out). From the chapter's structure the liner heat balance (R1, C1 vs R2, C2 to the annulus air) supports the recovery argument in §2 but does not give a cycle-loss fraction. | **no number to adopt**; user may supply a page-cited value |
| ξ ≈ 0.1 % (order of magnitude) | Casing ≈ 2 m² at ≈ T3 (≈ 900 K) losing to bay air by convection + radiation ≈ 0.05–0.1 MW against ≈ 100 MW take-off heat release. | an estimate this document made, **not a source**; prohibited by P4.5(1) |
| ξ ≤ 0.27 % | Headroom of the fitted η_b to its 0.999 bound. | a property of the fit, not physics; **not a source** |
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

## 5. Decision needed from the user (P4.5 is blocked here)

Options, in the order this record recommends them:

**A. ξ = 0, sourced.** Production stays at ξ = 0, now *justified* rather than defaulted: the combustor is
adiabatic to the cycle because liner heat is recovered by the annulus air (Lefebvre & Ballal ch. 8
heat-balance structure), external casing loss is below the resolution of the fitted η_b, and η_b is
declared the lumped heat-delivery efficiency (already so labelled). The Phase-2.6 sweep is reported as a
bound on the blend effect size, as it is now. Reviewer 1's point is answered in the model's *structure*
(fitted η_b) and in the physics, not by a limitations paragraph. Consequence for Phase 4: v5 differs
from v4 only through the P4.3/P4.4 component adjudication; no combustor recalibration.

**B. The user supplies a page-cited ξ.** If the user has Lefebvre & Ballal (or Walsh & Fletcher) to hand
and can cite a casing/cycle heat-loss fraction with page and engine class, this document is updated
with the citation and P4.5 steps 2–5 run exactly as planned (ξ fixed, not fitted; η_b re-fitted with
ξ recorded under `fixed_parameters`). Note that any ξ above ≈ 0.27 % will move v5 numbers.

**C. ξ fixed at an estimate.** Not recommended: it is the "learnable factor absorbing model error"
problem in different clothes, and P4.5(1) forbids it.

Until one of these is chosen, P4.5(2)–(5) and the combustor side of P4.6 do not run.
