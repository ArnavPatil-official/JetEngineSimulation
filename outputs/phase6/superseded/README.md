# Superseded Phase 6 P6.1 registration (R0) — REJECTED before any pilot or fit

| File | Original path | SHA-256 (byte-identical to commit `59d9665`) |
|---|---|---|
| `p61_registration_R0_REJECTED.json` | `outputs/phase6/p61_registration.json` | `89e7d11630f2aa0bd1cbfb1e2a8fccf8e57c98b69132388ffb2e7ddb35158dc8` |
| `phase6_p61_registration_R0_REJECTED.md` | `docs/phase6_p61_registration.md` | `c11222550a4d010ff674f5d7173f13405d357d26f5b7207df6e4b010f5f26445` |

**Registered:** commit `59d9665` (2026-09-26). **Rejected:** planner review
`docs/plan_phase6_review.md` (commit `c55e182`), finding R6-A, before any pilot,
v5 calibration or held-out evaluation had run. Kept unmodified as evidence.

**Reason.** R0 fixed the burner-zone air fraction β = 0.8 over [0.70, 0.90]
labelled ILLUSTRATIVE with no source. Decision 2 of `docs/plan.md` requires an
engine parameter without a citable range to be data-identified or dropped; the
"illustrative" exception in P6.2 covers surrogate compositions only. R0 also
used an approximate fan polytropic-to-isentropic adjustment ("−0.005") and did
not document the η_b proxy assumptions (R6-C).

**Replaced by:** amendment A1 — `outputs/phase6/p61_registration.json` and
`docs/phase6_p61_registration.md` (commit recorded in
`outputs/phase6_execution_status.md`). A1 keeps R0's split, objective, weighting,
fitted set and box, baselines, margins and thresholds unchanged.
