# Phase 6 P6.3 — pre-registration: SAF blends at matched thrust (2026-09-26)

Machine-readable and authoritative: `outputs/phase6/p63_registration.json`.
Driver: `scripts/optimization/blend_matched_thrust_v5.py`; analysis:
`scripts/analysis/variance_decomposition.py --v5`. Committed before the first run.

- **Engine and operating points.** Trent 1000-AE3 inputs, v5 calibration
  (`outputs/calibration_v5_A2.json`), A1 central fixed values. Every blend runs at
  the ICAO thrust of each LTO mode (take-off primary); φ is solved, not chosen.
- **Blend basis.** Mass fractions of the four component surrogates, converted to
  mixture mole fractions for Cantera. `make_saf_blend` mixes on a mole basis (a
  nominal 50 % ATJ blend is 42.5 % ATJ by mass), which is inconsistent with the
  per-kg lifecycle accounting, so it is not used here. SAF ≤ 50 % by mass (ASTM
  D7566 limits are volumetric; densities are not modelled).
- **Design.** 256-point scrambled Sobol sample (seed 42) plus Jet-A1, HEFA-50,
  FT-50, ATJ-50. CORSIA Doc 06 triangular draws: one per design point (variance
  decomposition) and 1000 common draws (bands, rank stability).
- **P6.2 conditioning.** Reference blends re-run under each of the 64 P6.2 draws.
- **Ranking rule (gating).** A ranking between two blends is claimed only if the
  central difference exceeds both (a) the Monte-Carlo spread (P5–P95 width of the
  paired difference over the common CORSIA draws; 0 for the deterministic cycle
  quantities) and (b) the P6.2 band (P5–P95 width of the quantity for Jet-A1).
  Reported, not gating: the paired P6.2 difference and its sign agreement.
- **NOx** is the ICAO-derived correlation of OPR and fuel flow; it has no
  composition term, so blend NOx differences only restate fuel-flow differences.
- **Supersedes** the `optimize_blend.py` free-φ and fixed-φ studies (unequal
  thrust), which remain only as the v4 reproduction path.
