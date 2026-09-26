# outputs/archive/pre_phase6/ — superseded by Phase 6 (v5). Dead numbers: never cite.

Moved in P6.6 (2026-09-26) with `git mv`; `MAPPING.json` gives every old → new
path. Reverse any move with the opposite `git mv`. Protected files among them
are verified by content at these paths by
`scripts/validation/verify_protected_hashes.py` (the baseline is not regenerated).

| Group | Superseded by (manifest row) | Why |
|---|---|---|
| `holdout_icao_validation*` (v1–v4) + holdout plots | V3 (model vs B0 vs B1) | Finding F-A: a rated-thrust rescaling rule, not a test of the cycle (V4 cites the v4 CSV here) |
| `design_point_summary_v4.csv` | V5 | fixed-φ design point, 17 % take-off thrust gap |
| `identifiability_profile_v4.*` | V2 | v4 objective identified φ_to only |
| `takeoff_thrust_gap*` | V1 (fitted W_ref vs hand-set 79.9 kg/s) | the gap is now a fitted airflow |
| `results/*` (free-φ and fixed-φ studies, variance decomposition, rank stability, representative solution) + their plots | B1–B3 | blends compared at unequal thrust |
| `nox_dual_path.csv`, `plots/nox_path_comparison.png` | E3 | v3 states |
| `heat_loss_sensitivity.csv` + plot | E4 | fixed φ (fuel flow could not respond) |
| `airflow_split_sensitivity.csv` + plot | — | β dropped in registration A1 (single-zone) |

The scripts that wrote these files keep their v3/v4 code paths. Re-running one
of those paths writes to `outputs/` again; reproduce into a scratch copy instead.
