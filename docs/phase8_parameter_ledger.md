# Phase 8 parameter ledger

One row per new or retired parameter, extended in the same commit as the
code that introduces it. The P8.0 ledger template and A1 penalty guard in
`docs/phase8_registration.md` govern this file. Fixed ranges are sensitivity
envelopes, not fitted confidence intervals.

| Name / unit | Component / ladder | Status | Central; range | Constraining source and meaning | Identifiability |
|---|---|---|---|---|---|
| W_ref / kg s^-1 | v6 matching / A0–A3 | `calibrated:v6:Trent` pending retirement at A4 | 103.2852; frozen v6 box | `outputs/phase7/calibration_v6.json` | A1 FAIL, penalty-dependent |
| a_thrust / 1 | v6 matching / A0–A3 | `calibrated:v6:Trent` pending retirement at A4 | 1.10982; frozen v6 box | same | A1 FAIL, penalty-dependent |
| k_pi / 1 | v6 matching / A0–A3 | `calibrated:v6:Trent` pending retirement at A4 | 1.34644; frozen v6 box | same | A1 FAIL, penalty-dependent |
| k_mdot / 1 | v6 matching / A0–A3 | `calibrated:v6:Trent` pending retirement at A4 | 0.413812; frozen v6 box | same | A1 FAIL, penalty-dependent |
| delta_h_vap / MJ kg^-1 fuel | liquid fuel / A1 | `fixed-cited` | 0.360; 0–0.360 mathematical gas-to-liquid limit | `data/fuel_properties_v7.yaml`, Viton et al. 1996 n-dodecane value; zero is gas-basis limit | n/a (fixed) |
| f_NGV / fraction of core inlet air | HP vane cooling / A2 | `fixed-cited` | 0.0641; 0–0.0641 no-cooling/design range | [NASA CR-168189](https://ntrs.nasa.gov/citations/19850021643), §3.2.3, Fig. 3.2.3-4 (design, other-machine rig) | n/a (fixed) |
| f_rotor / fraction of core inlet air | HP blade cooling / A2 | `fixed-cited` | 0.0275; 0–0.0275 no-cooling/design range | NASA CR-168189, §3.2.3, Fig. 3.2.3-2 (design, other-machine rig) | n/a (fixed) |
| eta_poly / 1 | HP/IP/LP turbine / A2 | `fixed-cited` | 0.90; frozen v6 fixed range | `outputs/phase7/p72_registration.json` `fixed_central.eta_turbine_polytropic` | n/a (fixed) |
| N_pressure / steps | turbine integration / A2 | numerical setting | 50; verification 100 | `docs/plan.md` P8.2 and P8.0 G1 step-doubling rule | n/a |

The NASA rig's **total** cooling flow is 14.56% design and 12.36% measured
(Table 5.3.1-I). It includes flows not represented by `f_NGV+f_rotor`; do
not fit those totals to the Trent model or treat their difference as a
confidence interval for either parameter.


## Prospective P8.3-A2 / A4c additions — 2026-10-03

These rows apply to the new A4c model. Earlier A4 values and results remain
historical. The authoritative ranges, source caveats and C1 numerical rule are
in `docs/phase8_p83_a2_registration.json` and `docs/phase8_p84c_registration.json`.
No fit or identifiability result exists yet. Existing P8-R2 stage 4 combustor
calibration remains intact; A4c is additive engine calibration.

| Name / unit | Component / ladder | Status | Central; range | Constraining source and meaning | Identifiability |
|---|---|---|---|---|---|
| Cv_core, Cv_bypass / 1 | nozzle / A4c | fixed analog prior | 0.985; 0.95–1.00 | P8.3-A2: full-scale NASA TP-2171 Cfg is a distinct gross-thrust coefficient; pyCycle velocity coefficients are generic references; bounds are chosen engineering endpoints | n/a; OAT sensitivity pending |
| Cd_core, Cd_bypass / 1 | nozzle / A4c | fixed analog prior | 0.96; 0.90–0.99 | P8.3-A2: TP-2171 measured Cd 0.965/0.957 is distinct from Cfg; chosen envelope, not a Trent measurement | n/a; OAT sensitivity pending |
| T4_K / K | shared engine / A4c | calibration pending | 1800; 1700–1900 | Martinez sister-engine reference; P8.4b engineering endpoints | A1/profile pending |
| FPR / 1 | shared engine / A4c | calibration pending | 1.45; 1.40–1.55 | frozen v6 central; upper Martinez reference, lower engineering endpoint | A1/profile pending |
| eff_comp_offset / 1 | shared fan/IPC/HPC / A4c | calibration pending | 0; −0.02–0.02 | offsets to pinned pyCycle centers; chosen endpoints | A1/profile pending |
| eff_turb_offset / 1 | shared HPT/IPT/LPT / A4c | calibration pending | 0; −0.02–0.02 | offsets to pinned pyCycle centers; chosen endpoints | A1/profile pending |
| cooling_scale / 1 | shared HPT cooling / A4c | calibration pending | 1; 0–1 | multiplies CR-168189 NGV 0.0641/rotor 0.0275; no-cooling/cited-design limits | A1/profile pending |
| design_W_max_kg_s / kg s^-1 | numerical design box / A4c | opt-in numerical setting | default 0; new paths max(3000×0.45359237, 4×rated_N/220) | C1: public-thrust-derived search box; zero retains exact old architecture caps; no physical-flow or validation claim | n/a; compiled/default and all-20 checks pending |
