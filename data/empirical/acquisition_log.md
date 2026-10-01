# Empirical source acquisition log (Phase 8, P8.6)

One row per candidate source examined, entered or not. Files are not in git
(`data/empirical/pdf/`, `data/empirical/raw/`); their sha256 identifies them.
"Data form" says whether values are tabulated (can be transcribed and
cross-checked) or plotted only (need digitisation per vocabulary v2: three
repeats, axis calibration, class B/C).

| Source | NTRS / URL | sha256 (file) | Component | Data form | State |
|---|---|---|---|---|---|
| NASA CR-168189, Leach 1983 (P&W E3 HPT cooled rig) | 19850021643 | 9b70c6c1…a3db754d | turbine | tabulated (Table 5.3.1-II, 27 points; uncertainties Table 3.5.3-I/II) | **entered** (class A) |
| NASA TN D-6967, Kofskey & Nusbaum 1972 (two-stage cold-air turbine, small turbofan) | 19720024422 | 38c90cc6…22b96536 | turbine | plotted only (work, torque, flow, efficiency maps); design tables I–III only | needs digitisation |
| NASA TP-2991, Berrier & Taylor 1990 (gimballed axisymmetric C-D and 2-D C-D vectoring nozzles) | 19900009884 | 670c2f53…a760acb8 | nozzle | Cd and thrust ratio plotted; tables are wall static-pressure ratios | poor fit for the P8.3 convergent nozzle (C-D, vectoring, (NPR)d 5.9–8.8); keep for the C-D option only |
| NACA TR-933 (1949) / TN-1757 (1948), Grey & Wilsted, conical jet nozzles, flow and velocity coefficients vs pressure ratio | 19930091998 / 19930082415 | 516f8d65…861abe3a / 7d773bdf…9ebdc00 | nozzle | plotted only (coefficients vs pressure ratio across choking); Table I is geometry | best-matched convergent-nozzle source; needs digitisation |
| NASA CR-168289, Timko 1984 (GE E3 two-stage HPT warm-air rig) | [NTRS 19900019237](https://ntrs.nasa.gov/citations/19900019237) | 2e6f8a69…35d6d | turbine | Table XIII TEST column, report p. 82/PDF p. 92: measured PR and flow function; abstract: thermodynamic efficiency. Detailed maps plotted. | **entered** (one measured design point, class B); maps need digitisation |
| NASA CR-168290, Bridgeman, Cherry & Pedersen 1983 (GE E3 five-stage LPT scaled air rig) | [NTRS 19900019247](https://ntrs.nasa.gov/citations/19900019247) | 2a900305…63ddec | turbine | Appendix H tabulates measured Block I/II runs (report pp. 222–237); Table XI is design intent. | **entered partially** (four legible Configuration 5 runs, class C); full scan review remains |
| NASA CR-168069, Stearns et al. 1982 (GE E3 core design/performance) | [NTRS 19900019243](https://ntrs.nasa.gov/citations/19900019243) | ee3abbf9…eaf8a | combustor/core | Table XIX, report p. 239/PDF p. 265, tabulates raw measured P3, T3, FAR, P4 and EI; Table XX is cycle-adjusted and excluded. | **entered** (19 station rows, 17 with emissions, class B); W36 kg/s versus lbm/s inconsistent by about 4%, withheld |
| NASA TP-2721, Bare & Reubush 1987 (Langley static internal nozzle performance) | [NTRS 19870014999](https://ntrs.nasa.gov/citations/19870014999) | not downloaded | nozzle | two-dimensional C-D nozzle with thrust vectoring; NPR up to 10 | poor match for separate convergent P8.3 nozzles; not entered |
| NASA TP-3411, Wing 1994 (Langley skewed-throat static nozzle) | [NTRS 19940029666](https://ntrs.nasa.gov/citations/19940029666) | not downloaded | nozzle | C-D multiaxis thrust-vectoring nozzle, NPR 2–11.5 | poor match for the fixed-area convergent P8.3 nozzle; not entered |
| ICAO Aircraft Engine Emissions Databank, EASA "Emissions Databank (03/2026)" | easa.europa.eu/en/downloads/131424/en | 57a9ff57…69302530 | engine | tabulated | downloaded; identifier columns only read (`outputs/phase8/icao_edb_families.*`); no target column decoded |

| NASA TP-2171, Straight & Cullom 1983 (full-scale 2-D C-D nozzle on a J85 turbojet at altitude) | [NTRS 19830018568](https://ntrs.nasa.gov/citations/19830018568) | not stored | nozzle | abstract: corrected gross thrust coefficients >= 0.985 for NPR > 4 | **reference only** (2-D C-D, fighter geometry): not entered; shows full-scale Cfg near 0.985-0.99, relevant to the P8.3 Cv = 0.95 analog prior |
| NASA TM-2000-209948, Saiyed, Mikkelsen & Bridges 2000 (Separate Flow Nozzle Test, high-BPR core + fan convergent nozzles, model scale) | [NTRS 20000083968](https://ntrs.nasa.gov/citations/20000083968) | not stored | nozzle | thrust coefficient C_T (accuracy +-0.25 points) static and M 0.28; tables and Figs. 17-23 give only losses relative to baseline 3BB; no absolute C_T, Cd or Cv | **dead end for absolute coefficients** (2026-10-01); geometry matches P8.3 separate convergent nozzles |

| NASA CR-2000-210039, Janardan et al. 2000 (GE, SFNT concept evaluation) | [NTRS 20010061345](https://ntrs.nasa.gov/citations/20010061345) | not stored | nozzle | acoustics only; Section 8 recommends that thrust and flow coefficients still be measured | **dead end** (2026-10-01) |
Accessed 2026-09-29 for every row except the three rows dated 2026-10-01.
