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
| ICAO Aircraft Engine Emissions Databank, EASA "Emissions Databank (03/2026)" | easa.europa.eu/en/downloads/131424/en | 57a9ff57…69302530 | engine | tabulated | downloaded; identifier columns only read (`outputs/phase8/icao_edb_families.*`); no target column decoded |

Accessed 2026-09-29 for every row.
