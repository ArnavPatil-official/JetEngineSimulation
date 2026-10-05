# P8.4b prospective registration: Trent 1000 three-shaft model (ladder A4)

Date: 2026-10-01. Parents: `docs/phase8_r2_plan.md` (P8.4), P8.4
registration and P8.4-A1, P8.4 G1 PASS (`eaf61f5`), P8-A3 (`5958df1`: A3
merged into A4, A4 without knobs). Committed before any P8.4b build, solve,
calibration-row prediction or held-out score.

Known when written: A1 scored once (1.835 %); A2 not yet calibrated; the
two-shaft solver reproduces pyCycle to 3.9e-5; no Trent three-shaft solve
has been run. No held-out row is used below except in the single A4 score.

## 1. Architecture (EASA TCDS E.036, Trent 1000)

Three shafts: LP = single-stage fan + 6-stage LPT; IP = 8-stage IPC +
single-stage IPT; HP = 6-stage HPC + single-stage HPT. Separate-flow core
and bypass convergent nozzles. Station graph: inlet -> fan -> splitter ->
duct -> IPC -> duct -> HPC -> HPT cooling offtake -> burner -> HPT -> duct ->
IPT -> duct -> LPT -> duct -> core nozzle; bypass duct -> bypass nozzle.
Element equations are the P8.4 (G1-verified) ones; the IP spool is a third
instance of the same compressor/turbine/shaft equations.

## 2. Thermodynamics and component laws

- Production thermo (CRECK, Dooley 2012 surrogate, air `O2:1, N2:3.76`),
  frozen outside the burner; burner HP equilibrium of air + fuel gas at the
  burner-inlet temperature with liquid-basis fuel enthalpy (−360 kJ/kg), then
  the v6/P8.2 eta_b convention: `T_out = T_in + eta_b (T_eq − T_in)` at the
  equilibrium composition, eta_b = the frozen v6 per-mode value of the row's
  ICAO mode (`p72_registration.json fixed_central.eta_b`). This keeps the
  burner identical to ladder A1/A2.
- Nozzles: the P8.3 law (real-gas p*, `m = Cd A G`, `u = Cv u_ideal`,
  pressure thrust on geometric area, frozen Y) with the P8.3 fixed analog
  priors Cd = 0.96, Cv = 0.95 for both nozzles. Geometric areas are set at
  the design point and frozen.

## 3. Design point (per ICAO engine record)

Sea-level static ISA (288.15 K, 101325 Pa, M = 0), the record's rated
thrust, OPR and BPR (ICAO design columns, as v6). Fixed, cited inputs:

| Input | Central | Sensitivity range | Source |
|---|---|---|---|
| T4 at take-off | 1800 K | 1700–1900 K | Martinez, *Aerospace engine data* (UPM, Isidoro Martínez): Trent XWB maximum TET 1800 K, OPR 50; outer envelope 1500–2200 K from the Trent 1000-A performance study (ResearchGate 329944773). Data-limited; sister engine (Trent XWB is a cross-family held-out family; only its published specification is used). |
| FPR (both streams) | 1.45 | 1.40–1.55 | frozen v6 `fpr_rated`; upper end from the Martinez table (1.55 for older lower-BPR engines) |
| IPC/HPC split | equal pressure ratio per stage over 8 + 6 stages | ±20 % in ln PR_IPC share | TCDS E.036 stage counts; rule, not data |
| Isentropic efficiencies | fan 0.8948, IPC 0.9243, HPC 0.8707, HPT 0.8888, IPT 0.8888, LPT 0.8996 | ±0.02 each | pinned pyCycle HBTF example (generic NPSS values; IPC = LPC, IPT = HPT) |
| Duct dP/P | fan–IPC 0.0048, IPC–HPC 0.0101, HPT–IPT 0.0051, IPT–LPT 0.0051, LPT–nozzle 0.0107, bypass 0.0149 | 0 to 2× | pyCycle HBTF example (IPT–LPT reuses duct11) |
| Burner dP/P | 0.045 | frozen v6 | `fixed_central.combustor_pressure_loss` |
| HPT cooling | NGV 0.0641 (enters before rotor), rotor 0.0275 (after), of HPC exit flow; IPT/LPT 0 | 0 to cited | CR-168189 (as P8.2) |
| Inlet recovery | 0.999 | 0.99–1.0 | pyCycle example |
| Customer bleed, power offtake | 0 | — | ICAO Annex 16 Vol II certification test conditions |
| Maps | fan FanMap, IPC LPCMap, HPC HPCMap, HPT and IPT HPTMap, LPT LPTMap; design at map defaults | — | pyCycle 4.4.0 export |

The IPC and HPC ratios satisfy `FPR (1−dP_fan-IPC) PR_IPC (1−dP_IPC-HPC) PR_HPC = OPR`
with `ln PR_IPC / ln(PR_IPC PR_HPC) = 8/14`. Design spool speeds are
normalisation constants (results are invariant to them with scaled maps).
Design unknowns: W, FAR, PR_HPT, PR_IPT, PR_LPT; residuals: Fn − rated,
T4 − 1800 K, three shaft balances.

## 4. Off-design (LTO rows)

Same static ISA conditions; throttle Fn = x · rated (ICAO mode x: take-off
1.00, approach 0.30, idle 0.07, as `lto_v5.MODE_X`). Unknowns (12): W, FAR,
BPR, N_LP, N_IP, N_HP, Rline fan/IPC/HPC, PR HPT/IPT/LPT; residuals (12):
thrust, core and bypass nozzle capacity at their fixed areas, three shaft
balances, three compressor and three turbine map flows. Path: continuation
in thrust from the design point in steps of 0.10 · rated, each converged
point initialising the next; only registered modes are outputs. Numerics as
P8.4 (damped Newton, central FD Jacobian, max scaled residual < 1e-10,
50 iterations per point). A non-converging row is unreachable with its
reason; it is scored as the v6 rule scores unreachable rows.

## 5. Checks before scoring (write-once `outputs/phase8/p84b_g1.json`)

1. At every AE3 mode: mass and energy closure < 1e-8, element < 1e-10.
2. The off-design solve at x = 1.00 reproduces its own design point
   (W, FAR, all PRs) to 1e-8 relative.
3. The two-shaft P8.4 G1 comparison still passes unchanged (regression of
   the refactored code).
4. Report (not gating) map extrapolation and the sensitivity of AE3 fuel flow
   at each mode to every range end in section 3, one at a time.

## 6. Ladder A4 score

After `p84b_g1.json` is committed: predict all 93 calibration rows (in-sample,
no fit) and score the 87 Trent held-out rows **once** with the frozen P7.2
held-out tables (as `ablation_ladder.py`). Report group-weighted and
mode-wise MAPE, B0/B1, signs and unreachable count, even if worse than v6.

## Relevant files

`cpp/catjet_core/offdesign.{hpp,cpp}` (three-shaft path), `cpp/bindings/catjet_core.cpp`,
`scripts/phase8/trent_p84b.py`, `tests/test_phase8_p84.py`.
