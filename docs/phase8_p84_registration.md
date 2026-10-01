# P8.4 prospective registration: design point and off-design matching

Date: 2026-09-30. Parent: `docs/phase8_r2_plan.md` (P8-R2, `1974420`),
section P8.4. Committed before any P8.4 build, solve or comparison.

Known when written: G0 PASS (`74f53c8`); P8.2 G1 PASS (`df3a6c1`); P8.3 G1
PASS (`ef4e928`); the pinned pyCycle 4.4.0 HBTF reference
(`outputs/phase8/pycycle_hbtf_reference.json`, `42a3003`) reproduces 26/26
upstream values. The P8-A2 ladder A1 is scored (1.835 %); A2 is not yet
calibrated; A3 is blocked on its definition. No P8.4 code exists and no
off-design or design-point result has been computed.

## 1. Scope of this registration

P8.4 delivers, in order: (a) generic map tables and scaling; (b) a
design-point solver; (c) a damped bounded Newton off-design solver; (d) the
code-to-code check against the pinned pyCycle HBTF example (G1). Applying
P8.4 to the Trent 1000 (three-shaft) and retiring the four v6 knobs is
registered separately (P8.4b) **after** this G1, together with the cited
design-point inputs; this file fixes only the solver, its numerics and its
check.

## 2. Architecture and station graph (two-shaft HBTF, mirrors pyCycle)

Flight conditions (US 1976 standard atmosphere, `dTs`) -> inlet (ram
recovery) -> fan -> splitter (BPR) -> core: duct4 -> LPC -> duct6 -> HPC
(bleeds cool1, cool2, cust) -> bld3 (cool3, cool4) -> burner -> HPT (cool3
re-enters before the rotor, frac_P 1; cool4 after, frac_P 0) -> duct11 ->
LPT (cool1 before, cool2 after) -> duct13 -> core nozzle; bypass: byp_bld
(bypBld) -> duct15 -> bypass nozzle. LP shaft: fan, LPC, LPT; HP shaft: HPC,
HPT, power extraction HPX. Element parameters (dPqP, Cv, bleed frac_W,
frac_P, frac_work, ram recovery, HPX, design PRs/efficiencies, Mach numbers,
design spool speeds, T4, Fn) are exactly those of the pinned
`envs/pycycle/upstream/example_cycles/high_bypass_turbofan.py`.

Bleeds follow pyCycle: a compressor bleed with `frac_P` leaves at
`P_in + frac_P (P_out − P_in)` with enthalpy `h_in + frac_work (h_out − h_in)`;
turbine re-entry with `frac_P` mixes at `P_in + frac_P (P_out − P_in)`
(1 = before the rotor, 0 = after) by enthalpy and composition (P8.2 mixing
rule). Nozzles are convergent (`CV`) with Cv on velocity, using the P8.3
real-gas choking module with Cd = 1 (pyCycle CV nozzle has no Cd).

## 3. Thermodynamics: two modes

- **Thermo-matched mode (G1 only).** A Cantera YAML generated once by
  `scripts/phase8/pycycle/export_janaf.py` from the pinned pyCycle
  `thermo/cea/thermo_data/janaf.py` (NASA 9-coefficient polynomials, the
  same species list) and pyCycle's `CEA_AIR_COMPOSITION` element ratios,
  with fuel `Jet-A(g)` as in the example. The file and the exporter are
  hashed. Equilibrium at each burner/mixer uses Cantera HP equilibrium over
  that species set. This makes the comparison test the solver, maps and
  component equations, not the thermochemistry.
- **Production mode.** CRECK (`data/creck_c1c16_full.yaml`) with the
  registered Dooley 2012 surrogate and air `O2:1, N2:3.76`, liquid-fuel
  enthalpy as P8.2. Differences between modes are **reported**, never
  compared at a tolerance and never tuned.

## 4. Maps

The five pyCycle generic maps (`FanMap`, `LPCMap`, `HPCMap`, `HPTMap`,
`LPTMap`) are exported once to `data/maps/pycycle_4.4.0/*.json` (Apache-2.0,
attribution and source hash recorded) by `scripts/phase8/pycycle/export_maps.py`.
Interpolation: multilinear on the map's own grids, linear extrapolation
outside them (pyCycle `slinear`, `extrapolate=True`); any extrapolated
evaluation is flagged in the solution record.

Scaling at the design point (pyCycle convention): with the design
`RlineMap` and the map's `Nc_DES`/`alphaMap`, `s_Nc = Nc_design / NcMap_des`,
`s_PR = (PR_des − 1)/(PRmap_des − 1)`, `s_Wc = Wc_des / WcMap_des`,
`s_eff = eff_des / effMap_des`; turbines analogously on `(NpMap, PRmap)` with
`s_Np`, `s_PR`, `s_Wp`, `s_eff`. Scalars are frozen after the design point.

## 5. Unknowns and residuals

**Design point** (4 x 4): unknowns W, FAR, HPT PR, LPT PR; residuals
Fn − Fn_DES, Tt4 − T4_MAX, HP power balance, LP power balance. Areas
(component exit areas from design Mach numbers, nozzle throat areas) and
map scalars are outputs.

**Off-design** (10 x 10): unknowns W, FAR, BPR, N_LP, N_HP, Rline_fan,
Rline_LPC, Rline_HPC, PR_HPT, PR_LPT; residuals: throttle (Tt4 − T4_MAX, or
Fn − PC·Fn_max), core-nozzle throat area − design, bypass-nozzle throat area
− design, LP and HP power balances, fan/LPC/HPC map corrected flow −
flowpath corrected flow, HPT/LPT map flow parameter − flowpath flow
parameter. Power residuals are normalised by the design shaft power, flow
residuals by design flows, area residuals by design areas, temperature by
T4_MAX, thrust by Fn_DES.

## 6. Numerics

Damped Newton with central finite-difference Jacobian (relative step 1e-6
on box-scaled unknowns), Armijo backtracking (rho 0.75, up to 6 steps),
bounds W 10–1000 lbm/s equivalent, BPR 2–10, Rline 1.0–3.0, PR 1.001–8,
spool speeds 0.3–1.3 of design, FAR 1e-4–0.06. Convergence: max |scaled
residual| < 1e-10, at most 50 iterations. Initialisation: the design-point
solution for the first off-design point of a sequence, then the previous
converged point (pyCycle's sequence order). A point that fails reports its
residual vector and reason; it is not restarted from a tuned guess.

## 7. G1 (write-once `outputs/phase8/p84_g1.json`)

Compared quantities at DESIGN, OD_full_pwr and OD_part_pwr (PC 0.8) of the
pinned reference: W, FAR, OPR, Fn, Fg, TSFC, BPR, Tt3, Tt4, both spool
speeds (off-design), HPT/LPT PR, fan/LPC/HPC PR and efficiency, and the
station Tt, Pt, W listed by the reference record.

1. Thermo-matched mode: every compared quantity within **0.10 %** relative
   of the pyCycle reference (temperatures in K, absolute 0.5 K if smaller).
2. Converged audit at all three points: mass closure (all streams incl.
   bleeds and fuel) and energy closure (shaft powers, HPX, nozzle kinetic
   energy) relative < 1e-8; element closure < 1e-10 with production thermo.
   (1e-8 rather than 1e-10 for energy: the map interpolation and Newton
   tolerance bound the shaft-balance residual, not round-off.)
3. Finite, bounded solutions; no map extrapolation at the three points
   (extrapolation reported if it occurs).
4. Production mode at the same three points: reported deltas only.

A G1 failure is recorded; P8.4b and the A4 ladder step do not start;
P8.5 and the data tracks continue.

## Relevant files

`cpp/catjet_core/maps.{hpp,cpp}`, `offdesign.{hpp,cpp}`,
`cpp/tests/test_offdesign.cpp`, `cpp/bindings/catjet_core.cpp`,
`scripts/phase8/pycycle/export_maps.py`, `export_janaf.py`,
`compare_hbtf.py`, `data/maps/pycycle_4.4.0/`, `data/thermo/pycycle_janaf.yaml`,
`tests/test_phase8_p84.py`.
