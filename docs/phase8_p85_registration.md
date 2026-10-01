# P8.5 prospective registration: combustor reactor network

Date: 2026-09-30. Parent: `docs/phase8_r2_plan.md` (P8-R2, `1974420`),
section P8.5; gates in `docs/phase8_registration.md`. Committed before any
reactor-network build, test or numerical result.

Known when written: G0 PASS (`74f53c8`); P8.2 G1 PASS (`df3a6c1`); P8.3 G1
PASS (`ef4e928`). `data/A2NOx.yaml` loads in Cantera 3.2.0 with 201 species,
1589 reactions, elements O, H, C, N, Ar, He and lumped Jet A fuel
`POSF10325`; on load Cantera reports NASA-polynomial discontinuities at the
mid temperature for CH2OCH, C4H5-2, C4H6-2, C2H3CHOCH2, CH3CHCHCHO and
iC4H7-1 (warnings only). No network has been solved and no NOx, CO or η_b
value has been computed. No calibration is authorized by this file.

## 1. Network (per operating point)

Inputs: combustor-inlet air `T3, P3` and mass flow `m_air` (the burner air of
the cycle step that calls the network), fuel mass flow `m_f` (set by the
cycle's thrust match), the four free parameters of section 3.

1. **Pressure.** Every reactor is at `P = P3 (1 − Δp)` with the frozen v6
   `combustor_pressure_loss` Δp = 0.045. Adiabatic walls (v6 heat loss 0).
2. **Inlet streams.** Air is `O2:1, N2:3.76` (molar) at `T3`. Fuel is
   `POSF10325` gas at `T3` with liquid-basis enthalpy
   `h_gas(T3) − 360 kJ/kg` (the protected v7 heat of vaporisation, applied
   once, as in P8.2).
3. **Primary zone (PZ).** Air `α_pz m_air` and all fuel. The PZ is K = 7
   parallel perfectly stirred reactors at the Gauss–Hermite nodes `x_k` with
   probability weights `w_k = ω_k/√π`. PSR k receives air `w_k α_pz m_air`
   and equivalence ratio `φ_k = φ_pz + √2 σ_φ x_k`, so its fuel is
   `φ_k FAR_st w_k α_pz m_air`; `Σ w_k φ_k = φ_pz` holds exactly, so fuel
   is conserved. `FAR_st` is the stoichiometric mass ratio of POSF10325 in
   this air from the mechanism. All `φ_k` must be positive (section 3).
4. **Mixing.** Outlets of the PZ PSRs mix adiabatically at P by mass-weighted
   enthalpy and Y (P8.2 mixing rule, HP inversion tolerance 1e-13).
5. **Quick quench (QQ).** One PSR fed by the mixed PZ gas and air `α_qq m_air`.
6. **Lean zone.** A plug-flow reactor represented by N = 10 equal-volume PSRs
   in series.
7. **Dilution.** Air `α_dil m_air`, `α_dil = 1 − α_pz − α_qq`, mixed
   adiabatically with frozen composition (no reaction).

**Volumes.** Zone volume fractions are fixed: PZ 0.30 (split by `w_k`),
QQ 0.10, lean zone 0.60 (0.06 per PSR). The total volume is
`V = s · V_ref`, where `V_ref` makes the total reference residence time
`m_total/(ρ_mean)` equal 4.0 ms at the AE3 take-off design inlet with
`s = 1`, ρ_mean evaluated at the HP-equilibrium state of the overall
mixture. `V_ref` is computed once per engine at its design point and then
held fixed (hardware), so residence times vary with operating point.

**PSR solution.** Each PSR is a Cantera `IdealGasConstPressureReactor` fed by
mass-flow controllers from reservoirs, exhausting through a pressure
controller, initialised at the HP-equilibrium state of its inlet mixture
(burning branch), advanced with `ReactorNet::advanceToSteadyState` at
rtol 1e-9, atol 1e-20, residual threshold 1e-9 (relative). A PSR that does
not converge, or extinguishes (outlet T within 50 K of its inlet mixture
temperature), is reported as such; it is not silently re-initialised.

**Threads.** The K PZ PSRs of one point are solved on separate `std::thread`s,
each owning its own Cantera `Solution`. Python modules use the static-Cantera
hidden-symbol recipe (`cpp/CMakeLists.txt`).

## 2. Outputs

- Exit `T, P, Y` after dilution; `η_b` (below); EI NOx as NO2 equivalent
  `1000 (Y_NO·M_NO2/M_NO + Y_NO2) m_exit / m_f` g/kg; EI CO, EI UHC.
- **η_b energy basis.** `η_b = 1 − Σ_i m_exit Y_i LHV_i / (m_f LHV_f)`, summed
  over every exit species containing C or H other than CO2 and H2O (CO, H2,
  and all hydrocarbons/oxygenates; N-containing species with C or H count
  too). `LHV_i` is the lower heating value at 298.15 K from this mechanism's
  thermo, with complete products CO2, H2O(g), N2. `LHV_f` is that of
  POSF10325 on the same basis, minus 360 kJ/kg (liquid basis).
- NOx and CO are model outputs, never derived from the ICAO correlation (D4).

## 3. Free parameters and admissible domain

| Parameter | Meaning | Admissible numerical domain |
|---|---|---|
| `φ_pz,design` | PZ φ at the engine's take-off design point; sets the hardware split `α_pz = φ_global,TO/φ_pz,design` | 1.2–2.5 |
| `σ_φ/φ_pz` | unmixedness | 0–0.25 (K = 7 positivity needs < 1/(√2·2.6520) = 0.2666) |
| `α_qq` | QQ air fraction of burner air | 0.20–0.70, with `α_dil ≥ 0` |
| `s` | volume scale | 0.25–4 |

These are numerical domains, **not** cited priors. Before any calibration,
P8.7 registers cited central values and ranges (range rule) and the data
that would constrain each parameter. Until then every result uses the G1
test values below and is not a prediction.

## 4. G1 (write-once `outputs/phase8/p85_g1.json`)

Test values: `φ_pz,design = 1.8`, `σ_φ/φ_pz = 0.10`, `α_qq = 0.45`, `s = 1`,
at the AE3 TAKE-OFF, APPROACH and IDLE combustor inlets of the frozen v6
rows (T3, P3, burner air, fuel flow from `calibration_v6_rows.csv`).

1. **Long-residence limit.** With `α_dil = 0` and `s = 1e4`, the lean-zone
   exit equals HP equilibrium of the total inlet mixture (same mechanism and
   enthalpy basis): |ΔT| ≤ 0.1 K and |ΔY_k| ≤ 1e-5 for every species with
   equilibrium Y_k > 1e-6, at all three points.
2. **Closure.** Element mass and energy (enthalpy flux) closure of the whole
   network and of each mixer, relative error < 1e-10 with the P8.2
   denominators.
3. **σ_φ = 0 limit.** All K PZ PSRs give identical states (relative 1e-12).
4. **Quadrature.** K = 7 vs K = 9 at the test values: report the change in
   exit T, EI NOx, EI CO and η_b (recorded, not a pass/fail threshold).
5. **Mechanism consistency (reported, not gating).** (a) For species shared
   with CRECK (`data/creck_c1c16_full.yaml`), max relative difference of
   h(T) and cp(T) on 300–2500 K; (b) HP-equilibrium T of the AE3 take-off
   mixture with A2 (POSF10325) and with CRECK (registered Dooley 2012
   surrogate) at the same FAR; (c) the polynomial discontinuities above.
   A difference is reported; it is never used to choose a mechanism.
6. **Mechanism spread (reported).** η_b and EI CO at the test values with
   CRECK in place of A2NOx (CRECK has no N chemistry, so no NOx), as a
   spread, not a selection.

A G1 failure is recorded and blocks P8.5 use in P8.7; P8.4 and the data
tracks continue.

## 5. Not in scope here

Coupling the network into the matched-thrust cycle (it replaces the v6
η_b temperature scaling) is registered with P8.7 together with the
calibration data and stopping rule. No D4 claim is made from G1.

## Relevant files

`cpp/catjet_core/reactor_network.{hpp,cpp}`, `cpp/tests/test_reactor_network.cpp`,
`cpp/bindings/catjet_core.cpp` (binding), `cpp/CMakeLists.txt`,
`scripts/phase8/reactor_validation.py` (G1 record), `tests/test_phase8_p85.py`.
