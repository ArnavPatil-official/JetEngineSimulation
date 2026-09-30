# P8.2 prospective numerical registration (A1 and A2)

Date: 2026-09-29. Parent plan: `docs/plan.md`; ablation order:
`docs/phase8_registration.md`, amendment P8-A2 (`421022e`). This file is
committed before the first P8.2 build, test, calibration or G1 computation.
Known when written: G0 PASS (`74f53c8`), the AE3 frozen v6 parameters, and
NASA CR-168189 design cooling fractions and rig total cooling flows. No
P8.2 result has been seen. The existing G0 C++ class and Python v6 code
remain callable and unchanged by the new physics path.

## A1: state, liquid fuel and dilution

- `GasState` is `(T [K], P [Pa], Y [mechanism-order mass fractions])`. A
  station has a mass-flow rate alongside its state. One Cantera CRECK phase
  supplies `h(T,Y)`, `s(T,P,Y)`, `cp`, `R`, element fractions and density.
  Air at new stations is the combustor's exact `O2:1, N2:3.76` molar mixture;
  no implicit change of Y may occur at a station.
- Compressor pressure ratio and eta_c retain the frozen v6 configuration.
  Its ideal outlet is at `(s_in,p_out)`; actual outlet enthalpy is
  `h_in+(h_ideal-h_in)/eta_c`, solved at `p_out` with frozen inlet Y.
  The fan remains at its v6 analytic setting until P8.4 map matching; its
  inlet/outlet GasStates carry the same air Y for downstream bookkeeping.
- Keep the v6 HP-equilibrium products and eta_b temperature-scaling
  convention, with no mechanism change. With product mass
  `m_prod=m_burn_air+m_fuel`, set liquid-basis product enthalpy to
  `h_product,gas - (m_fuel/m_prod)*360000 J/kg`. Invert `h(T,Y)` at the
  combustor outlet P to obtain T. The 0.360 MJ/kg is the protected
  `data/fuel_properties_v7.yaml` value, a common n-dodecane approximation.
  Compute and report the explicit heat rejection implied by eta_b and the
  existing heat-loss hook; do not describe it as energy nonclosure or apply
  vaporisation twice.
- At equal static pressure, mix streams with
  `Y_out=sum(m_i Y_i)/sum(m_i)` and
  `h_out=sum(m_i h_i)/sum(m_i)`, then solve `T_out` at `(h_out,P,Y_out)`.
  Cooling/dilution air may throttle isenthalpically to this pressure.
  A1 uses the existing beta split: `m_burn_air=beta*m_core`,
  `m_dilution=(1-beta)*m_core`. It then uses the v6 analytic turbine and
  nozzle, with the new mixed-state cp/R/gamma at turbine inlet. This is the
  A1 ladder branch.

## A2: turbine and cooling

- Fixed central HP cooling fractions of core inlet air are `f_NGV=0.0641`
  before the HP rotor and `f_rotor=0.0275` after that rotor. Source: Leach,
  NASA CR-168189, section 3.2.3 and Figures 3.2.3-2/-4,
  [NTRS 19850021643](https://ntrs.nasa.gov/citations/19850021643).
  Table 5.3.1-I reports total design cooling 14.56% versus rig 12.36%; it
  includes other cooling/leakage flows and is **not** the sum of these two
  fractions. The sensitivity range for each implemented fraction is zero
  (the exact v6/no-cooling limit) through its cited design value. This is a
  physical-limit/design range, not a measured confidence interval. Use no
  additional fitted cooling parameters or Trent-specific rig fractions.
- For A2, split core air as `m_NGV=f_NGV*m_core`,
  `m_rotor=f_rotor*m_core`, `m_burn_air=(beta-f_NGV-f_rotor)*m_core`, and
  `m_dilution=(1-beta)*m_core`; require `beta>f_NGV+f_rotor` and all streams
  nonnegative. This reduces exactly to A1 when both cooling fractions are
  zero. Mix dilution after the burner, NGV air before HP expansion, and rotor
  air after HP expansion, at common pressure by enthalpy and Y. Do not
  silently count the same air in both burner and coolant.
- Supply nonnegative HP, IP and LP shaft-work demands separately. For the
  present v6-component ladder, assign compressor demand to HP, zero demand
  to IP, and fan demand to LP. The zero-work IP stage is the two-shaft limit;
  the three-shaft allocation is set by P8.4 maps, not an invented split.
  Each stage has eta_poly=0.9 from frozen v6 fixed settings and no stage
  pressure loss beyond the modeled expansion. IP/LP cooling defaults zero.
- Within a stage, Y is frozen. The ideal-gas polytropic relation is
  `dh=eta_poly*R(Y)*T*d(ln P)`. For target work `W`, find the outlet T from
  `h_out=h_in-W/m_in` at inlet pressure with Cantera; infer outlet pressure
  from `ln(P_out/P_in)=[s(T_out,P_in,Y)-s(T_in,P_in,Y)]/(eta_poly*R(Y))`.
  Trace the path with 50 equal steps in `ln P`, using Cantera entropy
  inversion at each step. Repeat with 100 steps for convergence. No map
  efficiencies or geometry scale factors are introduced in P8.2.
- The constant-cp test mode of the same stage API uses supplied `cp,R` and
  the exact limit `T_out=T_in-W/(m*cp)` and
  `P_out=P_in*(T_out/T_in)^(cp/(eta_poly*R))`, identical to the v6 analytic
  turbine. Invalid work, pressure, composition or cooling split returns an
  explicit cycle-does-not-close reason.

## G1 and record

Before scoring any calibration or held-out row, verify:

1. Constant-cp no-cooling stage at frozen v6 input reproduces its T5 and p5
   to relative 1e-10. The zero-cooling A2 path and A1 path agree at the
   turbine interface to the same tolerance.
2. For every stream mix and stage, mass closure, element-mass closure and
   energy closure (including stated shaft work or burner heat rejection)
   have relative error below 1e-10. The denominator is the sum of absolute
   inlet enthalpy fluxes plus absolute work/heat terms, floored at 1 J/s;
   element denominator is total incoming element mass flow, floored at
   1e-12 kg/s. These are numerical closure checks, not validation against
   experiment.
3. At the frozen AE3 take-off state and a second lower-work state, doubling
   polytropic pressure steps 50→100 changes T and P by less than 1e-8
   relative. Record both values, not just pass/fail.
4. The pre-existing G0 component and 180-row results remain unchanged on
   the v6 backend. Check the protected hashes and full pytest suite.

The write-once verdict is `outputs/phase8/p82_g1.json`; include source
commit, compiler/Cantera versions, fuel/config hashes, all numerical errors
and failed cases. Do not run A1/A2 ladder calibration or held-out scoring
until this G1 record has been committed. A gate failure is recorded and
blocks A3, but independent tracks continue.
