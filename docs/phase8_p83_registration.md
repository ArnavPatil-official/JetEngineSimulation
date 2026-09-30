# P8.3 prospective registration: fixed-area convergent nozzles

Date: 2026-09-29. Parent: `docs/plan.md` and P8.0 G1 in
`docs/phase8_registration.md`. This registration is committed before the
first P8.3 build or numerical result. G0 is passed; P8.2 G1 is pending at
registration. P8.3 test execution and full-cycle integration wait for P8.2
G1. The frozen v6 nozzle remains callable for the A0–A2 ladder branches.

## Physical and numerical contract

- The core and bypass have **separate fixed geometric exit areas**, separate
  stagnation GasStates `(T,P,Y)`, and one common ambient pressure. Nozzle Y
  is frozen along each isentrope; `h`, `s`, `rho` and sound speed come from
  Cantera. The core and bypass flows are not mixed.
- At trial exit pressure `p`, ideal velocity is
  `u(p)=sqrt(2[h0-h(s0,p,Y)])`, and ideal mass flux is `G(p)=rho(s0,p,Y)u(p)`.
  Locate the maximum of `G` through the equivalent sonic root `u²=a²`,
  bracketed by pressure halving and solved with the unchanged SciPy-compatible
  Brent implementation. Check that mass flux on both sides is lower. This
  determines the critical pressure from the thermodynamic model rather than
  a fixed gamma. The constant-cp analytic pressure ratio is a G1 limit.
- If `p_ambient >= p*`, set `p_exit=p_ambient`; otherwise set `p_exit=p*`
  and include pressure thrust. In either regime use
  `m_dot=Cd*A*G(p_exit)`, `u_exit=Cv*u(p_exit)`, and
  `F=m_dot*u_exit+(p_exit-p_ambient)*A`. A zero or reverse pressure head
  produces zero forward flow. Cd and Cv are positive dimensionless
  coefficients. This form makes both mass flow and force continuous at p*.
- The P8.3 component API reports nozzle mass-flow **capacity** and pressure
  thrust. A v6 upstream flow imposed independently need not equal that
  capacity. P8.4's shaft/map/nozzle matching solves that residual; no
  upstream mass flow is silently replaced in the v6 path.
- The unchoked v6 limit is tested with `Cd=Cv=1` and an area set once from
  the v6 effective area at that *test state* (`m_dot/G(p_ambient)`). This
  tests the equation limit, not a claim that the v6 cycle had fixed area.

## Fixed coefficient prior and data boundary

Until three-repeat data have passed `scripts/phase8/wpd_import.py` QA, use
`Cd=0.96` and `Cv=0.95` for both nozzles as **fixed analog priors**. The
sensitivity ranges are `Cd=0.90–0.99` and `Cv=0.90–1.02`. These are rounded,
deliberately broad readings of small and moderate cone-angle measurements
around nozzle pressure ratio 2 in Grey and Wilsted, [NACA TN-1757](https://ntrs.nasa.gov/citations/19930082415),
Figures 4(c,d) and 8 (1948), supported qualitatively by Stitt,
[NASA RP-1235](https://ntrs.nasa.gov/citations/19900011721), Figures 3-2 to
3-4 (1990). They are engineering sensitivity bounds, **not** confidence
intervals or Trent-specific measured coefficients. Figure 8 defines Cv on
outlet velocity; this one-dimensional model uses it as the effective axial
velocity factor and has no extra convergence-angle parameter. If later
digitised measurements show this convention is incompatible, amend the
definition prospectively before scoring those rows.

The digitised TN-1757 curves are a *calibration/validation source only after*
their source split and three-repeat QA are registered. They may constrain
dimensionless Cd/Cv relations under the P8.7 stopping rule; these fixed
priors are not fitted to Trent fuel flow. Core and bypass geometric areas
are frozen from a declared design-point procedure before any A3 full-cycle
score; the P8.3 component G1 tests use declared synthetic areas only.

## G1 and outputs

Use the P8.0 G1 thresholds unchanged:

1. An exactly constant-cp Cantera ideal-gas fixture reproduces
   `p*/p0=[2/(gamma+1)]^[gamma/(gamma-1)]` to relative `1e-10`.
2. Across `p_ambient=p*(1±1e-9)`, mass flow and force jumps are below
   `1e-8` relative, including the pressure-thrust term.
3. With `Cd=Cv=1`, the unchoked fixed-area test state reproduces the v6
   fully expanded force to relative `1e-10`.
4. Separate core/bypass cases have distinct areas and states. Re-run G0
   parity, protected hashes and the full pytest suite after integration.

The write-once verdict is `outputs/phase8/p83_g1.json` with the source
commit, mechanism and fixture hashes, all numerical errors and any failed
case. Do not begin A3 recalibration or held-out scoring before G1 passes
and its verdict is committed. A failure closes dependent P8.4 work; the
independent data and infrastructure tracks continue.
