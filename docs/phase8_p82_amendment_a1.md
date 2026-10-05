# P8.2-A1 — numerical pressure-path amendment

Date: 2026-09-29. Commit before the first P8.2 build, test or physics
computation. Known when written: G0 passed; the original P8.2 registration
`2b0fe17` was committed, but no P8.2 executable has been built or run and
no P8.2 result has been observed. The original method inferred the exit
pressure analytically and merely traced it in 50 steps, making the 50→100
convergence check vacuous. This amendment replaces that pressure-path method
with a numerical integration that determines the outlet pressure and work.

At frozen Y, integrate `dT/d(ln P)=eta_poly*R(Y)*T/cp(T,Y)` from the inlet
over N equal steps in log pressure using classical RK4. Obtain cp at every
RK4 substep from the same Cantera phase at that substep's T, P and frozen Y.
For a requested shaft work W, root-find the outlet log-pressure ratio with
the existing C++ Brent solver until
`m*[h(T_in,Y)-h(T_out,Y)]-W=0`. Use 50 steps for the production result and
100 steps only for the G1 convergence check; report both outlet T and P.
The bracket begins at log ratio 0 and expands from -1 by doubling its
magnitude until the work residual changes sign, with a hard lower bound of
-20 and explicit failure if no physical bracket exists. Brent uses
`xtol=1e-12`, `rtol=4*epsilon`; each integration must retain positive,
finite temperature and pressure. A zero-work stage returns the inlet state
unchanged. The constant-cp branch remains the exact v6 analytic formula.

This change affects only the P8.2 numerical integrator. The A1/A2 physical
scope, cooling fractions, energy/element tolerances and all other gate
rules remain as registered. Since no P8.2 result existed, there is no
post-outcome selection. A failure to meet the registered 50→100 G1 threshold
is recorded; the production step count is not tuned after that result.
