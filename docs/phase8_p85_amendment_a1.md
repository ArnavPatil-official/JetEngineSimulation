# P8.5-A1 — PSR solver details from the Cantera 3.2 C++ API

Date: 2026-09-30. Prospective numerical amendment, committed before any
reactor-network build or result. Known when written: only the P8.5
registration (`bb5b61b`) and the Cantera 3.2 headers; no network solved.

1. **Steady-state routine.** `advance_to_steady_state` exists only in
   Cantera's Python layer; the C++ `ReactorNet` has `step()` and
   `getState()`. The network ports that Python loop to C++ unchanged:
   repeated `step()`; residual
   `‖(y − y_prev)/(max|y| seen + atol)‖₂ / √n`; stop below the threshold;
   fail after 10,000 steps. Cantera requires the threshold to exceed the
   integrator rtol, so the registered "threshold 1e-9 with rtol 1e-9" is
   replaced by Cantera's own default **threshold = 10 · rtol = 1e-8**, with
   rtol 1e-9, atol 1e-20 unchanged.
2. **Reactor type.** Use `ConstPressureReactor` (state variable: total
   enthalpy) instead of `IdealGasConstPressureReactor` (state variable:
   temperature). Each PSR starts at the HP-equilibrium state of its inlet,
   which has the inlet enthalpy, and for an adiabatic constant-pressure PSR
   `dH/dt = ṁ_in h_in − ṁ_out h` then keeps `h = h_in` to round-off. The
   temperature-based form holds it only to the integrator rtol, which
   cannot reliably meet the registered 1e-10 energy closure. Thermodynamics
   are identical (ideal-gas phase in both).
3. **Sequential solution.** The network has no recirculation, so each zone
   is solved in flow order as one PSR fed by a reservoir holding the
   adiabatic mixture of its inflows (upstream outlet + injected air). For
   a perfectly stirred reactor this is identical to feeding the streams
   separately: the steady state depends only on the summed species and
   enthalpy fluxes. The K primary PSRs are independent and run on separate
   threads.

All other P8.5 rules, parameters, outputs and G1 tolerances are unchanged.
