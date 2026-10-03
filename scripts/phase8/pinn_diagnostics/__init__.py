"""Phase 8 Track 4: isolated PINN diagnostics (registration P8-TRACK4-20261002-attempt1).

Diagnostic code only. Nothing here is imported by the production cycle, and
nothing here imports ``simulation.turbine`` (it sets a global float32 default)
or the parked ``phase7-p74-wip`` runner. Every numeric constant is read from
``docs/phase8_track4_registration.json``.

- ``turbine_map``: four-input CPU float64 MLP fitted to the analytic
  polytropic pressure ratio, scored once on a 65x65 grid.
- ``nozzle_verification``: literal Ma et al. Eqs. 22-26 residual with
  independent analytic forcing (method of manufactured solutions), Ma's
  current-loss weights, and exact quasi-1D nozzle references (no network).
- ``run_diagnostics``: write-once runner with resource checks and reports.
"""
