# P8.3-A1 — isentrope inversion accuracy correction

Date: 2026-09-30. Corrective numerical amendment committed before the P8.3
G1 verdict (`outputs/phase8/p83_g1.json` does not exist yet). The G1
thresholds in `docs/phase8_p83_registration.md` are unchanged.

Known when written: P8.2 G1 passed (`df3a6c1`). The first build of the P8.3
Catch2 contract tests (development, not G1) passed 4 of 5 cases. The
unchoked v6-limit case failed: on the constant-cp fixture at
T0 = 1100 K, P0 = 400 kPa, p_ambient = 300 kPa, an area set from
`mass_flux(p_ambient)` returned mass flow 50.0000000138 instead of 50
(relative 2.76e-10 > 1e-12), and the force check at 1e-10 failed with it.
Cause: each isentrope point calls Cantera `setState_SP(s0, p)` at its default
relative tolerance 1e-9, starting from whatever temperature the phase last
held, so the same `(s0, p)` gives states that differ at ~1e-10.

Rule: every isentrope state in the P8.3 nozzle uses
`setState_SP(s0, p, 1e-13)`, matching the registered 1e-13 for the exit
HP inversion. No model, coefficient, area or tolerance of the G1 checks
changes. If G1 still fails under this rule, the failure is recorded and
P8.4 nozzle matching does not start.

Not changed: the P8.2 compressor's ideal-outlet `setState_SP` keeps its
default tolerance. P8.2 G1 passed with it, and changing it after that
verdict would alter a gated result; its ~1e-9 precision is noted in the
status file for P8.4.
