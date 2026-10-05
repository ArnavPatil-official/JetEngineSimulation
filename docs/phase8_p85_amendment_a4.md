# P8.5-A4 — design at take-off in the G1 script; unit-flow PSR solves

Date: 2026-10-02. Before G1 re-run 2 (`outputs/phase8/p85_g1_rev2.json`).
Records `p85_g1.json` and `p85_g1_rev1.json` stay as they are.

1. **Script conformance.** The registration (section 1) fixes `alpha_pz` and
   `V_ref` at the engine's take-off design point. `reactor_validation.py`
   recomputed the design at every mode; it now computes it once at AE3
   TAKE-OFF and uses it for all three modes. This is a correction to match
   the registration, not a change to it.
2. **Unit-flow PSR solve (numerical).** A PSR's steady state depends only on
   its inlet and residence time; each PSR is now solved at 1 kg/s with
   volume `V/mdot` (exact), so the very unequal Gauss-Hermite reactors are
   equally conditioned.

Development results known before the re-run (AE3, take-off design):
sigma = 0 primary-PSR T spread 2.5e-12 (TO), 4.4e-13 (APP), 7.6e-13 (IDLE);
long-residence exit minus HP equilibrium: TO −1.4e-6 K, APP −2.5 K,
IDLE −0.041 K. Checks and tolerances are unchanged; the closure check still
fails at ~3e-8 because of the protected mechanism (user decision pending).
