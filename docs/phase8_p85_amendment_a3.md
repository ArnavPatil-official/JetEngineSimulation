# P8.5-A3 — closure root cause; no cloning; temperature-based PSR

Date: 2026-10-02. Prospective numerical amendment before any P8.5 G1 re-run.
The failed G1 record (`outputs/phase8/p85_g1.json`, `722a0b6`) stays as is.

## What the development diagnosis found (`scripts/phase8/p85_closure_diagnosis.py` and probes)

- A single A2NOx PSR built the registered way, with **cloned** Solutions
  (`Reservoir`/`Reactor` `clone=True`; C++ `newReservoir(sol, true)`,
  `newReactor4(..., true)`), gives C +3.8e-7, H +4.7e-7, sum(Y) - 1 = +4.3e-8.
  The kinetic source terms at that state create exactly that mass and carbon
  (V·Σ ω_k W_k = 4.3076e-8 kg/s vs sum(Y) - 1 = 4.3075e-8; C creation
  3.57e-8 kg/s vs observed 3.57e-8 kg/s).
- The cloned phase's lumped HyChem reaction 0 has a mass imbalance of
  **+6.09e-5 kg/kmol**; the same reaction loaded from `data/A2NOx.yaml`
  has **−4.01e-6 kg/kmol** (−3e-7 C, −4e-7 H atoms). Cloning re-serialises
  the mechanism and rounds the non-integer stoichiometric coefficients,
  changing the imbalance by 15× and its sign.
- With separately loaded Solutions (`clone=False`), the PSR error is
  C −2.7e-8, H −1.8e-8, sum(Y) − 1 = −2.8e-9: exactly the protected
  mechanism's own imbalance (−3e-7 C atoms per C11H22 event = −2.7e-8).
- The G1 record's non-gating diagnostic used the file's stoichiometry, which
  is why it did not match the cloned network; P8.5-A2's attribution to the
  mechanism imbalance was right in kind; its rounding mechanism was unknown.
- `ConstPressureReactor` (enthalpy state, P8.5-A1) fails at this state
  (CVODES error −4) at rtol 1e-9, 1e-11 and 1e-13; `IdealGasConstPressureReactor`
  (temperature state) converges. Tighter rtol (1e-11, 1e-13) fails CVODES's
  error test with either reactor.

## Rule changes (numerical only)

1. Every Reservoir and Reactor gets its own `Solution` created from the
   mechanism file and is constructed with `clone=false`. Per thread, the
   network keeps one Solution each for inlet, reactor and exhaust.
2. PSRs use `IdealGasConstPressureReactor`, replacing P8.5-A1 item 2.
   Steady-state loop, tolerances and every G1 check are unchanged.

Expected: the closure check still fails at about 3e-8 because of the
protected mechanism's genuine imbalance; changing the 1e-10 rule or using a
rebalanced mechanism copy is the user's decision. The re-run writes
`outputs/phase8/p85_g1_rev1.json`.
