# P8.5-A2 — development probe before G1 (no rule change)

Date: 2026-09-30. Committed before the P8.5 G1 verdict
(`outputs/phase8/p85_g1.json` does not exist). The registered G1 rules and
tolerances (`docs/phase8_p85_registration.md`, P8.5-A1) are **unchanged**.

Known when written, from uncommitted development probes (not G1 records):

- One network run at the AE3 take-off inlet with the G1 test values
  converged in every PSR (primary-zone outlet 1914–2687 K over the seven
  phi nodes, exit 1684.7 K) in about 8 s. Mixer closure was 1e-15, but the
  whole-network closure was energy 7.0e-9 and elements 4.6e-7 (C 3.7e-7,
  H 4.6e-7; O and N 8.8e-9; mass exact). This would fail the registered
  1e-10 closure.
- Diagnosis: Cantera `equilibrate("HP")` conserves C, H, O and N to 1e-16,
  and a Newton `solve_steady` polish of a time-marched PSR leaves the error
  unchanged. Seven reactions of the protected `data/A2NOx.yaml` (the HyChem
  POSF10325 decomposition steps, reactions 0–6) are element-imbalanced by
  2–4e-7 C and H atoms per event (rounded lumped coefficients). In a single
  PSR only C and H drift, and the reactor mass fractions sum to 1 + 4.3e-8.
  The energy error follows from that sum.

Decision: the G1 runs as registered and is scored by its registered rules.
The record adds a **non-gating diagnostic**: the element creation predicted
by integrating each imbalanced reaction's rate over every PSR at its outlet
state, compared with the observed network element error. The diagnostic
cannot turn a failed closure check into a pass. Using a rebalanced copy of
the mechanism, or a registered closure tolerance tied to it, is a
scientific choice left to the user.
