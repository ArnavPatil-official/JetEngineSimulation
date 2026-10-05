# P8-A3 — ablation ladder: A3 merged into A4; A4 has no knobs

Date: 2026-10-01. User decisions of 2026-10-01, recorded before any A3 or A4
ladder computation. Amends P8-A2 (`docs/phase8_registration.md`); P8-R2
(`docs/phase8_r2_plan.md`) governs where they differed.

Known when written: ladder A1 scored once (1.835 %, `99a0578`); A2
calibration attempt 1 stopped on battery (no parameters); P8.3 G1 and P8.4
G1 passed; no A2, A3 or A4 parameters or scores exist.

1. **A3 is not separately scorable.** Fixed-area choking nozzles with the v6
   imposed core/bypass flows do not form a closed cycle (P8.3 registration:
   capacity is reported, not matched, until P8.4). A3's nozzles therefore
   enter the ladder together with P8.4 matching, at A4. The ladder reports
   A3 as "merged into A4" with this reason; no A3 calibration or score is run.
2. **A4 has no knobs.** P8-R2 retires W_ref, a_thrust, k_pi and k_mdot at
   P8.4 and forbids reintroducing them as free fit parameters. A4 is the
   P8.4b Trent model with its registered design-point procedure and cited
   inputs only; it is scored once on the 87 Trent held-out rows with the
   frozen P7.2 held-out tables. Calibration of identifiable design-point
   scale variables is P8.7 stage 2 under its own registration.
3. Unchanged: A1 and A2 refit exactly the four knobs with the frozen v6
   procedure and are scored once each; A2 is still to be calibrated.
