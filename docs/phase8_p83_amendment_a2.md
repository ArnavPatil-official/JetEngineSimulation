# P8.3-A2 — nozzle coefficient re-citation and Cv sensitivity

Date: 2026-10-03. User decisions of 2026-10-02. Machine-readable copy:
`docs/phase8_p83_a2_registration.json` (authoritative for the numbers).

Written **after** ladder A4 was scored once (`2dd705f`: 21.24 % held-out,
with the P8.3 priors Cd = 0.96, Cv = 0.95). This amendment is not backdated
and does not rescore A4. The A4 record, `docs/phase8_p83_registration.md`,
`docs/phase8_p84b_registration.md`, `scripts/phase8/trent_p84b.py` and every
A4 output stay exactly as committed. The new prior applies only to models
registered after this commit (first user: A4c, `docs/phase8_p84c_registration.md`).

## Rule

| Coefficient | Applies to | Central | Sensitivity envelope |
|---|---|---|---|
| Cv (velocity coefficient) | core and bypass, each | **0.985** | 0.95–1.00 |
| Cd (discharge coefficient) | core and bypass, each | **0.96** (unchanged) | 0.90–0.99 (unchanged) |

The model definitions are the P8.3 ones and do not change:
`m_dot = Cd A G(p_exit)` and `u_exit = Cv u_ideal(p_exit)`; the kinetic-energy
shortfall stays in gas enthalpy. Cv multiplies **velocity**. Both ranges are
declared engineering envelopes for one-at-a-time sensitivity. They are not
confidence intervals, they are not fitted, and they are not Trent
measurements.

## What each number rests on

| Number | Kind | Basis |
|---|---|---|
| Cv = 0.985 | engineering choice (central reference) | Informed by the level of the corrected gross-thrust coefficient in TP-2171 and the generic pyCycle Cv values below. Not a measured Cv. |
| Cv = 0.95 (low end) | engineering bound | The former P8.3 central (rounded TN-1757 reading, older analog). It also brackets the low-NPR TP-2171 Cfg level (0.958 at NPR 2.5). |
| Cv = 1.00 (high end) | mathematical limit | No velocity loss. The old P8.3 upper end 1.02 is not kept. |
| Cd = 0.96 | engineering central (unchanged) | P8.3 reading of NACA TN-1757. TP-2171 measured 0.965 and 0.957 on a different nozzle (below) are consistent with it. |
| Cd = 0.90–0.99 | engineering bounds (unchanged) | P8.3 envelope from TN-1757 / RP-1235. |

### Primary evidence (pages and code checked 2026-10-03)

1. **NASA TP-2171** (Straight and Cullom, June 1983), *Thrust performance of
   a variable-geometry, nonaxisymmetric, two-dimensional, convergent-divergent
   exhaust nozzle on a turbojet engine at altitude*,
   <https://ntrs.nasa.gov/api/citations/19830018568/downloads/19830018568.pdf>
   (40 PDF pages, sha256 `b47ac6fa…4793`, downloaded 2026-10-03).
   - The test article was a full-scale 2D convergent-divergent nozzle on a
     J85-13 **turbojet** in an altitude facility, at NPR 1.6–14. It is not a
     separate-flow convergent turbofan nozzle and not a Trent measurement.
   - Printed p. 14 (Fig. 16 text): with leakage and bypass-coolant
     corrections, the dry-cruise throat setting gives a peak corrected gross
     thrust coefficient **0.985–0.990 for NPR > 4.0**. Below NPR 4 it falls
     to **0.958 at NPR 2.5**. The maximum-afterburning setting gives 0.967
     at NPR 2.0, 0.996 at 6.5 and 0.987 at 14.0.
   - Printed p. 14: the discharge coefficient is the effective throat area,
     computed from the corrected throat flow, mixed gas temperature and
     measured total pressure, divided by the measured throat area. Above
     NPR 5 it is nearly constant: **0.965 (dry cruise)** and **0.957
     (maximum-afterburning)**.
   - Printed p. 15: the full-scale Cd values are about 2 % below the
     scale-model values, and they fall at NPR below 5.
   - Printed p. 30 (Summary of Results 1, 3 and 5) repeats these values.
   - **Definitions.** The gross thrust coefficient is the measured gross
     thrust divided by the ideal one-dimensional isentropic gross thrust of
     the (corrected) flows, `C_F = F_g / F_i` (eq. (1), printed pp. 8–9).
     That is a *thrust* ratio. Its denominator is not the
     `m_dot u_ideal(p_exit)` used with our Cv, and its corrected flows are
     not our `Cd A G`. **Cfg is not equated with Cd, and no Cv value is
     derived from it.** It serves only as an order-of-magnitude reference
     for the central choice. Our LTO nozzles run at low NPR (AE3 take-off
     core p0/pa = 1.87, `p83_g1.json`). In TP-2171 that is the region where
     Cfg and Cd both fall, which is one reason the low end stays at 0.95.
2. **pyCycle 4.4.0 HBTF example**,
   <https://raw.githubusercontent.com/OpenMDAO/pyCycle/4.4.0/example_cycles/high_bypass_turbofan.py>
   (local unmodified copy, sha256 `feace41f…d9420`, matching
   `envs/pycycle/upstream/SOURCES.json`), lines 329 and 331:
   `core_nozz.Cv = 0.9933`, `byp_nozz.Cv = 0.9939`. Both nozzles are built
   with `Nozzle(nozzType='CV', lossCoef='Cv')`. pyCycle's
   `elements/nozzle.py` (4.4.0) computes
   `Fg = W V_actual Cv Cang CmixCorr + (Ps_actual − Ps_amb) A_actual`, so
   Cv multiplies velocity, as ours does. These are **generic NPSS example
   values (modelling reference), not measurements.**
3. **Older analog context (kept, not primary):** Grey and Wilsted,
   NACA TN-1757 (1948), Figs. 4(c,d) and 8, and Stitt, NASA RP-1235 (1990),
   Figs. 3-2 to 3-4, as cited in the P8.3 registration. They are small
   convergent-nozzle rig data near NPR 2, not full-scale Trent
   measurements.

No held-out target, held-out prediction or A4 row-level result was opened
to choose these numbers. The only A4 facts known are its committed
aggregates.

## Not changed

The P8.3 law, the G1 thresholds and verdict (`p83_g1.json`), the
1e-13 inversion tolerances (P8.3-A1), the TN-1757 digitisation route
(P8.7 component stage) and the rule that digitised measurements may later
constrain dimensionless Cd/Cv relations only after their own registration.
