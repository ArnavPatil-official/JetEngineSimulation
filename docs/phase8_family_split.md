# D3 cross-family split registration (Track C3)

Date: 2026-10-01. Parent: `docs/phase8_r2_plan.md` (P8.6/D3, G2 as amended by
P8-A1), user directive of 2026-09-29 (≥ 8 held-out families, ≥ 8 calibration
families, stratified by thrust class and seeded, bootstrap clusters =
families), and the user's decision of 2026-10-01 to **admit geared
turbofans** (only 15 direct-drive families were eligible).

Committed before the split script runs. Only identifier/design columns are
read (`scripts/phase8/icao_families.py` reader: manufacturer, engine
identification, engine type, rated thrust, pressure ratio, bypass ratio,
status/superseded markers); no fuel-flow, emission, smoke or nvPM value has
been decoded for any family other than the Trent 1000 rows already in
`data/icao_engine_data.csv`.

## Eligibility (record level, then family)

A databank record is eligible when all hold:
1. `Eng Type` = `TF` (separate-flow turbofan; `MTF` mixed-flow needs a mixer
   element the P8.4 solver does not have);
2. `Current Engine Status` is blank (in production; "Out of production" in any
   capitalisation and "Out of service" are excluded);
3. `Data Superseded` is not `Yes`;
4. pressure ratio, bypass ratio and rated thrust are present and positive.

Families: the `icao_families.family` rule (databank model designation), with
one grouping override already used in the 2026-09-29 audit: every
`Trent7000-*` designation is the single family `Trent 7000`. Direct-drive and
geared (`PW1xxxG`) families are both eligible; the architecture flag (geared
iff the identification matches `^PW1\d{3}G`) is recorded for P8.4b. A family
is eligible if it has at least one eligible record.

## Draw

Thrust class from the family median rated thrust of its eligible records:
S1 < 100 kN, S2 100–200 kN, S3 > 200 kN. The held-out count is 8,
allocated to strata in proportion to their family counts by largest
remainder (ties: S3, then S1, then S2). The Trent 1000 is fixed in the
calibration pool and is not drawable. Within each stratum, eligible
families are sorted by name and the stratum's held-out families are drawn
without replacement with `numpy.random.default_rng(20261001)`, strata in the
order S1, S2, S3, one generator for all strata. Every other eligible family
is in the calibration pool. Abort (no split written) if fewer than 8 families
remain in the calibration pool.

## Use and guards

- Held-out families' target rows are opened only by the registered G2
  scoring step after this file, the script and
  `outputs/phase8/cross_family_split.json` (with its SHA-256) are committed.
- Bootstrap clusters are families (P8-A1, 10,000 replicates,
  `default_rng(20260929)`).
- **Pre-registered sensitivity (reported, not gating):** designations that
  share a programme (LEAP-1A/1B/1C; PW1100G/1200G/1400G and PW1500G/1900G;
  CF34-8/CF34-10; D-36/D-436) are separate families under the rule above, so
  siblings may sit on both sides of the split. G2 scoring also reports the
  result with every held-out family whose sibling is in the calibration pool
  removed.
- The v6 Trent split (`outputs/phase6/split_p61.json`) is unchanged and
  remains a separate check.

## Relevant files

`scripts/phase8/cross_family_split.py`, `outputs/phase8/cross_family_split.json`.
