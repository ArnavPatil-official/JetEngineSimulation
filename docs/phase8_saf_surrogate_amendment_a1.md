# P8-S amendment A1: four-size paired learning curve

Registered date: 2026-10-04. Amendment ID: `P8-S-20261004-A1`.
The corresponding JSON is authoritative. This is a prospective additive
amendment to `P8-S-20261004`, authorized by the user's 2026-10-04 deadline
change. The working deadline is Wednesday, 2026-10-07; work unfinished at
Tuesday, 2026-10-06 23:59 America/New_York remains explicitly in progress.

## Exact change

Use **N = 64, 256, 1024, 4096**, with the original **three seeds 42, 43, 44**
for each of **M-phys and M-data** (`Mphys` and `Mdata` internally). Each of the
**24** arm/size/seed fits starts fresh. Remove the intermediate sizes 128,
512 and 2048; do not execute their 18 fits. The JSON lists every retained fit.

The parent JSON's seven-size schedule and references to 42 fits/model hashes
are superseded only where listed in A1's exact JSON-pointer overrides.
Model-dependent artifact coverage is reduced from 294 to 168 paths: seven
registered artifacts for each of 24 fits. Its 44 other required outputs stay
unchanged, giving **212 exact required paths**. The A1 override supplies the
complete path set. Seal all 24 model hashes, validation tables, selections and
locked prediction arrays before any sole-score target opening.

## Preserved registration

The parent JSON and Markdown remain byte-for-byte historical records. All
scientific bars, model definitions, loss weights, optimizer settings, 2,000
epochs, initialization, paired batches, precision, fixed seeds, split rules,
label budgets, locked-test/sole-score/freeze rules and failure disclosures are
unchanged. Validation may select only among the four retained sizes; the
predeclared failed diagnostic fallback remains N=4096.

N still counts attempted simulator calls total. The complete 4,096-row nested
TRAIN manifest, 1,024 validation, 2,048 ordinary test, 4,096 ranking and 68 named
central-property calls remain registered. There are no replacement samples or
extra labels. Nozzle property coverage remains every 4,096 TRAIN and 68 named
case ID, including repeated coefficients and failed/missing rows. The timing
protocol, startup cost/break-even accounting, core provenance and AC/idle/lease
requirements remain unchanged. Product scope, other registrations and the
existing deferrals are unchanged.

## Identity and execution requirements

Commit this amendment and reviewed amendment-aware code **before** any P8-S
generation, training, numerical checks or target access. Verify the parent
JSON and Markdown hashes recorded in A1. Apply only the exact before/after
JSON-pointer overrides; reject any parent-value mismatch. Hash the full
resolved contract using the canonical JSON encoding specified in A1.

Keep `registration_id=P8-S-20261004` and the existing parent-JSON meaning of
`registration_sha256` and `source_registration_sha256`. Add and validate
`amendment_id`, the actual committed `amendment_sha256`, and
`effective_registration_sha256`. Freeze **all four** paths, byte hashes, Git
blobs and modes in the registration set:

- `docs/phase8_saf_surrogate_registration.json`
- `docs/phase8_saf_surrogate_registration.md`
- `docs/phase8_saf_surrogate_amendment_a1.json`
- `docs/phase8_saf_surrogate_amendment_a1.md`

The shared source-extension manifest must explicitly enumerate both new
amendment files. Producer manifests/row provenance, retained models and score
receipts, nozzle property-source consumers and screening deployment/verification
consumers must bind and validate the full identity set. Base-hash-only checks,
legacy seven-size loaders or legacy 42-fit completion checks cannot authorize
training or acceptance. Missing/dirty/uncommitted amendment bytes or mismatched
contract hashes block work. Preserve the original terminal/G0/AC/lease/source
guards and sealed-target read restrictions.

## Static review and rollback

While the benchmark is active, validation is limited to JSON parsing, parent
hash checks, the exact override whitelist, full resolved-contract hashing,
24 unique fit identities and 212 unique artifact paths, and equality of every
untouched parent JSON leaf. No simulator, MLX, pytest or scientific imports.

Before arming, rollback is a revert of this additive commit. After arming,
retain all write-once evidence and actual failure/incomplete states. No
historical registration edits, history rewrite, output overwrite or push.
