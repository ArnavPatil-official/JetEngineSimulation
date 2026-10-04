# P8-NOZZLE-ODE-A1 — blend-product-conditioned smooth nozzle diagnostic

Prospective registration dated 2026-10-04, based on `59eab6c2ac52865a470884e55f2c7f3e0539631a`.
Machine-readable protocol: `docs/phase8_nozzle_ode_registration.json`.
The user authorized additive implementation and diagnostic execution after the
registration and reviewed code are committed and every declared gate passes.
The active chain remains unchanged. There is no production integration or
empirical claim.

The proposed network predicts a smooth quasi-1D nozzle profile from **x, NPR,
gamma and R**. Its four outputs are density, velocity, temperature and pressure.
An otherwise identical control receives the same interior and boundary labels
with differential/EOS residual weight zero. The study asks whether the added
residual improves an unseen-condition exact-solution comparison. It does not
validate a real nozzle or restore the retired nozzle network.

## Product-property source and interpretation

The dependency is P8-S-20261004, in
`docs/phase8_saf_surrogate_registration.json`. Consume its future
`outputs/phase8/saf_surrogate/attempt_001/teacher_rows.csv` and
`teacher_species.npz`, containing **TRAIN4096 only**, using **C++ teacher** `gamma4`, `R4_J_kg_K` and
`cp4_J_kg_K`, with the full CRECK product mass fractions and species ordering
retained as provenance. These correspond to `CppEngine.run_at_thrust()`'s
`combustor.gamma_out`, `R_out`, `cp_out` and `Y_out`. The reduced protected
`solve_task()` row does not contain these properties. No property predicted by
the P8-S student supplies an exact nozzle input. Separately consume its
`named_central_properties.csv`, the input-only central property table containing
exactly 17 fuels × four modes (68 fixed `fuel|op` IDs). Its whitelist contains
input identity/status, cp/R/gamma and provenance, never source ff/T4 or sealed
named species. P8-S validation, test, ranking and sealed named targets are
forbidden to this consumer.

The source is the selected frozen Phase 7 v6 calibration, fixed registered
64 engine draws and single-zone beta=1 full-equilibrium C++ engine. Mixture
fractions, source draw/design IDs and all producer identities remain
attached to each condition. Source fuel-flow/emission targets are not nozzle
training labels. Fix all `train_000000`–`train_004095` rows (prefix indices
0–4095, 64 total per draw) and all 68 named IDs before reading coefficients.
Score every row independently, retaining repeated coefficients. There is no
value-based subset or unrestricted design×64 crossproduct. Missing or unconverged source
rows block a claim covering the full registered blend range.
Require finite coherent cp/R/gamma, aligned nonnegative normalized species mass
TRAIN species fractions and valid mass-simplex identities. The named alternate
`JetA_dooley2010` remains an included context case with `in_product_API=false`
and null four-component fractions, verified against its separate frozen fuel
provenance; it is never relabelled as the production Jet A. Named sealed species
are not opened. Malformed evidence blocks before labels. Exact bars are in JSON.

**gamma and R are frozen at the simulator's burner-outlet source state.**
Here they define a calorically-perfect diagnostic gas; they are not asserted
constant along a real product-mixture expansion. The fixed study stagnation
temperature below is separate from the source temperature. Species chemistry,
variable cp, cooling, reaction and external expansion are outside scope. Fuel
fractions are provenance rather than extra network features: two mixtures with
identical gamma/R represent the same gas in these equations.

## Fixed gas box, geometry and smooth regimes

The prospective box is gamma **1.20–1.40**, R **260–330 J/(kg K)**, with
`cp=gamma*R/(gamma-1)`. These are deliberately broad diagnostic bounds chosen
before reading blend-product values, not measured mixture bounds. If a source
pair falls outside them, record `BLOCKED_SCOPE`; do not clip it or widen the
box after labels or scores.

Use x from -1 to 1 m, throat at zero, `A=1+0.5*x^2` m², and fixed
`p0=100000 Pa`, `T0=1000 K`. Thus inlet and exit areas are both 1.5 times the
throat area. Unit length/area are diagnostic scales rather than engine geometry.
The existing exact oracle supports this area law. Its area-Mach construction
and invariants follow the public [NASA area-ratio relations](https://www.grc.nasa.gov/www/k-12/BGP/astar.html).

Define NPR as `p0/p_back`. Register two disconnected intervals:

| Regime | NPR | Branch and exit condition |
|---|---|---|
| Smooth subcritical | 1.02–1.08 | Entire profile subsonic; exit static pressure equals back pressure. |
| Smooth choked | 8–12 | Sonic throat, subsonic convergent section, supersonic divergent section; exit pressure follows area ratio, while lower back pressure permits external expansion outside this study. |

For each gamma, invert `Ae/At=1.5` on the subsonic and supersonic branches to
obtain `M_e^-` and `M_e^+`. Before labels, prove/check across the entire gamma
interval that the subcritical upper bound is below
`[1+(gamma-1)*(M_e^-)^2/2]^(gamma/(gamma-1))`, and the choked lower bound is
at least the analogous expression using `M_e^+`. Require endpoint/extremum
reasoning plus a prospective 1001-point gamma check. This check has not run.
Failure blocks the study; it does not select replacement bounds.

For subcritical cases, derive exit Mach directly from NPR and call the existing
`QuasiOneD.smooth_subsonic(M_in)`, with `M_in=M_exit` because endpoint areas
match. For choked cases call `choked_isentropic()`. The internal choked profile
is independent of NPR over the admitted high-ratio interval. **Do not impose
`p_exit=p_back` there.** The exit branch's dependence on area ratio follows the
[NASA nozzle derivation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/nozzle-design/).

Every other NPR is refused. This includes the choking transition, internal
normal shocks, shock at the exit and overexpanded exit cases. There is no claim
of interpolation through the gap or of external-jet prediction.

## Original oracle remains visible

Reuse the exact functions in
`scripts/phase8/pinn_diagnostics/nozzle_verification.py`, without changing that
file or `docs/phase8_track4_registration.json`. A separate additive wrapper
may select the NPR-consistent subcritical inlet Mach and configure dimensions.
Root tolerance remains `1e-13`; exact mass/enthalpy/total-pressure/area-roundtrip
checks must pass at `1e-10` before the oracle supplies any labels.
Before training, these checks use old independent oracle cases and training
conditions only. Test-profile invariants are computed within the sealed final
scoring pass after all checkpoints are frozen.

Require hash-verified original Track 4 **rung 2** (`2_choked_isentropic`) and
**rung 3** (`3_shock_oracle`) PASS evidence. Keep the old rational normal-shock
check and its evidence fields in the quantitative record. It is a conservation oracle, not a
trained shock-resolving network. Excluding shock cases from this new smooth
claim neither repairs nor conceals the separately registered shock work.

## Scaling and governing residuals

Use fixed `R_ref=287 J/(kg K)`, `rho_ref=p0/(R_ref*T0)` and
`u_ref=sqrt(R_ref*T0)`, not per-case output scales. Let xi=x/L_ref,
`r=R/R_ref`, `a=1+0.5*xi²`, and `c=gamma*r/(gamma-1)`. Independent positive
network outputs are rho_hat, u_hat, T_hat and p_hat. The normalized residuals are

```text
mass:     d(rho_hat*u_hat*a)/dxi
momentum: rho_hat*u_hat*d(u_hat)/dxi + d(p_hat)/dxi
energy:   c*d(T_hat)/dxi + u_hat*d(u_hat)/dxi
EOS:      p_hat-r*rho_hat*T_hat
```

Their dimensional scales are respectively rho_ref*u_ref*A_throat/L_ref,
p0/L_ref, R_ref*T0/L_ref and p0. Differentiate x only while holding the three
conditioning inputs fixed. Apply input chain factors. Do not divide by
`M²-1` at the throat; these primitive residuals remain finite there. This is
inviscid quasi-1D conservation, distinct from Ma's 2D viscous/RANS verifier.

## Matched training and disjoint evaluation

Both arms use a 4→64→64→64→4 tanh network, independent
`softplus(z)+1e-12` outputs, CPU float64 and one thread. There is no hardwired
EOS, branch, exact skip or analytic correction. For seeds 20261004, 20261014
and 20261024, copy the same initial state into each arm pair.

Use 64 scrambled Sobol condition tuples per regime for training. Each receives
eight fixed interior x labels plus identical primitive boundary/throat labels
at x=-1,0,1 in both arms. Physics collocation is the 65-point inclusive x grid
at training conditions only. Training data and boundary MSE weights are 1;
residual weight is 1 for physics-on and 0 for data-only. Residual components
have equal weights, without adaptive balancing or weight search. “Data-only”
therefore includes the same supervised boundary labels and no differential or
EOS penalty; this isolates the effect of the added residual.

Fix 1500 full-batch Adam steps at lr=0.001 followed by LBFGS with max_iter=300,
max_eval=375, strong-Wolfe search and the precise tolerances in the JSON. Keep
the final optimizer state. Budgets are the same optimizer caps, not a claim
of equal arithmetic cost or runtime.

Validation uses a separate seed and 32 conditions per regime, solely for final
reporting. Synthetic test uses another seed and 64 conditions per regime plus
fixed gamma/R corners at NPR endpoints/midpoints. Product test uses all
TRAIN4096 and named68 rows at NPR 1.02/1.05/1.08 and 8/10/12, retaining IDs and
repeated coefficients.
All evaluation profiles use 161 x points including endpoints and throat.
Split whole condition tuples; never distribute x points of one condition
across splits or use test conditions for physics collocation. Freeze/check
input identities before labels and reject overlap rather than adjusting samples.

Freeze all six final checkpoints and optimizer records before computing or
opening any exact test field. Score all once in one synchronous pass. No best
seed, extra iterations, replacement checkpoint or retry follows a test score.

## Prospective numerical bars

Publish each field, condition, regime, panel and seed; pooling cannot hide a
failure. Both synthetic and product panels must pass separately. For every
physics-on seed require:

- Profile maximum relative error ≤0.5% and RMS relative error ≤0.1% for each
  of rho/u/T/p, using positive exact pointwise denominators.
- Mass flow, total enthalpy, recovered total pressure and subcritical exit
  pressure errors ≤0.5%; maximum normalized EOS error ≤0.001.
- Each residual RMS ≤0.001 and maximum absolute residual ≤0.01.
- Positive finite fields; correct subsonic/supersonic branches; throat Mach
  error ≤0.005. Check the choked branches outside |xi|<0.05.
- Choked prediction spread over NPR 8/10/12 ≤0.1% of the exact field at
  corresponding x, for every product property pair.

Profile maxima/RMS and residual maxima/RMS apply separately to each condition
and component over its 161 x locations. Invariant bars apply to each condition's
worst point; reporting an average across conditions cannot pass a failed case.

Demonstrated physics benefit additionally requires, on each panel/regime,
median paired primitive-RMSE ratio physics-on/data-only ≤0.8 and improvement
for at least two of three seed pairs. Zero control error cannot establish an
improvement. A failed/missing control is retained and cannot be replaced.
Accuracy pass without this comparison supports no incremental-residual claim.
These bars are prospective diagnostic choices, not experimental uncertainty.

## Shared post-chain ownership and evidence

Implementation is authorized in isolation; numerical work and tests wait until
the active benchmark ends and all execution gates pass. Before a later launch,
require the current main AC chain's validated terminal records and released
ownership, AC power, fail-closed idle/process checks, nice≥15 and committed
reviewed code. Freeze a reviewed append-only manifest of P8-S product rows,
species, registrations, oracle/implementation, binary and all case identities
before any nozzle labels/training. Unknown input or identity is a blocker.

Use the separately registered `scripts/phase8/scientific_workflow_gate.py`
and `docs/phase8_screening_operations_registration.json`. The shared helper
validates `outputs/phase8/screening_operations/main_dependency.json`, exported
by the original strict idle validator before additive source/test integration.
It verifies raw original completion/owner-release evidence, original source
blobs/modes/bytes and built-module hashes. Only exact new paths/hashes in its
committed source-extension manifest may extend the old scientific roots; there
is no wildcard exception or reinterpretation of old evidence.

Call `prepare_context(..., expected_consumer_identity=identity, require_g0=True)`,
`Context.require_idle_ac()` and `Context.acquire_run(...)`. Use the shared
`outputs/phase8/screening_operations/owner.lease.json`, check
`Run.assert_current()` before stages and at exit, and finalize through
`Run.release(terminal)`. The helper exclusively writes terminal evidence and
downgrades completion on drift or missing proof. Every existing lease blocks
idle authorization. Require the reviewed fresh G0 receipt with raw table/verdict
hashes and actual selected core identity. No names-based waiter, log phrase or
dead PID establishes completion or permission to start.

Future outputs are write-once under
`outputs/phase8/nozzle_ode/attempt_001/`, including input/split/source-coverage
manifests, oracle gate, configuration/environment, all six checkpoints/logs,
validation/test predictions and scores, quantitative JSON, execution log and final hashes.
Start identity is frozen and rechecked at exit. Drift, incomplete evidence or
non-finite results produce a nonzero terminal status; interrupted evidence is
retained. No protected weights, data, old registration or historical output is
overwritten. A changed protocol requires a new prospective attempt preserving
earlier evidence. Generated artifacts use quantitative JSON/CSV/NPZ and a
technical `README.md` describing schema, units, commands and limitations;
there is no `report.md` or narrative results prose.

### Dated prospective implementation clarification — 2026-10-04

Before any numerical execution, the implementation resolves two wording details
without changing counts, models, seeds, budgets, thresholds or scientific scope.
The duplicate-profile restriction applies to synthetic conditions and cross-split
identity collisions. Every one of the 4164 product source-row identities remains
in the product panel, including equal coefficients and equal exact profiles.
Canonical named source order follows P8-S: lexicographic fuel name, then the four
original modes. The listed named fuels define the required set.

Property proofs use `property_inputs_manifest.json`, `generation_terminal.json`
and `property_manifest.json`, in that order. Binary, dependency, generation-output
and log paths are relative to main; final prior-proof pointers are basenames.
Only permitted TRAIN, input-property and provenance files are read or hashed.

The reviewed code automatically reviews the new input manifest by writing its
complete input/case/source identity once and revalidating all hashes and gates
before any training exact field is generated. Test fields remain uncomputed until
all six final checkpoint hashes and the single score reservation are fixed.

### Dated prospective proof clarification — 2026-10-04

The consumer additionally requires the completed, released and committed P8-S
run metadata, validated by `validate_consumer_terminal(..., artifact_paths=...)`.
Its explicit allowlist contains permitted TRAIN/property/input and raw generation
proof files. It never opens or hashes excluded validation/test/ranking or sealed
targets; their hash values remain opaque. Independently join the actual generation
command spec, child handshake, waited exit, log and archived owner lease to the
same producer registration/source/input/core identity and reservation. The inline
launch summary must match these raw records. Later producer terminal evidence can
hash the earlier property manifest; the manifest has no reverse terminal hash.
This tightens provenance only, with no scientific protocol change.
