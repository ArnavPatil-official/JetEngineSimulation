# P8-S: MLX SAF surrogate, prospective registration

Registered design date: 2026-10-04. Registration ID: `P8-S-20261004`.
The corresponding JSON is authoritative. The reviewed registration precedes
computation and fixes the10% effect bar. The user authorized additive isolated
implementation and post-chain diagnostic execution after all registered gates
pass. No simulator, MLX training, numerical test or named target has been run
or opened for this registration work.

## Objective

Measure the number of simulator runs needed for an MLX surrogate to reproduce
the frozen v6 engine's fuel flow and burner-outlet temperature over the V7
four-component mass simplex and thrust fractions 0.07 through 1.00. Compare
otherwise identical data-only and physics-informed models. Derive emissions
indices and lifecycle rates through the existing formulas; do not learn those
formulas as extra labels. Demonstrate practical screening cost only after the
fidelity, ranking, precision, and provenance gates pass.

This is conditional emulation of a particular engine simulator, not empirical
validation of engine performance or fuel certification. All earlier scientific
registrations and results remain historical records with their existing gates.

## Constraints and repo context

The active benchmark chain stays unchanged. Work is additive and isolated.
Data YAML files, earlier registrations, existing trained models, v6 source,
the protected Python engine, and the old C++ benchmark binary are read-only.
No computation runs on battery, beside an active heavy process, before this
registration is committed, or before the existing main chain has terminal
record evidence. No empirical engine target is accessed. The 64 uncertainty
rows contain only the originally fixed engine parameters; their per-draw
refit columns are excluded. They are conditional sensitivity draws, not fuel
composition draws, calibration refits, or confidence intervals.

The current core already returns compressor enthalpies, burner `h_out`,
`cp_out`, `R_out`, `gamma_out`, and the full product mass-fraction vector.
The old flattened `solve_task` drops some of these. A new adapter captures the
full core result without changing that function or the bindings. v6 uses a
single-zone burner (`beta=1`), HP-equilibrium product composition, an analytic
turbine, and a fully expanded analytic nozzle. That scope is retained.

## Inputs, budget, and splits

The external API accepts `(f_JetA, f_HEFA, f_FT, f_ATJ, thrust_fraction,
draw_id)`, with nonnegative mass fractions summing to one and draw IDs 00–63.
The central parameter set is additionally admitted for the named reference
check. The network sees three independent mass fractions, thrust, and eight
fixed draw values, never a categorical identity embedding. Fractions are not
volume fractions or certification limits. ATJ uses the existing, explicitly
formulated `ATJ_C1matched` surrogate; it is not described as a published
composition.

Each train size N=64,128,256,512,1024,2048,4096 is **N simulator runs total**,
not N times 64. A scrambled five-dimensional Sobol sequence supplies three
sorted-uniform spacings for a uniform four-component simplex, one thrust
coordinate, and one coordinate mapped to `floor(64*u)` for the draw ID.
The digital-net prefixes must contain exactly N/64 points of every draw ID.
Training sets are nested prefixes. Separate seeded sequences define 1,024
validation and 2,048 ordinary test runs. Any exact canonical input collision
between splits or with named inputs aborts rather than reallocates points.

An independent locked ranking test contains 64 Sobol compositions at take-off,
each crossed with all 64 draws: 4,096 **test**, not training, runs. This paired
cohort supports common-draw rankings. An independent 2,048-row unlabeled
physics collocation set supplies no additional C++ labels. All manifests are
written before the first simulator call. There is no adaptive acquisition,
resampling of failed points, pilot training, or extra label budget.

Combustor efficiency is piecewise linear through thrust knots
`(0.07, eta_idle), (0.30, eta_approach), (0.85, eta_takeoff),
(1.00, eta_takeoff)`. This preserves all four named operating points exactly;
the original 85% point remains extrapolated context. The frozen mode-state
relations derive core airflow, compressor ratio, fan ratio, and target thrust
from public AE3 design inputs and the selected v6 calibration. Compressor T3
is an input-only NASA7 entropy solve using the frozen compressor air and
efficiency convention. It is neither a learned target nor a concealed
equilibrium solve during deployment.

Unreachable or invalid teacher points retain their original positions and
reasons. N counts attempted simulator runs. Accuracy on converged rows may be
reported conditionally, but any invalid reference prevents an unqualified
full-domain product pass. No domain shrink or replacement sample is allowed.

The named-reference prerequisite is the separately registered
`docs/phase7_p73_a1_registration.{json,md}` and its fresh write-once root
`outputs/phase7/p73_a1_cpp_20261004/`. These are new computations, not
preexisting production results. The trainer and selector never decode them.
Because its registered flattened producer lacks full product state, add
exactly **68** sealed central-property simulator calls: the original 17 named
fuels at four modes, including the alternate JetA2010 context. Save matching
cp/R/gamma/T4 and full 492-species Y with source/core identity; parity with
P7.3-A1 ff/T4 is checked only during the sole score pass. These 68 calls are
metadata/test work, not training N. An input-only property projection carries
canonical fuel/mode IDs and gamma/R/cp/provenance; ff/T4/Y stay in separate
sealed full-state artifacts. Drawn named rows without Y have explicitly
unavailable auxiliary/energy comparisons; there are no hidden 4,096 property
repeats. Ordinary test and paired ranking labels retain complete state.

## Paired models and training

Both models use four hidden layers of 128 SiLU units. The outputs are log fuel
flow, log T4, and 492 product-species logits in the frozen CRECK phase order.
The full-species softmax is a supervised auxiliary head in **both** models,
using exactly the same labels. It makes temperature-dependent mixture
enthalpy computable at deployment without a teacher-only composition or HP
oracle. It is not an emissions chemistry prediction claim.

Seeds are 42, 43, and 44. Each size and arm starts fresh. Initialization,
training batches, labels, Adam settings, 2,000 epochs, learning rate 0.001,
batch size 256, and absence of early stopping are fixed. Training and MLX
weights are explicitly float32. Feature scaling is fitted from that N's
training inputs only. Primary target maps use the fixed references 1 kg/s
and 1,000 K; no future-label target normalization is used. All final scoring
reruns the exported weights and output maps in NumPy float64 on CPU.
Promoting float32 weights does not recover float64 training precision.

Data loss is mean squared log fuel-flow error plus mean squared log T4 error
plus 0.1 times mean per-row squared product-composition L2 error. Physics loss
adds the fixed soft energy, element-conservation, and finite-difference
thrust-monotonicity terms in the JSON. No loss coefficient, epoch count,
activation, architecture, or seed can be changed after results appear.

## Energy relation and its simulator floor

For species k, use the frozen NASA7 polynomial, molecular weight, and the
same gas constant and atomic weights as the teacher. At reference temperature
Tr=298.15 K, let `h_ref(Y)=sum(Y_k*h_k(Tr))` and
`h_s(T,Y)=sum(Y_k*(h_k(T)-h_k(Tr)))`, both in J/kg. The standard NASA7
enthalpy and entropy equations and unit convention are given in the
[Cantera species thermodynamic model documentation](https://cantera.org/stable/reference/thermo/species-thermo.html).
Every species uses its own frozen branch break temperature; unsupported
thermo or temperatures outside valid ranges are errors, not silently clamped.

For burner air mass flow ma, predicted fuel mass flow mf, known fuel mass
fractions Yf, burner-air composition Ya, and predicted product composition Yp,

```
Qchem = ma*h_ref(Ya) + mf*h_ref(Yf) - (ma+mf)*h_ref(Yp)              [W]
Es = (ma+mf)*h_s(T4,Yp) - ma*h_s(T3,Ya) - mf*h_s(T3,Yf)            [W]
rE = (Es - eta_b*(1-xi)*Qchem) / (ma*1e6 J/kg)                     [1]
```

This includes fuel inlet sensible enthalpy and handles chemical reference
enthalpy once. Qchem includes the predicted incomplete/dissociated product
state; it is not automatically the complete-combustion LHV. The additional
complete-combustion engineering diagnostic substitutes `mf*LHV_gas` and is
reported separately, with its dissociation/incomplete-combustion caveat.
The cycle remains on its gas-phase basis; the 0.360 MJ/kg vaporization
correction appears once in liquid-basis lifecycle postprocessing only.

The teacher applies `T4=T3+eta_b*(1-xi)*(T_HP-T3)` and leaves HP product
composition unchanged. This is temperature-rise scaling, not the physical
enthalpy efficiency relation above. Consequently its rE can be nonzero even
with exact teacher outputs. Report that teacher floor on every full-result
split and the 68 central named property rows, both models' absolute residuals,
their signed discrepancies from the teacher, and
accuracy together. Do not subtract a learned correction, fit a test-derived
offset, claim guaranteed zero, or count a zero exact postprocessing identity
as evidence of learned physics. The compressor and burner use slightly
different registered air conventions; the energy inlet uses burner air.

Single-zone beta=1 is unchanged for every valid V7 draw. Dilution is unsupported
in this product registration: v6's beta<1 branch changes mixed T4/cp/R while
leaving h/Y pre-dilution. Any later extension must first register post-dilution
mass mixing and a recomputed `h(T4,Ymix)`; stale fields cannot be combined.

Monotonicity is the soft penalty on negative fuel-flow slope at paired thrust
points separated by 0.001, with every input-only state recomputed at both
endpoints. It is evaluated within one interpolation segment. It does not
assume that eta_b, T4, or blend ordering is monotone. Product mass positivity
and unit sum come from the shared output map. Element-flux conservation is a
separate meaningful residual; neither it nor energy is an algebraic identity
that disappears merely because emissions postprocessing is shared.

## Exact postprocessing

Computed EI_CO2 is the existing complete-combustion carbon balance; rate is
fuel flow times EI. P8-S learns only fuel flow, T4 and shared physical
auxiliaries; its requested derived outputs are EI_CO2, CO2 rate, lifecycle
rate and domain-limited Brem percent change.

Lifecycle rate is `ff * sum_i(f_i*LHV_liquid_i*LCEF_i)`, with kg/s, MJ/kg,
and gCO2e/MJ giving gCO2e/s. The existing CORSIA YAML edition and its
triangular scenarios remain fixed; the dated source is an input, not an
automatically updated regulation. Common seed-42 pathway scenario draws are
shared between compared fuels. The registered [ICAO source document](https://www.icao.int/sites/default/files/environmental-protection/CORSIA/Documents/CORSIA%20Eligible%20Fuels/ICAO-document-06-Default-Life-Cycle-Emissions-November-2025.pdf)
is already cited by the unchanged repository input.

nvPM output is Brem **percent change of number EI relative to Jet A**, never
absolute nvPM or mass EI. Compute `dH=1.5*(f_HEFA+f_FT+f_ATJ)` percentage
points and `F_pct=100*x`. Return a screening number only when F_pct>30 and
0<=dH<0.6; otherwise return null and a reason. Do not extrapolate, substitute
actual surrogate hydrogen for the existing reference convention, or use the
excluded alternate relation. Exact identity/unit/domain checks are structural
software gates, not evidence that the network learned physics.

## Selection, sole score pass, and acceptance

Validation alone selects the smallest N for which all three seeds meet fixed
relative fuel-flow MAE<=0.0001 and maximum<=0.001, T4 MAE<=0.5 K and
maximum<=2 K, and the auxiliary/provenance gates. Select separately for each
arm. If none qualifies, freeze N=4096 as the failed diagnostic candidate.
The deployment prediction is the three-seed mean at the selected physics
size; seeds and both selected arms are reported separately. All 42 models,
validation tables, selections, and prediction arrays for locked inputs are
hashed before any locked labels are opened.

One exclusive, write-once score reservation opens ordinary test labels,
the paired ranking cohort, and newly generated P7.3-A1 named results in the same score
pass. It cannot retrain, change selection, pick another model using test
results, or reopen after a failure. Crash recovery may resume the identical
hashed reservation; partial results remain visible. New generations require
a new prospective registration, not a retry chosen from the test.

The fixed operational fuel-flow bar is **10%** of the smallest positive
central named SAF-versus-JetA2012 fuel-flow effect over HEFA/FT/ATJ
10/20/30/50% and all four original matched-thrust modes. Treat absolute
effects<=1e-12 kg/s as numeric zero and enumerate them. Undefined, missing,
nonfinite, unmatched, or unreachable effects are enumerated and prevent a
fidelity pass; an empty eligible set never produces an infinite or vacuous
threshold. Report the 64-draw minimum separately without changing the bar.
Both mean absolute and maximum absolute error must meet the operational bar
on ordinary test, paired ranking rows, and the named product inputs; T4 uses
the fixed 0.5 K mean/2 K maximum bar. The alternative JetA2010 reference is a
locked representation-spread diagnostic outside the four-component API.

On the paired 64-candidate ranking cohort, rank the 95th percentile over the
64 fixed-draw lifecycle rates at the central LCEF values (NumPy linear
quantile), with deterministic design-ID tie breaks. This is a conditional
sensitivity screening score, not a confidence bound. Require Kendall tau-b
>=0.98, at least 9/10 true top-ten overlap, and >=99% paired ordering agreement
for reference score differences>1e-12 gCO2e/s. An undefined tau, absent
eligible pairs, or invalid reference fails. Named common-draw sign and the
historical claim-rule agreement are reported in full; no new empirical claim
is inferred from emulator accuracy.

Physics learning is a separate claim. Require >=20% lower RMS physical
energy residual for the physics mean than the paired data-only mean at the
same selected physics N, both evaluated on the ordinary test, with all
accuracy/auxiliary gates passing. Each seed's energy residual and spread,
monotonicity violations, element residual, and teacher floor are reported.
An already-zero data-only baseline, undefined denominator, or degraded
accuracy cannot yield a physics benefit pass. A fidelity pass alone does not
imply a physics benefit; a physics benefit alone does not authorize deployment.

## Runtime, break-even, and 640,000 predictions

Final scientific timing uses the actual three-model NumPy float64 CPU product
path, including input-only preparation, output maps, exact postprocessing,
uncertainty aggregation, and transfers. Batch sizes 1,64,4096 and a full
640,000-row run are separate records. Use one warmup and seven timed
repetitions for the three smaller batch sizes; report median, range, cold
start, peak memory, worker/thread counts, versions, power, and thermal state.
The C++ reference uses the same shared-context, provenance-verified core
selection as P7.3-A1, with one worker per physical CPU core and one thread
per worker, using the identical frozen queries. The actual core can be the
old or next validated binary; the original benchmark binary stays retained. Do not charge a batched simulator against a
single surrogate call, or disguise float32 MLX inference timing as float64
production timing. Attempt and record MLX float32 Metal GPU latency at batches1,64,4096 and
640,000 separately, with all3 models, synchronization, output maps, transfers
and exactpostprocessing cost. Unsupported hardware/device errors are explicit
unavailable records and make operational measurements INCOMPLETE; never
silently skip the requested measurement. Test precision parity.
[MLX documents the float64-to-float32 conversion](https://ml-explore.github.io/mlx/build/html/usage/numpy.html);
this study labels the training and deployment precision explicitly.

Require a strictly positive measured median end-to-end float64 bulk speed
advantage (>1x) at batch4096 over the all-core simulator, and a
measured/projection-labeled total-cost break-even at or below 640,000 queries.
Break-even includes every teacher train/validation/test/ranking acquisition,
all 42 model trainings, exports, selection, physics preparation, scoring,
the 68 named property calls, and the separately measured prospective
P7.3-A1 prerequisite cost counted once. A 10x speedup is an optional reported
target, not the primary pass bar. If simulator and surrogate per-query costs do not
give a positive saving, break-even is undefined and the operational gate fails.

The large screening run uses 10,000 independent Sobol compositions, each
at take-off and all 64 fixed draws: exactly 640,000 product predictions.
Fixed thrust prevents idle from trivially winning the ranking. Report LCA,
fuel flow, T4, Brem availability, sensitivity bands, and deterministic top ten.
Verify those ten compositions with all 64 teacher draws (640 new runs),
reporting accuracy and ranking **within selected candidates only**. Project
full simulator cost from registered measured batches. Global top-ten overlap
and regret are demonstrated only by the separately locked 64-candidate
reference cohort. A later optional full 640,000-run teacher reference is
clearly marked in progress until complete; no sampled or selected-only audit
is represented as global optimum verification.

Timing makes exactly 37,449 repeated full-cycle calls: one warmup, seven
timed repetitions and one cold call for each of batches 1,64,4096. These are
measurement calls, not additional training points. The fixed P8-S full-cycle
budget is 49,421 calls (11,264 split acquisition +68 central properties
+37,449 timing +640 selected audit), plus the separately registered P7.3-A1
prerequisite. Source-only thermo tests use 3,444 property states and 24
compressor-only calls, explicitly outside full-cycle training N and charged
in setup. All actual counts/times and failures are retained.

## Provenance, launch, and artifact contract

Before any new source is integrated, export a write-once main-chain terminal
attestation at `outputs/phase8/screening_operations/main_dependency.json`
while the existing strict idle validator still matches its frozen
source identity. New modules would otherwise change that whole tracked tree.
The shared operational registration is
`docs/phase8_screening_operations_registration.{json,md}`. Before compute, a
write-once, committed source-extension manifest enumerates every exact new
source/test path and hash from the scientific registrations and shared helper;
no wildcard permits additions. The future product launcher checks every raw
evidence hash and full record semantics, the original frozen Git blob IDs and
modes against the launch tree and working bytes, the original registrations,
and all retained original built-module hashes. No original path can be changed,
removed, or replaced. Only the explicitly enumerated new paths are allowed.
After proving that original identity, construct the original `Workflow` with
that identity injected for read-only validation of the old records; never
weaken or edit the original helper. Freeze the new source, registration,
input, binary, and environment identity separately. Existing
terminal failures are reported; execution completion is never relabeled a
scientific pass. Missing, foreign, mismatched, live, or drifted records block.
No process-name or ancestor shortcut can release this gate. Acquire a new
exclusive product lease after main releases ownership; AC and identities are
checked before every expensive stage and on exit. Drift produces ERROR,
never PASS. No battery work or change to the active chain is authorized.

The new `scripts.phase8.scientific_workflow_gate` API is
`export_main_context(root)` before new integration, then
`prepare_context(root, registration, allow_new_paths=...,`
`expected_consumer_identity=..., require_g0=True)`. It returns a context with
verified `identity`, `binary_path`, `binary_sha256`, and `original_context`,
plus `require_idle_ac()` and `acquire_run(output_dir,registration_sha256,`
`identity=None)`. A run has `assert_current()` and `release(terminal)`.
The shared exclusive lease is
`outputs/phase8/screening_operations/owner.lease.json`. Mandatory G0 evidence
is the reviewed, committed
`outputs/phase8/screening_operations/g0_evidence.json` receipt, with raw table,
verdict hashes, and actual selected core provenance. Missing evidence blocks
instead of authorizing a new fit or treating prior results as current.

Future artifact root is `outputs/phase8/saf_surrogate/attempt_001/`, with
write-once manifest, split manifests, input-only frozen properties, teacher
TRAIN4096-only CSV/species arrays, separate validation arrays, sealed
test/ranking/named full-state arrays, an input-only named property table, progress JSONL, training logs/weights, validation selection,
score reservation, test/ranking/named metrics, timings, study predictions,
top-ten verification, and a numeric `report.json`. A technical README
documents units, schema, commands and limitations; no prose result report or
`report.md` is produced. Weights stay below that new artifact root;
existing `models/*.pt` are never replaced. A source-state teacher CSV includes
`cp4_J_kg_K`, `R4_J_kg_K`, `gamma4`, T4, and the matching full-species array.
Predicted cp/R/gamma use predicted T4/Y and have independent auxiliary checks;
they are not labeled oracle-exact properties.

`teacher_rows.csv` and `teacher_species.npz` contain TRAIN4096 only, ordered
by prefix_index 0–4095 with IDs train_000000 through train_004095. Every draw
appears exactly64 times. Validation is separate; test/ranking and the full
named central target states are under sealed/. The nozzle consumer may read
the complete TRAIN4096 list and all68 input-only named-property IDs after
fixing the case list, never P8-S validation/test targets or value-selected
property subsets. The alternate JetA2010 named row has in_product_API=false,
null four-component fractions and its explicit original fuel_parts; it remains
a nozzle coefficient case but is never relabeled as JetA2012 or scored through
the P8-S product API. Each row carries a canonical input hash; manifest proof
links registration, source commit/blobs, actual selected core, original main
evidence, G0 receipt, source extension, and split queries.

The exact proof contract uses `property_inputs_manifest.json` before
generation, then `generation_terminal.json` with completed command/exit/source
identity, exact 11,332 acquisition requests and output hashes, then final
`property_manifest.json` binding the prior two records and allowed TRAIN/named
property artifact hashes. Row property_manifest_sha256 refers to the input
manifest, avoiding a self-hash cycle. The final run terminal may hash the final
property manifest afterward. Nozzle verifies only allowed source artifacts;
sealed val/test/named target hashes are opaque producer evidence and their
files are never opened or hashed by that consumer.

Relevant source paths and SHA256 pins are in the JSON. Empirical target and
named result paths are listed as sealed references but were not read or
hashed for this draft. The sole reservation freezes their bytes before decode
and validates their newly generated P7.3-A1 provenance and paired key completeness.

## Implementation phases and file-level edits

1. Commit these reviewed registration documents after staticchecks; do not run models while the original chain is active.
2. After terminal main evidence, export the dependency attestation and implement
   additive modules under `scripts/phase8/saf_surrogate/` and additive tests.
3. Pure tests verify identities, splits, units, thermo algebra, derivative
   preservation, failures, score sealing, and runtime accounting. Any live-core
   or MLX numerical checks remain AC-only and separately recorded.
4. Freeze source, binary, whitelisted inputs, properties, and every query list;
   generate exactly the fixed labels and retain failed rows.
5. Train all fixed arms/sizes/seeds, validate, export and seal selections.
6. Perform the one locked score pass. Collect registered operational timing,
   large screening and selected teacher verification even when fidelity fails;
   label those measurements diagnostic/unsafe and gate deployment and ranking
   claims. Input/provenance/AC/ownership failures still block execution. Do not
   fit further or change selection after a failed score.

Planned new modules are `run.py`, `registration.py`, `inputs.py`, `teacher.py`,
`thermo.py`, `postprocess.py`, `models.py`, `train.py`, `score.py`, `timing.py`,
and `study.py`. Existing core, engine, fuel, optimization, and MLX modules are
read-only references. No prior registration is edited. The user-authorized executor scope is additive modules/tests in this isolated
worktree. Static implementation review may proceed while the main benchmark
exists; scientific imports and jobs wait for the guarded post-chain launch.

## Commands and tests

Prospective commands are fixed in the JSON and are unavailable until the new
CLI exists. Pure tests use a specific additive test module and must not import
the live core or open targets. Registered numerical checks verify NASA7
enthalpy/cp/entropy against the retained teacher for fixed source-only states,
MLX gradients against finite differences without NumPy detachment, output
export parity, teacher input-state parity, and deterministic Sobol counts.
The original full pytest and protected-hash obligations remain applicable
when the new implementation is integrated after the main chain.

## Rollback notes and escalation guidance

Revert only a new additive implementation or registration commit; retain all
run evidence and prior records. Partial attempts remain write-once with their
terminal status. A missing input, bad provenance, solver failure, physics
approximation, failed operational bar, or unfinished reference is flagged
explicitly rather than bypassed. The executor work is nontrivial (MLX
autodifferentiation, source identity, thermochemistry units, and score sealing)
and needs a high-capability code executor and independent static review.
Scientific thresholds change only through a prospective amendment before a
new experiment, with this attempt retained. No history rewrite or push.
