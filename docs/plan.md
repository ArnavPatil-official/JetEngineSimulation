# Shared ML backend and resumable Python-v6 PC study

## Objective

Complete the user three-task request: push all branches/tags now, add a shared
Torch-primary/MLX-optional backend, deliver the resumable Python-v6 PC study,
commit and fixture-test, then push all branches/tags again without force.

## Constraints

Work only in the isolated pc-backend worktree. Preserve the original checkout,
running chain, refs, lease, operational files and untracked files. No project
training/generation/scoring runs here. Commit the prospective amendment before
any training fixture. Never overwrite data YAML, trained weights or consumed
score locks. Stage only exact task files. Existing Codex executor override applies.

## Repo Context

SAF uses separateMLX32/Torch32 routines, nozzleTorch64; oldPC requiresC++Linux.
New PC uses full Python-v6 and actual local source/host/PID provenance. Mac C++
and historical chain remain optional and unchanged in the original checkout.

## Relevant Files

CREATE simulation/ml_backend/{__init__,interface,numpy_reference,torch_backend,
mlx_backend}.py and tests/test_ml_backend.py.
CREATE docs/phase8_backend_pc_amendment.json/.md; MODIFY prospective SAF/nozzle
registration JSON files for requested backend/dtype andfour-size schedule.
MODIFY scripts/phase8/saf_surrogate/{models,train,train_torch,thermo,registration,
score,run,timing,teacher}.py; nozzle_ode/{model,run,score}.py; p73_a1_cpp.py.
CREATE scripts/pc_pipeline.py, scripts/phase8/python_pc.py and necessary new
portable runtime helper. CREATE docs/PC_SETUP.md and fixtures for stage/guards,
resume/one-shot protection/real dispatch/scopedGitpublication.
MODIFY rootPC_SETUP.md/pc_pipeline.py asforwarding compatibility entries,
README.md, FIXES.md and execution status. Audit currenttraining/scoringCLIs.

## Implementation Phases

### Phase1 — Publish and register prospectively

Push --all then --tags; reportrejections, noforce. Commitamendment beforefits.
Record no scientific P8-S/nozzle result exists. Preserve oldchain/checkoutevidence.

### Phase2 — Shared backend and consumer integration

One interface for arrays/dtype, MLP, Adam/optionalTorchL-BFGS, losses, parameter
andinputfirst/secondderivatives, seeds/devices andneutralNPZ checkpoints.
--backend orCATJET_ML_BACKEND selectsTorch(default) orMLX. TorchCPU64training
andscoring; optionalMLX32training/NumPyCPU64score. Preservefloat64weights.
SAF fourNs64/256/1024/4096,twoarms,3seeds. Primarynozzle keepsregisteredTorch
Adam/L-BFGScaps;optionalMLXAdamonly disclosesmissingpolish. PreserveSobolsamples.

### Phase3 — Python PC workflow, review and final publication

No C++orMaccommands required onPC. Platform-awarepower/host/PIDleases;Macbehavior
retained. Actual resumablestages: environment/hardware;20frozenPythonv6rowsparity
rtol1e-9 stopfailure;P73;SAFgeneration;24fits;solelockedscore;nozzle;measured1/10
workerssimulatorandsurrogatetiming/break-even;commitownoutputstopc-run-date/push.
Resume only byte-verified completedstage receipts. Never reopenconsumedscore or
replacetrainedmodel. Generation-stage proofs,notfabricatedfullproducerterminals,
authorizenozzleTRAINpropertyprojectionsbeforetimingstage. Defaults10workers,
CPU,OMP/OPENBLAS1. DocumentWSL2/Miniforge/pipCantera3.2.0/TorchCPUclone/run.

## File-Level Edits

Shared modules implement actual selectedframework operations andindependentNumPy
reference. Consumerspreserveinputs/losses/labels/seeds whiledelegatingbackend.
Pythonadapters reusefullv6 andregisteredproperty/nozzle procedures. CLI runsreal
stages/checkpoints andscopedGitpublication. Doccommands matchimplementedflags.

## Commands to Run

Use originalcheckout .venv Python withcwdthisworktree forfixturepytest/syntax.
Useinstalledcatjet-mlxPython for tinyactualbackendparity afteramendmentcommit.
NoPC --run here. gitdiff--check. Fullpytestonlyif noactiveheavyowner,reportinherited
Macgatefailures. Finally gitpushorigin--all, then--tags,verifyremoterefs.

## Tests

Same neutralweights outputs/losses/parameterandinputfirst/secondgradients:
relative1e-5float32;Torch64vsindependentNumPy1e-10. MLXabsentskipscleanly.
Neutralroundtrip/float64preservation/backendCLIenvdefault/optimizeroptions.
Macpowerbehavior,Linuxno-batterymains,host/PIDownership;parityFAILstops;
resumebytesdrift/oneshotscore/failedstageretention/scopedGitoutputs. Fixtureonly.

## Acceptance Criteria

Pushorder/noforce/untrackedunchanged;amendmentcommitprecedesfixtures. Backend
actuallyselectsframework/dtype andparitypasses. PCcommandexecutesa-iwithoutC++
andisresumablewithprotectedoneshotscore. Setupcommandsagree. Reportrealvalidation
limits andinheritedgatefailures; no projectresults invented.

## Rollback Notes

RevertisolatedtaskcommitsbeforePCcomputation. Retainconsumedattemptsandlocks.
Never resetoriginalcheckout,chain,weightsorhistoricalevidence.

## Escalation Guidance

Highcomplexity ML/PINN/PCintegration;independentagentscoordinateAPIandfileownership.
Rootreviews/commits. ExistingexplicitCodexexecutoroverridepersists;userauthorizes
thistaskincludingprospectiveamendment. Noadditionalregistrationscope/handoffrules.
