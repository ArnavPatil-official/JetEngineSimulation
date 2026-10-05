# Phase 8 defects and pending verification

Updated 2026-10-04. User decisions are registered; no further scientific
approval is pending. Registration, one-shot scoring and protected files remain mandatory.
The user authorized publishing phase8 on 2026-10-04. Existing results are preserved.

## Corrected before AC launch

Codex completed the remaining implementation after the user's explicit
executor override. All corrections were reviewed in isolated worktrees and
integrated before any numerical launch.

| Item | Correction / evidence | State |
|---|---|---|
| Queue lifecycle and cached evidence | Shared strict completion validator; historical and fresh launch evidence; PID/birth lease; handshake before go; AC/source/dependency rechecks; nonzero finalization | Fixed `8635403`; 57 queue/G0 tests passed |
| A5 workflow and binary identity | Require terminal main records and validated separate-core build; recheck before reactor work; end drift is ERROR | Registered `7d545ed`, fixed `a79312d`; 26 tests passed |
| A4c frozen fit/profile/fallback evidence | Exact dependencies, canonical profile replay, full artifacts and source/input/core identity before reservation | Fixed `ca60227`; 38 tests and four subtests passed |
| Input-only convergence and flow domain | C1 opt-in public-input bound, unchanged defaults, separate build, actual all-20 assertion and recorded full-pytest child exception | Registered `9e4db00`, fixed `ca60227`; actual solves pending on AC |
| OAT failed-design label and architecture provenance | Check converged flag; cite verified primary sources and label remaining proxies | Fixed `ca60227`; numerical error budget pending |

Historical pre-launch assessment (superseded by retirement of the Mac chain):
No launch blocker remained from implementation review. Compilation and
scientific verification are pending in the registered AC sequence; their
results must be reported from actual records, not inferred from fixture tests.

## Non-blocking defects and limits

- Old benchmark records lack clean power evidence in two timings and have
  registered overlap flags. Preserve them and report flags; the two existing
  rerun jobs are the only supplementary benchmark jobs registered.
- A4c coefficient/efficiency/cooling envelopes are declared engineering
  ranges informed by cited evidence, not measured Trent confidence intervals.
- Error-budget shares are calibration-engine responses normalized by known
  aggregates; they are not an additive allocation of held-out error.
- The all-family check is representative design convergence, with public
  input records and declared architecture proxies; it is not geared
  off-design validation or empirical engine validation.
- Track 4 Ma Eq. 25 thermal-unit ambiguity remains flagged. Diagnostic
  scores, loss balancing and future empirical/Sajben work retain their
  previously registered scope and order.

- Run-2 arm4b_W4_11w reference comparison: warm-up internally matches,
  nine core-thrust cells fail the frozen comparison. Retain **FAIL**, raw
  evidence and absent timing; no tolerance change, repeat or active-core edit.
  This does not promote the cached/approximate arm to a valid speed result.
- New study dependencies: the existing MLX Python is 3.12, but static package
  metadata did not show SciPy/PyYAML/Cantera. Verify and prepare that environment
  only after the main lease releases, before new data generation; no protected
  requirement/config or original environment changes during benchmarks.

## Concrete PC portability defects — 2026-10-04

- The historical Mac chain and downstream gate require A2 although the user
  deferred it. The parked chain was stopped before A2 started; it is retired.
  Preserve its source and benchmark evidence. Repair historical consumers later;
  the new PC sequence omits A2.
- The existing Mac execution gate pins a Mac binary and calls pmset/sysctl/sw_vers.
  A rebuilt PC core cannot satisfy that identity. PC execution needs actual local
  binary/source provenance, rather than reusing Mac numerical evidence.

## PyTorch backend and remaining portability limits — 2026-10-04

The SAF sources (saf-surrogate worktree, `6811085`) and shared consumers
(screening-operations worktree, `0a4096b`) are now copied onto phase8, plus
`train_torch.py`. `run.py run --backend torch --device auto|cpu|cuda` trains
without MLX; scoring stays NumPy CPU64. `pc_pipeline.py` now runs the PC route
through actual local source/core/G0 evidence. Existing registrations are unchanged.

- The default historical SAF entry still uses the Mac execution gate. The PC
  entry supplies Linux power/hardware and its actual local core/G0/source proof.
  Repair of historical Mac consumers after A2 deferral remains deferred.
- `study.py` imports `resource` at module scope. That is fine on Linux/WSL2,
  but the module will not import on native Windows. `train.py` now treats it as optional.
- Torch CUDA timing is implemented and explicitly labeled `torch_cuda` in the
  existing GPU timing artifact. It uses synchronized float32 inference,
  transfers and CPU64 postprocessing. CPU mode records GPU UNAVAILABLE; the
  GPU-dependent operational gate cannot pass there. Actual PC/CUDA measurements
  are pending and no measured speed result is claimed.
- The registration names MLX keyed initialization and MLX Adam. Torch uses the
  same U[-1/sqrt(fan_in), +1/sqrt(fan_in)] family, keyed by a CPU
  `torch.Generator`, and `torch.optim.Adam` with identical hyperparameters and
  bias-corrected update. Its weights are not bitwise MLX replicates.
  `training_backend` in each fit record and `precision.json` discloses this.
- MLX CPU device defect on this Mac: `mlx.nn.silu` is `mx.compile`d, and MLX's CPU
  JIT fails to link against the CommandLineTools MacOSX27.0 SDK
  (`tapi ... unknown architecture arm64e.x1`). The original MLX
  `source_diagnostics` branch sets `mx.cpu` for export parity, so on this Mac it
  would raise before label acquisition. The GPU device works.
- `cpp/build_pc.sh` and `cpp/pc/CMakeLists.txt` provide a separate Linux build
  with GNU version scripts. The original Mac build files are unchanged. Actual
  Linux compilation/import/G0 validation remains pending on the PC.
- The registered `commands` still name `~/miniforge3/envs/catjet-mlx`;
  `environment.json` now records the selected backend's package version.
- `train_torch.py` and `tests/test_phase8_saf_surrogate_torch.py` are not in the
  SAF registration's `implementation_create` list. The shared gate accepts only
  declared new paths in its source-extension manifest, so it will reject them as
  unregistered additions. The PC manifest hashes the actual Torch sources.
  Historical gate repair is deferred; no registration amendments are added.
- Optional public top-K simulator verification still invokes the historical
  Mac execution gate and cannot run on PC yet. The PC pipeline tests public
  CPU64 screening and failed-deployment refusal without this optional feature.
- The SAF and nozzle numerical fixture tests error until
  `outputs/phase8/screening_operations/{main_dependency,source_extension_manifest}.json`
  are committed. This is the unchanged historical gate requirement.
- `test_all20_eligible_families` still uses the historical chain validator.
  After source integration its original benchmark completion identity differs
  from the extended current source tree. The post-commit recheck fails before
  any all-20 solve with `completion source identity differs from current sources`.
  This concrete historical test-gate defect remains deferred.

Final full-suite review: 767 passed, 4 skipped, 1 failed, 16 setup errors and
four passed subtests (235.33 s). The single failure is the historical all-20
workflow gate; ten nozzle and six SAF fixtures fail before scientific bodies
because the historical committed context evidence is absent. Preserve these
results; no registrations or missing receipts are fabricated to turn them green.
After concrete PC implementation corrections, final focused checks report
211 passed and one MLX-only skip. Help/dry-run, syntax and protected hashes pass.
The post-commit all-20 recheck also fails (0.51 s) at the historical benchmark
completion identity, confirming this remains an inherited consumer defect after
the task files are committed. No historical evidence was changed.

## New backend review defects — 2026-10-04

- Torch was imported before thread limits were set in run.pipeline; an already
  imported Torch kept multiple CPU threads despite thread_environment=1.
- CUDA execution used current_device while metadata named GPU0. These new
  implementation defects are corrected and covered by targeted tests.

## Corrected new PC implementation defects — 2026-10-04

The following concrete defects were logged during implementation review and
corrected before the final push, with manufactured regression checks.

- `cpp/pc/CMakeLists.txt` initially used GNU ld `--retain-symbols-file` as an
  export whitelist. That option strips the regular symbol table and does not
  localize dynamic Cantera exports, so it cannot provide the required isolation
  from the pip Cantera wheel. Use a per-module anonymous version script with
  only `PyInit_<module>` global and all other symbols local. This concrete new
  implementation defect is corrected with anonymous version scripts. References:
  [GNU ld options](https://sourceware.org/binutils/docs/ld/Options.html) and
  [GNU ld version scripts](https://sourceware.org/binutils/docs/ld/VERSION.html).

- Initial `PCRun.assert_current()` checked lease ownership and power but did not
  recheck the captured local source/core/registration hashes. Revalidate those
  bytes before each scientific boundary; a stored fingerprint alone does not
  detect execution-time drift. This is a concrete new implementation defect,
  separate from the deferred historical Mac gate.

- The first local PC producer validator did not bind the G0 result file bytes,
  only its path and in-memory result. Record and recheck the actual G0 SHA256
  alongside its PASS/core identity before accepting local producer evidence.

- Linux power detection initially recognized only Mains/USB supplies, rejecting
  powered laptops whose kernel reports USB_C/USB_PD/USB_PD_DRP. Accept the actual
  online USB supply variants and retain the battery/offline refusal.

- Initial PC build path validation accepted `cpp/` itself and descendants of
  the historical build trees. Reject those paths so the portable build cannot
  write into the source directory or original build evidence.
- The initial PC light epoch check rehashed all 74.6 MB of protected/source
  files on each of 84,000 epochs, adding about 6 TB of repeated reads. Keep the
  existing full checks at fit boundaries and every 100 epochs; the per-epoch
  check verifies ownership, power and small registration/G0 fingerprints.
- Initial Torch CUDA timing had only CPU32 export precision checks. Compare
  actual CUDA output from the selected exported weights against CPU64 on fixed
  queries using the existing ff/T4/species precision bounds; invalid output
  must not produce a COMPLETE GPU timing receipt.
- The unavailable-CUDA unit test originally assumed the host had no GPU and
  would fail the PC's focused checks on a CUDA machine. It now simulates that
  branch instead of depending on the host's hardware.

## Shared backend / Python PC implementation review — 2026-10-04

The user requested Torch CPU float64 primary, optional MLX, and a Python v6 PC
pipeline. The prospective amendment is committed as `d3767dc`, before tiny
training fixtures. This work is isolated on `phase8-pc-backend-20261004`; the
original checkout and operational files are untouched. No new P8-S or nozzle
scientific study results exist. A2 remains deferred and is absent from this PC
sequence. Historical Mac consumer defects above remain deferred.

Concrete implementation corrections in this task:

- Primary Torch training previously used float32, and checkpoint export could
  downcast float64 weights. Shared models and training now use CPU float64 and
  preserve native parameter dtype in neutral NPZ. Optional MLX exports retain
  float32 until CPU float64 scoring.
- Backend flags could describe Torch while executing NumPy inference. Primary
  scoring now executes native Torch float64 on CPU; optional MLX uses NumPy64.
- Native MLX SiLU/softplus wrappers could invoke the failing CPU JIT noted above.
  The shared implementation uses equivalent sigmoid/logaddexp expressions.
  Actual MLX CPU fixtures pass without changing system SDKs or environments.
- Cross-framework neutral nozzle loading needs activation metadata. Shared NPZ
  archives now retain the tanh architecture and trained precision.
- Python-only producers have no C++ binary hash. Their empty CSV binary field is
  checked as absent, while authenticated generation receipts bind the actual
  Python simulator identity. Existing scientific property projections remain
  unchanged; no extra registration columns or fabricated C++ identity are added.

The independent fixture comparison has maximum relative family errors of
`2.87e-16` for Torch64 versus analytic NumPy64 and `1.92e-7` for Torch32 versus
actual MLX32, below `1e-10` and `1e-5` respectively. MLX-installed fixtures:
17 passed. Torch-only environment: 12 passed, 5 skipped. These are manufactured
checks, not study fits or locked test results.
- MLX Adam does not accept Torch's zero weight-decay/disabled AMSGrad keyword
  arguments. The shared adapter accepts those neutral defaults and rejects
  unsupported non-default variants; real MLX optimizer fixtures pass.
- Native MLX modules subclass dict. Neutral-weight detection now checks for
  canonical layers.i keys, preventing a native nozzle model from being mistaken
  for a parameter archive. Real MLX native scoring/cross-load fixtures cover it.
- Completed PC learning-curve receipts initially omitted the actual model and
  selection files. The stage now seals all 24 model/log artifacts and the
  selection/validation/progress bytes before subsequent stages.
- The new PC timing route initially inherited a GPU prerequisite, making the
  requested CPU-only PC fail regardless of CPU measurements. Primary PC timing
  uses its measured CPU gate; optional GPU availability stays explicit.
- Publication retries initially collided with the consumed stage marker and
  existing run branch after a network failure. A retry now verifies and pushes
  the exact owned recorded commit without repeating scientific stages.
- The 640,000-row study JSON exceeds GitHub's 100 MiB regular-file limit, so
  plain output publication would be rejected. Oversized outputs now use lossless
  gzip chunks of at most 50 MiB with original/chunk SHA256 manifests and a
  scoped restore command. Originals remain unchanged locally. This transport
  changes no scientific rows or registration. Source:
  [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- Checkpoint validation initially resolved paths before testing symlinks, which
  could accept replacement symlinks with identical bytes. It now rejects the
  original symlink path before resolving; a manufactured resume fixture passes.
- Git text filters or autocrlf could change sealed artifact bytes during staging.
  Scoped publication disables autocrlf for its add command and verifies raw
  staged blobs before commit; retry also verifies the committed bytes. The
  CRLF fixture passes without changing project/global Git configuration.
