"""
Train the nozzle LE-PINN on the Sajben weak-shock case (P4.3).

Training data: ``data/processed/sajben_wind_dataset.pt`` — the NASA/WIND
Spalart-Allmaras RANS solution decoded from the NPARC archive, with the
train / evaluation split declared in the dataset by
``scripts/validation/sajben_split.py`` (route (a) of the P4.1 audit).  The
experiment (``data/raw/data.Mach46.txt``) is the evaluation set and never
enters training; the split record is copied into the checkpoint so
``sajben_validation.py`` can verify that.

PRE-REGISTERED ATTEMPT 1 (committed before the run; ``ATTEMPT`` below)
-----------------------------------------------------------------------
Changing any of these values makes a new attempt with its own record; every
attempt is reported (docs/plan.md, P4.3).

* fresh initialisation (no pretrained checkpoint), seed 42, CPU
* 5000 full-batch epochs, AdamW lr 1e-3 (wd 1e-5), ReduceLROnPlateau on the
  data loss (factor 0.5, patience 20) — the optimiser/scheduler already in
  ``finetune_on_cfd_data``
* loss = 1.0 * data_MSE + 0.05 * ramp(t) * physics_residual, i.e. the
  existing data + RANS-residual structure with ``PhysicsWarmupWeighting``:
  the data weight is constant and the physics weight ramps linearly 0 -> 1
  over the first half of the run.  The historical schedule decayed the
  data weight to 0.08 and drove long runs to the trivial constant field
  (P4.2); it is not used.
* disclosed exploratory smokes (100 epochs, seed 42, before registration,
  none of them a gate attempt): constant 0.5/0.5 weighting reached internal
  val MSE 0.122 with 107 alive units in the narrowest layer; the warm-up
  ramp 0.040 / 151; data-only 0.015 / 162.  The ramp is registered because
  the claim under test is "physics-informed"; the data-only configuration
  is run alongside as REFERENCE 1 (``--physics-weight 0
  --attempt-id P4.3-reference-1-data-only``), an ablation that is reported
  next to attempt 1 but does not compete for the gate.
* physics residual as implemented (planar geometry, Sutherland laminar
  viscosity in ``_safe_physics_loss``; with ReLU activations its
  second-derivative terms vanish, so it is effectively an Euler residual).
  Recorded as a known limitation of attempt 1, not changed here.
* internal validation split 20 % (seed 42) for best-weight restore and
  collapse monitoring; the gate is scored on the experiment afterwards by
  ``sajben_validation.py --model models/le_pinn_sajben_v5.pt``
* output ``models/le_pinn_sajben_v5.pt`` (never overwrites an existing file)

Gate (fixed in docs/plan.md before this run): held-out wall-Cp shape-L2
< 0.10 pass / 0.10-0.25 partial / > 0.25 fail.  The training data's own
score on that metric is 0.089 / 0.084 (P4.1 §3).

ATTEMPT 3 — TERMINAL (registered 2026-09-19, before the run)
------------------------------------------------------------
Attempts 1 and 2 (0.258 fail, 0.245 partial on one seed) and the data-only
ablation (0.105) share one diagnosed defect, verified independently of any
score in ``outputs/physics_residual_defect.md``: the physics loss enforced
inviscid Euler. (i) ReLU is piecewise linear, so every second derivative in
``compute_rans_residuals`` — the viscous and thermal-diffusion terms, the
only second-derivative terms — is exactly zero; (ii) ``mu_eff`` was
overwritten with Sutherland molecular viscosity while the field is a
turbulent RANS solution with mu_t/mu_l up to 1750 in the boundary layer.
Neither fix works alone. Attempt 3 changes exactly these two things,
together, on top of attempt 2's configuration:

* ``activation = "tanh"`` (C-infinity) in both sub-networks — recorded in
  the checkpoint; ReLU checkpoints still load;
* ``physics_mu_source = "data"``: the residual uses the WIND field's
  ``mu_l + mu_t`` at the collocation points (dataset target column 8).
  The residual keeps its existing Laplacian form mu_eff * lap(u); the
  grad(mu) . grad(u) term of the full divergence form is NOT added and is
  recorded as a known approximation of the formulation.

Design fixed before the run:

* seeds {42, 43, 44}; each seed trained twice, physics-on (weight 0.05,
  warm-up as before) and a MATCHED data-only ablation (weight 0.0);
* the band is called on the DISTRIBUTION over seeds: on the mean worse-wall
  L2, and only claimed if every seed falls in the same band — otherwise the
  result is reported as straddling, with the range;
* supplementary, not a gate change: the ceiling-relative error
  L2 - 0.089 is reported beside each score, since the training data itself
  scores 0.089 on the upper wall;
* interpretation, fixed now: physics-on within the seed spread of data-only
  = physics consistency at no accuracy cost; physics-on below data-only =
  the paper's positive result; physics-on above data-only = a negative
  result about this residual formulation, reported as such.
* This is the last attempt the executor registers. Whatever it shows is
  the P4.3 outcome.

    python scripts/validation/train_sajben.py --attempt 3 --seed 42
    python scripts/validation/train_sajben.py --attempt 3 --seed 42 --physics-weight 0 --attempt-id P4.3-attempt-3-dataonly

Usage::

    python scripts/validation/train_sajben.py                      # attempt 1 as registered
    python scripts/validation/train_sajben.py --epochs 200 --out /tmp/x.pt   # smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.nozzle.le_pinn import PhysicsWarmupWeighting, finetune_on_cfd_data

# ---------------------------------------------------------------------------
# Dataset path, output path and the pre-registered configuration
# ---------------------------------------------------------------------------
DATASET_PATH = str(_ROOT / "data" / "processed" / "sajben_wind_dataset.pt")
SAVE_PATH = str(_ROOT / "models" / "le_pinn_sajben_v5.pt")

ATTEMPTS = {
    1: {
        "id": "P4.3-attempt-1",
        "registered": "2026-09-18",
        "n_epochs": 5000,
        "lr": 1e-3,
        "lr_schedule": "plateau",
        "physics_loss_weight": 0.05,
        "loss_weighting": {"schedule": "PhysicsWarmupWeighting", "data": 1.0, "warmup_fraction": 0.5},
        "physics_max_points": None,
        "val_fraction": 0.2,
        "seed": 42,
        "geometry": "planar",
        "pretrained": None,
        "out": "models/le_pinn_sajben_v5.pt",
    },
    # Registered 2026-09-18 AFTER attempt 1 scored 0.258 (fail band), with
    # exactly one diagnosed defect fixed: the ReduceLROnPlateau scheduler,
    # stepped on the per-epoch data loss, halved the learning rate every
    # 21 epochs once the physics warm-up made that loss non-monotonic and
    # reached 1.5e-8 by epoch ~1000 (best validation at epoch 350 of 5000).
    # Attempt 2 uses a cosine schedule (1e-3 -> 1e-5 over the run) that
    # does not depend on the loss trajectory. Everything else identical.
    # Reported next to attempt 1; no other parameter was touched.
    2: {
        "id": "P4.3-attempt-2",
        "registered": "2026-09-18",
        "n_epochs": 5000,
        "lr": 1e-3,
        "lr_schedule": "cosine",
        "physics_loss_weight": 0.05,
        "loss_weighting": {"schedule": "PhysicsWarmupWeighting", "data": 1.0, "warmup_fraction": 0.5},
        "physics_max_points": None,
        "val_fraction": 0.2,
        "seed": 42,
        "geometry": "planar",
        "pretrained": None,
        "out": "models/le_pinn_sajben_v5_a2.pt",
    },
    # TERMINAL attempt, registered 2026-09-19 before the run. Fixes the
    # residual-formulation defect (outputs/physics_residual_defect.md):
    # tanh activation (non-zero second derivatives) + mu_eff from the WIND
    # field (mu_l + mu_t) at the collocation points. Three seeds, matched
    # data-only ablation, band called on the distribution. See docstring.
    3: {
        "id": "P4.3-attempt-3",
        "registered": "2026-09-19",
        "n_epochs": 5000,
        "lr": 1e-3,
        "lr_schedule": "cosine",
        "activation": "tanh",
        "physics_mu_source": "data",
        "physics_loss_weight": 0.05,
        "loss_weighting": {"schedule": "PhysicsWarmupWeighting", "data": 1.0, "warmup_fraction": 0.5},
        "physics_max_points": None,
        "val_fraction": 0.2,
        "seed": 42,
        "seeds": [42, 43, 44],
        "geometry": "planar",
        "pretrained": None,
        "out": "models/le_pinn_sajben_v5_a3_s{seed}.pt",
        "out_dataonly": "models/le_pinn_sajben_v5_a3_dataonly_s{seed}.pt",
        "band_rule": "mean over seeds; claimed only if all seeds agree, else reported as straddling",
        "terminal": True,
    },
}
ATTEMPT = ATTEMPTS[1]   # rebound by main() from --attempt


def resolve_device(name: str) -> str:
    """``auto`` -> cpu (reproducible everywhere); explicit names pass through."""
    import torch
    if name == "auto":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return name


def train_sajben_le_pinn(
    n_epochs: int = ATTEMPT["n_epochs"],
    lr: float = ATTEMPT["lr"],
    save_path: str | None = None,
    device: str = "cpu",
    verbose: bool = True,
    physics_loss_weight: float = ATTEMPT["physics_loss_weight"],
    physics_debug: bool = False,
    seed: int = ATTEMPT["seed"],
    attempt_id: str = ATTEMPT["id"],
) -> tuple:
    """
    Train the LE-PINN from scratch on the declared Sajben training set.

    Returns ``(model, history)``.
    """
    import torch

    if save_path is None:
        save_path = SAVE_PATH

    # Validate dataset existence
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Sajben dataset not found: {DATASET_PATH}\n"
            f"Run 'python scripts/validation/sajben_split.py' to generate it."
        )

    # Load and validate schema
    try:
        dataset = torch.load(DATASET_PATH, weights_only=False)
    except TypeError:
        dataset = torch.load(DATASET_PATH)

    required_keys = {"inputs", "targets"}
    missing = required_keys - set(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset schema validation failed. Missing keys: {missing}\n"
            f"Expected keys: {required_keys}"
        )
    split = dataset.get("split")
    if split is None:
        raise ValueError(
            "Dataset carries no declared train/eval split record; refusing to train. "
            "Build it with scripts/validation/sajben_split.py."
        )

    if verbose:
        print("Dataset validation passed:")
        print(f"  Resolved path: {Path(DATASET_PATH).resolve()}")
        print(f"  Inputs shape : {dataset['inputs'].shape}")
        print(f"  Targets shape: {dataset['targets'].shape}")
        print(f"  Split        : train={split['train_source']} ({split['train_rows']} rows), "
              f"eval={split['eval_source']} (rows in train: {split['eval_rows_in_train']})")
        print(f"  Attempt      : {attempt_id}  (seed {seed}, epochs {n_epochs}, lr {lr}, "
              f"physics {physics_loss_weight}, data weight 1.0, physics warm-up over first "
              f"{int(100*ATTEMPT['loss_weighting']['warmup_fraction'])} % of epochs)")
        print(f"  Geometry mode: planar (Sajben 2D diffuser)")
        print()

    weighting = PhysicsWarmupWeighting(
        max_epochs=n_epochs,
        warmup_fraction=ATTEMPT["loss_weighting"]["warmup_fraction"],
    )

    model, history = finetune_on_cfd_data(
        dataset_path=DATASET_PATH,
        pretrained_path=None,  # Fresh init (no pretrained checkpoint)
        save_path=save_path,
        n_epochs=n_epochs,
        lr=lr,
        physics_loss_weight=physics_loss_weight,
        physics_max_points=ATTEMPT["physics_max_points"],
        val_fraction=ATTEMPT["val_fraction"],
        device=device,
        verbose=verbose,
        geometry="planar",  # Sajben is 2D planar, not axisymmetric
        physics_debug=physics_debug,
        seed=seed,
        loss_weighting=weighting,
        lr_schedule=ATTEMPT.get("lr_schedule", "plateau"),
        activation=ATTEMPT.get("activation", "relu"),
        physics_mu_source=ATTEMPT.get("physics_mu_source", "sutherland"),
        extra_payload={"split": split, "attempt": {**ATTEMPT, "id": attempt_id,
                                                   "n_epochs": n_epochs, "lr": lr,
                                                   "physics_loss_weight": physics_loss_weight,
                                                   "seed": seed}},
    )

    return model, history


class _Tee:
    """Write-through copy of a stream into the run log (flushed per write, so a
    killed run leaves everything it printed — unlike a buffered redirect)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def default_log_path(attempt: int, seed: int, dataonly: bool) -> Path:
    """Registered log location; attempts 1/2 keep their historical names."""
    tag = f"a{attempt}_dataonly_s{seed}" if dataonly else f"a{attempt}_s{seed}"
    return _ROOT / "outputs" / "logs" / f"train_sajben_v5_{tag}.log"


def done_marker_path(log_path: Path) -> Path:
    return log_path.with_suffix(".done")


def guarded_run(log_path: Path, checkpoint: str, run) -> int:
    """
    P5.1 launch guard. Tees stdout/stderr into ``log_path`` and, when the run
    ends by any route Python can observe (success, exception, SIGTERM/SIGINT/
    SIGHUP), writes ``<log>.done`` recording the exit code. A run killed with
    SIGKILL or by power loss leaves no marker, which the reporter treats as a
    failed run. Refuses to start over an existing marker or non-empty log.
    """
    import hashlib
    import json
    import os
    import platform
    import signal
    import time
    import traceback

    import torch

    marker = done_marker_path(log_path)
    if marker.exists() or (log_path.exists() and log_path.stat().st_size > 0):
        raise SystemExit(f"refusing to launch: {log_path} or {marker} already exists "
                         "(logs and completion markers are never overwritten)")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_signal(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    fh = open(log_path, "a", encoding="utf-8")
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, fh), _Tee(err, fh)
    print(f"[launch-guard] pid {os.getpid()}  started {started}  argv {' '.join(sys.argv)}")
    exit_code, error = 1, None
    try:
        run()
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        error = f"SystemExit({exc.code})"
    except BaseException:  # noqa: BLE001 — recorded, then re-signalled via exit code
        error = traceback.format_exc()
        print(error)
    finally:
        sha = None
        if exit_code == 0 and Path(checkpoint).exists():
            sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        record = {
            "exit_code": exit_code,
            "error": error,
            "checkpoint": str(Path(checkpoint).resolve().relative_to(_ROOT))
            if Path(checkpoint).resolve().is_relative_to(_ROOT) else str(checkpoint),
            "checkpoint_sha256": sha,
            "log": str(log_path.relative_to(_ROOT)) if log_path.is_relative_to(_ROOT) else str(log_path),
            "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "argv": sys.argv,
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        }
        print(f"[launch-guard] exit_code {exit_code}")
        sys.stdout, sys.stderr = out, err
        fh.close()
        marker.write_text(json.dumps(record, indent=2) + "\n")
    return exit_code


def main() -> None:
    global ATTEMPT, SAVE_PATH
    parser = argparse.ArgumentParser(
        description="Train LE-PINN on the Sajben WIND dataset (pre-registered attempt)"
    )
    parser.add_argument("--attempt", type=int, default=1, choices=sorted(ATTEMPTS),
                        help="registered attempt to run (see ATTEMPTS)")
    parser.add_argument("--epochs", type=int, default=ATTEMPT["n_epochs"],
                        help=f"Number of training epochs (default: {ATTEMPT['n_epochs']})")
    parser.add_argument("--lr", type=float, default=ATTEMPT["lr"],
                        help=f"Learning rate (default: {ATTEMPT['lr']})")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device (default: auto -> cpu)")
    parser.add_argument("--seed", type=int, default=ATTEMPT["seed"],
                        help=f"RNG seed (default: {ATTEMPT['seed']})")
    parser.add_argument("--physics-weight", type=float, default=ATTEMPT["physics_loss_weight"],
                        dest="physics_weight",
                        help=f"Physics loss weight (default: {ATTEMPT['physics_loss_weight']})")
    parser.add_argument("--physics-debug", action="store_true",
                        help="Print per-term normalized physics residual losses during training")
    parser.add_argument("--out", type=str, default=SAVE_PATH,
                        help=f"Checkpoint path (default: {SAVE_PATH}); existing files are not overwritten")
    parser.add_argument("--attempt-id", type=str, default=ATTEMPT["id"],
                        help="Attempt label recorded in the checkpoint")
    parser.add_argument("--log", type=str, default=None,
                        help="Run log (tee'd) with a .done exit-code marker beside it. Default for "
                             "attempt 3: outputs/logs/train_sajben_v5_a3[_dataonly]_s{seed}.log; "
                             "'-' disables the guard (smoke runs only)")
    args = parser.parse_args()
    ATTEMPT = ATTEMPTS[args.attempt]
    out_key = "out_dataonly" if (args.physics_weight == 0 and "out_dataonly" in ATTEMPT) else "out"
    SAVE_PATH = str(_ROOT / ATTEMPT[out_key].format(seed=args.seed))
    if args.attempt != 1:
        # defaults above were bound to attempt 1; rebind the registered values
        if args.attempt_id == ATTEMPTS[1]["id"]:
            args.attempt_id = ATTEMPT["id"] + ("-dataonly" if out_key == "out_dataonly" else "")
        if args.out == str(_ROOT / ATTEMPTS[1]["out"]):
            args.out = SAVE_PATH
    if "seeds" in ATTEMPT and args.seed not in ATTEMPT["seeds"]:
        print(f"NOTE: seed {args.seed} is not one of the registered seeds {ATTEMPT['seeds']}.")

    registered = (args.epochs == ATTEMPT["n_epochs"] and args.lr == ATTEMPT["lr"]
                  and args.physics_weight in (ATTEMPT["physics_loss_weight"], 0.0)
                  and args.seed in ATTEMPT.get("seeds", [ATTEMPT["seed"]]))
    if not registered and args.attempt_id == ATTEMPT["id"]:
        print(f"NOTE: hyperparameters differ from the registered {ATTEMPT['id']}; "
              "pass --attempt-id to label this run as a new attempt.")

    def run() -> None:
        model, history = train_sajben_le_pinn(
            n_epochs=args.epochs,
            lr=args.lr,
            save_path=args.out,
            device=resolve_device(args.device),
            physics_loss_weight=args.physics_weight,
            physics_debug=args.physics_debug,
            seed=args.seed,
            attempt_id=args.attempt_id,
            verbose=True,
        )
        print(f"\nEpochs run: {len(history['loss_total'])}  "
              f"final data loss {history['loss_data'][-1]:.3e}  "
              f"best val {min(history['val_loss']):.3e}  "
              f"min alive units {min(history['alive_min'])}")
        print("Next: python scripts/validation/sajben_validation.py --model", args.out)

    log = args.log
    if log is None and ATTEMPT.get("terminal"):
        log = str(default_log_path(args.attempt, args.seed, out_key == "out_dataonly"))
    if log is None or log == "-":
        run()
        return
    sys.exit(guarded_run(Path(log).resolve(), args.out, run))


if __name__ == "__main__":
    main()
