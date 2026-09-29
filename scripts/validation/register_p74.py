#!/usr/bin/env python3
"""
P7.4 — write the registration of the new LE-PINN study BEFORE any training:
``outputs/phase7/p74_split.json`` (held-out WIND test rows, the training pool,
nested per-seed subsets, all as WIND grid indices g = j*81 + i) and
``outputs/phase7/p74_registration.json`` (design, runs, hyperparameters,
scoring, claim rule, compute budget, commands). Both are write-once.

Usage: .venv/bin/python scripts/validation/register_p74.py [--benchmark-json PATH] [--dry-run DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from simulation.nozzle import le_pinn_ma as lm  # noqa: E402

SEEDS = (42, 43, 44)
FRACTIONS = {"f002": 0.02, "f005": 0.05, "f010": 0.10, "f025": 0.25, "f100": 1.00}
ARMS = ("physics", "dataonly")
TEST_FRACTION = 0.20
SPLIT_SEED = 7401
SUBSET_SEED_OFFSET = 7410

TRAINING = {
    "epochs": 5000,                       # established repo budget (P4.3 attempts 1-3)
    "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-5,
    "lr_schedule": "CosineAnnealingLR(T_max = epochs, eta_min = lr * eta_min_factor), stepped per epoch",
    "eta_min_factor": 0.01, "grad_clip": 1.0,
    "batching": "full batch over the run's training rows every epoch",
    "n_collocation": 3264,                # = the 100 % pool size, the P4.3 attempt-3 physics point count
    "collocation_batch": None,            # None = the whole collocation pool every epoch
    "collocation_seed_offset": 7500, "collocation_batch_seed_offset": 7600,
    "weight_eps": 1e-8,
    "checkpoint_every": 250, "history_every": 10, "log_every": 50,
    "checkpoint_selection": "final epoch; no validation set, no early stopping, no held-out or experimental data",
    "restart_policy": "infrastructure restarts resume from the last atomic checkpoint (model, optimizer, "
                      "scheduler, RNG states) with an identical configuration hash; a changed configuration "
                      "is refused; a non-finite loss or any other error ends the run as FAILED with no retry",
}
MODEL = {"architecture": "dual network (global 2->400, 6 hidden x 400 -> 6; boundary 2->100, 6 hidden x 100 -> 2 "
                         "(p, T) fused within fusion_delta of a wall), tanh, Xavier-uniform weights, zero biases",
         "width": 400, "n_hidden": 6, "b_width": 100, "b_hidden": 6, "fusion_delta_m": lm.FUSION_DELTA,
         "activation": "tanh",
         "inputs": "physical (x, y), normalised inside the model to [-1, 1] by the GEOMETRY bounding box "
                   "(x_min..x_max, 0..y_max of the grid coordinates; no flow data)",
         "outputs": "standardised [rho, u, v, p, T, log1p(mu_t / mu_ref)]; mean/std from the run's TRAINING rows "
                    "only (shared by the paired arms, which use the same rows); mu_t = mu_ref expm1(.) clamped >= 0"}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(compute: dict, benchmark: dict | None, out_dir: Path, training: dict) -> tuple[dict, dict]:
    sol = lm.load_wind()
    nj, ni = sol.nj, sol.ni
    sp = lm.make_split(nj, ni, TEST_FRACTION, SPLIT_SEED)
    subsets = {}
    for s in SEEDS:
        sub = lm.nested_subsets(sp["pool"], list(FRACTIONS.values()), SUBSET_SEED_OFFSET + s)
        subsets[str(s)] = {k: [int(i) for i in sub[f]] for k, f in FRACTIONS.items()}
    split = {
        "description": "P7.4 WIND row split (grid index g = j*ni + i, ni = 81; inflow column i = 0 excluded)",
        "ni": ni, "nj": nj, "n_rows": int(len(sp["test"]) + len(sp["pool"])),
        "test_fraction": TEST_FRACTION, "split_seed": SPLIT_SEED,
        "test": [int(i) for i in sp["test"]], "pool": [int(i) for i in sp["pool"]],
        "subset_rule": "per seed s: numpy default_rng(7410 + s).permutation(pool); fraction f uses the first "
                       "round(f * 3264) rows (nested by construction)",
        "subsets": subsets,
    }
    split_path = out_dir / "p74_split.json"
    runs = []
    order = [(fk, s, arm) for fk in FRACTIONS for s in SEEDS for arm in ARMS]
    for fk, s, arm in order:
        rid = f"p74_s{s}_{fk}_{arm}"
        ids = subsets[str(s)][fk]
        runs.append({
            "run_id": rid, "seed": s, "init_seed": s, "fraction": FRACTIONS[fk], "fraction_key": fk, "arm": arm,
            "n_train_rows": len(ids), "train_ids_sha256": lm.ids_hash(ids),
            "checkpoint": f"models/le_pinn_ma_p74_s{s}_{fk}_{arm}.pt",
            "record": f"outputs/phase7/p74_runs/{rid}.json",
        })
    inputs = ["data/raw/cfd_datasets/nasa/transdif01/sajben.cgd", "data/raw/cfd_datasets/nasa/transdif01/sajben.cfl",
              "data/raw/data.Mach46.txt", "data/raw/sajben.x.fmt", "simulation/nozzle/wind_cff.py"]
    reg = {
        "phase": "P7.4", "registered": "2026-09-27",
        "study": "P7.4 new LE-PINN study: Ma-form RANS residual + current-loss weights, data efficiency "
                 "(NOT P4.3 attempt 4; nothing is promoted to the production cycle)",
        "registered_before": "any training or scoring of this study. Known: P4.3 attempts 1-3 and their "
                             "diagnoses, the P6.5 audit. Compute-only benchmark (timing, no scores) before "
                             "the budget below was fixed (see compute.benchmark).",
        "data_access": {
            "question": "sent to the user by the interactive session before this dispatch (viscosity access)",
            "answer_recorded": "no answer is present in the repository, the plan or this executor's input "
                               "(headless dispatch, started 2026-09-27 11:16, cannot receive messages); the plan's "
                               "stated recommended assumption is therefore applied",
            "assumption": "every WIND field label, including viscosity (mu_l, mu_t), is available ONLY at the "
                          "sampled training rows, identically for both arms. The physics arm gets no held-out "
                          "viscosity or derivative. mu_t is a network output trained on WIND mu_t at the training "
                          "rows only (a training-only learned viscosity; information budget = the run's rows, the "
                          "same as the data-only arm, which learns the same six outputs). Molecular viscosity is "
                          "Sutherland's law of the predicted T (a closure, not a label). Geometry (grid "
                          "coordinates, wall shape and normals), the WIND file's freestream reference constants "
                          "and the physical wall conditions are not held-out flow measurements.",
        },
        "data": {
            "source": "WIND S-A RANS solution of the Sajben weak-shock case (sajben.cgd/.cfl, NPARC archive): one "
                      "flow case; fields rho, u, v, p, T, mu_l, mu_t; no measured Reynolds stresses, no k",
            "rows": "every grid node except the inflow column i = 0: 80 x 51 = 4080",
            "heldout_test": f"fixed {int(TEST_FRACTION*100)} % of the 4080 rows ({len(sp['test'])} rows), numpy "
                            f"default_rng({SPLIT_SEED}).permutation; never used in training, scalers, selection "
                            "or stopping",
            "denominator_100pct": f"the {len(sp['pool'])}-row training POOL = all 4080 rows minus the held-out test "
                                  "rows. Fractions 2, 5, 10, 25, 100 % of this pool: "
                                  + ", ".join(f"{k}={len(subsets['42'][k])}" for k in FRACTIONS) + " rows",
            "validation": "none registered (final-epoch checkpoint)",
            "experiment": "data/raw/data.Mach46.txt wall pressures: external scoring only; never used to "
                          "select, checkpoint or stop",
            "labels_used": "rho, u, v, p, T, mu_t at the run's training rows (mu_l is not used)",
        },
        "split": {"file": "outputs/phase7/p74_split.json", "sha256": None,
                  "test_sha256": lm.ids_hash(sp["test"]), "pool_sha256": lm.ids_hash(sp["pool"])},
        "design": {"fractions": FRACTIONS, "arms": list(ARMS), "seeds": list(SEEDS), "n_runs": len(runs),
                   "pairing": "for each (seed, fraction) the physics and data-only runs share the training rows, "
                              "scalers, initial weights (torch.manual_seed(seed) immediately before model "
                              "construction), architecture, optimizer, schedule and epoch budget; the only "
                              "difference is the loss",
                   "arms": {"physics": "L = lambda_data L_data + lambda_phys L_phys + lambda_bc L_bc with the "
                                       "detached current-loss weights (le_pinn_ma.ma_weights)",
                            "dataonly": "L = L_data (weight 1)"}},
        "physics": {
            "equations": "le_pinn_ma module docstring (planar steady compressible RANS, conservative fluxes, "
                         "variable mu_eff = Sutherland(T) + mu_t, Boussinesq stress combined with the molecular "
                         "stress in ONE flux, heat flux cp (mu/Pr + mu_t/Pr_t) grad T, EOS p = rho R T)",
            "Pr": lm.PR, "Pr_t": lm.PR_T, "sutherland": {"C1": lm.SUTH_C1, "S_K": lm.SUTH_S},
            "gamma_R": "from the WIND reference record (gamma 1.4, R 286.96)",
            "not_modelled": "isotropic 2/3 rho k (S-A gives no k; k is not in H either): an approximation, not an "
                            "exact Ma replication",
            "derivatives": "autograd w.r.t. physical coordinates (exact chain rule through the input "
                           "normalisation and output de-standardisation)",
            "scaling": "residuals / (rho_r a_r / L), (rho_r a_r^2 / L), (rho_r a_r^3 / L), (rho_r a_r^2 for EOS); "
                       "rho_r, a_r, T_r = WIND reference; L = throat height",
            "L_phys": "sum over {continuity, x_mom, y_mom, energy, eos} of the mean squared scaled residual over "
                      "the collocation points",
            "L_bc": "mean (u/a_r)^2 + mean (v/a_r)^2 + mean ((L/T_r) dT/dn)^2 over both walls' nodes (i >= 1); "
                    "n = unit normal of the actual wall from its coordinates (upper wall curved)",
            "L_data": "mean squared error of the six standardised outputs over the training rows",
            "weights": "Ma et al. (2026) Eqs. 30-33 (p. 5) in the forms fixed by the Phase 7 plan: "
                       "lambda_data = .1 + .9 sigmoid((Lphys + Lbc - Ldata)/(Ldata + eps)); "
                       "lambda_phys = .1 + .9 sigmoid((Ldata - Lphys)/(Lphys + eps)); "
                       "lambda_bc = .1 + .9 sigmoid((Ldata - Lbc)/(Lbc + eps)); eps = 1e-8; recomputed every "
                       "epoch from the current losses and DETACHED (no gradient through the weights)",
            "collocation": "fixed pool per seed: uniform draws in continuous grid-index space (xi in [1, 80], "
                           "eta in [0, 50]) mapped by bilinear interpolation of the grid coordinates "
                           "(le_pinn_ma.collocation_points, default_rng(7500 + seed)); follows the grid's wall "
                           "clustering; geometry only",
            "boundary_points": "the 80 lower-wall and 80 upper-wall grid nodes i = 1..80",
        },
        "model": MODEL, "training": training,
        "work_dir": "outputs/phase7/p74_work",
        "runs": runs,
        "scoring": {
            "script": "scripts/validation/report_sajben_ma.py (runs only when all 30 runs are complete)",
            "experimental": "upper and lower wall P/P_in shape-L2 against data.Mach46.txt with the existing "
                            "scorer (scripts/validation/sajben_validation.py: build_sajben_grid 60 x 25 from "
                            "sajben.x.fmt, compute_wall_cp_errors with the throat-height mapping), existing bands "
                            "< 0.10 pass / 0.10-0.25 partial / > 0.25 fail; degenerate (flat) predictions flagged",
            "primary_scalar": "worse of the two walls: max(l2_upper, l2_lower)",
            "wind_heldout": "on the 816 held-out WIND rows: relative L2 ||pred - ref|| / ||ref|| and RMSE for each "
                            "of rho, u, v, p, T; declared aggregate = mean of the five relative L2; viscosity "
                            "separately: relative L2 and RMSE of mu_t and of mu_eff = Sutherland(T_pred) + mu_t "
                            "vs WIND mu_l + mu_t",
            "labels": "WIND held-out = in-case spatial interpolation on the same flow case; experiment = external "
                      "comparison on the same geometry and condition; no cross-case or cross-geometry claim",
        },
        "claim": {
            "per_fraction": "three seed scores per arm on the PRIMARY scalar (worse-wall experimental shape-L2)",
            "seed_spread": "max(sampleSD(physics), sampleSD(data-only)), ddof = 1, per fraction",
            "benefit": "mean(data-only primary) - mean(physics primary) > seed spread, at >= 2 of the fractions "
                       "{2, 5, 10} %",
            "reported_regardless": "per-seed paired differences, mean/SD per arm, all five fractions, both the "
                                   "experimental and the WIND held-out evaluations, bands, degenerate flags, "
                                   "final training-side losses, failures",
            "completeness": "a terminal claim requires all 30 registered runs COMPLETE (exit 0, record, checkpoint "
                            "hash); the report refuses otherwise",
            "no_selection": "no wall, metric, seed, checkpoint, fraction or run is chosen after outcomes; no "
                            "held-out score chooses hyperparameters, splits, stopping rules or winning runs",
            "failure": "a new-study failure is a result, not permission for another unregistered attempt",
        },
        "compute": compute,
        "report_outputs": ["outputs/phase7/p74_scores.csv", "outputs/phase7/p74_report.json",
                           "outputs/phase7/p74_report.md"],
        "commands": {
            "launch": "bash scripts/run_phase7.sh launch pinn",
            "run_in_snapshot": ".venv/bin/python scripts/validation/train_sajben_ma.py --run-id <run_id>",
            "report_in_snapshot": ".venv/bin/python scripts/validation/report_sajben_ma.py",
            "queue_order": [r["run_id"] for r in runs] + ["p74_report"],
            "inspect": "bash scripts/run_phase7.sh status; outputs/phase7/runs/pinn/<run_id>/attempt_<n>.log",
        },
        "inputs_sha256": {p: sha256(ROOT / p) for p in inputs},
    }
    if benchmark is not None:
        reg["compute"]["benchmark"] = benchmark
    return reg, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--max-concurrent", type=int, required=True)
    ap.add_argument("--collocation-batch", type=int, default=None)
    ap.add_argument("--benchmark-json", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase7"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    training = dict(TRAINING, collocation_batch=a.collocation_batch)
    compute = {"device": "cpu", "dtype": "float32", "torch_threads_per_run": a.threads,
               "max_concurrent_runs": a.max_concurrent,
               "max_threads": a.threads * a.max_concurrent,
               "note": "the supervisor never runs more than max_threads torch threads at once; the v6 "
                       "calibration/blend queues use 6 process workers of their own"}
    bench = json.loads(Path(a.benchmark_json).read_text()) if a.benchmark_json else None
    reg, split = build(compute, bench, out_dir, training)
    split_path = out_dir / "p74_split.json"
    with open(split_path, "x") as fh:
        fh.write(json.dumps(split) + "\n")
    reg["split"]["file"] = str(split_path.resolve().relative_to(ROOT)) if split_path.resolve().is_relative_to(ROOT) \
        else str(split_path.resolve())
    reg["split"]["sha256"] = sha256(split_path)
    with open(out_dir / "p74_registration.json", "x") as fh:
        fh.write(json.dumps(reg, indent=2) + "\n")
    print(json.dumps({k: reg[k] for k in ("compute",)}, indent=2))
    print({r["run_id"]: r["n_train_rows"] for r in reg["runs"][:10]})


if __name__ == "__main__":
    main()
