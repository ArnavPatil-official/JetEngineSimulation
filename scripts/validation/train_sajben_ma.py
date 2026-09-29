#!/usr/bin/env python3
"""
P7.4 — train ONE registered run of the new LE-PINN study (Ma-form RANS residual,
current-loss weights) on a nested subset of the WIND Sajben training pool.

Every setting comes from ``outputs/phase7/p74_registration.json``; the command
line only names the run. Nothing is scored here: the held-out WIND rows and the
experiment are never loaded into training (the WIND field is reduced to the
run's training rows before anything else happens), and the final-epoch weights
are the registered checkpoint (no validation-based selection, no early stop).

Durability: resumable checkpoints are written atomically every
``checkpoint_every`` epochs under ``outputs/phase7/p74_work/<run_id>/`` (model,
optimizer, scheduler, RNG states, loss history). A restart with the same
registered configuration resumes from the last one and continues the identical
trajectory; a checkpoint with a different configuration hash is refused.
The final checkpoint (models/...) and the run record (outputs/phase7/p74_runs/)
are write-once; the record carries the checkpoint SHA-256.

Usage:
    python scripts/validation/train_sajben_ma.py --run-id p74_s42_f002_physics
    python scripts/validation/train_sajben_ma.py --run-id ... --smoke-epochs 3 --out-root /tmp/x
        (NON-STUDY implementation check: temporary outputs only)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from simulation.nozzle import le_pinn_ma as lm  # noqa: E402

REGISTRATION = ROOT / "outputs" / "phase7" / "p74_registration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registration(path: Path = REGISTRATION) -> dict:
    reg = json.loads(Path(path).read_text())
    if reg.get("phase") != "P7.4":
        raise ValueError(f"{path} is not the P7.4 registration")
    for rel, h in reg["inputs_sha256"].items():
        if sha256(ROOT / rel) != h:
            raise RuntimeError(f"registered input {rel} changed since registration")
    return reg


def run_spec(reg: dict, run_id: str) -> dict:
    for r in reg["runs"]:
        if r["run_id"] == run_id:
            return r
    raise SystemExit(f"{run_id} is not a registered P7.4 run")


def load_split(reg: dict) -> dict:
    split = json.loads((ROOT / reg["split"]["file"]).read_text())
    if sha256(ROOT / reg["split"]["file"]) != reg["split"]["sha256"]:
        raise RuntimeError("split file changed since registration")
    return split


def build_training_batch(reg: dict, run: dict, sol=None) -> dict:
    """Everything the optimiser may see for one run: geometry, the run's
    training rows (coordinates + six labels), training-only scalers, the
    collocation pool and the wall points. The held-out rows are never
    extracted; the full field object is dropped before returning."""
    split = load_split(reg)
    sol = sol if sol is not None else lm.load_wind()
    geom = lm.geometry_from(sol)
    train_ids = np.asarray(split["subsets"][str(run["seed"])][run["fraction_key"]], dtype=np.int64)
    test_ids = np.asarray(split["test"], dtype=np.int64)
    if np.intersect1d(train_ids, test_ids).size:
        raise RuntimeError("training rows intersect the held-out WIND test set")
    if lm.ids_hash(train_ids) != run["train_ids_sha256"]:
        raise RuntimeError("training subset differs from the registration")
    tab = lm.label_table(sol, train_ids)
    del sol
    scalers = lm.fit_output_scalers(tab, geom.ref["mu_ref"])
    tr = reg["training"]
    colloc = lm.collocation_points(geom, tr["n_collocation"], tr["collocation_seed_offset"] + run["seed"])
    wxy, wn = lm.wall_points(geom)
    return {
        "geom": geom, "train_ids": train_ids, "scalers": scalers,
        "xy": torch.tensor(tab[:, :2], dtype=torch.float32),
        "labels": torch.tensor(tab[:, 2:8], dtype=torch.float32),
        "colloc": torch.tensor(colloc, dtype=torch.float32),
        "wall_xy": torch.tensor(wxy, dtype=torch.float32),
        "wall_n": torch.tensor(wn, dtype=torch.float32),
    }


def run_config(reg: dict, run: dict) -> dict:
    """What must match for a resume to continue the same trajectory."""
    return {"run": run, "training": reg["training"], "model": reg["model"],
            "split_sha256": reg["split"]["sha256"]}


def make_model(batch: dict, reg: dict, seed: int) -> lm.MaLEPINN:
    torch.manual_seed(seed)
    m = reg["model"]
    return lm.MaLEPINN(batch["geom"], batch["scalers"], width=m["width"], n_hidden=m["n_hidden"],
                       b_width=m["b_width"], b_hidden=m["b_hidden"], delta=m["fusion_delta_m"])


def losses(model, batch, physics: bool, colloc_idx=None, want_phys: bool | None = None) -> dict:
    geom = batch["geom"]
    L_data = lm.data_loss(model, batch["xy"], batch["labels"])
    out = {"data": L_data}
    if physics if want_phys is None else want_phys:
        xy = batch["colloc"] if colloc_idx is None else batch["colloc"][colloc_idx]
        xy = xy.clone().requires_grad_(True)
        res = lm.rans_residuals(xy, model.fields, geom.ref, geom.L_ref)
        out["phys"] = lm.physics_loss(res)
        out["phys_terms"] = {k: float(torch.mean(v.detach() ** 2)) for k, v in res.items()}
        wxy = batch["wall_xy"].clone().requires_grad_(True)
        out["bc"], bt = lm.wall_bc_loss(wxy, batch["wall_n"], model.fields, geom.ref, geom.L_ref)
        out["bc_terms"] = {k: float(v.detach()) for k, v in bt.items()}
    return out


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def source_commit() -> dict:
    snap = ROOT / ".snapshot.json"
    if snap.exists():
        return json.loads(snap.read_text())
    import subprocess
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return {"commit": sha, "working_tree_dirty": dirty, "note": "not a frozen snapshot"}


def train(run_id: str, reg_path: Path = REGISTRATION, out_root: Path = ROOT,
          smoke_epochs: int | None = None, sol=None, verbose: bool = True) -> dict:
    reg = load_registration(reg_path)
    run = run_spec(reg, run_id)
    tr = reg["training"]
    n_epochs = smoke_epochs if smoke_epochs is not None else tr["epochs"]
    physics = run["arm"] == "physics"
    threads = int(os.environ.get("P7_TORCH_THREADS", reg["compute"]["torch_threads_per_run"]))
    torch.set_num_threads(threads)
    ckpt_path = Path(out_root) / run["checkpoint"]
    rec_path = Path(out_root) / run["record"]
    work = Path(out_root) / reg["work_dir"] / run_id
    if ckpt_path.exists() or rec_path.exists():
        raise SystemExit(f"{run_id}: final checkpoint or record already exists (write-once)")
    cfg = run_config(reg, run)
    if smoke_epochs is not None:
        cfg = dict(cfg, smoke_epochs=smoke_epochs)
    chash = lm.config_hash(cfg)

    batch = build_training_batch(reg, run, sol=sol)
    model = make_model(batch, reg, run["init_seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1),
                                                       eta_min=tr["lr"] * tr["eta_min_factor"])
    gen = torch.Generator().manual_seed(tr["collocation_batch_seed_offset"] + run["seed"])
    hist: dict[str, list] = {k: [] for k in ("epoch", "L_data", "L_phys", "L_bc", "lam_data",
                                             "lam_phys", "lam_bc", "lr", "grad_norm")}
    start_epoch, resumed_from = 0, None
    latest = work / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["config_hash"] != chash:
            raise SystemExit(f"{latest}: configuration hash differs from the registration; refusing to resume")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        hist = ck["history"]
        start_epoch, resumed_from = ck["epoch"], ck["epoch"]
        if verbose:
            print(f"[resume] {run_id} from epoch {start_epoch} ({latest})", flush=True)
    t0 = time.time()
    n_c = batch["colloc"].shape[0]
    cb = tr["collocation_batch"]
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    for epoch in range(start_epoch, n_epochs):
        model.train()
        idx = None
        if physics and cb is not None and cb < n_c:
            idx = torch.randperm(n_c, generator=gen)[:cb]
        L = losses(model, batch, physics, idx)
        if physics:
            lam = lm.ma_weights(L["data"], L["phys"], L["bc"], tr["weight_eps"])
            total = lam["data"] * L["data"] + lam["phys"] * L["phys"] + lam["bc"] * L["bc"]
        else:
            lam = None
            total = L["data"]
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at epoch {epoch}: registered failure, no retry")
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tr["grad_clip"]))
        opt.step()
        sched.step()
        if epoch % tr["history_every"] == 0 or epoch == n_epochs - 1:
            hist["epoch"].append(epoch)
            hist["L_data"].append(float(L["data"]))
            hist["L_phys"].append(float(L["phys"]) if physics else None)
            hist["L_bc"].append(float(L["bc"]) if physics else None)
            for k in ("data", "phys", "bc"):
                hist[f"lam_{k}"].append(float(lam[k]) if lam else None)
            hist["lr"].append(float(opt.param_groups[0]["lr"]))
            hist["grad_norm"].append(gn)
        if verbose and (epoch % tr["log_every"] == 0 or epoch == n_epochs - 1):
            ph = (f" L_phys {float(L['phys']):.3e} L_bc {float(L['bc']):.3e} "
                  f"lam {float(lam['data']):.3f}/{float(lam['phys']):.3f}/{float(lam['bc']):.3f}") if physics else ""
            print(f"[{run_id}] epoch {epoch} L_data {float(L['data']):.4e}{ph} "
                  f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % tr["checkpoint_every"] == 0 and epoch + 1 < n_epochs:
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scheduler": sched.state_dict(), "generator": gen.get_state(),
                         "torch_rng": torch.get_rng_state(), "history": hist, "epoch": epoch + 1,
                         "config_hash": chash, "saved": time.time()}, latest)

    # final-epoch diagnostics on training-side points only (no held-out data)
    final = losses(model, batch, physics, None, want_phys=True)
    record = {
        "run_id": run_id, "phase": "P7.4", "study": reg["study"], "arm": run["arm"], "seed": run["seed"],
        "fraction": run["fraction"], "n_train_rows": int(len(batch["train_ids"])),
        "train_ids_sha256": lm.ids_hash(batch["train_ids"]), "config_hash": chash,
        "epochs": n_epochs, "smoke": smoke_epochs is not None, "resumed_from_epoch": resumed_from,
        "started": started, "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "wall_seconds_this_process": time.time() - t0, "torch_threads": threads,
        "torch": torch.__version__, "source": source_commit(),
        "registration_sha256": sha256(reg_path),
        "final_training_side": {"L_data": float(final["data"]), "L_phys": float(final["phys"]),
                                "L_bc": float(final["bc"]), "phys_terms": final["phys_terms"],
                                "bc_terms": final["bc_terms"],
                                "note": "evaluated on training rows, the full collocation pool and the wall points"},
        "checkpoint_selection": "final epoch (registered); no validation or held-out data used",
        "history": hist,
    }
    payload = {"model_state_dict": model.state_dict(), "scalers": batch["scalers"],
               "model_config": reg["model"], "record": {k: v for k, v in record.items() if k != "history"},
               "geometry_ref": batch["geom"].ref, "L_ref": batch["geom"].L_ref}
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
    torch.save(payload, tmp)
    if ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} appeared during the run; refusing to overwrite")
    os.replace(tmp, ckpt_path)
    record["final_checkpoint"] = str(run["checkpoint"])
    record["final_checkpoint_sha256"] = sha256(ckpt_path)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rec_path, "x") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
    if verbose:
        print(f"[{run_id}] done: final L_data {record['final_training_side']['L_data']:.4e} "
              f"checkpoint {ckpt_path} sha256 {record['final_checkpoint_sha256'][:12]}", flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--registration", default=str(REGISTRATION))
    ap.add_argument("--smoke-epochs", type=int, default=None,
                    help="NON-STUDY implementation check; requires --out-root outside the repo outputs")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    if a.smoke_epochs is not None and a.out_root is None:
        raise SystemExit("--smoke-epochs requires --out-root (temporary outputs)")
    train(a.run_id, Path(a.registration), Path(a.out_root) if a.out_root else ROOT, a.smoke_epochs)


if __name__ == "__main__":
    main()
