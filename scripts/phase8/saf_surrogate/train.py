"""Fixed paired MLX/Torch training; target readers are explicitly split-restricted."""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

from .inputs import feature_rows
from .models import forward64, make_model, output_map
from .registration import append_progress, read_json, sha256_file, write_once
from .thermo import Thermo

try:
    import resource
except ImportError:  # Windows: peak RSS is an optional POSIX metric.
    resource = None

DATASETS = {
    "train": ("teacher_rows.csv", "teacher_species.npz"),
    "validation": ("validation_teacher_rows.csv", "validation_teacher_species.npz"),
    "test": ("sealed/test_teacher_rows.csv", "sealed/test_teacher_species.npz"),
    "ranking_test": ("sealed/ranking_teacher_rows.csv", "sealed/ranking_teacher_species.npz"),
}


def read_dataset(output, name, *, score_reservation=None):
    import numpy as np
    if name not in ("train", "validation") and score_reservation is None:
        raise ValueError("Locked dataset access requires the sole score reservation")
    output = Path(output)
    if score_reservation is not None:
        if read_json(output / "score_reservation.json") != score_reservation:
            raise ValueError("Foreign score reservation")
    csv_path, species_path = (output / path for path in DATASETS[name])
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    queries=read_json(output/f"splits/{name}.json")
    if len(rows)!=len(queries):
        raise ValueError("Teacher query coverage differs from frozen manifest")
    for row,query in zip(rows,queries):
        if any(str(row[key])!=str(query[key]) for key in ("design_id","draw_id","prefix_index","input_sha256")):
            raise ValueError("Teacher rows differ from frozen input identities")
    with np.load(species_path, allow_pickle=False) as archive:
        species = np.asarray(archive["Y4"], dtype=np.float64)
        proof = {key: archive[key].copy() for key in ("species_order", "design_id", "draw_id", "prefix_index", "input_sha256")}
    if species.ndim != 2 or species.shape[1] != 492:
        raise ValueError("Corrupt product composition dimensions")
    if proof["species_order"].tolist()!=read_json(output/"frozen_properties.json")["species_order"]:
        raise ValueError("Product species order differs from frozen mechanism")
    used=[]
    ff, T4, Y = [], [], []
    for row in rows:
        if row["status"] != "converged":
            ff.append(float("nan")); T4.append(float("nan")); Y.append(np.full(492, np.nan))
            continue
        index = int(row["species_row_index"])
        used.append(index)
        for key in ("design_id", "draw_id", "input_sha256"):
            if str(proof[key][index]) != row[key]:
                raise ValueError("Species/teacher row identity mismatch")
        if int(proof["prefix_index"][index]) != int(row["prefix_index"]):
            raise ValueError("Species prefix alignment mismatch")
        fuel, temperature = float(row["ff_kg_s"]), float(row["T4_K"])
        mass = species[index]
        if not np.isfinite(fuel) or fuel <= 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid positive target")
        if not np.isfinite(mass).all() or (mass < 0).any() or abs(float(mass.sum())-1) > 1e-10:
            raise ValueError("Invalid teacher composition")
        ff.append(fuel); T4.append(temperature); Y.append(mass)
    if used!=list(range(len(species))):
        raise ValueError("Species rows duplicated, missing or unexpectedly reordered")
    return rows, np.asarray(ff), np.asarray(T4), np.asarray(Y), proof


def monotonic_endpoints(queries):
    plus, minus, widths = [], [], []
    for query in queries:
        x = query["thrust_fraction"]
        if x < .30:
            lo, hi = .07, .30
        elif x < .85:
            lo, hi = .30, .85
        else:
            lo, hi = .85, 1.0
        a, b = max(lo, x-.0005), min(hi, x+.0005)
        if b <= a:
            raise ValueError("Invalid monotonic endpoint separation")
        minus.append(dict(query, thrust_fraction=a))
        plus.append(dict(query, thrust_fraction=b))
        widths.append(b-a)
    return minus, plus, widths


def prediction_metrics(ff, T4, Y, true_ff, true_T4, true_Y, thermo):
    import numpy as np
    ff,T4,Y,true_ff,true_T4,true_Y=[np.asarray(value,dtype=np.float64) for value in (ff,T4,Y,true_ff,true_T4,true_Y)]
    valid=(np.isfinite(ff)&np.isfinite(T4)&np.isfinite(true_ff)&np.isfinite(true_T4)&
        np.isfinite(Y).all(axis=-1)&np.isfinite(true_Y).all(axis=-1)&(ff>0)&(true_ff>0)&
        (Y>=0).all(axis=-1)&(true_Y>=0).all(axis=-1))
    lo,hi=float(thermo.bounds[:,0].max()),float(thermo.bounds[:,2].min())
    valid &= (T4>=lo)&(T4<=hi)&(true_T4>=lo)&(true_T4<=hi)
    indices=np.flatnonzero(valid)
    if len(indices):
        for temperature,mass in ((T4,Y),(true_T4,true_Y)):
            cp=np.sum(mass[indices]*thermo.species_cp(temperature[indices]),axis=-1)
            gas_R=thermo.Ru*np.sum(mass[indices]/thermo.mw,axis=-1)
            valid[indices] &= np.isfinite(cp)&np.isfinite(gas_R)&(cp>gas_R)&(gas_R>0)
    summary={"state":"SCORED" if valid.all() else "INVALID", "complete":bool(valid.all()),
             "requested_rows":len(valid),"conditional_valid_rows":int(valid.sum()),"invalid_rows":int((~valid).sum())}
    if not valid.any():
        return summary
    ff,T4,Y,true_ff,true_T4,true_Y=[value[valid] for value in (ff,T4,Y,true_ff,true_T4,true_Y)]
    ff_error, T_error = np.abs(ff-true_ff), np.abs(T4-true_T4)
    species_error = np.abs(Y-true_Y).sum(axis=-1)
    pred, true = thermo.mixture(T4, Y), thermo.mixture(true_T4, true_Y)
    return {**summary, "ff_MAE_kg_s": float(ff_error.mean()),
        "ff_max_kg_s": float(ff_error.max()), "ff_relative_MAE": float((ff_error/true_ff).mean()),
        "ff_relative_max": float((ff_error/true_ff).max()), "T4_MAE_K": float(T_error.mean()),
        "T4_max_K": float(T_error.max()), "composition_mean_L1": float(species_error.mean()),
        "composition_max_L1": float(species_error.max()),
        "cp_relative_max": float(np.max(np.abs(pred["cp"]-true["cp"])/true["cp"])),
        "R_relative_max": float(np.max(np.abs(pred["R"]-true["R"])/true["R"])),
        "gamma_absolute_max": float(np.max(np.abs(pred["gamma"]-true["gamma"])))}


def validation_pass(metrics, reg):
    limits = reg["validation_selection"]
    if not metrics.get("complete"):
        return False
    checks = (("ff_relative_MAE", "ff_relative_MAE_max"), ("ff_relative_max", "ff_relative_absolute_max"),
              ("T4_MAE_K", "T4_MAE_K_max"), ("T4_max_K", "T4_absolute_max_K"),
              ("composition_mean_L1", "composition_mean_L1_max"), ("composition_max_L1", "composition_max_L1_max"),
              ("cp_relative_max", "auxiliary_cp_relative_max"), ("R_relative_max", "auxiliary_R_relative_max"),
              ("gamma_absolute_max", "auxiliary_gamma_absolute_max"))
    return all(metrics[key] <= limits[limit] for key, limit in checks)


def peak_rss_bytes():
    """Process peak RSS in bytes; None where the POSIX module is absent."""
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024*value)


def registered_loss(model, arm, thermo, xp, x, target, composition, p, pm, pp, state, width):
    """The fixed data/physics objective, written once for every backend."""
    z = model(x)
    _, _, mass = output_map(z, xp)
    value = xp.mean((z[:, 0]-target[:, 0])**2+(z[:, 1]-target[:, 1])**2
                    + .1*xp.sum((mass-composition)**2, axis=-1))
    if arm == "Mphys":
        ff, temperature, species = output_map(model(p), xp)
        energy, element = thermo.residuals(ff, temperature, species, state, xp)
        ffminus, _, _ = output_map(model(pm), xp)
        ffplus, _, _ = output_map(model(pp), xp)
        mono = xp.maximum(-(ffplus-ffminus)/width, 0)
        value = value+.1*xp.mean(energy**2)+.1*xp.mean(xp.sum(element**2, axis=-1))+.01*xp.mean(mono**2)
    return value


def prepare_training(output):
    """Backend-neutral CPU64 inputs; only train/validation labels are read."""
    import numpy as np
    output = Path(output)
    properties, public, draws = (read_json(output / name) for name in
                                ("frozen_properties.json", "public_inputs.json", "fixed_draws.json"))
    thermo = Thermo(properties)
    queries = read_json(output / "splits/train.json")
    validation_queries = read_json(output / "splits/validation.json")
    physics_queries = read_json(output / "splits/physics.json")
    _, labels_ff, labels_T4, labels_Y, _ = read_dataset(output, "train")
    _, val_ff, val_T4, val_Y, _ = read_dataset(output, "validation")
    minus, plus, widths = monotonic_endpoints(physics_queries)
    return {"thermo": thermo, "features": feature_rows(queries, draws, public),
            "labels": (labels_ff, labels_T4, labels_Y),
            "validation": (feature_rows(validation_queries, draws, public), val_ff, val_T4, val_Y),
            "physics": feature_rows(physics_queries, draws, public),
            "minus": feature_rows(minus, draws, public), "plus": feature_rows(plus, draws, public),
            "states": thermo.input_states(physics_queries, public, draws), "widths": np.asarray(widths)}


def prefix_arrays(data, N):
    """First-N scaler and CPU64-normalized arrays for one registered size."""
    import numpy as np
    features = data["features"]
    labels_ff, labels_T4, labels_Y = data["labels"]
    # The scaler uses all N fixed inputs, never later-prefix/validation/test values.
    mean, scale = features[:N].mean(axis=0), features[:N].std(axis=0)
    zero_scale = np.flatnonzero(scale == 0).tolist()
    scale[scale == 0] = 1
    valid = np.isfinite(labels_ff[:N])
    if not valid.any():
        raise ValueError("No converged training point in registered prefix")
    X = (features[:N][valid]-mean)/scale
    target = np.column_stack((np.log(labels_ff[:N][valid]), np.log(labels_T4[:N][valid]/1000)))
    P, Pminus, Pplus = ((data[key]-mean)/scale for key in ("physics", "minus", "plus"))
    return {"mean": mean, "scale": scale, "zero_scale": zero_scale, "valid": valid,
            "arrays": (X, target, labels_Y[:N][valid], P, Pminus, Pplus)}


def fit_loop(rows, seed, batch, check, log, *, epochs=2000, physics_rows=2048):
    """Registered epoch, batch and physics-cycle order shared by every backend.

    ``batch(index, pindex)`` takes one optimizer step and returns its loss;
    ``check(epoch)`` runs the owned-run guard before each epoch.
    """
    import numpy as np
    generator, physics_generator = np.random.default_rng(seed), np.random.default_rng(seed+100000)
    physics_order, position, cycle = physics_generator.permutation(physics_rows), 0, 0
    steps = 0
    for epoch in range(epochs):
        check(epoch)
        order, total = generator.permutation(rows), 0.0
        for start in range(0, len(order), 256):
            index = order[start:start+256]
            if position+64 > physics_rows:
                physics_order, position = physics_generator.permutation(physics_rows), 0
                cycle += 1
            pindex = physics_order[position:position+64]; position += 64
            current = batch(index, pindex)
            if not np.isfinite(current):
                raise FloatingPointError("Nonfinite loss; no restart or changed budget")
            total += current*len(index); steps += 1
        append_progress(log, {"epoch": epoch+1, "loss": total/rows, "steps": steps,
                              "physics_cycle": cycle, "physics_position": position})
    return steps


def export_member(output, base, checkpoint, params, metadata):
    """Canonical layers.i NPZ and metadata binding both checkpoint hashes."""
    import numpy as np
    if not all(np.isfinite(value).all() for value in params.values()):
        raise FloatingPointError("Training produced nonfinite weights")
    stream = io.BytesIO(); np.savez(stream, **params)
    npz = str(base.relative_to(output)) + ".npz"
    write_once(output / npz, stream.getvalue())
    metadata = dict(metadata, safetensors_sha256=sha256_file(checkpoint), npz_sha256=sha256_file(output / npz))
    write_once(base.with_suffix(".json"), metadata)
    return params, npz


class MLXBackend:
    """The original MLX fit on MLX's default device; imported only when used."""

    def __init__(self):
        import mlx.core as mx
        self.mx = mx

    def start(self):
        self.mx.reset_peak_memory()

    def info(self):
        import importlib.metadata
        return {"backend": "mlx", "device": self.device_label(), "dtype": "float32",
                "version": importlib.metadata.version("mlx")}

    def device_label(self):
        return str(self.mx.default_device())

    def memory(self):
        return {"peak_mlx_bytes": int(self.mx.get_peak_memory()),
                "process_peak_rss_bytes_Mac": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}

    def fit(self, arm, seed, arrays, states, widths, thermo, check, log, *, epochs=2000, physics_rows=2048):
        import numpy as np
        import mlx.nn as nn
        import mlx.optimizers as optim
        from functools import partial
        mx = self.mx
        model = make_model(seed)
        mx.random.seed(seed)
        optimizer = optim.Adam(learning_rate=.001, betas=(.9, .999), eps=1e-8, bias_correction=True)
        optimizer.init(model.trainable_parameters())
        def loss(model, *args):
            return registered_loss(model, arm, thermo, mx, *args)
        loss_grad = nn.value_and_grad(model, loss)
        def step(*args):
            value, grads = loss_grad(model, *args)
            optimizer.update(model, grads)
            return value
        compiled_state = [model.state, optimizer.state, mx.random.state]
        step = partial(mx.compile, inputs=compiled_state, outputs=compiled_state)(step)
        X, target, Y, P, Pminus, Pplus = arrays
        def batch(index, pindex):
            # Normalize on CPU64 before the explicit float32 cast. This
            # avoids catastrophic subtraction of near-unit efficiencies.
            values = [mx.array(value.astype(np.float32)) for value in
                      (X[index], target[index], Y[index], P[pindex], Pminus[pindex], Pplus[pindex])]
            state = {key: mx.array(value[pindex].astype(np.float32)) for key, value in states.items()}
            width = mx.array(widths[pindex].astype(np.float32))
            value = step(*values, state, width)
            mx.eval(model.parameters(), optimizer.state, value)
            return float(value)
        return model, fit_loop(len(X), seed, batch, check, log, epochs=epochs, physics_rows=physics_rows)

    def save(self, model, output, arm, N, seed, metadata):
        import numpy as np
        from mlx.utils import tree_flatten
        output = Path(output)
        base = output / f"models/{arm}/N{N}/seed{seed}"
        base.parent.mkdir(parents=True, exist_ok=True)
        target = base.with_suffix(".safetensors")
        temporary = base.parent / (".part-"+uuid.uuid4().hex+".safetensors")
        model.save_weights(str(temporary))
        try:
            os.link(temporary, target)
        finally:
            temporary.unlink()
        params = {key: np.asarray(value, dtype=np.float32).astype(np.float64)
                  for key, value in tree_flatten(model.parameters())}
        return export_member(output, base, target, params, metadata)


def make_backend(backend="mlx", device="auto"):
    """MLX keeps its default device; --device selects only the Torch device."""
    if backend == "mlx":
        if device != "auto":
            raise ValueError("--device applies to --backend torch; MLX uses its default device")
        return MLXBackend()
    if backend == "torch":
        from .train_torch import TorchBackend
        return TorchBackend(device)
    raise ValueError(f"Unknown training backend {backend!r}")


def train_all(output, reg, registration_sha256, run, *, backend="mlx", device="auto"):
    import numpy as np
    from .run import light_training_check

    output = Path(output)
    trainer = make_backend(backend, device)
    data = prepare_training(output)
    thermo = data["thermo"]
    validation_features, val_ff, val_T4, val_Y = data["validation"]
    def check(epoch):
        if epoch%100==0:run.assert_current()
        else:light_training_check(run)
    results = []
    for arm in reg["model"]["arms"]:
        for N in reg["sampling"]["train_sizes"]:
            prefix = prefix_arrays(data, N)
            mean, scale, valid = prefix["mean"], prefix["scale"], prefix["valid"]
            for seed in reg["model"]["seeds"]:
                run.assert_current()
                fit_started=time.perf_counter();trainer.start()
                log = output / f"training/{arm}/N{N}/seed{seed}.jsonl"
                model, steps = trainer.fit(arm, seed, prefix["arrays"], data["states"], data["widths"], thermo, check, log)
                run.assert_current()
                metadata = {"arm": arm, "N": N, "seed": seed, "registration_sha256": registration_sha256,
                            "feature_scaler": {"mean": mean.tolist(), "scale": scale.tolist(), "zero_scale": prefix["zero_scale"]},
                            "epochs": 2000, "steps": steps, "train_requested": N,
                            "train_converged": int(valid.sum()), "source_manifest_sha256": sha256_file(output / "manifest.json"),
                            "training_backend": trainer.info()}
                metadata["training_compute_accounting"]={"training_wall_seconds":time.perf_counter()-fit_started,
                    **trainer.memory(),
                    "data_forward_rows":2000*int(valid.sum()),"physics_residual_forward_rows":64*steps if arm=="Mphys" else 0,
                    "monotonic_forward_rows":128*steps if arm=="Mphys" else 0}
                params, weight_path = trainer.save(model, output, arm, N, seed, metadata)
                predicted = forward64(params, (validation_features-mean)/scale)
                try:
                    metrics = prediction_metrics(*predicted, val_ff, val_T4, val_Y, thermo)
                except ValueError:
                    metrics = {"state": "INVALID_THERMO", "complete": False}
                results.append({"arm": arm, "N": N, "seed": seed, "weight_path": weight_path,
                                "feature_scaler": metadata["feature_scaler"], "metrics": metrics,
                                "validation_pass": validation_pass(metrics, reg)})
                # Each row exposes the full fixed fit budget, including failed
                # validation; validation outcomes never trigger another fit.
                results[-1]["compute_accounting"]={"fit_wall_seconds":time.perf_counter()-fit_started,
                    **trainer.memory(),
                    "device":trainer.device_label(),"epochs":2000,"optimizer_steps":steps,
                    "data_forward_rows":2000*int(valid.sum()),"physics_residual_forward_rows":64*steps if arm=="Mphys" else 0,
                    "monotonic_forward_rows":128*steps if arm=="Mphys" else 0,
                    "teacher_requests":N,"teacher_converged":int(valid.sum())}
                run.assert_current()
                results[-1]["compute_accounting"]["fit_wall_seconds"]=time.perf_counter()-fit_started
                append_progress(output / "progress.jsonl", {"stage": "train", "arm": arm, "N": N, "seed": seed,
                    "validation_pass": results[-1]["validation_pass"]})
    write_once(output / "validation.json", {"registration_sha256": registration_sha256, "members": results})
    selection = {}
    for arm in reg["model"]["arms"]:
        passing = [N for N in reg["sampling"]["train_sizes"]
                   if all(row["validation_pass"] for row in results if row["arm"] == arm and row["N"] == N)]
        selection[arm] = {"N": min(passing) if passing else 4096, "validation_pass": bool(passing)}
    write_once(output / "selection.json", {"registration_sha256": registration_sha256, "selection": selection,
                                          "validation_sha256": sha256_file(output / "validation.json")})
    return results, selection
