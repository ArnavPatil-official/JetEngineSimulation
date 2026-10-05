"""Shared native model factory and receipt-gated CPU64 product API."""
from __future__ import annotations

from pathlib import Path

from .inputs import canonical_query, feature_rows
from .postprocess import derived_outputs
from .registration import contained, read_json, sha256_file, verify_artifacts
from .thermo import Thermo


def make_model(seed, backend=None, device="cpu", *, dtype=None):
    """Native shared-backend MLP; Torch float64, optional MLX float32."""
    from simulation.ml_backend import get_backend
    selected = get_backend(backend, device=device, dtype=dtype)
    return selected.mlp(12, (128, 128, 128, 128), 494, activation="silu", seed=seed)


def output_map(z, xp):
    logits = z[..., 2:]
    logits = logits - xp.max(logits, axis=-1, keepdims=True)
    species = xp.exp(logits)
    species = species / xp.sum(species, axis=-1, keepdims=True)
    return xp.exp(z[..., 0]), 1000*xp.exp(z[..., 1]), species


def forward64(params, features):
    import numpy as np
    value = np.asarray(features, dtype=np.float64)
    for i in range(5):
        value = value @ params[f"layers.{i}.weight"].T + params[f"layers.{i}.bias"]
        if i < 4:
            # Stable SiLU without changing the registered activation.
            positive = value >= 0
            sigmoid = np.empty_like(value)
            sigmoid[positive] = 1/(1+np.exp(-value[positive]))
            exp_value = np.exp(value[~positive])
            sigmoid[~positive] = exp_value/(1+exp_value)
            value = value*sigmoid
    return output_map(value, np)


def cpu64_model(params):
    """Restore neutral weights into the primary CPU Torch float64 model."""
    from simulation.ml_backend import get_backend
    selected = get_backend("torch", device="cpu", dtype="float64")
    network = selected.mlp(12, (128, 128, 128, 128), 494, activation="silu", seed=0)
    return selected.set_parameters(network, params)


def forward_cpu64(params, features, backend=None, *, model=None):
    """Torch scores on CPU in float64; optional MLX exports score in NumPy64."""
    from simulation.ml_backend import get_backend, resolve_backend
    if resolve_backend(backend) == "mlx":
        return forward64(params, features)
    selected = get_backend("torch", device="cpu", dtype="float64")
    with selected.torch.no_grad():
        values = output_map(selected.forward(model if model is not None else cpu64_model(params), features), selected.ops)
        return tuple(selected.to_numpy(value) for value in values)


def ensemble_outputs(members):
    """Mean in CPU64, with the registered mean-species renormalization."""
    import numpy as np
    ff,T4,species=(np.stack([member[i] for member in members],axis=0) for i in range(3))
    mean_species=species.mean(axis=0)
    with np.errstate(invalid="ignore",divide="ignore"):
        mean_species=mean_species/mean_species.sum(axis=-1,keepdims=True)
    return {"ff":ff,"T4":T4,"Y4":species,"mean_ff":ff.mean(axis=0),
            "mean_T4":T4.mean(axis=0),"mean_Y4":mean_species}


def ensemble_prediction(members):
    arrays=ensemble_outputs(members)
    return arrays["mean_ff"],arrays["mean_T4"],arrays["mean_Y4"]


def load_product(bundle_path, *, require_deployment=True, backend=None):
    """Load the sealed three-seed product; unvalidated use must be explicit."""
    return Product(bundle_path, require_deployment=require_deployment, backend=backend)


class Product:
    def __init__(self, bundle_path, *, require_deployment=True, backend=None):
        import numpy as np
        self.bundle_path = Path(bundle_path).resolve()
        self.output = self.bundle_path.parent
        self.bundle = read_json(self.bundle_path)
        if self.bundle.get("registration_id") != "P8-S-20261004":
            raise ValueError("Foreign product bundle")
        if self.bundle["model"]["seeds"] != [42, 43, 44] or self.bundle["model"]["output_dimension"] != 494:
            raise ValueError("Foreign model architecture/aggregation")
        verify_artifacts(self.output, self.bundle["artifacts"])
        self.diagnostic = not require_deployment
        if require_deployment:
            root = Path(__file__).resolve().parents[3]
            if sha256_file(root / "docs/phase8_saf_surrogate_registration.json") != self.bundle["registration_sha256"]:
                raise ValueError("Current registration differs from the validated product")
            verify_artifacts(root, self.bundle["scientific_sources"])
            receipt = read_json(self.output / "deployment_receipt.json")
            if (receipt.get("state") != "PASS" or receipt.get("product_sha256") != sha256_file(self.bundle_path)
                    or receipt.get("registration_sha256") != self.bundle["registration_sha256"]):
                raise ValueError("Product has no matching successful deployment receipt")
            verify_artifacts(self.output, receipt["evidence_sha256"])
            if not all(receipt.get("gates", {}).get(name) == "PASS" for name in
                       ("fidelity", "ranking", "precision", "provenance", "operational")):
                raise ValueError("Deployment evidence contains a failed gate")
            projected={str(self.bundle_path.relative_to(root)),str((self.output/"deployment_receipt.json").relative_to(root))}
            projected.update(str((self.output/path).relative_to(root)) for path in self.bundle["artifacts"])
            projected.update(str((self.output/path).relative_to(root)) for path in receipt["evidence_sha256"])
            if self.bundle.get("simulator", {}).get("name") == "python-v6":
                from scripts.phase8.python_pc import validate_product_provenance
                metadata = validate_product_provenance(root, self.output,
                    artifact_paths=sorted(projected), expected_simulator_identity_sha256=self.bundle["simulator_identity_sha256"])
                if metadata["status"] != "COMPLETE":
                    raise ValueError("Python product has no completed authentic producer")
            elif self.bundle.get("execution_profile") == "pc":
                from scripts.phase8.pc_saf import validate_local_terminal as validate_consumer_terminal
            else:
                from scripts.phase8.scientific_workflow_gate import validate_consumer_terminal
            if self.bundle.get("simulator", {}).get("name") != "python-v6":
                metadata=validate_consumer_terminal(root,"docs/phase8_saf_surrogate_registration.json",self.output,
                    expected_binary_sha256=self.bundle["binary_sha256"],artifact_paths=sorted(projected))
                if metadata["status"] not in ("PASS","COMPLETE"):
                    raise ValueError("Deployment producer has no successful authenticated release")
        self.properties = read_json(contained(self.output, self.bundle["properties_path"]))
        self.public = read_json(contained(self.output, self.bundle["public_inputs_path"]))
        self.draws = read_json(contained(self.output, self.bundle["fixed_draws_path"]))
        self.thermo = Thermo(self.properties)
        self.params = []
        self.scalers = []
        for entry in self.bundle["members"]:
            with np.load(contained(self.output, entry["weights_path"]), allow_pickle=False) as archive:
                params = {key: np.asarray(archive[key], dtype=np.float64)
                          for key in archive.files if key.startswith("layers.")}
            expected = [(12, 128), (128, 128), (128, 128), (128, 128), (128, 494)]
            for i, (incoming, outgoing) in enumerate(expected):
                if params[f"layers.{i}.weight"].shape != (outgoing, incoming) or params[f"layers.{i}.bias"].shape != (outgoing,):
                    raise ValueError("Corrupt model dimensions")
            if not all(np.isfinite(value).all() for value in params.values()):
                raise ValueError("Nonfinite model weights")
            scaler = entry["feature_scaler"]
            mean, scale = np.asarray(scaler["mean"], dtype=np.float64), np.asarray(scaler["scale"], dtype=np.float64)
            if mean.shape != (12,) or scale.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Invalid saved feature scaler")
            self.params.append(params)
            self.scalers.append((mean, scale))
        if len(self.params) != 3 or [row["seed"] for row in self.bundle["members"]] != [42, 43, 44]:
            raise ValueError("Exactly the three fixed ordered seeds are required")
        from simulation.ml_backend import resolve_backend
        self.backend = resolve_backend(backend)
        self.cpu_models = [cpu64_model(params) for params in self.params] if self.backend == "torch" else [None]*3

    def arrays(self, queries):
        """Internal full-species results, shared by scoring and exact postprocess."""
        import numpy as np
        queries = self.canonical_queries(queries)
        features = feature_rows(queries, self.draws, self.public)
        members = [forward_cpu64(params, (features-mean)/scale, self.backend, model=network)
                   for params, (mean, scale), network in zip(self.params, self.scalers, self.cpu_models)]
        return queries, ensemble_outputs(members)

    def canonical_queries(self,queries):
        return [canonical_query(query, self.draws, self.public) | {
            key: query[key] for key in ("query_id", "candidate_id", "design_id", "prefix_index", "named_case_id")
            if key in query} for query in queries]

    def predict(self, queries):
        import numpy as np
        if not queries:
            return []
        queries, arrays = self.arrays(queries)
        return self.postprocess(queries,arrays)

    def postprocess(self,queries,arrays):
        """Identical CPU64 physical output, seed uncertainty and flags for CPU/GPU."""
        import numpy as np
        states = self.thermo.input_states(queries, self.public, self.draws)
        results = []
        for i, query in enumerate(queries):
            ff, T4, Y = arrays["mean_ff"][i], arrays["mean_T4"][i], arrays["mean_Y4"][i]
            row = {"input_sha256": query["input_sha256"], "diagnostic_unsafe": self.diagnostic,
                   "draw_id": query["draw_id"],
                   "ff_kg_s": float(ff) if np.isfinite(ff) else None,
                   "T4_K": float(T4) if np.isfinite(T4) else None}
            for key, values in (("seed_sd_ff_kg_s", arrays["ff"][:, i]), ("seed_sd_T4_K", arrays["T4"][:, i])):
                deviation = values.std(ddof=1)
                row[key] = float(deviation) if np.isfinite(deviation) else None
            row.update({key: query[key] for key in ("query_id", "candidate_id", "design_id") if key in query})
            row["within_train4096_ranges"] = within_envelope(query, self.bundle["training_envelope"])
            row["within_selected_training_ranges"] = within_envelope(query, self.bundle["selected_training_envelope"])
            if not np.isfinite(ff) or ff <= 0 or not np.isfinite(T4) or T4 <= 0 or not np.isfinite(Y).all():
                row.update(status="invalid_prediction", physical_aux=None)
                row.update(EI_CO2_kg_kg=None, CO2_g_s=None, lifecycle_g_s=None,
                           nvpm_dEI_number_pct=None, nvpm_status="unavailable", nvpm_reason="invalid_prediction")
            else:
                row.update(derived_outputs(query, ff, self.properties))
                spread=row["seed_sd_ff_kg_s"]
                row.update(seed_sd_EI_CO2_kg_kg=0.0,
                    seed_sd_CO2_g_s=1000*row["EI_CO2_kg_kg"]*spread,
                    seed_sd_lifecycle_g_s=(row["lifecycle_g_s"]/ff)*spread,
                    seed_sd_nvpm_dEI_number_pct=0.0 if row["nvpm_dEI_number_pct"] is not None else None)
                try:
                    mixture = self.thermo.mixture(T4, Y)
                    single_state = {key: value[i:i+1] for key, value in states.items()}
                    energy, element = self.thermo.residuals(np.array([ff]), np.array([T4]), Y[None, :], single_state)
                    row.update(status="predicted", physical_aux={
                        "cp4_J_kg_K": float(mixture["cp"]), "R4_J_kg_K": float(mixture["R"]),
                        "gamma4": float(mixture["gamma"]), "energy_residual": float(energy[0]),
                        "element_residual_max": float(np.max(np.abs(element[0])))})
                except ValueError as error:
                    row.update(status="invalid_thermo", physical_aux=None, thermo_reason=str(error))
            results.append(row)
        return results


def training_envelope(queries):
    return {"fractions": {f"f_{fuel}": [min(q[f"f_{fuel}"] for q in queries),
                            max(q[f"f_{fuel}"] for q in queries)]
                          for fuel in ("JetA", "HEFA", "FT", "ATJ")},
            "thrust_fraction": [min(q["thrust_fraction"] for q in queries),
                                max(q["thrust_fraction"] for q in queries)],
            "draw_ids": sorted({q["draw_id"] for q in queries})}


def within_envelope(query, envelope):
    return (all(lo <= query[key] <= hi for key, (lo, hi) in envelope["fractions"].items())
            and envelope["thrust_fraction"][0] <= query["thrust_fraction"] <= envelope["thrust_fraction"][1]
            and query["draw_id"] in envelope["draw_ids"])


def summarize_draws(queries, predictions):
    """Shared fixed-draw bands and rank aggregation, included in CPU64 timing."""
    if len(queries) != len(predictions):
        raise ValueError("Query/prediction alignment mismatch")
    groups = {}
    for query, prediction in zip(queries, predictions):
        for identity_key in ("input_sha256", "query_id", "candidate_id", "design_id"):
            if identity_key in query and identity_key in prediction and query[identity_key] != prediction[identity_key]:
                raise ValueError("Prediction/query identity mismatch")
        if prediction.get("draw_id", query["draw_id"]) != query["draw_id"]:
            raise ValueError("Prediction/query draw identity mismatch")
        prediction = dict(prediction, draw_id=query["draw_id"])
        key = query.get("candidate_id", query.get("design_id", query.get("query_id")))
        if key is None:
            key = query["input_sha256"]
        groups.setdefault(str(key), []).append((query, prediction))
    result = []
    for key, rows in sorted(groups.items()):
        draw_ids = [query["draw_id"] for query, _ in rows]
        if len(rows) == 64 and set(draw_ids) != {f"draw_{i:02d}" for i in range(64)}:
            raise ValueError("Fixed-draw group lacks the exact64 paired IDs")
        if len(rows) == 64 and draw_ids != [f"draw_{i:02d}" for i in range(64)]:
            raise ValueError("Fixed-draw group must be in canonical ascending order")
        if len(set(draw_ids)) != len(draw_ids):
            raise ValueError("Duplicate draw in aggregation group")
        fractions = {tuple(query[f"f_{fuel}"] for fuel in ("JetA", "HEFA", "FT", "ATJ"))
                     + (query["thrust_fraction"],) for query, _ in rows}
        if len(fractions) != 1:
            raise ValueError("Aggregation requires matched composition/thrust")
        summary = {"candidate_id": key, "draw_count": len(rows),
                   "conditional_fixed_draw_bands": True,
                   "diagnostic_unsafe": any(pred.get("diagnostic_unsafe", False) for _, pred in rows)}
        valid = all(pred.get("status") in ("predicted", "converged") for _, pred in rows)
        summary["status"] = "complete" if valid else "invalid_prediction"
        if len(rows) == 64 and valid:
            from scripts.phase8.screening_product import summarize_draws as shared_summary
            summary.update(shared_summary([prediction for _, prediction in rows]))
        elif len(rows) == 64:
            summary["fixed_draw_bands"] = None
            summary["ranking_q95_lifecycle_g_s"] = None
            summary["invalid_draw_ids"] = [q["draw_id"] for q,p in rows if p.get("status") not in ("predicted","converged")]
            summary["reason"] = "invalid predictions; no complete fixed-draw band or rank"
        else:
            summary["fixed_draw_bands"] = None
            summary["ranking_q95_lifecycle_g_s"] = None
            summary["partial_draw_group"] = True
        result.append(summary)
    result.sort(key=lambda row: (row["ranking_q95_lifecycle_g_s"]
                                 if row.get("ranking_q95_lifecycle_g_s") is not None
                                 else float("inf"), row["candidate_id"]))
    for rank, row in enumerate(result, start=1):
        row["conditional_lifecycle_rank"] = rank
    return result
