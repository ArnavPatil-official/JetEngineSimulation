"""Conditional v6 SAF pre-screening API; exact fuel relations stay computed.

Draw bands describe the 64 frozen parameter scenarios. They are sensitivity
bands, not empirical confidence intervals. Default model loading requires the
registered deployment receipt; this API performs no training or test scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

FRACTIONS = ("f_JetA", "f_HEFA", "f_FT", "f_ATJ")
ALIASES = ("JetA", "HEFA", "FT", "ATJ")
DRAW_IDS = tuple(f"draw_{i:02d}" for i in range(64))
METRICS = ("ff_kg_s", "T4_K", "EI_CO2_kg_kg", "CO2_g_s", "lifecycle_g_s", "nvpm_dEI_number_pct")
REGISTRATION = "docs/phase8_screening_tool_registration.json"
ROOT = Path(__file__).resolve().parents[2]


def simplex_grid(step):
    step = float(step)
    if not math.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("grid_step must be finite and between zero and one")
    n = round(1 / step)
    if n < 1 or not math.isclose(n * step, 1, rel_tol=0, abs_tol=1e-10):
        raise ValueError("grid_step must divide one into an integer grid")
    return [{"id": f"grid_{a}_{b}_{c}_{n-a-b-c}",
             **dict(zip(FRACTIONS, (a/n, b/n, c/n, (n-a-b-c)/n)))}
            for a in range(n+1) for b in range(n-a+1) for c in range(n-a-b+1)]


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not physical inputs")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Physical inputs must be finite")
    return value


def normalize_candidate(candidate, index, thrust_fraction):
    if not isinstance(candidate, dict):
        raise ValueError("Each candidate must be an object")
    ident = str(candidate.get("id", f"blend_{index:06d}"))
    fractions = []
    for field, alias in zip(FRACTIONS, ALIASES):
        if field in candidate and alias in candidate:
            raise ValueError(f"Provide either {field} or {alias}, once")
        fractions.append(_number(candidate.get(field, candidate.get(alias, 0.0))))
    thrust = _number(candidate.get("thrust_fraction", thrust_fraction))
    if any(f < 0 or f > 1 for f in fractions) or not math.isclose(sum(fractions), 1, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Blend mass fractions must be nonnegative and sum to one")
    if not 0.07 <= thrust <= 1.0:
        raise ValueError("thrust_fraction is outside the nominal 0.07 to 1.0 range")
    total = sum(fractions)
    return {"candidate_id": ident, **dict(zip(FRACTIONS, (f/total for f in fractions))), "thrust_fraction": thrust}


def envelope_flags(query, envelope):
    if not isinstance(envelope, dict) or set(envelope.get("fractions", {})) != set(FRACTIONS):
        raise ValueError("Frozen empirical training envelope is missing")
    if envelope.get("draw_ids") != list(DRAW_IDS):
        raise ValueError("Frozen draw envelope differs from the registered 64 draws")
    flags = []
    for field in (*FRACTIONS, "thrust_fraction"):
        bounds = envelope["fractions"][field] if field in FRACTIONS else envelope.get(field)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Frozen envelope for {field} is missing")
        low, high = map(_number, bounds)
        if low > high:
            raise ValueError("Frozen envelope bounds are reversed")
        if query[field] < low - 1e-12 or query[field] > high + 1e-12:
            flags.append(f"{field}_outside_observed_training_range")
    return flags


def summarize_draws(predictions):
    """Fixed-draw sensitivity summaries, also shared by speed measurements."""
    import numpy as np
    if len(predictions) != 64:
        raise ValueError("Exactly 64 draw predictions are required")
    if [p.get("draw_id") for p in predictions] != list(DRAW_IDS):
        raise ValueError("Draw predictions must preserve all registered IDs and order")
    result = {}
    for field in METRICS:
        values = [p.get(field) for p in predictions]
        available = [v for v in values if v is not None]
        if field != "nvpm_dEI_number_pct" and len(available) != 64:
            raise ValueError(f"Missing physical prediction: {field}")
        if not available:
            result[field] = {"available_draws": 0, "mean": None, "q025": None, "q50": None, "q975": None, "min": None, "max": None}
            continue
        arr = np.asarray(available, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite physical prediction: {field}")
        if field != "nvpm_dEI_number_pct" and (arr <= 0).any():
            raise ValueError(f"Nonpositive physical prediction: {field}")
        q = np.quantile(arr, [0.025, 0.5, 0.975], method="linear")
        result[field] = {"available_draws": len(available), "mean": float(arr.mean()),
                         "q025": float(q[0]), "q50": float(q[1]), "q975": float(q[2]),
                         "min": float(arr.min()), "max": float(arr.max())}
    result["ranking_q95_lifecycle_g_s"] = float(np.quantile([p["lifecycle_g_s"] for p in predictions], 0.95, method="linear"))
    result["nvpm_statuses"] = sorted(set(p.get("nvpm_status", "unavailable") for p in predictions))
    for field in ("seed_sd_ff_kg_s", "seed_sd_T4_K"):
        values = np.asarray([p.get(field, float("nan")) for p in predictions], dtype=np.float64)
        result[field] = {"mean": float(values.mean()), "max": float(values.max())} if np.isfinite(values).all() and (values >= 0).all() else None
    return result


def _load_product(model):
    from scripts.phase8.saf_surrogate.models import load_product
    return load_product(model, require_deployment=True)


def _bundle_path(model):
    path = Path(model).resolve()
    return path / "product.json" if path.is_dir() else path


def screen_blends(candidates=None, *, model, thrust_fraction=1.0, grid_step=None, verify_top_k=0, verification_out=None):
    """Predict candidate blends or a deterministic grid, with frozen draw bands.

    Invalid physical inputs are retained as flagged rows without predictions.
    Nominal inputs beyond observed training ranges retain explicit flags.
    Optional simulator verification requires idle AC and a fresh owned output.
    """
    if candidates is not None and grid_step is not None:
        raise ValueError("Choose candidates or grid_step")
    if candidates is None:
        if grid_step is None:
            raise ValueError("Provide candidates or grid_step")
        candidates = simplex_grid(grid_step)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("Candidates must be a nonempty sequence")
    if isinstance(verify_top_k, bool) or not isinstance(verify_top_k, int) or verify_top_k < 0:
        raise ValueError("verify_top_k must be a nonnegative integer")
    path = _bundle_path(model)
    bundle_bytes = path.read_bytes()
    bundle = json.loads(bundle_bytes)
    product = _load_product(path)
    envelope = bundle.get("training_envelope")
    selected_envelope = bundle.get("selected_training_envelope")
    rows, queries, seen = [], [], set()
    start = time.perf_counter()
    for i, candidate in enumerate(candidates):
        ident = str(candidate.get("id", f"blend_{i:06d}")) if isinstance(candidate, dict) else f"blend_{i:06d}"
        if ident in seen:
            raise ValueError("Candidate IDs must be unique")
        seen.add(ident)
        try:
            q = normalize_candidate(candidate, i, thrust_fraction)
        except (ValueError, TypeError) as exc:
            rows.append({"id": ident, "status": "INVALID_INPUT", "outside_training_envelope": True,
                         "flags": [str(exc)], "predictions": None})
            continue
        flags = envelope_flags(q, envelope)
        flags.extend("selected_N:" + flag for flag in envelope_flags(q, selected_envelope))
        rows.append({"id": ident, "status": "PREDICTED", "outside_training_envelope": bool(flags),
                     "flags": flags, "inputs": q, "predictions": None})
        queries.extend({**q, "query_id": f"{ident}:{draw}", "draw_id": draw} for draw in DRAW_IDS)
    predictions = []
    for offset in range(0, len(queries), 4096):
        chunk = queries[offset:offset+4096]
        results = product.predict(chunk)
        if len(results) != len(chunk):
            raise ValueError("Product returned incomplete prediction coverage")
        for q, p in zip(chunk, results):
            if p.get("query_id", q["query_id"]) != q["query_id"] or p.get("draw_id", q["draw_id"]) != q["draw_id"]:
                raise ValueError("Product prediction identity/order mismatch")
            predictions.append({**p, "query_id": q["query_id"], "draw_id": q["draw_id"]})
    pos = 0
    for row in rows:
        if row["status"] == "PREDICTED":
            drawn = predictions[pos:pos+64]
            invalid = any(p.get("diagnostic_unsafe") is True or p.get("status", "predicted") not in {"predicted", "converged"} for p in drawn)
            try:
                if invalid:
                    raise ValueError("Model returned unsafe or invalid physical outputs")
                row["predictions"] = summarize_draws(drawn)
            except (ValueError, TypeError) as exc:
                row.update(status="MODEL_INVALID", predictions=None)
                row["flags"].append(str(exc))
            pos += 64
    if path.read_bytes() != bundle_bytes:
        raise RuntimeError("Model bundle changed during prediction")
    ranked = sorted((r for r in rows if r["status"] == "PREDICTED"),
                    key=lambda r: (r["predictions"]["ranking_q95_lifecycle_g_s"], r["id"]))
    result = {"schema": 1, "conditional_label": "conditional on v6 calibration",
              "draw_count": 64, "bands": "fixed-draw conditional sensitivity bands; not empirical confidence intervals",
              "model_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
              "ranking": [r["id"] for r in ranked], "candidates": rows,
              "prediction_wall_s": time.perf_counter()-start, "verification": None}
    if verify_top_k:
        if verify_top_k > len(ranked):
            raise ValueError("verify_top_k exceeds predicted candidate count")
        if verification_out is None:
            raise ValueError("Simulator verification requires a fresh verification_out directory")
        result["verification"] = _verify(ranked[:verify_top_k], queries, predictions, path, bundle_bytes, verification_out)
    return result


def _verify(selected, queries, predictions, bundle_path, bundle_bytes, output_dir):
    from scripts.phase8 import scientific_workflow_gate as gate
    from scripts.phase8.saf_surrogate.teacher import load_verifier
    ctx = gate.prepare_context(ROOT, REGISTRATION)
    run = ctx.acquire_run(output_dir, ctx.identity["registration_sha256"])
    ids = [r["id"] for r in selected]
    selected_queries = [q for q in queries if q["candidate_id"] in set(ids)]
    selected_predictions = [p for q,p in zip(queries, predictions) if q["candidate_id"] in set(ids)]
    started = time.perf_counter()
    status, errors, hashes = "ERROR", [], {}
    try:
        reservation = gate.read(run.out / "reservation.json")
        command = {"owner_pid": reservation["owner_pid"], "owner_birth": reservation["owner_birth"],
                   "argv": reservation["argv"], "identity": ctx.identity,
                   "core": ctx.identity["core"], "selected_queries": selected_queries,
                   "model_sha256": hashlib.sha256(bundle_bytes).hexdigest()}
        gate.write_once(run.out / "command_spec.json", command)
        gate.write_once(run.out / "selected.json", {"ids": ids, "queries": selected_queries,
                       "predictions": selected_predictions, "model_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
                       "identity": ctx.identity, "global_overlap": "unavailable; selected-only verification"})
        run.assert_current()
        verifier = load_verifier(ctx, bundle_path, assert_current=run.assert_current)
        import sys
        loaded = sys.modules.get("catjet_core")
        if loaded is None or Path(loaded.__file__).resolve() != ctx.binary_path.resolve() \
                or gate.digest(loaded.__file__) != ctx.binary_sha256:
            raise RuntimeError("Verifier loaded a foreign C++ core")
        core_proof = {"path": str(Path(loaded.__file__).resolve()), "sha256": gate.digest(loaded.__file__),
                      "owner_pid": reservation["owner_pid"], "owner_birth": reservation["owner_birth"],
                      "identity": ctx.identity}
        gate.write_once(run.out / "core_proof.json", core_proof)
        actual = verifier(selected_queries)
        gate.write_once(run.out / "raw_verification.json", {"actual": actual, "identity": ctx.identity})
        if len(actual) != len(selected_queries):
            raise RuntimeError("Simulator verification coverage is incomplete")
        for q,p in zip(selected_queries, actual):
            if p.get("query_id", q["query_id"]) != q["query_id"]:
                raise RuntimeError("Simulator verification identity/order mismatch")
        if bundle_path.read_bytes() != bundle_bytes:
            raise RuntimeError("Model bundle changed during verification")
        import numpy as np
        differences = {}
        actual_bands = {}
        for ident in ids:
            indices = [i for i,q in enumerate(selected_queries) if q["candidate_id"] == ident]
            actual_bands[ident] = summarize_draws([{**actual[i], "draw_id": selected_queries[i]["draw_id"]} for i in indices])
        for field in ("ff_kg_s", "T4_K", "lifecycle_g_s"):
            diff = np.asarray([_number(p[field])-_number(a[field]) for p,a in zip(selected_predictions,actual)], dtype=np.float64)
            differences[field] = {"mean_abs": float(np.abs(diff).mean()), "max_abs": float(np.abs(diff).max())}
        summary = {"selected_ids": ids, "draw_count": 64, "simulator_requests": len(actual),
                   "actual_bands": actual_bands, "differences": differences, "wall_s": time.perf_counter()-started,
                   "global_overlap": "unavailable; selected-only verification", "identity": ctx.identity}
        gate.write_once(run.out / "verification.json", summary)
        status = "COMPLETE"
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    gate.write_once(run.out / "command.exit.json", {"exit_code": 0 if status == "COMPLETE" else 1,
                   "status": status,
                   "in_process_completed": True, "waited_child_exit": "not_applicable_synchronous_owner",
                   "owner_pid": reservation["owner_pid"], "owner_birth": reservation["owner_birth"],
                   "argv": reservation["argv"], "identity": ctx.identity,
                   "core": ctx.identity["core"], "errors": errors, "wall_s": time.perf_counter()-started})
    for p in run.out.rglob("*"):
        if p.is_file():
            hashes[str(p.relative_to(ROOT))] = gate.digest(p)
    expected = [str((run.out/n).relative_to(ROOT)) for n in ("command_spec.json", "core_proof.json", "command.exit.json", "selected.json", "raw_verification.json", "verification.json")]
    terminal = run.release({"status": status, "exit_code": 0 if status == "COMPLETE" else 1,
                            "errors": errors, "artifact_hashes": hashes, "expected_outputs": expected,
                            "outputs_complete": status == "COMPLETE"})
    if terminal["status"] != "COMPLETE":
        raise RuntimeError(f"Simulator verification failed; evidence retained at {run.out}")
    return {"selected_ids": ids, "simulator_requests": len(selected_queries),
            "out_dir": str(run.out.relative_to(ROOT)), "wall_s": time.perf_counter()-started,
            "global_overlap": "unavailable; selected-only verification"}
