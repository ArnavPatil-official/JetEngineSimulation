"""Package published quantitative screening evidence; never train or reopen labels.

The product registration declares every readable input and output. Figures are
standalone PNG/PDF exports. Missing or invalid evidence yields INCOMPLETE; this
utility never creates or moves the local freeze tag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = "docs/phase8_screening_tool_registration.json"
LABEL = "conditional on v6 calibration; original A1 FAIL (penalty-dependent); original blend gate closed"
PREFIXES = ("speed_break_even", "learning_curve", "surrogate_simulator_parity",
            "ranking_agreement", "blend_screening_bands", "nozzle_pinn_exact",
            "pycycle_match", "g0_parity")
INPUT_NAMES = {
    "speed_break_even": "timing.json", "learning_curve": "all_candidate_diagnostic_metrics.csv",
    "surrogate_simulator_parity": "test_predictions.csv", "surrogate_metrics": "test_metrics.json",
    "surrogate_precision": "precision.json", "ranking_agreement": "ranking_metrics.json",
    "blend_screening_bands": "study_bands.csv", "nozzle_pinn_exact": "test_scores.csv",
    "nozzle_report": "report.json", "pycycle_match": "p84_g1.json",
    "pycycle_reference": "pycycle_hbtf_reference.csv", "G0": "g0_parity.json",
}
REQUIRED_ROLES = set(PREFIXES) - {"g0_parity"} | {"G0", "surrogate_metrics", "surrogate_precision", "nozzle_report", "pycycle_reference"}
META = ("command_spec.json", "command.exit.json", "execution.log", "freeze_receipt.json")


class Incomplete(RuntimeError):
    pass


def workflow_gate():
    from scripts.phase8 import scientific_workflow_gate
    return scientific_workflow_gate


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts \
            or any(part.lower().startswith("sealed") for part in path.parts):
        raise Incomplete("Input is not a declared published path")
    result = Path(root) / path
    if not result.resolve().is_relative_to((Path(root) / "outputs").resolve()) \
            or result.is_symlink():
        raise Incomplete("Published input is outside outputs or redirects through a symlink")
    if any(parent.is_symlink() for parent in result.parents if parent != Path(root)):
        raise Incomplete("Published input parent redirects through a symlink")
    return result


def number(value, *, positive=False):
    if isinstance(value, bool):
        raise Incomplete("Boolean supplied as a numerical metric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Incomplete("Required published numerical metric is missing") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise Incomplete("Published metric is nonfinite or outside its physical range")
    return result


def truth(value):
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    raise Incomplete("Published categorical gate is missing or malformed")


class Published:
    def __init__(self, context, registration, gate):
        self.context, self.gate, self.root = context, gate, context.root
        self.descriptors, self.hashes, self.proofs, self.statuses = {}, {}, {}, {}
        declared = registration.get("published_inputs")
        if not isinstance(declared, list):
            raise Incomplete("Registration has no explicit published input list")
        for entry in declared:
            if not isinstance(entry, dict) or entry.get("role") not in INPUT_NAMES:
                raise Incomplete("Unrecognized published input descriptor")
            role = entry["role"]
            if role in self.descriptors or Path(entry.get("path", "")).name != INPUT_NAMES[role]:
                raise Incomplete("Duplicate role or foreign published artifact filename")
            if entry.get("format") != Path(INPUT_NAMES[role]).suffix.removeprefix("."):
                raise Incomplete("Unsupported published artifact format")
            if "expected_sha256" in entry and role not in {"pycycle_match", "pycycle_reference"}:
                raise Incomplete("Fresh scientific evidence requires its owned producer proof")
            safe_path(self.root, entry["path"])
            self.descriptors[role] = entry
        missing = REQUIRED_ROLES - set(self.descriptors)
        if missing:
            raise Incomplete("Missing registered published roles: " + ", ".join(sorted(missing)))
        consumers = defaultdict(list)
        for role, entry in self.descriptors.items():
            path = safe_path(self.root, entry["path"])
            if not path.is_file():
                raise Incomplete(f"Published artifact missing: {entry['path']}")
            if "expected_sha256" in entry:
                if sha256(path) != entry["expected_sha256"]:
                    raise Incomplete(f"Original registered artifact changed: {entry['path']}")
                gate.require_committed(self.root, [entry["path"]])
                self.statuses[role] = "HASH_BOUND_ORIGINAL"
            elif role == "G0":
                receipt = gate.read(self.root / context.op["paths"]["g0_evidence"])
                if receipt.get("files", {}).get(entry["path"]) != sha256(path):
                    raise Incomplete("Published G0 report differs from verified raw receipt")
                self.proofs[role] = {"g0_sha256": context.identity["g0_sha256"]}
                self.statuses[role] = "PASS"
            else:
                producer = entry.get("producer_registration")
                output = entry.get("producer_output_dir")
                if not isinstance(producer, str) or not isinstance(output, str):
                    raise Incomplete("Published artifact has no registered producer proof")
                if not path.resolve().is_relative_to(safe_path(self.root, output).resolve()):
                    raise Incomplete("Published artifact escaped its registered producer")
                consumers[(producer, output)].append(role)
            self.hashes[entry["path"]] = sha256(path)
        for (producer, output), roles in consumers.items():
            allowed = [self.descriptors[role]["path"] for role in roles]
            proof = gate.validate_consumer_terminal(self.root, producer, output,
                expected_binary_sha256=context.binary_sha256, artifact_paths=allowed, allow_scientific_fail=True)
            for role in roles:
                if proof["artifact_hashes"].get(self.descriptors[role]["path"]) != self.hashes[self.descriptors[role]["path"]]:
                    raise Incomplete("Published artifact differs from its owned producer terminal")
                self.proofs[role], self.statuses[role] = proof, proof["status"]
        self.assert_current()

    def assert_current(self):
        for name, expected in self.hashes.items():
            if sha256(safe_path(self.root, name)) != expected:
                raise Incomplete(f"Published input drift: {name}")

    def json(self, role):
        entry = self.descriptors[role]
        if entry["format"] != "json":
            raise Incomplete("Published format differs from required JSON schema")
        value = json.loads(safe_path(self.root, entry["path"]).read_text())
        if not isinstance(value, dict):
            raise Incomplete("Published JSON must be an object")
        return value

    def rows(self, role, columns):
        entry = self.descriptors[role]
        if entry["format"] != "csv":
            raise Incomplete("Published format differs from required CSV schema")
        with safe_path(self.root, entry["path"]).open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                raise Incomplete(f"Published CSV is empty: {role}")
            if len(header) != len(set(header)) or not set(columns) <= set(header):
                raise Incomplete(f"Published CSV columns missing/duplicated: {role}")
            indices = {key: header.index(key) for key in columns}
            for row in reader:
                if len(row) != len(header):
                    raise Incomplete("Malformed published CSV row")
                yield {key: row[index] for key, index in indices.items()}


def metric(rows, figure, group, name, value, unit, status):
    rows.append({"figure": figure, "group": str(group), "metric": name,
                 "value": value if isinstance(value, bool) else number(value), "unit": unit, "producer_status": status})


def save(figure, prefix, out, plt, guard, verdict):
    guard()
    figure.suptitle(LABEL + "\nPublished producer verdict: " + verdict, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, .96))
    for extension in ("png", "pdf"):
        with (out / f"{prefix}.{extension}").open("xb") as stream:
            figure.savefig(stream, format=extension, dpi=180, bbox_inches="tight")
            stream.flush()
            os.fsync(stream.fileno())
    plt.close(figure)
    guard()


def speed(data, plt, rows):
    record = data.json("speed_break_even")
    batches = record.get("batches")
    if not isinstance(batches, list) or not batches:
        raise Incomplete("No published timing batches")
    if [batch.get("rows") for batch in batches] != [1, 64, 4096] \
            or record.get("state") != "COMPLETE" or record.get("cpu64_end_to_end") is not True:
        raise Incomplete("Published timing does not cover the registered CPU64 full workload")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts, cpp, product = [], [], []
    for batch in batches:
        count = number(batch["rows"], positive=True)
        a, b = number(batch["cpp_median_seconds"], positive=True), number(batch["product_median_seconds"], positive=True)
        counts.append(count); cpp.append(a); product.append(b)
        for name, unit in (("cpp_median_seconds", "s"), ("product_median_seconds", "s"), ("speedup", "ratio"),
                           ("cpp_cold_seconds", "s"), ("product_cold_seconds", "s")):
            metric(rows, "speed_break_even", int(count), name, batch[name], unit, data.statuses["speed_break_even"])
    axes[0].loglog(counts, cpp, "o-", label="Full physical C++ (measured median)")
    axes[0].loglog(counts, product, "o-", label="CPU64 product (measured median)")
    axes[0].set(xlabel="Batch queries", ylabel="Measured seconds"); axes[0].legend()
    setup = number(record["total_setup_seconds"])
    if setup < 0:
        raise Incomplete("Negative published setup cost")
    bulk = batches[-1]
    per_cpp = number(bulk["cpp_median_seconds"], positive=True) / number(bulk["rows"], positive=True)
    per_product = number(bulk["product_median_seconds"], positive=True) / number(bulk["rows"], positive=True)
    queries = [0, 640000]
    axes[1].plot(queries, [q * per_cpp for q in queries], label="C++ projected from measured bulk")
    axes[1].plot(queries, [setup + q * per_product for q in queries], label="Setup + product projection")
    crossover = record.get("break_even_queries")
    if crossover is not None:
        value = number(crossover)
        if value < 0:
            raise Incomplete("Negative published break-even count")
        axes[1].axvline(value, linestyle=":", color="black", label=f"Published break-even: {value:g}")
        metric(rows, "speed_break_even", "overall", "break_even_queries", value, "queries", data.statuses["speed_break_even"])
    metric(rows, "speed_break_even", "overall", "break_even_available", crossover is not None, "boolean", data.statuses["speed_break_even"])
    for name in ("total_setup_seconds", "projected_simulator_640k_seconds", "measured_product_640k_seconds"):
        metric(rows, "speed_break_even", "overall", name, record[name], "s", data.statuses["speed_break_even"])
    for name, unit in (("named_prerequisite_seconds", "s"), ("cpu64_bulk_speedup", "ratio"),
                       ("all_physical_cpp_workers", "workers"), ("teacher_full_cycle_requests", "requests")):
        metric(rows, "speed_break_even", "overall", name, record[name], unit, data.statuses["speed_break_even"])
    for name in ("operational_pass", "gpu_available"):
        metric(rows, "speed_break_even", "overall", name, truth(record[name]), "boolean", data.statuses["speed_break_even"])
    axes[1].set(xlabel="Queries (projection)", ylabel="Total seconds"); axes[1].legend(fontsize=8)
    return fig


def learning(data, plt, rows):
    columns = ("arm", "N", "seed", "dataset", "ff_MAE_kg_s", "T4_MAE_K", "fidelity_pass")
    records = list(data.rows("learning_curve", columns))
    test = [row for row in records if row["dataset"] == "test"]
    if not test:
        raise Incomplete("Published learning curve contains no scored test rows")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    grouped = defaultdict(list)
    identities = set()
    for record in test:
        key = (record["arm"], int(number(record["seed"])))
        identity = (*key, int(number(record["N"], positive=True)))
        if identity in identities:
            raise Incomplete("Duplicate scored learning-curve member")
        identities.add(identity); grouped[key].append(record)
        for field, unit in (("ff_MAE_kg_s", "kg/s"), ("T4_MAE_K", "K")):
            value = number(record[field])
            if value < 0:
                raise Incomplete("Negative published error metric")
            metric(rows, "learning_curve", "/".join(map(str, identity)), field, value, unit, data.statuses["learning_curve"])
        metric(rows, "learning_curve", "/".join(map(str, identity)), "fidelity_pass", truth(record["fidelity_pass"]), "boolean", data.statuses["learning_curve"])
    for (arm, seed), group in sorted(grouped.items()):
        group.sort(key=lambda record: number(record["N"]))
        for axis, field in zip(axes, ("ff_MAE_kg_s", "T4_MAE_K")):
            axis.plot([number(record["N"]) for record in group], [number(record[field]) for record in group], "o-", label=f"{arm}, seed {seed}")
    for axis, label in zip(axes, ("Published test FF MAE (kg/s)", "Published test T4 MAE (K)")):
        axis.set(xlabel="Training queries N", ylabel=label)
        axis.set_xscale("log", base=2)
        axis.legend(fontsize=7)
    return fig


def parity(data, plt, rows):
    record = list(data.rows("surrogate_simulator_parity", ("ff_kg_s", "T4_K", "reference_ff_kg_s", "reference_T4_K")))
    if not record:
        raise Incomplete("No published sole-score parity predictions")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, predicted, reference, unit in ((axes[0], "ff_kg_s", "reference_ff_kg_s", "kg/s"),
                                            (axes[1], "T4_K", "reference_T4_K", "K")):
        x, y = [number(row[reference], positive=True) for row in record], [number(row[predicted], positive=True) for row in record]
        axis.scatter(x, y, s=5, alpha=.35)
        low, high = min(x + y), max(x + y)
        axis.plot([low, high], [low, high], "k:")
        axis.set(xlabel=f"Published full C++ reference ({unit})", ylabel=f"Published selected product ({unit})")
    metrics = data.json("surrogate_metrics")
    scored = metrics["ensemble"]["test"]
    for name, unit in (("ff_MAE_kg_s", "kg/s"), ("ff_max_kg_s", "kg/s"), ("T4_MAE_K", "K"), ("T4_max_K", "K")):
        metric(rows, "surrogate_simulator_parity", "selected ensemble/test", name, scored[name], unit, data.statuses["surrogate_metrics"])
    precision = data.json("surrogate_precision")
    if len(precision["export_ff_T4_max_normalized"]) != 2:
        raise Incomplete("Published export parity requires separate FF and T4 values")
    for name in ("T3_max_abs_K", "export_species_L1_max"):
        metric(rows, "surrogate_simulator_parity", "published precision", name, precision[name], "K" if name.startswith("T3") else "L1", data.statuses["surrogate_precision"])
    for index, value in enumerate(precision["export_ff_T4_max_normalized"]):
        metric(rows, "surrogate_simulator_parity", "published export/" + ("ff" if index == 0 else "T4"), "normalized_discrepancy", value, "ratio", data.statuses["surrogate_precision"])
    metric(rows, "surrogate_simulator_parity", "test", "published_prediction_rows", len(record), "rows", data.statuses["surrogate_simulator_parity"])
    return fig


def ranking(data, plt, rows):
    record = data.json("ranking_agreement")
    ranked = record.get("rows")
    if not isinstance(ranked, list) or len(ranked) != 64 or len({row["design_id"] for row in ranked}) != 64:
        raise Incomplete("Ranking publication lacks the registered 64 candidate rows")
    fig, axis = plt.subplots(figsize=(6, 5))
    x = [number(row["reference_q95_g_s"], positive=True) for row in ranked]
    y = [number(row["predicted_q95_g_s"], positive=True) for row in ranked]
    axis.scatter(x, y, s=18)
    low, high = min(x + y), max(x + y)
    axis.plot([low, high], [low, high], "k:")
    axis.set(xlabel="Published reference q95 lifecycle (g/s)", ylabel="Published product q95 lifecycle (g/s)")
    for field, unit in (("kendall_tau_b", "tau-b"), ("top10_overlap_count", "candidates"),
                        ("paired_ordering_agreement", "fraction"), ("eligible_pairs", "pairs")):
        metric(rows, "ranking_agreement", "64 candidates", field, record[field], unit, data.statuses["ranking_agreement"])
    metric(rows, "ranking_agreement", "64 candidates", "ranking_pass", truth(record["pass"]), "boolean", data.statuses["ranking_agreement"])
    return fig


def bands(data, plt, rows):
    names = ("ff_kg_s", "T4_K", "lifecycle_g_s")
    fields = ("candidate_id", "ranking_q95_lifecycle_g_s", "conditional_lifecycle_rank", "status", "diagnostic_unsafe") \
        + tuple(name + "_" + stat for name in names for stat in ("mean", "q025", "q975", "available_draws"))
    records = list(data.rows("blend_screening_bands", fields))
    if len(records) != 10000 or len({row["candidate_id"] for row in records}) != 10000:
        raise Incomplete("Published blend screening requires the registered 10000 candidates")
    diagnostic_count = 0
    ranks = []
    for record in records:
        diagnostic_count += int(truth(record["diagnostic_unsafe"]))
        rank = number(record["conditional_lifecycle_rank"], positive=True)
        if rank % 1:
            raise Incomplete("Published conditional rank is not an integer")
        ranks.append(int(rank))
        if record["status"] != "complete" or any(number(record[name + "_available_draws"]) != 64 for name in names):
            raise Incomplete("Published blend bands include invalid or incomplete candidate predictions")
        number(record["ranking_q95_lifecycle_g_s"], positive=True)
        for name in names:
            for stat in ("mean", "q025", "q975"):
                number(record[name + "_" + stat], positive=True)
            if number(record[name + "_q025"]) > number(record[name + "_q975"]):
                raise Incomplete("Published sensitivity quantiles are reversed")
    if set(ranks) != set(range(1, 10001)):
        raise Incomplete("Published conditional rank coverage differs")
    records.sort(key=lambda row: number(row["conditional_lifecycle_rank"]))
    shown = records[:20]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, name, unit in zip(axes, names, ("kg/s", "K", "g/s")):
        for index, record in enumerate(shown):
            low, mean, high = (number(record[name + "_" + stat], positive=True) for stat in ("q025", "mean", "q975"))
            axis.plot([index, index], [low, high], color="tab:blue")
            axis.scatter([index], [mean], color="black", s=12)
            for stat in ("q025", "mean", "q975"):
                field = name + "_" + stat
                metric(rows, "blend_screening_bands", record["candidate_id"], field, record[field], unit, data.statuses["blend_screening_bands"])
        axis.set(ylabel=name + " (" + unit + ")\n64 fixed-draw sensitivity band")
    axes[-1].set(xticks=range(len(shown)), xticklabels=[row["candidate_id"] for row in shown],
                 xlabel=f"First 20 of published conditional rank; diagnostic flags retained: {diagnostic_count}/10000")
    axes[-1].tick_params(axis="x", rotation=75, labelsize=7)
    metric(rows, "blend_screening_bands", "all", "published_candidates", len(records), "candidates", data.statuses["blend_screening_bands"])
    metric(rows, "blend_screening_bands", "all", "diagnostic_flagged_candidates", diagnostic_count, "candidates", data.statuses["blend_screening_bands"])
    for record in shown:
        metric(rows, "blend_screening_bands", record["candidate_id"], "diagnostic_unsafe", truth(record["diagnostic_unsafe"]), "boolean", data.statuses["blend_screening_bands"])
    return fig


def nozzle(data, plt, rows):
    fields = ("rho", "u", "T", "p")
    columns = ("panel", "regime", "case_id", "seed", "arm", "finite", "accuracy_pass") + tuple(f"{field}_relative_{stat}" for field in fields for stat in ("max", "rms"))
    worst, counts, failures = {}, defaultdict(int), defaultdict(int)
    seen = set()
    for record in data.rows("nozzle_pinn_exact", columns):
        identity = tuple(record[key] for key in ("panel", "regime", "case_id", "seed", "arm"))
        if identity in seen:
            raise Incomplete("Duplicate published nozzle case/arm/seed")
        seen.add(identity)
        key = tuple(record[key] for key in ("panel", "regime", "arm"))
        counts[key] += 1; failures[key] += int(not truth(record["accuracy_pass"]))
        if not truth(record["finite"]):
            raise Incomplete("Published nozzle errors lack finite per-case quantities")
        for field in fields:
            for stat in ("max", "rms"):
                name = f"{field}_relative_{stat}"
                value = number(record[name])
                if value < 0:
                    raise Incomplete("Negative published nozzle error")
                worst[(*key, field, stat)] = max(value, worst.get((*key, field, stat), value))
    panels = [(panel, regime) for panel in ("synthetic_test", "product_test") for regime in ("smooth_subcritical", "smooth_choked")]
    if set(counts) != {(panel, regime, arm) for panel, regime in panels for arm in ("physics_on", "data_only")}:
        raise Incomplete("Published nozzle panel/regime/arm coverage incomplete")
    producer = data.descriptors["nozzle_pinn_exact"]["producer_registration"]
    bars = data.gate.read(data.root / producer)["acceptance"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for column, (panel, regime) in enumerate(panels):
        for row_index, stat in enumerate(("max", "rms")):
            axis = axes[row_index][column]
            for arm, offset in (("physics_on", -.12), ("data_only", .12)):
                key = (panel, regime, arm)
                values = [worst[(*key, field, stat)] for field in fields]
                axis.plot([index + offset for index in range(4)], values, "o", label=arm)
                for field, value in zip(fields, values):
                    metric(rows, "nozzle_pinn_exact", "/".join((*key, field)), f"worst_case_relative_{stat}", value, "ratio", data.statuses["nozzle_pinn_exact"])
            threshold = number(bars["profile_max_relative_error_max" if stat == "max" else "profile_rms_relative_error_max"])
            axis.axhline(threshold, color="black", linestyle=":", label="Registered per-case bar")
            axis.set(xticks=range(4), xticklabels=fields, ylabel=f"Worst per-case {stat} error", title=f"{panel}\n{regime}")
            axis.legend(fontsize=6)
    for key, count in sorted(counts.items()):
        metric(rows, "nozzle_pinn_exact", "/".join(key), "scored_case_seed_rows", count, "rows", data.statuses["nozzle_pinn_exact"])
        metric(rows, "nozzle_pinn_exact", "/".join(key), "failed_per_case_bars", failures[key], "rows", data.statuses["nozzle_pinn_exact"])
    report = data.json("nozzle_report")
    for name in ("physics_on_accuracy_pass", "physics_benefit_pass", "registered_comparison_pass"):
        metric(rows, "nozzle_pinn_exact", "report", name, truth(report[name]), "boolean", report["status"])
    return fig


def pycycle(data, plt, rows):
    report = data.json("pycycle_match")
    compared = report.get("comparison_matched")
    if not isinstance(compared, dict) or set(compared) != {"DESIGN", "OD_full_pwr", "OD_part_pwr"}:
        raise Incomplete("Original pyCycle matched comparison coverage differs")
    values, labels, colors = [], [], []
    for case, record in compared.items():
        if not isinstance(record, dict) or not record:
            raise Incomplete("Empty original pyCycle comparison")
        for quantity, pair in record.items():
            relative = number(pair["rel"])
            values.append(100 * relative); labels.append(case + "/" + quantity)
            colors.append("tab:blue" if truth(pair["pass"]) else "tab:red")
            for name in ("catjet", "pycycle", "rel"):
                metric(rows, "pycycle_match", case + "/" + quantity, name, pair[name], "reported quantity" if name != "rel" else "ratio", "HASH_BOUND_ORIGINAL")
            metric(rows, "pycycle_match", case + "/" + quantity, "pass", truth(pair["pass"]), "boolean", "HASH_BOUND_ORIGINAL")
    if len(values) != int(number(report["n_compared"])):
        raise Incomplete("Original pyCycle comparison count differs")
    fig, axis = plt.subplots(figsize=(16, 5))
    axis.bar(range(len(values)), values, color=colors)
    axis.axhline(100 * number(report["tolerances"]["relative"]), color="black", linestyle=":", label="Registered relative tolerance")
    axis.set(xticks=range(len(values)), xticklabels=labels, ylabel="Published relative discrepancy (%)")
    axis.tick_params(axis="x", rotation=90, labelsize=5); axis.legend()
    metric(rows, "pycycle_match", "all", "n_compared", report["n_compared"], "quantities", "HASH_BOUND_ORIGINAL")
    metric(rows, "pycycle_match", "all", "worst_rel", report["worst_rel"], "ratio", "HASH_BOUND_ORIGINAL")
    return fig


def g0(data, plt, rows):
    report = data.json("G0")
    if report.get("verdict") != "PASS":
        raise Incomplete("Fresh G0 is not a verified PASS")
    labels, values, colors = [], [], []
    for backend, record in report["backends"].items():
        for name, check in record["checks"].items():
            numeric = [column for column in check["columns"].values() if column.get("numeric") is True]
            if not numeric:
                raise Incomplete("Published G0 check has no numerical column metrics")
            value = max(number(column["max_rel_diff"]) for column in numeric)
            labels.append(backend + "/" + name); values.append(value)
            colors.append("tab:blue" if truth(check["match"]) else "tab:red")
            metric(rows, "g0_parity", backend + "/" + name, "max_reported_column_discrepancy", value, "original compare field", "PASS" if truth(check["match"]) else "FAIL")
            metric(rows, "g0_parity", backend + "/" + name, "match", truth(check["match"]), "boolean", "PASS" if truth(check["match"]) else "FAIL")
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(len(values)), values, color=colors)
    axis.set(xticks=range(len(values)), xticklabels=labels, ylabel="Published maximum discrepancy field\n(original zero-denominator rule retained)")
    axis.tick_params(axis="x", rotation=30)
    return fig


DRAW = dict(zip(PREFIXES, (speed, learning, parity, ranking, bands, nozzle, pycycle, g0)))


def numbers_table(out, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    with (out / "NUMBERS.md").open("x") as stream:
        stream.write("# Quantitative freeze\n\n" + LABEL + "\n\n")
        stream.write("| Figure | Group | Metric | Value | Unit | Producer status |\n|---|---|---|---:|---|---|\n")
        for row in rows:
            value = str(row["value"]).lower() if isinstance(row["value"], bool) else format(row["value"], ".12g")
            stream.write("| " + " | ".join(cell(value if key == "value" else row[key]) for key in ("figure", "group", "metric", "value", "unit", "producer_status")) + " |\n")
        stream.flush(); os.fsync(stream.fileno())


def tag_guard(root, receipt_path, *, verification_receipt):
    """Read-only eligibility check; root alone creates the local tag afterward."""
    gate = workflow_gate()
    root = Path(root).resolve()
    if root != ROOT:
        raise Incomplete("Tag guard runs only from the committed registered main implementation")
    receipt_path, verification_receipt = Path(receipt_path), Path(verification_receipt)
    if receipt_path.is_absolute(): receipt_path = receipt_path.relative_to(root)
    if verification_receipt.is_absolute(): verification_receipt = verification_receipt.relative_to(root)
    if str(receipt_path) != "outputs/freeze/freeze_receipt.json":
        raise Incomplete("Foreign freeze receipt path")
    receipt_path = safe_path(root, receipt_path)
    verification_receipt = safe_path(root, verification_receipt)
    reg = gate.read(root / REGISTRATION)
    receipt = gate.read(receipt_path)
    context = gate.prepare_context(root, REGISTRATION)
    context.require_idle_ac()
    if receipt.get("status") not in {"COMPLETE", "FAIL"} or receipt.get("identity") != context.identity \
            or receipt.get("errors") or receipt.get("required_outputs") != required_outputs():
        raise Incomplete("Freeze receipt is incomplete or scientific identity changed")
    if receipt.get("completed_figures") != list(PREFIXES) or number(receipt.get("numeric_rows"), positive=True) <= 0:
        raise Incomplete("Freeze does not cover every registered figure and quantitative table")
    data = Published(context, reg, gate)
    if receipt.get("input_sha256") != data.hashes or receipt.get("producer_status") != data.statuses \
            or receipt.get("producer_proofs") != data.proofs:
        raise Incomplete("Freeze no longer matches its registered published producers")
    out = root / "outputs/freeze"
    proof = gate.validate_consumer_terminal(root, REGISTRATION, "outputs/freeze",
        expected_binary_sha256=context.binary_sha256, artifact_paths=terminal_outputs(), allow_scientific_fail=True)
    terminal, reservation = gate.read(out / "terminal.json"), gate.read(out / "reservation.json")
    command, spec = gate.read(out / "command.exit.json"), gate.read(out / "command_spec.json")
    if proof["status"] != receipt["status"] or terminal.get("execution_complete") is not True \
            or terminal.get("expected_outputs") != terminal_outputs() \
            or receipt.get("expected_outputs") != terminal_outputs() \
            or terminal["artifact_hashes"].get(str(receipt_path.relative_to(root))) != sha256(receipt_path):
        raise Incomplete("Freeze receipt is not bound to the complete authoritative terminal")
    other_hashes = {name: value for name, value in terminal["artifact_hashes"].items()
                    if name != str(receipt_path.relative_to(root))}
    if receipt.get("artifact_hashes") != other_hashes \
            or receipt.get("scientific_verdict") != terminal.get("scientific_verdict") \
            or receipt.get("execution_complete") is not True or receipt.get("exit_code") != terminal.get("exit_code"):
        raise Incomplete("Freeze receipt differs from authoritative artifact coverage or verdict")
    for record in (command, spec):
        if record.get("identity") != context.identity or record.get("owner_pid") != reservation.get("owner_pid") \
                or record.get("owner_birth") != reservation.get("owner_birth") or record.get("argv") != reservation.get("argv"):
            raise Incomplete("Freeze command proof differs from its actual reservation owner")
    if command.get("status") != terminal["status"] or command.get("exit_code") != terminal["exit_code"] \
            or command.get("in_process_completed") is not True \
            or command.get("waited_child_exit") != "not_applicable_synchronous_owner" \
            or command.get("command_spec_sha256") != sha256(out / "command_spec.json") \
            or command.get("log_sha256") != sha256(out / "execution.log"):
        raise Incomplete("Freeze body completion is unproven")
    for name, expected in {**receipt["input_sha256"], **receipt["artifact_hashes"]}.items():
        if sha256(safe_path(root, name)) != expected:
            raise Incomplete("Freeze input/artifact drift before local tag")
    gate.require_committed(root, [str(receipt_path.relative_to(root)), *receipt["artifact_hashes"], *receipt["input_sha256"], str(verification_receipt.relative_to(root))])
    validate_verification(root, reg, context, receipt, verification_receipt, gate)
    context.require_idle_ac(); data.assert_current()
    return {"status": "READY_FOR_ROOT_REVIEW", "scientific_verdict": receipt["scientific_verdict"],
            "local_tag": "freeze-2026-10-18", "receipt_sha256": sha256(receipt_path), "tag_created": False}


def validate_verification(root, reg, context, receipt, path, gate):
    declared = reg["quantitative_freeze_2026_10_04"]["verification_receipt"]
    if str(path.relative_to(root)) != declared["path"]:
        raise Incomplete("Foreign quantitative verification receipt")
    verification = gate.read(path)
    if verification.get("status") != "PASS" or verification.get("exit_code") != 0 \
            or verification.get("waited") is not True or verification.get("identity") != context.identity \
            or verification.get("test_files") != receipt.get("test_files") \
            or set(verification.get("test_files", {})) != set(declared["registered_tests"]) \
            or number(verification.get("passed"), positive=True) % 1 \
            or type(verification.get("skipped")) is not int or verification["skipped"] != 0:
        raise Incomplete("Exact frozen tests have no matching committed PASS receipt")
    raw = verification.get("raw_sha256")
    names = [declared[key] for key in ("command_spec", "command_log", "command_exit", "junit_xml")]
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise Incomplete("Verification lacks exact registered raw command evidence")
    for name in names:
        if sha256(safe_path(root, name)) != raw[name]:
            raise Incomplete("Raw verification command evidence changed")
    gate.require_committed(root, [str(path.relative_to(root)), *names, *declared["registered_tests"]])
    spec, command = gate.read(safe_path(root, declared["command_spec"])), verification.get("command")
    end = gate.read(safe_path(root, declared["command_exit"]))
    if not isinstance(command, dict) or spec.get("command") != command or end.get("command") != command:
        raise Incomplete("Verification command identity differs between raw records")
    for record in (spec, end):
        if record.get("identity") != context.identity or record.get("test_files") != verification["test_files"]:
            raise Incomplete("Verification raw records do not prove the frozen scientific and test identities")
    argv = command.get("argv")
    suite = ["-m", "pytest", *declared["registered_tests"]]
    allowed_tails = (["-v", "--junitxml=" + declared["junit_xml"]],)
    if not isinstance(argv, list) or argv[1:1 + len(suite)] != suite \
            or argv[1 + len(suite):] not in allowed_tails or command.get("workdir") != str(root) \
            or any(not isinstance(command.get(key), int) or command[key] <= 0 for key in ("owner_pid", "child_pid")) \
            or any(not isinstance(command.get(key), str) or not command[key] for key in ("owner_birth", "child_birth")):
        raise Incomplete("Verification did not run the registered suite in the actual recorded child")
    interpreter = Path(command.get("interpreter", ""))
    launched = Path(argv[0])
    if not launched.is_absolute(): launched = root / launched
    if not interpreter.is_absolute() or launched.resolve() != interpreter.resolve() \
            or interpreter.resolve() != (root / ".venv/bin/python").resolve() \
            or sha256(interpreter) != command.get("interpreter_sha256"):
        raise Incomplete("Verification interpreter provenance is invalid")
    if end.get("status") != "PASS" or end.get("exit_code") != 0 or end.get("waited") is not True \
            or end.get("passed") != verification["passed"] or end.get("skipped") != 0 \
            or end.get("log_sha256") != raw[declared["command_log"]] \
            or end.get("command_spec_sha256") != raw[declared["command_spec"]] \
            or end.get("junit_sha256") != raw[declared["junit_xml"]] \
            or not verification.get("ended_utc") or end.get("ended_utc") != verification["ended_utc"]:
        raise Incomplete("Actual waited verification exit and receipt disagree")
    log = safe_path(root, declared["command_log"]).read_text()
    log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log)
    passed = re.findall(r"\b(\d+) passed\b", log)
    if len(passed) != 1 or int(passed[0]) != verification["passed"] \
            or re.search(r"\b[1-9]\d* (?:failed|errors?|skipped|deselected)\b", log):
        raise Incomplete("Raw pytest log does not prove the reported complete PASS count")
    from xml.etree import ElementTree
    try:
        junit = ElementTree.fromstring(safe_path(root, declared["junit_xml"]).read_bytes())
        suites = list(junit.iter("testsuite"))
        cases = list(junit.iter("testcase"))
        total = sum(int(suite.attrib["tests"]) for suite in suites)
        invalid = any(int(suite.attrib[key]) != 0 for suite in suites for key in ("errors", "failures", "skipped"))
        failed_cases = any(list(case.iter("failure")) or list(case.iter("error")) or list(case.iter("skipped")) for case in cases)
    except (ElementTree.ParseError, KeyError, ValueError) as exc:
        raise Incomplete("Verification JUnit is absent or malformed") from exc
    if not suites or not cases or total != verification["passed"] or len(cases) != total or invalid or failed_cases:
        raise Incomplete("Actual JUnit does not prove every registered test passed")


def required_outputs():
    return ["NUMBERS.md", *[prefix + "." + extension for prefix in PREFIXES for extension in ("png", "pdf")]]


def terminal_outputs():
    return ["outputs/freeze/" + name for name in required_outputs() + list(META)]


def run(root, registration=REGISTRATION):
    gate = workflow_gate()
    root = Path(root).resolve()
    if root != ROOT or registration != REGISTRATION:
        raise Incomplete("Freeze runs only from the committed registered main implementation")
    reg = gate.read(root / registration)
    if reg.get("id") != "P8-SCREENING-TOOL-20261004" or reg.get("outputs", {}).get("root") != "outputs/freeze" \
            or reg.get("outputs", {}).get("local_tag") != "freeze-2026-10-18" \
            or reg.get("quantitative_freeze_2026_10_04", {}).get("required") != ["NUMBERS.md", *PREFIXES]:
        raise Incomplete("Freeze output contract differs from registration")
    expected = {"registration_sha256": sha256(root / registration), "files": {
        "scripts/phase8/freeze_screening.py": gate.file_identity(root / "scripts/phase8/freeze_screening.py"),
        "tests/test_phase8_freeze_screening.py": gate.file_identity(root / "tests/test_phase8_freeze_screening.py"),
        registration: gate.file_identity(root / registration)}}
    context = gate.prepare_context(root, registration, expected_consumer_identity=expected)
    owned = context.acquire_run("outputs/freeze", expected["registration_sha256"], identity=context.identity)
    out, errors, completed, rows, inputs, statuses, proofs = owned.out, [], [], [], {}, {}, {}
    started, status, mpl_version = utc(), "INCOMPLETE", None
    test_files = {}
    reservation = gate.read(out / "reservation.json")
    owner = {key: reservation[key] for key in ("owner_pid", "owner_birth", "argv")}
    try:
        gate.write_once(out / "command_spec.json", {**owner, "identity": context.identity,
            "workdir": str(root), "interpreter": sys.executable, "started_utc": started,
            "registration_sha256": expected["registration_sha256"], "published_inputs": reg["published_inputs"]})
        original = gate.read(root / context.op["paths"]["main_dependency"])
        source_files = {**original["original_files"], **context.identity["new_files"]}
        declared_tests = reg["quantitative_freeze_2026_10_04"]["verification_receipt"]["registered_tests"]
        test_files = {name: source_files[name] for name in declared_tests}
        data = Published(context, reg, gate)
        inputs, statuses, proofs = data.hashes, data.statuses, data.proofs
        def guard():
            owned.assert_current(); data.assert_current()
        guard()
        if importlib.util.find_spec("matplotlib") is None:
            raise Incomplete("Matplotlib is unavailable; scientific figure export cannot complete")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mpl_version = matplotlib.__version__
        matplotlib.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
        for prefix in PREFIXES:
            before = len(rows)
            try:
                guard()
                figure = DRAW[prefix](data, plt, rows)
                verdict = data.statuses["G0" if prefix == "g0_parity" else prefix]
                save(figure, prefix, out, plt, guard, verdict)
                completed.append(prefix)
            except Exception as exc:
                del rows[before:]
                errors.append({"figure": prefix, "error": f"{type(exc).__name__}: {exc}"})
                plt.close("all")
        if completed == list(PREFIXES) and not errors:
            guard(); numbers_table(out, rows); guard()
            status = "FAIL" if "FAIL" in statuses.values() else "COMPLETE"
    except Exception as exc:
        errors.append({"stage": "published_evidence", "error": f"{type(exc).__name__}: {exc}"})
    execution_complete = status in {"COMPLETE", "FAIL"}
    verdict = "FAIL" if status == "FAIL" else "PASS" if status == "COMPLETE" else "INCOMPLETE"
    gate.write_once(out / "execution.log", {"started_utc": started, "finished_utc": utc(), "status": status,
        "completed_figures": completed, "errors": errors})
    gate.write_once(out / "command.exit.json", {**owner, "identity": context.identity, "status": status,
        "exit_code": 0 if status == "COMPLETE" else 1, "in_process_completed": True,
        "waited_child_exit": "not_applicable_synchronous_owner", "started_utc": started, "ended_utc": utc(),
        "command_spec_sha256": sha256(out / "command_spec.json") if (out / "command_spec.json").is_file() else None,
        "log_sha256": sha256(out / "execution.log"), "errors": errors})
    hashes = {str(path.relative_to(root)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}
    receipt = {"schema": 1, "status": status, "execution_complete": execution_complete,
        "scientific_verdict": verdict, "exit_code": 0 if status == "COMPLETE" else 1,
        "identity": context.identity, "registration_sha256": expected["registration_sha256"],
        "conditional_label": LABEL, "created_utc": started, "finished_utc": utc(),
        "input_sha256": inputs, "producer_status": statuses, "producer_proofs": proofs,
        "test_files": test_files, "environment": {"python": platform.python_version(), "platform": platform.platform(), "matplotlib": mpl_version},
        "completed_figures": completed, "required_outputs": required_outputs(), "expected_outputs": terminal_outputs(), "numeric_rows": len(rows),
        "artifact_hashes": dict(hashes), "errors": errors, "local_tag": "freeze-2026-10-18", "tag_created": False,
        "tag_requires_root_verification": True}
    gate.write_once(out / "freeze_receipt.json", receipt)
    hashes[str((out / "freeze_receipt.json").relative_to(root))] = sha256(out / "freeze_receipt.json")
    terminal = owned.release({"status": status, "exit_code": receipt["exit_code"], "errors": [str(error) for error in errors],
        "outputs_complete": status == "COMPLETE", "expected_outputs": terminal_outputs(),
        "artifact_hashes": hashes, "execution_complete": execution_complete, "scientific_verdict": verdict})
    return 0 if terminal["status"] == "COMPLETE" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-root", type=Path, default=ROOT)
    parser.add_argument("--registration", default=REGISTRATION)
    parser.add_argument("--verify-receipt", type=Path)
    parser.add_argument("--verification-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_receipt:
            if not args.verification_receipt:
                raise Incomplete("Tag guard requires the committed exact-test verification receipt")
            print(json.dumps(tag_guard(args.main_root, args.verify_receipt, verification_receipt=args.verification_receipt)))
            return 0
        return run(args.main_root, args.registration)
    except Exception as exc:
        print(json.dumps({"status": "INCOMPLETE", "exit_code": 1, "error": f"{type(exc).__name__}: {exc}", "tag_created": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
