"""Whitelisted source-property evidence and immutable whole-condition identities.

This module uses only the standard library. It never opens sealed target paths.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


class Blocked(RuntimeError):
    """A registered input, scope or proof is unavailable."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise Blocked(f"not a JSON object: {path}")
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def read_projection(path, allowed, required):
    """Select allowed columns before numeric decoding; no alternate file routing."""
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)) or not set(required) <= set(header):
            raise Blocked(f"missing/duplicate source columns: {path}")
        indices = {name: i for i, name in enumerate(header) if name in allowed}
        rows = []
        for values in reader:
            if len(values) != len(header):
                raise Blocked(f"malformed CSV row: {path}")
            rows.append({name: values[index] for name, index in indices.items()})
        return rows


def finite_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Blocked(f"invalid property {name}") from exc
    if not math.isfinite(number):
        raise Blocked(f"nonfinite property {name}")
    return number


def expected_source_ids(reg):
    dep = reg["dependencies"]
    return ([f"train_{i:06d}" for i in range(4096)],
            [f"{fuel}|{op}" for fuel in sorted(dep["named_fuels"]) for op in dep["named_modes"]])


def validate_source_ids(train, named, reg):
    """Only input identities are examined here; coefficients are not decoded."""
    expected_train, expected_named = expected_source_ids(reg)
    if len(train) != 4096 or [r.get("design_id") for r in train] != expected_train:
        raise Blocked("TRAIN4096 ID/order/count mismatch")
    if [r.get("prefix_index") for r in train] != [str(i) for i in range(4096)]:
        raise Blocked("TRAIN4096 prefix index mismatch")
    draws = {}
    for row in train:
        if row.get("split") != "train":
            raise Blocked("non-training source row refused")
        draws[row.get("draw_id")] = draws.get(row.get("draw_id"), 0) + 1
    if draws != {f"draw_{i:02d}": 64 for i in range(64)}:
        raise Blocked("TRAIN4096 fixed-draw balance mismatch")
    if len(named) != 68 or [r.get("named_case_id") for r in named] != expected_named:
        raise Blocked("named68 input ID/order/count mismatch")
    for row in named:
        if row["named_case_id"] != f"{row.get('fuel')}|{row.get('op')}":
            raise Blocked("named source key mismatch")


def fraction_fields(row, alternate=False):
    fields = [row.get(f"f_{name}") for name in ("JetA", "HEFA", "FT", "ATJ")]
    if alternate:
        if row.get("in_product_API", "").lower() != "false" or any(v not in ("", "null", "None") for v in fields):
            raise Blocked("alternate Jet A must retain its separate non-product identity")
        if json.loads(row.get("fuel_parts", "null")) != {"JetA_dooley2010": 1.0}:
            raise Blocked("alternate Jet A fuel_parts changed")
        return None
    result = [finite_number(v, "mass_fraction") for v in fields]
    if "in_product_API" in row and row["in_product_API"].lower() != "true":
        raise Blocked("ordinary named source is outside product API")
    if any(v < 0.0 or v > 1.0 for v in result) or abs(sum(result) - 1.0) > 1e-12:
        raise Blocked("invalid four-component mass simplex")
    return result


def property_pair(row, box):
    gamma = finite_number(row.get("gamma4"), "gamma4")
    gas_R = finite_number(row.get("R4_J_kg_K"), "R4")
    cp = finite_number(row.get("cp4_J_kg_K"), "cp4")
    if gamma <= 1.0 or gas_R <= 0.0 or cp <= gas_R or abs((cp / (cp - gas_R)) / gamma - 1.0) > 1e-10:
        raise Blocked("incoherent source cp/R/gamma")
    return gamma, gas_R, (box["gamma"][0] <= gamma <= box["gamma"][1]
                           and box["R_J_kg_K"][0] <= gas_R <= box["R_J_kg_K"][1])


def property_cases(train, named, reg, context, producer_sha, property_input_sha):
    """Retain every fixed row; invalid/out-of-box rows block rather than disappear."""
    cases, failures = [], []
    for source, rows in (("train", train), ("named", named)):
        for row in rows:
            key = row["design_id"] if source == "train" else row["named_case_id"]
            identity = {"source": source, "source_id": key}
            try:
                if row.get("status") != "converged":
                    raise Blocked("SOURCE_UNCONVERGED")
                for name, expected in (("source_registration_sha256", producer_sha),
                                       ("binary_sha256", context.binary_sha256),
                                       ("property_manifest_sha256", property_input_sha)):
                    if row.get(name) != expected:
                        raise Blocked(f"SOURCE_PROOF_MISMATCH:{name}")
                if not row.get("input_sha256") or not row.get("source_commit"):
                    raise Blocked("SOURCE_IDENTITY_MISSING")
                fraction_fields(row, alternate=source == "named" and row["fuel"] == "JetA_dooley2010")
                gamma, gas_R, inside = property_pair(row, reg["gas_and_geometry"])
                if not inside:
                    raise Blocked("OUT_OF_ENVELOPE")
                cases.append({**identity, "gamma": gamma, "R": gas_R,
                              "input_sha256": row["input_sha256"], "row": row})
            except (Blocked, ValueError, TypeError) as exc:
                failures.append({**identity, "status": str(exc)})
    return cases, failures


def condition_key(case):
    return (case["regime"], *(float(case[k]).hex() for k in ("NPR", "gamma", "R")))


def assert_disjoint(splits):
    """Reject any train/validation/test condition collision; retain test-panel provenance."""
    keys = {name: {condition_key(case) for case in rows} for name, rows in splits.items()}
    if len(keys["train"]) != len(splits["train"]) or len(keys["validation"]) != len(splits["validation"]):
        raise Blocked("duplicate training/validation condition")
    if len(keys["synthetic_test"]) != len(splits["synthetic_test"]):
        raise Blocked("duplicate synthetic test condition")
    for left, right in (("train", "validation"), ("train", "synthetic_test"),
                        ("train", "product_test"), ("validation", "synthetic_test"),
                        ("validation", "product_test")):
        if keys[left] & keys[right]:
            raise Blocked(f"condition split overlap: {left}/{right}")


def safe_path(root, name):
    value = Path(name)
    if value.is_absolute() or ".." in value.parts or name in ("", ".") or "sealed" in value.parts:
        raise Blocked("unsafe or sealed source evidence path")
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise Blocked("source evidence escaped its root")
    for component in (path,*path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise Blocked("source evidence must not redirect through symlinks")
    return path


def verify_source_commit(root, commit, expected_files):
    """The claimed producer commit must contain the already-proven source blobs."""
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise Blocked("malformed producer source commit")
    try:
        raw = subprocess.check_output(["git","ls-tree","-r",commit,"--",*expected_files],cwd=root,text=True)
    except (OSError,subprocess.CalledProcessError) as exc:
        raise Blocked("producer source commit is unreadable") from exc
    actual = {}
    for line in raw.splitlines():
        metadata,name = line.split("\t",1)
        mode,kind,blob = metadata.split()
        actual[name] = (mode,blob) if kind == "blob" else None
    if actual != {name:(entry["mode"],entry["blob"]) for name,entry in expected_files.items()}:
        raise Blocked("producer source commit differs from proven scientific sources")


def generation_command_proof(main_root, root, dep, producer, producer_sha, producer_identity, binary, pre_sha, generation, context):
    """Join raw waited command records and archived ownership without label access."""
    raw = generation.get("raw_command", {})
    paths = {"spec":"proofs/generation_command_spec.json", "handshake":"proofs/generation_handshake.json",
             "exit":"proofs/generation_exit.json", "log":"proofs/generation.log"}
    if set(raw) != set(paths):
        raise Blocked("generation lacks the exact raw command proof set")
    files = []
    records = {}
    for name, suffix in paths.items():
        path = safe_path(main_root,str((root/suffix).relative_to(main_root)))
        if raw[name] != {"path":str(path.relative_to(main_root)), "sha256":sha256(path)}:
            raise Blocked("raw generation command path/hash mismatch")
        files.append(path)
        if name != "log":
            records[name] = read_object(path)
    spec, handshake, exited = (records[name] for name in ("spec","handshake","exit"))
    common = {"schema_version":1, "stage":"generate", "registration_path":dep["source_registration"],
              "registration_sha256":producer_sha, "consumer_identity":producer_identity,
              "start_identity":producer_identity, "binary":binary,
              "property_inputs_manifest_sha256":pre_sha}
    for record in records.values():
        if any(record.get(key) != expected for key,expected in common.items()):
            raise Blocked("raw generation source/input/core identity differs")
    expected_sources = set(producer["relevant_files"]["implementation_create"]+producer["relevant_files"]["registration_create"])
    source_hashes = {name:sha256(safe_path(main_root,name)) for name in expected_sources}
    if spec.get("source_hashes") != source_hashes:
        raise Blocked("raw generation lacks complete registered source hashes")
    argv = spec.get("argv")
    if not isinstance(argv,list) or len(argv) != 6 or argv[1:] != ["-m","scripts.phase8.saf_surrogate.run","_generate","--spec",str(root/paths["spec"])] \
            or spec.get("root") != str(main_root) or spec.get("output") != str(root):
        raise Blocked("raw generation command differs from exact registered child")
    for record in (handshake,exited):
        if any(record.get(key) != spec.get(key) for key in ("owner_pid","owner_birth","owner_lease","source_hashes","argv","started_utc","root","output")):
            raise Blocked("raw generation records are from different commands")
    if not isinstance(handshake.get("pid"),int) or handshake["pid"] <= 0 or not handshake.get("birth") \
            or exited.get("pid") != handshake["pid"] or exited.get("birth") != handshake["birth"] \
            or exited.get("waited") is not True or exited.get("exit_code") != 0 \
            or exited.get("end_identity") != producer_identity or exited.get("identity_problems") != []:
        raise Blocked("raw generation has no actual successful joined wait")
    launch = generation.get("launch", {})
    expected_launch = {key:exited[key] for key in ("pid","birth","argv","started_utc","finished_utc","exit_code","waited")}
    expected_launch.update(log_path=raw["log"]["path"],log_sha256=raw["log"]["sha256"])
    if launch != expected_launch:
        raise Blocked("inline generation launch differs from raw waited command")
    snapshot_path = safe_path(main_root,str((root/"proofs/generation_owner_lease.json").relative_to(main_root)))
    snapshot = read_object(snapshot_path)
    lease_pointer = spec.get("owner_lease", {})
    if lease_pointer.get("path") not in {context.op["paths"]["owner_lease"],str(snapshot_path.relative_to(main_root))} \
            or lease_pointer.get("sha256") != sha256(snapshot_path):
        raise Blocked("generation command is outside exact archived owner lease")
    reservation_path = safe_path(main_root,str((root/"reservation.json").relative_to(main_root)))
    reservation = read_object(reservation_path)
    for record in (snapshot,reservation):
        if record.get("identity") != producer_identity or record.get("owner_pid") != spec.get("owner_pid") \
                or record.get("owner_birth") != spec.get("owner_birth") \
                or record.get("registration_sha256") != producer_sha or record.get("output_dir") != str(root.relative_to(main_root)):
            raise Blocked("generation owner differs from released producer reservation")
    if snapshot.get("reservation_sha256") != sha256(reservation_path):
        raise Blocked("generation archived lease reservation changed")
    return files+[snapshot_path], source_hashes


def source_artifacts(main_root, reg, context, out):
    """Validate the allowed projection and producer manifests after acquiring ownership."""
    dep = reg["dependencies"]
    # The complete case ID set is recorded before opening coefficients.
    train_ids, named_ids = expected_source_ids(reg)
    write_json(out / "case_selection.json", {"train_ids": train_ids, "named_ids": named_ids})
    producer_path = safe_path(main_root,dep["source_registration"])
    producer = read_object(producer_path)
    if producer.get("id") != dep["source_registration_id"]:
        raise Blocked("foreign P8-S source registration")
    root = safe_path(main_root,dep["product_rows"]).parent
    manifest = read_object(safe_path(main_root,dep["property_output_manifest"]))
    property_input = safe_path(main_root,str((root/"property_inputs_manifest.json").relative_to(main_root)))
    input_proof = read_object(property_input)
    generation_path = safe_path(main_root,str((root/"generation_terminal.json").relative_to(main_root)))
    generation = read_object(generation_path)
    producer_sha, pre_sha = sha256(producer_path), sha256(property_input)
    producer_identity = {**context.identity, "registration_sha256": producer_sha}
    binary = {"path": str(context.binary_path.relative_to(main_root)), "sha256": context.binary_sha256}
    for proof in (input_proof, generation, manifest):
        if proof.get("schema_version") != 1 or proof.get("registration_id") != dep["source_registration_id"] \
                or proof.get("registration_sha256") != producer_sha:
            raise Blocked("P8-S proof registration/schema mismatch")
    for proof in (input_proof, manifest):
        if proof.get("binary") != binary or proof.get("consumer_identity") != producer_identity \
                or proof.get("counts") != {"train": 4096, "named_central": 68}:
            raise Blocked("P8-S property input/core/count proof differs from shared context")
    if not input_proof.get("source_commit") or manifest.get("source_commit") != input_proof["source_commit"]:
        raise Blocked("P8-S producer commit mismatch")
    if generation.get("state") != "COMPLETE" or generation.get("stage") != "generate" \
            or generation.get("scientific_coverage") != "PASS" or generation.get("identity_problems") != [] \
            or generation.get("start_identity") != producer_identity or generation.get("end_identity") != producer_identity \
            or generation.get("binary_path") != binary["path"] or generation.get("binary_sha256") != binary["sha256"] \
            or generation.get("property_inputs_manifest_sha256") != pre_sha:
        raise Blocked("P8-S generation identity/coverage proof failed")
    counts = generation.get("counts", {})
    for name, count in {"train_requested":4096, "validation_requested":1024, "test_requested":2048,
                        "ranking_requested":4096, "named_central_requested":68, "total_requested":11332}.items():
        if counts.get(name) != count:
            raise Blocked("P8-S generation request count mismatch")
    launch = generation.get("launch", {})
    if launch.get("exit_code") != 0 or launch.get("waited") is not True \
            or not isinstance(launch.get("pid"), int) or launch["pid"] <= 0 or not launch.get("birth") \
            or not isinstance(launch.get("argv"), list) or not launch["argv"] \
            or not launch.get("started_utc") or not launch.get("finished_utc"):
        raise Blocked("P8-S generation has no successful waited command")
    log_path = safe_path(main_root, launch["log_path"])
    if sha256(log_path) != launch.get("log_sha256"):
        raise Blocked("P8-S generation log changed")
    if manifest.get("property_inputs_manifest") != {"path":"property_inputs_manifest.json", "sha256":pre_sha} \
            or manifest.get("producer_terminal") != {"path":"generation_terminal.json", "sha256":sha256(generation_path)}:
        raise Blocked("property pre/post manifest binding mismatch")
    files = {name:safe_path(main_root,str((root/name).relative_to(main_root)))
             for name in ("teacher_rows.csv", "teacher_species.npz", "named_central_properties.csv", "frozen_properties.json")}
    expected_hashes = manifest.get("artifacts")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(files):
        raise Blocked("P8-S allowed artifact set mismatch")
    for name, path in files.items():
        actual = sha256(path)
        recorded = generation.get("outputs", {}).get(str(path.relative_to(main_root)))
        if expected_hashes[name] != actual or recorded != actual:
            raise Blocked("allowed P8-S artifact generation/final/byte hashes differ")
    dependencies = input_proof.get("dependencies", {})
    expected_dependencies = {"main_dependency":context.op["paths"]["main_dependency"],
        "g0_evidence":context.op["paths"]["g0_evidence"],
        "source_extension_manifest":context.op["paths"]["source_extension_manifest"],
        "frozen_properties":str(files["frozen_properties.json"].relative_to(main_root)),
        "train_queries":str((root / "splits/train.json").relative_to(main_root)),
        "named_queries":str((root / "splits/named_central.json").relative_to(main_root))}
    dependency_files = []
    for name, expected in expected_dependencies.items():
        entry = dependencies.get(name, {})
        if entry.get("path") != expected:
            raise Blocked("P8-S property dependency path mismatch")
        path = safe_path(main_root, expected)
        if entry.get("sha256") != sha256(path):
            raise Blocked("P8-S property dependency hash mismatch")
        dependency_files.append(path)
    command_files, command_source_hashes = generation_command_proof(main_root,root,dep,producer,producer_sha,
        producer_identity,binary,pre_sha,generation,context)
    # The overall terminal is later than property_manifest and may hash it;
    # validate that release independently, without a reverse hash or sealed reads.
    from scripts.phase8 import scientific_workflow_gate as gate
    producer_artifacts = {str(path.relative_to(main_root)) for path in [property_input,generation_path,
        main_root/dep["property_output_manifest"],*files.values(),*command_files,
        safe_path(main_root,expected_dependencies["train_queries"]),safe_path(main_root,expected_dependencies["named_queries"])]}
    for name in ("terminal.json","reservation.json","released_lease.json"):
        safe_path(main_root,str((root/name).relative_to(main_root)))
    release = gate.validate_consumer_terminal(main_root,dep["source_registration"],root,
        expected_binary_sha256=context.binary_sha256,artifact_paths=sorted(producer_artifacts))
    if release.get("identity") != producer_identity or set(release.get("verified_artifact_paths",[])) != producer_artifacts:
        raise Blocked("producer release artifact projection differs")
    original_attestation = read_object(safe_path(main_root,expected_dependencies["main_dependency"]))
    verify_source_commit(main_root,input_proof["source_commit"],
                         {**original_attestation["original_files"],**context.identity["new_files"]})
    query_tables = {}
    query_columns = {"split", "design_id", "prefix_index", "draw_id", "named_case_id", "fuel", "op",
                     "in_product_API", "fuel_parts", "input_sha256", "thrust_fraction", "f_JetA", "f_HEFA", "f_FT", "f_ATJ"}
    for name in ("train_queries", "named_queries"):
        raw = json.loads(safe_path(main_root, expected_dependencies[name]).read_text())
        if not isinstance(raw,list) or any(not isinstance(row,dict) for row in raw):
            raise Blocked("P8-S source query manifest must be a plain object list")
        query_tables[name] = [{key:value for key,value in row.items() if key in query_columns} for row in raw]
    identities = input_proof.get("cases", {})
    if manifest.get("case_identity") != identities:
        raise Blocked("P8-S property case manifests differ")
    train_proof, named_proof = identities.get("train", []), identities.get("named_central", [])
    if [r.get("design_id") for r in train_proof] != train_ids or [r.get("named_case_id") for r in named_proof] != named_ids:
        raise Blocked("P8-S pregeneration canonical input identities differ")
    if [r.get("design_id") for r in query_tables["train_queries"]] != train_ids \
            or [r.get("named_case_id") for r in query_tables["named_queries"]] != named_ids:
        raise Blocked("P8-S source query canonical identities differ")
    allowed = set(dep["columns_read"])
    required_train = {"split", "prefix_index", "design_id", "draw_id", "status", "input_sha256",
                      "gamma4", "R4_J_kg_K", "cp4_J_kg_K", "species_row_index",
                      "source_registration_sha256", "binary_sha256", "property_manifest_sha256", "source_commit",
                      "f_JetA", "f_HEFA", "f_FT", "f_ATJ", "thrust_fraction"}
    required_named = {"named_case_id", "fuel", "op", "status", "in_product_API", "fuel_parts",
                      "input_sha256", "gamma4", "R4_J_kg_K", "cp4_J_kg_K", "source_registration_sha256",
                      "binary_sha256", "property_manifest_sha256", "source_commit",
                      "prerequisite_registration_sha256", "full_state_sha256", "f_JetA", "f_HEFA", "f_FT", "f_ATJ", "thrust_fraction"}
    train = read_projection(files["teacher_rows.csv"], allowed, required_train)
    named = read_projection(files["named_central_properties.csv"], allowed, required_named)
    validate_source_ids(train, named, reg)
    for row, proof in zip(train, train_proof):
        if any(str(proof.get(key)) != row.get(key) for key in ("design_id", "prefix_index", "draw_id", "input_sha256")):
            raise Blocked("TRAIN row differs from pregeneration input identity")
    for row, proof in zip(named, named_proof):
        if any(str(proof.get(key)) != row.get(key) for key in ("named_case_id", "fuel", "op", "input_sha256")) \
                or str(proof.get("in_product_API")).lower() != row.get("in_product_API", "").lower() \
                or json.loads(row.get("fuel_parts", "null")) != proof.get("fuel_parts"):
            raise Blocked("named row differs from pregeneration input identity")
    for rows, queries, keys in ((train,query_tables["train_queries"],("split","design_id","prefix_index","draw_id","input_sha256")),
                                (named,query_tables["named_queries"],("named_case_id","fuel","op","input_sha256"))):
        for row,query in zip(rows,queries):
            if any(str(query.get(key)) != row.get(key) for key in keys):
                raise Blocked("property row differs from frozen source query keys")
            if finite_number(row["thrust_fraction"],"thrust_fraction") != query.get("thrust_fraction"):
                raise Blocked("property row thrust fraction differs from frozen query")
            alternate = row.get("fuel") == "JetA_dooley2010"
            fractions = fraction_fields(row,alternate=alternate)
            expected_fractions = [query.get(f"f_{name}") for name in ("JetA","HEFA","FT","ATJ")]
            if alternate:
                if expected_fractions != [None]*4:
                    raise Blocked("alternate source query entered product simplex")
            elif fractions != expected_fractions:
                raise Blocked("property row mass fractions differ from frozen source query")
    prerequisite_sha = sha256(main_root / "docs/phase7_p73_a1_registration.json")
    if any(row.get("source_commit") != input_proof["source_commit"] for row in train + named) \
            or any(row.get("prerequisite_registration_sha256") != prerequisite_sha for row in named):
        raise Blocked("source row commit/prerequisite mismatch")
    cases, failures = property_cases(train, named, reg, context, producer_sha, pre_sha)
    coverage = {"expected": 4164, "valid": len(cases), "failed": len(failures), "failures": failures,
                "all_requested_ids_retained":len(cases)+len(failures)==4164,
                "out_of_envelope":sum(row["status"]=="OUT_OF_ENVELOPE" for row in failures),
                "unconverged":sum(row["status"]=="SOURCE_UNCONVERGED" for row in failures)}
    write_json(out / "source_property_coverage.json", coverage)
    if failures:
        raise Blocked("SOURCE_COVERAGE_OR_SCOPE_FAILED")
    import numpy as np  # lazy, ownership already held
    with np.load(files["teacher_species.npz"], allow_pickle=False) as species:
        Y = species["Y4"]
        if Y.ndim != 2 or Y.shape[1] != 492 or not np.isfinite(Y).all() or (Y < 0).any() or np.max(np.abs(Y.sum(axis=1) - 1.0)) > 1e-10:
            raise Blocked("invalid TRAIN species mixture matrix")
        properties = read_object(files["frozen_properties.json"])
        if "species_order" not in species or list(species["species_order"].astype(str)) != properties.get("species_order"):
            raise Blocked("missing canonical TRAIN species order")
        indices = [int(row["species_row_index"]) for row in train]
        if indices != list(range(4096)) or Y.shape[0] != 4096:
            raise Blocked("TRAIN species row alignment mismatch")
        for key in ("design_id", "draw_id", "prefix_index", "input_sha256"):
            if key not in species or list(species[key].astype(str)) != [row[key] for row in train]:
                raise Blocked("TRAIN species canonical row key mismatch")
    frozen = {str(path.relative_to(main_root)): sha256(path) for path in [producer_path, property_input,
              generation_path, log_path, main_root / dep["property_output_manifest"],
              main_root / "docs/phase7_p73_a1_registration.json", *dependency_files, *files.values(),*command_files,
              *(root/name for name in ("terminal.json","reservation.json","released_lease.json"))]}
    frozen.update(command_source_hashes)
    return cases, frozen
