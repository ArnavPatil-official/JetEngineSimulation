"""Portable nozzle, public-product checks and published-only quantitative freeze.

These stages record local PC evidence. They never dispatch ladder A2, infer a
historical Track 4 result, reopen sealed labels, or create a freeze tag.
Scientific imports and computations occur only inside an acquired local run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

NOZZLE_REGISTRATION = "docs/phase8_nozzle_ode_registration.json"
PRODUCT_REGISTRATION = "docs/phase8_screening_tool_registration.json"
SAF_REGISTRATION = "docs/phase8_saf_surrogate_registration.json"
SAF_OUTPUT = "outputs/phase8/saf_surrogate/attempt_001"
CHECK_OUTPUT = "outputs/phase8/screening_operations/product_verification_run"


class FinishError(RuntimeError):
    """Execution, source identity or published evidence is invalid."""


def _runtime():
    from scripts.phase8 import pc_runtime
    return pc_runtime


def _validate(*args, **kwargs):
    from scripts.phase8.pc_saf import validate_local_terminal
    return validate_local_terminal(*args, **kwargs)


def _context(root, registration, factory):
    return factory(Path(root).resolve(), registration)


def _terminal(root, output):
    terminal = _runtime().read_json(Path(root) / output / "terminal.json")
    if terminal.get("status") == "ERROR":
        raise FinishError("; ".join(terminal.get("errors", [])) or "PC stage failed")
    return terminal


def _source_properties_pc(root, reg, context, out):
    """Read only TRAIN/named property projections from an authenticated PC fit."""
    from scripts.phase8.nozzle_ode import inputs
    rt = _runtime()
    dep = reg["dependencies"]
    train_ids, named_ids = inputs.expected_source_ids(reg)
    inputs.write_json(out / "case_selection.json", {"train_ids":train_ids, "named_ids":named_ids})
    producer = inputs.safe_path(root, dep["product_rows"]).parent
    names = ("property_inputs_manifest.json", "property_manifest.json", "generation_terminal.json",
             "teacher_rows.csv", "teacher_species.npz", "named_central_properties.csv", "frozen_properties.json",
             "splits/train.json", "splits/named_central.json", "proofs/generation_command_spec.json",
             "proofs/generation_handshake.json", "proofs/generation_exit.json", "proofs/generation.log",
             "proofs/generation_owner_lease.json")
    paths = {name:inputs.safe_path(root, str((producer / name).relative_to(root))) for name in names}
    projection = [str(path.relative_to(root)) for path in paths.values()]
    release = _validate(root, dep["source_registration"], producer,
                        expected_binary_sha256=context.binary_sha256,
                        artifact_paths=projection, allow_scientific_fail=True)
    producer_sha = inputs.sha256(root / dep["source_registration"])
    producer_identity = {**context.identity, "registration_sha256":producer_sha}
    if release.get("identity") != producer_identity:
        raise FinishError("PC property producer identity differs from local nozzle context")
    pre, manifest, generated = (inputs.read_object(paths[name]) for name in names[:3])
    pre_sha = inputs.sha256(paths["property_inputs_manifest.json"])
    binary = {"path":str(context.binary_path.relative_to(root)), "sha256":context.binary_sha256}
    for proof in (pre, manifest):
        if proof.get("schema_version") != 1 or proof.get("registration_sha256") != producer_sha \
                or proof.get("registration_id") != dep["source_registration_id"] \
                or proof.get("consumer_identity") != producer_identity or proof.get("binary") != binary \
                or proof.get("counts") != {"train":4096, "named_central":68}:
            raise FinishError("PC property manifest registration, identity, core or counts changed")
    if manifest.get("property_inputs_manifest") != {"path":"property_inputs_manifest.json", "sha256":pre_sha} \
            or manifest.get("producer_terminal") != {"path":"generation_terminal.json", "sha256":inputs.sha256(paths["generation_terminal.json"])} \
            or manifest.get("case_identity") != pre.get("cases"):
        raise FinishError("PC property pre-generation and published manifests disagree")
    if generated.get("state") != "COMPLETE" or generated.get("scientific_coverage") != "PASS" \
            or generated.get("start_identity") != producer_identity or generated.get("end_identity") != producer_identity \
            or generated.get("binary_sha256") != context.binary_sha256:
        raise FinishError("PC property generation did not complete with unchanged local identity")
    counts = generated.get("counts", {})
    expected_counts = {"train_requested":4096, "validation_requested":1024, "test_requested":2048,
                       "ranking_requested":4096, "named_central_requested":68, "total_requested":11332}
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise FinishError("PC property generation did not retain the fixed acquisition budget")
    launch = generated.get("launch", {})
    handshake = inputs.read_object(paths["proofs/generation_handshake.json"])
    exited = inputs.read_object(paths["proofs/generation_exit.json"])
    if launch.get("waited") is not True or launch.get("exit_code") != 0 \
            or exited.get("waited") is not True or exited.get("exit_code") != 0 \
            or exited.get("end_identity") != producer_identity or exited.get("identity_problems") != [] \
            or any(exited.get(key) != handshake.get(key) or exited.get(key) != launch.get(key)
                   for key in ("pid", "birth", "argv")):
        raise FinishError("PC property generation has no consistent successful waited child")
    allowed_artifacts = {name:inputs.sha256(paths[name]) for name in
                         ("teacher_rows.csv", "teacher_species.npz", "named_central_properties.csv", "frozen_properties.json")}
    if manifest.get("artifacts") != allowed_artifacts:
        raise FinishError("PC published property artifact hashes differ")
    for name, value in allowed_artifacts.items():
        if generated.get("outputs", {}).get(str(paths[name].relative_to(root))) != value:
            raise FinishError("PC property generation output hash differs from published bytes")
    allowed = set(dep["columns_read"])
    common = {"status", "input_sha256", "gamma4", "R4_J_kg_K", "cp4_J_kg_K",
              "source_registration_sha256", "binary_sha256", "property_manifest_sha256", "source_commit",
              "f_JetA", "f_HEFA", "f_FT", "f_ATJ", "thrust_fraction"}
    train = inputs.read_projection(paths["teacher_rows.csv"], allowed,
        common | {"split", "prefix_index", "design_id", "draw_id", "species_row_index"})
    named = inputs.read_projection(paths["named_central_properties.csv"], allowed,
        common | {"named_case_id", "fuel", "op", "in_product_API", "fuel_parts", "prerequisite_registration_sha256", "full_state_sha256"})
    inputs.validate_source_ids(train, named, reg)
    cases = pre.get("cases", {})
    for name, rows, keys in (("train", train, ("design_id", "prefix_index", "draw_id", "input_sha256")),
                             ("named_central", named, ("named_case_id", "fuel", "op", "input_sha256"))):
        frozen = cases.get(name, [])
        queries = json.loads(paths["splits/" + name + ".json"].read_text())
        if len(frozen) != len(rows) or len(queries) != len(rows):
            raise FinishError("PC property case/query count changed")
        for row, proof, query in zip(rows, frozen, queries):
            if any(row.get(key) != str(proof.get(key)) or row.get(key) != str(query.get(key)) for key in keys):
                raise FinishError("PC property row differs from frozen canonical case")
            alternate = row.get("fuel") == "JetA_dooley2010"
            fractions = inputs.fraction_fields(row, alternate=alternate)
            if not alternate and fractions != [query.get("f_" + fuel) for fuel in ("JetA", "HEFA", "FT", "ATJ")]:
                raise FinishError("PC property source simplex changed")
            if inputs.finite_number(row["thrust_fraction"], "thrust_fraction") != query.get("thrust_fraction"):
                raise FinishError("PC property source thrust fraction changed")
            if row.get("source_commit") != pre.get("source_commit"):
                raise FinishError("PC property source commit changed")
    property_cases, failures = inputs.property_cases(train, named, reg, context, producer_sha, pre_sha)
    inputs.write_json(out / "source_property_coverage.json", {
        "expected":4164, "valid":len(property_cases), "failed":len(failures), "failures":failures,
        "all_requested_ids_retained":len(property_cases)+len(failures)==4164,
        "producer_status":release["status"], "producer_scientific_verdict":release.get("scientific_verdict"),
        "producer_terminal_sha256":release["terminal_sha256"], "execution_profile":"pc"})
    if failures:
        raise inputs.Blocked("SOURCE_COVERAGE_OR_SCOPE_FAILED")
    import numpy as np
    with np.load(paths["teacher_species.npz"], allow_pickle=False) as species:
        y = species["Y4"]
        if y.shape != (4096, 492) or not np.isfinite(y).all() or (y < 0).any() \
                or np.max(np.abs(y.sum(axis=1)-1)) > 1e-10:
            raise FinishError("Invalid complete TRAIN4096 species mixture matrix")
        properties = inputs.read_object(paths["frozen_properties.json"])
        if list(species["species_order"].astype(str)) != properties.get("species_order") \
                or [int(row["species_row_index"]) for row in train] != list(range(4096)):
            raise FinishError("TRAIN species canonical row/order mismatch")
        for key in ("design_id", "draw_id", "prefix_index", "input_sha256"):
            if list(species[key].astype(str)) != [row[key] for row in train]:
                raise FinishError("TRAIN species case identity mismatch")
    frozen = {name:rt.sha256_file(root/name) for name in projection}
    for name in (dep["source_registration"], "docs/phase7_p73_a1_registration.json"):
        frozen[name] = rt.sha256_file(root/name)
    return property_cases, frozen


def run_nozzle(root, context_factory):
    """Run the unchanged CPU float64 study, with honest inherited evidence status."""
    root = Path(root).resolve()
    rt = _runtime()
    reg = rt.read_json(root / NOZZLE_REGISTRATION)
    dep = reg["dependencies"]
    required = [dep[key] for key in ("product_rows", "product_species", "named_properties", "property_output_manifest")]
    missing = [name for name in required if not (root/name).is_file()]
    if missing:
        return {"status":"INCOMPLETE", "execution_complete":False, "missing":missing,
                "reason":"Nozzle requires complete published SAF source properties"}
    from scripts.phase8.nozzle_ode import run as nozzle
    nozzle.run(root, context_factory=context_factory, source_loader=_source_properties_pc, portable=True)
    terminal = _terminal(root, reg["outputs"]["root"])
    if terminal["status"] == "BLOCKED":
        terminal = {**terminal, "status":"INCOMPLETE"}
    report_path = root / reg["outputs"]["root"] / "report.json"
    if report_path.is_file():
        report = rt.read_json(report_path)
        terminal = {**terminal, "registered_status":report["registered_status"],
                    "original_shock_oracle_status":report["original_shock_oracle_status"]}
    return terminal


def _junit(path):
    doc = ElementTree.parse(path).getroot()
    suites, cases = list(doc.iter("testsuite")), list(doc.iter("testcase"))
    totals = {key:sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if not suites or totals["tests"] != len(cases) or not cases:
        raise FinishError("Product test JUnit has incomplete test coverage")
    return totals


def check_product(root, context_factory):
    """Test the actual public CPU64 product and wait for pure regression tests."""
    root = Path(root).resolve()
    rt = _runtime()
    context = _context(root, PRODUCT_REGISTRATION, context_factory)
    owned = context.acquire_run(CHECK_OUTPUT, context.identity["registration_sha256"])
    output, started = owned.out, time.perf_counter()
    errors, status, complete = [], "ERROR", False
    smoke, totals, child = {}, {}, None
    try:
        owned.assert_current()
        bundle = root / SAF_OUTPUT / "product.json"
        if not bundle.is_file():
            status = "INCOMPLETE"
            smoke = {"status":"INCOMPLETE", "reason":"Published Product bundle is absent"}
        else:
            from scripts.phase8 import screening_product
            producer = _validate(root, SAF_REGISTRATION, SAF_OUTPUT,
                expected_binary_sha256=context.binary_sha256,
                artifact_paths=[str(bundle.relative_to(root))], allow_scientific_fail=True)
            candidates = [{"id":"JetA", "JetA":1.0}, {"id":"invalid", "JetA":-1.0, "HEFA":2.0}]
            if producer["status"] == "FAIL":
                try:
                    screening_product.screen_blends(candidates, model=bundle)
                except ValueError as exc:
                    smoke = {"status":"PASS", "deployment_ready":False,
                             "reason":"Completed scientific FAIL is refused by default loader", "loader_error":str(exc)}
                else:
                    raise FinishError("Scientific FAIL was accepted by the public deployment loader")
            else:
                result = screening_product.screen_blends(candidates, model=bundle)
                rows = result["candidates"]
                if result["draw_count"] != 64 or rows[0]["status"] != "PREDICTED" \
                        or rows[1]["status"] != "INVALID_INPUT" or rows[1]["predictions"] is not None \
                        or result["ranking"] != ["JetA"]:
                    raise FinishError("Public Product draw/invalid-input/ranking check failed")
                smoke = {"status":"PASS", "deployment_ready":True, "response":result}
            owned.assert_current()
            tests = ["tests/test_phase8_screening_product.py", "tests/test_phase8_pc_finish.py"]
            junit = output / "junit.xml"
            argv = [sys.executable, "-m", "pytest", *tests, "-v", "--junitxml="+str(junit)]
            spec = {"argv":argv, "workdir":str(root), "interpreter":sys.executable,
                    "interpreter_sha256":rt.sha256_file(Path(sys.executable).resolve()),
                    "identity":context.identity, "tests":{name:rt.sha256_file(root/name) for name in tests},
                    "execution_profile":"pc", "started_utc":rt.utc()}
            rt.write_once(output / "command_spec.json", spec)
            with (output / "pytest.log").open("xb") as log:
                child = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                birth = rt.process_birth(child.pid)
                if not birth or birth == rt.DEAD:
                    child.terminate(); child.wait()
                    raise FinishError("Product test child birth was unreadable")
                owned.record_children([{"pid":child.pid, "birth":birth, "argv":argv}])
                exit_code = child.wait()
            totals = _junit(junit)
            owned.assert_current()
            status = "PASS" if exit_code == 0 and not any(totals[key] for key in ("failures", "errors", "skipped")) else "FAIL"
            rt.write_once(output / "command.exit.json", {**spec, "pid":child.pid, "birth":birth,
                "waited":True, "exit_code":exit_code, "ended_utc":rt.utc(), "junit":totals,
                "in_process_completed":True, "waited_child_exit":"recorded_pytest_child",
                "junit_sha256":rt.sha256_file(junit), "log_sha256":rt.sha256_file(output/"pytest.log")})
            complete = True
        rt.write_once(output / "product_checks.json", {"status":status, "execution_profile":"pc", "smoke":smoke,
            "junit":totals, "legacy_verification_receipt":"INCOMPLETE; this local check is not a Mac receipt"})
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        status = "ERROR"
    finally:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait()
        hashes = {str(p.relative_to(root)):rt.sha256_file(p) for p in output.rglob("*") if p.is_file()}
        terminal = owned.release({"status":status, "exit_code":0 if status == "PASS" else 1, "errors":errors,
            "execution_complete":complete, "outputs_complete":complete,
            "scientific_verdict":"PASS" if status == "PASS" else status,
            "expected_outputs":list(hashes), "artifact_hashes":hashes,
            "execution_profile":"pc", "wall_s":time.perf_counter()-started})
    if terminal["status"] == "ERROR":
        raise FinishError("; ".join(terminal.get("errors", [])))
    return terminal


class _PublishedPC:
    """Adapter to the existing figure readers, authenticating actual PC outputs."""
    def __init__(self, context, reg):
        from scripts.phase8 import freeze_screening as figures
        rt = _runtime()
        self.root, self.context = context.root, context
        self.hashes, self.statuses, self.proofs, self.descriptors = {}, {}, {}, {}
        self.gate = SimpleNamespace(read=rt.read_json)
        groups = {}
        g0_path = context.identity.get("g0_record_path") or context.original_context["g0"].get("evidence_path")
        for original in reg["published_inputs"]:
            entry = dict(original)
            role = entry["role"]
            if role == "G0":
                if not g0_path:
                    raise figures.Incomplete("Actual local PC G0 evidence path is missing")
                actual = Path(g0_path)
                entry["path"] = str(actual.relative_to(self.root)) if actual.is_absolute() else str(actual)
            if role not in figures.INPUT_NAMES or role in self.descriptors \
                    or Path(entry["path"]).name != figures.INPUT_NAMES[role]:
                raise figures.Incomplete("Published role/filename differs from quantitative figure schema")
            path = figures.safe_path(self.root, entry["path"])
            if not path.is_file():
                raise figures.Incomplete("Published artifact missing: "+entry["path"])
            digest = rt.sha256_file(path)
            self.descriptors[role], self.hashes[entry["path"]] = entry, digest
            if "expected_sha256" in entry:
                if digest != entry["expected_sha256"]:
                    raise FinishError("Original pyCycle published evidence hash changed")
                self.statuses[role] = "HASH_BOUND_ORIGINAL"
            elif role == "G0":
                current = rt.read_json(path)
                if current != context.original_context["g0"]:
                    # evidence_path is metadata attached by the pipeline, not a scientific result field.
                    proven = {k:v for k,v in context.original_context["g0"].items() if k != "evidence_path"}
                    if current != proven:
                        raise FinishError("Published local G0 differs from captured PC context")
                self.statuses[role] = current["verdict"]
                self.proofs[role] = {"path":entry["path"], "sha256":digest, "execution_profile":"pc"}
            else:
                groups.setdefault((entry["producer_registration"], entry["producer_output_dir"]), []).append(role)
        if not figures.REQUIRED_ROLES <= set(self.descriptors):
            raise figures.Incomplete("Quantitative published role coverage is incomplete")
        for (registration, output), roles in groups.items():
            proof = _validate(self.root, registration, output, expected_binary_sha256=context.binary_sha256,
                artifact_paths=[self.descriptors[role]["path"] for role in roles], allow_scientific_fail=True)
            for role in roles:
                name = self.descriptors[role]["path"]
                if proof["artifact_hashes"].get(name) != self.hashes[name]:
                    raise FinishError("Published quantitative input differs from producer terminal")
                self.proofs[role], self.statuses[role] = proof, proof["status"]

    def assert_current(self):
        from scripts.phase8 import freeze_screening as figures
        for name, value in self.hashes.items():
            if _runtime().sha256_file(figures.safe_path(self.root, name)) != value:
                raise FinishError("Published PC input changed: "+name)

    def json(self, role):
        from scripts.phase8.freeze_screening import Published
        return Published.json(self, role)

    def rows(self, role, columns):
        from scripts.phase8.freeze_screening import Published
        return Published.rows(self, role, columns)


def freeze(root, context_factory):
    """Export real published figures/numbers, retaining missing inherited proof."""
    from scripts.phase8 import freeze_screening as figures
    root = Path(root).resolve()
    rt = _runtime()
    context = _context(root, PRODUCT_REGISTRATION, context_factory)
    reg = rt.read_json(root / PRODUCT_REGISTRATION)
    owned = context.acquire_run(reg["outputs"]["root"], context.identity["registration_sha256"])
    out, rows, completed, errors = owned.out, [], [], []
    status, data, inherited = "INCOMPLETE", None, {}
    rt.write_once(out / "command_spec.json", {"identity":context.identity, "execution_profile":"pc",
        "owner_pid":os.getpid(), "argv":list(sys.argv), "workdir":str(root),
        "registration_sha256":context.identity["registration_sha256"], "started_utc":rt.utc()})
    try:
        data = _PublishedPC(context, reg)
        nozzle_report = data.json("nozzle_report")
        inherited["original_track4"] = nozzle_report.get("original_shock_oracle_status", "INCOMPLETE")
        def guard():
            owned.assert_current(); data.assert_current()
        guard()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"font.size":9, "pdf.fonttype":42})
        for prefix in figures.PREFIXES:
            before = len(rows)
            try:
                guard()
                figure = figures.DRAW[prefix](data, plt, rows)
                verdict = data.statuses["G0" if prefix == "g0_parity" else prefix]
                verdict += "; local PC diagnostic; original Track 4 " + inherited["original_track4"]
                figures.save(figure, prefix, out, plt, guard, verdict)
                completed.append(prefix)
            except figures.Incomplete as exc:
                del rows[before:]
                errors.append({"figure":prefix, "missing":str(exc)})
                plt.close("all")
        if rows:
            figures.numbers_table(out, rows)
            with (out/"NUMBERS.md").open("a") as stream:
                stream.write("\nLocal PC diagnostic package. Original Track 4 prerequisite: "
                             + inherited["original_track4"] + ".\n")
        checks = root / CHECK_OUTPUT / "terminal.json"
        inherited["local_product_checks"] = rt.read_json(checks)["status"] if checks.is_file() else "INCOMPLETE"
        if inherited["local_product_checks"] == "PASS":
            _validate(root, PRODUCT_REGISTRATION, CHECK_OUTPUT,
                expected_binary_sha256=context.binary_sha256,
                artifact_paths=[str((checks.parent/name).relative_to(root)) for name in
                                ("product_checks.json", "command_spec.json", "command.exit.json", "pytest.log", "junit.xml")])
        if completed == list(figures.PREFIXES) and not errors and all(value == "PASS" for value in inherited.values()):
            status = "FAIL" if "FAIL" in data.statuses.values() else "PASS"
        guard()
    except figures.Incomplete as exc:
        errors.append({"stage":"published_evidence", "missing":str(exc)})
    except Exception as exc:
        status = "ERROR"
        errors.append({"stage":"freeze", "error":f"{type(exc).__name__}: {exc}"})
    expected = [str((out/name).relative_to(root)) for name in figures.required_outputs()]
    rt.write_once(out / "command.exit.json", {"identity":context.identity, "execution_profile":"pc",
        "owner_pid":os.getpid(), "status":status, "exit_code":0 if status == "PASS" else 1,
        "in_process_completed":True, "waited_child_exit":"not_applicable_synchronous_owner", "ended_utc":rt.utc()})
    expected += [str((out/name).relative_to(root)) for name in ("command_spec.json", "command.exit.json")]
    hashes = {str(p.relative_to(root)):rt.sha256_file(p) for p in out.rglob("*") if p.is_file()}
    receipt = {"schema":1, "status":status, "execution_profile":"pc", "identity":context.identity,
        "scientific_verdict":"FAIL" if data and "FAIL" in data.statuses.values() else "PASS" if status == "PASS" else "INCOMPLETE",
        "conditional_label":figures.LABEL, "completed_figures":completed, "numeric_rows":len(rows),
        "input_sha256":data.hashes if data else {}, "producer_status":data.statuses if data else {},
        "inherited_prerequisites":inherited, "artifact_hashes":hashes, "errors":errors,
        "tag_created":False, "local_tag":reg["outputs"]["local_tag"]}
    rt.write_once(out / "freeze_receipt.json", receipt)
    hashes[str((out/"freeze_receipt.json").relative_to(root))] = rt.sha256_file(out/"freeze_receipt.json")
    terminal = owned.release({"status":status, "exit_code":0 if status == "PASS" else 1,
        "scientific_verdict":receipt["scientific_verdict"], "execution_complete":status in {"PASS", "FAIL"},
        "outputs_complete":status in {"PASS", "FAIL"}, "expected_outputs":expected+[str((out/"freeze_receipt.json").relative_to(root))],
        "artifact_hashes":hashes, "errors":[str(e) for e in errors], "execution_profile":"pc", "tag_created":False})
    if terminal["status"] == "ERROR":
        raise FinishError("; ".join(terminal.get("errors", [])))
    return terminal
