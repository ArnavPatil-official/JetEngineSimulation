"""Actual scientific stages for the resumable, Python-v6 PC workflow."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from .pc_python_runtime import AMENDMENT, PythonContext, source_hashes
from .pc_runtime import read_json, sha256_file, utc, write_once

REGISTRATIONS = {"p73": "docs/phase7_p73_a1_registration.json",
    "saf": "docs/phase8_saf_surrogate_registration.json", "nozzle": "docs/phase8_nozzle_ode_registration.json"}


def artifacts(root, *folders):
    return {p.relative_to(root).as_posix(): sha256_file(p) for folder in folders
            for p in sorted(Path(folder).rglob("*")) if p.is_file() and not p.is_symlink()}


def platform_info():
    from simulation.runtime import power_status
    physical, physical_source = None, "unknown"
    cpu_model, cpu_model_source = platform.processor() or None, "platform.processor"
    memory_bytes, memory_source = None, "unavailable"
    if platform.system() == "Linux":
        try:
            values = [line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.split(":",1)[0].strip() == "model name" and ":" in line]
            if values:
                cpu_model,cpu_model_source=values[0],"/proc/cpuinfo model name"
        except OSError:
            pass
        try:
            fields = next(line.split() for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemTotal:"))
            if len(fields)==3 and fields[2]=="kB":
                memory_bytes,memory_source=int(fields[1])*1024,"/proc/meminfo MemTotal"
        except (OSError,ValueError,StopIteration):
            pass
        try:
            pairs = {(int((p/"topology/physical_package_id").read_text()), int((p/"topology/core_id").read_text()))
                     for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*")
                     if not (p/"online").exists() or (p/"online").read_text().strip() == "1"}
            if pairs and all(a>=0 and b>=0 for a,b in pairs):
                physical, physical_source = len(pairs), "visible online Linux sysfs CPU topology"
        except (OSError,ValueError):
            pass
    elif platform.system() == "Darwin":
        import subprocess
        physical = int(subprocess.check_output(["sysctl","-n","hw.physicalcpu"],text=True))
        physical_source = "sysctl hw.physicalcpu"
    return {"execution_profile": "pc", "platform": platform.platform(), "hardware": {
        "machine": platform.machine(), "processor": platform.processor(), "logical_cores": os.cpu_count(),
        "cpu_model":cpu_model,"cpu_model_source":cpu_model_source,
        "memory_bytes":memory_bytes,"memory_source":memory_source,
        "kernel": platform.release(), "physical_core_count_source":physical_source},
        "physical_cores": physical, "power": power_status()}


class ScientificWorkflow:
    def __init__(self, root, metadata, *, workers=10, backend=None, device="cpu"):
        from simulation.ml_backend import resolve_backend
        self.root, self.metadata = Path(root).resolve(), Path(metadata).resolve()
        self.workers, self.backend, self.device = workers, resolve_backend(backend), device
        self.sources = None
        self.parity = None
        self.reg = {name: read_json(self.root / path) for name, path in REGISTRATIONS.items()}
        self.saf = self.root / self.reg["saf"]["artifact_root"]
        self.p73 = self.root / self.reg["p73"]["outputs"]["directory"]
        self.nozzle = self.root / self.reg["nozzle"]["outputs"]["root"]

    def context(self, registration, *, require_g0=True, **kwargs):
        if self.sources is None:
            self.sources = read_json(self.metadata / "scientific_sources.json")
        if self.parity is None and (self.metadata / "parity20.json").is_file():
            self.parity = read_json(self.metadata / "parity20.json")
        return PythonContext(self.root, registration, sources=self.sources, metadata_dir=self.metadata,
            parity=self.parity, workers=self.workers, require_g0=require_g0, **kwargs)

    def context_factory(self, root, registration, **kwargs):
        if Path(root).resolve() != self.root:
            raise ValueError("Scientific consumer is from another checkout")
        return self.context(registration, **kwargs)

    def owned(self, letter, registration):
        context = self.context(registration)
        run = context.acquire_run(self.metadata / "runs" / letter, context.identity["registration_sha256"])
        run._saf_children = []
        run.record_children([])
        return context, run

    def finish(self, run, folders, *, scientific_verdict="PASS", extra=None, extra_artifacts=None):
        hashes = artifacts(self.root, *folders)
        hashes.update(extra_artifacts or {})
        terminal = run.release({"status": "COMPLETE", "state": "COMPLETE", "exit_code": 0,
            "execution_complete": True, "outputs_complete": True, "scientific_verdict": scientific_verdict,
            "artifact_hashes": hashes, "expected_outputs": list(hashes), "errors": [], **(extra or {})})
        if terminal["status"] == "ERROR":
            raise RuntimeError("; ".join(terminal["errors"]))
        return {"status": "COMPLETE", "execution_complete": True, "scientific_verdict": scientific_verdict,
                "artifacts": {**artifacts(self.root, *folders, run.out), **(extra_artifacts or {})}, **(extra or {})}

    def a(self):
        from simulation.runtime import require_ac
        if sys.version_info[:2] != (3,12):
            raise RuntimeError("Use Python3.12 for the pinned study environment")
        os.environ["CATJET_SIMULATOR_BACKEND"] = "python"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            if os.environ.get(key, "1") != "1":
                raise RuntimeError(f"{key}=1 is required")
            os.environ[key] = "1"
        versions = {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "pandas", "cantera", "PyYAML", "pytest", "torch" if self.backend == "torch" else "mlx")}
        if versions["cantera"] != "3.2.0":
            raise RuntimeError("Python v6 requires Cantera3.2.0")
        if self.workers <= 0 or self.workers > (os.cpu_count() or 1):
            raise RuntimeError("Worker count must fit the visible logical CPU count")
        from simulation.ml_backend import get_backend
        selected = get_backend(self.backend, device=self.device)
        self.sources = source_hashes(self.root)
        write_once(self.metadata / "scientific_sources.json", self.sources)
        info = {"status": "COMPLETE", "execution_complete": True, "platform": platform_info(),
            "versions": versions, "workers": self.workers, "backend": self.backend, "device": self.device,
            "resolved_device":str(selected.device),
            "thread_limits":{key:os.environ[key] for key in
                ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")},
            "training_dtype": "float64" if self.backend == "torch" else "float32", "power": require_ac(),
            "python_executable": sys.executable, "python": sys.version, "cpp_required": False,
            "amendment_sha256": sha256_file(self.root / AMENDMENT)}
        write_once(self.metadata / "environment.json", info)
        return {**info, "artifacts": {str((self.metadata/name).relative_to(self.root)):sha256_file(self.metadata/name)
                                      for name in ("environment.json", "scientific_sources.json")}}

    def b(self):
        """Exactly20 frozen v6 calibration rows, recomputed by Python, no fit."""
        self.context(REGISTRATIONS["p73"],require_g0=False).assert_current()
        import pandas as pd
        for extra in (self.root, self.root / "scripts/optimization", self.root / "scripts/phase8"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        import lto_v5 as v5
        import lto_v6
        from v6_backend import make_model_v6
        from g0_parity import compare
        reg6, split = lto_v6.load_registration_v6(), v5.load_split()
        rows = v5.calibration_rows(split).iloc[:20].copy()
        if len(rows) != 20:
            raise RuntimeError("Frozen parity smoke requires exactly20 rows")
        fit = read_json(self.root / "outputs/phase7/calibration_v6.json")
        model = make_model_v6("python", self.workers)
        started = time.perf_counter()
        try:
            predictions = model.predict(fit["params"], rows)
        finally:
            model.close()
        out = self.metadata / "parity20"
        out.mkdir(exist_ok=False)
        actual = rows.drop(columns=["CO (g/kg)", "HC (g/kg)", "NOx (g/kg)"]).join(predictions)
        actual.to_csv(out / "actual.csv", index=False)
        pd.read_csv(self.root / "outputs/phase7/calibration_v6_rows.csv").iloc[:20].to_csv(out / "frozen.csv", index=False)
        checked = compare(out / "frozen.csv", out / "actual.csv")
        self.parity = {"status": "PASS" if checked["match"] else "FAIL", "rows": 20, "rtol": 1e-9,
            "atol": 1e-12, "backend": "python-v6", "check": checked, "seconds": time.perf_counter()-started,
            "source_hashes": self.sources or read_json(self.metadata / "scientific_sources.json")}
        write_once(self.metadata / "parity20.json", self.parity)
        return {"status": "COMPLETE", "execution_complete": True, "scientific_verdict": self.parity["status"],
                "artifacts": artifacts(self.root, out) | {str((self.metadata / "parity20.json").relative_to(self.root)):
                    sha256_file(self.metadata / "parity20.json")}}

    def c(self):
        from . import p73_a1_cpp as p73
        context = self.context(REGISTRATIONS["p73"])
        reg = self.reg["p73"]
        p73.validate_contract(self.root, reg)
        contract = reg["implementation_contract"]
        fit, profile, hold, historical = (read_json(self.root / contract[key]) for key in
            ("fit_path", "profile_path", "gate_path", "historical_closed_gate_path"))
        p73.validate_exception(profile, fit, hold, historical)
        run = context.acquire_run(self.p73, context.identity["registration_sha256"])
        run.record_children([])
        started = utc()
        try:
            write_once(self.p73 / "environment.json", {"identity": context.identity, "simulator": context.simulator_identity,
                "python": sys.version, "workers": self.workers, "conditional_label": p73.LABEL, **platform_info()})
            (self.p73 / "workers").mkdir()
            (self.p73 / "parity").mkdir()
            write_once(self.p73 / "parity/parity.json", {"status": "PASS", "checks": {"frozen20": self.parity},
                "coverage": "20 frozen Python-v6 calibration rows; no C++ cross-backend claim",
                "simulator": context.simulator_identity, "conditional_label": p73.LABEL})
            protocol, backend = p73.import_protocol(self.root)
            reg6, split = protocol.v6.load_registration_v6(), protocol.v5.load_split()
            ae3 = protocol.v5.load_rows([protocol.AE3_UID], with_targets=False).iloc[0]
            fuels, draws = protocol.fuel_parts(), protocol.load_draws()
            if [name for name, _ in draws] != [f"draw_{i:02d}" for i in range(64)]:
                raise RuntimeError("P7.3 fixed64 draw identities changed")
            with (self.p73 / "command.log").open("x") as log, (self.p73 / "partial_rows.jsonl").open("x") as journal:
                import contextlib
                with contextlib.redirect_stdout(log):
                    def checkpoint(stage, case, frame):
                        journal.write(p73.checkpoint_record(stage, case, frame, context.identity)); journal.flush()
                    summary = p73.study_stage(protocol, backend, reg, reg6, split, fit, ae3, fuels, draws,
                                               context, run, self.p73, checkpoint)
            for name, expected in summary["_sealed_outputs"].items():
                if sha256_file(self.p73 / name) != expected:
                    raise RuntimeError("Fresh P7.3 scientific output drift")
            write_once(self.p73 / "command.exit.json", {"identity": context.identity, "started": started,
                "finished": utc(), "exit_code": 0, "in_process_completed": True, "backend": "python-v6"})
            write_once(self.p73 / "artifact_hashes.json", {"artifacts_sha256": {
                p.relative_to(self.p73).as_posix(): sha256_file(p) for p in self.p73.rglob("*") if p.is_file()},
                "identity": context.identity, "conditional_label": p73.LABEL})
            return self.finish(run, [self.p73], extra={"conditional_label": p73.LABEL})
        except BaseException as exc:
            if not run._released:
                run.release({"status": "ERROR", "exit_code": 1, "errors": [str(exc)], "outputs_complete": False})
            raise

    def d(self):
        from .saf_surrogate import run as saf_run
        from .saf_surrogate.timing import source_diagnostics
        if self.saf.exists():
            raise FileExistsError("SAF generation directory is already consumed")
        context, run = self.owned("d", REGISTRATIONS["saf"])
        reg, reg_sha = self.reg["saf"], context.identity["registration_sha256"]
        try:
            self.saf.mkdir(parents=True)
            # Authenticate producer completion without opening or hashing named targets before the sole score.
            verify_stage_producer(self.root, self.p73, expected_simulator=context.simulator_identity_sha256,
                selected=[str((self.p73/name).relative_to(self.root)) for name in
                          ("environment.json","command.exit.json","artifact_hashes.json","parity/parity.json")])
            prereq = {"registration": REGISTRATIONS["p73"], "registration_sha256": sha256_file(self.root / REGISTRATIONS["p73"]),
                "output": str(self.p73), "terminal_sha256": sha256_file(self.p73 / "terminal.json")}
            write_once(self.saf / "prerequisite.json", prereq)
            write_once(self.saf / "pc_origin.json", {"simulator": context.simulator_identity,
                "simulator_identity_sha256": context.simulator_identity_sha256, "parity20": self.parity,
                "historical_mac_chain_completion_claimed": False})
            saf_run.freeze(self.root, self.saf, reg, reg_sha, context, run, self.backend,
                platform_info=platform_info, source_hashes=self.sources,
                provenance_dependencies={"local_pc_origin":str((self.saf / "pc_origin.json").relative_to(self.root)),
                    "named_prerequisite_terminal":str((self.p73 / "terminal.json").relative_to(self.root))})
            source_diagnostics(self.root, self.saf, reg, context, run, self.backend)
            saf_run.wait_generation(self.root, self.saf, context, run, reg_sha)
            return self.finish(run, [self.saf])
        except BaseException as exc:
            if not run._released:
                run.release({"status":"ERROR", "exit_code":1, "errors":[str(exc)], "outputs_complete":False})
            raise

    def e(self):
        from .saf_surrogate.train import train_all
        context, run = self.owned("e", REGISTRATIONS["saf"])
        try:
            train_all(self.saf, self.reg["saf"], context.identity["registration_sha256"], run,
                      backend=self.backend, device=self.device)
            members = read_json(self.saf / "validation.json")["members"]
            if len(members) != 24 or {(m["arm"],m["N"],m["seed"]) for m in members} != {
                    (arm, size, seed) for arm in ("Mdata","Mphys") for size in (64,256,1024,4096) for seed in (42,43,44)}:
                raise RuntimeError("Completed learning curve does not contain the fixed24 fits")
            hashes = artifacts(self.root, self.saf / "models", self.saf / "training")
            hashes.update({str((self.saf/name).relative_to(self.root)):sha256_file(self.saf/name)
                           for name in ("selection.json", "validation.json", "progress.jsonl")})
            result = self.finish(run, [self.saf / "models", self.saf / "training"], extra={"fit_count":24}, extra_artifacts=hashes)
            result["artifacts"].update(hashes)
            return result
        except BaseException as exc:
            if not run._released:
                run.release({"status":"ERROR", "exit_code":1, "errors":[str(exc)], "outputs_complete":False})
            raise

    def f(self):
        from .saf_surrogate.score import seal_predictions, score_all
        if (self.saf / "score_reservation.json").exists() or (self.saf / "predictions_freeze.json").exists():
            raise RuntimeError("Sole score/prediction seal is consumed; interrupted scoring cannot resume")
        context, run = self.owned("f", REGISTRATIONS["saf"])
        try:
            seal_predictions(self.saf, self.reg["saf"], context.identity["registration_sha256"], context, run, backend=self.backend)
            scored = score_all(self.root, self.saf, self.reg["saf"], context.identity["registration_sha256"], context, run, backend=self.backend)
            write_once(self.saf / "sole_score_summary.json", scored)
            return self.finish(run, [self.saf], scientific_verdict="PASS" if all(scored[k] for k in
                ("fidelity_pass", "ranking_pass", "precision_pass")) else "FAIL")
        except BaseException as exc:
            if not run._released:
                run.release({"status":"ERROR", "exit_code":1, "errors":[str(exc)], "outputs_complete":False})
            raise

    def g(self):
        from .nozzle_ode import run as nozzle
        nozzle.run(self.root, context_factory=self.context_factory, source_loader=load_source_properties,
                   portable=True, backend=self.backend)
        terminal = read_json(self.nozzle / "terminal.json")
        if terminal["status"] == "ERROR" or terminal.get("execution_complete") is not True or terminal.get("outputs_complete") is not True:
            raise RuntimeError("Nozzle execution did not complete: "+terminal["status"])
        return {"status":"COMPLETE", "execution_complete":True, "scientific_verdict":terminal["status"],
                "artifacts":artifacts(self.root, self.nozzle)}

    def h(self):
        return run_timing(self)


def verify_stage_producer(root, output, *, expected_simulator=None, selected=None):
    root, output = Path(root).resolve(), Path(output)
    terminal, reservation, released = (read_json(output/name) for name in
                                      ("terminal.json", "reservation.json", "released_lease.json"))
    identity = terminal["identity"]
    if identity.get("schema") != "pc-python-v6-v1" or terminal.get("status") != "COMPLETE" or terminal.get("exit_code") != 0:
        raise ValueError("Producer is not a complete Python-v6 stage")
    if expected_simulator is not None and identity["simulator_identity_sha256"] != expected_simulator:
        raise ValueError("Producer used another Python simulator identity")
    if reservation["identity"] != identity or released["identity"] != identity or released["state"] != "RELEASED":
        raise ValueError("Producer source/ownership records disagree")
    if terminal["reservation_sha256"] != sha256_file(output/"reservation.json") or released["terminal_sha256"] != sha256_file(output/"terminal.json"):
        raise ValueError("Producer release bytes changed")
    hashes = terminal["artifact_hashes"]
    for name in selected if selected is not None else hashes:
        if name not in hashes or sha256_file(root/name) != hashes[name]:
            raise ValueError("Producer artifact drift: "+name)
    return terminal


def load_source_properties(root, reg, context, out):
    """Validate genuine generation proof; decode only permitted TRAIN/named properties."""
    from .nozzle_ode import inputs
    from .saf_surrogate.registration import verify_artifacts
    root, out = Path(root), Path(out)
    source = root / read_json(root/REGISTRATIONS["saf"])["artifact_root"]
    pre, manifest, generation = (read_json(source/name) for name in
        ("property_inputs_manifest.json", "property_manifest.json", "generation_terminal.json"))
    if generation.get("state") != "COMPLETE" or generation.get("simulator_identity_sha256") != context.simulator_identity_sha256:
        raise ValueError("Nozzle properties lack actual completed Python generation proof")
    if pre["simulator_identity_sha256"] != context.simulator_identity_sha256:
        raise ValueError("Nozzle/source Python identity differs")
    verify_artifacts(source, manifest["artifacts"])
    train_ids, named_ids = inputs.expected_source_ids(reg)
    inputs.write_json(out / "case_selection.json", {"train_ids":train_ids, "named_ids":named_ids})
    producer_sha = sha256_file(root / REGISTRATIONS["saf"])
    pre_sha = sha256_file(source / "property_inputs_manifest.json")
    if pre.get("registration_sha256") != producer_sha or manifest.get("registration_sha256") != producer_sha \
            or manifest.get("property_inputs_manifest") != {"path":"property_inputs_manifest.json", "sha256":pre_sha} \
            or manifest.get("producer_terminal") != {"path":"generation_terminal.json", "sha256":sha256_file(source/"generation_terminal.json")}:
        raise ValueError("Nozzle generation property manifest binding differs")
    stage = Path(context.pc_lease_path).parent / "runs/d"
    projection = [str((source/name).relative_to(root)) for name in
        ("property_inputs_manifest.json", "property_manifest.json", "generation_terminal.json", "teacher_rows.csv",
         "teacher_species.npz", "named_central_properties.csv", "frozen_properties.json",
         "splits/train.json", "splits/named_central.json")]
    verify_stage_producer(root, stage, expected_simulator=context.simulator_identity_sha256, selected=projection)
    allowed = set(reg["dependencies"]["columns_read"])
    common = {"status", "input_sha256", "gamma4", "R4_J_kg_K", "cp4_J_kg_K", "source_registration_sha256",
              "binary_sha256", "property_manifest_sha256", "source_commit", "f_JetA", "f_HEFA", "f_FT", "f_ATJ", "thrust_fraction"}
    train = inputs.read_projection(source/"teacher_rows.csv", allowed,
        common | {"split", "prefix_index", "design_id", "draw_id", "species_row_index"})
    named = inputs.read_projection(source/"named_central_properties.csv", allowed,
        common | {"named_case_id", "fuel", "op", "in_product_API", "fuel_parts", "prerequisite_registration_sha256", "full_state_sha256"})
    inputs.validate_source_ids(train, named, reg)
    for name, rows, keys in (("train", train, ("design_id","prefix_index","draw_id","input_sha256")),
                            ("named_central", named, ("named_case_id","fuel","op","input_sha256"))):
        cases = pre["cases"][name]
        queries = read_json(source / f"splits/{name}.json")
        if len(cases) != len(rows) or len(queries) != len(rows):
            raise ValueError("Nozzle property case count changed")
        for row, case, query in zip(rows, cases, queries):
            if any(row[key] != str(case[key]) or row[key] != str(query[key]) for key in keys):
                raise ValueError("Nozzle property case identity differs from frozen input")
    properties, failures = inputs.property_cases(train, named, reg, context, producer_sha, pre_sha)
    if failures or len(properties) != 4164:
        inputs.write_json(out/"source_property_coverage.json", {"valid":len(properties), "failures":failures})
        raise ValueError("Complete4164 property coverage is required; failed rows were retained")
    import numpy as np
    with np.load(source/"teacher_species.npz", allow_pickle=False) as archive:
        y = archive["Y4"]
        if y.shape != (4096,492) or not np.isfinite(y).all() or (y<0).any() or np.max(np.abs(y.sum(axis=1)-1))>1e-10:
            raise ValueError("TRAIN species properties are invalid")
        frozen_properties = read_json(source/"frozen_properties.json")
        if list(archive["species_order"].astype(str)) != frozen_properties["species_order"] \
                or [int(row["species_row_index"]) for row in train] != list(range(4096)):
            raise ValueError("TRAIN species canonical ordering changed")
        for key in ("design_id","draw_id","prefix_index","input_sha256"):
            if list(archive[key].astype(str)) != [row[key] for row in train]:
                raise ValueError("TRAIN species identity changed")
    hashes = {str((source/name).relative_to(root)):sha256_file(source/name) for name in
        ("property_inputs_manifest.json", "property_manifest.json", "generation_terminal.json", "teacher_rows.csv",
         "teacher_species.npz", "named_central_properties.csv", "frozen_properties.json",
         "splits/train.json", "splits/named_central.json")}
    inputs.write_json(out / "source_manifest.json", {"simulator":context.simulator_identity,
        "simulator_identity_sha256":context.simulator_identity_sha256, "source_hashes":hashes,
        "property_count":len(properties), "columns_read":sorted(allowed)})
    return properties, hashes


def run_timing(workflow):
    """Measure actual Python-v6 at1/10 workers and the selected CPU64 evaluator."""
    from .saf_surrogate.models import load_product, summarize_draws
    from .saf_surrogate.run import command_spec
    from .saf_surrogate.teacher import ParallelTeacher
    from .saf_surrogate.timing import break_even, gpu_measurements
    from .saf_surrogate.study import screen
    from .saf_surrogate.score import deployment_receipt
    context, run = workflow.owned("h", REGISTRATIONS["saf"])
    output, root, reg = workflow.saf, workflow.root, workflow.reg["saf"]
    started = time.perf_counter()
    try:
        product = load_product(output/"product.json", require_deployment=False, backend=workflow.backend)
        queries = read_json(output/"splits/study.json")
        def timed(call, batch):
            before = time.perf_counter_ns()
            predictions = call(batch)
            summarize_draws(batch, predictions)
            return (time.perf_counter_ns()-before)/1e9
        measurements, requests = [], 0
        for workers in (1,10):
            stage = f"timing_python_workers{workers}"
            spec = command_spec(root, output, context, context.identity["registration_sha256"], stage,
                                [sys.executable,"P8-S",stage])
            spec["workers"] = workers
            write_once(root/spec["owner_lease"]["snapshot_path"], run.lease_path.read_bytes())
            write_once(root/spec["command_spec_path"], spec)
            teacher = ParallelTeacher(spec, product.properties, product.public, product.draws, workers, run=run)
            batches = []
            try:
                for count in (1,64,4096):
                    batch = queries[:count]
                    cold = timed(teacher,batch); requests += count
                    product_cold = timed(product.predict,batch)
                    warm = timed(teacher,batch); requests += count
                    product_warm = timed(product.predict,batch)
                    sim, surrogate = [], []
                    for _ in range(7):
                        run.assert_current(); sim.append(timed(teacher,batch)); requests += count
                        surrogate.append(timed(product.predict,batch))
                    batches.append({"rows":count, "python_cold_seconds":cold, "product_cold_seconds":product_cold,
                        "python_warmup_seconds":warm, "product_warmup_seconds":product_warm,
                        "python_seconds":sim, "product_seconds":surrogate,
                        "python_median_seconds":statistics.median(sim), "product_median_seconds":statistics.median(surrogate),
                        "speedup":statistics.median(sim)/statistics.median(surrogate)})
            finally:
                teacher.close()
            measurements.append({"workers":workers, "batches":batches})
        study = screen(root, output, reg, context.identity["registration_sha256"], context, run, backend=workflow.backend)
        scored = read_json(output/"sole_score_summary.json")
        gpu = gpu_measurements(output, product, queries, run, workflow.backend, workflow.device)
        bulk = measurements[-1]["batches"][-1]
        # Charge actual completed acquisition/training/scoring and this phase.
        setup = sum(read_json(workflow.metadata/"checkpoints"/f"{letter}.json")["elapsed_seconds"]
                    for letter in "cdefg") + time.perf_counter()-started
        sim, surrogate = bulk["python_median_seconds"]/4096, bulk["product_median_seconds"]/4096
        crossover = break_even(setup, sim, surrogate)
        cpu_pass = bool(scored["fidelity_pass"] and scored["ranking_pass"] and scored["precision_pass"]
            and bulk["speedup"]>1 and crossover is not None and crossover<=640000
            and study["invalid_prediction_rows"]==0 and study["selected_reference_converged"]==640)
        timing = {"state":"COMPLETE", "simulator":"python-v6", "simulator_identity_sha256":context.simulator_identity_sha256,
            "scoring_backend":"torch" if workflow.backend=="torch" else "numpy", "scoring_dtype":"float64",
            "worker_measurements":measurements, "teacher_full_cycle_requests":requests,
            "total_setup_seconds":setup, "break_even_queries":crossover,
            "cpu64_bulk_speedup":bulk["speedup"], "gpu_available":gpu["available"],
            "cpu_operational_pass":cpu_pass, "operational_pass":cpu_pass,
            "operational_scope":"Prospective Python PC CPU64 timing/break-even; no GPU prerequisite or GPU speed claim",
            "projected_simulator_640k_seconds":640000*sim, "measured_product_640k_seconds":study["cpu64_total_seconds"]}
        write_once(output/"pc_timing_break_even.json", timing)
        write_once(output/"timing.json", timing)
        report = {"registration_id":reg["id"], "registration_sha256":context.identity["registration_sha256"],
            "execution_profile":"pc", "simulator":context.simulator_identity, "metrics":scored,
            "timing":timing, "study":study, "historical_mac_chain_completion_claimed":False}
        write_once(output/"report.json", report)
        deployment_receipt(output,reg,context.identity["registration_sha256"],scored,timing)
        verdict = read_json(output/"deployment_receipt.json")["state"]
        result = workflow.finish(run,[output],scientific_verdict=verdict)
        history = {"schema":"pc-python-stage-history-v1", "simulator":context.simulator_identity,
            "simulator_identity_sha256":context.simulator_identity_sha256, "source_hashes":context.identity["source_hashes"],
            "stages":{letter:{"path":str((workflow.metadata/"runs"/letter).relative_to(root)),
                "terminal_sha256":sha256_file(workflow.metadata/"runs"/letter/"terminal.json")} for letter in "defh"}}
        write_once(output/"pc_pipeline_provenance.json",history)
        result["artifacts"].update({str((output/"pc_pipeline_provenance.json").relative_to(root)):
                                      sha256_file(output/"pc_pipeline_provenance.json")})
        return result
    except BaseException as exc:
        if not run._released:
            run.release({"status":"ERROR", "exit_code":1, "errors":[str(exc)], "outputs_complete":False})
        raise


def validate_product_provenance(root, output_dir, *, artifact_paths=None, expected_simulator_identity_sha256=None,
                                expected_binary_sha256=None):
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Product leaves the current checkout")
    history = read_json(output/"pc_pipeline_provenance.json")
    if history.get("schema") != "pc-python-stage-history-v1" or (expected_simulator_identity_sha256 is not None
            and history["simulator_identity_sha256"] != expected_simulator_identity_sha256):
        raise ValueError("Product Python-v6 provenance differs")
    from .saf_surrogate.registration import verify_artifacts
    verify_artifacts(root, history["source_hashes"])
    proofs = {}
    for letter in "defh":
        entry = history["stages"][letter]
        folder = (root/entry["path"]).resolve()
        if not folder.is_relative_to(root) or sha256_file(folder/"terminal.json") != entry["terminal_sha256"]:
            raise ValueError("Product stage receipt changed")
        proofs[letter] = verify_stage_producer(root,folder,expected_simulator=history["simulator_identity_sha256"],
            selected=artifact_paths if letter=="h" else [])
    if history["source_hashes"] != proofs["h"]["identity"]["source_hashes"] or history["simulator"] != proofs["h"]["identity"]["simulator"]:
        raise ValueError("Product history sources differ from authenticated stage identity")
    return {"status":"COMPLETE", "identity":proofs["h"]["identity"],
            "simulator_identity_sha256":history["simulator_identity_sha256"], "core_sha256":None}
