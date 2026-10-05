"""Full-state v6 adapter and public verification factory without target fitting."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .inputs import state_for
from .postprocess import derived_outputs
from .registration import read_json, sha256_file
from .thermo import Thermo


def load_selected_core(binary_path, expected_sha256):
    path = Path(binary_path).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("Selected core hash drift")
    existing = sys.modules.get("catjet_core")
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError("Foreign core was already imported")
        return existing
    spec = importlib.util.spec_from_file_location("catjet_core", path)
    if spec is None or spec.loader is None:
        raise ValueError("Selected core cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules["catjet_core"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("catjet_core", None)
        raise
    if Path(module.__file__).resolve() != path or sha256_file(path) != expected_sha256:
        raise ValueError("Imported core identity mismatch")
    return module


class Verifier:
    def __init__(self, root, binary_path, binary_sha256, properties, public, draws, *, assert_current):
        assert_current()
        self.root = Path(root)
        self.binary_sha256 = binary_sha256
        self.properties, self.public, self.draws = properties, public, draws
        self.thermo = Thermo(properties)
        self.assert_current = assert_current
        core = load_selected_core(binary_path, binary_sha256)
        self.engine = core.V6Engine(str(self.root / "data/creck_c1c16_full.yaml"))

    def full_state(self, query):
        q = query
        if query.get("in_product_API") is False:
            q = dict(query, f_JetA=1.0, f_HEFA=0.0, f_FT=0.0, f_ATJ=0.0)
        state = state_for(q, self.public, self.draws)
        config = self.engine.config
        config.mass_flow_core = state["ma"]
        config.bypass_ratio = state["bpr"]
        config.fpr, config.eta_fan = state["fpr"], state["eta_fan"]
        config.pi_c, config.eta_c = state["pi_c"], state["eta_c"]
        config.eta_polytropic = state["eta_t"]
        config.combustor_pressure_loss = state["pressure_loss"]
        config.combustor_heat_loss_fraction = 0.0
        config.combustor_air_fraction = 1.0
        config.A_combustor_exit, config.A_nozzle_exit = .207, .340
        config.T_ambient, config.P_ambient = 288.15, 101325.0
        self.engine.config = config
        fuel, species = self.thermo.fuel_args(query)
        result = self.engine.run_at_thrust(state["target_kN"], fuel, species, state["eta_b"],
                                         .05, 1.0, 3800*5/9, 1e-12, None)
        if result["status"] != "converged":
            return {"status": result["status"], "reason": result.get("reason", ""), "input_state": state}
        return {"status": "converged", "reason": "", "ff_kg_s": result["performance"]["fuel_mass_flow"],
                "T3_K": result["compressor"]["T_out"], "T4_K": result["combustor"]["T_out"],
                "h3_in_J_kg": result["compressor"]["h_in"], "h3_out_J_kg": result["compressor"]["h_out"],
                "h4_out_J_kg": result["combustor"]["h_out"],
                "p3_Pa": result["combustor"]["p_out"], "mcore_kg_s": state["ma"],
                "cp4_J_kg_K": result["combustor"]["cp_out"],
                "R4_J_kg_K": result["combustor"]["R_out"], "gamma4": result["combustor"]["gamma_out"],
                "Y4": result["combustor"]["Y_out"], "phi": result["thrust_match"]["phi"],
                "thrust_residual_kN": result["thrust_match"]["residual_kN"],
                "n_cycle_evaluations": result["thrust_match"]["n_cycle_evaluations"], "input_state": state}

    def __call__(self, queries):
        self.assert_current()
        output = []
        for query in queries:
            result = self.full_state(query)
            row = {key: result[key] for key in ("status", "reason")}
            row["draw_id"] = query["draw_id"]
            row.update({key: query[key] for key in ("query_id", "candidate_id", "design_id", "input_sha256") if key in query})
            if result["status"] == "converged":
                row.update(ff_kg_s=float(result["ff_kg_s"]), T4_K=float(result["T4_K"]),
                           seed_sd_ff_kg_s=0.0, seed_sd_T4_K=0.0, diagnostic_unsafe=False)
                row.update(derived_outputs(query, result["ff_kg_s"], self.properties))
                row["physical_aux"] = {key: float(result[key]) for key in ("cp4_J_kg_K", "R4_J_kg_K", "gamma4")}
            else:
                row.update(ff_kg_s=None, T4_K=None, physical_aux=None, EI_CO2_kg_kg=None,
                           CO2_g_s=None, lifecycle_g_s=None, nvpm_dEI_number_pct=None,
                           nvpm_status="unavailable", nvpm_reason="teacher_unreachable")
            output.append(row)
        self.assert_current()
        return output


class PythonVerifier(Verifier):
    """The protected Python v6 full-equilibrium engine, with full-state outputs."""
    def __init__(self, root, simulator_identity, properties, public, draws, *, assert_current):
        assert_current()
        import contextlib
        import io
        import os
        root = Path(root).resolve()
        for path in (root, root / "scripts/optimization"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import lto_v5 as v5
        if Path(v5.__file__).resolve() != root / "scripts/optimization/lto_v5.py":
            raise ValueError("Python-v6 module is from another checkout")
        with contextlib.redirect_stdout(io.StringIO()):
            v5._init_worker(v5.load_split()["heldout_models"])
        self.root, self.simulator_identity = root, simulator_identity
        self.properties, self.public, self.draws = properties, public, draws
        self.thermo, self.assert_current, self.engine = Thermo(properties), assert_current, v5._ENGINE
        assert_current()

    def full_state(self, query):
        from integrated_engine import LocalFuelBlend, ThrustTargetUnreachable
        q = query if query.get("in_product_API") is not False else dict(
            query, f_JetA=1.0, f_HEFA=0.0, f_FT=0.0, f_ATJ=0.0)
        state = state_for(q, self.public, self.draws)
        self.engine.design_point.update(mass_flow_core=state["ma"], bypass_ratio=state["bpr"],
            fpr=state["fpr"], eta_fan=state["eta_fan"], pi_c=state["pi_c"],
            combustor_pressure_loss=state["pressure_loss"], combustor_heat_loss_fraction=0.0,
            combustor_air_fraction=1.0, A_combustor_exit=.207, A_nozzle_exit=.340,
            T_ambient=288.15, P_ambient=101325.0)
        self.engine.compressor.eta_c = state["eta_c"]
        self.engine.turbine_design["eta_polytropic"] = state["eta_t"]
        fuel_text, _ = self.thermo.fuel_args(query)
        parts = {name.strip(): float(value) for name, value in
                 (component.split(":", 1) for component in fuel_text.split(","))}
        try:
            result = self.engine.run_at_thrust(state["target_kN"], LocalFuelBlend("pc-saf", parts),
                combustor_efficiency=state["eta_b"], phi_bounds=(.05, 1.0),
                t4_max_K=3800*5/9, phi_xtol=1e-12, phi_guess=None)
        except ThrustTargetUnreachable as exc:
            return {"status": "unreachable", "reason": exc.reason, "input_state": state}
        comp, burner, match = result["compressor"], result["combustor"], result["thrust_match"]
        return {"status": "converged", "reason": "", "input_state": state,
            "ff_kg_s": float(result["performance"]["fuel_mass_flow"]), "T3_K": float(comp["T_out"]),
            "T4_K": float(burner["T_out"]), "h3_in_J_kg": float(comp["h_in"]),
            "h3_out_J_kg": float(comp["h_out"]), "h4_out_J_kg": float(burner["h_out"]),
            "p3_Pa": float(burner["p_out"]), "mcore_kg_s": float(state["ma"]),
            "cp4_J_kg_K": float(burner["cp_out"]), "R4_J_kg_K": float(burner["R_out"]),
            "gamma4": float(burner["gamma_out"]), "Y4": burner["Y_out"].tolist(),
            "phi": float(match["phi"]), "thrust_residual_kN": float(match["residual_kN"]),
            "n_cycle_evaluations": int(match["n_cycle_evaluations"])}


def make_verifier(context, properties, public, draws, *, assert_current):
    if getattr(context, "simulator_backend", None) == "python":
        return PythonVerifier(context.root, context.simulator_identity, properties, public, draws,
                              assert_current=assert_current)
    return Verifier(context.root, context.binary_path, context.binary_sha256, properties, public, draws,
                    assert_current=assert_current)


def load_verifier(context, bundle_path=None, *, assert_current=None):
    """Verification is authorized only under the caller's fresh shared run lease."""
    if assert_current is None:
        raise ValueError("A live owned shared Run.assert_current callback is required")
    root = Path(context.root)
    output = Path(bundle_path).resolve().parent if bundle_path is not None else root / "outputs/phase8/saf_surrogate/attempt_001"
    properties = read_json(output / "frozen_properties.json")
    public, draws = read_json(output / "public_inputs.json"), read_json(output / "fixed_draws.json")
    if bundle_path is not None:
        bundle = read_json(bundle_path)
        if getattr(context,"simulator_backend",None)=="python":
            if bundle.get("simulator_identity_sha256") != context.simulator_identity_sha256:
                raise ValueError("Verification Python simulator does not match the product")
        elif bundle["binary_sha256"] != context.binary_sha256:
            raise ValueError("Verification core does not match the product")
    return make_verifier(context, properties, public, draws, assert_current=assert_current)


_worker = None


def _initialize_worker(spec, properties, public, draws):
    """Spawn-only worker; exact parent/spec authorization is checked first."""
    import os
    import time
    from .run import check_child_authorization,birth,native_command
    from .registration import write_once
    worker_handshake=Path(spec["output"])/f"proofs/workers/{spec['stage']}_{os.getpid()}.json"
    write_once(worker_handshake,{"pid":os.getpid(),"birth":birth(os.getpid()),"argv":list(sys.orig_argv),
        "native_command":native_command(os.getpid()),"parent_pid":os.getppid(),
        "spec_sha256":sha256_file(Path(spec["root"])/spec["command_spec_path"])})
    go=Path(spec["output"])/f"proofs/go/worker_{os.getpid()}.json"
    deadline=time.monotonic()+60
    while not go.exists():
        if time.monotonic()>deadline:
            raise RuntimeError("Owner did not publish registered worker GO")
        time.sleep(.05)
    if read_json(go)["spec_sha256"] != sha256_file(Path(spec["root"])/spec["command_spec_path"]):
        raise RuntimeError("Worker GO differs from exact command spec")
    check = lambda: check_child_authorization(spec)
    check()
    global _worker
    if spec.get("simulator", {}).get("name") == "python-v6":
        _worker = PythonVerifier(spec["root"], spec["simulator"], properties, public, draws, assert_current=check)
    else:
        _worker = Verifier(spec["root"], Path(spec["root"]) / spec["binary"]["path"],
                           spec["binary"]["sha256"], properties, public, draws, assert_current=check)
    check()
    actual=Path(sys.modules["integrated_engine" if spec.get("simulator") else "catjet_core"].__file__).resolve()
    write_once(Path(spec["output"])/f"proofs/workers/{spec['stage']}_{os.getpid()}_core.json",{
        "pid":os.getpid(),"birth":birth(os.getpid()),"parent_pid":os.getppid(),
        **({"actual_simulator": spec["simulator"], "actual_python_module": {
            "path":str(actual.relative_to(Path(spec["root"]))),"sha256":sha256_file(actual)}} if spec.get("simulator") else {
            "actual_core":{"path":str(actual.relative_to(Path(spec["root"]))),"sha256":sha256_file(actual)}}),
        "input_source_hashes":spec["input_source_hashes"],
        "spec_sha256":sha256_file(Path(spec["root"])/spec["command_spec_path"]),
        "handshake_sha256":sha256_file(worker_handshake),"go_sha256":sha256_file(go)})


def _full_chunk(queries):
    _worker.assert_current()
    rows = [_worker.full_state(query) for query in queries]
    _worker.assert_current()
    return rows


class ParallelTeacher:
    """Identical, ordered full-state requests with one thread per spawned worker."""
    def __init__(self, spec, properties, public, draws, workers, *, run=None):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        import os
        from .registration import write_once
        import time
        from .run import birth,native_command
        folder=Path(spec["output"])/"proofs/spawns"
        self.receipts={}
        receipts=self.receipts
        spec_path=Path(spec["root"])/spec["command_spec_path"]
        class TrackedPool(ProcessPoolExecutor):
            serial=0
            def _spawn_process(self):
                slot=f"{spec['stage']}_pool_spawn_{self.serial}"; self.serial+=1
                write_once(folder/f"{slot}.starting.json",{"state":"STARTING","parent_pid":os.getpid(),"spec_sha256":sha256_file(spec_path)})
                before=set(self._processes)
                super()._spawn_process()
                added=set(self._processes)-before
                if len(added)!=1:
                    raise RuntimeError("Ambiguous pool spawn")
                pid=added.pop()
                worker_birth=birth(pid)
                handshake_path=Path(spec["output"])/f"proofs/workers/{spec['stage']}_{pid}.json"
                deadline=time.monotonic()+60
                while not handshake_path.exists():
                    if time.monotonic()>deadline:raise RuntimeError("Spawned worker handshake absent")
                    time.sleep(.05)
                handshake=read_json(handshake_path)
                if handshake["pid"]!=pid or handshake["birth"]!=worker_birth or handshake["parent_pid"]!=os.getpid() or handshake["spec_sha256"]!=sha256_file(spec_path) or handshake["native_command"]!=native_command(pid):
                    raise RuntimeError("Spawned worker handshake differs from native command/spec")
                receipt={"state":"CREATED","parent_pid":os.getpid(),"pid":pid,"birth":worker_birth,
                    "argv":handshake["argv"],"native_command":handshake["native_command"],"logical_slot":slot,
                    "handshake_path":str(handshake_path.relative_to(Path(spec["root"]))),
                    "handshake_sha256":sha256_file(handshake_path),"spec_sha256":sha256_file(spec_path)}
                receipt_path=folder/f"{slot}.complete.json"; write_once(receipt_path,receipt)
                receipts[pid]=receipt
                if run is not None:
                    run._saf_children.append({key:receipt[key] for key in ("pid","birth","argv")})
                    run.record_children(run._saf_children); run.assert_current()
                    write_once(Path(spec["output"])/f"proofs/go/worker_{pid}.json",{"receipt_sha256":sha256_file(receipt_path),"spec_sha256":sha256_file(spec_path)})
        self.spec, self.properties = spec, properties
        self.pool = TrackedPool(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker, initargs=(spec, properties, public, draws))

    def full_states(self, queries):
        chunks = [queries[start:start+16] for start in range(0, len(queries), 16)]
        return [row for chunk in self.pool.map(_full_chunk, chunks) for row in chunk]

    def __call__(self, queries):
        rows = []
        for query, state in zip(queries, self.full_states(queries)):
            row = {key: state[key] for key in ("status", "reason")}
            row.update({key: query[key] for key in ("draw_id", "design_id", "candidate_id", "query_id", "input_sha256") if key in query})
            if state["status"] == "converged":
                row.update(ff_kg_s=float(state["ff_kg_s"]), T4_K=float(state["T4_K"]),
                           seed_sd_ff_kg_s=0.0, seed_sd_T4_K=0.0, diagnostic_unsafe=False)
                row.update(derived_outputs(query, state["ff_kg_s"], self.properties))
            else:
                row.update(ff_kg_s=None, T4_K=None, CO2_g_s=None, EI_CO2_kg_kg=None, lifecycle_g_s=None,
                    nvpm_dEI_number_pct=None, nvpm_status="unavailable", nvpm_reason="teacher_unreachable")
            rows.append(row)
        return rows

    def close(self):
        from .registration import write_once
        processes=dict(self.pool._processes or {})
        self.pool.shutdown(wait=True, cancel_futures=True)
        for pid,process in processes.items():
            receipt=self.receipts[pid]
            write_once(Path(self.spec["output"])/f"proofs/workers/{self.spec['stage']}_{pid}_exit.json",{
                "pid":pid,"birth":receipt["birth"],"argv":receipt["argv"],"parent_pid":receipt["parent_pid"],
                "waited":True,"exit_code":process.exitcode,"spec_sha256":receipt["spec_sha256"]})
        failed={pid:process.exitcode for pid,process in processes.items() if process.exitcode!=0}
        if failed:
            raise RuntimeError(f"Simulator worker shutdown failed: {failed}")
