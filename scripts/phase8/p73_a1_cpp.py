#!/usr/bin/env python3
"""P7.3-A1 conditional blends; lazy scientific imports after the shared gate.

This additive consumer preserves the protected P7.3 quantitative schemas and
algorithm. It requires the separate original-main attestation/source-extension
gate and verified fresh G0 provenance; it never edits the original closed gate.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import numbers
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = "docs/phase7_p73_a1_registration.json"
OWN_PATHS = ("scripts/phase8/p73_a1_cpp.py", "tests/test_phase7_p73_a1_cpp.py",
             REGISTRATION, "docs/phase7_p73_a1_registration.md")
LABEL = ("Conditional on the frozen v6 calibration; A1 FAIL (penalty-dependent); "
         "original gate closed; P7.3-A1 exception.")
TABLE_NAMES = ("p73_blends_v6_central.csv", "p73_blends_v6_draws.csv",
               "p73_blends_v6_claims.csv", "p73_blends_v6_nvpm.csv",
               "p73_blends_v6_lifecycle_corsia.csv")
EXPECTED_OUTPUTS = (*TABLE_NAMES, "p73_blends_v6.json", "README.md",
                    "environment.json", "command.log", "command.exit.json",
                    "partial_rows.jsonl", "parity/parity.json")


class Blocked(RuntimeError):
    """A required prospective dependency or identity is not established."""


class ParityFailure(RuntimeError):
    """The registered cross-backend comparison did not pass."""


class ChildLifecycleError(RuntimeError):
    """A pool shutdown could not prove that every owned worker ended."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise Blocked(f"expected JSON object: {path}")
    return value


def write_new(path: Path, value: str) -> None:
    # Every caller supplies a path inside the exclusively reserved run directory.
    with Path(path).open("x") as stream:
        stream.write(value)


def json_text(value) -> str:
    return json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"


def consumer_identity(root: Path, registration: Path) -> dict:
    """Freeze committed additive files separately from original-main identity."""
    root = Path(root).resolve()
    registration = Path(registration).resolve()
    if registration != root / REGISTRATION:
        raise Blocked("registration must be the committed P7.3-A1 record in the consumer tree")
    status = subprocess.run(["git", "status", "--porcelain", "--", *OWN_PATHS],
                            cwd=root, text=True, capture_output=True, check=True).stdout
    if status.strip():
        raise Blocked("additive consumer/registration files are uncommitted or dirty")
    tree = subprocess.run(["git", "ls-tree", "HEAD", "--", *OWN_PATHS], cwd=root,
                          text=True, capture_output=True, check=True).stdout
    modes = {}
    for line in tree.splitlines():
        meta, rel = line.split("\t", 1)
        mode, kind, _blob = meta.split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise Blocked(f"unsupported additive file mode: {rel}")
        modes[rel] = mode
    if set(modes) != set(OWN_PATHS):
        raise Blocked("required additive files are missing from the committed tree")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                          capture_output=True, check=True).stdout.strip()
    files = {}
    for rel in OWN_PATHS:
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise Blocked(f"additive file is absent or a symlink: {rel}")
        disk_mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if disk_mode != modes[rel]:
            raise Blocked(f"additive file mode drift: {rel}")
        files[rel] = {"sha256": sha256(path), "mode": modes[rel]}
    return {"git_head": head, "consumer_root": str(root), "files": files,
            "registration_sha256": sha256(registration)}


def validate_contract(root: Path, registration: dict) -> None:
    if registration.get("registration_id") != "P7.3-A1":
        raise Blocked("not the registered P7.3-A1 consumer contract")
    parent = read_json(root / registration["parent"]["path"])
    inherited = registration["unchanged_protocol"]
    if inherited != {key: parent[key] for key in inherited}:
        raise Blocked("scientific protocol differs from the protected parent")
    for rel, expected in registration["inputs_sha256"].items():
        if sha256(root / rel) != expected:
            raise Blocked(f"registered input changed: {rel}")
    # These define the reused algorithm; original main operation records are
    # separately proven by the shared gate against their original identity.
    for rel in ("scripts/optimization/blend_matched_thrust_v6.py",
                "scripts/phase8/v6_backend.py", "simulation/catjet_backend.py"):
        if sha256(root / rel) != registration["base_source_and_registration_sha256"][rel]:
            raise Blocked(f"registered protected algorithm/adapter changed: {rel}")
    parity = registration["backend_parity"]["study_parity"]
    if parity["fixed_draws"] != ["draw_00", "draw_31", "draw_63"]:
        raise Blocked("unregistered backend parity coverage")
    if registration["implementation_contract"]["task_contract"]["workers"] != 6:
        raise Blocked("worker count differs from registration")


def validate_exception(profile: dict, fit: dict, hold: dict, historical: dict) -> None:
    """Only the recorded penalty-guard-only failure has additive authorization."""
    free = fit.get("free")
    if not isinstance(free, list) or not free or len(set(free)) != len(free):
        raise Blocked("frozen fit has no valid fitted parameter set")
    if profile.get("free") != free or set(profile.get("verdicts", {})) != set(free):
        raise Blocked("profile does not cover the frozen fitted parameter set")
    if profile.get("A1") != "FAIL" or profile.get("identified") != []:
        raise Blocked("the registered penalty-dependent A1 failure is not established")
    if set(profile.get("not_identified", [])) != set(free):
        raise Blocked("profile failed-parameter set differs from the frozen fit")
    for name in free:
        verdict = profile["verdicts"][name]
        if not (verdict.get("IDENTIFIED_existing_rule") is True
                and verdict.get("IDENTIFIED") is False
                and verdict.get("penalty_dependent") is True
                and isinstance(verdict.get("verdict_points_penalised"), list)
                and bool(verdict["verdict_points_penalised"])):
            raise Blocked(f"A1 failure is not solely the registered penalty guard: {name}")
    a2 = hold.get("A2", {}).get("verdict")
    if not isinstance(a2, str) or not a2 or a2.startswith("ESCALATE"):
        raise Blocked("A2 missing or B0 escalation triggered")
    for gate in (hold.get("blend_gate"), historical.get("gate")):
        if not isinstance(gate, dict) or gate.get("open") is not False or gate.get("reason") != "A1 FAIL":
            raise Blocked("historical A1 closed gate is absent or changed")


def load_module(name: str, path: Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Blocked(f"cannot load registered module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise Blocked(f"module origin mismatch: {name}")
    return module


def load_selected_core(path: Path, expected: str):
    """Explicit selection, also used in every spawned C++ worker."""
    path = Path(path).resolve()
    if sha256(path) != expected:
        raise Blocked("selected core hash changed")
    loaded = sys.modules.get("catjet_core")
    if loaded is not None and Path(loaded.__file__).resolve() != path:
        raise Blocked("another core is already imported; no fallback permitted")
    core = loaded if loaded is not None else load_module("catjet_core", path)
    if Path(core.__file__).resolve() != path or sha256(path) != expected:
        raise Blocked("actual imported core identity differs from the selected core")
    if not hasattr(core, "V6Engine"):
        raise Blocked("selected core lacks full-equilibrium V6Engine")
    return core


def import_protocol(root: Path):
    root = Path(root).resolve()
    sys.path[:0] = [str(root), str(root / "scripts" / "optimization"),
                    str(root / "scripts" / "phase8")]
    protocol = load_module("_p73_protected_protocol", root / "scripts/optimization/blend_matched_thrust_v6.py")
    for module, rel in ((protocol.v5, "scripts/optimization/lto_v5.py"),
                        (protocol.v6, "scripts/optimization/lto_v6.py"),
                        (protocol.fuels_v7, "simulation/fuels_v7.py")):
        if Path(module.__file__).resolve() != root / rel:
            raise Blocked(f"protected module imported from another checkout: {rel}")
    backend = load_module("_p73_registered_backend", root / "scripts/phase8/v6_backend.py")
    return protocol, backend


def worker_record(workers_dir, pool_tag, backend, binary=None, expected_hash=None):
    birth = subprocess.run(["ps", "-p", str(os.getpid()), "-o", "lstart="],
                           capture_output=True, text=True, check=True).stdout
    birth = " ".join(birth.split())
    command = subprocess.run(["ps", "-ww", "-p", str(os.getpid()), "-o", "command="],
                             capture_output=True, text=True, check=True).stdout.strip()
    if not birth or not command:
        raise Blocked("worker birth or command identity unreadable")
    path = Path(workers_dir) / f"{pool_tag}-{os.getpid()}.json"
    write_new(path, json_text({"pid": os.getpid(), "birth": birth, "pool": pool_tag,
                              "argv": [command], "argv_format": "ps command (single string)",
                              "binary_path": None if binary is None else str(Path(binary).resolve()),
                              "binary_sha256": expected_hash, "backend": backend}))


def init_cpp_worker(root, binary, expected_hash, excluded, workers_dir, pool_tag):
    root = Path(root).resolve()
    _protocol, backend = import_protocol(root)
    core = load_selected_core(Path(binary), expected_hash)
    backend.init_worker_cpp(excluded)
    adapter = importlib.import_module("simulation.catjet_backend")
    if Path(adapter.__file__).resolve() != root / "simulation/catjet_backend.py":
        raise Blocked("worker imported another checkout's adapter")
    if adapter.load_core() is not core:
        raise Blocked("worker adapter did not use the selected full-equilibrium core")
    worker_record(workers_dir, pool_tag, "cpp-full-equilibrium-v6", binary, expected_hash)


def init_python_worker(root, excluded, workers_dir, pool_tag):
    protocol, _backend = import_protocol(Path(root))
    protocol.v5._init_worker(excluded)
    worker_record(workers_dir, pool_tag, "python-v6")


def make_model(protocol, backend, reg6, split, kind, context, output_dir, tag):
    model = backend.V6Model(reg6["fixed_central"], split["heldout_models"], n_workers=6,
                            fuel=protocol.v6.fuel_composition(reg6), backend=kind)
    # Keep the registered V6Model and solve_task path; initializers add only
    # explicit module provenance and worker evidence around protected setup.
    model.pool.shutdown(wait=True, cancel_futures=True)
    if kind == "cpp":
        model.pool = ProcessPoolExecutor(
            max_workers=6, initializer=init_cpp_worker,
            initargs=(str(protocol.ROOT), str(context.binary_path), context.binary_sha256,
                      split["heldout_models"], str(output_dir / "workers"), tag))
    else:
        model.pool = ProcessPoolExecutor(
            max_workers=6, initializer=init_python_worker,
            initargs=(str(protocol.ROOT), split["heldout_models"], str(output_dir / "workers"), "parity_python"))
    return model


def sync_worker_children(run, out):
    children = []
    for path in sorted((out / "workers").glob("*.json")):
        proof = read_json(path)
        if (not isinstance(proof.get("pid"), int) or proof["pid"] <= 0
                or not isinstance(proof.get("birth"), str) or not proof["birth"]
                or not isinstance(proof.get("argv"), list) or not proof["argv"]):
            raise Blocked("worker ownership proof is incomplete")
        children.append({"pid": proof["pid"], "birth": proof["birth"], "argv": proof["argv"]})
    run.record_children(children)


def close_model(model, run, out):
    try:
        model.close()
    except BaseException as exc:
        try:
            sync_worker_children(run, out)
        except BaseException:
            pass
        raise ChildLifecycleError("worker pool shutdown unproven; preserve owner lease") from exc
    sync_worker_children(run, out)


def missing(value) -> bool:
    return value is None or (isinstance(value, numbers.Real) and math.isnan(float(value)))


def compare_records(a: list[dict], b: list[dict]) -> dict:
    problems = []
    if len(a) != len(b):
        return {"match": False, "problems": ["row count differs"]}
    for row, (left, right) in enumerate(zip(a, b)):
        if list(left) != list(right):
            problems.append(f"row {row}: columns/order differ")
            continue
        if left.get("status") not in {"converged", "unreachable"} or right.get("status") not in {"converged", "unreachable"}:
            problems.append(f"row {row}: unknown solve status")
        for key, x in left.items():
            y = right[key]
            if key in {"status", "reason", "fuel", "op"}:
                ok = isinstance(x, str) and isinstance(y, str) and x == y
            elif missing(x) or missing(y):
                ok = missing(x) and missing(y) and left.get("status") == right.get("status") == "unreachable"
            elif isinstance(x, numbers.Real) and isinstance(y, numbers.Real) and not isinstance(x, bool) and not isinstance(y, bool):
                ok = math.isfinite(float(x)) and math.isfinite(float(y)) and math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-12)
            else:
                ok = type(x) is type(y) and x == y
            if not ok:
                problems.append(f"row {row}, column {key}: registered parity mismatch")
    return {"match": not problems, "rows": len(a), "problems": problems}


def write_frame(path: Path, frame) -> None:
    with path.open("x", newline="") as stream:
        frame.to_csv(stream, index=False)


def validate_coverage(protocol, central, draws, fuels) -> None:
    modes = list(protocol.OPERATING_POINTS)
    expected_c = [(fuel, mode) for fuel in fuels for mode in modes]
    got_c = list(central[["fuel", "op"]].itertuples(index=False, name=None))
    study = [fuel for fuel in fuels if fuel != protocol.JETA_ALT]
    expected_d = [(f"draw_{i:02d}", fuel, mode) for i in range(protocol.N_DRAWS)
                  for fuel in study for mode in modes]
    got_d = list(draws[["draw", "fuel", "op"]].itertuples(index=False, name=None))
    if got_c != expected_c or got_d != expected_d:
        raise RuntimeError("registered central/draw/fuel/mode rows missing, duplicated or reordered")
    required = [quantity for quantity in protocol.QUANTITIES if quantity != "lifecycle_g_s"]
    for label, frame in (("central", central), ("draws", draws)):
        if not frame["status"].isin(["converged", "unreachable"]).all():
            raise RuntimeError(f"{label}: unknown solve status")
        converged = frame["status"] == "converged"
        for column in required:
            if converged.any() and (column not in frame or not frame.loc[converged, column].map(
                    lambda value: isinstance(value, numbers.Real) and math.isfinite(float(value))).all()):
                raise RuntimeError(f"{label}: nonfinite or missing converged solve quantity {column}")


def parity_stage(protocol, backend, reg6, split, params, ae3, fuels, fixed_draws, context, run, out, checkpoint=None):
    directory = out / "parity"
    directory.mkdir()
    checks = {}
    results = {}
    study = {key: value for key, value in fuels.items() if key != protocol.JETA_ALT}
    for kind in ("python", "cpp"):
        run.assert_current()
        model = make_model(protocol, backend, reg6, split, kind, context, out, "parity_cpp")
        try:
            batches = {"central": protocol.run(model, params, reg6["fixed_central"], ae3, fuels)}
            sync_worker_children(run, out)
            if checkpoint is not None:
                checkpoint(f"parity-{kind}", "central", batches["central"])
            for case, row in fixed_draws:
                if case not in {"draw_00", "draw_31", "draw_63"}:
                    continue
                run.assert_current()
                batches[case] = protocol.run(model, params, protocol.draw_fixed(row, reg6["fixed_central"]), ae3, study)
                sync_worker_children(run, out)
                if checkpoint is not None:
                    checkpoint(f"parity-{kind}", case, batches[case])
            for name, frame in batches.items():
                write_frame(directory / f"{name}_{kind}.csv", frame)
            results[kind] = batches
        finally:
            close_model(model, run, out)
    if list(results["python"]) != ["central", "draw_00", "draw_31", "draw_63"]:
        raise RuntimeError("registered parity draw coverage missing or reordered")
    for name in results["python"]:
        left, right = results["python"][name], results["cpp"][name]
        if list(left.columns) != list(right.columns):
            checks[name] = {"match": False, "problems": ["columns/order differ"]}
        else:
            checks[name] = compare_records(left.to_dict("records"), right.to_dict("records"))
    record = {"status": "PASS" if all(c["match"] for c in checks.values()) else "FAIL",
              "checks": checks, "rule": "math.isclose(rel_tol=1e-9, abs_tol=1e-12); missing patterns and strings exact",
              "binary_path": str(context.binary_path), "binary_sha256": context.binary_sha256,
              "coverage": "All central fuels/four modes; draw_00, draw_31, draw_63 for all study fuels/four modes. Other draw parity untested.",
              "conditional_label": LABEL}
    run.assert_current()
    write_new(directory / "parity.json", json_text(record))
    if record["status"] != "PASS":
        raise ParityFailure("registered backend parity failed; full blend stage blocked")
    return record


def postprocess(protocol, registration, params, cen, draws, fuels):
    """Same registered main postprocessing, calling protected scientific rules."""
    np, pd = protocol.np, protocol.pd
    c = protocol.corsia()
    lc_c = protocol.corsia_central(c)
    for df in (cen, draws):
        df["saf_mass_fraction"] = [protocol.saf_fraction(f) for f in df["fuel"]]
        df["lifecycle_factor_gCO2e_per_kg"] = [protocol.lifecycle_factor(fuels[f], lc_c) for f in df["fuel"]]
        df["lifecycle_g_s"] = df["ff"] * df["lifecycle_factor_gCO2e_per_kg"]
        df["in_calibration_domain"] = [protocol.OPERATING_POINTS[o][2] for o in df["op"]]
    cen["lhv_liquid_MJ_kg"] = [sum(w * protocol.lhv_liquid(s) for s, w in fuels[f].items()) for f in cen["fuel"]]
    cen["lhv_gas_MJ_kg"] = [protocol.fuels_v7.properties(protocol.fuels_v7.mass_blend(fuels[f]))["lhv_gas_MJ_kg"] for f in cen["fuel"]]
    inherited = registration["unchanged_protocol"]
    cd = protocol.corsia_common_draws(c, inherited["lifecycle"]["n_corsia_common"], inherited["lifecycle"]["seed"])
    C, D = cen.set_index(["fuel", "op"]), draws.set_index(["draw", "fuel", "op"])
    rows, lc_rows = [], []
    for pair in protocol.pairs():
        a, b, family = pair["a"], pair["b"], pair["family"]
        for op, (_x, _eta, domain) in protocol.OPERATING_POINTS.items():
            ca, cb = C.loc[(a, op)], C.loc[(b, op)]
            converged = bool((draws.loc[(draws["fuel"].isin([a, b])) & (draws["op"] == op), "status"] == "converged").all()) and ca["status"] == cb["status"] == "converged"
            for quantity in protocol.QUANTITIES:
                delta = float(ca[quantity] - cb[quantity])
                pa = D.xs((a, op), level=("fuel", "op"))[quantity]
                pb = D.xs((b, op), level=("fuel", "op"))[quantity]
                paired = (pa - pb).to_numpy(dtype=float)
                spread = abs(float(C.loc[(protocol.JETA, op), quantity] - C.loc[(protocol.JETA_ALT, op), quantity]))
                extra = None
                if quantity == "lifecycle_g_s":
                    la = np.array([protocol.lifecycle_factor(fuels[a], item) for item in cd]) * ca["ff"]
                    lb = np.array([protocol.lifecycle_factor(fuels[b], item) for item in cd]) * cb["ff"]
                    dd = la - lb
                    extra = float(np.mean(np.sign(dd) == np.sign(delta)))
                    lc_rows.append({"a": a, "b": b, "family": family, "op": op, "delta_central": delta,
                                    "corsia_p5": float(np.percentile(dd, 5)), "corsia_p95": float(np.percentile(dd, 95)),
                                    "corsia_sign_agreement": extra})
                ok, why = protocol.claim(delta, paired, spread, domain, converged, extra)
                rows.append({"family": family, "a": a, "b": b, "op": op, "quantity": quantity,
                             "central_a": float(ca[quantity]), "central_b": float(cb[quantity]), "delta_central": delta,
                             "delta_rel_pct_of_b": 100.0 * delta / float(cb[quantity]) if cb[quantity] else np.nan,
                             "spread_S_dooley2012_vs_2010": spread,
                             "paired_sign_agreement": float(np.mean(np.sign(paired) == np.sign(delta))),
                             "paired_n": int(len(paired)), "paired_delta_p5": float(np.percentile(paired, 5)),
                             "paired_delta_p95": float(np.percentile(paired, 95)), "paired_delta_min": float(paired.min()),
                             "paired_delta_max": float(paired.max()), "corsia_sign_agreement": extra,
                             "in_domain": domain, "all_converged": converged, "claimed": ok,
                             "rejection_reasons": "; ".join(why)})
    claims = pd.DataFrame(rows)
    nv = []
    for fuel in fuels:
        if fuel == protocol.JETA_ALT:
            continue
        for op, (x, _eta, _domain) in protocol.OPERATING_POINTS.items():
            result = protocol.nvpm_brem(fuel, 100.0 * x, inherited["nvpm"])
            nv.append({"fuel": fuel, "op": op, "thrust_pct": 100.0 * x, **result,
                       "label": ("screening at an extrapolated engine condition (85 % climb)" if op == "CLIMB85"
                                 else "screening (empirical relation), not validated engine nvPM")
                       if result["status"] == "screening" else "unavailable (outside the registered validity)"})
    nvpm = pd.DataFrame(nv)
    summary = {"registration": REGISTRATION, "parent_registration": "outputs/phase7/p73_registration.json",
               "calibration": "outputs/phase7/calibration_v6.json", "v6_params_fixed_for_all_fuels_and_draws": params,
               "n_draws": protocol.N_DRAWS, "n_unconverged_central": int((cen["status"] != "converged").sum()),
               "n_unconverged_draws": int((draws["status"] != "converged").sum()),
               "spread_S": {op: {q: abs(float(C.loc[(protocol.JETA, op), q] - C.loc[(protocol.JETA_ALT, op), q])) for q in protocol.QUANTITIES} for op in protocol.OPERATING_POINTS},
               "claims": claims[claims["claimed"]][["family", "a", "b", "op", "quantity", "delta_rel_pct_of_b"]].to_dict("records"),
               "n_comparisons": int(len(claims)), "n_claimed": int(claims["claimed"].sum()),
               "n_claimed_by_family": claims.groupby("family")["claimed"].sum().astype(int).to_dict(),
               "n_in_domain_comparisons": int(claims["in_domain"].sum()), "labels": inherited["labels"],
               "conditional_label": LABEL, "original_A1": "FAIL", "original_blend_gate_open": False,
               "backend": "cpp-full-equilibrium-v6"}
    return (cen, draws, claims, nvpm, pd.DataFrame(lc_rows)), summary


def study_stage(protocol, backend, registration, reg6, split, fit, ae3, fuels, fixed_draws, context, run, out, checkpoint=None):
    run.assert_current()
    model = make_model(protocol, backend, reg6, split, "cpp", context, out, "study_cpp")
    try:
        central = protocol.run(model, fit["params"], reg6["fixed_central"], ae3, fuels)
        sync_worker_children(run, out)
        if checkpoint is not None:
            checkpoint("study-cpp", "central", central)
        frames = []
        study = {name: value for name, value in fuels.items() if name != protocol.JETA_ALT}
        for case, row in fixed_draws:
            run.assert_current()
            frame = protocol.run(model, fit["params"], protocol.draw_fixed(row, reg6["fixed_central"]), ae3, study)
            sync_worker_children(run, out)
            frame.insert(0, "draw", case)
            if checkpoint is not None:
                checkpoint("study-cpp", case, frame)
            frames.append(frame)
            print(f"{case}: {int((frame['status'] != 'converged').sum())} unconverged", flush=True)
        draws = protocol.pd.concat(frames, ignore_index=True)
    finally:
        close_model(model, run, out)
    validate_coverage(protocol, central, draws, fuels)
    run.assert_current()
    tables, summary = postprocess(protocol, registration, fit["params"], central, draws, fuels)
    for name, table in zip(TABLE_NAMES, tables):
        write_frame(out / name, table)
    write_new(out / "p73_blends_v6.json", protocol.v5._json(summary))
    write_new(out / "README.md", technical_readme())
    # Keep the freshly written proof in memory until terminal validation; no
    # self-referential JSON hash and no trust in an independently edited table.
    summary["_sealed_outputs"] = {name: sha256(out / name)
                                  for name in (*TABLE_NAMES, "p73_blends_v6.json", "README.md")}
    summary["_sealed_schemas"] = {name: {"columns": list(table.columns), "rows": int(len(table))}
                                  for name, table in zip(TABLE_NAMES, tables)}
    return summary


def technical_readme() -> str:
    return ("# P7.3-A1 quantitative artifact contract\n\n" + LABEL + "\n\n"
            "The original gate/profile records are unchanged. This separately registered exception retains the original conditional comparison rule; it does not demonstrate identifiable calibration or real-world blend validity.\n\n"
            "CSV schemas follow protected P7.3: central rows, exactly 64 paired fixed draws, every comparison/rejection reason, Brem screening and common CORSIA scenarios. Neat context remains separate; CLIMB85 is extrapolated and never claimed. NOx has no fuel-composition term. Mass fractions are not volume certification limits; no operational approval is implied.\n\n"
            "Lifecycle uses the registered liquid-LHV correction once and 1000 common seed-42 CORSIA draws. Unavailable nvPM cells contain no number; Brem screening is not validated engine nvPM. Sensitivity bands are conditional fixed-calibration sensitivity, not confidence intervals.\n\n"
            "parity/parity.json records all central fuel/mode checks and draw_00/draw_31/draw_63 cross-backend checks at the original tolerance. Other draws have not been cross-backend recomputed by that subset. partial_rows.jsonl retains each finished solve batch as append-only failure evidence without extra simulator requests. environment.json, artifact_hashes.json and terminal.json describe original-main/additive identity, the selected actual binary, verified fresh G0 evidence, ownership and command outcome. This is a technical README, not a narrative results report.\n")


def checkpoint_record(stage, case, frame, identity) -> str:
    def encode(value):
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
            return {"nonfinite_raw_value": str(float(value))}
        return value.item() if hasattr(value, "item") else value
    return json.dumps({"stage": stage, "case": case, "identity": identity,
                       "rows": encode(frame.to_dict("records"))}, default=str, allow_nan=False) + "\n"


def require_outputs(out: Path, *, before_exit_record=False, binary_path=None, binary_sha256=None,
                    expected_hashes=None, expected_schemas=None) -> None:
    absent = [rel for rel in EXPECTED_OUTPUTS
              if not (before_exit_record and rel == "command.exit.json") and not (out / rel).is_file()]
    if absent:
        raise RuntimeError("missing expected outputs: " + ", ".join(absent))
    if expected_hashes is not None:
        expected_names = {*TABLE_NAMES, "p73_blends_v6.json", "README.md"}
        if set(expected_hashes) != expected_names or set(expected_schemas or {}) != set(TABLE_NAMES):
            raise RuntimeError("fresh quantitative artifact proof is incomplete")
        for rel, expected in expected_hashes.items():
            if sha256(out / rel) != expected:
                raise RuntimeError(f"fresh quantitative output drift: {rel}")
        for rel, spec in expected_schemas.items():
            with (out / rel).open(newline="") as stream:
                reader = csv.reader(stream)
                columns = next(reader, None)
                rows = 0
                for row in reader:
                    if len(row) != len(spec["columns"]):
                        raise RuntimeError(f"malformed quantitative CSV: {rel}")
                    rows += 1
            if columns != spec["columns"] or rows != spec["rows"]:
                raise RuntimeError(f"quantitative CSV schema/coverage drift: {rel}")
    verdict = read_json(out / "parity/parity.json")
    if verdict.get("status") != "PASS" or list(verdict.get("checks", {})) != ["central", "draw_00", "draw_31", "draw_63"] or any(c.get("match") is not True for c in verdict["checks"].values()):
        raise RuntimeError("missing or failed parity evidence")
    summary = read_json(out / "p73_blends_v6.json")
    if summary.get("n_draws") != 64 or summary.get("conditional_label") != LABEL or summary.get("original_blend_gate_open") is not False:
        raise RuntimeError("quantitative summary lacks original draw count/conditional gate disclosure")
    if binary_path is not None:
        if verdict.get("binary_path") != str(binary_path) or verdict.get("binary_sha256") != binary_sha256:
            raise RuntimeError("parity proof does not match selected core")
        workers = [read_json(p) for p in sorted((out / "workers").glob("*.json"))]
        if not workers or {record.get("pool") for record in workers} != {"parity_python", "parity_cpp", "study_cpp"}:
            raise RuntimeError("missing C++ worker provenance for parity or full study")
        for record in workers:
            cpp = record.get("pool") != "parity_python"
            if ((cpp and (record.get("binary_path") != str(Path(binary_path).resolve())
                          or record.get("binary_sha256") != binary_sha256
                          or record.get("backend") != "cpp-full-equilibrium-v6"))
                    or (not cpp and record.get("backend") != "python-v6")
                    or not isinstance(record.get("pid"), int) or record["pid"] <= 0
                    or not isinstance(record.get("birth"), str) or not record["birth"]
                    or not isinstance(record.get("argv"), list) or not record["argv"]):
                raise RuntimeError("invalid selected-core worker provenance")


def environment(context, identity) -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "cantera", "pybind11"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "versions": versions, "argv": list(sys.argv), "priority": os.getpriority(os.PRIO_PROCESS, 0),
            "original_main_context": context.original_context, "identity": context.identity,
            "consumer_identity": identity, "binary_path": str(context.binary_path),
            "binary_sha256": context.binary_sha256, "conditional_label": LABEL}


def execute(main_root: Path, registration_path: Path, *, consumer_root: Path = ROOT, gate_factory=None,
            scientific_runner=None) -> dict:
    main_root, registration_path = Path(main_root).resolve(), Path(registration_path).resolve()
    registration = read_json(registration_path)
    identity = consumer_identity(consumer_root, registration_path)
    if gate_factory is None:
        sys.path.insert(0, str(main_root))
        gate = importlib.import_module("scripts.phase8.scientific_workflow_gate")
        if Path(gate.__file__).resolve() != main_root / "scripts/phase8/scientific_workflow_gate.py":
            raise Blocked("shared gate imported from an unvalidated checkout")
        gate_factory = gate.prepare_context
    context = gate_factory(main_root, REGISTRATION, expected_consumer_identity=identity, require_g0=True)
    context.require_idle_ac()
    validate_contract(main_root, registration)
    fit = read_json(main_root / registration["implementation_contract"]["fit_path"])
    profile = read_json(main_root / registration["implementation_contract"]["profile_path"])
    hold = read_json(main_root / registration["implementation_contract"]["gate_path"])
    historical = read_json(main_root / registration["implementation_contract"]["historical_closed_gate_path"])
    validate_exception(profile, fit, hold, historical)
    out = main_root / registration["outputs"]["directory"]
    run = context.acquire_run(out, identity["registration_sha256"], identity=context.identity)
    reservation = read_json(out / "reservation.json")
    owner = {key: reservation[key] for key in ("owner_pid", "owner_birth", "argv")}
    status, errors, summary = "ERROR", [], None
    ambiguous_child = False
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        run.assert_current()
        write_new(out / "environment.json", json_text(environment(context, identity)))
        (out / "workers").mkdir()
        with (out / "command.log").open("x") as log, (out / "partial_rows.jsonl").open("x") as journal, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            def checkpoint(stage, case, frame):
                journal.write(checkpoint_record(stage, case, frame, context.identity))
                journal.flush()
            if scientific_runner is not None:
                summary = scientific_runner(main_root, registration, fit, context, run, out)
            else:
                load_selected_core(context.binary_path, context.binary_sha256)
                protocol, backend = import_protocol(main_root)
                reg6, split = protocol.v6.load_registration_v6(), protocol.v5.load_split()
                ae3 = protocol.v5.load_rows([protocol.AE3_UID], with_targets=False).iloc[0]
                fuels, fixed_draws = protocol.fuel_parts(), protocol.load_draws()
                if [name for name, _ in fixed_draws] != [f"draw_{i:02d}" for i in range(64)]:
                    raise RuntimeError("original fixed-draw coverage missing or reordered")
                parity_stage(protocol, backend, reg6, split, fit["params"], ae3, fuels, fixed_draws, context, run, out, checkpoint)
                run.assert_current()
                summary = study_stage(protocol, backend, registration, reg6, split, fit, ae3, fuels, fixed_draws, context, run, out, checkpoint)
        run.assert_current()
        status = "COMPLETE"
    except BaseException as exc:
        status = "FAIL" if isinstance(exc, ParityFailure) else "ERROR"
        errors.append(f"{type(exc).__name__}: {exc}")
        ambiguous_child = isinstance(exc, ChildLifecycleError)
    try:
        if status == "COMPLETE":
            if not isinstance(summary, dict) or "_sealed_outputs" not in summary or "_sealed_schemas" not in summary:
                raise RuntimeError("scientific runner supplied no freshly sealed quantitative output proof")
            require_outputs(out, before_exit_record=True, binary_path=context.binary_path,
                            binary_sha256=context.binary_sha256, expected_hashes=summary["_sealed_outputs"],
                            expected_schemas=summary["_sealed_schemas"])
            run.assert_current()
    except BaseException as exc:
        status = "ERROR"
        errors.append(f"{type(exc).__name__}: {exc}")
    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        write_new(out / "command.exit.json", json_text({"exit_code": 0 if status == "COMPLETE" else 1,
                  "status": status, "started": started, "finished": finished, "errors": errors,
                  "identity": context.identity, "in_process_completed": True, **owner,
                  "scope": "consumer computation plus required-output validation; terminal release revalidates identity"}))
    except BaseException as exc:
        status = "ERROR"
        errors.append(f"{type(exc).__name__}: {exc}")
    # A failed run retains its reservation/logs/partial outputs. Hash only the
    # owned evidence; the helper separately verifies reservation/lease identity.
    try:
        files = [p for p in out.rglob("*") if p.is_file()]
        if any(p.is_symlink() for p in files):
            raise RuntimeError("symlink in owned output evidence")
        hashes = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(files)}
        write_new(out / "artifact_hashes.json", json_text({"artifacts_sha256": hashes,
                  "conditional_label": LABEL, "identity": context.identity}))
        hashes["artifact_hashes.json"] = sha256(out / "artifact_hashes.json")
    except BaseException as exc:
        status = "ERROR"
        errors.append(f"{type(exc).__name__}: {exc}")
        hashes = {}
    canonical_hashes = {(out / name).relative_to(main_root).as_posix(): value for name, value in hashes.items()}
    canonical_coverage = [(out / name).relative_to(main_root).as_posix() for name in (*EXPECTED_OUTPUTS, "artifact_hashes.json")]
    terminal = {"status": status, "registration_id": "P7.3-A1", "registration_sha256": identity["registration_sha256"],
                "identity": context.identity, "consumer_identity": identity, "binary_path": str(context.binary_path),
                "binary_sha256": context.binary_sha256, "conditional_label": LABEL, "started": started,
                "finished": finished, "exit_code": 0 if status == "COMPLETE" else 1, "errors": errors,
                "output_dir": out.relative_to(main_root).as_posix(), "artifacts_sha256": hashes,
                "artifact_hashes": canonical_hashes, "expected_outputs": canonical_coverage,
                "expected_output_names": list(EXPECTED_OUTPUTS), "outputs_complete": status == "COMPLETE",
                "ambiguous_child": ambiguous_child,
                "quantitative_tables": {} if summary is None else summary.get("_sealed_schemas", {}),
                "quantitative_summary": None if summary is None else {key: summary.get(key) for key in ("n_draws", "n_comparisons", "n_claimed")}}
    return run.release(terminal)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, default=ROOT / REGISTRATION)
    parser.add_argument("--workers", type=int, choices=[6], default=6)
    args = parser.parse_args(argv)
    try:
        terminal = execute(args.main_root, args.registration)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json_text({"status": terminal.get("status"), "conditional_label": LABEL,
                     "errors": terminal.get("errors", [])}), end="")
    return 0 if terminal.get("status") == "COMPLETE" and terminal.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
