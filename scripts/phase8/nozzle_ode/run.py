"""Execute the committed smooth-nozzle diagnostic after all shared gates pass.

Usage: caffeinate -i nice -n 15 .venv/bin/python -m scripts.phase8.nozzle_ode.run
No scientific import, source property read or exact label occurs before ownership.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import sys
import time
from pathlib import Path

from .inputs import Blocked, assert_disjoint, read_object, sha256, source_artifacts, write_json

SOURCE_ROOT = Path(__file__).resolve().parents[3]
REGISTRATION = "docs/phase8_nozzle_ode_registration.json"


def make_splits(reg, properties):
    import torch
    box = reg["gas_and_geometry"]
    splits = {}
    for name, count, seed in (("train",64,20261004), ("validation",32,20261005), ("synthetic_test",64,20261006)):
        points = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw_base2(int(math.log2(count)), dtype=torch.float64).tolist()
        rows = []
        for regime in ("smooth_subcritical", "smooth_choked"):
            npr_lo, npr_hi = reg["regimes"][regime]["NPR"]
            for i, unit in enumerate(points):
                rows.append({"case_id":f"{name}|{regime}|{i:06d}", "regime":regime,
                             "NPR":npr_lo+(npr_hi-npr_lo)*unit[0],
                             "gamma":box["gamma"][0]+(box["gamma"][1]-box["gamma"][0])*unit[1],
                             "R":box["R_J_kg_K"][0]+(box["R_J_kg_K"][1]-box["R_J_kg_K"][0])*unit[2]})
            if name == "synthetic_test":
                for gamma in box["gamma"]:
                    for gas_R in box["R_J_kg_K"]:
                        for npr in (npr_lo, (npr_lo+npr_hi)/2, npr_hi):
                            rows.append({"case_id":f"stress|{regime}|{gamma}|{gas_R}|{npr}",
                                         "regime":regime, "NPR":npr, "gamma":gamma, "R":gas_R})
        splits[name] = rows
    product = []
    for source in properties:
        for regime, nprs in (("smooth_subcritical",reg["splits"]["product_test"]["NPR_subcritical"]),
                             ("smooth_choked",reg["splits"]["product_test"]["NPR_choked"])):
            for npr in nprs:
                product.append({"case_id":f"{source['source']}|{source['source_id']}|{regime}|{npr}",
                    "regime":regime, "NPR":npr, "gamma":source["gamma"], "R":source["R"],
                    "source":source["source"], "source_id":source["source_id"], "input_sha256":source["input_sha256"]})
    splits["product_test"] = product
    assert_disjoint(splits)
    return splits


def old_evidence(root, reg):
    old_reg_path = root / reg["dependencies"]["old_registration"]
    old = read_object(old_reg_path)
    output = root / old["output_dir"]
    hashes_path = output / "hashes.json"
    hashes = read_object(hashes_path)
    frozen = {str(old_reg_path.relative_to(root)):sha256(old_reg_path),
              str(hashes_path.relative_to(root)):sha256(hashes_path)}
    for name in ("report.json", "nozzle_scores.json"):
        if hashes.get(name) != sha256(output / name):
            raise Blocked("original Track 4 evidence hash mismatch")
        frozen[str((output / name).relative_to(root))] = sha256(output / name)
    report, scores = read_object(output / "report.json"), read_object(output / "nozzle_scores.json")
    identity = report.get("start_identity", {})
    if identity.get("registration_sha256") != sha256(old_reg_path) or report.get("identity_drift") != []:
        raise Blocked("original Track 4 registration/identity drift")
    for name, expected in identity.get("source_sha256", {}).items():
        path = root / "scripts/phase8/pinn_diagnostics" / name
        if sha256(path) != expected:
            raise Blocked("original Track 4 source differs from scored oracle")
        frozen[str(path.relative_to(root))] = expected
    if identity.get("source_sha256", {}).get("nozzle_verification.py") is None:
        raise Blocked("original Track 4 oracle proof missing")
    rungs = {name:scores.get(name, {}) for name in reg["dependencies"]["old_rungs_required"]}
    if any(row.get("status") != "PASS" for row in rungs.values()) or report.get("nozzle") != scores:
        raise Blocked("original choked/shock oracle rung is not a verified PASS")
    return {"rungs":rungs, "evidence_sha256":frozen}, frozen


def freeze_case_inputs(out, splits):
    fields = ["split", "case_id", "regime", "NPR", "gamma", "R", "source", "source_id", "input_sha256"]
    with (out / "case_inputs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, cases in splits.items():
            writer.writerows({"split":name, **case} for case in cases)
    write_json(out / "split_manifest.json", {name:{"conditions":len(cases), "case_ids":[case["case_id"] for case in cases]}
                                               for name, cases in splits.items()})


def technical_readme(out):
    with (out / "README.md").open("x") as stream:
        stream.write("""Smooth nozzle diagnostic artifact schema

Run command: caffeinate -i nice -n 15 .venv/bin/python -m scripts.phase8.nozzle_ode.run

case_inputs.csv freezes whole conditions. Primitive values in test_predictions.csv
are rho/rho_ref, u/u_ref, T/T0, p/p0; x is in metres (xi=x/1m).
p0=100000Pa, T0=1000K, R_ref=287J/kg/K, A=1+0.5xi^2m^2.
test_scores.csv records each condition/field/residual bar for every arm and seed.
report.json contains quantitative coverage, accuracy and paired comparison gates.

Source properties are frozen C++ burner-state coefficients, used as a constant
calorically-perfect gas. gamma in[1.2,1.4] and R in[260,330]J/kg/K are a prospective
diagnostic envelope. source_property_coverage.json retains all4164 requested IDs;
invalid/out-of-envelope values block the registered coverage without clipping.

Subcritical NPR[1.02,1.08] and smooth choked NPR[8,12] only. Shocks, choking
transition, overexpanded exits, external jet expansion, variable cp, reaction and
viscous effects are excluded. oracle_gate.json preserves original rung3 shock
evidence and independent exact-reference checks. Exact fields are references;
models use identical supervised interior/boundary/throat labels in both arms.

All six final checkpoints are frozen before score_reservation.json and exact-test
evaluation. No held-out stopping, checkpoint choice, retries or production claim.
Each physics seed must satisfy every field/case/regime/panel accuracy bar. The
paired ratio comparison separately tests incremental residual benefit. Original
shock evidence remains visible; this study does not establish empirical validity.
""")


def run(main_root):
    root = Path(main_root).resolve()
    if root != SOURCE_ROOT:
        raise Blocked("execute only the committed implementation integrated into main")
    reg = read_object(root / REGISTRATION)
    if reg.get("registration_id") != "P8-NOZZLE-ODE-A1-20261004":
        raise Blocked("foreign nozzle registration")
    from scripts.phase8 import scientific_workflow_gate as gate
    sources = {name:gate.file_identity(root/name) for name in reg["relevant_new_files"]}
    expected = {"registration_sha256":sha256(root/REGISTRATION), "files":sources}
    context = gate.prepare_context(root, REGISTRATION, expected_consumer_identity=expected, require_g0=True)
    context.require_idle_ac()
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        raise Blocked("registered nice>=15 is required")
    owned = context.acquire_run(reg["outputs"]["root"], expected["registration_sha256"], identity=context.identity)
    out, errors, frozen, status = owned.out, [], {}, "ERROR"
    started = time.perf_counter()
    output_hashes, expected_outputs, complete = {}, [], False
    log = (out / "execution.log").open("x")
    def record(code):
        log.write(f"{gate.utc()} {code}\n")
        log.flush()
    def guard():
        owned.assert_current()
        for name, digest in frozen.items():
            if sha256(root/name) != digest:
                raise Blocked(f"frozen input/checkpoint drift: {name}")
    def freeze_finished():
        # Call only between phases, when every file except execution.log is
        # closed. Finished artifacts remain immutable throughout later phases.
        guard()
        for path in out.rglob("*"):
            if path.is_file() and path.name != "execution.log":
                frozen.setdefault(str(path.relative_to(root)),sha256(path))
    try:
        record("RESERVED")
        for directory in ("training_logs", "checkpoints"):
            (out / directory).mkdir()
        technical_readme(out)
        write_json(out / "config.json", reg)
        properties, source_hashes = source_artifacts(root, reg, context, out)
        frozen.update(source_hashes)
        historical, historical_hashes = old_evidence(root, reg)
        frozen.update(historical_hashes)
        freeze_finished()
        # Scientific imports are confined to the owned, verified execution path.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from . import model, oracle, score
        write_json(out / "environment.json", {"python":platform.python_version(), "platform":platform.platform(),
            "torch":torch.__version__, "numpy":np.__version__, "dtype":"float64", "device":"cpu", "threads":1,
            "nice":os.getpriority(os.PRIO_PROCESS,0), "context_identity":context.identity,
            "source_commit":gate.git(root,"rev-parse","HEAD").strip()})
        splits = make_splits(reg, properties)
        freeze_case_inputs(out, splits)
        frozen.update({str((out/name).relative_to(root)):sha256(out/name) for name in ("case_inputs.csv", "split_manifest.json")})
        # Freeze and revalidate the complete pre-label identity before opening
        # even training-only exact fields. The reviewed registration/code fix
        # the manifest schema and all case-selection rules prospectively.
        write_json(out / "input_manifest.json", {"registration_sha256":expected["registration_sha256"],
            "source_files":sources, "input_hashes":dict(frozen), "context_identity":context.identity,
            "case_counts":{name:len(cases) for name,cases in splits.items()},
            "declared_outputs":reg["outputs"]["required"]})
        frozen[str((out/"input_manifest.json").relative_to(root))] = sha256(out/"input_manifest.json")
        freeze_finished()
        original = oracle.load_original(root)
        admissible = oracle.admissibility(original, reg)
        reference = oracle.ExactReference(original, reg)
        training_oracle_errors = []
        for case in splits["train"]:
            _, values = reference.profile(case, np.linspace(-1,1,161))
            training_oracle_errors.append({"case_id":case["case_id"], **values})
        write_json(out / "oracle_gate.json", {"status":"PASS", "admissibility":admissible,
            "original_evidence":historical, "training_reference_errors":training_oracle_errors,
            "test_reference_state":"SEALED_UNCOMPUTED", "excluded_shocks":True})
        frozen[str((out/"oracle_gate.json").relative_to(root))] = sha256(out/"oracle_gate.json")
        freeze_finished()
        networks = {}
        for seed in reg["models"]["paired_seeds"]:
            record(f"TRAIN_SEED_{seed}")
            pair = model.train_pair(reg,seed,splits["train"],reference,out,guard,
                                    {"context":context.identity,"input_manifest_sha256":sha256(out/"input_manifest.json")})
            for arm, network in pair.items():
                networks[(seed,arm)] = network
                checkpoint = out / "checkpoints" / f"{arm}-seed{seed}.pt"
                frozen[str(checkpoint.relative_to(root))] = sha256(checkpoint)
            freeze_finished()
        guard()
        record("VALIDATION_FINAL_CHECKPOINTS")
        validation = score.score_panels(reg,{"validation":splits["validation"]},networks,reference,out,guard,final_test=False)
        freeze_finished()
        checkpoint_hashes = {name:digest for name,digest in frozen.items() if name.endswith(".pt")}
        if len(checkpoint_hashes) != 6:
            raise Blocked("six final checkpoints are required before score reservation")
        write_json(out / "score_reservation.json", {"registration_sha256":expected["registration_sha256"],
            "checkpoint_hashes":checkpoint_hashes, "input_manifest_sha256":sha256(out/"input_manifest.json"),
            "test_state":"OPENING_ONCE", "created_utc":gate.utc()})
        freeze_finished()
        record("SINGLE_FINAL_TEST_PASS")
        tests = score.score_panels(reg,{name:splits[name] for name in ("synthetic_test","product_test")},
                                   networks,reference,out,guard,final_test=True)
        for (panel, regime, seed, arm), values in tests.items():
            expected_count = sum(case["regime"] == regime for case in splits[panel])
            if values["cases"] != expected_count:
                raise Blocked("incomplete final test coverage")
        decisions = score.paired_decisions(tests,reg)
        status = "PASS" if decisions["registered_comparison_pass"] else "FAIL"
        write_json(out / "report.json", {"registration_id":reg["registration_id"], "status":status, **decisions,
            "scores":[{"panel":k[0],"regime":k[1],"seed":k[2],"arm":k[3],**v} for k,v in tests.items()],
            "validation_scores":[{"panel":k[0],"regime":k[1],"seed":k[2],"arm":k[3],**v} for k,v in validation.items()],
            "source_rows":len(properties), "synthetic_test_conditions":len(splits["synthetic_test"]),
            "product_test_conditions":len(splits["product_test"]), "original_shock_oracle_status":"PASS",
            "wall_s":time.perf_counter()-started, "start_identity":context.identity, "end_identity":context.identity})
        freeze_finished()
        complete = True
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        status = "BLOCKED" if isinstance(exc, Blocked) else "ERROR"
        record(status)
    finally:
        record("FINALIZING")
        log.close()
        files = [p for p in sorted(out.rglob("*")) if p.is_file()]
        output_hashes = {str(p.relative_to(root)):sha256(p) for p in files}
        write_json(out / "hashes.json", output_hashes)
        output_hashes[str((out/"hashes.json").relative_to(root))] = sha256(out/"hashes.json")
        expected_outputs = [str((out/name).relative_to(root)) for name in reg["outputs"]["required"] if not name.endswith("/")]
        expected_outputs += [str((out/"checkpoints"/f"{arm}-seed{seed}.pt").relative_to(root))
                             for seed in reg["models"]["paired_seeds"] for arm in reg["models"]["arms"]]
        expected_outputs += [str((out/"training_logs"/f"{arm}-seed{seed}.jsonl").relative_to(root))
                             for seed in reg["models"]["paired_seeds"] for arm in reg["models"]["arms"]]
        expected_outputs += [str((out/"score_reservation.json").relative_to(root)), str((out/"case_selection.json").relative_to(root))]
        try:
            guard()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            status, complete = "ERROR", False
        terminal = owned.release({"status":status, "exit_code":0 if status == "PASS" else 1, "errors":errors,
            "outputs_complete":bool(complete and set(expected_outputs) <= set(output_hashes)),
            "expected_outputs":expected_outputs, "artifact_hashes":output_hashes,
            "wall_s":time.perf_counter()-started})
    print(terminal["status"])
    return 0 if terminal["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args(argv)
    try:
        return run(args.main_root)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
