#!/usr/bin/env python3
"""P8-A2 ablation ladder (docs/phase8_registration.md, amendment P8-A2).

Each step refits exactly the four v6 knobs with the frozen P7.2/v6
procedure (`lto_v6.run_calibration`, called unchanged: same starts, bounds,
seed, penalty, weights, fuel, fixed settings and 93 calibration rows). The
only substitution is the worker engine: the C++ P8.2 cycle at the step's
ablation level, solved by the v6 matched-thrust procedure
(cpp/catjet_core/thrust_match.hpp, tested identical to the G0 solver).

  --phase calibrate  write-once outputs/phase8/ladder/<step>/calibration_p8_<step>*.
  --phase holdout    score the 87 Trent held-out rows ONCE with the frozen
                     P7.2 held-out tables; refuses unless that step's
                     calibration JSON is committed and unmodified.

Steps: A1 = P8.2 state/liquid fuel/enthalpy dilution (level 1);
       A2 = A1 + enthalpy HP/IP/LP turbine and cited cooling (level 2).
A3 (P8.3 nozzles) needs a declared fixed-area procedure first; not here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for path in (ROOT, ROOT / "scripts" / "optimization", ROOT / "scripts" / "phase8"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lto_v5 as v5  # noqa: E402
import lto_v6  # noqa: E402

LEVELS = {"A1": 1, "A2": 2}
OUT_ROOT = ROOT / "outputs" / "phase8" / "ladder"


def _command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LadderEngine:
    """The solve_task engine interface over the C++ P8.2 cycle."""

    def __init__(self, level: int, nox_fit_exclude_models):
        from simulation.catjet_backend import CppEngine, load_core

        with contextlib.redirect_stdout(io.StringIO()):
            self.base = CppEngine(nox_fit_exclude_models=nox_fit_exclude_models)
        self.core = load_core()
        self.level = level
        self.p82 = self.core.P82Engine(self.base.mechanism)
        # the same objects solve_task mutates on the v6 engine
        self.design_point = self.base.design_point
        self.compressor = self.base.compressor
        self.turbine_design = self.base.turbine_design
        self.emissions = getattr(self.base, "emissions", None)

    def run_at_thrust(self, target_kN, fuel_blend, combustor_efficiency=None, phi_guess=None):
        from integrated_engine import ThrustTargetUnreachable

        self.base.push_config()
        c = self.p82.config
        c.base = self.base.core.config
        self.p82.config = c
        d = self.p82.run_at_thrust(target_kN, *self.base.fuel_args(fuel_blend),
                                   combustor_efficiency, self.level, phi_guess=phi_guess)
        if d["status"] != "converged":
            raise ThrustTargetUnreachable(d["reason"], target_kN, dict(d["info"]))
        return d


def _init_ladder_worker(level: int, nox_fit_exclude_models):
    logging.getLogger("cantera").setLevel(logging.ERROR)
    os.chdir(ROOT)
    v5._ENGINE = LadderEngine(level, set(nox_fit_exclude_models))
    excluded = getattr(getattr(v5._ENGINE.base, "emissions", None), "nox_fit_exclude_models", None)
    if excluded is not None and excluded != set(nox_fit_exclude_models):
        raise RuntimeError("NOx held-out exclusion not applied")


class LadderModel(v5.V5Model):
    """V5Model task creation and scoring; workers run the ladder engine."""

    def __init__(self, level: int, fixed: dict, nox_fit_exclude_models, n_workers: int, fuel):
        if fixed.get("eta_b") is None or not nox_fit_exclude_models:
            raise ValueError("fixed eta_b and held-out NOx exclusions are required")
        self.fixed = fixed
        self.fuel = fuel
        self.guess = {}
        self.n_evals = 0
        self.nox_fit_exclude_models = sorted(nox_fit_exclude_models)
        self.pool = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_ladder_worker,
                                        initargs=(level, self.nox_fit_exclude_models))


def _provenance(step: str) -> dict:
    core_so = next((ROOT / "cpp" / "build").glob("catjet_core*.so"))
    return {"step": step, "ablation_level": LEVELS[step],
            "git_sha": _command("git", "rev-parse", "HEAD"),
            "dirty_source": _command("git", "status", "--porcelain", "--", "cpp", "scripts", "simulation"),
            "module_sha256": _sha256(core_so),
            "p82_defaults": {"vaporization_J_kg": 360000.0, "ngv_fraction": 0.0641,
                             "rotor_fraction": 0.0275, "pressure_steps": 50},
            "registration": "docs/phase8_registration.md P8-A2; docs/phase8_p82_registration.md",
            "solver": "cpp/catjet_core/thrust_match.hpp (v6 procedure, cycle callback replaced)"}


def _use_ladder(step: str) -> None:
    level = LEVELS[step]
    lto_v6.make_model = lambda reg6, split, n_workers: LadderModel(
        level, reg6["fixed_central"], split["heldout_models"], n_workers,
        lto_v6.fuel_composition(reg6))
    lto_v6.FIT_JSON = f"calibration_p8_{step}.json"
    lto_v6.FIT_EVALS = f"calibration_p8_{step}_evaluations.csv"
    lto_v6.FIT_ROWS = f"calibration_p8_{step}_rows.csv"


def calibrate(step: str, n_workers: int) -> dict:
    out = OUT_ROOT / step
    if (out / f"calibration_p8_{step}.json").exists():
        raise SystemExit(f"{step} calibration exists; write-once")
    out.mkdir(parents=True, exist_ok=True)
    prov = out / f"calibration_p8_{step}_provenance.json"
    with prov.open("x") as f:
        f.write(json.dumps(_provenance(step), indent=2) + "\n")
    _use_ladder(step)
    result = lto_v6.run_calibration(out, n_workers=n_workers)
    print(f"{step} calibration: SSE {result['sse']:.6g}, in-sample MAPE "
          f"{result['calibration_weighted_mape_pct']:.3f} %, params {result['params']}, "
          f"unreachable {result['n_unreachable']}")
    return result


def holdout(step: str, n_workers: int) -> dict:
    out = OUT_ROOT / step
    fit_path = out / f"calibration_p8_{step}.json"
    rel = str(fit_path.relative_to(ROOT))
    if subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                      capture_output=True).returncode != 0 or \
            subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0:
        raise SystemExit("commit the step's fitted parameters before scoring held-out rows (P8-A2)")
    paths = [out / f"holdout_p8_{step}{suffix}" for suffix in (".csv", "_summary.csv", ".json")]
    if any(p.exists() for p in paths):
        raise SystemExit(f"{step} held-out score exists; held-out rows are scored once")
    _use_ladder(step)
    reg6 = lto_v6.load_registration_v6()
    base = lto_v6.base_registration(reg6)
    split = v5.load_split()
    fitted = json.loads(fit_path.read_text())
    cal = v5.calibration_rows(split)
    held = v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True),
                            split["heldout_groups"])
    model = lto_v6.make_model(reg6, split, n_workers)
    try:
        pred = model.predict(fitted["params"], held)
    finally:
        model.close()
    df, summary, fields = v5.holdout_tables(base, cal, held, pred)
    per_mode = summary.set_index("Scope")
    result = {
        "step": step, "calibration": rel, "fitted_params": fitted["params"],
        **fields,
        "per_mode_group_weighted_mape_pct": {
            s: {t: float(per_mode.loc[s, f"{t} group-weighted MAPE (%)"]) for t in ("Model", "B0", "B1")}
            for s in per_mode.index},
        "A4": lto_v6.a4_informativeness(df),
        "n_unreachable": int((pred["status"] == "unreachable").sum()),
        "v6_reference_pct": 1.830,
        "provenance": _provenance(step),
        "note": "scored once after the committed calibration; not used for any selection (P8-A2)",
    }
    df.to_csv(paths[0], index=False)
    summary.to_csv(paths[1], index=False)
    v5._write_new(paths[2], v5._json(result))
    print(f"{step} held-out: model {fields['primary_group_weighted_mape_pct']['model']:.3f} %, "
          f"unreachable {result['n_unreachable']}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", choices=tuple(LEVELS), required=True)
    ap.add_argument("--phase", choices=("calibrate", "holdout"), required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    (calibrate if a.phase == "calibrate" else holdout)(a.step, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
