#!/usr/bin/env python3
"""A4c generic A1 profile (docs/phase8_p84c_registration.md section 4).

The lto_v5.profile algorithm over the registered A4c box: 17 uniform grid
points per fitted parameter, the others re-fitted by the registered polish
(40-evaluation cap), warm-started from the neighbouring grid point (upwards
from the optimum's nearest grid point, then downwards);
D = 27 ln(SSE_profile / SSE_min), threshold 3.841. The penalty guard is
``lto_v6.apply_penalty_guard`` itself, with unreachable counts taken from the
objective log at each returned optimum (re-evaluated once if absent).

    .venv/bin/python scripts/phase8/a4c_profile.py --stage primary|fallback

Refuses unless fit_<stage>.json is committed and clean (and the runtime gate
passes); writes profile_<stage>.{json,csv} and a progress log once.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import trent_p84c as A  # noqa: E402

v5, lto_v6, IC = A.v5, A.lto_v6, A.IC
PR = A.REG["profile"]


def profile(obj, free: list[str], opt: dict, bounds: dict, grid_points: int, inner_max_nfev: int,
            n_eff: int, polish, progress=None) -> dict:
    """lto_v5.profile with explicit bounds and polish (identical algorithm and verdict rule)."""
    rows = []
    for name in free:
        others = [k for k in free if k != name]
        grid = np.linspace(*bounds[name], grid_points)
        start = int(np.argmin(np.abs(grid - opt["params"][name])))
        for direction in (range(start, grid_points), range(start - 1, -1, -1)):
            x_prev = [opt["params"][k] for k in others]
            for i in direction:
                obj.fixed_fit[name] = float(grid[i])
                if others:
                    res = polish(obj, others, x_prev, inner_max_nfev)
                else:
                    fun = np.asarray(obj.residuals([], []), float)
                    res = SimpleNamespace(x=[], fun=fun, nfev=1, status=1)
                x_prev = res.x
                rows.append(dict(param=name, i=i, value=float(grid[i]), sse=float(res.fun @ res.fun),
                                 nfev=int(res.nfev), status=int(res.status),
                                 inner_optimizer=bool(others),
                                 inner_message="optimizer" if others else "direct evaluation; no inner fit",
                                 **{f"fit_{k}": float(v) for k, v in zip(others, res.x)}))
                if progress:
                    progress(rows[-1])
        del obj.fixed_fit[name]
    df = pd.DataFrame(rows).sort_values(["param", "i"]).reset_index(drop=True)
    return profile_statistics(df, free, opt["sse"], bounds, n_eff)


def profile_statistics(table: pd.DataFrame, free: list[str], fit_sse: float, bounds: dict,
                       n_eff: int, threshold: float = v5.CHI2_1_95) -> dict:
    """Recompute the registered D statistic and unguarded interval rule from SSE evidence."""
    if not free or set(table["param"]) != set(free):
        raise ValueError("profile table must cover exactly the fitted parameters")
    df = table.copy()
    values = df["sse"].to_numpy(float)
    if not np.isfinite(fit_sse) or fit_sse < 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("profile SSE evidence must be finite and nonnegative")
    sse_min = min(float(fit_sse), float(values.min()))
    if sse_min <= 0:
        raise ValueError("the registered logarithmic profile statistic requires positive SSE_min")
    df["D"] = n_eff * np.log(df["sse"] / sse_min)
    verdicts = {}
    for name in free:
        g = df[df["param"] == name].sort_values("value")
        v, d = g["value"].to_numpy(), g["D"].to_numpy()
        lo_b, hi_b = bounds[name]
        edges_ok = bool(d[0] >= threshold and d[-1] >= threshold)
        inside = np.where(d < threshold)[0]
        if len(inside):
            i0, i1 = inside.min(), inside.max()
            a = v[i0] if i0 == 0 else np.interp(threshold, [d[i0], d[i0 - 1]], [v[i0], v[i0 - 1]])
            b = v[i1] if i1 == len(v) - 1 else np.interp(threshold, [d[i1], d[i1 + 1]], [v[i1], v[i1 + 1]])
        else:
            j = int(np.argmin(d))
            a, b = v[max(j - 1, 0)], v[min(j + 1, len(v) - 1)]
        width_frac = float((b - a) / (hi_b - lo_b))
        verdicts[name] = {"D_at_lower_edge": float(d[0]), "D_at_upper_edge": float(d[-1]),
                          "interval_95": [float(a), float(b)], "interval_width_frac_of_box": width_frac,
                          "interval_contains_no_grid_point": bool(len(inside) == 0),
                          "grid_argmin": float(v[int(np.argmin(d))]),
                          "IDENTIFIED": bool(edges_ok and width_frac <= 0.5)}
    return {"sse_min": sse_min, "sse_min_source": "fit" if sse_min == fit_sse else "profile",
            "n_eff": n_eff, "verdicts": verdicts, "table": df}


def guarded(table: pd.DataFrame, raw_verdicts: dict, free: list[str]) -> dict:
    """Registered guard and A1 verdict of one pass."""
    verdicts = lto_v6.apply_penalty_guard(table, raw_verdicts, PR["inner_max_nfev"])
    identified = [k for k in free if verdicts[k]["IDENTIFIED"]]
    return {"verdicts": verdicts, "identified": identified,
            "not_identified": [k for k in free if k not in identified],
            "failed_profiles": {k: {"existing_rule": verdicts[k]["IDENTIFIED_existing_rule"],
                                    "penalty_dependent": verdicts[k]["penalty_dependent"]}
                                for k in free if k not in identified},
            "A1": "PASS" if free and len(identified) == len(free) else "FAIL"}


def run(stage: str, n_workers: int) -> int:
    stem = A.OUT_DIR / f"profile_{stage}"
    paths = [stem.with_suffix(".json"), stem.with_suffix(".csv"), Path(f"{stem}_progress.log")]
    if any(p.exists() for p in paths):
        print(f"A4c {stage} profile exists; write-once")
        return 2
    fit_rel = A._rel(f"fit_{stage}.json")
    extra = A.artifact_paths("fit", stage)
    if stage == "fallback":
        extra = A.artifact_paths("fit", "primary") + A.artifact_paths("profile", "primary") + extra
    blockers = IC.live_gate(A.REG, extra)
    if blockers:
        print("BLOCKED:\n- " + "\n- ".join(blockers))
        return 3
    IC.check_registered_hashes(A.REG)
    current = IC.identity(A.REG)
    if stage == "fallback":
        A.validate_record(A._rel("fit_primary.json"), current, [])
        A.validate_record(A._rel("profile_primary.json"), current, A.artifact_paths("fit", "primary"))
        fit_dependencies = A.artifact_paths("fit", "primary") + A.artifact_paths("profile", "primary")
    else:
        fit_dependencies = []
    opt = A.validate_record(fit_rel, current, fit_dependencies)
    free = opt["free"]
    if not free:
        print("no fitted parameter remains (all fixed at cited centrals): A1 n/a, nothing to profile")
        return 2
    rows = A.read_calibration_rows()
    started, ident = IC.utc(), IC.identity(A.REG, extra)
    IC.ensure_ac()
    model = A.A4cModel(n_workers)
    obj = v5.Objective(model, rows, opt["fixed"])
    extra_nu: dict = {}

    def progress(r):
        nu = lto_v6._lookup_unreachable(obj.log, r, free)
        if nu is None:      # not in the log: evaluate the returned point once more
            p = {**opt["fixed"], r["param"]: r["value"], **{k: r[f"fit_{k}"] for k in free if k != r["param"]}}
            nu = int((model.predict(p, rows)["status"] == "unreachable").sum())
        extra_nu[(r["param"], r["i"])] = nu
        with open(paths[2], "a") as fh:
            fh.write(f"{r['param']} i={r['i']} value={r['value']:.6g} sse={r['sse']:.6g} "
                     f"nfev={r['nfev']} status={r['status']} n_unreachable={nu}\n")
    try:
        prof = profile(obj, free, opt, A.BOUNDS, PR["grid_points"], PR["inner_max_nfev"], PR["n_eff"],
                       A.polish, progress=progress)
    except Exception as exc:
        end_ident, drift = IC.finish_identity(A.REG, ident, extra)
        A._write_once(paths[0], v5._json({"status": "ERROR", "stage": f"A4c {stage} profile",
                    "error": f"{type(exc).__name__}: {exc}", "drift": drift,
                    "start_identity": ident, "end_identity": end_ident,
                    "started_utc": started, "finished_utc": IC.utc()}))
        return 1
    finally:
        model.close()
    table = prof.pop("table")
    table["n_unreachable"] = [extra_nu.get((p, i)) for p, i in zip(table["param"], table["i"])]
    raw_verdicts = prof.pop("verdicts")
    g = guarded(table, raw_verdicts, free)
    end_ident, drift = IC.finish_identity(A.REG, ident, extra)
    doc = {"status": "ERROR" if drift else "COMPLETE", "error": drift,
           "raw_verdicts": raw_verdicts, **prof, **g, "stage": f"A4c {stage} profile", "fit": fit_rel, "free": free, "fixed": opt["fixed"],
           "rule": PR["rule"], "penalty_guard": PR["penalty_guard"], "threshold": v5.CHI2_1_95,
           "grid_points": PR["grid_points"], "inner_max_nfev": PR["inner_max_nfev"],
           "n_inner_fits": int(table["inner_optimizer"].sum()),
           "n_direct_evaluations": int((~table["inner_optimizer"]).sum()),
           "n_inner_fits_at_cap": int(((table["status"] == 0) | (table["nfev"] >= PR["inner_max_nfev"])).sum()),
           "n_points_with_unreachable_rows": int((table["n_unreachable"].fillna(1) > 0).sum()),
           "jtj_condition_box_scaled_at_fit": opt["jtj_condition_box_scaled"],
           "profile_min_below_fit_sse": bool(float(table["sse"].min()) < opt["sse"]),
           "G2": "closed" if g["A1"] != "PASS" else "A1 passed for this pass only",
           "started_utc": started, "finished_utc": IC.utc(), "start_identity": ident,
           "end_identity": end_ident}
    table.to_csv(paths[1], index=False, mode="x")
    doc["artifacts_sha256"] = {str(p.relative_to(ROOT)): IC.sha256(p) for p in paths[1:]}
    A._write_once(paths[0], v5._json(doc))
    print(f"A4c {stage} profile: A1 {doc['A1']}; identified {g['identified']}; not {g['not_identified']}")
    return 1 if drift else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("primary", "fallback"), required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    return run(a.stage, a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
