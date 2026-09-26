#!/usr/bin/env python3
"""
P5.2 Step 1 — Identifiability profile of the LTO calibration.

For every fitted parameter of a frozen calibration, fix it at points across its
search box, re-optimise every other fitted parameter (bounded Powell, warm-
started along the profile), and record the best objective at each point — the
profile of the objective. Then run the explicit ridge check of Phase 5 finding
F2 (docs/plan.md): move k_mdot while re-solving phi_idle / phi_app so that
idle and approach fuel flow are unchanged, and see whether the objective moves.

VERDICT RULE (fixed before any profile was run; objective units are the
calibration's own objective — mean |relative error|, so 0.001 = 0.1 pp):

A parameter is IDENTIFIED iff all three hold:
  1. the profile rises by at least MARGIN above its minimum at BOTH box edges;
  2. the DELTA-sublevel interval {theta : profile(theta) <= min + DELTA} is
     interior (touches neither box edge) — a minimum on a search-box edge is
     set by the prior box, not by the data;
  3. that interval is narrower than WIDTH_FRAC of the box — a flat valley
     (a ridge of equal objective, F2) rises at the box edges only because the
     other parameters run into their own boxes; its width exposes it.
Otherwise it is NOT IDENTIFIED. Condition 3 is what makes the test able to
see the F2 ridge at all: with conditions 1-2 alone, k_mdot on the v4
objective would pass because the phi boxes stop the ridge.

Speed: the cycle re-parses the CRECK mechanism in every ct.Solution() call
(~1 s each). Within this script only, ct.Solution is served from a small pool
of preloaded objects; every call site sets the full thermodynamic state before
reading (TP + set_equivalence_ratio, or TPY), so results are unchanged. The
script verifies bit-identical fuel flow and thrust at the calibrated point
with and without the pool before profiling, and records that check.

Outputs (tag from the calibration file name, e.g. ``_v4``):
    outputs/identifiability_profile<tag>.json   verdicts, criteria, ridge check
    outputs/identifiability_profile<tag>.csv    every profile point

Usage::

    python scripts/validation/identifiability_profile.py \\
        --calibration outputs/calibration_trent1000_ae3_v4.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "optimization"))

import cantera as ct  # noqa: E402

import calibrate_lto as cal  # noqa: E402

logging.getLogger("cantera").setLevel(logging.ERROR)

# ---- verdict rule (see module docstring; fixed before any run) ----
MARGIN = 0.005       # rise at both box edges: 0.5 percentage points
DELTA = 1e-4         # sublevel threshold: 0.01 percentage points
WIDTH_FRAC = 0.02    # identified interval narrower than 2 % of the box
# ---- profile resolution ----
N_GRID = 15          # grid points across each box (plus the calibrated value)
N_BISECT = 10        # bisection steps locating each end of the DELTA interval
RIDGE_K = (0.58, 0.60)   # F2 ridge points (the calibrated k_mdot is added)


# --------------------------------------------------------------------------
# Calibration record and fast evaluator
# --------------------------------------------------------------------------
def load_calibration(path: Path) -> dict:
    rec = json.loads(path.read_text())
    fixed = rec.get("fixed_parameters", {})
    best = dict(rec["best_params"])
    # v5+ records list what was fitted and what was fixed; v1-v4 fitted all seven
    fitted = list(rec.get("fitted_parameters") or best.keys())
    full = {k: best.get(k, fixed.get(k)) for k in cal.PARAM_BOUNDS}
    missing = [k for k, v in full.items() if v is None]
    if missing:
        raise ValueError(f"{path.name}: no value for {missing}")
    bounds = {k: tuple(v) for k, v in (rec.get("search_bounds") or cal.PARAM_BOUNDS).items()}
    return {
        "path": path, "record": rec, "fitted": fitted, "values": full, "bounds": bounds,
        "beta": fixed.get("combustor_air_fraction", 1.0),
        "xi": fixed.get("combustor_heat_loss_fraction", 0.0),
        "objective": (rec.get("objective") or {}).get("name", "fuel_flow_v4"),
    }


@contextlib.contextmanager
def pooled_solutions(size: int = 4):
    """Serve ct.Solution(...) from a rotating pool of preloaded objects."""
    orig, pools = ct.Solution, {}

    def pooled(*a, **k):
        key = (a, tuple(sorted(k.items())))
        if key not in pools:
            pools[key] = itertools.cycle([orig(*a, **k) for _ in range(size)])
        return next(pools[key])

    ct.Solution = pooled
    try:
        yield
    finally:
        ct.Solution = orig


class Evaluator:
    """Objective of a full parameter set, with per-mode caching.

    A mode's cycle depends only on (eta_b, p_loss, k_pi, k_mdot, its own phi),
    so a Powell line search along one phi re-runs one mode, not three.
    """

    def __init__(self, c: dict):
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine = cal.IntegratedTurbofanEngine()
        self.c = c
        self.cache: dict = {}
        self.n_cycles = 0

    def perf(self, p: dict) -> dict:
        out = {}
        for mode in cal.ICAO_TARGETS:
            key = (mode, p["eta_combustor"], p["pressure_loss"], p["k_pi"], p["k_mdot"],
                   p[cal.PHI_KEYS[mode]])
            if key not in self.cache:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.cache[key] = cal.run_lto_modes(self.engine, p, self.c["beta"],
                                                        self.c["xi"], modes=(mode,))[mode]
                self.n_cycles += 1
            out[mode] = self.cache[key]
        return out

    def __call__(self, p: dict) -> float:
        try:
            return cal.OBJECTIVES[self.c["objective"]](self.perf(p))
        except Exception:
            return cal.CRASH_PENALTY


def pool_equality_check(c: dict) -> dict:
    """Fuel flow and thrust at the calibrated point, pooled vs plain Cantera."""
    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            eng = cal.IntegratedTurbofanEngine()
            perf = cal.run_lto_modes(eng, c["values"], c["beta"], c["xi"])
        return {m: (float(v["fuel_mass_flow"]), float(v["thrust_N"])) for m, v in perf.items()}
    plain = run()
    with pooled_solutions():
        pooled = run()
    if plain != pooled:
        raise RuntimeError(f"pooled Cantera differs from plain: {plain} vs {pooled}")
    return {"identical": True, "per_mode_fuel_flow_thrust": plain}


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------
def inner_min(ev: Evaluator, c: dict, name: str, value: float, start: dict) -> tuple[float, dict]:
    """min over the other fitted parameters with `name` fixed at `value` (normalised Powell)."""
    free = [k for k in c["fitted"] if k != name]
    lo = np.array([c["bounds"][k][0] for k in free])
    hi = np.array([c["bounds"][k][1] for k in free])
    base = dict(c["values"])
    base[name] = value

    def unpack(z):
        p = dict(base)
        p.update(zip(free, lo + np.clip(z, 0.0, 1.0) * (hi - lo)))
        return p

    if not free:
        return ev(base), base
    z0 = np.clip((np.array([start[k] for k in free]) - lo) / (hi - lo), 0.0, 1.0)
    res = minimize(lambda z: ev(unpack(z)), z0, method="Powell",
                   bounds=[(0.0, 1.0)] * len(free),
                   options={"xtol": 1e-7, "ftol": 1e-12, "maxfev": 6000})
    best = unpack(res.x)
    return ev(best), best


def profile_param(ev: Evaluator, c: dict, name: str) -> dict:
    lo, hi = c["bounds"][name]
    cal_val = float(np.clip(c["values"][name], lo, hi))
    grid = np.unique(np.append(np.linspace(lo, hi, N_GRID), cal_val))
    i0 = int(np.argmin(np.abs(grid - cal_val)))
    f = np.full(len(grid), np.nan)
    sols: list = [None] * len(grid)
    f[i0], sols[i0] = inner_min(ev, c, name, grid[i0], c["values"])
    for order in (range(i0 + 1, len(grid)), range(i0 - 1, -1, -1)):   # continuation
        prev = sols[i0]
        for i in order:
            f[i], sols[i] = inner_min(ev, c, name, grid[i], prev)
            # a second start from the calibrated point guards the continuation
            f2, s2 = inner_min(ev, c, name, grid[i], c["values"])
            if f2 < f[i]:
                f[i], sols[i] = f2, s2
            prev = sols[i]
    points = [(float(g), float(v)) for g, v in zip(grid, f)]
    fmin = float(np.min(f))
    inside = np.where(f <= fmin + DELTA)[0]
    a, b = int(inside.min()), int(inside.max())

    def crossing(i_in: int, i_out: int) -> float:
        """Bisect between an in-set and an out-of-set grid point for profile = fmin + DELTA."""
        x_in, x_out = grid[i_in], grid[i_out]
        warm = sols[i_in]
        for _ in range(N_BISECT):
            mid = 0.5 * (x_in + x_out)
            fm, sm = inner_min(ev, c, name, mid, warm)
            points.append((float(mid), float(fm)))
            if fm <= fmin + DELTA:
                x_in, warm = mid, sm
            else:
                x_out = mid
        return float(0.5 * (x_in + x_out))

    touches_lo, touches_hi = a == 0, b == len(grid) - 1
    iv_lo = float(grid[0]) if touches_lo else crossing(a, a - 1)
    iv_hi = float(grid[-1]) if touches_hi else crossing(b, b + 1)
    rise_lo, rise_hi = float(f[0] - fmin), float(f[-1] - fmin)
    width_frac = (iv_hi - iv_lo) / (hi - lo)
    identified = (rise_lo >= MARGIN and rise_hi >= MARGIN
                  and not touches_lo and not touches_hi and width_frac < WIDTH_FRAC)
    reasons = []
    if rise_lo < MARGIN or rise_hi < MARGIN:
        reasons.append(f"edge rise {rise_lo:.2e} / {rise_hi:.2e} < margin {MARGIN}")
    if touches_lo or touches_hi:
        reasons.append("minimum interval touches the "
                       + " and ".join(s for s, t in (("lower", touches_lo), ("upper", touches_hi)) if t)
                       + " box edge")
    if width_frac >= WIDTH_FRAC:
        reasons.append(f"flat valley: interval {width_frac:.1%} of the box >= {WIDTH_FRAC:.0%}")
    return {
        "parameter": name, "bounds": [lo, hi], "calibrated": cal_val,
        "profile_min": fmin, "argmin": float(grid[int(np.argmin(f))]),
        "edge_rise_lower": rise_lo, "edge_rise_upper": rise_hi,
        "interval": [iv_lo, iv_hi], "interval_width_frac": width_frac,
        "touches_lower": touches_lo, "touches_upper": touches_hi,
        "identified": bool(identified),
        "why_not": "; ".join(reasons) if not identified else "",
        "points": sorted(points),
        "argmin_params": sols[int(np.argmin(f))],
    }


def ridge_check(ev: Evaluator, c: dict) -> dict:
    """F2: along k_mdot, re-solve phi_idle / phi_app so idle and approach fuel flow are unchanged.

    Fuel flow = phi * f_st * beta * m_rated * x^k_mdot with f linear in phi, so
    phi_mode(k) = phi_mode(k0) * x_mode^(k0 - k) holds each mode's fuel flow fixed.
    Box limits are deliberately ignored: the question is what the data can see.
    """
    v = c["values"]
    k0 = v["k_mdot"]
    rows = []
    for k in sorted(set(RIDGE_K) | {k0}):
        p = dict(v)
        p["k_mdot"] = k
        for mode, key in (("Idle", "phi_idle"), ("Approach", "phi_app")):
            p[key] = v[key] * cal.ICAO_TARGETS[mode]["power_fraction"] ** (k0 - k)
        perf = ev.perf(p)
        rows.append({
            "k_mdot": k, "phi_idle": p["phi_idle"], "phi_app": p["phi_app"],
            "objective": ev(p),
            "fuel_flow_mape_pct": 100 * cal.fuel_flow_error(perf),
            **{f"thrust_{m}_kN": float(perf[m]["thrust_N"]) / 1e3 for m in cal.ICAO_TARGETS},
            "in_box": all(c["bounds"][q][0] <= p[q] <= c["bounds"][q][1] for q in ("phi_idle", "phi_app")),
        })
    objs = [r["objective"] for r in rows]
    thr = [r["thrust_Idle_kN"] for r in rows]
    return {
        "rows": rows,
        "objective_spread": float(max(objs) - min(objs)),
        "idle_thrust_spread_pct": float(100 * (max(thr) - min(thr)) / np.mean(thr)),
        "ridge_present": bool(max(objs) - min(objs) <= DELTA),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--params", nargs="*", help="profile only these (default: every fitted parameter)")
    ap.add_argument("--out-tag", default=None, help="output suffix (default from the calibration file name)")
    args = ap.parse_args()

    c = load_calibration(args.calibration.resolve())
    tag = args.out_tag if args.out_tag is not None else "_" + args.calibration.stem.rsplit("_", 1)[-1]
    print(f"Calibration {args.calibration}  objective {c['objective']}  fitted {c['fitted']}")
    pool = pool_equality_check(c)
    print("Pooled-Cantera check: identical fuel flow and thrust at the calibrated point")

    t0 = time.time()
    with pooled_solutions():
        ev = Evaluator(c)
        f_cal = ev(c["values"])
        print(f"Objective at the calibrated point: {f_cal:.6f}")
        ridge = ridge_check(ev, c)
        print(f"Ridge check (F2): objective spread {ridge['objective_spread']:.2e} "
              f"-> ridge {'PRESENT' if ridge['ridge_present'] else 'broken'}; "
              f"idle thrust spread {ridge['idle_thrust_spread_pct']:.1f} %")
        profiles = []
        for name in (args.params or c["fitted"]):
            t = time.time()
            pr = profile_param(ev, c, name)
            profiles.append(pr)
            print(f"  {name:<14} min {pr['profile_min']:.6f} at {pr['argmin']:.4f}  "
                  f"edge rise {pr['edge_rise_lower']:.4f}/{pr['edge_rise_upper']:.4f}  "
                  f"interval [{pr['interval'][0]:.4f}, {pr['interval'][1]:.4f}] "
                  f"({pr['interval_width_frac']:.1%})  -> "
                  f"{'IDENTIFIED' if pr['identified'] else 'not identified'}  ({time.time() - t:.0f} s)")

    out = {
        "calibration": str(c["path"].relative_to(REPO_ROOT)),
        "objective": c["objective"],
        "objective_at_calibration": f_cal,
        "fitted": c["fitted"],
        "criteria": {"MARGIN": MARGIN, "DELTA": DELTA, "WIDTH_FRAC": WIDTH_FRAC,
                     "N_GRID": N_GRID, "N_BISECT": N_BISECT},
        "pooled_cantera_check": pool,
        "ridge_check": ridge,
        "identified": [p["parameter"] for p in profiles if p["identified"]],
        "not_identified": [p["parameter"] for p in profiles if not p["identified"]],
        "profiles": [{k: v for k, v in p.items() if k != "points"} for p in profiles],
        "cycle_evaluations": ev.n_cycles,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_json = REPO_ROOT / "outputs" / f"identifiability_profile{tag}.json"
    out_csv = REPO_ROOT / "outputs" / f"identifiability_profile{tag}.csv"
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    pd.DataFrame([{"parameter": p["parameter"], "value": x, "profile_objective": y}
                  for p in profiles for x, y in p["points"]]).to_csv(out_csv, index=False)
    print(f"\nIdentified: {out['identified']}\nNot identified: {out['not_identified']}")
    print(f"Written: {out_json.relative_to(REPO_ROOT)}, {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
