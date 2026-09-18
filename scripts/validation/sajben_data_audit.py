#!/usr/bin/env python3
"""
P4.1 — Sajben data-reality gate (GO / NO-GO for the nozzle LE-PINN retrain).

This is a written finding, not a training run.  It answers three questions
before any epoch is spent:

1. What does ``data/processed/master_shock_dataset.pt`` actually contain?
   (Does any row carry the shock/boundary-layer physics that the Sajben
   benchmark measures?  Do the inputs even identify which flow case a row
   belongs to?)
2. What is the best any model trained on that data could score on the
   pre-registered gate metric — the held-out wall-Cp shape-L2 of
   ``sajben_validation.py`` — in principle?  We compute that ceiling by
   scoring the quasi-1D generator itself, every shock station it can
   produce, and the conditional-mean profile that a mean-squared regressor
   on the dataset converges to.
3. Which of the three routes to physics-bearing training data is adopted?
   (a) the NASA/WIND RANS solution shipped in the archive, decoded here;
   (b) a fresh 2-D RANS run; (c) train on the experiment with a declared
   split.

Outputs
-------
outputs/sajben_data_audit.md                 the finding (manifest source)
outputs/sajben_data_audit_ceiling.csv        per-station quasi-1D scores
outputs/plots/sajben_data_audit_wall_cp.png  experiment vs 1-D vs WIND

Usage::

    python scripts/validation/sajben_data_audit.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.le_pinn import (  # noqa: E402
    parse_sajben_experimental_data,
    parse_sajben_geometry,
)
from simulation.nozzle.wind_cff import load_wind_solution  # noqa: E402
from scripts.parse_sajben_cfd import (  # noqa: E402
    _build_supersonic_M,
    _mach_from_ar,
    _p_isen,
    _sutherland,
    _t_isen,
    parse_dat_configs,
    solve_fully_supersonic,
    solve_shock_at_station,
    solve_subsonic_unchoked,
)
from scripts.validation.sajben_validation import (  # noqa: E402
    _l2_relative,
    _normalise_01,
)

# ---------------------------------------------------------------------------
# Paths and the pre-registered gate
# ---------------------------------------------------------------------------
DATASET = REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt"
EXP_FILE = REPO_ROOT / "data" / "raw" / "data.Mach46.txt"
GEOM_FILE = REPO_ROOT / "data" / "raw" / "sajben.x.fmt"
NASA_DIR = REPO_ROOT / "data" / "raw" / "cfd_datasets" / "nasa"
WIND_CGD = NASA_DIR / "transdif01" / "sajben.cgd"
WIND_CFL = NASA_DIR / "transdif01" / "sajben.cfl"
OUT_MD = REPO_ROOT / "outputs" / "sajben_data_audit.md"
OUT_CSV = REPO_ROOT / "outputs" / "sajben_data_audit_ceiling.csv"
OUT_PNG = REPO_ROOT / "outputs" / "plots" / "sajben_data_audit_wall_cp.png"

GATE_PASS = 0.10          # docs/plan.md P4.3, fixed before any run
GATE_PARTIAL = 0.25
H_THROAT_M = 0.14435 * 0.3048   # 4.4014 cm, data.Mach46.txt header


# ---------------------------------------------------------------------------
# Metric: identical to sajben_validation.compute_wall_cp_errors, with the
# experimental x/H mapped by the length the experiment used (the throat
# height).  ``h_map`` is a parameter so the scorer's current mapping — the
# throat *radius*, i.e. H/2 — can be reported alongside.
# ---------------------------------------------------------------------------

def score_wall_cp(x_model_m: np.ndarray, cp_model: np.ndarray,
                  exp_wall: dict, x_thr_m: float, h_map_m: float) -> tuple[float, int]:
    x_exp = exp_wall["xh"] * h_map_m + x_thr_m
    m = (x_exp >= x_model_m.min()) & (x_exp <= x_model_m.max())
    if m.sum() < 2:
        return float("nan"), int(m.sum())
    cp_at_exp = np.interp(x_exp[m], x_model_m, cp_model)
    cp_exp = exp_wall["pp0"][m].astype(float)
    return _l2_relative(_normalise_01(cp_at_exp), _normalise_01(cp_exp)), int(m.sum())


def score_velocity_profiles(x_m: np.ndarray, y_m: np.ndarray, u: np.ndarray,
                            exp: dict, x_thr_m: float) -> dict:
    """Relative L2 of u(y) at the four LDV stations; y/H is in throat heights."""
    out = {}
    for lbl, vp in exp["vel_profiles"].items():
        i = int(np.argmin(np.abs(x_m[0] - (float(lbl) * H_THROAT_M + x_thr_m))))
        y_col, u_col = y_m[:, i], u[:, i]
        y_exp = vp["yh"].astype(float) * H_THROAT_M + y_col[0]
        u_exp = vp["u_ms"].astype(float)
        m = (y_exp >= y_col.min()) & (y_exp <= y_col.max())
        out[lbl] = _l2_relative(np.interp(y_exp[m], y_col, u_col), u_exp[m])
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# 1. Dataset content audit
# ---------------------------------------------------------------------------

def audit_dataset() -> dict:
    d = torch.load(DATASET, map_location="cpu", weights_only=False)
    X = d["inputs"].numpy().astype(np.float64)
    Y = d["targets"].numpy().astype(np.float64)
    W = d["sample_weights"].numpy().astype(np.float64)
    n, n_in = X.shape
    n_tgt = Y.shape[1]

    n_unique_in = len(np.unique(X, axis=0))
    ni, nj = 81, 51
    n_cases = n // (ni * nj)
    if n_cases * ni * nj != n:
        raise RuntimeError("dataset is not an integer number of 81x51 cases")
    X4 = X.reshape(n_cases, ni, nj, n_in)
    Y4 = Y.reshape(n_cases, ni, nj, n_tgt)
    # Row order is case-major, then i, then j (parse_sajben_cfd._pack_case)
    if not (np.ptp(X4[..., 0], axis=2).max() < 1e-9):
        raise RuntimeError("row order is not (case, i, j): x varies along j")

    # wall-normal variation of the primary targets within each (case, i)
    rel_span = {}
    for k, nm in enumerate(["rho", "u", "v", "P", "T"]):
        span = np.ptp(Y4[..., k], axis=2)
        scale = np.maximum(np.abs(Y4[..., k]).mean(axis=2), 1e-12)
        rel_span[nm] = float((span / scale).max())

    u_wall = Y4[:, :, 0, 1]
    u_mid = Y4[:, :, nj // 2, 1]
    wall_to_centre = u_wall / np.maximum(np.abs(u_mid), 1e-12)

    v = Y[:, 2]
    mu_eff = Y[:, 8]
    T = Y[:, 4]
    mu_suth = _sutherland(T)
    nan_cols = [int(k) for k in range(n_tgt) if np.isnan(Y[:, k]).all()]

    # spread of targets across cases at identical inputs
    P_spread = np.ptp(Y4[..., 3], axis=0) / np.maximum(np.abs(Y4[..., 3]).mean(axis=0), 1e-12)

    return {
        "n_rows": n, "n_inputs": n_in, "n_targets": n_tgt, "n_cases": n_cases,
        "unique_input_rows": n_unique_in,
        "repeats_per_input_row": n // n_unique_in,
        "constant_inputs": [nm for k, nm in enumerate(["x", "y", "A5", "A6", "P0", "T0"])
                            if len(np.unique(X[:, k])) == 1],
        "frac_v_nonzero": float((v != 0).mean()),
        "v_abs_max": float(np.abs(v).max()),
        "u_abs_max": float(np.abs(Y[:, 1]).max()),
        "wall_normal_rel_span": rel_span,
        "u_wall_over_u_centre_min": float(wall_to_centre.min()),
        "u_wall_over_u_centre_max": float(wall_to_centre.max()),
        "mu_eff_min": float(np.nanmin(mu_eff)), "mu_eff_max": float(np.nanmax(mu_eff)),
        "mu_eff_vs_sutherland_max_rel_dev": float(
            np.nanmax(np.abs(mu_eff - mu_suth) / mu_suth)),
        "nan_target_columns": nan_cols,
        "frac_input_rows_with_P_spread_gt_10pct": float((P_spread > 0.10).mean()),
        "P_spread_max": float(P_spread.max()),
        "weights_unique": sorted(set(np.unique(W).tolist())),
        "dataset_sha256": sha256(DATASET),
        "Y4": Y4, "X4": X4, "W4": W.reshape(n_cases, ni, nj),
    }


# ---------------------------------------------------------------------------
# 2. Quasi-1D analytic ceiling
# ---------------------------------------------------------------------------

def quasi_1d_ceiling(exp: dict, ds: dict) -> dict:
    geom = parse_sajben_geometry(str(GEOM_FILE))
    x_m = geom["x_m"]
    ni = geom["ni"]
    upper_y = geom["upper_wall_y_m"]
    lower_y = geom["lower_wall_y_m"]
    idx_throat = int(np.argmin(upper_y))
    h_vec = upper_y - lower_y
    h_throat = float(h_vec[idx_throat])
    AR_vec = np.maximum(h_vec / h_throat, 1.0)
    h_inlet_to_throat = float(h_vec[0]) / h_throat
    x_axial = x_m[:, 0]
    x_thr = float(x_axial[idx_throat])

    inlet = parse_dat_configs(NASA_DIR)
    P_s = inlet["P_s_psi"] * 6894.757
    fac = 1.0 + 0.2 * inlet["M"] ** 2
    P0 = P_s * fac ** 3.5
    T0 = inlet["T_s_R"] * (5.0 / 9.0) * fac

    M_sub_conv = _mach_from_ar(AR_vec, supersonic=False)
    M_sup = _build_supersonic_M(AR_vec, idx_throat)

    exp_exit_pp0 = float(np.mean(exp["top_wall"]["pp0"][-3:]))

    rows = []
    for k in range(idx_throat + 1, ni - 1):
        M_ax, P0_ax = solve_shock_at_station(AR_vec, M_sub_conv, M_sup, idx_throat, k, P0)
        pp0 = _p_isen(M_ax, 1.0) * P0_ax / P0
        l2_top, n_top = score_wall_cp(x_axial, pp0, exp["top_wall"], x_thr, H_THROAT_M)
        l2_bot, n_bot = score_wall_cp(x_axial, pp0, exp["bot_wall"], x_thr, H_THROAT_M)
        l2_top_half, _ = score_wall_cp(x_axial, pp0, exp["top_wall"], x_thr, H_THROAT_M / 2)
        l2_bot_half, _ = score_wall_cp(x_axial, pp0, exp["bot_wall"], x_thr, H_THROAT_M / 2)
        rows.append({
            "family": "A_shock", "station": k,
            "x_shock_over_H": (float(x_axial[k]) - x_thr) / H_THROAT_M,
            "M1": float(M_sup[k]), "exit_pp0": float(pp0[-1]),
            "l2_upper": l2_top, "l2_lower": l2_bot,
            "l2_upper_Hhalf": l2_top_half, "l2_lower_Hhalf": l2_bot_half,
            "pp0": pp0,
        })

    M_ax, P0_ax = solve_fully_supersonic(AR_vec, M_sub_conv, M_sup, idx_throat, P0)
    pp0 = _p_isen(M_ax, 1.0) * P0_ax / P0
    rows.append({
        "family": "B_supersonic", "station": -1, "x_shock_over_H": np.nan,
        "M1": float(M_ax[-1]), "exit_pp0": float(pp0[-1]),
        "l2_upper": score_wall_cp(x_axial, pp0, exp["top_wall"], x_thr, H_THROAT_M)[0],
        "l2_lower": score_wall_cp(x_axial, pp0, exp["bot_wall"], x_thr, H_THROAT_M)[0],
        "l2_upper_Hhalf": score_wall_cp(x_axial, pp0, exp["top_wall"], x_thr, H_THROAT_M / 2)[0],
        "l2_lower_Hhalf": score_wall_cp(x_axial, pp0, exp["bot_wall"], x_thr, H_THROAT_M / 2)[0],
        "pp0": pp0,
    })
    for M_in in np.linspace(0.20, 0.44, 5):
        M_ax, P0_ax = solve_subsonic_unchoked(AR_vec, h_inlet_to_throat, float(M_in), P0)
        pp0 = _p_isen(M_ax, 1.0) * P0_ax / P0
        rows.append({
            "family": "C_subsonic", "station": -1, "x_shock_over_H": np.nan,
            "M1": float(M_in), "exit_pp0": float(pp0[-1]),
            "l2_upper": score_wall_cp(x_axial, pp0, exp["top_wall"], x_thr, H_THROAT_M)[0],
            "l2_lower": score_wall_cp(x_axial, pp0, exp["bot_wall"], x_thr, H_THROAT_M)[0],
            "l2_upper_Hhalf": score_wall_cp(x_axial, pp0, exp["top_wall"], x_thr, H_THROAT_M / 2)[0],
            "l2_lower_Hhalf": score_wall_cp(x_axial, pp0, exp["bot_wall"], x_thr, H_THROAT_M / 2)[0],
            "pp0": pp0,
        })

    # Conditional mean of the dataset at each x (what MSE regression converges
    # to when every input row carries 31 different targets).  P is uniform in
    # j within a case, so the j=0 row is the wall value.
    Y4, W4 = ds["Y4"], ds["W4"]
    P_wall = Y4[:, :, 0, 3]                       # (cases, i)
    w = W4[:, :, 0]
    P_mean = (P_wall * w).sum(axis=0) / w.sum(axis=0)
    pp0_mean = P_mean / P0
    cond_mean = {
        "l2_upper": score_wall_cp(x_axial, pp0_mean, exp["top_wall"], x_thr, H_THROAT_M)[0],
        "l2_lower": score_wall_cp(x_axial, pp0_mean, exp["bot_wall"], x_thr, H_THROAT_M)[0],
        "pp0": pp0_mean,
    }

    shock_rows = [r for r in rows if r["family"] == "A_shock"]
    best = min(shock_rows, key=lambda r: 0.5 * (r["l2_upper"] + r["l2_lower"]))
    best_half = min(shock_rows, key=lambda r: 0.5 * (r["l2_upper_Hhalf"] + r["l2_lower_Hhalf"]))
    by_backpressure = min(shock_rows, key=lambda r: abs(r["exit_pp0"] - exp_exit_pp0))

    return {
        "rows": rows, "best": best, "best_Hhalf": best_half,
        "by_backpressure": by_backpressure, "cond_mean": cond_mean,
        "x_axial": x_axial, "x_thr": x_thr, "P0": P0, "T0": T0,
        "exp_exit_pp0": exp_exit_pp0, "n_shock_stations": len(shock_rows),
        "idx_throat": idx_throat,
    }


# ---------------------------------------------------------------------------
# 3. Route (a): the WIND RANS solution in the archive
# ---------------------------------------------------------------------------

def audit_wind(exp: dict) -> dict:
    sol = load_wind_solution(WIND_CGD, WIND_CFL)
    x_thr = sol.x_throat
    xu, yu = sol.upper_wall
    xl, yl = sol.lower_wall
    pp0_u = sol.p[-1] / sol.p0
    pp0_l = sol.p[0] / sol.p0
    l2_u, n_u = score_wall_cp(xu, pp0_u, exp["top_wall"], x_thr, H_THROAT_M)
    l2_l, n_l = score_wall_cp(xl, pp0_l, exp["bot_wall"], x_thr, H_THROAT_M)
    l2_u_half, _ = score_wall_cp(xu, pp0_u, exp["top_wall"], x_thr, H_THROAT_M / 2)
    l2_l_half, _ = score_wall_cp(xl, pp0_l, exp["bot_wall"], x_thr, H_THROAT_M / 2)
    vel = score_velocity_profiles(sol.x, sol.y, sol.u, exp, x_thr)
    jmid = sol.nj // 2
    return {
        "sol": sol, "l2_upper": l2_u, "l2_lower": l2_l, "n_upper": n_u, "n_lower": n_l,
        "l2_upper_Hhalf": l2_u_half, "l2_lower_Hhalf": l2_l_half,
        "velocity_l2": vel,
        "inlet_pp0": float(sol.p[jmid, 0] / sol.p0), "inlet_M": float(sol.mach[jmid, 0]),
        "exit_pp0": float(sol.p[jmid, -1] / sol.p0), "exit_M": float(sol.mach[jmid, -1]),
        "peak_M": float(sol.mach.max()),
        "wall_u_max": float(max(np.abs(sol.u[0]).max(), np.abs(sol.u[-1]).max())),
        "mut_over_mul_max": float((sol.mu_t / sol.mu_l).max()),
        "n_reverse_flow": int((sol.u < 0).sum()),
        "h_throat_m": sol.h_throat,
    }


# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------

def make_figure(exp: dict, ceil: dict, wind: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    x1d = (ceil["x_axial"] - ceil["x_thr"]) / H_THROAT_M
    sol = wind["sol"]
    xw_u = (sol.upper_wall[0] - sol.x_throat) / H_THROAT_M
    xw_l = (sol.lower_wall[0] - sol.x_throat) / H_THROAT_M

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, wall, xw, pp0_w in [
        (axes[0], "top_wall", xw_u, sol.p[-1] / sol.p0),
        (axes[1], "bot_wall", xw_l, sol.p[0] / sol.p0),
    ]:
        ax.plot(exp[wall]["xh"], exp[wall]["pp0"], "o", color="black", ms=4.5,
                mfc="white", mew=1.2, label="Experiment (Hsieh et al. 1987)", zorder=5)
        ax.plot(xw, pp0_w, "-", color="#0072B2", lw=2.0, label="WIND RANS (route a)", zorder=4)
        ax.plot(x1d, ceil["best"]["pp0"], "--", color="#D55E00", lw=1.6,
                label=f"Quasi-1D, best station (x/H={ceil['best']['x_shock_over_H']:.2f})")
        ax.plot(x1d, ceil["cond_mean"]["pp0"], ":", color="#009E73", lw=2.0,
                label="Dataset conditional mean (31 cases)")
        ax.set_xlabel("x / H  (throat height, throat at 0)")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlim(-4.3, 8.9)
    axes[0].set_ylabel("P / P$_0$")
    axes[0].set_title("Upper wall", loc="left", fontsize=10)
    axes[1].set_title("Lower wall", loc="left", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Sajben weak-shock case — what each data source reproduces of the wall pressure",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------

def write_csv(ceil: dict) -> None:
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["family", "station", "x_shock_over_H", "M1", "exit_pp0",
            "l2_upper", "l2_lower", "l2_upper_Hhalf", "l2_lower_Hhalf"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ceil["rows"]:
            w.writerow({c: r[c] for c in cols})


def band(v: float) -> str:
    if v < GATE_PASS:
        return "below 0.10 (gate band: pass)"
    if v < GATE_PARTIAL:
        return "0.10–0.25 (gate band: partial)"
    return "above 0.25 (gate band: fail)"


def write_report(ds: dict, ceil: dict, wind: dict, exp: dict) -> None:
    b, bh, bp, cm = ceil["best"], ceil["best_Hhalf"], ceil["by_backpressure"], ceil["cond_mean"]
    sol = wind["sol"]
    span = ds["wall_normal_rel_span"]
    vel_lines = "\n".join(
        f"| x/H = {lbl} | {v:.3f} |" for lbl, v in sorted(wind["velocity_l2"].items(), key=lambda kv: float(kv[0]))
    )
    L = []
    L.append(f"# Sajben data-reality audit (P4.1) — {date.today().isoformat()}\n")
    L.append(f"Generated by `scripts/validation/sajben_data_audit.py` at git `{git_sha()}`. "
             f"Gate metric: held-out wall-Cp shape-L2 from `sajben_validation.py` "
             f"(`_normalise_01` both profiles to [0,1], then `_l2_relative`), pre-registered "
             f"bands < {GATE_PASS:.2f} pass / {GATE_PASS:.2f}–{GATE_PARTIAL:.2f} partial / > {GATE_PARTIAL:.2f} fail "
             f"(`docs/plan.md`, P4.3).\n")

    ceiling = 0.5 * (b["l2_upper"] + b["l2_lower"])
    ceiling_above_gate = ceiling > GATE_PASS
    wind_worst = max(wind["l2_upper"], wind["l2_lower"])
    wind_under_gate = wind_worst < GATE_PASS

    L.append("## Verdict\n")
    if ceiling_above_gate and wind_under_gate:
        L.append(
            f"**GO, on route (a).** The existing training set cannot reach the gate for two "
            f"independent reasons (§1), and its analytic ceiling on the gate metric is "
            f"**{ceiling:.3f}** (§2) — above {GATE_PASS:.2f} for the best shock station the "
            f"generator can produce, so the recorded 0.714 / 0.723 was a domain-mismatch artifact, "
            f"not a training failure. The NASA archive already ships a converged Spalart-Allmaras "
            f"RANS solution of the weak-shock case on the same 81×51 grid; it decodes without new "
            f"dependencies and scores **{wind['l2_upper']:.3f} / {wind['l2_lower']:.3f}** "
            f"(upper / lower wall) against the experiment on the gate metric (§3). That is the "
            f"training data for P4.3.\n")
    elif ceiling_above_gate:
        L.append(
            f"**NO-GO on the archive solution; escalate.** The quasi-1D ceiling is **{ceiling:.3f}** "
            f"(> {GATE_PASS:.2f}), so the existing data cannot pass, but the WIND solution scores "
            f"{wind['l2_upper']:.3f} / {wind['l2_lower']:.3f} — not under the gate either. Route (b) or a "
            f"re-opened scope decision is required before P4.3 (`docs/plan.md`, P4.1 escalation).\n")
    else:
        L.append(
            f"**The quasi-1D ceiling is {ceiling:.3f}, under the gate.** The premise of F1 is not "
            f"confirmed by this metric; the data-content findings in §1 still stand, but the "
            f"'cannot pass on this data' claim must not be made. Re-open the route decision with the user.\n")

    L.append("## 1. What `master_shock_dataset.pt` contains\n")
    L.append(f"`{DATASET.relative_to(REPO_ROOT)}` (sha256 `{ds['dataset_sha256'][:16]}…`): "
             f"{ds['n_rows']:,} rows = {ds['n_cases']} cases × 81 × 51, "
             f"{ds['n_inputs']} inputs / {ds['n_targets']} targets, sample weights {ds['weights_unique']}.\n")
    L.append("| Question | Measured | Meaning |\n|---|---|---|")
    L.append(f"| Rows with v ≠ 0 | {100*ds['frac_v_nonzero']:.1f} % (max \\|v\\| {ds['v_abs_max']:.1f} m/s vs max \\|u\\| {ds['u_abs_max']:.0f} m/s) | "
             f"v is `_estimate_transverse_velocity`: inviscid streamline deflection from the wall slope, not a solved momentum field |")
    L.append(f"| Wall-normal variation of ρ, u, P, T within a station | "
             f"ρ {span['rho']:.1e}, u {span['u']:.1e}, P {span['P']:.1e}, T {span['T']:.1e} (relative span, max over all stations) | "
             f"zero by construction — `_pack_case` broadcasts the 1-D axial solution with `np.repeat` over all 51 wall-normal points |")
    L.append(f"| u(wall) / u(centreline) | {ds['u_wall_over_u_centre_min']:.3f} – {ds['u_wall_over_u_centre_max']:.3f} | "
             f"no row encodes a boundary layer; the wall velocity equals the core velocity everywhere |")
    L.append(f"| μ_eff | {ds['mu_eff_min']:.2e} – {ds['mu_eff_max']:.2e} Pa·s; max deviation from Sutherland(T) {100*ds['mu_eff_vs_sutherland_max_rel_dev']:.2g} % | "
             f"molecular viscosity only — no eddy viscosity, hence no turbulence model |")
    L.append(f"| Target columns all-NaN | {ds['nan_target_columns']} | Reynolds-stress slots; masked out of the data loss, docstring calls them zero-padded (P4.7 fix) |")
    L.append(f"| Inputs that are a single constant | {ds['constant_inputs']} | **the flow case is not identifiable from the inputs** |")
    L.append(f"| Unique input rows | {ds['unique_input_rows']:,} of {ds['n_rows']:,} — each input row repeats {ds['repeats_per_input_row']}× with different targets | "
             f"the dataset is a one-to-many map |")
    L.append(f"| Input rows whose P target varies > 10 % across cases | {100*ds['frac_input_rows_with_P_spread_gt_10pct']:.0f} % (max spread {100*ds['P_spread_max']:.0f} %) | "
             f"a mean-squared regressor converges to the case-average, i.e. a smeared ramp, not a shock |\n")
    L.append(
        "**Finding 1a — no physics.** Every row is a quasi-1D inviscid, adiabatic, laminar-viscosity "
        "state broadcast across the channel. The Sajben benchmark measures shock/boundary-layer "
        "interaction: a curved-wall suction peak, a λ-shock foot smeared over ~1 H, asymmetric upper/lower "
        "wall recovery, and no-slip velocity profiles. None of that is in the training signal.\n")
    L.append(
        "**Finding 1b — no conditioning variable.** `A5, A6, P0, T0` are identical for all 31 cases, "
        "so the back pressure that sets the shock position is not an input. The same (x, y) carries "
        "31 different pressures. This is not fixable by epochs, architecture, or loss weights: the target "
        "is not a function of the inputs. Any model trained on this set is fitting the average over shock "
        "positions (green dotted curve in the figure).\n")

    L.append("## 2. Analytic ceiling of the quasi-1D generator on the gate metric\n")
    L.append(f"Experimental conditions from `sajben.dat.*`: P₀ = {ceil['P0']:.0f} Pa, T₀ = {ceil['T0']:.1f} K, "
             f"M_in = 0.46; experimental exit P/P₀ = {ceil['exp_exit_pp0']:.3f}. All "
             f"{ceil['n_shock_stations']} diverging-section shock stations were scored "
             f"(`outputs/sajben_data_audit_ceiling.csv`). A 1-D solution has identical upper and lower "
             f"wall pressure, so it cannot reproduce the measured wall asymmetry at all.\n")
    L.append("| Quasi-1D candidate | Upper L2 | Lower L2 | Mean | Band |\n|---|---|---|---|---|")
    L.append(f"| Best of all {ceil['n_shock_stations']} shock stations (x_shock/H = {b['x_shock_over_H']:.2f}, M₁ = {b['M1']:.3f}) — **the ceiling** | "
             f"{b['l2_upper']:.3f} | {b['l2_lower']:.3f} | {0.5*(b['l2_upper']+b['l2_lower']):.3f} | {band(0.5*(b['l2_upper']+b['l2_lower']))} |")
    L.append(f"| Station selected by the experimental back pressure (exit P/P₀ = {bp['exit_pp0']:.3f}, x_shock/H = {bp['x_shock_over_H']:.2f}) | "
             f"{bp['l2_upper']:.3f} | {bp['l2_lower']:.3f} | {0.5*(bp['l2_upper']+bp['l2_lower']):.3f} | {band(0.5*(bp['l2_upper']+bp['l2_lower']))} |")
    L.append(f"| Dataset conditional mean (what MSE regression on the one-to-many set converges to) | "
             f"{cm['l2_upper']:.3f} | {cm['l2_lower']:.3f} | {0.5*(cm['l2_upper']+cm['l2_lower']):.3f} | {band(0.5*(cm['l2_upper']+cm['l2_lower']))} |")
    L.append(f"| Best shock station under the scorer's current x-mapping (H/2, see §4) | "
             f"{bh['l2_upper_Hhalf']:.3f} | {bh['l2_lower_Hhalf']:.3f} | {0.5*(bh['l2_upper_Hhalf']+bh['l2_lower_Hhalf']):.3f} | — |\n")
    if ceiling_above_gate:
        f2_claim = (f"**No model trained on this data can pass the gate; this is computed, not argued.** "
                    f"The recorded 0.714 / 0.723 for `le_pinn_sajben.pt` is therefore a domain-mismatch artifact.")
    else:
        f2_claim = (f"The ceiling is under the gate on this metric, so the 'cannot pass' claim is **not** "
                    f"supported; only the content findings in §1 stand.")
    L.append(
        f"**Finding 2.** The ceiling is {ceiling:.3f} ({'above' if ceiling_above_gate else 'below'} the "
        f"{GATE_PASS:.2f} gate) for the single most favourable member of the training family; the member "
        f"the physics actually selects (matching the measured exit pressure) scores "
        f"{0.5*(bp['l2_upper']+bp['l2_lower']):.3f}, and the profile a regressor would converge to scores "
        f"{0.5*(cm['l2_upper']+cm['l2_lower']):.3f}. {f2_claim} What a quasi-1D inviscid solution *can* "
        f"reproduce, at best: the isentropic run-up to the throat, a sharp jump at one chosen station, and "
        f"the subsonic recovery level. What it cannot: the upper/lower asymmetry, the smeared shock foot, "
        f"the post-shock plateau, and any velocity profile. Note that the inviscid solution that matches "
        f"the measured exit pressure puts its shock at x/H = {bp['x_shock_over_H']:.2f}, downstream of the "
        f"measured shock near x/H ≈ 1.4: the boundary-layer displacement and total-pressure loss that set "
        f"the real shock position are exactly the physics the generator lacks.\n")

    L.append("## 3. Route (a) verified — the WIND RANS solution in the archive\n")
    L.append(
        f"`{WIND_CFL.relative_to(REPO_ROOT)}` is not a CGNS file that needs pyCGNS; it is a 1997 ADF "
        f"database (the container under CGNS v1/v2) written by WIND 1.144 in NPARC Study #1 (C. Towne). "
        f"`simulation/nozzle/wind_cff.py` (new, NumPy only) decodes it. The run listing `sajben.lis.Z` "
        f"confirms: Spalart-Allmaras, adiabatic no-slip walls, uniform M = 0.46 inflow, exit static pressure "
        f"ramped over 20 restarts to the experimental 16.055 psi, 20 000 iterations, final RMS residual "
        f"~3×10⁻⁶ in a low-amplitude limit cycle (this case is mildly self-excited, per Bogar 1986). The "
        f"file's modification stamp ({sol.provenance['cfl_modified']}) matches the listing's end time.\n")
    L.append("| Check | WIND | Experiment / expected |\n|---|---|---|")
    L.append(f"| Grid | {sol.ni} × {sol.nj}, throat height {100*wind['h_throat_m']:.3f} cm | 81 × 51 (`sajben.x.fmt`), 4.407 cm |")
    L.append(f"| Inlet P/P₀, M | {wind['inlet_pp0']:.3f}, {wind['inlet_M']:.3f} | 0.864 (first tap), 0.46 |")
    L.append(f"| Exit P/P₀, M | {wind['exit_pp0']:.3f}, {wind['exit_M']:.3f} | {ceil['exp_exit_pp0']:.3f} (last taps), ≈0.51 (NPARC page) |")
    L.append(f"| Peak Mach | {wind['peak_M']:.3f} | 'just under 1.3' (NPARC page) |")
    L.append(f"| Wall velocity (no-slip) | max \\|u\\| = {wind['wall_u_max']:.1e} m/s | 0 |")
    L.append(f"| Eddy / laminar viscosity | up to {wind['mut_over_mul_max']:.0f}× | turbulent boundary layers present |")
    L.append(f"| Reverse-flow points | {wind['n_reverse_flow']} of {sol.ni*sol.nj} | weak-shock case: no shock-induced separation |")
    L.append(f"| **Wall-Cp shape-L2 (gate metric)** | **upper {wind['l2_upper']:.3f} ({wind['n_upper']} taps), lower {wind['l2_lower']:.3f} ({wind['n_lower']} taps)** | gate < {GATE_PASS:.2f} |")
    L.append(f"| Same, under the scorer's current H/2 mapping | upper {wind['l2_upper_Hhalf']:.3f}, lower {wind['l2_lower_Hhalf']:.3f} | (see §4) |\n")
    L.append("Axial-velocity profiles, relative L2 against the LDV data (y/H in throat heights from the lower wall):\n")
    L.append("| Station | L2 |\n|---|---|")
    L.append(vel_lines + "\n")
    L.append(
        f"**Finding 3.** The archive solution contains the physics the benchmark measures and, scored "
        f"exactly as a checkpoint would be, sits at {wind['l2_upper']:.3f} / {wind['l2_lower']:.3f} — "
        f"{'under' if wind_under_gate else 'NOT under'} the gate. This is also the honest ceiling for "
        f"route (a): a surrogate that reproduced WIND perfectly would score this, and the margin to "
        f"{GATE_PASS:.2f} is {GATE_PASS - wind_worst:+.3f}. The pre-registered bands stand as written; this "
        f"audit records that the pass band is reachable only by a near-perfect surrogate of the CFD, which "
        f"is what a PINN on a single case should be.\n")

    L.append("## 4. A scorer defect found while reproducing the metric\n")
    L.append(
        "`compute_wall_cp_errors` and `compute_velocity_profile_errors` in `sajben_validation.py` set "
        "`H_m = sqrt(A5/π)`. With `A5 = π (H/2)²` from `build_sajben_grid`, that is the throat *radius* — "
        "half the throat height. Both functions then map the experimental x/H and y/H with it, so every "
        "experimental coordinate is placed at half its true distance from the throat (the measured shock at "
        "x/H ≈ 1.4 is compared against the model at x/H ≈ 0.7). The velocity profiles are compressed the "
        "same way. Every Sajben number in the repo — 0.714 / 0.723 and the fine-tuned 0.805 / 1.091 — was "
        "scored under this mapping. The fix is one line per function (`H_m = 2·sqrt(A5/π)`, or pass the "
        "geometry's `H_m` through) and lands in P4.2 before any checkpoint is re-scored, with old- and "
        "new-metric numbers reported side by side. It does not change the conclusions above: the table in "
        "§2 shows the quasi-1D ceiling under both mappings, and §3 shows WIND under both.\n")

    L.append("## 5. Routes and decision\n")
    L.append("| Route | What it gives | Cost | External-validation claim | Verdict |\n|---|---|---|---|---|")
    L.append(
        "| **(a) NASA/WIND archive solution** | one converged S-A RANS field, 4 131 points, ρ u v P T μ_l μ_t, "
        "same grid, weak-shock case only | done — `wind_cff.py`, no dependency | **kept**: trained on CFD, "
        "scored on an independent experiment | **adopted** |")
    L.append(
        "| (b) Fresh 2-D RANS (SU2) | a parametric family over back pressure; reusable | SU2 install + mesh + "
        "case setup + convergence study; the vendored `nozzle_flow_cfd-main` is a generic AI-generated SU2 "
        "desktop GUI with de Laval templates and self-declared minimal testing, with no Sajben case | kept | "
        "not needed for the gate; the natural follow-up if a "
        "back-pressure-parametric surrogate is wanted later |")
    L.append(
        "| (c) Train on the experiment with a declared split | 71 wall taps + 116 LDV points | trivial | "
        "**forfeited** — held-out set is small and spatially correlated | rejected while (a) exists |\n")
    if not wind_under_gate:
        L.append("**Decision: escalate.** Route (a) does not clear the gate on its own data; see Verdict.\n")
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text("\n".join(L))
        return
    L.append(
        "**Decision: route (a).** Justification: it is the only route that is both already in the repo and "
        "physics-complete; it keeps the external-validation language because the experiment never enters "
        "training; and it makes the P4.3 leakage guard trivial by construction — a checkpoint's recorded "
        "training source is the WIND file hash, and the scorer refuses any checkpoint whose provenance names "
        "`data.Mach46.txt`.\n")
    L.append("**What route (a) changes for P4.3, recorded now:**\n")
    L.append(
        "- The model becomes a single-condition surrogate (weak-shock case, one back pressure). The preprint "
        "must say so; it is not a parametric nozzle model.\n"
        "- Training inputs must not repeat the current defect: with one case, `(x, y)` alone identify the "
        "target, and `A5, A6, P0, T0` are constants. Either drop them or keep them as documented constants.\n"
        "- The RANS residual in `_safe_physics_loss` overwrites `μ_eff` with Sutherland molecular viscosity "
        "(`le_pinn.py` ~L1469), so on RANS data the physics term is a laminar residual that contradicts the "
        "data in the boundary layer. P4.3 must feed `μ_l + μ_t` from the WIND field (or drop the viscous "
        "term) before the physics weight is turned on.\n"
        "- Reynolds-stress target columns 5–7 stay zero (not NaN) unless derived from μ_t via Boussinesq.\n"
        "- The inflow column (i = 0) carries the uniform-inflow condition imposed on no-slip walls, which "
        "produces a wall-pressure spike at the two corner points (visible at x/H ≈ −4 in the figure). The "
        "experimental taps begin just downstream of it, so the gate metric never samples it; the P4.3 "
        "training set should either exclude i = 0 or document that it is a boundary artifact.\n"
        "- The strong-shock case is not in the archive; nothing here supports claims about it.\n")

    L.append("## Artifacts\n")
    L.append(f"- `{OUT_MD.relative_to(REPO_ROOT)}` — this finding\n"
             f"- `{OUT_CSV.relative_to(REPO_ROOT)}` — quasi-1D sweep, all stations, both x-mappings\n"
             f"- `{OUT_PNG.relative_to(REPO_ROOT)}` — experiment vs WIND vs best quasi-1D vs conditional mean\n"
             f"- WIND source: `{WIND_CFL.relative_to(REPO_ROOT)}` sha256 `{sol.provenance['cfl_sha256']}`; "
             f"grid `{WIND_CGD.relative_to(REPO_ROOT)}` sha256 `{sol.provenance['cgd_sha256']}`\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L))


# ---------------------------------------------------------------------------

def main() -> dict:
    print("=" * 72)
    print("P4.1  SAJBEN DATA-REALITY AUDIT")
    print("=" * 72)
    for f in (DATASET, EXP_FILE, GEOM_FILE, WIND_CGD, WIND_CFL):
        if not f.exists():
            raise FileNotFoundError(f)
    exp = parse_sajben_experimental_data(str(EXP_FILE))

    print("\n[1/4] Dataset content ...")
    ds = audit_dataset()
    print(f"  {ds['n_rows']:,} rows, {ds['n_cases']} cases; unique input rows {ds['unique_input_rows']:,} "
          f"(each repeats {ds['repeats_per_input_row']}x); constant inputs {ds['constant_inputs']}")
    print(f"  v!=0 on {100*ds['frac_v_nonzero']:.1f}% rows; wall-normal span of u {ds['wall_normal_rel_span']['u']:.1e}; "
          f"u_wall/u_centre in [{ds['u_wall_over_u_centre_min']:.3f}, {ds['u_wall_over_u_centre_max']:.3f}]")
    print(f"  mu_eff = Sutherland to {100*ds['mu_eff_vs_sutherland_max_rel_dev']:.2g}%; NaN target cols {ds['nan_target_columns']}")

    print("\n[2/4] Quasi-1D analytic ceiling ...")
    ceil = quasi_1d_ceiling(exp, ds)
    b, bp, cm = ceil["best"], ceil["by_backpressure"], ceil["cond_mean"]
    print(f"  best station   : x/H={b['x_shock_over_H']:.2f}  L2 upper {b['l2_upper']:.3f} lower {b['l2_lower']:.3f}")
    print(f"  by back-press. : x/H={bp['x_shock_over_H']:.2f}  L2 upper {bp['l2_upper']:.3f} lower {bp['l2_lower']:.3f}")
    print(f"  cond. mean     :            L2 upper {cm['l2_upper']:.3f} lower {cm['l2_lower']:.3f}")

    print("\n[3/4] Route (a): WIND archive solution ...")
    wind = audit_wind(exp)
    print(f"  inlet P/P0 {wind['inlet_pp0']:.3f} M {wind['inlet_M']:.3f}; exit P/P0 {wind['exit_pp0']:.3f}; "
          f"peak M {wind['peak_M']:.3f}; mut/mul max {wind['mut_over_mul_max']:.0f}")
    print(f"  wall-Cp shape-L2: upper {wind['l2_upper']:.3f}  lower {wind['l2_lower']:.3f}   (gate < {GATE_PASS})")
    for lbl, v in sorted(wind["velocity_l2"].items(), key=lambda kv: float(kv[0])):
        print(f"  u-profile x/H={lbl}: L2 {v:.3f}")

    print("\n[4/4] Writing artifacts ...")
    write_csv(ceil)
    make_figure(exp, ceil, wind)
    write_report(ds, ceil, wind, exp)
    print(f"  {OUT_MD.relative_to(REPO_ROOT)}\n  {OUT_CSV.relative_to(REPO_ROOT)}\n  {OUT_PNG.relative_to(REPO_ROOT)}")

    verdict = "GO (route a)" if max(wind["l2_upper"], wind["l2_lower"]) < GATE_PASS else "NO-GO on route (a)"
    print(f"\nVERDICT: {verdict}; quasi-1D ceiling {0.5*(b['l2_upper']+b['l2_lower']):.3f} "
          f"{'>' if 0.5*(b['l2_upper']+b['l2_lower']) > GATE_PASS else '<'} {GATE_PASS}")
    return {"dataset": ds, "ceiling": ceil, "wind": wind, "verdict": verdict}


if __name__ == "__main__":
    main()
