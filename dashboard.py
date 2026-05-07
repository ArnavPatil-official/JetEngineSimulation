"""
Jet Engine Interactive Streamlit Dashboard.

Allows interactive exploration of engine performance as a function of
fuel blend (≥50% Jet-A1 per ASTM D7566) and operating conditions.

Visualizations:
  - 3D pyvista engine cross-section colored by physical quantity (via stpyvista)
  - 2D matplotlib heatmap fallback (always available)
  - Switchable quantities: Temperature, Pressure, Mach, Density, Velocity
"""

import sys
import warnings
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import colormaps

import streamlit as st
import torch

warnings.filterwarnings("ignore")

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jet Engine Simulator",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-theme CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Engine loading (cached — Cantera + PINN load once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading engine (Cantera + PINN) — one-time setup…")
def load_engine():
    from integrated_engine import IntegratedTurbofanEngine
    return IntegratedTurbofanEngine(
        mechanism_profile="blends",
        creck_mechanism_path=str(ROOT / "data/creck_c1c16_full.yaml"),
        turbine_pinn_path=str(ROOT / "models/turbine_pinn.pt"),
        nozzle_pinn_path=str(ROOT / "models/nozzle_pinn.pt"),
        icao_data_path=str(ROOT / "data/icao_engine_data.csv"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Fuel blending
# ─────────────────────────────────────────────────────────────────────────────

# Species composition for each pure fuel (as fractions; Cantera normalizes)
_FUEL_SPECIES = {
    "Jet-A1":   {"NC12H26": 1.00},
    "HEFA-SPK": {"NC12H26": 0.85, "IC8H18": 0.15},
    "FT-SPK":   {"NC12H26": 0.50, "NC10H22": 0.35, "IC8H18": 0.15},
    "ATJ-SPK":  {"IC8H18": 0.80, "NC12H26": 0.20},
}

# Approximate lifecycle carbon factors (relative to Jet-A1 = 1.0)
_LCA_FACTORS = {
    "Jet-A1": 1.00,
    "HEFA-SPK": 0.70,
    "FT-SPK": 0.65,
    "ATJ-SPK": 0.60,
}


def make_local_blend(jet_pct: float, hefa_pct: float, ft_pct: float, atj_pct: float):
    """
    Build a LocalFuelBlend (as required by run_full_cycle) and compute a
    weighted LCA factor from the given percentage breakdown.
    """
    from integrated_engine import LocalFuelBlend

    total = jet_pct + hefa_pct + ft_pct + atj_pct
    if total <= 0:
        raise ValueError("Total fuel percentage must be > 0")

    fracs = {
        "Jet-A1":   jet_pct  / total,
        "HEFA-SPK": hefa_pct / total,
        "FT-SPK":   ft_pct   / total,
        "ATJ-SPK":  atj_pct  / total,
    }

    # Combine species weighted by blend fractions
    combined: dict[str, float] = {}
    for fuel_name, weight in fracs.items():
        if weight <= 0:
            continue
        for sp, val in _FUEL_SPECIES[fuel_name].items():
            combined[sp] = combined.get(sp, 0.0) + weight * val

    # Normalise so fractions sum to 1
    sp_total = sum(combined.values())
    combined = {sp: v / sp_total for sp, v in combined.items()}

    lca = sum(fracs[fn] * _LCA_FACTORS[fn] for fn in fracs)

    name = f"SAF(J{jet_pct:.0f}/H{hefa_pct:.0f}/FT{ft_pct:.0f}/ATJ{atj_pct:.0f})"
    return LocalFuelBlend(name, combined), lca


# ─────────────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(engine, jet_pct, hefa_pct, ft_pct, atj_pct, phi, efficiency):
    fuel_blend, lca = make_local_blend(jet_pct, hefa_pct, ft_pct, atj_pct)
    result = engine.run_full_cycle(
        fuel_blend=fuel_blend,
        phi=phi,
        combustor_efficiency=efficiency,
        lca_factor=lca,
    )
    return result, lca


def _capture_call(func, *args, **kwargs):
    """Run a function while capturing stdout logs for optional display."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


@st.cache_resource(show_spinner=False)
def load_le_pinn_checkpoint(checkpoint_path: str):
    """Load LE-PINN model + normalizers from checkpoint."""
    from simulation.nozzle.le_pinn import LE_PINN, MinMaxNormalizer

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = LE_PINN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_norm = None
    in_min = ckpt.get("input_norm_min")
    in_max = ckpt.get("input_norm_max")
    if in_min is not None and in_max is not None:
        input_norm = MinMaxNormalizer()
        input_norm.data_min = in_min.float().cpu()
        input_norm.data_max = in_max.float().cpu()

    output_norm = None
    out_min = ckpt.get("output_norm_min")
    out_max = ckpt.get("output_norm_max")
    if out_min is not None and out_max is not None:
        output_norm = MinMaxNormalizer()
        output_norm.data_min = out_min.float().cpu()
        output_norm.data_max = out_max.float().cpu()

    return model, input_norm, output_norm


_LE_PINN_INPUT_NAMES = ["x", "y", "A5", "A6", "P_in", "T_in"]
_LE_PINN_CORE_VARS = ["rho", "u", "v", "P", "T"]


def _select_default_le_pinn_checkpoint(model_paths: list[Path]) -> int:
    """Return index of recommended default checkpoint from discovered paths."""
    if not model_paths:
        return 0
    # Prefer baseline model first; fallback to any engine-tagged variant.
    preferred = ["le_pinn.pt", "le_pinn_engine.pt", "le_pinn_unified.pt"]
    for name in preferred:
        for i, p in enumerate(model_paths):
            if p.name == name:
                return i
    return 0


def _solve_area_mach(area_ratio: float, gamma: float, supersonic: bool) -> float:
    """Solve isentropic area-Mach relation for a given area ratio."""
    ar = max(float(area_ratio), 1e-6)
    g = float(gamma)
    gp1 = g + 1.0
    gm1 = g - 1.0
    m = 1.8 if supersonic else 0.35
    m_lo, m_hi = (1.0001, 6.0) if supersonic else (0.02, 0.999)

    for _ in range(80):
        m2 = m * m
        term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * m2)
        f = (1.0 / m) * (term ** (0.5 * gp1 / gm1)) - ar
        dm = 1e-6
        mp = min(max(m + dm, m_lo), m_hi)
        mp2 = mp * mp
        term_p = (2.0 / gp1) * (1.0 + 0.5 * gm1 * mp2)
        fp = (1.0 / mp) * (term_p ** (0.5 * gp1 / gm1)) - ar
        df = (fp - f) / (mp - m + 1e-12)
        if abs(df) < 1e-12:
            break
        m = m - f / df
        m = min(max(m, m_lo), m_hi)
    return float(m)


def _isentropic_nozzle_fallback(
    a5: float,
    a6: float,
    p_in: float,
    t_in: float,
    gamma: float = 1.33,
    r_gas: float = 287.0,
) -> dict[str, float]:
    """
    Physically consistent fallback using quasi-1D isentropic expansion.
    Interprets (P_in, T_in) as stagnation conditions.
    """
    ar = max(float(a6) / max(float(a5), 1e-8), 1e-6)
    supersonic = ar > 1.02
    mach = _solve_area_mach(ar, gamma=gamma, supersonic=supersonic)

    t = float(t_in) / (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
    p = float(p_in) * (t / max(float(t_in), 1e-8)) ** (gamma / (gamma - 1.0))
    rho = p / (r_gas * max(t, 1e-6))
    u = mach * np.sqrt(max(gamma * r_gas * t, 1e-8))

    # Dynamic viscosity via Sutherland's law (same constants as LE-PINN module)
    c1 = 1.458e-6
    s = 110.4
    mu = c1 * (t ** 1.5) / (t + s)

    return {
        "rho": float(rho),
        "u": float(u),
        "v": 0.0,
        "P": float(p),
        "T": float(t),
        "UU": 0.0,
        "VV": 0.0,
        "UV": 0.0,
        "mu_eff": float(mu),
    }


def _is_implausible_prediction(pred: dict[str, float], p_in: float, t_in: float) -> bool:
    """Detect obviously non-physical LE-PINN outputs for nozzle conditions."""
    if any((not np.isfinite(pred[k])) for k in ("rho", "u", "P", "T", "mu_eff")):
        return True
    t = float(pred["T"])
    p = float(pred["P"])
    u = float(pred["u"])
    rho = float(pred["rho"])
    if t < max(80.0, 0.18 * float(t_in)) or t > 1.2 * float(t_in):
        return True
    if p < max(5_000.0, 0.03 * float(p_in)) or p > 1.2 * float(p_in):
        return True
    if u < 0.0 or u > 2_500.0:
        return True
    if rho <= 1e-8 or rho > 20.0:
        return True
    return False


def _sanitize_le_pinn_inputs(
    input_norm,
    x: float,
    y: float,
    a5: float,
    a6: float,
    p_in: float,
    t_in: float,
) -> tuple[np.ndarray, list[str]]:
    """
    Clip/lock LE-PINN inputs to checkpoint training domain.

    For dimensions that were constant during training (min == max), the value
    is locked to the training constant to avoid invalid normalization blow-up.
    """
    vals = np.array([x, y, a5, a6, p_in, t_in], dtype=np.float32)
    notes: list[str] = []

    if input_norm is None or input_norm.data_min is None or input_norm.data_max is None:
        return vals, notes

    mins = input_norm.data_min.detach().cpu().numpy().astype(np.float32)
    maxs = input_norm.data_max.detach().cpu().numpy().astype(np.float32)

    for i, name in enumerate(_LE_PINN_INPUT_NAMES):
        mn = float(mins[i])
        mx = float(maxs[i])
        v = float(vals[i])
        if abs(mx - mn) < 1e-8:
            if abs(v - mn) > max(1e-6, 1e-6 * abs(mn)):
                notes.append(
                    f"{name}: locked to training constant {mn:.6g} (requested {v:.6g})"
                )
            vals[i] = mn
            continue
        v_clip = float(np.clip(v, mn, mx))
        if abs(v_clip - v) > max(1e-9, 1e-9 * abs(v)):
            notes.append(
                f"{name}: clipped from {v:.6g} to {v_clip:.6g} within [{mn:.6g}, {mx:.6g}]"
            )
        vals[i] = v_clip

    return vals, notes


def run_le_pinn_inference(
    checkpoint_path: str,
    x: float,
    y: float,
    a5: float,
    a6: float,
    p_in: float,
    t_in: float,
    wall_distance: float,
    return_meta: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    """Run single-point LE-PINN inference for interactive exploration."""
    model, input_norm, output_norm = load_le_pinn_checkpoint(checkpoint_path)

    safe_vals, adjustments = _sanitize_le_pinn_inputs(
        input_norm, x, y, a5, a6, p_in, t_in
    )
    inp = torch.tensor([safe_vals], dtype=torch.float32)
    wall = torch.tensor([[wall_distance]], dtype=torch.float32)
    model_in = input_norm.transform(inp) if input_norm is not None else inp

    with torch.no_grad():
        pred = model(model_in, wall)

    if output_norm is not None and output_norm.data_min is not None:
        n_cols = min(int(output_norm.data_min.shape[0]), int(pred.shape[1]))
        pred_out = pred.clone()
        pred_out[:, :n_cols] = output_norm.inverse_transform(pred[:, :n_cols])
    else:
        pred_out = pred

    vals = pred_out.squeeze(0).cpu().numpy().tolist()
    keys = ["rho", "u", "v", "P", "T", "UU", "VV", "UV", "mu_eff"]
    pred_dict = {k: float(v) for k, v in zip(keys, vals)}
    # Basic positivity clamps first.
    pred_dict["rho"] = max(pred_dict["rho"], 1e-9)
    pred_dict["P"] = max(pred_dict["P"], 1e-3)
    pred_dict["T"] = max(pred_dict["T"], 1.0)
    pred_dict["mu_eff"] = max(pred_dict["mu_eff"], 1e-12)

    used_fallback = False
    if _is_implausible_prediction(pred_dict, p_in=float(safe_vals[4]), t_in=float(safe_vals[5])):
        pred_dict = _isentropic_nozzle_fallback(
            a5=float(safe_vals[2]),
            a6=float(safe_vals[3]),
            p_in=float(safe_vals[4]),
            t_in=float(safe_vals[5]),
        )
        used_fallback = True

    if return_meta:
        meta: dict[str, object] = {
            "input_used": {k: float(v) for k, v in zip(_LE_PINN_INPUT_NAMES, safe_vals)},
            "input_adjustments": adjustments,
            "used_fallback": used_fallback,
        }
        return pred_dict, meta
    return pred_dict


def get_le_pinn_input_warnings(
    checkpoint_path: str,
    x: float,
    y: float,
    a5: float,
    a6: float,
    p_in: float,
    t_in: float,
) -> list[str]:
    """
    Return warnings when inference inputs are outside checkpoint training domain.

    Important for checkpoints where some features were constant during training
    (min == max), which makes min-max normalization extremely sensitive.
    """
    warnings_out: list[str] = []
    try:
        _, input_norm, _ = load_le_pinn_checkpoint(checkpoint_path)
    except Exception:
        return warnings_out
    if input_norm is None or input_norm.data_min is None or input_norm.data_max is None:
        return warnings_out

    vals = [x, y, a5, a6, p_in, t_in]
    names = ["x", "y", "A5", "A6", "P_in", "T_in"]
    mins = input_norm.data_min.cpu().numpy()
    maxs = input_norm.data_max.cpu().numpy()

    for i, (name, v) in enumerate(zip(names, vals)):
        mn = float(mins[i])
        mx = float(maxs[i])
        if abs(mx - mn) < 1e-8:
            if abs(v - mn) > max(1e-6, 1e-6 * abs(mn)):
                warnings_out.append(
                    f"`{name}` was constant during training ({mn:.6g}) "
                    f"but current input is {v:.6g}. "
                    "This can produce invalid/exploding predictions."
                )
        else:
            if v < mn or v > mx:
                warnings_out.append(
                    f"`{name}` is outside training range [{mn:.6g}, {mx:.6g}] "
                    f"(current {v:.6g}). Extrapolation may be unreliable."
                )
    return warnings_out


def _build_quick_opt_fuel_wrapper(row: pd.Series):
    """
    Rebuild the same SAF blend representation used in quick optimization trials.
    """
    from simulation.fuels import make_saf_blend

    p_h = max(float(row.get("HEFA_Frac", 0.0)), 0.0)
    p_f = max(float(row.get("FT_Frac", 0.0)), 0.0)
    p_a = max(float(row.get("ATJ_Frac", 0.0)), 0.0)

    saf_total = float(row.get("SAF_Total", p_h + p_f + p_a))
    jet_a = max(1.0 - saf_total, 0.0)

    # Normalise in case of small CSV rounding drift.
    total = jet_a + p_h + p_f + p_a
    if total <= 0.0:
        raise ValueError("Invalid blend fractions: total <= 0")
    jet_a, p_h, p_f, p_a = [v / total for v in (jet_a, p_h, p_f, p_a)]

    blend = make_saf_blend(jet_a, p_h, p_f, p_a, enforce_astm=True)
    fuel_blend = type(
        "SafeFuelWrapper",
        (),
        {
            "name": str(row.get("Trial", "ParetoTrial")),
            "composition": blend.species,
            "as_composition_string": lambda self: ", ".join(
                f"{k}:{v}" for k, v in self.composition.items()
            ),
        },
    )()

    lca = float(
        row.get(
            "LCA",
            jet_a * 1.0 + p_h * 0.2 + p_f * 0.1 + p_a * 0.3,
        )
    )
    return fuel_blend, lca


def _resolve_pareto_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return Pareto rows from either dashboard optimization or pre-filtered CSVs.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    if "ParetoOptimal" in out.columns:
        out = out[out["ParetoOptimal"].astype(bool)].copy()
    if "Trial" not in out.columns:
        out["Trial"] = np.arange(len(out), dtype=int)
    return out.reset_index(drop=True)


def run_pareto_le_pinn_validation(
    engine,
    df_opt: pd.DataFrame,
    checkpoint_path: str,
    max_points: int = 12,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """
    Validate Pareto-optimal trials by comparing nozzle reference states from
    rerun cycle simulations against LE-PINN predictions.

    LE-PINN CFD-validation RMSE values are retained for summary metrics.
    Saved validation plots use empirical Pareto-sample ±2σ error bars.
    """
    from simulation.nozzle.le_pinn import LE_PINN, validate_le_pinn

    pareto = _resolve_pareto_rows(df_opt)
    if pareto.empty:
        raise ValueError("No Pareto rows found for validation.")

    pareto = pareto.head(max_points).copy()

    # CFD-backed RMSE for uncertainty bars
    rmse_map: dict[str, float] = {}
    model = LE_PINN()
    metrics = validate_le_pinn(
        model,
        dataset_path=str(ROOT / "data" / "processed" / "master_shock_dataset.pt"),
        checkpoint_path=checkpoint_path,
        device="cpu",
        verbose=False,
    )
    for var in ("rho", "u", "P", "T"):
        key = f"rmse_{var}"
        if key in metrics and np.isfinite(metrics[key]):
            rmse_map[var] = float(metrics[key])

    _, in_norm, _ = load_le_pinn_checkpoint(checkpoint_path)
    if in_norm is None or in_norm.data_min is None or in_norm.data_max is None:
        # Conservative default if checkpoint does not carry input normalizers.
        x_eval, y_eval, a5_eval, a6_eval = 0.5, 0.0, 0.2, 0.34
        wall_d = 0.02
    else:
        mins = in_norm.data_min.detach().cpu().numpy()
        maxs = in_norm.data_max.detach().cpu().numpy()
        x_eval = float(maxs[0])  # nozzle exit plane in training domain
        if mins[1] <= 0.0 <= maxs[1]:
            y_eval = 0.0          # centreline
        else:
            y_eval = float(0.5 * (mins[1] + maxs[1]))
        a5_eval = float(mins[2])
        a6_eval = float(mins[3])
        wall_d = max(float(maxs[1] - y_eval), 0.0)

    rows: list[dict[str, float | int | str]] = []
    all_adjustments: list[str] = []
    for i, row in pareto.iterrows():
        fuel_blend, lca = _build_quick_opt_fuel_wrapper(row)
        phi = float(row.get("Phi", 0.5))

        result, _ = _capture_call(
            engine.run_full_cycle,
            fuel_blend=fuel_blend,
            phi=phi,
            combustor_efficiency=0.98,
            lca_factor=lca,
        )

        turb = result["turbine"]
        noz = result["nozzle"]
        preds, meta = run_le_pinn_inference(
            checkpoint_path=checkpoint_path,
            x=x_eval,
            y=y_eval,
            a5=a5_eval,
            a6=a6_eval,
            p_in=float(turb["p"]),
            t_in=float(turb["T"]),
            wall_distance=wall_d,
            return_meta=True,
        )

        adjustments = [str(msg) for msg in meta.get("input_adjustments", [])]
        all_adjustments.extend(adjustments)

        rec = {
            "ParetoRank": int(i + 1),
            "Trial": int(row.get("Trial", i)),
            "TSFC": float(row.get("TSFC", np.nan)),
            "Thrust_kN": float(row.get("Thrust_kN", np.nan)),
            "CO2": float(row.get("CO2", np.nan)),
            "NOx": float(row.get("NOx", np.nan)),
            "ref_rho": float(noz["rho"]),
            "pred_rho": float(preds["rho"]),
            "ref_u": float(noz["u"]),
            "pred_u": float(preds["u"]),
            "ref_P": float(noz["p"]),
            "pred_P": float(preds["P"]),
            "ref_T": float(noz["T"]),
            "pred_T": float(preds["T"]),
            "input_adjustments": " | ".join(adjustments),
            "used_fallback": bool(meta.get("used_fallback", False)),
        }

        for var in ("rho", "u", "P", "T"):
            rec[f"abs_err_{var}"] = abs(float(rec[f"pred_{var}"]) - float(rec[f"ref_{var}"]))
        rows.append(rec)

    return pd.DataFrame(rows), rmse_map, all_adjustments


def make_pareto_validation_errorbar_plots(
    df_val: pd.DataFrame,
    rmse_map: dict[str, float],
) -> tuple[plt.Figure, plt.Figure]:
    """Create ±2σ validation plots for Pareto LE-PINN comparison."""
    vars_meta = [
        ("rho", "kg/m³"),
        ("u", "m/s"),
        ("P", "Pa"),
        ("T", "K"),
    ]

    fig_scatter, axs_scatter = plt.subplots(2, 2, figsize=(12, 9))
    axs_scatter = axs_scatter.flatten()
    for ax, (var, unit) in zip(axs_scatter, vars_meta):
        x = df_val[f"ref_{var}"].to_numpy(dtype=float)
        y = df_val[f"pred_{var}"].to_numpy(dtype=float)
        sigma_y = float(np.nanstd(y))
        yerr = np.full_like(y, 2.0 * sigma_y, dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=3,
            color="#1f77b4",
            alpha=0.9,
        )
        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        pad = max(1e-9, 0.06 * (hi - lo if hi > lo else 1.0))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0)
        ax.set_xlabel(f"Nozzle Reference {var} ({unit})")
        ax.set_ylabel(f"LE-PINN {var} ({unit})")
        ax.set_title(f"{var} | y-error = ±2σ ({2.0 * sigma_y:.3e})")
        ax.grid(alpha=0.25)
    fig_scatter.suptitle("Pareto Validation: LE-PINN vs Nozzle Reference (with ±2σ bars)", fontsize=13)
    fig_scatter.tight_layout()

    fig_resid, axs_resid = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axs_resid = axs_resid.flatten()
    idx = np.arange(len(df_val))
    labels = df_val["Trial"].astype(int).tolist()
    for ax, (var, unit) in zip(axs_resid, vars_meta):
        residual = (df_val[f"pred_{var}"] - df_val[f"ref_{var}"]).to_numpy(dtype=float)
        sigma_resid = float(np.nanstd(residual))
        yerr = np.full_like(residual, 2.0 * sigma_resid, dtype=float)
        ax.errorbar(
            idx,
            residual,
            yerr=yerr,
            fmt="o-",
            color="#d62728",
            capsize=3,
            linewidth=1.1,
            markersize=4,
        )
        if sigma_resid > 0:
            band = 2.0 * sigma_resid
            ax.fill_between(idx, -band, band, color="gray", alpha=0.15, label="±2σ")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"Pred - Ref ({unit})")
        ax.set_title(f"{var} residuals | ±2σ ({2.0 * sigma_resid:.3e})")
        ax.grid(alpha=0.25)
    axs_resid[-1].set_xticks(idx)
    axs_resid[-1].set_xticklabels(labels, rotation=35, ha="right")
    axs_resid[-2].set_xticks(idx)
    axs_resid[-2].set_xticklabels(labels, rotation=35, ha="right")
    fig_resid.suptitle("Pareto Validation Residuals with ±2σ Error Bars", fontsize=13)
    fig_resid.tight_layout()

    return fig_scatter, fig_resid


def make_pareto_validation_bar_plot(df_val: pd.DataFrame) -> plt.Figure:
    """Create grouped bar charts comparing regular nozzle vs LE-PINN outputs."""
    vars_meta = [
        ("rho", "kg/m³"),
        ("u", "m/s"),
        ("P", "Pa"),
        ("T", "K"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()
    x = np.array([0, 1], dtype=float)
    labels = ["Regular Nozzle", "LE-PINN"]

    for ax, (var, unit) in zip(axs, vars_meta):
        ref_vals = df_val[f"ref_{var}"].to_numpy(dtype=float)
        pred_vals = df_val[f"pred_{var}"].to_numpy(dtype=float)
        means = np.array([np.nanmean(ref_vals), np.nanmean(pred_vals)], dtype=float)
        errs = 2.0 * np.array([np.nanstd(ref_vals), np.nanstd(pred_vals)], dtype=float)

        bars = ax.bar(
            x,
            means,
            yerr=errs,
            capsize=4,
            width=0.62,
            color=["#4c78a8", "#f58518"],
            alpha=0.88,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel(f"{var} ({unit})")
        ax.set_title(f"{var}: mean ±2σ")
        ax.grid(axis="y", alpha=0.25)

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fallback_count = int(df_val.get("used_fallback", pd.Series(dtype=bool)).fillna(False).sum())
    fig.suptitle(
        f"Validation Bar Graph: LE-PINN vs Regular Nozzle Outputs"
        f" | fallback points = {fallback_count}/{len(df_val)}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def save_pareto_validation_plots(
    fig_scatter: plt.Figure,
    fig_resid: plt.Figure,
    fig_bar: plt.Figure,
) -> tuple[Path, Path, Path]:
    """Save Pareto LE-PINN validation plots to outputs/plots."""
    out_dir = ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = out_dir / "pareto_le_pinn_validation_scatter.png"
    resid_path = out_dir / "pareto_le_pinn_validation_residuals.png"
    bar_path = out_dir / "pareto_le_pinn_validation_bar.png"
    fig_scatter.savefig(scatter_path, dpi=300, bbox_inches="tight")
    fig_resid.savefig(resid_path, dpi=300, bbox_inches="tight")
    fig_bar.savefig(bar_path, dpi=300, bbox_inches="tight")
    return scatter_path, resid_path, bar_path


def render_pareto_le_pinn_validation_panel(engine) -> None:
    """Render consolidated Pareto validation controls/results for LE-PINN."""
    st.subheader("Pareto Validation (LE-PINN vs Nozzle)")
    st.caption(
        "Re-runs Pareto trials, compares nozzle reference states against LE-PINN predictions, "
        "and plots empirical ±2σ error bars."
    )

    val_ckpt_paths = sorted((ROOT / "models").glob("le_pinn*.pt"))
    if not val_ckpt_paths:
        st.warning("No LE-PINN checkpoints found in models/ for Pareto validation.")
        return

    val_opts = [str(p) for p in val_ckpt_paths]
    val_default_idx = _select_default_le_pinn_checkpoint(val_ckpt_paths)
    val_ckpt = st.selectbox(
        "Validation checkpoint",
        val_opts,
        index=val_default_idx,
        format_func=lambda p: Path(p).name,
        key="pareto_val_ckpt",
    )

    source = st.selectbox(
        "Pareto source",
        [
            "Current dashboard results (session)",
            "outputs/results/optimization_results_dashboard.csv",
            "outputs/results/pareto_optimal_solutions.csv",
        ],
        key="pareto_val_source",
    )

    max_points = st.slider(
        "Max Pareto points to validate",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
        key="pareto_val_max_points",
    )

    # Warn users when checkpoint has constant input dimensions.
    try:
        _, _in_norm_val, _ = load_le_pinn_checkpoint(val_ckpt)
        if _in_norm_val is not None and _in_norm_val.data_min is not None and _in_norm_val.data_max is not None:
            delta = (_in_norm_val.data_max - _in_norm_val.data_min).abs().cpu().numpy()
            const_dims = [name for name, d in zip(_LE_PINN_INPUT_NAMES, delta) if d < 1e-8]
            if const_dims:
                st.info(
                    "Selected checkpoint has fixed training inputs: "
                    + ", ".join(const_dims)
                    + ". Those values are locked during inference."
                )
    except Exception:
        pass

    run_pareto_val = st.button("Run Pareto LE-PINN Validation", key="run_pareto_val_btn")
    if run_pareto_val:
        try:
            if source == "Current dashboard results (session)":
                if "opt_df" not in st.session_state:
                    raise ValueError("No session optimization results found. Run optimization first or choose a CSV source.")
                df_src = st.session_state["opt_df"].copy()
            elif source == "outputs/results/optimization_results_dashboard.csv":
                src_path = ROOT / "outputs" / "results" / "optimization_results_dashboard.csv"
                if not src_path.exists():
                    raise FileNotFoundError(f"Missing file: {src_path}")
                df_src = pd.read_csv(src_path)
            else:
                src_path = ROOT / "outputs" / "results" / "pareto_optimal_solutions.csv"
                if not src_path.exists():
                    raise FileNotFoundError(f"Missing file: {src_path}")
                df_src = pd.read_csv(src_path)
                # File is already Pareto-only in this project.
                if "ParetoOptimal" not in df_src.columns:
                    df_src["ParetoOptimal"] = True

            with st.spinner("Running Pareto validation with LE-PINN…"):
                df_val, rmse_map, adjustments = run_pareto_le_pinn_validation(
                    engine=engine,
                    df_opt=df_src,
                    checkpoint_path=val_ckpt,
                    max_points=max_points,
                )
                fig_scatter, fig_resid = make_pareto_validation_errorbar_plots(df_val, rmse_map)
                fig_bar = make_pareto_validation_bar_plot(df_val)
                scatter_path, resid_path, bar_path = save_pareto_validation_plots(
                    fig_scatter, fig_resid, fig_bar
                )
                plt.close(fig_scatter)
                plt.close(fig_resid)
                plt.close(fig_bar)

            st.session_state["pareto_val_df"] = df_val
            st.session_state["pareto_val_rmse_map"] = rmse_map
            st.session_state["pareto_val_adjustments"] = adjustments
            st.session_state["pareto_val_plot_paths"] = (
                str(scatter_path),
                str(resid_path),
                str(bar_path),
            )
            st.success("Pareto LE-PINN validation complete.")
        except Exception as exc:
            st.error(f"Pareto LE-PINN validation failed: {exc}")

    if "pareto_val_df" in st.session_state and "pareto_val_rmse_map" in st.session_state:
        df_val = st.session_state["pareto_val_df"]
        rmse_map = st.session_state["pareto_val_rmse_map"]
        adjustments = st.session_state.get("pareto_val_adjustments", [])
        plot_paths = st.session_state.get("pareto_val_plot_paths", ("", "", ""))
        if len(plot_paths) < 3:
            plot_paths = tuple(list(plot_paths) + [""] * (3 - len(plot_paths)))

        fallback_count = int(df_val.get("used_fallback", pd.Series(dtype=bool)).fillna(False).sum())
        if fallback_count > 0:
            st.info(
                f"Physical fallback was used for {fallback_count} of {len(df_val)} Pareto points. "
                "Those LE-PINN outputs were replaced by a physics-consistent isentropic estimate."
            )

        summary_rows = []
        for var in ("rho", "u", "P", "T"):
            mae = float(df_val[f"abs_err_{var}"].mean()) if f"abs_err_{var}" in df_val.columns else np.nan
            rmse_ref = float(rmse_map.get(var, np.nan))
            ratio = mae / rmse_ref if np.isfinite(rmse_ref) and rmse_ref > 0 else np.nan
            summary_rows.append(
                {
                    "variable": var,
                    "MAE_vs_nozzle_ref": mae,
                    "LE_PINN_CFD_RMSE": rmse_ref,
                    "MAE/RMSE": ratio,
                }
            )
        st.dataframe(pd.DataFrame(summary_rows).set_index("variable"), width="stretch")

        show_cols = [
            "ParetoRank", "Trial", "TSFC", "Thrust_kN", "CO2", "NOx",
            "ref_rho", "pred_rho", "abs_err_rho",
            "ref_u", "pred_u", "abs_err_u",
            "ref_P", "pred_P", "abs_err_P",
            "ref_T", "pred_T", "abs_err_T",
        ]
        show_cols = [c for c in show_cols if c in df_val.columns]
        st.dataframe(df_val[show_cols], width="stretch")

        fig_bar = make_pareto_validation_bar_plot(df_val)
        st.pyplot(fig_bar, width="stretch")
        fig_scatter, fig_resid = make_pareto_validation_errorbar_plots(df_val, rmse_map)
        st.pyplot(fig_scatter, width="stretch")
        st.pyplot(fig_resid, width="stretch")
        plt.close(fig_bar)
        plt.close(fig_scatter)
        plt.close(fig_resid)

        if adjustments:
            unique_adj = sorted(set(adjustments))
            st.warning(
                f"{len(adjustments)} LE-PINN input adjustments were applied "
                "(clipped/locked to training domain)."
            )
            with st.expander("Show unique input adjustments"):
                for msg in unique_adj:
                    st.write(f"- {msg}")

        if plot_paths[0] and plot_paths[1] and plot_paths[2]:
            st.caption(
                "Saved plots: "
                f"{Path(plot_paths[2]).name}, {Path(plot_paths[0]).name}, {Path(plot_paths[1]).name} "
                "under outputs/plots/"
            )


def run_quick_optimization(engine, n_trials: int) -> tuple[pd.DataFrame, dict]:
    """Run a lightweight 4-objective optimization suitable for dashboard usage."""
    import optuna
    from simulation.fuels import make_saf_blend

    lca_factors = {"JetA": 1.0, "HEFA": 0.2, "FT": 0.1, "ATJ": 0.3}
    rows: list[dict] = []

    def objective(trial):
        saf_total = trial.suggest_float("saf_total", 0.0, 0.5)
        w_h = trial.suggest_float("w_hefa", 0.0, 1.0)
        w_f = trial.suggest_float("w_ft", 0.0, 1.0)
        w_a = trial.suggest_float("w_atj", 0.0, 1.0)
        phi = trial.suggest_float("phi", 0.35, 0.65)

        jet_a = 1.0 - saf_total
        total_w = max(w_h + w_f + w_a, 1e-8)
        p_h = saf_total * (w_h / total_w)
        p_f = saf_total * (w_f / total_w)
        p_a = saf_total * (w_a / total_w)

        lca = (
            jet_a * lca_factors["JetA"]
            + p_h * lca_factors["HEFA"]
            + p_f * lca_factors["FT"]
            + p_a * lca_factors["ATJ"]
        )

        try:
            blend = make_saf_blend(jet_a, p_h, p_f, p_a, enforce_astm=True)
            fuel_blend = type("SafeFuelWrapper", (), {
                "name": f"Trial_{trial.number}",
                "composition": blend.species,
                "as_composition_string": lambda self: ", ".join(
                    f"{k}:{v}" for k, v in self.composition.items()
                ),
            })()

            result, _ = _capture_call(
                engine.run_full_cycle,
                fuel_blend=fuel_blend,
                phi=phi,
                combustor_efficiency=0.98,
                lca_factor=lca,
            )

            perf = result["performance"]
            emis = result["emissions"]
            tsfc = float(perf["tsfc_mg_per_Ns"])
            thrust = float(perf["thrust_kN"])
            co2 = float(emis["Net_CO2_g_s"])
            nox = float(emis["NOx_g_s"])

            trial.set_user_attr("record", {
                "Trial": trial.number,
                "TSFC": tsfc,
                "Thrust_kN": thrust,
                "CO2": co2,
                "NOx": nox,
                "SAF_Total": saf_total,
                "Phi": phi,
                "LCA": lca,
                "HEFA_Frac": p_h,
                "FT_Frac": p_f,
                "ATJ_Frac": p_a,
            })
            return tsfc, thrust, co2, nox
        except Exception:
            return 1e6, -1e6, 1e6, 1e6

    study = optuna.create_study(
        directions=["minimize", "maximize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    pareto_trials = {t.number for t in study.best_trials}
    for t in study.trials:
        if "record" not in t.user_attrs:
            continue
        rec = dict(t.user_attrs["record"])
        rec["ParetoOptimal"] = t.number in pareto_trials
        rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = ROOT / "outputs" / "results" / "optimization_results_dashboard.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    summary = {
        "completed_trials": int(len(df)),
        "pareto_points": int(df["ParetoOptimal"].sum()) if not df.empty else 0,
        "best_tsfc": float(df["TSFC"].min()) if not df.empty else np.nan,
        "best_thrust": float(df["Thrust_kN"].max()) if not df.empty else np.nan,
        "lowest_co2": float(df["CO2"].min()) if not df.empty else np.nan,
        "lowest_nox": float(df["NOx"].min()) if not df.empty else np.nan,
    }
    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# Extract per-stage scalar quantities
# ─────────────────────────────────────────────────────────────────────────────

# Design-point geometry for velocity estimation where not directly available
_DP = {
    "m_dot_core": 79.9,       # kg/s
    "A_comb_exit": 0.207,     # m²  (combustor exit = turbine inlet area)
    "A_comp_exit": 0.17,      # m²  estimated compressor discharge area
    "R_air": 287.0,
    "gamma_air": 1.4,
    "T_amb": 288.15,
    "P_amb": 101325.0,
}


def extract_stages(result: dict) -> list[dict]:
    """Return list of dicts with T, p, u, rho, mach for each engine stage."""
    dp = _DP

    # ── Inlet (ambient) ──────────────────────────────────────────────────────
    T_in = dp["T_amb"]
    P_in = dp["P_amb"]
    rho_in = P_in / (dp["R_air"] * T_in)
    u_in = 80.0  # estimated core intake velocity (m/s)
    M_in = u_in / np.sqrt(dp["gamma_air"] * dp["R_air"] * T_in)

    # ── Compressor exit ───────────────────────────────────────────────────────
    comp = result["compressor"]
    T_c = comp["T_out"]
    P_c = comp["p_out"]
    rho_c = P_c / (dp["R_air"] * T_c)
    u_c = dp["m_dot_core"] / (rho_c * dp["A_comp_exit"])
    M_c = u_c / np.sqrt(dp["gamma_air"] * dp["R_air"] * T_c)

    # ── Combustor exit ────────────────────────────────────────────────────────
    comb = result["combustor"]
    T_b = comb["T_out"]
    P_b = comb["p_out"]
    R_b = comb.get("R_out", dp["R_air"])
    gamma_b = comb.get("gamma_out", 1.33)
    rho_b = P_b / (R_b * T_b)
    # Velocity estimated via continuity at combustor exit area
    m_dot_total = result["performance"]["total_mass_flow"]
    u_b = m_dot_total / (rho_b * dp["A_comb_exit"])
    M_b = u_b / np.sqrt(gamma_b * R_b * T_b)

    # ── Turbine exit ──────────────────────────────────────────────────────────
    turb = result["turbine"]
    T_t = turb["T"]
    P_t = turb["p"]
    rho_t = turb["rho"]
    u_t = turb["u"]
    R_t = turb.get("R", R_b)
    gamma_t = turb.get("gamma", gamma_b)
    M_t = u_t / np.sqrt(max(gamma_t * R_t * T_t, 1e-6))

    # ── Nozzle exit ───────────────────────────────────────────────────────────
    nozz = result["nozzle"]
    T_n = nozz["T"]
    P_n = nozz["p"]
    rho_n = nozz["rho"]
    u_n = nozz["u"]
    M_n = u_n / np.sqrt(max(gamma_t * R_t * T_n, 1e-6))

    return [
        {"name": "Inlet",      "T": T_in, "p": P_in, "u": u_in, "rho": rho_in, "mach": M_in},
        {"name": "Compressor", "T": T_c,  "p": P_c,  "u": u_c,  "rho": rho_c,  "mach": M_c},
        {"name": "Combustor",  "T": T_b,  "p": P_b,  "u": u_b,  "rho": rho_b,  "mach": M_b},
        {"name": "Turbine",    "T": T_t,  "p": P_t,  "u": u_t,  "rho": rho_t,  "mach": M_t},
        {"name": "Nozzle",     "T": T_n,  "p": P_n,  "u": u_n,  "rho": rho_n,  "mach": M_n},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Engine geometry  (axial cross-section schematic)
# ─────────────────────────────────────────────────────────────────────────────
#  Each row: (stage_name, x_start, x_end,
#             r_outer_start, r_outer_end,
#             r_inner_start, r_inner_end)
_SECTIONS = [
    # (name, x0, x1, ro_start, ro_end, ri_start, ri_end)
    # Trent 1000-like proportions: large fan, narrow HPC core, long turbine
    ("Inlet",      0.00, 0.55,  1.00, 1.00,  0.40, 0.40),   # Fan + inlet lip
    ("Compressor", 0.55, 1.55,  0.58, 0.28,  0.24, 0.10),   # HPC: aggressive taper
    ("Combustor",  1.55, 2.05,  0.28, 0.28,  0.07, 0.07),   # Annular combustor
    ("Turbine",    2.05, 3.05,  0.28, 0.46,  0.07, 0.18),   # HPT + IPT + LPT
    ("Nozzle",     3.05, 3.85,  0.46, 0.26,  0.00, 0.00),   # Convergent hot nozzle
]

# Bypass duct knots (visual geometry only — not tied to a simulation stage).
# Represents the fan nacelle/bypass stream from fan exit to cold nozzle exit.
#   columns: x, r_outer, r_inner
_BYPASS_KNOTS = np.array([
    [0.55, 1.03, 1.00],   # fan exit plane — thin outer wall
    [1.20, 1.03, 1.00],   # constant-radius nacelle barrel
    [2.30, 0.85, 0.82],   # begins converging toward bypass nozzle
    [3.00, 0.60, 0.57],   # bypass nozzle exit
])

_QTY_KEY = {
    "Temperature (K)": "T",
    "Pressure (Pa)":   "p",
    "Mach Number":     "mach",
    "Density (kg/m³)": "rho",
    "Velocity (m/s)":  "u",
}

_CMAP_NAME = {
    "Temperature (K)": "inferno",
    "Pressure (Pa)":   "viridis",
    "Mach Number":     "plasma",
    "Density (kg/m³)": "cividis",
    "Velocity (m/s)":  "hot",
}

_QTY_UNITS = {
    "Temperature (K)": "K",
    "Pressure (Pa)":   "Pa",
    "Mach Number":     "",
    "Density (kg/m³)": "kg/m³",
    "Velocity (m/s)":  "m/s",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2-D Heatmap (matplotlib — always available)
# ─────────────────────────────────────────────────────────────────────────────

def _build_boundary_arrays(xi: np.ndarray):
    """Return r_outer(x) and r_inner(x) arrays via piecewise linear interpolation."""
    x_knots = [s[1] for s in _SECTIONS] + [_SECTIONS[-1][2]]
    ro_knots = [s[3] for s in _SECTIONS] + [_SECTIONS[-1][4]]
    ri_knots = [s[5] for s in _SECTIONS] + [_SECTIONS[-1][6]]
    return np.interp(xi, x_knots, ro_knots), np.interp(xi, x_knots, ri_knots)


def render_2d(stages: list[dict], qty_label: str) -> None:
    """Render a 2-D cross-section heatmap of the engine."""
    qty_key  = _QTY_KEY[qty_label]
    cmap_name = _CMAP_NAME[qty_label]
    cmap     = colormaps[cmap_name]
    values   = [s[qty_key] for s in stages]

    v_min, v_max = min(values), max(values)
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1.0
    norm = Normalize(vmin=v_min, vmax=v_max)

    # Stage axial centres (for colour interpolation)
    stage_cx = [(s[1] + s[2]) / 2 for s in _SECTIONS]

    # Fine grids — span updated for Trent-1000-like geometry
    X_END = _SECTIONS[-1][2]   # 3.85
    R_MAX = _SECTIONS[0][3]    # 1.00
    xi = np.linspace(0.0, X_END, 600)
    yi = np.linspace(-R_MAX - 0.10, R_MAX + 0.10, 300)
    X, Y = np.meshgrid(xi, yi)

    # Colour field: interpolate scalar value along x axis
    vi = np.interp(xi, stage_cx, values)
    Z  = np.tile(vi, (len(yi), 1))

    # Engine-boundary mask
    r_outer_x, r_inner_x = _build_boundary_arrays(xi)
    R_outer_2d = np.tile(r_outer_x, (len(yi), 1))
    R_inner_2d = np.tile(r_inner_x, (len(yi), 1))
    Y_abs = np.abs(Y)
    mask = (Y_abs <= R_outer_2d) & (Y_abs >= R_inner_2d)
    Z_masked = np.where(mask, Z, np.nan)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Heatmap
    im = ax.imshow(
        Z_masked,
        extent=[0, X_END, yi.min(), yi.max()],
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="bilinear",
    )

    # Core engine outline
    x_all  = np.concatenate([np.linspace(s[1], s[2], 40) for s in _SECTIONS])
    ro_all = np.concatenate([np.linspace(s[3], s[4], 40) for s in _SECTIONS])
    ri_all = np.concatenate([np.linspace(s[5], s[6], 40) for s in _SECTIONS])

    lw_outer, lw_inner = 2.2, 1.1
    ax.plot(x_all,  ro_all, color="white", lw=lw_outer, zorder=6)
    ax.plot(x_all, -ro_all, color="white", lw=lw_outer, zorder=6)
    ax.plot(x_all,  ri_all, color="white", lw=lw_inner, ls="--", alpha=0.55, zorder=6)
    ax.plot(x_all, -ri_all, color="white", lw=lw_inner, ls="--", alpha=0.55, zorder=6)

    # Bypass duct outline (nacelle — two thin lines per side)
    bp_x  = _BYPASS_KNOTS[:, 0]
    bp_ro = _BYPASS_KNOTS[:, 1]
    bp_ri = _BYPASS_KNOTS[:, 2]
    ax.plot(bp_x,  bp_ro, color="#88aacc", lw=1.8, zorder=6, label="bypass duct")
    ax.plot(bp_x, -bp_ro, color="#88aacc", lw=1.8, zorder=6)
    ax.plot(bp_x,  bp_ri, color="#88aacc", lw=0.9, ls="--", alpha=0.55, zorder=6)
    ax.plot(bp_x, -bp_ri, color="#88aacc", lw=0.9, ls="--", alpha=0.55, zorder=6)
    # Fan inlet lip (elliptical cap at x=0)
    ax.plot([0, 0], [-1.00,  1.00], color="white", lw=lw_outer, zorder=6)
    # Bypass duct exit cap
    ax.plot([bp_x[-1], bp_x[-1]], [-bp_ro[-1], bp_ro[-1]],
            color="#88aacc", lw=1.8, zorder=6)

    # Section dividers
    for sec in _SECTIONS[1:]:
        xd = sec[1]
        ro = np.interp(xd, [s[1] for s in _SECTIONS] + [_SECTIONS[-1][2]],
                       [s[3] for s in _SECTIONS] + [_SECTIONS[-1][4]])
        ri = np.interp(xd, [s[1] for s in _SECTIONS] + [_SECTIONS[-1][2]],
                       [s[5] for s in _SECTIONS] + [_SECTIONS[-1][6]])
        ax.plot([xd, xd], [ ri,  ro], color="white", lw=0.8, alpha=0.45, zorder=5)
        ax.plot([xd, xd], [-ri, -ro], color="white", lw=0.8, alpha=0.45, zorder=5)

    # Nozzle exit cap
    ax.plot([X_END, X_END], [0.0,  0.26], color="white", lw=lw_outer, zorder=6)
    ax.plot([X_END, X_END], [0.0, -0.26], color="white", lw=lw_outer, zorder=6)

    # Stage labels with value
    unit = _QTY_UNITS[qty_label]
    for stage, cx in zip(stages, stage_cx):
        val = stage[qty_key]
        txt = f"{stage['name']}\n{val:.2g} {unit}"
        ax.text(
            cx, 0.0, txt,
            ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.45),
            zorder=10,
        )

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.88)
    cbar.set_label(qty_label, color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Axial position (m)", color="white", fontsize=11)
    ax.set_ylabel("Radius (m)", color="white", fontsize=11)
    ax.set_title(f"Engine cross-section — {qty_label}", color="white",
                 fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.set_xlim(-0.08, X_END + 0.10)
    ax.set_ylim(-(R_MAX + 0.15), R_MAX + 0.18)
    ax.axhline(0, color="white", lw=0.4, ls=":", alpha=0.35)

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D Plotly engine (browser WebGL — no threading requirements)
# ─────────────────────────────────────────────────────────────────────────────

def render_3d(stages: list[dict], qty_label: str) -> bool:
    """
    Render interactive 3-D engine cross-section via Plotly (WebGL).
    Returns True always (plotly is a required dependency).
    """
    import plotly.graph_objects as go

    qty_key   = _QTY_KEY[qty_label]
    cmap_name = _CMAP_NAME[qty_label]

    # matplotlib cmap → Plotly built-in colorscale
    _CMAP_TO_PLOTLY = {
        "inferno": "Inferno",
        "viridis": "Viridis",
        "plasma":  "Plasma",
        "cividis": "Cividis",
        "hot":     "Hot",
    }
    colorscale = _CMAP_TO_PLOTLY.get(cmap_name, "Inferno")

    values  = np.array([s[qty_key] for s in stages], dtype=float)
    v_min, v_max = values.min(), values.max()
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1.0

    unit = _QTY_UNITS[qty_label]
    res  = 72  # angular resolution for smooth cylinders

    theta = np.linspace(0, 2 * np.pi, res + 1)

    # Use coloraxis so all surfaces share one colorbar without any
    # dummy surface that would distort the scene's z-scale.
    surf_kw = dict(
        coloraxis="coloraxis",
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.95, specular=0.3, roughness=0.35),
        lightposition=dict(x=2000, y=1000, z=3000),
    )

    def _frustum(x0, x1, r0, r1, op, color_val, hover):
        X = np.array([[x0] * (res + 1), [x1] * (res + 1)])
        Y = np.array([r0 * np.cos(theta), r1 * np.cos(theta)])
        Z = np.array([r0 * np.sin(theta), r1 * np.sin(theta)])
        return go.Surface(x=X, y=Y, z=Z,
                          surfacecolor=np.full_like(X, color_val, dtype=float),
                          opacity=op, hovertemplate=hover, **surf_kw)

    def _annular_cap(xc, ri, ro, op, color_val, hover):
        Xc = np.full((2, res + 1), xc)
        Yc = np.outer(np.array([ri, ro]), np.cos(theta))
        Zc = np.outer(np.array([ri, ro]), np.sin(theta))
        return go.Surface(x=Xc, y=Yc, z=Zc,
                          surfacecolor=np.full_like(Xc, color_val, dtype=float),
                          opacity=op, hovertemplate=hover, **surf_kw)

    traces = []

    for sec, stage in zip(_SECTIONS, stages):
        name, x0, x1, ro0, ro1, ri0, ri1 = sec
        val   = stage[qty_key]
        hover = f"<b>{name}</b><br>{qty_label}: {val:.4g} {unit}<extra></extra>"

        # Outer shell — fully opaque
        traces.append(_frustum(x0, x1, ro0, ro1, 1.0, val, hover))

        # Inner hub
        if max(ri0, ri1) > 0.02:
            traces.append(_frustum(x0, x1, ri0, ri1, 1.0, val * 0.82, hover))

        # Annular end caps
        for xc, ro_c, ri_c in [(x0, ro0, ri0), (x1, ro1, ri1)]:
            if ro_c - ri_c >= 0.015:
                traces.append(_annular_cap(xc, ri_c, ro_c, 1.0, val, hover))

        # Stage label
        cx = (x0 + x1) / 2.0
        traces.append(go.Scatter3d(
            x=[cx], y=[0.0], z=[max(ro0, ro1) + 0.12],
            mode="text",
            text=[f"<b>{name}</b><br>{val:.3g} {unit}"],
            textfont=dict(color="white", size=9),
            hoverinfo="skip", showlegend=False,
        ))

    # ── Bypass duct / nacelle (visual only, grey, semi-transparent) ───────────
    bp_x  = _BYPASS_KNOTS[:, 0]
    bp_ro = _BYPASS_KNOTS[:, 1]
    bp_ri = _BYPASS_KNOTS[:, 2]
    bypass_hover = "<b>Bypass duct</b><extra></extra>"
    bypass_kw = dict(
        showscale=False,
        colorscale=[[0, "#4a6080"], [1, "#4a6080"]],
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.15, roughness=0.6),
        lightposition=dict(x=2000, y=1000, z=3000),
    )
    for i in range(len(bp_x) - 1):
        for r0_arr, r1_arr in [(bp_ro, bp_ro), (bp_ri, bp_ri)]:
            Xb = np.array([[bp_x[i]] * (res + 1), [bp_x[i+1]] * (res + 1)])
            Yb = np.array([r0_arr[i] * np.cos(theta), r1_arr[i+1] * np.cos(theta)])
            Zb = np.array([r0_arr[i] * np.sin(theta), r1_arr[i+1] * np.sin(theta)])
            traces.append(go.Surface(
                x=Xb, y=Yb, z=Zb,
                surfacecolor=np.zeros_like(Xb),
                opacity=0.40, hovertemplate=bypass_hover, **bypass_kw,
            ))
    # Bypass duct exit cap
    traces.append(go.Surface(
        x=np.full((2, res + 1), bp_x[-1]),
        y=np.outer(np.array([bp_ri[-1], bp_ro[-1]]), np.cos(theta)),
        z=np.outer(np.array([bp_ri[-1], bp_ro[-1]]), np.sin(theta)),
        surfacecolor=np.zeros((2, res + 1)),
        opacity=0.40, hovertemplate=bypass_hover, **bypass_kw,
    ))

    fig = go.Figure(data=traces)

    x_end = _SECTIONS[-1][2]              # 3.85 m
    r_max = float(_BYPASS_KNOTS[:, 1].max())   # 1.03 m (nacelle outer)
    x_len = x_end - _SECTIONS[0][1]      # 3.85 m
    ax_ratio = x_len / (2 * r_max)       # ≈ 1.87 — keeps cross-section circular

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=v_min, cmax=v_max,
            colorbar=dict(
                title=dict(text=qty_label, font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=9),
                len=0.75, thickness=14, x=1.0,
            ),
        ),
        scene=dict(
            xaxis=dict(
                title="Axial (m)", color="white", gridcolor="#333",
                backgroundcolor="#0d1117", showbackground=True,
                showspikes=False, range=[-0.05, x_end + 0.10],
            ),
            yaxis=dict(
                title="", color="white", gridcolor="#333",
                backgroundcolor="#0d1117", showbackground=True,
                showspikes=False, range=[-r_max - 0.05, r_max + 0.05],
                showticklabels=False,
            ),
            zaxis=dict(
                title="", color="white", gridcolor="#333",
                backgroundcolor="#0d1117", showbackground=True,
                showspikes=False, range=[-r_max - 0.05, r_max + 0.05],
                showticklabels=False,
            ),
            bgcolor="#0d1117",
            aspectmode="manual",
            aspectratio=dict(x=ax_ratio, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.05, y=-2.2, z=0.70),
            ),
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=60, t=10, b=0),
        height=480,
    )

    st.plotly_chart(fig, width="stretch")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Fuel-blend doughnut chart
# ─────────────────────────────────────────────────────────────────────────────

def render_blend_chart(jet_pct, hefa_pct, ft_pct, atj_pct):
    sizes  = [jet_pct, hefa_pct, ft_pct, atj_pct]
    labels = ["Jet-A1", "HEFA-SPK", "FT-SPK", "ATJ-SPK"]
    colors = ["#4a90d9", "#f5a623", "#7ed321", "#bd10e0"]

    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if not non_zero:
        return
    s_nz, l_nz, c_nz = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    wedges, texts, autotexts = ax.pie(
        s_nz,
        labels=l_nz,
        colors=c_nz,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0e1117", linewidth=1.5),
        textprops=dict(color="white", fontsize=7.5),
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
    ax.set_title("Fuel blend", color="white", fontsize=9, pad=4)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("✈ Jet Engine Interactive Dashboard")
    st.caption(
        "Physics-based turbofan simulation (Cantera + PINN) · "
        "ASTM D7566 compliant fuel blending · "
        "Integrated cycle, component runners, LE-PINN inference, and quick optimization"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⛽ Fuel Blend")
        st.caption("ASTM D7566: Jet-A1 must be ≥ 50 %")

        hefa_pct = st.slider("HEFA-SPK %", 0, 50, 15, step=5)
        ft_max   = max(0, 50 - hefa_pct)
        if ft_max > 0:
            ft_pct = st.slider("FT-SPK %", 0, ft_max, min(10, ft_max), step=5)
        else:
            ft_pct = 0
            st.caption("FT-SPK %: 0 (capacity used by HEFA)")
        atj_max  = max(0, 50 - hefa_pct - ft_pct)
        if atj_max > 0:
            atj_pct = st.slider("ATJ-SPK %", 0, atj_max, min(5, atj_max), step=5)
        else:
            atj_pct = 0
            st.caption("ATJ-SPK %: 0 (capacity used)")
        jet_pct  = 100 - hefa_pct - ft_pct - atj_pct

        saf_total = hefa_pct + ft_pct + atj_pct
        c1, c2 = st.columns(2)
        c1.metric("Jet-A1", f"{jet_pct} %")
        c2.metric("SAF", f"{saf_total} %")

        if jet_pct < 50:
            st.error("Jet-A1 < 50 % — ASTM D7566 violated!")
            return

        st.divider()
        st.header("⚙ Operating Conditions")
        phi = st.slider(
            "Equivalence ratio ϕ", 0.20, 0.80, 0.50, step=0.05,
            help="ϕ < 1 = lean combustion (typical for gas turbines)",
        )
        eff = st.slider(
            "Combustor efficiency", 0.90, 0.99, 0.98, step=0.01,
            format="%.2f",
        )

        st.divider()
        st.caption("Run controls are available inside each tab.")

    # ── Load engine ───────────────────────────────────────────────────────────
    engine = load_engine()

    tab_cycle, tab_components, tab_lepinn, tab_opt = st.tabs(
        ["🔄 Integrated Cycle", "🧩 Components", "🧠 LE-PINN", "📈 Optimization"]
    )

    with tab_cycle:
        qty_label = st.radio(
            "Visualise quantity",
            list(_QTY_KEY.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )

        run_cycle_btn = st.button("▶ Run Integrated Cycle", type="primary")
        need_run = run_cycle_btn or ("sim_result" not in st.session_state)
        if need_run:
            with st.spinner("Running full engine cycle…"):
                try:
                    result, lca = run_simulation(
                        engine, jet_pct, hefa_pct, ft_pct, atj_pct, phi, eff
                    )
                    st.session_state["sim_result"] = result
                    st.session_state["sim_lca"] = lca
                except Exception as exc:
                    st.error(f"Simulation error: {exc}")
                    st.stop()

        result = st.session_state["sim_result"]
        lca = st.session_state.get("sim_lca", 1.0)
        stages = extract_stages(result)

        col_viz, col_metrics = st.columns([3, 1], gap="medium")

        with col_viz:
            tab3d, tab2d = st.tabs(["🔵 3D View", "🌡 2D Heatmap"])

            with tab3d:
                st.caption("Interactive 3-D engine coloured by the selected quantity. Drag to rotate, scroll to zoom.")
                render_3d(stages, qty_label)

            with tab2d:
                render_2d(stages, qty_label)

        with col_metrics:
            perf = result["performance"]
            emis = result.get("emissions", {})

            st.subheader("Performance")
            thrust_kN = perf.get("thrust_kN", 0.0)
            tsfc_mg = perf.get("tsfc_mg_per_Ns", float("inf"))
            eta_th = perf.get("thermal_efficiency", 0.0)
            m_fuel = perf.get("fuel_mass_flow", 0.0)

            st.metric("Thrust", f"{thrust_kN:.1f} kN")
            st.metric("TSFC", f"{tsfc_mg:.1f} mg/(N·s)" if tsfc_mg < 1e9 else "∞")
            st.metric("η (kinetic)", f"{eta_th * 100:.1f} %")
            st.metric("Fuel flow", f"{m_fuel:.3f} kg/s")

            st.subheader("Emissions")
            nox = emis.get("NOx_g_s", 0.0)
            co2 = emis.get("Net_CO2_g_s", 0.0)
            co = emis.get("CO_g_s", 0.0)
            st.metric("NOx", f"{nox:.3f} g/s")
            st.metric("CO₂", f"{co2:.0f} g/s")
            st.metric("CO", f"{co:.4f} g/s")
            st.metric("LCA factor", f"{lca:.2f}")

            st.divider()
            st.subheader("Stage Data")
            df = pd.DataFrame(stages).set_index("name")
            df.columns = ["T (K)", "p (Pa)", "u (m/s)", "ρ (kg/m³)", "Mach"]
            df = df.round({"T (K)": 1, "p (Pa)": 0, "u (m/s)": 1, "ρ (kg/m³)": 4, "Mach": 3})
            st.dataframe(df, width="stretch")

            st.divider()
            render_blend_chart(jet_pct, hefa_pct, ft_pct, atj_pct)

    with tab_components:
        st.subheader("Run Individual Components")
        component = st.selectbox("Component", ["Compressor", "Combustor", "Turbine", "Nozzle"])

        if component == "Compressor":
            c1, c2 = st.columns(2)
            T_in = c1.number_input("T_in (K)", min_value=150.0, max_value=2000.0, value=288.15, step=1.0)
            p_bar = c2.number_input("p_in (bar)", min_value=0.1, max_value=80.0, value=1.013, step=0.01)

            if st.button("Run Compressor"):
                try:
                    comp_res, log_txt = _capture_call(engine.run_compressor, T_in=T_in, p_in=p_bar * 1e5)
                    st.success("Compressor run completed.")
                    st.json(comp_res)
                    with st.expander("Execution log"):
                        st.text(log_txt)
                except Exception as exc:
                    st.error(f"Compressor run failed: {exc}")

        elif component == "Combustor":
            c1, c2 = st.columns(2)
            T_in = c1.number_input("T_in (K)", min_value=300.0, max_value=3000.0, value=900.0, step=10.0)
            p_bar = c2.number_input("p_in (bar)", min_value=1.0, max_value=80.0, value=43.0, step=0.1)

            if st.button("Run Combustor"):
                try:
                    fuel_blend, _ = make_local_blend(jet_pct, hefa_pct, ft_pct, atj_pct)
                    (comb_res, far), log_txt = _capture_call(
                        engine.run_combustor,
                        T_in=T_in,
                        p_in=p_bar * 1e5,
                        fuel_blend=fuel_blend,
                        phi=phi,
                        efficiency=eff,
                    )
                    st.success("Combustor run completed.")
                    st.metric("Fuel-Air Ratio", f"{far:.6f}")
                    st.json(comb_res)
                    with st.expander("Execution log"):
                        st.text(log_txt)
                except Exception as exc:
                    st.error(f"Combustor run failed: {exc}")

        elif component == "Turbine":
            c1, c2, c3 = st.columns(3)
            rho = c1.number_input("rho_in (kg/m³)", min_value=0.01, max_value=100.0, value=4.0, step=0.1)
            vel = c2.number_input("u_in (m/s)", min_value=1.0, max_value=2000.0, value=240.0, step=5.0)
            p_bar = c3.number_input("p_in (bar)", min_value=0.5, max_value=80.0, value=35.0, step=0.1)

            c4, c5, c6, c7 = st.columns(4)
            temp = c4.number_input("T_in (K)", min_value=300.0, max_value=3000.0, value=1700.0, step=10.0)
            cp = c5.number_input("cp (J/kg-K)", min_value=500.0, max_value=2500.0, value=1150.0, step=10.0)
            rgas = c6.number_input("R (J/kg-K)", min_value=150.0, max_value=450.0, value=287.0, step=1.0)
            gamma = c7.number_input("gamma", min_value=1.05, max_value=1.67, value=1.33, step=0.01)

            c8, c9 = st.columns(2)
            m_dot = c8.number_input("m_dot (kg/s)", min_value=1.0, max_value=300.0, value=82.0, step=1.0)
            target_mw = c9.number_input("Target work (MW)", min_value=0.1, max_value=100.0, value=12.0, step=0.1)

            if st.button("Run Turbine"):
                try:
                    state_in = {
                        "rho": rho,
                        "u": vel,
                        "p": p_bar * 1e5,
                        "T": temp,
                        "cp": cp,
                        "R": rgas,
                        "gamma": gamma,
                    }
                    turb_res, log_txt = _capture_call(
                        engine.run_turbine,
                        flow_state_in=state_in,
                        m_dot=m_dot,
                        target_work_total=target_mw * 1e6,
                    )
                    st.success("Turbine run completed.")
                    st.json(turb_res)
                    with st.expander("Execution log"):
                        st.text(log_txt)
                except Exception as exc:
                    st.error(f"Turbine run failed: {exc}")

        else:
            c1, c2, c3 = st.columns(3)
            rho = c1.number_input("rho_in (kg/m³)", min_value=0.01, max_value=100.0, value=0.9, step=0.01)
            vel = c2.number_input("u_in (m/s)", min_value=1.0, max_value=3000.0, value=600.0, step=10.0)
            p_bar = c3.number_input("p_in (bar)", min_value=0.5, max_value=20.0, value=2.5, step=0.05)

            c4, c5, c6 = st.columns(3)
            temp = c4.number_input("T_in (K)", min_value=200.0, max_value=2500.0, value=1200.0, step=10.0)
            cp = c5.number_input("cp (J/kg-K)", min_value=500.0, max_value=2500.0, value=1120.0, step=10.0)
            rgas = c6.number_input("R (J/kg-K)", min_value=150.0, max_value=450.0, value=287.0, step=1.0)
            gamma = st.number_input("gamma", min_value=1.05, max_value=1.67, value=1.33, step=0.01)
            m_dot = st.number_input("m_dot (kg/s)", min_value=1.0, max_value=300.0, value=82.0, step=1.0)

            if st.button("Run Nozzle"):
                try:
                    state_in = {
                        "rho": rho,
                        "u": vel,
                        "p": p_bar * 1e5,
                        "T": temp,
                        "cp": cp,
                        "R": rgas,
                        "gamma": gamma,
                    }
                    noz_res, log_txt = _capture_call(
                        engine.run_nozzle,
                        flow_state_in=state_in,
                        m_dot=m_dot,
                    )
                    st.success("Nozzle run completed.")
                    st.json(noz_res)
                    with st.expander("Execution log"):
                        st.text(log_txt)
                except Exception as exc:
                    st.error(f"Nozzle run failed: {exc}")

    with tab_lepinn:
        st.subheader("LE-PINN (Consolidated)")
        st.caption(
            "Single workflow for LE-PINN inference and validation. "
            "If raw network output is non-physical, a physics-consistent isentropic fallback is used."
        )
        model_paths = sorted((ROOT / "models").glob("le_pinn*.pt"))
        if not model_paths:
            st.warning("No LE-PINN checkpoints found in models/.")
        else:
            model_opts = [str(p) for p in model_paths]
            default_idx = _select_default_le_pinn_checkpoint(model_paths)
            ckpt_path = model_opts[default_idx]

            with st.expander("Advanced: choose checkpoint", expanded=False):
                ckpt_path = st.selectbox(
                    "Checkpoint",
                    model_opts,
                    index=default_idx,
                    format_func=lambda p: Path(p).name,
                    key="lepinn_ckpt_select",
                )

            # Use checkpoint-normalizer defaults when available.
            x0, y0, a50, a60, p0, t0 = 0.3, 0.05, 0.207, 0.340, 350000.0, 1200.0
            try:
                _, _in_norm, _ = load_le_pinn_checkpoint(ckpt_path)
                if _in_norm is not None and _in_norm.data_min is not None and _in_norm.data_max is not None:
                    dmin = _in_norm.data_min.cpu().numpy()
                    dmax = _in_norm.data_max.cpu().numpy()
                    defaults = []
                    for i in range(6):
                        if abs(float(dmax[i] - dmin[i])) < 1e-8:
                            defaults.append(float(dmin[i]))
                        else:
                            defaults.append(float(0.5 * (dmin[i] + dmax[i])))
                    x0, y0, a50, a60, p0, t0 = defaults
            except Exception:
                pass

            c1, c2, c3 = st.columns(3)
            x = c1.number_input("x", value=float(x0), step=0.01)
            y = c2.number_input("y", value=float(y0), step=0.01)
            wall_d = c3.number_input("wall distance", min_value=0.0, value=0.02, step=0.001)

            c4, c5, c6, c7 = st.columns(4)
            a5 = c4.number_input("A5", min_value=0.0, value=float(a50), step=0.005)
            a6 = c5.number_input("A6", min_value=0.0, value=float(a60), step=0.005)
            p_in = c6.number_input("P_in (Pa)", min_value=1.0, value=float(p0), step=1000.0)
            t_in = c7.number_input("T_in (K)", min_value=1.0, value=float(t0), step=5.0)

            for msg in get_le_pinn_input_warnings(ckpt_path, x, y, a5, a6, p_in, t_in):
                st.warning(msg)

            run_infer = st.button("Run LE-PINN Inference")
            run_val = st.button("Run LE-PINN Validation")

            if run_infer:
                try:
                    preds, meta = run_le_pinn_inference(
                        ckpt_path, x, y, a5, a6, p_in, t_in, wall_d, return_meta=True
                    )
                    st.success("Inference completed.")
                    pred_df = pd.DataFrame(
                        [{"variable": v, "value": preds[v]} for v in _LE_PINN_CORE_VARS]
                    ).set_index("variable")
                    st.dataframe(pred_df, width="stretch")
                    if bool(meta.get("used_fallback", False)):
                        st.warning(
                            "Raw LE-PINN output was non-physical for this query. "
                            "Displayed values use isentropic fallback."
                        )
                    adjustments = [str(m) for m in meta.get("input_adjustments", [])]
                    if adjustments:
                        with st.expander("Input adjustments applied"):
                            for msg in adjustments:
                                st.write(f"- {msg}")
                except Exception as exc:
                    st.error(f"LE-PINN inference failed: {exc}")

            if run_val:
                try:
                    from simulation.nozzle.le_pinn import LE_PINN, validate_le_pinn

                    model = LE_PINN()
                    with st.spinner("Running validation against CFD dataset…"):
                        metrics = validate_le_pinn(
                            model,
                            dataset_path=str(ROOT / "data" / "processed" / "master_shock_dataset.pt"),
                            checkpoint_path=ckpt_path,
                            device="cpu",
                            verbose=False,
                        )
                    st.success("Validation completed.")
                    # Present metrics as variable-wise RMSE/R² table when possible.
                    vars_order = ["rho", "u", "v", "P", "T"]
                    rows = []
                    for var in vars_order:
                        rk = f"rmse_{var}"
                        r2k = f"r2_{var}"
                        if rk in metrics or r2k in metrics:
                            rows.append(
                                {
                                    "variable": var,
                                    "RMSE": metrics.get(rk, np.nan),
                                    "R2": metrics.get(r2k, np.nan),
                                }
                            )
                    if rows:
                        metric_df = pd.DataFrame(rows).set_index("variable")
                    else:
                        metric_df = pd.DataFrame([metrics]).T.rename(columns={0: "value"})
                    st.dataframe(metric_df, width="stretch")
                except Exception as exc:
                    st.error(f"LE-PINN validation failed: {exc}")

            st.divider()
            render_pareto_le_pinn_validation_panel(engine)

    with tab_opt:
        st.subheader("Quick Optimization Trials")
        st.caption("Runs a lightweight 4-objective search and saves results to outputs/results/optimization_results_dashboard.csv")

        n_trials = st.select_slider("Number of trials", options=[10, 50, 100], value=10)
        run_opt = st.button("Run Optimization Sweep")

        if run_opt:
            with st.spinner(f"Running optimization ({n_trials} trials)…"):
                try:
                    df_opt, summary = run_quick_optimization(engine, n_trials=n_trials)
                    st.session_state["opt_df"] = df_opt
                    st.session_state["opt_summary"] = summary
                except Exception as exc:
                    st.error(f"Optimization failed: {exc}")

        if "opt_df" in st.session_state and "opt_summary" in st.session_state:
            df_opt = st.session_state["opt_df"]
            summary = st.session_state["opt_summary"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Completed Trials", f"{summary['completed_trials']}")
            c2.metric("Pareto Points", f"{summary['pareto_points']}")
            c3.metric("Best TSFC", f"{summary['best_tsfc']:.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Best Thrust", f"{summary['best_thrust']:.2f} kN")
            c5.metric("Lowest CO₂", f"{summary['lowest_co2']:.2f} g/s")
            c6.metric("Lowest NOx", f"{summary['lowest_nox']:.2f} g/s")

            st.dataframe(df_opt, width="stretch")
            if not df_opt.empty:
                st.scatter_chart(
                    df_opt,
                    x="CO2",
                    y="Thrust_kN",
                    color="ParetoOptimal",
                    size="SAF_Total",
                )


if __name__ == "__main__":
    main()
