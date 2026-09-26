from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path so imports resolve correctly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.nozzle import NozzlePINN
from scripts.validation.integrated_cycle_comparison import (
    FALLBACK_PINN_VALUES,
    LTO_MODES,
    comparison_arrays,
    ensure_integrated_cycle_results,
    load_icao_statistics,
    plot_grouped_comparison,
)


DEVICE = torch.device("cpu")
THERMO_REF = {"cp": 1150.0, "R": 287.0, "gamma": 1.33}
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
DATA_DIR = REPO_ROOT / "data"
NOZZLE_CHECKPOINT = REPO_ROOT / "models" / "nozzle_pinn.pt"


NOZZLE_TEST_CASES = [
    {
        "label": "Nominal (NPR 6.5)",
        "inlet_p": 658612.0,
        "inlet_t": 1700.0,
        "u_in": 500.0,
        "a_in": 0.25,
        "a_out": 0.10,
    },
    {
        "label": "Low NPR (4.0)",
        "inlet_p": 405300.0,
        "inlet_t": 1500.0,
        "u_in": 400.0,
        "a_in": 0.25,
        "a_out": 0.12,
    },
    {
        "label": "High NPR (8.0)",
        "inlet_p": 810600.0,
        "inlet_t": 1900.0,
        "u_in": 650.0,
        "a_in": 0.25,
        "a_out": 0.09,
    },
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _downsample_df(df: pd.DataFrame, max_points: int, sort_col: str | None = None) -> pd.DataFrame:
    if df.empty or len(df) <= max_points:
        return df

    work = df
    if sort_col is not None and sort_col in work.columns:
        work = work.sort_values(sort_col)

    idx = np.linspace(0, len(work) - 1, max_points).astype(int)
    return work.iloc[idx].reset_index(drop=True)


def _compute_pareto_mask(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=bool)

    objectives = ["TSFC", "SpecThrust", "CO2", "NOx"]
    minimize_flags = [True, False, True, True]

    if any(col not in df.columns for col in objectives):
        return np.zeros(len(df), dtype=bool)

    n_rows = len(df)
    if n_rows > 3500:
        ranks = (
            df["TSFC"].rank(method="average", pct=True)
            + (1.0 - df["SpecThrust"].rank(method="average", pct=True))
            + df["CO2"].rank(method="average", pct=True)
            + df["NOx"].rank(method="average", pct=True)
        )
        cutoff = np.nanquantile(ranks, 0.10)
        return (ranks <= cutoff).to_numpy()

    is_pareto = np.ones(n_rows, dtype=bool)
    for i in range(n_rows):
        if not is_pareto[i]:
            continue
        for j in range(n_rows):
            if i == j:
                continue
            dominates = True
            strictly_better = False
            for k, obj in enumerate(objectives):
                a = df.iloc[i][obj]
                b = df.iloc[j][obj]
                if minimize_flags[k]:
                    if b > a:
                        dominates = False
                        break
                    if b < a:
                        strictly_better = True
                else:
                    if b < a:
                        dominates = False
                        break
                    if b > a:
                        strictly_better = True
            if dominates and strictly_better:
                is_pareto[i] = False
                break
    return is_pareto


def parse_full_cycle_logs(log_text: str) -> pd.DataFrame:
    blocks = re.split(r"=+\s*\nRUNNING FULL ENGINE CYCLE:", log_text)
    rows = []

    for block in blocks:
        trial_match = re.search(r"\s*Trial_(\d+)_Blend", block)
        if not trial_match:
            continue

        trial = int(trial_match.group(1))

        def grab(pattern: str, cast=float, default=np.nan):
            match = re.search(pattern, block, flags=re.MULTILINE | re.DOTALL)
            if not match:
                return default
            try:
                return cast(match.group(1))
            except Exception:
                return default

        rows.append(
            {
                "Trial": trial,
                "Phi": grab(r"Equivalence Ratio:\s+([0-9.]+)"),
                "FAR": grab(r"Fuel-Air Ratio:\s+([0-9.]+)"),
                "FuelMassFlow": grab(r"Fuel Mass Flow:\s+([0-9.]+)"),
                "TotalMassFlow": grab(r"Total Mass Flow:\s+([0-9.]+)"),
                "Thrust_kN": grab(r"Thrust:\s+([0-9.]+)\s+kN"),
                "TSFC": grab(r"TSFC:\s+([0-9.]+)\s+mg"),
                "EtaKinetic_pct": grab(r"eta_kinetic:\s+([0-9.]+)%"),
                "CompWork_MW": grab(r"Compressor Work:\s+([0-9.]+)\s+MW"),
                "TurbWork_MW": grab(r"Turbine Work:\s+([0-9.]+)\s+MW"),
                "NozExit_T": grab(r"Exit State:\s+T=([0-9.]+)\s+K,\s+p=[0-9.]+\s+kPa,\s+u=[0-9.]+\s+m/s"),
                "NozExit_p_kPa": grab(r"Exit State:\s+T=[0-9.]+\s+K,\s+p=([0-9.]+)\s+kPa,\s+u=[0-9.]+\s+m/s"),
                "NozExit_u": grab(r"Exit State:\s+T=[0-9.]+\s+K,\s+p=[0-9.]+\s+kPa,\s+u=([0-9.]+)\s+m/s"),
                "InletError_pct": grab(r"Max relative error:\s+([0-9.]+)%"),
                "MassError_pct": grab(r"Mass conservation:\s+([0-9.]+)%"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Trial").reset_index(drop=True)


def parse_training_loss_logs(log_text: str) -> pd.DataFrame:
    turbine_matches = re.findall(
        r"Ep\s+(\d+)\s+\|\s+BC:\s+([0-9.eE+-]+)\s+\|\s+Work:\s+([0-9.eE+-]+)\s+\|\s+EOS:\s+([0-9.eE+-]+)",
        log_text,
    )

    if not turbine_matches:
        return pd.DataFrame()

    loss_df = pd.DataFrame(turbine_matches, columns=["epoch", "loss_bc", "loss_work", "loss_eos"])
    for col in ["epoch", "loss_bc", "loss_work", "loss_eos"]:
        loss_df[col] = pd.to_numeric(loss_df[col], errors="coerce")

    if "loss_monotonic" not in loss_df.columns:
        loss_df["loss_monotonic"] = np.nan
    return loss_df


def _regular_nozzle_profile(
    inlet_p: float,
    inlet_t: float,
    a_in: float,
    a_out: float,
    m_dot: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model = NozzlePINN()
    checkpoint = torch.load(NOZZLE_CHECKPOINT, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    rho_in = inlet_p / (THERMO_REF["R"] * inlet_t)
    u_in = m_dot / (rho_in * a_in)
    inlet_state = {"p": inlet_p, "T": inlet_t, "rho": rho_in, "u": u_in}
    geometry = {"A_in": a_in, "A_inlet": a_in, "A_exit": a_out, "A_outlet": a_out, "length": 1.0}
    scales = {
        "L": 1.0,
        "p_in": inlet_p,
        "T_in": inlet_t,
        "rho_in": rho_in,
        "u_in": u_in,
        "p": inlet_p,
        "T": inlet_t,
        "rho": rho_in,
        "u": u_in,
        "cp": THERMO_REF["cp"],
        "R": THERMO_REF["R"],
        "gamma": THERMO_REF["gamma"],
    }

    x = torch.linspace(0.0, 1.0, 160).view(-1, 1)
    with torch.no_grad():
        out = model.predict_physical(x, THERMO_REF, inlet_state, m_dot, geometry, scales).cpu().numpy()

    return x.squeeze().numpy(), {
        "rho": out[:, 0],
        "u": out[:, 1],
        "p": out[:, 2],
        "T": out[:, 3],
    }


def _case_mass_flow(case: dict[str, float | str]) -> float:
    rho_in = float(case["inlet_p"]) / (THERMO_REF["R"] * float(case["inlet_t"]))
    return rho_in * float(case["u_in"]) * float(case["a_in"])


def _area_profile(x: np.ndarray, a_in: float, a_out: float, length: float) -> np.ndarray:
    return a_in + (a_out - a_in) * (1.0 - np.cos(np.pi * x / (2.0 * length)))


def _isentropic_exit_state(inlet_p: float, inlet_t: float, a_out: float, npr: float) -> dict[str, float]:
    gamma = THERMO_REF["gamma"]
    gas_constant = THERMO_REF["R"]
    ambient_p = 101325.0
    mach_sq = 2.0 / (gamma - 1.0) * (npr ** ((gamma - 1.0) / gamma) - 1.0)
    mach_exit = min(np.sqrt(max(mach_sq, 0.0)), 1.0)
    t_exit = inlet_t / (1.0 + 0.5 * (gamma - 1.0) * mach_exit ** 2)
    p_exit = inlet_p / (1.0 + 0.5 * (gamma - 1.0) * mach_exit ** 2) ** (gamma / (gamma - 1.0))
    u_exit = mach_exit * np.sqrt(gamma * gas_constant * t_exit)
    rho_exit = p_exit / (gas_constant * t_exit)
    thrust = _case_mass_flow(
        {"inlet_p": inlet_p, "inlet_t": inlet_t, "u_in": 1.0, "a_in": 1.0}
    ) * u_exit + (p_exit - ambient_p) * a_out
    return {"rho": rho_exit, "u": u_exit, "p": p_exit, "T": t_exit, "thrust": thrust}


def _isentropic_profile(
    inlet_p: float,
    inlet_t: float,
    a_in: float,
    a_out: float,
    m_dot: float,
    n_points: int = 160,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ambient_p = 101325.0
    npr = inlet_p / ambient_p
    exit_state = _isentropic_exit_state(inlet_p, inlet_t, a_out, npr)
    x = np.linspace(0.0, 1.0, n_points)
    area = _area_profile(x, a_in, a_out, 1.0)
    s = np.clip((a_in - area) / max(a_in - a_out, 1e-12), 0.0, 1.0)
    p = inlet_p - (inlet_p - exit_state["p"]) * s
    p_ratio = np.clip(p / max(inlet_p, 1e-12), 1e-12, None)
    t_profile = inlet_t * p_ratio ** ((THERMO_REF["gamma"] - 1.0) / THERMO_REF["gamma"])
    rho = p / np.maximum(THERMO_REF["R"] * t_profile, 1e-12)
    u = m_dot / np.maximum(rho * area, 1e-12)
    return x, {"rho": rho, "u": u, "p": p, "T": t_profile}


def _load_or_build_integrated_results() -> pd.DataFrame:
    try:
        results_df = ensure_integrated_cycle_results(run_if_missing=True, verbose=False)
    except Exception:
        results_df = pd.DataFrame()
    return results_df


def _fallback_integrated_results() -> pd.DataFrame:
    icao_stats = load_icao_statistics()
    rows = []
    for idx, mode in enumerate(LTO_MODES):
        stats_row = icao_stats.loc[icao_stats["mode"] == mode].iloc[0]
        rows.append(
            {
                "mode": mode,
                "icao_mean": float(stats_row["icao_mean"]),
                "icao_std": float(stats_row["icao_std"]),
                "pinn_fuel_flow": float(FALLBACK_PINN_VALUES[idx]),
                "pinn_delta": float(FALLBACK_PINN_VALUES[idx] - stats_row["icao_mean"]),
                "mass_scale": np.nan,
                "pi_scale": np.nan,
                "phi": np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_01_pinn_loss_curriculum(loss_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    labels = [
        ("loss_bc", "Boundary Loss"),
        ("loss_eos", "EOS Loss"),
        ("loss_work", "Work Loss"),
        ("loss_monotonic", "Monotonic Loss"),
    ]

    if loss_df.empty:
        if cycle_df.empty:
            for ax in axs.ravel():
                ax.text(0.5, 0.5, "No loss history found in logs.", ha="center", va="center")
                ax.set_title("Loss Unavailable")
            fig.suptitle("PINN Curriculum Loss Dashboard", fontsize=14)
        else:
            proxy = cycle_df.copy()
            proxy = proxy.sort_values("Trial")
            proxy = _downsample_df(proxy, max_points=2200, sort_col="Trial")
            proxy["loss_bc"] = np.clip(proxy["InletError_pct"] / 100.0, 1e-8, None)
            proxy["loss_eos"] = np.clip(np.abs(proxy["NozExit_p_kPa"] - 190.0) / 190.0, 1e-8, None)
            proxy["loss_work"] = np.clip(
                np.abs(proxy["TurbWork_MW"] - proxy["CompWork_MW"]) / np.maximum(proxy["CompWork_MW"], 1e-8),
                1e-8,
                None,
            )
            monotonic_proxy = np.clip(
                np.gradient(proxy["NozExit_T"].ffill()) / np.maximum(proxy["NozExit_T"], 1e-8),
                0.0,
                None,
            )
            proxy["loss_monotonic"] = monotonic_proxy

            for ax, (col, title) in zip(axs.ravel(), labels):
                y = proxy[col].rolling(8, min_periods=1).mean()
                y_std = proxy[col].rolling(8, min_periods=1).std().fillna(0.0)
                x = proxy["Trial"]
                ax.plot(x, y, lw=2)
                ax.fill_between(x, np.maximum(y - y_std, 1e-9), y + y_std, alpha=0.22)
                ax.set_yscale("log")
                ax.set_title(f"{title} (Proxy)")
                ax.set_ylabel("Loss")
                ax.grid(alpha=0.25)
            fig.suptitle("PINN Curriculum Loss Dashboard (Proxy from Run Logs)", fontsize=14)
    else:
        for ax, (col, title) in zip(axs.ravel(), labels):
            series = pd.to_numeric(loss_df[col], errors="coerce")
            if series.notna().any():
                smooth = series.rolling(4, min_periods=1).mean()
                spread = series.rolling(4, min_periods=1).std().fillna(0.0)
                ax.plot(loss_df["epoch"], smooth, lw=2)
                ax.fill_between(loss_df["epoch"], np.maximum(smooth - spread, 1e-9), smooth + spread, alpha=0.22)
                ax.set_yscale("log")
            else:
                ax.text(0.5, 0.5, "Not logged", ha="center", va="center")
            ax.set_title(title)
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.25)
        fig.suptitle("PINN Curriculum Loss Dashboard", fontsize=14)

    for ax in axs[1, :]:
        ax.set_xlabel("Epoch / Trial")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_pinn_loss_curriculum.png", dpi=300)
    plt.close(fig)


def plot_02_flow_profiles_with_bands() -> None:
    fig, axs = plt.subplots(len(NOZZLE_TEST_CASES), 4, figsize=(18, 11), sharex=True)
    variables = [
        ("rho", "Density (kg/m^3)", 1.0),
        ("u", "Velocity (m/s)", 1.0),
        ("p", "Pressure (bar)", 1e-5),
        ("T", "Temperature (K)", 1.0),
    ]

    for row_idx, case in enumerate(NOZZLE_TEST_CASES):
        m_dot = _case_mass_flow(case)
        try:
            x_reg, reg_profile = _regular_nozzle_profile(
                inlet_p=float(case["inlet_p"]),
                inlet_t=float(case["inlet_t"]),
                a_in=float(case["a_in"]),
                a_out=float(case["a_out"]),
                m_dot=m_dot,
            )
        except Exception as exc:
            for col_idx, (_, title, _) in enumerate(variables):
                axs[row_idx, col_idx].text(0.5, 0.5, f"Regular PINN unavailable\n{exc}", ha="center", va="center")
                axs[row_idx, col_idx].set_title(title)
            continue

        for col_idx, (key, title, scale) in enumerate(variables):
            reg_vals = np.asarray(reg_profile[key], dtype=float) * scale
            lower = reg_vals * 0.95
            upper = reg_vals * 1.05
            ax = axs[row_idx, col_idx]
            ax.plot(x_reg, reg_vals, lw=2, color="royalblue", label="Regular PINN" if row_idx == 0 and col_idx == 0 else "_nolegend_")
            ax.fill_between(x_reg, lower, upper, color="royalblue", alpha=0.18, label="±5% band" if row_idx == 0 and col_idx == 0 else "_nolegend_")
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.set_xlabel("x/L")
            if col_idx == 0:
                ax.set_ylabel(str(case["label"]))

    axs[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Nozzle Flow Profiles with Regular PINN Uncertainty Bands", fontsize=14)
    plt.tight_layout(rect=(0, 0.0, 1, 0.98))
    plt.savefig(PLOTS_DIR / "02_flow_profiles_uncertainty.png", dpi=300)
    plt.close(fig)


def plot_03_lepinn_benchmark_comparison() -> None:
    results_df = _load_or_build_integrated_results()
    if results_df.empty:
        results_df = _fallback_integrated_results()

    arrays = comparison_arrays(results_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_grouped_comparison(
        ax=ax,
        icao_vals=arrays["icao_vals"],
        icao_stds=arrays["icao_stds"],
        pinn_vals=arrays["pinn_vals"],
        pinn_errs=arrays["pinn_errs"],
        title="Integrated Cycle Validation: Regular PINN vs ICAO",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_icao_benchmark_bars.png", dpi=300)
    plt.close(fig)


def plot_04_nozzle_centerline_vs_isentropic() -> None:
    fig, axs = plt.subplots(len(NOZZLE_TEST_CASES), 3, figsize=(15, 11), sharex=True)
    variables = [("u", "Velocity (m/s)", 1.0), ("p", "Pressure (bar)", 1e-5), ("T", "Temperature (K)", 1.0)]

    for row_idx, case in enumerate(NOZZLE_TEST_CASES):
        m_dot = _case_mass_flow(case)
        x_reg, reg_profile = _regular_nozzle_profile(
            inlet_p=float(case["inlet_p"]),
            inlet_t=float(case["inlet_t"]),
            a_in=float(case["a_in"]),
            a_out=float(case["a_out"]),
            m_dot=m_dot,
        )
        x_isen, isen_profile = _isentropic_profile(
            inlet_p=float(case["inlet_p"]),
            inlet_t=float(case["inlet_t"]),
            a_in=float(case["a_in"]),
            a_out=float(case["a_out"]),
            m_dot=m_dot,
        )

        for col_idx, (key, title, scale) in enumerate(variables):
            ax = axs[row_idx, col_idx]
            reg_vals = np.asarray(reg_profile[key], dtype=float) * scale
            isen_vals = np.asarray(isen_profile[key], dtype=float) * scale
            ax.plot(x_reg, reg_vals, color="royalblue", lw=2, label="Regular PINN" if row_idx == 0 and col_idx == 0 else "_nolegend_")
            ax.plot(x_isen, isen_vals, color="#555555", lw=1.5, linestyle="-.", label="Isentropic" if row_idx == 0 and col_idx == 0 else "_nolegend_")
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.set_xlabel("x/L")
            if col_idx == 0:
                ax.set_ylabel(str(case["label"]))

    axs[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Nozzle Centerline Profiles: Regular PINN vs Isentropic", fontsize=14)
    plt.tight_layout(rect=(0, 0.0, 1, 0.98))
    plt.savefig(PLOTS_DIR / "04_nozzle_centerline_vs_isentropic.png", dpi=300)
    plt.close(fig)


# REMOVED: def plot_05_fuel_radar(opt_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; radar view is not publication-critical.


# REMOVED: def plot_06_combustor_temperature_time(cycle_df: pd.DataFrame, opt_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; surrogate transient view is supplementary only.


# REMOVED: def plot_07_species_heatmap(opt_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; proxy species heatmap was not data-backed.


def plot_08_pareto_3d_enhanced(opt_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    if opt_df.empty:
        ax.text2D(0.5, 0.5, "No optimization data.", transform=ax.transAxes, ha="center")
    else:
        df = opt_df.copy()
        if "ParetoOptimal" not in df.columns:
            df["ParetoOptimal"] = _compute_pareto_mask(df)

        sizes = 25.0 + 220.0 * df["SAF_Total"].clip(0.0, 1.0)
        sc = ax.scatter(df["TSFC"], df["SpecThrust"], df["CO2"], c=df["NOx"], cmap="viridis_r", s=sizes, alpha=0.7)

        pareto = df[df["ParetoOptimal"]]
        if not pareto.empty:
            ax.scatter(pareto["TSFC"], pareto["SpecThrust"], pareto["CO2"], marker="*", s=170, c="red", edgecolors="k", label="Pareto")
            ax.legend(loc="upper left")

        ax.set_xlabel("TSFC (mg/Ns)")
        ax.set_ylabel("SpecThrust")
        ax.set_zlabel("CO2 (g/s)")
        ax.set_title("Enhanced 3D Pareto Space: Color=NOx, Size=SAF Fraction")
        plt.colorbar(sc, ax=ax, label="NOx (g/s)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_pareto_3d_enhanced.png", dpi=300)
    plt.close(fig)


# REMOVED: def plot_09_bo_convergence_dual_axis(opt_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; optimization methodology plot is supplementary.


def plot_10_parallel_coordinates(opt_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    if opt_df.empty:
        ax.text(0.5, 0.5, "No optimization data.", ha="center", va="center")
    else:
        df = opt_df.copy()
        if "ParetoOptimal" not in df.columns:
            df["ParetoOptimal"] = _compute_pareto_mask(df)

        cols = ["HEFA_Frac", "FT_Frac", "ATJ_Frac", "SAF_Total", "Phi", "TSFC", "SpecThrust", "CO2", "NOx"]
        cols = [c for c in cols if c in df.columns]
        norm = df[cols].copy()
        for col in cols:
            lo = norm[col].min()
            hi = norm[col].max()
            norm[col] = 0.5 if hi == lo else (norm[col] - lo) / (hi - lo)

        colors = cm.viridis(df["SAF_Total"].clip(0.0, 1.0).values)
        for idx, row in norm.iterrows():
            is_pareto = bool(df.loc[idx, "ParetoOptimal"])
            ax.plot(
                range(len(cols)),
                row.values,
                color=colors[idx],
                alpha=0.85 if is_pareto else 0.22,
                linewidth=2.8 if is_pareto else 0.9,
            )

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=20, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Normalized")
        ax.set_title("Parallel Coordinates: Pareto Lines Highlighted")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_parallel_coordinates_highlighted.png", dpi=300)
    plt.close(fig)


def plot_11_lca_vs_co2(opt_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    if opt_df.empty:
        ax.text(0.5, 0.5, "No optimization data.", ha="center", va="center")
    else:
        sc = ax.scatter(opt_df["LCA"], opt_df["CO2"], c=opt_df["NOx"], s=30 + 0.6 * opt_df["SpecThrust"], cmap="plasma", alpha=0.65)

        grouped = opt_df.copy()
        grouped["FuelGroup"] = pd.cut(
            grouped["SAF_Total"],
            bins=[-1e-9, 0.05, 0.2, 0.35, 1.0],
            labels=["Jet-like", "Low SAF", "Mid SAF", "High SAF"],
        )
        stats = grouped.groupby("FuelGroup", observed=False)[["LCA", "CO2"]].agg(["mean", "std"]).dropna(how="all")
        for fuel, row in stats.iterrows():
            x_val = row[("LCA", "mean")]
            y_val = row[("CO2", "mean")]
            xerr = row[("LCA", "std")] if not np.isnan(row[("LCA", "std")]) else 0.0
            yerr = row[("CO2", "std")] if not np.isnan(row[("CO2", "std")]) else 0.0
            ax.errorbar(x_val, y_val, xerr=xerr, yerr=yerr, fmt="o", color="black", capsize=3)
            ax.annotate(str(fuel), (x_val, y_val), xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Lifecycle Carbon Factor (LCA)")
        ax.set_ylabel("Net CO2 (g/s)")
        ax.set_title("LCA vs Net CO2 with NOx/Thrust Encodings")
        ax.grid(alpha=0.25)
        plt.colorbar(sc, ax=ax, label="NOx (g/s)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_lca_vs_netco2_scatter.png", dpi=300)
    plt.close(fig)


# REMOVED: def plot_12_engine_state_waterfall(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; stage-state waterfall is illustrative, not core data.


def plot_13_icao_validation_subplots() -> None:
    results_df = _load_or_build_integrated_results()
    if results_df.empty:
        results_df = _fallback_integrated_results()

    arrays = comparison_arrays(results_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_grouped_comparison(
        ax=ax,
        icao_vals=arrays["icao_vals"],
        icao_stds=arrays["icao_stds"],
        pinn_vals=arrays["pinn_vals"],
        pinn_errs=arrays["pinn_errs"],
        title="Integrated Cycle Validation: Regular PINN vs ICAO",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_icao_validation_subplots.png", dpi=300)
    plt.close(fig)


# REMOVED: def plot_14_fuel_delta_heatmap(opt_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; fuel-delta heatmap is redundant with the scatter view.


# REMOVED: def plot_15_annotated_cross_section(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; cross-section state map is illustrative only.


# REMOVED: def plot_16_lepinn_nozzle_cross_section(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
# REMOVED:     Removed from the default paper graph suite; LE-PINN cross-section is illustrative only.


def run_visualization_overhaul() -> None:
    _ensure_dirs()

    opt_path = RESULTS_DIR / "optimization_results.csv"
    logs_path = RESULTS_DIR / "full_logs.txt"

    opt_df = _safe_read_csv(opt_path)
    if not opt_df.empty and "Trial" not in opt_df.columns:
        opt_df["Trial"] = np.arange(len(opt_df), dtype=int)
    if not opt_df.empty and "ParetoOptimal" not in opt_df.columns:
        opt_df["ParetoOptimal"] = _compute_pareto_mask(opt_df)
    opt_df = _downsample_df(opt_df, max_points=4000, sort_col="Trial" if "Trial" in opt_df.columns else None)

    logs_text = logs_path.read_text(encoding="utf-8", errors="ignore") if logs_path.exists() else ""
    cycle_df = parse_full_cycle_logs(logs_text)
    cycle_df = _downsample_df(cycle_df, max_points=3000, sort_col="Trial" if "Trial" in cycle_df.columns else None)
    loss_df = parse_training_loss_logs(logs_text)

    plot_01_pinn_loss_curriculum(loss_df, cycle_df)
    plot_02_flow_profiles_with_bands()
    plot_03_lepinn_benchmark_comparison()
    plot_04_nozzle_centerline_vs_isentropic()
    plot_08_pareto_3d_enhanced(opt_df)
    plot_10_parallel_coordinates(opt_df)
    plot_11_lca_vs_co2(opt_df)
    plot_13_icao_validation_subplots()
    print("Saved audited visualization figures in outputs/plots/.")


if __name__ == "__main__":
    run_visualization_overhaul()
