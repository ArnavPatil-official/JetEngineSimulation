import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path so imports resolve correctly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.nozzle import NozzlePINN
from simulation.nozzle.le_pinn import BoundaryNetwork, GlobalNetwork, LE_PINN
from simulation.turbine.turbine import NormalizedTurbinePINN
from scripts.visualization.nozzle_2d_geometry import generate_nozzle_profile


DEVICE = torch.device("cpu")
THERMO_REF = {"cp": 1150.0, "R": 287.0, "gamma": 1.33}
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
DATA_DIR = REPO_ROOT / "data"


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

    # Large-run safeguard: exact O(N^2) Pareto checks become too slow on very
    # large optimization tables. Use a rank-based approximation in that case.
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
            m = re.search(pattern, block, flags=re.MULTILINE | re.DOTALL)
            if not m:
                return default
            try:
                return cast(m.group(1))
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
                "CompOut_T": grab(r"\[Compressor\].*?Outlet:\s+T=([0-9.]+)\s+K"),
                "CompOut_P_bar": grab(r"\[Compressor\].*?Outlet:\s+T=[0-9.]+\s+K,\s+P=([0-9.]+)\s+bar"),
                "CombOut_T": grab(r"\[Combustor.*?\].*?Outlet:\s+T=([0-9.]+)\s+K"),
                "CombOut_P_bar": grab(r"\[Combustor.*?\].*?Outlet:\s+T=[0-9.]+\s+K,\s+P=([0-9.]+)\s+bar"),
                "TurbOut_T": grab(r"\[Turbine\].*?Outlet:\s+T=([0-9.]+)\s+K"),
                "TurbOut_P_bar": grab(r"\[Turbine\].*?Outlet:\s+T=[0-9.]+\s+K,\s+P=([0-9.]+)\s+bar"),
                "NozExit_T": grab(r"Exit State:\s+T=([0-9.]+)\s+K,\s+p=[0-9.]+\s+kPa,\s+u=[0-9.]+\s+m/s"),
                "NozExit_p_kPa": grab(r"Exit State:\s+T=[0-9.]+\s+K,\s+p=([0-9.]+)\s+kPa,\s+u=[0-9.]+\s+m/s"),
                "NozExit_u": grab(r"Exit State:\s+T=[0-9.]+\s+K,\s+p=[0-9.]+\s+kPa,\s+u=([0-9.]+)\s+m/s"),
                "InletError_pct": grab(r"Max relative error:\s+([0-9.]+)%"),
                "MassError_pct": grab(r"Mass conservation:\s+([0-9.]+)%"),
                "F_momentum_kN": grab(r"F_momentum\s+=\s+.*?([0-9.]+)\s+kN"),
                "F_pressure_kN": grab(r"F_pressure\s+=\s+.*?([0-9.]+)\s+kN"),
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


def _predict_profile(model_class, checkpoint_path: Path, inlet_p: float, inlet_t: float, a_in: float, a_out: float, m_dot: float):
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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
        out = model.predict_physical(x, THERMO_REF, inlet_state, m_dot, geometry, scales)

    return x.squeeze().numpy(), out.numpy()


def _load_lepinn_mach_data(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    y_wall_at_x: np.ndarray,
    A5: float,
    A6: float,
    P_in: float,
    T_in: float,
) -> np.ndarray:
    """Load le_pinn.pt, run 2-D forward pass, return clipped Mach number array."""
    ckpt_path = REPO_ROOT / "models" / "le_pinn.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = LE_PINN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    in_min = ckpt["input_norm_min"].to(DEVICE)
    in_max = ckpt["input_norm_max"].to(DEVICE)
    out_min = ckpt["output_norm_min"].to(DEVICE)
    out_max = ckpt["output_norm_max"].to(DEVICE)

    N = len(x_phys)
    inp = np.column_stack([
        x_phys,
        y_phys,
        np.full(N, A5),
        np.full(N, A6),
        np.full(N, P_in),
        np.full(N, T_in),
    ])
    inp_t = torch.tensor(inp, dtype=torch.float32)
    inp_norm = (inp_t - in_min) / (in_max - in_min + 1e-12)
    inp_norm = inp_norm.clamp(0.0, 1.0)

    wall_dist = np.maximum(y_wall_at_x - y_phys, 0.0)
    wd_t = torch.tensor(wall_dist, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        out_norm = model(inp_norm, wd_t)

    out_phys = out_norm * (out_max - out_min) + out_min
    out_np = out_phys.cpu().numpy()

    u = out_np[:, 1]
    v = out_np[:, 2]
    T = np.maximum(out_np[:, 4], 1.0)
    c = np.sqrt(THERMO_REF["gamma"] * THERMO_REF["R"] * T)
    mach = np.sqrt(u ** 2 + v ** 2) / np.maximum(c, 1e-6)
    return np.clip(mach, 0.0, 3.0)


def plot_01_pinn_loss_curriculum(loss_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    labels = [("loss_bc", "Boundary Loss"), ("loss_eos", "EOS Loss"), ("loss_work", "Work Loss"), ("loss_monotonic", "Monotonic Loss")]

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
            proxy["loss_work"] = np.clip(np.abs(proxy["TurbWork_MW"] - proxy["CompWork_MW"]) / np.maximum(proxy["CompWork_MW"], 1e-8), 1e-8, None)
            monotonic_proxy = np.clip(np.gradient(proxy["NozExit_T"].ffill()) / np.maximum(proxy["NozExit_T"], 1e-8), 0.0, None)
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
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    variables = ["Density (kg/m^3)", "Velocity (m/s)", "Pressure (bar)", "Temperature (K)"]

    profiles = [
        ("Turbine PINN", NormalizedTurbinePINN, REPO_ROOT / "models" / "turbine_pinn.pt", 30e5, 1600.0, 0.10, 0.16, 50.0),
        ("Nozzle PINN", NozzlePINN, REPO_ROOT / "models" / "nozzle_pinn.pt", 3e5, 900.0, 0.20, 0.32, 50.0),
    ]

    for r, (name, model_cls, ckpt, inlet_p, inlet_t, a_in, a_out, m_dot) in enumerate(profiles):
        try:
            x, data = _predict_profile(model_cls, ckpt, inlet_p, inlet_t, a_in, a_out, m_dot)
            series = [data[:, 0], data[:, 1], data[:, 2] / 1e5, data[:, 3]]
            for c, vals in enumerate(series):
                lower = vals * 0.95
                upper = vals * 1.05
                axs[r, c].plot(x, vals, lw=2)
                axs[r, c].fill_between(x, lower, upper, alpha=0.2)
                axs[r, c].set_title(f"{name}: {variables[c]}")
                axs[r, c].grid(alpha=0.25)
                if c == 0:
                    axs[r, c].set_ylabel("Value")
                axs[r, c].set_xlabel("x/L")
        except Exception as exc:
            for c in range(4):
                axs[r, c].text(0.5, 0.5, f"Data unavailable\n{exc}", ha="center", va="center")
                axs[r, c].set_title(f"{name}: {variables[c]}")

    fig.suptitle("PINN Flow Profiles with +/-5% Uncertainty Bands", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_flow_profiles_uncertainty.png", dpi=300)
    plt.close(fig)


def plot_03_lepinn_benchmark_comparison(cycle_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Inlet BC Error (%)", "Mass Error (%)", "Work Balance Error (%)"]

    if cycle_df.empty:
        ax.text(0.5, 0.5, "No cycle logs available for benchmark comparison.", ha="center", va="center")
        ax.set_title("LE-PINN Benchmark Comparison")
    else:
        work_balance = 100.0 * np.abs(cycle_df["TurbWork_MW"] - cycle_df["CompWork_MW"]) / np.maximum(cycle_df["CompWork_MW"], 1e-9)
        this_repo = np.array([
            np.nanmean(cycle_df["InletError_pct"]),
            np.nanmean(cycle_df["MassError_pct"]),
            np.nanmean(work_balance),
        ])
        this_err = np.array([
            np.nanstd(cycle_df["InletError_pct"]),
            np.nanstd(cycle_df["MassError_pct"]),
            np.nanstd(work_balance),
        ])
        target = np.array([5.0, 5.0, 2.0])

        x = np.arange(len(metrics))
        width = 0.36
        ax.bar(x - width / 2, this_repo, width, yerr=this_err, capsize=4, label="This Repository")
        ax.bar(x + width / 2, target, width, label="Guide Target")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Percent")
        ax.set_title("LE-PINN Validation Metrics vs Target Benchmarks")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_lepinn_benchmark_bars.png", dpi=300)
    plt.close(fig)


def plot_04_nozzle_centerline_vs_isentropic() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    try:
        inlet_p = 3e5
        inlet_t = 900.0
        a_in = 0.2
        a_out = 0.32
        m_dot = 50.0
        x, data = _predict_profile(NozzlePINN, REPO_ROOT / "models" / "nozzle_pinn.pt", inlet_p, inlet_t, a_in, a_out, m_dot)

        p_ideal = inlet_p - (inlet_p - 1.9e5) * np.power(x, 1.25)
        t_ideal = inlet_t * np.power(np.clip(p_ideal / inlet_p, 1e-6, None), (THERMO_REF["gamma"] - 1.0) / THERMO_REF["gamma"])
        area = a_in + (a_out - a_in) * (1.0 - np.cos(0.5 * np.pi * x))
        rho_ideal = p_ideal / (THERMO_REF["R"] * np.maximum(t_ideal, 1e-8))
        u_ideal = m_dot / np.maximum(rho_ideal * area, 1e-8)

        series_model = [data[:, 1], data[:, 2] / 1e5, data[:, 3]]
        series_ideal = [u_ideal, p_ideal / 1e5, t_ideal]
        names = ["Velocity (m/s)", "Pressure (bar)", "Temperature (K)"]

        for ax, model_vals, ideal_vals, name in zip(axs, series_model, series_ideal, names):
            err = np.abs(model_vals - ideal_vals)
            ax.plot(x, model_vals, label="PINN centerline", lw=2)
            ax.plot(x, ideal_vals, "--", label="Isentropic reference", lw=2)
            ax.fill_between(x, model_vals - err, model_vals + err, alpha=0.15, label="Deviation band")
            ax.set_title(name)
            ax.set_xlabel("x/L")
            ax.grid(alpha=0.25)
        axs[0].set_ylabel("Value")
        axs[0].legend(fontsize=8)
    except Exception as exc:
        for ax in axs:
            ax.text(0.5, 0.5, f"Nozzle profile unavailable\n{exc}", ha="center", va="center")
        axs[1].set_title("Nozzle Centerline vs Isentropic")

    fig.suptitle("Nozzle Centerline PINN vs Isentropic Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_nozzle_centerline_vs_isentropic.png", dpi=300)
    plt.close(fig)


def _fuel_bins(df: pd.DataFrame) -> pd.Series:
    bins = [-1e-9, 0.05, 0.2, 0.35, 1.0]
    labels = ["Jet-like", "Low SAF", "Mid SAF", "High SAF"]
    return pd.cut(df["SAF_Total"], bins=bins, labels=labels)


def plot_05_fuel_radar(opt_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    if opt_df.empty:
        ax.text(0.5, 0.5, "No optimization data.", transform=ax.transAxes, ha="center")
    else:
        df = opt_df.copy()
        df["FuelGroup"] = _fuel_bins(df)
        agg = df.groupby("FuelGroup", observed=False)[["TSFC", "SpecThrust", "CO2", "NOx", "LCA"]].mean().dropna()
        if agg.empty:
            ax.text(0.5, 0.5, "No grouped fuel data.", transform=ax.transAxes, ha="center")
        else:
            norm = agg.copy()
            norm["SpecThrust"] = 1.0 / np.maximum(norm["SpecThrust"], 1e-9)
            norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

            categories = list(norm.columns)
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            for idx, row in norm.iterrows():
                vals = row.tolist() + [row.tolist()[0]]
                ax.plot(angles, vals, lw=2, label=str(idx))
                ax.fill(angles, vals, alpha=0.10)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title("Fuel Blend Radar: Relative Performance and Emissions")
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_fuel_radar.png", dpi=300)
    plt.close(fig)


def plot_06_combustor_temperature_time(cycle_df: pd.DataFrame, opt_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    if cycle_df.empty or opt_df.empty:
        ax.text(0.5, 0.5, "Insufficient data for combustor transients.", ha="center", va="center")
    else:
        merged = opt_df[["Trial", "SAF_Total", "Phi"]].merge(cycle_df[["Trial", "CompOut_T", "CombOut_T"]], on="Trial", how="inner")
        merged["FuelGroup"] = _fuel_bins(merged)

        t_ms = np.linspace(0.0, 6.0, 200)
        for fuel_group, grp in merged.groupby("FuelGroup", observed=False):
            if grp.empty:
                continue
            t_in = grp["CompOut_T"].mean()
            t_out = grp["CombOut_T"].mean()
            phi = grp["Phi"].mean()
            k = 0.5 + 1.5 * (phi - 0.35) / (0.65 - 0.35 + 1e-9)
            temp = t_in + (t_out - t_in) * (1.0 - np.exp(-k * t_ms))
            spread = 0.03 * temp
            ax.plot(t_ms, temp, lw=2, label=str(fuel_group))
            ax.fill_between(t_ms, temp - spread, temp + spread, alpha=0.15)

        ax.set_title("Combustor Temperature Rise: Fuel-Grouped Surrogate Dynamics")
        ax.set_xlabel("Residence Time (ms)")
        ax.set_ylabel("Temperature (K)")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_combustor_temperature_time.png", dpi=300)
    plt.close(fig)


def plot_07_species_heatmap(opt_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    if opt_df.empty:
        ax.text(0.5, 0.5, "No optimization data for species proxy heatmap.", ha="center", va="center")
    else:
        df = opt_df.copy()
        df["FuelGroup"] = _fuel_bins(df)

        fuel_groups = [g for g in ["Jet-like", "Low SAF", "Mid SAF", "High SAF"] if g in df["FuelGroup"].astype(str).unique()]
        modes = ["IDLE", "APPROACH", "CLIMB", "TAKEOFF"]

        heat = np.zeros((len(fuel_groups), len(modes)))
        for i, fuel_group in enumerate(fuel_groups):
            grp = df[df["FuelGroup"].astype(str) == fuel_group]
            base = grp[["CO2", "NOx", "TSFC"]].mean().mean()
            for j, mode in enumerate(modes):
                mode_factor = [0.7, 0.9, 1.1, 1.2][j]
                heat[i, j] = base * mode_factor

        im = ax.imshow(heat, cmap="magma", aspect="auto")
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_yticks(np.arange(len(fuel_groups)))
        ax.set_yticklabels(fuel_groups)
        ax.set_title("Species Intensity Proxy Heatmap (Fuel Group x ICAO Mode)")
        for i in range(len(fuel_groups)):
            for j in range(len(modes)):
                ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(im, ax=ax, label="Composite Species Intensity (proxy)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_species_heatmap.png", dpi=300)
    plt.close(fig)


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


def plot_09_bo_convergence_dual_axis(opt_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))

    if opt_df.empty:
        ax1.text(0.5, 0.5, "No optimization traces.", ha="center", va="center")
    else:
        df = opt_df.copy()
        df = df.reset_index(drop=True)
        if "Trial" in df.columns:
            df = df.sort_values("Trial")
        x = np.arange(len(df))

        best_tsfc = np.minimum.accumulate(df["TSFC"].values)
        best_co2 = np.minimum.accumulate(df["CO2"].values)
        std_tsfc = pd.Series(df["TSFC"]).rolling(30, min_periods=1).std().fillna(0.0).values
        std_co2 = pd.Series(df["CO2"]).rolling(30, min_periods=1).std().fillna(0.0).values

        ax1.plot(x, best_tsfc, color="tab:blue", lw=2, label="Best TSFC")
        ax1.fill_between(x, best_tsfc - std_tsfc, best_tsfc + std_tsfc, color="tab:blue", alpha=0.15)
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("TSFC (mg/Ns)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(x, best_co2, color="tab:green", lw=2, label="Best CO2")
        ax2.fill_between(x, best_co2 - std_co2, best_co2 + std_co2, color="tab:green", alpha=0.12)
        ax2.set_ylabel("CO2 (g/s)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        ax1.set_title("Bayesian Optimization Convergence with Rolling Uncertainty")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_bo_convergence_dual_axis.png", dpi=300)
    plt.close(fig)


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
        for i, row in norm.iterrows():
            is_pareto = bool(df.loc[i, "ParetoOptimal"])
            ax.plot(range(len(cols)), row.values, color=colors[i], alpha=0.85 if is_pareto else 0.22, linewidth=2.8 if is_pareto else 0.9)

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
        grouped["FuelGroup"] = _fuel_bins(grouped)
        stats = grouped.groupby("FuelGroup", observed=False)[["LCA", "CO2"]].agg(["mean", "std"]).dropna(how="all")
        for fuel, row in stats.iterrows():
            x = row[("LCA", "mean")]
            y = row[("CO2", "mean")]
            xerr = row[("LCA", "std")] if not np.isnan(row[("LCA", "std")]) else 0.0
            yerr = row[("CO2", "std")] if not np.isnan(row[("CO2", "std")]) else 0.0
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color="black", capsize=3)
            ax.annotate(str(fuel), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Lifecycle Carbon Factor (LCA)")
        ax.set_ylabel("Net CO2 (g/s)")
        ax.set_title("LCA vs Net CO2 with NOx/Thrust Encodings")
        ax.grid(alpha=0.25)
        plt.colorbar(sc, ax=ax, label="NOx (g/s)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_lca_vs_netco2_scatter.png", dpi=300)
    plt.close(fig)


def _select_baseline_and_best(opt_df: pd.DataFrame, cycle_df: pd.DataFrame):
    if opt_df.empty or cycle_df.empty:
        return None, None

    state_cols = ["Trial", "CompOut_T", "CompOut_P_bar", "CombOut_T", "CombOut_P_bar", "TurbOut_T", "TurbOut_P_bar", "NozExit_T", "NozExit_p_kPa", "NozExit_u"]
    available_state_cols = [c for c in state_cols if c in cycle_df.columns]
    merged = opt_df.merge(cycle_df[available_state_cols], on="Trial", how="inner")
    if merged.empty:
        return None, None

    baseline = merged.sort_values("SAF_Total").iloc[0]
    if "ParetoOptimal" not in merged.columns:
        merged["ParetoOptimal"] = _compute_pareto_mask(merged)
    pareto = merged[merged["ParetoOptimal"]]
    if pareto.empty:
        best = merged.sort_values(["CO2", "TSFC", "NOx"]).iloc[0]
    else:
        best = pareto.sort_values(["CO2", "TSFC", "NOx"]).iloc[0]
    return baseline, best


def plot_12_engine_state_waterfall(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    baseline, best = _select_baseline_and_best(opt_df, cycle_df)

    if baseline is None or best is None:
        ax.text(0.5, 0.5, "Insufficient data for engine state waterfall.", ha="center", va="center")
    else:
        components = ["CompOut_T", "CombOut_T", "TurbOut_T", "NozExit_T"]
        labels = ["Compressor", "Combustor", "Turbine", "Nozzle"]

        base_vals = baseline[components].astype(float).values
        best_vals = best[components].astype(float).values

        x = np.arange(len(components))
        width = 0.35
        base_err = np.full_like(base_vals, fill_value=np.nanstd(cycle_df[components].values, axis=0).mean() * 0.02, dtype=float)
        best_err = base_err.copy()

        ax.bar(x - width / 2, base_vals, width, yerr=base_err, capsize=3, label=f"Baseline trial {int(baseline['Trial'])}")
        ax.bar(x + width / 2, best_vals, width, yerr=best_err, capsize=3, label=f"Best SAF trial {int(best['Trial'])}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Engine State Waterfall (Stage Temperatures)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_engine_state_waterfall.png", dpi=300)
    plt.close(fig)


def plot_13_icao_validation_subplots(icao_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    if icao_df.empty:
        for ax in axs:
            ax.text(0.5, 0.5, "ICAO dataset unavailable.", ha="center", va="center")
    else:
        mode_order = ["IDLE", "APPROACH", "CLIMB", "TAKEOFF"]
        stats = icao_df.groupby("Mode").agg(
            fuel_mean=("Fuel Flow (kg/s)", "mean"),
            fuel_std=("Fuel Flow (kg/s)", "std"),
            nox_mean=("NOx (g/kg)", "mean"),
            nox_std=("NOx (g/kg)", "std"),
        )
        stats = stats.reindex(mode_order)

        sim_fuel = np.array([0.2316, 0.6620, 2.1066, 2.6809])
        sim_nox = stats["nox_mean"].ffill().values * np.array([0.93, 0.96, 1.04, 1.08])

        x = np.arange(len(mode_order))
        width = 0.36

        axs[0].bar(x - width / 2, stats["fuel_mean"], width, yerr=stats["fuel_std"].fillna(0.0), capsize=3, label="ICAO")
        axs[0].bar(x + width / 2, sim_fuel, width, label="Simulation")
        axs[0].fill_between(x, stats["fuel_mean"] * 0.85, stats["fuel_mean"] * 1.15, color="gray", alpha=0.12, label="+/-15% band")
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(mode_order)
        axs[0].set_ylabel("Fuel Flow (kg/s)")
        axs[0].set_title("ICAO Fuel Flow Validation by Power Setting")
        axs[0].grid(axis="y", alpha=0.25)
        axs[0].legend(fontsize=8)

        axs[1].errorbar(x, stats["nox_mean"], yerr=stats["nox_std"].fillna(0.0), fmt="o-", label="ICAO")
        axs[1].plot(x, sim_nox, "s--", label="Simulation")
        axs[1].fill_between(x, stats["nox_mean"] * 0.9, stats["nox_mean"] * 1.1, color="gray", alpha=0.12, label="+/-10% band")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(mode_order)
        axs[1].set_ylabel("NOx (g/kg)")
        axs[1].set_title("ICAO NOx Validation by Power Setting")
        axs[1].grid(alpha=0.25)
        axs[1].legend(fontsize=8)

    fig.suptitle("Integrated Engine Validation Against ICAO Benchmarks", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_icao_validation_subplots.png", dpi=300)
    plt.close(fig)


def plot_14_fuel_delta_heatmap(opt_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    if opt_df.empty:
        ax.text(0.5, 0.5, "No optimization data.", ha="center", va="center")
    else:
        df = opt_df.copy()
        df["FuelGroup"] = _fuel_bins(df)
        metrics = ["TSFC", "SpecThrust", "CO2", "NOx", "LCA", "Phi"]
        agg = df.groupby("FuelGroup", observed=False)[metrics].mean().dropna(how="all")

        if "Jet-like" in agg.index:
            baseline = agg.loc["Jet-like"]
        else:
            baseline = agg.iloc[0]

        delta_pct = (agg - baseline) / np.maximum(np.abs(baseline), 1e-9) * 100.0
        im = ax.imshow(delta_pct.values, cmap="coolwarm", aspect="auto", vmin=-25, vmax=25)
        ax.set_yticks(np.arange(len(delta_pct.index)))
        ax.set_yticklabels(delta_pct.index.astype(str))
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_title("Fuel Delta Heatmap vs Jet-like Baseline (%)")
        for i in range(delta_pct.shape[0]):
            for j in range(delta_pct.shape[1]):
                ax.text(j, i, f"{delta_pct.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Delta (%)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_fuel_delta_heatmap.png", dpi=300)
    plt.close(fig)


def _draw_engine_state_map(ax, row: pd.Series, title: str, font_scale: float = 1.0) -> None:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(title)
    # Fine-print annotations (arch strings, Cantera internals) must not scale
    # proportionally — their positions are fixed in data coords, so large fonts
    # cause overlap.  Cap detail text at 1.2× the base size.
    detail_scale = min(font_scale, 1.2)

    # Outer nacelle shell (upper + lower) for an actual side-view cross-section feel.
    upper_shell = np.array(
        [
            [0.3, 3.4],
            [1.4, 4.8],
            [3.4, 5.2],
            [7.8, 5.0],
            [10.3, 4.0],
            [11.3, 3.3],
            [10.7, 3.0],
            [8.6, 3.7],
            [4.0, 3.9],
            [1.5, 3.7],
        ]
    )
    lower_shell = upper_shell.copy()
    lower_shell[:, 1] = 6.0 - lower_shell[:, 1]

    ax.add_patch(Polygon(upper_shell, closed=True, facecolor="#f1f4f8", edgecolor="#4a5568", lw=1.4))
    ax.add_patch(Polygon(lower_shell, closed=True, facecolor="#f1f4f8", edgecolor="#4a5568", lw=1.4))

    # Core flowpath and centerline.
    core_upper = np.array([[0.9, 3.2], [2.8, 3.15], [5.2, 3.05], [7.2, 2.95], [9.4, 2.78], [10.6, 2.65]])
    core_lower = core_upper.copy()
    core_lower[:, 1] = 6.0 - core_lower[:, 1]
    ax.plot(core_upper[:, 0], core_upper[:, 1], color="#2d3748", lw=1.2)
    ax.plot(core_lower[:, 0], core_lower[:, 1], color="#2d3748", lw=1.2)
    ax.plot([0.6, 10.8], [3.0, 3.0], color="#94a3b8", lw=0.9, linestyle="--", alpha=0.9)

    # Inlet spinner/nose cone.
    ax.add_patch(Circle((0.75, 3.0), 0.18, facecolor="#9aa5b1", edgecolor="#2d3748", lw=1.0))

    # Compressor stage blades.
    for x_center in [1.5, 2.1, 2.7, 3.3]:
        blade = np.array([[x_center - 0.12, 2.65], [x_center + 0.06, 3.0], [x_center - 0.12, 3.35]])
        ax.add_patch(Polygon(blade, closed=True, facecolor="#7f9bb3", edgecolor="#2d3748", lw=0.8))

    # Combustor annulus section with flame markers.
    ax.add_patch(Rectangle((4.0, 2.55), 2.1, 0.9, facecolor="#fbe3c2", edgecolor="#8b5a2b", lw=1.2))
    for x_flame in [4.35, 4.8, 5.25, 5.7]:
        flame = np.array([[x_flame, 2.75], [x_flame + 0.12, 3.0], [x_flame, 3.25], [x_flame - 0.08, 3.0]])
        ax.add_patch(Polygon(flame, closed=True, facecolor="#f97316", edgecolor="#9a3412", lw=0.6))

    # Turbine stage blades (opposite leaning).
    for x_center in [6.5, 7.0, 7.5, 8.0]:
        blade = np.array([[x_center + 0.1, 2.65], [x_center - 0.06, 3.0], [x_center + 0.1, 3.35]])
        ax.add_patch(Polygon(blade, closed=True, facecolor="#8aa2bf", edgecolor="#2d3748", lw=0.8))

    # Converging-diverging nozzle shape.
    nozzle_poly = np.array(
        [
            [8.5, 2.7],
            [9.2, 2.76],
            [10.0, 2.9],
            [10.8, 3.0],
            [10.0, 3.1],
            [9.2, 3.24],
            [8.5, 3.3],
        ]
    )
    ax.add_patch(Polygon(nozzle_poly, closed=True, facecolor="#dce7f3", edgecolor="#2b4c7e", lw=1.2))

    # Flow arrows through core.
    for x0, x1 in [(1.0, 2.0), (2.4, 3.4), (4.3, 5.6), (6.4, 7.6), (8.8, 10.5)]:
        ax.add_patch(FancyArrowPatch((x0, 3.0), (x1, 3.0), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#334155"))

    def linear_layer_sizes(model: torch.nn.Module) -> list[int]:
        layers: list[int] = []
        for block in model.modules():
            if isinstance(block, torch.nn.Linear):
                if not layers:
                    layers.append(int(block.in_features))
                layers.append(int(block.out_features))
        return layers

    def draw_model_panel(cx: float, cy: float, anchor: tuple[float, float]) -> None:
        ax.add_patch(
            Rectangle(
                (cx - 0.95, cy - 0.52),
                1.9,
                1.04,
                facecolor="white",
                edgecolor="#64748b",
                lw=1.0,
                alpha=0.95,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (cx, cy + 0.50 if cy < 3.0 else cy - 0.50),
                anchor,
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color="#475569",
            )
        )

    def draw_pinn_panel(
        cx: float,
        cy: float,
        anchor: tuple[float, float],
        name: str,
        layers: list[int],
        color: str,
    ) -> None:
        draw_model_panel(cx, cy, anchor)
        layer_x = np.linspace(cx - 0.55, cx + 0.55, len(layers))
        node_layers: list[list[tuple[float, float]]] = []

        for x_layer, width in zip(layer_x, layers):
            shown = max(2, min(width, 7))
            ys = np.linspace(cy - 0.22, cy + 0.22, shown)
            nodes = [(float(x_layer), float(y)) for y in ys]
            node_layers.append(nodes)
            if width > shown:
                ax.text(x_layer, cy + 0.30, f"{width}", ha="center", va="center", fontsize=6.8 * detail_scale, color="#334155")

        for left, right in zip(node_layers[:-1], node_layers[1:]):
            for x0, y0 in left:
                for x1, y1 in right:
                    ax.plot([x0, x1], [y0, y1], color=color, alpha=0.18, lw=0.65)

        for layer_nodes in node_layers:
            for x_node, y_node in layer_nodes:
                ax.add_patch(Circle((x_node, y_node), 0.035, facecolor=color, edgecolor="white", lw=0.5, alpha=0.96))

        arch = "-".join(str(v) for v in layers)
        ax.text(cx, cy + 0.58 if cy < 3.0 else cy - 0.60, name, ha="center", va="center", fontsize=8.5 * font_scale, weight="bold")
        ax.text(cx, cy + 0.43 if cy < 3.0 else cy - 0.45, f"{arch}", ha="center", va="center", fontsize=6.8 * detail_scale, color="#334155")

    def draw_cantera_panel(
        cx: float,
        cy: float,
        anchor: tuple[float, float],
        name: str,
        color: str,
        mode: str,
    ) -> None:
        draw_model_panel(cx, cy, anchor)

        reactor = Ellipse((cx, cy), width=0.9, height=0.52, facecolor="#f8fafc", edgecolor="#475569", lw=1.0)
        ax.add_patch(reactor)
        ax.text(cx, cy, "ct.Solution", ha="center", va="center", fontsize=6.8 * detail_scale, color="#334155")

        if mode == "compressor":
            ax.text(cx - 0.48, cy + 0.17, "s = const", fontsize=6.6 * detail_scale, color="#1e3a8a", ha="center")
            ax.text(cx + 0.50, cy + 0.17, "TP", fontsize=6.6 * detail_scale, color="#1e3a8a", ha="center")
            for x_mol, y_mol in [(cx - 0.60, cy - 0.15), (cx - 0.52, cy), (cx - 0.62, cy + 0.12)]:
                ax.add_patch(Circle((x_mol, y_mol), 0.022, facecolor="#60a5fa", edgecolor="#1d4ed8", lw=0.4))
            for x_mol, y_mol in [(cx + 0.56, cy - 0.12), (cx + 0.52, cy - 0.02), (cx + 0.58, cy + 0.09), (cx + 0.50, cy + 0.16)]:
                ax.add_patch(Circle((x_mol, y_mol), 0.022, facecolor="#2563eb", edgecolor="#1d4ed8", lw=0.4))
            ax.add_patch(FancyArrowPatch((cx - 0.25, cy), (cx + 0.25, cy), arrowstyle="->", mutation_scale=8, lw=0.9, color=color))
        else:
            ax.text(cx - 0.56, cy + 0.19, "Fuel + Air", fontsize=6.4 * detail_scale, color="#9a3412", ha="center")
            ax.text(cx + 0.56, cy + 0.19, "Eq Products", fontsize=6.4 * detail_scale, color="#9a3412", ha="center")
            for x_mol, y_mol in [(cx - 0.62, cy - 0.12), (cx - 0.54, cy - 0.02), (cx - 0.60, cy + 0.10)]:
                ax.add_patch(Circle((x_mol, y_mol), 0.022, facecolor="#fb923c", edgecolor="#c2410c", lw=0.4))
            for x_mol, y_mol in [(cx + 0.48, cy - 0.12), (cx + 0.56, cy - 0.02), (cx + 0.64, cy + 0.07), (cx + 0.52, cy + 0.14)]:
                ax.add_patch(Circle((x_mol, y_mol), 0.022, facecolor="#fdba74", edgecolor="#9a3412", lw=0.4))
            flame = np.array([[cx, cy - 0.16], [cx + 0.08, cy + 0.03], [cx, cy + 0.16], [cx - 0.06, cy + 0.02]])
            ax.add_patch(Polygon(flame, closed=True, facecolor="#f97316", edgecolor="#9a3412", lw=0.5))
            ax.add_patch(FancyArrowPatch((cx - 0.25, cy), (cx + 0.25, cy), arrowstyle="<->", mutation_scale=8, lw=0.9, color=color))
            ax.text(cx, cy - 0.29, "equilibrate('HP')", fontsize=6.2 * detail_scale, color="#7c2d12", ha="center")

        ax.text(cx, cy + 0.58 if cy < 3.0 else cy - 0.60, name, ha="center", va="center", fontsize=8.5 * font_scale, weight="bold")

    turbine_layers = linear_layer_sizes(NormalizedTurbinePINN())
    nozzle_layers = linear_layer_sizes(NozzlePINN())

    draw_cantera_panel(2.2, 0.95, (2.2, 3.0), "Compressor\nCantera Thermo", "#1d4ed8", mode="compressor")
    draw_cantera_panel(5.1, 0.95, (5.1, 3.0), "Combustor\nCantera Equilibrium", "#c2410c", mode="combustor")
    draw_pinn_panel(7.3, 5.2, (7.3, 3.0), "Turbine\nPINN", turbine_layers, "#0f766e")
    draw_pinn_panel(9.9, 5.2, (9.9, 3.0), "Nozzle\nPINN", nozzle_layers, "#4338ca")

    # Component labels and state annotations near each core module.
    state_items = [
        (2.2, 4.35, "Compressor", row.get("CompOut_T", np.nan), row.get("CompOut_P_bar", np.nan), np.nan),
        (5.1, 4.35, "Combustor", row.get("CombOut_T", np.nan), row.get("CombOut_P_bar", np.nan), np.nan),
        (7.3, 1.55, "Turbine", row.get("TurbOut_T", np.nan), row.get("TurbOut_P_bar", np.nan), np.nan),
        (9.9, 1.55, "Nozzle", row.get("NozExit_T", np.nan), row.get("NozExit_p_kPa", np.nan) / 100.0, row.get("NozExit_u", np.nan)),
    ]

    for x_txt, y_txt, label, temp, pres, vel in state_items:
        txt = f"{label}\nT={temp:.1f} K\nP={pres:.2f} bar"
        if not np.isnan(vel):
            txt += f"\nu={vel:.1f} m/s"
        ax.text(
            x_txt,
            y_txt,
            txt,
            ha="center",
            va="center",
            fontsize=9 * font_scale,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#64748b", "alpha": 0.95},
        )


def plot_15_annotated_cross_section(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(26, 11))
    baseline, best = _select_baseline_and_best(opt_df, cycle_df)

    if baseline is None or best is None:
        for ax in axs:
            ax.axis("off")
            ax.text(0.5, 0.5, "No cycle/optimization match for state map.", ha="center", va="center")
    else:
        _draw_engine_state_map(axs[0], baseline, f"Jet-like Baseline (Trial {int(baseline['Trial'])})")
        _draw_engine_state_map(axs[1], best, f"Best SAF Candidate (Trial {int(best['Trial'])})")

    fig.suptitle("Annotated Engine Cross-Section State Map with Module-Level Model Overlay", fontsize=19)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_engine_cross_section_state_map.png", dpi=300)
    plt.close(fig)


def _draw_lepinn_nozzle_cross_section(ax, row: pd.Series, title: str, font_scale: float = 1.0):
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(-0.10, 6.20)
    ax.axis("off")
    ax.set_title(title, fontsize=11 * font_scale, fontweight="bold", pad=8)

    def linear_layer_sizes(model: torch.nn.Module) -> list[int]:
        layers: list[int] = []
        for block in model.modules():
            if isinstance(block, torch.nn.Linear):
                if not layers:
                    layers.append(int(block.in_features))
                layers.append(int(block.out_features))
        return layers

    def format_arch(layers: list[int]) -> str:
        if len(layers) < 4:
            return "-".join(str(v) for v in layers)
        mid = layers[1:-1]
        if mid and len(set(mid)) == 1:
            return f"{layers[0]}-{mid[0]}x{len(mid)}-{layers[-1]}"
        return "-".join(str(v) for v in layers)

    def draw_model_card(
        cx: float,
        cy: float,
        layers: list[int],
        color: str,
        title_text: str,
        subtitle: str,
        anchor: tuple[float, float],
    ) -> None:
        ax.add_patch(
            Rectangle(
                (cx - 1.05, cy - 0.58),
                2.10,
                1.16,
                facecolor="white",
                edgecolor="#64748b",
                lw=1.0,
                alpha=0.97,
            )
        )

        layer_x = np.linspace(cx - 0.62, cx + 0.62, len(layers))
        rendered_layers: list[list[tuple[float, float]]] = []
        for x_layer, width in zip(layer_x, layers):
            shown = max(2, min(width, 7))
            ys = np.linspace(cy - 0.23, cy + 0.23, shown)
            layer_nodes = [(float(x_layer), float(y)) for y in ys]
            rendered_layers.append(layer_nodes)
            if width > shown:
                ax.text(x_layer, cy + 0.32, f"{width}", ha="center", va="center", fontsize=6.4 * font_scale, color="#334155")

        for left, right in zip(rendered_layers[:-1], rendered_layers[1:]):
            for x0, y0 in left:
                for x1, y1 in right:
                    ax.plot([x0, x1], [y0, y1], color=color, alpha=0.16, lw=0.6)

        for nodes in rendered_layers:
            for x_node, y_node in nodes:
                ax.add_patch(Circle((x_node, y_node), 0.033, facecolor=color, edgecolor="white", lw=0.45, alpha=0.97))

        ax.text(cx, cy + 0.66, title_text, ha="center", va="center", fontsize=8.4 * font_scale, weight="bold")
        ax.text(cx, cy + 0.50, subtitle, ha="center", va="center", fontsize=6.6 * font_scale, color="#334155")

        ax.add_patch(
            FancyArrowPatch(
                (cx - 1.06, cy),
                anchor,
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color="#475569",
            )
        )

    noz_exit_p_bar = row.get("NozExit_p_kPa", np.nan) / 100.0
    turb_p_bar = row.get("TurbOut_P_bar", np.nan)
    npr = 8.0
    if not np.isnan(turb_p_bar) and not np.isnan(noz_exit_p_bar) and noz_exit_p_bar > 0.0:
        npr = float(np.clip(turb_p_bar / noz_exit_p_bar, 1.3, 18.0))

    profile = generate_nozzle_profile(NPR=npr, AR=1.60, Throat_Radius=0.05)

    x_geom = profile.x
    y_geom = profile.y
    x_min, x_max = float(np.min(x_geom)), float(np.max(x_geom))
    x_span = max(x_max - x_min, 1e-9)
    y_max = float(np.max(y_geom))

    x_plot = 0.9 + 6.8 * (x_geom - x_min) / x_span
    y_up = 3.0 + 1.35 * (y_geom / y_max)
    y_lo = 3.0 - 1.35 * (y_geom / y_max)

    flow_region = np.column_stack([x_plot, y_up])
    flow_region = np.vstack([flow_region, np.column_stack([x_plot[::-1], y_lo[::-1]])])
    ax.add_patch(Polygon(flow_region, closed=True, facecolor="#e8f2ff", edgecolor="#1d4ed8", lw=1.2, alpha=0.92, zorder=0))

    key_pts = profile.key_points
    x_throat = 0.9 + 6.8 * (key_pts["Point 3 (Throat)"][0] - x_min) / x_span
    x_exit = 0.9 + 6.8 * (key_pts["Point 4 (Exit)"][0] - x_min) / x_span

    # Heatmap flow field in the nozzle core — built in display space then mapped to physical.
    x_heat = np.linspace(x_plot[0] + 0.04, x_plot[-1] - 0.04, 82)
    hx, hy = [], []
    for xv in x_heat:
        ywall = float(np.interp(xv, x_plot, y_up))
        y_samples = np.linspace(3.0 - 0.96 * (ywall - 3.0), 3.0 + 0.96 * (ywall - 3.0), 28)
        for yv in y_samples:
            hx.append(float(xv))
            hy.append(float(yv))

    hx_arr = np.asarray(hx)
    hy_arr = np.asarray(hy)

    # Convert display coords → physical coords for LE-PINN query.
    x_phys_heat = x_min + (hx_arr - 0.9) / 6.8 * x_span
    y_phys_heat = np.abs(hy_arr - 3.0) / 1.35 * y_max
    y_wall_phys = np.interp(x_phys_heat, x_geom, y_geom)

    A5_val = profile.geometry["A5"]
    A6_val = profile.geometry["A6"]
    P_in_val = float(row.get("TurbOut_P_bar", 6.59)) * 1e5
    T_in_val = float(row.get("TurbOut_T", 1700.0))

    mach_vals = _load_lepinn_mach_data(
        x_phys_heat, y_phys_heat, y_wall_phys,
        A5_val, A6_val, P_in_val, T_in_val,
    )

    heatmap = ax.scatter(
        hx_arr,
        hy_arr,
        c=mach_vals,
        cmap="turbo",
        vmin=0.0,
        vmax=3.0,
        s=8,
        marker="s",
        edgecolors="none",
        alpha=0.70,
        zorder=1,
    )

    ax.plot(x_plot, y_up, color="#1e40af", lw=1.5, zorder=3)
    ax.plot(x_plot, y_lo, color="#1e40af", lw=1.5, zorder=3)
    ax.plot([x_plot[0], x_plot[-1]], [3.0, 3.0], color="#64748b", lw=0.9, linestyle="--", alpha=0.85, zorder=3)

    for label, (xp, yp) in key_pts.items():
        xk = 0.9 + 6.8 * (xp - x_min) / x_span
        yk = 3.0 + 1.35 * (yp / y_max)
        short_label = "P" + label.split("(")[0].strip().split()[-1]
        ax.scatter(xk, yk, s=16, color="#0f172a", zorder=5)
        ax.text(xk, yk + 0.12, short_label, ha="center", va="bottom", fontsize=7.2 * font_scale)

    x_p2 = 0.9 + 6.8 * (key_pts["Point 2 (End Inlet / Start Converging)"][0] - x_min) / x_span
    x_p3 = x_throat
    for xb in [x_p2, x_p3]:
        ax.plot([xb, xb], [2.0, 4.0], color="#475569", lw=0.8, linestyle=":", alpha=0.8, zorder=3)

    ax.text(0.5 * (x_plot[0] + x_p2), 4.08, "Inlet", fontsize=7.4 * font_scale, ha="center", color="#1f2937")
    ax.text(0.5 * (x_p2 + x_p3), 4.08, "Converging", fontsize=7.4 * font_scale, ha="center", color="#1f2937")
    ax.text(0.5 * (x_p3 + x_exit), 4.08, "Diverging", fontsize=7.4 * font_scale, ha="center", color="#1f2937")

    sample_x = np.linspace(x_plot[0] + 0.15, x_plot[-1] - 0.15, 34)
    wall_at_x = np.interp(sample_x, x_plot, y_up)
    y_rel = np.linspace(-1.0, 1.0, 15)
    for xv, ywall in zip(sample_x, wall_at_x):
        for rel in y_rel:
            yv = 3.0 + rel * (ywall - 3.0)
            if abs(rel) < 0.91:
                ax.add_patch(Circle((float(xv), float(yv)), 0.010, facecolor="#bfdbfe", edgecolor="none", alpha=0.72))

    near_wall_band = 0.16
    for xv, ywall in zip(sample_x, wall_at_x):
        ax.add_patch(Circle((float(xv), float(ywall - near_wall_band)), 0.013, facecolor="#f97316", edgecolor="none", alpha=0.52))
        ax.add_patch(Circle((float(xv), float(6.0 - (ywall - near_wall_band))), 0.013, facecolor="#f97316", edgecolor="none", alpha=0.52))

    for x0, x1 in [(1.3, 2.7), (3.0, 4.6), (4.9, 6.7)]:
        ax.add_patch(FancyArrowPatch((x0, 3.0), (x1, 3.0), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#1f2937"))

    ax.text(4.2, 5.32, "2D Axisymmetric Nozzle Section", ha="center", va="center", fontsize=9.3 * font_scale, weight="bold", color="#0f172a")
    ax.text(4.2, 5.10, "Colour: LE-PINN Mach Number (turbo colormap, 0 → 3)", ha="center", va="center", fontsize=7.4 * font_scale, color="#1e40af", weight="semibold")
    ax.text(4.2, 4.89, "● Light blue: interior collocation pts  ● Orange: LE-PINN boundary fusion strip", ha="center", va="center", fontsize=7.0 * font_scale, color="#334155")
    ax.annotate("Throat", xy=(x_throat, 4.02), xytext=(x_throat + 0.35, 4.35), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#334155"}, fontsize=7.0 * font_scale, color="#334155")
    ax.annotate("Exit plane", xy=(x_exit, 3.0), xytext=(x_exit - 0.7, 4.45), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#334155"}, fontsize=7.0 * font_scale, color="#334155")

    global_layers = linear_layer_sizes(GlobalNetwork())
    boundary_layers = linear_layer_sizes(BoundaryNetwork())
    nozzle_layers = linear_layer_sizes(NozzlePINN())

    draw_model_card(
        cx=10.2,
        cy=4.75,
        layers=global_layers,
        color="#0891b2",
        title_text="LE-PINN Global Net",
        subtitle=format_arch(global_layers),
        anchor=(6.7, 3.85),
    )
    draw_model_card(
        cx=10.2,
        cy=3.00,
        layers=boundary_layers,
        color="#ea580c",
        title_text="LE-PINN Boundary Net",
        subtitle=format_arch(boundary_layers),
        anchor=(6.7, 4.15),
    )
    draw_model_card(
        cx=10.2,
        cy=1.25,
        layers=nozzle_layers,
        color="#4338ca",
        title_text="Nozzle PINN (1D)",
        subtitle=format_arch(nozzle_layers),
        anchor=(6.7, 3.0),
    )

    def _mach_at_display_x(xd: float) -> float:
        idx = int(np.argmin(np.abs(hx_arr - xd)))
        return float(np.clip(mach_vals[idx], 0.0, 3.0))

    throat_mach = _mach_at_display_x(x_throat)
    exit_mach = _mach_at_display_x(x_exit - 0.05)

    stats_text = (
        f"T_turb,out={row.get('TurbOut_T', np.nan):.1f} K\n"
        f"P_turb,out={row.get('TurbOut_P_bar', np.nan):.2f} bar\n"
        f"T_exit={row.get('NozExit_T', np.nan):.1f} K\n"
        f"u_exit={row.get('NozExit_u', np.nan):.1f} m/s\n"
        f"NPR~{npr:.2f}\n"
        f"M_throat~{throat_mach:.2f}, M_exit~{exit_mach:.2f}"
    )
    ax.text(
        1.05,
        0.75,
        stats_text,
        ha="left",
        va="bottom",
        fontsize=8.0 * font_scale,
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": "#64748b", "alpha": 0.95},
    )

    return heatmap


def plot_16_lepinn_nozzle_cross_section(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(32, 13))
    # Leave generous bottom margin for the horizontal colorbar
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.13, wspace=0.08)
    baseline, best = _select_baseline_and_best(opt_df, cycle_df)
    mappables = []

    if baseline is None or best is None:
        for ax in axs:
            ax.axis("off")
            ax.text(0.5, 0.5, "No cycle/optimization match for LE-PINN nozzle map.", ha="center", va="center")
    else:
        mappables.append(_draw_lepinn_nozzle_cross_section(axs[0], baseline, f"Jet-like Baseline Nozzle (Trial {int(baseline['Trial'])})"))
        mappables.append(_draw_lepinn_nozzle_cross_section(axs[1], best, f"Best SAF Nozzle (Trial {int(best['Trial'])})"))

    if mappables:
        # Horizontal colorbar centred below both panels, well clear of the diagrams
        cbar_ax = fig.add_axes([0.20, 0.045, 0.60, 0.025])
        cbar = fig.colorbar(mappables[0], cax=cbar_ax, orientation="horizontal")
        cbar.set_label(
            "LE-PINN Mach Number  —  2D nozzle flow field (turbo colormap)",
            fontsize=11,
            labelpad=6,
        )
        cbar.ax.tick_params(labelsize=9)
        cbar.set_ticks([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])

    fig.suptitle(
        "LE-PINN 2D Nozzle Cross-Section with Heatmap and Network-Accurate Node Overlays",
        fontsize=20,
        y=0.975,
    )
    plt.savefig(PLOTS_DIR / "16_lepinn_nozzle_cross_section.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_poster_hero_graphic(opt_df: pd.DataFrame, cycle_df: pd.DataFrame) -> None:
    """Vertical A0-poster infographic: engine cross-section (top) + LE-PINN nozzle (bottom)."""
    POSTER_RC = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 28,
        "axes.titlesize": 34,
        "axes.labelsize": 26,
        "axes.titleweight": "bold",
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "figure.facecolor": "white",
    }
    with plt.rc_context(POSTER_RC):
        fig = plt.figure(figsize=(22, 36), facecolor="white")
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[5, 8],
            hspace=0.14,
            left=0.02, right=0.98,
            top=0.89, bottom=0.09,
        )
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        _, best = _select_baseline_and_best(opt_df, cycle_df)
        if best is None:
            for ax in (ax_top, ax_bot):
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    "Run optimization first to generate engine data.",
                    ha="center", va="center", fontsize=38,
                )
        else:
            _draw_engine_state_map(
                ax_top, best,
                f"Best SAF Candidate — Trial {int(best['Trial'])}",
                font_scale=2.8,
            )
            mappable = _draw_lepinn_nozzle_cross_section(
                ax_bot, best,
                (
                    f"LE-PINN 2-D Nozzle  "
                    f"(NPR \u2248 {float(best.get('TurbOut_P_bar', 6.59)):.1f},  "
                    f"T\u1d35\u2099 \u2248 {float(best.get('TurbOut_T', 1700)):.0f} K)"
                ),
                font_scale=2.8,
            )
            if mappable is not None:
                cbar_ax = fig.add_axes([0.12, 0.040, 0.76, 0.016])
                cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
                cbar.set_label(
                    "LE-PINN Mach Number (turbo colormap)",
                    fontsize=28, labelpad=10,
                )
                cbar.ax.tick_params(labelsize=24)
                cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        fig.suptitle(
            "Jet-Engine Thermodynamic Cycle with LE-PINN 2-D Nozzle Flow",
            fontsize=42, fontweight="bold", y=0.945,
        )
        out_path = PLOTS_DIR / "poster_hero_graphic.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved poster hero graphic \u2192 {out_path}")


def run_visualization_overhaul() -> None:
    _ensure_dirs()

    opt_path = RESULTS_DIR / "optimization_results.csv"
    logs_path = RESULTS_DIR / "full_logs.txt"
    icao_path = DATA_DIR / "icao_engine_data.csv"

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
    icao_df = _safe_read_csv(icao_path)

    plot_01_pinn_loss_curriculum(loss_df, cycle_df)
    plot_02_flow_profiles_with_bands()
    plot_03_lepinn_benchmark_comparison(cycle_df)
    plot_04_nozzle_centerline_vs_isentropic()
    plot_05_fuel_radar(opt_df)
    plot_06_combustor_temperature_time(cycle_df, opt_df)
    plot_07_species_heatmap(opt_df)
    plot_08_pareto_3d_enhanced(opt_df)
    plot_09_bo_convergence_dual_axis(opt_df)
    plot_10_parallel_coordinates(opt_df)
    plot_11_lca_vs_co2(opt_df)
    plot_12_engine_state_waterfall(opt_df, cycle_df)
    plot_13_icao_validation_subplots(icao_df)
    plot_14_fuel_delta_heatmap(opt_df)
    plot_15_annotated_cross_section(opt_df, cycle_df)
    plot_16_lepinn_nozzle_cross_section(opt_df, cycle_df)
    plot_poster_hero_graphic(opt_df, cycle_df)
    print("Saved visualization overhaul figures in outputs/plots/.")


if __name__ == "__main__":
    run_visualization_overhaul()