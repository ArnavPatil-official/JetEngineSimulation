#!/usr/bin/env python3
"""
Compare the regular nozzle PINN and LE-PINN across a nozzle condition sweep.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde, probplot


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.le_pinn import run_le_pinn
from simulation.nozzle.nozzle import analytical_isentropic_nozzle, run_nozzle_pinn


NOZZLE_MODEL_PATH = REPO_ROOT / "models" / "nozzle_pinn.pt"
LE_MODEL_REQUEST_PATH = REPO_ROOT / "models" / "le_pinn_unified.pt"
LE_MODEL_FALLBACK_PATH = REPO_ROOT / "models" / "le_pinn_engine_unified.pt"

NPR_VALUES = [4.0, 5.0, 6.0, 6.5, 7.0, 8.0]
THERMO_CONFIGS = {
    "JetA1": {"cp": 1150.0, "R": 287.0, "gamma": 1.33},
    "HEFA50": {"cp": 1200.0, "R": 287.0, "gamma": 1.30},
    "BioSPK": {"cp": 1250.0, "R": 287.0, "gamma": 1.28},
}
A_IN = 0.25
LENGTH = 1.0
M_DOT = 50.0
AMBIENT_P = 101325.0
T_IN = 1700.0
U_IN = 500.0


def _ensure_checkpoints() -> None:
    if not NOZZLE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Regular PINN checkpoint missing: {NOZZLE_MODEL_PATH}")
    if not LE_MODEL_REQUEST_PATH.exists() and not LE_MODEL_FALLBACK_PATH.exists():
        raise FileNotFoundError(
            "LE-PINN checkpoint missing: neither "
            f"{LE_MODEL_REQUEST_PATH} nor {LE_MODEL_FALLBACK_PATH} exists"
        )


def _make_inlet(T_in: float, P_in: float, u_in: float, gas_constant: float) -> dict[str, float]:
    return {"rho": P_in / (gas_constant * T_in), "u": u_in, "p": P_in, "T": T_in}


def _area_profile(x: np.ndarray, A_in: float, A_exit: float, length: float) -> np.ndarray:
    return A_in + (A_exit - A_in) * (1.0 - np.cos(np.pi * x / (2.0 * length)))


def _profile_to_exit_state(
    inlet_state: dict[str, float],
    exit_state: dict[str, float],
    A_in: float,
    A_exit: float,
    length: float,
    thermo_props: dict[str, float],
    n_points: int = 50,
) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, length, n_points, dtype=np.float64)
    A_x = _area_profile(x, A_in, A_exit, length)
    gamma = thermo_props["gamma"]
    cp = thermo_props["cp"]
    gas_constant = thermo_props["R"]

    s = (A_in - A_x) / max(A_in - A_exit, 1e-12)
    s = np.clip(s, 0.0, 1.0)

    p_in = float(inlet_state["p"])
    T_inlet = float(inlet_state["T"])
    u_inlet = float(inlet_state["u"])
    p_exit = float(exit_state["p"])

    p = p_in - (p_in - p_exit) * s
    p_ratio = np.clip(p / max(p_in, 1e-12), 1e-12, None)
    T = T_inlet * p_ratio ** ((gamma - 1.0) / gamma)
    u_sq = u_inlet ** 2 + 2.0 * cp * (T_inlet - T)
    u = np.sqrt(np.maximum(u_sq, 0.0))
    rho = p / np.maximum(gas_constant * T, 1e-12)

    return {
        "x": x.astype(np.float32),
        "rho": rho.astype(np.float32),
        "u": u.astype(np.float32),
        "p": p.astype(np.float32),
        "T": T.astype(np.float32),
    }


def _build_scalar_consistent_analytical_profile(
    inlet_state: dict[str, float],
    ambient_p: float,
    A_in: float,
    A_exit: float,
    length: float,
    thermo_props: dict[str, float],
    n_points: int = 50,
) -> dict[str, np.ndarray]:
    exit_state = analytical_isentropic_nozzle(
        inlet_state=inlet_state,
        ambient_p=ambient_p,
        A_exit=A_exit,
        thermo_props=thermo_props,
        m_dot=M_DOT,
    )
    return _profile_to_exit_state(
        inlet_state=inlet_state,
        exit_state=exit_state,
        A_in=A_in,
        A_exit=A_exit,
        length=length,
        thermo_props=thermo_props,
        n_points=n_points,
    )


def _compute_isentropic_reference(
    npr: float,
    thermo_props: dict[str, float],
    A_exit: float,
) -> dict[str, float]:
    gamma = thermo_props["gamma"]
    gas_constant = thermo_props["R"]
    mach_sq = 2.0 / (gamma - 1.0) * (npr ** ((gamma - 1.0) / gamma) - 1.0)
    M_exit = min(math.sqrt(max(mach_sq, 0.0)), 1.0)
    T_exit = T_IN / (1.0 + 0.5 * (gamma - 1.0) * M_exit ** 2)
    P_in = npr * AMBIENT_P
    P_exit = P_in / (1.0 + 0.5 * (gamma - 1.0) * M_exit ** 2) ** (gamma / (gamma - 1.0))
    u_exit = M_exit * math.sqrt(gamma * gas_constant * T_exit)
    rho_exit = P_exit / (gas_constant * T_exit)
    thrust = M_DOT * u_exit + (P_exit - AMBIENT_P) * A_exit
    return {
        "rho": rho_exit,
        "u": u_exit,
        "p": P_exit,
        "T": T_exit,
        "thrust": thrust,
    }


def _normalize_profile(profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in profile.items()}


def _select_profile(
    result: dict,
    inlet_state: dict[str, float],
    A_in: float,
    A_exit: float,
    length: float,
    thermo_props: dict[str, float],
    n_points: int = 50,
) -> dict[str, np.ndarray]:
    if result.get("used_fallback") or "profiles" not in result:
        profile = _profile_to_exit_state(
            inlet_state=inlet_state,
            exit_state=result["exit_state"],
            A_in=A_in,
            A_exit=A_exit,
            length=length,
            thermo_props=thermo_props,
            n_points=n_points,
        )
    else:
        profile = result["profiles"]
    return _normalize_profile(profile)


def _eos_error(profile: dict[str, np.ndarray], gas_constant: float) -> float:
    p = np.asarray(profile["p"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    T = np.asarray(profile["T"], dtype=float)
    residual = np.abs(p - rho * gas_constant * T) / np.maximum(np.abs(p), 1e-12)
    return float(np.mean(residual))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _plot_hist_with_kde(ax, values: np.ndarray, color: str, label: str) -> None:
    ax.hist(values, bins=8, density=True, alpha=0.5, color=color, label=label)
    if values.size >= 2 and float(np.std(values)) > 1e-12:
        grid = np.linspace(float(values.min()), float(values.max()), 200)
        kde = gaussian_kde(values)
        ax.plot(grid, kde(grid), color=color, linewidth=2)


def main() -> None:
    _ensure_checkpoints()
    output_results = REPO_ROOT / "outputs" / "results"
    output_plots = REPO_ROOT / "outputs" / "plots"
    output_results.mkdir(parents=True, exist_ok=True)
    output_plots.mkdir(parents=True, exist_ok=True)

    rows = []
    selected_profiles: dict[str, dict[str, np.ndarray]] = {}

    for npr in NPR_VALUES:
        A_exit = A_IN / npr * 3.0
        for fuel_name, thermo in THERMO_CONFIGS.items():
            P_in = npr * AMBIENT_P
            inlet_state = _make_inlet(T_IN, P_in, U_IN, thermo["R"])

            reg_result = run_nozzle_pinn(
                model_path=str(NOZZLE_MODEL_PATH),
                inlet_state=inlet_state,
                ambient_p=AMBIENT_P,
                A_in=A_IN,
                A_exit=A_exit,
                length=LENGTH,
                thermo_props=thermo,
                m_dot=M_DOT,
                device="cpu",
                return_profile=True,
                thrust_model="static_test_stand",
            )
            le_result = run_le_pinn(
                model_path=str(LE_MODEL_REQUEST_PATH),
                inlet_state=inlet_state,
                ambient_p=AMBIENT_P,
                A_in=A_IN,
                A_exit=A_exit,
                length=LENGTH,
                thermo_props=thermo,
                m_dot=M_DOT,
                n_axial=50,
                n_radial=20,
                device="cpu",
                return_profile=True,
                thrust_model="static_test_stand",
            )
            isen_exit = _compute_isentropic_reference(npr, thermo, A_exit)
            isen_profile = _profile_to_exit_state(
                inlet_state=inlet_state,
                exit_state=isen_exit,
                A_in=A_IN,
                A_exit=A_exit,
                length=LENGTH,
                thermo_props=thermo,
                n_points=50,
            )

            reg_profile = _select_profile(reg_result, inlet_state, A_IN, A_exit, LENGTH, thermo, n_points=50)
            le_profile = _select_profile(le_result, inlet_state, A_IN, A_exit, LENGTH, thermo, n_points=50)

            mass_err_reg = float(reg_result["mass_conservation"]["error_pct"]) / 100.0
            mass_err_le = float(le_result["mass_conservation"]["max_error"])
            eos_err_reg = _eos_error(reg_profile, thermo["R"])
            eos_err_le = _eos_error(le_profile, thermo["R"])

            rows.append(
                {
                    "NPR": npr,
                    "fuel": fuel_name,
                    "T_exit_reg": float(reg_result["exit_state"]["T"]),
                    "T_exit_le": float(le_result["exit_state"]["T"]),
                    "T_exit_isen": float(isen_exit["T"]),
                    "P_exit_reg": float(reg_result["exit_state"]["p"]),
                    "P_exit_le": float(le_result["exit_state"]["p"]),
                    "P_exit_isen": float(isen_exit["p"]),
                    "u_exit_reg": float(reg_result["exit_state"]["u"]),
                    "u_exit_le": float(le_result["exit_state"]["u"]),
                    "u_exit_isen": float(isen_exit["u"]),
                    "thrust_reg": float(reg_result["thrust_total"]),
                    "thrust_le": float(le_result["thrust_total"]),
                    "thrust_isen": float(isen_exit["thrust"]),
                    "mass_err_reg": mass_err_reg,
                    "mass_err_le": mass_err_le,
                    "fallback_reg": bool(reg_result["used_fallback"]),
                    "fallback_le": bool(le_result["used_fallback"]),
                    "eos_err_reg": eos_err_reg,
                    "eos_err_le": eos_err_le,
                }
            )

            if math.isclose(npr, 6.5) and fuel_name == "JetA1":
                selected_profiles = {
                    "regular": reg_profile,
                    "le": le_profile,
                    "isen": _normalize_profile(isen_profile),
                }

    df = pd.DataFrame(rows)
    csv_path = output_results / "pinn_comparison_results.csv"
    df.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(16, 20))
    outer = fig.add_gridspec(4, 2)
    marker_map = {"JetA1": "o", "HEFA50": "s", "BioSPK": "^"}

    ax_a = fig.add_subplot(outer[0, 0])
    for fuel_name, marker in marker_map.items():
        fuel_df = df[df["fuel"] == fuel_name].sort_values("NPR")
        ax_a.plot(fuel_df["NPR"], fuel_df["thrust_reg"], color="tab:blue", marker=marker, label=f"{fuel_name} PINN")
        ax_a.plot(fuel_df["NPR"], fuel_df["thrust_le"], color="tab:orange", marker=marker, label=f"{fuel_name} LE-PINN")
        ax_a.plot(fuel_df["NPR"], fuel_df["thrust_isen"], color="gray", linestyle="--", marker=marker, label=f"{fuel_name} Isentropic")
        ax_a.fill_between(
            fuel_df["NPR"],
            fuel_df["thrust_isen"] * 0.90,
            fuel_df["thrust_isen"] * 1.10,
            color="lightgray",
            alpha=0.15,
        )
    ax_a.set_title("Thrust vs NPR — all fuels")
    ax_a.set_xlabel("NPR")
    ax_a.set_ylabel("Thrust [N]")
    handles, labels = ax_a.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax_a.legend(dedup.values(), dedup.keys(), fontsize=8, ncol=2)

    ax_b = fig.add_subplot(outer[0, 1])
    ax_b.scatter(df["T_exit_isen"], df["T_exit_reg"], color="tab:blue", label="Regular PINN")
    ax_b.scatter(df["T_exit_isen"], df["T_exit_le"], color="tab:orange", label="LE-PINN")
    t_min = float(min(df["T_exit_isen"].min(), df["T_exit_reg"].min(), df["T_exit_le"].min()))
    t_max = float(max(df["T_exit_isen"].max(), df["T_exit_reg"].max(), df["T_exit_le"].max()))
    ref_line = np.linspace(t_min, t_max, 200)
    ax_b.plot(ref_line, ref_line, color="black")
    ax_b.plot(ref_line, ref_line * 1.05, color="black", linestyle="--")
    ax_b.plot(ref_line, ref_line * 0.95, color="black", linestyle="--")
    ax_b.set_title("Exit temperature vs isentropic reference")
    ax_b.set_xlabel("T_exit_isen [K]")
    ax_b.set_ylabel("T_exit [K]")
    ax_b.legend()

    ax_c = fig.add_subplot(outer[1, 0])
    thrust_err_reg = (df["thrust_reg"].to_numpy() - df["thrust_isen"].to_numpy()) / df["thrust_isen"].to_numpy() * 100.0
    thrust_err_le = (df["thrust_le"].to_numpy() - df["thrust_isen"].to_numpy()) / df["thrust_isen"].to_numpy() * 100.0
    _plot_hist_with_kde(ax_c, thrust_err_reg, "tab:blue", "Regular PINN")
    _plot_hist_with_kde(ax_c, thrust_err_le, "tab:orange", "LE-PINN")
    ax_c.axvline(0.0, color="black", linestyle="--")
    ax_c.set_title("Thrust error distribution vs isentropic (%)")
    ax_c.set_xlabel("Percent error [%]")
    ax_c.set_ylabel("Density")
    ax_c.legend()

    ax_d = fig.add_subplot(outer[1, 1])
    thrust_mean = (df["thrust_reg"].to_numpy() + df["thrust_le"].to_numpy()) / 2.0
    thrust_diff = df["thrust_reg"].to_numpy() - df["thrust_le"].to_numpy()
    mean_diff = float(np.mean(thrust_diff))
    std_diff = float(np.std(thrust_diff, ddof=0))
    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff
    ax_d.scatter(thrust_mean, thrust_diff, color="tab:purple")
    ax_d.axhline(0.0, color="black", linestyle=":")
    ax_d.axhline(mean_diff, color="tab:blue")
    ax_d.axhline(upper, color="tab:blue", linestyle="--")
    ax_d.axhline(lower, color="tab:blue", linestyle="--")
    ax_d.text(0.98, 0.90, f"Mean={mean_diff:.1f}", transform=ax_d.transAxes, ha="right")
    ax_d.text(0.98, 0.84, f"+1.96σ={upper:.1f}", transform=ax_d.transAxes, ha="right")
    ax_d.text(0.98, 0.78, f"-1.96σ={lower:.1f}", transform=ax_d.transAxes, ha="right")
    ax_d.set_title("Bland-Altman: PINN vs LE-PINN thrust")
    ax_d.set_xlabel("Mean thrust [N]")
    ax_d.set_ylabel("PINN - LE-PINN [N]")

    inner_e = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[2, 0], wspace=0.25, hspace=0.30)
    panel_e_axes = [fig.add_subplot(inner_e[i, j]) for i in range(2) for j in range(2)]
    variables = [("T", "T [K]"), ("p", "P [Pa]"), ("u", "u [m/s]"), ("rho", "rho [kg/m³]")]
    for ax, (var, ylabel) in zip(panel_e_axes, variables):
        for label, profile, color, style in [
            ("PINN", selected_profiles["regular"], "tab:blue", "-"),
            ("LE-PINN", selected_profiles["le"], "tab:orange", "-"),
            ("Isentropic", selected_profiles["isen"], "gray", "--"),
        ]:
            x_profile = np.asarray(profile["x"], dtype=float)
            x_norm = x_profile / max(float(np.max(x_profile)), 1e-12)
            ax.plot(x_norm, np.asarray(profile[var], dtype=float), color=color, linestyle=style, label=label)
        ax.set_xlabel("x/L")
        ax.set_ylabel(ylabel)
    panel_e_axes[0].legend(fontsize=8)
    panel_e_axes[0].set_title("Axial profiles — NPR 6.5, Jet-A1")

    ax_f = fig.add_subplot(outer[2, 1])
    grouped = df.groupby("NPR")[["mass_err_reg", "mass_err_le"]].max().reset_index()
    x_idx = np.arange(len(grouped))
    width = 0.35
    ax_f.bar(x_idx - width / 2, grouped["mass_err_reg"] * 100.0, width=width, color="tab:blue", label="PINN")
    ax_f.bar(x_idx + width / 2, grouped["mass_err_le"] * 100.0, width=width, color="tab:orange", label="LE-PINN")
    ax_f.axhline(2.0, color="red", linestyle="--")
    ax_f.set_xticks(x_idx)
    ax_f.set_xticklabels([f"{value:.1f}" for value in grouped["NPR"]])
    ax_f.set_xlabel("NPR")
    ax_f.set_ylabel("Max mass conservation error [%]")
    ax_f.set_title("Mass conservation error (%)")
    ax_f.legend()

    inner_g = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3, 0], wspace=0.30)
    ax_g1 = fig.add_subplot(inner_g[0, 0])
    ax_g2 = fig.add_subplot(inner_g[0, 1])
    probplot(thrust_err_reg, dist="norm", plot=ax_g1)
    probplot(thrust_err_le, dist="norm", plot=ax_g2)
    ax_g1.set_title("Regular PINN")
    ax_g2.set_title("LE-PINN")
    ax_g1.set_ylabel("Residual quantiles")
    ax_g2.set_ylabel("Residual quantiles")
    ax_g1.set_xlabel("Theoretical quantiles")
    ax_g2.set_xlabel("Theoretical quantiles")
    ax_g1.text(0.5, 1.08, "Q-Q: thrust residuals", transform=ax_g1.transAxes, ha="center")

    ax_h = fig.add_subplot(outer[3, 1])
    ax_h.axis("off")
    metric_rows = []
    row_labels = []
    cell_colours = []
    for var, reg_col, le_col, ref_col in [
        ("Thrust", "thrust_reg", "thrust_le", "thrust_isen"),
        ("T_exit", "T_exit_reg", "T_exit_le", "T_exit_isen"),
        ("P_exit", "P_exit_reg", "P_exit_le", "P_exit_isen"),
        ("u_exit", "u_exit_reg", "u_exit_le", "u_exit_isen"),
    ]:
        ref = df[ref_col].to_numpy(dtype=float)
        reg = df[reg_col].to_numpy(dtype=float)
        le = df[le_col].to_numpy(dtype=float)
        reg_rmse = float(np.sqrt(np.mean((reg - ref) ** 2)))
        le_rmse = float(np.sqrt(np.mean((le - ref) ** 2)))
        reg_mae = float(np.mean(np.abs(reg - ref)))
        le_mae = float(np.mean(np.abs(le - ref)))
        reg_r2 = _r2_score(ref, reg)
        le_r2 = _r2_score(ref, le)
        metric_rows.append([reg_rmse, le_rmse, reg_mae, le_mae, reg_r2, le_r2])
        row_labels.append(var)
        winner_is_pinn = reg_rmse <= le_rmse
        row_colors = []
        for col_idx in range(6):
            if winner_is_pinn and col_idx in (0, 2, 4):
                row_colors.append("#dbeafe")
            elif (not winner_is_pinn) and col_idx in (1, 3, 5):
                row_colors.append("#ffedd5")
            else:
                row_colors.append("white")
        cell_colours.append(row_colors)

    table = ax_h.table(
        cellText=[[f"{value:.3e}" if idx < 4 else f"{value:.4f}" for idx, value in enumerate(row)] for row in metric_rows],
        rowLabels=row_labels,
        colLabels=["PINN_RMSE", "LE_RMSE", "PINN_MAE", "LE_MAE", "PINN_R²", "LE_R²"],
        cellColours=cell_colours,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    ax_h.set_title("Metrics vs isentropic reference")

    fig.suptitle("LE-PINN vs Regular PINN — Nozzle Benchmark", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = output_plots / "pinn_le_pinn_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    thrust_rmse_reg_pct = float(np.sqrt(np.mean(((df["thrust_reg"] - df["thrust_isen"]) / df["thrust_isen"]) ** 2)) * 100.0)
    thrust_rmse_le_pct = float(np.sqrt(np.mean(((df["thrust_le"] - df["thrust_isen"]) / df["thrust_isen"]) ** 2)) * 100.0)
    temp_rmse_reg_pct = float(np.sqrt(np.mean(((df["T_exit_reg"] - df["T_exit_isen"]) / df["T_exit_isen"]) ** 2)) * 100.0)
    temp_rmse_le_pct = float(np.sqrt(np.mean(((df["T_exit_le"] - df["T_exit_isen"]) / df["T_exit_isen"]) ** 2)) * 100.0)
    press_rmse_reg_pct = float(np.sqrt(np.mean(((df["P_exit_reg"] - df["P_exit_isen"]) / df["P_exit_isen"]) ** 2)) * 100.0)
    press_rmse_le_pct = float(np.sqrt(np.mean(((df["P_exit_le"] - df["P_exit_isen"]) / df["P_exit_isen"]) ** 2)) * 100.0)
    mass_err_reg_pct = float(np.mean(df["mass_err_reg"]) * 100.0)
    mass_err_le_pct = float(np.mean(df["mass_err_le"]) * 100.0)

    summary_rows = [
        ("Thrust RMSE%", thrust_rmse_reg_pct, thrust_rmse_le_pct),
        ("T_exit RMSE%", temp_rmse_reg_pct, temp_rmse_le_pct),
        ("P_exit RMSE%", press_rmse_reg_pct, press_rmse_le_pct),
        ("Mass err%", mass_err_reg_pct, mass_err_le_pct),
    ]

    wins = 0
    print("=== BENCHMARK SUMMARY ===")
    print(f"{'Metric':<18} {'Regular PINN':>14} {'LE-PINN':>12} {'Winner':>10}")
    for metric, reg_value, le_value in summary_rows:
        winner = "Regular PINN" if reg_value <= le_value else "LE-PINN"
        wins += int(winner == "LE-PINN")
        print(f"  {metric:<16} {reg_value:>12.3f} {le_value:>12.3f} {winner:>12}")
    print(f"LE-PINN vs Regular PINN: {wins}/{len(summary_rows)} metrics won")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
