#!/usr/bin/env python3
"""
Research-paper comparison: Regular Nozzle PINN vs isentropic reference.

Produces a single 4-panel figure (2×2) containing only publication-essential
panels:
  A (top-left)  — Axial flow profiles at NPR 6.5, Jet-A1 (2×2 sub-grid)
  B (top-right) — Thrust vs NPR across all fuels
  C (bottom-left)  — Exit temperature parity scatter vs isentropic reference
  D (bottom-right) — Quantitative metrics table (RMSE / MAE / R²)

Usage
-----
    python3 scripts/validation/compare_pinn_le_pinn.py

Outputs
-------
  outputs/results/pinn_comparison_results.csv
  outputs/plots/pinn_comparison.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.nozzle import run_nozzle_pinn


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
NOZZLE_MODEL_PATH = REPO_ROOT / "models" / "nozzle_pinn.pt"
# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
NPR_VALUES = [4.0, 5.0, 6.0, 6.5, 7.0, 8.0]
THERMO_CONFIGS = {
    "Jet-A1":  {"cp": 1150.0, "R": 287.0, "gamma": 1.33},
    "HEFA-50": {"cp": 1200.0, "R": 287.0, "gamma": 1.30},
    "Bio-SPK": {"cp": 1250.0, "R": 287.0, "gamma": 1.28},
}
A_IN = 0.25
LENGTH = 1.0
M_DOT = 50.0
AMBIENT_P = 101325.0
T_IN = 1700.0
U_IN = 500.0

# Visual style
_COLOR_PINN = "tab:blue"
_COLOR_ISEN = "#555555"
_MARKER_MAP = {"Jet-A1": "o", "HEFA-50": "s", "Bio-SPK": "^"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_checkpoints() -> None:
    if not NOZZLE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Regular PINN checkpoint missing: {NOZZLE_MODEL_PATH}")


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
    return {"rho": rho_exit, "u": u_exit, "p": P_exit, "T": T_exit, "thrust": thrust}


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
    """Return axial profile array; synthesise analytical profile when fallback used."""
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


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _panel_b_thrust_vs_npr(ax, df: pd.DataFrame) -> None:
    """Thrust vs NPR for all fuels, regular PINN against isentropic reference."""
    for fuel_name, marker in _MARKER_MAP.items():
        fuel_df = df[df["fuel"] == fuel_name].sort_values("NPR")
        ax.plot(fuel_df["NPR"], fuel_df["thrust_reg"],
                color=_COLOR_PINN, marker=marker, linewidth=1.5, markersize=5, zorder=3)
        ax.plot(fuel_df["NPR"], fuel_df["thrust_isen"],
                color=_COLOR_ISEN, linestyle="--", marker=marker,
                linewidth=1.2, markersize=4, zorder=2)
        # ±10 % isentropic band (shared across fuels, draw once)
    # Shade ±10% band using JetA1 as representative
    ja_df = df[df["fuel"] == "Jet-A1"].sort_values("NPR")
    if len(ja_df) > 0:
        ax.fill_between(
            ja_df["NPR"],
            ja_df["thrust_isen"] * 0.90,
            ja_df["thrust_isen"] * 1.10,
            color="lightgray", alpha=0.20, label="_nolegend_",
        )

    # Build a clean 5-entry legend: 2 model entries + 3 fuel-marker entries
    model_handles = [
        mlines.Line2D([], [], color=_COLOR_PINN, linewidth=1.5, label="Regular PINN"),
        mlines.Line2D([], [], color=_COLOR_ISEN, linewidth=1.2, linestyle="--", label="Isentropic"),
    ]
    fuel_handles = [
        mlines.Line2D([], [], color="black", marker=m, linestyle="None", markersize=5, label=f)
        for f, m in _MARKER_MAP.items()
    ]
    ax.legend(handles=model_handles + fuel_handles, fontsize=8, ncol=2, loc="upper left")

    ax.set_xlabel("Nozzle Pressure Ratio (NPR)", fontsize=9)
    ax.set_ylabel("Thrust [N]", fontsize=9)
    ax.set_title("(b) Thrust vs NPR — regular PINN vs isentropic", fontsize=9, loc="left")


def _panel_c_parity_scatter(ax, df: pd.DataFrame) -> None:
    """Exit temperature parity scatter for the regular PINN only."""
    ax.scatter(df["T_exit_isen"], df["T_exit_reg"],
               color=_COLOR_PINN, s=30, alpha=0.85, label="Regular PINN", zorder=3)

    all_T = np.concatenate([
        df["T_exit_isen"].to_numpy(),
        df["T_exit_reg"].to_numpy(),
    ])
    t_min, t_max = float(np.min(all_T)), float(np.max(all_T))
    margin = (t_max - t_min) * 0.05
    ref = np.linspace(t_min - margin, t_max + margin, 200)
    ax.plot(ref, ref,        color="black",      linewidth=1.2, label="Perfect agreement")
    ax.plot(ref, ref * 1.05, color="black",      linewidth=0.8, linestyle="--")
    ax.plot(ref, ref * 0.95, color="black",      linewidth=0.8, linestyle="--",
            label="±5 % band")

    ax.set_xlabel("T_exit isentropic [K]", fontsize=9)
    ax.set_ylabel("T_exit predicted [K]", fontsize=9)
    ax.set_title("(c) Exit temperature parity", fontsize=9, loc="left")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")


def _panel_d_metrics_table(ax, df: pd.DataFrame) -> None:
    """Quantitative regular-PINN accuracy table against the isentropic reference."""
    ax.axis("off")

    metric_rows = []
    row_labels = []

    for var, reg_col, ref_col in [
        ("Thrust",  "thrust_reg",  "thrust_isen"),
        ("T_exit",  "T_exit_reg",  "T_exit_isen"),
        ("P_exit",  "P_exit_reg",  "P_exit_isen"),
        ("u_exit",  "u_exit_reg",  "u_exit_isen"),
    ]:
        ref = df[ref_col].to_numpy(dtype=float)
        reg = df[reg_col].to_numpy(dtype=float)

        reg_rmse = float(np.sqrt(np.mean((reg - ref) ** 2)))
        reg_mae  = float(np.mean(np.abs(reg - ref)))
        reg_r2   = _r2_score(ref, reg)

        metric_rows.append([reg_rmse, reg_mae, reg_r2])
        row_labels.append(var)

    # Format: RMSE/MAE in scientific notation; R² to 4 decimal places
    formatted = []
    for row in metric_rows:
        formatted.append([
            f"{row[0]:.3e}",
            f"{row[1]:.3e}",
            f"{row[2]:.4f}",
        ])

    table = ax.table(
        cellText=formatted,
        rowLabels=row_labels,
        colLabels=["PINN\nRMSE", "PINN\nMAE", "PINN\nR²"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.7)
    ax.set_title("(d) Regular PINN accuracy vs isentropic reference", fontsize=8, loc="left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_checkpoints()

    output_results = REPO_ROOT / "outputs" / "results"
    output_plots   = REPO_ROOT / "outputs" / "plots"
    output_results.mkdir(parents=True, exist_ok=True)
    output_plots.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Data collection sweep
    # -----------------------------------------------------------------------
    rows: list[dict] = []
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

            reg_profile = _select_profile(
                reg_result, inlet_state, A_IN, A_exit, LENGTH, thermo, n_points=50)

            mass_err_reg = float(reg_result["mass_conservation"]["error_pct"]) / 100.0

            rows.append({
                "NPR":        npr,
                "fuel":       fuel_name,
                # Exit states
                "T_exit_reg":   float(reg_result["exit_state"]["T"]),
                "T_exit_isen":  float(isen_exit["T"]),
                "P_exit_reg":   float(reg_result["exit_state"]["p"]),
                "P_exit_isen":  float(isen_exit["p"]),
                "u_exit_reg":   float(reg_result["exit_state"]["u"]),
                "u_exit_isen":  float(isen_exit["u"]),
                # Thrust
                "thrust_reg":   float(reg_result["thrust_total"]),
                "thrust_isen":  float(isen_exit["thrust"]),
                # Diagnostics
                "mass_err_reg": mass_err_reg,
                "fallback_reg": bool(reg_result["used_fallback"]),
            })

            # Capture NPR 6.5, Jet-A1 profiles for axial profile panel
            if math.isclose(npr, 6.5) and fuel_name == "Jet-A1":
                selected_profiles = {
                    "regular": reg_profile,
                    "isen":    _normalize_profile(isen_profile),
                }

    df = pd.DataFrame(rows)
    csv_path = output_results / "pinn_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    if not selected_profiles:
        raise RuntimeError("Failed to collect panel-A profiles for NPR 6.5, Jet-A1")

    # -----------------------------------------------------------------------
    # Figure: 2×2 research-paper layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 11))
    outer = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.38)

    # ------------------------------------------------------------------
    # Panel A — Axial profiles 2×2 sub-grid (top-left)
    # ------------------------------------------------------------------
    # We create the sub-grid inside outer[0, 0] using GridSpecFromSubplotSpec.
    # The individual sub-axes are added directly to fig; we use a placeholder
    # invisible axis to carry the panel title.
    ax_a_holder = fig.add_subplot(outer[0, 0])
    ax_a_holder.set_visible(False)

    inner_a = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0],
                                      wspace=0.40, hspace=0.45)
    variables = [
        ("T",   "Temperature [K]"),
        ("p",   "Pressure [Pa]"),
        ("u",   "Velocity [m/s]"),
        ("rho", "Density [kg/m³]"),
    ]
    model_styles = [
        ("Regular PINN", "regular", _COLOR_PINN, "-",  1.8),
        ("Isentropic",   "isen",    _COLOR_ISEN, "--", 1.2),
    ]
    for k, (var, ylabel) in enumerate(variables):
        ax = fig.add_subplot(inner_a[k // 2, k % 2])
        for label, key, color, style, lw in model_styles:
            prof = selected_profiles[key]
            x_arr  = np.asarray(prof["x"], dtype=float)
            x_norm = x_arr / max(float(np.max(x_arr)), 1e-12)
            ax.plot(x_norm, np.asarray(prof[var], dtype=float),
                    color=color, linestyle=style, linewidth=lw,
                    label=label if k == 0 else "_nolegend_")
        ax.set_xlabel("x / L", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="best")

    # Super-title for the sub-grid
    fig.text(
        ax_a_holder.get_position().x0 + ax_a_holder.get_position().width / 2,
        ax_a_holder.get_position().y1 + 0.005,
        "(a) Axial flow profiles — NPR 6.5, Jet-A1",
        ha="center", va="bottom", fontsize=9, fontweight="semibold",
    )

    # ------------------------------------------------------------------
    # Panel B — Thrust vs NPR (top-right)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[0, 1])
    _panel_b_thrust_vs_npr(ax_b, df)

    # ------------------------------------------------------------------
    # Panel C — Exit temperature parity scatter (bottom-left)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[1, 0])
    _panel_c_parity_scatter(ax_c, df)

    # ------------------------------------------------------------------
    # Panel D — Metrics table (bottom-right)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(outer[1, 1])
    _panel_d_metrics_table(ax_d, df)

    # ------------------------------------------------------------------
    # Final layout & save
    # ------------------------------------------------------------------
    fig.suptitle(
        "Regular PINN — Nozzle Flow Benchmark\n"
        "NPR sweep 4–8, three SAF blends",
        fontsize=12, y=0.99,
    )
    fallback_reg = int(df["fallback_reg"].sum())
    if fallback_reg > 0:
        fig.text(
            0.5,
            0.015,
            (
                f"Note: analytical fallback was used in {fallback_reg}/{len(df)} sweep cases "
                "after the wrapper physics checks failed."
            ),
            ha="center",
            fontsize=8,
        )
    fig.savefig(output_plots / "pinn_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    thrust_rmse_reg = float(np.sqrt(np.mean(((df["thrust_reg"] - df["thrust_isen"]) / df["thrust_isen"]) ** 2)) * 100.0)
    temp_rmse_reg   = float(np.sqrt(np.mean(((df["T_exit_reg"] - df["T_exit_isen"]) / df["T_exit_isen"]) ** 2)) * 100.0)
    mass_reg        = float(np.mean(df["mass_err_reg"]) * 100.0)

    summary = [
        ("Thrust RMSE %", thrust_rmse_reg),
        ("T_exit RMSE %", temp_rmse_reg),
        ("Mass error %",  mass_reg),
    ]
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Regular PINN':>14}")
    print("-" * 60)
    for metric, reg_v in summary:
        print(f"  {metric:<18} {reg_v:>12.3f}")
    print("-" * 60)
    print(f"  Fallback triggered     {fallback_reg:>12d}")
    print()
    print(f"Saved CSV  : {csv_path}")
    print(f"Saved plot : {output_plots / 'pinn_comparison.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
