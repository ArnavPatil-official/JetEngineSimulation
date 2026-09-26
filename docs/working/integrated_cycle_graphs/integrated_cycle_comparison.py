#!/usr/bin/env python3
"""
Integrated ICAO LTO fuel-flow comparison for the regular PINN nozzle only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrated_engine import FUEL_LIBRARY, IntegratedTurbofanEngine


PLOTS_DIR = REPO_ROOT / "outputs" / "plots"
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
ICAO_CSV_PATH = REPO_ROOT / "data" / "icao_engine_data.csv"
RESULTS_CSV_PATH = RESULTS_DIR / "integrated_cycle_results.csv"
COMPARISON_PLOT_PATH = PLOTS_DIR / "integrated_cycle_comparison.png"
STATS_PLOT_PATH = PLOTS_DIR / "statistical_tests.png"

LTO_MODES = ["Idle", "Approach", "Climb", "Takeoff"]
LTO_SCALES = {
    "Idle": {"mass_scale": 0.15, "pi_scale": 0.15, "phi": 0.26},
    "Approach": {"mass_scale": 0.35, "pi_scale": 0.40, "phi": 0.35},
    "Climb": {"mass_scale": 0.85, "pi_scale": 0.90, "phi": 0.46},
    "Takeoff": {"mass_scale": 1.00, "pi_scale": 1.00, "phi": 0.55},
}
ICAO_FUEL_FLOW = {
    "Idle": 0.244,
    "Approach": 0.643,
    "Climb": 2.050,
    "Takeoff": 2.327,
}
MODE_TO_CSV_LABEL = {
    "Idle": "IDLE",
    "Approach": "APPROACH",
    "Climb": "CLIMB",
    "Takeoff": "TAKE-OFF",
}
FALLBACK_PINN_VALUES = np.array([0.2316, 0.6620, 2.1066, 2.6809], dtype=float)


def _ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_fuel_flow(performance: dict[str, Any]) -> float:
    return float(performance.get("fuel_flow", performance.get("fuel_mass_flow", float("nan"))))


def load_icao_statistics() -> pd.DataFrame:
    icao_df = pd.read_csv(ICAO_CSV_PATH)
    stats = icao_df.groupby("Mode")["Fuel Flow (kg/s)"].agg(["mean", "std"])

    rows: list[dict[str, float | str]] = []
    for mode in LTO_MODES:
        csv_label = MODE_TO_CSV_LABEL[mode]
        std_value = np.nan
        if csv_label in stats.index:
            std_value = float(stats.loc[csv_label, "std"])
        elif mode == "Climb":
            climb_std = float(
                np.nanmean(
                    [
                        stats.loc["APPROACH", "std"] if "APPROACH" in stats.index else np.nan,
                        stats.loc["TAKE-OFF", "std"] if "TAKE-OFF" in stats.index else np.nan,
                    ]
                )
            )
            std_value = 0.08 if np.isnan(climb_std) else climb_std

        rows.append(
            {
                "mode": mode,
                "icao_mean": float(ICAO_FUEL_FLOW[mode]),
                "icao_std": float(std_value),
            }
        )

    return pd.DataFrame(rows)


def get_design_base() -> dict[str, float]:
    engine = IntegratedTurbofanEngine()
    return {
        "mass_flow_core": float(engine.design_point["mass_flow_core"]),
        "pi_c": float(getattr(engine.compressor, "pi_c", 43.2)),
    }


def _run_lto_sweep(
    fuel_blend,
    nozzle_variant: str,
    design_base: dict[str, float],
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run the integrated cycle at all four LTO points.
    Returns dict mapping mode name -> fuel_flow (kg/s).
    """
    fuel_flows: dict[str, float] = {}
    for mode, cfg in LTO_SCALES.items():
        engine = IntegratedTurbofanEngine(nozzle_variant=nozzle_variant)
        engine.design_point["mass_flow_core"] = design_base["mass_flow_core"] * cfg["mass_scale"]

        base_pi_c = design_base.get("pi_c")
        if base_pi_c is not None:
            scaled_pi_c = float(base_pi_c) * cfg["pi_scale"]
            engine.compressor.pi_c = scaled_pi_c
            if "pi_c" in engine.design_point or "pi_c" in design_base:
                engine.design_point["pi_c"] = scaled_pi_c

        result = engine.run_full_cycle(fuel_blend=fuel_blend, phi=cfg["phi"])
        fuel_flows[mode] = _safe_fuel_flow(result["performance"])
        if verbose:
            print(
                f"[{nozzle_variant}] {mode:<8} "
                f"mass_scale={cfg['mass_scale']:.2f} pi_scale={cfg['pi_scale']:.2f} "
                f"phi={cfg['phi']:.2f} fuel_flow={fuel_flows[mode]:.4f} kg/s"
            )
    return fuel_flows


def build_results_dataframe(
    icao_stats: pd.DataFrame,
    pinn_flows: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for mode in LTO_MODES:
        icao_mean = float(icao_stats.loc[icao_stats["mode"] == mode, "icao_mean"].iloc[0])
        icao_std = float(icao_stats.loc[icao_stats["mode"] == mode, "icao_std"].iloc[0])
        pinn_val = float(pinn_flows[mode])
        rows.append(
            {
                "mode": mode,
                "icao_mean": icao_mean,
                "icao_std": icao_std,
                "pinn_fuel_flow": pinn_val,
                "pinn_delta": pinn_val - icao_mean,
                "mass_scale": float(LTO_SCALES[mode]["mass_scale"]),
                "pi_scale": float(LTO_SCALES[mode]["pi_scale"]),
                "phi": float(LTO_SCALES[mode]["phi"]),
            }
        )
    return pd.DataFrame(rows)


def load_integrated_cycle_results(path: Path = RESULTS_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_integrated_cycle_results(df: pd.DataFrame, path: Path = RESULTS_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ensure_integrated_cycle_results(run_if_missing: bool = False, verbose: bool = False) -> pd.DataFrame:
    df = load_integrated_cycle_results()
    if not df.empty or not run_if_missing:
        return df

    icao_stats = load_icao_statistics()
    design_base = get_design_base()
    pinn_flows = _run_lto_sweep(FUEL_LIBRARY["Jet-A1"], "pinn", design_base, verbose=verbose)
    df = build_results_dataframe(icao_stats, pinn_flows)
    save_integrated_cycle_results(df)
    return df


def _model_uncertainty(icao_stds: np.ndarray) -> np.ndarray:
    return np.maximum(0.5 * np.nan_to_num(icao_stds, nan=0.06), 0.03)


def comparison_arrays(results_df: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = results_df.set_index("mode").loc[LTO_MODES].reset_index()
    icao_vals = ordered["icao_mean"].to_numpy(dtype=float)
    icao_stds = ordered["icao_std"].to_numpy(dtype=float)
    pinn_vals = ordered["pinn_fuel_flow"].to_numpy(dtype=float)
    pinn_errs = _model_uncertainty(icao_stds)
    return {
        "icao_vals": icao_vals,
        "icao_stds": icao_stds,
        "pinn_vals": pinn_vals,
        "pinn_errs": pinn_errs,
    }


def plot_grouped_comparison(
    ax: plt.Axes,
    icao_vals: np.ndarray,
    icao_stds: np.ndarray,
    pinn_vals: np.ndarray,
    pinn_errs: np.ndarray,
    title: str,
) -> None:
    x = np.arange(len(LTO_MODES))
    width = 0.34
    model_err_label = "±0.5×ICAO σ propagated"

    ax.bar(
        x - width / 2.0,
        icao_vals,
        width,
        yerr=icao_stds,
        capsize=4,
        label="ICAO (mean ±1σ)",
        color="dimgray",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2.0,
        pinn_vals,
        width,
        yerr=pinn_errs,
        capsize=4,
        label=f"Regular PINN ({model_err_label})",
        color="royalblue",
    )

    for idx, delta in enumerate(pinn_vals - icao_vals):
        ax.text(
            x[idx] + width / 2.0,
            pinn_vals[idx] + pinn_errs[idx] + 0.04,
            f"Δ={delta:+.3f}",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Fuel Flow Rate (kg/s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LTO_MODES, fontsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    takeoff_idx = LTO_MODES.index("Takeoff")
    takeoff_delta = float(pinn_vals[takeoff_idx] - icao_vals[takeoff_idx])
    ax.text(
        0.985,
        0.02,
        (
            f"Model bars use {model_err_label}.\n"
            f"Takeoff drives the largest ICAO gap (Δ={takeoff_delta:+.3f} kg/s)."
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.95},
    )


def save_comparison_plot(results_df: pd.DataFrame, output_path: Path = COMPARISON_PLOT_PATH) -> None:
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
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _confidence_interval(diff: np.ndarray) -> tuple[float, float]:
    n_obs = len(diff)
    mean_diff = float(np.mean(diff))
    if n_obs < 2:
        return mean_diff, mean_diff

    sample_std = float(np.std(diff, ddof=1))
    if sample_std == 0.0:
        return mean_diff, mean_diff

    sem_value = sample_std / np.sqrt(n_obs)
    t_crit = float(t.ppf(0.975, df=n_obs - 1))
    margin = t_crit * sem_value
    return mean_diff - margin, mean_diff + margin


def _paired_ttest(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(x_vals, dtype=float) - np.asarray(y_vals, dtype=float)
    if np.allclose(diff, 0.0, atol=1e-12, rtol=0.0):
        return 0.0, 1.0

    t_stat, p_value = ttest_rel(x_vals, y_vals)
    return float(t_stat), float(p_value)


def run_statistical_tests(results_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    arrays = comparison_arrays(results_df)
    pinn_vals = arrays["pinn_vals"]
    icao_vals = arrays["icao_vals"]

    diff_a = pinn_vals - icao_vals
    t_a, p_a = _paired_ttest(pinn_vals, icao_vals)
    ci_low, ci_high = _confidence_interval(diff_a)
    mape = float(np.mean(np.abs(diff_a) / np.maximum(icao_vals, 1e-12)) * 100.0)
    bias = float(np.mean(diff_a))

    return {
        "test_a": {
            "label": "Regular PINN vs ICAO",
            "t_stat": t_a,
            "p_value": p_a,
            "bias": bias,
            "mape": mape,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    }


def save_statistical_plot(
    results_df: pd.DataFrame,
    stats_results: dict[str, dict[str, Any]],
    output_path: Path = STATS_PLOT_PATH,
) -> None:
    arrays = comparison_arrays(results_df)
    icao_vals = arrays["icao_vals"]
    pinn_vals = arrays["pinn_vals"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))

    all_left = np.concatenate([icao_vals, pinn_vals])
    left_min = float(np.min(all_left) * 0.9)
    left_max = float(np.max(all_left) * 1.1)
    ref_left = np.linspace(left_min, left_max, 200)
    ax_left.scatter(icao_vals, pinn_vals, color="royalblue", s=50)
    ax_left.plot(ref_left, ref_left, color="black", linewidth=1.2)
    ax_left.fill_between(ref_left, ref_left * 0.9, ref_left * 1.1, color="lightgray", alpha=0.25)
    ax_left.set_xlabel("ICAO Fuel Flow (kg/s)")
    ax_left.set_ylabel("Regular PINN Fuel Flow (kg/s)")
    ax_left.set_title("Test A: Regular PINN vs ICAO")
    ax_left.grid(alpha=0.25)
    ax_left.text(
        0.04,
        0.96,
        (
            f"t={stats_results['test_a']['t_stat']:.3f}, "
            f"p={stats_results['test_a']['p_value']:.4f}, "
            f"MAPE={stats_results['test_a']['mape']:.1f}%"
        ),
        transform=ax_left.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.95},
    )

    mode_labels = results_df.set_index("mode").loc[LTO_MODES].index.tolist()
    deltas = pinn_vals - icao_vals
    ax_right.axhline(0.0, color="black", linewidth=1.2)
    ax_right.bar(mode_labels, deltas, color="royalblue", alpha=0.85)
    ax_right.set_xlabel("LTO Mode")
    ax_right.set_ylabel("Regular PINN - ICAO (kg/s)")
    ax_right.set_title("Per-Mode Signed Bias")
    ax_right.grid(alpha=0.25)
    for idx, delta in enumerate(deltas):
        va = "bottom" if delta >= 0 else "top"
        ax_right.text(idx, delta, f"{delta:+.3f}", ha="center", va=va, fontsize=9)
    ax_right.text(
        0.04,
        0.96,
        (
            f"Bias={stats_results['test_a']['bias']:+.3f} kg/s\n"
            f"95% CI=[{stats_results['test_a']['ci_low']:+.3f}, "
            f"{stats_results['test_a']['ci_high']:+.3f}]"
        ),
        transform=ax_right.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_comparison_table(results_df: pd.DataFrame) -> None:
    ordered = results_df.set_index("mode").loc[LTO_MODES].reset_index()
    print("\n" + "=" * 78)
    print("INTEGRATED CYCLE LTO FUEL-FLOW COMPARISON")
    print("=" * 78)
    print(
        f"{'Mode':<10} {'ICAO':>10} {'Regular PINN':>14} {'Δ PINN':>10}"
    )
    print("-" * 78)
    for row in ordered.itertuples(index=False):
        print(
            f"{row.mode:<10} {row.icao_mean:>10.4f} {row.pinn_fuel_flow:>14.4f} "
            f"{row.pinn_delta:>+10.4f}"
        )


def print_statistical_results(stats_results: dict[str, dict[str, Any]]) -> None:
    print("-" * 78)
    print("STATISTICAL TEST A: INTEGRATED CYCLE (REGULAR PINN) VS ICAO")
    print("-" * 78)
    print(f"t-statistic:       {stats_results['test_a']['t_stat']:.6f}")
    print(f"p-value:           {stats_results['test_a']['p_value']:.6f}")
    print(f"Mean signed bias:  {stats_results['test_a']['bias']:+.6f} kg/s")
    print(f"MAPE:              {stats_results['test_a']['mape']:.3f}%")
    print(
        "95% CI on bias:    "
        f"[{stats_results['test_a']['ci_low']:+.6f}, {stats_results['test_a']['ci_high']:+.6f}] kg/s"
    )
    print("Interpretation:    n=4; interpret with caution.")
    if stats_results["test_a"]["p_value"] < 0.05:
        print("Conclusion:        Statistically distinguishable from ICAO at α=0.05.")
    else:
        print("Conclusion:        Insufficient evidence to reject H₀ at α=0.05.")
    print("Mode note:         Takeoff is the dominant model-reference deviation.")

    print("=" * 78 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated engine across ICAO LTO modes for the regular PINN "
            "nozzle, then save comparison and statistical validation figures."
        )
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-mode sweep details while running the integrated engine.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_dirs()

    try:
        icao_stats = load_icao_statistics()
        design_base = get_design_base()
        pinn_flows = _run_lto_sweep(FUEL_LIBRARY["Jet-A1"], "pinn", design_base, verbose=args.verbose)
        results_df = build_results_dataframe(icao_stats, pinn_flows)
        save_integrated_cycle_results(results_df)
        save_comparison_plot(results_df)
        stats_results = run_statistical_tests(results_df)
        save_statistical_plot(results_df, stats_results)
        print_comparison_table(results_df)
        print_statistical_results(stats_results)
        return 0
    except Exception as exc:
        print(f"[ERROR] Integrated cycle comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
