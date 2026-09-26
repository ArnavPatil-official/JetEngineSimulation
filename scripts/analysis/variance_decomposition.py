"""
Variance decomposition, objective correlations, and CORSIA Monte Carlo
rank-stability for the final optimization studies (Phase 2.7 / P2.3.4).

Consumes the trial databases written by scripts/optimization/optimize_blend.py:
- outputs/results/optimization_results.csv            (free phi)
- outputs/results/optimization_results_phi_frozen.csv (AB8: blends only)

Produces:
1. Variance shares of each objective attributable to phi, blend fractions,
   and the CORSIA LCA draw (random-forest permutation importance, normalized
   by explained variance; R^2 reported so unexplained variance is visible).
   -> outputs/results/variance_decomposition.csv (+ figure)
2. Objective correlation matrix -> outputs/results/objective_correlation.csv
   (+ heatmap figure)
3. CORSIA Monte Carlo rank stability (plan AB7): N_MC seeded common-scenario
   draws; per draw the lifecycle axis is recomputed for ALL trials and Pareto
   membership re-evaluated. Reports, per baseline Pareto member, the fraction
   of draws in which it remains Pareto-optimal, plus P5-P95 lifecycle bands.
   Decision rule: membership < 50% stable => all blend-selection statements
   become scenario-conditional in the manuscript.
   -> outputs/results/lca_rank_stability.csv, Pareto band figure.

Note (documented deviation): the plan named scripts/visualization/
pareto_visual.py / visualize_results.py for these figures; those files carry
uncommitted local modifications, so the Phase 2 analysis figures are emitted
here instead, leaving them untouched.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

RESULTS = PROJECT_ROOT / "outputs" / "results"
PLOTS = PROJECT_ROOT / "outputs" / "plots"

OBJECTIVES = ["TSFC", "SpecThrust", "Lifecycle_CO2e", "NOx_correlation"]
# Orientation for Pareto: convert to all-minimize
MINIMIZE_SIGNS = np.array([1.0, -1.0, 1.0, 1.0])

# dataviz reference palette
C1, C2, C3 = "#2a78d6", "#1baf7a", "#eda100"


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Vectorized Pareto front for an (n, k) array oriented all-minimize."""
    le = (values[None, :, :] <= values[:, None, :]).all(axis=2)
    lt = (values[None, :, :] < values[:, None, :]).any(axis=2)
    dominated = (le & lt).any(axis=1)
    return ~dominated


def variance_shares(df: pd.DataFrame, objective: str, features: list,
                    seed: int) -> dict:
    X = df[features].to_numpy()
    y = df[objective].to_numpy()
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    r2 = rf.score(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=10, random_state=seed,
                                n_jobs=-1)
    imp = np.clip(pi.importances_mean, 0.0, None)
    total = imp.sum() if imp.sum() > 0 else 1.0
    shares = {f: float(v / total) for f, v in zip(features, imp)}
    shares["R2"] = float(r2)
    return shares


V5_FEATURES = ["p_HEFA", "p_FT", "p_ATJ", "lcef_draw_HEFA", "lcef_draw_FT", "lcef_draw_ATJ"]
V5_OUTPUTS = ["ff", "tsfc_mg_Ns", "T4", "nox_corr_g_s", "lifecycle_g_s_point_draw"]


def main_v5(seed: int, n_mc: int) -> None:
    """P6.3 (outputs/phase6/p63_registration.json): blends at MATCHED take-off
    thrust. Variance shares, Spearman correlations and CORSIA rank stability
    over the 256 Sobol design points (reference blends excluded)."""
    outs = {k: RESULTS / f"{k}_v5.csv" for k in
            ("variance_decomposition", "objective_correlation", "lca_rank_stability")}
    for pth in outs.values():
        if pth.exists():
            raise SystemExit(f"{pth} exists; refusing to overwrite")
    df = pd.read_csv(RESULTS / "blend_matched_thrust_v5.csv")
    df = df[(df["Mode"] == "TAKE-OFF") & df["blend"].str.startswith("S")].reset_index(drop=True)
    if (df["status"] != "converged").any():
        raise SystemExit("unreachable take-off design points; registered analysis needs all 256")
    rows = []
    for obj in V5_OUTPUTS:
        sh = variance_shares(df, obj, V5_FEATURES, seed)
        sh.update({"Objective": obj, "rel_range_pct": float(100 * (df[obj].max() - df[obj].min())
                                                              / df[obj].mean())})
        rows.append(sh)
    shares = pd.DataFrame(rows)[["Objective", "R2", "rel_range_pct", *V5_FEATURES]]
    shares.to_csv(outs["variance_decomposition"], index=False)
    df[V5_OUTPUTS].corr(method="spearman").to_csv(outs["objective_correlation"])

    with open(PROJECT_ROOT / "data" / "corsia_lca_values.yaml") as fh:
        corsia = yaml.safe_load(fh)
    tri = {p: corsia["pathways"][p]["triangular"] for p in ("HEFA", "FT", "ATJ")}
    lcef_c = {"JetA": corsia["baseline_fossil_gCO2e_MJ"], **{p: t["mode"] for p, t in tri.items()}}
    from simulation.fuels import JET_A1, HEFA_SPK, FT_SPK, ATJ_SPK
    lhv = {"JetA": JET_A1.LHV_MJ_per_kg, "HEFA": HEFA_SPK.LHV_MJ_per_kg,
           "FT": FT_SPK.LHV_MJ_per_kg, "ATJ": ATJ_SPK.LHV_MJ_per_kg}

    def lifecycle(draw):
        return df["ff"].to_numpy() * sum(df[f"p_{k}"].to_numpy() * lhv[k] * draw[k] for k in lhv)

    fixed = df[["tsfc_mg_Ns", "nox_corr_g_s"]].to_numpy()
    central = pareto_mask(np.column_stack([fixed[:, 0], lifecycle(lcef_c), fixed[:, 1]]))
    rng = np.random.default_rng(seed)
    persist, samples = np.zeros(len(df)), np.empty((n_mc, len(df)))
    for d in range(n_mc):
        draw = {"JetA": lcef_c["JetA"], **{p: rng.triangular(t["min"], t["mode"], t["max"])
                                          for p, t in tri.items()}}
        samples[d] = lifecycle(draw)
        persist += pareto_mask(np.column_stack([fixed[:, 0], samples[d], fixed[:, 1]]))
    persist /= n_mc
    out = df[["blend", "p_JetA", "p_HEFA", "p_FT", "p_ATJ", "ff", "tsfc_mg_Ns", "T4",
              "nox_corr_g_s"]].copy()
    out["Lifecycle_central"] = lifecycle(lcef_c)
    out["ParetoOptimal_central"] = central
    out["Pareto_persistence"] = persist
    for q in (5, 50, 95):
        out[f"Lifecycle_P{q}"] = np.percentile(samples, q, axis=0)
    out.to_csv(outs["lca_rank_stability"], index=False)
    members = out[out["ParetoOptimal_central"]]
    stable = float((members["Pareto_persistence"] >= 0.5).mean()) if len(members) else float("nan")
    print(shares.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Central Pareto members: {len(members)}; stable in >= 50 % of {n_mc} draws: "
          f"{stable * 100:.1f} % -> blend-selection statements "
          f"{'may be stated with scenario bands' if stable >= 0.5 else 'must be scenario-conditional'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-mc", type=int, default=1000,
                    help="Monte Carlo CORSIA scenario draws (default 1000)")
    ap.add_argument("--v5", action="store_true",
                    help="P6.3 matched-thrust blend study (blend_matched_thrust_v5.csv)")
    args = ap.parse_args()
    if args.v5:
        main_v5(args.seed, args.n_mc)
        return

    with open(PROJECT_ROOT / "data" / "corsia_lca_values.yaml") as fh:
        corsia = yaml.safe_load(fh)
    tri = {p: corsia["pathways"][p]["triangular"] for p in ("HEFA", "FT", "ATJ")}
    baseline = corsia["baseline_fossil_gCO2e_MJ"]
    # Component LHVs must match optimize_blend.py
    from simulation.fuels import JET_A1, HEFA_SPK, FT_SPK, ATJ_SPK
    LHV = {"JetA": JET_A1.LHV_MJ_per_kg, "HEFA": HEFA_SPK.LHV_MJ_per_kg,
           "FT": FT_SPK.LHV_MJ_per_kg, "ATJ": ATJ_SPK.LHV_MJ_per_kg}

    # ------------------------------------------------------------------
    # 1 + 2: variance shares and objective correlations, both studies
    # ------------------------------------------------------------------
    share_rows, corr_frames = [], {}
    for study, fname in (("free_phi", "optimization_results.csv"),
                         ("phi_frozen", "optimization_results_phi_frozen.csv")):
        path = RESULTS / fname
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        features = ["SAF_Total", "HEFA_Frac", "FT_Frac", "ATJ_Frac",
                    "LCEF_blend_gCO2e_MJ"]
        if study == "free_phi":
            features = ["Phi"] + features
        for obj in OBJECTIVES:
            s = variance_shares(df, obj, features, args.seed)
            s.update({"Study": study, "Objective": obj})
            share_rows.append(s)
        corr = df[OBJECTIVES].corr()
        corr_frames[study] = corr
        corr.to_csv(RESULTS / f"objective_correlation_{study}.csv")

    shares = pd.DataFrame(share_rows)
    cols = ["Study", "Objective", "R2", "Phi", "SAF_Total", "HEFA_Frac",
            "FT_Frac", "ATJ_Frac", "LCEF_blend_gCO2e_MJ"]
    shares = shares.reindex(columns=[c for c in cols if c in shares.columns])
    shares.to_csv(RESULTS / "variance_decomposition.csv", index=False)
    print("\nVariance shares (normalized permutation importance):")
    print(shares.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Figure: stacked variance shares per objective (free-phi study)
    sub = shares[shares.Study == "free_phi"].set_index("Objective")
    feat_cols = [c for c in ["Phi", "SAF_Total", "HEFA_Frac", "FT_Frac",
                             "ATJ_Frac", "LCEF_blend_gCO2e_MJ"]
                 if c in sub.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(sub))
    palette = [C1, C2, C3, "#008300", "#4a3aa7", "#e34948"]
    for color, feat in zip(palette, feat_cols):
        vals = sub[feat].to_numpy()
        ax.bar(sub.index, vals, bottom=bottom, label=feat, color=color,
               edgecolor="white", linewidth=1)
        bottom += vals
    ax.set_ylabel("Share of explained variance")
    ax.set_title("What drives each objective? (free-φ study; "
                 "RF permutation importance)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS / "variance_decomposition.png", dpi=300)
    plt.close(fig)

    # Correlation heatmap (free-phi)
    if "free_phi" in corr_frames:
        corr = corr_frames["free_phi"]
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(corr)), corr.index)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                        va="center", fontsize=9,
                        color="#1a1a19" if abs(corr.iloc[i, j]) < 0.6 else "white")
        fig.colorbar(im, label="Pearson r")
        ax.set_title("Objective correlations (free-φ study)")
        fig.tight_layout()
        fig.savefig(PLOTS / "objective_correlation_heatmap.png", dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3: CORSIA Monte Carlo rank stability (common scenario per draw)
    # ------------------------------------------------------------------
    path = RESULTS / "optimization_results.csv"
    if not path.exists():
        print("[skip] rank stability: results CSV missing")
        return
    df = pd.read_csv(path)
    n = len(df)
    ff = df["Fuel_Flow_kg_s"].to_numpy()
    pen = df["TIT_Penalty"].to_numpy()
    fr = {k: df[f"{k}_Frac"].to_numpy() for k in ("HEFA", "FT", "ATJ")}
    jet = df["JetA_Frac"].to_numpy()

    fixed_axes = df[["TSFC", "SpecThrust", "NOx_correlation"]].to_numpy()
    baseline_pareto = df["ParetoOptimal"].to_numpy(bool)

    rng = np.random.default_rng(args.seed)
    persist = np.zeros(n)
    lifecycle_samples = np.empty((args.n_mc, n))
    for d in range(args.n_mc):
        draw = {"JetA": baseline}
        for p in ("HEFA", "FT", "ATJ"):
            draw[p] = rng.triangular(tri[p]["min"], tri[p]["mode"], tri[p]["max"])
        lifecycle = ff * (jet * LHV["JetA"] * draw["JetA"] +
                          fr["HEFA"] * LHV["HEFA"] * draw["HEFA"] +
                          fr["FT"] * LHV["FT"] * draw["FT"] +
                          fr["ATJ"] * LHV["ATJ"] * draw["ATJ"]) * pen
        lifecycle_samples[d] = lifecycle
        vals = np.column_stack([fixed_axes[:, 0], fixed_axes[:, 1],
                                lifecycle, fixed_axes[:, 2]])
        vals = vals * MINIMIZE_SIGNS[None, :]
        persist += pareto_mask(vals)
    persist /= args.n_mc

    df_out = df[["Trial", "SAF_Total", "HEFA_Frac", "FT_Frac", "ATJ_Frac",
                 "Phi", "TSFC", "SpecThrust", "NOx_correlation",
                 "ParetoOptimal"]].copy()
    df_out["Pareto_persistence"] = persist
    df_out["Lifecycle_P5"] = np.percentile(lifecycle_samples, 5, axis=0)
    df_out["Lifecycle_P50"] = np.percentile(lifecycle_samples, 50, axis=0)
    df_out["Lifecycle_P95"] = np.percentile(lifecycle_samples, 95, axis=0)
    df_out.to_csv(RESULTS / "lca_rank_stability.csv", index=False)

    members = df_out[baseline_pareto]
    stable_frac = float((members["Pareto_persistence"] >= 0.5).mean())
    print(f"\nCORSIA MC rank stability ({args.n_mc} draws, seed {args.seed}):")
    print(f"  Baseline Pareto members: {int(baseline_pareto.sum())}")
    print(f"  Median persistence of members: "
          f"{members['Pareto_persistence'].median():.2f}")
    print(f"  Members stable in >=50% of draws: {stable_frac*100:.1f}%")
    print(f"  DECISION RULE (AB7): blend-selection statements "
          f"{'may be stated with scenario bands' if stable_frac >= 0.5 else 'must be scenario-conditional'}")

    # Representative balanced solution: free-phi Pareto member closest to
    # the normalized ideal point (selection rule recorded here so the
    # manuscript's representative solution is script-generated, not ad hoc).
    p = df[df["ParetoOptimal"]].copy()
    obj = p[OBJECTIVES].copy()
    obj["SpecThrust"] = -obj["SpecThrust"]  # orient all-minimize
    z = (obj - obj.min()) / (obj.max() - obj.min())
    p["ideal_distance"] = np.sqrt((z ** 2).sum(axis=1))
    rep = p.nsmallest(1, "ideal_distance")
    rep.to_csv(RESULTS / "representative_solution.csv", index=False)
    r0 = rep.iloc[0]
    print(f"\nRepresentative balanced solution (min normalized distance to ideal):")
    print(f"  Trial {int(r0['Trial'])}: TSFC {r0['TSFC']:.2f} mg/(N·s), "
          f"spec thrust {r0['SpecThrust']:.1f} N·s/kg, "
          f"lifecycle CO2e {r0['Lifecycle_CO2e']:.0f} g/s, "
          f"NOx(corr) {r0['NOx_correlation']:.1f} g/s, "
          f"SAF {r0['SAF_Total']*100:.1f}%, phi {r0['Phi']:.4f}")

    # Figure: TSFC vs lifecycle with P5-P95 bands for Pareto members
    fig, ax = plt.subplots(figsize=(8, 5.5))
    non = df_out[~baseline_pareto]
    ax.scatter(non["TSFC"], non["Lifecycle_P50"], s=12, color="#c9c8bf",
               label="Dominated trials", zorder=1)
    m = df_out[baseline_pareto]
    ax.errorbar(m["TSFC"], m["Lifecycle_P50"],
                yerr=[m["Lifecycle_P50"] - m["Lifecycle_P5"],
                      m["Lifecycle_P95"] - m["Lifecycle_P50"]],
                fmt="o", ms=6, color=C1, ecolor=C1, elinewidth=1.1,
                capsize=2.5, label="Pareto members (P5–P95 CORSIA band)",
                zorder=3)
    ax.set_xlabel("TSFC [mg/(N·s)] (TIT-penalized objective)")
    ax.set_ylabel("Lifecycle CO₂e [g/s] (CORSIA scenario median + band)")
    ax.set_title("Pareto set under CORSIA feedstock uncertainty")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS / "pareto_lca_bands.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'variance_decomposition.csv'}, "
          f"{RESULTS / 'lca_rank_stability.csv'}, figures in {PLOTS}")


if __name__ == "__main__":
    main()
