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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import colormaps

import streamlit as st

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

def _frustum_mesh(x0: float, x1: float, r0: float, r1: float, res: int = 56):
    """(Legacy helper — kept for reference, not called.)"""
    theta = np.linspace(0, 2 * np.pi, res + 1)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    pts_lo = np.column_stack([np.full(res + 1, x0), r0 * cos_t, r0 * sin_t])
    pts_hi = np.column_stack([np.full(res + 1, x1), r1 * cos_t, r1 * sin_t])
    points = np.vstack([pts_lo, pts_hi])  # shape (2*(res+1), 3)

    n = res + 1
    quads = []
    for j in range(res):
        quads += [4, j, j + 1, j + 1 + n, j + n]
    faces = np.array(quads, dtype=np.int_)
    return pv.PolyData(points, faces)


def _cap_disk(x: float, r_inner: float, r_outer: float, res: int = 56):
    """Flat annular cap at axial position x."""
    import pyvista as pv

    theta = np.linspace(0, 2 * np.pi, res + 1)
    pts_out = np.column_stack([np.full(res + 1, x), r_outer * np.cos(theta), r_outer * np.sin(theta)])
    pts_in  = np.column_stack([np.full(res + 1, x), r_inner * np.cos(theta), r_inner * np.sin(theta)])
    points  = np.vstack([pts_out, pts_in])
    n = res + 1
    quads = []
    for j in range(res):
        quads += [4, j, j + 1, j + 1 + n, j + n]
    return pv.PolyData(points, np.array(quads, dtype=np.int_))


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
        "Multi-quantity engine heatmap"
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
        run_btn = st.button("▶  Run Simulation", type="primary", width="stretch")

    # ── Quantity selector (top of main area) ──────────────────────────────────
    qty_label = st.radio(
        "Visualise quantity",
        list(_QTY_KEY.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Load engine ───────────────────────────────────────────────────────────
    engine = load_engine()

    # ── Run / restore simulation ──────────────────────────────────────────────
    need_run = run_btn or ("sim_result" not in st.session_state)
    if need_run:
        with st.spinner("Running engine simulation…"):
            try:
                result, lca = run_simulation(
                    engine, jet_pct, hefa_pct, ft_pct, atj_pct, phi, eff
                )
                st.session_state["sim_result"] = result
                st.session_state["sim_lca"]    = lca
            except Exception as exc:
                st.error(f"Simulation error: {exc}")
                st.stop()

    result = st.session_state["sim_result"]
    lca    = st.session_state.get("sim_lca", 1.0)
    stages = extract_stages(result)

    # ── Layout: visualisation | metrics ───────────────────────────────────────
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
        tsfc_mg   = perf.get("tsfc_mg_per_Ns", float("inf"))
        eta_th    = perf.get("thermal_efficiency", 0.0)
        m_fuel    = perf.get("fuel_mass_flow", 0.0)

        st.metric("Thrust",       f"{thrust_kN:.1f} kN")
        st.metric("TSFC",         f"{tsfc_mg:.1f} mg/(N·s)" if tsfc_mg < 1e9 else "∞")
        st.metric("η (kinetic)",  f"{eta_th * 100:.1f} %")
        st.metric("Fuel flow",    f"{m_fuel:.3f} kg/s")

        st.subheader("Emissions")
        nox = emis.get("NOx_g_s",     0.0)
        co2 = emis.get("Net_CO2_g_s", 0.0)
        co  = emis.get("CO_g_s",      0.0)
        st.metric("NOx",  f"{nox:.3f} g/s")
        st.metric("CO₂",  f"{co2:.0f} g/s")
        st.metric("CO",   f"{co:.4f} g/s")
        st.metric("LCA factor", f"{lca:.2f}")

        st.divider()

        # Stage data table
        st.subheader("Stage Data")
        df = pd.DataFrame(stages).set_index("name")
        df.columns = ["T (K)", "p (Pa)", "u (m/s)", "ρ (kg/m³)", "Mach"]
        df = df.round({"T (K)": 1, "p (Pa)": 0, "u (m/s)": 1, "ρ (kg/m³)": 4, "Mach": 3})
        st.dataframe(df, width="stretch")

        st.divider()

        # Fuel blend doughnut
        render_blend_chart(jet_pct, hefa_pct, ft_pct, atj_pct)


if __name__ == "__main__":
    main()
