"""
Sajben Transonic Diffuser — LE-PINN Validation

Validates the 2D Locally-Enhanced PINN against the Hseih, Wardlaw, Collins &
Coakley (1987) experimental dataset for the Sajben transonic diffuser.

Target: weak shock case (inlet Mach ≈ 0.46, no shock-induced separation).

Metrics reported
----------------
* L2 relative error — wall static-pressure coefficient Cp = P/P₀ (upper and
  lower surfaces).
* L2 relative error — axial velocity profiles at x/H = 1.73, 2.88, 4.61, 6.34.
* Mean absolute axisymmetric continuity residual across the domain.

Geometry flags
--------------
All geometry mismatches between the Sajben 2-D planar diffuser and the
LE-PINN's default axisymmetric turbofan-nozzle setup are printed to stdout
before the metrics.

Normalisation strategy
----------------------
The LE-PINN was trained on synthetic isentropic data generated for the
turbofan-nozzle geometry (NPR = 6.5, Throat_Radius = 0.05 m).  For this
cross-geometry validation the inputs are normalised with a MinMaxNormalizer
refitted on the Sajben domain; outputs are denormalised using a normalizer
refitted on isentropic predictions for the Sajben-equivalent geometry (see
``_build_sajben_normalizers``).  Predicted and experimental Cp distributions
are compared **after rescaling both to their respective [0,1] ranges** so
the error reflects shape similarity rather than absolute-scale accuracy.
Large errors are expected until the model is retrained on Sajben conditions.

Usage::

    python scripts/validation/sajben_validation.py

"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.nozzle.le_pinn import (
    LE_PINN,
    MinMaxNormalizer,
    compute_rans_residuals,
    parse_sajben_experimental_data,
    parse_sajben_geometry,
    _estimate_wall_distances,
    GAMMA,
    R_GAS,
    CP,
    SUTHERLAND_C1,
    SUTHERLAND_S,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE  = _ROOT / "data" / "raw" / "data.Mach46.txt"
GEOM_FILE  = _ROOT / "data" / "raw" / "sajben.x.fmt"
DEFAULT_MODEL = _ROOT / "models" / "le_pinn_sajben.pt"
FALLBACK_MODEL = _ROOT / "models" / "le_pinn.pt"

# ---------------------------------------------------------------------------
# Sajben physical conditions  (weak shock / Mach-0.46 inlet case)
# ---------------------------------------------------------------------------
# Stagnation conditions (low-speed wind-tunnel, room temperature, ~atmospheric)
P0_SAJBEN: float = 101_325.0   # Pa  — stagnation pressure
T0_SAJBEN: float = 293.0       # K   — stagnation temperature
GAMMA_SAJ: float = 1.4
R_SAJ:     float = 287.0       # J/(kg·K)

# Sajben nozzle geometry (from file / literature)
H_FT: float = 0.14435          # throat height in feet
H_M:  float = H_FT * 0.3048   # ≈ 0.044014 m

# LE-PINN default geometry (for mismatch reporting)
_LEPINN_THROAT_R  = 0.05      # m
_LEPINN_AR        = 1.53
_LEPINN_INLET_AR  = 1.30 ** 2  # inlet_radius_ratio² ≈ 1.69


# ===========================================================================
# 1.  GEOMETRY VERIFICATION
# ===========================================================================

def verify_geometry(geom: dict, verbose: bool = True) -> list:
    """
    Compare Sajben diffuser geometry against LE-PINN defaults.

    Parameters
    ----------
    geom : dict
        Output of :func:`parse_sajben_geometry`.
    verbose : bool
        If True, print a formatted table to stdout.

    Returns
    -------
    list of str
        One entry per check; each prefixed with ``OK``, ``MISMATCH``, or
        ``MISMATCH [CRITICAL]``.
    """
    H_m   = geom["H_m"]
    AR_en = geom["AR_entrance"]
    AR_ex = geom["AR_exit"]

    flags: list = []

    # --- Critical: flow topology ---
    flags.append(
        "MISMATCH [CRITICAL] Flow topology: Sajben diffuser is 2D PLANAR; "
        "LE-PINN models AXISYMMETRIC (RANS) flow.  "
        "There is no exact geometric correspondence."
    )

    # --- Throat scale ---
    r_eq = H_m / 2.0  # pseudo-radius = half-height
    rel  = abs(r_eq - _LEPINN_THROAT_R) / _LEPINN_THROAT_R
    tag  = "OK" if rel < 0.05 else "MISMATCH"
    flags.append(
        f"{tag} Throat scale: Sajben half-height = {r_eq * 100:.3f} cm; "
        f"LE-PINN Throat_Radius = {_LEPINN_THROAT_R * 100:.3f} cm "
        f"(relative diff = {rel * 100:.1f} %)"
    )

    # --- Exit area ratio ---
    rel = abs(AR_ex - _LEPINN_AR) / _LEPINN_AR
    tag = "OK" if rel < 0.05 else "MISMATCH"
    flags.append(
        f"{tag} Exit AR: Sajben = {AR_ex:.3f}; "
        f"LE-PINN default = {_LEPINN_AR:.3f} "
        f"(relative diff = {rel * 100:.1f} %)"
    )

    # --- Entrance area ratio ---
    rel = abs(AR_en - _LEPINN_INLET_AR) / _LEPINN_INLET_AR
    tag = "OK" if rel < 0.10 else "MISMATCH"
    flags.append(
        f"{tag} Entrance AR: Sajben = {AR_en:.3f}; "
        f"LE-PINN inlet AR ≈ {_LEPINN_INLET_AR:.3f} "
        f"(relative diff = {rel * 100:.1f} %)"
    )

    # --- Flow regime ---
    flags.append(
        "MISMATCH [CRITICAL] Flow regime: Sajben weak-shock case has "
        "M_inlet ≈ 0.46 with P₀ ≈ 1 atm; "
        "LE-PINN was trained for NPR = 6.5 (P_in ≈ 6.5 atm, supersonic nozzle)."
    )

    if verbose:
        print("\n--- Geometry / Condition Verification ---")
        for flag in flags:
            print(f"  {flag}")
        print()

    return flags


# ===========================================================================
# 2.  NORMALIZER CONSTRUCTION FOR SAJBEN CONDITIONS
# ===========================================================================

def _isentropic_mach_from_area_ratio(ar: float, supersonic: bool = False) -> float:
    """Newton-iterate to find M from A/A* (isentropic, γ=1.4)."""
    g, gp1, gm1 = GAMMA_SAJ, GAMMA_SAJ + 1.0, GAMMA_SAJ - 1.0
    M = 1.5 if supersonic else 0.5
    for _ in range(60):
        M2   = M * M
        term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2)
        f    = (1.0 / M) * term ** (0.5 * gp1 / gm1) - ar
        dM   = 1e-7
        M2p  = (M + dM) ** 2
        tp   = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2p)
        fp   = (1.0 / (M + dM)) * tp ** (0.5 * gp1 / gm1) - ar
        df   = (fp - f) / dM
        if abs(df) < 1e-14:
            break
        M = M - f / df
        M = max(0.01, min(M, 5.0))
    return M


def _build_sajben_normalizers(
    geom: dict,
    n_axial: int = 60,
    n_normal: int = 25,
) -> tuple:
    """
    Build input and output MinMaxNormalizers calibrated to the Sajben domain.

    Uses isentropic quasi-1D relations for the Sajben geometry and conditions
    (P₀ = 101325 Pa, T₀ = 293 K) to generate a representative set of
    physical values, then fits normalizers on those values.

    Returns
    -------
    norm_in  : MinMaxNormalizer for inputs  (N, 6)
    norm_out : MinMaxNormalizer for outputs (N, 5) — [ρ, u, v, P, T]
    """
    H_m  = geom["H_m"]
    x_full = geom["x_m"][:, 0]
    upper_y_full = geom["upper_wall_y_m"]
    x_vec = np.linspace(x_full.min(), x_full.max(), n_axial)
    upper_y = np.interp(x_vec, x_full, upper_y_full)

    x_pts_list, y_pts_list = [], []
    for xi, yu in zip(x_vec, upper_y):
        y_r = np.linspace(0.0, yu * 0.98, n_normal)
        x_pts_list.append(np.full(n_normal, xi))
        y_pts_list.append(y_r)
    x_pts = np.concatenate(x_pts_list)
    y_pts = np.concatenate(y_pts_list)
    N = len(x_pts)

    # Pseudo-axisymmetric throat / exit areas
    r_t = H_m / 2.0
    AR_ex = geom["AR_exit"]
    r_ex  = r_t * np.sqrt(AR_ex)
    A5 = np.pi * r_t ** 2
    A6 = np.pi * r_ex ** 2

    # Find throat x-location (min upper wall y)
    i_thr   = int(np.argmin(upper_y))
    x_thr_m = x_vec[i_thr]

    # --- Isentropic flow field ---
    # A(x) for pseudo-axisymmetric: π r(x)² where r(x) ≈ upper_y(x)
    r_local = np.interp(x_pts, x_vec, upper_y)
    A_local = np.pi * r_local ** 2
    AR_local = A_local / (np.pi * r_t ** 2)
    AR_local = np.maximum(AR_local, 1.0)

    # Subsonic Mach
    mach = np.where(x_pts <= x_thr_m, 0.5, 1.5)
    for _ in range(60):
        g, gp1, gm1 = GAMMA_SAJ, GAMMA_SAJ + 1.0, GAMMA_SAJ - 1.0
        M2   = mach ** 2
        term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2)
        ar_f = (1.0 / mach) * term ** (0.5 * gp1 / gm1)
        f    = ar_f - AR_local
        dM   = 1e-7
        M2p  = (mach + dM) ** 2
        tp   = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M2p)
        ar_p = (1.0 / (mach + dM)) * tp ** (0.5 * gp1 / gm1)
        df   = (ar_p - ar_f) / dM
        df   = np.where(np.abs(df) < 1e-14, 1e-14, df)
        mach = mach - f / df
        mach = np.clip(mach, 0.01, 3.0)

    T_loc  = T0_SAJBEN / (1.0 + 0.5 * (GAMMA_SAJ - 1.0) * mach ** 2)
    P_loc  = P0_SAJBEN * (T_loc / T0_SAJBEN) ** (GAMMA_SAJ / (GAMMA_SAJ - 1.0))
    rho_loc = P_loc / (R_SAJ * T_loc)
    a_loc  = np.sqrt(GAMMA_SAJ * R_SAJ * T_loc)
    u_loc  = mach * a_loc
    v_loc  = np.zeros(N)

    inputs_np = np.stack([
        x_pts, y_pts,
        np.full(N, A5), np.full(N, A6),
        np.full(N, P0_SAJBEN), np.full(N, T0_SAJBEN),
    ], axis=1).astype(np.float32)

    targets_np = np.stack([
        rho_loc, u_loc, v_loc, P_loc, T_loc
    ], axis=1).astype(np.float32)

    norm_in  = MinMaxNormalizer().fit(torch.from_numpy(inputs_np))
    norm_out = MinMaxNormalizer().fit(torch.from_numpy(targets_np))

    return norm_in, norm_out


# ===========================================================================
# 3.  DOMAIN GRID AND FORWARD PASS
# ===========================================================================

def build_sajben_grid(
    geom: dict,
    n_axial: int = 60,
    n_normal: int = 25,
) -> tuple:
    """
    Build a structured interior grid over the Sajben diffuser domain.

    Returns
    -------
    inputs_raw : (N, 6) float32 tensor  — [x, y, A5, A6, P₀, T₀] in SI units
    x_vec      : (n_axial,) x in metres
    upper_y    : (n_axial,) upper wall y in metres
    """
    H_m      = geom["H_m"]
    x_full   = geom["x_m"][:, 0]
    uy_full  = geom["upper_wall_y_m"]
    x_vec    = np.linspace(x_full.min(), x_full.max(), n_axial)
    upper_y  = np.interp(x_vec, x_full, uy_full)

    x_pts_list, y_pts_list = [], []
    for xi, yu in zip(x_vec, upper_y):
        y_pts = np.linspace(0.0, yu * 0.98, n_normal)
        x_pts_list.append(np.full(n_normal, xi))
        y_pts_list.append(y_pts)
    x_pts = np.concatenate(x_pts_list)
    y_pts = np.concatenate(y_pts_list)
    N = len(x_pts)

    r_t  = H_m / 2.0
    AR   = geom["AR_exit"]
    A5   = np.pi * r_t ** 2
    A6   = np.pi * (r_t * np.sqrt(AR)) ** 2

    inputs_raw = np.stack([
        x_pts, y_pts,
        np.full(N, A5), np.full(N, A6),
        np.full(N, P0_SAJBEN), np.full(N, T0_SAJBEN),
    ], axis=1).astype(np.float32)

    return torch.from_numpy(inputs_raw), x_vec, upper_y


def run_forward_pass(
    model: LE_PINN,
    inputs_raw: torch.Tensor,
    norm_in: MinMaxNormalizer,
    norm_out: MinMaxNormalizer,
) -> torch.Tensor:
    """
    Normalise inputs → forward pass → denormalise [ρ, u, v, P, T].

    Returns
    -------
    preds_phys : (N, 5) tensor in physical units [kg/m³, m/s, m/s, Pa, K]
    """
    wall_dists = _estimate_wall_distances(
        inputs_raw[:, :2],
        inputs_raw[:, 0],
        inputs_raw[:, 1].abs(),
    )
    inputs_n = norm_in.transform(inputs_raw)
    model.eval()
    with torch.no_grad():
        preds_n = model(inputs_n, wall_dists)

    preds_phys = norm_out.inverse_transform(preds_n)[:, :5]
    return preds_phys


# ===========================================================================
# 4.  METRIC FUNCTIONS
# ===========================================================================

def _l2_relative(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error ‖pred − ref‖₂ / ‖ref‖₂  (absolute if ‖ref‖≈0)."""
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(pred - ref)) / (denom + 1e-30)


def _normalise_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-30)


def compute_wall_cp_errors(
    preds_phys: torch.Tensor,
    inputs_raw: torch.Tensor,
    x_vec: np.ndarray,
    upper_y: np.ndarray,
    exp_data: dict,
    n_normal: int,
) -> dict:
    """
    Compute L2 errors for wall Cp = P/P₀ (upper and lower surfaces).

    Shape-comparison strategy: both predicted and experimental Cp profiles
    are normalised to [0,1] before computing the L2 error, so the metric
    reflects pressure distribution shape rather than absolute scale.

    The throat x-location (min upper-wall y) is used to map experimental
    X/H* coordinates to physical metres.

    Returns
    -------
    dict with keys ``l2_upper``, ``l2_bot``, ``x_throat_m``, ``H_m``,
    ``n_top_pts``, ``n_bot_pts``.
    """
    H_m = float((inputs_raw[0, 2].item() / np.pi) ** 0.5)  # r_t = sqrt(A5/π)
    P_pred = preds_phys[:, 3].numpy()          # (N,) — predicted P in Pa
    n_axial = len(x_vec)

    # Upper wall = last normal point in each axial slice
    upper_idx = [i * n_normal + (n_normal - 1) for i in range(n_axial)]
    P_upper_pred = P_pred[upper_idx]

    # Lower wall = first normal point in each axial slice  (y ≈ 0)
    lower_idx = [i * n_normal for i in range(n_axial)]
    P_lower_pred = P_pred[lower_idx]

    # Convert Cp_model to pressure ratio relative to predicted inlet P
    # (use the mean of the 3 most-upstream upper-wall points as reference)
    P_in_pred = float(np.mean(P_upper_pred[:3]))
    Cp_upper_pred = P_upper_pred / (P_in_pred + 1e-30)
    Cp_lower_pred = P_lower_pred / (P_in_pred + 1e-30)

    # Throat location in metres
    i_thr = int(np.argmin(upper_y))
    x_thr = x_vec[i_thr]

    exp_top = exp_data["top_wall"]
    exp_bot = exp_data["bot_wall"]

    # Map experimental X/H* to metres (x = x/H * H_m + x_throat)
    x_exp_top_m = exp_top["xh"] * H_m + x_thr
    x_exp_bot_m = exp_bot["xh"] * H_m + x_thr

    x_lo, x_hi = float(x_vec[0]), float(x_vec[-1])
    mask_top = (x_exp_top_m >= x_lo) & (x_exp_top_m <= x_hi)
    mask_bot = (x_exp_bot_m >= x_lo) & (x_exp_bot_m <= x_hi)

    result = {
        "l2_upper": None,
        "l2_bot":   None,
        "x_throat_m": x_thr,
        "H_m": H_m,
        "n_top_pts": int(mask_top.sum()),
        "n_bot_pts": int(mask_bot.sum()),
    }

    if mask_top.sum() > 1:
        # Interpolate model Cp to experimental x-locations
        Cp_model_at_exp_top = np.interp(x_exp_top_m[mask_top], x_vec, Cp_upper_pred)
        Cp_exp_top = exp_top["pp0"][mask_top].astype(float)
        # Normalise both to [0,1] for shape comparison
        Cp_model_norm = _normalise_01(Cp_model_at_exp_top)
        Cp_exp_norm   = _normalise_01(Cp_exp_top)
        result["l2_upper"] = _l2_relative(Cp_model_norm, Cp_exp_norm)

    if mask_bot.sum() > 1:
        Cp_model_at_exp_bot = np.interp(x_exp_bot_m[mask_bot], x_vec, Cp_lower_pred)
        Cp_exp_bot = exp_bot["pp0"][mask_bot].astype(float)
        Cp_model_norm = _normalise_01(Cp_model_at_exp_bot)
        Cp_exp_norm   = _normalise_01(Cp_exp_bot)
        result["l2_bot"] = _l2_relative(Cp_model_norm, Cp_exp_norm)

    return result


def compute_velocity_profile_errors(
    preds_phys: torch.Tensor,
    inputs_raw: torch.Tensor,
    x_vec: np.ndarray,
    upper_y: np.ndarray,
    exp_data: dict,
    n_normal: int,
) -> dict:
    """
    Compute L2 relative errors for axial velocity profiles at the four
    Sajben measurement stations (X/H = 1.729, 2.882, 4.611, 6.340).

    Returns
    -------
    dict keyed by station label; each value is a dict with
    ``l2_error``, ``x_station_m``, ``x_nearest_m``, ``n_exp_pts``.
    """
    H_m   = float((inputs_raw[0, 2].item() / np.pi) ** 0.5)
    i_thr = int(np.argmin(upper_y))
    x_thr = float(x_vec[i_thr])

    u_pred = preds_phys[:, 1].numpy()    # (N,) axial velocity m/s
    y_pred = inputs_raw[:, 1].numpy()    # (N,) y in metres

    results: dict = {}
    for label, vp in exp_data["vel_profiles"].items():
        x_st_m = float(label) * H_m + x_thr

        # Nearest axial index
        i_near = int(np.argmin(np.abs(x_vec - x_st_m)))
        x_near = float(x_vec[i_near])

        # Extract model profile slice at this axial station
        sl = slice(i_near * n_normal, (i_near + 1) * n_normal)
        y_sl = y_pred[sl]
        u_sl = u_pred[sl]

        # Sort by y for interpolation
        order = np.argsort(y_sl)
        y_sl  = y_sl[order]
        u_sl  = u_sl[order]

        # Experimental Y/H → metres
        y_exp_m  = vp["yh"].astype(float) * H_m
        u_exp_ms = vp["u_ms"].astype(float)

        # Clip to model y-range
        y_lo, y_hi = float(y_sl[0]), float(y_sl[-1])
        mask = (y_exp_m >= y_lo) & (y_exp_m <= y_hi)
        if mask.sum() < 2:
            results[label] = {
                "l2_error": None,
                "x_station_m": x_st_m,
                "x_nearest_m": x_near,
                "n_exp_pts": int(mask.sum()),
            }
            continue

        u_model_interp = np.interp(y_exp_m[mask], y_sl, u_sl)
        err = _l2_relative(u_model_interp, u_exp_ms[mask])

        results[label] = {
            "l2_error": err,
            "x_station_m": x_st_m,
            "x_nearest_m": x_near,
            "n_exp_pts": int(mask.sum()),
        }

    return results


def compute_continuity_error(
    model: LE_PINN,
    inputs_raw: torch.Tensor,
    norm_in: MinMaxNormalizer,
) -> float:
    """
    Compute mean absolute axisymmetric continuity residual.

    Uses :func:`compute_rans_residuals` (mass equation only) with
    autograd through the model.

    Returns
    -------
    float — mean |res_mass|
    """
    wall_dists = _estimate_wall_distances(
        inputs_raw[:, :2],
        inputs_raw[:, 0],
        inputs_raw[:, 1].abs(),
    )
    inputs_n = norm_in.transform(inputs_raw).requires_grad_(True)
    # model.train() is needed for autograd (dropout etc. off in eval, but we
    # need the computational graph)
    model.train()
    preds = model(inputs_n, wall_dists)
    res_mass, _, _, _, _ = compute_rans_residuals(inputs_n, preds)
    with torch.no_grad():
        mean_abs = float(res_mass.abs().mean().item())
    model.eval()
    return mean_abs


# ===========================================================================
# 5.  MAIN
# ===========================================================================

def main(model_file: Path | None = None) -> dict:
    print("=" * 72)
    print("LE-PINN  ×  SAJBEN TRANSONIC DIFFUSER  —  VALIDATION REPORT")
    print("Weak shock case (inlet Mach ≈ 0.46, no shock-induced separation)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1 — Parse experimental data
    # ------------------------------------------------------------------
    print("\n[1/5] Parsing experimental data ...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Experimental data not found: {DATA_FILE}")
    exp_data = parse_sajben_experimental_data(str(DATA_FILE))

    n_top  = len(exp_data["top_wall"]["xh"])
    n_bot  = len(exp_data["bot_wall"]["xh"])
    print(f"  Top-wall  pressure: {n_top} points, "
          f"x/H ∈ [{exp_data['top_wall']['xh'].min():.3f}, "
          f"{exp_data['top_wall']['xh'].max():.3f}]")
    print(f"  Bot-wall  pressure: {n_bot} points, "
          f"x/H ∈ [{exp_data['bot_wall']['xh'].min():.3f}, "
          f"{exp_data['bot_wall']['xh'].max():.3f}]")
    for lbl in sorted(exp_data["vel_profiles"], key=float):
        vp = exp_data["vel_profiles"][lbl]
        print(f"  Velocity X/H={lbl}: {len(vp['yh'])} profile pts, "
              f"u ∈ [{vp['u_ms'].min():.1f}, {vp['u_ms'].max():.1f}] m/s")

    # ------------------------------------------------------------------
    # Step 2 — Parse Sajben geometry
    # ------------------------------------------------------------------
    print("\n[2/5] Parsing Sajben geometry (sajben.x.fmt) ...")
    if not GEOM_FILE.exists():
        raise FileNotFoundError(f"Geometry file not found: {GEOM_FILE}")
    geom = parse_sajben_geometry(str(GEOM_FILE))

    print(f"  Grid: {geom['ni']} × {geom['nj']}  (kmax = 1)")
    print(f"  Throat height H = {geom['H_in']:.4f} in = "
          f"{geom['H_m'] * 100:.4f} cm  (expected 4.407 cm)")
    print(f"  Entrance AR  = {geom['AR_entrance']:.4f}  (expected ≈ 1.40)")
    print(f"  Exit AR      = {geom['AR_exit']:.4f}  (expected ≈ 1.50)")
    print(f"  x range (in) = [{geom['x_vec'][0]:.3f}, {geom['x_vec'][-1]:.3f}]")

    # Throat-height sanity check
    H_expected_in = H_M / 0.0254  # 1.7322 in
    H_err = abs(geom["H_in"] - H_expected_in) / H_expected_in
    tag   = "OK" if H_err < 0.02 else "MISMATCH"
    print(f"  Throat H check: {geom['H_in']:.4f} in vs expected "
          f"{H_expected_in:.4f} in  → {tag} (rel err {H_err*100:.2f} %)")

    # Area ratio checks
    for label, measured, expected in [
        ("Entrance AR", geom["AR_entrance"], 1.40),
        ("Exit AR",     geom["AR_exit"],     1.50),
    ]:
        err = abs(measured - expected) / expected
        tag = "OK" if err < 0.02 else "MISMATCH"
        print(f"  {label}: measured {measured:.4f} vs expected {expected:.2f}"
              f"  → {tag} (rel err {err*100:.2f} %)")

    # ------------------------------------------------------------------
    # Step 3 — Geometry / condition verification
    # ------------------------------------------------------------------
    print("\n[3/5] Geometry and condition verification ...")
    mismatch_flags = verify_geometry(geom, verbose=True)

    # ------------------------------------------------------------------
    # Step 4 — Build grid, normalizers, load model, run forward pass
    # ------------------------------------------------------------------
    N_AXIAL, N_NORMAL = 60, 25
    print("[4/5] Building grid, normalizers, and running forward pass ...")

    inputs_raw, x_vec, upper_y = build_sajben_grid(
        geom, n_axial=N_AXIAL, n_normal=N_NORMAL
    )
    print(f"  Domain: {len(inputs_raw)} interior points "
          f"({N_AXIAL} axial × {N_NORMAL} normal)")

    # Resolve model file
    if model_file is None:
        model_file = DEFAULT_MODEL if DEFAULT_MODEL.exists() else FALLBACK_MODEL

    model = LE_PINN()
    ckpt = None
    if model_file.exists():
        try:
            try:
                ckpt = torch.load(str(model_file), map_location="cpu",
                                  weights_only=True)
            except TypeError:
                ckpt = torch.load(str(model_file), map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded weights: {model_file.name}")
        except Exception as exc:
            warnings.warn(
                f"Could not load model weights ({exc}); using random init.",
                RuntimeWarning, stacklevel=1,
            )
            ckpt = None
    else:
        warnings.warn(
            f"Model file not found ({model_file}); using random init.",
            RuntimeWarning, stacklevel=1,
        )

    # --- Normalizers ---
    # Input normalizer: always fit on the Sajben grid so spatial inputs map to
    # [0,1] exactly as the model expects during training (regardless of the
    # physical coordinate range).
    # Output normalizer: use the checkpoint's if available so denormalized
    # outputs reflect the physical scale the model was trained on.
    norm_in_fresh, norm_out_fresh = _build_sajben_normalizers(
        geom, N_AXIAL, N_NORMAL
    )
    norm_in = norm_in_fresh  # always use Sajben-domain input normalizer

    if ckpt is not None and "output_norm_min" in ckpt:
        norm_out = MinMaxNormalizer()
        norm_out.data_min = ckpt["output_norm_min"]
        norm_out.data_max = ckpt["output_norm_max"]
        print(f"  Input normalizer: fitted on Sajben domain.")
        print(f"  Output normalizer: loaded from checkpoint.")
    else:
        norm_out = norm_out_fresh
        print(f"  Normalizers fitted on Sajben-calibrated isentropic data.")

    preds_phys = run_forward_pass(model, inputs_raw, norm_in, norm_out)
    print(f"  Forward pass complete. "
          f"P range: [{preds_phys[:,3].min():.1f}, "
          f"{preds_phys[:,3].max():.1f}] Pa  "
          f"u range: [{preds_phys[:,1].min():.1f}, "
          f"{preds_phys[:,1].max():.1f}] m/s")

    # ------------------------------------------------------------------
    # Step 5 — Compute metrics
    # ------------------------------------------------------------------
    print("\n[5/5] Computing validation metrics ...")

    cp_result = compute_wall_cp_errors(
        preds_phys, inputs_raw, x_vec, upper_y,
        exp_data, n_normal=N_NORMAL,
    )

    vel_result = compute_velocity_profile_errors(
        preds_phys, inputs_raw, x_vec, upper_y,
        exp_data, n_normal=N_NORMAL,
    )

    cont_err = compute_continuity_error(model, inputs_raw, norm_in)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    print("\n  Wall Cp = P/P₀  (normalised shape L2 relative error):")
    print("  ─" * 36)
    if cp_result["l2_upper"] is not None:
        print(f"    Upper wall L2 error : {cp_result['l2_upper']:.4f}  "
              f"({cp_result['n_top_pts']} exp points in model x-range)")
    else:
        print("    Upper wall          : insufficient overlap with model domain")
    if cp_result["l2_bot"] is not None:
        print(f"    Lower wall L2 error : {cp_result['l2_bot']:.4f}  "
              f"({cp_result['n_bot_pts']} exp points in model x-range)")
    else:
        print("    Lower wall          : insufficient overlap with model domain")

    print(f"\n  Axial velocity profiles (L2 relative error vs Sajben data):")
    print("  ─" * 36)
    for lbl in sorted(vel_result.keys(), key=float):
        info = vel_result[lbl]
        if info["l2_error"] is not None:
            print(f"    X/H = {lbl}  L2 = {info['l2_error']:.4f}  "
                  f"(model x = {info['x_nearest_m']*100:.2f} cm, "
                  f"{info['n_exp_pts']} exp pts)")
        else:
            print(f"    X/H = {lbl}  — insufficient profile overlap")

    print(f"\n  Axisymmetric continuity residual (mean |res_mass|): "
          f"{cont_err:.4e}")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("─" * 72)
    print("  * Cp errors reflect pressure SHAPE mismatch (both profiles")
    print("    normalised to [0,1]). Values < 0.10 indicate good shape match.")
    print("  * Velocity errors are absolute L2 relative to experimental values.")
    print("  * Continuity residual measures PDE satisfaction (physics loss).")
    print("  * LARGE ERRORS EXPECTED: model was trained on NPR=6.5 axisymmetric")
    print("    geometry; Sajben is 2-D planar at ~1 atm.  Retrain on Sajben")
    print("    conditions for quantitative validation.")
    print("=" * 72)

    return {
        "l2_cp_upper":        cp_result["l2_upper"],
        "l2_cp_lower":        cp_result["l2_bot"],
        "vel_profile_errors": vel_result,
        "continuity_error":   cont_err,
        "mismatch_flags":     mismatch_flags,
    }


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Sajben diffuser validation")
    _parser.add_argument(
        "--model", type=str, default=None,
        help=f"Path to model checkpoint (default: {DEFAULT_MODEL})",
    )
    _args = _parser.parse_args()
    _mpath = Path(_args.model) if _args.model else None
    results = main(model_file=_mpath)
