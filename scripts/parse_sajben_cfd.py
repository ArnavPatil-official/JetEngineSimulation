#!/usr/bin/env python3
"""
Generate LE-PINN training dataset from the NASA NPARC Sajben transonic diffuser.

Data sources used
-----------------
- ``data/raw/sajben.x.fmt``               : actual 81 × 51 Sajben geometry
- ``data/raw/cfd_datasets/nasa/sajben.dat.{1..20}`` : inlet BCs + back pressures

The binary ADF/CGNS files (``sajben.cfl``, ``sajben.cgd``) require pyCGNS
(not in requirements.txt) and are skipped.  Instead, a quasi-1D normal-shock
solver is run on the real Sajben geometry to produce three families of flow:

  A) Shock inside domain  — 25 shock stations swept across the diverging section
  B) Fully-supersonic     —  5 cases (no internal shock, overexpanded exit)
  C) Subsonic throughout  —  5 cases (unchoked, varying inlet Mach)

Total: 35 cases × 4 131 grid points ≈ 144 k training points.

Output
------
``data/processed/master_shock_dataset.pt``
    ``{"inputs": Tensor(N, 6), "targets": Tensor(N, 9)}``
    inputs  : [x, y, A5, A6, P_in, T_in] — physical units (m, m², Pa, K)
    targets : [ρ, u, v, P, T, 0, 0, 0, μ_eff]

Run
---
    python scripts/parse_sajben_cfd.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.le_pinn import parse_sajben_geometry

# ---------------------------------------------------------------------------
# Physical constants  (mirror le_pinn.py exactly)
# ---------------------------------------------------------------------------
GAMMA: float = 1.4
R_GAS: float = 287.0
SUTHERLAND_C1: float = 1.458e-6
SUTHERLAND_S: float = 110.4      # K
PSI_TO_PA: float = 6_894.757
RANKINE_TO_K: float = 5.0 / 9.0


# ============================================================================
# 1.  Parse .dat run-configuration files
# ============================================================================

def parse_dat_configs(dat_dir: Path) -> dict:
    """Return inlet conditions dict from the first parseable .dat file."""
    for n in range(1, 21):
        fpath = dat_dir / f"sajben.dat.{n}"
        if not fpath.exists():
            continue
        m = re.search(
            r"Freestream static\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            fpath.read_text(),
        )
        if m:
            return {
                "M":       float(m.group(1)),
                "P_s_psi": float(m.group(2)),
                "T_s_R":   float(m.group(3)),
            }
    # Fallback: documented Sajben weak-shock inlet conditions
    return {"M": 0.46, "P_s_psi": 16.937, "T_s_R": 504.26}


# ============================================================================
# 2.  Isentropic / normal-shock helpers
# ============================================================================

def _area_mach(M: np.ndarray, g: float = GAMMA) -> np.ndarray:
    """A/A* for Mach M (vectorised)."""
    gm1, gp1 = g - 1.0, g + 1.0
    return (1.0 / np.maximum(M, 1e-9)) * (
        (2.0 / gp1) * (1.0 + 0.5 * gm1 * M ** 2)
    ) ** (0.5 * gp1 / gm1)


def _mach_from_ar(
    AR: np.ndarray,
    supersonic: bool = False,
    g: float = GAMMA,
    n_iter: int = 80,
) -> np.ndarray:
    """Solve A/A* = AR for M (Newton, scalar or array)."""
    AR = np.asarray(AR, dtype=np.float64)
    M = np.full_like(AR, 1.5 if supersonic else 0.5)
    lo, hi = (1.0, 20.0) if supersonic else (1e-4, 1.0 - 1e-6)
    for _ in range(n_iter):
        f = _area_mach(M, g) - AR
        dM = 1e-6
        df = (_area_mach(M + dM, g) - _area_mach(M, g)) / dM
        df = np.where(np.abs(df) < 1e-14, 1e-14 * np.sign(df + 1e-20), df)
        M = np.clip(M - f / df, lo, hi)
    return M


def _normal_shock(M1: float, g: float = GAMMA):
    """Return (M2, P02/P01) for normal shock at upstream Mach M1."""
    gm1, gp1 = g - 1.0, g + 1.0
    M1sq = max(float(M1) ** 2, 1.0)
    M2 = float(np.sqrt(max((M1sq + 2.0 / gm1) / (2.0 * g / gm1 * M1sq - 1.0), 0.0)))
    t1 = ((gp1 / 2.0 * M1sq) / (1.0 + gm1 / 2.0 * M1sq)) ** (g / gm1)
    t2 = (2.0 * g / gp1 * M1sq - gm1 / gp1) ** (-1.0 / gm1)
    return M2, float(t1 * t2)


def _p_isen(M: np.ndarray, P0: float, g: float = GAMMA) -> np.ndarray:
    return P0 / (1.0 + 0.5 * (g - 1.0) * M ** 2) ** (g / (g - 1.0))


def _t_isen(M: np.ndarray, T0: float, g: float = GAMMA) -> np.ndarray:
    return T0 / (1.0 + 0.5 * (g - 1.0) * M ** 2)


def _sutherland(T: np.ndarray) -> np.ndarray:
    return SUTHERLAND_C1 * T ** 1.5 / (T + SUTHERLAND_S)


# ============================================================================
# 3.  Quasi-1D flow solvers  (three regimes)
# ============================================================================

def _build_supersonic_M(AR_vec: np.ndarray, idx_throat: int) -> np.ndarray:
    """Supersonic Mach distribution (diverging section only)."""
    ni = len(AR_vec)
    M_sup = np.ones(ni)
    if idx_throat + 1 < ni:
        M_sup[idx_throat + 1:] = _mach_from_ar(
            AR_vec[idx_throat + 1:], supersonic=True
        )
    return np.clip(M_sup, 1.0, 20.0)


def solve_shock_at_station(
    AR_vec: np.ndarray,
    M_sub_conv: np.ndarray,
    M_sup: np.ndarray,
    idx_throat: int,
    k_shock: int,
    P0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quasi-1D flow with a normal shock fixed at diverging station *k_shock*.

    Upstream  : subsonic converging → M=1 at throat → supersonic.
    Downstream: subsonic isentropic with post-shock stagnation P0_post.

    Returns
    -------
    M_axial   : (ni,) Mach distribution
    P0_axial  : (ni,) local stagnation pressure
    """
    ni = len(AR_vec)
    M_axial = np.copy(M_sub_conv)
    M_axial[idx_throat] = 1.0
    M_axial[idx_throat + 1: k_shock] = M_sup[idx_throat + 1: k_shock]

    M1 = float(M_sup[k_shock])
    M2, P02_P01 = _normal_shock(M1)
    P0_post = P0 * P02_P01
    M_axial[k_shock] = M2

    # Effective subsonic A* downstream of shock
    ar_k_sub = float(_area_mach(np.array([M2]))[0])
    A_star_ratio = AR_vec[k_shock] / ar_k_sub   # A*_sub_eff / A_geom_throat
    for i in range(k_shock + 1, ni):
        AR_eff = max(float(AR_vec[i]) / A_star_ratio, 1.0 + 1e-6)
        M_axial[i] = float(
            _mach_from_ar(np.array([AR_eff]), supersonic=False)[0]
        )

    P0_axial = np.full(ni, P0)
    P0_axial[k_shock:] = P0_post
    return M_axial, P0_axial


def solve_fully_supersonic(
    AR_vec: np.ndarray,
    M_sub_conv: np.ndarray,
    M_sup: np.ndarray,
    idx_throat: int,
    P0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully choked + supersonic diverging section (no internal shock).
    """
    ni = len(AR_vec)
    M_axial = np.copy(M_sub_conv)
    M_axial[idx_throat] = 1.0
    M_axial[idx_throat + 1:] = M_sup[idx_throat + 1:]
    P0_axial = np.full(ni, P0)
    return M_axial, P0_axial


def solve_subsonic_unchoked(
    AR_vec: np.ndarray,
    h_inlet_to_throat: float,
    M_in: float,
    P0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully subsonic (unchoked) flow referenced to the inlet Mach.

    The inlet Mach M_in < 0.46 drives a flow that accelerates slightly into
    the throat then decelerates through the diffuser — never reaching M=1.

    Returns
    -------
    M_axial   : (ni,) Mach distribution
    P0_axial  : (ni,) stagnation pressure (constant, no shocks)
    """
    ni = len(AR_vec)
    # Effective area ratio: A(x)/A_inlet × (A_inlet/A*_sub)
    AR_inlet_star = float(_area_mach(np.array([M_in]))[0])   # A_inlet / A*_sub
    # A(x)/A*_sub = (h_vec[x]/h_throat) / h_inlet_to_throat × AR_inlet_star
    #             = AR_vec[x] / h_inlet_to_throat × AR_inlet_star
    AR_sub_eff = AR_vec / h_inlet_to_throat * AR_inlet_star   # (ni,)
    AR_sub_eff = np.maximum(AR_sub_eff, 1.0 + 1e-6)

    M_axial = _mach_from_ar(AR_sub_eff, supersonic=False)
    P0_axial = np.full(ni, P0)
    return M_axial, P0_axial


# ============================================================================
# 4.  Pack one flow case into input/target arrays
# ============================================================================

def _pack_case(
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    ni: int,
    nj: int,
    M_axial: np.ndarray,
    P0_axial: np.ndarray,
    T0: float,
    A5: float,
    A6: float,
    P0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast 1-D axial solution to the full 2-D grid."""
    T_axial  = _t_isen(M_axial, T0)
    P_axial  = _p_isen(M_axial, 1.0) * P0_axial   # P = P_ratio × P0_local
    rho_axial = P_axial / (R_GAS * T_axial)
    u_axial  = M_axial * np.sqrt(GAMMA * R_GAS * T_axial)
    mu_axial = _sutherland(T_axial)

    # Each axial station i applies to all nj radial points
    T_2d   = np.repeat(T_axial,   nj)
    P_2d   = np.repeat(P_axial,   nj)
    rho_2d = np.repeat(rho_axial, nj)
    u_2d   = np.repeat(u_axial,   nj)
    mu_2d  = np.repeat(mu_axial,  nj)
    v_2d   = np.zeros(ni * nj)
    zero   = np.zeros(ni * nj)

    inputs = np.column_stack([
        x_flat, y_flat,
        np.full(ni * nj, A5),
        np.full(ni * nj, A6),
        np.full(ni * nj, P0),
        np.full(ni * nj, T0),
    ]).astype(np.float32)

    targets = np.column_stack([
        rho_2d, u_2d, v_2d, P_2d, T_2d,
        zero, zero, zero, mu_2d,
    ]).astype(np.float32)

    return inputs, targets


# ============================================================================
# 5.  Main dataset generator
# ============================================================================

def generate_sajben_dataset(
    geom_path: str,
    dat_dir: str,
    n_shock_cases: int = 25,
    n_supersonic: int = 5,
    n_subsonic: int = 5,
    output_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build and save ``master_shock_dataset.pt``.

    Three flow families are generated:
      A) Shock in domain   : *n_shock_cases* stations swept across the diverging section
      B) Fully supersonic  : *n_supersonic* cases (no internal shock)
      C) Subsonic unchoked : *n_subsonic* cases (varying inlet Mach, no shock)
    """
    if output_path is None:
        output_path = str(
            REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt"
        )

    # ------------------------------------------------------------------
    # Load geometry
    # ------------------------------------------------------------------
    print("Loading Sajben geometry …")
    geom = parse_sajben_geometry(geom_path)
    x_m = geom["x_m"]                    # (ni, nj)
    y_m = geom["y_m"]                    # (ni, nj)
    ni, nj = geom["ni"], geom["nj"]
    H_m  = geom["H_m"]
    upper_y = geom["upper_wall_y_m"]     # (ni,)
    lower_y = geom["lower_wall_y_m"]     # (ni,)
    AR_exit = geom["AR_exit"]

    idx_throat = int(np.argmin(upper_y))
    h_vec = upper_y - lower_y
    h_throat = float(h_vec[idx_throat])
    h_inlet  = float(h_vec[0])
    AR_vec   = np.maximum(h_vec / h_throat, 1.0)       # A/A_throat, ≥ 1
    h_inlet_to_throat = h_inlet / h_throat              # ≈ 1.30

    # A5, A6: axisymmetric equivalents (mirror train_sajben.py / le_pinn.py)
    r_throat = H_m / 2.0
    A5 = float(np.pi * r_throat ** 2)
    A6 = A5 * float(AR_exit)

    x_flat = x_m.ravel()   # (ni*nj,)  C-order: index = i*nj + j
    y_flat = y_m.ravel()

    print(f"  Grid: {ni} × {nj}  |  throat idx={idx_throat}"
          f"  |  A5={A5:.4e} m²  A6={A6:.4e} m²")
    print(f"  x range: {x_m.min():.3f} … {x_m.max():.3f} m")
    print(f"  h_inlet/h_throat = {h_inlet_to_throat:.4f}")

    # ------------------------------------------------------------------
    # Pre-compute Mach distributions on the full grid
    # ------------------------------------------------------------------
    # Subsonic (converging) branch — used for choked cases upstream of throat
    M_sub_conv = _mach_from_ar(AR_vec, supersonic=False)

    # Supersonic branch (diverging section)
    M_sup = _build_supersonic_M(AR_vec, idx_throat)

    # ------------------------------------------------------------------
    # Parse inlet stagnation conditions from .dat files
    # ------------------------------------------------------------------
    inlet = parse_dat_configs(Path(dat_dir))
    M_in_ref = inlet["M"]
    P_s_in   = inlet["P_s_psi"] * PSI_TO_PA
    T_s_in   = inlet["T_s_R"]   * RANKINE_TO_K
    fac = 1.0 + 0.5 * (GAMMA - 1.0) * M_in_ref ** 2
    P0 = P_s_in * fac ** (GAMMA / (GAMMA - 1.0))
    T0 = T_s_in * fac

    print(f"\nInlet:       M={M_in_ref}, P_s={P_s_in:.0f} Pa, T_s={T_s_in:.1f} K")
    print(f"Stagnation:  P0={P0:.0f} Pa,  T0={T0:.1f} K\n")

    # ------------------------------------------------------------------
    # Family A: shock at k_shock ∈ {idx_throat+1, …, ni-2}
    # ------------------------------------------------------------------
    # Unique M_sup values (last ~5 stations are flat at M_sup=1.854)
    div_all = list(range(idx_throat + 1, ni - 1))
    # Find stations where M_sup is still changing (filter flat plateau near exit)
    M_sup_div = M_sup[div_all]
    unique_end = int(np.searchsorted(np.diff(M_sup_div) < 1e-4, True))
    unique_end = max(unique_end, len(div_all) - 1)  # fallback: use all
    div_valid = div_all[: unique_end + 1]

    shock_stations = np.round(
        np.linspace(0, len(div_valid) - 1, min(n_shock_cases, len(div_valid)))
    ).astype(int)
    shock_stations = [div_valid[i] for i in shock_stations]

    all_inputs:  list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    print(f"Family A — shock in domain ({len(shock_stations)} cases):")
    for ci, k in enumerate(shock_stations):
        M_axial, P0_axial = solve_shock_at_station(
            AR_vec, M_sub_conv, M_sup, idx_throat, k, P0
        )
        inp, tgt = _pack_case(x_flat, y_flat, ni, nj, M_axial, P0_axial,
                               T0, A5, A6, P0)
        all_inputs.append(inp);  all_targets.append(tgt)
        P_back_approx = float(_p_isen(M_axial[-1:], P0_axial[-1])[0])
        print(f"  [{ci+1:02d}/{len(shock_stations)}] k={k:2d}"
              f"  x_shock={x_m[k,0]:.3f} m"
              f"  M1={M_sup[k]:.3f} → M2={M_axial[k]:.3f}"
              f"  P_back≈{P_back_approx:.0f} Pa")

    # ------------------------------------------------------------------
    # Family B: fully supersonic (no internal shock)
    # ------------------------------------------------------------------
    print(f"\nFamily B — fully supersonic ({n_supersonic} cases):")
    M_axial_sup, P0_axial_sup = solve_fully_supersonic(
        AR_vec, M_sub_conv, M_sup, idx_throat, P0
    )
    # All n_supersonic copies are identical flow fields with same P0, T0
    for ci in range(n_supersonic):
        inp, tgt = _pack_case(x_flat, y_flat, ni, nj,
                               M_axial_sup, P0_axial_sup, T0, A5, A6, P0)
        all_inputs.append(inp);  all_targets.append(tgt)
    P_exit_sup = float(_p_isen(M_axial_sup[-1:], P0)[0])
    print(f"  M_exit={M_axial_sup[-1]:.3f},  P_exit≈{P_exit_sup:.0f} Pa  "
          f"(×{n_supersonic} copies for weighting)")

    # ------------------------------------------------------------------
    # Family C: subsonic unchoked (varying inlet Mach)
    # ------------------------------------------------------------------
    print(f"\nFamily C — subsonic unchoked ({n_subsonic} cases):")
    M_in_values = np.linspace(0.10, 0.44, n_subsonic)
    for ci, M_in_sub in enumerate(M_in_values):
        M_axial_sub, P0_axial_sub = solve_subsonic_unchoked(
            AR_vec, h_inlet_to_throat, M_in_sub, P0
        )
        inp, tgt = _pack_case(x_flat, y_flat, ni, nj,
                               M_axial_sub, P0_axial_sub, T0, A5, A6, P0)
        all_inputs.append(inp);  all_targets.append(tgt)
        print(f"  [{ci+1:02d}/{n_subsonic}] M_in={M_in_sub:.3f}"
              f"  M_throat≈{M_axial_sub[idx_throat]:.3f}"
              f"  M_exit≈{M_axial_sub[-1]:.3f}")

    # ------------------------------------------------------------------
    # Concatenate and clean
    # ------------------------------------------------------------------
    inputs_all  = np.concatenate(all_inputs,  axis=0)
    targets_all = np.concatenate(all_targets, axis=0)

    valid = (
        np.isfinite(inputs_all).all(axis=1)
        & np.isfinite(targets_all).all(axis=1)
        & (targets_all[:, 0] > 0)    # ρ > 0
        & (targets_all[:, 3] > 0)    # P > 0
        & (targets_all[:, 4] > 0)    # T > 0
    )
    inputs_all  = inputs_all[valid]
    targets_all = targets_all[valid]

    n_dropped = (~valid).sum()
    print(f"\nTotal valid points : {len(inputs_all):,}"
          f"  (dropped {n_dropped:,} non-finite/unphysical)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inputs":  torch.from_numpy(inputs_all),
            "targets": torch.from_numpy(targets_all),
        },
        out,
    )
    print(f"Saved → {out}")
    print(f"  inputs  shape: {inputs_all.shape}")
    print(f"  targets shape: {targets_all.shape}")
    return inputs_all, targets_all


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    generate_sajben_dataset(
        geom_path=str(REPO_ROOT / "data" / "raw" / "sajben.x.fmt"),
        dat_dir=str(REPO_ROOT / "data" / "raw" / "cfd_datasets" / "nasa"),
        n_shock_cases=25,
        n_supersonic=5,
        n_subsonic=5,
        output_path=str(
            REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt"
        ),
    )
