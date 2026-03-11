"""Generate and plot 2D axisymmetric aero-engine nozzle contours.

Based on the notation used in Ma et al. (2026), this script builds an
upper-half (y >= 0) nozzle wall profile with three sections:
- Inlet (A2)
- Converging (A3)
- Diverging (A4)

A1, A5, A6 are represented through cross-sectional areas derived from
inlet/throat/exit radii in axisymmetric form (A = pi * r^2).
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NozzleProfile:
    """Container for geometric profile and metadata."""

    x: np.ndarray
    y: np.ndarray
    key_points: Dict[str, tuple[float, float]]
    geometry: Dict[str, float]


def smoothstep(s: np.ndarray) -> np.ndarray:
    """C1-continuous interpolation from 0 to 1."""
    return s * s * (3.0 - 2.0 * s)


def area_ratio_from_npr_isentropic(npr: float, gamma: float = 1.4) -> float:
    """Approximate AR using isentropic relations.

    AR = A_exit / A_throat = A/A*
    with NPR interpreted as P0/Pe.
    """
    if npr <= 1.0:
        raise ValueError("NPR must be greater than 1.0 for supersonic expansion.")

    m2 = (2.0 / (gamma - 1.0)) * (npr ** ((gamma - 1.0) / gamma) - 1.0)
    m2 = max(m2, 1e-12)
    me = np.sqrt(m2)

    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m2)
    ar = (1.0 / me) * term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return float(ar)


def generate_nozzle_profile(
    NPR: float,
    AR: Optional[float],
    Throat_Radius: float,
    inlet_length_factor: float = 1.5,
    converging_length_factor: float = 1.0,
    inlet_radius_ratio: float = 1.30,
    n_inlet: int = 80,
    n_converging: int = 90,
    n_diverging: int = 120,
) -> NozzleProfile:
    """Generate upper-wall contour of a 2D axisymmetric nozzle.

    Required arguments follow the requested interface:
    - NPR: Nozzle Pressure Ratio
    - AR: Area ratio A6/A5 (if None, estimated from isentropic relation)
    - Throat_Radius: throat radius (A5 from pi*r_t^2)

    Geometry assumptions (from Fig. 4-inspired proportions):
    - A2 = inlet_length_factor * throat_height, throat_height = 2 * r_t
    - A3 = converging_length_factor * throat_height
    - A4 grows with AR so divergent section length changes with expansion ratio
    """
    if Throat_Radius <= 0.0:
        raise ValueError("Throat_Radius must be positive.")

    if AR is None:
        AR = area_ratio_from_npr_isentropic(NPR)
    if AR <= 1.0:
        raise ValueError("AR must be greater than 1.0 for converging-diverging nozzles.")

    r_t = float(Throat_Radius)
    h_t = 2.0 * r_t
    r_inlet = inlet_radius_ratio * r_t
    r_exit = r_t * np.sqrt(AR)

    # Axial lengths corresponding to A2, A3, A4, A7.
    A2 = inlet_length_factor * h_t
    A3 = converging_length_factor * h_t
    A4 = h_t * (2.0 + 1.25 * (AR - 1.0))
    A7 = A2 + A3 + A4

    x0 = 0.0
    x1 = A2
    x2 = A2 + A3
    x3 = A7

    # Section 1: inlet (constant radius).
    x_in = np.linspace(x0, x1, n_inlet)
    y_in = np.full_like(x_in, r_inlet)

    # Section 2: converging to throat.
    x_cv = np.linspace(x1, x2, n_converging)
    s_cv = (x_cv - x1) / max(x2 - x1, 1e-12)
    y_cv = r_inlet + (r_t - r_inlet) * smoothstep(s_cv)

    # Section 3: diverging to exit.
    x_dv = np.linspace(x2, x3, n_diverging)
    s_dv = (x_dv - x2) / max(x3 - x2, 1e-12)
    y_dv = r_t + (r_exit - r_t) * smoothstep(s_dv)

    x = np.concatenate([x_in, x_cv[1:], x_dv[1:]])
    y = np.concatenate([y_in, y_cv[1:], y_dv[1:]])

    key_points = {
        "Point 1 (Inlet Start)": (x0, r_inlet),
        "Point 2 (End Inlet / Start Converging)": (x1, r_inlet),
        "Point 3 (Throat)": (x2, r_t),
        "Point 4 (Exit)": (x3, r_exit),
    }

    geometry = {
        "NPR": float(NPR),
        "AR": float(AR),
        "A1": float(np.pi * r_inlet**2),
        "A2": float(A2),
        "A3": float(A3),
        "A4": float(A4),
        "A5": float(np.pi * r_t**2),
        "A6": float(np.pi * r_exit**2),
        "A7": float(A7),
        "r_inlet": float(r_inlet),
        "r_throat": float(r_t),
        "r_exit": float(r_exit),
    }

    return NozzleProfile(x=x, y=y, key_points=key_points, geometry=geometry)


def plot_nozzle_profiles() -> None:
    """Run requested test cases and plot both contours."""
    test_cases = [
        {"NPR": 6.5, "AR": 1.53, "Throat_Radius": 0.05},
        {"NPR": 12.0, "AR": 2.14, "Throat_Radius": 0.05},
    ]

    profiles: list[NozzleProfile] = []
    for case in test_cases:
        profiles.append(
            generate_nozzle_profile(
                NPR=case["NPR"],
                AR=case["AR"],
                Throat_Radius=case["Throat_Radius"],
            )
        )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#1f77b4", "#d62728"]

    for i, profile in enumerate(profiles):
        geom = profile.geometry
        label = f"Case {i+1}: NPR={geom['NPR']:.1f}, AR={geom['AR']:.2f}"
        ax.plot(profile.x, profile.y, color=colors[i], lw=2.2, label=label)

    # Centerline for axisymmetric half-model.
    x_max = max(p.geometry["A7"] for p in profiles)
    ax.plot([0.0, x_max], [0.0, 0.0], "k--", lw=1.2, label="Centerline (y=0)")

    # Mark key points on first profile to avoid clutter.
    p0 = profiles[0]
    for idx, (_, (xp, yp)) in enumerate(p0.key_points.items(), start=1):
        ax.scatter(xp, yp, color="black", s=20, zorder=4)
        ax.text(xp, yp + 0.006, f"P{idx}", ha="center", va="bottom", fontsize=9)

    # Section labels from first profile geometry.
    x0 = p0.key_points["Point 1 (Inlet Start)"][0]
    x1 = p0.key_points["Point 2 (End Inlet / Start Converging)"][0]
    x2 = p0.key_points["Point 3 (Throat)"][0]
    x3 = p0.key_points["Point 4 (Exit)"][0]
    y_annot = max(max(p.y) for p in profiles) * 0.93
    ax.text(0.5 * (x0 + x1), y_annot, "Inlet", ha="center", fontsize=10)
    ax.text(0.5 * (x1 + x2), y_annot, "Converging", ha="center", fontsize=10)
    ax.text(0.5 * (x2 + x3), y_annot, "Diverging", ha="center", fontsize=10)

    ax.set_xlabel("Axial coordinate, x")
    ax.set_ylabel("Radius, y (upper half only)")
    ax.set_title("2D Axisymmetric Nozzle Wall Profiles (Ma et al., Fig. 4 style)")
    ax.grid(alpha=0.25)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.002)
    ax.legend(loc="lower left")

    output_path = "outputs/plots/nozzle_profiles_test_cases.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    print(f"Saved plot: {output_path}")

    # Print key geometric values for transparency.
    for i, p in enumerate(profiles, start=1):
        g = p.geometry
        print(f"\nCase {i}: NPR={g['NPR']:.1f}, AR={g['AR']:.2f}")
        print(
            f"A1={g['A1']:.6f}, A2={g['A2']:.6f}, A3={g['A3']:.6f}, "
            f"A4={g['A4']:.6f}, A5={g['A5']:.6f}, A6={g['A6']:.6f}, A7={g['A7']:.6f}"
        )
        for k, (xp, yp) in p.key_points.items():
            print(f"  {k}: x={xp:.6f}, y={yp:.6f}")


if __name__ == "__main__":
    plot_nozzle_profiles()
