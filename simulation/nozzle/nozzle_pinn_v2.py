"""
Legacy compatibility helpers for nozzle choking checks.

This module preserves the v2 test import path used by
``tests/test_choking_detection.py``.
"""

from __future__ import annotations

from typing import Tuple


def compute_critical_pressure_ratio(gamma: float) -> float:
    """
    Return critical static-to-stagnation pressure ratio at M=1.

    Formula:
        p*/p0 = (2/(gamma+1))**(gamma/(gamma-1))
    """
    if gamma <= 1.0:
        raise ValueError(f"Unphysical gamma={gamma}. Expected gamma > 1.")
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def detect_choking(
    p_inlet: float,
    p_ambient: float,
    gamma: float = 1.33,
) -> Tuple[bool, float]:
    """
    Detect whether nozzle flow is choked based on back-pressure ratio.

    Returns:
        ``(is_choked, p_critical)``
    """
    if p_inlet <= 0.0 or p_ambient < 0.0:
        raise ValueError(
            f"Invalid pressures: p_inlet={p_inlet}, p_ambient={p_ambient}. "
            "Expected p_inlet > 0 and p_ambient >= 0."
        )

    pr_critical = compute_critical_pressure_ratio(gamma)
    p_critical = p_inlet * pr_critical

    # Choked when back pressure is at or below critical pressure.
    is_choked = p_ambient <= p_critical

    return bool(is_choked), float(p_critical)
