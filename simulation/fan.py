"""
0-D fan / bypass-stream model (Phase 2.2, V4).

Models the bypass stream of a high-bypass turbofan at the same fidelity as the
rest of the cycle (0-D station calculations):

- The fan raises the bypass stream from ambient (T0, p0) to p0 * FPR with
  isentropic efficiency eta_fan (temperature rise dT = dT_ideal / eta_fan).
- The bypass nozzle expands the fan stream isentropically back to ambient;
  static-test-stand thrust F_bypass = m_dot_bypass * u_exit (no ram drag).
- Fan shaft work W_fan = m_dot_bypass * cp * dT must be supplied by the
  turbine in addition to compressor work (wired in
  IntegratedTurbofanEngine.run_full_cycle).

Modeling choices (documented for the manuscript):
- The fan is modeled on the BYPASS stream only. The fan-root compression of
  the core stream is not modeled separately: the core's overall pressure
  ratio pi_c is defined compressor-inlet-to-combustor and already represents
  the full core compression, so adding a separate core fan stage would
  double-count work.
- The bypass stream is pure air; constant cp = 1005 J/(kg K) and gamma = 1.4
  are used (standard dry-air values; the ~40 K fan temperature rise does not
  move cp materially).
- Design values FPR = 1.45, eta_fan = 0.90 are standard modern high-BPR
  civil-turbofan magnitudes (e.g., Mattingly, "Elements of Gas Turbine
  Propulsion": civil fan stage FPR ~1.4-1.6, fan polytropic efficiency
  ~0.89-0.91). They are design-class values, not measured Trent 1000 data.
"""

from __future__ import annotations
from typing import Dict
import numpy as np


class Fan:
    """
    0-D bypass fan stage with isentropic-efficiency compression and an
    isentropic bypass nozzle.

    Attributes:
        fpr: Fan pressure ratio (p_exit / p_inlet), > 1
        eta_fan: Fan isentropic efficiency in (0, 1]
        cp: Bypass-air specific heat at constant pressure [J/(kg K)]
        gamma: Bypass-air heat capacity ratio
    """

    def __init__(self, fpr: float = 1.45, eta_fan: float = 0.90,
                 cp: float = 1005.0, gamma: float = 1.4):
        if fpr < 1.0:
            raise ValueError(f"Fan pressure ratio must be >= 1, got {fpr}")
        if not 0.0 < eta_fan <= 1.0:
            raise ValueError(f"Fan efficiency must be in (0, 1], got {eta_fan}")
        self.fpr = fpr
        self.eta_fan = eta_fan
        self.cp = cp
        self.gamma = gamma

    def run(self, T0: float, p0: float, m_dot_bypass: float) -> Dict[str, float]:
        """
        Compute fan exit state, shaft-work demand, and static bypass thrust.

        Args:
            T0: Fan inlet (ambient) temperature [K]
            p0: Fan inlet (ambient) pressure [Pa]
            m_dot_bypass: Bypass-stream mass flow [kg/s]

        Returns:
            Dict with:
                T_exit, p_exit: fan exit stagnation state [K], [Pa]
                dT: actual temperature rise [K]
                work_total: fan shaft work demand [W]
                u_bypass_exit: bypass-nozzle exit velocity [m/s]
                thrust_bypass: static bypass thrust [N]
        """
        if m_dot_bypass < 0:
            raise ValueError("Bypass mass flow must be non-negative")

        exponent = (self.gamma - 1.0) / self.gamma

        # Fan compression with isentropic-efficiency correction
        dT_ideal = T0 * (self.fpr ** exponent - 1.0)
        dT = dT_ideal / self.eta_fan
        T_exit = T0 + dT
        p_exit = p0 * self.fpr
        work_total = m_dot_bypass * self.cp * dT

        # Bypass nozzle: isentropic expansion back to ambient pressure.
        # u = sqrt(2 cp T_fan_exit (1 - (p0/p_exit)^((gamma-1)/gamma)))
        expansion_factor = max(0.0, 1.0 - (p0 / p_exit) ** exponent)
        u_exit = float(np.sqrt(2.0 * self.cp * T_exit * expansion_factor))

        # Static test stand: no inlet momentum (ram) term
        thrust_bypass = m_dot_bypass * u_exit

        return {
            'T_exit': T_exit,
            'p_exit': p_exit,
            'dT': dT,
            'work_total': work_total,
            'u_bypass_exit': u_exit,
            'thrust_bypass': thrust_bypass,
        }
