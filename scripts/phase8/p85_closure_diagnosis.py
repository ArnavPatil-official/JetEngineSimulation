#!/usr/bin/env python3
"""P8.5 development diagnosis of the PSR element-closure error (not a gate record).

Single PSR at the P8.5-A2 probe conditions (A2NOx, phi 1.8, T3 880 K,
P 38 bar, mdot 1 kg/s). Varies one thing at a time and prints the relative
C/H element error and sum(Y) - 1:
  rtol 1e-9 / 1e-11 / 1e-13 (does the error scale with the integrator?),
  ConstPressureReactor vs IdealGasConstPressureReactor,
  advance_to_steady_state alone vs + solve_steady,
  and the element-balanced CRECK mechanism (n-dodecane) as a control.
Output is printed; findings go into a P8.5 amendment before any re-run.
"""

import sys
import warnings
from pathlib import Path

import cantera as ct
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
warnings.filterwarnings("ignore")
ct.suppress_thermo_warnings()


def psr(mech, fuel, rtol, reactor_cls, polish, vol=0.004, phi=1.8, T3=880.0, P=38e5):
    g = ct.Solution(str(ROOT / "data" / mech))
    g.TP = T3, P
    g.set_equivalence_ratio(phi, fuel, "O2:1, N2:3.76")
    els = [e for e in ("C", "H", "O", "N") if e in g.element_names]
    z0 = np.array([g.elemental_mass_fraction(e) for e in els])
    src = ct.Reservoir(g)
    g.equilibrate("HP")
    r = reactor_cls(g, clone=True)
    r.volume = vol
    ex = ct.Reservoir(g)
    m = ct.MassFlowController(src, r, mdot=1.0)
    ct.PressureController(r, ex, primary=m, K=0.01)
    net = ct.ReactorNet([r])
    net.rtol, net.atol = rtol, 1e-20
    try:
        net.advance_to_steady_state()
        if polish:
            net.solve_steady()
    except Exception as e:  # incl. Cantera 3.2 NameError in its own error path
        lines = [l for l in str(e).splitlines() if l.strip() and "***" not in l]
        return {"error": f"{type(e).__name__}: " + (lines[1] if len(lines) > 1 else str(e))[:80]}
    Y = r.phase.Y
    z = np.array([r.phase.elemental_mass_fraction(e) for e in els])
    return {"T": r.phase.T, "sumY-1": float(Y.sum() - 1.0),
            **{f"d{e}": float((zi - z0i) / z0i) for e, zi, z0i in zip(els, z, z0)}}


def main() -> int:
    cases = []
    for rtol in (1e-9, 1e-11, 1e-13):
        cases.append(("A2NOx ConstP rtol %.0e" % rtol, "A2NOx.yaml", "POSF10325:1", rtol, ct.ConstPressureReactor, False))
    cases.append(("A2NOx ConstP 1e-9 + solve_steady", "A2NOx.yaml", "POSF10325:1", 1e-9, ct.ConstPressureReactor, True))
    for rtol in (1e-9, 1e-11, 1e-13):
        cases.append(("A2NOx IdealGasConstP rtol %.0e" % rtol, "A2NOx.yaml", "POSF10325:1", rtol,
                      ct.IdealGasConstPressureReactor, False))
        cases.append(("CRECK IdealGasConstP rtol %.0e" % rtol, "creck_c1c16_full.yaml", "NC12H26:1", rtol,
                      ct.IdealGasConstPressureReactor, False))
    for name, mech, fuel, rtol, cls, pol in cases:
        out = psr(mech, fuel, rtol, cls, pol)
        print(f"{name:38s}", {k: (f"{v:+.2e}" if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
