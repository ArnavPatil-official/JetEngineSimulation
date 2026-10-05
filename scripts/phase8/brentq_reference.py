#!/usr/bin/env python3
"""
P8.1: SciPy brentq reference roots for the C++ port's Catch2 test
(cpp/tests/test_brentq.cpp). Writes cpp/tests/brentq_reference.txt with the
root and function-call count of scipy.optimize.brentq (full_output) for a
few test functions, brackets and tolerances, including the v6 settings
(xtol 1e-12, rtol 4 eps).

Usage: .venv/bin/python scripts/phase8/brentq_reference.py
"""
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "brentq_reference.txt"
FUNCS = {"cubic": lambda x: x * x * x - 2 * x - 5, "cos": lambda x: math.cos(x) - x,
         "exp": lambda x: math.exp(x) - 3.0, "tanh": lambda x: math.tanh(x - 0.3)}
CASES = [("cubic", 2.0, 3.0), ("cubic", 0.0, 4.0), ("cos", 0.0, 1.0), ("cos", -2.0, 3.0),
         ("exp", 0.0, 2.0), ("exp", -5.0, 10.0), ("tanh", 0.0, 3.0), ("tanh", -1.0, 1.5)]
TOLS = [(2e-12, float(4 * np.finfo(float).eps)), (1e-12, float(4 * np.finfo(float).eps)), (1e-6, 1e-10)]

lines = [f"# scipy {scipy.__version__}: name a b xtol rtol root funcalls"]
for name, a, b in CASES:
    for xtol, rtol in TOLS:
        root, r = brentq(FUNCS[name], a, b, xtol=xtol, rtol=rtol, full_output=True)
        lines.append(f"{name} {a!r} {b!r} {xtol!r} {rtol!r} {root:.17g} {r.function_calls}")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(lines) - 1} cases)")
