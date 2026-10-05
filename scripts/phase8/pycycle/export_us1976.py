#!/usr/bin/env python3
"""P8.4: export pyCycle 4.4.0's US 1976 atmosphere table (run in catjet-pycycle).

Writes data/thermo/pycycle_us1976.json once (alt ft, T degR, P psi) with the
source SHA-256. pyCycle interpolates it with scipy's Akima1DInterpolator; the
C++ solver ports that interpolation (cpp/catjet_core/offdesign.cpp).
"""

import hashlib
import json
import sys
from pathlib import Path

import pycycle
from pycycle.elements import US1976

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "thermo" / "pycycle_us1976.json"

if pycycle.__version__ != "4.4.0":
    sys.exit("pinned pyCycle 4.4.0 required")
if OUT.exists():
    sys.exit(f"{OUT} exists; write-once")
d = US1976.USatm1976Data
src = Path(US1976.__file__)
doc = {"source": "pycycle/elements/US1976.py", "pycycle_version": pycycle.__version__,
       "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
       "license": "Apache-2.0 (pyCycle); table from digitaldutch.com per the source docstring",
       "alt_ft": d.alt.tolist(), "T_degR": d.T.tolist(), "P_psi": d.P.tolist()}
with OUT.open("x") as f:
    f.write(json.dumps(doc) + "\n")
print(f"wrote {OUT}: {len(doc['alt_ft'])} points")
