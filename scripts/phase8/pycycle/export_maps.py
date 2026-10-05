#!/usr/bin/env python3
"""P8.4: export the pinned pyCycle 4.4.0 generic maps to JSON (run in catjet-pycycle).

Writes data/maps/pycycle_4.4.0/<Map>.json once per map: every array and
setting of the MapData object, the pyCycle version and the SHA-256 of its
source file. pyCycle is Apache-2.0 (NASA/OpenMDAO); attribution is kept in
each file and in data/maps/pycycle_4.4.0/README.md.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pycycle
import pycycle.api as pyc

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "maps" / "pycycle_4.4.0"
MAPS = ("FanMap", "LPCMap", "HPCMap", "HPTMap", "LPTMap")


def plain(v):
    if isinstance(v, np.ndarray):
        return {"shape": list(v.shape), "data": v.tolist()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v


def main() -> int:
    if pycycle.__version__ != "4.4.0":
        sys.exit(f"pinned pyCycle 4.4.0 required, found {pycycle.__version__}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in MAPS:
        path = OUT / f"{name}.json"
        if path.exists():
            sys.exit(f"{path} exists; write-once")
        m = getattr(pyc, name)
        module = sys.modules[[k for k, v in sys.modules.items()
                              if k.startswith("pycycle.maps") and getattr(v, name, None) is m][0]]
        src = Path(module.__file__)
        doc = {"map": name, "pycycle_version": pycycle.__version__,
               "source_file": f"pycycle/maps/{src.name}",
               "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
               "license": "Apache-2.0 (pyCycle, NASA Glenn / OpenMDAO); data unchanged",
               "fields": {k: plain(v) for k, v in vars(m).items()}}
        with path.open("x") as f:
            f.write(json.dumps(doc, indent=1) + "\n")
        print(name, sorted(doc["fields"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
