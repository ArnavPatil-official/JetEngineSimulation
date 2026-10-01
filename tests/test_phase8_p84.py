"""P8.4 numerical contracts: map interpolation and the US 1976 Akima port."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import Akima1DInterpolator, RegularGridInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(os.environ.get("CATJET_BUILD", ROOT / "cpp" / "build"))))
core = pytest.importorskip("catjet_core")
if not hasattr(core, "Hbtf"):
    pytest.skip("catjet_core built without P8.4", allow_module_level=True)
sys.path.insert(0, str(ROOT / "scripts" / "phase8" / "pycycle"))
import compare_hbtf as C  # noqa: E402


@pytest.mark.parametrize("name", ["FanMap", "HPCMap", "HPTMap"])
def test_map_interpolation_matches_scipy_linear_with_extrapolation(name):
    m = C.load_map(name)
    f = json.loads((C.MAPS / f"{name}.json").read_text())["fields"]
    grids = [np.array(p["values"]["data"] if isinstance(p["values"], dict) else p["values"], float)
             for p in f["param_data"]]
    out = f["output_data"][0]
    data = np.array(out["values"]["data"], float)
    ref = RegularGridInterpolator(grids, data, method="linear", bounds_error=False, fill_value=None)
    rng = np.random.default_rng(20261001)
    for _ in range(200):
        x = [rng.uniform(g[0] - 0.1 * (g[-1] - g[0]), g[-1] + 0.1 * (g[-1] - g[0])) for g in grids]
        value, _ = m.outputs[out["name"]](x)
        assert value == pytest.approx(float(ref(x)[0]), rel=1e-12, abs=1e-12)


def test_us1976_akima_matches_scipy():
    e = C.make_engine("matched")
    atm = json.loads(C.US1976.read_text())
    T = Akima1DInterpolator(atm["alt_ft"], atm["T_degR"])
    P = Akima1DInterpolator(atm["alt_ft"], atm["P_psi"])
    for alt_ft in (0.0, 1234.5, 35000.0, 36123.0, 47777.7, 99000.0):
        Tk, Pa = e.us1976(alt_ft * C.FT)
        assert Tk / C.R2K == pytest.approx(float(T(alt_ft)), rel=1e-12)
        assert Pa / C.PSI == pytest.approx(float(P(alt_ft)), rel=1e-12)
