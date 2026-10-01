"""P8.5 reactor-network numerical contracts (no network study here)."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(os.environ.get("CATJET_BUILD", ROOT / "cpp" / "build"))))
core = pytest.importorskip("catjet_core")
if not hasattr(core, "ReactorNetwork"):
    pytest.skip("catjet_core built without the P8.5 network", allow_module_level=True)


@pytest.mark.parametrize("K", [7, 9])
def test_gauss_hermite_matches_numpy(K):
    x, w = core.gauss_hermite(K)
    xn, wn = np.polynomial.hermite.hermgauss(K)
    assert np.allclose(x, xn, rtol=0, atol=1e-14)
    assert np.allclose(w, wn / math.sqrt(math.pi), rtol=1e-13, atol=1e-18)
    assert math.isclose(sum(w), 1.0, rel_tol=1e-14)
    # phi_k = phi + sqrt(2) sigma x_k has mean phi and standard deviation sigma
    assert math.isclose(sum(wi * 2 * xi * xi for xi, wi in zip(x, w)), 1.0, rel_tol=1e-13)


def test_lhv_basis_is_standard():
    net = core.ReactorNetwork(str(ROOT / "data" / "A2NOx.yaml"), "POSF10325", 1)
    assert 49.5e6 < net.lhv_mass("CH4") < 50.5e6   # 50.0 MJ/kg
    assert 119e6 < net.lhv_mass("H2") < 121e6      # 120.0 MJ/kg
    assert 10.0e6 < net.lhv_mass("CO") < 10.2e6    # 10.1 MJ/kg
    assert 42.5e6 < net.lhv_mass("POSF10325") < 44.5e6
