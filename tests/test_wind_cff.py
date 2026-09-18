"""
Tests for simulation/nozzle/wind_cff.py — the NPARC/WIND common-file reader
adopted by the P4.1 Sajben data audit (route a).

The checks pin the decoded solution to facts stated independently in the
archive: the NPARC page (81 x 51 grid, M = 0.46 inflow, exit static pressure
16.055 psi), the WIND listing (reference conditions), and the experimental
data file (inlet P/P0 ~ 0.864, exit P/P0 ~ 0.82).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.nozzle.wind_cff import load_wind_solution, read_adf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NASA = REPO / "data" / "raw" / "cfd_datasets" / "nasa" / "transdif01"
CGD, CFL = NASA / "sajben.cgd", NASA / "sajben.cfl"

pytestmark = pytest.mark.skipif(
    not (CGD.exists() and CFL.exists()), reason="NASA archive files not present"
)


@pytest.fixture(scope="module")
def sol():
    return load_wind_solution(CGD, CFL)


def test_adf_tree_has_expected_nodes():
    tree = read_adf(CFL)
    for key in ("rho", "rho*u", "rho*v", "rho*e0", "mul", "mut"):
        assert f"/ADF MotherNode/ZONE   1/{key}" in tree
        assert tree[f"/ADF MotherNode/ZONE   1/{key}"].size == 81 * 51


def test_grid_and_shape(sol):
    assert (sol.ni, sol.nj) == (81, 51)
    assert sol.x.shape == sol.p.shape == (51, 81)
    # x runs along i (axial), y along j (wall-normal); lower wall is y = 0
    assert np.all(np.diff(sol.x[0]) > 0)
    assert np.allclose(sol.y[0], 0.0)
    assert sol.h_throat == pytest.approx(0.14435 * 0.3048, rel=2e-3)


def test_reference_conditions_match_listing(sol):
    ref = sol.reference
    assert ref["M_ref"] == pytest.approx(0.46, abs=1e-3)
    assert ref["p_ref"] == pytest.approx(16.937 * 6894.757, rel=1e-3)
    assert ref["T_ref"] == pytest.approx(504.26 * 5 / 9, rel=1e-3)
    assert ref["p0"] == pytest.approx(19.581 * 6894.757, rel=1e-3)
    assert ref["gamma"] == pytest.approx(1.4)


def test_flow_state_matches_experiment_endpoints(sol):
    jm = sol.nj // 2
    assert sol.p[jm, 0] / sol.p0 == pytest.approx(0.864, abs=0.01)     # first tap
    assert sol.mach[jm, 0] == pytest.approx(0.46, abs=0.01)
    assert sol.p[jm, -1] / sol.p0 == pytest.approx(16.055 / 19.581, abs=0.01)  # exit BC
    assert 1.2 < sol.mach.max() < 1.35                                   # "just under 1.3"


def test_walls_are_no_slip_and_turbulent(sol):
    assert np.abs(sol.u[0]).max() == 0.0 and np.abs(sol.u[-1]).max() == 0.0
    assert np.abs(sol.v[0]).max() == 0.0 and np.abs(sol.v[-1]).max() == 0.0
    assert (sol.mu_t / sol.mu_l).max() > 100.0      # eddy viscosity present
    assert np.all(sol.rho > 0) and np.all(sol.T > 0) and np.all(sol.p > 0)


def test_provenance_recorded(sol):
    assert len(sol.provenance["cfl_sha256"]) == 64
    assert "1997" in sol.provenance["cfl_modified"]
    assert "Weak shock" in sol.provenance["title"]
