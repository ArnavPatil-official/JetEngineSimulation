"""
P5.2 Step 1 — identifiability of the LTO calibration (docs/plan.md, F2).

On the frozen v4 record the profile must find φ_to identified and the other six
parameters not, with the k_mdot / φ_idle / φ_app ridge measured, not asserted.
The closed-form fuel-flow objective the profile uses is checked against the
real Cantera cycle, and the import-safe ``calibrate_lto`` refactor is checked
to reproduce the v4 objective and to refuse overwriting a frozen record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import calibrate_lto as cal  # noqa: E402
import identifiability_profile as ip  # noqa: E402

V4 = ROOT / "outputs" / "calibration_trent1000_ae3_v4.json"


@pytest.fixture(scope="module")
def c4():
    return ip.load_calibration(V4)


@pytest.fixture(scope="module")
def engine():
    return ip.new_engine()


@pytest.fixture(scope="module")
def v4_profile():
    return ip.run(V4, cycle_checks=False)


@pytest.fixture(scope="module")
def cycle(c4, engine):
    return ip.CycleRunner(c4, engine)


# ---- verdict on v4 (F2) ------------------------------------------------------

def test_v4_identifies_phi_to_alone(v4_profile):
    assert v4_profile["identified"] == ["phi_to"]
    assert set(v4_profile["not_identified"]) == {
        "eta_combustor", "pressure_loss", "k_pi", "k_mdot", "phi_idle", "phi_app"}


def test_v4_inert_parameters_have_flat_profiles(v4_profile):
    prof = {p["parameter"]: p for p in v4_profile["profiles"]}
    for k in ("eta_combustor", "pressure_loss", "k_pi"):
        assert prof[k]["edge_rise_lower"] == 0.0 and prof[k]["edge_rise_upper"] == 0.0
        assert prof[k]["interval_width_frac"] == 1.0


def test_v4_ridge_is_a_flat_valley_not_a_box_artifact(v4_profile):
    prof = {p["parameter"]: p for p in v4_profile["profiles"]}
    # k_mdot's profile rises at both box edges (the phi boxes stop the ridge) — only the
    # flat-valley width check exposes it; a rule without that check would call it identified
    k = prof["k_mdot"]
    assert k["edge_rise_lower"] >= ip.MARGIN and k["edge_rise_upper"] >= ip.MARGIN
    assert not k["touches_lower"] and not k["touches_upper"]
    assert k["interval_width_frac"] >= ip.WIDTH_FRAC
    assert "flat valley" in k["why_not"]
    for q in ("phi_idle", "phi_app"):
        assert prof[q]["interval_width_frac"] >= ip.WIDTH_FRAC


def test_v4_phi_to_interval_is_narrow_and_interior(v4_profile):
    p = next(p for p in v4_profile["profiles"] if p["parameter"] == "phi_to")
    lo, hi = p["bounds"]
    assert lo < p["interval"][0] < p["interval"][1] < hi
    assert p["interval_width_frac"] < ip.WIDTH_FRAC
    # identifiable is not "the sampler found the optimum": v4's 0.5409 lies outside the interval
    assert not (p["interval"][0] <= p["calibrated"] <= p["interval"][1])


def test_v4_ridge_measured(v4_profile):
    rc = v4_profile["ridge_check"]
    assert rc["ridge_present"]
    assert rc["objective_spread"] < 1e-12
    objs = {round(r["k_mdot"], 2): r["objective_closed_form"] for r in rc["rows"]}
    assert objs[0.58] == pytest.approx(0.016190398, abs=1e-9)   # F2: 1.619 % at every point


def test_non_fuel_objective_is_refused(tmp_path):
    rec = json.loads(V4.read_text())
    rec["objective"] = {"name": "fuel_flow_and_thrust"}
    p = tmp_path / "calibration_trent1000_ae3_vX.json"
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="No closed-form profile"):
        ip.run(p, cycle_checks=False)


# ---- closed form vs real cycle ----------------------------------------------

def test_closed_form_matches_cycle_at_v4_and_on_the_ridge(c4, engine, cycle):
    model = ip.FuelFlowProfile(c4, engine)
    rows = [ip.cross_check(cycle, model, p, "ridge") for p in ip.ridge_points(c4)]
    assert all(r["agrees"] for r in rows), rows
    objs = [r["objective_cycle"] for r in rows]
    assert max(objs) - min(objs) < 1e-12
    idle = [r["thrust_Idle_kN"] for r in rows]
    assert (max(idle) - min(idle)) / min(idle) > 0.05   # F2: same objective, different engine


def test_refactored_calibration_reproduces_v4_objective(c4, engine):
    rec = json.loads(V4.read_text())
    got = cal.objective_value(engine, rec["best_params"], c4["beta"])
    assert got == pytest.approx(rec["best_mean_abs_pct_error"], rel=1e-12)


def test_calibration_refuses_to_overwrite_a_frozen_record(monkeypatch):
    before = V4.read_bytes()
    monkeypatch.setattr(sys, "argv", ["calibrate_lto.py", "--tag", "v4", "--n-trials", "1"])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cal.main()
    assert V4.read_bytes() == before
