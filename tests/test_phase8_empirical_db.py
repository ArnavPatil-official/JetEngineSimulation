"""P8.6 empirical database: vocabulary, SI conversion and QA checks (plan P8.6)."""

import math
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))

import empirical_db as edb  # noqa: E402

VOCAB = edb.load_vocabulary()


@pytest.fixture
def conn(tmp_path):
    c = edb.connect(tmp_path / "t.sqlite", create=True)
    edb.add_source(c, "S1", "Test report 1", "nasa_report", "2026-09-29")
    edb.add_source(c, "S2", "Test paper reusing rig 1", "journal", "2026-09-29")
    edb.add_experiment(c, "E1", "S1", "rigA", "Turbine rig A", "turbine", "cold_air",
                       "experiment", 4, "B", facility="Lab X")
    edb.add_operating_point(c, "E1-r1", "E1", "run 1")
    yield c
    c.close()


def test_si_round_trip_every_unit():
    for q, spec in VOCAB["quantities"].items():
        for unit in VOCAB["units"][spec["dim"]]:
            for v in (0.37, 1.0, 523.7, 1.9e4):
                si = edb.to_si(VOCAB, q, v, unit)
                assert math.isclose(edb.from_si(VOCAB, q, si, unit), v, rel_tol=1e-12, abs_tol=1e-12)


def test_known_conversions():
    assert edb.to_si(VOCAB, "Tt_in", 518.67, "degR") == pytest.approx(288.15, rel=1e-12)
    assert edb.to_si(VOCAB, "Tt_in", 59.0, "degF") == pytest.approx(288.15, rel=1e-12)
    assert edb.to_si(VOCAB, "Pt_in", 14.695948775513449, "psia") == pytest.approx(101325.0, rel=1e-12)
    assert edb.to_si(VOCAB, "mdot", 1.0, "lbm/s") == 0.45359237
    assert edb.to_si(VOCAB, "N", 60.0, "rpm") == pytest.approx(2 * math.pi, rel=1e-15)
    assert edb.to_si(VOCAB, "EI_NOx", 25.0, "g/kg") == pytest.approx(0.025, rel=1e-15)
    # an uncertainty scales without the offset
    assert edb.sigma_to_si(VOCAB, "Tt_in", 1.8, "degF") == pytest.approx(1.0, rel=1e-12)


def test_rejects_unknown_quantity_unit_and_nonfinite(conn):
    with pytest.raises(ValueError):
        edb.add_observation(conn, "E1-r1", "efficiency", 0.9, "1", "output", "T1")
    with pytest.raises(ValueError):
        edb.add_observation(conn, "E1-r1", "Tt_in", 500.0, "Kelvin", "input", "T1")
    with pytest.raises(ValueError):
        edb.add_observation(conn, "E1-r1", "Tt_in", float("nan"), "K", "input", "T1")
    with pytest.raises(ValueError):
        edb.add_observation(conn, "E1-r1", "Tt_in", 500.0, "K", "input", "T1", sigma=1.0)


def test_round_trip_qa_passes_on_stored_rows(conn):
    edb.add_observation(conn, "E1-r1", "Tt_in", 1000.0, "degR", "input", "Table 3")
    edb.add_observation(conn, "E1-r1", "Pt_in", 45.0, "psia", "input", "Table 3", 0.1, "reported")
    assert edb.qa_round_trip(conn) == []


def test_corrected_flow_and_speed_recomputed(conn):
    T, P, m, N = 700.0, 3.0e5, 10.0, 1000.0
    ref = VOCAB["reference_state"]
    mc = m * math.sqrt(T / ref["T_ref_K"]) / (P / ref["p_ref_Pa"])
    for q, v, u in (("Tt_in", T, "K"), ("Pt_in", P, "Pa"), ("mdot", m, "kg/s"),
                    ("mdot_corr", mc, "kg/s"), ("N", N, "rad/s"),
                    ("N_corr", N / math.sqrt(T / ref["T_ref_K"]), "rad/s")):
        edb.add_observation(conn, "E1-r1", q, v, u, "input", "T1")
    assert edb.qa_corrected(conn) == []
    conn.execute("UPDATE observation SET value_si = value_si * 1.02 WHERE quantity = 'mdot_corr'")
    kinds = [i.kind for i in edb.qa_corrected(conn)]
    assert kinds == ["corrected_flow"]


def test_plausibility_catches_efficiency_above_one_and_turbine_heating(conn):
    edb.add_observation(conn, "E1-r1", "eta_tt", 1.02, "1", "output", "T1")
    edb.add_observation(conn, "E1-r1", "Tt_in", 500.0, "K", "input", "T1")
    edb.add_observation(conn, "E1-r1", "Tt_out", 520.0, "K", "output", "T1")
    kinds = {i.kind for i in edb.qa_plausibility(conn) if i.level == "error"}
    assert kinds == {"range", "turbine_temperature"}


def test_energy_balance_cold_air(conn):
    import cantera as ct
    air = ct.Solution("air.yaml")
    air.TPX = 420.0, 2.0e5, "O2:0.21, N2:0.78, AR:0.01"
    h1 = air.enthalpy_mass
    air.TPX = 330.0, 2.0e5, "O2:0.21, N2:0.78, AR:0.01"
    w = h1 - air.enthalpy_mass
    edb.add_observation(conn, "E1-r1", "Tt_in", 420.0, "K", "input", "T1")
    edb.add_observation(conn, "E1-r1", "Tt_out", 330.0, "K", "output", "T1")
    edb.add_observation(conn, "E1-r1", "Pt_in", 2.0e5, "Pa", "input", "T1")
    edb.add_observation(conn, "E1-r1", "work_specific", w, "J/kg", "output", "T1")
    assert edb.qa_energy_balance(conn) == []
    conn.execute("UPDATE observation SET value_si = value_si * 1.1 WHERE quantity = 'work_specific'")
    assert [i.kind for i in edb.qa_energy_balance(conn)] == ["energy_balance"]


def test_one_rig_in_two_papers_counts_once(conn):
    edb.add_experiment(conn, "E2", "S2", "rigA", "Turbine rig A", "turbine", "cold_air",
                       "experiment", 4, "C", facility="Lab X")
    assert edb.independent_sources(conn, "turbine") == 1
    assert edb.qa_duplicates(conn) == []
    # same article and facility entered under a different key is flagged
    edb.add_experiment(conn, "E3", "S2", "rigA-bis", "Turbine rig A", "turbine", "cold_air",
                       "experiment", 4, "C", facility="Lab X")
    assert edb.independent_sources(conn, "turbine") == 2
    assert [i.kind for i in edb.qa_duplicates(conn)] == ["same_article_different_key"]


def test_duplicate_operating_point_flagged(conn):
    edb.add_operating_point(conn, "E1-r2", "E1", "run 2")
    for op in ("E1-r1", "E1-r2"):
        edb.add_observation(conn, op, "Tt_in", 500.0, "K", "input", f"T1 {op}")
        edb.add_observation(conn, op, "Pt_in", 2.0e5, "Pa", "input", f"T1 {op}")
    assert [i.kind for i in edb.qa_duplicates(conn)] == ["duplicate_operating_point"]


def test_validation_split_is_tier4_only(conn):
    edb.add_experiment(conn, "C1", "S1", "cfdA", "Nozzle CFD", "nozzle", "cold_air", "cfd", 3, "B")
    conn.execute("INSERT INTO split VALUES ('C1', 'calibration', 'h')")      # CFD may calibrate
    conn.execute("INSERT INTO split VALUES ('E1', 'locked_test', 'h')")      # tier 4 may validate
    edb.add_experiment(conn, "C2", "S1", "cfdB", "Nozzle CFD 2", "nozzle", "cold_air", "cfd", 3, "B")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO split VALUES ('C2', 'validation', 'h')")
    with pytest.raises(sqlite3.IntegrityError):   # an experiment is always tier 4
        edb.add_experiment(conn, "E9", "S1", "k9", "rig 9", "turbine", "cold_air", "experiment", 3, "B")


def test_export_is_deterministic(conn, tmp_path):
    edb.add_observation(conn, "E1-r1", "Tt_in", 500.0, "K", "input", "T1")
    a = edb.export_csv(conn, tmp_path / "a")
    b = edb.export_csv(conn, tmp_path / "b")
    assert a == b and set(a) == {f"{t}.csv" for t in edb.TABLES}


def test_database_refuses_changed_vocabulary(tmp_path, monkeypatch):
    edb.connect(tmp_path / "t.sqlite", create=True).close()
    changed = tmp_path / "vocab.yaml"
    changed.write_text(edb.VOCAB.read_text() + "\n# edit\n")
    monkeypatch.setattr(edb, "VOCAB", changed)
    with pytest.raises(RuntimeError):
        edb.connect(tmp_path / "t.sqlite")
