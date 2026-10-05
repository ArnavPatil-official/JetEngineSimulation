"""P8.6 WebPlotDigitizer importer: CSV/JSON parsing, repeat matching, database entry.

Fixtures reproduce the WPD 4.x/5.x file layouts (automeris-io/WebPlotDigitizer:
plotData.js serialize(), dataExport.js generateCSV(), dataTable.js makeTable()).
The pixel -> data cases are ports of WPD's own tests/xy_axes_tests.js.
"""

import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))

import empirical_db as edb  # noqa: E402
import wpd_import as wpd  # noqa: E402


# ---------------------------------------------------------------- fixture writers
def _cal(px_py_dx_dy):
    """Four calibration points X1, X2, Y1, Y2 as WPD stores them (typed values as strings)."""
    return [{"px": px, "py": py, "dx": str(dx), "dy": str(dy), "dz": None}
            for px, py, dx, dy in px_py_dx_dy]


# Grey & Wilsted-like panel at 150 dpi: x 1.0..2.8 over px 100..910, y 0.60..1.00 over py 1200..300
LIN_CAL = _cal([(100.0, 1200.0, "1.0", "0.60"), (910.0, 1200.0, "2.8", "0.60"),
                (100.0, 1200.0, "1.0", "0.60"), (100.0, 300.0, "1.0", "1.00")])
# log y axis: 1e-3..1e1 (4 decades) over py 800..100; x linear 0..10
LOG_CAL = _cal([(50.0, 800.0, "0", "1e-3"), (750.0, 800.0, "10", "1e-3"),
                (50.0, 800.0, "0", "1e-3"), (50.0, 100.0, "0", "10")])


def _axes(cal, log_x=False, log_y=False, name="XY"):
    ax = {"name": name, "type": "XYAxes", "isLogX": log_x, "isLogY": log_y, "noRotation": False,
          "calibrationPoints": cal}
    return ax


def _data_to_pixel(ax_json, xy):
    """Inverse of the WPD XY map, for building fixtures (as WPD's dataToPixel)."""
    ax = wpd.WPDAxes(name="t", type="XYAxes", is_log_x=ax_json["isLogX"], is_log_y=ax_json["isLogY"],
                     calibration_points=[{k: (float(v) if v is not None else None) for k, v in cp.items()
                                          if k != "dz"} for cp in ax_json["calibrationPoints"]])
    probe = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = wpd.xy_pixel_to_data(ax, probe)
    t = np.array(d, dtype=float)
    if ax.is_log_x:
        t[:, 0] = np.log10(t[:, 0])
    if ax.is_log_y:
        t[:, 1] = np.log10(t[:, 1])
    A = np.column_stack([t[1] - t[0], t[2] - t[0]])
    xy = np.array(xy, dtype=float)
    if ax.is_log_x:
        xy[:, 0] = np.log10(xy[:, 0])
    if ax.is_log_y:
        xy[:, 1] = np.log10(xy[:, 1])
    return (np.linalg.solve(A, (xy - t[0]).T)).T


def write_project(path, datasets, ax_json, version=(4, 2)):
    """WPD project JSON exactly as plotData.serialize() writes it."""
    coll = []
    for name, pts in datasets.items():
        pix = _data_to_pixel(ax_json, pts)
        coll.append({"name": name, "axesName": ax_json["name"], "colorRGB": [200, 0, 0, 255],
                     "metadataKeys": [],
                     "data": [{"x": float(px), "y": float(py), "value": [float(x), float(y)]}
                              for (px, py), (x, y) in zip(pix, pts)],
                     "autoDetectionData": None})
    doc = {"version": list(version), "axesColl": [ax_json], "datasetColl": coll, "measurementColl": []}
    Path(path).write_text(json.dumps(doc))
    return Path(path)


def write_single_csv(path, pts, sep=", ", decimal_comma=False):
    """'View Data' -> 'Download .CSV': no header, x<sep>y per row."""
    rows = []
    for x, y in pts:
        sx, sy = repr(float(x)), repr(float(y))
        if decimal_comma:
            sx, sy = sx.replace(".", ","), sy.replace(".", ",")
        rows.append(f"{sx}{sep}{sy}")
    Path(path).write_text("\n".join(rows) + "\n")
    return Path(path)


def write_wide_csv(path, datasets):
    """'Export all datasets' -> 'Download .CSV' (wpd_datasets.csv)."""
    names = list(datasets)
    n = max(len(v) for v in datasets.values())
    header = ",".join(sum(([nm, ""] for nm in names), []))      # "name1,,name2,"
    lines = [header, ",".join(["X", "Y"] * len(names))]
    for i in range(n):
        cells = []
        for nm in names:
            pts = datasets[nm]
            cells += [repr(float(pts[i][0])), repr(float(pts[i][1]))] if i < len(pts) else ["", ""]
        lines.append(",".join(cells))
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


TRUE_CURVE = np.array([[1.03, 0.901], [1.11, 0.918], [1.11, 0.912], [1.30, 0.934], [1.54, 0.947],
                       [1.77, 0.957], [2.06, 0.969], [2.14, 0.970]])


def noisy_repeats(true, seed=7, sx=0.0015, sy=0.0006):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        r = true + rng.normal(0.0, [sx, sy], size=true.shape)
        out.append(r[rng.permutation(len(r))])       # file order is not assumed
    return out


# ---------------------------------------------------------------- CSV parsing
def test_single_dataset_csv(tmp_path):
    pts = [[1.05, 0.931], [1.2, 0.944], [2.5, 0.973]]
    got = wpd.read_wpd_csv(write_single_csv(tmp_path / "a.csv", pts))
    assert list(got) == [""] and np.allclose(got[""], pts)
    # locale with decimal comma: WPD switches the column separator to "; "
    got = wpd.load_curve_csv(write_single_csv(tmp_path / "b.csv", pts, sep="; ", decimal_comma=True))
    assert np.allclose(got, pts)
    got = wpd.load_curve_csv(write_single_csv(tmp_path / "c.csv", pts, sep="\t"))
    assert np.allclose(got, pts)


def test_single_csv_rejects_extra_columns_and_text(tmp_path):
    (tmp_path / "x.csv").write_text("1.0, 0.9, Bar0\n1.1, 0.91, Bar1\n")
    with pytest.raises(ValueError, match="2 columns"):
        wpd.read_wpd_csv(tmp_path / "x.csv")
    (tmp_path / "y.csv").write_text("1.0, 0.9\n1.1, abc\n")
    with pytest.raises(ValueError, match="not a number"):
        wpd.read_wpd_csv(tmp_path / "y.csv")
    with pytest.raises(FileNotFoundError):
        wpd.read_wpd_csv(tmp_path / "missing.csv")


def test_export_all_datasets_csv(tmp_path):
    sets = {"alpha05": [[1.1, 0.93], [1.5, 0.95], [2.0, 0.97]], "alpha13": [[1.1, 0.89], [2.5, 0.955]]}
    p = write_wide_csv(tmp_path / "wpd_datasets.csv", sets)
    assert p.read_text().splitlines()[:2] == ["alpha05,,alpha13,", "X,Y,X,Y"]
    got = wpd.read_wpd_csv(p)
    assert set(got) == set(sets)
    for k in sets:
        assert np.allclose(got[k], sets[k])
    assert np.allclose(wpd.load_curve_csv(p, "alpha13"), sets["alpha13"])
    with pytest.raises(ValueError, match="several datasets"):
        wpd.load_curve_csv(p)
    with pytest.raises(KeyError):
        wpd.load_curve_csv(p, "alpha30")


def test_export_all_rejects_comma_in_name_and_gaps(tmp_path):
    (tmp_path / "a.csv").write_text("a,b,,c,\nX,Y,X,Y\n1,2,3,4\n")
    with pytest.raises(ValueError, match="comma"):
        wpd.read_wpd_csv(tmp_path / "a.csv")
    (tmp_path / "b.csv").write_text("s1,,s2,\nX,Y,X,Y\n1,2,,\n,,3,4\n")
    with pytest.raises(ValueError, match="out-of-order"):
        wpd.read_wpd_csv(tmp_path / "b.csv")


# ---------------------------------------------------------------- pixel -> data (WPD's own cases)
def _ax(points, log_x=False, log_y=False, no_rot=False):
    cps = [{"px": px, "py": py, "dx": float(dx), "dy": float(dy)} for px, py, dx, dy in points]
    return wpd.WPDAxes("t", "XYAxes", log_x, log_y, no_rot, cps)


@pytest.mark.parametrize("no_rot", [False, True])
def test_pixel_to_data_linear(no_rot):
    ax = _ax([(0, 99, 0, 0), (99, 99, 100, 0), (0, 99, 0, 0), (0, 0, 0, 10)], no_rot=no_rot)
    assert np.allclose(wpd.xy_pixel_to_data(ax, [[99 / 2, 99 / 2]]), [[50, 5]], atol=1e-13)


@pytest.mark.parametrize("no_rot", [False, True])
def test_pixel_to_data_axes_at_90_deg(no_rot):
    ax = _ax([(0, 99, 0, 0), (0, 0, 10, 0), (0, 99, 0, 0), (99, 99, 0, 100)], no_rot=no_rot)
    assert np.allclose(wpd.xy_pixel_to_data(ax, [[99 / 2, 99 / 2]]), [[5, 50]], atol=1e-13)


@pytest.mark.parametrize("sign,no_rot", [(1, False), (1, True), (-1, False)])
def test_pixel_to_data_log(sign, no_rot):
    ax = _ax([(0, 99, sign * 1e-5, 0), (99, 99, sign * 1e12, 0), (0, 99, sign * 1e-5, sign * 1e-20),
              (0, 0, sign * 1e-5, sign * 1.0)], log_x=True, log_y=True, no_rot=no_rot)
    px = [[99 * (6 + 5) / (12 + 5), 99 * (1 - (-3 + 20) / (0 + 20))]]
    d = wpd.xy_pixel_to_data(ax, px)[0]
    assert d[0] == pytest.approx(sign * 1e6, rel=1e-12) and d[1] == pytest.approx(sign * 1e-3, rel=1e-12)


# ---------------------------------------------------------------- project JSON
def test_project_json_linear_axes(tmp_path):
    ax = _axes(LIN_CAL)
    p = write_project(tmp_path / "fig4d_r1.json", {"alpha06_d091": TRUE_CURVE.tolist()}, ax)
    proj = wpd.read_wpd_project(p)
    assert proj.version == [4, 2] and len(proj.axes) == 1 and len(proj.datasets) == 1
    a = proj.axes[0]
    assert (a.type, a.is_log_x, a.is_log_y) == ("XYAxes", False, False)
    assert a.x_ticks == (1.0, 2.8) and a.y_ticks == (0.6, 1.0)
    assert a.span("x") == pytest.approx(1.8) and a.span("y") == pytest.approx(0.4)
    ds = proj.dataset("alpha06_d091")
    assert ds.axes_name == "XY" and np.allclose(ds.values, TRUE_CURVE)
    assert np.allclose(wpd.xy_pixel_to_data(a, ds.pixels), TRUE_CURVE, atol=1e-12)


def test_project_json_log_axis_and_tar(tmp_path):
    ax = _axes(LOG_CAL, log_y=True)
    pts = [[1.0, 2e-3], [4.0, 0.05], [9.0, 3.0]]
    p = write_project(tmp_path / "log.json", {"c": pts}, ax)
    proj = wpd.read_wpd_project(p)
    a = proj.axes[0]
    assert a.is_log_y and not a.is_log_x
    assert a.y_ticks == (1e-3, 10.0) and a.span("y") == pytest.approx(4.0)   # decades
    assert a.calibration_points[3] == {"px": 50.0, "py": 100.0, "dx": 0.0, "dy": 10.0}
    assert np.allclose(wpd.xy_pixel_to_data(a, proj.dataset("c").pixels), pts, rtol=1e-12)
    # "Download Project File (.tar)": <name>/info.json + <name>/wpd.json + image
    tp = tmp_path / "proj.tar"
    with tarfile.open(tp, "w") as tf:
        tf.add(p, arcname="proj/wpd.json")
    assert wpd.read_wpd_project(tp).sha256 == proj.sha256


def test_project_json_rejects_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        wpd.read_wpd_project(tmp_path / "none.json")
    (tmp_path / "v3.json").write_text(json.dumps({"wpd": {"version": [3, 8]}}))
    with pytest.raises(ValueError, match="3.x"):
        wpd.read_wpd_project(tmp_path / "v3.json")
    # a stored value that does not follow from the calibration (hand-edited file)
    p = write_project(tmp_path / "t.json", {"c": TRUE_CURVE.tolist()}, _axes(LIN_CAL))
    doc = json.loads(p.read_text())
    doc["datasetColl"][0]["data"][2]["value"][1] += 0.01
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="calibration"):
        wpd.read_wpd_project(p)


# ---------------------------------------------------------------- repeats
def test_combine_repeats_sigma_and_spread():
    r1 = [[1.0, 0.90], [2.0, 0.95]]
    r2 = [[2.01, 0.952], [1.01, 0.902]]            # file order differs
    r3 = [[0.99, 0.901], [1.99, 0.947]]
    c = wpd.combine_repeats(r1, r2, r3, x_span=1.8, y_span=0.4)
    assert np.allclose(c.x_repeats, [[1.0, 1.01, 0.99], [2.0, 2.01, 1.99]])
    assert np.allclose(c.y_repeats, [[0.90, 0.902, 0.901], [0.95, 0.952, 0.947]])
    assert np.allclose(c.x_mean, [1.0, 2.0]) and np.allclose(c.y_mean, [0.901, 0.9496666666666667])
    assert np.allclose(c.x_sigma, [0.01, 0.01])                     # sample SD, ddof = 1
    assert c.y_sigma[1] == pytest.approx(np.std([0.95, 0.952, 0.947], ddof=1))
    assert np.allclose(c.y_spread, [0.002, 0.005]) and np.allclose(c.x_spread, [0.02, 0.02])


def test_combine_repeats_keeps_points_that_share_an_x():
    reps = noisy_repeats(TRUE_CURVE)
    c = wpd.combine_repeats(*reps, x_span=1.8, y_span=0.4)
    # the two symbols at x = 1.11 stay paired with themselves in every repeat
    same_x = np.isclose(c.x_mean, 1.11, atol=0.01)
    assert same_x.sum() == 2
    assert np.allclose(np.sort(c.y_mean[same_x]), [0.912, 0.918], atol=0.002)
    assert np.all(c.y_spread < 0.004)


def test_combine_repeats_rejects_mismatches():
    reps = noisy_repeats(TRUE_CURVE)
    with pytest.raises(ValueError, match="point counts"):
        wpd.combine_repeats(reps[0], reps[1], reps[2][:-1], x_span=1.8, y_span=0.4)
    shifted = reps[2].copy()
    shifted[np.argmax(shifted[:, 0]), 0] += 0.05       # 2.8 % of the 1.8 span
    with pytest.raises(ValueError, match="misaligned"):
        wpd.combine_repeats(reps[0], reps[1], shifted, x_span=1.8, y_span=0.4)
    with pytest.raises(ValueError, match="identical"):
        wpd.combine_repeats(reps[0], reps[1], reps[0][::-1], x_span=1.8, y_span=0.4)
    # a different point marked in one repeat (same count, far away)
    other = reps[1].copy()
    other[0] = [2.6, 0.62]
    with pytest.raises(ValueError, match="misaligned"):
        wpd.combine_repeats(reps[0], other, reps[2], x_span=1.8, y_span=0.4)


def test_combine_repeats_log_axis_tolerance_in_decades():
    true = np.array([[1.0, 2e-3], [5.0, 0.05], [9.0, 3.0]])
    r = [true * [1, f] for f in (1.0, 1.02, 0.98)]        # 1-2 % in value, < 0.01 decade
    c = wpd.combine_repeats(*r, x_span=10.0, y_span=4.0, y_log=True)
    assert np.allclose(c.y_mean, true[:, 1] * (1.0 + 1.02 + 0.98) / 3)
    bad = true * [1, 1.2]                                 # 0.08 decade > 1 % of 4 decades
    with pytest.raises(ValueError, match="misaligned"):
        wpd.combine_repeats(true, r[1], bad, x_span=10.0, y_span=4.0, y_log=True)


# ---------------------------------------------------------------- database entry
@pytest.fixture
def db(tmp_path):
    c = edb.connect(tmp_path / "t.sqlite", create=True)
    edb.add_source(c, "NACA-TN-1757", "Grey & Wilsted (1948), test fixture", "nasa_report", "2026-09-29")
    edb.add_experiment(c, "GW-a06-d091", "NACA-TN-1757", "NACA-Lewis-conical-nozzles-1947",
                       "conical nozzle 6 deg, D2/D1 0.91", "nozzle", "cold_air", "experiment", 4, "B",
                       geometry={"cone_half_angle_deg": 6, "D2_over_D1": 0.91})
    yield c
    c.close()


def _write_curve(tmp_path, curve, fig, reps, ax_json, wide=False):
    d = tmp_path / "digitised" / "NACA-TN-1757" / fig
    d.mkdir(parents=True, exist_ok=True)
    csvs, jsons = [], []
    for k, r in enumerate(reps, 1):
        if wide:
            csvs.append(write_wide_csv(d / f"{curve}_r{k}.csv", {curve: r, "other": [[1.5, 0.8]]}))
        else:
            csvs.append(write_single_csv(d / f"{curve}_r{k}.csv", r))
        jsons.append(write_project(d / f"{fig}_r{k}.json", {curve: r, "other": [[1.5, 0.8]]}, ax_json))
    return csvs, jsons


def test_enter_curve_full_insert_and_qa(tmp_path, db):
    reps = noisy_repeats(TRUE_CURVE)
    csvs, jsons = _write_curve(tmp_path, "alpha06_d091", "fig4d", reps, _axes(LIN_CAL))
    ops = wpd.enter_digitised_curve(db, edb, "GW-a06-d091-fig4d", "GW-a06-d091", "NPR", "1", "Cd", "1",
                                    csvs, jsons, "NACA TN-1757 Figure 4(d)", "Figure 4(d) (PDF p. 23)")
    db.commit()
    assert ops == [f"GW-a06-d091-fig4d-p{i:02d}" for i in range(1, 9)]
    assert [i for i in edb.run_qa(db) if i.level == "error"] == []
    rows = db.execute("SELECT o.*, d.figure, d.tool, d.axis_calibration_json, d.repeats_json, d.spread_si "
                      "FROM observation o JOIN digitisation d USING(obs_id) ORDER BY o.op_id, o.quantity"
                      ).fetchall()
    assert len(rows) == 16 and {r["quantity"] for r in rows} == {"NPR", "Cd"}
    assert all(r["sigma_kind"] == "digitisation" for r in rows)
    comb = wpd.combine_repeats(*reps, x_span=1.8, y_span=0.4)
    for i, op in enumerate(ops):
        by_q = {r["quantity"]: r for r in rows if r["op_id"] == op}
        for q, col, mean, sd, rep in (("NPR", 0, comb.x_mean, comb.x_sigma, comb.x_repeats),
                                      ("Cd", 1, comb.y_mean, comb.y_sigma, comb.y_repeats)):
            r = by_q[q]
            assert r["role"] == ("input" if q == "NPR" else "output")
            assert r["value_si"] == pytest.approx(mean[i], abs=1e-15)
            assert r["sigma_si"] == pytest.approx(sd[i], abs=1e-15)
            assert json.loads(r["repeats_json"]) == pytest.approx(list(rep[i]), abs=0)
            assert r["spread_si"] == pytest.approx(max(rep[i]) - min(rep[i]), abs=1e-15)
            assert r["value_si"] == pytest.approx(np.mean(json.loads(r["repeats_json"])), abs=1e-15)
            assert r["location"] == "Figure 4(d) (PDF p. 23)"
            assert r["figure"] == "NACA TN-1757 Figure 4(d)" and r["tool"].startswith("WebPlotDigitizer")
            cal = json.loads(r["axis_calibration_json"])
            assert [c["repeat"] for c in cal["calibrations"]] == [1, 2, 3]
            assert all(c["axes"]["calibrationPoints"][1]["dx"] == 2.8 for c in cal["calibrations"])
            assert all(c["dataset"] == "alpha06_d091" for c in cal["calibrations"])
            assert [c["file"] for c in cal["repeat_csv"]] == [p.name for p in csvs]
            assert cal["matching"]["x_span"] == pytest.approx(1.8)
    assert db.execute("SELECT COUNT(*) FROM operating_point").fetchone()[0] == 8
    assert db.execute("SELECT COUNT(*) FROM digitisation").fetchone()[0] == 16


def test_enter_curve_wide_csv_single_project_and_fixed_input(tmp_path, db):
    edb.add_source(db, "NASA-TN-D-6967", "Kofskey & Nusbaum (1972), test fixture", "nasa_report",
                   "2026-09-29")
    edb.add_experiment(db, "TND6967-2stage", "NASA-TN-D-6967", "NASA-Lewis-TND6967-turbine",
                       "two-stage turbine", "turbine", "cold_air", "experiment", 4, "B")
    true = np.array([[2.4, 1.98], [3.0, 2.01], [3.6, 2.02], [4.4, 2.021]])
    reps = noisy_repeats(true, seed=3, sx=0.004, sy=0.0002)
    cal = _cal([(80.0, 700.0, "2.0", "1.94"), (900.0, 700.0, "5.2", "1.94"),
                (80.0, 700.0, "2.0", "1.94"), (80.0, 200.0, "2.0", "2.06")])
    csvs, jsons = _write_curve(tmp_path, "speed100", "fig16", reps, _axes(cal), wide=True)
    ops = wpd.enter_digitised_curve(
        db, edb, "TND6967-fig16-N100", "TND6967-2stage", "PR", "1", "mdot_corr", "kg/s", csvs,
        jsons[0], "NASA TN D-6967 Figure 16", "Figure 16 (PDF p. 23)", dataset="speed100",
        fixed_observations=[{"quantity": "N_corr", "value": 15336.0, "unit": "rpm", "role": "input",
                             "location": "Figure 16 legend (100 %) x Table I (15 336 rpm)"}])
    assert len(ops) == 4
    assert [i for i in edb.run_qa(db) if i.level == "error"] == []
    n = db.execute("SELECT COUNT(*) FROM observation WHERE quantity='N_corr'").fetchone()[0]
    assert n == 4
    assert db.execute("SELECT COUNT(*) FROM digitisation d JOIN observation o USING(obs_id) "
                      "WHERE o.quantity='N_corr'").fetchone()[0] == 0
    cal_rec = json.loads(db.execute("SELECT axis_calibration_json FROM digitisation LIMIT 1").fetchone()[0])
    assert len(cal_rec["calibrations"]) == 1 and cal_rec["calibrations"][0]["repeat"] is None


def test_enter_curve_refuses_bad_inputs(tmp_path, db):
    reps = noisy_repeats(TRUE_CURVE)
    csvs, jsons = _write_curve(tmp_path, "alpha06_d091", "fig4d", reps, _axes(LIN_CAL))
    args = (db, edb, "op", "GW-a06-d091", "NPR", "1", "Cd", "1")
    with pytest.raises(FileNotFoundError):
        wpd.enter_digitised_curve(*args, [csvs[0], csvs[1], tmp_path / "gone.csv"], jsons, "F", "L")
    with pytest.raises(FileNotFoundError):
        wpd.enter_digitised_curve(*args, csvs, [jsons[0], jsons[1], tmp_path / "gone.json"], "F", "L")
    with pytest.raises(ValueError, match="three"):
        wpd.enter_digitised_curve(*args, csvs[:2], jsons, "F", "L")
    # CSV of repeat 2 paired with the project of repeat 1: not the same digitisation
    with pytest.raises(ValueError, match="not the same repeat"):
        wpd.enter_digitised_curve(*args, [csvs[1], csvs[1], csvs[2]], jsons, "F", "L",
                                  dataset="alpha06_d091")
    # unknown quantity is refused by the database layer
    with pytest.raises(ValueError, match="vocabulary"):
        wpd.enter_digitised_curve(db, edb, "op2", "GW-a06-d091", "NPR", "1", "Cv_e", "1",
                                  csvs, jsons, "F", "L")


def test_check_command_on_documented_layout(tmp_path, capsys):
    reps = noisy_repeats(TRUE_CURVE)
    csvs, _ = _write_curve(tmp_path, "alpha06_d091", "fig4d", reps, _axes(LIN_CAL))
    fig_dir = csvs[0].parent
    assert wpd.main(["check", str(fig_dir), "alpha06_d091"]) == 0
    assert "OK: alpha06_d091, 8 points" in capsys.readouterr().out
    (fig_dir / "alpha06_d091_r3.csv").unlink()
    assert wpd.main(["check", str(fig_dir), "alpha06_d091"]) == 1
    assert "NOT OK" in capsys.readouterr().out
