#!/usr/bin/env python3
"""
P8.6: import WebPlotDigitizer (WPD) digitisations into the empirical database.

A plotted-only curve is digitised three times, independently (vocabulary v2:
sigma kind "digitisation"). Each repeat gives one CSV export and one project
JSON. This module reads both, matches the three repeats point by point, and
enters one operating point per matched point with an x observation (role
input) and a y observation (role output), each with a ``digitisation`` row
holding the axis calibration of every repeat and the three repeat values.
Everything goes through ``empirical_db`` (vocabulary, unit and SI checks).
Nothing is invented: a missing file, an unreadable file, a repeat with a
different number of points or a point that does not line up raises.

User guide: data/empirical/DIGITISING.md. Check one curve before import:
    .venv/bin/python scripts/phase8/wpd_import.py check data/empirical/digitised/<source>/<figure> <curve>

WPD formats read (checked against automeris-io/WebPlotDigitizer master,
javascript/core/plotData.js, services/dataExport.js, widgets/dataTable.js;
WPD 4.x and 5.x both write project format ``"version": [4, x]``):

* CSV, single dataset ("View Data" -> "Download .CSV"): no header, one row per
  point, ``x<sep>y``; default separator ", " (";" when the browser locale uses a
  decimal comma, in which case numbers are written with ","). Row order is the
  sort order chosen in the dialog.
* CSV, all datasets ("File > Export > Export all datasets" -> "Download .CSV",
  ``wpd_datasets.csv``): row 1 = dataset names, each followed by an empty cell
  (``name1,,name2,``); row 2 = axis labels (``X,Y,X,Y`` for XY axes); then one
  row per point index, blank cells where a dataset has fewer points. Always
  comma separated; names are not quoted, so a name containing a comma is
  rejected.
* Project JSON ("File > Save Project" -> "Download JSON", or ``wpd.json`` inside
  the "Download Project File (.tar)"). Fields relied on:
  ``version`` ([4, n]); ``axesColl[]``: ``name``, ``type`` ("XYAxes"),
  ``isLogX``, ``isLogY``, ``noRotation`` (may be absent), ``calibrationPoints[]``
  = {``px``, ``py``, ``dx``, ``dy``, ``dz``} in the order X1, X2, Y1, Y2 (pixel
  position and the typed tick value; ``dx``/``dy`` are usually strings; X1/X2
  carry the x values, Y1/Y2 the y values); ``datasetColl[]``: ``name``,
  ``axesName``, ``data[]`` = {``x``, ``y`` (pixels), ``value`` ([x, y] data
  coordinates, written when the dataset has axes)}. Older 3.x projects
  (``{"wpd": {"version": [3, ...]}}``) are rejected. The pixel -> data mapping
  of XYAxes is re-implemented (``xy_pixel_to_data``) and checked against the
  stored ``value`` of every point.

Matching (``combine_repeats``): repeats 2 and 3 are paired with repeat 1 by the
assignment that minimises the total distance in axis-normalised coordinates
(Hungarian algorithm), then points are ordered by mean x. Plain sorting by x
mis-pairs points that share an x value (several Grey & Wilsted panels have
two symbols at the same pressure ratio). Every matched point must agree within
``tol_frac`` (default 1 %) of the axis span in x AND in y, where the span is
the calibrated tick range (log axes: in decades) or, without a calibration,
the data range. sigma = sample SD (ddof = 1) of the three repeat values; the
spread (max - min) is written by ``empirical_db.add_digitisation``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

TOL_FRAC = 0.01       # matched repeats must agree within 1 % of the axis span (x and y)
CHECK_FRAC = 1e-4     # CSV vs project-JSON values of the same repeat (formatting only)


# ---------------------------------------------------------------- CSV
def _to_float(cell: str, decimal_comma: bool, where: str) -> float:
    s = cell.strip()
    if decimal_comma:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{where}: {cell!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{where}: non-finite value {cell!r}")
    return v


def read_wpd_csv(path) -> dict[str, np.ndarray]:
    """Read a WPD CSV export. Returns {dataset name: (n, 2) array of x, y} in file
    order. A single-dataset export has no names: its key is ``""``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WPD CSV not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty file")
    # "Export all datasets" layout: row 2 holds only X/Y axis labels
    if len(lines) >= 2 and all(c.strip() in ("X", "Y") for c in lines[1].split(",")) \
            and lines[1].strip():
        return _read_wide(path, lines)
    return {"": _read_single(path, lines)}


def _read_single(path: Path, lines: list[str]) -> np.ndarray:
    first = lines[0]
    if ";" in first:
        sep, dec_comma = ";", True
    elif "\t" in first:
        sep, dec_comma = "\t", False
    else:
        sep, dec_comma = ",", False
    rows = []
    for i, ln in enumerate(lines, 1):
        cells = [c for c in ln.split(sep)]
        if len(cells) != 2:
            raise ValueError(f"{path} line {i}: expected 2 columns (x{sep}y), got {len(cells)}; "
                             "export an XY dataset without point groups/metadata")
        if dec_comma and any("." in c for c in cells):
            raise ValueError(f"{path} line {i}: ';'-separated file with '.' in a number")
        rows.append([_to_float(c, dec_comma, f"{path} line {i}") for c in cells])
    return np.asarray(rows, dtype=float)


def _read_wide(path: Path, lines: list[str]) -> dict[str, np.ndarray]:
    names_row = next(csv.reader([lines[0]]))
    labels = [c.strip() for c in lines[1].split(",")]
    if len(names_row) != len(labels):
        raise ValueError(f"{path}: {len(names_row)} name cells vs {len(labels)} axis labels "
                         "(a dataset name containing a comma?)")
    if len(labels) % 2 or labels != ["X", "Y"] * (len(labels) // 2):
        raise ValueError(f"{path}: axis-label row {labels} is not X,Y pairs (not XY axes?)")
    names = [names_row[2 * k].strip() for k in range(len(labels) // 2)]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{path}: empty or repeated dataset names {names}")
    if any(names_row[2 * k + 1].strip() for k in range(len(names))):
        raise ValueError(f"{path}: unexpected text in a Y header cell of row 1")
    out: dict[str, list] = {n: [] for n in names}
    ended = {n: False for n in names}
    for i, ln in enumerate(lines[2:], 3):
        cells = ln.split(",")
        if len(cells) != len(labels):
            raise ValueError(f"{path} line {i}: {len(cells)} cells, expected {len(labels)}")
        for k, n in enumerate(names):
            cx, cy = cells[2 * k].strip(), cells[2 * k + 1].strip()
            if not cx and not cy:
                ended[n] = True
                continue
            if not cx or not cy or ended[n]:
                raise ValueError(f"{path} line {i}: dataset {n!r} has a half-empty or out-of-order row")
            out[n].append([_to_float(cx, False, f"{path} line {i}"),
                           _to_float(cy, False, f"{path} line {i}")])
    return {n: np.asarray(v, dtype=float).reshape(-1, 2) for n, v in out.items()}


def load_curve_csv(path, dataset: str | None = None) -> np.ndarray:
    """The (n, 2) points of one curve from a WPD CSV (either layout)."""
    sets = read_wpd_csv(path)
    if dataset is not None and "" not in sets:
        if dataset not in sets:
            raise KeyError(f"{path}: no dataset {dataset!r} (has {sorted(sets)})")
        pts = sets[dataset]
    elif len(sets) == 1:
        pts = next(iter(sets.values()))
    else:
        raise ValueError(f"{path}: several datasets {sorted(sets)}; name the one to import")
    if len(pts) == 0:
        raise ValueError(f"{path}: dataset has no points")
    return pts


# ---------------------------------------------------------------- project JSON
@dataclass
class WPDAxes:
    name: str
    type: str
    is_log_x: bool = False
    is_log_y: bool = False
    no_rotation: bool = False
    calibration_points: list[dict] = field(default_factory=list)   # px, py, dx, dy (floats)

    @property
    def x_ticks(self) -> tuple[float, float]:
        return self.calibration_points[0]["dx"], self.calibration_points[1]["dx"]

    @property
    def y_ticks(self) -> tuple[float, float]:
        return self.calibration_points[2]["dy"], self.calibration_points[3]["dy"]

    def span(self, axis: str) -> float:
        a, b = self.x_ticks if axis == "x" else self.y_ticks
        log = self.is_log_x if axis == "x" else self.is_log_y
        return abs(math.log10(abs(b)) - math.log10(abs(a))) if log else abs(b - a)

    def as_record(self) -> dict:
        return {"name": self.name, "type": self.type, "isLogX": self.is_log_x,
                "isLogY": self.is_log_y, "noRotation": self.no_rotation,
                "calibrationPoints": self.calibration_points}


@dataclass
class WPDDataset:
    name: str
    axes_name: str
    pixels: np.ndarray            # (n, 2)
    values: np.ndarray | None     # (n, 2) data coordinates as stored by WPD, or None


@dataclass
class WPDProject:
    path: Path
    sha256: str
    version: list
    axes: list[WPDAxes]
    datasets: list[WPDDataset]

    def dataset(self, name: str) -> WPDDataset:
        hits = [d for d in self.datasets if d.name == name]
        if len(hits) != 1:
            raise KeyError(f"{self.path}: {len(hits)} datasets named {name!r} "
                           f"(has {[d.name for d in self.datasets]})")
        return hits[0]

    def axes_for(self, ds: WPDDataset) -> WPDAxes:
        hits = [a for a in self.axes if a.name == ds.axes_name]
        if len(hits) != 1:
            raise KeyError(f"{self.path}: dataset {ds.name!r} has no unique axes {ds.axes_name!r}")
        return hits[0]


def _num(v, where: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: calibration value {v!r} is not a number (dates are not "
                         "supported)") from None
    if not math.isfinite(x):
        raise ValueError(f"{where}: non-finite calibration value {v!r}")
    return x


def _load_json_bytes(path: Path) -> bytes:
    if path.suffix == ".tar":
        with tarfile.open(path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith("/wpd.json")]
            if len(members) != 1:
                raise ValueError(f"{path}: expected one */wpd.json in the project tar")
            return tf.extractfile(members[0]).read()
    return path.read_bytes()


def read_wpd_project(path) -> WPDProject:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WPD project file not found: {path}")
    raw = _load_json_bytes(path)
    data = json.loads(raw.decode("utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("wpd"), dict):
        raise ValueError(f"{path}: WPD 3.x project; open and re-save it with WPD 4.x or later")
    ver = data.get("version") if isinstance(data, dict) else None
    if not isinstance(ver, list) or not ver or ver[0] != 4:
        raise ValueError(f"{path}: not a WPD project JSON (version {ver!r}, expected [4, n])")
    axes = []
    for k, ax in enumerate(data.get("axesColl") or []):
        where = f"{path} axesColl[{k}]"
        cps = []
        if ax.get("type") == "XYAxes":
            raw_cps = ax.get("calibrationPoints") or []
            if len(raw_cps) != 4:
                raise ValueError(f"{where}: XY axes need 4 calibration points, got {len(raw_cps)}")
            for j, cp in enumerate(raw_cps):
                cps.append({"px": _num(cp["px"], where), "py": _num(cp["py"], where),
                            "dx": _num(cp["dx"], where) if j < 2 else
                            (None if cp.get("dx") in (None, "") else _num(cp["dx"], where)),
                            "dy": _num(cp["dy"], where) if j >= 2 else
                            (None if cp.get("dy") in (None, "") else _num(cp["dy"], where))})
        axes.append(WPDAxes(name=str(ax.get("name")), type=str(ax.get("type")),
                            is_log_x=bool(ax.get("isLogX", False)),
                            is_log_y=bool(ax.get("isLogY", False)),
                            no_rotation=bool(ax.get("noRotation", False)),
                            calibration_points=cps))
    datasets = []
    for k, ds in enumerate(data.get("datasetColl") or []):
        pts = ds.get("data") or []
        pix = np.asarray([[float(p["x"]), float(p["y"])] for p in pts], dtype=float).reshape(-1, 2)
        vals = None
        if pts and all(isinstance(p.get("value"), list) and len(p["value"]) >= 2 for p in pts):
            vals = np.asarray([[float(p["value"][0]), float(p["value"][1])] for p in pts])
        datasets.append(WPDDataset(name=str(ds.get("name")), axes_name=str(ds.get("axesName", "")),
                                   pixels=pix, values=vals))
    proj = WPDProject(path, hashlib.sha256(raw).hexdigest(), ver, axes, datasets)
    # the stored values must follow from the stored calibration
    for ds in datasets:
        if ds.values is None or len(ds.pixels) == 0:
            continue
        ax = proj.axes_for(ds)
        if ax.type != "XYAxes":
            continue
        calc = xy_pixel_to_data(ax, ds.pixels)
        for col, axis in ((0, "x"), (1, "y")):
            log = ax.is_log_x if axis == "x" else ax.is_log_y
            a, b = (calc[:, col], ds.values[:, col])
            if log:
                a, b = np.log10(np.abs(a)), np.log10(np.abs(b))
            if not np.allclose(a, b, rtol=0, atol=1e-6 * max(ax.span(axis), 1e-300)):
                raise ValueError(f"{path}: dataset {ds.name!r} {axis} values do not follow from "
                                 "the stored calibration (edited file?)")
    return proj


def xy_pixel_to_data(ax: WPDAxes, pixels: np.ndarray) -> np.ndarray:
    """WPD XYAxes pixel -> data mapping (javascript/core/axes/xy.js, processCalibration and
    pixelToData): affine map fitted to X1, X2 (x) and Y1, Y2 (y), log10 space on log axes."""
    if ax.type != "XYAxes" or len(ax.calibration_points) != 4:
        raise ValueError(f"axes {ax.name!r}: only calibrated XYAxes are supported")
    c1, c2, c3, c4 = ax.calibration_points
    xmin, xmax = c1["dx"], c2["dx"]
    ymin, ymax = c3["dy"], c4["dy"]
    neg_x = ax.is_log_x and xmin < 0 and xmax < 0
    neg_y = ax.is_log_y and ymin < 0 and ymax < 0
    if ax.is_log_x:
        xmin, xmax = math.log10(abs(xmin)), math.log10(abs(xmax))
    if ax.is_log_y:
        ymin, ymax = math.log10(abs(ymin)), math.log10(abs(ymax))
    x1, y1, x2, y2 = c1["px"], c1["py"], c2["px"], c2["py"]
    x3, y3, x4, y4 = c3["px"], c3["py"], c4["px"], c4["py"]
    dat = np.array([[xmin - xmax, 0.0], [0.0, ymin - ymax]])
    pix = np.array([[x1 - x2, x3 - x4], [y1 - y2, y3 - y4]])
    a = dat @ np.linalg.inv(pix)
    if ax.no_rotation:
        if abs(a[0, 0] * a[1, 1]) > abs(a[0, 1] * a[1, 0]):
            a = np.array([[(xmax - xmin) / (x2 - x1), 0.0], [0.0, (ymax - ymin) / (y4 - y3)]])
        else:
            a = np.array([[0.0, (xmax - xmin) / (y2 - y1)], [(ymax - ymin) / (x4 - x3), 0.0]])
    c = np.array([xmin - a[0, 0] * x1 - a[0, 1] * y1, ymin - a[1, 0] * x3 - a[1, 1] * y3])
    out = np.asarray(pixels, dtype=float) @ a.T + c
    if ax.is_log_x:
        out[:, 0] = -10.0 ** out[:, 0] if neg_x else 10.0 ** out[:, 0]
    if ax.is_log_y:
        out[:, 1] = -10.0 ** out[:, 1] if neg_y else 10.0 ** out[:, 1]
    return out


def _axis_coords(pts: np.ndarray, ax: WPDAxes) -> np.ndarray:
    """Points in the axis' linear coordinates (log10 on a log axis)."""
    out = np.array(pts, dtype=float)
    if ax.is_log_x:
        out[:, 0] = np.log10(np.abs(out[:, 0]))
    if ax.is_log_y:
        out[:, 1] = np.log10(np.abs(out[:, 1]))
    return out


# ---------------------------------------------------------------- repeats
@dataclass
class Combined:
    x_mean: np.ndarray
    y_mean: np.ndarray
    x_sigma: np.ndarray     # sample SD (ddof=1) of the three x repeats
    y_sigma: np.ndarray
    x_spread: np.ndarray    # max - min of the three repeats
    y_spread: np.ndarray
    x_repeats: np.ndarray   # (n, 3): column k = repeat k+1
    y_repeats: np.ndarray


def combine_repeats(r1, r2, r3, *, x_span: float | None = None, y_span: float | None = None,
                    x_log: bool = False, y_log: bool = False,
                    tol_frac: float = TOL_FRAC) -> Combined:
    """Three repeats of one curve ((n, 2) arrays of x, y) -> per-point mean, SD, spread.

    Repeats 2 and 3 are paired with repeat 1 by minimum total axis-normalised
    distance; points are then ordered by mean x. Raises if the point counts
    differ, if two repeats are identical (not independent), or if any matched
    point differs from repeat 1 by more than ``tol_frac`` of the axis span in x
    or in y (span in decades on a log axis)."""
    reps = [np.asarray(r, dtype=float).reshape(-1, 2) for r in (r1, r2, r3)]
    counts = [len(r) for r in reps]
    if len(set(counts)) != 1:
        raise ValueError(f"repeats have different point counts {counts}: re-digitise so that every "
                         "repeat marks the same points")
    if counts[0] == 0:
        raise ValueError("repeats have no points")
    for i in range(3):
        for j in range(i + 1, 3):
            if np.array_equal(reps[i][np.lexsort(reps[i].T[::-1])], reps[j][np.lexsort(reps[j].T[::-1])]):
                raise ValueError(f"repeats {i + 1} and {j + 1} are identical: the three repeats must "
                                 "be independent digitisations")

    def tr(v, log):
        if log:
            if np.any(v <= 0):
                raise ValueError("non-positive value on a log axis")
            return np.log10(v)
        return v

    tx = [tr(r[:, 0], x_log) for r in reps]
    ty = [tr(r[:, 1], y_log) for r in reps]
    if x_span is None:
        x_span = float(np.ptp(np.concatenate(tx)))
    if y_span is None:
        y_span = float(np.ptp(np.concatenate(ty)))
    if not (x_span > 0 and y_span > 0):
        raise ValueError(f"axis span must be positive (x {x_span}, y {y_span})")
    tol_x, tol_y = tol_frac * x_span, tol_frac * y_span

    order1 = np.argsort(tx[0], kind="stable")
    idx = [order1]
    for k in (1, 2):
        cost = np.hypot((tx[0][order1][:, None] - tx[k][None, :]) / x_span,
                        (ty[0][order1][:, None] - ty[k][None, :]) / y_span)
        _, cols = linear_sum_assignment(cost)
        idx.append(cols)
    X = np.column_stack([reps[k][idx[k], 0] for k in range(3)])
    Y = np.column_stack([reps[k][idx[k], 1] for k in range(3)])
    TX = np.column_stack([tx[k][idx[k]] for k in range(3)])
    TY = np.column_stack([ty[k][idx[k]] for k in range(3)])
    for i in range(len(X)):
        for k in (1, 2):
            dx, dy = abs(TX[i, k] - TX[i, 0]), abs(TY[i, k] - TY[i, 0])
            if dx > tol_x or dy > tol_y:
                raise ValueError(
                    f"point near x={X[i, 0]:.6g}, y={Y[i, 0]:.6g} (repeat 1): repeat {k + 1} is at "
                    f"x={X[i, k]:.6g}, y={Y[i, k]:.6g}; |dx|={dx:.3g} (tol {tol_x:.3g}), "
                    f"|dy|={dy:.3g} (tol {tol_y:.3g}) — misaligned or different points; re-digitise")
    xm, ym = X.mean(axis=1), Y.mean(axis=1)
    o = np.argsort(xm, kind="stable")
    X, Y, xm, ym = X[o], Y[o], xm[o], ym[o]
    return Combined(xm, ym, X.std(axis=1, ddof=1), Y.std(axis=1, ddof=1),
                    X.max(axis=1) - X.min(axis=1), Y.max(axis=1) - Y.min(axis=1), X, Y)


# ---------------------------------------------------------------- database entry
def _repeat_label(path: Path) -> str:
    """Curve name from ``<curve>_rN.csv``."""
    return re.sub(r"_r\d+$", "", path.stem)


@dataclass
class LoadedCurve:
    name: str                 # dataset / curve name
    combined: Combined
    calibration_record: dict  # what goes into digitisation.axis_calibration_json
    projects: list[WPDProject]


def load_curve(repeats_paths: Sequence, project_json_path, *, dataset: str | None = None,
               tol_frac: float = TOL_FRAC) -> LoadedCurve:
    """Read and check the three repeats of one curve (no database access).

    ``repeats_paths``: the three WPD CSV exports. ``project_json_path``: one
    project JSON per repeat (sequence of three, recommended) or a single one.
    With three, each CSV is checked against the same-named dataset (``dataset``,
    default: the CSV name without ``_rN``) of its own project. Raises on a
    missing file, a CSV/project mismatch, or repeats that do not line up."""
    paths = [Path(p) for p in repeats_paths]
    if len(paths) != 3:
        raise ValueError(f"three repeat files are required, got {len(paths)}")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"repeat file not found: {p}")
    jpaths = [Path(project_json_path)] if isinstance(project_json_path, (str, Path)) \
        else [Path(p) for p in project_json_path]
    if len(jpaths) not in (1, 3):
        raise ValueError("give one project JSON, or one per repeat (three)")
    projects = [read_wpd_project(p) for p in jpaths]   # raises FileNotFoundError

    reps = [load_curve_csv(p, dataset) for p in paths]
    name = dataset if dataset is not None else _repeat_label(paths[0])

    calib, axes_list = [], []
    for k, proj in enumerate(projects):
        try:
            ds = proj.dataset(name)
        except KeyError:
            if len(projects) == 3:
                raise
            ds = None
        if ds is not None:
            ax = proj.axes_for(ds)
        else:
            xy = [a for a in proj.axes if a.type == "XYAxes"]
            if len(xy) != 1:
                raise ValueError(f"{proj.path}: no dataset {name!r} and {len(xy)} XY axes")
            ax = xy[0]
        if ax.type != "XYAxes":
            raise ValueError(f"{proj.path}: axes {ax.name!r} are {ax.type}, only XYAxes supported")
        axes_list.append(ax)
        if len(projects) == 3:
            pts = reps[k]
            vals = ds.values if ds.values is not None else xy_pixel_to_data(ax, ds.pixels)
            if len(vals) != len(pts):
                raise ValueError(f"{paths[k]} has {len(pts)} points but dataset {name!r} in "
                                 f"{proj.path} has {len(vals)}: CSV and project are not the same repeat")
            a = _axis_coords(pts, ax)
            b = _axis_coords(vals, ax)
            spans = np.array([ax.span("x"), ax.span("y")])
            cost = np.linalg.norm((a[:, None, :] - b[None, :, :]) / spans, axis=2)
            _, cols = linear_sum_assignment(cost)
            for col, axis in ((0, "x"), (1, "y")):
                if not np.allclose(a[:, col], b[cols, col], rtol=0, atol=CHECK_FRAC * spans[col]):
                    raise ValueError(f"{paths[k]} {axis} values differ from dataset {name!r} in "
                                     f"{proj.path}: CSV and project are not the same repeat")
        calib.append({"repeat": k + 1 if len(projects) == 3 else None, "file": proj.path.name,
                      "sha256": proj.sha256, "wpd_project_version": proj.version,
                      "dataset": name if ds is not None else None, "axes": ax.as_record()})

    ax0 = axes_list[0]
    for ax in axes_list[1:]:
        if (ax.is_log_x, ax.is_log_y) != (ax0.is_log_x, ax0.is_log_y):
            raise ValueError("repeats disagree on log/linear axis scales")
    x_span = float(np.median([a.span("x") for a in axes_list]))
    y_span = float(np.median([a.span("y") for a in axes_list]))
    comb = combine_repeats(*reps, x_span=x_span, y_span=y_span, x_log=ax0.is_log_x,
                           y_log=ax0.is_log_y, tol_frac=tol_frac)

    csv_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    calibration_record = {"calibrations": calib,
                          "repeat_csv": [{"repeat": k + 1, "file": p.name, "sha256": h}
                                         for k, (p, h) in enumerate(zip(paths, csv_hashes))],
                          "matching": {"tol_frac": tol_frac, "x_span": x_span, "y_span": y_span,
                                       "x_log": ax0.is_log_x, "y_log": ax0.is_log_y}}
    return LoadedCurve(name, comb, calibration_record, projects)


def enter_digitised_curve(conn, edb, op_prefix: str, experiment_id: str, x_quantity: str,
                          x_unit: str, y_quantity: str, y_unit: str,
                          repeats_paths: Sequence, project_json_path, figure: str, location: str,
                          *, dataset: str | None = None, x_role: str = "input",
                          y_role: str = "output", tol_frac: float = TOL_FRAC,
                          fixed_observations: Sequence[dict] = (), tool: str | None = None
                          ) -> list[str]:
    """Enter one digitised curve (three repeats) into the database.

    Files and checks as in ``load_curve``. The axis calibration(s) and file
    hashes go into ``digitisation.axis_calibration_json``.

    Per matched point: operating point ``{op_prefix}-pNN``, x observation (mean
    of the repeats, sigma = SD of the repeats, kind "digitisation"), same for y,
    and one digitisation row per observation. ``fixed_observations``: optional
    dicts {quantity, value, unit, role[, sigma, sigma_kind, location]} added to
    every operating point unchanged (a curve parameter read from the legend or a
    table, e.g. corrected speed); they are not digitised and get no digitisation
    row. Returns the operating-point ids."""
    cur = load_curve(repeats_paths, project_json_path, dataset=dataset, tol_frac=tol_frac)
    name, comb, calibration_record = cur.name, cur.combined, cur.calibration_record
    if tool is None:
        tool = f"WebPlotDigitizer (project format {'.'.join(map(str, cur.projects[0].version))})"

    # vocabulary/unit check before anything is written (no half-entered curve)
    vocab = edb.load_vocabulary()
    for q, unit in [(x_quantity, x_unit), (y_quantity, y_unit)] + \
            [(fo["quantity"], fo["unit"]) for fo in fixed_observations]:
        edb.to_si(vocab, q, 1.0, unit)

    op_ids = []
    for i in range(len(comb.x_mean)):
        op = f"{op_prefix}-p{i + 1:02d}"
        edb.add_operating_point(conn, op, experiment_id,
                                f"{figure}, {name}: digitised point {i + 1} of {len(comb.x_mean)}")
        for q, unit, role, mean, sd, rep in (
                (x_quantity, x_unit, x_role, comb.x_mean[i], comb.x_sigma[i], comb.x_repeats[i]),
                (y_quantity, y_unit, y_role, comb.y_mean[i], comb.y_sigma[i], comb.y_repeats[i])):
            oid = edb.add_observation(conn, op, q, float(mean), unit, role, location,
                                      sigma=float(sd), sigma_kind="digitisation")
            edb.add_digitisation(conn, oid, figure, tool, calibration_record,
                                 [float(v) for v in rep])
        for fo in fixed_observations:
            edb.add_observation(conn, op, fo["quantity"], float(fo["value"]), fo["unit"], fo["role"],
                                fo.get("location", location), sigma=fo.get("sigma"),
                                sigma_kind=fo.get("sigma_kind"))
        op_ids.append(op)
    return op_ids


# ---------------------------------------------------------------- command line
def check_curve(figure_dir, curve: str, tol_frac: float = TOL_FRAC) -> LoadedCurve:
    """Check ``<figure_dir>/<curve>_r{1,2,3}.csv`` against ``<figure_dir>/<figure>_r{1,2,3}.json``
    (layout of data/empirical/DIGITISING.md) without touching the database."""
    d = Path(figure_dir)
    return load_curve([d / f"{curve}_r{k}.csv" for k in (1, 2, 3)],
                      [d / f"{d.name}_r{k}.json" for k in (1, 2, 3)], dataset=curve,
                      tol_frac=tol_frac)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Check three WPD repeats of one curve (no database write).")
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("figure_dir", type=Path, help="e.g. data/empirical/digitised/NACA-TN-1757/fig4c")
    ap.add_argument("curve", help="curve/dataset name, e.g. alpha15_d080")
    a = ap.parse_args(argv)
    try:
        cur = check_curve(a.figure_dir, a.curve)
    except (OSError, ValueError, KeyError) as e:
        print(f"NOT OK: {e}")
        return 1
    c = cur.combined
    m = cur.calibration_record["matching"]
    print(f"OK: {a.curve}, {len(c.x_mean)} points; tolerance {m['tol_frac']:.0%} of span "
          f"(x {m['x_span']:.4g}, y {m['y_span']:.4g})")
    print(f"{'#':>3} {'x mean':>10} {'x SD':>9} {'y mean':>10} {'y SD':>9} {'y spread':>9}")
    for i in range(len(c.x_mean)):
        print(f"{i + 1:>3} {c.x_mean[i]:>10.5g} {c.x_sigma[i]:>9.2g} {c.y_mean[i]:>10.5g} "
              f"{c.y_sigma[i]:>9.2g} {c.y_spread[i]:>9.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
