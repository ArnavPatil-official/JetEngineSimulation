"""
Reader for NPARC/WIND "Common File" (CFF) grid and solution files.

The NPARC Alliance validation archive ships the Sajben transonic-diffuser
benchmark with two binary files produced by the WIND code (version 1.144,
October 1997):

* ``sajben.cgd`` — common *grid* file (81 x 51 x 1, inches)
* ``sajben.cfl`` — common *solution* file (converged Spalart-Allmaras RANS
  solution, weak-shock case, 20 000 iterations)

Both are ADF databases ("Advanced Data Format", the original container
underneath CGNS v1/v2).  ``scripts/parse_sajben_cfd.py`` skipped them because
reading ADF appeared to require pyCGNS.  It does not: ADF is a documented,
self-describing tree of fixed-layout records, and this module reads it with
NumPy alone so the repo takes on no new dependency.

ADF on-disk layout (all pointers are 12 ASCII hex chars: 8 block + 4 offset,
block size 4096 bytes)::

    file header  186 B  "@(#)ADF Database Version ..." + dates + numeric
                        format ('B' = big-endian IEEE) + root-node pointer
    node record  246 B  "NoDe" name[32] label[32] n_sub[8] n_entries[8]
                        sub_table_ptr[12] data_type[32] n_dims[2]
                        dims[12*8] n_chunks[4] chunk_ptr[12] "TaiL"
    sub-node tbl        "SNTb" end_ptr[12] (name[32] child_ptr[12])* "snTE"
    data chunk          "DaTa" end_ptr[12] <raw bytes> "dATA"
    chunk table         "DCtb" end_ptr[12] (start_ptr[12] end_ptr[12])* "dcTB"

WIND non-dimensionalisation (decoded from the ``RefScl`` record and verified
against the run listing ``sajben.lis``):

    rho     / rho_ref            rho_ref = p_ref / (R T_ref)
    rho*u   / (rho_ref a_ref)    a_ref   = sqrt(gamma R T_ref)
    rho*e0  / (rho_ref a_ref^2)
    mul,mut / mu_ref             (laminar / eddy viscosity)
    x, y    in inches

where ``p_ref, T_ref, M_ref, mu_ref, gamma, R`` are the freestream reference
conditions stored in the root ``rdat_cff`` record.

Usage::

    from simulation.nozzle.wind_cff import load_wind_solution
    sol = load_wind_solution("data/raw/cfd_datasets/nasa/transdif01/sajben.cgd",
                             "data/raw/cfd_datasets/nasa/transdif01/sajben.cfl")
    sol.p / sol.p0          # wall-pressure ratio field, shape (nj, ni)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_BLOCK = 4096
_NODE_LEN = 246
_DTYPES = {"I4": ">i4", "I8": ">i8", "R4": ">f4", "R8": ">f8", "C1": "S1", "B1": "u1"}
INCH_TO_M = 0.0254


# ---------------------------------------------------------------------------
# Low-level ADF reader
# ---------------------------------------------------------------------------

def _ptr(raw: bytes) -> int:
    s = raw.decode("ascii")
    return int(s[:8], 16) * _BLOCK + int(s[8:12], 16)


def _read_node(buf: bytes, off: int) -> dict:
    n = buf[off:off + _NODE_LEN]
    if n[:4] != b"NoDe" or n[242:246] != b"TaiL":
        raise ValueError(f"ADF node record expected at byte {off}, got {n[:4]!r}")
    ndim = int(n[128:130], 16)
    return {
        "name": n[4:36].decode("ascii", "replace").rstrip(),
        "label": n[36:68].decode("ascii", "replace").rstrip(),
        "n_sub": int(n[68:76], 16),
        "sub_ptr": _ptr(n[84:96]),
        "dtype": n[96:128].decode("ascii", "replace").rstrip(),
        "dims": [int(n[130 + 8 * i:138 + 8 * i], 16) for i in range(ndim)],
        "n_chunks": int(n[226:230], 16),
        "chunk_ptr": _ptr(n[230:242]),
    }


def _children(buf: bytes, node: dict) -> list:
    if node["n_sub"] == 0:
        return []
    off = node["sub_ptr"]
    if buf[off:off + 4] != b"SNTb":
        raise ValueError(f"ADF sub-node table expected at byte {off}")
    out = []
    p = off + 16
    for _ in range(node["n_sub"]):
        name = buf[p:p + 32].decode("ascii", "replace").rstrip()
        out.append((name, _ptr(buf[p + 32:p + 44])))
        p += 44
    return out


def _read_data(buf: bytes, node: dict, big_endian: bool) -> Optional[np.ndarray]:
    code = node["dtype"][:2]
    if code == "MT" or node["n_chunks"] == 0 or code not in _DTYPES:
        return None
    dt = np.dtype(_DTYPES[code])
    if not big_endian and dt.byteorder == ">":
        dt = dt.newbyteorder("<")
    n_el = int(np.prod(node["dims"])) if node["dims"] else 0
    nbytes = n_el * dt.itemsize
    chunks = []
    if node["n_chunks"] == 1:
        off = node["chunk_ptr"]
        if buf[off:off + 4] != b"DaTa":
            raise ValueError(f"ADF data chunk expected at byte {off}")
        chunks.append(buf[off + 16:off + 16 + nbytes])
    else:
        off = node["chunk_ptr"]
        if buf[off:off + 4] != b"DCtb":
            raise ValueError(f"ADF data-chunk table expected at byte {off}")
        p = off + 16
        for _ in range(node["n_chunks"]):
            s, e = _ptr(buf[p:p + 12]), _ptr(buf[p + 12:p + 24])
            p += 24
            if buf[s:s + 4] != b"DaTa":
                raise ValueError(f"ADF data chunk expected at byte {s}")
            chunks.append(buf[s + 16:e - 4])
    raw = b"".join(chunks)[:nbytes]
    arr = np.frombuffer(raw, dtype=dt)
    # ADF stores Fortran (column-major) order; expose as C-order with the
    # first ADF dimension varying fastest.
    return arr.reshape(node["dims"][::-1]) if n_el else arr


def read_adf(path: str | Path) -> Dict[str, Optional[np.ndarray]]:
    """Read every node of an ADF file into ``{"/path/to/node": array_or_None}``."""
    buf = Path(path).read_bytes()
    if not buf.startswith(b"@(#)ADF Database Version"):
        raise ValueError(f"{path} is not an ADF database")
    big_endian = chr(buf[100]) == "B"
    root = _ptr(buf[134:146])
    out: Dict[str, Optional[np.ndarray]] = {}

    def walk(off: int, prefix: str) -> None:
        node = _read_node(buf, off)
        key = f"{prefix}/{node['name']}"
        out[key] = _read_data(buf, node, big_endian)
        for _, cptr in _children(buf, node):
            walk(cptr, key)

    walk(root, "")
    return out


def adf_dates(path: str | Path) -> tuple[str, str]:
    """Creation and last-modification date strings from the ADF file header."""
    hdr = Path(path).read_bytes()[:186]
    return hdr[36:64].decode("ascii").strip(), hdr[68:96].decode("ascii").strip()


# ---------------------------------------------------------------------------
# WIND common-file decoding
# ---------------------------------------------------------------------------

_ZONE = "/ADF MotherNode/ZONE   1"
_ROOT = "/ADF MotherNode"


@dataclass
class WindSolution:
    """A WIND common-file solution on its grid, in SI units, shape (nj, ni)."""

    ni: int
    nj: int
    x: np.ndarray            # m
    y: np.ndarray            # m
    rho: np.ndarray          # kg/m^3
    u: np.ndarray            # m/s
    v: np.ndarray            # m/s
    p: np.ndarray            # Pa
    T: np.ndarray            # K
    mu_l: np.ndarray         # Pa s  (laminar)
    mu_t: np.ndarray         # Pa s  (eddy)
    reference: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)

    # convenience -------------------------------------------------------
    @property
    def p0(self) -> float:
        return self.reference["p0"]

    @property
    def T0(self) -> float:
        return self.reference["T0"]

    @property
    def mach(self) -> np.ndarray:
        g, R = self.reference["gamma"], self.reference["R"]
        return np.sqrt(self.u ** 2 + self.v ** 2) / np.sqrt(g * R * self.T)

    @property
    def lower_wall(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[0], self.y[0]

    @property
    def upper_wall(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[-1], self.y[-1]

    @property
    def i_throat(self) -> int:
        return int(np.argmin(self.y[-1] - self.y[0]))

    @property
    def x_throat(self) -> float:
        return float(self.x[-1, self.i_throat])

    @property
    def h_throat(self) -> float:
        i = self.i_throat
        return float(self.y[-1, i] - self.y[0, i])

    def as_flat_arrays(self) -> Dict[str, np.ndarray]:
        """Flatten every field to (ni*nj,) in C order of the (nj, ni) grids."""
        return {
            k: getattr(self, k).ravel()
            for k in ("x", "y", "rho", "u", "v", "p", "T", "mu_l", "mu_t")
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wind_solution(cgd_path: str | Path, cfl_path: str | Path) -> WindSolution:
    """
    Decode a WIND common grid + common solution pair into SI fields.

    Raises ``ValueError`` if the two files disagree on grid size or if any
    of the expected variable nodes are missing.
    """
    cgd_path, cfl_path = Path(cgd_path), Path(cfl_path)
    g = read_adf(cgd_path)
    s = read_adf(cfl_path)

    idat = s[f"{_ROOT}/idat_cff"]
    rdat = s[f"{_ROOT}/rdat_cff"]
    if idat is None or rdat is None:
        raise ValueError("solution file lacks the CFF idat/rdat records")
    ni, nj, nk = (int(v) for v in idat[:3])
    n_pts = int(idat[4])
    if nk != 1 or n_pts != ni * nj:
        raise ValueError(f"expected a 2-D single-zone file, got ni={ni} nj={nj} nk={nk}")

    for key in ("x", "y"):
        if g.get(f"{_ZONE}/{key}") is None:
            raise ValueError(f"grid file lacks node {key}")
    for key in ("rho", "rho*u", "rho*v", "rho*e0", "mul", "mut"):
        if s.get(f"{_ZONE}/{key}") is None:
            raise ValueError(f"solution file lacks node {key}")
    if g[f"{_ZONE}/x"].size != n_pts:
        raise ValueError("grid and solution files disagree on point count")

    # Reference conditions (root rdat_cff layout decoded from the WIND listing)
    M_ref = float(rdat[4])
    p_ref = float(rdat[5])
    T_ref = float(rdat[6])
    mu_ref = float(rdat[11])
    Re_ref = float(rdat[12])
    gamma = float(rdat[23])
    R = float(rdat[24])
    rho_ref = p_ref / (R * T_ref)
    a_ref = np.sqrt(gamma * R * T_ref)
    fac = 1.0 + 0.5 * (gamma - 1.0) * M_ref ** 2
    p0 = p_ref * fac ** (gamma / (gamma - 1.0))
    T0 = T_ref * fac

    shape = (nj, ni)
    f64 = lambda key: s[f"{_ZONE}/{key}"].reshape(shape).astype(np.float64)  # noqa: E731
    x = g[f"{_ZONE}/x"].reshape(shape).astype(np.float64) * INCH_TO_M
    y = g[f"{_ZONE}/y"].reshape(shape).astype(np.float64) * INCH_TO_M
    rho = f64("rho") * rho_ref
    ru = f64("rho*u") * rho_ref * a_ref
    rv = f64("rho*v") * rho_ref * a_ref
    re0 = f64("rho*e0") * rho_ref * a_ref ** 2
    mu_l = f64("mul") * mu_ref
    mu_t = f64("mut") * mu_ref

    u = ru / rho
    v = rv / rho
    p = (gamma - 1.0) * (re0 - 0.5 * rho * (u ** 2 + v ** 2))
    T = p / (rho * R)

    cdat = s[f"{_ROOT}/cdat_cff"]
    title = cdat.tobytes().decode("ascii", "replace").strip() if cdat is not None else ""
    created, modified = adf_dates(cfl_path)

    return WindSolution(
        ni=ni, nj=nj, x=x, y=y, rho=rho, u=u, v=v, p=p, T=T, mu_l=mu_l, mu_t=mu_t,
        reference={
            "M_ref": M_ref, "p_ref": p_ref, "T_ref": T_ref, "rho_ref": rho_ref,
            "a_ref": float(a_ref), "mu_ref": mu_ref, "Re_ref": Re_ref,
            "gamma": gamma, "R": R, "p0": float(p0), "T0": float(T0),
        },
        provenance={
            "cgd_path": str(cgd_path), "cgd_sha256": _sha256(cgd_path),
            "cfl_path": str(cfl_path), "cfl_sha256": _sha256(cfl_path),
            "cfl_created": created, "cfl_modified": modified,
            "title": title,
            "solver": "WIND 1.144 (NPARC Alliance validation archive, Study #1, C. Towne)",
            "turbulence_model": "Spalart-Allmaras",
            "walls": "adiabatic no-slip",
        },
    )
