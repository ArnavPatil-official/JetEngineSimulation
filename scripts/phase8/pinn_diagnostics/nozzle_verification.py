"""Track 4b: Ma et al. residual verifier and exact quasi-1D nozzle references.

1. ``ma_residuals`` evaluates the numbered equations of Ma et al. (AST 168,
   111002, printed p. 5) literally, by autograd:

     Eq22  d(rho u)/dx + d(rho v)/dy
     Eq23  rho(u u_x + v u_y) + p_x - d/dx(mu u_x) - d/dy(mu u_y)
           + d(rho UU)/dx + d(rho UV)/dy
     Eq24  rho(u v_x + v v_y) + p_y - d/dx(mu v_x) - d/dy(mu v_y)
           + d(rho UV)/dx + d(rho VV)/dy
     Eq25  rho cp (u T_x + v T_y) - d/dx(kappa T_x) - d/dy(kappa T_y) - Phi,
           kappa = conductivity + mu_t/Pr  (exactly as printed)
     Eq26  p - rho R T
     Phi = mu (4/3 (u_x^2 + v_y^2 - u_x v_y) + (u_y + v_x)^2)   (from Eqs. 7-10)

   mu is the molecular viscosity; UU, VV, UV are independent Reynolds-stress
   covariances; mu_t appears only in the printed thermal coefficient. The
   printed coefficient adds a conductivity (W/(m K)) to mu_t/Pr (Pa s); its
   dimensional meaning is unresolved, so this is a **dimensionless algebra
   verifier**, not a dimensional replication. The mu_t term is NOT multiplied
   by cp here.

2. ``analytic_forcing`` evaluates the same equations for the registered
   manufactured fields q = base + amplitude*h(ax x + ay y) from closed-form
   activation derivatives in NumPy, without calling the residual or autograd.
   The verifier checks residual - forcing ~ 0; omission controls must not.

3. ``ma_loss_weights``: Ma Eqs. 31-33 (asymmetric, current-loss, detached).

4. Exact quasi-1D nozzle references (no network): isentropic subsonic flow,
   the choked subsonic/supersonic branches, the normal-shock oracle and the
   back-pressure -> shock-position inversion. The shock is algebraic; nothing
   is differentiated through it. These are reference solutions, not PINN results.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
import torch
from scipy.optimize import brentq

DTYPE = torch.float64
FIELD_NAMES = ("rho", "u", "v", "T", "mu", "conductivity", "mu_t", "UU", "VV", "UV")
FIELDS_EXPR = "base+amplitude*activation(ax*x+ay*y), p=rho*R*T"
THERMAL_EXPR = "conductivity+mu_t/Pr"
AREA_EXPR = "1+0.5*x*x"
EQUATIONS = ("mass", "xmom", "ymom", "energy", "eos")
OMISSIONS = ("viscosity_gradient", "stress_divergence", "conductivity_gradient", "dissipation")
RUNGS = ("1_smooth_subsonic", "2_choked_isentropic", "3_shock_oracle", "4_back_pressure_shock")
SHOCK_ERROR_KEYS = ("exit_pressure_rel", "mass_jump_rel", "momentum_jump_rel", "total_enthalpy_jump_rel",
                    "pressure_jump_vs_rankine_hugoniot_rel", "downstream_mach_vs_rankine_hugoniot_rel",
                    "total_pressure_loss_from_sides_rel", "mass_flow_profile_rel", "total_enthalpy_profile_rel",
                    "total_pressure_profile_rel")


# ---------------------------------------------------------------------------
# Activations: closed forms (NumPy) and the torch versions used by the fields
# ---------------------------------------------------------------------------

def activation_numpy(name: str, z):
    """Return (h, h', h'') of the activation from closed forms."""
    z = np.asarray(z, dtype=float)
    if name == "tanh":
        h = np.tanh(z)
        return h, 1.0 - h * h, -2.0 * h * (1.0 - h * h)
    if name == "silu":
        s = 1.0 / (1.0 + np.exp(-z))
        return z * s, s + z * s * (1.0 - s), 2.0 * s * (1.0 - s) + z * s * (1.0 - s) * (1.0 - 2.0 * s)
    raise ValueError(f"unsupported activation {name!r}")


def activation_torch(name: str):
    if name == "tanh":
        return torch.tanh
    if name == "silu":
        return torch.nn.functional.silu
    raise ValueError(f"unsupported activation {name!r}")


def min_max_normalize(X, lo: float, hi: float):
    """Ma Eq. 16 input scaling. Physical derivatives pick up 1/(hi - lo) per order."""
    return (X - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Manufactured fields
# ---------------------------------------------------------------------------

def manufactured_grid(mcfg: dict) -> tuple:
    lo, hi = mcfg["coordinates"]
    x, y = np.meshgrid(np.linspace(lo, hi, mcfg["grid_x"]), np.linspace(lo, hi, mcfg["grid_y"]), indexing="ij")
    return x.ravel(), y.ravel()


def manufactured_fields_torch(x: torch.Tensor, y: torch.Tensor, coeffs: dict, activation: str, R: float) -> dict:
    h = activation_torch(activation)
    f = {}
    for name in FIELD_NAMES:
        b, a, ax, ay = coeffs[name]
        f[name] = b + a * h(ax * x + ay * y)
    f["p"] = f["rho"] * R * f["T"]
    return f


def _check_coefficients(coeffs: dict) -> None:
    for name in FIELD_NAMES:
        b, a, _, _ = coeffs[name]
        if name in ("rho", "T", "mu", "conductivity", "mu_t") and b - abs(a) <= 0.0:
            raise ValueError(f"manufactured {name} must stay positive")


def analytic_forcing(x, y, coeffs: dict, activation: str, R: float, cp: float, Pr: float) -> dict:
    """Eqs. 22-26 evaluated by hand-expanded closed forms (NumPy only)."""
    q = {}
    for name in FIELD_NAMES:
        b, a, ax, ay = coeffs[name]
        h, h1, h2 = activation_numpy(activation, ax * np.asarray(x) + ay * np.asarray(y))
        q[name] = {"v": b + a * h, "x": a * ax * h1, "y": a * ay * h1,
                   "xx": a * ax * ax * h2, "yy": a * ay * ay * h2}
    rho, u, v, T, mu = (q[k] for k in ("rho", "u", "v", "T", "mu"))
    k, mt, UU, VV, UV = (q[n] for n in ("conductivity", "mu_t", "UU", "VV", "UV"))
    px = R * (rho["x"] * T["v"] + rho["v"] * T["x"])
    py = R * (rho["y"] * T["v"] + rho["v"] * T["y"])
    mass = rho["x"] * u["v"] + rho["v"] * u["x"] + rho["y"] * v["v"] + rho["v"] * v["y"]
    xmom = (rho["v"] * (u["v"] * u["x"] + v["v"] * u["y"]) + px
            - (mu["x"] * u["x"] + mu["v"] * u["xx"]) - (mu["y"] * u["y"] + mu["v"] * u["yy"])
            + (rho["x"] * UU["v"] + rho["v"] * UU["x"]) + (rho["y"] * UV["v"] + rho["v"] * UV["y"]))
    ymom = (rho["v"] * (u["v"] * v["x"] + v["v"] * v["y"]) + py
            - (mu["x"] * v["x"] + mu["v"] * v["xx"]) - (mu["y"] * v["y"] + mu["v"] * v["yy"])
            + (rho["x"] * UV["v"] + rho["v"] * UV["x"]) + (rho["y"] * VV["v"] + rho["v"] * VV["y"]))
    kap, kap_x, kap_y = k["v"] + mt["v"] / Pr, k["x"] + mt["x"] / Pr, k["y"] + mt["y"] / Pr
    phi = mu["v"] * (4.0 * (u["x"] ** 2 + v["y"] ** 2 - u["x"] * v["y"]) / 3.0 + (u["y"] + v["x"]) ** 2)
    energy = (rho["v"] * cp * (u["v"] * T["x"] + v["v"] * T["y"])
              - (kap_x * T["x"] + kap * T["xx"]) - (kap_y * T["y"] + kap * T["yy"]) - phi)
    return {"mass": mass, "xmom": xmom, "ymom": ymom, "energy": energy, "eos": np.zeros_like(mass)}


# ---------------------------------------------------------------------------
# Literal residual (autograd)
# ---------------------------------------------------------------------------

def _d(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(q, w, torch.ones_like(q), create_graph=True)[0]


def ma_residuals(x: torch.Tensor, y: torch.Tensor, f: dict, R: float, cp: float, Pr: float,
                 omit: tuple = ()) -> dict:
    """Ma Eqs. 22-26 as printed; ``omit`` drops one term family (negative controls)."""
    bad = set(omit) - set(OMISSIONS)
    if bad:
        raise ValueError(f"unknown omission {sorted(bad)}")
    rho, u, v, T, p, mu = f["rho"], f["u"], f["v"], f["T"], f["p"], f["mu"]
    ux, uy, vx, vy = _d(u, x), _d(u, y), _d(v, x), _d(v, y)
    Tx, Ty, px, py = _d(T, x), _d(T, y), _d(p, x), _d(p, y)
    mass = _d(rho * u, x) + _d(rho * v, y)
    if "viscosity_gradient" in omit:          # Laplacian form: mu treated as constant
        visc_u = mu * (_d(ux, x) + _d(uy, y))
        visc_v = mu * (_d(vx, x) + _d(vy, y))
    else:
        visc_u = _d(mu * ux, x) + _d(mu * uy, y)
        visc_v = _d(mu * vx, x) + _d(mu * vy, y)
    if "stress_divergence" in omit:
        st_u = st_v = torch.zeros_like(u)
    else:
        st_u = _d(rho * f["UU"], x) + _d(rho * f["UV"], y)
        st_v = _d(rho * f["UV"], x) + _d(rho * f["VV"], y)
    xmom = rho * (u * ux + v * uy) + px - visc_u + st_u
    ymom = rho * (u * vx + v * vy) + py - visc_v + st_v
    kappa = f["conductivity"] + f["mu_t"] / Pr    # literal printed coefficient (dimensionless verifier)
    if "conductivity_gradient" in omit:
        cond = kappa * (_d(Tx, x) + _d(Ty, y))
    else:
        cond = _d(kappa * Tx, x) + _d(kappa * Ty, y)
    phi = torch.zeros_like(u) if "dissipation" in omit else \
        mu * (4.0 * (ux ** 2 + vy ** 2 - ux * vy) / 3.0 + (uy + vx) ** 2)
    energy = rho * cp * (u * Tx + v * Ty) - cond - phi
    eos = p - rho * R * T
    return {"mass": mass, "xmom": xmom, "ymom": ymom, "energy": energy, "eos": eos}


def verify_manufactured(mcfg: dict, activation: str) -> dict:
    """Forced-residual check plus the four omission controls for one activation."""
    if mcfg["fields"] != FIELDS_EXPR or mcfg["literal_thermal_coefficient"] != THERMAL_EXPR:
        raise ValueError("registration manufactured fields/thermal coefficient differ from the implemented ones")
    coeffs = {k: [float(c) for c in v] for k, v in mcfg["coefficients"].items()}
    _check_coefficients(coeffs)
    R, cp, Pr = float(mcfg["R"]), float(mcfg["cp"]), float(mcfg["Pr"])
    xs, ys = manufactured_grid(mcfg)
    forcing = analytic_forcing(xs, ys, coeffs, activation, R, cp, Pr)

    def residual(omit=()):
        x = torch.tensor(xs, dtype=DTYPE, requires_grad=True)
        y = torch.tensor(ys, dtype=DTYPE, requires_grad=True)
        res = ma_residuals(x, y, manufactured_fields_torch(x, y, coeffs, activation, R), R, cp, Pr, omit)
        return {k: t.detach().numpy() for k, t in res.items()}

    full = residual()
    per_eq = {}
    for eq in EQUATIONS:
        diff = np.abs(full[eq] - forcing[eq])
        scale = float(np.max(np.abs(forcing[eq])))
        per_eq[eq] = {"max_abs_forced_residual": float(diff.max()), "max_abs_forcing": scale,
                      "relative_forcing_disagreement": float(diff.max() / scale) if scale > 0 else None}
    controls = {}
    for om in OMISSIONS:
        r = residual((om,))
        controls[om] = max(float(np.max(np.abs(r[eq] - forcing[eq]))) for eq in EQUATIONS)
    abs_max = max(v["max_abs_forced_residual"] for v in per_eq.values())
    rel_max = max(v["relative_forcing_disagreement"] for v in per_eq.values()
                  if v["relative_forcing_disagreement"] is not None)
    ok = (abs_max <= mcfg["absolute_residual_max"] and rel_max <= mcfg["relative_forcing_disagreement_max"]
          and all(c > mcfg["negative_control_min_discrepancy"] for c in controls.values()))
    return {"status": "PASS" if ok else "FAIL", "activation": activation, "grid_points": int(xs.size),
            "max_abs_forced_residual": abs_max, "max_relative_forcing_disagreement": rel_max,
            "per_equation": per_eq, "negative_control_discrepancy": controls,
            "tolerances": {"absolute_residual_max": mcfg["absolute_residual_max"],
                           "relative_forcing_disagreement_max": mcfg["relative_forcing_disagreement_max"],
                           "negative_control_min_discrepancy": mcfg["negative_control_min_discrepancy"]},
            "interpretation": mcfg["interpretation"]}


# ---------------------------------------------------------------------------
# Ma Eqs. 31-33 current-loss weights
# ---------------------------------------------------------------------------

def ma_loss_weights(L_data, L_phys, L_bc, eps: float) -> dict:
    """Ma Eqs. 31-33. Asymmetric: data compares with the sum of the other two,
    physics and BC each compare with the data loss only. Detached from the graph.
    Always CPU float64, also for Python-scalar inputs (torch would make those float32)."""
    Ld, Lp, Lb = (torch.as_tensor(t, device="cpu", dtype=DTYPE).detach() for t in (L_data, L_phys, L_bc))
    return {"data": 0.1 + 0.9 * torch.sigmoid((Lp + Lb - Ld) / (Ld + eps)),
            "phys": 0.1 + 0.9 * torch.sigmoid((Ld - Lp) / (Lp + eps)),
            "bc": 0.1 + 0.9 * torch.sigmoid((Ld - Lb) / (Lb + eps))}


# ---------------------------------------------------------------------------
# Quasi-1D nozzle: exact references
# ---------------------------------------------------------------------------

def _check_gamma(g: float) -> None:
    if not (math.isfinite(g) and g > 1.0):
        raise ValueError("gamma must be finite and > 1")


def area(x):
    return 1.0 + 0.5 * np.asarray(x, dtype=float) ** 2


def area_mach_ratio(M, g: float):
    """A/A* of isentropic flow at Mach M."""
    _check_gamma(g)
    M = np.asarray(M, dtype=float)
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area_ratio(ratio: float, g: float, branch: str, xtol: float) -> float:
    """Invert A/A* on the subsonic or supersonic branch (brentq)."""
    _check_gamma(g)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("A/A* must be >= 1")
    if ratio == 1.0:
        return 1.0
    f = lambda M: float(area_mach_ratio(M, g)) - ratio  # noqa: E731
    if branch == "subsonic":
        return brentq(f, 1e-8, 1.0, xtol=xtol)
    if branch == "supersonic":
        hi = 2.0
        while f(hi) < 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=xtol)
    raise ValueError(f"unknown branch {branch!r}")


def isentropic_state(M, g: float, R: float, p0: float, T0: float) -> dict:
    M = np.asarray(M, dtype=float)
    T = T0 / (1.0 + 0.5 * (g - 1.0) * M * M)
    p = p0 * (T / T0) ** (g / (g - 1.0))
    rho = p / (R * T)
    return {"M": M, "T": T, "p": p, "rho": rho, "u": M * np.sqrt(g * R * T)}


def normal_shock(M1: float, g: float) -> dict:
    _check_gamma(g)
    if not M1 >= 1.0:
        raise ValueError("normal shock needs M1 >= 1")
    if M1 == 1.0:          # sonic limit: no jump (exact, avoids ulp-level ratios at the throat)
        return {"M2": 1.0, "p_ratio": 1.0, "rho_ratio": 1.0, "T_ratio": 1.0, "p0_ratio": 1.0}
    m2 = M1 * M1
    rho_ratio = (g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)
    p_ratio = 1.0 + 2.0 * g / (g + 1.0) * (m2 - 1.0)
    return {"M2": math.sqrt((1.0 + 0.5 * (g - 1.0) * m2) / (g * m2 - 0.5 * (g - 1.0))),
            "p_ratio": p_ratio, "rho_ratio": rho_ratio, "T_ratio": p_ratio / rho_ratio,
            "p0_ratio": rho_ratio ** (g / (g - 1.0)) * (1.0 / p_ratio) ** (1.0 / (g - 1.0))}


def shock_oracle(ocfg: dict, tol: float) -> dict:
    """gamma 1.4, M1 2: M2 = 1/sqrt(3), p2/p1 = 9/2, rho2/rho1 = 8/3, T2/T1 = 27/16."""
    exact = {"M2": 1.0 / math.sqrt(3.0), "p_ratio": float(Fraction(9, 2)),
             "rho_ratio": float(Fraction(8, 3)), "T_ratio": float(Fraction(27, 16))}
    reg = {"M2": ocfg["M2"], "pressure_ratio": ocfg["pressure_ratio"], "density_ratio": ocfg["density_ratio"],
           "temperature_ratio": ocfg["temperature_ratio"]}
    if reg != {"M2": "1/sqrt(3)", "pressure_ratio": 4.5, "density_ratio": "8/3", "temperature_ratio": "27/16"}:
        raise ValueError("registered oracle differs from the implemented rational values")
    got = normal_shock(float(ocfg["M1"]), float(ocfg["gamma"]))
    err = {k: abs(got[k] / v - 1.0) for k, v in exact.items()}
    return {"status": "PASS" if max(err.values()) <= tol else "FAIL", "relative_errors": err,
            "tolerance": tol, "computed": {k: got[k] for k in exact}}


@dataclass(frozen=True)
class ShockCase:
    """Complete input of one back-pressure shock case."""
    name: str
    gamma: float
    R: float
    p0: float
    T0: float
    area: str
    x_domain: tuple
    grid_points: int
    back_pressure: float

    def identity(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def validate_cases(cases: list) -> list:
    """Reject invalid inputs and inconsistent duplicates; drop exact duplicates."""
    seen, out = {}, []
    for c in cases:
        _check_gamma(c.gamma)
        if c.area != AREA_EXPR:
            raise ValueError(f"unsupported area law {c.area!r}")
        if not (math.isfinite(c.back_pressure) and c.back_pressure > 0.0):
            raise ValueError("back pressure must be finite and positive")
        if c.name in seen:
            if seen[c.name] != c.identity():
                raise ValueError(f"inconsistent duplicate input rows for case {c.name!r}")
            continue
        seen[c.name] = c.identity()
        out.append(c)
    return out


class QuasiOneD:
    """Area 1 + 0.5 x^2 on x_domain; throat at x = 0 (A_t = 1), exit at x_domain[1]."""

    def __init__(self, gamma: float, R: float, p0: float, T0: float, x_domain, grid_points: int, xtol: float):
        _check_gamma(gamma)
        lo, hi = x_domain
        if not (lo < 0.0 < hi):
            raise ValueError("domain must contain the throat at x = 0")
        self.g, self.R, self.p0, self.T0, self.xtol = gamma, R, p0, T0, xtol
        self.cp = gamma * R / (gamma - 1.0)
        self.x = np.linspace(lo, hi, grid_points)
        self.x_exit = float(hi)
        self.A_t = float(area(0.0))

    @classmethod
    def from_case(cls, case: ShockCase, xtol: float) -> "QuasiOneD":
        return cls(case.gamma, case.R, case.p0, case.T0, case.x_domain, case.grid_points, xtol)

    def _mach(self, ratio, branch):
        return mach_from_area_ratio(float(ratio), self.g, branch, self.xtol)

    def _invariants(self, st: dict, A, p0_local) -> dict:
        mdot = st["rho"] * st["u"] * A
        h0 = self.cp * st["T"] + 0.5 * st["u"] ** 2
        p0_rec = st["p"] * (1.0 + 0.5 * (self.g - 1.0) * st["M"] ** 2) ** (self.g / (self.g - 1.0))
        return {"mdot": mdot, "h0": h0, "p0_rel": np.abs(p0_rec / p0_local - 1.0)}

    def smooth_subsonic(self, M_in: float) -> dict:
        A = area(self.x)
        A_star = float(A[0] / area_mach_ratio(M_in, self.g))
        if not A_star < self.A_t:
            raise ValueError("inlet Mach chokes the throat; not a smooth subsonic case")
        M = np.array([self._mach(a / A_star, "subsonic") for a in A])
        st = isentropic_state(M, self.g, self.R, self.p0, self.T0)
        inv = self._invariants(st, A, self.p0)
        return {"M": M, "state": st, "errors": {
            "mass_flow_rel": float(np.max(np.abs(inv["mdot"] / inv["mdot"][0] - 1.0))),
            "total_enthalpy_rel": float(np.max(np.abs(inv["h0"] / (self.cp * self.T0) - 1.0))),
            "total_pressure_rel": float(np.max(inv["p0_rel"])),
            "area_ratio_roundtrip_rel": float(np.max(np.abs(area_mach_ratio(M, self.g) * A_star / A - 1.0))),
            "inlet_mach_abs": float(abs(M[0] - M_in)),
            "max_mach": float(M.max())}}

    def choked_mass_flow(self) -> float:
        g = self.g
        return self.A_t * self.p0 * math.sqrt(g / (self.R * self.T0)) \
            * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))

    def choked_isentropic(self) -> dict:
        A = area(self.x)
        M = np.array([self._mach(a / self.A_t, "subsonic" if x < 0.0 else "supersonic")
                      for x, a in zip(self.x, A)])
        st = isentropic_state(M, self.g, self.R, self.p0, self.T0)
        inv = self._invariants(st, A, self.p0)
        up, down = self.x < 0.0, self.x > 0.0
        return {"M": M, "state": st, "errors": {
            "mass_flow_vs_choked_rel": float(np.max(np.abs(inv["mdot"] / self.choked_mass_flow() - 1.0))),
            "total_enthalpy_rel": float(np.max(np.abs(inv["h0"] / (self.cp * self.T0) - 1.0))),
            "total_pressure_rel": float(np.max(inv["p0_rel"])),
            "area_ratio_roundtrip_rel": float(np.max(np.abs(area_mach_ratio(M, self.g) * self.A_t / A - 1.0))),
            "throat_mach_abs": float(np.min(np.abs(M - 1.0))),
            "branches_ok": bool(np.all(M[up] < 1.0) and np.all(M[down] > 1.0))}}

    def shock_jump(self, xs: float) -> dict:
        """The two sides of a normal shock at throat <= xs <= exit, as the profile evaluates them.

        Upstream: supersonic branch with (p0, A* = A_t). Downstream: subsonic
        branch with (p02, A*2 = A_t p0/p02), i.e. the downstream state comes
        from the area inversion, not from the Rankine-Hugoniot M2, so the side
        conservation checks are not tautological.
        """
        if not 0.0 <= xs <= self.x_exit:
            raise ValueError("shock position outside the divergent section")
        a = float(area(xs))
        M1 = self._mach(a / self.A_t, "supersonic")
        sh = normal_shock(M1, self.g)
        p02 = self.p0 * sh["p0_ratio"]
        A_star2 = self.A_t / sh["p0_ratio"]
        r2 = a / A_star2
        if 1.0 - 1e-12 < r2 < 1.0:      # analytically (A/A*)(M2) >= 1; a few ulp low only as M1 -> 1
            r2 = 1.0
        M2_branch = self._mach(r2, "subsonic")
        s1 = isentropic_state(M1, self.g, self.R, self.p0, self.T0)
        s2 = isentropic_state(M2_branch, self.g, self.R, p02, self.T0)
        return {"M1": M1, "M2_branch": M2_branch, "shock": sh, "p02": p02, "A_star2": A_star2,
                "up": s1, "down": s2}

    def exit_pressure(self, xs: float) -> float:
        j = self.shock_jump(xs)
        Me = self._mach(area(self.x_exit) / j["A_star2"], "subsonic")
        return float(isentropic_state(Me, self.g, self.R, j["p02"], self.T0)["p"])

    def internal_shock_range(self) -> tuple:
        """(lowest, highest) back pressure that places a normal shock inside [throat, exit]."""
        return self.exit_pressure(self.x_exit), self.exit_pressure(0.0)

    def invert_back_pressure(self, pb: float) -> float:
        lo, hi = self.internal_shock_range()
        if not (math.isfinite(pb) and lo <= pb <= hi):
            raise ValueError(f"back pressure {pb!r} outside the internal-shock range [{lo!r}, {hi!r}]")
        if pb == hi:            # both endpoints are valid: shock at the throat ...
            return 0.0
        if pb == lo:            # ... or at the exit
            return self.x_exit
        return brentq(lambda xs: self.exit_pressure(xs) - pb, 0.0, self.x_exit, xtol=self.xtol)

    def shock_profile(self, xs: float) -> dict:
        """Grid profile. Jump convention: points with x < xs are upstream; a point at
        x == xs is downstream, so a shock at the exit shows the post-shock exit pressure."""
        j = self.shock_jump(xs)
        A = area(self.x)
        M = np.empty_like(self.x)
        p0_local = np.empty_like(self.x)
        for i, (x, a) in enumerate(zip(self.x, A)):
            if x < 0.0:
                M[i], p0_local[i] = self._mach(a / self.A_t, "subsonic"), self.p0
            elif x < xs:
                M[i], p0_local[i] = self._mach(a / self.A_t, "supersonic"), self.p0
            else:
                M[i], p0_local[i] = self._mach(a / j["A_star2"], "subsonic"), j["p02"]
        T = self.T0 / (1.0 + 0.5 * (self.g - 1.0) * M * M)
        p = p0_local * (T / self.T0) ** (self.g / (self.g - 1.0))
        st = {"M": M, "T": T, "p": p, "rho": p / (self.R * T), "u": M * np.sqrt(self.g * self.R * T)}
        return {"jump": j, "state": st, "p0_local": p0_local, "area": A}


def shock_scores(nz: QuasiOneD, xs: float, pb: float, xs_ref: float) -> dict:
    """Independent scores for one recovered shock: location, exit pressure, jumps, invariants.

    The jumps compare the two branch states the profile actually uses at xs
    (``shock_jump``), and the post-shock total-pressure loss recovered from
    those states is compared with the closed-form normal-shock p02/p01.
    """
    prof = nz.shock_profile(xs)
    j, st = prof["jump"], prof["state"]
    u1, u2 = j["up"], j["down"]
    m1, m2 = u1["rho"] * u1["u"], u2["rho"] * u2["u"]
    f1, f2 = u1["p"] + m1 * u1["u"], u2["p"] + m2 * u2["u"]
    H1 = nz.cp * u1["T"] + 0.5 * u1["u"] ** 2
    H2 = nz.cp * u2["T"] + 0.5 * u2["u"] ** 2
    mdot = st["rho"] * st["u"] * prof["area"]
    h0 = nz.cp * st["T"] + 0.5 * st["u"] ** 2
    ex = nz.g / (nz.g - 1.0)
    p0_rec = st["p"] * (1.0 + 0.5 * (nz.g - 1.0) * st["M"] ** 2) ** ex
    p01_side = float(u1["p"]) * (1.0 + 0.5 * (nz.g - 1.0) * float(u1["M"]) ** 2) ** ex
    p02_side = float(u2["p"]) * (1.0 + 0.5 * (nz.g - 1.0) * float(u2["M"]) ** 2) ** ex
    sh = j["shock"]
    return {
        "shock_position": float(xs), "reference_position": float(xs_ref),
        "position_abs_error": float(abs(xs - xs_ref)),
        "back_pressure": float(pb), "exit_pressure": float(st["p"][-1]),
        "exit_pressure_rel": float(abs(st["p"][-1] / pb - 1.0)),
        "mass_jump_rel": float(abs(float(m2) / float(m1) - 1.0)),
        "momentum_jump_rel": float(abs(float(f2) / float(f1) - 1.0)),
        "total_enthalpy_jump_rel": float(abs(float(H2) / float(H1) - 1.0)),
        "pressure_jump_vs_rankine_hugoniot_rel": float(abs(float(u2["p"]) / float(u1["p"]) / sh["p_ratio"] - 1.0)),
        "downstream_mach_vs_rankine_hugoniot_rel": float(abs(j["M2_branch"] / sh["M2"] - 1.0)),
        "total_pressure_loss_from_sides_rel": float(abs(p02_side / p01_side / sh["p0_ratio"] - 1.0)),
        "mass_flow_profile_rel": float(np.max(np.abs(mdot / nz.choked_mass_flow() - 1.0))),
        "total_enthalpy_profile_rel": float(np.max(np.abs(h0 / (nz.cp * nz.T0) - 1.0))),
        "total_pressure_profile_rel": float(np.max(np.abs(p0_rec / prof["p0_local"] - 1.0))),
        "M1": float(j["M1"]), "M2": float(sh["M2"]), "M2_branch": float(j["M2_branch"]),
        "p02_over_p01": float(sh["p0_ratio"]), "total_pressure_loss": float(1.0 - sh["p0_ratio"]),
    }


def run_ladder(reg: dict) -> dict:
    """Sequential rungs: smooth -> choked -> shock oracle -> back-pressure shocks.
    A rung runs only if the one before it passed; otherwise it is BLOCKED."""
    n = reg["nozzle"]
    if n["area"] != AREA_EXPR:
        raise ValueError("registered area law differs from the implemented one")
    tol, pos_tol, xtol = n["invariant_relative_max"], n["shock_position_absolute_max"], n["root_xtol"]
    nz = QuasiOneD(n["gamma"], n["R"], n["p0"], n["T0"], n["x_domain"], n["grid_points"], xtol)
    rungs = {}

    def blocked(why):
        return {"status": "BLOCKED", "reason": why}

    sm = nz.smooth_subsonic(n["smooth_inlet_mach"])["errors"]
    sm_ok = all(sm[k] <= tol for k in ("mass_flow_rel", "total_enthalpy_rel", "total_pressure_rel",
                                       "area_ratio_roundtrip_rel", "inlet_mach_abs")) and sm["max_mach"] < 1.0
    rungs["1_smooth_subsonic"] = {"status": "PASS" if sm_ok else "FAIL", "errors": sm, "tolerance": tol}

    if rungs["1_smooth_subsonic"]["status"] != "PASS":
        rungs["2_choked_isentropic"] = blocked("rung 1 did not pass")
    else:
        ch = nz.choked_isentropic()["errors"]
        ch_ok = all(ch[k] <= tol for k in ("mass_flow_vs_choked_rel", "total_enthalpy_rel", "total_pressure_rel",
                                           "area_ratio_roundtrip_rel", "throat_mach_abs")) and ch["branches_ok"]
        rungs["2_choked_isentropic"] = {"status": "PASS" if ch_ok else "FAIL", "errors": ch, "tolerance": tol}

    if rungs["2_choked_isentropic"]["status"] != "PASS":
        rungs["3_shock_oracle"] = blocked("rung 2 did not pass")
    else:
        rungs["3_shock_oracle"] = shock_oracle(n["oracle"], tol)

    if rungs["3_shock_oracle"]["status"] != "PASS":
        rungs["4_back_pressure_shock"] = blocked("rung 3 did not pass")
        return rungs

    # reference back pressures from the fixed manufactured positions, fixed before any inversion
    xs_refs = [float(v) for v in n["manufactured_shock_positions"]]
    cases = validate_cases([ShockCase(f"shock_x{xr}", n["gamma"], n["R"], n["p0"], n["T0"], n["area"],
                                      tuple(n["x_domain"]), n["grid_points"], nz.exit_pressure(xr))
                            for xr in xs_refs])
    lo, hi = nz.internal_shock_range()
    scan = np.linspace(0.0, nz.x_exit, 41)
    pe_scan = np.array([nz.exit_pressure(s) for s in scan])
    monotone = bool(np.all(np.diff(pe_scan) < 0.0))
    per_case = {}
    for c, xr in zip(cases, xs_refs):
        cnz = QuasiOneD.from_case(c, xtol)
        xs = cnz.invert_back_pressure(c.back_pressure)
        per_case[c.name] = {"case_identity": c.identity(), **shock_scores(cnz, xs, c.back_pressure, xr)}
    rec = [per_case[c.name]["shock_position"] for c in cases]
    pbs = [c.back_pressure for c in cases]
    upstream_with_pb = all((pbs[i] - pbs[k]) * (rec[i] - rec[k]) < 0.0
                           for i in range(len(cases)) for k in range(i + 1, len(cases)))
    rejects = {}
    for label, pb in (("above_range", hi * (1.0 + 1e-6)), ("below_range", lo * (1.0 - 1e-6))):
        try:
            nz.invert_back_pressure(pb)
            rejects[label] = False
        except ValueError:
            rejects[label] = True
    ok = (monotone and upstream_with_pb and all(rejects.values())
          and all(v["position_abs_error"] <= pos_tol and all(v[k] <= tol for k in SHOCK_ERROR_KEYS)
                  and v["total_pressure_loss"] > 0.0 for v in per_case.values()))
    rungs["4_back_pressure_shock"] = {
        "status": "PASS" if ok else "FAIL", "cases": per_case,
        "internal_shock_back_pressure_range": [lo, hi],
        "exit_pressure_strictly_decreasing_with_shock_position": monotone,
        "higher_back_pressure_moves_shock_upstream": upstream_with_pb,
        "out_of_range_back_pressure_rejected": rejects,
        "tolerances": {"invariant_relative_max": tol, "shock_position_absolute_max": pos_tol, "root_xtol": xtol}}
    return rungs
