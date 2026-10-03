# P8.5-A5 — mechanism-conditioned closure, trace-species sigma0 rule, temperature convergence toward HP

Date: 2026-10-03 (registration). The decisions are the user's, of 2026-10-02.
Prospective amendment, committed before any A5 computation: no mechanism
audit has been run, the allowance `B` below is unknown, and no network has
been solved under these rules. The executable gates and paths are in
`docs/phase8_p85_a5_registration.json`; where the two differ, the JSON governs.

Records `p85_g1.json`, `p85_g1_rev1.json` and `p85_g1_rev2.json` stay as they
are; their verdicts (all FAIL) and scores are not changed by this amendment.
The registration, P8.5-A1 to A4, the A3/A4 numerics (file-loaded, never
cloned Solutions; temperature-state PSRs; unit-flow solves; take-off design)
and every C++ source are retained unchanged. No mechanism file is edited.

## Known when written

G1 rev2 (`outputs/phase8/p85_g1_rev2.json`, `43adda9`, FAIL), AE3 take-off
design, test values:

| Check | TAKE-OFF | APPROACH | IDLE |
|---|---|---|---|
| G1.1 dT, s = 1e4, no dilution (K) | 1.37e-6 | 2.499 | 0.0410 |
| G1.1 max dY (Y_eq > 1e-6) | 1.6e-10 | 8.95e-4 | 2.18e-5 |
| G1.2 worst closure (tol 1e-10) | 2.63e-8 | 2.68e-8 | 2.71e-8 |
| G1.3 T relative spread (tol 1e-12) | 2.5e-12 | 4.4e-13 | 7.6e-13 |
| G1.3 Y relative spread, all species | 1.0 | 1.21 | 0.889 |

Mechanism SHA-256: `data/A2NOx.yaml` `af3bc190…a230a087`,
`data/creck_c1c16_full.yaml` `c4fc7197…08a1edfa` (full digests in the JSON;
both equal the Phase 8 protected manifest). P8.5-A2/A3 reported that the
lumped HyChem POSF10325 steps (reactions 0–6) are element-imbalanced by
2–4e-7 C and H atoms per event in the file; that is a development probe, not
this audit, and is not used to set any number here.

## 1. Independent stoichiometry audit (`scripts/phase8/p85_audit.py`)

Each mechanism is loaded as a fresh `cantera.Solution` directly from the
unchanged file (SHA-256 checked before and after loading against the values
above), never from a network phase or a clone. Every reaction j and every
element e is audited from the loaded float64 stoichiometric coefficients
(`reactant_stoich_coeffs`, `product_stoich_coeffs`) and atom counts, in exact
rational arithmetic (`fractions.Fraction` of each float64 value); the float64
values are reported beside the exact ones.

- Reactant and product turnover: `R_ej = Σ_k a_ek ν'_kj`, `P_ej = Σ_k a_ek ν''_kj`.
- Signed defect `δ_ej = Σ_k a_ek (ν''_kj − ν'_kj) = P_ej − R_ej` (atoms per event).
- Denominator `D_ej = max(R_ej, P_ej)`; pair defect `d_ej = |δ_ej| / D_ej`.
  A pair with `D_ej = 0` and `δ_ej = 0` contributes 0; `D_ej = 0` with
  `δ_ej ≠ 0` is an audit error (the audit fails).
- No threshold: every exact non-zero defect, however small or cyclic, is kept.
  For each such reaction the record keeps its index, equation, signed atoms
  per element (exact and float), both turnover counts, and its mass defect
  `Σ_k W_k (ν''_kj − ν'_kj)` (kg/kmol, exact rational of the float64 molecular
  weights, and float).
- `B = max_{j,e} d_ej` over A2NOx, exact. It must be finite and strictly
  positive, else the audit fails.

**Honest scope.** `B` bounds the *local* stoichiometric defect of a single
reaction event, normalised by its turnover. The closure tolerance `10·B`
below is the user's mechanism-conditioned allowance. It is not a proof that
the accumulated network element or enthalpy error is bounded by `10·B`, and
no claim is made that renormalising species mass fractions would preserve
any stoichiometric proof. `B` is never derived from observed network
residuals, outlets or closures.

**Balanced control (CRECK).** `data/creck_c1c16_full.yaml` is audited the
same way. Equality rule, fixed now: CRECK is balanced to floating
representation accuracy if it has no audit error and every pair defect
`d_ej ≤ 2^-52` (exact comparison). Rationale: each decimal coefficient is
stored with relative error at most `u = 2^-53`; atom counts are integers; a
decimally balanced reaction then has `|δ| ≤ u (R + P) ≤ 2u·max(R, P)`. This
rule is not tuned after the audit is seen.

The audit writes `outputs/phase8/p85_a5_audit.json` once, only from a clean
commit in which this amendment, the JSON registration and `p85_audit.py` are
committed and unmodified, and records HEAD, the registration, source and
mechanism hashes, Cantera version and the protected-manifest check.

## 2. G1 under A5 (rerun 4, `outputs/phase8/p85_g1_rerun4.json`)

Same network, test values, AE3 TAKE-OFF/APPROACH/IDLE inlets of the frozen v6
rows, take-off design for every mode (P8.5-A4). `B` is read from the
committed audit and recomputed from the mechanism before any network solve;
the two exact values must be equal.

**G1.1 (replaced): temperature convergence toward HP.** No dilution; volume
scales `s = 1e4, 1e5, 1e6`, called `tau, 10tau, 100tau`; one HP-equilibrium
reference per mode from the same inputs. Pass at a mode iff every solve
converged (`all_converged`), the absolute temperature errors
`e(s) = |T_lean_exit(s) − T_eq|` are finite and non-increasing exactly
(`e(10tau) ≤ e(tau)` and `e(100tau) ≤ e(10tau)`, float comparison, no slack),
and `e(100tau) < 0.1 K` strictly. No posterior slack at a numerical floor.
All PSR residence times and outlet states, extinguished flags and integrator
errors of these runs are reported; a shortfall at APPROACH/IDLE is reported
as the kinetic/extinction physics of the network at those residence times.
The old composition criterion (max |ΔY| ≤ 1e-5 over species with
Y_eq > 1e-6) is **reported only**, at every scale. If it fails, no full-state
equilibrium is claimed; G1.1 then shows temperature convergence only.

**G1.2 closure.** A2NOx: whole-network energy, whole-network element, max
mixer energy and max mixer element relative errors (P8.2 denominators,
unchanged) must all be finite and **strictly** `< 10·B` (exact rational
comparison), for the test-value run and the three G1.1 runs at every mode.
CRECK control (separate tolerance): the CRECK audit must pass the equality
rule; its three base-value runs (test values, take-off design; the existing
G1.6 runs) must all converge and all four closures must be strictly
`< 1e-10`. A failing audit or control blocks PASS. `10·B` is never applied
to CRECK.

**G1.3 sigma0.** `σ_φ = 0`. Every primary PSR converged; T relative spread
`(max − min)/max ≤ 1e-12`. For each species, reference `r = max |Y|` over the
primary PSRs and spread `Δ = max Y − min Y`: if `r > 1e-8`, `Δ/r ≤ 1e-12`;
otherwise (including `r = 1e-8` and `r = 0`) `Δ ≤ 1e-12`. No division by a
near-zero value. Species-level evidence (reference, spread, rule, metric) is
retained. This mixed trace rule applies only to sigma0.

**Reported, unchanged.** K = 7 vs 9 quadrature, A2NOx/CRECK thermo
consistency, CRECK mechanism spread and the P8.5-A2 imbalance diagnostic.

**Verdict.** PASS iff the A2NOx audit gives a finite positive `B` without
error and G1.1, G1.2 (A2NOx and CRECK control, including the CRECK audit)
and G1.3 pass at all three modes. Identity drift during the run makes the
record ERROR.

## 3. Run guards and order

The audit and rerun 4 are queued after the main benchmark/A2/Track 4
sequence. Rerun 4 refuses (nothing written) unless: the registration, this
amendment, `p85_audit.py` and `reactor_validation.py` are committed and
unmodified; the audit output is committed, unmodified, and matches the
current registration, audit source and mechanism hashes; `cpp/` and
`scripts/` have no uncommitted changes; the Mac is on AC power; no main
workflow owner is active (any Python process running a `scripts/phase8/`
script or module, the v6 LTO/calibration scripts, or a
`run_benchmark_queue.sh` shell); and the protected manifest matches. Both
outputs are write-once; rerun 4 is selected only with `--a5` and its path is
fixed by the registration (the historical default, the occupied rev2 path,
is unchanged and still refuses). Pending commands, in order:

```
nice -n 15 .venv/bin/python scripts/phase8/p85_audit.py
git add outputs/phase8/p85_a5_audit.json && git commit   # audit record
bash cpp/build.sh                                       # dev module from the committed cpp/
nice -n 15 .venv/bin/python scripts/phase8/reactor_validation.py --a5
```
