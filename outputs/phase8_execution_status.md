# Phase 8 execution status

Plan: `docs/plan.md` (Phase 8). Registration: `docs/phase8_registration.md`.
Branch `phase8`, from the Phase 7 freeze `7524a7a`.
Scope authorized now: the STS slice (D1): P8.0, P8.1, P8.2, P8.3 and the
first P8.6 database sources.

**Phase 8 is NOT complete.** The table says what is done, running or not started.

Test floor at the freeze `7524a7a`: 231 passed, 1 skipped (the orphan-output failure seen at
`8c1d502` predates the freeze and was fixed in `7524a7a`).

| Step | Commit | State |
|---|---|---|
| P8.0 protected list (184 files) + registration | (this commit) | registered |
| P8.0 cProfile of Python v6 `run_at_thrust` | — | not started |
| P8.1 toolchain | — | not started; needs Arnav's OK to install Miniforge/cmake |
| P8.1 C++ port, G0, benchmark | — | not started |
| P8.2, P8.3 | — | not started |
| P8.6 database | — | not started |
