"""
Phase 8: the v6 model (lto_v5.V5Model) with a selectable cycle backend.

backend="python" is lto_v5.V5Model unchanged (arm 1 / A0). backend="cpp"
runs the same lto_v5.solve_task in each worker, with the worker engine
replaced by simulation.catjet_backend.CppEngine (the C++ core). Everything
else - rows, mode states, warm-start guesses, NOx held-out exclusion, the
scoring functions - is the protected v6 code.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "scripts" / "optimization"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import lto_v5 as v5  # noqa: E402


def init_worker_cpp(nox_fit_exclude_models):
    """Worker initialiser: lto_v5._ENGINE becomes the C++ engine (same NOx exclusion)."""
    import logging
    logging.getLogger("cantera").setLevel(logging.ERROR)
    for p in (ROOT, ROOT / "scripts" / "optimization"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(ROOT)
    from simulation.catjet_backend import CppEngine
    with contextlib.redirect_stdout(io.StringIO()):
        v5._ENGINE = CppEngine(nox_fit_exclude_models=set(nox_fit_exclude_models))
    if v5._ENGINE.emissions.nox_fit_exclude_models != set(nox_fit_exclude_models):
        raise RuntimeError("NOx held-out exclusion not applied")


class V6Model(v5.V5Model):
    """V5Model with backend selection; backend='python' is the original class."""

    def __init__(self, fixed: dict, nox_fit_exclude_models, n_workers: int = 6, fuel="Jet-A1",
                 backend: str = "python"):
        if backend == "python":
            super().__init__(fixed, nox_fit_exclude_models, n_workers=n_workers, fuel=fuel)
        elif backend == "cpp":
            super().__init__(fixed, nox_fit_exclude_models, n_workers=1, fuel=fuel)
            self.pool.shutdown()            # replace the Python-engine pool
            self.pool = ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker_cpp,
                                            initargs=(self.nox_fit_exclude_models,))
        else:
            raise ValueError(f"backend must be 'python' or 'cpp', got {backend!r}")
        self.backend = backend


def make_model_v6(backend: str, n_workers: int = 6) -> V6Model:
    """The P7.2 v6 model (registered fuel, fixed values, NOx exclusion) on the chosen backend."""
    import lto_v6
    reg6 = lto_v6.load_registration_v6()
    split = v5.load_split()
    return V6Model(reg6["fixed_central"], nox_fit_exclude_models=split["heldout_models"],
                   n_workers=n_workers, fuel=lto_v6.fuel_composition(reg6), backend=backend)
