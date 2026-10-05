"""
pytest entry point for pyCycle's own HBTF regression test (upstream, unmodified).

Upstream names the test `benchmark_case1` in `benchmark_hbtf.py` (a unittest
TestCase meant for `testflo -b`). pytest only collects unittest methods whose
name starts with `test`, so this shim subclasses the upstream TestCase and
calls the upstream method unchanged. The upstream setUp, run_model call,
reference numbers and tolerance are all used as published; nothing is copied.

Run inside the catjet-pycycle env only (never .venv):
    ~/miniforge3/envs/catjet-pycycle/bin/python -m pytest \
        scripts/phase8/pycycle/upstream_benchmark_pytest_shim.py -q -p no:cacheprovider
The file name does not match `test_*.py`, so the repo's own pytest run never
collects it. run_hbtf_reference.py runs it from a temporary directory so the
OpenMDAO report folders do not land in the repo.
"""

import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[3] / "envs" / "pycycle" / "upstream"
sys.path.insert(0, str(UPSTREAM))

import example_cycles.tests.benchmark_hbtf as _upstream  # noqa: E402


class TestUpstreamHBTFBenchmark(_upstream.HBTFTestCase):
    """Collects upstream HBTFTestCase.benchmark_case1 under a pytest-visible name."""

    def test_benchmark_case1(self):
        self.benchmark_case1()
