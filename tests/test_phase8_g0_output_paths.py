"""Exercise only the G0 path selector, without importing empirical-data code."""
import ast
from pathlib import Path

import pytest


@pytest.fixture
def selector(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/phase8/g0_parity.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "output_paths")
    namespace = {"Path": Path, "ROOT": tmp_path, "OUT_DIR": tmp_path / "outputs/phase8/g0",
                 "OUT_JSON": tmp_path / "outputs/phase8/g0_parity.json"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["output_paths"], namespace


def test_historical_defaults_preserved(selector):
    choose, ns = selector
    assert choose(None) == (ns["OUT_DIR"], ns["OUT_JSON"])


@pytest.mark.parametrize("absolute", [False, True])
def test_new_output_is_isolated_without_creation(selector, absolute):
    choose, ns = selector
    relative = Path("outputs/phase8/g0_rerun_new")
    target = ns["ROOT"] / relative
    assert choose(str(target if absolute else relative)) == (target, target / "g0_parity.json")
    assert not target.exists()


@pytest.mark.parametrize("path", ["outputs/phase8/g0", "outputs/phase7/new", "../outside", "outputs/phase8/../../outside"])
def test_default_or_outside_directory_is_refused(selector, path):
    choose, _ = selector
    with pytest.raises(ValueError): choose(path)


def test_symlink_escape_is_refused(selector, tmp_path):
    choose, ns = selector
    base = ns["ROOT"] / "outputs/phase8"; base.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): choose("outputs/phase8/link/new")
