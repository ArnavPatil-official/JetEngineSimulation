"""Pure A4c boundary fixtures; no compiled core, fit, targets, or engine solve."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/phase8'))
import trent_p84c as A
import a4c_profile as P
import a4_error_budget as E
import p84_input_convergence as IC


def fake_identity():
    return {'git_head': 'a' * 40, 'sha256': {'source.py': 'b' * 64},
            'core_module_sha256': 'c' * 64, 'build_provenance_sha256': 'd' * 64,
            'dependencies_sha256': {}}


def calibration_fixture(tmp_path):
    models = [f'M{i}' for i in range(9)]
    ids = [f'C{i:02}' for i in range(31)]
    rows = []
    for i, uid in enumerate(ids):
        for mode in A.v5.MODES:
            row = {k: 1.0 for k in A.REG['calibration_data']['whitelist_columns']}
            row.update({'Unique ID': uid, 'Model': models[i % 9], 'Mode': mode, 'Group': i % 9,
                        'w': 1 / 93, 'Fuel Flow (kg/s)': 1.0, 'old_pred_ff': 'poison ignored'})
            rows.append(row)
    path = tmp_path / 'cal.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    split = {'calibration_records': ids, 'calibration_models': models,
             'calibration_groups': [[m] for m in models], 'heldout_records': ['H0'],
             'heldout_models': ['HMODEL']}
    sp = tmp_path / 'split.json'
    sp.write_text(json.dumps(split))
    return path, sp


def test_shared_bounds_and_no_engine_specific_knobs():
    assert len(A.ORDER) == 5
    assert A.midpoint()['FPR'] == 1.475
    assert A.engine_inputs(A.central())['Cv_core'] == .985
    assert A.old_a4_inputs()['Cv_core'] == .95
    with pytest.raises(ValueError):
        A.engine_inputs({**A.central(), 'engine_year': 2000})
    with pytest.raises(ValueError):
        A.check_shared({**A.central(), 'T4_K': float('nan')})
    assert A.design_flow_bound(500) == max(3000 * .45359237, 4 * 500000 / 220)


def test_calibration_reader_ignores_predictions_and_never_calls_target_reader(tmp_path, monkeypatch):
    path, split = calibration_fixture(tmp_path)
    monkeypatch.setattr(A.v5, 'load_rows', lambda *a, **k: pytest.fail('target reader invoked'))
    rows = A.read_calibration_rows(path, split, check_hash=False)
    assert len(rows) == 93 and 'old_pred_ff' not in rows
    df = pd.read_csv(path)
    df.loc[0, 'Unique ID'] = 'H0'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='held-out record'):
        A.read_calibration_rows(path, split, check_hash=False)


def test_candidate_tie_and_single_fallback_rule():
    candidates = [{'name': k, 'sse': 2} for k in reversed(A.CANDIDATE_ORDER)]
    assert A.select_candidate(candidates)['name'] == A.CANDIDATE_ORDER[0]
    verdicts = {k: {'IDENTIFIED': k == 'T4_K'} for k in A.ORDER}
    plan = A.fallback_plan({'free': list(A.ORDER), 'verdicts': verdicts})
    assert plan['remaining'] == ['T4_K']
    assert plan['fixed'] == {k: A.CENTRAL[k] for k in A.ORDER if k != 'T4_K'}


def test_singleton_profile_direct_evaluations_and_guard():
    class Objective:
        fixed_fit = {}
        calls = 0
        def residuals(self, free, x):
            assert free == [] and x == []
            self.calls += 1
            return np.array([(self.fixed_fit['T4_K'] - 1800) / 100, .1])
    obj = Objective()
    prof = P.profile(obj, ['T4_K'], {'params': A.central(), 'sse': .01}, A.BOUNDS,
                     17, 40, 27, lambda *a: pytest.fail('empty optimizer invoked'))
    table = prof['table']
    assert obj.calls == 17 and len(table) == 17
    assert (table['nfev'] == 1).all() and (table['status'] == 1).all()
    assert not table['inner_optimizer'].any() and obj.fixed_fit == {}
    table['n_unreachable'] = 0
    clean = P.guarded(table, prof['verdicts'], ['T4_K'])
    assert clean['A1'] == 'PASS'
    table.loc[0, 'n_unreachable'] = np.nan
    assert P.guarded(table, prof['verdicts'], ['T4_K'])['A1'] == 'FAIL'


def test_oat_failed_nonempty_design_is_flagged():
    res = {'design': {'converged': False, 'reason': 'failure'}, 'modes': {
        m: {'status': 'unreachable', 'reason': 'failure'} for m in E.MODES}}
    assert E.summarise('bad', res, {'obs_ff': {m: 1 for m in E.MODES}}, None)['design_status'] == 'unreachable'
    assert {'Cv_core', 'Cv_bypass', 'Cv_joint'} <= {c['parameter'] for c in E.oat_cases()}


def test_identity_drift_invalidates_even_when_head_changes_only_for_artifacts():
    before = fake_identity()
    after = copy.deepcopy(before)
    after['git_head'] = 'e' * 40
    IC.assert_same_identity(before, after)
    after['core_module_sha256'] = 'f' * 64
    with pytest.raises(RuntimeError, match='drift'):
        IC.assert_same_identity(before, after)


def test_main_context_uses_exact_record_validator_and_child_policy(monkeypatch):
    calls = []
    def validate(root, registration, **kwargs):
        calls.append(kwargs)
        return {'stages': {'a2_calibration': {'state': 'PASS'}, 'validation_build': {'state': 'PASS'}}}
    monkeypatch.setitem(sys.modules, 'ac_workflow', SimpleNamespace(validate_terminal_context=validate))
    IC.main_context()
    IC.main_context(allow_full_pytest_child=True)
    assert calls == [{'require_idle': True, 'allow_active_stage': None},
                     {'require_idle': False, 'allow_active_stage': 'full_pytest'}]
    monkeypatch.setitem(sys.modules, 'ac_workflow', SimpleNamespace(
        validate_terminal_context=lambda *a, **k: {'stages': {'a2_calibration': {'state': 'FAIL'}}}))
    with pytest.raises(RuntimeError, match='PASS evidence'):
        IC.main_context()


def test_build_binary_mismatch_refuses_before_export(tmp_path, monkeypatch):
    monkeypatch.setattr(IC, 'ROOT', tmp_path)
    core = tmp_path / 'cpp/build_next/core.so'
    core.parent.mkdir(parents=True)
    core.write_bytes(b'fixture binary')
    ctx = {'identity': {'tracked_tree_sha256': 'tree'}, 'stages': {'validation_build': {
        'state': 'PASS', 'exit_code': 0, 'identity': {'git_head': 'a'*40, 'tracked_tree_sha256': 'tree'},
        'launch': {'went': True, 'waited': True, 'exit_code': 0, 'log_sha256': 'x'},
        'outputs': {'cpp/build_next/core.so': 'foreign'}}}}
    with pytest.raises(RuntimeError, match='binary differs'):
        IC.ensure_build_provenance(ctx, core, export=True)
    assert not (tmp_path / IC.BUILD_RECORD).exists()


def test_score_refuses_foreign_evidence_before_any_target_or_reservation(tmp_path):
    def bad_frozen():
        raise RuntimeError('source drift')
    result = A.run_score(1, out_dir=tmp_path, frozen=bad_frozen,
                         read_targets=lambda: pytest.fail('target read before validation'))
    assert result == 3 and not (tmp_path / A.RESERVATION).exists()


def test_score_gate_blocks_before_target_read(tmp_path):
    fm = {'params': A.central(), 'required_records': ['missing.csv']}
    rc = A.run_score(1, out_dir=tmp_path, frozen=lambda: fm, gate=lambda *a: ['missing evidence'],
                     read_targets=lambda: pytest.fail('target read despite gate'))
    assert rc == 3 and not (tmp_path / A.RESERVATION).exists()


def test_exclusive_reservation_is_retained_after_target_reader_failure(tmp_path, monkeypatch):
    fm = {'params': A.central(), 'required_records': [], 'A1_primary': 'FAIL'}
    monkeypatch.setattr(IC, 'check_registered_hashes', lambda *a: None)
    monkeypatch.setattr(IC, 'identity', lambda *a: fake_identity())
    monkeypatch.setattr(IC, 'validate_convergence_record', lambda *a: None)
    monkeypatch.setattr(A, '_load', lambda *a: {'start_identity': fake_identity(),
                                             'end_identity': fake_identity(), 'verdict': 'FAIL'})
    def fail_reader():
        assert (tmp_path / A.RESERVATION).exists()
        raise RuntimeError('synthetic reader failure')
    assert A.run_score(1, out_dir=tmp_path, frozen=lambda: fm, gate=lambda *a: [], read_targets=fail_reader) == 1
    assert (tmp_path / A.RESERVATION).exists()
    assert json.loads((tmp_path / A.SCORE_FILES[2]).read_text())['status'] == 'ERROR'
    assert A.run_score(1, out_dir=tmp_path, read_targets=lambda: pytest.fail('second read')) == 2


def test_record_missing_csv_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'ROOT', tmp_path)
    monkeypatch.setattr(IC, 'ROOT', tmp_path)
    monkeypatch.setattr(A, 'OUT_DIR', tmp_path / 'out')
    A.OUT_DIR.mkdir()
    doc = {'status': 'COMPLETE', 'start_identity': fake_identity(), 'end_identity': fake_identity(),
           'artifacts_sha256': {}}
    (A.OUT_DIR / 'fit_primary.json').write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match='incomplete artifact'):
        A.validate_record('out/fit_primary.json', fake_identity(), [])


import subprocess
import tempfile
import unittest

class DesignFlowScopeContract(unittest.TestCase):
    def test_public_field_defaults_to_zero_and_has_a_binding(self):
        header = (ROOT / "cpp/catjet_core/offdesign.hpp").read_text()
        binding = (ROOT / "cpp/bindings/catjet_core.cpp").read_text()
        self.assertRegex(header, r"double\s+design_W_max_kg_s\s*=\s*0\.0\s*;")
        self.assertIn('.def_readwrite("design_W_max_kg_s", &HbtfSpec::design_W_max_kg_s)', binding)

    def test_opt_in_field_is_used_only_in_the_design_solver(self):
        source = (ROOT / "cpp/catjet_core/offdesign.cpp").read_text()
        before, design_and_after = source.split("SolveResult Hbtf::solve_design", 1)
        design, after = design_and_after.split("SolveResult Hbtf::solve_offdesign", 1)
        self.assertNotIn("design_W_max_kg_s", before)
        self.assertIn("spec.design_W_max_kg_s", design)
        self.assertNotIn("design_W_max_kg_s", after)


class BuildDirectoryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = ROOT / "cpp/build.sh"
        # Exercise the actual argument parser, stopping before dependencies or CMake.
        cls.parser = cls.script.read_text().split("export DEVELOPER_DIR=", 1)[0]
        cls.parser += '\nprintf "%s\\n" "$BUILD_DIR"\n'

    def parse(self, *args):
        with tempfile.TemporaryDirectory(prefix="p84c-build-option-") as cwd:
            return subprocess.run(
                ["bash", "-c", self.parser, str(self.script), *args],
                cwd=cwd, capture_output=True, text=True, check=False,
            )

    def test_default_directory_is_unchanged_from_another_working_directory(self):
        result = self.parse()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), ROOT / "cpp/build")

    def test_registered_relative_directory_is_repository_relative(self):
        result = self.parse("--build-dir", "cpp/build_next")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), ROOT / "cpp/build_next")

    def test_absolute_directory_with_spaces_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="p84c-build-option-") as base:
            target = Path(base) / "validation build"
            result = self.parse("--build-dir", str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), target)

    def test_invalid_options_refuse_before_dependencies(self):
        for args in (("--build-dir",), ("--unknown", "cpp/build_next"),
                     ("--build-dir", ""), ("--build-dir", "cpp/build_next", "extra")):
            with self.subTest(args=args):
                result = self.parse(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Usage:", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(self.script)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)




def profile_record_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'ROOT', tmp_path)
    monkeypatch.setattr(IC, 'ROOT', tmp_path)
    monkeypatch.setattr(A, 'OUT_DIR', tmp_path / 'out')
    A.OUT_DIR.mkdir()
    fit = {'free': ['T4_K'], 'fixed': {k: A.CENTRAL[k] for k in A.ORDER if k != 'T4_K'}, 'sse': 1.0}
    fit_path = A.OUT_DIR / 'fit_primary.json'
    fit_path.write_text(json.dumps(fit))
    grid = np.linspace(*A.BOUNDS['T4_K'], 17)
    table = pd.DataFrame({'param': 'T4_K', 'i': range(17), 'value': grid,
                          'sse': 1 + ((grid-1800)/50)**2, 'nfev': 1, 'status': 1,
                          'n_unreachable': 0})
    calc = P.profile_statistics(table, ['T4_K'], 1.0, A.BOUNDS, 27)
    table = calc.pop('table')
    raw = calc.pop('verdicts')
    doc = {'status': 'COMPLETE', **calc, **P.guarded(table, raw, ['T4_K']),
           'raw_verdicts': raw, **fit, 'fit': 'out/fit_primary.json',
           'threshold': A.REG['profile']['threshold'], 'grid_points': 17, 'inner_max_nfev': 40}
    ident = fake_identity()
    ident['dependencies_sha256'] = {'out/fit_primary.json': IC.sha256(fit_path)}
    doc.update(start_identity=ident, end_identity=copy.deepcopy(ident))
    csv = A.OUT_DIR / 'profile_primary.csv'
    log = A.OUT_DIR / 'profile_primary_progress.log'
    table.to_csv(csv, index=False)
    log.write_text(''.join(f'T4_K i={i} value={grid[i]}\n' for i in range(17)))
    jp = A.OUT_DIR / 'profile_primary.json'
    def save():
        doc['artifacts_sha256'] = {str(p.relative_to(tmp_path)): IC.sha256(p) for p in (csv, log)}
        jp.write_text(json.dumps(doc))
    save()
    return doc, table, csv, save


@pytest.mark.parametrize('corruption', ['D', 'raw_verdict', 'foreign_fit', 'negative_count'])
def test_profile_semantic_evidence_rejected_even_with_matching_artifact_hashes(tmp_path, monkeypatch, corruption):
    doc, table, csv, save = profile_record_fixture(tmp_path, monkeypatch)
    current = fake_identity()
    assert A.validate_record('out/profile_primary.json', current, ['out/fit_primary.json'])['A1'] == 'PASS'
    if corruption == 'D':
        table.loc[0, 'D'] += 1
        table.to_csv(csv, index=False)
    elif corruption == 'raw_verdict':
        doc['raw_verdicts']['T4_K']['IDENTIFIED'] = False
    elif corruption == 'foreign_fit':
        doc['fit'] = 'out/foreign_fit.json'
    else:
        table.loc[0, 'n_unreachable'] = -1
        table.to_csv(csv, index=False)
    save()
    with pytest.raises(RuntimeError):
        A.validate_record('out/profile_primary.json', current, ['out/fit_primary.json'])


def test_all_fixed_fit_empty_evaluation_csv_is_valid(tmp_path, monkeypatch):
    path, split = calibration_fixture(tmp_path)
    expected = A.read_calibration_rows(path, split, check_hash=False)
    monkeypatch.setattr(A, 'read_calibration_rows', lambda: expected)
    monkeypatch.setattr(A, 'ROOT', tmp_path)
    monkeypatch.setattr(IC, 'ROOT', tmp_path)
    monkeypatch.setattr(A, 'OUT_DIR', tmp_path / 'out')
    A.OUT_DIR.mkdir()
    paths = [tmp_path / p for p in A.artifact_paths('fit', 'fallback')]
    pd.DataFrame(columns=[*A.ORDER, 'sse', 'n_unreachable']).to_csv(paths[1], index=False)
    expected.assign(status='converged', ff=1.0).to_csv(paths[2], index=False)
    doc = {'status': 'COMPLETE', 'start_identity': fake_identity(), 'end_identity': fake_identity(),
           'params': A.central(), 'free': [], 'fixed': A.central(), 'sse': 0., 'n_rows': 93,
           'n_evaluations': 0, 'artifacts_sha256': {str(p.relative_to(tmp_path)): IC.sha256(p) for p in paths[1:]}}
    paths[0].write_text(json.dumps(doc))
    assert A.validate_record('out/fit_fallback.json', fake_identity(), [])['free'] == []
