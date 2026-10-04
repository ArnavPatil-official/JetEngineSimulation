"""Input-only pure fixtures and an explicitly gated actual 20-family assertion."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/phase8'))
import p84_input_convergence as IC
import public_engine_inputs as P
import trent_p84c as A


def good_design():
    return {'converged': True, 'reason': '', 'iterations': 3, 'x': [100, .03, 3, 2, 5],
            'scalars': {k: 1 for k in IC.POSITIVE_SCALARS}, 'residuals': [0]*5,
            'stations': {'inlet': {'W': 100, 'Tt': 288, 'Pt': 101325}},
            'mass_closure': 0, 'energy_closure': 0, 'element_closure': 0, 'extrapolated_maps': []}


def synthetic_reps():
    table = P.architecture_table(A.REG)
    families = {name: {'uids': [f'{i:02}b', f'{i:02}a'], 'geared': a['geared']}
                for i, (name, a) in enumerate(sorted(table.items()))}
    split = {'families': families, 'n_eligible_families': 20}
    return P.select_representatives(split, A.REG)


def test_nine_starts_order_and_shaft_dimensions():
    starts = IC.design_starts(220, 3, A.REG)
    assert len(starts) == 9
    assert starts[:3] == [[220000/280, f, 3.5, 2, 5] for f in (.03, .02, .04)]
    assert starts[3][0] == 1000
    assert len(IC.design_starts(220, 2, A.REG)[0]) == 4


def test_first_valid_start_selected_not_best_fuel():
    good = good_design()
    invalid = {**good, 'mass_closure': 1}
    responses = iter([invalid, good])
    eng = SimpleNamespace(solve_design=lambda start: next(responses))
    selected, tries = IC.solve_design_ordered(eng, 220, 3, A.REG)
    assert selected is good and len(tries) == 2
    assert not tries[0]['closure_valid'] and tries[1]['closure_valid']


def test_all_failed_starts_retained_and_no_success_inferred():
    eng = SimpleNamespace(solve_design=lambda start: {'converged': False, 'reason': 'failed', 'x': []})
    selected, tries = IC.solve_design_ordered(eng, 220, 2, A.REG)
    assert selected is None and len(tries) == 9
    assert all(not t['closure_valid'] for t in tries)


def test_default_old_input_builder_keeps_zero_flow_override(monkeypatch):
    specs = []
    class Engine:
        spec = SimpleNamespace()
    def build(*args):
        eng = Engine()
        eng.spec = SimpleNamespace(design_W_max_kg_s=0)
        specs.append(eng.spec)
        return eng
    B = SimpleNamespace(make_engine=build, CENTRAL=A.engine_inputs(A.central()))
    monkeypatch.setattr(A, '_p84b', lambda: B)
    old = A.make_three_shaft(40, 10, 300, A.old_a4_inputs(), use_flow_override=False)
    new = A.make_three_shaft(40, 10, 300, A.engine_inputs(A.central()))
    assert old.spec.design_W_max_kg_s == 0
    assert new.spec.design_W_max_kg_s == A.design_flow_bound(300)


def test_exact_20_representatives_plus_both_original_records():
    reps = synthetic_reps()
    assert len(reps) == 20 and all(r['uid'].endswith('a') for r in reps)
    values = {'opr': 40, 'bpr': 10, 'rated_kN': 300, 'identification': 'fixture'}
    cases = P.build_cases(reps, {r['uid']: values for r in reps},
                          {u: values for u in P.TRENT_1000_E}, A.REG)
    assert len(cases) == 22 and {c['uid'] for c in cases} >= set(P.TRENT_1000_E)
    assert {c['family'] for c in cases} == set(P.architecture_table(A.REG))


def test_missing_family_and_missing_binary_fail_gate():
    records = [{'family': r['family'], 'case': r['uid'], 'status': 'converged'} for r in synthetic_reps()]
    assert IC.gate_verdict(records, 20)['verdict'] == 'PASS'
    assert IC.gate_verdict(records[:-1], 20)['verdict'] == 'FAIL'
    records[0]['status'] = 'error'
    assert IC.gate_verdict(records, 20)['verdict'] == 'FAIL'


def test_track4_target_is_heavy_and_unknown_states_refuse():
    assert IC.is_heavy_target('scripts/phase8/pinn_diagnostics/run_diagnostics.py --dispatch-registered')
    assert IC.runtime_blockers(None, None, 1, None, None)
    parked = '2 /bin/bash scripts/phase8/pinn_diagnostics/run_diagnostics.py'
    assert IC.heavy_processes(parked, 1) == []
    python = '2 /tmp/python scripts/phase8/pinn_diagnostics/run_diagnostics.py'
    assert IC.heavy_processes(python, 1)


def test_converged_flag_alone_does_not_satisfy_closure():
    assert not IC.closure_problems(good_design())
    bad = good_design()
    bad['stations'] = {}
    assert 'no station states reported' in IC.closure_problems(bad)
    bad = good_design()
    bad['scalars']['W'] = float('nan')
    assert IC.closure_problems(bad)


@pytest.mark.parametrize("corruption", ["residual_nan", "residual_large", "iterations", "dimension", "negative_closure"])
def test_design_flag_cannot_override_solver_contract(corruption):
    bad = good_design()
    if corruption == "residual_nan":
        bad["residuals"][0] = float("nan")
    elif corruption == "residual_large":
        bad["residuals"][0] = 1e-8
    elif corruption == "iterations":
        bad["iterations"] = 51
    elif corruption == "negative_closure":
        bad["mass_closure"] = -1
    else:
        bad["residuals"] = [0]
    assert IC.closure_problems(bad)


def test_public_input_columns_never_include_empirical_targets():
    assert not any('fuel' in c.lower() or 'emission' in c.lower() for c in P.ICAO_CSV_INPUT_COLUMNS)
    assert not any('fuel' in c.lower() or 'emission' in c.lower() for c in P.DATABANK_INPUT_COLUMNS)


def test_all20_eligible_families():
    """Actual solves: missing AC/build/records FAIL explicitly; no family is skipped.

    The registered full_pytest child may run this inside the main chain;
    standalone use needs the validated terminal idle context and committed
    build provenance. Battery fixtures exclude this test.
    """
    context = IC.main_context(allow_full_pytest_child=True)
    active_child = context["lease"] is not None
    blockers = IC.live_gate(A.REG, allow_full_pytest_child=active_child)
    assert not blockers, '\n'.join(blockers)
    IC.check_registered_hashes(A.REG)
    if not IC.OUT.exists():
        assert IC.main([], allow_full_pytest_child=active_child) == 0
    doc = json.loads(IC.OUT.read_text())
    IC.validate_convergence_record(doc, A.REG)
    IC.assert_same_identity(doc['start_identity'], doc['end_identity'])
    assert IC.scientific_identity(doc['start_identity']) == IC.scientific_identity(IC.identity(A.REG))
    assert doc['coverage'] == 20 and doc['verdict'] == 'PASS'
    assert all(c['status'] == 'converged' for c in doc['cases'])
    assert set(P.TRENT_1000_E) <= {c['uid'] for c in doc['cases']}
    assert A._p84b().core.HbtfSpec().design_W_max_kg_s == 0
