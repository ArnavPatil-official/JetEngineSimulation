"""P8.5-A5 pure gate and audit-fixture contracts (no mechanism load, no network)."""

import ast
import json
import math
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import p85_audit as a5  # noqa: E402

ELEMENTS = ["C", "H", "N"]
# species: CH2, C2H4, X (one C), Y (one C), Z (one C)
ATOMS = [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
WEIGHTS = [14.027, 28.054, 12.011, 12.011, 12.011]


def rxn(reactants, products, eq="fixture"):
    return {"equation": eq, "reactants": reactants, "products": products}


def fake_audit(reactions):
    return a5.audit_reactions(ELEMENTS, ATOMS, WEIGHTS, reactions)


LUMPED = rxn({0: 2.0}, {1: 0.9999999}, "2 CH2 => 0.9999999 C2H4")   # rounded lumped coefficient
BALANCED = rxn({0: 2.0}, {1: 1.0}, "2 CH2 => C2H4")


# --- source audit ------------------------------------------------------------

def test_audit_exact_defect_and_bound():
    audit = fake_audit([BALANCED, LUMPED])
    B = a5.audit_B(audit)
    assert B == 1 - Fraction(0.9999999)            # |2 nu - 2| / max(2, 2 nu), exact
    assert audit["argmax"]["index"] == 1
    assert audit["n_reactions_with_nonzero_defect"] == 1
    rec = audit["reactions_with_nonzero_defect"][0]
    assert Fraction(rec["signed_atoms_exact"]["C"]) == 2 * Fraction(0.9999999) - 2
    assert Fraction(rec["reactant_turnover"]["C"]) == 2 and "N" not in rec["signed_atoms_exact"]
    assert Fraction(rec["mass_defect_kg_kmol_exact"]) == Fraction(28.054) * Fraction(0.9999999) - 2 * Fraction(14.027)
    assert rec["signed_atoms_float64"]["H"] == 2.0 * -2.0 + 4.0 * 0.9999999   # float64 evaluation
    assert a5.allowance_gate(audit)["pass"]


def test_zero_turnover_pairs_contribute_zero():
    audit = fake_audit([BALANCED])
    assert a5.audit_B(audit) == 0 and audit["errors"] == []
    assert audit["max_by_element"]["N"]["exact"] == "0"
    gate = a5.allowance_gate(audit)
    assert not gate["pass"] and "strictly positive" in gate["reasons"][0]
    with pytest.raises(ValueError):
        a5.closure_tolerance(a5.audit_B(audit))


def test_nonzero_defect_at_zero_turnover_is_an_error():
    audit = fake_audit([LUMPED, rxn({2: -1.0}, {})])     # R_C = -1, P_C = 0: denominator 0, delta 1
    assert audit["errors"] and audit["errors"][0]["index"] == 1
    assert not a5.allowance_gate(audit)["pass"]


def test_tiny_cyclic_defect_is_kept_no_threshold():
    audit = fake_audit([rxn({2: 0.1, 3: 0.2}, {4: 0.3})])
    delta = Fraction(0.3) - Fraction(0.1) - Fraction(0.2)
    assert delta != 0 and a5.audit_B(audit) == abs(delta) / max(Fraction(0.1) + Fraction(0.2), Fraction(0.3))
    assert audit["n_reactions_with_nonzero_defect"] == 1 and audit["threshold"] is None


def test_balanced_control_rule():
    decimal = fake_audit([BALANCED, rxn({2: 0.1, 3: 0.2}, {4: 0.3})])
    assert a5.control_gate(decimal)["pass"]                          # representation-level only
    assert not a5.control_gate(fake_audit([LUMPED]))["pass"]         # 1e-7 defect
    assert not a5.control_gate(fake_audit([rxn({2: -1.0}, {})]))["pass"]   # audit error
    edge = {"errors": [], "B_exact": str(Fraction(1, 2 ** 52))}
    assert a5.control_gate(edge)["pass"]
    edge["B_exact"] = str(Fraction(1, 2 ** 52) + Fraction(1, 2 ** 80))
    assert not a5.control_gate(edge)["pass"]


# --- closure -----------------------------------------------------------------

def test_tolerance_independent_of_observed_residuals():
    B = a5.audit_B(fake_audit([BALANCED, LUMPED]))
    tol = a5.closure_tolerance(B)
    assert tol == 10 * B
    small = {"test_values": {**dict.fromkeys(a5.CLOSURE_METRICS, 1e-12), "all_converged": True}}
    large = {"test_values": {**dict.fromkeys(a5.CLOSURE_METRICS, 3e-6), "all_converged": True}}
    g_small, g_large = a5.closure_gate(small, tol), a5.closure_gate(large, tol)
    assert g_small["tolerance_exact"] == g_large["tolerance_exact"] == str(10 * B)
    assert g_small["pass"] and not g_large["pass"]
    # stoichiometry alone moves the tolerance
    assert a5.closure_tolerance(a5.audit_B(fake_audit([rxn({0: 2.0}, {1: 0.999999})]))) != tol


def test_closure_strict_equality_and_finite():
    tol = a5.closure_tolerance(Fraction(1, 2 ** 30))              # 10 * 2^-30 is a float exactly
    edge = float(tol)
    assert Fraction(edge) == tol
    assert not a5.strictly_below(edge, tol)
    assert a5.strictly_below(math.nextafter(edge, 0.0), tol)
    odd = a5.closure_tolerance(Fraction(1, 3) * Fraction(1, 10 ** 8))   # not a float
    assert a5.strictly_below(float(odd), odd) == (Fraction(float(odd)) < odd)
    for bad in (math.nan, math.inf, None, True):
        assert not a5.strictly_below(bad, tol)
    case = {**dict.fromkeys(a5.CLOSURE_METRICS, 0.0), "all_converged": True}
    assert a5.closure_gate({"c": case}, tol)["pass"]
    assert not a5.closure_gate({"c": {**case, "element_relative": edge}}, tol)["pass"]
    assert not a5.closure_gate({"c": {**case, "max_mixer_energy_relative": math.nan}}, tol)["pass"]
    assert not a5.closure_gate({"c": {k: v for k, v in case.items() if k != "energy_relative"}}, tol)["pass"]
    assert not a5.closure_gate({"c": case}, None)["pass"]


def test_closure_requires_convergence_of_every_case():
    tol = a5.closure_tolerance(Fraction(1, 2 ** 30))
    ok = {**dict.fromkeys(a5.CLOSURE_METRICS, 0.0), "all_converged": True}
    stalled = {**dict.fromkeys(a5.CLOSURE_METRICS, 0.0), "all_converged": False}   # near-initial state
    cases = {"test_values": stalled, "tau": ok, "10tau": ok, "100tau": ok}
    g = a5.closure_gate(cases, tol)
    assert not g["pass"] and not g["cases"]["test_values"]["all_converged"]["pass"]
    assert not a5.closure_gate({"test_values": {k: v for k, v in ok.items() if k != "all_converged"}}, tol)["pass"]
    assert not a5.closure_gate({"test_values": {**ok, "all_converged": 1}}, tol)["pass"]
    assert a5.closure_gate({**cases, "test_values": ok}, tol)["pass"]


def test_creck_control_closure():
    ok = {m: {**dict.fromkeys(a5.CLOSURE_METRICS, 1e-15), "all_converged": True}
          for m in ("TAKE-OFF", "APPROACH", "IDLE")}
    assert a5.control_closure_gate(ok, True)["pass"]
    assert not a5.control_closure_gate(ok, False)["pass"]                      # failing audit blocks
    assert not a5.control_closure_gate({**ok, "IDLE": {**ok["IDLE"], "all_converged": False}}, True)["pass"]
    assert not a5.control_closure_gate({**ok, "IDLE": {**ok["IDLE"], "energy_relative": 1e-10}}, True)["pass"]
    assert a5.control_closure_gate({**ok, "IDLE": {**ok["IDLE"], "energy_relative": 9.9e-11}}, True)["pass"]


# --- sigma0 ------------------------------------------------------------------

def test_sigma0_mixed_trace_rule():
    names = ["major", "edge", "just_above", "trace", "absent"]
    edge_ref = 1e-8
    above = math.nextafter(edge_ref, 1.0)
    Y = [[0.2, edge_ref, above, 1e-12, 0.0],
         [0.2, edge_ref, above, 0.0, 0.0]]
    g = a5.sigma0_gate([1800.0, 1800.0], Y, [True, True], names)
    rules = {s["name"]: s["rule"] for s in g["species"]}
    assert rules == {"major": "relative", "edge": "absolute", "just_above": "relative",
                     "trace": "absolute", "absent": "absolute"}
    assert g["pass"]                                   # trace spread exactly 1e-12 passes
    Y[1][3] = -1e-13                                   # spread 1.1e-12 > 1e-12
    g = a5.sigma0_gate([1800.0, 1800.0], Y, [True, True], names)
    assert not g["pass"] and g["failing_species"] == ["trace"]
    assert {"reference", "spread", "metric"} <= set(g["species"][0])


def test_sigma0_convergence_temperature_and_finite():
    Y = [[0.5, 0.5], [0.5, 0.5]]
    assert not a5.sigma0_gate([1800.0, 1800.0], Y, [True, False], ["a", "b"])["pass"]
    assert not a5.sigma0_gate([1800.0, 1800.0 * (1 + 1e-11)], Y, [True, True], ["a", "b"])["pass"]
    assert not a5.sigma0_gate([1800.0, math.nan], Y, [True, True], ["a", "b"])["pass"]
    assert not a5.sigma0_gate([1800.0, 1800.0], [[0.5, math.nan], [0.5, 0.5]], [True, True], ["a", "b"])["pass"]
    assert a5.sigma0_gate([1800.0, 1800.0], Y, [True, True], ["a", "b"])["pass"]


# --- G1.1 temperature convergence and composition (reported only) -------------

def runs(errors, converged=(True, True, True)):
    return [{"label": lab, "scale": s, "converged": c, "T": e}
            for lab, s, e, c in zip(("tau", "10tau", "100tau"), (1e4, 1e5, 1e6), errors, converged)]


def test_shortfall_labels():
    assert a5.shortfall_label("IDLE", True, 0.04, []) == "none"
    for mode in ("APPROACH", "IDLE"):
        assert a5.shortfall_label(mode, True, 2.5, []).startswith("kinetics/extinction limit")
        for conv, err, dT in ((False, [], 2.5), (True, ["CVODES error -3"], 2.5), (True, [], math.nan)):
            assert a5.shortfall_label(mode, conv, dT, err).startswith("numerical failure")
    assert "not labelled physics" in a5.shortfall_label("TAKE-OFF", True, 2.5, [])
    g = a5.temperature_convergence_gate(
        [{"label": "tau", "scale": 1e4, "converged": True, "T": 2.5},
         {"label": "10tau", "scale": 1e5, "converged": False, "T": 1.0, "errors": ["CVODES"]},
         {"label": "100tau", "scale": 1e6, "converged": True, "T": 0.5}], 0.0, 0.1, "APPROACH")
    assert g["shortfall"]["tau"].startswith("kinetics") and g["shortfall"]["10tau"].startswith("numerical")
    assert not g["pass"]


def test_temperature_convergence_gate():
    assert a5.temperature_convergence_gate(runs([0.5, 0.25, 0.0625]), 0.0)["pass"]
    assert a5.temperature_convergence_gate(runs([0.05, 0.05, 0.05]), 0.0)["pass"]       # ties allowed
    assert not a5.temperature_convergence_gate(runs([0.5, 0.25, 0.1]), 0.0)["pass"]     # strict < 0.1
    up = a5.temperature_convergence_gate(runs([0.01, 0.02, 0.015]), 0.0)
    assert not up["pass"] and not up["monotone_non_increasing"] and up["final_pass"]
    assert not a5.temperature_convergence_gate(runs([0.5, 0.25, 0.0625], (True, False, True)), 0.0)["pass"]
    assert not a5.temperature_convergence_gate(runs([0.5, 0.25, math.nan]), 0.0)["pass"]
    below = a5.temperature_convergence_gate(runs([975.0, 974.0, 973.0]), 975.27)   # T below T_eq
    assert below["abs_dT_K"]["tau"] == pytest.approx(0.27) and not below["monotone_non_increasing"]


def test_composition_is_reported_only():
    names = ["N2", "CO", "trace"]
    rep = a5.composition_report([0.7, 0.01, 1e-7], [0.7, 0.0109, 5e-7], names)
    assert rep["reported_only"] and not rep["criterion_met"] and rep["argmax_species"] == "CO"
    assert "no full-state equilibrium" in rep["claim"]
    g11 = {m: {**a5.temperature_convergence_gate(runs([0.5, 0.25, 0.0625]), 0.0),
               "composition_reported_only": {"100tau": rep}} for m in ("TAKE-OFF", "APPROACH", "IDLE")}
    allowance = {"pass": True}
    g12 = {m: {"pass": True} for m in g11}
    verdict, checks = a5.a5_verdict(allowance, g11, g12, {"pass": True}, {m: {"pass": True} for m in g11})
    assert verdict == "PASS" and checks["G1.1_temperature_convergence_toward_HP"]
    verdict, _ = a5.a5_verdict(allowance, g11, g12, {"pass": False}, {m: {"pass": True} for m in g11})
    assert verdict == "FAIL"
    verdict, _ = a5.a5_verdict({"pass": False}, g11, g12, {"pass": True}, {m: {"pass": True} for m in g11})
    assert verdict == "FAIL"


# --- run guards and registration ---------------------------------------------

QREG = {"id": a5.QUEUE_ID, "lease_path": "outputs/phase8/operations/20261003_recovery/owner.lease.json"}
AC = "Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged\n"
BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 100%; discharging\n"


def chain(status, rid=a5.QUEUE_ID, sha="q"):
    return {"registration_id": rid, "registration_sha256": sha, "session": "s1", "status": status}


def test_workflow_blockers_use_lease_and_terminal_chain_record():
    for status in a5.TERMINAL_CHAIN:
        assert a5.workflow_blockers(QREG, "q", False, [chain(status)]) == []
    assert a5.workflow_blockers(None, None, False, [chain("COMPLETE")])                 # absent here: blocker
    assert any("lease" in b for b in a5.workflow_blockers(QREG, "q", True, [chain("COMPLETE")]))
    assert any("terminal" in b for b in a5.workflow_blockers(QREG, "q", False, []))
    for bad in (chain("ABORTED"), chain(None), chain("COMPLETE", rid="other"),
                chain("COMPLETE", sha="older registration"), None, ["not a record"]):
        assert any("terminal" in b for b in a5.workflow_blockers(QREG, "q", False, [bad]))
    assert a5.workflow_blockers({**QREG, "id": "X"}, "q", False, [chain("COMPLETE")])
    assert a5.workflow_blockers(QREG, "q", False, [chain("ABORTED"), chain("COMPLETE")]) == []


def test_run_blockers_need_ac_and_idle_workflow():
    done = [chain("COMPLETE")]
    assert a5.run_blockers(AC, QREG, "q", False, done) == []
    assert any("AC" in b for b in a5.run_blockers(BATTERY, QREG, "q", False, done))
    assert any("AC" in b for b in a5.run_blockers(None, QREG, "q", False, done))
    assert len(a5.run_blockers(BATTERY, QREG, "q", True, [])) == 3


def test_commit_and_audit_match_blockers():
    paths = ["docs/phase8_p85_a5_registration.json", "outputs/phase8/p85_a5_audit.json"]
    assert a5.commit_blockers(paths, "\n".join(paths) + "\n", "") == []
    assert a5.commit_blockers(paths, paths[0] + "\n", "") == [f"{paths[1]} is not committed"]
    assert a5.commit_blockers(paths, "\n".join(paths), " M docs/phase8_p85_a5_registration.json\n")
    assert a5.commit_blockers(paths, None, "")
    ident = {"registration_sha256": "r", "audit_source_sha256": "s", "a2nox_sha256": "a", "creck_sha256": "c"}
    doc = {"identity": dict(ident), "a2nox": {"B_exact": "1/3"}, "identity_drift": [],
           "allowance_gate": {"pass": True, "reasons": []}, "creck_control_gate": {"pass": True, "reasons": []}}
    assert a5.audit_match_blockers(doc, ident) == []
    assert a5.audit_match_blockers(doc, {**ident, "a2nox_sha256": "changed"})
    assert a5.audit_match_blockers({**doc, "a2nox": {}}, ident)


def test_matching_hash_but_drifted_or_failed_audit_is_rejected():
    ident = {"registration_sha256": "r", "audit_source_sha256": "s", "a2nox_sha256": "a", "creck_sha256": "c"}
    good = {"identity": dict(ident), "a2nox": {"B_exact": "1/3"}, "identity_drift": [],
            "allowance_gate": {"pass": True}, "creck_control_gate": {"pass": True}}
    assert a5.audit_match_blockers(good, ident) == []
    bad = [{**good, "identity_drift": ["git_head"]},
           {**good, "allowance_gate": {"pass": False, "reasons": ["B is not finite and strictly positive"]}},
           {**good, "creck_control_gate": {"pass": False}},
           {**good, "creck_control_gate": {"pass": "true"}}]
    for doc in bad:
        assert len(a5.audit_match_blockers(doc, ident)) == 1
    for key in ("identity_drift", "allowance_gate", "creck_control_gate"):           # missing metadata
        missing = {k: v for k, v in good.items() if k != key}
        assert len(a5.audit_match_blockers(missing, ident)) == 1


def test_registration_matches_module_and_new_paths():
    reg = json.loads(a5.REGISTRATION.read_text())
    assert reg["registered_date"] == "2026-10-03" and reg["user_decisions_date"] == "2026-10-02"
    assert reg["audit"]["closure_tolerance_multiplier"] == a5.CLOSURE_MULTIPLIER
    assert Fraction(reg["audit"]["control_rule"]["max_pair_defect_at_most_float"]) == a5.CONTROL_BOUND
    assert reg["g1"]["G1.2_closure"]["creck_control"]["strictly_below"] == a5.CONTROL_CLOSURE
    assert tuple(reg["g1"]["G1.2_closure"]["metrics"]) == a5.CLOSURE_METRICS
    assert reg["g1"]["G1.1_temperature_convergence"]["volume_scales"] == [1e4, 1e5, 1e6]
    old = reg["known_when_registered"]["historical_records_unchanged"]
    assert reg["outputs"]["g1"] not in old and reg["outputs"]["audit"] not in old
    assert reg["outputs"] == {"audit": "outputs/phase8/p85_a5_audit.json",
                              "g1": "outputs/phase8/p85_g1_rerun4.json"}
    assert reg["known_when_registered"]["a5_audit"].startswith("not computed")
    corr = reg["prospective_corrections"][0]
    assert corr["date"] == "2026-10-03" and corr["before_any_a5_calculation"] is True
    assert reg["g1"]["G1.2_closure"]["a2nox"]["require_all_converged_every_case"] is True
    assert reg["g1"]["G1.1_temperature_convergence"]["shortfall_label_modes"] == list(a5.SHORTFALL_MODES)
    wf = reg["guards"]["main_workflow"]
    assert wf["registration"] == str(a5.QUEUE_REGISTRATION.relative_to(ROOT)) and wf["id"] == a5.QUEUE_ID
    assert wf["terminal_chain_statuses"] == list(a5.TERMINAL_CHAIN)
    assert reg["prospective_corrections"][1]["before_any_a5_calculation"] is True


def test_validator_keeps_historical_default_and_explicit_a5():
    tree = ast.parse(a5.VALIDATOR.read_text())
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "outputs/phase8/p85_g1_rev2.json" in strings          # historical default unchanged
    assert "--a5" in strings and "outputs/phase8/p85_g1_rerun4.json" not in strings   # A5 path from registration
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"main", "main_a5", "a5_blockers"} <= funcs


def test_runtime_uses_strict_shared_validator_and_refuses_missing_evidence(monkeypatch):
    calls = []
    helper = types.ModuleType("ac_workflow")

    def validate(root, **kw):
        calls.append((root, kw))
        raise RuntimeError("completion manifest is missing")

    helper.validate_terminal_context = validate
    monkeypatch.setitem(sys.modules, "ac_workflow", helper)
    monkeypatch.setattr(a5, "cmd", lambda argv: AC)
    assert a5.live_run_blockers() == [
        "main workflow evidence invalid or unavailable: completion manifest is missing"]
    assert calls == [(ROOT, {"require_idle": True})]
    helper.validate_terminal_context = lambda root, **kw: {"registration_id": a5.QUEUE_ID}
    assert a5.live_run_blockers() == []
    monkeypatch.setattr(a5, "cmd", lambda argv: BATTERY)
    assert any("AC" in b for b in a5.live_run_blockers())
    helper.validate_terminal_context = lambda root, **kw: {"registration_id": "foreign"}
    assert any("foreign" in b for b in a5.live_run_blockers())


def test_unreadable_end_identity_and_extra_fields_are_drift():
    def unreadable():
        raise OSError("core disappeared")

    end = a5.capture_identity(unreadable)
    assert "identity_error" in end
    assert a5.identity_differences({"binary": "old"}, end) == ["binary", "identity_error"]
    assert a5.identity_differences({"binary": "same"}, {"binary": "same"}) == []


def test_audit_match_includes_workflow_source_identity():
    current = {"workflow_validator_sha256": "new", "workflow_registration_sha256": "registered"}
    audit = {"identity": dict(current), "a2nox": {"B_exact": "1/3"}, "identity_drift": [],
             "allowance_gate": {"pass": True}, "creck_control_gate": {"pass": True}}
    assert a5.audit_match_blockers(audit, current) == []
    assert a5.audit_match_blockers(audit, {**current, "workflow_validator_sha256": "changed"})


def test_rerun_requires_proven_separate_build_binary(tmp_path, monkeypatch):
    import reactor_validation as rv

    monkeypatch.setattr(rv, "ROOT", tmp_path)
    binary = tmp_path / "cpp/build_next/catjet_core.fixture.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic binary")
    rel = str(binary.relative_to(tmp_path))
    context = {"stages": {"validation_build": {"state": "PASS", "outputs": {rel: a5.sha256(binary)}}}}
    assert rv.validation_binary_blockers(context, binary) == []
    assert rv.validation_binary_blockers(context, None)
    assert rv.validation_binary_blockers(context, tmp_path / "cpp/build/catjet_core.fixture.so")
    assert rv.validation_binary_blockers({"stages": {}}, binary)
    context["stages"]["validation_build"]["outputs"][rel] = None
    assert rv.validation_binary_blockers(context, binary)
    context["stages"]["validation_build"]["outputs"][rel] = "0" * 64
    assert rv.validation_binary_blockers(context, binary)


def test_rerun_identity_detects_source_input_binary_and_power_drift(tmp_path, monkeypatch):
    import reactor_validation as rv

    monkeypatch.setattr(rv, "ROOT", tmp_path)
    monkeypatch.setattr(a5, "git_head", lambda: "a" * 40)
    monkeypatch.setattr(a5, "current_identity", lambda rb, reg: {"registration_sha256": "fixture"})
    paths = ["cpp/core.cpp", "outputs/phase7/calibration_v6_rows.csv",
             "outputs/phase7/p72_registration.json", "outputs/phase8/protected_sha256_phase8.json",
             "outputs/audit.json", "validator.py", "cpp/build_next/core.so"]
    for path in paths:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture")
    monkeypatch.setattr(a5, "VALIDATOR", tmp_path / "validator.py")
    monkeypatch.setattr(rv, "core", types.SimpleNamespace(__file__=str(tmp_path / paths[-1])))
    monkeypatch.setattr(a5, "cmd", lambda argv: "cpp/core.cpp\n" if argv[0] == "git" else AC)
    reg = {"outputs": {"audit": "outputs/audit.json"}}
    start = rv.a5_identity(b"fixture", reg)
    for path, key in [(paths[0], "cpp_sources_sha256"), (paths[1], "inputs_sha256"),
                      (paths[-1], "module_sha256")]:
        p = tmp_path / path
        p.write_bytes(b"changed")
        assert key in a5.identity_differences(start, rv.a5_identity(b"fixture", reg))
        p.write_bytes(b"fixture")
    monkeypatch.setattr(a5, "cmd", lambda argv: "cpp/core.cpp\n" if argv[0] == "git" else BATTERY)
    assert "on_ac" in a5.identity_differences(start, rv.a5_identity(b"fixture", reg))


def test_rerun_refuses_malformed_audit_before_any_network(tmp_path, monkeypatch):
    import reactor_validation as rv

    audit = tmp_path / "audit.json"
    audit.write_text("[]")
    monkeypatch.setattr(rv, "ROOT", tmp_path)
    monkeypatch.setattr(rv, "core", types.SimpleNamespace(ReactorNetwork=object, __file__="fixture.so"))
    monkeypatch.setattr(a5, "git_commit_blockers", lambda paths: [])
    monkeypatch.setattr(a5, "cmd", lambda argv: "")
    monkeypatch.setattr(a5, "git_head", lambda: "a" * 40)
    monkeypatch.setattr(a5, "live_run_blockers", lambda: [])
    monkeypatch.setattr(a5, "strict_workflow_context", lambda: {})
    monkeypatch.setattr(rv, "validation_binary_blockers", lambda context, path: [])
    monkeypatch.setattr(rv, "protected_check", lambda: {"mismatches": []})
    blockers, _, _ = rv.a5_blockers({"outputs": {"audit": "audit.json"}}, b"fixture")
    assert "audit output is unreadable or malformed" in blockers
