"""Focused regression tests for omni_bench_system_adapters. Fakes below only stand in for tasks.py to test WIRING and MAPPING; they never produce reported results.
The real-pipeline test runs only when the real tasks.py is importable (otherwise it is SKIPPED with the exact blocker). Run: python -m unittest test_omni_bench_system_adapters"""
import ast
import copy
import os
import tempfile
import types
import unittest

try:  # package use (python -m unittest app.services.module1_compliance.test_omni_bench_system_adapters) or flat-directory use
    from . import omni_bench_system_adapters as A
    from .omni_bench_adversarial_eval import (NOT_MEASURED, _ALLOWED_PRED_KEYS, compare_systems, evaluate_adversarial_predictions)
    from .omni_bench_adversarial_generator import COMP, INS, NC, generate_adversarial_suite
except ImportError:
    import omni_bench_system_adapters as A
    from omni_bench_adversarial_eval import (NOT_MEASURED, _ALLOWED_PRED_KEYS, compare_systems, evaluate_adversarial_predictions)
    from omni_bench_adversarial_generator import COMP, INS, NC, generate_adversarial_suite

RECORDS = [r.to_dict() for r in generate_adversarial_suite()[:12]]  # fixed leading slice of the real generator output (no selection)
GT_KEYS = ("expected_decision", "expected_escalation", "expected_evidence_roles", "expected_contradiction_behavior", "outcome_change")


def _ref(eid, fn): return {"evidence_id": eid, "filename": fn, "document_id": f"doc_{fn}"}


def _result(outcome="SATISFIED", status="VERIFIED", contra=False, sup=("d1.txt",), con=(), names=None):
    return {"verdict": outcome, "outcome": outcome, "verification_status": status, "contradiction_predicted": contra,
            "supporting_evidence": [_ref(f"ev_{f}", f) for f in sup], "contradicting_evidence": list(con),
            "filename_to_doc_id": names or {"d1.txt": "d1", "d2.txt": "d2"}}


class FakeTasks:
    """Stands in for tasks.py: one Decision per record; records every call so tests can see what was invoked."""
    def __init__(self, outcome="SATISFIED", n_decisions=1, v1_text="NON_COMPLIANT because d1 contradicts d2", fail_on=None):
        self.outcome, self.n, self.v1_text, self.fail_on, self.calls = outcome, n_decisions, v1_text, fail_on, []

    def build_evidence_graph(self, paths, rb):
        self.calls.append(("graph", [os.path.basename(p) for p in paths], os.path.basename(rb)))
        assert all(os.path.isfile(p) for p in paths) and os.path.isfile(rb)
        if self.fail_on and self.fail_on in [os.path.basename(p) for p in paths]: raise RuntimeError("scripted failure")
        return {"first": os.path.basename(paths[0]), "n": self.n}

    def _nodes_of_type(self, G, t): return [(f"dec{i}", {}) for i in range(G["n"])]

    def self_verification_variant_predictions(self, G, d):
        sup = [_ref("ev1", G["first"]), _ref("evR", A.RULEBOOK_FILENAME)]
        return {A.COMPLETE_VARIANT: {"outcome": self.outcome, "verification_status": "VERIFIED", "contradiction_predicted": False, "supporting_evidence": sup}}

    def verify_policy_applicability(self, G, d): return {"contradicting_evidence": []}
    def _rulebook_text_only(self, rb):
        with open(rb, encoding="utf-8") as fh: return fh.read()
    def _build_v1_payload(self, paths, rb_text, G): return "PAYLOAD"

    def analyze_compliance_with_matrix(self, payload, rb_text, objective, is_v2=False):
        assert is_v2 is False and objective == A.OMNI_BENCH_OBJECTIVE
        self.calls.append(("v1", payload))
        return self.v1_text


class SpyCompare:
    def __init__(self): self.args = None
    def __call__(self, records, v1, complete, **kw):
        self.args = (copy.deepcopy(list(records)), v1, complete)
        return compare_systems(records, v1, complete, **kw)


class MappingTests(unittest.TestCase):
    def test_vocabulary_mapping_and_priority(self):
        f = lambda *o: A.complete_prediction_from_results(RECORDS[0], [_result(x) for x in o])["predicted_decision"]
        self.assertEqual((f("SATISFIED"), f("VIOLATION"), f("NO_CONCLUSION"), f("ESCALATE")), (COMP, NC, INS, INS))
        self.assertEqual((f("SATISFIED", "NO_CONCLUSION"), f("SATISFIED", "VIOLATION", "ESCALATE")), (INS, NC))

    def test_escalation_only_from_existing_verification_status(self):
        p = A.complete_prediction_from_results(RECORDS[0], [_result("ESCALATE", status="ESCALATE")])
        q = A.complete_prediction_from_results(RECORDS[0], [_result("NO_CONCLUSION", status="FAILED")])
        self.assertEqual((p["predicted_escalation"], q["predicted_escalation"]), (True, False))

    def test_evidence_roles_use_doc_ids_and_never_cite_the_rulebook_or_unknown_files(self):
        con = [{"source": "CONTRADICTS_edge", "documents": [{"filename": "d2.txt"}, {"filename": A.RULEBOOK_FILENAME}]},
               {"source": "contradiction_finding", "claim_a": {"filename": "d1.txt", "provenance": [{"evidence_id": "ev_d1.txt"}]},
                "claim_b": {"filename": "d2.txt", "provenance": [{"evidence_id": "ev_x"}]}}]
        p = A.complete_prediction_from_results(RECORDS[0], [_result(sup=("d1.txt", A.RULEBOOK_FILENAME, "other.txt"), con=con)])
        self.assertEqual((p["predicted_supporting_evidence"], p["predicted_contradicting_evidence"]), (["d1"], ["d2"]))

    def test_unavailable_fields_stay_none_never_defaulted(self):
        p = A.complete_prediction_from_results(RECORDS[0], [_result(contra="NOT_MEASURED")])
        self.assertIsNone(p["predicted_missing_evidence"])
        self.assertIsNone(p["predicted_contradiction_detected"])
        self.assertIs(A.complete_prediction_from_results(RECORDS[0], [_result(contra=True)])["predicted_contradiction_detected"], True)

    def test_no_decision_means_no_prediction(self):
        self.assertIsNone(A.complete_prediction_from_results(RECORDS[0], []))


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.spy = FakeTasks(), SpyCompare()
        self.before = copy.deepcopy(RECORDS)
        self.out = A.run_paired(RECORDS, self.tasks, compare=self.spy)

    def test_identical_perturbation_ids_and_records_for_both_systems(self):
        ids = [r["perturbation_id"] for r in RECORDS]
        self.assertEqual(self.out["record_ids"], ids)
        self.assertEqual([p["perturbation_id"] for p in self.out["complete_predictions"]], ids)  # same ids, same order, none dropped or added
        self.assertEqual(self.spy.args[0], RECORDS)  # ONE record list is passed to the evaluator for both systems

    def test_identical_ground_truth_and_records_untouched(self):
        self.assertEqual(RECORDS, self.before)
        res = evaluate_adversarial_predictions(RECORDS, self.out["complete_predictions"])
        for r, x in zip(RECORDS, res["per_perturbation"]): self.assertEqual(x["expected_decision"], r["expected_decision"])
        for p in self.out["complete_predictions"]: self.assertFalse(set(GT_KEYS) & set(p))  # a prediction never carries ground truth

    def test_prediction_schema_and_evaluator_compatibility(self):
        for p in self.out["complete_predictions"]:
            self.assertLessEqual(set(p), _ALLOWED_PRED_KEYS)
            self.assertIn(p["predicted_decision"], (COMP, NC, INS))
        res = evaluate_adversarial_predictions(RECORDS, self.out["complete_predictions"])
        self.assertEqual((res["status"], res["case_count"], res["coverage"]), ("MEASURED", len(RECORDS), 1.0))
        self.assertEqual(res["overall"]["evidence_grounding"]["missing_flag_accuracy"], NOT_MEASURED)  # field left unavailable -> NOT_MEASURED, not 0

    def test_v1_complete_pairing(self):
        c = self.out["comparison"]
        self.assertEqual((c["status"], c["paired"]["paired_count"]), ("NOT_MEASURED", 0))  # V1 has no valid prediction -> nothing is paired or scored
        v1_fixture = [{"perturbation_id": r["perturbation_id"], "predicted_decision": INS} for r in RECORDS]  # TEST INPUT standing in for a future structured V1
        c2 = compare_systems(RECORDS, v1_fixture, self.out["complete_predictions"])
        self.assertEqual((c2["status"], c2["paired"]["paired_count"], c2["paired"]["same_records_for_both"]), ("MEASURED", len(RECORDS), True))
        self.assertEqual(c2["run_provenance"]["records_sha256"], evaluate_adversarial_predictions(RECORDS, self.out["complete_predictions"])["run_provenance"]["records_sha256"])

    def test_provenance_preserved(self):
        for r, p in zip(RECORDS, self.out["complete_predictions"]):
            self.assertEqual(p["provenance"]["perturbation_provenance"], r["provenance"])
            self.assertEqual(p["provenance"]["system"], "COMPLETE")
        p0 = self.out["complete_predictions"][0]
        p0["provenance"]["perturbation_provenance"]["marker"] = "changed"
        self.assertNotIn("marker", RECORDS[0]["provenance"])  # copied, never aliased
        res = evaluate_adversarial_predictions(RECORDS, self.out["complete_predictions"])
        self.assertIn("perturbation_provenance", res["per_perturbation"][0]["provenance"])

    def test_deterministic(self):
        again = A.run_paired(RECORDS, FakeTasks())
        self.assertEqual(again["complete_predictions"], self.out["complete_predictions"])

    def test_documents_and_policy_written_verbatim(self):
        graph_calls = [c for c in self.tasks.calls if c[0] == "graph"]
        self.assertEqual(graph_calls[0][1], [f"{d['doc_id']}.txt" for d in RECORDS[0]["perturbed_document_set"]])
        self.assertEqual(graph_calls[0][2], A.RULEBOOK_FILENAME)


class VersionSelectionTests(unittest.TestCase):
    def test_all_v1_to_v8_accepted_with_provenance(self):
        for v in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"):
            self.assertEqual(A.validate_version(v), v)
            out = A.run_paired(RECORDS[:2], FakeTasks(), version=v)
            self.assertEqual(out["selected_version"], v)
            self.assertTrue(all(p["provenance"]["selected_version"] == v for p in out["complete_predictions"]))
        self.assertEqual(A.VERSIONS, tuple(f"V{i}" for i in range(1, 9)))

    def test_invalid_versions_rejected_before_anything_runs(self):
        for bad in ("V0", "V9", "v1", "", " V1", "V1 ", None, 1, ["V1"]):
            with self.assertRaises(ValueError): A.validate_version(bad)
        tasks = FakeTasks()
        with self.assertRaises(ValueError): A.run_paired(RECORDS[:2], tasks, version="V9")
        self.assertEqual(tasks.calls, [])
        self.assertIsNone(A._ACTIVE_CONFIG["version"])

    def test_legacy_default_unchanged(self):
        out = A.run_paired(RECORDS[:3], FakeTasks())
        self.assertNotIn("selected_version", out)
        self.assertTrue(all("selected_version" not in p["provenance"] for p in out["complete_predictions"]))
        sel = A.run_paired(RECORDS[:3], FakeTasks(), version="V8")
        strip = lambda ps: [{**p, "provenance": {k: v for k, v in p["provenance"].items() if k not in ("selected_version", "version_logic")}} for p in ps]
        self.assertEqual(strip(sel["complete_predictions"]), out["complete_predictions"])  # selection changes provenance only

    def test_v1_selected_still_free_text_never_parsed(self):
        out = A.run_paired(RECORDS[:2], FakeTasks(), run_v1=True, version="V1")
        self.assertIsNone(out["v1_predictions"])
        self.assertTrue(all(t["status"] == "RAN" for t in out["v1_traces"]))

    def test_config_restored_after_success_and_failure(self):
        tasks = FakeTasks()
        tasks.knob = "orig"
        out = A.run_paired(RECORDS[:2], tasks, version="V3", config_overrides={"knob": "temp", "new_attr": 1})
        self.assertEqual((tasks.knob, hasattr(tasks, "new_attr"), A._ACTIVE_CONFIG["version"]), ("orig", False, None))
        self.assertEqual(out["selected_version"], "V3")
        seen = {}
        def boom(*a, **k):
            seen["during"] = (tasks.knob, A._ACTIVE_CONFIG["version"])
            raise RuntimeError("compare failed")
        with self.assertRaises(RuntimeError): A.run_paired(RECORDS[:2], tasks, version="V5", config_overrides={"knob": "temp"}, compare=boom)
        self.assertEqual(seen["during"], ("temp", "V5"))  # override really applied during the run
        self.assertEqual((tasks.knob, A._ACTIVE_CONFIG["version"]), ("orig", None))  # and restored although the run raised

    def test_blocked_tasks_none_with_version(self):
        out = A.run_paired(RECORDS[:2], None, version="V2", blocker="x")
        self.assertEqual((out["selected_version"], out["comparison"]["status"], A._ACTIVE_CONFIG["version"]), ("V2", "NOT_MEASURED", None))


class FakeGraph:
    def __init__(self, verdicts): self.nodes = {f"dec{i}": {"verdict": v} for i, v in enumerate(verdicts)}


class FakeV2Tasks:
    """tasks.py stand-in with the real flag names; every V3+ component is a trap that fails the test if V2 calls it."""
    ENABLE_CROSS_DOCUMENT_LINKING = ENABLE_CROSS_DOCUMENT_REASONING = ENABLE_CONTRADICTION_HEURISTIC = COMPILED_POLICY_ENABLED = True
    def __init__(self, verdicts=("SATISFIED",)): self.verdicts, self.seen, self.traps = verdicts, [], []
    def build_evidence_graph(self, paths, rb):
        self.seen.append({f: getattr(self, f) for f in A.V2_COMPONENT_FLAGS})
        return FakeGraph(self.verdicts)
    def _nodes_of_type(self, G, t): return [(k, {}) for k in G.nodes]
    def _sv7d_outcome(self, verdict, status): return verdict if verdict in ("VIOLATION", "SATISFIED") else "NO_CONCLUSION"
    def _trap(self, name, *a, **k): self.traps.append(name); raise AssertionError(f"V2 must not call {name}")
    def apply_compiled_policy(self, *a, **k): self._trap("apply_compiled_policy")
    def self_verification_variant_predictions(self, *a, **k): self._trap("self_verification_variant_predictions")
    def verify_policy_applicability(self, *a, **k): self._trap("verify_policy_applicability")
    def build_counterfactual(self, *a, **k): self._trap("build_counterfactual")
    def build_uncertainty_assessment(self, *a, **k): self._trap("build_uncertainty_assessment")
    def analyze_compliance_with_matrix(self, *a, **k): self._trap("analyze_compliance_with_matrix")


class V2Tests(unittest.TestCase):
    def test_v2_disables_v3_plus_components(self):
        t = FakeV2Tasks(("VIOLATION",))
        out = A.run_version(RECORDS[:3], t, "V2")
        self.assertTrue(t.seen and all(not any(s.values()) for s in t.seen))  # every flag was False DURING the graph build
        self.assertEqual(t.traps, [])
        st = out["component_state"]
        for c in ("cross_document_linking", "cross_document_reasoning", "contradiction_detection", "compiled_policy"): self.assertEqual((st[c]["enabled"], st[c]["invoked"]), (False, False))
        for c in A.V2_NO_FLAG_COMPONENTS: self.assertEqual((st[c]["flag"], st[c]["enabled"], st[c]["invoked"]), (None, False, False))
        for p in out["predictions"]:
            self.assertEqual((p["predicted_decision"], p["provenance"]["selected_version"], p["provenance"]["system"]), (NC, "V2", "V2"))
            self.assertEqual(p["provenance"]["component_state"], st)

    def test_v2_does_not_fabricate_unavailable_fields(self):
        out = A.run_version(RECORDS[:3], FakeV2Tasks(("SATISFIED", "NO_CONCLUSION")), "V2")
        for p in out["predictions"]:
            self.assertEqual(p["predicted_decision"], INS)
            self.assertFalse(set(A.V2_UNAVAILABLE_FIELDS) & set(p))  # omitted, not defaulted
            self.assertLessEqual(set(p), _ALLOWED_PRED_KEYS | {"provenance"})
            self.assertFalse(set(GT_KEYS) & set(p))
        res = evaluate_adversarial_predictions(RECORDS[:3], out["predictions"])
        self.assertEqual(res["overall"]["evidence_grounding"]["missing_flag_accuracy"], NOT_MEASURED)

    def test_v2_no_decision_or_missing_flag_gives_no_guess(self):
        out = A.run_version(RECORDS[:2], FakeV2Tasks(()), "V2")
        self.assertEqual((out["predictions"], len(out["unanswered"])), (None, 2))
        with self.assertRaises(RuntimeError): A.v2_component_state(types.SimpleNamespace(ENABLE_CROSS_DOCUMENT_LINKING=True))  # flags absent -> cannot be honestly graph-only

    def test_config_restored_after_v2_success_and_failure(self):
        t = FakeV2Tasks()
        A.run_version(RECORDS[:2], t, "V2")
        self.assertEqual([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], [True] * 4)
        self.assertIsNone(A._ACTIVE_CONFIG["version"])
        t.verdicts = None  # makes the fake graph build raise inside the run
        t.build_evidence_graph = lambda p, r: (_ for _ in ()).throw(RuntimeError("boom"))
        out = A.run_version(RECORDS[:2], t, "V2")  # per-record failure is reported, not raised
        self.assertEqual(len(out["unanswered"]), 2)
        orig = A.predict_v2
        A.predict_v2 = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hard fail"))
        try:
            with self.assertRaises(RuntimeError): A.run_version(RECORDS[:2], t, "V2")
        finally: A.predict_v2 = orig
        self.assertEqual([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], [True] * 4)
        self.assertIsNone(A._ACTIVE_CONFIG["version"])

    def test_v1_unchanged_and_legacy_complete_unaffected(self):
        t = FakeV2Tasks()
        out = A.run_version(RECORDS[:2], t, "V1")
        self.assertEqual((out["predictions"], out["blocker"], t.seen), (None, A.V1_NOT_MEASURED_REASON, []))
        self.assertEqual(A.run_paired(RECORDS[:2], FakeTasks())["complete_predictions"], A.run_paired(RECORDS[:2], FakeTasks())["complete_predictions"])


class V3Tests(unittest.TestCase):
    """Real tasks.py only (skipped where it is not importable). No metrics, no fabricated compiler success: whatever the real compiler does is what is observed."""
    def test_v3_runs_real_tasks_orders_compile_before_rules_and_restores_config(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        keys = tuple(A.V3_COMPONENT_FLAGS) + ("apply_compiled_policy", "evaluate_policy_rules")
        before = {k: getattr(mod, k) for k in keys}
        order, real_apply, real_eval = [], mod.apply_compiled_policy, mod.evaluate_policy_rules
        def spy_apply(G, text): order.append("apply_compiled_policy"); return real_apply(G, text)
        def spy_eval(G): order.append("evaluate_policy_rules"); return real_eval(G)
        mod.apply_compiled_policy, mod.evaluate_policy_rules = spy_apply, spy_eval
        try:
            out = A.run_version(RECORDS[:1], mod, "V3")  # must not raise NotImplementedError (or anything)
        finally: mod.apply_compiled_policy, mod.evaluate_policy_rules = real_apply, real_eval
        self.assertEqual(out["selected_version"], "V3")
        self.assertEqual(len(out["predictions"] or []) + len(out["unanswered"]), 1)  # a prediction OR an honest unanswered entry, never silence
        for u in out["unanswered"]: self.assertTrue(u["reason"])
        for p in out["predictions"] or []: self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V3", "V2"))
        if "apply_compiled_policy" in order: self.assertLess(order.index("apply_compiled_policy"), order.index("evaluate_policy_rules"))  # compile precedes rule evaluation
        self.assertEqual({k: getattr(mod, k) for k in keys}, before)  # flags and callables restored
        self.assertIsNone(A._ACTIVE_CONFIG["version"])

    def test_v8_no_longer_blocked_and_invalid_rejected(self):
        t = FakeV2Tasks()
        self.assertEqual(A._VERSION_BLOCKERS, {})
        with self.assertRaises(ValueError): A.run_version(RECORDS[:1], t, "V9")
        self.assertEqual((t.seen, t.traps, [getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([], [], [True] * 4, None))  # invalid version rejected before anything runs
        out = A.run_version(RECORDS[:1], t, "V8")  # selectable now: never NotImplementedError; the fake's traps are reported as honest unanswered entries
        self.assertEqual((out["selected_version"], out["predictions"], len(out["unanswered"])), ("V8", None, 1))
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([True] * 4, None))


class V4Tests(unittest.TestCase):
    def test_v4_enables_linking_reasoning_with_v3_compiled_policy_and_restores_config(self):
        t = FakeV2Tasks(("VIOLATION",))
        for f in A.V2_COMPONENT_FLAGS: setattr(t, f, False)
        t.compiled, order = [], []
        t.apply_compiled_policy = lambda G, text: order.append("apply_compiled_policy")
        t.evaluate_policy_rules = lambda G: order.append("evaluate_policy_rules") or []
        t._rulebook_text_only = lambda rb: "RB"
        t.build_evidence_graph = lambda paths, rb: (t.seen.append({f: getattr(t, f) for f in A.V2_COMPONENT_FLAGS}), t.evaluate_policy_rules(None), FakeGraph(t.verdicts))[2]
        out = A.run_version(RECORDS[:2], t, "V4")
        self.assertEqual(A.V4_COMPONENT_FLAGS, {**A.V3_COMPONENT_FLAGS, "ENABLE_CROSS_DOCUMENT_LINKING": True, "ENABLE_CROSS_DOCUMENT_REASONING": True})
        self.assertTrue(t.seen and all(s == A.V4_COMPONENT_FLAGS for s in t.seen))  # linking/reasoning/compiled ON, contradiction heuristic OFF, DURING the build
        self.assertEqual(order, ["apply_compiled_policy", "evaluate_policy_rules"] * len(t.seen))  # V3 ordering preserved
        st = out["component_state"]
        self.assertEqual([st[c]["enabled"] for c in ("cross_document_linking", "cross_document_reasoning", "compiled_policy", "contradiction_detection")], [True, True, True, False])
        self.assertEqual(out["selected_version"], "V4")
        self.assertEqual(len(out["predictions"]), 2)
        for p in out["predictions"]: self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V4", "V2"))
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))

    def test_v4_flags_restored_when_run_raises(self):
        t = FakeV2Tasks(None)  # fake graph build raises inside the run
        A.run_version(RECORDS[:1], t, "V4")
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([True] * 4, None))

    def test_v4_runs_real_tasks_cross_document_path(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        keys = tuple(A.V4_COMPONENT_FLAGS)
        before, seen = {k: getattr(mod, k) for k in keys}, []
        real_build = mod.build_evidence_graph
        def spy_build(*a, **k): seen.append({x: getattr(mod, x) for x in keys}); return real_build(*a, **k)
        mod.build_evidence_graph = spy_build
        try: out = A.run_version(RECORDS[:1], mod, "V4")
        finally: mod.build_evidence_graph = real_build
        self.assertEqual(seen, [A.V4_COMPONENT_FLAGS])
        self.assertEqual(len(out["predictions"] or []) + len(out["unanswered"]), 1)
        self.assertEqual({k: getattr(mod, k) for k in keys}, before)


class V5Tests(unittest.TestCase):
    def test_v5_enables_contradiction_heuristic_on_top_of_v4_and_restores_config(self):
        t = FakeV2Tasks(("VIOLATION",))
        for f in A.V2_COMPONENT_FLAGS: setattr(t, f, False)
        order = []
        t.apply_compiled_policy = lambda G, text: order.append("apply_compiled_policy")
        t.evaluate_policy_rules = lambda G: order.append("evaluate_policy_rules") or []
        t._rulebook_text_only = lambda rb: "RB"
        t.build_evidence_graph = lambda paths, rb: (t.seen.append({f: getattr(t, f) for f in A.V2_COMPONENT_FLAGS}), t.evaluate_policy_rules(None), FakeGraph(t.verdicts))[2]
        out = A.run_version(RECORDS[:2], t, "V5")
        self.assertEqual(A.V5_COMPONENT_FLAGS, {**A.V4_COMPONENT_FLAGS, "ENABLE_CONTRADICTION_HEURISTIC": True})
        self.assertEqual([k for k, v in A.V5_COMPONENT_FLAGS.items() if v != A.V4_COMPONENT_FLAGS[k]], ["ENABLE_CONTRADICTION_HEURISTIC"])  # only delta vs V4
        self.assertTrue(t.seen and all(s == A.V5_COMPONENT_FLAGS and all(s.values()) for s in t.seen))  # all four ON DURING the build
        self.assertEqual(order, ["apply_compiled_policy", "evaluate_policy_rules"] * len(t.seen))  # V3/V4 ordering preserved
        st = out["component_state"]
        self.assertTrue(all(st[c]["enabled"] for c in ("cross_document_linking", "cross_document_reasoning", "compiled_policy", "contradiction_detection")))
        for c in A.V2_NO_FLAG_COMPONENTS: self.assertEqual((st[c]["enabled"], st[c]["invoked"]), (False, False))  # no self-verification/counterfactual/uncertainty
        self.assertEqual(t.traps, [])
        self.assertEqual((out["selected_version"], len(out["predictions"])), ("V5", 2))
        for p in out["predictions"]: self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V5", "V2"))
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))

    def test_v5_flags_restored_when_run_raises(self):
        t = FakeV2Tasks(None)
        A.run_version(RECORDS[:1], t, "V5")
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([True] * 4, None))

    def test_v5_runs_real_tasks_with_contradiction_heuristic_on(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        keys = tuple(A.V5_COMPONENT_FLAGS)
        before, seen = {k: getattr(mod, k) for k in keys}, []
        real_build = mod.build_evidence_graph
        def spy_build(*a, **k): seen.append({x: getattr(mod, x) for x in keys}); return real_build(*a, **k)
        mod.build_evidence_graph = spy_build
        try: out = A.run_version(RECORDS[:1], mod, "V5")
        finally: mod.build_evidence_graph = real_build
        self.assertTrue(seen and all(s == A.V5_COMPONENT_FLAGS for s in seen))  # flags during the real graph build
        for f in ("ENABLE_CONTRADICTION_HEURISTIC", "ENABLE_CROSS_DOCUMENT_LINKING", "ENABLE_CROSS_DOCUMENT_REASONING", "COMPILED_POLICY_ENABLED"): self.assertTrue(seen[0][f], f)
        self.assertEqual(out["selected_version"], "V5")
        self.assertEqual(len(out["predictions"] or []) + len(out["unanswered"]), 1)  # a prediction OR an honest unanswered entry
        for u in out["unanswered"]: self.assertTrue(u["reason"])
        for p in out["predictions"] or []: self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V5", "V2"))
        self.assertEqual({k: getattr(mod, k) for k in keys}, before)  # flags restored
        self.assertIs(mod.build_evidence_graph, real_build)
        self.assertIsNone(A._ACTIVE_CONFIG["version"])

    def test_v4_unchanged_contradiction_still_off(self):
        self.assertFalse(A.V4_COMPONENT_FLAGS["ENABLE_CONTRADICTION_HEURISTIC"])


class V6Tests(unittest.TestCase):
    def _tasks(self, verdicts=("VIOLATION",), status="ESCALATE"):
        t = FakeV2Tasks(verdicts)
        for f in A.V2_COMPONENT_FLAGS: setattr(t, f, False)
        t.sv_calls, order = [], []
        t.apply_compiled_policy = lambda G, text: order.append("apply_compiled_policy")
        t.evaluate_policy_rules = lambda G: order.append("evaluate_policy_rules") or []
        t._rulebook_text_only = lambda rb: "RB"
        t.build_evidence_graph = lambda paths, rb: (t.seen.append({f: getattr(t, f) for f in A.V2_COMPONENT_FLAGS}), t.evaluate_policy_rules(None), FakeGraph(t.verdicts))[2]
        t.self_verification_variant_predictions = lambda G, d: (t.sv_calls.append(d), {A.V6_VARIANT: {"outcome": "VIOLATION", "verification_status": status}})[1]
        t.order = order
        return t

    def test_v6_keeps_v5_flags_invokes_only_self_verification_and_restores_config(self):
        t = self._tasks()
        out = A.run_version(RECORDS[:2], t, "V6")
        self.assertEqual(A.V6_COMPONENT_FLAGS, A.V5_COMPONENT_FLAGS)
        self.assertTrue(t.seen and all(s == A.V5_COMPONENT_FLAGS and all(s.values()) for s in t.seen))  # all four V5 flags ON during the build
        self.assertEqual(t.order, ["apply_compiled_policy", "evaluate_policy_rules"] * len(t.seen))  # compile before rules preserved
        self.assertEqual(len(t.sv_calls), 2)  # self-verification invoked once per Decision per record
        self.assertEqual(t.traps, [])  # no applicability / counterfactual / uncertainty / V1 call
        st = out["component_state"]
        self.assertEqual((st["self_verification"]["enabled"], st["self_verification"]["invoked"], st["self_verification"]["invocations"]), (True, True, 2))
        self.assertTrue(all(st[c]["enabled"] for c in ("cross_document_linking", "cross_document_reasoning", "compiled_policy", "contradiction_detection")))
        for c in ("counterfactual", "uncertainty"): self.assertEqual((st[c]["enabled"], st[c]["invoked"]), (False, False))
        self.assertEqual((out["selected_version"], len(out["predictions"])), ("V6", 2))
        for p in out["predictions"]:
            self.assertEqual((p["predicted_decision"], p["predicted_escalation"]), (NC, True))
            self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"], p["provenance"]["variant"]), ("V6", "V2", A.V6_VARIANT))
            for f in A.V6_UNAVAILABLE_FIELDS: self.assertNotIn(f, p)
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))

    def test_v6_missing_variant_or_no_decision_gives_unanswered_not_a_guess(self):
        t = self._tasks()
        t.self_verification_variant_predictions = lambda G, d: {}
        out = A.run_version(RECORDS[:1], t, "V6")
        self.assertEqual((out["predictions"], len(out["unanswered"])), (None, 1))
        self.assertIn("KeyError", out["unanswered"][0]["reason"])
        self.assertEqual(A.run_version(RECORDS[:1], self._tasks(()), "V6")["unanswered"][0]["reason"], "pipeline produced no Decision node")

    def test_v6_flags_restored_when_run_raises(self):
        t = FakeV2Tasks(None)
        A.run_version(RECORDS[:1], t, "V6")
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([True] * 4, None))

    def test_v5_unchanged_never_invokes_self_verification(self):
        t = self._tasks()
        out = A.run_version(RECORDS[:1], t, "V5")
        self.assertEqual((t.sv_calls, A.V5_COMPONENT_FLAGS["ENABLE_CONTRADICTION_HEURISTIC"], out["component_state"]["self_verification"]["invoked"]), ([], True, False))
        for p in out["predictions"]: self.assertEqual(p["provenance"]["selected_version"], "V5")

    def test_v6_runs_real_tasks_with_self_verification(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        keys = tuple(A.V6_COMPONENT_FLAGS)
        before, seen, sv = {k: getattr(mod, k) for k in keys}, [], []
        real_build = mod.build_evidence_graph
        real_verify = getattr(mod, "verify_self_verification_result", None)
        if not callable(real_verify): self.skipTest("tasks.py has no verify_self_verification_result (7B) function")
        real_applicability = getattr(mod, "verify_policy_applicability", None)
        applicability_calls = []
        def spy_build(*a, **k):
            seen.append({x: getattr(mod, x) for x in keys})
            return real_build(*a, **k)
        def spy_verify(*a, **k):
            r = real_verify(*a, **k)
            sv.append(r)
            return r
        def spy_applicability(*a, **k):
            applicability_calls.append((a, k))
            return real_applicability(*a, **k)
        mod.build_evidence_graph = spy_build
        mod.verify_self_verification_result = spy_verify
        if callable(real_applicability): mod.verify_policy_applicability = spy_applicability
        try:
            out = A.run_version(RECORDS[:1], mod, "V6")
        finally:
            mod.build_evidence_graph = real_build
            mod.verify_self_verification_result = real_verify
            if callable(real_applicability): mod.verify_policy_applicability = real_applicability
        self.assertTrue(seen and all(s == A.V5_COMPONENT_FLAGS and all(s.values()) for s in seen))
        self.assertEqual(out["selected_version"], "V6")
        self.assertTrue(sv, "real verify_self_verification_result (7B) was never invoked/returned")
        self.assertTrue(all(isinstance(x, dict) for x in sv))
        self.assertFalse(applicability_calls, "V6 must not invoke 7C policy applicability")
        self.assertTrue(out["component_state"]["self_verification"]["invoked"])
        self.assertFalse(out["component_state"]["policy_applicability"]["enabled"])
        self.assertEqual(len(out["predictions"] or []) + len(out["unanswered"]), 1)
        for u in out["unanswered"]: self.assertTrue(u["reason"])
        for p in out["predictions"] or []: self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V6", "V2"))
        self.assertEqual({k: getattr(mod, k) for k in keys}, before)
        self.assertIs(mod.build_evidence_graph, real_build)
        self.assertIs(mod.verify_self_verification_result, real_verify)
        self.assertIsNone(A._ACTIVE_CONFIG["version"])


class V7Tests(unittest.TestCase):
    CF = {"counterfactual_id": "cf::dec0", "in_scope": True, "scope_class": "NON_COMPLIANT", "status": "UNRESOLVED", "change_type": None, "unresolved_reason": "why",
          "missing_requirement": NOT_MEASURED, "recommended_corrective_condition": NOT_MEASURED, "expected_resulting_state": NOT_MEASURED, "minimum_changes": [],
          "requires_reevaluation": False, "satisfaction_asserted": False, "decision_id": "dec0", "extra_internal": "x"}

    def _tasks(self, verdicts=("VIOLATION",), cf=None):
        t = V6Tests()._tasks(verdicts)
        t.cf_calls = []
        t.build_counterfactual = lambda G, d: (t.cf_calls.append((G, d)), copy.deepcopy(self.CF if cf is None else cf))[1]
        return t

    def test_v7_keeps_v5_flags_invokes_sv_and_counterfactual_never_uncertainty_and_restores_config(self):
        t = self._tasks()
        out = A.run_version(RECORDS[:2], t, "V7")
        self.assertEqual(A.V7_COMPONENT_FLAGS, A.V5_COMPONENT_FLAGS)
        self.assertTrue(t.seen and all(s == A.V5_COMPONENT_FLAGS and all(s.values()) for s in t.seen))
        self.assertEqual(len(t.sv_calls), 2)  # V6 path still runs
        self.assertEqual([d for _, d in t.cf_calls], ["dec0", "dec0"])  # existing counterfactual invoked as build_counterfactual(G, decision_id), once per Decision
        self.assertTrue(all(isinstance(G, FakeGraph) for G, _ in t.cf_calls))
        self.assertEqual(t.traps, [])  # build_uncertainty_assessment / applicability / V1 never called
        st = out["component_state"]
        self.assertEqual((st["counterfactual"]["enabled"], st["counterfactual"]["invoked"], st["counterfactual"]["invocations"]), (True, True, 2))
        self.assertEqual((st["uncertainty"]["enabled"], st["uncertainty"]["invoked"]), (False, False))
        self.assertEqual((out["selected_version"], len(out["predictions"])), ("V7", 2))
        for p in out["predictions"]:
            pv = p["provenance"]
            self.assertEqual((pv["selected_version"], pv["system"], pv["variant"]), ("V7", "V2", A.V7_VARIANT))
            self.assertEqual(len(pv["counterfactual"]), 1)
            e = pv["counterfactual"][0]
            self.assertEqual(set(e), {"decision_id"} | set(A.V7_COUNTERFACTUAL_FIELDS))  # only existing fields; extra_internal / decision_id from the result not copied through
            self.assertEqual((e["status"], e["missing_requirement"], e["satisfaction_asserted"]), ("UNRESOLVED", NOT_MEASURED, False))  # values passed through untouched
            for f in A.V7_UNAVAILABLE_FIELDS: self.assertNotIn(f, p)
            self.assertEqual(set(p) - {"provenance"}, {"perturbation_id", "predicted_decision", "predicted_escalation"})  # no counterfactual key on the record: evaluator rejects extras
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))
        res = evaluate_adversarial_predictions(RECORDS[:2], out["predictions"])
        self.assertNotIn(res["status"], ("INVALID_RECORDS", "INVALID_PREDICTIONS"))  # schema-compatible

    def test_v6_unchanged_and_v7_decision_identical_to_v6(self):
        t6, t7 = self._tasks(), self._tasks()
        o6, o7 = A.run_version(RECORDS[:2], t6, "V6"), A.run_version(RECORDS[:2], t7, "V7")
        self.assertEqual(t6.cf_calls, [])  # V6 never calls counterfactual
        self.assertEqual((o6["component_state"]["counterfactual"]["enabled"], o6["component_state"]["counterfactual"]["invoked"]), (False, False))
        for p in o6["predictions"]: self.assertEqual((p["provenance"]["selected_version"], "counterfactual" in p["provenance"]), ("V6", False))
        strip = lambda ps: [{k: v for k, v in p.items() if k != "provenance"} for p in ps]
        self.assertEqual(strip(o6["predictions"]), strip(o7["predictions"]))  # decision / escalation identical; counterfactual supplies none

    def test_v7_missing_counterfactual_output_is_unanswered_never_fabricated(self):
        for bad in (None, "x", []):
            t = self._tasks(cf=bad)
            t.build_counterfactual = lambda G, d, b=bad: b
            out = A.run_version(RECORDS[:1], t, "V7")
            self.assertEqual((out["predictions"], len(out["unanswered"])), (None, 1))
            self.assertIn("counterfactual output missing", out["unanswered"][0]["reason"])
        t = self._tasks()
        def boom(G, d): raise RuntimeError("cf broke")
        t.build_counterfactual = boom
        out = A.run_version(RECORDS[:1], t, "V7")
        self.assertEqual((out["predictions"], out["unanswered"][0]["reason"]), (None, "RuntimeError: cf broke"))
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))  # restored after the failure
        sparse = A.run_version(RECORDS[:1], self._tasks(cf={"status": "NOT_REQUIRED"}), "V7")  # fields the pipeline did not return stay omitted
        self.assertEqual(sparse["predictions"][0]["provenance"]["counterfactual"], [{"decision_id": "dec0", "status": "NOT_REQUIRED"}])

    def test_v7_flags_restored_when_run_raises(self):
        t = FakeV2Tasks(None)
        A.run_version(RECORDS[:1], t, "V7")
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([True] * 4, None))

    def test_v6_still_never_invokes_counterfactual(self):
        t = self._tasks()
        A.run_version(RECORDS[:1], t, "V6")
        self.assertEqual(t.cf_calls, [])

    def test_v7_runs_real_tasks_with_real_counterfactual(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        keys = tuple(A.V7_COMPONENT_FLAGS)
        before, seen, cf, unc = {k: getattr(mod, k) for k in keys}, [], [], []
        real_build, real_cf, real_unc = mod.build_evidence_graph, mod.build_counterfactual, mod.build_uncertainty_assessment
        def spy_build(*a, **k): seen.append({x: getattr(mod, x) for x in keys}); return real_build(*a, **k)
        def spy_cf(*a, **k): r = real_cf(*a, **k); cf.append(r); return r
        def spy_unc(*a, **k): unc.append(a); return real_unc(*a, **k)
        mod.build_evidence_graph, mod.build_counterfactual, mod.build_uncertainty_assessment = spy_build, spy_cf, spy_unc
        try: out = A.run_version(RECORDS[:1], mod, "V7")
        finally: mod.build_evidence_graph, mod.build_counterfactual, mod.build_uncertainty_assessment = real_build, real_cf, real_unc
        self.assertTrue(seen and all(s == A.V5_COMPONENT_FLAGS and all(s.values()) for s in seen))
        self.assertTrue(cf, "real tasks.build_counterfactual was never invoked/returned")  # an exception there must fail here, not pass as unanswered
        self.assertTrue(all(r["status"] in mod.CF_STATUSES for r in cf))
        self.assertEqual(unc, [])  # uncertainty not invoked
        self.assertEqual(out["selected_version"], "V7")
        self.assertTrue(out["component_state"]["counterfactual"]["invoked"])
        self.assertEqual(len(out["predictions"] or []) + len(out["unanswered"]), 1)
        for p in out["predictions"] or []:
            self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V7", "V2"))
            self.assertEqual([e["status"] for e in p["provenance"]["counterfactual"]], [r["status"] for r in cf])
        self.assertEqual({k: getattr(mod, k) for k in keys}, before)
        self.assertIs(mod.build_counterfactual, real_cf)
        self.assertIsNone(A._ACTIVE_CONFIG["version"])


class V8Tests(unittest.TestCase):
    UNC = {"contract_version": "UNC1", "decision_id": "dec0", "found": True, "decision": {"decision_id": "dec0", "verdict": "VIOLATION"}, "uncertainty_status": "CONDITIONAL",
           "uncertainty_state_reason": "why", "decision_confidence": {"value": None, "status": NOT_MEASURED, "basis": "b"}, "evidence_completeness": {"value": "COMPLETE", "status": "MEASURED", "basis": "b"},
           "contradiction_severity": {"value": None, "status": NOT_MEASURED, "basis": "b"}, "policy_alignment": {"value": "UNESTABLISHED", "status": "MEASURED", "basis": "b"},
           "risk_level": {"value": "HIGH", "status": "MEASURED", "basis": "b"}, "escalation_required": True, "escalation_reasons": [{"reason": "HIGH_RISK", "detail": "d"}],
           "verification_status": "VERIFIED", "policy_applicability_status": "UNESTABLISHED", "supporting_evidence": ["internal"], "gaps": ["internal"]}

    def _tasks(self, verdicts=("VIOLATION",), unc=None, validate=None):
        t = V7Tests()._tasks(verdicts)
        t.unc_calls = []
        t.build_uncertainty_assessment = lambda G, d: (t.unc_calls.append((G, d)), copy.deepcopy(self.UNC if unc is None else unc))[1]
        if validate is not None: t.validate_uncertainty_assessment = validate
        return t

    def test_v8_keeps_v7_flags_invokes_uncertainty_and_uses_its_decision_and_restores_config(self):
        t = self._tasks()
        out = A.run_version(RECORDS[:2], t, "V8")
        self.assertEqual(A.V8_COMPONENT_FLAGS, A.V7_COMPONENT_FLAGS)
        self.assertTrue(t.seen and all(s == A.V7_COMPONENT_FLAGS and all(s.values()) for s in t.seen))  # all V7 flags still ON during the build
        self.assertEqual((len(t.sv_calls), [d for _, d in t.cf_calls], [d for _, d in t.unc_calls]), (2, ["dec0"] * 2, ["dec0"] * 2))  # V6 + V7 + uncertainty, once per Decision
        self.assertTrue(all(isinstance(G, FakeGraph) for G, _ in t.unc_calls))
        self.assertEqual(t.traps, [])  # the uncertainty trap was replaced; no applicability / V1 call
        st = out["component_state"]
        for c in ("self_verification", "counterfactual", "uncertainty"): self.assertEqual((st[c]["enabled"], st[c]["invoked"], st[c]["invocations"]), (True, True, 2))
        self.assertEqual((out["selected_version"], len(out["predictions"]), out["unanswered"]), ("V8", 2, []))
        for p in out["predictions"]:
            pv = p["provenance"]
            self.assertEqual((pv["selected_version"], pv["system"], pv["variant"]), ("V8", "V2", A.V8_VARIANT))
            self.assertEqual((p["predicted_decision"], p["predicted_escalation"]), (INS, True))  # CONDITIONAL -> INS (vocabulary map); escalation_required passed through
            self.assertEqual((pv["decision_source"], pv["v7_predicted_decision"], pv["v7_predicted_escalation"]), ("uncertainty_status", NC, True))  # V7 value kept visible, not silently replaced
            self.assertEqual(len(pv["counterfactual"]), 1)
            e = pv["uncertainty"][0]
            self.assertEqual(set(e), {"decision_id"} | {k for k in A.V8_UNCERTAINTY_FIELDS if k in self.UNC})  # only fields the real result returns; supporting_evidence / gaps / decision not copied
            self.assertEqual((e["uncertainty_status"], e["decision_confidence"], e["escalation_reasons"]), ("CONDITIONAL", self.UNC["decision_confidence"], self.UNC["escalation_reasons"]))  # untouched (confidence stays NOT_MEASURED)
            for f in A.V8_UNAVAILABLE_FIELDS: self.assertNotIn(f, p)
            self.assertEqual(set(p) - {"provenance"}, {"perturbation_id", "predicted_decision", "predicted_escalation"})  # no confidence / probability / calibration key anywhere on the record
            self.assertNotIn("confidence", str(sorted(p)))
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))
        res = evaluate_adversarial_predictions(RECORDS[:2], out["predictions"])
        self.assertNotIn(res["status"], ("INVALID_RECORDS", "INVALID_PREDICTIONS"))

    def test_v8_state_to_label_mapping_and_aggregation(self):
        for st, lab in (("COMPLIANT", COMP), ("NON_COMPLIANT", NC), ("CONDITIONAL", INS), ("INSUFFICIENT_EVIDENCE", INS)):
            out = A.run_version(RECORDS[:1], self._tasks(unc={**self.UNC, "uncertainty_status": st, "escalation_required": False, "escalation_reasons": []}), "V8")
            self.assertEqual(out["predictions"][0]["predicted_decision"], lab)
            self.assertIs(out["predictions"][0]["predicted_escalation"], False)
        t = self._tasks()
        t.build_evidence_graph = lambda paths, rb: (t.seen.append({}), FakeGraph(("VIOLATION", "SATISFIED")))[1]
        res = iter(("COMPLIANT", "INSUFFICIENT_EVIDENCE"))
        t.build_uncertainty_assessment = lambda G, d: {**self.UNC, "uncertainty_status": next(res), "escalation_required": False, "escalation_reasons": []}
        t._nodes_of_type = lambda G, ty: [("dec0", {}), ("dec1", {})]
        out = A.run_version(RECORDS[:1], t, "V8")
        self.assertEqual(out["predictions"][0]["predicted_decision"], INS)  # same priority as V7: violation > unresolved > pass

    def test_v7_unchanged_never_invokes_uncertainty_and_v8_differs_only_by_uncertainty_fields(self):
        t7, t8 = self._tasks(), self._tasks()
        o7, o8 = A.run_version(RECORDS[:2], t7, "V7"), A.run_version(RECORDS[:2], t8, "V8")
        self.assertEqual(t7.unc_calls, [])  # V7 never calls uncertainty
        self.assertEqual((o7["component_state"]["uncertainty"]["enabled"], o7["component_state"]["uncertainty"]["invoked"]), (False, False))
        for p in o7["predictions"]: self.assertEqual(("uncertainty" in p["provenance"], p["provenance"]["selected_version"], p["predicted_decision"], p["predicted_escalation"]), (False, "V7", NC, True))  # V6 fake status ESCALATE -> V7 escalation True
        self.assertEqual(len(t8.unc_calls), 2)
        # V8 with an uncertainty layer that agrees with V7 reproduces V7's decision / escalation exactly
        agree = {**self.UNC, "uncertainty_status": "NON_COMPLIANT", "escalation_required": True}
        o8b = A.run_version(RECORDS[:2], self._tasks(unc=agree), "V8")
        strip = lambda ps: [{k: v for k, v in p.items() if k != "provenance"} for p in ps]
        self.assertEqual(strip(o7["predictions"]), strip(o8b["predictions"]))

    def test_v8_missing_or_invalid_uncertainty_output_is_unanswered_never_fabricated(self):
        bad_cases = {"none": None, "str": "x", "list": [], "not_found": {**self.UNC, "found": False}, "no_status": {k: v for k, v in self.UNC.items() if k != "uncertainty_status"},
                     "not_measured_status": {**self.UNC, "uncertainty_status": NOT_MEASURED}, "unknown_status": {**self.UNC, "uncertainty_status": "MAYBE"},
                     "esc_not_bool": {**self.UNC, "escalation_required": "yes"}, "esc_missing": {k: v for k, v in self.UNC.items() if k != "escalation_required"}}
        for name, bad in bad_cases.items():
            t = self._tasks()
            t.build_uncertainty_assessment = lambda G, d, b=bad: b
            out = A.run_version(RECORDS[:1], t, "V8")
            self.assertEqual((out["predictions"], len(out["unanswered"])), (None, 1), name)
            self.assertIn("uncertainty output", out["unanswered"][0]["reason"], name)
            self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None), name)
        t = self._tasks(validate=lambda G, a: ["contract broken"])  # existing contract validator reports problems -> unanswered
        out = A.run_version(RECORDS[:1], t, "V8")
        self.assertEqual(out["predictions"], None)
        self.assertIn("contract broken", out["unanswered"][0]["reason"])
        t = self._tasks()
        def boom(G, d): raise RuntimeError("unc broke")
        t.build_uncertainty_assessment = boom
        out = A.run_version(RECORDS[:1], t, "V8")
        self.assertEqual((out["predictions"], out["unanswered"][0]["reason"]), (None, "RuntimeError: unc broke"))
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))  # restored after the failure
        sparse = A.run_version(RECORDS[:1], self._tasks(unc={"found": True, "uncertainty_status": "COMPLIANT", "escalation_required": False}), "V8")  # fields the pipeline did not return stay omitted
        self.assertEqual(sparse["predictions"][0]["provenance"]["uncertainty"], [{"decision_id": "dec0", "uncertainty_status": "COMPLIANT", "escalation_required": False}])

    def test_v8_valid_validator_passes_and_no_confidence_fabricated(self):
        t = self._tasks(validate=lambda G, a: [])
        out = A.run_version(RECORDS[:1], t, "V8")
        e = out["predictions"][0]["provenance"]["uncertainty"][0]
        self.assertEqual((e["decision_confidence"]["value"], e["decision_confidence"]["status"]), (None, NOT_MEASURED))

    def test_v8_flags_restored_when_run_raises(self):
        t = FakeV2Tasks(None)
        A.run_version(RECORDS[:1], t, "V8")
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([True] * 4, None))
        t = self._tasks()
        orig = A.predict_v8
        A.predict_v8 = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hard fail"))
        try:
            with self.assertRaises(RuntimeError): A.run_version(RECORDS[:1], t, "V8")
        finally: A.predict_v8 = orig
        self.assertEqual(([getattr(t, f) for f in A.V2_COMPONENT_FLAGS], A._ACTIVE_CONFIG["version"]), ([False] * 4, None))

    def test_v8_runs_real_tasks_with_real_uncertainty(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        keys = tuple(A.V8_COMPONENT_FLAGS)
        before, seen, unc = {k: getattr(mod, k) for k in keys}, [], []
        real_build, real_unc = mod.build_evidence_graph, mod.build_uncertainty_assessment
        def spy_build(*a, **k): seen.append({x: getattr(mod, x) for x in keys}); return real_build(*a, **k)
        def spy_unc(*a, **k): r = real_unc(*a, **k); unc.append(r); return r
        mod.build_evidence_graph, mod.build_uncertainty_assessment = spy_build, spy_unc
        try:
            out = A.run_version(RECORDS[:1], mod, "V8")
            v7 = A.run_version(RECORDS[:1], mod, "V7")
        finally: mod.build_evidence_graph, mod.build_uncertainty_assessment = real_build, real_unc
        self.assertTrue(seen and all(s == A.V5_COMPONENT_FLAGS and all(s.values()) for s in seen))
        self.assertTrue(unc, "real tasks.build_uncertainty_assessment was never invoked/returned")  # an exception there must fail here, not pass as unanswered
        n_v8 = len(unc)
        self.assertTrue(all(r["uncertainty_status"] in mod.UNC_STATES and isinstance(r["escalation_required"], bool) for r in unc))  # the real structure / keys V8 reads
        self.assertTrue(out["component_state"]["uncertainty"]["invoked"])
        self.assertFalse(v7["component_state"]["uncertainty"]["invoked"])
        self.assertEqual(len(unc), n_v8)  # the V7 run added no uncertainty call
        self.assertEqual(out["selected_version"], "V8")
        self.assertEqual(len(out["predictions"] or []) + len(out["unanswered"]), 1)
        for u in out["unanswered"]: self.assertTrue(u["reason"])
        for p in out["predictions"] or []:
            self.assertEqual((p["provenance"]["selected_version"], p["provenance"]["system"]), ("V8", "V2"))
            self.assertEqual([e["uncertainty_status"] for e in p["provenance"]["uncertainty"]], [r["uncertainty_status"] for r in unc])  # real returned key used
            self.assertEqual(p["predicted_escalation"], any(r["escalation_required"] for r in unc))
            for e in p["provenance"]["uncertainty"]: self.assertEqual(e["decision_confidence"]["value"], None)  # real layer never supplies a confidence
        self.assertEqual({k: getattr(mod, k) for k in keys}, before)
        self.assertIs(mod.build_uncertainty_assessment, real_unc)
        self.assertIsNone(A._ACTIVE_CONFIG["version"])


class V1Tests(unittest.TestCase):
    def test_no_invented_v1_parsing(self):
        tasks = FakeTasks(v1_text="Verdict: NON_COMPLIANT. COMPLIANT? INSUFFICIENT_EVIDENCE. d1 contradicts d2. Escalate.")
        out = A.run_paired(RECORDS[:3], tasks, run_v1=True)
        self.assertEqual(len(out["v1_traces"]), 3)
        self.assertTrue(all(t["status"] == "RAN" and t["raw_text"] == tasks.v1_text for t in out["v1_traces"]))  # raw text kept verbatim, inert
        self.assertIsNone(out["v1_predictions"])
        self.assertIsNone(A.v1_prediction_records(out["v1_traces"]))
        self.assertEqual(A.V1_STRUCTURED_FIELDS, ())
        c = compare_systems(RECORDS[:3], out["v1_predictions"], out["complete_predictions"])
        self.assertEqual(c["status"], "NOT_MEASURED")
        self.assertIn("v1", c["overall"]["metrics"]["decision_accuracy"])
        self.assertEqual(c["overall"]["metrics"]["decision_accuracy"]["v1"], NOT_MEASURED)
        with open(A.__file__, encoding="utf-8") as fh: src = fh.read()
        mods = {n.names[0].name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import)}
        self.assertFalse(mods & {"re", "json"})  # no regex / JSON machinery that could parse V1 text

    def test_v1_is_not_run_unless_requested_and_failures_are_reported(self):
        tasks = FakeTasks()
        A.run_paired(RECORDS[:2], tasks)
        self.assertFalse([c for c in tasks.calls if c[0] == "v1"])
        bad = types.SimpleNamespace(build_evidence_graph=lambda p, r: (_ for _ in ()).throw(ConnectionError("no LLM")))
        with tempfile.TemporaryDirectory() as wd: t = A.run_v1_trace(RECORDS[0], bad, wd)
        self.assertEqual((t["status"], t["raw_text"]), ("BLOCKED", None))
        self.assertIn("ConnectionError", t["error"])


class NoFabricationTests(unittest.TestCase):
    def test_blocked_environment_gives_not_measured_everywhere(self):
        out = A.run_paired(RECORDS, None, blocker="ModuleNotFoundError: No module named 'app'")
        c = out["comparison"]
        self.assertEqual((c["status"], c["paired"]["paired_count"], out["blocker"]), ("NOT_MEASURED", 0, "ModuleNotFoundError: No module named 'app'"))
        self.assertIsNone(out["complete_predictions"])
        self.assertTrue(all(m["v1"] == NOT_MEASURED and m["complete"] == NOT_MEASURED for m in c["overall"]["metrics"].values()))

    def test_failing_or_empty_pipeline_yields_no_prediction_not_a_guess(self):
        out = A.run_paired(RECORDS[:3], FakeTasks(fail_on=f"{RECORDS[1]['perturbed_document_set'][0]['doc_id']}.txt", n_decisions=1))
        failed = [u for u in out["complete_unanswered"] if "scripted failure" in u["reason"]]
        self.assertTrue(failed)
        self.assertTrue(set(u["perturbation_id"] for u in failed).isdisjoint(p["perturbation_id"] for p in out["complete_predictions"] or []))
        empty = A.run_paired(RECORDS[:2], FakeTasks(n_decisions=0))
        self.assertEqual(len(empty["complete_unanswered"]), 2)
        self.assertIsNone(empty["complete_predictions"])
        self.assertEqual(empty["comparison"]["status"], "NOT_MEASURED")

    def test_load_tasks_reports_exact_blocker_without_substituting(self):
        mod, why = A.load_tasks(os.path.join("no", "such", "tasks.py"), "definitely_missing_pkg.tasks")
        self.assertIsNone(mod)
        self.assertIn("ModuleNotFoundError", why)
        self.assertIn("tasks file not found", why)


class RealPipelineTests(unittest.TestCase):
    """Runs the REAL frozen pipeline only where tasks.py is importable. Asserts shape/compatibility only; prints and asserts no performance."""
    def test_real_complete_pipeline_on_a_few_records(self):
        mod, why = A.load_tasks(os.environ.get("OMNICHECK_TASKS_PATH"), os.environ.get("OMNICHECK_TASKS_MODULE", "app.services.module1_compliance.tasks"))
        if mod is None: self.skipTest(f"real tasks.py not importable here: {why}")
        preds, unanswered = A.predict_complete(RECORDS[:3], mod)
        res = evaluate_adversarial_predictions(RECORDS[:3], preds or None)
        self.assertNotIn(res["status"], ("INVALID_RECORDS", "INVALID_PREDICTIONS"))
        self.assertEqual(len(preds) + len(unanswered), 3)


if __name__ == "__main__":
    unittest.main()

class CompiledPolicyProofTests(unittest.TestCase):
    def test_graph_proof_requires_effective_compiled_decision_evidence(self):
        class FakeGraph:
            graph = {}
            def __init__(self):
                self._nodes = [
                    ("p1", {"type": "Policy", "compiled_policy": {
                        "status": "COMPILED_WITH_REVIEW",
                        "stats": {"valid": 2, "needs_review": 1, "rejected": 0},
                        "unmapped_rules": [],
                    }}),
                    ("r1", {"type": "PolicyRule", "compiled_rules": [{
                        "rule_id": "R-1", "status": "VALID", "mapping_basis": "clause_within_rule_line",
                        "result": {"executed": True, "verdict": "SATISFIED"},
                    }]}),
                    ("d1", {"type": "Decision", "evaluation_engine": "compiled_rule_engine", "verdict": "SATISFIED"}),
                    ("d2", {"type": "Decision", "evaluation_engine": "legacy_dsl", "verdict": "SATISFIED"}),
                ]
            def nodes(self, data=False):
                return self._nodes if data else [node_id for node_id, _ in self._nodes]
            def number_of_nodes(self):
                return len(self._nodes)

        proof = A._graph_component_proof(FakeGraph())
        self.assertEqual(proof["compiled_policy_summaries"][0]["status"], "COMPILED_WITH_REVIEW")
        self.assertEqual(proof["compiled_policy_execution"]["executed_rule_count"], 1)
        self.assertEqual(proof["compiled_policy_execution"]["compiled_decision_count"], 1)
        self.assertTrue(proof["compiled_policy_execution"]["effective_on_decision"])

    def test_graph_proof_does_not_treat_compilation_alone_as_effective(self):
        class FakeGraph:
            graph = {}
            def nodes(self, data=False):
                rows = [
                    ("p1", {"type": "Policy", "compiled_policy": {"status": "COMPILED_WITH_REVIEW", "stats": {"valid": 1}}}),
                    ("d1", {"type": "Decision", "evaluation_engine": "legacy_dsl", "verdict": "NO_CONCLUSION"}),
                ]
                return rows if data else [node_id for node_id, _ in rows]
            def number_of_nodes(self): return 2

        proof = A._graph_component_proof(FakeGraph())
        self.assertFalse(proof["compiled_policy_execution"]["effective_on_decision"])
        self.assertEqual(proof["compiled_policy_execution"]["compiled_decision_count"], 0)
