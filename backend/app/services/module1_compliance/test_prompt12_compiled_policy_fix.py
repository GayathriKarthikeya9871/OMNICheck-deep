"""Focused regression tests for Prompt 12 Phase 4 compiled-policy input semantics."""
import importlib
import copy
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_tasks():
    # Mirror the existing cross-document test's dependency stubs; no DB/provider calls occur.
    for name in ("celery", "dotenv", "app", "app.db", "app.db.database", "app.models", "app.models.domain"):
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
    class _Celery:
        def __init__(self, *args, **kwargs): pass
        def task(self, *args, **kwargs): return lambda f: f
    if not hasattr(sys.modules["celery"], "Celery"):
        sys.modules["celery"].Celery = _Celery
    if not hasattr(sys.modules["dotenv"], "load_dotenv"):
        sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None
    if not hasattr(sys.modules["app.db.database"], "SessionLocal"):
        sys.modules["app.db.database"].SessionLocal = None
    for name in ("InvestigationRecord", "EvidenceNode", "EvidenceEdge"):
        if not hasattr(sys.modules["app.models.domain"], name):
            setattr(sys.modules["app.models.domain"], name, object)
    pkg_name = "omnicheck_phase4_test_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(HERE)]
    sys.modules[pkg_name] = pkg
    return importlib.import_module(pkg_name + ".tasks")


T = _load_tasks()
P = importlib.import_module("omnicheck_phase4_test_pkg.policy_compiler")
S = importlib.import_module("omnicheck_phase4_test_pkg.policy_schema")

POLICY = "EXP-4.2: Any single expense above $2,000 must have written approval from a Director attached to the expense report before payment."
RAW_RULE = {
    "rule_type": "REQUIRE",
    "entity": "expense",
    "condition": {"entity": "expense", "field": "amount", "operator": ">", "value": 2000, "unit": "USD"},
    "temporal": {"kind": "before", "entity": "payment", "field": "date", "reference": "payment.date"},
    "required_evidence": [{"type": "written_approval", "description": "written approval from a Director attached to the expense report", "mandatory": True, "min_count": 1}],
    "exception": [], "severity": "high", "action": "attach_written_approval_before_payment", "confidence": 0.90,
    "source_text": "Any single expense above $2,000 must have written approval from a Director attached to the expense report before payment.",
    "ambiguities": ["Timing of \"before payment\" is interpreted as approval must exist prior to the expense payment date, but the policy does not define a specific field for approval timestamp."],
}


class CompilerNormalizationTests(unittest.TestCase):
    def test_explicit_approval_timing_becomes_evidence_match_not_payment_date_before_now(self):
        rule, errors = P.validate_raw_rule(RAW_RULE, "POL_TEST", 1, POLICY, 0.7)
        self.assertEqual(errors, [])
        self.assertIsNotNone(rule)
        self.assertIsNone(rule.temporal)
        self.assertEqual(rule.required_evidence[0].match, {"timing": "before_payment"})
        self.assertEqual(rule.status, "VALID")
        self.assertFalse(any("approval timestamp" in x.lower() for x in rule.ambiguities))
        self.assertTrue(any("represented as required_evidence.match.timing" in x for x in rule.issues))

    def test_legacy_now_reference_is_also_normalized_for_explicit_approval_timing(self):
        raw = copy.deepcopy(RAW_RULE)
        raw["temporal"]["reference"] = "now"
        rule, errors = P.validate_raw_rule(raw, "POL_TEST", 1, POLICY, 0.7)
        self.assertEqual(errors, [])
        self.assertIsNotNone(rule)
        self.assertIsNone(rule.temporal)
        self.assertEqual(rule.required_evidence[0].match, {"timing": "before_payment"})

    def test_payment_date_field_alias_is_normalized_too(self):
        raw = copy.deepcopy(RAW_RULE)
        raw["temporal"] = {"kind": "before", "entity": "expense", "field": "payment_date", "reference": "payment"}
        rule, errors = P.validate_raw_rule(raw, "POL_TEST", 1, POLICY, 0.7)
        self.assertEqual(errors, [])
        self.assertIsNotNone(rule)
        self.assertIsNone(rule.temporal)
        self.assertEqual(rule.required_evidence[0].match, {"timing": "before_payment"})

    def test_no_timing_inference_when_source_does_not_say_before_payment(self):
        raw = copy.deepcopy(RAW_RULE)
        raw["source_text"] = "Any single expense above $2,000 must have written approval attached."
        raw["temporal"] = None
        raw["ambiguities"] = []
        self.assertFalse(P._normalize_explicit_approval_timing(raw))
        self.assertIsNone(raw["required_evidence"][0].get("match"))


class EvidenceMappingTests(unittest.TestCase):
    def _compiled_rule(self):
        rule, errors = P.validate_raw_rule(RAW_RULE, "POL_TEST", 1, POLICY, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual(rule.status, "VALID")
        return rule

    def _graph(self, evidence_text):
        import networkx as nx
        G = nx.MultiDiGraph()
        G.add_node("doc1", type="Document", filename="expense_report.txt")
        G.add_node("ev1", type="Evidence", text=evidence_text, source_file="expense_report.txt", source_location="Lines 1-1", context_only=False)
        G.add_edge("ev1", "doc1", relation="DERIVED_FROM")
        return G

    def test_explicit_approval_and_timing_are_typed_from_source_evidence(self):
        G = self._graph("Expense report ER-101: Attachment: written approval from the Director of Finance, dated before payment.")
        inputs = T.graph_to_rule_inputs(G, [self._compiled_rule()])
        self.assertEqual(inputs["evidence"], [{"type": "written_approval", "evidence_id": "ev1", "file": "expense_report.txt", "location": "Lines 1-1", "timing": "before_payment"}])

    def test_negated_approval_is_not_typed_as_present(self):
        G = self._graph("Expense report ER-101: No written approval was attached before payment.")
        inputs = T.graph_to_rule_inputs(G, [self._compiled_rule()])
        self.assertIsNone(inputs["evidence"])

    def test_diagnostic_input_snapshot_recovers_attached_compiled_requirements(self):
        G = self._graph("Expense report ER-101: Attachment: written approval from the Director, dated before payment.")
        rule = self._compiled_rule()
        G.add_node("policy_rule", type="PolicyRule", compiled_rules=[
            {"compiled_rule": rule.model_dump(mode="json")}
        ])
        # Same call signature used by the trace utility: graph_to_rule_inputs(G), with no explicit rules.
        inputs = T.graph_to_rule_inputs(G)
        self.assertTrue(inputs["evidence"])
        self.assertEqual(inputs["evidence"][0]["type"], "written_approval")
        self.assertEqual(inputs["evidence"][0]["timing"], "before_payment")


class DerivedTriggerFactTests(unittest.TestCase):
    def test_confirmed_violation_false_is_derived_only_after_all_base_rules_are_determinate(self):
        Condition, Operator, RuleType = S.Condition, S.Operator, S.RuleType
        base = types.SimpleNamespace(rule_id="R1", entity="expense", rule_type=RuleType.REQUIRE, status="VALID")
        trigger = types.SimpleNamespace(rule_id="R2", entity="expense_violation", rule_type=RuleType.TRIGGER, status="VALID",
            condition=Condition(entity="expense_violation", field="confirmed", operator=Operator.EQ, value=True, unit=None))
        compiled = types.SimpleNamespace(rules=[base, trigger], rejected=[], unparsed_statements=[])
        facts = {"expense": [{"amount": {"value": 3400, "unit": "USD"}}]}
        result = T._derive_confirmed_violation_facts([base, trigger], [{"rule_id": "R1", "record_index": 0, "verdict": "COMPLIANT"}], compiled, facts)
        self.assertTrue(result)
        self.assertEqual(facts["expense_violation"], [{"confirmed": False}])

    def test_indeterminate_base_rule_does_not_create_confirmation_fact(self):
        Condition, Operator, RuleType = S.Condition, S.Operator, S.RuleType
        base = types.SimpleNamespace(rule_id="R1", entity="expense", rule_type=RuleType.REQUIRE, status="VALID")
        trigger = types.SimpleNamespace(rule_id="R2", entity="expense_violation", rule_type=RuleType.TRIGGER, status="VALID",
            condition=Condition(entity="expense_violation", field="confirmed", operator=Operator.EQ, value=True, unit=None))
        compiled = types.SimpleNamespace(rules=[base, trigger], rejected=[], unparsed_statements=[])
        facts = {"expense": [{"amount": {"value": 3400, "unit": "USD"}}]}
        result = T._derive_confirmed_violation_facts([base, trigger], [{"rule_id": "R1", "record_index": 0, "verdict": "INDETERMINATE"}], compiled, facts)
        self.assertFalse(result)
        self.assertNotIn("expense_violation", facts)


class ConfirmedTriggerNormalizationTests(unittest.TestCase):
    def test_live_trace_generic_violation_confirmed_form_is_normalized(self):
        rule = S.CompiledRule.model_validate({
            "policy_id": "POL_TEST", "rule_id": "R2", "rule_type": "TRIGGER", "entity": "violation",
            "condition": {"entity": "violation", "field": "confirmed", "operator": "==", "value": True},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "medium",
            "action": "escalate_to_finance_compliance", "confidence": 0.9,
            "source_text": "Confirmed violations must be escalated to Finance Compliance.", "ambiguities": [],
        })
        base = S.CompiledRule.model_validate({
            "policy_id": "POL_TEST", "rule_id": "R1", "rule_type": "REQUIRE", "entity": "expense",
            "condition": {"entity": "expense", "field": "amount", "operator": ">", "value": 2000, "unit": "USD"},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "high",
            "action": "attach_written_approval", "confidence": 0.9,
            "source_text": "Any single expense above $2,000 must have written approval before payment.", "ambiguities": [],
        })
        changed = T._normalize_confirmed_violation_trigger_aliases([base, rule])
        self.assertEqual(changed, ["R2"])
        self.assertEqual(rule.entity, "expense_violation")
        self.assertEqual(rule.condition.entity, "expense_violation")
        self.assertEqual(rule.condition.field, "confirmed")
        self.assertIs(rule.condition.value, True)

    def test_canonical_compiler_trigger_form_is_normalized(self):
        rule = S.CompiledRule.model_validate({
            "policy_id": "POL_TEST", "rule_id": "R2", "rule_type": "TRIGGER", "entity": "expense",
            "condition": {"entity": "expense", "field": "violation_confirmed", "operator": "==", "value": True},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "high",
            "action": "escalate_to_finance_compliance", "confidence": 0.9,
            "source_text": "Confirmed violations must be escalated to Finance Compliance.", "ambiguities": [],
        })
        base = S.CompiledRule.model_validate({
            "policy_id": "POL_TEST", "rule_id": "R1", "rule_type": "REQUIRE", "entity": "expense",
            "condition": {"entity": "expense", "field": "amount", "operator": ">", "value": 2000, "unit": "USD"},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "high",
            "action": "attach_written_approval", "confidence": 0.9,
            "source_text": "Any single expense above $2,000 must have written approval before payment.", "ambiguities": [],
        })
        changed = T._normalize_confirmed_violation_trigger_aliases([base, rule])
        self.assertEqual(changed, ["R2"])
        self.assertEqual(rule.entity, "expense_violation")
        self.assertEqual(rule.condition.entity, "expense_violation")
        self.assertEqual(rule.condition.field, "confirmed")
        self.assertIs(rule.condition.value, True)


class CompiledDecisionIntegrationTests(unittest.TestCase):
    def test_indeterminate_compiled_sibling_prevents_partial_compliant_decision(self):
        rd = {"compiled_rules": [
            {"result": {"executed": True, "verdict": "COMPLIANT", "counts": {"COMPLIANT": 1},
                        "violating_node_ids": [], "satisfying_node_ids": ["txn"], "reasons": []},
             "expression": "expense.amount > 2000", "action_required": None},
            {"result": {"executed": True, "verdict": "INDETERMINATE", "counts": {"INDETERMINATE": 1},
                        "violating_node_ids": [], "satisfying_node_ids": [], "reasons": ["missing fact"]},
             "expression": "violation.status == confirmed", "action_required": None},
        ]}
        self.assertIsNone(T._compiled_node_result(rd))

    def test_explicit_approval_propagates_through_real_compiler_adapter_and_decision_path(self):
        import networkx as nx
        from types import SimpleNamespace
        clause1 = "Any single expense above $2,000 must have written approval from a Director attached to the expense report before payment."
        clause2 = "Confirmed violations must be escalated to Finance Compliance."
        policy_text = "EXP-4.2: " + clause1 + " " + clause2
        raw1 = copy.deepcopy(RAW_RULE)
        rule1, errors = P.validate_raw_rule(raw1, "POL_TEST", 1, policy_text, 0.7)
        self.assertEqual(errors, [])
        raw2 = {
            # Match the actual Prompt 12 trace output, not just the alternate status alias.
            "rule_type": "TRIGGER", "entity": "violation",
            "condition": {"entity": "violation", "field": "confirmed", "operator": "==", "value": True, "unit": None},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "medium",
            "action": "escalate_to_finance_compliance", "confidence": 0.95, "source_text": clause2, "ambiguities": [],
        }
        rule2, errors = P.validate_raw_rule(raw2, "POL_TEST", 2, policy_text, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual((rule1.status, rule2.status), ("VALID", "VALID"))
        compiled = SimpleNamespace(policy_id="POL_TEST", rules=[rule1, rule2], status="COMPILED", stats={"extracted": 2, "valid": 2, "needs_review": 0},
                                   ambiguous_policy=False, ambiguity_reasons=[], rejected=[], unparsed_statements=[])

        G = nx.MultiDiGraph()
        G.add_node("policy_doc", type="Document", filename="policy_rulebook.txt", file_hash="rulebookhash")
        G.add_node("case_doc", type="Document", filename="expense_report.txt", file_hash="casedochash")
        G.add_node("policy", type="Policy", name="Investigation Framework", source_document_id="policy_doc", source_file="policy_rulebook.txt")
        G.add_node("policy_rule", type="PolicyRule", condition=policy_text, original_text=policy_text, source_file="policy_rulebook.txt", source_document_id="policy_doc", source_location="line 1")
        G.add_edge("policy_rule", "policy", relation="BELONGS_TO")
        G.add_node("txn", type="Transaction", amount=3400.0, currency="USD", amount_role=T.ROLE_TXN, attributes={})
        G.add_node("evidence", type="Evidence", text="Expense report ER-101: Attachment: written approval from the Director of Finance, dated before payment.",
                   source_file="expense_report.txt", source_location="Lines 1-1", context_only=False)
        G.add_edge("evidence", "case_doc", relation="DERIVED_FROM")
        G.add_edge("evidence", "txn", relation="SUPPORTS")

        old_flags = (T.COMPILED_POLICY_ENABLED, T.HAS_POLICY_COMPILER, T._compile_policy, T._COMPILER_MIN_CONFIDENCE)
        try:
            T.COMPILED_POLICY_ENABLED = True
            T.HAS_POLICY_COMPILER = True
            T._compile_policy = lambda *args, **kwargs: compiled
            T._COMPILER_MIN_CONFIDENCE = 0.7
            summary = T.apply_compiled_policy(G, policy_text)
            self.assertEqual(summary["status"], "COMPILED")
            self.assertTrue(summary["inputs"]["evidence_supplied"])
            self.assertEqual(summary["inputs"]["records"]["expense"], 1)
            self.assertEqual(summary["normalized_confirmed_violation_trigger_aliases"], ["POL_TEST-R002"])
            decisions = T.evaluate_policy_rules(G)
            self.assertEqual(len(decisions), 1)
            decision = G.nodes[decisions[0]]
            self.assertEqual(decision["evaluation_engine"], "compiled_rule_engine")
            self.assertEqual(decision["compiled_verdict"], "COMPLIANT")
            self.assertEqual(decision["verdict"], "SATISFIED")
            self.assertEqual(decision["result_source"], "compiled_policy_engine")
            attached = G.nodes["policy_rule"]["compiled_rules"]
            self.assertEqual([x["result"]["verdict"] for x in attached], ["COMPLIANT", "NOT_APPLICABLE"])
            diagnostic_inputs = T.graph_to_rule_inputs(G)
            self.assertTrue(diagnostic_inputs["evidence"])
            self.assertEqual(decision["compiled_verdict"], "COMPLIANT")
        finally:
            T.COMPILED_POLICY_ENABLED, T.HAS_POLICY_COMPILER, T._compile_policy, T._COMPILER_MIN_CONFIDENCE = old_flags


# ---------------------------------------------------------------------------------------------------------------------
# Prompt 12 (final): conditional-escalation typing, base-obligation partition, decision integration, cache key.
# These drive the REAL compiler validation (P.validate_raw_rule), the REAL tasks.py pipeline (apply_compiled_policy ->
# evaluate_policy_rules) and the REAL rule engine. Only the LLM text is replaced by raw rule dicts.
# ---------------------------------------------------------------------------------------------------------------------
import hashlib

R = importlib.import_module("omnicheck_phase4_test_pkg.rule_engine")
_C1 = "Any single expense above $2,000 must have written approval from a Director attached to the expense report before payment."
_C2 = "Confirmed violations must be escalated to Finance Compliance."
_POL = "EXP-4.2: " + _C1 + " " + _C2
_OK_EVIDENCE = "Expense report ER-101: Priya Nair purchased conference registration for $3,400. Attachment: written approval from the Director of Finance, dated before payment."
_IRRELEVANT = ["Facilities bulletin: the east stairwell will be repainted over the coming quarter.",
               "Social club note: the book club chose its next reading for the autumn meeting."]


def _raw_rules(r1_type="REQUIRE", r2_type="REQUIRE", amount_field="amount"):
    raw1 = copy.deepcopy(RAW_RULE)
    raw1.update(rule_type=r1_type, source_text=_C1)
    raw1["condition"]["field"] = amount_field  # the real compiler emitted "total_amount"; the graph exposes "amount"
    raw1["required_evidence"][0]["type"] = "written_approval_from_a_director"
    raw1.pop("severity")  # not stated by the policy: schema default applies
    raw2 = {"rule_type": r2_type, "entity": "violation", "temporal": None, "required_evidence": [], "exception": [],
            "condition": {"entity": "violation", "field": "confirmed", "operator": "==", "value": True, "unit": None},
            "action": "escalate_to_finance_compliance", "confidence": 0.95, "source_text": _C2, "ambiguities": []}
    return raw1, raw2


def _run_case(evidence_texts, r1_type="REQUIRE", r2_type="REQUIRE", with_txn=True, amount_field="amount", real_compile=False):
    """Compile (validation only) -> attach to a graph -> evaluate through the production path. Returns (summary, decision node, attached)."""
    import networkx as nx
    from types import SimpleNamespace
    raw1, raw2 = _raw_rules(r1_type, r2_type, amount_field)
    rule1, e1 = P.validate_raw_rule(raw1, "POL_T", 1, _POL, 0.7)
    rule2, e2 = P.validate_raw_rule(raw2, "POL_T", 2, _POL, 0.7)
    assert e1 == [] and e2 == [], (e1, e2)
    assert (rule1.status, rule2.status) == ("VALID", "VALID")
    compiled = SimpleNamespace(policy_id="POL_T", rules=[rule1, rule2], status="COMPILED", stats={"extracted": 2, "valid": 2, "needs_review": 0},
                               ambiguous_policy=False, ambiguity_reasons=[], rejected=[], unparsed_statements=[])
    G = nx.MultiDiGraph()
    G.add_node("policy_doc", type="Document", filename="policy_rulebook.txt", file_hash="rulebookhash")
    G.add_node("case_doc", type="Document", filename="expense_report.txt", file_hash="casedochash")
    G.add_node("policy", type="Policy", name="Investigation Framework", source_document_id="policy_doc", source_file="policy_rulebook.txt")
    G.add_node("policy_rule", type="PolicyRule", condition=_POL, original_text=_POL, source_file="policy_rulebook.txt", source_document_id="policy_doc", source_location="line 1")
    G.add_edge("policy_rule", "policy", relation="BELONGS_TO")
    if with_txn:
        G.add_node("txn", type="Transaction", amount=3400.0, currency="USD", amount_role=T.ROLE_TXN, attributes={})
    for i, text in enumerate(evidence_texts):
        G.add_node(f"ev{i}", type="Evidence", text=text, source_file="expense_report.txt", source_location=f"Lines {i + 1}-{i + 1}", context_only=False)
        G.add_edge(f"ev{i}", "case_doc", relation="DERIVED_FROM")
        if with_txn and i == 0:  # only the expense report supports the transaction; unrelated notes are not linked to it
            G.add_edge(f"ev{i}", "txn", relation="SUPPORTS")
    old = (T.COMPILED_POLICY_ENABLED, T.HAS_POLICY_COMPILER, T._compile_policy, T._COMPILER_MIN_CONFIDENCE)
    try:
        if real_compile:  # production compile_policy (prompt build, JSON parse, validation, normalization) with the LLM text injected
            import json
            llm_text = json.dumps({"rules": [raw1, raw2], "unparsed_statements": []})
            compile_fn = lambda text, pid=None, minc=0.7, **k: P.compile_policy(text, pid, minc, llm_fn=lambda system, user: llm_text)
        else:
            compile_fn = lambda *a, **k: compiled
        T.COMPILED_POLICY_ENABLED, T.HAS_POLICY_COMPILER, T._compile_policy, T._COMPILER_MIN_CONFIDENCE = True, True, compile_fn, 0.7
        summary = T.apply_compiled_policy(G, _POL)
        decisions = T.evaluate_policy_rules(G)
        assert len(decisions) == 1
        return summary, G.nodes[decisions[0]], G.nodes["policy_rule"]["compiled_rules"]
    finally:
        T.COMPILED_POLICY_ENABLED, T.HAS_POLICY_COMPILER, T._compile_policy, T._COMPILER_MIN_CONFIDENCE = old


# the four type combinations the compiler may emit for the two clauses (the real trace emitted REQUIRE/REQUIRE)
_COMBOS = [("REQUIRE", "REQUIRE"), ("REQUIRE", "TRIGGER"), ("TRIGGER", "TRIGGER"), ("TRIGGER", "REQUIRE")]


class ConfirmedViolationRuleTypeTests(unittest.TestCase):
    def test_real_trace_require_form_is_retyped_to_trigger_with_audit_note(self):
        _, raw2 = _raw_rules("REQUIRE", "REQUIRE")
        rule, errors = P.validate_raw_rule(raw2, "POL_T", 2, _POL, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual(rule.rule_type.value, "TRIGGER")
        self.assertTrue(any("REQUIRE -> TRIGGER" in x for x in rule.issues))

    def test_status_alias_form_is_retyped_too(self):
        _, raw2 = _raw_rules("REQUIRE", "REQUIRE")
        raw2["condition"] = {"entity": "violation", "field": "status", "operator": "==", "value": "confirmed", "unit": None}
        rule, errors = P.validate_raw_rule(raw2, "POL_T", 2, _POL, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual(rule.rule_type.value, "TRIGGER")

    def test_other_require_rules_are_never_retyped(self):
        raw1, _ = _raw_rules("REQUIRE", "REQUIRE")
        self.assertFalse(P._normalize_confirmed_violation_rule_type(raw1))
        self.assertEqual(raw1["rule_type"], "REQUIRE")
        # confirmation flag but the clause does not say "confirmed violation(s)" -> untouched
        _, raw2 = _raw_rules("REQUIRE", "REQUIRE")
        raw2["source_text"] = "Violations must be escalated to Finance Compliance."
        self.assertFalse(P._normalize_confirmed_violation_rule_type(raw2))
        # extra predicate in the condition -> untouched
        _, raw3 = _raw_rules("REQUIRE", "REQUIRE")
        raw3["condition"] = {"logic": "AND", "children": [raw3["condition"], {"entity": "violation", "field": "severity", "operator": "==", "value": "high", "unit": None}]}
        self.assertFalse(P._normalize_confirmed_violation_rule_type(raw3))
        # no action stated -> untouched
        _, raw4 = _raw_rules("REQUIRE", "REQUIRE")
        raw4["action"] = ""
        self.assertFalse(P._normalize_confirmed_violation_rule_type(raw4))


class EndToEndCompiledDecisionTests(unittest.TestCase):
    def test_1_valid_approval_is_compliant_without_escalation_for_every_type_combination(self):
        for combo in _COMBOS:
            with self.subTest(combo=combo):
                summary, decision, attached = _run_case([_OK_EVIDENCE], *combo)
                self.assertEqual([x["result"]["verdict"] for x in attached], ["COMPLIANT", "NOT_APPLICABLE"])
                self.assertEqual([x["result"]["missing_facts"] for x in attached], [[], []])
                self.assertEqual([x["result"]["action_required"] for x in attached], [None, None])  # no escalation
                self.assertEqual(summary["normalized_confirmed_violation_trigger_aliases"], ["POL_T-R002"])
                self.assertEqual(attached[0]["result"]["satisfying_node_ids"], ["txn"])

    def test_10_compiled_decision_is_authoritative_and_has_provenance(self):
        for combo in _COMBOS:
            with self.subTest(combo=combo):
                _, d, attached = _run_case([_OK_EVIDENCE], *combo)
                self.assertEqual(d["evaluation_engine"], "compiled_rule_engine")
                self.assertEqual(d["result_source"], "compiled_policy_engine")
                self.assertEqual(d["compiled_verdict"], "COMPLIANT")
                self.assertEqual(d["verdict"], "SATISFIED")
                self.assertEqual(d["legacy_verdict"], "UNEVALUATED")  # legacy DSL could not decide; it did not produce this verdict
                self.assertEqual(d["compiled_missing_facts"], [])
                self.assertEqual(d["supporting_evidence_ids"], ["ev0"])
                self.assertEqual(attached[0]["compiled_rule"]["required_evidence"][0]["type"], "written_approval_from_a_director")
                self.assertEqual(attached[0]["compiled_rule"]["required_evidence"][0]["match"], {"timing": "before_payment"})
                self.assertEqual(attached[0]["severity"], "medium")  # omitted by the policy -> schema default, not an LLM guess

    def test_9_unrelated_documents_do_not_change_the_decision(self):
        _, base, _ = _run_case([_OK_EVIDENCE])
        _, noisy, attached = _run_case([_OK_EVIDENCE] + _IRRELEVANT)
        self.assertEqual((noisy["verdict"], noisy["compiled_verdict"], noisy["evaluation_engine"]), (base["verdict"], base["compiled_verdict"], base["evaluation_engine"]))
        self.assertEqual(noisy["supporting_evidence_ids"], ["ev0"])

    def _assert_not_compliant_and_not_fabricated(self, evidence_text):
        for combo in _COMBOS:
            with self.subTest(combo=combo):
                _, d, attached = _run_case([evidence_text], *combo)
                self.assertNotEqual(d["verdict"], "SATISFIED")
                self.assertNotEqual(d["compiled_verdict"], "COMPLIANT")
                # nothing is silently turned into "no violation": the escalation condition stays unknown, evidence stays unsupplied
                self.assertEqual(attached[0]["result"]["verdict"], "INDETERMINATE")
                self.assertEqual(attached[1]["result"]["verdict"], "INDETERMINATE")
                self.assertEqual(attached[1]["result"]["missing_facts"], ["expense_violation.confirmed"])
                self.assertIsNone(attached[1]["result"]["action_required"])

    def test_2_missing_approval_is_not_compliant_and_stays_indeterminate(self):
        self._assert_not_compliant_and_not_fabricated("Expense report ER-101: Priya Nair purchased conference registration for $3,400.")

    def test_3_approval_after_payment_is_not_compliant(self):
        self._assert_not_compliant_and_not_fabricated("Expense report ER-101: purchased for $3,400. Attachment: written approval from the Director of Finance, dated after payment.")

    def test_4_wrong_approver_is_not_compliant(self):
        self._assert_not_compliant_and_not_fabricated("Expense report ER-101: purchased for $3,400. Attachment: written approval from the Manager, dated before payment. The Director of Finance was copied.")

    def test_5_explicit_negation_is_not_approval(self):
        self._assert_not_compliant_and_not_fabricated("Expense report ER-101: purchased for $3,400. No written approval from the Director of Finance was attached, dated before payment.")

    def test_6_missing_amount_fact_stays_indeterminate(self):
        _, d, attached = _run_case([_OK_EVIDENCE], with_txn=False)
        self.assertNotEqual(d["verdict"], "SATISFIED")
        self.assertNotEqual(attached[0]["result"]["verdict"], "COMPLIANT")


class ConditionalEscalationSemanticsTests(unittest.TestCase):
    def _escalation_rule(self):
        return S.CompiledRule.model_validate({
            "policy_id": "POL_T", "rule_id": "R2", "rule_type": "TRIGGER", "entity": "expense_violation",
            "condition": {"entity": "expense_violation", "field": "confirmed", "operator": "==", "value": True, "unit": None},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "medium", "action": "escalate_to_finance_compliance",
            "confidence": 0.9, "source_text": _C2, "ambiguities": []})

    def _derive(self, base_type, verdict):
        base_raw, _ = _raw_rules(base_type, "TRIGGER")
        base, errors = P.validate_raw_rule(base_raw, "POL_T", 1, _POL, 0.7)
        self.assertEqual(errors, [])
        trig = self._escalation_rule()
        compiled = types.SimpleNamespace(rules=[base, trig], rejected=[], unparsed_statements=[])
        facts = {"expense": [{"amount": {"value": 3400, "unit": "USD"}}]}
        derived = T._derive_confirmed_violation_facts([base, trig], [{"rule_id": base.rule_id, "record_index": 0, "verdict": verdict}], compiled, facts)
        return derived, facts, trig

    def test_8_confirmed_violation_activates_finance_compliance_escalation(self):
        for base_type in ("REQUIRE", "TRIGGER"):
            with self.subTest(base_type=base_type):
                derived, facts, trig = self._derive(base_type, "VIOLATION")
                self.assertTrue(derived)
                self.assertEqual(facts["expense_violation"], [{"confirmed": True}])
                out = R.evaluate_policy([trig], facts, None)
                row = out["rule_results"][0]
                self.assertEqual(row["verdict"], "ACTION_REQUIRED")
                self.assertEqual(row["action_required"], "escalate_to_finance_compliance")
                self.assertEqual(out["decision"], "ACTION_REQUIRED")

    def test_7_false_trigger_condition_is_not_blocking_and_does_not_escalate(self):
        for base_type in ("REQUIRE", "TRIGGER"):
            with self.subTest(base_type=base_type):
                derived, facts, trig = self._derive(base_type, "COMPLIANT")
                self.assertTrue(derived)
                self.assertEqual(facts["expense_violation"], [{"confirmed": False}])
                row = R.evaluate_policy([trig], facts, None)["rule_results"][0]
                self.assertEqual((row["verdict"], row["action_required"], row["missing_facts"]), ("NOT_APPLICABLE", None, []))

    def test_indeterminate_base_never_becomes_false_for_either_base_type(self):
        for base_type in ("REQUIRE", "TRIGGER"):
            with self.subTest(base_type=base_type):
                derived, facts, _ = self._derive(base_type, "INDETERMINATE")
                self.assertFalse(derived)
                self.assertNotIn("expense_violation", facts)

    def test_base_obligation_partition(self):
        raw1, raw2 = _raw_rules("TRIGGER", "TRIGGER")
        t_with_evidence, _ = P.validate_raw_rule(raw1, "POL_T", 1, _POL, 0.7)
        escalation, _ = P.validate_raw_rule(raw2, "POL_T", 2, _POL, 0.7)
        self.assertTrue(T._is_base_obligation_rule(t_with_evidence))   # evidence-bearing trigger yields COMPLIANT/VIOLATION/NOT_APPLICABLE
        self.assertFalse(T._is_base_obligation_rule(escalation))       # escalation trigger is never a base obligation
        action_only = copy.deepcopy(raw1); action_only["required_evidence"] = []
        rule, _ = P.validate_raw_rule(action_only, "POL_T", 3, _POL, 0.7)
        self.assertFalse(T._is_base_obligation_rule(rule))              # action-only triggers keep their old (non-base) treatment


class SchemaAndCacheTests(unittest.TestCase):
    def test_11_invalid_rule_type_and_severity_are_rejected_not_coerced(self):
        raw1, _ = _raw_rules("REQUIRE", "REQUIRE")
        bad_type = dict(raw1, rule_type="ENFORCE")
        rule, errors = P.validate_raw_rule(bad_type, "POL_T", 1, _POL, 0.7)
        self.assertIsNone(rule)
        self.assertTrue(errors and errors[0].startswith("rule_type:"))
        for bad in ("High", None, 3, "severe"):
            rule, errors = P.validate_raw_rule(dict(raw1, severity=bad), "POL_T", 1, _POL, 0.7)
            self.assertIsNone(rule, bad)
            self.assertTrue(errors[0].startswith("severity: Input should be") and "received" in errors[0], errors)
        # a valid severity is preserved only when the source clause states it (see SeverityGroundingTests)
        crit_clause = _C1[:-1] + " (severity: critical)."
        rule, errors = P.validate_raw_rule(dict(raw1, severity="critical", source_text=crit_clause), "POL_T", 1, "EXP-4.2: " + crit_clause, 0.7)
        self.assertEqual((errors, rule.severity.value), ([], "critical"))

    def test_12_prompt_change_or_version_change_cannot_reuse_a_cached_response(self):
        key = lambda system, user: hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()  # same keying as the trace runner / adapter cache
        user = P._build_user_prompt(_POL)
        self.assertTrue(P.SYSTEM_PROMPT.rstrip().endswith("COMPILER_VERSION: " + P.COMPILER_VERSION))
        bumped = P.SYSTEM_PROMPT.replace("COMPILER_VERSION: " + P.COMPILER_VERSION, "COMPILER_VERSION: next")
        self.assertNotEqual(key(P.SYSTEM_PROMPT, user), key(bumped, user))
        old_prompt = P.SYSTEM_PROMPT.split("Conditional obligations:")[0]  # a compiler without the conditional-obligation guidance
        self.assertNotEqual(key(P.SYSTEM_PROMPT, user), key(old_prompt, user))
        for needle in ("Conditional obligations:", "a REQUIRE rule is a VIOLATION whenever its condition is false", "omit the key so the schema default applies"):
            self.assertIn(needle, P.SYSTEM_PROMPT)


# ---------------------------------------------------------------------------------------------------------------------
# Prompt 12 compiler defects: (1) confirmation-form constraints, (2) severity grounding. Compiler/schema only.
# ---------------------------------------------------------------------------------------------------------------------
def _confirm_raw(condition, **over):
    raw = {"rule_type": "REQUIRE", "entity": "violation", "temporal": None, "required_evidence": [], "exception": [],
           "condition": condition, "action": "escalate_to_finance_compliance", "confidence": 0.95, "source_text": _C2, "ambiguities": []}
    raw.update(over)
    return raw


def _leaf(entity, field, op="==", value=True, **extra):
    return dict({"entity": entity, "field": field, "operator": op, "value": value, "unit": None}, **extra)


class ConfirmationFormAcceptanceTests(unittest.TestCase):
    def test_recognized_forms_are_retyped_with_audit_note_and_provenance(self):
        forms = [
            (_leaf("violation", "confirmed"), "violation"),
            (_leaf("violation", "violation_confirmed"), "violation"),
            (_leaf("violation", "status", value="confirmed"), "violation"),
            (_leaf("violation", "state", value=" Confirmed "), "violation"),
            (_leaf("expense_violation", "confirmed"), "expense_violation"),
            (_leaf("expense_violation", "status", value="confirmed"), "expense_violation"),
            (_leaf("expense", "violation_confirmed"), "expense"),  # base-entity alias: the field itself says 'violation'
        ]
        for cond, top in forms:
            with self.subTest(cond=cond):
                rule, errors = P.validate_raw_rule(_confirm_raw(cond, entity=top), "POL_T", 2, _POL, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual(rule.rule_type.value, "TRIGGER")
                self.assertEqual(rule.source_text, _C2)  # provenance untouched
                self.assertEqual(rule.action, "escalate_to_finance_compliance")
                self.assertEqual(sum("rule_type normalized REQUIRE -> TRIGGER" in x for x in rule.issues), 1)

    def test_omitted_top_level_entity_is_allowed(self):
        raw = _confirm_raw(_leaf("violation", "confirmed"))
        del raw["entity"]
        self.assertTrue(P._normalize_confirmed_violation_rule_type(raw))


class ConfirmationUnrelatedAndMalformedTests(unittest.TestCase):
    def _assert_untouched(self, raw):
        before = copy.deepcopy(raw)
        self.assertFalse(P._normalize_confirmed_violation_rule_type(raw))
        self.assertEqual(raw, before)  # nothing mutated
        rule, errors = P.validate_raw_rule(before, "POL_T", 2, _POL, 0.7)
        if rule is not None:
            self.assertEqual(rule.rule_type.value, str(before["rule_type"]).upper())  # type unchanged
            self.assertFalse(any("REQUIRE -> TRIGGER" in x for x in rule.issues))

    def test_unrelated_entities_are_not_retyped(self):
        for cond in (_leaf("employee", "confirmed"), _leaf("expense", "confirmed"),
                     _leaf("employee", "status", value="confirmed"), _leaf("invoice", "state", value="confirmed")):
            with self.subTest(cond=cond):
                self._assert_untouched(_confirm_raw(cond, entity=cond["entity"]))

    def test_top_level_entity_must_agree_with_condition_entity(self):
        self._assert_untouched(_confirm_raw(_leaf("violation", "confirmed"), entity="employee"))
        self._assert_untouched(_confirm_raw(_leaf("expense", "violation_confirmed"), entity="violation"))

    def test_malformed_leaf_forms_are_not_retyped(self):
        bad = [
            _leaf("violation", "confirmed", op="!="),
            _leaf("violation", "confirmed", value=False),
            _leaf("violation", "confirmed", value="true"),
            _leaf("violation", "confirmed", value=1),
            _leaf("violation", "status", value="pending"),
            _leaf("violation", "status", value=True),
            _leaf("violation", "confirmed", op="exists", value=None),
            _leaf("violation", "severity", value="confirmed"),
            _leaf("violation", "confirmed", unit="USD"),
            _leaf("violation", "confirmed", inclusive=False),
            {"entity": "violation", "field": "confirmed", "operator": "==", "value": True, "extra": 1},
            {"entity": None, "field": "confirmed", "operator": "==", "value": True},
            {"entity": "violation", "field": 7, "operator": "==", "value": True},
        ]
        for cond in bad:
            with self.subTest(cond=cond):
                self._assert_untouched(_confirm_raw(cond))

    def test_extra_predicates_groups_and_other_clause_shapes_are_not_retyped(self):
        conf = _leaf("violation", "confirmed")
        self._assert_untouched(_confirm_raw({"logic": "AND", "children": [conf, _leaf("violation", "severity", value="high")]}))
        self._assert_untouched(_confirm_raw({"logic": "NOT", "children": [conf]}))
        self._assert_untouched(_confirm_raw(conf, temporal={"kind": "after", "entity": "violation", "field": "date", "reference": "now"}))
        self._assert_untouched(_confirm_raw(conf, source_text="Violations must be escalated to Finance Compliance."))
        self._assert_untouched(_confirm_raw(conf, action=""))
        self._assert_untouched(_confirm_raw(None))
        self._assert_untouched(_confirm_raw("violation.confirmed == true"))
        self._assert_untouched(_confirm_raw(conf, rule_type="PROHIBIT"))


class OrdinaryRequireRulesPreservedTests(unittest.TestCase):
    def test_threshold_require_even_when_clause_mentions_confirmed_violations(self):
        clause = "Confirmed violations above 500 USD must be escalated to Finance Compliance."
        raw = _confirm_raw(_leaf("violation", "amount", op=">", value=500, unit="USD"), source_text=clause)
        self.assertFalse(P._normalize_confirmed_violation_rule_type(raw))
        rule, errors = P.validate_raw_rule(raw, "POL_T", 1, clause, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual(rule.rule_type.value, "REQUIRE")
        self.assertFalse(any("REQUIRE -> TRIGGER" in x for x in rule.issues))

    def test_threshold_obligation_with_evidence_stays_require(self):
        raw1, _ = _raw_rules("REQUIRE", "REQUIRE")
        rule, errors = P.validate_raw_rule(raw1, "POL_T", 1, _POL, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual(rule.rule_type.value, "REQUIRE")
        self.assertEqual(rule.condition.operator.value, ">")
        self.assertEqual(rule.required_evidence[0].type, "written_approval_from_a_director")

    def test_property_require_stays_require(self):
        clause = "Passwords must be at least 12 characters."
        raw = {"rule_type": "REQUIRE", "entity": "password", "temporal": None, "required_evidence": [], "exception": [],
               "condition": _leaf("password", "length", op=">=", value=12, unit="characters"), "action": "enforce_password_length",
               "confidence": 0.95, "source_text": clause, "ambiguities": []}
        rule, errors = P.validate_raw_rule(raw, "POL_T", 1, clause, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual(rule.rule_type.value, "REQUIRE")


class SeverityGroundingTests(unittest.TestCase):
    def _compile(self, severity="__omit__", clause=_C1, policy=None):
        raw1, _ = _raw_rules("REQUIRE", "REQUIRE")  # raw1 carries no severity key
        raw1["source_text"] = clause
        if severity != "__omit__":
            raw1["severity"] = severity
        return P.validate_raw_rule(raw1, "POL_T", 1, policy or ("EXP-4.2: " + clause), 0.7)

    def test_schema_default_is_unchanged(self):
        self.assertEqual(S.CompiledRule.model_fields["severity"].default, S.Severity.MEDIUM)
        self.assertEqual(S.CompiledRule.model_fields["severity_source"].default, "schema_default")

    def test_omitted_severity_uses_schema_default_and_says_so(self):
        rule, errors = self._compile()
        self.assertEqual(errors, [])
        self.assertEqual((rule.severity.value, rule.severity_source, rule.status), ("medium", "schema_default", "VALID"))
        self.assertFalse(any("severity" in x for x in rule.issues + rule.ambiguities))

    def test_unsupported_inferred_severity_is_discarded_not_accepted_as_policy_fact(self):
        for level in ("low", "medium", "high", "critical"):
            with self.subTest(level=level):
                rule, errors = self._compile(severity=level)
                self.assertEqual(errors, [])
                self.assertEqual((rule.severity.value, rule.severity_source), ("medium", "schema_default"))
                self.assertTrue(any(f"returned severity '{level}' discarded" in x for x in rule.issues), rule.issues)
                self.assertEqual(rule.status, "VALID")  # discard is auditable via issues; wording was not ambiguous

    def test_unsupported_severity_from_unverified_quote_is_not_trusted(self):
        clause = _C1[:-1] + " (severity: critical)."
        rule, _ = self._compile(severity="critical", clause=clause, policy="EXP-4.2: " + _C1)  # source_text is not in the policy
        self.assertEqual((rule.severity.value, rule.severity_source), ("medium", "schema_default"))
        self.assertEqual(rule.status, "NEEDS_REVIEW")  # existing grounding check still flags the non-quote

    def test_explicitly_stated_severity_is_preserved_and_marked_policy_stated(self):
        cases = [("high severity", "high"), ("(severity: critical)", "critical"), ("(priority: low)", "low"),
                 ("classified as medium", "medium"), ("critical-severity", "critical")]
        for phrase, level in cases:
            with self.subTest(phrase=phrase):
                clause = _C1[:-1] + f" ({phrase})." if "(" not in phrase else _C1[:-1] + f" {phrase}."
                rule, errors = self._compile(severity=level, clause=clause)
                self.assertEqual(errors, [])
                self.assertEqual((rule.severity.value, rule.severity_source, rule.status), (level, "policy_stated", "VALID"), rule.ambiguities)

    def test_returned_severity_contradicting_the_policy_is_discarded_and_sent_to_review(self):
        clause = _C1[:-1] + " (severity: critical)."
        rule, errors = self._compile(severity="low", clause=clause)
        self.assertEqual(errors, [])
        self.assertEqual((rule.severity.value, rule.severity_source, rule.status), ("medium", "schema_default", "NEEDS_REVIEW"))
        self.assertTrue(any("clause states 'critical'" in x for x in rule.ambiguities))

    def test_ambiguous_wording_uses_existing_review_mechanism(self):
        clause = _C1[:-1] + "; a breach is a major finding."
        rule, errors = self._compile(severity="high", clause=clause)
        self.assertEqual(errors, [])
        self.assertEqual((rule.severity.value, rule.severity_source, rule.status), ("high", "ambiguous", "NEEDS_REVIEW"))
        self.assertLessEqual(rule.confidence, P.CONFIDENCE_CAP_WITH_ISSUES)
        self.assertTrue(any("does not state a severity level outright" in x for x in rule.ambiguities))
        # same wording but the returned level does not fit the cue -> unsupported -> default
        rule, _ = self._compile(severity="low", clause=clause)
        self.assertEqual((rule.severity.value, rule.severity_source), ("medium", "schema_default"))

    def test_conflicting_stated_levels_go_to_review(self):
        clause = _C1[:-1] + " (high severity, or low severity if waived)."
        rule, _ = self._compile(severity="high", clause=clause)
        self.assertEqual((rule.severity_source, rule.status), ("ambiguous", "NEEDS_REVIEW"))

    def test_invalid_severity_values_are_rejected_never_coerced_or_dropped(self):
        stated = _C1[:-1] + " (severity: urgent)."
        for bad in ("urgent", "High", "HIGH", "", " high", None, 3, ["high"], True):
            for clause in (_C1, stated):
                with self.subTest(bad=bad, clause=clause[-20:]):
                    rule, errors = self._compile(severity=bad, clause=clause)
                    self.assertIsNone(rule)
                    self.assertTrue(errors and errors[0].startswith("severity:"), errors)

    def test_direct_schema_construction_distinguishes_default_from_supplied(self):
        base = {"policy_id": "P", "rule_id": "R1", "rule_type": "REQUIRE", "entity": "expense", "action": "do_the_thing", "confidence": 0.9,
                "condition": {"entity": "expense", "field": "amount", "operator": ">", "value": 1, "unit": "USD"}}
        self.assertEqual(S.CompiledRule.model_validate(base).severity_source, "schema_default")
        r = S.CompiledRule.model_validate(dict(base, severity="high"))
        self.assertEqual((r.severity.value, r.severity_source), ("high", "unverified"))


# ---------------------------------------------------------------------------------------------------------------------
# Prompt consistency for severity, and severity_source integration at the real pipeline boundaries.
# ---------------------------------------------------------------------------------------------------------------------
import json
import re

_SERVER_CLAUSE = "Servers storing customer data must not be reachable from the public internet."


def _prompt_examples():
    """(source text, rule dict) for every worked example embedded in the compiler system prompt."""
    found = re.findall(r'^Text: "(.+)"\n-> (\{.*\})$', P.SYSTEM_PROMPT, re.M)
    return [(text, json.loads(raw)) for text, raw in found]


class PromptSeverityConsistencyTests(unittest.TestCase):
    def test_prompt_examples_are_found(self):
        self.assertGreaterEqual(len(_prompt_examples()), 3)

    def test_examples_do_not_teach_unsupported_severity(self):
        for text, rule in _prompt_examples():
            with self.subTest(text=text[:50]):
                states_one = any(rx.search(text) for rx in P._SEV_CONTEXT_RES)
                self.assertEqual("severity" in rule, states_one,
                                 "an example may carry a severity key if and only if its own source text states a severity")
                if "severity" in rule:
                    verdict = P._assess_severity(dict(rule), text)
                    self.assertEqual((verdict["source"], verdict["drop"]), ("policy_stated", False))

    def test_examples_validate_through_the_real_validator_with_correct_severity_source(self):
        for text, rule in _prompt_examples():
            with self.subTest(text=text[:50]):
                compiled, errors = P.validate_raw_rule(rule, "POL_EX", 1, text, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual(compiled.status, "VALID", compiled.ambiguities)
                if "severity" in rule:
                    self.assertEqual((compiled.severity.value, compiled.severity_source), (rule["severity"], "policy_stated"))
                else:
                    self.assertEqual((compiled.severity, compiled.severity_source), (S.Severity.MEDIUM, "schema_default"))
                    self.assertFalse(any("severity" in x for x in compiled.issues))

    def test_output_schema_presents_severity_as_optional(self):
        lines = [l for l in P.SYSTEM_PROMPT.splitlines() if l.startswith('  "severity":')]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("OPTIONAL", lines[0])
        self.assertIn("omit it unless the policy text itself explicitly states", lines[0])
        self.assertNotIn('"severity": "low|medium|high|critical",', P.SYSTEM_PROMPT)  # the old unconditional field line

    def test_severity_instruction_is_explicit_and_consistent(self):
        for needle in ('Severity (optional key): include "severity" ONLY when the policy text itself explicitly states',
                       "omit the key so the schema default applies",
                       'is NOT a severity classification',
                       "Never infer or guess a severity"):
            self.assertIn(needle, P.SYSTEM_PROMPT)
        self.assertNotIn('for example "critical", "minor"', P.SYSTEM_PROMPT)  # bare label words were presented as explicit severity

    def test_every_form_the_prompt_calls_explicit_is_recognised_by_the_validator(self):
        for phrase, level in (("high severity", "high"), ("severity: critical", "critical"), ("priority: low", "low"), ("classified as medium", "medium")):
            with self.subTest(phrase=phrase):
                self.assertIn(f'"{phrase}"', P.SYSTEM_PROMPT)
                clause = f"Backups must be encrypted ({phrase})."
                self.assertEqual(P._assess_severity({"severity": level, "source_text": clause}, clause)["source"], "policy_stated")

    def test_version_was_bumped_for_this_prompt_change(self):
        self.assertNotEqual(P.COMPILER_VERSION, "2026-10-10.3-confirmation-entity-severity-grounding")
        self.assertTrue(P.SYSTEM_PROMPT.rstrip().endswith("COMPILER_VERSION: " + P.COMPILER_VERSION))

    def test_old_example_behaviour_is_still_discarded_by_the_validator(self):
        # the previous prompt showed "high" for this clause although the clause states no severity
        server = next(rule for text, rule in _prompt_examples() if text == _SERVER_CLAUSE)
        compiled, errors = P.validate_raw_rule(dict(server, severity="high"), "POL_EX", 1, _SERVER_CLAUSE, 0.7)
        self.assertEqual(errors, [])
        self.assertEqual((compiled.severity, compiled.severity_source), (S.Severity.MEDIUM, "schema_default"))
        self.assertTrue(any("returned severity 'high' discarded" in x for x in compiled.issues))

    def test_ordinary_critical_wording_is_not_an_explicit_severity(self):
        server = next(rule for text, rule in _prompt_examples() if text == _SERVER_CLAUSE)
        clause = "Servers hosting critical systems must not be reachable from the public internet."
        compiled, _ = P.validate_raw_rule(dict(server, severity="critical", source_text=clause), "POL_EX", 1, clause, 0.7)
        self.assertNotEqual(compiled.severity_source, "policy_stated")
        self.assertEqual((compiled.severity_source, compiled.status), ("ambiguous", "NEEDS_REVIEW"))
        compiled, _ = P.validate_raw_rule(dict(server, severity="low", source_text=clause), "POL_EX", 1, clause, 0.7)
        self.assertEqual((compiled.severity, compiled.severity_source), (S.Severity.MEDIUM, "schema_default"))


def _rules_for_severity_source():
    """One compiled rule per severity_source outcome, produced by the real validator."""
    raw1, _ = _raw_rules("REQUIRE", "REQUIRE")
    out = {}
    out["schema_default"], _ = P.validate_raw_rule(copy.deepcopy(raw1), "POL_T", 1, _POL, 0.7)
    stated = _C1[:-1] + " (severity: critical)."
    out["policy_stated"], _ = P.validate_raw_rule(dict(copy.deepcopy(raw1), severity="critical", source_text=stated), "POL_T", 1, "EXP-4.2: " + stated, 0.7)
    vague = _C1[:-1] + "; a breach is a major finding."
    out["ambiguous"], _ = P.validate_raw_rule(dict(copy.deepcopy(raw1), severity="high", source_text=vague), "POL_T", 1, "EXP-4.2: " + vague, 0.7)
    for key, rule in out.items():
        assert rule is not None and rule.severity_source == key, (key, rule)
    return out


class SeveritySourceIntegrationTests(unittest.TestCase):
    """severity_source is DIAGNOSTIC metadata: the compiler decides it, it is persisted with the rule dump and revalidated
    by tasks.py, and neither tasks.py nor rule_engine.py reads it to decide anything."""

    def test_serialization_round_trip_preserves_value_and_source(self):
        for key, rule in _rules_for_severity_source().items():
            with self.subTest(source=key):
                dump = json.loads(json.dumps(rule.model_dump(mode="json")))  # graph storage is JSON-like
                self.assertEqual(dump["severity_source"], key)
                back = S.CompiledRule.model_validate(dump)  # same call tasks.graph_to_rule_inputs uses for stored rules
                self.assertEqual((back.severity, back.severity_source), (rule.severity, key))

    def test_stored_dump_without_the_field_is_accepted_but_never_upgraded(self):
        for key, rule in _rules_for_severity_source().items():
            with self.subTest(source=key):
                dump = rule.model_dump(mode="json")
                del dump["severity_source"]  # dump written before this field existed
                back = S.CompiledRule.model_validate(dump)
                self.assertEqual(back.severity_source, "unverified")  # neither policy_stated nor schema_default is claimed

    def test_llm_supplied_severity_source_never_overrides_the_compiler_decision(self):
        raw1, _ = _raw_rules("REQUIRE", "REQUIRE")
        stated = _C1[:-1] + " (severity: critical)."
        vague = _C1[:-1] + "; a breach is a major finding."
        cases = [
            ({}, _C1, "schema_default"),                                                         # omitted severity
            ({"severity": "low"}, _C1, "schema_default"),                                        # unsupported severity
            ({"severity": "high", "source_text": vague}, vague, "ambiguous"),                    # ambiguous wording
            ({"severity": "low", "source_text": stated}, stated, "schema_default"),              # contradicts the policy
            ({"severity": "critical", "source_text": stated}, stated, "policy_stated"),          # genuinely stated
        ]
        for extra, clause, expected in cases:
            for claimed in ("policy_stated", "schema_default", "ambiguous", "unverified", "bogus", None):
                with self.subTest(extra=extra, claimed=claimed):
                    raw = dict(copy.deepcopy(raw1), **extra)
                    raw["severity_source"] = claimed
                    before = copy.deepcopy(raw)
                    rule, errors = P.validate_raw_rule(raw, "POL_T", 1, "EXP-4.2: " + clause, 0.7)
                    self.assertEqual(errors, [])
                    self.assertEqual(rule.severity_source, expected)
                    self.assertEqual(raw, before)  # caller's dict untouched

    def test_graph_attachment_records_source_without_changing_execution(self):
        for combo in _COMBOS:
            with self.subTest(combo=combo):
                _, decision, attached = _run_case([_OK_EVIDENCE], *combo)
                self.assertEqual(attached[0]["compiled_rule"]["severity_source"], "schema_default")
                self.assertEqual(attached[0]["compiled_rule"]["severity"], "medium")
                self.assertEqual(attached[0]["severity"], "medium")
                self.assertEqual((decision["verdict"], decision["compiled_verdict"]), ("SATISFIED", "COMPLIANT"))  # unchanged by the new field
                self.assertNotIn("severity_source", attached[0])  # the entry exposes only `severity`; the source lives in the stored rule dump

    def test_diagnostic_recovery_revalidates_stored_dumps_with_and_without_the_field(self):
        import networkx as nx
        rule, errors = P.validate_raw_rule(*(lambda r: (r[0], "POL_TEST", 1, POLICY, 0.7))(_raw_rules("REQUIRE", "REQUIRE")))
        self.assertEqual(errors, [])
        with_field = rule.model_dump(mode="json")
        without_field = dict(with_field); del without_field["severity_source"]
        for label, dump in (("with severity_source", with_field), ("stored before the field existed", without_field)):
            with self.subTest(dump=label):
                G = nx.MultiDiGraph()
                G.add_node("doc1", type="Document", filename="expense_report.txt")
                G.add_node("ev1", type="Evidence", text="Expense report ER-101: Attachment: written approval from the Director of Finance, dated before payment.",
                           source_file="expense_report.txt", source_location="Lines 1-1", context_only=False)
                G.add_edge("ev1", "doc1", relation="DERIVED_FROM")
                G.add_node("policy_rule", type="PolicyRule", compiled_rules=[{"compiled_rule": dump}])
                inputs = T.graph_to_rule_inputs(G)  # recovery path: CompiledRule.model_validate on the stored dump
                self.assertTrue(inputs["evidence"], "stored rule was dropped by revalidation")
                self.assertEqual(inputs["evidence"][0]["type"], "written_approval_from_a_director")

    def test_rule_engine_decides_the_same_whatever_severity_source_says(self):
        clause = "Passwords must be at least 12 characters."
        raw = {"rule_type": "REQUIRE", "entity": "password", "temporal": None, "required_evidence": [], "exception": [],
               "condition": {"entity": "password", "field": "length", "operator": ">=", "value": 12, "unit": None},
               "action": "enforce_password_length", "confidence": 0.95, "source_text": clause, "ambiguities": []}
        rule, errors = P.validate_raw_rule(raw, "POL_T", 1, clause, 0.7)
        self.assertEqual((errors, rule.severity_source), ([], "schema_default"))
        facts = {"password": [{"length": 8}]}
        baseline = R.evaluate_policy([rule], facts, None, evaluation_date="2026-10-10")  # pinned: the result embeds the evaluation date
        for source in ("policy_stated", "ambiguous", "unverified"):
            with self.subTest(source=source):
                other = rule.model_copy(update={"severity_source": source})
                self.assertEqual(R.evaluate_policy([other], facts, None, evaluation_date="2026-10-10"), baseline)
        self.assertEqual(baseline["rule_results"][0]["verdict"], "VIOLATION")
        self.assertEqual(baseline["rule_results"][0]["severity"], "medium")  # the default value, consumed as before


# ---------------------------------------------------------------------------------------------------------------------
# Explicit severity labels must not trip the vague-term scan ("high"/"low" are in _VAGUE), without hiding other ambiguity.
# ---------------------------------------------------------------------------------------------------------------------
def _reset_raw(clause, severity=None):
    raw = {"rule_type": "REQUIRE", "entity": "password_reset", "temporal": None, "required_evidence": [], "exception": [],
           "condition": {"entity": "password_reset", "field": "identity_verified", "operator": "==", "value": True, "unit": None},
           "action": "verify_identity", "confidence": 0.95, "source_text": clause, "ambiguities": []}
    if severity is not None:
        raw["severity"] = severity
    return raw


def _compile_one(clause, severity=None):
    raw = _reset_raw(clause, severity)
    return P.compile_policy(clause, policy_id="POL_T", llm_fn=lambda system, user: json.dumps({"rules": [raw], "unparsed_statements": []}))


class SeverityLabelVagueScanTests(unittest.TestCase):
    def test_mask_blanks_only_explicit_severity_spans_and_keeps_length(self):
        text = "Resets with high costs, low confidence and critical systems are high severity (priority: low) or classified as medium."
        masked = P._mask_severity_labels(text)
        self.assertEqual(len(masked), len(text))
        for kept in ("high costs", "low confidence", "critical systems"):
            self.assertIn(kept, masked)
        for gone in ("high severity", "priority: low", "classified as medium"):
            self.assertNotIn(gone, masked)
        self.assertEqual(P._mask_severity_labels(""), "")
        self.assertEqual(P._mask_severity_labels(None), "")

    def test_policy_level_scan_ignores_explicit_severity_without_number(self):
        for sentence in ("Password reset is high severity.", "Password reset is low severity.", "Password reset has severity: high.",
                         "Password reset has priority: low.", "Password reset is classified as high."):
            with self.subTest(sentence=sentence):
                self.assertEqual(P.lexical_ambiguity_reasons(sentence), [])

    def test_validator_accepts_explicit_severity_without_number_as_valid(self):
        for phrase, level in (("high severity", "high"), ("low severity", "low"), ("severity: high", "high"), ("priority: low", "low"),
                              ("classified as high", "high"), ("severity: critical", "critical")):
            with self.subTest(phrase=phrase):
                clause = f"Password reset requests must be verified ({phrase})."
                rule, errors = P.validate_raw_rule(_reset_raw(clause, level), "POL_T", 1, clause, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual((rule.status, rule.severity.value, rule.severity_source), ("VALID", level, "policy_stated"), rule.ambiguities)
                self.assertEqual(rule.source_text, clause)  # provenance untouched
                self.assertFalse(any("vague term" in x for x in rule.ambiguities))

    def test_compile_policy_end_to_end_has_no_review_for_explicit_severity(self):
        result = _compile_one("Password reset requests must be verified (high severity).", "high")
        self.assertEqual((result.status, result.ambiguity_reasons, result.ambiguous_policy), ("COMPILED", [], False))
        self.assertEqual((result.rules[0].severity.value, result.rules[0].severity_source), ("high", "policy_stated"))

    def test_explicit_severity_does_not_hide_other_vague_or_advisory_wording(self):
        cases = [("Password reset requests must be verified promptly (high severity).", "promptly", "high"),
                 ("Password reset requests should be verified (high severity).", "should", "high"),
                 ("Password reset requests must be verified with reasonable care (low severity).", "reasonable", "low"),
                 ("Resets with high costs must be verified (high severity).", "high", "high"),    # unrelated 'high' still counts
                 ("Low confidence resets must be verified (priority: low).", "Low", "low")]
        for clause, word, level in cases:
            with self.subTest(clause=clause):
                rule, errors = P.validate_raw_rule(_reset_raw(clause, level), "POL_T", 1, clause, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual(rule.status, "NEEDS_REVIEW")
                self.assertTrue(any(word in x and ("vague term" in x or "advisory wording" in x) for x in rule.ambiguities), rule.ambiguities)
                self.assertEqual((rule.severity.value, rule.severity_source), (level, "policy_stated"))  # severity grounding unaffected
                reasons = P.lexical_ambiguity_reasons(clause)
                self.assertTrue(reasons and any(word in r for r in reasons), reasons)
                self.assertIn(clause[:60], reasons[0])  # reported against the ORIGINAL sentence, not the masked one
                self.assertEqual(_compile_one(clause, level).status, "COMPILED_WITH_REVIEW")

    def test_ordinary_severity_like_words_still_count_as_vague(self):
        cases = [("High costs must be approved by finance.", "high"), ("Low confidence results must be reviewed.", "low"),
                 ("Resets of high value accounts must be verified.", "high")]
        for clause, word in cases:
            with self.subTest(clause=clause):
                self.assertTrue(any(f"'{word}'" in r.lower() for r in P.lexical_ambiguity_reasons(clause)))
                rule, errors = P.validate_raw_rule(_reset_raw(clause), "POL_T", 1, clause, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual((rule.status, rule.severity_source), ("NEEDS_REVIEW", "schema_default"))
                self.assertTrue(any(f"vague term '{word}'" in x.lower() for x in rule.ambiguities), rule.ambiguities)

    def test_ordinary_words_do_not_become_explicit_severity(self):
        for clause in ("Servers hosting critical systems must be patched.", "Resets with high costs must be verified.",
                       "Low confidence resets must be verified."):
            with self.subTest(clause=clause):
                self.assertEqual(P._mask_severity_labels(clause), clause)
                for level in ("critical", "high", "low"):
                    verdict = P._assess_severity({"severity": level, "source_text": clause}, clause)
                    self.assertNotEqual(verdict["source"], "policy_stated")

    def test_severity_grounding_default_and_version(self):
        self.assertEqual(S.CompiledRule.model_fields["severity"].default, S.Severity.MEDIUM)
        clause = "Password reset requests must be verified."
        rule, _ = P.validate_raw_rule(_reset_raw(clause, "high"), "POL_T", 1, clause, 0.7)  # unstated: still discarded
        self.assertEqual((rule.severity, rule.severity_source), (S.Severity.MEDIUM, "schema_default"))
        self.assertNotIn(P.COMPILER_VERSION, ("2026-10-10.3-confirmation-entity-severity-grounding", "2026-10-10.4-severity-prompt-consistency", "2026-10-10.5-severity-label-vague-masking", "2026-10-10.6-severity-label-no-of-connector"))
        self.assertTrue(P.SYSTEM_PROMPT.rstrip().endswith("COMPILER_VERSION: " + P.COMPILER_VERSION))


class SeverityOfConnectorTests(unittest.TestCase):
    """'severity of high costs' is a description of costs, not a severity classification ('of' is not a connector)."""
    CLAUSES = ("Resets with severity of high costs must be verified.", "Resets with priority of high cost must be verified.")

    def test_severity_of_phrase_is_not_policy_stated_and_is_not_masked(self):
        for clause in self.CLAUSES + ("Resets with severity of critical impact must be verified.",):
            with self.subTest(clause=clause):
                self.assertEqual(P._mask_severity_labels(clause), clause)
                for level in ("low", "high", "critical"):
                    self.assertNotEqual(P._assess_severity({"severity": level, "source_text": clause}, clause)["source"], "policy_stated")

    def test_high_stays_visible_to_both_vague_scans(self):
        for clause in self.CLAUSES:
            with self.subTest(clause=clause):
                self.assertTrue(any("vague term 'high'" in r for r in P.lexical_ambiguity_reasons(clause)))
                rule, errors = P.validate_raw_rule(_reset_raw(clause, "high"), "POL_T", 1, clause, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual((rule.status, rule.severity, rule.severity_source), ("NEEDS_REVIEW", S.Severity.MEDIUM, "schema_default"))  # unstated severity discarded
                self.assertTrue(any("vague term 'high'" in x for x in rule.ambiguities), rule.ambiguities)
                self.assertEqual(_compile_one(clause, "high").status, "COMPILED_WITH_REVIEW")

    def test_documented_forms_are_still_explicit_and_masked(self):
        for phrase, level in (("high severity", "high"), ("severity: critical", "critical"), ("priority: low", "low"), ("classified as medium", "medium")):
            with self.subTest(phrase=phrase):
                clause = f"Password reset requests must be verified ({phrase})."
                self.assertEqual(P._assess_severity({"severity": level, "source_text": clause}, clause)["source"], "policy_stated")
                self.assertNotIn(phrase, P._mask_severity_labels(clause))
                self.assertEqual(P.lexical_ambiguity_reasons(clause), [])


class SeverityLabelConnectorTests(unittest.TestCase):
    """A severity/priority label needs a real connector (is / as / = / :). Adjacent words are a description, not a label."""
    SUPPORTED = (("high severity", "high"), ("severity: critical", "critical"), ("priority: low", "low"), ("classified as medium", "medium"),
                 ("severity is high", "high"), ("priority level = medium", "medium"), ("severity:critical", "critical"), ("severity level is low", "low"))
    REJECTED = ("Resets with severity high costs must be verified.", "Resets with priority high cost must be verified.",
                "Resets with severity critical impact must be verified.", "Resets with severityis high costs must be verified.",
                "Resets with priority level high cost must be verified.")

    def test_supported_forms_are_explicit_and_masked(self):
        for phrase, level in self.SUPPORTED:
            with self.subTest(phrase=phrase):
                clause = f"Password reset requests must be verified ({phrase})."
                verdict = P._assess_severity({"severity": level, "source_text": clause}, clause)
                self.assertEqual((verdict["source"], verdict["drop"]), ("policy_stated", False))
                masked = P._mask_severity_labels(clause)
                self.assertEqual(len(masked), len(clause))
                self.assertNotIn(phrase, masked)
                self.assertEqual(P.lexical_ambiguity_reasons(clause), [])
                rule, errors = P.validate_raw_rule(_reset_raw(clause, level), "POL_T", 1, clause, 0.7)
                self.assertEqual((errors, rule.status, rule.severity.value, rule.severity_source), ([], "VALID", level, "policy_stated"), rule.ambiguities)

    def test_adjacent_word_forms_are_not_explicit_and_are_not_masked(self):
        for clause in self.REJECTED:
            with self.subTest(clause=clause):
                self.assertEqual(P._mask_severity_labels(clause), clause)
                for level in ("low", "high", "critical"):
                    self.assertNotEqual(P._assess_severity({"severity": level, "source_text": clause}, clause)["source"], "policy_stated")

    def test_words_in_rejected_forms_stay_visible_to_both_vague_scans(self):
        for clause in self.REJECTED[:2] + self.REJECTED[3:]:  # these contain the vague word 'high'
            with self.subTest(clause=clause):
                self.assertTrue(any("vague term 'high'" in r for r in P.lexical_ambiguity_reasons(clause)))
                rule, errors = P.validate_raw_rule(_reset_raw(clause, "high"), "POL_T", 1, clause, 0.7)
                self.assertEqual(errors, [])
                self.assertEqual((rule.status, rule.severity, rule.severity_source), ("NEEDS_REVIEW", S.Severity.MEDIUM, "schema_default"))
                self.assertTrue(any("vague term 'high'" in x for x in rule.ambiguities), rule.ambiguities)


class AmountFieldGroundingTests(unittest.TestCase):
    """Exact failure mode of benchmark record ADV-01-OB-01-1A: the compiler named the amount field ``total_amount`` while the
    graph exposes the transaction fact as ``amount``. The engine and its fail-closed handling are exercised unmodified."""

    _ALIAS_ROW = [{"rule_id": "POL_T-R001", "entity": "expense", "from_field": "total_amount", "to_field": "amount"}]

    def test_1_total_amount_is_grounded_to_the_existing_amount_fact_and_decision_is_authoritative(self):
        for combo in _COMBOS:
            with self.subTest(combo=combo):
                summary, d, attached = _run_case([_OK_EVIDENCE], *combo, amount_field="total_amount")
                self.assertEqual(summary["normalized_transaction_amount_field_aliases"], self._ALIAS_ROW)
                self.assertIn("expense.amount", attached[0]["expression"])
                self.assertNotIn("total_amount", attached[0]["expression"])
                self.assertEqual([x["result"]["verdict"] for x in attached], ["COMPLIANT", "NOT_APPLICABLE"])
                self.assertEqual([x["result"]["missing_facts"] for x in attached], [[], []])
                self.assertEqual([x["result"]["action_required"] for x in attached], [None, None])  # no escalation: nothing was violated
                self.assertEqual(attached[0]["result"]["satisfying_node_ids"], ["txn"])
                self.assertEqual((d["evaluation_engine"], d["result_source"], d["compiled_verdict"], d["verdict"]),
                                 ("compiled_rule_engine", "compiled_policy_engine", "COMPLIANT", "SATISFIED"))
                self.assertEqual(d["compiled_missing_facts"], [])

    def test_2_failure_modes_behave_exactly_as_with_the_canonical_field(self):
        cases = {
            "missing approval": "Expense report ER-101: Priya Nair purchased conference registration for $3,400.",
            "approval after payment": "Expense report ER-101: purchased for $3,400. Attachment: written approval from the Director of Finance, dated after payment.",
            "wrong approver": "Expense report ER-101: purchased for $3,400. Attachment: written approval from the Manager, dated before payment. The Director of Finance was copied.",
            "negation": "Expense report ER-101: purchased for $3,400. No written approval from the Director of Finance was attached, dated before payment.",
        }
        for name, text in cases.items():
            for combo in _COMBOS:
                with self.subTest(case=name, combo=combo):
                    _, d_ref, att_ref = _run_case([text], *combo, amount_field="amount")
                    _, d, att = _run_case([text], *combo, amount_field="total_amount")
                    self.assertNotEqual(d["verdict"], "SATISFIED")
                    self.assertNotEqual(d["compiled_verdict"], "COMPLIANT")
                    self.assertEqual([(x["result"]["verdict"], x["result"]["missing_facts"], x["result"]["action_required"]) for x in att],
                                     [(x["result"]["verdict"], x["result"]["missing_facts"], x["result"]["action_required"]) for x in att_ref])
                    self.assertEqual((d["verdict"], d["compiled_verdict"], d["evaluation_engine"]), (d_ref["verdict"], d_ref["compiled_verdict"], d_ref["evaluation_engine"]))

    def test_3_genuinely_missing_amount_fact_is_not_grounded_or_invented(self):
        summary, d, attached = _run_case([_OK_EVIDENCE], with_txn=False, amount_field="total_amount")
        self.assertEqual(summary["normalized_transaction_amount_field_aliases"], [])
        self.assertEqual(attached[0]["result"]["verdict"], "INDETERMINATE")
        self.assertIn("expense.total_amount", attached[0]["result"]["missing_facts"])
        self.assertNotEqual(d["verdict"], "SATISFIED")

    def test_4_real_compile_policy_path_with_compiler_style_total_amount(self):
        # production compile_policy (not a SimpleNamespace): the LLM text is injected, everything after it is real code
        summary, d, attached = _run_case([_OK_EVIDENCE], "TRIGGER", "TRIGGER", amount_field="total_amount", real_compile=True)
        self.assertEqual(summary["status"], "COMPILED")
        self.assertEqual(len(summary["normalized_transaction_amount_field_aliases"]), 1)
        self.assertEqual([x["result"]["verdict"] for x in attached], ["COMPLIANT", "NOT_APPLICABLE"])
        self.assertEqual((d["evaluation_engine"], d["verdict"]), ("compiled_rule_engine", "SATISFIED"))
        self.assertEqual([x["result"]["action_required"] for x in attached], [None, None])

    def _rule(self, field, entity="expense", extra_condition=None, **over):
        raw = {"rule_type": "TRIGGER", "entity": entity, "temporal": None, "exception": [], "action": "attach_written_approval_from_director", "confidence": 0.9,
               "source_text": _C1, "ambiguities": [],
               "condition": extra_condition or {"entity": entity, "field": field, "operator": ">", "value": 2000, "unit": "USD"},
               "required_evidence": [{"type": "written_approval_from_a_director", "description": "d", "mandatory": True, "min_count": 1}]}
        raw.update(over)
        rule, errors = P.validate_raw_rule(raw, "POL_T", 1, _POL, 0.7)
        self.assertEqual(errors, [])
        return rule

    @staticmethod
    def _facts(records=None, entity="expense"):
        return {entity: records if records is not None else [{"amount": {"value": 3400.0, "unit": "USD"}, "currency": "USD"}]}

    def test_5_alias_only_when_grounded(self):
        norm = T._normalize_transaction_amount_field_aliases
        r = self._rule("total_amount"); self.assertEqual(len(norm([r], self._facts())), 1); self.assertEqual(r.condition.field, "amount")
        r = self._rule("expense_amount"); self.assertEqual(len(norm([r], self._facts())), 1); self.assertEqual(r.condition.field, "amount")
        untouched = {
            "fact already has the named field": (self._rule("total_amount"), self._facts([{"amount": 1, "total_amount": 3400}])),
            "no amount fact": (self._rule("total_amount"), self._facts([{"currency": "USD"}])),
            "mixed records (one lacks amount)": (self._rule("total_amount"), self._facts([{"amount": 1}, {"currency": "USD"}])),
            "no fact records": (self._rule("total_amount"), {}),
            "non-transaction entity": (self._rule("total_amount", entity="purchase_order"), self._facts(entity="purchase_order")),
            "unrelated field": (self._rule("balance"), self._facts()),
            "other entity's amount field": (self._rule("payment_amount"), self._facts()),
        }
        for name, (rule, facts) in untouched.items():
            with self.subTest(name):
                before = (rule.condition.field, rule.expression)
                self.assertEqual(norm([rule], facts), [])
                self.assertEqual((rule.condition.field, rule.expression), before)

    def test_6_ambiguous_rule_reading_both_fields_is_untouched(self):
        both = {"logic": "AND", "children": [{"entity": "expense", "field": "total_amount", "operator": ">", "value": 2000, "unit": "USD"},
                                             {"entity": "expense", "field": "amount", "operator": "<", "value": 9000, "unit": "USD"}]}
        r = self._rule(None, extra_condition=both)
        self.assertEqual(T._normalize_transaction_amount_field_aliases([r], self._facts()), [])
        self.assertEqual([lf.field for lf in T._compiled_rule_leaves(r.condition)], ["total_amount", "amount"])

    def test_7_exception_leaf_is_grounded_and_expression_rebuilt(self):
        r = self._rule("total_amount", exception=[{"description": "small", "condition": {"entity": "expense", "field": "total", "operator": "<", "value": 50, "unit": "USD"}}])
        rows = T._normalize_transaction_amount_field_aliases([r], self._facts())
        self.assertEqual(sorted(x["from_field"] for x in rows), ["total", "total_amount"])
        self.assertEqual((r.condition.field, r.exception[0].condition.field), ("amount", "amount"))
        self.assertNotIn("total_amount", r.expression)
        self.assertEqual(r.severity_source, "schema_default")  # revalidation did not turn a default severity into "unverified"


if __name__ == "__main__":
    unittest.main()