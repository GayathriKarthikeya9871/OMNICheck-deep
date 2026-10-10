"""Offline focused tests for Prompt 12 Phase 3 gate semantics.

Loads only the pure realization-gate functions from the experiment source, so no
Celery/database/provider setup is needed. Adapter source checks are structural,
not a substitute for the project's integration tests.
"""
import ast
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "omni_bench_experiment_phase3.py"
if not EXPERIMENT.exists():
    EXPERIMENT = ROOT / "omni_bench_experiment.py"
ADAPTER = ROOT / "omni_bench_system_adapters_phase3.py"
if not ADAPTER.exists():
    ADAPTER = ROOT / "omni_bench_system_adapters.py"
NOT_MEASURED = "NOT_MEASURED"


def load_gate_functions():
    tree = ast.parse(EXPERIMENT.read_text(encoding="utf-8"))
    names = {"_proof_rows", "_compile_ok", "_compiled_record_classification", "version_realization_report"}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    ns = {"NOT_MEASURED": NOT_MEASURED, "Dict": Dict, "List": List, "Iterable": Iterable, "Any": Any}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(EXPERIMENT), "exec"), ns)
    return ns


GATE = load_gate_functions()


def make_pred(rule_rows, *, call_log=None, effective=False, compiled_count=0, inputs=None, state_invoked=True):
    proof = {
        "compiled_policy_summaries": [{"status": "COMPILED_WITH_REVIEW", "error": None, "evaluation_error": None, "inputs": inputs or {"evidence_supplied": True}}],
        "compiled_policy_call_log": call_log if call_log is not None else [{"status": "CALLED", "result_status": "COMPILED_WITH_REVIEW", "error": None}],
        "compiled_policy_rule_proof": rule_rows,
        "compiled_policy_execution": {"effective_on_decision": effective, "compiled_decision_count": compiled_count, "legacy_decision_count": 0 if effective else 1},
        "contradiction_findings_present": False,
        "cross_document_summary_status": "DISABLED",
    }
    return {"perturbation_id": "record-1", "provenance": {
        "component_state": {"compiled_policy": {"enabled": True, "invoked": state_invoked}},
        "component_proof": proof,
    }}


class Phase3GateTests(unittest.TestCase):
    def report(self, pred):
        return GATE["version_realization_report"]({"V3": {"predictions": [pred]}}, ("V3",))["V3"]

    def test_indeterminate_missing_fact_is_partial_not_success(self):
        pred = make_pred([{"rule_id": "R2", "mapped": True, "status": "VALID", "executed": True,
                           "verdict": "INDETERMINATE", "missing_facts": ["expense_violation.confirmed"], "required_evidence": []}])
        report = self.report(pred)
        self.assertEqual(report["status"], "PARTIAL")
        row = report["checks"]["compiled_policy_record_classifications"][0]
        self.assertEqual(row["status"], "INDETERMINATE_MISSING_FACTS")
        self.assertEqual(row["missing_facts"], ["expense_violation.confirmed"])
        self.assertEqual(report["checks"]["compiled_policy_effective_decision_records"], 0)

    def test_missing_structured_evidence_is_reported_separately(self):
        pred = make_pred([
            {"rule_id": "R1", "mapped": True, "status": "NEEDS_REVIEW", "executed": False,
             "verdict": "INDETERMINATE", "missing_facts": [], "required_evidence": [{"type": "written_approval"}]},
            {"rule_id": "R2", "mapped": True, "status": "VALID", "executed": True,
             "verdict": "INDETERMINATE", "missing_facts": ["expense_violation.confirmed"], "required_evidence": []},
        ], inputs={"evidence_supplied": False})
        report = self.report(pred)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "EVIDENCE_NOT_SUPPLIED")

    def test_determinate_compiled_verdict_not_propagated_is_hard_failure(self):
        pred = make_pred([{"rule_id": "R1", "mapped": True, "status": "VALID", "executed": True,
                           "verdict": "VIOLATION", "missing_facts": [], "required_evidence": []}])
        report = self.report(pred)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "DETERMINATE_VERDICT_NOT_PROPAGATED")

    def test_determinate_compiled_decision_is_pass_when_other_gates_pass(self):
        pred = make_pred([{"rule_id": "R1", "mapped": True, "status": "VALID", "executed": True,
                           "verdict": "VIOLATION", "missing_facts": [], "required_evidence": []}], effective=True, compiled_count=1)
        report = self.report(pred)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "DETERMINATE_COMPILED_DECISION")

    def test_stage_not_called_is_hard_failure(self):
        pred = make_pred([{"rule_id": "R1", "mapped": True, "status": "VALID", "executed": False,
                           "verdict": "INDETERMINATE", "missing_facts": [], "required_evidence": []}], call_log=[], state_invoked=False)
        report = self.report(pred)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "NOT_CALLED")

    def test_no_mapped_rules_is_hard_failure(self):
        pred = make_pred([{"rule_id": "R1", "mapped": False, "status": "REJECTED", "executed": False,
                           "verdict": "INDETERMINATE", "missing_facts": [], "required_evidence": []}])
        report = self.report(pred)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "NO_MAPPED_RULES")

    def test_adapter_records_compiled_calls_for_v6_v8(self):
        source = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("compile_log: Optional[List[Dict[str, Any]]] = None", source)
        self.assertIn("G = _build_graph(record, tasks, workdir, True, compile_log)", source)
        self.assertIn('proof["compiled_policy_call_log"] = copy.deepcopy(compile_log or [])', source)
        self.assertIn('state["compiled_policy"] = {"flag": "COMPILED_POLICY_ENABLED"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
