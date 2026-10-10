"""Thin adapters that connect the EXISTING OMNICheck systems to the EXISTING OMNI-Bench adversarial evaluator. No reasoning pipeline lives here.

Reuses unchanged: omni_bench_adversarial_generator records, omni_bench_adversarial_eval.compare_systems / evaluate_adversarial_predictions,
tasks.build_evidence_graph, tasks.self_verification_variant_predictions, tasks.verify_policy_applicability, tasks._build_v1_payload, tasks.analyze_compliance_with_matrix.

COMPLETE: perturbation record -> files on disk (<doc_id>.txt + rulebook) -> real build_evidence_graph -> per-Decision structured results already produced by the
frozen pipeline (variant EVIDENCE_GRAPH_SELF_VERIFICATION_POLICY_APPLICABILITY) -> one prediction record. Only vocabulary mapping happens here (see OUTCOME_TO_LABEL).
V1: the real free-text path is invoked only on request (run_v1=True) and its raw text is kept as an inert trace. It is NEVER parsed. V1 exposes no structured field,
so NO V1 prediction record is built (predicted_decision is required by the evaluator) and the evaluator reports NOT_MEASURED for V1.
Nothing is fabricated: a record the system cannot answer yields no prediction; a field the system cannot supply is left None (evaluator -> NOT_MEASURED).
This module computes no metrics and runs no benchmark by itself.
"""
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from .omni_bench_adversarial_eval import NOT_MEASURED, compare_systems
    from .omni_bench_adversarial_generator import COMP, INS, NC
except ImportError:
    from omni_bench_adversarial_eval import NOT_MEASURED, compare_systems
    from omni_bench_adversarial_generator import COMP, INS, NC

SYSTEM_V1, SYSTEM_COMPLETE = "V1", "COMPLETE"
COMPLETE_VARIANT = "EVIDENCE_GRAPH_SELF_VERIFICATION_POLICY_APPLICABILITY"  # last stage of the frozen chain (7A lineage -> 7B verification -> 7C applicability)
OUTCOME_TO_LABEL = {"VIOLATION": NC, "SATISFIED": COMP, "NO_CONCLUSION": INS, "ESCALATE": INS}  # vocabulary of tasks.SV7D_OUTCOMES -> OMNI-Bench labels
OMNI_BENCH_OBJECTIVE = "Assess whether the supplied documents comply with the supplied policy."  # fixed; identical for every perturbation
RULEBOOK_FILENAME = "policy_rulebook.txt"
V1_STRUCTURED_FIELDS: Tuple[str, ...] = ()  # V1 yields free text only: no field is validly available
V1_NOT_MEASURED_REASON = "V1 returns a free-text report with no structured decision/evidence/contradiction/escalation field; no parser exists or is invented, so no V1 prediction is built"
VERSIONS: Tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")  # Phase 12B: selectable identifiers. Selection + provenance only; no per-version reasoning exists here yet
V2_COMPONENT_FLAGS: Dict[str, bool] = {"ENABLE_CROSS_DOCUMENT_LINKING": False, "ENABLE_CROSS_DOCUMENT_REASONING": False, "ENABLE_CONTRADICTION_HEURISTIC": False, "COMPILED_POLICY_ENABLED": False}  # existing tasks.py module flags, set for a V2 run only
V2_COMPONENT_MAP = {"cross_document_linking": "ENABLE_CROSS_DOCUMENT_LINKING", "cross_document_reasoning": "ENABLE_CROSS_DOCUMENT_REASONING",
                    "contradiction_detection": "ENABLE_CONTRADICTION_HEURISTIC", "compiled_policy": "COMPILED_POLICY_ENABLED"}
V2_NO_FLAG_COMPONENTS = ("self_verification", "counterfactual", "uncertainty")  # no tasks.py flag exists; V2 simply never calls them
V2_UNAVAILABLE_FIELDS = ("predicted_escalation", "predicted_contradicting_evidence", "predicted_missing_evidence", "predicted_contradiction_detected")  # omitted, not defaulted
V3_COMPONENT_FLAGS: Dict[str, bool] = {**V2_COMPONENT_FLAGS, "COMPILED_POLICY_ENABLED": True}  # V3 = V2 graph-only run + existing compiled-policy stage; nothing else changes
V4_COMPONENT_FLAGS: Dict[str, bool] = {**V3_COMPONENT_FLAGS, "ENABLE_CROSS_DOCUMENT_LINKING": True, "ENABLE_CROSS_DOCUMENT_REASONING": True}  # V4 = V3 + existing cross-document linking/reasoning; contradiction heuristic stays off
V5_COMPONENT_FLAGS: Dict[str, bool] = {**V4_COMPONENT_FLAGS, "ENABLE_CONTRADICTION_HEURISTIC": True}  # V5 = V4 + existing contradiction heuristic; nothing else changes
V6_COMPONENT_FLAGS: Dict[str, bool] = dict(V5_COMPONENT_FLAGS)  # V6 = V5 flags unchanged; self-verification has no tasks.py flag, it is invoked explicitly (see run_v6_graph)
V6_VARIANT = "EVIDENCE_GRAPH_SELF_VERIFICATION"  # key of the 7B (self-verification, pre-applicability) variant in tasks.self_verification_variant_predictions; absent key -> honest unanswered, never a fallback
V6_UNAVAILABLE_FIELDS = ("predicted_missing_evidence",)  # omitted, not defaulted
V7_COMPONENT_FLAGS: Dict[str, bool] = dict(V6_COMPONENT_FLAGS)  # V7 = V6 flags unchanged; counterfactual has no tasks.py flag, it is invoked explicitly (see run_v6_graph cf_log)
V7_VARIANT = V6_VARIANT  # V7 decision/escalation come from the V6 path unchanged
V7_UNAVAILABLE_FIELDS = V6_UNAVAILABLE_FIELDS  # tasks.build_counterfactual returns no decision, evidence-role or contradiction field the evaluator can compare: nothing is added to the prediction record
V7_COUNTERFACTUAL_FIELDS = ("counterfactual_id", "in_scope", "scope_class", "status", "change_type", "unresolved_reason", "missing_requirement", "recommended_corrective_condition",
                            "expected_resulting_state", "minimum_changes", "requires_reevaluation", "satisfaction_asserted")  # read from the dict tasks.build_counterfactual(G, decision_id) returns; copied only if present, values (incl. NOT_MEASURED) untouched. Carried in provenance only: the evaluator rejects extra prediction keys
V8_COMPONENT_FLAGS: Dict[str, bool] = dict(V7_COMPONENT_FLAGS)  # V8 = V7 flags unchanged; uncertainty has no tasks.py flag, it is invoked explicitly (see run_v6_graph unc_log)
V8_VARIANT = V7_VARIANT  # underlying 7B variant is the V7/V6 one; V8 only adds the uncertainty layer on top
V8_UNCERTAINTY_STATES: Tuple[str, ...] = ("COMPLIANT", "NON_COMPLIANT", "CONDITIONAL", "INSUFFICIENT_EVIDENCE")  # tasks.UNC_STATES (a found Decision never returns NOT_MEASURED here); anything else is invalid output
V8_STATE_TO_LABEL = {"COMPLIANT": COMP, "NON_COMPLIANT": NC, "CONDITIONAL": INS, "INSUFFICIENT_EVIDENCE": INS}  # vocabulary mapping only (CONDITIONAL = applicability/major contradiction unresolved = no final determination, like ESCALATE -> INS)
V8_UNCERTAINTY_FIELDS = ("contract_version", "uncertainty_status", "uncertainty_state_reason", "decision_confidence", "evidence_completeness", "contradiction_severity", "policy_alignment", "risk_level",
                         "escalation_required", "escalation_reasons", "verification_status", "policy_applicability_status")  # read from the dict tasks.build_uncertainty_assessment(G, decision_id) returns; copied only if present, values (incl. NOT_MEASURED / None confidence) untouched. Provenance only: the evaluator rejects extra prediction keys
V8_UNAVAILABLE_FIELDS = V7_UNAVAILABLE_FIELDS  # uncertainty layer returns no evaluator-comparable evidence-role / contradiction-detected field; none is added
_VERSION_FLAGS = {"V2": V2_COMPONENT_FLAGS, "V3": V3_COMPONENT_FLAGS, "V4": V4_COMPONENT_FLAGS, "V5": V5_COMPONENT_FLAGS, "V6": V6_COMPONENT_FLAGS, "V7": V7_COMPONENT_FLAGS, "V8": V8_COMPONENT_FLAGS}
_COMPILED_POLICY_VERSIONS = ("V3", "V4", "V5", "V6", "V7", "V8")
_VERSION_BLOCKERS: Dict[str, str] = {}  # no version remains blocked (V1..V8 all selectable)
_ACTIVE_CONFIG: Dict[str, Any] = {"version": None}  # adapter-level selection state; snapshotted/restored around every run
_MISSING = object()
_VERDICT_RANK = {"VIOLATION": 0, "INDETERMINATE": 1, "SATISFIED": 2}


def _sha(text: str) -> str: return hashlib.sha256(text.encode("utf-8")).hexdigest()
def _rec(r: Any) -> Dict[str, Any]: return r.to_dict() if hasattr(r, "to_dict") else copy.deepcopy(r)


# Cross-version compile-response cache: exact prompt hashes only; raw provider text is never
# written to experiment outputs. The same compile prompt replays identically in V4-V8.
_COMPILER_RESPONSE_CACHE: Dict[str, Optional[str]] = {}
_COMPILER_CACHE_LOG: List[Dict[str, Any]] = []

@contextlib.contextmanager
def cached_compiler_responses(tasks: Any):
    """Temporarily cache policy_compiler._call_llm by SHA-256 of the exact prompt pair."""
    compile_fn = getattr(tasks, "_compile_policy", None)
    module_name = getattr(compile_fn, "__module__", None)
    compiler_module = sys.modules.get(module_name) if module_name else None
    original = getattr(compiler_module, "_call_llm", None) if compiler_module else None
    if not callable(original):
        yield {"status": "UNAVAILABLE", "reason": "compiler _call_llm hook not found", "hits": 0, "misses": 0}
        return
    start = len(_COMPILER_CACHE_LOG)
    def _cached(system: str, user: str):
        key = _sha(str(system) + "\0" + str(user))
        if key in _COMPILER_RESPONSE_CACHE:
            value = _COMPILER_RESPONSE_CACHE[key]
            _COMPILER_CACHE_LOG.append({"prompt_sha256": key, "status": "HIT", "response_sha256": _sha(value) if isinstance(value, str) else None})
            return value
        value = original(system, user)
        if value is not None:
            _COMPILER_RESPONSE_CACHE[key] = value
        _COMPILER_CACHE_LOG.append({"prompt_sha256": key, "status": "MISS", "response_sha256": _sha(value) if isinstance(value, str) else None})
        return value
    setattr(compiler_module, "_call_llm", _cached)
    try:
        yield {"status": "ACTIVE", "start_index": start}
    finally:
        setattr(compiler_module, "_call_llm", original)

def compiler_cache_summary(start_index: int = 0) -> Dict[str, Any]:
    rows = _COMPILER_CACHE_LOG[start_index:]
    return {"status": "MEASURED", "calls_observed": len(rows),
            "cache_hits": sum(1 for x in rows if x["status"] == "HIT"),
            "cache_misses": sum(1 for x in rows if x["status"] == "MISS"),
            "entries": [{k: v for k, v in x.items()} for x in rows],
            "cache_entries_total": len(_COMPILER_RESPONSE_CACHE),
            "cache_key": "SHA-256(system_prompt + NUL + user_prompt)",
            "raw_response_storage": "in-memory only; response hashes in provenance"}

# ----------------------------------------------------------------------------- version selection / config
def validate_version(version: Any) -> str:
    """Accept exactly V1..V8 (case-sensitive str). Anything else raises ValueError; nothing is coerced."""
    if not isinstance(version, str) or version not in VERSIONS: raise ValueError(f"invalid version {version!r}; expected one of {VERSIONS}")
    return version


def snapshot_config(tasks: Optional[Any] = None, keys: Iterable[str] = ()) -> Dict[str, Any]:
    """Copy adapter state and the named attributes of tasks (absent attribute recorded as missing)."""
    return {"active": dict(_ACTIVE_CONFIG), "tasks": {k: getattr(tasks, k, _MISSING) for k in keys} if tasks is not None else {}}


def restore_config(snap: Dict[str, Any], tasks: Optional[Any] = None) -> None:
    """Put adapter state and tasks attributes back exactly as snapshotted (missing attributes are removed again)."""
    _ACTIVE_CONFIG.clear(); _ACTIVE_CONFIG.update(snap["active"])
    if tasks is None: return
    for k, v in snap["tasks"].items():
        if v is _MISSING:
            if hasattr(tasks, k): delattr(tasks, k)
        else: setattr(tasks, k, v)


# ----------------------------------------------------------------------------- inputs
def materialize_record(record: Any, workdir: str) -> Tuple[List[str], str, Dict[str, str]]:
    """Write the record's perturbed documents and policy to disk, unchanged. Returns (doc paths in record order, rulebook path, filename -> doc_id)."""
    r = _rec(record)
    names: Dict[str, str] = {}
    paths: List[str] = []
    for d in r["perturbed_document_set"]:
        fn = f"{d['doc_id']}.txt"
        if fn in names or fn == RULEBOOK_FILENAME or os.sep in fn: raise ValueError(f"{r['perturbation_id']}: doc_id {d['doc_id']!r} cannot be used as a file name")
        names[fn] = d["doc_id"]
        paths.append(os.path.join(workdir, fn))
        with open(paths[-1], "w", encoding="utf-8", newline="") as fh: fh.write(d["text"])
    rb = os.path.join(workdir, RULEBOOK_FILENAME)
    with open(rb, "w", encoding="utf-8", newline="") as fh: fh.write(r["perturbed_policy"])
    return paths, rb, names


def load_tasks(tasks_path: Optional[str] = None, module_name: Optional[str] = "app.services.module1_compliance.tasks") -> Tuple[Optional[Any], Optional[str]]:
    """Import the real tasks.py: first as its package module (it needs its package for relative imports), else from a path. Returns (module, None) or (None, exact blocker).
    Never substitutes a fake."""
    errs: List[str] = []
    if module_name:
        try: return importlib.import_module(module_name), None
        except Exception as e: errs.append(f"import {module_name}: {type(e).__name__}: {e}")  # noqa: BLE001
    if tasks_path:
        if not os.path.isfile(tasks_path): errs.append(f"tasks file not found: {tasks_path}")
        else:
            try:
                spec = importlib.util.spec_from_file_location("omnicheck_tasks", tasks_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod, None
            except Exception as e: errs.append(f"load {tasks_path}: {type(e).__name__}: {e}")  # noqa: BLE001
    return None, "; ".join(errs) or "no module_name or tasks_path given"


# ----------------------------------------------------------------------------- COMPLETE
def run_complete_graph(record: Any, tasks: Any, workdir: str) -> List[Dict[str, Any]]:
    """Run the REAL pipeline on one record; return one structured result per Decision node (read-only calls on the existing contract)."""
    paths, rb, names = materialize_record(record, workdir)
    G = tasks.build_evidence_graph(paths, rb)
    out = []
    for dec_id, _ in sorted(tasks._nodes_of_type(G, "Decision"), key=lambda kv: kv[0]):
        var = tasks.self_verification_variant_predictions(G, dec_id)[COMPLETE_VARIANT]
        res = tasks.verify_policy_applicability(G, dec_id)
        out.append({"outcome": var["outcome"], "verification_status": var["verification_status"],
                    "contradiction_predicted": var["contradiction_predicted"], "supporting_evidence": var["supporting_evidence"],
                    "contradicting_evidence": res.get("contradicting_evidence") or [], "filename_to_doc_id": dict(names)})
    return out


def _docs(refs: Iterable[Dict[str, Any]], names: Dict[str, str]) -> set:
    return {names[x["filename"]] for x in refs if isinstance(x, dict) and x.get("filename") in names}  # rulebook / unknown files are never cited


def _contra_docs(items: Iterable[Dict[str, Any]], sup_eids: set, names: Dict[str, str]) -> set:
    out: set = set()
    for c in items:
        if c.get("source") == "contradiction_finding":  # the claim side NOT resting on this decision's supporting evidence is the disputing one
            for side in (c.get("claim_a"), c.get("claim_b")):
                if side and not {p.get("evidence_id") for p in side.get("provenance") or []} & sup_eids: out |= _docs([side], names)
        else: out |= _docs(c.get("documents") or [], names)
    return out


def _aggregate_decision(labels: List[str]) -> str: return NC if NC in labels else INS if INS in labels else COMP


def complete_prediction_from_results(record: Any, results: List[Dict[str, Any]], version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pure mapping of existing per-Decision results to one evaluator prediction record. No Decision -> None (no prediction; nothing invented)."""
    r = _rec(record)
    if not results: return None
    labels = [OUTCOME_TO_LABEL[x["outcome"]] for x in results]
    decision = _aggregate_decision(labels)  # a violation wins; an unresolved/escalated rule outranks a pass (as in the engine's own priority)
    names: Dict[str, str] = {}
    for x in results: names.update(x["filename_to_doc_id"])
    sup_eids = {s.get("evidence_id") for x in results for s in x["supporting_evidence"]}
    sup = sorted(set().union(*(_docs(x["supporting_evidence"], names) for x in results)))
    con = sorted(set().union(*(_contra_docs(x["contradicting_evidence"], sup_eids, names) for x in results)))
    cflags = [x["contradiction_predicted"] for x in results]
    contra = True if any(f is True for f in cflags) else False if all(f is False for f in cflags) else None  # NOT_MEASURED from the system -> unavailable
    prov_extra = {"selected_version": validate_version(version), "version_logic": "none: selection recorded only; COMPLETE path unchanged"} if version is not None else {}
    return {"perturbation_id": r["perturbation_id"], "predicted_decision": decision,
            "predicted_escalation": any(x["verification_status"] == "ESCALATE" for x in results),
            "predicted_supporting_evidence": sup, "predicted_contradicting_evidence": con,
            "predicted_missing_evidence": None,  # no existing field yields evaluator-comparable missing-evidence strings
            "predicted_contradiction_detected": contra,
            "provenance": {"system": SYSTEM_COMPLETE, "interface": "build_evidence_graph + self_verification_variant_predictions + verify_policy_applicability",
                           "variant": COMPLETE_VARIANT, "decision_count": len(results), "decision_outcomes": sorted(x["outcome"] for x in results), **prov_extra,
                           "perturbation_provenance": copy.deepcopy(r["provenance"])}}


def predict_complete(records: Iterable[Any], tasks: Any, version: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Real COMPLETE pipeline over the SAME records, in order. Returns (predictions, unanswered[{perturbation_id, reason}]). A per-record failure is reported, never filled in."""
    preds, unanswered = [], []
    for rec in records:
        r = _rec(rec)
        try:
            with tempfile.TemporaryDirectory(prefix="omnibench_") as wd:
                p = complete_prediction_from_results(r, run_complete_graph(r, tasks, wd), version)
        except Exception as e:  # noqa: BLE001
            unanswered.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {e}"}); continue
        if p is None: unanswered.append({"perturbation_id": r["perturbation_id"], "reason": "pipeline produced no Decision node"})
        else: preds.append(p)
    return preds, unanswered


# ----------------------------------------------------------------------------- V2 (graph-only)
def v2_component_state(tasks: Any) -> Dict[str, Any]:
    """Actual component state read from tasks at call time (never assumed). Raises if a required flag does not exist (V2 could not be honestly graph-only)."""
    missing = [f for f in V2_COMPONENT_FLAGS if not hasattr(tasks, f)]
    if missing: raise RuntimeError(f"V2 blocker: tasks has no flag(s) {missing}; the component cannot be disabled")
    st: Dict[str, Any] = {c: {"flag": f, "enabled": bool(getattr(tasks, f)), "invoked": False} for c, f in V2_COMPONENT_MAP.items()}
    st.update({c: {"flag": None, "enabled": False, "invoked": False, "note": "no tasks.py flag; V2 path never calls it"} for c in V2_NO_FLAG_COMPONENTS})
    return st


def v2_prediction_from_results(record: Any, outcomes: List[str], component_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Graph-only: Decision outcomes only. Fields V2 has no mechanism for are OMITTED (evaluator -> NOT_MEASURED). No Decision -> None."""
    r = _rec(record)
    if not outcomes: return None
    return {"perturbation_id": r["perturbation_id"], "predicted_decision": _aggregate_decision([OUTCOME_TO_LABEL[o] for o in outcomes]),
            "provenance": {"system": "V2", "selected_version": "V2", "interface": "build_evidence_graph (flags off) + _nodes_of_type + _sv7d_outcome(verdict, None)",
                           "decision_count": len(outcomes), "decision_outcomes": sorted(outcomes), "component_state": copy.deepcopy(component_state),
                           "unavailable_fields": list(V2_UNAVAILABLE_FIELDS), "perturbation_provenance": copy.deepcopy(r["provenance"])}}


@contextlib.contextmanager
def compiled_policy_before_rules(tasks: Any, rulebook_text: str, log: Optional[List[Dict[str, Any]]] = None):
    """V3: for the duration of the with-block, tasks.evaluate_policy_rules is a wrapper that first calls the EXISTING tasks.apply_compiled_policy(G, rulebook_text)
    (same try/except-and-continue as process_document_batch_task), then the ORIGINAL evaluate_policy_rules. Works because build_evidence_graph resolves
    evaluate_policy_rules as a tasks module global at call time. tasks.py is not modified; the original is restored in finally."""
    original = tasks.evaluate_policy_rules
    def wrapped(G: Any) -> List[str]:
        try:
            summary = tasks.apply_compiled_policy(G, rulebook_text)
            if log is not None: log.append({"status": "CALLED", "result_status": (summary or {}).get("status") if isinstance(summary, dict) else None, "error": (summary or {}).get("error") if isinstance(summary, dict) else None})
        except Exception as e:  # noqa: BLE001  mirrors the task: continue with the legacy engine
            if log is not None: log.append({"status": "RAISED", "error": f"{type(e).__name__}: {e}"})
        return original(G)
    tasks.evaluate_policy_rules = wrapped
    try: yield
    finally: tasks.evaluate_policy_rules = original


def _build_graph(record: Any, tasks: Any, workdir: str, compiled_policy: bool, compile_log: Optional[List[Dict[str, Any]]] = None) -> Any:
    """Build one graph with the selected version's components isolated.

    tasks.build_evidence_graph calls classify_contradictions unconditionally in some
    repository revisions, so V2-V4 temporarily replace that call with a no-op. The
    original callable is always restored. This is runner-side only; tasks.py is not edited.
    """
    paths, rb, _ = materialize_record(record, workdir)
    version = _ACTIVE_CONFIG.get("version")
    original_classifier = getattr(tasks, "classify_contradictions", None)
    suppress_classifier = version in ("V2", "V3", "V4") and callable(original_classifier)
    if suppress_classifier:
        def _disabled_classifier(G):
            # Deliberately do not attach contradiction_findings to G.graph.
            return {"status": "DISABLED", "findings": 0, "note": "disabled by controlled V2-V4 configuration"}
        tasks.classify_contradictions = _disabled_classifier
    try:
        if compiled_policy:
            with compiled_policy_before_rules(tasks, tasks._rulebook_text_only(rb), compile_log):
                return tasks.build_evidence_graph(paths, rb)
        return tasks.build_evidence_graph(paths, rb)
    finally:
        if suppress_classifier:
            tasks.classify_contradictions = original_classifier


def _graph_component_proof(G: Any) -> Dict[str, Any]:
    """Compact, per-record proof of component realization from the actual graph.

    Compilation being called is not enough to prove that V3 affected a decision.
    This records rule-engine results attached to PolicyRule nodes and whether a
    Decision node actually selected the compiled engine over the legacy engine.
    """
    graph_meta = getattr(G, "graph", {})
    policies: List[Dict[str, Any]] = []
    attached_rules: List[Dict[str, Any]] = []
    decision_nodes: List[Dict[str, Any]] = []
    try:
        for node_id, data in G.nodes(data=True):
            node_type = data.get("type")
            if node_type == "Policy":
                cp = data.get("compiled_policy")
                if isinstance(cp, dict):
                    stats = cp.get("stats") if isinstance(cp.get("stats"), dict) else {}
                    inputs = cp.get("inputs") if isinstance(cp.get("inputs"), dict) else {}
                    policies.append({
                        "node_id": str(node_id),
                        "status": cp.get("status"),
                        "error": cp.get("error"),
                        "valid_rule_count": stats.get("valid", 0),
                        "needs_review_rule_count": stats.get("needs_review", 0),
                        "rejected_rule_count": stats.get("rejected", 0),
                        "evaluation_error": cp.get("evaluation_error"),
                        "unmapped_rule_count": len(cp.get("unmapped_rules") or []),
                        "inputs": {k: inputs.get(k) for k in (
                            "evaluation_date", "evaluation_date_source", "evidence_supplied",
                            "evidence_note", "fact_entities", "records", "strict_units", "fx_rates"
                        )},
                    })
            if node_type == "PolicyRule":
                for entry in data.get("compiled_rules") or []:
                    if not isinstance(entry, dict):
                        continue
                    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
                    compiled_rule = entry.get("compiled_rule") if isinstance(entry.get("compiled_rule"), dict) else {}
                    attached_rules.append({
                        "rule_id": entry.get("rule_id"),
                        "status": entry.get("status"),
                        "executed": result.get("executed") is True,
                        "verdict": result.get("verdict"),
                        "mapped": bool(entry.get("mapping_basis")),
                        "mapping_basis": entry.get("mapping_basis"),
                        "rule_type": entry.get("rule_type"),
                        "confidence": entry.get("confidence"),
                        "expression": entry.get("expression"),
                        "source_text": entry.get("source_text"),
                        "ambiguities": copy.deepcopy(entry.get("ambiguities") or []),
                        "required_evidence": copy.deepcopy(compiled_rule.get("required_evidence") or []),
                        "missing_facts": copy.deepcopy(result.get("missing_facts") or []),
                        "reasons": copy.deepcopy(result.get("reasons") or []),
                    })
            if node_type == "Decision":
                decision_nodes.append({
                    "node_id": str(node_id),
                    "evaluation_engine": data.get("evaluation_engine"),
                    "result_source": data.get("result_source"),
                    "verdict": data.get("verdict"),
                    "legacy_verdict": data.get("legacy_verdict"),
                    "compiled_verdict": data.get("compiled_verdict"),
                    "compiled_missing_facts": copy.deepcopy(data.get("compiled_missing_facts") or []),
                    "compiled_verdict_downgraded": data.get("compiled_verdict_downgraded"),
                })
    except Exception:  # noqa: BLE001
        # Keep the proof schema complete even if a custom graph object is malformed.
        policies = policies or []
        attached_rules = attached_rules or []
        decision_nodes = decision_nodes or []
    xdoc = graph_meta.get("cross_document_summary") if isinstance(graph_meta, dict) else None
    findings = graph_meta.get("contradiction_findings") if isinstance(graph_meta, dict) else None
    compiled_decision_count = sum(1 for row in decision_nodes if row.get("evaluation_engine") == "compiled_rule_engine")
    executed_rule_count = sum(1 for row in attached_rules if row.get("executed"))
    determinate_rule_count = sum(1 for row in attached_rules if row.get("executed") and row.get("verdict") not in (None, "INDETERMINATE"))
    return {
        "cross_document_summary_status": xdoc.get("status") if isinstance(xdoc, dict) else NOT_MEASURED,
        "contradiction_findings_present": isinstance(graph_meta, dict) and "contradiction_findings" in graph_meta,
        "contradiction_findings_count": len(findings or []) if isinstance(findings, list) else NOT_MEASURED,
        "compiled_policy_summaries": policies,
        "compiled_policy_rule_proof": attached_rules,
        "decision_nodes": decision_nodes,
        "compiled_policy_execution": {
            "policy_summary_count": len(policies),
            "attached_rule_count": len(attached_rules),
            "executed_rule_count": executed_rule_count,
            "determinate_executed_rule_count": determinate_rule_count,
            "decision_count": len(decision_nodes),
            "compiled_decision_count": compiled_decision_count,
            "legacy_decision_count": sum(1 for row in decision_nodes if row.get("evaluation_engine") != "compiled_rule_engine"),
            "effective_on_decision": compiled_decision_count > 0,
        },
        "graph_node_count": G.number_of_nodes() if callable(getattr(G, "number_of_nodes", None)) else NOT_MEASURED,
    }


def run_v6_graph(record: Any, tasks: Any, workdir: str, calls: List[str], cf_log: Optional[List[Tuple[str, Any]]] = None, unc_log: Optional[List[Tuple[str, Any, Any]]] = None, compile_log: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """V6: build the V5 graph with compiled-policy call proof, then invoke the existing 7B verification per Decision.
    V6 does not call 7C, counterfactual or uncertainty. Each verification invocation is appended to calls.
    V7 only: when cf_log is given, the EXISTING tasks.build_counterfactual(G, dec_id) is called once per Decision AFTER the V6 read and its raw return is appended as (dec_id, result). Returned items are unchanged.
    V8 only: when unc_log is given, the EXISTING tasks.build_uncertainty_assessment(G, dec_id) is called once per Decision AFTER the counterfactual and its raw return is appended as (dec_id, result, problems), problems being the
    existing tasks.validate_uncertainty_assessment(G, result) list when that function exists (else None). V7 passes no unc_log, so it never calls uncertainty."""
    _paths, _rb, filename_to_doc_id = materialize_record(record, workdir)
    G = _build_graph(record, tasks, workdir, True, compile_log)
    proof = _graph_component_proof(G)
    proof["filename_to_doc_id"] = dict(filename_to_doc_id)
    proof["compiled_policy_call_log"] = copy.deepcopy(compile_log or [])
    out = []
    selected_version = _ACTIVE_CONFIG.get("version")
    for dec_id, _ in sorted(tasks._nodes_of_type(G, "Decision"), key=lambda kv: kv[0]):
        calls.append(dec_id)
        # Do not call self_verification_variant_predictions here: that helper eagerly
        # computes 7B AND 7C. V6/V7 need 7B only; V8 explicitly adds 7C.
        if selected_version == "V8" and callable(getattr(tasks, "verify_policy_applicability", None)) and callable(getattr(tasks, "verify_self_verification_result", None)):
            verified = tasks.verify_policy_applicability(G, dec_id)
            verdict = G.nodes[dec_id].get("verdict")
            outcome = tasks._sv7d_outcome(verdict, verified.get("verification_status"))
        elif callable(getattr(tasks, "verify_self_verification_result", None)):
            verified = tasks.verify_self_verification_result(G, dec_id)
            verdict = G.nodes[dec_id].get("verdict")
            outcome = tasks._sv7d_outcome(verdict, verified.get("verification_status"))
        else:
            # Compatibility with minimal test doubles; production tasks.py exposes 7B/7C directly.
            var = tasks.self_verification_variant_predictions(G, dec_id)[V6_VARIANT]
            verified = var
            outcome = var["outcome"]
        row = {"outcome": outcome, "verification_status": verified.get("verification_status"), "component_proof": copy.deepcopy(proof)}
        if isinstance(verified, dict) and "supporting_evidence" in verified:
            row["supporting_evidence"] = copy.deepcopy(verified.get("supporting_evidence") or [])
        if isinstance(verified, dict) and "contradicting_evidence" in verified:
            row["contradicting_evidence"] = copy.deepcopy(verified.get("contradicting_evidence") or [])
            verdict = G.nodes[dec_id].get("verdict")
            asserting = verdict in ("VIOLATION", "SATISFIED")
            row["contradiction_predicted"] = (any(item.get("source") == "contradiction_finding" and item.get("category") == "MAJOR_CONTRADICTION"
                                                      for item in row["contradicting_evidence"]) if asserting else None)
        out.append(row)
        if cf_log is not None: cf_log.append((dec_id, tasks.build_counterfactual(G, dec_id)))
        if unc_log is not None:
            a = tasks.build_uncertainty_assessment(G, dec_id)
            validate = getattr(tasks, "validate_uncertainty_assessment", None)
            unc_log.append((dec_id, a, validate(G, a) if callable(validate) and isinstance(a, dict) else None))
    return out


def v6_prediction_from_results(record: Any, results: List[Dict[str, Any]], component_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """7B decision/escalation plus the evidence and contradiction fields actually supplied by verification."""
    r = _rec(record)
    if not results: return None
    prediction = {"perturbation_id": r["perturbation_id"],
                  "predicted_decision": _aggregate_decision([OUTCOME_TO_LABEL[x["outcome"]] for x in results]),
                  "predicted_escalation": any(x["verification_status"] == "ESCALATE" for x in results)}
    proof = results[0].get("component_proof") or {}
    names = proof.get("filename_to_doc_id") or {}
    if any("supporting_evidence" in x for x in results):
        sup_refs = [ref for x in results for ref in (x.get("supporting_evidence") or [])]
        prediction["predicted_supporting_evidence"] = sorted({names[ref["filename"]] for ref in sup_refs
                                                               if isinstance(ref, dict) and ref.get("filename") in names})
    if any("contradicting_evidence" in x for x in results):
        con_items = [item for x in results for item in (x.get("contradicting_evidence") or [])]
        sup_refs = [ref for x in results for ref in (x.get("supporting_evidence") or [])]
        sup_eids = {ref.get("evidence_id") for ref in sup_refs if isinstance(ref, dict)}
        prediction["predicted_contradicting_evidence"] = sorted(_contra_docs(con_items, sup_eids, names))
        flags = [x.get("contradiction_predicted") for x in results if isinstance(x.get("contradiction_predicted"), bool)]
        prediction["predicted_contradiction_detected"] = (any(flags) if flags else None)
    prediction["provenance"] = {"system": "V2", "selected_version": "V6",
        "interface": "build_evidence_graph (V5 flags) + verify_self_verification_result (7B)",
        "variant": V6_VARIANT, "decision_count": len(results),
        "decision_outcomes": sorted(x["outcome"] for x in results),
        "component_state": copy.deepcopy(component_state),
        "unavailable_fields": list(V6_UNAVAILABLE_FIELDS),
        "component_proof": [copy.deepcopy(x.get("component_proof")) for x in results],
        "perturbation_provenance": copy.deepcopy(r["provenance"])}
    return prediction



def predict_v6(records: Iterable[Any], tasks: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Caller must already have applied V6 flags (see run_version). component_state is read from tasks; self_verification is marked enabled and invoked from the real call log."""
    state, calls, compiled_calls = v2_component_state(tasks), [], []
    answered, unanswered = [], []
    for rec in records:
        r = _rec(rec)
        compile_log: List[Dict[str, Any]] = []
        try:
            with tempfile.TemporaryDirectory(prefix="omnibench_v6_") as wd: res = run_v6_graph(r, tasks, wd, calls, compile_log=compile_log)
        except Exception as e:  # noqa: BLE001
            compiled_calls.extend(compile_log)
            unanswered.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {e}"}); continue
        compiled_calls.extend(compile_log)
        if not res: unanswered.append({"perturbation_id": r["perturbation_id"], "reason": "pipeline produced no Decision node"})
        else: answered.append((r, res))
    state["compiled_policy"] = {"flag": "COMPILED_POLICY_ENABLED", "enabled": bool(getattr(tasks, "COMPILED_POLICY_ENABLED", False)),
                                "invoked": any(isinstance(x, dict) and x.get("status") == "CALLED" for x in compiled_calls),
                                "invocations": sum(1 for x in compiled_calls if isinstance(x, dict) and x.get("status") == "CALLED"),
                                "call_log_records": len(compiled_calls)}
    state["self_verification"] = {"flag": None, "enabled": True, "invoked": bool(calls), "invocations": len(calls), "note": "existing tasks.verify_self_verification_result (7B); no tasks.py flag"}
    state["policy_applicability"] = {"flag": None, "enabled": False, "invoked": False, "invocations": 0, "note": "7C is disabled for V6"}
    return [v6_prediction_from_results(r, res, state) for r, res in answered], unanswered, state


def v7_counterfactual_entries(cf_log: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
    """Per-Decision counterfactual provenance: only V7_COUNTERFACTUAL_FIELDS actually present in the existing result. A non-dict result is an error (record unanswered), never a default."""
    out = []
    for dec_id, cf in cf_log:
        if not isinstance(cf, dict): raise ValueError(f"counterfactual output missing for {dec_id}: build_counterfactual returned {type(cf).__name__}")
        out.append({"decision_id": dec_id, **{k: copy.deepcopy(cf[k]) for k in V7_COUNTERFACTUAL_FIELDS if k in cf}})
    return out


def v7_prediction_from_results(record: Any, results: List[Dict[str, Any]], cf_entries: List[Dict[str, Any]], component_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """V6 prediction unchanged (decision/escalation identical), plus counterfactual provenance. The counterfactual layer supplies no decision-level result, so none is invented."""
    p = v6_prediction_from_results(record, results, component_state)
    if p is None: return None
    pv = p["provenance"]
    pv.update({"selected_version": "V7", "interface": "build_evidence_graph (V5 flags) + verify_self_verification_result (7B) + build_counterfactual",
               "unavailable_fields": list(V7_UNAVAILABLE_FIELDS), "counterfactual": copy.deepcopy(cf_entries)})
    return p


def predict_v7(records: Iterable[Any], tasks: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Caller must already have applied V7 flags (see run_version). V6 path + existing counterfactual per Decision. Uncertainty is never called."""
    state, calls, cf_calls, compiled_calls = v2_component_state(tasks), [], [], []
    answered, unanswered = [], []
    for rec in records:
        r = _rec(rec)
        cf_log: List[Tuple[str, Any]] = []
        compile_log: List[Dict[str, Any]] = []
        try:
            with tempfile.TemporaryDirectory(prefix="omnibench_v7_") as wd: res = run_v6_graph(r, tasks, wd, calls, cf_log, compile_log=compile_log)
            cf_calls.extend(d for d, _ in cf_log)
            entries = v7_counterfactual_entries(cf_log)
        except Exception as e:  # noqa: BLE001
            compiled_calls.extend(compile_log)
            cf_calls.extend(d for d, _ in cf_log)
            unanswered.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {e}"}); continue
        compiled_calls.extend(compile_log)
        if not res: unanswered.append({"perturbation_id": r["perturbation_id"], "reason": "pipeline produced no Decision node"})
        else: answered.append((r, res, entries))
    state["compiled_policy"] = {"flag": "COMPILED_POLICY_ENABLED", "enabled": bool(getattr(tasks, "COMPILED_POLICY_ENABLED", False)),
                                "invoked": any(isinstance(x, dict) and x.get("status") == "CALLED" for x in compiled_calls),
                                "invocations": sum(1 for x in compiled_calls if isinstance(x, dict) and x.get("status") == "CALLED"),
                                "call_log_records": len(compiled_calls)}
    state["self_verification"] = {"flag": None, "enabled": True, "invoked": bool(calls), "invocations": len(calls), "note": "existing tasks.verify_self_verification_result (7B); no tasks.py flag"}
    state["counterfactual"] = {"flag": None, "enabled": True, "invoked": bool(cf_calls), "invocations": len(cf_calls), "note": "existing tasks.build_counterfactual; no tasks.py flag"}
    state["policy_applicability"] = {"flag": None, "enabled": False, "invoked": False, "invocations": 0, "note": "7C is disabled for V7"}
    return [v7_prediction_from_results(r, res, ent, state) for r, res, ent in answered], unanswered, state


def v8_uncertainty_entries(unc_log: List[Tuple[str, Any, Any]]) -> List[Dict[str, Any]]:
    """Per-Decision uncertainty entries: only V8_UNCERTAINTY_FIELDS actually present in the existing result. Output that is not a dict, is not for a found Decision, lacks a valid uncertainty_status / boolean escalation_required,
    or fails the existing contract validation is an error (record unanswered), never a default."""
    out = []
    for dec_id, a, probs in unc_log:
        if not isinstance(a, dict): raise ValueError(f"uncertainty output missing for {dec_id}: build_uncertainty_assessment returned {type(a).__name__}")
        if a.get("found") is not True: raise ValueError(f"uncertainty output invalid for {dec_id}: decision not found")
        if a.get("uncertainty_status") not in V8_UNCERTAINTY_STATES: raise ValueError(f"uncertainty output invalid for {dec_id}: uncertainty_status {a.get('uncertainty_status')!r}")
        if not isinstance(a.get("escalation_required"), bool): raise ValueError(f"uncertainty output invalid for {dec_id}: escalation_required {a.get('escalation_required')!r}")
        if probs: raise ValueError(f"uncertainty output invalid for {dec_id}: {'; '.join(str(x) for x in probs[:3])}")
        out.append({"decision_id": dec_id, **{k: copy.deepcopy(a[k]) for k in V8_UNCERTAINTY_FIELDS if k in a}})
    return out


def v8_prediction_from_results(record: Any, results: List[Dict[str, Any]], cf_entries: List[Dict[str, Any]], unc_entries: List[Dict[str, Any]], component_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """V7 prediction, then decision / escalation taken from the existing uncertainty layer's own uncertainty_status / escalation_required (aggregated over Decisions like V7). Nothing else is added; no confidence or probability is created.
    The V7-derived decision / escalation are kept in provenance for transparency. No Decision -> None."""
    p = v7_prediction_from_results(record, results, cf_entries, component_state)
    if p is None: return None
    if len(unc_entries) != len(results): raise ValueError(f"uncertainty output missing: {len(unc_entries)} assessment(s) for {len(results)} Decision(s)")
    pv = p["provenance"]
    v7_decision, v7_escalation = p["predicted_decision"], p["predicted_escalation"]
    p["predicted_decision"] = _aggregate_decision([V8_STATE_TO_LABEL[e["uncertainty_status"]] for e in unc_entries])
    p["predicted_escalation"] = any(e["escalation_required"] for e in unc_entries)
    pv.update({"selected_version": "V8", "interface": "build_evidence_graph (V5 flags) + verify_self_verification_result (7B) + verify_policy_applicability (7C) + build_counterfactual + build_uncertainty_assessment",
               "unavailable_fields": list(V8_UNAVAILABLE_FIELDS), "uncertainty": copy.deepcopy(unc_entries), "decision_source": "uncertainty_status", "escalation_source": "escalation_required",
               "v7_predicted_decision": v7_decision, "v7_predicted_escalation": v7_escalation})
    return p


def predict_v8(records: Iterable[Any], tasks: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Caller must already have applied V8 flags (see run_version). V7 path + existing uncertainty assessment per Decision. A record whose uncertainty output is missing / invalid / raising is unanswered, never filled in."""
    state, calls, cf_calls, unc_calls, compiled_calls = v2_component_state(tasks), [], [], [], []
    answered, unanswered = [], []
    for rec in records:
        r = _rec(rec)
        cf_log: List[Tuple[str, Any]] = []
        unc_log: List[Tuple[str, Any, Any]] = []
        compile_log: List[Dict[str, Any]] = []
        try:
            with tempfile.TemporaryDirectory(prefix="omnibench_v8_") as wd: res = run_v6_graph(r, tasks, wd, calls, cf_log, unc_log, compile_log)
            cf_calls.extend(d for d, _ in cf_log); unc_calls.extend(u[0] for u in unc_log)
            cf_entries = v7_counterfactual_entries(cf_log)
            unc_entries = v8_uncertainty_entries(unc_log)
        except Exception as e:  # noqa: BLE001
            compiled_calls.extend(compile_log)
            cf_calls.extend(d for d, _ in cf_log); unc_calls.extend(u[0] for u in unc_log)
            unanswered.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {e}"}); continue
        compiled_calls.extend(compile_log)
        if not res: unanswered.append({"perturbation_id": r["perturbation_id"], "reason": "pipeline produced no Decision node"})
        else: answered.append((r, res, cf_entries, unc_entries))
    state["self_verification"] = {"flag": None, "enabled": True, "invoked": bool(calls), "invocations": len(calls), "note": "existing tasks.verify_self_verification_result (7B); no tasks.py flag"}
    state["counterfactual"] = {"flag": None, "enabled": True, "invoked": bool(cf_calls), "invocations": len(cf_calls), "note": "existing tasks.build_counterfactual; no tasks.py flag"}
    state["uncertainty"] = {"flag": None, "enabled": True, "invoked": bool(unc_calls), "invocations": len(unc_calls), "note": "existing tasks.build_uncertainty_assessment; no tasks.py flag"}
    state["policy_applicability"] = {"flag": None, "enabled": True, "invoked": bool(calls), "invocations": len(calls), "note": "existing tasks.verify_policy_applicability (7C); V8 only"}
    preds = []
    for r, res, cfe, ue in answered:
        try: preds.append(v8_prediction_from_results(r, res, cfe, ue, state))
        except Exception as e: unanswered.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {e}"})  # noqa: BLE001
    return preds, unanswered, state


def run_v2_graph(record: Any, tasks: Any, workdir: str, compiled_policy: bool = False, compile_log: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """REAL graph build on one record, then each Decision's stored verdict -> outcome via the existing _sv7d_outcome(verdict, None). No verification is invoked.
    compiled_policy=True (V3): apply_compiled_policy runs inside build_evidence_graph, immediately before evaluate_policy_rules (see compiled_policy_before_rules)."""
    G = _build_graph(record, tasks, workdir, compiled_policy, compile_log)
    return [tasks._sv7d_outcome(G.nodes[d].get("verdict"), None) for d, _ in sorted(tasks._nodes_of_type(G, "Decision"), key=lambda kv: kv[0])]


def predict_v2(records: Iterable[Any], tasks: Any, compiled_policy: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """V2-V5 graph path. Uses 7A read-only evidence lineage only; 7B/7C are not invoked.

    V2-V4 expose supporting evidence only. V5 additionally exposes contradiction
    evidence/flag because the contradiction classifier is enabled. Missing fields
    remain absent so the evaluator records NOT_MEASURED.
    """
    state = v2_component_state(tasks)
    version = _ACTIVE_CONFIG.get("version") or ("V3" if compiled_policy else "V2")
    state["self_verification"] = {"flag": None, "enabled": False, "invoked": False, "invocations": 0,
                                  "note": "V2-V5 use 7A read-only evidence lineage only; 7B self-verification is off"}
    state["counterfactual"] = {"flag": None, "enabled": False, "invoked": False, "invocations": 0}
    state["uncertainty"] = {"flag": None, "enabled": False, "invoked": False, "invocations": 0}
    state["policy_applicability"] = {"flag": None, "enabled": False, "invoked": False, "invocations": 0,
                                     "note": "7C is disabled for V2-V5"}
    preds, unanswered = [], []
    for rec in records:
        r = _rec(rec)
        try:
            with tempfile.TemporaryDirectory(prefix=f"omnibench_{version.lower()}_") as wd:
                # Build exactly once for this record, then retain filename -> doc_id mapping.
                paths, rb, names = materialize_record(r, wd)
                compile_log: List[Dict[str, Any]] = []
                if compiled_policy:
                    with compiled_policy_before_rules(tasks, tasks._rulebook_text_only(rb), compile_log):
                        G = _build_graph(r, tasks, wd, compiled_policy=False)
                else:
                    G = _build_graph(r, tasks, wd, compiled_policy=False)
                proof = _graph_component_proof(G)
                proof["filename_to_doc_id"] = dict(names)
                proof["compiled_policy_call_log"] = copy.deepcopy(compile_log)
                if compiled_policy:
                    # Record observed invocation, not merely the configured flag.
                    # A CALLED entry means apply_compiled_policy returned; the graph proof
                    # separately records whether compilation produced a usable status.
                    invoked = any(isinstance(row, dict) and row.get("status") == "CALLED" for row in compile_log)
                    state["compiled_policy"]["invoked"] = bool(invoked)
                    state["compiled_policy"]["invocations"] = int(state["compiled_policy"].get("invocations", 0)) + int(bool(invoked))
                decision_nodes = sorted(tasks._nodes_of_type(G, "Decision"), key=lambda kv: kv[0])
                if not decision_nodes:
                    unanswered.append({"perturbation_id": r["perturbation_id"], "reason": "pipeline produced no Decision node"})
                    continue
                outcomes, supporting_refs, contradicting_items, contradiction_flags = [], [], [], []
                evidence_available = callable(getattr(tasks, "build_self_verification_result", None))
                for dec_id, _dd in decision_nodes:
                    dd = G.nodes[dec_id]
                    verdict = dd.get("verdict")
                    outcomes.append(tasks._sv7d_outcome(verdict, None))
                    builder = getattr(tasks, "build_self_verification_result", None)
                    if callable(builder):
                        result = builder(G, dec_id)  # 7A structure only; does not run 7B/7C
                        if not isinstance(result, dict) or result.get("found") is not True:
                            raise ValueError(f"7A evidence result missing for Decision {dec_id}")
                        supporting_refs.extend(result.get("supporting_evidence") or [])
                        if version == "V5":
                            contradicting_items.extend(result.get("contradicting_evidence") or [])
                            contradiction_flags.append(any(
                                item.get("source") == "contradiction_finding" and item.get("category") == "MAJOR_CONTRADICTION"
                                for item in (result.get("contradicting_evidence") or [])
                            ))
                prediction = v2_prediction_from_results(r, outcomes, state)
                if prediction is None:
                    unanswered.append({"perturbation_id": r["perturbation_id"], "reason": "pipeline produced no Decision node"})
                    continue
                # V2-V5 support evidence is available from 7A lineage, but never expose 7C/7B fields.
                if evidence_available:
                    prediction["predicted_supporting_evidence"] = sorted({
                        names[ref["filename"]] for ref in supporting_refs
                        if isinstance(ref, dict) and ref.get("filename") in names
                    })
                if version == "V5" and callable(getattr(tasks, "build_self_verification_result", None)):
                    sup_eids = {x.get("evidence_id") for x in supporting_refs if isinstance(x, dict)}
                    contra = sorted(_contra_docs(contradicting_items, sup_eids, names))
                    prediction["predicted_contradicting_evidence"] = contra
                    # Contradiction classification is not meaningful for a record with no assertive decision.
                    asserting = any(x in ("VIOLATION", "SATISFIED") for x in outcomes)
                    prediction["predicted_contradiction_detected"] = (any(contradiction_flags) if asserting else None)
                prediction["provenance"].update({
                    "selected_version": version,
                    "component_state": copy.deepcopy(state),
                    "evidence_source": "7A read-only lineage; filenames mapped to supplied doc_ids",
                    "component_proof": copy.deepcopy(proof),
                })
                preds.append(prediction)
        except Exception as e:  # noqa: BLE001
            unanswered.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {e}"})
    return preds or [], unanswered, state


def run_version(records: Iterable[Any], tasks: Any, version: str) -> Dict[str, Any]:
    """Phase 12B version runner. V1: no structured prediction (unchanged). V2: graph-only. V3: V2 + compiled policy before rule evaluation. V4: V3 + existing cross-document linking/reasoning flags. V5: V4 + existing contradiction heuristic flag. V6: V5 + existing self-verification (7B variant). V7: V6 + existing counterfactual (tasks.build_counterfactual per Decision; decision unchanged from V6). V8: V7 + existing uncertainty assessment (tasks.build_uncertainty_assessment per Decision; decision / escalation come from its own uncertainty_status / escalation_required; missing / invalid output -> unanswered). No version is blocked.
    Config (adapter state + the V2 flags on tasks) is snapshotted and restored in finally."""
    validate_version(version)
    if version in _VERSION_BLOCKERS: raise NotImplementedError(_VERSION_BLOCKERS[version])  # empty: kept so a future blocker stays honest
    recs = [_rec(r) for r in records]
    out: Dict[str, Any] = {"selected_version": version, "record_ids": [r["perturbation_id"] for r in recs], "predictions": None, "unanswered": [], "component_state": None, "blocker": None}
    if version == "V1":
        out["blocker"] = V1_NOT_MEASURED_REASON; return out
    flags = _VERSION_FLAGS[version]
    snap = snapshot_config(tasks, flags)
    try:
        _ACTIVE_CONFIG["version"] = version
        for k, v in flags.items():
            if hasattr(tasks, k): setattr(tasks, k, v)
        cache_start = len(_COMPILER_CACHE_LOG)
        if version in _COMPILED_POLICY_VERSIONS:
            with cached_compiler_responses(tasks):
                if version == "V8": preds, out["unanswered"], out["component_state"] = predict_v8(recs, tasks)
                elif version == "V7": preds, out["unanswered"], out["component_state"] = predict_v7(recs, tasks)
                elif version == "V6": preds, out["unanswered"], out["component_state"] = predict_v6(recs, tasks)
                else: preds, out["unanswered"], out["component_state"] = predict_v2(recs, tasks, True)
        else:
            preds, out["unanswered"], out["component_state"] = predict_v2(recs, tasks, False)
        if version in _COMPILED_POLICY_VERSIONS:
            cache_summary = compiler_cache_summary(cache_start)
            if isinstance(out.get("component_state"), dict): out["component_state"]["compiler_cache"] = cache_summary
            for pr in preds: pr["provenance"].update({"selected_version": version, "compiler_cache": cache_summary})  # system stays V2: same graph decision path
        out["predictions"] = preds or None
    finally: restore_config(snap, tasks)
    return out


# ----------------------------------------------------------------------------- V1
def run_v1_trace(record: Any, tasks: Any, workdir: str) -> Dict[str, Any]:
    """Invoke the REAL V1 path (payload + analyze_compliance_with_matrix, is_v2=False) on one record. The raw text is stored verbatim as an inert trace and never parsed."""
    r = _rec(record)
    base = {"perturbation_id": r["perturbation_id"], "system": SYSTEM_V1, "perturbation_provenance": copy.deepcopy(r["provenance"])}
    try:
        paths, rb, _ = materialize_record(r, workdir)
        G = tasks.build_evidence_graph(paths, rb)
        rb_text = tasks._rulebook_text_only(rb)
        text = tasks.analyze_compliance_with_matrix(tasks._build_v1_payload(paths, rb_text, G), rb_text, OMNI_BENCH_OBJECTIVE, is_v2=False)
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "BLOCKED", "error": f"{type(e).__name__}: {e}", "raw_text": None, "raw_text_sha256": None}
    return {**base, "status": "RAN", "error": None, "raw_text": text, "raw_text_sha256": _sha(text) if isinstance(text, str) else None}


def v1_prediction_records(traces: Optional[Iterable[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """V1 has no structured fields (V1_STRUCTURED_FIELDS is empty), so no valid prediction record exists. None -> evaluator reports NOT_MEASURED. Trace text is never read."""
    return None


# ----------------------------------------------------------------------------- pairing
def run_paired(records: Iterable[Any], tasks: Optional[Any], run_v1: bool = False, blocker: Optional[str] = None,
               compare: Callable[..., Dict[str, Any]] = compare_systems, version: Optional[str] = None,
               config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Identical records to both systems, one evaluator call. tasks None -> nothing is run and the comparison is NOT_MEASURED (blocker reported).
    version (V1..V8, default None = legacy Prompt 11 behavior) is validated BEFORE anything runs and recorded in provenance. config_overrides
    (attr -> value) are set on tasks for the run only; adapter state and those attributes are restored in finally, even on error."""
    if version is not None: validate_version(version)
    overrides = dict(config_overrides or {})
    snap = snapshot_config(tasks, overrides)
    try:
        if version is not None: _ACTIVE_CONFIG["version"] = version
        if tasks is not None:
            for k, v in overrides.items(): setattr(tasks, k, v)
        out = _run_paired_core(records, tasks, run_v1, blocker, compare, version)
    finally: restore_config(snap, tasks)
    if version is not None: out["selected_version"] = version
    return out


def _run_paired_core(records: Iterable[Any], tasks: Optional[Any], run_v1: bool, blocker: Optional[str], compare: Callable[..., Dict[str, Any]], version: Optional[str]) -> Dict[str, Any]:
    recs = [_rec(r) for r in records]
    out: Dict[str, Any] = {"record_ids": [r["perturbation_id"] for r in recs], "blocker": blocker, "v1_traces": [], "complete_unanswered": [], "v1_not_measured_reason": V1_NOT_MEASURED_REASON}
    complete = None
    if tasks is not None:
        complete, out["complete_unanswered"] = predict_complete(recs, tasks, version)
        if run_v1:
            for r in recs:
                with tempfile.TemporaryDirectory(prefix="omnibench_v1_") as wd: out["v1_traces"].append(run_v1_trace(r, tasks, wd))
    out["v1_predictions"], out["complete_predictions"] = v1_prediction_records(out["v1_traces"]), (complete or None)
    out["comparison"] = compare(recs, out["v1_predictions"], out["complete_predictions"])
    return out