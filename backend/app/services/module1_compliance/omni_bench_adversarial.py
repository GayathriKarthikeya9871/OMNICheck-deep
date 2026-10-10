"""OMNI-Bench adversarial evaluation runner: scores SUPPLIED system predictions against the independent expected labels of generated perturbation records.

Deterministic and offline (no LLM, network, wall-clock or randomness). It never produces, completes or guesses predictions: no predictions -> NOT_MEASURED.
Ground truth (expected decision, escalation, evidence roles, contradiction behaviour, outcome change, provenance) is read ONLY from the perturbation record;
a prediction can never supply or alter it, and any ground-truth field inside a prediction makes the whole prediction set INVALID_PREDICTIONS.
Every metric returns NOT_MEASURED when it has no valid denominator. Perturbation provenance is preserved verbatim in every per-perturbation result.
Document text (including injected instructions) is inert data: this module never reads it to derive labels or scores.
Scope: scoring (evaluate_adversarial_predictions) plus a V1-vs-COMPLETE comparison over SUPPLIED predictions (compare_systems) and a static interface probe
(probe_system_interfaces). evaluate_adversarial_predictions / compare_systems never invoke a system. The 11C adapters (section below) drive the COMPLETE pipeline only through an explicitly injected
tasks module and never run V1 (an LLM call): V1 decisions come only from human reviews of its reports; every other V1 metric is NOT_MEASURED with a reason.
If a system cannot be driven, no performance number is produced.

Prediction record (all optional except perturbation_id and predicted_decision):
  perturbation_id, predicted_decision, predicted_escalation (bool), predicted_supporting_evidence / predicted_contradicting_evidence (unique doc_id lists),
  predicted_missing_evidence (unique strings), predicted_contradiction_detected (bool: any document OR policy contradiction flagged), provenance (system-side, preserved),
  and optional echo fields source_case_id / source_family_id / perturbation_type / split, which must match the record.

Metric definitions
  decision_accuracy, decision_macro_f1 : over the labels COMPLIANT / NON_COMPLIANT / INSUFFICIENT_EVIDENCE present in expected or predicted decisions.
  evidence_grounding : grounded_citation_rate = cited doc_ids (supporting + contradicting) that exist in the perturbed document set / cited doc_ids (hallucinated citations lower it);
      per-role and pooled doc_id precision/recall/F1 against expected roles; missing evidence by normalized exact string match plus a missing_flag_accuracy
      (did the system flag that something is missing when, and only when, the record expects it).
  contradiction_detection : predicted_contradiction_detected vs expected contradiction_present (accuracy, precision, recall, F1, false-positive rate).
  unsupported_decision_rate : committed decision (COMPLIANT / NON_COMPLIANT) where the expected label is INSUFFICIENT_EVIDENCE, over all expected-INSUFFICIENT_EVIDENCE records
      (and, separately, over all committed decisions).
  escalation : accuracy; and change vs the source case's expected escalation (change_recall over records whose expected escalation changed, change_false_alarm_rate over those that did not).

Run the embedded tests with:  python -m unittest omni_bench_adversarial_eval   (or: python omni_bench_adversarial_eval.py)
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unittest
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # package use or flat-directory use
    from .omni_bench_adversarial_generator import (COMP, INS, NC, OUTCOME_ALTERED, OUTCOME_UNCHANGED, PERTURBATION_TYPES, PerturbationRecord, generate_adversarial_suite,
                                         validate_adversarial_suite)
except ImportError:
    from omni_bench_adversarial_generator import (COMP, INS, NC, OUTCOME_ALTERED, OUTCOME_UNCHANGED, PERTURBATION_TYPES, PerturbationRecord, generate_adversarial_suite,
                                        validate_adversarial_suite)

EVALUATOR_NAME, EVALUATOR_VERSION = "omni_bench_adversarial_eval", "1.0.0"
NOT_MEASURED = "NOT_MEASURED"
STATUS_MEASURED, STATUS_NOT_MEASURED = "MEASURED", "NOT_MEASURED"
STATUS_INVALID_RECORDS, STATUS_INVALID_PREDICTIONS = "INVALID_RECORDS", "INVALID_PREDICTIONS"
LABELS = (COMP, NC, INS)
OUTCOME_ORDER = (OUTCOME_UNCHANGED, OUTCOME_ALTERED)
SPLIT_ORDER = ("TRAIN", "DEV", "TEST")
_RECORD_FIELDS = tuple(PerturbationRecord.__dataclass_fields__)
_ALLOWED_PRED_KEYS = frozenset(("perturbation_id", "predicted_decision", "predicted_escalation", "predicted_supporting_evidence", "predicted_contradicting_evidence",
                                "predicted_missing_evidence", "predicted_contradiction_detected", "provenance", "source_case_id", "source_family_id", "perturbation_type", "split"))
_ROLES = ("supporting", "contradicting")


# ----------------------------------------------------------------------------- small helpers
def _ratio(n, d): return n / d if d else NOT_MEASURED
def _norm(s: str) -> str: return " ".join(s.split()).casefold()
def _canon(o: Any) -> str: return json.dumps(o, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
def _sha(o: Any) -> str: return hashlib.sha256(_canon(o).encode("utf-8")).hexdigest()
def _has(p: Dict[str, Any], k: str) -> bool: return p.get(k) is not None
def _str_list(v) -> bool: return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def _prf(tp: int, pred: int, exp: int) -> Dict[str, Any]:
    """precision / recall / F1 from counts; each NOT_MEASURED when its own denominator is 0 (F1 = 2tp / (pred + exp))."""
    return {"precision": _ratio(tp, pred), "recall": _ratio(tp, exp), "f1": _ratio(2 * tp, pred + exp)}


def _as_dict(r: Any) -> Any: return r.to_dict() if isinstance(r, PerturbationRecord) else copy.deepcopy(r)


# ----------------------------------------------------------------------------- validation
def _check_records(records) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not isinstance(records, (list, tuple)): return ["records must be a list or tuple of perturbation records"], []
    errs: List[str] = []
    out: List[Dict[str, Any]] = []
    seen = set()
    for i, raw in enumerate(records):
        d = _as_dict(raw)
        if not isinstance(d, dict): errs.append(f"record[{i}]: not an object"); continue
        pid = d.get("perturbation_id")
        tag = f"record[{i}] ({pid})"
        if not isinstance(pid, str) or not pid: errs.append(f"record[{i}]: missing or invalid perturbation_id"); continue
        if pid in seen: errs.append(f"{tag}: duplicate perturbation_id")
        seen.add(pid)
        missing = [f for f in _RECORD_FIELDS if f not in d]
        extra = sorted(str(k) for k in d if k not in _RECORD_FIELDS)
        if missing: errs.append(f"{tag}: missing fields {missing}")
        if extra: errs.append(f"{tag}: unknown fields {extra} (predictions are never part of a record)")
        if missing: continue
        if d["perturbation_type"] not in PERTURBATION_TYPES: errs.append(f"{tag}: unknown perturbation_type")
        if d["expected_decision"] not in LABELS: errs.append(f"{tag}: invalid expected_decision")
        if not isinstance(d["expected_escalation"], bool): errs.append(f"{tag}: expected_escalation must be a bool")
        if d["outcome_change"] not in OUTCOME_ORDER: errs.append(f"{tag}: invalid outcome_change")
        roles = d["expected_evidence_roles"]
        if not (isinstance(roles, dict) and set(roles) == {"supporting", "contradicting", "missing"} and all(_str_list(v) for v in roles.values())): errs.append(f"{tag}: malformed expected_evidence_roles")
        cb = d["expected_contradiction_behavior"]
        if not (isinstance(cb, dict) and isinstance(cb.get("contradiction_present"), bool)): errs.append(f"{tag}: expected_contradiction_behavior.contradiction_present must be a bool")
        docs = d["perturbed_document_set"]
        if not (isinstance(docs, list) and all(isinstance(x, dict) and isinstance(x.get("doc_id"), str) for x in docs)): errs.append(f"{tag}: malformed perturbed_document_set")
        prov = d["provenance"]
        if not (isinstance(prov, dict) and isinstance(prov.get("source_split"), str) and isinstance(prov.get("source_expected_escalation"), bool)):
            errs.append(f"{tag}: provenance must carry source_split and source_expected_escalation")
        out.append(d)
    return sorted(errs), out


def _resolve_filter(value, order: Tuple[str, ...], label: str, universe) -> Optional[Tuple[str, ...]]:
    if value is None: return None
    req = [value] if isinstance(value, str) else list(value)
    allowed = set(order) | set(universe)
    if not req or any(v not in allowed for v in req): raise ValueError(f"{label} must be a non-empty selection of {sorted(allowed)}")
    return tuple(req)


def _check_predictions(predictions, by_id: Dict[str, Dict[str, Any]], scope_ids) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    if not isinstance(predictions, (list, tuple)): return [f"predictions must be a list or tuple of records, got {type(predictions).__name__}"], {}
    errs: List[str] = []
    accepted: Dict[str, Dict[str, Any]] = {}
    dups = set()
    for i, p in enumerate(predictions):
        if not isinstance(p, dict): errs.append(f"prediction[{i}]: not an object"); continue
        pid = p.get("perturbation_id")
        if not isinstance(pid, str) or not pid: errs.append(f"prediction[{i}]: missing or invalid perturbation_id"); continue
        tag = f"prediction[{i}] ({pid})"
        if pid not in by_id: errs.append(f"{tag}: unknown perturbation_id"); continue
        if pid in accepted: dups.add(pid)
        r = by_id[pid]
        extra = sorted(str(k) for k in p if k not in _ALLOWED_PRED_KEYS)
        if extra: errs.append(f"{tag}: unknown or ground-truth field(s) not allowed in a prediction: {extra}")
        if pid not in scope_ids: errs.append(f"{tag}: perturbation is outside the evaluated scope")
        if not isinstance(p.get("predicted_decision"), str) or p["predicted_decision"] not in LABELS: errs.append(f"{tag}: missing or invalid predicted_decision")
        for k in ("predicted_escalation", "predicted_contradiction_detected"):
            if _has(p, k) and not isinstance(p[k], bool): errs.append(f"{tag}: {k} must be a bool")
        if _has(p, "provenance") and not (isinstance(p["provenance"], dict) or (isinstance(p["provenance"], str) and p["provenance"].strip())): errs.append(f"{tag}: malformed provenance")
        for e in _ROLES:
            k, v = f"predicted_{e}_evidence", p.get(f"predicted_{e}_evidence")
            if v is not None and (not _str_list(v) or len(set(v)) != len(v)): errs.append(f"{tag}: malformed {k} (list of unique doc_id strings)")
        v = p.get("predicted_missing_evidence")
        if v is not None and (not _str_list(v) or len({_norm(x) for x in v}) != len(v)): errs.append(f"{tag}: malformed predicted_missing_evidence (list of unique non-empty strings)")
        for k, truth in (("source_case_id", r["source_case_id"]), ("source_family_id", r["source_family_id"]), ("perturbation_type", r["perturbation_type"]), ("split", r["provenance"]["source_split"])):
            if _has(p, k) and p[k] != truth: errs.append(f"{tag}: {k} mismatch (record has '{truth}')")
        accepted.setdefault(pid, p)
    errs.extend(f"duplicate prediction for perturbation_id {pid}" for pid in sorted(dups))
    return sorted(errs), accepted


# ----------------------------------------------------------------------------- per-perturbation scoring
def _role_metrics(expected: List[str], predicted: Optional[List[str]], norm: bool) -> Dict[str, Any]:
    n = _norm if norm else (lambda x: x)
    exp = {n(x) for x in expected}
    if predicted is None: return {"precision": NOT_MEASURED, "recall": NOT_MEASURED, "f1": NOT_MEASURED, "true_positives": None, "predicted_count": None, "expected_count": len(exp)}
    pred = {n(x) for x in predicted}
    tp = len(exp & pred)
    return {**_prf(tp, len(pred), len(exp)), "true_positives": tp, "predicted_count": len(pred), "expected_count": len(exp)}


def _eval_record(r: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    g = lambda k: p.get(k) if _has(p, k) else None
    roles, prov = r["expected_evidence_roles"], r["provenance"]
    doc_ids = {d["doc_id"] for d in r["perturbed_document_set"]}
    ed, pd, ee, pe, src_esc = r["expected_decision"], p["predicted_decision"], r["expected_escalation"], g("predicted_escalation"), prov["source_expected_escalation"]
    ps, pc, pm, det = g("predicted_supporting_evidence"), g("predicted_contradicting_evidence"), g("predicted_missing_evidence"), g("predicted_contradiction_detected")
    sup, con = _role_metrics(roles["supporting"], ps, False), _role_metrics(roles["contradicting"], pc, False)
    supplied = [m for m in (sup, con) if m["true_positives"] is not None]
    pooled = ({**_prf(sum(m["true_positives"] for m in supplied), sum(m["predicted_count"] for m in supplied), sum(m["expected_count"] for m in supplied)),
               "true_positives": sum(m["true_positives"] for m in supplied), "predicted_count": sum(m["predicted_count"] for m in supplied), "expected_count": sum(m["expected_count"] for m in supplied)}
              if supplied else _role_metrics([], None, False))
    cited = [x for x in (ps or []) + (pc or [])]
    cited_total = len(cited) if (ps is not None or pc is not None) else None
    cited_valid = sum(1 for x in cited if x in doc_ids) if cited_total is not None else None
    exp_missing_flag = bool(roles["missing"])
    exp_change = ee != src_esc
    exp_contra = r["expected_contradiction_behavior"]["contradiction_present"]
    return {"perturbation_id": r["perturbation_id"], "source_case_id": r["source_case_id"], "source_family_id": r["source_family_id"],
            "perturbation_type": r["perturbation_type"], "split": prov["source_split"], "outcome_change": r["outcome_change"],
            "expected_decision": ed, "predicted_decision": pd, "decision_correct": pd == ed,
            "unsupported_decision": pd in (COMP, NC) and ed == INS, "committed_decision": pd in (COMP, NC),
            "expected_escalation": ee, "predicted_escalation": pe, "source_expected_escalation": src_esc,
            "escalation_correct": NOT_MEASURED if pe is None else pe == ee,
            "expected_escalation_change": exp_change, "predicted_escalation_change": NOT_MEASURED if pe is None else pe != src_esc,
            "escalation_change_correct": NOT_MEASURED if pe is None else (pe != src_esc) == exp_change,
            "supporting_evidence_metrics": sup, "contradicting_evidence_metrics": con, "pooled_evidence_metrics": pooled,
            "missing_evidence_metrics": _role_metrics(roles["missing"], pm, True),
            "missing_flag_correct": NOT_MEASURED if pm is None else bool(pm) == exp_missing_flag, "expected_missing_flag": exp_missing_flag,
            "cited_doc_count": cited_total, "grounded_cited_doc_count": cited_valid,
            "grounded_citation_rate": NOT_MEASURED if not cited_total else cited_valid / cited_total,
            "expected_contradiction_present": exp_contra, "predicted_contradiction_detected": det,
            "contradiction_detection_correct": NOT_MEASURED if det is None else det == exp_contra,
            "expected_contradiction_behavior": copy.deepcopy(r["expected_contradiction_behavior"]),
            "expected_evidence_roles": copy.deepcopy(roles),
            "provenance": {"perturbation_provenance": copy.deepcopy(prov), "prediction_provenance": copy.deepcopy(p.get("provenance"))}}


# ----------------------------------------------------------------------------- aggregation
def _bool_acc(rs, key):
    v = [x[key] for x in rs if isinstance(x[key], bool)]
    return _ratio(sum(v), len(v)), len(v)


def _role_agg(rs, key) -> Dict[str, Any]:
    m = [x[key] for x in rs if x[key]["true_positives"] is not None]
    tp, pred, exp = (sum(x[k] for x in m) for k in ("true_positives", "predicted_count", "expected_count"))
    return {**_prf(tp, pred, exp), "true_positives": tp, "predicted_count": pred, "expected_count": exp, "records_measured": len(m)}


def _aggregate(rs: List[Dict[str, Any]], in_scope: int) -> Dict[str, Any]:
    n = len(rs)
    labels = {}
    for lab in LABELS:
        tp = sum(1 for x in rs if x["expected_decision"] == lab and x["predicted_decision"] == lab)
        pred, exp = sum(1 for x in rs if x["predicted_decision"] == lab), sum(1 for x in rs if x["expected_decision"] == lab)
        labels[lab] = {**_prf(tp, pred, exp), "true_positives": tp, "predicted_count": pred, "expected_count": exp}
    f1s = [v["f1"] for v in labels.values() if isinstance(v["f1"], float)]
    dec_acc, _ = _bool_acc(rs, "decision_correct")
    cm = [x for x in rs if isinstance(x["contradiction_detection_correct"], bool)]
    tp = sum(1 for x in cm if x["expected_contradiction_present"] and x["predicted_contradiction_detected"])
    fp = sum(1 for x in cm if not x["expected_contradiction_present"] and x["predicted_contradiction_detected"])
    fn = sum(1 for x in cm if x["expected_contradiction_present"] and not x["predicted_contradiction_detected"])
    tn = len(cm) - tp - fp - fn
    gm = [x for x in rs if x["cited_doc_count"]]
    cited, valid = sum(x["cited_doc_count"] for x in gm), sum(x["grounded_cited_doc_count"] for x in gm)
    miss_acc, miss_n = _bool_acc(rs, "missing_flag_correct")
    exp_ins, unsupported, committed = (sum(1 for x in rs if x["expected_decision"] == INS), sum(1 for x in rs if x["unsupported_decision"]),
                                       sum(1 for x in rs if x["committed_decision"]))
    em = [x for x in rs if isinstance(x["escalation_correct"], bool)]
    changed = [x for x in em if x["expected_escalation_change"]]
    unchanged = [x for x in em if not x["expected_escalation_change"]]
    esc_acc, _ = _bool_acc(rs, "escalation_correct")
    chg_acc, _ = _bool_acc(rs, "escalation_change_correct")
    return {
        "records_in_scope": in_scope, "case_count": n, "coverage": _ratio(n, in_scope),
        "decision": {"accuracy": dec_acc, "macro_f1": sum(f1s) / len(f1s) if f1s else NOT_MEASURED, "by_label": labels},
        "evidence_grounding": {"grounded_citation_rate": _ratio(valid, cited), "cited_doc_count": cited, "grounded_cited_doc_count": valid,
                               "supporting": _role_agg(rs, "supporting_evidence_metrics"), "contradicting": _role_agg(rs, "contradicting_evidence_metrics"),
                               "pooled_doc_roles": _role_agg(rs, "pooled_evidence_metrics"), "missing_strict": _role_agg(rs, "missing_evidence_metrics"),
                               "missing_flag_accuracy": miss_acc, "missing_flag_records_measured": miss_n},
        "contradiction_detection": {"accuracy": _ratio(tp + tn, len(cm)), **_prf(tp, tp + fp, tp + fn), "false_positive_rate": _ratio(fp, fp + tn),
                                    "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn, "records_measured": len(cm)},
        "unsupported_decision": {"rate": _ratio(unsupported, exp_ins), "rate_among_committed": _ratio(unsupported, committed),
                                 "unsupported_count": unsupported, "expected_insufficient_count": exp_ins, "committed_count": committed},
        "escalation": {"accuracy": esc_acc, "change_accuracy": chg_acc,
                       "change_recall": _ratio(sum(1 for x in changed if x["predicted_escalation_change"]), len(changed)),
                       "change_false_alarm_rate": _ratio(sum(1 for x in unchanged if x["predicted_escalation_change"]), len(unchanged)),
                       "expected_change_count": len(changed), "expected_no_change_count": len(unchanged), "records_measured": len(em)}}


def _ordered(values: Iterable[str], preferred: Tuple[str, ...]) -> List[str]:
    vs = set(values)
    return [v for v in preferred if v in vs] + sorted(vs - set(preferred))


def evaluate_adversarial_predictions(records: Iterable[Any], predictions: Optional[Iterable[Dict[str, Any]]], source_cases: Optional[List[Dict[str, Any]]] = None,
                                     splits=None, perturbation_types=None) -> Dict[str, Any]:
    """Score supplied predictions against the expected labels carried by `records` (PerturbationRecord objects or their dicts).
    `predictions` must be None (nothing supplied) or a list/tuple of prediction records; only predicted perturbations are scored and coverage is reported.
    `source_cases` (optional) additionally validates the records against their source cases. Returns status MEASURED / NOT_MEASURED / INVALID_RECORDS / INVALID_PREDICTIONS."""
    res: Dict[str, Any] = {"evaluator": {"name": EVALUATOR_NAME, "version": EVALUATOR_VERSION}, "errors": []}
    errs, recs = _check_records(records)
    if not errs and source_cases is not None: errs = validate_adversarial_suite(recs, source_cases)
    if errs: return {**res, "status": STATUS_INVALID_RECORDS, "errors": errs}
    sel_splits = _resolve_filter(splits, SPLIT_ORDER, "splits", {r["provenance"]["source_split"] for r in recs})
    sel_types = _resolve_filter(perturbation_types, PERTURBATION_TYPES, "perturbation_types", ())
    scope = [r for r in recs if (sel_splits is None or r["provenance"]["source_split"] in sel_splits) and (sel_types is None or r["perturbation_type"] in sel_types)]
    res.update({"splits_evaluated": list(sel_splits) if sel_splits else None, "perturbation_types_evaluated": list(sel_types) if sel_types else None, "records_in_scope": len(scope)})
    if predictions is not None and not isinstance(predictions, (list, tuple)):
        return {**res, "status": STATUS_INVALID_PREDICTIONS, "errors": [f"predictions must be a list or tuple of records, got {type(predictions).__name__}"]}
    accepted: Dict[str, Dict[str, Any]] = {}
    if predictions:
        perrs, accepted = _check_predictions(predictions, {r["perturbation_id"]: r for r in recs}, {r["perturbation_id"] for r in scope})
        if perrs: return {**res, "status": STATUS_INVALID_PREDICTIONS, "errors": perrs}
    results = [_eval_record(r, accepted[r["perturbation_id"]]) for r in scope if r["perturbation_id"] in accepted]  # record order => stable
    group = lambda key, order: {g: _aggregate([x for x in results if x[key] == g], sum(1 for r in scope if _key(r, key) == g))
                                for g in _ordered((_key(r, key) for r in scope), order)}
    res.update({"status": STATUS_MEASURED if results else STATUS_NOT_MEASURED, "reason": "" if results else "no predictions supplied; nothing was measured",
                "case_count": len(results), "coverage": _ratio(len(results), len(scope)), "complete": bool(results) and len(results) == len(scope),
                "overall": _aggregate(results, len(scope)), "by_perturbation_type": group("perturbation_type", PERTURBATION_TYPES), "by_split": group("split", SPLIT_ORDER),
                "by_outcome_change": group("outcome_change", OUTCOME_ORDER), "per_perturbation": results,
                "run_provenance": {"records_sha256": _sha(recs), "predictions_sha256": _sha(list(predictions)) if predictions else None, "randomness": "none", "network": "none",
                                   "ground_truth_source": "perturbation records only; predictions never supply or alter labels"}})
    return res


def _key(r: Dict[str, Any], key: str) -> str: return r["provenance"]["source_split"] if key == "split" else r[key]


# ----------------------------------------------------------------------------- V1 vs COMPLETE comparison (supplied predictions only; additive, evaluate_adversarial_predictions is unchanged)
SYSTEM_V1, SYSTEM_COMPLETE = "V1", "COMPLETE"
COMPARISON_CLAIM = ("Descriptive comparison of two SUPPLIED prediction sets on the paired perturbations only; no superiority claim, no general accuracy claim. "
                    "Neither system is run by this module and ground truth comes only from the perturbation records.")
_METRICS = (  # (name, path into an aggregate, higher_is_better; None = descriptive rate, no direction)
    ("decision_accuracy", ("decision", "accuracy"), True), ("decision_macro_f1", ("decision", "macro_f1"), True),
    *((f"decision_f1[{lab}]", ("decision", "by_label", lab, "f1"), True) for lab in LABELS),
    ("grounded_citation_rate", ("evidence_grounding", "grounded_citation_rate"), True), ("supporting_evidence_f1", ("evidence_grounding", "supporting", "f1"), True),
    ("contradicting_evidence_f1", ("evidence_grounding", "contradicting", "f1"), True), ("pooled_evidence_f1", ("evidence_grounding", "pooled_doc_roles", "f1"), True),
    ("missing_flag_accuracy", ("evidence_grounding", "missing_flag_accuracy"), True),
    ("contradiction_accuracy", ("contradiction_detection", "accuracy"), True), ("contradiction_recall", ("contradiction_detection", "recall"), True),
    ("contradiction_f1", ("contradiction_detection", "f1"), True), ("contradiction_false_positive_rate", ("contradiction_detection", "false_positive_rate"), False),
    ("unsupported_decision_rate", ("unsupported_decision", "rate"), False), ("unsupported_rate_among_committed", ("unsupported_decision", "rate_among_committed"), False),
    ("escalation_accuracy", ("escalation", "accuracy"), True), ("escalation_change_recall", ("escalation", "change_recall"), True),
    ("escalation_change_false_alarm_rate", ("escalation", "change_false_alarm_rate"), False))
_DIRECTION = {name: hib for name, _, hib in _METRICS}
_DIRECTION.update({"escalation_rate": None, "expected_escalation_rate": None})


def _get(agg: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    for k in path: agg = agg[k]
    return agg


def _is_num(v) -> bool: return isinstance(v, (int, float)) and not isinstance(v, bool)


def _delta(v1, cx) -> Dict[str, Any]:
    """absolute_change = COMPLETE - V1 (both must be measured); relative_change = absolute / V1 only when V1 is a valid non-zero denominator."""
    ok = _is_num(v1) and _is_num(cx)
    ab = cx - v1 if ok else NOT_MEASURED
    return {"v1": v1, "complete": cx, "absolute_change": ab, "relative_change": ab / v1 if ok and v1 != 0 else NOT_MEASURED}


def _flat(agg: Dict[str, Any], per: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {name: _get(agg, path) for name, path, _ in _METRICS}
    m = [x for x in per if isinstance(x["predicted_escalation"], bool)]
    out["escalation_rate"] = _ratio(sum(1 for x in m if x["predicted_escalation"]), len(m))  # share of paired perturbations where the system escalates
    out["expected_escalation_rate"] = _ratio(sum(1 for x in m if x["expected_escalation"]), len(m))  # ground-truth rate over the same perturbations
    return out


def _group_compare(a1, a2, per1, per2) -> Dict[str, Any]:
    f1, f2 = _flat(a1, per1), _flat(a2, per2)
    return {"case_count": a1["case_count"], "metrics": {k: {**_delta(f1[k], f2[k]), "higher_is_better": _DIRECTION[k]} for k in f1}}


def compare_systems(records: Iterable[Any], v1_predictions: Optional[Iterable[Dict[str, Any]]], complete_predictions: Optional[Iterable[Dict[str, Any]]],
                    source_cases: Optional[List[Dict[str, Any]]] = None, splits=None, perturbation_types=None) -> Dict[str, Any]:
    """Compare two SUPPLIED prediction sets (V1, COMPLETE) on the IDENTICAL perturbation records. Only perturbations predicted by BOTH systems are scored, for both systems;
    the rest are listed as unpaired and never scored. Every metric comes from evaluate_adversarial_predictions, so ground truth is read only from the records.
    absolute_change = COMPLETE - V1; relative_change = absolute / V1 when V1 is a valid non-zero measured value; otherwise NOT_MEASURED. Nothing is invoked or fabricated:
    a missing prediction set yields NOT_MEASURED."""
    kw = {"splits": splits, "perturbation_types": perturbation_types}
    out: Dict[str, Any] = {"comparison": {"systems": [SYSTEM_V1, SYSTEM_COMPLETE], "delta": "COMPLETE minus V1", "claim": COMPARISON_CLAIM}, "errors": {}}
    sets = ((SYSTEM_V1, v1_predictions), (SYSTEM_COMPLETE, complete_predictions))
    first = {n: evaluate_adversarial_predictions(records, p, source_cases=source_cases, **kw) for n, p in sets}  # full validation of records and of each prediction set
    bad = {n: r for n, r in first.items() if r["status"] in (STATUS_INVALID_RECORDS, STATUS_INVALID_PREDICTIONS)}
    if bad:
        status = STATUS_INVALID_RECORDS if any(r["status"] == STATUS_INVALID_RECORDS for r in bad.values()) else STATUS_INVALID_PREDICTIONS
        return {**out, "status": status, "errors": {n: r["errors"] for n, r in bad.items()}}
    ids = {n: {x["perturbation_id"] for x in first[n]["per_perturbation"]} for n, _ in sets}
    paired = ids[SYSTEM_V1] & ids[SYSTEM_COMPLETE]
    _, recs = _check_records(records)
    scope_ids = {x["perturbation_id"] for x in recs}
    paired_recs = [r for r in recs if r["perturbation_id"] in paired]  # identical records and order for both systems
    res = {n: evaluate_adversarial_predictions(paired_recs, [x for x in p if x["perturbation_id"] in paired] if paired else None, **kw) for n, p in sets}
    a1, a2 = res[SYSTEM_V1], res[SYSTEM_COMPLETE]
    assert a1["run_provenance"]["records_sha256"] == a2["run_provenance"]["records_sha256"] if paired else True
    group = lambda key, order: {g: _group_compare(a1[key][g], a2[key][g], [x for x in a1["per_perturbation"] if x[{"by_perturbation_type": "perturbation_type", "by_split": "split"}[key]] == g],
                                                    [x for x in a2["per_perturbation"] if x[{"by_perturbation_type": "perturbation_type", "by_split": "split"}[key]] == g]) for g in a1[key]}
    by2 = {x["perturbation_id"]: x for x in a2["per_perturbation"]}
    side = [{"perturbation_id": x["perturbation_id"], "source_case_id": x["source_case_id"], "perturbation_type": x["perturbation_type"], "split": x["split"], "outcome_change": x["outcome_change"],
             "expected_decision": x["expected_decision"], "v1_decision": x["predicted_decision"], "complete_decision": by2[x["perturbation_id"]]["predicted_decision"],
             "v1_correct": x["decision_correct"], "complete_correct": by2[x["perturbation_id"]]["decision_correct"],
             "perturbation_provenance": copy.deepcopy(x["provenance"]["perturbation_provenance"])} for x in a1["per_perturbation"]]
    out.update({"status": STATUS_MEASURED if paired else STATUS_NOT_MEASURED,
                "reason": "" if paired else "no perturbation has predictions from both systems; nothing was measured",
                "paired": {"records_in_scope": len(scope_ids), "paired_count": len(paired), "coverage": _ratio(len(paired), len(scope_ids)),
                           "v1_only": sorted(ids[SYSTEM_V1] - paired), "complete_only": sorted(ids[SYSTEM_COMPLETE] - paired), "same_records_for_both": True},
                "overall": _group_compare(a1["overall"], a2["overall"], a1["per_perturbation"], a2["per_perturbation"]),
                "by_perturbation_type": group("by_perturbation_type", PERTURBATION_TYPES), "by_split": group("by_split", SPLIT_ORDER),
                "per_perturbation": side, "systems": {SYSTEM_V1: a1, SYSTEM_COMPLETE: a2},
                "run_provenance": {"records_sha256": a1["run_provenance"]["records_sha256"] if paired else None,
                                   "v1_predictions_sha256": a1["run_provenance"]["predictions_sha256"] if paired else None,
                                   "complete_predictions_sha256": a2["run_provenance"]["predictions_sha256"] if paired else None, "randomness": "none", "network": "none",
                                   "ground_truth_source": "perturbation records only; neither system supplies or alters labels"}})
    return out


# ----------------------------------------------------------------------------- static interface probe (never imports or executes the project code)
V1_ENTRY, COMPLETE_ENTRY = "analyze_compliance_with_matrix", "build_evidence_graph"


def _importable(name: str) -> bool:
    try: return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError): return False


def probe_system_interfaces(tasks_path: str) -> Dict[str, Any]:
    """Statically inspect tasks.py (ast + importlib.util.find_spec only; nothing is imported or run) and report whether V1 and the complete pipeline can be driven
    on OMNI-Bench perturbations. Reports blockers; never returns a performance number."""
    if not os.path.isfile(tasks_path): return {"status": "UNAVAILABLE", "reason": f"tasks file not found: {tasks_path}", "performance_numbers": NOT_MEASURED}
    with open(tasks_path, "r", encoding="utf-8", errors="replace") as f: src = f.read()
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    mods, relative = set(), []
    for n in tree.body:
        if isinstance(n, ast.Import): mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level: relative.append("." * n.level + (n.module or ""))
            elif n.module: mods.add(n.module)
    missing = sorted(m for m in mods if not _importable(m))
    seg = lambda name: (ast.get_source_segment(src, funcs[name]) or "") if name in funcs else ""
    v1_net = bool(re.search(r"http_session\.|requests\.|\.post\(|ollama", seg(V1_ENTRY), re.I))
    adapters = sorted(n for n in funcs if re.search(r"omni_?bench", n, re.I) and re.search(r"predict|adapter|infer|run", n, re.I))
    verdicts = [v for v in ("VIOLATION", "SATISFIED", "INCONCLUSIVE", "NOT_APPLICABLE", "UNEVALUATED") if f'"{v}"' in src]
    blockers: List[str] = []
    if missing: blockers.append(f"tasks.py cannot be imported here: missing module(s) {missing}")
    if relative: blockers.append(f"tasks.py uses package-relative import(s) {sorted(set(relative))}; it must be imported inside its package")
    if V1_ENTRY not in funcs or v1_net: blockers.append(f"V1 decision path ({V1_ENTRY}, is_v2=False) is an LLM/network call returning free-text report; it has no deterministic offline decision output")
    if V1_ENTRY in funcs: blockers.append("V1 free-text verdicts have no existing mapping to COMPLIANT / NON_COMPLIANT / INSUFFICIENT_EVIDENCE, doc_id evidence, contradiction or escalation fields")
    if COMPLETE_ENTRY not in funcs: blockers.append(f"complete pipeline entry point {COMPLETE_ENTRY} not found")
    else: blockers.append(f"{COMPLETE_ENTRY} takes file paths (documents + rulebook) and returns an evidence graph whose Decision verdicts use {verdicts}; "
                          "there is no existing adapter from perturbation records to its inputs or from its graph to OMNI-Bench labels (decision, doc_id roles, contradiction, escalation)")
    if not adapters: blockers.append("no OMNI-Bench prediction adapter exists in tasks.py; writing one would be inventing a decision mapping and is deliberately not done here")
    return {"status": "BLOCKED" if blockers else "AVAILABLE_UNVERIFIED", "blockers": blockers, "missing_modules": missing, "relative_imports": sorted(set(relative)),
            "v1": {"entry_point": V1_ENTRY, "found": V1_ENTRY in funcs, "calls_llm_or_network": v1_net, "outputs_omni_bench_labels": False},
            "complete": {"entry_point": COMPLETE_ENTRY, "found": COMPLETE_ENTRY in funcs, "verdict_vocabulary": verdicts, "outputs_omni_bench_labels": False},
            "omni_bench_adapters_found": adapters, "performance_numbers": NOT_MEASURED,
            "required_to_unblock": "supply V1 and COMPLETE predictions produced by the real systems (e.g. in the full project environment with a vetted perturbation-to-input and output-to-label adapter), then pass them to compare_systems"}


# ----------------------------------------------------------------------------- 11C adapters: the SAME perturbation records through V1 (existing path) and COMPLETE (existing pipeline)
# Nothing below scores, derives labels, or invents a decision parser. evaluate_adversarial_predictions / compare_systems (above) stay pure and unchanged.
#   COMPLETE: complete_predictions() drives the EXISTING research pipeline through an explicitly INJECTED project module, calling its functions in the exact order of the
#             production task (tasks.process_document_batch_task, local files) INCLUDING the compiled-policy stage, then tasks.build_uncertainty_assessment, and copies the
#             structured fields. tasks.build_evidence_graph (used by every existing benchmark) is NOT that path: it omits apply_compiled_policy. This module never imports the project code.
#   V1      : the actual V1 path (tasks.run_v1_v2_experiment -> analyze_compliance_with_matrix, an LLM/network call returning free text) is NEVER run here.
#             v1_experiment_cases() only prepares its inputs; the decision of a V1 report exists only as a HUMAN reading (the existing `reviewed_verdicts` mechanism),
#             and every V1 metric without such a reading is NOT_MEASURED with a precise reason.
ADAPTER_VERSION = "11C-1.0"
V1_OBJECTIVE = "Assess whether the supplied documents comply with the supplied policy."  # fixed, label-free, identical for every record (analyze_compliance_with_matrix requires an objective)
UNC_STATE_TO_LABEL = {COMP: COMP, NC: NC, INS: INS}  # tasks.UNC_STATES minus CONDITIONAL, which names no OMNI-Bench label (never mapped unless the caller says so)
ENGINE_VERDICT_TO_LABEL = {"SATISFIED": COMP, "VIOLATION": NC, "INCONCLUSIVE": INS, "NOT_APPLICABLE": INS}  # mirrors tasks._UNCB_VERDICT_STATE; UNEVALUATED has no OMNI label
LABEL_TO_ENGINE_VERDICT = {COMP: "SATISFIED", NC: "VIOLATION", INS: "INCONCLUSIVE"}  # independent ground truth expressed in the existing engine vocabulary
_REVIEW_VERDICTS = frozenset(ENGINE_VERDICT_TO_LABEL) | {"UNEVALUATED"}
_REVIEW_KEYS = frozenset(("perturbation_id", "reviewed_verdict", "reviewer", "reviewed_at", "notes"))
_RULE_ID_RE = re.compile(r"\b[A-Z]{2,5}-\d+\.\d+")
_V1_REASON = {
    "decision": "V1 verdicts are free LLM text and no automatic decision parser exists (none is invented); measured only for perturbations whose V1 report has a human review (reviewed_verdicts)",
    "evidence": "a V1 report has no structured, role-separated doc_id evidence; the existing evaluate_report_citations is a string-match proxy reported separately as v1_report_citation_proxy",
    "contradiction": "V1 has no graph and no contradiction output (its report is free text); no automatic or reviewed contradiction field exists",
    "escalation": "V1 has no escalation output (its report is free text); no automatic or reviewed escalation field exists"}
_SUP, _CON, _MIS = "predicted_supporting_evidence", "predicted_contradicting_evidence", "predicted_missing_evidence"
_METRIC_FIELDS = {"grounded_citation_rate": (_SUP, _CON), "supporting_evidence_f1": (_SUP,), "contradicting_evidence_f1": (_CON,), "pooled_evidence_f1": (_SUP, _CON), "missing_flag_accuracy": (_MIS,),
                  "expected_escalation_rate": ("predicted_escalation",), "escalation": ("predicted_escalation",), "contradiction": ("predicted_contradiction_detected",),
                  "decision": ("predicted_decision",), "unsupported": ("predicted_decision",)}
_COMPLETE_MISSING = "the COMPLETE system emits missing-evidence KINDS, not OMNI-style descriptions, so the adapter supplies none (no mapping is invented)"


def _valid_records(records) -> List[Dict[str, Any]]:
    errs, recs = _check_records(records)
    if errs: raise ValueError("invalid perturbation records: " + "; ".join(errs[:5]))
    return recs


def _echo(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"perturbation_id": r["perturbation_id"], "source_case_id": r["source_case_id"], "source_family_id": r["source_family_id"],
            "perturbation_type": r["perturbation_type"], "split": r["provenance"]["source_split"]}


def _uniq(xs) -> List[str]:
    return list(dict.fromkeys(x for x in xs if isinstance(x, str) and x.strip()))


_PRODUCTION_STAGES = ("_init_graph", "_build_policy_nodes", "_ingest_facts", "link_shared_terms", "link_cross_documents", "apply_compiled_policy", "evaluate_policy_rules",
                      "detect_contradictions", "classify_contradictions")  # first-call order inside tasks.process_document_batch_task (checked statically by a regression test)


def _production_graph(tasks, r: Dict[str, Any], wd: str):
    """The EXISTING functions, in the exact order process_document_batch_task calls them for local files. Differs from tasks.build_evidence_graph ONLY by the apply_compiled_policy
    stage (which only the production task calls). Like the task, a failing link / compile / rule / contradiction stage is recorded and the pipeline continues."""
    paths = []
    for d in r["perturbed_document_set"]:
        if os.path.basename(d["doc_id"]) != d["doc_id"] or d["doc_id"] == "rulebook.txt": raise ValueError(f"{r['perturbation_id']}: doc_id {d['doc_id']!r} cannot be a file name")
        paths.append(os.path.join(wd, d["doc_id"]))
        with open(paths[-1], "w", encoding="utf-8") as fh: fh.write(d["text"])
    rb = os.path.join(wd, "rulebook.txt")
    with open(rb, "w", encoding="utf-8") as fh: fh.write(r["perturbed_policy"])
    G = tasks._init_graph()
    rulebook_text, cache, errs = tasks._build_policy_nodes(G, rb), {}, []
    for pth in paths:
        if not tasks.is_sensitive_file(pth): tasks._ingest_facts(G, tasks._add_document_node(G, pth), os.path.basename(pth), pth, cache)
    def stage(name, fn, *a):
        try: return fn(*a)
        except Exception as e: errs.append(f"{name}: {type(e).__name__}: {str(e)[:150]}")
    stage("link_shared_terms", tasks.link_shared_terms, G)
    stage("link_cross_documents", tasks.link_cross_documents, G)
    comp = stage("apply_compiled_policy", tasks.apply_compiled_policy, G, rulebook_text)
    stage("evaluate_policy_rules", tasks.evaluate_policy_rules, G)
    if getattr(tasks, "ENABLE_CONTRADICTION_HEURISTIC", True): stage("detect_contradictions", tasks.detect_contradictions, G)
    stage("classify_contradictions", tasks.classify_contradictions, G)
    ran = isinstance(comp, dict) and not comp.get("error") and not comp.get("evaluation_error")  # tasks.apply_compiled_policy records a rule-engine crash in evaluation_error, not error
    return G, {"stage_errors": errs, "compiled_policy_stage": {"ran": ran, "status": comp.get("status") if isinstance(comp, dict) else None,
                                                              "error": (comp.get("error") or (f"compiled rule evaluation failed: {comp['evaluation_error']}" if comp.get("evaluation_error") else None)) if isinstance(comp, dict) else "stage did not run (compiler disabled / unavailable / no rules / exception)"}}


def complete_predictions(records, tasks, conditional_label: Optional[str] = None, require_compiled_policy: bool = True) -> Dict[str, Any]:
    """COMPLETE-system predictions for EVERY record, from the frozen production research path via the injected project module `tasks` (needs the functions in _PRODUCTION_STAGES plus
    _add_document_node, is_sensitive_file, _nodes_of_type, build_uncertainty_assessment). The compiled-policy stage (an LLM call inside the injected pipeline) is part of that path:
    with require_compiled_policy=True (default) a record whose compile stage did not run cleanly is SKIPPED with the reason, never scored as a partial-pipeline result; False allows
    legacy-engine-only results (recorded in provenance). One Decision node per rule line: exactly one is required (no rule selector is guessed), otherwise SKIPPED.
    predicted_decision = uncertainty_status; CONDITIONAL names no OMNI-Bench label, so it is skipped unless `conditional_label` (a decision label) is passed explicitly.
    predicted_escalation = escalation_required; supporting / contradicting = source filenames (== doc_ids, files are written under their doc_id) of the assessment's evidence;
    predicted_contradiction_detected = any contradicting evidence (policy conflicts have no existing detector, so they are not claimed). predicted_missing_evidence is NOT supplied:
    the system emits missing-evidence KINDS, not OMNI-style descriptions, so those metrics stay NOT_MEASURED. Skipped records are never scored."""
    if conditional_label is not None and conditional_label not in LABELS: raise ValueError(f"conditional_label must be one of {LABELS} or None")
    recs, preds, skipped = _valid_records(records), [], []
    for r in recs:
        try:
            with tempfile.TemporaryDirectory() as wd:
                G, info = _production_graph(tasks, r, wd)
                decs = sorted(n for n, _ in tasks._nodes_of_type(G, "Decision"))
                if require_compiled_policy and not info["compiled_policy_stage"]["ran"]:
                    skipped.append({"perturbation_id": r["perturbation_id"], "reason": f"compiled-policy stage did not run cleanly ({info['compiled_policy_stage']['error']}); not scored as the production path"}); continue
                if len(decs) != 1: skipped.append({"perturbation_id": r["perturbation_id"], "reason": f"{len(decs)} Decision nodes; exactly one is required (no rule selector is guessed)"}); continue
                a = tasks.build_uncertainty_assessment(G, decs[0])
        except Exception as e:  # an adapter failure is reported, never silently scored
            skipped.append({"perturbation_id": r["perturbation_id"], "reason": f"{type(e).__name__}: {str(e)[:200]}"}); continue
        state = a.get("uncertainty_status")
        label = UNC_STATE_TO_LABEL.get(state) or (conditional_label if state == "CONDITIONAL" else None)
        if not a.get("found") or label is None:
            skipped.append({"perturbation_id": r["perturbation_id"], "reason": f"uncertainty_status {state!r} has no OMNI-Bench decision label; not mapped, no prediction made"}); continue
        con = [f for it in a.get("contradicting_evidence") or [] for f in [*[d.get("filename") for d in it.get("documents") or [] if isinstance(d, dict)],
                                                                               *[s.get("filename") for s in (it.get("claim_a"), it.get("claim_b")) if isinstance(s, dict)]]]
        preds.append({**_echo(r), "predicted_decision": label, "predicted_escalation": bool(a.get("escalation_required")),
                      "predicted_supporting_evidence": _uniq(x.get("filename") for x in a.get("supporting_evidence") or []), "predicted_contradicting_evidence": _uniq(con),
                      "predicted_contradiction_detected": bool(a.get("contradicting_evidence")),
                      "provenance": {"system": SYSTEM_COMPLETE, "adapter": ADAPTER_VERSION, "path": "tasks.process_document_batch_task stage order (incl. apply_compiled_policy) + tasks.build_uncertainty_assessment",
                                     "engine_verdict": (a.get("decision") or {}).get("verdict"), "uncertainty_status": state,
                                     "escalation_reason_codes": sorted(e.get("reason") for e in a.get("escalation_reasons") or []), "conditional_label": conditional_label, **info}})
    return {"predictions": preds, "skipped": skipped, "records_in_scope": len(recs),
            "adapter": {"version": ADAPTER_VERSION, "system": SYSTEM_COMPLETE, "require_compiled_policy": require_compiled_policy,
                        "randomness": "none in this module; the injected compiled-policy stage is an LLM call (policy_compiler) and may vary between runs", "llm_or_network": "only inside the injected pipeline's compile stage"}}


def _rule_selector(policy: str) -> Optional[str]:
    ids = {m.split("(")[0] for m in _RULE_ID_RE.findall(policy or "")}
    return next(iter(ids)) if len(ids) == 1 else None


def v1_experiment_cases(records, workdir: str, reviews: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Inputs for the EXISTING tasks.run_v1_v2_experiment (case_id == perturbation_id; files named by doc_id; rulebook = the perturbed policy; fixed V1_OBJECTIVE).
    ground_truth is built ONLY from the record's independent expected_decision (engine vocabulary) and, when given, the human `reviews` of the V1 report
    (-> the existing reviewed_verdicts). The rule selector is the policy's single rule id; with none or several, no expectation is written and the existing
    V1 verdict metrics stay NOT_MEASURED for that record. This function never runs V1 (run_v1_v2_experiment(..., run_llm=True) is an LLM call made by the caller)."""
    recs, by_rev = _valid_records(records), {x["perturbation_id"]: x for x in reviews or [] if isinstance(x, dict)}
    cases = []
    for r in recs:
        d = os.path.join(workdir, r["perturbation_id"]); os.makedirs(d, exist_ok=True)
        paths = []
        for doc in r["perturbed_document_set"]:
            if os.path.basename(doc["doc_id"]) != doc["doc_id"] or doc["doc_id"] == "rulebook.txt": raise ValueError(f"{r['perturbation_id']}: doc_id {doc['doc_id']!r} cannot be a file name")
            paths.append(os.path.join(d, doc["doc_id"]))
            with open(paths[-1], "w", encoding="utf-8") as fh: fh.write(doc["text"])
        rb = os.path.join(d, "rulebook.txt")
        with open(rb, "w", encoding="utf-8") as fh: fh.write(r["perturbed_policy"])
        sel, gt = _rule_selector(r["perturbed_policy"]), {"labeller": "OMNI-Bench perturbation record", "source": "perturbation record expected_decision (independent of both systems)"}
        if sel:
            gt["expected_verdicts"] = [{"rule_contains": sel, "verdict": LABEL_TO_ENGINE_VERDICT[r["expected_decision"]]}]
            rv = by_rev.get(r["perturbation_id"])
            if rv: gt["reviewed_verdicts"] = {"V1": [{"rule_contains": sel, "verdict": str(rv.get("reviewed_verdict")).strip().upper()}]}
        cases.append({"case_id": r["perturbation_id"], "files": paths, "rulebook": rb, "objective": V1_OBJECTIVE, "ground_truth": gt})
    return cases


def v1_predictions_from_reviews(records, reviews) -> Dict[str, Any]:
    """V1 decision predictions from HUMAN readings of the V1 reports: reviews = [{perturbation_id, reviewed_verdict (engine vocabulary, as in reviewed_verdicts), reviewer?, reviewed_at?, notes?}].
    Mapped only through ENGINE_VERDICT_TO_LABEL; UNEVALUATED (no OMNI label) is skipped. Only predicted_decision is produced: V1 supplies nothing else. Invalid reviews -> INVALID_REVIEWS."""
    recs = {r["perturbation_id"]: r for r in _valid_records(records)}
    errs, preds, skipped, seen = [], [], [], set()
    for i, rv in enumerate(reviews or []):
        pid = rv.get("perturbation_id") if isinstance(rv, dict) else None
        tag = f"review[{i}] ({pid})"
        if not isinstance(rv, dict): errs.append(f"review[{i}]: not an object"); continue
        if pid not in recs: errs.append(f"{tag}: unknown perturbation_id"); continue
        if pid in seen: errs.append(f"{tag}: duplicate review")
        seen.add(pid)
        extra = sorted(str(k) for k in rv if k not in _REVIEW_KEYS)
        if extra: errs.append(f"{tag}: unknown or ground-truth field(s) not allowed in a review: {extra}")
        v = str(rv.get("reviewed_verdict") or "").strip().upper()
        if v not in _REVIEW_VERDICTS: errs.append(f"{tag}: reviewed_verdict must be one of {sorted(_REVIEW_VERDICTS)}"); continue
        if v not in ENGINE_VERDICT_TO_LABEL: skipped.append({"perturbation_id": pid, "reason": f"reviewed_verdict {v} has no OMNI-Bench decision label; not mapped"}); continue
        preds.append({**_echo(recs[pid]), "predicted_decision": ENGINE_VERDICT_TO_LABEL[v],
                      "provenance": {"system": SYSTEM_V1, "adapter": ADAPTER_VERSION, "source": "manual review of the V1 report (reviewed_verdicts)", "reviewed_verdict": v,
                                     "reviewer": rv.get("reviewer"), "reviewed_at": rv.get("reviewed_at")}})
    if errs: return {"status": "INVALID_REVIEWS", "errors": sorted(errs), "predictions": [], "skipped": []}
    return {"status": STATUS_MEASURED if preds else STATUS_NOT_MEASURED, "errors": [], "predictions": preds, "skipped": skipped, "reviews_supplied": len(reviews or [])}


def v1_report_citation_metrics(v1_experiment_records, records) -> Dict[str, Any]:
    """Pools the citation-validity PROXY the existing build_experiment_record already stored in V1 experiment records (evaluate_report_citations: 'file @ location' citations that match
    real evidence). It describes what the V1 report cites, NOT decision correctness and NOT role-separated evidence, so it is never mixed into evidence_grounding."""
    recs = {r["perturbation_id"]: r for r in _valid_records(records)}
    rows = []
    for x in v1_experiment_records or []:
        if not isinstance(x, dict) or str(x.get("version")).upper() != "V1" or x.get("case_id") not in recs: continue
        cv = (x.get("measured") or {}).get("citation_validity")
        ok = isinstance(cv, dict) and cv.get("status") == "MEASURED"
        rows.append({"perturbation_id": x["case_id"], "status": "MEASURED" if ok else NOT_MEASURED, "valid": cv["valid"] if ok else None, "cited_syntactic": cv["cited_syntactic"] if ok else None})
    def pool(rs):
        m = [r for r in rs if r["status"] == "MEASURED"]
        v, c = sum(r["valid"] for r in m), sum(r["cited_syntactic"] for r in m)
        return {"reports": len(rs), "reports_with_citations": len(m), "valid": v, "cited_syntactic": c, "citation_validity": _ratio(v, c)}
    key = lambda r, k: recs[r["perturbation_id"]]["provenance"]["source_split"] if k == "split" else recs[r["perturbation_id"]][k]
    grp = lambda k, order: {g: pool([r for r in rows if key(r, k) == g]) for g in _ordered((key(r, k) for r in rows), order)}
    if not rows: return {"status": STATUS_NOT_MEASURED, "reason": "no V1 experiment records supplied (run_v1_v2_experiment(run_llm=True) is an LLM call that this module never makes)"}
    return {"status": STATUS_MEASURED, "method": "string_match_proxy (tasks.evaluate_report_citations via build_experiment_record)", "overall": pool(rows),
            "by_perturbation_type": grp("perturbation_type", PERTURBATION_TYPES), "by_split": grp("split", SPLIT_ORDER), "per_report": rows,
            "note": "what the V1 report cites; not decision correctness; not comparable to the structured evidence_grounding metrics"}


def _availability(system: str, preds: List[Dict[str, Any]], names) -> Dict[str, Any]:
    out = {}
    for n in names:
        key = n if n in _METRIC_FIELDS else next(k for k in _METRIC_FIELDS if n.startswith(k)) if any(n.startswith(k) for k in _METRIC_FIELDS) else "decision"
        have = sum(1 for p in preds if any(_has(p, f) for f in _METRIC_FIELDS[key]))
        grp = "evidence" if key in ("grounded_citation_rate", "supporting_evidence_f1", "contradicting_evidence_f1", "pooled_evidence_f1", "missing_flag_accuracy") else \
              "escalation" if "escalation" in key else "contradiction" if key == "contradiction" else "decision"
        if have: out[n] = {"status": "MEASURED_WHERE_SUPPLIED", "predictions_with_field": have}
        elif system == SYSTEM_V1: out[n] = {"status": NOT_MEASURED, "reason": _V1_REASON[grp]}
        else: out[n] = {"status": NOT_MEASURED, "reason": _COMPLETE_MISSING if key == "missing_flag_accuracy" else "no COMPLETE prediction carries this field (all records skipped or none supplied)"}
    return out


def compare_v1_vs_complete(records, v1_reviews, complete_output, v1_experiment_records=None, source_cases=None, splits=None, perturbation_types=None) -> Dict[str, Any]:
    """V1 vs COMPLETE on the IDENTICAL perturbation records. Delegates all scoring to compare_systems / evaluate_adversarial_predictions (unchanged; ground truth only from the records).
    Adds: per-system metric availability with precise NOT_MEASURED reasons, adapter skip accounting, the COMPLETE system over every record it predicted (so V1's review subset never
    shrinks it), and the V1 report-citation proxy. absolute_change = COMPLETE - V1; relative_change only where V1 is a valid non-zero denominator. No fabricated predictions."""
    v1 = v1_predictions_from_reviews(records, v1_reviews)
    if v1["status"] == "INVALID_REVIEWS": return {"status": "INVALID_REVIEWS", "errors": {SYSTEM_V1: v1["errors"]}}
    cx = complete_output or {"predictions": [], "skipped": [], "records_in_scope": None}
    kw = {"source_cases": source_cases, "splits": splits, "perturbation_types": perturbation_types}
    recs = _valid_records(records)  # the evaluator rejects out-of-scope predictions, so a split / type filter restricts BOTH prediction sets to the same scope first
    st, ty = _resolve_filter(splits, SPLIT_ORDER, "splits", {r["provenance"]["source_split"] for r in recs}), _resolve_filter(perturbation_types, PERTURBATION_TYPES, "perturbation_types", ())
    scope = {r["perturbation_id"] for r in recs if (st is None or r["provenance"]["source_split"] in st) and (ty is None or r["perturbation_type"] in ty)}
    dropped = {SYSTEM_V1: sum(1 for p in v1["predictions"] if p["perturbation_id"] not in scope), SYSTEM_COMPLETE: sum(1 for p in cx["predictions"] if p["perturbation_id"] not in scope)}
    v1 = {**v1, "predictions": [p for p in v1["predictions"] if p["perturbation_id"] in scope]}
    cx = {**cx, "predictions": [p for p in cx["predictions"] if p["perturbation_id"] in scope]}
    cmp = compare_systems(records, v1["predictions"] or None, cx["predictions"] or None, **kw)
    names = [n for n, _, _ in _METRICS] + ["escalation_rate", "expected_escalation_rate"]
    cmp["metric_availability"] = {SYSTEM_V1: _availability(SYSTEM_V1, v1["predictions"], names), SYSTEM_COMPLETE: _availability(SYSTEM_COMPLETE, cx["predictions"], names)}
    cmp["complete_system_all_predicted"] = evaluate_adversarial_predictions(records, cx["predictions"] or None, **kw)
    cmp["v1_report_citation_proxy"] = v1_report_citation_metrics(v1_experiment_records, records)
    cmp["adapter_report"] = {SYSTEM_V1: {"reviews_supplied": v1.get("reviews_supplied", 0), "predictions": len(v1["predictions"]), "skipped": v1["skipped"], "out_of_scope_dropped": dropped[SYSTEM_V1], "automatic_decisions": "none (no V1 decision parser)"},
                             SYSTEM_COMPLETE: {"predictions": len(cx["predictions"]), "skipped": cx["skipped"], "out_of_scope_dropped": dropped[SYSTEM_COMPLETE], "adapter_version": ADAPTER_VERSION}}
    cmp["comparison"] = {**cmp["comparison"], "adapter_claim": "Infrastructure result over supplied outputs only; V1 is never run here; no metric is reported without real system output."}
    return cmp


# ----------------------------------------------------------------------------- tests (fixtures are test inputs only; the runner never fabricates predictions)
def _rec(pid, ptype, split, outcome, dec, esc, src_esc, sup, con, missing, contra, docs, family="FAM-T-1"):
    return {"perturbation_id": pid, "source_case_id": f"SRC-{pid}", "source_family_id": family, "perturbation_type": ptype,
            "perturbed_document_set": [{"doc_id": d, "text": f"text {d}"} for d in docs], "perturbed_policy": "T-1.1: test policy.", "expected_decision": dec, "expected_escalation": esc,
            "expected_evidence_roles": {"supporting": sup, "contradicting": con, "missing": missing},
            "expected_contradiction_behavior": {"contradiction_present": contra, "conflicting_doc_ids": con, "conflicting_policy_rule_ids": [], "expected_behavior": "X", "rationale": "r"},
            "outcome_change": outcome, "outcome_change_reason": "t",
            "provenance": {"source_split": split, "source_expected_escalation": src_esc, "source_case_id": f"SRC-{pid}", "marker": f"prov-{pid}"}}


def _fixture():
    recs = [_rec("R1", "ocr_errors", "TRAIN", OUTCOME_UNCHANGED, COMP, False, False, ["d1"], [], [], False, ["d1"]),
            _rec("R2", "document_ordering", "TRAIN", OUTCOME_UNCHANGED, NC, True, True, ["d1", "d2"], [], [], False, ["d1", "d2"]),
            _rec("R3", "missing_evidence", "DEV", OUTCOME_ALTERED, INS, True, False, [], [], ["supporting evidence formerly in d1 (removed)"], False, ["d2"]),
            _rec("R4", "contradictory_evidence", "TEST", OUTCOME_ALTERED, INS, True, False, ["d1"], ["d2"], ["authoritative record resolving the dispute over d1"], True, ["d1", "d2"])]
    preds = [{"perturbation_id": "R1", "predicted_decision": COMP, "predicted_escalation": False, "predicted_supporting_evidence": ["d1"], "predicted_contradicting_evidence": [],
              "predicted_missing_evidence": [], "predicted_contradiction_detected": False, "provenance": {"system": "fixture-A"}},
             {"perturbation_id": "R2", "predicted_decision": COMP, "predicted_escalation": False, "predicted_supporting_evidence": ["d1", "d9"], "predicted_contradicting_evidence": [],
              "predicted_contradiction_detected": False},
             {"perturbation_id": "R3", "predicted_decision": COMP, "predicted_escalation": False, "predicted_supporting_evidence": ["d2"], "predicted_missing_evidence": [],
              "predicted_contradiction_detected": False},
             {"perturbation_id": "R4", "predicted_decision": INS, "predicted_escalation": True, "predicted_supporting_evidence": ["d1"], "predicted_contradicting_evidence": ["d2"],
              "predicted_missing_evidence": ["Authoritative  record resolving the dispute over d1"], "predicted_contradiction_detected": True}]
    return recs, preds


class AdversarialEvalTests(unittest.TestCase):
    def setUp(self):
        self.recs, self.preds = _fixture()
        self.res = evaluate_adversarial_predictions(self.recs, self.preds)

    def test_hand_computed_overall_metrics(self):
        o = self.res["overall"]
        self.assertEqual((self.res["status"], self.res["case_count"], self.res["coverage"], self.res["complete"]), (STATUS_MEASURED, 4, 1.0, True))
        self.assertAlmostEqual(o["decision"]["accuracy"], 0.5)
        self.assertAlmostEqual(o["decision"]["macro_f1"], (0.5 + 0.0 + 2 / 3) / 3)
        self.assertAlmostEqual(o["unsupported_decision"]["rate"], 0.5)
        self.assertAlmostEqual(o["unsupported_decision"]["rate_among_committed"], 1 / 3)
        e = o["escalation"]
        self.assertAlmostEqual(e["accuracy"], 0.5)
        self.assertAlmostEqual(e["change_accuracy"], 0.5)
        self.assertAlmostEqual(e["change_recall"], 0.5)
        self.assertAlmostEqual(e["change_false_alarm_rate"], 0.5)
        c = o["contradiction_detection"]
        self.assertEqual((c["accuracy"], c["precision"], c["recall"], c["f1"], c["false_positive_rate"]), (1.0, 1.0, 1.0, 1.0, 0.0))
        g = o["evidence_grounding"]
        self.assertAlmostEqual(g["grounded_citation_rate"], 5 / 6)  # d9 is a hallucinated citation
        self.assertEqual((g["supporting"]["precision"], g["supporting"]["recall"]), (3 / 5, 3 / 4))
        self.assertAlmostEqual(g["supporting"]["f1"], 2 / 3)
        self.assertEqual((g["contradicting"]["precision"], g["contradicting"]["recall"], g["contradicting"]["records_measured"]), (1.0, 1.0, 3))
        self.assertAlmostEqual(g["missing_flag_accuracy"], 2 / 3)
        self.assertEqual((g["missing_strict"]["precision"], g["missing_strict"]["recall"]), (1.0, 0.5))  # case/whitespace-normalized exact match

    def test_groups_by_type_split_and_outcome(self):
        r = self.res
        self.assertEqual(list(r["by_split"]), ["TRAIN", "DEV", "TEST"])
        self.assertEqual(list(r["by_outcome_change"]), [OUTCOME_UNCHANGED, OUTCOME_ALTERED])
        self.assertEqual(list(r["by_perturbation_type"]), ["document_ordering", "ocr_errors", "missing_evidence", "contradictory_evidence"])
        for key in ("by_perturbation_type", "by_split", "by_outcome_change"): self.assertEqual(sum(v["case_count"] for v in r[key].values()), 4)
        self.assertAlmostEqual(r["by_split"]["TRAIN"]["decision"]["accuracy"], 0.5)
        self.assertEqual(r["by_split"]["TEST"]["decision"]["accuracy"], 1.0)
        self.assertEqual(r["by_outcome_change"][OUTCOME_UNCHANGED]["unsupported_decision"]["rate"], NOT_MEASURED)  # no expected-INSUFFICIENT_EVIDENCE record in the group
        self.assertAlmostEqual(r["by_outcome_change"][OUTCOME_ALTERED]["unsupported_decision"]["rate"], 0.5)
        self.assertEqual(r["by_perturbation_type"]["ocr_errors"]["decision"]["accuracy"], 1.0)

    def test_no_predictions_is_not_measured_never_a_score(self):
        for preds in (None, [], ()):
            r = evaluate_adversarial_predictions(self.recs, preds)
            self.assertEqual((r["status"], r["case_count"], r["coverage"], r["complete"], r["per_perturbation"]), (STATUS_NOT_MEASURED, 0, 0.0, False, []))
            self.assertEqual(r["overall"]["decision"]["accuracy"], NOT_MEASURED)
            self.assertEqual(r["overall"]["unsupported_decision"]["rate"], NOT_MEASURED)
            self.assertEqual(r["overall"]["escalation"]["accuracy"], NOT_MEASURED)
            self.assertEqual(r["overall"]["contradiction_detection"]["accuracy"], NOT_MEASURED)
            self.assertEqual(r["overall"]["evidence_grounding"]["grounded_citation_rate"], NOT_MEASURED)

    def test_optional_fields_absent_gives_not_measured_not_zero(self):
        r = evaluate_adversarial_predictions(self.recs, [{"perturbation_id": "R1", "predicted_decision": COMP}])
        o = r["overall"]
        self.assertEqual(o["decision"]["accuracy"], 1.0)
        self.assertEqual(o["escalation"]["accuracy"], NOT_MEASURED)
        self.assertEqual(o["contradiction_detection"]["accuracy"], NOT_MEASURED)
        self.assertEqual(o["evidence_grounding"]["grounded_citation_rate"], NOT_MEASURED)
        self.assertEqual(o["evidence_grounding"]["supporting"]["f1"], NOT_MEASURED)
        self.assertEqual(o["evidence_grounding"]["missing_flag_accuracy"], NOT_MEASURED)
        self.assertEqual(r["per_perturbation"][0]["escalation_correct"], NOT_MEASURED)
        self.assertAlmostEqual(r["coverage"], 0.25)
        self.assertFalse(r["complete"])

    def test_empty_expected_and_predicted_evidence_has_no_denominator(self):
        r = evaluate_adversarial_predictions(self.recs, [{"perturbation_id": "R1", "predicted_decision": COMP, "predicted_contradicting_evidence": []}])
        self.assertEqual(r["overall"]["evidence_grounding"]["contradicting"]["f1"], NOT_MEASURED)

    def test_invalid_predictions_rejected_whole(self):
        bad = {"perturbation_id": "R1", "predicted_decision": COMP}
        cases = [({"R1": bad}, "list or tuple"), ([{**bad, "expected_decision": INS}], "ground-truth"), ([{**bad, "outcome_change": OUTCOME_UNCHANGED}], "ground-truth"),
                 ([{**bad, "perturbation_id": "NOPE"}], "unknown perturbation_id"), ([bad, bad], "duplicate"), ([{**bad, "predicted_decision": "MAYBE"}], "predicted_decision"),
                 ([{**bad, "predicted_escalation": "yes"}], "bool"), ([{**bad, "predicted_supporting_evidence": ["d1", "d1"]}], "malformed"),
                 ([{**bad, "split": "TEST"}], "mismatch"), (["R1"], "not an object"), ([{"predicted_decision": COMP}], "perturbation_id")]
        for preds, frag in cases:
            r = evaluate_adversarial_predictions(self.recs, preds)
            self.assertEqual(r["status"], STATUS_INVALID_PREDICTIONS, preds)
            self.assertTrue(any(frag in e for e in r["errors"]), (frag, r["errors"]))
            self.assertNotIn("overall", r)

    def test_invalid_records_rejected(self):
        for mutate in (lambda d: d.update(predicted_decision=COMP), lambda d: d.pop("provenance"), lambda d: d.update(expected_decision="MAYBE"),
                       lambda d: d["provenance"].pop("source_expected_escalation"), lambda d: d.update(perturbation_type="nope")):
            recs = copy.deepcopy(self.recs)
            mutate(recs[0])
            self.assertEqual(evaluate_adversarial_predictions(recs, self.preds)["status"], STATUS_INVALID_RECORDS)
        dup = [self.recs[0], self.recs[0]]
        self.assertEqual(evaluate_adversarial_predictions(dup, None)["status"], STATUS_INVALID_RECORDS)
        self.assertEqual(evaluate_adversarial_predictions("nope", None)["status"], STATUS_INVALID_RECORDS)

    def test_scope_filters(self):
        r = evaluate_adversarial_predictions(self.recs, self.preds[:2], splits="TRAIN")
        self.assertEqual((r["records_in_scope"], r["case_count"], r["complete"]), (2, 2, True))
        r = evaluate_adversarial_predictions(self.recs, self.preds[:2], perturbation_types=["ocr_errors"])
        self.assertEqual(r["status"], STATUS_INVALID_PREDICTIONS)  # R2 is outside the evaluated scope
        with self.assertRaises(ValueError): evaluate_adversarial_predictions(self.recs, None, splits="NOPE")
        with self.assertRaises(ValueError): evaluate_adversarial_predictions(self.recs, None, perturbation_types=["bogus"])

    def test_provenance_preserved_and_isolated(self):
        by = {x["perturbation_id"]: x for x in self.res["per_perturbation"]}
        for rec in self.recs:
            x = by[rec["perturbation_id"]]
            self.assertEqual(x["provenance"]["perturbation_provenance"], rec["provenance"])
            self.assertEqual((x["source_case_id"], x["source_family_id"], x["split"], x["outcome_change"]), (rec["source_case_id"], rec["source_family_id"], rec["provenance"]["source_split"], rec["outcome_change"]))
        self.assertEqual(by["R1"]["provenance"]["prediction_provenance"], {"system": "fixture-A"})
        self.assertIsNone(by["R2"]["provenance"]["prediction_provenance"])
        by["R1"]["provenance"]["perturbation_provenance"]["marker"] = "tampered"
        self.assertEqual(self.recs[0]["provenance"]["marker"], "prov-R1")

    def test_ground_truth_independent_of_predictions_and_inputs_unchanged(self):
        before_r, before_p = copy.deepcopy(self.recs), copy.deepcopy(self.preds)
        flipped = [{**p, "predicted_decision": INS if p["predicted_decision"] != INS else COMP} for p in self.preds]
        a, b = evaluate_adversarial_predictions(self.recs, self.preds), evaluate_adversarial_predictions(self.recs, flipped)
        for x, y in zip(a["per_perturbation"], b["per_perturbation"]):
            for k in ("expected_decision", "expected_escalation", "expected_evidence_roles", "expected_contradiction_behavior", "outcome_change", "source_expected_escalation"):
                self.assertEqual(x[k], y[k])
        self.assertNotEqual(a["overall"]["decision"]["accuracy"], b["overall"]["decision"]["accuracy"])
        self.assertEqual((self.recs, self.preds), (before_r, before_p))

    def test_deterministic(self):
        again = evaluate_adversarial_predictions(copy.deepcopy(self.recs), copy.deepcopy(self.preds))
        self.assertEqual(_canon(self.res), _canon(again))
        self.assertEqual(self.res["run_provenance"]["randomness"], "none")

    def test_works_on_generated_records_with_supplied_predictions(self):
        from_gen = generate_adversarial_suite()
        subset = [r for r in from_gen if r.perturbation_type in ("missing_evidence", "prompt_injection_inside_documents")][:6]
        preds = [{"perturbation_id": r.perturbation_id, "predicted_decision": COMP} for r in subset]  # test input: a deliberately naive supplied system
        res = evaluate_adversarial_predictions(from_gen, preds)
        self.assertEqual((res["status"], res["case_count"]), (STATUS_MEASURED, len(subset)))
        n_ins = sum(1 for r in subset if r.expected_decision == INS)
        self.assertEqual(res["overall"]["unsupported_decision"]["unsupported_count"], n_ins)
        self.assertEqual(res["per_perturbation"][0]["provenance"]["perturbation_provenance"], subset[0].provenance)
        self.assertEqual(evaluate_adversarial_predictions(from_gen, preds, source_cases=None)["status"], STATUS_MEASURED)

    def test_module_is_offline_and_random_free(self):
        src = open(__file__, encoding="utf-8").read()
        self.assertIsNone(re.search(r"^\s*(?:import|from)\s+(?:random|socket|urllib|http|requests|secrets|time|datetime)\b", src, re.M))


def _complete_preds():
    """Test fixture for a second system: deliberately different from the V1 fixture."""
    return [{"perturbation_id": "R1", "predicted_decision": COMP, "predicted_escalation": False, "predicted_supporting_evidence": ["d1"], "predicted_contradicting_evidence": [],
             "predicted_contradiction_detected": False},
            {"perturbation_id": "R2", "predicted_decision": NC, "predicted_escalation": True, "predicted_supporting_evidence": ["d1", "d2"], "predicted_contradicting_evidence": [],
             "predicted_contradiction_detected": False},
            {"perturbation_id": "R3", "predicted_decision": INS, "predicted_escalation": True, "predicted_supporting_evidence": [], "predicted_contradicting_evidence": [],
             "predicted_contradiction_detected": False},
            {"perturbation_id": "R4", "predicted_decision": INS, "predicted_escalation": True, "predicted_supporting_evidence": ["d1"], "predicted_contradicting_evidence": ["d2"],
             "predicted_contradiction_detected": True}]


class CompareSystemsTests(unittest.TestCase):
    def setUp(self):
        self.recs, self.v1 = _fixture()
        self.cx = _complete_preds()
        self.cmp = compare_systems(self.recs, self.v1, self.cx)

    def test_hand_computed_absolute_and_relative_change(self):
        m = self.cmp["overall"]["metrics"]
        self.assertEqual(self.cmp["status"], STATUS_MEASURED)
        d = m["decision_accuracy"]
        self.assertEqual((d["v1"], d["complete"], d["absolute_change"], d["relative_change"]), (0.5, 1.0, 0.5, 1.0))
        u = m["unsupported_decision_rate"]
        self.assertEqual((u["v1"], u["complete"], u["absolute_change"], u["relative_change"], u["higher_is_better"]), (0.5, 0.0, -0.5, -1.0, False))
        nc = m["decision_f1[NON_COMPLIANT]"]  # V1 F1 is 0.0: absolute change valid, relative change has no valid denominator
        self.assertEqual((nc["v1"], nc["complete"], nc["absolute_change"], nc["relative_change"]), (0.0, 1.0, 1.0, NOT_MEASURED))
        e = m["escalation_accuracy"]
        self.assertEqual((e["v1"], e["complete"], e["absolute_change"]), (0.5, 1.0, 0.5))
        self.assertEqual((m["escalation_rate"]["v1"], m["escalation_rate"]["complete"], m["expected_escalation_rate"]["v1"]), (0.25, 0.75, 0.75))
        self.assertAlmostEqual(m["decision_macro_f1"]["complete"], 1.0)

    def test_matches_single_system_evaluator_exactly(self):
        solo = evaluate_adversarial_predictions(self.recs, self.v1)
        self.assertEqual(self.cmp["systems"][SYSTEM_V1]["overall"], solo["overall"])  # comparison adds no scoring logic of its own
        self.assertEqual(self.cmp["overall"]["metrics"]["decision_accuracy"]["v1"], solo["overall"]["decision"]["accuracy"])

    def test_same_records_pairing_and_unpaired_never_scored(self):
        r = compare_systems(self.recs, self.v1, self.cx[:3])  # COMPLETE did not predict R4
        self.assertEqual((r["paired"]["paired_count"], r["paired"]["v1_only"], r["paired"]["complete_only"]), (3, ["R4"], []))
        self.assertTrue(r["paired"]["same_records_for_both"])
        self.assertEqual(r["systems"][SYSTEM_V1]["case_count"], r["systems"][SYSTEM_COMPLETE]["case_count"])
        self.assertEqual([x["perturbation_id"] for x in r["systems"][SYSTEM_V1]["per_perturbation"]], [x["perturbation_id"] for x in r["systems"][SYSTEM_COMPLETE]["per_perturbation"]])
        self.assertEqual(r["systems"][SYSTEM_V1]["run_provenance"]["records_sha256"], r["systems"][SYSTEM_COMPLETE]["run_provenance"]["records_sha256"])
        self.assertNotIn("R4", [x["perturbation_id"] for x in r["per_perturbation"]])
        self.assertEqual(r["overall"]["metrics"]["decision_accuracy"]["v1"], 1 / 3)  # V1 on R1-R3 only

    def test_missing_system_gives_not_measured_no_numbers(self):
        for v1, cx in ((None, self.cx), (self.v1, None), (None, None), ([], []), (self.v1[:2], self.cx[2:])):
            r = compare_systems(self.recs, v1, cx)
            self.assertEqual(r["status"], STATUS_NOT_MEASURED)
            self.assertEqual(r["paired"]["paired_count"], 0)
            for g in [r["overall"], *r["by_perturbation_type"].values(), *r["by_split"].values()]:
                for name, d in g["metrics"].items():
                    self.assertEqual((d["v1"], d["complete"], d["absolute_change"], d["relative_change"]), (NOT_MEASURED,) * 4, name)

    def test_invalid_input_in_either_system_rejected(self):
        bad = [{"perturbation_id": "R1", "predicted_decision": COMP, "expected_decision": INS}]
        r = compare_systems(self.recs, self.v1, bad)
        self.assertEqual(r["status"], STATUS_INVALID_PREDICTIONS)
        self.assertEqual(list(r["errors"]), [SYSTEM_COMPLETE])
        self.assertEqual(compare_systems(self.recs, {"R1": 1}, self.cx)["status"], STATUS_INVALID_PREDICTIONS)
        self.assertEqual(compare_systems("nope", self.v1, self.cx)["status"], STATUS_INVALID_RECORDS)
        self.assertNotIn("overall", r)

    def test_by_type_and_by_split_groups(self):
        c = self.cmp
        self.assertEqual(list(c["by_split"]), ["TRAIN", "DEV", "TEST"])
        self.assertEqual(sum(g["case_count"] for g in c["by_perturbation_type"].values()), 4)
        self.assertEqual(sum(g["case_count"] for g in c["by_split"].values()), 4)
        t = c["by_split"]["TRAIN"]["metrics"]["decision_accuracy"]  # R1 right/right, R2 V1 wrong, COMPLETE right
        self.assertEqual((t["v1"], t["complete"], t["absolute_change"]), (0.5, 1.0, 0.5))
        g = c["by_split"]["TEST"]["metrics"]["unsupported_decision_rate"]  # R4 expects INS; neither commits
        self.assertEqual((g["v1"], g["complete"], g["absolute_change"], g["relative_change"]), (0.0, 0.0, 0.0, NOT_MEASURED))
        self.assertEqual(c["by_split"]["TRAIN"]["metrics"]["unsupported_decision_rate"]["v1"], NOT_MEASURED)  # no expected-INSUFFICIENT_EVIDENCE record in TRAIN

    def test_ground_truth_independent_of_both_systems_and_order_symmetric(self):
        before = copy.deepcopy(self.recs)
        swapped = compare_systems(self.recs, self.cx, self.v1)
        for k in ("decision_accuracy", "unsupported_decision_rate", "escalation_accuracy"):
            a, b = self.cmp["overall"]["metrics"][k], swapped["overall"]["metrics"][k]
            self.assertEqual((a["v1"], a["complete"], a["absolute_change"]), (b["complete"], b["v1"], -b["absolute_change"]))
        for x, y in zip(self.cmp["systems"][SYSTEM_V1]["per_perturbation"], self.cmp["systems"][SYSTEM_COMPLETE]["per_perturbation"]):
            for k in ("expected_decision", "expected_escalation", "expected_evidence_roles", "expected_contradiction_behavior", "outcome_change"): self.assertEqual(x[k], y[k])
        self.assertEqual(self.recs, before)

    def test_provenance_preserved_per_system_and_side_by_side(self):
        by = {x["perturbation_id"]: x for x in self.cmp["per_perturbation"]}
        for rec in self.recs: self.assertEqual(by[rec["perturbation_id"]]["perturbation_provenance"], rec["provenance"])
        for name in (SYSTEM_V1, SYSTEM_COMPLETE):
            for x in self.cmp["systems"][name]["per_perturbation"]:
                self.assertEqual(x["provenance"]["perturbation_provenance"]["marker"], f"prov-{x['perturbation_id']}")
        by["R1"]["perturbation_provenance"]["marker"] = "tampered"
        self.assertEqual(self.recs[0]["provenance"]["marker"], "prov-R1")

    def test_deterministic_and_no_superiority_claim(self):
        again = compare_systems(copy.deepcopy(self.recs), copy.deepcopy(self.v1), copy.deepcopy(self.cx))
        self.assertEqual(_canon(self.cmp), _canon(again))
        self.assertIn("no superiority claim", self.cmp["comparison"]["claim"])
        self.assertNotIn("winner", _canon(self.cmp))

    def test_works_on_generated_records(self):
        gen = generate_adversarial_suite()
        sub = [r for r in gen if r.perturbation_type in ("missing_evidence", "ocr_errors")][:8]
        v1 = [{"perturbation_id": r.perturbation_id, "predicted_decision": COMP} for r in sub]  # test inputs standing in for supplied predictions
        cx = [{"perturbation_id": r.perturbation_id, "predicted_decision": INS} for r in sub]
        res = compare_systems(gen, v1, cx)
        self.assertEqual((res["status"], res["paired"]["paired_count"]), (STATUS_MEASURED, len(sub)))
        self.assertEqual(res["per_perturbation"][0]["perturbation_provenance"], sub[0].provenance)

    def test_probe_reports_blockers_without_importing_or_numbers(self):
        import sys, tempfile
        code = "import definitely_missing_pkg_xyz\nfrom .sibling import x\nimport requests\n\ndef analyze_compliance_with_matrix(a, b, c, is_v2=False):\n    return requests.post('u')\n\ndef build_evidence_graph(f, r=None):\n    return {'v': \"VIOLATION\"}\n"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fake_tasks.py")
            with open(path, "w", encoding="utf-8") as f: f.write(code)
            r = probe_system_interfaces(path)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("definitely_missing_pkg_xyz", r["missing_modules"])
        self.assertEqual(r["relative_imports"], [".sibling"])
        self.assertTrue(r["v1"]["found"] and r["v1"]["calls_llm_or_network"] and r["complete"]["found"])
        self.assertFalse(r["v1"]["outputs_omni_bench_labels"] or r["complete"]["outputs_omni_bench_labels"])
        self.assertEqual(r["omni_bench_adapters_found"], [])
        self.assertEqual(r["performance_numbers"], NOT_MEASURED)
        self.assertNotIn("definitely_missing_pkg_xyz", sys.modules)  # probe never imports the inspected code
        self.assertEqual(probe_system_interfaces(os.path.join("no", "such", "tasks.py"))["status"], "UNAVAILABLE")
        self.assertEqual(probe_system_interfaces("no_such.py")["performance_numbers"], NOT_MEASURED)

    def test_comparison_never_invokes_a_system(self):
        with open(__file__, encoding="utf-8") as fh: src = fh.read()
        self.assertIsNone(re.search(r"^\s*(?:import|from)\s+(?:\.?tasks|celery|networkx|pydantic)\b", src, re.M))


# ----------------------------------------------------------------------------- 11C adapter / comparison regression tests (stubs and fixtures are TEST INPUTS, never benchmark results)
class _FakeTasks:
    """Stand-in for the injected project module: records the stage calls the adapter makes and returns canned structured assessments (test plumbing only)."""
    ENABLE_CONTRADICTION_HEURISTIC = True

    def __init__(self, assessments, decisions=1, boom=None, compiled=None, enable_contra=True):
        self.assessments, self.n_dec, self.boom, self.order, self.calls = assessments, decisions, boom, [], []
        self.compiled = {"status": "VALID"} if compiled is None else compiled
        self.ENABLE_CONTRADICTION_HEURISTIC = enable_contra
    def _init_graph(self): self.order.append("_init_graph"); self.calls.append({"docs": {}, "rulebook": None}); return {"n": len(self.calls)}
    def _build_policy_nodes(self, G, rb):
        self.order.append("_build_policy_nodes")
        with open(rb, encoding="utf-8") as fh: self.calls[-1]["rulebook"] = fh.read()
        return "RULEBOOK-TEXT"
    def is_sensitive_file(self, p): return False
    def _add_document_node(self, G, p): return "doc:" + os.path.basename(p)
    def _ingest_facts(self, G, doc, fname, path, cache):
        self.order.append("_ingest_facts")
        if self.boom: raise RuntimeError(self.boom)
        with open(path, encoding="utf-8") as fh: self.calls[-1]["docs"][fname] = fh.read()
    def link_shared_terms(self, G): self.order.append("link_shared_terms")
    def link_cross_documents(self, G): self.order.append("link_cross_documents")
    def apply_compiled_policy(self, G, text):
        self.order.append("apply_compiled_policy"); self.calls[-1]["compiled_text"] = text
        if isinstance(self.compiled, Exception): raise self.compiled
        return self.compiled
    def evaluate_policy_rules(self, G): self.order.append("evaluate_policy_rules")
    def detect_contradictions(self, G): self.order.append("detect_contradictions")
    def classify_contradictions(self, G): self.order.append("classify_contradictions")
    def _nodes_of_type(self, G, t): return [(f"dec_{i}", {}) for i in range(self.n_dec)]
    def build_uncertainty_assessment(self, G, d): return self.assessments[(G["n"] - 1) % len(self.assessments)]


def _assess(state, esc=False, sup=(), con_docs=(), finding=None, verdict="VIOLATION", found=True):
    con = [{"source": "CONTRADICTS_edge", "documents": [{"document_id": f"doc_{x}", "filename": x}]} for x in con_docs]
    if finding: con.append({"source": "contradiction_finding", "claim_a": {"document_id": "doc_a", "filename": finding[0]}, "claim_b": {"document_id": "doc_b", "filename": finding[1]}})
    return {"found": found, "decision": {"verdict": verdict}, "uncertainty_status": state, "escalation_required": esc,
            "escalation_reasons": [{"reason": "UNRESOLVED_CONTRADICTION"}, {"reason": "HIGH_RISK"}] if esc else [],
            "supporting_evidence": [{"filename": f, "document_id": f"doc_{f}"} for f in sup], "contradicting_evidence": con}


class ComparisonAdapterTests(unittest.TestCase):
    def setUp(self):
        self.recs, _ = _fixture()
        self.rev = [{"perturbation_id": "R1", "reviewed_verdict": "SATISFIED", "reviewer": "t"}, {"perturbation_id": "R2", "reviewed_verdict": "SATISFIED"},
                    {"perturbation_id": "R3", "reviewed_verdict": "violation"}]

    # --- COMPLETE adapter
    def test_complete_maps_structured_fields_and_feeds_the_record(self):
        ft = _FakeTasks([_assess(NC, True, ["d1", "d1", "d2"], con_docs=["d2"], finding=("d1", "d2"))])
        out = complete_predictions(self.recs[:1], ft)
        p = out["predictions"][0]
        self.assertEqual((p["predicted_decision"], p["predicted_escalation"], p["predicted_supporting_evidence"], p["predicted_contradicting_evidence"], p["predicted_contradiction_detected"]),
                         (NC, True, ["d1", "d2"], ["d2", "d1"], True))
        self.assertNotIn("predicted_missing_evidence", p)  # kinds are not OMNI descriptions: never invented
        self.assertEqual((ft.calls[0]["docs"], ft.calls[0]["rulebook"], ft.calls[0]["compiled_text"]), ({"d1": "text d1"}, "T-1.1: test policy.", "RULEBOOK-TEXT"))
        self.assertEqual((p["perturbation_id"], p["source_case_id"], p["source_family_id"], p["perturbation_type"], p["split"]), ("R1", "SRC-R1", "FAM-T-1", "ocr_errors", "TRAIN"))
        self.assertEqual(p["provenance"]["escalation_reason_codes"], ["HIGH_RISK", "UNRESOLVED_CONTRADICTION"])
        self.assertEqual(evaluate_adversarial_predictions(self.recs, out["predictions"])["status"], STATUS_MEASURED)  # passes the strict prediction validator

    def test_complete_skips_instead_of_guessing(self):
        cond = complete_predictions(self.recs, _FakeTasks([_assess("CONDITIONAL", verdict="UNEVALUATED")]))
        self.assertEqual((cond["predictions"], len(cond["skipped"])), ([], 4))
        self.assertIn("no OMNI-Bench decision label", cond["skipped"][0]["reason"])
        forced = complete_predictions(self.recs[:1], _FakeTasks([_assess("CONDITIONAL")]), conditional_label=INS)  # only when the caller says so
        self.assertEqual(forced["predictions"][0]["predicted_decision"], INS)
        self.assertEqual(forced["predictions"][0]["provenance"]["uncertainty_status"], "CONDITIONAL")
        with self.assertRaises(ValueError): complete_predictions(self.recs, _FakeTasks([]), conditional_label="MAYBE")
        multi = complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP)], decisions=2))
        self.assertIn("2 Decision nodes", multi["skipped"][0]["reason"])
        self.assertIn("RuntimeError: kaput", complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP)], boom="kaput"))["skipped"][0]["reason"])
        self.assertEqual(complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP, found=False)]))["predictions"], [])

    def test_complete_is_deterministic_and_never_reads_labels(self):
        ft = _FakeTasks([_assess(COMP), _assess(NC), _assess(INS), _assess(COMP)])
        a, b = complete_predictions(self.recs, ft), complete_predictions(copy.deepcopy(self.recs), _FakeTasks([_assess(COMP), _assess(NC), _assess(INS), _assess(COMP)]))
        self.assertEqual(_canon(a), _canon(b))
        for c in ft.calls: self.assertNotIn("expected", json.dumps(c).lower())  # the pipeline only ever sees documents and policy
        with self.assertRaises(ValueError): complete_predictions([{"perturbation_id": "x"}], ft)

    # --- frozen-path equivalence (11C correction)
    def test_complete_adapter_follows_production_stage_order_including_compile(self):
        ft = _FakeTasks([_assess(COMP)])
        complete_predictions(self.recs[:2], ft)
        first = list(dict.fromkeys(ft.order))
        self.assertEqual(first, list(_PRODUCTION_STAGES))
        self.assertEqual(ft.order.count("apply_compiled_policy"), 2)  # once per record, before evaluate_policy_rules
        self.assertLess(first.index("apply_compiled_policy"), first.index("evaluate_policy_rules"))
        off = _FakeTasks([_assess(COMP)], enable_contra=False)
        complete_predictions(self.recs[:1], off)
        self.assertNotIn("detect_contradictions", off.order)  # same gate as the production task

    @unittest.skipUnless(os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.py")), "tasks.py not beside this module")
    def test_stage_order_matches_production_task_and_documents_build_evidence_graph_gap(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.py"), encoding="utf-8", errors="replace") as fh: tree = ast.parse(fh.read())
        fn = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        calls = lambda f: sorted(((c.lineno, c.col_offset, c.func.id) for c in ast.walk(fn[f]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)))
        prod = list(dict.fromkeys(n for _, _, n in calls("process_document_batch_task") if n in _PRODUCTION_STAGES))
        self.assertEqual(prod, list(_PRODUCTION_STAGES))  # adapter order == production task order
        bev = {n for _, _, n in calls("build_evidence_graph")}
        self.assertNotIn("apply_compiled_policy", bev)  # the benchmark path (_cfb_build -> build_evidence_graph) is NOT the production path
        self.assertTrue(set(_PRODUCTION_STAGES) - {"apply_compiled_policy"} <= bev | {"_ingest_facts"})
        self.assertEqual(sum(1 for f in fn.values() for _, _, n in calls(f.name) if n == "apply_compiled_policy" and f.name != "apply_compiled_policy"), 1)  # only the task calls it

    def test_compile_stage_not_run_is_skipped_by_default_never_scored_as_production(self):
        for comp in ({"status": "NOT_COMPILED", "error": "policy could not be compiled: x"}, RuntimeError("llm down")):
            out = complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP)], compiled=comp))
            self.assertEqual(out["predictions"], [])
            self.assertIn("compiled-policy stage did not run cleanly", out["skipped"][0]["reason"])
        class NoCompile(_FakeTasks):
            def apply_compiled_policy(self, G, text): self.order.append("apply_compiled_policy"); return None  # compiler disabled / unavailable / no rules
        out = complete_predictions(self.recs[:1], NoCompile([_assess(COMP)]))
        self.assertEqual((out["predictions"], len(out["skipped"])), ([], 1))
        legacy = complete_predictions(self.recs[:1], NoCompile([_assess(COMP)]), require_compiled_policy=False)
        self.assertEqual(legacy["predictions"][0]["provenance"]["compiled_policy_stage"]["ran"], False)  # explicit opt-in, and recorded
        ok = complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP)]))["predictions"][0]["provenance"]
        self.assertEqual((ok["compiled_policy_stage"]["ran"], ok["compiled_policy_stage"]["status"], ok["stage_errors"]), (True, "VALID", []))

    def test_compiled_rule_evaluation_error_is_skipped_not_scored_as_production(self):  # tasks.apply_compiled_policy reports an engine crash in evaluation_error (no "error" key)
        out = complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP)], compiled={"status": "VALID", "evaluation_error": "engine boom"}))
        self.assertEqual(out["predictions"], [])
        self.assertIn("compiled rule evaluation failed: engine boom", out["skipped"][0]["reason"])
        legacy = complete_predictions(self.recs[:1], _FakeTasks([_assess(COMP)], compiled={"status": "VALID", "evaluation_error": "engine boom"}), require_compiled_policy=False)
        self.assertFalse(legacy["predictions"][0]["provenance"]["compiled_policy_stage"]["ran"])

    def test_conditional_stays_unmapped_by_default_on_the_production_path(self):
        out = complete_predictions(self.recs, _FakeTasks([_assess("CONDITIONAL", verdict="UNEVALUATED")]))
        self.assertEqual((out["predictions"], len(out["skipped"])), ([], 4))
        self.assertNotIn("CONDITIONAL", UNC_STATE_TO_LABEL)

    def test_failing_non_fatal_stage_is_recorded_and_pipeline_continues_like_production(self):
        class Flaky(_FakeTasks):
            def link_cross_documents(self, G): self.order.append("link_cross_documents"); raise RuntimeError("xdoc down")
        out = complete_predictions(self.recs[:1], Flaky([_assess(COMP)]))
        self.assertEqual(out["predictions"][0]["provenance"]["stage_errors"], ["link_cross_documents: RuntimeError: xdoc down"])

    # --- V1: inputs, reviews, proxy
    def test_v1_cases_use_real_input_shape_and_independent_ground_truth(self):
        recs = copy.deepcopy(self.recs)
        recs[0]["perturbed_policy"] = "EXP-4.2: Any expense above $2,000 needs approval."
        import tempfile as tf
        with tf.TemporaryDirectory() as wd:
            cases = v1_experiment_cases(recs, wd, self.rev)
            c = cases[0]
            self.assertEqual(sorted(c), ["case_id", "files", "ground_truth", "objective", "rulebook"])
            self.assertEqual((c["case_id"], c["objective"], [os.path.basename(f) for f in c["files"]]), ("R1", V1_OBJECTIVE, ["d1"]))
            with open(c["files"][0], encoding="utf-8") as f1, open(c["rulebook"], encoding="utf-8") as f2: self.assertEqual((f1.read(), f2.read()), ("text d1", recs[0]["perturbed_policy"]))
            self.assertEqual(c["ground_truth"]["expected_verdicts"], [{"rule_contains": "EXP-4.2", "verdict": "SATISFIED"}])  # from the record, not a system
            self.assertEqual(c["ground_truth"]["reviewed_verdicts"], {"V1": [{"rule_contains": "EXP-4.2", "verdict": "SATISFIED"}]})
            self.assertNotIn("expected_verdicts", cases[1]["ground_truth"])  # "T-1.1" is not a single recognizable rule id: nothing is guessed
            self.assertNotIn("reviewed_verdicts", v1_experiment_cases(recs, wd)[0]["ground_truth"])
        for lab in LABELS: self.assertNotIn(lab, V1_OBJECTIVE)

    def test_v1_reviews_map_only_via_existing_vocabulary_and_are_validated(self):
        ok = v1_predictions_from_reviews(self.recs, self.rev + [{"perturbation_id": "R4", "reviewed_verdict": "UNEVALUATED"}])
        self.assertEqual([(p["perturbation_id"], p["predicted_decision"]) for p in ok["predictions"]], [("R1", COMP), ("R2", COMP), ("R3", NC)])
        self.assertEqual([s["perturbation_id"] for s in ok["skipped"]], ["R4"])
        for p in ok["predictions"]: self.assertEqual(sorted(k for k in p if k.startswith("predicted_")), ["predicted_decision"])
        self.assertEqual(ENGINE_VERDICT_TO_LABEL["NOT_APPLICABLE"], INS)
        for bad in ([{"perturbation_id": "Z", "reviewed_verdict": "SATISFIED"}], [{"perturbation_id": "R1", "reviewed_verdict": "MAYBE"}],
                    [{"perturbation_id": "R1", "reviewed_verdict": "SATISFIED"}] * 2, [{"perturbation_id": "R1", "reviewed_verdict": "SATISFIED", "expected_decision": NC}]):
            r = v1_predictions_from_reviews(self.recs, bad)
            self.assertEqual((r["status"], r["predictions"]), ("INVALID_REVIEWS", []))
        self.assertEqual(v1_predictions_from_reviews(self.recs, [])["status"], STATUS_NOT_MEASURED)

    def test_v1_report_citation_proxy_pools_existing_records_only(self):
        mk = lambda cid, v, valid, n, ver="V1": {"case_id": cid, "version": ver, "measured": {"citation_validity": ({"status": "MEASURED", "valid": valid, "cited_syntactic": n, "value": valid / n} if n else {"status": NOT_MEASURED})}}
        r = v1_report_citation_metrics([mk("R1", "", 2, 2), mk("R2", "", 1, 2), mk("R3", "", 0, 0), mk("R1", "", 9, 9, "V2"), mk("nope", "", 1, 1)], self.recs)
        self.assertEqual((r["overall"]["reports"], r["overall"]["reports_with_citations"], r["overall"]["valid"], r["overall"]["cited_syntactic"], r["overall"]["citation_validity"]), (3, 2, 3, 4, 0.75))
        self.assertEqual(r["by_split"]["TRAIN"]["citation_validity"], 0.75)
        self.assertEqual(r["by_split"]["DEV"]["citation_validity"], NOT_MEASURED)  # report without any citation: no valid denominator
        self.assertEqual(v1_report_citation_metrics([], self.recs)["status"], STATUS_NOT_MEASURED)
        self.assertEqual(v1_report_citation_metrics([mk("R1", "", 1, 1, "V2")], self.recs)["status"], STATUS_NOT_MEASURED)

    # --- comparison
    def _cx(self):
        return {"predictions": [dict(p, **{"provenance": {"system": "COMPLETE"}}) for p in _complete_preds()], "skipped": []}

    def test_v1_vs_complete_measures_only_what_is_legitimate(self):
        res = compare_v1_vs_complete(self.recs, self.rev, self._cx())
        self.assertEqual((res["status"], res["paired"]["paired_count"], res["paired"]["complete_only"]), (STATUS_MEASURED, 3, ["R4"]))
        m = res["overall"]["metrics"]
        d = m["decision_accuracy"]  # V1 reviews: R1 right, R2 wrong, R3 wrong; COMPLETE right on all three
        self.assertEqual((d["v1"], d["complete"]), (1 / 3, 1.0)); self.assertAlmostEqual(d["absolute_change"], 2 / 3); self.assertAlmostEqual(d["relative_change"], 2.0)
        for k in ("escalation_accuracy", "escalation_rate", "contradiction_f1", "contradiction_accuracy", "supporting_evidence_f1", "grounded_citation_rate", "pooled_evidence_f1", "missing_flag_accuracy"):
            self.assertEqual((m[k]["v1"], m[k]["absolute_change"], m[k]["relative_change"]), (NOT_MEASURED,) * 3, k)
        self.assertNotEqual(m["escalation_accuracy"]["complete"], NOT_MEASURED)
        av = res["metric_availability"]
        self.assertEqual(av[SYSTEM_V1]["decision_accuracy"]["status"], "MEASURED_WHERE_SUPPLIED")
        for k, word in (("escalation_accuracy", "escalation"), ("contradiction_f1", "contradiction"), ("supporting_evidence_f1", "role-separated")):
            self.assertEqual(av[SYSTEM_V1][k]["status"], NOT_MEASURED)
            self.assertIn(word, av[SYSTEM_V1][k]["reason"], k)
        self.assertEqual(av[SYSTEM_V1]["unsupported_decision_rate"]["status"], "MEASURED_WHERE_SUPPLIED")  # derivable from the reviewed decisions alone
        self.assertIn("KINDS", av[SYSTEM_COMPLETE]["missing_flag_accuracy"]["reason"])
        self.assertEqual(av[SYSTEM_COMPLETE]["escalation_accuracy"]["status"], "MEASURED_WHERE_SUPPLIED")
        self.assertEqual(res["complete_system_all_predicted"]["case_count"], 4)  # COMPLETE is also reported over everything it predicted
        self.assertEqual(res["systems"][SYSTEM_V1]["run_provenance"]["records_sha256"], res["systems"][SYSTEM_COMPLETE]["run_provenance"]["records_sha256"])
        self.assertEqual(list(res["by_split"]), ["TRAIN", "DEV"])  # only paired perturbations are grouped; R4 (TEST) has no V1 review
        self.assertEqual(res["v1_report_citation_proxy"]["status"], STATUS_NOT_MEASURED)

    def test_no_reviews_means_no_v1_numbers_and_no_fabrication(self):
        res = compare_v1_vs_complete(self.recs, [], self._cx())
        self.assertEqual((res["status"], res["paired"]["paired_count"]), (STATUS_NOT_MEASURED, 0))
        for g in [res["overall"], *res["by_perturbation_type"].values(), *res["by_split"].values()]:
            for name, d in g["metrics"].items(): self.assertEqual((d["v1"], d["complete"], d["absolute_change"], d["relative_change"]), (NOT_MEASURED,) * 4, name)
        self.assertEqual(res["adapter_report"][SYSTEM_V1]["predictions"], 0)
        self.assertEqual(compare_v1_vs_complete(self.recs, [], None)["complete_system_all_predicted"]["status"], STATUS_NOT_MEASURED)
        bad = compare_v1_vs_complete(self.recs, [{"perturbation_id": "R1", "reviewed_verdict": "X"}], self._cx())
        self.assertEqual(bad["status"], "INVALID_REVIEWS"); self.assertNotIn("overall", bad)

    def test_ground_truth_untouched_deterministic_and_filters_respected(self):
        before = copy.deepcopy(self.recs)
        a = compare_v1_vs_complete(self.recs, self.rev, self._cx())
        b = compare_v1_vs_complete(copy.deepcopy(self.recs), copy.deepcopy(self.rev), copy.deepcopy(self._cx()))
        self.assertEqual(_canon(a), _canon(b)); self.assertEqual(self.recs, before)
        s = compare_v1_vs_complete(self.recs, self.rev, self._cx(), splits=["TRAIN"])
        self.assertEqual((s["complete_system_all_predicted"]["records_in_scope"], s["paired"]["paired_count"], s["adapter_report"][SYSTEM_COMPLETE]["out_of_scope_dropped"]), (2, 2, 2))
        self.assertEqual(s["complete_system_all_predicted"]["case_count"], 2)
        for x in a["per_perturbation"]: self.assertEqual(x["perturbation_provenance"]["marker"], f"prov-{x['perturbation_id']}")
        self.assertIn("no superiority claim", a["comparison"]["claim"])

    def test_adapters_work_on_generated_records_and_skip_accounting_is_complete(self):
        gen = generate_adversarial_suite()
        out = complete_predictions(gen, _FakeTasks([_assess(INS, True), _assess("CONDITIONAL")]))  # plumbing only: stub outputs, no performance meaning
        self.assertEqual(len(out["predictions"]) + len(out["skipped"]), len(gen))
        self.assertEqual({p["perturbation_id"] for p in out["predictions"]} | {s["perturbation_id"] for s in out["skipped"]}, {r.perturbation_id for r in gen})
        p = out["predictions"][0]; src = next(r for r in gen if r.perturbation_id == p["perturbation_id"])
        self.assertEqual((p["source_case_id"], p["source_family_id"], p["split"]), (src.source_case_id, src.source_family_id, src.provenance["source_split"]))


if __name__ == "__main__":
    unittest.main()