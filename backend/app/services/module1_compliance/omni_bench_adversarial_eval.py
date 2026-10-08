"""OMNI-Bench adversarial evaluation runner: scores SUPPLIED system predictions against the independent expected labels of generated perturbation records.

Deterministic and offline (no LLM, network, wall-clock or randomness). It never produces, completes or guesses predictions: no predictions -> NOT_MEASURED.
Ground truth (expected decision, escalation, evidence roles, contradiction behaviour, outcome change, provenance) is read ONLY from the perturbation record;
a prediction can never supply or alter it, and any ground-truth field inside a prediction makes the whole prediction set INVALID_PREDICTIONS.
Every metric returns NOT_MEASURED when it has no valid denominator. Perturbation provenance is preserved verbatim in every per-perturbation result.
Document text (including injected instructions) is inert data: this module never reads it to derive labels or scores.
Scope: scoring (evaluate_adversarial_predictions) plus a V1-vs-COMPLETE comparison over SUPPLIED predictions (compare_systems) and a static interface probe
(probe_system_interfaces). Neither system is ever invoked here; if they cannot be invoked, no performance number is produced.

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


if __name__ == "__main__":
    unittest.main()