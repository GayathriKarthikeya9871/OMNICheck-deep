"""OMNI-Bench evaluation runner: scores a SUPPLIED prediction record against the existing benchmark ground truth.

Deterministic and offline (no LLM, network or randomness). It never produces predictions: no predictions -> NOT_MEASURED, never a score.

Ground truth (decision, escalation, rule, evidence, counterfactual, family, split, category) is read only from the benchmark case found by case_id;

a prediction can never supply or alter it. Free-text comparisons are strict: normalized (whitespace/case-folded) exact match; rules compare by rule-id set.

Counterfactuals are the one exception: they are judged by the semantic anchors (rule ids, decision labels, doc_ids, numbers, no-change marker) present in the benchmark's own

correction text, never by wording; if those anchors cannot prove or refute equivalence the result is NOT_MEASURED. Predictions must be a list/tuple of records.
"""

import copy
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


NOT_MEASURED = "NOT_MEASURED"

STATUS_MEASURED, STATUS_NOT_MEASURED = "MEASURED", "NOT_MEASURED"

STATUS_INVALID_PREDICTIONS, STATUS_INVALID_BENCHMARK = (
    "INVALID_PREDICTIONS",
    "INVALID_BENCHMARK",
)

_ALLOWED_KEYS = frozenset(
    (
        "case_id",
        "predicted_decision",
        "predicted_escalation",
        "predicted_applicable_policy_rule",
        "predicted_supporting_evidence",
        "predicted_contradicting_evidence",
        "predicted_missing_evidence",
        "predicted_counterfactual_correction",
        "provenance",
        "split",
        "family_id",
        "difficulty_category",
    )
)

_RULE_ID = re.compile(
    r"[A-Z]{2,5}-\d+\.\d+(?:\([a-z]\))?"
)

_DECISION_TOKEN = re.compile(
    r"(?<![A-Za-z_])(?:NON_COMPLIANT|COMPLIANT|INSUFFICIENT_EVIDENCE)(?![A-Za-z_])"
)

_NO_CHANGE = re.compile(
    r"\b(?:does not change|do not change|did not change|not change|no change|unchanged)\b"
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_EVIDENCE = ("supporting", "contradicting", "missing")

_SCALARS = (
    "decision_accuracy",
    "escalation_accuracy",
    "policy_rule_accuracy",
    "counterfactual_correctness",
)

_METRIC_KEYS = _SCALARS + tuple(
    f"{e}_evidence_{m}"
    for e in _EVIDENCE
    for m in ("precision", "recall", "f1")
)


def _bench():
    from . import tasks  # single source of truth for cases, splits, decisions and categories

    return tasks


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def _rule_key(s: str):
    return frozenset(_RULE_ID.findall(s)) or _norm(s)


def _ratio(n, d):
    return n / d if d else NOT_MEASURED


def _has(p: Dict[str, Any], k: str) -> bool:
    return p.get(k) is not None


def _f1(p, r):
    if p == NOT_MEASURED or r == NOT_MEASURED:
        return NOT_MEASURED

    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _prf(tp, pred, exp) -> Dict[str, Any]:
    p, r = _ratio(tp, pred), _ratio(tp, exp)

    return {
        "precision": p,
        "recall": r,
        "f1": _f1(p, r),
    }


def _cf_signature(
    text: str,
    doc_ids,
) -> Tuple[
    Tuple[frozenset, frozenset, frozenset, frozenset],
    bool,
]:
    """Semantic anchors of a counterfactual text: (rule ids, decision labels, case doc_ids, numbers) and whether it states 'no change'. Pure; uses only the text and the case's doc_ids."""

    rules = frozenset(_RULE_ID.findall(text))

    rest = _RULE_ID.sub(" ", text)

    decisions = frozenset(_DECISION_TOKEN.findall(rest))

    rest = _DECISION_TOKEN.sub(" ", rest)

    docs = set()

    for d in sorted(doc_ids):
        pat = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(d) + r"(?![A-Za-z0-9_])"
        )

        if pat.search(rest):
            docs.add(d)

        rest = pat.sub(" ", rest)

    return (
        (
            rules,
            decisions,
            frozenset(docs),
            frozenset(_NUMBER.findall(rest)),
        ),
        bool(_NO_CHANGE.search(_norm(text))),
    )


def _counterfactual_correct(c: Dict[str, Any], cf: Optional[str]):
    """True / False / NOT_MEASURED. Wording-independent: the prediction is correct iff it carries exactly the anchors of the benchmark's correction (rules, decision labels, doc_ids, numbers, no-change marker).

    Any extra or conflicting anchor -> False. Anchors missing, none at all, or none in the benchmark text to compare against -> NOT_MEASURED (never guessed).
    """

    exp = c.get("counterfactual_correction")

    if not (isinstance(exp, str) and exp.strip()) or cf is None:
        return NOT_MEASURED

    if _norm(cf) == _norm(exp):
        return True

    doc_ids = {d["doc_id"] for d in c["document_set"]}

    (e_anchors, e_neg), (p_anchors, p_neg) = (
        _cf_signature(exp, doc_ids),
        _cf_signature(cf, doc_ids),
    )

    if not any(e_anchors[:3]) or not any(p_anchors):
        return NOT_MEASURED

    if e_neg != p_neg or any(
        p - e for e, p in zip(e_anchors, p_anchors)
    ):
        return False

    if any(e - p for e, p in zip(e_anchors, p_anchors)):
        return NOT_MEASURED

    return True
def _resolve_splits(splits) -> Tuple[str, ...]:
    order = tuple(_bench().OMNI_BENCH_SPLITS)

    if splits is None:
        return order

    req = [splits] if isinstance(splits, str) else list(splits)

    if not req or any(s not in order for s in req):
        raise ValueError(
            f"splits must be a non-empty selection of {order}"
        )

    return tuple(s for s in order if s in req)


def _str_list(v) -> bool:
    return (
        isinstance(v, list)
        and all(isinstance(x, str) and x.strip() for x in v)
    )


def _check_records(
    predictions,
    cases,
    splits,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:

    t = _bench()

    if not isinstance(predictions, (list, tuple)):
        return [
            "predictions must be a list or tuple of records"
        ], {}

    by_id = {
        c["case_id"]: c
        for c in cases
    }

    errs: List[str] = []
    accepted: Dict[str, Dict[str, Any]] = {}

    dups = set()

    for i, p in enumerate(predictions):

        if not isinstance(p, dict):
            errs.append(
                f"prediction[{i}]: not an object"
            )
            continue

        cid = p.get("case_id")

        if not isinstance(cid, str) or not cid:
            errs.append(
                f"prediction[{i}]: missing or invalid case_id"
            )
            continue

        tag = f"prediction[{i}] ({cid})"

        if cid not in by_id:
            errs.append(
                f"{tag}: unknown case_id"
            )
            continue

        if cid in accepted:
            dups.add(cid)

        c = by_id[cid]

        extra = sorted(
            str(k)
            for k in p
            if k not in _ALLOWED_KEYS
        )

        if extra:
            errs.append(
                f"{tag}: unknown or ground-truth field(s) "
                f"not allowed in a prediction: {extra}"
            )

        if c["split"] not in splits:
            errs.append(
                f"{tag}: case belongs to split {c['split']}, "
                f"outside the evaluated splits {list(splits)}"
            )

        if (
            not isinstance(p.get("predicted_decision"), str)
            or p["predicted_decision"]
            not in t.OMNI_BENCH_DECISIONS
        ):
            errs.append(
                f"{tag}: missing or invalid predicted_decision"
            )

        if (
            _has(p, "predicted_escalation")
            and not isinstance(
                p["predicted_escalation"],
                bool,
            )
        ):
            errs.append(
                f"{tag}: predicted_escalation must be a bool"
            )

        if _has(
            p,
            "predicted_applicable_policy_rule",
        ) and not (
            isinstance(
                p["predicted_applicable_policy_rule"],
                str,
            )
            and p["predicted_applicable_policy_rule"].strip()
        ):
            errs.append(
                f"{tag}: malformed "
                f"predicted_applicable_policy_rule"
            )

        if _has(
            p,
            "predicted_counterfactual_correction",
        ) and not (
            isinstance(
                p["predicted_counterfactual_correction"],
                str,
            )
            and p[
                "predicted_counterfactual_correction"
            ].strip()
        ):
            errs.append(
                f"{tag}: malformed "
                f"predicted_counterfactual_correction"
            )

        if _has(p, "provenance") and not (
            isinstance(p["provenance"], dict)
            or (
                isinstance(p["provenance"], str)
                and p["provenance"].strip()
            )
        ):
            errs.append(
                f"{tag}: malformed provenance"
            )

        docs = {
            d["doc_id"]
            for d in c["document_set"]
        }

        for e in (
            "supporting",
            "contradicting",
        ):
            k = f"predicted_{e}_evidence"
            v = p.get(k)

            if v is None:
                continue

            if (
                not _str_list(v)
                or len(set(v)) != len(v)
            ):
                errs.append(
                    f"{tag}: malformed {k} "
                    f"(list of unique doc_id strings)"
                )

            elif not set(v) <= docs:
                errs.append(
                    f"{tag}: malformed {k} "
                    f"(doc_id not in this case's "
                    f"document_set: "
                    f"{sorted(set(v) - docs)})"
                )

        v = p.get(
            "predicted_missing_evidence"
        )

        if v is not None and (
            not _str_list(v)
            or len({_norm(x) for x in v}) != len(v)
        ):
            errs.append(
                f"{tag}: malformed "
                f"predicted_missing_evidence "
                f"(list of unique non-empty strings)"
            )

        for (
            k,
            valid,
            truth,
        ) in (
            (
                "split",
                t.OMNI_BENCH_SPLITS,
                c["split"],
            ),
            (
                "difficulty_category",
                t.OMNI_BENCH_CATEGORY_IDS,
                c["difficulty_category"],
            ),
            (
                "family_id",
                None,
                c["family_id"],
            ),
        ):

            if not _has(p, k):
                continue

            if not isinstance(p[k], str) or (
                valid is not None
                and p[k] not in valid
            ):
                errs.append(
                    f"{tag}: invalid {k} value"
                )

            elif p[k] != truth:
                errs.append(
                    f"{tag}: {k} mismatch "
                    f"(benchmark assigns '{truth}')"
                )

        accepted.setdefault(cid, p)

    errs.extend(
        f"duplicate prediction for case_id {cid}"
        for cid in sorted(dups)
    )

    return sorted(errs), accepted


def validate_omni_bench_predictions(
    predictions,
    cases: Optional[List[Dict[str, Any]]] = None,
    splits=None,
) -> List[str]:
    """Sorted list of problems with a prediction set (empty == well-formed)."""

    return _check_records(
        predictions,
        _bench().OMNI_BENCH_CASES
        if cases is None
        else cases,
        _resolve_splits(splits),
    )[0]


def _ev(
    expected: List[str],
    predicted: Optional[List[str]],
    norm: bool,
) -> Dict[str, Any]:

    n = _norm if norm else (lambda x: x)

    exp = {
        n(x)
        for x in expected
    }

    if predicted is None:
        return {
            "precision": NOT_MEASURED,
            "recall": NOT_MEASURED,
            "f1": NOT_MEASURED,
            "true_positives": None,
            "predicted_count": None,
            "expected_count": len(exp),
        }

    pred = {
        n(x)
        for x in predicted
    }

    tp = len(exp & pred)

    return {
        **_prf(
            tp,
            len(pred),
            len(exp),
        ),
        "true_positives": tp,
        "predicted_count": len(pred),
        "expected_count": len(exp),
    }
def _eval_case(
    c: Dict[str, Any],
    p: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:

    g = lambda k: p.get(k) if _has(p, k) else None

    pr = g("predicted_applicable_policy_rule")

    cf_exp = c.get("counterfactual_correction")
    cf = g("predicted_counterfactual_correction")

    cf_ok = _counterfactual_correct(c, cf)

    ps = g("predicted_supporting_evidence")
    pc = g("predicted_contradicting_evidence")
    pm = g("predicted_missing_evidence")

    return {
        "case_id": c["case_id"],
        "family_id": c["family_id"],
        "split": c["split"],
        "difficulty_category": c["difficulty_category"],

        "expected_decision": c["expected_decision"],
        "predicted_decision": p["predicted_decision"],
        "decision_correct": (
            p["predicted_decision"]
            == c["expected_decision"]
        ),

        "expected_escalation": c["expected_escalation"],
        "predicted_escalation": g(
            "predicted_escalation"
        ),
        "escalation_correct": (
            NOT_MEASURED
            if g("predicted_escalation") is None
            else (
                g("predicted_escalation")
                == c["expected_escalation"]
            )
        ),

        "expected_applicable_policy_rule": (
            c["applicable_policy_rule"]
        ),
        "predicted_applicable_policy_rule": pr,
        "rule_correct": (
            NOT_MEASURED
            if pr is None
            else (
                _rule_key(pr)
                == _rule_key(
                    c["applicable_policy_rule"]
                )
            )
        ),

        "expected_supporting_evidence": sorted(
            c["supporting_evidence"]
        ),
        "predicted_supporting_evidence": (
            None
            if ps is None
            else sorted(ps)
        ),
        "supporting_evidence_metrics": _ev(
            c["supporting_evidence"],
            ps,
            False,
        ),

        "expected_contradicting_evidence": sorted(
            c["contradicting_evidence"]
        ),
        "predicted_contradicting_evidence": (
            None
            if pc is None
            else sorted(pc)
        ),
        "contradicting_evidence_metrics": _ev(
            c["contradicting_evidence"],
            pc,
            False,
        ),

        "expected_missing_evidence": sorted(
            c["missing_evidence"]
        ),
        "predicted_missing_evidence": (
            None
            if pm is None
            else sorted(pm)
        ),
        "missing_evidence_metrics": _ev(
            c["missing_evidence"],
            pm,
            True,
        ),

        "expected_counterfactual_correction": cf_exp,
        "predicted_counterfactual_correction": cf,
        "counterfactual_correct": cf_ok,

        "provenance": {
            "benchmark_name": meta.get(
                "benchmark_name"
            ),
            "benchmark_version": meta.get(
                "benchmark_version"
            ),
            "schema_version": meta.get(
                "schema_version"
            ),
            "case_id": c["case_id"],
            "family_id": c["family_id"],
            "split": c["split"],
            "difficulty_category": c[
                "difficulty_category"
            ],
            "label_source": c.get(
                "label_source"
            ),
            "prediction_provenance": copy.deepcopy(
                p.get("provenance")
            ),
        },
    }


def _aggregate(
    rs: List[Dict[str, Any]]
) -> Dict[str, Any]:

    def acc(key):
        v = [
            x[key]
            for x in rs
            if isinstance(x[key], bool)
        ]

        return _ratio(
            sum(v),
            len(v),
        ), len(v)

    out: Dict[str, Any] = {
        "case_count": len(rs)
    }

    den: Dict[str, int] = {}

    for (
        name,
        key,
    ) in (
        (
            "decision_accuracy",
            "decision_correct",
        ),
        (
            "escalation_accuracy",
            "escalation_correct",
        ),
        (
            "policy_rule_accuracy",
            "rule_correct",
        ),
        (
            "counterfactual_correctness",
            "counterfactual_correct",
        ),
    ):

        out[name], den[name] = acc(key)

    for e in _EVIDENCE:

        m = [
            x[f"{e}_evidence_metrics"]
            for x in rs
            if x[
                f"{e}_evidence_metrics"
            ]["true_positives"]
            is not None
        ]

        tp = sum(
            x["true_positives"]
            for x in m
        )

        pred = sum(
            x["predicted_count"]
            for x in m
        )

        exp = sum(
            x["expected_count"]
            for x in m
        )

        for k, v in _prf(
            tp,
            pred,
            exp,
        ).items():

            out[
                f"{e}_evidence_{k}"
            ] = v

        den[
            f"{e}_evidence_predicted"
        ] = pred

        den[
            f"{e}_evidence_expected"
        ] = exp

    out["denominators"] = den

    return out


def _macro(
    by_cat: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:

    out: Dict[str, Any] = {}
    n: Dict[str, int] = {}

    for k in _METRIC_KEYS:

        v = [
            a[k]
            for a in by_cat.values()
            if isinstance(a[k], float)
        ]

        out[k], n[k] = (
            sum(v) / len(v)
            if v
            else NOT_MEASURED,
            len(v),
        )

    out["categories_averaged"] = n

    return out
def evaluate_omni_bench(
    predictions: Optional[Iterable[Dict[str, Any]]],
    cases: Optional[List[Dict[str, Any]]] = None,
    splits=None,
) -> Dict[str, Any]:
    """Evaluate supplied predictions against benchmark ground truth for one split, several, or all (splits=None).

    `predictions` must be None (nothing supplied) or a list/tuple of prediction records; any other container (dict, str, generator, ...) is INVALID_PREDICTIONS.

    Only predicted cases are scored; coverage is reported. Returns status MEASURED / NOT_MEASURED / INVALID_PREDICTIONS / INVALID_BENCHMARK.
    """

    t = _bench()

    cases = (
        t.OMNI_BENCH_CASES
        if cases is None
        else cases
    )

    sel = _resolve_splits(splits)

    meta = t.OMNI_BENCH_METADATA

    res: Dict[str, Any] = {
        "benchmark": {
            "name": meta.get("benchmark_name"),
            "version": meta.get("benchmark_version"),
            "schema_version": meta.get("schema_version"),
        },
        "splits_evaluated": list(sel),
    }

    bad = t.validate_omni_bench_cases(cases)

    if bad:
        return {
            **res,
            "status": STATUS_INVALID_BENCHMARK,
            "errors": bad,
        }

    scope = [
        c
        for c in cases
        if c["split"] in sel
    ]

    res["cases_in_scope"] = len(scope)

    if predictions is not None and not isinstance(
        predictions,
        (list, tuple),
    ):
        return {
            **res,
            "status": STATUS_INVALID_PREDICTIONS,
            "errors": [
                "predictions must be a list or tuple "
                f"of records, got "
                f"{type(predictions).__name__}"
            ],
        }

    accepted: Dict[str, Dict[str, Any]] = {}

    if predictions:
        errs, accepted = _check_records(
            predictions,
            cases,
            sel,
        )

        if errs:
            return {
                **res,
                "status": STATUS_INVALID_PREDICTIONS,
                "errors": errs,
            }

    results = [
        _eval_case(
            c,
            accepted[c["case_id"]],
            meta,
        )
        for c in scope
        if c["case_id"] in accepted
    ]

    by_cat = {
        cat: _aggregate(
            [
                r
                for r in results
                if r["difficulty_category"] == cat
            ]
        )
        for cat in t.OMNI_BENCH_CATEGORY_IDS
        if any(
            r["difficulty_category"] == cat
            for r in results
        )
    }

    res.update(
        {
            "status": (
                STATUS_MEASURED
                if results
                else STATUS_NOT_MEASURED
            ),
            "reason": (
                ""
                if results
                else "no predictions supplied; "
                "nothing was measured"
            ),
            "case_count": len(results),
            "category_count": len(by_cat),
            "coverage": _ratio(
                len(results),
                len(scope),
            ),
            "complete": (
                bool(results)
                and len(results) == len(scope)
            ),
            "overall": _aggregate(results),
            "by_category": by_cat,
            "macro_by_category": _macro(by_cat),
            "by_split": {
                s: _aggregate(
                    [
                        r
                        for r in results
                        if r["split"] == s
                    ]
                )
                for s in sel
            },
            "per_case": results,
            "errors": [],
        }
    )

    return res