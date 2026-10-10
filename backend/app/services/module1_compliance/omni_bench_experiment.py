"""OMNICheck Prompt 12 controlled experiment runner (12D-12H).

This module is an experiment/orchestration layer only.

It uses:
  - omni_bench_system_adapters.py for the frozen V1-V8 systems
  - omni_bench_adversarial_eval.py for the matching perturbation evaluator

It does not alter benchmark ground truth and does not fabricate unavailable
measurements. Missing measurements are recorded as NOT_MEASURED.

Outputs:
  experiment_results.json
  publication_metrics.csv
  statistics.json
  publication plots (PNG)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import random
import sys
import io
import json
import math
import os
import tempfile
import time
import tracemalloc
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VERSIONS = (
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
)

NOT_MEASURED = "NOT_MEASURED"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from . import omni_bench_system_adapters as adapter
    from . import omni_bench_adversarial_eval as evaluator
except ImportError:
    import omni_bench_system_adapters as adapter
    import omni_bench_adversarial_eval as evaluator


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _is_num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _get(root: Dict[str, Any], *path: str) -> Any:
    value: Any = root

    for key in path:
        if not isinstance(value, dict):
            return NOT_MEASURED

        value = value.get(
            key,
            NOT_MEASURED,
        )

    return value


def _records_to_dicts(
    records: Iterable[Any],
) -> List[Dict[str, Any]]:
    out = []

    for record in records:

        if hasattr(record, "to_dict"):
            out.append(
                deepcopy(
                    record.to_dict()
                )
            )

        elif isinstance(record, dict):
            out.append(
                deepcopy(record)
            )

        else:
            raise TypeError(
                "Unsupported record type: "
                f"{type(record).__name__}"
            )

    return out


# ---------------------------------------------------------------------------
# 12F — statistical analysis
# ---------------------------------------------------------------------------

def _wilson(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> Dict[str, Any]:
    """Wilson 95% confidence interval for a binomial proportion."""

    if n <= 0:
        return {
            "estimate": NOT_MEASURED,
            "low": NOT_MEASURED,
            "high": NOT_MEASURED,
            "n": 0,
        }

    p = successes / n

    den = 1.0 + z * z / n

    center = (
        p + z * z / (2.0 * n)
    ) / den

    half = (
        z
        * math.sqrt(
            (
                p * (1.0 - p)
                + z * z / (4.0 * n)
            )
            / n
        )
        / den
    )

    return {
        "estimate": p,
        "low": max(
            0.0,
            center - half,
        ),
        "high": min(
            1.0,
            center + half,
        ),
        "n": n,
    }


def _exact_mcnemar(
    correct_a: List[bool],
    correct_b: List[bool],
) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired boolean outcomes."""

    pairs = [
        (a, b)
        for a, b in zip(
            correct_a,
            correct_b,
        )
        if isinstance(a, bool)
        and isinstance(b, bool)
    ]

    b = sum(
        1
        for a, c in pairs
        if a and not c
    )

    c = sum(
        1
        for a, d in pairs
        if not a and d
    )

    discordant = b + c

    if discordant == 0:
        return {
            "status": NOT_MEASURED,
            "b": b,
            "c": c,
            "discordant": 0,
            "p_value": NOT_MEASURED,
        }

    k = min(
        b,
        c,
    )

    tail = sum(
        math.comb(
            discordant,
            i,
        )
        for i in range(
            k,
            discordant + 1,
        )
    )

    p_value = min(
        1.0,
        2.0 * tail / (2 ** discordant),
    )

    return {
        "status": "MEASURED",
        "b": b,
        "c": c,
        "discordant": discordant,
        "p_value": p_value,
    }


def _decision_pairs(
    evaluation_a: Dict[str, Any],
    evaluation_b: Dict[str, Any],
) -> Tuple[
    List[str],
    List[bool],
    List[bool],
]:
    a = {
        row["perturbation_id"]: row
        for row in evaluation_a.get(
            "per_perturbation",
            [],
        )
    }

    b = {
        row["perturbation_id"]: row
        for row in evaluation_b.get(
            "per_perturbation",
            [],
        )
    }

    ids = sorted(
        set(a) & set(b)
    )

    correct_a = []
    correct_b = []

    for perturbation_id in ids:

        a_ok = a[perturbation_id].get(
            "decision_correct"
        )

        b_ok = b[perturbation_id].get(
            "decision_correct"
        )

        if (
            isinstance(a_ok, bool)
            and isinstance(b_ok, bool)
        ):
            correct_a.append(a_ok)
            correct_b.append(b_ok)

    return (
        ids,
        correct_a,
        correct_b,
    )


def _cluster_bootstrap_accuracy_ci(rows: List[Dict[str, Any]], version: str, n_boot: int = 2000) -> Dict[str, Any]:
    """Deterministic family-cluster percentile CI; refuses undersized cluster sets."""
    usable = [r for r in rows if isinstance(r.get("decision_correct"), bool)
              and isinstance(r.get("source_family_id"), str) and r.get("source_family_id")]
    families: Dict[str, List[bool]] = {}
    for row in usable:
        families.setdefault(row["source_family_id"], []).append(row["decision_correct"])
    cluster_ids = sorted(families)
    if len(cluster_ids) < 20:
        return {"estimate": (sum(r["decision_correct"] for r in usable) / len(usable) if usable else NOT_MEASURED),
                "low": NOT_MEASURED, "high": NOT_MEASURED, "n_records": len(usable),
                "n_clusters": len(cluster_ids), "status": NOT_MEASURED,
                "reason": "cluster bootstrap requires at least 20 independent family_id clusters"}
    seed_material = version + "|" + "|".join(cluster_ids)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sampled = [cluster_ids[rng.randrange(len(cluster_ids))] for __ in cluster_ids]
        outcomes = [value for cluster in sampled for value in families[cluster]]
        if outcomes:
            values.append(sum(outcomes) / len(outcomes))
    values.sort()
    if not values:
        return {"estimate": NOT_MEASURED, "low": NOT_MEASURED, "high": NOT_MEASURED,
                "n_records": len(usable), "n_clusters": len(cluster_ids), "status": NOT_MEASURED}
    return {"estimate": sum(r["decision_correct"] for r in usable) / len(usable),
            "low": values[int(0.025 * (len(values)-1))],
            "high": values[int(0.975 * (len(values)-1))],
            "n_records": len(usable), "n_clusters": len(cluster_ids),
            "bootstrap_replicates": n_boot, "status": "MEASURED",
            "method": "deterministic percentile bootstrap resampling family_id clusters"}


def _holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (key, pval) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pval))
        adjusted[key] = running
    return adjusted


def statistical_summary(evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Family-cluster CIs, paired exact McNemar and Holm-adjusted perturbation subgroups."""
    result = {
        "accuracy_confidence_intervals": {},
        "paired_mcnemar": {},
        "paired_mcnemar_by_perturbation_type": {},
        "method": {
            "confidence_interval": "deterministic family-cluster percentile bootstrap (95%); requires >=20 family_id clusters",
            "paired_test": "exact two-sided McNemar test; reported only with >=10 discordant pairs",
            "pairing_key": "perturbation_id",
            "cluster_key": "source_family_id",
            "multiple_comparison_correction": "Holm correction within each version-pair across measured perturbation-type tests",
            "interpretation": "P-values are descriptive and do not alone establish superiority.",
        },
    }
    for version, evaluation in evaluations.items():
        result["accuracy_confidence_intervals"][version] = _cluster_bootstrap_accuracy_ci(evaluation.get("per_perturbation", []), version)
    versions = list(evaluations)
    for i, va in enumerate(versions):
        for vb in versions[i+1:]:
            ea, eb = evaluations[va], evaluations[vb]
            ids, ca, cb = _decision_pairs(ea, eb)
            exact = _exact_mcnemar(ca, cb)
            discordant = exact.get("discordant", 0) if isinstance(exact, dict) else 0
            if discordant < 10:
                exact = {"status": NOT_MEASURED, "reason": "exact McNemar test requires at least 10 discordant paired outcomes",
                         "b": exact.get("b", 0), "c": exact.get("c", 0), "discordant": discordant}
            result["paired_mcnemar"][f"{va}_vs_{vb}"] = {"n_paired": len(ca), "paired_ids": ids, **exact}

            rows_a = {r.get("perturbation_id"): r for r in ea.get("per_perturbation", [])}
            rows_b = {r.get("perturbation_id"): r for r in eb.get("per_perturbation", [])}
            common = sorted(set(rows_a) & set(rows_b))
            types = sorted({rows_a[k].get("perturbation_type") for k in common if rows_a[k].get("perturbation_type")})
            subgroup, raw_p = {}, {}
            for ptype in types:
                pair_ids = [k for k in common if rows_a[k].get("perturbation_type") == ptype
                            and isinstance(rows_a[k].get("decision_correct"), bool)
                            and isinstance(rows_b[k].get("decision_correct"), bool)]
                aa = [rows_a[k]["decision_correct"] for k in pair_ids]
                bb = [rows_b[k]["decision_correct"] for k in pair_ids]
                test = _exact_mcnemar(aa, bb)
                disc = test.get("discordant", 0)
                if disc < 10:
                    subgroup[ptype] = {"status": NOT_MEASURED, "n_paired": len(pair_ids), "discordant": disc,
                                       "reason": "subgroup exact McNemar requires >=10 discordant pairs"}
                else:
                    subgroup[ptype] = {"status": "MEASURED", "n_paired": len(pair_ids), **test}
                    raw_p[ptype] = test["p_value"]
            adjusted = _holm_adjust(raw_p)
            for ptype, p_adj in adjusted.items():
                subgroup[ptype]["p_value_holm"] = p_adj
            result["paired_mcnemar_by_perturbation_type"][f"{va}_vs_{vb}"] = {
                "tests": subgroup, "holm_family_size": len(raw_p),
                "correction": "Holm-Bonferroni over perturbation types with >=10 discordant pairs"}
    return result


# ---------------------------------------------------------------------------
# 12D — efficiency
# ---------------------------------------------------------------------------

def _run_version(
    records: List[Dict[str, Any]],
    tasks_module: Any,
    version: str,
    run_v1: bool,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    """Run one frozen V1-V8 version."""

    started = time.perf_counter()

    tracemalloc.start()

    try:

        if version == "V1":

            traces = []

            if run_v1:

                for record in records:

                    with tempfile.TemporaryDirectory(
                        prefix="omnibench_v1_"
                    ) as workdir:

                        traces.append(
                            adapter.run_v1_trace(
                                record,
                                tasks_module,
                                workdir,
                            )
                        )

            result = {
                "selected_version": "V1",
                "record_ids": [
                    r["perturbation_id"]
                    for r in records
                ],
                "predictions": None,
                "unanswered": [],
                "component_state": None,
                "blocker": None,
                "v1_traces": traces,
                "unavailable_fields": list(
                    adapter.V1_STRUCTURED_FIELDS
                ),
                "not_measured_reason": (
                    adapter.V1_NOT_MEASURED_REASON
                ),
            }

        else:

            result = adapter.run_version(
                records,
                tasks_module,
                version,
            )

        _current, peak_bytes = (
            tracemalloc.get_traced_memory()
        )

    finally:

        tracemalloc.stop()

    wall_seconds = (
        time.perf_counter()
        - started
    )

    prediction_count = len(
        result.get("predictions")
        or []
    )

    unanswered_count = len(
        result.get("unanswered")
        or []
    )

    record_count = len(
        records
    )

    component_state = result.get("component_state") or {}
    compiler_cache = component_state.get("compiler_cache") if isinstance(component_state, dict) else None
    if not isinstance(compiler_cache, dict):
        compiler_cache = next((p.get("provenance", {}).get("compiler_cache") for p in (result.get("predictions") or [])
                               if isinstance(p.get("provenance", {}).get("compiler_cache"), dict)), {})
    efficiency = {
        "wall_time_seconds": wall_seconds,

        "throughput_records_per_second": (
            record_count / wall_seconds
            if wall_seconds > 0
            else NOT_MEASURED
        ),

        "peak_python_traced_memory_bytes": (
            peak_bytes
        ),

        "prediction_count": (
            prediction_count
        ),

        "unanswered_count": (
            unanswered_count
        ),

        "record_count": (
            record_count
        ),

        "llm_calls": NOT_MEASURED,
        "policy_compiler_llm_call_attempts": compiler_cache.get("cache_misses", NOT_MEASURED),
        "policy_compiler_cache_hits": compiler_cache.get("cache_hits", NOT_MEASURED),
        "policy_compiler_cache_misses": compiler_cache.get("cache_misses", NOT_MEASURED),
        "tokens": NOT_MEASURED,
        "estimated_cost": NOT_MEASURED,

        "llm_metrics_reason": (
            "The frozen adapter/tasks interface "
            "does not expose a reproducible per-run "
            "provider call, token or cost counter."
        ),
    }

    return (
        _jsonable(result),
        efficiency,
    )


# ---------------------------------------------------------------------------
# 12C-12H — publication metric extraction
# ---------------------------------------------------------------------------

def _publication_row(
    version: str,
    evaluation: Dict[str, Any],
    efficiency: Dict[str, Any],
    statistics: Dict[str, Any],
) -> Dict[str, Any]:

    overall = evaluation.get(
        "overall",
        {},
    )

    decision = overall.get(
        "decision",
        {},
    )

    evidence = overall.get(
        "evidence_grounding",
        {},
    )

    contradiction = overall.get(
        "contradiction_detection",
        {},
    )

    verification = overall.get(
        "verification",
        {},
    )

    uncertainty = overall.get(
        "uncertainty",
        {},
    )

    policy = overall.get(
        "policy_reasoning",
        {},
    )

    human = overall.get(
        "human_factors",
        {},
    )

    escalation = overall.get(
        "escalation",
        {},
    )

    unsupported = overall.get(
        "unsupported_decision",
        {},
    )

    ci = (
        statistics
        .get(
            "accuracy_confidence_intervals",
            {},
        )
        .get(
            version,
            {},
        )
    )

    return {
        "version": version,

        "status": evaluation.get(
            "status",
            NOT_MEASURED,
        ),

        "case_count": evaluation.get(
            "case_count",
            0,
        ),

        "coverage": evaluation.get(
            "coverage",
            NOT_MEASURED,
        ),

        # ---------------------------------------------------------------
        # Decision quality
        # ---------------------------------------------------------------

        "accuracy": decision.get(
            "accuracy",
            NOT_MEASURED,
        ),

        "precision": decision.get(
            "macro_precision",
            NOT_MEASURED,
        ),

        "recall": decision.get(
            "macro_recall",
            NOT_MEASURED,
        ),

        "f1": decision.get(
            "macro_f1",
            NOT_MEASURED,
        ),

        "specificity": decision.get(
            "specificity",
            NOT_MEASURED,
        ),

        "mcc": decision.get(
            "mcc",
            NOT_MEASURED,
        ),

        # ---------------------------------------------------------------
        # Evidence quality
        # ---------------------------------------------------------------

        "evidence_precision": _get(
            evidence,
            "pooled_doc_roles",
            "precision",
        ),

        "evidence_recall": _get(
            evidence,
            "pooled_doc_roles",
            "recall",
        ),

        "evidence_f1": _get(
            evidence,
            "pooled_doc_roles",
            "f1",
        ),

        "evidence_completeness": NOT_MEASURED,

        "source_accuracy": evidence.get(
            "grounded_citation_rate",
            NOT_MEASURED,
        ),

        "missing_evidence_flag_accuracy": (
            evidence.get(
                "missing_flag_accuracy",
                NOT_MEASURED,
            )
        ),

        # ---------------------------------------------------------------
        # Policy reasoning
        # ---------------------------------------------------------------

        "rule_extraction_accuracy": (
            policy.get(
                "rule_extraction_accuracy",
                NOT_MEASURED,
            )
        ),

        "condition_accuracy": (
            policy.get(
                "condition_accuracy",
                NOT_MEASURED,
            )
        ),

        "exception_accuracy": (
            policy.get(
                "exception_accuracy",
                NOT_MEASURED,
            )
        ),

        "execution_accuracy": (
            policy.get(
                "execution_accuracy",
                NOT_MEASURED,
            )
        ),

        # ---------------------------------------------------------------
        # Verification
        # ---------------------------------------------------------------

        "unsupported_decision_rate": (
            verification.get(
                "unsupported_decision_rate",
                NOT_MEASURED,
            )
        ),

        "hallucination_rate": (
            verification.get(
                "hallucination_rate",
                NOT_MEASURED,
            )
        ),

        "contradiction_detection_f1": (
            verification.get(
                "contradiction_detection_f1",
                NOT_MEASURED,
            )
        ),

        # ---------------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------------

        "ece": uncertainty.get(
            "ece",
            NOT_MEASURED,
        ),

        "brier": uncertainty.get(
            "brier",
            NOT_MEASURED,
        ),

        "confidence_reliability": (
            uncertainty.get(
                "confidence_reliability",
                NOT_MEASURED,
            )
        ),

        # ---------------------------------------------------------------
        # Human factors
        # ---------------------------------------------------------------

        "escalation_rate": human.get(
            "escalation_rate",
            NOT_MEASURED,
        ),

        "automation_coverage": human.get(
            "automation_coverage",
            NOT_MEASURED,
        ),

        "expert_agreement": human.get(
            "expert_agreement",
            NOT_MEASURED,
        ),

        "review_time": human.get(
            "review_time",
            NOT_MEASURED,
        ),

        "escalation_accuracy": (
            escalation.get(
                "accuracy",
                NOT_MEASURED,
            )
        ),

        # ---------------------------------------------------------------
        # Efficiency
        # ---------------------------------------------------------------

        "latency_seconds": (
            efficiency.get(
                "wall_time_seconds",
                NOT_MEASURED,
            )
        ),

        "throughput_records_per_second": (
            efficiency.get(
                "throughput_records_per_second",
                NOT_MEASURED,
            )
        ),

        "llm_calls": efficiency.get(
            "llm_calls",
            NOT_MEASURED,
        ),

        "policy_compiler_llm_call_attempts": efficiency.get("policy_compiler_llm_call_attempts", NOT_MEASURED),
        "policy_compiler_cache_hits": efficiency.get("policy_compiler_cache_hits", NOT_MEASURED),
        "policy_compiler_cache_misses": efficiency.get("policy_compiler_cache_misses", NOT_MEASURED),

        "tokens": efficiency.get(
            "tokens",
            NOT_MEASURED,
        ),

        "peak_memory_bytes": (
            efficiency.get(
                "peak_python_traced_memory_bytes",
                NOT_MEASURED,
            )
        ),

        "estimated_cost": (
            efficiency.get(
                "estimated_cost",
                NOT_MEASURED,
            )
        ),

        # ---------------------------------------------------------------
        # Confidence interval
        # ---------------------------------------------------------------

        "accuracy_ci_low": ci.get(
            "low",
            NOT_MEASURED,
        ),

        "accuracy_ci_high": ci.get(
            "high",
            NOT_MEASURED,
        ),

        # ---------------------------------------------------------------
        # Denominator audit
        # ---------------------------------------------------------------

        "unsupported_count": (
            unsupported.get(
                "unsupported_count",
                NOT_MEASURED,
            )
        ),

        "expected_insufficient_count": (
            unsupported.get(
                "expected_insufficient_count",
                NOT_MEASURED,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main controlled experiment
# ---------------------------------------------------------------------------

def _proof_rows(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    proofs = []
    for pred in raw.get("predictions") or []:
        prov = pred.get("provenance") if isinstance(pred, dict) else None
        proof = prov.get("component_proof") if isinstance(prov, dict) else None
        if isinstance(proof, dict):
            proofs.append(proof)
        elif isinstance(proof, list):
            proofs.extend(x for x in proof if isinstance(x, dict))
    return proofs


def _compile_ok(proof: Dict[str, Any]) -> bool:
    rows = proof.get("compiled_policy_summaries") or []
    bad = {"NOT_COMPILED", "DISABLED", "FAILED", "ERROR", "INVALID", "NOT_MEASURED"}
    return bool(rows) and all(
        isinstance(x, dict)
        and x.get("status")
        and str(x.get("status")).upper() not in bad
        and not x.get("error")
        and not x.get("evaluation_error")
        for x in rows
    )


def _prediction_has_compiled_policy_proof(prediction: Dict[str, Any]) -> bool:
    """True only when compilation is recorded AND the compiled engine affected a Decision."""
    prov = prediction.get("provenance") or {}
    proof = prov.get("component_proof")
    proofs = [proof] if isinstance(proof, dict) else [x for x in proof or [] if isinstance(x, dict)] if isinstance(proof, list) else []
    return bool(proofs) and all(
        _compile_ok(x)
        and isinstance(x.get("compiled_policy_execution"), dict)
        and x["compiled_policy_execution"].get("effective_on_decision") is True
        for x in proofs
    )


_QUALITY_METRIC_KEYS = {
    "accuracy", "macro_f1", "macro_precision", "macro_recall", "specificity", "mcc",
    "precision", "recall", "f1", "specificity_by_label", "grounded_citation_rate",
    "missing_flag_accuracy", "unsupported_decision_rate", "false_positive_rate",
    "hallucination_rate", "ece", "brier", "confidence_reliability",
    "automation_coverage", "escalation_rate", "escalation_accuracy", "change_accuracy",
    "change_recall", "change_false_alarm_rate", "rule_extraction_accuracy",
    "condition_accuracy", "exception_accuracy", "execution_accuracy",
    "expert_agreement", "review_time", "source_accuracy", "evidence_completeness",
    "contradiction_detection_f1",
}


def _mask_unrealized_quality_metrics(value: Any) -> Any:
    """Recursively prevent publication of quality scores for a version that failed its proof gate."""
    if isinstance(value, dict):
        return {
            key: (NOT_MEASURED if key in _QUALITY_METRIC_KEYS else _mask_unrealized_quality_metrics(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_mask_unrealized_quality_metrics(item) for item in value]
    return value


def _compiled_record_classification(proof: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one compiled-policy graph without treating indeterminacy as success."""
    logs = proof.get("compiled_policy_call_log") or []
    summaries = proof.get("compiled_policy_summaries") or []
    rules = proof.get("compiled_policy_rule_proof") or []
    execution = proof.get("compiled_policy_execution") or {}
    decisions = proof.get("decision_nodes") or []

    called = any(isinstance(row, dict) and row.get("status") == "CALLED" for row in logs)
    raised = any(isinstance(row, dict) and row.get("status") == "RAISED" for row in logs)
    if not called:
        return {"status": "COMPILE_CALL_RAISED" if raised else "NOT_CALLED", "hard_failure": True,
                "reason": "compiled-policy call did not return successfully"}
    if not summaries:
        return {"status": "COMPILE_STATUS_UNAVAILABLE", "hard_failure": True,
                "reason": "no compiled-policy summary was attached"}
    bad_statuses = {"NOT_COMPILED", "DISABLED", "FAILED", "ERROR", "INVALID", "NOT_MEASURED"}
    for summary in summaries:
        if not isinstance(summary, dict) or str(summary.get("status", "")).upper() in bad_statuses:
            return {"status": "COMPILATION_FAILED", "hard_failure": True,
                    "reason": "compiled-policy summary reports an unusable status"}
        if summary.get("error") or summary.get("evaluation_error"):
            return {"status": "COMPILATION_OR_EVALUATION_ERROR", "hard_failure": True,
                    "reason": "compiled-policy summary contains an error"}

    mapped = [r for r in rules if isinstance(r, dict) and r.get("mapped") is True]
    if not rules or not mapped:
        return {"status": "NO_MAPPED_RULES", "hard_failure": True,
                "reason": "no mapped compiled rules are proven on the graph"}

    # A determinate executed rule that failed to reach the Decision is a true propagation defect.
    determinate = [r for r in mapped if r.get("executed") is True and r.get("verdict") not in (None, "INDETERMINATE", "UNEVALUATED", "NO_CONCLUSION")]
    if determinate and execution.get("effective_on_decision") is not True:
        return {"status": "DETERMINATE_VERDICT_NOT_PROPAGATED", "hard_failure": True,
                "reason": "a determinate compiled-rule verdict exists but the Decision did not use the compiled engine"}
    if execution.get("effective_on_decision") is True and int(execution.get("compiled_decision_count") or 0) > 0:
        return {"status": "DETERMINATE_COMPILED_DECISION", "hard_failure": False,
                "reason": "compiled decision is proven to have affected the Decision node"}

    needs_review = [r for r in mapped if str(r.get("status", "")).upper() == "NEEDS_REVIEW"]
    indeterminate = [r for r in mapped if r.get("executed") is True and str(r.get("verdict", "")).upper() == "INDETERMINATE"]
    evidence_required_but_absent = any(
        isinstance(r, dict) and r.get("mapped") is True and r.get("required_evidence")
        for r in rules
    ) and any(
        isinstance(s, dict) and isinstance(s.get("inputs"), dict) and s["inputs"].get("evidence_supplied") is False
        for s in summaries
    )
    evidence_missing = [r for r in indeterminate if r.get("required_evidence")]
    missing_fact_rules = [r for r in indeterminate if r.get("missing_facts")]
    if evidence_required_but_absent:
        return {"status": "EVIDENCE_NOT_SUPPLIED", "hard_failure": False,
                "reason": "a mapped rule requires evidence but the structured evidence input is absent",
                "rule_ids": [r.get("rule_id") for r in rules if isinstance(r, dict) and r.get("mapped") is True and r.get("required_evidence")]}
    if missing_fact_rules:
        return {"status": "INDETERMINATE_MISSING_FACTS", "hard_failure": False,
                "reason": "executed compiled rules are indeterminate because required fact paths are absent",
                "rule_ids": [r.get("rule_id") for r in missing_fact_rules],
                "missing_facts": sorted({str(f) for r in missing_fact_rules for f in (r.get("missing_facts") or [])})}
    if needs_review:
        return {"status": "NEEDS_REVIEW", "hard_failure": False,
                "reason": "mapped compiled rules were withheld from execution for ambiguity or low confidence",
                "rule_ids": [r.get("rule_id") for r in needs_review]}
    return {"status": "INDETERMINATE_UNCLASSIFIED", "hard_failure": False,
            "reason": "compiled stage ran, but no determinate compiled Decision or specific indeterminacy cause was proven"}


def version_realization_report(raw_results: Dict[str, Any], versions: Iterable[str]) -> Dict[str, Any]:
    """Validate component realization; distinguish execution defects from honest indeterminacy."""
    report: Dict[str, Any] = {}
    for version in versions:
        raw = raw_results.get(version, {})
        if version == "V1":
            report[version] = {"status": NOT_MEASURED, "reason": "baseline is free-text only; no structured component proof"}
            continue
        predictions = [p for p in (raw.get("predictions") or []) if isinstance(p, dict)]
        proofs = _proof_rows(raw)
        checks: Dict[str, Any] = {"proof_records": len(proofs)}
        failures, notes = [], []
        prediction_states = [(p.get("provenance") or {}).get("component_state") for p in predictions]
        prediction_states = [x for x in prediction_states if isinstance(x, dict)]

        if version in ("V2", "V3", "V4", "V5", "V6", "V7") and prediction_states:
            if any((x.get("policy_applicability") or {}).get("enabled") is True for x in prediction_states):
                failures.append(f"7C policy applicability must be disabled in {version}")
        if version == "V8" and prediction_states:
            if any(not ((x.get("policy_applicability") or {}).get("enabled") is True and (x.get("policy_applicability") or {}).get("invoked") is True) for x in prediction_states):
                failures.append("V8 did not prove that 7C policy applicability was invoked")
        if not proofs:
            failures.append("no per-record graph component proof was captured")
        if version in ("V2", "V3"):
            if proofs and any(x.get("contradiction_findings_present") is not False for x in proofs):
                failures.append("V2/V3 require contradiction findings to be absent")
            if proofs and any(x.get("cross_document_summary_status") != "DISABLED" for x in proofs):
                failures.append("V2/V3 require cross_document_summary.status=DISABLED")
        if version == "V2" and proofs and any(x.get("compiled_policy_summaries") for x in proofs):
            failures.append("V2 must not attach a compiled_policy summary")
        if version == "V4" and proofs and any(x.get("cross_document_summary_status") in ("DISABLED", NOT_MEASURED, None) for x in proofs):
            failures.append("V4 cross-document linking was not proven active")
        if version in ("V5", "V6", "V7", "V8") and proofs and any(x.get("contradiction_findings_present") is not True for x in proofs):
            failures.append("contradiction findings were not present for every inspected graph")

        if version in ("V3", "V4", "V5", "V6", "V7", "V8"):
            by_id = {p.get("perturbation_id"): p for p in predictions}
            classifications = []
            for index, proof in enumerate(proofs):
                classification = _compiled_record_classification(proof)
                classifications.append({"record_index": index, **classification})
                if classification.get("hard_failure"):
                    failures.append(f"compiled-policy realization defect ({classification['status']}): {classification['reason']}")
            checks["compiled_policy_record_classifications"] = classifications
            checks["compiled_policy_classification_counts"] = {
                status: sum(1 for row in classifications if row.get("status") == status)
                for status in sorted({row.get("status", "UNKNOWN") for row in classifications})
            }
            effective = [x for x in proofs if isinstance(x.get("compiled_policy_execution"), dict)
                         and x["compiled_policy_execution"].get("effective_on_decision") is True]
            compiled = [x for x in proofs if _compile_ok(x)]
            checks["compiled_policy_records"] = len(compiled)
            checks["compiled_policy_proof_coverage"] = len(compiled) / len(proofs) if proofs else NOT_MEASURED
            checks["compiled_policy_effective_decision_records"] = len(effective)
            checks["compiled_policy_effective_decision_coverage"] = len(effective) / len(proofs) if proofs else NOT_MEASURED
            if len(compiled) != len(proofs):
                notes.append("one or more records did not prove a usable compiled-policy summary")
            unresolved = [row for row in classifications if not row.get("hard_failure") and row.get("status") != "DETERMINATE_COMPILED_DECISION"]
            if unresolved:
                notes.append("compiled-policy execution was attempted, but one or more records remain indeterminate or require review; these records are not counted as determinate successes")
            if version == "V3":
                call_records = sum(1 for proof in proofs if any(
                    isinstance(row, dict) and row.get("status") == "CALLED"
                    for row in (proof.get("compiled_policy_call_log") or [])
                ))
                checks["compiled_policy_call_proof_records"] = call_records
                state_failures = [p.get("perturbation_id") for p in predictions
                                  if ((p.get("provenance") or {}).get("component_state") or {}).get("compiled_policy", {}).get("enabled") is not True
                                  or ((p.get("provenance") or {}).get("component_state") or {}).get("compiled_policy", {}).get("invoked") is not True]
                checks["compiled_policy_invoked_state_records"] = len(predictions) - len(state_failures)
                if state_failures:
                    failures.append("V3 component_state must record compiled_policy.enabled=true and invoked=true for every prediction")

        if version == "V7" and "V6" not in raw_results:
            failures.append("V7 parity gate requires V6 output in the same run")
        if version == "V7" and "V6" in raw_results:
            p6 = {p.get("perturbation_id"): (p.get("predicted_decision"), p.get("predicted_escalation")) for p in raw_results["V6"].get("predictions") or []}
            p7 = {p.get("perturbation_id"): (p.get("predicted_decision"), p.get("predicted_escalation")) for p in raw.get("predictions") or []}
            common = sorted(set(p6) & set(p7))
            parity = all(p6[k] == p7[k] for k in common) if common else False
            checks["v6_v7_decision_parity"] = {"status": "PASS" if parity else "FAIL", "paired_records": len(common)}
            if not parity: failures.append("V7 decision/escalation outputs are not identical to V6")

        if failures:
            status = "FAIL"
        elif notes:
            status = "PARTIAL"
        else:
            status = "PASS"
        report[version] = {"status": status, "checks": checks, "failures": failures, "notes": notes}
    return report


def run_controlled_experiment(
    records: Iterable[Any],
    tasks_module: Any,
    output_dir: str,
    versions: Tuple[str, ...] = VERSIONS,
    run_v1: bool = True,
) -> Dict[str, Any]:
    """Run the complete controlled V1-V8 experiment."""

    requested = tuple(
        versions
    )

    invalid = [
        v
        for v in requested
        if v not in VERSIONS
    ]

    if invalid:
        raise ValueError(
            f"Invalid version(s): {invalid}; "
            f"expected {VERSIONS}"
        )

    output = Path(
        output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    recs = _records_to_dicts(
        records
    )

    raw_results = {}
    efficiency = {}
    prediction_map = {}

    # ---------------------------------------------------------------
    # Run every version against the identical record list.
    # ---------------------------------------------------------------

    for version in requested:

        raw, eff = _run_version(
            recs,
            tasks_module,
            version,
            run_v1,
        )

        raw_results[version] = raw
        efficiency[version] = eff

        if version == "V1":
            prediction_map[version] = None

        else:
            prediction_map[version] = (
                raw.get("predictions")
            )

    # ---------------------------------------------------------------
    # Pre-publication component realization gates.
    # ---------------------------------------------------------------

    realization = version_realization_report(raw_results, requested)

    # ---------------------------------------------------------------
    # Unified evaluator.
    #
    # IMPORTANT:
    # This is the adversarial evaluator because the frozen adapter's
    # records are PerturbationRecord records, not base OMNI-Bench cases.
    # ---------------------------------------------------------------

    evaluations = (
        evaluator.evaluate_version_suite(
            recs,
            prediction_map,
            versions=requested,
        )
    )

    # V3+ can silently fall back to the legacy engine when compilation fails. Keep
    # the all-record evaluation above, and separately score only records whose graph
    # provenance proves compiled-policy realization.
    compiled_only_evaluations: Dict[str, Any] = {}
    rec_by_id = {r["perturbation_id"]: r for r in recs}
    for version in requested:
        if version not in ("V3", "V4", "V5", "V6", "V7", "V8"):
            compiled_only_evaluations[version] = {"status": NOT_MEASURED, "reason": "compiled-policy-only subset is not applicable"}
            continue
        selected = [p for p in (raw_results.get(version, {}).get("predictions") or []) if _prediction_has_compiled_policy_proof(p)]
        selected_ids = {p.get("perturbation_id") for p in selected}
        selected_records = [r for r in recs if r["perturbation_id"] in selected_ids]
        if selected_records:
            compiled_only_evaluations[version] = evaluator.evaluate_adversarial_predictions(selected_records, selected)
        else:
            compiled_only_evaluations[version] = {"status": NOT_MEASURED, "case_count": 0, "reason": "no records with observed compiled_policy realization"}
        compiled_only_evaluations[version]["subset_policy"] = "only records with observed compiled_policy summary AND proof that the compiled engine affected a Decision; missing or ineffective proof is excluded"

    by_version = evaluations.get(
        "by_version",
        {},
    )

    # A failed realization gate means that version did not execute the claimed
    # system configuration. Keep its raw predictions/proof for diagnosis, but
    # do not publish its evaluator quality metrics as system performance.
    for version in requested:
        if realization.get(version, {}).get("status") == "FAIL" and version in by_version:
            masked = _mask_unrealized_quality_metrics(by_version[version])
            if isinstance(masked, dict):
                masked["status"] = "VERSION_NOT_REALIZED"
                masked["quality_metrics_status"] = "NOT_MEASURED: realization gate failed"
                by_version[version] = masked
    if isinstance(evaluations, dict):
        evaluations["by_version"] = by_version

    # ---------------------------------------------------------------
    # Statistics.
    # ---------------------------------------------------------------

    statistics = statistical_summary(
        by_version
    )

    # ---------------------------------------------------------------
    # Publication table.
    # ---------------------------------------------------------------

    publication_rows = [
        _publication_row(
            version,
            by_version.get(version, {}),
            efficiency.get(version, {}),
            statistics,
        )
        for version in requested
    ]
    metric_columns = {
        "accuracy", "precision", "recall", "f1", "specificity", "mcc",
        "evidence_precision", "evidence_recall", "evidence_f1", "evidence_completeness", "source_accuracy",
        "missing_evidence_flag_accuracy", "rule_extraction_accuracy", "condition_accuracy",
        "exception_accuracy", "execution_accuracy", "unsupported_decision_rate",
        "hallucination_rate", "contradiction_detection_f1", "ece", "brier",
        "confidence_reliability", "escalation_rate", "automation_coverage",
        "expert_agreement", "review_time", "escalation_accuracy",
    }
    for row in publication_rows:
        gate = realization.get(row["version"], {}).get("status")
        if gate == "FAIL":
            row["status"] = "VERSION_NOT_REALIZED"
            for key in metric_columns: row[key] = NOT_MEASURED
        elif gate == "PARTIAL":
            row["status"] = "PARTIAL_REALIZATION"
        compiled_eval = compiled_only_evaluations.get(row["version"], {})
        compiled_overall = compiled_eval.get("overall", {}) if isinstance(compiled_eval, dict) else {}
        compiled_decision = compiled_overall.get("decision", {})
        row["compiled_only_case_count"] = compiled_eval.get("case_count", 0) if isinstance(compiled_eval, dict) else 0
        row["compiled_only_accuracy"] = compiled_decision.get("accuracy", NOT_MEASURED)
        row["compiled_only_macro_f1"] = compiled_decision.get("macro_f1", NOT_MEASURED)

    # ---------------------------------------------------------------
    # Complete machine-readable artifact.
    # ---------------------------------------------------------------

    artifacts = {
        "schema_version": (
            "12D-12H-2.0"
        ),

        "experiment": {
            "name": (
                "OMNICheck controlled "
                "V1-V8 experiment"
            ),

            "versions": list(
                requested
            ),

            "record_count": len(
                recs
            ),

            "run_v1": run_v1,

            "same_records_for_all_versions": True,

            "ground_truth_source": (
                "adversarial perturbation records; "
                "predictions never supply or alter "
                "ground truth"
            ),
        },

        "raw_results": raw_results,

        "version_realization": _jsonable(realization),

        "efficiency": efficiency,

        "evaluations": _jsonable(
            evaluations
        ),

        "compiled_only_evaluations": _jsonable(compiled_only_evaluations),

        "statistics": _jsonable(
            statistics
        ),

        "publication_rows": _jsonable(
            publication_rows
        ),

        "measurement_policy": {
            "unavailable_metrics": (
                NOT_MEASURED
            ),

            "llm_calls": (
                "NOT_MEASURED because the frozen "
                "adapter/tasks interface exposes "
                "no reproducible provider counter"
            ),

            "tokens": (
                "NOT_MEASURED because the frozen "
                "adapter/tasks interface exposes "
                "no reproducible token counter"
            ),

            "evidence_completeness": (
                "NOT_MEASURED because the frozen adversarial evaluator does not define a comparable completeness label/denominator for the full evidence set"
            ),

            "source_accuracy_caveat": (
                "grounded_citation_rate is a limited source-accuracy proxy; the adapter maps known filenames to supplied doc_ids and drops unknown filenames, making the measure potentially optimistic"
            ),

            "hallucination_rate_caveat": (
                "citation hallucination only, not semantic hallucination; unknown filenames may be dropped before scoring"
            ),

            "estimated_cost": (
                "NOT_MEASURED because no provider "
                "pricing/call data is exposed by "
                "the frozen interface"
            ),

            "expert_agreement": (
                "NOT_MEASURED because no human-review "
                "labels are supplied"
            ),

            "review_time": (
                "NOT_MEASURED because no human "
                "review-time observations are supplied"
            ),

            "policy_decomposition": (
                "NOT_MEASURED because the frozen "
                "V1-V8 prediction interface does not "
                "expose separate condition, exception "
                "or execution predictions"
            ),
        },
    }

    # ---------------------------------------------------------------
    # Save machine-readable outputs.
    # ---------------------------------------------------------------

    (
        output / "experiment_results.json"
    ).write_text(
        json.dumps(
            _jsonable(
                artifacts
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        output / "publication_metrics.csv"
    ).write_text(
        _rows_to_csv(
            publication_rows
        ),
        encoding="utf-8",
    )

    (
        output / "statistics.json"
    ).write_text(
        json.dumps(
            _jsonable(
                statistics
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Plots are supplementary. If matplotlib is unavailable, preserve all data and
    # record the plotting blocker rather than failing or fabricating a figure.
    try:
        artifacts["publication_plots"] = render_publication_plots(
            str(output / "experiment_results.json"), str(output / "plots")
        )
    except Exception as exc:  # noqa: BLE001
        artifacts["publication_plots"] = {
            "status": NOT_MEASURED,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    (output / "experiment_results.json").write_text(
        json.dumps(_jsonable(artifacts), indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifacts


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _rows_to_csv(
    rows: List[Dict[str, Any]],
) -> str:

    if not rows:
        return ""

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(
            rows[0].keys()
        ),
        extrasaction="ignore",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        rows
    )

    return buffer.getvalue()


def write_publication_table(
    results_json: str,
    output_csv: str,
) -> None:
    """Regenerate the publication CSV from experiment_results.json."""

    data = json.loads(
        Path(
            results_json
        ).read_text(
            encoding="utf-8"
        )
    )

    rows = data.get(
        "publication_rows",
        [],
    )

    Path(
        output_csv
    ).write_text(
        _rows_to_csv(
            rows
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 12H — publication plots
# ---------------------------------------------------------------------------

def render_publication_plots(
    results_json: str,
    output_dir: str,
) -> List[str]:
    """Create publication plots only for actually measured metrics.

    NOT_MEASURED values are omitted and never converted to zero.
    """

    import matplotlib.pyplot as plt

    data = json.loads(
        Path(
            results_json
        ).read_text(
            encoding="utf-8"
        )
    )

    rows = data.get(
        "publication_rows",
        [],
    )

    output = Path(
        output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_specs = [
        (
            "accuracy",
            "Decision accuracy",
            "decision_accuracy.png",
        ),
        (
            "f1",
            "Macro F1",
            "decision_macro_f1.png",
        ),
        (
            "evidence_f1",
            "Pooled evidence F1",
            "evidence_f1.png",
        ),
        (
            "contradiction_detection_f1",
            "Contradiction detection F1",
            "contradiction_detection_f1.png",
        ),
        (
            "unsupported_decision_rate",
            "Unsupported decision rate",
            "unsupported_decision_rate.png",
        ),
        (
            "ece",
            "Expected calibration error",
            "ece.png",
        ),
        (
            "latency_seconds",
            "Latency (seconds)",
            "latency_seconds.png",
        ),
        (
            "throughput_records_per_second",
            "Throughput (records/second)",
            "throughput.png",
        ),
        (
            "peak_memory_bytes",
            "Peak Python traced memory (bytes)",
            "peak_memory.png",
        ),
    ]

    created = []

    for (
        key,
        title,
        filename,
    ) in plot_specs:

        points = [
            (
                row["version"],
                row[key],
            )
            for row in rows
            if _is_num(
                row.get(key)
            )
        ]

        if not points:
            continue

        x = [
            p[0]
            for p in points
        ]

        y = [
            p[1]
            for p in points
        ]

        fig, ax = plt.subplots(
            figsize=(8.0, 4.8)
        )

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
        )

        ax.set_title(
            title,
            fontsize=13,
        )

        ax.set_xlabel(
            "System version"
        )

        ax.set_ylabel(
            title
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        path = (
            output / filename
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        created.append(
            str(path)
        )

    return created


# ---------------------------------------------------------------------------
# Embedded tests
# ---------------------------------------------------------------------------

class ExperimentFrameworkTests(
    unittest.TestCase
):

    def test_v3_realization_rejects_compiled_status_without_invoked_state(self):
        proof = {
            "compiled_policy_summaries": [{"status": "COMPILED"}],
            "compiled_policy_call_log": [{"status": "CALLED", "result_status": "COMPILED"}],
            "compiled_policy_rule_proof": [{"rule_id": "R1", "mapped": True, "status": "VALID", "executed": True, "verdict": "SATISFIED", "missing_facts": [], "required_evidence": []}],
            "contradiction_findings_present": False,
            "cross_document_summary_status": "DISABLED",
        }
        pred = {
            "perturbation_id": "pilot-1",
            "provenance": {
                "component_state": {"compiled_policy": {"enabled": True, "invoked": False}},
                "component_proof": proof,
            },
        }
        raw = {"V3": {"predictions": [pred]}}
        self.assertEqual(version_realization_report(raw, ("V3",))["V3"]["status"], "FAIL")
        pred["provenance"]["component_state"]["compiled_policy"]["invoked"] = True
        proof["compiled_policy_execution"] = {
            "effective_on_decision": True,
            "compiled_decision_count": 1,
        }
        report = version_realization_report(raw, ("V3",))["V3"]
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"]["compiled_policy_call_proof_records"], 1)
        self.assertEqual(report["checks"]["compiled_policy_effective_decision_records"], 1)

    def test_v3_legitimate_indeterminate_is_partial_not_a_false_pass_or_hard_failure(self):
        proof = {
            "compiled_policy_summaries": [{"status": "COMPILED_WITH_REVIEW", "inputs": {"evidence_supplied": False}}],
            "compiled_policy_call_log": [{"status": "CALLED", "result_status": "COMPILED_WITH_REVIEW"}],
            "compiled_policy_rule_proof": [
                {"rule_id": "R1", "mapped": True, "status": "NEEDS_REVIEW", "executed": False, "verdict": "INDETERMINATE", "missing_facts": [], "required_evidence": [{"type": "approval"}]},
                {"rule_id": "R2", "mapped": True, "status": "VALID", "executed": True, "verdict": "INDETERMINATE", "missing_facts": ["expense_violation.confirmed"], "required_evidence": []},
            ],
            "compiled_policy_execution": {"effective_on_decision": False, "compiled_decision_count": 0, "legacy_decision_count": 1},
            "contradiction_findings_present": False,
            "cross_document_summary_status": "DISABLED",
        }
        pred = {"perturbation_id": "pilot-2", "provenance": {"component_state": {"compiled_policy": {"enabled": True, "invoked": True}}, "component_proof": proof}}
        report = version_realization_report({"V3": {"predictions": [pred]}}, ("V3",))["V3"]
        self.assertEqual(report["status"], "PARTIAL")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "EVIDENCE_NOT_SUPPLIED")
        self.assertEqual(report["checks"]["compiled_policy_effective_decision_records"], 0)
        self.assertEqual(report["failures"], [])

    def test_determinate_compiled_result_not_propagated_is_hard_failure(self):
        proof = {
            "compiled_policy_summaries": [{"status": "COMPILED"}],
            "compiled_policy_call_log": [{"status": "CALLED", "result_status": "COMPILED"}],
            "compiled_policy_rule_proof": [{"rule_id": "R1", "mapped": True, "status": "VALID", "executed": True, "verdict": "VIOLATION", "missing_facts": [], "required_evidence": []}],
            "compiled_policy_execution": {"effective_on_decision": False, "compiled_decision_count": 0, "legacy_decision_count": 1},
            "contradiction_findings_present": False,
            "cross_document_summary_status": "DISABLED",
        }
        pred = {"perturbation_id": "pilot-3", "provenance": {"component_state": {"compiled_policy": {"enabled": True, "invoked": True}}, "component_proof": proof}}
        report = version_realization_report({"V3": {"predictions": [pred]}}, ("V3",))["V3"]
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("DETERMINATE_VERDICT_NOT_PROPAGATED" in x for x in report["failures"]))

    def test_compiled_policy_not_called_is_hard_failure(self):
        proof = {"compiled_policy_summaries": [{"status": "COMPILED"}], "compiled_policy_call_log": [],
                 "compiled_policy_rule_proof": [{"rule_id": "R1", "mapped": True, "status": "VALID", "executed": False, "verdict": "INDETERMINATE"}],
                 "contradiction_findings_present": False, "cross_document_summary_status": "DISABLED"}
        pred = {"perturbation_id": "pilot-4", "provenance": {"component_state": {"compiled_policy": {"enabled": True, "invoked": False}}, "component_proof": proof}}
        report = version_realization_report({"V3": {"predictions": [pred]}}, ("V3",))["V3"]
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"]["compiled_policy_record_classifications"][0]["status"], "NOT_CALLED")

    def test_unrealized_quality_metrics_are_masked_not_zeroed(self):
        source = {"decision": {"accuracy": 0.0, "mcc": "NOT_MEASURED", "true_positives": 0}, "case_count": 2}
        masked = _mask_unrealized_quality_metrics(source)
        self.assertEqual(masked["decision"]["accuracy"], NOT_MEASURED)
        self.assertEqual(masked["decision"]["mcc"], NOT_MEASURED)
        self.assertEqual(masked["decision"]["true_positives"], 0)
        self.assertEqual(masked["case_count"], 2)

    def test_wilson(self):

        result = _wilson(
            5,
            10,
        )

        self.assertAlmostEqual(
            result["estimate"],
            0.5,
        )

        self.assertLess(
            result["low"],
            0.5,
        )

        self.assertGreater(
            result["high"],
            0.5,
        )

    def test_mcnemar_no_discordance(
        self,
    ):

        result = _exact_mcnemar(
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
        )

        self.assertEqual(
            result["status"],
            NOT_MEASURED,
        )

    def test_mcnemar_symmetric_pair(
        self,
    ):

        result = _exact_mcnemar(
            [
                True,
                False,
            ],
            [
                False,
                True,
            ],
        )

        self.assertEqual(
            result["b"],
            1,
        )

        self.assertEqual(
            result["c"],
            1,
        )

        self.assertEqual(
            result["discordant"],
            2,
        )

        self.assertEqual(
            result["p_value"],
            1.0,
        )

    def test_not_measured_is_not_zero(
        self,
    ):

        result = _wilson(
            0,
            0,
        )

        self.assertEqual(
            result["estimate"],
            NOT_MEASURED,
        )

    def test_cluster_bootstrap_requires_twenty_families(self):
        rows = [{"decision_correct": True, "source_family_id": f"family-{i}"} for i in range(19)]
        result = _cluster_bootstrap_accuracy_ci(rows, "V2", n_boot=100)
        self.assertEqual(result["status"], NOT_MEASURED)
        self.assertEqual(result["n_clusters"], 19)

    def test_cluster_bootstrap_is_deterministic_with_enough_families(self):
        rows = [{"decision_correct": i % 3 != 0, "source_family_id": f"family-{i}"} for i in range(24)]
        a = _cluster_bootstrap_accuracy_ci(rows, "V4", n_boot=100)
        b = _cluster_bootstrap_accuracy_ci(rows, "V4", n_boot=100)
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "MEASURED")
        self.assertLessEqual(a["low"], a["estimate"])
        self.assertGreaterEqual(a["high"], a["estimate"])

    def test_mcnemar_small_discordant_count_is_not_measured(self):
        summary = statistical_summary({
            "V2": {"per_perturbation": [{"perturbation_id": "a", "decision_correct": True, "source_family_id": "f1", "perturbation_type": "x"}]},
            "V3": {"per_perturbation": [{"perturbation_id": "a", "decision_correct": False, "source_family_id": "f1", "perturbation_type": "x"}]},
        })
        self.assertEqual(summary["paired_mcnemar"]["V2_vs_V3"]["status"], NOT_MEASURED)

    def test_metric_extraction_preserves_missing(
        self,
    ):

        row = _publication_row(
            "V1",
            {
                "status": NOT_MEASURED,
                "case_count": 0,
                "coverage": 0.0,
                "overall": {},
            },
            {
                "wall_time_seconds": 1.0,
                "throughput_records_per_second": 10.0,
                "peak_python_traced_memory_bytes": 100,
                "llm_calls": NOT_MEASURED,
                "tokens": NOT_MEASURED,
                "estimated_cost": NOT_MEASURED,
            },
            {
                "accuracy_confidence_intervals": {
                    "V1": {
                        "low": NOT_MEASURED,
                        "high": NOT_MEASURED,
                    }
                }
            },
        )

        self.assertEqual(
            row["accuracy"],
            NOT_MEASURED,
        )

        self.assertEqual(
            row["llm_calls"],
            NOT_MEASURED,
        )

        self.assertEqual(
            row["latency_seconds"],
            1.0,
        )


def _load_experiment_records(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the frozen generator output; does not call any model/provider."""
    try:
        from .omni_bench_adversarial_generator import generate_adversarial_suite
    except ImportError:
        from omni_bench_adversarial_generator import generate_adversarial_suite
    records = _records_to_dicts(generate_adversarial_suite())
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError("adversarial generator returned zero records")
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OMNICheck Prompt 12 controlled V1-V8 experiment")
    parser.add_argument("--mode", choices=("pilot", "full", "tests"), default="pilot",
                        help="pilot is quota-conscious; full requires explicit confirmation")
    parser.add_argument("--limit", type=int, default=2, help="number of generated records for pilot mode")
    parser.add_argument("--versions", default=None, help="comma-separated versions; default is V2 for pilot and V1-V8 for full")
    parser.add_argument("--output-dir", default="prompt12_results", help="directory for JSON, CSV, statistics and plots")
    parser.add_argument("--run-v1", action="store_true", help="invoke baseline free-text path (can consume provider quota)")
    parser.add_argument("--confirm-expensive-run", action="store_true", help="required for full mode; full mode can make many provider calls")
    args = parser.parse_args(argv)

    if args.mode == "tests":
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    versions = VERSIONS if args.versions is None and args.mode == "full" else (("V2",) if args.versions is None else tuple(v.strip() for v in args.versions.split(",") if v.strip()))
    invalid = [v for v in versions if v not in VERSIONS]
    if invalid:
        parser.error(f"invalid version(s) {invalid}; expected {VERSIONS}")
    if args.mode == "full" and not args.confirm_expensive_run:
        parser.error("full mode can trigger thousands of model-backed executions; add --confirm-expensive-run after pilot/proof checks")
    if args.mode == "full" and "V1" in versions and not args.run_v1:
        parser.error("V1 is free-text baseline and must be invoked for a full V1-V8 comparison; add --run-v1 (provider quota may be used)")
    if args.mode == "pilot":
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        records = _load_experiment_records(args.limit)
    else:
        records = _load_experiment_records()
    try:
        try:
            from . import tasks as tasks_module
        except ImportError:
            import tasks as tasks_module
    except Exception as exc:
        print(f"Could not import local tasks.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    out = Path(args.output_dir)
    if args.mode == "full":
        # One-record proof gate before any scored full run. Fail fast to avoid wasting
        # provider quota on later versions after an early hard-gate failure.
        proof_dir = out / "pre_data_proof"
        proof_dir.mkdir(parents=True, exist_ok=True)
        proof_raw, proof_eff = {}, {}
        proof_report = {}
        for version in versions:
            raw, eff = _run_version(records[:1], tasks_module, version, args.run_v1)
            proof_raw[version], proof_eff[version] = raw, eff
            proof_report = version_realization_report(proof_raw, tuple(proof_raw))
            if proof_report.get(version, {}).get("status") == "FAIL":
                (proof_dir / "proof_results.json").write_text(
                    json.dumps(_jsonable({"versions": list(versions), "realization": proof_report,
                                          "raw_results": proof_raw, "efficiency": proof_eff}), indent=2, sort_keys=True),
                    encoding="utf-8")
                print(f"STOPPED before full scored run: proof gate failed for {version}", file=sys.stderr)
                print(f"Inspect {proof_dir / 'proof_results.json'}; no full benchmark was run.", file=sys.stderr)
                return 3
        (proof_dir / "proof_results.json").write_text(
            json.dumps(_jsonable({"versions": list(versions), "realization": proof_report,
                                  "raw_results": proof_raw, "efficiency": proof_eff}), indent=2, sort_keys=True),
            encoding="utf-8")
    artifacts = run_controlled_experiment(records, tasks_module, str(out), versions=versions, run_v1=args.run_v1)
    source_paths = [Path(__file__).resolve(), Path(adapter.__file__).resolve(), Path(evaluator.__file__).resolve(),
                    Path(getattr(tasks_module, "__file__", "")).resolve()]
    try:
        from . import omni_bench_adversarial_generator as generator_module, omni_bench_cases as cases_module
    except ImportError:
        import omni_bench_adversarial_generator as generator_module
        import omni_bench_cases as cases_module
    source_paths.extend([Path(generator_module.__file__).resolve(), Path(cases_module.__file__).resolve()])
    source_hashes = {}
    for source_path in source_paths:
        if source_path.is_file(): source_hashes[str(source_path)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "12D-12H-2.2",
        "python_version": platform.python_version(),
        "record_count_requested": len(records),
        "versions_requested": list(versions),
        "mode": args.mode,
        "run_v1": bool(args.run_v1),
        "source_sha256": source_hashes,
        "note": "Pilot output is not publication evidence. Run the pre-data gates and confirm version realization before a full scored run.",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finished {args.mode} run: {len(records)} records; versions={','.join(versions)}")
    print(f"Results: {out.resolve()}")
    print("Check experiment_results.json for NOT_MEASURED fields and unanswered records before interpreting metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
