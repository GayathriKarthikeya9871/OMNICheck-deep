"""OMNI-Bench dataset documentation + deterministic structural validation report.
Offline, no LLM / network / randomness. Structural checks only: this module never produces or reports benchmark performance scores.
It reads the benchmark (tasks.OMNI_BENCH_CASES, built by omni_bench_cases.py) and the evaluator (omni_bench_eval.py) and changes neither."""
import ast
import copy
import hashlib
import json
import pathlib
from typing import Any, Dict, List

EXPECTED_CASE_COUNT = 128
EXPECTED_SPLIT_COUNTS = {"TRAIN": 64, "DEV": 32, "TEST": 32}
EXPECTED_CATEGORY_COUNT = 15
EXPECTED_RULE_ONLY_COUNTERFACTUALS = 16
# sha256 of the canonical JSON of the 128 cases (sort_keys, compact separators): any edit to a case or label changes it, so it must be a deliberate, reviewed change.
OMNI_BENCH_CASES_SHA256 = "e47e5d7b15bc658c29366b26f6fcc29f4dc628cbedba0eea2135e1c755bdc4bb"
_FORBIDDEN_IMPORTS = frozenset(("random", "secrets", "requests", "socket", "urllib", "urllib3", "http", "httpx", "aiohttp", "openai", "anthropic", "time", "uuid", "numpy"))
_BENCH_MODULES = ("omni_bench_cases.py", "omni_bench_eval.py", "omni_bench_docs.py")

OMNI_BENCH_DATASET_CARD: Dict[str, Any] = {
    "purpose": ("Measure whether a document-compliance reasoning system reaches the right decision (COMPLIANT / NON_COMPLIANT / INSUFFICIENT_EVIDENCE), escalation, applicable policy rule, "
                "evidence roles (supporting / contradicting / missing) and counterfactual correction on small, fully specified multi-document cases."),
    "schema": {
        "benchmark_name": "OMNI-Bench", "benchmark_version": "0.1.0", "schema_version": "1.0",
        "case_fields": ["case_id", "family_id", "split", "document_set", "policy", "expected_decision", "applicable_policy_rule", "supporting_evidence", "contradicting_evidence",
                        "missing_evidence", "expected_escalation", "counterfactual_correction", "difficulty_category", "label_source"],
        "prediction_fields": ["case_id", "predicted_decision (required)", "predicted_escalation", "predicted_applicable_policy_rule", "predicted_supporting_evidence",
                              "predicted_contradicting_evidence", "predicted_missing_evidence", "predicted_counterfactual_correction", "provenance"],
        "case_count": EXPECTED_CASE_COUNT, "category_count": EXPECTED_CATEGORY_COUNT, "split_counts": dict(EXPECTED_SPLIT_COUNTS)},
    "split_methodology": ("Three disjoint splits assigned by the case author, never by randomness: TRAIN (64) for method/prompt development, DEV (32) for tuning and model selection, "
                          "TEST (32) held out for final reporting only."),
    "leakage_prevention": ("Cases that are variants of one another share a family_id and must all sit in one split; case_id is globally unique; no identical document_set+policy content "
                           "may appear in two splits (checked by SHA-256 of the canonical content); ground truth is never derived from system predictions."),
    "case_generation_methodology": ("Hand-authored synthetic cases built deterministically from fixed parameter tables and string templates in omni_bench_cases.py "
                                    "(build_omni_bench_cases is a pure function; each call returns fresh, equal objects)."),
    "ground_truth_construction": ("Every label is a literal chosen by the case author from the policy text quoted in the case (label_source = synthetic_construction); no label is computed by, "
                                  "or copied from, the system under test. Embedded instructions in documents are inert data and never influence labels."),
    "reproducibility": ("No randomness (seed: none; SEED=0 recorded), no wall-clock input, no network, no LLM. Identical inputs give identical cases, validation and evaluation results. "
                        "A SHA-256 fingerprint of the 128 cases is pinned in this module to make unreviewed edits detectable."),
    "evaluation_protocol": {
        "scope": "Only the predictions supplied are scored; predictions never supply or alter ground truth; coverage is reported. Predictions must be a list/tuple of records.",
        "decision_escalation": "exact match against expected_decision / expected_escalation (accuracy over cases where a value was predicted).",
        "policy_rule": "match by the set of rule ids found in the text (falls back to normalized text when none).",
        "evidence": "set precision / recall / F1 for supporting, contradicting (doc_ids) and missing (normalized text), micro-aggregated; also macro over categories.",
        "counterfactual": ("wording-independent anchor check on the benchmark's own text: rule ids, decision labels, doc_ids, numbers and a no-change marker. "
                           "Same anchors -> correct; extra or conflicting anchor -> incorrect; anchors missing or not comparable -> NOT_MEASURED."),
        "reporting": "overall, per difficulty category, macro over categories and per split. TEST is for final reporting only."},
    "not_measured_meaning": ("NOT_MEASURED means no valid score can be computed (nothing supplied, an empty denominator, or a counterfactual whose equivalence cannot be proven "
                             "deterministically). It is never a zero and never a guess, and is excluded from the denominator of that metric."),
    "known_limitations": [
        "Small (128 cases), synthetic, hand-authored and English-only; it checks reasoning on constructed cases, not real-world document variety.",
        f"{EXPECTED_RULE_ONLY_COUNTERFACTUALS} counterfactual_correction labels are rule-only (they carry a rule id but no decision label, doc_id or other anchor): they provide weaker semantic "
        "verification than cases with rule + decision/other anchors, because a prediction repeating the rule id can match while changing the meaning. These cases are intentionally not altered.",
        "No counterfactual label references a doc_id, so the doc_id anchor is never exercised on the real cases.",
        "A counterfactual that omits every anchor is NOT_MEASURED, not wrong; a paraphrase that drops the rule id or decision label likewise cannot be proven correct.",
        "Missing-evidence predictions are compared as normalized exact text, which is strict for free-form wording.",
        "A perfect-prediction fixture built from the labels validates infrastructure only; it is not a performance result and no baseline scores are reported here."]}


def get_omni_bench_dataset_card() -> Dict[str, Any]:
    """The dataset card plus the 15 category definitions read from the benchmark (single source of truth). Returns a deep copy."""
    from . import tasks
    card = copy.deepcopy(OMNI_BENCH_DATASET_CARD)
    card["categories"] = [{"id": c[0], "name": c[1], "definition": c[2]} for c in tasks.OMNI_BENCH_CATEGORIES]
    card["splits"] = dict(tasks.OMNI_BENCH_SPLITS)
    return card


def omni_bench_cases_fingerprint(cases=None) -> str:
    from . import tasks
    cases = tasks.OMNI_BENCH_CASES if cases is None else cases
    return hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _card_problems(card: Dict[str, Any]) -> List[str]:
    probs = [f"dataset card: missing or empty '{k}'" for k in ("purpose", "schema", "split_methodology", "leakage_prevention", "case_generation_methodology", "ground_truth_construction",
                                                                  "reproducibility", "evaluation_protocol", "not_measured_meaning", "known_limitations") if not card.get(k)]
    if len(card.get("categories") or []) != EXPECTED_CATEGORY_COUNT or any(not (c.get("id") and c.get("name") and c.get("definition")) for c in card.get("categories") or []):
        probs.append("dataset card: must document all 15 categories with id, name and definition")
    if not any("rule-only" in str(x) and str(EXPECTED_RULE_ONLY_COUNTERFACTUALS) in str(x) for x in card.get("known_limitations") or []):
        probs.append("dataset card: known_limitations must document the rule-only counterfactual cases")
    return probs


def _rule_only_case_ids(cases) -> List[str]:
    from . import omni_bench_eval as ev
    out = []
    for c in cases:
        sig, _ = ev._cf_signature(c["counterfactual_correction"], {d["doc_id"] for d in c["document_set"]})
        if sig[0] and not sig[1] and not sig[2]: out.append(c["case_id"])  # rule id present, no decision label and no doc anchor
    return sorted(out)


def _source_problems() -> List[str]:
    probs = []
    here = pathlib.Path(__file__).resolve().parent
    for name in _BENCH_MODULES:
        path = here / name
        if not path.exists(): continue
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            probs.extend(f"{name}: forbidden import '{m}' (offline/deterministic modules only)" for m in mods if m.split(".")[0] in _FORBIDDEN_IMPORTS)
        probs.extend(f"{name}: wall-clock call '{w}'" for w in ("datetime.now", "date.today", "utcnow") if w in src.replace(f'"{w}"', ""))
    return probs


def _perfect_fixture(cases) -> List[Dict[str, Any]]:
    """Predictions copied from the labels, ONLY to exercise the evaluation contract; the result is never reported as performance."""
    return [{"case_id": c["case_id"], "predicted_decision": c["expected_decision"], "predicted_escalation": c["expected_escalation"], "predicted_applicable_policy_rule": c["applicable_policy_rule"],
             "predicted_supporting_evidence": list(c["supporting_evidence"]), "predicted_contradicting_evidence": list(c["contradicting_evidence"]),
             "predicted_missing_evidence": list(c["missing_evidence"]), "predicted_counterfactual_correction": c["counterfactual_correction"]} for c in cases]


def validate_omni_bench_release() -> Dict[str, Any]:
    """Deterministic structural report: {status: PASS|FAIL, checks: {name: bool}, counts, problems}. Contains structural counts only, never benchmark performance scores."""
    from . import tasks, omni_bench_eval as ev
    cases = tasks.OMNI_BENCH_CASES
    snapshot = copy.deepcopy(cases)
    probs: List[str] = []
    checks: Dict[str, bool] = {}

    def chk(name: str, ok: bool, msg: str = ""):
        checks[name] = bool(ok)
        if not ok: probs.append(msg or name)

    split_counts = {s: sum(c["split"] == s for c in cases) for s in tasks.OMNI_BENCH_SPLITS}
    cat_counts = {k: sum(c["difficulty_category"] == k for c in cases) for k in tasks.OMNI_BENCH_CATEGORY_IDS}
    fam_splits: Dict[str, set] = {}
    content_splits: Dict[str, set] = {}
    for c in cases:
        fam_splits.setdefault(c["family_id"], set()).add(c["split"])
        content_splits.setdefault(hashlib.sha256(json.dumps([c["document_set"], c["policy"]], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(), set()).add(c["split"])
    chk("case_count_128", len(cases) == EXPECTED_CASE_COUNT, f"expected {EXPECTED_CASE_COUNT} cases, found {len(cases)}")
    chk("split_counts_64_32_32", split_counts == EXPECTED_SPLIT_COUNTS, f"split counts {split_counts} != {EXPECTED_SPLIT_COUNTS}")
    chk("all_15_categories_represented", len(tasks.OMNI_BENCH_CATEGORY_IDS) == EXPECTED_CATEGORY_COUNT and all(n > 0 for n in cat_counts.values()), f"category counts: {cat_counts}")
    chk("unique_case_ids", len({c["case_id"] for c in cases}) == len(cases), "duplicate case_id")
    chk("family_ids_do_not_cross_splits", all(len(v) == 1 for v in fam_splits.values()), "a family_id appears in more than one split")
    chk("no_cross_split_content_leakage", all(len(v) == 1 for v in content_splits.values()), "identical document_set+policy content appears in more than one split")
    schema = tasks.validate_omni_bench_cases(cases) + tasks.validate_omni_bench_metadata()
    chk("schema_valid", not schema, f"schema problems: {schema[:5]}")
    a, b = tasks.build_omni_bench_cases(), tasks.build_omni_bench_cases()
    chk("dataset_generation_deterministic", a == b == cases and a is not b and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), "repeated generation differs from the loaded benchmark")
    chk("dataset_fingerprint_matches", omni_bench_cases_fingerprint(cases) == OMNI_BENCH_CASES_SHA256, "case fingerprint changed: benchmark cases or labels were modified")
    fixture = _perfect_fixture(cases)
    r1, r2 = ev.evaluate_omni_bench(fixture, cases=cases), ev.evaluate_omni_bench(copy.deepcopy(fixture), cases=cases)
    chk("evaluation_deterministic", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True) and r1["status"] == "MEASURED" and r1["case_count"] == len(cases), "repeated evaluation differs or did not cover every case")
    chk("not_measured_without_predictions", ev.evaluate_omni_bench(None, cases=cases)["status"] == "NOT_MEASURED" and ev.evaluate_omni_bench([], cases=cases)["status"] == "NOT_MEASURED", "no predictions must be NOT_MEASURED")
    chk("prediction_container_strict", ev.evaluate_omni_bench({"x": 1}, cases=cases)["status"] == "INVALID_PREDICTIONS", "a dict must be rejected as a prediction collection")
    rule_only = _rule_only_case_ids(cases)
    chk("rule_only_limitation_documented", len(rule_only) == EXPECTED_RULE_ONLY_COUNTERFACTUALS, f"{len(rule_only)} rule-only counterfactual cases, documented {EXPECTED_RULE_ONLY_COUNTERFACTUALS}")
    chk("dataset_card_complete", not _card_problems(get_omni_bench_dataset_card()), "; ".join(_card_problems(get_omni_bench_dataset_card())))
    src = _source_problems()
    chk("no_llm_network_or_randomness", not src, "; ".join(src))
    chk("benchmark_not_mutated_by_validation", cases == snapshot, "validation mutated the benchmark")
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "counts": {"cases": len(cases), "splits": split_counts, "categories": cat_counts, "families": len(fam_splits), "rule_only_counterfactual_cases": len(rule_only)},
            "rule_only_counterfactual_case_ids": rule_only, "cases_sha256": omni_bench_cases_fingerprint(cases), "problems": sorted(probs),
            "note": "structural validation only; no benchmark performance is measured or reported"}