"""Prompt 12 / Phase 3, mandatory first step: READ-ONLY one-record trace of the real V3 execution path.

Run from C:\\Users\\DELL\\Downloads\\OMNICheck-deep\\backend :

    python -m app.services.module1_compliance.omni_bench_trace_one_record --record-id ADV-01-OB-01-1A --out prompt12_trace

What it does (nothing is invented; an unavailable field is reported as "UNAVAILABLE: <reason>"):
  1. loads the frozen generator record and prints its documents + rulebook text;
  2. applies the exact V3 configuration the adapter uses (V3_COMPONENT_FLAGS, adapter _ACTIVE_CONFIG, cached_compiler_responses,
     compiled_policy_before_rules, _build_graph) to ONE record, then reads the resulting graph;
  3. dumps Policy.compiled_policy, every PolicyRule.compiled_rules entry, graph_to_rule_inputs(G), every Decision node's
     legacy/compiled fields, compile status / evaluation errors, compiler prompt + response hashes;
  4. runs adapter.run_version([record], tasks, "V3") (compile response comes from the cache, so no second provider call)
     and reports the final prediction and its provenance;
  5. optionally compares compiler prompt hashes with a pilot experiment_results.json (--pilot-json).

It never edits tasks.py / policy_compiler.py / policy_schema.py / rule_engine.py / router.py, never prints environment variables or
credentials (every string is passed through tasks.redact_secrets when available), and restores all flags/monkeypatches in finally.
It makes at most ONE compile call to the configured provider chain (Groq -> Gemini -> Ollama), exactly as the pilot did.
"""
import argparse
import contextlib
import copy
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

try:
    from . import omni_bench_system_adapters as adapter
    from . import omni_bench_experiment as experiment
except ImportError:  # flat-directory use
    import omni_bench_system_adapters as adapter
    import omni_bench_experiment as experiment

UNAVAILABLE = "UNAVAILABLE"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(obj: Any, redact) -> Any:
    """JSON-safe copy; every string is secret-redacted; sets/datetimes/unknown objects are stringified."""
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v, redact) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, redact) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_plain(v, redact) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    return redact(str(obj))


def _get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _rule_entries(G: Any) -> List[Dict[str, Any]]:
    rows = []
    for node_id, data in sorted(G.nodes(data=True), key=lambda kv: str(kv[0])):
        if data.get("type") != "PolicyRule":
            continue
        entries = []
        for e in data.get("compiled_rules") or []:
            if not isinstance(e, dict):
                continue
            res = e.get("result") if isinstance(e.get("result"), dict) else {}
            entries.append({
                "compiled_rule_id": e.get("rule_id"), "status": e.get("status"), "rule_type": e.get("rule_type"),
                "severity": e.get("severity"), "confidence": e.get("confidence"), "expression": e.get("expression"),
                "source_text": e.get("source_text"), "source_span": e.get("source_span"),
                "mapping_basis": e.get("mapping_basis"), "ambiguities": e.get("ambiguities"), "issues": e.get("issues"),
                "compiled_rule_entity": _get(e.get("compiled_rule"), "entity"),
                "compiled_rule_required_evidence": _get(e.get("compiled_rule"), "required_evidence"),
                "compiled_rule_exception": _get(e.get("compiled_rule"), "exception"),
                "result": {k: res.get(k) for k in ("executed", "verdict", "counts", "missing_facts", "reasons", "exception_applied",
                                                    "action_required", "severity", "violating_node_ids", "satisfying_node_ids", "record_results")}})
        rows.append({"policy_rule_node_id": str(node_id), "legacy_condition": data.get("condition"), "original_text": data.get("original_text"),
                     "legacy_parsed": data.get("legacy_parsed"), "compiled_authoritative": data.get("compiled_authoritative"),
                     "engine_run": data.get("engine_run"), "executable": data.get("executable"), "evaluated": data.get("evaluated"),
                     "unevaluated_reason": data.get("unevaluated_reason"), "compiled_rules": entries})
    return rows


def _decision_rows(G: Any) -> List[Dict[str, Any]]:
    keys = ("verdict", "rule_id", "evaluation_engine", "result_source", "legacy_verdict", "legacy_recognized", "legacy_unevaluated_reason", "compiled_verdict",
            "compiled_missing_facts", "compiled_verdict_downgraded", "legacy_disagrees", "violation_status", "supporting_evidence_ids", "contradicting_evidence_ids")
    out = []
    for node_id, data in sorted(G.nodes(data=True), key=lambda kv: str(kv[0])):
        if data.get("type") == "Decision":
            row = {"decision_node_id": str(node_id)}
            row.update({k: (data[k] if k in data else UNAVAILABLE + ": key absent on node") for k in keys})
            row["compiled_rules_summary"] = data.get("compiled_rules")
            row["rationale"] = data.get("rationale")
            out.append(row)
    return out


def _policy_rows(G: Any) -> List[Dict[str, Any]]:
    out = []
    for node_id, data in sorted(G.nodes(data=True), key=lambda kv: str(kv[0])):
        if data.get("type") == "Policy":
            out.append({"policy_node_id": str(node_id), "name": data.get("name"), "source_file": data.get("source_file"),
                        "compiled_policy": data.get("compiled_policy", UNAVAILABLE + ": Policy node has no compiled_policy attribute")})
    return out


def _inputs_view(tasks: Any, G: Any) -> Any:
    fn = getattr(tasks, "graph_to_rule_inputs", None)
    if not callable(fn):
        return UNAVAILABLE + ": tasks.graph_to_rule_inputs not found"
    try:
        return fn(G)
    except Exception as e:  # noqa: BLE001
        return f"{UNAVAILABLE}: graph_to_rule_inputs raised {type(e).__name__}: {e}"


def _compiler_hashes(tasks: Any, rulebook_text: str) -> Dict[str, Any]:
    """Prompt hashes recomputed exactly as apply_compiled_policy builds them (redact_pii -> compile_policy prompts)."""
    try:
        compile_fn = getattr(tasks, "_compile_policy")
        mod = sys.modules[compile_fn.__module__]
        text = tasks.redact_pii(rulebook_text)
        system, user = mod.SYSTEM_PROMPT, mod._build_user_prompt(text)
        return {"system_prompt_sha256": _sha(system), "user_prompt_sha256": _sha(user), "cache_key_sha256": _sha(str(system) + "\0" + str(user)),
                "redacted_rulebook_text_sha256": _sha(text), "min_confidence": getattr(tasks, "_COMPILER_MIN_CONFIDENCE", UNAVAILABLE)}
    except Exception as e:  # noqa: BLE001
        return {"status": f"{UNAVAILABLE}: {type(e).__name__}: {e}"}


def _pilot_compiler_entries(path: Optional[str], version: str = "V3") -> Any:
    if not path:
        return UNAVAILABLE + ": --pilot-json not given"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        preds = (d.get("raw_results") or {}).get(version, {}).get("predictions") or []
        return [{"perturbation_id": p.get("perturbation_id"), "compiler_cache": (p.get("provenance") or {}).get("compiler_cache")} for p in preds]
    except Exception as e:  # noqa: BLE001
        return f"{UNAVAILABLE}: could not read {path}: {type(e).__name__}: {e}"


def trace(record_id: str, tasks: Any, pilot_json: Optional[str]) -> Dict[str, Any]:
    redact = getattr(tasks, "redact_secrets", None) or (lambda s: s)
    records = experiment._load_experiment_records()
    record = next((r for r in records if r.get("perturbation_id") == record_id), None)
    if record is None:
        raise SystemExit(f"record {record_id!r} not found among {len(records)} generated records")
    out: Dict[str, Any] = {"record_id": record_id, "trace_kind": "read-only; real V3 path (adapter._build_graph + compiled_policy_before_rules)"}
    out["record"] = {"perturbation_type": record.get("perturbation_type"), "source_case_id": record.get("source_case_id"),
                     "expected_decision (label, read-only context)": record.get("expected_decision"),
                     "documents": record.get("perturbed_document_set"), "rulebook_text": record.get("perturbed_policy")}
    flags = adapter.V3_COMPONENT_FLAGS
    snap = adapter.snapshot_config(tasks, flags)
    compile_log: List[Dict[str, Any]] = []
    cache_start = len(adapter._COMPILER_CACHE_LOG)
    try:
        adapter._ACTIVE_CONFIG["version"] = "V3"
        for k, v in flags.items():
            if hasattr(tasks, k):
                setattr(tasks, k, v)
        out["flags_applied"] = {k: getattr(tasks, k, UNAVAILABLE + ": attribute absent") for k in flags}
        out["compiled_policy_module_flags"] = {"COMPILED_POLICY_ENABLED": getattr(tasks, "COMPILED_POLICY_ENABLED", UNAVAILABLE),
                                               "HAS_POLICY_COMPILER": getattr(tasks, "HAS_POLICY_COMPILER", UNAVAILABLE),
                                               "COMPILED_POLICY_STRICT_UNITS": getattr(tasks, "COMPILED_POLICY_STRICT_UNITS", UNAVAILABLE)}
        out["compiler_prompt_hashes"] = _compiler_hashes(tasks, record["perturbed_policy"])
        with adapter.cached_compiler_responses(tasks) as cache_state:
            out["compiler_cache_hook"] = cache_state
            with tempfile.TemporaryDirectory(prefix="omnibench_trace_") as wd:
                paths, rb, names = adapter.materialize_record(record, wd)
                with adapter.compiled_policy_before_rules(tasks, tasks._rulebook_text_only(rb), compile_log):
                    G = adapter._build_graph(record, tasks, wd, compiled_policy=False)
                out["compiled_policy_call_log"] = compile_log
                out["policy_nodes"] = _policy_rows(G)
                out["policy_rule_nodes"] = _rule_entries(G)
                out["graph_to_rule_inputs"] = _inputs_view(tasks, G)
                out["decision_nodes"] = _decision_rows(G)
                out["graph_component_proof (adapter._graph_component_proof)"] = adapter._graph_component_proof(G)
        out["compiler_cache_log_this_trace"] = adapter.compiler_cache_summary(cache_start)
        first_rows = out["compiler_cache_log_this_trace"].get("entries") or []
        first_response_cacheable = bool(first_rows) and all(
            isinstance(row, dict) and row.get("status") in ("MISS", "HIT") and row.get("response_sha256")
            for row in first_rows
        )
        if first_response_cacheable:
            run = adapter.run_version([record], tasks, "V3")  # permitted only after first response is proven cacheable
            preds = run.get("predictions") or []
            out["run_version_V3"] = {"status": "RAN_FROM_CACHED_RESPONSE", "unanswered": run.get("unanswered"), "blocker": run.get("blocker"), "component_state": run.get("component_state"),
                                     "prediction_count": len(preds),
                                     "final_prediction": ({k: v for k, v in preds[0].items() if k != "provenance"} if preds else UNAVAILABLE + ": no prediction"),
                                     "final_prediction_provenance": (preds[0].get("provenance") if preds else UNAVAILABLE + ": no prediction")}
        else:
            out["run_version_V3"] = {"status": "SKIPPED_TO_PREVENT_SECOND_PROVIDER_CALL",
                                     "reason": "first compiler response was absent or not proven cacheable; final run_version prediction omitted to enforce at-most-one provider call"}
        out["compiler_cache_log_after_run_version"] = adapter.compiler_cache_summary(cache_start)
    finally:
        adapter.restore_config(snap, tasks)
    out["pilot_compiler_cache_entries (for hash comparison)"] = _pilot_compiler_entries(pilot_json)
    out["notes"] = ["Fields reported as UNAVAILABLE were not present on the real objects; none were invented.",
                    "Compare compiler_cache_log_this_trace[*].prompt_sha256 with the pilot entries; equal prompt hashes mean the same compile request.",
                    "response_sha256 equality across calls shows whether the provider returned identical text; a MISS means a fresh provider call."]
    return _plain(out, redact)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only one-record V3 trace")
    ap.add_argument("--record-id", default="ADV-01-OB-01-1A")
    ap.add_argument("--out", default="prompt12_trace")
    ap.add_argument("--pilot-json", default=None, help="path to the pilot experiment_results.json for compiler-hash comparison")
    args = ap.parse_args(argv)
    errs: List[str] = []
    tasks_module = None
    for how in ("package", "flat"):
        try:
            if how == "package":
                from . import tasks as tasks_module
            else:
                import tasks as tasks_module
            break
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{how} import: {type(exc).__name__}: {exc}")
    if tasks_module is None:
        print("BLOCKER: cannot import tasks.py: " + " | ".join(errs), file=sys.stderr)
        return 2
    result = trace(args.record_id, tasks_module, args.pilot_json)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"trace_{args.record_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nTrace written to {os.path.abspath(path)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())