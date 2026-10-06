"""
rule_engine.py  (NEW FILE)
Location: backend/app/services/module1_compliance/rule_engine.py

DETERMINISTIC executor for CompiledRule objects. No LLM, no network, no randomness.
The compliance decision is produced here, by evaluating structured rules against facts + evidence.

Inputs
  facts     {"<entity>": {"<field>": value | {"value": v, "unit": u}, ...} | [records...], ...}   (any domain)
  evidence  [{"type": "<evidence_type>", ...attrs}, ...]  or None when evidence was not supplied

Logic is three-valued (Kleene): True / False / None(unknown). A missing fact, unparseable value or
non-convertible unit gives UNKNOWN, which propagates to INDETERMINATE: never a guessed pass or fail.
"""
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .policy_schema import (
    CompiledRule, Condition, Logic, Operator, RuleType, Temporal,
    LIST_OPS, ORDERING_OPS, RANGE_OPS,
    add_delta, as_number, is_date_string, parse_datetime, render_condition, render_temporal,
)

DECISION_SOURCE = "deterministic_rule_engine"

# ----------------------------------------------------------------------------- units
_ALIASES = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "£": "GBP", "gbp": "GBP",
    "%": "percent", "percent": "percent", "pct": "percent",
    "second": "seconds", "sec": "seconds", "secs": "seconds", "minute": "minutes", "min": "minutes", "mins": "minutes",
    "hour": "hours", "hr": "hours", "hrs": "hours", "day": "days", "week": "weeks", "wk": "weeks",
    "month": "months", "year": "years", "yr": "years", "yrs": "years",
    "byte": "b", "bytes": "b", "kilobyte": "kb", "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb",
    "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "meter": "m", "metre": "m", "meters": "m", "metres": "m", "kilometer": "km", "kilometre": "km", "kilometers": "km",
}
_CURRENCIES = {"INR", "USD", "EUR", "GBP", "AED", "JPY", "CAD", "AUD", "SGD", "CNY", "CHF"}
_FACTORS = {
    "time": {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800, "months": 2592000, "years": 31536000},  # month=30d, year=365d (approximation for DURATIONS only)
    "data": {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4},
    "mass": {"mg": 1e-3, "g": 1, "kg": 1000, "lb": 453.59237, "oz": 28.349523},
    "length": {"mm": 1e-3, "cm": 1e-2, "m": 1, "km": 1000},
}
_UNIT_DIM = {u: d for d, t in _FACTORS.items() for u in t}


def norm_unit(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    s = str(u).strip()
    k = s.lower()
    if k in _ALIASES:
        return _ALIASES[k]
    if s.upper() in _CURRENCIES:
        return s.upper()
    return k


def _dim(u: str) -> str:
    if u in _CURRENCIES:
        return "currency"
    if u == "percent":
        return "percent"
    return _UNIT_DIM.get(u, "other")


def convert(value: float, src: str, dst: str, fx: Dict[str, float]) -> Tuple[Optional[float], Optional[str]]:
    if src == dst:
        return value, None
    ds, dd = _dim(src), _dim(dst)
    if ds == "currency" and dd == "currency":
        if f"{src}->{dst}" in fx:
            return value * fx[f"{src}->{dst}"], None
        if f"{dst}->{src}" in fx and fx[f"{dst}->{src}"]:
            return value / fx[f"{dst}->{src}"], None
        return None, f"no fx rate supplied for {src}->{dst}"
    if ds == dd and ds in _FACTORS:
        return value * _FACTORS[ds][src] / _FACTORS[ds][dst], None
    return None, f"cannot convert unit '{src}' to '{dst}'"


_QTY = re.compile(r"^\s*(₹|rs\.?|inr|\$|usd|€|eur|£|gbp)?\s*(-?[\d,]+(?:\.\d+)?)\s*(%|[A-Za-z]+\.?)?\s*$", re.I)


def parse_quantity(raw: Any) -> Optional[Tuple[float, Optional[str]]]:
    """-> (number, unit|None) from a number, '₹12,000', '30 days', or {'value':..,'unit':..}."""
    if isinstance(raw, dict) and "value" in raw:
        q = parse_quantity(raw.get("value"))
        if q is None:
            return None
        return q[0], norm_unit(raw.get("unit") or raw.get("currency")) or q[1]
    n = as_number(raw) if not isinstance(raw, str) else None
    if n is not None:
        return n, None
    if isinstance(raw, str):
        m = _QTY.match(raw)
        if not m:
            return None
        num = as_number(m.group(2))
        if num is None:
            return None
        return num, norm_unit(m.group(1) or m.group(3))
    return None


# ----------------------------------------------------------------------------- Kleene logic
def k_and(vals: List[Optional[bool]]) -> Optional[bool]:
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


def k_or(vals: List[Optional[bool]]) -> Optional[bool]:
    if any(v is True for v in vals):
        return True
    return None if any(v is None for v in vals) else False


def k_not(v: Optional[bool]) -> Optional[bool]:
    return None if v is None else (not v)


# ----------------------------------------------------------------------------- context / fact access
class _Ctx:
    def __init__(self, now: datetime, fx: Dict[str, float], strict_units: bool):
        self.now, self.fx, self.strict_units = now, fx, strict_units
        self.missing: set = set()
        self.notes: List[str] = []


def _lookup_ci(d: Dict[str, Any], key: str) -> Tuple[bool, Any]:
    if key in d:
        return True, d[key]
    lk = key.lower()
    for k, v in d.items():
        if str(k).lower() == lk:
            return True, v
    return False, None


def _resolve(facts: Dict[str, Any], entity: str, field: str) -> Tuple[str, Any]:
    """-> ('ok'|'missing'|'list', value)."""
    ok, cur = _lookup_ci(facts, entity)
    if not ok:
        return "missing", None
    if isinstance(cur, (list, tuple)):
        return "list", None
    for part in field.split("."):
        if isinstance(cur, dict) and not ("value" in cur and part == "value"):
            ok, cur = _lookup_ci(cur, part)
            if not ok:
                return "missing", None
        else:
            return "missing", None
    return "ok", cur


def _present(status: str, raw: Any) -> bool:
    return status == "ok" and raw is not None and not (isinstance(raw, str) and not raw.strip())


def _safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_safe(x) for x in v]
    return str(v)


def _norm_str(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().casefold())


def _to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y"):
            return True
        if s in ("false", "no", "n"):
            return False
    return None


# ----------------------------------------------------------------------------- leaf evaluation
def _number_in_rule_unit(raw: Any, rule_unit: Optional[str], ctx: _Ctx, path: str) -> Tuple[Optional[float], Optional[str]]:
    q = parse_quantity(raw)
    if q is None:
        return None, f"fact '{path}' value {raw!r} is not numeric"
    num, fu = q
    ru = norm_unit(rule_unit)
    if ru and fu and fu != ru:
        conv, err = convert(num, fu, ru, ctx.fx)
        if conv is None:
            return None, err
        return conv, None
    if ru and not fu:
        if ctx.strict_units:
            return None, f"fact '{path}' has no unit but the rule is in '{ru}' (strict_units)"
        ctx.notes.append(f"fact '{path}' has no unit; assumed '{ru}'")
    return num, None


def _eq(raw: Any, rule_val: Any, rule_unit: Optional[str], ctx: _Ctx, path: str) -> Tuple[Optional[bool], Optional[str]]:
    if isinstance(rule_val, bool):
        b = _to_bool(raw)
        return (None, f"fact '{path}' is not boolean") if b is None else (b == rule_val, None)
    rn = as_number(rule_val)
    if rn is not None and not isinstance(rule_val, str):
        n, err = _number_in_rule_unit(raw, rule_unit, ctx, path)
        return (None, err) if n is None else (math.isclose(n, rn, rel_tol=1e-9, abs_tol=1e-9), None)
    if isinstance(rule_val, str) and is_date_string(rule_val):
        a, b = parse_datetime(raw), parse_datetime(rule_val)
        if a is not None and b is not None:
            return a == b, None
    if rn is not None and as_number(raw) is not None:  # numeric-looking string on both sides
        return math.isclose(as_number(raw), rn, rel_tol=1e-9, abs_tol=1e-9), None
    return _norm_str(raw) == _norm_str(rule_val), None


def _eval_leaf(c: Condition, facts: Dict[str, Any], ctx: _Ctx) -> Tuple[Optional[bool], Dict[str, Any]]:
    path = f"{c.entity}.{c.field}"
    tr: Dict[str, Any] = {"type": "condition", "expr": render_condition(c), "fact": path}
    status, raw = _resolve(facts, c.entity, c.field)
    present = _present(status, raw)
    op = c.operator

    def done(res: Optional[bool], reason: Optional[str] = None):
        tr["result"] = res
        if reason:
            tr["reason"] = reason
        return res, tr

    if op == Operator.EXISTS:
        tr["actual"] = _safe(raw) if present else None
        return done(present)
    if op == Operator.NOT_EXISTS:
        tr["actual"] = _safe(raw) if present else None
        return done(not present)
    if status == "list":
        return done(None, f"'{c.entity}' is a list of records; evaluate one record at a time")
    if not present:
        ctx.missing.add(path)
        return done(None, f"fact '{path}' is missing")
    tr["actual"] = _safe(raw)

    if op in ORDERING_OPS or op in RANGE_OPS:
        rv = c.value[0] if op in RANGE_OPS else c.value
        if isinstance(rv, str):  # date mode
            a = parse_datetime(raw)
            if a is None:
                return done(None, f"fact '{path}' value {raw!r} is not a recognisable date")
            if op in RANGE_OPS:
                lo, hi = parse_datetime(c.value[0]), parse_datetime(c.value[1])
                res = (lo <= a <= hi) if c.inclusive else (lo < a < hi)
                return done(res if op == Operator.BETWEEN else not res)
            b = parse_datetime(c.value)
            return done({Operator.GT: a > b, Operator.GTE: a >= b, Operator.LT: a < b, Operator.LTE: a <= b}[op])
        x, err = _number_in_rule_unit(raw, c.unit, ctx, path)
        if x is None:
            return done(None, err)
        if op in RANGE_OPS:
            lo, hi = c.value
            res = (lo <= x <= hi) if c.inclusive else (lo < x < hi)
            return done(res if op == Operator.BETWEEN else not res)
        v = c.value
        return done({Operator.GT: x > v, Operator.GTE: x >= v, Operator.LT: x < v, Operator.LTE: x <= v}[op])

    if op in (Operator.EQ, Operator.NEQ):
        res, err = _eq(raw, c.value, c.unit, ctx, path)
        if res is None:
            return done(None, err)
        return done(res if op == Operator.EQ else not res)

    if op in LIST_OPS:
        hit, errs = False, []
        for item in c.value:
            r, e = _eq(raw, item, c.unit, ctx, path)
            if r is True:
                hit = True
                break
            if r is None and e:
                errs.append(e)
        if not hit and errs and len(errs) == len(c.value):
            return done(None, errs[0])
        return done(hit if op == Operator.IN else not hit)

    # contains / not_contains
    if isinstance(raw, (list, tuple, set)):
        res = any(_norm_str(x) == _norm_str(c.value) for x in raw)
    elif isinstance(raw, str):
        res = _norm_str(c.value) in _norm_str(raw)
    else:
        return done(None, f"fact '{path}' is neither text nor a list")
    return done(res if op == Operator.CONTAINS else not res)


def _eval_condition(c: Condition, facts: Dict[str, Any], ctx: _Ctx) -> Tuple[Optional[bool], Dict[str, Any]]:
    if not c.children:
        return _eval_leaf(c, facts, ctx)
    pairs = [_eval_condition(ch, facts, ctx) for ch in c.children]
    vals = [p[0] for p in pairs]
    if c.logic == Logic.AND:
        res = k_and(vals)
    elif c.logic == Logic.OR:
        res = k_or(vals)
    elif c.logic == Logic.NOT:
        res = k_not(vals[0])
    else:  # IMPLIES
        res = k_or([k_not(vals[0]), vals[1]])
    return res, {"type": "group", "logic": c.logic.value, "result": res, "children": [p[1] for p in pairs]}


# ----------------------------------------------------------------------------- temporal
def _eval_temporal(t: Temporal, facts: Dict[str, Any], ctx: _Ctx) -> Tuple[Optional[bool], Dict[str, Any]]:
    path = f"{t.entity}.{t.field}"
    tr: Dict[str, Any] = {"type": "temporal", "expr": render_temporal(t), "fact": path}

    def done(res, reason=None):
        tr["result"] = res
        if reason:
            tr["reason"] = reason
        return res, tr

    status, raw = _resolve(facts, t.entity, t.field)
    if not _present(status, raw):
        ctx.missing.add(path)
        return done(None, f"date fact '{path}' is missing" if status != "list" else f"'{t.entity}' is a list of records")
    tr["actual"] = _safe(raw)
    dt = parse_datetime(raw)
    if dt is None:
        return done(None, f"fact '{path}' value {raw!r} is not a recognisable date (dd/mm vs mm/dd is never guessed)")
    if t.kind == "between":
        return done(parse_datetime(t.start) <= dt <= parse_datetime(t.end))
    r = (t.reference or "").strip()
    if r.lower() in ("now", "today", "evaluation_date"):
        ref = ctx.now
    else:
        ref = parse_datetime(r)
        if ref is None and re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z0-9_.]+", r.lower()):
            ent, _, fld = r.lower().partition(".")
            rs, rraw = _resolve(facts, ent, fld)
            if not _present(rs, rraw):
                ctx.missing.add(r)
                return done(None, f"reference date fact '{r}' is missing")
            ref = parse_datetime(rraw)
    if ref is None:
        return done(None, f"reference '{r}' could not be resolved to a date")
    tr["reference_resolved"] = ref.isoformat()
    if t.kind == "before":
        return done(dt < ref)
    if t.kind == "after":
        return done(dt > ref)
    if t.kind == "older_than":
        return done(dt < add_delta(ref, t.amount, t.unit, -1))
    if t.direction == "after":
        return done(ref <= dt <= add_delta(ref, t.amount, t.unit, 1))
    if t.direction == "before":
        return done(add_delta(ref, t.amount, t.unit, -1) <= dt <= ref)
    return done(add_delta(ref, t.amount, t.unit, -1) <= dt <= add_delta(ref, t.amount, t.unit, 1))


# ----------------------------------------------------------------------------- evidence
def _norm_type(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _evidence_items(evidence: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    if evidence is None:
        return None
    out = []
    for e in evidence:
        if isinstance(e, str):
            out.append({"type": e})
        elif isinstance(e, dict):
            t = e.get("type", e.get("kind"))
            if t:
                out.append({**e, "type": t})
    return out


def _check_evidence(rule: CompiledRule, items: Optional[List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """-> (status OK|MISSING|NOT_SUPPLIED, missing mandatory requirements, satisfied types)."""
    if not rule.required_evidence:
        return "OK", [], []
    mandatory = [r for r in rule.required_evidence if r.mandatory]
    if items is None:
        return ("NOT_SUPPLIED" if mandatory else "OK"), [], []
    missing, found = [], []
    for req in rule.required_evidence:
        n = 0
        for it in items:
            if _norm_type(it["type"]) != req.type:
                continue
            if req.match and not all(k in it and _norm_str(it[k]) == _norm_str(v) for k, v in req.match.items()):
                continue
            n += 1
        if n >= req.min_count:
            found.append(req.type)
        elif req.mandatory:
            missing.append({"type": req.type, "required": req.min_count, "found": n, "description": req.description})
    return ("MISSING" if missing else "OK"), missing, found


# ----------------------------------------------------------------------------- rule / policy evaluation
_SEV_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def evaluate_rule(rule: CompiledRule, facts: Dict[str, Any], evidence: Optional[List[Any]], ctx: _Ctx, include_needs_review: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {"policy_id": rule.policy_id, "rule_id": rule.rule_id, "expression": rule.expression, "rule_type": rule.rule_type.value,
                           "verdict": None, "severity": None, "action_required": None, "reasons": [], "missing_facts": [],
                           "missing_evidence": [], "evidence_satisfied": [], "exception_applied": None, "trace": {}}
    if rule.status != "VALID" and not include_needs_review:
        res.update(verdict="INDETERMINATE")
        res["reasons"].append("rule is NEEDS_REVIEW (ambiguous or low confidence) and was not executed: " + "; ".join(rule.ambiguities or ["see issues"]))
        return res

    ctx.missing, ctx.notes = set(), []
    parts: List[Optional[bool]] = []
    if rule.condition is not None:
        r, tr = _eval_condition(rule.condition, facts, ctx)
        parts.append(r)
        res["trace"]["condition"] = tr
    if rule.temporal is not None:
        r, tr = _eval_temporal(rule.temporal, facts, ctx)
        parts.append(r)
        res["trace"]["temporal"] = tr
    pred = k_and(parts) if parts else True
    res["trace"]["predicate"] = pred

    ev_status, missing_ev, satisfied = ("OK", [], [])
    engaged = rule.rule_type == RuleType.REQUIRE or (rule.rule_type == RuleType.TRIGGER and pred is True)
    if engaged:
        ev_status, missing_ev, satisfied = _check_evidence(rule, _evidence_items(evidence))
    res["missing_evidence"], res["evidence_satisfied"] = missing_ev, satisfied

    verdict: str
    if pred is None:
        verdict = "INDETERMINATE"
        res["reasons"].append("condition could not be evaluated (unknown facts); no pass/fail assigned")
    elif rule.rule_type == RuleType.PROHIBIT:
        verdict = "VIOLATION" if pred else "COMPLIANT"
    elif rule.rule_type == RuleType.REQUIRE:
        if not pred:
            verdict = "VIOLATION"
            res["reasons"].append("required condition does not hold")
        elif ev_status == "MISSING":
            verdict = "VIOLATION"
            res["reasons"].append("mandatory evidence is missing")
        elif ev_status == "NOT_SUPPLIED":
            verdict = "INDETERMINATE"
            res["reasons"].append("rule requires evidence but no evidence list was supplied")
        else:
            verdict = "COMPLIANT"
    else:  # TRIGGER
        if not pred:
            verdict = "NOT_APPLICABLE"
            res["reasons"].append("trigger condition not met")
        elif ev_status == "MISSING":
            verdict = "VIOLATION"
            res["reasons"].append(f"trigger fired; action '{rule.action}' not evidenced: mandatory evidence missing")
        elif ev_status == "NOT_SUPPLIED":
            verdict = "INDETERMINATE"
            res["reasons"].append("trigger fired but no evidence list was supplied")
        elif rule.required_evidence:
            verdict = "COMPLIANT"
            res["reasons"].append(f"trigger fired; required evidence present for action '{rule.action}'")
        else:
            verdict = "ACTION_REQUIRED"
            res["reasons"].append(f"trigger fired; action '{rule.action}' is required")

    if verdict in ("VIOLATION", "ACTION_REQUIRED") and rule.exception:
        ex_res, ex_traces = [], []
        for ex in rule.exception:
            r, tr = _eval_condition(ex.condition, facts, ctx)
            ex_res.append(r)
            ex_traces.append({"description": ex.description, "result": r, "trace": tr})
        res["trace"]["exceptions"] = ex_traces
        if any(r is True for r in ex_res):
            hit = next(e["description"] for e, r in zip(ex_traces, ex_res) if r is True)
            verdict, res["exception_applied"] = "EXEMPT", hit
            res["reasons"].append(f"exception applies: {hit}")
        elif any(r is None for r in ex_res):
            verdict = "INDETERMINATE"
            res["reasons"].append("an exception could not be evaluated (unknown facts); cannot confirm the rule applies")

    res["verdict"] = verdict
    if verdict in ("VIOLATION", "ACTION_REQUIRED"):
        res["severity"] = rule.severity.value
        res["action_required"] = rule.action
    res["missing_facts"] = sorted(ctx.missing)
    if ctx.notes:
        res["notes"] = sorted(set(ctx.notes))
    return res


_RANK = ["VIOLATION", "ACTION_REQUIRED", "INDETERMINATE", "COMPLIANT"]


def _overall(results: List[Dict[str, Any]]) -> str:
    vs = {r["verdict"] for r in results}
    if "VIOLATION" in vs:
        return "NON_COMPLIANT"
    if "ACTION_REQUIRED" in vs:
        return "ACTION_REQUIRED"
    if "INDETERMINATE" in vs:
        return "INDETERMINATE"
    if "COMPLIANT" in vs:
        return "COMPLIANT"
    return "NOT_APPLICABLE"


def evaluate_policy(rules: List[Any], facts: Dict[str, Any], evidence: Optional[List[Any]] = None, evaluation_date: Optional[str] = None,
                    include_needs_review: bool = False, fx_rates: Optional[Dict[str, float]] = None, strict_units: bool = False) -> Dict[str, Any]:
    """Evaluate compiled rules against facts + evidence. If facts[rule.entity] is a list, the rule runs once per record."""
    if not isinstance(facts, dict):
        raise ValueError("facts must be an object {entity: {...} | [...]}")
    now = parse_datetime(evaluation_date) if evaluation_date else datetime.utcnow()
    if now is None:
        raise ValueError(f"evaluation_date {evaluation_date!r} is not a recognisable date")
    fx = {}
    for k, v in (fx_rates or {}).items():
        n = as_number(v)
        if n is None or n <= 0:
            raise ValueError(f"fx rate {k!r} must be a positive number")
        a, _, b = str(k).replace(" ", "").partition("->")
        fx[f"{norm_unit(a)}->{norm_unit(b)}"] = n
    ctx = _Ctx(now, fx, strict_units)

    compiled = [r if isinstance(r, CompiledRule) else CompiledRule.model_validate(r) for r in rules]
    results: List[Dict[str, Any]] = []
    for rule in compiled:
        _, ent = _lookup_ci(facts, rule.entity or "")
        if isinstance(ent, (list, tuple)):
            if not ent:
                r = evaluate_rule(rule, {**facts, rule.entity: {}}, evidence, ctx, include_needs_review)
                r["record_index"] = None
                results.append(r)
            for i, rec in enumerate(ent):
                r = evaluate_rule(rule, {**facts, rule.entity: rec if isinstance(rec, dict) else {}}, evidence, ctx, include_needs_review)
                r["record_index"] = i
                results.append(r)
        else:
            results.append(evaluate_rule(rule, facts, evidence, ctx, include_needs_review))

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    viol = [r for r in results if r["verdict"] == "VIOLATION"]
    return {
        "decision": _overall(results),
        "decision_source": DECISION_SOURCE,
        "llm_used": False,
        "evaluation_date": now.isoformat(),
        "highest_severity": max((r["severity"] for r in viol), key=lambda s: _SEV_RANK.get(s, 0)) if viol else None,
        "actions_required": sorted({r["action_required"] for r in results if r["action_required"]}),
        "summary": counts,
        "rules_evaluated": len(compiled),
        "rule_results": results,
        "note": ("Decision derived by evaluating structured rules against the supplied facts and evidence. INDETERMINATE means required facts/evidence/units "
                 "were missing or the rule needs review: it is neither a pass nor a fail."),
    }