"""
policy_compiler.py  (NEW FILE)
Location: backend/app/services/module1_compliance/policy_compiler.py

Natural-language policy -> validated structured rules.

  Stage 1 (LLM INTERPRETATION): the model only translates text into JSON. It never judges compliance.
  Stage 2 (DETERMINISTIC VALIDATION): schema validation + grounding / ambiguity / consistency checks.
          A rule is VALID, NEEDS_REVIEW (ambiguous: never silently accepted) or REJECTED (malformed).
  Execution lives in rule_engine.py (no LLM).

Domain-neutral: entity / field / unit / action names come from the policy text itself.
"""
import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .policy_schema import (
    CompiledRule, CompileResult, Condition, Logic, Operator, RejectedRule, RuleType,
    NEGATED_OPS, RANGE_OPS, TIME_UNITS, as_number, has_not_group, iter_leaves,
)

logger = logging.getLogger(__name__)

MAX_POLICY_CHARS = 20000
DEFAULT_MIN_CONFIDENCE = 0.7
CONFIDENCE_CAP_WITH_ISSUES = 0.49
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class CompilationError(Exception):
    """LLM unavailable or returned unusable output (distinct from a policy that is merely ambiguous)."""


# ============================================================================ STAGE 1: LLM INTERPRETATION
SYSTEM_PROMPT = """You are a POLICY-TO-RULE COMPILER. Translate natural-language policy text from ANY domain (finance, HR, IT security, procurement, safety, data governance, ...) into structured JSON rules.
You only TRANSLATE. You never decide whether anything complies.

STRICT RULES
1. Output ONLY one JSON object, no prose, no markdown fences.
2. Use ONLY facts stated in the policy. Never invent thresholds, units, entities, exceptions or evidence.
3. If a clause is vague (no measurable threshold, unclear subject, advisory wording such as "should"/"may", words like "large", "reasonable", "promptly"), DO NOT guess a number. Put it in "unparsed_statements" with a reason.
4. Entity = singular snake_case noun the policy talks about (e.g. "invoice", "employee", "server", "dataset"). Field = snake_case attribute of that entity.
5. Numbers: write plain numbers. Convert "10,000" -> 10000, "5 lakh" -> 500000, "2 weeks" may stay 2 with unit "weeks". Copy the unit as written in the policy (currency code/symbol, days, %, GB, ...). Use null when the policy states no unit.
6. Keep the policy's own wording in "source_text" (an exact quote of the clause).
7. One rule per independent obligation. Preserve negation, ranges, AND/OR, exceptions and time limits.

OUTPUT SCHEMA
{"rules":[{
  "rule_type": "REQUIRE" | "PROHIBIT" | "TRIGGER",
     REQUIRE  = the condition/temporal must hold ("Passwords must be at least 12 characters")
     PROHIBIT = the condition must not hold ("Contractors must not access production data")
     TRIGGER  = when the condition holds, the action is required ("Contracts above X need legal review")
  "entity": "snake_case primary entity",
  "condition": <condition or null>,
  "temporal": null | {"kind":"within|older_than|before|after|between","entity":"..","field":"..(a date field)","reference":"now|ISO date|entity.field","amount":number,"unit":"minutes|hours|days|weeks|months|years","direction":"after|before|either","start":"ISO date","end":"ISO date"},
  "required_evidence": [{"type":"snake_case_evidence_type","description":"..","mandatory":true,"min_count":1}],
  "exception": [{"description":"..","condition": <condition>}],
  "severity": "low|medium|high|critical",
  "action": "snake_case_action (what must happen / the consequence)",
  "confidence": 0.0-1.0,
  "source_text": "exact clause quote",
  "ambiguities": ["anything uncertain about your interpretation"]
 }],
 "unparsed_statements":[{"text":"..","reason":".."}]}

CONDITION = a LEAF or a GROUP (never both)
 leaf : {"entity":"..","field":"..","operator":"> | >= | < | <= | == | != | in | not_in | between | not_between | contains | not_contains | exists | not_exists","value":..,"unit":"..|null"}
        between/not_between: "value":[low,high] (inclusive unless "inclusive":false). in/not_in: "value":[...]. exists/not_exists: no value.
 group: {"logic":"AND|OR|NOT|IMPLIES","children":[...]}   NOT has 1 child. IMPLIES has 2 children [IF, THEN] and encodes a conditional requirement.
Negation: use not_* operators, "!=", or a NOT group. Never drop a "not"/"no"/"never".
Exceptions ("unless", "except", "other than", "exempt"): put them in "exception", not in the main condition.
Time limits ("within 30 days of ...", "before ...", "older than ...", "between <date> and <date>"): use "temporal".
Mandatory documents/proof the policy demands ("with receipt", "signed approval", "attach ..."): use "required_evidence".

ILLUSTRATIVE EXAMPLES (different domains; do not copy their entities)
Text: "Servers storing customer data must not be reachable from the public internet."
-> {"rule_type":"PROHIBIT","entity":"server","condition":{"logic":"AND","children":[{"entity":"server","field":"stores_customer_data","operator":"==","value":true,"unit":null},{"entity":"server","field":"publicly_reachable","operator":"==","value":true,"unit":null}]},"temporal":null,"required_evidence":[],"exception":[],"severity":"high","action":"block_and_report","confidence":0.9,"source_text":"Servers storing customer data must not be reachable from the public internet.","ambiguities":[]}
Text: "Purchase orders above 50,000 USD require legal review unless issued under an approved framework agreement."
-> {"rule_type":"TRIGGER","entity":"purchase_order","condition":{"entity":"purchase_order","field":"total_value","operator":">","value":50000,"unit":"USD"},"temporal":null,"required_evidence":[{"type":"legal_review","mandatory":true,"min_count":1}],"exception":[{"description":"issued under an approved framework agreement","condition":{"entity":"purchase_order","field":"under_framework_agreement","operator":"==","value":true,"unit":null}}],"severity":"medium","action":"legal_review_required","confidence":0.9,"source_text":"Purchase orders above 50,000 USD require legal review unless issued under an approved framework agreement.","ambiguities":[]}
"""


def _build_user_prompt(policy_text: str) -> str:
    return f"POLICY TEXT:\n<policy>\n{policy_text.strip()}\n</policy>\nReturn the JSON object now."


def _call_llm(system: str, user: str) -> Optional[str]:
    """Same provider chain and privacy gates as tasks.analyze_compliance_with_matrix (Groq -> Gemini -> local Ollama);
    every outbound payload is passed through redact_pii. Returns raw model text or None."""
    from . import tasks as t  # lazy: keeps schema/engine importable without celery/networkx

    sys_p, usr_p = t.redact_pii(system), t.redact_pii(user)
    if t.EXTERNAL_AI_ENABLED:
        deadline = time.monotonic() + t.AI_EXTERNAL_BUDGET
        for key in t.GROQ_KEYS:
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                break
            try:
                resp = t.http_session.post("https://api.groq.com/openai/v1/chat/completions",
                                           headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                           json={"model": t.GROQ_MODEL, "temperature": 0,
                                                 "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]},
                                           timeout=min(remaining, 45))
                if resp.status_code == 200:
                    txt = t._extract_groq_text(resp.json())
                    if txt.strip():
                        return txt
                else:
                    t._log_provider_failure("Groq", resp)
            except Exception as e:
                logger.warning(f"Policy compiler Groq error: {e}")
        for key in t.GEMINI_KEYS:
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                break
            try:
                resp = t.http_session.post(f"https://generativelanguage.googleapis.com/v1beta/models/{t.GEMINI_MODEL}:generateContent",
                                           headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                           json={"system_instruction": {"parts": [{"text": sys_p}]},
                                                 "contents": [{"role": "user", "parts": [{"text": usr_p}]}],
                                                 "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
                                           timeout=min(remaining, 45))
                if resp.status_code == 200:
                    txt = t._extract_gemini_text(resp.json())
                    if txt.strip():
                        return txt
                else:
                    t._log_provider_failure("Gemini", resp)
            except Exception as e:
                logger.warning(f"Policy compiler Gemini error: {e}")
    try:
        resp = t.http_session.post(f"{t.OLLAMA_URL}/api/chat",
                                   json={"model": t.OLLAMA_PRO_MODEL, "stream": False, "options": {"temperature": 0},
                                         "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]},
                                   timeout=max(t.OLLAMA_TIMEOUT, 120))
        if resp.status_code == 200:
            txt = re.sub(r"<think>.*?(?:</think>|$)", "", resp.json().get("message", {}).get("content", ""), flags=re.DOTALL).strip()
            if txt:
                return txt
        else:
            t._log_provider_failure("Ollama", resp)
    except Exception as e:
        logger.error(f"Policy compiler Ollama error: {e}")
    return None


def _extract_json(text: str) -> Dict[str, Any]:
    s = re.sub(r"<think>.*?(?:</think>|$)", "", text or "", flags=re.DOTALL).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    for cand in (s, s[s.find("{"):s.rfind("}") + 1] if "{" in s else ""):
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except ValueError:
            continue
        if isinstance(data, list):
            return {"rules": data}
        if isinstance(data, dict):
            return data
    raise CompilationError("LLM output was not valid JSON")


# ============================================================================ STAGE 2: DETERMINISTIC VALIDATION
_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
              "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100, "thousand": 1000}
_MULT = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
         "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|k\b|thousand|lakhs?|lacs?|crores?|million|mn|billion|bn)?", re.I)
_TIME_RE = re.compile(r"\b(\d+(?:\.\d+)?|a|an|one)\s*-?\s*(minute|hour|day|week|month|year)s?\b", re.I)
_TIME_MIN = {"minute": 1, "hour": 60, "day": 1440, "week": 10080}
_HAS_NUMBER = re.compile(r"\d|\b(?:" + "|".join(_WORD_NUMS) + r")\b", re.I)


def numbers_in_text(text: str) -> List[float]:
    """Every number the policy states (digits with k/lakh/crore/million..., percent as 20 and 0.2, number words,
    and exact time-unit equivalents such as '2 weeks' -> 14 days), used to ground rule values."""
    out: List[float] = []
    for m in _NUM_RE.finditer(text or ""):
        n = as_number(m.group(1))
        if n is None:
            continue
        suf = (m.group(2) or "").lower()
        out.append(n)
        if suf == "%":
            out.append(n / 100.0)
        elif suf in _MULT:
            out.append(n * _MULT[suf])
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in _WORD_NUMS:
            out.append(float(_WORD_NUMS[w]))
    for m in _TIME_RE.finditer(text or ""):
        raw = m.group(1).lower()
        n = 1.0 if raw in ("a", "an", "one") else float(raw)
        unit = m.group(2).lower()
        out.append(n)
        if unit in _TIME_MIN:
            mins = n * _TIME_MIN[unit]
            out.extend(mins / f for f in _TIME_MIN.values())
    return out


def _rule_numbers(rule: CompiledRule) -> List[Tuple[str, float]]:
    vals: List[Tuple[str, float]] = []

    def leaf_vals(c: Optional[Condition], where: str):
        for lf in iter_leaves(c):
            raw = lf.value if isinstance(lf.value, list) else [lf.value]
            for x in raw:
                if isinstance(x, bool) or x is None or isinstance(x, str):
                    continue
                if as_number(x) is not None:
                    vals.append((f"{where}:{lf.entity}.{lf.field}", float(x)))

    leaf_vals(rule.condition, "condition")
    for ex in rule.exception:
        leaf_vals(ex.condition, "exception")
    if rule.temporal and rule.temporal.amount is not None:
        vals.append(("temporal.amount", float(rule.temporal.amount)))
    for ev in rule.required_evidence:
        if ev.min_count > 1:
            vals.append(("required_evidence.min_count", float(ev.min_count)))
    return vals


_VAGUE = re.compile(r"\b(large|small|big|significant(?:ly)?|substantial(?:ly)?|reasonabl[ey]|appropriate(?:ly)?|excessive(?:ly)?|adequate(?:ly)?|sufficient(?:ly)?|"
                    r"material(?:ly)?|prompt(?:ly)?|timely|soon|as soon as possible|asap|frequent(?:ly)?|high|low|unusual|suspicious|as needed|where possible|"
                    r"if (?:necessary|possible)|etc\.?|at (?:its|their|the manager'?s?) discretion|generally|normally|typically|usually|approximately|"
                    r"around|some|several|many|few|acceptable|proper(?:ly)?|good|bad)\b", re.I)
_ADVISORY = re.compile(r"\b(should|might|ideally|encouraged|recommended|advisable|preferably|may(?!\s+not))\b", re.I)
_ORDER_PHRASES = re.compile(r"\b(?:no|not)\s+(?:later|more|less|earlier|fewer|greater|longer|shorter|higher|lower)\s+than\b|\bnot\s+(?:to\s+)?exceed(?:ing)?\b|\bno\s+(?:more|less)\b", re.I)
_NEG_CUE = re.compile(r"\b(not|no|never|cannot|can't|cannot|without|prohibit(?:ed|s)?|forbid(?:den|s)?|neither|nor|disallow(?:ed|s)?|banned?)\b", re.I)
_EXC_CUE = re.compile(r"\b(unless|except(?:ion)?s?|other than|excluding|exempt(?:ed|s|ion)?|save for|apart from)\b", re.I)
_TEMP_CUE = re.compile(r"\b(within\s+(?:the\s+)?(?:\d+|a|an|one|two|three|four|five|six|seven|ten|fifteen|thirty|sixty|ninety)\s*-?\s*(?:minute|hour|day|week|month|year)s?|"
                       r"older than|no later than|not later than|later than|before\s+\d|after\s+\d|last\s+\d+\s*(?:minute|hour|day|week|month|year)s?|"
                       r"(?:minute|hour|day|week|month|year)s?\s+(?:of|from|after|before|since)\b)", re.I)
_RANGE_CUE = re.compile(r"\bbetween\b[^.;]{0,60}?\band\b|\bfrom\b[^.;]{0,40}?\bto\b[^.;]{0,40}?\d", re.I)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _sentences(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"(?<=[.;!?])\s+|\n+", text or "") if x and x.strip()]


def lexical_ambiguity_reasons(text: str) -> List[str]:
    """Policy-level scan: sentences that contain vague/advisory wording and state no measurable number."""
    out = []
    for s in _sentences(text):
        if _HAS_NUMBER.search(s):
            continue
        v = _VAGUE.search(s)
        a = _ADVISORY.search(s)
        if v:
            out.append(f"vague term '{v.group(0)}' with no measurable value: \"{s[:140]}\"")
        elif a:
            out.append(f"advisory wording '{a.group(0)}' (obligation strength unclear): \"{s[:140]}\"")
    return out


def _has_negation_construct(rule: CompiledRule) -> bool:
    if rule.rule_type == RuleType.PROHIBIT or has_not_group(rule.condition):
        return True
    return any(lf.operator in NEGATED_OPS or lf.operator == Operator.NEQ for lf in iter_leaves(rule.condition))


def _has_range_construct(rule: CompiledRule) -> bool:
    leaves = list(iter_leaves(rule.condition))
    if any(lf.operator in RANGE_OPS for lf in leaves):
        return True
    lows, highs = set(), set()
    for lf in leaves:
        key = (lf.entity, lf.field)
        if lf.operator in (Operator.GT, Operator.GTE):
            lows.add(key)
        if lf.operator in (Operator.LT, Operator.LTE):
            highs.add(key)
    return bool(lows & highs)


def semantic_crosschecks(rule: CompiledRule, clause: str) -> List[str]:
    """Deterministic consistency checks between the policy clause and the compiled structure.
    They catch silent LLM drops (negation, exception, time limit, range); findings send the rule to NEEDS_REVIEW."""
    out: List[str] = []
    stripped = _ORDER_PHRASES.sub(" ", clause)
    if _NEG_CUE.search(stripped) and not _has_negation_construct(rule):
        out.append(f"clause contains negation ('{_NEG_CUE.search(stripped).group(0)}') but the compiled rule has no negation (no PROHIBIT / NOT / negated operator)")
    if _EXC_CUE.search(clause) and not rule.exception:
        out.append(f"clause contains an exception cue ('{_EXC_CUE.search(clause).group(0)}') but the compiled rule has no exception")
    time_leaf = any((lf.unit or "").lower().rstrip("s") in {u.rstrip("s") for u in TIME_UNITS} for lf in iter_leaves(rule.condition))
    if _TEMP_CUE.search(clause) and rule.temporal is None and not time_leaf:
        out.append(f"clause contains a time constraint ('{_TEMP_CUE.search(clause).group(0)}') but the compiled rule has no temporal constraint")
    if _RANGE_CUE.search(clause) and not _has_range_construct(rule):
        out.append("clause describes a range (between ... and ...) but the compiled rule has no range / two-sided bound")
    if rule.exception and not _EXC_CUE.search(clause):
        out.append("compiled rule has an exception that the clause does not mention (possible hallucination)")
    return out


def _grounding_issues(rule: CompiledRule, clause: str, policy_text: str) -> List[str]:
    out: List[str] = []
    if not clause.strip():
        out.append("rule has no source_text quote; cannot verify it against the policy")
    elif _norm_ws(clause) not in _norm_ws(policy_text):
        out.append("source_text is not an exact quote of the policy (possible hallucination or paraphrase)")
    pool = numbers_in_text(clause if (clause.strip() and _norm_ws(clause) in _norm_ws(policy_text)) else policy_text)
    for where, v in _rule_numbers(rule):
        if not any(math.isclose(v, p, rel_tol=1e-9, abs_tol=1e-9) for p in pool):
            out.append(f"value {v:g} ({where}) does not appear in the policy text (possible hallucinated threshold)")
    return out


def _fmt_errors(e: ValidationError) -> List[str]:
    return [f"{'.'.join(str(x) for x in err['loc']) or 'rule'}: {err['msg']}" for err in e.errors()]


_IGNORABLE_KEYS = {"description", "notes", "explanation", "id", "name", "title", "policy_id", "rule_id", "status", "issues", "expression"}


def validate_raw_rule(raw: Any, policy_id: str, index: int, policy_text: str, min_confidence: float) -> Tuple[Optional[CompiledRule], List[str]]:
    """-> (rule, errors). rule is None when malformed (REJECTED). Otherwise VALID or NEEDS_REVIEW."""
    if not isinstance(raw, dict):
        return None, ["rule is not a JSON object"]
    data = {k: v for k, v in raw.items() if k not in _IGNORABLE_KEYS}
    data["policy_id"], data["rule_id"] = policy_id, f"{policy_id}-R{index:03d}"
    llm_amb = data.get("ambiguities")
    if llm_amb is not None and not (isinstance(llm_amb, list) and all(isinstance(x, str) for x in llm_amb)):
        return None, ["ambiguities must be a list of strings"]
    try:
        rule = CompiledRule.model_validate(data)
    except ValidationError as e:
        return None, _fmt_errors(e)

    clause = rule.source_text or ""
    amb: List[str] = [f"LLM-reported: {x}" for x in rule.ambiguities]
    amb += _grounding_issues(rule, clause, policy_text)
    amb += semantic_crosschecks(rule, clause or policy_text)
    scope = clause if clause.strip() else policy_text
    if not _HAS_NUMBER.search(scope):
        v, a = _VAGUE.search(scope), _ADVISORY.search(scope)
        if v:
            amb.append(f"clause uses vague term '{v.group(0)}' with no measurable value")
        elif a:
            amb.append(f"clause uses advisory wording '{a.group(0)}' (not a clear obligation)")
    issues: List[str] = []
    for lf in iter_leaves(rule.condition):
        if lf.operator in (Operator.GT, Operator.GTE, Operator.LT, Operator.LTE) and isinstance(lf.value, float) and not lf.unit:
            issues.append(f"numeric threshold on {lf.entity}.{lf.field} has no unit (acceptable for counts; verify)")
    if rule.confidence < min_confidence:
        amb.append(f"confidence {rule.confidence:.2f} is below the minimum {min_confidence:.2f}")

    rule.ambiguities = list(dict.fromkeys(amb))
    rule.issues = issues
    if rule.ambiguities:
        rule.status = "NEEDS_REVIEW"
        rule.confidence = round(min(rule.confidence, CONFIDENCE_CAP_WITH_ISSUES), 3)
    else:
        rule.status = "VALID"
    return rule, []


def make_policy_id(policy_text: str, policy_id: Optional[str] = None) -> str:
    if policy_id:
        if not _ID_RE.match(policy_id):
            raise ValueError("policy_id may only contain letters, digits, '_', '.', '-' (max 64)")
        return policy_id
    return "POL-" + hashlib.sha1(_norm_ws(policy_text).encode("utf-8")).hexdigest()[:10].upper()


def compile_policy(policy_text: str, policy_id: Optional[str] = None, min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                   llm_fn: Optional[Callable[[str, str], Optional[str]]] = None) -> CompileResult:
    """Natural language -> CompileResult. `llm_fn(system, user) -> raw text` can be injected (used by the evaluation harness)."""
    if not policy_text or not policy_text.strip():
        raise ValueError("policy_text is empty")
    if len(policy_text) > MAX_POLICY_CHARS:
        raise ValueError(f"policy_text exceeds {MAX_POLICY_CHARS} characters; split it into smaller policies")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    pid = make_policy_id(policy_text, policy_id)

    raw_text = (llm_fn or _call_llm)(SYSTEM_PROMPT, _build_user_prompt(policy_text))
    if not raw_text or not raw_text.strip():
        raise CompilationError("no LLM provider returned a response")
    data = _extract_json(raw_text)
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise CompilationError("LLM JSON has no 'rules' list")

    rules: List[CompiledRule] = []
    rejected: List[RejectedRule] = []
    for i, raw in enumerate(raw_rules, 1):
        rule, errors = validate_raw_rule(raw, pid, i, policy_text, min_confidence)
        if rule is None:
            rejected.append(RejectedRule(raw=raw, errors=errors, source_text=(raw.get("source_text") if isinstance(raw, dict) else None)))
        else:
            rules.append(rule)

    unparsed: List[Dict[str, Any]] = []
    for u in data.get("unparsed_statements") or []:
        if isinstance(u, dict) and u.get("text"):
            unparsed.append({"text": str(u["text"])[:500], "reason": str(u.get("reason") or "not compilable")[:300]})

    reasons: List[str] = []
    reasons += [f"{r.rule_id}: {a}" for r in rules for a in r.ambiguities]
    reasons += [f"rejected rule #{i}: {'; '.join(x.errors)}" for i, x in enumerate(rejected, 1)]
    reasons += [f"unparsed statement: \"{u['text'][:120]}\" ({u['reason']})" for u in unparsed]
    if not rules and not rejected and not unparsed:
        reasons.append("no rule could be extracted from the policy text")
    for s in lexical_ambiguity_reasons(policy_text):  # vague/advisory sentences with no measurable value, even if the LLM stayed silent
        snippet = s.split('"')[1][:40].lower()
        if not any(snippet in r.lower() for r in reasons):
            reasons.append(s)

    status = "REJECTED" if not rules else ("COMPILED_WITH_REVIEW" if reasons else "COMPILED")
    return CompileResult(
        policy_id=pid, source_text=policy_text, status=status, rules=rules, rejected=rejected, unparsed_statements=unparsed,
        ambiguous_policy=bool(reasons), ambiguity_reasons=reasons,
        stats={"extracted": len(raw_rules), "valid": sum(r.status == "VALID" for r in rules), "needs_review": sum(r.status == "NEEDS_REVIEW" for r in rules),
               "rejected": len(rejected), "unparsed_statements": len(unparsed)},
    )