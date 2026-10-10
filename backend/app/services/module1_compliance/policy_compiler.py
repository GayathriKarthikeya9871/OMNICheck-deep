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
import copy
import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .policy_schema import (
    CompiledRule, CompileResult, Condition, Logic, Operator, RejectedRule, RuleType, Severity,
    NEGATED_OPS, RANGE_OPS, TIME_UNITS, as_number, has_not_group, iter_leaves,
)

logger = logging.getLogger(__name__)

MAX_POLICY_CHARS = 20000
DEFAULT_MIN_CONFIDENCE = 0.7
CONFIDENCE_CAP_WITH_ISSUES = 0.49
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class CompilationError(Exception):
    """LLM unavailable or returned unusable output (distinct from a policy that is merely ambiguous)."""


# Bump whenever the prompt, normalization or validation changes the compiled output. It is appended to
# SYSTEM_PROMPT so any cache keyed on the prompt cannot replay a result from an incompatible compiler.
COMPILER_VERSION = "2026-10-10.7-severity-label-required-connector"

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
  "severity": OPTIONAL KEY: omit it unless the policy text itself explicitly states a severity level (see "Severity" below); when present it is exactly one of "low", "medium", "high", "critical",
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
Conditional obligations: when a clause applies only to some entities or states ("Any X above N must have Y", "If A then B", "Confirmed incidents must be reported to ..."), use TRIGGER: put the applicability test (A) in "condition" and the demanded proof or consequence in "required_evidence" and/or "action". Use REQUIRE only when the condition itself is the property every entity must have (e.g. "Passwords must be at least 12 characters"). Never use REQUIRE for a condition that merely says when the clause applies: a REQUIRE rule is a VIOLATION whenever its condition is false. For a "confirmed <thing> must be <action>" clause, emit a TRIGGER whose condition is the confirmation flag (entity "<thing>", field "confirmed", operator "==", value true) and whose action is the stated escalation.
Severity (optional key): include "severity" ONLY when the policy text itself explicitly states a severity level for that obligation, for example "high severity", "severity: critical", "priority: low" or "classified as medium". Otherwise omit the key so the schema default applies. Never infer or guess a severity from how serious, important, risky or costly the obligation sounds, from its consequence, or from its domain. A word such as "critical" in an ordinary description ("critical systems", "critical infrastructure", "a critical review") is NOT a severity classification. When given it must be exactly one of the lowercase strings low, medium, high, critical. The compiler checks every returned severity against the clause and discards any it cannot find there.
Mandatory documents/proof the policy demands ("with receipt", "signed approval", "attach ..."): use "required_evidence".
The evidence "type" must be the policy's own contiguous wording for that document as snake_case, INCLUDING who must issue it when the policy names a role (e.g. "a written sign-off from a Security Officer" -> "written_sign_off_from_a_security_officer"). Never drop the role, and never keep it only in "description": the evaluator can only verify what is part of the type.
If the policy constrains the evidence itself in time (for example, written approval must be attached before payment), represent that relation as an explicit evidence match attribute such as `"match":{"timing":"before_payment"}`. Do NOT translate that relation into `payment_date before now`: it is about the approval evidence relative to payment, not about whether payment occurred before the current date. The evaluator must still verify the relation from source evidence; if the evidence does not explicitly establish it, the result remains INDETERMINATE.

ILLUSTRATIVE EXAMPLES (different domains; do not copy their entities)
Text: "Servers storing customer data must not be reachable from the public internet."
-> {"rule_type":"PROHIBIT","entity":"server","condition":{"logic":"AND","children":[{"entity":"server","field":"stores_customer_data","operator":"==","value":true,"unit":null},{"entity":"server","field":"publicly_reachable","operator":"==","value":true,"unit":null}]},"temporal":null,"required_evidence":[],"exception":[],"action":"block_and_report","confidence":0.9,"source_text":"Servers storing customer data must not be reachable from the public internet.","ambiguities":[]}
Text: "Purchase orders above 50,000 USD require legal review unless issued under an approved framework agreement."
-> {"rule_type":"TRIGGER","entity":"purchase_order","condition":{"entity":"purchase_order","field":"total_value","operator":">","value":50000,"unit":"USD"},"temporal":null,"required_evidence":[{"type":"legal_review","mandatory":true,"min_count":1}],"exception":[{"description":"issued under an approved framework agreement","condition":{"entity":"purchase_order","field":"under_framework_agreement","operator":"==","value":true,"unit":null}}],"action":"legal_review_required","confidence":0.9,"source_text":"Purchase orders above 50,000 USD require legal review unless issued under an approved framework agreement.","ambiguities":[]}
Text: "Backup archives must be encrypted at rest (severity: critical)."
-> {"rule_type":"REQUIRE","entity":"backup_archive","condition":{"entity":"backup_archive","field":"encrypted_at_rest","operator":"==","value":true,"unit":null},"temporal":null,"required_evidence":[],"exception":[],"severity":"critical","action":"encrypt_backup_archive","confidence":0.9,"source_text":"Backup archives must be encrypted at rest (severity: critical).","ambiguities":[]}
(The first two examples omit "severity" because their source text states none; the third includes it because its source text states it.)
"""


SYSTEM_PROMPT = SYSTEM_PROMPT + f"\nCOMPILER_VERSION: {COMPILER_VERSION}\n"


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
        masked = _mask_severity_labels(s)  # an explicit severity label ("high severity") is not vague wording
        v = _VAGUE.search(masked)
        a = _ADVISORY.search(masked)
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
    out = []
    for err in e.errors():
        msg = f"{'.'.join(str(x) for x in err['loc']) or 'rule'}: {err['msg']}"
        if tuple(err["loc"]) == ("severity",):  # diagnostic only: the rule is still rejected
            msg += f" (received {type(err.get('input')).__name__} {repr(err.get('input'))[:60]})"
        out.append(msg)
    return out


_IGNORABLE_KEYS = {"description", "notes", "explanation", "id", "name", "title", "policy_id", "rule_id", "status", "issues", "expression", "severity_source"}  # severity_source is compiler-decided (see _assess_severity), never LLM input


def _normalize_explicit_approval_timing(data: Dict[str, Any]) -> bool:
    """Represent an explicit approval-before-payment relation as evidence metadata.

    This narrow normalization applies only when the source clause explicitly says approval
    must be before payment and the compiler attached an approval evidence requirement. It
    avoids the invalid inference `payment_date before now`; it does NOT invent calendar dates.
    The evidence must independently state the timing relation at evaluation time.
    """
    clause = str(data.get("source_text") or "")
    if not re.search(r"\bapproval\b[^.\n]{0,180}\bbefore\s+(?:the\s+)?payment\b", clause, re.I):
        return False
    reqs = data.get("required_evidence")
    if not isinstance(reqs, list):
        return False
    approval_reqs = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        label = f"{req.get('type', '')} {req.get('description', '')}"
        if re.search(r"\bapproval\b", label, re.I):
            approval_reqs.append(req)
    if not approval_reqs:
        return False

    temporal = data.get("temporal")
    if temporal is not None:
        if not isinstance(temporal, dict):
            return False
        entity = str(temporal.get("entity") or "").strip().lower()
        field = str(temporal.get("field") or "").strip().lower()
        kind = str(temporal.get("kind") or "").strip().lower()
        reference = str(temporal.get("reference") or "").strip().lower()
        # Real provider output has appeared in both forms:
        #   payment.payment_date before payment
        #   payment.date before payment.date
        # Both are a malformed representation of the *approval evidence* being before
        # payment, not a rule requiring the payment date to precede itself/current time.
        payment_date_field = field in ("payment_date", "date_of_payment") or (
            entity in ("payment", "payments", "the_payment")
            and field in ("date", "payment_date", "date_of_payment")
        )
        payment_reference = reference in (
            "now", "today", "evaluation_date", "payment", "payments", "the_payment",
            "payment.date", "payments.date", "the_payment.date", "payment.payment_date",
            "payment_date", "date_of_payment",
        ) or reference == f"{entity}.{field}"
        if not (payment_date_field and kind == "before" and payment_reference):
            return False
        data["temporal"] = None

    for req in approval_reqs:
        match = req.get("match")
        match = dict(match) if isinstance(match, dict) else {}
        match["timing"] = "before_payment"
        req["match"] = match

    # Drop only the model's explicit concern that this relation needs an approval timestamp;
    # the relation is now checked against the source evidence, not guessed from a date field.
    ambiguities = data.get("ambiguities")
    if isinstance(ambiguities, list):
        data["ambiguities"] = [
            x for x in ambiguities
            if not (isinstance(x, str) and re.search(r"approval timestamp|field for approval timestamp", x, re.I))
        ]
    return True


_CONFIRMED_VIOLATION_RE = re.compile(r"\bconfirmed\s+violations?\b", re.I)
_CONFIRM_COND_KEYS = {"entity", "field", "operator", "value", "unit"}
_VIOLATION_ENTITY_RE = re.compile(r"^(?:[a-z][a-z0-9_]*_)?violation$")  # 'violation' or '<base>_violation'
_ENTITY_SPACES = re.compile(r"[\s\-]+")


def _confirmation_form(cond: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """(entity, field) when `cond` is EXACTLY one recognized confirmation leaf, else None. Recognized forms:
         <violation|X_violation>.confirmed|violation_confirmed == true
         <violation|X_violation>.status|state == "confirmed"
         <base_entity>.violation_confirmed == true   (base-entity alias: the field name itself says 'violation')
       A bare '<other>.confirmed' is NOT recognized: 'employee.confirmed' / 'expense.confirmed' do not mean a violation."""
    if not isinstance(cond, dict) or not set(cond) <= _CONFIRM_COND_KEYS or cond.get("unit") not in (None, ""):
        return None
    ent, fld, op = cond.get("entity"), cond.get("field"), cond.get("operator")
    if not all(isinstance(x, str) for x in (ent, fld, op)) or op.strip() != "==":
        return None
    ent, fld = _ENTITY_SPACES.sub("_", ent.strip().lower()), _ENTITY_SPACES.sub("_", fld.strip().lower())
    val = cond.get("value")
    violation_entity = bool(_VIOLATION_ENTITY_RE.match(ent))
    if val is True:  # strictly boolean true: not 1, not "true"
        if fld == "violation_confirmed" and re.match(r"^[a-z][a-z0-9_]*$", ent):
            return ent, fld
        if fld == "confirmed" and violation_entity:
            return ent, fld
    elif isinstance(val, str) and val.strip().lower() == "confirmed" and violation_entity and fld in ("status", "state"):
        return ent, fld
    return None


def _normalize_confirmed_violation_rule_type(data: Dict[str, Any]) -> bool:
    """REQUIRE -> TRIGGER for an explicit "confirmed violation(s) must be <action>" clause.

    The clause is conditional: IF a violation is confirmed THEN the action is required. Typed REQUIRE, the
    confirmation flag becomes a property every record must have (a false flag is a VIOLATION, an unknown one is
    INDETERMINATE), which is never what the policy says. Narrow by construction: the clause must literally say
    "confirmed violation(s)", the condition must be the single recognized confirmation leaf (see _confirmation_form:
    entity, field, operator and value must agree; no extra predicates, unit or keys), the rule's own `entity` (when
    given) must match the condition's entity, an action must be stated and there must be no temporal constraint.
    Anything else is left untouched (a threshold REQUIRE, a malformed or unrelated condition). Only `rule_type` is
    changed: source_text and every other field keep their provenance; the caller records the audit note.
    """
    if str(data.get("rule_type") or "").strip().upper() != "REQUIRE" or data.get("temporal"):
        return False
    if not _CONFIRMED_VIOLATION_RE.search(str(data.get("source_text") or "")) or not str(data.get("action") or "").strip():
        return False
    form = _confirmation_form(data.get("condition"))
    if form is None:
        return False
    top = data.get("entity")
    if top not in (None, "") and (not isinstance(top, str) or _ENTITY_SPACES.sub("_", top.strip().lower()) != form[0]):
        return False
    data["rule_type"] = "TRIGGER"
    return True


# ---- severity grounding: a returned severity is a policy-stated fact only when the source clause says so.
_SEV_L = r"(low|medium|high|critical)"
_SEV_CONTEXT_RES = (
    re.compile(rf"\b{_SEV_L}[\s-]+(?:severity|priority)\b", re.I),                                        # "high severity"
    re.compile(rf"\b(?:severity|priority)(?:\s+level)?(?:\s+(?:is|as)\s+|\s*[=:]\s*){_SEV_L}\b", re.I),  # "severity: critical", "severity is high"; a connector is REQUIRED ("severity high costs", "severity of high costs" are not labels)
    re.compile(rf"\b(?:classified|rated|marked|labell?ed|categori[sz]ed|designated)\s+as\s+(?:an?\s+)?{_SEV_L}\b", re.I),
)
# Bare label words are only a CUE: they may or may not be a severity ("critical systems"), so they go to review.
_SEV_BARE_CUES = {"critical": {"critical"}, "minor": {"low"}, "moderate": {"medium"},
                  "major": {"high", "critical"}, "severe": {"high", "critical"}, "serious": {"high", "critical"}}
_SEV_BARE_RE = re.compile(r"\b(" + "|".join(_SEV_BARE_CUES) + r")\b", re.I)


def _mask_severity_labels(text: str) -> str:
    """Copy of `text` with ONLY the spans matching the explicit-severity patterns (_SEV_CONTEXT_RES) blanked out, same length so
    offsets are unchanged. Used solely before the vague/advisory scans: "high severity" must not be reported as the vague term
    'high'. Bare words ("high costs", "low confidence", "critical systems") do not match those patterns and are left alone, as is
    every other vague/advisory word. Never applied to source_text, the policy text or severity grounding."""
    out = text or ""
    for rx in _SEV_CONTEXT_RES:
        out = rx.sub(lambda m: " " * len(m.group(0)), out)
    return out


def _assess_severity(data: Dict[str, Any], policy_text: str) -> Dict[str, Any]:
    """Classify the severity the LLM returned against the source clause. -> {source, drop, issue, ambiguity}.
    Only a clause that is an exact policy quote is trusted. Never invents a severity; an unsupported value is dropped
    so the schema default applies and the discard is recorded. Values that are not valid severities are left for the
    schema to REJECT (not coerced, not silently dropped)."""
    out: Dict[str, Any] = {"source": "schema_default", "drop": False, "issue": None, "ambiguity": None}
    if "severity" not in data:
        return out
    raw = data["severity"]
    try:
        level = Severity(raw).value if isinstance(raw, str) else None
    except ValueError:
        level = None
    if level is None:
        out["source"] = None  # invalid: CompiledRule validation rejects the rule
        return out
    clause = str(data.get("source_text") or "")
    text = clause if clause.strip() and _norm_ws(clause) in _norm_ws(policy_text) else ""
    stated = {m.group(1).lower() for rx in _SEV_CONTEXT_RES for m in rx.finditer(text)}
    cues = {w.lower() for w in _SEV_BARE_RE.findall(text)}
    plausible = set().union(*(_SEV_BARE_CUES[c] for c in cues)) if cues else set()
    if len(stated) == 1 and level in stated:
        out["source"] = "policy_stated"
    elif len(stated) > 1:
        out.update(source="ambiguous", ambiguity=f"severity wording is ambiguous: the clause states several levels ({', '.join(sorted(stated))}); compiler returned '{level}'")
    elif len(stated) == 1:
        out.update(source="schema_default", drop=True,
                   issue=f"returned severity '{level}' discarded: the clause states '{next(iter(stated))}'; schema default applies",
                   ambiguity=f"compiler returned severity '{level}' but the clause states '{next(iter(stated))}'; schema default applied")
    elif level in plausible:
        out.update(source="ambiguous", ambiguity=f"severity '{level}' is inferred from the wording {sorted(cues)}, which does not state a severity level outright")
    else:
        out.update(source="schema_default", drop=True,
                   issue=f"returned severity '{level}' discarded: the source clause does not state a severity; schema default applies")
    return out


def validate_raw_rule(raw: Any, policy_id: str, index: int, policy_text: str, min_confidence: float) -> Tuple[Optional[CompiledRule], List[str]]:
    """-> (rule, errors). rule is None when malformed (REJECTED). Otherwise VALID or NEEDS_REVIEW."""
    if not isinstance(raw, dict):
        return None, ["rule is not a JSON object"]
    data = copy.deepcopy({k: v for k, v in raw.items() if k not in _IGNORABLE_KEYS})
    data["policy_id"], data["rule_id"] = policy_id, f"{policy_id}-R{index:03d}"
    evidence_timing_normalized = _normalize_explicit_approval_timing(data)
    trigger_retyped = _normalize_confirmed_violation_rule_type(data)
    sev = _assess_severity(data, policy_text)
    if sev["drop"]:
        data.pop("severity")
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
    crosschecks = semantic_crosschecks(rule, clause or policy_text)
    if evidence_timing_normalized:
        # The explicit temporal relation is represented on the evidence requirement above,
        # so the generic "time cue has no temporal object" warning no longer applies.
        crosschecks = [x for x in crosschecks if not ("clause contains a time constraint" in x and "no temporal constraint" in x)]
    amb += crosschecks
    scope = clause if clause.strip() else policy_text
    if not _HAS_NUMBER.search(scope):
        masked_scope = _mask_severity_labels(scope)  # only explicit severity-label spans are excluded; other vague/advisory wording still counts
        v, a = _VAGUE.search(masked_scope), _ADVISORY.search(masked_scope)
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

    if sev["ambiguity"]:
        amb.append(sev["ambiguity"])
    if sev["issue"]:
        issues.append(sev["issue"])
    if sev["source"]:
        rule.severity_source = sev["source"]
    rule.ambiguities = list(dict.fromkeys(amb))
    if evidence_timing_normalized:
        issues.append("explicit approval-before-payment relation represented as required_evidence.match.timing; source evidence must explicitly establish this relation")
    if trigger_retyped:
        issues.append("rule_type normalized REQUIRE -> TRIGGER: a confirmed-violation escalation clause is conditional (fires only when a violation is confirmed), not an unconditional requirement")
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