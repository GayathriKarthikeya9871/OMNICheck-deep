"""
policy_schema.py  (NEW FILE)
Location: backend/app/services/module1_compliance/policy_schema.py

Domain-neutral structured rule representation for the Policy-to-Executable Compliance Compiler.
Nothing here knows about expenses, HR, vendors, etc. Entities and fields are free-form snake_case
names taken from the policy text. Pure data + validation: no LLM, no I/O.
"""
import calendar
import math
import re
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CONDITION_DEPTH = 6
MAX_CONDITION_NODES = 50


class Operator(str, Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"            # range, value = [low, high]
    NOT_BETWEEN = "not_between"    # negated range
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


ORDERING_OPS = {Operator.GT, Operator.GTE, Operator.LT, Operator.LTE}
RANGE_OPS = {Operator.BETWEEN, Operator.NOT_BETWEEN}
LIST_OPS = {Operator.IN, Operator.NOT_IN}
NEGATED_OPS = {Operator.NEQ, Operator.NOT_IN, Operator.NOT_BETWEEN, Operator.NOT_CONTAINS, Operator.NOT_EXISTS}


class Logic(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"          # exactly 1 child
    IMPLIES = "IMPLIES"  # exactly 2 children: IF children[0] THEN children[1]  (conditional requirement)


class RuleType(str, Enum):
    REQUIRE = "REQUIRE"    # the predicate (condition AND temporal) MUST hold, else VIOLATION
    PROHIBIT = "PROHIBIT"  # the predicate must NOT hold, else VIOLATION
    TRIGGER = "TRIGGER"    # when the predicate holds, `action` is required (and required_evidence must exist)


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


TIME_UNITS = ("minutes", "hours", "days", "weeks", "months", "years")
_TIME_ALIASES = {"minute": "minutes", "min": "minutes", "mins": "minutes", "hour": "hours", "hr": "hours", "hrs": "hours",
                 "day": "days", "week": "weeks", "wk": "weeks", "month": "months", "year": "years", "yr": "years", "yrs": "years"}

_ENTITY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


# ----------------------------------------------------------------------------- helpers shared with the engine
def as_number(v: Any) -> Optional[float]:
    """Number from int/float or numeric string ('10,000'). bool and non-finite are NOT numbers."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


_DATE_FORMATS = ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y")


def parse_datetime(x: Any) -> Optional[datetime]:
    """ISO-8601 or 'D Mon YYYY' / 'Mon D, YYYY'. dd/mm vs mm/dd is NEVER guessed (returns None). Result is naive UTC."""
    dt: Optional[datetime] = None
    if isinstance(x, datetime):
        dt = x
    elif isinstance(x, date):
        dt = datetime(x.year, x.month, x.day)
    elif isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)
        except ValueError:
            for f in _DATE_FORMATS:
                try:
                    dt = datetime.strptime(s, f)
                    break
                except ValueError:
                    continue
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def is_date_string(v: Any) -> bool:
    return isinstance(v, str) and parse_datetime(v) is not None and bool(re.search(r"\d{4}", v))


def add_delta(dt: datetime, amount: float, unit: str, sign: int = 1) -> datetime:
    """Calendar-aware shift. months/years clamp the day (31 Jan + 1 month = 28/29 Feb)."""
    n = amount * sign
    if unit in ("months", "years"):
        months = int(n) * (12 if unit == "years" else 1)
        y = dt.year + (dt.month - 1 + months) // 12
        m = (dt.month - 1 + months) % 12 + 1
        return dt.replace(year=y, month=m, day=min(dt.day, calendar.monthrange(y, m)[1]))
    return dt + timedelta(**{unit: n})


def _scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) and not (isinstance(v, float) and not math.isfinite(v))


# ----------------------------------------------------------------------------- condition tree
class Condition(BaseModel):
    """Either a LEAF (entity, field, operator, value, unit) or a GROUP (logic + children). Never both."""
    model_config = ConfigDict(extra="forbid")

    logic: Optional[Logic] = None
    children: Optional[List["Condition"]] = None
    entity: Optional[str] = None
    field: Optional[str] = None
    operator: Optional[Operator] = None
    value: Any = None
    unit: Optional[str] = None
    inclusive: bool = True  # between / not_between bounds

    @field_validator("entity", mode="before")
    @classmethod
    def _entity(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("entity must be a string")
        s = re.sub(r"[\s\-]+", "_", v.strip().lower())
        if not _ENTITY_RE.match(s):
            raise ValueError(f"'{v}' is not a valid snake_case entity name")
        return s

    @field_validator("field", mode="before")
    @classmethod
    def _field(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("field must be a string")
        s = re.sub(r"[\s\-]+", "_", v.strip().lower())
        if not _FIELD_RE.match(s):
            raise ValueError(f"'{v}' is not a valid snake_case field name/path")
        return s

    @field_validator("unit", mode="before")
    @classmethod
    def _unit(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("unit must be a string or null")
        v = v.strip()
        return v or None

    @model_validator(mode="after")
    def _structure(self):
        is_group = self.logic is not None or self.children is not None
        is_leaf = any(x is not None for x in (self.entity, self.field, self.operator)) or self.value is not None
        if is_group and is_leaf:
            raise ValueError("a condition node cannot be both a group (logic/children) and a leaf (entity/field/operator/value)")
        if not is_group and not is_leaf:
            raise ValueError("empty condition node")
        if is_group:
            if self.logic is None or not self.children:
                raise ValueError("a group needs both 'logic' and a non-empty 'children' list")
            n = len(self.children)
            if self.logic == Logic.NOT and n != 1:
                raise ValueError("NOT takes exactly 1 child")
            if self.logic == Logic.IMPLIES and n != 2:
                raise ValueError("IMPLIES takes exactly 2 children [if, then]")
            if self.logic in (Logic.AND, Logic.OR) and n < 2:
                raise ValueError(f"{self.logic.value} needs at least 2 children")
            return self
        self._check_leaf()
        return self

    def _check_leaf(self) -> None:
        if self.entity is None or self.field is None or self.operator is None:
            raise ValueError("a leaf condition needs entity, field and operator")
        op, v = self.operator, self.value
        if op in (Operator.EXISTS, Operator.NOT_EXISTS):
            if v is not None:
                raise ValueError(f"operator '{op.value}' takes no value")
        elif op in RANGE_OPS:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(f"'{op.value}' needs value = [low, high]")
            nums = [as_number(x) for x in v]
            if all(n is not None for n in nums):
                lo, hi = nums
                self.value = [lo, hi]
            elif all(is_date_string(x) for x in v):
                lo, hi = parse_datetime(v[0]), parse_datetime(v[1])
                self.value = [str(v[0]).strip(), str(v[1]).strip()]
            else:
                raise ValueError("range bounds must be two numbers or two ISO dates")
            if lo > hi:
                raise ValueError(f"range low bound {v[0]} is greater than high bound {v[1]}")
        elif op in LIST_OPS:
            if not isinstance(v, (list, tuple)) or not v or not all(_scalar(x) for x in v):
                raise ValueError(f"'{op.value}' needs a non-empty list of scalar values")
            self.value = list(v)
        elif op in ORDERING_OPS:
            n = as_number(v)
            if n is not None:
                self.value = n
            elif is_date_string(v):
                self.value = v.strip()
            else:
                raise ValueError(f"'{op.value}' needs a numeric or ISO-date value (got {v!r}); vague or missing thresholds are not accepted")
        elif op in (Operator.EQ, Operator.NEQ, Operator.CONTAINS, Operator.NOT_CONTAINS):
            if v is None or not _scalar(v):
                raise ValueError(f"'{op.value}' needs a scalar value")


Condition.model_rebuild()


def iter_leaves(c: Optional[Condition]) -> Iterator[Condition]:
    if c is None:
        return
    if c.children:
        for ch in c.children:
            yield from iter_leaves(ch)
    else:
        yield c


def cond_depth(c: Optional[Condition]) -> int:
    if c is None:
        return 0
    return 1 + (max(cond_depth(ch) for ch in c.children) if c.children else 0)


def cond_size(c: Optional[Condition]) -> int:
    if c is None:
        return 0
    return 1 + (sum(cond_size(ch) for ch in c.children) if c.children else 0)


def has_not_group(c: Optional[Condition]) -> bool:
    if c is None:
        return False
    if c.logic == Logic.NOT:
        return True
    return any(has_not_group(ch) for ch in (c.children or []))


def render_condition(c: Optional[Condition]) -> str:
    if c is None:
        return "ALWAYS"
    if c.children:
        parts = [render_condition(x) for x in c.children]
        if c.logic == Logic.NOT:
            return f"NOT ({parts[0]})"
        if c.logic == Logic.IMPLIES:
            return f"({parts[0]} IMPLIES {parts[1]})"
        return "(" + f" {c.logic.value} ".join(parts) + ")"
    u = f" {c.unit}" if c.unit else ""
    path = f"{c.entity}.{c.field}"
    if c.operator in (Operator.EXISTS, Operator.NOT_EXISTS):
        return f"{path} {c.operator.value}"
    if c.operator in RANGE_OPS:
        lo, hi = c.value
        br = "[" if c.inclusive else "("
        er = "]" if c.inclusive else ")"
        return f"{path} {c.operator.value} {br}{lo}, {hi}{er}{u}"
    return f"{path} {c.operator.value} {c.value}{u}"


# ----------------------------------------------------------------------------- temporal / evidence / exception
class Temporal(BaseModel):
    """Constraint on a DATE field. kinds:
       within     field lies within `amount unit` of `reference` (direction after|before|either)
       older_than field is more than `amount unit` before `reference`
       before / after   field < / > reference
       between    start <= field <= end
       reference = 'now' (evaluation date) | ISO date | 'entity.field' path."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["within", "older_than", "before", "after", "between"]
    entity: str
    field: str
    reference: Optional[str] = None
    amount: Optional[float] = None
    unit: Optional[str] = None
    direction: Literal["after", "before", "either"] = "after"
    start: Optional[str] = None
    end: Optional[str] = None

    @field_validator("entity", mode="before")
    @classmethod
    def _entity(cls, v):
        s = re.sub(r"[\s\-]+", "_", str(v).strip().lower())
        if not _ENTITY_RE.match(s):
            raise ValueError(f"'{v}' is not a valid snake_case entity name")
        return s

    @field_validator("field", mode="before")
    @classmethod
    def _field(cls, v):
        s = re.sub(r"[\s\-]+", "_", str(v).strip().lower())
        if not _FIELD_RE.match(s):
            raise ValueError(f"'{v}' is not a valid snake_case field name/path")
        return s

    @field_validator("unit", mode="before")
    @classmethod
    def _unit(cls, v):
        if v is None:
            return None
        s = str(v).strip().lower()
        return _TIME_ALIASES.get(s, s)

    @model_validator(mode="after")
    def _check(self):
        k = self.kind
        if k in ("within", "older_than"):
            if self.amount is None or self.amount <= 0 or not math.isfinite(self.amount):
                raise ValueError(f"temporal '{k}' needs amount > 0")
            if self.unit not in TIME_UNITS:
                raise ValueError(f"temporal unit must be one of {TIME_UNITS}")
            if self.unit in ("months", "years") and self.amount != int(self.amount):
                raise ValueError("months/years must be whole numbers")
        if k in ("within", "older_than", "before", "after"):
            r = (self.reference or "").strip()
            if not r:
                raise ValueError(f"temporal '{k}' needs a reference ('now', an ISO date, or 'entity.field')")
            if not (r.lower() in ("now", "today", "evaluation_date") or parse_datetime(r) is not None or _FIELD_RE.match(r.lower())):
                raise ValueError(f"unusable temporal reference '{r}'")
            self.reference = r
        if k == "between":
            s, e = parse_datetime(self.start), parse_datetime(self.end)
            if s is None or e is None:
                raise ValueError("temporal 'between' needs ISO start and end dates")
            if s > e:
                raise ValueError("temporal start is after end")
        return self


def render_temporal(t: Optional[Temporal]) -> str:
    if t is None:
        return ""
    p = f"{t.entity}.{t.field}"
    if t.kind == "between":
        return f"{p} between {t.start} and {t.end}"
    if t.kind in ("before", "after"):
        return f"{p} {t.kind} {t.reference}"
    if t.kind == "older_than":
        return f"{p} older_than {t.amount:g} {t.unit} vs {t.reference}"
    return f"{p} within {t.amount:g} {t.unit} {t.direction} {t.reference}"


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str                                  # free-form evidence type, e.g. 'manager_approval', 'signed_contract'
    description: Optional[str] = None
    mandatory: bool = True
    min_count: int = Field(default=1, ge=1, le=1000)
    match: Optional[Dict[str, Any]] = None     # attribute equality constraints on the evidence item

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("evidence type must be a non-empty string")
        s = re.sub(r"[^a-z0-9]+", "_", v.strip().lower()).strip("_")
        if not s:
            raise ValueError("evidence type has no usable characters")
        return s


class ExceptionClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    condition: Condition


# ----------------------------------------------------------------------------- the rule
class CompiledRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    rule_id: str
    rule_type: RuleType
    condition: Optional[Condition] = None      # None = unconditional
    # Summary of the PRIMARY leaf (auto-derived; the `condition` tree is the source of truth for execution)
    entity: Optional[str] = None
    operator: Optional[Operator] = None
    value: Any = None
    unit: Optional[str] = None
    required_evidence: List[EvidenceRequirement] = Field(default_factory=list)
    exception: List[ExceptionClause] = Field(default_factory=list)
    temporal: Optional[Temporal] = None
    severity: Severity = Severity.MEDIUM
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    # --- provenance / validation outcome
    source_text: str = ""
    expression: Optional[str] = None
    status: Literal["VALID", "NEEDS_REVIEW"] = "VALID"
    ambiguities: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)

    @field_validator("policy_id", "rule_id")
    @classmethod
    def _ids(cls, v):
        if not _ID_RE.match(v or ""):
            raise ValueError("ids may only contain letters, digits, '_', '.', '-' (max 64)")
        return v

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, v):
        if not isinstance(v, str):
            raise ValueError("action must be a string")
        s = re.sub(r"[^a-z0-9]+", "_", v.strip().lower()).strip("_")
        if not _ACTION_RE.match(s):
            raise ValueError(f"action '{v}' cannot be normalised to snake_case (3-64 chars)")
        return s

    @field_validator("entity", mode="before")
    @classmethod
    def _entity(cls, v):
        if v is None:
            return v
        s = re.sub(r"[\s\-]+", "_", str(v).strip().lower())
        if not _ENTITY_RE.match(s):
            raise ValueError(f"'{v}' is not a valid snake_case entity name")
        return s

    @model_validator(mode="after")
    def _rule_checks(self):
        if self.condition is not None:
            if cond_depth(self.condition) > MAX_CONDITION_DEPTH or cond_size(self.condition) > MAX_CONDITION_NODES:
                raise ValueError("condition tree too deep/large")
        for ex in self.exception:
            if cond_depth(ex.condition) > MAX_CONDITION_DEPTH or cond_size(ex.condition) > MAX_CONDITION_NODES:
                raise ValueError("exception condition tree too deep/large")
        if not (self.condition or self.temporal or self.required_evidence):
            raise ValueError("rule has no condition, temporal constraint or required evidence: nothing to execute")
        if self.rule_type in (RuleType.PROHIBIT, RuleType.TRIGGER) and not (self.condition or self.temporal):
            raise ValueError(f"{self.rule_type.value} rule needs a condition or temporal constraint (otherwise it would always fire)")
        if self.rule_type == RuleType.PROHIBIT and self.required_evidence:
            raise ValueError("required_evidence is not meaningful for a PROHIBIT rule")
        leaves = list(iter_leaves(self.condition))
        if leaves:
            p = leaves[0]
            self.entity, self.operator, self.value, self.unit = p.entity, p.operator, p.value, p.unit
        elif self.temporal:
            self.entity = self.temporal.entity
            self.operator, self.value, self.unit = None, None, None
        if not self.entity:
            raise ValueError("entity could not be determined (give 'entity' for evidence-only rules)")
        parts = [render_condition(self.condition)] if self.condition else []
        if self.temporal:
            parts.append(render_temporal(self.temporal))
        pred = " AND ".join(parts) or "ALWAYS"
        self.expression = f"[{self.rule_type.value}] {pred} -> {self.action}"
        return self


# ----------------------------------------------------------------------------- compile output
class RejectedRule(BaseModel):
    raw: Any = None
    errors: List[str]
    source_text: Optional[str] = None


class CompileResult(BaseModel):
    policy_id: str
    source_text: str
    status: Literal["COMPILED", "COMPILED_WITH_REVIEW", "REJECTED"]
    rules: List[CompiledRule] = Field(default_factory=list)          # VALID + NEEDS_REVIEW (never REJECTED)
    rejected: List[RejectedRule] = Field(default_factory=list)
    unparsed_statements: List[Dict[str, Any]] = Field(default_factory=list)
    ambiguous_policy: bool = False
    ambiguity_reasons: List[str] = Field(default_factory=list)
    stats: Dict[str, int] = Field(default_factory=dict)
    llm_used: bool = True
    note: str = ("Rules were INTERPRETED by an LLM and then validated deterministically. Compliance decisions are made only by the "
                 "deterministic rule engine (rule_engine.py); confidence is LLM self-reported and capped by validation findings, not calibrated.")