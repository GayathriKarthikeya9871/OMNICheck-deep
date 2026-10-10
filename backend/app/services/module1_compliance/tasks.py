# FILE TO REPLACE: app/services/module1_compliance/tasks.py
import os
import re
import io
import json
import math
import time
import stat
import uuid
import hmac
import shutil
import socket
import logging
import zipfile
import base64
import hashlib
import tempfile
import requests
import ipaddress
import threading
import operator
import networkx as nx
from itertools import islice
from collections import Counter, defaultdict, deque
from urllib.parse import urlparse, urljoin, unquote, parse_qsl, quote
from datetime import datetime, timezone
from celery import Celery
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.util import connection as urllib3_connection
import urllib3

# Ensure env variables are loaded before configuration
_backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(_backend_dir, ".env"))

from app.db.database import SessionLocal
from app.models.domain import InvestigationRecord, EvidenceNode, EvidenceEdge

logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- STRICT IMPORT GUARDS ---
HAS_BS4 = HAS_PDFPLUMBER = HAS_DOCX = HAS_FITZ = HAS_PYTESSERACT = HAS_PHONENUMBERS = HAS_OPENPYXL = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError: logger.warning("BeautifulSoup not installed. HTML parsing degraded.")
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError: logger.warning("pdfplumber not installed.")
try:
    import docx
    HAS_DOCX = True
except ImportError: logger.warning("python-docx not installed.")
try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError: logger.warning("PyMuPDF (fitz) not installed.")
try:
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
    Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard (Pillow raises above 2x this)
    HAS_PYTESSERACT = True
except ImportError: logger.warning("Pillow or pytesseract not installed.")
try:
    import phonenumbers
    HAS_PHONENUMBERS = True
except ImportError: logger.warning("phonenumbers not installed.")
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError: logger.warning("openpyxl not installed.")

# Transient DB errors (retry-worthy). Optional import so module still loads without SQLAlchemy symbols.
try:
    from sqlalchemy.exc import OperationalError as _SAOperationalError, InterfaceError as _SAInterfaceError
    _DB_TRANSIENT_ERRORS: tuple = (_SAOperationalError, _SAInterfaceError)
except ImportError:
    _DB_TRANSIENT_ERRORS = ()

# Only these are retried by Celery. Everything else = permanent (bad input, bug) -> fail once, no retry.
TRANSIENT_ERRORS = (requests.exceptions.RequestException, ConnectionError, TimeoutError) + _DB_TRANSIENT_ERRORS

# --- Policy compiler integration: LLM-compiled structured rules executed by the deterministic rule_engine (separate modules).
# If the modules cannot be imported the legacy policy DSL evaluator below keeps working unchanged.
COMPILED_POLICY_ENABLED = os.getenv("COMPILED_POLICY_ENABLED", "true").lower() == "true"
COMPILED_POLICY_STRICT_UNITS = os.getenv("COMPILED_POLICY_STRICT_UNITS", "true").lower() == "true"  # unit-less fact => INDETERMINATE, never assumed
try:
    from .policy_compiler import compile_policy as _compile_policy, CompilationError as _CompilationError, DEFAULT_MIN_CONFIDENCE as _COMPILER_MIN_CONFIDENCE
    from .rule_engine import evaluate_policy as _evaluate_compiled_rules, norm_unit as _compiled_norm_unit
    from .policy_schema import CompiledRule as _CompiledRule, Condition as _CompiledCondition, Operator as _CompiledOperator, render_condition as _render_compiled_condition
    HAS_POLICY_COMPILER = True
except ImportError as _pc_err:
    HAS_POLICY_COMPILER = False
    logger.warning(f"Policy compiler / rule engine not importable ({_pc_err}); legacy policy DSL only.")


class PermanentTaskError(Exception):
    """Raised for malformed input. Never retried."""


# --- CONFIGURATION ---
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

def _nid(prefix: str) -> str:
    """Graph node id. 64 bits of randomness (was 32): negligible collision risk per investigation."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"

celery_app = Celery("compliance_tasks", broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"), backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"))

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("localhost", "127.0.0.1")
OLLAMA_PRO_MODEL = os.getenv("OLLAMA_PRO_MODEL", "deepseek-r1:8b")
OLLAMA_TIMEOUT = _env_int("OLLAMA_TIMEOUT_SECONDS", 60)
GROQ_KEYS = [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip("\"'").strip()
GEMINI_KEYS = [k for k in [os.getenv(f"GEMINI_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # NOTE: verify model still served; override via env.

AI_TEMPERATURE = 0.1  # same for every provider
# Set EXTERNAL_AI_ENABLED=false to keep ALL data local (Ollama only). Default true = old behaviour.
EXTERNAL_AI_ENABLED = os.getenv("EXTERNAL_AI_ENABLED", "true").lower() == "true"
# Max seconds spent across ALL external providers (Groq+Gemini). Ollama gets its own timeout after.
AI_EXTERNAL_BUDGET = _env_int("AI_EXTERNAL_BUDGET_SECONDS", 60)

# PII storage policy. Default: DB stores masked/redacted values only. Set STORE_RAW_PII=true to restore old behaviour.
# NOTE: STORE_RAW_PII only affects what is PERSISTED in the graph DB. Every payload sent to an external LLM
# is still passed through redact_pii() regardless of this flag.
STORE_RAW_PII = os.getenv("STORE_RAW_PII", "false").lower() == "true"
# Hard gate: raw storage ONLY if operator also sets STORE_RAW_PII_ACK=I_UNDERSTAND (access controls + retention policy in place).
if STORE_RAW_PII and os.getenv("STORE_RAW_PII_ACK", "") != "I_UNDERSTAND":
    logger.error("STORE_RAW_PII=true IGNORED: set STORE_RAW_PII_ACK=I_UNDERSTAND to confirm access controls and a retention policy. Masking stays ON.")
    STORE_RAW_PII = False
PII_HASH_KEY = os.getenv("PII_HASH_KEY", "").encode("utf-8")  # HMAC key for correlating masked entities
if STORE_RAW_PII:
    logger.warning("STORE_RAW_PII=true: raw phone numbers / GovIDs / unredacted evidence text are written to the graph DB. "
                   "Restrict DB access and retention accordingly.")

# V2 retrieval tuning
V2_TOP_K = max(1, _env_int("V2_TOP_K", 10))
V2_EXPANSION_LIMIT = max(0, _env_int("V2_EXPANSION_LIMIT", 5))
V2_MAX_HOPS = max(1, _env_int("V2_MAX_HOPS", 3))  # graph traversal depth from lexical seeds (incoming AND outgoing edges)
V2_TRAVERSAL_EDGE_BUDGET = max(10, _env_int("V2_TRAVERSAL_EDGE_BUDGET", 2000))  # max edges examined per traversal; hitting it is reported, never hidden
V2_TRAVERSAL_HUB_DEGREE = max(5, _env_int("V2_TRAVERSAL_HUB_DEGREE", 50))  # nodes with more incident edges are listed but not expanded (reported as hubs)
V2_NODE_CONTEXT_LIMIT = max(0, _env_int("V2_NODE_CONTEXT_LIMIT", 15))  # non-evidence nodes (Decision/Risk/Rule/Policy/Transaction/Document) returned as context
EVAL_GROUND_TRUTH_PATH = os.getenv("EVAL_GROUND_TRUTH_PATH", "").strip()  # optional JSON ground-truth file read by the Celery task (see evaluation section)

# Contradiction heuristic (same label + currency, different amount, different documents). Set false to disable.
ENABLE_CONTRADICTION_HEURISTIC = os.getenv("ENABLE_CONTRADICTION_HEURISTIC", "true").lower() == "true"
MAX_CONTRADICTION_EDGES = max(0, _env_int("MAX_CONTRADICTION_EDGES", 100))

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_DOWNLOAD_REDIRECTS = 3
MAX_DOWNLOAD_SECONDS = max(5, _env_int("MAX_DOWNLOAD_SECONDS", 60))  # overall wall-clock deadline PER download (all hops)
MAX_LOCAL_FILE_BYTES = max(1, _env_int("MAX_LOCAL_FILE_MB", 100)) * 1024 * 1024  # uploaded / extracted file cap
MAX_EMBEDDED_URLS_PER_DOC = 5  # links FETCHED per document (the rest are still recorded, as NOT_ATTEMPTED)
MAX_EMBEDDED_URLS_SCANNED_PER_DOC = 50  # links RECORDED per document
MAX_TEXT_READ_BYTES = 10 * 1024 * 1024  # plain-text read cap per file
MAX_PDF_PAGES = max(1, _env_int("MAX_PDF_PAGES", 200))
MAX_XLSX_ROWS = max(1, _env_int("MAX_XLSX_ROWS", 20000))
# Downloads ignore HTTP(S)_PROXY env by default so the connect-time SSRF guard sees the real target address.
# Set DOWNLOAD_TRUST_ENV_PROXY=true only if the worker MUST egress via a proxy (guard then only protects the proxy hop).
DOWNLOAD_TRUST_ENV_PROXY = os.getenv("DOWNLOAD_TRUST_ENV_PROXY", "false").lower() == "true"

# Global HTTP Sessions with Retry Backoff.
# allowed_methods explicit (urllib3 default idempotent set, so LLM POSTs are NOT blindly retried).
# respect_retry_after_header=False: hostile server cannot stall worker with huge Retry-After.
# http_session  = LLM / vision calls (honours proxy env).
# download_session = untrusted URL downloads (separate so proxy policy differs). requests.Session is safe for
# concurrent reads in practice; Celery prefork (default) gives each worker process its own copy anyway.
_RETRY_KWARGS = dict(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=False)
try:
    retries = Retry(allowed_methods=frozenset(["HEAD", "GET", "OPTIONS", "PUT", "DELETE", "TRACE"]), **_RETRY_KWARGS)
except TypeError:  # urllib3 < 1.26
    retries = Retry(method_whitelist=frozenset(["HEAD", "GET", "OPTIONS", "PUT", "DELETE", "TRACE"]), **_RETRY_KWARGS)
http_session = requests.Session()
http_session.mount('http://', HTTPAdapter(max_retries=retries))
http_session.mount('https://', HTTPAdapter(max_retries=retries))
download_session = requests.Session()
download_session.trust_env = DOWNLOAD_TRUST_ENV_PROXY
download_session.mount('http://', HTTPAdapter(max_retries=retries))
download_session.mount('https://', HTTPAdapter(max_retries=retries))

# --- VALIDATION & REDACTION UTILS ---
VERHOEFF_D = [[0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5], [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7], [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1], [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3], [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0]]
VERHOEFF_P = [[0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4], [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7], [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1], [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8]]

# One GovID pattern for detection, redaction AND graph extraction (first digit 2-9, 12 digits, optional separators, not inside longer digit run).
# SHAPE ONLY: 12 digits, optional 4-4-4 separators, not inside a longer digit run / card-like 16-digit group.
# Shape is neither "this is a Government ID" nor "this ID is valid". Redaction uses the shape (over-redaction is the
# privacy-safe side); entity creation uses find_govid_candidates(); validity uses verify_govid().
GOVID_PATTERN = re.compile(r'(?<!\d)(?<!\d[- ])\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)(?![- ]\d)')
_GOVID_CONTEXT = re.compile(r'(?i)(aadhaar|aadhar|\buid\b|gov(?:ernment)?[\s_-]?id|national[\s_-]?id|\bid[\s_-]?(?:no|number|num)\b|tax[\s_-]?id|\bssn\b)')
# Jurisdiction whose validation method is implemented. Unset/unknown => IDs are kept but stay UNVERIFIED.
GOVID_JURISDICTION = os.getenv("GOVID_JURISDICTION", "").strip().upper()
# Government-ID validation is performed ONLY when jurisdiction + ID format + checksum algorithm are ALL configured.
#   GOVID_JURISDICTION=IN  GOVID_ID_FORMAT=[2-9]\d{11}  GOVID_CHECKSUM_ALGORITHM=VERHOEFF
# Selecting a jurisdiction that has a built-in profile (GOVID_PROFILES) supplies format/algorithm for it; anything else
# must be configured explicitly. No generic checksum is ever assumed.
GOVID_ID_FORMAT = os.getenv("GOVID_ID_FORMAT", "").strip()
GOVID_CHECKSUM_ALGORITHM = os.getenv("GOVID_CHECKSUM_ALGORITHM", "").strip().upper()
GOVID_PROFILES = {"IN": {"format": r"[2-9]\d{11}", "algorithm": "VERHOEFF", "method": "IN_aadhaar_verhoeff"}}
# ONE redaction representation, used by V1 and V2 (reports, graph labels, evidence text, provenance, logs). No partial digits, ever.
REDACTED_GOVID = "[REDACTED_GOVID]"
REDACTED_PHONE = "[REDACTED_PHONE]"
MONEY_PATTERN = re.compile(r'(?<![A-Za-z])(Rs\.?|INR|USD|EUR|GBP|₹|\$)\s*(\d[\d,]*(?:\.\d+)?)', re.IGNORECASE)

_SECRET_PATTERNS = [
    (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.DOTALL), '[REDACTED_PRIVATE_KEY]'),
    (re.compile(r'\bAKIA[0-9A-Z]{16}\b'), '[REDACTED_SECRET]'),
    (re.compile(r'\bgh[pousr]_[A-Za-z0-9]{36,}\b'), '[REDACTED_SECRET]'),
    (re.compile(r'\bAIza[0-9A-Za-z_\-]{35}\b'), '[REDACTED_SECRET]'),
    (re.compile(r'\bsk-[A-Za-z0-9_\-]{20,}\b'), '[REDACTED_SECRET]'),
    (re.compile(r'\bxox[baprs]-[A-Za-z0-9\-]{10,}\b'), '[REDACTED_SECRET]'),
    (re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b'), '[REDACTED_JWT]'),
    (re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}'), 'Bearer [REDACTED_SECRET]'),
    (re.compile(r'(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret)\b(\s*[:=]\s*)(["\']?)[^\s"\',;]{4,}'), r'\1\2\3[REDACTED_SECRET]'),
]
_URL_CREDENTIALS = re.compile(r'(https?://)[^/\s@]+@')
_URL_QUERY = re.compile(r'(https?://[^\s"\'<>?#]+)\?[^\s"\'<>#]*')

def validate_verhoeff(num_str: str) -> bool:
    if not re.fullmatch(r'\d+', num_str): return False
    try:
        c = 0
        for i, n in enumerate(reversed(num_str)): c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(n)]]
        return c == 0
    except Exception: return False

def find_govid_candidates(text: str) -> List[Dict[str, Any]]:
    """DETECTION only (no validity claim). A 12-digit run is an ID-like candidate only if it is written in 4-4-4 groups
    OR an ID keyword (Aadhaar / Gov ID / UID / ...) appears just before or after it. Plain 12-digit numbers elsewhere
    (amounts, references, timestamps) are NOT treated as Government IDs. First digit is deliberately not filtered here."""
    out: List[Dict[str, Any]] = []
    for m in GOVID_PATTERN.finditer(text or ""):
        raw = m.group()
        grouped = bool(re.search(r'[- ]', raw))
        near = (text[max(0, m.start() - 40):m.start()] + " " + text[m.end():m.end() + 20])
        if grouped: basis = "grouped_4_4_4"
        elif _GOVID_CONTEXT.search(near): basis = "id_keyword_context"
        else: continue
        out.append({"start": m.start(), "end": m.end(), "raw": raw, "digits": re.sub(r'\D', '', raw), "basis": basis})
    return out

GOVID_ALGORITHMS = {"VERHOEFF": validate_verhoeff}

def govid_validation_config() -> Dict[str, Any]:
    """Resolved Government-ID validation configuration. complete=True only if jurisdiction, ID format AND a supported
    checksum algorithm are all available. `missing` lists what is not."""
    prof = GOVID_PROFILES.get(GOVID_JURISDICTION) if GOVID_JURISDICTION else None
    fmt = GOVID_ID_FORMAT or (prof["format"] if prof else "")
    algo = GOVID_CHECKSUM_ALGORITHM or (prof["algorithm"] if prof else "")
    missing: List[str] = []
    if not GOVID_JURISDICTION: missing.append("jurisdiction")
    if not fmt: missing.append("ID format")
    if not algo: missing.append("checksum algorithm")
    elif algo not in GOVID_ALGORITHMS: missing.append(f"supported checksum algorithm ('{algo}' is not implemented)")
    pattern = None
    if fmt:
        try: pattern = re.compile(fmt)
        except re.error: missing.append("valid ID-format pattern (configured pattern does not compile)")
    method = None
    if not missing:
        method = prof["method"] if (prof and not GOVID_ID_FORMAT and not GOVID_CHECKSUM_ALGORITHM) else f"{GOVID_JURISDICTION}_{algo.lower()}_configured"
    return {"complete": not missing, "missing": missing, "jurisdiction": GOVID_JURISDICTION or None, "pattern": pattern, "algorithm": algo or None, "method": method}

def govid_unvalidated_reason(cfg: Optional[Dict[str, Any]] = None) -> str:
    """The ONE canonical explanation for an unvalidated Government ID (V1 and V2)."""
    cfg = cfg or govid_validation_config()
    return ("validation was not performed because the required jurisdiction / ID-format / checksum-algorithm configuration is unavailable"
            + (f" (missing: {', '.join(cfg['missing'])})" if cfg.get("missing") else ""))

GOVID_NEXT_STEP = ("Conditional next step: (1) identify the jurisdiction and ID format; (2) determine whether that ID type has an applicable checksum validation method and whether the required "
                   "configuration is available; (3) run checksum validation only if both are true. If no applicable method exists or the configuration is unavailable, the ID remains UNVERIFIED and an "
                   "appropriate independent verification method should be requested. A checksum result, if any, shows arithmetic consistency only and is not proof of authenticity, ownership or document validity. "
                   "The outcome (VALID/INVALID) is not predetermined.")

def verify_govid(digits: str) -> Dict[str, Any]:
    """VERIFICATION, separate from detection. Runs ONLY when jurisdiction, ID format and checksum algorithm are all configured.
    Statuses: UNVERIFIED (validation not performed; is_valid=None) | INVALID_FORMAT | CHECKSUM_FAILED | CHECKSUM_PASSED.
    CHECKSUM_PASSED is arithmetic consistency under the configured method only, NOT proof the ID was issued, is authentic or belongs to anyone."""
    cfg = govid_validation_config()
    if not cfg["complete"]:
        return {"status": "UNVERIFIED", "is_valid": None, "method": None, "jurisdiction": cfg["jurisdiction"], "reason": govid_unvalidated_reason(cfg)}
    if not cfg["pattern"].fullmatch(digits or ""):
        return {"status": "INVALID_FORMAT", "is_valid": False, "method": cfg["method"], "jurisdiction": cfg["jurisdiction"], "reason": "does not match the configured ID format"}
    ok = bool(GOVID_ALGORITHMS[cfg["algorithm"]](digits))
    return {"status": "CHECKSUM_PASSED" if ok else "CHECKSUM_FAILED", "is_valid": ok, "method": cfg["method"], "jurisdiction": cfg["jurisdiction"],
            "reason": "checksum arithmetic only (configured method); authenticity not verified"}

def redact_secrets(text: str) -> str:
    """Masks API keys, tokens, private keys, passwords-in-assignments, URL credentials/query strings.
    FOR OUTBOUND / STORED / LOGGED TEXT ONLY. Never run this on a URL you still intend to download
    (query strings such as Drive ?id=... are needed to fetch the file)."""
    if not text: return ""
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    text = _URL_CREDENTIALS.sub(r'\1[REDACTED_CREDENTIALS]@', text)
    text = _URL_QUERY.sub(r'\1?[REDACTED_QUERY]', text)
    return text

def _safe_log_url(url: str) -> str:
    """URL for logs: credentials + query stripped. Never used for the actual request."""
    return redact_secrets(url or "")[:300]

# --- URL EXTRACTION / CLEANUP / CANONICAL KEY / LINK RECORDS / LINK RENDERING ---
# One path for every link: extract_urls() (document text) and _clean_extracted_url() (uploaded URL list) strip stray surrounding
# delimiters (Markdown brackets, escaped \] , unmatched closers, encoded %5D artifacts, sentence punctuation); canonical_url_key() gives a
# stable comparison key (hashed, so credentials/query values are never held in plain text in a record); _record_link() keeps ONE record
# per unique link, merged across input paths; normalize_link_log() is what BOTH V1 and V2 report sections read, so the two reports can
# never disagree about a link. The URL actually fetched is never altered beyond removing stray delimiters (query strings needed for
# retrieval are kept); only displayed/logged forms are redacted. Reports are Markdown (rendered by the client), so links are emitted as
# Markdown links [label](url); blue/underline styling is applied by the Markdown renderer, not by this code.
_URL_CANDIDATE = re.compile(r'https?://[^\s"\'<>`]+')
_URL_TRAIL_CHARS = ".,;:!?*'\""  # sentence / Markdown-emphasis punctuation at the END of a candidate
_URL_LEAD_CHARS = " \t<([{\"'`*"  # Markdown / quoting characters BEFORE the scheme
_URL_PAIRS = {')': '(', ']': '[', '}': '{'}
_URL_PAIR_ENC = {')': ('%29', '%28'), ']': ('%5D', '%5B'), '}': ('%7D', '%7B')}  # percent-encoded closer / opener
_MD_LINK_JOIN = re.compile(r'(?i)(?:\]|%5D)\(')  # '[text](url)' joint, plain or with the bracket percent-encoded
_MD_ESCAPED = re.compile(r'\\([\\`*_{}\[\]()<>#+\-.!|~"\'])')  # Markdown backslash escapes ( \] \) \_ ... ); a real URL never contains '\'
_ENC_TRAIL = re.compile(r'(?i)%(5D|7D|29)$')
_ENC_ARTIFACT_TAIL = re.compile(r'(?i)(?:%5C|%22|%27|%3E|%60|%2A|%E2%80%8B|&#93;|&#41;|&#125;|&rsqb;|&rpar;|&rcub;)$')  # encoded backslash / quote / '>' / backtick / '*' / zero-width space / HTML-entity closer left at the END of a link
_FULLWIDTH_TRAIL = "\u200b\u200c\u200d\ufeff\uff09\uff3d\uff5d\u3011\u300d\u300f\u300b\u3009"  # zero-width characters and full-width closers
_ENC_TO_CLOSER = {'5D': ']', '7D': '}', '29': ')'}

def _delim_unbalanced(u: str, closer: str) -> bool:
    """True when `closer` (plain or percent-encoded) occurs more often than its opener inside u (so the trailing one is stray)."""
    opener = _URL_PAIRS[closer]
    enc_c, enc_o = _URL_PAIR_ENC[closer]
    up = u.upper()
    return (u.count(closer) + up.count(enc_c)) > (u.count(opener) + up.count(enc_o))

def _clean_extracted_url(url: str) -> str:
    """Removes surrounding Markdown/sentence punctuation from a link DESTINATION. Closing ) ] } (plain or %-encoded, e.g. a stray '%5D' that
    came from a Markdown ']') are stripped ONLY when unbalanced inside the URL, so 'https://x/a_(b)' keeps its ')' while '[https://x/a]',
    'https://x/a\\]' and '(see https://x/a)' lose theirs. Query strings, fragments, path characters and balanced punctuation are kept.
    Never percent-encodes anything. Only the extracted destination is cleaned; source document text is never modified."""
    u = (url or "").strip()
    u = _MD_ESCAPED.sub(r'\1', u)  # '\]' -> ']' so the unmatched-delimiter rule below can see it
    u = u.lstrip(_URL_LEAD_CHARS)
    m = _MD_LINK_JOIN.search(u)  # Markdown [text](url): the URL ends where '](' begins
    if m and m.start() > 0: u = u[:m.start()]
    while u:
        last = u[-1]
        am_ = _ENC_ARTIFACT_TAIL.search(u)  # checked first: '&#93;' ends in ';', which would otherwise be stripped as sentence punctuation
        if am_ and am_.start() > 0:
            u = u[:am_.start()]; continue
        if last in _URL_TRAIL_CHARS or last in "\\>" or last in _FULLWIDTH_TRAIL:
            u = u[:-1]; continue
        if last in _URL_PAIRS and _delim_unbalanced(u, last):
            u = u[:-1]; continue
        em = _ENC_TRAIL.search(u)
        if em and _delim_unbalanced(u, _ENC_TO_CLOSER[em.group(1).upper()]):
            u = u[:-3]; continue
        break
    return u

_URL_ARTIFACT_TAIL = re.compile(r'(?i)(?:%5D|%7D|%29|%5C|\\|\])+')

def _strip_url_artifacts(text: str) -> str:
    """DISPLAY-ONLY: removes stray bracket / backslash / percent-encoded closers glued to the END of a URL inside an excerpt (for example 'https://x/y%5D'),
    so the same link never appears twice in a report, once clean and once malformed. Ordinary sentence punctuation and every other character is left
    exactly as written. The source file and the stored Evidence text are never modified."""
    def sub(m):
        raw = m.group(0)
        cleaned = _clean_extracted_url(raw)
        if cleaned == raw or not re.match(r'(?i)https?://[^/\s?#]+', cleaned): return raw
        removed = raw[len(cleaned):] if raw.startswith(cleaned) else ""
        return cleaned + _URL_ARTIFACT_TAIL.sub("", removed)
    return _URL_CANDIDATE.sub(sub, text or "")

def _url_key_noquery(url: str) -> str:
    """Comparison key that IGNORES the query string and fragment. Evidence text is secret-redacted before storage (query values become '[REDACTED_QUERY]'),
    so a link that carries a query string (for example a Drive '?id=...' link) can only be matched to its Evidence through scheme + host + path."""
    parts = _canonical_parts(url or "")
    if parts is None: return hashlib.sha256(_clean_extracted_url(url or "").encode("utf-8", "ignore")).hexdigest()[:16]
    scheme, host, path, _q, _shown = parts
    return hashlib.sha256(f"{scheme}://{host}{path}".encode("utf-8", "ignore")).hexdigest()[:16]

def _strip_stray_path_delims(path: str) -> str:
    """Removes trailing UNBALANCED closers (plain or %-encoded, e.g. '%5D') and slashes from a URL PATH. _clean_extracted_url only sees the end of
    the whole string, so a stray '%5D' followed by '/' ('.../repo%5D/') survives it and would otherwise become a second, 'different' link.
    Balanced pairs ('/a_(b)') are kept. Only the derived key/display form is affected; source text is never modified."""
    path = path.rstrip("/")
    while path:
        if path[-1] in _URL_PAIRS and _delim_unbalanced(path, path[-1]):
            path = path[:-1].rstrip("/"); continue
        em = _ENC_TRAIL.search(path)
        if em and _delim_unbalanced(path, _ENC_TO_CLOSER[em.group(1).upper()]):
            path = path[:-3].rstrip("/"); continue
        break
    return path

_ESC = re.compile(r'%([0-9A-Fa-f]{2})')

def _unquote_unreserved(s: str) -> str:
    """Decodes ONLY percent-escapes of unreserved characters (A-Z a-z 0-9 - . _ ~), which are equivalent to the literal character. Reserved
    escapes (%2F, %3F, %23, %20 ...) are kept (upper-cased), because they can change the meaning of a URL and so denote a different link."""
    def sub(m):
        ch = chr(int(m.group(1), 16))
        return ch if (ch.isascii() and (ch.isalnum() or ch in "-._~")) else "%" + m.group(1).upper()
    return _ESC.sub(sub, s)

def _github_fold(path: str, fold_case: bool) -> str:
    """GitHub owner and repository names are case-insensitive and the repo may carry '.git'. Deeper path segments (branch, file names)
    are case-sensitive and are never folded or trimmed, so '/o/r/blob/main/README.md' and '/o/r/blob/main/readme.md' stay distinct."""
    segs = path.split("/")  # ['', owner, repo, ...]
    if fold_case:
        for i in (1, 2):
            if len(segs) > i: segs[i] = segs[i].lower()
    if len(segs) > 2 and segs[2].lower().endswith(".git"): segs[2] = segs[2][:-4]
    return "/".join(segs)

def _canonical_parts(url: str) -> Optional[Tuple[str, str, str, str, str]]:
    """(scheme, host[:port], path, query, display_path) of the cleaned URL in canonical form, or None if unparsable. Formatting variants of one
    destination (case of scheme/host, default port, trailing '/', github 'www.' / '.git' / path case / http vs https, fragment, query-pair order,
    percent-encoding of unreserved path characters) map to the same tuple (first four fields). Case of deeper github path segments and reserved
    percent-escapes are significant, so genuinely distinct URLs keep distinct keys."""
    try:
        p = urlparse(_clean_extracted_url(url))
        scheme, host, port = p.scheme.lower(), (p.hostname or "").lower().rstrip("."), p.port
    except ValueError:
        return None
    if not scheme or not host: return None
    host_disp = f"[{host}]" if ":" in host else host  # IPv6 literal keeps its brackets
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)): host_port = f"{host_disp}:{port}"
    else: host_port = host_disp
    shown_path = _strip_stray_path_delims(re.sub(r'/{2,}', '/', p.path or ""))  # also drops '%5D' left before a trailing '/'
    path = _unquote_unreserved(shown_path)
    if host in ("github.com", "www.github.com") and not port:
        scheme, host_port = "https", "github.com"
        path = _github_fold(path, True)
        shown_path = _github_fold(shown_path, False)
    path = path.rstrip("/") or "/"
    shown_path = shown_path.rstrip("/") or "/"
    query = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(p.query, keep_blank_values=True)))
    return scheme, host_port, path, query, shown_path

def canonical_url_key(url: str) -> str:
    """Stable comparison key (16 hex chars of SHA-256) over the canonical parts above; userinfo and fragment are ignored."""
    parts = _canonical_parts(url)
    if parts is None:
        return hashlib.sha256(_clean_extracted_url(url or "").encode("utf-8", "ignore")).hexdigest()[:16]
    scheme, host, path, query, _ = parts
    return hashlib.sha256(f"{scheme}://{host}{path}?{query}".encode("utf-8", "ignore")).hexdigest()[:16]

def _display_url(url: str) -> Tuple[str, bool]:
    """(canonical url safe for reports/logs, had_query). Credentials, query and fragment are dropped; secret-looking tokens are redacted.
    Host is lower-cased, trailing '/' and github '.git' removed. This is NEVER the URL used for retrieval."""
    u = _clean_extracted_url(url)
    parts = _canonical_parts(u)
    try:
        had_query = bool(urlparse(u).query)
    except ValueError:
        had_query = False
    if parts is None: return redact_secrets(u)[:300], had_query
    scheme, host, _path, _query, shown_path = parts
    shown = f"{scheme}://{host}" + ("" if shown_path == "/" else shown_path)
    return redact_secrets(shown)[:300], had_query

def extract_url_occurrences(text: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """EVERY URL occurrence in document text, in order of appearance: {"url": cleaned URL, "raw": the source text exactly as it appeared
    (before any delimiter cleanup), "key": canonical key}. Occurrences of the same link are all returned (so each keeps its own original text);
    `limit` caps the number of UNIQUE links (by canonical key), never the repeat occurrences of a link already admitted."""
    out: List[Dict[str, str]] = []
    seen: set = set()
    for m in _URL_CANDIDATE.finditer(text or ""):
        for piece in _MD_LINK_JOIN.split(m.group(0)):  # '[http://a](http://b)' yields both URLs
            if not re.match(r'(?i)https?://', piece): continue
            u = _clean_extracted_url(piece)
            if not re.match(r'(?i)^https?://[^/\s?#]+', u): continue
            k = canonical_url_key(u)
            if k not in seen:
                if limit and len(seen) >= limit: continue
                seen.add(k)
            out.append({"url": u, "raw": piece, "key": k, "start": m.start()})
    return out

def extract_urls(text: str, limit: Optional[int] = None) -> List[str]:
    """URLs from document text: cleaned, host required, de-duplicated by canonical key, in order of appearance."""
    out: List[str] = []
    seen: set = set()
    for o in extract_url_occurrences(text, limit):
        if o["key"] in seen: continue
        seen.add(o["key"])
        out.append(o["url"])
    return out

# Link status model (what actually happened to a link). Priority when merging records of one link: RETRIEVED > BLOCKED/NOT_RETRIEVED > NOT_ATTEMPTED.
LINK_RETRIEVED = "RETRIEVED"
LINK_BLOCKED_SSRF = "BLOCKED_BY_SSRF_PROTECTION"
LINK_NOT_RETRIEVED = "NOT_RETRIEVED"
LINK_NOT_ATTEMPTED = "NOT_ATTEMPTED"
_LINK_STATUS_WORDS = {LINK_RETRIEVED: "retrieved", LINK_BLOCKED_SSRF: "blocked by SSRF protection", LINK_NOT_RETRIEVED: "retrieval failed", LINK_NOT_ATTEMPTED: "not attempted"}

def _link_status_of(retrieved: Optional[bool], reason: Optional[str]) -> str:
    if retrieved is True: return LINK_RETRIEVED
    if retrieved is None: return LINK_NOT_ATTEMPTED
    return LINK_BLOCKED_SSRF if re.search(r'(?i)\bssrf\b', str(reason or "")) else LINK_NOT_RETRIEVED

def _clean_reason(reason: Optional[str], default: str) -> str:
    return re.sub(r'\s+', ' ', redact_secrets(str(reason or default)))[:300]

def _merge_link(out: List[Dict[str, Any]], key: str, url: str, query_withheld: bool, retrieved: Optional[bool], reason: Optional[str], sources: List[str], status: Optional[str] = None, locations: Optional[List[str]] = None, originals: Optional[List[Tuple[str, str]]] = None) -> None:
    """Merge one observation into the unique-link list. retrieved=None = observed but no retrieval attempted by that path (it never downgrades a
    record and only adds a source to an existing one). RETRIEVED wins over any failed attempt for the same link; an explicit SSRF-blocked status is
    kept (never turned into 'retrieved' or generic failure); among failures the reasons are combined; NOT_ATTEMPTED is replaced by any real attempt."""
    new_status = status or _link_status_of(retrieved, reason)
    rec = next((r for r in out if r.get("key") == key), None)
    if rec is None:
        rec = {"url": url, "key": key, "query_withheld": bool(query_withheld), "retrieved": False, "status": LINK_NOT_ATTEMPTED, "reason": None, "sources": [], "locations": [], "originals": []}
        out.append(rec)
    else:
        rec["query_withheld"] = bool(rec.get("query_withheld")) or bool(query_withheld)
        rec.setdefault("status", LINK_RETRIEVED if rec.get("retrieved") else LINK_NOT_ATTEMPTED)
        if not rec.get("url") and url: rec["url"] = url
    cur = rec["status"]
    if cur == LINK_RETRIEVED:
        pass  # a real retrieval is never overwritten by a failed or skipped duplicate
    elif new_status == LINK_RETRIEVED:
        rec["status"], rec["reason"] = LINK_RETRIEVED, None
    elif new_status in (LINK_BLOCKED_SSRF, LINK_NOT_RETRIEVED):
        r = _clean_reason(reason, "no reason recorded")
        if cur == LINK_NOT_ATTEMPTED: rec["status"], rec["reason"] = new_status, r
        else:
            rec["status"] = LINK_BLOCKED_SSRF if LINK_BLOCKED_SSRF in (cur, new_status) else LINK_NOT_RETRIEVED
            existing = rec.get("reason") or ""
            if not existing: rec["reason"] = r
            elif r not in existing: rec["reason"] = (existing + "; " + r)[:500]
    else:  # NOT_ATTEMPTED observation
        if cur == LINK_NOT_ATTEMPTED and reason:
            r = _clean_reason(reason, "")
            existing = rec.get("reason") or ""
            if not existing: rec["reason"] = r
            elif r not in existing: rec["reason"] = (existing + "; " + r)[:500]
    rec["retrieved"] = rec["status"] == LINK_RETRIEVED
    if rec["status"] == LINK_NOT_ATTEMPTED and not rec.get("reason"): rec["reason"] = "no retrieval was attempted"
    for s in sources:
        if s and s not in rec["sources"]: rec["sources"].append(s)
    locs = rec.setdefault("locations", [])  # original source location / provenance of every retained occurrence (file @ sheet/row/page, or URL-list entry)
    for loc in (locations or []):
        loc = re.sub(r'\s+', ' ', redact_secrets(str(loc or "")))[:200]
        if loc and loc not in locs and len(locs) < 10: locs.append(loc)
    orig = rec.setdefault("originals", [])  # ORIGINAL source text of each occurrence (as written, redacted for secrets) + where it was found; the normalized "url" is a separate field
    for item in (originals or []):
        o_text, o_loc = (item if isinstance(item, (tuple, list)) and len(item) == 2 else (item, ""))
        entry = {"text": re.sub(r'\s+', ' ', redact_secrets(str(o_text or "")))[:300], "location": re.sub(r'\s+', ' ', redact_secrets(str(o_loc or "")))[:200]}
        if entry["text"] and entry not in orig and len(orig) < 10: orig.append(entry)

def _record_link(link_log: Optional[List[Dict[str, Any]]], raw_url: str, retrieved: Optional[bool], reason: Optional[str] = None, source: Optional[str] = None, location: Optional[str] = None, original: Optional[str] = None) -> None:
    """`original` = the link text exactly as it appeared in the source (stored in "originals" with its location, separate from the normalized "url")."""
    if link_log is None: return
    shown, has_q = _display_url(raw_url)
    _merge_link(link_log, canonical_url_key(raw_url), shown, has_q, retrieved, reason, [source] if source else [], None, [location] if location else [],
                [(original, location or "")] if original else None)

def normalize_link_log(link_log: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The single de-duplicated view of the link records used by every report section (V1 and V2). Idempotent; never mutates its input;
    records whose URL differs only in formatting collapse to one record (status/reason/sources/query flag merged by _merge_link)."""
    out: List[Dict[str, Any]] = []
    for l in (link_log or []):
        raw = l.get("url") or ""
        shown, has_q = _display_url(raw) if raw else ("", False)
        key = l.get("key") or (canonical_url_key(raw) if raw else hashlib.sha256(b"").hexdigest()[:16])
        status = l.get("status") or (LINK_RETRIEVED if l.get("retrieved") else _link_status_of(False, l.get("reason")))
        _merge_link(out, key, shown or raw, bool(l.get("query_withheld")) or has_q, status == LINK_RETRIEVED, l.get("reason"), list(l.get("sources") or []), status, list(l.get("locations") or []),
                    [(o.get("text"), o.get("location")) for o in (l.get("originals") or []) if isinstance(o, dict)])
    return out

def _links_for_output(links: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Structured, redacted link records (clean canonical url, status, reason, sources) for any downstream renderer; same normalized records as the reports."""
    return [{"url": l.get("url"), "status": l.get("status"), "retrieved": bool(l.get("retrieved")), "reason": l.get("reason"), "sources": list(l.get("sources") or []), "locations": list(l.get("locations") or []), "originals": [dict(o) for o in (l.get("originals") or [])],
             "query_withheld": bool(l.get("query_withheld")), "href": _link_href(l.get("url") or ""), "markdown": _link_md(l)} for l in normalize_link_log(links)]  # href/markdown: ready for a clickable renderer

def _attach_link_locations(link_log: Optional[List[Dict[str, Any]]], G: nx.MultiDiGraph, doc_node_id: str, file_name: str) -> None:
    """Adds 'file @ location' provenance to each already-recorded link that appears in this document's Evidence text (spreadsheet cell, page, section).
    Nothing is invented: a link whose text cannot be matched to an Evidence node keeps only its source channel."""
    if link_log is None: return
    by_key = {r.get("key"): r for r in link_log}
    by_nq = {_url_key_noquery(r.get("url") or ""): r for r in link_log if r.get("url")}  # Evidence text is query-redacted: fall back to scheme + host + path
    for ev, _, d in G.in_edges(doc_node_id, data=True):
        if d.get("relation") != "DERIVED_FROM" or G.nodes[ev].get("type") != "Evidence": continue
        for occ in extract_url_occurrences(G.nodes[ev].get("text") or ""):
            rec = by_key.get(occ["key"]) or by_nq.get(_url_key_noquery(occ["url"]))
            if rec is not None:
                where = f"{file_name} @ {d.get('location') or 'n/a'}"
                _merge_link(link_log, rec["key"], rec.get("url") or "", bool(rec.get("query_withheld")), None, None, [], None, [where])

def _attach_link_originals(link_log: Optional[List[Dict[str, Any]]], items: List[Tuple[str, str, str]]) -> None:
    """Adds the ORIGINAL text of uploaded-URL-list entries ((cleaned url, entry as given, location)) to their already-recorded link. Status is never touched."""
    if link_log is None: return
    by_key = {r.get("key"): r for r in link_log}
    for url, raw, loc in items:
        rec = by_key.get(canonical_url_key(url))
        if rec is not None: _merge_link(link_log, rec["key"], rec.get("url") or "", bool(rec.get("query_withheld")), None, None, [], None, None, [(raw, loc)])

def v2_trace_section(stats: Dict[str, Any]) -> str:
    """Deterministic V2 appendix block from the current investigation's traversal data. Reports edges examined (outgoing and incoming separately), followed, displayed and
    omitted, hop limit / deepest hop, hubs and frontier nodes NOT expanded, and whether the edge budget was exhausted. The list is labelled 'Complete trace' (every examined edge
    displayed) or 'Sample trace' (X of N shown). Every edge shown is a real traversal result with its real relation and direction; nothing is inferred or substituted.
    Never claims complete evidence lineage."""
    t = (stats or {}).get("trace") or {}
    out = "\n### Graph traversal (V2, deterministic)\n\n"
    if not t.get("inspected"):
        seeds_n = t.get("seeds", 0)
        why = "no lexical seed was available" if not seeds_n else f"none of the {seeds_n} lexical seed(s) has an incident edge to examine, or expansion is disabled"
        return out + f"Edges examined: **0**; displayed: **0**. No edge was examined ({why}). No traversal path is shown. This is not a complete evidence lineage.\n\n"
    n, edges = t["inspected"], (t.get("edges") or [])
    shown = len(edges)
    omitted = n - shown
    complete = omitted == 0
    out += (f"Traversal: breadth-first from {t.get('seeds', 0)} lexical seed(s) over edges in both directions, up to **{t.get('max_hops')}** hop(s) (deepest hop reached: **{t.get('max_hop_reached', 0)}**). "
            f"Edges examined: **{n}** (outgoing **{t.get('outgoing_inspected', 0)}**, incoming **{t.get('incoming_inspected', 0)}**); followed: **{t.get('followed', 0)}**; "
            f"edges displayed: **{shown}**; omitted: **{omitted}**. Nodes expanded: **{t.get('nodes_expanded', 0)}**.\n\n")
    if t.get("frontier_not_expanded"): out += f"{t['frontier_not_expanded']} node(s) at the hop limit were reached but their edges were NOT examined.\n\n"
    if t.get("hubs_not_expanded"): out += f"Hub node(s) reached but NOT expanded: {', '.join(f'`{h[0]}` ({h[1]}, degree {h[2]})' for h in t['hubs_not_expanded'])}.\n\n"
    if t.get("budget_exhausted"): out += f"**Edge budget ({t.get('edge_budget')}) exhausted**: the traversal stopped early and further edges exist that were NOT examined.\n\n"
    if complete: out += f"**Complete trace** (all {n} of {n} examined edges are displayed below; complete for the examined edges only).\n\n"
    else: out += (f"**Sample trace** ({shown} of {n} edges shown). The other {omitted} edge(s) were examined but are NOT displayed; this list does not represent them. "
                  "Selection basis: edges on a path to returned evidence/nodes are listed first, then the remaining examined edges in traversal order up to the display cap; it is not a random or representative sample.\n\n")
    out += "".join(f"{i}. `{e}`\n" for i, e in enumerate(edges, 1)) + "\n"
    paths = t.get("expansion_paths") or []
    if paths:
        out += f"Graph-expanded evidence reached (**{t.get('expanded', len(paths))}**" + ("; the expansion limit was reached, so further evidence may exist" if t.get("limit_reached") else "") + "):\n\n"
        for i, p in enumerate(paths, 1):
            nos = p.get("path_edge_nos") or []
            ref = (f"{p.get('hops', 1)} hop(s); path edges displayed as {nos}" if nos else f"{p.get('hops', 1)} hop(s); its path edges were examined but are not displayed") + (f"; {p['path_edges_not_displayed']} of its path edge(s) not displayed" if p.get("path_edges_not_displayed") else "")
            out += f"{i}. `{' ; '.join(p.get('relations') or [])}` => `{p.get('evidence_id')}` ({ref})\n"
        out += "\n"
    else:
        out += f"Paths that led to additional graph-expanded evidence: **{t.get('expanded', 0)}**.\n\n"
    npaths = t.get("node_paths") or []
    if npaths:
        out += f"Non-evidence nodes reached (context only; **{len(npaths)}**):\n\n"
        for i, p in enumerate(npaths, 1):
            out += f"{i}. `{' ; '.join(p.get('relations') or [])}` => `{p.get('node_id')}` ({p.get('type')}; {p.get('hops')} hop(s); path edges displayed as {p.get('path_edge_nos') or []})\n"
        out += "\n"
    out += ("This trace covers the edges incident to the nodes expanded within the hop limit and edge budget (outgoing and incoming). Edges beyond the hop limit, at hub nodes, past the budget, or between "
            "nodes never reached were not examined, so it is not a complete evidence lineage and does not show that all relevant evidence was found. Any individual evidence path shown elsewhere in this report is an EXAMPLE, not the full graph trace.\n\n")
    return out

def link_status_summary(links: List[Dict[str, Any]]) -> str:
    """'4 unique link(s): 1 retrieved, 1 blocked by SSRF protection, ...' computed from the normalized records only."""
    counts = Counter(l.get("status") or (LINK_RETRIEVED if l.get("retrieved") else LINK_NOT_RETRIEVED) for l in links)
    parts = [f"{counts[s]} {_LINK_STATUS_WORDS[s]}" for s in (LINK_RETRIEVED, LINK_BLOCKED_SSRF, LINK_NOT_RETRIEVED, LINK_NOT_ATTEMPTED) if counts.get(s)]
    return f"{len(links)} unique link(s): " + (", ".join(parts) or "none")

def _md_escape_label(s: str) -> str:
    return re.sub(r'([\\\[\]`*_<>|])', r'\\\1', s)

def _link_label(l: Dict[str, Any]) -> str:
    """Readable link text: host + path (no scheme, query or credentials), shortened."""
    u = l.get("url") or ""
    try:
        p = urlparse(u)
        label = (p.hostname or "") + ((p.path or "").rstrip("/") if p.path not in ("", "/") else "")
    except ValueError:
        label = ""
    label = label or re.sub(r'(?i)^https?://', '', u)
    if len(label) > 70: label = label[:67].rstrip("/") + "..."
    return _md_escape_label(label or "link")

def _link_href(url: str) -> str:
    """Destination for a Markdown link: the cleaned canonical display URL with the PATH percent-encoded so ( ) [ ] | and spaces cannot end the
    link early or break a table. Existing %XX escapes are kept. Query/credentials are never included (they are withheld from reports)."""
    try:
        p = urlparse(url or "")
    except ValueError:
        return ""
    if p.scheme not in ("http", "https") or not p.netloc: return ""
    path = re.sub(r'%(?![0-9A-Fa-f]{2})', '%25', p.path or "")
    return f"{p.scheme}://{p.netloc}" + quote(path, safe="/%:@!$&'*+,;=-._~")

def _link_md(l: Dict[str, Any]) -> str:
    """Live Markdown hyperlink [label](href); label is readable text, href is the cleaned canonical destination."""
    href = _link_href(l.get("url") or "")
    return f"[{_link_label(l)}]({href})" if href else _md_escape_label(l.get("url") or "")

def _link_ref(l: Dict[str, Any], md: bool = False) -> str:
    ref = _link_md(l) if md else (l.get("url") or "")
    if l.get("query_withheld"): ref += " [query parameters withheld]"
    if l.get("sources"): ref += f" (found via: {', '.join(l['sources'])})"
    locs = l.get("locations") or []
    if locs: ref += f" [source location: {'; '.join(locs[:3])}" + (f"; +{len(locs) - 3} more" if len(locs) > 3 else "") + "]"
    arts = [o for o in (l.get("originals") or []) if o.get("text") and o["text"] != _clean_extracted_url(o["text"])]  # only where an extraction artifact was removed
    if md and arts:
        _al = list(dict.fromkeys(o["location"] for o in arts if o.get("location")))
        ref += (f" [the source text had a formatting artifact (stray bracket, backslash or percent-encoded delimiter) at {len(arts)} occurrence(s)" + (f" ({'; '.join(_al[:2])})" if _al else "") +
                "; it was removed from the normalized URL above, so this is ONE link listed once; the malformed text is not repeated here and is kept only in the structured link data]")
    return ref

def _link_status_line(l: Dict[str, Any], md: bool = False, for_llm: bool = False) -> str:
    """One report line for a normalized link record. Wording follows the status exactly: nothing is said to be inspected unless retrieval succeeded."""
    st = l.get("status") or (LINK_RETRIEVED if l.get("retrieved") else LINK_NOT_RETRIEVED)
    ref = _link_ref(l, md)
    if st == LINK_RETRIEVED:
        return (f"- {ref}: RETRIEVED; extracted content appears " + ("below" if for_llm else "in the evidence") +
                " as unverified source text (retrieval alone does not validate the link or the expense).\n")
    if st == LINK_BLOCKED_SSRF:
        return (f"- {ref}: BLOCKED BY SSRF PROTECTION ({l.get('reason')}); its content was NOT inspected; "
                "the block says nothing about the link or the underlying expense.\n")
    if st == LINK_NOT_ATTEMPTED:
        return (f"- {ref}: NOT ATTEMPTED ({l.get('reason')}); content NOT inspected; nothing is known about the link or the underlying expense.\n")
    return (f"- {ref}: NOT RETRIEVED ({l.get('reason')}); content NOT inspected; "
            + ("so nothing is known about what the link contains; the failure says nothing about whether the link or the underlying expense is valid. "
               "Request an invoice, receipt, or other relevant supporting evidence through an approved, accessible channel. The retrieval failure is separate from the validity of the expense.\n"
               if for_llm else "this says nothing about the link or the underlying expense.\n"))

# ONE wording for two different things that must never be mixed up (used in the appendix, the V1 brief, the V2 context and the LLM rules).
LINK_VS_FINDING_NOTE = ("Terminology: NOT RETRIEVED, BLOCKED BY SSRF PROTECTION and NOT ATTEMPTED are RETRIEVAL OUTCOMES: they record only whether the system could fetch a link; "
                        "they say nothing about the link's content, validity or the underlying expense. INCONCLUSIVE is a FINDING (a policy-rule verdict): the rule was attempted but the evidence was "
                        "insufficient for a pass or a violation. A link is never itself 'inconclusive' or 'failed'; an unretrieved link only means its content was not inspected, which can be one reason "
                        "a related finding stays INCONCLUSIVE. INCONCLUSIVE and UNEVALUATED are neither passed nor failed.")

# --- V2 relevant-link reporting (system data only; the LLM never decides retrieval status) ---
_OBJECTIVE_LINK_TERMS = re.compile(r'(?i)\b(?:links?|urls?|hyperlinks?|supporting\s+(?:documents?|evidence|links?)|proof\s+of\s+(?:purchase|payment|expense))\b')  # the objective must refer to links / supporting documents themselves; expense words such as 'invoice' or 'receipt' alone do not make every link relevant
_LINK_STATUS_CAPS = {LINK_NOT_RETRIEVED: "NOT RETRIEVED", LINK_BLOCKED_SSRF: "BLOCKED BY SSRF PROTECTION", LINK_NOT_ATTEMPTED: "NOT ATTEMPTED"}

def relevant_unretrieved_links(G: nx.MultiDiGraph, links: Optional[List[Dict[str, Any]]], evidence_ids: Optional[List[str]], objective: str = "") -> List[Dict[str, Any]]:
    """Unretrieved links (from the normalized records) that matter to THIS investigation: the link's URL occurs in Evidence that this
    investigation actually used (lexical seeds + graph-expanded), or the objective itself refers to links / receipts / supporting documents.
    Other unretrieved links stay only in the Supporting links list. Nothing is inferred about link content."""
    ev_set = set(evidence_ids or [])
    by_key: Dict[str, List[str]] = defaultdict(list)  # keyed WITHOUT the query string: stored Evidence text is query-redacted, so a link with a query string could never match its own Evidence
    for ev, d in _investigation_evidence(G):
        for u in extract_urls(d.get("text") or ""):
            k = _url_key_noquery(u)
            if ev not in by_key[k]: by_key[k].append(ev)
    used_docs = {x["document_id"] for e in ev_set for x in _evidence_source_docs(G, e)}
    obj_refers = bool(_OBJECTIVE_LINK_TERMS.search(objective or ""))
    out: List[Dict[str, Any]] = []
    for l in (links or []):
        if l.get("retrieved") or l.get("status") == LINK_RETRIEVED: continue
        evs = by_key.get(_url_key_noquery(l.get("url") or ""), [])
        hit = [e for e in evs if e in ev_set]
        same_doc = [e for e in evs if e not in ev_set and any(x["document_id"] in used_docs for x in _evidence_source_docs(G, e))]
        if hit: out.append({"link": l, "evidence_ids": hit, "basis": "its URL appears in evidence used in this investigation"})
        elif same_doc: out.append({"link": l, "evidence_ids": same_doc[:5], "basis": "its URL appears in a document from which evidence was used in this investigation"})
        elif obj_refers: out.append({"link": l, "evidence_ids": evs[:5], "basis": "the objective refers to supporting links or documents"})
    return out

def v2_unretrieved_link_note(rel: List[Dict[str, Any]]) -> str:
    """One deterministic sentence block (no URLs: the Supporting links list stays the only link list). Retrieval status is kept apart from finding status."""
    if not rel: return ""
    statuses = sorted({_LINK_STATUS_CAPS.get(r["link"].get("status"), "NOT RETRIEVED") for r in rel})
    ids = sorted({e for r in rel for e in r.get("evidence_ids", [])})
    return (f"> **Relevant supporting link(s) not retrieved (system):** {len(rel)} link(s) relevant to this investigation have retrieval status {' / '.join(statuses)}"
            + (f" (linked to evidence {', '.join(ids)})" if ids else "") + "; their contents were not inspected. Retrieval status only records whether the system could fetch a link. "
            "It is separate from the assessment status of any finding (for example INCONCLUSIVE) and does not show that a link or the underlying expense is valid, invalid, or proof for or against any claim. "
            "See the Supporting links list in the Audit Appendix.")

def v2_unretrieved_link_row(rel: List[Dict[str, Any]]) -> List[str]:
    """Cells for ONE Key Findings table row (Finding | Evidence / Source | Status | Severity) about relevant unretrieved links. System data only: no URL, no claim about the link."""
    if not rel: return []
    ids = sorted({e for r in rel for e in r.get("evidence_ids", [])})
    statuses = sorted({_LINK_STATUS_CAPS.get(r["link"].get("status"), "NOT RETRIEVED") for r in rel})
    return [f"{len(rel)} relevant supporting link(s) were not retrieved; their contents were not inspected",
            "System link records; Supporting links list (Audit Appendix)" + (f"; linked evidence {', '.join(ids)}" if ids else ""),
            f"{' / '.join(statuses)} (retrieval status, not a finding status)",
            "Not assessed (nothing is known about the link or the underlying expense)"]

def _add_kf_table_row(section: str, cells: List[str]) -> str:
    """Appends one row to the first Markdown table inside the Key Findings section (needs >= 4 columns; otherwise the section is returned unchanged)."""
    if not cells: return section
    lines = section.split("\n")
    first = last = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|"):
            if first is None: first = i
            last = i
        elif first is not None: break
    if first is None or last is None or last - first < 1: return section
    ncols = len([c for c in lines[first].strip().strip("|").split("|")])
    if ncols < 4: return section
    row = [c.replace("|", "¦") for c in cells[:4]] + ["n/a"] * (ncols - 4)
    lines.insert(last + 1, "| " + " | ".join(row) + " |")
    return "\n".join(lines)

_LINK_NOTE_PRESENT = re.compile(r'(?is)(?:\blinks?\b|\burls?\b)[^.\n]{0,200}?(?:not\s+retrieved|not\s+inspected|retrieval\s+(?:failed|was\s+blocked)|blocked\s+by\s+ssrf)|(?:not\s+retrieved|not\s+inspected|blocked\s+by\s+ssrf)[^.\n]{0,200}?(?:\blinks?\b|\burls?\b)')

_KEY_FINDINGS_H = re.compile(r'(?mi)^#{1,6}\s*(?:\*\*)?\s*(?:\d+[.):]?\s*)?Key\s+Findings\b')
_ANY_HEADING = re.compile(r'(?m)^#{1,6}\s')

def ensure_unretrieved_link_note(llm_text: str, note: str, row_cells: Optional[List[str]] = None) -> str:
    """Deterministic retrieval-limitation note, placed at the END OF THE KEY FINDINGS SECTION (before the next heading), added only when that section does
    not already say a relevant link was not retrieved / not inspected. Falls back to just before Recommended Actions / the appendix / the end."""
    if not note or not llm_text: return llm_text
    am = _APPENDIX_RE.search(llm_text)
    body_end = am.start() if am else len(llm_text)
    kf = _KEY_FINDINGS_H.search(llm_text, 0, body_end)
    if kf:
        nxt = _ANY_HEADING.search(llm_text, kf.end(), body_end)
        end = nxt.start() if nxt else body_end
        if _LINK_NOTE_PRESENT.search(llm_text[kf.start():end]): return llm_text
        section = _add_kf_table_row(llm_text[kf.start():end], row_cells or [])  # the unretrieved link is a visible Key Findings ROW (retrieval status) as well as a note
        return (llm_text[:kf.start()] + section).rstrip() + "\n\n" + note + "\n\n" + llm_text[end:].lstrip("\n")
    body = llm_text[:body_end]
    if _LINK_NOTE_PRESENT.search(body): return llm_text
    h = re.search(r'(?m)^#{1,6}\s*(?:\*\*)?\s*3\b', body)
    if h: return llm_text[:h.start()].rstrip() + "\n\n" + note + "\n\n" + llm_text[h.start():]
    if am: return body.rstrip() + "\n\n" + note + "\n\n" + llm_text[am.start():]
    return llm_text.rstrip() + "\n\n" + note

def redact_pii(text: str) -> str:
    """Masks secrets, 12-digit GovID-shaped numbers and phone numbers before external API submission / DB storage."""
    if not text: return ""
    text = redact_secrets(text)
    text = GOVID_PATTERN.sub(REDACTED_GOVID, text)
    if HAS_PHONENUMBERS:
        try:
            # Collect spans on the CURRENT text first, then splice back-to-front so earlier offsets stay valid.
            spans = sorted(((m.start, m.end) for m in phonenumbers.PhoneNumberMatcher(text, "IN")), reverse=True)
            for start, end in spans:
                text = text[:start] + REDACTED_PHONE + text[end:]
        except Exception as e: logger.warning(f"PII Redact Phone Error: {e}")
    return text

def _phone_matches(text: str) -> list:
    """Phone matches that do NOT overlap a GovID-shaped number (a 12-digit ID must never become a Phone entity)."""
    if not HAS_PHONENUMBERS or not text: return []
    id_spans = [(m.start(), m.end()) for m in GOVID_PATTERN.finditer(text)]
    return [m for m in phonenumbers.PhoneNumberMatcher(text, "IN") if not any(m.start < e and m.end > s for s, e in id_spans)]

def _pii_hash(value: str) -> Optional[str]:
    """Keyed hash for correlation without storing raw value. None if PII_HASH_KEY unset (unkeyed hash is brute-forceable)."""
    if not PII_HASH_KEY: return None
    return hmac.new(PII_HASH_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()

def _store_pii_value(raw: str, kind: str = "phone") -> str:
    """Stored/displayed entity value. Default: full redaction token (NO partial/last-4 digits). Raw only with STORE_RAW_PII."""
    if STORE_RAW_PII: return raw
    return REDACTED_GOVID if kind == "govid" else REDACTED_PHONE

def _json_safe(obj: Any) -> Any:
    """DB-safe property payload: no NaN/Inf, no NUL bytes (Postgres rejects both), no non-JSON types."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, int): return obj
    if isinstance(obj, float): return obj if math.isfinite(obj) else None
    if isinstance(obj, str): return obj.replace("\x00", "")
    if isinstance(obj, dict): return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [_json_safe(v) for v in obj]
    return str(obj)

def generate_system_validation_report(text: str) -> str:
    """Report never contains raw IDs/phones (ordinal labels only) so intermediate text stays clean."""
    report_lines = []
    try:
        for i, match in enumerate(_phone_matches(text), start=1):
            status = "PASSED (Format Valid)" if phonenumbers.is_valid_number(match.number) else "FAILED (Format Invalid)"
            report_lines.append(f"Phone #{i}: {status}")
    except Exception as e: logger.warning(f"Validation Phone Error: {e}")
    for i, cand in enumerate(find_govid_candidates(text), start=1):
        ver = verify_govid(cand["digits"])
        if ver["status"] == "UNVERIFIED": status = f"UNVERIFIED ({govid_unvalidated_reason()}; validity and authenticity not claimed). {GOVID_NEXT_STEP}"
        elif ver["status"] == "CHECKSUM_PASSED": status = f"CHECKSUM PASSED ({ver['method']}; arithmetic check only, issuance/authenticity NOT verified)"
        else: status = f"{ver['status']} ({ver['method']})"
        report_lines.append(f"Gov ID #{i} [{REDACTED_GOVID}]: {status}")
    if not report_lines: return ""
    return "\n--- [SYSTEM ALGORITHMIC PRE-CHECK REPORT] ---\n" + "\n".join(report_lines) + "\n--------------------------------------------\n\n"

def generate_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""): sha256.update(block)
        return sha256.hexdigest()
    except Exception: return "hash_error"

# --- SENSITIVE FILE POLICY ---
_SENSITIVE_EXTS = {"env", "pem", "key", "p12", "pfx", "jks", "keystore"}
_SENSITIVE_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".npmrc", ".netrc", ".pgpass", ".htpasswd", "credentials"}
_CODE_EXTS = {"py", "js", "sh", "c", "cpp", "java", "yml", "yaml", "json"}

def is_sensitive_file(file_path: str) -> bool:
    """Credential/secret stores are never ingested (.env, private keys, etc.)."""
    base = os.path.basename(file_path or "").lower()
    if not base: return False
    ext = os.path.splitext(base)[1].lstrip('.')
    return base.startswith(".env") or ext in _SENSITIVE_EXTS or base in _SENSITIVE_NAMES

# --- VISION (Gemini) ---
# Restored ID/passport-photo compliance prompt (was reduced to a generic "extract text" prompt).
# PRIVACY: the IMAGE ITSELF is sent to Gemini (pixels cannot be text-redacted). Only the model's text reply is
# redacted. Gated by ENABLE_VISION_API=true AND EXTERNAL_AI_ENABLED=true AND a Gemini key.
VISION_PROMPT = (
    "You are an ID / passport photo compliance checker. Analyze this image factually and report each item:\n"
    "1. FACE: Is a human face present? How many faces? Is it frontal and fully visible?\n"
    "2. QUALITY: Is the image blurry, poorly lit, cropped, low resolution, or visibly edited/modified (filters, retouching, "
    "composited or tampered regions)? Say what you observe.\n"
    "3. ID/PASSPORT PHOTO CRITERIA: Does it meet typical criteria (frontal pose, neutral expression, plain background, "
    "no obstruction of face, adequate lighting and resolution)? Give PASS/FAIL per criterion with a short reason.\n"
    "4. TEXT: Transcribe all visible text exactly as written.\n"
    "5. OBJECTS: List other visible objects or document elements.\n"
    "Be factual. Do not guess or state the identity of any person. If something is not visible, say 'not determinable'."
)
_VISION_CACHE: Dict[Tuple[str, int, float], str] = {}
_VISION_CACHE_MAX = 256
MAX_VISION_IMAGE_BYTES = 15 * 1024 * 1024
# Requirements differ per country/document. Set e.g. VISION_DOCUMENT_STANDARD="India passport photo (MEA spec)".
# Unset => generic criteria, and the output says so.
VISION_DOCUMENT_STANDARD = os.getenv("VISION_DOCUMENT_STANDARD", "").strip()

def _vision_prompt() -> str:
    if VISION_DOCUMENT_STANDARD:
        std = (f"Evaluate item 3 against THIS standard: {VISION_DOCUMENT_STANDARD}. "
               "State which requirements you can and cannot verify from the image alone.\n")
    else:
        std = ("No jurisdiction/document standard was supplied. For item 3 use GENERIC criteria only and state that "
               "country/document-specific requirements were NOT checked.\n")
    return VISION_PROMPT + std

def analyze_image_with_vision(image_path: str) -> str:
    if os.getenv("ENABLE_VISION_API", "false").lower() != "true": return ""
    if not EXTERNAL_AI_ENABLED: return ""
    if not GEMINI_KEYS: return ""
    try:
        ext = os.path.splitext(image_path)[1].lstrip('.').lower()
        mime_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "bmp": "image/bmp"}.get(ext)
        if not mime_type: return ""
        size = os.path.getsize(image_path)
        if size > MAX_VISION_IMAGE_BYTES:
            logger.warning(f"Vision skipped (image too large): {os.path.basename(image_path)}")
            return ""
        cache_key = (image_path, size, os.path.getmtime(image_path))  # same image is read by V1 + provenance paths: call API once
        if cache_key in _VISION_CACHE: return _VISION_CACHE[cache_key]
        with open(image_path, "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
        payload = {"contents": [{"parts": [{"text": _vision_prompt()}, {"inline_data": {"mime_type": mime_type, "data": img_data}}]}],
                   "generationConfig": {"temperature": AI_TEMPERATURE}}
        for key in GEMINI_KEYS:
            resp = http_session.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent", headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=payload, timeout=20.0)
            if resp.status_code == 200:
                raw_text = _extract_gemini_text(resp.json())
                if not raw_text.strip():
                    logger.warning("Vision API returned empty text. Trying next key.")
                    continue
                std_note = VISION_DOCUMENT_STANDARD or "none supplied (generic criteria only; NOT a jurisdiction-specific check)"
                result = f"\n--- [SYSTEM VISION PRE-CHECK: PRELIMINARY OBSERVATION, NOT A FORMAL COMPLIANCE DECISION | standard: {std_note}] ---\n{redact_pii(raw_text)}\n"
                if len(_VISION_CACHE) >= _VISION_CACHE_MAX: _VISION_CACHE.clear()
                _VISION_CACHE[cache_key] = result
                return result
            logger.warning(f"Vision API HTTP {resp.status_code}")
    except Exception as e: logger.warning(f"Vision API Error: {e}")
    return ""

# --- SECURITY UTILS (SSRF HARDENED) ---
def _is_public_ip(ip_str: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str.split('%')[0])
    except ValueError:
        return False
    if ip_obj.version == 6 and ip_obj.ipv4_mapped: ip_obj = ip_obj.ipv4_mapped
    return bool(ip_obj.is_global and not (ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_reserved
                                          or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified))

def is_safe_url(url: str) -> bool:
    """
    Pre-flight SSRF check: scheme, no userinfo, and EVERY resolved address must be public.
    DNS-rebinding TOCTOU is closed at connect time by the guarded create_connection below
    (active only inside download_url_to_temp): the address actually connected to is re-resolved,
    validated and pinned there.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'): return False
        if parsed.username or parsed.password: return False
        if not parsed.hostname: return False

        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        addr_info = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        if not addr_info: return False
        return all(_is_public_ip(ai[4][0]) for ai in addr_info)
    except Exception as e:
        logger.warning(f"URL Validation failed for {_safe_log_url(url)}: {e}")
        return False

# Connect-time SSRF guard. Thread-local flag => only download traffic is restricted
# (LLM calls to local Ollama on 127.0.0.1 are unaffected). Assumes direct egress (download_session ignores proxy env
# unless DOWNLOAD_TRUST_ENV_PROXY=true). Re-verify after upgrading urllib3/requests: this patches an internal hook.
_ssrf_local = threading.local()
_orig_create_connection = getattr(urllib3_connection.create_connection, "_ssrf_orig", urllib3_connection.create_connection)

def _guarded_create_connection(address, *args, **kwargs):
    if not getattr(_ssrf_local, "active", False):
        return _orig_create_connection(address, *args, **kwargs)
    host, port = address
    host = host.strip("[]") if isinstance(host, str) else host
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ips = [ai[4][0] for ai in infos]
    if not ips or not all(_is_public_ip(ip) for ip in ips):
        raise OSError(f"SSRF BLOCKED: {host} resolves to non-public address")
    last_err: Optional[Exception] = None
    for ip in dict.fromkeys(ips):  # pinned: connect to validated IP literal, no 2nd DNS lookup
        try:
            return _orig_create_connection((ip, port), *args, **kwargs)
        except OSError as e:
            last_err = e
    raise last_err or OSError("SSRF guard: connection failed")

_guarded_create_connection._ssrf_orig = _orig_create_connection
urllib3_connection.create_connection = _guarded_create_connection

def _transform_download_url(url: str) -> str:
    """Dropbox/Drive direct-download rewriting. Result is re-validated by caller.
    Operates on the RAW url (query string intact); redaction is never applied to download targets."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return url
    if host == "dropbox.com" or host.endswith(".dropbox.com"):
        return url.replace("?dl=0", "?dl=1")
    if host == "drive.google.com":
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', parsed.path)
        if match: return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url

def _sniff_extension(path: str) -> Optional[str]:
    """Real file type from magic bytes (never trust Content-Type / URL text)."""
    try:
        with open(path, 'rb') as f: head = f.read(16)
    except OSError:
        return None
    if head.startswith(b'%PDF'): return 'pdf'
    if head.startswith(b'\x89PNG\r\n\x1a\n'): return 'png'
    if head.startswith(b'\xff\xd8\xff'): return 'jpg'
    if head[:6] in (b'GIF87a', b'GIF89a'): return 'gif'
    if head.startswith(b'RIFF') and head[8:12] == b'WEBP': return 'webp'
    if head.startswith(b'PK\x03\x04'):
        try:
            with zipfile.ZipFile(path) as z: names = z.namelist()[:2000]
            if any(n.startswith('word/') for n in names): return 'docx'
            if any(n.startswith('xl/') for n in names): return 'xlsx'
            return 'zip'
        except Exception:
            return None
    return None

def download_url_to_temp(url: str, dest_dir: str) -> Tuple[str, str]:
    """Download ONE url into dest_dir. Always pass the task's managed temp dir (cleaned in the task's finally).
    Overall wall-clock deadline (MAX_DOWNLOAD_SECONDS) spans all redirect hops and the body stream."""
    current = _transform_download_url(url)  # transform FIRST, then validate the real target
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'}
    deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
    _ssrf_local.active = True
    try:
        for _ in range(MAX_DOWNLOAD_REDIRECTS + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1: return "", "DOWNLOAD TIMEOUT: overall deadline exceeded."
            if not is_safe_url(current): return "", "SSRF BLOCKED: Restricted/Invalid address."
            with download_session.get(current, headers=headers, stream=True, timeout=min(15, remaining), allow_redirects=False) as resp:
                if resp.status_code in [301, 302, 303, 307, 308]:
                    location = resp.headers.get('Location')
                    if not location: return "", "REDIRECT BLOCKED: Missing Location header."
                    current = urljoin(current, location)  # every hop re-validated at loop top
                    continue
                resp.raise_for_status()

                c_len = resp.headers.get('Content-Length')
                try:
                    if c_len and int(c_len) > MAX_DOWNLOAD_BYTES: return "", "FILE TOO LARGE."
                except ValueError: pass

                header_ext = 'txt'
                c_type = resp.headers.get('Content-Type', '').lower()
                if 'pdf' in c_type: header_ext = 'pdf'
                elif 'zip' in c_type or 'archive' in current: header_ext = 'zip'
                elif 'jpeg' in c_type or 'jpg' in c_type: header_ext = 'jpg'
                elif 'png' in c_type: header_ext = 'png'
                elif 'csv' in c_type: header_ext = 'csv'
                elif 'spreadsheetml' in c_type: header_ext = 'xlsx'
                elif 'wordprocessingml' in c_type: header_ext = 'docx'
                elif 'html' in c_type: header_ext = 'html'

                tmp_path = os.path.join(dest_dir, f"{uuid.uuid4().hex}.{header_ext}")
                downloaded_size = 0
                with open(tmp_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        downloaded_size += len(chunk)
                        if downloaded_size > MAX_DOWNLOAD_BYTES:
                            f.close()
                            os.remove(tmp_path)
                            return "", "FILE TOO LARGE."
                        if time.monotonic() > deadline:
                            f.close()
                            os.remove(tmp_path)
                            return "", "DOWNLOAD TIMEOUT: overall deadline exceeded."
                        f.write(chunk)

                # Verify real content type vs claimed one.
                sniffed = _sniff_extension(tmp_path)
                if sniffed: final_ext = sniffed
                elif header_ext in ('pdf', 'zip', 'jpg', 'png', 'xlsx', 'docx'): final_ext = 'bin'  # claimed binary, content disagrees
                else: final_ext = header_ext
                if final_ext != header_ext:
                    new_path = os.path.join(dest_dir, f"{uuid.uuid4().hex}.{final_ext}")
                    os.replace(tmp_path, new_path)
                    tmp_path = new_path
                return tmp_path, "Success"
        return "", "REDIRECT BLOCKED: Too many redirects."
    except Exception as e: return "", f"Network/System Error: {redact_secrets(str(e))[:300]}"
    finally:
        _ssrf_local.active = False

def safe_extract_zip(zip_path: str, extract_to: str) -> List[str]:
    """Hardened against Zip-Slips, Zip-Bombs, Symlink/special-file entries. Extracts into its OWN subdir,
    so a failure only removes this archive's files (never sibling downloads in extract_to).
    This is the ONLY archive extraction path in this module (no extractall anywhere)."""
    extracted_files = []
    MAX_SIZE = 100 * 1024 * 1024
    MAX_FILE_SIZE = 25 * 1024 * 1024
    MAX_ENTRIES = 1000
    total_size, entries = 0, 0
    os.makedirs(extract_to, exist_ok=True)
    base_extract_dir = os.path.realpath(tempfile.mkdtemp(prefix="zip_", dir=extract_to))

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                if info.is_dir(): continue
                entries += 1
                if entries > MAX_ENTRIES: raise ValueError("Zip Bomb: Too many entries.")

                if info.compress_size > 0 and (info.file_size / info.compress_size) > 100:
                    raise ValueError("Zip Bomb: Highly compressed ratio.")

                target_path = os.path.realpath(os.path.join(base_extract_dir, info.filename))
                if os.path.commonpath([base_extract_dir, target_path]) != base_extract_dir or target_path == base_extract_dir:
                    raise ValueError(f"Zip Slip: Path traversal attempt to {target_path}")

                mode = info.external_attr >> 16
                if stat.S_IFMT(mode) and not stat.S_ISREG(mode): continue  # symlinks/devices/fifos/sockets (many zips store permission bits only: those are regular files)
                if info.file_size > MAX_FILE_SIZE:
                    logger.warning(f"ZIP entry skipped (> per-file limit): {info.filename}")
                    continue

                total_size += info.file_size
                if total_size > MAX_SIZE: raise ValueError("Zip Bomb: Uncompressed size too large.")

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with z.open(info) as source, open(target_path, "wb") as target: shutil.copyfileobj(source, target)
                extracted_files.append(target_path)
    except Exception as e:
        logger.error(f"ZIP Extraction failed: {e}")
        shutil.rmtree(base_extract_dir, ignore_errors=True)  # intentional: discard ALL of this archive's output
        return []
    return extracted_files

def _expand_input(path: str, dest_dir: str) -> Tuple[List[str], str]:
    """Single entry for ALL untrusted inputs (uploads, GitHub zips, URL downloads, embedded links).
    ZIPs (by extension or magic bytes; docx/xlsx sniff as their own type) go through safe_extract_zip.
    Returns (files, log_message). Empty files + message when an archive was rejected."""
    if path.lower().endswith('.zip') or _sniff_extension(path) == 'zip':
        files = safe_extract_zip(path, dest_dir)
        if not files: return [], f"Archive {os.path.basename(path)} rejected or empty (see worker log)."
        return files, ""
    return [path], ""

# --- BASELINE V1 TEXT EXTRACTION ---
_PLAIN_TEXT_EXTS = ["txt", "csv", "md", "py", "js", "html", "json", "yml", "yaml", "c", "cpp", "java", "sh"]

def _load_xlsx_pair(file_path: str):
    """(formula workbook, cached-value workbook). Formula cells carry cached results only in the data_only copy."""
    wb_f = openpyxl.load_workbook(file_path, data_only=False)
    try: wb_v = openpyxl.load_workbook(file_path, data_only=True)
    except Exception: wb_v = None
    return wb_f, wb_v

def _cell_display(fcell, vcell) -> str:
    f = fcell.value
    v = vcell.value if vcell is not None else None
    if isinstance(f, str) and f.startswith("="):
        return f"{v} [formula: {f}]" if v is not None else f"[formula: {f}] (no cached value)"
    return "" if f is None else str(f)

def _xlsx_rows(file_path: str):
    """Yield (sheet_title, row_number, [(coordinate, display_text), ...]) for non-empty rows.
    Shows cached value AND formula, so formula cells are never silently empty. Capped at MAX_XLSX_ROWS."""
    wb_f, wb_v = _load_xlsx_pair(file_path)
    emitted = 0
    try:
        for sheet in wb_f.worksheets:
            vsheet = wb_v[sheet.title] if (wb_v is not None and sheet.title in wb_v.sheetnames) else None
            for row in sheet.iter_rows():
                cells = []
                for cell in row:
                    vcell = vsheet[cell.coordinate] if vsheet is not None else None
                    val = _cell_display(cell, vcell)
                    link = getattr(cell, "hyperlink", None)
                    if link is not None and getattr(link, "target", None): val += f" [URL: {link.target}]"
                    if val.strip(): cells.append((cell.coordinate, val.strip()))
                if cells:
                    emitted += 1
                    if emitted > MAX_XLSX_ROWS:
                        logger.warning(f"XLSX row cap ({MAX_XLSX_ROWS}) reached for {os.path.basename(file_path)}; remaining rows skipped.")
                        return
                    row_num = getattr(row[0], "row", None) or emitted
                    yield sheet.title, row_num, cells
    finally:
        try: wb_f.close()
        except Exception: pass
        if wb_v is not None:
            try: wb_v.close()
            except Exception: pass

def extract_text_from_file(file_path: str, redact_code: bool = True) -> str:
    """redact_code=True (default): code/config text is secret-redacted at extraction.
    The batch task passes False for its URL-harvest pass because redact_secrets() strips URL query strings
    (which breaks Drive/Dropbox links); everything is redacted again before leaving the process anyway."""
    if not file_path or not os.path.exists(file_path): return ""
    if is_sensitive_file(file_path):
        logger.warning(f"Sensitive file skipped (not ingested): {os.path.basename(file_path)}")
        return ""
    ext = os.path.splitext(os.path.basename(file_path).lower())[1].lstrip('.')
    text = ""
    try:
        if ext in _PLAIN_TEXT_EXTS or ext == "":
            if ext == "":  # extensionless: skip binary blobs
                with open(file_path, "rb") as fb:
                    if b"\x00" in fb.read(8192): return ""
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f: raw_content = f.read(MAX_TEXT_READ_BYTES)
            if ext == "html" and HAS_BS4: text = BeautifulSoup(raw_content, "html.parser").get_text(separator="\n", strip=True)
            else: text = re.sub(r'<[^>]+>', ' ', raw_content) if ext == "html" else raw_content
            if redact_code and ext in _CODE_EXTS: text = redact_secrets(text)  # source/config can embed credentials
        elif ext == "pdf":
            if HAS_PDFPLUMBER:
                with pdfplumber.open(file_path) as pdf: text = "\n".join([page.extract_text() or "" for page in pdf.pages[:MAX_PDF_PAGES]])
            elif HAS_FITZ:  # no pdfplumber: still get the native text layer
                with fitz.open(file_path) as pdf_file:
                    text = "\n".join([pdf_file[i].get_text() for i in range(min(len(pdf_file), MAX_PDF_PAGES))])
            if HAS_FITZ and HAS_PYTESSERACT:
                try:
                    seen_imgs = set()  # same logo/stamp repeated on every page is OCR'd once
                    with fitz.open(file_path) as pdf_file:
                        for page_index in range(min(len(pdf_file), 10)):
                            page_text = pdf_file[page_index].get_text().strip()
                            if len(page_text) < 50: # Only OCR if native text is sparse
                                for img in pdf_file[page_index].get_images(full=True):
                                    xref = img[0]
                                    base_image = pdf_file.extract_image(xref)
                                    img_digest = hashlib.sha256(base_image["image"]).hexdigest()
                                    if img_digest in seen_imgs: continue
                                    seen_imgs.add(img_digest)
                                    img_obj = Image.open(io.BytesIO(base_image["image"]))
                                    text += f"\n[OCR from PDF Image Page {page_index+1}]:\n{pytesseract.image_to_string(img_obj).strip()}\n"
                except Exception as e: logger.warning(f"PDF Extraction Error: {e}")
        elif ext in ["docx", "xlsx"]:
            if ext == "docx" and HAS_DOCX:
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                for t_idx, table in enumerate(doc.tables):
                    for r_idx, row in enumerate(table.rows):
                        row_data = " | ".join([c.text.strip() for c in row.cells if c.text.strip()])
                        if row_data: text += f"\nTable {t_idx+1} Row {r_idx+1}: {row_data}"
            if ext == "xlsx" and HAS_OPENPYXL:
                current_sheet = None
                for sheet_title, row_num, cells in _xlsx_rows(file_path):
                    if sheet_title != current_sheet:
                        text += f"\nSheet: {sheet_title}\n"
                        current_sheet = sheet_title
                    text += f"Row {row_num}: " + " ¦ ".join(t.replace("|", "¦") for _, t in cells) + "\n"
            if HAS_PYTESSERACT:
                try:
                    with zipfile.ZipFile(file_path, 'r') as z:
                        for item in z.namelist()[:1000]:
                            if item.startswith('word/media/') or item.startswith('xl/media/'):
                                info = z.getinfo(item)
                                if info.file_size > 15 * 1024 * 1024: continue # Skip huge images
                                with z.open(info) as img_file:
                                    img_obj = Image.open(io.BytesIO(img_file.read(15 * 1024 * 1024)))
                                    text += f"\n[OCR Embedded Image - {item}]:\n{pytesseract.image_to_string(img_obj).strip()}\n"
                except Exception: pass
        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "img"]:
            if HAS_PYTESSERACT:
                try: text += pytesseract.image_to_string(Image.open(file_path))
                except Exception: pass
            vision_report = analyze_image_with_vision(file_path)
            if vision_report: text = vision_report + "\n" + text
    except Exception as e: return f"[Extraction Error] {e}"
    return text

# --- NEW V2 LAYER: STRICT GRAPH SCHEMAS ---
# (Graph-only pydantic models. They are NOT the SQLAlchemy EvidenceNode/EvidenceEdge rows;
#  persistence maps each graph node to EvidenceNode(node_id, node_type, properties=<json-safe dict>).)
class GraphNodeSchema(BaseModel):
    node_id: str
    type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DocumentNodeSchema(GraphNodeSchema): type: str = "Document"; filename: str; source: str = "upload"; file_hash: str
class EvidenceNodeSchema(GraphNodeSchema): type: str = "Evidence"; text: str; provenance: dict
class ClaimNodeSchema(GraphNodeSchema): type: str = "Claim"; claim_type: str; value: str
class EntityNodeSchema(GraphNodeSchema): type: str = "Entity"; entity_type: str; value: str; is_valid: Optional[bool] = None; value_hash: Optional[str] = None; verification_status: Optional[str] = None; verification_method: Optional[str] = None; jurisdiction: Optional[str] = None; detection_basis: Optional[str] = None
class TransactionNodeSchema(GraphNodeSchema): type: str = "Transaction"; amount: float; currency: str = "Any"; label: Optional[str] = None; attributes: Optional[dict] = None
class PolicyNodeSchema(GraphNodeSchema): type: str = "Policy"; name: str
class PolicyRuleNodeSchema(GraphNodeSchema): type: str = "PolicyRule"; condition: str
class DecisionNodeSchema(GraphNodeSchema): type: str = "Decision"; verdict: str; rule_id: Optional[str] = None; rationale: Optional[str] = None
class RiskNodeSchema(GraphNodeSchema): type: str = "Risk"; severity: str; rule_id: Optional[str] = None

# Relations used by the pipeline:
#   Evidence --DERIVED_FROM--> Document      Document --LINKED_FROM--> Document
#   Evidence --SUPPORTS-->     Entity/Claim/Transaction/Decision
#   Evidence --CONTRADICTS-->  Transaction   (heuristic, see detect_contradictions)
#   Entity/Transaction/Evidence --VIOLATES|SATISFIES--> PolicyRule   (deterministic rule engine)
#   Decision --EVALUATES--> PolicyRule   Decision --BELONGS_TO--> Policy   Decision --HAS_RISK--> Risk
#   PolicyRule --BELONGS_TO--> Policy
ALLOWED_RELATIONS = ["DERIVED_FROM", "LINKED_FROM", "SUPPORTS", "CONTRADICTS", "VIOLATES", "SATISFIES", "EVALUATES", "HAS_RISK", "BELONGS_TO", "CROSS_DOCUMENT_LINK"]
#   Document --CROSS_DOCUMENT_LINK--> Document   (Phase A cross-document linking; see link_cross_documents. Not traversed by V2 retrieval; makes no compliance decision)

ALLOWED_NODE_TYPES = ["Document", "Evidence", "Entity", "Transaction", "Policy", "PolicyRule", "Claim", "Decision", "Risk"]

# --- V2 LAYER 1: SPATIALLY AWARE PROVENANCE EXTRACTION ---
FALLBACK_CHUNK_LINES = max(1, _env_int("FALLBACK_CHUNK_LINES", 40))
_IMAGE_EXTS = ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "img"]

def _fact(text: str, location: str, method: str, conf_value: Optional[float] = None, conf_basis: str = "not_measured", **extra) -> Dict[str, Any]:
    """Provenance fact. 'confidence_metadata' stays 'uncalibrated' (backward compatible).
    confidence_value/confidence_basis carry what was actually MEASURED:
      - native text/cell layers: 1.0, basis 'native_*_exact' (no recognition step, so no recognition error; NOT a correctness probability)
      - OCR: mean word confidence reported by Tesseract (0-1), basis 'tesseract_mean_word_conf' (engine-reported, not calibrated to ground truth)
      - otherwise None / 'not_measured'."""
    fact = {"text": text, "location": location, "method": method, "confidence_metadata": "uncalibrated",
            "confidence_value": conf_value, "confidence_basis": conf_basis}
    fact.update(extra)
    return fact

def _ocr_image(img_obj) -> Tuple[str, Optional[float]]:
    """OCR text + mean word confidence (0-1) as reported by Tesseract. Confidence None if unavailable."""
    text = pytesseract.image_to_string(img_obj).strip()
    conf = None
    try:
        data = pytesseract.image_to_data(img_obj, output_type=pytesseract.Output.DICT)
        vals = [float(c) for c, t in zip(data.get("conf", []), data.get("text", [])) if str(t).strip() and float(c) >= 0]
        if vals: conf = round(sum(vals) / len(vals) / 100.0, 4)
    except Exception: pass
    return text, conf

def _line_chunk_facts(text: str, suffix: str = "") -> List[Dict[str, Any]]:
    """Plain-text fallback with real spatial info: line ranges (no more 'unavailable')."""
    lines = text.splitlines()
    facts = []
    for start in range(0, len(lines), FALLBACK_CHUNK_LINES):
        block = lines[start:start + FALLBACK_CHUNK_LINES]
        block_text = "\n".join(block)
        if block_text.strip():
            facts.append(_fact(block_text, f"Lines {start+1}-{start+len(block)}{suffix}", "text_line_chunk", 1.0, "native_text_exact"))
    return facts

def extract_provenance_facts(file_path: str) -> List[Dict[str, Any]]:
    if not file_path or not os.path.exists(file_path): return []
    if is_sensitive_file(file_path): return []
    filename = os.path.basename(file_path).lower()
    ext = os.path.splitext(filename)[1].lstrip('.')
    facts = []
    try:
        if ext == "pdf" and HAS_FITZ:
            with fitz.open(file_path) as doc:
                seen_ocr = set()  # dedupe: identical OCR text (repeated logos/stamps) recorded once
                uncaptured_pages: List[int] = []  # pages that produced no Evidence: reported on the Document, never silently dropped
                for page_index in range(min(len(doc), 50)):
                    page = doc[page_index]
                    text = page.get_text().strip()
                    _before = len(facts)
                    if len(text) > 50:
                        facts.append(_fact(text, f"Page {page_index+1}", "pymupdf_native", 1.0, "native_text_exact"))
                    elif HAS_PYTESSERACT and page_index < 10:
                        for img in page.get_images(full=True):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                img_obj = Image.open(io.BytesIO(base_image["image"]))
                                ocr_text, ocr_conf = _ocr_image(img_obj)
                                if not ocr_text: continue
                                ocr_key = hashlib.sha256(ocr_text.encode("utf-8", "ignore")).hexdigest()
                                if ocr_key in seen_ocr: continue
                                seen_ocr.add(ocr_key)
                                facts.append(_fact(ocr_text, f"Page {page_index+1} (Image)", "pytesseract_embedded", ocr_conf, "tesseract_mean_word_conf" if ocr_conf is not None else "not_measured"))
                            except Exception: pass
                    if len(facts) == _before: uncaptured_pages.append(page_index + 1)
                if uncaptured_pages:
                    facts.append(_fact(f"No text/OCR evidence was extracted from page(s): {uncaptured_pages[:50]}. Absence of text there is not proof of absence of content.", "unavailable", "extraction_note"))
                if len(doc) > 50:
                    facts.append(_fact(f"PDF has {len(doc)} pages; only the first 50 were examined, pages 51-{len(doc)} were NOT extracted.", "unavailable", "extraction_note"))
            return facts

        if ext == "pdf" and HAS_PDFPLUMBER:  # no PyMuPDF: still keep page-level provenance
            with pdfplumber.open(file_path) as pdf:
                for page_index, page in enumerate(pdf.pages[:50]):
                    text = (page.extract_text() or "").strip()
                    if text: facts.append(_fact(text, f"Page {page_index+1}", "pdfplumber_native", 1.0, "native_text_exact"))
                if len(pdf.pages) > 50:
                    facts.append(_fact(f"PDF has {len(pdf.pages)} pages; only the first 50 were examined, the rest were NOT extracted.", "unavailable", "extraction_note"))
            if facts: return facts

        if ext == "docx" and HAS_DOCX:
            doc = docx.Document(file_path)
            section = None  # nearest preceding Heading/Title paragraph actually present in the file; never invented
            for i, p in enumerate(doc.paragraphs):
                if p.text.strip():
                    try: sname = (p.style.name or "").lower()
                    except Exception: sname = ""
                    if sname.startswith("heading") or sname == "title": section = p.text.strip()[:120]
                    extra = {"section": section} if section else {}
                    facts.append(_fact(p.text, f"Paragraph {i+1}", "docx_native", 1.0, "native_text_exact", **extra))
            for t_idx, table in enumerate(doc.tables):
                for r_idx, row in enumerate(table.rows):
                    row_data = " | ".join([c.text.strip() for c in row.cells if c.text.strip()])
                    if row_data: facts.append(_fact(row_data, f"Table {t_idx+1}, Row {r_idx+1}", "docx_table", 1.0, "native_text_exact"))
            return facts

        if ext == "xlsx" and HAS_OPENPYXL:
            for sheet_title, row_num, cells in _xlsx_rows(file_path):
                coords = [c for c, _ in cells]
                row_text = " | ".join(t for _, t in cells)
                facts.append(_fact(row_text, f"{sheet_title}, Row {row_num}", "openpyxl_cells", 1.0, "native_cell_value",
                                   cells=coords[:200], cell_range=f"{coords[0]}:{coords[-1]}", sheet=sheet_title))
            return facts

        fallback_text = extract_text_from_file(file_path)
        if fallback_text.strip():
            if ext in _PLAIN_TEXT_EXTS or ext == "":
                facts.extend(_line_chunk_facts(fallback_text, " (extracted text)" if ext == "html" else ""))
            elif ext in _IMAGE_EXTS:
                conf, basis = None, "not_measured"
                if HAS_PYTESSERACT:
                    try:
                        _, conf = _ocr_image(Image.open(file_path))
                        if conf is not None: basis = "tesseract_mean_word_conf"
                    except Exception: pass
                facts.append(_fact(fallback_text, "Image (whole)", "image_ocr", conf, basis))
            else:
                facts.append(_fact(fallback_text, "unavailable", "v1_fallback_extractor"))
    except Exception as e:
        logger.error(f"Provenance Extraction Error: {e}")
        facts.append(_fact(f"Error: {e}", "unavailable", "error"))
    return facts

_STOP_WORDS = {"the", "a", "an", "is", "of", "and", "to", "in", "for", "on", "with", "this", "that"}

def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r'\w+', text.lower()) if w not in _STOP_WORDS]

def compute_lexical_overlap(query: str, text: str) -> float:
    """Legacy single-pair score (kept for backward compatibility). Raw heuristic, NOT a probability/confidence.
    The pipeline now ranks with bm25_scores()."""
    q_words = _tokenize(query)
    t_words = _tokenize(text)
    if not q_words or not t_words: return 0.0

    q_counts = Counter(q_words)
    t_counts = Counter(t_words)
    intersection = set(q_counts.keys()) & set(t_counts.keys())
    if not intersection: return 0.0

    numerator = sum(q_counts[w] * t_counts[w] for w in intersection)
    return numerator / (math.log(len(t_words) + 1) + 1)

def bm25_scores(query: str, texts: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Okapi BM25 over the evidence corpus (IDF + saturated TF + length normalisation).
    Scores are RELATIVE ranks within one corpus. Not probabilities, not confidence."""
    n_docs = len(texts)
    q_terms = set(_tokenize(query))
    if n_docs == 0 or not q_terms: return [0.0] * n_docs
    tokenized = [_tokenize(t) for t in texts]
    avgdl = (sum(len(t) for t in tokenized) / n_docs) or 1.0
    tfs = [Counter(t) for t in tokenized]
    df = {term: sum(1 for tf in tfs if term in tf) for term in q_terms}
    scores = []
    for tokens, tf in zip(tokenized, tfs):
        dl = len(tokens)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0: continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores

# --- NEW V2 LAYER 2: GRAPH QUERIES ---
# Relation conventions (all edges point FROM the derived/supporting node TO its source/target):
#   Evidence  --SUPPORTS-->     Entity / Claim / Transaction / Decision
#   Evidence  --DERIVED_FROM--> Document          (extraction provenance)
#   Document  --LINKED_FROM-->  Document          (embedded doc fetched from a parent doc's link)
# Reaching the original document from ANY node: X <-SUPPORTS- Evidence -DERIVED_FROM-> Document
# (query_documents_supporting_claim). Expanded evidence keeps its own DERIVED_FROM edge, so _describe_evidence
# always reports the true source document, never an intermediate evidence node.
LINEAGE_RELATIONS = {"DERIVED_FROM", "LINKED_FROM"}

def query_documents_supporting_claim(G: nx.MultiDiGraph, target_id: str) -> List[Dict[str, Any]]:
    """Strict Semantic Traversal: Claim/Entity/Transaction/Decision <-[SUPPORTS]- Evidence -[DERIVED_FROM]-> Document.
    Direction verified against the ingestion code: Evidence --SUPPORTS--> target, so target's IN-edges are inspected.
    Evidence links directly to its source Document, so no recursion (and no leaking through LINKED_FROM to parents)."""
    docs_dict = {}
    if G.has_node(target_id):
        for u, _, d1 in G.in_edges(target_id, data=True):
            if d1.get("relation") == "SUPPORTS" and G.nodes[u].get("type") == "Evidence":
                for _, v, d2 in G.out_edges(u, data=True):
                    if d2.get("relation") == "DERIVED_FROM" and G.nodes[v].get("type") == "Document":
                        docs_dict[f"{v}|{u}"] = {"document": G.nodes[v], "evidence": G.nodes[u]}
    return list(docs_dict.values())

def query_evidence_lineage(G: nx.MultiDiGraph, start_node_id: str) -> list:
    """Paths from a node to source Document nodes. Two kinds of hop are kept apart in every result:
      semantic hops   (SUPPORTS, VIOLATES, SATISFIES, EVALUATES, HAS_RISK ...): WHY the evidence relates to the node; NOT source provenance;
      provenance hops (DERIVED_FROM Evidence->Document, LINKED_FROM Document->parent Document): the traceable source-document path.
    Start node types: Decision (-> query_decision_lineage), Risk (via the Decision that HAS_RISK it), Claim / Entity / Transaction
    (Evidence --SUPPORTS--> node, then provenance), Evidence / Document (provenance only). Other types return no paths (nothing is invented)."""
    lineages: list = []
    if not G.has_node(start_node_id): return lineages
    stype = G.nodes[start_node_id].get("type")
    if stype == "Decision":  # Decision has no provenance out-edges; use dedicated trace
        return [{"path": p["path"], "kind": p["kind"], "evidence_id": p["evidence_id"], "document_id": p["document_id"],
                 "semantic_part": p.get("semantic_part"), "provenance_part": p.get("provenance_part")}
                for p in query_decision_lineage(G, start_node_id)["lineage_paths"]]
    if stype == "Risk":
        for dec in _in_edge_sources(G, start_node_id, "HAS_RISK", "Decision"):
            for p in query_decision_lineage(G, dec["node_id"])["lineage_paths"]:
                lineages.append({"path": [start_node_id] + p["path"], "kind": p["kind"], "evidence_id": p["evidence_id"], "document_id": p["document_id"],
                                 "semantic_part": [start_node_id] + (p.get("semantic_part") or []), "provenance_part": p.get("provenance_part")})
        return lineages
    lineage_view = nx.subgraph_view(G, filter_edge=lambda u, v, k: G.edges[u, v, k].get("relation") in LINEAGE_RELATIONS)
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Document" and n != start_node_id]
    seen_paths = set()
    if stype in ("Claim", "Entity", "Transaction"):
        starts = [(e, [start_node_id]) for e in _supporting_evidence_ids(G, start_node_id)]  # semantic hop first: Evidence --SUPPORTS--> node
    else:
        starts = [(start_node_id, [])]
    for ev_start, prefix in starts:
        for target in doc_nodes:
            try:
                for path in nx.all_simple_paths(lineage_view, source=ev_start, target=target, cutoff=5):
                    key = tuple(prefix + path)
                    if key in seen_paths: continue  # multigraph can yield duplicates
                    seen_paths.add(key)
                    lineages.append({"path": prefix + path, "semantic_part": list(prefix), "provenance_part": list(path)})
            except (nx.NetworkXException, KeyError): continue
    return lineages

_LINEAGE_DISCOVERY_LIMIT = ("Evidence discovery covers only the channels listed in discovery_channels. The graph does not record which supporting items "
                            "SHOULD exist for a decision (the rule engine's evidence can be partial, and 'absence of text' evidence is never stored), so this search "
                            "cannot establish that EVERY relevant supporting item was found. Source-path status describes only the evidence that WAS discovered.")

def query_decision_lineage(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Decision -> evidence -> source Document, full structured trace.
    SEMANTIC links (EVALUATES, BELONGS_TO, VIOLATES/SATISFIES, SUPPORTS) are reported separately from PROVENANCE links
    (DERIVED_FROM Evidence->Document, LINKED_FROM Document->parent Document). Evidence is gathered from: (a) Evidence --SUPPORTS--> Decision,
    (b) the Decision's stored `evidence_used`, (c) evidence supporting the Decision's stored basis nodes (claim/entity/transaction).
    Nothing is invented: a missing document/edge/node is listed in `gaps`.
    TWO SEPARATE QUESTIONS are reported (they are NOT the same):
      1. status / source_path_status: does every evidence item that WAS discovered have a DERIVED_FROM source-document path?
         status: DISCOVERED_EVIDENCE_TRACED | DISCOVERED_EVIDENCE_PARTIALLY_TRACED | NO_EVIDENCE | NOT_FOUND.
         'TRACED' never means the decision's evidence set is complete.
      2. discovery_exhaustiveness_established: always False. This search cannot prove it found every relevant supporting item
         (see discovery_limitation / discovery_channels)."""
    res: Dict[str, Any] = {"decision_id": decision_id, "found": False, "verdict": None, "status": "NOT_FOUND", "source_path_status": "NOT_FOUND",
                           "discovery_exhaustiveness_established": False, "discovery_limitation": _LINEAGE_DISCOVERY_LIMIT,
                           "tracing_complete_for_discovered_evidence": False,
                           "discovery_channels": ["Evidence --SUPPORTS--> Decision edges", "Decision.evidence_used (stored by the rule engine)",
                                                  "evidence supporting Decision.basis_node_ids (stored basis nodes)",
                                                  "evidence supporting nodes that have a VIOLATES / SATISFIES edge to the evaluated PolicyRule (Claim / Entity / Transaction)"],
                           "rule": None, "policies": [], "risks": [], "basis_nodes": [], "evidence": [], "documents": [], "lineage_paths": [],
                           "semantic_links": [], "contradicting_evidence": [], "tracing": {}, "gaps": [],
                           "lineage_note": ("semantic_links explain WHY nodes relate to the decision (support / violation / satisfaction / evaluation / risk); only lineage_paths' provenance_part "
                                            "(Evidence -DERIVED_FROM-> Document, Document -LINKED_FROM-> parent) is a traceable source-document path. contradicting_evidence is reported "
                                            "separately: it is a (possibly heuristic) discrepancy, never support and never proof.")}
    if not G.has_node(decision_id) or G.nodes[decision_id].get("type") != "Decision":
        res["gaps"].append("start node missing or not a Decision")
        return res
    dd = G.nodes[decision_id]
    res.update(found=True, verdict=dd.get("verdict"))

    sem_seen: set = set()
    def _sem(rel: str, frm: str, to: str, **extra) -> None:  # semantic (NOT provenance) link; only recorded for edges that exist in the graph
        if (rel, frm, to) in sem_seen: return
        sem_seen.add((rel, frm, to))
        res["semantic_links"].append({"relation": rel, "from": frm, "from_type": G.nodes[frm].get("type"), "to": to, "to_type": G.nodes[to].get("type"),
                                      "kind": "semantic", "is_source_provenance": False, **extra})

    # --- semantic links (only edges that exist) ---
    rule_id: Optional[str] = None
    for _, v, d in G.out_edges(decision_id, data=True):
        if d.get("relation") == "EVALUATES" and G.nodes[v].get("type") == "PolicyRule":
            _sem("EVALUATES", decision_id, v)
            if rule_id is None: rule_id = v
        elif d.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy":
            _sem("BELONGS_TO", decision_id, v)
            if v not in res["policies"]: res["policies"].append(v)
        elif d.get("relation") == "HAS_RISK" and G.nodes[v].get("type") == "Risk":
            _sem("HAS_RISK", decision_id, v)
            res["risks"].append({"node_id": v, "severity": G.nodes[v].get("severity"), "rule_id": G.nodes[v].get("rule_id")})
    if rule_id:
        rn = G.nodes[rule_id]
        res["rule"] = {"node_id": rule_id, "condition": rn.get("condition"), "source_file": rn.get("source_file"), "source_location": rn.get("source_location")}
        for _, pol, d in G.out_edges(rule_id, data=True):
            if d.get("relation") == "BELONGS_TO" and G.nodes[pol].get("type") == "Policy":
                _sem("BELONGS_TO", rule_id, pol)
                if pol not in res["policies"]: res["policies"].append(pol)
    else:
        res["gaps"].append("decision has no EVALUATES -> PolicyRule edge")

    # --- policy-definition provenance: PolicyRule / Policy <-SUPPORTS- Evidence(rulebook) -DERIVED_FROM-> rulebook Document ---
    res["rule_source"], res["policy_source"], res["policy_documents"] = [], [], []
    def _definition_sources(target: str, bucket: str) -> None:
        found = False
        for s_ in _in_edge_sources(G, target, "SUPPORTS", "Evidence"):
            if not _is_policy_source_evidence(s_["node"]): continue
            found = True
            _sem("SUPPORTS", s_["node_id"], target, definition_source=True)
            docs_ = _evidence_source_docs(G, s_["node_id"])
            prov_ = s_["node"].get("provenance") or {}
            res[bucket].append({"target_id": target, "evidence_id": s_["node_id"], "documents": docs_, "provenance": prov_,
                                "path": [decision_id] + ([rule_id] if (rule_id and target != rule_id) else []) + [target, s_["node_id"]] + [x["document_id"] for x in docs_][:1]})
            for x_ in docs_:
                if not any(p_["document_id"] == x_["document_id"] for p_ in res["policy_documents"]): res["policy_documents"].append({"document_id": x_["document_id"], "filename": x_["filename"]})
                pth = [decision_id] + ([rule_id] if (rule_id and target != rule_id) else []) + [target, s_["node_id"], x_["document_id"]]
                res["lineage_paths"].append({"kind": "policy_definition", "path": pth, "evidence_id": s_["node_id"], "document_id": x_["document_id"],
                                             "semantic_part": pth[:-2], "provenance_part": [s_["node_id"], x_["document_id"]]})
        if not found: res["gaps"].append(f"{G.nodes[target].get('type')} {target} has no rulebook source Evidence (definition not traced to a rulebook Document)")
    if rule_id: _definition_sources(rule_id, "rule_source")
    for pol_ in res["policies"]: _definition_sources(pol_, "policy_source")

    # --- evidence collection (ordered, de-duplicated) ---
    ev: Dict[str, Dict[str, Any]] = {}
    paths_seen: set = set()
    def _add_ev(ev_id: str, origin: str, via: Optional[str] = None) -> None:
        item = ev.setdefault(ev_id, {"evidence_id": ev_id, "origins": [], "via_nodes": []})
        if origin not in item["origins"]: item["origins"].append(origin)
        if via and via not in item["via_nodes"]: item["via_nodes"].append(via)
    for s in _in_edge_sources(G, decision_id, "SUPPORTS", "Evidence"):
        _add_ev(s["node_id"], "SUPPORTS_edge")
        _sem("SUPPORTS", s["node_id"], decision_id)
    for e_id in (dd.get("evidence_used") or []):
        if G.has_node(e_id) and G.nodes[e_id].get("type") == "Evidence": _add_ev(e_id, "stored_evidence_used")
        else: res["gaps"].append(f"stored evidence reference {e_id} not found as Evidence node")
    basis_ids: List[str] = list(dict.fromkeys(dd.get("basis_node_ids") or []))
    stored_basis = set(basis_ids)
    if rule_id:  # nodes linked to the evaluated rule by VIOLATES / SATISFIES even if not stored on the Decision
        for rname in ("VIOLATES", "SATISFIES"):
            for src_ in _in_edge_sources(G, rule_id, rname):
                if src_["node_id"] not in stored_basis and src_["node_id"] not in basis_ids: basis_ids.append(src_["node_id"])
    for b_id in basis_ids:
        if not G.has_node(b_id):
            res["gaps"].append(f"stored basis node {b_id} not found in graph")
            continue
        rel = None
        if rule_id:
            for rname in ("VIOLATES", "SATISFIES"):
                if any(d.get("relation") == rname for d in (G.get_edge_data(b_id, rule_id) or {}).values()): rel = rname; break
        if rel and rule_id: _sem(rel, b_id, rule_id)
        elif rule_id: res["gaps"].append(f"basis node {b_id} has no VIOLATES/SATISFIES edge to the evaluated rule")
        b_evs = _supporting_evidence_ids(G, b_id)
        origin = "via_basis_node" if b_id in stored_basis else "via_rule_edge_node"
        res["basis_nodes"].append({"node_id": b_id, "type": G.nodes[b_id].get("type"), "relation_to_rule": rel, "evidence_ids": b_evs, "discovered_via": "stored_basis" if b_id in stored_basis else "rule_edge"})
        for e_id in b_evs:
            _add_ev(e_id, origin, b_id)
            if e_id != b_id: _sem("SUPPORTS", e_id, b_id)
        if not b_evs: res["gaps"].append(f"basis node {b_id} ({G.nodes[b_id].get('type')}) has no supporting Evidence")
        if G.nodes[b_id].get("type") == "Transaction":  # evidence that CONTRADICTS a basis transaction: heuristic discrepancy, never support
            for c in query_contradictory_evidence(G, b_id):
                if c["direction"] != "incoming": continue
                e = c["edge"]
                res["contradicting_evidence"].append({"transaction_id": b_id, "evidence_id": c["node_id"], "match_strength": e.get("match_strength"), "heuristic": bool(e.get("heuristic")),
                                                      "method": e.get("method"), "status": e.get("status"), "qualification": e.get("qualification"),
                                                      "documents": _evidence_source_docs(G, c["node_id"])})
                _sem("CONTRADICTS", c["node_id"], b_id, heuristic=bool(e.get("heuristic")))

    # --- provenance: Evidence -DERIVED_FROM-> Document (+ LINKED_FROM parents, cycle-safe) ---
    docs: Dict[str, Dict[str, Any]] = {}
    def _parents(doc_id: str) -> List[str]:
        chain, seen, stack = [], {doc_id}, [doc_id]
        while stack:
            cur = stack.pop()
            for _, p, d in G.out_edges(cur, data=True):
                if d.get("relation") == "LINKED_FROM" and p not in seen and G.nodes[p].get("type") == "Document":
                    seen.add(p); chain.append(p); stack.append(p)
        return chain
    for ev_id, item in ev.items():
        nd = G.nodes[ev_id]
        item["provenance"] = nd.get("provenance")  # method / confidence_value / confidence_basis preserved as stored
        item["context_only"] = bool(nd.get("context_only"))
        item["documents"] = []
        for _, doc, d in G.out_edges(ev_id, data=True):
            if d.get("relation") != "DERIVED_FROM" or G.nodes[doc].get("type") != "Document": continue
            if any(x["document_id"] == doc and x["location"] == d.get("location") for x in item["documents"]): continue
            dn = G.nodes[doc]
            item["documents"].append({"document_id": doc, "filename": dn.get("filename"), "location": d.get("location"), "file_hash": dn.get("file_hash"),
                                      "source": dn.get("source"), "linked_from": _parents(doc)})
            docs.setdefault(doc, {"document_id": doc, "filename": dn.get("filename"), "file_hash": dn.get("file_hash"), "source": dn.get("source")})
            for kind, base in (("direct", [decision_id]), ("via_basis", [decision_id] + ([rule_id] if rule_id else []))):
                if kind == "direct" and "SUPPORTS_edge" not in item["origins"] and "stored_evidence_used" not in item["origins"]: continue
                if kind == "via_basis" and not item["via_nodes"]: continue
                for via in (item["via_nodes"] if kind == "via_basis" else [None]):
                    path = tuple(base + ([via] if via else []) + [ev_id, doc])
                    if path not in paths_seen:
                        paths_seen.add(path)
                        res["lineage_paths"].append({"kind": kind, "path": list(path), "evidence_id": ev_id, "document_id": doc,
                                                     "semantic_part": list(path[:-1]),  # decision/rule/basis -> evidence (semantic: why it is related)
                                                     "provenance_part": [ev_id, doc] + _parents(doc)})  # evidence -DERIVED_FROM-> document -LINKED_FROM-> parents
        if not item["documents"]: res["gaps"].append(f"evidence {ev_id} has no DERIVED_FROM source document")
    res["evidence"] = list(ev.values())
    res["documents"] = list(docs.values())

    # --- status (source-path status of DISCOVERED evidence only; exhaustiveness is separately and always reported as not established) ---
    if not ev:
        res["status"] = res["source_path_status"] = "NO_EVIDENCE"
        if res["verdict"] in ("UNEVALUATED", "INCONCLUSIVE", "NOT_APPLICABLE"): res["gaps"].append(f"no supporting evidence recorded; none expected for verdict {res['verdict']}")
        else: res["gaps"].append("no evidence linked to this decision")
    else:
        res["status"] = res["source_path_status"] = ("DISCOVERED_EVIDENCE_TRACED" if all(i["documents"] for i in ev.values()) else "DISCOVERED_EVIDENCE_PARTIALLY_TRACED")
    traced_n = sum(1 for i in ev.values() if i["documents"])
    res["tracing"] = {"discovered_evidence": len(ev), "traced_to_source_document": traced_n, "untraced": len(ev) - traced_n,
                      "scope": "tracing completeness covers ONLY the discovered evidence; it says nothing about whether all relevant evidence was discovered"}
    res["tracing_complete_for_discovered_evidence"] = bool(ev) and traced_n == len(ev)  # separate from discovery (discovery_exhaustiveness_established stays False)
    res["chain"] = {"decision_to_rule": bool(rule_id), "rule_to_policy": bool(rule_id) and bool(res["policies"]),
                    "rule_to_source_document": bool(res["rule_source"]) and all(x["documents"] for x in res["rule_source"]),
                    "policy_to_source_document": bool(res["policy_source"]) and all(x["documents"] for x in res["policy_source"]),
                    "decision_to_evidence": bool(ev) if res["verdict"] in ("VIOLATION", "SATISFIED") else None,  # None = no evidence expected for this verdict
                    "evidence_to_document": res["tracing_complete_for_discovered_evidence"] if ev else None,
                    "risk_present_when_violation": (bool(res["risks"]) if res["verdict"] == "VIOLATION" else None)}
    res["chain_complete"] = all(v for v in res["chain"].values() if v is not None) and bool(rule_id)
    return res

def _in_edge_sources(G: nx.MultiDiGraph, node_id: str, relation: str, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    if not G.has_node(node_id): return out
    for u, _, d in G.in_edges(node_id, data=True):
        if d.get("relation") == relation and (source_type is None or G.nodes[u].get("type") == source_type):
            out.append({"node_id": u, "node": G.nodes[u]})
    return out

# Graph traversals now backed by data created by evaluate_policy_rules() / detect_contradictions().
def query_evidence_supporting_decision(G: nx.MultiDiGraph, decision_id: str) -> list:
    """Evidence --SUPPORTS--> Decision."""
    return _in_edge_sources(G, decision_id, "SUPPORTS", "Evidence")

def query_contradictory_evidence(G: nx.MultiDiGraph, node_id: str) -> list:
    """Works for BOTH ends of a heuristic Evidence --CONTRADICTS--> Transaction edge. Each result names the counterpart:
       - node is a Transaction -> incoming edges: direction='incoming', counterpart_role='contradicting_evidence'
       - node is Evidence      -> outgoing edges: direction='outgoing', counterpart_role='contradicted_transaction'
    Keys: node_id (counterpart), node, direction, counterpart_role, edge (method/label/amounts/heuristic). Possible, not proven."""
    out: list = []
    if not G.has_node(node_id): return out
    for u, _, d in G.in_edges(node_id, data=True):
        if d.get("relation") == "CONTRADICTS":
            out.append({"node_id": u, "node": G.nodes[u], "direction": "incoming", "counterpart_role": "contradicting_evidence", "edge": dict(d)})
    for _, v, d in G.out_edges(node_id, data=True):
        if d.get("relation") == "CONTRADICTS":
            out.append({"node_id": v, "node": G.nodes[v], "direction": "outgoing", "counterpart_role": "contradicted_transaction", "edge": dict(d)})
    return out

def query_policy_violation_source(G: nx.MultiDiGraph, target_id: str) -> list:
    """Which PolicyRule(s) does target violate? target --VIOLATES--> PolicyRule. Also resolves:
       - target is a Decision with verdict VIOLATION  -> its EVALUATES rule
       - target is Evidence                           -> rules violated by nodes it SUPPORTS (via='SUPPORTS')"""
    results, seen = [], set()
    if not G.has_node(target_id): return results
    def add(rule_id, via):
        if rule_id not in seen:
            seen.add(rule_id)
            results.append({"node_id": rule_id, "node": G.nodes[rule_id], "via": via})
    for _, v, d in G.out_edges(target_id, data=True):
        if d.get("relation") == "VIOLATES" and G.nodes[v].get("type") == "PolicyRule": add(v, "VIOLATES")
    ttype = G.nodes[target_id].get("type")
    if ttype == "Decision" and G.nodes[target_id].get("verdict") == "VIOLATION":
        for _, v, d in G.out_edges(target_id, data=True):
            if d.get("relation") == "EVALUATES": add(v, "EVALUATES")
    if ttype == "Evidence":
        for _, mid, d in G.out_edges(target_id, data=True):
            if d.get("relation") != "SUPPORTS": continue
            for _, v, d2 in G.out_edges(mid, data=True):
                if d2.get("relation") == "VIOLATES" and G.nodes[v].get("type") == "PolicyRule": add(v, "SUPPORTS")
    return results

def expand_evidence_by_shared_entities(G: nx.MultiDiGraph, seed_ids: List[str], limit: int) -> List[Tuple[str, str, str]]:
    """Graph-structure retrieval: other Evidence supporting the same Entity (phone/GovID) as a seed.
    Returns (evidence_id, via_entity_id, seed_id)."""
    results: List[Tuple[str, str, str]] = []
    if limit <= 0: return results
    seen = set(seed_ids)
    for seed in seed_ids:
        for _, ent, d in G.out_edges(seed, data=True):
            if d.get("relation") != "SUPPORTS" or G.nodes[ent].get("type") != "Entity": continue
            for other in _in_edge_sources(G, ent, "SUPPORTS", "Evidence"):
                if other["node_id"] in seen: continue
                seen.add(other["node_id"])
                results.append((other["node_id"], ent, seed))
                if len(results) >= limit: return results
    return results

# --- CROSS-DOCUMENT LINKS (heuristic) ---
# Entity extraction only knew phones/GovIDs, so documents about the same person/item were never connected unless
# they shared a phone/GovID. This adds Entity(entity_type="Term") nodes for Title-Case phrases (e.g. "John Doe",
# "Server Hardware") that occur in Evidence from >= 2 DIFFERENT documents. Regex heuristic: ALL-CAPS or lowercase
# names are missed, and a shared phrase proves co-mention only. Every Term node / edge carries heuristic=True.
_TERM_PATTERN = re.compile(r"\b[A-Z][a-z]{1,}(?:[ \t][A-Z][a-z]{1,}){1,2}\b")
_TERM_SKIP_WORDS = {"the", "this", "that", "these", "those", "please", "dear", "date", "total", "page", "sheet", "row", "table", "amount", "name", "type"}
MAX_TERM_NODES = max(0, _env_int("MAX_TERM_NODES", 200))

def link_shared_terms(G: nx.MultiDiGraph) -> int:
    """Creates Term entities shared across documents + Evidence --SUPPORTS--> Term edges. Idempotent. Returns Term nodes created."""
    term_ev: Dict[str, set] = defaultdict(set)
    term_docs: Dict[str, set] = defaultdict(set)
    for ev_id, d in _investigation_evidence(G):
        docs = {x["document_id"] for x in _evidence_source_docs(G, ev_id)}
        if not docs: continue
        found: List[str] = []
        for m in _TERM_PATTERN.finditer(d.get("text", "") or ""):
            term = re.sub(r"\s+", " ", m.group(0))
            first = term.lower().split()[0]
            if first in _TERM_SKIP_WORDS or first in _STOP_WORDS: continue
            found.append(term)
        for term in list(dict.fromkeys(found))[:10]:
            term_ev[term].add(ev_id)
            term_docs[term] |= docs
    existing = {d.get("value") for _, d in _nodes_of_type(G, "Entity", "Term")}
    created = 0
    for term, evs in sorted(term_ev.items(), key=lambda kv: (-len(term_docs[kv[0]]), kv[0])):
        if len(term_docs[term]) < 2 or term in existing: continue
        if created >= MAX_TERM_NODES: break
        ent = EntityNodeSchema(node_id=_nid("term"), entity_type="Term", value=term, is_valid=None).model_dump()
        G.add_node(ent["node_id"], **ent, method="title_case_phrase_shared_across_documents", heuristic=True)
        for ev in sorted(evs):
            G.add_edge(ev, ent["node_id"], relation="SUPPORTS", method="shared_term", heuristic=True)
        created += 1
    return created

def _entity_label(G: nx.MultiDiGraph, ent: str) -> Any:
    """Display label for an Entity. GovID digits (even masked/last-4) are never shown in reports/context."""
    d = G.nodes[ent]
    if d.get("entity_type") == "GovID": return f"ID present, {(d.get('verification_status') or 'unverified').lower()}"
    return d.get("value")

def _node_brief(G: nx.MultiDiGraph, node_id: str) -> str:
    d = G.nodes[node_id]
    t = d.get("type")
    if t == "Transaction": return f"{node_id} (Transaction {d.get('currency')} {d.get('amount')}, label={d.get('label')}, amount_role={d.get('amount_role', 'not_classified')})"
    if t == "Entity":
        ver = f", verification={d.get('verification_status')}" if d.get("verification_status") else ""
        if d.get("entity_type") == "GovID": return f"{node_id} (Entity/GovID: {_entity_label(G, node_id)}{ver}; digits withheld)"
        return f"{node_id} (Entity/{d.get('entity_type')} '{d.get('value')}', valid={d.get('is_valid')}{ver})"
    if t == "Claim" and d.get("claim_type") == "PolicyListedAmount": return f"{node_id} (Claim PolicyListedAmount {d.get('value')} [POLICY CONTEXT ONLY: not a transaction; no employee association])"
    if t == "Claim": return f"{node_id} (Claim {d.get('value')})"
    return f"{node_id} ({t})"

def expand_evidence_by_graph(G: nx.MultiDiGraph, seed_ids: List[str], limit: int) -> List[Dict[str, Any]]:
    """Graph retrieval from lexical seeds. Two traversals, every hop recorded:
      (a) seed --SUPPORTS--> Entity <--SUPPORTS-- other Evidence   (Phone / GovID exact match, or heuristic Term)
      (b) seed --CONTRADICTS--> Transaction <--SUPPORTS-- other Evidence, and
          other Evidence --CONTRADICTS--> Transaction <--SUPPORTS-- seed
    Returns dicts: evidence_id, seed_id, via_node_id, via_type, via_label, relations (readable edges), heuristic."""
    results: List[Dict[str, Any]] = []
    if limit <= 0: return results
    seen = set(seed_ids)
    def push(oid, seed, via, via_type, via_label, relations, heuristic, link_method=None):
        seen.add(oid)
        if via_type == "Entity/Term": lim = "shared term only (co-mention): does NOT establish identity, transaction ownership, policy compliance or a violation"
        elif via_type == "Transaction": lim = "reached through a discrepancy edge: context for reconciliation, not proof of a contradiction or violation"
        else: lim = f"shared {via_type} value: context link; does not by itself establish identity, ownership, compliance or a violation"
        pol_ctx = bool(G.nodes[oid].get("context_only"))
        if pol_ctx: lim = ("POLICY CONTEXT ONLY: this evidence is policy-listed information; its amount is not a transaction record, it neither corroborates nor contradicts any employee transaction, "
                           "and no employee-to-policy-amount association is implied. Underlying link: " + lim)
        results.append({"evidence_id": oid, "seed_id": seed, "via_node_id": via, "via_type": via_type, "limitation": lim, "policy_context": pol_ctx,
                        "via_label": via_label, "relations": relations, "heuristic": heuristic,
                        "match_strength": "heuristic" if heuristic else "strong", "link_method": link_method})
        return len(results) >= limit
    def _cedge(u, v):
        for dd in (G.get_edge_data(u, v) or {}).values():
            if dd.get("relation") == "CONTRADICTS": return dd
        return {}
    for seed in seed_ids:
        for _, ent, d in list(G.out_edges(seed, data=True)):
            if d.get("relation") != "SUPPORTS" or G.nodes[ent].get("type") != "Entity": continue
            heur = bool(d.get("heuristic")) or bool(G.nodes[ent].get("heuristic"))
            for other in _in_edge_sources(G, ent, "SUPPORTS", "Evidence"):
                oid = other["node_id"]
                if oid in seen: continue
                if push(oid, seed, ent, f"Entity/{G.nodes[ent].get('entity_type')}", _entity_label(G, ent),
                        [f"{seed} --SUPPORTS--> {ent}", f"{oid} --SUPPORTS--> {ent}"], heur,
                        d.get("method") or f"shared_{G.nodes[ent].get('entity_type')}_exact_value"): return results
        for _, txn, d in list(G.out_edges(seed, data=True)):
            if d.get("relation") == "CONTRADICTS":
                for oid in _supporting_evidence_ids(G, txn):
                    if oid in seen: continue
                    if push(oid, seed, txn, "Transaction", f"{G.nodes[txn].get('currency')} {G.nodes[txn].get('amount')}",
                            [f"{seed} --CONTRADICTS--> {txn}", f"{oid} --SUPPORTS--> {txn}"], bool(d.get("heuristic")), d.get("method")): return results
            elif d.get("relation") == "SUPPORTS" and G.nodes[txn].get("type") == "Transaction":
                for other in _in_edge_sources(G, txn, "CONTRADICTS", "Evidence"):
                    oid = other["node_id"]
                    if oid in seen: continue
                    if push(oid, seed, txn, "Transaction", f"{G.nodes[txn].get('currency')} {G.nodes[txn].get('amount')}",
                            [f"{oid} --CONTRADICTS--> {txn}", f"{seed} --SUPPORTS--> {txn}"],
                            bool(_cedge(oid, txn).get("heuristic", True)), _cedge(oid, txn).get("method")): return results
    return results

# --- MULTI-HOP GRAPH TRAVERSAL (V2 retrieval; incoming AND outgoing edges; honest trace) ---
# expand_evidence_by_graph (above) is the legacy single-hop expansion. The V2 pipeline uses traverse_graph_context(): a breadth-first walk from the lexical
# seeds over edges in BOTH directions, up to V2_MAX_HOPS hops, so lineage such as
#   Evidence -SUPPORTS-> Transaction -VIOLATES-> PolicyRule <-EVALUATES- Decision -HAS_RISK-> Risk <-SUPPORTS- Evidence
#   Evidence -DERIVED_FROM-> Document -LINKED_FROM-> Document <-DERIVED_FROM- Evidence
# can contribute to the V2 context. EVERY edge looked at is recorded (examined; followed or not, with the reason), with its real relation, real direction and the
# direction it was traversed in. Hubs (Policy nodes, nodes above V2_TRAVERSAL_HUB_DEGREE), nodes at the hop limit, an exhausted edge budget and same-document
# Evidence (a Document is reached, but Evidence that only shares a Document is not expanded) are reported as NOT examined, never hidden.
_TRAVERSAL_NODE_TYPES = {"Decision", "Risk", "PolicyRule", "Policy", "Transaction", "Document"}
_SIBLING_RELS = {"VIOLATES", "SATISFIES"}

def _via_label(G: nx.MultiDiGraph, node_id: str) -> Any:
    d = G.nodes[node_id]
    if d.get("type") == "Entity": return _entity_label(G, node_id)
    if d.get("type") == "Transaction": return f"{d.get('currency')} {d.get('amount')}"
    return _node_brief(G, node_id)

def _traversal_node_detail(G: nx.MultiDiGraph, node_id: str) -> str:
    d = G.nodes[node_id]
    t = d.get("type")
    if t == "Decision": return f"{node_id} (Decision verdict={d.get('verdict')}; {str(d.get('rationale') or '')[:200]})"
    if t == "Risk": return f"{node_id} (Risk severity={d.get('severity')}, severity_source={d.get('severity_source')})"
    if t == "PolicyRule": return f"{node_id} (PolicyRule '{str(d.get('condition') or '')[:160]}' source {d.get('source_file')} @ {d.get('source_location')})"
    if t == "Policy": return f"{node_id} (Policy '{d.get('name')}' from {d.get('source_file')})"
    if t == "Document": return f"{node_id} (Document {d.get('filename')}, source={d.get('source')}{', RULEBOOK' if d.get('role') == 'RULEBOOK' else ''})"
    if t == "Evidence": return f"{node_id} (rulebook source Evidence @ {d.get('source_location')})"
    return _node_brief(G, node_id)

def traverse_graph_context(G: nx.MultiDiGraph, seed_ids: List[str], evidence_limit: int, max_hops: Optional[int] = None,
                           edge_budget: Optional[int] = None, node_limit: Optional[int] = None) -> Dict[str, Any]:
    """-> {"evidence": [...], "nodes": [...], "examined": [edge records], "path_edge_keys": set, "stats": {...}}.
    evidence items keep the keys of expand_evidence_by_graph (evidence_id, seed_id, via_node_id, via_type, via_label, relations, heuristic, match_strength,
    link_method, limitation, policy_context) plus hops and path. `examined` lists every edge looked at, in examination order, each with from/to/relation (true edge
    direction), direction (outgoing/incoming relative to the walk), hop, via (node it was reached from), followed and skip_reason."""
    max_hops = V2_MAX_HOPS if max_hops is None else max(1, int(max_hops))
    edge_budget = V2_TRAVERSAL_EDGE_BUDGET if edge_budget is None else max(1, int(edge_budget))
    node_limit = V2_NODE_CONTEXT_LIMIT if node_limit is None else max(0, int(node_limit))
    seeds = list(dict.fromkeys(x for x in (seed_ids or []) if G.has_node(x)))
    stats: Dict[str, Any] = {"seeds": len(seeds), "max_hops": max_hops, "edge_budget": edge_budget, "evidence_limit": evidence_limit, "node_limit": node_limit,
                             "edges_examined": 0, "outgoing_examined": 0, "incoming_examined": 0, "edges_followed": 0, "nodes_expanded": 0, "nodes_reached": 0,
                             "max_hop_reached": 0, "frontier_not_expanded": 0, "hubs_not_expanded": [], "budget_exhausted": False, "evidence_limit_reached": False,
                             "node_limit_reached": False, "stopped_because": "queue_exhausted"}
    res: Dict[str, Any] = {"evidence": [], "nodes": [], "examined": [], "path_edge_keys": set(), "stats": stats}
    if evidence_limit <= 0 or not seeds:
        stats["stopped_because"] = "expansion_disabled" if evidence_limit <= 0 else "no_seeds"
        return res
    seed_set = set(seeds)
    visited: Dict[str, Dict[str, Any]] = {s_: {"hop": 0, "path": [], "seed": s_} for s_ in seeds}
    queue = deque((s_, 0) for s_ in seeds)
    examined_keys: set = set()
    ev_items: List[str] = []
    node_items: List[str] = []
    done = False
    while queue and not done:
        cur, hop = queue.popleft()
        if hop >= max_hops:
            stats["frontier_not_expanded"] += 1
            continue
        ctype = G.nodes[cur].get("type")
        info = visited[cur]
        last_rel = info["path"][-1]["relation"] if info["path"] else None
        via_linked = any(e_["relation"] == "LINKED_FROM" for e_ in info["path"])
        outs = [(u, v, k, d, "outgoing", v) for u, v, k, d in G.out_edges(cur, keys=True, data=True)]
        ins = [(u, v, k, d, "incoming", u) for u, v, k, d in G.in_edges(cur, keys=True, data=True)]
        if ctype == "Document":  # provenance hub: only LINKED_FROM is followed; Evidence of a LINKED document only if it was reached through a LINKED_FROM hop
            cand = [c for c in outs + ins if c[3].get("relation") == "LINKED_FROM"]
            der_in = [c for c in ins if c[3].get("relation") == "DERIVED_FROM"] if via_linked else []
            if len(der_in) > V2_TRAVERSAL_HUB_DEGREE:
                stats["hubs_not_expanded"].append({"node_id": cur, "type": ctype, "degree": len(der_in), "note": "evidence of linked document not expanded (too many)"})
                der_in = []
            cand += der_in
        else:
            if ctype == "Policy" or (G.degree(cur) > V2_TRAVERSAL_HUB_DEGREE and cur not in seed_set):
                stats["hubs_not_expanded"].append({"node_id": cur, "type": ctype, "degree": G.degree(cur)})
                continue
            cand = outs + ins
        cand.sort(key=lambda c: (str(c[3].get("relation", "")), c[5], c[4]))
        stats["nodes_expanded"] += 1
        for u, v, k, d, direction, nb in cand:
            ekey = (u, v, k)
            if ekey in examined_keys: continue
            if len(res["examined"]) >= edge_budget:
                stats["budget_exhausted"] = True
                stats["stopped_because"] = "edge_budget_exhausted"
                done = True
                break
            examined_keys.add(ekey)
            rel = str(d.get("relation", ""))
            rec = {"from": u, "to": v, "relation": rel, "direction": direction, "hop": hop + 1, "heuristic": bool(d.get("heuristic")), "method": d.get("method"),
                   "via": cur, "reached": nb, "followed": False, "skip_reason": None}
            res["examined"].append(rec)
            stats["edges_examined"] += 1
            stats[f"{direction}_examined"] += 1
            nbd = G.nodes[nb]
            if nbd.get("type") == "SystemSchema": rec["skip_reason"] = "system node"; continue
            if nb in visited: rec["skip_reason"] = "node already reached"; continue
            if ctype == "PolicyRule" and rel in _SIBLING_RELS and direction == "incoming" and last_rel in _SIBLING_RELS:
                rec["skip_reason"] = "other nodes linked to the same rule are not expanded"
                continue
            rec["followed"] = True
            stats["edges_followed"] += 1
            visited[nb] = {"hop": hop + 1, "path": info["path"] + [rec], "seed": info["seed"]}
            stats["nodes_reached"] += 1
            stats["max_hop_reached"] = max(stats["max_hop_reached"], hop + 1)
            queue.append((nb, hop + 1))
            ntype = nbd.get("type")
            if ntype == "Evidence" and not _is_policy_source_evidence(nbd):
                if len(ev_items) < evidence_limit: ev_items.append(nb)
                else: stats["evidence_limit_reached"] = True
            elif ntype in _TRAVERSAL_NODE_TYPES or (ntype == "Evidence" and _is_policy_source_evidence(nbd)):
                if len(node_items) < node_limit: node_items.append(nb)
                else: stats["node_limit_reached"] = True
        if len(ev_items) >= evidence_limit and len(node_items) >= node_limit and not done:
            stats["evidence_limit_reached"] = stats["evidence_limit_reached"] or bool(queue)
            stats["stopped_because"] = "result_limits_reached"
            done = True
    for oid in ev_items:
        info = visited[oid]
        path = info["path"]
        via = path[-1]["via"]
        vt = G.nodes[via].get("type")
        via_type = f"Entity/{G.nodes[via].get('entity_type')}" if vt == "Entity" else vt
        on_path = {x for e_ in path for x in (e_["via"], e_["reached"])} - {oid}
        heuristic = any(e_["heuristic"] for e_ in path) or any(G.nodes[n_].get("heuristic") for n_ in on_path)
        relset = {e_["relation"] for e_ in path}
        if via_type == "Entity/Term": lim = "shared term only (co-mention): does NOT establish identity, transaction ownership, policy compliance or a violation"
        elif "CONTRADICTS" in relset: lim = "reached through a discrepancy edge: context for reconciliation, not proof of a contradiction or violation"
        elif relset & {"VIOLATES", "SATISFIES", "EVALUATES", "HAS_RISK", "BELONGS_TO"}: lim = "reached through a policy-reasoning path (Decision / Rule / Risk): shows association with that decision's basis; not independent corroboration and not by itself proof of a violation"
        elif "LINKED_FROM" in relset: lim = "reached through a LINKED_FROM document relation: lineage context from a linked document, not corroboration"
        else: lim = f"shared {via_type} value: context link; does not by itself establish identity, ownership, compliance or a violation"
        if len(path) > 1: lim = f"{len(path)}-hop path (each extra hop weakens the association): " + lim
        pol_ctx = bool(G.nodes[oid].get("context_only"))
        if pol_ctx:
            lim = ("POLICY CONTEXT ONLY: this evidence is policy-listed information; its amount is not a transaction record, it neither corroborates nor contradicts any employee transaction, "
                   "and no employee-to-policy-amount association is implied. Underlying link: " + lim)
        res["evidence"].append({"evidence_id": oid, "seed_id": info["seed"], "via_node_id": via, "via_type": via_type, "limitation": lim, "policy_context": pol_ctx,
                                "via_label": _via_label(G, via), "relations": [f"{e_['from']} --{e_['relation']}--> {e_['to']}" for e_ in path], "heuristic": heuristic,
                                "match_strength": "heuristic" if heuristic else "strong", "link_method": next((e_["method"] for e_ in path if e_.get("method")), None) or f"graph_path_{len(path)}_hop",
                                "hops": len(path), "path": path})
        for e_ in path: res["path_edge_keys"].add((e_["from"], e_["relation"], e_["to"]))
    for nid in node_items:
        info = visited[nid]
        path = info["path"]
        on_path = {x for e_ in path for x in (e_["via"], e_["reached"])}
        heuristic = any(e_["heuristic"] for e_ in path) or any(G.nodes[n_].get("heuristic") for n_ in on_path)
        res["nodes"].append({"node_id": nid, "type": G.nodes[nid].get("type"), "detail": _traversal_node_detail(G, nid), "seed_id": info["seed"], "hops": len(path),
                             "relations": [f"{e_['from']} --{e_['relation']}--> {e_['to']}" for e_ in path], "heuristic": heuristic, "path": path})
        for e_ in path: res["path_edge_keys"].add((e_["from"], e_["relation"], e_["to"]))
    return res

# --- POLICY ENGINE (deterministic rule DSL -> Decision / Risk / VIOLATES / SATISFIES) ---
# This is NOT NLP. Only rulebook lines written in the DSL below are evaluated. Every other line stays
# context-only and gets a Decision with verdict UNEVALUATED (so the graph states honestly what was not checked).
#
#   REQUIRE|FORBID GOVID VALID                     (Verhoeff checksum of detected GovIDs)
#   REQUIRE|FORBID PHONE VALID                     (phonenumbers validity of detected phones)
#   REQUIRE|FORBID TRANSACTION <op> <n> [CUR]      (op: > >= < <= ; CUR e.g. INR, USD, Rs)
#   REQUIRE|FORBID KEYWORD "phrase"                (case-insensitive phrase in Evidence text)
#   optional suffix on any rule: [severity=low|medium|high|critical]   (default MEDIUM)
# Examples:  FORBID TRANSACTION > 50000 INR [severity=high]   |   REQUIRE GOVID VALID
# Verdicts: VIOLATION | SATISFIED | NOT_APPLICABLE (no relevant data found) | INCONCLUSIVE (rule parsed but extraction
#           incomplete, absence of text is NOT proof) | UNEVALUATED (rule not in DSL / ambiguous: never checked)
_RULE_SEVERITY = re.compile(r'[\[(]\s*severity\s*[=:]\s*(low|medium|high|critical)\s*[\])]', re.IGNORECASE)
_RULE_ENTITY = re.compile(r'^\s*(REQUIRE|FORBID)\s+(GOVID|PHONE)S?\s+(?:IS\s+|ARE\s+)?VALID\s*$', re.IGNORECASE)
_RULE_TXN = re.compile(r'^\s*(REQUIRE|FORBID)\s+TRANSACTIONS?(?:\s+AMOUNTS?)?\s*(>=|<=|>|<)\s*(?:([A-Za-z]{2,3}\.?|₹|\$)\s*)?(\d[\d,]*(?:\.\d+)?)\s*([A-Za-z]{2,3}\.?|₹|\$)?\s*$', re.IGNORECASE)
_KNOWN_CURRENCIES = {"INR", "USD", "EUR", "GBP"}  # anything else in a TRANSACTION rule is ambiguous -> UNEVALUATED
_LIST_MARKER = re.compile(r'^\s*(?:[-*•–—▪●◦○■□☐✓✔→➢➤>]+|\(?\d{1,3}(?:\.\d{1,3})*[.)]|\(?[A-Za-z][.)])\s+')
_RULE_LABEL = re.compile(r'(?i)^\s*(?:rule|policy|clause|section)\s*#?\s*\d+(?:\.\d+)*[A-Za-z]?\s*[:.)\-–—]\s*')
# Operator WORDS are rewritten to symbols only inside TRANSACTION rules (never inside quoted KEYWORD phrases).
_OP_WORDS = [(re.compile(r'(?i)\b(?:greater\s+than\s+or\s+equal\s+to|at\s+least|not\s+less\s+than|no\s+less\s+than)\b'), '>='),
             (re.compile(r'(?i)\b(?:less\s+than\s+or\s+equal\s+to|at\s+most|not\s+more\s+than|no\s+more\s+than|not\s+exceeding)\b'), '<='),
             (re.compile(r'(?i)\b(?:greater\s+than|more\s+than|exceeding|exceeds|over|above)\b'), '>'),
             (re.compile(r'(?i)\b(?:less\s+than|fewer\s+than|under|below)\b'), '<')]
_RULE_KW = re.compile(r'^\s*(REQUIRE|FORBID)\s+KEYWORD\s+["“\'](.+?)["”\']\s*$', re.IGNORECASE)
_CMP_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}

def _normalize_rule_line(line: str) -> str:
    """Formatting-only cleanup so a valid DSL rule is not rejected for decoration: list markers ("1.", "-", "a)"),
    markdown emphasis, trailing punctuation, unicode >= / <=, 'Gov ID' / 'Phone number' spellings. Meaning never changes."""
    s_ = re.sub(r'[\u00a0\u2000-\u200b\u202f\u3000\ufeff]', ' ', line or "")  # NBSP / unicode / zero-width whitespace
    s_ = s_.strip().strip("|").strip().strip("*_`").strip()  # markdown table pipes + emphasis
    for _ in range(2):  # nested markers such as "- 1."
        s_ = _LIST_MARKER.sub("", s_, count=1).strip().strip("*_`").strip()
    s_ = _RULE_LABEL.sub("", s_, count=1).strip()
    s_ = s_.replace("≥", ">=").replace("≤", "<=").replace("＞", ">").replace("＜", "<")
    s_ = re.sub(r'(?i)^(?:FORBIDS?|FORBIDDEN|PROHIBITS?|PROHIBITED)\b', 'FORBID', s_)
    s_ = re.sub(r'(?i)^(?:REQUIRES?|REQUIRED)\b', 'REQUIRE', s_)
    s_ = re.sub(r'(?i)^(REQUIRE|FORBID)\s*[:\-–—]\s*', r'\1 ', s_)
    s_ = re.sub(r'(?i)\bGOV(?:ERNMENT)?[\s_-]?IDS?\b', 'GOVID', s_)
    s_ = re.sub(r'(?i)\bPHONE[\s_-]?(?:NUMBERS?|NOS?)\b', 'PHONE', s_)
    if re.match(r'(?i)^(?:REQUIRE|FORBID)\s+TRANSACTIONS?\b', s_):
        for pat, sym in _OP_WORDS: s_ = pat.sub(sym, s_)
        s_ = re.sub(r'(?i)\bUS\$', '$', s_)
        s_ = re.sub(r'([<>])\s+=', r'\1=', s_)
        s_ = re.sub(r'(?<=\d)\s*/-', '', s_)  # Indian "5,000/-"
    s_ = re.sub(r'[.;,]+\s*$', '', s_)  # "Rs." survives: the regex accepts "Rs" without the dot
    return re.sub(r'\s+', ' ', s_).strip()

# Narrow plain-English support. A line is mapped ONLY when it is an unconditional, un-negated requirement that clearly names Government-ID
# checksum/validation, or clearly requires transaction AMOUNTS to match (no money value, no limit/threshold/policy wording). Anything else
# stays UNEVALUATED. Mapping a line never implies it can be evaluated: evaluation has its own configuration/linkage gates.
_PLAIN_MODAL = re.compile(r'(?i)\b(must|shall|should|required?|needs?\s+to|has\s+to|have\s+to|is\s+to|are\s+to)\b')
_PLAIN_NEG_COND = re.compile(r"(?i)\b(not|never|no|none|unless|except|excluding|only\s+if|if|when|whenever|where|provided|exempt\w*|waive\w*)\b|n't")
_PLAIN_GOVID_CHECK = re.compile(r'(?i)\b(checksum|check[\s-]?digit|validat\w*|valid)\b')
_PLAIN_AMT_SUBJECT = re.compile(r'(?i)\b(transactions?|expenses?|invoices?|payments?|records?)\b')
_PLAIN_AMT_WORD = re.compile(r'(?i)\bamounts?\b')
_PLAIN_MATCH_VERB = re.compile(r'(?i)\b(match(?:es|ed|ing)?|consistent|agree\w*|reconcil\w*|identical|equal)\b')
_PLAIN_AMT_BLOCK = re.compile(r'(?i)\b(limits?|thresholds?|maximum|max|exceed\w*|up\s+to|cap|ceiling|more\s+than|less\s+than|policy|policies|allowed|allowance)\b|[<>]=?')

def _parse_plain_english_rule(line: str, severity: str) -> Optional[Dict[str, Any]]:
    if not line or MONEY_PATTERN.search(line) or _PLAIN_NEG_COND.search(line) or not _PLAIN_MODAL.search(line): return None
    has_govid = bool(re.search(r'(?i)\bGOVID\b', line))
    if has_govid and _PLAIN_GOVID_CHECK.search(line) and not _PLAIN_AMT_WORD.search(line):
        return {"mode": "REQUIRE", "subject": "GOVID", "severity": severity, "form": "plain_english_govid_validation"}
    if (not has_govid and _PLAIN_AMT_SUBJECT.search(line) and _PLAIN_AMT_WORD.search(line) and _PLAIN_MATCH_VERB.search(line)
            and not _PLAIN_AMT_BLOCK.search(line)):
        return {"mode": "REQUIRE", "subject": "AMOUNT_MATCH", "severity": severity, "form": "plain_english_amount_match"}
    return None

def parse_policy_rule(condition: str) -> Optional[Dict[str, Any]]:
    """DSL line -> spec dict, or None if the line is free text / ambiguous (never guessed, stays UNEVALUATED)."""
    line = _normalize_rule_line(condition)
    severity = "MEDIUM"
    m = _RULE_SEVERITY.search(line)
    if m:
        severity = m.group(1).upper()
        line = _normalize_rule_line(_RULE_SEVERITY.sub("", line))
    m = _RULE_ENTITY.match(line)
    if m: return {"mode": m.group(1).upper(), "subject": m.group(2).upper(), "severity": severity}
    m = _RULE_TXN.match(line)
    if m:
        try: value = float(m.group(4).replace(",", ""))
        except ValueError: return None
        cur_pre = _normalize_currency(m.group(3)) if m.group(3) else None
        cur_suf = _normalize_currency(m.group(5)) if m.group(5) else None
        if cur_pre and cur_suf and cur_pre != cur_suf: return None  # contradictory currencies
        currency = cur_pre or cur_suf
        if currency is not None and currency not in _KNOWN_CURRENCIES: return None  # unknown word, not a currency
        return {"mode": m.group(1).upper(), "subject": "TRANSACTION", "op": m.group(2), "value": value,
                "currency": currency, "severity": severity}
    m = _RULE_KW.match(line)
    if m: return {"mode": m.group(1).upper(), "subject": "KEYWORD", "value": m.group(2).strip().lower(), "severity": severity}
    return _parse_plain_english_rule(line, severity)

def _unevaluated_reason(condition: str) -> str:
    """WHY a rule line is not evaluated. Description only: the line is never reinterpreted or guessed."""
    line = _normalize_rule_line(condition)
    if re.match(r'(?i)^(REQUIRE|FORBID)\s+TRANSACTIONS?\b', line):
        return "Written as a TRANSACTION rule, but the operator, amount or currency could not be read unambiguously (supported: > >= < <= with INR/USD/EUR/GBP); not guessed."
    if re.match(r'(?i)^(REQUIRE|FORBID)\b', line):
        return 'Starts with REQUIRE/FORBID but the subject/form is unsupported (supported: GOVID VALID, PHONE VALID, TRANSACTION <op> <amount> [CUR], KEYWORD "phrase").'
    if re.search(r'(?i)\bGOVID\b', line) and _PLAIN_GOVID_CHECK.search(line):
        return "Government ID validation statement that is conditional, negated or not clearly an unconditional requirement; its meaning is not guessed."
    if _PLAIN_AMT_SUBJECT.search(line) and _PLAIN_AMT_WORD.search(line) and _PLAIN_MATCH_VERB.search(line):
        return "Amount-matching statement that is conditional, negated, contains a value, or refers to limits/policy; it is not clearly an unconditional 'amounts must match' requirement and is not guessed."
    if MONEY_PATTERN.search(line) or re.search(r'(?i)[<>]=?|\b(limit|threshold|maximum|exceed\w*|up\s+to|cap)\b', line):
        return ("Plain-English statement with an amount/limit/threshold. It is not in a supported REQUIRE/FORBID form, and its meaning "
                "(limit vs. actual transaction, scope, exceptions) is not guessed.")
    return "Plain-English statement with no supported REQUIRE/FORBID form."

def _nodes_of_type(G: nx.MultiDiGraph, node_type: str, entity_type: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
    return [(n, d) for n, d in G.nodes(data=True)
            if d.get("type") == node_type and (entity_type is None or d.get("entity_type") == entity_type)]

def _supporting_evidence_ids(G: nx.MultiDiGraph, node_id: str) -> List[str]:
    if G.nodes[node_id].get("type") == "Evidence": return [node_id]
    return [s["node_id"] for s in _in_edge_sources(G, node_id, "SUPPORTS", "Evidence")]

def extraction_gaps(G: nx.MultiDiGraph) -> List[str]:
    """Document nodes whose extraction failed or produced no Evidence. Absence of text in these is NOT proof of anything."""
    gaps = []
    for n, d in _nodes_of_type(G, "Document"):
        if d.get("extraction_status") == "error":
            gaps.append(n)
            continue
        if not any(e.get("relation") == "DERIVED_FROM" for _, _, e in G.in_edges(n, data=True)):
            gaps.append(n)
    return gaps

def _evaluate_amount_match(G: nx.MultiDiGraph, spec: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    """REQUIRE amounts to match. Compares ONLY records reliably linked to the same transaction (same reference/invoice ID, or same date +
    vendor/person + expense type, same currency, different documents). A shared name/label and currency alone never qualify.
    SATISFIED/VIOLATION only from such pairs; INCONCLUSIVE when records from several documents exist but none are reliably linked;
    NOT_APPLICABLE when there are not records from at least two documents. Policy-listed amounts are never records."""
    recs = [(n, d) for n, d in _nodes_of_type(G, "Transaction") if not _is_policy_role(d.get("amount_role"))]
    def _docs(n: str) -> set:
        s: set = set()
        for _, dd in _txn_evidence_docs(G, n): s |= dd
        return s
    viol, sat, seen_pairs = [], [], set()
    n_viol_pairs = n_sat_pairs = 0
    for key, txns in _strong_txn_groups(G).items():
        if len(txns) < 2 or len(txns) > 50: continue
        for i in range(len(txns)):
            for j in range(i + 1, len(txns)):
                t1, t2 = txns[i], txns[j]
                pair = frozenset((t1, t2))
                n1, n2 = G.nodes[t1], G.nodes[t2]
                if pair in seen_pairs or n1.get("currency") != n2.get("currency"): continue
                if _match_transactions(n1, n2)[0] != "strong": continue
                d1, d2 = _docs(t1), _docs(t2)
                if not d1 or not d2 or (d1 & d2): continue
                seen_pairs.add(pair)
                if n1.get("amount") == n2.get("amount"): sat.extend([t1, t2]); n_sat_pairs += 1
                else: viol.extend([t1, t2]); n_viol_pairs += 1
    viol, sat = list(dict.fromkeys(viol)), list(dict.fromkeys(sat))
    if viol: return "VIOLATION", f"REQUIRE amounts to match: {n_viol_pairs} reliably linked record pair(s) (shared reference ID, or date + party + expense type) have DIFFERING amounts; {n_sat_pairs} linked pair(s) match.", viol, sat
    if sat: return "SATISFIED", f"REQUIRE amounts to match: {n_sat_pairs} reliably linked record pair(s) have matching amounts.", [], sat
    doc_ids = set()
    for n, _ in recs: doc_ids |= _docs(n)
    if len(recs) >= 2 and len(doc_ids) >= 2:
        return "INCONCLUSIVE", ("REQUIRE amounts to match: evaluation attempted, but no records are reliably linked to the same transaction (no shared reference/transaction ID or date + party + expense type). "
                                "A shared employee name and currency alone are not sufficient, so nothing was compared and no pass/violation was assigned."), [], []
    return "NOT_APPLICABLE", "REQUIRE amounts to match: fewer than two transaction records from different documents were found.", [], []

def _evaluate_spec(G: nx.MultiDiGraph, spec: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    """-> (verdict, rationale, violating_node_ids, satisfying_node_ids). UNEVALUATED = no evaluation performed (required configuration missing);
    INCONCLUSIVE = attempted but evidence insufficient; VIOLATION/SATISFIED only with sufficient evidence."""
    subject, mode = spec["subject"], spec["mode"]
    forced_violation = False
    if subject == "AMOUNT_MATCH": return _evaluate_amount_match(G, spec)
    cfg: Dict[str, Any] = {}
    if subject == "GOVID":
        cfg = govid_validation_config()
        if not cfg["complete"]:
            return "UNEVALUATED", f"Government ID rule not evaluated: {govid_unvalidated_reason(cfg)}.", [], []
    if subject in ("GOVID", "PHONE"):
        pop = _nodes_of_type(G, "Entity", "GovID" if subject == "GOVID" else "Phone")
        if not pop: return "NOT_APPLICABLE", f"No {subject} entities found.", [], []
        true_ids = [n for n, d in pop if d.get("is_valid") is True]
        false_ids = [n for n, d in pop if d.get("is_valid") is False]
        n_unverified = len(pop) - len(true_ids) - len(false_ids)
        if not true_ids and not false_ids:
            return "INCONCLUSIVE", (f"{len(pop)} {subject} value(s) detected but UNVERIFIED (no jurisdiction-specific validation available); "
                                    f"validity was not determined, so the rule could not be checked."), [], []
        viol, sat = (false_ids, true_ids) if mode == "REQUIRE" else (true_ids, false_ids)
        rationale = f"{mode} {subject} VALID: {len(true_ids)} valid, {len(false_ids)} invalid."
        if n_unverified: rationale += f" {n_unverified} UNVERIFIED value(s) not counted."
        if subject == "GOVID": rationale += f" (checksum arithmetic under configured method {cfg.get('method')} only; authenticity not verified)"
    elif subject == "TRANSACTION":
        all_pop = [(n, d) for n, d in _nodes_of_type(G, "Transaction") if not spec.get("currency") or d.get("currency") == spec["currency"]]
        excluded_policy = [n for n, d in all_pop if _is_policy_role(d.get("amount_role"))]
        excluded_policy += [n for n, d in _nodes_of_type(G, "Claim") if d.get("claim_type") == "PolicyListedAmount" and (not spec.get("currency") or d.get("currency") == spec["currency"])]
        pop = [(n, d) for n, d in all_pop if not _is_policy_role(d.get("amount_role"))]
        dropped = 0
        if not pop: return "NOT_APPLICABLE", "No matching Transaction nodes found." + (f" {len(excluded_policy)} amount(s) identified as policy limits/thresholds were not treated as transactions." if excluded_policy else ""), [], []
        cmp_fn = _CMP_OPS[spec["op"]]
        true_ids = [n for n, d in pop if cmp_fn(d.get("amount", 0.0), spec["value"])]
        _true_set = set(true_ids)
        false_ids = [n for n, d in pop if n not in _true_set]
        viol, sat = (false_ids, true_ids) if mode == "REQUIRE" else (true_ids, false_ids)
        _unclear = {n for n, d in pop if d.get("amount_role") == ROLE_UNCLEAR}
        if _unclear:
            _cv, _cs = [n for n in viol if n not in _unclear], [n for n in sat if n not in _unclear]
            dropped = (len(viol) - len(_cv)) + (len(sat) - len(_cs))
            viol, sat = _cv, _cs
            if dropped and not viol and not sat:
                return "INCONCLUSIVE", (f"{mode} TRANSACTION {spec['op']} {spec['value']:g} {spec.get('currency') or ''}: the only amount(s) examined have an UNCLEAR role "
                                        f"(not established as actual transactions), so no pass/fail was assigned."), [], []
        rationale = f"{mode} TRANSACTION {spec['op']} {spec['value']:g} {spec.get('currency') or ''}: {len(true_ids)} match condition of {len(pop)}."
        if excluded_policy: rationale += f" {len(excluded_policy)} policy limit/threshold amount(s) excluded (not transactions)."
        if dropped: rationale += f" {dropped} result(s) based only on amounts of UNCLEAR role were not counted."
    else:  # KEYWORD
        pop = _investigation_evidence(G)
        if not pop: return "NOT_APPLICABLE", "No Evidence nodes found.", [], []
        phrase = spec["value"]
        hit_ids = [n for n, d in pop if phrase in (d.get("text", "") or "").lower()]
        if mode == "REQUIRE":
            if not hit_ids:
                gaps = extraction_gaps(G)
                if gaps:  # phrase not found, but some documents were unreadable/empty: cannot conclude a violation
                    return "INCONCLUSIVE", (f"REQUIRE KEYWORD '{phrase}': not found, but extraction failed or was empty for "
                                            f"{len(gaps)} document(s); absence of text is not proof of a violation."), [], []
            viol, sat = [], hit_ids
            forced_violation = not hit_ids
            rationale = f"REQUIRE KEYWORD '{phrase}': found in {len(hit_ids)} evidence node(s)."
        else:
            viol, sat = hit_ids, []
            rationale = f"FORBID KEYWORD '{phrase}': found in {len(hit_ids)} evidence node(s)."
    if viol or forced_violation: verdict = "VIOLATION"
    elif sat or (subject == "KEYWORD" and mode == "FORBID"): verdict = "SATISFIED"
    else: verdict = "NOT_APPLICABLE"
    return verdict, rationale, viol, sat

def _basis_link_problem(G: nx.MultiDiGraph, node_id: str) -> Optional[str]:
    """None if `node_id` may be the basis of a VIOLATES / SATISFIES edge, else the reason it may not. A basis node needs NON-heuristic supporting Evidence
    that has a DERIVED_FROM source Document (an Evidence node qualifies by itself). Heuristic nodes/links (Term entities, CONTRADICTS-only relations) and
    policy-context evidence never qualify, so a heuristic contradiction can never independently produce a confirmed policy violation."""
    d = G.nodes[node_id]
    if d.get("heuristic"): return "node is heuristic"
    if d.get("type") == "Evidence":
        if _is_policy_source_evidence(d) or d.get("context_only"): return "policy-context evidence is not case evidence"
        return None if _evidence_source_docs(G, node_id) else "evidence has no DERIVED_FROM source document"
    firm = []
    for e in _supporting_evidence_ids(G, node_id):
        en = G.nodes[e]
        if en.get("context_only") or _is_policy_source_evidence(en) or not _evidence_source_docs(G, e): continue
        if any(not ed.get("heuristic") for ed in (G.get_edge_data(e, node_id) or {}).values() if ed.get("relation") == "SUPPORTS"): firm.append(e)
    return None if firm else "no non-heuristic supporting Evidence with a source Document"

def _gate_basis(G: nx.MultiDiGraph, nodes: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    keep, withheld = [], []
    for n in nodes:
        why = _basis_link_problem(G, n)
        if why is None: keep.append(n)
        else: withheld.append((n, why))
    return keep, withheld

# =====================================================================================================================
# COMPILED POLICY: rulebook text -> policy_compiler (LLM interprets, deterministic validation) -> graph-derived facts/evidence ->
# rule_engine (deterministic). Results live ON the existing PolicyRule / Decision / Policy nodes (no parallel policy graph).
# The legacy DSL evaluator (evaluate_policy_rules) is kept: it decides every rule the compiler path cannot determine.
# =====================================================================================================================
_COMPILED_TXN_ENTITIES = ("transaction", "transactions", "expense", "expenses", "payment", "payments")
_COMPILED_ENTITY_MAP = {"GovID": ("govid", "gov_id", "government_id"), "Phone": ("phone", "phone_number")}
_COMPILED_TO_LEGACY = {"VIOLATION": "VIOLATION", "COMPLIANT": "SATISFIED", "NOT_APPLICABLE": "NOT_APPLICABLE", "EXEMPT": "NOT_APPLICABLE", "ACTION_REQUIRED": "INCONCLUSIVE"}
_COMPILED_PRIORITY = ["VIOLATION", "ACTION_REQUIRED", "INDETERMINATE", "COMPLIANT", "EXEMPT", "NOT_APPLICABLE"]  # INDETERMINATE outranks a pass: a pass is never claimed over unchecked records
_COMPILED_SEV = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_COMPILED_REPORT_KEYS = ("evaluation_engine", "compiled_verdict", "legacy_verdict", "compiled_rules", "compiled_missing_facts", "supporting_evidence_ids", "contradicting_evidence_ids", "rulebook_provenance")

def _ws_norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def _compiled_source_span(text: str, clause: Optional[str]) -> Optional[List[int]]:
    """[start, end) offsets of the clause inside the text handed to the compiler (whitespace-tolerant), else None."""
    if not clause or not clause.strip(): return None
    m = re.search(r"\s+".join(re.escape(tok) for tok in clause.split()), text)
    return [m.start(), m.end()] if m else None

def _norm_compiled_evidence_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _explicit_evidence_timing(text: str, expected: Any) -> bool:
    """True only for a direct, affirmative statement that something was dated before payment."""
    val = _norm_compiled_evidence_text(expected).replace(" ", "_")
    if val != "before_payment":
        return False
    if re.search(r"\b(?:not|never|no|without)\b.{0,45}\bbefore\s+(?:the\s+)?payment\b", text, re.I):
        return False
    return bool(re.search(r"\b(?:dated|approved|signed|granted|issued|attached|provided)?\s*(?:on\s+)?before\s+(?:the\s+)?payment\b", text, re.I))


def _explicitly_negates_evidence(text: str, phrase: str) -> bool:
    """Conservative local negation guard for typed evidence inferred from exact source wording."""
    m = re.search(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", text, re.I)
    if not m:
        return True
    window = text[max(0, m.start() - 55):min(len(text), m.end() + 55)]
    return bool(re.search(r"\b(?:no|not|never|without|missing|lacks?|lacked)\b.{0,45}\b(?:" + re.escape(phrase.split()[0]) + r")\b|\b(?:not|never)\b.{0,35}\b(?:attached|provided|obtained|approved|present)\b", window, re.I))


def _typed_evidence_from_requirements(G: nx.MultiDiGraph, compiled_rules: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Create typed evidence inputs only when the evidence node explicitly supports the type and attributes.

    No LLM is used here. Type labels must appear verbatim as a phrase in the source evidence;
    non-timing match attributes must be explicitly labelled in the text. Policy/context evidence
    and negated evidence are excluded. Unsupported requirements remain unknown.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    requirements = []
    for rule in compiled_rules or []:
        for req in (getattr(rule, "required_evidence", None) or []):
            req_type = str(getattr(req, "type", "") or "").strip().lower()
            if req_type:
                requirements.append((req_type, getattr(req, "description", None), getattr(req, "match", None)))
    # Articles ("a/an/the") are ignored only for the contiguous type-phrase search and its negation guard,
    # so "approval from a Director" (policy) equals "approval from the Director" (evidence). Role words stay
    # bound inside the contiguous phrase; timing and labelled-attribute checks still use the original text.
    _strip_articles = lambda s: re.sub(r"\b(?:a|an|the)\b\s*", "", s).strip()
    for ev_id, data in _investigation_evidence(G):
        if data.get("context_only") or _is_policy_source_evidence(data):
            continue
        text = _norm_compiled_evidence_text(data.get("text", ""))
        if not text:
            continue
        text_phrase_view = _strip_articles(text)
        for req_type, description, match in requirements:
            phrase = _strip_articles(re.sub(r"[^a-z0-9]+", " ", req_type).strip())
            if not phrase or not re.search(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", text_phrase_view, re.I):
                continue
            if _explicitly_negates_evidence(text_phrase_view, phrase):
                continue
            attrs: Dict[str, Any] = {}
            match_map = match if isinstance(match, dict) else {}
            matches = True
            for key, expected in match_map.items():
                if str(key).lower() == "timing" and _norm_compiled_evidence_text(expected).replace(" ", "_") == "before_payment":
                    if not _explicit_evidence_timing(text, expected):
                        matches = False
                        break
                    attrs[str(key)] = "before_payment"
                else:
                    label = _norm_compiled_evidence_text(key)
                    value = _norm_compiled_evidence_text(expected)
                    if not label or not value or not re.search(r"\b" + re.escape(label).replace(r"\ ", r"\s+") + r"\s*[:=]\s*" + re.escape(value).replace(r"\ ", r"\s+") + r"\b", text, re.I):
                        matches = False
                        break
                    attrs[str(key)] = expected
            if not matches:
                continue
            item = {"type": req_type, "evidence_id": ev_id, "file": data.get("source_file"), "location": data.get("source_location"), **attrs}
            key = (ev_id, req_type, tuple(sorted((str(k), str(v)) for k, v in attrs.items())))
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def graph_to_rule_inputs(G: nx.MultiDiGraph, compiled_rules: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Existing structured graph data -> deterministic rule-engine inputs.
    Transaction facts remain restricted to unambiguous transaction-role amounts and extracted attributes.
    Evidence types / match attributes are added only when the source Evidence text explicitly supports them;
    unsupported or negated evidence is not inferred. Returns node_ids aligned with fact lists.

    When callers omit ``compiled_rules`` on an already evaluated graph, recover attached structured rules
    so diagnostic input snapshots use the same evidence typing requirements as actual execution.
    """
    if compiled_rules is None and HAS_POLICY_COMPILER:
        recovered = []
        for _, rule_data in _nodes_of_type(G, "PolicyRule"):
            for entry in (rule_data.get("compiled_rules") or []):
                raw_rule = entry.get("compiled_rule") if isinstance(entry, dict) else None
                if isinstance(raw_rule, dict):
                    try:
                        recovered.append(_CompiledRule.model_validate(raw_rule))
                    except Exception:
                        # Diagnostics remain read-only and tolerant of malformed historical entries.
                        continue
        compiled_rules = recovered or None

    facts: Dict[str, Any] = {}
    node_ids: Dict[str, List[str]] = {}
    txns = [(n, d) for n, d in _nodes_of_type(G, "Transaction") if d.get("amount_role") == ROLE_TXN and isinstance(d.get("amount"), (int, float))]
    if txns:
        recs: List[Dict[str, Any]] = []
        for _, d in txns:
            cur = d.get("currency") if d.get("currency") in _KNOWN_CURRENCIES else None
            rec: Dict[str, Any] = {"amount": ({"value": d["amount"], "unit": cur} if cur else d["amount"])}
            if cur: rec["currency"] = cur
            for k in _ATTR_KEYS:
                v = (d.get("attributes") or {}).get(k)
                if v: rec[k] = v
            recs.append(rec)
        for alias in _COMPILED_TXN_ENTITIES:
            facts[alias], node_ids[alias] = recs, [n for n, _ in txns]
    for etype, aliases in _COMPILED_ENTITY_MAP.items():
        if etype == "GovID" and not govid_validation_config()["complete"]: continue
        ents = [(n, d) for n, d in _nodes_of_type(G, "Entity", etype) if d.get("is_valid") is not None]
        if not ents: continue
        recs = [{"is_valid": bool(d["is_valid"])} for _, d in ents]
        for alias in aliases:
            facts[alias], node_ids[alias] = recs, [n for n, _ in ents]
    typed = [{"type": d["evidence_type"], "evidence_id": n, "file": d.get("source_file"), "location": d.get("source_location")}
             for n, d in _investigation_evidence(G) if d.get("evidence_type") and not d.get("context_only")]
    typed.extend(_typed_evidence_from_requirements(G, compiled_rules))
    # Deduplicate only exact evidence/type/attribute triples; different structured match attributes
    # must remain separate so one cannot accidentally satisfy another requirement.
    deduped, seen_evidence = [], set()
    for item in typed:
        key = (item.get("evidence_id"), item.get("type"), tuple(sorted((str(k), str(v)) for k, v in item.items() if k not in ("evidence_id", "type", "file", "location"))))
        if key not in seen_evidence:
            seen_evidence.add(key)
            deduped.append(item)
    return {"facts": facts, "evidence": (deduped or None), "node_ids": node_ids, "strict_units": COMPILED_POLICY_STRICT_UNITS, "evaluation_date": None, "fx_rates": None}

def _compiled_top(entries: List[Dict[str, Any]]) -> Optional[str]:
    vs = {(e.get("result") or {}).get("verdict") for e in entries if e.get("result")}
    if not vs: return None
    return next((v for v in _COMPILED_PRIORITY if v in vs), "INDETERMINATE")

def _compiled_entry_result(entry: Dict[str, Any], rrs: List[Dict[str, Any]], node_ids: Dict[str, List[str]], eval_error: Optional[str]) -> Dict[str, Any]:
    if entry["status"] != "VALID":  # NEEDS_REVIEW is never executed
        return {"executed": False, "verdict": "INDETERMINATE", "counts": {}, "missing_facts": [], "violating_node_ids": [], "satisfying_node_ids": [],
                "reasons": ["NEEDS_REVIEW compiled rule: NOT executed (ambiguous or low confidence): " + "; ".join(entry.get("ambiguities") or entry.get("issues") or ["see issues"])]}
    if eval_error or not rrs:
        return {"executed": False, "verdict": "INDETERMINATE", "counts": {}, "missing_facts": [], "violating_node_ids": [], "satisfying_node_ids": [],
                "reasons": [f"compiled rule engine did not produce a result: {eval_error or 'no result returned'}"]}
    counts = Counter(r["verdict"] for r in rrs)
    top = next((v for v in _COMPILED_PRIORITY if counts.get(v)), "INDETERMINATE")
    ids = node_ids.get(((entry["compiled_rule"].get("entity")) or "").lower()) or []
    def _nodes(v: str) -> List[str]:
        return list(dict.fromkeys(ids[r["record_index"]] for r in rrs if r["verdict"] == v and isinstance(r.get("record_index"), int) and r["record_index"] < len(ids)))
    sev = max((r["severity"] for r in rrs if r["verdict"] == "VIOLATION" and r.get("severity")), key=lambda s_: _COMPILED_SEV.get(s_, 0), default=None)
    return {"executed": True, "verdict": top, "counts": dict(counts), "missing_facts": sorted({m for r in rrs for m in r.get("missing_facts", [])}),
            "reasons": list(dict.fromkeys(x for r in rrs for x in r.get("reasons", [])))[:10],
            "exception_applied": next((r["exception_applied"] for r in rrs if r.get("exception_applied")), None),
            "action_required": next((r["action_required"] for r in rrs if r.get("action_required")), None), "severity": sev,
            "violating_node_ids": _nodes("VIOLATION"), "satisfying_node_ids": _nodes("COMPLIANT"),
            "record_results": [{"record_index": r.get("record_index"), "verdict": r["verdict"], "missing_facts": r.get("missing_facts", [])} for r in rrs[:50]]}

def _compiled_rule_leaves(condition: Any) -> List[Any]:
    if condition is None:
        return []
    children = getattr(condition, "children", None) or []
    if children:
        return [leaf for child in children for leaf in _compiled_rule_leaves(child)]
    return [condition]


def _is_confirmed_violation_trigger(rule: Any) -> bool:
    if getattr(getattr(rule, "rule_type", None), "value", None) != "TRIGGER":
        return False
    for leaf in _compiled_rule_leaves(getattr(rule, "condition", None)):
        entity = str(getattr(leaf, "entity", "") or "").lower()
        field = str(getattr(leaf, "field", "") or "").lower()
        op = getattr(getattr(leaf, "operator", None), "value", getattr(leaf, "operator", None))
        if entity.endswith("_violation") and field == "confirmed" and op == "==" and getattr(leaf, "value", None) is True:
            return True
    return False


def _is_confirmed_violation_candidate(rule: Any) -> bool:
    """TRIGGER whose source says "confirmed violation(s)" and whose only predicate is a confirmation flag/status
    (any alias form, before or after alias normalization). Used to tell escalation triggers from base obligations."""
    if getattr(getattr(rule, "rule_type", None), "value", None) != "TRIGGER":
        return False
    if not re.search(r"\bconfirmed\s+violations?\b", str(getattr(rule, "source_text", "") or ""), re.I):
        return False
    cond = getattr(rule, "condition", None)
    if cond is None or getattr(cond, "children", None):
        return False
    entity = str(getattr(cond, "entity", "") or "").lower()
    field = str(getattr(cond, "field", "") or "").lower()
    op = getattr(getattr(cond, "operator", None), "value", getattr(cond, "operator", None))
    value = getattr(cond, "value", None)
    return op == "==" and (
        (entity == "violation" and field in ("status", "state") and isinstance(value, str) and value.strip().lower() == "confirmed")
        or (field in ("confirmed", "violation_confirmed") and value is True))


def _is_base_obligation_rule(rule: Any) -> bool:
    """Rules whose verdict is a compliance outcome that a confirmed violation can be derived from.

    REQUIRE/PROHIBIT always; a TRIGGER only when it carries required evidence (it then yields COMPLIANT /
    VIOLATION / NOT_APPLICABLE, e.g. "expenses above N need approval"). Escalation triggers and action-only
    triggers are never base obligations."""
    if getattr(getattr(rule, "rule_type", None), "value", None) != "TRIGGER":
        return True
    if _is_confirmed_violation_trigger(rule) or _is_confirmed_violation_candidate(rule):
        return False
    return bool(getattr(rule, "required_evidence", None))


def _normalize_confirmed_violation_trigger_aliases(rules: List[Any]) -> List[str]:
    """Normalize a narrow compiler alias for an explicit confirmed-violation trigger.

    Compiler responses may represent the explicit phrase "confirmed violations" either as
    ``violation.status == "confirmed"`` OR as ``<base_entity>.violation_confirmed == True``.
    Derivation uses ``<base_entity>_violation.confirmed == True``. Normalize only these exact
    patterns, only with one base-rule entity, and only when the trigger source explicitly says
    "confirmed violation(s)". Ambiguous/multi-domain policies remain untouched.
    """
    base_entities = {
        str(getattr(rule, "entity", "") or "").strip().lower()
        for rule in rules
        if _is_base_obligation_rule(rule)
        and str(getattr(rule, "entity", "") or "").strip()
    }
    if len(base_entities) != 1:
        return []
    base_entity = next(iter(base_entities))
    target_entity = base_entity if base_entity.endswith("_violation") else f"{base_entity}_violation"
    changed: List[str] = []
    for rule in rules:
        if getattr(getattr(rule, "rule_type", None), "value", None) != "TRIGGER":
            continue
        source = str(getattr(rule, "source_text", "") or "")
        if not re.search(r"\bconfirmed\s+violations?\b", source, re.I):
            continue
        condition = getattr(rule, "condition", None)
        if condition is None or getattr(condition, "children", None):
            continue
        entity = str(getattr(condition, "entity", "") or "").lower()
        field = str(getattr(condition, "field", "") or "").lower()
        op = getattr(getattr(condition, "operator", None), "value", getattr(condition, "operator", None))
        value = getattr(condition, "value", None)
        alias_status_form = (
            entity == "violation" and field in ("status", "state") and op == "=="
            and isinstance(value, str) and value.strip().lower() == "confirmed"
        )
        canonical_flag_form = (
            entity == base_entity and field == "violation_confirmed" and op == "==" and value is True
        )
        # Some compiler responses already use the canonical field name but leave it
        # on the generic `violation` entity (violation.confirmed == True). Because
        # the source explicitly says "confirmed violations" and this policy has
        # exactly one base entity, normalize that alias to the derived entity too.
        generic_confirmed_form = (
            entity == "violation" and field == "confirmed" and op == "==" and value is True
        )
        if not (alias_status_form or canonical_flag_form or generic_confirmed_form):
            continue
        normalized = _CompiledCondition.model_validate({
            "entity": target_entity, "field": "confirmed", "operator": _CompiledOperator.EQ,
            "value": True, "unit": None,
        })
        rule.condition = normalized
        # The executor groups records using CompiledRule.entity as well as Condition.entity.
        # Keep both schema levels aligned; changing only the condition still leaves the trigger
        # looking for facts under the stale ``violation`` alias.
        rule.entity = target_entity
        rule.operator = _CompiledOperator.EQ
        rule.value = True
        rule.unit = None
        rule.expression = _render_compiled_condition(normalized)
        changed.append(str(getattr(rule, "rule_id", "")))
    return changed


_TXN_AMOUNT_FIELD_SYNONYMS = frozenset({"total_amount", "amount_total", "total", "total_value", "grand_total"})


def _normalize_transaction_amount_field_aliases(rules: List[Any], facts: Dict[str, Any]) -> List[Dict[str, str]]:
    """Re-point a compiler-chosen monetary-total field name to the graph's canonical transaction fact ``amount``.

    The compiler names fields freely (``total_amount``, ``expense_amount``, ...) but the graph exposes exactly one
    transaction-role monetary fact per record: ``<transaction-alias>.amount``. Without this, the engine reports the
    uncovered field as missing and every dependent rule stays INDETERMINATE although the amount is known.

    Fail-closed grounding: a leaf is rewritten only when (a) its entity is a transaction alias, (b) its field is a known
    total-amount synonym or ``<entity>_amount``, (c) EVERY fact record for that entity has an ``amount`` and none has the
    original field, and (d) the same rule does not also read ``amount`` (which could mean a different quantity).
    No fact is created or changed; unit/currency checks stay in the engine. Anything else is left untouched.
    """
    changed: List[Dict[str, str]] = []
    for rule in rules:
        leaves = list(_compiled_rule_leaves(getattr(rule, "condition", None)))
        for ex in (getattr(rule, "exception", None) or []):
            leaves.extend(_compiled_rule_leaves(getattr(ex, "condition", None)))
        txn_leaves = [lf for lf in leaves
                      if str(getattr(lf, "entity", "") or "").lower() in _COMPILED_TXN_ENTITIES and getattr(lf, "field", None)]
        rewritten: List[Dict[str, str]] = []
        reads_canonical = {str(o.entity).lower() for o in txn_leaves if str(o.field).lower() == "amount"}  # taken BEFORE any rewrite
        for leaf in txn_leaves:
            entity, field = str(leaf.entity).lower(), str(leaf.field).lower()
            if field != f"{entity}_amount" and field not in _TXN_AMOUNT_FIELD_SYNONYMS:
                continue
            if entity in reads_canonical:
                continue
            records = facts.get(entity)
            if not isinstance(records, list) or not records:
                continue
            if not all(isinstance(r, dict) and "amount" in r and field not in {str(k).lower() for k in r} for r in records):
                continue
            leaf.field = "amount"
            rewritten.append({"rule_id": str(getattr(rule, "rule_id", "")), "entity": entity, "from_field": field, "to_field": "amount"})
        if rewritten:
            try:  # rebuild the rendered expression exactly as the schema does
                rule.expression = _CompiledRule.model_validate(rule.model_dump(mode="json")).expression
            except Exception:
                for r in rewritten:
                    rule.expression = (rule.expression or "").replace(f"{r['entity']}.{r['from_field']}", f"{r['entity']}.amount")
            changed.extend(rewritten)
    return changed


def _derive_confirmed_violation_facts(valid_rules: List[Any], compiled_results: List[Dict[str, Any]],
                                      compile_result: Any, facts: Dict[str, Any]) -> bool:
    """Derive *_violation.confirmed only from complete, determinate non-trigger rule results.

    If any non-trigger obligation is unresolved/rejected/unparsed, the fact stays absent so the
    engine returns INDETERMINATE. This is a derived fact, not an LLM assertion.
    """
    rules = list(getattr(compile_result, "rules", []) or [])
    base_rules = [r for r in rules if _is_base_obligation_rule(r)]
    trigger_entities = sorted({str(getattr(leaf, "entity", "") or "").lower()
                               for r in valid_rules if _is_confirmed_violation_trigger(r)
                               for leaf in _compiled_rule_leaves(getattr(r, "condition", None))
                               if str(getattr(leaf, "entity", "") or "").lower().endswith("_violation")})
    if not base_rules or not trigger_entities:
        return False
    if getattr(compile_result, "rejected", None) or getattr(compile_result, "unparsed_statements", None):
        return False
    if any(getattr(r, "status", None) != "VALID" for r in base_rules):
        return False

    by_rule: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in compiled_results:
        if row.get("rule_id"):
            by_rule[row["rule_id"]].append(row)
    base_ids = [getattr(r, "rule_id", None) for r in base_rules]
    if any(not rid or not by_rule.get(rid) for rid in base_ids):
        return False
    # All base rules must address the same record domain (transaction aliases are equivalent).
    base_entities = {str(getattr(r, "entity", "") or "").lower() for r in base_rules}
    txn_aliases = set(_COMPILED_TXN_ENTITIES)
    for target in trigger_entities:
        base_entity = target[:-len("_violation")]
        compatible = (base_entity in txn_aliases and base_entities.issubset(txn_aliases)) or base_entities == {base_entity}
        if not compatible:
            continue
        row_maps = [{r.get("record_index"): r for r in by_rule[rid]} for rid in base_ids]
        indices = set(row_maps[0])
        if None in indices or any(set(m) != indices for m in row_maps):
            continue
        derived = []
        safe = True
        for idx in sorted(indices):
            rows = [m[idx] for m in row_maps]
            verdicts = [str(r.get("verdict")) for r in rows]
            if any(v not in ("COMPLIANT", "VIOLATION", "NOT_APPLICABLE", "EXEMPT") for v in verdicts):
                safe = False
                break
            derived.append({"confirmed": any(v == "VIOLATION" for v in verdicts)})
        if safe and derived:
            facts[target] = derived
    return any(target in facts for target in trigger_entities)


def apply_compiled_policy(G: nx.MultiDiGraph, rulebook_text: str) -> Optional[Dict[str, Any]]:
    """Compile the (already extracted) rulebook text, attach each compiled rule to its existing PolicyRule node (with rulebook provenance), and
    execute VALID rules deterministically against facts/evidence derived from the graph. NEEDS_REVIEW rules are never executed. Call BEFORE
    evaluate_policy_rules, which then uses the results as authoritative where determinate. Never raises into the caller's pipeline."""
    rule_nodes, pols = _nodes_of_type(G, "PolicyRule"), _nodes_of_type(G, "Policy")
    if not (COMPILED_POLICY_ENABLED and HAS_POLICY_COMPILER and rule_nodes and pols and (rulebook_text or "").strip()): return None
    pol_id, pol = pols[0]
    rb_doc = G.nodes.get(pol.get("source_document_id"), {})
    rulebook = {"filename": pol.get("source_file"), "document_id": pol.get("source_document_id"), "sha256": rb_doc.get("file_hash"),
                "span_basis": "character offsets in the redacted extracted rulebook text passed to the compiler"}
    summary: Dict[str, Any] = {"status": "NOT_COMPILED", "rulebook": rulebook,
                               "engine": "policy_compiler (LLM interprets rule text) + rule_engine (deterministic; the only executor)"}
    G.nodes[pol_id]["compiled_policy"] = summary
    text = redact_pii(rulebook_text)
    try:
        result = _compile_policy(text, None, _COMPILER_MIN_CONFIDENCE)
    except ValueError as e:  # e.g. rulebook too long: never truncated silently
        summary["error"] = f"not compiled: {e}"; return summary
    except _CompilationError as e:
        summary["error"] = f"policy could not be compiled: {e}"; return summary
    except Exception as e:
        logger.exception("Policy compilation crashed (continuing with legacy engine)")
        summary["error"] = f"unexpected compilation error: {str(e)[:200]}"; return summary

    normalized_trigger_aliases = _normalize_confirmed_violation_trigger_aliases(result.rules)
    lines = [(n, _ws_norm(redact_pii(d.get("original_text") or d.get("condition") or ""))) for n, d in rule_nodes]
    lines = [(n, ln) for n, ln in lines if len(ln) >= 8]
    inputs = graph_to_rule_inputs(G, result.rules)
    normalized_amount_aliases = _normalize_transaction_amount_field_aliases(result.rules, inputs["facts"])
    valid = [cr for cr in result.rules if cr.status == "VALID"]  # NEEDS_REVIEW rules are never executed
    base_rules = [cr for cr in valid if _is_base_obligation_rule(cr)]  # evaluated first; confirmed-violation facts are derived from these
    trigger_rules = [cr for cr in valid if not _is_base_obligation_rule(cr)]
    by_rule: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    eval_error: Optional[str] = None
    ev: Optional[Dict[str, Any]] = None
    all_results: List[Dict[str, Any]] = []
    try:
        if base_rules:
            base_ev = _evaluate_compiled_rules(base_rules, inputs["facts"], inputs["evidence"], inputs["evaluation_date"], False, inputs["fx_rates"], inputs["strict_units"])
            all_results.extend(base_ev.get("rule_results", []))
            _derive_confirmed_violation_facts(valid, all_results, result, inputs["facts"])
            if trigger_rules:
                trigger_ev = _evaluate_compiled_rules(trigger_rules, inputs["facts"], inputs["evidence"], inputs["evaluation_date"], False, inputs["fx_rates"], inputs["strict_units"])
                all_results.extend(trigger_ev.get("rule_results", []))
            ev = {**base_ev, "rule_results": all_results}
        elif valid:
            ev = _evaluate_compiled_rules(valid, inputs["facts"], inputs["evidence"], inputs["evaluation_date"], False, inputs["fx_rates"], inputs["strict_units"])
            all_results.extend(ev.get("rule_results", []))
        for rr in all_results:
            by_rule[rr["rule_id"]].append(rr)
    except Exception as e:
        logger.exception("Compiled rule evaluation failed (legacy engine continues)")
        eval_error = str(e)[:300]

    unmapped: List[Dict[str, Any]] = []
    for cr in result.rules:
        src = _ws_norm(cr.source_text)
        host, basis = None, None
        inside = [n for n, ln in lines if src and src in ln]
        if inside:
            host, basis = inside[0], "clause_within_rule_line" + ("" if len(inside) == 1 else f" (first of {len(inside)} matching lines)")
        else:
            within = [(n, ln) for n, ln in lines if src and ln in src]
            if within: host, basis = max(within, key=lambda x: len(x[1]))[0], "rule_line_within_clause"
        rd = G.nodes[host] if host else {}
        entry = {"rule_id": cr.rule_id, "policy_id": cr.policy_id, "status": cr.status, "rule_type": cr.rule_type.value, "severity": cr.severity.value, "confidence": cr.confidence,
                 "expression": cr.expression, "source_text": cr.source_text, "source_span": _compiled_source_span(text, cr.source_text), "ambiguities": cr.ambiguities, "issues": cr.issues,
                 "mapping_basis": basis, "compiled_rule": cr.model_dump(mode="json"),
                 "rulebook": {**rulebook, "location": rd.get("source_location"), "line": rd.get("source_line"), "evidence_id": rd.get("source_evidence_id")}}
        entry["result"] = _compiled_entry_result(entry, by_rule.get(cr.rule_id, []), inputs["node_ids"], eval_error)
        if host: G.nodes[host].setdefault("compiled_rules", []).append(entry)
        else: unmapped.append({k: entry[k] for k in ("rule_id", "status", "expression", "source_text", "source_span")} | {"verdict": entry["result"]["verdict"]})
    summary.update(status=result.status, policy_id=result.policy_id, stats=result.stats, ambiguous_policy=result.ambiguous_policy, ambiguity_reasons=result.ambiguity_reasons[:20],
                   needs_review_not_executed=[cr.rule_id for cr in result.rules if cr.status != "VALID"],
                   rejected=[{"errors": x.errors, "source_text": x.source_text} for x in result.rejected][:20], unparsed_statements=result.unparsed_statements[:20],
                   unmapped_rules=unmapped, evaluation_error=eval_error,
                   normalized_confirmed_violation_trigger_aliases=normalized_trigger_aliases,
                   normalized_transaction_amount_field_aliases=normalized_amount_aliases,
                   inputs={"fact_entities": sorted(inputs["facts"]), "records": {k: len(v) for k, v in inputs["node_ids"].items()},
                           "evidence_supplied": inputs["evidence"] is not None,
                           "evidence_note": ("typed Evidence nodes supplied" if inputs["evidence"] is not None else "no Evidence node carries a structured evidence_type: evidence NOT supplied; evidence-requiring rules are INDETERMINATE"),
                           "evaluation_date": (ev or {}).get("evaluation_date"), "evaluation_date_source": "engine default (today, UTC): no evaluation date is available to the pipeline",
                           "fx_rates": "none available: cross-currency comparisons are INDETERMINATE", "strict_units": inputs["strict_units"]})
    return summary

def _compiled_node_result(rd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use a compiled result only when every compiled rule attached to this PolicyRule is determinate.

    A determinate sibling must not mask an indeterminate or unexecuted obligation on the same rule line.
    Otherwise the compiled result is not authoritative and the legacy result remains visible.
    """
    entries = list(rd.get("compiled_rules") or [])
    if not entries:
        return None
    if any(
        not (entry.get("result") or {}).get("executed")
        or (entry.get("result") or {}).get("verdict") == "INDETERMINATE"
        for entry in entries
    ):
        return None
    raw = _compiled_top(entries)
    viol = list(dict.fromkeys(n for e in entries for n in e["result"]["violating_node_ids"]))  # record-level outcomes of every determinate entry
    sat = [n for n in dict.fromkeys(n for e in entries for n in e["result"]["satisfying_node_ids"]) if n not in viol]
    parts = []
    for e in entries:
        r = e["result"]
        parts.append(f"{e['expression']} => {r['verdict']} (records: {r.get('counts')}" + (f"; {'; '.join(r['reasons'][:3])}" if r.get("reasons") else "") + (f"; action required: {r['action_required']}" if r.get("action_required") else "") + ")")
    sevs = [e["result"].get("severity") for e in entries if e["result"]["verdict"] == "VIOLATION" and e["result"].get("severity")]
    sev = max(sevs, key=lambda s_: _COMPILED_SEV.get(s_, 0)).upper() if sevs else ("MEDIUM" if raw == "VIOLATION" else None)
    return {"verdict": _COMPILED_TO_LEGACY[raw], "raw_verdict": raw, "violating": viol, "satisfying": sat, "severity": sev,
            "rationale": "COMPILED RULE (rulebook text interpreted by an LLM, executed by the deterministic rule engine; authoritative for this rule): " + " | ".join(parts)}

def _decision_evidence_ids(G: nx.MultiDiGraph, nodes: List[str]) -> List[str]:
    return list(dict.fromkeys(e for n in nodes for e in _supporting_evidence_ids(G, n) if not G.nodes[e].get("context_only") and not _is_policy_source_evidence(G.nodes[e])))

def _compiled_decision_fields(G: nx.MultiDiGraph, rd: Dict[str, Any], legacy: Dict[str, Any], comp: Optional[Dict[str, Any]], verdict: str,
                              used_ev: List[str], viol: List[str], sat: List[str]) -> Dict[str, Any]:
    """Attributes that keep legacy and compiled results distinguishable on the Decision node (plus evidence for/against and rulebook provenance)."""
    entries = rd.get("compiled_rules") or []
    opposite = sat if verdict == "VIOLATION" else viol if verdict == "SATISFIED" else []
    f: Dict[str, Any] = {
        "evaluation_engine": "compiled_rule_engine" if comp else "legacy_dsl",
        "legacy_verdict": legacy["verdict"], "legacy_rationale": legacy["rationale"], "legacy_recognized": legacy["parsed"], "legacy_unevaluated_reason": legacy["unevaluated_reason"],
        "compiled_verdict": _compiled_top(entries),
        "compiled_rules": [{**{k: e.get(k) for k in ("rule_id", "status", "rule_type", "severity", "confidence", "expression", "source_text", "source_span", "mapping_basis")},
                            "verdict": (e.get("result") or {}).get("verdict"), "executed": (e.get("result") or {}).get("executed"),
                            "missing_facts": (e.get("result") or {}).get("missing_facts"), "reasons": ((e.get("result") or {}).get("reasons") or [])[:5]} for e in entries],
        "compiled_missing_facts": sorted({m for e in entries for m in ((e.get("result") or {}).get("missing_facts") or [])}),
        "supporting_evidence_ids": list(used_ev), "contradicting_evidence_ids": [e for e in _decision_evidence_ids(G, opposite) if e not in used_ev],
        "rulebook_provenance": {"file": rd.get("source_file"), "location": rd.get("source_location"), "line": rd.get("source_line"), "document_id": rd.get("source_document_id"),
                                "evidence_id": rd.get("source_evidence_id"), "rulebook_sha256": G.nodes.get(rd.get("source_document_id"), {}).get("file_hash"),
                                "compiled_source_spans": [e.get("source_span") for e in entries]}}
    if comp:
        f["legacy_disagrees"] = legacy["verdict"] in ("VIOLATION", "SATISFIED") and legacy["verdict"] != verdict
        f["compiled_verdict_downgraded"] = verdict != _COMPILED_TO_LEGACY.get(comp["raw_verdict"])
    return f

def evaluate_policy_rules(G: nx.MultiDiGraph) -> List[str]:
    """Evaluate every PolicyRule once. Creates Decision (+Risk on VIOLATION) nodes and VIOLATES/SATISFIES/
    EVALUATES/SUPPORTS/HAS_RISK/BELONGS_TO edges. Idempotent per rule. Returns new Decision node ids."""
    decision_ids: List[str] = []
    for rule_id, rd in list(G.nodes(data=True)):
        if rd.get("type") != "PolicyRule" or rd.get("engine_run"): continue
        spec = parse_policy_rule(rd.get("condition", ""))
        uneval_reason = _unevaluated_reason(rd.get("condition", "")) if spec is None else None
        if spec is None:
            verdict, rationale, viol, sat = "UNEVALUATED", (f"NOT EVALUATED: {uneval_reason} No pass/violation was assigned and compliance with this rule was NOT checked. "
                                                            "Context only; missing extracted text is not proof of absence."), [], []
            severity = None
        else:
            verdict, rationale, viol, sat = _evaluate_spec(G, spec)
            severity = spec["severity"]
            _absence_violation = spec["subject"] == "KEYWORD" and spec["mode"] == "REQUIRE" and verdict == "VIOLATION" and not viol
            _had_viol, _had_sat = bool(viol), bool(sat)
            viol, _w1 = _gate_basis(G, viol)  # no VIOLATES / SATISFIES edge without non-heuristic evidence that traces to a source Document
            sat, _w2 = _gate_basis(G, sat)
            if _w1 or _w2:
                rationale += f" {len(_w1) + len(_w2)} basis node(s) were NOT linked (VIOLATES/SATISFIES withheld: no qualifying supporting evidence)."
            if verdict == "VIOLATION" and _had_viol and not viol and not _absence_violation:
                verdict, rationale = "INCONCLUSIVE", rationale + " No violating node has qualifying evidence, so no violation was assigned."
            elif verdict == "SATISFIED" and _had_sat and not sat and not _had_viol:
                verdict, rationale = "INCONCLUSIVE", rationale + " No satisfying node has qualifying evidence, so no pass was assigned."
            if verdict == "UNEVALUATED":  # parsed, but required support/configuration is missing: nothing was evaluated
                uneval_reason = rationale
                rationale = f"NOT EVALUATED: {rationale} No pass/violation was assigned and compliance with this rule was NOT checked. Context only; missing extracted text is not proof of absence."
            _gaps = extraction_gaps(G)
            if _gaps and verdict in ("VIOLATION", "SATISFIED", "NOT_APPLICABLE"):
                rationale += f" WARNING: extraction incomplete for {len(_gaps)} document(s); result covers extracted data only."
        _legacy = {"verdict": verdict, "rationale": rationale, "parsed": spec is not None, "unevaluated_reason": uneval_reason}  # legacy DSL result, always preserved on the Decision
        _comp = _compiled_node_result(rd)  # determinate result of the deterministic compiled-rule engine (None -> legacy stays authoritative)
        if _comp is not None:
            verdict, rationale, severity, uneval_reason = _comp["verdict"], _comp["rationale"], _comp["severity"], None
            viol, sat = list(_comp["violating"]), list(_comp["satisfying"])
            _hv, _hs = bool(viol), bool(sat)
            viol, _w1 = _gate_basis(G, viol)  # same basis gate as legacy: no VIOLATES / SATISFIES edge without non-heuristic evidence tracing to a source Document
            sat, _w2 = _gate_basis(G, sat)
            if _w1 or _w2:
                rationale += f" {len(_w1) + len(_w2)} basis node(s) were NOT linked (VIOLATES/SATISFIES withheld: no qualifying supporting evidence)."
            if verdict == "VIOLATION" and not viol:
                verdict, rationale = "INCONCLUSIVE", rationale + " No violating node has qualifying evidence, so no violation was assigned."
            elif verdict == "SATISFIED" and not sat and not _hv:
                verdict, rationale = "INCONCLUSIVE", rationale + " No satisfying node could be mapped to the graph, so no pass was assigned."
            _gaps = extraction_gaps(G)
            if _gaps and verdict in ("VIOLATION", "SATISFIED", "NOT_APPLICABLE"):
                rationale += f" WARNING: extraction incomplete for {len(_gaps)} document(s); result covers extracted data only."
            rationale += f" [Legacy DSL result for this rule line: {_legacy['verdict']}; not authoritative here.]"
        logger.info(f"Policy rule {rule_id}: parsed={spec is not None} verdict={verdict} text={redact_pii(rd.get('condition', ''))[:120]!r}")
        G.nodes[rule_id]["engine_run"] = True
        G.nodes[rule_id]["legacy_parsed"] = spec is not None
        G.nodes[rule_id]["compiled_authoritative"] = _comp is not None
        G.nodes[rule_id]["parsed"] = (spec is not None or _comp is not None)  # RECOGNIZED only (legacy DSL or executed compiled rule): text mapped to the supported rule format; says nothing about being executable
        G.nodes[rule_id]["executable"] = (spec is not None or _comp is not None) and verdict != "UNEVALUATED"  # deterministic rule AND required support/configuration available
        G.nodes[rule_id]["attempted"] = G.nodes[rule_id]["executable"]  # evaluation actually started (never true for UNEVALUATED)
        G.nodes[rule_id]["evaluated"] = verdict in ("VIOLATION", "SATISFIED")  # a determinate result exists
        G.nodes[rule_id]["unevaluated_reason"] = uneval_reason
        G.nodes[rule_id]["normalized_text"] = _normalize_rule_line(rd.get("condition", ""))

        dec_data = DecisionNodeSchema(node_id=_nid("dec"), verdict=verdict, rule_id=rule_id, rationale=rationale).model_dump()
        dec_id = dec_data["node_id"]
        G.add_node(dec_id, **dec_data)
        G.add_edge(dec_id, rule_id, relation="EVALUATES")
        for _, pol, ed in list(G.out_edges(rule_id, data=True)):
            if ed.get("relation") == "BELONGS_TO" and G.nodes[pol].get("type") == "Policy":
                G.add_edge(dec_id, pol, relation="BELONGS_TO")

        for node in viol: G.add_edge(node, rule_id, relation="VIOLATES")
        for node in sat: G.add_edge(node, rule_id, relation="SATISFIES")

        basis_nodes = viol if verdict == "VIOLATION" else sat
        used_ev = list(dict.fromkeys(e for node in basis_nodes for e in _supporting_evidence_ids(G, node)
                                     if not G.nodes[e].get("context_only") and not _is_policy_source_evidence(G.nodes[e])))
        for ev_id in used_ev:
            G.add_edge(ev_id, dec_id, relation="SUPPORTS")
        _absence = _comp is None and spec is not None and spec["subject"] == "KEYWORD" and spec["mode"] == "REQUIRE" and verdict == "VIOLATION" and not viol
        G.nodes[dec_id].update(violation_status=("CONFIRMED_BY_DETERMINISTIC_RULE" if (verdict == "VIOLATION" and not _absence) else
                                                 "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE" if _absence else None),
                               heuristic_inputs_used=False,  # contradictions / shared terms are never inputs of a verdict
                               absence_scope_evidence_ids=([n for n, _ in _investigation_evidence(G)] if _absence else []),
                               result_source=("compiled_policy_engine" if _comp is not None else "deterministic_policy_engine"), evidence_used=used_ev, basis_node_ids=list(basis_nodes),
                               result_reason=rationale, parsed_spec=spec, rule_source_file=rd.get("source_file"), rule_source_location=rd.get("source_location"), unevaluated_reason=uneval_reason,
                               extraction_gap_documents=(extraction_gaps(G) if verdict == "INCONCLUSIVE" else []))
        G.nodes[dec_id].update(_compiled_decision_fields(G, rd, _legacy, _comp, verdict, used_ev, viol, sat))

        if verdict == "VIOLATION":
            risk_data = RiskNodeSchema(node_id=_nid("risk"), severity=severity or "MEDIUM", rule_id=rule_id).model_dump()
            G.add_node(risk_data["node_id"], **risk_data, severity_source=("compiled_rule" if _comp is not None else "rule_configured" if _RULE_SEVERITY.search(rd.get("condition", "")) else "default_when_unspecified"),
                       decision_id=dec_id, basis_node_ids=list(basis_nodes), evidence_ids=list(used_ev),
                       basis="deterministic_policy_violation" if not _absence else "required_text_absent_within_extracted_scope")
            G.add_edge(dec_id, risk_data["node_id"], relation="HAS_RISK")
            for ev_id in used_ev:  # Risk is tied to the same case evidence that produced the violation (only evidence that exists)
                G.add_edge(ev_id, risk_data["node_id"], relation="SUPPORTS", method="violation_basis")
        decision_ids.append(dec_id)
    return decision_ids

def _evidence_source_docs(G: nx.MultiDiGraph, ev_id: str) -> List[Dict[str, Any]]:
    out = []
    for _, doc, d in G.out_edges(ev_id, data=True):
        if d.get("relation") == "DERIVED_FROM" and G.nodes[doc].get("type") == "Document":
            out.append({"document_id": doc, "filename": G.nodes[doc].get("filename"), "location": d.get("location")})
    return out

# --- CONTRADICTION HEURISTIC ---
# Deterministic but heuristic: two Transactions with the same label (up to 3 words before the amount, >=2 words
# after stop-word removal) and currency, DIFFERENT amount, backed by evidence from DIFFERENT documents.
# Edges: Evidence(of one) --CONTRADICTS--> Transaction(of the other), tagged method="same_label_different_amount".
# This can false-positive (e.g. same label legitimately repeated); it is capped and always labelled heuristic.
def _txn_label(raw_text: str, match_start: int) -> Optional[str]:
    prefix = raw_text[:match_start].split("\n")[-1]
    tokens = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", prefix)[-3:]]
    tokens = [w for w in tokens if w not in _STOP_WORDS and w not in ("rs", "inr", "usd", "eur", "gbp")]
    return " ".join(tokens) if len(tokens) >= 2 else None

# --- AMOUNT ROLE: what does an amount represent? (limit/threshold vs actual transaction vs unclear) ---
# Decided from wording on the SAME line plus the source file name. A policy limit is never an actual transaction.
_POLICY_AMOUNT_CUES = re.compile(r'(?i)\b(limits?|thresholds?|maximum|max|cap|capped|ceiling|allowances?|allowed|permitted|permissible|up\s+to|not\s+(?:to\s+)?exceed|(?:may|must|shall)\s+not\s+exceed|exceed(?:s|ing)?|above\s+which|per\s+diem|approval\s+required|budget(?:ed)?|policy|policies)\b')
_TXN_AMOUNT_CUES = re.compile(r'(?i)\b(invoice|inv|receipt|paid|payment|payments|purchase|purchased|expenses?|reimburse\w*|claim(?:ed)?|voucher|txn|transaction|bill(?:ed)?|charged|spent)\b')
_POLICY_DOC_HINT = re.compile(r'(?i)(policy|policies|rulebook|guideline|procedure|handbook|\bsop\b|regulation)')
ROLE_POLICY, ROLE_TXN, ROLE_UNCLEAR = "POLICY_THRESHOLD", "TRANSACTION", "UNCLEAR"
# POLICY_CONTEXT = an amount LISTED inside a policy-type document whose meaning (limit, price, budget...) is not stated.
# Like POLICY_THRESHOLD it is context only: never a transaction record, never a side of a discrepancy, never tied to an employee.
ROLE_POLICY_CONTEXT = "POLICY_CONTEXT"

def _is_policy_role(role: Optional[str]) -> bool:
    return role in (ROLE_POLICY, ROLE_POLICY_CONTEXT)

def looks_like_policy_document(text: str, file_name: str = "") -> bool:
    """File name OR the document's first non-empty line (its title) says policy/rulebook/guideline/handbook/SOP/regulation."""
    if _POLICY_DOC_HINT.search(file_name or ""): return True
    for line in (text or "").splitlines():
        if line.strip(): return bool(_POLICY_DOC_HINT.search(line.strip()[:200]))
    return False

def classify_amount_role(text: str, start: int, end: int, file_name: str = "", attrs: Optional[Dict[str, str]] = None, doc_is_policy: bool = False) -> Tuple[str, str]:
    """-> (role, reason). Never guesses: conflicting or missing cues give UNCLEAR (or POLICY_CONTEXT inside a policy document)."""
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    le = len(text) if le < 0 else le
    window = text[max(ls, start - 100):min(le, end + 100)]
    pol, txn = _POLICY_AMOUNT_CUES.search(window), _TXN_AMOUNT_CUES.search(window)
    has_ref = bool((attrs or {}).get("ref_id"))
    doc_policy = doc_is_policy or bool(_POLICY_DOC_HINT.search(file_name or ""))
    if pol and not txn and not has_ref:
        return ROLE_POLICY, f"limit/threshold wording ('{pol.group(0).lower()}') next to the amount and no transaction wording or reference ID"
    if pol and (txn or has_ref):
        return ROLE_UNCLEAR, "both limit/threshold wording and transaction wording/reference ID appear near the amount"
    if txn or has_ref:
        return ROLE_TXN, ("transaction wording ('" + txn.group(0).lower() + "')" if txn else "a reference/invoice ID in the same row") + " near the amount, no limit/threshold wording"
    if doc_policy:
        return ROLE_POLICY_CONTEXT, "amount is policy-listed information inside a policy document; its meaning (limit, price, budget) is not stated; not a transaction record and not tied to any employee"
    return ROLE_UNCLEAR, "no wording or identifier near the amount states whether it is a transaction or a limit"

def _entry_location(text: str, line_start: int, file_name: str = "") -> str:
    """Source location of the line starting at line_start. Spreadsheet records are extracted as 'Row N:' under a 'Sheet:' header, so the
    worksheet row number (and sheet) is reported, never the extracted-text line number. Other files: 'extracted-text line N'."""
    m = re.match(r'Row (\d+):', text[line_start:line_start + 24])
    if m and (file_name or "").lower().endswith(".xlsx"):
        sh = text.rfind("\nSheet: ", 0, line_start)
        if sh >= 0:
            name = text[sh + 8:text.find("\n", sh + 8)].strip()
            return f"{name}, Row {m.group(1)}"
        return f"Row {m.group(1)}"
    return f"extracted-text line {text.count(chr(10), 0, line_start) + 1}"

def collect_amount_entries(text: str, file_name: str = "", policy_source: bool = False) -> List[Dict[str, Any]]:
    """Amounts found in ONE source text, with file, line, role and reason. context_only=True means the amount is policy/limit
    context (limit wording, a rulebook, or an amount listed in a policy document), never a transaction record."""
    out: List[Dict[str, Any]] = []
    doc_policy = looks_like_policy_document(text, file_name)
    for m in MONEY_PATTERN.finditer(text or ""):
        try: val = float(m.group(2).replace(',', ''))
        except ValueError: continue
        if not math.isfinite(val): continue
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end()); le = len(text) if le < 0 else le
        line_txt = text[ls:le]
        attrs = _extract_txn_attributes(line_txt) if len(MONEY_PATTERN.findall(line_txt)) == 1 else {}
        role, reason = classify_amount_role(text, m.start(), m.end(), file_name, attrs, doc_policy)
        context_only = False
        if policy_source:
            context_only, reason = True, "amount stated in the rulebook/policy text: " + reason
            if role == ROLE_UNCLEAR: role = ROLE_POLICY_CONTEXT
        elif _is_policy_role(role): context_only = True
        out.append({"file": file_name, "line": text.count("\n", 0, m.start()) + 1, "location": _entry_location(text, ls, file_name), "excerpt": line_txt, "currency": _normalize_currency(m.group(1)), "amount": val,
                    "role": role, "reason": reason, "attrs": attrs, "context_only": context_only})
        if len(out) >= 200: break
    return out

def generate_amount_role_report(text: str, file_name: str = "") -> str:
    """Pre-check for V1: states what each amount appears to represent so limits are not compared with transactions."""
    rows = [f"Amount #{i} ({e['currency']} {e['amount']:g}) in {e['file']}, {e['location']}: role={e['role']} ({e['reason']})" + (" [policy-listed context: NOT a transaction record]" if e["context_only"] else "")
            for i, e in enumerate(collect_amount_entries(text, file_name)[:30], 1)]
    if not rows: return ""
    return ("\n--- [SYSTEM AMOUNT-ROLE PRE-CHECK] (wording-based; not verified) ---\n" + "\n".join(rows) +
            "\nPOLICY_THRESHOLD / POLICY_CONTEXT = policy-listed information, NOT a transaction record (no employee association unless the source states one). UNCLEAR = meaning not established. "
            "Never present a policy-listed amount and a TRANSACTION amount as two conflicting transaction records.\n"
            "--------------------------------------------\n\n")

def _govid_occurrences(raw_text: str, file_name: str) -> List[str]:
    """'file @ location' for every ID-like value actually present in this file's extracted text (digits never kept)."""
    out: List[str] = []
    for c in find_govid_candidates(raw_text):
        ls = raw_text.rfind("\n", 0, c["start"]) + 1
        out.append(f"{file_name} @ {_entry_location(raw_text, ls, file_name)}")
    return out

def _govid_source_lines(G: nx.MultiDiGraph) -> List[str]:
    """'file @ location (xN)' for Evidence that actually contains a Government-ID-like Entity (Evidence --SUPPORTS--> GovID). No other document is implied."""
    cnt: Counter = Counter()
    for gid, _ in _nodes_of_type(G, "Entity", "GovID"):
        for e in _in_edge_sources(G, gid, "SUPPORTS", "Evidence"):
            for x in _evidence_source_docs(G, e["node_id"]): cnt[f"{x['filename']} @ {x['location']}"] += 1
    return [f"{k} (x{v})" if v > 1 else k for k, v in sorted(cnt.items())]

def build_v1_amount_brief(entries: List[Dict[str, Any]], cov: Optional[Dict[str, Any]] = None,
                          govid_statuses: Optional[List[str]] = None, link_log: Optional[List[Dict[str, Any]]] = None, govid_sources: Optional[List[str]] = None) -> str:
    """Deterministic, cross-document brief placed at the START of V1's context (V1 stays document-based: built from V1's own
    extracted text, not from the graph). Sections: A record amounts, B policy-listed context, C record-to-record comparisons,
    D Government ID status, E supporting links. Policy-listed amounts are never paired with record amounts; record-vs-record
    differences are POSSIBLE discrepancies unless a shared identifier links them (a shared name/currency is NOT enough)."""
    if not entries and not govid_statuses and not link_log: return ""
    ctx_items = [e for e in entries if e["context_only"]]
    recs = [e for e in entries if not e["context_only"]]
    def ref(e): return f"{e['file']}, {e['location']}"
    out = "--- [SYSTEM V1 AMOUNT RECONCILIATION BRIEF] (deterministic, wording-based; not verified; sources listed for lineage) ---\n"
    out += "A. RECORD AMOUNTS (candidate transaction records; SOURCE FACTS, roles are heuristic):\n"
    out += "".join(f"- {e['currency']} {e['amount']:g} | {ref(e)} | role={e['role']}\n" for e in recs[:20]) or "- none identified\n"
    out += "B. POLICY-LISTED INFORMATION / LIMIT CONTEXT (NOT transaction records; kept separate; source preserved; no employee or transaction association):\n"
    out += "".join(f"- {e['currency']} {e['amount']:g} | {ref(e)} | role={e['role']} | {e['reason']}\n" for e in ctx_items[:20]) or "- none identified\n"
    out += "C. RECORD-TO-RECORD COMPARISONS (different documents, same currency, different amount):\n"
    pairs, done = [], False
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            x, y = recs[i], recs[j]
            if x["file"] == y["file"] or x["currency"] != y["currency"] or x["amount"] == y["amount"]: continue
            strength, matched, _c = _match_transactions({"attributes": x["attrs"], "amount_role": x["role"]}, {"attributes": y["attrs"], "amount_role": y["role"]})
            if strength == "distinct": continue  # different reference IDs: different transactions
            if strength == "strong":
                out += (f"- {x['currency']} {x['amount']:g} ({ref(x)}) vs {y['currency']} {y['amount']:g} ({ref(y)}): amounts differ on records linked by "
                        f"{', '.join(matched)}; discrepancy needs review (still NOT a policy violation).\n")
            else:
                out += (f"- {x['currency']} {x['amount']:g} ({ref(x)}) vs {y['currency']} {y['amount']:g} ({ref(y)}): POSSIBLE DISCREPANCY requiring reconciliation. "
                        f"UNCERTAIN: no shared transaction/reference ID or date + party + expense type links these records. A shared employee name and/or currency does NOT show they are the same transaction, "
                        f"so they may be different transactions (roles: {x['role']} / {y['role']}).\n")
            pairs.append(1)
            if len(pairs) >= 10: done = True; break
        if done: break
    if not pairs: out += "- no differing record amounts across different documents\n"
    gstat = govid_statuses or []
    if gstat:
        cfg = govid_validation_config()
        out += f"D. GOVERNMENT ID ({len(gstat)} ID-like value(s) detected; digits never shown, represented as {REDACTED_GOVID}):\n"
        if not cfg["complete"]:
            out += (f"- Status: UNVERIFIED. Single explanation: {govid_unvalidated_reason(cfg)}. No validity, invalidity or authenticity claim is made.\n"
                    f"- {GOVID_NEXT_STEP}\n")
        else:
            out += (f"- Validation performed with the configured method ({cfg['method']}); results: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(gstat).items())) +
                    ". Arithmetic consistency only; authenticity and ownership are NOT verified.\n")
    if gstat:
        _gs = Counter(govid_sources or [])
        out += ("- ID-like value(s) detected ONLY in these extracted sources: " + (", ".join(f"{k} (x{v})" for k, v in sorted(_gs.items())) or "source location not recorded") +
                ". Attribute an ID only to these sources; a file not listed here had no ID-like value in its extracted text, and a shared employee name is NOT evidence that another document contains an ID.\n")
    if link_log:
        _links = normalize_link_log(link_log)
        out += f"E. SUPPORTING LINKS (system retrieval status only; each unique link listed once; {link_status_summary(_links)}):\n"
        for l in _links[:20]: out += _link_status_line(l, md=False, for_llm=True)
        out += LINK_VS_FINDING_NOTE + "\n"
    out += ("MANDATORY INTERPRETATION (applies to Executive Summary, Key Findings, Recommended Actions):\n"
            "1. Section B amounts are policy-listed context. NEVER compare a Section B amount with a Section A amount as a contradiction, never call one 'consistent with' or 'confirming' a transaction record, "
            "never treat one as a transaction, and never associate one with an employee unless the source states it.\n"
            "2. Section C pairs are 'possible discrepancy requiring reconciliation' unless the line says they are linked by an identifier; never 'confirmed contradiction', 'material discrepancy' or 'policy non-conformity'. A shared name is not proof of the same transaction.\n")
    if cov and cov.get("evaluated", 0) == 0:
        out += "3. NO policy rule was deterministically evaluated to a result: do not state or imply that a policy rule was violated. Statements about the rulebook are LLM INTERPRETATION.\n"
    else:
        out += "3. Only a deterministic VIOLATION verdict reported in the system note may be called a policy violation; amount differences, limits and readings of rulebook text may not.\n"
    out += ("4. Severity for amount differences is 'Undetermined, pending reconciliation', not High/Critical; the overall assessment is 'inconclusive pending reconciliation', not non-compliant. Next steps are reconciliation actions: obtain the source transaction records, "
            "confirm whether the records describe the same transaction, confirm what the policy-listed amount represents. Never recommend changing, resubmitting or matching a transaction to a Section B amount merely because the policy lists it, and never recommend linking or associating a Section B amount with an actual expense claim or employee. Do not invent identifiers or policy meaning.\n"
            "5. Government ID: say only 'ID present, UNVERIFIED' with the single explanation in Section D; keep the next step conditional and never promise VALID/INVALID.\n"
            "6. Links: an unretrieved link is unassessed, not invalid/fraudulent/legitimate; never present it as a receipt or proof of the expense. NOT RETRIEVED is a retrieval outcome; INCONCLUSIVE is a finding: never use one word for the other, and never call a link 'inconclusive'.\n"
            "7. Keep SOURCE FACTS, HEURISTIC INFERENCES, policy-listed context and deterministic policy results distinct, and keep every section consistent with these statuses.\n"
            "8. Source excerpts may contain malformed URLs (e.g. a trailing ']' or '%5D'). Never list or repeat URLs yourself: the system's Supporting links list is the only link list.\n"
            "9. Government ID attribution: name a document as containing an ID ONLY if Section D lists it as a source. Never infer an ID in a document from another document or from a shared employee name.\n"
            "---------------------------------------------------------------------\n\n")
    return out

# Transaction attributes (only from LABELED text in the same evidence row, and only when that row holds exactly ONE amount,
# otherwise the attribute could belong to another amount). Spreadsheet rows without header labels yield no attributes.
_ATTR_REF = re.compile(r'(?i)\b(?:invoice|inv|txn|transaction|trans|ref(?:erence)?|receipt|voucher|purchase\s+order|po|utr)\b\s*(?P<marker>no\.?|num(?:ber)?|id|#)?\s*[:=#\-]?\s*(?P<tok>[A-Z0-9][A-Z0-9\-/_]{2,31})')
_ATTR_LABELS = {"person": r'(?:employee|person|claimant|submitted\s+by|requested\s+by|customer|client|paid\s+by)',
                "vendor": r'(?:vendor|supplier|payee|merchant|paid\s+to|seller)',
                "expense_type": r'(?:expense\s+type|expense\s+category|expense|category|cost\s+type)'}
_ALL_LABELS = "|".join(_ATTR_LABELS.values())
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_PATTERNS = [(re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'), lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
                  (re.compile(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b'), lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"),  # day/month order NOT guessed
                  (re.compile(r'(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})\b'), lambda m: f"{m.group(3)}-{_MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"),
                  (re.compile(r'(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b'), lambda m: f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}")]
_ATTR_KEYS = ("ref_id", "date", "vendor", "person", "expense_type")

def _extract_txn_attributes(text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    refs = set()
    for m in _ATTR_REF.finditer(text or ""):
        tok = m.group("tok")
        if not re.search(r'\d', tok): continue
        if not m.group("marker") and not re.search(r'[A-Za-z]', tok) and ":" not in m.group(0): continue  # bare number after "transaction" is likely the amount
        norm = re.sub(r'[^A-Z0-9]', '', tok.upper())
        if len(norm) >= 3: refs.add(norm)
    if len(refs) == 1: attrs["ref_id"] = next(iter(refs))
    dates = set()
    for pat, fmt in _DATE_PATTERNS:
        for m in pat.finditer(text or ""):
            try: dates.add(fmt(m))
            except (KeyError, ValueError): pass
    if len(dates) == 1: attrs["date"] = next(iter(dates))
    for key, lab in _ATTR_LABELS.items():
        vals = set()
        for m in re.finditer(rf'(?i)\b{lab}\s*[:=]\s*([^|¦;\n]+?)(?=\s*(?:[|¦;\n]|$|\b(?:{_ALL_LABELS})\s*[:=]))', text or ""):
            v = re.sub(r'\s+', ' ', m.group(1)).strip().lower()[:60]
            if v and not MONEY_PATTERN.fullmatch(v): vals.add(v)
        if len(vals) == 1: attrs[key] = next(iter(vals))
    return attrs

def _match_transactions(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """-> (strength, matched attribute names, conflicting attribute names). strength:
      'strong'   = same reference/invoice ID, OR same date + same vendor/person + same expense type; no conflicting attribute
      'distinct' = both carry a reference ID and the IDs differ (different transactions: never linked)
      'weak'     = anything else (label / shared-name heuristics)"""
    aa, ab = a.get("attributes") or {}, b.get("attributes") or {}
    roles_ok = a.get("amount_role", ROLE_TXN) == ROLE_TXN and b.get("amount_role", ROLE_TXN) == ROLE_TXN  # legacy nodes: no role = as before
    matched = [k for k in _ATTR_KEYS if aa.get(k) and aa.get(k) == ab.get(k)]
    conflicts = [k for k in _ATTR_KEYS if aa.get(k) and ab.get(k) and aa[k] != ab[k]]
    if "ref_id" in conflicts: return "distinct", matched, conflicts
    if not conflicts:
        if "ref_id" in matched: return ("strong" if roles_ok else "weak"), matched, conflicts
        if "date" in matched and ("vendor" in matched or "person" in matched) and "expense_type" in matched: return ("strong" if roles_ok else "weak"), matched, conflicts
    return "weak", matched, conflicts

def _txn_evidence_docs(G: nx.MultiDiGraph, t: str) -> List[Tuple[str, set]]:
    return [(e, {x["document_id"] for x in _evidence_source_docs(G, e)}) for e in _supporting_evidence_ids(G, t)]

def _strong_txn_groups(G: nx.MultiDiGraph) -> Dict[tuple, List[str]]:
    """Transactions grouped by RELIABLE identity: same reference/invoice ID, or same date + vendor/person + expense type (same currency)."""
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for n, d in _nodes_of_type(G, "Transaction"):
        if _is_policy_role(d.get("amount_role")): continue  # a policy-listed amount is not a transaction record
        at = d.get("attributes") or {}
        cur = d.get("currency")
        if at.get("ref_id"): groups[("ref", at["ref_id"], cur)].append(n)
        if at.get("date") and (at.get("vendor") or at.get("person")) and at.get("expense_type"):
            groups[("combo", at["date"], at.get("vendor"), at.get("person"), at["expense_type"], cur)].append(n)
    return groups

def _link_pair(G: nx.MultiDiGraph, t1: str, t2: str, seen: set, budget: int, method: str, label: Optional[str], strength: str, reason: str) -> int:
    """ONE CONTRADICTS edge per unordered EVIDENCE pair (different documents), however many passes/nodes reach it. Canonical direction:
    lower evidence id --CONTRADICTS--> the Transaction derived from the other evidence. The comparison is record-vs-record: that Transaction node
    is NOT independent evidence for the record it came from. Edge stores both evidence ids + source refs, strength, matched/conflicting attributes
    (including the shared currency / name-or-label and the differing amount) and reason. Never creates a Decision/VIOLATES: not a policy finding."""
    n1, n2 = G.nodes[t1], G.nodes[t2]
    _, matched, conflicts = _match_transactions(n1, n2)
    strong = strength == "strong"
    roles = [n1.get("amount_role", ROLE_TXN), n2.get("amount_role", ROLE_TXN)]
    if not strong and matched:
        reason += f"; partial attribute overlap ({', '.join(matched)}) is insufficient for a confirmed link" + ("" if roles == [ROLE_TXN, ROLE_TXN] else " and at least one amount is not established as a transaction")
    uncertainty = "" if strong else ("Not linked by a shared reference/transaction ID or by date + party + expense type; a shared name/label and currency do not show the same transaction, so the amounts may belong to different transactions or periods, "
                                     f"or may not both be transaction records (amount roles: {roles[0]} / {roles[1]}). Requires reconciliation against source records.")
    cmp_matched, cmp_values = list(matched), {k: (n1.get("attributes") or {}).get(k) for k in matched}
    if n1.get("currency") and n1.get("currency") == n2.get("currency"):
        cmp_matched.append("currency"); cmp_values["currency"] = n1.get("currency")
    if method == "same_label_different_amount" and label:
        cmp_matched.append("normalized_label"); cmp_values["normalized_label"] = label
    elif method == "shared_term_different_amount" and label:
        cmp_matched.append("shared_term"); cmp_values["shared_term"] = label
    cmp_conflicts = list(conflicts)
    if n1.get("amount") != n2.get("amount"): cmp_conflicts.append("amount")
    info = {t1: _txn_evidence_docs(G, t1), t2: _txn_evidence_docs(G, t2)}
    created = 0
    for ev_a, docs_a in info[t1]:
        for ev_b, docs_b in info[t2]:
            if ev_a == ev_b or not docs_a or not docs_b or (docs_a & docs_b): continue  # must be different documents
            if G.nodes[ev_a].get("context_only") or G.nodes[ev_b].get("context_only"): continue  # policy context never takes part
            pair_key = "|".join(sorted((ev_a, ev_b)))
            if pair_key in seen: continue
            if created >= budget: return created
            seen.add(pair_key)
            if ev_a <= ev_b: src_ev, dst_ev, src_t, dst_t = ev_a, ev_b, t1, t2
            else: src_ev, dst_ev, src_t, dst_t = ev_b, ev_a, t2, t1
            refs = [{"evidence_id": e, "documents": _evidence_source_docs(G, e)} for e in sorted((ev_a, ev_b))]
            G.add_edge(src_ev, dst_t, relation="CONTRADICTS", method=method, label=label,
                       this_amount=G.nodes[src_t].get("amount"), other_amount=G.nodes[dst_t].get("amount"),
                       heuristic=not strong, match_strength=strength, confirmed=False, can_create_violation=False,
                       contradiction_label=("DISCREPANCY ON MATCHED RECORDS (unconfirmed; needs review)" if strong else "POSSIBLE CONTRADICTION (heuristic; not established)"),
                       confirmation_status="UNCONFIRMED",
                       status="DISCREPANCY_ON_MATCHED_TRANSACTION" if strong else "POSSIBLE_DISCREPANCY",
                       qualification="needs_review_not_a_policy_violation" if strong else "possible_not_proven",
                       link_reason=reason, matched_attributes=cmp_matched, matched_values=cmp_values, conflicting_attributes=cmp_conflicts,
                       source_transaction_id=src_t, target_transaction_id=dst_t, evidence_pair=pair_key, evidence_ids=sorted((ev_a, ev_b)),
                       counterpart_evidence_id=dst_ev, source_refs=refs,
                       comparison_basis="record_vs_record: Transaction nodes are derived from these evidence records and are not independent evidence",
                       amount_roles=roles, requires_reconciliation=not strong, uncertainty_reason=uncertainty)
            created += 1
    return created

def detect_contradictions(G: nx.MultiDiGraph, max_edges: Optional[int] = None) -> int:
    limit = MAX_CONTRADICTION_EDGES if max_edges is None else max_edges
    created = 0
    seen: set = set()
    linked: set = set()
    # Pass 1: STRONG matches (reference/invoice ID, or date + vendor/person + expense type), same currency, different amount.
    strong_groups = _strong_txn_groups(G)
    for key, txns in strong_groups.items():
        if len(txns) < 2 or len(txns) > 50: continue
        for i in range(len(txns)):
            for j in range(i + 1, len(txns)):
                t1, t2 = txns[i], txns[j]
                pair = frozenset((t1, t2))
                n1, n2 = G.nodes[t1], G.nodes[t2]
                if pair in linked or n1.get("amount") == n2.get("amount") or n1.get("currency") != n2.get("currency"): continue
                strength, matched, _c = _match_transactions(n1, n2)
                if strength != "strong": continue
                if created >= limit: return created
                reason = (f"matched on {', '.join(matched)} (+ same currency); amounts differ; evidence from different documents")
                c = _link_pair(G, t1, t2, seen, limit - created, "strong_attribute_match_different_amount", key[0], "strong", reason)
                if c: linked.add(pair); created += c
    # Pass 2: WEAK label heuristic (unchanged idea): same label + currency, different amount, different documents.
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for n, d in _nodes_of_type(G, "Transaction"):
        if _is_policy_role(d.get("amount_role")): continue
        if d.get("label"): groups[(d["label"], d.get("currency"))].append(n)
    for (label, _cur), txns in groups.items():
        if len(txns) < 2 or len(txns) > 50: continue
        for i in range(len(txns)):
            for j in range(i + 1, len(txns)):
                t1, t2 = txns[i], txns[j]
                n1, n2 = G.nodes[t1], G.nodes[t2]
                if n1.get("amount") == n2.get("amount") or frozenset((t1, t2)) in linked: continue
                strength, _m, _c = _match_transactions(n1, n2)
                if strength != "weak": continue  # strong handled above; distinct reference IDs = different transactions
                if created >= limit: return created
                c = _link_pair(G, t1, t2, seen, limit - created, "same_label_different_amount", label, "weak",
                               f"same normalized label '{label}' + currency, amounts differ; no reference/transaction ID matched (heuristic)")
                if c: linked.add(frozenset((t1, t2))); created += c
    return created + _detect_term_contradictions(G, max(0, limit - created), seen, linked)

def _detect_term_contradictions(G: nx.MultiDiGraph, remaining: int, seen: set, linked: Optional[set] = None) -> int:
    """Second heuristic: Evidence from DIFFERENT documents that share a Term entity, each carrying exactly ONE amount
    (same currency), amounts differ. Evidence with 0 or several amounts is skipped (ambiguous). WEAK: POSSIBLE_DISCREPANCY only."""
    created = 0
    linked = linked if linked is not None else set()
    if remaining <= 0: return 0
    for term_id, td in _nodes_of_type(G, "Entity", "Term"):
        evs = [e["node_id"] for e in _in_edge_sources(G, term_id, "SUPPORTS", "Evidence")]
        if len(evs) < 2 or len(evs) > 50: continue
        info = {}
        for ev in evs:
            txns = [v for _, v, d in G.out_edges(ev, data=True) if d.get("relation") == "SUPPORTS" and G.nodes[v].get("type") == "Transaction"
                    and not _is_policy_role(G.nodes[v].get("amount_role"))]
            docs = {x["document_id"] for x in _evidence_source_docs(G, ev)}
            if len(txns) == 1 and docs: info[ev] = (txns[0], docs)
        items = list(info.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (e1, (t1, d1)), (e2, (t2, d2)) = items[i], items[j]
                if d1 & d2: continue
                n1, n2 = G.nodes[t1], G.nodes[t2]
                if n1.get("currency") != n2.get("currency") or n1.get("amount") == n2.get("amount"): continue
                if frozenset((t1, t2)) in linked: continue
                strength, _m, _c = _match_transactions(n1, n2)
                if strength != "weak": continue
                if created >= remaining: return created
                c = _link_pair(G, t1, t2, seen, remaining - created, "shared_term_different_amount", td.get("value"), "weak",
                               f"evidence rows share the Title-Case term '{td.get('value')}' (co-mention only), same currency, amounts differ; no reference/transaction ID matched (heuristic)")
                if c: linked.add(frozenset((t1, t2))); created += c
    return created

# --- CROSS-DOCUMENT EVIDENCE LINKING (Prompt 5, Phase A) ---
# Connects DOCUMENTS that refer to the same vendor / transaction / compliance chain, using only what the pipeline already extracted:
# Evidence text + provenance, Transaction nodes (amount, currency), Entity nodes (GovID / Phone / Term), labelled vendor / person fields,
# reference IDs, dates and the existing lexical similarity (bm25_scores). Edges are Document --CROSS_DOCUMENT_LINK--> Document, ONE per
# unordered document pair, so the existing Evidence Graph is untouched and V1/V2 retrieval is unchanged (traversal never leaves a Document hub
# except over LINKED_FROM). Every edge carries relationship_type, link_kinds, match_methods, match_score, matched_fields (with source/target
# evidence ids + locations), conflicts, conflict_status and a plain-language link_reason.
#   link_status EXPLICIT    = at least one STRONG anchor (typed reference ID, valid GovID, labelled vendor/person name exact match), no conflict
#               WEAK        = weak anchor (cross-type token, phone, unverified GovID) or corroboration only (amount+date / semantic): NOT identity
#               CONFLICTING = a positive match exists, but fields disagree (different IDs of the same type, vendor, currency, amounts)
#               UNRESOLVED  = a signal exists (e.g. equal amount) but it is insufficient to associate the documents; never used for chains
# Semantic (lexical) similarity is only ever a corroborator: it can never create EXPLICIT, and a semantic-only edge is flagged semantic_only.
# match_score is rule-based (noisy-OR of signal weights), NOT a calibrated probability. No decision, verdict, VIOLATES/SATISFIES edge is created here.
XDOC_RELATION = "CROSS_DOCUMENT_LINK"
XDOC_EXPLICIT, XDOC_WEAK, XDOC_CONFLICT, XDOC_UNRESOLVED = "EXPLICIT", "WEAK", "CONFLICTING", "UNRESOLVED"

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except (TypeError, ValueError): return default

ENABLE_CROSS_DOCUMENT_LINKING = os.getenv("ENABLE_CROSS_DOCUMENT_LINKING", "true").lower() == "true"
MAX_XDOC_EDGES = max(0, _env_int("MAX_XDOC_EDGES", 500))
MAX_XDOC_DOCS = max(2, _env_int("MAX_XDOC_DOCS", 200))
XDOC_MAX_BUCKET = max(2, _env_int("XDOC_MAX_BUCKET", 50))  # a shared value held by more documents than this (e.g. a common date) is a hub: not used to pair documents
XDOC_MAX_SEMANTIC_DOCS = max(2, _env_int("XDOC_MAX_SEMANTIC_DOCS", 60))  # BM25 matrix is O(N^2): skipped (and reported) above this many documents
XDOC_SEMANTIC_THRESHOLD = _env_float("XDOC_SEMANTIC_THRESHOLD", 0.5)  # uncalibrated normalized BM25 overlap in [0,1]
XDOC_TXN_ROLES = ("PURCHASE_ORDER", "INVOICE", "PAYMENT", "APPROVAL")
XDOC_ROLE_RANK = {"PURCHASE_ORDER": 0, "INVOICE": 1, "PAYMENT": 2, "APPROVAL": 3}
XDOC_ROLE_CUES = {"INVOICE": re.compile(r'(?i)\b(invoice|inv|bill)\b'),
                  "PURCHASE_ORDER": re.compile(r'(?i)\b(purchase\s*order|po)\b'),
                  "PAYMENT": re.compile(r'(?i)\b(payment|paid|remittance|utr|receipt|voucher|bank\s+statement|neft|rtgs|upi)\b'),
                  "APPROVAL": re.compile(r'(?i)\b(approval|approved|approver|authori[sz]ation|authori[sz]ed|sign[\s-]*off)\b')}
_XDOC_TOK = r'(?P<marker>(?:\s*(?:(?:no|num(?:ber)?|id|utr|txn|ref(?:erence)?)\b\.?|#))*)\s*[:=#\-]?\s*(?P<tok>[A-Z0-9][A-Z0-9\-/_]{2,31})'  # lead-in words ("Payment UTR: X", "Invoice No. X") are skipped, not mistaken for the ID
_XDOC_REF_LABELLED = {"invoice": re.compile(r'(?i)\b(?:invoice|inv)\b' + _XDOC_TOK),
                      "purchase_order": re.compile(r'(?i)\b(?:purchase\s*order|po)\b' + _XDOC_TOK),
                      "payment": re.compile(r'(?i)\b(?:utr|txn|transaction|payment|receipt|voucher|remittance|cheque)\b' + _XDOC_TOK),
                      "approval": re.compile(r'(?i)\b(?:approval|approved|authori[sz]ation)\b' + _XDOC_TOK),
                      "ref": re.compile(r'(?i)\b(?:ref(?:erence)?)\b' + _XDOC_TOK)}
_XDOC_REF_PREFIXED = re.compile(r'(?<![A-Za-z0-9])(?P<pfx>INV|PO|UTR|TXN|PAY|APPR|REF)[-/_]?\d[A-Z0-9\-/_]{1,28}(?![A-Za-z0-9])')  # case-sensitive on purpose
_XDOC_PREFIX_TYPE = {"INV": "invoice", "PO": "purchase_order", "UTR": "payment", "TXN": "payment", "PAY": "payment", "APPR": "approval", "REF": "ref"}
_XDOC_COMPANY_SUFFIX = re.compile(r'\b(?:pvt|private|ltd|limited|inc|incorporated|llc|llp|corp|corporation|co|company)\b')
_XDOC_WEIGHTS = {"strong_id": 0.9, "strong_govid": 0.9, "strong_party": 0.6, "weak_anchor": 0.35, "amount": 0.15, "date": 0.1, "term": 0.05}

def _xdoc_norm_ref(tok: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', tok.upper())

def _xdoc_norm_party(v: str) -> str:
    s = re.sub(r'[^a-z0-9& ]', ' ', v.lower())
    return re.sub(r'\s+', ' ', _XDOC_COMPANY_SUFFIX.sub(' ', s)).strip()

def _xdoc_extract_refs(text: str) -> List[Tuple[str, str, str]]:
    """(type, normalized id, matched text). Labelled patterns (invoice / PO / payment / approval / generic reference) plus upper-case prefixed IDs (INV-001, PO-77, UTR...).
    A token must contain a digit, must not be a date, and a bare number without marker/letters/colon is skipped (it is probably an amount)."""
    out: List[Tuple[str, str, str]] = []
    for rtype, rx in _XDOC_REF_LABELLED.items():
        for m in rx.finditer(text or ""):
            tok = m.group("tok")
            if not re.search(r'\d', tok) or any(p.fullmatch(tok) for p, _ in _DATE_PATTERNS): continue
            if not m.group("marker") and not re.search(r'[A-Za-z]', tok) and ":" not in m.group(0): continue
            norm = _xdoc_norm_ref(tok)
            if len(norm) >= 3: out.append((rtype, norm, m.group(0)))
    for m in _XDOC_REF_PREFIXED.finditer(text or ""):
        norm = _xdoc_norm_ref(m.group(0))
        if len(norm) >= 3: out.append((_XDOC_PREFIX_TYPE[m.group("pfx")], norm, m.group(0)))
    return out

def _xdoc_extract_parties(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """Labelled vendor / person fields (same label vocabulary as _extract_txn_attributes) -> (normalized name, raw value)."""
    out: Dict[str, List[Tuple[str, str]]] = {"vendor": [], "person": []}
    for key in ("vendor", "person"):
        for m in re.finditer(rf'(?i)\b{_ATTR_LABELS[key]}\s*[:=]\s*([^|¦;\n]+?)(?=\s*(?:[|¦;\n]|$|\b(?:{_ALL_LABELS})\s*[:=]))', text or ""):
            raw = re.sub(r'\s+', ' ', m.group(1)).strip()[:60]
            if not raw or MONEY_PATTERN.fullmatch(raw): continue
            norm = _xdoc_norm_party(raw)
            if len(norm) >= 3 and not norm.isdigit(): out[key].append((norm, raw))
    return out

def _xdoc_infer_role(G: nx.MultiDiGraph, doc_id: str, fname: str, texts: List[str]) -> Tuple[str, str, Dict[str, int]]:
    """Document role from filename / wording cues. A HINT used only to name the relationship; linking never depends on it. Ambiguous or cue-less => UNKNOWN."""
    d = G.nodes[doc_id]
    if d.get("role") == "RULEBOOK" or d.get("policy_context"): return "POLICY", "document flagged as policy/rulebook context", {}
    name = re.sub(r'[_\-.]+', ' ', fname or "")
    body = "\n".join(texts)[:20000]
    scores = {r: 3 * len(rx.findall(name)) + min(5, len(rx.findall(body))) for r, rx in XDOC_ROLE_CUES.items()}
    scores = {r: s for r, s in scores.items() if s}
    name_roles = [r for r, rx in XDOC_ROLE_CUES.items() if rx.search(name)]
    if len(name_roles) == 1: return name_roles[0], "filename cue", scores
    if not scores: return "UNKNOWN", "no role cue found", scores
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and ranked[1][1] * 2 > ranked[0][1]: return "UNKNOWN", "ambiguous role cues", scores
    return ranked[0][0], "wording cues", scores

def _xdoc_prov(G: nx.MultiDiGraph, ev_id: str, doc_id: str, matched: Any) -> Dict[str, Any]:
    return {"document_id": doc_id, "filename": G.nodes[doc_id].get("filename"), "evidence_id": ev_id, "location": G.nodes[ev_id].get("source_location"), "matched_text": str(matched)[:80]}

def _xdoc_build_profiles(G: nx.MultiDiGraph) -> Dict[str, Dict[str, Any]]:
    """Per-document reference profile built from investigation Evidence (policy / context-only evidence excluded). Every value keeps its provenance list."""
    ev_by_doc: Dict[str, List[str]] = defaultdict(list)
    for ev_id, d in _investigation_evidence(G):
        if d.get("context_only"): continue
        for s in _evidence_source_docs(G, ev_id): ev_by_doc[s["document_id"]].append(ev_id)
    profiles: Dict[str, Dict[str, Any]] = {}
    for doc_id, dd in sorted(_nodes_of_type(G, "Document"), key=lambda kv: (str(kv[1].get("filename")), kv[0])):
        evs = sorted(ev_by_doc.get(doc_id, []))
        texts = [G.nodes[e].get("text") or "" for e in evs]
        role, basis, scores = _xdoc_infer_role(G, doc_id, dd.get("filename") or "", texts)
        G.nodes[doc_id].update(xdoc_role=role, xdoc_role_basis=basis, xdoc_role_scores=scores)
        if role == "POLICY" or not evs: continue
        p: Dict[str, Any] = {"doc_id": doc_id, "filename": dd.get("filename"), "role": role, "evidence_ids": evs, "text": "\n".join(texts)[:20000],
                             "ids": defaultdict(list), "dates": defaultdict(list), "amounts": defaultdict(list), "vendor": defaultdict(list), "person": defaultdict(list),
                             "entities": defaultdict(list), "entity_meta": {}}
        for ev, text in zip(evs, texts):
            for rtype, norm, raw in _xdoc_extract_refs(text): p["ids"][(rtype, norm)].append(_xdoc_prov(G, ev, doc_id, raw))
            for pat, fmt in _DATE_PATTERNS:
                for m in pat.finditer(text):
                    try: p["dates"][fmt(m)].append(_xdoc_prov(G, ev, doc_id, m.group(0)))
                    except (KeyError, ValueError): pass
            for key, vals in _xdoc_extract_parties(text).items():
                for norm, raw in vals: p[key][norm].append(_xdoc_prov(G, ev, doc_id, raw))
            for _, tgt, ed in G.out_edges(ev, data=True):
                if ed.get("relation") != "SUPPORTS": continue
                td = G.nodes[tgt]
                if td.get("type") == "Transaction" and not _is_policy_role(td.get("amount_role")) and td.get("amount") is not None:
                    p["amounts"][(td.get("currency"), td.get("amount"))].append(_xdoc_prov(G, ev, doc_id, f"{td.get('currency')} {td.get('amount')}"))
                elif td.get("type") == "Entity" and td.get("entity_type") in ("GovID", "Phone", "Term"):
                    p["entities"][tgt].append(_xdoc_prov(G, ev, doc_id, "[REDACTED]" if td.get("entity_type") != "Term" else td.get("value")))
                    p["entity_meta"][tgt] = (td.get("entity_type"), td.get("is_valid"), td.get("value") if td.get("entity_type") == "Term" else f"{td.get('entity_type')}:{td.get('value_hash') or tgt}")
        profiles[doc_id] = p
    return profiles

def _xdoc_semantic_matrix(profiles: Dict[str, Dict[str, Any]]) -> Optional[Dict[Tuple[str, str], float]]:
    """Symmetric normalized BM25 overlap (existing bm25_scores): mean of score(A->B)/score(A->A) and score(B->A)/score(B->B), clamped to [0,1]. Lexical, NOT identity."""
    ids = list(profiles)
    if len(ids) < 2 or len(ids) > XDOC_MAX_SEMANTIC_DOCS: return None
    texts = [profiles[i]["text"] for i in ids]
    rows = [bm25_scores(profiles[i]["text"], texts) for i in ids]
    sim: Dict[Tuple[str, str], float] = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            saa, sbb = rows[a][a], rows[b][b]
            ab = min(1.0, rows[a][b] / saa) if saa > 0 else 0.0
            ba = min(1.0, rows[b][a] / sbb) if sbb > 0 else 0.0
            sim[(ids[a], ids[b])] = round((ab + ba) / 2, 4)
    return sim

def _xdoc_complementary(ra: str, rb: str) -> bool:
    return ra in XDOC_TXN_ROLES and rb in XDOC_TXN_ROLES and ra != rb

def _xdoc_evaluate_pair(G: nx.MultiDiGraph, pa: Dict[str, Any], pb: Dict[str, Any], sim: Optional[float]) -> Optional[Dict[str, Any]]:
    """Signals -> status. Returns None when there is nothing worth recording (no fabricated relationship)."""
    sig: List[Dict[str, Any]] = []
    def add(field, kind, strength, method, value, provs_a, provs_b, weight, link_kind=None):
        sig.append({"field": field, "kind": kind, "strength": strength, "method": method, "value": value, "weight": weight, "link_kind": link_kind,
                    "source": provs_a[:5], "target": provs_b[:5]})
    strong_tokens = set()
    for key in sorted(set(pa["ids"]) & set(pb["ids"])):
        rtype, norm = key
        strong = rtype != "ref" or len(norm) >= 5
        add(f"{rtype}_id", "anchor", "strong" if strong else "weak", "identifier_exact_match", norm, pa["ids"][key], pb["ids"][key], _XDOC_WEIGHTS["strong_id" if strong else "weak_anchor"], "SAME_TRANSACTION_REFERENCE")
        if strong: strong_tokens.add(norm)
    for (t1, n1) in sorted(pa["ids"]):
        for (t2, n2) in sorted(pb["ids"]):
            if n1 == n2 and t1 != t2 and "ref" in (t1, t2) and len(n1) >= 5 and n1 not in strong_tokens:
                add("reference_token", "anchor", "weak", "identifier_token_cross_type", n1, pa["ids"][(t1, n1)], pb["ids"][(t2, n2)], _XDOC_WEIGHTS["weak_anchor"], "SAME_TRANSACTION_REFERENCE")
    terms: List[str] = []
    for ent in sorted(set(pa["entities"]) & set(pb["entities"])):
        etype, valid, label = pa["entity_meta"][ent]
        if etype == "GovID":
            strong = valid is True
            add("govid", "anchor", "strong" if strong else "weak", "shared_govid_entity" if strong else "shared_govid_entity_unverified", "[REDACTED_GOVID]", pa["entities"][ent], pb["entities"][ent],
                _XDOC_WEIGHTS["strong_govid" if strong else "weak_anchor"], "SAME_ENTITY_IDENTIFIER")
        elif etype == "Phone":
            add("phone", "anchor", "weak", "shared_phone_entity", "[REDACTED_PHONE]", pa["entities"][ent], pb["entities"][ent], _XDOC_WEIGHTS["weak_anchor"], "SAME_ENTITY_IDENTIFIER")
        else:
            terms.append(ent)
    for ent in terms[:5]:
        add("shared_term", "corroborator", "weak", "title_case_term_co_mention", pa["entity_meta"][ent][2], pa["entities"][ent], pb["entities"][ent], _XDOC_WEIGHTS["term"])
    for fld, kind in (("vendor", "SAME_VENDOR"), ("person", "SAME_PERSON")):
        for norm in sorted(set(pa[fld]) & set(pb[fld])):
            add(fld, "anchor", "strong", "structured_field_exact_match", norm, pa[fld][norm], pb[fld][norm], _XDOC_WEIGHTS["strong_party"], kind)
    for amt in sorted(set(pa["amounts"]) & set(pb["amounts"]), key=str):
        add("amount", "corroborator", "weak", "amount_currency_exact_match", f"{amt[0]} {amt[1]}", pa["amounts"][amt], pb["amounts"][amt], _XDOC_WEIGHTS["amount"])
    for dt in sorted(set(pa["dates"]) & set(pb["dates"])):
        add("date", "corroborator", "weak", "date_exact_match", dt, pa["dates"][dt], pb["dates"][dt], _XDOC_WEIGHTS["date"])
    if sim is not None and sim >= XDOC_SEMANTIC_THRESHOLD:
        sig.append({"field": "semantic_similarity", "kind": "corroborator", "strength": "weak", "method": "bm25_normalized_lexical_overlap", "value": sim, "weight": round(0.2 * sim, 4),
                    "link_kind": None, "source": [], "target": []})
    if not sig: return None
    strong_a = [s for s in sig if s["kind"] == "anchor" and s["strength"] == "strong"]
    weak_a = [s for s in sig if s["kind"] == "anchor" and s["strength"] == "weak"]
    corr = {s["field"] for s in sig if s["kind"] == "corroborator"}
    comp = _xdoc_complementary(pa["role"], pb["role"])
    semantic_only = False
    if strong_a: positive = XDOC_EXPLICIT
    elif weak_a: positive = XDOC_WEAK
    elif {"amount", "date"} <= corr or ("semantic_similarity" in corr and len(corr) > 1): positive = XDOC_WEAK
    elif corr == {"semantic_similarity"} and comp: positive, semantic_only = XDOC_WEAK, True
    elif "amount" in corr and comp: positive = XDOC_UNRESOLVED
    else: return None
    conflicts: List[Dict[str, Any]] = []
    if positive != XDOC_UNRESOLVED:
        for rtype in sorted({t for t, _ in pa["ids"]} & {t for t, _ in pb["ids"]}):
            if rtype == "ref": continue
            ia, ib = {n for t, n in pa["ids"] if t == rtype}, {n for t, n in pb["ids"] if t == rtype}
            if ia and ib and not (ia & ib): conflicts.append({"field": f"{rtype}_id", "kind": "different_identifiers", "source_values": sorted(ia)[:5], "target_values": sorted(ib)[:5]})
        for fld in ("vendor", "person"):
            if pa[fld] and pb[fld] and not (set(pa[fld]) & set(pb[fld])): conflicts.append({"field": fld, "kind": "different_names", "source_values": sorted(pa[fld])[:5], "target_values": sorted(pb[fld])[:5]})
        ca, cb = {c for c, _ in pa["amounts"]}, {c for c, _ in pb["amounts"]}
        if ca and cb and not (ca & cb): conflicts.append({"field": "currency", "kind": "different_currencies", "source_values": sorted(map(str, ca)), "target_values": sorted(map(str, cb))})
        for cur in sorted(ca & cb, key=str):
            va, vb = {a for c, a in pa["amounts"] if c == cur}, {a for c, a in pb["amounts"] if c == cur}
            if va and vb and not (va & vb): conflicts.append({"field": "amount", "kind": "different_amounts_same_currency", "currency": cur, "source_values": sorted(va)[:5], "target_values": sorted(vb)[:5],
                                                               "note": "may be a legitimate partial payment / line-item difference; requires reconciliation"})
    status = XDOC_CONFLICT if conflicts else positive
    score = 1.0
    for s in sig: score *= (1.0 - min(0.99, s["weight"]))
    return {"signals": sig, "positive_status": positive, "status": status, "conflicts": conflicts, "semantic_only": semantic_only, "score": round(1.0 - score, 4), "complementary_roles": comp}

def _xdoc_pair_order(pa: Dict[str, Any], pb: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ka = (XDOC_ROLE_RANK.get(pa["role"], 9), str(pa["filename"]), pa["doc_id"])
    kb = (XDOC_ROLE_RANK.get(pb["role"], 9), str(pb["filename"]), pb["doc_id"])
    return (pa, pb) if ka <= kb else (pb, pa)

def _xdoc_relationship(sr: Dict[str, Any], tr: Dict[str, Any], kinds: List[str], status: str) -> str:
    if status == XDOC_UNRESOLVED: return "UNRESOLVED_ASSOCIATION"
    if "SAME_TRANSACTION_REFERENCE" in kinds:
        if sr in XDOC_TXN_ROLES and tr in XDOC_TXN_ROLES: return f"{sr}_TO_{tr}" if sr != tr else f"SAME_ROLE_{sr}"
        return "RELATED_TRANSACTION_DOCUMENTS"
    for k in ("SAME_VENDOR", "SAME_PERSON", "SAME_ENTITY_IDENTIFIER"):
        if k in kinds: return k
    return "POSSIBLY_RELATED_DOCUMENTS"

def _xdoc_reason(signals: List[Dict[str, Any]], conflicts: List[Dict[str, Any]], status: str, semantic_only: bool) -> str:
    parts = [f"{s['field']} '{s['value']}' via {s['method']} ({s['strength']} {s['kind']})" for s in signals if s["field"] != "semantic_similarity"]
    sem = [s for s in signals if s["field"] == "semantic_similarity"]
    if sem: parts.append(f"lexical similarity {sem[0]['value']} (corroboration only; not identity)")
    txt = "; ".join(parts[:8]) + (f"; +{len(parts) - 8} more" if len(parts) > 8 else "")
    if conflicts: txt += " | CONFLICTS: " + "; ".join(f"{c['field']} ({c['kind']})" for c in conflicts)
    if semantic_only: txt += " | semantic similarity alone does not show the documents refer to the same entity or transaction"
    if status == XDOC_UNRESOLVED: txt += " | insufficient to associate the documents; no identifier or party anchor matched"
    return txt

def _xdoc_unresolved_references(profiles: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Typed references (e.g. a PO number inside an invoice) whose target document is not among the supplied documents. Own-role IDs and generic 'ref' are skipped."""
    tokens_by_doc = {d: {n for _, n in p["ids"]} for d, p in profiles.items()}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for did, p in profiles.items():
        if p["role"] not in XDOC_TXN_ROLES: continue
        own = p["role"].lower()
        for (rtype, norm), provs in sorted(p["ids"].items()):
            if rtype in ("ref", own) or any(norm in toks for d, toks in tokens_by_doc.items() if d != did): continue
            out.setdefault(did, []).append({"reference_type": rtype, "reference": norm, "reason": "referenced document/identifier is not present in any other supplied document", "provenance": provs[:3]})
    return out

def link_cross_documents(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Builds Document --CROSS_DOCUMENT_LINK--> Document edges (idempotent: previous cross-document edges/annotations are replaced). Phase A only: no decision,
    no verdict, no VIOLATES/SATISFIES, no policy link. Returns a summary (also kept in G.graph['cross_document_summary'])."""
    summary: Dict[str, Any] = {"status": "OK", "phase": "A", "edges_created": 0, "by_status": {}, "chains": 0, "documents_profiled": 0, "unresolved_references": 0, "truncated": False, "notes": []}
    G.remove_edges_from([(u, v, k) for u, v, k, d in list(G.edges(keys=True, data=True)) if d.get("relation") == XDOC_RELATION])
    for n, _ in _nodes_of_type(G, "Document"):
        for key in ("xdoc_chain_id", "xdoc_unresolved_references", "xdoc_identifiers", "xdoc_vendors", "xdoc_persons"): G.nodes[n].pop(key, None)
    if not ENABLE_CROSS_DOCUMENT_LINKING:
        summary["status"] = "DISABLED"; G.graph["cross_document_summary"] = summary; return summary
    profiles = _xdoc_build_profiles(G)
    summary["documents_profiled"] = len(profiles)
    if len(profiles) > MAX_XDOC_DOCS:
        summary["status"] = "SKIPPED"; summary["notes"].append(f"{len(profiles)} documents exceed MAX_XDOC_DOCS={MAX_XDOC_DOCS}; no cross-document links created")
        G.graph["cross_document_summary"] = summary; return summary
    for did, p in profiles.items():
        G.nodes[did]["xdoc_identifiers"] = [{"type": t, "value": n} for t, n in sorted(p["ids"])][:100]
        G.nodes[did]["xdoc_vendors"] = sorted(p["vendor"])[:20]; G.nodes[did]["xdoc_persons"] = sorted(p["person"])[:20]
    unresolved = _xdoc_unresolved_references(profiles)
    for did, items in unresolved.items(): G.nodes[did]["xdoc_unresolved_references"] = items[:50]
    summary["unresolved_references"] = sum(len(v) for v in unresolved.values())
    index: Dict[Tuple[str, Any], set] = defaultdict(set)
    for did, p in profiles.items():
        for _, n in p["ids"]: index[("token", n)].add(did)
        for e in p["entities"]: index[("entity", e)].add(did)
        for f in ("vendor", "person", "dates", "amounts"):
            for v in p[f]: index[(f, v)].add(did)
    pairs: set = set()
    hubs = 0
    for (_k, _v), docs in index.items():
        if len(docs) < 2: continue
        if len(docs) > XDOC_MAX_BUCKET: hubs += 1; continue
        ds = sorted(docs)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)): pairs.add((ds[i], ds[j]))
    if hubs: summary["notes"].append(f"{hubs} shared value(s) held by more than {XDOC_MAX_BUCKET} documents were not used to pair documents (hub values)")
    sims = _xdoc_semantic_matrix(profiles)
    if sims is None:
        summary["semantic"] = {"status": "SKIPPED", "reason": f"needs 2..{XDOC_MAX_SEMANTIC_DOCS} documents, got {len(profiles)}"}
    else:
        summary["semantic"] = {"status": "COMPUTED", "method": "bm25_normalized_lexical_overlap", "threshold": XDOC_SEMANTIC_THRESHOLD, "basis": "uncalibrated lexical overlap; never proof of identity"}
        for (a, b), s in sims.items():
            if s >= XDOC_SEMANTIC_THRESHOLD and _xdoc_complementary(profiles[a]["role"], profiles[b]["role"]): pairs.add((a, b))
    results = []
    for a, b in sorted(pairs):
        sim = None
        if sims is not None: sim = sims.get((a, b), sims.get((b, a)))
        ev = _xdoc_evaluate_pair(G, profiles[a], profiles[b], sim)
        if ev: results.append((a, b, ev))
    prio = {XDOC_EXPLICIT: 0, XDOC_CONFLICT: 1, XDOC_WEAK: 2, XDOC_UNRESOLVED: 3}
    results.sort(key=lambda r: (prio[r[2]["status"]], -r[2]["score"], r[0], r[1]))
    if len(results) > MAX_XDOC_EDGES:
        summary["truncated"] = True; summary["notes"].append(f"{len(results)} candidate links exceed MAX_XDOC_EDGES={MAX_XDOC_EDGES}; weakest dropped")
        results = results[:MAX_XDOC_EDGES]
    by_status: Counter = Counter()
    for a, b, ev in results:
        s, t = _xdoc_pair_order(profiles[a], profiles[b])
        kinds = sorted({x["link_kind"] for x in ev["signals"] if x["link_kind"]})
        strong_kinds = sorted({x["link_kind"] for x in ev["signals"] if x["link_kind"] and x["kind"] == "anchor" and x["strength"] == "strong"})  # a weak cross-type token must not make a link transaction-level
        status = ev["status"]
        sigs = ev["signals"]
        ev_ids = sorted(set(s["evidence_ids"]) | set(t["evidence_ids"]))
        used = sorted({p_["evidence_id"] for x in sigs for p_ in x["source"] + x["target"]})
        sem = next((x["value"] for x in sigs if x["field"] == "semantic_similarity"), None)
        G.add_edge(s["doc_id"], t["doc_id"], relation=XDOC_RELATION, method="cross_document_linking", phase="A",
                   link_status=status, positive_status=ev["positive_status"], conflict_status=("CONFLICTING" if ev["conflicts"] else "NONE"), conflicts=ev["conflicts"],
                   match_strength=("strong" if any(x["kind"] == "anchor" and x["strength"] == "strong" for x in sigs) else ("none" if status == XDOC_UNRESOLVED else "weak")),
                   match_score=ev["score"], score_basis="rule_based_noisy_or_of_signal_weights; not a calibrated probability",
                   relationship_type=_xdoc_relationship(s["role"], t["role"], strong_kinds or kinds, status), link_kinds=kinds, strong_link_kinds=strong_kinds,
                   match_methods=sorted({x["method"] for x in sigs}), matched_fields=sigs,
                   semantic_only=ev["semantic_only"], semantic_similarity=sem, semantic_proof_of_identity=False,
                   source_document_id=s["doc_id"], target_document_id=t["doc_id"], source_filename=s["filename"], target_filename=t["filename"],
                   source_role=s["role"], target_role=t["role"], evidence_ids=used or ev_ids,
                   source_refs=[{"evidence_id": e, "documents": _evidence_source_docs(G, e)} for e in used[:20]],
                   link_reason=_xdoc_reason(sigs, ev["conflicts"], status, ev["semantic_only"]),
                   uncertainty_reason=("" if status == XDOC_EXPLICIT else {XDOC_WEAK: "no strong identifier / party anchor; relationship is a possibility, not established",
                                                                         XDOC_CONFLICT: "a match exists but fields disagree; requires reconciliation against source records",
                                                                         XDOC_UNRESOLVED: "signal insufficient to associate the documents"}[status]),
                   heuristic=status != XDOC_EXPLICIT, confirmed=False, can_create_violation=False, decision_made=False)
        by_status[status] += 1
    summary["edges_created"] = len(results); summary["by_status"] = dict(by_status)
    summary["chains"] = len(_assign_xdoc_chains(G))
    G.graph["cross_document_summary"] = summary
    return summary

def _xdoc_edges(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    return [dict(d, source_id=u, target_id=v) for u, v, d in G.edges(data=True) if d.get("relation") == XDOC_RELATION]

def _assign_xdoc_chains(G: nx.MultiDiGraph) -> Dict[str, List[str]]:
    """Compliance chains = connected components of EXPLICIT, transaction-level links (SAME_TRANSACTION_REFERENCE). Vendor/person/identifier-only links, weak,
    conflicting and unresolved links never merge documents into a chain (one vendor has many transactions). Sets Document.xdoc_chain_id (components of >= 2 documents)."""
    parent: Dict[str, str] = {}
    def find(x):
        while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in _xdoc_edges(G):
        if e.get("link_status") == XDOC_EXPLICIT and "SAME_TRANSACTION_REFERENCE" in _xdoc_strong_kinds(e): parent[find(e["source_id"])] = find(e["target_id"])
    groups: Dict[str, List[str]] = defaultdict(list)
    for n in list(parent): groups[find(n)].append(n)
    chains: Dict[str, List[str]] = {}
    for members in groups.values():
        if len(members) < 2: continue
        key = "|".join(sorted(f"{G.nodes[m].get('file_hash')}:{G.nodes[m].get('filename')}" for m in members))
        cid = "xchain_" + hashlib.sha256(key.encode("utf-8", "ignore")).hexdigest()[:12]
        chains[cid] = sorted(members)
        for m in members: G.nodes[m]["xdoc_chain_id"] = cid
    return chains

# --- CROSS-DOCUMENT QUERIES (read-only; answer 'which documents belong together' without deciding compliance) ---
def query_cross_document_links(G: nx.MultiDiGraph, doc_id: Optional[str] = None, statuses: Optional[List[str]] = None, link_kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cross-document links, optionally for one document / status set / link kind (SAME_VENDOR, SAME_TRANSACTION_REFERENCE, SAME_PERSON, SAME_ENTITY_IDENTIFIER)."""
    out = []
    for e in _xdoc_edges(G):
        if doc_id and doc_id not in (e["source_id"], e["target_id"]): continue
        if statuses and e.get("link_status") not in statuses: continue
        if link_kind and link_kind not in (e.get("link_kinds") or []): continue
        out.append(e)
    return out

def query_documents_for_vendor(G: nx.MultiDiGraph, vendor: str, explicit_only: bool = True) -> List[Dict[str, Any]]:
    """Documents carrying the labelled vendor name (normalized: case, punctuation and company suffixes ignored). Exact normalized match only."""
    norm = _xdoc_norm_party(vendor or "")
    if not norm: return []
    return [{"document_id": n, "filename": d.get("filename"), "role": d.get("xdoc_role")} for n, d in _nodes_of_type(G, "Document") if norm in (d.get("xdoc_vendors") or [])]

def query_related_documents(G: nx.MultiDiGraph, doc_id: str, include_weak: bool = False) -> List[Dict[str, Any]]:
    """Documents linked to doc_id. EXPLICIT only by default; include_weak adds WEAK / CONFLICTING / UNRESOLVED (each keeps its status and reason)."""
    sts = None if include_weak else [XDOC_EXPLICIT]
    return [{"document_id": e["target_id"] if e["source_id"] == doc_id else e["source_id"], "link_status": e["link_status"], "relationship_type": e["relationship_type"],
             "link_kinds": e["link_kinds"], "match_score": e["match_score"], "conflict_status": e["conflict_status"], "link_reason": e["link_reason"]}
            for e in query_cross_document_links(G, doc_id, sts)]

def query_compliance_chains(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """Documents grouped into transaction chains (see _assign_xdoc_chains), each with its explicit links, plus weak / conflicting / unresolved links touching it and
    unresolved references. roles_present describes what was FOUND; it does not say what SHOULD exist (no compliance judgement)."""
    by_chain: Dict[str, List[str]] = defaultdict(list)
    for n, d in _nodes_of_type(G, "Document"):
        if d.get("xdoc_chain_id"): by_chain[d["xdoc_chain_id"]].append(n)
    edges = _xdoc_edges(G)
    out = []
    for cid, members in sorted(by_chain.items()):
        ms = set(members)
        touching = [e for e in edges if e["source_id"] in ms or e["target_id"] in ms]
        brief = lambda e: {"source": e["source_id"], "target": e["target_id"], "relationship_type": e["relationship_type"], "link_status": e["link_status"], "match_score": e["match_score"], "link_reason": e["link_reason"]}
        out.append({"chain_id": cid, "documents": [{"document_id": m, "filename": G.nodes[m].get("filename"), "role": G.nodes[m].get("xdoc_role")} for m in sorted(members)],
                    "roles_present": sorted({G.nodes[m].get("xdoc_role") for m in members}),
                    "explicit_links": [brief(e) for e in touching if e["link_status"] == XDOC_EXPLICIT and e["source_id"] in ms and e["target_id"] in ms],
                    "conflicting_links": [brief(e) for e in touching if e["link_status"] == XDOC_CONFLICT],
                    "weak_links": [brief(e) for e in touching if e["link_status"] == XDOC_WEAK],
                    "unresolved_links": [brief(e) for e in touching if e["link_status"] == XDOC_UNRESOLVED],
                    "unresolved_references": {m: G.nodes[m].get("xdoc_unresolved_references") for m in sorted(members) if G.nodes[m].get("xdoc_unresolved_references")}})
    return out

def query_chain_for_document(G: nx.MultiDiGraph, doc_id: str) -> Optional[Dict[str, Any]]:
    cid = G.nodes[doc_id].get("xdoc_chain_id") if G.has_node(doc_id) else None
    return next((c for c in query_compliance_chains(G) if c["chain_id"] == cid), None) if cid else None

# --- CROSS-DOCUMENT REASONING (Prompt 5, Phase B): retrieval through CROSS_DOCUMENT_LINK edges (V2 path only) ---
# NOT a second reasoning system and NOT a decision engine. When V2 has retrieved evidence from a document, this READS the Phase A CROSS_DOCUMENT_LINK edges around
# that document and adds the relevant evidence of related documents to the existing V2 context, each item labelled with how and why it was reached:
#   [XDOC-EXPLICIT]   EXPLICIT link (strong identifier / party anchor, no conflict, not semantic-only). Followed up to V2_XDOC_MAX_HOPS hops. Entity-level links
#                     (same vendor / person / ID value) are followed one hop from a directly retrieved document only, and only for related documents that are
#                     lexically relevant to the objective, because one vendor has many unrelated transactions.
#   [XDOC-POSSIBLE]   WEAK link: surfaced as a possible relationship, never traversed, never identity (semantic-only links land here, flagged semantic_only).
#   [XDOC-CONFLICT]   CONFLICTING link: surfaced with its conflicts as a reconciliation requirement, never traversed, never resolved.
#   [XDOC-UNRESOLVED] UNRESOLVED link: listed, no relationship established, no evidence text added.
#   [MISSING-REFERENCE] xdoc_unresolved_references of documents in the chain: a referenced document that was not supplied. Nothing is invented for it.
# Policy / rulebook documents are never traversed. The graph is never modified: no VIOLATES / SATISFIES / Decision / CONTRADICTS edge is created and no verdict changes;
# the deterministic compiled-policy / rule-engine path stays authoritative. Budgets (hops, related documents, evidence items, links inspected) are reported, never hidden.
ENABLE_CROSS_DOCUMENT_REASONING = os.getenv("ENABLE_CROSS_DOCUMENT_REASONING", "true").lower() == "true"
V2_XDOC_MAX_HOPS = max(1, _env_int("V2_XDOC_MAX_HOPS", 3))  # invoice -> PO -> payment -> approval is 3 hops from the invoice
V2_XDOC_DOC_LIMIT = max(0, _env_int("V2_XDOC_DOC_LIMIT", 4))  # related documents accepted for traversal per investigation
V2_XDOC_EVIDENCE_LIMIT = max(0, _env_int("V2_XDOC_EVIDENCE_LIMIT", 6))  # evidence items added through EXPLICIT links per investigation
V2_XDOC_EVIDENCE_PER_DOC = max(1, _env_int("V2_XDOC_EVIDENCE_PER_DOC", 2))
V2_XDOC_SIGNAL_LIMIT = max(0, _env_int("V2_XDOC_SIGNAL_LIMIT", 6))  # WEAK / CONFLICTING / UNRESOLVED links surfaced per investigation
V2_XDOC_LINK_BUDGET = max(10, _env_int("V2_XDOC_LINK_BUDGET", 500))  # links inspected per investigation
V2_XDOC_MISSING_LIMIT = 20
_XDOC_STATUS_PRIO = {XDOC_EXPLICIT: 0, XDOC_CONFLICT: 1, XDOC_WEAK: 2, XDOC_UNRESOLVED: 3}
XDOC_PHASE_B_LIMITS = ("Cross-document retrieval associates DOCUMENTS through deterministic links. It covers only the links, hops and budgets reported; it does not show that every related "
                       "document was found, and an EXPLICIT link does not by itself establish compliance, a violation, approval validity or that amounts reconcile.")

def _xdoc_strong_kinds(e: Dict[str, Any]) -> set:
    """Link kinds backed by a STRONG anchor only (a weak cross-type token never makes a link transaction-level)."""
    return {s.get("link_kind") for s in (e.get("matched_fields") or []) if s.get("kind") == "anchor" and s.get("strength") == "strong" and s.get("link_kind")}

def _xdoc_edge_scope(e: Dict[str, Any]) -> Optional[str]:
    sk = _xdoc_strong_kinds(e)
    return "TRANSACTION" if "SAME_TRANSACTION_REFERENCE" in sk else ("ENTITY" if sk else None)

def _xdoc_doc_ok(G: nx.MultiDiGraph, doc_id: str) -> bool:
    """A transaction document: exists, is not a policy / rulebook document."""
    if not G.has_node(doc_id): return False
    d = G.nodes[doc_id]
    return d.get("type") == "Document" and d.get("role") != "RULEBOOK" and not d.get("policy_context") and d.get("xdoc_role") != "POLICY"

def _xdoc_eligibility(G: nx.MultiDiGraph, e: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """May this link be TRAVERSED as an established relationship? -> (ok, reason when not, scope TRANSACTION|ENTITY)."""
    if e.get("link_status") != XDOC_EXPLICIT: return False, f"link status {e.get('link_status')} is not an established relationship", None
    if e.get("conflict_status") != "NONE" or e.get("conflicts"): return False, "link carries conflicts", None
    if e.get("semantic_only"): return False, "semantic similarity alone is not proof of identity", None
    scope = _xdoc_edge_scope(e)
    if scope is None: return False, "no strong anchor recorded on the link", None
    if not (_xdoc_doc_ok(G, e["source_id"]) and _xdoc_doc_ok(G, e["target_id"])): return False, "policy / rulebook documents are not traversed as transaction documents", None
    return True, "", scope

def _xdoc_edge_doc_evidence(e: Dict[str, Any], doc_id: str) -> List[str]:
    """Evidence ids on `doc_id`'s side of the link that carry the matched values (the evidence the relationship was built from)."""
    out: List[str] = []
    for s in e.get("matched_fields") or []:
        for p in (s.get("source") or []) + (s.get("target") or []):
            if p.get("document_id") == doc_id and p.get("evidence_id") and p["evidence_id"] not in out: out.append(p["evidence_id"])
    return out

def _xdoc_link_brief(e: Dict[str, Any]) -> Dict[str, Any]:
    return {"source_document_id": e["source_id"], "target_document_id": e["target_id"], "source_filename": e.get("source_filename"), "target_filename": e.get("target_filename"),
            "relationship_type": e.get("relationship_type"), "link_status": e.get("link_status"), "link_kinds": e.get("link_kinds"), "match_methods": e.get("match_methods"),
            "match_score": e.get("match_score"), "link_reason": e.get("link_reason")}

def traverse_cross_document_context(G: nx.MultiDiGraph, direct_evidence_ids: List[str], objective: str = "", max_hops: Optional[int] = None, doc_limit: Optional[int] = None,
                                    evidence_limit: Optional[int] = None, evidence_per_doc: Optional[int] = None, signal_limit: Optional[int] = None, link_budget: Optional[int] = None) -> Dict[str, Any]:
    """READ-ONLY. direct_evidence_ids = evidence V2 already retrieved (lexical hits + graph-expanded). Returns the cross-document evidence to add to the context plus
    instrumentation (see 'metrics'). Never modifies G, never creates a decision / VIOLATES / SATISFIES, never invents a document or evidence item."""
    max_hops = V2_XDOC_MAX_HOPS if max_hops is None else max(1, int(max_hops))
    doc_limit = V2_XDOC_DOC_LIMIT if doc_limit is None else max(0, int(doc_limit))
    evidence_limit = V2_XDOC_EVIDENCE_LIMIT if evidence_limit is None else max(0, int(evidence_limit))
    evidence_per_doc = V2_XDOC_EVIDENCE_PER_DOC if evidence_per_doc is None else max(1, int(evidence_per_doc))
    signal_limit = V2_XDOC_SIGNAL_LIMIT if signal_limit is None else max(0, int(signal_limit))
    link_budget = V2_XDOC_LINK_BUDGET if link_budget is None else max(1, int(link_budget))
    m: Dict[str, Any] = {"documents_direct": 0, "documents_via_cross_document_links": 0, "links_inspected": 0, "links_accepted_for_traversal": 0, "links_rejected": 0,
                         "links_between_retrieved_documents": 0, "weak_links_surfaced": 0, "conflicting_links_surfaced": 0, "unresolved_links_surfaced": 0,
                         "unresolved_references_surfaced": 0, "evidence_added_via_explicit_links": 0, "evidence_added_via_signal_links": 0, "evidence_already_retrieved": 0,
                         "doc_limit_reached": False, "evidence_limit_reached": False, "signal_limit_reached": False, "link_budget_exhausted": False,
                         "frontier_not_expanded": 0, "max_hops": max_hops}
    res: Dict[str, Any] = {"status": "OK", "enabled": True, "direct_documents": [], "related_documents": [], "explicit_items": [], "weak_signals": [], "conflict_signals": [],
                           "unresolved_signals": [], "missing_references": [], "rejected_links": [], "links_between_retrieved_documents": [], "metrics": m, "limitations": XDOC_PHASE_B_LIMITS}
    direct_ev = [e for e in dict.fromkeys(direct_evidence_ids or []) if G.has_node(e) and G.nodes[e].get("type") == "Evidence"
                 and not G.nodes[e].get("context_only") and not _is_policy_source_evidence(G.nodes[e])]
    direct_set = set(direct_ev)
    roots: Dict[str, List[str]] = {}
    for ev in direct_ev:
        for s in _evidence_source_docs(G, ev):
            if _xdoc_doc_ok(G, s["document_id"]): roots.setdefault(s["document_id"], []).append(ev)
    res["direct_documents"] = [{"document_id": d, "filename": G.nodes[d].get("filename"), "evidence_ids": evs} for d, evs in roots.items()]
    m["documents_direct"] = len(roots)
    if not roots:
        res["status"] = "NO_DIRECT_DOCUMENTS"; return res
    edges = _xdoc_edges(G)
    adj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges: adj[e["source_id"]].append(e); adj[e["target_id"]].append(e)
    inv = [(n, d) for n, d in _investigation_evidence(G) if not d.get("context_only")]
    doc_ev: Dict[str, List[str]] = defaultdict(list)
    for n, _ in inv:
        for s in _evidence_source_docs(G, n): doc_ev[s["document_id"]].append(n)
    lexv = dict(zip([n for n, _ in inv], bm25_scores(objective, [d.get("text", "") for _, d in inv]))) if (edges and objective) else {}
    members: Dict[str, Dict[str, Any]] = {d: {"hop": 0, "via": None, "path": []} for d in roots}
    seen_pairs: set = set()
    frontier = list(roots)
    accepted = 0
    root_of = lambda d: (members[d]["path"][0]["from_document_id"] if members[d]["path"] else d)

    def _prov(ev: str) -> Dict[str, Any]:
        src = _evidence_source_docs(G, ev)
        return {"evidence_id": ev, "document_id": src[0]["document_id"] if src else None, "filename": src[0]["filename"] if src else None, "location": G.nodes[ev].get("source_location")}

    def _signal(e: Dict[str, Any], doc: str, other: str, cls: str) -> Dict[str, Any]:
        evs = [x for x in _xdoc_edge_doc_evidence(e, other) if x in G and not G.nodes[x].get("context_only")]
        ev = evs[0] if (evs and cls != "XDOC_UNRESOLVED") else None
        base = {"class": cls, "document_id": other, "filename": G.nodes[other].get("filename"), "related_document_id": doc, "related_filename": G.nodes[doc].get("filename"),
                "relationship_type": e.get("relationship_type"), "link_status": e.get("link_status"), "link_reason": e.get("link_reason"), "match_methods": e.get("match_methods"),
                "match_score": e.get("match_score"), "semantic_only": bool(e.get("semantic_only")), "conflicts": e.get("conflicts") or [], "uncertainty_reason": e.get("uncertainty_reason"),
                "matched_evidence": [_prov(x) for x in evs[:4]], "treated_as_identity": False, "requires_reconciliation": cls == "XDOC_CONFLICT",
                "evidence_id": ev, "location": G.nodes[ev].get("source_location") if ev else None, "already_retrieved": bool(ev and ev in direct_set), "hops": members[doc]["hop"] + 1}
        return base

    for hop in range(max_hops):
        nxt: List[str] = []
        for doc in frontier:
            cands = sorted(adj.get(doc, []), key=lambda e: (_XDOC_STATUS_PRIO.get(e.get("link_status"), 9), 0 if _xdoc_edge_scope(e) == "TRANSACTION" else 1, -float(e.get("match_score") or 0),
                                                           str(e.get("target_filename") if e["source_id"] == doc else e.get("source_filename")), e["source_id"], e["target_id"]))
            for e in cands:
                pair = (e["source_id"], e["target_id"])
                if pair in seen_pairs: continue
                if m["links_inspected"] >= link_budget: m["link_budget_exhausted"] = True; break
                seen_pairs.add(pair); m["links_inspected"] += 1
                other = e["target_id"] if e["source_id"] == doc else e["source_id"]
                status = e.get("link_status")
                if status == XDOC_EXPLICIT:
                    ok, why, scope = _xdoc_eligibility(G, e)
                    if ok and other in members:
                        m["links_between_retrieved_documents"] += 1; res["links_between_retrieved_documents"].append(_xdoc_link_brief(e)); continue
                    if ok and scope == "ENTITY" and members[doc]["hop"] > 0: ok, why = False, "entity-level link (same vendor / person / ID value) is not chained beyond a directly retrieved document"
                    if ok and accepted >= doc_limit: ok, why = False, f"related-document limit ({doc_limit}) reached"; m["doc_limit_reached"] = True
                    if ok and scope == "ENTITY" and not any(lexv.get(x, 0.0) > 0.0 for x in doc_ev.get(other, [])): ok, why = False, "entity-level link and the related document has no lexical relevance to the objective"
                    if not ok:
                        m["links_rejected"] += 1; res["rejected_links"].append({**_xdoc_link_brief(e), "reason": why}); continue
                    step = {"from_document_id": doc, "to_document_id": other, "relationship_type": e.get("relationship_type"), "link_status": status, "link_scope": scope}
                    members[other] = {"hop": hop + 1, "via": doc, "path": members[doc]["path"] + [step]}
                    accepted += 1; m["links_accepted_for_traversal"] += 1; nxt.append(other)
                    matched = [x for x in _xdoc_edge_doc_evidence(e, other) if x in set(doc_ev.get(other, []))]
                    lex_ev = sorted([x for x in doc_ev.get(other, []) if lexv.get(x, 0.0) > 0.0 and x not in matched], key=lambda x: (-lexv.get(x, 0.0), x))
                    order = [(x, "matched_field") for x in matched] + [(x, "lexical") for x in lex_ev] if scope == "TRANSACTION" else [(x, "lexical") for x in lex_ev] + [(x, "matched_field") for x in matched]
                    res["related_documents"].append({"document_id": other, "filename": G.nodes[other].get("filename"), "hops": hop + 1, "via_document_id": doc, "root_document_id": root_of(other),
                                                     "relationship_type": e.get("relationship_type"), "link_status": status, "link_scope": scope, "path": members[other]["path"]})
                    taken = 0
                    for ev, basis in order:
                        if taken >= evidence_per_doc: break
                        if ev in direct_set: m["evidence_already_retrieved"] += 1; continue
                        if m["evidence_added_via_explicit_links"] >= evidence_limit: m["evidence_limit_reached"] = True; break
                        taken += 1; m["evidence_added_via_explicit_links"] += 1
                        res["explicit_items"].append({"class": "XDOC_EXPLICIT", "evidence_id": ev, "document_id": other, "filename": G.nodes[other].get("filename"), "location": G.nodes[ev].get("source_location"),
                                                      "related_document_id": doc, "related_filename": G.nodes[doc].get("filename"), "root_document_id": root_of(other), "anchoring_direct_evidence_ids": roots.get(root_of(other), [])[:5],
                                                      "relationship_type": e.get("relationship_type"), "link_status": status, "link_scope": scope, "link_reason": e.get("link_reason"),
                                                      "match_methods": e.get("match_methods"), "match_score": e.get("match_score"), "match_basis": basis, "hops": hop + 1, "path": members[other]["path"],
                                                      "treated_as_identity": False})
                else:
                    cls = {XDOC_WEAK: "XDOC_WEAK", XDOC_CONFLICT: "XDOC_CONFLICT"}.get(status, "XDOC_UNRESOLVED")
                    if len(res["weak_signals"]) + len(res["conflict_signals"]) + len(res["unresolved_signals"]) >= signal_limit: m["signal_limit_reached"] = True; continue
                    sig = _signal(e, doc, other, cls)
                    key = {"XDOC_WEAK": "weak_signals", "XDOC_CONFLICT": "conflict_signals", "XDOC_UNRESOLVED": "unresolved_signals"}[cls]
                    res[key].append(sig)
                    m[{"XDOC_WEAK": "weak_links_surfaced", "XDOC_CONFLICT": "conflicting_links_surfaced", "XDOC_UNRESOLVED": "unresolved_links_surfaced"}[cls]] += 1
                    if sig["evidence_id"] and not sig["already_retrieved"]: m["evidence_added_via_signal_links"] += 1
                    elif sig["already_retrieved"]: m["evidence_already_retrieved"] += 1
            if m["link_budget_exhausted"]: break
        frontier = nxt
        if not frontier or m["link_budget_exhausted"]: break
    m["frontier_not_expanded"] = len(frontier) if hop == max_hops - 1 else 0
    m["documents_via_cross_document_links"] = len(res["related_documents"])
    for d in members:  # missing referenced documents of the documents in the chain (existing Phase A data; nothing invented)
        for r in G.nodes[d].get("xdoc_unresolved_references") or []:
            if len(res["missing_references"]) >= V2_XDOC_MISSING_LIMIT: break
            res["missing_references"].append({"document_id": d, "filename": G.nodes[d].get("filename"), "reference_type": r.get("reference_type"), "reference": r.get("reference"),
                                              "reason": r.get("reason"), "provenance": r.get("provenance") or [], "document_reached_via_link": members[d]["hop"] > 0})
    m["unresolved_references_surfaced"] = len(res["missing_references"])
    if not edges and not res["missing_references"]: res["status"] = "NO_CROSS_DOCUMENT_LINKS"
    return res

def cross_document_has_content(xd: Optional[Dict[str, Any]]) -> bool:
    return bool(xd) and any(xd.get(k) for k in ("explicit_items", "weak_signals", "conflict_signals", "unresolved_signals", "missing_references", "related_documents"))

def format_cross_document_for_context(G: nx.MultiDiGraph, xd: Dict[str, Any]) -> str:
    """V2 context block. '' when there is nothing cross-document to say (so single-document / unlinked cases are unchanged)."""
    if not cross_document_has_content(xd): return ""
    m = xd["metrics"]
    out = ("--- CROSS-DOCUMENT EVIDENCE (Phase B: evidence from OTHER documents reached over deterministic CROSS_DOCUMENT_LINK edges from a directly retrieved document) ---\n"
           "Classes (kept separate from [LEXICAL] / [GRAPH-EXPANDED]): [XDOC-EXPLICIT] established link (strong identifier / party match); [XDOC-POSSIBLE] weak link, a possibility only; "
           "[XDOC-CONFLICT] linked documents disagree, reconciliation required; [XDOC-UNRESOLVED] signal too weak to associate documents; [MISSING-REFERENCE] a referenced document was not supplied.\n")
    for it in xd["explicit_items"]:
        hdr = (f"[XDOC-EXPLICIT] Evidence Node {it['evidence_id']} from Document {it['document_id']} ({it['filename']}) | reached from Document {it['related_document_id']} ({it['related_filename']}) "
               f"over a {it['relationship_type']} link (status EXPLICIT, scope {it['link_scope']}, {it['hops']} hop(s), basis {it['match_basis']}; methods {it['match_methods']}; score {it['match_score']}, rule-based, not a probability) | reason: {it['link_reason']}")
        out += _describe_evidence(G, it["evidence_id"], G.nodes[it["evidence_id"]], hdr, "CROSS-DOCUMENT (explicit link; NOT a lexical match unless basis is lexical)")
    for cls, tag, items, note in (("XDOC_WEAK", "[XDOC-POSSIBLE]", xd["weak_signals"], "WEAK link: a POSSIBLE relationship only. It does NOT establish that these documents refer to the same entity or transaction"),
                                  ("XDOC_CONFLICT", "[XDOC-CONFLICT]", xd["conflict_signals"], "CONFLICTING link: a match exists but fields disagree. RECONCILIATION REQUIRED; do not pick a side")):
        for it in items:
            hdr = (f"{tag} Document {it['document_id']} ({it['filename']}) vs Document {it['related_document_id']} ({it['related_filename']}) | {it['relationship_type']} | status {it['link_status']}"
                   f"{' | SEMANTIC-ONLY (lexical similarity is not identity)' if it['semantic_only'] else ''} | {note} | reason: {it['link_reason']}"
                   + (f" | conflicts: {[(c.get('field'), c.get('kind')) for c in it['conflicts']]}" if it["conflicts"] else ""))
            if it["evidence_id"] and not it["already_retrieved"]: out += _describe_evidence(G, it["evidence_id"], G.nodes[it["evidence_id"]], hdr, f"CROSS-DOCUMENT ({it['link_status']} link; context only)")
            else: out += hdr + (f" | matched evidence {it['evidence_id']} is already part of the retrieved evidence" if it["already_retrieved"] else " | no matched evidence text available") + "\n\n"
    for it in xd["unresolved_signals"]:
        out += (f"[XDOC-UNRESOLVED] Document {it['document_id']} ({it['filename']}) vs Document {it['related_document_id']} ({it['related_filename']}) | status UNRESOLVED | {it['link_reason']} | "
                "NO relationship is established; evidence text not included.\n\n")
    for r in xd["missing_references"]:
        pv = "; ".join(f"{p.get('filename')} @ {p.get('location')} (evidence {p.get('evidence_id')})" for p in r["provenance"][:2])
        out += (f"[MISSING-REFERENCE] Document {r['document_id']} ({r['filename']}) refers to {r['reference_type']} '{r['reference']}' [{pv}]; {r['reason']}. MISSING EVIDENCE: "
                "do not assume the contents of that document, and treat conclusions that depend on it as incomplete.\n\n")
    out += (f"Cross-document retrieval summary: directly retrieved documents {m['documents_direct']}; related documents reached {m['documents_via_cross_document_links']}; links inspected {m['links_inspected']}; "
            f"accepted for traversal {m['links_accepted_for_traversal']}; rejected {m['links_rejected']}; weak {m['weak_links_surfaced']}, conflicting {m['conflicting_links_surfaced']}, unresolved {m['unresolved_links_surfaced']} surfaced; "
            f"missing references {m['unresolved_references_surfaced']}; evidence items added through explicit links {m['evidence_added_via_explicit_links']}.\n")
    lim = [x for x, f in (("related-document limit", m["doc_limit_reached"]), ("evidence limit", m["evidence_limit_reached"]), ("signal limit", m["signal_limit_reached"]), ("link-inspection budget", m["link_budget_exhausted"])) if f]
    if lim: out += f"LIMITS REACHED: {', '.join(lim)}; further related evidence may exist. "
    if m["frontier_not_expanded"]: out += f"{m['frontier_not_expanded']} related document(s) sit at the hop limit and their links were not examined. "
    out += ("RULES: only [XDOC-EXPLICIT] items are established document relationships and they still do not prove compliance, a violation, approval validity or reconciled amounts. A cross-document link never creates "
            "VIOLATION or SATISFIED; only DETERMINISTIC policy results can. " + XDOC_PHASE_B_LIMITS + "\n\n")
    return out

def v2_cross_document_section(stats: Dict[str, Any]) -> str:
    """Deterministic V2 appendix block: which documents were reached through which link and why, kept apart by class. '' when there is no cross-document activity."""
    xd = (stats or {}).get("cross_document")
    if not cross_document_has_content(xd): return ""
    m = xd["metrics"]
    out = "\n### Cross-document evidence (V2 Phase B, deterministic)\n\n"
    out += (f"Directly retrieved documents: **{m['documents_direct']}**; related documents reached through explicit links: **{m['documents_via_cross_document_links']}**; links inspected: **{m['links_inspected']}**; "
            f"accepted for traversal: **{m['links_accepted_for_traversal']}**; rejected: **{m['links_rejected']}**; evidence items added through explicit links: **{m['evidence_added_via_explicit_links']}**.\n\n")
    def rows(title, items, cols):
        o = f"{title}\n\n"
        for i, it in enumerate(items, 1): o += f"{i}. " + "; ".join(f"{k}: `{it.get(v)}`" for k, v in cols) + "\n"
        return o + "\n"
    if xd["explicit_items"]:
        out += rows("**Retrieved through an EXPLICIT cross-document link** (established document relationship; not a direct retrieval and not a compliance finding):", xd["explicit_items"],
                    [("evidence", "evidence_id"), ("document", "filename"), ("location", "location"), ("reached from", "related_filename"), ("relationship", "relationship_type"), ("status", "link_status"), ("hops", "hops"), ("basis", "match_basis"), ("reason", "link_reason")])
    if xd["weak_signals"]:
        out += rows("**Possible (WEAK) cross-document links** (not identity; not traversed):", xd["weak_signals"], [("document", "filename"), ("related to", "related_filename"), ("relationship", "relationship_type"), ("semantic-only", "semantic_only"), ("reason", "link_reason")])
    if xd["conflict_signals"]:
        out += rows("**CONFLICTING cross-document links** (reconciliation required; not resolved):", xd["conflict_signals"], [("document", "filename"), ("related to", "related_filename"), ("relationship", "relationship_type"), ("conflicts", "conflicts"), ("reason", "link_reason")])
    if xd["unresolved_signals"]:
        out += rows("**UNRESOLVED cross-document links** (no relationship established):", xd["unresolved_signals"], [("document", "filename"), ("related to", "related_filename"), ("reason", "link_reason")])
    if xd["missing_references"]:
        out += rows("**Missing referenced documents** (not supplied; nothing was inferred about them):", xd["missing_references"], [("referenced by", "filename"), ("type", "reference_type"), ("reference", "reference"), ("reason", "reason")])
    lim = [x for x, f in (("related-document limit", m["doc_limit_reached"]), ("evidence limit", m["evidence_limit_reached"]), ("signal limit", m["signal_limit_reached"]), ("link-inspection budget", m["link_budget_exhausted"])) if f]
    if lim: out += f"Limits reached: {', '.join(lim)}; further related evidence may exist.\n\n"
    return out + XDOC_PHASE_B_LIMITS + "\n\n"

# --- CONTRADICTION CLASSIFIER (deterministic; builds on the Phase A conflict signals + provenance; no new graph nodes/edges) ---
# Classifies what the linked documents say about the SAME transaction: CONSISTENT / MINOR_CONTRADICTION / MAJOR_CONTRADICTION / MISSING_EVIDENCE / POLICY_VIOLATION / UNRESOLVED.
# * Only documents joined by a STRONG TRANSACTION-level link (identifier anchor, not semantic-only) are compared. Entity-level links (same vendor / person) do not show the
#   documents are the same transaction, so their Phase A conflicts are suppressed (counted, not classified); weak / semantic-only links yield UNRESOLVED, never a contradiction.
# * Both claims are kept side by side with document / evidence ids and locations (existing _xdoc_prov provenance). Nothing is resolved or picked.
# * Legitimate differences that can be recognised deterministically (payment that states it is partial, line items summing to a total) are CONSISTENT with the reason recorded.
# * POLICY_VIOLATION is only ever a COPY of an existing deterministic Decision with verdict VIOLATION. A contradiction is never converted into a violation here.
# * Results: Document-link edge properties contradiction_class / contradiction_findings + G.graph["contradiction_findings"]; recomputed from scratch each call (idempotent).
XCON_CATEGORIES = ("CONSISTENT", "MINOR_CONTRADICTION", "MAJOR_CONTRADICTION", "MISSING_EVIDENCE", "POLICY_VIOLATION", "UNRESOLVED")
XCON_CONTRADICTION = ("MINOR_CONTRADICTION", "MAJOR_CONTRADICTION")
_XCON_PRIO = {"MAJOR_CONTRADICTION": 0, "POLICY_VIOLATION": 1, "MINOR_CONTRADICTION": 2, "UNRESOLVED": 3, "MISSING_EVIDENCE": 4, "CONSISTENT": 5}
XCON_MINOR_REL = 0.05  # amounts / quantities within 5% of each other: minor
XCON_MINOR_DATE_DAYS = 7
_XCON_PARTIAL = re.compile(r'(?i)\b(partial|part[\s-]*payment|instal+ment|advance|on\s+account|balance)\b')
_XCON_DATE_LABEL = re.compile(r'(?i)\b(invoice|order|payment|paid|approval|delivery|due)\s+date\b\s*[:=]?\s*(.{0,30})')
_XCON_APPROVAL_LABELLED = re.compile(r'(?i)\bapproval\s+status\s*[:=]\s*([A-Za-z]{3,12})')
_XCON_STATUS_BARE = re.compile(r'(?i)\bstatus\s*[:=]\s*([A-Za-z]{3,12})')
_XCON_QTY = re.compile(r'(?i)\b(?:quantity|qty)\b\s*[:=]?\s*(\d+(?:\.\d+)?)')
_XCON_DELIVERY = re.compile(r'(?i)\bdelivery\s+status\s*[:=]\s*([A-Za-z ]{3,25})')
_XCON_PAYSTATUS = re.compile(r'(?i)\bpayment\s+status\s*[:=]\s*([A-Za-z ]{3,25})')

def _xcon_norm_approval(v: str) -> Optional[str]:
    v = v.lower()
    return "APPROVED" if v in ("approved", "granted", "accepted") else ("REJECTED" if v in ("rejected", "declined", "denied", "refused") else ("PENDING" if v == "pending" else None))

def _xcon_norm_delivery(v: str) -> Optional[str]:
    v = v.lower()
    return "NOT_DELIVERED" if re.search(r'\bnot\b|\bun|pending|awaiting|\bno\b', v) else ("PARTIAL" if "partial" in v else ("DELIVERED" if "deliver" in v else None))

def _xcon_norm_paystatus(v: str) -> Optional[str]:
    v = v.lower()
    return "UNPAID" if re.search(r'unpaid|not paid|pending|\bdue\b|overdue', v) else ("PARTIAL" if "partial" in v else ("PAID" if re.search(r'\bpaid\b|settled|complete', v) else None))

def _xcon_doc_claims(G: nx.MultiDiGraph, prof: Dict[str, Any]) -> Dict[str, Any]:
    """Labelled dates, approval status and factual claims of one document, each value with its provenance."""
    did = prof["doc_id"]
    out: Dict[str, Any] = {"dates": defaultdict(lambda: defaultdict(list)), "approval": defaultdict(list), "quantity": defaultdict(list), "delivery_status": defaultdict(list), "payment_status": defaultdict(list)}
    for ev in prof["evidence_ids"]:
        text = G.nodes[ev].get("text") or ""
        for m in _XCON_DATE_LABEL.finditer(text):
            best = None
            for pat, fmt in _DATE_PATTERNS:
                mm = pat.search(m.group(2))
                if mm and mm.start() <= 2 and (best is None or mm.start() < best[0]):
                    try: best = (mm.start(), fmt(mm), mm.group(0))
                    except (KeyError, ValueError): pass
            if best: out["dates"]["payment" if m.group(1).lower() == "paid" else m.group(1).lower()][best[1]].append(_xdoc_prov(G, ev, did, f"{m.group(1)} date {best[2]}"))
        for m in _XCON_APPROVAL_LABELLED.finditer(text):
            s = _xcon_norm_approval(m.group(1))
            if s: out["approval"][s].append(_xdoc_prov(G, ev, did, m.group(0)))
        if prof["role"] == "APPROVAL":
            for m in _XCON_STATUS_BARE.finditer(text):
                s = _xcon_norm_approval(m.group(1))
                if s: out["approval"][s].append(_xdoc_prov(G, ev, did, m.group(0)))
        for m in _XCON_QTY.finditer(text): out["quantity"][float(m.group(1))].append(_xdoc_prov(G, ev, did, m.group(0)))
        for rx, key, norm in ((_XCON_DELIVERY, "delivery_status", _xcon_norm_delivery), (_XCON_PAYSTATUS, "payment_status", _xcon_norm_paystatus)):
            for m in rx.finditer(text):
                s = norm(m.group(1))
                if s: out[key][s].append(_xdoc_prov(G, ev, did, m.group(0)))
    return out

def _xcon_side(G: nx.MultiDiGraph, doc_id: str, value: Any, provs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"document_id": doc_id, "filename": G.nodes[doc_id].get("filename"), "value": value, "provenance": provs[:5]}

def _xcon_mk(cat: str, field: str, a: Dict[str, Any], b: Optional[Dict[str, Any]], reason: str, link: Optional[Dict[str, Any]] = None, severity: Optional[str] = None, **extra) -> Dict[str, Any]:
    key = json.dumps([cat, field, sorted(x["filename"] or "" for x in (a, b) if x), [({k: v for k, v in x["value"].items() if not k.endswith("_id")} if isinstance(x["value"], dict) else x["value"]) for x in (a, b) if x], extra.get("subfield")], sort_keys=True, default=str)
    return {"finding_id": "xcon_" + hashlib.sha256(key.encode()).hexdigest()[:12], "category": cat, "field": field, "severity": severity or {"MAJOR_CONTRADICTION": "major", "MINOR_CONTRADICTION": "minor"}.get(cat),
            "claim_a": a, "claim_b": b, "reason": reason, "link": link, "resolution": "NOT_RESOLVED", "is_policy_violation": False, "heuristic": cat != "POLICY_VIOLATION", **extra}

def _xcon_rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)

def _xcon_edit1(a: str, b: str) -> bool:
    if a == b: return True
    if abs(len(a) - len(b)) > 1: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))

def _xcon_compare_pair(G: nx.MultiDiGraph, e: Dict[str, Any], pa: Dict[str, Any], pb: Dict[str, Any], ca: Dict[str, Any], cb: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    A, B = pa["doc_id"], pb["doc_id"]
    link = {"relationship_type": e.get("relationship_type"), "link_status": e.get("link_status"), "link_reason": e.get("link_reason"), "match_methods": e.get("match_methods")}
    conf = {c["field"] for c in e.get("conflicts") or []}
    F: List[Dict[str, Any]] = []
    compared: List[str] = []
    flat = lambda d, pred=lambda k: True: [p for k, ps in d.items() if pred(k) for p in ps]
    # amounts (trigger: Phase A amount / currency conflict; details recomputed from the same profile values)
    if pa["amounts"] and pb["amounts"]:
        common = sorted({c for c, _ in pa["amounts"]} & {c for c, _ in pb["amounts"]}, key=str)
        if not common:
            compared.append("amount")
            if "currency" in conf:
                F.append(_xcon_mk("UNRESOLVED", "amount", _xcon_side(G, A, sorted({str(c) for c, _ in pa["amounts"]}), flat(pa["amounts"])), _xcon_side(G, B, sorted({str(c) for c, _ in pb["amounts"]}), flat(pb["amounts"])),
                                  "amounts are in different currencies and no exchange rate is applied, so they cannot be compared", link))
        for cur in common:
            compared.append("amount")
            va, vb = sorted({x for c, x in pa["amounts"] if c == cur}), sorted({x for c, x in pb["amounts"] if c == cur})
            if set(va) & set(vb): continue
            sa = _xcon_side(G, A, {"currency": cur, "amounts": va}, flat(pa["amounts"], lambda k: k[0] == cur))
            sb = _xcon_side(G, B, {"currency": cur, "amounts": vb}, flat(pb["amounts"], lambda k: k[0] == cur))
            if any(abs(sum(va) - x) <= 0.01 for x in vb) or any(abs(sum(vb) - x) <= 0.01 for x in va):
                F.append(_xcon_mk("CONSISTENT", "amount", sa, sb, "line items of one document sum to the amount in the other", link, legitimate_difference="line_item_total")); continue
            if {pa["role"], pb["role"]} >= {"PAYMENT"} and pa["role"] != pb["role"]:
                pay, oth = (pa, pb) if pa["role"] == "PAYMENT" else (pb, pa)
                vp, vo = (va, vb) if pay is pa else (vb, va)
                if max(vp) > max(vo): F.append(_xcon_mk("MAJOR_CONTRADICTION", "amount", sa, sb, "payment exceeds the amount of the document it settles", link)); continue
                if _XCON_PARTIAL.search(pay["text"]):
                    F.append(_xcon_mk("CONSISTENT", "amount", sa, sb, "payment states it is partial / on account and is smaller than the billed amount", link, legitimate_difference="partial_payment")); continue
                F.append(_xcon_mk("UNRESOLVED", "amount", sa, sb, "payment is smaller than the billed amount and nothing states whether it is partial", link)); continue
            d = min(_xcon_rel(x, y) for x in va for y in vb)
            F.append(_xcon_mk("MINOR_CONTRADICTION" if d <= XCON_MINOR_REL else "MAJOR_CONTRADICTION", "amount", sa, sb, f"amounts differ by {d:.1%} (minor threshold {XCON_MINOR_REL:.0%})", link))
    # transaction ids of the same type
    for t in sorted(({x for x, _ in pa["ids"]} & {x for x, _ in pb["ids"]}) - {"ref"}):
        compared.append("transaction_id")
        ia, ib = {n for x, n in pa["ids"] if x == t}, {n for x, n in pb["ids"] if x == t}
        if ia & ib: continue
        sa, sb = _xcon_side(G, A, {"type": t, "ids": sorted(ia)}, flat(pa["ids"], lambda k: k[0] == t)), _xcon_side(G, B, {"type": t, "ids": sorted(ib)}, flat(pb["ids"], lambda k: k[0] == t))
        if pa["role"] == pb["role"] == t.upper():
            if set(pa["amounts"]) & set(pb["amounts"]): F.append(_xcon_mk("MAJOR_CONTRADICTION", "transaction_id", sa, sb, "two documents of the same kind with different numbers carry the same amount against one anchor (duplicate-document signature)", link))
            else: F.append(_xcon_mk("UNRESOLVED", "transaction_id", sa, sb, "two documents of the same kind with different numbers against one anchor: separate (e.g. partial) documents or a duplicate cannot be told apart", link))
        elif len(ia) > 1 or len(ib) > 1: F.append(_xcon_mk("UNRESOLVED", "transaction_id", sa, sb, "one document lists several identifiers of this type; a mismatch cannot be established", link))
        elif any(_xcon_edit1(x, y) for x in ia for y in ib): F.append(_xcon_mk("MINOR_CONTRADICTION", "transaction_id", sa, sb, "identifiers differ by one character (possible typo; not resolved)", link))
        else: F.append(_xcon_mk("MAJOR_CONTRADICTION", "transaction_id", sa, sb, "linked documents cite different identifiers of the same type", link))
    # vendor / person names, valid GovIDs
    for fld, field in (("vendor", "vendor"), ("person", "identity")):
        if not (pa[fld] and pb[fld]): continue
        compared.append(field)
        if set(pa[fld]) & set(pb[fld]): continue
        sa, sb = _xcon_side(G, A, sorted(pa[fld]), flat(pa[fld])), _xcon_side(G, B, sorted(pb[fld]), flat(pb[fld]))
        close = any((lambda x, y: x <= y or y <= x or len(x & y) / len(x | y) >= 0.5)(set(p.split()), set(q.split())) for p in pa[fld] for q in pb[fld])
        F.append(_xcon_mk("MINOR_CONTRADICTION" if close else "MAJOR_CONTRADICTION", field, sa, sb, "names differ but share most words (variant spelling / suffix; not resolved)" if close else "linked documents name different parties", link))
    ga, gb = {k for k in pa["entities"] if pa["entity_meta"][k][0] == "GovID"}, {k for k in pb["entities"] if pb["entity_meta"][k][0] == "GovID"}
    if ga and gb:
        compared.append("identity")
        if not (ga & gb):
            ok = all(pa["entity_meta"][k][1] is True for k in ga) and all(pb["entity_meta"][k][1] is True for k in gb)
            F.append(_xcon_mk("MAJOR_CONTRADICTION" if ok else "UNRESOLVED", "identity", _xcon_side(G, A, "[REDACTED_GOVID]", flat({k: pa["entities"][k] for k in ga})), _xcon_side(G, B, "[REDACTED_GOVID]", flat({k: pb["entities"][k] for k in gb})),
                              "linked documents carry different Government IDs" if ok else "different Government IDs but at least one is unverified", link))
    # labelled dates
    for kind in sorted(set(ca["dates"]) & set(cb["dates"])):
        compared.append("date")
        da, db = ca["dates"][kind], cb["dates"][kind]
        if set(da) & set(db): continue
        sa, sb = _xcon_side(G, A, {"kind": kind, "dates": sorted(da)}, flat(da)), _xcon_side(G, B, {"kind": kind, "dates": sorted(db)}, flat(db))
        try:
            gap = min(abs((datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(y, "%Y-%m-%d")).days) for x in da for y in db)
            F.append(_xcon_mk("MINOR_CONTRADICTION" if gap <= XCON_MINOR_DATE_DAYS else "MAJOR_CONTRADICTION", "date", sa, sb, f"the {kind} date differs by {gap} day(s)", link, subfield=kind))
        except ValueError: F.append(_xcon_mk("UNRESOLVED", "date", sa, sb, f"the {kind} dates differ but are not in an unambiguous format", link, subfield=kind))
    for (cx, dx), (cy, dy) in (((ca, A), (cb, B)), ((cb, B), (ca, A))):  # an invoice dated before the order it cites
        if cx["dates"].get("order") and cy["dates"].get("invoice"):
            od, iv = min(cx["dates"]["order"]), min(cy["dates"]["invoice"])
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', od) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', iv) and iv < od:
                F.append(_xcon_mk("MAJOR_CONTRADICTION", "date", _xcon_side(G, dx, {"kind": "order", "dates": [od]}, cx["dates"]["order"][od]), _xcon_side(G, dy, {"kind": "invoice", "dates": [iv]}, cy["dates"]["invoice"][iv]),
                                  "the invoice is dated before the order it cites", link, subfield="sequence"))
                compared.append("date")
    # approval status
    if ca["approval"] and cb["approval"]:
        compared.append("approval_status")
        sa_, sb_ = set(ca["approval"]), set(cb["approval"])
        if not (sa_ & sb_):
            side = lambda d, c: _xcon_side(G, d, sorted(c), flat(c))
            hard = ("APPROVED" in sa_ and "REJECTED" in sb_) or ("REJECTED" in sa_ and "APPROVED" in sb_)
            F.append(_xcon_mk("MAJOR_CONTRADICTION" if hard else "UNRESOLVED", "approval_status", side(A, ca["approval"]), side(B, cb["approval"]),
                              "one document records the approval as approved and the other as rejected" if hard else "approval statuses differ but one is pending, so a later decision may simply have been recorded", link))
    # factual claims
    q_a, q_b = ca["quantity"], cb["quantity"]
    if q_a and q_b:
        compared.append("factual_claim")
        if not (set(q_a) & set(q_b)):
            d = min(_xcon_rel(x, y) for x in q_a for y in q_b)
            F.append(_xcon_mk("MINOR_CONTRADICTION" if d <= XCON_MINOR_REL else "MAJOR_CONTRADICTION", "factual_claim", _xcon_side(G, A, {"claim": "quantity", "values": sorted(q_a)}, flat(q_a)),
                              _xcon_side(G, B, {"claim": "quantity", "values": sorted(q_b)}, flat(q_b)), f"quantities differ by {d:.1%}", link, subfield="quantity"))
    for key, hard_pairs in (("delivery_status", {frozenset(("DELIVERED", "NOT_DELIVERED"))}), ("payment_status", {frozenset(("PAID", "UNPAID"))})):
        if ca[key] and cb[key]:
            compared.append("factual_claim")
            if not (set(ca[key]) & set(cb[key])):
                hard = any(frozenset((x, y)) in hard_pairs for x in ca[key] for y in cb[key])
                F.append(_xcon_mk("MAJOR_CONTRADICTION" if hard else "MINOR_CONTRADICTION", "factual_claim", _xcon_side(G, A, {"claim": key, "values": sorted(ca[key])}, flat(ca[key])),
                                  _xcon_side(G, B, {"claim": key, "values": sorted(cb[key])}, flat(cb[key])), "the documents state opposite " + key.replace("_", " ") if hard else key.replace("_", " ") + " differs (partial vs complete)", link, subfield=key))
    return F, sorted(set(compared))

def classify_contradictions(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Recomputed from scratch on every call (idempotent): adds no node or edge. See the header comment for the rules."""
    summary: Dict[str, Any] = {"status": "OK", "pairs_compared": 0, "suppressed_entity_level_conflicts": 0, "by_category": {}, "note": "heuristic classification; no finding is a compliance violation (POLICY_VIOLATION findings copy existing deterministic Decisions)"}
    live = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == XDOC_RELATION]
    for _, _, d in live: d["contradiction_findings"], d["contradiction_class"] = [], "NOT_ASSESSED"
    profiles = _xdoc_build_profiles(G)
    claims = {k: _xcon_doc_claims(G, p) for k, p in profiles.items()}
    allf: List[Dict[str, Any]] = []
    for u, v, d in live:
        e = dict(d, source_id=u, target_id=v)
        if u not in profiles or v not in profiles: continue
        a, b = profiles[u], profiles[v]
        scope = _xdoc_edge_scope(e)
        if e.get("link_status") in (XDOC_EXPLICIT, XDOC_CONFLICT) and scope == "TRANSACTION" and not e.get("semantic_only"):
            F, compared = _xcon_compare_pair(G, e, a, b, claims[u], claims[v])
            summary["pairs_compared"] += 1
            if not F and compared:
                F = [_xcon_mk("CONSISTENT", "*", _xcon_side(G, u, None, []), _xcon_side(G, v, None, []), "all compared fields agree: " + ", ".join(compared), {"relationship_type": e.get("relationship_type"), "link_status": e.get("link_status"), "link_reason": e.get("link_reason")}, compared_fields=compared)]
        elif e.get("link_status") == XDOC_CONFLICT and scope == "ENTITY": F = []; summary["suppressed_entity_level_conflicts"] += 1  # same vendor / person only: different transactions legitimately differ
        elif e.get("link_status") == XDOC_CONFLICT and scope is None:
            F = [_xcon_mk("UNRESOLVED", "link", _xcon_side(G, u, sorted({c["field"] for c in e["conflicts"]}), []), _xcon_side(G, v, None, []), "fields differ but the documents are linked only weakly" + (" (semantic similarity)" if e.get("semantic_only") else "") + ", so it is not established they describe the same transaction",
                              {"relationship_type": e.get("relationship_type"), "link_status": e.get("link_status"), "link_reason": e.get("link_reason")}, conflicting_fields=sorted({c["field"] for c in e["conflicts"]}))]
        else: F = []
        F.sort(key=lambda f: (_XCON_PRIO[f["category"]], f["field"], f["finding_id"]))
        d["contradiction_findings"] = F
        d["contradiction_class"] = F[0]["category"] if F else "NOT_ASSESSED"
        allf += F
    for did, p in profiles.items():  # information needed to resolve is absent: a referenced document was not supplied (Phase A xdoc_unresolved_references)
        for r in G.nodes[did].get("xdoc_unresolved_references") or []:
            allf.append(_xcon_mk("MISSING_EVIDENCE", "reference", _xcon_side(G, did, {"reference_type": r.get("reference_type"), "reference": r.get("reference")}, r.get("provenance") or []), None,
                                 "the document refers to a document that was not supplied; nothing is assumed about it"))
    viol_by_ev: Dict[str, List[str]] = defaultdict(list)
    for dec, dd in _nodes_of_type(G, "Decision"):  # POLICY_VIOLATION = existing deterministic verdicts only
        if dd.get("verdict") != "VIOLATION": continue
        evs = _supporting_evidence_ids(G, dec)
        for ev in evs: viol_by_ev[ev].append(dec)
        provs = [{"evidence_id": ev, "document_id": (_evidence_source_docs(G, ev) or [{}])[0].get("document_id"), "filename": (_evidence_source_docs(G, ev) or [{}])[0].get("filename"), "location": G.nodes[ev].get("source_location")} for ev in evs]
        if provs:
            f = _xcon_mk("POLICY_VIOLATION", "policy", {"document_id": provs[0]["document_id"], "filename": provs[0]["filename"], "value": {"decision_id": dec, "rule_id": dd.get("rule_id"), "rule": G.nodes[dd["rule_id"]].get("condition") if dd.get("rule_id") in G else None}, "provenance": provs[:5]}, None,
                         "existing deterministic Decision with verdict VIOLATION (copied, not derived from any contradiction)", None, severity="deterministic", source="existing_deterministic_decision", derived_from_contradiction=False,
                         files=sorted({p["filename"] for p in provs if p["filename"]}))
            allf.append(f)
    for f in allf:  # information only: a contradictory claim's evidence that also underlies an existing violation decision (category unchanged)
        if f["category"] in XCON_CONTRADICTION:
            f["policy_decisions_on_same_evidence"] = sorted({dec for s in (f["claim_a"], f["claim_b"]) if s for p in s["provenance"] for dec in viol_by_ev.get(p.get("evidence_id"), [])})
    allf.sort(key=lambda f: (_XCON_PRIO[f["category"]], f["field"], f["finding_id"]))
    G.graph["contradiction_findings"] = allf
    summary["by_category"] = dict(Counter(f["category"] for f in allf))
    summary["findings"] = len(allf)
    G.graph["contradiction_summary"] = summary
    return summary

def v2_contradiction_section(G: nx.MultiDiGraph) -> str:
    """Deterministic report block; '' when there is nothing to report."""
    fs = [f for f in G.graph.get("contradiction_findings") or [] if f["category"] != "CONSISTENT"]
    if not fs: return ""
    out = ("\n### Contradiction classification (deterministic, heuristic)\n\nFindings compare what linked documents say about the same transaction. A contradiction is NOT a compliance violation, nothing here is resolved, "
           "and POLICY_VIOLATION entries are copies of existing deterministic policy decisions.\n\n")
    for i, f in enumerate(fs, 1):
        cl = lambda s: "-" if not s else f"{s['filename']}: `{s['value']}` [" + "; ".join(f"{p.get('filename')} @ {p.get('location')} (evidence {p.get('evidence_id')})" for p in s["provenance"][:2]) + "]"
        out += f"{i}. **{f['category']}** ({f['field']}): {f['reason']}. Claim A: {cl(f['claim_a'])}. Claim B: {cl(f['claim_b'])}.\n"
    return out + "\n"

# --- SELF-VERIFICATION RESULT CONTRACT (Phase 7A: structure only) ---
# One structured, read-only view of an EXISTING Decision for later self-verification phases. It only REUSES query_decision_lineage, the stored Decision attributes, the
# contradiction findings (G.graph["contradiction_findings"]) and the Evidence provenance. It creates no node / edge, changes no verdict and invents no evidence: every evidence
# reference is an Evidence node that exists in the graph, with its stored document id / location / provenance. No verification is performed here, so verification_status is
# NOT_VERIFIED and escalation_reason is None; confidence.value is None (NOT_MEASURED) because no calibrated decision-level confidence exists (measured components are listed, never aggregated).
SELF_VERIFICATION_FIELDS = ("decision", "confidence", "supporting_evidence", "contradicting_evidence", "policy_rules", "missing_evidence", "verification_status", "escalation_reason")
SELF_VERIFICATION_STATUSES = ("NOT_VERIFIED", "VERIFIED", "FAILED", "ESCALATE")  # 7A produced NOT_VERIFIED only; 7B (verify_self_verification_result) may set the rest

def _sv_json(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))

def _sv_evidence_refs(G: nx.MultiDiGraph, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    eid = item["evidence_id"]
    base = {"evidence_id": eid, "origins": list(item.get("origins") or []), "via_nodes": list(item.get("via_nodes") or []), "context_only": bool(item.get("context_only")), "provenance": _sv_json(item.get("provenance"))}
    docs = item.get("documents") or []
    if not docs: return [{**base, "document_id": None, "filename": None, "location": G.nodes[eid].get("source_location"), "file_hash": None}]
    return [{**base, "document_id": d["document_id"], "filename": d.get("filename"), "location": d.get("location"), "file_hash": d.get("file_hash")} for d in docs]

def build_self_verification_result(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Read-only; recomputed on every call (idempotent). All eight contract fields are always present."""
    res: Dict[str, Any] = {"contract_version": "7A", "decision_id": decision_id, "found": False, "decision": None,
                           "confidence": {"value": None, "status": "NOT_MEASURED", "basis": "no calibrated decision-level confidence exists; measured components are reported as stored and never aggregated", "components": {"compiled_rules": [], "evidence_extraction": []}},
                           "supporting_evidence": [], "contradicting_evidence": [], "policy_rules": [], "missing_evidence": [],
                           "verification_status": "NOT_VERIFIED", "escalation_reason": None, "gaps": []}
    if not G.has_node(decision_id) or G.nodes[decision_id].get("type") != "Decision":
        res["gaps"].append("start node missing or not a Decision"); return res
    dd = G.nodes[decision_id]
    lin = query_decision_lineage(G, decision_id)
    res["found"] = True
    res["gaps"] = list(lin.get("gaps") or [])
    res["decision"] = {"decision_id": decision_id, "verdict": dd.get("verdict"), "rule_id": dd.get("rule_id"), "rationale": dd.get("rationale"), "result_source": dd.get("result_source"),
                       "evaluation_engine": dd.get("evaluation_engine"), "violation_status": dd.get("violation_status")}
    sup_ids: set = set()
    sup_docs: set = set()
    for item in lin.get("evidence") or []:
        refs = _sv_evidence_refs(G, item)
        res["supporting_evidence"] += refs
        sup_ids.add(item["evidence_id"]); sup_docs |= {r["document_id"] for r in refs if r["document_id"]}
        m = (item.get("provenance") or {})
        if m.get("confidence_value") is not None: res["confidence"]["components"]["evidence_extraction"].append({"evidence_id": item["evidence_id"], "value": m.get("confidence_value"), "basis": m.get("confidence_basis")})
        if not item.get("documents"): res["missing_evidence"].append({"kind": "evidence_without_source_document", "evidence_id": item["evidence_id"], "location": G.nodes[item["evidence_id"]].get("source_location")})
    for c in lin.get("contradicting_evidence") or []:  # heuristic Evidence --CONTRADICTS--> Transaction edges on the basis nodes (possible, not proven)
        if G.has_node(c["evidence_id"]):
            res["contradicting_evidence"].append({"source": "CONTRADICTS_edge", "evidence_id": c["evidence_id"], "transaction_id": c.get("transaction_id"), "heuristic": c.get("heuristic"), "match_strength": c.get("match_strength"),
                                                  "method": c.get("method"), "documents": _sv_json(c.get("documents")), "location": G.nodes[c["evidence_id"]].get("source_location"), "provenance": _sv_json(G.nodes[c["evidence_id"]].get("provenance"))})
    for eid in dd.get("contradicting_evidence_ids") or []:  # evidence behind the OPPOSITE verdict of the same rule (stored by the rule engine)
        if G.has_node(eid) and G.nodes[eid].get("type") == "Evidence":
            res["contradicting_evidence"].append({"source": "opposite_verdict_basis", "evidence_id": eid, "documents": _sv_json(_evidence_source_docs(G, eid)), "location": G.nodes[eid].get("source_location"), "provenance": _sv_json(G.nodes[eid].get("provenance"))})
    for f in G.graph.get("contradiction_findings") or []:  # classifier findings whose claims rest on this decision's supporting evidence; both claims kept as stored
        sides = [s for s in (f.get("claim_a"), f.get("claim_b")) if s]
        touches = any(p.get("evidence_id") in sup_ids for s in sides for p in s.get("provenance") or [])
        if f.get("category") in XCON_CONTRADICTION and touches:
            res["contradicting_evidence"].append({"source": "contradiction_finding", "finding_id": f["finding_id"], "category": f["category"], "field": f["field"], "reason": f.get("reason"), "claim_a": _sv_json(f.get("claim_a")),
                                                  "claim_b": _sv_json(f.get("claim_b")), "resolution": f.get("resolution"), "is_policy_violation": f.get("is_policy_violation")})
        elif f.get("category") == "MISSING_EVIDENCE" and f.get("claim_a") and f["claim_a"].get("document_id") in sup_docs:
            res["missing_evidence"].append({"kind": "referenced_document_not_supplied", "finding_id": f["finding_id"], "claim": _sv_json(f["claim_a"]), "reason": f.get("reason")})
    if lin.get("rule"):
        res["policy_rules"].append({"rule_id": lin["rule"]["node_id"], "condition": lin["rule"].get("condition"), "source_file": lin["rule"].get("source_file"), "source_location": lin["rule"].get("source_location"),
                                    "policies": list(lin.get("policies") or []), "rulebook_provenance": _sv_json(dd.get("rulebook_provenance")), "compiled_rules": _sv_json(dd.get("compiled_rules") or []),
                                    "rule_source_evidence": [{"evidence_id": x["evidence_id"], "documents": _sv_json(x.get("documents"))} for x in lin.get("rule_source") or []]})
    for cr in dd.get("compiled_rules") or []:
        if cr.get("confidence") is not None: res["confidence"]["components"]["compiled_rules"].append({"rule_id": cr.get("rule_id"), "value": cr.get("confidence"), "basis": "policy_compiler_reported"})
    for mf in dd.get("compiled_missing_facts") or []: res["missing_evidence"].append({"kind": "compiled_rule_missing_fact", "fact": mf})
    for g_ in dd.get("extraction_gap_documents") or []: res["missing_evidence"].append({"kind": "extraction_gap", "document": g_})
    if dd.get("verdict") in ("VIOLATION", "SATISFIED") and not sup_ids: res["missing_evidence"].append({"kind": "no_supporting_evidence", "reason": "no evidence linked to this decision"})
    if dd.get("verdict") in ("UNEVALUATED", "INCONCLUSIVE") and dd.get("unevaluated_reason"): res["missing_evidence"].append({"kind": "rule_not_evaluated", "reason": dd.get("unevaluated_reason")})
    return res

def build_self_verification_results(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    return [build_self_verification_result(G, n) for n, _ in sorted(_nodes_of_type(G, "Decision"), key=lambda kv: kv[0])]

def validate_self_verification_result(G: nx.MultiDiGraph, r: Dict[str, Any]) -> List[str]:
    """Contract check: all fields present, status allowed, and every evidence / document reference exists in the graph (nothing invented)."""
    probs = [f"missing field {k}" for k in SELF_VERIFICATION_FIELDS if k not in r]
    if r.get("verification_status") not in SELF_VERIFICATION_STATUSES: probs.append(f"unknown verification_status {r.get('verification_status')}")
    if r.get("escalation_reason") is not None and not isinstance(r["escalation_reason"], str): probs.append("escalation_reason must be None or str")
    for ref in r.get("supporting_evidence") or []:
        eid = ref.get("evidence_id")
        if not (G.has_node(eid) and G.nodes[eid].get("type") == "Evidence"): probs.append(f"supporting evidence {eid} not in graph"); continue
        if ref.get("document_id") and ref["document_id"] not in {x["document_id"] for x in _evidence_source_docs(G, eid)}: probs.append(f"evidence {eid} is not derived from document {ref['document_id']}")
    if (r.get("confidence") or {}).get("value") is not None: probs.append("confidence.value must stay None (NOT_MEASURED)")
    if (r.get("verification_status") == "ESCALATE") != bool(r.get("escalation_reason")): probs.append("escalation_reason must be set exactly when verification_status is ESCALATE")
    for mc in r.get("material_claims") or []:  # 7B: every cited evidence / document must exist and be derived from that document (nothing invented)
        for ref in mc.get("evidence") or []:
            eid = ref.get("evidence_id")
            if not (G.has_node(eid) and G.nodes[eid].get("type") == "Evidence"): probs.append(f"material claim {mc.get('claim_id')} cites evidence {eid} not in graph"); continue
            if ref.get("document_id") and ref["document_id"] not in {x["document_id"] for x in _evidence_source_docs(G, eid)}: probs.append(f"material claim evidence {eid} is not derived from document {ref['document_id']}")
    for c in r.get("contradicting_evidence") or []:
        if c.get("evidence_id") and not G.has_node(c["evidence_id"]): probs.append(f"contradicting evidence {c['evidence_id']} not in graph")
        for s in (c.get("claim_a"), c.get("claim_b")):
            for p in (s or {}).get("provenance") or []:
                if p.get("evidence_id") and not G.has_node(p["evidence_id"]): probs.append(f"claim evidence {p['evidence_id']} not in graph")
    return probs

# --- SELF-VERIFICATION OF MATERIAL CLAIMS (Phase 7B) ---
# Verifies the MATERIAL CLAIMS behind an existing VIOLATION / SATISFIED Decision: the basis nodes (Transaction / Claim / Entity / Evidence) recorded by the rule engine, plus the
# "required text absent within extracted scope" claim. Each claim is re-checked against its ACTUAL source evidence (non-heuristic, non-context Evidence --SUPPORTS--> node, with a
# DERIVED_FROM Document, and, for amounts, the stored evidence text must still contain that currency + amount). Reuses build_self_verification_result / query_decision_lineage /
# contradiction findings; creates no node / edge, changes no verdict, invents no evidence, and leaves confidence.value None (NOT_MEASURED).
#   claim grounding: GROUNDED | WEAK (grounded text-wise but no source location, or text redacted so the value cannot be re-confirmed, or extraction incomplete) | UNSUPPORTED
#   FAILED = any UNSUPPORTED material claim (incl. conclusion with no recorded basis) | ESCALATE = no UNSUPPORTED, but a WEAK claim or a MAJOR contradiction / opposite-verdict evidence is unresolved | VERIFIED otherwise.
#   Decisions asserting no conclusion (UNEVALUATED / INCONCLUSIVE / NOT_APPLICABLE) have nothing to ground: they stay NOT_VERIFIED with a note.
SV_ASSERTING_VERDICTS = ("VIOLATION", "SATISFIED")

def _sv_amount_in_text(text: Optional[str], currency: str, amount: float) -> bool:
    for m in MONEY_PATTERN.finditer(text or ""):
        try: v = float(m.group(2).replace(",", ""))
        except ValueError: continue
        if _normalize_currency(m.group(1)) == currency and abs(v - amount) < 1e-9: return True
    return False

def _sv_claim_spec(G: nx.MultiDiGraph, node_id: str) -> Dict[str, Any]:
    d = G.nodes[node_id]; t = d.get("type")
    spec: Dict[str, Any] = {"node_id": node_id, "node_type": t, "kind": "existence", "currency": None, "amount": None, "description": f"{t} {node_id}"}
    cur, amt = None, None
    if t == "Transaction": cur, amt = d.get("currency"), d.get("amount")
    elif t == "Claim" and d.get("claim_type") == "MonetaryAmount":
        try: cur, amt = str(d.get("value")).split()[0], float(str(d.get("value")).split()[1])
        except (IndexError, ValueError): pass
    if cur and isinstance(amt, (int, float)) and math.isfinite(amt):
        spec.update(kind="amount", currency=cur, amount=float(amt), description=f"{t} {cur} {amt:g}")
    return spec

def _sv_check_claim(G: nx.MultiDiGraph, spec: Dict[str, Any]) -> Dict[str, Any]:
    nid = spec["node_id"]
    out = {"claim_id": f"claim::{nid}", "node_id": nid, "node_type": spec["node_type"], "kind": spec["kind"], "description": spec["description"], "grounding": "UNSUPPORTED", "reason": None, "evidence": []}
    if not G.has_node(nid): out["reason"] = "basis node not found in graph"; return out
    ev_ids = []
    for e in _supporting_evidence_ids(G, nid):
        ed = G.nodes[e]
        if ed.get("type") != "Evidence" or ed.get("context_only") or _is_policy_source_evidence(ed): continue
        if e != nid and not any(not x.get("heuristic") for x in (G.get_edge_data(e, nid) or {}).values() if x.get("relation") == "SUPPORTS"): continue  # heuristic links never ground a claim
        ev_ids.append(e)
    if not ev_ids: out["reason"] = "no qualifying (non-heuristic, non-context) supporting Evidence"; return out
    matched, unconfirmed, mismatched, nodoc = [], [], [], []
    for e in ev_ids:
        docs = _evidence_source_docs(G, e)
        ref = {"evidence_id": e, "document_id": docs[0]["document_id"] if docs else None, "filename": docs[0]["filename"] if docs else None, "location": G.nodes[e].get("source_location") or (docs[0].get("location") if docs else None),
               "document_ids": [x["document_id"] for x in docs], "provenance": _sv_json(G.nodes[e].get("provenance"))}
        out["evidence"].append(ref)
        if not docs: nodoc.append(e); continue
        if spec["kind"] != "amount": matched.append(e)
        elif _sv_amount_in_text(G.nodes[e].get("text"), spec["currency"], spec["amount"]): matched.append(e)
        elif "[REDACTED" in str(G.nodes[e].get("text") or ""): unconfirmed.append(e)
        else: mismatched.append(e)
    located = lambda e: bool(next(r for r in out["evidence"] if r["evidence_id"] == e)["location"])
    if matched and any(located(e) for e in matched): out["grounding"], out["reason"] = "GROUNDED", f"found in {len(matched)} source evidence item(s)"
    elif matched: out["grounding"], out["reason"] = "WEAK", "supported by evidence that has no recorded source location"
    elif unconfirmed: out["grounding"], out["reason"] = "WEAK", "evidence text is redacted, so the claimed value cannot be re-confirmed from source text"
    else: out["reason"] = ("supporting evidence has no source document" if nodoc and not mismatched else f"claimed {spec['currency']} {spec['amount']:g} not found in the supporting evidence text") if spec["kind"] == "amount" else "supporting evidence has no source document"
    return out

def verify_self_verification_result(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Read-only (no graph mutation); idempotent. Returns the 7A result with verification_status / escalation_reason / gaps populated and a `material_claims` list added."""
    res = build_self_verification_result(G, decision_id)
    res["material_claims"] = []
    if not res["found"]: return res
    dd = G.nodes[decision_id]; verdict = dd.get("verdict")
    if verdict not in SV_ASSERTING_VERDICTS:
        res["gaps"].append(f"verification not applicable: verdict {verdict} asserts no conclusion, so there are no material claims to ground"); return res
    res["contract_version"] = "7B"
    claims: List[Dict[str, Any]] = []
    if dd.get("violation_status") == "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE":
        scope = list(dd.get("absence_scope_evidence_ids") or [])
        ok = [e for e in scope if G.has_node(e) and G.nodes[e].get("type") == "Evidence"]
        c = {"claim_id": f"claim::{decision_id}::absence", "node_id": None, "node_type": "Decision", "kind": "absence_within_scope", "description": "required text absent within the extracted evidence scope",
             "grounding": "UNSUPPORTED", "reason": "no scope evidence recorded", "evidence": [{"evidence_id": e, "document_id": (_evidence_source_docs(G, e) or [{}])[0].get("document_id"), "filename": (_evidence_source_docs(G, e) or [{}])[0].get("filename"),
                                                                                           "location": G.nodes[e].get("source_location"), "document_ids": [x["document_id"] for x in _evidence_source_docs(G, e)], "provenance": _sv_json(G.nodes[e].get("provenance"))} for e in ok]}
        if ok and len(ok) == len(scope):
            gaps = extraction_gaps(G)
            c["grounding"], c["reason"] = ("WEAK", f"absence holds only within extracted scope; extraction incomplete for {len(gaps)} document(s)") if gaps else ("GROUNDED", "absence verified against every recorded scope evidence item")
        elif scope: c["reason"] = "some recorded scope evidence is not in the graph"
        claims.append(c)
    else:
        lin = query_decision_lineage(G, decision_id)
        ids = list(dict.fromkeys(list(dd.get("basis_node_ids") or []) + [b["node_id"] for b in lin.get("basis_nodes") or []]))
        for b in ids:
            claims.append(_sv_check_claim(G, _sv_claim_spec(G, b)) if G.has_node(b) else _sv_check_claim(G, {"node_id": b, "node_type": None, "kind": "existence", "description": f"basis node {b}"}))
        if not ids: claims.append({"claim_id": f"claim::{decision_id}::basis", "node_id": None, "node_type": None, "kind": "decision_basis", "description": f"{verdict} conclusion has no recorded basis",
                                   "grounding": "UNSUPPORTED", "reason": "no basis node recorded for the conclusion", "evidence": []})
    res["material_claims"] = claims
    bad, weak = [c for c in claims if c["grounding"] == "UNSUPPORTED"], [c for c in claims if c["grounding"] == "WEAK"]
    majors = [c for c in res["contradicting_evidence"] if c.get("category") == "MAJOR_CONTRADICTION" or c.get("source") == "opposite_verdict_basis"]
    for c in bad: res["gaps"].append(f"unsupported material claim {c['claim_id']}: {c['reason']}")
    for c in weak: res["gaps"].append(f"weakly grounded material claim {c['claim_id']}: {c['reason']}")
    if bad: res["verification_status"] = "FAILED"
    elif weak or majors:
        res["verification_status"] = "ESCALATE"
        why = ([f"{len(weak)} weakly grounded material claim(s)"] if weak else []) + ([f"{len(majors)} unresolved major contradiction / opposite-verdict evidence item(s)"] if majors else [])
        res["escalation_reason"] = "; ".join(why) + "; review required"
    else: res["verification_status"] = "VERIFIED"
    return res

def verify_self_verification_results(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    return [verify_self_verification_result(G, n) for n, _ in sorted(_nodes_of_type(G, "Decision"), key=lambda kv: kv[0])]

# --- POLICY-RULE APPLICABILITY VERIFICATION (Phase 7C) ---
# Independent, read-only check that the rule the Decision cites actually applies to the verified material facts. It REUSES verify_self_verification_result (7B), the stored Decision
# attributes (rule_id, parsed_spec, compiled_rules, rulebook_provenance, basis_node_ids) and the PolicyRule / Policy nodes; it does NOT rerun or replace the policy engine, creates no node / edge,
# never changes the verdict, invents nothing and keeps confidence None (NOT_MEASURED). Result is ADDITIVE: the eight contract fields stay; details go in `policy_applicability`.
#   checks: rule identity (EVALUATES edge == Decision.rule_id) | policy scope (Decision policies subset of the rule's owning policies) | rule text still equals the spec the Decision used |
#           required facts present (amount / currency unit / validity / phrase) | basis values meet the rule's condition for the stored verdict | compiled rule mapped, VALID, no missing facts.
#   status: FAILED (7B) stays FAILED | non-asserting verdict stays NOT_VERIFIED | applicability established -> keeps 7B status (VERIFIED) | mismatch or cannot be established -> ESCALATE.
def _pa_basis_problems(G: nx.MultiDiGraph, dd: Dict[str, Any], spec: Dict[str, Any], basis: List[str]) -> Tuple[List[str], List[str]]:
    bad: List[str] = []; miss: List[str] = []
    verdict, subj = dd.get("verdict"), spec.get("subject")
    want = (spec.get("mode") == "FORBID") == (verdict == "VIOLATION")  # True: basis must MEET the condition (FORBID+VIOLATION, REQUIRE+SATISFIED)
    if dd.get("violation_status") == "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE":
        phrase = str(spec.get("value") or "")
        if not (subj == "KEYWORD" and spec.get("mode") == "REQUIRE" and verdict == "VIOLATION" and phrase): bad.append("absence conclusion does not match a REQUIRE KEYWORD rule")
        elif any(phrase in str(G.nodes[e].get("text") or "").lower() for e in dd.get("absence_scope_evidence_ids") or [] if G.has_node(e)): bad.append(f"required phrase '{phrase}' is present in the recorded scope evidence")
        return bad, miss
    for n in basis:
        d = G.nodes[n] if G.has_node(n) else {}
        t = d.get("type")
        if subj in ("TRANSACTION", "AMOUNT_MATCH"):
            if t != "Transaction": bad.append(f"basis node {n} is {t}, not a Transaction"); continue
            if _is_policy_role(d.get("amount_role")) or d.get("amount_role") == ROLE_UNCLEAR: bad.append(f"basis node {n} amount role {d.get('amount_role')} is not an actual transaction"); continue
            amt = d.get("amount")
            if not isinstance(amt, (int, float)) or not math.isfinite(amt): miss.append(f"{n}: amount"); continue
            if subj == "TRANSACTION":
                cur = spec.get("currency")
                if cur and d.get("currency") not in _KNOWN_CURRENCIES: miss.append(f"{n}: currency unit"); continue
                if cur and d.get("currency") != cur: bad.append(f"basis node {n} currency {d.get('currency')} differs from rule currency {cur}"); continue
                if _CMP_OPS[spec["op"]](amt, spec["value"]) != want: bad.append(f"basis node {n} amount {amt:g} does not {'meet' if want else 'fail'} the rule condition {spec['op']} {spec['value']:g} required for verdict {verdict}")
        elif subj in ("GOVID", "PHONE"):
            if t != "Entity" or d.get("entity_type") != ("GovID" if subj == "GOVID" else "Phone"): bad.append(f"basis node {n} is not a {subj} entity")
            elif d.get("is_valid") is None: miss.append(f"{n}: validity result")
            elif bool(d["is_valid"]) != want: bad.append(f"basis node {n} validity {d['is_valid']} contradicts the rule for verdict {verdict}")
        elif subj == "KEYWORD":
            if t != "Evidence": bad.append(f"basis node {n} is not an Evidence node")
            elif (str(spec.get("value")) in str(d.get("text") or "").lower()) != want: bad.append(f"basis evidence {n} {'lacks' if want else 'contains'} phrase '{spec.get('value')}' contrary to verdict {verdict}")
        else: miss.append(f"rule subject {subj} cannot be re-checked")
    if subj == "AMOUNT_MATCH" and not bad and not miss:  # relationship, not mere presence: reliably linked basis pairs must differ (VIOLATION) / all agree (SATISFIED)
        pairs = [(a, b) for i, a in enumerate(basis) for b in basis[i + 1:] if G.nodes[a].get("currency") == G.nodes[b].get("currency") and _match_transactions(G.nodes[a], G.nodes[b])[0] == "strong"]
        diff = any(G.nodes[a]["amount"] != G.nodes[b]["amount"] for a, b in pairs); same = any(G.nodes[a]["amount"] == G.nodes[b]["amount"] for a, b in pairs)
        if not pairs: bad.append("no reliably linked basis pair (same reference / date + party + expense type, same currency) to compare amounts")
        elif verdict == "VIOLATION" and not diff: bad.append("linked basis amounts do not differ, contrary to verdict VIOLATION")
        elif verdict == "SATISFIED" and (diff or not same): bad.append("linked basis amounts do not all match, contrary to verdict SATISFIED")
    return bad, miss

def _pa_compiled_problems(G: nx.MultiDiGraph, dd: Dict[str, Any], rd: Dict[str, Any], basis: List[str]) -> Tuple[List[str], List[str]]:
    bad: List[str] = []; miss: List[str] = []
    ents = [e for e in dd.get("compiled_rules") or [] if e.get("executed") and e.get("verdict") not in (None, "INDETERMINATE")]
    if not ents: return bad, ["no executed compiled rule recorded on the Decision"]
    full = {e.get("rule_id"): e for e in rd.get("compiled_rules") or []}
    line = _ws_norm(redact_pii(rd.get("original_text") or rd.get("condition") or ""))
    for e in ents:
        rid, src = e.get("rule_id"), _ws_norm(e.get("source_text"))
        if e.get("status") != "VALID": bad.append(f"compiled rule {rid} status {e.get('status')} is not VALID")
        if not e.get("mapping_basis") or not src or not (src in line or line in src): bad.append(f"compiled rule {rid} is not mapped to the cited rule text")
        for m in e.get("missing_facts") or []: miss.append(f"{rid}: {m}")
        ent = str(((full.get(rid) or {}).get("compiled_rule") or {}).get("entity") or "").lower()
        types = {"Transaction"} if ent in _COMPILED_TXN_ENTITIES else {"Entity"} if any(ent in a for a in _COMPILED_ENTITY_MAP.values()) else None
        if types is None: miss.append(f"{rid}: entity {ent or 'unknown'} cannot be matched to basis nodes")
        else: bad += [f"basis node {n} is {G.nodes[n].get('type') if G.has_node(n) else None}, not {sorted(types)[0]} required by compiled rule {rid}" for n in basis if not G.has_node(n) or G.nodes[n].get("type") not in types]
    miss += [f"decision: {m}" for m in dd.get("compiled_missing_facts") or []]
    return bad, list(dict.fromkeys(miss))

def verify_policy_applicability(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Read-only; idempotent. 7B result + additive `policy_applicability` detail. Never changes the Decision verdict, the graph, or confidence."""
    res = verify_self_verification_result(G, decision_id)
    pa: Dict[str, Any] = {"status": "NOT_CHECKED", "rule": None, "checks": [], "mismatches": [], "missing_facts": [], "note": None}
    res["policy_applicability"] = pa
    if not res["found"]: pa["note"] = "decision not found"; return res
    dd = G.nodes[decision_id]
    if dd.get("verdict") not in SV_ASSERTING_VERDICTS: pa["note"] = f"verdict {dd.get('verdict')} asserts no conclusion: applicability not checked"; return res
    if res["verification_status"] == "FAILED": pa["note"] = "7B FAILED: applicability not assessed; status stays FAILED"; return res
    res["contract_version"] = "7C"
    rid = dd.get("rule_id"); rd = G.nodes[rid] if rid and G.has_node(rid) and G.nodes[rid].get("type") == "PolicyRule" else None
    pa["rule"] = {"rule_id": rid, "condition": (rd or {}).get("condition"), "source_file": (rd or {}).get("source_file"), "source_location": (rd or {}).get("source_location"),
                  "rulebook_provenance": _sv_json(dd.get("rulebook_provenance")), "parsed_spec": _sv_json(dd.get("parsed_spec")), "basis_node_ids": list(dd.get("basis_node_ids") or []), "result_source": dd.get("result_source")}
    bad, miss = pa["mismatches"], pa["missing_facts"]
    def chk(name: str, b0: int, m0: int) -> None: pa["checks"].append({"check": name, "result": "MISMATCH" if len(bad) > b0 else "MISSING" if len(miss) > m0 else "OK"})
    b0, m0 = len(bad), len(miss)
    if rd is None: bad.append(f"cited rule {rid} is not a PolicyRule in the graph")
    else:
        evaluated = [v for _, v, e in G.out_edges(decision_id, data=True) if e.get("relation") == "EVALUATES"]
        if evaluated != [rid]: bad.append(f"Decision evaluates {evaluated} but cites rule {rid}")
        if dd.get("rule_source_file") != rd.get("source_file") or dd.get("rule_source_location") != rd.get("source_location"): bad.append("Decision rule source file/location differs from the rule node")
    chk("rule_identity", b0, m0); b0, m0 = len(bad), len(miss)
    if rd is not None:
        rule_pols = {v for _, v, e in G.out_edges(rid, data=True) if e.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy"}
        dec_pols = {v for _, v, e in G.out_edges(decision_id, data=True) if e.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy"}
        if not rule_pols: miss.append("rule has no owning Policy: scope cannot be established")
        elif not dec_pols: miss.append("Decision records no Policy scope")
        elif not dec_pols <= rule_pols: bad.append(f"Decision policy scope {sorted(dec_pols - rule_pols)} is not a Policy that owns rule {rid}")
    chk("policy_scope", b0, m0); b0, m0 = len(bad), len(miss)
    basis = list(dd.get("basis_node_ids") or [])
    if rd is not None:
        if dd.get("result_source") == "compiled_policy_engine": nb, nm = _pa_compiled_problems(G, dd, rd, basis)
        else:
            spec = dd.get("parsed_spec")
            if not spec: nb, nm = [], ["no parsed rule spec recorded on the Decision"]
            elif parse_policy_rule(rd.get("condition", "")) != spec: nb, nm = ["rule text no longer parses to the spec the Decision used"], []
            else: nb, nm = _pa_basis_problems(G, dd, spec, basis)
        bad += nb; miss += nm
    chk("rule_conditions_facts_units", b0, m0)
    pa["status"] = "MISMATCH" if bad else "UNESTABLISHED" if miss else "VERIFIED"
    for x in bad: res["gaps"].append(f"policy rule mismatch: {x}")
    for x in miss: res["gaps"].append(f"policy rule fact missing: {x}")
    if pa["status"] != "VERIFIED":
        why = f"policy rule applicability {'mismatch' if bad else 'not established'}: " + "; ".join((bad + miss)[:3])
        res["verification_status"] = "ESCALATE"
        res["escalation_reason"] = f"{res['escalation_reason']}; {why}" if res.get("escalation_reason") else why
    return res

def verify_policy_applicability_results(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    return [verify_policy_applicability(G, n) for n, _ in sorted(_nodes_of_type(G, "Decision"), key=lambda kv: kv[0])]

# --- FULL-CHAIN TRACE: Document <- Evidence -> Claim/Entity/Transaction -> Rule <- Decision -> Risk ---
def trace_decision_chain(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    chain: Dict[str, Any] = {"decision_id": decision_id}
    if not G.has_node(decision_id) or G.nodes[decision_id].get("type") != "Decision": return chain
    dd = G.nodes[decision_id]
    chain.update(verdict=dd.get("verdict"), rationale=dd.get("rationale"))
    rules = [v for _, v, d in G.out_edges(decision_id, data=True) if d.get("relation") == "EVALUATES"]
    rule_id = rules[0] if rules else None
    chain["rule"] = {"node_id": rule_id, "condition": G.nodes[rule_id].get("condition")} if rule_id else None
    chain["risks"] = [{"node_id": v, "severity": G.nodes[v].get("severity")} for _, v, d in G.out_edges(decision_id, data=True) if d.get("relation") == "HAS_RISK"]
    pols = [v for _, v, d in G.out_edges(decision_id, data=True) if d.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy"]
    if rule_id:
        pols += [v for _, v, d in G.out_edges(rule_id, data=True) if d.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy" and v not in pols]
    chain["policies"] = pols
    chain["rule_source"] = ([{"evidence_id": s_["node_id"], "documents": _evidence_source_docs(G, s_["node_id"])}
                             for s_ in _in_edge_sources(G, rule_id, "SUPPORTS", "Evidence") if _is_policy_source_evidence(s_["node"])] if rule_id else [])
    nodes = []
    if rule_id:
        for rel in ("VIOLATES", "SATISFIES"):
            for src in _in_edge_sources(G, rule_id, rel):
                n = src["node"]
                evs = _supporting_evidence_ids(G, src["node_id"])
                nodes.append({"relation": rel, "node_id": src["node_id"], "type": n.get("type"),
                              "value": _entity_label(G, src["node_id"]) if n.get("entity_type") == "GovID" else n.get("value") if n.get("value") is not None else (f"{n.get('currency')} {n.get('amount')}" if n.get("type") == "Transaction" else None),
                              "evidence": [{"evidence_id": e, "documents": _evidence_source_docs(G, e)} for e in evs]})
    chain["nodes"] = nodes
    chain["supporting_evidence"] = [s["node_id"] for s in query_evidence_supporting_decision(G, decision_id)]
    return chain

def _compiled_coverage(G: nx.MultiDiGraph, rules: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Compiled-policy counts derived from the PolicyRule / Decision nodes (empty when no rulebook was compiled)."""
    ce = [e for _, rd in rules for e in (rd.get("compiled_rules") or [])]
    if not ce: return {}
    dec = {d.get("rule_id"): d for _, d in _nodes_of_type(G, "Decision")}
    return {"compiled_rules": len(ce), "compiled_authoritative": sum(1 for rid, _ in rules if (dec.get(rid) or {}).get("evaluation_engine") == "compiled_rule_engine"),
            "compiled_indeterminate": sum(1 for e in ce if (e.get("result") or {}).get("verdict") == "INDETERMINATE"),
            "compiled_needs_review_not_executed": sum(1 for e in ce if e.get("status") != "VALID")}

def policy_coverage(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Counts derived from each rule's own Decision, so they always agree with the per-rule results.
    discovered = PolicyRule nodes | recognized (key 'parsed') = text recognized/stored in the supported rule format; NOT necessarily executable
    executable = recognized AND required support/configuration available (verdict != UNEVALUATED) | attempted = evaluation actually started (== executable)
    evaluated_with_result = attempted AND verdict VIOLATION/SATISFIED | attempted_without_result = attempted AND NOT_APPLICABLE/INCONCLUSIVE
    unevaluated = discovered - attempted (UNEVALUATED: not attempted; support, configuration or unambiguous interpretation unavailable). UNEVALUATED is never attempted.
    discovered == evaluated_with_result + attempted_without_result + unevaluated."""
    rules = _nodes_of_type(G, "PolicyRule")
    verdict_by_rule = {d.get("rule_id"): d.get("verdict") for _, d in _nodes_of_type(G, "Decision")}
    discovered = len(rules)
    recognized = attempted = evaluated = 0
    verdicts: Counter = Counter()
    for rid, rd in rules:
        v = verdict_by_rule.get(rid)
        is_rec = bool(rd.get("parsed"))
        recognized += is_rec
        if v: verdicts[v] += 1
        if is_rec and v and v != "UNEVALUATED":
            attempted += 1
            if v in ("VIOLATION", "SATISFIED"): evaluated += 1
    return {**_compiled_coverage(G, rules), "total": discovered, "discovered": discovered, "parsed": recognized, "recognized": recognized, "executable": attempted, "attempted": attempted,
            "evaluated": evaluated, "evaluated_with_result": evaluated, "attempted_without_result": attempted - evaluated, "not_determined": attempted - evaluated,
            "unevaluated": discovered - attempted, "recognized_not_executable": recognized - attempted, "parsed_but_not_evaluable": recognized - attempted, "verdicts": dict(verdicts)}

def _coverage_sentence_base(c: Dict[str, Any]) -> str:
    return (f"{c['discovered']} rule(s) discovered; {c['recognized']} recognized in the supported rule format (recognized/stored only; not necessarily executable); "
            f"{c['executable']} executable (deterministic rule with required support/configuration available); {c['attempted']} attempted (evaluation started); "
            f"{c['evaluated_with_result']} evaluated with a result (SATISFIED/VIOLATION); {c['attempted_without_result']} attempted without a result (INCONCLUSIVE/NOT_APPLICABLE); "
            f"{c['unevaluated']} UNEVALUATED (not attempted: support, configuration or unambiguous interpretation unavailable; NOT checked; not counted as attempted)")

def _coverage_sentence(c: Dict[str, Any]) -> str:
    s = _coverage_sentence_base(c)
    if c.get("compiled_rules"):
        s += (f"; compiled-policy layer: {c['compiled_rules']} compiled rule(s) mapped to rulebook lines, {c['compiled_authoritative']} rule(s) decided by the deterministic compiled-rule engine "
              f"(authoritative), {c['compiled_indeterminate']} compiled rule(s) INDETERMINATE (required facts/evidence not available in the graph, or NEEDS_REVIEW), "
              f"{c['compiled_needs_review_not_executed']} NEEDS_REVIEW compiled rule(s) never executed; the legacy DSL result applies wherever the compiled result is not determinate")
    return s

def policy_coverage_note(G: nx.MultiDiGraph) -> str:
    c = policy_coverage(G)
    vtxt = ", ".join(f"{k}={v}" for k, v in sorted(c["verdicts"].items(), key=lambda kv: str(kv[0]))) or "none"
    note = f"> **Policy coverage:** {_coverage_sentence(c)}. Verdicts: {vtxt}."
    if any(d.get("relation") == "CONTRADICTS" for _, _, d in G.edges(data=True)):
        note += ("\n> **Amount discrepancies** are inferences from extracted amounts, not policy violations: strong = matched by reference ID or "
                 "date+party+expense type; weak = heuristic (POSSIBLE_DISCREPANCY). Each evidence pair is reported once. Policy-listed amounts are context only.")
    if c["evaluated"] == 0: note += "\n> **No policy rule produced a pass/violation result: policy compliance was NOT checked by the system.** Any compliance statement below is LLM interpretation."
    elif c["unevaluated"]: note += "\n> UNEVALUATED rules carry no pass/violation; each lists its reason. They must not be read as satisfied or as violated."
    if any(d.get("entity_type") == "GovID" and d.get("verification_status") in (None, "UNVERIFIED") for _, d in G.nodes(data=True)):
        note += f"\n> **Government ID:** UNVERIFIED; {govid_unvalidated_reason()}."
    note += "\n> Latency and retrieval counts are operational metrics, not evidence that V2 is more accurate than V1."
    return note + "\n\n"

def v1_system_note(G: nx.MultiDiGraph) -> str:
    """Deterministic caveats prepended to the V1 report. Computed from the graph build, NOT by V1's LLM; V1 analysis itself is unchanged."""
    c = policy_coverage(G)
    roles = Counter(d.get("amount_role", "not_classified") for _, d in _nodes_of_type(G, "Transaction"))
    roles.update(d.get("amount_role", ROLE_POLICY_CONTEXT) for _, d in _nodes_of_type(G, "Claim") if d.get("claim_type") == "PolicyListedAmount")
    out = ("> **System note (deterministic; not part of the LLM analysis below):**\n"
           f"> Policy rules: {_coverage_sentence(c)}.")
    if c["evaluated"] == 0 and c["discovered"]: out += " The system did NOT check policy compliance; rule-related statements below are LLM interpretation."
    if roles: out += ("\n> Amounts detected by role: " + ", ".join(f"{k}={v}" for k, v in sorted(roles.items())) + ". POLICY_THRESHOLD / POLICY_CONTEXT amounts are policy-listed context, not transactions, and are never tied to an employee; "
                      "UNCLEAR amounts need reconciliation before being compared. Amount differences are possible discrepancies requiring reconciliation, not confirmed contradictions, "
                      "and no policy violation is confirmed unless a rule received a deterministic VIOLATION verdict.")
    gids = _nodes_of_type(G, "Entity", "GovID")
    if gids:
        out += (f"\n> Government ID(s) present: {len(gids)} (represented as {REDACTED_GOVID}); verification: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(d.get('verification_status') or 'UNVERIFIED' for _, d in gids).items())) + ".")
        if any((d.get("verification_status") or "UNVERIFIED") == "UNVERIFIED" for _, d in gids):
            out += f" UNVERIFIED: {govid_unvalidated_reason()}. {GOVID_NEXT_STEP}"
    return out + "\n\n"

def _md_literal(t: str) -> str:
    """Inline code span: shows source text verbatim and stops renderers from auto-linking URLs inside it (the text itself is not changed)."""
    if not t: return t
    fence = "`" * (max((len(r) for r in re.findall(r'`+', t)), default=0) + 1)
    return f"{fence} {t} {fence}"

def _md_cell(v: Any, limit: int = 300, literal: bool = False) -> str:
    """Safe Markdown table cell: redacted, single-line, '|' -> '¦' (cell separator), explicit truncation notice."""
    t = GOVID_PATTERN.sub(REDACTED_GOVID, redact_pii(str(v if v is not None else "")))
    t = re.sub(r'\s+', ' ', t.replace("|", "¦")).strip()
    t = t if len(t) <= limit else t[:limit].rstrip() + f" …[truncated; {len(t)} chars]"
    return _md_literal(t) if literal else t

def v1_audit_section(entries: List[Dict[str, Any]]) -> str:
    """Deterministic V1 'Evidence & Audit Details': source records exactly as extracted (no values added or inferred)."""
    out = "## 4. Evidence & Audit Details\n\n*Deterministic system output: source records as extracted; no LLM interpretation. Source excerpts are shown verbatim in code format: any URL inside an excerpt is source text and is NOT a supporting link (a stray closing bracket or percent-encoded delimiter glued to the end of a URL is removed for display only; the source file is unchanged); the normalized, de-duplicated supporting links are listed once in the Audit Appendix.*\n\n"
    if not entries: return out + "No amount-bearing source records were extracted.\n"
    ordered = [e for e in entries if not e["context_only"]] + [e for e in entries if e["context_only"]]
    out += ("Source records as extracted (location = sheet and worksheet row for spreadsheets; `¦` separates cells; nothing added or inferred):\n\n"
            "| # | Source file | Location | Amount (role) | Source excerpt |\n|---|---|---|---|---|\n")
    for i, e in enumerate(ordered[:15], 1):
        ex = re.sub(r'^Row \d+:\s*', '', e.get("excerpt", ""))
        tag = "; policy-listed context, not a transaction" if e["context_only"] else ""
        out += f"| {i} | {_md_cell(e['file'], 120)} | {_md_cell(e['location'], 80)} | {_md_cell(str(e['currency']) + ' ' + format(e['amount'], 'g') + ' (' + str(e['role']) + tag + ')', 160)} | {_md_cell(_strip_url_artifacts(ex), 500, literal=True)} |\n"
    if len(ordered) > 15: out += f"\n{len(ordered) - 15} further amount record(s) not shown.\n"
    return out

def v2_audit_section(G: nx.MultiDiGraph) -> str:
    """Deterministic V2 'Evidence & Audit Details': every discrepancy lists BOTH evidence IDs, each mapped to its own source and amount."""
    out = "## 4. Evidence & Audit Details\n\n*Deterministic system output; amount pairings below are POSSIBLE discrepancies (heuristic or attribute-matched, always unconfirmed), not verified contradictions, and they never create a policy violation by themselves.*\n\n"
    items = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == "CONTRADICTS"]
    if not items: return out + "No record-vs-record amount discrepancy was detected in the extracted data.\n"
    items.sort(key=lambda x: (0 if x[2].get("match_strength") == "strong" else 1, str(x[2].get("evidence_pair") or x[0])))
    def src(e): return ", ".join(f"{x['filename']} @ {x['location']}" for x in _evidence_source_docs(G, e)) or "no source link"
    def amt(a, t): return f"{G.nodes.get(t, {}).get('currency') or ''} {a:g}".strip() if isinstance(a, (int, float)) else f"{a}"
    out += ("Amount differences between records (each pair listed once; both evidence IDs sit together in the Evidence Pair cell):\n\n"
            "| Evidence Pair | Discrepancy Details | Status |\n|---|---|---|\n")
    for u, v, d in items[:20]:
        b_id = d.get("counterpart_evidence_id") or (_supporting_evidence_ids(G, v) or ["?"])[0]
        a_t, b_t = d.get("source_transaction_id") or "", d.get("target_transaction_id") or v
        a_amt, b_amt = amt(d.get("this_amount"), a_t), amt(d.get("other_amount"), b_t)
        a_src, b_src = src(u), src(b_id)
        diff = ""
        if isinstance(d.get("this_amount"), (int, float)) and isinstance(d.get("other_amount"), (int, float)) and G.nodes.get(a_t, {}).get("currency") == G.nodes.get(b_t, {}).get("currency"):
            diff = f" (difference {abs(d['this_amount'] - d['other_amount']):g})"
        status = ("POSSIBLE DISCREPANCY (weak match); records may describe different transactions; requires reconciliation; not a confirmed contradiction or violation"
                  if d.get("match_strength", "weak") != "strong" else "DISCREPANCY ON MATCHED TRANSACTION; needs review; not a policy violation")
        out += (f"| A: {_md_cell(u, 60)} ({_md_cell(a_src, 160)}) / B: {_md_cell(b_id, 60)} ({_md_cell(b_src, 160)}) "
                f"| {_md_cell(a_amt, 40)} in A vs {_md_cell(b_amt, 40)} in B{diff}; matched: {_md_cell(', '.join(map(str, d.get('matched_attributes', []))) or 'none', 120)}; "
                f"conflicting: {_md_cell(', '.join(map(str, d.get('conflicting_attributes', []))) or 'none', 80)} | {status} |\n")
    return out

def _coverage_short(c: Dict[str, Any]) -> str:
    return (f"{c['discovered']} policy rule(s) discovered, {c['executable']} executable/attempted, {c['evaluated_with_result']} evaluated with a result, "
            f"{c['attempted_without_result']} attempted without a result, {c['unevaluated']} UNEVALUATED (not checked)")

REPORT_BASIS_LEGEND = ("> **Report basis:** sections 1-3 are LLM-generated interpretation of the extracted data; the Assessment limits, Evidence & Audit Details and Audit Appendix "
                       "are deterministic system output; amount pairings and shared-term links are heuristics and are labelled as such.")

def report_key_limits(G: nx.MultiDiGraph, link_log: Optional[List[Dict[str, Any]]] = None) -> str:
    """ONE short banner stating the general limits once; the rest of the report refers to it instead of repeating them."""
    c = policy_coverage(G)
    parts: List[str] = []
    if c["discovered"]:
        parts.append(_coverage_short(c) + ("; policy compliance was NOT checked by the system" if c["evaluated_with_result"] == 0 else "; UNEVALUATED rules carry no pass/violation"))
    if any(d.get("relation") == "CONTRADICTS" for _, _, d in G.edges(data=True)) or len(_nodes_of_type(G, "Transaction")) > 1:
        parts.append("amount differences are possible discrepancies pending reconciliation, not confirmed contradictions or violations")
    if any(d.get("entity_type") == "GovID" and d.get("verification_status") in (None, "UNVERIFIED") for _, d in G.nodes(data=True)):
        parts.append("Government ID(s) UNVERIFIED (validation not performed; no authenticity claim)")
    _links = normalize_link_log(link_log)
    nb = sum(1 for l in _links if not l.get("retrieved"))
    if nb: parts.append(f"{nb} of {len(_links)} unique supporting link(s) not retrieved ({link_status_summary(_links).split(': ', 1)[1]}); the content of those links was not inspected")
    legend = (REPORT_BASIS_LEGEND + ("\n>\n" if parts else "\n\n"))
    return legend + (("> **Assessment limits** (full definitions and explanations: see Audit Appendix): " + "; ".join(parts) + ".\n\n") if parts else "")

def report_appendix_details(G: nx.MultiDiGraph, latency: float, link_log: Optional[List[Dict[str, Any]]] = None) -> str:
    """Deterministic supporting details moved out of the main report: policy rule processing, amount roles, Government ID, links, metrics."""
    c = policy_coverage(G)
    out = ("### Policy rule processing\n\n" + _coverage_sentence(c) + ".\n\n"
           "Definitions: discovered = rules identified in the policy; recognized = rule text mapped to the supported rule format (stored only; not necessarily executable); "
           "executable = deterministic rule whose required support/configuration is available; attempted = evaluation actually started (UNEVALUATED is never attempted); "
           "evaluated with a result = SATISFIED or VIOLATION; attempted without a result = INCONCLUSIVE or NOT_APPLICABLE; UNEVALUATED = not attempted because support, configuration or unambiguous interpretation was unavailable. "
           "INCONCLUSIVE and UNEVALUATED are neither passed nor failed: no pass or violation is implied, and no result is inferred for them.\n\n")
    rows = policy_rule_report(G)
    if rows:
        out += "| Rule | Source | Recognized | Executable | Attempted | Verdict | Engine | Compiled verdict | Legacy verdict | Reason |\n|---|---|---|---|---|---|---|---|---|---|\n"
        for r in rows[:30]:
            src = f"{r.get('source_file') or ''} @ {r.get('source_location') or r.get('source_line') or 'n/a'}"
            out += (f"| {_md_cell(r['condition'], 140)} | {_md_cell(src, 100)} | {'yes' if r['recognized'] else 'no'} | {'yes' if r['executable'] else 'no'} | "
                    f"{'yes' if r['attempted'] else 'no'} | {r['verdict']} | {r.get('evaluation_engine') or ''} | {r.get('compiled_verdict') or 'none'} | {r.get('legacy_verdict') or ''} | {_md_cell(r.get('unevaluated_reason') or r.get('rationale'), 240)} |\n")
        out += "\n"
    roles = Counter(d.get("amount_role", "not_classified") for _, d in _nodes_of_type(G, "Transaction"))
    roles.update(d.get("amount_role", ROLE_POLICY_CONTEXT) for _, d in _nodes_of_type(G, "Claim") if d.get("claim_type") == "PolicyListedAmount")
    if roles: out += "Amounts by role: " + ", ".join(f"{k}={v}" for k, v in sorted(roles.items())) + ". Policy-listed amounts are context, not transactions, and are not tied to any employee.\n\n"
    gids = _nodes_of_type(G, "Entity", "GovID")
    if gids:
        out += (f"Government ID(s): {len(gids)} present (shown only as {REDACTED_GOVID}); verification: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(d.get('verification_status') or 'UNVERIFIED' for _, d in gids).items())) + ".")
        if any((d.get("verification_status") or "UNVERIFIED") == "UNVERIFIED" for _, d in gids): out += f" {govid_unvalidated_reason().capitalize()}. {GOVID_NEXT_STEP}"
        _gsrc = _govid_source_lines(G)
        out += " ID-like value detected only in: " + ("; ".join(_md_cell(x, 160) for x in _gsrc) if _gsrc else "no source location recorded") + " (no other document is implied)."
        out += "\n\n"
    links = normalize_link_log(link_log)  # the SAME normalized records feed V1 and V2 (and the V1 brief, V2 context and Assessment limits)
    if links:
        out += (f"Supporting links ({link_status_summary(links)}; each unique link listed once; V1 and V2 use the same records; query parameters and credentials are never shown; "
                "a clickable link is not evidence that the destination was retrieved or verified):\n" + "".join(_link_status_line(l, md=True) for l in links[:20]))
        if len(links) > 20: out += f"- {len(links) - 20} further unique link(s) not listed.\n"
        out += "\n" + LINK_VS_FINDING_NOTE + "\n\n"
    out += f"Operational: latency {latency:.2f}s (a metric, not evidence of accuracy; V1 and V2 are not ranked by it).\n"
    return out

# LLM-written text can echo malformed destinations copied from source excerpts (e.g. 'https://github.com/o/r%5D') both as bare URLs AND inside
# its own Markdown links. Every http(s) destination in the LLM sections is therefore re-cleaned and re-rendered through the same cleaning /
# canonical form / label code as the system link lists, and list items that merely repeat an already-listed link are dropped. Source excerpts in
# the deterministic audit tables are NOT touched (evidence text stays faithful). Output format = Markdown string stored in the DB / task result;
# the client renderer is outside this file, so blue/clickable rendering is NOT verifiable here (valid link data is also returned as "supporting_links").
_URL_DEST = r'(?:[^()\s]|\([^()\s]*\))*'
_REPORT_LINK_TOKEN = re.compile(r'\[([^\]\n]*)\]\((https?://' + _URL_DEST + r')\)|(<(https?://[^>\s]+)>)|(https?://[^\s<>"\'`|]+)')
_LINK_LIST_ITEM = re.compile(r'^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+')
_MD_LINK_ANY = re.compile(r'\[[^\]\n]*\]\((https?://' + _URL_DEST + r')\)')

def _linkify_report_text(text: str) -> str:
    if not text: return text
    def _sub(m):
        label = None
        if m.group(2) is not None:
            raw, label = m.group(2), m.group(1)
            if re.match(r'(?i)^\s*https?://', label): label = None  # label that is itself a (possibly malformed) URL is regenerated
        else: raw = m.group(4) or m.group(5)
        cleaned = _clean_extracted_url(raw)
        if not re.match(r'(?i)^https?://[^/\s?#]+', cleaned): return m.group(0)
        shown, _ = _display_url(cleaned)  # credentials / query values are never put into a report link
        href = _link_href(shown)
        if not href: return m.group(0)
        rec = {"url": shown}
        link = f"[{label if label is not None else _link_label(rec)}]({href})"
        if m.group(5):
            rest = raw[len(cleaned):] if raw.startswith(cleaned) else ""
            link += "".join(ch for ch in rest if ch in ".,;:!?)")  # keep surrounding sentence punctuation; stray delimiters are dropped
        return link
    def _plain(c: str) -> str:
        parts = re.split(r'(`[^`\n]+`)', c)  # inline code: left verbatim unless it holds nothing but a URL (then it becomes the normal link)
        out_ = []
        for j, p in enumerate(parts):
            if j % 2 and re.match(r'(?i)^`\s*https?://\S+\s*`$', p): out_.append(_REPORT_LINK_TOKEN.sub(_sub, p.strip("` ")))
            elif j % 2: out_.append(p)
            else: out_.append(_REPORT_LINK_TOKEN.sub(_sub, p))
        return "".join(out_)
    chunks = re.split(r'(```.*?```)', text, flags=re.S)
    return "".join(c if i % 2 else _plain(c) for i, c in enumerate(chunks))

def _drop_repeated_link_items(text: str, known_keys: Optional[set] = None) -> str:
    """Removes LLM list items that only repeat a link (same canonical key) already listed earlier or in the system link list; items with
    substantive commentary are kept."""
    if not text: return text
    seen = set(known_keys or ())
    out: List[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"): in_fence = not in_fence
        if not in_fence and _LINK_LIST_ITEM.match(line):
            urls = [m_.group(1) for m_ in _MD_LINK_ANY.finditer(line)]
            keys = {canonical_url_key(u) for u in urls}
            if len(keys) == 1:
                k = next(iter(keys))
                remainder = _LINK_LIST_ITEM.sub("", _MD_LINK_ANY.sub("", line)).strip()
                if k in seen and len(remainder) < 160: continue
                seen.add(k)
        out.append(line)
    return "\n".join(out)

_APPENDIX_RE = re.compile(r'(?mi)^#{1,6}\s*(?:\*\*)?\s*Appendix\b')
def assemble_report(llm_text: str, audit_section: str, appendix_extra: str, banner: str = "", link_keys: Optional[set] = None) -> str:
    """banner + LLM sections 1-3 + deterministic 'Evidence & Audit Details' + Appendix (LLM appendix, if any, then deterministic details).
    The report is Markdown text: links are Markdown links; the LLM's own bare URLs are converted to the same live links."""
    llm_text = _drop_repeated_link_items(_linkify_report_text((llm_text or "").strip()), link_keys)  # destinations cleaned; one link list (system appendix) per report
    m = _APPENDIX_RE.search(llm_text)
    if m:
        return banner + llm_text[:m.start()].rstrip() + "\n\n" + audit_section + "\n\n" + llm_text[m.start():].strip() + "\n\n" + appendix_extra
    return banner + llm_text + "\n\n" + audit_section + "\n\n## Audit Appendix (optional): Supporting Details\n\n" + appendix_extra

def policy_rule_report(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """One row per rulebook line: text, parsed?, verdict, rationale. Used for diagnostics / metrics."""
    dec_by_rule = {d.get("rule_id"): d for _, d in _nodes_of_type(G, "Decision")}
    rows = []
    for rid, rd in _nodes_of_type(G, "PolicyRule"):
        dd = dec_by_rule.get(rid, {})
        rows.append({"rule_id": rid, "condition": redact_pii(rd.get("condition", "")), "recognized": bool(rd.get("parsed")), "parsed": bool(rd.get("parsed")), "executable": bool(rd.get("executable")), "attempted": bool(rd.get("attempted")),
                     "verdict": dd.get("verdict"), "rationale": dd.get("rationale"),
                     "source_file": rd.get("source_file"), "source_location": rd.get("source_location"), "source_line": rd.get("source_line"),
                     "evidence_used": dd.get("evidence_used", []), "unevaluated_reason": dd.get("unevaluated_reason"),
                     "evaluated": dd.get("verdict") in ("VIOLATION", "SATISFIED") and bool(rd.get("parsed")),
                     **{k: dd.get(k) for k in _COMPILED_REPORT_KEYS}})
    return rows

def _compiled_policy_context(G: nx.MultiDiGraph) -> str:
    cp = next((d.get("compiled_policy") for _, d in _nodes_of_type(G, "Policy") if d.get("compiled_policy")), None)
    if not cp: return ""
    rb, inp = cp.get("rulebook") or {}, cp.get("inputs") or {}
    out = ("--- COMPILED POLICY [rulebook text interpreted by an LLM into structured rules; ONLY the deterministic rule engine executes them; NEEDS_REVIEW rules are never executed] ---\n"
           f"Status: {cp.get('status')}; stats: {cp.get('stats')}; rulebook: {rb.get('filename')} (sha256 {rb.get('sha256')}).\n")
    if cp.get("error"): out += f"Compilation did not complete: {cp['error']}. Legacy DSL results apply.\n"
    if inp: out += (f"Facts derived from the graph: entities {inp.get('fact_entities')}, records {inp.get('records')}; evidence: {inp.get('evidence_note')}; evaluation date: {inp.get('evaluation_date')} ({inp.get('evaluation_date_source')}); "
                    f"fx rates: {inp.get('fx_rates')}; strict units: {inp.get('strict_units')}.\n")
    if cp.get("evaluation_error"): out += f"Compiled evaluation error: {cp['evaluation_error']}.\n"
    if cp.get("needs_review_not_executed"): out += f"NEEDS_REVIEW compiled rules NOT executed (INDETERMINATE): {cp['needs_review_not_executed']}.\n"
    if cp.get("unmapped_rules"): out += "Compiled rules not matched to any PolicyRule node (informational only, not authoritative): " + "; ".join(f"{u['rule_id']} {u.get('verdict')}" for u in cp["unmapped_rules"][:10]) + ".\n"
    if cp.get("ambiguity_reasons"): out += "Ambiguities: " + " | ".join(str(a_) for a_ in cp["ambiguity_reasons"][:5]) + "\n"
    out += (f"Rejected rules: {len(cp.get('rejected') or [])}; unparsed statements: {len(cp.get('unparsed_statements') or [])}.\n"
            "Compiled verdicts: COMPLIANT / VIOLATION / EXEMPT / ACTION_REQUIRED / NOT_APPLICABLE are deterministic results; INDETERMINATE = required facts/evidence/units unavailable or rule needs review (neither pass nor fail). "
            "A Decision names its engine: compiled_rule_engine (authoritative) or legacy_dsl; the other engine's result is shown separately and never merged.\n\n")
    return out

def _compiled_context_lines(dd: Dict[str, Any]) -> str:
    if not dd.get("evaluation_engine"): return ""
    out = (f"  Evaluation engine (authoritative): {dd.get('evaluation_engine')}; compiled-rule verdict: {dd.get('compiled_verdict') or 'none (no compiled rule mapped to this rule line)'}; "
           f"legacy DSL verdict: {dd.get('legacy_verdict')}\n")
    for cr in dd.get("compiled_rules") or []:
        out += (f"  Compiled rule {cr.get('rule_id')} [{cr.get('status')}, {cr.get('rule_type')}, confidence {cr.get('confidence')}]: {cr.get('expression')} => {cr.get('verdict')}"
                + ("" if cr.get("executed") else " (NOT executed)") + (f"; missing facts: {cr['missing_facts']}" if cr.get("missing_facts") else "")
                + (f"; {'; '.join(cr['reasons'][:2])}" if cr.get("reasons") else "") + f"\n    Rule text: {cr.get('source_text')!r} (span {cr.get('source_span')} of the compiler input)\n")
    if dd.get("legacy_disagrees"): out += "  NOTE: the legacy DSL verdict differs from the compiled result; the compiled result is authoritative.\n"
    if dd.get("compiled_verdict_downgraded"): out += "  NOTE: the compiled verdict was downgraded to INCONCLUSIVE because no qualifying evidence-backed basis node exists.\n"
    if dd.get("compiled_rules"):
        out += f"  Supporting evidence: {dd.get('supporting_evidence_ids') or 'none'}; contradicting evidence: {dd.get('contradicting_evidence_ids') or 'none'}\n"
    prov = dd.get("rulebook_provenance") or {}
    out += f"  Rulebook provenance: {prov.get('file')} @ {prov.get('location') or prov.get('line')} (sha256 {prov.get('rulebook_sha256')})\n"
    return out

def format_policy_results_for_context(G: nx.MultiDiGraph, max_decisions: int = 30, max_nodes: int = 5) -> str:
    decisions = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "Decision"]
    if not decisions: return ""
    cov = policy_coverage(G)
    out = (f"--- POLICY EVALUATION RESULTS [DETERMINISTIC; not LLM interpretation] (UNEVALUATED = no evaluation performed: unsupported/ambiguous rule or required configuration missing, NOT checked; INCONCLUSIVE = attempted, evidence insufficient) ---\n"
           f"Coverage: {_coverage_sentence(cov)}.\n")
    out += _compiled_policy_context(G)
    order = {"VIOLATION": 0, "SATISFIED": 1, "NOT_APPLICABLE": 2, "INCONCLUSIVE": 3, "UNEVALUATED": 4}
    decisions.sort(key=lambda x: order.get(x[1].get("verdict"), 9))
    for dec_id, dd in decisions[:max_decisions]:
        ch = trace_decision_chain(G, dec_id)
        rule = ch.get("rule") or {}
        sev = ",".join(r["severity"] for r in ch.get("risks", []))
        out += f"Decision {dec_id} [{dd.get('verdict')}] Rule {rule.get('node_id')}: {rule.get('condition')}\n  Rationale: {dd.get('rationale')}\n"
        if sev: out += f"  Risk severity: {sev}\n"
        _rn = G.nodes.get(rule.get("node_id"), {})
        out += f"  Rule status: recognized={'yes' if _rn.get('parsed') else 'no'}; executable={'yes' if _rn.get('executable') else 'no'}; attempted={'yes' if _rn.get('attempted') else 'no'}; result={'yes' if dd.get('verdict') in ('VIOLATION', 'SATISFIED') else 'no'}\n"
        out += _compiled_context_lines(dd)
        if dd.get("unevaluated_reason"): out += f"  NOT EVALUATED, no compliance conclusion. Reason: {dd.get('unevaluated_reason')}\n"
        if dd.get("rule_source_location"): out += f"  Rule source: {dd.get('rule_source_file')} @ {dd.get('rule_source_location')}\n"
        if ch.get("rule_source"):
            out += "  Rule definition lineage: Decision -EVALUATES-> PolicyRule" + (f" -BELONGS_TO-> Policy {', '.join(ch.get('policies', []))}" if ch.get("policies") else "") + " <-SUPPORTS- " + "; ".join(
                f"Evidence {r_['evidence_id']} -DERIVED_FROM-> " + (", ".join(f"{x['filename']} ({x['location']})" for x in r_["documents"]) or "NO SOURCE DOCUMENT") for r_ in ch["rule_source"][:2]) + "\n"
        if dd.get("violation_status"): out += f"  Violation status: {dd.get('violation_status')} (heuristic contradictions / shared terms were not used)\n"
        if dd.get("evidence_used"): out += f"  Evidence used: {dd.get('evidence_used')}\n"
        elif dd.get("verdict") in ("NOT_APPLICABLE", "INCONCLUSIVE"): out += "  Evidence used: none (no determination; absence of extracted text is not proof of absence)\n"
        for item in ch.get("nodes", [])[:max_nodes]:
            ev_txt = "; ".join(f"{e['evidence_id']} <- " + ", ".join(f"{x['filename']} ({x['location']})" for x in e["documents"]) for e in item["evidence"][:2]) or "no evidence link"
            out += f"  {item['relation']}: {item['type']} {item['node_id']} {item['value'] if item['value'] is not None else ''} | {ev_txt}\n"
    out += "\n"
    return out

def format_contradictions_for_context(G: nx.MultiDiGraph, max_items: int = 20) -> str:
    items = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == "CONTRADICTS"]
    if not items: return ""
    items.sort(key=lambda x: (0 if x[2].get("match_strength") == "strong" else 1, str(x[2].get("evidence_pair") or x[0])))
    out = ("--- AMOUNT DISCREPANCIES (inferences from extracted amounts, NOT policy violations; each evidence pair is listed ONCE. match_strength=strong: same reference ID, or same "
           "date + vendor/person + expense type. match_strength=weak: heuristic label/shared-name link, status POSSIBLE_DISCREPANCY, not proven; always different documents; "
           "record-vs-record comparison: Transaction nodes are derived from the records and are not independent evidence) ---\n")
    def srcs(e): return ", ".join(f"{x['filename']} @ {x['location']}" for x in _evidence_source_docs(G, e)) or "no source link"
    for u, v, d in items[:max_items]:
        other_ev = d.get("counterpart_evidence_id") or (_supporting_evidence_ids(G, v) or ["?"])[0]
        out += (f"Discrepancy {d.get('evidence_pair') or f'{u}|{other_ev}'}: Evidence {u} [{srcs(u)}] amount {d.get('this_amount')} vs Evidence {other_ev} [{srcs(other_ev)}] amount {d.get('other_amount')}; "
                f"method={d.get('method')}, linked via '{d.get('label')}'\n"
                f"  label={d.get('contradiction_label', 'POSSIBLE CONTRADICTION (heuristic; not established)')}; confirmed={d.get('confirmed', False)}; can_create_violation=False\n"
                f"  status={d.get('status', 'POSSIBLE_DISCREPANCY')}, match_strength={d.get('match_strength', 'weak')}, amount_roles={d.get('amount_roles', 'not_classified')}\n"
                f"  Matched Attributes: {d.get('matched_attributes', [])} values={d.get('matched_values', {})}; Conflicting Attributes: {d.get('conflicting_attributes', [])}\n"
                f"  basis: {d.get('comparison_basis', 'record_vs_record')}\n  reason: {d.get('link_reason', 'same label/shared term, different amount (heuristic)')}\n"
                + (f"  UNCERTAIN (requires reconciliation): {d.get('uncertainty_reason')}\n" if d.get("uncertainty_reason") else ""))
    return out + "\n"

def build_v2_context(G: nx.MultiDiGraph, objective: str, link_log: Optional[List[Dict[str, Any]]] = None, cross_document: Optional[bool] = None) -> Tuple[str, Dict[str, Any]]:
    """V2 payload: [LEXICAL] BM25 hits, [GRAPH-EXPANDED] evidence reached by edges, the traversal trace,
    policy decisions/risks, heuristic contradictions, rule context."""
    evidence_items = _investigation_evidence(G)
    bm25 = bm25_scores(objective, [data.get("text", "") for _, data in evidence_items])
    scored_nodes = [(s, n, data) for s, (n, data) in zip(bm25, evidence_items) if s > 0.0]
    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    ctx = (f"--- LEXICAL RETRIEVAL PAYLOAD (Objective: {objective}) ---\n"
           "(Items tagged [LEXICAL] matched the objective by BM25 word overlap; score = relative rank, not confidence. "
           "Items tagged [GRAPH-EXPANDED] were NOT lexical matches: they were reached by following graph edges. Evidence classes are kept separate: "
           "[LEXICAL] direct word overlap; [GRAPH-EXPANDED] context via a graph path of 1..N hops over incoming/outgoing edges (strong = exact entity/ID/policy-lineage link, heuristic = shared phrase/label or a heuristic edge on the path); "
           "heuristic associations/discrepancies; deterministic policy results. A graph association alone never establishes a violation.)\n")
    _warn = extraction_gaps(G)
    if _warn:
        ctx += "--- EXTRACTION WARNINGS (not evidence; absence of text from these documents proves nothing) ---\n"
        for _d in _warn: ctx += f"Document {_d} ({G.nodes[_d].get('filename')}): extraction_status={G.nodes[_d].get('extraction_status', 'no_evidence_extracted')} {G.nodes[_d].get('extraction_error', '')}\n"
        ctx += "\n"
    _lex = {n: sc for sc, n, _ in scored_nodes}

    trace_info: Dict[str, Any] = {"inspected": 0, "edges": [], "all_listed": True, "expanded": 0, "limit_reached": False, "seeds": 0, "nodes_reached": 0, "outgoing_inspected": 0, "incoming_inspected": 0, "max_hops": V2_MAX_HOPS}
    retrieved_evidence = 0
    top_hits = scored_nodes[:V2_TOP_K]
    for score, n, data in top_hits:
        ctx += _describe_evidence(G, n, data, f"[LEXICAL] Evidence Node {n} (BM25 Score: {score:.2f})", "LEXICAL (BM25 match to objective)")
        retrieved_evidence += 1

    expanded: List[Dict[str, Any]] = []
    reached_nodes: List[Dict[str, Any]] = []
    expansion_trace: List[Dict[str, Any]] = []
    seeds = [n for _, n, _ in top_hits]
    if retrieved_evidence == 0:
        ctx += "No relevant graph evidence.\n\n"
    else:
        trav = traverse_graph_context(G, seeds, V2_EXPANSION_LIMIT)
        expanded, reached_nodes, tst = trav["evidence"], trav["nodes"], trav["stats"]
        examined = trav["examined"]
        inspected = len(examined)
        MAX_TRACE_EDGES_SHOWN = 60
        _pk = trav["path_edge_keys"]
        _ordered = sorted(examined, key=lambda e_: 0 if (e_["from"], e_["relation"], e_["to"]) in _pk else 1)  # stable: same edges; edges on a returned path come first
        _shown = _ordered[:MAX_TRACE_EDGES_SHOWN]
        _edge_no: Dict[Tuple[str, str, str], int] = {}
        for _i, _e in enumerate(_shown, 1): _edge_no.setdefault((_e["from"], _e["relation"], _e["to"]), _i)
        def _path_nos(path_):
            shown_ = [_edge_no[(e_["from"], e_["relation"], e_["to"])] for e_ in path_ if (e_["from"], e_["relation"], e_["to"]) in _edge_no]
            return shown_, len(path_) - len(shown_)
        def _fmt_edge(e_):
            tag = f" [traversed {e_['direction']} from {e_['via']}, hop {e_['hop']}" + ("; heuristic" if e_["heuristic"] else "") + ("; followed" if e_["followed"] else f"; examined, not followed: {e_['skip_reason']}") + "]"
            return f"{e_['from']} --{e_['relation']}--> {e_['to']} ({G.nodes[e_['to']].get('type')}){tag}"
        ctx += f"--- GRAPH-EXPANDED EVIDENCE (NOT lexical matches; reached by following graph edges in BOTH directions, up to {tst['max_hops']} hop(s), from a lexical hit) ---\n"
        if expanded:
            for item in expanded:
                kind = "HEURISTIC link" if item["heuristic"] else "deterministic exact-match link"
                has_lex = _lex.get(item["evidence_id"], 0.0) > 0.0
                role = ("POLICY CONTEXT ONLY: policy-listed information, never corroboration of or contradiction to a transaction" if item.get("policy_context")
                        else "lexical overlap with objective exists but ranked below top-K" if has_lex
                        else "CONTEXT ONLY: no lexical match to the objective, does not by itself support any claim")
                item["supports_objective_lexically"] = has_lex
                _nos, _hidden = _path_nos(item["path"])
                expansion_trace.append({**{k: item[k] for k in ("evidence_id", "seed_id", "via_node_id", "via_type", "via_label", "relations", "match_strength", "link_method", "limitation", "policy_context", "supports_objective_lexically", "hops")},
                                        "path_edge_nos": _nos, "path_edges_not_displayed": _hidden})
                ctx += _describe_evidence(G, item["evidence_id"], G.nodes[item["evidence_id"]],
                                          f"[GRAPH-EXPANDED] Evidence Node {item['evidence_id']} (seed {item['seed_id']} via {item['via_type']} {item['via_node_id']} '{item['via_label']}'; {item['hops']}-hop path; {kind}; match={item['match_strength']}; link_method={item.get('link_method')}; {role}; LIMITATION: {item['limitation']})",
                                          f"GRAPH-EXPANDED ({kind}; {'has lexical overlap' if has_lex else 'context only'})")
        else:
            ctx += "None: no graph path (incoming or outgoing, within the hop limit) leads from the lexical hits to additional evidence.\n\n"
        if reached_nodes:
            ctx += ("--- GRAPH-REACHED NODES (Decision / Risk / PolicyRule / Policy / Transaction / Document reached by traversal; CONTEXT ONLY: reaching a node does not by itself establish "
                    "a violation; a Decision verdict is a deterministic policy result reported separately) ---\n")
            for nd_ in reached_nodes:
                _nos, _hidden = _path_nos(nd_["path"])
                ctx += (f"{nd_['detail']} | seed {nd_['seed_id']} | {nd_['hops']} hop(s): " + " ; ".join(nd_["relations"]) + (" [path includes a heuristic link]" if nd_["heuristic"] else "")
                        + f" | path edges displayed as {_nos}" + (f"; {_hidden} path edge(s) inspected but not displayed" if _hidden else "") + "\n")
            ctx += "\n"
        ctx += (f"--- GRAPH TRAVERSAL TRACE ---\nSeeds (lexical hits): {seeds}\n"
                f"Traversal: breadth-first from the seeds over edges in BOTH directions (outgoing and incoming), up to {tst['max_hops']} hop(s); edge budget {tst['edge_budget']}.\n"
                f"Edges examined: {inspected} (outgoing {tst['outgoing_examined']}, incoming {tst['incoming_examined']}); followed: {tst['edges_followed']}; nodes expanded: {tst['nodes_expanded']}; "
                f"nodes reached: {tst['nodes_reached']}; deepest hop reached: {tst['max_hop_reached']}.\n")
        if tst["hubs_not_expanded"]: ctx += f"Hub nodes reached but NOT expanded (Policy nodes or degree above {V2_TRAVERSAL_HUB_DEGREE}): {[(h_['node_id'], h_['type'], h_['degree']) for h_ in tst['hubs_not_expanded']][:20]}\n"
        if tst["frontier_not_expanded"]: ctx += f"{tst['frontier_not_expanded']} node(s) at the hop limit were reached but their edges were NOT examined.\n"
        if tst["budget_exhausted"]: ctx += "EDGE BUDGET EXHAUSTED: the traversal stopped early; further edges exist that were NOT examined.\n"
        ctx += "Documents are reached, but Evidence that merely shares a source Document is not expanded (unless the Document was reached through a LINKED_FROM hop).\n"
        if inspected:
            shown_n = len(_shown)
            ctx += (f"Complete trace: all {inspected} of {inspected} examined edges are displayed (edges examined = {inspected}; displayed = {inspected}; omitted = 0):\n" if inspected <= MAX_TRACE_EDGES_SHOWN else
                    f"Sample trace: {shown_n} of {inspected} edges shown (edges examined = {inspected}; displayed = {shown_n}; omitted = {inspected - shown_n}); the other {inspected - shown_n} edges were examined but are not displayed:\n")
            for i, e_ in enumerate(_shown, 1): ctx += f"  ({i}) {_fmt_edge(e_)}\n"
            if inspected > shown_n: ctx += "Selection basis (sample): edges on a path to returned evidence/nodes are listed first, then the remaining examined edges in traversal order up to the display cap; not a random or representative sample. Any edge list you write must be labelled 'Sample trace (X of N edges shown)' or 'Complete trace' as above.\n"
        ctx += (f"Paths that led to additional (graph-expanded) evidence: {len(expanded)} (expansion limit {V2_EXPANSION_LIMIT}" +
                ("; limit reached, further evidence may exist" if (len(expanded) >= V2_EXPANSION_LIMIT or tst["evidence_limit_reached"]) else "") + f"); non-evidence nodes reached and listed: {len(reached_nodes)}.\n")
        if expanded:
            for i, item in enumerate(expanded, 1):
                docs = ", ".join(f"{x['filename']} @ {x['location']}" for x in _evidence_source_docs(G, item["evidence_id"]))
                _nos, _hidden = _path_nos(item["path"])
                ctx += (f"[{i}] " + "  ;  ".join(item["relations"]) + f"  => retrieved {item['evidence_id']} [{docs}] (graph-expanded CONTEXT; {item['hops']} hop(s); path edges displayed as {_nos}"
                        + (f"; {_hidden} path edge(s) examined but not displayed" if _hidden else "") + f"; {item['limitation']})\n")
        else:
            ctx += "No traversal produced new evidence.\n"
        trace_info.update(seeds=len(seeds), inspected=inspected, all_listed=inspected <= MAX_TRACE_EDGES_SHOWN, expanded=len(expanded), nodes_reached=len(reached_nodes),
                          limit_reached=bool(len(expanded) >= V2_EXPANSION_LIMIT or tst["evidence_limit_reached"]), max_hops=tst["max_hops"], max_hop_reached=tst["max_hop_reached"],
                          outgoing_inspected=tst["outgoing_examined"], incoming_inspected=tst["incoming_examined"], followed=tst["edges_followed"], nodes_expanded=tst["nodes_expanded"],
                          frontier_not_expanded=tst["frontier_not_expanded"], hubs_not_expanded=[(h_["node_id"], h_["type"], h_["degree"]) for h_ in tst["hubs_not_expanded"]][:20],
                          budget_exhausted=tst["budget_exhausted"], edge_budget=tst["edge_budget"], stopped_because=tst["stopped_because"],
                          edges=[_fmt_edge(e_) for e_ in _shown],
                          expansion_paths=[{"evidence_id": it_["evidence_id"], "relations": list(it_["relations"]), "hops": it_["hops"], "path_edge_nos": _path_nos(it_["path"])[0],
                                            "path_edges_not_displayed": _path_nos(it_["path"])[1]} for it_ in expanded],
                          node_paths=[{"node_id": n_["node_id"], "type": n_["type"], "relations": list(n_["relations"]), "hops": n_["hops"], "path_edge_nos": _path_nos(n_["path"])[0]} for n_ in reached_nodes])
        ctx += ("LIMITATION: this trace covers the edges incident to the nodes expanded within the hop limit and edge budget (outgoing AND incoming). Edges beyond the hop limit, at hub nodes, "
                "past the budget, or between nodes that were never reached were NOT examined, so it is not a complete evidence lineage and does not show that all relevant evidence was found.\n")
        ctx += "\n"

    xd: Dict[str, Any] = {"status": "NOT_RUN", "enabled": False}  # Phase B: cross-document reasoning (read-only; V2 path only; cross_document=False reproduces the pre-Phase-B context exactly)
    if ENABLE_CROSS_DOCUMENT_REASONING if cross_document is None else bool(cross_document):
        if retrieved_evidence == 0: xd = {"status": "NO_SEEDS", "enabled": True}
        else:
            try:
                xd = traverse_cross_document_context(G, list(seeds) + [i_["evidence_id"] for i_ in expanded], objective)
                ctx += format_cross_document_for_context(G, xd)
            except Exception as e_:
                logger.exception(f"Cross-document context failed (continuing without it): {e_}")
                xd = {"status": "ERROR", "enabled": True, "reason": str(e_)[:200]}
    ctx += format_policy_results_for_context(G)
    ctx += format_contradictions_for_context(G)
    _gsrc = _govid_source_lines(G)
    if _gsrc:
        ctx += ("--- GOVERNMENT ID SOURCES (deterministic) ---\nID-like value detected ONLY in: " + "; ".join(_gsrc) +
                ". Attribute an ID only to these sources; never infer one for another document or from a shared employee name. Status stays as reported (UNVERIFIED when validation configuration is unavailable).\n\n")
    ctx += ("--- REPORTING RULES ---\nDo not list URLs yourself (source excerpts may hold malformed URLs; the system Supporting links list is the only link list). "
            "Do not describe the graph traversal as a single path or as complete lineage: the system Graph traversal appendix lists the inspected edges. "
            "Policy-listed amounts, thresholds and limits are policy context, not transaction records: never recommend making a transaction or submission match one merely because a policy lists it, and never recommend linking or associating a policy-listed amount with an actual expense claim or employee.\n\n")
    _links = normalize_link_log(link_log)  # same normalized link records as V1 (system retrieval status, not LLM output)
    if _links:
        ctx += f"--- SUPPORTING LINKS (system retrieval status; {link_status_summary(_links)}) ---\n" + "".join(_link_status_line(l, md=False, for_llm=True) for l in _links[:20]) + LINK_VS_FINDING_NOTE + "\n\n"

    rel_links = relevant_unretrieved_links(G, _links, list(seeds) + [i_["evidence_id"] for i_ in expanded], objective)  # from the normalized records + this investigation's evidence
    if rel_links:
        ctx += ("--- LINKS RELEVANT TO THIS INVESTIGATION (system data; retrieval status NOT RETRIEVED / BLOCKED / NOT ATTEMPTED: contents NOT inspected) ---\n"
                "Where a related finding is reported, say the link was not retrieved and its contents were not inspected. That is a retrieval outcome, separate from the finding's Status (e.g. INCONCLUSIVE). "
                "Never call these links invalid, fraudulent, legitimate, or proof for or against a claim. Do not list them again.\n")
        for r_ in rel_links[:20]:
            ctx += _link_status_line(r_["link"], md=False, for_llm=True).rstrip() + f" Relevance: {r_['basis']}" + (f"; evidence {', '.join(r_['evidence_ids'])}" if r_["evidence_ids"] else "") + ".\n"
        ctx += "\n"

    ctx += "--- APPLICABLE POLICY RULES (Context Only) ---\n"
    for n, data in G.nodes(data=True):
        if data.get("type") == "PolicyRule": ctx += f"Rule {n}: {data.get('condition')}\n"
    trace_edges = sum(len(i["relations"]) for i in expanded)
    return ctx, {"trace": trace_info, "inspected_edges": trace_info["inspected"], "retrieved": retrieved_evidence, "expanded": len(expanded), "nodes_reached": len(reached_nodes), "traversal_edges": trace_edges,
                 "context_only_expanded": sum(1 for i_ in expanded if not i_.get("supports_objective_lexically") or i_.get("policy_context")), "expansion_trace": expansion_trace,
                 "relevant_unretrieved_links": [{"key": r_["link"].get("key"), "status": r_["link"].get("status"), "evidence_ids": r_["evidence_ids"], "basis": r_["basis"]} for r_ in rel_links],
                 "relevant_unretrieved_note": v2_unretrieved_link_note(rel_links), "relevant_unretrieved_row": v2_unretrieved_link_row(rel_links), "cross_document": xd}

# --- EVALUATION: evidence retrieval + graph queries (structure only; NO scores are shipped) ---
# Ground truth is keyed by STABLE source keys (file, optional location), never by node ids (those are random per run).
# ground_truth = [{"file": "expenses.xlsx", "location": "Sheet1, Row 4"}, ...]   location omitted => any location in that file counts (file-level).
# Relevance rule: an Evidence node is relevant iff its source (filename, location) matches a ground-truth entry.
# Missing/ambiguous handling:
#   - ground_truth None/empty or no entry with a "file"  -> status NOT_MEASURED (no number produced)
#   - entries without "file"                              -> ignored, counted in ambiguous_entries
#   - entries matching no Evidence in the graph           -> unmatched_ground_truth (extraction miss; they count as NOT recalled)
#   - precision undefined (nothing retrieved)             -> None, never 0
# precision@k is over retrieved Evidence nodes; recall@k is over ground-truth ENTRIES covered by the retrieved set.
def _norm_key(x: Any) -> str:
    return str(x or "").strip().lower()

def _prf(retrieved: set, relevant: set) -> Dict[str, Any]:
    tp = len(retrieved & relevant)
    p = tp / len(retrieved) if retrieved else None
    r = tp / len(relevant) if relevant else None
    f1 = None if (p is None or r is None) else (0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return {"retrieved": len(retrieved), "relevant": len(relevant), "true_positives": tp, "precision": p, "recall": r, "f1": f1}

def _gt_entry_matches(G: nx.MultiDiGraph, ev_id: str, entry: Dict[str, Any]) -> bool:
    f, loc = str(entry["file"]).strip().lower(), entry.get("location")
    for x in _evidence_source_docs(G, ev_id):
        if str(x.get("filename") or "").strip().lower() != f: continue
        if not loc or str(x.get("location") or "").strip().lower() == str(loc).strip().lower(): return True
    return False

def ranked_evidence_ids(G: nx.MultiDiGraph, objective: str) -> List[str]:
    """Same BM25 ranking build_v2_context uses (lexical stage), exposed for evaluation."""
    items = _investigation_evidence(G)
    sc = bm25_scores(objective, [d.get("text", "") for _, d in items])
    return [n for s, n in sorted(((s, n) for s, (n, _) in zip(sc, items) if s > 0.0), key=lambda x: x[0], reverse=True)]

def evaluate_evidence_retrieval(G: nx.MultiDiGraph, objective: str, ground_truth: Optional[List[Dict[str, Any]]] = None,
                                k: Optional[int] = None, ground_truth_source: Optional[str] = None) -> Dict[str, Any]:
    k = k or V2_TOP_K
    ranked = ranked_evidence_ids(G, objective)
    top = ranked[:k]
    seeds = list(top)
    expanded = [i["evidence_id"] for i in traverse_graph_context(G, seeds, V2_EXPANSION_LIMIT)["evidence"]] if seeds else []  # same traversal the V2 context uses
    valid = [e for e in (ground_truth or []) if isinstance(e, dict) and str(e.get("file") or "").strip()]
    out: Dict[str, Any] = {"k": k, "objective": objective, "ground_truth_source": ground_truth_source or "unspecified",
                           "scope_note": ("Retrieval metrics are computed only against the supplied ground truth. They are NOT a V1 vs V2 comparison and do not "
                                          "establish overall accuracy; NOT_MEASURED means no usable ground truth was supplied."),
                           "ambiguous_entries": len(ground_truth or []) - len(valid), "retrieved_top_k": top, "graph_expanded": expanded}
    xd_explicit: List[str] = []
    xd_signal: List[str] = []
    if ENABLE_CROSS_DOCUMENT_REASONING and seeds:  # Phase B: same inputs build_v2_context gives the cross-document traversal (lexical top-K + graph-expanded)
        xd = traverse_cross_document_context(G, list(dict.fromkeys(top + expanded)), objective)
        xd_explicit = [i["evidence_id"] for i in xd["explicit_items"]]
        xd_signal = [i["evidence_id"] for k in ("weak_signals", "conflict_signals") for i in xd[k] if i.get("evidence_id")]
        out["cross_document"] = {"status": xd["status"], "metrics": xd["metrics"], "explicit_evidence": xd_explicit, "signal_evidence": xd_signal, "related_documents": [r["filename"] for r in xd["related_documents"]],
                                  "note": "counts describe what the traversal did; they are not accuracy. Precision/recall appear only with ground truth (below)."}
    else: out["cross_document"] = {"status": "NOT_RUN", "reason": "cross-document reasoning disabled or no lexical seed"}
    if not valid:
        out.update(status="NOT_MEASURED", reason="no usable ground truth supplied (needs entries with a 'file'); no metrics computed")
        return out
    evs = [n for n, _ in _investigation_evidence(G)]
    pool = {e for e in evs if any(_gt_entry_matches(G, e, g) for g in valid)}
    unmatched = [g for g in valid if not any(_gt_entry_matches(G, e, g) for e in evs)]
    def score(ids: List[str]) -> Dict[str, Any]:
        s = set(ids)
        covered = sum(1 for g in valid if any(_gt_entry_matches(G, e, g) for e in s))
        m = _prf(s, pool)
        m["recall"] = covered / len(valid)  # entry-level recall (see header)
        m["gt_entries_covered"], m["gt_entries_total"] = covered, len(valid)
        m["f1"] = None if m["precision"] is None else (0.0 if (m["precision"] + m["recall"]) == 0 else 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]))
        return m
    out.update(status="MEASURED", unmatched_ground_truth=unmatched, relevant_evidence_in_graph=len(pool),
               lexical_top_k=score(top), lexical_plus_graph_expansion=score(list(dict.fromkeys(top + expanded))),
               lexical_plus_graph_plus_cross_document=score(list(dict.fromkeys(top + expanded + xd_explicit))), cross_document_added_only=score(xd_explicit) if xd_explicit else {"status": "NOT_MEASURED", "reason": "no evidence was added through cross-document links"},
               variant_notes={"v2_existing_graph": "lexical_plus_graph_expansion", "v2_phase_a": "Phase A adds CROSS_DOCUMENT_LINK edges that retrieval does not follow, so its retrieval equals lexical_plus_graph_expansion by construction; see cross_document.metrics for link counts",
                              "v2_phase_b": "lexical_plus_graph_plus_cross_document", "v1": "no retrieval stage (NOT_MEASURED)"})
    return out

def evaluate_graph_queries(G: nx.MultiDiGraph, expectations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """structural_checks: ground-truth-free invariants (consistency, NOT accuracy). accuracy: only with expectations
    [{"rule_contains": "text", "expected_documents": ["a.pdf", ...]}]; a selector matching 0 or >1 Decisions is ambiguous and skipped.
    decision_lineage_status counts the SOURCE-PATH status of discovered evidence only (see query_decision_lineage); it does not claim exhaustive discovery."""
    decisions = _nodes_of_type(G, "Decision")
    lin = {d_id: query_decision_lineage(G, d_id) for d_id, _ in decisions}
    evs = [n for n, _ in _nodes_of_type(G, "Evidence")]
    contr = [d for _, _, d in G.edges(data=True) if d.get("relation") == "CONTRADICTS"]
    structural = {
        "decisions": len(decisions),
        "decision_lineage_status": dict(Counter(l["status"] for l in lin.values())),
        "evidence_without_source_document": sum(1 for e in evs if not _evidence_source_docs(G, e)),
        "lineage_documents_missing_from_graph": sum(1 for l in lin.values() for x in l["documents"] if not G.has_node(x["document_id"])),
        "weak_contradictions_not_marked_heuristic": sum(1 for d in contr if d.get("match_strength", "weak") != "strong" and not d.get("heuristic")),
        "violation_decisions_without_evidence": sum(1 for d_id, l in lin.items() if l["verdict"] == "VIOLATION" and not l["evidence"] and G.nodes[d_id].get("violation_status") != "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE"),
        "absence_based_violation_decisions": sum(1 for d_id, _ in decisions if G.nodes[d_id].get("violation_status") == "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE"),  # REQUIRE KEYWORD not found in extracted scope: no supporting evidence exists by definition
        "semantic_links_flagged_as_source_provenance": sum(1 for l in lin.values() for x in l.get("semantic_links", []) if x.get("is_source_provenance")),  # must be 0
        "decisions_with_untraced_discovered_evidence": sum(1 for l in lin.values() if l["evidence"] and not l.get("tracing_complete_for_discovered_evidence")),
        "evidence_with_provenance_defects": sum(1 for g_ in evidence_provenance_gaps(G) if g_["defects"]),  # must be 0 (unavailable location / unmeasured confidence are honest gaps, not defects)
        "evidence_with_location_unavailable": sum(1 for g_ in evidence_provenance_gaps(G) if "location_unavailable" in g_["honest_gaps"]),
        "policy_rules_without_source_evidence": sum(1 for r_, _ in _nodes_of_type(G, "PolicyRule") if not any(_is_policy_source_evidence(x["node"]) for x in _in_edge_sources(G, r_, "SUPPORTS", "Evidence"))),
        "decisions_without_rule_policy_chain": sum(1 for l in lin.values() if not l["rule"] or not l["policies"]),
        "decisions_whose_rule_is_not_traced_to_rulebook_document": sum(1 for l in lin.values() if l["rule"] and not l.get("chain", {}).get("rule_to_source_document")),
        **_heuristic_labelling_checks(G),
    }
    out: Dict[str, Any] = {"structural_checks": structural}
    if not expectations:
        out["accuracy"] = {"status": "NOT_MEASURED", "reason": "no expected lineage ground truth supplied"}
        return out
    rows = []
    for exp in expectations:
        key = str((exp or {}).get("rule_contains") or "").strip().lower()
        exp_docs = {str(x).strip().lower() for x in (exp or {}).get("expected_documents") or [] if str(x).strip()}
        if not key or not exp_docs:
            rows.append({"selector": key, "status": "AMBIGUOUS", "reason": "missing rule_contains or expected_documents"}); continue
        hits = [d_id for d_id, dd in decisions if key in str((lin[d_id]["rule"] or {}).get("condition") or "").lower()]
        if len(hits) != 1:
            rows.append({"selector": key, "status": "AMBIGUOUS", "reason": f"selector matched {len(hits)} decisions (need exactly 1)"}); continue
        got = {str(x.get("filename") or "").strip().lower() for x in lin[hits[0]]["documents"]}
        rows.append({"selector": key, "status": "MEASURED", "decision_id": hits[0], **_prf(got, exp_docs)})
    out["accuracy"] = {"status": "MEASURED" if any(r["status"] == "MEASURED" for r in rows) else "NOT_MEASURED", "per_expectation": rows}
    return out

# --- V1 vs V2 EXPERIMENT: METHODOLOGY (this text is not a result) + IMPLEMENTATION (run_v1_v2_experiment / build_experiment_record below) ---
# The methodology is implemented: running it records ACTUAL retrieval and decision-quality metrics against human-labelled ground truth. A metric that was not run
# (no ground truth, no LLM run, no manual review) stays NOT_MEASURED; nothing is estimated or reported as a result without a run.
V1_V2_EXPERIMENT_DESIGN = """
V1 = document-based LLM reasoning (build_v1_amount_brief + extracted text). V2 = evidence-graph reasoning (build_v2_context).
INPUTS (identical per case): same files, rulebook, objective. Fix EXTERNAL_AI_ENABLED=false + one local OLLAMA_PRO_MODEL so V1/V2 use the same model
  (provider fallback is not returned by analyze_compliance_with_matrix; otherwise record it manually). Repeat each case N>=3 runs (LLM variance).
  Cases: >=1 of each: reliable-ID amount mismatch, weak-link amount difference, policy-listed amount, valid/unconfigured GovID, unreadable doc, rule not in DSL.
GROUND TRUTH (human-labelled BEFORE running, stored with case): relevant evidence (file, location); expected verdict per rule
  (VIOLATION/SATISFIED/INCONCLUSIVE/UNEVALUATED/NOT_APPLICABLE); expected source documents per decision; labeller + date. Ambiguous labels: exclude and count.
METRICS
  Retrieval (V2): evaluate_evidence_retrieval -> precision@k, recall@k, F1, with and without graph expansion; unmatched_ground_truth.
  Retrieval proxy (V1, no retrieval stage): precision/recall of (file, location) citations in the report vs the same ground truth (manual or script).
  Graph queries (V2): evaluate_graph_queries -> structural checks + lineage document precision/recall.
  Decision quality (both): verdict agreement per rule vs label; false-VIOLATION count; correct INCONCLUSIVE/UNEVALUATED where evidence is insufficient;
    unsupported-claim count (statement with no cited evidence, manual review); citation validity (cited file@location exists); heuristic items labelled as heuristic.
  Operational (not accuracy): latency (v1_latency, v2_latency in metrics).
RECORDING: one JSON per (case, run, version). build_experiment_record() / run_v1_v2_experiment() fill "measured" from the real run against the labelled ground truth;
  every metric that was not run stays "NOT_MEASURED" (never None-as-zero, never estimated). summarize_experiment_records() aggregates ONLY recorded values.
ANALYSIS: report per-case raw values and mean/min/max across runs; paired per-case comparison. No superiority claim unless pre-specified criteria are met
  on actual recorded data with an adequate case count; otherwise report 'no conclusion'.
"""

NOT_MEASURED = "NOT_MEASURED"
_MEASURED_FIELDS = ("retrieval", "graph_queries", "verdict_agreement", "false_violations", "correct_inconclusive_unevaluated", "unsupported_claims",
                    "citation_validity", "report_citations", "latency_s", "context_build_latency_s")
_VERDICTS = {"VIOLATION", "SATISFIED", "NOT_APPLICABLE", "INCONCLUSIVE", "UNEVALUATED"}

def _nm(reason: str) -> Dict[str, Any]:
    return {"status": NOT_MEASURED, "reason": reason}

def _is_measured(x: Any) -> bool:
    return (isinstance(x, dict) and x.get("status") == "MEASURED") or (isinstance(x, (int, float)) and not isinstance(x, bool))

def experiment_results_template(case_id: str, version: str, run: int = 1) -> Dict[str, Any]:
    """Blank record. Everything under 'measured' is NOT_MEASURED until a real run fills it."""
    return {"case_id": case_id, "version": version, "run": run, "status": NOT_MEASURED, "labels": {"labeller": None, "date": None, "ground_truth_source": None},
            "config": {"model": None, "provider": None, "external_ai_enabled": None, "k": V2_TOP_K},
            "measured": {k: NOT_MEASURED for k in _MEASURED_FIELDS}, "notes": None}

def _heuristic_labelling_checks(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Ground-truth-free invariants for requirement 'heuristic contradictions are never facts / never alone a violation'. All counts must be 0."""
    contr = [d for _, _, d in G.edges(data=True) if d.get("relation") == "CONTRADICTS"]
    on_heur = 0
    no_evidence = 0
    for u, v, d in G.edges(data=True):
        if d.get("relation") not in ("VIOLATES", "SATISFIES"): continue
        if G.nodes[u].get("type") != "Evidence" and not _supporting_evidence_ids(G, u): no_evidence += 1
        if d.get("relation") == "VIOLATES" and _basis_link_problem(G, u): on_heur += 1
    return {"contradictions_marked_confirmed": sum(1 for d in contr if d.get("confirmed")),
            "contradictions_without_unconfirmed_label": sum(1 for d in contr if not d.get("contradiction_label") or d.get("confirmation_status") != "UNCONFIRMED"),
            "contradictions_allowed_to_create_violation": sum(1 for d in contr if d.get("can_create_violation", False)),
            "violates_edges_without_qualifying_evidence": on_heur,
            "violates_or_satisfies_edges_without_supporting_evidence": no_evidence}

def graph_chain_summary(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """What part of the nine-node chain exists in THIS graph. A node type is absent when no extracted evidence or configured rule supports it; nothing is created to fill the chain."""
    types = Counter(d.get("type") for _, d in G.nodes(data=True) if d.get("type") in ALLOWED_NODE_TYPES)
    rels = Counter(d.get("relation") for _, _, d in G.edges(data=True))
    rules, decs = _nodes_of_type(G, "PolicyRule"), _nodes_of_type(G, "Decision")
    viol = [(n, d) for n, d in decs if d.get("verdict") == "VIOLATION"]
    return {"node_types": {t: types.get(t, 0) for t in ALLOWED_NODE_TYPES}, "absent_node_types": [t for t in ALLOWED_NODE_TYPES if not types.get(t)],
            "relations": dict(rels),
            "links": {"rules_belonging_to_policy": sum(1 for r, _ in rules if any(d.get("relation") == "BELONGS_TO" for _, _, d in G.out_edges(r, data=True))),
                      "rules_with_rulebook_source_evidence": sum(1 for r, _ in rules if any(_is_policy_source_evidence(x["node"]) for x in _in_edge_sources(G, r, "SUPPORTS", "Evidence"))),
                      "decisions_evaluating_a_rule": sum(1 for n, _ in decs if any(d.get("relation") == "EVALUATES" for _, _, d in G.out_edges(n, data=True))),
                      "decisions_with_supporting_evidence": sum(1 for n, _ in decs if _in_edge_sources(G, n, "SUPPORTS", "Evidence")),
                      "violation_decisions": len(viol), "violation_decisions_with_risk": sum(1 for n, _ in viol if any(d.get("relation") == "HAS_RISK" for _, _, d in G.out_edges(n, data=True))),
                      "transactions_with_supporting_evidence": sum(1 for n, _ in _nodes_of_type(G, "Transaction") if _in_edge_sources(G, n, "SUPPORTS", "Evidence")),
                      "violates_or_satisfies_edges": rels.get("VIOLATES", 0) + rels.get("SATISFIES", 0)},
            "note": ("Absent node types are absent because no extracted evidence or configured rule supports them (for example no Transaction without an amount in the evidence, no Risk without a "
                     "deterministic VIOLATION, no Policy without a readable rulebook); nothing is created to fill the chain.")}

# --- DECISION QUALITY (against human-labelled expected verdicts) ---
def _compare_verdicts(expected: List[Dict[str, Any]], predicted_for) -> Dict[str, Any]:
    """expected = [{"rule_contains": "...", "verdict": "..."}]; predicted_for(key) -> (verdict | None, n_matches). A selector that does not resolve to exactly one prediction is
    AMBIGUOUS and excluded (and counted). Returns agreement, false_violations, missed_violations and correct INCONCLUSIVE/UNEVALUATED/NOT_APPLICABLE handling."""
    rows, ambiguous = [], 0
    for exp in expected or []:
        key, want = _norm_key((exp or {}).get("rule_contains")), str((exp or {}).get("verdict") or "").strip().upper()
        if not key or want not in _VERDICTS:
            ambiguous += 1; rows.append({"selector": key, "status": "AMBIGUOUS", "reason": "missing rule_contains or unknown expected verdict"}); continue
        got, n = predicted_for(key)
        if n != 1 or got is None:
            ambiguous += 1; rows.append({"selector": key, "status": "AMBIGUOUS", "reason": f"selector matched {n} prediction(s) (need exactly 1)"}); continue
        rows.append({"selector": key, "status": "MEASURED", "expected": want, "predicted": got, "correct": got == want})
    m = [r for r in rows if r["status"] == "MEASURED"]
    if not m: return {"verdict_agreement": _nm("no labelled expectation resolved to exactly one prediction"), "false_violations": _nm("no resolved expectation"),
                      "correct_inconclusive_unevaluated": _nm("no resolved expectation"), "missed_violations": _nm("no resolved expectation"), "rows": rows, "ambiguous": ambiguous}
    correct = sum(1 for r in m if r["correct"])
    insuff = [r for r in m if r["expected"] in ("INCONCLUSIVE", "UNEVALUATED", "NOT_APPLICABLE")]
    fv = sum(1 for r in m if r["predicted"] == "VIOLATION" and r["expected"] != "VIOLATION")
    mv = sum(1 for r in m if r["expected"] == "VIOLATION" and r["predicted"] != "VIOLATION")
    return {"verdict_agreement": {"status": "MEASURED", "agreement": correct / len(m), "correct": correct, "matched": len(m)},
            "false_violations": {"status": "MEASURED", "count": fv, "of_matched": len(m)},
            "missed_violations": {"status": "MEASURED", "count": mv, "of_matched": len(m)},
            "correct_inconclusive_unevaluated": ({"status": "MEASURED", "rate": sum(1 for r in insuff if r["correct"]) / len(insuff), "correct": sum(1 for r in insuff if r["correct"]), "total": len(insuff)}
                                                 if insuff else _nm("no labelled expectation with an insufficient-evidence verdict (INCONCLUSIVE/UNEVALUATED/NOT_APPLICABLE)")),
            "rows": rows, "ambiguous": ambiguous}

def evaluate_decision_quality(G: nx.MultiDiGraph, expected_verdicts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Verdicts of the deterministic policy engine (Decision nodes in the graph) vs labelled expected verdicts. Also reports the structural heuristic-labelling invariants."""
    decs = [(n, d, str((G.nodes.get(d.get("rule_id")) or {}).get("condition") or "").lower()) for n, d in _nodes_of_type(G, "Decision")]
    def pred(key: str) -> Tuple[Optional[str], int]:
        hits = [d for _, d, cond in decs if key in cond]
        return (hits[0].get("verdict") if len(hits) == 1 else None), len(hits)
    out = _compare_verdicts(expected_verdicts or [], pred) if expected_verdicts else {"verdict_agreement": _nm("no expected verdicts supplied"), "false_violations": _nm("no expected verdicts supplied"),
                                                                                   "correct_inconclusive_unevaluated": _nm("no expected verdicts supplied"), "missed_violations": _nm("no expected verdicts supplied"), "rows": [], "ambiguous": 0}
    out["source"] = "deterministic_policy_engine (Decision nodes of the graph)"
    out["heuristic_labelling"] = _heuristic_labelling_checks(G)
    return out

# --- REPORT CITATIONS (string-match proxy; works for V1 and V2 report text) ---
_CITE_RE = re.compile(r'([\w.\-/!:%]+\.(?:pdf|docx|xlsx|csv|txt|md|png|jpe?g|bmp|tiff|webp|json|html|py|js|yml|yaml))\s*@\s*([^\n;|)\]]+)', re.IGNORECASE)

def _evidence_source_pairs(G: nx.MultiDiGraph) -> List[Tuple[str, str]]:
    pairs = set()
    for ev, _ in _investigation_evidence(G):
        for x in _evidence_source_docs(G, ev):
            if x.get("filename") and x.get("location"): pairs.add((str(x["filename"]), str(x["location"])))
    return sorted(pairs)

def _loc_prefix_match(cited: str, known: str) -> bool:
    return cited == known or (cited.startswith(known) and not cited[len(known):len(known) + 1].isalnum())

def evaluate_report_citations(G: nx.MultiDiGraph, report_text: Optional[str], ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """PROXY metrics from report TEXT (LLM output): a (file, location) is 'cited' when the location string occurs in the text with the file name within 250 characters.
    citation_validity = syntactic 'file @ location' citations that match a real Evidence source (location match on a word boundary). Retrieval proxy = precision / recall of the
    cited (file, location) pairs vs the labelled relevant evidence. This measures what the report cites, not whether its conclusions are correct."""
    if not report_text or not str(report_text).strip(): return _nm("no report text supplied")
    text, low = str(report_text), str(report_text).lower()
    pairs = _evidence_source_pairs(G)[:5000]
    cited = []
    for fn, loc in pairs:
        fl, ll, start = fn.lower(), loc.lower(), 0
        while True:
            i = low.find(ll, start)
            if i < 0: break
            if fl in low[max(0, i - 250):i + len(ll) + 250]:
                cited.append((fn, loc)); break
            start = i + 1
    known = [(f.lower(), l.lower()) for f, l in pairs]
    syn = [(m.group(1).strip().lower(), m.group(2).strip().lower().rstrip(".,`*:")) for m in _CITE_RE.finditer(text)][:500]
    valid = sum(1 for fn, loc in syn if any(fn == kf and _loc_prefix_match(loc, kl) for kf, kl in known))
    out: Dict[str, Any] = {"status": "MEASURED", "method": "string_match_proxy", "cited_pairs": len(cited), "cited_pairs_sample": cited[:50],
                           "citation_validity": {"cited_syntactic": len(syn), "valid": valid, "value": (valid / len(syn)) if syn else None,
                                                 "note": "value None = the text contains no 'file @ location' citation to check (not zero)"}}
    gt = [e for e in (ground_truth or []) if isinstance(e, dict) and str(e.get("file") or "").strip()]
    if not gt:
        out["retrieval_proxy"] = _nm("no usable ground truth supplied")
        return out
    def hit(pair, e):
        if str(pair[0]).strip().lower() != str(e["file"]).strip().lower(): return False
        return not e.get("location") or str(pair[1]).strip().lower() == str(e["location"]).strip().lower()
    tp = sum(1 for p in cited if any(hit(p, e) for e in gt))
    covered = sum(1 for e in gt if any(hit(p, e) for p in cited))
    prec = (tp / len(cited)) if cited else None
    rec = covered / len(gt)
    out["retrieval_proxy"] = {"status": "MEASURED", "precision": prec, "recall": rec, "f1": (None if prec is None else (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))),
                              "cited_relevant": tp, "cited_total": len(cited), "gt_entries_covered": covered, "gt_entries_total": len(gt)}
    return out

# --- RECORD / RUN / SUMMARISE ---
def build_experiment_record(case_id: str, version: str, run: int, G: nx.MultiDiGraph, objective: str, report_text: Optional[str] = None, latency_s: Optional[float] = None,
                            ground_truth: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, context_build_latency_s: Optional[float] = None) -> Dict[str, Any]:
    """One (case, run, version) record with ACTUAL measurements against labelled ground truth.
    ground_truth = {"labeller","date","source","relevant_evidence":[{"file","location"}], "expected_verdicts":[{"rule_contains","verdict"}],
                    "expected_lineage":[{"rule_contains","expected_documents":[...]}],
                    "reviewed_verdicts":{"V1":[{"rule_contains","verdict"}],"V2":[...]}   (a human's reading of the LLM report; optional),
                    "manual_review":{"V1":{"unsupported_claims":int},"V2":{...}}           (optional)}
    V2: retrieval + graph-query metrics and deterministic-engine verdict agreement come from the graph. V1 has no retrieval/graph stage (NOT_MEASURED), and its LLM-report
    verdicts are free text, so verdict agreement for V1 needs `reviewed_verdicts`. Report-citation metrics need report text. Latency needs a real run. Anything else stays NOT_MEASURED."""
    ver = str(version).upper()
    gt = ground_truth or {}
    rec = experiment_results_template(case_id, ver, run)
    rec["labels"].update(labeller=gt.get("labeller"), date=gt.get("date"), ground_truth_source=gt.get("source") or ("supplied" if gt else None))
    if config: rec["config"].update(config)
    m = rec["measured"]
    if ver == "V2":
        m["retrieval"] = evaluate_evidence_retrieval(G, objective, gt.get("relevant_evidence"), ground_truth_source=rec["labels"]["ground_truth_source"])
        gq = evaluate_graph_queries(G, gt.get("expected_lineage"))
        m["graph_queries"] = {"status": "MEASURED", **gq}
        dq = evaluate_decision_quality(G, gt.get("expected_verdicts"))
        for k in ("verdict_agreement", "false_violations", "correct_inconclusive_unevaluated"): m[k] = dq[k]
        m["missed_violations"] = dq["missed_violations"]
        m["heuristic_labelling"] = dq["heuristic_labelling"]
    else:
        m["retrieval"] = _nm("V1 has no retrieval stage (the whole extracted text is passed); see report_citations for the report-citation proxy")
        m["graph_queries"] = _nm("V1 has no graph")
        reviewed = (gt.get("reviewed_verdicts") or {}).get("V1")
        if reviewed and gt.get("expected_verdicts"):
            lut = {_norm_key(r.get("rule_contains")): str(r.get("verdict") or "").strip().upper() for r in reviewed if isinstance(r, dict)}
            dq = _compare_verdicts(gt["expected_verdicts"], lambda key: ((lut.get(key), 1) if key in lut else (None, 0)))
            for k in ("verdict_agreement", "false_violations", "correct_inconclusive_unevaluated"): m[k] = dq[k]
            m["missed_violations"] = dq["missed_violations"]
            m["verdict_source"] = "manual review of the V1 report (reviewed_verdicts)"
        else:
            for k in ("verdict_agreement", "false_violations", "correct_inconclusive_unevaluated"):
                m[k] = _nm("V1 verdicts are free LLM text; supply reviewed_verdicts (manual reading) and expected_verdicts to measure")
    mr = ((gt.get("manual_review") or {}).get(ver) or {})
    m["unsupported_claims"] = ({"status": "MEASURED", "count": int(mr["unsupported_claims"]), "source": "manual review"} if isinstance(mr.get("unsupported_claims"), int) and not isinstance(mr.get("unsupported_claims"), bool)
                               else _nm("needs manual review of the report (statements with no cited evidence)"))
    rc = evaluate_report_citations(G, report_text, gt.get("relevant_evidence")) if report_text else _nm("no LLM report was produced in this run")
    m["report_citations"] = rc
    m["citation_validity"] = ({"status": "MEASURED", "value": rc["citation_validity"]["value"], "valid": rc["citation_validity"]["valid"], "cited_syntactic": rc["citation_validity"]["cited_syntactic"]}
                              if rc.get("status") == "MEASURED" and rc["citation_validity"]["cited_syntactic"] else _nm("no report text or no 'file @ location' citation in it"))
    m["latency_s"] = round(float(latency_s), 4) if latency_s is not None else NOT_MEASURED
    m["context_build_latency_s"] = round(float(context_build_latency_s), 4) if context_build_latency_s is not None else NOT_MEASURED
    rec["status"] = "MEASURED" if any(_is_measured(v) for v in m.values()) else NOT_MEASURED
    return rec

def _scalar_metrics(rec: Dict[str, Any]) -> Dict[str, float]:
    m = rec.get("measured") or {}
    out: Dict[str, Any] = {}
    r = m.get("retrieval")
    if isinstance(r, dict) and r.get("status") == "MEASURED":
        for name, key in (("retrieval_lexical_top_k", "lexical_top_k"), ("retrieval_with_graph_expansion", "lexical_plus_graph_expansion"), ("retrieval_with_cross_document", "lexical_plus_graph_plus_cross_document")):
            for met in ("precision", "recall", "f1"): out[f"{name}_{met}"] = (r.get(key) or {}).get(met)
    for name, key, sub in (("verdict_agreement", "verdict_agreement", "agreement"), ("false_violations", "false_violations", "count"), ("missed_violations", "missed_violations", "count"),
                           ("correct_inconclusive_unevaluated", "correct_inconclusive_unevaluated", "rate"), ("unsupported_claims", "unsupported_claims", "count"), ("citation_validity", "citation_validity", "value")):
        v = m.get(key)
        if isinstance(v, dict) and v.get("status") == "MEASURED": out[name] = v.get(sub)
    rc = m.get("report_citations")
    if isinstance(rc, dict) and rc.get("status") == "MEASURED" and isinstance(rc.get("retrieval_proxy"), dict) and rc["retrieval_proxy"].get("status") == "MEASURED":
        for met in ("precision", "recall", "f1"): out[f"report_citation_{met}"] = rc["retrieval_proxy"].get(met)
    for k in ("latency_s", "context_build_latency_s"):
        if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool): out[k] = m[k]
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

def summarize_experiment_records(records: List[Dict[str, Any]], criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-version mean/min/max over recorded runs, per-case raw values, paired per-case V1/V2 comparison. NO superiority claim unless `criteria` (pre-specified) is met on
    recorded data: criteria = {"metric": "verdict_agreement", "min_cases": 10, "min_mean_delta": 0.1, "higher_is_better": True}. Otherwise status NO_CONCLUSION."""
    per_case: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    not_measured: Dict[str, Counter] = defaultdict(Counter)
    for r in records or []:
        v = str(r.get("version")).upper()
        for k, val in _scalar_metrics(r).items(): per_case[str(r.get("case_id"))][v][k].append(val)
        for k, val in (r.get("measured") or {}).items():
            if not _is_measured(val): not_measured[v][k] += 1
    agg: Dict[str, Dict[str, Any]] = defaultdict(dict)
    pool: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for case, vers in per_case.items():
        for v, mets in vers.items():
            for k, vals in mets.items(): pool[v][k].extend(vals)
    for v, mets in pool.items():
        for k, vals in mets.items(): agg[v][k] = {"n": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    paired = []
    for case, vers in sorted(per_case.items()):
        if "V1" in vers and "V2" in vers:
            for k in sorted(set(vers["V1"]) & set(vers["V2"])):
                a_, b_ = sum(vers["V1"][k]) / len(vers["V1"][k]), sum(vers["V2"][k]) / len(vers["V2"][k])
                paired.append({"case_id": case, "metric": k, "v1_mean": a_, "v2_mean": b_, "delta_v2_minus_v1": b_ - a_, "runs_v1": len(vers["V1"][k]), "runs_v2": len(vers["V2"][k])})
    conclusion: Dict[str, Any] = {"status": "NO_CONCLUSION", "reason": "no pre-specified criteria supplied; raw recorded values only"}
    if criteria:
        rows = [p for p in paired if p["metric"] == criteria.get("metric")]
        need = int(criteria.get("min_cases", 10))
        if len(rows) < need: conclusion = {"status": "NO_CONCLUSION", "reason": f"only {len(rows)} paired case(s) with recorded '{criteria.get('metric')}' (criteria need {need})"}
        else:
            md = sum(p["delta_v2_minus_v1"] for p in rows) / len(rows)
            ok = md >= float(criteria.get("min_mean_delta", 0.0)) if criteria.get("higher_is_better", True) else md <= -float(criteria.get("min_mean_delta", 0.0))
            conclusion = {"status": "CRITERIA_MET" if ok else "CRITERIA_NOT_MET", "cases": len(rows), "mean_delta_v2_minus_v1": md, "criteria": criteria,
                          "note": "descriptive result on the recorded cases only; not a general accuracy claim"}
    return {"records": len(records or []), "aggregate_by_version": {v: dict(m) for v, m in agg.items()}, "per_case": {c: {v: {k: list(vs) for k, vs in m.items()} for v, m in vers.items()} for c, vers in per_case.items()},
            "paired_comparison": paired, "not_measured_counts": {v: dict(c) for v, c in not_measured.items()}, "conclusion": conclusion}

def _rulebook_text_only(rulebook_path: Optional[str]) -> str:
    if not rulebook_path: return ""
    return "\n".join(f["text"] for f in extract_provenance_facts(rulebook_path) if f.get("method") not in ("error", "extraction_note"))

def _build_v1_payload(file_paths: List[str], rulebook_text: str, G: nx.MultiDiGraph) -> str:
    """V1 payload exactly as the task builds it from local files (amount brief + system pre-checks + full extracted text); no URL retrieval in experiment mode."""
    entries: List[Dict[str, Any]] = collect_amount_entries(rulebook_text, "rulebook", True) if rulebook_text else []
    statuses: List[str] = []
    sources: List[str] = []
    combined = ""
    for fp in file_paths:
        if is_sensitive_file(fp): continue
        try:
            if os.path.getsize(fp) > MAX_LOCAL_FILE_BYTES: continue
        except OSError: continue
        fn = os.path.basename(fp)
        raw = extract_text_from_file(fp, redact_code=False)
        if raw.strip() and "[Extraction Error]" not in raw:
            entries.extend(collect_amount_entries(raw, fn))
            statuses.extend(verify_govid(c_["digits"])["status"] for c_ in find_govid_candidates(raw))
            sources.extend(_govid_occurrences(raw, fn))
            if len(combined) < 100000:
                combined += f"\n\n========================================\nFILE NAME: {fn}\n========================================\n\n{generate_system_validation_report(raw)}{generate_amount_role_report(raw, fn)}{raw}"
    return build_v1_amount_brief(entries, policy_coverage(G), statuses, [], sources) + combined

# --- CROSS-DOCUMENT BENCHMARK + EVALUATION (evaluation layer only; no linking / retrieval / decision logic is changed) ---
# Deterministic labelled cases + an evaluator that runs the EXISTING pipeline per variant on the SAME cases and scores it against the labels:
#   V1              = documents present in the V1 payload (_build_v1_payload); no retrieval stage, no graph, no links, no contradiction signals, free-text LLM verdicts
#   EVIDENCE_GRAPH = existing V2 retrieval: lexical top-K + graph expansion (ranked_evidence_ids + traverse_graph_context); Phase A edges exist but are not followed
#   CROSS_DOCUMENT = EVIDENCE_GRAPH + Phase B (traverse_cross_document_context): evidence reached through EXPLICIT links and evidence of CONFLICTING-link documents
# No randomness (seed: none). No LLM. Aggregates are derived from the raw per-case records only. Metrics that cannot legitimately be computed are NOT_MEASURED.
# The benchmark decision is an EVALUATION PROTOCOL applied identically to every variant's retrieved evidence (not a change to the compliance decision logic):
#   RECONCILIATION_REQUIRED if the variant produced a cross-document contradiction signal involving a required document (a heuristic signal, NEVER a VIOLATION);
#   else SUPPORTED if every required document is covered by retrieved evidence; else INSUFFICIENT_EVIDENCE.
# V1 decisions / contradictions / links are NOT_MEASURED: V1 verdicts are free LLM text and V1 has no graph.
XBENCH_ID, XBENCH_VERSION = "omnicheck-xdoc-bench", "1.0"
XBENCH_VARIANTS = ("V1", "EVIDENCE_GRAPH", "CROSS_DOCUMENT")
XBENCH_DECISIONS = ("SUPPORTED", "RECONCILIATION_REQUIRED", "INSUFFICIENT_EVIDENCE")
XBENCH_PROTOCOL = {
    "seed": None, "randomness": "none", "llm_used": False,
    "decision": "RECONCILIATION_REQUIRED if a contradiction signal involves a required document; else SUPPORTED if all required documents are covered by retrieved evidence; else INSUFFICIENT_EVIDENCE. A contradiction signal is never a violation.",
    "retrieval": "evidence-level precision = retrieved evidence from relevant (required+supporting+contradicting) documents / retrieved evidence; recall = relevant ground-truth documents covered by retrieved evidence / relevant documents (entry-level, as in evaluate_evidence_retrieval); completeness = required documents covered / required documents. CROSS_DOCUMENT retrieved = EVIDENCE_GRAPH + evidence reached through EXPLICIT links + evidence of CONFLICTING-link documents; WEAK-link documents are surfaced as possibilities and are NOT counted as retrieved.",
    "entity_links": "unordered document pairs. EVIDENCE_GRAPH: documents sharing a GovID/Phone Entity node (strong only for a valid GovID; Term co-mentions are heuristic phrase overlap and are not counted). CROSS_DOCUMENT adds CROSS_DOCUMENT_LINK edges with status EXPLICIT/WEAK/CONFLICTING (strength = the edge's match_strength); UNRESOLVED edges are not links. Semantic similarity alone is never strong.",
    "contradictions": "unordered cross-document pairs. EVIDENCE_GRAPH: CONTRADICTS edges between evidence of different documents. CROSS_DOCUMENT adds conflict signals surfaced by the cross-document traversal. false_contradiction_rate = FP / predicted; missed_contradiction_rate = FN / expected; case_false_positive_rate = share of cases with no expected contradiction where one was predicted.",
    "metrics_not_measured_when": "denominator is empty (nothing expected / nothing predicted), the variant has no such mechanism (V1), or ground truth does not exist.",
}

def _xb_case(cid, category, difficulty, objective, docs, decision, required, policy=None, supporting=(), contradicting=(), missing=(), links=(), contradictions=(), distractors=(), provenance=()):
    f = lambda xs: [{"file": x} for x in xs]
    return {"case_id": cid, "category": category, "difficulty": difficulty, "objective": objective, "policy_context": policy, "documents": dict(docs), "expected_decision": decision,
            "required_evidence": f(required), "supporting_evidence": f(supporting), "contradicting_evidence": f(contradicting),
            "missing_evidence": [{"reference_type": t, "reference": r, "file": fl} for t, r, fl in missing],
            "expected_entity_links": [{"a": a, "b": b, "strength": s, "basis": bs} for a, b, s, bs in links],
            "expected_contradictions": [{"a": a, "b": b, "field": fl} for a, b, fl in contradictions],
            "distractors": list(distractors),
            "expected_provenance": [{"file": fl, "reached_from": rf, "relationship_type": rt, "link_status": st} for fl, rf, rt, st in provenance]}

CROSS_DOCUMENT_BENCHMARK: List[Dict[str, Any]] = [
    _xb_case("XB-01-one-document", "one_document_sufficient", "easy", "review billed amount",
             {"invoice_001.txt": "Invoice No: INV-1001\nVendor: Zenith Tools\nBilled amount INR 4,000\nPayment due in 30 days\n"},
             "SUPPORTED", ["invoice_001.txt"], policy="An invoice must state its vendor and billed amount."),
    _xb_case("XB-02-two-documents-identifier", "two_document_identifier_link", "easy", "review billed amount",
             {"invoice_002.txt": "Invoice No: INV-2002\nPO Number: PO-2002\nVendor: Bluefin Traders\nBilled amount INR 9,000\n",
              "po_002.txt": "Purchase Order No: PO-2002\nVendor: Bluefin Traders\nOrder value INR 9,000\nItems: cables\n"},
             "SUPPORTED", ["invoice_002.txt", "po_002.txt"], policy="An invoice must be backed by the purchase order it references.",
             links=[("invoice_002.txt", "po_002.txt", "strong", "identifier PO-2002 and vendor name")],
             provenance=[("po_002.txt", "invoice_002.txt", "PURCHASE_ORDER_TO_INVOICE", "EXPLICIT")]),
    _xb_case("XB-03-three-document-chain", "three_document_chain", "medium", "review billed amount",
             {"invoice_003.txt": "Invoice No: INV-3003\nPO Number: PO-3003\nBilled amount INR 33,000\n",
              "po_003.txt": "Purchase Order No: PO-3003\nApproval ID: APPR-3003\nItems: chairs\n",
              "approval_003.txt": "Approval ID: APPR-3003\nGranted by the finance head\n"},
             "SUPPORTED", ["invoice_003.txt", "po_003.txt", "approval_003.txt"], policy="An invoice requires a purchase order and a recorded approval of that order.",
             links=[("invoice_003.txt", "po_003.txt", "strong", "identifier PO-3003"), ("po_003.txt", "approval_003.txt", "strong", "identifier APPR-3003")],
             provenance=[("po_003.txt", "invoice_003.txt", "PURCHASE_ORDER_TO_INVOICE", "EXPLICIT"), ("approval_003.txt", "po_003.txt", "PURCHASE_ORDER_TO_APPROVAL", "EXPLICIT")]),
    _xb_case("XB-04-contradictory-amount", "contradictory_documents", "medium", "review billed amount",
             {"invoice_005.txt": "Invoice No: INV-5005\nPO Number: PO-5005\nVendor: Corvid Metals\nBilled amount INR 20,000\n",
              "po_005.txt": "Purchase Order No: PO-5005\nVendor: Corvid Metals\nOrder value INR 12,000\n"},
             "RECONCILIATION_REQUIRED", ["invoice_005.txt", "po_005.txt"], policy="Billed amount must not exceed the purchase order value.", contradicting=["po_005.txt"],
             links=[("invoice_005.txt", "po_005.txt", "strong", "identifier PO-5005 and vendor name")], contradictions=[("invoice_005.txt", "po_005.txt", "amount")],
             provenance=[("po_005.txt", "invoice_005.txt", "PURCHASE_ORDER_TO_INVOICE", "CONFLICTING")]),
    _xb_case("XB-05-contradictory-vendor", "contradictory_documents", "hard", "review billed amount",
             {"invoice_006.txt": "Invoice No: INV-6006\nPO Number: PO-6006\nVendor: Alder Foods\nBilled amount INR 5,000\n",
              "po_006.txt": "Purchase Order No: PO-6006\nVendor: Birch Foods\nOrder value INR 5,000\n"},
             "RECONCILIATION_REQUIRED", ["invoice_006.txt", "po_006.txt"], policy="The invoicing vendor must be the vendor named on the purchase order.", contradicting=["po_006.txt"],
             links=[("invoice_006.txt", "po_006.txt", "strong", "identifier PO-6006")], contradictions=[("invoice_006.txt", "po_006.txt", "vendor")],
             provenance=[("po_006.txt", "invoice_006.txt", "PURCHASE_ORDER_TO_INVOICE", "CONFLICTING")]),
    _xb_case("XB-06-missing-required-evidence", "missing_required_evidence", "medium", "review billed amount",
             {"invoice_007.txt": "Invoice No: INV-7007\nPO Number: PO-7777\nVendor: Dune Freight\nBilled amount INR 6,500\n"},
             "INSUFFICIENT_EVIDENCE", ["invoice_007.txt", "po_007.txt"], policy="An invoice requires the purchase order it references.",
             missing=[("purchase_order", "PO7777", "po_007.txt")]),
    _xb_case("XB-07-distractors", "distractor_documents", "medium", "review billed amount",
             {"invoice_008.txt": "Invoice No: INV-8008\nPO Number: PO-8008\nBilled amount INR 11,000\n",
              "po_008.txt": "Purchase Order No: PO-8008\nOrder value INR 11,000\nItems: laptops\n",
              "stray_008a.txt": "Payment UTR: UTR111222\nVendor: Globex Corp\nCatering for the picnic INR 300\n",
              "stray_008b.txt": "Meeting minutes\nAgenda: office plants\n",
              "stray_008c.txt": "Purchase Order No: PO-8999\nVendor: Hollis Paper\nOrder value INR 450\n"},
             "SUPPORTED", ["invoice_008.txt", "po_008.txt"], policy="An invoice must be backed by the purchase order it references.",
             links=[("invoice_008.txt", "po_008.txt", "strong", "identifier PO-8008")], distractors=["stray_008a.txt", "stray_008b.txt", "stray_008c.txt"],
             provenance=[("po_008.txt", "invoice_008.txt", "PURCHASE_ORDER_TO_INVOICE", "EXPLICIT")]),
    _xb_case("XB-08-identifier-invoice-payment", "identifier_entity_link", "easy", "review billed amount",
             {"invoice_009.txt": "Invoice No: INV-9009\nBilled amount INR 2,500\n",
              "payment_009.txt": "Payment UTR: UTR9009 settled against INV-9009\nSettled INR 2,500\n"},
             "SUPPORTED", ["invoice_009.txt", "payment_009.txt"], policy="An invoice must have a recorded payment that cites its invoice number.",
             links=[("invoice_009.txt", "payment_009.txt", "strong", "identifier INV-9009")],
             provenance=[("payment_009.txt", "invoice_009.txt", "INVOICE_TO_PAYMENT", "EXPLICIT")]),
    _xb_case("XB-09-date-and-amount-weak", "date_relationship", "medium", "review billed amount",
             {"invoice_010.txt": "Invoice No: INV-1010\nInvoice date 2024-06-15\nBilled amount INR 3,300\n",
              "payment_010.txt": "Payment advice\nDated 2024-06-15\nSettled INR 3,300\n"},
             "SUPPORTED", ["invoice_010.txt"], policy="An invoice states its billed amount.", links=[("invoice_010.txt", "payment_010.txt", "weak", "same date and amount, no identifier")],
             distractors=["payment_010.txt"]),
    _xb_case("XB-10-amount-only-unresolved", "amount_relationship", "medium", "review billed amount",
             {"invoice_011.txt": "Invoice No: INV-1111\nBilled amount INR 7,500\n", "payment_011.txt": "Payment advice\nSettled INR 7,500\n"},
             "SUPPORTED", ["invoice_011.txt"], policy="An invoice states its billed amount.", distractors=["payment_011.txt"]),
    _xb_case("XB-11-structured-field-person", "structured_field_relationship", "medium", "review billed amount",
             {"expense_012.txt": "Expense claim\nEmployee: Priya Nair\nBilled amount INR 2,100\n", "approval_012.txt": "Approval note\nEmployee: Priya Nair\nStatus: granted\n"},
             "SUPPORTED", ["expense_012.txt", "approval_012.txt"], policy="An expense claim requires an approval naming the same employee.",
             links=[("expense_012.txt", "approval_012.txt", "strong", "structured field Employee: Priya Nair")]),
    _xb_case("XB-12-semantic-similarity-only", "semantic_similarity", "hard", "review billed amount",
             {"invoice_013.txt": "Invoice for consulting services rendered to the client during the quarter\nBilled amount INR 90,000\n",
              "payment_013.txt": "Payment for consulting services rendered to the client during the quarter\nSettled in full\n"},
             "SUPPORTED", ["invoice_013.txt"], policy="An invoice states its billed amount.", links=[("invoice_013.txt", "payment_013.txt", "weak", "semantic similarity only")],
             distractors=["payment_013.txt"]),
    _xb_case("XB-13-weak-cross-type-token", "weak_unresolved_link", "hard", "review billed amount",
             {"invoice_014.txt": "Invoice No: INV-1414\nReference: ABC12345\nBilled amount INR 1,900\n", "payment_014.txt": "Payment UTR: ABC12345\nSettled by bank transfer\n"},
             "SUPPORTED", ["invoice_014.txt"], policy="An invoice states its billed amount.", links=[("invoice_014.txt", "payment_014.txt", "weak", "same token under different reference types")],
             distractors=["payment_014.txt"]),
    _xb_case("XB-14-unrelated-documents", "unrelated_documents", "easy", "review billed amount",
             {"invoice_015.txt": "Invoice No: INV-1500\nVendor: Quill Stationers\nBilled amount INR 800\n", "memo_015.txt": "Office memo\nThe plants in the east wing need watering twice weekly\n"},
             "SUPPORTED", ["invoice_015.txt"], policy="An invoice states its vendor and billed amount.", distractors=["memo_015.txt"]),
    _xb_case("XB-15-duplicate-billing-conflict", "contradictory_documents", "hard", "review billed amount",
             {"invoice_016a.txt": "Invoice No: INV-1601\nPO Number: PO-1601\nBilled amount INR 8,000\n",
              "invoice_016b.txt": "Invoice No: INV-1602\nPO Number: PO-1601\nBilled amount INR 8,000\n",
              "po_016.txt": "Purchase Order No: PO-1601\nOrder value INR 8,000\n"},
             "RECONCILIATION_REQUIRED", ["invoice_016a.txt", "invoice_016b.txt", "po_016.txt"], policy="A purchase order is billed by one invoice only.", contradicting=["invoice_016b.txt"],
             links=[("invoice_016a.txt", "po_016.txt", "strong", "identifier PO-1601"), ("invoice_016b.txt", "po_016.txt", "strong", "identifier PO-1601"), ("invoice_016a.txt", "invoice_016b.txt", "strong", "identifier PO-1601")],
             contradictions=[("invoice_016a.txt", "invoice_016b.txt", "invoice_id")],
             provenance=[("po_016.txt", "invoice_016a.txt", "PURCHASE_ORDER_TO_INVOICE", "EXPLICIT")]),
    _xb_case("XB-16-four-document-chain-with-distractors", "four_plus_document_chain", "hard", "review billed amount",
             {"po_017.txt": "Purchase Order No: PO-1701\nItems: monitors\n",
              "invoice_017.txt": "Invoice No: INV-1701\nPO Number: PO-1701\nBilled amount INR 14,000\n",
              "payment_017.txt": "Payment UTR: UTR1701 settled against INV-1701\nSettled INR 14,000\n",
              "approval_017.txt": "Approval ID: APPR-1701\nApproved payment UTR1701\n",
              "stray_017a.txt": "Meeting minutes\nAgenda: parking policy\n", "stray_017b.txt": "Payment UTR: UTR5555\nVendor: Nile Print\nSettled INR 220\n"},
             "SUPPORTED", ["po_017.txt", "invoice_017.txt", "payment_017.txt", "approval_017.txt"], policy="An invoice requires a purchase order, a payment that cites it, and an approval of that payment.",
             links=[("po_017.txt", "invoice_017.txt", "strong", "identifier PO-1701"), ("invoice_017.txt", "payment_017.txt", "strong", "identifier INV-1701"), ("payment_017.txt", "approval_017.txt", "strong", "identifier UTR1701")],
             distractors=["stray_017a.txt", "stray_017b.txt"],
             provenance=[("po_017.txt", "invoice_017.txt", "PURCHASE_ORDER_TO_INVOICE", "EXPLICIT"), ("payment_017.txt", "invoice_017.txt", "INVOICE_TO_PAYMENT", "EXPLICIT"),
                         ("approval_017.txt", "payment_017.txt", "PAYMENT_TO_APPROVAL", "EXPLICIT")]),
]

def validate_cross_document_benchmark(cases: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Structural consistency of the labels (unique ids, known decisions, every referenced file is a supplied document or a declared missing one). [] = consistent."""
    cases = CROSS_DOCUMENT_BENCHMARK if cases is None else cases
    probs: List[str] = []
    ids = [c.get("case_id") for c in cases]
    probs += [f"duplicate case_id {i}" for i, n in Counter(ids).items() if n > 1]
    for c in cases:
        cid, docs = c.get("case_id"), c.get("documents") or {}
        missing_files = {m["file"] for m in c.get("missing_evidence") or []}
        if c.get("expected_decision") not in XBENCH_DECISIONS: probs.append(f"{cid}: unknown expected_decision")
        for k in ("category", "difficulty", "objective", "required_evidence"):
            if not c.get(k): probs.append(f"{cid}: missing {k}")
        for k in ("required_evidence", "supporting_evidence", "contradicting_evidence"):
            for e in c.get(k) or []:
                if e["file"] not in docs and e["file"] not in missing_files: probs.append(f"{cid}: {k} file {e['file']} is neither a document nor declared missing")
        for k, keys in (("expected_entity_links", ("a", "b")), ("expected_contradictions", ("a", "b")), ("expected_provenance", ("file", "reached_from"))):
            for e in c.get(k) or []:
                for f in keys:
                    if e[f] not in docs: probs.append(f"{cid}: {k} references unknown document {e[f]}")
        probs += [f"{cid}: distractor {d} is not a document" for d in c.get("distractors") or [] if d not in docs]
        probs += [f"{cid}: missing file {f} is also supplied" for f in missing_files if f in docs]
    return probs

# ---- metric helpers ----
def _xb_r(x: Any) -> Any:
    return round(x, 4) if isinstance(x, float) else x

def _xb_prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    return {"precision": _xb_r(tp / (tp + fp)) if tp + fp else NOT_MEASURED, "recall": _xb_r(tp / (tp + fn)) if tp + fn else NOT_MEASURED,
            "f1": _xb_r(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else NOT_MEASURED}

def _xb_set_metrics(pred: set, exp: set) -> Dict[str, Any]:
    tp, fp, fn = len(pred & exp), len(pred - exp), len(exp - pred)
    return {"tp": tp, "fp": fp, "fn": fn, **_xb_prf(tp, fp, fn)}

def _xb_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _xb_mean(vals: List[Any]) -> Any:
    v = [x for x in vals if _xb_num(x)]
    return _xb_r(sum(v) / len(v)) if v else NOT_MEASURED

def _xb_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def _xb_files_of(G: nx.MultiDiGraph, ev: str) -> List[str]:
    return [s["filename"] for s in _evidence_source_docs(G, ev) if s.get("filename")]

def _xb_eg_links(G: nx.MultiDiGraph) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ent, d in _nodes_of_type(G, "Entity"):
        if d.get("entity_type") not in ("GovID", "Phone"): continue
        files = sorted({f for ev in _supporting_evidence_ids(G, ent) for f in _xb_files_of(G, ev)})
        strength = "strong" if (d.get("entity_type") == "GovID" and d.get("is_valid") is True) else "weak"
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                k = _xb_pair(files[i], files[j])
                if out.get(k, {}).get("strength") != "strong": out[k] = {"strength": strength, "basis": f"shared {d.get('entity_type')} entity", "status": "SHARED_ENTITY"}
    return out

def _xb_cd_links(G: nx.MultiDiGraph) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    for e in _xdoc_edges(G):
        k = _xb_pair(str(e.get("source_filename")), str(e.get("target_filename")))
        if e.get("link_status") == XDOC_UNRESOLVED:
            unresolved.append({"a": k[0], "b": k[1], "link_reason": e.get("link_reason")}); continue
        strength = "strong" if e.get("match_strength") == "strong" else "weak"
        if out.get(k, {}).get("strength") != "strong":
            out[k] = {"strength": strength, "basis": e.get("link_reason"), "status": e.get("link_status"), "relationship_type": e.get("relationship_type"), "match_methods": e.get("match_methods"), "semantic_only": bool(e.get("semantic_only"))}
    return out, unresolved

def _xb_eg_contradictions(G: nx.MultiDiGraph) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for u, v, d in G.edges(data=True):
        if d.get("relation") != "CONTRADICTS": continue
        other = d.get("counterpart_evidence_id") or (_supporting_evidence_ids(G, v) or [None])[0]
        if not other or not G.has_node(other): continue
        for fa in _xb_files_of(G, u):
            for fb in _xb_files_of(G, other):
                if fa == fb: continue
                out.setdefault(_xb_pair(fa, fb), {"source": "CONTRADICTS edge", "match_strength": d.get("match_strength"), "heuristic": bool(d.get("heuristic")), "label": d.get("label"), "is_violation": False,
                                                  "evidence": sorted(({"file": f, "location": G.nodes[e_].get("source_location"), "evidence": "stable key: file @ location"} for e_, f in ((u, fa), (other, fb))), key=lambda x: (str(x["file"]), str(x["location"])))})
    return out

def _xb_decide(required: List[str], retrieved_files: set, contra: Dict[Tuple[str, str], Any]) -> str:
    req = set(required)
    if any(a in req or b in req for a, b in contra): return "RECONCILIATION_REQUIRED"
    return "SUPPORTED" if req <= retrieved_files else "INSUFFICIENT_EVIDENCE"

def _xb_ev_loc(G: nx.MultiDiGraph, ev: str) -> Dict[str, Any]:
    return {"file": (_xb_files_of(G, ev) or [None])[0], "location": G.nodes[ev].get("source_location")}

def _xb_run_case(case: Dict[str, Any], workdir: str) -> List[Dict[str, Any]]:
    paths = []
    for name, text in case["documents"].items():
        p = os.path.join(workdir, name)
        with open(p, "w", encoding="utf-8") as fh: fh.write(text)
        paths.append(p)
    G = build_evidence_graph(paths, None)
    obj = case["objective"]
    inv = [n for n, _ in _investigation_evidence(G)]
    inv_set = set(inv)
    top = ranked_evidence_ids(G, obj)[:V2_TOP_K]
    expanded = [i["evidence_id"] for i in traverse_graph_context(G, top, V2_EXPANSION_LIMIT)["evidence"]] if top else []
    eg_ev = [e for e in dict.fromkeys(top + expanded) if e in inv_set]
    xd = traverse_cross_document_context(G, eg_ev, obj) if eg_ev else {"explicit_items": [], "weak_signals": [], "conflict_signals": [], "missing_references": [], "unresolved_signals": []}
    cd_ev = list(dict.fromkeys(eg_ev + [i["evidence_id"] for i in xd["explicit_items"]] + [s["evidence_id"] for s in xd["conflict_signals"] if s.get("evidence_id")]))
    payload = _build_v1_payload(paths, "", G)
    v1_files = {n for n in case["documents"] if f"FILE NAME: {n}\n" in payload}
    v1_ev = [e for e in inv if set(_xb_files_of(G, e)) & v1_files]
    eg_links, (cd_links, unresolved_links) = _xb_eg_links(G), _xb_cd_links(G)
    eg_contra = _xb_eg_contradictions(G)
    cd_contra = dict(eg_contra)
    for s in xd["conflict_signals"]:
        k = _xb_pair(str(s["filename"]), str(s["related_filename"]))
        cd_contra.setdefault(k, {"source": "CONFLICTING cross-document link", "link_reason": s["link_reason"], "conflicts": s["conflicts"], "is_violation": False, "requires_reconciliation": True,
                                 "evidence": sorted(({"file": p_.get("filename"), "location": p_.get("location"), "evidence": "stable key: file @ location"} for p_ in s["matched_evidence"]), key=lambda x: (str(x["file"]), str(x["location"])))})
    cd_links_all = {**eg_links, **cd_links}
    for k, v in eg_links.items():
        if v["strength"] == "strong": cd_links_all[k] = v
    obs_prov = [{"file": i["filename"], "reached_from": i["related_filename"], "relationship_type": i["relationship_type"], "link_status": i["link_status"], "link_reason": i["link_reason"],
                 "match_methods": i["match_methods"], "hops": i["hops"], "location": i["location"]} for i in xd["explicit_items"]]
    obs_prov += [{"file": s["filename"], "reached_from": s["related_filename"], "relationship_type": s["relationship_type"], "link_status": s["link_status"], "link_reason": s["link_reason"],
                  "match_methods": s["match_methods"], "hops": s.get("hops"), "location": s["location"]} for s in xd["conflict_signals"]]
    missing_pred = {(m["reference_type"], m["reference"]) for m in xd["missing_references"]}
    shared = dict(case=case, G=G, inv=inv)
    return [_xb_record(**shared, variant="V1", retrieved=v1_ev, links=None, contra=None, missing_pred=None, obs_prov=None,
                       notes=["V1 has no retrieval stage / graph: retrieved = documents present in the V1 payload; decision, links and contradictions are NOT_MEASURED (free-text LLM verdicts)"]),
            _xb_record(**shared, variant="EVIDENCE_GRAPH", retrieved=eg_ev, links=eg_links, contra=eg_contra, missing_pred=None, obs_prov=None,
                       notes=["missing-evidence detection and provenance relationships are Phase A/B features: NOT_MEASURED for this variant"]),
            _xb_record(**shared, variant="CROSS_DOCUMENT", retrieved=cd_ev, links=cd_links_all, contra=cd_contra, missing_pred=missing_pred, obs_prov=obs_prov,
                       notes=[f"weak-link documents surfaced (not counted as retrieved): {sorted({s['filename'] for s in xd['weak_signals']})}", f"unresolved links (not entity links): {unresolved_links}"])]

def _xb_record(case, G, inv, variant, retrieved, links, contra, missing_pred, obs_prov, notes) -> Dict[str, Any]:
    NM = NOT_MEASURED
    req = [e["file"] for e in case["required_evidence"]]
    rel = list(dict.fromkeys(e["file"] for k in ("required_evidence", "supporting_evidence", "contradicting_evidence") for e in case[k]))
    ret_set = set(retrieved)
    ret_files = sorted({f for e in retrieved for f in _xb_files_of(G, e)})
    pool = {e for e in inv if set(_xb_files_of(G, e)) & set(rel)}
    covered = lambda files: sum(1 for f in files if any(f in _xb_files_of(G, e) for e in ret_set))
    tp_ev = len(ret_set & pool)
    p = _xb_r(tp_ev / len(ret_set)) if ret_set else NM
    r = _xb_r(covered(rel) / len(rel)) if rel else NM
    f1 = _xb_r(2 * p * r / (p + r)) if _xb_num(p) and _xb_num(r) and (p + r) > 0 else (0.0 if _xb_num(p) and _xb_num(r) else NM)
    rec: Dict[str, Any] = {"case_id": case["case_id"], "system_variant": variant, "category": case["category"], "difficulty": case["difficulty"], "expected_decision": case["expected_decision"],
                           "required_documents": req, "retrieved_documents": ret_files, "evidence_precision": p, "evidence_recall": r, "evidence_f1": f1,
                           "evidence_completeness": _xb_r(covered(req) / len(req)) if req else NM, "evidence_counts": {"retrieved": len(ret_set), "true_positive": tp_ev, "relevant_entries": len(rel), "entries_covered": covered(rel), "required": len(req), "required_covered": covered(req)}}
    exp_links = {_xb_pair(e["a"], e["b"]): e["strength"] for e in case["expected_entity_links"]}
    exp_contra = {_xb_pair(e["a"], e["b"]) for e in case["expected_contradictions"]}
    rec["expected_entity_links"] = [{"a": k[0], "b": k[1], "strength": s} for k, s in sorted(exp_links.items())]
    rec["expected_contradictions"] = [{"a": a, "b": b} for a, b in sorted(exp_contra)]
    if links is None:
        rec.update(predicted_entity_links=NM, entity_link_precision=NM, entity_link_recall=NM, entity_link_f1=NM, entity_link_counts=NM, entity_link_strong=NM, entity_link_weak=NM,
                   predicted_decision=NM, decision_correct=NM, predicted_contradictions=NM, contradiction_precision=NM, contradiction_recall=NM, contradiction_f1=NM,
                   false_contradiction_rate=NM, missed_contradiction_rate=NM, contradiction_counts=NM)
    else:
        rec["predicted_entity_links"] = [{"a": k[0], "b": k[1], "strength": v["strength"], "status": v.get("status"), "basis": v.get("basis")} for k, v in sorted(links.items())]
        m = _xb_set_metrics(set(links), set(exp_links))
        rec.update(entity_link_precision=m["precision"], entity_link_recall=m["recall"], entity_link_f1=m["f1"], entity_link_counts={k: m[k] for k in ("tp", "fp", "fn")})
        for name, strength in (("entity_link_strong", "strong"), ("entity_link_weak", "weak")):  # weaker links are reported separately; semantic similarity alone is never strong
            sm = _xb_set_metrics({k for k, v in links.items() if v["strength"] == strength}, {k for k, s in exp_links.items() if s == strength})
            rec[name] = {k: sm[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
        rec["predicted_contradictions"] = [{"a": k[0], "b": k[1], "is_violation": False} for k in sorted(contra)]
        cm = _xb_set_metrics(set(contra), exp_contra)
        rec.update(contradiction_precision=cm["precision"], contradiction_recall=cm["recall"], contradiction_f1=cm["f1"],
                   false_contradiction_rate=_xb_r(cm["fp"] / len(contra)) if contra else NM, missed_contradiction_rate=_xb_r(cm["fn"] / len(exp_contra)) if exp_contra else NM,
                   contradiction_counts={k: cm[k] for k in ("tp", "fp", "fn")})
        rec["predicted_decision"] = _xb_decide(req, set(ret_files), contra)
        rec["decision_correct"] = rec["predicted_decision"] == case["expected_decision"]
    exp_missing = {(m_["reference_type"], m_["reference"]) for m_ in case["missing_evidence"]}
    mm = _xb_set_metrics(missing_pred, exp_missing) if missing_pred is not None else None
    rec["missing_evidence"] = {"expected": [{"reference_type": t, "reference": x} for t, x in sorted(exp_missing)],
                               "predicted": ([{"reference_type": t, "reference": x} for t, x in sorted(missing_pred)] if missing_pred is not None else NM),
                               "counts": ({k: mm[k] for k in ("tp", "fp", "fn")} if mm else NM), "precision": mm["precision"] if mm else NM, "recall": mm["recall"] if mm else NM}
    dis = case["distractors"]
    rec["distractors"] = {"expected": dis, "retrieved": sorted(set(dis) & set(ret_files)), "retrieval_rate": _xb_r(len(set(dis) & set(ret_files)) / len(dis)) if dis else NM}
    exp_prov = case["expected_provenance"]
    if obs_prov is None: rec["provenance"] = {"expected": exp_prov, "observed": NM, "expected_matched": NM, "recall": NM}
    else:
        key = lambda x: (x["file"], x["reached_from"], x["relationship_type"], x["link_status"])
        matched = sum(1 for e in exp_prov if key(e) in {key(o) for o in obs_prov})
        rec["provenance"] = {"expected": exp_prov, "observed": obs_prov, "expected_matched": matched, "recall": _xb_r(matched / len(exp_prov)) if exp_prov else NM}
        rec["provenance"]["contradiction_evidence"] = {f"{a}|{b}": v.get("evidence") for (a, b), v in sorted(contra.items())}
    rec["notes"] = notes
    return rec

# ---- aggregation (derived from the raw records only) ----
def _xb_pool(recs: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    rows = [r[field] for r in recs if isinstance(r.get(field), dict) and "tp" in r[field]]
    if not rows: return {"status": NOT_MEASURED, "reason": "no case produced this measurement"}
    tp, fp, fn = (sum(x[k] for x in rows) for k in ("tp", "fp", "fn"))
    return {"status": "MEASURED", "cases_measured": len(rows), "tp": tp, "fp": fp, "fn": fn, "micro": _xb_prf(tp, fp, fn)}

def _xb_aggregate_group(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(recs)
    meas = [r for r in recs if isinstance(r.get("decision_correct"), bool)]
    if meas:
        classes = {}
        for c in XBENCH_DECISIONS:
            tp = sum(1 for r in meas if r["expected_decision"] == c and r["predicted_decision"] == c)
            fp = sum(1 for r in meas if r["expected_decision"] != c and r["predicted_decision"] == c)
            fn = sum(1 for r in meas if r["expected_decision"] == c and r["predicted_decision"] != c)
            classes[c] = {"support": tp + fn, "predicted": tp + fp, **_xb_prf(tp, fp, fn)}
        used = [v for v in classes.values() if v["support"] or v["predicted"]]
        decision = {"status": "MEASURED", "cases_measured": len(meas), "accuracy": _xb_r(sum(r["decision_correct"] for r in meas) / len(meas)), "per_class": classes,
                    "macro_precision": _xb_mean([v["precision"] for v in used]), "macro_recall": _xb_mean([v["recall"] for v in used]), "macro_f1": _xb_mean([v["f1"] for v in used]),
                    "confusion": dict(Counter(f"{r['expected_decision']}->{r['predicted_decision']}" for r in meas))}
    else: decision = {"status": NOT_MEASURED, "reason": "decision not derivable for this variant (V1 verdicts are free LLM text) or no cases"}
    ec = [r["evidence_counts"] for r in recs]
    tp_ev, ret_n, ent_cov, ent_n, rq_cov, rq_n = (sum(c[k] for c in ec) for k in ("true_positive", "retrieved", "entries_covered", "relevant_entries", "required_covered", "required"))
    mp, mr = (tp_ev / ret_n if ret_n else NOT_MEASURED), (ent_cov / ent_n if ent_n else NOT_MEASURED)
    retrieval = {"status": "MEASURED" if ec else NOT_MEASURED, "micro": {"precision": _xb_r(mp), "recall": _xb_r(mr), "f1": _xb_r(2 * mp * mr / (mp + mr)) if _xb_num(mp) and _xb_num(mr) and mp + mr > 0 else NOT_MEASURED},
                 "macro": {"precision": _xb_mean([r["evidence_precision"] for r in recs]), "recall": _xb_mean([r["evidence_recall"] for r in recs]), "f1": _xb_mean([r["evidence_f1"] for r in recs])},
                 "evidence_completeness": {"micro": _xb_r(rq_cov / rq_n) if rq_n else NOT_MEASURED, "macro": _xb_mean([r["evidence_completeness"] for r in recs])}}
    ent = _xb_pool(recs, "entity_link_counts")
    if ent["status"] == "MEASURED":
        ent["macro"] = {k: _xb_mean([r[f"entity_link_{k}"] for r in recs]) for k in ("precision", "recall", "f1")}
        ent["strong"], ent["weak"] = _xb_pool(recs, "entity_link_strong"), _xb_pool(recs, "entity_link_weak")
    con = _xb_pool(recs, "contradiction_counts")
    if con["status"] == "MEASURED":
        con["macro"] = {k: _xb_mean([r[f"contradiction_{k}"] for r in recs]) for k in ("precision", "recall", "f1")}
        con["false_contradiction_rate"] = _xb_r(con["fp"] / (con["tp"] + con["fp"])) if con["tp"] + con["fp"] else NOT_MEASURED
        con["missed_contradiction_rate"] = _xb_r(con["fn"] / (con["tp"] + con["fn"])) if con["tp"] + con["fn"] else NOT_MEASURED
        none_exp = [r for r in recs if not r["expected_contradictions"] and isinstance(r["predicted_contradictions"], list)]
        con["case_false_positive_rate"] = _xb_r(sum(1 for r in none_exp if r["predicted_contradictions"]) / len(none_exp)) if none_exp else NOT_MEASURED
        con["note"] = "heuristic conflict signals; none is a compliance violation"
    me = [r["missing_evidence"] for r in recs if isinstance(r["missing_evidence"].get("counts"), dict)]
    miss = {"status": NOT_MEASURED, "reason": "variant has no missing-reference mechanism"} if not me else {"status": "MEASURED", "cases_measured": len(me), **{k: sum(x["counts"][k] for x in me) for k in ("tp", "fp", "fn")}}
    if me: miss["micro"] = _xb_prf(miss["tp"], miss["fp"], miss["fn"])
    dis_exp = sum(len(r["distractors"]["expected"]) for r in recs)
    prv = [r["provenance"] for r in recs if _xb_num(r["provenance"].get("expected_matched"))]
    pe = sum(len(x["expected"]) for x in prv)
    return {"cases": n, "decision": decision, "retrieval": retrieval, "entity_links": ent if ent["status"] == "MEASURED" else {"status": NOT_MEASURED, "reason": "variant produces no document links"},
            "contradictions": con if con["status"] == "MEASURED" else {"status": NOT_MEASURED, "reason": "variant produces no contradiction signals"}, "missing_evidence": miss,
            "distractors": {"expected": dis_exp, "retrieved": sum(len(r["distractors"]["retrieved"]) for r in recs), "retrieval_rate": _xb_r(sum(len(r["distractors"]["retrieved"]) for r in recs) / dis_exp) if dis_exp else NOT_MEASURED},
            "provenance_relationships": {"status": "MEASURED", "expected": pe, "matched": sum(x["expected_matched"] for x in prv), "recall": _xb_r(sum(x["expected_matched"] for x in prv) / pe) if pe else NOT_MEASURED} if prv else {"status": NOT_MEASURED, "reason": "variant has no relationship provenance"}}

def aggregate_cross_document_results(records: List[Dict[str, Any]], cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cases = CROSS_DOCUMENT_BENCHMARK if cases is None else cases
    by_var = {v: [r for r in records if r["system_variant"] == v] for v in XBENCH_VARIANTS}
    agg = {v: _xb_aggregate_group(rs) for v, rs in by_var.items()}
    per_cat = {v: {c: _xb_aggregate_group([r for r in rs if r["category"] == c]) for c in sorted({r["category"] for r in rs})} for v, rs in by_var.items()}
    def pick(a, path):
        for k in path: a = a.get(k) if isinstance(a, dict) else None
        return a if a is not None else NOT_MEASURED
    metrics = {"decision_accuracy": ("decision", "accuracy"), "decision_macro_f1": ("decision", "macro_f1"), "retrieval_micro_f1": ("retrieval", "micro", "f1"), "retrieval_macro_f1": ("retrieval", "macro", "f1"),
               "evidence_completeness_micro": ("retrieval", "evidence_completeness", "micro"), "entity_link_micro_f1": ("entity_links", "micro", "f1"), "contradiction_micro_f1": ("contradictions", "micro", "f1"),
               "false_contradiction_rate": ("contradictions", "false_contradiction_rate"), "missed_contradiction_rate": ("contradictions", "missed_contradiction_rate")}
    comparison: Dict[str, Any] = {"order": list(XBENCH_VARIANTS), "note": "same benchmark cases for every variant; raw values only, no superiority claim; a delta exists only where both values are measured"}
    for name, path in metrics.items():
        vals = [pick(agg[v], path) for v in XBENCH_VARIANTS]
        comparison[name] = {"values": dict(zip(XBENCH_VARIANTS, vals)), "delta_EG_vs_V1": _xb_r(vals[1] - vals[0]) if _xb_num(vals[0]) and _xb_num(vals[1]) else NOT_MEASURED,
                            "delta_CD_vs_EG": _xb_r(vals[2] - vals[1]) if _xb_num(vals[1]) and _xb_num(vals[2]) else NOT_MEASURED}
    return {"benchmark": {"id": XBENCH_ID, "version": XBENCH_VERSION, "size": len(cases), "category_counts": dict(Counter(c["category"] for c in cases)), "difficulty_counts": dict(Counter(c["difficulty"] for c in cases)),
                          "expected_decision_counts": dict(Counter(c["expected_decision"] for c in cases)), "seed": None},
            "variants": agg, "per_category": per_cat, "comparison": comparison, "protocol": XBENCH_PROTOCOL}

def run_cross_document_benchmark(cases: Optional[List[Dict[str, Any]]] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Runs the benchmark offline (no LLM / network). Returns {"records": raw per-case/per-variant results, "aggregate": derived from them}; writes JSON only if output_path is given."""
    import tempfile
    cases = CROSS_DOCUMENT_BENCHMARK if cases is None else cases
    probs = validate_cross_document_benchmark(cases)
    if probs: raise ValueError("benchmark labels inconsistent: " + "; ".join(probs[:5]))
    records: List[Dict[str, Any]] = []
    for c in cases:
        with tempfile.TemporaryDirectory() as td: records += _xb_run_case(c, td)
    result = _json_safe({"benchmark_id": XBENCH_ID, "version": XBENCH_VERSION, "records": records, "aggregate": aggregate_cross_document_results(records, cases)})
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh: json.dump(result, fh, indent=2, ensure_ascii=False)
    return result

# --- CONTRADICTION BENCHMARK + EVALUATION (extends the cross-document benchmark; scores classify_contradictions against deterministic labels) ---
# A "contradiction" = a finding in MINOR_/MAJOR_CONTRADICTION. Findings are matched by (documents, field). Precision / recall / F1 / false- and missed-contradiction rates use that
# match; severity accuracy = among expected contradictions that were detected, the share classified with the labelled MINOR/MAJOR category; category accuracy covers every expected finding
# (all six categories). NOT_MEASURED when a denominator is empty. No LLM, no randomness. Labels are written from the document semantics, not from system output.
def _xc_case(cid, category, difficulty, docs, findings, rulebook=None):
    return {"case_id": cid, "category": category, "difficulty": difficulty, "documents": dict(docs), "rulebook": rulebook,
            "expected_findings": [{"a": a, "b": b, "field": f, "category": c} for a, b, f, c in findings]}

CONTRADICTION_BENCHMARK: List[Dict[str, Any]] = [
    _xc_case("XC-01-consistent", "consistent", "easy",
             {"invoice_c01.txt": "Invoice No: INV-C011\nPO Number: PO-C011\nVendor: Aster Foods\nBilled amount INR 5,000\nQuantity: 10\n", "po_c01.txt": "Purchase Order No: PO-C011\nVendor: Aster Foods\nOrder value INR 5,000\nQuantity: 10\n"},
             [("invoice_c01.txt", "po_c01.txt", "*", "CONSISTENT")]),
    _xc_case("XC-02-amount-major", "amount_contradiction", "easy",
             {"invoice_c02.txt": "Invoice No: INV-C021\nPO Number: PO-C021\nVendor: Boreal Metals\nBilled amount INR 20,000\n", "po_c02.txt": "Purchase Order No: PO-C021\nVendor: Boreal Metals\nOrder value INR 12,000\n"},
             [("invoice_c02.txt", "po_c02.txt", "amount", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-03-amount-minor", "amount_contradiction", "medium",
             {"invoice_c03.txt": "Invoice No: INV-C031\nPO Number: PO-C031\nVendor: Boreal Metals\nBilled amount INR 10,200\n", "po_c03.txt": "Purchase Order No: PO-C031\nVendor: Boreal Metals\nOrder value INR 10,000\n"},
             [("invoice_c03.txt", "po_c03.txt", "amount", "MINOR_CONTRADICTION")]),
    _xc_case("XC-04-legitimate-partial-payment", "legitimate_partial_payment", "medium",
             {"invoice_c04.txt": "Invoice No: INV-4041\nVendor: Cobalt Tools\nBilled amount INR 20,000\n", "payment_c04.txt": "Payment UTR: UTR-4041 partial payment against INV-4041\nSettled INR 8,000\n"},
             [("invoice_c04.txt", "payment_c04.txt", "amount", "CONSISTENT")]),
    _xc_case("XC-05-date-contradiction", "date_contradiction", "medium",
             {"invoice_c05.txt": "Invoice No: INV-5051\nInvoice date: 2024-03-10\nBilled amount INR 3,000\n", "payment_c05.txt": "Payment UTR: UTR-5051 against INV-5051\nInvoice date: 2024-04-20\nSettled INR 3,000\n"},
             [("invoice_c05.txt", "payment_c05.txt", "date", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-06-vendor-major", "vendor_contradiction", "easy",
             {"invoice_c06.txt": "Invoice No: INV-C061\nPO Number: PO-C061\nVendor: Alder Foods\nBilled amount INR 5,000\n", "po_c06.txt": "Purchase Order No: PO-C061\nVendor: Birch Foods\nOrder value INR 5,000\n"},
             [("invoice_c06.txt", "po_c06.txt", "vendor", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-07-vendor-minor-variant", "vendor_contradiction", "hard",
             {"invoice_c07.txt": "Invoice No: INV-C071\nPO Number: PO-C071\nVendor: Birch Foods\nBilled amount INR 5,000\n", "po_c07.txt": "Purchase Order No: PO-C071\nVendor: Birch Foods India\nOrder value INR 5,000\n"},
             [("invoice_c07.txt", "po_c07.txt", "vendor", "MINOR_CONTRADICTION")]),
    _xc_case("XC-08-identity-person", "identity_contradiction", "medium",
             {"claim_c08.txt": "Expense claim\nReference: REF-C0801\nEmployee: Priya Nair\nBilled amount INR 2,100\n", "approval_c08.txt": "Approval note\nReference: REF-C0801\nEmployee: Rahul Menon\nStatus: granted\n"},
             [("approval_c08.txt", "claim_c08.txt", "identity", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-09-transaction-id-major", "transaction_id_contradiction", "medium",
             {"invoice_c09.txt": "Invoice No: INV-C091\nPO Number: PO-C090\nBilled amount INR 4,000\n", "payment_c09.txt": "Payment UTR: UTR-C091\nPO Number: PO-C090\nInvoice No: INV-X778\nSettled INR 4,000\n"},
             [("invoice_c09.txt", "payment_c09.txt", "transaction_id", "MAJOR_CONTRADICTION"), ("payment_c09.txt", None, "reference", "MISSING_EVIDENCE")]),
    _xc_case("XC-10-transaction-id-typo", "transaction_id_contradiction", "hard",
             {"invoice_c10.txt": "Invoice No: INV-C101\nPO Number: PO-C100\nBilled amount INR 4,000\n", "payment_c10.txt": "Payment UTR: UTR-C101\nPO Number: PO-C100\nInvoice No: INV-C102\nSettled INR 4,000\n"},
             [("invoice_c10.txt", "payment_c10.txt", "transaction_id", "MINOR_CONTRADICTION"), ("payment_c10.txt", None, "reference", "MISSING_EVIDENCE")]),
    _xc_case("XC-11-approval-contradiction", "approval_contradiction", "medium",
             {"po_c11.txt": "Purchase Order No: PO-C110\nApproval ID: APPR-C11\nApproval status: approved\n", "approval_c11.txt": "Approval ID: APPR-C11\nApproval status: rejected\n"},
             [("approval_c11.txt", "po_c11.txt", "approval_status", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-12-factual-quantity", "factual_claim_contradiction", "medium",
             {"invoice_c12.txt": "Invoice No: INV-C121\nPO Number: PO-C121\nQuantity: 10\nBilled amount INR 1,000\n", "po_c12.txt": "Purchase Order No: PO-C121\nQuantity: 25\nOrder value INR 1,000\n"},
             [("invoice_c12.txt", "po_c12.txt", "factual_claim", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-13-factual-delivery", "factual_claim_contradiction", "medium",
             {"invoice_c13.txt": "Invoice No: INV-C131\nPO Number: PO-C131\nDelivery status: delivered\nBilled amount INR 1,500\n", "po_c13.txt": "Purchase Order No: PO-C131\nDelivery status: not delivered\nOrder value INR 1,500\n"},
             [("invoice_c13.txt", "po_c13.txt", "factual_claim", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-14-missing-evidence", "missing_evidence", "medium",
             {"invoice_c14.txt": "Invoice No: INV-C141\nPO Number: PO-C149\nVendor: Dune Freight\nBilled amount INR 6,500\n"},
             [("invoice_c14.txt", None, "reference", "MISSING_EVIDENCE")]),
    _xc_case("XC-15-unresolved-payment", "unresolved_ambiguity", "hard",
             {"invoice_c15.txt": "Invoice No: INV-1511\nBilled amount INR 9,000\n", "payment_c15.txt": "Payment UTR: UTR-1511 against INV-1511\nSettled INR 4,000\n"},
             [("invoice_c15.txt", "payment_c15.txt", "amount", "UNRESOLVED")]),
    _xc_case("XC-16-multiple-contradictions", "multiple_contradictions", "hard",
             {"invoice_c16.txt": "Invoice No: INV-C161\nPO Number: PO-C161\nVendor: Alder Foods\nQuantity: 10\nBilled amount INR 20,000\n", "po_c16.txt": "Purchase Order No: PO-C161\nVendor: Birch Foods\nQuantity: 25\nOrder value INR 12,000\n"},
             [("invoice_c16.txt", "po_c16.txt", "vendor", "MAJOR_CONTRADICTION"), ("invoice_c16.txt", "po_c16.txt", "amount", "MAJOR_CONTRADICTION"), ("invoice_c16.txt", "po_c16.txt", "factual_claim", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-17-distractors", "distractors", "hard",
             {"invoice_c17.txt": "Invoice No: INV-C171\nPO Number: PO-C171\nVendor: Boreal Metals\nBilled amount INR 20,000\n", "po_c17.txt": "Purchase Order No: PO-C171\nVendor: Boreal Metals\nOrder value INR 12,000\n",
              "stray_c17a.txt": "Payment UTR: UTR-Z1\nVendor: Globex Corp\nSettled INR 900\n", "stray_c17b.txt": "Meeting minutes\nAgenda: office plants\n", "stray_c17c.txt": "Invoice No: INV-Z9\nVendor: Globex Corp\nBilled amount INR 77\n"},
             [("invoice_c17.txt", "po_c17.txt", "amount", "MAJOR_CONTRADICTION")]),
    _xc_case("XC-18-semantic-only", "semantic_only", "hard",
             {"invoice_c18.txt": "Invoice for consulting services rendered to the client during the quarter\nBilled amount INR 90,000\n", "payment_c18.txt": "Payment for consulting services rendered to the client during the quarter\nSettled INR 80,000\n"},
             []),
    _xc_case("XC-19-policy-violation-existing-decision", "policy_violation", "medium",
             {"invoice_c19.txt": "Invoice No: INV-C191\nBilled amount INR 5,000\n"},
             [("invoice_c19.txt", None, "policy", "POLICY_VIOLATION")], rulebook="FORBID TRANSACTION > INR 1000\n"),
    _xc_case("XC-20-contradiction-is-not-a-violation", "contradiction_not_violation", "hard",
             {"invoice_c20.txt": "Invoice No: INV-C201\nPO Number: PO-C201\nVendor: Boreal Metals\nBilled amount INR 20,000\n", "po_c20.txt": "Purchase Order No: PO-C201\nVendor: Boreal Metals\nOrder value INR 12,000\n"},
             [("invoice_c20.txt", "po_c20.txt", "amount", "MAJOR_CONTRADICTION")], rulebook="FORBID TRANSACTION > INR 100000\n"),
]

def validate_contradiction_benchmark(cases: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    cases = CONTRADICTION_BENCHMARK if cases is None else cases
    probs = [f"duplicate case_id {i}" for i, n in Counter(c["case_id"] for c in cases).items() if n > 1]
    for c in cases:
        for f in c["expected_findings"]:
            if f["category"] not in XCON_CATEGORIES: probs.append(f"{c['case_id']}: unknown category {f['category']}")
            for k in ("a", "b"):
                if f[k] is not None and f[k] not in c["documents"]: probs.append(f"{c['case_id']}: {k}={f[k]} is not a document")
    return probs

def _xc_key(files: List[str], field: str) -> Tuple[Tuple[str, ...], str]:
    return (tuple(sorted(set(f for f in files if f))), field)

def _xc_predicted(G: nx.MultiDiGraph) -> Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]]:
    out: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = {}
    for f in G.graph.get("contradiction_findings") or []:
        files = f["files"] if f.get("files") else [s["filename"] for s in (f["claim_a"], f["claim_b"]) if s]
        k = _xc_key(files, f["field"])
        if k not in out or _XCON_PRIO[f["category"]] < _XCON_PRIO[out[k]["category"]]:
            out[k] = {"category": f["category"], "severity": f["severity"], "finding_id": f["finding_id"], "reason": f["reason"], "is_policy_violation": f["is_policy_violation"],
                      "claim_a": f["claim_a"] and {"filename": f["claim_a"]["filename"], "value": ({k: v for k, v in f["claim_a"]["value"].items() if not k.endswith("_id")} if isinstance(f["claim_a"]["value"], dict) else f["claim_a"]["value"]), "provenance": [{"file": p.get("filename"), "location": p.get("location")} for p in f["claim_a"]["provenance"]]},
                      "claim_b": f["claim_b"] and {"filename": f["claim_b"]["filename"], "value": ({k: v for k, v in f["claim_b"]["value"].items() if not k.endswith("_id")} if isinstance(f["claim_b"]["value"], dict) else f["claim_b"]["value"]), "provenance": [{"file": p.get("filename"), "location": p.get("location")} for p in f["claim_b"]["provenance"]]}}
    return out

def _xc_run_case(case: Dict[str, Any], workdir: str) -> Dict[str, Any]:
    paths = []
    for name, text in case["documents"].items():
        p = os.path.join(workdir, name)
        with open(p, "w", encoding="utf-8") as fh: fh.write(text)
        paths.append(p)
    rb = None
    if case.get("rulebook"):
        rb = os.path.join(workdir, "rulebook.txt")
        with open(rb, "w", encoding="utf-8") as fh: fh.write(case["rulebook"])
    G = build_evidence_graph(paths, rb)
    pred = _xc_predicted(G)
    exp = {_xc_key([f["a"], f["b"]], f["field"]): f["category"] for f in case["expected_findings"]}
    exp_c = {k for k, c in exp.items() if c in XCON_CONTRADICTION}
    pred_c = {k for k, v in pred.items() if v["category"] in XCON_CONTRADICTION}
    m = _xb_set_metrics(pred_c, exp_c)
    detected = exp_c & pred_c
    sev_ok = sum(1 for k in detected if pred[k]["category"] == exp[k])
    cat_ok = sum(1 for k, c in exp.items() if pred.get(k, {}).get("category") == c)
    fp_pv = [k for k, v in pred.items() if v["category"] == "POLICY_VIOLATION" and exp.get(k) != "POLICY_VIOLATION"]
    return {"case_id": case["case_id"], "category": case["category"], "difficulty": case["difficulty"],
            "expected_findings": [{"files": list(k[0]), "field": k[1], "category": c} for k, c in sorted(exp.items())],
            "predicted_findings": [{"files": list(k[0]), "field": k[1], **v} for k, v in sorted(pred.items())],
            "contradiction_precision": m["precision"], "contradiction_recall": m["recall"], "contradiction_f1": m["f1"],
            "false_contradiction_rate": _xb_r(m["fp"] / len(pred_c)) if pred_c else NOT_MEASURED, "missed_contradiction_rate": _xb_r(m["fn"] / len(exp_c)) if exp_c else NOT_MEASURED,
            "counts": {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "predicted_contradictions": len(pred_c), "expected_contradictions": len(exp_c), "severity_correct": sev_ok, "severity_total": len(detected),
                       "category_correct": cat_ok, "category_total": len(exp), "false_policy_violations": len(fp_pv)},
            "severity_accuracy": _xb_r(sev_ok / len(detected)) if detected else NOT_MEASURED, "category_accuracy": _xb_r(cat_ok / len(exp)) if exp else NOT_MEASURED,
            "unexpected_findings": [{"files": list(k[0]), "field": k[1], "category": v["category"]} for k, v in sorted(pred.items()) if k not in exp and v["category"] in XCON_CONTRADICTION + ("POLICY_VIOLATION",)],
            "contradictions_flagged_as_violation": sum(1 for v in pred.values() if v["category"] in XCON_CONTRADICTION and v["is_policy_violation"])}

def _xc_group(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs: return {"cases": 0, "status": NOT_MEASURED}
    s = lambda k: sum(r["counts"][k] for r in recs)
    tp, fp, fn = s("tp"), s("fp"), s("fn")
    pooled = _xb_prf(tp, fp, fn)
    per_cat: Dict[str, Any] = {}
    for c in XCON_CATEGORIES:
        exp = sum(1 for r in recs for f in r["expected_findings"] if f["category"] == c)
        hit = sum(1 for r in recs for f in r["expected_findings"] if f["category"] == c and any(p["files"] == f["files"] and p["field"] == f["field"] and p["category"] == c for p in r["predicted_findings"]))
        per_cat[c] = {"expected": exp, "correctly_classified": hit, "recall": _xb_r(hit / exp) if exp else NOT_MEASURED}
    return {"cases": len(recs), "status": "MEASURED",
            "contradiction": {"tp": tp, "fp": fp, "fn": fn, "precision": pooled["precision"], "recall": pooled["recall"], "f1": pooled["f1"],
                              "macro": {k: _xb_mean([r[f"contradiction_{k}"] for r in recs]) for k in ("precision", "recall", "f1")},
                              "false_contradiction_rate": _xb_r(fp / (tp + fp)) if tp + fp else NOT_MEASURED, "missed_contradiction_rate": _xb_r(fn / (tp + fn)) if tp + fn else NOT_MEASURED,
                              "case_false_positive_rate": (lambda ne: _xb_r(sum(1 for r in ne if r["counts"]["predicted_contradictions"]) / len(ne)) if ne else NOT_MEASURED)([r for r in recs if not r["counts"]["expected_contradictions"]])},
            "severity_classification": {"accuracy": _xb_r(s("severity_correct") / s("severity_total")) if s("severity_total") else NOT_MEASURED, "correct": s("severity_correct"), "detected_contradictions": s("severity_total")},
            "category_classification": {"accuracy": _xb_r(s("category_correct") / s("category_total")) if s("category_total") else NOT_MEASURED, "correct": s("category_correct"), "expected_findings": s("category_total"), "per_category": per_cat},
            "false_policy_violations": s("false_policy_violations"), "contradictions_flagged_as_violation": sum(r["contradictions_flagged_as_violation"] for r in recs)}

def aggregate_contradiction_results(records: List[Dict[str, Any]], cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cases = CONTRADICTION_BENCHMARK if cases is None else cases
    return {"benchmark": {"id": "omnicheck-contradiction-bench", "version": "1.0", "size": len(cases), "category_counts": dict(Counter(c["category"] for c in cases)), "difficulty_counts": dict(Counter(c["difficulty"] for c in cases)),
                          "expected_category_counts": dict(Counter(f["category"] for c in cases for f in c["expected_findings"])), "seed": None, "llm_used": False},
            "overall": _xc_group(records), "per_category": {c: _xc_group([r for r in records if r["category"] == c]) for c in sorted({r["category"] for r in records})},
            "scope_note": "Evaluates the deterministic classifier only. Severity is measured only for expected contradictions that were detected. Heuristic contradictions are never compliance violations."}

def run_contradiction_benchmark(cases: Optional[List[Dict[str, Any]]] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    import tempfile
    cases = CONTRADICTION_BENCHMARK if cases is None else cases
    probs = validate_contradiction_benchmark(cases)
    if probs: raise ValueError("contradiction benchmark labels inconsistent: " + "; ".join(probs[:5]))
    records = []
    for c in cases:
        with tempfile.TemporaryDirectory() as td: records.append(_xc_run_case(c, td))
    result = _json_safe({"records": records, "aggregate": aggregate_contradiction_results(records, cases)})
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh: json.dump(result, fh, indent=2, ensure_ascii=False)
    return result

# --- SELF-VERIFICATION EVALUATION BENCHMARK (Phase 7D; evaluation layer only: 7A-7C, the policy engine, contradiction detection and lineage are REUSED unchanged) ---
# Deterministic labelled cases; each case builds its OWN private graph with the existing offline pipeline (the production graph is never touched). The label block of a case (expected_*) is written by hand and is
# NEVER read from system output; predictions are produced by the existing functions and scored against the labels afterwards. No LLM, no network, no randomness (seed: none).
# Variants (all start from the SAME initial engine Decision; none changes it):
#   BASELINE                                                   = the initial Decision verdict only: no evidence lineage, no grounding / contradiction / verification mechanism.
#   EVIDENCE_GRAPH                                             = verdict + 7A lineage (supporting evidence ids / provenance, contradiction findings); evidence EXISTENCE is its only grounding signal; status NOT_MEASURED.
#   EVIDENCE_GRAPH_SELF_VERIFICATION                           = verify_self_verification_result (7B): FAILED -> conclusion rejected (NO_CONCLUSION), ESCALATE -> ESCALATE, VERIFIED / non-asserting verdict as-is.
#   EVIDENCE_GRAPH_SELF_VERIFICATION_POLICY_APPLICABILITY      = verify_policy_applicability (7C), same mapping.
# Ground truth per case: expected_outcome (VIOLATION | SATISFIED | NO_CONCLUSION | ESCALATE = the correct SUPPORTABLE handling of the case), decision_supported, claims_grounded (None = not applicable),
# expected_major_contradiction (None = not assessed), expected_verification_status, policy_applicable. Escalation truth = (expected_verification_status == ESCALATE).
# Metrics are {value, numerator, denominator}; value is NOT_MEASURED (never 0) when the denominator is empty or the variant has no such mechanism.
SV7D_ID, SV7D_VERSION = "omnicheck-selfverification-bench", "1.0"
SV7D_VARIANTS = ("BASELINE", "EVIDENCE_GRAPH", "EVIDENCE_GRAPH_SELF_VERIFICATION", "EVIDENCE_GRAPH_SELF_VERIFICATION_POLICY_APPLICABILITY")
SV7D_OUTCOMES = ("VIOLATION", "SATISFIED", "NO_CONCLUSION", "ESCALATE")
SV7D_CATEGORIES = ("positive", "negative", "ambiguous", "missing_evidence", "contradiction", "policy_applicability", "wrong_but_supported", "unsupported")
SV7D_PROTOCOL = {"seed": None, "randomness": "none", "llm_used": False, "network_used": False, "confidence": "NOT_MEASURED (None) everywhere; never aggregated or invented",
                 "ground_truth": "hand-written per-case labels, independent of every system output; the benchmark never derives a label from a prediction",
                 "faults": "a case may inject a declared fault into its OWN private graph copy (e.g. evidence text no longer contains the claimed value) to create a wrong-but-superficially-supported decision or a grounded-but-inapplicable rule",
                 "metrics_not_measured_when": "the denominator is empty or the variant has no such mechanism (BASELINE has no evidence / verification; EVIDENCE_GRAPH has no verification status)",
                 "id_policy": "records keep evidence provenance and stable aliases; real (random) graph ids are verified against the graph at run time (id_integrity) rather than stored, so records are reproducible"}
_SV7D_FAULTS = ("evidence_text_5000_to_500", "evidence_text_to_1", "heuristic_support", "remove_support", "rule_condition_9000", "basis_amount_none", "wrong_scope", "weak_location")
_SV7D_RB = "FORBID TRANSACTION > INR 1000\n"
_SV7D_INV = "Invoice No: INV-{n}\nBilled amount INR {amt}\n"
_SV7D_CON_INV, _SV7D_CON_PO = "Invoice No: INV-1101\nPO Number: PO-1101\nVendor: Boreal Metals\nBilled amount INR 20,000\n", "Purchase Order No: PO-1101\nVendor: Boreal Metals\nOrder value INR {v}\n"

def _sv7d_case(cid, category, docs, rulebook, target, outcome, status, supported, grounded, contradiction, applicable=True, fault=None, note=""):
    return {"case_id": cid, "category": category, "documents": dict(docs), "rulebook": rulebook, "target_rule": target, "fault": fault, "note": note,
            "expected_outcome": outcome, "expected_verification_status": status, "decision_supported": supported, "claims_grounded": grounded, "expected_major_contradiction": contradiction, "policy_applicable": applicable}

SELF_VERIFICATION_BENCHMARK: List[Dict[str, Any]] = [
    _sv7d_case("sv7d_pos_violation", "positive", {"invoice_a.txt": _SV7D_INV.format(n=9001, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "VIOLATION", "VERIFIED", True, True, False, note="grounded violation of an applicable rule"),
    _sv7d_case("sv7d_pos_satisfied", "positive", {"invoice_b.txt": _SV7D_INV.format(n=9002, amt="5,000")}, "FORBID TRANSACTION > INR 10000\n", "FORBID TRANSACTION", "SATISFIED", "VERIFIED", True, True, False),
    _sv7d_case("sv7d_neg_under_limit", "negative", {"invoice_c.txt": _SV7D_INV.format(n=9003, amt="800")}, _SV7D_RB, "FORBID TRANSACTION", "SATISFIED", "VERIFIED", True, True, False, note="no violation exists; a pass is supported"),
    _sv7d_case("sv7d_pos_absence_in_scope", "positive", {"memo_a.txt": "Meeting notes\nNothing else to report\n"}, 'REQUIRE KEYWORD "approval"\n', "REQUIRE KEYWORD", "VIOLATION", "VERIFIED", True, True, False, note="absence holds within the extracted scope"),
    _sv7d_case("sv7d_ambiguous_free_text_rule", "ambiguous", {"invoice_d.txt": _SV7D_INV.format(n=9004, amt="5,000")}, _SV7D_RB + "Vendors should behave reasonably in spirit.\n", "Vendors should behave", "NO_CONCLUSION", "NOT_VERIFIED", False, None, None, note="rule not expressible deterministically: no conclusion"),
    _sv7d_case("sv7d_missing_unlinked_records", "missing_evidence", {"invoice_e.txt": _SV7D_CON_INV, "po_e.txt": _SV7D_CON_PO.format(v="12,000")}, "Transaction amounts must match.\n", "amounts must match", "NO_CONCLUSION", "NOT_VERIFIED", False, None, None, note="records not reliably linked: nothing may be compared"),
    _sv7d_case("sv7d_contradiction_major", "contradiction", {"invoice_f.txt": _SV7D_CON_INV, "po_f.txt": _SV7D_CON_PO.format(v="12,000")}, "FORBID TRANSACTION > INR 15000\n", "FORBID TRANSACTION", "ESCALATE", "ESCALATE", False, True, True, note="grounded claims, unresolved major cross-document contradiction"),
    _sv7d_case("sv7d_contradiction_consistent", "contradiction", {"invoice_g.txt": _SV7D_CON_INV, "po_g.txt": _SV7D_CON_PO.format(v="20,000")}, "FORBID TRANSACTION > INR 15000\n", "FORBID TRANSACTION", "VIOLATION", "VERIFIED", True, True, False, note="documents agree: nothing to escalate"),
    _sv7d_case("sv7d_wrong_but_supported_text", "wrong_but_supported", {"invoice_h.txt": _SV7D_INV.format(n=9005, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "NO_CONCLUSION", "FAILED", False, False, None, fault="evidence_text_5000_to_500",
                 note="initial VIOLATION has linked evidence, but the source text says 500: superficially supported, wrong"),
    _sv7d_case("sv7d_wrong_but_supported_heuristic", "wrong_but_supported", {"invoice_i.txt": _SV7D_INV.format(n=9006, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "NO_CONCLUSION", "FAILED", False, False, None, fault="heuristic_support",
                 note="evidence is linked only by heuristic edges: not grounding"),
    _sv7d_case("sv7d_unsupported_no_evidence", "unsupported", {"invoice_j.txt": _SV7D_INV.format(n=9007, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "NO_CONCLUSION", "FAILED", False, False, None, fault="remove_support", note="decision asserted with no supporting evidence"),
    _sv7d_case("sv7d_failed_beats_escalate", "unsupported", {"invoice_k.txt": _SV7D_CON_INV, "po_k.txt": _SV7D_CON_PO.format(v="12,000")}, "FORBID TRANSACTION > INR 15000\n", "FORBID TRANSACTION", "NO_CONCLUSION", "FAILED", False, False, None, fault="evidence_text_to_1",
                 note="contradiction present AND claim ungrounded: FAILED takes precedence"),
    _sv7d_case("sv7d_policy_rule_text_changed", "policy_applicability", {"invoice_l.txt": _SV7D_INV.format(n=9008, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "ESCALATE", "ESCALATE", False, True, False, applicable=False, fault="rule_condition_9000",
                 note="evidence grounded, but the rule on file is no longer the rule the Decision used"),
    _sv7d_case("sv7d_policy_missing_fact", "policy_applicability", {"invoice_m.txt": _SV7D_INV.format(n=9009, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "ESCALATE", "ESCALATE", False, True, False, applicable=False, fault="basis_amount_none",
                 note="required amount fact absent on the basis node: applicability cannot be established"),
    _sv7d_case("sv7d_policy_wrong_scope", "policy_applicability", {"invoice_n.txt": _SV7D_INV.format(n=9010, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "ESCALATE", "ESCALATE", False, True, False, applicable=False, fault="wrong_scope",
                 note="Decision belongs to a Policy that does not own the cited rule"),
    _sv7d_case("sv7d_ambiguous_weak_grounding", "ambiguous", {"invoice_o.txt": _SV7D_INV.format(n=9011, amt="5,000")}, _SV7D_RB, "FORBID TRANSACTION", "ESCALATE", "ESCALATE", False, True, False, fault="weak_location",
                 note="claim present in text but source location lost: weakly grounded, human review"),
]

def validate_self_verification_benchmark(cases: Optional[List[Dict[str, Any]]] = None, require_coverage: bool = True) -> List[str]:
    cases = SELF_VERIFICATION_BENCHMARK if cases is None else cases
    probs: List[str] = []
    ids = [c.get("case_id") for c in cases]
    probs += [f"duplicate case_id {i}" for i, n in Counter(ids).items() if n > 1]
    for c in cases:
        cid = c.get("case_id")
        for k in ("category", "documents", "rulebook", "target_rule", "expected_outcome", "expected_verification_status", "decision_supported", "claims_grounded", "expected_major_contradiction", "policy_applicable"):
            if k not in c: probs.append(f"{cid}: missing {k}")
        if c.get("expected_outcome") not in SV7D_OUTCOMES: probs.append(f"{cid}: unknown expected_outcome {c.get('expected_outcome')}")
        if c.get("expected_verification_status") not in SELF_VERIFICATION_STATUSES: probs.append(f"{cid}: unknown expected_verification_status")
        if c.get("fault") is not None and c["fault"] not in _SV7D_FAULTS: probs.append(f"{cid}: unknown fault {c['fault']}")
        o, s = c.get("expected_outcome"), c.get("expected_verification_status")
        if c.get("decision_supported") != (o in SV_ASSERTING_VERDICTS): probs.append(f"{cid}: decision_supported inconsistent with expected_outcome")
        if (s == "VERIFIED") != (o in SV_ASSERTING_VERDICTS) or (s == "ESCALATE") != (o == "ESCALATE") or (s in ("FAILED", "NOT_VERIFIED") and o != "NO_CONCLUSION"): probs.append(f"{cid}: expected_verification_status inconsistent with expected_outcome")
        if s == "VERIFIED" and c.get("policy_applicable") is not True: probs.append(f"{cid}: VERIFIED requires an applicable rule")
    if require_coverage: probs += [f"category {x} not covered" for x in SV7D_CATEGORIES if x not in {c.get("category") for c in cases}]
    return probs

def _sv7d_outcome(verdict: Optional[str], status: Optional[str]) -> str:
    if verdict not in SV_ASSERTING_VERDICTS: return "NO_CONCLUSION"
    return "NO_CONCLUSION" if status == "FAILED" else "ESCALATE" if status == "ESCALATE" else verdict

def self_verification_variant_predictions(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Dict[str, Any]]:
    """Read-only (no graph mutation, idempotent). Per variant: outcome, verification_status, escalation_reason, grounding_predicted, contradiction_predicted, supporting_evidence (real ids + provenance as stored), confidence.
    A field a variant has no mechanism for is NOT_MEASURED. The Decision, its verdict and every verification result are only READ."""
    NM = NOT_MEASURED
    verdict = G.nodes[decision_id].get("verdict") if G.has_node(decision_id) else None
    asserting = verdict in SV_ASSERTING_VERDICTS
    base = build_self_verification_result(G, decision_id)
    r7b, r7c = verify_self_verification_result(G, decision_id), verify_policy_applicability(G, decision_id)
    conf = {"value": None, "status": "NOT_MEASURED"}
    out: Dict[str, Dict[str, Any]] = {"BASELINE": {"outcome": _sv7d_outcome(verdict, None), "verification_status": NM, "escalation_reason": NM, "grounding_predicted": NM, "contradiction_predicted": NM, "supporting_evidence": [], "confidence": dict(conf)}}
    out["EVIDENCE_GRAPH"] = {"outcome": _sv7d_outcome(verdict, None), "verification_status": NM, "escalation_reason": NM, "grounding_predicted": (bool(base["supporting_evidence"]) if asserting else NM),
                             "contradiction_predicted": (any(c.get("source") == "contradiction_finding" and c.get("category") == "MAJOR_CONTRADICTION" for c in base["contradicting_evidence"]) if asserting else NM),
                             "supporting_evidence": _sv_json(base["supporting_evidence"]), "confidence": dict(conf)}
    for name, r in (("EVIDENCE_GRAPH_SELF_VERIFICATION", r7b), ("EVIDENCE_GRAPH_SELF_VERIFICATION_POLICY_APPLICABILITY", r7c)):
        out[name] = {"outcome": _sv7d_outcome(verdict, r["verification_status"]), "verification_status": r["verification_status"], "escalation_reason": r["escalation_reason"],
                     "grounding_predicted": (r["verification_status"] != "FAILED" if asserting else NM), "contradiction_predicted": ("major contradiction" in (r["escalation_reason"] or "") if asserting else NM),
                     "supporting_evidence": _sv_json(r["supporting_evidence"]), "confidence": {"value": r["confidence"]["value"], "status": r["confidence"]["status"]}}
    return out

def _sv7d_stable_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evidence refs without the random graph ids: filename / location / provenance (minus the random source_document_id) kept, stable aliases E1.. assigned in a deterministic order."""
    rows = [{"filename": r.get("filename"), "location": r.get("location"), "file_hash": r.get("file_hash"), "provenance": {k: v for k, v in (r.get("provenance") or {}).items() if k != "source_document_id"}} for r in refs]
    rows.sort(key=lambda x: (str(x["filename"]), str(x["location"]), str((x["provenance"] or {}).get("source_text_sha256"))))
    return [{"evidence_alias": f"E{i + 1}", **x} for i, x in enumerate(rows)]

def _sv7d_snapshot(G: nx.MultiDiGraph) -> str:
    return json.dumps([sorted(G.nodes(data=True), key=lambda kv: kv[0]), sorted(((u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)), key=lambda e: (e[0], e[1], str(e[2]))), G.graph.get("contradiction_findings")], sort_keys=True, default=str)

def _sv7d_apply_fault(G: nx.MultiDiGraph, d: str, fault: Optional[str]) -> None:
    """Fault injection into the benchmark's PRIVATE graph only (declared in the case; never applied to a caller's graph)."""
    if not fault: return
    dd = G.nodes[d]; basis = list(dd.get("basis_node_ids") or [])
    ev = [e for b in basis for e in _supporting_evidence_ids(G, b)]
    if fault == "evidence_text_5000_to_500":
        for e in ev: G.nodes[e]["text"] = G.nodes[e]["text"].replace("5,000", "500")
    elif fault == "evidence_text_to_1":
        for e in ev: G.nodes[e]["text"] = "Billed amount INR 1"
    elif fault in ("heuristic_support", "remove_support"):
        for b in basis:
            for u, v, k, ed in list(G.in_edges(b, keys=True, data=True)):
                if ed.get("relation") != "SUPPORTS": continue
                if fault == "heuristic_support": ed["heuristic"] = True
                else: G.remove_edge(u, v, k)
    elif fault == "rule_condition_9000": G.nodes[dd["rule_id"]]["condition"] = "FORBID TRANSACTION > INR 9000"
    elif fault == "basis_amount_none":
        if basis: G.nodes[basis[0]]["amount"] = None
    elif fault == "wrong_scope":
        for u, v, k, e in list(G.out_edges(d, keys=True, data=True)):
            if e.get("relation") == "BELONGS_TO": G.remove_edge(u, v, k)
        G.add_node("pol_other_bench", type="Policy", name="other policy"); G.add_edge(d, "pol_other_bench", relation="BELONGS_TO")
    elif fault == "weak_location":
        for e in ev:
            G.nodes[e]["source_location"] = None
            for _, _, ed in G.out_edges(e, data=True):
                if ed.get("relation") == "DERIVED_FROM": ed["location"] = None

def _sv7d_target_decision(G: nx.MultiDiGraph, target: str) -> Optional[str]:
    hits = [n for n, x in sorted(_nodes_of_type(G, "Decision"), key=lambda kv: kv[0]) if target.lower() in str((G.nodes.get(x.get("rule_id")) or {}).get("condition") or "").lower()]
    return hits[0] if len(hits) == 1 else None

def _sv7d_records_for_graph(case: Dict[str, Any], G: nx.MultiDiGraph, d: str) -> List[Dict[str, Any]]:
    before = _sv7d_snapshot(G)
    preds = self_verification_variant_predictions(G, d)
    if _sv7d_snapshot(G) != before: raise RuntimeError("self-verification benchmark mutated the graph")
    recs = []
    for v in SV7D_VARIANTS:
        p = preds[v]
        ids = [r.get("evidence_id") for r in p["supporting_evidence"]]
        integ = {"evidence_ids_in_graph": all(G.has_node(i) and G.nodes[i].get("type") == "Evidence" for i in ids),
                 "provenance_matches_graph": all(r.get("provenance") == _sv_json(G.nodes[r["evidence_id"]].get("provenance")) for r in p["supporting_evidence"] if G.has_node(r.get("evidence_id")))}
        recs.append({"case_id": case["case_id"], "category": case["category"], "system_variant": v, "fault": case.get("fault"),
                     "expected_outcome": case["expected_outcome"], "predicted_outcome": p["outcome"], "decision_correct": p["outcome"] == case["expected_outcome"],
                     "decision_asserted": p["outcome"] in SV_ASSERTING_VERDICTS, "decision_supported_label": case["decision_supported"],
                     "claims_grounded_expected": case["claims_grounded"], "grounding_predicted": p["grounding_predicted"],
                     "expected_major_contradiction": case["expected_major_contradiction"], "contradiction_predicted": p["contradiction_predicted"],
                     "policy_applicable_expected": case["policy_applicable"],
                     "expected_verification_status": case["expected_verification_status"], "predicted_verification_status": p["verification_status"], "escalation_reason": (re.sub(r"\b([a-z]+)_[0-9a-f]{16}\b", r"<\1>", p["escalation_reason"]) if isinstance(p["escalation_reason"], str) else p["escalation_reason"]),  # random graph ids -> <type> so records are reproducible
                     "supporting_evidence": _sv7d_stable_refs(p["supporting_evidence"]), "id_integrity": integ, "graph_unchanged": True, "confidence": p["confidence"]})
    return recs

def _sv7d_run_case(case: Dict[str, Any], workdir: str) -> List[Dict[str, Any]]:
    paths = []
    for name, text in case["documents"].items():
        p = os.path.join(workdir, name)
        with open(p, "w", encoding="utf-8") as fh: fh.write(text)
        paths.append(p)
    rb = os.path.join(workdir, "rulebook.txt")
    with open(rb, "w", encoding="utf-8") as fh: fh.write(case["rulebook"])
    G = build_evidence_graph(paths, rb)
    d = _sv7d_target_decision(G, case["target_rule"])
    if d is None: raise ValueError(f"{case['case_id']}: target rule did not map to exactly one Decision")
    _sv7d_apply_fault(G, d, case.get("fault"))
    return _sv7d_records_for_graph(case, G, d)

def _sv7d_metric(num: int, den: int, reason: Optional[str] = None) -> Dict[str, Any]:
    if den == 0: return {"value": NOT_MEASURED, "numerator": None, "denominator": 0, "reason": reason or "no valid denominator"}
    return {"value": round(num / den, 4), "numerator": num, "denominator": den}

def _sv7d_variant_metrics(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    nm = lambda r: NOT_MEASURED
    asserted = [r for r in recs if r["decision_asserted"]]
    grd = [r for r in recs if isinstance(r["claims_grounded_expected"], bool) and isinstance(r["grounding_predicted"], bool)]
    con = [r for r in recs if isinstance(r["expected_major_contradiction"], bool) and isinstance(r["contradiction_predicted"], bool)]
    ver = [r for r in recs if r["predicted_verification_status"] != NOT_MEASURED]
    nov = [r for r in ver if r["expected_verification_status"] != "VERIFIED"]
    noe = [r for r in ver if r["expected_verification_status"] != "ESCALATE"]
    no_mech = "variant has no verification mechanism"
    return {"cases": len(recs),
            "decision_accuracy": _sv7d_metric(sum(r["decision_correct"] for r in recs), len(recs)),
            "unsupported_decision_rate": _sv7d_metric(sum(1 for r in asserted if r["decision_supported_label"] is False), len(asserted), "no asserted (VIOLATION / SATISFIED) decisions"),
            "evidence_grounding_accuracy": _sv7d_metric(sum(r["claims_grounded_expected"] == r["grounding_predicted"] for r in grd), len(grd), "variant has no grounding mechanism or no labelled case"),
            "contradiction_handling_accuracy": _sv7d_metric(sum(r["expected_major_contradiction"] == r["contradiction_predicted"] for r in con), len(con), "variant has no contradiction mechanism or no labelled case"),
            "verification_status_accuracy": _sv7d_metric(sum(r["predicted_verification_status"] == r["expected_verification_status"] for r in ver), len(ver), no_mech),
            "escalation_accuracy": _sv7d_metric(sum((r["predicted_verification_status"] == "ESCALATE") == (r["expected_verification_status"] == "ESCALATE") for r in ver), len(ver), no_mech),
            "false_verification_rate": _sv7d_metric(sum(r["predicted_verification_status"] == "VERIFIED" for r in nov), len(nov), no_mech + " or no case that must not be VERIFIED"),
            "false_escalation_rate": _sv7d_metric(sum(r["predicted_verification_status"] == "ESCALATE" for r in noe), len(noe), no_mech + " or no case that must not be ESCALATE")}

def aggregate_self_verification_results(records: List[Dict[str, Any]], cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Derived from the raw per-case / per-variant records only."""
    cases = SELF_VERIFICATION_BENCHMARK if cases is None else cases
    variants = {v: _sv7d_variant_metrics([r for r in records if r["system_variant"] == v]) for v in SV7D_VARIANTS}
    names = [k for k in variants[SV7D_VARIANTS[0]] if k != "cases"]
    val = lambda v, k: variants[v][k]["value"]
    isnum = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    comparison = {"order": list(SV7D_VARIANTS), "note": "same cases for every variant; raw values only; a delta exists only where both values are measured"}
    for k in names:
        vals = [val(v, k) for v in SV7D_VARIANTS]
        comparison[k] = {"values": dict(zip(SV7D_VARIANTS, vals)), **{f"delta_{a}_vs_{b}": (round(vals[j] - vals[i], 4) if isnum(vals[i]) and isnum(vals[j]) else NOT_MEASURED) for a, b, i, j in (("SV", "EG", 1, 2), ("SVPA", "SV", 2, 3))}}
    return {"benchmark": {"id": SV7D_ID, "version": SV7D_VERSION, "size": len(cases), "category_counts": dict(Counter(c["category"] for c in cases)), "expected_outcome_counts": dict(Counter(c["expected_outcome"] for c in cases)),
                          "expected_status_counts": dict(Counter(c["expected_verification_status"] for c in cases)), "seed": None, "llm_used": False},
            "variants": variants, "comparison": comparison, "protocol": SV7D_PROTOCOL}

def run_self_verification_benchmark(cases: Optional[List[Dict[str, Any]]] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Offline and deterministic. Every case builds its own private graph; returns {"records", "aggregate"}; writes JSON only if output_path is given."""
    import tempfile
    cases = SELF_VERIFICATION_BENCHMARK if cases is None else cases
    probs = validate_self_verification_benchmark(cases, require_coverage=cases is SELF_VERIFICATION_BENCHMARK)  # a custom / subset list is label-checked but need not cover every category
    if probs: raise ValueError("self-verification benchmark labels inconsistent: " + "; ".join(probs[:5]))
    records: List[Dict[str, Any]] = []
    for c in cases:
        with tempfile.TemporaryDirectory() as td: records += _sv7d_run_case(c, td)
    result = _json_safe({"benchmark_id": SV7D_ID, "version": SV7D_VERSION, "records": records, "aggregate": aggregate_self_verification_results(records, cases)})
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh: json.dump(result, fh, indent=2, ensure_ascii=False)
    return result

# --- COUNTERFACTUAL COMPLIANCE (Phase 8; READ-ONLY layer: decision / policy-rule / Evidence Graph / cross-document / contradiction / self-verification layers are REUSED unchanged) ---
# For every NON-COMPLIANT (VIOLATION) or CONDITIONAL (INCONCLUSIVE / UNEVALUATED) Decision: "what MINIMUM change or missing evidence would make this case compliant?". Everything is derived from the
# rule the Decision cites (its stored parsed_spec, never from the rule text of a particular scenario) and from the Evidence the Decision / contradiction findings already reference. No node / edge is created, no verdict,
# evidence or rule is altered, nothing is invented, no LLM / network / randomness. Recomputed on every call (idempotent). Result contract (all six requested fields always present):
#   current_decision | violated_rule | evidence_causing_violation | missing_requirement | recommended_corrective_condition | expected_resulting_state     (+ status, change_type, minimum_changes, ...)
#   status:      ESTABLISHED (policy-supported counterfactual) | UNRESOLVED (none can be established: fields not derivable are NOT_MEASURED, `unresolved_reason` says why) | NOT_REQUIRED (decision is not NON-COMPLIANT / CONDITIONAL)
#   change_type: CORRECTIVE_ACTION (the case itself must change: amount, identifier, offending text) | SUPPLY_EVIDENCE (the case may already be compliant; the rule cannot be checked until evidence exists)
# Compiled-rule Decisions (result_source compiled_policy_engine) are UNRESOLVED except ONE re-derivable semantic (see _cf_compiled: single leaf `amount <ordering-op> number` on a transaction entity); rules outside the DSL
# are UNRESOLVED: no corrective condition is guessed for rule semantics this layer cannot re-derive.
CF_VERSION = "CF1"
CF_FIELDS = ("current_decision", "violated_rule", "evidence_causing_violation", "missing_requirement", "recommended_corrective_condition", "expected_resulting_state")
CF_STATUSES = ("ESTABLISHED", "UNRESOLVED", "NOT_REQUIRED")
CF_CHANGE_TYPES = ("CORRECTIVE_ACTION", "SUPPLY_EVIDENCE")
CF_SCOPE_CLASS = {"VIOLATION": "NON_COMPLIANT", "INCONCLUSIVE": "CONDITIONAL", "UNEVALUATED": "CONDITIONAL"}
_CF_NEGATE = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}

def _cf_ev_ref(G: nx.MultiDiGraph, eid: str, role: str = "violation_basis") -> Dict[str, Any]:
    docs = _evidence_source_docs(G, eid)
    return {"evidence_id": eid, "role": role, "document_id": docs[0]["document_id"] if docs else None, "document_ids": [x["document_id"] for x in docs], "filename": docs[0]["filename"] if docs else None,
            "location": G.nodes[eid].get("source_location") or (docs[0].get("location") if docs else None), "provenance": _sv_json(G.nodes[eid].get("provenance"))}

def _cf_case_evidence(G: nx.MultiDiGraph, node_id: str) -> List[str]:
    """Qualifying case evidence behind a node: policy-source / context-only evidence never counts as the case."""
    if not G.has_node(node_id): return []
    return [e for e in _supporting_evidence_ids(G, node_id) if G.has_node(e) and G.nodes[e].get("type") == "Evidence" and not G.nodes[e].get("context_only") and not _is_policy_source_evidence(G.nodes[e])]

def _cf_rule_ref(G: nx.MultiDiGraph, dd: Dict[str, Any]) -> Dict[str, Any]:
    rid = dd.get("rule_id"); rd = G.nodes[rid] if rid and G.has_node(rid) and G.nodes[rid].get("type") == "PolicyRule" else {}
    pols = sorted(v for _, v, e in G.out_edges(rid, data=True) if e.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy") if rd else []
    return {"rule_id": rid, "condition": rd.get("condition"), "source_file": rd.get("source_file"), "source_location": rd.get("source_location"), "policy_ids": pols, "rulebook_provenance": _sv_json(dd.get("rulebook_provenance")),
            "parsed_spec": _sv_json(dd.get("parsed_spec")), "result_source": dd.get("result_source")}

def _cf_unresolved(res: Dict[str, Any], reason: str) -> Dict[str, Any]:
    res.update(status="UNRESOLVED", change_type=None, unresolved_reason=reason, missing_requirement=NOT_MEASURED, recommended_corrective_condition=NOT_MEASURED, expected_resulting_state=NOT_MEASURED, minimum_changes=[], required_condition=None)
    return res

def _cf_fmt(v: float) -> str:
    return f"{v:g}"

def _cf_transaction(G: nx.MultiDiGraph, dd: Dict[str, Any], spec: Dict[str, Any], basis: List[str]) -> Dict[str, Any]:
    op, val, cur = spec["op"], spec["value"], spec.get("currency")
    region = op if spec["mode"] == "REQUIRE" else _CF_NEGATE[op]  # the compliant region of an amount
    incl, direction = region in (">=", "<="), ("decrease" if region in ("<", "<=") else "increase")
    changes: List[Dict[str, Any]] = []
    for b in basis:
        d = G.nodes[b] if G.has_node(b) else {}; a = d.get("amount")
        if d.get("type") != "Transaction" or not isinstance(a, (int, float)) or not math.isfinite(a): return {"reason": f"basis node {b} has no usable transaction amount"}
        if _CMP_OPS[region](a, val): return {"reason": f"basis amount {_cf_fmt(a)} already meets {region} {_cf_fmt(val)}; stored Decision and rule disagree"}
        changes.append({"kind": "amount_change", "transaction_id": b, "currency": d.get("currency"), "current_value": a, "direction": direction, "boundary": val, "boundary_inclusive": incl,
                        "minimum_delta": round(abs(a - val), 10), "delta_is_strict_lower_bound": not incl, "target_value": (val if incl else None), "evidence_ids": _cf_case_evidence(G, b)})
    unit = cur or "(any currency)"
    return {"change_type": "CORRECTIVE_ACTION", "required_condition": {"subject": "TRANSACTION", "relation": region, "value": val, "currency": cur}, "minimum_changes": changes,
            "missing_requirement": f"each in-scope transaction amount must be {region} {_cf_fmt(val)} {unit} (rule {spec['mode']} TRANSACTION {op} {_cf_fmt(val)}); {len(changes)} amount(s) do not meet it",
            "recommended": (f"Correct each listed transaction amount so it is {region} {_cf_fmt(val)} {unit}; the smallest change moves it {direction} to the boundary {_cf_fmt(val)}" if incl else
                            f"Correct each listed transaction amount so it is {region} {_cf_fmt(val)} {unit}; the boundary itself is not compliant, so the amount must move {direction} by more than the listed minimum_delta"),
            "resulting": {"state": "COMPLIANT_WITH_RULE", "expected_verdict": "SATISFIED", "condition": f"every listed amount meets {region} {_cf_fmt(val)}; other rules are not re-evaluated"}}

def _cf_amount_pairs(G: nx.MultiDiGraph, basis: List[str]) -> List[Tuple[str, str]]:
    return [(a, b) for i, a in enumerate(basis) for b in basis[i + 1:] if G.has_node(a) and G.has_node(b) and G.nodes[a].get("currency") == G.nodes[b].get("currency") and _match_transactions(G.nodes[a], G.nodes[b])[0] == "strong"
            and G.nodes[a].get("amount") != G.nodes[b].get("amount")]

def _cf_amount_match(G: nx.MultiDiGraph, dd: Dict[str, Any], spec: Dict[str, Any], basis: List[str]) -> Dict[str, Any]:
    pairs = _cf_amount_pairs(G, basis)
    if not pairs: return {"reason": "no reliably linked basis pair with differing amounts is recorded"}
    ch = []
    for a, b in pairs:
        va, vb = G.nodes[a]["amount"], G.nodes[b]["amount"]
        ch.append({"kind": "amount_reconciliation", "transaction_ids": [a, b], "currency": G.nodes[a].get("currency"), "values": [va, vb], "minimum_delta": round(abs(va - vb), 10), "authoritative_record": NOT_MEASURED,
                   "evidence_ids": list(dict.fromkeys(_cf_case_evidence(G, a) + _cf_case_evidence(G, b)))})
    return {"change_type": "CORRECTIVE_ACTION", "required_condition": {"subject": "AMOUNT_MATCH", "relation": "==", "value": NOT_MEASURED, "currency": None}, "minimum_changes": ch,
            "missing_requirement": f"reliably linked records must carry the same amount (rule REQUIRE amounts to match); {len(ch)} linked pair(s) differ",
            "recommended": "Reconcile each listed pair so both records carry the same amount (smallest change = the listed minimum_delta applied to one record); the policy does not say which record is authoritative, so that is not decided here",
            "resulting": {"state": "COMPLIANT_WITH_RULE", "expected_verdict": "SATISFIED", "condition": "every linked pair carries equal amounts"}}

def _cf_entity(G: nx.MultiDiGraph, dd: Dict[str, Any], spec: Dict[str, Any], basis: List[str]) -> Dict[str, Any]:
    subj, verdict = spec["subject"], dd.get("verdict")
    if spec["mode"] != "REQUIRE": return {"reason": f"a FORBID {subj} VALID rule has no policy-supported corrective condition"}
    et = "GovID" if subj == "GOVID" else "Phone"
    ents = [n for n in basis if G.has_node(n) and G.nodes[n].get("type") == "Entity" and G.nodes[n].get("entity_type") == et]
    if not ents: return {"reason": f"no {subj} basis entity recorded"}
    if verdict == "VIOLATION":
        ch = [{"kind": "identifier_correction", "entity_id": n, "entity_type": et, "current_validity": G.nodes[n].get("is_valid"), "evidence_ids": _cf_case_evidence(G, n)} for n in ents]
        return {"change_type": "CORRECTIVE_ACTION", "required_condition": {"subject": subj, "relation": "is_valid", "value": True, "currency": None}, "minimum_changes": ch,
                "missing_requirement": f"every {subj} must be valid (rule REQUIRE {subj} VALID); {len(ch)} value(s) are invalid",
                "recommended": f"Correct each listed {subj} value in the source document so it passes the configured validation; which digits are wrong is not determined here",
                "resulting": {"state": "COMPLIANT_WITH_RULE", "expected_verdict": "SATISFIED", "condition": f"every listed {subj} validates"}}
    return {}

def _cf_keyword(G: nx.MultiDiGraph, dd: Dict[str, Any], spec: Dict[str, Any], basis: List[str]) -> Dict[str, Any]:
    phrase, verdict = str(spec.get("value") or ""), dd.get("verdict")
    if not phrase: return {"reason": "rule carries no phrase"}
    if spec["mode"] == "REQUIRE" and verdict == "VIOLATION":
        scope = [e for e in dd.get("absence_scope_evidence_ids") or [] if G.has_node(e) and G.nodes[e].get("type") == "Evidence"]
        return {"change_type": "SUPPLY_EVIDENCE", "required_condition": {"subject": "KEYWORD", "relation": "present", "value": phrase, "currency": None}, "evidence": [_cf_ev_ref(G, e, "absence_scope") for e in scope],
                "minimum_changes": [{"kind": "phrase_presence", "phrase": phrase, "action": "add", "evidence_ids": scope}],
                "missing_requirement": f"case evidence containing the phrase '{phrase}' (rule REQUIRE KEYWORD); it is absent within the extracted scope of {len(scope)} evidence item(s)",
                "recommended": f"Supply case evidence whose text contains '{phrase}'; the rule checks for that text only, so it does not establish whether the underlying action took place",
                "resulting": {"state": "RE_EVALUATION_REQUIRED", "expected_verdict": "SATISFIED", "condition": f"the supplied evidence text contains '{phrase}'"}}
    if spec["mode"] == "FORBID" and verdict == "VIOLATION":
        hits = list(dict.fromkeys(e for b in basis for e in ([b] if G.has_node(b) and G.nodes[b].get("type") == "Evidence" else [])))
        if not hits: return {"reason": "no offending evidence recorded as the basis"}
        return {"change_type": "CORRECTIVE_ACTION", "required_condition": {"subject": "KEYWORD", "relation": "absent", "value": phrase, "currency": None}, "evidence_ids_override": hits,
                "minimum_changes": [{"kind": "phrase_removal", "phrase": phrase, "action": "remove", "evidence_ids": hits}],
                "missing_requirement": f"the phrase '{phrase}' must not appear in case evidence (rule FORBID KEYWORD); it appears in {len(hits)} evidence item(s)",
                "recommended": f"Revise the documents behind the listed evidence so they no longer contain '{phrase}'",
                "resulting": {"state": "COMPLIANT_WITH_RULE", "expected_verdict": "SATISFIED", "condition": f"no case evidence contains '{phrase}'"}}
    if spec["mode"] == "REQUIRE" and verdict == "INCONCLUSIVE":
        gaps = [g_ for g_ in dd.get("extraction_gap_documents") or [] if G.has_node(g_)]
        if not gaps: return {"reason": "INCONCLUSIVE without recorded extraction-gap documents"}
        return {"change_type": "SUPPLY_EVIDENCE", "required_condition": {"subject": "KEYWORD", "relation": "present", "value": phrase, "currency": None}, "gap_documents": gaps,
                "minimum_changes": [{"kind": "readable_text_for_documents", "phrase": phrase, "document_ids": gaps, "filenames": [G.nodes[g_].get("filename") for g_ in gaps]}],
                "missing_requirement": f"readable extracted text for {len(gaps)} document(s) whose extraction failed or was empty, so the presence of '{phrase}' can be checked",
                "recommended": "Supply a readable version of each listed document; the rule can then be evaluated. The result may still be a violation if the phrase is not in them",
                "resulting": {"state": "RE_EVALUATION_REQUIRED", "expected_verdict": NOT_MEASURED, "condition": f"SATISFIED only if a re-extracted document contains '{phrase}'"}}
    return {}

def _cf_compiled(G: nx.MultiDiGraph, dd: Dict[str, Any], basis: List[str]) -> Dict[str, Any]:
    """Compiled-rule Decisions: exactly ONE semantic is re-derived, from the STORED compiled rule (no LLM, no text parsing): a single VALID, executed, VIOLATION compiled rule whose REQUIRE / PROHIBIT
    condition is one leaf `amount <, <=, >, >= number` on a transaction entity (no temporal, exception or evidence clause, no unit conversion). It maps to the existing TRANSACTION amount-boundary
    derivation (_cf_transaction). Anything else returns only a reason -> UNRESOLVED."""
    rid = dd.get("rule_id"); rn = G.nodes[rid] if rid and G.has_node(rid) and G.nodes[rid].get("type") == "PolicyRule" else None
    ents = (rn or {}).get("compiled_rules") or []
    if len(ents) != 1: return {"reason": "compiled semantics not re-derived: the cited rule line does not carry exactly one compiled rule"}
    e = ents[0]; cr = e.get("compiled_rule") or {}; r = e.get("result") or {}
    if e.get("status") != "VALID" or not r.get("executed") or r.get("verdict") != "VIOLATION": return {"reason": "compiled rule is not a VALID, executed VIOLATION"}
    if not basis or not set(basis) <= set(r.get("violating_node_ids") or []): return {"reason": "Decision basis is not a subset of the compiled rule's violating records"}
    if cr.get("rule_type") not in ("REQUIRE", "PROHIBIT") or cr.get("temporal") or cr.get("exception") or cr.get("required_evidence"): return {"reason": "compiled rule form (type / temporal / exception / evidence) is not a plain amount bound"}
    cond, ent = cr.get("condition") or {}, str(cr.get("entity") or "").lower()
    if ent not in _COMPILED_TXN_ENTITIES or cond.get("children") or str(cond.get("entity") or "").lower() != ent or str(cond.get("field") or "").lower() != "amount": return {"reason": "compiled condition is not a single leaf on a transaction amount"}
    op, val = cond.get("operator"), cond.get("value")
    if op not in _CMP_OPS or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val): return {"reason": "compiled condition is not an ordering comparison against a number"}
    unit = _compiled_norm_unit(cond.get("unit"))
    if unit is not None and (unit not in _KNOWN_CURRENCIES or any((G.nodes[b].get("currency") if G.has_node(b) else None) != unit for b in basis)): return {"reason": "compiled unit is not the basis currency: conversion is not re-derived"}
    return _cf_transaction(G, dd, {"mode": "REQUIRE" if cr["rule_type"] == "REQUIRE" else "FORBID", "op": op, "value": val, "currency": unit}, basis)

def _cf_conditional(G: nx.MultiDiGraph, dd: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """CONDITIONAL (INCONCLUSIVE) Decisions: the rule was attempted but evidence is insufficient; the counterfactual is the evidence the rule itself needs."""
    subj = spec["subject"]
    if subj == "KEYWORD": return _cf_keyword(G, dd, spec, [])
    if subj == "AMOUNT_MATCH":
        recs = [n for n, d in _nodes_of_type(G, "Transaction") if not _is_policy_role(d.get("amount_role"))]
        docs = {x["document_id"] for n in recs for e in _cf_case_evidence(G, n) for x in _evidence_source_docs(G, e)}
        if len(recs) < 2 or len(docs) < 2: return {"reason": "fewer than two records from different documents: nothing to link"}
        return {"change_type": "SUPPLY_EVIDENCE", "required_condition": {"subject": "AMOUNT_MATCH", "relation": "linked_records_equal", "value": NOT_MEASURED, "currency": None},
                "evidence_ids_override": list(dict.fromkeys(e for n in recs for e in _cf_case_evidence(G, n))),
                "minimum_changes": [{"kind": "link_evidence", "record_ids": recs, "requirement": "shared reference / transaction ID, or the same date + party + expense type, on the records to be compared"}],
                "missing_requirement": f"evidence that the {len(recs)} records from {len(docs)} documents refer to the same transaction (shared reference / transaction ID, or same date + party + expense type); without it nothing may be compared",
                "recommended": "Supply the linking reference (or date + party + expense type) for the records; no amount is changed. The rule then compares the linked amounts",
                "resulting": {"state": "RE_EVALUATION_REQUIRED", "expected_verdict": NOT_MEASURED, "condition": "SATISFIED only if the linked amounts are equal; a differing pair would be a VIOLATION"}}
    if subj in ("GOVID", "PHONE"):
        et = "GovID" if subj == "GOVID" else "Phone"
        pend = [n for n, d in _nodes_of_type(G, "Entity", et) if d.get("is_valid") is None]
        if not pend: return {"reason": f"no unverified {subj} entity recorded"}
        return {"change_type": "SUPPLY_EVIDENCE", "required_condition": {"subject": subj, "relation": "is_valid", "value": True, "currency": None}, "evidence_ids_override": list(dict.fromkeys(e for n in pend for e in _cf_case_evidence(G, n))),
                "minimum_changes": [{"kind": "validity_determination", "entity_ids": pend, "entity_type": et}],
                "missing_requirement": f"a validity determination for {len(pend)} detected {subj} value(s) (rule REQUIRE/FORBID {subj} VALID)",
                "recommended": f"Supply the validation result for each listed {subj} under a configured, jurisdiction-specific method; nothing is changed in the documents",
                "resulting": {"state": "RE_EVALUATION_REQUIRED", "expected_verdict": NOT_MEASURED, "condition": "outcome depends on the validation result"}}
    if subj == "TRANSACTION":
        unc = [n for n, d in _nodes_of_type(G, "Transaction") if d.get("amount_role") == ROLE_UNCLEAR and (not spec.get("currency") or d.get("currency") == spec["currency"])]
        if not unc: return {"reason": "no UNCLEAR-role amount recorded"}
        return {"change_type": "SUPPLY_EVIDENCE", "required_condition": {"subject": "TRANSACTION", "relation": spec["op"], "value": spec["value"], "currency": spec.get("currency")}, "evidence_ids_override": list(dict.fromkeys(e for n in unc for e in _cf_case_evidence(G, n))),
                "minimum_changes": [{"kind": "amount_role_clarification", "transaction_ids": unc}],
                "missing_requirement": f"evidence establishing whether {len(unc)} amount(s) are actual transactions (their role is UNCLEAR), so the rule can be applied",
                "recommended": "Supply source evidence that states what each listed amount represents; no amount is changed",
                "resulting": {"state": "RE_EVALUATION_REQUIRED", "expected_verdict": NOT_MEASURED, "condition": "outcome depends on what the amounts turn out to be"}}
    return {}

def _cf_contradiction_context(G: nx.MultiDiGraph, ev_ids: List[str], docs: set, region: Optional[Tuple[str, float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []; evs = set(ev_ids)
    for f in sorted(G.graph.get("contradiction_findings") or [], key=lambda x: x["finding_id"]):
        cat = f.get("category"); sides = [s for s in (f.get("claim_a"), f.get("claim_b")) if s]
        if cat not in XCON_CONTRADICTION and cat != "MISSING_EVIDENCE": continue
        touch = any(p.get("evidence_id") in evs for s in sides for p in s.get("provenance") or []) or (cat == "MISSING_EVIDENCE" and any(s.get("document_id") in docs for s in sides))
        if not touch: continue
        ent = {"finding_id": f["finding_id"], "category": cat, "field": f.get("field"), "reason": f.get("reason"), "claims": [{"document_id": s.get("document_id"), "filename": s.get("filename"), "value": _sv_json(s.get("value")),
               "evidence_ids": [p.get("evidence_id") for p in s.get("provenance") or [] if p.get("evidence_id")]} for s in sides], "other_side_amounts": [], "other_side_satisfies_rule": NOT_MEASURED}
        if f.get("field") == "amount" and region:
            others = sorted({x for s in sides if not any(p.get("evidence_id") in evs for p in s.get("provenance") or []) for x in ((s.get("value") or {}).get("amounts") or [])})
            ent["other_side_amounts"] = others
            if others: ent["other_side_satisfies_rule"] = all(_CMP_OPS[region[0]](x, region[1]) for x in others)
        out.append(ent)
    return out

def _cf_conflict(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Phase 8B: contradicting evidence already recorded for the Decision (self-verification layer, reused as stored). Metadata only: never changes status, change_type or the proposed change."""
    items = [x for x in build_self_verification_result(G, decision_id).get("contradicting_evidence") or [] if x.get("evidence_id") or x.get("finding_id")]
    ev = list(dict.fromkeys(x["evidence_id"] for x in items if x.get("evidence_id") and G.has_node(x["evidence_id"])))
    fin = list(dict.fromkeys(x["finding_id"] for x in items if x.get("finding_id")))
    return {"status": "UNRESOLVED_CONTRADICTION" if items else "NONE", "sources": sorted({x["source"] for x in items}), "finding_ids": fin, "evidence_ids": ev, "evidence": [_cf_ev_ref(G, e, "contradicting") for e in ev],
            "note": "recorded contradicting evidence is not resolved by this counterfactual: the proposed change presumes the cited evidence is the record that is wrong" if items else None}

def build_counterfactual(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Read-only; recomputed on every call (idempotent); deterministic. All six contract fields are always present."""
    NM = NOT_MEASURED
    res: Dict[str, Any] = {"contract_version": CF_VERSION, "counterfactual_id": f"cf::{decision_id}", "decision_id": decision_id, "target_kind": "decision", "found": False, "in_scope": False, "scope_class": None, "status": "NOT_REQUIRED", "change_type": None,
                           "unresolved_reason": None, "current_decision": NM, "violated_rule": NM, "evidence_causing_violation": [], "missing_requirement": NM, "recommended_corrective_condition": NM, "expected_resulting_state": NM,
                           "required_condition": None, "minimum_changes": [], "contradiction_context": [], "verification": {"verification_status": NM, "policy_applicability": NM}, "notes": [],
                           "satisfaction_asserted": False, "requires_reevaluation": False, "evidence_conflict": {"status": "NONE", "sources": [], "finding_ids": [], "evidence_ids": [], "evidence": [], "note": None}}
    if not G.has_node(decision_id) or G.nodes[decision_id].get("type") != "Decision": res["notes"].append("start node missing or not a Decision"); return res
    dd = G.nodes[decision_id]; verdict = dd.get("verdict"); res["found"] = True
    res["current_decision"] = {"decision_id": decision_id, "verdict": verdict, "scope_class": CF_SCOPE_CLASS.get(verdict), "rule_id": dd.get("rule_id"), "rationale": dd.get("rationale"), "result_source": dd.get("result_source"), "violation_status": dd.get("violation_status")}
    res["violated_rule"] = _cf_rule_ref(G, dd)
    if verdict not in CF_SCOPE_CLASS: res["notes"].append(f"verdict {verdict} is neither NON-COMPLIANT nor CONDITIONAL: no counterfactual required"); return res
    res["in_scope"], res["scope_class"] = True, CF_SCOPE_CLASS[verdict]
    res["evidence_conflict"] = _cf_conflict(G, decision_id)
    if verdict == "VIOLATION":  # reuse the self-verification layers: an unsupported / inapplicable decision is no premise for a counterfactual
        v = verify_policy_applicability(G, decision_id)
        res["verification"] = {"verification_status": v["verification_status"], "policy_applicability": (v.get("policy_applicability") or {}).get("status", NM), "escalation_reason": v.get("escalation_reason")}
    basis = list(dd.get("basis_node_ids") or [])
    if verdict == "UNEVALUATED": return _cf_unresolved(res, f"rule was never evaluated ({dd.get('unevaluated_reason') or 'unspecified'}); a corrective condition for an unchecked rule is not guessed")
    compiled = dd.get("result_source") == "compiled_policy_engine"
    d_c = _cf_compiled(G, dd, basis) if compiled and verdict == "VIOLATION" else None
    if compiled and not (d_c or {}).get("change_type"): return _cf_unresolved(res, "compiled-rule Decision: its expression semantics are not re-derived by this layer, so no corrective condition is inferred")
    spec = dd.get("parsed_spec")
    if not spec and d_c is None: return _cf_unresolved(res, "no parsed rule spec recorded on the Decision: the rule cannot be turned into a corrective condition")
    if res["verification"]["verification_status"] == "FAILED": return _cf_unresolved(res, "self-verification FAILED: the violation itself is not grounded, so no counterfactual rests on it")
    if res["verification"]["policy_applicability"] == "MISMATCH": return _cf_unresolved(res, "policy applicability MISMATCH: the cited rule does not apply to the verified facts")
    if res["verification"]["policy_applicability"] == "UNESTABLISHED": return _cf_unresolved(res, "policy applicability UNESTABLISHED: facts needed to show the cited rule applies are missing, so no counterfactual rests on it")
    rid_ = dd.get("rule_id"); rn_ = G.nodes[rid_] if rid_ and G.has_node(rid_) and G.nodes[rid_].get("type") == "PolicyRule" else None
    if d_c is None and (rn_ is None or parse_policy_rule(rn_.get("condition", "")) != spec): return _cf_unresolved(res, "cited rule is not a PolicyRule in the graph or its text no longer parses to the spec the Decision used: the corrective condition cannot be derived from it")
    subj = (spec or {}).get("subject")
    if d_c is not None: d_ = d_c
    elif verdict == "INCONCLUSIVE": d_ = _cf_conditional(G, dd, spec)
    elif subj == "TRANSACTION": d_ = _cf_transaction(G, dd, spec, basis)
    elif subj == "AMOUNT_MATCH": d_ = _cf_amount_match(G, dd, spec, basis)
    elif subj in ("GOVID", "PHONE"): d_ = _cf_entity(G, dd, spec, basis)
    elif subj == "KEYWORD": d_ = _cf_keyword(G, dd, spec, basis)
    else: d_ = {}
    if not d_.get("change_type"): return _cf_unresolved(res, d_.get("reason") or f"no policy-supported counterfactual exists for rule subject {subj} with verdict {verdict}")
    if not d_.get("minimum_changes"): return _cf_unresolved(res, "no policy-supported minimum change could be derived from the cited rule")
    ev_ids = list(d_.get("evidence_ids_override") or [])
    if "evidence" in d_: refs = d_["evidence"]
    else:
        if not ev_ids: ev_ids = list(dict.fromkeys(e for b in basis for e in _cf_case_evidence(G, b)))
        refs = [_cf_ev_ref(G, e) for e in ev_ids]
    refs += [{"evidence_id": None, "role": "extraction_gap_document", "document_id": g_, "document_ids": [g_], "filename": G.nodes[g_].get("filename"), "location": None, "provenance": None} for g_ in d_.get("gap_documents") or []]
    all_ev = [r["evidence_id"] for r in refs if r.get("evidence_id")]
    docs = {x for r in refs for x in r.get("document_ids") or []}
    region = (d_["required_condition"]["relation"], d_["required_condition"]["value"]) if d_["required_condition"]["subject"] == "TRANSACTION" and isinstance(d_["required_condition"]["value"], (int, float)) and d_["required_condition"]["relation"] in _CMP_OPS else None
    res.update(status="ESTABLISHED", change_type=d_["change_type"], evidence_causing_violation=refs, missing_requirement=d_["missing_requirement"], recommended_corrective_condition=d_["recommended"],
               expected_resulting_state={**d_["resulting"], "rule_id": dd.get("rule_id")}, required_condition=d_["required_condition"], minimum_changes=_sv_json(d_["minimum_changes"]), unresolved_reason=None,
               contradiction_context=_cf_contradiction_context(G, all_ev, docs, region), requires_reevaluation=True)
    return res

def build_counterfactuals(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """One counterfactual per NON-COMPLIANT / CONDITIONAL Decision, in a stable order."""
    return [r for r in (build_counterfactual(G, n) for n, _ in sorted(_nodes_of_type(G, "Decision"), key=lambda kv: kv[0])) if r["in_scope"]]

def build_finding_counterfactual(G: nx.MultiDiGraph, finding: Dict[str, Any]) -> Dict[str, Any]:
    """Counterfactual for a cross-document contradiction / missing-evidence finding. Policy-supported ONLY where an existing parsed policy rule (REQUIRE amounts to match) governs the finding's field; otherwise UNRESOLVED."""
    NM = NOT_MEASURED; fid = finding["finding_id"]
    res: Dict[str, Any] = {"contract_version": CF_VERSION, "counterfactual_id": f"cf::{fid}", "decision_id": None, "finding_id": fid, "target_kind": "finding", "found": True, "in_scope": True, "scope_class": "CONDITIONAL", "status": "UNRESOLVED", "change_type": None,
                           "unresolved_reason": None, "current_decision": {"finding_id": fid, "category": finding.get("category"), "field": finding.get("field"), "reason": finding.get("reason"), "resolution": finding.get("resolution")}, "violated_rule": NM,
                           "evidence_causing_violation": [], "missing_requirement": NM, "recommended_corrective_condition": NM, "expected_resulting_state": NM, "required_condition": None, "minimum_changes": [], "contradiction_context": [],
                           "verification": {"verification_status": NM, "policy_applicability": NM}, "notes": [],
                           "satisfaction_asserted": False, "requires_reevaluation": False, "evidence_conflict": {"status": "SUBJECT_OF_COUNTERFACTUAL", "sources": ["contradiction_finding"], "finding_ids": [fid], "evidence_ids": [], "evidence": [], "note": None}}
    sides = [s for s in (finding.get("claim_a"), finding.get("claim_b")) if s]
    evs = list(dict.fromkeys(p["evidence_id"] for s in sides for p in s.get("provenance") or [] if p.get("evidence_id") and G.has_node(p["evidence_id"]) and G.nodes[p["evidence_id"]].get("type") == "Evidence"))
    res["evidence_causing_violation"] = [_cf_ev_ref(G, e, "finding_claim") for e in evs]
    rule = next(((n, d) for n, d in sorted(_nodes_of_type(G, "PolicyRule"), key=lambda kv: kv[0]) if (parse_policy_rule(d.get("condition", "")) or {}).get("subject") == "AMOUNT_MATCH"), None)
    if finding.get("category") not in XCON_CONTRADICTION: return _cf_unresolved(res, f"{finding.get('category')} finding: no policy rule states what evidence would resolve it")
    if finding.get("field") != "amount" or rule is None: return _cf_unresolved(res, f"no policy rule in the graph requires '{finding.get('field')}' to agree between linked documents; no corrective condition is invented")
    amts = [((s.get("value") or {}).get("amounts") or []) for s in sides]; cur = {(s.get("value") or {}).get("currency") for s in sides}
    if len(sides) != 2 or any(len(a) != 1 for a in amts) or len(cur) != 1: return _cf_unresolved(res, "the contradicting claims do not carry one amount each in one currency")
    rid, rd = rule; spec = parse_policy_rule(rd["condition"]); delta = round(abs(amts[0][0] - amts[1][0]), 10)
    res["violated_rule"] = {"rule_id": rid, "condition": rd.get("condition"), "source_file": rd.get("source_file"), "source_location": rd.get("source_location"),
                            "policy_ids": sorted(v for _, v, e in G.out_edges(rid, data=True) if e.get("relation") == "BELONGS_TO" and G.nodes[v].get("type") == "Policy"), "rulebook_provenance": NM, "parsed_spec": _sv_json(spec), "result_source": "parsed_rule_text"}
    res.update(status="ESTABLISHED", change_type="CORRECTIVE_ACTION", required_condition={"subject": "AMOUNT_MATCH", "relation": "==", "value": NM, "currency": next(iter(cur))},
               minimum_changes=[{"kind": "amount_reconciliation", "currency": next(iter(cur)), "values": [amts[0][0], amts[1][0]], "minimum_delta": delta, "authoritative_record": NM, "evidence_ids": evs}],
               missing_requirement=f"the two linked documents state different amounts ({_cf_fmt(amts[0][0])} vs {_cf_fmt(amts[1][0])} {next(iter(cur))}); rule REQUIRE amounts to match requires them to agree",
               recommended_corrective_condition="Reconcile the two documents so they carry the same amount (smallest change = the listed minimum_delta applied to one document); the policy does not say which document is authoritative, so that is not decided here",
               requires_reevaluation=True, expected_resulting_state={"state": "AMOUNTS_CONSISTENT", "expected_verdict": NM, "rule_id": rid, "condition": "the rule engine compares only reliably linked records, so its verdict after the correction also depends on that linkage"})
    return res

def build_finding_counterfactuals(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    return [build_finding_counterfactual(G, f) for f in sorted(G.graph.get("contradiction_findings") or [], key=lambda x: x["finding_id"]) if f.get("category") in XCON_CONTRADICTION or f.get("category") == "MISSING_EVIDENCE"]

def validate_counterfactual(G: nx.MultiDiGraph, r: Dict[str, Any]) -> List[str]:
    """Contract check: fields present, status / change_type allowed, rule and evidence references exist and trace to their documents (nothing invented), ESTABLISHED is complete and UNRESOLVED carries NOT_MEASURED."""
    probs = [f"missing field {k}" for k in CF_FIELDS if k not in r]
    if r.get("status") not in CF_STATUSES: probs.append(f"unknown status {r.get('status')}")
    if r.get("change_type") is not None and r["change_type"] not in CF_CHANGE_TYPES: probs.append(f"unknown change_type {r.get('change_type')}")
    if (r.get("status") == "ESTABLISHED") != (r.get("change_type") is not None): probs.append("change_type must be set exactly when status is ESTABLISHED")
    if r.get("status") == "UNRESOLVED":
        if not r.get("unresolved_reason"): probs.append("UNRESOLVED needs unresolved_reason")
        probs += [f"UNRESOLVED must keep {k} NOT_MEASURED" for k in ("missing_requirement", "recommended_corrective_condition", "expected_resulting_state") if r.get(k) != NOT_MEASURED]
    if r.get("status") == "ESTABLISHED":
        rid = (r.get("violated_rule") or {}).get("rule_id") if isinstance(r.get("violated_rule"), dict) else None
        if not (rid and G.has_node(rid) and G.nodes[rid].get("type") == "PolicyRule"): probs.append("ESTABLISHED counterfactual must reference an existing PolicyRule")
        probs += [f"ESTABLISHED must set {k}" for k in ("missing_requirement", "recommended_corrective_condition", "expected_resulting_state") if r.get(k) in (None, NOT_MEASURED)]
        if not r.get("evidence_causing_violation"): probs.append("ESTABLISHED counterfactual must cite evidence")
        if rid and G.has_node(rid) and G.nodes[rid].get("type") == "PolicyRule" and r["violated_rule"].get("result_source") != "compiled_policy_engine" and not parse_policy_rule(G.nodes[rid].get("condition", "")): probs.append("ESTABLISHED counterfactual rests on a rule whose text is not a supported rule form")
        if not r.get("minimum_changes"): probs.append("ESTABLISHED counterfactual must list a minimum change")
        if r.get("satisfaction_asserted") is not False or r.get("requires_reevaluation") is not True: probs.append("ESTABLISHED counterfactual must not assert satisfaction and must require re-evaluation")
        if r.get("change_type") == "SUPPLY_EVIDENCE" and isinstance(r.get("expected_resulting_state"), dict) and r["expected_resulting_state"].get("state") == "COMPLIANT_WITH_RULE": probs.append("SUPPLY_EVIDENCE must not claim compliance: supplied evidence is not proof of the requirement")
    for ref in r.get("evidence_causing_violation") or []:
        eid = ref.get("evidence_id")
        if eid is None:
            if not (ref.get("document_id") and G.has_node(ref["document_id"]) and G.nodes[ref["document_id"]].get("type") == "Document"): probs.append(f"document reference {ref.get('document_id')} not in graph")
            continue
        if not (G.has_node(eid) and G.nodes[eid].get("type") == "Evidence"): probs.append(f"evidence {eid} not in graph"); continue
        if ref.get("document_id") and ref["document_id"] not in {x["document_id"] for x in _evidence_source_docs(G, eid)}: probs.append(f"evidence {eid} is not derived from document {ref['document_id']}")
    return probs

# --- COUNTERFACTUAL COMPLIANCE BENCHMARK (Phase 8 evaluation; layer only) ---
# Deterministic labelled cases. Each case builds its OWN private graph with the existing offline pipeline. The label block (expected_*) is hand-written and NEVER read from system output; predictions come from
# build_counterfactual / build_finding_counterfactual and are scored against the labels afterwards. No LLM, no network, no randomness. A case may declare a fault (reused from the 7D fault set) injected into its OWN graph.
# Counterfactual VALIDITY is checked by REPLAY: the predicted counterfactual is applied to a copy of the case documents (declared per case in `replay`; the replacement value / phrase comes from the PREDICTION), the
# unchanged pipeline is re-run and the target rule must reach the labelled verdict. Metrics are {value, numerator, denominator}; value is NOT_MEASURED (never 0.0) when the denominator is empty.
CFB_ID, CFB_VERSION = "omnicheck-counterfactual-bench", "1.0"
CFB_CATEGORIES = ("positive", "negative", "conditional", "missing_evidence", "contradiction", "ambiguous", "no_valid_counterfactual")
CFB_PROTOCOL = {"seed": None, "randomness": "none", "llm_used": False, "network_used": False, "confidence": "not produced; nothing is estimated",
                "ground_truth": "hand-written per-case labels (status, change type, required condition, minimum change, evidence files, replay outcome); never derived from a prediction",
                "validity": "replay: the predicted change is applied to a private copy of the documents and the unchanged engine must return the labelled verdict for the target rule",
                "metrics_not_measured_when": "the denominator is empty (no case that the metric applies to)",
                "id_policy": "records carry stable evidence aliases, filenames, locations and provenance; random graph ids are verified against the graph at run time (id_integrity) and not stored"}

def _cfb_case(cid, category, docs, rulebook, target, status, decision=None, change_type=None, requirement=None, changes=None, ev_files=None, replay=None, after=None, fault=None, kind="decision", field=None, finding_category=None, note=""):
    return {"case_id": cid, "category": category, "documents": dict(docs), "rulebook": rulebook, "target_rule": target, "target_kind": kind, "target_field": field, "target_finding_category": finding_category, "fault": fault, "note": note,
            "expected_status": status, "expected_decision": decision, "expected_change_type": change_type, "expected_required_condition": requirement, "expected_minimum_changes": changes, "expected_evidence_files": ev_files, "replay": replay, "expected_after_verdict": after}

_CFB_INV = "Invoice No: INV-{n}\nBilled amount INR {amt}\n"
_CFB_LINK_I, _CFB_LINK_R = "Invoice No: INV-77\nVendor: Boreal Metals\nInvoice date 2024-03-10\nBilled amount INR 20,000\n", "Receipt for Invoice No: INV-77\nVendor: Boreal Metals\nInvoice date 2024-03-10\nPaid amount INR {v}\n"
_CFB_CON_I, _CFB_CON_P = "Invoice No: INV-1101\nPO Number: PO-1101\nVendor: Boreal Metals\nBilled amount INR 20,000\n", "Purchase Order No: PO-1101\nVendor: Boreal Metals\nOrder value INR 12,000\n"
_CFB_AMT = lambda d, b, i, inc, dl, tv: {"kind": "amount_change", "direction": d, "boundary": b, "boundary_inclusive": inc, "minimum_delta": dl, "target_value": tv}

COUNTERFACTUAL_BENCHMARK: List[Dict[str, Any]] = [
    _cfb_case("cfb_pos_amount_over_limit", "positive", {"inv_a.txt": _CFB_INV.format(n=8001, amt="5,000")}, "FORBID TRANSACTION > INR 1000\n", "FORBID TRANSACTION", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "TRANSACTION", "relation": "<=", "value": 1000.0, "currency": "INR"}, [_CFB_AMT("decrease", 1000.0, 0, True, 4000.0, 1000.0)], ["inv_a.txt"], {"kind": "amount_replace", "file": "inv_a.txt", "old": "5,000", "step": 0.0}, "SATISFIED",
              note="incorrect transaction amount: the minimum change is the rule boundary"),
    _cfb_case("cfb_pos_require_minimum", "positive", {"inv_b.txt": _CFB_INV.format(n=8002, amt="800")}, "REQUIRE TRANSACTION >= INR 1000\n", "REQUIRE TRANSACTION", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "TRANSACTION", "relation": ">=", "value": 1000.0, "currency": "INR"}, [_CFB_AMT("increase", 1000.0, 0, True, 200.0, 1000.0)], ["inv_b.txt"], {"kind": "amount_replace", "file": "inv_b.txt", "old": "800", "step": 0.0}, "SATISFIED"),
    _cfb_case("cfb_pos_exclusive_boundary", "positive", {"inv_c.txt": _CFB_INV.format(n=8003, amt="5,000")}, "FORBID TRANSACTION >= INR 1000\n", "FORBID TRANSACTION", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "TRANSACTION", "relation": "<", "value": 1000.0, "currency": "INR"}, [_CFB_AMT("decrease", 1000.0, 0, False, 4000.0, None)], ["inv_c.txt"], {"kind": "amount_replace", "file": "inv_c.txt", "old": "5,000", "step": -1.0}, "SATISFIED",
              note="the boundary value itself is non-compliant: no exact minimum value exists, only a strict bound"),
    _cfb_case("cfb_pos_two_amounts", "positive", {"inv_d1.txt": _CFB_INV.format(n=8004, amt="5,000"), "inv_d2.txt": _CFB_INV.format(n=8005, amt="3,000")}, "FORBID TRANSACTION > INR 1000\n", "FORBID TRANSACTION", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "TRANSACTION", "relation": "<=", "value": 1000.0, "currency": "INR"}, [_CFB_AMT("decrease", 1000.0, 0, True, 4000.0, 1000.0), _CFB_AMT("decrease", 1000.0, 0, True, 2000.0, 1000.0)], ["inv_d1.txt", "inv_d2.txt"],
              {"kind": "amount_replace_many", "files": {"inv_d1.txt": "5,000", "inv_d2.txt": "3,000"}, "step": 0.0}, "SATISFIED", note="every violating amount needs its own minimum change"),
    _cfb_case("cfb_pos_missing_manager_approval", "missing_evidence", {"memo_a.txt": "Expense memo\nTravel booked for the team\n"}, 'REQUIRE KEYWORD "manager approval"\n', "REQUIRE KEYWORD", "ESTABLISHED", "VIOLATION", "SUPPLY_EVIDENCE",
              {"subject": "KEYWORD", "relation": "present", "value": "manager approval", "currency": None}, [{"kind": "phrase_presence", "phrase": "manager approval", "action": "add"}], ["memo_a.txt"], {"kind": "append_phrase", "file": "memo_a.txt"}, "SATISFIED",
              note="supplying evidence, not a corrective action on the case"),
    _cfb_case("cfb_pos_missing_receipt", "missing_evidence", {"claim_b.txt": "Claim form\nMeal expense for the team\n"}, 'REQUIRE KEYWORD "receipt attached"\n', "REQUIRE KEYWORD", "ESTABLISHED", "VIOLATION", "SUPPLY_EVIDENCE",
              {"subject": "KEYWORD", "relation": "present", "value": "receipt attached", "currency": None}, [{"kind": "phrase_presence", "phrase": "receipt attached", "action": "add"}], ["claim_b.txt"], {"kind": "append_phrase", "file": "claim_b.txt"}, "SATISFIED"),
    _cfb_case("cfb_pos_policy_exception", "positive", {"inv_e.txt": _CFB_INV.format(n=8006, amt="5,000")}, 'FORBID TRANSACTION > INR 1000\nREQUIRE KEYWORD "policy exception granted"\n', "policy exception", "ESTABLISHED", "VIOLATION", "SUPPLY_EVIDENCE",
              {"subject": "KEYWORD", "relation": "present", "value": "policy exception granted", "currency": None}, [{"kind": "phrase_presence", "phrase": "policy exception granted", "action": "add"}], ["inv_e.txt"], {"kind": "append_phrase", "file": "inv_e.txt"}, "SATISFIED",
              note="an exception requirement exists only as a policy rule; it is derived from that rule, not hard-coded"),
    _cfb_case("cfb_pos_forbidden_phrase", "positive", {"pay_f.txt": "Settlement note\nSettled by cash payment on delivery\n"}, 'FORBID KEYWORD "cash payment"\n', "FORBID KEYWORD", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "KEYWORD", "relation": "absent", "value": "cash payment", "currency": None}, [{"kind": "phrase_removal", "phrase": "cash payment", "action": "remove"}], ["pay_f.txt"], {"kind": "remove_phrase", "file": "pay_f.txt"}, "SATISFIED"),
    _cfb_case("cfb_pos_linked_amount_mismatch", "positive", {"inv_g.txt": _CFB_LINK_I, "rcpt_g.txt": _CFB_LINK_R.format(v="12,000")}, "Transaction amounts must match.\n", "amounts must match", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "AMOUNT_MATCH", "relation": "==", "value": NOT_MEASURED, "currency": None}, [{"kind": "amount_reconciliation", "minimum_delta": 8000.0}], ["inv_g.txt", "rcpt_g.txt"], {"kind": "amount_replace", "file": "rcpt_g.txt", "old": "12,000", "to": "20,000"}, "SATISFIED",
              note="which record is authoritative is not decided by policy; replay equalises the amounts"),
    _cfb_case("cfb_neg_satisfied", "negative", {"inv_h.txt": _CFB_INV.format(n=8007, amt="800")}, "FORBID TRANSACTION > INR 1000\n", "FORBID TRANSACTION", "NOT_REQUIRED", "SATISFIED", note="compliant: no counterfactual is required"),
    _cfb_case("cfb_neg_not_applicable", "negative", {"memo_i.txt": "Meeting notes\nNothing to report\n"}, "FORBID TRANSACTION > INR 1000\n", "FORBID TRANSACTION", "NOT_REQUIRED", "NOT_APPLICABLE", note="rule does not apply: no counterfactual is required"),
    _cfb_case("cfb_cond_unlinked_records", "conditional", {"inv_j.txt": _CFB_CON_I, "po_j.txt": _CFB_CON_P}, "Transaction amounts must match.\n", "amounts must match", "ESTABLISHED", "INCONCLUSIVE", "SUPPLY_EVIDENCE",
              {"subject": "AMOUNT_MATCH", "relation": "linked_records_equal", "value": NOT_MEASURED, "currency": None}, [{"kind": "link_evidence"}], ["inv_j.txt", "po_j.txt"], None, None, note="no amount may be changed: the records first have to be linked"),
    _cfb_case("cfb_cond_extraction_gap", "conditional", {"empty_k.txt": "", "memo_k.txt": "Meeting notes\n"}, 'REQUIRE KEYWORD "approval"\n', "REQUIRE KEYWORD", "ESTABLISHED", "INCONCLUSIVE", "SUPPLY_EVIDENCE",
              {"subject": "KEYWORD", "relation": "present", "value": "approval", "currency": None}, [{"kind": "readable_text_for_documents", "phrase": "approval"}], ["empty_k.txt"], None, None, note="the unreadable document is the missing evidence; it is a document reference, not an Evidence node"),
    _cfb_case("cfb_contradiction_decision", "contradiction", {"inv_l.txt": _CFB_CON_I, "po_l.txt": _CFB_CON_P}, "FORBID TRANSACTION > INR 15000\n", "FORBID TRANSACTION", "ESTABLISHED", "VIOLATION", "CORRECTIVE_ACTION",
              {"subject": "TRANSACTION", "relation": "<=", "value": 15000.0, "currency": "INR"}, [_CFB_AMT("decrease", 15000.0, 0, True, 5000.0, 15000.0)], ["inv_l.txt"], {"kind": "amount_replace", "file": "inv_l.txt", "old": "20,000", "step": 0.0}, "SATISFIED",
              note="the linked PO states 12,000, which already meets the rule: the contradiction context must say so"),
    _cfb_case("cfb_contradiction_amount_finding", "contradiction", {"inv_m.txt": _CFB_CON_I, "po_m.txt": _CFB_CON_P}, "Transaction amounts must match.\n", "amounts must match", "ESTABLISHED", None, "CORRECTIVE_ACTION",
              {"subject": "AMOUNT_MATCH", "relation": "==", "value": NOT_MEASURED, "currency": "INR"}, [{"kind": "amount_reconciliation", "minimum_delta": 8000.0}], ["inv_m.txt", "po_m.txt"], None, None, kind="finding", field="amount", finding_category="MAJOR_CONTRADICTION",
              note="the amount contradiction is governed by the amounts-must-match rule"),
    _cfb_case("cfb_contradiction_vendor_no_policy", "no_valid_counterfactual", {"inv_n.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 800\n", "po_n.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"},
              "FORBID TRANSACTION > INR 100000\n", "vendor", "UNRESOLVED", kind="finding", field="vendor", finding_category="MAJOR_CONTRADICTION", ev_files=["inv_n.txt", "po_n.txt"], note="vendor identity mismatch, but no rule requires vendors to agree: nothing may be invented"),
    _cfb_case("cfb_contradiction_date_no_policy", "no_valid_counterfactual", {"inv_o.txt": "Invoice No: INV-6\nPO Number: PO-6\nVendor: Boreal Metals\nInvoice date 2024-03-10\nBilled amount INR 800\n",
              "po_o.txt": "Purchase Order No: PO-6\nVendor: Boreal Metals\nInvoice date 2024-04-02\nOrder value INR 800\n"}, "FORBID TRANSACTION > INR 100000\n", "date", "UNRESOLVED", kind="finding", field="date", finding_category="MAJOR_CONTRADICTION",
              ev_files=["inv_o.txt", "po_o.txt"], note="date mismatch with no governing rule"),
    _cfb_case("cfb_ambiguous_free_text_rule", "ambiguous", {"inv_p.txt": _CFB_INV.format(n=8008, amt="5,000")}, "FORBID TRANSACTION > INR 1000\nVendors should behave reasonably in spirit.\n", "Vendors should behave", "UNRESOLVED", "UNEVALUATED",
              note="rule not expressible deterministically: no corrective condition"),
    _cfb_case("cfb_ambiguous_govid_unconfigured", "ambiguous", {"id_q.txt": "Employee record\nGov ID 1234 5678 9012\n"}, "REQUIRE GOVID VALID\n", "REQUIRE GOVID", "UNRESOLVED", "UNEVALUATED", note="validation configuration unavailable: the rule was never checked"),
    _cfb_case("cfb_unsupported_violation", "no_valid_counterfactual", {"inv_r.txt": _CFB_INV.format(n=8009, amt="5,000")}, "FORBID TRANSACTION > INR 1000\n", "FORBID TRANSACTION", "UNRESOLVED", "VIOLATION", fault="evidence_text_5000_to_500",
              ev_files=None, note="the violation is no longer grounded in its source text (self-verification FAILED): no counterfactual rests on it"),
    _cfb_case("cfb_inapplicable_rule", "no_valid_counterfactual", {"inv_s.txt": _CFB_INV.format(n=8010, amt="5,000")}, "FORBID TRANSACTION > INR 1000\n", "FORBID TRANSACTION", "UNRESOLVED", "VIOLATION", fault="rule_condition_9000",
              note="the rule on file is not the rule the Decision used (policy applicability MISMATCH)"),
]

def validate_counterfactual_benchmark(cases: Optional[List[Dict[str, Any]]] = None, require_coverage: bool = True) -> List[str]:
    cases = COUNTERFACTUAL_BENCHMARK if cases is None else cases
    probs: List[str] = []
    probs += [f"duplicate case_id {i}" for i, n in Counter(c.get("case_id") for c in cases).items() if n > 1]
    for c in cases:
        cid = c.get("case_id")
        for k in ("category", "documents", "rulebook", "target_rule", "target_kind", "expected_status", "expected_decision", "expected_change_type", "expected_required_condition", "expected_minimum_changes", "expected_evidence_files", "replay", "expected_after_verdict"):
            if k not in c: probs.append(f"{cid}: missing {k}")
        if c.get("category") not in CFB_CATEGORIES: probs.append(f"{cid}: unknown category {c.get('category')}")
        if c.get("expected_status") not in CF_STATUSES: probs.append(f"{cid}: unknown expected_status {c.get('expected_status')}")
        if c.get("target_kind") not in ("decision", "finding"): probs.append(f"{cid}: unknown target_kind")
        if c.get("fault") is not None and c["fault"] not in _SV7D_FAULTS: probs.append(f"{cid}: unknown fault {c['fault']}")
        est = c.get("expected_status") == "ESTABLISHED"
        if est != (c.get("expected_change_type") in CF_CHANGE_TYPES): probs.append(f"{cid}: expected_change_type must be set exactly when expected_status is ESTABLISHED")
        if est and (not c.get("expected_required_condition") or not c.get("expected_minimum_changes")): probs.append(f"{cid}: ESTABLISHED needs expected_required_condition and expected_minimum_changes")
        if not est and (c.get("expected_required_condition") is not None or c.get("expected_minimum_changes") is not None or c.get("replay") is not None): probs.append(f"{cid}: only ESTABLISHED cases carry condition / minimum change / replay labels")
        if (c.get("replay") is None) != (c.get("expected_after_verdict") is None): probs.append(f"{cid}: replay and expected_after_verdict go together")
        if c.get("expected_status") == "NOT_REQUIRED" and c.get("expected_decision") not in ("SATISFIED", "NOT_APPLICABLE"): probs.append(f"{cid}: NOT_REQUIRED only for SATISFIED / NOT_APPLICABLE decisions")
        if c.get("target_kind") == "decision" and c.get("expected_decision") is None: probs.append(f"{cid}: decision target needs expected_decision")
        if c.get("target_kind") == "finding" and not (c.get("target_field") and c.get("target_finding_category")): probs.append(f"{cid}: finding target needs target_field and target_finding_category")
    if require_coverage: probs += [f"category {x} not covered" for x in CFB_CATEGORIES if x not in {c.get("category") for c in cases}]
    return probs

def _cfb_target(G: nx.MultiDiGraph, case: Dict[str, Any]) -> Optional[Tuple[str, Optional[str]]]:
    """(kind, id) of the unique target: a Decision whose rule text contains target_rule, or a finding with the labelled category + field whose claims touch the target documents."""
    if case["target_kind"] == "decision":
        d = _sv7d_target_decision(G, case["target_rule"])
        return ("decision", d) if d else None
    hits = [f for f in sorted(G.graph.get("contradiction_findings") or [], key=lambda x: x["finding_id"]) if f.get("category") == case["target_finding_category"] and f.get("field") == case["target_field"]]
    return ("finding", hits[0]["finding_id"]) if len(hits) == 1 else None

def _cfb_predict(G: nx.MultiDiGraph, case: Dict[str, Any], target: Tuple[str, Optional[str]]) -> Dict[str, Any]:
    if target[0] == "decision": return build_counterfactual(G, target[1])
    return build_finding_counterfactual(G, next(f for f in G.graph.get("contradiction_findings") or [] if f["finding_id"] == target[1]))

def _cfb_norm_change(c: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple:
    return tuple((k, (round(c[k], 6) if isinstance(c.get(k), float) else c.get(k))) for k in keys if k in c)

def _cfb_change_match(pred: List[Dict[str, Any]], exp: List[Dict[str, Any]]) -> bool:
    """Predicted minimum changes equal the labelled ones (as multisets), compared only on the keys the label states."""
    if len(pred) != len(exp): return False
    left = [dict(p) for p in pred]
    for e in exp:
        keys = tuple(sorted(e))
        hit = next((i for i, p in enumerate(left) if _cfb_norm_change(p, keys) == _cfb_norm_change(e, keys)), None)
        if hit is None: return False
        left.pop(hit)
    return True

def _cfb_replay_docs(case: Dict[str, Any], pred: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The case documents with the PREDICTED counterfactual applied (replacement value / phrase taken from the prediction). None when the prediction cannot be applied."""
    rp, docs = case["replay"], dict(case["documents"]); mc = pred.get("minimum_changes") or []
    def fmt(v: float) -> str: return f"{int(v):,}" if float(v).is_integer() else f"{v:,}"
    try:
        if rp["kind"] == "amount_replace":
            if "to" in rp: new = rp["to"]  # amounts-must-match: the replay equalises the two records (which one is authoritative is a human decision, declared by the case)
            else:
                m = mc[0]; tgt = m["boundary"] if m["boundary_inclusive"] else m["boundary"] + rp["step"]; new = fmt(tgt)
            docs[rp["file"]] = docs[rp["file"]].replace(rp["old"], new)
        elif rp["kind"] == "amount_replace_many":
            for (fn, old), m in zip(sorted(rp["files"].items()), sorted(mc, key=lambda x: -x["current_value"])):
                tgt = m["boundary"] if m["boundary_inclusive"] else m["boundary"] + rp["step"]; docs[fn] = docs[fn].replace(old, fmt(tgt))
        elif rp["kind"] == "append_phrase": docs[rp["file"]] = docs[rp["file"]] + f"Note: {mc[0]['phrase']}\n"
        elif rp["kind"] == "remove_phrase": docs[rp["file"]] = re.sub(re.escape(mc[0]["phrase"]), "bank transfer", docs[rp["file"]], flags=re.IGNORECASE)
        else: return None
    except (KeyError, IndexError, TypeError): return None
    return docs

def _cfb_build(docs: Dict[str, str], rulebook: str, workdir: str) -> nx.MultiDiGraph:
    paths = []
    for name, text in docs.items():
        p = os.path.join(workdir, name)
        with open(p, "w", encoding="utf-8") as fh: fh.write(text)
        paths.append(p)
    rb = os.path.join(workdir, "rulebook.txt")
    with open(rb, "w", encoding="utf-8") as fh: fh.write(rulebook)
    return build_evidence_graph(paths, rb)

def _cfb_stable_text(x: Any) -> Any:
    return re.sub(r"\b([a-z]+)_[0-9a-f]{16}\b", r"<\1>", x) if isinstance(x, str) else x

def _cfb_record(case: Dict[str, Any], G: nx.MultiDiGraph, workdir: str) -> Dict[str, Any]:
    import tempfile
    target = _cfb_target(G, case)
    if target is None: raise ValueError(f"{case['case_id']}: target did not map to exactly one {case['target_kind']}")
    if case["target_kind"] == "decision": _sv7d_apply_fault(G, target[1], case.get("fault"))
    before = _sv7d_snapshot(G)
    pred = _cfb_predict(G, case, target); pred2 = _cfb_predict(G, case, target)
    if _sv7d_snapshot(G) != before: raise RuntimeError("counterfactual benchmark mutated the graph")
    refs = pred.get("evidence_causing_violation") or []
    ids = [r["evidence_id"] for r in refs if r.get("evidence_id")]
    integ = {"evidence_ids_in_graph": all(G.has_node(i) and G.nodes[i].get("type") == "Evidence" for i in ids),
             "provenance_matches_graph": all(r.get("provenance") == _sv_json(G.nodes[r["evidence_id"]].get("provenance")) for r in refs if r.get("evidence_id") and G.has_node(r["evidence_id"])),
             "contract_problems": validate_counterfactual(G, pred)}
    rule = pred.get("violated_rule") if isinstance(pred.get("violated_rule"), dict) else {}
    rule_ok = bool(rule.get("rule_id") and G.has_node(rule["rule_id"]) and G.nodes[rule["rule_id"]].get("type") == "PolicyRule")
    rule_is_target = bool(rule_ok and case["target_rule"].lower() in str(G.nodes[rule["rule_id"]].get("condition") or "").lower())
    after = None
    if pred["status"] == "ESTABLISHED" and case.get("replay"):
        rd = _cfb_replay_docs(case, pred)
        if rd is not None:
            with tempfile.TemporaryDirectory() as td2:
                G2 = _cfb_build(rd, case["rulebook"], td2); d2 = _sv7d_target_decision(G2, case["target_rule"])
                after = G2.nodes[d2].get("verdict") if d2 else None
    stable = _sv7d_stable_refs([{"filename": r.get("filename"), "location": r.get("location"), "file_hash": None, "provenance": r.get("provenance")} for r in refs if r.get("evidence_id")])
    files = sorted({r.get("filename") for r in refs if r.get("filename")})
    return {"case_id": case["case_id"], "category": case["category"], "fault": case.get("fault"), "target_kind": case["target_kind"], "counterfactual_id_stable": pred["counterfactual_id"] == pred2["counterfactual_id"],
            "deterministic": json.dumps(_sv_json(pred), sort_keys=True) == json.dumps(_sv_json(pred2), sort_keys=True),
            "expected_status": case["expected_status"], "predicted_status": pred["status"], "status_correct": pred["status"] == case["expected_status"],
            "expected_change_type": case["expected_change_type"], "predicted_change_type": pred["change_type"],
            "expected_required_condition": case["expected_required_condition"], "predicted_required_condition": _sv_json(pred.get("required_condition")),
            "expected_minimum_changes": case["expected_minimum_changes"], "predicted_minimum_changes": [{k: v for k, v in m.items() if not k.endswith("_id") and not k.endswith("_ids")} for m in pred.get("minimum_changes") or []],
            "expected_evidence_files": case["expected_evidence_files"], "predicted_evidence_files": files, "evidence_refs": stable,
            "rule_referenced": rule_ok, "rule_is_target": rule_is_target, "rule_condition": rule.get("condition"), "rule_source_location": rule.get("source_location"),
            "replay_expected_verdict": case["expected_after_verdict"], "replay_verdict": after if after is not None else (NOT_MEASURED if case.get("replay") is None else "REPLAY_NOT_APPLICABLE"),
            "unresolved_reason": _cfb_stable_text(pred.get("unresolved_reason")), "contradiction_context": [{k: v for k, v in c.items() if k not in ("claims",)} for c in pred.get("contradiction_context") or []],
            "id_integrity": integ, "graph_unchanged": True}

def _cfb_run_case(case: Dict[str, Any], workdir: str) -> Dict[str, Any]:
    G = _cfb_build(case["documents"], case["rulebook"], workdir)
    return _cfb_record(case, G, workdir)

def _cfb_metric(num: int, den: int, reason: Optional[str] = None) -> Dict[str, Any]:
    return _sv7d_metric(num, den, reason)

def aggregate_counterfactual_results(records: List[Dict[str, Any]], cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Derived from the raw per-case records only. A metric with an empty denominator is NOT_MEASURED, never 0.0."""
    cases = COUNTERFACTUAL_BENCHMARK if cases is None else cases
    est = [r for r in records if r["expected_status"] == "ESTABLISHED" and r["predicted_status"] == "ESTABLISHED"]
    rep = [r for r in est if r["replay_expected_verdict"] is not None]
    minc = [r for r in records if r["expected_status"] == "ESTABLISHED" and r["expected_minimum_changes"] is not None]
    cond = [r for r in records if r["expected_required_condition"] is not None]
    grd = [r for r in records if r["expected_evidence_files"] is not None]
    nonest = [r for r in records if r["expected_status"] != "ESTABLISHED"]
    ctx = [r for r in records if r["predicted_status"] == "ESTABLISHED"]
    m = {"counterfactual_validity": _cfb_metric(sum(r["replay_verdict"] == r["replay_expected_verdict"] for r in rep), len(rep), "no established counterfactual with a replay label"),
         "policy_consistency": _cfb_metric(sum(r["rule_referenced"] and r["rule_is_target"] and r["predicted_required_condition"] == r["expected_required_condition"] for r in cond), len(cond), "no case labelled with a required condition"),
         "evidence_grounding": _cfb_metric(sum(r["predicted_evidence_files"] == sorted(r["expected_evidence_files"]) and r["id_integrity"]["evidence_ids_in_graph"] and r["id_integrity"]["provenance_matches_graph"] and not r["id_integrity"]["contract_problems"] for r in grd), len(grd), "no case labelled with expected evidence files"),
         "minimum_change_accuracy": _cfb_metric(sum(r["predicted_status"] == "ESTABLISHED" and r["predicted_change_type"] == r["expected_change_type"] and _cfb_change_match(r["predicted_minimum_changes"], r["expected_minimum_changes"]) for r in minc), len(minc), "no established case labelled with a minimum change"),
         "status_accuracy": _cfb_metric(sum(r["status_correct"] for r in records), len(records)),
         "change_type_accuracy": _cfb_metric(sum(r["predicted_change_type"] == r["expected_change_type"] for r in records if r["expected_status"] == "ESTABLISHED"), len([r for r in records if r["expected_status"] == "ESTABLISHED"]), "no established case"),
         "invented_counterfactual_rate": _cfb_metric(sum(r["predicted_status"] == "ESTABLISHED" for r in nonest), len(nonest), "no case where no counterfactual may exist"),
         "contract_validity": _cfb_metric(sum(not r["id_integrity"]["contract_problems"] for r in ctx), len(ctx), "no established counterfactual")}
    return {"benchmark": {"id": CFB_ID, "version": CFB_VERSION, "size": len(cases), "category_counts": dict(Counter(c["category"] for c in cases)), "expected_status_counts": dict(Counter(c["expected_status"] for c in cases)),
                          "expected_change_type_counts": dict(Counter(str(c["expected_change_type"]) for c in cases)), "seed": None, "llm_used": False}, "metrics": m, "protocol": CFB_PROTOCOL}

def run_counterfactual_benchmark(cases: Optional[List[Dict[str, Any]]] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Offline and deterministic. Every case builds its own private graph; returns {"records", "aggregate"}; writes JSON only if output_path is given."""
    import tempfile
    cases = COUNTERFACTUAL_BENCHMARK if cases is None else cases
    probs = validate_counterfactual_benchmark(cases, require_coverage=cases is COUNTERFACTUAL_BENCHMARK)
    if probs: raise ValueError("counterfactual benchmark labels inconsistent: " + "; ".join(probs[:5]))
    records: List[Dict[str, Any]] = []
    for c in cases:
        with tempfile.TemporaryDirectory() as td: records.append(_cfb_run_case(c, td))
    result = _json_safe({"benchmark_id": CFB_ID, "version": CFB_VERSION, "records": records, "aggregate": aggregate_counterfactual_results(records, cases)})
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh: json.dump(result, fh, indent=2, ensure_ascii=False)
    return result

# --- UNCERTAINTY LAYER (Phase 9; READ-ONLY, additive: the Decision verdict, rule engine, evidence graph, contradiction, self-verification (7A-7C) and counterfactual (Phase 8) layers are REUSED unchanged) ---
# Maps each existing evidence-backed Decision into ONE of four uncertainty states and derives measurable indicators from EXISTING deterministic information only (7C result, stored Decision / Risk / PolicyRule
# attributes, contradiction findings). No node / edge is created, no verdict / evidence / rule / provenance is altered, nothing is invented, no LLM / network / randomness. Recomputed on every call (idempotent).
#   uncertainty_status:   COMPLIANT | NON_COMPLIANT | CONDITIONAL | INSUFFICIENT_EVIDENCE   (NOT_MEASURED only when the Decision does not exist)
#     asserting verdicts (VIOLATION / SATISFIED), first match wins:
#       7B FAILED or any UNSUPPORTED material claim ................................ INSUFFICIENT_EVIDENCE (the conclusion is not grounded)
#       policy applicability MISMATCH / UNESTABLISHED (7C) ............................ CONDITIONAL
#       critical evidence missing or a WEAKLY grounded material claim ................ INSUFFICIENT_EVIDENCE
#       unresolved MAJOR contradiction / opposite-verdict evidence ................... CONDITIONAL
#       otherwise ..................................................................... NON_COMPLIANT (VIOLATION) / COMPLIANT (SATISFIED)
#     INCONCLUSIVE -> INSUFFICIENT_EVIDENCE | UNEVALUATED (rule not interpretable / configurable) -> CONDITIONAL | NOT_APPLICABLE (no fact the rule applies to) -> INSUFFICIENT_EVIDENCE (never COMPLIANT: absence of facts is not compliance)
#   Indicators are {value, status, basis}: value is a categorical label or None; status MEASURED | NOT_MEASURED. No numeric score is produced (no calibrated confidence / completeness ratio exists).
#     decision_confidence    always None / NOT_MEASURED (no calibrated decision-level confidence); stored components are listed, never aggregated; `low_components` = compiler-reported rule confidences below the compiler's own minimum
#     evidence_completeness  COMPLETE | INCOMPLETE | NOT_MEASURED     contradiction_severity  NONE | MINOR | MAJOR | UNCLASSIFIED | NOT_MEASURED (NONE only when the contradiction classifier has run)
#     policy_alignment       ALIGNED | MISALIGNED | UNESTABLISHED | NOT_APPLICABLE | NOT_MEASURED (from 7C / verdict)     risk_level  stored Risk.severity (LOW | MEDIUM | HIGH | CRITICAL) or None / NOT_MEASURED
#   escalation_required is True exactly when escalation_reasons is non-empty; reasons: LOW_CONFIDENCE | CRITICAL_EVIDENCE_MISSING | UNRESOLVED_CONTRADICTION | POLICY_APPLICABILITY_UNESTABLISHED | HIGH_RISK (HIGH / CRITICAL).
UNC_VERSION = "UNC1"
UNC_STATES = ("COMPLIANT", "NON_COMPLIANT", "CONDITIONAL", "INSUFFICIENT_EVIDENCE")
UNC_ESCALATION_CODES = ("LOW_CONFIDENCE", "CRITICAL_EVIDENCE_MISSING", "UNRESOLVED_CONTRADICTION", "POLICY_APPLICABILITY_UNESTABLISHED", "HIGH_RISK")
UNC_INDICATORS = ("decision_confidence", "evidence_completeness", "contradiction_severity", "policy_alignment", "risk_level")
UNC_FIELDS = ("decision_id", "found", "decision", "uncertainty_status") + UNC_INDICATORS + ("escalation_required", "escalation_reasons", "uncertainty_state_reason")
UNC_HIGH_RISK = ("HIGH", "CRITICAL")

def _unc_ind(value: Optional[str], basis: str, **extra) -> Dict[str, Any]:
    return {"value": value, "status": "MEASURED" if value is not None else NOT_MEASURED, "basis": basis, **extra}

def build_uncertainty_assessment(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Any]:
    """Read-only; recomputed on every call (idempotent). All contract fields are always present; the Decision verdict is copied, never changed."""
    res: Dict[str, Any] = {"contract_version": UNC_VERSION, "decision_id": decision_id, "found": False, "decision": None, "uncertainty_status": NOT_MEASURED, "uncertainty_state_reason": None,
                           **{k: _unc_ind(None, "decision not found") for k in UNC_INDICATORS}, "escalation_required": False, "escalation_reasons": [],
                           "verification_status": None, "policy_applicability_status": None, "supporting_evidence": [], "contradicting_evidence": [], "policy_rules": [], "missing_evidence": [], "gaps": []}
    if not G.has_node(decision_id) or G.nodes[decision_id].get("type") != "Decision":
        res["gaps"].append("start node missing or not a Decision"); return res
    vr = verify_policy_applicability(G, decision_id)  # reused unchanged: 7A lineage + 7B material claims + 7C applicability
    dd = G.nodes[decision_id]; verdict = dd.get("verdict")
    pa = (vr.get("policy_applicability") or {}).get("status", "NOT_CHECKED"); vs = vr.get("verification_status")
    res.update(found=True, decision={k: vr["decision"].get(k) for k in ("decision_id", "verdict", "rule_id", "result_source", "violation_status")}, verification_status=vs, policy_applicability_status=pa,
               supporting_evidence=_sv_json(vr["supporting_evidence"]), contradicting_evidence=_sv_json(vr["contradicting_evidence"]), policy_rules=_sv_json(vr["policy_rules"]), gaps=list(vr.get("gaps") or []))
    claims = vr.get("material_claims") or []
    ungrounded, weak = [c for c in claims if c["grounding"] == "UNSUPPORTED"], [c for c in claims if c["grounding"] == "WEAK"]
    asserting = verdict in SV_ASSERTING_VERDICTS
    absence_ok = dd.get("violation_status") == "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE" and any(c["kind"] == "absence_within_scope" and c["grounding"] != "UNSUPPORTED" for c in claims)
    miss = [m for m in vr["missing_evidence"] if m.get("kind") != "rule_not_evaluated" and not (absence_ok and m.get("kind") == "no_supporting_evidence")]  # an unevaluated rule is a policy problem (policy_alignment), not missing evidence
    res["missing_evidence"] = _sv_json(miss)
    # decision_confidence: never aggregated, never invented
    comps = vr["confidence"]["components"]; floor = globals().get("_COMPILER_MIN_CONFIDENCE")
    _num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    if not _num(floor): floor = None
    low = [c for c in comps["compiled_rules"] if floor is not None and _num(c.get("value")) and c["value"] < floor]
    res["decision_confidence"] = _unc_ind(None, "no calibrated decision-level confidence exists; stored components are listed, never aggregated", components=_sv_json(comps), low_components=_sv_json(low), low_confidence_floor=floor)
    # evidence_completeness
    if asserting:
        gone = bool(miss or ungrounded or weak)
        res["evidence_completeness"] = _unc_ind("INCOMPLETE" if gone else "COMPLETE", f"{len(miss)} missing-evidence item(s), {len(ungrounded)} unsupported and {len(weak)} weakly grounded material claim(s), {len(vr['supporting_evidence'])} supporting evidence reference(s)")
    elif verdict == "INCONCLUSIVE": res["evidence_completeness"] = _unc_ind("INCOMPLETE", f"verdict INCONCLUSIVE: the rule was attempted without a determinate result; {len(miss)} recorded missing-evidence item(s)")
    else: res["evidence_completeness"] = _unc_ind(None, f"verdict {verdict} asserts no conclusion to complete evidence for")
    # contradiction_severity
    cs = [c for c in vr["contradicting_evidence"] if c.get("source") == "contradiction_finding"]
    major = [c for c in cs if c.get("category") == "MAJOR_CONTRADICTION"] + [c for c in vr["contradicting_evidence"] if c.get("source") == "opposite_verdict_basis"]
    minor = [c for c in cs if c.get("category") == "MINOR_CONTRADICTION"]
    edges = [c for c in vr["contradicting_evidence"] if c.get("source") == "CONTRADICTS_edge"]
    unresolved_c = [c for c in cs if c.get("resolution") != "RESOLVED"] + [c for c in vr["contradicting_evidence"] if c.get("source") != "contradiction_finding"]
    major_open = [c for c in major if c.get("source") != "contradiction_finding" or c.get("resolution") != "RESOLVED"]  # MAJOR severity is reported as classified; only an UNRESOLVED major drives the state
    cb = f"{len(major)} major / opposite-verdict, {len(minor)} minor classified finding(s); {len(edges)} heuristic CONTRADICTS edge(s) without a severity classification"
    if major: res["contradiction_severity"] = _unc_ind("MAJOR", cb)
    elif minor: res["contradiction_severity"] = _unc_ind("MINOR", cb)
    elif edges: res["contradiction_severity"] = _unc_ind("UNCLASSIFIED", cb)
    elif G.graph.get("contradiction_findings") is not None: res["contradiction_severity"] = _unc_ind("NONE", "contradiction classifier ran: no finding touches this decision's supporting evidence")
    else: res["contradiction_severity"] = _unc_ind(None, "contradiction classifier has not run on this graph")
    # policy_alignment
    pal = {"VERIFIED": "ALIGNED", "MISMATCH": "MISALIGNED", "UNESTABLISHED": "UNESTABLISHED"}.get(pa) if asserting else {"UNEVALUATED": "UNESTABLISHED", "NOT_APPLICABLE": "NOT_APPLICABLE"}.get(verdict)
    res["policy_alignment"] = _unc_ind(pal, f"7C policy applicability {pa}" if asserting else f"verdict {verdict}" + (f": {dd.get('unevaluated_reason')}" if verdict == "UNEVALUATED" and dd.get("unevaluated_reason") else ""))
    # risk_level: only a stored Risk node (created by the rule engine for VIOLATION) is a risk classification
    rk = [(v, G.nodes[v]) for _, v, e in G.out_edges(decision_id, data=True) if e.get("relation") == "HAS_RISK" and G.has_node(v) and G.nodes[v].get("type") == "Risk" and str(G.nodes[v].get("severity")).lower() in _COMPILED_SEV]  # unrecognised severity text is not a risk classification
    if rk:
        top = max(rk, key=lambda x: (_COMPILED_SEV.get(str(x[1]["severity"]).lower(), 0), x[0]))
        res["risk_level"] = _unc_ind(str(top[1]["severity"]).upper(), "stored Risk node severity (highest when several)", risk_ids=sorted(v for v, _ in rk), severity_source=top[1].get("severity_source"))
    else: res["risk_level"] = _unc_ind(None, "no Risk node recorded for this decision (the rule engine records one only for VIOLATION)")
    # uncertainty_status
    if asserting:
        if vs == "FAILED" or ungrounded: st, why = "INSUFFICIENT_EVIDENCE", f"{len(ungrounded)} unsupported material claim(s): the {verdict} conclusion is not grounded in source evidence"
        elif pa in ("MISMATCH", "UNESTABLISHED"): st, why = "CONDITIONAL", f"policy applicability {pa}: the cited rule is not established to apply to the verified facts"
        elif miss or weak: st, why = "INSUFFICIENT_EVIDENCE", f"{len(miss)} missing-evidence item(s) and {len(weak)} weakly grounded material claim(s): the {verdict} conclusion is not adequately supported"
        elif major_open: st, why = "CONDITIONAL", f"{len(major_open)} unresolved major contradiction / opposite-verdict evidence item(s)"
        elif pa != "VERIFIED": st, why = "CONDITIONAL", f"policy applicability {pa}: not established"
        else: st, why = ("COMPLIANT", "SATISFIED with established policy applicability, complete evidence and no unresolved major contradiction") if verdict == "SATISFIED" else ("NON_COMPLIANT", "VIOLATION adequately supported: grounded evidence, established policy applicability, no unresolved major contradiction")
    elif verdict == "INCONCLUSIVE": st, why = "INSUFFICIENT_EVIDENCE", "verdict INCONCLUSIVE: the rule was attempted but evidence was insufficient for a determinate result"
    elif verdict == "UNEVALUATED": st, why = "CONDITIONAL", "verdict UNEVALUATED: the rule was not interpretable / configurable, so policy applicability is unestablished and compliance was not checked"
    elif verdict == "NOT_APPLICABLE": st, why = "INSUFFICIENT_EVIDENCE", "verdict NOT_APPLICABLE: no fact the rule applies to was found; absence of facts is not evidence of compliance"
    else: st, why = "INSUFFICIENT_EVIDENCE", f"verdict {verdict} has no defined uncertainty mapping"; res["gaps"].append(why)
    res["uncertainty_status"], res["uncertainty_state_reason"] = st, why
    # escalation: every reason is backed by existing deterministic information
    R = res["escalation_reasons"]
    if low: R.append({"reason": "LOW_CONFIDENCE", "detail": f"{len(low)} compiler-reported rule confidence value(s) below the compiler minimum {floor}"})
    if miss or ungrounded or weak or vs == "FAILED" or verdict == "INCONCLUSIVE":
        R.append({"reason": "CRITICAL_EVIDENCE_MISSING", "detail": "; ".join(([f"{len(miss)} missing-evidence item(s): " + ", ".join(sorted({m.get('kind') for m in miss}))] if miss else []) + ([f"{len(ungrounded)} unsupported material claim(s)"] if ungrounded else []) + ([f"{len(weak)} weakly grounded material claim(s)"] if weak else []) + (["verdict INCONCLUSIVE"] if verdict == "INCONCLUSIVE" else []) + (["7B self-verification FAILED"] if vs == "FAILED" else []))})
    if unresolved_c: R.append({"reason": "UNRESOLVED_CONTRADICTION", "detail": f"{len(unresolved_c)} unresolved contradicting evidence item(s); severity {res['contradiction_severity']['value']}"})
    if (asserting and vs != "FAILED" and pa != "VERIFIED") or verdict == "UNEVALUATED": R.append({"reason": "POLICY_APPLICABILITY_UNESTABLISHED", "detail": (f"7C policy applicability {pa}" if asserting else "verdict UNEVALUATED: rule not evaluated")})
    if res["risk_level"]["value"] in UNC_HIGH_RISK: R.append({"reason": "HIGH_RISK", "detail": f"stored Risk severity {res['risk_level']['value']}"})
    res["escalation_required"] = bool(R)
    return res

def build_uncertainty_assessments(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """One assessment per Decision, in a stable order."""
    return [build_uncertainty_assessment(G, n) for n, _ in sorted(_nodes_of_type(G, "Decision"), key=lambda kv: kv[0])]

def validate_uncertainty_assessment(G: nx.MultiDiGraph, r: Dict[str, Any]) -> List[str]:
    """Contract check: fields present, state allowed, no numeric indicator, verdict equals the stored Decision verdict, every cited id exists (nothing invented)."""
    probs = [f"missing field {k}" for k in UNC_FIELDS if k not in r]
    codes = [e.get("reason") for e in r.get("escalation_reasons") or []]
    if r.get("found"):
        if r.get("uncertainty_status") not in UNC_STATES: probs.append(f"unknown uncertainty_status {r.get('uncertainty_status')}")
        if G.has_node(r.get("decision_id")) and (r.get("decision") or {}).get("verdict") != G.nodes[r["decision_id"]].get("verdict"): probs.append("verdict differs from the stored Decision verdict")
        for ref in r.get("supporting_evidence") or []:
            if not (G.has_node(ref.get("evidence_id")) and G.nodes[ref["evidence_id"]].get("type") == "Evidence"): probs.append(f"supporting evidence {ref.get('evidence_id')} not in graph")
        for ru in r.get("policy_rules") or []:
            if not (G.has_node(ru.get("rule_id")) and G.nodes[ru["rule_id"]].get("type") == "PolicyRule"): probs.append(f"policy rule {ru.get('rule_id')} not in graph")
    for k in UNC_INDICATORS:
        ind = r.get(k) or {}
        if isinstance(ind.get("value"), (int, float)) and not isinstance(ind.get("value"), bool): probs.append(f"{k}.value must not be numeric")
        if ind.get("status") not in ("MEASURED", NOT_MEASURED): probs.append(f"{k}: unknown status {ind.get('status')}")
        if (ind.get("value") is None) != (ind.get("status") == NOT_MEASURED): probs.append(f"{k}: value None exactly when status is NOT_MEASURED")
    if (r.get("decision_confidence") or {}).get("value") is not None: probs.append("decision_confidence.value must stay None (NOT_MEASURED)")
    if r.get("found"):  # state / escalation must agree with the reused 7B / 7C statuses and the stored risk
        st, vs_, pa_ = r.get("uncertainty_status"), r.get("verification_status"), r.get("policy_applicability_status")
        asserting_ = (r.get("decision") or {}).get("verdict") in SV_ASSERTING_VERDICTS
        if vs_ == "FAILED" and (st != "INSUFFICIENT_EVIDENCE" or "CRITICAL_EVIDENCE_MISSING" not in codes): probs.append("7B FAILED requires INSUFFICIENT_EVIDENCE with CRITICAL_EVIDENCE_MISSING")
        if vs_ == "FAILED" and "POLICY_APPLICABILITY_UNESTABLISHED" in codes: probs.append("7B FAILED must not carry a policy applicability reason")
        if asserting_ and vs_ != "FAILED" and pa_ != "VERIFIED" and (st not in ("CONDITIONAL", "INSUFFICIENT_EVIDENCE") or "POLICY_APPLICABILITY_UNESTABLISHED" not in codes): probs.append("unverified policy applicability requires a non-final state with POLICY_APPLICABILITY_UNESTABLISHED")
        if st in ("COMPLIANT", "NON_COMPLIANT") and (vs_ == "FAILED" or (asserting_ and pa_ != "VERIFIED")): probs.append(f"{st} requires verified self-verification and policy applicability")
        if (r.get("risk_level") or {}).get("value") not in (None, "LOW", "MEDIUM", "HIGH", "CRITICAL"): probs.append("risk_level.value is not a recognised stored severity")
        fl = (r.get("decision_confidence") or {}).get("low_confidence_floor")
        if fl is not None and (isinstance(fl, bool) or not isinstance(fl, (int, float)) or not math.isfinite(fl)): probs.append("low_confidence_floor must be a finite number or None")
    if r.get("escalation_required") != bool(r.get("escalation_reasons")): probs.append("escalation_required must be True exactly when escalation_reasons is non-empty")
    probs += [f"unknown escalation reason {c}" for c in codes if c not in UNC_ESCALATION_CODES]
    if len(codes) != len(set(codes)): probs.append("duplicate escalation reasons")
    if "HIGH_RISK" in codes and (r.get("risk_level") or {}).get("value") not in UNC_HIGH_RISK: probs.append("HIGH_RISK without a stored HIGH / CRITICAL risk")
    if "LOW_CONFIDENCE" in codes and not (r.get("decision_confidence") or {}).get("low_components"): probs.append("LOW_CONFIDENCE without a low compiler confidence component")
    if "UNRESOLVED_CONTRADICTION" in codes and (r.get("contradiction_severity") or {}).get("value") in (None, "NONE"): probs.append("UNRESOLVED_CONTRADICTION without a recorded contradiction")
    return probs


# --- UNCERTAINTY EVALUATION BENCHMARK (Phase 9C; READ-ONLY / ADDITIVE: only CALLS the 9A/9B layer, the 7B/7C verification and the rule engine; changes none of them) ---
# Deterministic, offline (no LLM / network / randomness). Every case builds its OWN private graph. Expected labels are HAND-WRITTEN literals in `ground_truth` (state, escalation, reasons); they are never read from, or computed by, any system output
# (validate_uncertainty_benchmark rejects anything else). Three variants are scored against the same labels:
#   binary_verdict            VIOLATION -> NON_COMPLIANT, every other verdict -> COMPLIANT; never escalates                          (naive baseline: compliance by absence of violation)
#   verdict_with_verification verdict -> state (SATISFIED C, VIOLATION NC, INCONCLUSIVE / NOT_APPLICABLE INSUFFICIENT, UNEVALUATED CONDITIONAL); escalates iff 7C verification_status is FAILED / ESCALATE   (stronger baseline)
#   uncertainty_aware         the existing 9A / 9B assessment, unchanged
# Metric definitions (UNCB_DEFINITIONS) are explicit; CONDITIONAL / INSUFFICIENT_EVIDENCE are never silently folded into COMPLIANT / NON_COMPLIANT. Every metric is {status, value, numerator, denominator, ...}: MEASURED (value may be a genuine 0.0),
# NOT_MEASURED with reason_code EMPTY_DENOMINATOR (nothing to divide by) or NO_NUMERIC_CONFIDENCE (calibration inputs absent). ECE / Brier / calibration use ONLY a genuine finite numeric decision_confidence in [0, 1]; the existing layer returns
# NOT_MEASURED confidence, so on this benchmark they are NOT_MEASURED by design (no probability is invented, no categorical value is converted). The arithmetic is exercised on records that carry a genuine number.
UNCB_ID, UNCB_VERSION = "omni-uncertainty-bench", "1.0"
UNCB_VARIANTS = ("binary_verdict", "verdict_with_verification", "uncertainty_aware")
UNCB_FINAL_STATES = ("COMPLIANT", "NON_COMPLIANT")
UNCB_CATEGORIES = ("compliant", "non_compliant", "conditional", "insufficient_evidence", "missing_critical_evidence", "unresolved_contradiction", "resolved_contradiction", "policy_mismatch", "low_confidence", "high_risk", "unsupported_ambiguous", "not_applicable")
UNCB_CASE_KEYS = {"case_id", "category", "documents", "rulebook", "target_rule", "fault", "setup", "ground_truth", "note"}
UNCB_GT_KEYS = {"state", "escalation", "reasons"}
UNCB_BINS = 10
UNCB_DEFINITIONS = {
    "state_accuracy": "predicted uncertainty state == labelled state, over all cases",
    "false_positive_rate": "predicted NON_COMPLIANT among cases labelled COMPLIANT (a violation flagged where the case is compliant); denominator = cases labelled COMPLIANT",
    "false_negative_rate": "predicted COMPLIANT among cases labelled NON_COMPLIANT (a violation missed); denominator = cases labelled NON_COMPLIANT",
    "unsafe_finalization_rate": "predicted COMPLIANT / NON_COMPLIANT among cases labelled CONDITIONAL / INSUFFICIENT_EVIDENCE (an unresolved case closed as final); denominator = those cases",
    "escalation_rate": "escalated cases / all cases (predicted); expected_escalation_rate is the same quantity from the labels only",
    "escalation_false_positive_rate": "escalated among cases labelled no-escalation; denominator = cases labelled no-escalation",
    "escalation_false_negative_rate": "not escalated among cases labelled escalation; denominator = cases labelled escalation",
    "escalation_agreement": "predicted escalation == labelled escalation, over all cases",
    "automation_coverage": "cases eligible for automated final handling / all cases; eligible = predicted state in {COMPLIANT, NON_COMPLIANT} AND not escalated",
    "automated_decision_accuracy": "among automation-eligible cases, predicted state == labelled state; NOT_MEASURED (EMPTY_DENOMINATOR) when none is eligible",
    "calibration / ece / brier": "confidence = genuine finite numeric decision_confidence in [0, 1]; outcome = predicted state == labelled state; 10 equal-width bins; ECE = sum(n_b / N * |accuracy_b - mean_confidence_b|); Brier = mean((confidence - outcome)^2); NOT_MEASURED (NO_NUMERIC_CONFIDENCE) when no case carries one"}
UNCB_LIMITATIONS = ("The existing uncertainty layer returns decision_confidence NOT_MEASURED (no calibrated decision-level probability exists), so calibration, ECE and Brier are NOT_MEASURED on this benchmark; no confidence model was added and nothing is estimated.",
                    "Labels are small hand-written fixtures (single-rule cases); rates over small denominators are descriptive, not statistically powered; no expert / human ground-truth evaluation is included.")
UNCB_PROTOCOL = {"seed": None, "randomness": "none", "llm_used": False, "network_used": False, "variants": list(UNCB_VARIANTS), "definitions": UNCB_DEFINITIONS, "limitations": list(UNCB_LIMITATIONS),
                 "isolation": "every case builds a private graph; setups / faults touch only that graph; a graph snapshot taken before scoring must equal the one after, else the run aborts"}
_UNCB_FLOOR = 0.6

def _uncb_set_low(G: nx.MultiDiGraph, d: str) -> None: G.nodes[d]["compiled_rules"] = [{"rule_id": "bench_component", "confidence": 0.2}]
def _uncb_set_high(G: nx.MultiDiGraph, d: str) -> None: G.nodes[d]["compiled_rules"] = [{"rule_id": "bench_component", "confidence": 0.9}]
def _uncb_inconclusive(G: nx.MultiDiGraph, d: str) -> None: G.nodes[d]["verdict"] = "INCONCLUSIVE"
def _uncb_resolve_major(G: nx.MultiDiGraph, d: str) -> None:
    for f in G.graph.get("contradiction_findings") or []:
        if f["category"] == "MAJOR_CONTRADICTION": f["resolution"] = "RESOLVED"
def _uncb_minor(G: nx.MultiDiGraph, d: str) -> None:
    for f in G.graph.get("contradiction_findings") or []:
        if f["category"] == "MAJOR_CONTRADICTION": f["category"], f["severity"] = "MINOR_CONTRADICTION", "minor"
UNCB_SETUPS = {"verdict_inconclusive": (_uncb_inconclusive, None), "resolve_major": (_uncb_resolve_major, None), "minor_contradiction": (_uncb_minor, None),
               "low_confidence_component": (_uncb_set_low, _UNCB_FLOOR), "confidence_component_above_floor": (_uncb_set_high, _UNCB_FLOOR)}  # name -> (private-graph fixture, compiler floor in force while scoring)
UNCB_FAULTS = ("evidence_text_5000_to_500", "remove_support", "rule_condition_9000", "basis_amount_none", "wrong_scope", "weak_location")

def _uncb_case(cid, category, docs, rulebook, state, escalation, reasons=(), target="TRANSACTION", fault=None, setup=None, note=""):
    return {"case_id": cid, "category": category, "documents": docs, "rulebook": rulebook, "target_rule": target, "fault": fault, "setup": setup, "note": note,
            "ground_truth": {"state": state, "escalation": escalation, "reasons": sorted(reasons)}}

_UB_INV5, _UB_INV8 = "Invoice No: INV-1\nBilled amount INR 5,000\n", "Invoice No: INV-2\nBilled amount INR 800\n"
_UB_R, _UB_KW = "FORBID TRANSACTION > INR 1000\n", 'REQUIRE KEYWORD "manager approval"\n'
_UB_POC = {"inv_c.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 800\n", "po_c.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"}
_UB_REF = {"inv_r.txt": "Invoice No: INV-7\nPO Number: PO-777\nVendor: Boreal\nBilled amount INR {amt}\n"}
# Labels follow the POLICY reading of each fixture, written before any output was inspected: a final state needs grounded evidence, established rule applicability and no open major contradiction; escalation needs one of the five reasons.
UNCERTAINTY_BENCHMARK: List[Dict[str, Any]] = [
    _uncb_case("ub_compliant_amount", "compliant", {"i.txt": _UB_INV8}, _UB_R, "COMPLIANT", False, note="grounded amount under the limit"),
    _uncb_case("ub_compliant_keyword", "compliant", {"m.txt": "manager approval granted\n"}, _UB_KW, "COMPLIANT", False, target="KEYWORD"),
    _uncb_case("ub_noncompliant_amount", "non_compliant", {"i.txt": _UB_INV5}, _UB_R, "NON_COMPLIANT", False, note="default MEDIUM risk alone does not escalate"),
    _uncb_case("ub_noncompliant_absence", "non_compliant", {"m.txt": "notes\n"}, _UB_KW, "NON_COMPLIANT", False, target="KEYWORD", note="required text absent within a complete extraction scope"),
    _uncb_case("ub_noncompliant_low_risk", "non_compliant", {"i.txt": _UB_INV5}, "FORBID TRANSACTION > INR 1000 [severity=low]\n", "NON_COMPLIANT", False),
    _uncb_case("ub_high_risk", "high_risk", {"i.txt": _UB_INV5}, "FORBID TRANSACTION > INR 1000 [severity=high]\n", "NON_COMPLIANT", True, ["HIGH_RISK"]),
    _uncb_case("ub_critical_risk", "high_risk", {"i.txt": _UB_INV5}, "FORBID TRANSACTION > INR 1000 [severity=critical]\n", "NON_COMPLIANT", True, ["HIGH_RISK"]),
    _uncb_case("ub_high_risk_and_missing_ref", "high_risk", {k: v.format(amt="5,000") for k, v in _UB_REF.items()}, "FORBID TRANSACTION > INR 1000 [severity=high]\n", "INSUFFICIENT_EVIDENCE", True, ["CRITICAL_EVIDENCE_MISSING", "HIGH_RISK"], note="violation present but a referenced document was never supplied"),
    _uncb_case("ub_inconclusive", "insufficient_evidence", {"i.txt": _UB_INV8}, _UB_R, "INSUFFICIENT_EVIDENCE", True, ["CRITICAL_EVIDENCE_MISSING"], setup="verdict_inconclusive"),
    _uncb_case("ub_missing_referenced_document", "missing_critical_evidence", {k: v.format(amt="800") for k, v in _UB_REF.items()}, _UB_R, "INSUFFICIENT_EVIDENCE", True, ["CRITICAL_EVIDENCE_MISSING"], note="satisfied on its face, but the referenced PO is missing"),
    _uncb_case("ub_text_no_longer_supports_claim", "missing_critical_evidence", {"i.txt": _UB_INV5}, _UB_R, "INSUFFICIENT_EVIDENCE", True, ["CRITICAL_EVIDENCE_MISSING"], fault="evidence_text_5000_to_500"),
    _uncb_case("ub_weak_source_location", "missing_critical_evidence", {"i.txt": _UB_INV5}, _UB_R, "INSUFFICIENT_EVIDENCE", True, ["CRITICAL_EVIDENCE_MISSING"], fault="weak_location"),
    _uncb_case("ub_unsupported_violation", "unsupported_ambiguous", {"i.txt": _UB_INV5}, _UB_R, "INSUFFICIENT_EVIDENCE", True, ["CRITICAL_EVIDENCE_MISSING"], fault="remove_support", note="violation with no qualifying supporting evidence"),
    _uncb_case("ub_rule_free_text", "unsupported_ambiguous", {"i.txt": _UB_INV5}, "Vendors should behave in spirit.\n", "CONDITIONAL", True, ["POLICY_APPLICABILITY_UNESTABLISHED"], target="Vendors", note="rule text cannot be interpreted"),
    _uncb_case("ub_rule_unknown_currency", "conditional", {"i.txt": _UB_INV5}, "FORBID TRANSACTION > JPY 1000\n", "CONDITIONAL", True, ["POLICY_APPLICABILITY_UNESTABLISHED"], note="unit outside the supported set: compliance not checked"),
    _uncb_case("ub_wrong_policy_scope", "policy_mismatch", {"i.txt": _UB_INV5}, _UB_R, "CONDITIONAL", True, ["POLICY_APPLICABILITY_UNESTABLISHED"], fault="wrong_scope"),
    _uncb_case("ub_rule_text_changed", "policy_mismatch", {"i.txt": _UB_INV5}, _UB_R, "CONDITIONAL", True, ["POLICY_APPLICABILITY_UNESTABLISHED"], fault="rule_condition_9000"),
    _uncb_case("ub_basis_amount_unknown", "policy_mismatch", {"i.txt": _UB_INV5}, _UB_R, "CONDITIONAL", True, ["POLICY_APPLICABILITY_UNESTABLISHED"], fault="basis_amount_none", note="required fact missing: applicability not established"),
    _uncb_case("ub_major_contradiction_vendor", "unresolved_contradiction", _UB_POC, _UB_R, "CONDITIONAL", True, ["UNRESOLVED_CONTRADICTION"]),
    _uncb_case("ub_major_contradiction_amount", "unresolved_contradiction", {"inv.txt": "Invoice No: INV-9\nPO Number: PO-9\nVendor: Boreal\nBilled amount INR 900\n", "po.txt": "Purchase Order No: PO-9\nVendor: Boreal\nOrder value INR 700\n"}, _UB_R, "CONDITIONAL", True, ["UNRESOLVED_CONTRADICTION"]),
    _uncb_case("ub_minor_contradiction_unresolved", "unresolved_contradiction", _UB_POC, _UB_R, "COMPLIANT", True, ["UNRESOLVED_CONTRADICTION"], setup="minor_contradiction", note="unresolved but minor: escalates, does not block the state"),
    _uncb_case("ub_major_contradiction_resolved", "resolved_contradiction", _UB_POC, _UB_R, "COMPLIANT", False, setup="resolve_major", note="the major finding is recorded RESOLVED: neither state nor escalation"),
    _uncb_case("ub_not_applicable", "not_applicable", {"memo.txt": "Meeting notes\nNothing to report\n"}, _UB_R, "INSUFFICIENT_EVIDENCE", False, note="absence of facts is not compliance; nothing evidential is missing"),
    _uncb_case("ub_low_confidence_component", "low_confidence", {"i.txt": _UB_INV8}, _UB_R, "COMPLIANT", True, ["LOW_CONFIDENCE"], setup="low_confidence_component", note="compiler-reported component 0.2 below the compiler floor 0.6"),
    _uncb_case("ub_confidence_above_floor", "low_confidence", {"i.txt": _UB_INV8}, _UB_R, "COMPLIANT", False, setup="confidence_component_above_floor", note="component 0.9 above the floor: automation proceeds")]

def validate_uncertainty_benchmark(cases: Optional[List[Dict[str, Any]]] = None, require_coverage: bool = True) -> List[str]:
    """Label consistency and independence: labels are literal data of fixed shape inside the case (never callables / outputs / extra keys), escalation == bool(reasons), every referenced fault / setup exists."""
    cases = UNCERTAINTY_BENCHMARK if cases is None else cases
    probs: List[str] = []
    ids = [c.get("case_id") for c in cases]
    if len(ids) != len(set(ids)): probs.append("duplicate case_id")
    for c in cases:
        cid = c.get("case_id")
        if set(c) != UNCB_CASE_KEYS: probs.append(f"{cid}: case keys {sorted(set(c) ^ UNCB_CASE_KEYS)} differ from the allowed schema (no prediction / output fields may be attached to a case)")
        if c.get("category") not in UNCB_CATEGORIES: probs.append(f"{cid}: unknown category {c.get('category')}")
        if c.get("fault") is not None and c["fault"] not in UNCB_FAULTS: probs.append(f"{cid}: unknown fault {c['fault']}")
        if c.get("setup") is not None and c["setup"] not in UNCB_SETUPS: probs.append(f"{cid}: unknown setup {c['setup']}")
        gt = c.get("ground_truth")
        if not isinstance(gt, dict) or set(gt) != UNCB_GT_KEYS: probs.append(f"{cid}: ground_truth must have exactly {sorted(UNCB_GT_KEYS)}"); continue
        if gt["state"] not in UNC_STATES: probs.append(f"{cid}: label state {gt['state']} not one of {UNC_STATES}")
        if not isinstance(gt["escalation"], bool): probs.append(f"{cid}: label escalation must be a bool literal")
        if not (isinstance(gt["reasons"], list) and all(isinstance(x, str) and x in UNC_ESCALATION_CODES for x in gt["reasons"]) and gt["reasons"] == sorted(set(gt["reasons"]))): probs.append(f"{cid}: label reasons must be a sorted unique list of known escalation codes")
        elif gt["escalation"] != bool(gt["reasons"]): probs.append(f"{cid}: label escalation must be True exactly when label reasons is non-empty")
        if gt["state"] in ("COMPLIANT", "NON_COMPLIANT") and not gt["escalation"] and c.get("category") in ("missing_critical_evidence", "unresolved_contradiction", "policy_mismatch", "insufficient_evidence"): probs.append(f"{cid}: category {c['category']} cannot be a non-escalated final state")
        try: json.dumps(c)
        except (TypeError, ValueError): probs.append(f"{cid}: case is not plain JSON data (labels must be literals)")
    if require_coverage:
        got = {c["category"] for c in cases if "category" in c}
        probs += [f"category {x} not covered" for x in UNCB_CATEGORIES if x not in got]
        gts = [c["ground_truth"] for c in cases if isinstance(c.get("ground_truth"), dict) and "state" in c["ground_truth"]]
        probs += [f"state {s} not covered" for s in UNC_STATES if s not in {g["state"] for g in gts}]
        probs += [f"escalation={b} not covered" for b in (True, False) if b not in {g["escalation"] for g in gts}]
        probs += [f"reason {x} not covered" for x in UNC_ESCALATION_CODES if x not in {y for g in gts for y in g["reasons"]}]
    return probs

def _uncb_floor(floor: Optional[float], fn):
    """Runs fn with the compiler confidence floor a case declares (restored afterwards); None leaves the module untouched."""
    if floor is None: return fn()
    had, old = "_COMPILER_MIN_CONFIDENCE" in globals(), globals().get("_COMPILER_MIN_CONFIDENCE")
    globals()["_COMPILER_MIN_CONFIDENCE"] = floor
    try: return fn()
    finally:
        if had: globals()["_COMPILER_MIN_CONFIDENCE"] = old
        else: globals().pop("_COMPILER_MIN_CONFIDENCE", None)

_UNCB_VERDICT_STATE = {"SATISFIED": "COMPLIANT", "VIOLATION": "NON_COMPLIANT", "INCONCLUSIVE": "INSUFFICIENT_EVIDENCE", "UNEVALUATED": "CONDITIONAL", "NOT_APPLICABLE": "INSUFFICIENT_EVIDENCE"}

def uncertainty_variant_predictions(G: nx.MultiDiGraph, decision_id: str) -> Dict[str, Dict[str, Any]]:
    """The three compared variants for one Decision, from EXISTING outputs only (read-only)."""
    a = build_uncertainty_assessment(G, decision_id); vr = verify_policy_applicability(G, decision_id); verdict = G.nodes[decision_id].get("verdict")
    return {"binary_verdict": {"state": "NON_COMPLIANT" if verdict == "VIOLATION" else "COMPLIANT", "escalated": False},
            "verdict_with_verification": {"state": _UNCB_VERDICT_STATE.get(verdict, "INSUFFICIENT_EVIDENCE"), "escalated": vr.get("verification_status") in ("FAILED", "ESCALATE")},
            "uncertainty_aware": {"state": a["uncertainty_status"], "escalated": a["escalation_required"], "reasons": sorted(e["reason"] for e in a["escalation_reasons"])}}

def _uncb_numeric_confidence(a: Dict[str, Any]) -> Optional[float]:
    v = (a.get("decision_confidence") or {}).get("value")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0 else None

def _uncb_run_case(case: Dict[str, Any], workdir: str) -> Dict[str, Any]:
    G = _cfb_build(case["documents"], case["rulebook"], workdir)
    d = _sv7d_target_decision(G, case["target_rule"])
    if d is None: raise ValueError(f"{case['case_id']}: target rule {case['target_rule']!r} does not select exactly one Decision")
    _sv7d_apply_fault(G, d, case["fault"])
    fix, floor = UNCB_SETUPS[case["setup"]] if case["setup"] else (None, None)
    if fix: fix(G, d)
    before = _sv7d_snapshot(G)
    def score():
        a1, a2 = build_uncertainty_assessment(G, d), build_uncertainty_assessment(G, d)
        return a1, a2, uncertainty_variant_predictions(G, d)
    a, a2, preds = _uncb_floor(floor, score)
    if _sv7d_snapshot(G) != before: raise RuntimeError("uncertainty benchmark mutated the graph")
    refs = a["supporting_evidence"]
    gt = case["ground_truth"]
    return {"case_id": case["case_id"], "category": case["category"], "setup": case["setup"], "fault": case["fault"], "verdict": G.nodes[d].get("verdict"),
            "expected_state": gt["state"], "expected_escalation": gt["escalation"], "expected_reasons": list(gt["reasons"]), "predictions": preds, "predicted_confidence": _uncb_numeric_confidence(a),
            "deterministic": json.dumps(_sv_json(a), sort_keys=True) == json.dumps(_sv_json(a2), sort_keys=True), "graph_unchanged": True,
            "id_integrity": {"evidence_ids_in_graph": all(G.has_node(r["evidence_id"]) and G.nodes[r["evidence_id"]].get("type") == "Evidence" for r in refs),
                             "provenance_matches_graph": all(r.get("provenance") == _sv_json(G.nodes[r["evidence_id"]].get("provenance")) for r in refs if G.has_node(r["evidence_id"])),
                             "rule_ids_in_graph": all(G.has_node(x["rule_id"]) and G.nodes[x["rule_id"]].get("type") == "PolicyRule" for x in a["policy_rules"]),
                             "verdict_preserved": (a["decision"] or {}).get("verdict") == G.nodes[d].get("verdict"), "contract_problems": validate_uncertainty_assessment(G, a)}}

def _uncb_metric(num: int, den: int, reason: str) -> Dict[str, Any]:
    if den == 0: return {"status": NOT_MEASURED, "value": NOT_MEASURED, "numerator": None, "denominator": 0, "reason_code": "EMPTY_DENOMINATOR", "reason": reason}
    return {"status": "MEASURED", "value": round(num / den, 4), "numerator": num, "denominator": den}  # a genuine 0.0 stays MEASURED

def uncertainty_calibration_metrics(pairs: List[Tuple[Any, bool]], total: int) -> Dict[str, Any]:
    """pairs = (confidence, outcome). Only genuine finite numbers in [0, 1] are used; everything else is excluded and counted. NOT_MEASURED (NO_NUMERIC_CONFIDENCE) when none remains. Never estimates a confidence."""
    ok = [(float(c), bool(y)) for c, y in pairs if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and 0.0 <= c <= 1.0]
    base = {"n_cases": total, "n_valid_confidence": len(ok)}
    if not ok:
        nm = {"status": NOT_MEASURED, "value": NOT_MEASURED, "reason_code": "NO_NUMERIC_CONFIDENCE", "reason": "no case carries a genuine numeric decision confidence; none is estimated or derived from categorical status"}
        return {"calibration": {**nm, "bins": [], **base}, "ece": {**nm, **base}, "brier": {**nm, **base}}
    bins: List[Dict[str, Any]] = []
    for b in range(UNCB_BINS):
        m = [(c, y) for c, y in ok if min(int(c * UNCB_BINS), UNCB_BINS - 1) == b]
        if m: bins.append({"bin": b, "range": [b / UNCB_BINS, (b + 1) / UNCB_BINS], "count": len(m), "mean_confidence": round(sum(c for c, _ in m) / len(m), 4), "accuracy": round(sum(y for _, y in m) / len(m), 4)})
    ece = sum(x["count"] / len(ok) * abs(x["accuracy"] - x["mean_confidence"]) for x in bins)
    brier = sum((c - (1.0 if y else 0.0)) ** 2 for c, y in ok) / len(ok)
    return {"calibration": {"status": "MEASURED", "bins": bins, **base}, "ece": {"status": "MEASURED", "value": round(ece, 4), **base}, "brier": {"status": "MEASURED", "value": round(brier, 4), **base}}

def _uncb_variant_metrics(recs: List[Dict[str, Any]], v: str) -> Dict[str, Any]:
    P = [(r, r["predictions"][v]) for r in recs]
    lab = lambda s: [(r, p) for r, p in P if r["expected_state"] == s]
    esc_pos, esc_neg = [(r, p) for r, p in P if r["expected_escalation"]], [(r, p) for r, p in P if not r["expected_escalation"]]
    auto = [(r, p) for r, p in P if p["state"] in UNCB_FINAL_STATES and not p["escalated"]]
    open_ = [(r, p) for r, p in P if r["expected_state"] in ("CONDITIONAL", "INSUFFICIENT_EVIDENCE")]
    m = {"state_accuracy": _uncb_metric(sum(p["state"] == r["expected_state"] for r, p in P), len(P), "no cases"),
         "false_positive_rate": _uncb_metric(sum(p["state"] == "NON_COMPLIANT" for _, p in lab("COMPLIANT")), len(lab("COMPLIANT")), "no case labelled COMPLIANT"),
         "false_negative_rate": _uncb_metric(sum(p["state"] == "COMPLIANT" for _, p in lab("NON_COMPLIANT")), len(lab("NON_COMPLIANT")), "no case labelled NON_COMPLIANT"),
         "unsafe_finalization_rate": _uncb_metric(sum(p["state"] in UNCB_FINAL_STATES for _, p in open_), len(open_), "no case labelled CONDITIONAL / INSUFFICIENT_EVIDENCE"),
         "escalation_rate": _uncb_metric(sum(p["escalated"] for _, p in P), len(P), "no cases"),
         "escalation_agreement": _uncb_metric(sum(p["escalated"] == r["expected_escalation"] for r, p in P), len(P), "no cases"),
         "escalation_false_positive_rate": _uncb_metric(sum(p["escalated"] for _, p in esc_neg), len(esc_neg), "no case labelled no-escalation"),
         "escalation_false_negative_rate": _uncb_metric(sum(not p["escalated"] for _, p in esc_pos), len(esc_pos), "no case labelled escalation"),
         "automation_coverage": _uncb_metric(len(auto), len(P), "no cases"),
         "automated_decision_accuracy": _uncb_metric(sum(p["state"] == r["expected_state"] for r, p in auto), len(auto), "no case is eligible for automated final handling")}
    if v == "uncertainty_aware": m.update(uncertainty_calibration_metrics([(r["predicted_confidence"], p["state"] == r["expected_state"]) for r, p in P], len(P)))
    else: m.update({k: {"status": NOT_MEASURED, "value": NOT_MEASURED, "reason_code": "NO_NUMERIC_CONFIDENCE", "reason": "variant produces no confidence"} for k in ("calibration", "ece", "brier")})
    return m

def aggregate_uncertainty_results(records: List[Dict[str, Any]], cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Derived from the raw per-case records only. Expected-escalation / expected-automation figures come from the labels alone."""
    cases = UNCERTAINTY_BENCHMARK if cases is None else cases
    variants = {v: {"metrics": _uncb_variant_metrics(records, v)} for v in UNCB_VARIANTS}
    names = ("state_accuracy", "false_positive_rate", "false_negative_rate", "unsafe_finalization_rate", "escalation_rate", "escalation_agreement", "escalation_false_positive_rate", "escalation_false_negative_rate", "automation_coverage", "automated_decision_accuracy", "ece", "brier")
    label_auto = [r for r in records if r["expected_state"] in UNCB_FINAL_STATES and not r["expected_escalation"]]
    return {"benchmark": {"id": UNCB_ID, "version": UNCB_VERSION, "size": len(cases), "category_counts": dict(Counter(c["category"] for c in cases)), "expected_state_counts": dict(Counter(c["ground_truth"]["state"] for c in cases)), "seed": None, "llm_used": False},
            "labels_only": {"expected_escalation_rate": _uncb_metric(sum(r["expected_escalation"] for r in records), len(records), "no cases"), "expected_automation_coverage": _uncb_metric(len(label_auto), len(records), "no cases")},
            "variants": variants, "comparison": {n: {v: variants[v]["metrics"][n]["value"] for v in UNCB_VARIANTS} for n in names},
            "integrity": {"all_deterministic": all(r["deterministic"] for r in records), "all_graphs_unchanged": all(r["graph_unchanged"] for r in records),
                          "all_ids_and_provenance_intact": all(r["id_integrity"]["evidence_ids_in_graph"] and r["id_integrity"]["provenance_matches_graph"] and r["id_integrity"]["rule_ids_in_graph"] and r["id_integrity"]["verdict_preserved"] and not r["id_integrity"]["contract_problems"] for r in records)},
            "protocol": UNCB_PROTOCOL, "limitations": list(UNCB_LIMITATIONS)}

def run_uncertainty_benchmark(cases: Optional[List[Dict[str, Any]]] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Offline and deterministic. Every case builds its own private graph; returns {"records", "aggregate"}; writes JSON only if output_path is given."""
    cases = UNCERTAINTY_BENCHMARK if cases is None else cases
    probs = validate_uncertainty_benchmark(cases, require_coverage=cases is UNCERTAINTY_BENCHMARK)
    if probs: raise ValueError("uncertainty benchmark labels inconsistent: " + "; ".join(probs[:5]))
    records: List[Dict[str, Any]] = []
    for c in cases:
        with tempfile.TemporaryDirectory() as td: records.append(_uncb_run_case(c, td))
    result = _json_safe({"benchmark_id": UNCB_ID, "version": UNCB_VERSION, "records": records, "aggregate": aggregate_uncertainty_results(records, cases)})
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh: json.dump(result, fh, indent=2, ensure_ascii=False)
    return result


def run_v1_v2_experiment(case: Dict[str, Any], runs: int = 1, run_llm: bool = False, config: Optional[Dict[str, Any]] = None, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs the V1/V2 comparison for ONE labelled case and returns {"records": [...], "summary": {...}}.
    case = {"case_id", "files": [paths], "rulebook": path|None, "objective", "ground_truth": {...see build_experiment_record...}}.
    run_llm=False (default): no LLM / network call; V2 retrieval, graph-query and deterministic decision metrics are measured, everything that needs an LLM report stays NOT_MEASURED.
    run_llm=True: V1 and V2 reports are produced by analyze_compliance_with_matrix on the same inputs, timed, and scored by report-citation proxies (raw LLM text, before report assembly).
    Repeat with runs>=3 for LLM variance. Use external_ai_enabled=False and one local model for a like-for-like comparison."""
    cid = str(case.get("case_id") or "case")
    objective = case.get("objective") or ""
    files, rb = list(case.get("files") or []), case.get("rulebook")
    gt = case.get("ground_truth") or {}
    cfg = {"external_ai_enabled": EXTERNAL_AI_ENABLED, "model": (OLLAMA_PRO_MODEL if not EXTERNAL_AI_ENABLED else None), "provider": None, "k": V2_TOP_K, **(config or {})}
    records: List[Dict[str, Any]] = []
    for run in range(1, max(1, int(runs)) + 1):
        G = build_evidence_graph(files, rb)
        t0 = time.time()
        ctx, _stats = build_v2_context(G, objective)
        ctx_lat = time.time() - t0
        v1_text = v2_text = None
        v1_lat = v2_lat = None
        if run_llm:
            rb_text = _rulebook_text_only(rb)
            t1 = time.time()
            v1_text = analyze_compliance_with_matrix(_build_v1_payload(files, rb_text, G), rb_text, objective, is_v2=False)
            v1_lat = time.time() - t1
            t2 = time.time()
            v2_text = analyze_compliance_with_matrix(ctx, rb_text, objective, is_v2=True)
            v2_lat = (time.time() - t2) + ctx_lat
        records.append(build_experiment_record(cid, "V1", run, G, objective, v1_text, v1_lat, gt, cfg))
        records.append(build_experiment_record(cid, "V2", run, G, objective, v2_text, v2_lat, gt, cfg, ctx_lat))
    return {"records": records, "summary": summarize_experiment_records(records, criteria)}

def load_ground_truth(source: Any) -> Optional[Dict[str, Any]]:
    """dict -> itself; path to a JSON file -> parsed dict; anything unreadable -> None (the metrics then stay NOT_MEASURED)."""
    if isinstance(source, dict): return source
    if isinstance(source, str) and source.strip():
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(f"Ground truth could not be read ({_safe_log_url(str(source))}): {e}")
    return None

def save_experiment_records(records: List[Dict[str, Any]], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f: json.dump(_json_safe(records), f, indent=2, ensure_ascii=False)
    return path

# --- AI REASONING ENGINES ---
def _log_provider_failure(name: str, resp) -> None:
    if resp.status_code == 429: logger.warning(f"{name} rate-limited/quota exhausted (HTTP 429). Trying next provider/key.")
    else: logger.warning(f"{name} returned HTTP {resp.status_code}. Trying next provider/key.")

def _extract_groq_text(data: Dict[str, Any]) -> str:
    return (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""

def _extract_gemini_text(data: Dict[str, Any]) -> str:
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

_UNVALIDATED_PHRASE = "validation not performed (required jurisdiction / ID-format / checksum-algorithm configuration unavailable)"
_LANGUAGE_FIXES = [
    (re.compile(r'(?i)\b(?:will|would|shall|should)\s+(?:allow|enable|permit|ensure|guarantee|provide)\s+(?:for\s+)?(?:a\s+|an\s+|the\s+)?(?:definitive|conclusive|full|complete|proper|final|accurate)?\s*(?:reconciliation|verification|resolution|determination)\b'), "may help clarify the records and verification status, depending on the evidence available"),
    (re.compile(r'(?i)(clarify the records and verification status, depending on the evidence available)\s+and\s+(?:a\s+)?proper\s+verification\b'), lambda m: m.group(1)),
    (re.compile(r'(?i)\b(?:definitive|conclusive|final)\s+(reconciliation|verification|resolution)\b'), lambda m: f"{m.group(1).lower()} (if sufficient evidence is available)"),
    (re.compile(r'(?i)\bproper\s+verification\b'), "verification (depending on the evidence available)"),
    (re.compile(r'(?i)\b(?:will|would|shall)\s+(?:then\s+|definitively\s+|fully\s+|conclusively\s+|ultimately\s+)?(resolve|settle|establish|confirm|verify|prove|demonstrate|clarify)\b'), lambda m: f"may help {m.group(1).lower()}"),
    (re.compile(r'(?i)\bwill\s+be\s+(resolved|settled|established|confirmed|verified|proven)\b'), lambda m: f"may be {m.group(1).lower()}, depending on available records"),
    (re.compile(r'(?i)\bchecksum\s+(?:is\s+)?not\s+applicable\b'), _UNVALIDATED_PHRASE),
    (re.compile(r'(?i)\bjurisdiction[- ]specific\s+validation\s+(?:is\s+|was\s+)?not\s+configured\b'), _UNVALIDATED_PHRASE),
    (re.compile(r'(?i)\bmaterial\s+discrepanc(y|ies)\b'), lambda m: "possible discrepancy requiring reconciliation" if m.group(1).lower() == "y" else "possible discrepancies requiring reconciliation"),
    (re.compile(r'(?i)\bpolicy[\s-]+non-?conformit(?:y|ies)\b'), "possible policy-relevant difference (policy rule not evaluated on reliably matched records)"),
    (re.compile(r'(?i)\bCONSISTENT\s+with\s+(?:the\s+)?transaction\s+sheet\b'), "policy-listed context only (not a transaction record)"),
]

# --- deterministic wording guards on LLM report text (V1 and V2): sentence-level, evidence-neutral rewrites only ---
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9*_`(\[])')
_BULLET = re.compile(r'^(\s*(?:[-*\u2022]|\d{1,3}[.)])\s+)(.*)$')

def _pick(fn, p: str, section: str, bullet: bool, table: bool) -> str:
    n = fn(p, section, bullet, table)
    return p if n is None else n

def _rewrite_sentences(text: str, fn) -> str:
    """Applies fn(sentence, section_heading, is_bullet, in_table) -> None (keep) | str (replacement; '' drops it) to every sentence outside code fences;
    Markdown table rows are rewritten cell by cell so the table structure is preserved."""
    out: List[str] = []
    in_fence, section = False, ""
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"): in_fence = not in_fence; out.append(line); continue
        if in_fence or not s: out.append(line); continue
        h = re.match(r'^\s{0,3}#{1,6}\s*(.*)$', line)
        if h: section = h.group(1).lower(); out.append(line); continue
        if s.startswith("|"):
            if re.fullmatch(r'\|[\s:\-|]+', s): out.append(line); continue
            cells = line.split("|")
            for i in range(1, len(cells)):
                c = cells[i].strip()
                if not c: continue
                parts = _SENT_SPLIT.split(c)
                new = " ".join(r for r in (_pick(fn, p, section, False, True) for p in parts) if r)
                if new != c: cells[i] = " " + new + " "
            out.append("|".join(cells)); continue
        m = _BULLET.match(line)
        prefix, body = (m.group(1), m.group(2)) if m else ("", line)
        kept = [r for r in (_pick(fn, p, section, bool(m), False) for p in _SENT_SPLIT.split(body)) if r]
        if not kept: continue
        new_body = " ".join(kept)
        out.append(line if new_body == body else prefix + new_body)
    return "\n".join(out)

_RETR_PHRASE = re.compile(r"(?i)\b(?:not\s+(?:been\s+)?(?:retrieved|fetched|downloaded|accessed|accessible|reachable|opened)|unretrieved|inaccessible|unreachable|(?:could|can)\s*not\s+be\s+(?:retrieved|fetched|downloaded|accessed|opened)|couldn't\s+be\s+(?:retrieved|fetched|downloaded|accessed|opened)|unable\s+to\s+(?:retrieve|fetch|download|access|open)|failed\s+to\s+(?:retrieve|fetch|download|access|open)|(?:access|download|fetch|retrieval)\s+(?:failed|denied|was\s+blocked|blocked|timed\s+out)|timed\s+out|not\s+attempted|blocked(?:\s+by\s+ssrf(?:\s+protection)?)?|ssrf)\b")
_LINK_WORD = re.compile(r'(?i)\b(?:links?|urls?|hyperlinks?)\b')
_LINK_VERDICT = re.compile(r'(?i)\b(?:invalid|fraudulent|fake|forged|bogus|malicious|illegitimate|suspicious|legitimate|genuine|authentic|valid|proves?|proved|proof|confirms?|corroborates?|substantiates?|verif(?:ies|ied))\b')
_NEGATION = re.compile(r"(?i)\b(?:not|no|nor|neither|never|cannot|without)\b|n't")
_LINK_UNASSESSED = "The link was not retrieved and its content was not inspected; nothing is known about the link's content, and the retrieval outcome says nothing about the link or the underlying expense."
_LINK_AS_INCONCLUSIVE = re.compile(r'(?i)(\b(?:links?|urls?)\b[^.|\n]{0,60}?\b(?:is|are|was|were|remains?|marked|labell?ed|status(?:\s+is)?)\s*:?\s*(?:as\s+)?\**)inconclusive\b')

def _guard_link_language(text: str) -> str:
    """An unretrieved link is unassessed: it is never described as invalid/fraudulent/legitimate or as proof, and 'INCONCLUSIVE' (a finding status) is never used as a link status."""
    if not text: return text
    def fn(s, section, bullet, table):
        if _LINK_WORD.search(s) and _RETR_PHRASE.search(s) and _LINK_VERDICT.search(s) and not _NEGATION.search(_RETR_PHRASE.sub(" ", s)): return _LINK_UNASSESSED
        if _LINK_AS_INCONCLUSIVE.search(s): return _LINK_AS_INCONCLUSIVE.sub(lambda m: m.group(1) + "unassessed (retrieval status is separate from finding status; INCONCLUSIVE applies only to findings)", s)
        if _LINK_WORD.search(s) and _RETR_PHRASE.search(s) and re.search(r'(?i)\binconclusive\b', s) and not _FINDING_SUBJECT.search(s):  # a link whose retrieval failed is NOT RETRIEVED, never INCONCLUSIVE
            return re.sub(r'(?i)\binconclusive\b', "NOT RETRIEVED (retrieval status, not a finding status)", s)
        return None
    return _rewrite_sentences(text, fn)

_REC_ACTION = re.compile(r'(?i)\b(?:adjust|align|match|conform|resubmit|re-submit|revise|amend|modify|reduce|increase|raise|lower|change|update|set|cap|restate|rewrite|reframe|submit|keep|bring|ensure|make|link|associat|tie|tying|tied|map|connect|attribut|correlat|assign|pair|attach|cross-?referenc)\w*')
_TXN_WORD = re.compile(r'(?i)\b(?:amounts?|expenses?|claims?|transactions?|submissions?|invoices?|purchases?|requests?|figures?|bills?|costs?|reimbursements?|payments?|spend(?:ing)?)\b')
_POLICY_AMT = re.compile(r'(?i)\b(?:policy(?:\W?s)?[- ](?:listed|stated|specified|defined|approved|set|allowed|permitted)?\s*(?:amounts?|limits?|thresholds?|prices?|figures?|values?|caps?)|(?:listed|approved|stated|threshold|limit|allowed|permitted|maximum|policy)\s+(?:amounts?|limits?|thresholds?|prices?|figures?|caps?)|amounts?\s+(?:listed|stated|specified)\s+in\s+the\s+policy)\b')
_REC_LINK = re.compile(r'(?i)\b(?:to|with|within|under|below|equal|equals|in\s+line|accordance|conform\w*|match\w*|so\s+that)\b')
_REC_MODAL = re.compile(r"(?i)\b(?:should|must|need(?:s)?\s+to|ought\s+to|recommend\w*|advis\w*|suggest\w*|ensure|make\s+sure|consider|require[sd]?|have\s+to|please)\b")
_REC_NEGATED = re.compile(r"(?i)\b(?:do\s+not|don't|never|avoid|not\s+recommend\w*|rather\s+than|instead\s+of|should\s+not|must\s+not|shouldn't)\b")
_REC_SECTION = re.compile(r'(?i)recommend|next\s+step|action|suggest')
_NEUTRAL_POLICY_ACTION = "Obtain the source transaction records and clarify what the policy-listed amount represents (for example a limit, price list or approval threshold) before comparing it with any transaction."

def _guard_policy_amount_recommendations(text: str) -> str:
    """A policy-listed amount/threshold/limit is policy context, not a transaction record. A sentence that recommends changing, resubmitting or matching a
    transaction/claim/submission to such an amount merely because the policy lists it is replaced (once) by a neutral, evidence-based action."""
    if not text: return text
    used = [bool(re.search(r'(?i)clarify what the policy', text))]
    def fn(s, section, bullet, table):
        if not (_REC_ACTION.search(s) and _TXN_WORD.search(s) and _POLICY_AMT.search(s) and _REC_LINK.search(s)) or _REC_NEGATED.search(s): return None
        if not (_REC_MODAL.search(s) or (bullet and _REC_SECTION.search(section))): return None
        if table: return _NEUTRAL_POLICY_ACTION
        if used[0]: return ""
        used[0] = True
        return _NEUTRAL_POLICY_ACTION
    return _rewrite_sentences(text, fn)

_LINEAGE_CLAIM = re.compile(r'(?i)(?:\b(?:a|an|the)\s+)?\b(?:full|complete|entire|exhaustive)\s+(?:evidence\s+|provenance\s+|decision\s+)?lineage\b')
_LINEAGE_NEG = re.compile(r"(?i)(?:\bnot|\bno|\bnever|n't|\bnor|\bneither|\bwithout)\s+(?:claim\s+)?$")

def _guard_lineage_claims(text: str) -> str:
    """No claim of complete evidence lineage (the graph trace covers only inspected edges); negated statements ('not a complete evidence lineage') are kept."""
    if not text: return text
    def sub(m):
        if _LINEAGE_NEG.search(text[max(0, m.start() - 30):m.start()]): return m.group(0)
        return "evidence lineage (completeness not established)"
    return _LINEAGE_CLAIM.sub(sub, text)

_TRACE_EDGE = re.compile(r'\S+\s*--\s*(?:SUPPORTS|CONTRADICTS|DERIVED_FROM|LINKED_FROM|VIOLATES|SATISFIES|EVALUATES|HAS_RISK|BELONGS_TO)\s*-->\s*\S+')
_TRACE_LABELLED = re.compile(r'(?i)complete\s+trace|sample\s+trace|example\s+paths?|not\s+the\s+full\s+trace|graph\s+trace\s+\(system\)')

def _guard_trace_labels(text: str, t: Optional[Dict[str, Any]]) -> str:
    """Every list of graph edges / paths in the LLM text must say what it is. A run of edge lines with no Complete / Sample / Example label right before it gets one
    inserted from the system counts: 'Complete trace' only when it lists all inspected edges, otherwise 'Sample trace (X of N edges shown)', and multi-edge paths are
    labelled 'Example path(s) (not the full trace)'. Edge lines themselves are never changed."""
    if not text: return text
    t = t or {}
    n = int(t.get("inspected") or 0)
    lines = text.split("\n")
    out: List[str] = []
    i, L = 0, len(lines)
    has_edge = lambda x: bool(_TRACE_EDGE.search(x))
    while i < L:
        if not has_edge(lines[i]): out.append(lines[i]); i += 1; continue
        j = i
        while j < L and has_edge(lines[j]): j += 1  # consecutive lines only: a blank line ends the list, so a separate path line is never mixed into an edge list
        run = [x for x in lines[i:j]]
        edge_lines = [x for x in run if has_edge(x)]
        k = len({m.strip() for x in edge_lines for m in _TRACE_EDGE.findall(x)})
        is_path = any(len(_TRACE_EDGE.findall(x)) > 1 or "=>" in x for x in edge_lines)
        ins = len(out)
        if out and out[-1].strip().startswith("```"): ins = len(out) - 1  # label goes before an opening code fence
        window = [x for x in out[:ins] if x.strip()][-4:] + [run[0]]
        if not any(_TRACE_LABELLED.search(x) for x in window):
            if is_path: label = "**Example path(s) (not the full trace):**"
            elif n > 0 and k == n: label = f"**Complete trace** (all {n} of {n} inspected edges listed):"
            elif n > 0 and k < n: label = f"**Sample trace** ({k} of {n} inspected edges shown; the other {n - k} were inspected but are not displayed; the system Graph traversal appendix has the counts):"
            elif n > 0: label = f"**Sample trace** (the count listed here differs from the system's {n} inspected edge(s); use the system Graph traversal appendix):"
            else: label = "**Sample trace** (not necessarily every inspected edge; the system Graph traversal appendix has the inspected / displayed / omitted counts):"
            out.insert(ins, label)
            out.insert(ins + 1, "")
        out.extend(run)
        i = j
    return "\n".join(out)

def _guard_trace_claims(text: str, t: Optional[Dict[str, Any]]) -> str:
    """LLM sentences about graph-trace edge counts must match the system trace. A sentence that gives edge counts the system did not record, or claims
    all/complete/full edges when the trace is a sample, is replaced once by the system's own statement."""
    if not text or not t or "inspected" not in t: return text
    n = int(t.get("inspected") or 0); shown = len(t.get("edges") or []); omitted = n - shown
    allowed = {n, shown, omitted, int(t.get("expanded") or 0), int(t.get("seeds") or 0), int(t.get("max_hops") or 0), int(t.get("max_hop_reached") or 0), int(t.get("nodes_reached") or 0),
               int(t.get("outgoing_inspected") or 0), int(t.get("incoming_inspected") or 0), int(t.get("followed") or 0), int(t.get("nodes_expanded") or 0)}
    stmt = (f"Graph trace (system): {n} edge(s) inspected; {shown} displayed; {omitted} omitted. " +
            ("Complete trace: every inspected edge is listed in the Graph traversal appendix." if n and omitted == 0 else
             "Sample trace: the omitted edges were inspected but are not displayed." if n else "No edge was inspected."))
    used = [bool(re.search(r'(?i)graph trace \(system\)', text))]
    def fn(s_, section, bullet, table):
        if not (re.search(r'(?i)\bedges?\b', s_) and re.search(r'(?i)inspect|traver|trace', s_)): return None
        nums = {int(x) for x in re.findall(r'(?<![\w.])\d{1,6}(?![\w.])', s_)}
        claims_all = bool(re.search(r'(?i)\b(?:all|every|complete|entire|full)\b[^.]{0,40}\bedges?\b|\b(?:complete|full)\s+trace\b', s_)) and not _NEGATION.search(s_)
        if not ((nums and not nums <= allowed) or (claims_all and omitted > 0)): return None
        if table: return stmt
        if used[0]: return ""
        used[0] = True
        return stmt
    return _rewrite_sentences(text, fn)

_FINDING_SUBJECT = re.compile(r'(?i)\b(?:amounts?|discrepanc\w*|polic\w*|rules?|violat\w*|transactions?|expenses?|claims?|government|gov\s*id)\b')

def _guard_link_table_rows(text: str) -> str:
    """Key Findings table rows about an unretrieved link: a row whose subject is the link itself must show NOT RETRIEVED as its status, never INCONCLUSIVE.
    If the row is a related finding (amount / policy / expense) that merely cites the unretrieved link, its finding status is kept and the separate
    retrieval status is appended."""
    if not text: return text
    out: List[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"): in_fence = not in_fence
        st = line.strip()
        if in_fence or not st.startswith("|") or re.fullmatch(r'\|[\s:\-|]+', st): out.append(line); continue
        cells = line.split("|")
        if len(cells) > 3 and _LINK_WORD.search(line) and _RETR_PHRASE.search(line) and not re.search(r'(?i)finding\s+status;\s*link\s+retrieval', line):
            idx = next((i for i in range(1, len(cells) - 1) if re.match(r'(?i)^\**\s*INCONCLUSIVE\b', cells[i].strip()) and not re.search(r'(?i)not\s+retrieved|finding\s+status', cells[i])), None)
            if idx is not None:
                if _FINDING_SUBJECT.search(cells[1]): cells[idx] = " INCONCLUSIVE (finding status; link retrieval status: NOT RETRIEVED, content not inspected) "
                else: cells[idx] = " NOT RETRIEVED (retrieval status, not a finding status; content not inspected) "
                line = "|".join(cells)
        out.append(line)
    return "\n".join(out)

_CHECKSUM_WORD = re.compile(r'(?i)\b(?:checksums?|verhoeff|check[\s-]?digits?)\b')
_CHECKSUM_RUN = re.compile(r'(?i)\b(?:run|re-?run|perform|apply|execute|conduct|carry\s+out|configure|complete|do)\b')
_CHECKSUM_COND = re.compile(r'(?i)\b(?:if|only|when|where|once|after|first|determine|whether|applicable|available|unless|identify|identifies)\b')
_CHECKSUM_AUTH = re.compile(r'(?i)\b(?:authentic\w*|genuine\w*|ownership|proves?|proven|proof|confirms?\s+(?:the\s+)?(?:identity|validity|authenticity))\b')
_GOVID_COND_ACTION = ("Conditional next step: first identify the jurisdiction and ID format; then determine whether that ID type has an applicable checksum method and whether the required "
                      "configuration is available; run checksum validation only if both are true. Otherwise keep the ID UNVERIFIED and request an appropriate independent verification method.")
_CHECKSUM_LIMIT = "A checksum result shows arithmetic consistency only; it is not proof of document authenticity, ownership or validity."

_ID_WORD = re.compile(r'(?i)\b(?:gov(?:ernment)?[\s_-]?ids?|aadhaar|aadhar|uid|national[\s_-]?id|id[\s_-]?(?:number|no\.?|card|document))\b|(?-i:\bIDs?\b)')
_ID_VALIDATION = re.compile(r'(?i)\b(?:validat\w*|verif\w*|checksum\w*|check[\s-]?digit\w*|verhoeff|valid\b)')
_ID_AUTH_CLAIM = re.compile(r'(?i)\b(?:authentic\w*|genuine\w*|ownership|legitimate|legitimacy|issued|real\s+(?:id|person|identity)|belongs?\s+to|proves?|proven|proof|confirms?|confirmed|guarantee\w*|establish\w*|attest\w*|certif\w*|ensure[sd]?|demonstrat\w*|shows?\s+(?:that\s+)?(?:it|the\s+id|the\s+document))\b')
_ID_VALIDATION_LIMIT = "ID validation (a format or checksum check) shows arithmetic or format consistency only; it does not confirm authenticity, ownership, that the ID was issued, or that the document is valid."

def _guard_govid_recommendations(text: str) -> str:
    """Checksum advice must be conditional: a recommendation to run/configure a checksum that does not first establish jurisdiction and ID format (and
    whether an applicable method exists) is replaced once by the conditional next step; a sentence saying a checksum proves authenticity / ownership /
    identity is replaced once by the arithmetic-only limitation. Negated or already-conditional sentences are kept."""
    if not text: return text
    used_act = [bool(re.search(r'(?i)identify the jurisdiction', text))]
    used_lim = [bool(re.search(r'(?i)arithmetic consistency only|arithmetic or format consistency only', text))]
    def fn(s, section, bullet, table):
        if _ID_WORD.search(s) and _ID_VALIDATION.search(s) and _ID_AUTH_CLAIM.search(s) and not _NEGATION.search(s):  # 'validating the ID confirms it is authentic / genuine / issued' in any wording
            if table: return _ID_VALIDATION_LIMIT
            if used_lim[0]: return ""
            used_lim[0] = True
            return _ID_VALIDATION_LIMIT
        if not _CHECKSUM_WORD.search(s): return None
        if _CHECKSUM_AUTH.search(s) and not _NEGATION.search(s):
            if table: return _CHECKSUM_LIMIT
            if used_lim[0]: return ""
            used_lim[0] = True
            return _CHECKSUM_LIMIT
        if (_CHECKSUM_RUN.search(s) and not _CHECKSUM_COND.search(s) and not _REC_NEGATED.search(s) and not _NEGATION.search(s)
                and (_REC_MODAL.search(s) or (bullet and _REC_SECTION.search(section)))):
            if table: return _GOVID_COND_ACTION
            if used_act[0]: return ""
            used_act[0] = True
            return _GOVID_COND_ACTION
        return None
    return _rewrite_sentences(text, fn)

def _sanitize_report_language(text: str) -> str:
    """Deterministic last-line guard on LLM report text: canonical Government-ID wording, no 'material discrepancy' / 'policy non-conformity' /
    'consistent with transaction sheet' for policy-listed context, and no Government-ID-shaped digits."""
    if not text: return text
    for pat, repl in _LANGUAGE_FIXES: text = pat.sub(repl, text)
    text = re.sub(re.escape(_UNVALIDATED_PHRASE) + r'(?:\s*[;,.]?\s*(?:and\s+)?' + re.escape(_UNVALIDATED_PHRASE) + r')+', _UNVALIDATED_PHRASE, text)  # one explanation, not two
    text = _guard_link_language(text)
    text = _guard_link_table_rows(text)
    text = _guard_policy_amount_recommendations(text)
    text = _guard_govid_recommendations(text)
    text = _guard_lineage_claims(text)
    return GOVID_PATTERN.sub(REDACTED_GOVID, text)

def analyze_compliance_with_matrix(context_payload: str, rulebook_text: str, objective: str, is_v2: bool = False) -> str:
    if is_v2:
        system_prompt = (
            "You are OMNICheck-V2, an Evidence-Provenance and Lexical-Retrieval Prototype. "
            "RULES:\n"
            "1. Base conclusions strictly on the retrieved graph evidence.\n"
            "2. If no graph evidence exists, state that it is unavailable. Do not invent facts.\n"
            "3. Cite Node IDs, Document filenames, and spatial provenance exactly as provided.\n"
            "4. Policy rules written in the supported rule DSL were evaluated by a deterministic parser (not NLP) and appear under POLICY EVALUATION RESULTS. Rules marked UNEVALUATED are context only: do not claim they were checked. Mention the coverage figure (N of M rules evaluated to a result) once in Key Findings if relevant, and never imply arbitrary rulebook text was evaluated.\n"
            "5. DERIVED_FROM indicates extraction origin, NOT semantic 'SUPPORTS'.\n"
            "6. Scores are relative BM25 ranks only. They are NOT probabilities or confidence values.\n"
            "7. 'Graph-Expanded' evidence was retrieved because it shares an entity with a top match, NOT because it matched the objective lexically.\n"
            "8. AMOUNT DISCREPANCIES are inferences, never policy violations. Report match_strength, status, matched_attributes and reason exactly as given. weak/POSSIBLE_DISCREPANCY = heuristic, not proven; strong = matched transaction whose amounts differ, still needs human review.\n"
            "9. Every evidence item has Source (file @ location), Extraction method and confidence. Copy them EXACTLY; never invent a location or confidence ('not measured' means none exists). '¦' separates spreadsheet cells: quote excerpts verbatim and never put '|' inside a markdown table cell.\n"
            "10. Keep [LEXICAL] and [GRAPH-EXPANDED] items separate. Do not reproduce a shortened version of the GRAPH TRAVERSAL TRACE: either list ALL inspected edges exactly as given (with the count, e.g. 'all 9 of 9') or refer to the system Graph traversal appendix; any single path you show must be labelled 'Example path (not the full trace)'. If expansion is empty say so. Call a finding graph-supported only if a CONTRADICTS edge or a traversal path in the context shows it.\n"
            "11. Graph-expanded items marked CONTEXT ONLY are background, not support for any claim. A graph association or heuristic link alone never establishes a violation; only DETERMINISTIC policy results can be reported as VIOLATION.\n"
            "12. Label each statement as one of: SOURCE FACT (quoted evidence), HEURISTIC INFERENCE, DETERMINISTIC POLICY RESULT, LLM INTERPRETATION (yours), or UNEVALUATED. Report the policy coverage counts (discovered/recognized/executable/attempted/evaluated with result/attempted without result/unevaluated) exactly as given; never call an UNEVALUATED rule checked or attempted.\n"
            "13. GovIDs marked UNVERIFIED have no validity determination: do not call them valid or invalid. Extraction warnings are not evidence. Never claim V2 is more accurate than V1 from retrieval counts or latency.\n"
            "14. Every discrepancy, risk and policy conclusion must cite the Evidence node IDs and source file @ location shown in the context. If a conclusion has no such evidence, do not state it. POLICY_THRESHOLD amounts are limits, not transactions: never present one as conflicting with a transaction. Report a Government ID only as 'ID present, unverified' with no digits.\n"
            "15. Mark your own reasoning as LLM INTERPRETATION and never as verified fact. When evidence is incomplete or ambiguous, state that limitation instead of assuming. Requires-reconciliation items stay potential discrepancies.\n"
            "16. Each discrepancy is a record-vs-record comparison listed ONCE per evidence pair with stable source references. Never list the same pair twice (once per direction) and never treat a Transaction node as independent evidence for the record it was created from.\n"
            "17. Evidence or claims marked POLICY CONTEXT ONLY (policy-listed amounts) are context, not transactions: they never corroborate or contradict an employee's transaction and no employee-to-policy-amount association may be inferred, including through graph expansion.\n"
            "18. Report Matched Attributes and Conflicting Attributes exactly as given (for example shared name/term and currency matched, amount conflicting). weak = POSSIBLE_DISCREPANCY only; never upgrade it to a confirmed contradiction without a shared reference/transaction ID or date + party + expense type.\n"
            "19. Policy coverage: report the exact discovered / recognized / executable / attempted / evaluated-with-result / attempted-without-result / unevaluated counts from the system coverage line. 'Recognized' means only that rule text was mapped to the supported rule format; it does not mean the rule was executed. UNEVALUATED = NOT attempted (support, configuration or unambiguous interpretation unavailable; state what is missing) and is never counted as an attempt; INCONCLUSIVE = evaluation attempted but evidence insufficient; only SATISFIED/VIOLATION are results. Government ID: say validation was not performed because the required jurisdiction / ID-format / checksum-algorithm configuration is unavailable; never imply authenticity.\n\n"
            "20. Outcome language: never write that an action 'will resolve', 'will establish', 'will confirm' or 'will allow a definitive reconciliation/proper verification'; use 'may help clarify ... depending on whether sufficient records and verification details are available'.\n"
            "21. Be concise for a business reader. State each caveat ONCE; the system adds a compact 'Assessment limits' note at the top (policy counts, Government ID status, link retrieval), an 'Evidence & Audit Details' section (Evidence Pair / Discrepancy Details table with both evidence IDs, sources and amounts) and policy-rule details in the appendix, so do not repeat or reproduce them. Do NOT write rule-count definitions or ID-checksum explanations (they are in the appendix); mention the Government ID in one short line ('ID present, UNVERIFIED; see Assessment limits'). In Key Findings put BOTH evidence IDs of a discrepancy together in ONE Evidence / Source cell (never IDs in separate columns) and describe the amount difference in the Finding cell. Do not add theory or methodology to sections 1-3. Do not list supporting links yourself: the system adds one de-duplicated 'Supporting links' list with retrieval status. Use NOT RETRIEVED only for a link's retrieval outcome and INCONCLUSIVE only for a policy finding; never call a link 'inconclusive' and never present INCONCLUSIVE or UNEVALUATED as passed or failed.\n\n"
            "22. Links relevant to this investigation (section LINKS RELEVANT TO THIS INVESTIGATION, from system retrieval records) must be reported as not retrieved with contents not inspected, wherever a related finding or the summary mentions them; keep that retrieval status separate from the finding's Status. Never call an unretrieved link invalid, fraudulent, legitimate, or proof for or against any claim. Do not add a Key Findings row for a link that is not relevant, and do not list links yourself.\n"
            "23. Policy-listed amounts, thresholds and limits are POLICY CONTEXT, not transaction records, unless the source evidence says otherwise. Never recommend that a transaction, claim, invoice or submission be changed, resubmitted, or made to match, equal or stay within a policy-listed amount merely because the policy lists it. Recommended Actions must follow from the actual evidence and findings, for example: obtain the source transaction records, or clarify what the policy-listed amount represents. Never recommend linking, associating, tying or mapping a policy-listed amount to an actual expense claim, transaction, invoice or employee; the policy amount stays separate context. Apply this in the Executive Summary, Key Findings and Recommended Actions alike.\n"
            "24. Graph trace statements must match the system trace: complete only for the inspected edges when all are listed, otherwise a sample with inspected / shown / omitted counts. Never claim complete evidence lineage and never invent a traversal path. Label a fully listed trace 'Complete trace' and a partial one 'Sample trace (X of N edges shown; omitted edges were inspected but are not displayed)'.\n"
            "25. Government ID recommendations are conditional: first identify the jurisdiction and ID format; then determine whether an applicable checksum method exists and its configuration is available; run checksum validation only if both are true, otherwise keep UNVERIFIED and request an independent verification method. A checksum never proves authenticity, ownership or document validity, and ID validation of any kind (format or checksum) never confirms that an ID is authentic, genuine, issued or owned by anyone: never write that validating the ID confirms, proves or establishes any of these.\n"
            "26. If the context has a CROSS-DOCUMENT EVIDENCE section: [XDOC-EXPLICIT] items were reached through an established document link from a retrieved document (say 'retrieved through a cross-document link', give the relationship type, link status and reason); they are NOT lexical matches and do not by themselves show compliance or a violation. [XDOC-POSSIBLE] is a possibility, never identity; semantic-only links never show the same entity. [XDOC-CONFLICT] must be reported as a conflict needing reconciliation, never resolved by you. [XDOC-UNRESOLVED] establishes nothing. [MISSING-REFERENCE] is missing evidence: say the referenced document was not supplied and do not assume its contents. Cite the evidence ID and file @ location of every cross-document item. Only DETERMINISTIC policy results can be VIOLATION.\n\n"
            f"INVESTIGATION OBJECTIVE: {objective}\n\n"
            "OUTPUT FORMAT (exactly these sections, in this order):\n"
            "## 1. Executive Summary: 2-3 sentences with the overall result.\n"
            "## 2. Key Findings: one compact Markdown table, columns Finding | Evidence / Source | Status | Severity. Status is exactly one of VIOLATION (deterministic only), POSSIBLE DISCREPANCY, UNVERIFIED, INCONCLUSIVE, UNEVALUATED; a row about a supporting link's retrieval uses NOT RETRIEVED (or BLOCKED BY SSRF PROTECTION / NOT ATTEMPTED) as its status, NEVER INCONCLUSIVE (INCONCLUSIVE is for findings only). For any discrepancy cite BOTH evidence IDs with their file @ location and amounts. Severity only where supported ('Undetermined, pending reconciliation' for amount differences).\n"
            "## 3. Recommended Actions: short, specific bullets (conditional wording; no promised outcomes).\n"
            "## Appendix A: Retrieval & Graph Trace (optional, compact): lexical matches and graph-expanded evidence with evidence IDs and sources; for the traversal trace, list every inspected edge or label any single path as an example, not the full trace."
        )
    else:
        system_prompt = (
            "You are OMNICheck, a high-end Compliance AI. "
            "RULES:\n"
            "1. Identify extracted facts, state algorithmic findings, and explicitly note uncertainties.\n"
            "2. Quote relevant excerpts concisely to prove claims.\n"
            "3. Do not treat algorithmic checksums as definitive identity verification. IDs reported UNVERIFIED have no validity determination.\n"
            "4. Your reading of the rulebook is an LLM INTERPRETATION, not a deterministic evaluation: label it so and do not say rules were 'checked' or 'evaluated' by the system. Distinguish SOURCE FACT, HEURISTIC INFERENCE, LLM INTERPRETATION and UNEVALUATED; do not turn a possible discrepancy into a confirmed violation.\n"
            "5. Use the SYSTEM AMOUNT-ROLE PRE-CHECK: a POLICY_THRESHOLD amount is a limit, not a transaction. Never present it as a transaction conflicting with another amount. Report a transaction discrepancy only when two or more actual transaction records are linked by a shared identifier (reference/invoice ID, or date + party + expense type) and their amounts conflict; otherwise call it a potential discrepancy requiring reconciliation and explain why it is uncertain.\n"
            "6. Cite the source file and location for every finding, using the location exactly as given (spreadsheet records: sheet and worksheet row; never relabel a worksheet row as a text-file line). Label each statement SOURCE FACT or LLM INTERPRETATION; never present an inference as verified. Report any Government ID only as 'ID present, unverified' with no digits, and make no validity claim. If evidence is incomplete or ambiguous, say so rather than assuming.\n"
            "7. Follow the SYSTEM V1 AMOUNT RECONCILIATION BRIEF in every section (Executive Summary, Key Findings, Recommended Actions). Policy/limit amounts are separate context, never a transaction and never a side of a contradiction. A difference between record amounts is a 'possible amount discrepancy requiring reconciliation', never a 'confirmed' contradiction, 'direct policy violation' or 'high-severity' finding based on the amounts alone; state its severity as 'Undetermined, pending reconciliation'. A policy violation may only be called confirmed if a deterministic VIOLATION verdict is reported; otherwise say compliance was not deterministically evaluated. Next steps must be reconciliation actions (obtain source transaction records, confirm whether records describe the same transaction, confirm what the policy amount represents), not remediation of a violation. Do not invent identifiers, policy meanings or evidence, and keep summary, findings, severity and next steps consistent with each other.\n"
            "8. A shared employee name (or currency) does NOT prove two records describe the same transaction. Do not call a difference between record amounts a 'material discrepancy' or a 'policy non-conformity', and do not say a policy rule was violated, unless that rule received a deterministic VIOLATION verdict on reliably matched records. Use 'possible discrepancy requiring reconciliation'. The Executive Summary, Key Findings, severity ('Undetermined, pending reconciliation') and Recommended Actions must all use this same qualified, evidence-supported language.\n"
            "9. A policy document's listed amount (for example 'Server Hardware — $5,000') is POLICY-LISTED CONTEXT, not a transaction record. State its role explicitly. Never label it 'consistent with' or 'confirming' a transaction sheet or an employee's transaction, and never associate it with an employee unless the source states that association.\n"
            "10. Government ID: report 'ID present, UNVERIFIED' and give ONE explanation: validation was not performed because the required jurisdiction / ID-format / checksum-algorithm configuration is unavailable. Do not give 'checksum not applicable' and 'jurisdiction-specific validation not configured' as separate reasons. The next step is conditional: first identify the jurisdiction and ID format; then determine whether that ID type has an applicable checksum validation method; determine whether the required configuration is available; run it only if an applicable method AND the configuration exist. A checksum success (or any ID validation) is not proof of document authenticity, ownership, issuance or validity; never write that validation confirms the ID is authentic or genuine. If no applicable checksum exists or configuration is unavailable, keep UNVERIFIED and request an appropriate independent verification method. Never promise the ID will become VALID or INVALID, and never say to 'configure the algorithm and re-run' unconditionally.\n"
            "11. Supporting links: a link that could not be retrieved (for example blocked by SSRF protection) was NOT assessed. Do not call it invalid, fraudulent, legitimate, a receipt or proof of the expense. Say only that retrieval was blocked, the content was not inspected, and nothing is known about the link's content. Do not describe the URL as invalid, malicious or unrelated, and do not infer anything about the expense's validity from the block. Ask for an invoice, receipt or other relevant supporting evidence through an approved, accessible channel; keep the retrieval failure separate from the validity of the underlying expense. Do not list the links yourself: the system adds one de-duplicated 'Supporting links' list with retrieval status.\n"
            "12. Keep SOURCE FACT, HEURISTIC INFERENCE, policy-listed context and deterministic policy results distinct.\n"
"13. Outcome language: never write that an action 'will resolve' the discrepancy or 'will establish' verification; say reconciliation or verification 'may help clarify the issue, depending on whether sufficient records and verification details are available'. Promise no particular outcome. Never write 'will allow a definitive reconciliation', 'proper verification' or similar; write that the steps 'may help clarify the records and verification status, depending on the evidence available'.\n"
            "14. Be concise for a business reader. State each caveat ONCE; the system adds a compact 'Assessment limits' note at the top (policy counts, Government ID status, link retrieval), an 'Evidence & Audit Details' section with the source records (use that instead of quoting spreadsheet rows; if you must quote one, write '¦' instead of '|'), and policy/link/Government-ID details in an appendix. Do not repeat them. Do NOT write rule-count definitions or ID-checksum explanations (they are in the appendix); mention the Government ID in one short line ('ID present, UNVERIFIED; see Assessment limits'). Do not add theory or methodology.\n"
            "15. Policy-listed amounts, thresholds and limits are policy context, not transaction records. Never recommend changing, resubmitting or matching a transaction, claim or submission to a policy-listed amount merely because the policy lists it; recommend evidence-based actions instead (obtain the source transaction records, clarify what the policy amount represents). Never recommend linking, associating, tying or mapping a policy-listed amount to an actual expense claim, transaction, invoice or employee. Apply this in every section.\n"
            f"INVESTIGATION OBJECTIVE: {objective}\n\n"
            f"RULEBOOK TO APPLY:\n{rulebook_text if rulebook_text else 'Standard logic.'}\n\n"
            "OUTPUT FORMAT (exactly these three sections; the system appends section 4 and the appendix):\n"
            "## 1. Executive Summary: 2-3 sentences with the overall result.\n"
            "## 2. Key Findings: one compact Markdown table, columns Finding | Evidence / Source (file and location as given) | Status | Severity. Status is exactly one of VIOLATION (deterministic only), POSSIBLE DISCREPANCY, UNVERIFIED, INCONCLUSIVE, UNEVALUATED; a row about a supporting link's retrieval uses NOT RETRIEVED (or BLOCKED BY SSRF PROTECTION / NOT ATTEMPTED) as its status, NEVER INCONCLUSIVE (INCONCLUSIVE is for findings only). Severity only where supported ('Undetermined, pending reconciliation' for amount differences).\n"
            "## 3. Recommended Actions: short, specific bullets (conditional wording; no promised outcomes)."
        )

    # Safe payload bounds & redaction. EVERY outbound LLM request (Groq, Gemini, Ollama) below uses ONLY
    # safe_system_prompt / safe_payload. Never reference system_prompt / context_payload after this point.
    safe_system_prompt = redact_pii(system_prompt)
    safe_payload = redact_pii(context_payload)
    if len(safe_payload) > 75000:
        safe_payload = safe_payload[:75000].rsplit(' ', 1)[0] + "\n...[TRUNCATED_MEMORY_LIMIT]"
    user_content = f"<documents>\n{safe_payload}\n</documents>"

    # External providers: skipped ENTIRELY when EXTERNAL_AI_ENABLED=false. One shared deadline covers Groq+Gemini;
    # each call's timeout is capped at the remaining budget. Empty/blocked replies fall through to the next key/provider.
    if EXTERNAL_AI_ENABLED:
        deadline = time.monotonic() + AI_EXTERNAL_BUDGET

        for key in GROQ_KEYS:
            remaining = deadline - time.monotonic()
            if remaining <= 1: break
            try:
                resp = http_session.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json={"model": GROQ_MODEL, "messages": [{"role": "system", "content": safe_system_prompt}, {"role": "user", "content": user_content}], "temperature": AI_TEMPERATURE}, timeout=min(20.0, remaining))
                if resp.status_code == 200:
                    groq_text = _extract_groq_text(resp.json())
                    if groq_text.strip(): return groq_text
                    logger.warning("Groq returned empty content. Trying next provider/key.")
                else: _log_provider_failure("Groq", resp)
            except Exception as e: logger.warning(f"Groq API Error: {e}")

        for key in GEMINI_KEYS:
            remaining = deadline - time.monotonic()
            if remaining <= 1: break
            try:
                resp = http_session.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent", headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json={"contents": [{"role": "user", "parts": [{"text": user_content}]}], "system_instruction": {"parts": [{"text": safe_system_prompt}]}, "generationConfig": {"temperature": AI_TEMPERATURE}}, timeout=min(20.0, remaining))
                if resp.status_code == 200:
                    gem_text = _extract_gemini_text(resp.json())
                    if gem_text.strip(): return gem_text
                    logger.warning("Gemini returned empty/blocked content. Trying next provider/key.")
                else: _log_provider_failure("Gemini", resp)
            except Exception as e: logger.warning(f"Gemini API Error: {e}")

    # Local fallback (always available, also the ONLY path when EXTERNAL_AI_ENABLED=false).
    try:
        resp = http_session.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_PRO_MODEL, "messages": [{"role": "system", "content": safe_system_prompt}, {"role": "user", "content": user_content}], "stream": False, "options": {"temperature": AI_TEMPERATURE}}, timeout=float(OLLAMA_TIMEOUT))
        if resp.status_code == 200:
            local_text = re.sub(r"<think>.*?(?:</think>|$)", "", resp.json().get("message", {}).get("content", ""), flags=re.DOTALL).strip()
            if local_text: return local_text
        else: _log_provider_failure("Ollama", resp)
    except Exception as e: logger.error(f"Ollama API Error: {e}")
    return "Investigation Failed: All matrices exhausted or timed out."

def _location_kind(loc: Optional[str]) -> str:
    """Kind of location, read from the location string the extractor produced. 'unavailable' is kept as such, never replaced by a guess."""
    l = (loc or "").strip().lower()
    if not l or l == "unavailable": return "unavailable"
    if l.startswith("page"): return "pdf_page_image" if "(image)" in l else "pdf_page"
    if l.startswith("paragraph"): return "docx_paragraph"
    if l.startswith("table"): return "docx_table_row"
    if l.startswith("lines"): return "text_line_range"
    if l.startswith("image"): return "image_whole"
    if re.search(r",\s*row\s+\d+$", l): return "sheet_row"
    return "other"

def _provenance_dict(doc_node_id: str, file_label: str, fact: Dict[str, Any], doc_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evidence provenance. Only values the extractor actually produced are stored: location 'unavailable' stays 'unavailable' (location_status
    UNAVAILABLE), confidence_value stays None with basis 'not_measured' when no confidence exists. source_text_sha256 / length describe the RAW
    extracted text (the stored text may be redacted, see text_stored_redacted), so the source text can be re-verified without storing PII."""
    loc = fact.get("location")
    raw = fact.get("text") or ""
    prov = {"source_document_id": doc_node_id, "file": file_label, "location": loc, "method": fact["method"],
            "location_status": "UNAVAILABLE" if _location_kind(loc) == "unavailable" else "RECORDED", "location_kind": _location_kind(loc),
            "confidence_metadata": fact["confidence_metadata"], "confidence_value": fact.get("confidence_value"),
            "confidence_basis": fact.get("confidence_basis", "not_measured"), "confidence_available": fact.get("confidence_value") is not None,
            "source_text_sha256": hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest(), "source_text_length": len(raw),
            "text_stored_redacted": not STORE_RAW_PII}
    if doc_data:
        prov["document_file_hash"] = doc_data.get("file_hash")
        prov["document_source"] = doc_data.get("source")
    for extra_key in ("cells", "cell_range", "sheet", "section"):  # Excel cell-level provenance / docx section
        if extra_key in fact: prov[extra_key] = fact[extra_key]
    return prov

def _make_evidence_node(G: nx.MultiDiGraph, doc_node_id: str, file_label: str, fact: Dict[str, Any], policy_source: bool = False) -> str:
    """Creates Evidence node + Evidence --DERIVED_FROM--> Document. Stored text is redacted unless STORE_RAW_PII.
    The node keeps source document, location, source text, extraction method and (only if measured) confidence in `provenance`; source_file / source_location
    are copied to the node for direct querying. policy_source=True marks Evidence that holds RULEBOOK text (it defines a PolicyRule; it is not investigation evidence)."""
    stored_text = fact["text"] if STORE_RAW_PII else redact_pii(fact["text"])
    doc_data = G.nodes[doc_node_id] if G.has_node(doc_node_id) else {}
    ev_data = EvidenceNodeSchema(node_id=_nid("evid"), text=stored_text, provenance=_provenance_dict(doc_node_id, file_label, fact, doc_data)).model_dump()
    ev_id = ev_data["node_id"]
    G.add_node(ev_id, **ev_data, source_file=file_label, source_location=fact.get("location"))
    if policy_source:
        G.nodes[ev_id].update(evidence_role="POLICY_SOURCE", context_only=True, context_role="POLICY_SOURCE: rulebook text that defines PolicyRule nodes; not investigation evidence")
    G.add_edge(ev_id, doc_node_id, relation="DERIVED_FROM", location=fact.get("location"))
    return ev_id

def _is_policy_source_evidence(d: Dict[str, Any]) -> bool:
    return d.get("evidence_role") == "POLICY_SOURCE"

def _investigation_evidence(G: nx.MultiDiGraph) -> List[Tuple[str, Dict[str, Any]]]:
    """Evidence nodes that come from investigated documents (rulebook-source Evidence excluded: it defines rules, it is not searched/retrieved as case evidence)."""
    return [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "Evidence" and not _is_policy_source_evidence(d)]

_PROVENANCE_REQUIRED = ("source_document_id", "file", "location", "method", "confidence_basis")

def evidence_provenance_gaps(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """Evidence nodes whose provenance is incomplete. 'location_unavailable' / 'confidence_not_measured' are reported honestly (the system never invents
    either); the others (missing document link, empty text, missing keys) are defects."""
    gaps: List[Dict[str, Any]] = []
    for n, d in G.nodes(data=True):
        if d.get("type") != "Evidence": continue
        prov = d.get("provenance") or {}
        miss = [k for k in _PROVENANCE_REQUIRED if k not in prov or prov.get(k) in (None, "")]
        info: List[str] = []
        if "location" in miss and prov.get("location_status") == "UNAVAILABLE": miss.remove("location")
        if prov.get("location_status") == "UNAVAILABLE" or str(prov.get("location") or "").lower() == "unavailable": info.append("location_unavailable")
        if prov.get("confidence_value") is None: info.append("confidence_not_measured")
        if not (d.get("text") or "").strip(): miss.append("text")
        docs = _evidence_source_docs(G, n)
        if not docs: miss.append("DERIVED_FROM_document_edge")
        elif prov.get("source_document_id") and prov["source_document_id"] not in {x["document_id"] for x in docs}: miss.append("source_document_id_mismatch")
        if miss or info: gaps.append({"evidence_id": n, "defects": miss, "honest_gaps": info})
    return gaps

def _process_embedded_url(e_url: str, parent_doc_node_id: str, G: nx.MultiDiGraph, combined_text: str, master_temp_dir: str, access_logs: str, entity_cache: Optional[Dict[str, str]] = None, link_log: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
    """Download an embedded link INTO master_temp_dir (managed + cleaned by the task). Archives go through
    safe_extract_zip. Embedded content is ingested exactly like uploads (Evidence + Entity/Claim/Transaction)
    so the policy engine sees it too. e_url is the RAW (delimiter-cleaned, never query-redacted) url; logs use _safe_log_url.
    The link record says RETRIEVED only when files were actually obtained; it is merged by canonical key with any other record of the same link."""
    if entity_cache is None: entity_cache = {}
    dl_path, status_msg = download_url_to_temp(e_url, master_temp_dir)
    safe_url = _safe_log_url(e_url)
    if not dl_path: _record_link(link_log, e_url, False, status_msg, "embedded_link")
    if dl_path:
        try:
            members, arc_msg = _expand_input(dl_path, master_temp_dir)
            _record_link(link_log, e_url, bool(members), None if members else (arc_msg or "no files could be extracted from the download"), "embedded_link")
            if arc_msg:
                access_logs += f"\n[SYSTEM LOG] Embedded URL {safe_url}: {arc_msg}\n"
            texts = []
            for mp in members:
                t = extract_text_from_file(mp)
                if t.strip(): texts.append(t)
            v1_text = "\n".join(texts)
            if v1_text.strip():
                combined_text += f"\n--- Embedded Link ({safe_url}) ---\nStatus: Accessible\n{v1_text[:5000]}\n"
            else:
                combined_text += f"\n--- Embedded Link ({safe_url}) ---\nStatus: Accessible (no extractable text)\n"

            # Document node ALWAYS recorded once download succeeded (even if no text extracted).
            e_hash = generate_file_hash(dl_path)
            emb_doc_data = DocumentNodeSchema(node_id=_nid("doc"), filename=safe_url, source="embedded_link", file_hash=e_hash).model_dump()
            emb_doc_id = emb_doc_data["node_id"]
            G.add_node(emb_doc_id, **emb_doc_data)
            # LINKED_FROM (not DERIVED_FROM): keeps the Evidence->Document provenance convention unambiguous.
            G.add_edge(emb_doc_id, parent_doc_node_id, relation="LINKED_FROM")

            for mp in members:
                label = safe_url if len(members) == 1 else f"{safe_url}!{os.path.basename(mp)}"
                _ingest_facts(G, emb_doc_id, label, mp, entity_cache)
        finally:
            try: os.remove(dl_path)
            except Exception: pass
    else:
        access_logs += f"\n[SYSTEM LOG] Embedded URL {safe_url} NOT RETRIEVED: {status_msg}. Link content was not inspected; the block does not establish that the link is invalid, malicious, unrelated or a receipt, and says nothing about the expense itself.\n"
        combined_text += f"\n--- Embedded Link ({safe_url}) ---\nStatus: NOT RETRIEVED ({status_msg}); content NOT assessed (a retrieval failure says nothing about the link's validity)\n"
    return combined_text, access_logs

def chunked_iterable(iterable, size):
    """Yield successive chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def _batched(iterable, size):
    """Lazy batches from any iterable/generator (keeps DB object memory bounded)."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk: return
        yield chunk

def _normalize_currency(sym: str) -> str:
    c = sym.upper().rstrip('.')
    return {"RS": "INR", "₹": "INR", "$": "USD"}.get(c, c)

MAX_EXCERPT_CHARS = 1500

def _describe_evidence(G: nx.MultiDiGraph, n: str, data: Dict[str, Any], header: str, retrieval: str = "") -> str:
    """Full provenance block: source file @ location, extraction method, measured confidence (or 'not measured'),
    Excel cell range, COMPLETE excerpt up to MAX_EXCERPT_CHARS (explicit truncation notice beyond), supported nodes."""
    prov = data.get("provenance") or {}
    conf = prov.get("confidence_value")
    conf_txt = f"{conf} (basis: {prov.get('confidence_basis')})" if conf is not None else f"not measured (basis: {prov.get('confidence_basis', 'not_measured')})"
    src = "; ".join(f"{x['filename']} @ {x['location']}" for x in _evidence_source_docs(G, n)) or "NO SOURCE DOCUMENT LINK"
    full_text = data.get("text") or ""
    shown = full_text.replace(" | ", " ¦ ")  # '¦' = cell separator; avoids breaking markdown tables downstream
    if len(shown) > MAX_EXCERPT_CHARS:
        shown = shown[:MAX_EXCERPT_CHARS] + f"\n[TRUNCATED: showing {MAX_EXCERPT_CHARS} of {len(full_text)} chars]"
    supported = [_node_brief(G, v) for _, v, d in G.out_edges(n, data=True) if d.get("relation") == "SUPPORTS"]
    out = f"{header}\n"
    if retrieval: out += f"Retrieval: {retrieval}\n"
    if data.get("context_only"): out += "Role: POLICY CONTEXT ONLY (policy-listed information; any amount here is NOT a transaction record and does not corroborate or contradict any transaction)\n"
    out += f"Source: {src}\nExtraction: method={prov.get('method')}, confidence={conf_txt}\n"
    if prov.get("cell_range"): out += f"Cells: {prov.get('cell_range')} (sheet {prov.get('sheet')})\n"
    out += f"Excerpt ({len(full_text)} chars; '¦' separates spreadsheet cells):\n{shown}\nSupported nodes: {supported}\n\n"
    return out

def _ingest_facts(G: nx.MultiDiGraph, doc_node_id: str, file_name: str, file_path: str, entity_cache: Dict[str, str]) -> None:
    """Provenance facts -> Evidence nodes, and Entity / Claim / Transaction nodes with SUPPORTS edges.
    Amounts inside a policy-type document become Claim(PolicyListedAmount) nodes (context only), NEVER Transaction nodes."""
    facts = extract_provenance_facts(file_path)
    good_facts = [f for f in facts if f.get("method") not in ("error", "extraction_note")]
    policy_like = looks_like_policy_document(good_facts[0]["text"] if good_facts else "", file_name)
    if policy_like:
        G.nodes[doc_node_id]["policy_context"] = True
        G.nodes[doc_node_id]["context_role"] = "POLICY_CONTEXT: policy-listed information; not transaction records"
    for fact in facts:
        if fact.get("method") == "error":  # extraction failure is recorded on the Document, never stored as Evidence
            G.nodes[doc_node_id]["extraction_status"] = "error"
            G.nodes[doc_node_id]["extraction_error"] = redact_secrets(str(fact.get("text", "")))[:300]
            continue
        if fact.get("method") == "extraction_note":  # coverage limits (pages skipped, page cap): a note on the Document, never Evidence
            G.nodes[doc_node_id].setdefault("extraction_notes", []).append(redact_secrets(str(fact.get("text", "")))[:300])
            continue
        raw_fact_text = fact["text"]  # entities detected on RAW text; graph stores redacted values by default
        ev_id = _make_evidence_node(G, doc_node_id, file_name, fact)
        if policy_like:
            G.nodes[ev_id]["context_only"] = True
            G.nodes[ev_id]["context_role"] = "POLICY_CONTEXT"

        if HAS_PHONENUMBERS:
            try:
                for match in _phone_matches(raw_fact_text):
                    norm_phone = re.sub(r'\D', '', match.raw_string)
                    cache_key = f"phone_{norm_phone}"
                    if cache_key not in entity_cache:
                        ent_data = EntityNodeSchema(node_id=_nid("phone"), entity_type="Phone", value=_store_pii_value(match.raw_string, "phone"), is_valid=bool(phonenumbers.is_valid_number(match.number)), value_hash=_pii_hash(norm_phone)).model_dump()
                        G.add_node(ent_data["node_id"], **ent_data)
                        entity_cache[cache_key] = ent_data["node_id"]
                    G.add_edge(ev_id, entity_cache[cache_key], relation="SUPPORTS")
            except Exception: pass

        for cand in find_govid_candidates(raw_fact_text):  # detection (ID-like) kept separate from verification
            id_str = cand["digits"]
            cache_key = f"govid_{id_str}"
            if cache_key not in entity_cache:
                ver = verify_govid(id_str)
                ent_data = EntityNodeSchema(node_id=_nid("govid"), entity_type="GovID", value=_store_pii_value(id_str, "govid"), is_valid=ver["is_valid"], value_hash=_pii_hash(id_str),
                                            verification_status=ver["status"], verification_method=ver["method"], jurisdiction=ver["jurisdiction"], detection_basis=cand["basis"]).model_dump()
                ent_data["verification_note"] = ver.get("reason")
                G.add_node(ent_data["node_id"], **ent_data)
                entity_cache[cache_key] = ent_data["node_id"]
            G.add_edge(ev_id, entity_cache[cache_key], relation="SUPPORTS", method="govid_shape_detection", detection_basis=cand["basis"])

        money_matches = list(MONEY_PATTERN.finditer(raw_fact_text))
        txn_attrs = _extract_txn_attributes(raw_fact_text) if len(money_matches) == 1 else {}  # attributes only when unambiguous
        for match in money_matches:
            currency = _normalize_currency(match.group(1))
            try: val = float(match.group(2).replace(',', ''))
            except ValueError: continue
            if not math.isfinite(val): continue  # absurdly long digit run -> inf; would break JSON/DB persistence
            role, role_reason = classify_amount_role(raw_fact_text, match.start(), match.end(), file_name, txn_attrs, policy_like)
            if _is_policy_role(role):
                # Policy-listed / limit amount: context only. Not a Transaction, so it can never corroborate or contradict an employee transaction.
                claim_data = ClaimNodeSchema(node_id=_nid("claim"), claim_type="PolicyListedAmount", value=f"{currency} {val}").model_dump()
                G.add_node(claim_data["node_id"], **claim_data, amount=val, currency=currency, amount_role=role, amount_role_reason=role_reason, context_only=True,
                           source_file=file_name, label=_txn_label(raw_fact_text, match.start()),
                           note="policy-listed amount: context only, not a transaction record; no employee or transaction association is implied")
                G.add_edge(ev_id, claim_data["node_id"], relation="SUPPORTS", context_only=True)
                continue
            claim_data = ClaimNodeSchema(node_id=_nid("claim"), claim_type="MonetaryAmount", value=f"{currency} {val}").model_dump()
            G.add_node(claim_data["node_id"], **claim_data)
            G.add_edge(ev_id, claim_data["node_id"], relation="SUPPORTS")
            txn_data = TransactionNodeSchema(node_id=_nid("txn"), amount=val, currency=currency, label=_txn_label(raw_fact_text, match.start()), attributes=txn_attrs or None).model_dump()
            txn_data["amount_count_in_evidence"] = len(money_matches)
            txn_data["amount_role"], txn_data["amount_role_reason"] = role, role_reason
            txn_data["source_file"] = file_name
            G.add_node(txn_data["node_id"], **txn_data)
            G.add_edge(ev_id, txn_data["node_id"], relation="SUPPORTS")

def _init_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("system_schema", type="SystemSchema", allowed_types=ALLOWED_NODE_TYPES, allowed_relations=ALLOWED_RELATIONS)
    return G

def _build_policy_nodes(G: nx.MultiDiGraph, rulebook_path: Optional[str]) -> str:
    """Rulebook -> Document(role=RULEBOOK) <-DERIVED_FROM- Evidence(POLICY_SOURCE) -SUPPORTS-> PolicyRule / Policy, and PolicyRule -BELONGS_TO-> Policy.
    Nodes exist ONLY when the rulebook was actually read: no rulebook (or no readable rule line) => no Policy / PolicyRule / Document node is invented.
    Every rule line is traceable to its rulebook Document and exact location (page / paragraph / table row / sheet row / line range) through its Evidence node.
    Returns rulebook text (unchanged V1 behaviour)."""
    if not rulebook_path: return ""
    rule_facts = extract_provenance_facts(rulebook_path)
    good = [f for f in rule_facts if f.get("method") not in ("error", "extraction_note")]
    errs = [f for f in rule_facts if f.get("method") == "error"]
    notes = [f for f in rule_facts if f.get("method") == "extraction_note"]
    rulebook_text = "\n".join([f["text"] for f in good])
    if not rule_facts or is_sensitive_file(rulebook_path): return rulebook_text
    rb_name = os.path.basename(rulebook_path)
    doc_id = _add_document_node(G, rulebook_path)
    G.nodes[doc_id].update(source="rulebook", role="RULEBOOK", policy_context=True, context_role="POLICY_SOURCE: rulebook; defines PolicyRule nodes; not investigation evidence")
    if errs:  # extraction failure is a warning on the rulebook Document, never a rule line
        G.nodes[doc_id]["extraction_status"] = "error"
        G.nodes[doc_id]["extraction_error"] = redact_secrets("; ".join(str(f.get("text", "")) for f in errs))[:300]
    if notes: G.nodes[doc_id]["extraction_notes"] = [redact_secrets(str(f.get("text", "")))[:300] for f in notes]
    pol_id: Optional[str] = None
    for fact in good:
        ev_id: Optional[str] = None
        for line_no, line in enumerate(fact["text"].split('\n'), start=1):
            if not line.strip(): continue
            if pol_id is None:  # Policy only once a rule line exists in the rulebook
                pol_id = _nid("pol")
                G.add_node(pol_id, **PolicyNodeSchema(node_id=pol_id, name="Investigation Framework").model_dump(), source_file=rb_name, source_document_id=doc_id)
            if ev_id is None:
                ev_id = _make_evidence_node(G, doc_id, rb_name, fact, policy_source=True)
                G.add_edge(ev_id, pol_id, relation="SUPPORTS", method="rulebook_text", context_only=True)
            r_id = _nid("rule")
            G.add_node(r_id, **PolicyRuleNodeSchema(node_id=r_id, condition=line.strip()).model_dump(),
                       original_text=line.rstrip("\r"), source_file=rb_name, source_document_id=doc_id, source_evidence_id=ev_id,
                       source_location=fact.get("location"), source_line=line_no, extraction_method=fact.get("method"))
            G.add_edge(r_id, pol_id, relation="BELONGS_TO")
            G.add_edge(ev_id, r_id, relation="SUPPORTS", method="rulebook_line", source_line=line_no, context_only=True)
    return rulebook_text

def _add_document_node(G: nx.MultiDiGraph, file_path: str) -> str:
    doc_data = DocumentNodeSchema(node_id=_nid("doc"), filename=os.path.basename(file_path), file_hash=generate_file_hash(file_path)).model_dump()
    G.add_node(doc_data["node_id"], **doc_data)
    return doc_data["node_id"]

def build_evidence_graph(file_paths: List[str], rulebook_path: Optional[str] = None) -> nx.MultiDiGraph:
    """Offline pipeline (no DB, no network, no LLM): ingestion -> policy engine -> contradictions.
    Same helpers the Celery task uses; used by tests and by evaluation.py."""
    G = _init_graph()
    _build_policy_nodes(G, rulebook_path)
    entity_cache: Dict[str, str] = {}
    for path in file_paths:
        if is_sensitive_file(path): continue
        doc_id = _add_document_node(G, path)
        _ingest_facts(G, doc_id, os.path.basename(path), path, entity_cache)
    link_shared_terms(G)
    link_cross_documents(G)
    evaluate_policy_rules(G)
    if ENABLE_CONTRADICTION_HEURISTIC: detect_contradictions(G)
    classify_contradictions(G)
    return G

def _mark_record(db, inv_id: str, objective: str, message: str) -> None:
    """Best-effort status write after a failure (fresh transaction)."""
    try:
        db.rollback()
        rec = db.query(InvestigationRecord).filter_by(id=inv_id).first()
        if rec is None:
            rec = InvestigationRecord(id=inv_id, objective=objective, baseline_v1_result=message, research_v2_result=message)
            db.add(rec)
        else:
            rec.baseline_v1_result = message
            rec.research_v2_result = message
        db.commit()
    except Exception as e:
        logger.error(f"Could not persist failure status: {e}")
        try: db.rollback()
        except Exception: pass

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_batch_task(self, file_paths: list, url_list: list, rulebook_path: str, objective: str, ground_truth: Optional[dict] = None):
    # Everything that allocates a resource (temp dir, DB session) lives INSIDE try so finally always cleans up.
    master_temp_dir: Optional[str] = None
    db = None
    combined_text = ""
    access_logs = ""
    G = _init_graph()
    inv_id = self.request.id if self.request.id else str(uuid.uuid4())

    try:
        master_temp_dir = tempfile.mkdtemp(prefix="omni_")  # ONE managed dir, passed to every download/extract call
        db = SessionLocal()

        file_paths = file_paths or []
        url_list = url_list or []
        _url_originals: List[Tuple[str, str, str]] = []  # (cleaned url, entry exactly as given, location): original text kept apart from the normalized URL
        if not isinstance(file_paths, (list, tuple)) or not isinstance(url_list, (list, tuple)):
            raise PermanentTaskError("file_paths and url_list must be lists.")
        if not all(isinstance(p, str) for p in file_paths) or not all(isinstance(u, str) for u in url_list):
            raise PermanentTaskError("file_paths and url_list must contain strings only.")

        # 1. Early DB Registration (Ensures Record exists even if task fails). Retry resets stale failure text.
        existing_record = db.query(InvestigationRecord).filter_by(id=inv_id).first()
        if not existing_record:
            inv_record = InvestigationRecord(id=inv_id, objective=objective, baseline_v1_result="Processing...", research_v2_result="Processing...")
            db.add(inv_record)
            db.commit()
        else:
            inv_record = existing_record
            inv_record.baseline_v1_result = "Processing..."
            inv_record.research_v2_result = "Processing..."
            db.commit()

        rulebook_text = _build_policy_nodes(G, rulebook_path)

        # Every untrusted archive (upload, GitHub zip, URL download) goes through _expand_input -> safe_extract_zip.
        files_to_process: List[str] = []
        for path in file_paths:
            members, arc_msg = _expand_input(path, master_temp_dir)
            files_to_process.extend(members)
            if arc_msg: access_logs += f"\n[SYSTEM LOG] {arc_msg}\n"

        fetched_urls = 0
        MAX_GLOBAL_URLS = 20
        link_log: List[Dict[str, Any]] = []  # ONE record per unique link (canonical key), merged across the URL list and embedded links; read by V1 and V2 reports
        attempted_keys: set = set()  # canonical keys of links already handled: a link is fetched and ingested at most once

        for _u_idx, url in enumerate(url_list, 1):
            loc = f"uploaded URL list, entry {_u_idx}"
            raw_entry = url
            url = _clean_extracted_url(url)  # strips stray Markdown/sentence delimiters only; balanced () [] {} and the query string are kept
            if not url: continue
            _url_originals.append((url, raw_entry, loc))
            u_key = canonical_url_key(url)
            if u_key in attempted_keys:  # duplicate in the list (or already handled): add its source, never re-fetch / re-ingest
                _record_link(link_log, url, None, None, "uploaded_list", loc)
                continue
            if fetched_urls >= MAX_GLOBAL_URLS:  # recorded (never silently dropped) but not fetched
                _record_link(link_log, url, None, f"not attempted: global limit of {MAX_GLOBAL_URLS} URL retrievals reached", "uploaded_list", loc)
                continue
            attempted_keys.add(u_key)
            safe_url = _safe_log_url(url)
            if not is_safe_url(url):
                _reason = "blocked by SSRF protection (restricted, non-public or unvalidated address)"
                access_logs += f"\n[SYSTEM LOG] URL {safe_url} NOT RETRIEVED: {_reason}. Link content was not inspected; the block does not establish that the link is invalid, malicious, unrelated or a receipt, and says nothing about the expense itself.\n"
                _record_link(link_log, url, False, _reason, "uploaded_list", loc)
                continue

            gh_match = re.match(r'^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?(?:[?#].*)?$', url)
            if gh_match:
                base_repo = f"https://github.com/{gh_match.group(1)}/{gh_match.group(2)}"
                dl_path, status = "", "No branch archive downloaded."
                for branch in ("main", "master"):
                    candidate = f"{base_repo}/archive/refs/heads/{branch}.zip"
                    if not is_safe_url(candidate):  # constructed URL validated too (and again inside the downloader)
                        status = "SSRF BLOCKED: Restricted/Invalid address."
                        break
                    dl_path, status = download_url_to_temp(candidate, master_temp_dir)
                    if dl_path and os.path.exists(dl_path) and os.path.getsize(dl_path) >= 1000: break
                members, arc_msg = [], ""
                if dl_path:
                    members, arc_msg = _expand_input(dl_path, master_temp_dir)
                    files_to_process.extend(members)
                    if arc_msg: access_logs += f"\n[SYSTEM LOG] Repo URL {safe_url}: {arc_msg}\n"
                else: access_logs += f"\n[SYSTEM LOG] Repo URL {safe_url} NOT RETRIEVED: {status}. Link content was not assessed.\n"
                _record_link(link_log, url, bool(members), None if members else (arc_msg or status), "uploaded_list", loc)  # RETRIEVED only if files were actually obtained
                fetched_urls += 1
            else:
                dl_path, status = download_url_to_temp(url, master_temp_dir)
                members, arc_msg = [], ""
                if dl_path:
                    members, arc_msg = _expand_input(dl_path, master_temp_dir)  # downloaded ZIPs were previously NOT unpacked
                    files_to_process.extend(members)
                    if arc_msg: access_logs += f"\n[SYSTEM LOG] Document URL {safe_url}: {arc_msg}\n"
                else: access_logs += f"\n[SYSTEM LOG] Document URL {safe_url} NOT RETRIEVED: {status}. Link content was not assessed.\n"
                _record_link(link_log, url, bool(members), None if members else (arc_msg or status), "uploaded_list", loc)  # RETRIEVED only if files were actually obtained
                fetched_urls += 1

        _attach_link_originals(link_log, _url_originals)
        total_files = len(files_to_process)
        combined_text += access_logs

        if total_files == 0:  # nothing to analyze (access logs alone are not evidence)
            # Raise (Celery state FAILURE, no retry): the except block persists the failure on the record.
            raise PermanentTaskError("Extraction Blocked: no files or text could be obtained (inputs empty, rejected, or URLs blocked)." + (f" Log: {redact_secrets(access_logs.strip())[:800]}" if access_logs.strip() else ""))

        entity_cache: Dict[str, str] = {}
        v1_govid_sources: List[str] = []  # 'file @ location' of each ID-like value actually present in extracted text
        v1_govid_statuses: List[str] = []  # V1 brief section D (statuses only; digits are never kept)
        v1_amount_entries: List[Dict[str, Any]] = collect_amount_entries(rulebook_text, "rulebook", True) if rulebook_text else []  # V1-only: from V1's own extracted text

        for idx, file_path in enumerate(files_to_process):
            file_name = os.path.basename(file_path)
            self.update_state(state='PROGRESS', meta={'current': idx+1, 'total': total_files, 'file': file_name})

            if is_sensitive_file(file_path):
                combined_text += f"\n[SYSTEM LOG] File {file_name} SKIPPED: sensitive credential/secret file type.\n"
                continue
            try:
                if os.path.getsize(file_path) > MAX_LOCAL_FILE_BYTES:
                    combined_text += f"\n[SYSTEM LOG] File {file_name} SKIPPED: exceeds {MAX_LOCAL_FILE_BYTES // (1024*1024)} MB limit.\n"
                    continue
            except OSError:
                combined_text += f"\n[SYSTEM LOG] File {file_name} SKIPPED: unreadable.\n"
                continue

            # redact_code=False: URL harvesting needs intact query strings (Drive ?id=...). Nothing here leaves the
            # process unredacted: combined_text is redact_pii()'d inside analyze_compliance_with_matrix.
            raw_text = extract_text_from_file(file_path, redact_code=False)
            if raw_text.strip() and "[Extraction Error]" not in raw_text:
                v1_amount_entries.extend(collect_amount_entries(raw_text, file_name))
                v1_govid_statuses.extend(verify_govid(c_["digits"])["status"] for c_ in find_govid_candidates(raw_text))
                v1_govid_sources.extend(_govid_occurrences(raw_text, file_name))
                if len(combined_text) < 100000: # Protect RAM before payload cap
                    combined_text += f"\n\n========================================\nFILE NAME: {file_name}\n========================================\n\n{generate_system_validation_report(raw_text)}{generate_amount_role_report(raw_text, file_name)}{raw_text}"

            doc_node_id = _add_document_node(G, file_path)

            if raw_text.strip():
                doc_attempts = 0
                for e_url in extract_urls(raw_text, MAX_EMBEDDED_URLS_SCANNED_PER_DOC):  # cleaned (no stray ] ) > quotes / sentence punctuation), de-duplicated by canonical key
                    e_key = canonical_url_key(e_url)
                    if e_key in attempted_keys:  # same link already handled via the URL list or another document: record the source, do not fetch/ingest again
                        _record_link(link_log, e_url, None, None, "embedded_link")
                        continue
                    if doc_attempts >= MAX_EMBEDDED_URLS_PER_DOC or fetched_urls >= MAX_GLOBAL_URLS:  # recorded (never silently dropped) but not fetched
                        _record_link(link_log, e_url, None, "not attempted: per-document / global link retrieval limit reached", "embedded_link")
                        continue
                    attempted_keys.add(e_key)
                    doc_attempts += 1
                    combined_text, access_logs = _process_embedded_url(e_url, doc_node_id, G, combined_text, master_temp_dir, access_logs, entity_cache, link_log)
                    fetched_urls += 1
                for occ in extract_url_occurrences(raw_text, MAX_EMBEDDED_URLS_SCANNED_PER_DOC):  # provenance pass: EVERY occurrence keeps its exact source text + location (any file type; status is never changed)
                    _ls = raw_text.rfind("\n", 0, occ["start"]) + 1
                    _record_link(link_log, occ["url"], None, None, None, f"{file_name} @ {_entry_location(raw_text, _ls, file_name)}", occ["raw"])

            _ingest_facts(G, doc_node_id, file_name, file_path, entity_cache)
            _attach_link_locations(link_log, G, doc_node_id, file_name)

        # Deterministic policy engine + contradiction heuristic (graph-only; failures never kill the task)
        decisions_created = contradiction_edges = term_links = 0
        try:
            term_links = link_shared_terms(G)
        except Exception as e: logger.exception(f"Term linking failed (continuing): {e}")
        xdoc_summary: Dict[str, Any] = {"status": "NOT_RUN"}
        try:  # Phase A cross-document evidence linking: graph-only, no decision / verdict; failure never kills the task
            xdoc_summary = link_cross_documents(G)
        except Exception as e:
            xdoc_summary = {"status": "ERROR", "reason": str(e)[:200]}
            logger.exception(f"Cross-document linking failed (continuing): {e}")
        try:  # rulebook -> compiler -> graph-derived facts/evidence -> deterministic rule engine; results go onto the PolicyRule nodes used by evaluate_policy_rules below
            apply_compiled_policy(G, rulebook_text)
        except Exception as e: logger.exception(f"Compiled policy stage failed (continuing with legacy engine): {e}")
        try:
            decisions_created = len(evaluate_policy_rules(G))
        except Exception as e: logger.exception(f"Policy engine failed (continuing without decisions): {e}")
        try:
            if ENABLE_CONTRADICTION_HEURISTIC: contradiction_edges = detect_contradictions(G)
        except Exception as e: logger.exception(f"Contradiction detection failed (continuing): {e}")
        xcon_summary: Dict[str, Any] = {"status": "NOT_RUN"}
        try: xcon_summary = classify_contradictions(G)  # graph-only classification of linked-document contradictions; never a violation
        except Exception as e:
            xcon_summary = {"status": "ERROR", "reason": str(e)[:200]}
            logger.exception(f"Contradiction classification failed (continuing): {e}")

        link_log = normalize_link_log(link_log)  # ONE de-duplicated record set; V1 and V2 report sections below both read exactly this

        t0 = time.time()
        ai_report_v1 = analyze_compliance_with_matrix(build_v1_amount_brief(v1_amount_entries, policy_coverage(G), v1_govid_statuses, link_log, v1_govid_sources) + combined_text, rulebook_text, objective, is_v2=False)
        v1_latency = time.time() - t0
        ai_report_v1 = _sanitize_report_language(ai_report_v1)
        _rel_v1 = relevant_unretrieved_links(G, link_log, ranked_evidence_ids(G, objective)[:V2_TOP_K], objective)
        ai_report_v1 = ensure_unretrieved_link_note(ai_report_v1, v2_unretrieved_link_note(_rel_v1), v2_unretrieved_link_row(_rel_v1))  # same system relevance rule and wording as V2 (lexical top-K evidence, or an objective that refers to links); irrelevant links are not added
        _llm_v1_text = ai_report_v1  # LLM-written text only (before deterministic sections are appended): what the evaluation scores
        ai_report_v1 = assemble_report(ai_report_v1, v1_audit_section(v1_amount_entries), report_appendix_details(G, v1_latency, link_log), report_key_limits(G, link_log), {l["key"] for l in link_log})

        t1 = time.time()
        v2_subgraph_context, v2_stats = build_v2_context(G, objective, link_log)

        ai_report_v2 = _sanitize_report_language(analyze_compliance_with_matrix(v2_subgraph_context, rulebook_text, objective, is_v2=True))
        ai_report_v2 = _guard_trace_claims(ai_report_v2, v2_stats.get("trace"))  # LLM trace statements must match the system counts/labels
        ai_report_v2 = _guard_trace_labels(ai_report_v2, v2_stats.get("trace"))  # any edge list the LLM writes is labelled Complete / Sample / Example path
        ai_report_v2 = ensure_unretrieved_link_note(ai_report_v2, v2_stats.get("relevant_unretrieved_note", ""), v2_stats.get("relevant_unretrieved_row"))
        v2_latency = time.time() - t1
        _llm_v2_text = ai_report_v2

        # Evaluation against labelled ground truth (task argument, or JSON file named by EVAL_GROUND_TRUTH_PATH). No ground truth => everything stays NOT_MEASURED.
        evaluation: Dict[str, Any]
        _gt = load_ground_truth(ground_truth) or load_ground_truth(EVAL_GROUND_TRUTH_PATH)
        if _gt:
            try:
                _cfg = {"external_ai_enabled": EXTERNAL_AI_ENABLED, "model": (OLLAMA_PRO_MODEL if not EXTERNAL_AI_ENABLED else None), "provider": None, "k": V2_TOP_K}
                _recs = [build_experiment_record(inv_id, "V1", 1, G, objective, _llm_v1_text, v1_latency, _gt, _cfg),
                         build_experiment_record(inv_id, "V2", 1, G, objective, _llm_v2_text, v2_latency, _gt, _cfg)]
                evaluation = {"status": "MEASURED", "records": _recs, "summary": summarize_experiment_records(_recs)}
            except Exception as e:
                logger.exception(f"Evaluation failed (continuing): {e}")
                evaluation = {"status": "ERROR", "reason": f"evaluation failed: {str(e)[:300]}"}
        else:
            evaluation = {"status": NOT_MEASURED, "reason": "no labelled ground truth supplied (task argument ground_truth or EVAL_GROUND_TRUTH_PATH); no retrieval or decision-quality metric was computed or estimated"}
        try: _chain_summary = graph_chain_summary(G)
        except Exception as e: _chain_summary = {"status": "ERROR", "reason": str(e)[:200]}
        try: _gq_structural = evaluate_graph_queries(G)["structural_checks"]
        except Exception as e: _gq_structural = {"status": "ERROR", "reason": str(e)[:200]}
        ai_report_v2 = assemble_report(ai_report_v2, v2_audit_section(G), report_appendix_details(G, v2_latency, link_log) + v2_trace_section(v2_stats) + v2_cross_document_section(v2_stats) + v2_contradiction_section(G), report_key_limits(G, link_log), {l["key"] for l in link_log})

        # Final DB Update & Batch Saving (idempotent: a retried task replaces, never duplicates, graph rows)
        inv_record.baseline_v1_result = _json_safe(ai_report_v1)
        inv_record.research_v2_result = _json_safe(ai_report_v2)
        inv_record.metrics = _json_safe({"v1_latency": v1_latency, "v2_latency": v2_latency, "ceg_nodes": G.number_of_nodes(), "ceg_edges": G.number_of_edges(), "decisions": decisions_created, "violations": sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Decision" and d.get("verdict") == "VIOLATION"), "contradiction_edges": contradiction_edges, "policy_coverage": policy_coverage(G), "policy_rules": policy_rule_report(G), "supporting_links": _links_for_output(link_log), "term_links": term_links, "cross_document_links": xdoc_summary, "cross_document_reasoning": v2_stats.get("cross_document"), "contradiction_classification": xcon_summary, "evaluation": evaluation, "graph_chain": _chain_summary, "graph_query_structural_checks": _gq_structural, "v2_nodes_reached": v2_stats.get("nodes_reached", 0), "v2_retrieved": v2_stats["retrieved"], "v2_expanded": v2_stats["expanded"], "v2_traversal_edges": v2_stats.get("traversal_edges", 0), "v2_inspected_edges": v2_stats.get("inspected_edges", 0), "v2_context_only_expanded": v2_stats.get("context_only_expanded", 0), "v2_expansion_trace": v2_stats.get("expansion_trace", [])})
        db.add(inv_record)

        db.query(EvidenceEdge).filter(EvidenceEdge.investigation_id == inv_id).delete(synchronize_session=False)
        db.query(EvidenceNode).filter(EvidenceNode.investigation_id == inv_id).delete(synchronize_session=False)

        # Generators + lazy batches: ORM objects exist only 500 at a time.
        # properties go through _json_safe (no NaN/Inf/NUL, no non-JSON types) so JSON columns never reject a row.
        # Edges only reference nodes that exist in G, so edge source/target FKs always resolve.
        node_objs = (EvidenceNode(investigation_id=inv_id, node_id=u, node_type=str(d.get("type", "Unknown")), properties=_json_safe(d)) for u, d in G.nodes(data=True))
        edge_objs = (EvidenceEdge(investigation_id=inv_id, source_id=u, target_id=v, relation=str(d.get("relation", "UNKNOWN")), properties=_json_safe(d)) for u, v, k, d in G.edges(keys=True, data=True))

        for chunk in _batched(node_objs, 500): db.bulk_save_objects(chunk)
        for chunk in _batched(edge_objs, 500): db.bulk_save_objects(chunk)

        db.commit()

        return {"status": "completed", "supporting_links": _links_for_output(link_log), "results": [
            {"file": "V1 Monolith", "findings": ai_report_v1},
            {"file": f"V2 Graph Prototype ({G.number_of_nodes()} Nodes)", "findings": ai_report_v2}
        ]}
    except Exception as e:
        logger.error(f"Task Failed: {redact_secrets(str(e))[:500]}")
        is_transient = isinstance(e, TRANSIENT_ERRORS)
        will_retry = is_transient and self.request.retries < self.max_retries
        msg = f"Task Failed: {e}"[:2000]
        if will_retry:
            msg = f"Task attempt {self.request.retries + 1} failed (retrying): {e}"[:2000]
        if db is not None: _mark_record(db, inv_id, objective, msg)
        if will_retry:
            raise self.retry(exc=e, countdown=60)
        raise  # permanent error, or retries exhausted: record already marked failed
    finally:
        if db is not None:
            try: db.close()
            except Exception: pass
        if master_temp_dir and not os.getenv("DEBUG_KEEP_FILES"):
            shutil.rmtree(master_temp_dir, ignore_errors=True)

# --- OMNI-BENCH FOUNDATION (schema + leakage validation only; no cases populated, no runner, no scoring; offline/deterministic, no LLM/network) ---
OMNI_BENCH_CATEGORIES: Tuple[Tuple[str, str, str], ...] = (
    ("simple_compliance", "Simple compliance", "Single document, single clear rule, unambiguous outcome."),
    ("multi_document_compliance", "Multi-document compliance", "Outcome requires combining evidence across several documents."),
    ("contradictory_evidence", "Contradictory evidence", "Documents disagree on a fact relevant to the rule."),
    ("missing_evidence", "Missing evidence", "Evidence required by the rule is absent from the document set."),
    ("policy_exceptions", "Policy exceptions", "A stated exception overrides the general rule."),
    ("temporal_violations", "Temporal violations", "Dates, deadlines or validity windows decide the outcome."),
    ("entity_mismatch", "Entity mismatch", "Evidence refers to a different entity than the one under review."),
    ("distractor_documents", "Distractor documents", "Irrelevant documents are mixed in with the relevant ones."),
    ("ocr_noise", "OCR noise", "Document text contains OCR-style corruption."),
    ("policy_paraphrasing", "Policy paraphrasing", "The same policy is expressed in different wording."),
    ("ambiguous_policies", "Ambiguous policies", "Policy wording admits more than one reasonable reading."),
    ("adversarial_document_content", "Adversarial document content", "Documents contain misleading or manipulative claims."),
    ("prompt_injection_inside_documents", "Prompt injection inside documents", "Documents embed instructions aimed at the system."),
    ("conflicting_policies", "Conflicting policies", "Two applicable policies give incompatible requirements."),
    ("evidence_removal", "Evidence removal", "Decisive evidence has been removed relative to a base case."),
)
OMNI_BENCH_CATEGORY_IDS = tuple(c[0] for c in OMNI_BENCH_CATEGORIES)
OMNI_BENCH_SPLITS: Dict[str, str] = {
    "TRAIN": "Cases usable for development of methods and prompts.",
    "DEV": "Cases usable for tuning and model selection; not for final reporting.",
    "TEST": "Held-out cases for final reporting only; never used for tuning.",
}
OMNI_BENCH_DECISIONS = ("COMPLIANT", "NON_COMPLIANT", "INSUFFICIENT_EVIDENCE")
OMNI_BENCH_REQUIRED_FIELDS = ("case_id", "family_id", "split", "document_set", "policy", "expected_decision", "applicable_policy_rule",
                              "supporting_evidence", "contradicting_evidence", "missing_evidence", "expected_escalation",
                              "counterfactual_correction", "difficulty_category")
OMNI_BENCH_LABEL_SOURCES = ("human_annotation", "rule_derived", "synthetic_construction")
OMNI_BENCH_FORBIDDEN_KEYS = ("prediction", "predictions", "predicted_decision", "system_output", "model_output", "system_prediction", "model_prediction")
OMNI_BENCH_METADATA: Dict[str, Any] = {
    "benchmark_name": "OMNI-Bench", "benchmark_version": "0.1.0", "schema_version": "1.0",
    "split_definitions": dict(OMNI_BENCH_SPLITS),
    "category_definitions": {c[0]: {"name": c[1], "definition": c[2]} for c in OMNI_BENCH_CATEGORIES},
    "leakage_policy": ("Cases sharing a family_id (related or near-duplicate variants) must all be in one split; case_id is globally unique; "
                       "no identical document_set+policy content may appear in two splits; ground truth is never derived from system predictions."),
    "reproducibility_policy": "Offline, deterministic, no LLM or network calls; identical inputs and seed produce identical cases and validation results.",
    "generation_methodology": "UNSPECIFIED: to be defined before the benchmark is populated.",
    "seed_policy": "Any randomized generation must use an explicit integer seed recorded with the benchmark; no unseeded randomness or wall-clock input.",
}
OMNI_BENCH_CASES: List[Dict[str, Any]] = []  # intentionally empty: the benchmark is not populated yet


def _omni_bench_has_forbidden_key(o: Any) -> bool:
    if isinstance(o, dict):
        return any(str(k).lower() in OMNI_BENCH_FORBIDDEN_KEYS or str(k).lower().startswith("predicted_") or _omni_bench_has_forbidden_key(v) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return any(_omni_bench_has_forbidden_key(v) for v in o)
    return False


def validate_omni_bench_metadata(meta: Optional[Dict[str, Any]] = None) -> List[str]:
    meta = OMNI_BENCH_METADATA if meta is None else meta
    probs = [f"metadata: missing/empty '{k}'" for k in ("benchmark_name", "benchmark_version", "schema_version", "split_definitions", "category_definitions",
                                                         "leakage_policy", "reproducibility_policy", "generation_methodology", "seed_policy") if not meta.get(k)]
    if set(meta.get("split_definitions") or {}) != set(OMNI_BENCH_SPLITS): probs.append("metadata: split_definitions must be exactly TRAIN/DEV/TEST")
    if list(meta.get("category_definitions") or {}) != list(OMNI_BENCH_CATEGORY_IDS): probs.append("metadata: category_definitions must be the 15 defined categories in order")
    return probs


def validate_omni_bench_cases(cases: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Return a sorted list of problems (empty == valid). Pure and deterministic."""
    cases = OMNI_BENCH_CASES if cases is None else cases
    probs: List[str] = []
    seen_ids: set = set()
    fam_split: Dict[str, str] = {}
    content_split: Dict[str, Tuple[str, str]] = {}
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            probs.append(f"case[{i}]: not an object"); continue
        cid = c.get("case_id")
        tag = cid if isinstance(cid, str) and cid else f"case[{i}]"
        missing = [f for f in OMNI_BENCH_REQUIRED_FIELDS if f not in c or c[f] is None or c[f] == "" or (f in ("document_set", "policy", "applicable_policy_rule") and not c[f])]
        for f in missing: probs.append(f"{tag}: missing required field '{f}'")
        if isinstance(cid, str) and cid:
            if cid in seen_ids: probs.append(f"{tag}: duplicate case_id")
            seen_ids.add(cid)
        if _omni_bench_has_forbidden_key(c): probs.append(f"{tag}: prediction-derived ground truth (forbidden prediction field)")
        if "label_source" in c and c["label_source"] not in OMNI_BENCH_LABEL_SOURCES: probs.append(f"{tag}: label_source must be one of {OMNI_BENCH_LABEL_SOURCES}")
        if "difficulty_category" not in missing and c["difficulty_category"] not in OMNI_BENCH_CATEGORY_IDS: probs.append(f"{tag}: invalid difficulty_category")
        split = c.get("split")
        if "split" not in missing and split not in OMNI_BENCH_SPLITS: probs.append(f"{tag}: invalid split assignment")
        fam = c.get("family_id")
        if "family_id" not in missing:
            if not isinstance(fam, str): probs.append(f"{tag}: family_id must be a string")
            elif split in OMNI_BENCH_SPLITS:
                if fam_split.setdefault(fam, split) != split: probs.append(f"{tag}: family_id '{fam}' appears in multiple splits")
        # ground-truth shape
        if "expected_decision" not in missing and c["expected_decision"] not in OMNI_BENCH_DECISIONS: probs.append(f"{tag}: malformed expected_decision")
        if "expected_escalation" not in missing and not isinstance(c["expected_escalation"], bool): probs.append(f"{tag}: malformed expected_escalation (bool required)")
        for f in ("policy", "applicable_policy_rule", "counterfactual_correction"):
            if f not in missing and not isinstance(c[f], str): probs.append(f"{tag}: malformed {f} (string required)")
        doc_ids: set = set()
        if "document_set" not in missing:
            ds = c["document_set"]
            if not isinstance(ds, list) or not all(isinstance(d, dict) and isinstance(d.get("doc_id"), str) and d["doc_id"] and isinstance(d.get("text"), str) for d in ds):
                probs.append(f"{tag}: malformed document_set (list of {{doc_id, text}})")
            else:
                doc_ids = {d["doc_id"] for d in ds}
                if len(doc_ids) != len(ds): probs.append(f"{tag}: duplicate doc_id in document_set")
        for f in ("supporting_evidence", "contradicting_evidence", "missing_evidence"):
            if f in missing: continue
            v = c[f]
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v): probs.append(f"{tag}: malformed {f} (list of strings)"); continue
            if f != "missing_evidence" and doc_ids and not set(v) <= doc_ids: probs.append(f"{tag}: {f} references doc_id not in document_set")
        if c.get("expected_decision") == "INSUFFICIENT_EVIDENCE" and isinstance(c.get("missing_evidence"), list) and not c["missing_evidence"]:
            probs.append(f"{tag}: INSUFFICIENT_EVIDENCE requires non-empty missing_evidence")
        # near-duplicate content leakage across splits
        if not missing and split in OMNI_BENCH_SPLITS:
            try: h = hashlib.sha256(json.dumps([c["document_set"], c["policy"]], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            except (TypeError, ValueError): continue
            if content_split.setdefault(h, (split, tag))[0] != split: probs.append(f"{tag}: identical document_set+policy content also in split '{content_split[h][0]}' (case {content_split[h][1]})")
    return sorted(probs)


# --- OMNI-BENCH POPULATION (case data lives in omni_bench_cases.py beside this file; deterministic, offline; no labels come from running this system) ---
from .omni_bench_cases import (OMNI_BENCH_DOCUMENT_CONTENT_POLICY, OMNI_BENCH_GENERATION_METHODOLOGY, OMNI_BENCH_SEED_POLICY, build_omni_bench_cases)
OMNI_BENCH_METADATA.update({"generation_methodology": OMNI_BENCH_GENERATION_METHODOLOGY, "seed_policy": OMNI_BENCH_SEED_POLICY,
                            "document_content_policy": OMNI_BENCH_DOCUMENT_CONTENT_POLICY})
OMNI_BENCH_CASES.extend(build_omni_bench_cases())