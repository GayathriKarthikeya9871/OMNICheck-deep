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
ALLOWED_RELATIONS = ["DERIVED_FROM", "LINKED_FROM", "SUPPORTS", "CONTRADICTS", "VIOLATES", "SATISFIES", "EVALUATES", "HAS_RISK", "BELONGS_TO"]

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
        logger.info(f"Policy rule {rule_id}: parsed={spec is not None} verdict={verdict} text={redact_pii(rd.get('condition', ''))[:120]!r}")
        G.nodes[rule_id]["engine_run"] = True
        G.nodes[rule_id]["parsed"] = spec is not None  # RECOGNIZED only: text mapped to the supported rule format; says nothing about being executable
        G.nodes[rule_id]["executable"] = spec is not None and verdict != "UNEVALUATED"  # deterministic rule AND required support/configuration available
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
        _absence = spec is not None and spec["subject"] == "KEYWORD" and spec["mode"] == "REQUIRE" and verdict == "VIOLATION" and not viol
        G.nodes[dec_id].update(violation_status=("CONFIRMED_BY_DETERMINISTIC_RULE" if (verdict == "VIOLATION" and not _absence) else
                                                 "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE" if _absence else None),
                               heuristic_inputs_used=False,  # contradictions / shared terms are never inputs of a verdict
                               absence_scope_evidence_ids=([n for n, _ in _investigation_evidence(G)] if _absence else []),
                               result_source="deterministic_policy_engine", evidence_used=used_ev, basis_node_ids=list(basis_nodes),
                               result_reason=rationale, parsed_spec=spec, rule_source_file=rd.get("source_file"), rule_source_location=rd.get("source_location"), unevaluated_reason=uneval_reason,
                               extraction_gap_documents=(extraction_gaps(G) if verdict == "INCONCLUSIVE" else []))

        if verdict == "VIOLATION":
            risk_data = RiskNodeSchema(node_id=_nid("risk"), severity=severity or "MEDIUM", rule_id=rule_id).model_dump()
            G.add_node(risk_data["node_id"], **risk_data, severity_source=("rule_configured" if _RULE_SEVERITY.search(rd.get("condition", "")) else "default_when_unspecified"),
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
    return {"total": discovered, "discovered": discovered, "parsed": recognized, "recognized": recognized, "executable": attempted, "attempted": attempted,
            "evaluated": evaluated, "evaluated_with_result": evaluated, "attempted_without_result": attempted - evaluated, "not_determined": attempted - evaluated,
            "unevaluated": discovered - attempted, "recognized_not_executable": recognized - attempted, "parsed_but_not_evaluable": recognized - attempted, "verdicts": dict(verdicts)}

def _coverage_sentence(c: Dict[str, Any]) -> str:
    return (f"{c['discovered']} rule(s) discovered; {c['recognized']} recognized in the supported rule format (recognized/stored only; not necessarily executable); "
            f"{c['executable']} executable (deterministic rule with required support/configuration available); {c['attempted']} attempted (evaluation started); "
            f"{c['evaluated_with_result']} evaluated with a result (SATISFIED/VIOLATION); {c['attempted_without_result']} attempted without a result (INCONCLUSIVE/NOT_APPLICABLE); "
            f"{c['unevaluated']} UNEVALUATED (not attempted: support, configuration or unambiguous interpretation unavailable; NOT checked; not counted as attempted)")

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
        out += "| Rule | Source | Recognized | Executable | Attempted | Verdict | Reason |\n|---|---|---|---|---|---|---|\n"
        for r in rows[:30]:
            src = f"{r.get('source_file') or ''} @ {r.get('source_location') or r.get('source_line') or 'n/a'}"
            out += (f"| {_md_cell(r['condition'], 140)} | {_md_cell(src, 100)} | {'yes' if r['recognized'] else 'no'} | {'yes' if r['executable'] else 'no'} | "
                    f"{'yes' if r['attempted'] else 'no'} | {r['verdict']} | {_md_cell(r.get('unevaluated_reason') or r.get('rationale'), 240)} |\n")
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
                     "evaluated": dd.get("verdict") in ("VIOLATION", "SATISFIED") and bool(rd.get("parsed"))})
    return rows

def format_policy_results_for_context(G: nx.MultiDiGraph, max_decisions: int = 30, max_nodes: int = 5) -> str:
    decisions = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "Decision"]
    if not decisions: return ""
    cov = policy_coverage(G)
    out = (f"--- POLICY EVALUATION RESULTS [DETERMINISTIC; not LLM interpretation] (UNEVALUATED = no evaluation performed: unsupported/ambiguous rule or required configuration missing, NOT checked; INCONCLUSIVE = attempted, evidence insufficient) ---\n"
           f"Coverage: {_coverage_sentence(cov)}.\n")
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

def build_v2_context(G: nx.MultiDiGraph, objective: str, link_log: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
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
                 "relevant_unretrieved_note": v2_unretrieved_link_note(rel_links), "relevant_unretrieved_row": v2_unretrieved_link_row(rel_links)}

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
               lexical_top_k=score(top), lexical_plus_graph_expansion=score(list(dict.fromkeys(top + expanded))))
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
        for name, key in (("retrieval_lexical_top_k", "lexical_top_k"), ("retrieval_with_graph_expansion", "lexical_plus_graph_expansion")):
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
            "25. Government ID recommendations are conditional: first identify the jurisdiction and ID format; then determine whether an applicable checksum method exists and its configuration is available; run checksum validation only if both are true, otherwise keep UNVERIFIED and request an independent verification method. A checksum never proves authenticity, ownership or document validity, and ID validation of any kind (format or checksum) never confirms that an ID is authentic, genuine, issued or owned by anyone: never write that validating the ID confirms, proves or establishes any of these.\n\n"
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
    evaluate_policy_rules(G)
    if ENABLE_CONTRADICTION_HEURISTIC: detect_contradictions(G)
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
        try:
            decisions_created = len(evaluate_policy_rules(G))
        except Exception as e: logger.exception(f"Policy engine failed (continuing without decisions): {e}")
        try:
            if ENABLE_CONTRADICTION_HEURISTIC: contradiction_edges = detect_contradictions(G)
        except Exception as e: logger.exception(f"Contradiction detection failed (continuing): {e}")

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
        ai_report_v2 = assemble_report(ai_report_v2, v2_audit_section(G), report_appendix_details(G, v2_latency, link_log) + v2_trace_section(v2_stats), report_key_limits(G, link_log), {l["key"] for l in link_log})

        # Final DB Update & Batch Saving (idempotent: a retried task replaces, never duplicates, graph rows)
        inv_record.baseline_v1_result = _json_safe(ai_report_v1)
        inv_record.research_v2_result = _json_safe(ai_report_v2)
        inv_record.metrics = _json_safe({"v1_latency": v1_latency, "v2_latency": v2_latency, "ceg_nodes": G.number_of_nodes(), "ceg_edges": G.number_of_edges(), "decisions": decisions_created, "violations": sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Decision" and d.get("verdict") == "VIOLATION"), "contradiction_edges": contradiction_edges, "policy_coverage": policy_coverage(G), "policy_rules": policy_rule_report(G), "supporting_links": _links_for_output(link_log), "term_links": term_links, "evaluation": evaluation, "graph_chain": _chain_summary, "graph_query_structural_checks": _gq_structural, "v2_nodes_reached": v2_stats.get("nodes_reached", 0), "v2_retrieved": v2_stats["retrieved"], "v2_expanded": v2_stats["expanded"], "v2_traversal_edges": v2_stats.get("traversal_edges", 0), "v2_inspected_edges": v2_stats.get("inspected_edges", 0), "v2_context_only_expanded": v2_stats.get("context_only_expanded", 0), "v2_expansion_trace": v2_stats.get("expansion_trace", [])})
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