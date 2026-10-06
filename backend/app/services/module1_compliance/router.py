import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import List
from fastapi.templating import Jinja2Templates
from .tasks import process_document_batch_task

# --- Policy-to-Executable Compliance Compiler (added) ---
import json
import re
from typing import Any, Dict, Optional
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from .policy_compiler import compile_policy, CompilationError, DEFAULT_MIN_CONFIDENCE
from .policy_schema import CompiledRule
from .rule_engine import evaluate_policy

compliance_router = APIRouter()

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates = Jinja2Templates(directory=os.path.join(_base_dir, "templates"))

@compliance_router.get("/portal")
async def compliance_portal_ui(request: Request):
    return templates.TemplateResponse("compliance.html", {"request": request})

COMPLIANCE_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../uploads/compliance")
os.makedirs(COMPLIANCE_UPLOAD_DIR, exist_ok=True)

@compliance_router.post("/process-documents")
async def process_compliance_documents(
    files: List[UploadFile] = File(default=[]),
    rulebook: UploadFile = File(None),
    objective: str = Form("General Compliance Check"),
    urls: str = Form(None),
):
    saved_file_paths = []
    rulebook_path = None
    url_list = [u.strip() for u in urls.split(",")] if urls else []
    
    if files:
        for file in files:
            if file.filename:
                unique_name = f"{uuid.uuid4()}_{file.filename}"
                file_path = os.path.join(COMPLIANCE_UPLOAD_DIR, unique_name)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_file_paths.append(file_path)
            
    if rulebook and rulebook.filename:
        rb_name = f"rulebook_{uuid.uuid4()}_{rulebook.filename}"
        rulebook_path = os.path.join(COMPLIANCE_UPLOAD_DIR, rb_name)
        with open(rulebook_path, "wb") as buffer:
            shutil.copyfileobj(rulebook.file, buffer)

    # Thin: queue the existing pipeline. Rulebook compilation + deterministic evaluation happen INSIDE process_document_batch_task (tasks.py).
    task = process_document_batch_task.delay(saved_file_paths, url_list, rulebook_path, objective)

    return JSONResponse({
        "status": "processing_started",
        "message": f"Successfully queued {len(saved_file_paths) + len(url_list)} sources for deep compliance investigation.",
        "task_id": task.id,
        "objective": objective
    })

@compliance_router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    task_result = process_document_batch_task.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }

    if task_result.state == 'PROGRESS':
        response["progress"] = task_result.info 
    elif task_result.state == 'SUCCESS':
        response["result"] = task_result.result

    return JSONResponse(jsonable_encoder(response))


# =====================================================================================
# POLICY-TO-EXECUTABLE COMPLIANCE COMPILER (added; existing endpoints above unchanged)
#   /compile-policy      LLM INTERPRETATION + deterministic validation  -> structured rules
#   /evaluate-compiled   DETERMINISTIC rule engine (no LLM)             -> compliance decision
# =====================================================================================
COMPILED_POLICY_DIR = os.path.join(COMPLIANCE_UPLOAD_DIR, "compiled_policies")
os.makedirs(COMPILED_POLICY_DIR, exist_ok=True)
_POLICY_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class CompilePolicyRequest(BaseModel):
    policy_text: str = Field(..., min_length=1)
    policy_id: Optional[str] = None
    min_confidence: float = Field(DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)
    persist: bool = True


class EvaluateCompiledRequest(BaseModel):
    policy_id: Optional[str] = None                  # load rules saved by /compile-policy ...
    rules: Optional[List[Dict[str, Any]]] = None     # ... or pass compiled rules inline (exactly one of the two)
    facts: Dict[str, Any]                            # {"<entity>": {"<field>": value | {"value":..,"unit":..}} | [records]}
    evidence: Optional[List[Any]] = None             # [{"type": "<evidence_type>", ...}] ; omit if evidence was not supplied
    evaluation_date: Optional[str] = None            # ISO date used for "now"-relative temporal rules (default: today, UTC)
    include_needs_review: bool = False               # execute NEEDS_REVIEW (ambiguous) rules too; default: they yield INDETERMINATE
    fx_rates: Optional[Dict[str, float]] = None      # {"USD->INR": 83.2}; without it, cross-currency comparison is INDETERMINATE
    strict_units: bool = False                       # True: a unit-less fact is INDETERMINATE instead of assumed in the rule's unit


def _policy_path(policy_id: str) -> str:
    if not _POLICY_ID_RE.match(policy_id or ""):
        raise HTTPException(status_code=400, detail="invalid policy_id")
    return os.path.join(COMPILED_POLICY_DIR, f"{policy_id}.json")


@compliance_router.post("/compile-policy")
async def compile_policy_endpoint(req: CompilePolicyRequest):
    try:
        result = await run_in_threadpool(compile_policy, req.policy_text, req.policy_id, req.min_confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CompilationError as e:
        raise HTTPException(status_code=502, detail=f"policy could not be compiled: {e}")
    payload = result.model_dump(mode="json")
    if req.persist and result.rules:
        with open(_policy_path(result.policy_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        payload["persisted"] = True
    else:
        payload["persisted"] = False
    return JSONResponse(jsonable_encoder(payload))


@compliance_router.get("/compiled-policy/{policy_id}")
async def get_compiled_policy(policy_id: str):
    path = _policy_path(policy_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="compiled policy not found")
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@compliance_router.post("/evaluate-compiled")
async def evaluate_compiled_endpoint(req: EvaluateCompiledRequest):
    if bool(req.policy_id) == bool(req.rules):
        raise HTTPException(status_code=400, detail="provide exactly one of 'policy_id' or 'rules'")
    rules = req.rules
    if req.policy_id:
        path = _policy_path(req.policy_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="compiled policy not found")
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f).get("rules", [])
    try:
        compiled = [CompiledRule.model_validate(r) for r in rules]
        result = evaluate_policy(compiled, req.facts, req.evidence, req.evaluation_date, req.include_needs_review, req.fx_rates, req.strict_units)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"invalid compiled rule: {e.errors()[0].get('msg')}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(jsonable_encoder(result))