import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import List
from fastapi.templating import Jinja2Templates
from .tasks import process_document_batch_task

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
    urls: str = Form(None)
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
        
    return JSONResponse(response)