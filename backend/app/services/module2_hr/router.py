import os
import io
import PyPDF2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Request, Form, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
import httpx

from app.db.database import get_db
from app.models.domain import Employee, ActivityLog, EmployeeAssessment, JobRequirement

hr_router = APIRouter()

# Setup Templates Directory
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates = Jinja2Templates(directory=os.path.join(_base_dir, "templates"))

# Dynamic Groq Key Fallback & Model Assignment
GROQ_KEYS = [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip("\"'").strip()

def get_groq_key():
    return GROQ_KEYS[0] if GROQ_KEYS else None

async def extract_file_text(file: UploadFile):
    """Helper to extract text from TXT or PDF uploads."""
    if not file or not file.filename:
        return ""
    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            print(f"[PDF Extraction Error] {e}")
            return "Error extracting PDF text."
    return content.decode("utf-8", errors="ignore")

@hr_router.post("/create_jd")
async def create_jd(
    title: str = Form(...), 
    jd_text: str = Form(""), 
    jd_file: UploadFile = File(None), 
    db: Session = Depends(get_db)
):
    # Extract text from file if provided, otherwise fallback to the pasted text
    file_text = await extract_file_text(jd_file)
    final_text = file_text.strip() if file_text.strip() else jd_text.strip()
    
    new_jd = JobRequirement(title=title, jd_text=final_text)
    db.add(new_jd)
    db.commit()
    return RedirectResponse(url="/api/hr/portal", status_code=303)

@hr_router.post("/enroll")
async def enroll_candidate(
    employee_name: str = Form(...), 
    jd_id: str = Form(...), 
    db: Session = Depends(get_db)
):
    job = db.query(JobRequirement).filter(JobRequirement.id == jd_id).first()
    assessment = EmployeeAssessment(
        employee_name=employee_name,
        job_role=job.title if job else "Unknown",
        department="Engineering",
        jd_text=job.jd_text if job else "",
        status="Pending Resume"
    )
    db.add(assessment)
    db.commit()
    return RedirectResponse(url="/api/hr/portal", status_code=303)

@hr_router.post("/submit_resume/{employee_id}")
async def submit_resume(
    employee_id: str, 
    resume_text: str = Form(""), 
    resume_file: UploadFile = File(None), 
    db: Session = Depends(get_db)
):
    assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
    if assessment:
        # Extract text from file if provided, otherwise fallback to the pasted text
        file_text = await extract_file_text(resume_file)
        final_text = file_text.strip() if file_text.strip() else resume_text.strip()
        
        assessment.resume_text = final_text
        assessment.status = "Ready for Interview"
        db.commit()
    return RedirectResponse(url=f"/api/hr/interview/{employee_id}", status_code=303)

@hr_router.get("/portal")
async def hr_dashboard(request: Request, db: Session = Depends(get_db)):
    assessments = db.query(EmployeeAssessment).order_by(EmployeeAssessment.created_at.desc()).all()
    jds = db.query(JobRequirement).order_by(JobRequirement.created_at.desc()).all()
    return templates.TemplateResponse("hr_portal.html", {"request": request, "assessments": assessments, "jds": jds})

@hr_router.get("/report/{employee_id}")
async def view_qa_report(request: Request, employee_id: str, db: Session = Depends(get_db)):
    assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
    return templates.TemplateResponse("qa_report.html", {"request": request, "assessment": assessment})

@hr_router.get("/interview/{employee_id}")
async def candidate_interview_ui(request: Request, employee_id: str, db: Session = Depends(get_db)):
    assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
    return templates.TemplateResponse("interview_room.html", {"request": request, "employee_id": employee_id, "assessment": assessment})

@hr_router.websocket("/ws/interview/{employee_id}")
async def interview_websocket(websocket: WebSocket, employee_id: str, db: Session = Depends(get_db)):
    await websocket.accept()
    
    # 1. Fetch Context from DB
    assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
    jd = assessment.jd_text if assessment and assessment.jd_text else "General Software Engineering Role"
    resume = assessment.resume_text if assessment and assessment.resume_text else "No resume provided. Ask general situational questions based on the JD."
    
    # 2. Strict 15-Question Interview Prompt
    sys_prompt = f"""You are OMNICheck's elite AI HR Interviewer. You will conduct a rigorous 15-question interview based on the candidate's Resume and the Job Description.

    CRITICAL RULES:
    1. DO NOT use markdown headers or announce your phases (e.g., NEVER say "**Phase 1**"). Transition naturally like a real human.
    2. You must ask EXACTLY 15 questions in total. (Roughly 5 on Resume, 5 technical JD questions, 5 behavioral).
    3. Ask ONLY ONE specific question at a time. Wait for the answer.
    4. NEVER disclose the candidate's performance, score, or hiring status during the chat. If they ask how they are doing, politely state that HR will review the final transcript.
    5. Once you have asked your 15th question and received the answer, thank the candidate and state: "This concludes our interview. You may now close the window." Do not ask any further questions.
    
    CANDIDATE RESUME:\n{resume}\n
    ROLE JD:\n{jd}
    """
    
    chat_history = [{"role": "system", "content": sys_prompt}]
    
    try:
        # 2. Start the Interview
        welcome_msg = "Hello. I am your AI interviewer. Please ensure your camera is clearly framing your face and your microphone is active. Shall we begin?"
        chat_history.append({"role": "assistant", "content": welcome_msg})
        await websocket.send_json({"type": "text", "content": welcome_msg})

        while True:
            data = await websocket.receive_json()
            
            # 3. Handle Silent Proctoring Alerts (Edge Computer Vision triggers this)
            if data.get("type") == "proctor_alert":
                log = ActivityLog(
                    employee_id=employee_id,
                    event_category="PROCTOR_FLAG",
                    description=data.get("content", "Suspicious activity detected"),
                    risk_weight=5,
                    timestamp=datetime.utcnow()
                )
                db.add(log)
                db.commit()
                continue # Do not interrupt the interview loop

            # 4. Handle Candidate Speech/Text Responses
            if data.get("type") == "candidate_response":
                user_text = data.get("content")
                chat_history.append({"role": "user", "content": user_text})
                
                active_key = get_groq_key()
                if not active_key:
                    await websocket.send_json({"type": "error", "content": "Groq API key not configured."})
                    continue

                # Route through Groq for instant cognitive follow-ups
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {active_key}", "Content-Type": "application/json"},
                        json={
                            "model": GROQ_MODEL,
                            "messages": chat_history,
                            "temperature": 0.3,
                            "max_tokens": 500
                        }
                    )
                    if resp.status_code == 200:
                        ai_reply = resp.json()["choices"][0]["message"]["content"]
                        chat_history.append({"role": "assistant", "content": ai_reply})
                        await websocket.send_json({"type": "text", "content": ai_reply})
                    else:
                        print(f"[Module 2 Router Error] Groq HTTP {resp.status_code}: {resp.text}")
                        await websocket.send_json({"type": "error", "content": "Network timeout or rate limit. Retrying your prompt..."})

    except WebSocketDisconnect:
        print(f"[Module 2] Candidate {employee_id} disconnected. Handing off to Celery Assessment Worker.")
        # Isolate the import to prevent circular dependencies
        from app.services.module2_hr.tasks import evaluate_interview_task
        # .delay() pushes it to the background Redis queue instantly
        evaluate_interview_task.delay(employee_id, chat_history)