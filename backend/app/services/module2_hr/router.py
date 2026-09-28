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

@hr_router.get("/completed")
async def interview_completed_ui(request: Request):
    """Secure exit page for candidates so they never see the HR portal."""
    return templates.TemplateResponse("interview_completed.html", {"request": request})

@hr_router.websocket("/ws/interview/{employee_id}")
async def interview_websocket(websocket: WebSocket, employee_id: str, db: Session = Depends(get_db)):
    await websocket.accept()
    
    # 1. Fetch Context from DB
    assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
    candidate_name = assessment.employee_name if assessment and assessment.employee_name else "Candidate"
    
    raw_jd = assessment.jd_text if assessment and assessment.jd_text else "General Software Engineering Role"
    raw_resume = assessment.resume_text if assessment and assessment.resume_text else "No resume provided."

    # --- DYNAMIC RESUME SPLITTING & ADAPTIVE PROMPT (Pseudo-RAG) ---
    def generate_system_prompt(turn_count: int):
        resume_len = len(raw_resume)
        if resume_len > 1500:
            chunk_size = resume_len // 3
            if turn_count < 5:
                # Milestones 1-5: Education, Background, Early Experience
                safe_resume = raw_resume[:chunk_size + 400] + "\n\n... [Remaining data available in next phase]"
            elif turn_count < 10:
                # Milestones 6-10: Core Projects, Tech Skills, Tools
                start = max(0, chunk_size - 400)
                end = min(resume_len, (chunk_size * 2) + 400)
                safe_resume = "[Earlier sections omitted] ...\n\n" + raw_resume[start:end] + "\n\n... [Remaining data available in next phase]"
            else:
                # Milestones 11-15: Certifications, JD Alignment, Behavioral
                start = max(0, (chunk_size * 2) - 400)
                safe_resume = "[Earlier sections omitted] ...\n\n" + raw_resume[start:]
        else:
            safe_resume = raw_resume
            
        safe_jd = raw_jd[:2000] + "\n...(truncated to core duties)" if len(raw_jd) > 2000 else raw_jd
        first_name = candidate_name.split()[0]
        
        return f"""You are OMNICheck's elite AI Senior Interviewer conducting a spoken, spoken-word assessment of {candidate_name}.

CANDIDATE RESUME (Phase { (turn_count // 5) + 1 }):
{safe_resume}

ROLE JOB DESCRIPTION:
{safe_jd}

STRICT OUTPUT FORMATTING LAWS:
1. NEVER output markdown headers, bold labels, bullet points, asterisks, or titles. Speak in pure conversational prose.
2. Output ONLY the question and your natural conversational transition. Nothing else.
3. ASK EXACTLY ONE QUESTION AT A TIME.
4. Address the candidate occasionally as {first_name} when transitioning between topics.
5. ZERO-COUNT FOLLOW-UP RULE: If {first_name}'s answer is shallow, vague, or suspicious, ask an immediate micro-follow-up to clarify. These micro-clarifications do NOT advance your internal 15-question counter.
6. NEVER disclose scores or hiring decisions.
7. EARLY TERMINATION RULE: If the candidate explicitly asks to end, quit, or stop the interview entirely, you MUST respond with EXACTLY and ONLY this trigger word: [END_INTERVIEW]. Do not output any conversational text.

DYNAMIC ADAPTABILITY & IN-CONVERSATION OVERRIDE:
1. RESUME-BASED SKIPS: You must dynamically tailor these 15 topics. If a candidate lacks a specific section (e.g., zero work experience, no certifications), DO NOT ask about it. Replace it with a deeper technical deep-dive.
2. IN-CONVERSATION OVERRIDE: If the candidate explicitly asks to skip a question, says they don't know, or requests to move on (e.g., "I want to skip projects"), YOU MUST OBEY IMMEDIATELY. Do NOT ask follow-up questions about the skipped topic. Instantly pivot to the next milestone category or a completely different JD-based scenario. You must still reach exactly 15 total questions.

YOUR ADAPTIVE 15-QUESTION ROADMAP (Track silently):
1. Work Experience (CONDITIONAL: If no formal experience exists in the resume, SKIP to Education or ask about their general engineering passion).
2. Formal education and foundational computer science theory.
3. Flagship resume project: Deep dive into their specific architecture and bottlenecks.
4. Secondary project: Design choices and trade-offs.
5. Primary technical skills: Scenario-based test of their core programming languages mentioned in the resume.
6. Analytical problem solving: A live distributed systems or concurrency failure puzzle.
7. The Honeypot Trap: Present a completely synthetic, fake algorithm (e.g., "The Voss-Kessler Cascade Protocol"). Evaluate whether they bluff or honestly state they have not encountered it.
8. Supporting tooling: Infrastructure, databases, caching, or Docker mentioned in their stack.
9. Certifications (CONDITIONAL: If no certifications/courses are listed, SKIP and ask a secondary technical skill question).
10. Achievements/Research (CONDITIONAL: If no publications, patents, or hackathons are listed, SKIP and ask an extra problem-solving question).
11. Job description alignment: Direct comparison of their skills to the JD requirements.
12. Day-one Execution: A realistic production challenge taken directly from the JD.
13. Booming industry trends relevant to the JD (e.g., LLMs, edge computing).
14. Behavioral dynamics: Handling disagreements or tight deadline prioritization.
15. Cultural expectations, engineering standards, and mutual career fit.

CONCLUDING:
After the candidate answers question 15, politely conclude:
"Thank you for your time, {first_name}. This concludes our interview. You may now close the window."
"""

    # Initialize the chat safely with Turn 0 (This fixes the UnboundLocalError)
    chat_history = [{"role": "system", "content": generate_system_prompt(0)}]
    
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
                raw_user_text = data.get("content")
                
                active_key = get_groq_key()
                if not active_key:
                    await websocket.send_json({"type": "error", "content": "Groq API key not configured."})
                    continue

                # --- STEP 4.1: GROQ GRAMMAR & JARGON INTERCEPTOR (Fast 8B Model) ---
                cleaned_text = raw_user_text
                try:
                    # Provide full context to grammar interceptor so it catches dynamic terms across the whole resume
                    interceptor_context = f"Candidate Resume Entities:\n{raw_resume[:1500]}\n\nTarget JD:\n{raw_jd[:800]}"
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        clean_resp = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {active_key}", "Content-Type": "application/json"},
                            json={
                                "model": "llama-3.1-8b-instant",
                                "messages": [
                                    {
                                        "role": "system", 
                                        "content": (
                                            f"You are an intelligent Indian English audio transcript post-processor. "
                                            f"The raw transcript is often phonetically garbled due to speech-to-text misinterpretations. "
                                            f"DYNAMIC PHONETIC MATCHING: Cross-reference the garbled text against the Candidate's Resume and Job Description provided here:\n{interceptor_context}\n\n"
                                            f"Do NOT rely on predefined rules. Listen for phonetic similarities. If a garbled phrase sounds like a specific project name, company, framework, or skill from the context (e.g., if it sounds like a tech stack or school mentioned in the resume), correct it to match the exact spelling in the context. "
                                            f"Preserve the candidate's meaning perfectly. Output ONLY the cleaned transcript with no quotes or preamble."
                                        )
                                    },
                                    {"role": "user", "content": raw_user_text}
                                ],
                                "temperature": 0.0,
                                "max_tokens": 300
                            }
                        )
                        if clean_resp.status_code == 200:
                            cleaned_candidate_text = clean_resp.json()["choices"][0]["message"]["content"].strip().strip('"')
                            if cleaned_candidate_text:
                                cleaned_text = cleaned_candidate_text
                except Exception as e:
                    print(f"[Grammar Intercept Failed] Falling back to raw text: {e}")
                
                # Send the CLEANED text back to the UI so the user sees the fixed technical terms
                await websocket.send_json({"type": "user_text", "content": cleaned_text})

                # Append the CLEANED text to the chat history for the main Interviewer
                chat_history.append({"role": "user", "content": cleaned_text})

                # --- DYNAMIC CONTEXT REFRESH ---
                # Recalculate which phase of the resume we should feed the AI for the next question
                current_turn = len(chat_history) // 2
                chat_history[0] = {"role": "system", "content": generate_system_prompt(current_turn)}

                # History Slicer: Always preserve the system prompt (index 0) + only the last 6 messages
                # This drops token payload by over 60%, preventing Groq 429 TPM limits
                if len(chat_history) > 7:
                    payload_messages = [chat_history[0]] + chat_history[-6:]
                else:
                    payload_messages = chat_history

                # Route through Groq with key rotation on rate limits
                response_success = False
                for api_key in GROQ_KEYS:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": GROQ_MODEL,
                                "messages": payload_messages,
                                "temperature": 0.3,
                                "max_tokens": 350
                            }
                        )
                        if resp.status_code == 200:
                            ai_reply = resp.json()["choices"][0]["message"]["content"]
                            chat_history.append({"role": "assistant", "content": ai_reply})
                            await websocket.send_json({"type": "text", "content": ai_reply})
                            response_success = True
                            break
                        elif resp.status_code == 429:
                            print(f"[Module 2 Router] Key rate-limited (429). Attempting fallback key...")
                            continue
                        else:
                            print(f"[Module 2 Router Error] Groq HTTP {resp.status_code}: {resp.text}")
                            break

                if not response_success:
                    await websocket.send_json({"type": "error", "content": "The system is experiencing high traffic. Please retry in a moment."})

    except WebSocketDisconnect:
        print(f"[Module 2] Candidate {employee_id} disconnected. Handing off to Celery Assessment Worker.")
        # Isolate the import to prevent circular dependencies
        from app.services.module2_hr.tasks import evaluate_interview_task
        # .delay() pushes it to the background Redis queue instantly
        evaluate_interview_task.delay(employee_id, chat_history)