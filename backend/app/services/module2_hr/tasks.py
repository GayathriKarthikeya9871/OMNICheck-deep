import os
import json
import httpx
from celery import Celery
from dotenv import load_dotenv
from pathlib import Path

# Bulletproof Absolute Path to .env (Navigates up from tasks.py -> module2_hr -> services -> app -> backend)
_current_file = Path(__file__).resolve()
_backend_dir = _current_file.parent.parent.parent.parent
_env_path = _backend_dir / ".env"
load_dotenv(dotenv_path=_env_path)

from app.db.database import SessionLocal
from app.models.domain import EmployeeAssessment, ActivityLog

# Celery Setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("hr_tasks", broker=REDIS_URL)

@celery_app.task
def evaluate_interview_task(employee_id: str, chat_history: list):
    """
    Background worker that aggregates transcripts & proctoring logs,
    and uses Groq to generate a definitive HR employment recommendation.
    """
    db = SessionLocal()
    try:
        # Load keys INSIDE the task to ensure they are read after load_dotenv executes in the worker process
        groq_keys = [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip("\"'").strip()
        active_key = groq_keys[0] if groq_keys else None

        if not active_key:
            print(f"[Module 2 Worker] Fatal: No Groq API key available. Checked path: {_env_path}")
            return
        # 1. Fetch Assessment Data (JD & Resume)
        assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
        jd_text = assessment.jd_text if assessment and assessment.jd_text else "No JD provided."
        resume_text = assessment.resume_text if assessment and assessment.resume_text else "No Resume provided."

        # 2. Aggregate Proctoring & Integrity Events
        logs = db.query(ActivityLog).filter(ActivityLog.employee_id == employee_id).all()
        if logs:
            proctor_events = [f"[{log.timestamp.strftime('%H:%M:%S')}] {log.event_category}: {log.description}" for log in logs]
        else:
            proctor_events = ["No integrity violations detected. Completely clean session."]
            
        # 3. Format the Raw Interview Transcript
        transcript = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history if msg['role'] != 'system'])
        
        # 4. Construct the Strict JSON AI Prompt
        system_prompt = """You are an elite HR Assessor & Auditor. Review the interview transcript, JD, Resume, and proctoring logs.
        Evaluate the candidate strictly against the Job Description based on their actual answers.
        
        CRITICAL RULES:
        1. PROPORTIONAL SCORING: If the interview was aborted early, base your score ONLY on the questions the candidate actually answered. If they answered zero questions, the score is 0. If they answered 3 questions brilliantly and then the interview ended, grade their competency based on those 3 answers.
        2. Do not give pity points. If the answer is irrelevant or missing, score it 0.
        3. Evaluate every single question asked in the qa_report.

        You MUST return ONLY valid JSON in this exact structure:
        {
            "overall_suitability": 85.5,
            "proctoring_status": "Clean",
            "competency_scores": {"Communication": 90, "Technical": 80, "Problem Solving": 85, "Integrity": 100},
            "hr_decision": "Recommend Hire",
            "qa_report": [
                {
                    "question": "The interviewer's question",
                    "candidate_answer": "Exact summary of what the candidate said",
                    "ideal_answer": "What the perfect, correct answer should have been",
                    "score_out_of_10": 8
                }
            ]
        }
        Do not include markdown blocks. Output just the raw JSON object.
        """
        
        user_prompt = f"--- ROLE JD ---\n{jd_text}\n\n--- CANDIDATE RESUME ---\n{resume_text}\n\n--- PROCTORING LOGS ---\n{chr(10).join(proctor_events)}\n\n--- INTERVIEW TRANSCRIPT ---\n{transcript}"

        print(f"[Module 2 Worker] Commencing evaluation for candidate {employee_id}...")

        # 4. Synchronous HTTP request to Groq
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {active_key}", "Content-Type": "application/json"},
                json={
                    "model": groq_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
            )
            
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                result = json.loads(content)
                
                # 5. Save Final Decision to Existing DB Record
                assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
                if assessment:
                    assessment.status = "Completed"
                    assessment.overall_suitability = result.get("overall_suitability", 0.0)
                    assessment.competency_scores = result.get("competency_scores", {})
                    assessment.proctoring_status = result.get("proctoring_status", "Warning")
                    assessment.hr_decision = result.get("hr_decision", "Flag for Review")
                    assessment.qa_report = result.get("qa_report", [])
                    db.commit()
                print(f"[Module 2 Worker] Successfully finalized grading for {employee_id}.")
            else:
                print(f"[Module 2 Worker] Groq API Error: {resp.text}")
                
    except Exception as e:
        print(f"[Module 2 Worker] Assessment Task Error: {e}")
    finally:
        db.close()