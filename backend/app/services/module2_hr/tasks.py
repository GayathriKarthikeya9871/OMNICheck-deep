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
        system_prompt = """You are an elite HR Assessor & Auditor. Review the interview transcript, Job Description, Candidate Resume, and proctoring logs.
Evaluate the candidate strictly against the role requirements across the 15-milestone framework:
1. Work Experience, 2. Education & Academics, 3. Flagship Project, 4. Secondary Project, 5. Primary Tech Skills, 
6. Problem Solving, 7. The Honeypot (Fake Algorithm), 8. Supporting Tools, 9. Certifications, 10. Achievements & Research, 
11. JD Alignment/Gaps, 12. JD Execution Scenario, 13. Emerging Tech, 14. Behavioral Dynamics, 15. Role Fit & Expectations.

CRITICAL EVALUATION RULES:
1. PROPORTIONAL SCORING: If the interview concluded early or was terminated due to security violations, base competency scores ONLY on the questions actually completed. If zero questions were answered, total score is 0.
2. HONEYPOT TRAP EVALUATION (Milestone 7):
   - Did the candidate honestly acknowledge they have not heard of the fake algorithm (High Integrity)?
   - Or did they bluff, pretend to know it, or invent an answer? If they BLUFFED, you MUST apply a -40 penalty to their base Integrity score and a -30 penalty to their Problem Solving score. (The Python backend will separately calculate browser proctoring penalties).
3. COMPETENCY SCORING:
   - Technical (0-100): Depth in primary skills, projects, and architecture.
   - Problem Solving (0-100): Logic, debugging approach, and adaptability.
   - Communication (0-100): Clarity, conciseness, and articulation.
   - Integrity (0-100): Intellectual honesty on the honeypot + proctoring compliance.
4. QA REPORT: Every question asked (including micro follow-ups) must appear with the candidate's actual response summary, the ideal expected response, and a score out of 10.

You MUST return ONLY valid JSON in this exact structure:
{
    "overall_suitability": 82.5,
    "proctoring_status": "Clean",
    "competency_scores": {
        "Communication": 85,
        "Technical": 80,
        "Problem Solving": 85,
        "Integrity": 90
    },
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
Do not include markdown blocks (no ```json). Output just the raw JSON object.
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
                
                # --- MATHEMATICAL VERIFICATION & ENFORCEMENT ---
                qa_report = result.get("qa_report", [])
                
                # 1. Exact mathematical average of answered questions
                if qa_report:
                    total_points = sum([float(q.get("score_out_of_10", 0.0)) for q in qa_report])
                    max_possible = len(qa_report) * 10.0
                    calculated_qa_score = round((total_points / max_possible) * 100.0, 1)
                else:
                    calculated_qa_score = 0.0

                # 2. Cumulative Integrity Evaluation based on proctor logs
                base_integrity = 100
                proctoring_status = "Clean"
                
                for event in proctor_events:
                    event_lower = event.lower()
                    if "fatal strike 3" in event_lower or "auto-terminated" in event_lower:
                        base_integrity = 0
                        proctoring_status = "Terminated"
                        break
                    elif "exited full-screen" in event_lower:
                        base_integrity -= 10
                    elif "switched tabs" in event_lower or "lost focus" in event_lower:
                        base_integrity -= 15
                    elif "multiple" in event_lower and "faces" in event_lower:
                        base_integrity -= 20
                    elif "no face detected" in event_lower or "left camera" in event_lower:
                        base_integrity -= 20
                    elif "clipboard" in event_lower or "context menu" in event_lower:
                        base_integrity -= 25
                
                # Clamp score between 0 and 100
                base_integrity = max(0, min(100, base_integrity))
                
                if base_integrity < 100 and proctoring_status != "Terminated":
                    proctoring_status = "Warning/Violations Detected"
                    
                # Incorporate AI's Honeypot penalty if it flagged integrity in its JSON
                ai_integrity = float(result.get("competency_scores", {}).get("Integrity", 100))
                integrity_score = min(base_integrity, int(ai_integrity))

                # 3. Pull competency scores and clamp them to reality
                raw_competencies = result.get("competency_scores", {})
                
                # Competency scores cannot exceed the QA performance by more than 15 points
                communication = min(float(raw_competencies.get("Communication", calculated_qa_score)), calculated_qa_score + 15)
                technical = min(float(raw_competencies.get("Technical", calculated_qa_score)), calculated_qa_score + 10)
                problem_solving = min(float(raw_competencies.get("Problem Solving", calculated_qa_score)), calculated_qa_score + 10)
                
                # Final Mathematical Suitability (70% Question Competency + 30% Integrity)
                final_overall_score = round((calculated_qa_score * 0.70) + (integrity_score * 0.30), 1)

                # Automated Decision Enforcement
                if final_overall_score < 50.0 or proctoring_status == "Violation":
                    hr_decision = "Do Not Hire"
                elif final_overall_score < 70.0:
                    hr_decision = "Consider with Reservations"
                else:
                    hr_decision = "Recommend Hire"

                # 5. Save Verified Decision to DB
                assessment = db.query(EmployeeAssessment).filter(EmployeeAssessment.id == employee_id).first()
                if assessment:
                    assessment.status = "Completed"
                    assessment.overall_suitability = final_overall_score
                    assessment.competency_scores = {
                        "Communication": round(communication, 1),
                        "Technical": round(technical, 1),
                        "Problem Solving": round(problem_solving, 1),
                        "Integrity": integrity_score
                    }
                    assessment.proctoring_status = proctoring_status
                    assessment.hr_decision = hr_decision
                    assessment.qa_report = qa_report
                    db.commit()
                print(f"[Module 2 Worker] Successfully finalized grading for {employee_id}. Final Math Score: {final_overall_score}%")
            else:
                print(f"[Module 2 Worker] Groq API Error: {resp.text}")
                
    except Exception as e:
        print(f"[Module 2 Worker] Assessment Task Error: {e}")
    finally:
        db.close()