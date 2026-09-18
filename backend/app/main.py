import os
import time
from dotenv import load_dotenv

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_backend_dir, ".env")
load_dotenv(_env_path)
from app.services.module1_compliance.router import compliance_router
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from typing import List, Optional
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from datetime import datetime
import shutil
import zipfile
import uuid
import httpx
import asyncio
import html
import re
from pathlib import Path

try:
    import markdown
except ImportError:
    markdown = None

import pdfplumber
import docx

from app.db.database import engine, Base, get_db
from app.models.domain import ChatSession, ChatMessage, UploadedDocument, EmployeeAssessment

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def render_content_to_html(raw_text: str) -> str:
    if markdown:
        return markdown.markdown(
            raw_text,
            extensions=["extra", "tables", "sane_lists", "nl2br"]
        )
    escaped = html.escape(raw_text)
    return escaped.replace("\n", "<br>")

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    if db.query(ChatSession).count() == 0:
        initial_session = ChatSession(title="General Knowledge & Inquiry", created_at=datetime.now(), updated_at=datetime.now())
        db.add(initial_session)
        db.commit()
    yield

app = FastAPI(
    title="OMNICheck-deep Universal AI",
    description="Universal AI Investigation & Workforce Intelligence Platform",
    lifespan=lifespan
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.include_router(compliance_router, prefix="/api/compliance", tags=["Module 1: Compliance"])

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("localhost", "127.0.0.1")
OLLAMA_LITE_MODEL = os.getenv("OLLAMA_LITE_MODEL", "qwen2.5:3b")
OLLAMA_PRO_MODEL = os.getenv("OLLAMA_PRO_MODEL", "deepseek-r1:8b")

# API KEYS SETUP & INDEPENDENT MATRIX PIPELINES
GROQ_KEYS = [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip("\"'").strip() # Failsafe default

SERPER_KEYS = [k for k in [os.getenv(f"SERPER_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
GEMINI_KEYS = [k for k in [os.getenv(f"GEMINI_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]

OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "3"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))

_groq_key_index = 0
_serper_key_index = 0
_gemini_key_index = 0

def get_next_groq_key() -> str | None:
    global _groq_key_index
    if not GROQ_KEYS: return None
    key = GROQ_KEYS[_groq_key_index % len(GROQ_KEYS)]
    _groq_key_index = (_groq_key_index + 1) % len(GROQ_KEYS)
    return key

def get_next_serper_key() -> str | None:
    global _serper_key_index
    if not SERPER_KEYS: return None
    key = SERPER_KEYS[_serper_key_index % len(SERPER_KEYS)]
    _serper_key_index = (_serper_key_index + 1) % len(SERPER_KEYS)
    return key

def get_next_gemini_key() -> str | None:
    global _gemini_key_index
    if not GEMINI_KEYS: return None
    key = GEMINI_KEYS[_gemini_key_index % len(GEMINI_KEYS)]
    _gemini_key_index = (_gemini_key_index + 1) % len(GEMINI_KEYS)
    return key

async def is_ollama_online():
    try:
        timeout = httpx.Timeout(3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            return response.status_code == 200
    except Exception:
        return False

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_silence():
    return JSONResponse({})

@app.get("/")
async def get_app_root(request: Request, session_id: str = None, db: Session = Depends(get_db)):
    if not session_id:
        new_sess = ChatSession(title="New Chat", created_at=datetime.now(), updated_at=datetime.now())
        db.add(new_sess)
        db.commit()
        db.refresh(new_sess)
        return RedirectResponse(url=f"/?session_id={new_sess.id}")

    # --- NEW: ABANDONED SESSION CLEANUP ---
    other_sessions = db.query(ChatSession).filter(ChatSession.id != session_id).all()
    cleanup_needed = False
    for s in other_sessions:
        if not db.query(ChatMessage).filter(ChatMessage.session_id == s.id).first():
            db.delete(s)
            cleanup_needed = True
    
    if cleanup_needed:
        db.commit()
    # --------------------------------------

    sessions = db.query(ChatSession).order_by(ChatSession.is_pinned.desc(), ChatSession.updated_at.desc()).all()
    active_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        
    messages = []
    if active_session:
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == active_session.id).order_by(ChatMessage.created_at.asc()).all()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sessions": sessions,
            "active_session": active_session,
            "messages": messages
        }
    )

@app.post("/api/sessions/new")
async def create_new_session(db: Session = Depends(get_db)):
    new_sess = ChatSession(title="New Chat", created_at=datetime.now(), updated_at=datetime.now())
    db.add(new_sess)
    db.commit()
    db.refresh(new_sess)
    return JSONResponse({"id": new_sess.id, "title": new_sess.title})

@app.post("/api/sessions/{session_id}/pin")
async def pin_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.is_pinned = not session.is_pinned
        db.commit()
    return {"status": "success"}

@app.post("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, title: str = Form(...), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.title = title[:35]
        db.commit()
    return {"status": "success"}

@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str, request: Request, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return templates.TemplateResponse(
        "chat_feed.html",
        {"request": request, "messages": messages, "session_id": session_id}
    )

# --- INDEPENDENT SEARCH MATRIX ENGINE ---
async def call_serper_search(query: str, client: httpx.AsyncClient, timeout=4.0) -> list:
    if not SERPER_KEYS: return []
    
    attempts = len(SERPER_KEYS)
    for _ in range(attempts):
        key = get_next_serper_key()
        if not key: continue
        try:
            headers = {"X-API-KEY": key, "Content-Type": "application/json"}
            payload = {"q": query, "num": 5}
            
            resp = await client.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=httpx.Timeout(timeout))
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("organic", []):
                    results.append({
                        "title": item.get("title", ""),
                        "body": item.get("snippet", ""),
                        "href": item.get("link", "")
                    })
                return results
            if resp.status_code in [401, 403, 429]:
                continue
        except Exception as e:
            print(f"[Serper Error] {e}")
            continue
    return []

# --- THE 3-TIER CASCADING ENGINES ---

# TIER 1: GROQ (ROUND-ROBIN MATRIX)
async def call_groq_chat(messages_payload: list, client: httpx.AsyncClient, max_tokens=1500, timeout=8.0) -> str | None:
    if not GROQ_KEYS: return None

    attempts = len(GROQ_KEYS)
    for _ in range(attempts):
        key = get_next_groq_key()
        if not key: continue
        try:
            payload = {"model": GROQ_MODEL, "messages": messages_payload, "temperature": 0.2, "max_tokens": max_tokens}
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=httpx.Timeout(timeout))
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content and str(content).strip(): return content.strip()
            else:
                print(f"[Groq API Error] Status Code: {resp.status_code} - {resp.text}")
            
            if resp.status_code in [401, 403, 429]: 
                continue
        except Exception as e:
            print(f"[Groq Exception] {e}")
            continue
    return None

# TIER 2: GEMINI (NATIVE GOOGLE GROUNDING MULTI-KEY)
async def call_gemini_chat(system_prompt: str, messages_payload: list, client: httpx.AsyncClient, timeout=8.0) -> str | None:
    if not GEMINI_KEYS: return None
    
    contents = []
    last_role = None
    for m in messages_payload:
        if m["role"] == "system": continue
        role = "user" if m["role"] == "user" else "model"
        if role == last_role:
            contents[-1]["parts"][0]["text"] += f"\n\n{m['content']}"
        else:
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        last_role = role

    # Enable native Google Search tool inside Gemini request payload
    payload = {
        "contents": contents,
        "tools": [{"googleSearch": {}}]
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    # UPDATED to Gemini 3.8 Flash
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    
    attempts = len(GEMINI_KEYS)
    for _ in range(attempts):
        key = get_next_gemini_key()
        if not key: continue
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        try:
            resp = await client.post(endpoint, headers=headers, json=payload, timeout=httpx.Timeout(timeout))
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"[Gemini API Error] Status Code: {resp.status_code} - {resp.text}")

            if resp.status_code in [401, 403, 429]: 
                continue
        except Exception as e:
            print(f"[Gemini Exception] {e}")
            continue
    return None

# CASCADING SEARCH ROUTER
async def determine_search_query(prompt: str, history: list, client: httpx.AsyncClient) -> str | None:
    current_date = datetime.now().strftime("%Y-%m-%d")
    recent = [msg for msg in history if not str(msg.content).startswith("⚠️")][-4:]
    hist_text = "\n".join([f"{m.sender.upper()}: {m.content}" for m in recent]) if recent else "No prior history."
    
    force_triggers = ["where", "price", "who", "what", "is", "iphone", "stock", "rate", "full form", "buy", "shop", "paper", "research", "news"]
    prompt_lower = prompt.lower()
    
    router_prompt = (
        f"You are an autonomous search query architect. Today is {current_date}.\n"
        "Your task is to transform the user's latest prompt and conversational context into a powerful, comprehensive web search query.\n"
        "RULES:\n"
        "1. Resolve any conversational pronouns (it, that, them, he, she) using the chat history so the search query is fully self-contained.\n"
        "2. Include specific shopping items, product categories, research paper keywords, or real-time entities mentioned.\n"
        "3. If the user is just saying 'hi', 'hello', or asking for pure code/math with no real-world entities, output exactly: NONE\n"
        "4. Output ONLY the raw search query text without quotes, markdown, or conversational filler.\n\n"
        f"--- Chat History ---\n{hist_text}\n\n"
        f"User Latest Message: {prompt}\n"
        "Optimized Search Query:"
    )

    router_payload = [{"role": "user", "content": router_prompt}]
    
    # Fast 3.0 second timeouts for the router to prevent delays
    result = await call_groq_chat(router_payload, client, max_tokens=30, timeout=10.0)
    if not result:
        result = await call_gemini_chat("", router_payload, client, timeout=10.0)
        
    if result:
        cleaned = result.replace('"', '').replace("'", "").strip()
        if "NONE" in cleaned.upper() and not any(t in prompt_lower for t in force_triggers):
            return None
        return cleaned.split("\n")[0].strip() if "NONE" not in cleaned.upper() else prompt
        
    return prompt

@app.post("/api/chat/send")
async def handle_chat_message(
    session_id: str = Form(...),
    prompt: str = Form(...),
    selected_mode: str = Form("cloud"),
    external_url: Optional[str] = Form(None),
    rulebook_text: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        session = ChatSession(id=session_id, title=prompt[:35] or "New Session", created_at=datetime.now(), updated_at=datetime.now())
        db.add(session)
        db.commit()

    if session.title in ["New Chat", "New Investigation", "General Knowledge & Inquiry"] and prompt:
        session.title = prompt[:30]
    
    session.updated_at = datetime.now()
    db.commit()

    history_messages = list(reversed(
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
        .all()
    ))

    user_msg = ChatMessage(session_id=session.id, sender="user", content=prompt, created_at=datetime.now())
    db.add(user_msg)
    db.commit()

    extracted_doc_text = ""
    if files:
        for file in files:
            if file.filename:
                file_id = str(uuid.uuid4())
                safe_filename = Path(file.filename).name
                save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
                
                with open(save_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                ext = safe_filename.lower().split('.')[-1]

                try:
                    text = ""
                    ext = safe_filename.lower().split('.')[-1]

                    # 1. Jupyter Notebook Extraction
                    if ext == "ipynb":
                        import json
                        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                            try:
                                nb = json.load(f)
                                cells = ["".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") in ["code", "markdown"]]
                                text = "\n\n".join(cells)
                            except Exception:
                                f.seek(0)
                                text = f.read()

                    # 2. PDF Documents
                    elif ext == "pdf":
                        with pdfplumber.open(save_path) as pdf:
                            text = "".join(page.extract_text() + "\n" for page in pdf.pages if page.extract_text())
                            
                    # 3. Word Documents
                    elif ext == "docx":
                        doc = docx.Document(save_path)
                        text = "\n".join([p.text for p in doc.paragraphs])
                        
                    # 4. Audio & Video Transcription (via Groq Whisper)
                    elif ext in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
                        if GROQ_KEYS and os.path.getsize(save_path) < 25 * 1024 * 1024:
                            try:
                                async with httpx.AsyncClient(timeout=60.0) as ac:
                                    with open(save_path, "rb") as media_file:
                                        resp = await ac.post(
                                            "https://api.groq.com/openai/v1/audio/transcriptions",
                                            headers={"Authorization": f"Bearer {GROQ_KEYS[0]}"},
                                            data={"model": "whisper-large-v3"},
                                            files={"file": (safe_filename, media_file)}
                                        )
                                        if resp.status_code == 200:
                                            text = f"[Transcribed Media Content]:\n{resp.json().get('text', '')}"
                                        else:
                                            text = f"[System Note: Transcription Failed - {resp.status_code}]"
                            except Exception as e:
                                text = f"[System Note: Media Error - {str(e)}]"
                        else:
                            text = "[System Note: Media file exceeds 25MB API limit or API key is missing.]"

                    # 5. Universal Catch-All (For EVERY other text, code, or data file)
                    else:
                        try:
                            with open(save_path, "r", encoding="utf-8") as f:
                                text = f.read()
                        except UnicodeDecodeError:
                            text = f"[System Note: {safe_filename} is an unreadable binary format.]"

                    if text.strip():
                        extracted_doc_text += f"\n--- Content of {safe_filename} ---\n{text[:25000]}\n"
                except Exception as e:
                    print(f"Error extracting {safe_filename}: {e}")
                    pass

                doc = UploadedDocument(
                    session_id=session.id,
                    filename=safe_filename,
                    file_type=ext,
                    file_path=save_path,
                    created_at=datetime.now()
                )
                db.add(doc)
    db.commit()

    fetched_url_text = ""
    if external_url:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                url_resp = await client.get(external_url)
                clean_text = re.sub(r'<[^>]+>', ' ', url_resp.text)
                fetched_url_text = f"\n[Data from {external_url}]:\n{clean_text[:3000]}"
        except Exception:
            pass

    if selected_mode.lower() == "local":
        target_model = OLLAMA_PRO_MODEL
        target_predict = 512
        target_ctx = 8192
        is_cloud = False
    else:
        target_model = OLLAMA_PRO_MODEL
        target_predict = 512
        target_ctx = 8192
        is_cloud = True

    web_context = ""
    search_results_data = []
    
    request_started = time.perf_counter()
    timeout_config = httpx.Timeout(30.0)

    used_engine_label = "Ollama (DeepSeek-R1 8B)" 

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        if is_cloud:
            search_query = await determine_search_query(prompt, history_messages, client)
            if search_query:
                results = await call_serper_search(search_query, client)
                if results:
                    search_results_data = results
                    web_context = f"\n=== LIVE WEB SEARCH & SHOPPING EVIDENCE FOR QUERY: '{search_query}' ===\n"
                    for idx, r in enumerate(results, 1):
                        web_context += (
                            f"{idx}. Title: {r.get('title', '')}\n"
                            f"   Snippet: {r.get('body', '')}\n"
                            f"   Link: {r.get('href', '')}\n"
                        )
                    web_context += "=== END SEARCH EVIDENCE ===\n"

        current_date_str = datetime.now().strftime("%A, %B %d, %Y")
        
        system_instruction = (
            f"You are OMNICHECK, an advanced, autonomous AI intelligence and live research engine. Today's date is {current_date_str}.\n"
            "INSTRUCTIONS:\n"
            "1. Leverage the provided [LIVE WEB SEARCH & SHOPPING EVIDENCE] to answer queries about products, shopping options, devices, research papers, recent developments, or general knowledge instantly.\n"
            "2. Maintain full conversational context across follow-up prompts smoothly and accurately.\n"
            "3. Provide direct, highly accurate, structured answers (with markdown tables, bullet points, or exact specifications where applicable).\n"
            "4. For shopping queries, include direct references to the provided product links and pricing data found in the search evidence.\n"
            "5. For research papers or academic inquiries, cite authors, titles, and exact paper URLs from the search results.\n"
            "6. For coding, mathematics, and general reasoning, use your full foundational knowledge freely. Do not refuse to answer if search evidence is absent for these topics.\n"
            "7. CRITICAL RULE: NEVER use phrases like 'Based on the provided search evidence', 'According to the web search', or 'Here are the results'. Speak naturally and directly as if you inherently know the information."
        )

        current_content = prompt
        if web_context: current_content += f"\n{web_context}"
        if extracted_doc_text: current_content += f"\n{extracted_doc_text}"
        if fetched_url_text: current_content += f"\n{fetched_url_text}"
        if rulebook_text: current_content += f"\n[Rulebook Applied]:\n{rulebook_text}"

        messages_payload = [{"role": "system", "content": system_instruction}]
        for past_msg in history_messages:
            if not str(past_msg.content).strip() or "⚠️" in str(past_msg.content):
                continue
            messages_payload.append({
                "role": "user" if past_msg.sender == "user" else "assistant",
                "content": past_msg.content
            })
        messages_payload.append({"role": "user", "content": current_content})

        reply_text = ""

        if is_cloud:
            reply_text = await call_groq_chat(messages_payload, client, max_tokens=1500, timeout=8.0)
            if reply_text:
                used_engine_label = "Groq (120B) + Serper Matrix"
            
            if not reply_text:
                reply_text = await call_gemini_chat(system_instruction, messages_payload, client, timeout=8.0)
                if reply_text:
                    used_engine_label = "Gemini Native Grounding"

        if not reply_text:
            ollama_online = await is_ollama_online()
            if ollama_online:
                try:
                    request_payload = {
                        "model": target_model,
                        "messages": messages_payload,
                        "stream": False,
                        "keep_alive": OLLAMA_KEEP_ALIVE,
                        "options": {"num_predict": target_predict, "num_ctx": target_ctx, "temperature": 0.1}
                    }
                    resp = await client.post(f"{OLLAMA_URL}/api/chat", json=request_payload, timeout=httpx.Timeout(15.0))
                    if resp.status_code == 200:
                        raw_reply = resp.json().get("message", {}).get("content", "")
                        cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_reply, flags=re.DOTALL).strip()
                        reply_text = cleaned if cleaned else raw_reply.strip()
                        if reply_text:
                            used_engine_label = f"Ollama ({target_model})"
                except Exception as e:
                    print(f"[Ollama Error] {e}")

        # Strict Regex Firewall to strip robotic introductions
        if reply_text:
            robotic_patterns = [
                r"^(Based on the provided search evidence,?|According to the web search,?|According to the search results,?|Based on the context,?|Based on the live web search,?|Here is the information:?)\s*"
            ]
            for pattern in robotic_patterns:
                reply_text = re.sub(pattern, "", reply_text, flags=re.IGNORECASE).strip()
            
            if reply_text:
                reply_text = reply_text[0].upper() + reply_text[1:]

        if not reply_text or not str(reply_text).strip():
            reply_text = "⚠️ I processed your request through all available cloud and local nodes, but could not retrieve information at this moment. Please verify your query or connectivity."
            used_engine_label = "System Error"

    # CRITICAL FIX: Only append verified sources if the system successfully generated an answer
    if search_results_data and used_engine_label != "System Error" and "⚠️" not in reply_text:
        reply_text += "\n\n**Verified Sources & Direct Links:**\n"
        seen_urls = set()
        count = 0
        for r in search_results_data:
            url = r.get('href', '')
            title = r.get('title', 'Source Link')
            if url and url not in seen_urls and count < 4:
                reply_text += f"* [{title}]({url})\n"
                seen_urls.add(url)
                count += 1

    db_ai_msg = ChatMessage(session_id=session.id, sender="assistant", content=reply_text, created_at=datetime.now())
    db.add(db_ai_msg)
    db.commit()

    elapsed = time.perf_counter() - request_started
    print(f"[Chat Execution] mode={selected_mode} elapsed={elapsed:.2f}s search={bool(web_context)}")
    formatted_html_reply = render_content_to_html(reply_text)
    time_str = datetime.now().strftime('%I:%M %p')

    return HTMLResponse(
        f"""
        <div class="flex items-start gap-4 mb-6" id="ai-bubble-{db_ai_msg.id}">
            <img src="/static/logo.png" class="w-8 h-8 rounded-full aspect-square object-cover shadow-sm mt-1 shrink-0" alt="OC">
            <div class="flex flex-col items-start flex-1 max-w-[90%]">
                <div class="ai-rendered-response text-slate-800 leading-relaxed text-[15px] font-sans bg-white border border-slate-200 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm w-full overflow-x-auto">
                    <style>
                        .ai-rendered-response h1 {{ font-size: 1.4rem; font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem; }}
                        .ai-rendered-response h2 {{ font-size: 1.25rem; font-weight: 700; margin-top: 0.875rem; margin-bottom: 0.5rem; }}
                        .ai-rendered-response h3 {{ font-size: 1.1rem; font-weight: 600; margin-top: 0.75rem; margin-bottom: 0.375rem; }}
                        .ai-rendered-response p {{ margin-bottom: 0.65rem; }}
                        .ai-rendered-response p:last-child {{ margin-bottom: 0; }}
                        .ai-rendered-response ul {{ list-style-type: disc; padding-left: 1.5rem; margin-bottom: 0.75rem; }}
                        .ai-rendered-response ol {{ list-style-type: decimal; padding-left: 1.5rem; margin-bottom: 0.75rem; }}
                        .ai-rendered-response li {{ margin-bottom: 0.25rem; }}
                        .ai-rendered-response strong {{ font-weight: 600; color: #0f172a; }}
                        .ai-rendered-response hr {{ border: 0; border-top: 1px solid #e2e8f0; margin: 1rem 0; }}
                        .ai-rendered-response table {{ width: 100%; border-collapse: collapse; margin: 0.875rem 0; font-size: 0.875rem; }}
                        .ai-rendered-response th, .ai-rendered-response td {{ border: 1px solid #cbd5e1; padding: 0.5rem 0.75rem; text-align: left; }}
                        .ai-rendered-response th {{ background-color: #f8fafc; font-weight: 600; color: #334155; }}
                        .ai-rendered-response tr:nth-child(even) {{ background-color: #fdfdfd; }}
                        .ai-rendered-response code {{ background-color: #f1f5f9; padding: 0.125rem 0.25rem; border-radius: 0.25rem; font-size: 0.875rem; font-family: monospace; }}
                        .ai-rendered-response pre {{ background-color: #0f172a; color: #f8fafc; padding: 0.75rem; border-radius: 0.5rem; overflow-x: auto; margin: 0.75rem 0; }}
                        .ai-rendered-response pre code {{ background-color: transparent; color: inherit; padding: 0; }}
                        .ai-rendered-response a {{ color: #4f46e5; text-decoration: underline; word-break: break-all; }}
                    </style>
                    {formatted_html_reply}
                </div>
                <div class="flex items-center gap-2 mt-1 pl-1">
                    <div class="text-[10px] text-slate-400 font-medium">{time_str}</div>
                    <div class="text-[9px] text-indigo-500 font-bold bg-indigo-50 border border-indigo-100 px-1.5 py-0.5 rounded flex items-center gap-1"><i class="fa-solid fa-bolt"></i> {used_engine_label}</div>
                </div>
            </div>
        </div>
        """
    )

@app.post("/api/auth/verify-hr")
async def verify_hr_access(access_key: str = Form(...)):
    HR_ACCESS_KEY = os.getenv("HR_ACCESS_KEY", "ADMIN2026")
    if access_key.strip() == HR_ACCESS_KEY:
        return JSONResponse({"status": "granted", "redirect": "/hr-portal"})
    return JSONResponse({"status": "denied", "message": "Invalid Role Authorization Key."}, status_code=403)

@app.get("/hr-portal")
async def get_hr_portal(request: Request, db: Session = Depends(get_db)):
    assessments = db.query(EmployeeAssessment).all()
    return templates.TemplateResponse(
        "hr_portal.html",
        {"request": request, "assessments": assessments}
    )