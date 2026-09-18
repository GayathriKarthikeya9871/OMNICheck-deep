import os
import requests
import tempfile
import re
import zipfile
import io
import shutil
import base64
from celery import Celery
from dotenv import load_dotenv

try:
    from bs4 import BeautifulSoup
except ImportError:
    pass

try:
    import pdfplumber
    import docx
except ImportError:
    pass

try:
    import pymupdf as fitz 
except ImportError:
    try:
        import fitz
    except ImportError:
        pass

try:
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    pass

try:
    import phonenumbers
except ImportError:
    pass

try:
    import git
except ImportError:
    pass

_backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(_backend_dir, ".env"))

celery_app = Celery(
    "compliance_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

GROQ_KEYS = [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip("\"'").strip()
GEMINI_KEYS = [k for k in [os.getenv(f"GEMINI_API_KEY_{i}", "").strip("\"'").strip() for i in range(1, 4)] if k]

VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5], [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7], [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3], [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]

VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4], [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7], [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8]
]

def validate_verhoeff(num_str):
    try:
        c = 0
        reversed_arr = [int(x) for x in reversed(str(num_str))]
        for i, n in enumerate(reversed_arr):
            c = VERHOEFF_D[c][VERHOEFF_P[i % 8][n]]
        return c == 0
    except:
        return False

def generate_system_validation_report(text: str) -> str:
    report_lines = []
    try:
        if 'phonenumbers' in globals():
            for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
                is_valid = phonenumbers.is_valid_number(match.number)
                status = "PASSED (Carrier/Format Valid)" if is_valid else "FAILED (Fake/Invalid Number)"
                report_lines.append(f"Phone {match.raw_string}: {status}")
    except Exception:
        pass

    id_matches = re.finditer(r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b', text)
    for match in id_matches:
        id_str = match.group().replace(" ", "")
        is_valid = validate_verhoeff(id_str)
        status = "PASSED (Mathematical Checksum Valid)" if is_valid else "FAILED (Fake/Invalid Checksum)"
        report_lines.append(f"Gov ID {id_str}: {status}")

    if not report_lines:
        return ""
    return "\n--- [SYSTEM ALGORITHMIC PRE-CHECK REPORT] ---\n" + "\n".join(report_lines) + "\n--------------------------------------------\n\n"

def analyze_image_with_vision(image_path: str) -> str:
    if not GEMINI_KEYS:
        return ""
    try:
        ext = image_path.split('.')[-1].lower()
        mime_type = f"image/{'png' if ext == 'png' else 'jpeg'}"
        
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
            
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Analyze this image for a compliance audit. 1) Is there a human face? 2) Is it blurry, poorly lit, cropped, or modified? 3) Does it meet standard passport/ID photo criteria? 4) Describe any text or objects visible. Be concise and factual."},
                    {"inline_data": {"mime_type": mime_type, "data": img_data}}
                ]
            }]
        }
        
        endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
        
        for key in GEMINI_KEYS:
            headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=15.0)
            if resp.status_code == 200:
                vision_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return f"\n--- [SYSTEM VISION PRE-CHECK REPORT] ---\n{vision_text.strip()}\n----------------------------------------\n"
    except Exception:
        pass
    return ""

def download_url_to_temp(url: str) -> tuple:
    try:
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "?dl=1")
        elif "drive.google.com" in url:
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
            if match:
                file_id = match.group(1)
                url = f"https://drive.google.com/uc?export=download&id={file_id}"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code in [401, 403]:
            return "", "HTTP 403: Access Denied. The link is private or requires authentication."
        elif resp.status_code == 404:
            return "", "HTTP 404: Not Found. The link is dead or the file was removed."
            
        resp.raise_for_status()
        
        content_type = resp.headers.get('Content-Type', '').lower()
        ext = url.split('?')[0].split('.')[-1].lower()
        
        if "archive" in url and "zip" in url:
            ext = "zip"
        elif 'image/jpeg' in content_type:
            ext = 'jpg'
        elif 'image/png' in content_type:
            ext = 'png'
        elif 'text/html' in content_type:
            ext = 'html'
        elif 'application/pdf' in content_type:
            ext = 'pdf'
        elif 'application/zip' in content_type:
            ext = 'zip'
        elif ext not in ['pdf', 'txt', 'csv', 'docx', 'png', 'jpg', 'jpeg', 'zip', 'xlsx', 'html']:
            ext = 'txt'
            
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tmp.write(resp.content)
        tmp.close()
        return tmp.name, "Success"
    except requests.exceptions.RequestException as e:
        return "", f"Network Error: {str(e)}"
    except Exception as e:
        return "", f"System Error: {str(e)}"

def extract_text_from_file(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
        
    filename = os.path.basename(file_path).lower()
    ext = filename.split('.')[-1] if '.' in filename else ""
    text = ""
    
    try:
        if ext in ["txt", "csv", "md", "py", "js", "html", "json", "env", "yml", "c", "cpp", "java", "sh"] or ext == "":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()
                
            if ext == "html" or ("<html" in raw_content.lower() and "<body" in raw_content.lower()):
                if 'BeautifulSoup' in globals():
                    soup = BeautifulSoup(raw_content, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                else:
                    text = re.sub(r'<[^>]+>', ' ', raw_content)
            else:
                text = raw_content

        elif ext == "pdf":
            if 'pdfplumber' in globals():
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            
            if ('fitz' in globals() or 'pymupdf' in globals()) and 'pytesseract' in globals():
                try:
                    pdf_file = fitz.open(file_path)
                    for page_index in range(len(pdf_file)):
                        page = pdf_file[page_index]
                        for img in page.get_images(full=True):
                            xref = img[0]
                            base_image = pdf_file.extract_image(xref)
                            img_obj = Image.open(io.BytesIO(base_image["image"]))
                            ocr_text = pytesseract.image_to_string(img_obj)
                            if ocr_text.strip():
                                text += f"\n[OCR from PDF Image Page {page_index+1}]:\n{ocr_text.strip()}\n"
                except Exception:
                    pass

        elif ext in ["docx", "xlsx"]:
            if ext == "docx" and 'docx' in globals():
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                
            if 'pytesseract' in globals():
                try:
                    with zipfile.ZipFile(file_path, 'r') as z:
                        for item in z.namelist():
                            if item.startswith('word/media/') or item.startswith('xl/media/'):
                                with z.open(item) as img_file:
                                    img_obj = Image.open(io.BytesIO(img_file.read()))
                                    ocr_text = pytesseract.image_to_string(img_obj)
                                    if ocr_text.strip():
                                        text += f"\n[OCR from Embedded {ext.upper()} Image - {item}]:\n{ocr_text.strip()}\n"
                except Exception:
                    pass

        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "img"]:
            if 'pytesseract' in globals():
                try:
                    text = pytesseract.image_to_string(Image.open(file_path))
                except Exception:
                    pass
            vision_report = analyze_image_with_vision(file_path)
            if vision_report:
                text = vision_report + "\n" + text
            
    except Exception as e:
        return f"[Extraction Error] {e}"
    
    return text

def analyze_compliance_with_matrix(combined_documents_text: str, rulebook_text: str, objective: str) -> str:
    system_prompt = (
        "You are OMNICheck, a high-end, zero-bias Compliance and Investigation AI. "
        "You have been provided with one or more documents, raw source code, or images, and potentially SYSTEM ALGORITHMIC and VISION PRE-CHECK REPORTS. "
        "Your absolute priority is MULTI-DOCUMENT SYNTHESIS and DATA MAPPING. \n\n"
        "RULES:\n"
        "1. If a System Pre-Check explicitly marks an ID/Phone as FAILED, or a URL as ACCESS DENIED/403/404, you MUST flag that entity immediately.\n"
        "2. If a Vision Pre-Check report describes an image as blurry, non-compliant, low-light, or lacking a face when an ID or portrait photo is expected, flag it as a non-compliance issue or policy violation.\n"
        "3. You must cross-reference dates, monetary amounts, and claims across all files. Calculate exact variances.\n"
        "4. DO NOT hide facts. Output the absolute truth.\n\n"
        f"INVESTIGATION OBJECTIVE: {objective}\n\n"
        "RULEBOOK / TEMPLATE LAWS TO APPLY:\n"
        f"{rulebook_text if rulebook_text else 'No custom rulebook provided. Apply standard global compliance logic.'}\n\n"
        "OUTPUT FORMAT: Provide a structured Markdown report including:\n"
        "1. Executive Synthesis (including Access Denied logs and Vision flags)\n"
        "2. Data Mapping / Discrepancy Table\n"
        "3. Finding (Pass/Fail/Requires Review) & Severity Score\n"
        "4. Explicit Evidence (quote logs verbatim)\n"
        "5. Actionable Next Steps"
    )

    if GROQ_KEYS:
        for key in GROQ_KEYS:
            try:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"DOCUMENTS TO INVESTIGATE:\n{combined_documents_text[:80000]}"} 
                    ],
                    "temperature": 0.1
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=45.0)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                continue

    if GEMINI_KEYS:
        for key in GEMINI_KEYS:
            try:
                endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
                headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": f"DOCUMENTS TO INVESTIGATE:\n{combined_documents_text[:80000]}"}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]}
                }
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=45.0)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                continue
            
    return "Investigation Failed: Both Groq and Gemini AI routing matrices are exhausted, blocked, or timed out."

@celery_app.task(bind=True)
def process_document_batch_task(self, file_paths: list, url_list: list, rulebook_path: str, objective: str):
    rulebook_text = ""
    if rulebook_path:
        self.update_state(state='PROGRESS', meta={'status': 'Extracting Rulebook/Template...'})
        rulebook_text = extract_text_from_file(rulebook_path)

    files_to_process = []
    temp_dirs_to_cleanup = []
    access_logs = ""

    for path in file_paths:
        if path.lower().endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            temp_dirs_to_cleanup.append(temp_dir)
            try:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        files_to_process.append(os.path.join(root, f))
            except Exception:
                pass
        else:
            files_to_process.append(path)

    for url in url_list:
        url = url.strip()
        if not url: continue
        
        if "github.com" in url:
            clean_url = url.split('.git')[0]
            zip_url_main = f"{clean_url}/archive/refs/heads/main.zip"
            downloaded_path, status = download_url_to_temp(zip_url_main)
            
            if not downloaded_path or os.path.getsize(downloaded_path) < 1000:
                zip_url_master = f"{clean_url}/archive/refs/heads/master.zip"
                downloaded_path, status = download_url_to_temp(zip_url_master)
                
            if downloaded_path and downloaded_path.lower().endswith('.zip'):
                temp_dir = tempfile.mkdtemp()
                temp_dirs_to_cleanup.append(temp_dir)
                try:
                    with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    for root, _, files in os.walk(temp_dir):
                        for f in files:
                            files_to_process.append(os.path.join(root, f))
                except Exception:
                    pass
            else:
                access_logs += f"\n[SYSTEM LOG] Repo URL {url} FAILED: {status}\n"
        else:
            downloaded_path, status = download_url_to_temp(url)
            if downloaded_path:
                if downloaded_path.lower().endswith('.zip'):
                    temp_dir = tempfile.mkdtemp()
                    temp_dirs_to_cleanup.append(temp_dir)
                    try:
                        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        for root, _, files in os.walk(temp_dir):
                            for f in files:
                                files_to_process.append(os.path.join(root, f))
                    except Exception:
                        pass
                else:
                    files_to_process.append(downloaded_path)
            else:
                access_logs += f"\n[SYSTEM LOG] Document URL {url} FAILED: {status}\n"

    total_files = len(files_to_process)
    combined_text = access_logs
    current_idx = 0

    for file_path in files_to_process:
        current_idx += 1
        file_name = os.path.basename(file_path)
        if len(file_name) > 36 and '-' in file_name[:36]:
            file_name = file_name.split('_', 1)[-1]
            
        self.update_state(state='PROGRESS', meta={'current': current_idx, 'total': total_files, 'file': f"Extracting {file_name}..."})
        
        raw_text = extract_text_from_file(file_path)
        
        if raw_text.strip():
            embedded_urls = re.findall(r'(https?://[^\s"\'<>]+)', raw_text)
            unique_embedded_urls = list(dict.fromkeys(embedded_urls))[:15] 
            
            if unique_embedded_urls:
                combined_text += f"\n[System Log: Found {len(unique_embedded_urls)} embedded URLs in {file_name}. Fetching...]\n"
                for e_url in unique_embedded_urls:
                    dl_path, status_msg = download_url_to_temp(e_url)
                    if dl_path:
                        e_text = extract_text_from_file(dl_path)
                        if e_text.strip():
                            combined_text += f"\n--- Data from Embedded Link ({e_url}) ---\nStatus: Accessible\n{e_text}\n"
                        try:
                            os.remove(dl_path)
                        except Exception:
                            pass
                    else:
                        combined_text += f"\n--- Data from Embedded Link ({e_url}) ---\nStatus: FAILED - {status_msg}\n"
        
        if raw_text.strip() and "[Extraction Error]" not in raw_text:
            validation_report = generate_system_validation_report(raw_text)
            combined_text += f"\n\n========================================\n"
            combined_text += f"FILE NAME: {file_name}\n"
            combined_text += f"========================================\n\n"
            combined_text += validation_report + raw_text

    for tmp_dir in temp_dirs_to_cleanup:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
            
    if not combined_text.strip():
        return {
            "status": "completed", 
            "objective": objective,
            "total_processed": total_files, 
            "results": [{"file": "Batch Processing", "status": "failed", "findings": "Could not extract text or OCR data. URL Access logs indicate blocking."}]
        }

    self.update_state(state='PROGRESS', meta={'current': total_files, 'total': total_files, 'file': 'Executing AI Compliance Matrix...'})
    
    ai_report = analyze_compliance_with_matrix(combined_text, rulebook_text, objective)
    
    return {
        "status": "completed", 
        "objective": objective,
        "total_processed": total_files, 
        "results": [{
            "file": f"Synthesized Audit of {total_files} Extracted Files",
            "status": "processed",
            "findings": ai_report
        }]
    }