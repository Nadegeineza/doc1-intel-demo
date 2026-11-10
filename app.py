"""
GPU Document Intelligence - demo app v3

Upload a PDF/image -> OCR -> field extraction -> compliance check ->
inconsistency check -> summary -> searchable PDF, with a GPU/CPU run log.

Run:  python app.py     then open http://127.0.0.1:5000
"""
import io
import re
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF: reads PDFs and merges PDFs
import pytesseract
from PIL import Image
from flask import Flask, request, render_template_string, send_file, abort
from werkzeug.exceptions import HTTPException

# Use Tesseract from PATH, else the default Windows install location.
# If yours is elsewhere, change the path below.
_DEFAULT_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not shutil.which("tesseract") and Path(_DEFAULT_TESS).exists():
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # reject uploads over 20 MB
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_PAGES = 20  # keep demo runs short
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

ENGINES = {
    "tesseract": "Tesseract (CPU)",
    "easyocr_gpu": "EasyOCR (GPU if available)",
    "easyocr_cpu": "EasyOCR (CPU)",
}
RUNS = []  # run log used as GPU vs CPU evidence (newest first)
_readers = {}

# ---------------------------------------------------------------------------
# CHECKLIST: what must be present on the certificate
# ---------------------------------------------------------------------------
TEXT_CHECKS = {
    "Document title": r"certificate\s+of\s+(domestic\s+|foreign\s+)?(company\s+)?registration",
    "Issuing body (Registrar General / RDB)": r"registrar\s+general|rwanda\s+development\s+board|\bRDB\b",
}

DATE = r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})"
FIELD_PATTERNS = {
    "Company name": r"company\s*name\s*[:\-][ \t]*(\S.*)",
    "Category": r"category\s*[:\-][ \t]*(\S.*)",
    "Company code": r"(?:company\s*(?:code|no\.?|number)|registration\s*(?:no\.?|number)|\bTIN\b)\s*[:\-]?[ \t]*([\w/\-]{5,})",
    "Registration date": r"registration\s*date\s*[:\-]?[ \t]*" + DATE,
    "Date of issuance": r"date\s+of\s+issuance\s*[:\-]?[ \t]*" + DATE,
    "Registered office address": r"registered\s+office\s+address\s*[:\-]?[ \t]*([A-Za-z0-9].*)",
}
REQUIRED_FIELDS = list(FIELD_PATTERNS)

STOPWORDS = set(
    "the a an and or of to in on at for with by is are was were be been this that "
    "these those it its as from not no has have had will shall may can".split()
)


# ---------------------------------------------------------------------------
# 1. Load pages
# ---------------------------------------------------------------------------
def load_pages(data: bytes, filename: str):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200, alpha=False)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return pages
    return [Image.open(io.BytesIO(data)).convert("RGB")]


# ---------------------------------------------------------------------------
# 2. OCR engines + GPU evidence
# ---------------------------------------------------------------------------
def gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name}, {p.total_memory / 1024**3:.1f} GB memory, CUDA available"
        return "No CUDA GPU detected: GPU runs fall back to CPU"
    except ImportError:
        return "PyTorch not installed: EasyOCR engines unavailable"


def gpu_peak_mb(reset=False):
    try:
        import torch
        if torch.cuda.is_available():
            if reset:
                torch.cuda.reset_peak_memory_stats()
                return 0
            return round(torch.cuda.max_memory_allocated() / 1024**2)
    except ImportError:
        pass
    return None


def get_reader(use_gpu: bool):
    import easyocr
    import torch
    gpu = use_gpu and torch.cuda.is_available()
    if gpu not in _readers:
        _readers[gpu] = easyocr.Reader(["en", "fr"], gpu=gpu)
    return _readers[gpu], gpu


def prepare_engine(engine: str):
    """Load the model BEFORE timing, so model loading is not counted as OCR speed."""
    t0 = time.time()
    if engine == "tesseract":
        return "CPU", 0.0
    _, gpu = get_reader(engine == "easyocr_gpu")
    return ("GPU (CUDA)" if gpu else "CPU"), round(time.time() - t0, 2)


def ocr_text(img, engine: str) -> str:
    if engine == "tesseract":
        return pytesseract.image_to_string(img)
    import numpy as np
    reader, _ = get_reader(engine == "easyocr_gpu")
    return "\n".join(reader.readtext(np.array(img), detail=0, paragraph=False))


# ---------------------------------------------------------------------------
# 3. Information extraction: pull each field's VALUE out of the text
# ---------------------------------------------------------------------------
def extract_fields(text: str) -> dict:
    fields = {}
    for name, pattern in FIELD_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        fields[name] = m.group(1).strip() if m else None
    return fields


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# 4. Compliance check (missing) and consistency check (contradictions)
# ---------------------------------------------------------------------------
def check_compliance(text: str, fields: dict):
    found, missing = [], []
    for name, pattern in TEXT_CHECKS.items():
        (found if re.search(pattern, text, re.IGNORECASE) else missing).append(name)
    for name in REQUIRED_FIELDS:
        (found if fields.get(name) else missing).append(name)
    return found, missing


def check_inconsistencies(text: str, fields: dict):
    issues = []
    reg = parse_date(fields.get("Registration date"))
    iss = parse_date(fields.get("Date of issuance"))
    if reg and iss and reg > iss:
        issues.append("Registration date is later than the date of issuance.")

    law = re.search(r"law\s*n[°o]?\s*[\w/]+\s*of\s*" + DATE, text, re.IGNORECASE)
    law_date = parse_date(law.group(1)) if law else None
    if iss and law_date and iss < law_date:
        issues.append(
            f"Certificate issued {fields['Date of issuance']} cites a law dated "
            f"{law.group(1)}, which did not exist yet at issuance."
        )
    for label in ("Registration date", "Date of issuance"):
        if fields.get(label) and not parse_date(fields[label]):
            issues.append(f"{label} could not be read as a valid date: {fields[label]}")
    if re.search(r"\bDUPLICATE\b", text):
        issues.append("Document is marked DUPLICATE: confirm it is an authorised copy.")
    return issues


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
def summarize(text: str, fields: dict, n: int = 3) -> str:
    present = {k: v for k, v in fields.items() if v}
    title = re.search(r"certificate\s+of\s+[a-z ]*registration", text, re.IGNORECASE)
    if len(present) >= 3:
        head = f"{title.group(0).title()}. " if title else ""
        return head + "; ".join(f"{k}: {v}" for k, v in present.items()) + "."
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 25]
    if not sentences:
        return "Not enough readable text to summarize."
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", text.lower()) if w not in STOPWORDS]
    freq = Counter(words)
    scored = []
    for i, s in enumerate(sentences):
        tokens = [w for w in re.findall(r"[a-zA-Z]{3,}", s.lower()) if w not in STOPWORDS]
        scored.append((sum(freq[w] for w in tokens) / (len(tokens) + 1), i, s))
    top = sorted(sorted(scored, reverse=True)[:n], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


# ---------------------------------------------------------------------------
# 6. Searchable PDF
# ---------------------------------------------------------------------------
def make_searchable_pdf(images) -> bytes:
    merged = fitz.open()
    for img in images:
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension="pdf")
        merged.insert_pdf(fitz.open("pdf", pdf_bytes))
    return merged.tobytes()


PAGE = """
<!doctype html><title>Document Intelligence</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}
 .card{border:1px solid #ddd;border-radius:10px;padding:1rem 1.25rem;margin:1rem 0}
 .ok{color:#127a3a}.bad{color:#b3261e;font-weight:600}.warn{color:#8a5a00;font-weight:600}
 pre{white-space:pre-wrap;background:#f6f6f6;padding:.75rem;border-radius:8px;max-height:300px;overflow:auto}
 button{padding:.5rem 1.1rem;font-size:1rem;cursor:pointer}
 .alert{background:#fdecea;border:1px solid #f5c2c0}.pass{background:#e8f5ec;border:1px solid #b7dfc4}
 .flag{background:#fff6e0;border:1px solid #f0d9a0}
 table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e3e3e3;padding:.35rem .5rem;text-align:left}
 .scroll{overflow-x:auto}.muted{color:#666;font-size:.9rem}
</style>
<h1>GPU Document Intelligence</h1>
<p class="muted">Hardware: {{ gpu }}</p>
{% if error %}<div class="card alert"><p class="bad">{{ error }}</p></div>{% endif %}
<form id="f" class="card" method="post" enctype="multipart/form-data">
  <p><input type="file" name="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" required></p>
  <p>OCR engine:
    <select name="engine">
      {% for key, label in engines.items() %}<option value="{{ key }}">{{ label }}</option>{% endfor %}
    </select></p>
  <button id="go" type="submit">Analyze document</button>
  <p id="busy" class="muted" style="display:none">Processing... OCR can take several seconds (the first EasyOCR run also downloads and loads its model). Please wait.</p>
</form>
<script>
document.getElementById('f').addEventListener('submit', function () {
  var b = document.getElementById('go'); b.disabled = true; b.textContent = 'Processing...';
  document.getElementById('busy').style.display = 'block';
});
</script>

{% if r %}
  <div class="card {{ 'alert' if r.missing else 'pass' }}">
    <h2>Compliance check</h2>
    {% if r.missing %}
      <p class="bad">ALERT: {{ r.missing|length }} required item(s) missing</p>
      <ul>{% for m in r.missing %}<li class="bad">{{ m }}</li>{% endfor %}</ul>
    {% else %}<p class="ok">All required items found.</p>{% endif %}
    {% if r.found %}<p class="ok">Found: {{ r.found|join(', ') }}</p>{% endif %}
  </div>

  <div class="card {{ 'flag' if r.issues else 'pass' }}">
    <h2>Consistency check</h2>
    {% if r.issues %}
      <p class="warn">{{ r.issues|length }} potential inconsistenc{{ 'y' if r.issues|length == 1 else 'ies' }} for review</p>
      <ul>{% for i in r.issues %}<li class="warn">{{ i }}</li>{% endfor %}</ul>
    {% else %}<p class="ok">No inconsistencies detected.</p>{% endif %}
  </div>

  <div class="card"><h2>Extracted information</h2>
    <table>{% for k, v in r.fields.items() %}
      <tr><th>{{ k }}</th><td>{% if v %}{{ v }}{% else %}<span class="bad">not found</span>{% endif %}</td></tr>
    {% endfor %}</table>
  </div>

  <div class="card"><h2>Summary</h2><p>{{ r.summary }}</p></div>
  <div class="card"><h2>Extracted text</h2><pre>{{ r.text }}</pre></div>
  <div class="card">
    <p>{{ r.pages }} page(s) | {{ r.engine_label }} | device: <b>{{ r.device }}</b> |
       OCR {{ r.ocr_seconds }} s ({{ r.per_page }} s/page) | model load {{ r.load_seconds }} s
       {% if r.peak_mb is not none %}| peak GPU memory {{ r.peak_mb }} MB{% endif %}</p>
    <a href="/download/{{ r.id }}.pdf">Download searchable PDF</a> |
    <a href="/download/{{ r.id }}.txt">Download text</a>
  </div>
{% endif %}

{% if runs %}
  <div class="card scroll"><h2>Run log (CPU vs GPU evidence)</h2>
    <table><tr><th>Time</th><th>File</th><th>Engine</th><th>Device</th><th>Pages</th>
      <th>OCR s</th><th>s/page</th><th>GPU peak MB</th></tr>
    {% for x in runs %}<tr><td>{{ x.time }}</td><td>{{ x.file }}</td><td>{{ x.engine }}</td>
      <td>{{ x.device }}</td><td>{{ x.pages }}</td><td>{{ x.ocr_seconds }}</td>
      <td>{{ x.per_page }}</td><td>{{ x.peak_mb if x.peak_mb is not none else '-' }}</td></tr>{% endfor %}
    </table>
    <p class="muted">Run the same file with each engine to compare. The first EasyOCR run loads the model, which is reported separately from OCR time.</p>
  </div>
{% endif %}
"""


class UserError(Exception):
    """An error whose message is safe and useful to show to the user."""


def process_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        raise UserError("Please choose a file first.")
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise UserError(f"Unsupported file type '{ext or 'none'}'. Upload a PDF, PNG, JPG or TIFF.")
    engine = request.form.get("engine", "tesseract")
    if engine not in ENGINES:
        engine = "tesseract"

    try:
        images = load_pages(f.read(), f.filename)
    except Exception:
        app.logger.exception("Could not open file")
        raise UserError("This file could not be opened. It may be corrupt or password-protected.") from None
    if not images:
        raise UserError("The document has no pages.")
    if len(images) > MAX_PAGES:
        raise UserError(f"Too many pages ({len(images)}). This demo accepts up to {MAX_PAGES} pages.")

    try:
        device, load_s = prepare_engine(engine)
    except ModuleNotFoundError:
        raise UserError("EasyOCR is not installed. Run: python -m pip install easyocr, "
                        "or choose Tesseract.") from None

    gpu_peak_mb(reset=True)
    try:
        t0 = time.time()
        text = "\n\n".join(ocr_text(img, engine) for img in images)
        ocr_s = round(time.time() - t0, 2)
        peak = gpu_peak_mb() if device.startswith("GPU") else None
        pdf_bytes = make_searchable_pdf(images)
    except pytesseract.TesseractNotFoundError:
        raise UserError("Tesseract was not found. Install it, or set its path at the top of app.py.") from None

    fields = extract_fields(text)
    found, missing = check_compliance(text, fields)
    issues = check_inconsistencies(text, fields)

    job = uuid.uuid4().hex[:10]
    (OUT / f"{job}.txt").write_text(text, encoding="utf-8")
    (OUT / f"{job}.pdf").write_bytes(pdf_bytes)

    per_page = round(ocr_s / len(images), 2)
    RUNS.insert(0, dict(
        time=datetime.now().strftime("%H:%M:%S"), file=f.filename,
        engine=ENGINES[engine], device=device, pages=len(images),
        ocr_seconds=ocr_s, per_page=per_page, peak_mb=peak,
    ))
    del RUNS[10:]
    return dict(
        id=job, text=text, summary=summarize(text, fields), fields=fields,
        found=found, missing=missing, issues=issues, pages=len(images),
        engine_label=ENGINES[engine], device=device, ocr_seconds=ocr_s,
        per_page=per_page, load_seconds=load_s, peak_mb=peak,
    )


def render(result=None, error=None, status=200):
    html = render_template_string(PAGE, r=result, runs=RUNS, engines=ENGINES,
                                  gpu=gpu_info(), error=error)
    return html, status


@app.errorhandler(413)
def too_large(_e):
    return render(error="That file is too large. The limit is 20 MB.", status=413)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            return render(result=process_upload())
        except UserError as e:
            return render(error=str(e), status=400)
        except HTTPException:
            raise  # let Flask handle it (e.g. 413 file too large)
        except Exception:
            app.logger.exception("Processing failed")
            return render(error="Something went wrong while processing this file. "
                                "Try another file or engine. Details are in the terminal.", status=500)
    return render()


@app.route("/download/<name>")
def download(name):
    if not re.fullmatch(r"[0-9a-f]{10}\.(pdf|txt)", name):
        abort(404)
    path = OUT / name
    if not path.exists():
        abort(404)
    return send_file(path.resolve(), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False)  # debug off: no interactive traceback page
