"""
GPU Document Intelligence - demo app v4 (new UI, faster results)

Upload a PDF/image -> OCR -> field extraction -> compliance check ->
inconsistency check -> summary -> searchable PDF, with a GPU/CPU run log.

What changed from v3:
  * OCR runs in a background thread with a live progress screen
  * the searchable PDF is built AFTER the results show (it used to run OCR a 2nd time first)
  * large photos are scaled down before OCR; phone photos are auto-rotated
  * every result has its own page (/result/<id>), ready for the database step

Run:  python app.py     then open http://127.0.0.1:5000
"""
import io
import json
import re
import shutil
import threading
import time
import uuid
from functools import lru_cache
from collections import Counter
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF: reads PDFs and merges PDFs
import pytesseract
from PIL import Image, ImageOps
from flask import Flask, request, render_template, send_file, abort, jsonify
from jinja2 import DictLoader

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
MAX_SIDE = 2400  # pixels: bigger photos are scaled down before OCR (much faster, usually no loss for printed text)


def shrink(img):
    longest = max(img.size)
    if longest <= MAX_SIDE:
        return img
    scale = MAX_SIDE / longest
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def load_pages(data: bytes, filename: str):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200, alpha=False)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return pages
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))  # fix sideways phone photos
    return [img.convert("RGB")]


# ---------------------------------------------------------------------------
# 2. OCR engines + GPU evidence
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)  # import torch once, not on every page load
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



# ---------------------------------------------------------------------------
# 7. Background jobs: OCR runs in a thread so the page never freezes
# ---------------------------------------------------------------------------
class UserError(Exception):
    """An error whose message is safe and useful to show to the user."""


JOBS = {}  # job id -> live status while a document is being processed
JOBS_LOCK = threading.Lock()
PDF_LOCKS = {}
JOB_RE = re.compile(r"[0-9a-f]{10}")
STEPS = ["Opening the file", "Reading the text (OCR)", "Checking the document", "Done"]


def set_status(job, **changes):
    with JOBS_LOCK:
        JOBS[job].update(changes)


def ensure_pdf(job):
    """Build the searchable PDF once (a second OCR pass), then reuse it."""
    path = OUT / f"{job}.pdf"
    lock = PDF_LOCKS.setdefault(job, threading.Lock())
    with lock:
        if not path.exists():
            src = next(OUT.glob(f"{job}_source.*"), None)
            if src is None:
                return None
            images = [shrink(i) for i in load_pages(src.read_bytes(), src.name)]
            path.write_bytes(make_searchable_pdf(images))
    return path


def run_job(job, data, filename, ext, engine):
    try:
        try:
            images = load_pages(data, filename)
        except Exception:
            app.logger.exception("Could not open file")
            raise UserError("This file could not be opened. It may be corrupt or password-protected.") from None
        if not images:
            raise UserError("The document has no pages.")
        if len(images) > MAX_PAGES:
            raise UserError(f"Too many pages ({len(images)}). This demo accepts up to {MAX_PAGES} pages.")
        images = [shrink(i) for i in images]
        (OUT / f"{job}_source{ext}").write_bytes(data)
        preview = images[0].copy()
        preview.thumbnail((900, 1200))
        preview.save(OUT / f"{job}_preview.jpg", quality=80)

        set_status(job, step=1, message="Getting the OCR engine ready")
        try:
            device, load_s = prepare_engine(engine)
        except ModuleNotFoundError:
            raise UserError("EasyOCR is not installed. Run: python -m pip install easyocr, "
                            "or choose Tesseract.") from None

        gpu_peak_mb(reset=True)
        t0 = time.time()
        pieces = []
        for n, img in enumerate(images, 1):
            set_status(job, message=f"Reading page {n} of {len(images)}")
            pieces.append(ocr_text(img, engine))
        ocr_s = round(time.time() - t0, 2)
        peak = gpu_peak_mb() if device.startswith("GPU") else None
        text = "\n\n".join(pieces)

        set_status(job, step=2, message="Checking required items")
        fields = extract_fields(text)
        found, missing = check_compliance(text, fields)
        issues = check_inconsistencies(text, fields)
        per_page = round(ocr_s / len(images), 2)

        rows = [dict(name=n, ok=n in found, value=None) for n in TEXT_CHECKS]
        rows += [dict(name=n, ok=n in found, value=fields.get(n)) for n in REQUIRED_FIELDS]
        result = dict(
            id=job, created=datetime.now().isoformat(timespec="seconds"), filename=filename, text=text, summary=summarize(text, fields),
            rows=rows, missing=missing, issues=issues, total=len(rows), pages=len(images),
            engine_label=ENGINES[engine], device=device, ocr_seconds=ocr_s,
            per_page=per_page, load_seconds=load_s, peak_mb=peak,
        )
        (OUT / f"{job}.txt").write_text(text, encoding="utf-8")
        (OUT / f"{job}.json").write_text(json.dumps(result), encoding="utf-8")
        set_status(job, step=3, state="done", message="Done")
    except UserError as e:
        set_status(job, state="error", error=str(e))
        return
    except pytesseract.TesseractNotFoundError:
        set_status(job, state="error", error="Tesseract was not found. Install it, or set its path at the top of app.py.")
        return
    except Exception:
        app.logger.exception("Processing failed")
        set_status(job, state="error", error="Something went wrong while processing this file. "
                                              "Try another file or engine. Details are in the terminal.")
        return
    try:
        ensure_pdf(job)  # built while the person reads the results
    except Exception:
        app.logger.exception("Searchable PDF failed (it will be retried on download)")


# ---------------------------------------------------------------------------
# 8. Pages (look and feel)
# ---------------------------------------------------------------------------
BASE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}Document Intelligence{% endblock %}</title>
<style>
:root{--ink:#14222b;--muted:#5a6b75;--bg:#f5f7f9;--surface:#fff;--line:#d9e1e6;--brand:#1459a8;--brand-dark:#0f4a8c;--sun:#f2c230;
 --ok:#17693f;--ok-bg:#e3f1e8;--warn:#7a4f00;--warn-bg:#fdf0d2;--bad:#a32a22;--bad-bg:#fbe6e3}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif;min-height:100vh;display:flex;flex-direction:column}
main{flex:1}
.wrap{max-width:1120px;margin:0 auto;padding:0 1.25rem}
a:focus-visible,button:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid var(--brand);outline-offset:2px}
.muted{color:var(--muted)}.small{font-size:.9rem;color:var(--muted)}

/* top bar */
.nav{background:#fff;border-bottom:1px solid var(--line)}
.navbar{display:flex;align-items:center;gap:.5rem 1.5rem;min-height:68px;flex-wrap:wrap;padding-top:.5rem;padding-bottom:.5rem}
.logo{display:flex;align-items:center;gap:.65rem;color:var(--ink);font-weight:700;font-size:1.15rem;text-decoration:none;letter-spacing:-.01em}
.links{display:flex;gap:.3rem;margin-right:auto}
.links a{color:var(--muted);text-decoration:none;padding:.45rem .85rem;border-radius:8px;font-weight:500}
.links a:hover{color:var(--ink);background:#eef2f5}
.links a[aria-current=page]{color:var(--brand);background:#e8f0fa;font-weight:600}
.btn{display:inline-block;background:var(--brand);color:#fff;border:1px solid var(--brand);border-radius:10px;padding:.65rem 1.3rem;font:inherit;font-weight:600;cursor:pointer;text-decoration:none;text-align:center}
.btn:hover{background:var(--brand-dark);border-color:var(--brand-dark)}
.btn.alt{background:#fff;color:var(--brand)}.btn.alt:hover{background:#eef4fb}
.btn.big{padding:.95rem 2.1rem;font-size:1.1rem;border-radius:12px}
.btn:disabled{opacity:.5;cursor:not-allowed}

/* home */
.hero{text-align:center;padding:3.2rem 0 1rem}
.hero h1{font-size:clamp(2.1rem,4.6vw,3.3rem);line-height:1.1;letter-spacing:-.025em;font-weight:700;max-width:17ch;margin:0 auto .9rem}
.lead{color:var(--muted);font-size:1.1rem;max-width:56ch;margin:0 auto 2.2rem}
.stage{position:relative;max-width:1000px;margin:0 auto}
.uploader{position:relative;z-index:2;max-width:700px;margin:0 auto;background:#fff;border:1px solid var(--line);border-radius:22px;padding:1rem;box-shadow:0 30px 60px -35px rgba(20,60,110,.45);text-align:left}
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.9rem;position:relative;min-height:290px;border:2px dashed #9db4c6;border-radius:16px;text-align:center;padding:2rem 1rem;
 background-color:#fbfdff;background-image:repeating-linear-gradient(to bottom,transparent 0 35px,#e6edf3 35px 36px);cursor:pointer}
.drop.over{border-color:var(--brand);background-color:#eaf2fc}
.drop:focus-within{outline:3px solid var(--brand);outline-offset:2px}
.drop .small,.drop .fname{background:#fbfdff;padding:0 .5rem}
.fname{font-weight:600;word-break:break-all}
.sr{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
.options{display:flex;align-items:center;gap:.6rem 1rem;flex-wrap:wrap;padding:1rem .8rem .3rem}
.options label{font-weight:600}
.options select{font:inherit;padding:.55rem .7rem;border:1px solid #9db4c6;border-radius:8px;background:#fff;max-width:100%}
.options .btn{margin-left:auto}
.options-note{padding:0 .8rem .6rem;margin:.2rem 0 0}
.notice{border-radius:10px;padding:.7rem 1rem;margin:0 auto 1rem;max-width:700px;text-align:left}
.notice.bad{background:var(--bad-bg);color:var(--bad);border:1px solid #efb9b4}
.working{padding:1.2rem 1.4rem 1rem}
.working h2{margin:0 0 .4rem;font-size:1.3rem}
.steps{list-style:none;margin:1rem 0;padding:0}
.steps li{display:flex;gap:.7rem;align-items:center;padding:.4rem 0;color:var(--muted)}
.steps li.current{color:var(--ink);font-weight:600}
.steps li.done{color:var(--ok)}
.dot{width:1.2rem;height:1.2rem;border-radius:50%;border:2px solid #b4c2c8;flex:none;text-align:center;font-size:.72rem;line-height:1rem;color:#fff}
.done .dot{background:var(--ok);border-color:var(--ok)}
.done .dot::after{content:"\2713"}
.current .dot{border-color:var(--brand);border-top-color:transparent;animation:spin .9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.current .dot{animation:none;border-top-color:var(--brand)}}
.float{position:absolute;top:2.4rem;width:200px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:.9rem;box-shadow:0 18px 40px -22px rgba(20,60,110,.5);z-index:1;text-align:left}
.float.left{left:-2rem;transform:rotate(-7deg)}
.float.right{right:-2rem;top:4.6rem;transform:rotate(6deg)}
.sheet b{display:block;font-size:.72rem;line-height:1.25;margin-bottom:.6rem;max-width:130px}
.sheet i{display:block;height:6px;border-radius:3px;background:#e1e8ee;margin:.5rem 0}
.seal{float:right;width:26px;height:26px;border-radius:50%;background:var(--sun)}
.mini strong{display:block;font-size:.85rem;margin-bottom:.5rem}
.mini ul{list-style:none;margin:0;padding:0;font-size:.78rem}
.mini li{padding:.28rem 0;border-top:1px solid #edf1f4}
.mini .ok::before{content:"\2713  ";color:var(--ok);font-weight:700}
.mini .bad::before{content:"\2717  ";color:var(--bad);font-weight:700}
@media(max-width:1100px){.float{display:none}}
.facts{max-width:760px;margin:2.6rem auto 0;text-align:center}
.facts h2{font-size:1rem;color:var(--muted);font-weight:600;margin:0 0 .8rem}
.chips{list-style:none;display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;margin:0;padding:0}
.chips li{background:#fff;border:1px solid var(--line);border-radius:999px;padding:.3rem .85rem;font-size:.92rem}
.foot{padding-top:2rem;padding-bottom:2rem;text-align:center;color:var(--muted);font-size:.9rem}

/* uploads + performance */
.pagehead{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;margin:2.2rem 0 1.4rem}
.pagehead h1{font-size:2rem;margin:0;letter-spacing:-.015em}
.pagehead p{margin:.3rem 0 0;color:var(--muted)}
.filters{display:flex;gap:.5rem;margin-bottom:1.2rem;flex-wrap:wrap}
.chip{font:inherit;background:#fff;border:1px solid var(--line);border-radius:999px;padding:.35rem .95rem;cursor:pointer;color:var(--ink)}
.chip[aria-pressed=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:1.1rem;padding-bottom:2rem}
.up{display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;color:inherit;text-decoration:none}
.up:hover{border-color:var(--brand);box-shadow:0 12px 28px -20px rgba(20,60,110,.7)}
.thumb{height:190px;background:#eef2f5;overflow:hidden}
.thumb img{display:block;width:100%;height:100%;object-fit:cover;object-position:top}
.meta{padding:.8rem .9rem 1rem;display:flex;flex-direction:column;gap:.25rem}
.meta strong{word-break:break-all;line-height:1.3}
.badge{align-self:flex-start;font-size:.82rem;font-weight:600;border-radius:999px;padding:.1rem .7rem}
.badge.ok{background:var(--ok-bg);color:var(--ok)}.badge.warn{background:var(--warn-bg);color:var(--warn)}.badge.bad{background:var(--bad-bg);color:var(--bad)}
.empty{text-align:center;background:#fff;border:1px dashed #9db4c6;border-radius:16px;padding:3rem 1rem;margin-bottom:2rem}
.empty h2{margin:0 0 .4rem}.empty p{margin:0 0 1.2rem;color:var(--muted)}
.note{background:#fff;border:1px solid var(--line);border-radius:10px;padding:.7rem 1rem;margin-bottom:1rem}

/* results */
.verdict{border-left:8px solid;border-radius:10px;padding:1.1rem 1.4rem;margin:2rem 0 1.75rem;background:var(--surface);border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.verdict.ok{border-left-color:var(--ok)}.verdict.warn{border-left-color:#c98a00}.verdict.bad{border-left-color:var(--bad)}
.verdict h1{margin:0 0 .3rem;font-size:1.7rem;letter-spacing:-.01em}
.verdict.ok h1{color:var(--ok)}.verdict.warn h1{color:var(--warn)}.verdict.bad h1{color:var(--bad)}
.verdict p{margin:0;color:var(--muted)}
.actions{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1rem}
.grid{display:grid;grid-template-columns:minmax(0,300px) minmax(0,1fr);gap:1.75rem;align-items:start;padding-bottom:2rem}
.preview{position:sticky;top:1rem;margin:0}
.preview img{display:block;width:100%;border:1px solid var(--line);border-radius:10px;background:#fff}
@media(max-width:800px){.grid{grid-template-columns:1fr}.preview{position:static;max-width:320px}}
h2{font-size:1.15rem;margin:0 0 .6rem;font-weight:650}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:var(--surface)}
th,td{text-align:left;padding:.6rem .8rem;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-size:.85rem;color:var(--muted);font-weight:600}
tr.missing td{background:var(--bad-bg)}
.mark{font-weight:600;white-space:nowrap}.mark.ok{color:var(--ok)}.mark.bad{color:var(--bad)}
.review{background:var(--warn-bg);border:1px solid #ecd391;border-radius:10px;padding:.8rem 1rem;color:var(--warn);margin-bottom:1.5rem}
.review ul{margin:.3rem 0 0 1.1rem;padding:0}
.block{margin-top:1.75rem}.block p{max-width:70ch;margin:0}
details{margin-top:1rem;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.7rem 1rem}
summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;background:#f3f6f7;padding:.75rem;border-radius:6px;max-height:320px;overflow:auto;margin:.7rem 0 0}
dl.tech{display:grid;grid-template-columns:max-content 1fr;gap:.3rem 1.2rem;margin:.7rem 0 0}
dl.tech dt{color:var(--muted)}dl.tech dd{margin:0}
</style>
</head>
<body>
<header class="nav"><div class="wrap navbar">
  <a class="logo" href="/">
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><rect width="34" height="34" rx="9" fill="#1459a8"/><path d="M10 8h10l5 5v13a1 1 0 0 1-1 1H10a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" fill="#fff"/><path d="M20 8v5h5z" fill="#c5d8ec"/><circle cx="24" cy="24" r="6.5" fill="#f2c230"/><path d="M21.2 24.2l2 2 3.6-3.8" stroke="#14222b" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <span>Document Intelligence</span>
  </a>
  <nav class="links" aria-label="Main">
    <a href="/uploads"{% if active == 'uploads' %} aria-current="page"{% endif %}>My uploads</a>
    <a href="/runs"{% if active == 'runs' %} aria-current="page"{% endif %}>Performance</a>
  </nav>
  <a class="btn" href="/">New check</a>
</div></header>
<main class="wrap">{% block body %}{% endblock %}</main>
<footer class="foot wrap">Checks RDB Certificates of Registration. More Rwandan document types are being added.</footer>
{% block script %}{% endblock %}
</body>
</html>"""

HOME = r"""{% extends "base.html" %}
{% block body %}
<section class="hero">
  <h1>Know if a business document is complete</h1>
  <p class="lead">Upload a registration certificate. Get the key details, a checklist of anything missing, and a searchable PDF in seconds.</p>

  <p id="err" class="notice bad" role="alert" hidden></p>
  <div class="stage">
    <div class="float left" aria-hidden="true"><div class="sheet"><span class="seal"></span><b>Certificate of Domestic Company Registration</b><i></i><i style="width:70%"></i><i></i><i style="width:55%"></i><i style="width:80%"></i></div></div>
    <div class="float right" aria-hidden="true"><div class="mini"><strong>7 of 8 items found</strong>
      <ul><li class="ok">Company name</li><li class="ok">Company code</li><li class="ok">Registration date</li><li class="bad">Date of issuance</li></ul></div></div>

    <div class="uploader">
      <form id="f" method="post" action="/analyze" enctype="multipart/form-data">
        <label class="drop" id="drop">
          <input class="sr" id="file" type="file" name="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" required>
          <span class="btn big" id="pick">Upload a document</span>
          <span class="small" id="hint">or drop a file here</span>
          <strong class="fname" id="fname" hidden></strong>
          <span class="small">PDF, PNG, JPG or TIFF. Up to 20 MB and {{ max_pages }} pages.</span>
        </label>
        <div class="options">
          <label for="engine">Reading method</label>
          <select id="engine" name="engine">
            {% for key, label in engines.items() %}<option value="{{ key }}">{{ label }}</option>{% endfor %}
          </select>
          <button id="go" class="btn" type="submit" disabled>Analyze document</button>
        </div>
        <p class="small options-note">Tesseract is fastest on a laptop CPU. EasyOCR is faster when a GPU is available.</p>
      </form>

      <div id="panel" class="working" hidden>
        <h2>Analyzing your document</h2>
        <ol class="steps" id="steps">
          {% for s in steps %}<li data-i="{{ loop.index0 }}"><span class="dot"></span>{{ s }}</li>{% endfor %}
        </ol>
        <p id="msg" class="muted" aria-live="polite" style="margin:0"></p>
        <p id="elapsed" class="small" style="margin:.2rem 0 0"></p>
      </div>
    </div>
  </div>

  <div class="facts">
    <h2>What the app checks</h2>
    <ul class="chips">{% for c in checks %}<li>{{ c }}</li>{% endfor %}</ul>
  </div>
</section>
{% endblock %}
{% block script %}
<script>
const form = document.getElementById('f'), input = document.getElementById('file');
const drop = document.getElementById('drop'), fname = document.getElementById('fname');
const pick = document.getElementById('pick'), hint = document.getElementById('hint');
const go = document.getElementById('go'), err = document.getElementById('err');
const panel = document.getElementById('panel');
const msg = document.getElementById('msg'), elapsed = document.getElementById('elapsed');

function showError(text) { err.textContent = text; err.hidden = false; }
function chosen() {
  if (!input.files.length) return;
  fname.textContent = input.files[0].name; fname.hidden = false; hint.hidden = true;
  pick.textContent = 'Choose a different file'; pick.classList.add('alt');
  go.disabled = false; err.hidden = true;
}
input.addEventListener('change', chosen);
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; chosen(); } });

form.addEventListener('submit', async e => {
  e.preventDefault();
  err.hidden = true;
  if (!input.files.length) { showError('Choose a file first.'); return; }
  go.disabled = true; go.textContent = 'Uploading...';
  try {
    const res = await fetch('/analyze', { method: 'POST', body: new FormData(form) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'The upload failed.');
    track(data.job);
  } catch (ex) {
    showError(ex instanceof TypeError ? 'Could not reach the app. Check that it is still running.' : ex.message);
    go.disabled = false; go.textContent = 'Analyze document';
  }
});

function paint(s) {
  document.querySelectorAll('#steps li').forEach(li => {
    const i = +li.dataset.i;
    li.className = i < s.step ? 'done' : (i === s.step ? 'current' : '');
  });
  msg.textContent = s.message;
  elapsed.textContent = 'Working for ' + s.elapsed + ' s';
}

function track(job) {
  form.hidden = true; panel.hidden = false;
  const timer = setInterval(poll, 700);
  async function poll() {
    try {
      const s = await (await fetch('/status/' + job)).json();
      paint(s);
      if (s.state === 'done') { clearInterval(timer); location.href = '/result/' + job; }
      if (s.state === 'error') {
        clearInterval(timer); panel.hidden = true; form.hidden = false;
        go.disabled = false; go.textContent = 'Analyze document'; showError(s.error);
      }
    } catch (ex) { /* one failed check is fine; the next one retries */ }
  }
  poll();
}
</script>
{% endblock %}"""

UPLOADS = r"""{% extends "base.html" %}
{% block title %}My uploads{% endblock %}
{% block body %}
<div class="pagehead">
  <div><h1>My uploads</h1><p>Documents you have checked. Open one to see its results again.</p></div>
  <a class="btn" href="/">Check a document</a>
</div>
{% if items %}
<div class="filters" role="group" aria-label="Filter uploads">
  <button class="chip" data-f="all" aria-pressed="true">All ({{ items|length }})</button>
  <button class="chip" data-f="ok" aria-pressed="false">Complete ({{ n_ok }})</button>
  <button class="chip" data-f="attention" aria-pressed="false">Needs attention ({{ items|length - n_ok }})</button>
</div>
<div class="gallery">
  {% for x in items %}
  <a class="up" href="/result/{{ x.id }}" data-tone="{{ x.tone }}">
    <div class="thumb"><img src="/preview/{{ x.id }}.jpg" alt="" loading="lazy"></div>
    <div class="meta">
      <strong>{{ x.filename }}</strong>
      <span class="badge {{ x.tone }}">{{ x.label }}</span>
      <span class="small">{{ x.created|when }}</span>
      <span class="small">{{ x.pages }} page{{ '' if x.pages == 1 else 's' }}, {{ x.engine_label }}</span>
    </div>
  </a>
  {% endfor %}
</div>
{% else %}
<div class="empty">
  <h2>No uploads yet</h2>
  <p>Check your first document and it will appear here.</p>
  <a class="btn big" href="/">Check a document</a>
</div>
{% endif %}
{% endblock %}
{% block script %}
<script>
const chips = document.querySelectorAll('.chip');
chips.forEach(c => c.addEventListener('click', () => {
  chips.forEach(x => x.setAttribute('aria-pressed', x === c));
  document.querySelectorAll('.up').forEach(card => {
    const t = card.dataset.tone, f = c.dataset.f;
    card.hidden = !(f === 'all' || (f === 'ok' && t === 'ok') || (f === 'attention' && t !== 'ok'));
  });
}));
</script>
{% endblock %}"""

RUNS_PAGE = r"""{% extends "base.html" %}
{% block title %}Performance{% endblock %}
{% block body %}
<div class="pagehead">
  <div><h1>Performance</h1><p>Every run is saved here, so you can compare CPU and GPU. Run the same file with each engine.</p></div>
</div>
<p class="note"><strong>This computer:</strong> {{ gpu }}</p>
{% if runs %}
<div class="scroll"><table>
  <thead><tr><th>When</th><th>File</th><th>Engine</th><th>Device</th><th>Pages</th><th>OCR s</th><th>s/page</th><th>Model load s</th><th>GPU peak MB</th><th></th></tr></thead>
  {% for x in runs %}<tr>
    <td>{{ x.created|when }}</td><td>{{ x.filename }}</td><td>{{ x.engine_label }}</td><td>{{ x.device }}</td>
    <td>{{ x.pages }}</td><td>{{ x.ocr_seconds }}</td><td>{{ x.per_page }}</td><td>{{ x.load_seconds }}</td>
    <td>{{ x.peak_mb if x.peak_mb is not none else '-' }}</td><td><a href="/result/{{ x.id }}">Open</a></td>
  </tr>{% endfor %}
</table></div>
<p class="small" style="margin:.8rem 0 2rem">Model load time is the one-off cost of loading EasyOCR. It is shown separately so it does not count against OCR speed.</p>
{% else %}
<div class="empty">
  <h2>No runs yet</h2>
  <p>Check a document and its timings will appear here.</p>
  <a class="btn big" href="/">Check a document</a>
</div>
{% endif %}
{% endblock %}"""

RESULT = r"""{% extends "base.html" %}
{% block title %}Results for {{ r.filename }}{% endblock %}
{% block body %}
<section class="verdict {{ tone }}">
  <h1>{{ headline }}</h1>
  <p>{{ r.filename }}: {{ r.pages }} page{{ '' if r.pages == 1 else 's' }} read with {{ r.engine_label }} in {{ r.ocr_seconds }} s.</p>
  <div class="actions">
    <a class="btn" href="/download/{{ r.id }}.pdf">Download searchable PDF</a>
    <a class="btn alt" href="/download/{{ r.id }}.txt">Download text</a>
    <a class="btn alt" href="/uploads">My uploads</a>
    <a class="btn alt" href="/">Check another document</a>
  </div>
</section>

<div class="grid">
  <figure class="preview"><img src="/preview/{{ r.id }}.jpg" alt="First page of the uploaded document"></figure>
  <div>
    {% if r.issues %}
    <div class="review"><strong>{{ r.issues|length }} thing{{ '' if r.issues|length == 1 else 's' }} to review</strong>
      <ul>{% for i in r.issues %}<li>{{ i }}</li>{% endfor %}</ul></div>
    {% endif %}

    <h2>Checklist</h2>
    <div class="scroll"><table>
      <thead><tr><th>Item</th><th>Status</th><th>What was read</th></tr></thead>
      {% for row in r.rows %}
      <tr class="{{ '' if row.ok else 'missing' }}">
        <td>{{ row.name }}</td>
        <td>{% if row.ok %}<span class="mark ok">&#10003; Found</span>{% else %}<span class="mark bad">&#10007; Missing</span>{% endif %}</td>
        <td>{% if row.value %}{{ row.value }}{% elif row.ok %}<span class="muted">Present in the text</span>{% else %}<span class="muted">Not found in the text</span>{% endif %}</td>
      </tr>
      {% endfor %}
    </table></div>

    <div class="block"><h2>Summary</h2><p>{{ r.summary }}</p></div>

    <details><summary>Show the extracted text</summary><pre>{{ r.text }}</pre></details>
    <details><summary>Technical details</summary>
      <dl class="tech">
        <dt>OCR engine</dt><dd>{{ r.engine_label }}</dd>
        <dt>Device</dt><dd>{{ r.device }}</dd>
        <dt>Pages</dt><dd>{{ r.pages }}</dd>
        <dt>OCR time</dt><dd>{{ r.ocr_seconds }} s ({{ r.per_page }} s per page)</dd>
        <dt>Model load time</dt><dd>{{ r.load_seconds }} s</dd>
        {% if r.peak_mb is not none %}<dt>Peak GPU memory</dt><dd>{{ r.peak_mb }} MB</dd>{% endif %}
      </dl>
    </details>
  </div>
</div>
{% endblock %}"""

app.jinja_loader = DictLoader({"base.html": BASE, "home.html": HOME, "uploads.html": UPLOADS,
                               "runs.html": RUNS_PAGE, "result.html": RESULT})


# ---------------------------------------------------------------------------
# 9. Reading saved results (today: JSON files. Next step: PostgreSQL.)
# ---------------------------------------------------------------------------
def load_results(limit=None):
    items = []
    for p in OUT.glob("*.json"):
        if not JOB_RE.fullmatch(p.stem):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        r.setdefault("created", datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
        items.append(r)
    items.sort(key=lambda r: r["created"], reverse=True)
    return items[:limit] if limit else items


def verdict(r):
    """Returns (tone, headline, short label) for a saved result."""
    n_missing, n_issues = len(r["missing"]), len(r["issues"])
    if n_missing:
        return "bad", f"{n_missing} of {r['total']} required items are missing", f"{n_missing} missing"
    if n_issues:
        head = ("All required items found, but 1 inconsistency needs review" if n_issues == 1 else
                f"All required items found, but {n_issues} inconsistencies need review")
        return "warn", head, "Needs review"
    return "ok", "All required items found", "Complete"


@app.template_filter("when")
def when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return iso


# ---------------------------------------------------------------------------
# 10. Routes
# ---------------------------------------------------------------------------
@app.context_processor
def common():
    return dict(gpu=gpu_info(), engines=ENGINES, steps=STEPS, active=None)


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="That file is too large. The limit is 20 MB."), 413


@app.route("/")
def index():
    return render_template("home.html", max_pages=MAX_PAGES, checks=list(TEXT_CHECKS) + REQUIRED_FIELDS)


@app.route("/uploads")
def uploads():
    items = []
    for r in load_results():
        tone, _head, label = verdict(r)
        items.append(dict(r, tone=tone, label=label))
    n_ok = sum(1 for x in items if x["tone"] == "ok")
    return render_template("uploads.html", active="uploads", items=items, n_ok=n_ok)


@app.route("/runs")
def runs():
    return render_template("runs.html", active="runs", runs=load_results(30))


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a file first."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"Unsupported file type '{ext or 'none'}'. Upload a PDF, PNG, JPG or TIFF."), 400
    engine = request.form.get("engine", "tesseract")
    if engine not in ENGINES:
        engine = "tesseract"
    job = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job] = dict(state="running", step=0, message="Opening the file",
                         started=time.time(), error=None)
    threading.Thread(target=run_job, args=(job, f.read(), f.filename, ext, engine), daemon=True).start()
    return jsonify(job=job)


@app.route("/status/<job>")
def status(job):
    if not JOB_RE.fullmatch(job):
        abort(404)
    with JOBS_LOCK:
        s = dict(JOBS.get(job) or {})
    if not s:
        if (OUT / f"{job}.json").exists():
            return jsonify(state="done", step=3, message="Done", elapsed=0, error=None)
        abort(404)
    return jsonify(state=s["state"], step=s["step"], message=s["message"],
                   error=s["error"], elapsed=round(time.time() - s["started"]))


@app.route("/result/<job>")
def result(job):
    path = OUT / f"{job}.json"
    if not JOB_RE.fullmatch(job) or not path.exists():
        abort(404)
    r = json.loads(path.read_text(encoding="utf-8"))
    tone, headline, _label = verdict(r)
    return render_template("result.html", r=r, tone=tone, headline=headline)


@app.route("/preview/<job>.jpg")
def preview(job):
    path = OUT / f"{job}_preview.jpg"
    if not JOB_RE.fullmatch(job) or not path.exists():
        abort(404)
    return send_file(path.resolve())


@app.route("/download/<name>")
def download(name):
    m = re.fullmatch(r"([0-9a-f]{10})\.(pdf|txt)", name)
    if not m:
        abort(404)
    job, kind = m.groups()
    path = ensure_pdf(job) if kind == "pdf" else OUT / name
    if not path or not path.exists():
        abort(404)
    return send_file(path.resolve(), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False)  # debug off: no interactive traceback page
