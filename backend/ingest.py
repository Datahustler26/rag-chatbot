"""
backend/ingest.py — PDF ingestion pipeline

Stages:
  1. Extract text (PyMuPDF native + Tesseract OCR fallback)
  2. Clean & normalize
  3. Chunk with overlap
  4. Embed chunks
  5. Upsert into Qdrant
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import fitz                        # PyMuPDF
import pytesseract
import tiktoken
from langdetect import detect, LangDetectException
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_TOKENS,
    OCR_DPI, OCR_LANG, OCR_MIN_CHARS,
)

# ── Tokenizer ──────────────────────────────────────────────────────────────
_ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))

def token_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into token-sized chunks with overlap."""
    tokens = _ENC.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(_ENC.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += size - overlap
    return chunks


# ── Data model ─────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:  str
    pdf_id:    str
    filename:  str
    page:      int
    text:      str
    tokens:    int
    language:  str
    section:   str = ""
    ocr_used:  bool = False


@dataclass
class PDFMeta:
    pdf_id:   str
    filename: str
    pages:    int
    chunks:   list[Chunk] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────
_HEADER_FOOTER_RE = re.compile(
    r"^\s*(\d+\s*$|Page\s+\d+|[\w\s]+\|\s*\d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_WHITESPACE_RE  = re.compile(r"\s{3,}")
_HYPHEN_BREAK   = re.compile(r"(\w)-\n(\w)")

def clean_text(raw: str) -> str:
    """Normalize unicode, strip headers/footers, collapse whitespace."""
    text = unicodedata.normalize("NFKC", raw)
    text = _HEADER_FOOTER_RE.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # rejoin hyphenated words
    text = text.replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()

def detect_language(text: str) -> str:
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "en"

def pdf_id_from_path(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]

def page_to_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ── Section heading detection ───────────────────────────────────────────────
_HEADING_RE = re.compile(
    r"^(\d+\.?\s+[A-Z][^\n]{3,60}|[A-Z][A-Z\s]{4,40})$",
    re.MULTILINE,
)

def extract_section(text: str) -> str:
    """Return the last heading found before this text block."""
    m = _HEADING_RE.search(text)
    return m.group(0).strip() if m else ""


# ── Core extraction ────────────────────────────────────────────────────────
def extract_page_text(page: fitz.Page, use_ocr_threshold: int = OCR_MIN_CHARS) -> tuple[str, bool]:
    """
    Extract text from a single page.
    Falls back to Tesseract OCR if native text is sparse (scanned page).
    Returns (text, ocr_used).
    """
    native = page.get_text("text")
    native_clean = clean_text(native)

    if len(native_clean) >= use_ocr_threshold:
        return native_clean, False

    # OCR fallback
    try:
        img = page_to_image(page)
        ocr_raw = pytesseract.image_to_string(img, lang=OCR_LANG)
        return clean_text(ocr_raw), True
    except Exception as e:
        print(f"[Warning] OCR failed (Tesseract may not be installed or configured): {e}")
        return native_clean, False


def ingest_pdf(pdf_path: Path) -> PDFMeta:
    """
    Full ingestion pipeline for a single PDF.
    Returns a PDFMeta with all Chunk objects.
    """
    pdf_path = Path(pdf_path)
    pid = pdf_id_from_path(pdf_path)
    doc = fitz.open(str(pdf_path))
    meta = PDFMeta(pdf_id=pid, filename=pdf_path.name, pages=len(doc))

    chunk_index = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        text, ocr_used = extract_page_text(page)

        if not text:
            continue

        lang    = detect_language(text)
        section = extract_section(text)

        for chunk_text in token_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP):
            tok = count_tokens(chunk_text)
            if tok < MIN_CHUNK_TOKENS:
                continue

            cid = f"{pid}_{page_num}_{chunk_index}"
            meta.chunks.append(Chunk(
                chunk_id=cid,
                pdf_id=pid,
                filename=pdf_path.name,
                page=page_num + 1,          # 1-indexed
                text=chunk_text,
                tokens=tok,
                language=lang,
                section=section,
                ocr_used=ocr_used,
            ))
            chunk_index += 1

    doc.close()
    return meta


def ingest_directory(
    pdf_dir: Path,
    on_progress: callable | None = None,
) -> Generator[PDFMeta, None, None]:
    """
    Ingest all PDFs in a directory.
    Yields PDFMeta objects as each PDF is processed.
    """
    pdf_dir = Path(pdf_dir)
    pdfs    = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    for i, pdf_path in enumerate(tqdm(pdfs, desc="Ingesting PDFs")):
        meta = ingest_pdf(pdf_path)
        if on_progress:
            on_progress(i + 1, len(pdfs), pdf_path.name, len(meta.chunks))
        yield meta
