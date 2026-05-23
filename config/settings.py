"""
config/settings.py — Central configuration for RAGCore
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PDF_DIR  = BASE_DIR / "pdfs"
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Qdrant ─────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")

# ── Embedding model (free / OSS) ───────────────────────────────────────────
# BAAI/bge-base-en-v1.5: 768-dim, MIT license, strong retrieval performance
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM     = 768
EMBEDDING_BATCH   = 512          # chunks per forward pass
EMBEDDING_DEVICE  = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" if GPU

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE        = 750          # tokens per chunk
CHUNK_OVERLAP     = 150          # overlap tokens (~20%)
MIN_CHUNK_TOKENS  = 50           # discard tiny chunks

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K   = 20           # ANN candidates
RERANK_TOP_K      = 5            # final chunks after reranking
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HNSW_EF           = 128          # ef at query time (higher = more accurate)

# ── HNSW index params ──────────────────────────────────────────────────────
HNSW_M            = 32           # connections per node
HNSW_EF_CONSTRUCT = 200          # ef during construction

# ── BM25 hybrid ────────────────────────────────────────────────────────────
BM25_WEIGHT       = 0.3          # weight in RRF fusion
DENSE_WEIGHT      = 0.7

# ── LLM Providers ──────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "ollama")  # "ollama", "gemini", "openai", "anthropic"

# Gemini Settings
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI Settings
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama Settings
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3")

# Anthropic API (Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-sonnet-4-20250514"

# Common LLM settings
LLM_MAX_TOKENS    = 1024
LLM_TEMPERATURE   = 0.1          # low temp for factual answers

# ── OCR ────────────────────────────────────────────────────────────────────
OCR_DPI           = 300          # render DPI for scanned pages
OCR_LANG          = "eng"        # Tesseract language code(s), e.g. "eng+fra"
OCR_MIN_CHARS     = 100          # pages with fewer native chars trigger OCR

# ── Server ─────────────────────────────────────────────────────────────────
HOST              = "0.0.0.0"
PORT              = 8000
CORS_ORIGINS      = ["*"]        # restrict in production
