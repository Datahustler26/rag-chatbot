"""
backend/server.py — FastAPI server

Endpoints:
  POST /api/chat          — RAG query endpoint
  POST /api/ingest        — Ingest a new PDF
  GET  /api/status        — Collection stats
  GET  /                  — Serve frontend
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CORS_ORIGINS, HOST, PORT
from backend.embedder import get_embedder
from backend.vector_store import get_vector_store
from backend.retriever import get_retriever
from backend.generator import get_generator
from backend.ingest import ingest_pdf

app = FastAPI(title="RAGCore", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Pydantic schemas ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str = Field(..., min_length=1, max_length=1000)
    filters:    dict | None = None   # e.g. {"pdf_id": "abc123"}
    top_k:      int = Field(5, ge=1, le=20)

class ChatResponse(BaseModel):
    answer:              str
    citations:           list[dict]
    sources:             list[dict]
    total_time_ms:       float
    retrieval_time_ms:   float
    gen_time_ms:         float


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    # Warm up all singletons
    print("[Server] Warming up models...")
    embedder = get_embedder()
    vs = get_vector_store()
    retriever = get_retriever()
    generator = get_generator()
    
    # Warm up and build BM25 index from existing database points
    try:
        result, _ = vs.client.scroll(
            collection_name=vs.collection,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        chunks = []
        for point in result:
            payload = point.payload
            if payload and "chunk_id" in payload and "text" in payload:
                chunks.append({
                    "chunk_id": payload["chunk_id"],
                    "text": payload["text"]
                })
        if chunks:
            retriever.load_bm25_corpus(chunks)
            print(f"[Server] Rebuilt BM25 index with {len(chunks)} existing chunks.")
    except Exception as e:
        print(f"[Server] Warning: Failed to restore BM25 index on startup: {e}")
        
    print("[Server] Ready [OK]")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "RAGCore running", "docs": "/api/docs"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main RAG query endpoint."""
    try:
        retriever = get_retriever()
        generator = get_generator()

        retrieval = retriever.retrieve(
            query=req.question,
            rerank_k=req.top_k,
            filters=req.filters,
        )

        if not retrieval.chunks:
            return ChatResponse(
                answer="No relevant documents found for your question.",
                citations=[],
                sources=[],
                total_time_ms=retrieval.total_time_ms,
                retrieval_time_ms=retrieval.total_time_ms,
                gen_time_ms=0.0,
            )

        result = generator.generate(question=req.question, retrieval=retrieval)
        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    """Ingest a single PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        # Ingest
        meta = ingest_pdf(tmp_path)
        
        # Override temporary filename with original uploaded filename
        meta.filename = file.filename
        for c in meta.chunks:
            c.filename = file.filename

        # Embed & upsert
        embedder     = get_embedder()
        vector_store = get_vector_store()

        texts = [c.text for c in meta.chunks]
        vecs  = embedder.embed_passages(texts, show_progress=True)

        chunk_dicts = []
        for c, vec in zip(meta.chunks, vecs):
            d = c.__dict__.copy()
            d["embedding"] = vec
            chunk_dicts.append(d)

        upserted = vector_store.upsert_chunks(chunk_dicts)

        # Rebuild BM25
        retriever = get_retriever()
        all_payload = [
            {"chunk_id": c["chunk_id"], "text": c["text"]}
            for c in chunk_dicts
        ]
        retriever.load_bm25_corpus(all_payload)

        tmp_path.unlink(missing_ok=True)

        return {
            "status":    "ok",
            "filename":  file.filename,
            "pdf_id":    meta.pdf_id,
            "pages":     meta.pages,
            "chunks":    len(meta.chunks),
            "upserted":  upserted,
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/status")
async def status():
    """Collection health and stats."""
    try:
        vs   = get_vector_store()
        info = vs.collection_info()
        return {
            "status":     "ok",
            "collection": vs.collection,
            **info,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/documents")
async def get_documents():
    """Retrieve unique indexed files from the vector store."""
    try:
        vs = get_vector_store()
        return vs.get_unique_documents()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/collection")
async def reset_collection():
    """Delete and recreate the collection (destructive!)."""
    vs = get_vector_store()
    vs.delete_collection()
    vs._ensure_collection()
    return {"status": "reset"}


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host=HOST, port=PORT, reload=True)
