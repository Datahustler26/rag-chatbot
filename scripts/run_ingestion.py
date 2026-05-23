"""
scripts/run_ingestion.py — CLI: ingest a folder of PDFs into Qdrant

Usage:
  python scripts/run_ingestion.py --pdf_dir ./pdfs
  python scripts/run_ingestion.py --pdf_dir ./pdfs --collection my_docs --reset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.embedder import get_embedder
from backend.ingest import ingest_directory
from backend.retriever import get_retriever
from backend.vector_store import VectorStore
from config.settings import QDRANT_COLLECTION


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into RAGCore vector store")
    parser.add_argument("--pdf_dir",    required=True, help="Directory containing PDF files")
    parser.add_argument("--collection", default=QDRANT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--reset",      action="store_true", help="Delete and recreate collection before ingesting")
    parser.add_argument("--batch",      type=int, default=256, help="Upsert batch size")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"[ERROR] Directory not found: {pdf_dir}")
        sys.exit(1)

    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs found in {pdf_dir}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  RAGCore Ingestion Pipeline")
    print(f"{'='*55}")
    print(f"  PDFs found : {len(pdfs)}")
    print(f"  Collection : {args.collection}")
    print(f"  Reset      : {args.reset}")
    print(f"{'='*55}\n")

    # Setup
    print("[1/5] Initializing vector store ...")
    vs = VectorStore(collection=args.collection)
    if args.reset:
        print("  WARNING: Resetting collection ...")
        vs.delete_collection()
        vs._ensure_collection()

    print("[2/5] Loading embedding model ...")
    embedder = get_embedder()

    # Ingest
    print("[3/5] Extracting & chunking PDFs ...")
    t0 = time.perf_counter()
    all_chunks = []
    total_pages = 0

    for meta in ingest_directory(pdf_dir):
        print(f"  [OK]  {meta.filename:40s} {meta.pages:4d} pages -> {len(meta.chunks):5d} chunks")
        all_chunks.extend(meta.chunks)
        total_pages += meta.pages

    extract_time = time.perf_counter() - t0
    print(f"\n  Total: {len(pdfs)} PDFs | {total_pages} pages | {len(all_chunks)} chunks")
    print(f"  Extraction time: {extract_time:.1f}s\n")

    # Embed
    print("[4/5] Embedding chunks ...")
    t1 = time.perf_counter()
    texts = [c.text for c in all_chunks]
    vecs  = embedder.embed_passages(texts, show_progress=True)
    embed_time = time.perf_counter() - t1
    print(f"  Embedding time: {embed_time:.1f}s  ({len(texts)/embed_time:.0f} chunks/sec)\n")

    # Upsert
    print("[5/5] Upserting into Qdrant ...")
    t2 = time.perf_counter()

    chunk_dicts = []
    for c, vec in zip(all_chunks, vecs):
        d = c.__dict__.copy()
        d["embedding"] = vec
        chunk_dicts.append(d)

    total_upserted = vs.upsert_chunks(chunk_dicts, batch_size=args.batch)
    upsert_time = time.perf_counter() - t2

    print(f"  Upserted: {total_upserted} chunks in {upsert_time:.1f}s\n")

    # Build BM25 for hybrid retrieval
    print("[+] Building BM25 index ...")
    retriever = get_retriever()
    bm25_corpus = [{"chunk_id": c.chunk_id, "text": c.text} for c in all_chunks]
    retriever.load_bm25_corpus(bm25_corpus)

    total_time = time.perf_counter() - t0
    info = vs.collection_info()

    print(f"\n{'='*55}")
    print(f"  [SUCCESS] Ingestion complete!")
    print(f"  Total time  : {total_time:.1f}s")
    print(f"  Chunks in DB: {info['vectors_count']}")
    print(f"  Collection  : {args.collection}")
    print(f"{'='*55}\n")
    print("  Start the server:")
    print("    uvicorn backend.server:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
