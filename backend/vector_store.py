"""
backend/vector_store.py — Qdrant vector DB interface

Creates a collection with HNSW index (m=32, ef_construct=200).
Supports upsert, ANN search, and payload filtering.
"""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance, VectorParams, HnswConfigDiff,
    PointStruct, Filter, FieldCondition, MatchValue,
    ScalarQuantizationConfig, ScalarType, QuantizationConfig,
    ScalarQuantization,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_DIM, HNSW_M, HNSW_EF_CONSTRUCT, HNSW_EF,
    RETRIEVAL_TOP_K,
)


class VectorStore:
    """
    Wrapper around Qdrant with:
    - HNSW indexing (m=32, ef_construct=200)
    - Scalar int8 quantization for memory efficiency
    - Payload filters (pdf_id, language, page range)
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = QDRANT_COLLECTION,
    ):
        self.collection = collection
        self.client = QdrantClient(host=host, port=port, timeout=30)
        self._ensure_collection()

    # ── Collection management ─────────────────────────────────────────────

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            print(f"[VectorStore] Using existing collection '{self.collection}'")
            return

        print(f"[VectorStore] Creating collection '{self.collection}' …")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=HNSW_M,
                    ef_construct=HNSW_EF_CONSTRUCT,
                    full_scan_threshold=10_000,
                    on_disk=False,
                ),
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )
        # Create payload indexes for fast filtering
        for field in ("pdf_id", "filename", "language"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="page",
            field_schema=qmodels.PayloadSchemaType.INTEGER,
        )
        print(f"[VectorStore] Collection created with HNSW m={HNSW_M}, ef={HNSW_EF_CONSTRUCT}")

    def collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count if info.vectors_count is not None else info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }

    # ── Upsert ────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict],      # list of {chunk_id, text, embedding, **metadata}
        batch_size: int = 256,
    ):
        """
        Upsert chunks into Qdrant in batches.
        Each chunk dict must have keys: chunk_id, text, embedding (np.ndarray), 
        plus metadata: pdf_id, filename, page, tokens, language, section, ocr_used.
        """
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []
            for c in batch:
                vec = c["embedding"]
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                payload = {k: v for k, v in c.items() if k not in ("embedding",)}
                points.append(PointStruct(
                    id=self._chunk_id_to_uuid(c["chunk_id"]),
                    vector=vec,
                    payload=payload,
                ))
            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
            total += len(batch)
        return total

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = RETRIEVAL_TOP_K,
        filters: dict[str, Any] | None = None,
        ef: int = HNSW_EF,
    ) -> list[dict]:
        """
        ANN search. Returns list of dicts with payload + score.
        Optional filters: {"pdf_id": "abc", "language": "en"}
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            search_params=qmodels.SearchParams(
                hnsw_ef=ef,
                exact=False,
            ),
            with_payload=True,
            with_vectors=False,
        )

        hits = []
        for r in results:
            item = dict(r.payload)
            item["score"] = float(r.score)
            hits.append(item)
        return hits

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        # Deterministic UUID from chunk_id string
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    @staticmethod
    def _build_filter(filters: dict) -> Filter:
        conditions = []
        for key, val in filters.items():
            conditions.append(FieldCondition(
                key=key,
                match=MatchValue(value=val),
            ))
        return Filter(must=conditions)

    def get_unique_documents(self) -> list[dict]:
        """Retrieve all unique documents (filenames + pdf_ids + max pages + chunk count) from Qdrant."""
        try:
            result, _ = self.client.scroll(
                collection_name=self.collection,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            
            docs = {}
            for point in result:
                payload = point.payload
                if not payload:
                    continue
                pdf_id = payload.get("pdf_id")
                filename = payload.get("filename")
                page = payload.get("page", 1)
                
                if pdf_id and pdf_id not in docs:
                    docs[pdf_id] = {
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "pages": page,
                        "chunks_count": 1
                    }
                elif pdf_id:
                    docs[pdf_id]["pages"] = max(docs[pdf_id]["pages"], page)
                    docs[pdf_id]["chunks_count"] += 1
                    
            return list(docs.values())
        except Exception as e:
            print(f"[VectorStore] Error fetching unique documents: {e}")
            return []

    def delete_collection(self):
        self.client.delete_collection(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection).count


# ── Singleton ──────────────────────────────────────────────────────────────
_store: VectorStore | None = None

def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
