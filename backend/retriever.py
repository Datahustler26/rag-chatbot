"""
backend/retriever.py — Hybrid retrieval pipeline

1. Dense ANN search (Qdrant HNSW, top-20)
2. BM25 sparse retrieval (in-memory, top-20)
3. Reciprocal Rank Fusion (RRF) merge
4. Cross-encoder reranking (MiniLM, top-5)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    RETRIEVAL_TOP_K, RERANK_TOP_K, RERANK_MODEL,
    BM25_WEIGHT, DENSE_WEIGHT,
)
from backend.embedder import get_embedder
from backend.vector_store import get_vector_store


# ── BM25 (lightweight in-memory implementation) ────────────────────────────
class BM25:
    """
    Okapi BM25 scorer over a fixed corpus.
    Rebuilt on first query after ingestion.
    """
    K1 = 1.5
    B  = 0.75

    def __init__(self):
        self._corpus: list[dict]  = []
        self._tf:     list[dict]  = []
        self._df:     dict        = defaultdict(int)
        self._avgdl:  float       = 0.0
        self._built:  bool        = False

    def build(self, chunks: list[dict]):
        """chunks: list of {chunk_id, text, ...}"""
        self._corpus = chunks
        self._tf     = []
        self._df     = defaultdict(int)

        for c in chunks:
            tokens = c["text"].lower().split()
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self._tf.append(tf)
            for t in set(tokens):
                self._df[t] += 1

        self._avgdl = sum(len(c["text"].split()) for c in chunks) / max(len(chunks), 1)
        self._built = True

    def score(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[int, float]]:
        """Return (chunk_index, bm25_score) sorted desc."""
        if not self._built:
            return []
        q_tokens = query.lower().split()
        N = len(self._corpus)
        scores = []
        for idx, tf in enumerate(self._tf):
            dl  = sum(tf.values())
            s   = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log((N - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1)
                tfd = tf[t] * (self.K1 + 1) / (tf[t] + self.K1 * (1 - self.B + self.B * dl / self._avgdl))
                s  += idf * tfd
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ── RRF fusion ─────────────────────────────────────────────────────────────
def rrf_merge(
    dense_hits:  list[dict],
    sparse_hits: list[tuple[int, float]],
    corpus:      list[dict],
    k:           int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of dense and sparse result lists.
    Returns merged list sorted by RRF score.
    """
    rrf: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, 1):
        cid = hit["chunk_id"]
        rrf[cid]      += DENSE_WEIGHT * (1.0 / (k + rank))
        chunk_map[cid] = hit

    for rank, (idx, _) in enumerate(sparse_hits, 1):
        c   = corpus[idx]
        cid = c["chunk_id"]
        rrf[cid]      += BM25_WEIGHT * (1.0 / (k + rank))
        if cid not in chunk_map:
            chunk_map[cid] = c

    merged = sorted(chunk_map.values(), key=lambda c: -rrf[c["chunk_id"]])
    for c in merged:
        c["rrf_score"] = rrf[c["chunk_id"]]
    return merged


# ── Cross-encoder reranker ─────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        print(f"[Reranker] Loading {model_name} ...")
        self.model = CrossEncoder(model_name, max_length=512)
        print("[Reranker] Ready")

    def rerank(self, query: str, chunks: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        if not chunks:
            return []
        pairs  = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
        result = []
        for chunk, score in ranked[:top_k]:
            chunk = dict(chunk)
            chunk["rerank_score"] = float(score)
            result.append(chunk)
        return result


# ── Main Retriever ─────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    chunks:        list[dict]
    dense_time_ms:  float
    rerank_time_ms: float
    total_time_ms:  float
    query:         str


class Retriever:
    def __init__(self):
        self.embedder     = get_embedder()
        self.vector_store = get_vector_store()
        self.bm25         = BM25()
        self.reranker     = Reranker()
        self._corpus_cache: list[dict] = []

    def load_bm25_corpus(self, chunks: list[dict]):
        """Call after ingestion to populate BM25 index."""
        self._corpus_cache = chunks
        self.bm25.build(chunks)
        print(f"[Retriever] BM25 built over {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int     = RETRIEVAL_TOP_K,
        rerank_k: int  = RERANK_TOP_K,
        filters: dict  | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()

        # 1. Embed query
        q_vec = self.embedder.embed_query(query)

        # 2. Dense ANN search
        dense_hits = self.vector_store.search(
            query_vector=q_vec,
            top_k=top_k,
            filters=filters,
        )
        t_dense = (time.perf_counter() - t0) * 1000

        # 3. BM25 sparse search (if corpus loaded)
        sparse_hits = self.bm25.score(query, top_k=top_k)

        # 4. RRF fusion
        if sparse_hits and self._corpus_cache:
            merged = rrf_merge(dense_hits, sparse_hits, self._corpus_cache)
        else:
            merged = dense_hits

        # 5. Cross-encoder rerank
        t_r0 = time.perf_counter()
        final = self.reranker.rerank(query, merged[:top_k], top_k=rerank_k)
        t_rerank = (time.perf_counter() - t_r0) * 1000

        total = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            chunks=final,
            dense_time_ms=round(t_dense, 1),
            rerank_time_ms=round(t_rerank, 1),
            total_time_ms=round(total, 1),
            query=query,
        )


# ── Singleton ──────────────────────────────────────────────────────────────
_retriever: Retriever | None = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
