"""
backend/embedder.py — Embedding model wrapper

Uses BAAI/bge-base-en-v1.5 (768-dim, free/OSS, MIT license).
BGE models use a query prefix for retrieval; we handle that here.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH, EMBEDDING_DEVICE
)


class Embedder:
    """
    Thread-safe embedding wrapper.
    BGE models require the prefix "Represent this sentence: " for passages
    and "query: " for queries (asymmetric retrieval).
    """

    QUERY_PREFIX   = "Represent this question for searching relevant passages: "
    PASSAGE_PREFIX = ""    # BGE-base doesn't need prefix for passages

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        print(f"[Embedder] Loading {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim   = EMBEDDING_DIM
        print(f"[Embedder] Ready - dim={self.dim}")

    # ── Public API ────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns L2-normalised float32 array."""
        prefixed = self.QUERY_PREFIX + text
        vec = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of passages in batches.
        Returns shape (N, dim) float32 array, L2-normalised.
        """
        prefixed = [self.PASSAGE_PREFIX + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Alias — embed one passage (not a query)."""
        return self.embed_passages([text], show_progress=False)[0]


# ── Module-level singleton (lazy) ──────────────────────────────────────────
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
