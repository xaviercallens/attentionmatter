"""FAISS-based vector store for efficient ANN search at scale (10k+ entries)."""

from __future__ import annotations

import numpy as np

from .base import SearchResult, VectorStore


class FAISSStore(VectorStore):
    """
    FAISS vector store for efficient approximate nearest neighbor search.
    Supports 10k-10M+ entries with sub-millisecond search latency.

    Uses IndexFlatIP (inner product on normalized vectors = cosine similarity)
    for exact search on smaller stores, and IndexIVFFlat for ANN on larger stores.
    """

    def __init__(self, dimension: int, use_ivf: bool = False, nlist: int = 100) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu required for FAISS vector store. "
                "Install with: pip install faiss-cpu"
            )

        self._dim = dimension
        self._use_ivf = use_ivf
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._is_trained = False

        if use_ivf:
            quantizer = faiss.IndexFlatIP(dimension)
            self._index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            self._index = faiss.IndexFlatIP(dimension)
            self._is_trained = True

    def add(self, text: str, embedding: np.ndarray, metadata: dict | None = None) -> int:
        idx = len(self._texts)
        vec = embedding.astype(np.float32).reshape(1, -1)

        if self._use_ivf and not self._is_trained:
            # IVF requires training before adding; buffer until trained
            self._texts.append(text)
            self._metadata.append(metadata or {})
            return idx

        self._index.add(vec)
        self._texts.append(text)
        self._metadata.append(metadata or {})
        return idx

    def train(self, embeddings: np.ndarray) -> None:
        """Train the IVF index (required before searching if use_ivf=True)."""
        if self._use_ivf and not self._is_trained:
            import faiss
            vecs = embeddings.astype(np.float32)
            self._index.train(vecs)
            self._index.add(vecs)
            self._is_trained = True

    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query = query_vec.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(
                index=int(idx),
                score=float(score),
                text=self._texts[idx],
                metadata=self._metadata[idx],
            ))
        return results

    def size(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        import faiss
        if self._use_ivf:
            quantizer = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIVFFlat(quantizer, self._dim, 100)
            self._is_trained = False
        else:
            self._index = faiss.IndexFlatIP(self._dim)
        self._texts.clear()
        self._metadata.clear()
