"""Brute-force vector store using numpy cosine similarity."""

from __future__ import annotations

import numpy as np

from .base import SearchResult, VectorStore


class BruteForceStore(VectorStore):
    """
    In-memory brute-force vector search.
    Efficient for up to ~5k entries. Use FAISS for larger stores.
    """

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def add(self, text: str, embedding: np.ndarray, metadata: dict | None = None) -> int:
        idx = len(self._embeddings)
        self._embeddings.append(embedding)
        self._texts.append(text)
        self._metadata.append(metadata or {})
        return idx

    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        if not self._embeddings:
            return []

        # Stack all embeddings and compute similarities in one op
        matrix = np.array(self._embeddings, dtype=np.float32)
        similarities = matrix @ query_vec.astype(np.float32)

        # Get top-K indices
        k = min(top_k, len(self._embeddings))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            SearchResult(
                index=int(idx),
                score=float(similarities[idx]),
                text=self._texts[idx],
                metadata=self._metadata[idx],
            )
            for idx in top_indices
        ]

    def size(self) -> int:
        return len(self._embeddings)

    def clear(self) -> None:
        self._embeddings.clear()
        self._texts.clear()
        self._metadata.clear()
