"""Multi-query context scorer — scores against a window of recent queries."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..embeddings.base import EmbeddingBackend
from ..models import Candidate


class MultiQueryScorer:
    """
    Scores candidates against the last N queries (not just latest).
    Uses max or weighted-avg aggregation across the query window.
    """

    def __init__(
        self, embedding: EmbeddingBackend, window_size: int = 3,
        aggregation: str = "max", recency_weight: float = 0.7,
    ):
        self._embedding = embedding
        self._window_size = window_size
        self._aggregation = aggregation
        self._recency_weight = recency_weight
        self._query_window: deque[str] = deque(maxlen=window_size)
        self._query_vecs: deque[np.ndarray] = deque(maxlen=window_size)

    def push_query(self, query: str) -> None:
        self._query_window.append(query)
        self._query_vecs.append(self._embedding.embed(query))

    def score_candidate(self, candidate: Candidate) -> float:
        if not self._query_vecs:
            return 0.0
        if candidate.embedding is None:
            candidate.embedding = self._embedding.embed(candidate.text)
        sims = [max(0.0, float(np.dot(qv, candidate.embedding))) for qv in self._query_vecs]
        if self._aggregation == "max":
            return max(sims)
        # weighted_avg
        n = len(sims)
        weights = [self._recency_weight ** (n - 1 - i) for i in range(n)]
        return sum(s * w for s, w in zip(sims, weights)) / sum(weights)

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        return [self.score_candidate(c) for c in candidates]

    @property
    def window(self) -> list[str]:
        return list(self._query_window)

    def clear(self) -> None:
        self._query_window.clear()
        self._query_vecs.clear()
