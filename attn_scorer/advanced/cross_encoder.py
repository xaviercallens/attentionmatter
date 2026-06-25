"""Cross-encoder re-ranking for top-K candidate refinement."""

from __future__ import annotations

import re

import numpy as np

from ..models import ScoredCandidate


class CrossEncoderReranker:
    """
    Re-ranks top-K candidates using a cross-encoder or heuristic.

    Modes:
    - "neural": sentence-transformers CrossEncoder (accurate but slower)
    - "heuristic": lexical boost re-ranking (fast, no model)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 20,
        mode: str = "heuristic",
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._mode = mode
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        scored_candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """Re-rank top-K candidates; rest retain original positions."""
        if len(scored_candidates) <= 1:
            return scored_candidates

        k = min(self._top_k, len(scored_candidates))
        top_k = scored_candidates[:k]
        rest = scored_candidates[k:]

        if self._mode == "neural":
            reranked = self._rerank_neural(query, top_k)
        else:
            reranked = self._rerank_heuristic(query, top_k)

        return reranked + rest

    def _rerank_neural(
        self, query: str, candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """Re-rank using cross-encoder neural model."""
        model = self._load_model()
        pairs = [(query, sc.candidate.text) for sc in candidates]
        scores = model.predict(pairs)
        for sc, new_score in zip(candidates, scores):
            sc.score = float(new_score)
        candidates.sort(key=lambda sc: sc.score, reverse=True)
        return candidates

    def _rerank_heuristic(
        self, query: str, candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """Lexical-boost heuristic re-ranking (no model needed)."""
        q_words = set(query.lower().split())
        q_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9-]+\b', query))
        q_numbers = set(re.findall(r'\b\d[\d-]*\d\b|\b\d+\b', query))

        for sc in candidates:
            text = sc.candidate.text
            c_words = set(text.lower().split())
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            c_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9-]+\b', text))
            entity_boost = (
                len(q_entities & c_entities) / max(len(q_entities), 1)
                if q_entities else 0
            ) * 0.2
            c_numbers = set(re.findall(r'\b\d[\d-]*\d\b|\b\d+\b', text))
            number_boost = (
                len(q_numbers & c_numbers) / max(len(q_numbers), 1)
                if q_numbers else 0
            ) * 0.2
            sc.score *= (1.0 + overlap * 0.3 + entity_boost + number_boost)

        candidates.sort(key=lambda sc: sc.score, reverse=True)
        return candidates
