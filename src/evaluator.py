"""Quality evaluator: checks whether the expected key fact is present in the answer."""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import EmbeddingService


@dataclass
class QualityResult:
    """Evaluation outcome for a single run."""
    passed: bool
    similarity: float | None = None


class Evaluator:
    """Scores LLM answers against expected key facts."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embedding = embedding_service

    def score(self, answer: str, key_fact: str) -> QualityResult:
        """
        Evaluate whether the key fact appears in the answer.

        Primary: case-insensitive substring check.
        Secondary (optional): cosine similarity between answer and key_fact embeddings.
        """
        # Primary check
        passed = key_fact.lower() in answer.lower()

        # Optional similarity
        similarity: float | None = None
        if self._embedding is not None:
            try:
                ans_vec = self._embedding.embed(answer)
                fact_vec = self._embedding.embed(key_fact)
                similarity = EmbeddingService.cosine_similarity(ans_vec, fact_vec)
            except Exception:
                similarity = None

        return QualityResult(passed=passed, similarity=similarity)
