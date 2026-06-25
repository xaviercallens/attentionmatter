"""Multi-modal scorer: handles scoring across content types."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import ScorerConfig
from ..embeddings.base import EmbeddingBackend
from ..models import Candidate, ContextResult, ScoredCandidate
from ..scorer import Scorer
from .extractors import (
    CodeExtractor, ContentExtractor, ImageExtractor,
    StructuredDataExtractor, TableExtractor,
)
from .types import ModalityType, MultiModalCandidate


# Default modality weights (adjustable)
DEFAULT_MODALITY_WEIGHTS: dict[str, float] = {
    ModalityType.TEXT.value: 1.0,
    ModalityType.CODE.value: 1.1,     # code is often important
    ModalityType.TABLE.value: 1.05,   # tables contain structured facts
    ModalityType.IMAGE.value: 0.7,    # images have less precise embeddings
    ModalityType.STRUCTURED.value: 1.1,
}


class MultiModalScorer:
    """
    Extends the base Scorer to handle multi-modal candidates.

    Each modality has a content extractor that produces embeddable text,
    and a weight that adjusts its relevance score.

    Usage:
        scorer = MultiModalScorer(config, embedding)
        scorer.add_candidate(MultiModalCandidate(
            text_repr="...", raw_content="def foo(): ...",
            modality=ModalityType.CODE, language="python",
        ))
        result = scorer.score_and_select(query, budget=4096)
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        embedding: EmbeddingBackend | None = None,
        token_counter: Callable[[str], int] | None = None,
        modality_weights: dict[str, float] | None = None,
    ) -> None:
        self._config = config or ScorerConfig()
        self._scorer = Scorer(config=self._config, embedding=embedding,
                              token_counter=token_counter)
        self._modality_weights = modality_weights or DEFAULT_MODALITY_WEIGHTS
        self._extractors: dict[ModalityType, ContentExtractor] = {
            ModalityType.CODE: CodeExtractor(),
            ModalityType.TABLE: TableExtractor(),
            ModalityType.IMAGE: ImageExtractor(),
            ModalityType.STRUCTURED: StructuredDataExtractor(),
        }
        self._candidates: list[MultiModalCandidate] = []

    def register_extractor(self, modality: ModalityType, extractor: ContentExtractor):
        """Register a custom extractor for a modality."""
        self._extractors[modality] = extractor

    def add_candidate(self, candidate: MultiModalCandidate) -> None:
        """Add a multi-modal candidate for scoring."""
        # Auto-extract text representation if not provided
        if not candidate.text_repr and candidate.modality in self._extractors:
            extractor = self._extractors[candidate.modality]
            candidate.text_repr = extractor.extract_text(
                candidate.raw_content, candidate.metadata
            )
            if candidate.token_count == 0:
                candidate.token_count = extractor.estimate_tokens(candidate.raw_content)
        self._candidates.append(candidate)

    def add_candidates(self, candidates: list[MultiModalCandidate]) -> None:
        """Add multiple candidates."""
        for c in candidates:
            self.add_candidate(c)

    def score_and_select(
        self, query: str, token_budget: int | None = None,
    ) -> ContextResult:
        """
        Score all multi-modal candidates and select within budget.

        Returns ContextResult with selected items ordered appropriately.
        """
        budget = token_budget or self._config.default_token_budget

        # Convert MultiModalCandidates to base Candidates for scoring
        base_candidates = []
        for mmc in self._candidates:
            ctype = "memory" if mmc.modality == ModalityType.IMAGE else "history"
            base_candidates.append(Candidate(
                text=mmc.text_repr,
                ctype=ctype,
                age=mmc.age,
                turn=mmc.turn,
                metadata={
                    "modality": mmc.modality.value,
                    "prompt_text": mmc.prompt_text,
                    **mmc.metadata,
                },
                embedding=mmc.embedding,
            ))

        # Score using the base scorer
        scored = self._scorer.score(query, base_candidates)

        # Apply modality weights
        for sc in scored:
            modality = sc.candidate.metadata.get("modality", "text")
            weight = self._modality_weights.get(modality, 1.0)
            sc.score *= weight
            sc.type_weight *= weight

        # Re-sort after weight adjustment
        scored.sort(key=lambda s: s.score, reverse=True)

        # Select within budget (using prompt_text token counts)
        selected: list[ScoredCandidate] = []
        omitted: list[ScoredCandidate] = []
        total_tokens = 0
        counter = self._scorer._token_counter

        for sc in scored:
            prompt_text = sc.candidate.metadata.get("prompt_text", sc.candidate.text)
            tc = counter(prompt_text)
            sc.token_count = tc
            if total_tokens + tc <= budget:
                selected.append(sc)
                total_tokens += tc
            else:
                omitted.append(sc)

        budget_used = total_tokens / budget if budget > 0 else 0.0

        return ContextResult(
            selected=selected,
            omitted=omitted,
            token_count=total_tokens,
            budget_used=budget_used,
        )

    def clear(self) -> None:
        """Clear all stored candidates."""
        self._candidates.clear()

    @property
    def candidates(self) -> list[MultiModalCandidate]:
        return list(self._candidates)
