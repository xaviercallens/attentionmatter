"""Data models for the attn_scorer module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Candidate:
    """A piece of context eligible for inclusion in the prompt."""

    text: str
    ctype: str = "history"  # "history", "memory", "fact", "chit_chat"
    age: int = 0  # 0 = most recent; higher = older
    turn: int = -1  # original turn index (-1 for memories)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class ScoredCandidate:
    """A candidate with its computed relevance score."""

    candidate: Candidate
    score: float
    cosine_similarity: float
    decay_multiplier: float
    type_weight: float
    token_count: int = 0


@dataclass
class ContextResult:
    """Output of context selection."""

    selected: list[ScoredCandidate]
    omitted: list[ScoredCandidate]
    token_count: int
    budget_used: float  # ratio of budget consumed (0-1)
    scoring_time_ms: float = 0.0

    @property
    def selected_texts(self) -> list[str]:
        """Return just the text of selected items."""
        return [sc.candidate.text for sc in self.selected]

    @property
    def reduction_vs_full(self) -> float:
        """Token reduction percentage vs including everything."""
        full = self.token_count + sum(sc.token_count for sc in self.omitted)
        if full == 0:
            return 0.0
        return (1 - self.token_count / full) * 100
