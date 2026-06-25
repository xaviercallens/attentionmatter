"""
attn_scorer — Attention-inspired context scoring and selection for LLMs.

A standalone module that scores conversation context by semantic relevance
to a query, with configurable recency decay, and selects the optimal subset
within a token budget.

Usage:
    from attn_scorer import Scorer, ScorerConfig

    scorer = Scorer(ScorerConfig(decay_factor=0.95))
    result = scorer.build_context(
        query="What is my booking code?",
        messages=[...],
        memories=[...],
        token_budget=4096,
    )
    print(result.selected_texts)
    print(result.token_count)
"""

from .config import ScorerConfig
from .models import Candidate, ContextResult, ScoredCandidate
from .scorer import Scorer
from .embeddings.base import EmbeddingBackend
from .embeddings.local import LocalEmbeddingBackend
from .vector_store.base import VectorStore
from .vector_store.brute_force import BruteForceStore

__all__ = [
    "Scorer",
    "ScorerConfig",
    "Candidate",
    "ScoredCandidate",
    "ContextResult",
    "EmbeddingBackend",
    "LocalEmbeddingBackend",
    "VectorStore",
    "BruteForceStore",
]

__version__ = "1.1.0"
