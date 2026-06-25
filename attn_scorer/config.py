"""Configuration for the attn_scorer module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScorerConfig:
    """All tunable parameters for the scorer."""

    # Scoring
    decay_factor: float = 0.95
    memory_age: int = 0  # LTM entries: no decay penalty (already relevance-filtered)

    # Budget
    default_token_budget: int = 4096

    # Dynamic budgeting
    dynamic_budget_enabled: bool = False
    budget_min_ratio: float = 0.3  # minimum budget as ratio of max
    budget_max_ratio: float = 0.9  # maximum budget as ratio of max
    max_context_tokens: int = 8192

    # Retrieval
    ltm_top_k: int = 5

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "local"  # "local", "openai", "cohere"

    # OpenAI / Cohere (when embedding_backend != "local")
    api_key: str = ""
    api_model: str = ""  # e.g. "text-embedding-3-small" for OpenAI

    # Vector store
    vector_store_backend: str = "brute_force"  # "brute_force", "faiss"

    # Performance
    batch_size: int = 64
    cache_embeddings: bool = True

    # Type weighting (multipliers applied after cosine × decay)
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "memory": 1.2,   # boost persistent knowledge
        "fact": 1.1,     # boost factual statements
        "history": 1.0,  # normal weight for conversation
        "chit_chat": 0.8,  # penalize small talk
    })

    @property
    def budget_min(self) -> int:
        return int(self.max_context_tokens * self.budget_min_ratio)

    @property
    def budget_max(self) -> int:
        return int(self.max_context_tokens * self.budget_max_ratio)
