"""Central configuration for the Adaptive Attention Token Reduction PoC."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass
class Config:
    """All tunable parameters for the PoC pipeline."""

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_4bit: bool = True
    max_new_tokens: int = 256

    # Context budget
    max_context_tokens: int = 8192
    token_budget_ratio: float = 0.8

    # Scoring
    decay_factor: float = 0.95
    large_age_offset: int = 50  # LTM treated as ~50 turns older than conversation end

    # Sliding window
    sliding_window_messages: int = 4

    # LTM retrieval
    ltm_top_k: int = 5

    # A3TK heuristic
    summarization_threshold_tokens: int = 512
    importance_keywords: list[str] = field(default_factory=lambda: [
        "booking", "code", "reference", "number", "account", "name",
        "preference", "vegetarian", "vegan", "allergy", "address",
        "phone", "email", "flight", "confirmation", "password", "id",
    ])

    # STM capacity
    stm_capacity: int = 200

    # Reproducibility
    random_seed: int = 42

    # Output
    results_path: str = "results/poc_results.csv"

    @property
    def token_budget(self) -> int:
        """Derived token budget leaving headroom for the response."""
        return int(self.max_context_tokens * self.token_budget_ratio)

    def seed_everything(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch is not None:
            torch.manual_seed(self.random_seed)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load config overrides from a JSON file."""
        with open(path) as f:
            overrides = json.load(f)
        return cls(**{k: v for k, v in overrides.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> Config:
        """Build config with environment variable overrides."""
        kwargs: dict = {}
        env_map = {
            "POC_EMBEDDING_MODEL": ("embedding_model", str),
            "POC_LLM_MODEL": ("llm_model", str),
            "POC_USE_4BIT": ("use_4bit", lambda v: v.lower() in ("1", "true", "yes")),
            "POC_MAX_CONTEXT_TOKENS": ("max_context_tokens", int),
            "POC_TOKEN_BUDGET_RATIO": ("token_budget_ratio", float),
            "POC_DECAY_FACTOR": ("decay_factor", float),
            "POC_SLIDING_WINDOW": ("sliding_window_messages", int),
            "POC_LTM_TOP_K": ("ltm_top_k", int),
            "POC_RANDOM_SEED": ("random_seed", int),
            "POC_RESULTS_PATH": ("results_path", str),
        }
        for env_key, (field_name, converter) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                kwargs[field_name] = converter(val)
        return cls(**kwargs)
