"""Shared test fixtures."""

import numpy as np
import pytest

from attn_scorer import Candidate, Scorer, ScorerConfig
from attn_scorer.embeddings.base import EmbeddingBackend


class MockEmbeddingBackend(EmbeddingBackend):
    """Deterministic mock embedding for tests."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**32))
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def mock_embedding():
    return MockEmbeddingBackend()


@pytest.fixture
def config():
    return ScorerConfig(decay_factor=0.95, default_token_budget=100)


@pytest.fixture
def scorer(config, mock_embedding):
    return Scorer(
        config=config,
        embedding=mock_embedding,
        token_counter=lambda t: len(t.split()),
    )


@pytest.fixture
def sample_candidates():
    """A set of candidates with known properties."""
    return [
        Candidate(text="Your booking code is XYZ789.", ctype="fact", age=50, turn=5),
        Candidate(text="Nice weather today.", ctype="chit_chat", age=10, turn=45),
        Candidate(text="It is sunny.", ctype="chit_chat", age=9, turn=46),
        Candidate(text="Anything else?", ctype="history", age=1, turn=54),
        Candidate(text="Booking code XYZ789 flight AB123", ctype="memory", age=0, turn=-1),
    ]
