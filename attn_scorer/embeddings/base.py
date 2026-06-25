"""Base interface for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """Abstract base for all embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalized embedding vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch-embed texts. Returns shape (n, dim)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit-normalized vectors."""
        return float(np.dot(a, b))
