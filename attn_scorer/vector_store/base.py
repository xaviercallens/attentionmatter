"""Base interface for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    """A single vector search result."""
    index: int
    score: float
    text: str
    metadata: dict


class VectorStore(ABC):
    """Abstract base for vector stores (memory retrieval)."""

    @abstractmethod
    def add(self, text: str, embedding: np.ndarray, metadata: dict | None = None) -> int:
        """Add an entry. Returns the assigned index."""
        ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        """Search for the top-K most similar entries."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Return the number of stored entries."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        ...
