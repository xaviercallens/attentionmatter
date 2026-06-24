"""In-memory STM and LTM stores simulating Redis and Couchbase."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .embedding import EmbeddingService


@dataclass
class Message:
    """A single conversation message (STM unit)."""
    text: str
    role: str  # "user" | "assistant"
    turn: int
    important: bool = False


@dataclass
class MemoryRecord:
    """A long-term memory entry (LTM unit)."""
    embedding: np.ndarray
    text: str
    source_session: str = "default"
    importance: float = 1.0


class MemoryManager:
    """Manages STM (conversation history) and LTM (long-term facts)."""

    def __init__(self, cfg: Config, embedding_service: EmbeddingService) -> None:
        self._cfg = cfg
        self._embedding = embedding_service
        self._stm: deque[Message] = deque(maxlen=cfg.stm_capacity)
        self._ltm: list[MemoryRecord] = []

    # --- STM operations ---

    def add_message(self, msg: Message) -> None:
        """Append a message to STM; oldest evicted if at capacity."""
        self._stm.append(msg)

    def get_stm(self) -> list[Message]:
        """Return STM messages in chronological order."""
        return list(self._stm)

    # --- LTM operations ---

    def insert_memory(self, text: str, source_session: str = "default",
                      importance: float = 1.0) -> None:
        """Compute embedding and store a MemoryRecord in LTM."""
        vec = self._embedding.embed(text)
        self._ltm.append(MemoryRecord(
            embedding=vec,
            text=text,
            source_session=source_session,
            importance=importance,
        ))

    def search_ltm(self, query_vec: np.ndarray, top_k: int | None = None) -> list[MemoryRecord]:
        """Brute-force cosine similarity search over LTM. Returns top-K sorted desc."""
        if not self._ltm:
            return []
        k = top_k if top_k is not None else self._cfg.ltm_top_k
        scored = []
        for rec in self._ltm:
            sim = float(np.dot(query_vec, rec.embedding))
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:k]]

    def get_all_ltm(self) -> list[MemoryRecord]:
        """Return all LTM records (used by No-Pruning baseline)."""
        return list(self._ltm)

    # --- Lifecycle ---

    def reset(self) -> None:
        """Clear both STM and LTM for the next scenario."""
        self._stm.clear()
        self._ltm.clear()
