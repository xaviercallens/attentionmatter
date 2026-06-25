"""Streaming context assembly for real-time chat support."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Iterator

import numpy as np

from .config import ScorerConfig
from .embeddings.base import EmbeddingBackend
from .models import Candidate, ContextResult, ScoredCandidate
from .scorer import Scorer


@dataclass
class StreamEvent:
    """An event in the streaming context pipeline."""
    event_type: str  # "message_added", "context_updated", "budget_exceeded"
    candidate: Candidate | None = None
    context_result: ContextResult | None = None
    timestamp: float = 0.0


class StreamingContextManager:
    """
    Real-time context assembly that updates incrementally as messages arrive.

    Instead of recomputing the full selection on every query, this manager:
    1. Maintains a scored candidate buffer
    2. Incrementally updates scores as new messages arrive
    3. Emits context updates via a stream/callback interface
    4. Supports lazy recomputation (only re-score when queried)

    This is designed for chat interfaces where messages arrive one at a time
    and the context window must be kept current with minimal latency.

    Usage:
        stream = StreamingContextManager(config, embedding)
        stream.set_query("What is my booking code?")
        stream.on_message(Message(text="Your code is XYZ789", ...))
        stream.on_message(Message(text="How's the weather?", ...))
        result = stream.get_current_context(budget=4096)
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        embedding: EmbeddingBackend | None = None,
        token_counter: Callable[[str], int] | None = None,
        max_buffer_size: int = 1000,
    ) -> None:
        self._config = config or ScorerConfig()
        self._scorer = Scorer(config=self._config, embedding=embedding,
                              token_counter=token_counter)
        self._buffer: deque[ScoredCandidate] = deque(maxlen=max_buffer_size)
        self._current_query: str = ""
        self._query_vec: np.ndarray | None = None
        self._dirty: bool = True
        self._last_result: ContextResult | None = None
        self._listeners: list[Callable[[StreamEvent], None]] = []
        self._embedding = embedding

    def set_query(self, query: str) -> None:
        """Set or update the current scoring query. Marks buffer as dirty."""
        self._current_query = query
        if self._embedding:
            self._query_vec = self._embedding.embed(query)
        self._dirty = True
        self._rescore_buffer()

    def on_message(self, candidate: Candidate) -> None:
        """
        Handle a new message arriving in the conversation.

        Scores it immediately against the current query and inserts into buffer.
        """
        if not self._current_query:
            # No query set yet — just buffer it unscored
            self._buffer.append(ScoredCandidate(
                candidate=candidate, score=0.0, cosine_similarity=0.0,
                decay_multiplier=1.0, type_weight=1.0,
                token_count=self._scorer._token_counter(candidate.text),
            ))
            self._dirty = True
            return

        # Score the new candidate
        if candidate.embedding is None and self._embedding:
            candidate.embedding = self._embedding.embed(candidate.text)

        cos = 0.0
        if candidate.embedding is not None and self._query_vec is not None:
            cos = max(0.0, float(np.dot(self._query_vec, candidate.embedding)))

        decay = self._config.decay_factor ** candidate.age
        type_w = self._config.type_weights.get(candidate.ctype, 1.0)
        score = cos * decay * type_w
        tc = self._scorer._token_counter(candidate.text)

        sc = ScoredCandidate(
            candidate=candidate, score=score, cosine_similarity=cos,
            decay_multiplier=decay, type_weight=type_w, token_count=tc,
        )
        self._buffer.append(sc)
        self._dirty = True

        # Age all existing candidates by 1
        for existing in self._buffer:
            if existing is not sc:
                existing.candidate.age += 1
                existing.decay_multiplier = self._config.decay_factor ** existing.candidate.age
                existing.score = (existing.cosine_similarity *
                                  existing.decay_multiplier * existing.type_weight)

        self._emit(StreamEvent(
            event_type="message_added", candidate=candidate,
            timestamp=time.time(),
        ))

    def get_current_context(self, budget: int | None = None) -> ContextResult:
        """
        Get the current optimal context selection.

        Re-sorts and selects from the buffer. Caches result until buffer changes.
        """
        if not self._dirty and self._last_result is not None:
            return self._last_result

        token_budget = budget or self._config.default_token_budget

        # Sort buffer by score
        sorted_buffer = sorted(self._buffer, key=lambda s: s.score, reverse=True)

        # Select within budget
        selected = []
        omitted = []
        total = 0
        for sc in sorted_buffer:
            if total + sc.token_count <= token_budget:
                selected.append(sc)
                total += sc.token_count
            else:
                omitted.append(sc)

        # Re-order selected chronologically
        history = sorted(
            [s for s in selected if s.candidate.ctype != "memory"],
            key=lambda s: s.candidate.turn,
        )
        memories = [s for s in selected if s.candidate.ctype == "memory"]
        selected = history + memories

        result = ContextResult(
            selected=selected, omitted=omitted,
            token_count=total,
            budget_used=total / token_budget if token_budget > 0 else 0,
        )

        self._last_result = result
        self._dirty = False

        self._emit(StreamEvent(
            event_type="context_updated", context_result=result,
            timestamp=time.time(),
        ))
        return result

    def on_listener(self, callback: Callable[[StreamEvent], None]) -> None:
        """Register a listener for stream events."""
        self._listeners.append(callback)

    def _emit(self, event: StreamEvent) -> None:
        for listener in self._listeners:
            listener(event)

    def _rescore_buffer(self) -> None:
        """Re-score all buffered candidates against the new query."""
        if not self._query_vec is not None:
            return
        for sc in self._buffer:
            if sc.candidate.embedding is not None and self._query_vec is not None:
                sc.cosine_similarity = max(
                    0.0, float(np.dot(self._query_vec, sc.candidate.embedding))
                )
                sc.decay_multiplier = self._config.decay_factor ** sc.candidate.age
                sc.type_weight = self._config.type_weights.get(sc.candidate.ctype, 1.0)
                sc.score = sc.cosine_similarity * sc.decay_multiplier * sc.type_weight

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._current_query = ""
        self._query_vec = None
        self._dirty = True
        self._last_result = None
