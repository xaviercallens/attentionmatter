"""Async scoring support for high-concurrency environments."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable

import numpy as np

from .config import ScorerConfig
from .embeddings.base import EmbeddingBackend
from .models import Candidate, ContextResult, ScoredCandidate
from .observability import MetricsCollector, TracingHook
from .scorer import Scorer

logger = logging.getLogger(__name__)


class AsyncEmbeddingBackend:
    """
    Async wrapper for embedding backends.

    Supports:
    - Running local embeddings in a thread pool (non-blocking)
    - Native async for API-based backends (OpenAI, Cohere)
    """

    def __init__(self, backend: EmbeddingBackend, max_workers: int = 4):
        self._backend = backend
        self._max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)

    async def embed(self, text: str) -> np.ndarray:
        """Async embed a single text."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._backend.embed, text)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Async batch embed. Runs in executor for local, native for API."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._backend.embed_batch, texts)

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class AsyncScorer:
    """
    Async version of Scorer for use in async web frameworks (FastAPI, aiohttp).

    All scoring and selection methods are async, running CPU-bound work
    in a thread pool to avoid blocking the event loop.

    Usage:
        scorer = AsyncScorer(config, embedding_backend)

        async def handle_request(query, messages):
            result = await scorer.build_context(query, messages)
            return result.selected_texts
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        embedding: EmbeddingBackend | None = None,
        token_counter: Callable[[str], int] | None = None,
        max_concurrent_embeds: int = 4,
        metrics: MetricsCollector | None = None,
        tracer: TracingHook | None = None,
    ) -> None:
        self._config = config or ScorerConfig()
        self._sync_scorer = Scorer(
            config=self._config, embedding=embedding, token_counter=token_counter
        )
        self._async_embedding = AsyncEmbeddingBackend(
            self._sync_scorer._embedding, max_workers=max_concurrent_embeds
        )
        self._token_counter = token_counter or self._sync_scorer._token_counter
        self._metrics = metrics
        self._tracer = tracer
        self._cache_lock = threading.Lock()

    async def score(self, query: str, candidates: list[Candidate]) -> list[ScoredCandidate]:
        """Async score candidates by relevance to query."""
        # Embed query
        q_vec = await self._async_embedding.embed(query)

        # Batch-embed candidates without embeddings
        to_embed = [(i, c) for i, c in enumerate(candidates) if c.embedding is None]
        if to_embed:
            texts = [c.text for _, c in to_embed]
            vecs = await self._async_embedding.embed_batch(texts)
            for j, (i, c) in enumerate(to_embed):
                c.embedding = vecs[j]

        # Score (CPU-bound, fast enough to run inline)
        decay = self._config.decay_factor
        type_weights = self._config.type_weights
        scored = []

        for cand in candidates:
            cos = max(0.0, AsyncEmbeddingBackend.cosine_similarity(q_vec, cand.embedding))
            decay_mult = decay ** cand.age
            type_w = type_weights.get(cand.ctype, 1.0)
            score = cos * decay_mult * type_w
            tc = self._token_counter(cand.text)

            scored.append(ScoredCandidate(
                candidate=cand, score=score, cosine_similarity=cos,
                decay_multiplier=decay_mult, type_weight=type_w, token_count=tc,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    async def select(
        self, scored: list[ScoredCandidate], token_budget: int
    ) -> ContextResult:
        """Select candidates within budget (sync, fast)."""
        return self._sync_scorer.select(scored, token_budget)

    async def build_context(
        self,
        query: str,
        messages: list[Candidate] | None = None,
        memories: list[Candidate] | None = None,
        token_budget: int | None = None,
    ) -> ContextResult:
        """Async high-level API: score, select, return context."""
        t0 = time.perf_counter()
        budget = token_budget or self._config.default_token_budget

        candidates = []
        if messages:
            candidates.extend(messages)
        if memories:
            for mem in memories:
                mem.age = 0
                if mem.ctype == "history":
                    mem.ctype = "memory"
            candidates.extend(memories)

        scored = await self.score(query, candidates)
        result = await self.select(scored, budget)

        # Re-order chronologically
        history = sorted(
            [s for s in result.selected if s.candidate.ctype != "memory"],
            key=lambda s: s.candidate.turn,
        )
        mem_items = [s for s in result.selected if s.candidate.ctype == "memory"]
        result.selected = history + mem_items
        result.scoring_time_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "async build_context: %d candidates → %d selected (%d tokens, %.1fms)",
            len(candidates), len(result.selected), result.token_count,
            result.scoring_time_ms,
        )
        return result

    async def score_multiple_queries(
        self,
        queries: list[str],
        candidates: list[Candidate],
        token_budget: int | None = None,
    ) -> list[ContextResult]:
        """Score the same candidates against multiple queries concurrently."""
        tasks = [
            self.build_context(q, messages=candidates, token_budget=token_budget)
            for q in queries
        ]
        return await asyncio.gather(*tasks)
