"""Core scorer: the main public API for attn_scorer."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .config import ScorerConfig
from .embeddings.base import EmbeddingBackend
from .embeddings.local import LocalEmbeddingBackend
from .models import Candidate, ContextResult, ScoredCandidate
from .vector_store.base import VectorStore
from .vector_store.brute_force import BruteForceStore


class Scorer:
    """
    Attention-inspired context scorer and selector.

    Core API:
        scorer = Scorer(config)
        result = scorer.build_context(query, messages, memories, token_budget)

    Lower-level API:
        scored = scorer.score(query, candidates)
        selected = scorer.select(scored, token_budget, token_counter)
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        embedding: EmbeddingBackend | None = None,
        vector_store: VectorStore | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._config = config or ScorerConfig()
        self._embedding = embedding or self._create_embedding()
        self._vector_store = vector_store or BruteForceStore()
        self._token_counter = token_counter or self._default_token_counter

    def _create_embedding(self) -> EmbeddingBackend:
        if self._config.embedding_backend == "openai":
            from .embeddings.openai import OpenAIEmbeddingBackend
            return OpenAIEmbeddingBackend(self._config)
        return LocalEmbeddingBackend(self._config)

    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Fallback token counter: words × 1.3."""
        if not text or not text.strip():
            return 0
        return int(len(text.split()) * 1.3)

    # --- Public API ---

    def score(self, query: str, candidates: list[Candidate]) -> list[ScoredCandidate]:
        """
        Score candidates by relevance to the query.

        Returns a list of ScoredCandidate sorted by descending score.
        """
        q_vec = self._embedding.embed(query)

        # Batch-embed candidates that don't already have embeddings
        to_embed = [(i, c) for i, c in enumerate(candidates) if c.embedding is None]
        if to_embed:
            texts = [c.text for _, c in to_embed]
            vecs = self._embedding.embed_batch(texts)
            for j, (i, c) in enumerate(to_embed):
                c.embedding = vecs[j]

        # Score each candidate
        decay = self._config.decay_factor
        type_weights = self._config.type_weights
        scored = []

        for cand in candidates:
            cos = max(0.0, EmbeddingBackend.cosine_similarity(q_vec, cand.embedding))
            decay_mult = decay ** cand.age
            type_w = type_weights.get(cand.ctype, 1.0)
            final_score = cos * decay_mult * type_w
            token_count = self._token_counter(cand.text)

            scored.append(ScoredCandidate(
                candidate=cand,
                score=final_score,
                cosine_similarity=cos,
                decay_multiplier=decay_mult,
                type_weight=type_w,
                token_count=token_count,
            ))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored

    def select(
        self,
        scored_candidates: list[ScoredCandidate],
        token_budget: int,
    ) -> ContextResult:
        """
        Select candidates within the token budget (already scored and sorted).

        Returns ContextResult with selected/omitted items and token count.
        """
        selected: list[ScoredCandidate] = []
        omitted: list[ScoredCandidate] = []
        total_tokens = 0

        for sc in scored_candidates:
            if total_tokens + sc.token_count <= token_budget:
                selected.append(sc)
                total_tokens += sc.token_count
            else:
                omitted.append(sc)

        budget_used = total_tokens / token_budget if token_budget > 0 else 0

        return ContextResult(
            selected=selected,
            omitted=omitted,
            token_count=total_tokens,
            budget_used=budget_used,
        )

    def build_context(
        self,
        query: str,
        messages: list[Candidate] | None = None,
        memories: list[Candidate] | None = None,
        token_budget: int | None = None,
    ) -> ContextResult:
        """
        High-level API: score, select, and return context within budget.

        Args:
            query: The user's current query.
            messages: Conversation history candidates (with age set).
            memories: LTM candidates (age=0, already relevance-filtered).
            token_budget: Max tokens for context. Uses config default if None.

        Returns:
            ContextResult with selected items in chronological order.
        """
        t0 = time.perf_counter()

        # Resolve budget
        budget = token_budget or self._resolve_budget(query)

        # Build candidate list
        candidates = []
        if messages:
            candidates.extend(messages)
        if memories:
            for mem in memories:
                mem.age = self._config.memory_age  # ensure age=0 for memories
                if mem.ctype == "history":
                    mem.ctype = "memory"
            candidates.extend(memories)

        # Score
        scored = self.score(query, candidates)

        # Select
        result = self.select(scored, budget)

        # Re-order selected: history by turn (chronological), then memories
        history_items = sorted(
            [sc for sc in result.selected if sc.candidate.ctype in ("history", "fact", "chit_chat")],
            key=lambda sc: sc.candidate.turn,
        )
        memory_items = [sc for sc in result.selected if sc.candidate.ctype == "memory"]
        result.selected = history_items + memory_items

        t1 = time.perf_counter()
        result.scoring_time_ms = (t1 - t0) * 1000

        return result

    def _resolve_budget(self, query: str) -> int:
        """Determine token budget, optionally adjusting by query complexity."""
        if not self._config.dynamic_budget_enabled:
            return self._config.default_token_budget

        # Dynamic budgeting: longer/complex queries get more budget
        query_tokens = self._token_counter(query)
        # Scale: short queries (< 10 tokens) → budget_min,
        # long queries (> 50 tokens) → budget_max
        ratio = min(1.0, max(0.0, (query_tokens - 5) / 45))
        budget_range = self._config.budget_max - self._config.budget_min
        return int(self._config.budget_min + ratio * budget_range)

    # --- Vector store operations (for LTM management) ---

    def add_memory(self, text: str, metadata: dict | None = None) -> int:
        """Add a memory entry to the vector store."""
        vec = self._embedding.embed(text)
        return self._vector_store.add(text, vec, metadata)

    def search_memories(self, query: str, top_k: int | None = None) -> list[Candidate]:
        """Search LTM for relevant memories. Returns Candidate objects."""
        k = top_k or self._config.ltm_top_k
        q_vec = self._embedding.embed(query)
        results = self._vector_store.search(q_vec, k)

        candidates = []
        for r in results:
            candidates.append(Candidate(
                text=r.text,
                ctype="memory",
                age=self._config.memory_age,
                turn=-1,
                metadata=r.metadata,
                embedding=q_vec if r.index == 0 else None,  # avoid redundant storage
            ))
            # Re-embed properly
            candidates[-1].embedding = self._embedding.embed(r.text)

        return candidates

    def clear_memories(self) -> None:
        """Clear all stored memories."""
        self._vector_store.clear()
