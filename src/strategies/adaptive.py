"""Adaptive (Attention Filter) strategy: embedding relevance + recency decay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..embedding import EmbeddingService
from ..memory import MemoryManager
from ..tokenizer_service import TokenizerService
from .base import SYSTEM_PROMPT, SelectionResult, format_prompt


@dataclass
class Candidate:
    """A scored candidate for context inclusion."""
    text: str
    ctype: str  # "history" | "memory"
    embedding: np.ndarray
    age: int
    turn: int  # original turn index (for chronological ordering)
    score: float = 0.0


class AdaptiveStrategy:
    """
    Attention-inspired context selection:
    - Score each candidate by cosine similarity to query × recency decay.
    - Select top-scoring candidates within token budget.
    - Assemble selected history chronologically.
    """

    name: str = "Adaptive"

    def __init__(self, cfg: Config, tokenizer: TokenizerService,
                 embedding: EmbeddingService) -> None:
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._embedding = embedding

    def build_prompt(self, query: str, memory: MemoryManager) -> SelectionResult:
        q_vec = self._embedding.embed(query)
        stm = memory.get_stm()
        candidates: list[Candidate] = []

        # Add STM messages as candidates
        for i, msg in enumerate(stm):
            age = len(stm) - 1 - i  # 0 = most recent
            vec = self._embedding.embed(msg.text)
            candidates.append(Candidate(
                text=f"[Turn {msg.turn} - {msg.role}]: {msg.text}",
                ctype="history",
                embedding=vec,
                age=age,
                turn=msg.turn,
            ))

        # Add top-K LTM entries as candidates
        # LTM entries are persistent knowledge — don't penalize by conversation length.
        # They already passed relevance filtering (vector search top-K).
        # Give them age=0 so only cosine similarity determines their score.
        ltm_records = memory.search_ltm(q_vec, self._cfg.ltm_top_k)
        for rec in ltm_records:
            candidates.append(Candidate(
                text=f"[Memory]: {rec.text}",
                ctype="memory",
                embedding=rec.embedding,
                age=0,  # no decay penalty for durable memories
                turn=-1,  # memories don't have a turn; placed separately
            ))

        # Score each candidate
        decay = self._cfg.decay_factor
        for cand in candidates:
            cos = max(0.0, EmbeddingService.cosine_similarity(q_vec, cand.embedding))
            cand.score = cos * (decay ** cand.age)

        # Sort by descending score
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Select within token budget
        overhead = self._tokenizer.count(format_prompt(SYSTEM_PROMPT, [], query))
        budget = self._cfg.token_budget - overhead

        selected_cands: list[Candidate] = []
        omitted_cands: list[Candidate] = []
        total_tokens = 0

        for cand in candidates:
            t = self._tokenizer.count(cand.text)
            if total_tokens + t > budget:
                omitted_cands.append(cand)
            else:
                selected_cands.append(cand)
                total_tokens += t

        # Assemble: history in chronological order, then memories
        history_cands = sorted(
            [c for c in selected_cands if c.ctype == "history"],
            key=lambda c: c.turn,
        )
        memory_cands = [c for c in selected_cands if c.ctype == "memory"]

        context_blocks = [c.text for c in history_cands] + [c.text for c in memory_cands]

        prompt = format_prompt(SYSTEM_PROMPT, context_blocks, query)
        final_token_count = overhead + total_tokens

        return SelectionResult(
            prompt=prompt,
            selected=[c.text for c in selected_cands],
            omitted=[c.text for c in omitted_cands],
            token_count=final_token_count,
        )
