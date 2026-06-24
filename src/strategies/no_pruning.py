"""No-Pruning baseline: include everything up to model maximum."""

from __future__ import annotations

from ..config import Config
from ..embedding import EmbeddingService
from ..memory import MemoryManager
from ..tokenizer_service import TokenizerService
from .base import SYSTEM_PROMPT, SelectionResult, format_prompt


class NoPruningStrategy:
    """Includes the entire conversation and all LTM entries, truncating only at max."""

    name: str = "No-Pruning"

    def __init__(self, cfg: Config, tokenizer: TokenizerService,
                 embedding: EmbeddingService) -> None:
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._embedding = embedding

    def build_prompt(self, query: str, memory: MemoryManager) -> SelectionResult:
        stm = memory.get_stm()
        ltm = memory.get_all_ltm()

        # Build candidate text blocks
        context_blocks: list[str] = []
        for msg in stm:
            context_blocks.append(f"[Turn {msg.turn} - {msg.role}]: {msg.text}")
        for rec in ltm:
            context_blocks.append(f"[Memory]: {rec.text}")

        # Use full max_context_tokens (no budget ratio) for this baseline
        max_tokens = self._cfg.max_context_tokens
        overhead = self._tokenizer.count(
            format_prompt(SYSTEM_PROMPT, [], query)
        )

        selected: list[str] = []
        omitted: list[str] = []
        total_tokens = overhead

        for block in context_blocks:
            t = self._tokenizer.count(block)
            if total_tokens + t > max_tokens:
                omitted.append(block)
            else:
                selected.append(block)
                total_tokens += t

        prompt = format_prompt(SYSTEM_PROMPT, selected, query)
        return SelectionResult(
            prompt=prompt,
            selected=selected,
            omitted=omitted,
            token_count=total_tokens,
        )
