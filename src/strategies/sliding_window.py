"""Sliding-Window baseline: keep only the last N messages, ignore LTM."""

from __future__ import annotations

from ..config import Config
from ..memory import MemoryManager
from ..tokenizer_service import TokenizerService
from .base import SYSTEM_PROMPT, SelectionResult, format_prompt


class SlidingWindowStrategy:
    """Includes only the most recent N messages from STM; ignores LTM entirely."""

    name: str = "Sliding-Window"

    def __init__(self, cfg: Config, tokenizer: TokenizerService) -> None:
        self._cfg = cfg
        self._tokenizer = tokenizer

    def build_prompt(self, query: str, memory: MemoryManager) -> SelectionResult:
        stm = memory.get_stm()
        window_size = self._cfg.sliding_window_messages

        # Take only the last N messages
        windowed = stm[-window_size:] if len(stm) > window_size else stm
        older = stm[:-window_size] if len(stm) > window_size else []

        selected: list[str] = []
        for msg in windowed:
            selected.append(f"[Turn {msg.turn} - {msg.role}]: {msg.text}")

        omitted: list[str] = []
        for msg in older:
            omitted.append(f"[Turn {msg.turn} - {msg.role}]: {msg.text}")
        # All LTM entries are also omitted
        for rec in memory.get_all_ltm():
            omitted.append(f"[Memory]: {rec.text}")

        prompt = format_prompt(SYSTEM_PROMPT, selected, query)
        token_count = self._tokenizer.count(prompt)

        return SelectionResult(
            prompt=prompt,
            selected=selected,
            omitted=omitted,
            token_count=token_count,
        )
