"""Shared strategy interface and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..memory import MemoryManager


@dataclass
class SelectionResult:
    """Output of a context-selection strategy."""
    prompt: str
    selected: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    token_count: int = 0


class ContextStrategy(Protocol):
    """Protocol that all context-management strategies implement."""

    name: str

    def build_prompt(self, query: str, memory: MemoryManager) -> SelectionResult:
        """Assemble a prompt for the LLM given the current query and memory state."""
        ...


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the provided "
    "conversation context and memory facts. If the answer is in the context, state "
    "it clearly. If you cannot find the answer, say so."
)


def format_prompt(system: str, context_blocks: list[str], query: str) -> str:
    """Assemble a standard prompt from system instruction, context, and query."""
    parts = [f"[System]\n{system}\n"]
    if context_blocks:
        parts.append("[Context]\n" + "\n".join(context_blocks) + "\n")
    parts.append(f"[User Query]\n{query}\n")
    parts.append("[Assistant]\n")
    return "\n".join(parts)
