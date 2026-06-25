"""Plugin interface for orchestrator integration (e.g., A3TK)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .config import ScorerConfig
from .models import Candidate, ContextResult
from .scorer import Scorer


@dataclass
class Message:
    """Standard message format for orchestrator integration."""
    text: str
    role: str  # "user" | "assistant" | "system"
    turn: int
    metadata: dict[str, Any] | None = None


@dataclass
class MemoryEntry:
    """Standard memory entry format for orchestrator integration."""
    text: str
    source: str = "default"
    importance: float = 1.0
    metadata: dict[str, Any] | None = None


class ContextPlugin(ABC):
    """
    Abstract plugin interface for integrating the scorer into an orchestrator.

    Orchestrators implement the abstract methods to provide their data sources.
    The plugin handles scoring and selection.
    """

    @abstractmethod
    def get_conversation_history(self) -> list[Message]:
        """Return the current conversation history (STM)."""
        ...

    @abstractmethod
    def get_memory_entries(self, query: str) -> list[MemoryEntry]:
        """Return relevant memory entries for the query (LTM retrieval)."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the token count for the given text."""
        ...

    @abstractmethod
    def get_token_budget(self) -> int:
        """Return the current token budget for context assembly."""
        ...


class AttentionContextManager:
    """
    Drop-in context manager that wraps the scorer for orchestrator use.

    Usage in an orchestrator:
        class MyPlugin(ContextPlugin):
            def get_conversation_history(self): return self.stm.get_all()
            def get_memory_entries(self, q): return self.ltm.search(q)
            def count_tokens(self, t): return self.tokenizer.count(t)
            def get_token_budget(self): return 4096

        manager = AttentionContextManager(plugin=MyPlugin(), config=ScorerConfig())
        result = manager.assemble_context(user_query)
        # result.selected_texts contains the ordered context blocks
    """

    def __init__(
        self,
        plugin: ContextPlugin,
        config: ScorerConfig | None = None,
    ) -> None:
        self._plugin = plugin
        self._config = config or ScorerConfig()
        self._scorer = Scorer(
            config=self._config,
            token_counter=plugin.count_tokens,
        )

    def assemble_context(self, query: str) -> ContextResult:
        """
        Assemble context for the given query using the plugin's data sources.

        1. Retrieves conversation history and memory entries.
        2. Converts to Candidate objects with appropriate age/type.
        3. Scores and selects within budget.
        4. Returns chronologically-ordered selected context.
        """
        # Get data from plugin
        history = self._plugin.get_conversation_history()
        memories = self._plugin.get_memory_entries(query)
        budget = self._plugin.get_token_budget()

        # Convert to candidates
        messages = self._history_to_candidates(history)
        mem_candidates = self._memories_to_candidates(memories)

        # Score and select
        return self._scorer.build_context(
            query=query,
            messages=messages,
            memories=mem_candidates,
            token_budget=budget,
        )

    def _history_to_candidates(self, history: list[Message]) -> list[Candidate]:
        """Convert orchestrator messages to scorer candidates."""
        n = len(history)
        candidates = []
        for i, msg in enumerate(history):
            age = n - 1 - i  # 0 = most recent
            candidates.append(Candidate(
                text=f"[{msg.role}]: {msg.text}",
                ctype="history",
                age=age,
                turn=msg.turn,
                metadata=msg.metadata or {},
            ))
        return candidates

    def _memories_to_candidates(self, memories: list[MemoryEntry]) -> list[Candidate]:
        """Convert orchestrator memory entries to scorer candidates."""
        return [
            Candidate(
                text=mem.text,
                ctype="memory",
                age=0,  # no decay for persistent memories
                turn=-1,
                metadata={"source": mem.source, "importance": mem.importance,
                          **(mem.metadata or {})},
            )
            for mem in memories
        ]

    @property
    def scorer(self) -> Scorer:
        """Access the underlying scorer for advanced usage."""
        return self._scorer
