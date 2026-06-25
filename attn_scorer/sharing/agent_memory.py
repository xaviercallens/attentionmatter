"""Cross-agent memory sharing bus and per-agent stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embeddings.base import EmbeddingBackend
from ..models import Candidate
from .access_control import AccessLevel, AccessPolicy


@dataclass
class MemoryEntry:
    """A memory entry owned by an agent."""
    text: str
    topic: str = ""
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_agent: str = ""


class AgentMemoryStore:
    """Per-agent memory store with access policy."""

    def __init__(self, agent_id: str, embedding: EmbeddingBackend,
                 policy: AccessPolicy | None = None):
        self._agent_id = agent_id
        self._embedding = embedding
        self._policy = policy or AccessPolicy(agent_id=agent_id)
        self._memories: list[MemoryEntry] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def policy(self) -> AccessPolicy:
        return self._policy

    def add(self, text: str, topic: str = "", **metadata) -> None:
        vec = self._embedding.embed(text)
        self._memories.append(MemoryEntry(
            text=text, topic=topic, embedding=vec,
            metadata=metadata, owner_agent=self._agent_id,
        ))

    def search(self, query_vec: np.ndarray, top_k: int = 5,
               topic: str = "") -> list[MemoryEntry]:
        candidates = self._memories
        if topic:
            candidates = [m for m in candidates if m.topic == topic or not m.topic]

        scored = []
        for m in candidates:
            if m.embedding is not None:
                sim = float(np.dot(query_vec, m.embedding))
                scored.append((sim, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def get_shared(self, requester: str, query_vec: np.ndarray,
                   top_k: int = 5, topic: str = "") -> list[MemoryEntry]:
        """Get memories accessible to a requesting agent."""
        access = self._policy.can_access(requester, topic)
        if access == AccessLevel.NONE:
            return []

        # Find the grant to get max_entries limit
        max_k = top_k
        for grant in self._policy.grants:
            if grant.target_agent == requester:
                max_k = min(top_k, grant.max_entries)
                break

        return self.search(query_vec, min(top_k, max_k), topic)

    @property
    def size(self) -> int:
        return len(self._memories)

    def clear(self) -> None:
        self._memories.clear()


class SharedMemoryBus:
    """
    Central bus for cross-agent memory sharing.

    Agents register their stores and can query other agents'
    memories (subject to access policies).

    Usage:
        bus = SharedMemoryBus(embedding)
        bus.register_agent("booking_agent", policy=...)
        bus.register_agent("support_agent", policy=...)

        # booking_agent shares memories with support_agent
        bus.get_store("booking_agent").policy.grant_access("support_agent")

        # support_agent queries booking_agent's memories
        results = bus.query_across_agents(
            requester="support_agent",
            query="booking code",
            top_k=3,
        )
    """

    def __init__(self, embedding: EmbeddingBackend):
        self._embedding = embedding
        self._stores: dict[str, AgentMemoryStore] = {}

    def register_agent(self, agent_id: str,
                       policy: AccessPolicy | None = None) -> AgentMemoryStore:
        store = AgentMemoryStore(agent_id, self._embedding, policy)
        self._stores[agent_id] = store
        return store

    def get_store(self, agent_id: str) -> AgentMemoryStore | None:
        return self._stores.get(agent_id)

    def query_across_agents(
        self, requester: str, query: str,
        top_k: int = 5, topic: str = "",
    ) -> list[Candidate]:
        """
        Query all accessible agent stores and return unified candidates.

        Returns Candidate objects with metadata indicating the source agent.
        """
        q_vec = self._embedding.embed(query)
        all_results: list[tuple[float, MemoryEntry, str]] = []

        for agent_id, store in self._stores.items():
            if agent_id == requester:
                continue  # don't query own store
            entries = store.get_shared(requester, q_vec, top_k, topic)
            for entry in entries:
                if entry.embedding is not None:
                    sim = float(np.dot(q_vec, entry.embedding))
                    all_results.append((sim, entry, agent_id))

        # Sort by similarity and take top-K overall
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_results = all_results[:top_k]

        candidates = []
        for sim, entry, source_agent in top_results:
            candidates.append(Candidate(
                text=entry.text,
                ctype="memory",
                age=0,
                turn=-1,
                metadata={
                    "source_agent": source_agent,
                    "topic": entry.topic,
                    "similarity": sim,
                    **entry.metadata,
                },
                embedding=entry.embedding,
            ))
        return candidates

    @property
    def agents(self) -> list[str]:
        return list(self._stores.keys())

    def clear_all(self) -> None:
        for store in self._stores.values():
            store.clear()
