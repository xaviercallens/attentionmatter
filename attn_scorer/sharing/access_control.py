"""Access control for cross-agent memory sharing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AccessLevel(Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class MemoryGrant:
    """A grant allowing one agent to access another's memories."""
    source_agent: str
    target_agent: str
    access_level: AccessLevel = AccessLevel.READ
    topic_filter: list[str] = field(default_factory=list)
    max_entries: int = 10
    expires_at: str | None = None  # ISO timestamp or None for permanent


@dataclass
class AccessPolicy:
    """
    Defines what memories an agent can share and with whom.

    Policies support:
    - Topic-based filtering (only share memories matching topics)
    - Entry limits (max N memories per request)
    - Expiration (time-limited grants)
    - Role-based access (read, write, admin)
    """
    agent_id: str
    default_access: AccessLevel = AccessLevel.NONE
    grants: list[MemoryGrant] = field(default_factory=list)
    shared_topics: list[str] = field(default_factory=list)
    private_topics: list[str] = field(default_factory=list)

    def can_access(self, requester: str, topic: str = "") -> AccessLevel:
        """Check if a requester can access memories under a topic."""
        if topic in self.private_topics:
            return AccessLevel.NONE

        for grant in self.grants:
            if grant.target_agent == requester:
                if not grant.topic_filter or topic in grant.topic_filter:
                    return grant.access_level

        if self.shared_topics and topic in self.shared_topics:
            return AccessLevel.READ

        return self.default_access

    def grant_access(self, target_agent: str, level: AccessLevel = AccessLevel.READ,
                     topics: list[str] | None = None, max_entries: int = 10):
        """Grant access to another agent."""
        self.grants.append(MemoryGrant(
            source_agent=self.agent_id,
            target_agent=target_agent,
            access_level=level,
            topic_filter=topics or [],
            max_entries=max_entries,
        ))

    def revoke_access(self, target_agent: str) -> None:
        """Revoke all grants for an agent."""
        self.grants = [g for g in self.grants if g.target_agent != target_agent]
