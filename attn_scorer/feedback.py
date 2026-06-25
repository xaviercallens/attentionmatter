"""Feedback learning loop — learn from user corrections."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ScorerConfig
from .models import Candidate, ScoredCandidate


@dataclass
class FeedbackSignal:
    """A feedback signal from the user or system."""
    query: str
    candidate_text: str
    was_needed: bool  # True = user needed this context, False = it was useless
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackStore:
    """Stores and retrieves feedback signals for learning."""

    def __init__(self, path: str | Path | None = None):
        self._signals: list[FeedbackSignal] = []
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._load()

    def add(self, signal: FeedbackSignal) -> None:
        self._signals.append(signal)
        if self._path:
            self._save_incremental(signal)

    def get_all(self) -> list[FeedbackSignal]:
        return list(self._signals)

    def get_for_query(self, query: str) -> list[FeedbackSignal]:
        q_words = set(query.lower().split())
        relevant = []
        for s in self._signals:
            s_words = set(s.query.lower().split())
            if len(q_words & s_words) / max(len(q_words | s_words), 1) > 0.3:
                relevant.append(s)
        return relevant

    @property
    def size(self) -> int:
        return len(self._signals)

    def _save_incremental(self, signal: FeedbackSignal) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps({
                "query": signal.query,
                "candidate_text": signal.candidate_text,
                "was_needed": signal.was_needed,
                "timestamp": signal.timestamp,
                "metadata": signal.metadata,
            }) + "\n")

    def _load(self) -> None:
        with open(self._path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self._signals.append(FeedbackSignal(**d))


class FeedbackLearner:
    """
    Learns from feedback to adjust scoring parameters.

    Adjustments:
    1. Type weight adjustment: if memories are often needed, boost memory weight
    2. Decay factor tuning: if old items are often needed, reduce decay
    3. Keyword boosting: learn which terms signal important content
    4. Per-topic relevance patterns

    The learner modifies the ScorerConfig in-place based on accumulated feedback.
    """

    def __init__(self, config: ScorerConfig, store: FeedbackStore | None = None):
        self._config = config
        self._store = store or FeedbackStore()
        self._keyword_boosts: dict[str, float] = {}
        self._adjustment_history: list[dict] = []

    def record_feedback(
        self, query: str, candidate: Candidate | ScoredCandidate,
        was_needed: bool,
    ) -> None:
        """Record that a candidate was or wasn't needed for the query."""
        text = (candidate.candidate.text if isinstance(candidate, ScoredCandidate)
                else candidate.text)
        ctype = (candidate.candidate.ctype if isinstance(candidate, ScoredCandidate)
                 else candidate.ctype)

        signal = FeedbackSignal(
            query=query,
            candidate_text=text,
            was_needed=was_needed,
            metadata={"ctype": ctype},
        )
        self._store.add(signal)

    def record_omitted_was_needed(self, query: str, text: str) -> None:
        """Record that an omitted piece of context was actually needed."""
        self._store.add(FeedbackSignal(
            query=query, candidate_text=text, was_needed=True,
            metadata={"was_omitted": True},
        ))

    def learn(self, min_signals: int = 10) -> dict[str, Any]:
        """
        Analyze feedback and adjust config parameters.

        Returns a dict describing what was adjusted.
        """
        signals = self._store.get_all()
        if len(signals) < min_signals:
            return {"status": "insufficient_data", "signals": len(signals)}

        adjustments = {}

        # 1. Analyze if old items are frequently needed → reduce decay
        needed = [s for s in signals if s.was_needed]
        not_needed = [s for s in signals if not s.was_needed]

        # 2. Analyze type weights
        type_needed: dict[str, int] = {}
        type_total: dict[str, int] = {}
        for s in signals:
            ctype = s.metadata.get("ctype", "history")
            type_total[ctype] = type_total.get(ctype, 0) + 1
            if s.was_needed:
                type_needed[ctype] = type_needed.get(ctype, 0) + 1

        for ctype, total in type_total.items():
            if total >= 5:
                needed_ratio = type_needed.get(ctype, 0) / total
                current_weight = self._config.type_weights.get(ctype, 1.0)
                # Adjust weight toward the needed ratio
                new_weight = current_weight * 0.8 + (needed_ratio * 2.0) * 0.2
                new_weight = max(0.5, min(2.0, new_weight))  # clamp
                if abs(new_weight - current_weight) > 0.05:
                    self._config.type_weights[ctype] = new_weight
                    adjustments[f"type_weight_{ctype}"] = {
                        "old": current_weight, "new": new_weight,
                    }

        # 3. Learn keyword boosts from needed signals
        keyword_counts: dict[str, int] = {}
        for s in needed:
            words = set(s.candidate_text.lower().split())
            for w in words:
                if len(w) > 3:  # skip short words
                    keyword_counts[w] = keyword_counts.get(w, 0) + 1

        # Top keywords that appear often in needed content
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for kw, count in top_keywords:
            if count >= 3:
                self._keyword_boosts[kw] = min(1.5, 1.0 + count * 0.05)

        if self._keyword_boosts:
            adjustments["keyword_boosts"] = dict(list(self._keyword_boosts.items())[:5])

        # 4. Check if omitted items were needed → signal decay is too aggressive
        omitted_needed = [s for s in needed if s.metadata.get("was_omitted")]
        if len(omitted_needed) > len(needed) * 0.2:  # >20% of needed were omitted
            old_decay = self._config.decay_factor
            new_decay = min(1.0, old_decay + 0.01)
            if new_decay != old_decay:
                self._config.decay_factor = new_decay
                adjustments["decay_factor"] = {"old": old_decay, "new": new_decay}

        self._adjustment_history.append({
            "timestamp": time.time(),
            "signals_used": len(signals),
            "adjustments": adjustments,
        })

        return {"status": "adjusted", "adjustments": adjustments}

    def get_keyword_boost(self, text: str) -> float:
        """Get accumulated keyword boost for a text."""
        if not self._keyword_boosts:
            return 1.0
        words = set(text.lower().split())
        boosts = [self._keyword_boosts[w] for w in words if w in self._keyword_boosts]
        return max(boosts) if boosts else 1.0

    @property
    def adjustment_history(self) -> list[dict]:
        return list(self._adjustment_history)

    def save_state(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "keyword_boosts": self._keyword_boosts,
                "history": self._adjustment_history,
            }, f)

    def load_state(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._keyword_boosts = data.get("keyword_boosts", {})
        self._adjustment_history = data.get("history", [])
