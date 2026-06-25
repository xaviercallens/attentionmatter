"""Unit tests for the feedback learning module."""

import pytest

from attn_scorer import Candidate, ScorerConfig
from attn_scorer.feedback import FeedbackLearner, FeedbackSignal, FeedbackStore


class TestFeedbackStore:
    def test_add_and_retrieve(self):
        store = FeedbackStore()
        store.add(FeedbackSignal(query="test", candidate_text="hello", was_needed=True))
        assert store.size == 1
        assert store.get_all()[0].was_needed is True

    def test_get_for_query_similarity(self):
        store = FeedbackStore()
        store.add(FeedbackSignal(query="booking code info", candidate_text="a", was_needed=True))
        store.add(FeedbackSignal(query="weather forecast", candidate_text="b", was_needed=False))
        related = store.get_for_query("booking code")
        assert len(related) >= 1
        assert related[0].query == "booking code info"


class TestFeedbackLearner:
    def test_insufficient_data(self):
        config = ScorerConfig()
        learner = FeedbackLearner(config)
        result = learner.learn(min_signals=10)
        assert result["status"] == "insufficient_data"

    def test_learns_type_weights(self):
        config = ScorerConfig()
        original_memory_weight = config.type_weights["memory"]
        learner = FeedbackLearner(config)

        # Simulate: memories are always needed, chit_chat never
        for _ in range(20):
            learner.record_feedback(
                "question",
                Candidate(text="important fact", ctype="memory", age=0, turn=-1),
                was_needed=True,
            )
            learner.record_feedback(
                "question",
                Candidate(text="random chat", ctype="chit_chat", age=5, turn=10),
                was_needed=False,
            )

        result = learner.learn(min_signals=10)
        assert result["status"] == "adjusted"
        # Memory weight should increase
        assert config.type_weights["memory"] >= original_memory_weight
        # Chit chat weight should decrease
        assert config.type_weights["chit_chat"] < 0.8

    def test_keyword_boost(self):
        config = ScorerConfig()
        learner = FeedbackLearner(config)

        for _ in range(15):
            learner.record_feedback(
                "booking info",
                Candidate(text="booking code XYZ789 confirmed", ctype="fact", age=0, turn=1),
                was_needed=True,
            )

        learner.learn(min_signals=10)
        boost = learner.get_keyword_boost("booking code XYZ789")
        assert boost > 1.0

    def test_decay_adjustment_on_omitted(self):
        config = ScorerConfig(decay_factor=0.90)
        learner = FeedbackLearner(config)

        # Simulate: many omitted items were needed
        for _ in range(15):
            learner.record_feedback(
                "query",
                Candidate(text="needed but omitted", ctype="history", age=100, turn=1),
                was_needed=True,
            )
            learner._store.get_all()[-1].metadata["was_omitted"] = True

        learner.learn(min_signals=10)
        # Decay should increase (less aggressive)
        assert config.decay_factor > 0.90
