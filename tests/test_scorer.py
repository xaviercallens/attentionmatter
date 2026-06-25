"""Unit tests for the core Scorer."""

import numpy as np
import pytest

from attn_scorer import Candidate, ContextResult, Scorer, ScorerConfig


class TestScoring:
    """Tests for the score() method."""

    def test_score_returns_sorted_descending(self, scorer, sample_candidates):
        scored = scorer.score("booking code", sample_candidates)
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_score_all_non_negative(self, scorer, sample_candidates):
        scored = scorer.score("booking code", sample_candidates)
        assert all(s.score >= 0.0 for s in scored)

    def test_score_cosine_in_range(self, scorer, sample_candidates):
        scored = scorer.score("booking code", sample_candidates)
        for s in scored:
            assert -1.0 <= s.cosine_similarity <= 1.0

    def test_score_decay_multiplier_correct(self, scorer, sample_candidates):
        scored = scorer.score("booking code", sample_candidates)
        for s in scored:
            expected_decay = 0.95 ** s.candidate.age
            assert abs(s.decay_multiplier - expected_decay) < 1e-6

    def test_memory_age_zero_no_decay(self, scorer):
        candidates = [
            Candidate(text="memory fact", ctype="memory", age=0, turn=-1),
        ]
        scored = scorer.score("query", candidates)
        assert scored[0].decay_multiplier == 1.0

    def test_high_age_low_decay(self, scorer):
        candidates = [
            Candidate(text="old message", ctype="history", age=100, turn=1),
        ]
        scored = scorer.score("query", candidates)
        expected = 0.95 ** 100
        assert abs(scored[0].decay_multiplier - expected) < 1e-6

    def test_type_weight_applied(self, config, mock_embedding):
        config.type_weights = {"memory": 2.0, "history": 1.0, "chit_chat": 0.5}
        scorer = Scorer(config=config, embedding=mock_embedding,
                        token_counter=lambda t: len(t.split()))
        # Use different text so cosine with query varies; but same age=0
        candidates = [
            Candidate(text="booking code reference info", ctype="memory", age=0, turn=-1),
            Candidate(text="booking code reference info", ctype="chit_chat", age=0, turn=2),
        ]
        scored = scorer.score("booking code reference info", candidates)
        memory_sc = next(s for s in scored if s.candidate.ctype == "memory")
        chit_sc = next(s for s in scored if s.candidate.ctype == "chit_chat")
        # Same cosine (same text), same age, different type weight
        assert memory_sc.type_weight == 2.0
        assert chit_sc.type_weight == 0.5
        assert memory_sc.score >= chit_sc.score


class TestSelection:
    """Tests for the select() method."""

    def test_select_respects_budget(self, scorer, sample_candidates):
        scored = scorer.score("booking", sample_candidates)
        result = scorer.select(scored, token_budget=10)
        assert result.token_count <= 10

    def test_select_empty_candidates(self, scorer):
        result = scorer.select([], token_budget=100)
        assert result.token_count == 0
        assert len(result.selected) == 0

    def test_select_large_budget_includes_all(self, scorer, sample_candidates):
        scored = scorer.score("booking", sample_candidates)
        result = scorer.select(scored, token_budget=10000)
        assert len(result.selected) == len(sample_candidates)
        assert len(result.omitted) == 0

    def test_select_budget_used_ratio(self, scorer, sample_candidates):
        scored = scorer.score("booking", sample_candidates)
        result = scorer.select(scored, token_budget=50)
        assert 0.0 <= result.budget_used <= 1.0

    def test_reduction_vs_full(self, scorer, sample_candidates):
        scored = scorer.score("booking", sample_candidates)
        result = scorer.select(scored, token_budget=10)
        if result.omitted:
            assert result.reduction_vs_full > 0.0


class TestBuildContext:
    """Tests for the build_context() high-level API."""

    def test_build_context_basic(self, scorer):
        messages = [
            Candidate(text="Hello there", ctype="history", age=2, turn=1),
            Candidate(text="How can I help?", ctype="history", age=1, turn=2),
        ]
        result = scorer.build_context("help me", messages=messages)
        assert isinstance(result, ContextResult)
        assert result.token_count > 0

    def test_build_context_with_memories(self, scorer):
        messages = [Candidate(text="chat", ctype="history", age=1, turn=1)]
        memories = [Candidate(text="user name: Alice", ctype="memory", age=5, turn=-1)]
        result = scorer.build_context("name", messages=messages, memories=memories)
        # Memories should have age reset to 0
        mem_items = [s for s in result.selected if s.candidate.ctype == "memory"]
        for m in mem_items:
            assert m.candidate.age == 0

    def test_build_context_chronological_order(self, scorer):
        messages = [
            Candidate(text="first", ctype="history", age=3, turn=1),
            Candidate(text="second", ctype="history", age=2, turn=2),
            Candidate(text="third", ctype="history", age=1, turn=3),
        ]
        result = scorer.build_context("query", messages=messages, token_budget=100)
        history_items = [s for s in result.selected if s.candidate.ctype != "memory"]
        turns = [s.candidate.turn for s in history_items]
        assert turns == sorted(turns)

    def test_scoring_time_recorded(self, scorer, sample_candidates):
        result = scorer.build_context("booking", messages=sample_candidates)
        assert result.scoring_time_ms >= 0.0
