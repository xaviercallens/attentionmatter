"""Unit tests for advanced scoring modules."""

import numpy as np
import pytest

from attn_scorer import Candidate, ScorerConfig
from attn_scorer.advanced.classifier import RelevanceClassifier
from attn_scorer.advanced.cross_encoder import CrossEncoderReranker
from attn_scorer.advanced.positional_bias import LearnablePositionalBias
from attn_scorer.models import ScoredCandidate


class TestRelevanceClassifier:
    def test_predict_returns_probability(self):
        clf = RelevanceClassifier()
        cand = Candidate(text="booking code XYZ789", ctype="memory", age=0, turn=-1)
        prob = clf.predict("booking code", cand, cosine_sim=0.8)
        assert 0.0 <= prob <= 1.0

    def test_high_cosine_high_probability(self):
        clf = RelevanceClassifier()
        cand = Candidate(text="relevant content", ctype="fact", age=0, turn=1)
        high = clf.predict("relevant", cand, cosine_sim=0.9)
        low = clf.predict("relevant", cand, cosine_sim=0.1)
        assert high > low

    def test_batch_prediction(self):
        clf = RelevanceClassifier()
        cands = [
            Candidate(text="one", ctype="history", age=0, turn=1),
            Candidate(text="two", ctype="history", age=5, turn=2),
        ]
        probs = clf.predict_batch("query", cands, [0.5, 0.3])
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)


class TestCrossEncoderReranker:
    def test_rerank_preserves_count(self):
        reranker = CrossEncoderReranker(mode="heuristic", top_k=5)
        scored = [
            ScoredCandidate(
                candidate=Candidate(text=f"item {i}", ctype="history", age=i, turn=i),
                score=1.0 - i * 0.1,
                cosine_similarity=0.5,
                decay_multiplier=1.0,
                type_weight=1.0,
            )
            for i in range(10)
        ]
        reranked = reranker.rerank("query term", scored)
        assert len(reranked) == 10

    def test_rerank_boosts_keyword_match(self):
        reranker = CrossEncoderReranker(mode="heuristic", top_k=10)
        scored = [
            ScoredCandidate(
                candidate=Candidate(text="booking code XYZ789", ctype="fact", age=0, turn=1),
                score=0.5, cosine_similarity=0.5, decay_multiplier=1.0, type_weight=1.0,
            ),
            ScoredCandidate(
                candidate=Candidate(text="weather is nice", ctype="chit_chat", age=0, turn=2),
                score=0.6, cosine_similarity=0.6, decay_multiplier=1.0, type_weight=1.0,
            ),
        ]
        reranked = reranker.rerank("booking code", scored)
        # The booking item should be boosted above weather
        assert reranked[0].candidate.text == "booking code XYZ789"


class TestLearnablePositionalBias:
    def test_bias_at_zero_age(self):
        pb = LearnablePositionalBias(initial_slope=-0.01)
        bias = pb.bias(0)
        assert abs(bias) < 0.01  # near zero at age=0

    def test_bias_decreases_with_age(self):
        pb = LearnablePositionalBias(initial_slope=-0.01)
        assert pb.bias(100) < pb.bias(0)

    def test_bias_batch(self):
        pb = LearnablePositionalBias()
        biases = pb.bias_batch([0, 10, 50, 100])
        assert len(biases) == 4
        # Should be monotonically decreasing with negative slope
        assert biases[0] >= biases[-1]

    def test_train_reduces_loss(self):
        pb = LearnablePositionalBias()
        ages = [0, 5, 10, 20, 50, 100, 200]
        # Target: slight negative bias that plateaus
        targets = [0.0, -0.01, -0.02, -0.03, -0.04, -0.04, -0.04]
        result = pb.train(ages, targets, lr=0.0001, epochs=500)
        assert result["loss"] < 0.1  # converges to reasonable loss
