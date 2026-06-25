"""Evaluation framework for comparing scoring strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ScorerConfig
from ..embeddings.base import EmbeddingBackend
from ..models import Candidate, ContextResult
from ..scorer import Scorer
from .classifier import RelevanceClassifier
from .cross_encoder import CrossEncoderReranker
from .multi_query import MultiQueryScorer
from .positional_bias import LearnablePositionalBias


@dataclass
class EvalScenario:
    """A scenario for comparative evaluation."""
    id: str
    query: str
    candidates: list[Candidate]
    key_fact: str
    relevant_indices: list[int] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of a single strategy on a scenario."""
    strategy_name: str
    scenario_id: str
    token_count: int
    key_fact_found: bool
    recall_at_k: float  # fraction of relevant items in selection
    latency_ms: float
    reduction_pct: float


def evaluate_strategies(
    scenarios: list[EvalScenario],
    embedding: EmbeddingBackend,
    config: ScorerConfig,
    token_counter=None,
    token_budget: int = 4096,
) -> list[EvalResult]:
    """
    Run all scoring strategies on all scenarios and compare.

    Strategies compared:
    1. Cosine + Decay (baseline from Phase 1-3)
    2. Classifier (feature-based)
    3. Cosine + Decay + Cross-encoder rerank
    4. Cosine + Decay + Positional bias
    """
    counter = token_counter or (lambda t: int(len(t.split()) * 1.3))
    results: list[EvalResult] = []

    scorer = Scorer(config, embedding=embedding, token_counter=counter)
    classifier = RelevanceClassifier(config)
    reranker = CrossEncoderReranker(mode="heuristic", top_k=20)
    pos_bias = LearnablePositionalBias()

    for scenario in scenarios:
        full_tokens = sum(counter(c.text) for c in scenario.candidates)

        # --- Strategy 1: Cosine + Decay (baseline) ---
        t0 = time.perf_counter()
        scored = scorer.score(scenario.query, scenario.candidates)
        result = scorer.select(scored, token_budget)
        t1 = time.perf_counter()

        found = any(
            scenario.key_fact.lower() in sc.candidate.text.lower()
            for sc in result.selected
        )
        recall = _recall(result, scenario.relevant_indices)
        reduction = (1 - result.token_count / full_tokens) * 100

        results.append(EvalResult(
            strategy_name="cosine_decay",
            scenario_id=scenario.id,
            token_count=result.token_count,
            key_fact_found=found,
            recall_at_k=recall,
            latency_ms=(t1 - t0) * 1000,
            reduction_pct=reduction,
        ))

        # --- Strategy 2: Classifier ---
        t0 = time.perf_counter()
        cosine_sims = [sc.cosine_similarity for sc in scored]
        probs = classifier.predict_batch(
            scenario.query, scenario.candidates, cosine_sims
        )
        # Re-sort by classifier probability
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        selected_cls = []
        cls_tokens = 0
        for idx, prob in indexed:
            tc = counter(scenario.candidates[idx].text)
            if cls_tokens + tc <= token_budget:
                selected_cls.append(idx)
                cls_tokens += tc
        t1 = time.perf_counter()

        found_cls = any(
            scenario.key_fact.lower() in scenario.candidates[i].text.lower()
            for i in selected_cls
        )
        recall_cls = (
            len(set(selected_cls) & set(scenario.relevant_indices))
            / max(len(scenario.relevant_indices), 1)
        )
        reduction_cls = (1 - cls_tokens / full_tokens) * 100

        results.append(EvalResult(
            strategy_name="classifier",
            scenario_id=scenario.id,
            token_count=cls_tokens,
            key_fact_found=found_cls,
            recall_at_k=recall_cls,
            latency_ms=(t1 - t0) * 1000,
            reduction_pct=reduction_cls,
        ))

        # --- Strategy 3: Cosine + Decay + Cross-encoder ---
        t0 = time.perf_counter()
        scored2 = scorer.score(scenario.query, scenario.candidates)
        reranked = reranker.rerank(scenario.query, scored2)
        result_ce = scorer.select(reranked, token_budget)
        t1 = time.perf_counter()

        found_ce = any(
            scenario.key_fact.lower() in sc.candidate.text.lower()
            for sc in result_ce.selected
        )
        recall_ce = _recall(result_ce, scenario.relevant_indices)
        reduction_ce = (1 - result_ce.token_count / full_tokens) * 100

        results.append(EvalResult(
            strategy_name="cosine_decay_rerank",
            scenario_id=scenario.id,
            token_count=result_ce.token_count,
            key_fact_found=found_ce,
            recall_at_k=recall_ce,
            latency_ms=(t1 - t0) * 1000,
            reduction_pct=reduction_ce,
        ))

        # --- Strategy 4: Cosine + Positional bias ---
        t0 = time.perf_counter()
        scored3 = scorer.score(scenario.query, scenario.candidates)
        ages = [sc.candidate.age for sc in scored3]
        biases = pos_bias.bias_batch(ages)
        for sc, bias in zip(scored3, biases):
            sc.score = max(0.0, sc.score + bias)
        scored3.sort(key=lambda s: s.score, reverse=True)
        result_pb = scorer.select(scored3, token_budget)
        t1 = time.perf_counter()

        found_pb = any(
            scenario.key_fact.lower() in sc.candidate.text.lower()
            for sc in result_pb.selected
        )
        recall_pb = _recall(result_pb, scenario.relevant_indices)
        reduction_pb = (1 - result_pb.token_count / full_tokens) * 100

        results.append(EvalResult(
            strategy_name="cosine_positional_bias",
            scenario_id=scenario.id,
            token_count=result_pb.token_count,
            key_fact_found=found_pb,
            recall_at_k=recall_pb,
            latency_ms=(t1 - t0) * 1000,
            reduction_pct=reduction_pb,
        ))

    return results


def _recall(result: ContextResult, relevant_indices: list[int]) -> float:
    """Compute recall of relevant items in selection."""
    if not relevant_indices:
        return 1.0
    selected_turns = {sc.candidate.turn for sc in result.selected}
    found = sum(1 for idx in relevant_indices if idx in selected_turns)
    return found / len(relevant_indices)


def print_eval_table(results: list[EvalResult]) -> str:
    """Format evaluation results as a comparison table."""
    from collections import defaultdict

    by_strategy: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)

    lines = [
        f"{'Strategy':<25} {'Avg Tokens':>10} {'Fact Found':>11} "
        f"{'Recall':>7} {'Latency':>9} {'Reduction':>10}",
        "-" * 80,
    ]

    for strat, res in by_strategy.items():
        avg_tok = sum(r.token_count for r in res) / len(res)
        fact_rate = sum(r.key_fact_found for r in res) / len(res) * 100
        avg_recall = sum(r.recall_at_k for r in res) / len(res) * 100
        avg_lat = sum(r.latency_ms for r in res) / len(res)
        avg_red = sum(r.reduction_pct for r in res) / len(res)
        lines.append(
            f"{strat:<25} {avg_tok:>10.0f} {fact_rate:>10.1f}% "
            f"{avg_recall:>6.1f}% {avg_lat:>8.2f}ms {avg_red:>9.1f}%"
        )

    table = "\n".join(lines)
    print(table)
    return table
