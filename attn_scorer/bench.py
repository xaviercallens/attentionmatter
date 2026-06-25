"""Latency benchmarking utilities for attn_scorer."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import ScorerConfig
from .models import Candidate
from .scorer import Scorer


@dataclass
class BenchmarkResult:
    """Result of a latency benchmark run."""
    num_candidates: int
    scoring_time_ms: float
    selection_time_ms: float
    total_time_ms: float
    embedding_time_ms: float
    candidates_per_second: float


def benchmark_scoring(
    scorer: Scorer,
    num_candidates: int = 200,
    embedding_dim: int = 384,
    runs: int = 5,
) -> BenchmarkResult:
    """
    Benchmark scoring latency with synthetic candidates.

    Args:
        scorer: Scorer instance to benchmark.
        num_candidates: Number of candidates to score.
        embedding_dim: Dimension of embeddings.
        runs: Number of runs to average over.

    Returns:
        BenchmarkResult with timing breakdown.
    """
    # Create synthetic candidates with pre-computed embeddings
    rng = np.random.default_rng(42)
    candidates = []
    for i in range(num_candidates):
        vec = rng.standard_normal(embedding_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        candidates.append(Candidate(
            text=f"This is candidate message number {i} with some content.",
            ctype="history" if i > 5 else "memory",
            age=num_candidates - 1 - i,
            turn=i,
            embedding=vec,
        ))

    query = "What is the important information?"
    q_vec = rng.standard_normal(embedding_dim).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Warm up
    scorer.score(query, candidates[:10])

    # Benchmark scoring
    scoring_times = []
    selection_times = []
    total_times = []

    for _ in range(runs):
        # Reset embeddings (to measure scoring, not embedding)
        for c in candidates:
            pass  # keep pre-computed embeddings

        t0 = time.perf_counter()
        scored = scorer.score(query, candidates)
        t1 = time.perf_counter()
        scorer.select(scored, token_budget=2000)
        t2 = time.perf_counter()

        scoring_times.append((t1 - t0) * 1000)
        selection_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    avg_scoring = sum(scoring_times) / len(scoring_times)
    avg_selection = sum(selection_times) / len(selection_times)
    avg_total = sum(total_times) / len(total_times)

    return BenchmarkResult(
        num_candidates=num_candidates,
        scoring_time_ms=avg_scoring,
        selection_time_ms=avg_selection,
        total_time_ms=avg_total,
        embedding_time_ms=0.0,  # embeddings pre-computed
        candidates_per_second=num_candidates / (avg_total / 1000),
    )


def run_latency_suite(scorer: Scorer) -> list[BenchmarkResult]:
    """Run benchmarks at various candidate counts."""
    sizes = [50, 100, 200, 500, 1000, 5000]
    results = []
    for n in sizes:
        result = benchmark_scoring(scorer, num_candidates=n)
        results.append(result)
        print(
            f"  {n:>5} candidates: "
            f"score={result.scoring_time_ms:6.2f}ms "
            f"select={result.selection_time_ms:6.2f}ms "
            f"total={result.total_time_ms:6.2f}ms "
            f"({result.candidates_per_second:.0f} cand/s)"
        )
    return results
