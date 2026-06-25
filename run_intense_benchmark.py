#!/usr/bin/env python3
"""
Intense production benchmark with REAL sentence-transformer embeddings.

Tests at scale: 100/200/500/1000 turns with full embedding computation.
Measures real latency, token reduction, and key fact preservation.
Also benchmarks the advanced scoring strategies (classifier + cross-encoder rerank).
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from attn_scorer import Scorer, ScorerConfig, Candidate
from attn_scorer.embeddings.local import LocalEmbeddingBackend
from attn_scorer.advanced.classifier import RelevanceClassifier
from attn_scorer.advanced.cross_encoder import CrossEncoderReranker
from attn_scorer.advanced.positional_bias import LearnablePositionalBias
from attn_scorer.bench import run_latency_suite
from attn_scorer.evaluation.prod_dataset import ProdDatasetGenerator, DatasetConfig
from attn_scorer.evaluation.prod_benchmark import ProdBenchmarkRunner, BenchmarkReport


def run_scale_benchmark():
    """Run multi-scale benchmark with real embeddings."""
    print("=" * 70)
    print("INTENSE BENCHMARK — Real Embeddings (all-MiniLM-L6-v2)")
    print("=" * 70)
    print()

    config = ScorerConfig(decay_factor=0.95, default_token_budget=6553)
    print("Loading embedding model...")
    t0 = time.perf_counter()
    emb = LocalEmbeddingBackend(config)
    # Warm up
    emb.embed("warmup text")
    load_time = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_time:.0f}ms")
    print(f"  Dimension: {emb.dimension}")
    print()

    # --- Part 1: Scale test across conversation lengths ---
    print("=" * 70)
    print("PART 1: Scale Test (varying conversation length)")
    print("=" * 70)
    print()

    turn_configs = [100, 200, 500]
    for num_turns in turn_configs:
        print(f"--- {num_turns} turns ---")
        gen = ProdDatasetGenerator(DatasetConfig(
            num_scenarios=3, turns_options=[num_turns], seed=42 + num_turns
        ))
        scenarios = gen.generate()

        runner = ProdBenchmarkRunner(config=config, embedding=emb, token_budget=6553)
        report = runner.run_sync(scenarios)

        print(f"  Avg reduction: {report.avg_reduction:.1f}%")
        print(f"  Key fact rate: {report.avg_key_fact_rate * 100:.1f}%")
        print(f"  Avg latency:   {report.avg_latency_ms:.1f}ms")
        print(f"  FP rate:       {report.avg_false_positive_rate * 100:.1f}%")
        for r in report.results:
            fact_str = "PASS" if r.key_fact_rate == 1.0 else "FAIL"
            print(f"    {r.scenario_id[:40]:<40} {r.reduction_pct:.0f}%  {fact_str}  {r.latency_ms:.0f}ms")
        print()

    # --- Part 2: Strategy comparison on 200-turn scenarios ---
    print("=" * 70)
    print("PART 2: Strategy Comparison (200 turns, real embeddings)")
    print("=" * 70)
    print()

    gen = ProdDatasetGenerator(DatasetConfig(
        num_scenarios=6, turns_options=[200], seed=999
    ))
    scenarios = gen.generate()

    scorer = Scorer(config=config, embedding=emb,
                    token_counter=lambda t: int(len(t.split()) * 1.3))
    classifier = RelevanceClassifier(config)
    reranker = CrossEncoderReranker(mode="heuristic", top_k=30)
    pos_bias = LearnablePositionalBias(initial_slope=-0.002)

    strategies = {
        "cosine_decay": lambda q, cands: scorer.select(
            scorer.score(q, cands), 6553
        ),
        "cosine_decay_rerank": lambda q, cands: scorer.select(
            reranker.rerank(q, scorer.score(q, cands)), 6553
        ),
        "cosine_positional_bias": lambda q, cands: _with_bias(
            scorer, pos_bias, q, cands, 6553
        ),
        "classifier_ranked": lambda q, cands: _with_classifier(
            scorer, classifier, q, cands, 6553
        ),
    }

    results_table = []
    for strat_name, strat_fn in strategies.items():
        facts_found = 0
        total_tokens = 0
        total_latency = 0.0
        total_scenarios = 0

        for scenario in scenarios:
            t0 = time.perf_counter()
            result = strat_fn(scenario.query, scenario.candidates)
            latency = (time.perf_counter() - t0) * 1000

            selected_text = " ".join(sc.candidate.text for sc in result.selected)
            found = all(
                fact.lower() in selected_text.lower()
                for fact in scenario.key_facts
            )
            facts_found += int(found)
            total_tokens += result.token_count
            total_latency += latency
            total_scenarios += 1

        avg_tokens = total_tokens / total_scenarios
        fact_rate = facts_found / total_scenarios * 100
        avg_latency = total_latency / total_scenarios
        full_tokens = sum(
            sum(int(len(c.text.split()) * 1.3) for c in s.candidates)
            for s in scenarios
        ) / len(scenarios)
        reduction = (1 - avg_tokens / full_tokens) * 100

        results_table.append((strat_name, avg_tokens, reduction, fact_rate, avg_latency))

    print(f"{'Strategy':<25} {'Tokens':>7} {'Red%':>6} {'Facts':>7} {'Latency':>9}")
    print("-" * 60)
    for name, tok, red, facts, lat in results_table:
        print(f"{name:<25} {tok:>7.0f} {red:>5.1f}% {facts:>6.1f}% {lat:>8.1f}ms")
    print()

    # --- Part 3: Raw latency benchmark (pre-computed embeddings) ---
    print("=" * 70)
    print("PART 3: Raw Scoring Latency (pre-computed embeddings)")
    print("=" * 70)
    print()
    run_latency_suite(scorer)
    print()

    # --- Part 4: Embedding throughput ---
    print("=" * 70)
    print("PART 4: Embedding Throughput")
    print("=" * 70)
    print()

    texts = [f"This is test sentence number {i} with some varied content about topic {i % 10}." for i in range(100)]
    t0 = time.perf_counter()
    emb.embed_batch(texts)
    batch_time = (time.perf_counter() - t0) * 1000
    print(f"  100 embeddings (batch): {batch_time:.1f}ms ({100/(batch_time/1000):.0f} emb/s)")

    t0 = time.perf_counter()
    for t in texts[:20]:
        emb.embed(t + " unique")  # avoid cache
    single_time = (time.perf_counter() - t0) * 1000
    print(f"  20 embeddings (single): {single_time:.1f}ms ({20/(single_time/1000):.0f} emb/s)")
    print()

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


def _with_bias(scorer, pos_bias, query, candidates, budget):
    scored = scorer.score(query, candidates)
    ages = [sc.candidate.age for sc in scored]
    biases = pos_bias.bias_batch(ages)
    for sc, b in zip(scored, biases):
        sc.score = max(0.0, sc.score + b)
    scored.sort(key=lambda s: s.score, reverse=True)
    return scorer.select(scored, budget)


def _with_classifier(scorer, classifier, query, candidates, budget):
    scored = scorer.score(query, candidates)
    cosine_sims = [sc.cosine_similarity for sc in scored]
    probs = classifier.predict_batch(query, candidates, cosine_sims)
    # Blend: 0.6 * cosine_decay + 0.4 * classifier
    for sc, prob in zip(scored, probs):
        sc.score = 0.6 * sc.score + 0.4 * prob
    scored.sort(key=lambda s: s.score, reverse=True)
    return scorer.select(scored, budget)


if __name__ == "__main__":
    run_scale_benchmark()
