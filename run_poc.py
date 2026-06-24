#!/usr/bin/env python3
"""
Adaptive Attention Token Reduction PoC — Main Entry Point

Usage:
    python run_poc.py                          # Full run with real LLM (requires GPU)
    python run_poc.py --dummy-llm              # Fast run with stub LLM (no GPU needed)
    python run_poc.py --decay-factor 1.0       # Override decay factor
    python run_poc.py --dry-run                # Print config and scenarios, no execution
    python run_poc.py --config path/to/cfg.json # Load config from file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.embedding import DummyEmbeddingService, EmbeddingService
from src.evaluator import Evaluator
from src.llm import DummyLLMClient, LLMClient
from src.memory import MemoryManager
from src.reporter import Reporter
from src.runner import ExperimentRunner
from src.scenarios import load_scenarios
from src.strategies.adaptive import AdaptiveStrategy
from src.strategies.a3tk_heuristic import A3TKHeuristicStrategy
from src.strategies.no_pruning import NoPruningStrategy
from src.strategies.sliding_window import SlidingWindowStrategy
from src.tokenizer_service import DummyTokenizerService, TokenizerService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Attention Token Reduction PoC"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file with parameter overrides."
    )
    parser.add_argument(
        "--dummy-llm", action="store_true",
        help="Use DummyLLMClient (no GPU required, for structural testing)."
    )
    parser.add_argument(
        "--decay-factor", type=float, default=None,
        help="Override the decay factor (e.g., 1.0 for no bias, 0.95 default)."
    )
    parser.add_argument(
        "--token-budget-ratio", type=float, default=None,
        help="Override the token budget ratio (default 0.8)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and scenario list without running the experiment."
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="Skip chart generation."
    )
    parser.add_argument(
        "--real-embeddings", action="store_true",
        help="Use real sentence-transformer embeddings (requires model download). "
             "Combines with --dummy-llm to test scoring without GPU."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Configuration ---
    if args.config:
        cfg = Config.from_file(args.config)
    else:
        cfg = Config()

    # Apply CLI overrides
    if args.decay_factor is not None:
        cfg.decay_factor = args.decay_factor
    if args.token_budget_ratio is not None:
        cfg.token_budget_ratio = args.token_budget_ratio

    cfg.seed_everything()

    # --- Dry run ---
    if args.dry_run:
        print("=== Configuration ===")
        for field_name in cfg.__dataclass_fields__:
            print(f"  {field_name}: {getattr(cfg, field_name)}")
        print(f"  token_budget (derived): {cfg.token_budget}")
        print()
        scenarios = load_scenarios()
        print(f"=== Scenarios ({len(scenarios)}) ===")
        for s in scenarios:
            print(f"  {s.id}: {s.description}")
            print(f"    conversation_turns={len(s.conversation)}, "
                  f"seed_memories={len(s.seed_memories)}, "
                  f"key_fact='{s.key_fact}'")
        print("\nDry run complete. No experiment executed.")
        return

    # --- Initialize services ---
    print("Initializing services...")
    if args.dummy_llm:
        if args.real_embeddings:
            print("Using real embeddings + DummyTokenizer + DummyLLM.")
            embedding = EmbeddingService(cfg)
            tokenizer = DummyTokenizerService(cfg)
        else:
            print("Using DummyLLMClient + offline embedding/tokenizer (no network needed).")
            embedding = DummyEmbeddingService(cfg)
            tokenizer = DummyTokenizerService(cfg)
        llm = DummyLLMClient(cfg)
    else:
        embedding = EmbeddingService(cfg)
        tokenizer = TokenizerService(cfg)
        print(f"Loading LLM: {cfg.llm_model} (4-bit={cfg.use_4bit})...")
        llm = LLMClient(cfg)

    # --- Strategies ---
    strategies = [
        NoPruningStrategy(cfg, tokenizer, embedding),
        SlidingWindowStrategy(cfg, tokenizer),
        A3TKHeuristicStrategy(cfg, tokenizer, embedding, llm_client=None),
        AdaptiveStrategy(cfg, tokenizer, embedding),
    ]

    # --- Scenarios ---
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios.")

    # --- Evaluator ---
    evaluator = Evaluator(embedding_service=embedding)

    # --- Runner ---
    runner = ExperimentRunner(
        cfg=cfg,
        strategies=strategies,
        llm=llm,
        evaluator=evaluator,
        scenarios=scenarios,
        embedding_service=embedding,
        tokenizer_service=tokenizer,
    )

    # --- Execute ---
    print(f"\nRunning {len(scenarios)} scenarios × {len(strategies)} strategies...")
    records = runner.run()

    # --- Report ---
    reporter = Reporter()
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(reporter.to_table(records))
    print(reporter.summary_stats(records))

    # Persist
    reporter.persist(records, cfg.results_path)

    # Chart
    if not args.no_chart:
        chart_path = cfg.results_path.replace(".csv", "_chart.png")
        reporter.chart(records, chart_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
