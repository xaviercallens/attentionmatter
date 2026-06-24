# Adaptive Attention Token Reduction PoC

An attention-inspired context filtering mechanism that reduces LLM prompt token
length by 30-56% while maintaining 100% answer quality on critical information
retrieval tasks.

## What It Does

In conversational AI systems, long conversations accumulate context that exceeds
LLM token limits. This PoC implements a **relevance scoring function** that acts as
a pseudo-attention mechanism over conversation history and long-term memory:

```
score(chunk) = cosine_similarity(query, chunk) × decay_factor^age
```

Only the highest-scoring chunks are included in the prompt, keeping it within a token
budget while preserving the information the user actually needs.

## Results

Tested across 7 scenarios (5-101 conversation turns):

| Strategy | Avg Token Savings | Key Fact Pass Rate |
|----------|------------------|--------------------|
| No-Pruning (baseline) | 0% | 100% |
| Sliding-Window | ~80% | 14% |
| A3TK-Heuristic | ~0-5% | 100% |
| **Adaptive (ours)** | **30-56%** | **100%** |

The Adaptive strategy achieves the best trade-off: significant token reduction
without losing critical information.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with dummy LLM (no GPU needed, structural validation)
python run_poc.py --dummy-llm

# Run with real embeddings + dummy LLM (shows actual semantic scoring)
python run_poc.py --dummy-llm --real-embeddings

# Run with real LLM (requires GPU with 16GB+ VRAM)
pip install -r requirements-gpu.txt
python run_poc.py

# Dry run (print config and scenarios only)
python run_poc.py --dry-run
```

## CLI Options

```
--dummy-llm            Use stub LLM (no GPU required)
--real-embeddings      Use real sentence-transformer embeddings with dummy LLM
--decay-factor FLOAT   Override decay factor (default: 0.95)
--token-budget-ratio F Override token budget ratio (default: 0.8)
--config PATH          Load config from JSON file
--no-chart             Skip chart generation
--dry-run              Print config without running
```

## Architecture

```
run_poc.py
├── src/config.py              # Central configuration
├── src/embedding.py           # Embedding service (all-MiniLM-L6-v2)
├── src/tokenizer_service.py   # Token counting
├── src/memory.py              # STM (conversation) + LTM (facts)
├── src/llm.py                 # LLM client + DummyLLM
├── src/strategies/
│   ├── base.py                # Strategy protocol
│   ├── no_pruning.py          # Full context baseline
│   ├── sliding_window.py      # Last-N messages
│   ├── a3tk_heuristic.py      # Keyword importance + summarization
│   └── adaptive.py            # Cosine scoring × recency decay (core)
├── src/scenarios.py           # 7 test scenarios
├── src/evaluator.py           # Key-fact presence check
├── src/runner.py              # Experiment matrix runner
└── src/reporter.py            # Results table + CSV + charts
```

## Azure Benchmark

For full GPU benchmarking on Azure:

```bash
./azure/provision.sh     # Create V100 VM (~$3/hr)
./azure/deploy.sh docker # Upload + run benchmark
./azure/teardown.sh      # Delete resources (stops billing!)
```

See [BENCHMARK.md](BENCHMARK.md) for detailed instructions.

## Backup & Restore

Archive results and model caches to Azure Blob Storage for fast restart:

```bash
./azure/backup.sh        # Archive to blob storage
./azure/restore.sh       # Restore from latest archive
```

## How the Scoring Works

1. Compute embedding for the user's current query.
2. Compute embeddings for each candidate context chunk (messages + memories).
3. Score each candidate: `cosine_sim(query, candidate) × 0.95^age_in_turns`
4. Sort by score descending.
5. Select top candidates until token budget is reached.
6. Reassemble selected items chronologically for the prompt.

The recency decay factor (`0.95^age`) gently penalizes older items, analogous to
ALiBi's distance bias in Transformer attention. But high semantic relevance can
override recency — a booking code from turn 4 still ranks above chit-chat from
turn 60 if the user asks about it.

## Project Documentation

- [memory.md](memory.md) — Current state, achievements, quick restart guide
- [roadmap.md](roadmap.md) — Development phases and milestones
- [todo.md](todo.md) — Actionable task list
- [ll.md](ll.md) — Lessons learned
- [BENCHMARK.md](BENCHMARK.md) — Azure benchmark instructions

## License

Internal PoC — not for external distribution.
