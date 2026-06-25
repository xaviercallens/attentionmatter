# Phase 6 Design — Production-Ready Release

## Overview

This phase adds three production capabilities to `attn_scorer`:

1. **Async support** — `AsyncScorer` for non-blocking scoring in async frameworks.
2. **Real-world evaluation** — production-representative dataset and benchmark.
3. **PyPI publication** — automated build, quality gates, and publish on release tag.

The design preserves backward compatibility: the sync `Scorer` API is unchanged.
Async is additive. Observability is opt-in.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  FastAPI / aiohttp / A3TK Orchestrator                          │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
     │ AsyncScorer  │    │   Scorer    │    │  Observability  │
     │  (asyncio)   │    │   (sync)    │    │  Metrics/Trace  │
     └──────┬───────┘    └──────┬──────┘    └────────┬────────┘
            │                   │                    │
     ┌──────▼───────────────────▼─────┐              │
     │       EmbeddingBackend          │◄─────────────┘
     │  ┌────────┐  ┌───────────────┐ │   (records latency)
     │  │ Local  │  │ OpenAI Async  │ │
     │  │(thread)│  │ (httpx.Async) │ │
     │  └────────┘  └───────────────┘ │
     └─────────────────────────────────┘
              │
     ┌────────▼────────┐
     │   VectorStore   │    (FAISS / brute-force)
     └─────────────────┘
```

### Data Flow (Async)

1. Application `await`s `AsyncScorer.build_context(query, messages, memories)`.
2. `AsyncScorer` acquires semaphore slot, embeds query via `run_in_executor`
   (local) or `httpx.AsyncClient` (API).
3. Batch-embeds unembedded candidates concurrently.
4. Runs scoring (CPU-bound, sub-1ms inline — no executor needed).
5. Selects within budget, reorders chronologically.
6. Records metrics if collector is active.
7. Returns `ContextResult`.

---

## Component Design

### 1. AsyncScorer (`attn_scorer/async_scorer.py`)

```python
class AsyncScorer:
    def __init__(self, config, embedding, token_counter, max_concurrent_embeds=4): ...
    async def score(self, query, candidates) -> list[ScoredCandidate]: ...
    async def select(self, scored, budget) -> ContextResult: ...
    async def build_context(self, query, messages, memories, budget) -> ContextResult: ...
    async def score_multiple_queries(self, queries, candidates, budget) -> list[ContextResult]: ...
```

**Concurrency control:**
- `asyncio.Semaphore(max_concurrent_embeds)` prevents overwhelming the embedding
  backend.
- Local embeddings: wrapped in `loop.run_in_executor(None, ...)` using the default
  thread pool.
- API embeddings: native async HTTP calls (no thread pool needed).

**Thread safety:**
- The scorer is stateless per call (embeddings are stored on candidates).
- The embedding cache (dict) is NOT thread-safe for writes in the executor model.
  Solution: use a thread-safe cache (`threading.Lock` around dict updates) or
  accept rare duplicate computations (idempotent, no correctness issue).

### 2. Async OpenAI Backend (`attn_scorer/embeddings/openai_async.py`)

```python
class AsyncOpenAIEmbeddingBackend:
    async def embed(self, text: str) -> np.ndarray: ...
    async def embed_batch(self, texts: list[str]) -> np.ndarray: ...
```

- Uses `httpx.AsyncClient` for native async HTTP.
- Rate-limit aware: respects `Retry-After` headers.
- Connection pooling via persistent client instance.

### 3. Observability (`attn_scorer/observability.py`)

Already partially implemented. Design completes:

```python
class MetricsCollector:
    def record_latency(self, operation, duration_ms, labels): ...
    def record_counter(self, name, value): ...
    def get_summary(self) -> dict: ...
    def export_prometheus(self) -> str: ...

class TracingHook:
    def register(self, event, callback): ...
    def emit(self, event, **kwargs): ...

class HealthCheck:
    def check(self) -> dict: ...
```

**Integration point:** The `Scorer` and `AsyncScorer` accept an optional
`MetricsCollector` and `TracingHook`. If None (default), no overhead is imposed.

### 4. Production Evaluation Dataset (`attn_scorer/evaluation/prod_dataset.py`)

```python
@dataclass
class ProdScenario:
    id: str
    conversations: list[SessionConversation]  # multi-session
    query: str
    key_facts: list[str]  # multiple facts to find
    hard_negatives: list[str]  # distractors

class ProdDatasetGenerator:
    def generate(self, config: DatasetConfig) -> list[ProdScenario]: ...
```

**Dataset characteristics:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Messages per turn | 50-200 words | Mirrors real agent responses |
| Turns per conversation | 500-2000 | Production-length sessions |
| Total tokens | 15k-60k | Exceeds 8k budget → forces pruning |
| Info density | 30% relevant | Not all filler, but not all signal |
| Hard negatives | 3 per scenario | Tests false-positive resilience |
| Sessions per scenario | 1-3 | Multi-session memory retrieval |
| Scenarios | 10-15 | Statistical significance |

**Message generation:** Uses Markov chain or templated paragraph generators to
produce realistic multi-sentence messages (not single-line chit-chat).

### 5. Production Benchmark Runner (`attn_scorer/evaluation/prod_benchmark.py`)

```python
class ProdBenchmarkRunner:
    def run_sync(self, scenarios, strategies, budget) -> BenchmarkReport: ...
    async def run_async(self, scenarios, concurrency=50) -> BenchmarkReport: ...

@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult]
    summary: dict  # avg reduction, pass rate, latency stats
    async_stats: dict  # concurrency metrics
```

**What it measures:**

| Metric | Target | How |
|--------|--------|-----|
| Token reduction | 30-50% | (full - selected) / full |
| Key fact preservation | ≥ 95% | Substring presence check |
| Scoring latency (sync) | < 50ms / 500 cand | `time.perf_counter` |
| Scoring latency (async, 50 conc) | < 100ms avg | `asyncio.gather` timing |
| Throughput | > 10k cand/s | candidates / total_time |
| False positive rate | < 5% | Hard negatives selected / total |

### 6. PyPI Build & Publish

**pyproject.toml** (already exists, needs minor additions):

```toml
[project]
name = "attn-scorer"
version = "1.1.0"
# ... existing config ...

[project.optional-dependencies]
async = ["httpx>=0.25"]
# ... existing extras ...
```

**GitHub Actions publish workflow** (`.github/workflows/publish.yml`):

```yaml
on:
  release:
    types: [published]
jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # OIDC for trusted publishing
    steps:
      - checkout
      - setup-python
      - pip install build twine
      - python -m build
      - twine check dist/*
      - pypi-publish (trusted publisher)
```

**Quality gates before publish:**
1. CI must pass (lint + type check + tests)
2. `twine check` validates metadata
3. Test install in clean venv (via separate CI job)

---

## Data Models

| Model | Fields | Purpose |
|-------|--------|---------|
| `ProdScenario` | id, conversations, query, key_facts, hard_negatives | Eval scenario |
| `SessionConversation` | session_id, messages, seed_memories | One session |
| `ProdMessage` | text, role, turn, word_count, topic | Realistic message |
| `BenchmarkResult` | scenario_id, strategy, token_count, facts_found, latency_ms | Per-run result |
| `BenchmarkReport` | results, summary, async_stats | Full report |

---

## Error Handling

| Condition | Handling |
|-----------|----------|
| Embedding API rate limit (429) | Exponential backoff with jitter; retry 3× |
| Embedding API timeout | Raise after 30s; log; return partial result |
| Concurrent semaphore exhausted | Queue (asyncio.Semaphore handles this) |
| PyPI upload auth failure | Fail workflow; alert via GitHub notification |
| Health check embedding fails | Return unhealthy; log error details |
| Build produces invalid metadata | `twine check` catches; CI fails |

---

## Testing Strategy

### Unit Tests (new for Phase 6)

- `test_async_scorer.py`: async scoring correctness, concurrency behavior,
  semaphore limiting.
- `test_observability.py`: metrics recording, Prometheus export format,
  health check pass/fail.
- `test_prod_dataset.py`: generated data meets size/density requirements.

### Integration Tests

- Async benchmark with mock embedding: 50 concurrent requests, verify no race
  conditions.
- Full sync + async result equivalence: same input → same output.
- Package build + install in clean venv (CI job).

### Performance Tests

- Latency regression: assert < 50ms for 500 candidates.
- Throughput regression: assert > 10k candidates/second.
- Async overhead: assert < 2× single-request latency at 100 concurrency.

---

## Verification Against Success Criteria

| Criterion | How Verified |
|-----------|-------------|
| Async < 2× latency at 100 conc | `test_async_benchmark` measures |
| 30-50% reduction on 2000 turns | `prod_benchmark` with GPT-2 tokenizer |
| ≥ 95% key fact preservation | Substring evaluator on prod scenarios |
| < 50ms scoring latency | `bench.py` regression test |
| Clean PyPI install | CI job: fresh venv + import test |
| Prometheus export valid | Unit test parses output |
| CI green on tag | All jobs must pass before publish |

---

## Migration & Compatibility

- **No breaking changes.** `Scorer` API is unchanged.
- `AsyncScorer` is a new class; existing sync users are unaffected.
- Observability is opt-in (pass `metrics=None` to disable).
- New `[async]` extra adds `httpx`; base install unchanged.
- `attn_scorer.__version__` bumps to `1.1.0`.

---

## Timeline Estimate

| Task | Effort |
|------|--------|
| Async scorer + async OpenAI backend | ~2 hours |
| Observability finalization (metrics in scorer, health check) | ~1 hour |
| Production dataset generator | ~2 hours |
| Production benchmark runner + results | ~1 hour |
| PyPI build + publish workflow | ~1 hour |
| Tests + quality gates | ~1 hour |
| Documentation + release | ~30 min |
| **Total** | **~8-9 hours** |
