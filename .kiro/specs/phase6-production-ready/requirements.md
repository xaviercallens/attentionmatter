# Phase 6 Requirements — Production-Ready Release

## Introduction

This phase transforms `attn_scorer` from a validated PoC into a production-grade
Python library suitable for real adoption in high-concurrency LLM orchestration
systems. The three pillars are:

1. **Async support** — non-blocking scoring for async web frameworks and
   multi-tenant orchestrators handling hundreds of concurrent requests.
2. **Real-world evaluation** — validation against production-representative
   conversation data with realistic message lengths, topic distributions, and
   multi-session patterns.
3. **PyPI publication** — a properly packaged, versioned, and distributable
   library that teams can install with `pip install attn-scorer`.

### Glossary

- **Async scorer:** An `asyncio`-native scorer that embeds and scores without
  blocking the event loop.
- **Production-like data:** Synthetic conversations modeled on real orchestrator
  traffic — 50-200 words per message, mixed intents, multi-session histories, and
  realistic token counts (8k-32k total per conversation).
- **PyPI:** The Python Package Index; the standard registry for Python libraries.
- **Observability:** Metrics (latency histograms, counters), tracing hooks, and
  health checks exposable to monitoring systems.

---

## Requirements

### Requirement 1: Async Embedding Backend

**User Story:** As an orchestrator developer, I want to call the scorer from an
async request handler without blocking the event loop, so that my service can
handle hundreds of concurrent scoring requests.

#### Acceptance Criteria

1. WHEN `AsyncScorer.score()` is awaited THEN it SHALL embed the query and
   candidates without blocking the event loop (using `run_in_executor` for local
   models or native async for API-based backends).
2. WHEN `AsyncScorer.build_context()` is awaited THEN it SHALL return a
   `ContextResult` identical in structure and correctness to the sync `Scorer`.
3. WHEN multiple `build_context()` coroutines run concurrently THEN the system
   SHALL limit concurrent embedding calls to a configurable `max_concurrent_embeds`
   (default 4) via a semaphore.
4. WHEN an API-based embedding backend (OpenAI) is used THEN the async scorer
   SHALL use native `httpx.AsyncClient` calls rather than wrapping sync calls in
   an executor.
5. WHEN the async scorer is benchmarked with 100 concurrent requests THEN average
   latency SHALL be < 2× the single-request latency (demonstrating concurrency
   benefit rather than serial bottleneck).

### Requirement 2: Async Selection and Context Assembly

**User Story:** As an orchestrator developer, I want the full scoring-to-assembly
pipeline to be async, so I can integrate it into FastAPI/aiohttp handlers without
special threading.

#### Acceptance Criteria

1. WHEN `AsyncScorer.build_context()` is called THEN all sub-operations (embed,
   score, select, reorder) SHALL be awaitable.
2. WHEN `score_multiple_queries()` is called with N queries THEN it SHALL execute
   all N concurrently via `asyncio.gather` and return N `ContextResult` objects.
3. WHEN the async scorer is used inside a `pytest-asyncio` test THEN it SHALL
   function correctly without special configuration.

### Requirement 3: Observability — Metrics

**User Story:** As a platform engineer, I want scoring latency, throughput, and
cache hit rates exposed as metrics, so I can monitor the scorer in production.

#### Acceptance Criteria

1. WHEN any scoring call completes THEN the system SHALL record
   `scoring_latency_ms` and `selection_latency_ms` in a `MetricsCollector`.
2. WHEN metrics are requested THEN the system SHALL provide summary statistics:
   count, mean, p50, p95, p99, max for each latency metric.
3. WHEN `export_prometheus()` is called THEN the system SHALL return a valid
   Prometheus text-format string with all histograms and counters.
4. WHEN a global metrics collector is enabled THEN it SHALL be opt-in (not active
   unless explicitly initialized by the application).
5. WHEN counters are recorded THEN the system SHALL track:
   `candidates_scored`, `candidates_selected`, `candidates_omitted`,
   `cache_hits`, `cache_misses`, `budget_exceeded_count`.

### Requirement 4: Observability — Tracing Hooks

**User Story:** As a platform engineer, I want to integrate scorer events into my
distributed tracing system, so I can correlate scoring latency with end-to-end
request traces.

#### Acceptance Criteria

1. WHEN a tracing hook is registered for `on_score_start` THEN the system SHALL
   invoke it with `(query, num_candidates)` at the beginning of every score call.
2. WHEN a tracing hook is registered for `on_score_end` THEN the system SHALL
   invoke it with `(query, num_selected, latency_ms)` after selection completes.
3. WHEN hook callbacks raise exceptions THEN the system SHALL log a warning and
   continue without interrupting the scoring operation.
4. WHEN no hooks are registered THEN the system SHALL impose zero overhead (no
   function calls or allocations on the hot path).

### Requirement 5: Observability — Health Check

**User Story:** As a platform engineer, I want a health check endpoint I can wire
to a readiness probe, so that load balancers only route to healthy scorer instances.

#### Acceptance Criteria

1. WHEN `HealthCheck.check()` is called THEN it SHALL verify that the embedding
   backend can embed a short test string and the scorer can score a synthetic
   candidate.
2. WHEN all checks pass THEN it SHALL return `{"healthy": True, "checks": {...}}`.
3. WHEN any check fails THEN it SHALL return `{"healthy": False, "checks": {...}}`
   with the specific failure reason.
4. WHEN the health check is called THEN it SHALL complete in < 500ms under normal
   conditions.

### Requirement 6: Production-Scale Evaluation Dataset

**User Story:** As a PoC researcher, I want evaluation data that mirrors real
orchestrator traffic, so that benchmark results predict production behavior.

#### Acceptance Criteria

1. WHEN the dataset is generated THEN each conversation SHALL have messages of
   50-200 words (realistic length, not short chit-chat).
2. WHEN the dataset is generated THEN it SHALL include conversations of 500, 1000,
   and 2000 turns with total token counts of 15k, 30k, and 60k+ tokens.
3. WHEN the dataset is generated THEN at least 30% of messages SHALL be
   domain-relevant (not pure filler) to simulate realistic information density.
4. WHEN the dataset is generated THEN it SHALL include multi-session scenarios
   where key facts span 2-3 sessions stored only in LTM.
5. WHEN the dataset is generated THEN it SHALL include at least 3 "hard negatives" —
   messages semantically similar to the query but NOT the correct answer.
6. WHEN token counts are measured THEN the system SHALL use the GPT-2 tokenizer
   for consistency across environments.

### Requirement 7: Production-Scale Benchmark

**User Story:** As a PoC researcher, I want benchmark results on production-scale
data that prove the scorer meets latency and quality targets under realistic load.

#### Acceptance Criteria

1. WHEN the benchmark runs THEN it SHALL process conversations exceeding the token
   budget at default ratio (80% of 8192 = 6553 tokens), demonstrating natural
   pruning without artificial budget tightening.
2. WHEN the benchmark completes THEN it SHALL report: token reduction (target
   30-50%), key fact preservation (target ≥ 95%), scoring latency (target < 50ms
   for 500 candidates), and throughput (candidates/second).
3. WHEN the async benchmark runs with 50 concurrent requests THEN average latency
   SHALL remain below 100ms per request.
4. WHEN results are collected THEN the system SHALL produce a comparison table and
   persist to CSV.

### Requirement 8: PyPI Package Build

**User Story:** As a developer, I want to install attn-scorer via
`pip install attn-scorer`, so I can add it to my project without cloning the repo.

#### Acceptance Criteria

1. WHEN `python -m build` is run THEN it SHALL produce both sdist (.tar.gz) and
   wheel (.whl) artifacts without errors.
2. WHEN the wheel is installed in a fresh virtualenv THEN
   `from attn_scorer import Scorer` SHALL work without ImportError.
3. WHEN installed with `pip install attn-scorer[local]` THEN sentence-transformers
   and torch SHALL be installed as extras.
4. WHEN installed with `pip install attn-scorer` (no extras) THEN only numpy SHALL
   be required; the module SHALL raise clear ImportErrors when optional backends are
   used without their dependencies.
5. WHEN `attn_scorer.__version__` is accessed THEN it SHALL return the correct
   version string matching pyproject.toml.

### Requirement 9: PyPI Publication

**User Story:** As a library maintainer, I want an automated publish workflow, so
that tagging a release on GitHub automatically publishes to PyPI.

#### Acceptance Criteria

1. WHEN a GitHub release is tagged (v*) THEN a GitHub Actions workflow SHALL build
   and upload the package to PyPI using trusted publishing (OIDC).
2. WHEN the package is uploaded THEN `pip install attn-scorer` SHALL resolve from
   PyPI within 5 minutes.
3. WHEN the package metadata is viewed on PyPI THEN it SHALL show the correct
   description, version, author, keywords, and links.
4. WHEN a pre-release (e.g., v1.1.0rc1) is tagged THEN it SHALL be published as a
   pre-release on PyPI (not shown by default in `pip install`).

### Requirement 10: Package Quality Gates

**User Story:** As a library maintainer, I want automated quality checks before
every release, so that broken code never reaches PyPI.

#### Acceptance Criteria

1. WHEN CI runs THEN it SHALL execute: lint (ruff), type check (mypy), unit tests
   (pytest), and a smoke test (examples/usage.py).
2. WHEN any quality gate fails THEN the publish workflow SHALL NOT proceed.
3. WHEN the package is built THEN it SHALL pass `twine check` (metadata validation).
4. WHEN tests run THEN coverage SHALL be ≥ 80% on the `attn_scorer/` package.

---

## Success Criteria

- Async scorer handles 100 concurrent requests with < 2× single-request latency.
- Production-scale benchmark shows 30-50% token reduction on 2000-turn
  conversations with ≥ 95% key fact preservation and < 50ms scoring latency.
- Package installs cleanly from PyPI in a fresh virtualenv.
- CI pipeline (lint + type check + test + build + publish) runs green on tag.
- Prometheus metrics export produces valid output parseable by a scraper.

## Out of Scope

- Hosting a documentation website (Sphinx/MkDocs) — future work.
- Production deployment automation (Kubernetes manifests, Helm charts).
- Load testing beyond 100 concurrent requests.
- Support for languages other than Python.
