# Phase 6 Implementation Tasks

## Task 1: Finalize AsyncScorer and Async OpenAI Backend

### Description
Complete the async scoring pipeline with native async for API backends and
thread-pool delegation for local backends. Add concurrency controls.

### Requirements Addressed
- Requirement 1 (Async Embedding Backend)
- Requirement 2 (Async Selection and Context Assembly)

### Steps
1. Finalize `attn_scorer/async_scorer.py` (already scaffolded):
   - Ensure semaphore correctly limits concurrency.
   - Add thread-safe embedding cache (threading.Lock).
   - Add `score_multiple_queries()` using `asyncio.gather`.
2. Create `attn_scorer/embeddings/openai_async.py`:
   - Use `httpx.AsyncClient` for native async embedding calls.
   - Handle rate limits with exponential backoff.
   - Connection pooling via persistent client.
3. Add tests: `tests/test_async_scorer.py`
   - Test basic async scoring correctness.
   - Test concurrent execution (10 simultaneous calls).
   - Test semaphore limiting (verify max_concurrent_embeds honored).
   - Test result equivalence with sync scorer.

### Acceptance Criteria
- `await AsyncScorer.build_context(...)` returns correct ContextResult.
- 100 concurrent calls complete with < 2× single-call latency.
- Sync and async scorers produce equivalent results for same input.

---

## Task 2: Finalize Observability (Metrics, Tracing, Health)

### Description
Complete the observability module and integrate metrics recording into the
Scorer and AsyncScorer hot paths.

### Requirements Addressed
- Requirement 3 (Metrics)
- Requirement 4 (Tracing Hooks)
- Requirement 5 (Health Check)

### Steps
1. Finalize `attn_scorer/observability.py` (already scaffolded):
   - Ensure `export_prometheus()` produces valid output.
   - Add counter tracking: cache_hits, cache_misses, budget_exceeded.
2. Integrate into `scorer.py` and `async_scorer.py`:
   - Accept optional `metrics: MetricsCollector | None` parameter.
   - Record scoring_latency_ms and selection_latency_ms after each call.
   - Emit tracing hooks (on_score_start, on_score_end) if registered.
   - Zero overhead when metrics=None (guard with `if self._metrics:`).
3. Add tests: `tests/test_observability.py`
   - Verify latency recording.
   - Verify Prometheus export format (parseable).
   - Verify health check returns correct structure.
   - Verify hooks fire and exceptions don't crash scoring.

### Acceptance Criteria
- Prometheus export is valid text format.
- Health check completes in < 500ms.
- Metrics recording adds < 0.1ms overhead per call.

---

## Task 3: Production-Scale Dataset Generator

### Description
Build a dataset generator that produces realistic production-representative
conversation data with long messages, mixed topics, and hard negatives.

### Requirements Addressed
- Requirement 6 (Production-Scale Evaluation Dataset)

### Steps
1. Create `attn_scorer/evaluation/__init__.py` and
   `attn_scorer/evaluation/prod_dataset.py`:
   - Define `ProdScenario`, `SessionConversation`, `ProdMessage` dataclasses.
   - Implement `ProdDatasetGenerator`:
     - Message generator: 50-200 words per message using templated paragraphs.
     - Topic mixer: 30% domain-relevant, 70% tangential/filler.
     - Hard negative injector: 3 messages per scenario that are semantically
       similar to the query but NOT the correct answer.
     - Multi-session support: key facts split across 2-3 sessions.
   - Implement `generate(config) -> list[ProdScenario]` with configs for
     500, 1000, and 2000 turn scenarios.
2. Use GPT-2 tokenizer to verify token counts exceed budget.
3. Add tests: verify generated data meets density/size requirements.

### Acceptance Criteria
- Generated 2000-turn conversation exceeds 60k tokens (GPT-2).
- 30% of messages contain domain-relevant content.
- Each scenario has 3+ hard negatives.
- At least 2 scenarios span multiple sessions.

---

## Task 4: Production Benchmark Runner

### Description
Implement a benchmark runner that executes scoring strategies on the production
dataset and reports latency, token reduction, and quality metrics.

### Requirements Addressed
- Requirement 7 (Production-Scale Benchmark)

### Steps
1. Create `attn_scorer/evaluation/prod_benchmark.py`:
   - `ProdBenchmarkRunner` with `run_sync()` and `run_async()`.
   - Measures: token reduction, key fact preservation, scoring latency,
     false positive rate (hard negatives in selection).
   - `BenchmarkReport` with summary statistics and per-scenario breakdown.
2. Create `run_prod_benchmark.py` (entry point script):
   - Loads or generates dataset.
   - Runs sync benchmark with all strategies.
   - Runs async benchmark with 50 concurrent requests.
   - Prints report and persists CSV.
3. Add assertions: reduction >= 30%, key_fact_rate >= 95%, latency < 50ms.

### Acceptance Criteria
- Benchmark produces results table with all metrics.
- Sync latency < 50ms for 500 candidates.
- Async (50 concurrent) avg latency < 100ms.
- Token reduction 30-50% on 2000-turn scenarios.

---

## Task 5: PyPI Build and Publish Workflow

### Description
Ensure the package builds correctly and add a GitHub Actions workflow that
publishes to PyPI on release tag.

### Requirements Addressed
- Requirement 8 (PyPI Package Build)
- Requirement 9 (PyPI Publication)
- Requirement 10 (Package Quality Gates)

### Steps
1. Update `pyproject.toml`:
   - Bump version to 1.1.0.
   - Add `[async]` extra dependency (`httpx>=0.25`).
   - Verify all metadata (description, keywords, URLs, classifiers).
   - Add `[project.scripts]` for optional CLI entry point.
2. Verify local build:
   - `pip install build && python -m build`
   - `twine check dist/*`
   - Install wheel in fresh venv, verify import works.
3. Create `.github/workflows/publish.yml`:
   - Trigger on release published.
   - Build sdist + wheel.
   - Run `twine check`.
   - Publish via PyPI trusted publishing (OIDC).
4. Update `.github/workflows/ci.yml`:
   - Add mypy type checking step.
   - Add coverage reporting (target ≥ 80%).
   - Add build verification step (`python -m build`).

### Acceptance Criteria
- `python -m build` produces valid .tar.gz and .whl.
- `twine check dist/*` passes.
- Fresh venv install + `from attn_scorer import Scorer` works.
- Publish workflow triggers on release tag.

---

## Task 6: Final Integration, Docs, and Release

### Description
Run the full production benchmark, update all documentation, and tag v1.1.0.

### Requirements Addressed
- All requirements (final validation)

### Steps
1. Run `run_prod_benchmark.py` — verify all targets met.
2. Run full test suite (`pytest tests/ -v`).
3. Run `python -m build && twine check dist/*`.
4. Update documentation:
   - `README.md`: add async usage example, PyPI install instructions.
   - `memory.md`: update with Phase 6 achievements.
   - `roadmap.md`: mark Phase 6 complete.
   - `ll.md`: add Phase 6 lessons.
   - `CHANGELOG.md`: create with all releases.
5. Commit, push, tag v1.1.0, create GitHub release.
6. (Optional) Publish to PyPI if trusted publishing is configured.

### Acceptance Criteria
- All tests pass (31+ existing + new async/observability/benchmark tests).
- Production benchmark meets all targets.
- v1.1.0 release created on GitHub.
- Package builds cleanly.

---

## Dependency Order

```
Task 1 (Async scorer)       ─┐
Task 2 (Observability)       ├── can run in parallel
Task 3 (Prod dataset)       ─┘
     │
     ▼
Task 4 (Prod benchmark) ← depends on Tasks 1-3
     │
     ▼
Task 5 (PyPI build/publish) ← depends on Task 4 (validates package)
     │
     ▼
Task 6 (Final release) ← depends on all above
```

Tasks 1, 2, and 3 can proceed in parallel.
Task 4 integrates them.
Tasks 5 and 6 are sequential finalization.
