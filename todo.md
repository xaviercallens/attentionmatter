# TODO

## Phase 2: Scale & Validate (COMPLETE)

- [x] Create scenario generator for 500+ turn conversations
- [x] Add 3 scenarios at 500, 750, and 1000 turns
- [x] Implement GPT-2 tokenizer as offline fallback
- [x] Run full benchmark showing 46-50% reduction at realistic budget
- [x] Fix LTM scoring (age=0 for persistent memories)
- [x] Validate 100% key fact preservation on all long scenarios
- [x] Document findings and update memory/goals/ll

## Phase 3: Production Integration (IN PROGRESS)

### High Priority

- [ ] Extract scorer as standalone module (`attn_scorer/`)
- [ ] Define clean public API: `score()`, `select()`, `build_context()`
- [ ] Add FAISS backend for 10k+ memory entry vector search
- [ ] Implement dynamic token budgeting (adjust by query complexity)
- [ ] Add OpenAI Ada / Cohere embedding support (API-based interface)
- [ ] Create plugin interface for A3TK orchestrator integration
- [ ] Package as pip-installable module (`pyproject.toml`)

### Medium Priority

- [ ] Add latency benchmarking (time per scoring pass, target < 50ms)
- [ ] Add batch scoring mode for multiple candidates
- [ ] Implement embedding cache with TTL (disk-backed)
- [ ] Add configuration validation and error messages
- [ ] Write integration tests with mock orchestrator
- [ ] Support async embedding calls (for API-based providers)

### Lower Priority

- [ ] Add type-aware weighting (facts=1.2x, narrative=1.0x, chit-chat=0.8x)
- [ ] Add multi-query context (score against last 3 queries)
- [ ] GitHub Actions CI (lint + test on push)
- [ ] Add Prometheus-style metrics (scoring latency, cache hit rate)
- [ ] Release v0.2.0

## Phase 4: Advanced Scoring (PLANNED)

- [ ] Train binary relevance classifier on conversation data
- [ ] Explore trainable ALiBi-style positional biases
- [ ] Evaluate cross-encoder re-ranking for top candidates
- [ ] Test with 128k+ context models (does pruning still help?)
- [ ] Multi-modal memory scoring (images, code blocks)

## Technical Debt

- [ ] Add unit tests for scoring math
- [ ] Add mypy strict mode
- [ ] Remove temporary scripts from repo
- [ ] Standardize logging (replace print with logging module)
- [ ] Add py.typed marker for type checkers
