# Roadmap

## Phase 1: PoC Validation (COMPLETE)

- [x] Design attention-inspired scoring function (cosine sim × decay)
- [x] Implement 4 context strategies (No-Pruning, Sliding-Window, A3TK, Adaptive)
- [x] Build 7 test scenarios (far-back recall, cross-session, irrelevant-heavy)
- [x] Validate locally with real embeddings: 34-56% token reduction, 100% quality
- [x] Create Azure GPU benchmark infrastructure
- [x] Execute Azure benchmark on T4 VM (northeurope)
- [x] Implement backup/restore to Azure Blob Storage
- [x] Release v0.1.0

## Phase 2: Scale & Validate (COMPLETE)

- [x] Add 500/750/1000-turn scenarios to stress default budget
- [x] Implement GPT-2 tokenizer as offline fallback (accurate counts)
- [x] Fix LTM scoring: persistent memories get age=0 (no decay penalty)
- [x] Validate 46-50% reduction on long conversations, 100% fact preservation
- [x] Add scenario generator for parameterized long conversations
- [x] Update documentation (memory, goals, ll, roadmap, todo)
- [x] Push Phase 2 results to GitHub

**Key Result:** 46-50% token reduction on 500-1000 turn conversations with
100% key fact preservation. Core hypothesis validated at scale.

## Phase 3: Production Integration (IN PROGRESS)

### 3a: Standalone Scorer Module

- [ ] Extract `attn_scorer/` package with clean public API
- [ ] `Scorer.score(query, candidates) → list[ScoredCandidate]`
- [ ] `Scorer.select(query, candidates, budget) → SelectionResult`
- [ ] `build_context(query, stm, ltm, budget) → ContextResult`
- [ ] Support pluggable embedding backends (local, OpenAI, Cohere)
- [ ] Support pluggable vector stores (brute-force, FAISS, external)

### 3b: Performance & Scale

- [ ] FAISS backend for 10k+ memory entries (ANN search)
- [ ] Latency target: < 50ms for scoring 200 candidates
- [ ] Batch embedding computation
- [ ] Disk-backed embedding cache with TTL
- [ ] Async support for API-based embeddings

### 3c: Integration

- [ ] Plugin interface for A3TK orchestrator
- [ ] Dynamic token budgeting (query complexity → budget adjustment)
- [ ] OpenAI Ada / Cohere embedding support
- [ ] pip-installable package (`pip install attn-scorer`)
- [ ] Configuration via environment variables or config object

### 3d: Validation

- [ ] Benchmark with GPT-4.1 on production-like conversations
- [ ] Latency profiling under load (100 concurrent requests)
- [ ] Memory usage profiling (10k LTM entries)
- [ ] Release v0.2.0

## Phase 4: Advanced Scoring (COMPLETE)

- [x] Train binary relevance classifier (BERT-small, fine-tuned)
- [x] Cross-encoder re-ranking for top-K candidates
- [x] Trainable positional biases (learnable ALiBi variant)
- [x] Multi-turn query context (score against query window, not just latest)
- [x] Evaluation framework comparing all strategies
- [x] Synthetic training dataset generator (12 templates)
- [ ] Evaluate on 128k+ context models — measure if pruning still adds value

## Phase 5: Scale & Generalize

- [ ] Multi-modal memory scoring (images, tables, code blocks)
- [ ] Cross-agent memory sharing with selective context
- [ ] Real-time adaptation: update scores as conversation evolves
- [ ] Feedback loop: learn from user corrections
- [ ] Streaming context assembly (for real-time chat)
- [ ] Publish findings and open-source the scorer module
