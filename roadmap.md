# Roadmap

## Phase 1: PoC Validation (COMPLETE)

- [x] Design attention-inspired scoring function (cosine sim × decay)
- [x] Implement 4 context strategies (No-Pruning, Sliding-Window, A3TK, Adaptive)
- [x] Build 7 test scenarios (far-back recall, cross-session, irrelevant-heavy)
- [x] Validate locally with real embeddings: 30-56% token reduction, 100% quality
- [x] Create Azure GPU benchmark infrastructure
- [x] Execute Azure benchmark on T4 VM (northeurope)
- [x] Implement backup/restore to Azure Blob Storage
- [x] Release v0.1.0

## Phase 2: Scale & Validate (IN PROGRESS)

- [ ] Add 500+ turn scenarios to stress default budget
- [ ] Implement GPT-2 tokenizer as offline fallback (accurate counts)
- [ ] Add scenario generator for parameterized long conversations
- [ ] Run with real Mistral-7B on GPU for LLM quality validation
- [ ] Optimize decay factor for long conversations (sweep 0.92-0.98)
- [ ] Add latency benchmarking (time per scoring pass)
- [ ] Generate matplotlib comparison charts
- [ ] Add GitHub Actions CI (--dummy-llm on every push)
- [ ] Implement "regression" scenarios (where cosine scoring may fail)
- [ ] Release v0.2.0 with scale results

## Phase 3: Production Integration

- [ ] Package relevance scorer as standalone pip module (`attn-scorer`)
- [ ] Define plugin interface for A3TK orchestrator
- [ ] Integrate into A3TK context assembly pipeline
- [ ] Replace brute-force with FAISS for 10k+ memory stores
- [ ] Add OpenAI Ada / Cohere embedding support (API-based)
- [ ] Benchmark with GPT-4.1 on production-like conversations
- [ ] Implement dynamic token budgeting based on query complexity
- [ ] A/B test in staging environment
- [ ] Release v1.0.0

## Phase 4: Advanced Scoring

- [ ] Train a small binary classifier for relevance (better than cosine)
- [ ] Explore trainable positional biases (learnable ALiBi variant)
- [ ] Add type-aware weighting (facts > narrative > chit-chat)
- [ ] Multi-turn query context (score against last N queries, not just latest)
- [ ] Evaluate on 128k+ context models — does pruning still help?

## Phase 5: Scale & Generalize

- [ ] Multi-modal memory scoring (images, tables, code blocks)
- [ ] Cross-agent memory sharing with selective context
- [ ] Real-time adaptation: update scores as conversation evolves
- [ ] Feedback loop: learn from user corrections (was pruned info needed?)
- [ ] Publish findings and open-source the scorer module
