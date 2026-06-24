# Roadmap

## Phase 1: PoC Validation (COMPLETE)

- [x] Design attention-inspired scoring function (cosine sim × decay)
- [x] Implement 4 context strategies (No-Pruning, Sliding-Window, A3TK, Adaptive)
- [x] Build 7 test scenarios (far-back recall, cross-session, irrelevant-heavy)
- [x] Validate locally with real embeddings: 30-56% token reduction, 100% quality
- [x] Create Azure GPU benchmark infrastructure
- [x] Implement backup/restore to Azure Blob Storage

## Phase 2: GPU Benchmark (IN PROGRESS)

- [ ] Run full benchmark on Azure V100 with Mistral-7B
- [ ] Compare decay factor variants: 0.90, 0.95, 1.0
- [ ] Compare budget ratios: 0.6, 0.8
- [ ] Collect LLM-generated answers and validate quality
- [ ] Generate comparison charts and final report
- [ ] Tag v0.1.0 release with results

## Phase 3: Production Integration

- [ ] Package relevance scorer as a standalone module/plugin
- [ ] Integrate into A3TK orchestrator's context assembly pipeline
- [ ] Replace brute-force similarity with FAISS or ANN index
- [ ] Add OpenAI Ada / Cohere embedding support for API-based scoring
- [ ] Benchmark with GPT-4.1 on production-like conversations
- [ ] Implement dynamic token budgeting based on query complexity
- [ ] A/B test in staging environment

## Phase 4: Advanced Scoring

- [ ] Train a small binary classifier for relevance (replace cosine heuristic)
- [ ] Explore trainable positional biases (learnable ALiBi variant)
- [ ] Add type-aware weighting (facts vs narrative vs code)
- [ ] Multi-turn query context (score against last N queries, not just latest)
- [ ] Evaluate on 128k+ context models — does pruning still help?

## Phase 5: Scale & Generalize

- [ ] Multi-modal memory scoring (images, tables, code blocks)
- [ ] Cross-agent memory sharing with selective context
- [ ] Real-time adaptation: update scores as conversation evolves
- [ ] Feedback loop: learn from user corrections (was pruned info needed?)
- [ ] Publish findings and open-source the scorer module
