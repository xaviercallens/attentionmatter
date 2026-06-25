# Goals

## Vision

Demonstrate that an attention-inspired context filtering mechanism can reduce LLM
prompt token usage by 30–50% while maintaining answer quality, paving the way for
production integration in conversational memory management systems.

---

## Phase 1: PoC Validation (COMPLETE)

**Goal:** Prove the core algorithm works on synthetic scenarios.

| Objective | Target | Actual | Status |
|-----------|--------|--------|--------|
| Implement adaptive scoring | cosine × decay | Done | COMPLETE |
| Compare 4 strategies | Fair comparison | 7 scenarios × 4 strategies | COMPLETE |
| Token reduction (tight budget) | 30–50% | 34–56% | EXCEEDED |
| Key fact preservation | >90% | 100% | EXCEEDED |
| Azure benchmark | Run on GPU VM | T4 in northeurope | COMPLETE |
| Cost under $5 | < $5 | ~$0.40 | EXCEEDED |

**Key Finding:** The algorithm works. Reduction scales with conversation length.
Short conversations (< budget) show minimal savings; long ones (100+ turns with
irrelevant content) show 17–56% savings while preserving all critical information.

---

## Phase 2: Scale & Validate (IN PROGRESS)

**Goal:** Prove the algorithm works at production scale with real LLM quality.

| Objective | Target | Status |
|-----------|--------|--------|
| 500+ turn scenarios | 3+ scenarios at scale | TODO |
| Real tokenizer (Mistral/GPT-2) | Accurate token counts | TODO |
| Real LLM answers (Mistral-7B) | Quality validated by model | TODO |
| 30–50% reduction at default budget | On long conversations | TODO |
| Decay factor optimization | Find optimal for 500+ turns | TODO |
| CI/CD pipeline | Auto-run on push | TODO |
| v0.2.0 release | With scale results | TODO |

**Hypothesis:** With 500+ turn conversations and the real tokenizer, the Adaptive
strategy will achieve 30–50% reduction at the default 80% budget, with ≥95% key
fact preservation.

---

## Phase 3: Production Integration

**Goal:** Ship the scorer as a production-ready component in A3TK.

| Objective | Target | Status |
|-----------|--------|--------|
| Standalone scorer module | pip-installable | NOT STARTED |
| A3TK orchestrator integration | Drop-in plugin | NOT STARTED |
| FAISS/ANN backend | 10k+ memory entries | NOT STARTED |
| OpenAI Ada embeddings support | API-based scoring | NOT STARTED |
| GPT-4.1 quality validation | Production-grade | NOT STARTED |
| Dynamic token budgeting | Query-adaptive | NOT STARTED |
| A/B test in staging | Measurable improvement | NOT STARTED |

---

## Phase 4: Advanced Scoring

**Goal:** Move beyond cosine similarity to learned relevance scoring.

| Objective | Target | Status |
|-----------|--------|--------|
| Train binary relevance classifier | Better than cosine | NOT STARTED |
| Learnable positional biases | Trainable ALiBi variant | NOT STARTED |
| Type-aware weighting | Facts > narrative | NOT STARTED |
| Multi-turn query context | Score against last N queries | NOT STARTED |
| 128k+ context evaluation | Does pruning still help? | NOT STARTED |

---

## Phase 5: Scale & Generalize

**Goal:** Extend to multi-modal and multi-agent scenarios.

| Objective | Target | Status |
|-----------|--------|--------|
| Multi-modal scoring | Images, tables, code | NOT STARTED |
| Cross-agent memory sharing | Selective context | NOT STARTED |
| Real-time adaptation | Live score updates | NOT STARTED |
| Feedback learning | Learn from user corrections | NOT STARTED |
| Open-source publication | Community release | NOT STARTED |

---

## Success Metrics

| Metric | Phase 1 | Phase 2 Target | Phase 3 Target |
|--------|---------|----------------|----------------|
| Token reduction | 17–56% | 30–50% (default budget) | 30–50% (production) |
| Key fact preservation | 100% | ≥95% | ≥98% |
| Scenarios tested | 7 | 15+ | Production traffic |
| Max conversation length | 101 turns | 500+ turns | Unbounded |
| Latency overhead | N/A | < 100ms | < 50ms |
| Cost per benchmark | $0.40 | < $5 | N/A |
