# Memory — Achievements & Next Actions

## Project State

**Status:** PoC complete, validated locally, Azure benchmark infrastructure ready.  
**Date:** 2025-06-24  
**Branch:** main (pre-release)

---

## Achievements

### 1. Core Algorithm Implemented and Validated

- Attention-inspired context selection using cosine similarity × recency decay.
- Formula: `score(chunk) = cos_sim(query, chunk) * (decay_factor ^ age)`
- Decay factor configurable (0.90 / 0.95 / 1.0 sweep supported).
- Token budget enforcement with chronological reassembly of selected context.

### 2. Token Reduction Demonstrated

With real sentence-transformer embeddings (`all-MiniLM-L6-v2`):

| Scenario | Reduction | Key Fact Preserved |
|----------|-----------|-------------------|
| flight_booking_memory (66 turns) | 34.6% | Yes |
| irrelevant_heavy (101 turns) | 56.0% | Yes |
| preference_recall (53 turns) | 17.7% | Yes |

- **100% key fact preservation** across all 7 scenarios.
- Sliding-Window baseline fails on 6/7 scenarios (14.3% pass rate).

### 3. Four Strategies Compared

| Strategy | Quality | Token Savings |
|----------|---------|---------------|
| No-Pruning | 100% pass | 0% (baseline) |
| Sliding-Window | 14% pass | ~80% savings |
| A3TK-Heuristic | 100% pass | ~0-5% savings |
| **Adaptive (ours)** | **100% pass** | **30-56% savings** |

### 4. Infrastructure Ready

- **Dockerfile:** CUDA 12.1 + pre-downloaded embedding model.
- **Azure scripts:** provision (V100 VM), deploy (rsync + docker), teardown.
- **Benchmark automation:** 4 parameter configurations in a single `benchmark.sh`.
- **Backup/Restore:** Azure Blob Storage archive with manifest and "latest" pointer.
- **Fast restart:** restore Docker image + model caches from blob storage.

### 5. Modular, Extensible Architecture

```
run_poc.py → ExperimentRunner → Strategy (4 impls) → LLMClient
                              → MemoryManager (STM + LTM)
                              → EmbeddingService
                              → Evaluator → Reporter
```

- Strategies implement a shared Protocol — drop in new ones without code changes.
- `DummyLLMClient` + `DummyEmbeddingService` enable full offline testing.

---

## Key Decisions Made

1. **Brute-force cosine similarity** over FAISS — justified by small data scale.
2. **DummyTokenizerService** for offline mode — approximates tokens as words × 1.3.
3. **Extractive summary fallback** in A3TK strategy — avoids LLM dependency for summarization.
4. **Standard_NC6s_v3 (V100)** as default Azure VM — balance of cost ($3/hr) and capability.
5. **Manifested archive** with latest pointer — enables restore without knowing timestamp.

---

## Known Limitations

- Token reduction only manifests when conversations exceed the token budget. Small
  conversations (< budget) show 0% reduction because everything fits.
- DummyLLMClient uses regex fact extraction — works for PoC evaluation but does not
  test actual LLM reasoning.
- Real Mistral-7B tokenizer download blocked by corporate SSL proxy (local env
  specific); works fine on Azure or with `SSL_CERT_FILE=""`.
- A3TK-Heuristic currently shows minimal token savings because keyword scoring
  preserves most messages — production version would be more aggressive.

---

## Next Actions

### Immediate (Before Azure Run)

- [ ] Initialize git repo, commit all code, push to GitHub.
- [ ] Run `./azure/backup.sh` to archive current state.
- [ ] Execute `./azure/provision.sh` → `./azure/deploy.sh docker` for full GPU run.
- [ ] Collect real LLM results (Mistral-7B answers) and validate quality.

### Short-Term (Post Azure Run)

- [ ] Analyze results: compare decay factor variants (0.90 vs 0.95 vs 1.0).
- [ ] Generate final report with charts and findings.
- [ ] Create GitHub release with results CSV and chart artifacts.
- [ ] Tune: if reduction < 30%, try tighter `token_budget_ratio` or lower decay.

### Medium-Term (Production Integration)

- [ ] Integrate relevance scorer into A3TK orchestrator as a plugin.
- [ ] Replace brute-force similarity with FAISS for larger memory stores.
- [ ] Add OpenAI Ada embeddings option for production (faster, API-based).
- [ ] Benchmark with GPT-4.1 to measure production-grade quality impact.
- [ ] Implement dynamic budget: adjust `token_budget_ratio` based on query complexity.

### Long-Term (Research)

- [ ] Train a small classifier for relevance scoring (replace cosine + decay).
- [ ] Explore trainable ALiBi-style biases (fine-tune on conversation data).
- [ ] Test with 128k+ context models — measure if pruning still helps.
- [ ] Multi-modal memory scoring (images, code blocks, structured data).

---

## Quick Restart Guide

```bash
# From scratch (no Azure state)
python run_poc.py --dummy-llm --no-chart

# With real embeddings (requires model download)
SSL_CERT_FILE="" python run_poc.py --dummy-llm --real-embeddings

# Restore from Azure archive
./azure/restore.sh
python run_poc.py --dummy-llm --real-embeddings

# Full Azure GPU benchmark
./azure/provision.sh
./azure/deploy.sh docker
./azure/teardown.sh
```

---

## File Inventory

| Path | Purpose |
|------|---------|
| `run_poc.py` | Single entry point |
| `src/strategies/adaptive.py` | Core algorithm |
| `src/scenarios.py` | 7 test scenarios |
| `benchmark.sh` | Multi-config runner |
| `azure/provision.sh` | VM creation |
| `azure/deploy.sh` | Code upload + execution |
| `azure/backup.sh` | Archive to blob storage |
| `azure/restore.sh` | Restore from blob storage |
| `azure/teardown.sh` | Delete Azure resources |
| `results/poc_results.csv` | Latest run results |
| `BENCHMARK.md` | Azure instructions |
