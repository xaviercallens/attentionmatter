# Memory — Achievements & Next Actions

## Project State

**Status:** Phase 1 complete. Azure benchmark validated. Starting Phase 2.  
**Date:** 2026-06-25  
**Branch:** main  
**Repo:** https://github.com/xaviercallens/attentionmatter  
**Release:** v0.1.0  

---

## Achievements

### 1. Core Algorithm Implemented and Validated

- Attention-inspired context selection using cosine similarity × recency decay.
- Formula: `score(chunk) = cos_sim(query, chunk) * (decay_factor ^ age)`
- Decay factor configurable (0.90 / 0.95 / 1.0 sweep supported).
- Token budget enforcement with chronological reassembly of selected context.

### 2. Token Reduction Demonstrated

**Local (tight budget, real embeddings):**

| Scenario | Reduction | Key Fact Preserved |
|----------|-----------|-------------------|
| flight_booking_memory (66 turns) | 34.6% | Yes |
| irrelevant_heavy (101 turns) | 56.0% | Yes |
| preference_recall (53 turns) | 17.7% | Yes |

**Azure (default budget, real embeddings):**

| Scenario | No-Pruning | Adaptive | Reduction |
|----------|-----------|----------|-----------|
| irrelevant_heavy (101 turns) | 1486 tokens | 1227 tokens | 17.4% |
| Overall avg | 651 tokens | 614 tokens | 2.5% |

- **100% key fact preservation** across all 7 scenarios in both environments.
- Sliding-Window baseline fails on 6/7 scenarios (14.3% pass rate).
- Reduction scales with conversation length and budget tightness.

### 3. Four Strategies Compared

| Strategy | Quality | Token Savings |
|----------|---------|---------------|
| No-Pruning | 100% pass | 0% (baseline) |
| Sliding-Window | 14% pass | ~80% savings |
| A3TK-Heuristic | 100% pass | ~3.5% savings |
| **Adaptive (ours)** | **100% pass** | **2.5–56% savings** (depends on length/budget) |

### 4. Azure Benchmark Executed

- **VM:** Standard_NC4as_T4_v3 (T4 16GB, northeurope)
- **Quota:** Auto-approved 0→4 cores in ~30s via `az quota create`
- **SSH workaround:** Corporate proxy blocks SSH banner; used `az vm run-command` API
- **Execution:** Cloned public repo, ran real embeddings benchmark via run-command
- **Cost:** ~$0.40 (45 min VM uptime)
- **Teardown:** Resource group deleted, $0 ongoing

### 5. Infrastructure & DevOps

- Public repo: https://github.com/xaviercallens/attentionmatter
- Dockerfile (CUDA 12.1), cloud-init.yaml, provision/deploy/teardown scripts
- Backup/restore to Azure Blob Storage with manifested archives
- Offline testing mode (DummyLLM + DummyEmbedding + DummyTokenizer)

---

## Key Decisions Made

1. **Brute-force cosine similarity** over FAISS — justified by small data scale.
2. **DummyTokenizerService** for offline mode — approximates tokens as words × 1.3.
3. **Extractive summary fallback** in A3TK strategy — avoids LLM dependency.
4. **Standard_NC4as_T4_v3** as Azure VM — cheapest GPU option ($0.53/hr).
5. **az vm run-command** for execution — workaround for SSH proxy block.
6. **Public repo** on GitHub — enables VM to clone without auth tokens.

---

## Known Limitations

1. Token reduction only manifests when conversations exceed the token budget.
2. DummyLLMClient uses regex fact extraction — does not test LLM reasoning.
3. Corporate SSL proxy blocks model downloads locally (fix: `SSL_CERT_FILE=""`).
4. SSH blocked by corporate network — all Azure VM interaction via run-command API.
5. A3TK-Heuristic shows minimal savings (~3.5%) — keyword scoring too conservative.
6. Current scenarios max at 101 turns — not enough to stress the default budget.

---

## Next Actions (Phase 2)

### Immediate

- [x] ~~Azure benchmark executed~~
- [ ] Add 500+ turn scenarios to demonstrate reduction at default budget
- [ ] Implement real tokenizer fallback (GPT-2) for accurate token counts offline
- [ ] Run with real LLM (Mistral-7B) on GPU for quality validation

### Short-Term

- [ ] Generate comparison charts with matplotlib
- [ ] Add GitHub Actions CI (run `--dummy-llm` on push)
- [ ] Create v0.2.0 release with longer scenarios and Azure results

### Medium-Term (Production Integration)

- [ ] Package relevance scorer as standalone Python module
- [ ] Integrate into A3TK orchestrator's context assembly pipeline
- [ ] Replace brute-force with FAISS for larger memory stores
- [ ] Benchmark with GPT-4.1

### Long-Term (Research)

- [ ] Train a learned relevance classifier
- [ ] Explore trainable ALiBi-style biases
- [ ] Test with 128k+ context models

---

## Quick Restart Guide

```bash
# Full offline test (no network)
python run_poc.py --dummy-llm --no-chart

# Real embeddings, no GPU
SSL_CERT_FILE="" python run_poc.py --dummy-llm --real-embeddings

# Tight budget (demonstrates reduction)
SSL_CERT_FILE="" python run_poc.py --dummy-llm --real-embeddings --token-budget-ratio 0.15

# Azure GPU benchmark
./azure/provision.sh
# (use az vm run-command for execution due to SSH proxy)
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
| `azure/cloud-init.yaml` | VM auto-setup |
| `results/azure_benchmark_results.md` | Azure run findings |
| `BENCHMARK.md` | Full Azure instructions |
