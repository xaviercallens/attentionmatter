# Intense Benchmark Results

**Date:** 2026-06-25  
**Hardware:** Apple M-series (ARM64), single core  
**Embedding Model:** all-MiniLM-L6-v2 (384-dim, 90MB)  
**Token Budget:** 6553 (80% of 8192)  
**Decay Factor:** 0.95  

## Key Findings

### 1. Classifier outperforms cosine on quality

| Strategy | Token Reduction | Key Fact Rate | Notes |
|----------|----------------|---------------|-------|
| cosine_decay (baseline) | 63.8% | 50% | Misses facts with low lexical overlap |
| cosine + rerank | 63.8% | 50% | Heuristic rerank doesn't help much |
| cosine + positional bias | 63.8% | 50% | Position not the bottleneck |
| **classifier_ranked** | 63.7% | **100%** | **Lexical features catch what cosine misses** |

The classifier's Jaccard overlap and keyword features capture relevance signals
that pure cosine similarity misses — especially when the query uses different
vocabulary than the stored fact (e.g., "antibiotic allergy" vs "amoxicillin").

### 2. Token reduction scales with conversation length

| Turns | Reduction | Key Fact Rate | Latency (incl. embedding) |
|-------|-----------|---------------|---------------------------|
| 100 | 27% | 100% | 256ms |
| 200 | 64% | 67% | 152ms |
| 500 | 86% | 67% | 199ms |

### 3. Scoring is sub-millisecond with pre-computed embeddings

| Candidates | Score + Select | Throughput |
|-----------|---------------|-----------|
| 200 | 1.1ms | 181k cand/s |
| 500 | 3.2ms | 157k cand/s |
| 1000 | 5.5ms | 182k cand/s |
| 5000 | 31.5ms | 159k cand/s |

The 50ms latency target is easily met for all practical conversation sizes.

### 4. Embedding is the bottleneck

| Mode | Throughput | Per-embedding |
|------|-----------|---------------|
| Batch (100) | 144 emb/s | 7ms |
| Single | 13 emb/s | 77ms |

**Production recommendation:** Pre-compute and cache embeddings on message arrival.
Scoring at request time uses only cached vectors → sub-5ms total latency.

### 5. False positive rate varies

At 200+ turns, hard negatives sometimes score higher than the key fact because
they share domain vocabulary with the query. The classifier mitigates this by
considering lexical overlap patterns and type indicators.

## Recommendations for Production

1. **Use classifier_ranked strategy** — it achieves 100% key fact rate where
   cosine alone achieves 50% on production-like data.
2. **Pre-compute embeddings** — cache on message arrival, not at query time.
3. **Batch embed** new messages — 7x faster than single-embed.
4. **Budget headroom** — at 100 turns, 27% reduction; at 500 turns, 86%.
   The algorithm's value increases with conversation length.
5. **Consider dynamic budget** — shorter conversations don't need pruning;
   scale budget with conversation token count.
