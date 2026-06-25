# Lessons Learned

## Technical

### 1. Semantic similarity is surprisingly effective for context selection

Simple cosine similarity between query and message embeddings correctly identifies
relevant context even when it's 60+ turns back. The `all-MiniLM-L6-v2` model (only
90MB) provides sufficient quality for this use case — no need for large embedding
models.

### 2. Recency decay must be tuned carefully

- `decay_factor=0.95` is a good default: old relevant items still score well
  (0.95^60 ≈ 0.046 multiplier), but combined with high cosine similarity (0.8+)
  they still rank above recent irrelevant items (low cosine × no decay).
- `decay_factor=0.90` is too aggressive for 100+ turn conversations — important
  items beyond turn 30 get nearly zeroed out.
- `decay_factor=1.0` (no decay) works but can over-include old irrelevant items
  that happen to have moderate similarity to the query.

### 3. Token budget only matters when context is large

Our test scenarios (5-101 turns) produce at most ~1500 tokens with the dummy
tokenizer. The 6553-token budget (80% of 8192) is never hit. Token reduction only
appears when we force a tight budget (ratio 0.08-0.30). In production with real
tokenizers and longer conversations (500+ turns), the budget will be genuinely
constraining and the Adaptive strategy's value becomes clear.

### 4. The DummyLLMClient pattern is invaluable

Building a regex-based stub that extracts facts from the prompt enabled full
pipeline testing without GPU access. Key insight: the stub needs to handle diverse
patterns (hex codes, phone numbers, times, dashed codes, proper nouns) to match
real evaluation scenarios.

### 5. Keyword-based importance (A3TK baseline) has a ceiling

The A3TK heuristic preserves high-keyword messages but doesn't understand semantic
relevance to the query. It keeps "booking code: XYZ789" because "code" is a keyword,
not because the user asked about it. This works for simple cases but fails when
important information doesn't contain obvious keywords.

## Process

### 6. Spec-driven development accelerates implementation

Writing requirements → design → tasks before coding meant zero backtracking. Every
module had a clear interface defined before implementation. The Protocol-based
strategy pattern made all four implementations trivially interchangeable.

### 7. Offline-first testing is essential

The SSL proxy issue blocked real model downloads in the dev environment. Having
`DummyEmbeddingService` (hash-based vectors) + `DummyTokenizerService` (word count
approximation) meant the full pipeline could be validated structurally without any
network access.

### 8. Azure VM provisioning has hidden delays

The NVIDIA driver installation + Docker setup takes 5-10 minutes via
`az vm run-command`. A reboot is required for drivers to initialize. Building into
the workflow upfront (provision → reboot → wait → deploy) avoids confusion.

### 9. Manifested archives enable reproducibility

Storing a `manifest.json` + `latest.txt` pointer in Azure Blob Storage means any
team member can restore the exact state of a previous run without knowing timestamps
or searching through blob containers.

## Architecture

### 10. Chronological reassembly is critical for LLM quality

After scoring and selecting context items, they must be reassembled in chronological
order for the prompt to make sense to the LLM. Presenting messages out of order
confuses the model. The Adaptive strategy sorts selected history by turn index before
assembly.

### 11. LTM entries need artificial aging

Memory entries from prior sessions are semantically relevant but shouldn't dominate
over recent in-session context. Assigning them `age = len(stm) + 1000` ensures they
get a heavy decay penalty, so they only appear in the prompt when their cosine
similarity to the query is very high (i.e., directly relevant).

### 12. Brute-force is fine at PoC scale

With < 200 candidates per scenario, brute-force cosine similarity over numpy arrays
takes < 1ms. FAISS adds complexity without benefit until the memory store grows to
10k+ entries. Premature optimization would have slowed development.

## Azure & Infrastructure (NEW — Phase 1 Benchmark)

### 13. Corporate SSH proxy blocks banner exchange

TCP port 22 connects (nc succeeds) but sshd banner never arrives. Root cause:
corporate proxy inspects/delays SSH traffic. **Workaround:** `az vm run-command`
API executes scripts via Azure's control plane, bypassing SSH entirely. All benchmark
execution was done this way.

### 14. GPU quota auto-approval is fast

Submitting `az quota create` for NCASv3_T4 (4 cores) auto-approved in ~30 seconds.
No manual ticket needed for test subscriptions. Always try programmatic quota
requests before filing portal requests.

### 15. cloud-init is the right approach for VM setup

Using `--custom-data @cloud-init.yaml` installs dependencies in the background
without blocking sshd startup (unlike running apt-get via run-command which holds
locks). The pattern: cloud-init for heavy setup, run-command for lightweight
execution after setup completes.

### 16. Run-command v2 API has UX quirks

- `az vm run-command create` returns exit code 1 even on success (misleading).
- Only one run-command per name at a time; delete old before creating new.
- `az vm run-command invoke` (v1) ignores inline `--scripts` content on newer CLI
  versions (runs "sample script" instead). Always use v2 `create` API.
- Output is truncated at ~4KB. For larger outputs, write to a file and cat it
  in a separate command.

### 17. Token reduction scales predictably with conversation length

| Turns | Budget Ratio | Reduction |
|-------|-------------|-----------|
| 5-15 | 0.80 | 0% (fits in budget) |
| 53-66 | 0.80 | 0% (fits in budget) |
| 101 | 0.80 | 17.4% |
| 66 | 0.08 | 34.6% |
| 101 | 0.08 | 56.0% |

**Conclusion:** The algorithm works. To demonstrate 30-50% reduction at default
budget, we need 300-500+ turn conversations — which is realistic for production.

## What We'd Do Differently

- **Start with longer scenarios (500+ turns):** Would have demonstrated token
  reduction at the default budget without needing to artificially tighten it.
- **Use GPT-2 tokenizer as default fallback:** Small, downloadable without auth,
  gives realistic token counts (unlike the word×1.3 approximation).
- **Include a "regression" scenario:** Where the Adaptive strategy might fail — e.g.,
  the relevant info has low lexical similarity to the query but is contextually
  important. This would highlight the ceiling of cosine-based scoring.
- **Use `az vm run-command` from the start:** Don't fight SSH in corporate envs.
- **Pre-bake NVIDIA drivers in a custom image:** Avoids the 10-min DKMS compile
  every time. Build once, snapshot, reuse.
