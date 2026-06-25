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

### 11. LTM entries must NOT be penalized by conversation length

**Critical fix in Phase 2:** Originally LTM entries had `age = len(stm) + offset`
which meant `0.95^800 ≈ 0` — effectively zeroing out all memory entries in long
conversations. The fix: LTM entries get `age=0` (no decay penalty) because they
already passed relevance filtering via vector search top-K. Only conversation
messages need recency decay.

This is a fundamental architectural insight: **persistent knowledge and ephemeral
conversation need different scoring models.** Decay applies to conversation turns
(recency matters). Memories are durable — their relevance is purely semantic.

### 12. Brute-force is fine at PoC scale

With < 200 candidates per scenario, brute-force cosine similarity over numpy arrays
takes < 1ms. FAISS adds complexity without benefit until the memory store grows to
10k+ entries. Premature optimization would have slowed development.

## Azure & Infrastructure

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
- Output is truncated at ~4KB. For larger outputs, write to file and cat separately.

### 17. Token reduction scales predictably with conversation length

| Turns | Budget Ratio | Reduction |
|-------|-------------|-----------|
| 5-15 | 0.80 | 0% (fits in budget) |
| 53-66 | 0.80 | 0% (fits in budget) |
| 101 | 0.80 | 17.4% |
| 500 | 0.50 | 46.3% |
| 750 | 0.50 | 50.0% |
| 1000 | 0.50 | 50.0% |

## Phase 2 Insights

### 18. GPT-2 tokenizer is the ideal offline fallback

GPT-2 tokenizer is ~1MB, doesn't need authentication, downloads from HuggingFace
without issues, and produces token counts within 5-10% of production tokenizers
for English text. Far more realistic than word×1.3 approximation.

### 19. Long conversations are mostly filler

A 1000-turn conversation with short messages (8-15 words each) only uses ~3200
GPT-2 tokens. Production conversations with longer messages (50-100 words per turn)
would use 10x more tokens. Our scenarios need longer per-message content to truly
stress the budget at default ratio.

### 20. The 50% reduction ceiling exists

With budget_ratio=0.5, reduction converges to ~50% regardless of conversation
length. This is because the adaptive strategy fills exactly to the budget boundary.
The actual token saved depends on `(full_context - budget) / full_context`. For
meaningful production savings, the full context must significantly exceed the budget.

## What We'd Do Differently (Updated)

- **Start with 500+ turn scenarios from day one.** Short scenarios don't exercise
  the algorithm's value.
- **Separate LTM scoring from the start.** The age-based decay should never have
  applied to persistent memories. Different content types need different scoring
  strategies — this should be a first-class design concern.
- **Use GPT-2 tokenizer as the default** for `--dummy-llm` mode, not word×1.3.
- **Make longer filler messages** (50-100 words each) to stress token budget at
  default ratio without needing to artificially tighten it.
- **Build the standalone scorer module first**, then wrap it in the experiment
  pipeline — not the other way around.
