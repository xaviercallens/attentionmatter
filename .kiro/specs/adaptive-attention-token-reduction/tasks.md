# Implementation Tasks

## Task 1: Project Scaffolding and Configuration

### Description
Set up the project structure, dependencies, and the central configuration module.

### Requirements Addressed
- Requirement 10 (Configuration and Reproducibility)

### Steps
1. Create the project directory layout:
   ```
   attentionmatter/
   ├── src/
   │   ├── __init__.py
   │   ├── config.py
   │   ├── embedding.py
   │   ├── tokenizer_service.py
   │   ├── memory.py
   │   ├── llm.py
   │   ├── evaluator.py
   │   ├── runner.py
   │   ├── reporter.py
   │   ├── scenarios.py
   │   └── strategies/
   │       ├── __init__.py
   │       ├── base.py
   │       ├── no_pruning.py
   │       ├── sliding_window.py
   │       ├── a3tk_heuristic.py
   │       └── adaptive.py
   ├── results/
   ├── requirements.txt
   ├── run_poc.py            # single entry point
   └── README.md (optional)
   ```
2. Create `requirements.txt` with pinned dependencies:
   - `transformers` (for LLM + tokenizer)
   - `torch` (backend)
   - `sentence-transformers` (embedding model)
   - `numpy`
   - `bitsandbytes` (4-bit quantization)
   - `accelerate`
   - `faiss-cpu` (optional, for vector search)
   - `matplotlib` (optional, for charting)
3. Implement `src/config.py`:
   - Define `Config` dataclass with all parameters from the design document.
   - Derive `token_budget` property (`int(max_context_tokens * token_budget_ratio)`).
   - Accept optional overrides from environment variables or a JSON config file.
   - Set the random seed (numpy + torch) upon instantiation.

### Acceptance Criteria
- Running `python run_poc.py --help` prints available options without import errors.
- All package imports succeed after `pip install -r requirements.txt`.
- `Config` defaults match the design document values.

---

## Task 2: Embedding Service

### Description
Implement the embedding service that computes and caches text embeddings using a
local sentence-transformer model.

### Requirements Addressed
- Requirement 2 (Embedding and Tokenization) — AC 1, 2, 4

### Steps
1. Implement `src/embedding.py` with class `EmbeddingService`:
   - Load the model specified in `Config.embedding_model` on first use (lazy init).
   - `embed(text: str) -> np.ndarray` — returns a unit-normalized vector.
   - `embed_batch(texts: list[str]) -> np.ndarray` — batch computation.
   - `cosine_similarity(a, b) -> float` — dot product of unit-normalized vectors.
   - Add an LRU dict cache (`dict[str, np.ndarray]`) keyed by text; return cached
     value if available.
2. Verify dimension matches config expectation (384 for MiniLM-L6).

### Acceptance Criteria
- `embed("hello")` returns a numpy array of shape (384,).
- Two identical texts return the same cached array instance.
- `cosine_similarity(embed(x), embed(x))` ≈ 1.0.

---

## Task 3: Tokenizer Service

### Description
Implement the token-counting service using the tokenizer that matches the target LLM.

### Requirements Addressed
- Requirement 2 (Embedding and Tokenization) — AC 3

### Steps
1. Implement `src/tokenizer_service.py` with class `TokenizerService`:
   - Load `AutoTokenizer.from_pretrained(Config.llm_model)` on init.
   - `count(text: str) -> int` — encode text and return number of tokens.
   - Handle edge cases: empty string returns 0.
2. Expose a module-level convenience function `count_tokens(text)` for quick use.

### Acceptance Criteria
- `count("")` returns 0.
- `count("Hello world")` returns a positive integer consistent with the model's
  tokenizer.

---

## Task 4: Memory Manager (STM + LTM)

### Description
Implement in-memory STM and LTM stores simulating Redis and Couchbase.

### Requirements Addressed
- Requirement 1 (Conversation and Memory Simulation)

### Steps
1. Define data models in `src/memory.py`:
   - `Message` dataclass: `text`, `role`, `turn`, `important`.
   - `MemoryRecord` dataclass: `embedding`, `text`, `source_session`, `importance`.
2. Implement `MemoryManager`:
   - `__init__(cfg, embedding_service)` — creates STM deque with configurable max
     capacity and empty LTM list.
   - `add_message(msg: Message)` — appends to STM; oldest evicted if at capacity.
   - `get_stm() -> list[Message]` — returns list in chronological order.
   - `insert_memory(text, **meta)` — computes embedding and stores a `MemoryRecord`.
   - `search_ltm(query_vec, top_k) -> list[MemoryRecord]` — brute-force cosine
     similarity, returns top-K records sorted descending by similarity.
   - `reset()` — clears STM and LTM for the next scenario.

### Acceptance Criteria
- Adding 200 messages to a capacity-100 STM leaves exactly 100 messages (the last
  100).
- `search_ltm` on a store with 5 entries returns them in descending similarity order.
- `reset()` empties both stores.

---

## Task 5: Strategy Interface and No-Pruning Baseline

### Description
Define the shared strategy protocol and implement the simplest baseline — No-Pruning.

### Requirements Addressed
- Requirement 5 — AC 1
- Requirement 4 (shared interface)

### Steps
1. Define `src/strategies/base.py`:
   - `SelectionResult` dataclass: `prompt`, `selected`, `omitted`, `token_count`.
   - `ContextStrategy` Protocol with attribute `name: str` and method
     `build_prompt(query: str, memory: MemoryManager) -> SelectionResult`.
2. Implement `src/strategies/no_pruning.py` (`NoPruningStrategy`):
   - Concatenates system prompt + all STM messages + all LTM records (via
     `search_ltm` with `top_k` = all entries or a large K).
   - Truncates from the beginning if total exceeds `max_context_tokens`.
   - Records everything as selected; nothing omitted (unless truncated).

### Acceptance Criteria
- With conversation within model max, everything is included (omitted list empty).
- With conversation exceeding model max, prompt is truncated and omitted list is
  non-empty.

---

## Task 6: Sliding-Window Strategy

### Description
Implement the static sliding-window baseline that only keeps the most recent N
messages.

### Requirements Addressed
- Requirement 5 — AC 2

### Steps
1. Implement `src/strategies/sliding_window.py` (`SlidingWindowStrategy`):
   - Takes the last `Config.sliding_window_messages` messages from STM.
   - Ignores LTM entirely.
   - Assembles prompt from system prompt + selected messages.
   - Everything older than the window and all LTM are marked omitted.

### Acceptance Criteria
- With 100 messages and window=4, only the last 4 appear in `selected`.
- LTM entries never appear in the prompt.

---

## Task 7: A3TK Heuristic Strategy

### Description
Implement the A3TK-style heuristic with keyword importance scoring, pruning,
summarization, and LTM retrieval.

### Requirements Addressed
- Requirement 5 — AC 3, 4, 5

### Steps
1. Implement `src/strategies/a3tk_heuristic.py` (`A3TKHeuristicStrategy`):
   - **Keyword importance score:** Define a list of high-signal keywords (numbers,
     names, domain terms like "booking", "code", "reference"). Score each message as
     the count of keyword matches, plus a bonus for the `important` flag.
   - **Pruning:** Sort messages by age ascending (oldest first). Drop older
     low-importance messages (score below a threshold) until remaining messages fit
     within `token_budget`.
   - **Summarization fallback:** If the dropped portion exceeds
     `Config.summarization_threshold_tokens`, generate a condensed summary of the
     dropped messages. Use `LLMClient.generate()` with a summarization prompt; or,
     for speed, use an extractive fallback (first + last sentence of each dropped
     message). Insert the summary at the top of the assembled prompt.
   - **LTM retrieval:** Retrieve `Config.ltm_top_k` memory entries via
     `MemoryManager.search_ltm` and append them.
   - Record selected/omitted and token count.
2. Accept an optional `LLMClient` reference for summarization (or pass `None` to use
   extractive fallback).

### Acceptance Criteria
- A message containing "booking code" keeps a high importance score and is not pruned.
- An older message with no keywords is pruned first.
- When summarization fires, the summary text appears in the prompt.
- LTM entries are included.

---

## Task 8: Adaptive (Attention Filter) Strategy

### Description
Implement the core attention-inspired strategy using embedding relevance scoring and
recency decay.

### Requirements Addressed
- Requirement 3 (Relevance Scoring Function)
- Requirement 4 (Adaptive Context Selection)

### Steps
1. Implement `src/strategies/adaptive.py` (`AdaptiveStrategy`):
   - Compute query embedding via `EmbeddingService`.
   - Build candidate list:
     - Each STM message becomes a candidate with `age = len(stm) - 1 - i` (0 = most
       recent).
     - Top-K LTM memories (from `search_ltm`) become candidates with
       `age = len(stm) + LARGE_AGE_OFFSET` (e.g., 1000).
   - Score each candidate:
     ```
     cos = max(0.0, cosine_similarity(q_vec, cand.embedding))
     score = cos * (decay_factor ** age)
     ```
   - Sort candidates by descending score.
   - Accumulate candidates within `token_budget`, skipping any that would exceed it.
   - Assemble prompt:
     - Order selected history messages by turn (chronological).
     - Group memory facts before or after history (design choice).
   - Record selected/omitted/token_count.

### Acceptance Criteria
- A far-back message with high cosine similarity to the query outscores recent but
  irrelevant messages.
- With `decay_factor=1.0` no recency bias is applied.
- With `decay_factor=0.95` a message at age 20 has its score multiplied by 0.95^20 ≈
  0.358.
- Total prompt tokens do not exceed `token_budget`.
- Selected messages appear chronologically in the final prompt.

---

## Task 9: LLM Client

### Description
Implement the local LLM wrapper for generating answers from assembled prompts.

### Requirements Addressed
- Requirement 6 (LLM Invocation)

### Steps
1. Implement `src/llm.py` with class `LLMClient`:
   - `__init__(cfg: Config)`:
     - Load model with `AutoModelForCausalLM.from_pretrained(cfg.llm_model)`.
     - If `cfg.use_4bit`, pass quantization config (`BitsAndBytesConfig`).
     - Load tokenizer from same model id.
   - `generate(prompt: str, max_new_tokens: int = 256) -> str`:
     - Tokenize prompt, run model generation, decode output (excluding prompt tokens).
   - Fail with a clear error message if model download fails or GPU memory is
     insufficient.
2. Provide a `DummyLLMClient` (returns the prompt or a canned response) for fast
   testing without GPU.

### Acceptance Criteria
- `LLMClient` loads without error on suitable hardware; raises `RuntimeError` with
  model name when unavailable.
- `DummyLLMClient.generate(prompt)` returns a deterministic string.
- Generated text length is bounded by `max_new_tokens`.

---

## Task 10: Scenarios

### Description
Define 5–10 test scenarios covering far-back recall, cross-session memory, and
irrelevant-heavy conversations.

### Requirements Addressed
- Requirement 7 — AC 1, 3, 4

### Steps
1. Implement `src/scenarios.py`:
   - Define `Scenario` dataclass: `id`, `conversation: list[Message]`,
     `seed_memories: list[tuple[str, dict]]`, `query: str`, `key_fact: str`.
   - Implement generator functions or constants for each scenario. Each must supply
     at least 15–20 turns of conversation with a mix of irrelevant chit-chat and
     critical facts. At least one scenario should have 100+ turns.
   - Scenarios (minimum set):
     1. **flight_booking_memory** — booking code "XYZ789" given at turn 4, stored in
        LTM; user asks for it after ~60 irrelevant turns.
     2. **support_original_problem** — issue description at turn 2; user asks "remind
        me the original problem" at turn 15.
     3. **preference_recall** — user preference ("I'm vegetarian") stated 50 turns
        ago; user asks "what dietary preference do you have on file?"
     4. **cross_session_name** — user's name provided in a prior session (LTM only);
        new session asks "what's my name?"
     5. **irrelevant_heavy** — 100 turns of weather/sports chit-chat with one
        important fact (account number) at turn 40; user asks for it at turn 101.
     6. **multi_fact** — two distinct facts at turns 5 and 30; final query needs both.
     7. **no_memory_needed** — answer is in the last 3 messages; tests that all
        strategies pass (sanity check).
   - Provide a `load_scenarios() -> list[Scenario]` function.

### Acceptance Criteria
- `load_scenarios()` returns at least 5 scenarios.
- Each scenario's `key_fact` is a string that appears somewhere in the conversation or
  seed memories.
- At least one scenario has 100+ turns.

---

## Task 11: Evaluator

### Description
Implement the quality evaluator that checks whether the expected key fact is present
in the LLM output.

### Requirements Addressed
- Requirement 8 (Quality Evaluation)

### Steps
1. Implement `src/evaluator.py`:
   - `QualityResult` dataclass: `passed: bool`, `similarity: float | None`.
   - `Evaluator` class with method `score(answer: str, key_fact: str) -> QualityResult`:
     - Primary check: case-insensitive substring match of `key_fact` in `answer`.
     - Secondary (optional): compute cosine similarity between answer embedding and
       key_fact embedding.
   - Accept `EmbeddingService` for optional similarity scoring.

### Acceptance Criteria
- `score("Your code is XYZ789", "XYZ789")` → `passed=True`.
- `score("I don't know", "XYZ789")` → `passed=False`.
- Similarity value is in [0, 1] when computed.

---

## Task 12: Experiment Runner

### Description
Implement the runner that executes every scenario through all four strategies and
collects results.

### Requirements Addressed
- Requirement 7 — AC 2, 5

### Steps
1. Implement `src/runner.py`:
   - `RunRecord` dataclass: `scenario_id`, `strategy`, `token_count`, `quality:
     QualityResult`, `answer`.
   - `ExperimentRunner`:
     - `__init__(cfg, strategies, llm, evaluator, scenarios)`.
     - `run() -> list[RunRecord]`:
       - For each scenario, reset and populate MemoryManager.
       - For each strategy, call `build_prompt`, call `llm.generate`, call
         `evaluator.score`.
       - Append `RunRecord`.
     - Print progress (scenario/strategy) to stdout.

### Acceptance Criteria
- With 7 scenarios × 4 strategies, `run()` returns 28 `RunRecord` objects.
- Each record has a non-negative `token_count` and a valid `QualityResult`.

---

## Task 13: Reporter

### Description
Implement the results reporter that produces a table, computes summary statistics,
and persists outputs.

### Requirements Addressed
- Requirement 9 (Results Reporting)

### Steps
1. Implement `src/reporter.py`:
   - `Reporter`:
     - `to_table(records) -> str` — Markdown table (scenario × strategy) with columns
       token_count and quality.
     - `avg_reduction(records) -> float` — average token reduction of Adaptive vs
       No-Pruning (percentage).
     - `persist(records, path)` — write CSV with columns: scenario_id, strategy,
       token_count, passed, similarity, answer (truncated).
     - `chart(records, path)` (optional) — bar or scatter plot of token count vs
       pass rate per strategy.
   - Print summary to stdout: average tokens per strategy, pass rate per strategy,
     and the reduction percentage.

### Acceptance Criteria
- `to_table` returns a valid Markdown table parseable by any viewer.
- `persist` writes a CSV file at the configured path.
- `avg_reduction` returns a value between 0 and 100.

---

## Task 14: Main Entry Point and Integration

### Description
Wire everything together in a single `run_poc.py` script that initializes all
components, runs the experiment, and produces the report.

### Requirements Addressed
- Requirement 10 — AC 4

### Steps
1. Implement `run_poc.py`:
   - Parse CLI arguments (optional JSON config path, optional `--dummy-llm` flag for
     fast testing).
   - Instantiate `Config` (with overrides if provided).
   - Instantiate shared services: `EmbeddingService`, `TokenizerService`,
     `MemoryManager`, `LLMClient` (or `DummyLLMClient`).
   - Instantiate all four strategies.
   - Load scenarios.
   - Instantiate `Evaluator` and `ExperimentRunner`.
   - Call `runner.run()`.
   - Pass records to `Reporter` — print table, persist CSV, optionally chart.
   - Exit 0 on success.
2. Add a `--decay-factor` CLI flag to support the 1.0 vs 0.95 comparison without code
   changes (Requirement 10.3).
3. Add a `--dry-run` flag that only prints the scenario list and config without
   running the LLM.

### Acceptance Criteria
- `python run_poc.py --dummy-llm` runs end-to-end without a GPU and produces a
  results CSV.
- `python run_poc.py --decay-factor 1.0 --dummy-llm` overrides decay and completes.
- `python run_poc.py --dry-run` prints config and exits without generating answers.

---

## Task 15: Validation and Tuning Run

### Description
Run the full pipeline on actual hardware (or with the dummy LLM for CI) and verify
success criteria.

### Requirements Addressed
- Success Criteria (all)

### Steps
1. Run `python run_poc.py` with the real LLM on GPU (or `--dummy-llm` on CPU for
   structural validation).
2. Review the results table:
   - Confirm Adaptive achieves 30–50% fewer tokens than No-Pruning on average.
   - Confirm Adaptive Pass rate matches or exceeds No-Pruning on far-back / LTM
     scenarios.
   - Confirm Sliding-Window fails (Pass=0) on at least one far-back scenario.
3. If Adaptive token reduction is below 30%:
   - Try a stricter `token_budget_ratio` (e.g., 0.6).
   - Try a lower `decay_factor` (e.g., 0.90) to drop more old irrelevant messages.
4. If Adaptive quality drops (Pass rate much lower than No-Pruning):
   - Increase `ltm_top_k`.
   - Increase `decay_factor` toward 1.0 to retain more context.
5. Document final tuned parameters in a short `results/TUNING_NOTES.md`.

### Acceptance Criteria
- Results CSV exists at `results/poc_results.csv` with 28+ rows.
- Average token reduction (Adaptive vs No-Pruning) is between 30% and 50%.
- Adaptive Pass rate is within 10 percentage points of No-Pruning.
- Sliding-Window Pass rate is measurably lower than Adaptive on far-back scenarios.

---

## Dependency Order

```
Task 1 (scaffold/config)
  ├── Task 2 (embedding)
  ├── Task 3 (tokenizer)
  └── Task 4 (memory)
         ├── Task 5 (strategy interface + no-pruning)
         ├── Task 6 (sliding-window)
         ├── Task 7 (A3TK heuristic) ← depends on Task 9 (LLM) for summarization
         └── Task 8 (adaptive)
Task 9 (LLM client)
Task 10 (scenarios)
Task 11 (evaluator) ← depends on Task 2 (embedding) for optional similarity
Task 12 (runner) ← depends on Tasks 5–11
Task 13 (reporter)
Task 14 (entry point) ← depends on all above
Task 15 (validation) ← depends on Task 14
```

Tasks 2, 3, 4, 9, 10, 11, 13 can proceed in parallel after Task 1.
Tasks 5–8 depend on Tasks 2–4 being complete.
Task 12 is the integration point.
Task 14 wires everything; Task 15 validates.
