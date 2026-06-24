# Design Document

## Overview

This PoC implements an attention-inspired context filtering pipeline for
conversational memory management. The central idea is to apply a pseudo-attention
relevance filter at the prompt-engineering layer: each candidate piece of context
(a conversation message/chunk or a long-term memory entry) is scored by semantic
similarity to the current user query and modulated by a recency decay factor. Only
the highest-scoring candidates that fit within a token budget are assembled into the
final prompt.

The pipeline is a single-machine Python application. It simulates short-term memory
(STM, normally Redis) and long-term memory (LTM, normally Couchbase vector search)
with in-memory structures, and calls a local open-source LLM (Mistral 7B / Qwen 7B).

The design supports four interchangeable context-management strategies sharing a
common interface:

1. **No-Pruning** — full context up to model maximum (quality ceiling).
2. **Sliding-Window** — last N messages only, ignores LTM (token floor).
3. **A3TK heuristic** — keyword importance pruning + summarization + LTM retrieval.
4. **Adaptive (Attention filter)** — embedding relevance score + recency decay.

An experiment runner executes every scenario through every strategy, an evaluator
scores answer quality, and a reporter produces a comparison table and summary.

### Design Goals

- **Comparability:** all strategies share the same embedding model, tokenizer, LLM,
  token budget, and LTM retrieval count (except No-Pruning, which fills to max).
- **Modularity:** strategies implement a single interface so new ones can be added.
- **Reproducibility:** all key parameters and the random seed are configurable from
  one place.
- **Simplicity:** brute-force cosine similarity is acceptable given small data; FAISS
  is optional.

### Non-Goals

- Production infrastructure integration, learned scoring models, LLM fine-tuning, or
  large hosted models. These are explicitly future work.

## Architecture

```
                          ┌─────────────────────────────┐
                          │      ExperimentRunner       │
                          │  (scenarios × strategies)   │
                          └──────────────┬──────────────┘
                                         │
          ┌──────────────┬───────────────┼───────────────┬───────────────┐
          ▼              ▼               ▼               ▼               ▼
   ┌────────────┐ ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐
   │ Scenario   │ │  Memory    │  │ Strategy   │  │   LLM      │  │Evaluator │
   │ Loader     │ │  Manager   │  │ (4 impls)  │  │  Client    │  │          │
   └────────────┘ │ STM + LTM  │  └─────┬──────┘  └────────────┘  └──────────┘
                  └─────┬──────┘        │
                        │               ▼
                  ┌─────▼──────┐  ┌──────────────┐
                  │ Embedding  │  │  Tokenizer   │
                  │  Service   │  │   Service    │
                  └────────────┘  └──────────────┘
                                         │
                                         ▼
                                  ┌────────────┐
                                  │  Reporter  │
                                  │ table/CSV  │
                                  └────────────┘
```

### Data Flow (per scenario, per strategy)

1. `ScenarioLoader` provides the conversation, seed LTM facts, final query, and
   expected key fact.
2. `MemoryManager` populates STM with the conversation and inserts seed facts into
   LTM (with embeddings).
3. The selected `ContextStrategy` builds candidate context, scores/selects/assembles
   it within the token budget, and returns a prompt plus a selection record.
4. `TokenizerService` reports the prompt token count.
5. `LLMClient` generates an answer from the prompt.
6. `Evaluator` checks whether the expected key fact is present (Pass/Fail).
7. `Reporter` records `(scenario, strategy, token_count, quality)`.

## Components and Interfaces

### Configuration (`config.py`)

A single dataclass/`Config` object holds all tunable parameters (Requirement 10).

```python
@dataclass
class Config:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_4bit: bool = True
    max_context_tokens: int = 8192
    token_budget_ratio: float = 0.8       # budget = ratio * max_context_tokens
    decay_factor: float = 0.95            # 1.0 disables recency bias
    sliding_window_messages: int = 4
    ltm_top_k: int = 5                    # M memories retrieved
    summarization_threshold_tokens: int = 512
    random_seed: int = 42
    results_path: str = "results/poc_results.csv"
```

The derived token budget is `int(max_context_tokens * token_budget_ratio)`.

### EmbeddingService (`embedding.py`)

Wraps the local embedding model and provides cached embeddings (Requirement 2).

```python
class EmbeddingService:
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float: ...
```

- Uses one model instance for both queries and candidates.
- An LRU/dict cache keyed by text avoids recomputation.

### TokenizerService (`tokenizer.py`)

Counts tokens using the tokenizer that matches the target LLM (Requirement 2/6).

```python
class TokenizerService:
    def count(self, text: str) -> int: ...
```

### MemoryManager (`memory.py`)

Simulates STM (Redis) and LTM (Couchbase vector search) in memory (Requirement 1).

```python
@dataclass
class Message:
    text: str
    role: str            # "user" | "assistant"
    turn: int
    important: bool = False

@dataclass
class MemoryRecord:
    embedding: np.ndarray
    text: str
    source_session: str
    importance: float

class MemoryManager:
    def add_message(self, msg: Message) -> None: ...      # append + evict oldest
    def get_stm(self) -> list[Message]: ...
    def insert_memory(self, text: str, **meta) -> None: ...
    def search_ltm(self, query_vec: np.ndarray, top_k: int) -> list[MemoryRecord]: ...
```

- STM is a `deque(maxlen=capacity)`; appending past capacity evicts the oldest.
- `search_ltm` uses brute-force cosine similarity (FAISS optional behind the same
  interface).

### ContextStrategy interface (`strategies/base.py`)

All four strategies share one interface so the runner is strategy-agnostic
(Requirement 4/5).

```python
@dataclass
class SelectionResult:
    prompt: str
    selected: list[str]
    omitted: list[str]
    token_count: int

class ContextStrategy(Protocol):
    name: str
    def build_prompt(self, query: str, memory: MemoryManager) -> SelectionResult: ...
```

#### NoPruningStrategy

Includes the whole conversation plus all relevant memories, truncating only at the
model maximum (Requirement 5.1).

#### SlidingWindowStrategy

Keeps the most recent N messages (or last X tokens), ignores LTM (Requirement 5.2).

#### A3TKHeuristicStrategy

Emulates the production heuristic (Requirement 5.3/5.5):

1. Keyword importance score per message (numbers, names, domain keywords, the
   `important` flag).
2. Drop oldest low-importance messages until within budget.
3. If too much is dropped (exceeds `summarization_threshold_tokens`), summarize the
   dropped chunk via the LLM (or a deterministic extractive fallback) and insert the
   summary.
4. Retrieve top-K LTM entries and append.

#### AdaptiveStrategy

Implements the attention-inspired filter (Requirement 3/4). Pseudocode reflecting the
specification:

```python
def build_prompt(self, query, memory):
    q_vec = embed(query)
    candidates = []

    stm = memory.get_stm()
    for i, msg in enumerate(stm):
        age = len(stm) - 1 - i            # 0 = most recent
        candidates.append(Candidate(msg.text, "history",
                                     embed(msg.text), age))

    for mem in memory.search_ltm(q_vec, top_k=cfg.ltm_top_k):
        candidates.append(Candidate(mem.text, "memory",
                                    mem.embedding, age=len(stm) + LARGE))

    for c in candidates:
        cos = max(0.0, cosine_similarity(q_vec, c.embedding))
        c.score = cos * (cfg.decay_factor ** c.age)

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected, omitted, total = [], [], 0
    for c in candidates:
        t = tokenizer.count(c.text)
        if total + t > token_budget:
            omitted.append(c); continue
        selected.append(c); total += t

    prompt = assemble_chronologically(selected, query)
    return SelectionResult(prompt, ..., token_count=total)
```

- `assemble_chronologically` orders selected history messages by turn so the prompt
  reads coherently, while memory facts are grouped (Requirement 4.4).
- Selection stops before exceeding the budget; remaining candidates are recorded as
  Omitted (Requirement 4.3/4.5).

### LLMClient (`llm.py`)

Wraps local model loading and generation (Requirement 6).

```python
class LLMClient:
    def __init__(self, cfg: Config): ...        # optional 4-bit quantization
    def generate(self, prompt: str) -> str: ...
```

- Uses Hugging Face Transformers; loads with `load_in_4bit=True` when `use_4bit`.
- Raises a clear error if model/dependencies are unavailable (Requirement 6.4).

### Evaluator (`evaluator.py`)

Scores answer quality (Requirement 8).

```python
class Evaluator:
    def score(self, answer: str, expected_key_fact: str) -> QualityResult: ...
    # QualityResult: passed: bool, similarity: float | None
```

- Primary check: case-insensitive substring presence of the key fact → Pass/Fail.
- Optional: cosine similarity between answer and expected answer via EmbeddingService.

### ExperimentRunner (`runner.py`)

Drives the full matrix (Requirement 7).

```python
class ExperimentRunner:
    def run(self) -> list[RunRecord]:
        for scenario in scenarios:
            memory = build_memory(scenario)
            for strategy in [NoPruning, SlidingWindow, A3TK, Adaptive]:
                result = strategy.build_prompt(scenario.query, memory)
                answer = llm.generate(result.prompt)
                quality = evaluator.score(answer, scenario.key_fact)
                records.append(RunRecord(scenario.id, strategy.name,
                                         result.token_count, quality, answer))
        return records
```

### Reporter (`reporter.py`)

Produces the comparison table and summary (Requirement 9).

```python
class Reporter:
    def to_table(self, records) -> str: ...          # Markdown table
    def avg_reduction_vs_full(self, records) -> float: ...
    def persist(self, records, path) -> None: ...     # CSV / Markdown
    def chart(self, records, path) -> None: ...        # optional token-vs-quality
```

### Scenarios (`scenarios.py`)

Defines 5–10 scenarios (Requirement 7.1). Each scenario provides a conversation
(100+ turns where required), optional seed LTM facts, a final query, and the expected
key fact.

```python
@dataclass
class Scenario:
    id: str
    conversation: list[Message]
    seed_memories: list[tuple[str, dict]]   # (text, metadata)
    query: str
    key_fact: str                            # expected critical info
```

Representative scenarios:

- **flight_booking_memory:** booking code given early (turn 4) and stored in LTM;
  user asks for it after unrelated chit-chat (far-back + cross-session).
- **support_original_problem:** issue described at turn 2; user asks to recall it at
  turn 15 (far-back recall).
- Additional scenarios cover preferences recalled across sessions and
  irrelevant-heavy conversations.

## Data Models

| Model | Key Fields | Purpose |
|-------|-----------|---------|
| `Message` | text, role, turn, important | STM conversation unit |
| `MemoryRecord` | embedding, text, source_session, importance | LTM entry |
| `Candidate` | text, type, embedding, age, score | Scored context unit |
| `SelectionResult` | prompt, selected, omitted, token_count | Strategy output |
| `Scenario` | id, conversation, seed_memories, query, key_fact | Test case |
| `QualityResult` | passed, similarity | Evaluation outcome |
| `RunRecord` | scenario_id, strategy, token_count, quality, answer | Result row |

## Scoring Model Details

- **Raw relevance:** `cos = max(0, cosine_similarity(q_vec, cand_vec))` keeps scores
  in [0, 1] (Requirement 3.1).
- **Recency decay:** `score = cos * (decay_factor ** age)`, age 0 = most recent
  (Requirement 3.2). This is a simple analog to ALiBi's distance bias.
- **Memory aging:** LTM candidates get `age = len(stm) + LARGE_AGE_OFFSET` so they are
  treated as older than in-session messages (Requirement 3.3).
- **Decay sweep:** `decay_factor` is configurable; runs at 1.0 (no bias) and 0.95 are
  compared (Requirement 3.4 / 10.3).
- **Optional type weighting:** an optional multiplier can upweight `memory`/factual
  candidates over narrative ones (Requirement 3.5).

## Token Budget Handling

- Budget = `int(max_context_tokens * token_budget_ratio)` (default 80%), leaving
  headroom for the response (Requirement 4.6).
- All strategies enforce the same budget and LTM top-K except No-Pruning, which fills
  to `max_context_tokens` and truncates only at the hard limit (Requirement 5.4).
- Selection accumulates tokens and stops before the budget is exceeded
  (Requirement 4.3).

## Error Handling

| Condition | Handling |
|-----------|----------|
| Embedding/LLM model not downloadable or out of memory | Fail fast with an actionable message naming the model and suggesting 4-bit/smaller model. |
| Prompt exceeds model max even after selection | Truncate oldest-first and log a warning. |
| LTM search on empty store | Return empty list; strategies proceed without memory. |
| Empty/whitespace candidate text | Skip candidate, do not count tokens. |
| Tokenizer mismatch with LLM | Load tokenizer from the same model id; warn if overridden. |

## Testing Strategy

Tests are not auto-generated; this section describes the intended approach for when
tests are requested.

- **Unit tests:** cosine similarity correctness, decay math (`score` at ages 0/1/N and
  decay 1.0 vs 0.95), token budget cutoff (stops before exceeding), STM eviction
  order, LTM top-K ranking, evaluator Pass/Fail on substring presence.
- **Strategy tests:** with a stubbed LLM and deterministic embeddings, verify each
  strategy's selection behavior — Sliding-Window ignores LTM, No-Pruning includes
  everything, A3TK keeps keyword-flagged messages, Adaptive ranks the far-back key
  fact highly.
- **Integration test:** run the flight-booking scenario end-to-end with a stub LLM
  that echoes selected context; assert Adaptive preserves the key fact while
  Sliding-Window drops it, and that Adaptive token count < No-Pruning token count.
- **Reproducibility test:** fixed seed yields identical selection and ordering across
  runs.

## Verification Against Success Criteria

- **Token reduction (30–50%):** Reporter computes average Adaptive token count versus
  No-Pruning across scenarios.
- **Quality preservation:** Evaluator Pass-rate of Adaptive compared to No-Pruning and
  Sliding-Window, especially on far-back / LTM-only scenarios.
- **Reporting:** Reporter persists a scenario × strategy table (token count + quality)
  and an optional trade-off chart.

## Future Work (informational)

- Replace cosine scoring with a learned gate/classifier for relevance.
- Integrate the selector into a real A3TK orchestrator with Redis/Couchbase.
- Evaluate with larger/hosted models and real user interactions.
- Explore trainable ALiBi-style biases instead of a fixed decay factor.
