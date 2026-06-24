# Requirements Document

## Introduction

This feature is a proof-of-concept (PoC) that demonstrates an attention-inspired
context filtering mechanism for conversational memory management. The goal is to
reduce prompt token length sent to a large language model (LLM) by a significant
margin (targeting 30–50% reduction versus full context) while preserving the
quality of the model's answers.

The PoC emulates a learnable-attention approach (semantic relevance scoring plus a
positional/recency decay bias) at the prompt-engineering level rather than inside
the model. It selects only the most relevant pieces of conversation history and
long-term memory to include in each prompt. The adaptive strategy is evaluated
against three baselines: (1) no pruning / full context, (2) static sliding window,
and (3) an A3TK-style heuristic (importance pruning plus summarization).

The system is a self-contained, single-machine Python pipeline. It simulates
short-term memory (STM, normally Redis) and long-term memory (LTM, normally
Couchbase vector search) with in-memory data structures, and runs against a local
open-source LLM (e.g., Mistral 7B or Qwen 7B). No production infrastructure
integration is in scope.

### Glossary

- **STM (Short-Term Memory):** Recent conversation history, simulated in-memory.
- **LTM (Long-Term Memory):** Persisted facts/memories with embeddings, simulated
  in-memory with brute-force or FAISS vector search.
- **Candidate:** A unit of context (a conversation message/chunk or a memory entry)
  eligible for inclusion in the prompt.
- **Relevance score:** Cosine similarity between a candidate embedding and the query
  embedding, modulated by a recency decay factor.
- **Token budget:** Maximum number of context tokens allowed in a prompt
  (configured as ~80% of the model's max context length).
- **A3TK baseline:** A simplified emulation of the existing production heuristic
  (importance keyword pruning + summarization + RAG retrieval).

## Requirements

### Requirement 1: Conversation and Memory Simulation

**User Story:** As a PoC researcher, I want a simulated orchestrator with in-memory
STM and LTM, so that I can run end-to-end context-management experiments without
production infrastructure.

#### Acceptance Criteria

1. WHEN the pipeline is initialized THEN the system SHALL provide an STM structure
   (list or deque) where each item holds at least a message text, a speaker role,
   a turn index/timestamp, and an optional importance flag.
2. WHEN the pipeline is initialized THEN the system SHALL provide an LTM store where
   each record holds at least an embedding vector, the source text, and metadata
   (source session and importance).
3. WHEN a test scenario is loaded THEN the system SHALL support a conversation of at
   least 100 turns that mixes irrelevant chit-chat with critical information.
4. WHEN seed memory facts are defined for a scenario THEN the system SHALL insert
   them into the LTM store prior to running that scenario.
5. WHEN a new message is added to STM THEN the system SHALL append it, and WHEN STM
   exceeds its configured capacity THEN the system SHALL evict the oldest items
   (emulating Redis list operations).

### Requirement 2: Embedding and Tokenization

**User Story:** As a PoC researcher, I want consistent embedding and token-counting
utilities, so that relevance scoring and budget enforcement are accurate and
comparable across strategies.

#### Acceptance Criteria

1. WHEN any text or query needs scoring THEN the system SHALL compute embeddings
   using a single configured local embedding model (e.g., all-MiniLM) producing
   384–768 dimensional vectors.
2. WHEN the query and candidates are embedded THEN the system SHALL use the same
   embedding model for both to ensure comparable similarity scores.
3. WHEN token counting is required THEN the system SHALL count tokens using the
   tokenizer that corresponds to the target LLM.
4. WHEN embeddings are computed repeatedly for the same text THEN the system SHOULD
   cache results to avoid redundant computation.

### Requirement 3: Relevance Scoring Function

**User Story:** As a PoC researcher, I want a pseudo-attention relevance scoring
function, so that each candidate context piece is weighted by its semantic
relevance to the current query and its recency.

#### Acceptance Criteria

1. WHEN scoring a candidate THEN the system SHALL compute the cosine similarity
   between the candidate embedding and the query embedding as the raw relevance
   score in the range [0, 1] (clamped if negative).
2. WHEN a recency decay factor is configured THEN the system SHALL compute the final
   score as `cos_sim(candidate, query) * (decay_factor ** age_in_turns)`, where the
   most recent message has age 0.
3. WHEN a candidate is an LTM memory entry THEN the system SHALL assign it an age
   value beyond the current conversation length so memories are treated as older
   than in-session messages.
4. WHEN the decay factor is set to 1.0 THEN the system SHALL apply no recency bias,
   and WHEN it is set below 1.0 (e.g., 0.95) THEN the system SHALL down-weight older
   items accordingly.
5. WHERE candidate type matters THE system MAY allow type-based weighting (e.g.,
   upweight factual memories versus narrative chit-chat).

### Requirement 4: Adaptive Context Selection

**User Story:** As a PoC researcher, I want an adaptive selection algorithm that
keeps only the highest-scoring context within a token budget, so that prompts stay
short while retaining critical information.

#### Acceptance Criteria

1. WHEN `select_context(query, stm_history, ltm_memories, token_budget)` is called
   THEN the system SHALL build a candidate list from the N most recent STM messages,
   any older-segment summaries available, and the top-M LTM retrievals by similarity.
2. WHEN candidates are assembled THEN the system SHALL score each candidate using the
   relevance scoring function and sort candidates by descending score.
3. WHEN selecting within the budget THEN the system SHALL accumulate candidates from
   highest score downward and stop before the next candidate would exceed
   `token_budget`.
4. WHEN the prompt is constructed THEN the system SHALL order the selected
   conversation items chronologically while still respecting which items were chosen
   by score.
5. WHEN selection completes THEN the system SHALL record which candidates were
   Selected and which were Omitted for later analysis.
6. WHERE the token budget is defined THE system SHALL default it to approximately 80%
   of the model's maximum context length to leave headroom for the response.

### Requirement 5: Baseline Strategies

**User Story:** As a PoC researcher, I want the three baseline strategies
implemented alongside the adaptive one, so that I can compare token usage and
quality fairly.

#### Acceptance Criteria

1. WHEN the No-Pruning strategy runs THEN the system SHALL include the entire
   conversation and all relevant memory items, truncating only if the model's
   maximum context length is exceeded.
2. WHEN the Sliding-Window strategy runs THEN the system SHALL include only the most
   recent N messages (or last X tokens) from STM and SHALL ignore LTM.
3. WHEN the A3TK heuristic strategy runs THEN the system SHALL assign each message a
   keyword-based importance score, drop older low-score messages until within budget,
   summarize a chunk of dropped messages when too much context is removed, and
   retrieve top-K LTM entries.
4. WHEN comparing strategies THEN the system SHALL apply the same token budget and
   LTM retrieval count across all strategies, except No-Pruning which fills up to the
   model maximum.
5. WHEN the A3TK summarization step is invoked THEN the system SHALL produce a shorter
   summary of the dropped messages (using the local LLM or an approximation) and
   include it in place of the dropped content.

### Requirement 6: LLM Invocation

**User Story:** As a PoC researcher, I want to call a local open-source LLM with the
assembled prompt, so that I can generate and evaluate answers per strategy.

#### Acceptance Criteria

1. WHEN a prompt is assembled THEN the system SHALL invoke a configured local LLM
   (e.g., Mistral 7B or Qwen 7B) and capture the generated answer text.
2. WHERE limited hardware is used THE system SHALL support loading the model with
   4-bit quantization.
3. WHEN a prompt is sent THEN the system SHALL record the exact prompt token count
   used for that call.
4. IF the model or its dependencies are unavailable THEN the system SHALL fail with a
   clear, actionable error message.

### Requirement 7: Experiment Runner and Scenarios

**User Story:** As a PoC researcher, I want a runner that executes every scenario
across all four strategies, so that I can collect comparable measurements in one
pass.

#### Acceptance Criteria

1. WHEN scenarios are defined THEN the system SHALL support 5–10 scenarios, each with
   a conversation, optional seed LTM facts, a final user query, and an expected
   answer / key fact.
2. WHEN the runner executes THEN the system SHALL run each scenario through all four
   strategies (No-Pruning, Sliding-Window, A3TK, Adaptive).
3. WHEN at least one scenario requires far-back context THEN the system SHALL include
   a scenario where the needed fact is early in the conversation.
4. WHEN at least one scenario requires cross-session memory THEN the system SHALL
   include a scenario where the needed fact exists only in LTM.
5. WHEN a strategy run completes for a scenario THEN the system SHALL capture the
   prompt token count, the LLM output text, and a quality outcome.

### Requirement 8: Quality Evaluation

**User Story:** As a PoC researcher, I want an automated quality check on each
answer, so that I can score whether the critical information was preserved.

#### Acceptance Criteria

1. WHEN an answer is produced THEN the system SHALL evaluate whether the expected key
   fact (e.g., a booking code) is present in the output.
2. WHEN the key fact is present THEN the system SHALL mark the result as Pass (1), and
   WHEN it is absent THEN the system SHALL mark it as Fail (0).
3. WHERE a finer measure is desired THE system MAY also compute a similarity score
   between the output and the expected answer.
4. WHEN evaluation completes THEN the system SHALL store the quality outcome alongside
   the token count for that scenario/strategy pair.

### Requirement 9: Results Reporting

**User Story:** As a PoC researcher, I want a consolidated results report, so that I
can judge the token-versus-quality trade-off of each strategy.

#### Acceptance Criteria

1. WHEN all runs complete THEN the system SHALL produce a results table indexed by
   scenario and strategy showing prompt token count and quality outcome.
2. WHEN the table is produced THEN the system SHALL compute the average token
   reduction of the Adaptive strategy versus the No-Pruning baseline.
3. WHERE charting is enabled THE system MAY produce a chart of token count versus
   quality per strategy.
4. WHEN reporting completes THEN the system SHALL persist the results to a file
   (e.g., CSV or Markdown) for later review.

### Requirement 10: Configuration and Reproducibility

**User Story:** As a PoC researcher, I want all key parameters configurable and runs
reproducible, so that I can tune and re-run experiments reliably.

#### Acceptance Criteria

1. WHEN the pipeline starts THEN the system SHALL read configuration for the decay
   factor, token budget, sliding-window size N, LTM top-K/M, embedding model, and LLM
   model from a single configuration source.
2. WHEN a run uses randomness THEN the system SHALL allow setting a fixed random seed
   for reproducibility.
3. WHEN the decay factor is varied (e.g., 1.0 vs 0.95) THEN the system SHALL allow
   running the comparison without code changes.
4. WHEN the pipeline runs THEN the system SHALL be executable as a single Python entry
   point.

## Success Criteria

- The Adaptive strategy achieves an average token reduction of 30–50% versus the
  No-Pruning baseline across scenarios.
- The Adaptive strategy preserves the critical key fact in its answers at a rate
  comparable to No-Pruning, and clearly higher than the naive Sliding-Window baseline
  on scenarios where the needed information is far back or only in LTM.
- A results table and summary report are produced capturing token count and quality
  per scenario and strategy.

## Out of Scope

- Integration with production Redis or Couchbase infrastructure.
- Training a learned scoring model or fine-tuning the LLM (noted as future work).
- Using large hosted models such as GPT-4.1 for the PoC runs.
- Production-grade performance, concurrency, or deployment concerns.
