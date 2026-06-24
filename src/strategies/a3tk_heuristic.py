"""A3TK-style heuristic: keyword importance pruning + summarization + LTM retrieval."""

from __future__ import annotations

from ..config import Config
from ..embedding import EmbeddingService
from ..memory import MemoryManager, Message
from ..tokenizer_service import TokenizerService
from .base import SYSTEM_PROMPT, SelectionResult, format_prompt


class A3TKHeuristicStrategy:
    """
    Emulates the production A3TK heuristic:
    1. Keyword-based importance score per message.
    2. Drop oldest low-importance messages until within budget.
    3. Summarize dropped content if it exceeds a threshold.
    4. Retrieve top-K LTM entries.
    """

    name: str = "A3TK-Heuristic"

    def __init__(self, cfg: Config, tokenizer: TokenizerService,
                 embedding: EmbeddingService, llm_client=None) -> None:
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._embedding = embedding
        self._llm = llm_client  # optional; uses extractive fallback if None

    def _importance_score(self, msg: Message) -> float:
        """Compute a simple keyword-based importance score."""
        score = 0.0
        text_lower = msg.text.lower()
        for kw in self._cfg.importance_keywords:
            if kw in text_lower:
                score += 1.0
        if msg.important:
            score += 3.0
        # Numbers often indicate important data (codes, amounts)
        if any(ch.isdigit() for ch in msg.text):
            score += 0.5
        return score

    def _extractive_summary(self, messages: list[Message]) -> str:
        """Fallback: extract first sentence of each dropped message."""
        sentences = []
        for msg in messages:
            first_sentence = msg.text.split(".")[0].strip()
            if first_sentence:
                sentences.append(f"{msg.role}: {first_sentence}.")
        return "Summary of earlier conversation: " + " ".join(sentences)

    def _llm_summary(self, messages: list[Message]) -> str:
        """Use the LLM to summarize dropped messages."""
        text_block = "\n".join(f"{m.role}: {m.text}" for m in messages)
        prompt = (
            "Summarize the following conversation in 2-3 sentences, "
            "preserving any important facts, numbers, or references:\n\n"
            f"{text_block}\n\nSummary:"
        )
        return self._llm.generate(prompt)

    def build_prompt(self, query: str, memory: MemoryManager) -> SelectionResult:
        stm = memory.get_stm()
        q_vec = self._embedding.embed(query)

        # Retrieve LTM
        ltm_records = memory.search_ltm(q_vec, self._cfg.ltm_top_k)
        ltm_blocks = [f"[Memory]: {rec.text}" for rec in ltm_records]

        # Score all messages
        scored_msgs = [(self._importance_score(msg), msg) for msg in stm]

        # Determine overhead (system + query + LTM)
        overhead_prompt = format_prompt(SYSTEM_PROMPT, ltm_blocks, query)
        overhead_tokens = self._tokenizer.count(overhead_prompt)
        available_tokens = self._cfg.token_budget - overhead_tokens

        # Sort by importance ascending for pruning (lowest first to drop)
        # But we want to preserve order, so we mark which to keep
        keep_flags = [True] * len(scored_msgs)

        # Calculate total tokens of all messages
        msg_tokens = []
        for _, msg in scored_msgs:
            block = f"[Turn {msg.turn} - {msg.role}]: {msg.text}"
            msg_tokens.append(self._tokenizer.count(block))

        total_msg_tokens = sum(msg_tokens)

        # Drop oldest low-importance messages until within budget
        if total_msg_tokens > available_tokens:
            # Create index sorted by (importance ASC, age DESC = oldest first)
            indices_by_priority = sorted(
                range(len(scored_msgs)),
                key=lambda i: (scored_msgs[i][0], -scored_msgs[i][1].turn)
            )
            tokens_to_free = total_msg_tokens - available_tokens
            freed = 0
            for idx in indices_by_priority:
                if freed >= tokens_to_free:
                    break
                keep_flags[idx] = False
                freed += msg_tokens[idx]

        # Separate kept and dropped
        kept_msgs: list[Message] = []
        dropped_msgs: list[Message] = []
        for i, (_, msg) in enumerate(scored_msgs):
            if keep_flags[i]:
                kept_msgs.append(msg)
            else:
                dropped_msgs.append(msg)

        # Summarize dropped if exceeding threshold
        summary_block: str | None = None
        if dropped_msgs:
            dropped_tokens = sum(
                self._tokenizer.count(m.text) for m in dropped_msgs
            )
            if dropped_tokens > self._cfg.summarization_threshold_tokens:
                if self._llm is not None:
                    summary_block = self._llm_summary(dropped_msgs)
                else:
                    summary_block = self._extractive_summary(dropped_msgs)

        # Assemble context blocks
        context_blocks: list[str] = []
        if summary_block:
            context_blocks.append(f"[Summary]: {summary_block}")
        for msg in kept_msgs:
            context_blocks.append(f"[Turn {msg.turn} - {msg.role}]: {msg.text}")
        context_blocks.extend(ltm_blocks)

        prompt = format_prompt(SYSTEM_PROMPT, context_blocks, query)
        token_count = self._tokenizer.count(prompt)

        # Build selected/omitted lists
        selected = context_blocks[:]
        omitted = [f"[Turn {msg.turn} - {msg.role}]: {msg.text}" for msg in dropped_msgs]

        return SelectionResult(
            prompt=prompt,
            selected=selected,
            omitted=omitted,
            token_count=token_count,
        )
