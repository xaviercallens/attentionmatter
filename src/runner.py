"""Experiment runner: executes every scenario through all strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .embedding import EmbeddingService
from .evaluator import Evaluator, QualityResult
from .memory import MemoryManager
from .scenarios import Scenario
from .strategies.base import ContextStrategy, SelectionResult
from .tokenizer_service import TokenizerService


@dataclass
class RunRecord:
    """Result of a single scenario × strategy execution."""
    scenario_id: str
    strategy: str
    token_count: int
    quality: QualityResult
    answer: str


class ExperimentRunner:
    """Drives the full scenario × strategy matrix."""

    def __init__(
        self,
        cfg: Config,
        strategies: list[ContextStrategy],
        llm,
        evaluator: Evaluator,
        scenarios: list[Scenario],
        embedding_service: EmbeddingService,
        tokenizer_service: TokenizerService,
    ) -> None:
        self._cfg = cfg
        self._strategies = strategies
        self._llm = llm
        self._evaluator = evaluator
        self._scenarios = scenarios
        self._embedding = embedding_service
        self._tokenizer = tokenizer_service

    def _populate_memory(self, scenario: Scenario, memory: MemoryManager) -> None:
        """Load a scenario's conversation and seed memories into the manager."""
        # Insert seed LTM facts first
        for text, meta in scenario.seed_memories:
            memory.insert_memory(
                text,
                source_session=meta.get("source_session", "default"),
                importance=meta.get("importance", 1.0),
            )
        # Add conversation messages to STM
        for msg in scenario.conversation:
            memory.add_message(msg)

    def run(self) -> list[RunRecord]:
        """Execute every scenario through every strategy and collect results."""
        records: list[RunRecord] = []
        total = len(self._scenarios) * len(self._strategies)
        current = 0

        for scenario in self._scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario.id} — {scenario.description}")
            print(f"{'='*60}")

            for strategy in self._strategies:
                current += 1
                print(f"  [{current}/{total}] Strategy: {strategy.name}...", end=" ")

                # Fresh memory for each strategy run
                memory = MemoryManager(self._cfg, self._embedding)
                self._populate_memory(scenario, memory)

                # Build prompt
                result: SelectionResult = strategy.build_prompt(scenario.query, memory)

                # Generate answer
                answer = self._llm.generate(result.prompt)

                # Evaluate
                quality = self._evaluator.score(answer, scenario.key_fact)

                status = "PASS" if quality.passed else "FAIL"
                print(f"tokens={result.token_count}, quality={status}")

                records.append(RunRecord(
                    scenario_id=scenario.id,
                    strategy=strategy.name,
                    token_count=result.token_count,
                    quality=quality,
                    answer=answer,
                ))

        return records
