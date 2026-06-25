"""Production-scale benchmark runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import ScorerConfig
from ..embeddings.base import EmbeddingBackend
from ..models import ContextResult
from ..scorer import Scorer
from .prod_dataset import ProdScenario


@dataclass
class BenchmarkResult:
    """Result for a single scenario run."""
    scenario_id: str
    num_turns: int
    total_tokens: int
    selected_tokens: int
    reduction_pct: float
    key_facts_found: list[bool]
    key_fact_rate: float
    hard_negatives_selected: int
    false_positive_rate: float
    latency_ms: float


@dataclass
class BenchmarkReport:
    """Full benchmark report across all scenarios."""
    results: list[BenchmarkResult] = field(default_factory=list)
    avg_reduction: float = 0.0
    avg_key_fact_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_false_positive_rate: float = 0.0
    total_scenarios: int = 0

    def compute_summary(self) -> None:
        if not self.results:
            return
        self.total_scenarios = len(self.results)
        self.avg_reduction = sum(r.reduction_pct for r in self.results) / len(self.results)
        self.avg_key_fact_rate = sum(r.key_fact_rate for r in self.results) / len(self.results)
        self.avg_latency_ms = sum(r.latency_ms for r in self.results) / len(self.results)
        self.avg_false_positive_rate = sum(r.false_positive_rate for r in self.results) / len(self.results)

    def print_report(self) -> str:
        self.compute_summary()
        lines = [
            "=" * 70,
            "PRODUCTION BENCHMARK REPORT",
            "=" * 70,
            f"Scenarios: {self.total_scenarios}",
            f"Avg Token Reduction: {self.avg_reduction:.1f}%",
            f"Avg Key Fact Preservation: {self.avg_key_fact_rate * 100:.1f}%",
            f"Avg Scoring Latency: {self.avg_latency_ms:.2f}ms",
            f"Avg False Positive Rate: {self.avg_false_positive_rate * 100:.1f}%",
            "",
            f"{'Scenario':<45} {'Turns':>6} {'Tokens':>7} {'Red%':>5} {'Facts':>6} {'Lat':>7}",
            "-" * 70,
        ]
        for r in self.results:
            fact_str = f"{r.key_fact_rate*100:.0f}%"
            lines.append(
                f"{r.scenario_id:<45} {r.num_turns:>6} "
                f"{r.selected_tokens:>7} {r.reduction_pct:>4.0f}% "
                f"{fact_str:>6} {r.latency_ms:>6.1f}ms"
            )
        lines.append("=" * 70)
        report = "\n".join(lines)
        print(report)
        return report


class ProdBenchmarkRunner:
    """Runs the production benchmark suite."""

    def __init__(
        self,
        config: ScorerConfig,
        embedding: EmbeddingBackend,
        token_counter=None,
        token_budget: int | None = None,
    ) -> None:
        self._config = config
        self._scorer = Scorer(
            config=config, embedding=embedding, token_counter=token_counter
        )
        self._token_counter = token_counter or (lambda t: int(len(t.split()) * 1.3))
        self._budget = token_budget or config.default_token_budget

    def run_sync(self, scenarios: list[ProdScenario]) -> BenchmarkReport:
        """Run benchmark synchronously."""
        report = BenchmarkReport()

        for scenario in scenarios:
            result = self._run_scenario(scenario)
            report.results.append(result)

        report.compute_summary()
        return report

    def _run_scenario(self, scenario: ProdScenario) -> BenchmarkResult:
        """Run a single scenario."""
        # Compute full token count
        full_tokens = sum(self._token_counter(c.text) for c in scenario.candidates)

        # Score and select
        t0 = time.perf_counter()
        scored = self._scorer.score(scenario.query, scenario.candidates)
        result = self._scorer.select(scored, self._budget)
        latency = (time.perf_counter() - t0) * 1000

        # Check key facts
        selected_text = " ".join(sc.candidate.text for sc in result.selected)
        facts_found = [
            fact.lower() in selected_text.lower()
            for fact in scenario.key_facts
        ]
        fact_rate = sum(facts_found) / len(facts_found) if facts_found else 1.0

        # Check false positives (hard negatives in selection)
        selected_turns = {sc.candidate.turn for sc in result.selected}
        hn_selected = sum(
            1 for idx in scenario.hard_negative_indices if idx in selected_turns
        )
        fp_rate = (
            hn_selected / len(scenario.hard_negative_indices)
            if scenario.hard_negative_indices else 0.0
        )

        # Reduction
        reduction = (1 - result.token_count / full_tokens) * 100 if full_tokens > 0 else 0

        return BenchmarkResult(
            scenario_id=scenario.id,
            num_turns=scenario.num_turns,
            total_tokens=full_tokens,
            selected_tokens=result.token_count,
            reduction_pct=reduction,
            key_facts_found=facts_found,
            key_fact_rate=fact_rate,
            hard_negatives_selected=hn_selected,
            false_positive_rate=fp_rate,
            latency_ms=latency,
        )
