"""Reporter: produces comparison tables, CSV output, and summary statistics."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

from .runner import RunRecord


class Reporter:
    """Produces results table, CSV persistence, and summary statistics."""

    def to_table(self, records: list[RunRecord]) -> str:
        """Generate a Markdown table of results (scenario × strategy)."""
        # Group by scenario
        scenarios_order: list[str] = []
        seen = set()
        for r in records:
            if r.scenario_id not in seen:
                scenarios_order.append(r.scenario_id)
                seen.add(r.scenario_id)

        strategies_order: list[str] = []
        seen_s = set()
        for r in records:
            if r.strategy not in seen_s:
                strategies_order.append(r.strategy)
                seen_s.add(r.strategy)

        # Build lookup
        lookup: dict[tuple[str, str], RunRecord] = {}
        for r in records:
            lookup[(r.scenario_id, r.strategy)] = r

        # Header
        header = "| Scenario | " + " | ".join(
            f"{s} (tokens / quality)" for s in strategies_order
        ) + " |"
        separator = "|" + "---|" * (len(strategies_order) + 1)

        rows = []
        for scen_id in scenarios_order:
            cells = [scen_id]
            for strat in strategies_order:
                rec = lookup.get((scen_id, strat))
                if rec:
                    q = "PASS" if rec.quality.passed else "FAIL"
                    cells.append(f"{rec.token_count} / {q}")
                else:
                    cells.append("—")
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header, separator] + rows)

    def summary_stats(self, records: list[RunRecord]) -> str:
        """Compute and format summary statistics."""
        by_strategy: dict[str, list[RunRecord]] = defaultdict(list)
        for r in records:
            by_strategy[r.strategy].append(r)

        lines = ["\n--- Summary Statistics ---\n"]

        for strat, recs in by_strategy.items():
            avg_tokens = sum(r.token_count for r in recs) / len(recs)
            pass_rate = sum(1 for r in recs if r.quality.passed) / len(recs) * 100
            lines.append(
                f"  {strat:20s}  avg_tokens={avg_tokens:7.0f}  "
                f"pass_rate={pass_rate:5.1f}%"
            )

        # Token reduction: Adaptive vs No-Pruning
        reduction = self.avg_reduction(records)
        lines.append(f"\n  Adaptive token reduction vs No-Pruning: {reduction:.1f}%")
        return "\n".join(lines)

    def avg_reduction(self, records: list[RunRecord]) -> float:
        """Average token reduction of Adaptive vs No-Pruning (percentage)."""
        no_pruning: dict[str, int] = {}
        adaptive: dict[str, int] = {}

        for r in records:
            if r.strategy == "No-Pruning":
                no_pruning[r.scenario_id] = r.token_count
            elif r.strategy == "Adaptive":
                adaptive[r.scenario_id] = r.token_count

        if not no_pruning or not adaptive:
            return 0.0

        reductions = []
        for scen_id in no_pruning:
            if scen_id in adaptive and no_pruning[scen_id] > 0:
                reduction = (1 - adaptive[scen_id] / no_pruning[scen_id]) * 100
                reductions.append(reduction)

        return sum(reductions) / len(reductions) if reductions else 0.0

    def persist(self, records: list[RunRecord], path: str) -> None:
        """Write results to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario_id", "strategy", "token_count",
                "passed", "similarity", "answer_preview"
            ])
            for r in records:
                writer.writerow([
                    r.scenario_id,
                    r.strategy,
                    r.token_count,
                    int(r.quality.passed),
                    f"{r.quality.similarity:.4f}" if r.quality.similarity is not None else "",
                    r.answer[:200].replace("\n", " "),
                ])

        print(f"\nResults saved to: {path}")

    def chart(self, records: list[RunRecord], path: str) -> None:
        """Optional: generate a bar chart of token count and pass rate per strategy."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping chart generation.")
            return

        by_strategy: dict[str, list[RunRecord]] = defaultdict(list)
        for r in records:
            by_strategy[r.strategy].append(r)

        strategies = list(by_strategy.keys())
        avg_tokens = [
            sum(r.token_count for r in recs) / len(recs)
            for recs in by_strategy.values()
        ]
        pass_rates = [
            sum(1 for r in recs if r.quality.passed) / len(recs) * 100
            for recs in by_strategy.values()
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Token count bar chart
        ax1.bar(strategies, avg_tokens, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"])
        ax1.set_ylabel("Average Prompt Tokens")
        ax1.set_title("Token Usage by Strategy")
        ax1.tick_params(axis="x", rotation=15)

        # Pass rate bar chart
        ax2.bar(strategies, pass_rates, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"])
        ax2.set_ylabel("Pass Rate (%)")
        ax2.set_title("Answer Quality by Strategy")
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Chart saved to: {path}")
