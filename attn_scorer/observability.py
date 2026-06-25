"""Observability: metrics, tracing hooks, and health checks."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Lightweight metrics collector for scoring operations.

    Tracks:
    - scoring_latency_ms: time per scoring call
    - selection_latency_ms: time per selection call
    - candidates_scored: count per call
    - tokens_selected: tokens in result
    - cache_hit_rate: embedding cache effectiveness
    - budget_utilization: how much of budget is used

    Exportable to Prometheus, JSON, or custom sinks.
    """

    def __init__(self):
        self._metrics: list[MetricPoint] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def record_latency(self, operation: str, duration_ms: float,
                       labels: dict[str, str] | None = None):
        """Record a latency measurement."""
        self._histograms[f"{operation}_ms"].append(duration_ms)
        self._metrics.append(MetricPoint(
            name=f"{operation}_ms", value=duration_ms,
            timestamp=time.time(), labels=labels or {},
        ))

    def record_counter(self, name: str, value: float = 1.0):
        """Increment a counter."""
        self._counters[name] += value

    def record_gauge(self, name: str, value: float,
                     labels: dict[str, str] | None = None):
        """Record a gauge value."""
        self._metrics.append(MetricPoint(
            name=name, value=value,
            timestamp=time.time(), labels=labels or {},
        ))

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary: dict[str, Any] = {"counters": dict(self._counters)}
        for name, values in self._histograms.items():
            if values:
                import numpy as np
                arr = np.array(values)
                summary[name] = {
                    "count": len(values),
                    "mean": float(np.mean(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(np.max(arr)),
                }
        return summary

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, values in self._histograms.items():
            if values:
                import numpy as np
                arr = np.array(values)
                safe_name = name.replace(".", "_")
                lines.append(f"# TYPE attn_scorer_{safe_name} summary")
                lines.append(f'attn_scorer_{safe_name}{{quantile="0.5"}} {np.percentile(arr, 50):.4f}')
                lines.append(f'attn_scorer_{safe_name}{{quantile="0.95"}} {np.percentile(arr, 95):.4f}')
                lines.append(f'attn_scorer_{safe_name}{{quantile="0.99"}} {np.percentile(arr, 99):.4f}')
                lines.append(f"attn_scorer_{safe_name}_count {len(values)}")
                lines.append(f"attn_scorer_{safe_name}_sum {sum(values):.4f}")
        for name, value in self._counters.items():
            safe_name = name.replace(".", "_")
            lines.append(f"# TYPE attn_scorer_{safe_name} counter")
            lines.append(f"attn_scorer_{safe_name} {value}")
        return "\n".join(lines)

    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._histograms.clear()


class TracingHook:
    """
    Hook interface for distributed tracing (OpenTelemetry-compatible).

    Register callbacks that fire on scoring events:
    - on_score_start(query, num_candidates)
    - on_score_end(query, num_selected, latency_ms)
    - on_cache_hit(text)
    - on_budget_exceeded(candidates_omitted)
    """

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def register(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        self._hooks[event].append(callback)

    def emit(self, event: str, **kwargs) -> None:
        """Emit an event to all registered callbacks."""
        for cb in self._hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception as e:
                logger.warning("Tracing hook error on %s: %s", event, e)


class HealthCheck:
    """Health check for the scorer service."""

    def __init__(self, scorer=None, embedding=None):
        self._scorer = scorer
        self._embedding = embedding

    def check(self) -> dict[str, Any]:
        """Run health checks and return status."""
        status: dict[str, Any] = {"healthy": True, "checks": {}}

        # Check embedding backend
        if self._embedding:
            try:
                vec = self._embedding.embed("health check")
                status["checks"]["embedding"] = {
                    "status": "ok",
                    "dimension": len(vec),
                }
            except Exception as e:
                status["healthy"] = False
                status["checks"]["embedding"] = {
                    "status": "error",
                    "error": str(e),
                }

        # Check scorer
        if self._scorer:
            try:
                from .models import Candidate
                cand = Candidate(text="test", ctype="history", age=0, turn=0)
                scored = self._scorer.score("test", [cand])
                status["checks"]["scorer"] = {
                    "status": "ok",
                    "scored_count": len(scored),
                }
            except Exception as e:
                status["healthy"] = False
                status["checks"]["scorer"] = {
                    "status": "error",
                    "error": str(e),
                }

        return status


# Global metrics instance (opt-in)
_global_metrics: MetricsCollector | None = None
_global_tracer: TracingHook | None = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_tracer() -> TracingHook:
    """Get or create the global tracing hook."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = TracingHook()
    return _global_tracer
