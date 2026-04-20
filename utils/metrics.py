"""
In-process metrics collector — no external dependencies.

Tracks counters and timing histograms for the lifetime of the process.
Exposed via GET /metrics in the API layer.

Usage:
    from utils.metrics import metrics
    metrics.inc("llm_requests_total")
    with metrics.timer("command_latency_ms"):
        result = engine.run(...)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Generator, List


class _Histogram:
    """Stores a list of observed float values; reports p50/p95/p99."""

    __slots__ = ("_values", "_lock")

    def __init__(self) -> None:
        self._values: List[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self._values:
                return {"count": 0, "sum": 0.0, "min": None, "max": None,
                        "p50": None, "p95": None, "p99": None}
            sorted_vals = sorted(self._values)
            n = len(sorted_vals)

            def _pct(p: float) -> float:
                idx = max(0, int(n * p / 100) - 1)
                return round(sorted_vals[idx], 3)

            return {
                "count": n,
                "sum": round(sum(sorted_vals), 3),
                "min": round(sorted_vals[0], 3),
                "max": round(sorted_vals[-1], 3),
                "p50": _pct(50),
                "p95": _pct(95),
                "p99": _pct(99),
            }


class MetricsCollector:
    """
    Thread-safe in-process metrics store.

    Counters: monotonically increasing integers.
    Histograms: lists of float samples (timings, sizes, etc.).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, _Histogram] = {}

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = _Histogram()
            h = self._histograms[name]
        h.observe(value)

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """Context manager that records elapsed milliseconds into a histogram."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.observe(name, elapsed_ms)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            hist_names = list(self._histograms.keys())

        histograms = {name: self._histograms[name].snapshot() for name in hist_names}
        return {"counters": counters, "histograms": histograms}

    def reset(self) -> None:
        """Clear all data — intended for tests only."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Module-level singleton
metrics = MetricsCollector()
