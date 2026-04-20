"""
Tests for the Observability layer (Phase 8).

Covers:
  - MetricsCollector (counters, histograms, timer, snapshot, reset)
  - RequestLoggingMiddleware (status logging, X-Request-Id header, error counting)
  - /metrics endpoint
  - log_context helper
  - Engine + registry metric instrumentation

Run:
    pytest tests/test_observability.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import MetricsCollector
from utils.logger import log_context, get_logger


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    @pytest.fixture(autouse=True)
    def fresh(self):
        self.m = MetricsCollector()

    def test_inc_default(self):
        self.m.inc("hits")
        assert self.m.get_counter("hits") == 1

    def test_inc_amount(self):
        self.m.inc("hits", 5)
        assert self.m.get_counter("hits") == 5

    def test_inc_accumulates(self):
        self.m.inc("hits")
        self.m.inc("hits")
        self.m.inc("hits", 3)
        assert self.m.get_counter("hits") == 5

    def test_get_counter_missing_returns_zero(self):
        assert self.m.get_counter("nonexistent") == 0

    def test_observe_and_snapshot(self):
        for v in [10.0, 20.0, 30.0]:
            self.m.observe("latency", v)
        s = self.m.snapshot()
        h = s["histograms"]["latency"]
        assert h["count"] == 3
        assert h["min"] == 10.0
        assert h["max"] == 30.0
        assert h["sum"] == 60.0

    def test_histogram_percentiles(self):
        for v in range(1, 101):
            self.m.observe("vals", float(v))
        snap = self.m.snapshot()["histograms"]["vals"]
        assert snap["p50"] is not None
        assert snap["p99"] is not None
        assert snap["p99"] >= snap["p50"]

    def test_histogram_empty_snapshot(self):
        self.m.observe("empty_hist", 1.0)   # create it
        fresh = MetricsCollector()           # fresh instance has no data
        snap = fresh.snapshot()
        assert snap["histograms"] == {}

    def test_timer_records_positive_duration(self):
        with self.m.timer("op_ms"):
            time.sleep(0.01)
        snap = self.m.snapshot()
        assert "op_ms" in snap["histograms"]
        assert snap["histograms"]["op_ms"]["min"] >= 5.0   # at least 5 ms

    def test_timer_records_even_on_exception(self):
        try:
            with self.m.timer("fail_ms"):
                raise ValueError("boom")
        except ValueError:
            pass
        snap = self.m.snapshot()
        assert "fail_ms" in snap["histograms"]

    def test_snapshot_includes_counters(self):
        self.m.inc("alpha")
        self.m.inc("beta", 2)
        snap = self.m.snapshot()
        assert snap["counters"]["alpha"] == 1
        assert snap["counters"]["beta"] == 2

    def test_reset_clears_all(self):
        self.m.inc("x")
        self.m.observe("y", 1.0)
        self.m.reset()
        snap = self.m.snapshot()
        assert snap["counters"] == {}
        assert snap["histograms"] == {}

    def test_thread_safety(self):
        import threading
        threads = [
            threading.Thread(target=self.m.inc, args=("t_counter",))
            for _ in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert self.m.get_counter("t_counter") == 100


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware via TestClient
# ---------------------------------------------------------------------------

class TestRequestLoggingMiddleware:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import create_app
        app = create_app(memory_manager=None)
        return TestClient(app)

    def test_request_id_header_present(self, client):
        r = client.get("/health")
        assert "x-request-id" in r.headers
        assert len(r.headers["x-request-id"]) == 8

    def test_successful_request_increments_counter(self, client):
        from utils.metrics import metrics
        metrics.reset()
        client.get("/health")
        assert metrics.get_counter("api_requests_total") >= 1

    def test_failed_request_increments_error_counter(self, client):
        from utils.metrics import metrics
        metrics.reset()
        client.get("/nonexistent_route_xyz")
        assert metrics.get_counter("api_errors_total") >= 1

    def test_duration_recorded_in_histogram(self, client):
        from utils.metrics import metrics
        metrics.reset()
        client.get("/health")
        snap = metrics.snapshot()
        assert "api_request_duration_ms" in snap["histograms"]
        assert snap["histograms"]["api_request_duration_ms"]["count"] >= 1

    def test_each_request_gets_unique_id(self, client):
        ids = set()
        for _ in range(5):
            r = client.get("/health")
            ids.add(r.headers["x-request-id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import create_app
        return TestClient(create_app(memory_manager=None))

    def test_metrics_endpoint_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_endpoint_has_counters_and_histograms(self, client):
        r = client.get("/metrics")
        body = r.json()
        assert "counters" in body
        assert "histograms" in body

    def test_metrics_reflect_activity(self, client):
        from utils.metrics import metrics
        metrics.reset()
        client.get("/health")
        client.get("/health")
        r = client.get("/metrics")
        body = r.json()
        # At least 2 requests logged (the health checks themselves)
        assert body["counters"].get("api_requests_total", 0) >= 2


# ---------------------------------------------------------------------------
# Engine + Registry instrumentation
# ---------------------------------------------------------------------------

class TestEngineMetrics:
    def test_engine_run_increments_commands_total(self):
        from utils.metrics import metrics
        from agents.engine import AgentEngine, EngineResult
        from agents.base import AgentResult

        metrics.reset()
        engine = AgentEngine()

        mock_result = AgentResult(response="ok", success=True)
        with patch.object(engine, "_run_single_agent", return_value=mock_result):
            engine.run("test command")

        assert metrics.get_counter("commands_total") >= 1

    def test_engine_run_records_latency(self):
        from utils.metrics import metrics
        from agents.engine import AgentEngine
        from agents.base import AgentResult

        metrics.reset()
        engine = AgentEngine()

        mock_result = AgentResult(response="ok", success=True)
        with patch.object(engine, "_run_single_agent", return_value=mock_result):
            engine.run("measure me")

        snap = metrics.snapshot()
        assert "command_latency_ms" in snap["histograms"]

    def test_engine_failure_increments_errors(self):
        from utils.metrics import metrics
        from agents.engine import AgentEngine

        metrics.reset()
        engine = AgentEngine()

        with patch.object(engine, "_run_single_agent", side_effect=RuntimeError("fail")):
            engine.run("break it")

        assert metrics.get_counter("command_errors_total") >= 1


class TestRegistryMetrics:
    def test_tool_execution_increments_counter(self):
        from utils.metrics import metrics
        from tools.loader import load_all_tools
        from tools.registry import registry

        load_all_tools()
        metrics.reset()
        registry.execute("get_system_info")
        assert metrics.get_counter("tool_calls_total") >= 1

    def test_tool_execution_records_timing(self):
        from utils.metrics import metrics
        from tools.registry import registry

        metrics.reset()
        registry.execute("get_system_info")
        snap = metrics.snapshot()
        assert "tool_execution_ms" in snap["histograms"]

    def test_tool_error_increments_error_counter(self):
        from utils.metrics import metrics
        from tools.registry import registry

        metrics.reset()
        with pytest.raises(KeyError):
            registry.execute("nonexistent_tool_xyz")
        assert metrics.get_counter("tool_errors_total") >= 1


# ---------------------------------------------------------------------------
# log_context
# ---------------------------------------------------------------------------

class TestLogContext:
    def test_log_context_yields_adapter(self):
        import logging
        log = get_logger("test.ctx")
        with log_context(log, request_id="abc") as adapter:
            assert isinstance(adapter, logging.LoggerAdapter)
            assert adapter.extra["request_id"] == "abc"

    def test_log_context_multiple_fields(self):
        import logging
        log = get_logger("test.ctx2")
        with log_context(log, a=1, b="two", c=True) as adapter:
            assert adapter.extra == {"a": 1, "b": "two", "c": True}
