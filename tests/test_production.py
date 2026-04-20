"""
Tests for Phase 9: Production Readiness.

Covers:
  - Config validation (Settings.validate)
  - Startup health checks (run_startup_checks)
  - Graceful shutdown (ShutdownManager)
  - Command queue (CommandQueue)

Run:
    pytest tests/test_production.py -v
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_settings_returns_no_errors(self):
        from config import settings
        errors = settings.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_invalid_port_fails(self):
        from config import settings
        original = settings.api.port
        settings.api.port = 99999
        try:
            errors = settings.validate()
            assert any("port" in e for e in errors)
        finally:
            settings.api.port = original

    def test_invalid_base_url_fails(self):
        from config import settings
        original = settings.llm.base_url
        settings.llm.base_url = "not-a-url"
        try:
            errors = settings.validate()
            assert any("base_url" in e for e in errors)
        finally:
            settings.llm.base_url = original

    def test_max_tokens_zero_fails(self):
        from config import settings
        original = settings.llm.max_tokens
        settings.llm.max_tokens = 0
        try:
            errors = settings.validate()
            assert any("max_tokens" in e for e in errors)
        finally:
            settings.llm.max_tokens = original

    def test_temperature_out_of_range_fails(self):
        from config import settings
        original = settings.llm.temperature
        settings.llm.temperature = 3.5
        try:
            errors = settings.validate()
            assert any("temperature" in e for e in errors)
        finally:
            settings.llm.temperature = original

    def test_max_iterations_zero_fails(self):
        from config import settings
        original = settings.agent.max_iterations
        settings.agent.max_iterations = 0
        try:
            errors = settings.validate()
            assert any("max_iterations" in e for e in errors)
        finally:
            settings.agent.max_iterations = original

    def test_invalid_log_level_fails(self):
        from config import settings
        original = settings.logging.level
        settings.logging.level = "VERBOSE"
        try:
            errors = settings.validate()
            assert any("level" in e for e in errors)
        finally:
            settings.logging.level = original

    def test_multiple_errors_returned(self):
        from config import settings
        orig_port = settings.api.port
        orig_tokens = settings.llm.max_tokens
        settings.api.port = 0
        settings.llm.max_tokens = 0
        try:
            errors = settings.validate()
            assert len(errors) >= 2
        finally:
            settings.api.port = orig_port
            settings.llm.max_tokens = orig_tokens


# ---------------------------------------------------------------------------
# Startup Checks
# ---------------------------------------------------------------------------

class TestStartupChecks:
    def test_run_startup_checks_returns_report(self):
        from utils.startup import run_startup_checks, CheckStatus
        report = run_startup_checks(checks=[])
        assert report.ok_to_start is True
        assert report.has_warnings is False

    def test_failing_check_makes_report_not_ok(self):
        from utils.startup import run_startup_checks, CheckResult, CheckStatus

        def always_fail():
            return CheckResult("test_fail", CheckStatus.FAIL, "deliberate failure")

        report = run_startup_checks(checks=[always_fail])
        assert report.ok_to_start is False

    def test_warning_check_does_not_block_startup(self):
        from utils.startup import run_startup_checks, CheckResult, CheckStatus

        def warn():
            return CheckResult("test_warn", CheckStatus.WARN, "just a warning")

        report = run_startup_checks(checks=[warn])
        assert report.ok_to_start is True
        assert report.has_warnings is True

    def test_pass_check_is_ok(self):
        from utils.startup import run_startup_checks, CheckResult, CheckStatus

        def passing():
            return CheckResult("test_pass", CheckStatus.PASS, "all good")

        report = run_startup_checks(checks=[passing])
        assert report.ok_to_start is True
        assert not report.has_warnings

    def test_exception_in_check_is_treated_as_fail(self):
        from utils.startup import run_startup_checks

        def broken():
            raise RuntimeError("check exploded")

        report = run_startup_checks(checks=[broken])
        assert report.ok_to_start is False

    def test_summary_contains_check_names(self):
        from utils.startup import run_startup_checks, CheckResult, CheckStatus

        def c1():
            return CheckResult("alpha", CheckStatus.PASS, "ok")
        def c2():
            return CheckResult("beta", CheckStatus.WARN, "watch out")

        report = run_startup_checks(checks=[c1, c2])
        summary = report.summary()
        assert "alpha" in summary
        assert "beta" in summary

    def test_config_check_passes_with_valid_settings(self):
        from utils.startup import _check_config_valid, CheckStatus
        result = _check_config_valid()
        assert result.status == CheckStatus.PASS

    def test_directories_check_passes(self):
        from utils.startup import _check_directories_writable, CheckStatus
        result = _check_directories_writable()
        assert result.status == CheckStatus.PASS

    def test_llm_check_warns_when_unreachable(self):
        from utils.startup import _check_llm_reachable, CheckStatus
        import socket
        with patch("socket.create_connection", side_effect=OSError("refused")):
            result = _check_llm_reachable()
        assert result.status == CheckStatus.WARN

    def test_port_check_warns_when_in_use(self):
        from utils.startup import _check_port_free, CheckStatus
        import socket as sock
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.__enter__.return_value.bind.side_effect = OSError("in use")
            result = _check_port_free()
        assert result.status == CheckStatus.WARN


# ---------------------------------------------------------------------------
# Shutdown Manager
# ---------------------------------------------------------------------------

class TestShutdownManager:
    @pytest.fixture(autouse=True)
    def fresh_manager(self):
        from utils.shutdown import ShutdownManager
        self.mgr = ShutdownManager()

    def test_register_and_shutdown_calls_hook(self):
        called = []
        self.mgr.register("test", lambda: called.append(True))
        self.mgr.shutdown()
        assert called == [True]

    def test_shutdown_calls_hooks_in_lifo_order(self):
        order = []
        self.mgr.register("first", lambda: order.append("first"))
        self.mgr.register("second", lambda: order.append("second"))
        self.mgr.register("third", lambda: order.append("third"))
        self.mgr.shutdown()
        assert order == ["third", "second", "first"]

    def test_shutdown_is_idempotent(self):
        called = []
        self.mgr.register("once", lambda: called.append(1))
        self.mgr.shutdown()
        self.mgr.shutdown()
        assert len(called) == 1

    def test_failing_hook_does_not_stop_others(self):
        results = []
        self.mgr.register("good1", lambda: results.append("g1"))
        self.mgr.register("bad",   lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        self.mgr.register("good2", lambda: results.append("g2"))
        self.mgr.shutdown()
        # Both good hooks should run despite the bad one
        assert "g1" in results
        assert "g2" in results

    def test_is_shutdown_flag(self):
        assert self.mgr.is_shutdown is False
        self.mgr.shutdown()
        assert self.mgr.is_shutdown is True

    def test_wait_returns_true_after_shutdown(self):
        def _do_shutdown():
            time.sleep(0.05)
            self.mgr.shutdown()

        t = threading.Thread(target=_do_shutdown)
        t.start()
        result = self.mgr.wait(timeout=2.0)
        t.join()
        assert result is True

    def test_wait_returns_false_on_timeout(self):
        result = self.mgr.wait(timeout=0.05)
        assert result is False

    def test_reset_allows_re_use(self):
        called = []
        self.mgr.register("cb", lambda: called.append(1))
        self.mgr.shutdown()
        self.mgr.reset()
        self.mgr.register("cb2", lambda: called.append(2))
        self.mgr.shutdown()
        assert 1 in called
        assert 2 in called


# ---------------------------------------------------------------------------
# Command Queue
# ---------------------------------------------------------------------------

class TestCommandQueue:
    def _make_engine(self, response="done"):
        from agents.engine import EngineResult
        engine = MagicMock()
        engine.run.return_value = EngineResult(
            response=response, success=True, latency_ms=1.0, steps_taken=1, tool_calls=[]
        )
        return engine

    def test_submit_returns_future(self):
        from utils.queue import CommandQueue
        engine = self._make_engine()
        q = CommandQueue(engine, workers=1)
        q.start()
        try:
            f = q.submit("hello")
            result = f.result(timeout=5)
            assert result.response == "done"
        finally:
            q.stop()

    def test_submit_multiple_commands(self):
        from utils.queue import CommandQueue
        engine = self._make_engine("ok")
        q = CommandQueue(engine, workers=2)
        q.start()
        try:
            futures = [q.submit(f"cmd {i}") for i in range(5)]
            results = [f.result(timeout=5) for f in futures]
            assert all(r.response == "ok" for r in results)
        finally:
            q.stop()

    def test_engine_error_propagates_to_future(self):
        from utils.queue import CommandQueue
        engine = MagicMock()
        engine.run.side_effect = RuntimeError("boom")
        q = CommandQueue(engine, workers=1)
        q.start()
        try:
            f = q.submit("bad command")
            with pytest.raises(RuntimeError, match="boom"):
                f.result(timeout=5)
        finally:
            q.stop()

    def test_queue_full_raises(self):
        from utils.queue import CommandQueue
        engine = self._make_engine()
        q = CommandQueue(engine, workers=0, maxsize=2)  # no workers — queue fills up
        q.submit("cmd1")
        q.submit("cmd2")
        with pytest.raises(queue.Full):
            q.submit("cmd3")  # should raise immediately

    def test_stop_is_idempotent(self):
        from utils.queue import CommandQueue
        engine = self._make_engine()
        q = CommandQueue(engine, workers=1)
        q.start()
        q.stop()
        q.stop()   # should not raise

    def test_qsize_reflects_backlog(self):
        from utils.queue import CommandQueue
        engine = self._make_engine()
        q = CommandQueue(engine, workers=0, maxsize=10)  # no workers
        q.submit("a")
        q.submit("b")
        assert q.qsize == 2

    def test_metrics_recorded(self):
        from utils.queue import CommandQueue
        from utils.metrics import metrics
        metrics.reset()
        engine = self._make_engine()
        q = CommandQueue(engine, workers=1)
        q.start()
        try:
            q.submit("test").result(timeout=5)
        finally:
            q.stop()
        assert metrics.get_counter("queue_submitted_total") >= 1
        assert metrics.get_counter("queue_processing_total") >= 1
