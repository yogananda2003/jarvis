"""
Tests for the API Layer (Phase 7).

Uses FastAPI's TestClient — no real server needed, no real LLM called.
Memory manager is injected with a real (in-memory / tmp-db) instance.

Run:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import create_app
from memory.manager import MemoryManager
from memory.long_term import LongTermMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_manager(tmp_path):
    mgr = MemoryManager(session_id="test", enable_long_term=True)
    mgr.long_term = LongTermMemory(db_path=tmp_path / "test.db")
    return mgr


@pytest.fixture
def client(mem_manager):
    app = create_app(memory_manager=mem_manager)
    return TestClient(app)


@pytest.fixture
def client_no_memory():
    app = create_app(memory_manager=None)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "model" in body

    def test_status_returns_fields(self, client):
        with patch("api.routes.health.get_llm_client") as mock_llm:
            mock_llm.return_value.health_check.return_value = True
            r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert "llm_healthy" in body
        assert "uptime_s" in body
        assert "memory_stats" in body

    def test_status_degraded_when_llm_down(self, client):
        with patch("api.routes.health.get_llm_client") as mock_llm:
            mock_llm.return_value.health_check.return_value = False
            r = client.get("/status")
        assert r.json()["status"] == "degraded"
        assert r.json()["llm_healthy"] is False

    def test_status_no_memory_manager(self, client_no_memory):
        with patch("api.routes.health.get_llm_client") as mock_llm:
            mock_llm.return_value.health_check.return_value = True
            r = client_no_memory.get("/status")
        assert r.status_code == 200
        assert r.json()["memory_stats"] == {}


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

class TestCommandEndpoint:
    def _mock_engine_result(self, response="Done", success=True, latency=123.0):
        from agents.engine import EngineResult
        result = EngineResult(
            response=response,
            success=success,
            latency_ms=latency,
            steps_taken=1,
            tool_calls=[],
        )
        return result

    def test_command_success(self, client):
        with patch("api.routes.command._get_engine") as mock_get:
            engine = MagicMock()
            engine.run.return_value = self._mock_engine_result("Hello!")
            mock_get.return_value = engine

            r = client.post("/command", json={"input": "say hello"})

        assert r.status_code == 200
        body = r.json()
        assert body["response"] == "Hello!"
        assert body["success"] is True
        assert body["latency_ms"] == 123.0

    def test_command_records_memory(self, client, mem_manager):
        with patch("api.routes.command._get_engine") as mock_get:
            engine = MagicMock()
            engine.run.return_value = self._mock_engine_result("Recorded")
            mock_get.return_value = engine

            client.post("/command", json={"input": "test memory recording"})

        msgs = mem_manager.short_term.get_messages()
        assert any(m.content == "test memory recording" for m in msgs)
        assert any(m.content == "Recorded" for m in msgs)

    def test_command_empty_input_rejected(self, client):
        r = client.post("/command", json={"input": ""})
        assert r.status_code == 422   # Pydantic validation error

    def test_command_missing_input_rejected(self, client):
        r = client.post("/command", json={})
        assert r.status_code == 422

    def test_command_includes_tool_calls(self, client):
        from agents.engine import EngineResult
        from agents.base import ToolCall

        tc = ToolCall(name="open_app", arguments={"app_name": "chrome"})
        result = EngineResult(
            response="Chrome opened",
            success=True,
            latency_ms=200.0,
            steps_taken=1,
            tool_calls=[tc],
        )
        with patch("api.routes.command._get_engine") as mock_get:
            engine = MagicMock()
            engine.run.return_value = result
            mock_get.return_value = engine
            r = client.post("/command", json={"input": "open chrome"})

        assert r.status_code == 200
        calls = r.json()["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["name"] == "open_app"

    def test_command_engine_error_returns_500(self, client):
        with patch("api.routes.command._get_engine") as mock_get:
            engine = MagicMock()
            engine.run.side_effect = RuntimeError("LLM exploded")
            mock_get.return_value = engine
            r = client.post("/command", json={"input": "crash test"})
        assert r.status_code == 500

    def test_command_stream_returns_event_stream(self, client):
        with patch("api.routes.command._get_engine") as mock_get:
            engine = MagicMock()
            engine.run_stream.return_value = iter(["Hello ", "world"])
            mock_get.return_value = engine
            r = client.post("/command", json={"input": "hi", "stream": True})
        assert "text/event-stream" in r.headers["content-type"]
        body = r.text
        assert "Hello " in body
        assert "[DONE]" in body


# ---------------------------------------------------------------------------
# Memory — conversations
# ---------------------------------------------------------------------------

class TestMemoryConversationEndpoints:
    def test_list_memory_empty(self, client):
        r = client.get("/memory?session_id=test")
        assert r.status_code == 200
        assert r.json()["messages"] == []

    def test_list_memory_after_command(self, client, mem_manager):
        mem_manager.record_user("hi")
        mem_manager.record_assistant("hello")
        r = client.get("/memory?session_id=test")
        assert r.status_code == 200
        msgs = r.json()["messages"]
        assert len(msgs) == 2

    def test_list_memory_limit(self, client, mem_manager):
        for i in range(10):
            mem_manager.record_user(f"msg {i}")
        r = client.get("/memory?session_id=test&limit=3")
        assert r.status_code == 200
        assert len(r.json()["messages"]) == 3

    def test_search_memory(self, client, mem_manager):
        mem_manager.long_term.save("user", "open chrome browser", session_id="test")
        mem_manager.long_term.save("user", "play music", session_id="test")
        r = client.get("/memory/search?q=chrome")
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) >= 1
        assert any("chrome" in res["content"] for res in results)

    def test_search_memory_no_results(self, client):
        r = client.get("/memory/search?q=xyzzy_no_match")
        assert r.status_code == 200
        assert r.json()["results"] == []

    def test_clear_session(self, client, mem_manager):
        mem_manager.record_user("temp message")
        r = client.delete("/memory/session/test")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        assert len(mem_manager.short_term) == 0

    def test_memory_without_manager_returns_503(self, client_no_memory):
        r = client_no_memory.get("/memory")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Memory — facts
# ---------------------------------------------------------------------------

class TestMemoryFactsEndpoints:
    def test_set_and_get_fact(self, client):
        r = client.post("/memory/facts", json={"key": "username", "value": "Alice"})
        assert r.status_code == 201
        assert r.json()["key"] == "username"

        r2 = client.get("/memory/facts/username")
        assert r2.status_code == 200
        assert r2.json()["value"] == "Alice"

    def test_get_nonexistent_fact_returns_404(self, client):
        r = client.get("/memory/facts/does_not_exist")
        assert r.status_code == 404

    def test_list_facts(self, client):
        client.post("/memory/facts", json={"key": "color", "value": "blue"})
        client.post("/memory/facts", json={"key": "lang", "value": "python"})
        r = client.get("/memory/facts")
        assert r.status_code == 200
        keys = [f["key"] for f in r.json()["facts"]]
        assert "color" in keys
        assert "lang" in keys

    def test_delete_fact(self, client):
        client.post("/memory/facts", json={"key": "tmp", "value": "x"})
        r = client.delete("/memory/facts/tmp")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        r2 = client.get("/memory/facts/tmp")
        assert r2.status_code == 404

    def test_delete_nonexistent_fact_returns_404(self, client):
        r = client.delete("/memory/facts/ghost")
        assert r.status_code == 404

    def test_set_fact_empty_key_rejected(self, client):
        r = client.post("/memory/facts", json={"key": "", "value": "val"})
        assert r.status_code == 422

    def test_facts_without_manager_returns_503(self, client_no_memory):
        r = client_no_memory.get("/memory/facts")
        assert r.status_code == 503
