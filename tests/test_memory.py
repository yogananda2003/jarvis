"""
Tests for the Memory System (Phase 6).

Short-term: in-memory rolling window.
Long-term:  SQLite-backed (uses a temp DB per test).
Manager:    combines both + memory_provider callback.

Run:
    pytest tests/test_memory.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.short_term import ShortTermMemory, MemoryMessage
from memory.long_term import LongTermMemory
from memory.manager import MemoryManager


# ---------------------------------------------------------------------------
# Short-Term Memory
# ---------------------------------------------------------------------------

class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_messages=10)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].content == "Hi there"

    def test_rolling_eviction(self):
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add("user", f"msg {i}")
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0].content == "msg 2"
        assert msgs[-1].content == "msg 4"

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add("user", "something")
        mem.clear()
        assert len(mem) == 0

    def test_get_context_empty(self):
        mem = ShortTermMemory()
        assert mem.get_context() == ""

    def test_get_context_format(self):
        mem = ShortTermMemory()
        mem.add("user", "open chrome")
        mem.add("assistant", "Opening Chrome")
        ctx = mem.get_context()
        assert "User: open chrome" in ctx
        assert "Assistant: Opening Chrome" in ctx

    def test_get_context_truncation(self):
        mem = ShortTermMemory()
        # Add a very long message
        mem.add("user", "x" * 3000)
        ctx = mem.get_context(max_chars=500)
        assert len(ctx) <= 503   # "..." prefix adds 3 chars
        assert ctx.startswith("...")

    def test_len(self):
        mem = ShortTermMemory(max_messages=5)
        assert len(mem) == 0
        mem.add("user", "a")
        mem.add("user", "b")
        assert len(mem) == 2

    def test_thread_safety(self):
        import threading
        mem = ShortTermMemory(max_messages=100)
        threads = [
            threading.Thread(target=mem.add, args=("user", f"msg {i}"))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(mem) == 50


# ---------------------------------------------------------------------------
# Long-Term Memory
# ---------------------------------------------------------------------------

class TestLongTermMemory:
    @pytest.fixture
    def mem(self, tmp_path):
        db = tmp_path / "test_memory.db"
        return LongTermMemory(db_path=db)

    def test_save_and_get_recent(self, mem):
        mem.save("user", "Hello long term")
        mem.save("assistant", "Remembered")
        rows = mem.get_recent(limit=10)
        assert len(rows) == 2
        assert rows[0]["content"] == "Hello long term"
        assert rows[1]["role"] == "assistant"

    def test_get_recent_limit(self, mem):
        for i in range(10):
            mem.save("user", f"message {i}")
        rows = mem.get_recent(limit=3)
        assert len(rows) == 3

    def test_get_recent_by_session(self, mem):
        mem.save("user", "session A", session_id="A")
        mem.save("user", "session B", session_id="B")
        rows = mem.get_recent(session_id="A")
        assert len(rows) == 1
        assert rows[0]["content"] == "session A"

    def test_search_keyword(self, mem):
        mem.save("user", "open chrome browser")
        mem.save("user", "play some music")
        mem.save("assistant", "Opening Chrome now")
        results = mem.search("chrome")
        assert len(results) == 2
        assert all("chrome" in r["content"].lower() for r in results)

    def test_search_multi_word(self, mem):
        mem.save("user", "open chrome browser please")
        mem.save("user", "open firefox")
        results = mem.search("open chrome")
        assert len(results) == 1
        assert "chrome" in results[0]["content"]

    def test_search_empty_query(self, mem):
        mem.save("user", "something")
        assert mem.search("") == []
        assert mem.search("   ") == []

    def test_search_no_results(self, mem):
        mem.save("user", "hello world")
        assert mem.search("nonexistent_xyz") == []

    def test_delete_session(self, mem):
        mem.save("user", "keep", session_id="keep")
        mem.save("user", "delete me", session_id="gone")
        deleted = mem.delete_session("gone")
        assert deleted == 1
        remaining = mem.get_recent()
        assert all(r["content"] != "delete me" for r in remaining)

    def test_facts_set_get(self, mem):
        mem.set_fact("user_name", "Alice")
        assert mem.get_fact("user_name") == "Alice"

    def test_facts_upsert(self, mem):
        mem.set_fact("user_name", "Alice")
        mem.set_fact("user_name", "Bob")
        assert mem.get_fact("user_name") == "Bob"

    def test_facts_get_missing(self, mem):
        assert mem.get_fact("nonexistent") is None

    def test_facts_delete(self, mem):
        mem.set_fact("temp", "value")
        assert mem.delete_fact("temp") is True
        assert mem.get_fact("temp") is None
        assert mem.delete_fact("temp") is False

    def test_list_facts(self, mem):
        mem.set_fact("a", "1")
        mem.set_fact("b", "2")
        facts = mem.list_facts()
        keys = [f["key"] for f in facts]
        assert "a" in keys
        assert "b" in keys

    def test_stats(self, mem):
        mem.save("user", "hi")
        mem.set_fact("x", "y")
        s = mem.stats()
        assert s["conversations"] == 1
        assert s["facts"] == 1

    def test_persistence_across_instances(self, tmp_path):
        db = tmp_path / "persist.db"
        m1 = LongTermMemory(db_path=db)
        m1.save("user", "remember this")
        m1.set_fact("lang", "python")

        m2 = LongTermMemory(db_path=db)
        rows = m2.get_recent()
        assert any(r["content"] == "remember this" for r in rows)
        assert m2.get_fact("lang") == "python"


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------

class TestMemoryManager:
    @pytest.fixture
    def manager(self, tmp_path):
        db = tmp_path / "mgr.db"
        from unittest.mock import patch
        with patch("memory.long_term.settings") as mock_s:
            mock_s.long_term_db_path = db
            mock_s.features.long_term_memory = True
        mgr = MemoryManager(session_id="test-session", enable_long_term=True)
        # Patch the db path after construction
        mgr.long_term = LongTermMemory(db_path=db)
        return mgr

    def test_record_user_adds_to_short_term(self, manager):
        manager.record_user("hello")
        assert len(manager.short_term) == 1
        assert manager.short_term.get_messages()[0].content == "hello"

    def test_record_assistant_adds_to_short_term(self, manager):
        manager.record_assistant("hi there")
        msgs = manager.short_term.get_messages()
        assert msgs[0].role == "assistant"

    def test_record_persists_to_long_term(self, manager):
        manager.record_user("test message")
        rows = manager.long_term.get_recent()
        assert any(r["content"] == "test message" for r in rows)

    def test_memory_provider_returns_string(self, manager):
        manager.record_user("what's on my schedule")
        manager.record_assistant("Nothing scheduled today")
        ctx = manager.memory_provider("schedule")
        assert isinstance(ctx, str)

    def test_memory_provider_empty_when_no_history(self, manager):
        ctx = manager.memory_provider("anything")
        assert ctx == ""

    def test_memory_provider_includes_short_term(self, manager):
        manager.record_user("open chrome")
        ctx = manager.memory_provider("chrome")
        assert "open chrome" in ctx

    def test_memory_provider_includes_long_term_hits(self, manager):
        manager.long_term.save("user", "I prefer dark mode")
        ctx = manager.memory_provider("dark mode")
        assert "dark mode" in ctx

    def test_set_and_get_fact(self, manager):
        manager.set_fact("favorite_color", "blue")
        assert manager.get_fact("favorite_color") == "blue"

    def test_clear_session(self, manager):
        manager.record_user("temp")
        manager.clear_session()
        assert len(manager.short_term) == 0

    def test_stats(self, manager):
        manager.record_user("hi")
        s = manager.stats()
        assert "short_term_messages" in s
        assert s["short_term_messages"] == 1

    def test_no_long_term_mode(self, tmp_path):
        mgr = MemoryManager(enable_long_term=False)
        assert mgr.long_term is None
        mgr.record_user("test")   # should not raise
        assert mgr.get_fact("x") is None   # returns None gracefully
        ctx = mgr.memory_provider("test")
        assert "test" in ctx   # short-term still works


# ---------------------------------------------------------------------------
# Engine integration: memory_provider callback
# ---------------------------------------------------------------------------

class TestEngineMemoryIntegration:
    def test_engine_accepts_memory_provider(self):
        """AgentEngine should accept and store a memory_provider callable."""
        from agents.engine import AgentEngine

        provider_calls = []

        def fake_provider(query: str) -> str:
            provider_calls.append(query)
            return f"Memory context for: {query}"

        engine = AgentEngine(memory_provider=fake_provider)
        assert engine._memory_provider is fake_provider

    def test_engine_injects_memory_context(self):
        """_build_context should call memory_provider and store result."""
        from agents.engine import AgentEngine

        def provider(q):
            return "relevant memory"

        engine = AgentEngine(memory_provider=provider)
        ctx = engine._build_context("test query")
        assert ctx.get("memory_context") == "relevant memory"

    def test_engine_handles_memory_provider_failure(self):
        """If memory_provider raises, engine should continue without memory."""
        from agents.engine import AgentEngine

        def bad_provider(q):
            raise RuntimeError("DB down")

        engine = AgentEngine(memory_provider=bad_provider)
        ctx = engine._build_context("test")
        assert "memory_context" not in ctx
