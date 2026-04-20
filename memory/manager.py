"""
Memory Manager — combines short-term and long-term memory into a single interface.

The `memory_provider` callable is wired into AgentEngine so every request
automatically receives relevant memory context in the system prompt.
"""

from __future__ import annotations

from typing import List, Optional

from config import settings
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from utils.logger import get_logger

log = get_logger(__name__)


class MemoryManager:
    """
    Unified facade for short-term (session) and long-term (SQLite) memory.

    Typical lifecycle:
        manager = MemoryManager(session_id="abc123")
        engine  = AgentEngine(memory_provider=manager.memory_provider)

        manager.record_user("open chrome")
        manager.record_assistant("Opening Chrome...")
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        enable_long_term: Optional[bool] = None,
    ) -> None:
        self._session_id = session_id or "default"
        self._use_long_term = (
            enable_long_term
            if enable_long_term is not None
            else settings.features.long_term_memory
        )

        self.short_term = ShortTermMemory()
        self.long_term: Optional[LongTermMemory] = (
            LongTermMemory() if self._use_long_term else None
        )

        log.info(
            "MemoryManager ready",
            extra={
                "session": self._session_id,
                "long_term": self._use_long_term,
            },
        )

    # ------------------------------------------------------------------
    # Recording turns
    # ------------------------------------------------------------------

    def record_user(self, text: str) -> None:
        self.short_term.add("user", text)
        if self.long_term:
            self.long_term.save("user", text, session_id=self._session_id)

    def record_assistant(self, text: str) -> None:
        self.short_term.add("assistant", text)
        if self.long_term:
            self.long_term.save("assistant", text, session_id=self._session_id)

    # ------------------------------------------------------------------
    # Context retrieval (called by AgentEngine)
    # ------------------------------------------------------------------

    def memory_provider(self, query: str) -> str:
        """
        Build a compact memory context string for injection into the system prompt.
        Includes:
          1. Recent short-term exchanges (last N messages).
          2. Relevant long-term hits for the current query (if enabled).
        """
        parts: List[str] = []

        short_ctx = self.short_term.get_context()
        if short_ctx:
            parts.append(f"[Recent conversation]\n{short_ctx}")

        if self.long_term and query.strip():
            hits = self.long_term.search(query, limit=3)
            if hits:
                relevant = "\n".join(
                    f"{h['role'].capitalize()}: {h['content']}" for h in hits
                )
                parts.append(f"[Relevant past interactions]\n{relevant}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Facts delegation
    # ------------------------------------------------------------------

    def set_fact(self, key: str, value: str) -> None:
        if self.long_term:
            self.long_term.set_fact(key, value)

    def get_fact(self, key: str) -> Optional[str]:
        if self.long_term:
            return self.long_term.get_fact(key)
        return None

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def clear_session(self) -> None:
        """Clear short-term memory and optionally remove session from long-term."""
        self.short_term.clear()
        if self.long_term:
            self.long_term.delete_session(self._session_id)

    def stats(self) -> dict:
        result = {"short_term_messages": len(self.short_term)}
        if self.long_term:
            result.update(self.long_term.stats())
        return result
