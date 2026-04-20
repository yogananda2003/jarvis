"""
Short-term (session) memory — an in-memory rolling window of messages.

Thread-safe. Cleared when the session ends (process restart).
Max size controlled by config.memory.short_term.max_messages.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import settings


@dataclass
class MemoryMessage:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)


class ShortTermMemory:
    """
    Rolling window of recent messages for the current session.

    Usage:
        mem = ShortTermMemory()
        mem.add("user", "What time is it?")
        mem.add("assistant", "It is 3 PM.")
        context = mem.get_context()   # formatted string injected into prompts
    """

    def __init__(self, max_messages: Optional[int] = None) -> None:
        self._max = max_messages or settings.memory.short_term.max_messages
        self._messages: List[MemoryMessage] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a message, evicting the oldest if over capacity."""
        with self._lock:
            self._messages.append(MemoryMessage(role=role, content=content))
            if len(self._messages) > self._max:
                self._messages = self._messages[-self._max:]

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_messages(self) -> List[MemoryMessage]:
        with self._lock:
            return list(self._messages)

    def get_context(self, max_chars: int = 2000) -> str:
        """
        Return a compact text block of recent exchanges suitable for
        injection into a system prompt.  Truncates from the front if
        the total character budget is exceeded.
        """
        with self._lock:
            messages = list(self._messages)

        if not messages:
            return ""

        lines: List[str] = []
        for msg in messages:
            prefix = msg.role.capitalize()
            lines.append(f"{prefix}: {msg.content}")

        block = "\n".join(lines)
        if len(block) > max_chars:
            block = "..." + block[-max_chars:]
        return block

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)
