"""
Base class for all JARVIS agents.

Every agent shares:
- An LLM client
- A reference to the tool registry
- A structured `run()` method that subclasses implement
- Logging and timing built-in
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm import Message, OllamaClient, get_llm_client
from tools.registry import ToolRegistry, registry as global_registry
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A tool invocation requested by an agent."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = field(default=None, repr=False)


@dataclass
class AgentResult:
    """The outcome of an agent.run() call."""
    response: str                           # final text response for the user
    tool_calls: List[ToolCall] = field(default_factory=list)
    steps_taken: int = 0
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    All JARVIS agents inherit from this class.

    Subclasses must implement `_execute()`.
    `run()` wraps `_execute()` with timing, logging, and error handling.
    """

    def __init__(
        self,
        name: str,
        llm: Optional[OllamaClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        model: Optional[str] = None,
    ) -> None:
        self.name = name
        self._llm = llm or get_llm_client()
        self._registry = tool_registry or global_registry
        self._model = model   # None = use client default

        log.debug("Agent created", extra={"agent": self.name})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute the agent. Wraps `_execute()` with timing and error handling.
        """
        t_start = time.perf_counter()
        log.info(
            "Agent run start",
            extra={"agent": self.name, "input": user_input[:80]},
        )

        try:
            result = self._execute(user_input, context or {})
        except Exception as exc:
            log.exception("Agent run failed", extra={"agent": self.name})
            result = AgentResult(
                response="I encountered an error and could not complete that task.",
                success=False,
                error=str(exc),
            )

        result.latency_ms = (time.perf_counter() - t_start) * 1000

        log.info(
            "Agent run complete",
            extra={
                "agent": self.name,
                "success": result.success,
                "steps": result.steps_taken,
                "latency_ms": round(result.latency_ms, 1),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _execute(
        self, user_input: str, context: Dict[str, Any]
    ) -> AgentResult:
        """Implement the agent's core logic here."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all agents
    # ------------------------------------------------------------------

    def _chat(self, messages: List[Message], tools: Optional[List[dict]] = None) -> str:
        """Send messages to the LLM and return the text response."""
        response = self._llm.chat(messages, model=self._model, tools=tools)
        return response.content

    def _build_tool_schemas(self) -> List[dict]:
        """Return OpenAI-style schemas for all enabled tools."""
        return self._registry.all_schemas()

    def _execute_tool(self, call: ToolCall) -> Any:
        """Look up and run a tool, logging the action."""
        from utils.logger import get_audit_logger
        audit = get_audit_logger()

        audit.info(
            "Tool execution",
            extra={
                "agent": self.name,
                "tool": call.name,
                "tool_args": call.arguments,
            },
        )
        log.debug("Executing tool", extra={"tool": call.name, "args": call.arguments})

        try:
            result = self._registry.execute(call.name, **call.arguments)
            call.result = result   # store on ToolCall for downstream use
            log.debug("Tool result", extra={"tool": call.name, "result": str(result)[:200]})
            return result
        except Exception as exc:
            log.error(
                "Tool execution failed",
                extra={"tool": call.name, "error": str(exc)},
            )
            raise
