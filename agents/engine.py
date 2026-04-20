"""
Agent Engine — the single entry point for the rest of JARVIS.

    engine = AgentEngine()
    response = engine.run("Open Chrome and search for the weather")

Routing:
  - multi_agent=false  →  ExecutorAgent ReAct loop (single agent, fast)
  - multi_agent=true   →  Planner → Executor → Supervisor pipeline

The engine also injects short-term memory context into every request
(populated in Phase 6 — the memory module is optional here).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.base import AgentResult
from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent, SupervisorVerdict
from config import settings
from utils.logger import get_logger, get_audit_logger

log = get_logger(__name__)
audit = get_audit_logger()


# ---------------------------------------------------------------------------
# Engine result (wraps AgentResult with top-level convenience fields)
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    response: str
    success: bool
    latency_ms: float = 0.0
    steps_taken: int = 0
    tool_calls: list = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent Engine
# ---------------------------------------------------------------------------

class AgentEngine:
    """
    Orchestrates the full agent pipeline for a single user command.

    Thread-safe: each `run()` call is independent. The engine itself
    holds no per-request state — all state lives in the AgentResult objects
    passed between agents.
    """

    def __init__(
        self,
        memory_provider: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Args:
            memory_provider: optional callable that takes the user query
                             and returns a memory context string to inject.
                             Wired up in Phase 6.
        """
        self._memory_provider = memory_provider
        self._multi_agent: bool = settings.features.multi_agent

        # Lazy-init agents to avoid loading the LLM until first use
        self._executor: Optional[ExecutorAgent] = None
        self._planner: Optional[PlannerAgent] = None
        self._supervisor: Optional[SupervisorAgent] = None

        # Load all tools so they are registered before the first agent run
        from tools.loader import load_all_tools
        load_all_tools()

        log.info(
            "AgentEngine initialised",
            extra={"mode": "multi-agent" if self._multi_agent else "single-agent"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> EngineResult:
        """
        Process a user command end-to-end and return a response string.
        This is the only method the voice pipeline (or API) needs to call.
        """
        user_input = user_input.strip()
        if not user_input:
            return EngineResult(response="Please say a command.", success=False)

        from utils.metrics import metrics
        audit.info("User command received", extra={"input": user_input})
        metrics.inc("commands_total")
        t_start = time.perf_counter()

        context = self._build_context(user_input)

        try:
            if self._multi_agent:
                result = self._run_multi_agent(user_input, context)
            else:
                result = self._run_single_agent(user_input, context)
        except Exception as exc:
            log.exception("AgentEngine.run failed")
            metrics.inc("command_errors_total")
            result = AgentResult(
                response="I'm sorry, something went wrong. Please try again.",
                success=False,
                error=str(exc),
            )

        latency_ms = (time.perf_counter() - t_start) * 1000
        metrics.observe("command_latency_ms", latency_ms)
        if not result.success:
            metrics.inc("command_errors_total")

        audit.info(
            "Command processed",
            extra={
                "input": user_input,
                "success": result.success,
                "latency_ms": round(latency_ms, 1),
            },
        )

        return EngineResult(
            response=result.response,
            success=result.success,
            latency_ms=latency_ms,
            steps_taken=result.steps_taken,
            tool_calls=result.tool_calls,
            error=result.error,
            metadata=result.metadata,
        )

    def run_stream(self, user_input: str):
        """
        Stream-friendly variant: yields text chunks as they're generated.
        Falls back to a single yield of the full response if streaming
        is not supported by the current pipeline mode.

        Usage:
            for chunk in engine.run_stream("Tell me a joke"):
                print(chunk, end="", flush=True)
        """
        from core.llm import Message, get_llm_client

        user_input = user_input.strip()
        if not user_input:
            yield "Please say a command."
            return

        context = self._build_context(user_input)
        memory_ctx = context.get("memory_context", "")

        system = (
            "You are JARVIS, an offline AI assistant. "
            "Be concise and helpful."
        )
        if memory_ctx:
            system += f"\n\nRelevant memory:\n{memory_ctx}"

        messages = [
            Message("system", system),
            Message("user", user_input),
        ]

        client = get_llm_client()
        for chunk in client.stream(messages):
            if chunk.content:
                yield chunk.content
            if chunk.done:
                break

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _run_single_agent(
        self, user_input: str, context: Dict[str, Any]
    ) -> AgentResult:
        """Single-agent ReAct loop via ExecutorAgent."""
        executor = self._get_executor()
        return executor.run(user_input, context)

    def _run_multi_agent(
        self, user_input: str, context: Dict[str, Any]
    ) -> AgentResult:
        """Planner → Executor → Supervisor pipeline."""
        # 1. Plan
        planner = self._get_planner()
        plan_result = planner.run(user_input, context)
        steps = plan_result.metadata.get("steps", [])

        if not steps:
            log.warning("Planner returned no steps — falling back to single agent")
            return self._run_single_agent(user_input, context)

        # 2. Execute
        executor = self._get_executor()
        exec_context = {**context, "steps": steps}
        exec_result = executor.run(user_input, exec_context)

        # 3. Supervise (with one retry allowed)
        supervisor = self._get_supervisor()
        step_results = exec_result.metadata.get("step_results", [exec_result.response])
        summary = "\n".join(step_results)
        verdict: SupervisorVerdict = supervisor.validate(user_input, summary)

        if verdict.status == "retry":
            log.info(
                "Supervisor requested retry",
                extra={"reason": verdict.reason, "retry_step": verdict.retry_step},
            )
            # One retry pass — re-run from scratch
            exec_result = executor.run(user_input, exec_context)
            summary = "\n".join(
                exec_result.metadata.get("step_results", [exec_result.response])
            )
            verdict = supervisor.validate(user_input, summary)

        final_response = verdict.response or exec_result.response

        return AgentResult(
            response=final_response,
            tool_calls=exec_result.tool_calls,
            steps_taken=exec_result.steps_taken + 1,   # +1 for planner
            success=verdict.status == "success",
            metadata={
                **exec_result.metadata,
                "plan": plan_result.metadata.get("raw_plan", ""),
                "verdict": verdict.status,
            },
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(self, user_input: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {}

        if self._memory_provider:
            try:
                context["memory_context"] = self._memory_provider(user_input)
            except Exception:
                log.exception("Memory provider failed — continuing without memory context")

        return context

    # ------------------------------------------------------------------
    # Lazy agent accessors
    # ------------------------------------------------------------------

    def _get_executor(self) -> ExecutorAgent:
        if self._executor is None:
            self._executor = ExecutorAgent()
        return self._executor

    def _get_planner(self) -> PlannerAgent:
        if self._planner is None:
            self._planner = PlannerAgent()
        return self._planner

    def _get_supervisor(self) -> SupervisorAgent:
        if self._supervisor is None:
            self._supervisor = SupervisorAgent()
        return self._supervisor
