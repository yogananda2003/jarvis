"""
Executor Agent — runs a plan step by step, calling tools as needed.

Supports two tool-call modes:
1. Native Ollama tool calling (preferred — model returns structured calls)
2. Text extraction fallback (for models that don't support native calls)

Each step feeds its result back into the LLM context so later steps
can build on earlier ones.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.base import AgentResult, BaseAgent, ToolCall
from agents.tool_parser import PlanStep, parse_text_tool_calls
from core.llm import Message
from utils.logger import get_logger

log = get_logger(__name__)

_SYSTEM = """\
You are JARVIS, an AI assistant. To use a tool respond with ONLY this JSON (no other text):
{"name": "tool_name", "arguments": {"param": "value"}}

If no tool is needed, answer directly in plain text.
"""

# Keyword → tool mapping for small-model fallback
_KEYWORD_TOOL_MAP = [
    (["cpu", "ram", "memory", "disk", "system info", "system information", "usage", "performance"], "get_system_info", {}),
    (["process", "processes", "running", "task"], "list_processes", {}),
    (["git status", "git stat"], "git_status", {}),
    (["git log", "commits", "commit history"], "git_log", {"limit": 5}),
    (["git diff", "changes", "what changed"], "git_diff", {}),
]

# Keywords that require argument extraction — handled separately
_OPEN_KEYWORDS = ["open", "launch", "start", "run"]
_TYPE_KEYWORDS = ["type ", "write ", "input "]


class ExecutorAgent(BaseAgent):
    """
    Executes a list of PlanStep objects produced by the PlannerAgent.
    Can also run in single-step mode directly from the AgentEngine.
    """

    def __init__(self, **kwargs) -> None:
        from config import settings
        super().__init__(
            name="ExecutorAgent",
            model=settings.agent.executor_model,
            **kwargs,
        )
        self._max_iterations: int = settings.agent.max_iterations

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    def _execute(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        """
        Single-agent mode: run the full ReAct loop for the user input.
        Used when multi_agent=false (most common case).
        """
        steps = context.get("steps")
        if steps:
            return self._run_plan(user_input, steps, context)
        return self._run_react_loop(user_input, context)

    # ------------------------------------------------------------------
    # Multi-agent mode: run a structured plan
    # ------------------------------------------------------------------

    def _run_plan(
        self,
        original_request: str,
        steps: List[PlanStep],
        context: Dict[str, Any],
    ) -> AgentResult:
        """Execute a structured plan from the PlannerAgent."""
        tool_calls_made: List[ToolCall] = []
        step_results: List[str] = []
        conversation: List[Message] = [
            Message("system", _SYSTEM),
            Message(
                "user",
                f"Original request: {original_request}\n\n"
                f"Execute the following plan step by step.",
            ),
        ]

        for plan_step in steps:
            result_text = self._run_step(
                plan_step, conversation, step_results, tool_calls_made
            )
            step_results.append(f"Step {plan_step.step}: {result_text}")

        final = self._summarise(original_request, step_results, conversation)

        return AgentResult(
            response=final,
            tool_calls=tool_calls_made,
            steps_taken=len(steps),
            success=True,
            metadata={"step_results": step_results},
        )

    def _run_step(
        self,
        step: PlanStep,
        conversation: List[Message],
        prior_results: List[str],
        tool_calls_out: List[ToolCall],
    ) -> str:
        """Execute one plan step, returning its text result."""
        prior_ctx = "\n".join(prior_results) if prior_results else "None yet."
        prompt = (
            f"Step {step.step}: {step.description}\n"
            f"Tool to use: {step.tool or 'none'}\n"
            f"Arguments hint: {json.dumps(step.args) if step.args else '{}'}\n"
            f"Prior results:\n{prior_ctx}"
        )
        conversation.append(Message("user", prompt))

        tools = self._build_tool_schemas() if step.tool else None
        response_text = self._chat(conversation, tools=tools)
        conversation.append(Message("assistant", response_text))

        # Try tool call extraction
        tool_calls, remainder = parse_text_tool_calls(response_text)

        if not tool_calls and step.tool:
            # Build call from plan hint
            tool_calls = [ToolCall(name=step.tool, arguments=step.args or {})]

        result = response_text
        for tc in tool_calls:
            try:
                tool_result = self._execute_tool(tc)
                tool_calls_out.append(tc)
                result = str(tool_result)
                conversation.append(
                    Message("user", f"Tool '{tc.name}' returned: {result}")
                )
            except Exception as exc:
                result = f"Error running {tc.name}: {exc}"
                log.error("Step tool failed", extra={"tool": tc.name, "error": str(exc)})

        return result

    # ------------------------------------------------------------------
    # Single-agent ReAct loop
    # ------------------------------------------------------------------

    def _run_react_loop(
        self,
        user_input: str,
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        Reason-Act loop. Compact prompt for small models + keyword fallback
        when the LLM fails to produce a tool call.
        """
        # Fast path: deterministic keyword dispatch BEFORE calling the LLM.
        # Handles "open X", "launch X", "type X", etc. instantly and reliably.
        fast_path = self._keyword_fallback(user_input)
        if fast_path:
            tool_calls_made: List[ToolCall] = []
            for tc in fast_path:
                try:
                    self._execute_tool(tc)
                    tool_calls_made.append(tc)
                    log.info("Fast-path tool executed", extra={"tool": tc.name})
                except Exception as exc:
                    log.error("Fast-path tool failed", extra={"tool": tc.name, "error": str(exc)})
            if tool_calls_made:
                last = tool_calls_made[-1]
                response = str(last.result) if last.result is not None else f"Done: {last.name}"
                return AgentResult(
                    response=response,
                    tool_calls=tool_calls_made,
                    steps_taken=1,
                    success=True,
                )

        tool_schemas = self._build_tool_schemas()
        memory_context = context.get("memory_context", "")

        # Compact tool list: just names + one-line descriptions
        tool_summary = "\n".join(
            f"- {t['name']}: {t['description'][:80]}"
            for t in tool_schemas
        )

        system_prompt = _SYSTEM + f"\nAvailable tools:\n{tool_summary}"
        if memory_context:
            system_prompt += f"\n\nMemory:\n{memory_context}"

        conversation: List[Message] = [
            Message("system", system_prompt),
            Message("user", user_input),
        ]

        tool_calls_made: List[ToolCall] = []
        iterations = 0

        while iterations < self._max_iterations:
            iterations += 1
            response_text = self._chat(conversation)
            conversation.append(Message("assistant", response_text))

            tool_calls, _ = parse_text_tool_calls(response_text)
            used_fallback = False

            # If LLM returned no tool call, try keyword dispatch
            if not tool_calls:
                fallback = self._keyword_fallback(user_input)
                if fallback:
                    tool_calls = fallback
                    used_fallback = True
                    log.info("Keyword fallback", extra={"tool": tool_calls[0].name})

            if not tool_calls:
                # LLM gave a direct answer
                final = response_text.strip()
                if not final and tool_calls_made:
                    final = self._format_tool_results(user_input, tool_calls_made)
                return AgentResult(
                    response=final,
                    tool_calls=tool_calls_made,
                    steps_taken=iterations,
                    success=bool(final),
                )

            # Execute tools
            for tc in tool_calls:
                try:
                    result = self._execute_tool(tc)
                    tool_calls_made.append(tc)
                    log.info("Tool executed", extra={"tool": tc.name, "result": str(result)[:120]})
                except Exception as exc:
                    log.error("Tool failed", extra={"tool": tc.name, "error": str(exc)})

            # Keyword fallback: format result directly — don't trust small model to summarise
            if used_fallback:
                return AgentResult(
                    response=self._format_tool_results(user_input, tool_calls_made),
                    tool_calls=tool_calls_made,
                    steps_taken=iterations,
                    success=bool(tool_calls_made),
                )

            # LLM-driven tool call: feed result back and ask for summary
            observation = "\n".join(
                f"Tool '{tc.name}' returned: {json.dumps(tc.result, default=str)}"
                for tc in tool_calls_made[-len(tool_calls):]
            )
            conversation.append(Message("user", observation))
            conversation.append(Message(
                "user",
                "Using only the tool result above, write a concise answer for the user."
            ))

        log.warning("ReAct loop hit max iterations", extra={"max": self._max_iterations})
        final = self._format_tool_results(user_input, tool_calls_made) if tool_calls_made else ""
        return AgentResult(
            response=final,
            tool_calls=tool_calls_made,
            steps_taken=iterations,
            success=False,
            error="Max iterations reached",
        )

    def _keyword_fallback(self, user_input: str) -> List[ToolCall]:
        """Map common question patterns to tools when the LLM fails."""
        text = user_input.lower().strip()

        # "open/launch/start <app>" → open_app
        for kw in _OPEN_KEYWORDS:
            if text.startswith(kw + " "):
                app_name = user_input[len(kw):].strip().rstrip(".")
                if app_name and self._registry.get("open_app"):
                    return [ToolCall(name="open_app", arguments={"app_name": app_name})]

        # "type/write/input <text>" → type_text
        for kw in _TYPE_KEYWORDS:
            if text.startswith(kw):
                content = user_input[len(kw):].strip().strip('"').strip("'")
                if content and self._registry.get("type_text"):
                    return [ToolCall(name="type_text", arguments={"text": content})]

        # Static keyword map
        for keywords, tool_name, args in _KEYWORD_TOOL_MAP:
            if any(kw in text for kw in keywords):
                if self._registry.get(tool_name):
                    return [ToolCall(name=tool_name, arguments=args)]
        return []

    def _format_tool_results(self, user_input: str, tool_calls: List[ToolCall]) -> str:
        """Build a plain-text answer from tool call results when the LLM returns empty."""
        if not tool_calls:
            return ""
        last = tool_calls[-1]
        data = last.result if last.result is not None else last.arguments
        # Ask the LLM one more time with just the result — very short prompt
        try:
            resp = self._chat([
                Message("system", "You are JARVIS. Answer concisely in 2-3 sentences."),
                Message("user", f"Question: {user_input}"),
                Message("user", f"Data from {last.name}: {json.dumps(data, default=str)}"),
            ])
            if resp.strip():
                return resp.strip()
        except Exception:
            pass
        # Hard fallback: format the raw result directly
        if isinstance(data, dict):
            lines = [f"{k}: {v}" for k, v in data.items()]
            return f"Result from {last.name}:\n" + "\n".join(lines)
        return f"Result from {last.name}: {data}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarise(
        self,
        original_request: str,
        step_results: List[str],
        conversation: List[Message],
    ) -> str:
        """Ask the LLM to write a final user-facing response from step results."""
        summary_prompt = (
            f"Original request: {original_request}\n\n"
            f"Completed steps:\n" + "\n".join(step_results) + "\n\n"
            "Write a concise final response for the user summarising what was done."
        )
        conversation.append(Message("user", summary_prompt))
        return self._chat(conversation).strip()
