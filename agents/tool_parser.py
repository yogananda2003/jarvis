"""
Tool call extraction from LLM responses.

Ollama supports native tool calling for some models (returns structured
`tool_calls` in the response). For models that don't support it, we fall
back to extracting JSON from the response text.

This module centralises both strategies so the rest of the agent code
never has to think about parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.base import ToolCall
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Native tool call parsing (Ollama SDK format)
# ---------------------------------------------------------------------------

def parse_native_tool_calls(response_obj: Any) -> List[ToolCall]:
    """
    Extract tool calls from an Ollama ChatResponse that supports native
    function calling (message.tool_calls is populated).
    """
    calls: List[ToolCall] = []
    tool_calls = getattr(getattr(response_obj, "message", None), "tool_calls", None)
    if not tool_calls:
        return calls

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", "")
        args = getattr(fn, "arguments", {}) or {}
        if name:
            calls.append(ToolCall(name=name, arguments=dict(args)))
            log.debug("Native tool call parsed", extra={"tool": name, "args": args})

    return calls


# ---------------------------------------------------------------------------
# Text-based tool call extraction (fallback for non-native models)
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_INLINE_JSON = re.compile(r"\{[^{}]*\"tool\"[^{}]*\}", re.DOTALL)
_TOOL_CALL_SCHEMA = re.compile(
    r"\{[^{}]*\"name\"\s*:\s*\"(?P<name>[^\"]+)\"[^{}]*\"arguments\"\s*:\s*(?P<args>\{[^}]*\})[^{}]*\}",
    re.DOTALL,
)


def parse_text_tool_calls(text: str) -> Tuple[List[ToolCall], str]:
    """
    Try to extract tool call JSON from plain LLM text.

    Returns (tool_calls, cleaned_text) where cleaned_text has the JSON block removed.

    Expected format in LLM response:
        ```json
        {"name": "open_app", "arguments": {"app_name": "Chrome"}}
        ```
    or inline:
        {"name": "open_app", "arguments": {"app_name": "Chrome"}}
    """
    calls: List[ToolCall] = []
    remaining = text

    # Try fenced code blocks first (most explicit)
    for match in _JSON_BLOCK.finditer(text):
        raw = match.group(1)
        call = _try_parse_tool_json(raw)
        if call:
            calls.append(call)
            remaining = remaining.replace(match.group(0), "").strip()

    if calls:
        return calls, remaining

    # Try named-field pattern
    for match in _TOOL_CALL_SCHEMA.finditer(text):
        name = match.group("name")
        args_str = match.group("args")
        try:
            args = json.loads(args_str)
            calls.append(ToolCall(name=name, arguments=args))
            remaining = remaining.replace(match.group(0), "").strip()
            log.debug("Text tool call parsed", extra={"tool": name})
        except json.JSONDecodeError:
            pass

    return calls, remaining


def _try_parse_tool_json(raw: str) -> Optional[ToolCall]:
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    # Support both {"name": ..., "arguments": ...} and {"tool": ..., "args": ...}
    name = data.get("name") or data.get("tool")
    args = data.get("arguments") or data.get("args") or data.get("parameters") or {}

    if not name:
        return None

    return ToolCall(name=str(name), arguments=dict(args) if isinstance(args, dict) else {})


# ---------------------------------------------------------------------------
# Plan parsing (for PlannerAgent)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class PlanStep:
    step: int
    description: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)


def parse_plan(text: str) -> List[PlanStep]:
    """
    Extract a structured plan from the planner LLM response.

    Expected JSON in the response:
        {"plan": [{"step": 1, "description": "...", "tool": "...", "args": {...}}]}
    """
    # Try fenced JSON block
    for match in _JSON_BLOCK.finditer(text):
        steps = _try_parse_plan_json(match.group(1))
        if steps is not None:
            return steps

    # Try raw JSON anywhere in the response
    start = text.find('{"plan"')
    if start != -1:
        # Find the matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    steps = _try_parse_plan_json(text[start : i + 1])
                    if steps is not None:
                        return steps
                    break

    log.warning("Could not parse plan from LLM response — treating as single step")
    return [PlanStep(step=1, description=text.strip(), tool=None, args={})]


def _try_parse_plan_json(raw: str) -> Optional[List[PlanStep]]:
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    raw_steps = data.get("plan", [])
    if not isinstance(raw_steps, list):
        return None

    steps = []
    for s in raw_steps:
        steps.append(
            PlanStep(
                step=int(s.get("step", len(steps) + 1)),
                description=str(s.get("description", "")),
                tool=s.get("tool") or None,
                args=s.get("args") or s.get("arguments") or {},
            )
        )
    return steps
