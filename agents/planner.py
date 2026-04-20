"""
Planner Agent — decomposes a user request into a step-by-step plan.

Outputs a list of PlanStep objects that the ExecutorAgent runs one by one.
When multi_agent=false this agent is bypassed and the engine handles
everything in a single-agent loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.base import AgentResult, BaseAgent
from agents.tool_parser import PlanStep, parse_plan
from core.llm import Message
from core.prompts import PLANNER_PROMPT, render
from utils.logger import get_logger

log = get_logger(__name__)

_SYSTEM = """\
You are the Planner Agent for JARVIS. Decompose the user request into concrete
steps. Each step should map to one tool call OR a reasoning step with no tool.

Respond ONLY with valid JSON matching this schema:
{"plan": [{"step": 1, "description": "...", "tool": "<name or null>", "args": {...}}]}

Do not include any text outside the JSON block.
"""


class PlannerAgent(BaseAgent):
    """
    Given a user request, produces a structured list of steps.

    Returns an AgentResult where `response` is the JSON plan string and
    `metadata["steps"]` holds the parsed PlanStep list.
    """

    def __init__(self, **kwargs) -> None:
        from config import settings
        super().__init__(
            name="PlannerAgent",
            model=settings.agent.planner_model,
            **kwargs,
        )

    def _execute(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        tool_schemas = self._build_tool_schemas()
        schemas_text = json.dumps(tool_schemas, indent=2) if tool_schemas else "No tools available."

        messages = [
            Message("system", _SYSTEM),
            Message(
                "user",
                f"Available tools:\n{schemas_text}\n\nUser request: {user_input}",
            ),
        ]

        raw = self._chat(messages)
        log.debug("Planner raw response", extra={"response": raw[:300]})

        steps: List[PlanStep] = parse_plan(raw)

        log.info(
            "Plan created",
            extra={"steps": len(steps), "user_input": user_input[:60]},
        )
        for s in steps:
            log.debug(
                "Plan step",
                extra={"step": s.step, "description": s.description, "tool": s.tool},
            )

        return AgentResult(
            response=raw,
            steps_taken=1,
            success=True,
            metadata={"steps": steps, "raw_plan": raw},
        )
