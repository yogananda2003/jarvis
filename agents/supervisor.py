"""
Supervisor Agent — validates the executor's output against the original request.

Returns either:
  - "success" with the final response
  - "retry" with a reason and the step to re-run

Only active when multi_agent=true. In single-agent mode the engine
skips this for efficiency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.base import AgentResult, BaseAgent
from core.llm import Message
from utils.logger import get_logger

log = get_logger(__name__)

_SYSTEM = """\
You are the Supervisor Agent for JARVIS. You review a completed task and decide
if the result satisfies the original user request.

Respond ONLY with valid JSON — no other text:

If satisfied:
{"status": "success", "response": "<polished final answer for the user>"}

If not satisfied:
{"status": "retry", "reason": "<what is wrong>", "retry_step": <step number or 1>}
"""


@dataclass
class SupervisorVerdict:
    status: str                 # "success" | "retry"
    response: Optional[str] = None
    reason: Optional[str] = None
    retry_step: int = 1


class SupervisorAgent(BaseAgent):
    """
    Validates executor output. Called by the AgentEngine in multi-agent mode.
    """

    def __init__(self, **kwargs) -> None:
        from config import settings
        super().__init__(
            name="SupervisorAgent",
            model=settings.agent.planner_model,   # reuse planner model — supervisor is LLM-light
            **kwargs,
        )

    def _execute(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        execution_summary = context.get("execution_summary", "")
        verdict = self.validate(user_input, execution_summary)

        return AgentResult(
            response=verdict.response or "",
            success=verdict.status == "success",
            error=verdict.reason if verdict.status != "success" else None,
            metadata={"verdict": verdict},
        )

    def validate(
        self,
        original_request: str,
        execution_summary: str,
    ) -> SupervisorVerdict:
        """
        Ask the LLM to review the execution and return a structured verdict.
        """
        messages = [
            Message("system", _SYSTEM),
            Message(
                "user",
                f"Original request: {original_request}\n\n"
                f"Execution summary:\n{execution_summary}",
            ),
        ]

        raw = self._chat(messages)
        log.debug("Supervisor raw response", extra={"response": raw[:200]})

        verdict = self._parse_verdict(raw)
        log.info(
            "Supervisor verdict",
            extra={
                "status": verdict.status,
                "reason": verdict.reason or "",
            },
        )
        return verdict

    @staticmethod
    def _parse_verdict(text: str) -> SupervisorVerdict:
        """Parse the LLM JSON response into a SupervisorVerdict."""
        # Strip markdown fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean

        try:
            data = json.loads(clean)
            status = data.get("status", "success")
            return SupervisorVerdict(
                status=status,
                response=data.get("response"),
                reason=data.get("reason"),
                retry_step=int(data.get("retry_step", 1)),
            )
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat whatever the supervisor said as a success response
            log.warning("Supervisor response was not valid JSON — treating as success")
            return SupervisorVerdict(status="success", response=text.strip())
