"""
Tests for the Agent System (planner, executor, supervisor, engine).
All LLM calls are mocked — no Ollama server needed.

Run:
    pytest tests/test_agents.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_llm(response_text: str = "Done.") -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = MagicMock(content=response_text)
    return llm


def _mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.all_schemas.return_value = [
        {
            "name": "open_app",
            "description": "Open an application",
            "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
        }
    ]
    registry.execute.return_value = "App opened successfully."
    return registry


# ---------------------------------------------------------------------------
# ToolCall / AgentResult
# ---------------------------------------------------------------------------

class TestAgentDataTypes:
    def test_tool_call_defaults(self):
        from agents.base import ToolCall
        tc = ToolCall(name="open_app")
        assert tc.arguments == {}

    def test_agent_result_defaults(self):
        from agents.base import AgentResult
        r = AgentResult(response="Hello")
        assert r.success is True
        assert r.steps_taken == 0
        assert r.tool_calls == []


# ---------------------------------------------------------------------------
# Tool Parser
# ---------------------------------------------------------------------------

class TestToolParser:
    def test_parse_fenced_json(self):
        from agents.tool_parser import parse_text_tool_calls
        text = '```json\n{"name": "open_app", "arguments": {"app_name": "Chrome"}}\n```'
        calls, remainder = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "open_app"
        assert calls[0].arguments == {"app_name": "Chrome"}
        assert "open_app" not in remainder

    def test_parse_no_tool_call(self):
        from agents.tool_parser import parse_text_tool_calls
        text = "I cannot help with that."
        calls, remainder = parse_text_tool_calls(text)
        assert calls == []
        assert remainder == text

    def test_parse_plan_valid_json(self):
        from agents.tool_parser import parse_plan
        text = '{"plan": [{"step": 1, "description": "Open Chrome", "tool": "open_app", "args": {"app_name": "Chrome"}}]}'
        steps = parse_plan(text)
        assert len(steps) == 1
        assert steps[0].tool == "open_app"
        assert steps[0].step == 1

    def test_parse_plan_fallback(self):
        from agents.tool_parser import parse_plan
        # No JSON — should return single fallback step
        steps = parse_plan("Just open Chrome please.")
        assert len(steps) == 1
        assert steps[0].step == 1

    def test_parse_plan_fenced(self):
        from agents.tool_parser import parse_plan
        text = '```json\n{"plan": [{"step": 1, "description": "Search web", "tool": null, "args": {}}]}\n```'
        steps = parse_plan(text)
        assert len(steps) == 1
        assert steps[0].tool is None

    def test_parse_multiple_steps(self):
        from agents.tool_parser import parse_plan
        import json
        plan = {"plan": [
            {"step": 1, "description": "Step 1", "tool": "open_app", "args": {"app_name": "Chrome"}},
            {"step": 2, "description": "Step 2", "tool": None, "args": {}},
        ]}
        steps = parse_plan(json.dumps(plan))
        assert len(steps) == 2
        assert steps[1].tool is None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    def test_parse_success_verdict(self):
        from agents.supervisor import SupervisorAgent
        verdict = SupervisorAgent._parse_verdict(
            '{"status": "success", "response": "Task complete."}'
        )
        assert verdict.status == "success"
        assert verdict.response == "Task complete."

    def test_parse_retry_verdict(self):
        from agents.supervisor import SupervisorAgent
        verdict = SupervisorAgent._parse_verdict(
            '{"status": "retry", "reason": "Wrong app opened.", "retry_step": 1}'
        )
        assert verdict.status == "retry"
        assert verdict.retry_step == 1

    def test_parse_invalid_json_fallback(self):
        from agents.supervisor import SupervisorAgent
        verdict = SupervisorAgent._parse_verdict("The task looks complete.")
        assert verdict.status == "success"
        assert "complete" in verdict.response.lower()

    def test_parse_fenced_json(self):
        from agents.supervisor import SupervisorAgent
        verdict = SupervisorAgent._parse_verdict(
            '```\n{"status": "success", "response": "All done."}\n```'
        )
        assert verdict.status == "success"


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    def test_plan_created_from_llm_response(self):
        from agents.planner import PlannerAgent
        import json

        plan_json = json.dumps({"plan": [
            {"step": 1, "description": "Open Chrome", "tool": "open_app", "args": {"app_name": "Chrome"}},
        ]})

        agent = PlannerAgent(llm=_mock_llm(plan_json), tool_registry=_mock_registry())
        result = agent.run("Open Chrome")

        assert result.success
        steps = result.metadata["steps"]
        assert len(steps) == 1
        assert steps[0].tool == "open_app"

    def test_plan_fallback_on_bad_llm_output(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent(llm=_mock_llm("I will open Chrome for you."), tool_registry=_mock_registry())
        result = agent.run("Open Chrome")

        # Should fall back to single step, not crash
        assert result.success
        assert len(result.metadata["steps"]) >= 1


# ---------------------------------------------------------------------------
# ExecutorAgent
# ---------------------------------------------------------------------------

class TestExecutorAgent:
    def test_react_loop_no_tool_returns_direct_answer(self):
        from agents.executor import ExecutorAgent

        agent = ExecutorAgent(llm=_mock_llm("The weather is sunny."), tool_registry=_mock_registry())
        result = agent.run("What is the weather?", {})

        assert result.success
        assert "sunny" in result.response

    def test_react_loop_tool_call_is_executed(self):
        from agents.executor import ExecutorAgent
        from unittest.mock import call

        tool_response = '```json\n{"name": "open_app", "arguments": {"app_name": "Chrome"}}\n```'
        # First call returns tool invocation; second returns final answer
        llm = _mock_llm()
        llm.chat.side_effect = [
            MagicMock(content=tool_response),
            MagicMock(content="Chrome has been opened."),
        ]

        registry = _mock_registry()
        agent = ExecutorAgent(llm=llm, tool_registry=registry)
        result = agent.run("Open Chrome", {})

        registry.execute.assert_called_once_with("open_app", app_name="Chrome")
        assert result.success
        assert len(result.tool_calls) == 1

    def test_max_iterations_respected(self):
        from agents.executor import ExecutorAgent

        # Always returns a tool call — should eventually give up
        tool_json = '{"name": "open_app", "arguments": {"app_name": "Chrome"}}'
        llm = _mock_llm()
        llm.chat.return_value = MagicMock(content=f"```json\n{tool_json}\n```")

        registry = _mock_registry()
        agent = ExecutorAgent(llm=llm, tool_registry=registry)
        agent._max_iterations = 3
        result = agent.run("Open Chrome", {})

        assert result.steps_taken == 3
        assert result.success is False
        assert result.error == "Max iterations reached"

    def test_tool_failure_is_handled(self):
        from agents.executor import ExecutorAgent

        tool_json = '{"name": "open_app", "arguments": {"app_name": "Chrome"}}'
        llm = _mock_llm()
        llm.chat.side_effect = [
            MagicMock(content=f"```json\n{tool_json}\n```"),
            MagicMock(content="I could not open the app due to an error."),
        ]

        registry = _mock_registry()
        registry.execute.side_effect = RuntimeError("App not found")
        agent = ExecutorAgent(llm=llm, tool_registry=registry)
        result = agent.run("Open Chrome", {})

        # Should not raise — error is handled gracefully
        assert isinstance(result.response, str)


# ---------------------------------------------------------------------------
# AgentEngine
# ---------------------------------------------------------------------------

class TestAgentEngine:
    def _make_engine(self, llm_response="All done.") -> "AgentEngine":
        from agents.engine import AgentEngine
        from unittest.mock import patch

        engine = AgentEngine()
        mock_executor = MagicMock()
        from agents.base import AgentResult
        mock_executor.run.return_value = AgentResult(
            response=llm_response, success=True, steps_taken=1
        )
        engine._executor = mock_executor
        return engine

    def test_run_returns_engine_result(self):
        from agents.engine import AgentEngine, EngineResult
        engine = self._make_engine("Hello!")
        result = engine.run("Say hello")
        assert isinstance(result, EngineResult)
        assert result.response == "Hello!"
        assert result.success

    def test_empty_input_returns_error(self):
        from agents.engine import AgentEngine
        engine = AgentEngine()
        result = engine.run("   ")
        assert not result.success

    def test_memory_provider_is_called(self):
        from agents.engine import AgentEngine
        from agents.base import AgentResult

        memory_mock = MagicMock(return_value="Previous context here.")
        engine = AgentEngine(memory_provider=memory_mock)
        engine._executor = MagicMock()
        engine._executor.run.return_value = AgentResult(response="Done", success=True)

        engine.run("What did I say before?")
        memory_mock.assert_called_once_with("What did I say before?")

    def test_memory_provider_failure_does_not_crash(self):
        from agents.engine import AgentEngine
        from agents.base import AgentResult

        def bad_memory(_): raise RuntimeError("DB down")

        engine = AgentEngine(memory_provider=bad_memory)
        engine._executor = MagicMock()
        engine._executor.run.return_value = AgentResult(response="Done", success=True)

        result = engine.run("Hello")
        assert result.response == "Done"

    def test_exception_in_executor_returns_error_response(self):
        from agents.engine import AgentEngine
        engine = AgentEngine()
        engine._executor = MagicMock()
        engine._executor.run.side_effect = RuntimeError("Unexpected crash")

        result = engine.run("Do something")
        assert not result.success
        assert "wrong" in result.response.lower()

    @pytest.mark.integration
    def test_live_run(self):
        """Requires a running Ollama server with llama3.2:1b."""
        from agents.engine import AgentEngine
        engine = AgentEngine()
        result = engine.run("What is 2 + 2? Reply with just the number.")
        assert result.success
        assert "4" in result.response
