"""POST /command — run a user command through the AgentEngine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.models import CommandRequest, CommandResponse
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# Lazily initialised — avoids loading the LLM at import time
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from agents.engine import AgentEngine
        _engine = AgentEngine()
    return _engine


@router.post("", response_model=CommandResponse)
async def run_command(req: CommandRequest, request: Request):
    """
    Execute a natural-language command and return the agent's response.

    Set `stream=true` to receive a Server-Sent Events stream instead.
    """
    if req.stream:
        return _stream_response(req)

    memory_manager = request.app.state.memory_manager

    engine = _get_engine()
    if memory_manager is not None:
        engine._memory_provider = memory_manager.memory_provider

    try:
        result = engine.run(req.input)
    except Exception as exc:
        log.exception("Command execution failed", extra={"input": req.input})
        raise HTTPException(status_code=500, detail=str(exc))

    if memory_manager is not None:
        try:
            memory_manager.record_user(req.input)
            memory_manager.record_assistant(result.response)
        except Exception:
            pass

    tool_calls = [
        {"name": tc.name, "arguments": tc.arguments}
        for tc in result.tool_calls
    ]

    return CommandResponse(
        response=result.response,
        success=result.success,
        session_id=req.session_id,
        latency_ms=round(result.latency_ms, 1),
        steps_taken=result.steps_taken,
        tool_calls=tool_calls,
        error=result.error,
    )


def _stream_response(req: CommandRequest) -> StreamingResponse:
    """Return a Server-Sent Events stream of text chunks."""

    def generate():
        engine = _get_engine()
        try:
            for chunk in engine.run_stream(req.input):
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
