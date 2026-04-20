"""Pydantic request/response models for the JARVIS REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /command
# ---------------------------------------------------------------------------

class CommandRequest(BaseModel):
    input: str = Field(..., min_length=1, description="User command text")
    session_id: str = Field(default="default", description="Session identifier for memory grouping")
    stream: bool = Field(default=False, description="Stream response chunks (SSE)")


class CommandResponse(BaseModel):
    response: str
    success: bool
    session_id: str
    latency_ms: float
    steps_taken: int
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# /health  /status
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    version: str
    model: str


class StatusResponse(BaseModel):
    status: str
    version: str
    model: str
    llm_healthy: bool
    memory_stats: Dict[str, Any]
    uptime_s: float


# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------

class MemoryMessage(BaseModel):
    role: str
    content: str
    timestamp: float


class MemoryListResponse(BaseModel):
    session_id: str
    messages: List[MemoryMessage]


class MemorySearchResponse(BaseModel):
    query: str
    results: List[MemoryMessage]


class FactRequest(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class FactResponse(BaseModel):
    key: str
    value: str
    timestamp: Optional[float] = None


class FactsListResponse(BaseModel):
    facts: List[FactResponse]


class DeleteResponse(BaseModel):
    deleted: bool
    detail: str
