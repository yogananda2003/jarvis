"""Memory CRUD endpoints — /memory/*"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from api.models import (
    DeleteResponse,
    FactRequest,
    FactResponse,
    FactsListResponse,
    MemoryListResponse,
    MemoryMessage,
    MemorySearchResponse,
)
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


def _require_manager(request: Request):
    mgr = request.app.state.memory_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="Memory manager not available")
    return mgr


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@router.get("", response_model=MemoryListResponse)
async def list_memory(
    request: Request,
    session_id: str = Query(default="default"),
    limit: int = Query(default=20, ge=1, le=200),
):
    mgr = _require_manager(request)
    if mgr.long_term is None:
        # Fall back to short-term only
        msgs = [
            MemoryMessage(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in mgr.short_term.get_messages()[-limit:]
        ]
        return MemoryListResponse(session_id=session_id, messages=msgs)

    rows = mgr.long_term.get_recent(limit=limit, session_id=session_id)
    msgs = [MemoryMessage(**r) for r in rows]
    return MemoryListResponse(session_id=session_id, messages=msgs)


@router.get("/search", response_model=MemorySearchResponse)
async def search_memory(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
):
    mgr = _require_manager(request)
    if mgr.long_term is None:
        raise HTTPException(status_code=503, detail="Long-term memory not enabled")
    rows = mgr.long_term.search(q, limit=limit)
    msgs = [MemoryMessage(**r) for r in rows]
    return MemorySearchResponse(query=q, results=msgs)


@router.delete("/session/{session_id}", response_model=DeleteResponse)
async def clear_session(session_id: str, request: Request):
    mgr = _require_manager(request)
    mgr.short_term.clear()
    deleted = 0
    if mgr.long_term:
        deleted = mgr.long_term.delete_session(session_id)
    return DeleteResponse(
        deleted=True,
        detail=f"Cleared session '{session_id}' ({deleted} long-term rows removed)",
    )


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

@router.get("/facts", response_model=FactsListResponse)
async def list_facts(request: Request):
    mgr = _require_manager(request)
    if mgr.long_term is None:
        raise HTTPException(status_code=503, detail="Long-term memory not enabled")
    facts = mgr.long_term.list_facts()
    return FactsListResponse(
        facts=[FactResponse(key=f["key"], value=f["value"], timestamp=f["timestamp"]) for f in facts]
    )


@router.post("/facts", response_model=FactResponse, status_code=201)
async def set_fact(body: FactRequest, request: Request):
    mgr = _require_manager(request)
    if mgr.long_term is None:
        raise HTTPException(status_code=503, detail="Long-term memory not enabled")
    mgr.long_term.set_fact(body.key, body.value)
    return FactResponse(key=body.key, value=body.value)


@router.get("/facts/{key}", response_model=FactResponse)
async def get_fact(key: str, request: Request):
    mgr = _require_manager(request)
    if mgr.long_term is None:
        raise HTTPException(status_code=503, detail="Long-term memory not enabled")
    value = mgr.long_term.get_fact(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Fact '{key}' not found")
    return FactResponse(key=key, value=value)


@router.delete("/facts/{key}", response_model=DeleteResponse)
async def delete_fact(key: str, request: Request):
    mgr = _require_manager(request)
    if mgr.long_term is None:
        raise HTTPException(status_code=503, detail="Long-term memory not enabled")
    deleted = mgr.long_term.delete_fact(key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Fact '{key}' not found")
    return DeleteResponse(deleted=True, detail=f"Fact '{key}' deleted")
