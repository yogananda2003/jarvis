"""GET /health and GET /status endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from api.models import HealthResponse, StatusResponse
from config import settings
from core.llm import get_llm_client
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    return {
        "name": settings.jarvis.name,
        "version": settings.jarvis.version,
        "model": settings.llm.model,
        "endpoints": {
            "docs":    "/docs",
            "health":  "/health",
            "status":  "/status",
            "metrics": "/metrics",
            "command": "POST /command",
            "memory":  "/memory",
        },
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=settings.jarvis.version,
        model=settings.llm.model,
    )


@router.get("/status", response_model=StatusResponse)
async def status(request: Request):
    llm_ok = False
    try:
        llm_ok = get_llm_client().health_check()
    except Exception:
        pass

    memory_manager = request.app.state.memory_manager
    mem_stats: dict = {}
    if memory_manager is not None:
        try:
            mem_stats = memory_manager.stats()
        except Exception:
            pass

    uptime = time.time() - request.app.state.start_time

    return StatusResponse(
        status="ok" if llm_ok else "degraded",
        version=settings.jarvis.version,
        model=settings.llm.model,
        llm_healthy=llm_ok,
        memory_stats=mem_stats,
        uptime_s=round(uptime, 1),
    )


@router.get("/metrics")
async def get_metrics():
    """Return in-process metrics snapshot (counters + histograms)."""
    return metrics.snapshot()
