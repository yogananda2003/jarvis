"""
JARVIS FastAPI application factory.

Usage:
    from api.server import create_app
    app = create_app()

    # or run directly:
    uvicorn api.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level start time for uptime calculation
_START_TIME = time.time()


def create_app(
    memory_manager=None,     # Optional[MemoryManager] — injected at startup
) -> FastAPI:
    """
    Build and return the FastAPI application.

    Args:
        memory_manager: A MemoryManager instance shared with the voice pipeline.
                        If None, the API creates its own (separate session).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "API server started",
            extra={"host": settings.api.host, "port": settings.api.port},
        )
        yield
        log.info("API server shutting down")

    app = FastAPI(
        title=settings.jarvis.name,
        version=settings.jarvis.version,
        description="JARVIS — Offline AI Assistant REST API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Request logging + metrics (innermost — wraps all routes)
    from api.middleware import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)

    # CORS (outermost — before logging so preflight requests are handled first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store shared state on app.state so routes can access it
    app.state.memory_manager = memory_manager
    app.state.start_time = _START_TIME

    # Register routers
    from api.routes.health import router as health_router
    from api.routes.command import router as command_router
    from api.routes.memory import router as memory_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(command_router, prefix="/command", tags=["Command"])
    app.include_router(memory_router, prefix="/memory", tags=["Memory"])

    # Serve frontend
    from pathlib import Path as _Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    _frontend = _Path(__file__).resolve().parent.parent / "frontend"
    if _frontend.exists():
        app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

        @app.get("/app", include_in_schema=False)
        async def frontend():
            return FileResponse(str(_frontend / "index.html"))

    return app


# Default app instance (for uvicorn cli: uvicorn api.server:app)
app = create_app()
