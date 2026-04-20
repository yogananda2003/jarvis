"""
Request logging and error capture middleware for the JARVIS API.

Adds to every request:
  - Structured log: method, path, status, duration_ms, client IP
  - Metrics: api_requests_total, api_errors_total, api_request_duration_ms
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("jarvis.api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        # Inject request ID for downstream correlation
        request.state.request_id = request_id

        response: Response
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            metrics.inc("api_errors_total")
            log.error(
                "Unhandled request exception",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        metrics.inc("api_requests_total")
        metrics.observe("api_request_duration_ms", duration_ms)
        if response.status_code >= 400:
            metrics.inc("api_errors_total")

        log.info(
            "API request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else "unknown",
            },
        )

        response.headers["X-Request-Id"] = request_id
        return response
