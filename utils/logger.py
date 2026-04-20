"""
Structured logging for JARVIS.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("LLM response received", extra={"tokens": 128, "model": "llama3"})

Two log streams:
  - Application log  → logs/jarvis.log   (rotating, JSON or pretty)
  - Audit log        → logs/audit.log    (append-only, JSON always)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, MutableMapping, Optional, Tuple


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    _SKIP = frozenset(
        {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelno", "lineno", "module",
            "msecs", "msg", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "taskName", "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Attach any extra fields passed via `extra={...}`
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Pretty (human-readable) formatter
# ---------------------------------------------------------------------------

class _PrettyFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        base = f"{color}[{record.levelname:<8}]{self._RESET} {ts} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ---------------------------------------------------------------------------
# Module-level cache so we don't duplicate handlers
# ---------------------------------------------------------------------------

_configured: bool = False
_log_format: str = "json"
_log_level: str = "INFO"
_log_dir: Path = Path("logs")
_max_bytes: int = 10_485_760
_backup_count: int = 5
_audit_enabled: bool = True


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_dir: str | Path = "logs",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    audit_log: bool = True,
) -> None:
    """
    Bootstrap logging once at startup.
    Called by main.py before any other module initialises.
    Subsequent calls are no-ops.
    """
    global _configured, _log_format, _log_level, _log_dir, _max_bytes, _backup_count, _audit_enabled

    if _configured:
        return

    _log_format = fmt
    _log_level = level.upper()
    _log_dir = Path(log_dir)
    _max_bytes = max_bytes
    _backup_count = backup_count
    _audit_enabled = audit_log

    _log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(_log_level)

    # -- console handler -------------------------------------------------
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_log_level)
    console.setFormatter(
        _JSONFormatter() if fmt == "json" else _PrettyFormatter()
    )
    root.addHandler(console)

    # -- rotating file handler -------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=_log_dir / "jarvis.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(_log_level)
    file_handler.setFormatter(_JSONFormatter())
    root.addHandler(file_handler)

    # -- audit log (always JSON, append-only) ----------------------------
    if audit_log:
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=_log_dir / "audit.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        audit_handler.setLevel(logging.DEBUG)
        audit_handler.setFormatter(_JSONFormatter())
        audit_logger = logging.getLogger("jarvis.audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = False   # audit records stay in audit.log only

    # Silence noisy third-party libraries
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a standard application logger scoped to `name`."""
    return logging.getLogger(name)


def get_audit_logger() -> logging.Logger:
    """
    Return the append-only audit logger.
    Use for tracking user commands, tool executions, and sensitive actions.
    """
    return logging.getLogger("jarvis.audit")


@contextmanager
def log_context(logger: logging.Logger, **fields) -> Generator[None, None, None]:
    """
    Context manager that injects structured fields into every log call
    made within the block, via a LoggerAdapter.

    Usage:
        with log_context(log, request_id="abc123", user="alice"):
            log.info("Processing")   # record will include request_id, user
    """
    adapter = logging.LoggerAdapter(logger, fields)
    # Patch the module-level logger temporarily
    original_manager = logger.manager
    try:
        yield adapter
    finally:
        logger.manager = original_manager
