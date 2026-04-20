"""
Startup health checks — run once before JARVIS begins accepting requests.

Each check is a function returning a CheckResult. A FAIL aborts startup;
a WARN is logged but does not block. PASS is silent unless debug is on.

Usage:
    from utils.startup import run_startup_checks
    report = run_startup_checks()
    if not report.ok_to_start:
        sys.exit(1)
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == CheckStatus.FAIL


@dataclass
class StartupReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def ok_to_start(self) -> bool:
        return not any(r.failed for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.status == CheckStatus.WARN for r in self.results)

    def summary(self) -> str:
        lines = [f"  [{r.status.value:4}] {r.name}: {r.message}" for r in self.results]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_config_valid() -> CheckResult:
    """Validate required config values are sane."""
    from config import settings

    errors = []
    port = settings.api.port
    if not (1 <= port <= 65535):
        errors.append(f"api.port={port} is out of range (1-65535)")

    if not settings.llm.base_url.startswith(("http://", "https://")):
        errors.append(f"llm.base_url='{settings.llm.base_url}' must start with http(s)://")

    if settings.llm.max_tokens < 1:
        errors.append(f"llm.max_tokens={settings.llm.max_tokens} must be >= 1")

    if settings.agent.max_iterations < 1:
        errors.append(f"agent.max_iterations={settings.agent.max_iterations} must be >= 1")

    if errors:
        return CheckResult("config_valid", CheckStatus.FAIL, "; ".join(errors))
    return CheckResult("config_valid", CheckStatus.PASS, "All config values valid")


def _check_directories_writable() -> CheckResult:
    """Ensure log and data directories exist and are writable."""
    from config import settings

    dirs_to_check = [
        ("log_dir", settings.log_dir_path),
        ("data_dir", settings.long_term_db_path.parent),
    ]
    failures = []
    for label, path in dirs_to_check:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
        except Exception as exc:
            failures.append(f"{label} ({path}): {exc}")

    if failures:
        return CheckResult("directories_writable", CheckStatus.FAIL, "; ".join(failures))
    return CheckResult("directories_writable", CheckStatus.PASS, "All directories writable")


def _check_llm_reachable() -> CheckResult:
    """Ping the Ollama server. WARN (not FAIL) if unreachable — offline use still possible."""
    from config import settings
    import urllib.parse

    parsed = urllib.parse.urlparse(settings.llm.base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    try:
        with socket.create_connection((host, port), timeout=3):
            pass
        return CheckResult("llm_reachable", CheckStatus.PASS, f"Ollama reachable at {host}:{port}")
    except (OSError, socket.timeout) as exc:
        return CheckResult(
            "llm_reachable",
            CheckStatus.WARN,
            f"Ollama not reachable at {host}:{port} ({exc}). LLM features will fail.",
        )


def _check_model_available() -> CheckResult:
    """Check the configured model is pulled. WARN if not — user may pull it later."""
    from config import settings

    try:
        from core.llm import get_llm_client
        client = get_llm_client()
        healthy = client.health_check()
        if healthy:
            return CheckResult("model_available", CheckStatus.PASS,
                               f"Model '{settings.llm.model}' is available")
        return CheckResult("model_available", CheckStatus.WARN,
                           f"Model '{settings.llm.model}' not responding — run: ollama pull {settings.llm.model}")
    except Exception as exc:
        return CheckResult("model_available", CheckStatus.WARN, str(exc))


def _check_port_free() -> CheckResult:
    """Check the API port is not already in use."""
    from config import settings

    host = settings.api.host
    port = settings.api.port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return CheckResult("port_free", CheckStatus.PASS, f"Port {port} is free")
    except OSError:
        return CheckResult(
            "port_free",
            CheckStatus.WARN,
            f"Port {port} already in use — API server may fail to bind",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_CHECKS: List[Callable[[], CheckResult]] = [
    _check_config_valid,
    _check_directories_writable,
    _check_llm_reachable,
    _check_model_available,
    _check_port_free,
]


def run_startup_checks(checks: Optional[List[Callable[[], CheckResult]]] = None) -> StartupReport:
    """
    Execute all startup checks and return a consolidated report.
    Any FAIL result sets report.ok_to_start = False.
    """
    report = StartupReport()
    for check_fn in (checks or _CHECKS):
        try:
            result = check_fn()
        except Exception as exc:
            result = CheckResult(
                name=getattr(check_fn, "__name__", "unknown"),
                status=CheckStatus.FAIL,
                message=f"Check raised an exception: {exc}",
            )

        report.results.append(result)

        if result.status == CheckStatus.PASS:
            log.debug("Startup check passed", extra={"check": result.name})
        elif result.status == CheckStatus.WARN:
            log.warning("Startup check warning", extra={"check": result.name, "detail": result.message})
        else:
            log.error("Startup check FAILED", extra={"check": result.name, "detail": result.message})

    return report
