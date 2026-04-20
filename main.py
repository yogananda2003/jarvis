"""
JARVIS — Entry point.

Usage:
    python main.py                # start full system
    python main.py --debug        # enable debug logging
    python main.py --no-voice     # skip voice subsystem
    python main.py --api-only     # start API server only (no voice)
    python main.py --skip-checks  # skip startup health checks (dev shortcut)
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: config + logging MUST be initialised before any module import
# ---------------------------------------------------------------------------
from config import settings
from utils.logger import configure_logging, get_logger

configure_logging(
    level=settings.logging.level,
    fmt=settings.logging.format,
    log_dir=settings.log_dir_path,
    max_bytes=settings.logging.max_bytes,
    backup_count=settings.logging.backup_count,
    audit_log=settings.logging.audit_log,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{settings.jarvis.name} v{settings.jarvis.version} — Offline AI Assistant"
    )
    parser.add_argument("--debug",        action="store_true", help="Enable debug logging")
    parser.add_argument("--no-voice",     action="store_true", help="Disable voice subsystem")
    parser.add_argument("--api-only",     action="store_true", help="Start API server only")
    parser.add_argument("--skip-checks",  action="store_true", help="Skip startup health checks")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    banner = f"""
+------------------------------------------+
|   {settings.jarvis.name} -- Offline AI Assistant        |
|   Version : {settings.jarvis.version:<30} |
|   LLM     : {settings.llm.model:<30} |
|   Debug   : {'ON' if settings.debug else 'OFF':<30} |
+------------------------------------------+
""".strip()
    print(banner)


# ---------------------------------------------------------------------------
# Component launchers
# ---------------------------------------------------------------------------

def start_api_server(memory_manager=None) -> None:
    """Launch FastAPI server (blocking — call in a thread for concurrent use)."""
    import uvicorn
    from api.server import create_app

    application = create_app(memory_manager=memory_manager)
    log.info(
        "Starting API server",
        extra={"host": settings.api.host, "port": settings.api.port},
    )
    uvicorn.run(
        application,
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,   # use our own logger
    )


def start_voice_pipeline(engine=None) -> None:
    """Start the JARVIS voice pipeline in a background thread."""
    try:
        from voice.pipeline import VoicePipeline
        from agents.engine import AgentEngine

        _engine = engine or AgentEngine()

        def _handle_command(text: str) -> str:
            result = _engine.run(text)
            return result.response

        pipeline = VoicePipeline(on_command=_handle_command)
        pipeline.start()
        log.info("Voice pipeline started")
        return pipeline
    except ImportError as exc:
        log.warning("Voice pipeline unavailable", extra={"error": str(exc)})
    except Exception as exc:
        log.error("Voice pipeline failed to start", extra={"error": str(exc)})


def start_agent_engine() -> None:
    log.info("Agent engine start requested — not yet implemented (Phase 4)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # Apply CLI overrides to settings
    if args.debug:
        import logging as _logging
        _logging.getLogger().setLevel("DEBUG")
        settings.jarvis.debug = True

    _print_banner()

    # ------------------------------------------------------------------
    # Config validation (fast-fail before anything else starts)
    # ------------------------------------------------------------------
    config_errors = settings.validate()
    if config_errors:
        for err in config_errors:
            log.critical("Config validation error", extra={"error": err})
        print(f"\nFATAL: {len(config_errors)} config error(s). Fix config.yaml and retry.")
        return 1

    log.info(
        "JARVIS starting",
        extra={
            "version": settings.jarvis.version,
            "model": settings.llm.model,
            "debug": settings.debug,
            "voice_enabled": settings.features.voice and not args.no_voice,
            "api_enabled": settings.features.api_server,
        },
    )

    # ------------------------------------------------------------------
    # Startup health checks
    # ------------------------------------------------------------------
    if not args.skip_checks:
        from utils.startup import run_startup_checks
        report = run_startup_checks()
        print("\nStartup checks:")
        print(report.summary())
        print()

        if not report.ok_to_start:
            log.critical("Startup checks failed — aborting")
            return 1
        if report.has_warnings:
            log.warning("Startup checks completed with warnings")

    # ------------------------------------------------------------------
    # Shutdown coordinator
    # ------------------------------------------------------------------
    from utils.shutdown import shutdown_manager
    shutdown_manager.install_signal_handlers()

    try:
        # ------------------------------------------------------------------
        # Memory
        # ------------------------------------------------------------------
        from memory.manager import MemoryManager
        memory_manager = MemoryManager()
        shutdown_manager.register("memory_manager", lambda: memory_manager.short_term.clear())

        # ------------------------------------------------------------------
        # API server
        # ------------------------------------------------------------------
        if settings.features.api_server:
            if args.api_only:
                shutdown_manager.register("api_server", lambda: log.info("API server stopping"))
                start_api_server(memory_manager=memory_manager)
                return 0
            else:
                api_thread = threading.Thread(
                    target=start_api_server,
                    kwargs={"memory_manager": memory_manager},
                    daemon=False,
                    name="api-server",
                )
                api_thread.start()
                shutdown_manager.register("api_server", lambda: log.info("API thread will exit with process"))

        # ------------------------------------------------------------------
        # Voice + Agent
        # ------------------------------------------------------------------
        if settings.features.voice and not args.no_voice:
            start_voice_pipeline()


        log.info("JARVIS initialised successfully. Ready.")
        return 0

    except KeyboardInterrupt:
        log.info("JARVIS shutting down (KeyboardInterrupt)")
        shutdown_manager.shutdown()
        return 0
    except Exception as exc:
        log.critical("Fatal startup error", exc_info=True, extra={"error": str(exc)})
        shutdown_manager.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
