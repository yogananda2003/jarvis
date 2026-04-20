"""
Graceful shutdown coordinator.

Registers SIGTERM/SIGINT handlers and runs all registered callbacks
in LIFO order when the process receives a termination signal or
`shutdown()` is called directly.

Usage:
    from utils.shutdown import shutdown_manager

    # Register a cleanup callback
    shutdown_manager.register("close db", lambda: db.close())

    # Install OS signal handlers (call once at startup)
    shutdown_manager.install_signal_handlers()
"""

from __future__ import annotations

import signal
import threading
from typing import Callable, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


class ShutdownManager:
    """
    Thread-safe LIFO shutdown hook registry.

    Callbacks are called in reverse registration order so that resources
    opened last are closed first (dependency-safe teardown).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hooks: List[Tuple[str, Callable[[], None]]] = []
        self._shutdown_event = threading.Event()
        self._done = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, callback: Callable[[], None]) -> None:
        """Add a named shutdown hook."""
        with self._lock:
            self._hooks.append((name, callback))
        log.debug("Shutdown hook registered", extra={"hook": name})

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """
        Wire SIGTERM and SIGINT to the shutdown sequence.
        Safe to call multiple times (idempotent).
        """
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
            log.debug("Shutdown signal handlers installed (SIGTERM, SIGINT)")
        except (OSError, ValueError):
            # Signal setup may fail in non-main threads or on some platforms
            log.warning("Could not install signal handlers — not in main thread?")

    def _handle_signal(self, signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        log.info("Shutdown signal received", extra={"signal": sig_name})
        self.shutdown()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Run all registered hooks in LIFO order.
        Idempotent — safe to call multiple times; only executes once.
        """
        with self._lock:
            if self._done:
                return
            self._done = True
            hooks = list(reversed(self._hooks))

        log.info("Running shutdown hooks", extra={"count": len(hooks)})
        for name, callback in hooks:
            try:
                callback()
                log.debug("Shutdown hook completed", extra={"hook": name})
            except Exception as exc:
                log.error("Shutdown hook failed", extra={"hook": name, "error": str(exc)})

        self._shutdown_event.set()
        log.info("Shutdown complete")

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until shutdown() completes. Returns True if shutdown fired."""
        return self._shutdown_event.wait(timeout=timeout)

    @property
    def is_shutdown(self) -> bool:
        return self._done

    def reset(self) -> None:
        """Clear all hooks and reset state — intended for tests only."""
        with self._lock:
            self._hooks.clear()
            self._done = False
            self._shutdown_event.clear()


# Module-level singleton
shutdown_manager = ShutdownManager()
