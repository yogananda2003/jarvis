"""
Async command queue — decouple command submission from execution.

Allows voice pipeline, API, and other producers to submit commands
without blocking, while a pool of worker threads processes them.

Usage:
    from utils.queue import CommandQueue

    q = CommandQueue(engine, workers=2)
    q.start()

    future = q.submit("open chrome")
    result = future.result(timeout=30)   # blocks until done

    q.stop()
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)

_SENTINEL = object()   # signals worker threads to exit


@dataclass
class _Task:
    command: str
    future: Future
    session_id: str = "default"


class CommandQueue:
    """
    Thread-pool-backed command queue.

    Each submitted command is processed by the AgentEngine.run() method
    in a worker thread. Callers receive a Future they can wait on.
    """

    def __init__(
        self,
        engine,                          # AgentEngine instance
        workers: int = 1,
        maxsize: int = 50,
    ) -> None:
        self._engine = engine
        self._workers = workers
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._threads: list[threading.Thread] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self._workers):
            t = threading.Thread(
                target=self._worker,
                name=f"cmd-worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        log.info("CommandQueue started", extra={"workers": self._workers})

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._threads:
            self._q.put(_SENTINEL)
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        log.info("CommandQueue stopped")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, command: str, session_id: str = "default") -> Future:
        """
        Enqueue a command and return a Future for the EngineResult.
        Raises queue.Full if the queue is at capacity.
        """
        future: Future = Future()
        task = _Task(command=command, future=future, session_id=session_id)
        self._q.put_nowait(task)
        metrics.inc("queue_submitted_total")
        log.debug("Command queued", extra={"command": command[:60], "session": session_id})
        return future

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                self._q.task_done()
                break

            task: _Task = item
            try:
                metrics.inc("queue_processing_total")
                with metrics.timer("queue_processing_ms"):
                    result = self._engine.run(task.command)
                task.future.set_result(result)
            except Exception as exc:
                metrics.inc("queue_errors_total")
                log.error(
                    "CommandQueue worker error",
                    extra={"command": task.command[:60], "error": str(exc)},
                )
                task.future.set_exception(exc)
            finally:
                self._q.task_done()
