"""
Ollama LLM client for JARVIS.

Responsibilities:
- Manage the connection to the local Ollama server
- Send chat / generate requests with full streaming support
- Retry transient failures with exponential backoff
- Expose a clean, type-safe interface that the rest of the codebase uses

The client is intentionally thin — no agent logic, no memory, no prompts.
Those concerns live in their own modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional

import httpx
import ollama
from ollama import Client, ResponseError

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single turn in a conversation."""
    role: str       # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Returned by a non-streaming chat call."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    done: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StreamChunk:
    """A single token/chunk from a streaming response."""
    content: str
    done: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMConnectionError(RuntimeError):
    """Ollama server is unreachable."""

class LLMModelNotFoundError(RuntimeError):
    """Requested model is not available locally."""

class LLMInsufficientMemoryError(RuntimeError):
    """System does not have enough RAM to load the model."""

class LLMTimeoutError(RuntimeError):
    """Request exceeded the configured timeout."""

class LLMError(RuntimeError):
    """Generic LLM error."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thread-safe wrapper around the Ollama Python SDK.

    Usage (non-streaming):
        client = OllamaClient()
        response = client.chat([Message("user", "Hello!")])
        print(response.content)

    Usage (streaming):
        for chunk in client.stream([Message("user", "Tell me a story")]):
            print(chunk.content, end="", flush=True)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self._base_url = base_url or settings.llm.base_url
        self._model = model or settings.llm.model
        self._temperature = temperature if temperature is not None else settings.llm.temperature
        self._max_tokens = max_tokens or settings.llm.max_tokens
        self._timeout = timeout or settings.llm.timeout

        self._retry_max = settings.llm.retry.max_attempts
        self._retry_backoff = settings.llm.retry.backoff_factor

        self._client = Client(host=self._base_url)

        log.debug(
            "OllamaClient initialised",
            extra={"model": self._model, "base_url": self._base_url},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Return True if the Ollama server is reachable and the configured
        model is available locally. Raises on hard failures.
        """
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            log.debug("Ollama models available", extra={"models": available})

            if not any(self._model in m for m in available):
                raise LLMModelNotFoundError(
                    f"Model '{self._model}' not found locally. "
                    f"Run: ollama pull {self._model}"
                )
            return True

        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                "Is `ollama serve` running?"
            ) from exc

    def chat(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
    ) -> LLMResponse:
        """
        Send a conversation and return the full response (non-streaming).
        Retries automatically on transient errors.
        """
        return self._with_retry(
            self._chat_once,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

    def stream(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[StreamChunk]:
        """
        Stream a response token by token.

        Usage:
            for chunk in client.stream(messages):
                print(chunk.content, end="", flush=True)
                if chunk.done:
                    break
        """
        return self._stream_once(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Single-turn generation (no conversation history).
        Useful for one-shot classification or extraction tasks.
        """
        messages = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, model=model)

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        log.info("Switching LLM model", extra={"from": self._model, "to": value})
        self._model = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_options(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict:
        return {
            "temperature": temperature if temperature is not None else self._temperature,
            "num_predict": max_tokens or self._max_tokens,
        }

    def _chat_once(
        self,
        messages: List[Message],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[dict]],
    ) -> LLMResponse:
        model_name = model or self._model
        options = self._build_options(temperature, max_tokens)
        raw_messages = [m.to_dict() for m in messages]

        log.debug(
            "LLM chat request",
            extra={
                "model": model_name,
                "num_messages": len(messages),
                "temperature": options["temperature"],
            },
        )

        t_start = time.perf_counter()

        kwargs: dict = dict(
            model=model_name,
            messages=raw_messages,
            options=options,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat(**kwargs)
        except ResponseError as exc:
            msg = str(exc).lower()
            if "not found" in msg:
                raise LLMModelNotFoundError(str(exc)) from exc
            if "memory" in msg:
                raise LLMInsufficientMemoryError(
                    f"{exc}\nTip: switch to a smaller model, e.g. `ollama pull llama3.2:1b`"
                ) from exc
            raise LLMError(f"Ollama error: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        log.info(
            "LLM chat complete",
            extra={
                "model": model_name,
                "elapsed_ms": round(elapsed_ms, 1),
                "prompt_tokens": response.prompt_eval_count or 0,
                "completion_tokens": response.eval_count or 0,
            },
        )

        return LLMResponse(
            content=response.message.content,
            model=model_name,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_duration_ms=elapsed_ms,
            done=True,
        )

    def _stream_once(
        self,
        messages: List[Message],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Iterator[StreamChunk]:
        model_name = model or self._model
        options = self._build_options(temperature, max_tokens)
        raw_messages = [m.to_dict() for m in messages]

        log.debug(
            "LLM stream request",
            extra={"model": model_name, "num_messages": len(messages)},
        )

        try:
            stream = self._client.chat(
                model=model_name,
                messages=raw_messages,
                options=options,
                stream=True,
            )
            for chunk in stream:
                content = chunk.message.content or ""
                done = chunk.done or False
                yield StreamChunk(content=content, done=done)
                if done:
                    log.debug("LLM stream finished", extra={"model": model_name})
                    break

        except ResponseError as exc:
            msg = str(exc).lower()
            if "not found" in msg:
                raise LLMModelNotFoundError(str(exc)) from exc
            if "memory" in msg:
                raise LLMInsufficientMemoryError(
                    f"{exc}\nTip: switch to a smaller model, e.g. `ollama pull llama3.2:1b`"
                ) from exc
            raise LLMError(str(exc)) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc

    def _with_retry(self, fn, **kwargs):
        """
        Execute `fn(**kwargs)` with exponential backoff on transient errors.
        LLMModelNotFoundError is NOT retried (it won't fix itself).
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._retry_max + 1):
            try:
                return fn(**kwargs)

            except (LLMModelNotFoundError, LLMInsufficientMemoryError):
                raise   # permanent — don't retry

            except (LLMConnectionError, LLMTimeoutError, LLMError) as exc:
                last_exc = exc
                if attempt == self._retry_max:
                    break
                wait = self._retry_backoff ** (attempt - 1)
                log.warning(
                    "LLM transient error — retrying",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self._retry_max,
                        "wait_s": wait,
                        "error": str(exc),
                    },
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module-level singleton — lazily initialised
# ---------------------------------------------------------------------------

_client_instance: Optional[OllamaClient] = None


def get_llm_client() -> OllamaClient:
    """
    Return the shared OllamaClient singleton.
    Creates it on first call; reuses it afterwards.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance
