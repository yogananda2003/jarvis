"""
Tests for the Ollama LLM client.

Unit tests mock the Ollama SDK — no real server needed.
Integration tests (marked) hit a live Ollama instance.

Run unit tests only:
    pytest tests/test_llm.py -v -m "not integration"

Run all (requires running Ollama):
    pytest tests/test_llm.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import (
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMResponse,
    LLMTimeoutError,
    Message,
    OllamaClient,
    StreamChunk,
    get_llm_client,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_client(**kwargs) -> OllamaClient:
    """Return a client with a mocked Ollama SDK underneath."""
    with patch("core.llm.Client"):
        client = OllamaClient(**kwargs)
    return client


def _mock_chat_response(content: str = "Hello!") -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    resp.prompt_eval_count = 10
    resp.eval_count = 20
    resp.done = True
    return resp


def _mock_stream_chunks(tokens: list[str]) -> list[MagicMock]:
    chunks = []
    for i, token in enumerate(tokens):
        chunk = MagicMock()
        chunk.message.content = token
        chunk.done = i == len(tokens) - 1
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

class TestMessage:
    def test_to_dict(self):
        m = Message("user", "hi")
        assert m.to_dict() == {"role": "user", "content": "hi"}

    def test_roles(self):
        for role in ("system", "user", "assistant"):
            m = Message(role, "text")
            assert m.role == role


# ---------------------------------------------------------------------------
# OllamaClient — chat (non-streaming)
# ---------------------------------------------------------------------------

class TestOllamaClientChat:
    def test_successful_chat(self):
        client = _make_client()
        client._client.chat.return_value = _mock_chat_response("Test response")

        response = client.chat([Message("user", "Hello")])

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.done is True
        assert response.total_tokens == 30

    def test_chat_passes_tools(self):
        client = _make_client()
        client._client.chat.return_value = _mock_chat_response()
        tools = [{"name": "open_app", "description": "Opens an app"}]

        client.chat([Message("user", "open Chrome")], tools=tools)

        call_kwargs = client._client.chat.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_model_not_found_raises(self):
        from ollama import ResponseError
        client = _make_client()
        client._client.chat.side_effect = ResponseError("model 'xyz' not found")

        with pytest.raises(LLMModelNotFoundError):
            client.chat([Message("user", "hi")])

    def test_connection_error_raises(self):
        import httpx
        client = _make_client()
        client._client.chat.side_effect = httpx.ConnectError("refused")

        with pytest.raises(LLMConnectionError):
            client.chat([Message("user", "hi")])

    def test_retry_on_transient_error(self):
        import httpx
        client = _make_client()
        client._retry_max = 3
        client._retry_backoff = 0  # no sleep in tests

        # Fail twice, succeed on third attempt
        client._client.chat.side_effect = [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            _mock_chat_response("recovered"),
        ]

        with patch("core.llm.time.sleep"):
            response = client.chat([Message("user", "hi")])

        assert response.content == "recovered"
        assert client._client.chat.call_count == 3

    def test_exhausted_retries_raises(self):
        import httpx
        client = _make_client()
        client._retry_max = 2
        client._retry_backoff = 0

        client._client.chat.side_effect = httpx.ConnectError("refused")

        with patch("core.llm.time.sleep"):
            with pytest.raises(LLMConnectionError):
                client.chat([Message("user", "hi")])


# ---------------------------------------------------------------------------
# OllamaClient — streaming
# ---------------------------------------------------------------------------

class TestOllamaClientStream:
    def test_stream_yields_chunks(self):
        client = _make_client()
        tokens = ["Hello", " world", "!"]
        client._client.chat.return_value = iter(_mock_stream_chunks(tokens))

        chunks = list(client.stream([Message("user", "hi")]))

        assert len(chunks) == 3
        assert all(isinstance(c, StreamChunk) for c in chunks)
        assert "".join(c.content for c in chunks) == "Hello world!"
        assert chunks[-1].done is True

    def test_stream_last_chunk_is_done(self):
        client = _make_client()
        client._client.chat.return_value = iter(_mock_stream_chunks(["a", "b"]))

        chunks = list(client.stream([Message("user", "hi")]))
        assert chunks[-1].done is True
        assert chunks[0].done is False


# ---------------------------------------------------------------------------
# OllamaClient — generate
# ---------------------------------------------------------------------------

class TestOllamaClientGenerate:
    def test_generate_without_system(self):
        client = _make_client()
        client._client.chat.return_value = _mock_chat_response("result")

        response = client.generate("What is 2+2?")

        assert response.content == "result"
        call_messages = client._client.chat.call_args.kwargs["messages"]
        assert call_messages[0]["role"] == "user"

    def test_generate_with_system(self):
        client = _make_client()
        client._client.chat.return_value = _mock_chat_response("result")

        client.generate("What is 2+2?", system="You are a math tutor.")

        call_messages = client._client.chat.call_args.kwargs["messages"]
        assert call_messages[0]["role"] == "system"
        assert call_messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_passes_when_model_present(self):
        client = _make_client()
        mock_model = MagicMock()
        mock_model.model = "llama3:latest"
        client._client.list.return_value = MagicMock(models=[mock_model])
        client._model = "llama3"

        assert client.health_check() is True

    def test_health_check_raises_when_model_missing(self):
        client = _make_client()
        mock_model = MagicMock()
        mock_model.model = "codellama:latest"
        client._client.list.return_value = MagicMock(models=[mock_model])
        client._model = "llama3"

        with pytest.raises(LLMModelNotFoundError):
            client.health_check()

    def test_health_check_raises_on_connection_failure(self):
        import httpx
        client = _make_client()
        client._client.list.side_effect = httpx.ConnectError("refused")

        with pytest.raises(LLMConnectionError):
            client.health_check()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_llm_client_returns_same_instance(self):
        with patch("core.llm._client_instance", None):
            with patch("core.llm.Client"):
                c1 = get_llm_client()
                c2 = get_llm_client()
        assert c1 is c2


# ---------------------------------------------------------------------------
# Integration tests (require live Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLiveOllama:
    """These tests require `ollama serve` to be running with llama3 pulled."""

    def test_live_chat(self):
        client = OllamaClient()
        client.health_check()
        response = client.chat([Message("user", "Say the word PING and nothing else.")])
        assert "PING" in response.content.upper()

    def test_live_stream(self):
        client = OllamaClient()
        chunks = list(client.stream([Message("user", "Count to 3, one number per word.")]))
        full = "".join(c.content for c in chunks)
        assert len(full) > 0
        assert chunks[-1].done is True
