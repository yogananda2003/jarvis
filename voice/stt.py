"""
Offline Speech-to-Text using OpenAI Whisper.

Records audio from the microphone, detects end-of-speech via silence,
then transcribes with the local Whisper model.

Design:
- Blocking `listen()` call: records until silence, returns transcript
- Configurable silence threshold and duration
- No cloud dependency — Whisper runs entirely on-device
"""

from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import whisper

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float       # 0.0 – 1.0 (avg log-prob, normalised)
    duration_s: float       # how long the audio clip was
    latency_ms: float       # time taken to transcribe


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class STTError(RuntimeError):
    """Generic STT error."""

class MicrophoneError(STTError):
    """Microphone is unavailable or not permitted."""


# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

class WhisperSTT:
    """
    Records from the default microphone and transcribes with Whisper.

    Usage:
        stt = WhisperSTT()
        result = stt.listen()
        print(result.text)
    """

    SAMPLE_RATE = 16_000        # Whisper requires 16 kHz
    CHANNELS = 1
    DTYPE = np.float32

    def __init__(
        self,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        silence_threshold: float = 0.015,
        silence_duration_s: float = 1.5,
        max_record_s: float = 30.0,
    ) -> None:
        self._model_size = model_size or settings.voice.stt.model_size
        self._language = language or settings.voice.stt.language
        self._device = device or settings.voice.stt.device
        self._silence_threshold = silence_threshold
        self._silence_duration_s = silence_duration_s
        self._max_record_s = max_record_s

        self._model: Optional[whisper.Whisper] = None
        log.info(
            "WhisperSTT initialised",
            extra={"model_size": self._model_size, "device": self._device},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Explicitly load the Whisper model into memory (lazy by default)."""
        if self._model is not None:
            return
        log.info("Loading Whisper model", extra={"size": self._model_size})
        t = time.perf_counter()
        self._model = whisper.load_model(
            self._model_size, device=self._device
        )
        elapsed = (time.perf_counter() - t) * 1000
        log.info("Whisper model loaded", extra={"elapsed_ms": round(elapsed, 1)})

    def listen(self, prompt: Optional[str] = None) -> TranscriptionResult:
        """
        Block until the user speaks and then falls silent.
        Returns the transcription.

        Args:
            prompt: optional Whisper context hint (improves accuracy for
                    known vocabulary like app names, commands).
        """
        self._ensure_model()
        audio = self._record_until_silence()
        return self._transcribe(audio, prompt=prompt)

    def transcribe_file(self, path: str) -> TranscriptionResult:
        """Transcribe an existing audio file (WAV/MP3/etc.)."""
        self._ensure_model()
        audio = whisper.load_audio(path)
        return self._transcribe(audio)

    def transcribe_array(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a raw float32 numpy array (16 kHz mono)."""
        self._ensure_model()
        return self._transcribe(audio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is None:
            self.load_model()

    def _record_until_silence(self) -> np.ndarray:
        """
        Record audio until the user has been silent for `silence_duration_s`
        or `max_record_s` is reached.
        """
        chunk_duration = 0.1            # seconds per chunk
        chunk_size = int(self.SAMPLE_RATE * chunk_duration)
        silence_chunks_needed = int(self._silence_duration_s / chunk_duration)
        max_chunks = int(self._max_record_s / chunk_duration)

        frames: list[np.ndarray] = []
        silent_chunks = 0
        speech_started = False

        log.debug("Microphone: listening...")

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
            ) as stream:
                for _ in range(max_chunks):
                    chunk, _ = stream.read(chunk_size)
                    chunk_flat = chunk.flatten()
                    rms = float(np.sqrt(np.mean(chunk_flat ** 2)))

                    if rms > self._silence_threshold:
                        speech_started = True
                        silent_chunks = 0
                        frames.append(chunk_flat)
                    elif speech_started:
                        frames.append(chunk_flat)
                        silent_chunks += 1
                        if silent_chunks >= silence_chunks_needed:
                            break

        except sd.PortAudioError as exc:
            raise MicrophoneError(f"Microphone error: {exc}") from exc

        if not frames:
            log.debug("No speech detected")
            return np.array([], dtype=self.DTYPE)

        audio = np.concatenate(frames)
        log.debug(
            "Recording complete",
            extra={"duration_s": round(len(audio) / self.SAMPLE_RATE, 2)},
        )
        return audio

    def _transcribe(
        self,
        audio: np.ndarray,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        if len(audio) == 0:
            return TranscriptionResult(
                text="", language=self._language,
                confidence=0.0, duration_s=0.0, latency_ms=0.0,
            )

        duration_s = len(audio) / self.SAMPLE_RATE
        t_start = time.perf_counter()

        kwargs: dict = {
            "language": self._language if self._language != "auto" else None,
            "task": "transcribe",
            "fp16": False,          # avoid GPU fp16 issues on CPU
            "verbose": False,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt

        result = self._model.transcribe(audio, **kwargs)  # type: ignore[union-attr]

        latency_ms = (time.perf_counter() - t_start) * 1000
        text = result["text"].strip()

        # Average log-probability → rough confidence (whisper returns segments)
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            confidence = min(1.0, max(0.0, (avg_logprob + 1.0)))
        else:
            confidence = 0.5

        log.info(
            "Transcription complete",
            extra={
                "text": text[:80],
                "language": result.get("language", self._language),
                "duration_s": round(duration_s, 2),
                "latency_ms": round(latency_ms, 1),
            },
        )

        return TranscriptionResult(
            text=text,
            language=result.get("language", self._language),
            confidence=confidence,
            duration_s=duration_s,
            latency_ms=latency_ms,
        )
