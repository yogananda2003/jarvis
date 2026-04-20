"""
Wake word detection for JARVIS.

Provides a pluggable interface with two implementations:

1. WhisperWakeWord (default) — uses the already-loaded Whisper model to
   continuously transcribe short audio clips and check for the keyword.
   Works 100% offline, no API key needed.
   Trade-off: ~1–2 s latency per detection window, moderate CPU usage.

2. PorcupineWakeWord (optional, recommended for production) — uses Picovoice
   Porcupine for sub-100ms, low-CPU detection.
   Requires a free API key at https://console.picovoice.ai/

Switch via config.yaml:
    voice:
      wake_word:
        engine: "whisper"   # or "porcupine"
        keyword: "jarvis"
        sensitivity: 0.5
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class WakeWordDetector(ABC):
    """
    Listens continuously in the background.
    Calls `on_detected()` each time the wake word is heard.
    """

    def __init__(
        self,
        keyword: Optional[str] = None,
        sensitivity: Optional[float] = None,
        on_detected: Optional[Callable[[], None]] = None,
    ) -> None:
        self._keyword = (keyword or settings.voice.wake_word.keyword).lower()
        self._sensitivity = sensitivity or settings.voice.wake_word.sensitivity
        self._on_detected = on_detected
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, on_detected: Optional[Callable[[], None]] = None) -> None:
        """Start the background listener thread."""
        if on_detected:
            self._on_detected = on_detected
        if self._running.is_set():
            log.warning("WakeWordDetector already running")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="wake-word-listener"
        )
        self._thread.start()
        log.info(
            "Wake word listener started",
            extra={"keyword": self._keyword, "engine": self.__class__.__name__},
        )

    def stop(self) -> None:
        """Stop the background listener thread."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("Wake word listener stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def _fire(self) -> None:
        """Call the user callback when the wake word is detected."""
        log.info("Wake word detected", extra={"keyword": self._keyword})
        if self._on_detected:
            try:
                self._on_detected()
            except Exception:
                log.exception("Error in wake word callback")

    @abstractmethod
    def _listen_loop(self) -> None:
        """Blocking loop — runs on the background thread."""
        ...


# ---------------------------------------------------------------------------
# Whisper-based implementation (no external API key)
# ---------------------------------------------------------------------------

class WhisperWakeWord(WakeWordDetector):
    """
    Detects the wake word by transcribing 2-second audio windows with Whisper.

    Loads the 'tiny' model by default to minimise latency and RAM usage.
    The main STT uses a larger model for full transcription accuracy.
    """

    SAMPLE_RATE = 16_000
    WINDOW_S = 2.0          # how many seconds of audio to analyse per pass
    COOLDOWN_S = 1.5        # silence after detection before listening again

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = None  # lazy-loaded on first use

    def _load_model(self):
        import whisper
        if self._model is None:
            log.info("Loading Whisper tiny model for wake word detection")
            self._model = whisper.load_model("tiny")

    def _listen_loop(self) -> None:
        self._load_model()
        chunk_size = int(self.SAMPLE_RATE * self.WINDOW_S)

        while self._running.is_set():
            try:
                audio = sd.rec(
                    chunk_size,
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype=np.float32,
                )
                sd.wait()
                audio_flat = audio.flatten()

                # Skip near-silent windows to save CPU
                rms = float(np.sqrt(np.mean(audio_flat ** 2)))
                if rms < 0.008:
                    continue

                import whisper as _whisper
                result = self._model.transcribe(  # type: ignore[union-attr]
                    audio_flat,
                    fp16=False,
                    language="en",
                    verbose=False,
                )
                transcript = result["text"].strip().lower()

                if self._keyword in transcript:
                    self._fire()
                    time.sleep(self.COOLDOWN_S)

            except Exception:
                log.exception("Wake word listen error — continuing")
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Porcupine-based implementation (low CPU, fast, requires API key)
# ---------------------------------------------------------------------------

class PorcupineWakeWord(WakeWordDetector):
    """
    High-performance wake word detection via Picovoice Porcupine.

    Requires:
        pip install pvporcupine
        PICOVOICE_ACCESS_KEY set in .env

    Supports built-in keywords: 'jarvis', 'alexa', 'hey siri', etc.
    Custom keywords require a paid Picovoice plan.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._access_key: Optional[str] = None

    def _listen_loop(self) -> None:
        import os
        try:
            import pvporcupine
        except ImportError:
            log.error("pvporcupine not installed — run: pip install pvporcupine")
            return

        access_key = os.environ.get("PICOVOICE_ACCESS_KEY", "")
        if not access_key:
            log.error(
                "PICOVOICE_ACCESS_KEY not set. "
                "Get a free key at https://console.picovoice.ai/"
            )
            return

        # Map our keyword to a Porcupine built-in (best-effort)
        supported = pvporcupine.KEYWORDS
        keyword = self._keyword if self._keyword in supported else "jarvis"
        if keyword not in supported:
            log.warning(
                "Wake word not in Porcupine built-ins — falling back to 'jarvis'",
                extra={"requested": self._keyword, "available": list(supported)},
            )
            keyword = "jarvis"

        try:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[self._sensitivity],
            )
        except pvporcupine.PorcupineActivationError:
            log.error("Invalid Picovoice access key")
            return

        frame_length = porcupine.frame_length
        sample_rate = porcupine.sample_rate
        log.info(
            "Porcupine wake word ready",
            extra={"keyword": keyword, "sample_rate": sample_rate},
        )

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16,
            ) as stream:
                while self._running.is_set():
                    audio, _ = stream.read(frame_length)
                    pcm = audio.flatten().tolist()
                    index = porcupine.process(pcm)
                    if index >= 0:
                        self._fire()
        finally:
            porcupine.delete()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_wake_word_detector(
    engine: Optional[str] = None,
    **kwargs,
) -> WakeWordDetector:
    """
    Return the appropriate WakeWordDetector based on config or `engine` arg.

    engine: "whisper" | "porcupine"
    """
    engine = engine or getattr(settings.voice.wake_word, "engine", "whisper")

    if engine == "porcupine":
        log.info("Using Porcupine wake word engine")
        return PorcupineWakeWord(**kwargs)

    log.info("Using Whisper wake word engine")
    return WhisperWakeWord(**kwargs)
