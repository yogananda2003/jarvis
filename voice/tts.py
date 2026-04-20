"""
JARVIS Text-to-Speech engine.

Uses pyttsx3 with the David voice configured for the JARVIS personality:
- Deep, authoritative male voice
- JARVIS-style text preprocessing ("Sir," prefix, formal tone)
- Activation and response beep sounds (winsound on Windows)
- Streaming sentence-by-sentence for LLM output
"""

from __future__ import annotations

import platform
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import pyttsx3

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_OS = platform.system()

# ---------------------------------------------------------------------------
# JARVIS personality prefixes (rotated to avoid repetition)
# ---------------------------------------------------------------------------
_JARVIS_PREFIXES = [
    "Sir, ",
    "Of course. ",
    "Certainly, Sir. ",
    "Right away. ",
    "",
    "Sir, ",
    "",
    "Understood. ",
]
_prefix_index = 0
_prefix_lock = threading.Lock()


def _next_prefix() -> str:
    global _prefix_index
    with _prefix_lock:
        p = _JARVIS_PREFIXES[_prefix_index % len(_JARVIS_PREFIXES)]
        _prefix_index += 1
    return p


# ---------------------------------------------------------------------------
# Sound effects (Windows winsound, silent fallback on other OS)
# ---------------------------------------------------------------------------

def _beep(freq: int, duration_ms: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


def play_activation_sound() -> None:
    """JARVIS startup chime — ascending 3-tone sequence."""
    t = threading.Thread(target=_activation_beeps, daemon=True)
    t.start()


def play_ready_sound() -> None:
    """Single tone — JARVIS ready to listen."""
    threading.Thread(target=lambda: _beep(1100, 120), daemon=True).start()


def play_processing_sound() -> None:
    """Double low beep — JARVIS thinking."""
    def _go():
        _beep(600, 80)
        time.sleep(0.12)
        _beep(600, 80)
    threading.Thread(target=_go, daemon=True).start()


def play_done_sound() -> None:
    """Descending tone — response complete."""
    def _go():
        _beep(880, 80)
        time.sleep(0.08)
        _beep(660, 100)
    threading.Thread(target=_go, daemon=True).start()


def _activation_beeps() -> None:
    _beep(660, 90)
    time.sleep(0.07)
    _beep(880, 90)
    time.sleep(0.07)
    _beep(1100, 130)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TTSVoice:
    id: str
    name: str
    languages: List[str]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TTSError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# TTS Engine
# ---------------------------------------------------------------------------

class PyTTSTTS:
    """
    JARVIS-style TTS backed by pyttsx3.

    Configures the David (US English male) voice with a lower rate
    for an authoritative, clear JARVIS tone.

    Thread safety: all engine calls are serialised via an internal lock.
    """

    _SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

    def __init__(
        self,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_id: Optional[str] = None,
        jarvis_personality: bool = True,
    ) -> None:
        self._rate = rate or settings.voice.tts.rate
        self._volume = volume if volume is not None else settings.voice.tts.volume
        self._voice_id = voice_id or settings.voice.tts.voice_id
        self._jarvis_personality = jarvis_personality

        self._lock = threading.Lock()
        self._engine: Optional[pyttsx3.Engine] = None
        self._speaking = threading.Event()

        log.info("PyTTSTTS initialised", extra={"rate": self._rate, "volume": self._volume})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_engine(self) -> pyttsx3.Engine:
        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)

            # Auto-select David (deep male) voice if no override
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
            else:
                self._auto_select_voice(self._engine)

        return self._engine

    def _auto_select_voice(self, engine: pyttsx3.Engine) -> None:
        """Pick the best JARVIS-like voice available on this system."""
        voices = engine.getProperty("voices")
        # Prefer David (Windows), then any male voice, then fallback to first
        preferred_ids = ["david", "male", "en_us", "en-us"]
        for v in voices:
            name_lower = (v.name or "").lower()
            id_lower = (v.id or "").lower()
            if any(p in name_lower or p in id_lower for p in preferred_ids):
                engine.setProperty("voice", v.id)
                log.info("Auto-selected voice", extra={"voice": v.name})
                return
        # Fallback: use whatever the first voice is
        if voices:
            engine.setProperty("voice", voices[0].id)

    def close(self) -> None:
        with self._lock:
            if self._engine:
                self._engine.stop()
                self._engine = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, add_personality: bool = True) -> None:
        """Speak `text` synchronously. Adds JARVIS prefix if personality is on."""
        if not text.strip():
            return

        if self._jarvis_personality and add_personality:
            text = self._apply_jarvis_style(text)

        log.debug("TTS speaking", extra={"text": text[:60]})
        t_start = time.perf_counter()

        with self._lock:
            self._speaking.set()
            engine = self._get_engine()
            engine.say(text)
            engine.runAndWait()
            self._speaking.clear()

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        log.debug("TTS done", extra={"elapsed_ms": round(elapsed_ms, 1)})

    def speak_async(self, text: str, add_personality: bool = True) -> threading.Thread:
        t = threading.Thread(
            target=self.speak,
            args=(text,),
            kwargs={"add_personality": add_personality},
            daemon=True,
        )
        t.start()
        return t

    def speak_stream(self, chunks: Iterator, *, extract_content: bool = True) -> None:
        """Speak sentence-by-sentence as LLM tokens arrive."""
        buffer = ""
        first = True

        for chunk in chunks:
            token = chunk.content if extract_content else str(chunk)
            buffer += token

            sentences, buffer = self._split_sentences(buffer)
            for sentence in sentences:
                if sentence.strip():
                    # Only apply personality prefix to the first sentence
                    self.speak(sentence, add_personality=first)
                    first = False

            done = getattr(chunk, "done", False)
            if done:
                break

        if buffer.strip():
            self.speak(buffer, add_personality=first)

    def stop(self) -> None:
        with self._lock:
            if self._engine and self._speaking.is_set():
                self._engine.stop()
                self._speaking.clear()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    def list_voices(self) -> List[TTSVoice]:
        engine = self._get_engine()
        voices = engine.getProperty("voices")
        return [
            TTSVoice(id=v.id, name=v.name, languages=list(v.languages))
            for v in voices
        ]

    def set_voice(self, voice_id: str) -> None:
        with self._lock:
            engine = self._get_engine()
            engine.setProperty("voice", voice_id)
            self._voice_id = voice_id

    def set_rate(self, rate: int) -> None:
        with self._lock:
            engine = self._get_engine()
            engine.setProperty("rate", rate)
            self._rate = rate

    def set_volume(self, volume: float) -> None:
        with self._lock:
            engine = self._get_engine()
            engine.setProperty("volume", max(0.0, min(1.0, volume)))
            self._volume = volume

    # ------------------------------------------------------------------
    # JARVIS personality
    # ------------------------------------------------------------------

    def _apply_jarvis_style(self, text: str) -> str:
        """Prepend a JARVIS-style prefix unless the response already has one."""
        skip_prefixes = (
            "sir,", "of course", "certainly", "understood", "right away",
            "hello", "welcome", "good", "i'm", "i am", "jarvis",
            "yes,", "no,", "yes.", "no.", "done.", "error",
        )
        if text.lower().startswith(skip_prefixes):
            return text
        prefix = _next_prefix()
        return prefix + text if prefix else text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _split_sentences(cls, text: str) -> tuple[list[str], str]:
        parts = cls._SENTENCE_END.split(text)
        if len(parts) == 1:
            return [], text
        return parts[:-1], parts[-1]
