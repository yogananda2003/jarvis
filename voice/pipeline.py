"""
Voice Pipeline — orchestrates wake word → STT → agent → TTS.

Flow:
    IDLE ──(wake word / hotkey)──► LISTENING ──(silence)──► PROCESSING
      ▲                                                           │
      └─────────────── SPEAKING ◄──────────────(response)────────┘
"""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

from voice.stt import WhisperSTT
from voice.tts import (
    PyTTSTTS,
    play_activation_sound,
    play_ready_sound,
    play_processing_sound,
    play_done_sound,
)
from voice.wake_word import WakeWordDetector, create_wake_word_detector
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class PipelineState(Enum):
    IDLE       = auto()
    LISTENING  = auto()
    PROCESSING = auto()
    SPEAKING   = auto()
    STOPPED    = auto()


class VoicePipeline:
    """
    Full JARVIS voice loop.

    Args:
        on_command: callable that receives transcribed text, returns response string.
    """

    _BOOT_LINES = [
        "Good day. All systems are online.",
        "JARVIS is online. Ready to assist you, Sir.",
        "Systems initialised. Standing by.",
    ]

    def __init__(
        self,
        on_command: Callable[[str], str],
        *,
        stt: Optional[WhisperSTT] = None,
        tts: Optional[PyTTSTTS] = None,
        wake_word: Optional[WakeWordDetector] = None,
    ) -> None:
        self._on_command = on_command
        self._stt = stt or WhisperSTT()
        self._tts = tts or PyTTSTTS()
        self._wake_word = wake_word or create_wake_word_detector()

        self._state = PipelineState.STOPPED
        self._state_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._boot_index = 0

        log.info("VoicePipeline created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._state != PipelineState.STOPPED:
            log.warning("VoicePipeline already running")
            return

        self._set_state(PipelineState.IDLE)

        # Pre-load STT model in background so first listen() is fast
        threading.Thread(target=self._stt.load_model, daemon=True, name="stt-loader").start()

        # Start wake word listener
        self._wake_word.start(on_detected=self._on_wake_word)

        # Start main loop
        threading.Thread(target=self._main_loop, daemon=True, name="voice-pipeline").start()

        # JARVIS boot sequence
        play_activation_sound()
        time.sleep(0.4)
        boot_msg = self._BOOT_LINES[self._boot_index % len(self._BOOT_LINES)]
        self._tts.speak_async(boot_msg, add_personality=False)
        self._boot_index += 1

        log.info("VoicePipeline started")

    def stop(self) -> None:
        self._set_state(PipelineState.STOPPED)
        self._wake_word.stop()
        self._wake_event.set()
        self._tts.stop()
        log.info("VoicePipeline stopped")

    @property
    def state(self) -> PipelineState:
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_state(self, state: PipelineState) -> None:
        with self._state_lock:
            old = self._state
            self._state = state
        if old != state:
            log.debug("Pipeline state", extra={"from": old.name, "to": state.name})

    def _on_wake_word(self) -> None:
        if self.state == PipelineState.IDLE:
            log.info("Wake word detected")
            self._wake_event.set()
        else:
            log.debug("Wake word ignored — pipeline busy", extra={"state": self.state.name})

    def _main_loop(self) -> None:
        while self.state != PipelineState.STOPPED:
            self._wake_event.wait()
            self._wake_event.clear()

            if self.state == PipelineState.STOPPED:
                break

            # ── LISTEN ───────────────────────────────────────────────
            self._set_state(PipelineState.LISTENING)
            play_ready_sound()
            time.sleep(0.1)
            self._tts.speak("Yes, Sir?", add_personality=False)

            try:
                result = self._stt.listen()
            except Exception:
                log.exception("STT error — returning to idle")
                self._tts.speak("I'm sorry, my microphone seems to be unavailable.", add_personality=False)
                self._set_state(PipelineState.IDLE)
                continue

            user_text = result.text.strip()
            if not user_text:
                self._tts.speak("I didn't catch that, Sir. Please try again.", add_personality=False)
                self._set_state(PipelineState.IDLE)
                continue

            log.info("User said", extra={"text": user_text})

            # ── PROCESS ──────────────────────────────────────────────
            self._set_state(PipelineState.PROCESSING)
            play_processing_sound()

            try:
                response = self._on_command(user_text)
            except Exception:
                log.exception("Agent error")
                response = "I'm sorry Sir, I encountered an error."

            if not response:
                response = "Done."

            # ── SPEAK ────────────────────────────────────────────────
            self._set_state(PipelineState.SPEAKING)
            play_done_sound()
            time.sleep(0.15)
            self._tts.speak(response)

            self._set_state(PipelineState.IDLE)
