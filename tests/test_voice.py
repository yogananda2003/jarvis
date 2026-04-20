"""
Tests for the voice subsystem (STT, TTS, wake word, pipeline).
All hardware (mic, speakers, Whisper model) is mocked.

Run:
    pytest tests/test_voice.py -v
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# STT Tests
# ---------------------------------------------------------------------------

class TestWhisperSTT:
    def _make_stt(self):
        from voice.stt import WhisperSTT
        stt = WhisperSTT()
        stt._model = MagicMock()     # skip actual model load
        return stt

    def test_transcribe_array_returns_result(self):
        from voice.stt import TranscriptionResult
        stt = self._make_stt()
        stt._model.transcribe.return_value = {
            "text": "  open chrome  ",
            "language": "en",
            "segments": [{"avg_logprob": -0.2}],
        }

        audio = np.zeros(16_000, dtype=np.float32)
        result = stt.transcribe_array(audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "open chrome"
        assert result.language == "en"
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_audio_returns_empty_text(self):
        stt = self._make_stt()
        audio = np.array([], dtype=np.float32)
        result = stt.transcribe_array(audio)
        assert result.text == ""
        assert result.duration_s == 0.0

    def test_load_model_called_once(self):
        from voice.stt import WhisperSTT
        stt = WhisperSTT()

        with patch("voice.stt.whisper.load_model") as mock_load:
            mock_load.return_value = MagicMock()
            stt.load_model()
            stt.load_model()   # second call should be no-op

        mock_load.assert_called_once()

    def test_transcribe_strips_whitespace(self):
        stt = self._make_stt()
        stt._model.transcribe.return_value = {
            "text": "  hello jarvis  ",
            "language": "en",
            "segments": [],
        }
        audio = np.zeros(8_000, dtype=np.float32)
        result = stt.transcribe_array(audio)
        assert result.text == "hello jarvis"

    def test_latency_is_positive(self):
        stt = self._make_stt()
        stt._model.transcribe.return_value = {
            "text": "test", "language": "en", "segments": [],
        }
        audio = np.zeros(8_000, dtype=np.float32)
        result = stt.transcribe_array(audio)
        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# TTS Tests
# ---------------------------------------------------------------------------

class TestPyTTSTTS:
    def _make_tts(self):
        from voice.tts import PyTTSTTS
        tts = PyTTSTTS()
        mock_engine = MagicMock()
        tts._engine = mock_engine
        return tts, mock_engine

    def test_speak_calls_say_and_runAndWait(self):
        tts, engine = self._make_tts()
        tts.speak("Hello world")
        engine.say.assert_called_once_with("Hello world")
        engine.runAndWait.assert_called_once()

    def test_speak_empty_string_is_noop(self):
        tts, engine = self._make_tts()
        tts.speak("")
        engine.say.assert_not_called()

    def test_speak_whitespace_only_is_noop(self):
        tts, engine = self._make_tts()
        tts.speak("   ")
        engine.say.assert_not_called()

    def test_set_rate(self):
        tts, engine = self._make_tts()
        tts.set_rate(200)
        engine.setProperty.assert_called_with("rate", 200)
        assert tts._rate == 200

    def test_set_volume_clamps(self):
        tts, engine = self._make_tts()
        tts.set_volume(1.5)     # above max
        engine.setProperty.assert_called_with("volume", 1.0)

        tts.set_volume(-0.5)    # below min
        engine.setProperty.assert_called_with("volume", 0.0)

    def test_stop_calls_engine_stop(self):
        tts, engine = self._make_tts()
        tts._speaking.set()
        tts.stop()
        engine.stop.assert_called_once()

    def test_sentence_splitting(self):
        from voice.tts import PyTTSTTS
        sentences, remainder = PyTTSTTS._split_sentences(
            "Hello world. How are you? Fine! Still going"
        )
        assert sentences == ["Hello world.", "How are you?", "Fine!"]
        assert remainder == "Still going"

    def test_sentence_splitting_no_boundary(self):
        from voice.tts import PyTTSTTS
        sentences, remainder = PyTTSTTS._split_sentences("No boundary here")
        assert sentences == []
        assert remainder == "No boundary here"

    def test_speak_stream(self):
        from voice.stt import TranscriptionResult
        tts, engine = self._make_tts()

        class FakeChunk:
            def __init__(self, content, done=False):
                self.content = content
                self.done = done

        chunks = [
            FakeChunk("Hello world."),
            FakeChunk(" How are you?"),
            FakeChunk(" I am fine!", done=True),
        ]
        tts.speak_stream(iter(chunks))
        assert engine.say.call_count >= 1


# ---------------------------------------------------------------------------
# Wake Word Tests
# ---------------------------------------------------------------------------

class TestWakeWordDetector:
    def test_start_stop(self):
        from voice.wake_word import WhisperWakeWord
        detector = WhisperWakeWord(keyword="jarvis")
        detector._load_model = MagicMock()    # skip model load
        detector._listen_loop = MagicMock()   # skip actual loop

        callback = MagicMock()
        detector.start(on_detected=callback)
        assert detector.is_running
        detector.stop()
        assert not detector.is_running

    def test_fire_calls_callback(self):
        from voice.wake_word import WhisperWakeWord
        detector = WhisperWakeWord(keyword="jarvis")
        callback = MagicMock()
        detector._on_detected = callback
        detector._fire()
        callback.assert_called_once()

    def test_fire_with_no_callback_is_safe(self):
        from voice.wake_word import WhisperWakeWord
        detector = WhisperWakeWord(keyword="jarvis")
        detector._on_detected = None
        detector._fire()   # should not raise

    def test_factory_returns_whisper_by_default(self):
        from voice.wake_word import create_wake_word_detector, WhisperWakeWord
        with patch("voice.wake_word.settings") as mock_settings:
            mock_settings.voice.wake_word.engine = "whisper"
            mock_settings.voice.wake_word.keyword = "jarvis"
            mock_settings.voice.wake_word.sensitivity = 0.5
            detector = create_wake_word_detector()
        assert isinstance(detector, WhisperWakeWord)

    def test_factory_returns_porcupine(self):
        from voice.wake_word import create_wake_word_detector, PorcupineWakeWord
        detector = create_wake_word_detector(engine="porcupine")
        assert isinstance(detector, PorcupineWakeWord)

    def test_double_start_is_noop(self):
        from voice.wake_word import WhisperWakeWord
        detector = WhisperWakeWord(keyword="jarvis")
        detector._listen_loop = MagicMock()
        detector.start()
        thread_before = detector._thread
        detector.start()    # should not create a second thread
        assert detector._thread is thread_before
        detector.stop()


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestVoicePipeline:
    def _make_pipeline(self, handler=None):
        from voice.pipeline import VoicePipeline, PipelineState

        mock_stt = MagicMock()
        mock_stt.listen.return_value = MagicMock(text="open chrome")
        mock_stt.load_model = MagicMock()

        mock_tts = MagicMock()
        mock_tts.speak = MagicMock()
        mock_tts.speak_async = MagicMock()
        mock_tts.stop = MagicMock()

        mock_wake = MagicMock()
        mock_wake.start = MagicMock()
        mock_wake.stop = MagicMock()
        mock_wake.is_running = False

        on_command = handler or (lambda text: f"Handled: {text}")
        pipeline = VoicePipeline(
            on_command=on_command,
            stt=mock_stt,
            tts=mock_tts,
            wake_word=mock_wake,
        )
        return pipeline, mock_stt, mock_tts, mock_wake

    def test_initial_state_is_stopped(self):
        from voice.pipeline import PipelineState
        pipeline, *_ = self._make_pipeline()
        assert pipeline.state == PipelineState.STOPPED

    def test_start_changes_state_to_idle(self):
        from voice.pipeline import PipelineState
        pipeline, _, _, mock_wake = self._make_pipeline()
        pipeline.start()
        assert pipeline.state in (PipelineState.IDLE, PipelineState.LISTENING)
        pipeline.stop()

    def test_stop_calls_wake_word_stop(self):
        pipeline, _, _, mock_wake = self._make_pipeline()
        pipeline.start()
        pipeline.stop()
        mock_wake.stop.assert_called_once()

    def test_on_wake_word_triggers_listening(self):
        """Simulate wake word → verify STT is called."""
        from voice.pipeline import PipelineState
        import time

        results = []
        def handler(text):
            results.append(text)
            return "OK"

        pipeline, mock_stt, mock_tts, _ = self._make_pipeline(handler)
        pipeline.start()
        time.sleep(0.1)

        # Simulate wake word detection
        pipeline._on_wake_word()
        time.sleep(0.5)

        pipeline.stop()
        assert len(results) > 0 or mock_stt.listen.called

    def test_empty_transcription_returns_to_idle(self):
        from voice.pipeline import PipelineState
        import time

        pipeline, mock_stt, mock_tts, _ = self._make_pipeline()
        mock_stt.listen.return_value = MagicMock(text="")

        pipeline.start()
        time.sleep(0.1)
        pipeline._on_wake_word()
        time.sleep(0.5)
        pipeline.stop()

        # Should have spoken the "didn't catch that" message
        spoken = [call.args[0] for call in mock_tts.speak.call_args_list]
        assert any("catch" in s.lower() for s in spoken)
