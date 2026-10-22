"""Unit tests for services/api: FastAPI routes, audio conversion, and ServiceRuntime."""
from __future__ import annotations

import io
import queue
import wave
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from models import Message
from services.api.app import (
    _OneShotGenerationListener,
    _TranscriptCollector,
    _audio_to_wav_bytes,
    _parse_cors_origins,
    app,
    runtime,
)
from services.stt.src.stt_service import AudioQueueFull
from services.tts.src.tts_types import SpeechQueueFull


class TestAPIUtilities(unittest.TestCase):
    def test_parse_cors_origins(self) -> None:
        """CORS origins parser handles comma separated values and empty defaults."""
        self.assertEqual(_parse_cors_origins("http://localhost:3000, http://example.com"), ["http://localhost:3000", "http://example.com"])
        self.assertEqual(_parse_cors_origins(""), ["*"])
        self.assertEqual(_parse_cors_origins("  "), ["*"])

    def test_audio_to_wav_bytes(self) -> None:
        """Converts float32 audio numpy array to valid playable 16-bit PCM WAV bytes."""
        audio = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        wav_bytes = _audio_to_wav_bytes(audio, 16000)

        self.assertGreater(len(wav_bytes), 44)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnframes(), 1000)

    def test_one_shot_generation_listener(self) -> None:
        """_OneShotGenerationListener tracks text and unblocks on completion."""
        listener = _OneShotGenerationListener()
        self.assertFalse(listener._done.is_set())

        listener.on_generation_started()
        listener.on_generation_updated("Hello ")
        self.assertEqual(listener.result, "Hello ")

        listener.on_generation_completed("Hello world")
        self.assertEqual(listener.result, "Hello world")
        self.assertTrue(listener._done.is_set())
        self.assertTrue(listener.wait(0.1))

    def test_transcript_collector(self) -> None:
        """_TranscriptCollector records events and supports snapshot/clear."""
        collector = _TranscriptCollector()
        self.assertEqual(collector.snapshot(), [])

        collector.on_transcript_started("Hi")
        collector.on_transcript_updated("Hi there")
        collector.on_transcript_completed("Hi there friend")

        events = collector.snapshot()
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].kind, "started")
        self.assertEqual(events[1].kind, "updated")
        self.assertEqual(events[2].kind, "completed")

        collector.clear()
        self.assertEqual(collector.snapshot(), [])


class TestAPIRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_health_endpoint(self) -> None:
        """GET /health returns application status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("running", data)
        self.assertIn("api_host", data)
        self.assertIn("api_port", data)
        self.assertIn("audio_playing", data)

    def test_audio_status_endpoint(self) -> None:
        """GET /audio/status reports whether player is currently active."""
        mock_player = MagicMock()
        mock_player.is_playing = False
        with patch.object(runtime, "player", mock_player):
            response = self.client.get("/audio/status")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"is_playing": False})

    def test_conversation_barge_in(self) -> None:
        """POST /conversation/barge-in interrupts player, conversation, and TTS."""
        with patch.object(runtime.player, "interrupt") as mock_p, \
             patch.object(runtime.conversation, "barge_in") as mock_c, \
             patch.object(runtime.tts, "interrupt") as mock_t:
            response = self.client.post("/conversation/barge-in")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "interrupted"})
            mock_p.assert_called_once()
            mock_c.assert_called_once()
            mock_t.assert_called_once()

    def test_conversation_respond(self) -> None:
        """POST /conversation/respond sends text to conversation manager."""
        with patch.object(runtime.conversation, "submit_user_text") as mock_submit:
            response = self.client.post("/conversation/respond", json={"text": "Hello assistant"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "queued"})
            mock_submit.assert_called_once_with("Hello assistant")

    def test_conversation_history_get_and_clear(self) -> None:
        """GET and DELETE /conversation/history manage conversation turn history."""
        mock_msg = Message(role="user", content="Hi")
        mock_conv = MagicMock()
        mock_conv.history = [mock_msg]
        with patch.object(runtime, "conversation", mock_conv):
            get_resp = self.client.get("/conversation/history")
            self.assertEqual(get_resp.status_code, 200)
            self.assertEqual(get_resp.json(), [{"role": "user", "content": "Hi"}])

            del_resp = self.client.delete("/conversation/history")
            self.assertEqual(del_resp.status_code, 200)
            self.assertEqual(del_resp.json(), {"status": "cleared"})
            mock_conv.reset.assert_called_once()

    def test_stt_transcripts_get_and_clear(self) -> None:
        """GET and DELETE /stt/transcripts inspect and flush collected events."""
        with patch.object(runtime.transcript_collector, "snapshot", return_value=[]), \
             patch.object(runtime.transcript_collector, "clear") as mock_clear:
            get_resp = self.client.get("/stt/transcripts")
            self.assertEqual(get_resp.status_code, 200)
            self.assertEqual(get_resp.json(), [])

            del_resp = self.client.delete("/stt/transcripts")
            self.assertEqual(del_resp.status_code, 200)
            self.assertEqual(del_resp.json(), {"status": "cleared"})
            mock_clear.assert_called_once()

    def test_stt_push_success_and_overflow(self) -> None:
        """POST /stt/push forwards audio samples, returns 429 when queue is full."""
        with patch.object(runtime.stt, "push") as mock_push:
            ok_resp = self.client.post("/stt/push", json={"sample_rate": 16000, "samples": [0.1, 0.2]})
            self.assertEqual(ok_resp.status_code, 200)
            self.assertEqual(ok_resp.json(), {"status": "queued"})

            mock_push.side_effect = AudioQueueFull("Full")
            fail_resp = self.client.post("/stt/push", json={"sample_rate": 16000, "samples": [0.1, 0.2]})
            self.assertEqual(fail_resp.status_code, 429)

    def test_tts_speak_success_and_overflow(self) -> None:
        """POST /tts/speak queues speech text, returns 429 when queue is full."""
        with patch.object(runtime.tts, "speak") as mock_speak:
            ok_resp = self.client.post("/tts/speak", json={"text": "Hello world"})
            self.assertEqual(ok_resp.status_code, 200)
            self.assertEqual(ok_resp.json(), {"status": "queued"})

            mock_speak.side_effect = SpeechQueueFull("Full")
            fail_resp = self.client.post("/tts/speak", json={"text": "Hello world"})
            self.assertEqual(fail_resp.status_code, 429)

    def test_tts_synthesize_endpoint(self) -> None:
        """POST /tts/synthesize returns synthesized WAV audio."""
        fake_audio = np.zeros(1024, dtype=np.float32)
        with patch.object(runtime.tts, "synthesize_direct", return_value=(fake_audio, 24000)):
            resp = self.client.post("/tts/synthesize", json={"text": "Synthesize this"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("content-type"), "audio/wav")
            self.assertGreater(len(resp.content), 44)

    def test_llm_generate_endpoint_success(self) -> None:
        """POST /llm/generate dispatches prompt generation and returns text."""
        def fake_generate(messages):
            for listener in list(runtime.llm._listeners):
                listener.on_generation_completed("Generated response text")

        with patch.object(runtime.llm, "generate", side_effect=fake_generate):
            resp = self.client.post(
                "/llm/generate",
                json={"messages": [{"role": "user", "content": "What is AI?"}], "timeout_seconds": 5.0},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {"text": "Generated response text"})

    def test_llm_generate_endpoint_queue_full(self) -> None:
        """POST /llm/generate returns 429 when LLM queue is full."""
        with patch.object(runtime.llm, "generate", side_effect=queue.Full):
            resp = self.client.post(
                "/llm/generate",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
            self.assertEqual(resp.status_code, 429)


if __name__ == "__main__":
    unittest.main()
