"""Unit tests for services/api/app.py FastAPI HTTP endpoints and services/api/webrtc_handler.py."""
from __future__ import annotations

import os
# Ensure testing mode is activated so lifespan does not start real hardware/neural models
os.environ["TESTING"] = "1"

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi.testclient import TestClient

from services.api.app import (
    _audio_to_wav_bytes,
    _parse_cors_origins,
    app,
    runtime,
)
from services.api.webrtc_handler import _ice_servers


class TestAPIHelpers(unittest.TestCase):
    def test_parse_cors_origins(self) -> None:
        self.assertEqual(
            _parse_cors_origins("http://localhost:3000, http://127.0.0.1:5173"),
            ["http://localhost:3000", "http://127.0.0.1:5173"],
        )
        self.assertEqual(_parse_cors_origins(""), ["*"])

    def test_audio_to_wav_bytes(self) -> None:
        audio = np.sin(np.linspace(0, 2 * np.pi, 160)).astype(np.float32)
        wav_bytes = _audio_to_wav_bytes(audio, 16000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_bytes[:16])

    def test_ice_servers_parsing(self) -> None:
        with patch.dict(
            os.environ,
            {"VOICECHAT_ICE_SERVERS": "stun:stun1.l.google.com:19302, stun:stun2.l.google.com:19302"},
        ):
            servers = _ice_servers()
            self.assertEqual(len(servers), 2)
            self.assertEqual(servers[0].urls, "stun:stun1.l.google.com:19302")

        with patch.dict(
            os.environ,
            {"VOICECHAT_ICE_SERVERS": '[{"urls": ["stun:stun.custom.com:3478"]}]'},
        ):
            servers = _ice_servers()
            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0].urls, ["stun:stun.custom.com:3478"])

        with patch.dict(os.environ, {"VOICECHAT_ICE_SERVERS": ""}):
            self.assertEqual(_ice_servers(), [])


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        from unittest.mock import PropertyMock
        from services.audio.src.audio_player import AudioPlayer

        with patch.object(runtime, "running", True), patch.object(
            AudioPlayer, "is_playing", new_callable=PropertyMock, return_value=False
        ):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["running"])
            self.assertFalse(data["audio_playing"])


    def test_llm_generate_endpoint_success(self) -> None:
        with patch("services.api.app._OneShotGenerationListener") as mock_listener_cls:
            mock_listener = MagicMock()
            mock_listener.wait.return_value = True
            mock_listener.result = "Generated AI text."
            mock_listener_cls.return_value = mock_listener

            with patch.object(runtime.llm, "generate") as mock_generate:
                response = self.client.post(
                    "/llm/generate",
                    json={
                        "messages": [{"role": "user", "content": "Hello!"}],
                        "timeout_seconds": 5.0,
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["text"], "Generated AI text.")
                mock_generate.assert_called_once()

    def test_tts_speak_endpoint(self) -> None:
        with patch.object(runtime.tts, "speak") as mock_speak:
            response = self.client.post("/tts/speak", json={"text": "Hello world."})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "queued"})
            mock_speak.assert_called_once_with("Hello world.")

    def test_tts_synthesize_endpoint(self) -> None:
        audio = np.zeros(240, dtype=np.float32)
        with patch.object(runtime.tts, "synthesize_direct", return_value=(audio, 24000)):
            response = self.client.post("/tts/synthesize", json={"text": "Synthesize this."})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertTrue(response.content.startswith(b"RIFF"))

    def test_stt_push_audio_endpoint(self) -> None:
        with patch.object(runtime.stt, "push") as mock_push:
            response = self.client.post(
                "/stt/push",
                json={"sample_rate": 16000, "samples": [0.0, 0.1, -0.1]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "queued"})
            mock_push.assert_called_once()

    def test_stt_transcripts_endpoints(self) -> None:
        from services.api.app import TranscriptEvent

        mock_event = TranscriptEvent(
            kind="completed",
            text="Hello there",
            created_at="2026-10-23T00:00:00Z",
        )
        with patch.object(runtime.transcript_collector, "snapshot", return_value=[mock_event]):
            response = self.client.get("/stt/transcripts")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()), 1)
            self.assertEqual(response.json()[0]["text"], "Hello there")

        with patch.object(runtime.transcript_collector, "clear") as mock_clear:
            response = self.client.delete("/stt/transcripts")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "cleared"})
            mock_clear.assert_called_once()

    def test_conversation_endpoints(self) -> None:
        from models import Message

        with patch.object(runtime.conversation, "submit_user_text") as mock_submit:
            response = self.client.post("/conversation/respond", json={"text": "User says hi"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "queued"})
            mock_submit.assert_called_once_with("User says hi")

        from unittest.mock import PropertyMock
        from services.conversation.src.manager import ConversationManager

        with patch.object(
            ConversationManager,
            "history",
            new_callable=PropertyMock,
            return_value=[Message(role="user", content="Hi"), Message(role="assistant", content="Hello!")],
        ):
            response = self.client.get("/conversation/history")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()), 2)

        with patch.object(runtime.conversation, "reset") as mock_reset:
            response = self.client.delete("/conversation/history")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "cleared"})
            mock_reset.assert_called_once()


    def test_conversation_barge_in_endpoint(self) -> None:
        with patch.object(runtime.player, "interrupt") as mock_p_interrupt, patch.object(
            runtime.conversation, "barge_in"
        ) as mock_c_barge_in:
            response = self.client.post("/conversation/barge-in")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "interrupted"})
            mock_p_interrupt.assert_called_once()
            mock_c_barge_in.assert_called_once()


if __name__ == "__main__":
    unittest.main()
