"""Comprehensive unit tests for emotional voice cloning, parsing, multi-threaded TTS, and offloading."""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import constants
from constants import (
    AVAILABLE_EMOTIONS,
    AVAILABLE_GENDERS,
    CLONED_VOICES_DIR,
    get_emotion_voice_path,
)
from services.conversation.src.emotion_manager import EmotionConversationManager
from services.conversation.src.emotion_parser import (
    extract_turn_emotion,
    normalize_emotion,
    parse_sentence_emotion,
    strip_emotion_tags,
)
from services.llm.src.llm_provider import LLMProvider, LLMProviderConfig
from services.tts.src.emotion_tts_service import EmotionTTSService, EmotionVoiceWorker


class TestEmotionParser(unittest.TestCase):
    def test_normalize_emotion(self) -> None:
        self.assertEqual(normalize_emotion("happy"), "happy")
        self.assertEqual(normalize_emotion("SAD"), "sad")
        self.assertEqual(normalize_emotion("  angry  "), "angry")
        self.assertEqual(normalize_emotion("neutral"), "neutral")
        self.assertEqual(normalize_emotion("excited", default="neutral"), "neutral")
        self.assertEqual(normalize_emotion("", default="happy"), "happy")
        self.assertEqual(normalize_emotion(None, default="neutral"), "neutral")

    def test_parse_sentence_emotion_simple_bracket(self) -> None:
        emotion, clean = parse_sentence_emotion("[happy] Good morning! How can I help you?")
        self.assertEqual(emotion, "happy")
        self.assertEqual(clean, "Good morning! How can I help you?")

    def test_parse_sentence_emotion_case_insensitive(self) -> None:
        emotion, clean = parse_sentence_emotion("[ANGRY] That is totally unacceptable!")
        self.assertEqual(emotion, "angry")
        self.assertEqual(clean, "That is totally unacceptable!")

    def test_parse_sentence_emotion_prefix_syntax(self) -> None:
        emotion, clean = parse_sentence_emotion("[emotion: sad] I am sorry to hear that.")
        self.assertEqual(emotion, "sad")
        self.assertEqual(clean, "I am sorry to hear that.")

    def test_parse_sentence_emotion_xml_syntax(self) -> None:
        emotion, clean = parse_sentence_emotion("<emotion>neutral</emotion> System is ready.")
        self.assertEqual(emotion, "neutral")
        self.assertEqual(clean, "System is ready.")

    def test_parse_sentence_emotion_no_tag_inherits_current(self) -> None:
        emotion, clean = parse_sentence_emotion("I have no tag here.", current_emotion="happy")
        self.assertEqual(emotion, "happy")
        self.assertEqual(clean, "I have no tag here.")

    def test_extract_turn_emotion(self) -> None:
        self.assertEqual(extract_turn_emotion("[happy] Wonderful news today!"), "happy")
        self.assertEqual(extract_turn_emotion("[sad] Sad news today."), "sad")
        self.assertEqual(extract_turn_emotion("No emotion tag here.", default="neutral"), "neutral")

    def test_strip_emotion_tags(self) -> None:
        text = "[happy] First sentence. [sad] Second sentence. <emotion>angry</emotion> Done!"
        cleaned = strip_emotion_tags(text)
        self.assertEqual(cleaned, "First sentence. Second sentence. Done!")


class TestEmotionSafetensorsAndConstants(unittest.TestCase):
    def test_available_constants(self) -> None:
        self.assertEqual(AVAILABLE_EMOTIONS, ("neutral", "happy", "sad", "angry"))
        self.assertEqual(AVAILABLE_GENDERS, ("female", "male"))

    def test_generated_safetensors_exist_in_cloned_voices(self) -> None:
        """Verify that all 8 generated emotion safetensors exist and are valid sizes (>1MB)."""
        for gender in ("female", "male"):
            for emotion in ("neutral", "happy", "sad", "angry"):
                path = get_emotion_voice_path(gender, emotion)
                self.assertTrue(
                    path.is_file(),
                    f"Expected safetensors file for {gender}_{emotion} at {path}",
                )
                self.assertGreater(path.stat().st_size, 1_000_000)

    def test_get_emotion_voice_path_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            (p / "happy_female.safetensors").write_text("fake")
            resolved = get_emotion_voice_path("female", "happy", directory=p)
            self.assertEqual(resolved, (p / "happy_female.safetensors").resolve())


class TestEmotionTTSService(unittest.TestCase):
    def test_invalid_gender_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            EmotionTTSService(gender="robot")

    def test_worker_lifecycle_in_memory_and_offload(self) -> None:
        """Verify that a worker manages state in memory, offloads, and auto-reloads."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "happy_female.safetensors"
            model_file.write_text("model weights")

            mock_engine = MagicMock()
            fake_state = {"latent": [1, 2, 3]}
            mock_engine.get_state_for_audio_prompt.return_value = fake_state

            callbacks = {
                "started": MagicMock(),
                "chunk": MagicMock(),
                "completed": MagicMock(),
            }

            worker = EmotionVoiceWorker(
                emotion="happy",
                gender="female",
                model_path=model_file,
                engine_getter=lambda: mock_engine,
                engine_lock=threading.Lock(),
                dispatch_callbacks=callbacks,
            )

            # Initially not loaded
            self.assertFalse(worker.is_loaded)

            # Load into memory
            worker.load()
            self.assertTrue(worker.is_loaded)
            self.assertEqual(worker._voice_state, fake_state)

            # Offload from memory
            offloaded = worker.offload()
            self.assertTrue(offloaded)
            self.assertFalse(worker.is_loaded)
            self.assertIsNone(worker._voice_state)

            # Second offload does nothing
            self.assertFalse(worker.offload())

    @patch("services.tts.src.emotion_tts_service.load_local_first")
    def test_emotion_tts_service_threads_and_routing(self, mock_load_first) -> None:
        """Test EmotionTTSService initializes dedicated threads per emotion and dispatches correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            for emo in ("neutral", "happy", "sad", "angry"):
                (p / f"{emo}_female.safetensors").write_text("fake")

            mock_engine = MagicMock()
            mock_engine.get_state_for_audio_prompt.return_value = {"state": "loaded"}
            mock_engine.sample_rate = 24000
            import torch
            mock_engine.generate_audio_stream.return_value = [torch.zeros(512)]
            mock_load_first.return_value = mock_engine

            service = EmotionTTSService(
                gender="female",
                models_dir=p,
                offload_idle_seconds=1.0,
            )

            try:
                service.start()
                self.assertTrue(service.is_started)

                # Ensure 4 workers were created for female only
                statuses = service.get_statuses()
                self.assertEqual(set(statuses.keys()), {"neutral", "happy", "sad", "angry"})

                # Test speak dispatch
                mock_listener = MagicMock()
                service.add_listener(mock_listener)

                service.speak("Hello happy world!", emotion="happy")
                time.sleep(0.3)

                # Check that synthesis started was dispatched
                self.assertTrue(mock_listener.on_synthesis_started.called)

                # Test idle offloading
                # Backdate last_used_time for sad worker
                service._workers["sad"].last_used_time = time.time() - 10.0
                offloaded = service.offload_idle()
                self.assertIn("sad", offloaded)
                self.assertFalse(service._workers["sad"].is_loaded)

                # Neutral should NOT be offloaded by offload_idle
                self.assertTrue(service._workers["neutral"].is_loaded)

            finally:
                service.stop()
                self.assertFalse(service.is_started)


class TestEmotionConversationManager(unittest.TestCase):
    def test_manager_parses_emotion_and_dispatches_clean_speech(self) -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        manager = EmotionConversationManager(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
        )

        manager.start()
        try:
            # User submits text, triggering LLM generation
            manager.submit_user_text("Hello assistant!")

            # Simulate LLM generation events
            manager.on_generation_started()

            # Stream chunk with [happy] tag and sentence boundary
            manager.on_generation_updated("[happy] It is wonderful to meet you today! ")
            mock_tts.speak.assert_called_with("It is wonderful to meet you today!", emotion="happy")
            self.assertEqual(manager.active_emotion, "happy")

            # Stream completion with clean history
            manager.on_generation_completed("[happy] It is wonderful to meet you today! Have a great day.")
            history = manager.history
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].role, "user")
            self.assertEqual(history[1].role, "assistant")
            self.assertEqual(history[1].content, "It is wonderful to meet you today! Have a great day.")
        finally:
            manager.stop()


class TestLLMProviderEmotionPrompt(unittest.TestCase):
    def test_emotion_prompt_in_config(self) -> None:
        config = LLMProviderConfig(system_prompt=LLMProvider.EMOTION_SYSTEM_PROMPT)
        self.assertEqual(config.system_prompt, LLMProvider.EMOTION_SYSTEM_PROMPT)

        provider = LLMProvider(config)
        req = provider._build_request([])
        self.assertEqual(req["messages"][0]["content"], LLMProvider.EMOTION_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
