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
    extract_latest_emotion,
    extract_turn_emotion,
    normalize_emotion,
    parse_sentence_emotion,
    split_by_emotion,
    strip_emotion_tags,
)
from services.llm.src.llm_provider import LLMProvider, LLMProviderConfig
from services.tts.src.emotion_tts_service import EmotionTTSService, EmotionVoiceWorker
from services.tts.src.tts_types import TTSServiceError


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

    def test_bare_parenthetical_prose_is_preserved(self) -> None:
        emotion, clean = parse_sentence_emotion("I felt a bit (sad) about it.", current_emotion="neutral")
        self.assertEqual(emotion, "neutral")
        self.assertEqual(clean, "I felt a bit (sad) about it.")

    def test_split_by_emotion_preserves_mid_chunk_transitions(self) -> None:
        segments = split_by_emotion("[sad] Oh no! [happy] Wait, I fixed it!", current_emotion="neutral")
        self.assertEqual(segments, [("sad", "Oh no!"), ("happy", "Wait, I fixed it!")])

    def test_extract_latest_emotion(self) -> None:
        self.assertIsNone(extract_latest_emotion("No tags yet"))
        self.assertEqual(extract_latest_emotion("[sad] Sorry. [happy] Good news!"), "happy")


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

    @patch("services.tts.src.emotion_tts_service.load_local_first")
    def test_emotion_tts_service_preserves_fifo_order_and_interrupt(self, mock_load_first) -> None:
        """Multi-emotion speech requests must synthesize in strict FIFO order and support interrupt/is_busy."""
        import numpy as np

        class FakeChunk:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.zeros(64, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            for emo in ("neutral", "happy", "sad", "angry"):
                (p / f"{emo}_female.safetensors").write_text("fake")

            spoken_order: list[tuple[str, str]] = []
            mock_engine = MagicMock()
            mock_engine.get_state_for_audio_prompt.side_effect = lambda path: {"path": Path(path).stem}
            mock_engine.sample_rate = 24000

            def fake_stream(state, text):
                spoken_order.append((state["path"], text))
                yield FakeChunk()

            mock_engine.generate_audio_stream.side_effect = fake_stream
            mock_load_first.return_value = mock_engine

            service = EmotionTTSService(gender="female", models_dir=p, offload_idle_seconds=0)
            try:
                service.start()
                service.offload_emotion("sad")
                service.speak("First sad sentence.", emotion="sad")
                service.speak("Second neutral sentence.", emotion="neutral")
                service.speak("Third happy sentence.", emotion="happy")

                deadline = time.time() + 3.0
                while service.is_busy and time.time() < deadline:
                    time.sleep(0.02)

                self.assertFalse(service.is_busy)
                self.assertEqual(
                    spoken_order,
                    [
                        ("sad_female", "First sad sentence."),
                        ("neutral_female", "Second neutral sentence."),
                        ("happy_female", "Third happy sentence."),
                    ],
                )
            finally:
                service.stop()

    @patch("services.tts.src.emotion_tts_service.load_local_first")
    def test_emotion_tts_service_start_raises_on_missing_model(self, mock_load_first) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_load_first.return_value = MagicMock()
            service = EmotionTTSService(gender="female", models_dir=tmp_dir, offload_idle_seconds=0)
            with self.assertRaises(TTSServiceError):
                service.start()
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

    def test_manager_preserves_multi_emotion_in_single_chunk(self) -> None:
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        manager = EmotionConversationManager(stt=mock_stt, llm=mock_llm, tts=mock_tts)
        manager.start()
        try:
            manager.submit_user_text("Tell me something")
            manager.on_generation_started()
            manager.on_generation_updated("[sad] I am sorry that happened. [happy] Good news is on the way! ")
            self.assertEqual(
                mock_tts.speak.call_args_list,
                [
                    unittest.mock.call("I am sorry that happened.", emotion="sad"),
                    unittest.mock.call("Good news is on the way!", emotion="happy"),
                ],
            )
            self.assertEqual(manager.active_emotion, "happy")
        finally:
            manager.stop()


class TestLLMProviderEmotionPrompt(unittest.TestCase):
    def test_emotion_prompt_in_config(self) -> None:
        config = LLMProviderConfig(system_prompt=LLMProvider.EMOTION_SYSTEM_PROMPT)
        self.assertEqual(config.system_prompt, LLMProvider.EMOTION_SYSTEM_PROMPT)

        provider = LLMProvider(config)
        req = provider._build_request([])
        self.assertEqual(req["messages"][0]["content"], LLMProvider.EMOTION_SYSTEM_PROMPT)


class TestEmotionConversationUI(unittest.TestCase):
    def test_ui_instantiation_and_listener_interface(self) -> None:
        """Verify EmotionConversationUI instantiates and implements all AudioPlaybackListener methods."""
        from services.conversation.src.emotion_ui import EmotionConversationUI

        ui = EmotionConversationUI(gender="female", verbose=True)
        self.assertEqual(ui.gender, "Female")

        # Test STT events
        ui.on_transcript_started("hello")
        ui.on_transcript_updated("hello there")
        ui.on_transcript_completed("hello there")
        ui.on_user_turn("hello there")

        # Test LLM events
        ui.on_generation_started()
        ui.on_generation_updated("[happy] Glad to assist!")
        ui.on_generation_completed("[happy] Glad to assist! How are you?")

        # Test Audio Playback Listener events
        ui.on_playback_started()
        ui.on_playback_text_updated("Glad to assist!")
        ui.on_playback_completed()

        # Verify benchmark was recorded
        self.assertEqual(len(ui.benchmarks), 1)
        bm = ui.benchmarks[0]
        self.assertIn("e2e_dur", bm)
        self.assertIn("gate_dur", bm)

        # Test rendering (including brackets that would break unescaped Rich markup)
        ui.on_user_turn("Test [unclosed bracket and [/red]")
        ui.on_generation_started()
        ui.on_generation_updated("[sad] Sorry. [happy] Yay [excited]!")
        ui.on_generation_completed("[sad] Sorry. [happy] Yay [excited]!")
        ui.on_playback_completed()
        self.assertEqual(ui._status, "READY")

        from rich.console import Console
        console = Console(record=True, width=100)
        layout = ui.render(0, console=console)
        self.assertIsNotNone(layout)
        self.assertIsNotNone(layout["benchmark"])
        console.print(layout)


if __name__ == "__main__":
    unittest.main()
