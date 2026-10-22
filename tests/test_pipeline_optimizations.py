"""Unit tests for low-latency voice pipeline improvements.

Covers:
1. Unified In-Process Pipeline & Zero-Copy Shared Memory
2. Sentence Gating Strategy (First-Chunk Fast Path & Adaptive Clause Chunking)
3. VAD Thresholds, Pre-Roll Ring Buffer & Resampling Efficiency
4. LLM Sliding Context Window & Voice Formatting Prompt
5. Instant Interruption (Barge-In) & Acoustic Echo Cancellation
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from models import Message
from services.audio.src.audio_player import AudioPlayer
from services.audio_processing.src import (
    AcousticEchoCanceller,
    AECConfig,
    BargeInController,
    VADConfig,
    VADListener,
    VADService,
    resample_audio,
)
from services.conversation.src.manager import ConversationManager
from services.conversation.src.sentence_gate import ClauseChunker, SentenceGate
from services.llm.src.llm_provider import LLMProvider, LLMProviderConfig
from services.llm.src.llm_service import LLMService
from services.shared_memory import SharedAudioBuffer


class TestSentenceGatingAndClauseChunker(unittest.TestCase):
    def test_clause_chunker_first_chunk_fast_path(self):
        """First chunk of an assistant response should fire on 3-5 words."""
        chunker = ClauseChunker(first_chunk_min_words=3, clause_min_words=5)

        # First incoming tokens: "Sure, here are three things, and we will begin."
        text = "Sure, here are three things, and we will begin."
        chunks = chunker.process_stream(text)

        # Must fire immediately on first clause comma (5 words)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Sure, here are three things,")

    def test_clause_chunker_intermediate_clause_markers(self):
        """Should trigger on intermediate punctuation (, ; : —) and coordinating conjunctions."""
        chunker = ClauseChunker(first_chunk_min_words=3, clause_min_words=5)

        text = "First we will analyze the data; then we will review the charts."
        chunks = chunker.process_stream(text)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "First we will analyze the data;")

        # Finalize returns remainder
        remainder = chunker.finalize(text)
        self.assertEqual(remainder, "then we will review the charts.")

    def test_clause_chunker_conjunction_split(self):
        """Should trigger before conjunction words (because, but, and) after sufficient words."""
        chunker = ClauseChunker(first_chunk_min_words=3, clause_min_words=5)
        text = "I am very pleased to assist you because this is an important topic."
        chunks = chunker.process_stream(text)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "I am very pleased to assist you")

        remainder = chunker.finalize(text)
        self.assertEqual(remainder, "because this is an important topic.")


class TestVADAndResampling(unittest.TestCase):
    def test_vad_config_tuned_defaults(self):
        """VADConfig should default to 200ms silence timeout (180-220ms range) and 200ms pre-roll."""
        config = VADConfig()
        self.assertGreaterEqual(config.min_silence_duration_ms, 180)
        self.assertLessEqual(config.min_silence_duration_ms, 220)
        self.assertEqual(config.pre_speech_padding_ms, 200)

    def test_vad_listener_notifications(self):
        """VADService should dispatch on_speech_started and on_speech_ended."""
        class MockBackend:
            def __init__(self, events):
                self.events = iter(events)

            def process(self, frame):
                return next(self.events, None)

            def reset(self):
                pass

        class MockListener:
            def __init__(self):
                self.started = 0
                self.ended = 0

            def on_speech_started(self):
                self.started += 1

            def on_speech_ended(self):
                self.ended += 1

        listener = MockListener()
        backend = MockBackend([None, {"start": 512}, None, {"end": 1024}])
        service = VADService(backend=backend)
        service.add_listener(listener)
        service.start()

        frame = np.ones(512, dtype=np.float32)
        service.process(frame, 16000)
        self.assertEqual(listener.started, 0)

        # Second frame triggers start
        service.process(frame, 16000)
        self.assertEqual(listener.started, 1)

        # Third frame active speech
        service.process(frame, 16000)
        self.assertEqual(listener.ended, 0)

        # Fourth frame triggers end
        service.process(frame, 16000)
        self.assertEqual(listener.ended, 1)

    def test_resample_audio_polyphase(self):
        """resample_audio should preserve sample length and frequency ratio accurately."""
        # 48 kHz to 16 kHz (ratio 1/3)
        audio_48k = np.ones(4800, dtype=np.float32)
        resampled_16k = resample_audio(audio_48k, 48000, 16000)
        self.assertEqual(len(resampled_16k), 1600)
        self.assertEqual(resampled_16k.dtype, np.float32)


class TestLLMContextAndVoiceFormatting(unittest.TestCase):
    def test_sliding_context_window_in_manager(self):
        """ConversationManager should prune history beyond max_history_turns."""
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        manager = ConversationManager(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            max_history_turns=3,  # max 6 messages
        )

        for i in range(10):
            manager._trigger_llm(f"User message {i}")
            manager.on_generation_completed(f"Assistant reply {i}")

        # With max_history_turns=3, history must hold at most 6 messages (3 turns)
        self.assertLessEqual(len(manager._history), 6)
        self.assertEqual(manager._history[-1].content, "Assistant reply 9")
        self.assertEqual(manager._history[-2].content, "User message 9")

    def test_speech_targeted_system_prompt(self):
        """LLM system prompt must forbid markdown, tables, asterisks, and emojis."""
        prompt = LLMProvider._SYSTEM_PROMPT.lower()
        self.assertIn("plain spoken english words only", prompt)
        self.assertIn("markdown", prompt)
        self.assertIn("emojis", prompt)
        self.assertIn("asterisk", prompt)

    def test_llm_provider_sliding_window_guard(self):
        """LLMProvider should clamp messages payload to last 16 messages."""
        provider = LLMProvider(LLMProviderConfig())
        messages = [Message(role="user", content=f"msg {i}") for i in range(30)]
        req = provider._build_request(messages)

        # 1 system message + 16 conversational messages
        self.assertEqual(len(req["messages"]), 17)
        self.assertEqual(req["messages"][-1]["content"], "msg 29")

    def test_conversation_manager_preserves_tool_calls_in_history(self):
        """ConversationManager should record tool calls and tool outputs in history."""
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        manager = ConversationManager(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
        )

        manager._trigger_llm("What is the weather in Delhi?")
        manager.on_tool_started("duckduckgo_search", {"query": "weather in New Delhi"})
        manager.on_tool_completed("duckduckgo_search", "Sunny, 28C", duration=0.2)
        manager.on_generation_completed("It is sunny and 28 degrees in Delhi.")

        # History should have: user utterance, assistant tool call, tool response, final assistant speech
        roles = [m.role for m in manager.history]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertIsNotNone(manager.history[1].tool_calls)
        self.assertEqual(manager.history[1].tool_calls[0]["function"]["name"], "duckduckgo_search")
        self.assertEqual(manager.history[2].content, "Sunny, 28C")
        self.assertEqual(manager.history[3].content, "It is sunny and 28 degrees in Delhi.")

    def test_llm_provider_preserves_tool_messages_and_tool_calls(self):
        """LLMProvider._build_request should serialize tool roles and tool_calls properly."""
        provider = LLMProvider(LLMProviderConfig())
        messages = [
            Message(role="user", content="Check weather"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"function": {"name": "duckduckgo_search", "arguments": {"query": "weather"}}}],
            ),
            Message(role="tool", content="Sunny, 28C"),
            Message(role="assistant", content="The weather is sunny."),
        ]
        req = provider._build_request(messages)

        serialized = req["messages"]
        # Index 0 is system message
        self.assertEqual(serialized[1]["role"], "user")
        self.assertEqual(serialized[2]["role"], "assistant")
        self.assertEqual(
            serialized[2]["tool_calls"][0]["function"]["name"], "duckduckgo_search"
        )
        self.assertEqual(serialized[3]["role"], "tool")
        self.assertEqual(serialized[3]["content"], "Sunny, 28C")
        self.assertEqual(serialized[4]["role"], "assistant")


class TestBargeInAndEchoCancellation(unittest.TestCase):
    def test_barge_in_controller_interrupts_player_and_llm(self):
        """BargeInController should interrupt player, conversation, and tts when speech begins."""
        mock_player = MagicMock()
        mock_player.is_playing = True
        mock_manager = MagicMock()
        mock_manager._pending_generation = True
        mock_tts = MagicMock()

        controller = BargeInController(
            player=mock_player,
            manager=mock_manager,
            tts=mock_tts,
        )

        controller.on_speech_started()

        mock_player.interrupt.assert_called_once()
        mock_manager.barge_in.assert_called_once()
        mock_tts.interrupt.assert_called_once()

    def test_acoustic_echo_canceller(self):
        """AcousticEchoCanceller should feed reference and process mic audio without crashing."""
        aec = AcousticEchoCanceller(AECConfig(filter_length=64))
        ref_audio = np.random.randn(512).astype(np.float32)
        mic_audio = ref_audio * 0.5 + np.random.randn(512).astype(np.float32) * 0.1

        aec.feed_reference(ref_audio, 16000)
        cleaned = aec.process(mic_audio, 16000)

        self.assertEqual(len(cleaned), 512)
        self.assertEqual(cleaned.dtype, np.float32)


class TestSharedMemoryBuffer(unittest.TestCase):
    def test_shared_memory_write_read(self):
        """SharedAudioBuffer should allow zero-copy raw PCM writes and reads."""
        buf = SharedAudioBuffer(name="test_voicechat_shm", capacity_samples=2048, create=True)
        try:
            samples = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
            buf.write(samples, 16000)

            read_back, rate = buf.read_latest(512)
            self.assertEqual(rate, 16000)
            np.testing.assert_allclose(read_back, samples, rtol=1e-5, atol=1e-5)
        finally:
            buf.close()


if __name__ == "__main__":
    unittest.main()
