"""Unit tests for voice chat benchmarking and logging infrastructure."""
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from services.conversation.src.benchmark import BenchmarkTracker
from services.logging_config import setup_app_logging


class TestBenchmarkTracker(unittest.TestCase):
    """Test suite for BenchmarkTracker metrics, rolling averages, and log persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logs_path = Path(self.temp_dir)
        self.tracker = BenchmarkTracker(logs_dir=self.logs_path, enabled=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_benchmark_disabled_does_not_log_or_track(self):
        """Verify tracker is a complete no-op when benchmark is disabled."""
        disabled_tracker = BenchmarkTracker(logs_dir=self.logs_path, enabled=False)
        self.assertFalse(disabled_tracker.enabled)

        disabled_tracker.record_tool_execution("test_tool", {}, 0.5)
        disabled_tracker.record_filler_phrase("thinking...", "test", "neutral")
        disabled_tracker.record_followup_llm(0.2, 0.5)
        metric = disabled_tracker.record_turn("hi", "hello", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

        stats = disabled_tracker.get_cumulative_stats()
        self.assertEqual(stats["total_turns"], 0)
        self.assertEqual(stats["basic_turns_count"], 0)
        self.assertEqual(stats["tool_turns_count"], 0)

        self.assertEqual(disabled_tracker.format_turn_card(metric), "")
        self.assertEqual(disabled_tracker.format_averages_summary(), "")
        self.assertEqual(disabled_tracker.log_session_summary(), "")

        # Files must not have been created by disabled tracker
        jsonl_path = self.logs_path / "benchmarks.jsonl"
        self.assertFalse(jsonl_path.exists())

    def test_record_basic_conversation_turn(self):
        """Verify recording a basic turn (without tools) updates stats and logs."""
        metric = self.tracker.record_turn(
            user_text="What's the capital of France?",
            assistant_text="The capital of France is Paris.",
            gate_dur=0.35,
            first_token_dur=0.28,
            full_llm_dur=0.65,
            tts_first_dur=0.11,
            e2e_dur=0.86,
            total_dur=1.45,
            speech_duration=1.20,
        )

        self.assertEqual(metric.turn_index, 1)
        self.assertEqual(metric.turn_type, "basic_conversation")
        self.assertEqual(len(metric.tools), 0)

        stats = self.tracker.get_cumulative_stats()
        self.assertEqual(stats["total_turns"], 1)
        self.assertEqual(stats["basic_turns_count"], 1)
        self.assertEqual(stats["tool_turns_count"], 0)
        self.assertAlmostEqual(stats["basic_averages"]["first_token_dur"], 0.28)
        self.assertAlmostEqual(stats["basic_averages"]["e2e_dur"], 0.86)

        # Verify jsonl line was written to disk
        jsonl_path = self.logs_path / "benchmarks.jsonl"
        self.assertTrue(jsonl_path.exists())
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["user_text"], "What's the capital of France?")
        self.assertEqual(lines[0]["turn_type"], "basic_conversation")
        self.assertIn("cumulative_averages", lines[0])

        # Verify benchmark.log was written
        bench_log_path = self.logs_path / "benchmark.log"
        self.assertTrue(bench_log_path.exists())
        content = bench_log_path.read_text(encoding="utf-8")
        self.assertIn("BENCHMARK RECORD — Turn #1 [BASIC CONVERSATION]", content)
        self.assertIn("CUMULATIVE BENCHMARK AVERAGES", content)

    def test_record_tool_call_turn(self):
        """Verify recording a tool turn properly accounts for tool durations and follow-up LLM."""
        # 1. Simulate in-flight tool execution
        tool_metric = self.tracker.record_tool_execution(
            name="duckduckgo_search",
            arguments={"query": "latest news in AI"},
            duration=0.45,
            fallback=False,
            server="duckduckgo",
            status="SUCCESS",
            result_len=1200,
            result_preview="Latest AI headlines...",
        )
        self.assertEqual(tool_metric.duration_ms, 450.0)

        # 2. Simulate filler phrase and follow-up LLM stream
        self.tracker.record_filler_phrase(
            text="Searching for it.",
            category="search",
            emotion="curious",
        )
        self.tracker.record_followup_llm(
            first_token_dur=0.25,
            full_dur=0.80,
        )

        # 3. Finalize turn
        metric = self.tracker.record_turn(
            user_text="What's the latest news in AI?",
            assistant_text="Here are the latest AI headlines...",
            gate_dur=0.40,
            first_token_dur=0.30,  # tool decision latency
            full_llm_dur=0.35,
            tts_first_dur=0.12,
            e2e_dur=1.72,
            total_dur=2.60,
            speech_duration=1.50,
        )

        self.assertEqual(metric.turn_index, 1)
        self.assertEqual(metric.turn_type, "tool_call")
        self.assertEqual(len(metric.tools), 1)
        self.assertEqual(metric.filler_phrase, "Searching for it.")
        self.assertEqual(metric.followup_first_token_dur, 0.25)
        self.assertEqual(metric.followup_full_dur, 0.80)

        stats = self.tracker.get_cumulative_stats()
        self.assertEqual(stats["total_turns"], 1)
        self.assertEqual(stats["basic_turns_count"], 0)
        self.assertEqual(stats["tool_turns_count"], 1)
        self.assertAlmostEqual(stats["tool_averages"]["tool_execution_dur"], 0.45)
        self.assertAlmostEqual(stats["tool_averages"]["e2e_dur"], 1.72)

        # Check per-tool breakdown stats
        self.assertIn("duckduckgo_search", stats["per_tool_stats"])
        tool_stat = stats["per_tool_stats"]["duckduckgo_search"]
        self.assertEqual(tool_stat["count"], 1)
        self.assertAlmostEqual(tool_stat["total_sec"], 0.45)

        # Check formatted card output
        summary = self.tracker.format_averages_summary()
        self.assertIn("TOOL CALLING (1 turn):", summary)
        self.assertIn("duckduckgo_search", summary)

    def test_mixed_turns_running_averages(self):
        """Verify running averages calculate accurately across interleaved basic and tool turns."""
        # Turn 1: Basic
        self.tracker.record_turn(
            user_text="Hi",
            assistant_text="Hello!",
            gate_dur=0.30,
            first_token_dur=0.20,
            full_llm_dur=0.50,
            tts_first_dur=0.10,
            e2e_dur=0.80,
            total_dur=1.20,
        )

        # Turn 2: Basic
        self.tracker.record_turn(
            user_text="How are you?",
            assistant_text="I'm doing well!",
            gate_dur=0.40,
            first_token_dur=0.40,
            full_llm_dur=0.70,
            tts_first_dur=0.12,
            e2e_dur=1.00,
            total_dur=1.50,
        )

        # Turn 3: Tool
        self.tracker.record_tool_execution(
            name="calculator",
            arguments={"expression": "2+2"},
            duration=0.15,
        )
        self.tracker.record_turn(
            user_text="Calculate 2+2",
            assistant_text="4",
            gate_dur=0.35,
            first_token_dur=0.30,
            full_llm_dur=0.35,
            tts_first_dur=0.10,
            e2e_dur=0.90,
            total_dur=1.30,
        )

        stats = self.tracker.get_cumulative_stats()
        self.assertEqual(stats["total_turns"], 3)
        self.assertEqual(stats["basic_turns_count"], 2)
        self.assertEqual(stats["tool_turns_count"], 1)

        # Basic turn averages: TTFT = (0.20 + 0.40)/2 = 0.30, E2E = (0.80 + 1.00)/2 = 0.90
        self.assertAlmostEqual(stats["basic_averages"]["first_token_dur"], 0.30)
        self.assertAlmostEqual(stats["basic_averages"]["e2e_dur"], 0.90)

        # Tool turn averages: Tool exec = 0.15
        self.assertAlmostEqual(stats["tool_averages"]["tool_execution_dur"], 0.15)


class TestLoggingConfig(unittest.TestCase):
    """Test setup_app_logging and FlushingFileHandler."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logs_path = Path(self.temp_dir)
        self.handlers = []

    def tearDown(self):
        for h in self.handlers:
            try:
                h.close()
                logging.getLogger().removeHandler(h)
                logging.getLogger("waffle.benchmark").removeHandler(h)
            except Exception:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_app_logging_with_benchmark_enabled(self):
        self.handlers = setup_app_logging(self.logs_path, benchmark_enabled=True)
        self.assertEqual(len(self.handlers), 2)

        test_logger = logging.getLogger("voicechat.test")
        test_logger.info("Test message for voicechat log")

        bench_logger = logging.getLogger("waffle.benchmark")
        bench_logger.info("Test benchmark message")

        voicechat_file = self.logs_path / "voicechat.log"
        benchmark_file = self.logs_path / "benchmark.log"

        self.assertTrue(voicechat_file.exists())
        self.assertTrue(benchmark_file.exists())

        v_content = voicechat_file.read_text(encoding="utf-8")
        b_content = benchmark_file.read_text(encoding="utf-8")

        self.assertIn("Test message for voicechat log", v_content)
        self.assertIn("Test benchmark message", b_content)

    def test_setup_app_logging_with_benchmark_disabled(self):
        self.handlers = setup_app_logging(self.logs_path, benchmark_enabled=False)
        self.assertEqual(len(self.handlers), 1)

        test_logger = logging.getLogger("voicechat.test_disabled")
        test_logger.info("Normal message")

        voicechat_file = self.logs_path / "voicechat.log"
        benchmark_file = self.logs_path / "benchmark.log"

        self.assertTrue(voicechat_file.exists())
        self.assertFalse(benchmark_file.exists())


class TestLLMProviderBenchmarkIntegration(unittest.TestCase):
    """Test LLMProvider integration with benchmark metrics."""

    def test_tool_followup_llm_benchmarking_integration(self):
        """Verify that LLMProvider._execute_tools_and_stream_reply calculates durations and logs benchmarks without NameError."""
        from unittest.mock import MagicMock, patch
        from models import Message
        from services.llm.src.llm_provider import LLMCallbacks, LLMProvider, LLMProviderConfig

        class MockCallbacks(LLMCallbacks):
            def __init__(self):
                self.started = False
                self.completed = False
                self.followup_timings = None
                self.updates = []

            def on_generation_started(self):
                self.started = True

            def on_generation_updated(self, text: str):
                self.updates.append(text)

            def on_generation_completed(self, text: str):
                self.completed = True

            def record_followup_llm(self, first_token_dur: float, full_dur: float):
                self.followup_timings = (first_token_dur, full_dur)

        mock_tool_mgr = MagicMock()
        mock_tool_mgr.execute_tool.return_value = "Sunny, 20C"
        mock_tool_mgr.last_execution_info = {"duration": 0.1, "fallback": False, "server": "mcp"}
        provider = LLMProvider(LLMProviderConfig(tools_enabled=True), tool_manager=mock_tool_mgr)
        callbacks = MockCallbacks()

        # Mock streaming response from Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": "Here is "}}),
            json.dumps({"message": {"content": "the weather."}}),
        ]
        provider._client.send = MagicMock(return_value=mock_response)

        tool_calls = [{"function": {"name": "duckduckgo_search", "arguments": {"query": "weather"}}}]
        messages = [Message(role="user", content="What is the weather?")]

        with patch("services.llm.src.llm_provider.is_benchmark_enabled", return_value=True):
            # This must execute without NameError (fu_full_dur, fu_first_tok_dur)
            provider._execute_tools_and_stream_reply(messages, tool_calls, callbacks)

        self.assertTrue(callbacks.completed)
        self.assertIsNotNone(callbacks.followup_timings)
        self.assertGreaterEqual(callbacks.followup_timings[0], 0.0)
        self.assertGreaterEqual(callbacks.followup_timings[1], 0.0)


if __name__ == "__main__":
    unittest.main()

