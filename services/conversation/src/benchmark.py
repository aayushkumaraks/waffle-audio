"""Centralized latency benchmarking and performance metrics tracking for Waffle Audio.

Tracks, computes, and logs granular latencies for:
- Basic voice conversations (VAD -> STT -> Sentence Gate -> LLM TTFT -> LLM Full -> TTS -> Audio Playback)
- Tool calls (Intent -> Tool Decision -> Filler Phrase -> Tool Execution -> Fallback -> Follow-up LLM -> TTS -> Audio Playback)
- Cumulative rolling averages and README target comparisons

Writes structured human-readable cards and JSONL logs directly into the logs/ directory.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# README benchmark reference targets (seconds)
README_TARGET_GATE = 0.40          # Sentence gate silence wait
README_TARGET_FIRST_TOKEN = 0.45    # LLM first token (TTFT)
README_TARGET_LLM_FULL = 1.47       # LLM full reply
README_TARGET_TTS_FIRST = 1.82      # Pocket TTS first sentence audio synthesis
README_TARGET_E2E = 2.10           # End-to-end latency (user speech end to sound from speaker)

logger = logging.getLogger(__name__)
benchmark_logger = logging.getLogger("waffle.benchmark")


def _format_diff(observed: float, target: float) -> str:
    diff = observed - target
    if diff <= 0.05:
        return f"{diff:+.2f}s (Faster! ✓)"
    elif diff <= 0.35:
        return f"{diff:+.2f}s (On Target)"
    else:
        return f"{diff:+.2f}s (Slower)"


@dataclass
class ToolExecutionMetric:
    """Record of an individual tool execution within a turn."""
    name: str
    arguments: dict[str, Any]
    duration: float              # seconds
    duration_ms: float           # milliseconds
    fallback: bool = False       # True if fell back to DuckDuckGo
    server: str = "mcp"          # Server or provider name
    status: str = "SUCCESS"      # "SUCCESS" or "ERROR"
    result_len: int = 0
    result_preview: str = ""


@dataclass
class TurnMetric:
    """Granular record of an entire conversation or tool-calling turn."""
    turn_index: int
    timestamp: str               # ISO 8601 string
    turn_type: str               # "basic_conversation" or "tool_call"
    user_text: str
    assistant_text: str

    # Pipeline timestamps / latencies (in seconds)
    speech_duration: float = 0.0
    gate_dur: float = 0.0
    first_token_dur: float = 0.0
    full_llm_dur: float = 0.0
    tts_first_dur: float = 0.0
    e2e_dur: float = 0.0
    total_dur: float = 0.0

    # Tool specific metrics
    tools: list[ToolExecutionMetric] = field(default_factory=list)
    total_tool_dur: float = 0.0
    filler_phrase: str = ""
    filler_emotion: str = ""
    filler_category: str = ""
    followup_first_token_dur: float = 0.0
    followup_full_dur: float = 0.0


class BenchmarkTracker:
    """Thread-safe benchmark collector, rolling average calculator, and logger."""

    def __init__(
        self,
        log_dir: Path | str = "logs",
        benchmark_file: str = "benchmark.log",
        jsonl_file: str = "benchmarks.jsonl",
        logs_dir: Path | str | None = None,
    ) -> None:
        target_dir = logs_dir if logs_dir is not None else log_dir
        self.log_dir = Path(target_dir)
        self.benchmark_log_path = self.log_dir / benchmark_file
        self.jsonl_path = self.log_dir / jsonl_file

        self._lock = threading.Lock()
        self._turn_counter: int = 0
        self._session_start_time: float = time.time()

        # In-flight tool execution accumulator for current turn
        self._current_turn_tools: list[ToolExecutionMetric] = []
        self._current_filler: dict[str, str] = {}
        self._current_followup_ttft: float | None = None
        self._current_followup_done: float | None = None

        # Cumulative metric aggregates
        self._turns: list[TurnMetric] = []
        self._basic_turns_count: int = 0
        self._tool_turns_count: int = 0

        # Sums for basic conversation averages
        self._sum_gate_dur: float = 0.0
        self._sum_first_token_dur: float = 0.0
        self._sum_full_llm_dur: float = 0.0
        self._sum_tts_first_dur: float = 0.0
        self._sum_e2e_dur: float = 0.0
        self._sum_total_dur: float = 0.0

        # Sums for tool calling averages
        self._sum_tool_dur: float = 0.0
        self._sum_tool_turn_e2e: float = 0.0
        self._sum_tool_turn_total: float = 0.0
        self._sum_followup_ttft: float = 0.0
        self._sum_followup_full: float = 0.0

        # Per-tool breakdown: name -> {"count": N, "total_sec": S, "min_sec": min, "max_sec": max, "fallbacks": F}
        self._per_tool_stats: dict[str, dict[str, Any]] = {}

        # Ensure log directory exists
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # In-flight Tool Accumulation
    # ------------------------------------------------------------------

    def record_tool_execution(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        duration: float,
        fallback: bool = False,
        server: str = "mcp",
        status: str = "SUCCESS",
        result_len: int = 0,
        result_preview: str = "",
    ) -> ToolExecutionMetric:
        """Record an executed tool call within the active turn."""
        metric = ToolExecutionMetric(
            name=name,
            arguments=arguments or {},
            duration=round(duration, 4),
            duration_ms=round(duration * 1000.0, 2),
            fallback=fallback,
            server=server,
            status=status,
            result_len=result_len,
            result_preview=result_preview[:120] if result_preview else "",
        )
        with self._lock:
            self._current_turn_tools.append(metric)

            # Update per-tool stats
            stats = self._per_tool_stats.setdefault(
                name,
                {"count": 0, "total_sec": 0.0, "min_sec": duration, "max_sec": duration, "fallbacks": 0},
            )
            stats["count"] += 1
            stats["total_sec"] += duration
            stats["min_sec"] = min(stats["min_sec"], duration)
            stats["max_sec"] = max(stats["max_sec"], duration)
            if fallback:
                stats["fallbacks"] += 1

        return metric

    def record_filler_phrase(
        self, text: str, category: str = "", emotion: str = ""
    ) -> None:
        """Record the spoken filler phrase for the active turn."""
        with self._lock:
            self._current_filler = {
                "text": text,
                "category": category,
                "emotion": emotion,
            }

    def record_followup_llm(
        self, first_token_dur: float, full_dur: float
    ) -> None:
        """Record follow-up LLM stream timings after tool execution."""
        with self._lock:
            self._current_followup_ttft = first_token_dur
            self._current_followup_done = full_dur

    def clear_turn_tools(self) -> None:
        """Clear active tool accumulation on turn reset or barge-in."""
        with self._lock:
            self._current_turn_tools.clear()
            self._current_filler.clear()
            self._current_followup_ttft = None
            self._current_followup_done = None

    # ------------------------------------------------------------------
    # Turn Completion & Aggregation
    # ------------------------------------------------------------------

    def record_turn(
        self,
        user_text: str,
        assistant_text: str,
        gate_dur: float,
        first_token_dur: float,
        full_llm_dur: float,
        tts_first_dur: float,
        e2e_dur: float,
        total_dur: float,
        speech_duration: float = 0.0,
    ) -> TurnMetric:
        """Finalize and record a turn with all its latency metrics, update running averages, and write to logs."""
        with self._lock:
            self._turn_counter += 1
            turn_idx = self._turn_counter
            tools_snapshot = list(self._current_turn_tools)
            filler_snapshot = dict(self._current_filler)
            followup_ttft = self._current_followup_ttft or 0.0
            followup_full = self._current_followup_done or 0.0

            # Reset in-flight tool buffers for next turn
            self._current_turn_tools.clear()
            self._current_filler.clear()
            self._current_followup_ttft = None
            self._current_followup_done = None

            is_tool_turn = len(tools_snapshot) > 0
            turn_type = "tool_call" if is_tool_turn else "basic_conversation"
            total_tool_dur = sum(t.duration for t in tools_snapshot)

            metric = TurnMetric(
                turn_index=turn_idx,
                timestamp=datetime.now(timezone.utc).isoformat(),
                turn_type=turn_type,
                user_text=user_text.strip(),
                assistant_text=assistant_text.strip(),
                speech_duration=round(speech_duration, 3),
                gate_dur=round(gate_dur, 3),
                first_token_dur=round(first_token_dur, 3),
                full_llm_dur=round(full_llm_dur, 3),
                tts_first_dur=round(tts_first_dur, 3),
                e2e_dur=round(e2e_dur, 3),
                total_dur=round(total_dur, 3),
                tools=tools_snapshot,
                total_tool_dur=round(total_tool_dur, 3),
                filler_phrase=filler_snapshot.get("text", ""),
                filler_emotion=filler_snapshot.get("emotion", ""),
                filler_category=filler_snapshot.get("category", ""),
                followup_first_token_dur=round(followup_ttft, 3),
                followup_full_dur=round(followup_full, 3),
            )

            self._turns.append(metric)

            # Update running averages
            if is_tool_turn:
                self._tool_turns_count += 1
                self._sum_tool_dur += total_tool_dur
                self._sum_tool_turn_e2e += e2e_dur
                self._sum_tool_turn_total += total_dur
                self._sum_followup_ttft += followup_ttft
                self._sum_followup_full += followup_full
            else:
                self._basic_turns_count += 1
                self._sum_gate_dur += gate_dur
                self._sum_first_token_dur += first_token_dur
                self._sum_full_llm_dur += full_llm_dur
                self._sum_tts_first_dur += tts_first_dur
                self._sum_e2e_dur += e2e_dur
                self._sum_total_dur += total_dur

        # Format and output to disk log files
        self._write_logs(metric)
        return metric

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_turn_card(self, metric: TurnMetric) -> str:
        """Create a human-readable latency and benchmark card for a single turn."""
        lines = [
            "=" * 78,
            f"📊 BENCHMARK RECORD — Turn #{metric.turn_index} [{metric.turn_type.upper().replace('_', ' ')}]",
            "=" * 78,
            f"Timestamp : {metric.timestamp}",
            f"User      : \"{metric.user_text}\"",
            f"Assistant : \"{metric.assistant_text[:120]}{'...' if len(metric.assistant_text) > 120 else ''}\"",
            "",
            "⏱️  LATENCY TIMELINE BREAKDOWN:",
        ]

        if metric.speech_duration > 0:
            lines.append(f"  • User Speech Duration       : {metric.speech_duration:5.2f}s")

        lines.append(
            f"  1. Sentence Gate Silence Wait: {metric.gate_dur:5.2f}s  "
            f"(Target: {README_TARGET_GATE:4.2f}s  |  Δ {_format_diff(metric.gate_dur, README_TARGET_GATE)})"
        )

        if metric.turn_type == "tool_call":
            lines.append(
                f"  2. Initial LLM Tool Decision : {metric.first_token_dur:5.2f}s  "
                f"(Target: {README_TARGET_FIRST_TOKEN:4.2f}s  |  Δ {_format_diff(metric.first_token_dur, README_TARGET_FIRST_TOKEN)})"
            )
            if metric.filler_phrase:
                emotion_tag = f" [{metric.filler_emotion}]" if metric.filler_emotion else ""
                lines.append(f"  3. Spoken Filler Phrase      : \"{metric.filler_phrase}\"{emotion_tag}")

            lines.append(f"  4. Executed Tools ({len(metric.tools)}):")
            for idx, tool in enumerate(metric.tools, start=1):
                fb_note = " (DuckDuckGo fallback)" if tool.fallback else ""
                status_note = f"[{tool.status}]" if tool.status != "SUCCESS" else ""
                lines.append(
                    f"     {idx}. {tool.name:20}: {tool.duration:5.2f}s ({tool.duration_ms:6.1f}ms) "
                    f"[server={tool.server}]{fb_note} {status_note}"
                )
            lines.append(f"     Total Tool Execution Time : {metric.total_tool_dur:5.2f}s")

            if metric.followup_first_token_dur > 0 or metric.followup_full_dur > 0:
                lines.append(f"  5. Follow-Up LLM First Token : {metric.followup_first_token_dur:5.2f}s")
                lines.append(f"  6. Follow-Up LLM Full Reply  : {metric.followup_full_dur:5.2f}s")

            lines.append(f"  7. First Spoken Audio Output : {metric.tts_first_dur:5.2f}s")
            lines.append("  " + "-" * 74)
            lines.append(
                f"  ★ Total E2E Latency (Speech End -> Sound) : {metric.e2e_dur:5.2f}s  "
                f"(Target: {README_TARGET_E2E:4.2f}s  |  Δ {_format_diff(metric.e2e_dur, README_TARGET_E2E)})"
            )
            lines.append(f"  ★ Total Turn Duration (Complete Speech)   : {metric.total_dur:5.2f}s")

        else:
            lines.append(
                f"  2. LLM First Token (TTFT)    : {metric.first_token_dur:5.2f}s  "
                f"(Target: {README_TARGET_FIRST_TOKEN:4.2f}s  |  Δ {_format_diff(metric.first_token_dur, README_TARGET_FIRST_TOKEN)})"
            )
            lines.append(
                f"  3. LLM Full Reply Generation : {metric.full_llm_dur:5.2f}s  "
                f"(Target: {README_TARGET_LLM_FULL:4.2f}s  |  Δ {_format_diff(metric.full_llm_dur, README_TARGET_LLM_FULL)})"
            )
            lines.append(
                f"  4. TTS First Speech Output   : {metric.tts_first_dur:5.2f}s  "
                f"(Target: {README_TARGET_TTS_FIRST:4.2f}s  |  Δ {_format_diff(metric.tts_first_dur, README_TARGET_TTS_FIRST)})"
            )
            lines.append("  " + "-" * 74)
            lines.append(
                f"  ★ Total E2E Latency (Speech End -> Sound) : {metric.e2e_dur:5.2f}s  "
                f"(Target: {README_TARGET_E2E:4.2f}s  |  Δ {_format_diff(metric.e2e_dur, README_TARGET_E2E)})"
            )
            lines.append(f"  ★ Total Turn Duration (Complete Speech)   : {metric.total_dur:5.2f}s")

        lines.append("=" * 78)
        return "\n".join(lines)

    def format_averages_summary(self) -> str:
        """Create a human-readable cumulative averages summary across the entire session."""
        with self._lock:
            total_turns = self._turn_counter
            basic_count = self._basic_turns_count
            tool_count = self._tool_turns_count

            # Calculate basic conversation averages
            avg_gate = (self._sum_gate_dur / basic_count) if basic_count > 0 else 0.0
            avg_ttft = (self._sum_first_token_dur / basic_count) if basic_count > 0 else 0.0
            avg_full_llm = (self._sum_full_llm_dur / basic_count) if basic_count > 0 else 0.0
            avg_tts_first = (self._sum_tts_first_dur / basic_count) if basic_count > 0 else 0.0
            avg_e2e = (self._sum_e2e_dur / basic_count) if basic_count > 0 else 0.0
            avg_total = (self._sum_total_dur / basic_count) if basic_count > 0 else 0.0

            # Calculate tool calling averages
            avg_tool_exec = (self._sum_tool_dur / tool_count) if tool_count > 0 else 0.0
            avg_tool_e2e = (self._sum_tool_turn_e2e / tool_count) if tool_count > 0 else 0.0
            avg_tool_total = (self._sum_tool_turn_total / tool_count) if tool_count > 0 else 0.0
            avg_followup_ttft = (self._sum_followup_ttft / tool_count) if tool_count > 0 else 0.0
            avg_followup_full = (self._sum_followup_full / tool_count) if tool_count > 0 else 0.0

            per_tool_copy = {k: v.copy() for k, v in self._per_tool_stats.items()}

        lines = [
            "=" * 78,
            f"📈 CUMULATIVE BENCHMARK AVERAGES (Session Total: {total_turns} Turns | Basic: {basic_count} | Tool: {tool_count})",
            "=" * 78,
            f"README Warm Target Reference: E2E = {README_TARGET_E2E}s | TTFT = {README_TARGET_FIRST_TOKEN}s | TTS First = {README_TARGET_TTS_FIRST}s | LLM Full = {README_TARGET_LLM_FULL}s",
            "",
            f"💬 BASIC CONVERSATION ({basic_count} turn{'s' if basic_count != 1 else ''}):",
        ]

        if basic_count > 0:
            lines.extend([
                f"  • Avg Sentence Gate Wait   : {avg_gate:5.2f}s  (Target: {README_TARGET_GATE:4.2f}s | Δ {_format_diff(avg_gate, README_TARGET_GATE)})",
                f"  • Avg LLM First Token (TTFT): {avg_ttft:5.2f}s  (Target: {README_TARGET_FIRST_TOKEN:4.2f}s | Δ {_format_diff(avg_ttft, README_TARGET_FIRST_TOKEN)})",
                f"  • Avg LLM Full Reply       : {avg_full_llm:5.2f}s  (Target: {README_TARGET_LLM_FULL:4.2f}s | Δ {_format_diff(avg_full_llm, README_TARGET_LLM_FULL)})",
                f"  • Avg TTS First Speech     : {avg_tts_first:5.2f}s  (Target: {README_TARGET_TTS_FIRST:4.2f}s | Δ {_format_diff(avg_tts_first, README_TARGET_TTS_FIRST)})",
                f"  • Avg End-to-End Latency   : {avg_e2e:5.2f}s  (Target: {README_TARGET_E2E:4.2f}s | Δ {_format_diff(avg_e2e, README_TARGET_E2E)})",
                f"  • Avg Total Turn Duration  : {avg_total:5.2f}s",
            ])
        else:
            lines.append("  (No basic conversation turns recorded yet)")

        lines.extend([
            "",
            f"🛠️  TOOL CALLING ({tool_count} turn{'s' if tool_count != 1 else ''}):",
        ])

        if tool_count > 0:
            lines.extend([
                f"  • Avg Tool Execution Time  : {avg_tool_exec:5.2f}s ({avg_tool_exec*1000.0:6.1f}ms)",
                f"  • Avg Follow-Up LLM TTFT   : {avg_followup_ttft:5.2f}s",
                f"  • Avg Follow-Up LLM Full   : {avg_followup_full:5.2f}s",
                f"  • Avg Tool Turn E2E Latency: {avg_tool_e2e:5.2f}s",
                f"  • Avg Tool Turn Total Time : {avg_tool_total:5.2f}s",
                "",
                "  Per-Tool Execution Breakdown:",
            ])
            for tool_name, stats in sorted(per_tool_copy.items()):
                c = stats["count"]
                avg_t = (stats["total_sec"] / c) if c > 0 else 0.0
                fb_str = f" [fallbacks: {stats['fallbacks']}]" if stats["fallbacks"] > 0 else ""
                lines.append(
                    f"    - {tool_name:24}: avg {avg_t:5.2f}s ({avg_t*1000:6.1f}ms) "
                    f"over {c} call{'s' if c != 1 else ''} "
                    f"(min {stats['min_sec']:5.2f}s, max {stats['max_sec']:5.2f}s){fb_str}"
                )
        else:
            lines.append("  (No tool calls recorded yet)")

        lines.append("=" * 78)
        return "\n".join(lines)

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Return raw cumulative statistics for automated testing or export."""
        with self._lock:
            basic_count = self._basic_turns_count
            tool_count = self._tool_turns_count
            return {
                "total_turns": self._turn_counter,
                "basic_turns_count": basic_count,
                "tool_turns_count": tool_count,
                "basic_averages": {
                    "gate_dur": round((self._sum_gate_dur / basic_count), 3) if basic_count > 0 else 0.0,
                    "first_token_dur": round((self._sum_first_token_dur / basic_count), 3) if basic_count > 0 else 0.0,
                    "full_llm_dur": round((self._sum_full_llm_dur / basic_count), 3) if basic_count > 0 else 0.0,
                    "tts_first_dur": round((self._sum_tts_first_dur / basic_count), 3) if basic_count > 0 else 0.0,
                    "e2e_dur": round((self._sum_e2e_dur / basic_count), 3) if basic_count > 0 else 0.0,
                    "total_dur": round((self._sum_total_dur / basic_count), 3) if basic_count > 0 else 0.0,
                },
                "tool_averages": {
                    "tool_execution_dur": round((self._sum_tool_dur / tool_count), 3) if tool_count > 0 else 0.0,
                    "followup_first_token_dur": round((self._sum_followup_ttft / tool_count), 3) if tool_count > 0 else 0.0,
                    "followup_full_dur": round((self._sum_followup_full / tool_count), 3) if tool_count > 0 else 0.0,
                    "e2e_dur": round((self._sum_tool_turn_e2e / tool_count), 3) if tool_count > 0 else 0.0,
                    "total_dur": round((self._sum_tool_turn_total / tool_count), 3) if tool_count > 0 else 0.0,
                },
                "per_tool_stats": {k: v.copy() for k, v in self._per_tool_stats.items()},
            }

    # ------------------------------------------------------------------
    # Disk Logging
    # ------------------------------------------------------------------

    def _write_logs(self, metric: TurnMetric) -> None:
        """Write turn metrics and updated averages to all target files in logs/."""
        card = self.format_turn_card(metric)
        averages = self.format_averages_summary()

        # 1. Log to main application log (logs/voicechat.log)
        logger.info("\n%s\n\n%s", card, averages)

        # 2. Log to dedicated benchmark log (logs/benchmark.log)
        try:
            with open(self.benchmark_log_path, "a", encoding="utf-8") as bf:
                bf.write(f"\n{card}\n\n{averages}\n")
                bf.flush()
        except Exception as exc:
            logger.warning("Could not write to %s: %s", self.benchmark_log_path, exc)

        # 3. Append JSONL record to logs/benchmarks.jsonl
        try:
            json_record = asdict(metric)
            json_record["cumulative_averages"] = self.get_cumulative_stats()
            with open(self.jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(json_record) + "\n")
                jf.flush()
        except Exception as exc:
            logger.warning("Could not write to %s: %s", self.jsonl_path, exc)

    def log_session_summary(self) -> str:
        """Log and return the final cumulative averages summary at shutdown."""
        summary = self.format_averages_summary()
        logger.info("\n%s", summary)
        try:
            with open(self.benchmark_log_path, "a", encoding="utf-8") as bf:
                bf.write(f"\nFINAL SESSION BENCHMARK SUMMARY:\n{summary}\n")
                bf.flush()
        except Exception:
            pass
        return summary


# Global singleton instance
_GLOBAL_TRACKER: BenchmarkTracker | None = None
_TRACKER_LOCK = threading.Lock()


def get_benchmark_tracker() -> BenchmarkTracker:
    """Return the global BenchmarkTracker instance, creating it if needed."""
    global _GLOBAL_TRACKER
    with _TRACKER_LOCK:
        if _GLOBAL_TRACKER is None:
            _GLOBAL_TRACKER = BenchmarkTracker()
        return _GLOBAL_TRACKER
