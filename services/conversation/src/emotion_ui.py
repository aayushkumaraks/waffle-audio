"""Rich-based terminal UI for Emotional Voice Assistant with real-time emotion badges, thread status, and benchmark profiling."""
from __future__ import annotations

import threading
import time
from typing import Any, Literal

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from services.audio.src import AudioPlaybackListener
from services.conversation.src.emotion_parser import (
    extract_latest_emotion,
    extract_turn_emotion,
    normalize_emotion,
    split_by_emotion,
    strip_emotion_tags,
)
from services.llm.src import LLMListener
from services.stt.src import STTListener

StatusType = Literal["READY", "LISTENING", "PROCESSING"]

# Targets defined in README.md benchmarks (based on warm turn measurements)
README_TARGET_GATE = 0.40          # Sentence gate wait (0.4s complete to ~1.4s trailing)
README_TARGET_FIRST_TOKEN = 0.45    # LLM first token
README_TARGET_LLM_FULL = 1.47       # LLM full reply
README_TARGET_TTS_FIRST = 1.82      # Pocket TTS opening sentence synthesis
README_TARGET_E2E = 2.10           # End-to-end latency (speech end to sound from speaker)

EMOTION_STYLES: dict[str, dict[str, str]] = {
    "neutral": {"color": "cyan", "icon": "✦", "bg": "on blue"},
    "happy": {"color": "bright_yellow", "icon": "🌟", "bg": "on yellow"},
    "sad": {"color": "blue", "icon": "🌧", "bg": "on navy_blue"},
    "angry": {"color": "bright_red", "icon": "⚡", "bg": "on dark_red"},
}


def _format_diff(observed: float, target: float) -> str:
    diff = observed - target
    if diff <= 0.05:
        return f"[bold green]{diff:+.2f}s (Faster! ✓)[/bold green]"
    elif diff <= 0.35:
        return f"[green]{diff:+.2f}s (On Target)[/green]"
    else:
        return f"[yellow]{diff:+.2f}s (Slower)[/yellow]"


def _merge_emotion_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge consecutive (emotion, text) segments sharing the same emotion."""
    merged: list[tuple[str, str]] = []
    for emo, seg_text in segments:
        clean_seg = seg_text.strip()
        if not clean_seg:
            continue
        norm_emo = normalize_emotion(emo, default="neutral")
        if merged and merged[-1][0] == norm_emo:
            merged[-1] = (norm_emo, f"{merged[-1][1]} {clean_seg}")
        else:
            merged.append((norm_emo, clean_seg))
    return merged


class EmotionConversationUI(STTListener, LLMListener, AudioPlaybackListener):
    """Rich UI controller displaying dynamic emotion badges, active threads, memory status, and latency benchmarks."""

    def __init__(self, gender: str = "female", verbose: bool = False) -> None:
        self.gender = gender.capitalize()
        self._verbose = verbose
        self._lock = threading.Lock()
        self._status: StatusType = "READY"
        self._active_emotion: str = "neutral"
        self._worker_statuses: dict[str, dict[str, Any]] = {}

        self._history: list[dict[str, Any]] = []
        self._current_user: str = ""
        self._current_assistant: str = ""
        self._current_assistant_segments: list[tuple[str, str]] = []
        self._pending_assistant_segments: list[tuple[str, str]] = []
        self._user_fragments: list[str] = []
        self._pending_user_turn: str = ""
        self._llm_generated_text: str = ""
        self._is_generating: bool = False
        self._is_playing_audio: bool = False
        self._last_error: str = ""

        # Latency tracking timestamps for verbose mode
        self._t_speech_start: float | None = None
        self._t_speech_end: float | None = None
        self._t_gate_fired: float | None = None
        self._t_first_token: float | None = None
        self._t_llm_done: float | None = None
        self._t_playback_start: float | None = None
        self._t_playback_end: float | None = None

        self._latest_benchmark: dict | None = None
        self._benchmarks: list[dict] = []

        self._wave_frames = [
            " ▂▃▅▆▇▆▅▃ ",
            "▃▅▆▇█▇▆▅▃ ",
            "▅▆▇███▇▆▅ ",
            "▇███▓▒░▒▓█",
            "▅▆▇███▇▆▅ ",
            "▃▅▆▇█▇▆▅▃ ",
            " ▂▃▅▆▇▆▅▃ ",
            "  ▂▃▄▅▄▃▂ ",
        ]
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def set_emotion(self, emotion: str) -> None:
        with self._lock:
            self._active_emotion = emotion.strip().lower()

    def update_worker_statuses(self, statuses: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self._worker_statuses = statuses

    def sync_audio_playback(self, is_playing: bool) -> None:
        with self._lock:
            was_playing = self._is_playing_audio
            self._is_playing_audio = is_playing
            # Only return to READY when playback actually transitions True -> False;
            # otherwise keep PROCESSING visible while waiting for Ollama's first token.
            if was_playing and not is_playing and not self._is_generating and self._status == "PROCESSING":
                self._status = "READY"

    # ------------------------------------------------------------------
    # STTListener
    # ------------------------------------------------------------------

    def on_transcript_started(self, text: str) -> None:
        with self._lock:
            if self._status == "READY":
                self._user_fragments.clear()
                self._pending_user_turn = ""
            self._status = "LISTENING"
            prefix = " ".join(self._user_fragments)
            clean_text = text.strip()
            self._current_user = f"{prefix} {clean_text}".strip() if prefix else clean_text
            if self._t_speech_start is None:
                self._t_speech_start = time.perf_counter()

    def on_transcript_updated(self, text: str) -> None:
        with self._lock:
            self._status = "LISTENING"
            prefix = " ".join(self._user_fragments)
            clean_text = text.strip()
            self._current_user = f"{prefix} {clean_text}".strip() if prefix else clean_text

    def on_transcript_completed(self, text: str) -> None:
        clean_text = text.strip()
        with self._lock:
            if clean_text:
                self._user_fragments.append(clean_text)
                self._current_user = " ".join(self._user_fragments)
                self._pending_user_turn = self._current_user
                self._t_speech_end = time.perf_counter()

    def _commit_pending_assistant_locked(self) -> None:
        """Commit the completed/spoken assistant turn to history exactly once."""
        raw_segments = self._pending_assistant_segments or self._current_assistant_segments
        merged = _merge_emotion_segments(raw_segments)
        if merged:
            full_text = " ".join(seg_text for _, seg_text in merged)
            # Pick first non-neutral emotion as primary label when multiple emotions appear
            primary_emotion = next((emo for emo, _ in merged if emo != "neutral"), merged[0][0])
            self._history.append({
                "role": "assistant",
                "text": full_text,
                "emotion": primary_emotion,
                "segments": merged,
            })
        self._current_assistant = ""
        self._current_assistant_segments = []
        self._pending_assistant_segments = []

    def on_user_turn(self, text: str) -> None:
        clean_text = text.strip()
        with self._lock:
            self._commit_pending_assistant_locked()
            self._status = "PROCESSING"
            self._last_error = ""
            self._t_gate_fired = time.perf_counter()
            self._t_first_token = None
            self._t_llm_done = None
            self._t_playback_start = None
            self._t_playback_end = None
            final_user = (
                clean_text
                or self._pending_user_turn.strip()
                or " ".join(self._user_fragments).strip()
                or self._current_user.strip()
            )
            if final_user:
                self._history.append({"role": "user", "text": final_user, "emotion": ""})
            self._pending_user_turn = ""
            self._user_fragments.clear()
            self._current_user = ""
            self._current_assistant = ""
            self._current_assistant_segments = []
            self._pending_assistant_segments = []
            self._llm_generated_text = ""

    # ------------------------------------------------------------------
    # LLMListener
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        with self._lock:
            self._is_generating = True
            self._status = "PROCESSING"
            self._current_assistant = ""
            self._current_assistant_segments = []
            self._pending_assistant_segments = []
            self._active_emotion = "neutral"

    def on_generation_updated(self, text: str) -> None:
        with self._lock:
            now = time.perf_counter()
            if self._t_first_token is None:
                self._t_first_token = now
            self._status = "PROCESSING"
            self._llm_generated_text = text
            # Only set active emotion from LLM stream before playback begins;
            # once audio is playing, playback callbacks own the live emotion badge.
            if not self._is_playing_audio:
                detected = extract_latest_emotion(text)
                if detected is not None:
                    self._active_emotion = detected

    def on_generation_completed(self, text: str) -> None:
        with self._lock:
            now = time.perf_counter()
            self._is_generating = False
            self._t_llm_done = now
            raw = text.strip() or self._llm_generated_text.strip()
            self._llm_generated_text = raw
            self._pending_assistant_segments = _merge_emotion_segments(
                split_by_emotion(raw, current_emotion="neutral")
            )
            if not self._pending_assistant_segments and not self._is_playing_audio:
                self._status = "READY"

    def on_generation_failed(self, error: str) -> None:
        with self._lock:
            self._is_generating = False
            self._current_assistant = ""
            self._current_assistant_segments = []
            self._pending_assistant_segments = []
            self._last_error = error.strip() or "LLM generation failed"
            self._status = "READY"

    # ------------------------------------------------------------------
    # AudioPlaybackListener
    # ------------------------------------------------------------------

    def on_playback_started(self) -> None:
        with self._lock:
            now = time.perf_counter()
            if self._t_playback_start is None:
                self._t_playback_start = now
            self._is_playing_audio = True

    def on_playback_text_updated(
        self, text: str, emotion: str = "", segment_text: str = ""
    ) -> None:
        with self._lock:
            self._status = "PROCESSING"
            if emotion:
                self._active_emotion = normalize_emotion(emotion, default=self._active_emotion)
            self._current_assistant = text
            seg = strip_emotion_tags(segment_text or text)
            if seg:
                emo = self._active_emotion
                if segment_text:
                    self._current_assistant_segments = _merge_emotion_segments(
                        self._current_assistant_segments + [(emo, seg)]
                    )
                else:
                    self._current_assistant_segments = [(emo, seg)]

    def on_playback_completed(self) -> None:
        with self._lock:
            now = time.perf_counter()
            self._is_playing_audio = False
            self._t_playback_end = now
            self._commit_pending_assistant_locked()
            if not self._is_generating and self._status == "PROCESSING":
                self._status = "READY"

            # Record benchmark metrics
            last_user_text = "Unknown"
            for item in reversed(self._history):
                if item["role"] == "user":
                    last_user_text = item["text"]
                    break

            last_ast_text = "Unknown"
            for item in reversed(self._history):
                if item["role"] == "assistant":
                    last_ast_text = item["text"]
                    break

            t_speech_end = self._t_speech_end or (self._t_gate_fired or now)
            t_gate = self._t_gate_fired or t_speech_end
            gate_dur = max(0.0, t_gate - t_speech_end)

            t_first_tok = self._t_first_token or t_gate
            first_tok_dur = max(0.0, t_first_tok - t_gate)

            t_llm_done = self._t_llm_done or (t_first_tok + 1.0)
            llm_full_dur = max(0.0, t_llm_done - t_gate)

            t_play_start = self._t_playback_start or now
            tts_first_dur = max(0.0, t_play_start - t_gate)
            e2e_dur = max(0.0, t_play_start - t_speech_end)
            total_dur = max(0.0, now - t_speech_end)

            benchmark = {
                "user_text": last_user_text,
                "assistant_text": last_ast_text,
                "gate_dur": gate_dur,
                "first_token_dur": first_tok_dur,
                "full_llm_dur": llm_full_dur,
                "tts_first_dur": tts_first_dur,
                "e2e_dur": e2e_dur,
                "total_dur": total_dur,
            }
            self._latest_benchmark = benchmark
            self._benchmarks.append(benchmark)

            self._t_speech_start = None
            self._t_speech_end = None
            self._t_gate_fired = None
            self._t_first_token = None
            self._t_llm_done = None
            self._t_playback_start = None
            self._t_playback_end = None

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            # Include any uncommitted completed turn (e.g. if stopped mid-playback)
            items = list(self._history)
            if self._pending_assistant_segments or self._current_assistant_segments:
                merged = _merge_emotion_segments(
                    self._pending_assistant_segments or self._current_assistant_segments
                )
                if merged:
                    primary = next((emo for emo, _ in merged if emo != "neutral"), merged[0][0])
                    items.append({
                        "role": "assistant",
                        "text": " ".join(seg_text for _, seg_text in merged),
                        "emotion": primary,
                        "segments": merged,
                    })
            return items

    @property
    def benchmarks(self) -> list[dict]:
        with self._lock:
            return list(self._benchmarks)

    # ------------------------------------------------------------------
    # UI Rendering
    # ------------------------------------------------------------------

    def render(self, frame_idx: int, console: Console | None = None) -> Layout:
        with self._lock:
            status = self._status
            current_user = self._current_user
            current_assistant = self._current_assistant
            current_assistant_segments = list(self._current_assistant_segments)
            history = list(self._history)
            active_emotion = self._active_emotion
            is_playing = self._is_playing_audio
            statuses = dict(self._worker_statuses)
            latest_bm = self._latest_benchmark
            verbose = self._verbose
            last_error = self._last_error

        layout = Layout()
        if verbose:
            layout.split_column(
                Layout(name="header", size=4),
                Layout(name="body", ratio=1),
                Layout(name="benchmark", size=9),
                Layout(name="footer", size=7),
            )
        else:
            layout.split_column(
                Layout(name="header", size=4),
                Layout(name="body", ratio=1),
                Layout(name="footer", size=7),
            )

        # Header
        e_info = EMOTION_STYLES.get(active_emotion, EMOTION_STYLES["neutral"])
        badge_color = e_info["color"]
        badge_icon = e_info["icon"]

        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", ratio=1)
        header_table.add_column(justify="center", ratio=2)
        header_table.add_column(justify="right", ratio=1)

        title = Text.assemble(
            (" WAFFLE AUDIO ", "bold white on dark_green"),
            (" Emotional Voice Assistant ", "bold cyan"),
        )
        gender_badge = Text.assemble(
            (" Voice Gender: ", "dim"),
            (f"{self.gender} ", "bold magenta"),
        )
        emotion_badge = Text.assemble(
            (" Active Emotion: ", "dim"),
            (f" [{active_emotion.upper()} {badge_icon}] ", f"bold {badge_color}"),
        )
        header_table.add_row(title, gender_badge, emotion_badge)
        layout["header"].update(Panel(header_table, border_style="cyan"))

        # Body: History + Voice Model Threads
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="conversation", ratio=2),
            Layout(name="models", ratio=1),
        )

        chat_table = Table.grid(expand=True, padding=(0, 1))
        chat_table.add_column(style="dim", width=18)
        chat_table.add_column(style="white", ratio=1)

        for msg in history[-6:]:
            if msg["role"] == "user":
                chat_table.add_row("[bold cyan]You:[/bold cyan]", f"[cyan]{escape(msg['text'])}[/cyan]")
            else:
                segments = msg.get("segments") or [(msg.get("emotion", "neutral"), msg["text"])]
                for emo, seg_text in segments:
                    e_color = EMOTION_STYLES.get(emo, EMOTION_STYLES["neutral"])["color"]
                    chat_table.add_row(
                        f"[bold green]Waffle [{emo.upper()}]:[/bold green]",
                        f"[{e_color}]{escape(seg_text)}[/{e_color}]",
                    )

        if current_user:
            chat_table.add_row(
                "[bold cyan]You (speaking):[/bold cyan]",
                f"[bold white]{escape(current_user)}[/bold white]",
            )

        if current_assistant_segments:
            for emo, seg_text in current_assistant_segments:
                e_color = EMOTION_STYLES.get(emo, EMOTION_STYLES["neutral"])["color"]
                chat_table.add_row(
                    f"[bold green]Waffle [{emo.upper()}]:[/bold green]",
                    f"[{e_color}]{escape(seg_text)}[/{e_color}]",
                )
        elif current_assistant:
            clean_ast = strip_emotion_tags(current_assistant)
            e_color = EMOTION_STYLES.get(active_emotion, EMOTION_STYLES["neutral"])["color"]
            chat_table.add_row(
                f"[bold green]Waffle [{active_emotion.upper()}]:[/bold green]",
                f"[{e_color}]{escape(clean_ast)}[/{e_color}]",
            )

        body_layout["conversation"].update(
            Panel(chat_table, title="[bold]💬 Real-Time Conversation[/bold]", border_style="dim white")
        )

        # Right pane: Emotion Voice Models & Thread Status
        models_table = Table(expand=True, show_edge=False, box=None)
        models_table.add_column("Emotion", style="bold", width=9)
        models_table.add_column("Thread / RAM", width=14)
        models_table.add_column("Status", width=12)

        for emo in ("neutral", "happy", "sad", "angry"):
            info = statuses.get(emo, {})
            is_loaded = info.get("loaded", False)
            is_synth = info.get("synthesizing", False)
            idle_s = info.get("idle_seconds", 0.0)

            emo_style = EMOTION_STYLES[emo]["color"]
            ram_label = "[green]● In Memory[/green]" if is_loaded else "[dim]○ Offloaded[/dim]"

            if is_synth:
                status_label = "[bold yellow]Synthesizing[/bold yellow]"
            elif is_loaded:
                status_label = f"[dim]Idle ({idle_s:.0f}s)[/dim]"
            else:
                status_label = "[dim]Ready (auto)[/dim]"

            models_table.add_row(f"[{emo_style}]{emo.capitalize()}[/{emo_style}]", ram_label, status_label)

        body_layout["models"].update(
            Panel(models_table, title=f"[bold]🧠 Voice Models ({self.gender})[/bold]", border_style="magenta")
        )
        layout["body"].update(body_layout)

        # Benchmark Timeline Panel (if verbose)
        if verbose:
            if latest_bm:
                tb = Table(box=None, padding=(0, 2), expand=True)
                tb.add_column("Pipeline Stage", style="bold cyan")
                tb.add_column("Observed", justify="right", style="bold white")
                tb.add_column("README Benchmark", justify="right", style="dim white")
                tb.add_column("Performance vs Benchmark", justify="left")

                tb.add_row(
                    "1. Sentence Gate",
                    f"{latest_bm['gate_dur']:.2f} s",
                    f"{README_TARGET_GATE:.2f} s",
                    _format_diff(latest_bm['gate_dur'], README_TARGET_GATE),
                )
                tb.add_row(
                    "2. LLM First Token",
                    f"{latest_bm['first_token_dur']:.2f} s",
                    f"{README_TARGET_FIRST_TOKEN:.2f} s",
                    _format_diff(latest_bm['first_token_dur'], README_TARGET_FIRST_TOKEN),
                )
                tb.add_row(
                    "3. LLM Full Reply",
                    f"{latest_bm['full_llm_dur']:.2f} s",
                    f"{README_TARGET_LLM_FULL:.2f} s",
                    _format_diff(latest_bm['full_llm_dur'], README_TARGET_LLM_FULL),
                )
                tb.add_row(
                    "4. TTS First Speech",
                    f"{latest_bm['tts_first_dur']:.2f} s",
                    f"{README_TARGET_TTS_FIRST:.2f} s",
                    _format_diff(latest_bm['tts_first_dur'], README_TARGET_TTS_FIRST),
                )
                tb.add_row(
                    "5. Total E2E Latency",
                    f"{latest_bm['e2e_dur']:.2f} s",
                    f"{README_TARGET_E2E:.2f} s",
                    _format_diff(latest_bm['e2e_dur'], README_TARGET_E2E),
                )

                bm_panel = Panel(
                    tb,
                    title=f"⏱️ Turn Latency Timeline (Last Turn E2E: {latest_bm['e2e_dur']:.2f}s)",
                    border_style="magenta",
                )
            else:
                waiting_text = Text.from_markup(
                    "  [dim italic]Awaiting first voice interaction... "
                    f"Target benchmarks from README: E2E ~{README_TARGET_E2E}s | Gate ~{README_TARGET_GATE}s | "
                    f"LLM Token ~{README_TARGET_FIRST_TOKEN}s | TTS Speech ~{README_TARGET_TTS_FIRST}s[/dim italic]"
                )
                bm_panel = Panel(
                    Align.center(waiting_text),
                    title="⏱️ Turn Latency Timeline (vs README Benchmarks)",
                    border_style="magenta",
                )
            layout["benchmark"].update(bm_panel)

        # Footer: Status bar & Waveform
        spinner = self._spinner_frames[frame_idx % len(self._spinner_frames)]
        waveform = self._wave_frames[frame_idx % len(self._wave_frames)]

        footer_table = Table.grid(expand=True)
        footer_table.add_column(justify="left", ratio=1)
        footer_table.add_column(justify="center", ratio=1)
        footer_table.add_column(justify="right", ratio=1)

        if is_playing:
            state_text = f"[bold green]{waveform} Speaking with emotion [{active_emotion.upper()}][/bold green]"
        elif status == "LISTENING":
            state_text = f"[bold red]{waveform} Listening to microphone...[/bold red]"
        elif status == "PROCESSING":
            state_text = f"[bold yellow]{spinner} Generating emotional reply...[/bold yellow]"
        elif last_error:
            state_text = f"[bold red]⚠ LLM error: {escape(last_error[:60])}[/bold red]"
        else:
            state_text = f"[dim green]● Ready (Listening for speech...)[/dim green]"

        footer_table.add_row(
            state_text,
            f"[dim]Models: 4 Threads resident / auto-offload | Engine: PocketTTS[/dim]",
            "[dim]Press Ctrl+C to stop[/dim]",
        )
        layout["footer"].update(Panel(footer_table, border_style="green"))

        return layout
