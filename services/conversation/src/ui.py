"""Rich-based terminal UI for Waffle Audio CLI with synchronized playback and latency benchmarking."""
from __future__ import annotations

import threading
import time
from typing import Literal

from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from services.audio.src import AudioPlaybackListener
from services.llm.src import LLMListener
from services.stt.src import STTListener

StatusType = Literal["READY", "LISTENING", "PROCESSING"]

# Targets defined in README.md benchmarks (based on warm turn measurements)
README_TARGET_GATE = 0.40          # Sentence gate wait (0.4s complete to ~1.4s trailing)
README_TARGET_FIRST_TOKEN = 0.45    # LLM first token
README_TARGET_LLM_FULL = 1.47       # LLM full reply
README_TARGET_TTS_FIRST = 1.82      # Pocket TTS opening sentence synthesis
README_TARGET_E2E = 2.10           # End-to-end latency (speech end to sound from speaker)


def _format_diff(observed: float, target: float) -> str:
    diff = observed - target
    if diff <= 0.05:
        return f"[bold green]{diff:+.2f}s (Faster! ✓)[/bold green]"
    elif diff <= 0.35:
        return f"[green]{diff:+.2f}s (On Target)[/green]"
    else:
        return f"[yellow]{diff:+.2f}s (Slower)[/yellow]"


class ConversationUI(STTListener, LLMListener, AudioPlaybackListener):
    """Thread-safe UI controller and renderer for the terminal voice interface."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._lock = threading.Lock()
        self._status: StatusType = "READY"
        self._history: list[dict[str, str]] = []
        self._current_user: str = ""
        self._current_assistant: str = ""
        self._pending_user_turn: str = ""
        self._llm_generated_text: str = ""
        self._is_generating: bool = False

        # Latency tracking timestamps
        self._t_speech_start: float | None = None
        self._t_speech_end: float | None = None
        self._t_gate_fired: float | None = None
        self._t_first_token: float | None = None
        self._t_llm_done: float | None = None
        self._t_playback_start: float | None = None
        self._t_playback_end: float | None = None

        self._latest_benchmark: dict | None = None
        self._benchmarks: list[dict] = []

        # ASCII animation frames
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

    # ------------------------------------------------------------------
    # State inspection & mutation (Thread-safe)
    # ------------------------------------------------------------------

    @property
    def status(self) -> StatusType:
        with self._lock:
            return self._status

    def set_status(self, status: StatusType) -> None:
        with self._lock:
            self._status = status

    @property
    def is_generating(self) -> bool:
        with self._lock:
            return self._is_generating

    @property
    def history(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._history)

    @property
    def benchmarks(self) -> list[dict]:
        with self._lock:
            return list(self._benchmarks)

    @property
    def latest_benchmark(self) -> dict | None:
        with self._lock:
            return self._latest_benchmark.copy() if self._latest_benchmark else None

    def sync_audio_playback(self, is_playing: bool) -> None:
        """Called by the main loop to reconcile status when audio is playing or done."""
        with self._lock:
            if is_playing or self._is_generating:
                self._status = "PROCESSING"
            elif not self._is_generating and not is_playing and self._status == "PROCESSING":
                if self._current_assistant:
                    self._history.append(
                        {"role": "assistant", "text": self._current_assistant.strip()}
                    )
                    self._current_assistant = ""
                self._status = "READY"

    # ------------------------------------------------------------------
    # STT Listener callbacks
    # ------------------------------------------------------------------

    def on_transcript_started(self, text: str) -> None:
        with self._lock:
            self._status = "LISTENING"
            self._current_user = text.strip()
            if self._t_speech_start is None:
                self._t_speech_start = time.perf_counter()

    def on_transcript_updated(self, text: str) -> None:
        with self._lock:
            self._status = "LISTENING"
            self._current_user = text.strip()

    def on_transcript_completed(self, text: str) -> None:
        text = text.strip()
        with self._lock:
            if text:
                self._current_user = text
                self._pending_user_turn = text
                self._t_speech_end = time.perf_counter()

    # ------------------------------------------------------------------
    # LLM Listener callbacks
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        with self._lock:
            now = time.perf_counter()
            self._status = "PROCESSING"
            self._is_generating = True
            self._t_gate_fired = now
            self._t_first_token = None
            self._t_llm_done = None
            self._t_playback_start = None
            self._t_playback_end = None

            if self._pending_user_turn:
                self._history.append({"role": "user", "text": self._pending_user_turn})
                self._pending_user_turn = ""
                self._current_user = ""
            elif self._current_user:
                self._history.append({"role": "user", "text": self._current_user})
                self._current_user = ""
            self._current_assistant = ""
            self._llm_generated_text = ""

    def on_generation_updated(self, text: str) -> None:
        with self._lock:
            now = time.perf_counter()
            if self._t_first_token is None:
                self._t_first_token = now
            self._status = "PROCESSING"
            self._llm_generated_text = text

    def on_generation_completed(self, text: str) -> None:
        with self._lock:
            now = time.perf_counter()
            self._is_generating = False
            self._t_llm_done = now
            self._llm_generated_text = text.strip() or self._llm_generated_text.strip()

    # ------------------------------------------------------------------
    # Audio Playback Listener callbacks
    # ------------------------------------------------------------------

    def on_playback_started(self) -> None:
        with self._lock:
            now = time.perf_counter()
            if self._t_playback_start is None:
                self._t_playback_start = now
            self._status = "PROCESSING"

    def on_playback_text_updated(self, text: str) -> None:
        with self._lock:
            self._status = "PROCESSING"
            self._current_assistant = text

    def on_playback_completed(self) -> None:
        with self._lock:
            now = time.perf_counter()
            self._t_playback_end = now
            full_text = self._current_assistant.strip() or self._llm_generated_text.strip()
            if full_text:
                self._history.append({"role": "assistant", "text": full_text})

            # Calculate and record benchmark metrics
            last_user_text = (
                self._history[-2]["text"] if len(self._history) >= 2 else "Unknown"
            )

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
                "assistant_text": full_text,
                "gate_dur": gate_dur,
                "first_token_dur": first_tok_dur,
                "full_llm_dur": llm_full_dur,
                "tts_first_dur": tts_first_dur,
                "e2e_dur": e2e_dur,
                "total_dur": total_dur,
            }
            self._latest_benchmark = benchmark
            self._benchmarks.append(benchmark)

            self._current_assistant = ""
            self._llm_generated_text = ""
            self._t_speech_start = None
            self._status = "READY"

    # ------------------------------------------------------------------
    # Rich Layout Renderer
    # ------------------------------------------------------------------

    def render(self, anim_frame: int = 0) -> Layout:
        with self._lock:
            status = self._status
            history = list(self._history)
            current_user = self._current_user
            current_assistant = self._current_assistant
            is_generating = self._is_generating
            latest_bm = self._latest_benchmark.copy() if self._latest_benchmark else None
            verbose = self._verbose

        layout = Layout()
        if verbose:
            layout.split_column(
                Layout(name="header", size=5),
                Layout(name="body"),
                Layout(name="benchmark", size=9),
                Layout(name="footer", size=3),
            )
        else:
            layout.split_column(
                Layout(name="header", size=5),
                Layout(name="body"),
                Layout(name="footer", size=3),
            )

        # Header split: Title banner on left, Status Panel fixed on top right
        layout["header"].split_row(
            Layout(name="title"),
            Layout(name="status_box", size=26),
        )

        current_wave = self._wave_frames[anim_frame % len(self._wave_frames)]
        spin = self._spinner_frames[anim_frame % len(self._spinner_frames)]

        # --- Left Header: App title & ASCII wave ---
        title_text = Text()
        title_text.append(" 🧇 WAFFLE AUDIO CLI ", style="bold bright_white on dark_blue")
        title_text.append(f"  {current_wave}\n", style="bold cyan")
        title_text.append(" Real-Time Voice Assistant • Local & Private", style="dim italic")
        if verbose:
            title_text.append("  [VERBOSE TIMELINE ACTIVE]", style="bold magenta")
        layout["title"].update(Panel(title_text, border_style="blue", padding=(0, 1)))

        # --- Right Header: Status box (Pinned Top-Right Corner) ---
        if status == "READY":
            status_panel = Panel(
                Align.center(
                    Text.from_markup(
                        "[bold bright_white on dark_green]   ● READY   [/]\n"
                        "[dim green]Awaiting speech...[/dim green]"
                    )
                ),
                border_style="bright_green",
                title="[bold green]STATUS[/]",
                title_align="right",
            )
        elif status == "LISTENING":
            status_panel = Panel(
                Align.center(
                    Text.from_markup(
                        f"[bold bright_white on dark_blue] 🎤 LISTENING [/]\n"
                        f"[bright_cyan]{current_wave}[/]"
                    )
                ),
                border_style="bright_cyan",
                title="[bold cyan]STATUS[/]",
                title_align="right",
            )
        else:  # PROCESSING
            status_panel = Panel(
                Align.center(
                    Text.from_markup(
                        f"[bold black on dark_orange] {spin} PROCESSING [/]\n"
                        "[bold yellow]Thinking & speaking[/]"
                    )
                ),
                border_style="bright_yellow",
                title="[bold yellow]STATUS[/]",
                title_align="right",
            )

        layout["status_box"].update(status_panel)

        # --- Body: Real-time Conversation History & Current Utterance ---
        body_content = Text()

        if not history and not current_user and not current_assistant and status == "READY":
            body_content.append(
                "\n  ✨ Microphone is active. Speak naturally, and Waffle will answer out loud!\n\n",
                style="dim italic",
            )

        # Show the most recent conversation messages
        slice_count = 8 if verbose else 12
        for item in history[-slice_count:]:
            if item["role"] == "user":
                body_content.append("👤 You: ", style="bold bright_cyan")
                body_content.append(f"{item['text']}\n\n", style="white")
            else:
                body_content.append("🧇 Waffle: ", style="bold bright_yellow")
                body_content.append(f"{item['text']}\n\n", style="bright_white")

        # In-progress user speech
        if current_user:
            body_content.append("👤 You (speaking): ", style="bold cyan")
            body_content.append(f"{current_user} ▌\n\n", style="italic bright_cyan")

        # In-progress assistant response (streams as audio plays) or loader
        if current_assistant:
            body_content.append("🧇 Waffle: ", style="bold bright_yellow")
            body_content.append(f"{current_assistant} ▌\n\n", style="bright_white")
        elif (status == "PROCESSING" or is_generating) and not current_assistant:
            body_content.append("🧇 Waffle: ", style="bold bright_yellow")
            body_content.append(
                f"{spin} Thinking and preparing speech...\n\n",
                style="italic yellow",
            )

        layout["body"].update(
            Panel(body_content, title="💬 Real-Time Conversation", border_style="dim white")
        )

        # --- Verbose Timeline Panel (if enabled) ---
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

        # --- Footer ---
        footer_msg = (
            "[dim]Press [bold red]Ctrl+C[/bold red] to stop • Local AI pipeline active • "
            "Verbose benchmark logging enabled[/dim]"
            if verbose
            else "[dim]Press [bold red]Ctrl+C[/bold red] to stop • Local AI pipeline active[/dim]"
        )
        layout["footer"].update(Align.center(Text.from_markup(footer_msg)))

        return layout
