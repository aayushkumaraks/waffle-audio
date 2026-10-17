"""Rich-based terminal UI for Emotional Voice Assistant with real-time emotion badges and thread status."""
from __future__ import annotations

import threading
import time
from typing import Any, Literal

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from services.audio.src import AudioPlaybackListener
from services.conversation.src.emotion_parser import extract_turn_emotion, strip_emotion_tags
from services.llm.src import LLMListener
from services.stt.src import STTListener

StatusType = Literal["READY", "LISTENING", "PROCESSING"]

EMOTION_STYLES: dict[str, dict[str, str]] = {
    "neutral": {"color": "cyan", "icon": "✦", "bg": "on blue"},
    "happy": {"color": "bright_yellow", "icon": "🌟", "bg": "on yellow"},
    "sad": {"color": "blue", "icon": "🌧", "bg": "on navy_blue"},
    "angry": {"color": "bright_red", "icon": "⚡", "bg": "on dark_red"},
}


class EmotionConversationUI(STTListener, LLMListener, AudioPlaybackListener):
    """Rich UI controller displaying dynamic emotion badges, active threads, and memory status."""

    def __init__(self, gender: str = "female", verbose: bool = False) -> None:
        self.gender = gender.capitalize()
        self._verbose = verbose
        self._lock = threading.Lock()
        self._status: StatusType = "READY"
        self._active_emotion: str = "neutral"
        self._worker_statuses: dict[str, dict[str, Any]] = {}

        self._history: list[dict[str, str]] = []
        self._current_user: str = ""
        self._current_assistant: str = ""
        self._is_generating: bool = False
        self._is_playing_audio: bool = False

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
            self._is_playing_audio = is_playing

    # ------------------------------------------------------------------
    # STTListener
    # ------------------------------------------------------------------

    def on_transcript_started(self, text: str) -> None:
        with self._lock:
            self._status = "LISTENING"
            self._current_user = text

    def on_transcript_updated(self, text: str) -> None:
        with self._lock:
            self._status = "LISTENING"
            self._current_user = text

    def on_transcript_completed(self, text: str) -> None:
        with self._lock:
            self._current_user = text
            self._status = "PROCESSING"

    def on_user_turn(self, text: str) -> None:
        with self._lock:
            self._current_user = text
            self._status = "PROCESSING"
            self._current_assistant = ""

    # ------------------------------------------------------------------
    # LLMListener
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        with self._lock:
            self._is_generating = True
            self._status = "PROCESSING"
            self._current_assistant = ""
            self._active_emotion = "neutral"

    def on_generation_updated(self, text: str) -> None:
        with self._lock:
            self._current_assistant = text
            detected = extract_turn_emotion(text)
            if detected:
                self._active_emotion = detected

    def on_generation_completed(self, text: str) -> None:
        with self._lock:
            self._is_generating = False
            clean_text = strip_emotion_tags(text)
            detected = extract_turn_emotion(text)
            if self._current_user:
                self._history.append({"role": "user", "text": self._current_user, "emotion": ""})
            self._history.append({"role": "assistant", "text": clean_text, "emotion": detected or self._active_emotion})
            self._current_user = ""
            self._current_assistant = ""
            self._status = "READY"

    # ------------------------------------------------------------------
    # AudioPlaybackListener
    # ------------------------------------------------------------------

    def on_playback_started(self) -> None:
        with self._lock:
            self._is_playing_audio = True

    def on_playback_completed(self) -> None:
        with self._lock:
            self._is_playing_audio = False

    @property
    def history(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # UI Rendering
    # ------------------------------------------------------------------

    def render(self, frame_idx: int, console: Console | None = None) -> Layout:
        with self._lock:
            status = self._status
            current_user = self._current_user
            current_assistant = self._current_assistant
            history = list(self._history)
            active_emotion = self._active_emotion
            is_playing = self._is_playing_audio
            statuses = dict(self._worker_statuses)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=8),
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

        # Body: History + Active Turn
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="conversation", ratio=2),
            Layout(name="models", ratio=1),
        )

        chat_table = Table.grid(expand=True, padding=(0, 1))
        chat_table.add_column(style="dim", width=12)
        chat_table.add_column(style="white", ratio=1)

        for msg in history[-6:]:
            if msg["role"] == "user":
                chat_table.add_row("[bold cyan]You:[/bold cyan]", f"[cyan]{msg['text']}[/cyan]")
            else:
                emo = msg.get("emotion", "neutral")
                e_color = EMOTION_STYLES.get(emo, EMOTION_STYLES["neutral"])["color"]
                chat_table.add_row(
                    f"[bold green]Waffle [{emo.upper()}]:[/bold green]",
                    f"[{e_color}]{msg['text']}[/{e_color}]",
                )

        if current_user:
            chat_table.add_row("[bold cyan]You (speaking):[/bold cyan]", f"[bold white]{current_user}[/bold white]")

        if current_assistant:
            clean_ast = strip_emotion_tags(current_assistant)
            e_color = EMOTION_STYLES.get(active_emotion, EMOTION_STYLES["neutral"])["color"]
            chat_table.add_row(
                f"[bold green]Waffle [{active_emotion.upper()}]:[/bold green]",
                f"[{e_color}]{clean_ast}[/{e_color}]",
            )

        body_layout["conversation"].update(
            Panel(chat_table, title="[bold]Conversation[/bold]", border_style="dim white")
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
            Panel(models_table, title=f"[bold]Voice Models ({self.gender})[/bold]", border_style="magenta")
        )
        layout["body"].update(body_layout)

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
        else:
            state_text = f"[dim green]● Ready (Listening for speech...)[/dim green]"

        footer_table.add_row(
            state_text,
            f"[dim]Models: 4 Threads resident / auto-offload | Engine: PocketTTS[/dim]",
            "[dim]Press Ctrl+C to stop[/dim]",
        )
        layout["footer"].update(Panel(footer_table, border_style="green"))

        return layout
