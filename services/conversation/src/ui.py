"""Rich-based terminal UI for Waffle Audio CLI."""
from __future__ import annotations

import threading
from typing import Literal

from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from services.llm.src import LLMListener
from services.stt.src import STTListener

StatusType = Literal["READY", "LISTENING", "PROCESSING"]


class ConversationUI(STTListener, LLMListener):
    """Thread-safe UI controller and renderer for the terminal voice interface."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: StatusType = "READY"
        self._history: list[dict[str, str]] = []
        self._current_user: str = ""
        self._current_assistant: str = ""
        self._pending_user_turn: str = ""
        self._is_generating: bool = False

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

    def sync_audio_playback(self, is_playing: bool) -> None:
        """Called by the main loop to reconcile status when audio is playing or done."""
        with self._lock:
            if is_playing:
                self._status = "PROCESSING"
            elif not self._is_generating and not self._current_user and self._status == "PROCESSING":
                self._status = "READY"

    # ------------------------------------------------------------------
    # STT Listener callbacks
    # ------------------------------------------------------------------

    def on_transcript_started(self, text: str) -> None:
        with self._lock:
            self._status = "LISTENING"
            self._current_user = text.strip()

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

    # ------------------------------------------------------------------
    # LLM Listener callbacks
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        with self._lock:
            self._status = "PROCESSING"
            self._is_generating = True
            if self._pending_user_turn:
                self._history.append({"role": "user", "text": self._pending_user_turn})
                self._pending_user_turn = ""
                self._current_user = ""
            elif self._current_user:
                self._history.append({"role": "user", "text": self._current_user})
                self._current_user = ""
            self._current_assistant = ""

    def on_generation_updated(self, text: str) -> None:
        with self._lock:
            self._status = "PROCESSING"
            self._current_assistant = text

    def on_generation_completed(self, text: str) -> None:
        with self._lock:
            self._is_generating = False
            full_text = text.strip() or self._current_assistant.strip()
            if full_text:
                self._history.append({"role": "assistant", "text": full_text})
            self._current_assistant = ""

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

        layout = Layout()
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
        for item in history[-12:]:
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

        # In-progress assistant response or thinking loader
        if current_assistant:
            body_content.append("🧇 Waffle: ", style="bold bright_yellow")
            body_content.append(f"{current_assistant} ▌\n\n", style="bright_white")
        elif (status == "PROCESSING" or is_generating) and not current_assistant:
            body_content.append("🧇 Waffle: ", style="bold bright_yellow")
            body_content.append(
                f"{spin} Thinking and preparing voice response...\n\n",
                style="italic yellow",
            )

        layout["body"].update(
            Panel(body_content, title="💬 Real-Time Conversation", border_style="dim white")
        )

        # --- Footer ---
        footer_text = Text.from_markup(
            "[dim]Press [bold red]Ctrl+C[/bold red] to stop • Local AI pipeline active[/dim]"
        )
        layout["footer"].update(Align.center(footer_text))

        return layout
