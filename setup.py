#!/usr/bin/env python3
"""Interactive setup wizard for Waffle Audio.

Generates a local .env configuration file before running the assistant.
Prompts for:
  - Required user personalization (Name, Location, Timezone) WITHOUT default values.
  - System and runtime defaults (Ollama URL, Model, Assistant Name, Voice, Microphone, Tokens) WITH fallback defaults.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ANSI colors if rich is unavailable
BOLD = "\033[1m"
GREEN = "\033[38;5;46m"
CYAN = "\033[38;5;51m"
YELLOW = "\033[38;5;226m"
RED = "\033[38;5;196m"
DIM = "\033[2m"
RESET = "\033[0m"


def _detect_wsl_host_ip() -> str | None:
    """Detect the WSL host gateway IP if running inside WSL2."""
    try:
        res = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout:
            parts = res.stdout.strip().split()
            if "via" in parts:
                idx = parts.index("via") + 1
                if idx < len(parts):
                    return parts[idx]
            elif len(parts) >= 3:
                return parts[2]
    except Exception:
        pass
    return None


def _detect_best_ollama_url() -> str:
    """Probe localhost and WSL host IP to suggest the best reachable Ollama URL."""
    candidates = ["http://localhost:11434"]
    wsl_ip = _detect_wsl_host_ip()
    if wsl_ip:
        candidates.append(f"http://{wsl_ip}:11434")

    # Probe candidates with quick curl
    for url in candidates:
        try:
            res = subprocess.run(
                ["curl", "-s", "--max-time", "1", f"{url}/api/tags"],
                capture_output=True,
                timeout=1.5,
            )
            if res.returncode == 0:
                return url
        except Exception:
            pass

    # If neither responds right now, prefer wsl_ip if in WSL, otherwise localhost
    if wsl_ip and os.path.exists("/proc/version"):
        with open("/proc/version", "r", encoding="utf-8") as f:
            if "microsoft" in f.read().lower():
                return f"http://{wsl_ip}:11434"
    return "http://localhost:11434"


def _list_microphones() -> list[tuple[int | str, str]]:
    """List available audio input devices via sounddevice."""
    inputs: list[tuple[int | str, str]] = [("default", "System Default Microphone")]
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                name = dev.get("name", f"Device {idx}")
                inputs.append((idx, f"{name} (Input Device #{idx})"))
    except Exception:
        pass
    return inputs


def _list_voices() -> list[str]:
    """List available voices including cloned voices."""
    voices = ["anna", "af_sarah"]
    cloned_dir = Path("cloned_voices")
    if cloned_dir.is_dir():
        for p in sorted(cloned_dir.glob("*.safetensors")):
            if p.is_file() and not p.name.startswith("."):
                if p.stem not in voices:
                    voices.append(p.stem)
    return voices


def prompt_with_default(question: str, default: str, non_interactive: bool = False) -> str:
    """Prompt the user for an option that HAS a fallback default value."""
    if non_interactive:
        return default
    try:
        val = input(f"{BOLD}{CYAN}{question}{RESET} [{BOLD}{GREEN}{default}{RESET}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def prompt_without_default(question: str, non_interactive: bool = False) -> str:
    """Prompt the user for an option that DOES NOT have a fallback default value."""
    if non_interactive:
        return ""
    try:
        return input(f"{BOLD}{YELLOW}{question}{RESET}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def run_setup(env_path: Path = Path(".env"), force: bool = False, non_interactive: bool = False) -> None:
    """Execute the interactive setup wizard and write configuration to .env."""
    if env_path.is_file() and not force:
        print(f"{GREEN}✓ Existing configuration found at {env_path.resolve()}. Skipping setup.{RESET}")
        print(f"{DIM}(Run with --force or pass --setup to run.sh to reconfigure){RESET}")
        return

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}             Waffle Audio — Initial Setup & Personalization           {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")
    print(f"This one-time wizard personalizes your voice assistant and configures your local environment.")
    print(f"Configuration will be saved to {BOLD}.env{RESET} (which is ignored by git).\n")

    # -------------------------------------------------------------------------
    # 1. User Personalization (NO DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[1/3] User Personalization (No defaults){RESET}")
    print(f"{DIM}Tell your assistant who it is speaking with so it can personalize responses.{RESET}")
    user_name = prompt_without_default("  Your Name (e.g. Aayush, Alex)", non_interactive)
    user_location = prompt_without_default("  Your Location (City, Country - e.g. New Delhi, India)", non_interactive)
    user_timezone = prompt_without_default("  Your Timezone (e.g. Asia/Kolkata, America/New_York)", non_interactive)

    print()

    # -------------------------------------------------------------------------
    # 2. Assistant & Audio Settings (WITH DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[2/3] Assistant & Audio Hardware (Defaults provided){RESET}")
    assistant_name = prompt_with_default("  Assistant Name", "Waffle", non_interactive)

    # Voice selection
    available_voices = _list_voices()
    voice_desc = ", ".join(available_voices)
    print(f"  {DIM}Available voices: {voice_desc}{RESET}")
    assistant_voice = prompt_with_default("  Assistant Voice", "anna", non_interactive)

    # Microphone selection
    mics = _list_microphones()
    print(f"  {DIM}Detected audio inputs:{RESET}")
    for idx_opt, desc in mics:
        print(f"    [{idx_opt}] {desc}")
    mic_choice = prompt_with_default("  Microphone Device (choose index or 'default')", "default", non_interactive)

    print()

    # -------------------------------------------------------------------------
    # 3. LLM Runtime & Tokens (WITH DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[3/3] LLM Runtime & External Tokens (Defaults provided){RESET}")
    default_ollama_url = _detect_best_ollama_url()
    ollama_url = prompt_with_default("  Ollama Runtime URL", default_ollama_url, non_interactive)
    ollama_model = prompt_with_default("  Ollama Model Name", "qwen3.5:4b", non_interactive)

    github_token = prompt_with_default(
        "  GitHub Personal Access Token (for GitHub MCP tool, optional)",
        "",
        non_interactive,
    )

    # -------------------------------------------------------------------------
    # Generate and write .env
    # -------------------------------------------------------------------------
    env_content = f"""# ==============================================================================
# Waffle Audio Configuration (.env)
# Automatically generated by setup.py
# ==============================================================================

# User Personalization (No defaults)
USER_NAME="{user_name}"
USER_LOCATION="{user_location}"
USER_TIMEZONE="{user_timezone}"

# Assistant Identity & Voice (Defaults provided)
ASSISTANT_NAME="{assistant_name}"
VOICECHAT_TTS_VOICE="{assistant_voice}"

# Audio Hardware (Defaults provided)
AUDIO_INPUT_DEVICE="{mic_choice}"
AUDIO_OUTPUT_DEVICE="default"

# LLM & Ollama Runtime (Defaults provided)
OLLAMA_BASE_URL="{ollama_url}"
OLLAMA_MODEL="{ollama_model}"

# External Tokens & Tools (Defaults provided)
GITHUB_PERSONAL_ACCESS_TOKEN="{github_token}"
"""

    env_path.write_text(env_content, encoding="utf-8")
    print(f"\n{BOLD}{GREEN}✓ Configuration saved successfully to {env_path.resolve()}!{RESET}")
    print(f"{CYAN}You can edit .env anytime or re-run setup with: {BOLD}python setup.py --force{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Waffle Audio Setup Wizard")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force setup wizard even if .env already exists",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run non-interactively using default fallback values",
    )
    args = parser.parse_args()

    # Auto-detect non-interactive terminal
    non_interactive = args.non_interactive or (not sys.stdin.isatty())
    run_setup(force=args.force, non_interactive=non_interactive)


if __name__ == "__main__":
    main()
