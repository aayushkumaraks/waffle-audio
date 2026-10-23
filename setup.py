#!/usr/bin/env python3
"""Interactive setup wizard for Waffle Audio.

Generates a local .env configuration file before running the assistant.
Prompts for:
  - Required user personalization (Name, Location, Timezone) WITHOUT default values.
  - System and runtime defaults (Ollama URL, Model selection via API, Assistant Name, Voice, Microphone, Tokens) WITH fallback defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
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


def _list_ollama_models(ollama_url: str) -> list[str]:
    """Query the Ollama /api/tags endpoint to retrieve installed models."""
    url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WaffleSetup/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return models
    except Exception:
        pass
    return []


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


def _parse_existing_env(env_path: Path) -> dict[str, str]:
    """Parse key-value pairs from an existing .env file if it exists."""
    config: dict[str, str] = {}
    if not env_path.is_file():
        return config
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                    v = v[1:-1]
                config[k] = v
    except Exception:
        pass
    return config


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

    existing = _parse_existing_env(env_path)

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}             Waffle Audio — Initial Setup & Personalization           {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")
    print(f"This wizard personalizes your voice assistant and configures your local environment.")
    print(f"Configuration will be saved to {BOLD}.env{RESET} (which is ignored by git).\n")

    # -------------------------------------------------------------------------
    # 1. User Personalization (NO DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[1/4] User Personalization (No defaults){RESET}")
    print(f"{DIM}Tell your assistant who it is speaking with so it can personalize responses.{RESET}")
    user_name_default = existing.get("USER_NAME", "")
    user_loc_default = existing.get("USER_LOCATION", "")
    user_tz_default = existing.get("USER_TIMEZONE", "")

    if user_name_default and not non_interactive:
        user_name = prompt_with_default("  Your Name (e.g. Aayush, Alex)", user_name_default, non_interactive)
    else:
        user_name = prompt_without_default("  Your Name (e.g. Aayush, Alex)", non_interactive)

    if user_loc_default and not non_interactive:
        user_location = prompt_with_default("  Your Location (City, Country)", user_loc_default, non_interactive)
    else:
        user_location = prompt_without_default("  Your Location (City, Country - e.g. New Delhi, India)", non_interactive)

    if user_tz_default and not non_interactive:
        user_timezone = prompt_with_default("  Your Timezone", user_tz_default, non_interactive)
    else:
        user_timezone = prompt_without_default("  Your Timezone (e.g. Asia/Kolkata, America/New_York)", non_interactive)

    print()

    # -------------------------------------------------------------------------
    # 2. Assistant Identity & Voice Backend (WITH DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[2/4] Assistant Identity & Voice Engine (Defaults provided){RESET}")
    assistant_name = prompt_with_default(
        "  Assistant Name",
        existing.get("ASSISTANT_NAME", "Waffle"),
        non_interactive,
    )

    # Personality selection
    from services.llm.src.personalities import list_personalities, get_personality
    personalities = list_personalities()
    print(f"  {DIM}Available Personalities:{RESET}")
    default_pers = existing.get("VOICECHAT_PERSONALITY", existing.get("ASSISTANT_PERSONALITY", "natural")).lower()
    default_pers_idx = 1
    for idx, p in enumerate(personalities, 1):
        rec = f" {BOLD}{GREEN}(recommended){RESET}" if p.id == "natural" else ""
        if p.id == default_pers:
            default_pers_idx = idx
        print(f"    [{idx}] {p.name} {DIM}({p.id}): {p.description}{RESET}{rec}")

    if non_interactive:
        personality_choice = default_pers
    else:
        p_choice = prompt_with_default(
            f"  Assistant Personality (1-{len(personalities)} or name)",
            str(default_pers_idx),
            non_interactive,
        )
        if p_choice.isdigit() and 1 <= int(p_choice) <= len(personalities):
            personality_choice = personalities[int(p_choice) - 1].id
        elif p_choice:
            personality_choice = get_personality(p_choice).id
        else:
            personality_choice = default_pers

    tts_backend = prompt_with_default(
        "  TTS Backend (pocket or kokoro)",
        existing.get("VOICECHAT_TTS_BACKEND", "pocket"),
        non_interactive,
    ).lower()
    if tts_backend not in ("pocket", "kokoro"):
        tts_backend = "pocket"

    # Voice selection
    available_voices = _list_voices()
    voice_desc = ", ".join(available_voices)
    print(f"  {DIM}Available voices: {voice_desc}{RESET}")
    default_voice = existing.get("VOICECHAT_TTS_VOICE", "anna" if tts_backend == "pocket" else "af_sarah")
    assistant_voice = prompt_with_default("  Assistant Voice", default_voice, non_interactive)

    tts_speed = prompt_with_default(
        "  Speech Speed Multiplier (0.5 - 2.0)",
        existing.get("VOICECHAT_TTS_SPEED", "1.0"),
        non_interactive,
    )

    print()

    # -------------------------------------------------------------------------
    # 3. Audio Hardware & Interaction Modes (WITH DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[3/4] Audio Hardware & Interaction Modes (Defaults provided){RESET}")
    mics = _list_microphones()
    print(f"  {DIM}Detected audio inputs:{RESET}")
    for idx_opt, desc in mics:
        print(f"    [{idx_opt}] {desc}")
    mic_choice = prompt_with_default(
        "  Microphone Device (choose index or 'default')",
        existing.get("AUDIO_INPUT_DEVICE", "default"),
        non_interactive,
    )

    trigger_mode_choice = prompt_with_default(
        "  Trigger Mode (only respond when 'Hey <Assistant Name>' is spoken) [true/false]",
        existing.get("VOICECHAT_TRIGGER_MODE", "false"),
        non_interactive,
    ).lower()
    trigger_mode = "true" if trigger_mode_choice in ("true", "1", "yes", "y") else "false"

    barge_in_choice = prompt_with_default(
        "  Barge-In Interruption (allow interrupting assistant mid-speech) [true/false]",
        existing.get("VOICECHAT_BARGE_IN_ENABLED", "true"),
        non_interactive,
    ).lower()
    barge_in = "true" if barge_in_choice in ("true", "1", "yes", "y") else "false"

    aec_choice = prompt_with_default(
        "  Acoustic Echo Cancellation (AEC for loudspeaker playback) [true/false]",
        existing.get("VOICECHAT_AEC_ENABLED", "true"),
        non_interactive,
    ).lower()
    aec = "true" if aec_choice in ("true", "1", "yes", "y") else "false"

    print()

    # -------------------------------------------------------------------------
    # 4. LLM Runtime & Ollama Model API Selection (WITH DEFAULT VALUES)
    # -------------------------------------------------------------------------
    print(f"{BOLD}[4/4] LLM Runtime & Ollama Model (API-powered model discovery){RESET}")
    default_ollama_url = existing.get("OLLAMA_BASE_URL") or _detect_best_ollama_url()
    ollama_url = prompt_with_default("  Ollama Runtime URL", default_ollama_url, non_interactive)

    # Use Ollama API to discover and select available models
    models = _list_ollama_models(ollama_url)
    if non_interactive:
        ollama_model = existing.get("OLLAMA_MODEL", "qwen3.5:4b")
    else:
        if models:
            print(f"  {BOLD}{GREEN}✓ Connected to Ollama at {ollama_url}. Installed models found:{RESET}")
            default_choice_idx = None
            for idx, m_name in enumerate(models, 1):
                is_match = (m_name == existing.get("OLLAMA_MODEL")) or (
                    not existing.get("OLLAMA_MODEL") and (m_name == "qwen3.5:4b" or m_name.startswith("qwen3.5:4b"))
                )
                rec = f" {BOLD}{GREEN}(recommended){RESET}" if m_name.startswith("qwen3.5:4b") else ""
                if is_match and default_choice_idx is None:
                    default_choice_idx = idx
                print(f"    [{idx}] {m_name}{rec}")

            def_str = str(default_choice_idx) if default_choice_idx is not None else "1"
            choice = prompt_with_default(
                f"  Select model (1-{len(models)} or type custom model name)",
                def_str,
                non_interactive=False,
            )
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                ollama_model = models[int(choice) - 1]
            elif choice:
                ollama_model = choice
            else:
                ollama_model = models[0]
        else:
            print(f"  {DIM}(Could not query models automatically from {ollama_url}/api/tags; entering manually){RESET}")
            ollama_model = prompt_with_default(
                "  Ollama Model Name",
                existing.get("OLLAMA_MODEL", "qwen3.5:4b"),
                non_interactive=False,
            )

    github_token = prompt_with_default(
        "  GitHub Personal Access Token (for GitHub MCP tool, optional)",
        existing.get("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
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
VOICECHAT_PERSONALITY="{personality_choice}"
VOICECHAT_TTS_BACKEND="{tts_backend}"
VOICECHAT_TTS_VOICE="{assistant_voice}"
VOICECHAT_TTS_SPEED="{tts_speed}"

# Audio Hardware (Defaults provided)
AUDIO_INPUT_DEVICE="{mic_choice}"
AUDIO_OUTPUT_DEVICE="default"

# Interaction Modes & Audio Processing (Defaults provided)
VOICECHAT_TRIGGER_MODE="{trigger_mode}"
VOICECHAT_BARGE_IN_ENABLED="{barge_in}"
VOICECHAT_AEC_ENABLED="{aec}"

# LLM & Ollama Runtime (Defaults provided)
OLLAMA_BASE_URL="{ollama_url}"
OLLAMA_MODEL="{ollama_model}"
OLLAMA_KEEP_ALIVE="{existing.get('OLLAMA_KEEP_ALIVE', '30m')}"
OLLAMA_NUM_PREDICT="{existing.get('OLLAMA_NUM_PREDICT', '160')}"
OLLAMA_NUM_CTX="{existing.get('OLLAMA_NUM_CTX', '2048')}"

# Conversation & Compactor (Defaults provided)
VOICECHAT_MAX_HISTORY_TURNS="{existing.get('VOICECHAT_MAX_HISTORY_TURNS', '8')}"
VOICECHAT_COMPACTOR_ENABLED="{existing.get('VOICECHAT_COMPACTOR_ENABLED', 'true')}"
VOICECHAT_COMPACTOR_MAX_TURNS="{existing.get('VOICECHAT_COMPACTOR_MAX_TURNS', '6')}"

# Tools & External Tokens (Defaults provided)
VOICECHAT_TOOLS_ENABLED="{existing.get('VOICECHAT_TOOLS_ENABLED', 'true')}"
VOICECHAT_DDGS_ENABLED="{existing.get('VOICECHAT_DDGS_ENABLED', 'true')}"
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
