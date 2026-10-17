"""Emotion parsing and tag cleaning utilities for LLM conversation responses."""
from __future__ import annotations

import re

# Supported emotions
VALID_EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "angry")

# Matches:
#   [happy] or [HAPPY]
#   [emotion: happy] or [EMOTION: happy]
#   <emotion>happy</emotion>
#   (emotion: happy)
_EMOTION_TAG_PATTERN = re.compile(
    r"(?:\[(?:emotion:\s*)?(neutral|happy|sad|angry)\]|<emotion>(neutral|happy|sad|angry)</emotion>|\((?:emotion:\s*)?(neutral|happy|sad|angry)\))",
    re.IGNORECASE,
)


def normalize_emotion(emotion: str | None, default: str = "neutral") -> str:
    """Normalize emotion string to one of the 4 supported emotion keys."""
    if not emotion:
        return default
    cleaned = emotion.strip().lower()
    if cleaned in VALID_EMOTIONS:
        return cleaned
    return default


def parse_sentence_emotion(
    sentence: str,
    current_emotion: str = "neutral",
) -> tuple[str, str]:
    """Parse a single sentence for an emotion tag.

    Returns:
        (detected_or_current_emotion, clean_sentence_without_tags)
    """
    if not sentence:
        return current_emotion, ""

    match = _EMOTION_TAG_PATTERN.search(sentence)
    if not match:
        clean_text = sentence.strip()
        return current_emotion, clean_text

    # Extract the captured group that matched
    detected = next((g for g in match.groups() if g is not None), None)
    emotion = normalize_emotion(detected, default=current_emotion)

    # Strip all emotion tags from this sentence
    clean_text = _EMOTION_TAG_PATTERN.sub("", sentence).strip()
    # Collapse multiple whitespace
    clean_text = re.sub(r"\s+", " ", clean_text)

    return emotion, clean_text


def extract_turn_emotion(text: str, default: str = "neutral") -> str:
    """Extract the first valid emotion tag found in full text, or default."""
    if not text:
        return default

    match = _EMOTION_TAG_PATTERN.search(text)
    if not match:
        return default

    detected = next((g for g in match.groups() if g is not None), None)
    return normalize_emotion(detected, default=default)


def strip_emotion_tags(text: str) -> str:
    """Remove all emotion markers from text for clean display or playback."""
    if not text:
        return ""
    cleaned = _EMOTION_TAG_PATTERN.sub("", text).strip()
    return re.sub(r"\s+", " ", cleaned)
