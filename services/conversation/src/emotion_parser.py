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
    r"(?:\[(?:emotion:\s*)?(neutral|happy|sad|angry)\]|<emotion>(neutral|happy|sad|angry)</emotion>|\(emotion:\s*(neutral|happy|sad|angry)\))",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_emotion(emotion: str | None, default: str = "neutral") -> str:
    """Normalize emotion string to one of the 4 supported emotion keys."""
    if not emotion:
        return default
    cleaned = emotion.strip().lower()
    if cleaned in VALID_EMOTIONS:
        return cleaned
    return default


def _extract_match_emotion(match: re.Match[str], default: str) -> str:
    detected = next((g for g in match.groups() if g is not None), None)
    return normalize_emotion(detected, default=default)


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
        clean_text = _WHITESPACE_RE.sub(" ", sentence.strip())
        return current_emotion, clean_text

    emotion = _extract_match_emotion(match, default=current_emotion)

    # Strip all emotion tags from this sentence
    clean_text = _EMOTION_TAG_PATTERN.sub("", sentence).strip()
    clean_text = _WHITESPACE_RE.sub(" ", clean_text)

    return emotion, clean_text


def split_by_emotion(
    text: str,
    current_emotion: str = "neutral",
) -> list[tuple[str, str]]:
    """Split text into ordered (emotion, clean_text) segments whenever an emotion tag appears.

    Preserves mid-chunk emotion transitions such as
    ``"[sad] Oh no! [happy] Wait, I fixed it!"``.
    """
    if not text:
        return []

    segments: list[tuple[str, str]] = []
    active = normalize_emotion(current_emotion, default="neutral")
    cursor = 0

    for match in _EMOTION_TAG_PATTERN.finditer(text):
        before = _WHITESPACE_RE.sub(" ", text[cursor:match.start()].strip())
        if before:
            segments.append((active, before))
        active = _extract_match_emotion(match, default=active)
        cursor = match.end()

    tail = _WHITESPACE_RE.sub(" ", text[cursor:].strip())
    if tail:
        segments.append((active, tail))

    return segments


def extract_turn_emotion(text: str, default: str = "neutral") -> str:
    """Extract the first valid emotion tag found in full text, or default."""
    if not text:
        return default

    match = _EMOTION_TAG_PATTERN.search(text)
    if not match:
        return default

    return _extract_match_emotion(match, default=default)


def extract_latest_emotion(text: str) -> str | None:
    """Extract the most recent valid emotion tag found in text, or None if none present."""
    if not text:
        return None

    last_match: re.Match[str] | None = None
    for match in _EMOTION_TAG_PATTERN.finditer(text):
        last_match = match

    if last_match is None:
        return None
    return _extract_match_emotion(last_match, default="neutral")


def strip_emotion_tags(text: str) -> str:
    """Remove all emotion markers from text for clean display or playback."""
    if not text:
        return ""
    cleaned = _EMOTION_TAG_PATTERN.sub("", text).strip()
    return _WHITESPACE_RE.sub(" ", cleaned)
