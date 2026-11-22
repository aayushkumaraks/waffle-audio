"""Emotion parsing and tag cleaning utilities for LLM conversation responses."""
from __future__ import annotations

import re

# Supported emotions
VALID_EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "whisper")

# Common LLM emotion tag synonyms mapped to the 4 supported voice models
_EMOTION_SYNONYMS: dict[str, str] = {
    "neutral": "neutral",
    "normal": "neutral",
    "default": "neutral",
    "regular": "neutral",
    "standard": "neutral",
    "plain": "neutral",
    "calm": "neutral",
    "serious": "neutral",
    "curious": "neutral",
    "thoughtful": "neutral",
    "polite": "neutral",
    "helpful": "neutral",
    "casual": "neutral",
    "gentle": "neutral",
    "matter-of-fact": "neutral",
    "happy": "happy",
    "cheerful": "happy",
    "glad": "happy",
    "grateful": "happy",
    "joyful": "happy",
    "excited": "happy",
    "friendly": "happy",
    "warm": "happy",
    "playful": "happy",
    "amused": "happy",
    "delighted": "happy",
    "enthusiastic": "happy",
    "relieved": "happy",
    "hopeful": "happy",
    "proud": "happy",
    "sad": "sad",
    "sorry": "sad",
    "upset": "sad",
    "apologetic": "sad",
    "melancholy": "sad",
    "concerned": "sad",
    "worried": "sad",
    "sympathetic": "sad",
    "empathetic": "sad",
    "disappointed": "sad",
    "regretful": "sad",
    "whisper": "whisper",
    "whispering": "whisper",
    "whispered": "whisper",
    "soft": "whisper",
    "softly": "whisper",
    "quiet": "whisper",
    "quietly": "whisper",
    "hushed": "whisper",
    "secret": "whisper",
    "secretive": "whisper",
    "low": "whisper",
}

# Matches any bracketed/XML/explicit emotion or stage-direction tag so the TTS engine
# never speaks LLM-invented tags like [normal], [polite], or [sighs] aloud:
#   [happy], [NORMAL], [sighs], [matter-of-fact]
#   [emotion: happy] or [EMOTION: normal]
#   <emotion>happy</emotion>
#   (emotion: happy)
# Bare parenthetical prose like "(sad)" is intentionally NOT matched.
_TAG_WORD = r"[a-z][a-z_\-]*(?:\s+[a-z][a-z_\-]*)?"
_EMOTION_TAG_PATTERN = re.compile(
    rf"(?:\[(?:emotion:\s*)?({_TAG_WORD})\]|<emotion>\s*({_TAG_WORD})\s*</emotion>|\(emotion:\s*({_TAG_WORD})\))",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_emotion(emotion: str | None, default: str = "neutral") -> str:
    """Normalize emotion string (or known synonym) to one of the 4 supported emotion keys."""
    if not emotion:
        return default
    cleaned = emotion.strip().lower()
    if cleaned in VALID_EMOTIONS:
        return cleaned
    return _EMOTION_SYNONYMS.get(cleaned, default)


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
