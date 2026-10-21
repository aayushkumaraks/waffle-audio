"""
Adaptive sentence-completion gate.

Accumulates STT fragments and holds them until the user has genuinely
finished speaking, then forwards the joined text as a single utterance.

Strategy
--------
Every ``submit()`` call appends the new fragment to a running buffer and
resets a silence timer.  Two timeout values control when the gate fires:

* ``gap_timeout``  — how long to wait after the *last* fragment before
  treating the user as done speaking.  This is the primary knob; it
  adapts linearly based on recent behaviour.
* ``quick_fire_timeout`` — if the *accumulated* text already looks
  clearly complete (high completeness score) the gate fires after this
  shorter delay instead, giving snappier responses for clean sentences.

Adaptation (linear, never exponential)
---------------------------------------
A rolling window of recent completeness scores produces an
``adaptive_factor`` in [1.0, 1.5].  When the user tends to trail off in
fragments the factor pushes ``gap_timeout`` upward; fluent speakers keep
the base value.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)

# Words that strongly suggest the utterance has not yet ended.
_DANGLING_WORDS = frozenset({
    "a", "an", "the",
    "and", "but", "or", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "into",
    "about", "over", "under", "after", "before", "between",
    "i", "you", "he", "she", "we", "they", "it",
    "is", "are", "was", "were", "be", "been",
    "that", "which", "who", "what", "when", "where", "how",
    "not", "just", "also", "then", "than",
    "will", "would", "could", "should", "can", "may", "might",
})

_TERMINAL_PUNCT = frozenset(".!?")


def _completeness_score(text: str) -> float:
    """
    Return a score in [0.0, 1.0] estimating how complete the sentence is.

    Scoring breakdown:
    - 0.50  ends with sentence-ending punctuation  (strong signal)
    - 0.20  last word is not a dangling/incomplete word
    - 0.30  length bucket  (>=8 words → 0.30, >=4 words → 0.15)
    """
    text = text.strip()
    if not text:
        return 0.0

    score = 0.0

    if text[-1] in _TERMINAL_PUNCT:
        score += 0.50

    words = text.split()

    word_count = len(words)
    if word_count >= 8:
        score += 0.30
    elif word_count >= 4:
        score += 0.15

    last_word = re.sub(r"[^\w]", "", words[-1]).lower() if words else ""
    if last_word and last_word not in _DANGLING_WORDS:
        score += 0.20

    return min(score, 1.0)


class SentenceGate:
    """
    Adaptive silence-gap gate that accumulates STT fragments.

    Usage::

        gate = SentenceGate(callback=my_llm_call)
        # Each STT completion calls submit(); the gate joins them and
        # fires once the user has been silent for gap_timeout seconds.
        gate.submit("Hello I was thinking of a")     # fragment — wait
        gate.submit("dinosaur but I'm not sure")     # still open — reset timer
        # … 2 s of silence …
        # → callback("Hello I was thinking of a dinosaur but I'm not sure")

    Adaptive logic
    --------------
    ``adaptive_factor`` ∈ [1.0, 1.5] scales ``gap_timeout`` linearly
    based on the rolling average of recent completeness scores.  Fragment-
    heavy speakers get a longer patience window; fluent speakers get snappier
    responses.
    """

    _COMPLETE_THRESHOLD = 0.7  # score at/above which quick_fire_timeout is used

    def __init__(
        self,
        callback: Callable[[str], None],
        *,
        gap_timeout: float = 2.0,
        quick_fire_timeout: float = 0.8,
        history_size: int = 8,
    ) -> None:
        self._callback = callback
        self._gap_timeout = gap_timeout
        self._quick_fire_timeout = quick_fire_timeout

        # Rolling history of completeness scores at fire-time for adaptation.
        self._history: deque[float] = deque(maxlen=history_size)

        self._buffer: list[str] = []
        self._pending_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_speech_active(self) -> None:
        """Called when new speech is detected or actively streaming.

        Postpones the silence timer so the gate never interrupts the
        user while speech is actively in progress, but ensures any buffered
        utterance is still dispatched once speech ceases.
        """
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None

            if self._buffer:
                accumulated = " ".join(self._buffer)
                delay = self._compute_delay(accumulated)
                timer = threading.Timer(delay, self._fire)
                timer.daemon = True
                timer.start()
                self._pending_timer = timer
                logger.debug("[GATE] Speech active; postponed silence timer (delay=%.2fs, buffer='%s')", delay, accumulated)

    def submit(self, text: str) -> None:
        """Append a new fragment and reset the silence timer."""
        text = text.strip()
        with self._lock:
            if text:
                self._buffer.append(text)
            if not self._buffer:
                return

            accumulated = " ".join(self._buffer)

            if self._pending_timer is not None:
                self._pending_timer.cancel()

            score = _completeness_score(accumulated)
            delay = self._compute_delay(accumulated)
            timer = threading.Timer(delay, self._fire)
            timer.daemon = True
            timer.start()
            self._pending_timer = timer
            logger.debug(
                "[GATE] Fragment '%s' buffered -> accumulated='%s' (completeness=%.2f, delay=%.2fs)",
                text,
                accumulated,
                score,
                delay,
            )

    def cancel(self) -> None:
        """Cancel any pending callback and clear the buffer (call on shutdown)."""
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self) -> None:
        with self._lock:
            if not self._buffer:
                self._pending_timer = None
                return
            text = " ".join(self._buffer)
            self._buffer.clear()
            self._pending_timer = None

        score = _completeness_score(text)
        self._history.append(score)
        logger.info(
            "[GATE BENCHMARK] SentenceGate silence timer fired (score=%.2f, words=%d): '%s'",
            score,
            len(text.split()),
            text,
        )
        self._callback(text)

    def _compute_delay(self, accumulated_text: str) -> float:
        """Return the silence timeout to use given the current buffer state."""
        score = _completeness_score(accumulated_text)

        # Looks clearly complete → use quick fire path.
        if score >= self._COMPLETE_THRESHOLD:
            return self._quick_fire_timeout

        # Otherwise wait for a real silence gap, scaled by adaptation.
        avg_score = (
            sum(self._history) / len(self._history) if self._history else 0.5
        )

        # Linear adaptation: patient with habitual fragment speakers.
        # adaptive_factor ∈ [1.0, 1.5]
        adaptive_factor = 1.0 + (1.0 - avg_score) * 0.5

        return self._gap_timeout * adaptive_factor


class ClauseChunker:
    """Adaptive clause and sentence boundary chunker for streaming LLM tokens.

    Provides:
    1. First-Chunk Fast Path: Allows the very first chunk of an assistant
       response to be short (3–5 words: e.g., "Sure, I can help...") to dispatch
       immediately to TTS for under 200ms audio latency.
    2. Adaptive Clause Chunking: Triggers synthesis on intermediate clause markers
       (',', ';', ':', '—', '--') or coordinating conjunctions ('and', 'but',
       'because', 'however', 'although', 'since') once a minimum token count
       (4–6 words) is reached.
    3. Full Sentence Boundaries: Triggers on '.', '!', '?' as terminal boundaries.
    """

    _CLAUSE_PUNCT_RE = re.compile(r"([,;:\u2014]|\s--\s)[\"')\]]*\s")
    _SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s")
    _CONJUNCTION_RE = re.compile(
        r"\s+(and|but|because|however|although|since|so|yet)\s+",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        first_chunk_min_words: int = 3,
        clause_min_words: int = 5,
        sentence_min_words: int = 3,
    ) -> None:
        self._first_chunk_min_words = first_chunk_min_words
        self._clause_min_words = clause_min_words
        self._sentence_min_words = sentence_min_words
        self._cursor = 0
        self._is_first_chunk = True

    def reset(self) -> None:
        self._cursor = 0
        self._is_first_chunk = True

    @property
    def spoken_chars(self) -> int:
        return self._cursor

    def process_stream(self, text: str) -> list[str]:
        """Process incremental LLM stream text and return any newly ready chunks."""
        chunks: list[str] = []
        while self._cursor < len(text):
            unprocessed = text[self._cursor:]
            boundary = self._find_next_boundary(unprocessed, is_first=self._is_first_chunk)
            if boundary is None:
                break

            chunk = unprocessed[:boundary].strip()
            self._cursor += boundary
            if chunk:
                chunks.append(chunk)
                self._is_first_chunk = False

        return chunks

    def finalize(self, text: str) -> str | None:
        """Finalize stream at completion, returning any remaining unspoken text."""
        remainder = text[self._cursor:].strip()
        self._cursor = len(text)
        self._is_first_chunk = True
        return remainder if remainder else None

    def _find_next_boundary(self, subtext: str, is_first: bool) -> int | None:
        words = subtext.split()
        if not words:
            return None

        # Check sentence terminal boundaries first
        sent_match = self._SENTENCE_END_RE.search(subtext)
        if sent_match:
            chunk_words = subtext[:sent_match.end()].split()
            min_w = self._first_chunk_min_words if is_first else self._sentence_min_words
            if len(chunk_words) >= min_w:
                return sent_match.end()

        # Check intermediate clause punctuation: , ; : — --
        clause_match = self._CLAUSE_PUNCT_RE.search(subtext)
        if clause_match:
            chunk_words = subtext[:clause_match.end()].split()
            min_w = self._first_chunk_min_words if is_first else self._clause_min_words
            if len(chunk_words) >= min_w:
                return clause_match.end()

        # Check coordinating conjunctions (and, but, because, etc.)
        min_conj_words = self._first_chunk_min_words if is_first else self._clause_min_words
        if len(words) >= min_conj_words + 2:
            conj_match = self._CONJUNCTION_RE.search(subtext)
            if conj_match:
                prefix = subtext[:conj_match.start()]
                if len(prefix.split()) >= min_conj_words:
                    return conj_match.start()

        return None

