"""DuckDuckGo Search integration for web query fallbacks and information retrieval."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def clean_conversational_query(query: str) -> str:
    """Strip conversational filler, greetings, and stop question phrasing for search engines."""
    q = (query or "").strip()
    q = re.sub(r"\btool[\s_-]*calls?\b\s*,?\s*", "", q, flags=re.I)
    q = re.sub(
        r"^(hey|hello|hi|please|waffle|so|um|uh|can you|could you|would you)\b\s*",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(
        r"^(who\s+is\s+(the\s+)?|what\s+is\s+(the\s+)?|where\s+is\s+(the\s+)?|when\s+was\s+(the\s+)?|tell\s+me\s+(about\s+)?|can\s+you\s+(check\s+|tell\s+me\s+)?|check\s+(the\s+)?|search\s+(for\s+)?)",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(
        r"\b(right\s+now|at\s+this\s+moment|presently|currently|setting|today|please)\b",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"[?!.,;:]", "", q)
    return re.sub(r"\s+", " ", q).strip()


def is_news_query(query: str) -> bool:
    """Check if query is asking for news, headlines, or breaking updates."""
    return bool(re.search(r"\b(news|headline|headlines|breaking|updates?)\b", query, flags=re.I))


class DuckDuckGoSearch:
    """Wrapper around DuckDuckGo search (via duckduckgo_search or ddgs) with spoken-friendly text formatting."""

    def __init__(self, max_results: int = 4, timeout: float = 10.0) -> None:
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str, max_results: int | None = None) -> str:
        """Perform a web search and format results for LLM consumption.

        Returns a concise plain-text summary suitable for the speech assistant.
        """
        query = (query or "").strip()
        if not query:
            return "No search query provided."

        # 1. Authoritative real-time date and time interception
        date_time_pattern = re.compile(
            r"\b(today('?s)? date|current date|what date is it|what is today's date|what day is (it|today)|current time|what time is it)\b",
            re.I,
        )
        if date_time_pattern.search(query):
            from datetime import datetime
            now = datetime.now().astimezone()
            date_str = now.strftime("%A, %B %d, %Y")
            time_str = now.strftime("%I:%M %p %Z").strip()
            return f"Current authoritative date and time: Today is {date_str}, and the current local time is {time_str}."

        if DDGS is None:
            return "DuckDuckGo search encountered an issue: neither 'ddgs' nor 'duckduckgo_search' is installed."

        limit = max_results or self.max_results
        results: list[dict[str, Any]] = []
        is_news = is_news_query(query)
        cleaned = clean_conversational_query(query)

        last_error: Exception | None = None
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                # 1. For news queries, try ddgs.news first
                if is_news:
                    for candidate in (query, cleaned):
                        if not candidate:
                            continue
                        try:
                            news_res = list(ddgs.news(candidate, max_results=limit))
                            if news_res:
                                results = news_res
                                break
                        except Exception as e:
                            last_error = e
                            logger.debug("ddgs.news failed for '%s': %s", candidate, e)

                # 2. Try standard text search
                if not results:
                    try:
                        results = list(ddgs.text(query, max_results=limit))
                    except Exception as e:
                        last_error = e
                        logger.debug("ddgs.text failed for '%s': %s", query, e)

                # 3. If no results or error, try cleaned query text search
                if not results and cleaned and cleaned.lower() != query.lower():
                    try:
                        results = list(ddgs.text(cleaned, max_results=limit))
                    except Exception as e:
                        last_error = e
                        logger.debug("ddgs.text cleaned failed for '%s': %s", cleaned, e)

                # 4. If still no results and was a news query, try text search for candidate + 'news'
                if not results and is_news and cleaned:
                    news_text_query = f"{cleaned} news" if "news" not in cleaned.lower() else cleaned
                    try:
                        results = list(ddgs.text(news_text_query, max_results=limit))
                    except Exception as e:
                        last_error = e
                        logger.debug("ddgs.text news fallback failed for '%s': %s", news_text_query, e)

        except Exception as exc:
            logger.warning("DuckDuckGo search error for '%s': %s", query, exc)
            return f"DuckDuckGo search encountered an issue: {exc}"

        if not results:
            if last_error is not None and "no results found" not in str(last_error).lower():
                return f"DuckDuckGo search encountered an issue: {last_error}"
            return f"No search results found for query: '{query}'."

        formatted_lines = [f"Search results for '{query}':"]
        for idx, item in enumerate(results[:limit], start=1):
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()
            # Allow rich snippet up to 1000 characters so essential answers are preserved
            if len(body) > 1000:
                body = body[:997] + "..."
            formatted_lines.append(f"{idx}. {title}: {body}")

        return "\n".join(formatted_lines)
