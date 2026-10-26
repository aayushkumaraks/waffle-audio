import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable, Sequence

import httpx

from constants import (
    ASSISTANT_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    TOOLS_ENABLED,
    USER_LOCATION,
    USER_NAME,
    USER_TIMEZONE,
    is_benchmark_enabled,
)
from models import Message
from services.tools import PhraseManager, ToolManager

logger = logging.getLogger(__name__)


class LLMCallbacks(ABC):
    """Receives streaming events from an LLM provider."""

    @abstractmethod
    def on_generation_started(self) -> None:
        ...

    @abstractmethod
    def on_generation_updated(self, text: str) -> None:
        ...

    @abstractmethod
    def on_generation_completed(self, text: str) -> None:
        ...

    def on_tool_started(
        self, tool_name: str, arguments: dict[str, Any], phrase: str | None = None
    ) -> None:
        """Called when a tool execution is initiated."""
        pass

    def on_tool_completed(
        self,
        tool_name: str,
        result: str,
        duration: float = 0.0,
        fallback: bool = False,
        server: str = "mcp",
    ) -> None:
        """Called when a tool execution has finished."""
        pass

    def record_followup_llm(self, first_token_dur: float, full_dur: float) -> None:
        """Called to record follow-up LLM stream timings after tool execution."""
        pass


@dataclass(slots=True)
class LLMProviderConfig:
    """Configuration for the local Ollama provider."""

    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL
    system_prompt: str | None = None
    tools_enabled: bool = TOOLS_ENABLED
    assistant_name: str | None = None
    user_name: str | None = None
    user_location: str | None = None
    user_timezone: str | None = None


class LLMProviderError(Exception):
    """Raised when the LLM provider encounters an error."""


TOOL_RELATED_INTENT_PATTERN = re.compile(
    r"("
    # 1. Search keywords: search, look up, look it up, browse the web, google, duckduckgo
    r"\b(?:search|browse(?:\s+the\s+web)?|google|duckduckgo)\b|"
    r"\blook\s*(?:(?:it|this|that|things?|details?|events?|news)\s+)?up\b|"
    # 2. Find recent / latest / online / on the web / out / some / more
    r"\bfind\s+(?:recent|latest|new|current|online|on\s+the\s+web|out|some|more)\b|"
    # 3. Check keywords: check the web, check for that, check that out, check it, can you check
    r"\bcheck\s+(?:for\s+)?(?:that|it|this|out|into|the\s+web|online|the\s+internet|recent|latest|news|weather|forecast|time|date|status|who|what|where|when|if|whether)\b|"
    r"\bcheck\s+(?:it|that|this)\s*out\b|"
    r"\b(?:can\s+you\s+|could\s+you\s+|let'?s\s+|why\s+don'?t\s+we\s+)?check\b|"
    # 4. Explicit tool directives
    r"\b(?:tool[\s_-]*calls?|use\s+(?:a\s+|the\s+)?tool|call\s+(?:a\s+|the\s+)?tool)\b|"
    # 5. References to previous requests
    r"\b(?:as\s+i\s+asked|i\s+asked\s+you|asked\s+you\s+to\s+do|previously\s+asked|do\s+that|do\s+it)\b|"
    # 6. Real-time / dynamic information: weather, news, headlines, current time/date
    r"\b(?:weather|forecast|headlines)\b|"
    r"\b(?:latest|breaking|recent|current|local|today'?s)?\s*news\b|"
    r"\bwhat(?:'s|\s+is)\s+happening\b|"
    r"\b(?:what\s+time\s+is\s+it|current\s+time|today'?s\s+date|current\s+date|what\s+is\s+today'?s\s+date)\b|"
    # 7. Questions asking for real-time/current facts
    r"\b(?:who\s+is|what\s+is|who\s+are)\b.*\b(?:right\s+now|currently|at\s+present)\b"
    r")",
    re.IGNORECASE,
)


def has_tool_intent(
    text: str | None,
    tool_names: set[str] | Sequence[str] | None = None,
) -> bool:
    """Check if the user request is related to a tool (e.g. search, find recent, check the web, weather, etc.)."""
    if not text:
        return False

    if TOOL_RELATED_INTENT_PATTERN.search(text):
        return True

    # Check against any installed MCP tool names or space-separated tool names
    if tool_names:
        lowered = text.lower()
        for name in tool_names:
            if name.lower() in lowered:
                return True
            space_name = name.replace("_", " ").lower()
            if space_name in lowered:
                return True

    return False


# Backward-compatible alias
has_tool_call_keyword = has_tool_intent


TOOL_PROMISE_PATTERN = re.compile(
    r"\b(?:let\s+me|i'?ll|i\s+will|i'?m\s+going\s+to|i\s+am\s+going\s+to|allow\s+me\s+to|"
    r"give\s+me\s+a\s+(?:moment|second)|one\s+moment|right\s+now)\b.{0,60}?"
    r"\b(?:check|search|look|find|pull\s+up|get|fetch|see)\b",
    re.IGNORECASE | re.DOTALL,
)

_TOOL_SELECT_STOPWORDS = frozenset(
    {"which", "where", "about", "other", "their", "using", "given", "specific", "return", "returns", "tool", "tools"}
)


def is_tool_promise(text: str | None) -> bool:
    """True if the text is a short 'let me check that' style promise rather than an answer."""
    if not text:
        return False
    return len(text) < 300 and bool(TOOL_PROMISE_PATTERN.search(text))


def select_relevant_tools(
    tools: list[dict[str, Any]], user_text: str | None
) -> list[dict[str, Any]]:
    """Keep duckduckgo_search plus only tools whose name/description words appear in the user text.

    Sending every MCP tool schema bloats the prompt and confuses small models.
    """
    words = set(re.findall(r"[a-z]{4,}", (user_text or "").lower()))
    selected: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if name == "duckduckgo_search":
            selected.append(tool)
            continue
        keywords = set(re.findall(r"[a-z]{5,}", name.lower().replace("_", " ")))
        first_sentence = (fn.get("description") or "").lower().split(".")[0]
        keywords |= set(re.findall(r"[a-z]{5,}", first_sentence))
        keywords -= _TOOL_SELECT_STOPWORDS
        if keywords & words or name.lower() in (user_text or "").lower():
            selected.append(tool)
    return selected


class LLMProvider:
    """Communicates with the local Ollama server with tool calling & fallback support."""

    def __init__(
        self,
        config: LLMProviderConfig,
        tool_manager: ToolManager | None = None,
        phrase_manager: PhraseManager | None = None,
    ) -> None:
        self._config = config
        self._tool_manager = tool_manager or ToolManager(enable_tools=config.tools_enabled)
        self._phrase_manager = phrase_manager or PhraseManager()

        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(60.0),
        )
        self._abort_event = threading.Event()

    def abort(self) -> None:
        """Signal immediate abort of active LLM streaming."""
        self._abort_event.set()

    interrupt = abort

    def start(self) -> None:
        """Verify that the Ollama server is reachable and initialize tools & phrases."""
        try:
            response = self._client.get("/api/version")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"Unable to connect to Ollama at {self._config.base_url}."
            ) from exc

        # Initialize and discover tools
        try:
            self._tool_manager.initialize()
        except Exception as exc:
            logger.warning("Error initializing ToolManager: %s", exc)

        # Load the model into memory in the background
        threading.Thread(
            target=self._warm_up, name="llm-warmup", daemon=True
        ).start()

    def _warm_up(self) -> None:
        try:
            self._client.post(
                "/api/chat",
                json={
                    "model": self._config.model,
                    "messages": [],
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "num_ctx": OLLAMA_NUM_CTX,
                        "num_predict": OLLAMA_NUM_PREDICT,
                    },
                },
                timeout=httpx.Timeout(60.0),
            )
            logger.info("LLM model warmed up with context length %d.", OLLAMA_NUM_CTX)
        except Exception:
            # Includes RuntimeError raised if the client is closed during shutdown.
            logger.debug("LLM warm-up failed.", exc_info=True)

    def stop(self) -> None:
        """Release provider resources."""
        self.abort()
        self._client.close()

    @property
    def tool_manager(self) -> ToolManager:
        return self._tool_manager

    @property
    def phrase_manager(self) -> PhraseManager:
        return self._phrase_manager

    def set_tts(self, tts: Any) -> None:
        """Register the active TTSService for on-the-go filler phrase synthesis."""
        self._phrase_manager.set_tts(tts)

    def set_speak_callback(self, callback: Callable[[str], None] | None) -> None:
        """Register a callback (e.g. tts.speak) for on-the-go filler phrase synthesis."""
        self._phrase_manager.set_speak_callback(callback)

    def _guess_tool_intent(self, messages: Sequence[Message]) -> bool:
        """Estimate whether the user's turn has a probability of requiring tool calling.

        Uses regex heuristics on the user request, checks installed MCP/local tool names,
        and inspects conversational context for follow-up lookups or confirmations.
        """
        last_user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
        if not last_user_msg or not last_user_msg.content:
            return False

        # 1. Direct regex & installed tool name match on user text
        if has_tool_intent(
            last_user_msg.content,
            tool_names=self._tool_manager.installed_tool_names,
        ):
            return True

        # 2. Conversational context: check if the user is following up on a search/lookup context
        if len(messages) >= 2:
            prev_asst_msg = next(
                (m for m in reversed(messages[:-1]) if m.role == "assistant"), None
            )
            if prev_asst_msg and prev_asst_msg.content:
                asst_lower = prev_asst_msg.content.lower()
                search_context_prompts = (
                    "search", "searching", "look up", "looking up", "look this up", "look that up",
                    "check", "checking", "find something", "happening closer to today", "uplifting instead",
                    "give me a moment", "let me see", "shall i find", "would you like to hear"
                )
                user_lower = last_user_msg.content.lower().strip().rstrip(".!?")
                followup_affirmations = (
                    "yes", "yeah", "sure", "ok", "okay", "please", "why not", "why don't we",
                    "go ahead", "do it", "and", "and...", "what about", "how about", "tell me",
                    "can you", "could you", "do that", "anything else", "more", "look it up",
                    "check for that"
                )
                if any(p in asst_lower for p in search_context_prompts) and (
                    any(user_lower.startswith(f) or user_lower == f for f in followup_affirmations)
                    or len(user_lower.split()) <= 4
                ):
                    return True

        return False

    def generate(
        self,
        messages: Sequence[Message],
        callbacks: LLMCallbacks,
    ) -> None:
        """Generate a streamed response.

        The regex is used to guess if the user's request has a probability of requiring tools.
        When so, tool context is provided to the LLM to let it decide whether to call tools.
        If the LLM decides not to call a tool, its conversational reply is accepted as-is.
        """
        self._abort_event.clear()

        maybe_requires_tools = self._guess_tool_intent(messages)
        last_user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        tools = (
            select_relevant_tools(self._tool_manager.get_ollama_tools(), last_user_text)
            if (self._config.tools_enabled and maybe_requires_tools)
            else []
        )
        sys_prompt = (
            self._config.system_prompt
            or (
                self._build_tool_system_prompt(**self._get_persona_kwargs())
                if maybe_requires_tools
                else self._build_default_system_prompt(**self._get_persona_kwargs())
            )
        )
        payload = self._build_request(messages, tools=tools, system_prompt=sys_prompt)

        try:
            tool_calls, initial_content = self._stream_or_collect_tools(payload, callbacks)

            # Safety net: parse inline tool-call text if the stream helper returned none
            if not tool_calls and initial_content:
                tool_calls = self._detect_text_tool_calls(initial_content)

            # If the LLM decided to invoke tools, execute them and stream synthesized reply
            if tool_calls and not self._abort_event.is_set():
                logger.info("LLM decided to call %d tool(s)...", len(tool_calls))
                self._execute_tools_and_stream_reply(
                    messages=messages,
                    tool_calls=tool_calls,
                    callbacks=callbacks,
                )
            elif not tool_calls and not self._abort_event.is_set():
                # If tool context was provided and LLM chose not to invoke tools: "If not then good" -
                # the LLM's conversational reply has already been streamed.
                # Only if tools were omitted due to no tool intent guess AND the model explicitly refused
                # citing lack of real-time information do we invoke fallback search.
                promised_without_calling = bool(payload.get("tools")) and is_tool_promise(initial_content)
                if self._config.tools_enabled and (
                    promised_without_calling
                    or (
                        not payload.get("tools")
                        and self._should_fallback_to_ddg_search(messages, initial_content)
                    )
                ):
                    logger.info(
                        "Model %s; invoking DuckDuckGo fallback search.",
                        "promised a lookup without calling a tool"
                        if promised_without_calling
                        else "indicated lack of real-time info",
                    )
                    last_user_query = self._build_fallback_query(messages)
                    self._execute_tools_and_stream_reply(
                        messages=messages,
                        tool_calls=[
                            {
                                "function": {
                                    "name": "duckduckgo_search",
                                    "arguments": {"query": last_user_query},
                                }
                            }
                        ],
                        callbacks=callbacks,
                    )

        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            if not self._abort_event.is_set():
                raise LLMProviderError("Failed to generate response.") from exc

    @staticmethod
    def _build_fallback_query(messages: Sequence[Message]) -> str:
        """Build a standalone search query; short follow-ups borrow the previous user turn."""
        user_msgs = [m.content for m in messages if m.role == "user" and m.content]
        if not user_msgs:
            return ""
        query = user_msgs[-1]
        if len(query.split()) < 5 and len(user_msgs) > 1:
            query = f"{user_msgs[-2]} {query}"
        return query

    def _interrupt_speech(self) -> None:
        """Stop any speech already started (e.g. premature text before a tool call arrived)."""
        tts = getattr(self._phrase_manager, "_tts", None)
        interrupt = getattr(tts, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except Exception:
                logger.debug("TTS interrupt failed.", exc_info=True)

    def _stream_or_collect_tools(
        self,
        payload: dict[str, Any],
        callbacks: LLMCallbacks,
    ) -> tuple[list[dict[str, Any]], str]:
        """Stream response while checking for tool_calls.

        If tool calls are present, buffers content instead of sending speech chunks.
        If no tool calls are present, streams tokens to callbacks for instant speech.
        """
        tool_calls: list[dict[str, Any]] = []
        collected_content = ""
        generation_started = False
        inline_syntax_seen = False
        has_tools_config = bool(payload.get("tools"))
        buffer_threshold = 80 if has_tools_config else 0
        t_start = time.perf_counter()
        t_first_tok: float | None = None

        with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if self._abort_event.is_set():
                    logger.info("LLM generation aborted by signal.")
                    break

                if not line:
                    continue

                chunk = json.loads(line)
                msg = chunk.get("message", {})

                # Check for tool_calls in chunk
                chunk_tool_calls = msg.get("tool_calls")
                if chunk_tool_calls:
                    tool_calls.extend(chunk_tool_calls)
                    # If any premature speech started before tool_calls arrived, flush/reset
                    if generation_started:
                        generation_started = False
                        self._interrupt_speech()

                content = msg.get("content", "")
                if content:
                    collected_content += content

                    # Only stream to speech if no tool calls have appeared, not inline tool call syntax, and buffer threshold is met.
                    # The prefix check is sticky so we don't re-scan the whole buffer on every chunk.
                    if not inline_syntax_seen:
                        inline_syntax_seen = collected_content.lstrip().startswith(
                            ("<tool_call", "{\"name\"", "{\"function\"", "[tool_call", "```json")
                        )
                    is_inline_tool_syntax = inline_syntax_seen or (
                        has_tools_config and is_tool_promise(collected_content)
                    )
                    if not tool_calls and not is_inline_tool_syntax and len(collected_content) >= buffer_threshold:
                        if not generation_started:
                            t_first_tok = time.perf_counter()
                            callbacks.on_generation_started()
                            generation_started = True
                        callbacks.on_generation_updated(collected_content)

            # Check if text contained inline tool calls if native tool_calls was empty
            if not tool_calls and collected_content:
                parsed_inline_calls = self._detect_text_tool_calls(collected_content)
                if parsed_inline_calls:
                    tool_calls = parsed_inline_calls
                    if generation_started:
                        generation_started = False
                        self._interrupt_speech()

            # If no tool calls occurred, flush any unstreamed content and complete turn
            suppress_promise = has_tools_config and is_tool_promise(collected_content)
            if not tool_calls and not self._abort_event.is_set() and not suppress_promise:
                if not generation_started and collected_content.strip():
                    t_first_tok = time.perf_counter()
                    callbacks.on_generation_started()
                    generation_started = True
                    callbacks.on_generation_updated(collected_content)
                if generation_started:
                    callbacks.on_generation_completed(collected_content)

        if is_benchmark_enabled():
            elapsed = time.perf_counter() - t_start
            ttft = (t_first_tok - t_start) if t_first_tok else elapsed
            if tool_calls:
                logger.info(
                    "[LLM BENCHMARK] Model emitted %d tool call(s) in %.3fs (decision latency)",
                    len(tool_calls),
                    elapsed,
                )
            else:
                logger.info(
                    "[LLM BENCHMARK] Direct LLM generation completed in %.3fs (TTFT: %.3fs, len=%d chars)",
                    elapsed,
                    ttft,
                    len(collected_content),
                )

        return tool_calls, collected_content

    def _execute_tools_and_stream_reply(
        self,
        messages: Sequence[Message],
        tool_calls: list[dict[str, Any]],
        callbacks: LLMCallbacks,
    ) -> None:
        """Execute tool calls, format results, and stream final spoken assistant answer."""
        tool_results: list[dict[str, Any]] = []

        for call in tool_calls:
            if self._abort_event.is_set():
                return

            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}

            logger.info("Executing tool '%s' with args %s", name, args)

            # Play emotional filler sentence from phrases/ (e.g. "Hmm, let me search that for you.")
            filler_info = self._phrase_manager.play_filler(name, args)
            filler_text = filler_info.get("text", "") if filler_info else ""

            try:
                callbacks.on_tool_started(name, args, filler_text)
            except TypeError:
                callbacks.on_tool_started(name, args)

            # ToolManager executes the tool, or falls back to DuckDuckGo search
            t_exec_start = time.perf_counter()
            try:
                result_str = self._tool_manager.execute_tool(name, args)
            except Exception as exc:
                logger.warning("Tool '%s' raised: %s", name, exc)
                result_str = f"Tool '{name}' failed: {exc}"
            t_exec_dur = time.perf_counter() - t_exec_start

            exec_info = getattr(self._tool_manager, "last_execution_info", {})
            actual_dur = exec_info.get("duration", t_exec_dur)
            is_fallback = exec_info.get("fallback", False)
            server_name = exec_info.get("server", "mcp")

            try:
                callbacks.on_tool_completed(
                    name,
                    result_str,
                    duration=actual_dur,
                    fallback=is_fallback,
                    server=server_name,
                )
            except TypeError:
                callbacks.on_tool_completed(name, result_str)

            tool_results.append({"name": name, "result": result_str})

        if self._abort_event.is_set():
            return

        # Prepare follow-up messages including tool context
        sliding_messages = messages[-14:] if len(messages) > 14 else messages
        conversation_context = [
            {"role": m.role if m.role in {"user", "assistant"} else "user", "content": m.content}
            for m in sliding_messages
        ]

        followup_sys_prompt = self._config.system_prompt or self._build_followup_system_prompt(**self._get_persona_kwargs())
        followup_messages = [
            {"role": "system", "content": followup_sys_prompt},
            *conversation_context,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        ]
        for res in tool_results:
            followup_messages.append(
                {
                    "role": "tool",
                    "tool_name": res["name"],
                    "content": res["result"],
                }
            )

        payload = {
            "model": self._config.model,
            "messages": followup_messages,
            "stream": True,
            "think": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        }

        # Stream the final answer directly to speech callbacks
        t_followup_start = time.perf_counter()
        t_followup_first_tok: float | None = None

        req = self._client.build_request("POST", "/api/chat", json=payload)
        response = self._client.send(req, stream=True)
        try:
            if response.status_code != 200:
                response.close()
                # Fallback to prompt-based format if model or server rejected role: tool
                tool_context = "\n\n".join(
                    f"[{res['name']} Output]:\n{res['result']}" for res in tool_results
                )

                fallback_messages = [
                    {"role": "system", "content": followup_sys_prompt},
                    *conversation_context,
                    {
                        "role": "user",
                        "content": (
                            f"Tool Execution Results:\n{tool_context}\n\n"
                            "Answer the user's question directly and conversationally using these tool results. "
                            "Adhere strictly to speech rules (2-3 short spoken sentences, plain English words only, no markdown)."
                        ),
                    },
                ]
                payload["messages"] = fallback_messages
                req = self._client.build_request("POST", "/api/chat", json=payload)
                response = self._client.send(req, stream=True)
                response.raise_for_status()
        except BaseException:
            # Don't leak the streaming connection if the retry or status check fails
            response.close()
            raise

        generation_started = False
        generated_text = ""

        try:
            for line in response.iter_lines():
                if self._abort_event.is_set():
                    logger.info("LLM generation aborted by signal during tool follow-up.")
                    break

                if not line:
                    continue

                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")

                if not content:
                    continue

                if not generation_started:
                    t_followup_first_tok = time.perf_counter()
                    callbacks.on_generation_started()
                    generation_started = True

                generated_text += content
                callbacks.on_generation_updated(generated_text)

            t_followup_end = time.perf_counter()
            fu_full_dur = max(0.0, t_followup_end - t_followup_start)
            fu_first_tok_dur = max(
                0.0,
                (t_followup_first_tok - t_followup_start)
                if t_followup_first_tok is not None
                else fu_full_dur,
            )

            # Flush content if not yet started
            if not generation_started and generated_text.strip():
                t_followup_first_tok = time.perf_counter()
                callbacks.on_generation_started()
                generation_started = True
                callbacks.on_generation_updated(generated_text)

            if generation_started and not self._abort_event.is_set():
                if is_benchmark_enabled():
                    logger.info(
                        "[LLM BENCHMARK] Tool follow-up LLM stream completed in %.3fs (TTFT: %.3fs, len=%d chars)",
                        fu_full_dur,
                        fu_first_tok_dur,
                        len(generated_text),
                    )
                    if hasattr(callbacks, "record_followup_llm"):
                        try:
                            callbacks.record_followup_llm(fu_first_tok_dur, fu_full_dur)
                        except Exception:
                            pass
                callbacks.on_generation_completed(generated_text)
        finally:
            response.close()


    _TOOL_CALL_TAG_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    @staticmethod
    def _as_tool_call(data: Any) -> dict[str, Any] | None:
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            return None
        args = data.get("arguments", data.get("parameters", {}))
        return {"function": {"name": data["name"], "arguments": args}}

    def _detect_text_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls if the model output tool call syntax as raw text."""
        # Check for <tool_call>...</tool_call> or {"name": "...", "arguments": ...}
        tool_calls: list[dict[str, Any]] = []

        for match in self._TOOL_CALL_TAG_PATTERN.finditer(text):
            try:
                call = self._as_tool_call(json.loads(match.group(1).strip()))
            except ValueError:
                continue
            if call:
                tool_calls.append(call)

        if not tool_calls and "{" in text:
            # raw_decode is string-aware (braces inside strings) and linear per object.
            decoder = json.JSONDecoder()
            idx = text.find("{")
            while idx != -1:
                try:
                    data, end = decoder.raw_decode(text, idx)
                except ValueError:
                    idx = text.find("{", idx + 1)
                    continue
                # Require an arguments key so ordinary JSON in an answer isn't treated as a call
                if isinstance(data, dict) and ("arguments" in data or "parameters" in data):
                    call = self._as_tool_call(data)
                    if call:
                        tool_calls.append(call)
                idx = text.find("{", end)

        return tool_calls


    def _should_fallback_to_ddg_search(
        self, messages: Sequence[Message], assistant_text: str
    ) -> bool:
        """Detect if the assistant replied that it lacks information when asked a question."""
        if not messages or not assistant_text:
            return False

        asst = assistant_text.lower()
        refusal_phrases = [
            "don't have access to real-time",
            "do not have access to real-time",
            "cannot browse the internet",
            "can't browse the internet",
            "no access to current",
            "no access to real-time",
            "cannot access real-time",
            "can't access real-time",
            "my knowledge cutoff",
            "as of my knowledge cutoff",
            "as an ai, i don't have access to live",
            "as an ai, i do not have access to live",
            "as an ai, i cannot provide real-time",
            "do not have live information",
            "don't have live information",
            "unable to browse",
        ]
        return any(phrase in asst for phrase in refusal_phrases)

    def _get_persona_kwargs(self) -> dict[str, str | None]:
        return {
            "assistant_name": self._config.assistant_name if self._config.assistant_name is not None else ASSISTANT_NAME,
            "user_name": self._config.user_name if self._config.user_name is not None else USER_NAME,
            "user_location": self._config.user_location if self._config.user_location is not None else USER_LOCATION,
            "user_timezone": self._config.user_timezone if self._config.user_timezone is not None else USER_TIMEZONE,
        }

    _VOICE_RULES = (
        "SPEECH RULES (STRICT):\n"
        "Your final responses are read aloud by a text-to-speech engine (Kokoro/Pocket TTS), so strictly adhere to these speech rules:\n"
        "1. Reply in 1 to 3 short, spoken sentences only.\n"
        "2. Output plain spoken English words only. NEVER output markdown formatting (no bold, asterisks, bullet points, headers, or tables), emojis, URLs, or code blocks.\n"
        "3. NEVER include asterisked or bracketed actions or expressive tags (e.g. do NOT write *laughs*, *smiles*, [sighs], or *chuckles*).\n"
        "4. Spell out numbers, measurements, and symbols in full words (e.g., 'three kilometers' not '3km', 'percent' not '%').\n"
        "5. Use only standard conversational punctuation (periods, commas, question marks). Do not use asterisks, hashes, backticks, or dashes.\n"
        "6. Never address the user by any name unless their name is explicitly provided in the persona instructions above."
    )

    EMOTION_SYSTEM_PROMPT = (
        "You are a helpful voice assistant named Waffle. "
        "Your responses are spoken aloud by an emotional text-to-speech engine. "
        "You MUST follow these rules strictly:\n"
        "1. You MUST specify your emotional state at the very start of your response using one of the available emotion tags: "
        "[neutral], [happy], [sad], or [angry]. Example: '[happy] That is wonderful news! I am glad to assist.'\n"
        "2. If your emotion changes mid-reply, you may prefix sentences with a new emotion tag, e.g. '[sad] I am sorry that happened. [neutral] Let us figure out what to do.'\n"
        "3. Only use the 4 available emotion tags: [neutral], [happy], [sad], [angry].\n"
        "4. Reply in 2-3 short sentences only.\n"
        "5. Use plain English words only — no emojis, no symbols, no markdown, no bullet points, no asterisks, no hashtags.\n"
        "6. Write out numbers and units in full (e.g. 'three kilometres' not '3km').\n"
        "7. Do not include any special characters other than standard punctuation (period, comma, question mark, exclamation mark) and the bracketed emotion tag."
    )

    @classmethod
    def _build_persona_prefix(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
    ) -> str:
        asst_name = assistant_name if assistant_name is not None else ASSISTANT_NAME
        u_name = user_name if user_name is not None else USER_NAME
        u_loc = user_location if user_location is not None else USER_LOCATION
        u_tz = user_timezone if user_timezone is not None else USER_TIMEZONE

        parts = []
        if asst_name and asst_name.strip():
            parts.append(
                f"You are a helpful voice assistant named {asst_name.strip()}. You speak in a natural, calm, and friendly conversational tone."
            )
        else:
            parts.append(
                "You are a helpful voice assistant. You speak in a natural, calm, and friendly conversational tone."
            )

        if u_name and u_name.strip():
            parts.append(f"You are speaking with {u_name.strip()}.")
        else:
            parts.append(
                "The user's name is not provided. Never assume, invent, or use any name to address or refer to the user."
            )

        if u_loc and u_loc.strip():
            parts.append(f"The user's current location is {u_loc.strip()}.")
        if u_tz and u_tz.strip():
            parts.append(f"The user's timezone is {u_tz.strip()}.")

        now = datetime.now()
        if u_tz and u_tz.strip():
            try:
                now = datetime.now(ZoneInfo(u_tz.strip()))
            except Exception:
                logger.debug("Invalid timezone %r; using local time.", u_tz)
        parts.append(f"Current date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p')}.")
        return " ".join(parts)

    @classmethod
    def _build_default_system_prompt(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
    ) -> str:
        prefix = cls._build_persona_prefix(
            assistant_name=assistant_name,
            user_name=user_name,
            user_location=user_location,
            user_timezone=user_timezone,
        )
        return (
            f"{prefix}\n\n"
            "Be directly helpful, concise, and engaging in your spoken answers.\n\n"
            f"{cls._VOICE_RULES}"
        )

    @classmethod
    def _build_tool_system_prompt(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
    ) -> str:
        prefix = cls._build_persona_prefix(
            assistant_name=assistant_name,
            user_name=user_name,
            user_location=user_location,
            user_timezone=user_timezone,
        )
        return (
            f"{prefix}\n\n"
            "TOOL CALLING POLICY (MANDATORY):\n"
            "You have access to live tools, including DuckDuckGo search (duckduckgo_search) and MCP tools.\n"
            "Whenever the user asks for current news, weather, real-time facts, web search, time, dates, or actions handled by tools:\n"
            "- Proactively invoke the appropriate tool IMMEDIATELY in this turn instead of answering from pretraining memory or guessing.\n"
            "- For web searches, latest news, weather, or current events, invoke duckduckgo_search.\n"
            "- If an installed MCP tool matches the function, invoke that tool.\n"
            "- If the user's message does not require live information or external tools, reply directly with conversational speech without invoking any tools.\n"
            "- NEVER emit conversational text saying you are searching, looking up, checking updates, or asking for time (e.g. do NOT say 'I am searching', 'looking that up', 'checking updates', 'please give me a moment', or 'I will be with you in a moment'). You cannot search in the background. If you need information, you MUST execute the tool call right now.\n"
            "- Formulate complete, standalone search queries: if the user refers to previous context or pronouns (e.g., 'search it', 'look that up', 'check for that'), resolve the topic into a complete standalone search query (e.g. query='positive news events New Delhi').\n"
            "- NEVER say you cannot browse the internet, do not have real-time information, or cite a knowledge cutoff date. Always execute the appropriate tool to look up the answer.\n\n"
            f"{cls._VOICE_RULES}"
        )

    @classmethod
    def _build_followup_system_prompt(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
    ) -> str:
        prefix = cls._build_persona_prefix(
            assistant_name=assistant_name,
            user_name=user_name,
            user_location=user_location,
            user_timezone=user_timezone,
        )
        return (
            f"{prefix}\n\n"
            "You have executed tools and retrieved real-time information to answer the user's inquiry.\n"
            "Synthesize the tool results into a clear, direct, and conversational spoken answer to the user's question.\n\n"
            f"{cls._VOICE_RULES}"
        )

    _SYSTEM_PROMPT = _VOICE_RULES
    _DEFAULT_SYSTEM_PROMPT = _SYSTEM_PROMPT
    _TOOL_SYSTEM_PROMPT = _SYSTEM_PROMPT
    _FOLLOWUP_SYSTEM_PROMPT = _SYSTEM_PROMPT

    def _build_request(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build an Ollama chat request with sliding window guard and optional tools."""
        sliding_messages = messages[-16:] if len(messages) > 16 else messages

        system_message = {
            "role": "system",
            "content": (
                system_prompt
                or self._config.system_prompt
                or self._build_default_system_prompt(**self._get_persona_kwargs())
            ),
        }

        request_payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [system_message] + [
                {
                    "role": message.role if message.role in {"user", "assistant"} else "user",
                    "content": message.content,
                }
                for message in sliding_messages
            ],
            "stream": True,
            "think": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        }

        if tools:
            request_payload["tools"] = tools

        return request_payload
