import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import httpx

from constants import (
    ASSISTANT_NAME,
    ASSISTANT_PERSONALITY,
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
from .personalities import build_personality_prompt, get_personality

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
    tools_enabled: bool = TOOLS_ENABLED
    assistant_name: str | None = None
    user_name: str | None = None
    user_location: str | None = None
    user_timezone: str | None = None
    personality: str | None = None


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
        except httpx.HTTPError:
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

        tools = (
            self._tool_manager.get_ollama_tools()
            if self._config.tools_enabled
            else []
        )
        sys_prompt = (
            self._build_tool_system_prompt(**self._get_persona_kwargs())
            if self._config.tools_enabled
            else self._build_default_system_prompt(**self._get_persona_kwargs())
        )
        payload = self._build_request(messages, tools=tools, system_prompt=sys_prompt)

        try:
            tool_calls, initial_content = self._stream_or_collect_tools(payload, callbacks)

            # Check if text contained inline tool calls if native tool_calls was empty
            if not tool_calls and initial_content:
                parsed_calls = self._detect_text_tool_calls(initial_content)
                if parsed_calls:
                    tool_calls = parsed_calls

            # If the LLM decided to invoke tools, execute them and stream synthesized reply
            if tool_calls and not self._abort_event.is_set():
                logger.info("LLM decided to call %d tool(s)...", len(tool_calls))
                self._execute_tools_and_stream_reply(
                    messages=messages,
                    tool_calls=tool_calls,
                    callbacks=callbacks,
                )
            elif not tool_calls and not self._abort_event.is_set():
                # If the LLM output stalling phrases pretending to look things up or cited lack of real-time info,
                # immediately rescue with DuckDuckGo search so the user receives a real answer.
                if (
                    self._config.tools_enabled
                    and self._should_fallback_to_ddg_search(messages, initial_content)
                ):
                    logger.info("Model indicated lack of real-time info or stalled; invoking DuckDuckGo fallback search.")
                    fallback_query = self._extract_fallback_query(messages, initial_content)
                    self._execute_tools_and_stream_reply(
                        messages=messages,
                        tool_calls=[
                            {
                                "function": {
                                    "name": "duckduckgo_search",
                                    "arguments": {"query": fallback_query},
                                }
                            }
                        ],
                        callbacks=callbacks,
                    )

        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            if not self._abort_event.is_set():
                raise LLMProviderError("Failed to generate response.") from exc

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
                        if hasattr(self._phrase_manager, "_tts") and hasattr(
                            self._phrase_manager._tts, "interrupt"
                        ):
                            try:
                                self._phrase_manager._tts.interrupt()
                            except Exception:
                                pass

                content = msg.get("content", "")
                if content:
                    collected_content += content

                    # Only stream to speech if tools are not configured in payload
                    if not has_tools_config and not tool_calls:
                        if not generation_started:
                            t_first_tok = time.perf_counter()
                            callbacks.on_generation_started()
                            generation_started = True
                        callbacks.on_generation_updated(collected_content)

            # If no tool calls occurred, flush any unstreamed content and complete turn
            if not tool_calls and not self._abort_event.is_set():
                needs_fallback = (
                    self._config.tools_enabled
                    and self._should_fallback_to_ddg_search(payload.get("messages", []), collected_content)
                )
                if not needs_fallback:
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
        max_rounds: int = 3,
    ) -> None:
        """Execute tool calls, format results, and stream final spoken assistant answer across up to max_rounds."""
        sliding_messages = messages[-14:] if len(messages) > 14 else messages
        conversation_context = [
            {"role": m.role if m.role in {"user", "assistant"} else "user", "content": m.content}
            for m in sliding_messages
        ]

        followup_sys_prompt = self._build_followup_system_prompt(**self._get_persona_kwargs())
        followup_messages = [
            {"role": "system", "content": followup_sys_prompt},
            *conversation_context,
        ]

        current_tool_calls = tool_calls

        for round_idx in range(max_rounds):
            if self._abort_event.is_set():
                return

            round_tool_results: list[dict[str, Any]] = []

            for call in current_tool_calls:
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

                logger.info("Executing tool '%s' with args %s (round %d)", name, args, round_idx + 1)

                filler_info = self._phrase_manager.play_filler(name, args)
                filler_text = filler_info.get("text", "") if filler_info else ""

                try:
                    callbacks.on_tool_started(name, args, filler_text)
                except TypeError:
                    callbacks.on_tool_started(name, args)

                t_exec_start = time.perf_counter()
                result_str = self._tool_manager.execute_tool(name, args)
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

                round_tool_results.append({"name": name, "result": result_str})

            followup_messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": current_tool_calls,
                }
            )
            for res in round_tool_results:
                followup_messages.append(
                    {
                        "role": "tool",
                        "content": res["result"],
                    }
                )

            # Only provide tools for subsequent rounds if a preparatory tool (e.g. geocoding) was executed
            chainable_tools = {"geocode_location", "batch_geocode_locations"}
            needs_further_tool = any(
                call.get("function", {}).get("name") in chainable_tools
                for call in current_tool_calls
            )
            is_last_round = (round_idx + 1 >= max_rounds) or not needs_further_tool
            raw_tools = (
                self._tool_manager.get_ollama_tools()
                if (not is_last_round and self._config.tools_enabled)
                else None
            )
            tools_for_payload = raw_tools if isinstance(raw_tools, list) and raw_tools else None

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
            if tools_for_payload:
                payload["tools"] = tools_for_payload

            t_followup_start = time.perf_counter()
            t_followup_first_tok: float | None = None

            req = self._client.build_request("POST", "/api/chat", json=payload)
            response = self._client.send(req, stream=True)
            if response.status_code != 200:
                response.close()
                # Fallback to prompt-based format if model or server rejected role: tool
                tool_context_lines = ["Tool Execution Results:"]
                for res in round_tool_results:
                    tool_context_lines.append(f"[{res['name']} Output]:\n{res['result']}")
                tool_context = "\n\n".join(tool_context_lines)

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
                payload.pop("tools", None)
                req = self._client.build_request("POST", "/api/chat", json=payload)
                response = self._client.send(req, stream=True)
                response.raise_for_status()

            generation_started = False
            generated_text = ""
            next_tool_calls: list[dict[str, Any]] = []

            try:
                for line in response.iter_lines():
                    if self._abort_event.is_set():
                        logger.info("LLM generation aborted by signal during tool follow-up.")
                        break

                    if not line:
                        continue

                    chunk = json.loads(line)
                    msg = chunk.get("message", {})

                    chunk_tc = msg.get("tool_calls")
                    if chunk_tc:
                        next_tool_calls.extend(chunk_tc)

                    content = msg.get("content", "")
                    if content:
                        generated_text += content
                        # Only stream to speech if no tool calls are being made
                        if not next_tool_calls and not tools_for_payload:
                            if not generation_started:
                                t_followup_first_tok = time.perf_counter()
                                callbacks.on_generation_started()
                                generation_started = True
                            callbacks.on_generation_updated(generated_text)

                t_followup_end = time.perf_counter()
                fu_full_dur = max(0.0, t_followup_end - t_followup_start)
                fu_first_tok_dur = max(
                    0.0,
                    (t_followup_first_tok - t_followup_start)
                    if t_followup_first_tok is not None
                    else fu_full_dur,
                )

                # Check if text contained inline tool calls if native tool_calls was empty
                if not next_tool_calls and generated_text:
                    parsed = self._detect_text_tool_calls(generated_text)
                    if parsed:
                        next_tool_calls = parsed

                # If the model requested another tool call and we have rounds left, continue loop
                if next_tool_calls and not is_last_round and not self._abort_event.is_set():
                    current_tool_calls = next_tool_calls
                    continue

                # Final speech delivery
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
                    callbacks.on_generation_completed(generated_text)
                    if hasattr(callbacks, "record_followup_llm"):
                        try:
                            callbacks.record_followup_llm(fu_first_tok_dur, fu_full_dur)
                        except Exception:
                            pass
                break
            finally:
                response.close()

    def _detect_text_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls if the model output tool call syntax as raw text."""
        tool_calls: list[dict[str, Any]] = []

        # 1. Natural DuckDuckGo invocation in text: e.g. "DuckDuckGo, fetch latest headlines"
        ddg_match = re.search(
            r"\bduckduckgo[,\s]+(?:fetch|search|query|find|look up)\s+(?:the\s+)?([^.!?\n]+)",
            text,
            re.IGNORECASE,
        )
        if ddg_match:
            q = ddg_match.group(1).strip()
            from constants import USER_LOCATION
            loc = USER_LOCATION or "New Delhi"
            if loc and loc.lower().split(",")[0] not in q.lower():
                q = f"{q} {loc}"
            tool_calls.append(
                {
                    "function": {
                        "name": "duckduckgo_search",
                        "arguments": {"query": q},
                    }
                }
            )
            return tool_calls

        tag_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        for match in tag_pattern.finditer(text):
            try:
                data = json.loads(match.group(1).strip())
                if "name" in data:
                    tool_calls.append(
                        {
                            "function": {
                                "name": data["name"],
                                "arguments": data.get("arguments", {}),
                            }
                        }
                    )
            except Exception:
                pass

        if not tool_calls and "{" in text:
            start_idx = 0
            while start_idx < len(text):
                start = text.find("{", start_idx)
                if start == -1:
                    break
                brace_count = 0
                end = -1
                for i in range(start, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                if end != -1:
                    candidate = text[start:end]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "name" in data:
                            tool_calls.append(
                                {
                                    "function": {
                                        "name": data["name"],
                                        "arguments": data.get("arguments", {}),
                                    }
                                }
                            )
                    except Exception:
                        pass
                    start_idx = end
                else:
                    start_idx = start + 1

        return tool_calls

    _STALLING_PATTERN = re.compile(
        r"\b(?:"
        r"let\s+me\s+(?:(?:\w+)\s+)?(?:check|find|look|get|see|search|pull|grab|give|provide|fetch|share|tell)\b|"
        r"(?:i\s*['’]?ll|i\s+will|i\s+can)\s+(?:(?:\w+)\s+)?(?:check|look|get|find|see|search|pull|grab|give|provide|fetch|share|tell|help\s+you)\b|"
        r"(?:give|allow)\s+(?:me\s+)?(?:just\s+)?(?:a\s+)?(?:moment|sec|second|minute)\b|"
        r"(?:be\s+with\s+you\s+in\s+(?:just\s+)?a\s+moment)\b|"
        r"(?:searching|looking\s+up|checking)\s+(?:for|into|the|now)\b|"
        r"duckduckgo[,\s]+(?:fetch|search|find|query)\b|"
        r"hold\s+on\b|"
        r"moment\s+to\s+check\b|"
        r"don['’]?t\s+have\s+(?:access\s+to\s+)?(?:real-?time|live|current)\b|"
        r"do\s+not\s+have\s+(?:access\s+to\s+)?(?:real-?time|live|current)\b|"
        r"cannot\s+browse|can['’]?t\s+browse|unable\s+to\s+browse\b|"
        r"my\s+knowledge\s+cutoff\b|"
        r"as\s+an\s+ai\b|"
        r"stop\s+nagging\b"
        r")",
        re.IGNORECASE,
    )

    def _should_fallback_to_ddg_search(
        self, messages: Sequence[Message] | Sequence[dict[str, Any]], assistant_text: str
    ) -> bool:
        """Detect if the assistant replied with refusal, stalling, or promising to check without executing."""
        if not messages or not assistant_text:
            return False

        return bool(self._STALLING_PATTERN.search(assistant_text))

    @classmethod
    def _extract_fallback_query(
        cls, messages: Sequence[Message] | Sequence[dict[str, Any]], assistant_text: str
    ) -> str:
        """Derive an effective, standalone search query from assistant stalling text or conversation context."""
        asst = assistant_text.strip()
        from constants import USER_LOCATION
        loc = USER_LOCATION or "New Delhi"

        # 1. Check if the assistant text described what it was 'searching for'
        match = re.search(
            r"(?:searching for|looking up|checking (?:the )?latest updates (?:in|on)|search for|duckduckgo[,\s]+(?:fetch|search|find))\s+([^.!?]+)",
            asst,
            re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(
                r"\b(?:some|very soon|right now|please give me a moment|scheduled very soon|details)\b",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()
            if len(candidate) > 4:
                is_local = any(k in candidate.lower() for k in ["news", "weather", "events", "traffic", "local", "city"])
                if is_local and loc and loc.lower().split(",")[0] not in candidate.lower():
                    candidate = f"{candidate} {loc}"
                return re.sub(r"\s+", " ", candidate).strip()

        # 2. Check conversation history backwards for a topical user query
        for msg in reversed(messages):
            role = msg.role if hasattr(msg, "role") else msg.get("role")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == "user" and content:
                text = content.strip()
                text_lower = text.lower()
                if text_lower in {"and", "and...", "what about", "can you look it up?", "can you search it right now", "tell me", "no tell me", "no, tell me!"}:
                    continue
                is_local = any(k in text_lower for k in ["news", "weather", "temperature", "events", "traffic", "local", "city", "happenings"])
                if is_local and loc and loc.lower().split(",")[0] not in text_lower:
                    return f"{text} {loc}"
                return text

        # 3. Default back to last user message or general query
        last_user = ""
        if messages:
            last_msg = messages[-1]
            last_user = (last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")).strip()
        return last_user or (f"latest news {loc}".strip() if loc else "latest news")

    def _get_persona_kwargs(self) -> dict[str, str | None]:
        return {
            "assistant_name": self._config.assistant_name if self._config.assistant_name is not None else ASSISTANT_NAME,
            "user_name": self._config.user_name if self._config.user_name is not None else USER_NAME,
            "user_location": self._config.user_location if self._config.user_location is not None else USER_LOCATION,
            "user_timezone": self._config.user_timezone if self._config.user_timezone is not None else USER_TIMEZONE,
            "personality": self._config.personality if self._config.personality is not None else ASSISTANT_PERSONALITY,
        }

    _VOICE_RULES = (
        "SPEECH RULES (STRICT):\n"
        "Your final responses are read aloud by a text-to-speech engine (Kokoro/Pocket TTS), so strictly adhere to these speech rules:\n"
        "1. Reply in 1 to 3 short, spoken sentences only.\n"
        "2. Output plain spoken English words only. NEVER output markdown formatting (no bold, asterisks, bullet points, headers, or tables), emojis, URLs, or code blocks.\n"
        "3. NEVER include asterisked or bracketed actions or expressive tags (e.g. do NOT write *laughs*, *smiles*, [sighs], or *chuckles*).\n"
        "4. Spell out numbers, measurements, and symbols in full words (e.g., 'three kilometers' not '3km', 'percent' not '%').\n"
        "5. Use only standard conversational punctuation (periods, commas, question marks). Do not use asterisks, hashes, backticks, or dashes.\n"
        "6. Never address the user by any name unless their name is explicitly provided in the persona instructions above.\n"
        "7. Never address the user by their name unless the context is very serious.\n"
        "8. Never mention user location unless user has asked a question requiring their location data."
    )

    @classmethod
    def _build_persona_prefix(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
        personality: str | None = None,
    ) -> str:
        asst_name = assistant_name if assistant_name is not None else ASSISTANT_NAME
        u_name = user_name if user_name is not None else USER_NAME
        u_loc = user_location if user_location is not None else USER_LOCATION
        u_tz = user_timezone if user_timezone is not None else USER_TIMEZONE
        pers = personality if personality is not None else ASSISTANT_PERSONALITY

        parts = []
        parts.append(build_personality_prompt(asst_name, pers))

        if u_name and u_name.strip():
            parts.append(f"You are speaking with {u_name.strip()}. Address the user by this name as per the speech rules.")
        else:
            parts.append(
                "The user's name is not provided. Never assume, invent, or use any name to address or refer to the user."
            )

        if u_loc and u_loc.strip():
            parts.append(
                f"The user's current location is {u_loc.strip()}. "
                "Always assume questions about local weather, news, time, or events refer to this location unless the user explicitly specifies another place. "
                "Never ask the user for their location if it is already known."
            )
        if u_tz and u_tz.strip():
            parts.append(f"The user's timezone is {u_tz.strip()}.")

        now = datetime.now()
        if u_tz and u_tz.strip():
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(u_tz.strip()))
            except Exception:
                pass
        parts.append(f"Current date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p')}.")
        return " ".join(parts)

    @classmethod
    def _build_default_system_prompt(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
        personality: str | None = None,
    ) -> str:
        prefix = cls._build_persona_prefix(
            assistant_name=assistant_name,
            user_name=user_name,
            user_location=user_location,
            user_timezone=user_timezone,
            personality=personality,
        )
        return (
            f"{prefix}\n\n"
            "Be directly helpful, concise, and engaging in your spoken answers.\n"
            "Always maintain conversational continuity across turns. If the user refers to a topic, person, or location from earlier in the conversation, remember and use that context naturally.\n"
            "If the user asks whether you called a tool or looked something up, answer truthfully based on the tool calls in the conversation history.\n\n"
            f"{cls._VOICE_RULES}"
        )

    @classmethod
    def _build_tool_system_prompt(
        cls,
        assistant_name: str | None = None,
        user_name: str | None = None,
        user_location: str | None = None,
        user_timezone: str | None = None,
        personality: str | None = None,
    ) -> str:
        prefix = cls._build_persona_prefix(
            assistant_name=assistant_name,
            user_name=user_name,
            user_location=user_location,
            user_timezone=user_timezone,
            personality=personality,
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
            "- CONVERSATION CONTEXT & RESOLUTION: Always maintain conversational continuity across turns. If the user refers to previous context or pronouns (e.g., 'search it', 'look that up', 'check for that', 'what about the weather', 'is it raining?'), resolve the subject, topic, and location from the conversation history into a complete standalone tool call query (e.g. query='current weather in New Delhi India'). Never ask the user for information (like their city or what to search) if they already mentioned it or if their location is already known.\n"
            "- TOOL CALL MEMORY: When the user asks about past actions or tools used earlier in the conversation (e.g., 'Did you look that up?', 'What tool did you call?'), refer accurately to the tool execution history in the conversation.\n"
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
        personality: str | None = None,
    ) -> str:
        prefix = cls._build_persona_prefix(
            assistant_name=assistant_name,
            user_name=user_name,
            user_location=user_location,
            user_timezone=user_timezone,
            personality=personality,
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
        """Build an Ollama chat request preserving full message roles and tool calls."""
        sliding_messages = messages[-16:] if len(messages) > 16 else messages

        system_message = {
            "role": "system",
            "content": system_prompt or self._build_default_system_prompt(**self._get_persona_kwargs()),
        }

        serialized_messages: list[dict[str, Any]] = [system_message]
        for message in sliding_messages:
            msg_dict: dict[str, Any] = {
                "role": message.role,
                "content": message.content or "",
            }
            if message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
            serialized_messages.append(msg_dict)

        request_payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": serialized_messages,
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
