import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import httpx

from constants import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    TOOLS_ENABLED,
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

    def on_tool_completed(self, tool_name: str, result: str) -> None:
        """Called when a tool execution has finished."""
        pass



@dataclass(slots=True)
class LLMProviderConfig:
    """Configuration for the local Ollama provider."""

    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL
    tools_enabled: bool = TOOLS_ENABLED


class LLMProviderError(Exception):
    """Raised when the LLM provider encounters an error."""


TOOL_RELATED_INTENT_PATTERN = re.compile(
    r"("
    # 1. Search keywords: search, look up, browse the web, google, duckduckgo
    r"\b(?:search|look\s*up|browse(?:\s+the\s+web)?|google|duckduckgo)\b|"
    # 2. Find recent / latest / online / on the web / out
    r"\bfind\s+(?:recent|latest|new|current|online|on\s+the\s+web|out)\b|"
    # 3. Check the web / online / recent / latest / news / weather / time / date
    r"\bcheck\s+(?:the\s+web|online|the\s+internet|recent|latest|news|weather|forecast|time|date|status|who|what|where|when|if|whether)\b|"
    # 4. Explicit tool directives
    r"\b(?:tool[\s_-]*calls?|use\s+(?:a\s+|the\s+)?tool|call\s+(?:a\s+|the\s+)?tool)\b|"
    # 5. Real-time / dynamic information: weather, news, current time/date
    r"\b(?:weather|forecast)\b|"
    r"\b(?:latest|breaking|recent)\s+news\b|"
    r"\b(?:what\s+time\s+is\s+it|current\s+time|today'?s\s+date|current\s+date|what\s+is\s+today'?s\s+date)\b|"
    # 6. Questions asking for real-time/current facts
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

    def generate(
        self,
        messages: Sequence[Message],
        callbacks: LLMCallbacks,
    ) -> None:
        """Generate a streamed response. Tools are only sent if the user specifically requested 'Tool call'."""
        self._abort_event.clear()

        last_user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
        tool_call_requested = (
            has_tool_call_keyword(
                last_user_msg.content,
                tool_names=self._tool_manager.installed_tool_names,
            )
            if last_user_msg
            else False
        )

        tools = (
            self._tool_manager.get_ollama_tools()
            if (self._config.tools_enabled and tool_call_requested)
            else []
        )
        sys_prompt = self._TOOL_SYSTEM_PROMPT if tool_call_requested else self._DEFAULT_SYSTEM_PROMPT
        payload = self._build_request(messages, tools=tools, system_prompt=sys_prompt)

        try:
            tool_calls, initial_content = self._stream_or_collect_tools(payload, callbacks)

            # Check if text contained inline tool calls if native tool_calls was empty
            if tool_call_requested and not tool_calls and initial_content:
                parsed_calls = self._detect_text_tool_calls(initial_content)
                if parsed_calls:
                    tool_calls = parsed_calls

            # If tool calls were requested, execute them and generate final answer
            if tool_calls and not self._abort_event.is_set():
                logger.info("Processing %d tool call(s)...", len(tool_calls))
                self._execute_tools_and_stream_reply(
                    messages=messages,
                    tool_calls=tool_calls,
                    callbacks=callbacks,
                )
            elif not tool_calls and not self._abort_event.is_set():
                # Only fallback to search if the user explicitly requested a tool call
                if tool_call_requested and self._should_fallback_to_ddg_search(messages, initial_content):
                    logger.info("Model indicated lack of real-time info; invoking DuckDuckGo fallback search.")
                    last_user_query = messages[-1].content if messages else ""
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

                    # Only stream to speech if no tool calls have appeared and buffer threshold is met
                    if not tool_calls and len(collected_content) >= buffer_threshold:
                        if not generation_started:
                            callbacks.on_generation_started()
                            generation_started = True
                        callbacks.on_generation_updated(collected_content)

            # If no tool calls occurred, flush any unstreamed content and complete turn
            if not tool_calls and not self._abort_event.is_set():
                if not generation_started and collected_content.strip():
                    callbacks.on_generation_started()
                    generation_started = True
                    callbacks.on_generation_updated(collected_content)
                if generation_started:
                    callbacks.on_generation_completed(collected_content)

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
            result_str = self._tool_manager.execute_tool(name, args)
            callbacks.on_tool_completed(name, result_str)


            tool_results.append(
                {
                    "name": name,
                    "result": result_str,
                }
            )

        if self._abort_event.is_set():
            return

        # Prepare follow-up messages including tool context
        # Formulate a clear prompt with tool results for the assistant to speak
        tool_context_lines = ["Tool Execution Results:"]
        for res in tool_results:
            tool_context_lines.append(f"[{res['name']} Output]:\n{res['result']}")
        tool_context = "\n\n".join(tool_context_lines)

        sliding_messages = messages[-14:] if len(messages) > 14 else messages
        conversation_context = [
            {"role": m.role if m.role in {"user", "assistant"} else "user", "content": m.content}
            for m in sliding_messages
        ]

        # Use native Ollama tool execution schema: assistant tool_calls + role: tool results
        followup_messages = [
            {"role": "system", "content": self._FOLLOWUP_SYSTEM_PROMPT},
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
        req = self._client.build_request("POST", "/api/chat", json=payload)
        response = self._client.send(req, stream=True)
        if response.status_code != 200:
            response.close()
            # Fallback to prompt-based format if model or server rejected role: tool
            tool_context_lines = ["Tool Execution Results:"]
            for res in tool_results:
                tool_context_lines.append(f"[{res['name']} Output]:\n{res['result']}")
            tool_context = "\n\n".join(tool_context_lines)

            fallback_messages = [
                {"role": "system", "content": self._FOLLOWUP_SYSTEM_PROMPT},
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
                    callbacks.on_generation_started()
                    generation_started = True

                generated_text += content
                callbacks.on_generation_updated(generated_text)

            if generation_started and not self._abort_event.is_set():
                callbacks.on_generation_completed(generated_text)
        finally:
            response.close()

    def _detect_text_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls if the model output tool call syntax as raw text."""
        # Check for <tool_call>...</tool_call> or {"name": "...", "arguments": ...}
        tool_calls: list[dict[str, Any]] = []

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
            "don't have real-time",
            "do not have real-time",
            "my knowledge cutoff",
            "i don't have internet access",
            "i cannot provide current",
            "as an ai, i cannot",
        ]
        return any(phrase in asst for phrase in refusal_phrases)

    _DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful voice assistant named Waffle. You speak in a natural, calm, and friendly conversational tone.\n\n"
        "SPEECH RULES (STRICT):\n"
        "Your final responses are read aloud by a text-to-speech engine (Kokoro/Pocket TTS), so strictly adhere to these speech rules:\n"
        "1. Reply in 2-3 short, spoken sentences only.\n"
        "2. Output plain spoken English words only. NEVER output markdown tables, bullet points, headers, emojis, URLs, or code blocks.\n"
        "3. NEVER include asterisked or bracketed actions or expressive tags (e.g. do NOT write *laughs*, *smiles*, [sighs], or *chuckles*).\n"
        "4. Spell out numbers, measurements, and symbols in full words (e.g., 'three kilometers' not '3km', 'percent' not '%').\n"
        "5. Use only standard conversational punctuation (periods, commas, question marks). Do not use asterisks, hashes, backticks, or dashes."
    )

    _TOOL_SYSTEM_PROMPT = (
        "You are a helpful voice assistant named Waffle. You speak in a natural, calm, and friendly conversational tone.\n\n"
        "TOOL CALLING POLICY (MANDATORY):\n"
        "You have access to tools. Your internal pretraining memory is strictly historical and in the past. You do NOT have access to current real-world facts from memory.\n"
        "Whenever the user asks a question:\n"
        "- If an installed tool matches the function (for example, get_current_time for date/time/day/year, or filesystem/database tools), you MUST call that tool.\n"
        "- If no installed tool matches the function or you need external facts, news, weather, current officeholders, or web details, you MUST call duckduckgo_search.\n"
        "- NEVER guess or answer real-time facts, dates, time, or current affairs from internal memory. Always call the appropriate tool.\n\n"
        "SPEECH RULES (STRICT):\n"
        "Your final responses are read aloud by a text-to-speech engine (Kokoro/Pocket TTS), so strictly adhere to these speech rules:\n"
        "1. Reply in 2-3 short, spoken sentences only.\n"
        "2. Output plain spoken English words only. NEVER output markdown tables, bullet points, headers, emojis, URLs, or code blocks.\n"
        "3. NEVER include asterisked or bracketed actions or expressive tags (e.g. do NOT write *laughs*, *smiles*, [sighs], or *chuckles*).\n"
        "4. Spell out numbers, measurements, and symbols in full words (e.g., 'three kilometers' not '3km', 'percent' not '%').\n"
        "5. Use only standard conversational punctuation (periods, commas, question marks). Do not use asterisks, hashes, backticks, or dashes."
    )

    _SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT

    _FOLLOWUP_SYSTEM_PROMPT = (
        "You are a helpful voice assistant named Waffle. You speak in a natural, calm, and friendly conversational tone.\n\n"
        "You have executed tools and retrieved real-time information to answer the user's inquiry.\n"
        "Directly and conversationally answer the user's question using the provided tool results.\n\n"
        "SPEECH RULES (STRICT):\n"
        "Your final responses are read aloud by a text-to-speech engine (Kokoro/Pocket TTS), so strictly adhere to these speech rules:\n"
        "1. Reply in 2-3 short, spoken sentences only.\n"
        "2. Output plain spoken English words only. NEVER output markdown tables, bullet points, headers, emojis, URLs, or code blocks.\n"
        "3. NEVER include asterisked or bracketed actions or expressive tags (e.g. do NOT write *laughs*, *smiles*, [sighs], or *chuckles*).\n"
        "4. Spell out numbers, measurements, and symbols in full words (e.g., 'twenty-four degrees Celsius', 'percent' not '%').\n"
        "5. Use only standard conversational punctuation (periods, commas, question marks). Do not use asterisks, hashes, backticks, or dashes."
    )

    def _build_request(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build an Ollama chat request with sliding window guard and optional tools."""
        sliding_messages = messages[-16:] if len(messages) > 16 else messages

        system_message = {"role": "system", "content": system_prompt or self._DEFAULT_SYSTEM_PROMPT}

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
