"""Unit tests for MCP tool integration, DuckDuckGo search fallback, and ToolManager."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from models import Message
from services.llm.src.llm_provider import (
    LLMCallbacks,
    LLMProvider,
    LLMProviderConfig,
    has_tool_call_keyword,
)
from services.tools import (
    DuckDuckGoSearch,
    MCPClientManager,
    MCPServerConfig,
    MCPToolDefinition,
    PhraseManager,
    ToolManager,
)



class TestDuckDuckGoSearch(unittest.TestCase):
    def test_empty_query_returns_message(self) -> None:
        searcher = DuckDuckGoSearch()
        result = searcher.search("")
        self.assertIn("No search query provided", result)

    @patch("services.tools.src.search.DDGS")
    def test_successful_search_formatting(self, mock_ddgs_class: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {"title": "Result One", "body": "Snippet 1 text"},
            {"title": "Result Two", "body": "Snippet 2 text"},
        ]
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs_class.return_value = mock_instance

        searcher = DuckDuckGoSearch(max_results=2)
        res = searcher.search("test query")

        self.assertIn("Search results for 'test query':", res)
        self.assertIn("1. Result One: Snippet 1 text", res)
        self.assertIn("2. Result Two: Snippet 2 text", res)

    @patch("services.tools.src.search.DDGS")
    def test_search_error_handling(self, mock_ddgs_class: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.text.side_effect = RuntimeError("Network timeout")
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs_class.return_value = mock_instance

        searcher = DuckDuckGoSearch()
        res = searcher.search("bad query")
        self.assertIn("DuckDuckGo search encountered an issue", res)

    def test_query_cleaning_and_news_detection(self) -> None:
        from services.tools.src.search import clean_conversational_query, is_news_query
        self.assertTrue(is_news_query("latest news in New Delhi"))
        self.assertTrue(is_news_query("breaking headlines"))
        self.assertFalse(is_news_query("who is the president"))

        cleaned = clean_conversational_query("Who is the Prime Minister of United Kingdom right now?")
        self.assertEqual(cleaned, "Prime Minister of United Kingdom")

        cleaned_tool = clean_conversational_query("Tool call, check the latest news in New Delhi")
        self.assertEqual(cleaned_tool, "latest news in New Delhi")

        cleaned_web = clean_conversational_query("Check the web for latest news in New Delhi")
        self.assertEqual(cleaned_web, "latest news in New Delhi")

        cleaned_search = clean_conversational_query("Search the web for Python 3.13")
        self.assertEqual(cleaned_search, "Python 3.13")

        cleaned_find = clean_conversational_query("Find recent news about space")
        self.assertEqual(cleaned_find, "news about space")

    @patch("services.tools.src.search.DDGS")
    def test_news_query_routes_to_ddgs_news(self, mock_ddgs_class: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.news.return_value = [
            {"title": "Delhi News", "body": "Delhi property tax reform update."}
        ]
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs_class.return_value = mock_instance

        searcher = DuckDuckGoSearch()
        res = searcher.search("latest news in New Delhi")
        self.assertIn("Delhi News: Delhi property tax reform update.", res)
        mock_instance.news.assert_called()

    @patch("services.tools.src.search.DDGS")
    def test_retry_on_initial_failure_with_cleaned_query(self, mock_ddgs_class: MagicMock) -> None:
        mock_instance = MagicMock()
        # First call fails with 'No results found.', second call with cleaned query succeeds
        def side_effect(query, **kwargs):
            if "right now" in query:
                raise Exception("No results found.")
            return [{"title": "UK PM", "body": "The Prime Minister of the United Kingdom."}]

        mock_instance.text.side_effect = side_effect
        mock_instance.__enter__.return_value = mock_instance
        mock_ddgs_class.return_value = mock_instance

        searcher = DuckDuckGoSearch()
        res = searcher.search("Prime Minister of United Kingdom right now")
        self.assertIn("UK PM: The Prime Minister of the United Kingdom.", res)


class TestToolManager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp_dir.name) / "mcp_servers.json"
        self.config_path.write_text(json.dumps({"mcpServers": {}}))

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_schema_includes_duckduckgo_when_enabled(self) -> None:
        manager = ToolManager(config_path=self.config_path, enable_search=True, enable_tools=True)
        tools = manager.get_ollama_tools()

        names = [t["function"]["name"] for t in tools]
        self.assertIn("duckduckgo_search", names)

    def test_schema_empty_when_tools_disabled(self) -> None:
        manager = ToolManager(config_path=self.config_path, enable_tools=False)
        tools = manager.get_ollama_tools()
        self.assertEqual(tools, [])

    def test_execute_duckduckgo_search(self) -> None:
        manager = ToolManager(config_path=self.config_path)
        with patch.object(manager.search, "search", return_value="Mocked search answer") as mock_search:
            res = manager.execute_tool("duckduckgo_search", {"query": "weather in Tokyo"})
            mock_search.assert_called_once_with("weather in Tokyo")
            self.assertEqual(res, "Mocked search answer")

    def test_fallback_to_duckduckgo_when_tool_not_installed(self) -> None:
        """When user asks for a function that has no installed tool, automatically fall back to DDG search."""
        manager = ToolManager(config_path=self.config_path)
        with patch.object(manager.search, "search", return_value="DDG web results for weather") as mock_search:
            res = manager.execute_tool("get_weather", {"location": "London"})
            mock_search.assert_called_once()
            self.assertIn("No installed tool matching function 'get_weather'", res)
            self.assertIn("DDG web results for weather", res)

    def test_fallback_to_duckduckgo_when_installed_tool_fails(self) -> None:
        """When an installed tool encounters a runtime error, fall back to DuckDuckGo search."""
        manager = ToolManager(config_path=self.config_path)
        server_cfg = MCPServerConfig(name="broken_server", command="broken")
        tool_def = MCPToolDefinition(
            name="broken_tool",
            description="A broken tool",
            input_schema={"type": "object"},
            server_name="broken_server",
        )
        manager._tools["broken_tool"] = (server_cfg, tool_def)

        with patch.object(manager.mcp_client, "call_tool_sync", side_effect=Exception("Connection refused")):
            with patch.object(manager.search, "search", return_value="Fallback search results") as mock_search:
                res = manager.execute_tool("broken_tool", {"query": "fallback test"})
                mock_search.assert_called_once()
                self.assertIn("broken_tool' error: Connection refused", res)
                self.assertIn("Fallback DuckDuckGo Search", res)

    def test_fallback_to_duckduckgo_when_tool_returns_unhelpful_result(self) -> None:
        """When an installed tool returns unknown code or error string without exception, fall back to DuckDuckGo."""
        manager = ToolManager(config_path=self.config_path)
        server_cfg = MCPServerConfig(name="weather", command="weather_cmd")
        tool_def = MCPToolDefinition(
            name="batch_interpret_weather_codes",
            description="Interpret weather codes",
            input_schema={"type": "object"},
            server_name="weather",
        )
        manager._tools["batch_interpret_weather_codes"] = (server_cfg, tool_def)

        unhelpful_response = '{"results": [{"code": 76, "description": "Unknown weather code: 76"}]}'
        with patch.object(manager.mcp_client, "call_tool_sync", return_value=unhelpful_response):
            with patch.object(manager.search, "search", return_value="Current weather in New Delhi is 28C") as mock_search:
                res = manager.execute_tool("batch_interpret_weather_codes", {"weather_codes": "76"})
                mock_search.assert_called_once()
                self.assertEqual(res, "Current weather in New Delhi is 28C")

    def test_add_and_remove_server(self) -> None:
        manager = ToolManager(config_path=self.config_path)

        # Mock list_tools_sync on mcp_client
        mock_tools = [
            MCPToolDefinition(
                name="test_tool",
                description="Test description",
                input_schema={"type": "object"},
                server_name="test_server",
            )
        ]
        with patch.object(manager.mcp_client, "list_tools_sync", return_value=mock_tools):
            discovered = manager.add_server(
                name="test_server",
                command="uvx",
                args=["test-pkg"],
            )
            self.assertEqual(discovered, ["test_tool"])

        # Check that config file was updated
        data = json.loads(self.config_path.read_text())
        self.assertIn("test_server", data["mcpServers"])
        self.assertEqual(data["mcpServers"]["test_server"]["command"], "uvx")

        # Check remove
        self.assertTrue(manager.remove_server("test_server"))
        data_after = json.loads(self.config_path.read_text())
        self.assertNotIn("test_server", data_after["mcpServers"])

    def test_build_search_query_formats_intents_and_arguments(self) -> None:
        manager = ToolManager(config_path=self.config_path)

        # Explicit query key
        q1 = manager._build_search_query("duckduckgo_search", {"query": "weather in Tokyo"})
        self.assertEqual(q1, "weather in Tokyo")

        # Function name and location argument
        q2 = manager._build_search_query("get_current_weather", {"location": "New Delhi"})
        self.assertEqual(q2, "current weather New Delhi")

        # Stock ticker argument
        q3 = manager._build_search_query("get_stock_quote", {"ticker": "AAPL"})
        self.assertEqual(q3, "stock quote AAPL")

        # No arguments
        q4 = manager._build_search_query("get_time", {})
        self.assertEqual(q4, "time")


class DummyCallbacks(LLMCallbacks):
    def __init__(self) -> None:
        self.started = False
        self.updates: list[str] = []
        self.completed = False
        self.tools_started: list[tuple[str, dict]] = []
        self.tools_completed: list[tuple[str, str]] = []

    def on_generation_started(self) -> None:
        self.started = True

    def on_generation_updated(self, text: str) -> None:
        self.updates.append(text)

    def on_generation_completed(self, text: str) -> None:
        self.completed = True

    def on_tool_started(self, tool_name: str, arguments: dict) -> None:
        self.tools_started.append((tool_name, arguments))

    def on_tool_completed(self, tool_name: str, result: str) -> None:
        self.tools_completed.append((tool_name, result))


class TestLLMProviderToolCalling(unittest.TestCase):
    def test_detect_text_tool_calls(self) -> None:
        provider = LLMProvider(LLMProviderConfig(), tool_manager=MagicMock())

        # Test XML format
        xml_text = '<tool_call>{"name": "duckduckgo_search", "arguments": {"query": "current time"}}</tool_call>'
        calls = provider._detect_text_tool_calls(xml_text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "duckduckgo_search")
        self.assertEqual(calls[0]["function"]["arguments"], {"query": "current time"})

        # Test raw JSON format
        json_text = 'I will call: {"name": "get_stock", "arguments": {"ticker": "AAPL"}}'
        calls_json = provider._detect_text_tool_calls(json_text)
        self.assertEqual(len(calls_json), 1)
        self.assertEqual(calls_json[0]["function"]["name"], "get_stock")

    def test_should_fallback_to_ddg_search_detection(self) -> None:
        provider = LLMProvider(LLMProviderConfig(), tool_manager=MagicMock())

        messages = [Message(role="user", content="What is the stock price of Apple today?")]
        asst_refusal = "I do not have access to real-time information to check stock prices."
        self.assertTrue(provider._should_fallback_to_ddg_search(messages, asst_refusal))

        asst_normal = "Apples are a nutritious fruit grown across temperate regions."
        self.assertFalse(provider._should_fallback_to_ddg_search(messages, asst_normal))

    def test_has_tool_call_keyword(self) -> None:
        # Explicit tool directives
        self.assertTrue(has_tool_call_keyword("Tool call, what's the weather in London?"))
        self.assertTrue(has_tool_call_keyword("Can you do a tool-call to check time?"))
        self.assertTrue(has_tool_call_keyword("tool_call: check stock"))
        self.assertTrue(has_tool_call_keyword("use the tool to get time"))

        # Search phrases
        self.assertTrue(has_tool_call_keyword("search for latest news in New Delhi"))
        self.assertTrue(has_tool_call_keyword("search the web for python 3.13"))
        self.assertTrue(has_tool_call_keyword("can you search duckduckgo for restaurants"))
        self.assertTrue(has_tool_call_keyword("look up who won the award"))

        # Find recent phrases
        self.assertTrue(has_tool_call_keyword("find recent articles on artificial intelligence"))
        self.assertTrue(has_tool_call_keyword("find latest news about Mars"))

        # Check the web / real-time facts
        self.assertTrue(has_tool_call_keyword("check the web for open cafes"))
        self.assertTrue(has_tool_call_keyword("check online for ticket prices"))
        self.assertTrue(has_tool_call_keyword("check recent headlines"))
        self.assertTrue(has_tool_call_keyword("what is the weather in Seattle?"))
        self.assertTrue(has_tool_call_keyword("what time is it right now?"))
        self.assertTrue(has_tool_call_keyword("tell me today's date"))
        self.assertTrue(has_tool_call_keyword("Who is the Prime Minister of United Kingdom right now?"))

        # Installed tool names
        self.assertTrue(has_tool_call_keyword("can you run a sqlite query?", tool_names={"sqlite_query"}))
        self.assertTrue(has_tool_call_keyword("get current time please", tool_names={"get_current_time"}))

        # Standard conversational queries must NOT trigger tools (keeping context light)
        self.assertFalse(has_tool_call_keyword("Tell me a funny joke"))
        self.assertFalse(has_tool_call_keyword("How are you doing today?"))
        self.assertFalse(has_tool_call_keyword("Write a short poem about the ocean"))
        self.assertFalse(has_tool_call_keyword("Can you explain how transformers work?"))
        self.assertFalse(has_tool_call_keyword("What is the capital of France?"))
        self.assertFalse(has_tool_call_keyword("I need some tools for carpentry"))
        self.assertFalse(has_tool_call_keyword(""))
        self.assertFalse(has_tool_call_keyword(None))

    def test_generate_omits_tools_when_no_keyword(self) -> None:
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_ollama_tools.return_value = [{"type": "function", "function": {"name": "test_tool"}}]
        provider = LLMProvider(LLMProviderConfig(tools_enabled=True), tool_manager=mock_tool_manager)

        messages = [Message(role="user", content="What is the capital of France?")]
        callbacks = DummyCallbacks()

        with patch.object(provider, "_stream_or_collect_tools", return_value=([], "Paris.")) as mock_stream:
            provider.generate(messages, callbacks)
            mock_stream.assert_called_once()
            payload = mock_stream.call_args[0][0]
            # Tools omitted to keep context light
            self.assertNotIn("tools", payload)
            # Default lightweight prompt without tool calling policy
            system_msg = payload["messages"][0]["content"]
            self.assertNotIn("TOOL CALLING POLICY", system_msg)
            mock_tool_manager.get_ollama_tools.assert_not_called()

    def test_generate_includes_tools_when_keyword_present(self) -> None:
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_ollama_tools.return_value = [{"type": "function", "function": {"name": "test_tool"}}]
        provider = LLMProvider(LLMProviderConfig(tools_enabled=True), tool_manager=mock_tool_manager)

        messages = [Message(role="user", content="Tool call, what is the weather in Paris?")]
        callbacks = DummyCallbacks()

        with patch.object(provider, "_stream_or_collect_tools", return_value=([], "It's sunny.")) as mock_stream:
            provider.generate(messages, callbacks)
            mock_stream.assert_called_once()
            payload = mock_stream.call_args[0][0]
            # Tools included
            self.assertIn("tools", payload)
            self.assertEqual(len(payload["tools"]), 1)
            # Tool prompt with tool calling policy included
            system_msg = payload["messages"][0]["content"]
            self.assertIn("TOOL CALLING POLICY", system_msg)
            mock_tool_manager.get_ollama_tools.assert_called_once()

    def test_generate_fallback_when_model_refuses_without_keyword(self) -> None:
        """When the model replies with a refusal regarding real-time info, it falls back to search."""
        mock_tool_manager = MagicMock()
        provider = LLMProvider(LLMProviderConfig(tools_enabled=True), tool_manager=mock_tool_manager)

        messages = [Message(role="user", content="Who won the game?")]
        callbacks = DummyCallbacks()

        with patch.object(
            provider,
            "_stream_or_collect_tools",
            return_value=([], "I do not have access to real-time news."),
        ), patch.object(provider, "_execute_tools_and_stream_reply") as mock_exec:
            provider.generate(messages, callbacks)
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[1]
            self.assertEqual(call_args["tool_calls"][0]["function"]["name"], "duckduckgo_search")
            self.assertEqual(call_args["tool_calls"][0]["function"]["arguments"], {"query": "Who won the game?"})

    def test_has_tool_intent_conversational_phrases(self) -> None:
        """Verify conversational lookups, follow-up checks, and references trigger tool intent."""
        self.assertTrue(has_tool_call_keyword("Yeah why don't we check for that"))
        self.assertTrue(has_tool_call_keyword("Can you look it up?"))
        self.assertTrue(has_tool_call_keyword("Can you search it right now"))
        self.assertTrue(has_tool_call_keyword("but I asked you to do previously"))
        self.assertTrue(has_tool_call_keyword("look that up for me"))
        self.assertTrue(has_tool_call_keyword("check it out please"))

    def test_should_fallback_to_ddg_search_stalling_phrases(self) -> None:
        """Verify stalling phrases where assistant claims it is searching trigger fallback."""
        provider = LLMProvider(LLMProviderConfig(tools_enabled=True), tool_manager=MagicMock())
        messages = [Message(role="user", content="Can you look it up?")]

        stalling_1 = "I am searching for some recent local news or events scheduled very soon for Wednesday, October 21st. Please give me a moment to look this up for you."
        self.assertTrue(provider._should_fallback_to_ddg_search(messages, stalling_1))

        stalling_2 = "Yes, I am looking up positive news and local events happening very soon in New Delhi right now. I will be with you on that in a moment."
        self.assertTrue(provider._should_fallback_to_ddg_search(messages, stalling_2))

        stalling_3 = "I am currently searching for some good things for you right here in New Delhi. Just give me a little more time to find those details."
        self.assertTrue(provider._should_fallback_to_ddg_search(messages, stalling_3))

    def test_extract_fallback_query_from_stalling_assistant(self) -> None:
        """Verify fallback query derives clean standalone search query from stalling text."""
        messages = [
            Message(role="user", content="Yeah, can you find some positive news for me?"),
            Message(role="assistant", content="I found some good things!"),
            Message(role="user", content="Can you look it up?"),
        ]
        asst = "I am searching for some recent local news or events scheduled very soon for Wednesday, October 21st. Please give me a moment to look this up for you."
        query = LLMProvider._extract_fallback_query(messages, asst)
        self.assertIn("news", query.lower())
        self.assertNotIn("please give me a moment", query.lower())




class TestPhraseManager(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = PhraseManager()

    def test_match_category(self) -> None:
        self.assertEqual(self.manager.match_category("duckduckgo_search"), "search")
        self.assertEqual(self.manager.match_category("web_search"), "search")
        self.assertEqual(self.manager.match_category("sqlite_query"), "database")
        self.assertEqual(self.manager.match_category("db_read"), "database")
        self.assertEqual(self.manager.match_category("read_file"), "filesystem")
        self.assertEqual(self.manager.match_category("list_directory"), "filesystem")
        self.assertEqual(self.manager.match_category("get_time"), "time")
        self.assertEqual(self.manager.match_category("fetch_url"), "fetch")
        self.assertEqual(self.manager.match_category("get_current_weather"), "weather")
        self.assertEqual(self.manager.match_category("duckduckgo_search", {"query": "weather forecast"}), "weather")
        self.assertEqual(self.manager.match_category("random_tool"), "general")

    def test_select_phrase_no_repeats(self) -> None:
        """Sequential selections in a category must avoid repeating until exhausted."""
        from services.tools.src.phrases import FILLER_PHRASES

        search_phrases = FILLER_PHRASES["search"]
        selected_ids = []

        for _ in range(len(search_phrases)):
            phrase = self.manager.select_phrase("duckduckgo_search")
            self.assertNotIn(phrase["id"], selected_ids)
            selected_ids.append(phrase["id"])

        self.assertEqual(len(selected_ids), len(search_phrases))

    def test_select_phrase_resets_after_exhaustion(self) -> None:
        """When all phrases in a category have been used, it resets and allows new selections."""
        from services.tools.src.phrases import FILLER_PHRASES

        total = len(FILLER_PHRASES["search"])
        for _ in range(total):
            self.manager.select_phrase("duckduckgo_search")

        # The next selection should successfully pick a phrase from the reset pool
        next_phrase = self.manager.select_phrase("duckduckgo_search")
        self.assertIsNotNone(next_phrase)
        self.assertEqual(next_phrase["category"], "search")

    def test_play_filler_invokes_speak_callback(self) -> None:
        spoken = []

        def mock_speak(text: str) -> None:
            spoken.append(text)

        manager = PhraseManager(speak_callback=mock_speak)
        chosen = manager.play_filler("duckduckgo_search")

        self.assertIsNotNone(chosen)
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0], chosen["text"])

    def test_play_filler_invokes_tts_speak(self) -> None:
        class MockTTS:
            def __init__(self) -> None:
                self.spoken = []

            def speak(self, text: str) -> None:
                self.spoken.append(text)

        mock_tts = MockTTS()
        manager = PhraseManager(tts=mock_tts)
        chosen = manager.play_filler("sqlite_query")

        self.assertIsNotNone(chosen)
        self.assertEqual(len(mock_tts.spoken), 1)
        self.assertEqual(mock_tts.spoken[0], chosen["text"])

    def test_provider_set_tts(self) -> None:
        from services.llm.src.llm_provider import LLMProvider, LLMProviderConfig

        class MockTTS:
            def __init__(self) -> None:
                self.spoken = []

            def speak(self, text: str) -> None:
                self.spoken.append(text)

        provider = LLMProvider(LLMProviderConfig())
        mock_tts = MockTTS()
        provider.set_tts(mock_tts)
        self.assertEqual(provider.phrase_manager._tts, mock_tts)


if __name__ == "__main__":
    unittest.main()

