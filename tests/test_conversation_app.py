"""Unit tests for services/conversation/app.py entrypoint."""
from __future__ import annotations

import unittest

from services.conversation.cli import main as cli_main
import services.conversation.app as conversation_app


class TestConversationApp(unittest.TestCase):
    def test_conversation_app_references_cli_main(self) -> None:
        """Verify services.conversation.app exposes cli.main without executing the blocking CLI loop."""
        self.assertTrue(callable(conversation_app.main))
        self.assertIs(conversation_app.main, cli_main)


if __name__ == "__main__":
    unittest.main()
