from models import Message

from .src.llm_provider import LLMProvider, LLMProviderConfig
from .src.llm_service import LLMListener, LLMService


class ConsoleListener(LLMListener):
    def on_generation_started(self) -> None:
        print("Started")

    def on_generation_updated(self, text: str) -> None:
        print("Updated:", text)

    def on_generation_completed(self, text: str) -> None:
        print("Completed:", text)


def main() -> None:
    provider = LLMProvider(LLMProviderConfig())

    service = LLMService(provider)

    service.start()

    service.add_listener(ConsoleListener())

    service.generate(
        [
            Message(
                role="user",
                content="My name is Aayush.",
            ),
            Message(
                role="assistant",
                content="Nice to meet you!",
            ),
            Message(
                role="user",
                content="What's my name?",
            ),
        ]
    )

    service.stop()


if __name__ == "__main__":
    main()