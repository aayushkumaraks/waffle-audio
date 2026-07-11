import logging
import time

import numpy as np
import sounddevice as sd

from services.conversation.src import ConversationManager
from services.stt.src import STTService, AudioQueueFull
from services.llm.src import LLMProvider, LLMProviderConfig, LLMService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


stt = STTService()
provider = LLMProvider(
    LLMProviderConfig(),
)

llm = LLMService(provider)

manager = ConversationManager(stt, llm)

llm.start()
manager.start()
stt.start()


def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(status)

    audio = indata.astype(np.float32).flatten()
    try:
        stt.push(audio, 16000)
    except AudioQueueFull:
        pass


stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="float32",
    blocksize=512,
    callback=audio_callback,
)

stream.start()

logger.info("Listening... Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    logger.info("Stopping...")

finally:
    stream.stop()
    stream.close()

    stt.stop()
    manager.stop()
    llm.stop()

    print("\nConversation History")
    print("--------------------")

    for index, message in enumerate(manager.history, start=1):
        print(f"{index}. [{message.role}] {message.content}")