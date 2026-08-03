from __future__ import annotations

import logging
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from constants import (
    API_CORS_ALLOW_ORIGINS,
    API_HOST,
    API_PORT,
    TTS_MODEL_PATH,
    TTS_VOICES_PATH,
)
from models import Message
from services.audio.src import AudioPlayer
from services.conversation.src import ConversationManager
from services.llm.src import LLMListener, LLMProvider, LLMProviderConfig, LLMService
from services.stt.src import AudioQueueFull, STTConfig, STTListener, STTService
from services.tts.src import SpeechQueueFull, TTSConfig, TTSService

logger = logging.getLogger(__name__)


def _parse_cors_origins(value: str) -> list[str]:
    origins = [item.strip() for item in value.split(",") if item.strip()]
    return origins or ["*"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class LLMGenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    timeout_seconds: float = Field(default=90.0, ge=1.0, le=300.0)


class LLMGenerateResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)


class STTPushRequest(BaseModel):
    sample_rate: int = Field(gt=0)
    samples: list[float] = Field(min_length=1)


class ConversationRequest(BaseModel):
    text: str = Field(min_length=1)


class TranscriptEvent(BaseModel):
    kind: Literal["started", "updated", "completed"]
    text: str
    created_at: str


class AppStatus(BaseModel):
    running: bool
    api_host: str
    api_port: int
    audio_playing: bool


class _OneShotGenerationListener(LLMListener):
    def __init__(self) -> None:
        self._done = threading.Event()
        self.result: str = ""

    def on_generation_started(self) -> None:
        pass

    def on_generation_updated(self, text: str) -> None:
        self.result = text

    def on_generation_completed(self, text: str) -> None:
        self.result = text
        self._done.set()

    def wait(self, timeout_seconds: float) -> bool:
        return self._done.wait(timeout_seconds)


class _TranscriptCollector(STTListener):
    def __init__(self) -> None:
        self._events: list[TranscriptEvent] = []
        self._lock = threading.Lock()

    def on_transcript_started(self, text: str) -> None:
        self._add_event("started", text)

    def on_transcript_updated(self, text: str) -> None:
        self._add_event("updated", text)

    def on_transcript_completed(self, text: str) -> None:
        self._add_event("completed", text)

    def snapshot(self) -> list[TranscriptEvent]:
        with self._lock:
            return self._events.copy()

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def _add_event(self, kind: Literal["started", "updated", "completed"], text: str) -> None:
        event = TranscriptEvent(
            kind=kind,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._events.append(event)


@dataclass(slots=True)
class ServiceRuntime:
    stt: STTService
    llm: LLMService
    tts: TTSService
    player: AudioPlayer
    conversation: ConversationManager
    transcript_collector: _TranscriptCollector
    running: bool = False

    @classmethod
    def create(cls) -> "ServiceRuntime":
        stt = STTService(STTConfig())

        provider = LLMProvider(LLMProviderConfig())
        llm = LLMService(provider)

        tts = TTSService(
            TTSConfig(
                model_path=TTS_MODEL_PATH,
                voices_path=TTS_VOICES_PATH,
            )
        )

        player = AudioPlayer()
        tts.add_listener(player)

        conversation = ConversationManager(stt=stt, llm=llm, tts=tts)

        transcript_collector = _TranscriptCollector()
        stt.add_listener(transcript_collector)

        return cls(
            stt=stt,
            llm=llm,
            tts=tts,
            player=player,
            conversation=conversation,
            transcript_collector=transcript_collector,
        )

    def start(self) -> None:
        if self.running:
            return

        self.player.start()
        self.tts.start()
        self.llm.start()
        self.conversation.start()
        self.stt.start()

        self.running = True
        logger.info("All services started.")

    def stop(self) -> None:
        if not self.running:
            return

        self.stt.stop()
        self.conversation.stop()
        self.llm.stop()
        self.tts.stop()
        self.player.stop()

        self.running = False
        logger.info("All services stopped.")


runtime = ServiceRuntime.create()
_llm_request_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()

    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(
    title="VoiceChat Services API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(API_CORS_ALLOW_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=AppStatus)
def health() -> AppStatus:
    return AppStatus(
        running=runtime.running,
        api_host=API_HOST,
        api_port=API_PORT,
        audio_playing=runtime.player.is_playing,
    )


@app.post("/llm/generate", response_model=LLMGenerateResponse)
def generate_llm(request: LLMGenerateRequest) -> LLMGenerateResponse:
    listener = _OneShotGenerationListener()

    with _llm_request_lock:
        runtime.llm.add_listener(listener)

        try:
            runtime.llm.generate(
                [Message(role=message.role, content=message.content) for message in request.messages]
            )
        except queue.Full as exc:
            runtime.llm.remove_listener(listener)
            raise HTTPException(status_code=429, detail="LLM queue is full.") from exc

        completed = listener.wait(request.timeout_seconds)
        runtime.llm.remove_listener(listener)

    if not completed:
        raise HTTPException(status_code=504, detail="Timed out waiting for LLM response.")

    return LLMGenerateResponse(text=listener.result)


@app.post("/tts/speak")
def speak_tts(request: TTSRequest) -> dict[str, str]:
    try:
        runtime.tts.speak(request.text)
    except SpeechQueueFull as exc:
        raise HTTPException(status_code=429, detail="TTS queue is full.") from exc

    return {"status": "queued"}


@app.post("/stt/push")
def push_stt_audio(request: STTPushRequest) -> dict[str, str]:
    audio = np.asarray(request.samples, dtype=np.float32)

    try:
        runtime.stt.push(audio, request.sample_rate)
    except AudioQueueFull as exc:
        raise HTTPException(status_code=429, detail="STT queue is full.") from exc

    return {"status": "queued"}


@app.get("/stt/transcripts", response_model=list[TranscriptEvent])
def list_stt_transcripts() -> list[TranscriptEvent]:
    return runtime.transcript_collector.snapshot()


@app.delete("/stt/transcripts")
def clear_stt_transcripts() -> dict[str, str]:
    runtime.transcript_collector.clear()
    return {"status": "cleared"}


@app.post("/conversation/respond")
def submit_conversation_text(request: ConversationRequest) -> dict[str, str]:
    runtime.conversation.submit_user_text(request.text)
    return {"status": "queued"}


@app.get("/conversation/history")
def conversation_history() -> list[ChatMessage]:
    return [ChatMessage(role=message.role, content=message.content) for message in runtime.conversation.history]


@app.delete("/conversation/history")
def clear_conversation_history() -> dict[str, str]:
    runtime.conversation.reset()
    return {"status": "cleared"}


@app.get("/audio/status")
def audio_status() -> dict[str, bool]:
    return {"is_playing": runtime.player.is_playing}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "services.api.app:app",
        host=API_HOST,
        port=API_PORT,
    )


if __name__ == "__main__":
    main()
