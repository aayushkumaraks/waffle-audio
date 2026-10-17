"""Unified In-Process Voice Pipeline.

Consolidates VAD, AEC, STT, ConversationManager, LLM, TTS, and AudioPlayer
into a single in-process async pipeline passing raw NumPy float32 audio buffers
directly in memory without network or JSON serialization overhead.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from services.audio.src import AudioPlaybackListener, AudioPlayer
from services.audio_processing.src import (
    AcousticEchoCanceller,
    AECConfig,
    BargeInController,
    EchoCanceller,
    PassthroughEchoCanceller,
    VADConfig,
    VADService,
    resample_audio,
)
from services.conversation.src import ConversationManager
from services.llm.src import LLMListener, LLMProvider, LLMProviderConfig, LLMService
from services.stt.src import AudioQueueFull, STTConfig, STTListener, STTService
from services.tts.src import TTSConfig, TTSService

logger = logging.getLogger(__name__)

AudioBuffer = npt.NDArray[np.float32]


class UnifiedPipeline:
    """Unified in-process voice-to-voice assistant pipeline."""

    def __init__(
        self,
        *,
        vad_config: VADConfig | None = None,
        aec_config: AECConfig | None = None,
        stt_config: STTConfig | None = None,
        llm_config: LLMProviderConfig | None = None,
        tts_config: TTSConfig | None = None,
        enable_aec: bool = True,
        on_user_turn: Callable[[str], None] | None = None,
    ) -> None:
        self.vad = VADService(vad_config or VADConfig())
        self.echo_canceller: EchoCanceller = (
            AcousticEchoCanceller(aec_config or AECConfig())
            if enable_aec
            else PassthroughEchoCanceller()
        )
        self.stt = STTService(stt_config)
        self.provider = LLMProvider(llm_config or LLMProviderConfig())
        self.llm = LLMService(self.provider)
        self.tts = TTSService(tts_config or TTSConfig())
        self.player = AudioPlayer()

        # Connect TTS to Player
        self.tts.add_listener(self.player)

        # Wire conversation manager
        self.manager = ConversationManager(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            on_user_turn=on_user_turn,
        )

        # Wire instant interruption / barge-in controller
        self.barge_in_controller = BargeInController(
            player=self.player,
            manager=self.manager,
            tts=self.tts,
            llm=self.llm,
        )
        self.vad.add_listener(self.barge_in_controller)

        # Feed audio player playback monitor to AEC reference input
        self.player.add_playback_monitor(self.echo_canceller.feed_reference)

        self._started = False

    def start(self) -> None:
        """Start all underlying services in-process."""
        if self._started:
            return

        self.vad.start()
        self.player.start()
        self.tts.start()
        self.llm.start()
        self.manager.start()
        self.stt.start()
        self._started = True
        logger.info("Unified voice pipeline started in-process.")

    def stop(self) -> None:
        """Stop all underlying pipeline services cleanly."""
        if not self._started:
            return

        self.vad.stop()
        self.stt.stop()
        self.manager.stop()
        self.llm.stop()
        self.tts.stop()
        self.player.stop()
        self._started = False
        logger.info("Unified voice pipeline stopped.")

    def feed_microphone_audio(
        self,
        audio: AudioBuffer,
        sample_rate: int = 16000,
    ) -> None:
        """Feed raw microphone audio directly into the in-process pipeline.

        Applies:
        1. Acoustic echo cancellation (subtracts active speaker playback)
        2. Voice activity detection (gates silence and preserves 150-200ms pre-roll)
        3. Moonshine streaming STT
        """
        if not self._started or audio.size == 0:
            return

        flat_audio = audio.astype(np.float32, copy=False).flatten()

        # 1. Echo Cancellation
        cleaned_audio = self.echo_canceller.process(flat_audio, sample_rate)

        # Resample to 16 kHz if necessary
        if sample_rate != 16000:
            cleaned_audio = resample_audio(cleaned_audio, sample_rate, 16000)
            sample_rate = 16000

        # 2. Voice Activity Detection
        speech_audio = self.vad.process(cleaned_audio, sample_rate)
        if speech_audio.size == 0:
            return

        # 3. Moonshine STT
        try:
            self.stt.push(speech_audio, sample_rate)
        except AudioQueueFull:
            pass

    def barge_in(self) -> None:
        """Manually trigger barge-in interruption."""
        self.barge_in_controller.trigger_barge_in()

    def add_transcript_listener(self, listener: STTListener) -> None:
        self.stt.add_listener(listener)

    def add_llm_listener(self, listener: LLMListener) -> None:
        self.llm.add_listener(listener)

    def add_playback_listener(self, listener: AudioPlaybackListener) -> None:
        self.player.add_playback_listener(listener)
