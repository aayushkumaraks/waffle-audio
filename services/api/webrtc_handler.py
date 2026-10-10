"""WebRTC signaling and media handling for browser voice chat.

Flow: browser mic → audio preprocessing/VAD → STTService
      STTService → ConversationManager → LLMService → TTSService
      TTSService → TTSOutputTrack → RTCPeerConnection → browser speakers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

import av
import numpy as np
import numpy.typing as npt
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import AudioStreamTrack

from services.audio_processing.src import VADService
from services.stt.src.stt_service import AudioQueueFull, STTService
from services.tts.src.tts_service import AudioBuffer, TTSListener

logger = logging.getLogger(__name__)

_WEBRTC_RATE = 48_000
_FRAME_SAMPLES = 960
_STT_RATE = 16_000
_ICE_GATHERING_TIMEOUT_SECONDS = 10.0
_DEFAULT_STUN_SERVERS = "stun:stun.l.google.com:19302"

_peer_connections: set[RTCPeerConnection] = set()


def _ice_servers() -> list[RTCIceServer]:
    """Load ICE servers from JSON or a comma-separated URL list.

    JSON accepts standard RTCIceServer fields (urls, username, credential),
    allowing authenticated TURN relays. Keep credentials in the environment,
    never in source control. URL-list mode remains convenient for STUN-only use.
    """
    configured = os.getenv("VOICECHAT_ICE_SERVERS", _DEFAULT_STUN_SERVERS).strip()
    if not configured:
        return []

    try:
        parsed = json.loads(configured)
    except json.JSONDecodeError:
        urls = [value.strip() for value in configured.split(",") if value.strip()]
        return [RTCIceServer(urls=url) for url in urls]

    if not isinstance(parsed, list):
        raise ValueError("VOICECHAT_ICE_SERVERS JSON must be an array of ICE server objects.")

    servers: list[RTCIceServer] = []
    for item in parsed:
        if not isinstance(item, dict) or "urls" not in item:
            raise ValueError("Each VOICECHAT_ICE_SERVERS entry must be an object with 'urls'.")
        servers.append(
            RTCIceServer(
                urls=item["urls"],
                username=item.get("username"),
                credential=item.get("credential"),
            )
        )
    return servers


async def _wait_for_ice_gathering(
    pc: RTCPeerConnection,
    timeout_seconds: float = _ICE_GATHERING_TIMEOUT_SECONDS,
) -> None:
    """Wait for candidates to be embedded in the local SDP (non-trickle ICE)."""
    if pc.iceGatheringState == "complete":
        return

    completed = asyncio.Event()

    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            completed.set()

    pc.on("icegatheringstatechange", on_ice_gathering_state_change)
    try:
        if pc.iceGatheringState == "complete":
            return
        await asyncio.wait_for(completed.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Server ICE gathering did not complete within {timeout_seconds:.0f} seconds."
        ) from exc
    finally:
        try:
            pc.remove_listener("icegatheringstatechange", on_ice_gathering_state_change)
        except Exception:
            logger.debug("Could not remove ICE gathering listener", exc_info=True)


def _resample(
    audio: npt.NDArray[np.float32],
    src: int,
    dst: int,
) -> npt.NDArray[np.float32]:
    if src == dst:
        return audio
    n_out = max(1, round(len(audio) * dst / src))
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _frame_to_float32_mono(frame: av.AudioFrame) -> npt.NDArray[np.float32]:
    """Convert any av.AudioFrame to a float32 mono numpy array."""
    arr = frame.to_ndarray().astype(np.float32)
    if "s16" in frame.format.name:
        arr /= 32_768.0
    elif "s32" in frame.format.name:
        arr /= 2_147_483_648.0
    arr = arr.flatten()
    n_ch = len(frame.layout.channels)
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1)
    return arr


class TTSOutputTrack(AudioStreamTrack, TTSListener):
    """Streams selected TTS-backend audio to the browser via WebRTC."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        AudioStreamTrack.__init__(self)
        self._loop = loop
        self._buffer = np.array([], dtype=np.float32)

    def on_synthesis_started(self) -> None:
        pass

    def on_synthesis_completed(self) -> None:
        pass

    def on_audio_chunk(self, audio: AudioBuffer, sample_rate: int) -> None:
        self._loop.call_soon_threadsafe(self._ingest, audio.copy(), sample_rate)

    def _ingest(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> None:
        resampled = _resample(audio, sample_rate, _WEBRTC_RATE)
        self._buffer = np.concatenate([self._buffer, resampled])

    async def recv(self) -> av.AudioFrame:
        pts, time_base = await self.next_timestamp()

        if len(self._buffer) >= _FRAME_SAMPLES:
            samples = self._buffer[:_FRAME_SAMPLES]
            self._buffer = self._buffer[_FRAME_SAMPLES:]
        else:
            samples = np.zeros(_FRAME_SAMPLES, dtype=np.float32)

        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format="fltp",
            layout="mono",
        )
        frame.pts = pts
        frame.time_base = time_base
        frame.sample_rate = _WEBRTC_RATE
        return frame


async def _receive_browser_audio(
    track: MediaStreamTrack,
    stt: STTService,
    vad: VADService,
) -> None:
    """Continuously read browser audio, gate it with VAD, then push to STT."""
    try:
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break

            samples = _frame_to_float32_mono(frame)
            samples_16k = _resample(samples, frame.sample_rate, _STT_RATE)

            try:
                speech_audio = vad.process(samples_16k, _STT_RATE)
            except Exception:
                logger.exception("VAD processing failed.")
                continue

            if speech_audio.size == 0:
                continue

            try:
                stt.push(speech_audio, _STT_RATE)
            except AudioQueueFull:
                pass
    finally:
        vad.reset()


async def create_answer(
    sdp: str,
    sdp_type: str,
    tts,
    stt: STTService,
    vad: VADService,
) -> dict[str, str]:
    """Accept a browser offer and return an SDP answer with gathered ICE candidates."""
    loop = asyncio.get_running_loop()
    pc = RTCPeerConnection(RTCConfiguration(iceServers=_ice_servers()))
    _peer_connections.add(pc)

    tts_track = TTSOutputTrack(loop=loop)
    tts.add_listener(tts_track)
    pc.addTrack(tts_track)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            asyncio.create_task(_receive_browser_audio(track, stt, vad))

    @pc.on("connectionstatechange")
    async def on_state_change() -> None:
        logger.info("WebRTC peer state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            tts.remove_listener(tts_track)
            _peer_connections.discard(pc)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering(pc)

        local = pc.localDescription
        if local is None or "a=candidate:" not in local.sdp:
            raise RuntimeError(
                "Server gathered no ICE candidates. Check network/firewall and "
                "VOICECHAT_ICE_SERVERS configuration."
            )

        logger.info(
            "Returning WebRTC answer (ICE state=%s, candidate lines=%d)",
            pc.iceGatheringState,
            sum(line.startswith("a=candidate:") for line in local.sdp.splitlines()),
        )
        return {"sdp": local.sdp, "type": local.type}
    except Exception:
        tts.remove_listener(tts_track)
        _peer_connections.discard(pc)
        await pc.close()
        logger.exception("Failed to negotiate WebRTC offer/answer.")
        raise


async def close_all() -> None:
    """Close every active peer connection."""
    for pc in list(_peer_connections):
        await pc.close()
    _peer_connections.clear()
