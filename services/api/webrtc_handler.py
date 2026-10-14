"""WebRTC signaling and media handling for browser voice chat.

Flow: browser mic → audio preprocessing/VAD → STTService
      STTService → ConversationManager → LLMService → TTSService
      TTSService → TTSOutputTrack → RTCPeerConnection → browser speakers
"""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import os
import time
from typing import Optional

import aioice.ice
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
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from aiortc.sdp import candidate_to_sdp

from services.audio_processing.src import VADService
from services.stt.src.stt_service import AudioQueueFull, STTService
from services.tts.src.tts_service import AudioBuffer, TTSListener

logger = logging.getLogger(__name__)

# Monkey-patch aioice Connection._prune_components to prevent it from emptying
# self._components to set() when remote offers arrive with a=end-of-candidates
# but 0 remote candidates (which occurs in offline / local environments without STUN).
# Pruning components to set() prevents aioice from creating ANY local sockets/candidates.
_original_aioice_prune_components = aioice.ice.Connection._prune_components


def _safe_aioice_prune_components(self: aioice.ice.Connection) -> None:
    seen_components = set(map(lambda x: x.component, self._remote_candidates))
    if seen_components:
        missing_components = self._components - seen_components
        if missing_components:
            self._components = seen_components


aioice.ice.Connection._prune_components = _safe_aioice_prune_components

_WEBRTC_RATE = 48_000
_FRAME_SAMPLES = 960
_STT_RATE = 16_000
_ICE_GATHERING_TIMEOUT_SECONDS = 10.0
_DEFAULT_STUN_SERVERS = ""
_DEFAULT_ICE_PORT_MIN = 50000
_DEFAULT_ICE_PORT_MAX = 50100

_peer_connections: set[RTCPeerConnection] = set()


def _configure_aioice_port_range() -> None:
    """Configure aioice to bind its dynamic UDP sockets to the published port range."""
    minimum = int(os.getenv("VOICECHAT_WEBRTC_UDP_PORT_MIN", _DEFAULT_ICE_PORT_MIN))
    maximum = int(os.getenv("VOICECHAT_WEBRTC_UDP_PORT_MAX", _DEFAULT_ICE_PORT_MAX))
    if not (1 <= minimum <= maximum <= 65535):
        raise ValueError("VOICECHAT_WEBRTC_UDP_PORT_MIN/MAX must describe a valid UDP port range.")

    loop = asyncio.get_running_loop()
    if getattr(loop, "_waffle_original_create_datagram_endpoint", None) is not None:
        return

    original = loop.create_datagram_endpoint
    setattr(loop, "_waffle_original_create_datagram_endpoint", original)

    async def create_datagram_endpoint(protocol_factory, *args, **kwargs):
        local_addr = kwargs.get("local_addr")
        positional_local = local_addr is None and bool(args)
        if positional_local:
            local_addr = args[0]

        logger.debug("create_datagram_endpoint called: local_addr=%s, positional=%s", local_addr, positional_local)

        if local_addr and len(local_addr) >= 2 and local_addr[1] == 0:
            last_error = None
            host = local_addr[0]
            width = maximum - minimum + 1
            start = int.from_bytes(os.urandom(4), "big") % width
            for offset in range(width):
                port = minimum + ((start + offset) % width)
                address = (host, port, *local_addr[2:])
                args_with_port = list(args)
                kwargs_with_port = dict(kwargs)
                if positional_local:
                    args_with_port[0] = address
                else:
                    kwargs_with_port["local_addr"] = address
                try:
                    res = await original(protocol_factory, *args_with_port, **kwargs_with_port)
                    logger.debug("Bound UDP socket to %s", address)
                    return res
                except OSError as exc:
                    logger.debug("Failed to bind UDP %s: %s", address, exc)
                    last_error = exc
            logger.error("No free UDP ports in range %d-%d, last error: %s", minimum, maximum, last_error)
            raise OSError(
                f"No free UDP ports in configured WebRTC range {minimum}-{maximum}."
            ) from last_error
        return await original(protocol_factory, *args, **kwargs)

    loop.create_datagram_endpoint = create_datagram_endpoint


def _rewrite_advertised_candidates(sdp_text: str, client_host: str = "") -> str:
    """Rewrite container-internal ICE candidates to an address reachable by the browser."""
    advertised_ip = os.getenv("VOICECHAT_WEBRTC_ADVERTISED_IP", "").strip()
    if not advertised_ip:
        if client_host and client_host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            advertised_ip = client_host
        else:
            advertised_ip = "127.0.0.1"

    rewritten = []
    for line in sdp_text.splitlines():
        if line.startswith("a=candidate:"):
            fields = line.split()
            if len(fields) >= 8 and fields[7] == "host":
                fields[4] = advertised_ip
                line = " ".join(fields)
        rewritten.append(line)
    return "\r\n".join(rewritten) + "\r\n"


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
        super().__init__()
        self._loop = loop
        self._buffer = np.array([], dtype=np.float32)
        self._start: float | None = None
        self._timestamp = 0

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
        if self.readyState != "live":
            raise MediaStreamError

        sample_rate = _WEBRTC_RATE
        samples = _FRAME_SAMPLES

        if self._start is not None:
            self._timestamp += samples
            wait = self._start + (self._timestamp / sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        if len(self._buffer) >= _FRAME_SAMPLES:
            chunk = self._buffer[:_FRAME_SAMPLES]
            self._buffer = self._buffer[_FRAME_SAMPLES:]
        else:
            chunk = np.zeros(_FRAME_SAMPLES, dtype=np.float32)

        frame = av.AudioFrame.from_ndarray(
            chunk.reshape(1, -1),
            format="fltp",
            layout="mono",
        )
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, sample_rate)
        frame.sample_rate = sample_rate
        return frame


async def _receive_browser_audio(
    track: MediaStreamTrack,
    stt: STTService,
    vad: VADService,
) -> None:
    """Continuously read browser audio, gate it with VAD, then push to STT."""
    logger.info("Started receiving browser audio track: id=%s", track.id)
    frame_count = 0
    try:
        while True:
            try:
                frame = await track.recv()
            except Exception as exc:
                logger.info("Browser audio track ended or raised: %s", exc)
                break

            frame_count += 1
            if frame_count % 100 == 1:
                logger.info(
                    "Receiving audio frames (frame #%d, rate=%s, channels=%d)",
                    frame_count,
                    frame.sample_rate,
                    len(frame.layout.channels),
                )

            samples = _frame_to_float32_mono(frame)
            samples_16k = _resample(samples, frame.sample_rate, _STT_RATE)

            try:
                speech_audio = vad.process(samples_16k, _STT_RATE)
            except Exception:
                logger.exception("VAD processing failed.")
                continue

            if speech_audio.size == 0:
                continue

            logger.info("VAD detected speech (%d samples @ 16kHz), pushing to STT", speech_audio.size)
            try:
                stt.push(speech_audio, _STT_RATE)
            except AudioQueueFull:
                logger.warning("STT audio queue is full; dropping WebRTC speech frame.")
    finally:
        logger.info("Stopped receiving browser audio track (total frames: %d)", frame_count)
        vad.reset()


async def create_answer(
    sdp: str,
    sdp_type: str,
    tts,
    stt: STTService,
    vad: VADService,
    client_host: str = "",
) -> dict[str, str]:
    """Accept a browser offer and return an SDP answer with gathered ICE candidates."""
    loop = asyncio.get_running_loop()
    _configure_aioice_port_range()
    pc = RTCPeerConnection(RTCConfiguration(iceServers=_ice_servers()))
    _peer_connections.add(pc)

    tts_track = TTSOutputTrack(loop=loop)
    tts.add_listener(tts_track)
    pc.addTrack(tts_track)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        logger.info("WebRTC received track: kind=%s, id=%s", track.kind, track.id)
        if track.kind == "audio":
            asyncio.create_task(_receive_browser_audio(track, stt, vad))

    @pc.on("connectionstatechange")
    async def on_state_change() -> None:
        logger.info("WebRTC peer state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            tts.remove_listener(tts_track)
            _peer_connections.discard(pc)

    try:
        # If the remote offer contains a=end-of-candidates but no candidates (common in offline/local
        # browser environments), strip a=end-of-candidates before aiortc processes it so that
        # ICE components are not pruned and remote candidate completion is not signaled prematurely.
        if "a=candidate:" not in sdp:
            sdp = "\r\n".join(
                line for line in sdp.splitlines() if line.strip() != "a=end-of-candidates"
            ) + "\r\n"

        logger.debug("Received offer SDP (len=%d):\n%s", len(sdp), sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering(pc)

        local = pc.localDescription
        if local is None:
            raise RuntimeError("Failed to generate local WebRTC session description.")

        answer_sdp = local.sdp
        logger.debug("Local SDP after gathering (state=%s):\n%s", pc.iceGatheringState, answer_sdp)

        if "a=candidate:" not in answer_sdp:
            candidate_lines = []
            for t in pc.getTransceivers():
                dtls = getattr(t.receiver, "transport", None) or getattr(t.sender, "transport", None)
                ice = getattr(dtls, "transport", None) or getattr(dtls, "_transport", None)
                gatherer = getattr(ice, "iceGatherer", None)
                if gatherer:
                    for c in gatherer.getLocalCandidates():
                        candidate_lines.append(f"a=candidate:{candidate_to_sdp(c)}")
            if candidate_lines:
                answer_sdp = answer_sdp.strip() + "\r\n" + "\r\n".join(candidate_lines) + "\r\n"

        if "a=candidate:" not in answer_sdp:
            raise RuntimeError(
                "Server gathered no ICE candidates. Check network/firewall and "
                "VOICECHAT_ICE_SERVERS configuration."
            )

        logger.info(
            "Returning WebRTC answer (ICE state=%s, candidate lines=%d)",
            pc.iceGatheringState,
            sum(line.startswith("a=candidate:") for line in answer_sdp.splitlines()),
        )
        return {"sdp": _rewrite_advertised_candidates(answer_sdp, client_host=client_host), "type": local.type}
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
