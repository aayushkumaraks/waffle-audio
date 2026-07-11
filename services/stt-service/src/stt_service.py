from __future__ import annotations

from typing import Optional
import queue
import threading

from moonshine_voice import Transcriber, Stream
from moonshine_voice.download import get_model_for_language


class STTService:
  """Core Speech-to-Text service."""

  def __init__(self, language: str = "en") -> None:
    self._audio_queue = queue.Queue()
    self._worker_thread: Optional[threading.Thread] = None
    self._running = False
    self._language = language
    self._transcriber: Optional[Transcriber] = None
    self._stream: Optional[Stream] = None

  def start(self) -> None:
    if self._stream is not None:
      raise RuntimeError("STTService is already started.")

    model_path, model_arch = get_model_for_language(self._language)

    self._transcriber = Transcriber(
      model_path=model_path,
      model_arch=model_arch,
    )

    self._stream = self._transcriber.create_stream()
    self._stream.start()
    self._running = True

    self._worker_thread = threading.Thread(
      target=self._process_audio_queue,
      daemon=True,
    )

    self._worker_thread.start()

  def add_listener(self, listener) -> None:
    if self._stream is None:
      raise RuntimeError("STTService has not been started.")

    self._stream.add_listener(listener)

  def remove_listener(self, listener) -> None:
    if self._stream is None:
      raise RuntimeError("STTService has not been started.")

    self._stream.remove_listener(listener)

  def push(self, audio_data, sample_rate: int) -> None:
    """
    Push normalized float32 audio samples to the transcriber.

    Args:
      audio_data:
        A sequence (typically a NumPy float32 array) of normalized
        audio samples in the range [-1.0, 1.0].

      sample_rate:
        Sample rate of the incoming audio.
    """
    if self._stream is None:
      raise RuntimeError("STTService has not been started.")

    self._audio_queue.put((audio_data, sample_rate))

  def _process_audio_queue(self) -> None:
    while self._running:
        try:
            audio_data, sample_rate = self._audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if self._stream is None:
            continue

        self._stream.add_audio(audio_data, sample_rate)
  
  def stop(self) -> None:
    self._running = False

    if self._worker_thread is not None:
      self._worker_thread.join()
      self._worker_thread = None
      
    if self._stream is not None:
      self._stream.stop()
      self._stream.close()
      self._stream = None

    self._transcriber = None