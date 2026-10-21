"""Centralized logging configuration for Waffle Audio.

Sets up robust, unbuffered file logging to the logs/ directory:
- logs/voicechat.log: Comprehensive application execution log
- logs/benchmark.log: Dedicated benchmark cards and cumulative averages
- Suppresses excessive third-party noise while preserving all pipeline timings
"""
from __future__ import annotations

import logging
import os
from pathlib import Path


from constants import is_benchmark_enabled


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes immediately after every emit so logs appear in real-time."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_app_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    main_log_file: str = "voicechat.log",
    benchmark_log_file: str = "benchmark.log",
    benchmark_enabled: bool | None = None,
) -> list[logging.Handler]:
    """Initialize file logging for the application and benchmark systems."""
    is_bm = is_benchmark_enabled() if benchmark_enabled is None else bool(benchmark_enabled)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    voicechat_log = log_path / main_log_file
    benchmark_log = log_path / benchmark_log_file

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any stale FileHandlers pointing to voicechat.log
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith(main_log_file):
            root.removeHandler(h)

    main_handler = FlushingFileHandler(voicechat_log, encoding="utf-8")
    main_handler.setLevel(level)
    main_handler.setFormatter(formatter)
    root.addHandler(main_handler)

    handlers: list[logging.Handler] = [main_handler]

    # 2. Configure dedicated benchmark logger (only if benchmark is enabled)
    bm_logger = logging.getLogger("waffle.benchmark")
    bm_logger.setLevel(logging.INFO if is_bm else logging.WARNING)
    bm_logger.propagate = False  # Avoid duplicating in root if already writing directly

    for h in list(bm_logger.handlers):
        bm_logger.removeHandler(h)

    if is_bm:
        bm_handler = FlushingFileHandler(benchmark_log, encoding="utf-8")
        bm_handler.setLevel(logging.INFO)
        bm_handler.setFormatter(logging.Formatter("%(message)s"))
        bm_logger.addHandler(bm_handler)
        handlers.append(bm_handler)

    for name in [
        "httpx",
        "httpcore",
        "onnxruntime",
        "moonshine_voice",
        "urllib3",
        "pocket_tts",
        "kokoro_onnx",
        "asyncio",
        "multipart",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    return handlers
