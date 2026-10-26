"""Unit tests for services/logging_config.py centralized logging setup."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from services.logging_config import FlushingFileHandler, setup_app_logging


class TestLoggingConfig(unittest.TestCase):
    def test_flushing_file_handler_emits_and_flushes(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            log_file = f.name

        try:
            handler = FlushingFileHandler(log_file, encoding="utf-8")
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="test log message",
                args=(),
                exc_info=None,
            )
            handler.emit(record)
            handler.close()

            with open(log_file, "r", encoding="utf-8") as rf:
                content = rf.read()
            self.assertIn("test log message", content)
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_setup_app_logging_creates_handlers_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            handlers = setup_app_logging(
                log_dir=tmpdir,
                main_log_file="app.log",
                benchmark_log_file="bm.log",
                benchmark_enabled=True,
            )
            self.assertGreaterEqual(len(handlers), 2)

            root = logging.getLogger()
            bm_logger = logging.getLogger("waffle.benchmark")

            root.info("Root message into app.log")
            bm_logger.info("Benchmark message into bm.log")

            for h in handlers:
                h.flush()
                h.close()
                root.removeHandler(h)
                bm_logger.removeHandler(h)

            app_log = Path(tmpdir) / "app.log"
            bm_log = Path(tmpdir) / "bm.log"

            self.assertTrue(app_log.exists())
            self.assertTrue(bm_log.exists())
            self.assertIn("Root message into app.log", app_log.read_text(encoding="utf-8"))
            self.assertIn("Benchmark message into bm.log", bm_log.read_text(encoding="utf-8"))

    def test_setup_app_logging_benchmark_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            handlers = setup_app_logging(
                log_dir=tmpdir,
                main_log_file="app2.log",
                benchmark_log_file="bm2.log",
                benchmark_enabled=False,
            )
            self.assertEqual(len(handlers), 1)
            for h in handlers:
                h.close()
                logging.getLogger().removeHandler(h)


if __name__ == "__main__":
    unittest.main()
