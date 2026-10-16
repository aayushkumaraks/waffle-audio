"""Local-first Hugging Face loading.

Models are loaded from the local Hugging Face cache without any network
request. Only when the local data is missing do we go online, using a Hugging
Face token from the environment / token file, or one entered by the user.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Callable, TypeVar

from huggingface_hub import constants as hf_constants
from huggingface_hub import get_token

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _set_offline(offline: bool) -> None:
    # huggingface_hub reads the flag at import time, so toggle both the
    # environment (for child code paths) and the already-imported constant.
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    hf_constants.HF_HUB_OFFLINE = offline


def _ensure_token() -> None:
    """Make sure a token is available for the online request, if the user has one."""
    if get_token():
        return
    token = ""
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            token = input("Hugging Face token not found. Enter it (leave empty to continue without one): ").strip()
        except EOFError:
            token = ""
    else:
        logger.warning("No Hugging Face token found and no interactive terminal; continuing without one.")
    if token:
        os.environ["HF_TOKEN"] = token


def load_local_first(loader: Callable[[], T]) -> T:
    """Run ``loader`` offline first; if local data is missing, retry online."""
    previous = hf_constants.HF_HUB_OFFLINE
    _set_offline(True)
    try:
        result = loader()
        logger.info("Loaded model from the local Hugging Face cache.")
        return result
    except Exception as exc:
        logger.info("Local model data not available (%s: %s); contacting Hugging Face.", type(exc).__name__, exc)
    finally:
        _set_offline(previous)

    _ensure_token()
    return loader()
