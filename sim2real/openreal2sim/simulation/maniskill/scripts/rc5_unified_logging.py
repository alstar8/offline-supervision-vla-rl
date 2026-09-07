from __future__ import annotations

import os
import sys

from loguru import logger


def resolve_log_level() -> str:
    return str(os.environ.get("RC5_LOG_LEVEL", "INFO") or "INFO").strip().upper()


logger.remove()
logger.add(sys.stderr, level=resolve_log_level(), format="[{level}] [{name}] {message}")


def is_debug_enabled() -> bool:
    return resolve_log_level() == "DEBUG"
