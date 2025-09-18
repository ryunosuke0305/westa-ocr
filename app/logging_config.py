"""Logging configuration utilities."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured, concise format."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # When running under uvicorn there will already be handlers; update their levels instead.
        for handler in root_logger.handlers:
            handler.setLevel(level)
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name if name else __name__)


__all__ = ["configure_logging", "get_logger"]
