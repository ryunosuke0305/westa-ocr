"""Application configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os


def _read_int(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str
    gemini_model: str
    max_upload_size: int
    request_timeout: int


def load_config() -> AppConfig:
    """Load application configuration from the environment."""
    max_upload_size_mb = _read_int("MAX_UPLOAD_SIZE_MB", 15)
    return AppConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-latest"),
        max_upload_size=max_upload_size_mb * 1024 * 1024,
        request_timeout=_read_int("OCR_REQUEST_TIMEOUT", 300),
    )
