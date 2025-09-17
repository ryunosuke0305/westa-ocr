"""Application configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


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
    upload_dir: Path
    job_history_dir: Path
    response_history_dir: Path
    api_key_dir: Path


def load_config() -> AppConfig:
    """Load application configuration from the environment."""
    max_upload_size_mb = _read_int("MAX_UPLOAD_SIZE_MB", 15)
    data_root = Path(os.getenv("DATA_ROOT", "/data"))
    upload_dir = Path(os.getenv("UPLOAD_DIR", str(data_root / "uploads")))
    job_history_dir = Path(os.getenv("JOB_HISTORY_DIR", str(data_root / "jobs")))
    response_history_dir = Path(
        os.getenv("RESPONSE_HISTORY_DIR", str(data_root / "responses"))
    )
    api_key_dir = Path(os.getenv("API_KEY_DIR", str(data_root / "keys")))

    for directory in (upload_dir, job_history_dir, response_history_dir, api_key_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-latest"),
        max_upload_size=max_upload_size_mb * 1024 * 1024,
        request_timeout=_read_int("OCR_REQUEST_TIMEOUT", 300),
        upload_dir=upload_dir,
        job_history_dir=job_history_dir,
        response_history_dir=response_history_dir,
        api_key_dir=api_key_dir,
    )
