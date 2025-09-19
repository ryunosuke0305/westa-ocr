"""Application settings and configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class Settings:
    """Container for configuration values loaded from environment variables."""

    relay_token: str
    sqlite_path: Path
    data_dir: Path
    tmp_dir: Path
    worker_idle_sleep: float
    worker_count: int
    gemini_api_key: Optional[str]
    gemini_model: str
    webhook_timeout: float
    request_timeout: float
    log_level: str
    drive_service_account_json: Optional[Path]

def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


def _read_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}, got {value}")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read and memoise :class:`Settings` from environment variables."""

    data_dir = Path(os.getenv("DATA_DIR", "/data")).resolve()
    sqlite_path = Path(os.getenv("SQLITE_PATH", str(data_dir / "relay.db"))).resolve()
    tmp_dir = Path(os.getenv("TMP_DIR", str(data_dir / "tmp"))).resolve()

    relay_token = os.getenv("RELAY_TOKEN")
    if not relay_token:
        raise RuntimeError("RELAY_TOKEN environment variable must be set")

    drive_service_account = os.getenv("DRIVE_SERVICE_ACCOUNT_JSON")
    drive_service_account_path = (
        Path(drive_service_account).expanduser().resolve()
        if drive_service_account
        else None
    )

    return Settings(
        relay_token=relay_token,
        sqlite_path=sqlite_path,
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        worker_idle_sleep=_read_float("WORKER_IDLE_SLEEP", 1.0),
        worker_count=_read_int("WORKER_COUNT", 3, minimum=1),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        webhook_timeout=_read_float("WEBHOOK_TIMEOUT", 30.0),
        request_timeout=_read_float("REQUEST_TIMEOUT", 600.0),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        drive_service_account_json=drive_service_account_path,
    )


__all__ = ["Settings", "get_settings"]
