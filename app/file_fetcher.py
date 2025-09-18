"""Utility for retrieving PDF payloads."""

from __future__ import annotations

import httpx
from pathlib import Path
from typing import Optional

from .logging_config import get_logger

LOGGER = get_logger(__name__)


class FileFetcher:
    """Simple file retriever supporting HTTP(S) and local file paths."""

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def fetch(self, file_id: str) -> bytes:
        """Fetch a file referenced by ``file_id``."""

        if file_id.startswith("http://") or file_id.startswith("https://"):
            client = self._ensure_client()
            LOGGER.info("Fetching file via HTTP", extra={"fileId": file_id})
            response = client.get(file_id)
            response.raise_for_status()
            return response.content

        if file_id.startswith("file://"):
            path = Path(file_id[7:]).expanduser().resolve()
            LOGGER.info("Fetching file from local filesystem", extra={"path": str(path)})
            return path.read_bytes()

        if file_id.startswith("local:"):
            path = Path(file_id.split(":", 1)[1]).expanduser().resolve()
            LOGGER.info("Fetching file from local shorthand", extra={"path": str(path)})
            return path.read_bytes()

        raise ValueError(
            "Unsupported file identifier. Provide an HTTP(S) URL, file:// path, or "
            "extend FileFetcher to integrate with Google Drive."
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


__all__ = ["FileFetcher"]
