"""Utility for retrieving PDF payloads."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import httpx

from .logging_config import get_logger

try:  # pragma: no cover - optional dependency guard
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2 import service_account
except Exception:  # pragma: no cover - optional dependency guard
    GoogleAuthRequest = None  # type: ignore
    service_account = None  # type: ignore

LOGGER = get_logger(__name__)

_DRIVE_FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


class FileFetcher:
    """Simple file retriever supporting HTTP(S), local paths, and Google Drive."""

    def __init__(self, timeout: float, drive_service_account_json: Optional[Path] = None) -> None:
        self._timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._drive_service_account_json = drive_service_account_json
        self._drive_credentials = None

        if self._drive_service_account_json and (GoogleAuthRequest is None or service_account is None):
            raise RuntimeError(
                "google-auth is required for Google Drive support but is not available"
            )

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _ensure_drive_credentials(self):
        if not self._drive_service_account_json:
            return None

        if GoogleAuthRequest is None or service_account is None:
            raise RuntimeError(
                "google-auth is required for Google Drive support but is not available"
            )

        if self._drive_credentials is None:
            LOGGER.info(
                "Initialising Google Drive credentials",
                extra={"credentialsPath": str(self._drive_service_account_json)},
            )
            self._drive_credentials = service_account.Credentials.from_service_account_file(  # type: ignore[arg-type]
                str(self._drive_service_account_json),
                scopes=[_DRIVE_SCOPE],
            )

        if not self._drive_credentials.valid:
            LOGGER.debug("Refreshing Google Drive access token")
            request = GoogleAuthRequest()
            self._drive_credentials.refresh(request)

        return self._drive_credentials

    @staticmethod
    def _normalise_drive_file_id(file_id: str) -> Optional[str]:
        candidate = file_id.strip()
        if candidate.startswith("drive://"):
            candidate = candidate[8:]
        elif candidate.startswith("drive:"):
            candidate = candidate.split(":", 1)[1].strip()

        if _DRIVE_FILE_ID_PATTERN.fullmatch(candidate):
            return candidate
        return None

    def _fetch_from_drive(self, drive_file_id: str) -> bytes:
        credentials = self._ensure_drive_credentials()
        if credentials is None:
            raise ValueError(
                "Google Drive credentials not configured. Set DRIVE_SERVICE_ACCOUNT_JSON."
            )

        if not credentials.token:
            # Ensure token is available (refresh handles validity above but token may still be None)
            LOGGER.debug("Refreshing Google Drive token due to missing token")
            request = GoogleAuthRequest()
            credentials.refresh(request)

        client = self._ensure_client()
        LOGGER.info("Fetching file from Google Drive", extra={"fileId": drive_file_id})
        response = client.get(
            f"https://www.googleapis.com/drive/v3/files/{drive_file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
            headers={"Authorization": f"Bearer {credentials.token}"},
        )
        response.raise_for_status()
        return response.content

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

        drive_file_id = self._normalise_drive_file_id(file_id)
        if drive_file_id:
            return self._fetch_from_drive(drive_file_id)

        raise ValueError(
            "Unsupported file identifier. Provide an HTTP(S) URL, file:// path, or a valid "
            "Google Drive file ID."
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


__all__ = ["FileFetcher"]
