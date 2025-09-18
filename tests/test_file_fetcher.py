"""Tests for the :mod:`app.file_fetcher` module."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

os.environ.setdefault("RELAY_TOKEN", "test-token")

import pytest

from app.file_fetcher import FileFetcher


class DummyResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class DummyClient:
    def __init__(self, expected_content: bytes) -> None:
        self._content = expected_content
        self.requests: list[tuple[str, dict | None, dict | None]] = []

    def get(self, url: str, params=None, headers=None):  # noqa: D401 - simple stub
        self.requests.append((url, params, headers))
        return DummyResponse(self._content)


class DummyCredentials:
    def __init__(self, token: str = "token") -> None:
        self.token = token
        self.valid = True

    def refresh(self, _request) -> None:  # pragma: no cover - not used in tests
        self.valid = True
        self.token = "token"


def test_fetch_drive_requires_credentials() -> None:
    fetcher = FileFetcher(timeout=1.0, drive_service_account_json=None)
    with pytest.raises(ValueError) as exc:
        fetcher.fetch("1A2B3C4D5E6F7G8H9I0J")
    assert "Google Drive" in str(exc.value)


def test_fetch_drive_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = FileFetcher(timeout=1.0, drive_service_account_json=None)

    dummy_client = DummyClient(b"content")
    monkeypatch.setattr(fetcher, "_ensure_client", lambda: dummy_client)
    monkeypatch.setattr(fetcher, "_ensure_drive_credentials", lambda: DummyCredentials())

    # Pretend that credentials are configured so the Drive branch is used
    fetcher._drive_service_account_json = Path("dummy.json")  # type: ignore[attr-defined]

    result = fetcher.fetch("drive:1A2B3C4D5E6F7G8H9I0J")

    assert result == b"content"
    assert dummy_client.requests
    url, params, headers = dummy_client.requests[0]
    assert url.endswith("/1A2B3C4D5E6F7G8H9I0J")
    assert params == {"alt": "media", "supportsAllDrives": "true"}
    assert headers["Authorization"].startswith("Bearer ")
