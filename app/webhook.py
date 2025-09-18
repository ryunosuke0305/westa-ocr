"""Webhook dispatch utilities."""

from __future__ import annotations

import json
from typing import Dict, Optional

import httpx

from .logging_config import get_logger

LOGGER = get_logger(__name__)


class WebhookDispatcher:
    """Send webhook payloads."""

    def __init__(self, timeout: float) -> None:
        # Google Apps Script の Webhook など、302 リダイレクトで実体レスポンスを返す
        # エンドポイントにも対応できるよう、リダイレクトの追跡を有効化する。
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def send(self, url: str, payload: Dict, *, token: Optional[str] = None) -> httpx.Response:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        LOGGER.info(
            "Dispatching webhook",
            extra={"url": url, "event": payload.get("event"), "jobId": payload.get("jobId")},
        )
        response = self._client.post(url, content=raw, headers=headers)
        response.raise_for_status()
        return response


__all__ = ["WebhookDispatcher"]
