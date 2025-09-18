"""Webhook dispatch utilities."""

from __future__ import annotations

import json
from typing import Dict

import httpx

from .logging_config import get_logger

LOGGER = get_logger(__name__)


class WebhookDispatcher:
    """Send webhook payloads."""

    def __init__(self, timeout: float) -> None:
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def send(self, url: str, payload: Dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        LOGGER.info(
            "Dispatching webhook",
            extra={"url": url, "event": payload.get("event"), "jobId": payload.get("jobId")},
        )
        response = self._client.post(url, content=raw, headers=headers)
        response.raise_for_status()


__all__ = ["WebhookDispatcher"]
