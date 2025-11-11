"""Gemini API integration (with a graceful fallback for local development)."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx

from .logging_config import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class GeminiResult:
    text: str
    meta: Dict[str, object]


class GeminiClient:
    """Thin wrapper around the public Gemini REST API."""

    def __init__(self, api_key: Optional[str], *, default_model: str, timeout: float) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def default_model(self) -> str:
        return self._default_model

    def generate(
        self,
        *,
        model: Optional[str],
        prompt: str,
        page_bytes: bytes,
        mime_type: str,
        masters: Dict[str, str],
        api_key_override: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> GeminiResult:
        target_model = model or self._default_model
        api_key = api_key_override or self._api_key
        if not api_key:
            LOGGER.warning(
                "Gemini API key not configured; returning simulated output.",
                extra={"model": target_model},
            )
            preview = prompt[:200].replace("\n", " ")
            text = (
                "[simulated Gemini response]\n"
                f"model={target_model}\n"
                f"prompt_preview={preview}\n"
                f"masters_keys={list(masters.keys())}\n"
                "(Set GEMINI_API_KEY to enable live inference)"
            )
            return GeminiResult(text=text, meta={"model": target_model, "durationMs": 0, "tokensInput": None, "tokensOutput": None})

        ship_csv = masters.get("shipCsv") if isinstance(masters, dict) else None
        item_csv = masters.get("itemCsv") if isinstance(masters, dict) else None

        text_segments = [prompt]
        if ship_csv:
            text_segments.append(ship_csv)
        if item_csv:
            text_segments.append(item_csv)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(page_bytes).decode("utf-8"),
                            }
                        },
                        {"text": "\n\n".join(text_segments)},
                    ],
                }
            ],
            "generationConfig": {
                k: v
                for k, v in {
                    "temperature": temperature,
                    "topP": top_p,
                    "topK": top_k,
                    "maxOutputTokens": max_output_tokens,
                }.items()
                if v is not None
            },
        }

        client = self._ensure_client()
        start = time.perf_counter()
        response = client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent",
            params={"key": api_key},
            json=payload,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text") for part in parts if part.get("text")]
        text = "\n".join(text_parts)
        usage = data.get("usageMetadata", {})
        meta = {
            "model": target_model,
            "durationMs": duration_ms,
            "tokensInput": usage.get("promptTokenCount"),
            "tokensOutput": usage.get("candidatesTokenCount"),
        }
        return GeminiResult(text=text, meta=meta)


__all__ = ["GeminiClient", "GeminiResult"]
