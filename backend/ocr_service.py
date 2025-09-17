"""OCR service that integrates with Google Gemini."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PyPDF2 import PdfReader

from config import AppConfig

try:  # pragma: no cover - the library may not be installed during CI
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class OCRResult:
    pdf_id: str
    delivery_location: str
    customer_name: str
    customer_order_number: str
    order_date: str
    shipping_date: str
    customer_delivery_date: str
    customer_item_code: str
    internal_item_code: str
    product_name: str
    quantity: str
    unit: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_id": self.pdf_id,
            "delivery_location": self.delivery_location,
            "customer_name": self.customer_name,
            "customer_order_number": self.customer_order_number,
            "order_date": self.order_date,
            "shipping_date": self.shipping_date,
            "customer_delivery_date": self.customer_delivery_date,
            "customer_item_code": self.customer_item_code,
            "internal_item_code": self.internal_item_code,
            "product_name": self.product_name,
            "quantity": self.quantity,
            "unit": self.unit,
            "notes": self.notes,
        }


class OCRService:
    """Encapsulates Gemini based OCR processing with graceful fallback."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = None
        if config.gemini_api_key and genai is not None:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self._model = genai.GenerativeModel(config.gemini_model)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Failed to initialise Gemini client: %%s", exc)
                self._model = None
        elif config.gemini_api_key:
            LOGGER.warning(
                "google-generativeai is not available. Falling back to local extraction."
            )

    def extract(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract structured data from the supplied PDF."""
        if self._model is not None:
            try:
                return self._extract_with_gemini(file_bytes, filename)
            except Exception as exc:  # pragma: no cover - remote API failure
                LOGGER.error("Gemini extraction failed: %%s", exc)

        return self._fallback_extraction(file_bytes, filename)

    # ------------------------------------------------------------------
    def _extract_with_gemini(
        self, file_bytes: bytes, filename: str
    ) -> List[Dict[str, Any]]:
        assert self._model is not None  # for type checkers
        response = self._model.generate_content(
            [
                self._prompt_template(filename),
                {
                    "mime_type": "application/pdf",
                    "data": file_bytes,
                },
            ],
            request_options={"timeout": self._config.request_timeout},
        )
        text = self._extract_text_from_response(response)
        payload = json.loads(text)
        results: Iterable[Dict[str, Any]]
        if isinstance(payload, dict) and "results" in payload:
            results = payload["results"]
        elif isinstance(payload, list):
            results = payload
        else:  # pragma: no cover - defensive branch
            raise ValueError("Unexpected response structure from Gemini")
        return [self._normalise_result(item, filename) for item in results]

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        # Fallback: inspect candidate parts
        candidates = getattr(response, "candidates", [])
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", [])
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    return text
        raise ValueError("Could not extract text from Gemini response")

    # ------------------------------------------------------------------
    @staticmethod
    def _prompt_template(filename: str) -> str:
        return (
            "You are an OCR assistant. Extract structured order data from the PDF "
            f"named '{filename}'. Return an array of JSON objects using the keys: "
            "pdf_id, delivery_location, customer_name, customer_order_number, order_date, "
            "shipping_date, customer_delivery_date, customer_item_code, internal_item_code, "
            "product_name, quantity, unit, notes. Use ISO formatted dates and numbers."
        )

    # ------------------------------------------------------------------
    def _fallback_extraction(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        stream = BytesIO(file_bytes)
        try:
            reader = PdfReader(stream)
            pages = list(reader.pages)
        except Exception:
            pages = []
        base_id = Path(filename).stem or "document"
        results: List[OCRResult] = []
        if not pages:
            results.append(
                OCRResult(
                    pdf_id=base_id,
                    delivery_location="",
                    customer_name="",
                    customer_order_number="",
                    order_date="",
                    shipping_date="",
                    customer_delivery_date="",
                    customer_item_code="",
                    internal_item_code="",
                    product_name="",
                    quantity="0",
                    unit="",
                    notes="テキストを抽出できませんでした。Gemini APIキーを設定すると高精度の抽出が可能です。",
                )
            )
            return [item.to_dict() for item in results]

        for index, page in enumerate(pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            snippet = " ".join(part.strip() for part in text.splitlines() if part.strip())
            snippet = snippet[:200]
            results.append(
                OCRResult(
                    pdf_id=f"{base_id}-p{index}",
                    delivery_location="",
                    customer_name="",
                    customer_order_number="",
                    order_date="",
                    shipping_date="",
                    customer_delivery_date="",
                    customer_item_code="",
                    internal_item_code="",
                    product_name="",
                    quantity="0",
                    unit="",
                    notes=snippet,
                )
            )
        return [item.to_dict() for item in results]

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_result(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        defaults = OCRResult(
            pdf_id=Path(filename).stem or "document",
            delivery_location="",
            customer_name="",
            customer_order_number="",
            order_date="",
            shipping_date="",
            customer_delivery_date="",
            customer_item_code="",
            internal_item_code="",
            product_name="",
            quantity="0",
            unit="",
            notes="",
        ).to_dict()
        normalised = {**defaults, **{k: ("" if v is None else v) for k, v in data.items()}}
        # Ensure quantity is always serialisable as string to keep frontend simple
        quantity = normalised.get("quantity", "0")
        normalised["quantity"] = str(quantity)
        return normalised
