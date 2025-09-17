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
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

LOGGER = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = (
    "You are an OCR assistant. Extract structured order data from the PDF named "
    "'{filename}'. Respond in pure JSON (no markdown) that strictly matches the "
    "following schema. Each element of the array is an object that must include "
    "ALL of these keys (Japanese, case sensitive):\n"
    "- PDFのID (string): Identifier for the page or document.\n"
    "- 納入場所 (string).\n"
    "- 得意先 (string).\n"
    "- 得意先注文番号 (string).\n"
    "- 受注日 (string, ISO-8601 date).\n"
    "- 出荷予定日 (string, ISO-8601 date).\n"
    "- 顧客納期 (string, ISO-8601 date).\n"
    "- 得意先品目コード (string).\n"
    "- 自社品目コード (string).\n"
    "- 受注商品名称 (string).\n"
    "- 受注数 (number, do not quote).\n"
    "- 単位 (string).\n"
    "- 受注記事 (string).\n"
    "Use empty strings when information is missing, except set 受注数 to 0 when "
    "the quantity is unknown. Do not include any extra keys or commentary."
)


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
    quantity: float
    unit: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "PDFのID": self.pdf_id,
            "納入場所": self.delivery_location,
            "得意先": self.customer_name,
            "得意先注文番号": self.customer_order_number,
            "受注日": self.order_date,
            "出荷予定日": self.shipping_date,
            "顧客納期": self.customer_delivery_date,
            "得意先品目コード": self.customer_item_code,
            "自社品目コード": self.internal_item_code,
            "受注商品名称": self.product_name,
            "受注数": float(self.quantity),
            "単位": self.unit,
            "受注記事": self.notes,
        }


STRUCTURED_OUTPUT_SCHEMA = None
if genai_types is not None:
    STRUCTURED_OUTPUT_SCHEMA = genai_types.Schema(
        type=genai_types.Type.ARRAY,
        items=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "PDFのID": genai_types.Schema(type=genai_types.Type.STRING),
                "納入場所": genai_types.Schema(type=genai_types.Type.STRING),
                "得意先": genai_types.Schema(type=genai_types.Type.STRING),
                "得意先注文番号": genai_types.Schema(type=genai_types.Type.STRING),
                "受注日": genai_types.Schema(type=genai_types.Type.STRING),
                "出荷予定日": genai_types.Schema(type=genai_types.Type.STRING),
                "顧客納期": genai_types.Schema(type=genai_types.Type.STRING),
                "得意先品目コード": genai_types.Schema(type=genai_types.Type.STRING),
                "自社品目コード": genai_types.Schema(type=genai_types.Type.STRING),
                "受注商品名称": genai_types.Schema(type=genai_types.Type.STRING),
                "受注数": genai_types.Schema(type=genai_types.Type.NUMBER),
                "単位": genai_types.Schema(type=genai_types.Type.STRING),
                "受注記事": genai_types.Schema(type=genai_types.Type.STRING),
            },
            required=[
                "PDFのID",
                "納入場所",
                "得意先",
                "得意先注文番号",
                "受注日",
                "出荷予定日",
                "顧客納期",
                "得意先品目コード",
                "自社品目コード",
                "受注商品名称",
                "受注数",
                "単位",
                "受注記事",
            ],
        ),
    )


class OCRService:
    """Encapsulates Gemini based OCR processing with graceful fallback."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._prompt_template_path = config.prompt_template_path
        self._ensure_prompt_template()
        self._client = None
        if config.gemini_api_key and genai is not None and genai_types is not None:
            try:
                self._client = genai.Client(api_key=config.gemini_api_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Failed to initialise Gemini client: %s", exc)
                self._client = None
        elif config.gemini_api_key:
            LOGGER.warning(
                "google-genai is not available. Falling back to local extraction."
            )

    def extract(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract structured data from the supplied PDF."""
        if self._client is not None and genai_types is not None:
            try:
                return self._extract_with_gemini(file_bytes, filename)
            except Exception as exc:  # pragma: no cover - remote API failure
                LOGGER.error("Gemini extraction failed: %%s", exc)

        return self._fallback_extraction(file_bytes, filename)

    # ------------------------------------------------------------------
    def _extract_with_gemini(
        self, file_bytes: bytes, filename: str
    ) -> List[Dict[str, Any]]:
        assert self._client is not None and genai_types is not None  # for type checkers
        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(text=self._build_prompt(filename)),
                    genai_types.Part.from_bytes(
                        data=file_bytes, mime_type="application/pdf"
                    ),
                ],
            )
        ]
        generate_config = genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="application/json",
            response_schema=STRUCTURED_OUTPUT_SCHEMA,
            http_options=genai_types.HttpOptions(
                timeout=self._config.request_timeout
            ),
        )
        response = self._client.models.generate_content(
            model=self._config.gemini_model,
            contents=contents,
            config=generate_config,
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
        parts = getattr(response, "parts", None)
        if parts:
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    return text
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
    def get_prompt_template(self) -> str:
        """Return the current prompt template string."""
        return self._read_prompt_template()

    def update_prompt_template(self, prompt: str) -> None:
        """Persist a new prompt template."""
        self._prompt_template_path.write_text(prompt, encoding="utf-8")

    # ------------------------------------------------------------------
    def _ensure_prompt_template(self) -> None:
        if not self._prompt_template_path.exists():
            try:
                self._prompt_template_path.write_text(
                    DEFAULT_PROMPT_TEMPLATE, encoding="utf-8"
                )
            except OSError as exc:  # pragma: no cover - best effort initialisation
                LOGGER.warning("プロンプトテンプレートの初期化に失敗しました: %s", exc)

    def _read_prompt_template(self) -> str:
        try:
            contents = self._prompt_template_path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("プロンプトテンプレートの読み込みに失敗しました: %s", exc)
            contents = ""
        if not contents.strip():
            return DEFAULT_PROMPT_TEMPLATE
        return contents

    def _build_prompt(self, filename: str) -> str:
        template = self._read_prompt_template()
        try:
            return template.format(filename=filename)
        except Exception as exc:  # pragma: no cover - defensive formatting
            LOGGER.warning("プロンプトの整形に失敗したため生のテンプレートを使用します: %s", exc)
            return template

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
                    quantity=0.0,
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
                    quantity=0.0,
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
            quantity=0.0,
            unit="",
            notes="",
        ).to_dict()
        normalised = dict(defaults)
        allowed_keys = set(defaults)
        for key, value in data.items():
            if key not in allowed_keys:
                continue
            if key == "受注数":
                normalised[key] = OCRService._coerce_quantity(value)
            else:
                normalised[key] = "" if value is None else str(value)
        for key in allowed_keys - {"受注数"}:
            current = normalised.get(key, "")
            normalised[key] = "" if current is None else str(current)
        if "受注数" not in normalised:
            normalised["受注数"] = 0.0
        return normalised

    @staticmethod
    def _coerce_quantity(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return 0.0
            try:
                normalised = stripped.replace(",", "")
                return float(normalised)
            except ValueError:
                return 0.0
        return 0.0
