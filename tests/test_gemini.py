import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.environ.setdefault("RELAY_TOKEN", "test-token")

from app.gemini import GeminiClient  # noqa: E402


class _DummyResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.text = "{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "dummy-text"},
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        }


class _DummyHttpxClient:
    def __init__(self) -> None:
        self.captured_params = None

    def post(self, url: str, *, params: dict, json: dict):  # type: ignore[override]
        self.captured_params = params
        return _DummyResponse()

    def close(self) -> None:
        return None


def test_generate_prefers_override_api_key() -> None:
    client = GeminiClient(api_key=None, default_model="gemini-test", timeout=1.0)
    dummy_client = _DummyHttpxClient()
    client._client = dummy_client  # type: ignore[attr-defined]

    result = client.generate(
        model=None,
        prompt="テストプロンプト",
        page_bytes=b"pdf-bytes",
        mime_type="application/pdf",
        masters={"shipCsv": "ship", "itemCsv": "item"},
        api_key_override="override-key",
    )

    assert dummy_client.captured_params == {"key": "override-key"}
    assert result.text == "dummy-text"
    assert result.meta["tokensInput"] == 10
    assert result.meta["tokensOutput"] == 5
