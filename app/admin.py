"""Admin dashboard endpoints for manual operations and configuration tweaks."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from .file_fetcher import FileFetcher
from .gemini import GeminiClient
from .logging_config import configure_logging, get_logger
from .settings import Settings, get_settings
from .webhook import WebhookDispatcher
from .worker import JobWorker

LOGGER = get_logger(__name__)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@dataclass(slots=True)
class GeminiLogEntry:
    """Result of a manual Gemini invocation."""

    timestamp: datetime
    prompt_preview: str
    model: str
    mime_type: str
    success: bool
    response_text: Optional[str]
    meta: Optional[Dict]
    error: Optional[str]


@dataclass(slots=True)
class WebhookLogEntry:
    """Result of a manual webhook dispatch."""

    timestamp: datetime
    url: str
    payload: str
    success: bool
    status_code: Optional[int]
    response_text: Optional[str]
    error: Optional[str]


@dataclass(slots=True)
class AdminMessage:
    """Feedback banner to display on the dashboard."""

    timestamp: datetime
    category: str
    success: bool
    summary: str
    detail: Optional[str]


class AdminState:
    """Container keeping the latest admin operations."""

    def __init__(self) -> None:
        self._gemini_logs: List[GeminiLogEntry] = []
        self._webhook_logs: List[WebhookLogEntry] = []
        self._messages: List[AdminMessage] = []

    @property
    def gemini_logs(self) -> List[GeminiLogEntry]:
        return list(self._gemini_logs)

    @property
    def webhook_logs(self) -> List[WebhookLogEntry]:
        return list(self._webhook_logs)

    @property
    def messages(self) -> List[AdminMessage]:
        return list(self._messages)

    def add_gemini_log(self, entry: GeminiLogEntry) -> None:
        self._gemini_logs.insert(0, entry)
        del self._gemini_logs[10:]

    def add_webhook_log(self, entry: WebhookLogEntry) -> None:
        self._webhook_logs.insert(0, entry)
        del self._webhook_logs[10:]

    def add_message(self, entry: AdminMessage) -> None:
        self._messages.insert(0, entry)
        del self._messages[10:]


CONFIG_FIELDS: List[Dict[str, str]] = [
    {"env": "RELAY_TOKEN", "label": "Relay Token", "placeholder": "必須"},
    {"env": "DATA_DIR", "label": "Data Directory", "placeholder": "/data"},
    {"env": "SQLITE_PATH", "label": "SQLite Path", "placeholder": "<DATA_DIR>/relay.db"},
    {"env": "TMP_DIR", "label": "Temporary Directory", "placeholder": "<DATA_DIR>/tmp"},
    {"env": "WORKER_IDLE_SLEEP", "label": "Worker Idle Sleep", "placeholder": "1.0"},
    {"env": "GEMINI_API_KEY", "label": "Gemini API Key", "placeholder": ""},
    {"env": "GEMINI_MODEL", "label": "Gemini Model", "placeholder": "gemini-2.5-flash"},
    {"env": "WEBHOOK_TIMEOUT", "label": "Webhook Timeout", "placeholder": "30"},
    {"env": "REQUEST_TIMEOUT", "label": "Request Timeout", "placeholder": "60"},
    {"env": "LOG_LEVEL", "label": "Log Level", "placeholder": "INFO"},
]


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _build_dashboard_payload(settings: Settings, state: AdminState) -> Dict[str, Any]:
    config_fields = []
    for field in CONFIG_FIELDS:
        env_name = field["env"]
        config_fields.append(
            {
                "env": env_name,
                "label": field["label"],
                "placeholder": field.get("placeholder", ""),
                "value": os.getenv(env_name, ""),
                "type": "password"
                if env_name in {"RELAY_TOKEN", "GEMINI_API_KEY"}
                else "text",
            }
        )

    messages = [
        {
            "category": message.category,
            "success": message.success,
            "summary": message.summary,
            "detail": message.detail,
            "timestamp": _format_datetime(message.timestamp),
        }
        for message in state.messages
    ]

    gemini_logs = [
        {
            "timestamp": _format_datetime(entry.timestamp),
            "promptPreview": entry.prompt_preview,
            "model": entry.model,
            "mimeType": entry.mime_type,
            "success": entry.success,
            "responseText": entry.response_text,
            "meta": entry.meta,
            "error": entry.error,
        }
        for entry in state.gemini_logs
    ]

    webhook_logs = [
        {
            "timestamp": _format_datetime(entry.timestamp),
            "url": entry.url,
            "payload": entry.payload,
            "success": entry.success,
            "statusCode": entry.status_code,
            "responseText": entry.response_text,
            "error": entry.error,
        }
        for entry in state.webhook_logs
    ]

    return {
        "configFields": config_fields,
        "messages": messages,
        "geminiLogs": gemini_logs,
        "webhookLogs": webhook_logs,
        "defaults": {
            "mimeType": "text/plain",
            "masters": "{}",
            "webhookPayload": "{}",
            "geminiModel": settings.gemini_model,
        },
    }


def _parse_optional_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    return float(value)


def _parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    return int(value)


def _reload_components(app: FastAPI, settings: Settings) -> None:
    LOGGER.info("Reloading application components via admin console")
    worker: JobWorker = app.state.worker
    worker.stop()
    worker.join(timeout=10)

    file_fetcher: FileFetcher = app.state.file_fetcher
    gemini_client: GeminiClient = app.state.gemini_client
    webhook_dispatcher: WebhookDispatcher = app.state.webhook_dispatcher

    file_fetcher.close()
    gemini_client.close()
    webhook_dispatcher.close()

    new_file_fetcher = FileFetcher(settings.request_timeout)
    new_gemini = GeminiClient(
        settings.gemini_api_key,
        default_model=settings.gemini_model,
        timeout=settings.request_timeout,
    )
    new_webhook = WebhookDispatcher(settings.webhook_timeout)
    new_worker = JobWorker(
        repository=app.state.repository,
        job_queue=app.state.job_queue,
        file_fetcher=new_file_fetcher,
        gemini_client=new_gemini,
        webhook_dispatcher=new_webhook,
        idle_sleep=settings.worker_idle_sleep,
    )
    new_worker.start()

    app.state.file_fetcher = new_file_fetcher
    app.state.gemini_client = new_gemini
    app.state.webhook_dispatcher = new_webhook
    app.state.worker = new_worker


def register_admin_routes(app: FastAPI) -> None:
    state: AdminState = app.state.admin_state

    @app.get("/admin")
    async def admin_dashboard(request: Request):
        settings: Settings = request.app.state.settings
        payload = _build_dashboard_payload(settings, state)
        return templates.TemplateResponse("admin.html", {"request": request, "dashboard": payload})

    @app.post("/admin/gemini")
    async def send_gemini(
        request: Request,
        prompt: str = Form(...),
        input_mode: str = Form("text"),
        content: str = Form(""),
        mime_type: str = Form("text/plain"),
        masters: str = Form("{}"),
        model: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        top_k: str = Form(""),
        max_output_tokens: str = Form(""),
    ) -> RedirectResponse:
        prompt = prompt.strip()
        page_content = content.strip()
        if not prompt:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="gemini",
                    success=False,
                    summary="プロンプトを入力してください",
                    detail=None,
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        if not page_content:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="gemini",
                    success=False,
                    summary="入力データを指定してください",
                    detail=None,
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        try:
            masters_payload = json.loads(masters) if masters.strip() else {}
            if not isinstance(masters_payload, dict):
                raise ValueError("masters must be a JSON object")
        except Exception as exc:  # pragma: no cover - defensive
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="gemini",
                    success=False,
                    summary="マスター情報の読み込みに失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        try:
            temperature_value = _parse_optional_float(temperature)
            top_p_value = _parse_optional_float(top_p)
            top_k_value = _parse_optional_int(top_k)
            max_tokens_value = _parse_optional_int(max_output_tokens)
        except ValueError as exc:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="gemini",
                    success=False,
                    summary="数値パラメータの変換に失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        try:
            if input_mode == "base64":
                page_bytes = base64.b64decode(page_content, validate=True)
            else:
                page_bytes = page_content.encode("utf-8")
        except Exception as exc:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="gemini",
                    success=False,
                    summary="入力データの変換に失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        gemini_client: GeminiClient = request.app.state.gemini_client
        try:
            result = gemini_client.generate(
                model=model or None,
                prompt=prompt,
                page_bytes=page_bytes,
                mime_type=mime_type or "application/octet-stream",
                masters=masters_payload,
                temperature=temperature_value,
                top_p=top_p_value,
                top_k=top_k_value,
                max_output_tokens=max_tokens_value,
            )
        except Exception as exc:
            state.add_gemini_log(
                GeminiLogEntry(
                    timestamp=datetime.utcnow(),
                    prompt_preview=prompt[:80],
                    model=model or request.app.state.settings.gemini_model,
                    mime_type=mime_type,
                    success=False,
                    response_text=None,
                    meta=None,
                    error=str(exc),
                )
            )
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="gemini",
                    success=False,
                    summary="Gemini へのリクエストでエラーが発生しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        state.add_gemini_log(
            GeminiLogEntry(
                timestamp=datetime.utcnow(),
                prompt_preview=prompt[:80],
                model=(model or request.app.state.settings.gemini_model),
                mime_type=mime_type,
                success=True,
                response_text=result.text,
                meta=result.meta,
                error=None,
            )
        )
        state.add_message(
            AdminMessage(
                timestamp=datetime.utcnow(),
                category="gemini",
                success=True,
                summary="Gemini へのリクエストが完了しました",
                detail=f"モデル: {(model or request.app.state.settings.gemini_model)}",
            )
        )
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/admin/webhook")
    async def send_webhook(
        request: Request,
        url: str = Form(...),
        payload: str = Form(...),
    ) -> RedirectResponse:
        try:
            payload_dict = json.loads(payload)
            if not isinstance(payload_dict, dict):
                raise ValueError("payload must be a JSON object")
        except Exception as exc:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="webhook",
                    success=False,
                    summary="JSON ペイロードの解析に失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        dispatcher: WebhookDispatcher = request.app.state.webhook_dispatcher
        try:
            response = dispatcher.send(url, payload_dict)
            response_text = response.text
            status_code = response.status_code
        except Exception as exc:
            response_text = None
            status_code = None
            if hasattr(exc, "response") and exc.response is not None:
                status_code = exc.response.status_code
                response_text = exc.response.text
            state.add_webhook_log(
                WebhookLogEntry(
                    timestamp=datetime.utcnow(),
                    url=url,
                    payload=payload,
                    success=False,
                    status_code=status_code,
                    response_text=response_text,
                    error=str(exc),
                )
            )
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="webhook",
                    success=False,
                    summary="Webhook の送信でエラーが発生しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        state.add_webhook_log(
            WebhookLogEntry(
                timestamp=datetime.utcnow(),
                url=url,
                payload=json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":")),
                success=True,
                status_code=status_code,
                response_text=response_text,
                error=None,
            )
        )
        state.add_message(
            AdminMessage(
                timestamp=datetime.utcnow(),
                category="webhook",
                success=True,
                summary="Webhook を送信しました",
                detail=f"HTTP {status_code}",
            )
        )
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/admin/settings")
    async def update_settings(request: Request) -> RedirectResponse:
        form = await request.form()
        updates: Dict[str, Optional[str]] = {}
        for field in CONFIG_FIELDS:
            env_name = field["env"]
            raw_value = form.get(env_name)
            updates[env_name] = raw_value if isinstance(raw_value, str) else None

        if not (updates.get("RELAY_TOKEN") and updates["RELAY_TOKEN"].strip()):
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="settings",
                    success=False,
                    summary="Relay Token は必須です",
                    detail=None,
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        previous_env = {field["env"]: os.getenv(field["env"]) for field in CONFIG_FIELDS}

        def _apply() -> None:
            for key, value in updates.items():
                if value is None:
                    continue
                stripped = value.strip()
                if stripped == "":
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = stripped

        def _restore() -> None:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        try:
            _apply()
            get_settings.cache_clear()
            new_settings = get_settings()
        except Exception as exc:
            _restore()
            get_settings.cache_clear()
            restored_settings = get_settings()
            request.app.state.settings = restored_settings
            state.add_message(
                AdminMessage(
                    timestamp=datetime.utcnow(),
                    category="settings",
                    success=False,
                    summary="環境変数の更新に失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        request.app.state.settings = new_settings
        configure_logging(new_settings.log_level)
        _reload_components(request.app, new_settings)
        state.add_message(
            AdminMessage(
                timestamp=datetime.utcnow(),
                category="settings",
                success=True,
                summary="環境変数を更新しました",
                detail="ワーカーとクライアントを再初期化しました",
            )
        )
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


__all__ = ["AdminState", "register_admin_routes"]
