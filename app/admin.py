"""Admin dashboard endpoints for manual operations and configuration tweaks."""

from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from zoneinfo import ZoneInfo

from .file_fetcher import FileFetcher
from .gemini import GeminiClient
from .logging_config import configure_logging, get_logger
from .models import JobStatus
from .settings import Settings, get_settings, save_env_file
from .webhook import WebhookDispatcher
from .worker import JobWorker

LOGGER = get_logger(__name__)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

_JST = ZoneInfo("Asia/Tokyo")

DEFAULT_GEMINI_TEMPERATURE = 0.1
DEFAULT_GEMINI_TOP_P = 0.9


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
    token: Optional[str]


@dataclass(slots=True)
class RelayRequestLogEntry:
    """Incoming request from ProcessOrder_test_relay."""

    timestamp: datetime
    job_id: str
    order_id: str
    idempotency_key: str
    status: str
    payload: str
    message: Optional[str]


@dataclass(slots=True)
class RelayWebhookLogEntry:
    """Webhook payload dispatched during automatic processing."""

    timestamp: datetime
    job_id: str
    order_id: str
    event: str
    url: str
    payload: str
    success: bool
    status_code: Optional[int]
    response_text: Optional[str]
    error: Optional[str]
    token: Optional[str]


@dataclass(slots=True)
class AdminMessage:
    """Feedback banner to display on the dashboard."""

    timestamp: datetime
    category: str
    success: bool
    summary: str
    detail: Optional[str]


def _mask_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    if len(token) <= 4:
        return "*" * len(token)
    return f"{token[:4]}…{token[-2:]}"


class AdminState:
    """Container keeping the latest admin operations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._webhook_logs: List[WebhookLogEntry] = []
        self._relay_request_logs: List[RelayRequestLogEntry] = []
        self._relay_webhook_logs: List[RelayWebhookLogEntry] = []
        self._messages: List[AdminMessage] = []
        self._cancellation_requests: Set[str] = set()

    @property
    def webhook_logs(self) -> List[WebhookLogEntry]:
        with self._lock:
            return list(self._webhook_logs)

    @property
    def relay_request_logs(self) -> List[RelayRequestLogEntry]:
        with self._lock:
            return list(self._relay_request_logs)

    @property
    def relay_webhook_logs(self) -> List[RelayWebhookLogEntry]:
        with self._lock:
            return list(self._relay_webhook_logs)

    @property
    def messages(self) -> List[AdminMessage]:
        with self._lock:
            return list(self._messages)

    def add_webhook_log(self, entry: WebhookLogEntry) -> None:
        entry = replace(entry, token=_mask_token(entry.token))
        with self._lock:
            self._webhook_logs.insert(0, entry)
            del self._webhook_logs[10:]

    def add_relay_request_log(
        self,
        *,
        job_id: str,
        order_id: str,
        idempotency_key: str,
        status: str,
        payload: str,
        message: Optional[str],
    ) -> None:
        entry = RelayRequestLogEntry(
            timestamp=datetime.now(timezone.utc),
            job_id=job_id,
            order_id=order_id,
            idempotency_key=idempotency_key,
            status=status,
            payload=payload,
            message=message,
        )
        with self._lock:
            self._relay_request_logs.insert(0, entry)
            del self._relay_request_logs[50:]

    def add_relay_webhook_log(
        self,
        *,
        job_id: str,
        order_id: str,
        event: str,
        url: str,
        payload: str,
        success: bool,
        status_code: Optional[int],
        response_text: Optional[str],
        error: Optional[str],
        token: Optional[str],
    ) -> None:
        entry = RelayWebhookLogEntry(
            timestamp=datetime.now(timezone.utc),
            job_id=job_id,
            order_id=order_id,
            event=event,
            url=url,
            payload=payload,
            success=success,
            status_code=status_code,
            response_text=response_text,
            error=error,
            token=_mask_token(token),
        )
        with self._lock:
            self._relay_webhook_logs.insert(0, entry)
            del self._relay_webhook_logs[50:]

    def add_message(self, entry: AdminMessage) -> None:
        with self._lock:
            self._messages.insert(0, entry)
            del self._messages[10:]

    def request_job_cancellation(self, job_id: str) -> None:
        with self._lock:
            self._cancellation_requests.add(job_id)

    def clear_job_cancellation(self, job_id: str) -> None:
        with self._lock:
            self._cancellation_requests.discard(job_id)

    def is_cancellation_requested(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancellation_requests


CONFIG_FIELDS: List[Dict[str, str]] = [
    {"env": "RELAY_TOKEN", "label": "Relay Token", "placeholder": "必須"},
    {"env": "DATA_DIR", "label": "Data Directory", "placeholder": "/data"},
    {"env": "SQLITE_PATH", "label": "SQLite Path", "placeholder": "<DATA_DIR>/relay.db"},
    {"env": "TMP_DIR", "label": "Temporary Directory", "placeholder": "<DATA_DIR>/tmp"},
    {"env": "WORKER_IDLE_SLEEP", "label": "Worker Idle Sleep", "placeholder": "秒数 (例: 1.0)"},
    {"env": "WORKER_COUNT", "label": "Worker Count", "placeholder": "スレッド数 (例: 10)"},
    {"env": "GEMINI_API_KEY", "label": "Gemini API Key", "placeholder": "Google AI Studio の API キー"},
    {"env": "GEMINI_MODEL", "label": "Gemini Model", "placeholder": "例: gemini-2.5-flash"},
    {
        "env": "WEBHOOK_URL",
        "label": "Webhook URL Override",
        "placeholder": "例: https://example.com/webhook",
    },
    {"env": "WEBHOOK_TIMEOUT", "label": "Webhook Timeout", "placeholder": "秒数 (例: 30)"},
    {"env": "REQUEST_TIMEOUT", "label": "Request Timeout", "placeholder": "秒数 (例: 600)"},
    {"env": "LOG_LEVEL", "label": "Log Level", "placeholder": "例: INFO / DEBUG"},
]


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(_JST).strftime("%Y-%m-%d %H:%M:%S")


def _calculate_total_pages(total_count: int, page_size: int) -> int:
    if total_count <= 0:
        return 0
    return (total_count + page_size - 1) // page_size


def _build_dashboard_payload(
    settings: Settings,
    state: AdminState,
    gemini_logs: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    *,
    job_total_count: int,
    job_page: int,
    job_page_size: int,
    job_total_pages: int,
) -> Dict[str, Any]:
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

    gemini_log_payloads = []
    for entry in gemini_logs:
        gemini_log_payloads.append(
            {
                "id": entry["id"],
                "timestamp": _format_datetime(entry["timestamp"]),
                "source": entry["source"],
                "sourceLabel": entry["source"].upper(),
                "promptPreview": entry["prompt_preview"],
                "model": entry["model"],
                "mimeType": entry["mime_type"],
                "success": entry["success"],
                "responseText": entry["response_text"],
                "meta": entry["meta"],
                "error": entry["error"],
                "request": entry["request"],
            }
        )

    webhook_logs = [
        {
            "timestamp": _format_datetime(entry.timestamp),
            "url": entry.url,
            "payload": entry.payload,
            "success": entry.success,
            "statusCode": entry.status_code,
            "responseText": entry.response_text,
            "error": entry.error,
            "authorization": entry.token,
        }
        for entry in state.webhook_logs
    ]

    relay_request_logs = [
        {
            "timestamp": _format_datetime(entry.timestamp),
            "jobId": entry.job_id,
            "orderId": entry.order_id,
            "idempotencyKey": entry.idempotency_key,
            "status": entry.status,
            "message": entry.message,
            "payload": entry.payload,
        }
        for entry in state.relay_request_logs
    ]

    relay_webhook_logs = [
        {
            "timestamp": _format_datetime(entry.timestamp),
            "jobId": entry.job_id,
            "orderId": entry.order_id,
            "event": entry.event,
            "url": entry.url,
            "payload": entry.payload,
            "success": entry.success,
            "statusCode": entry.status_code,
            "responseText": entry.response_text,
            "error": entry.error,
            "authorization": entry.token,
        }
        for entry in state.relay_webhook_logs
    ]

    cancellable_statuses = {
        JobStatus.RECEIVED.value,
        JobStatus.ENQUEUED.value,
        JobStatus.PROCESSING.value,
    }
    job_rows = []
    for job in jobs:
        cancellation_requested = state.is_cancellation_requested(job["job_id"])
        job_rows.append(
            {
                "jobId": job["job_id"],
                "orderId": job["order_id"],
                "status": job["status"],
                "statusLabel": job["status"],
                "workerName": job["worker_name"],
                "totalPages": job["total_pages"],
                "processedPages": job["processed_pages"],
                "skippedPages": job["skipped_pages"],
                "lastError": job["last_error"],
                "createdAt": _format_datetime(job["created_at"])
                if job["created_at"]
                else "-",
                "updatedAt": _format_datetime(job["updated_at"])
                if job["updated_at"]
                else "-",
                "canCancel": job["status"] in cancellable_statuses and not cancellation_requested,
                "cancellationRequested": cancellation_requested,
            }
        )

    return {
        "configFields": config_fields,
        "messages": messages,
        "geminiLogs": gemini_log_payloads,
        "webhookLogs": webhook_logs,
        "relayRequestLogs": relay_request_logs,
        "relayWebhookLogs": relay_webhook_logs,
        "jobs": job_rows,
        "jobsPagination": {
            "page": job_page,
            "pageSize": job_page_size,
            "totalCount": job_total_count,
            "totalPages": job_total_pages,
            "hasPrev": job_total_pages > 0 and job_page > 1,
            "hasNext": job_total_pages > 0 and job_page < job_total_pages,
        },
        "defaults": {
            "masters": "{}",
            "webhookPayload": "{}",
            "webhookToken": "",
            "webhookUrl": settings.webhook_url or "",
            "geminiModel": settings.gemini_model,
            "geminiTemperature": DEFAULT_GEMINI_TEMPERATURE,
            "geminiTopP": DEFAULT_GEMINI_TOP_P,
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


def _parse_page_number(value: Optional[str], *, default: int = 1) -> int:
    if value is None:
        return default
    try:
        page = int(value)
    except (TypeError, ValueError):
        return default
    if page < 1:
        return default
    return page


def _build_admin_gemini_request_snapshot(
    *,
    prompt: str,
    filename: str,
    mime_type: str,
    page_bytes: bytes,
    masters: Dict[str, Any],
    model: str,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_output_tokens: Optional[int],
) -> Dict[str, Any]:
    preview_limit = 120
    sample_bytes = page_bytes[:preview_limit]
    input_snapshot: Dict[str, Any] = {
        "filename": filename,
        "mimeType": mime_type,
        "sizeBytes": len(page_bytes),
    }
    if sample_bytes:
        input_snapshot["base64Sample"] = base64.b64encode(sample_bytes).decode("ascii")
    input_snapshot["isBase64Truncated"] = len(page_bytes) > preview_limit
    parameters = {
        "model": model,
        "temperature": temperature,
        "topP": top_p,
        "topK": top_k,
        "maxOutputTokens": max_output_tokens,
    }
    return {
        "prompt": prompt,
        "promptLength": len(prompt),
        "masters": masters,
        "mastersKeys": sorted(masters.keys()),
        "input": input_snapshot,
        "parameters": parameters,
    }


def _reload_components(app: FastAPI, settings: Settings) -> None:
    LOGGER.info("Reloading application components via admin console")
    workers: List[JobWorker] = getattr(app.state, "workers", [])
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=10)

    file_fetcher: FileFetcher = app.state.file_fetcher
    gemini_client: GeminiClient = app.state.gemini_client
    webhook_dispatcher: WebhookDispatcher = app.state.webhook_dispatcher

    file_fetcher.close()
    gemini_client.close()
    webhook_dispatcher.close()

    new_file_fetcher = FileFetcher(
        settings.request_timeout,
        drive_service_account_json=settings.drive_service_account_json,
    )
    new_gemini = GeminiClient(
        settings.gemini_api_key,
        default_model=settings.gemini_model,
        timeout=settings.request_timeout,
    )
    new_webhook = WebhookDispatcher(settings.webhook_timeout)
    new_workers = [
        JobWorker(
            repository=app.state.repository,
            job_queue=app.state.job_queue,
            file_fetcher=new_file_fetcher,
            gemini_client=new_gemini,
            webhook_dispatcher=new_webhook,
            idle_sleep=settings.worker_idle_sleep,
            page_concurrency=settings.worker_page_concurrency,
            admin_state=app.state.admin_state,
            worker_number=index + 1,
            name=f"JobWorker-{index + 1}",
        )
        for index in range(settings.worker_count)
    ]
    for worker in new_workers:
        worker.start()

    app.state.file_fetcher = new_file_fetcher
    app.state.gemini_client = new_gemini
    app.state.webhook_dispatcher = new_webhook
    app.state.workers = new_workers


def register_admin_routes(app: FastAPI) -> None:
    state: AdminState = app.state.admin_state

    @app.get("/admin")
    async def admin_dashboard(request: Request):
        settings: Settings = request.app.state.settings
        repository = request.app.state.repository
        gemini_logs = repository.list_gemini_logs(limit=10)
        job_page_size = 20
        requested_page = _parse_page_number(request.query_params.get("job_page"))
        total_jobs = repository.count_jobs()
        total_pages = _calculate_total_pages(total_jobs, job_page_size)
        if total_pages == 0:
            job_page = 1
        else:
            job_page = min(requested_page, total_pages)
        offset = (job_page - 1) * job_page_size if total_jobs else 0
        jobs = repository.list_recent_jobs(limit=job_page_size, offset=offset)
        payload = _build_dashboard_payload(
            settings,
            state,
            gemini_logs,
            jobs,
            job_total_count=total_jobs,
            job_page=job_page,
            job_page_size=job_page_size,
            job_total_pages=total_pages,
        )
        return templates.TemplateResponse("admin.html", {"request": request, "dashboard": payload})

    @app.post("/admin/gemini")
    async def send_gemini(
        request: Request,
        prompt: str = Form(...),
        masters: str = Form("{}"),
        model: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        top_k: str = Form(""),
        max_output_tokens: str = Form(""),
        pdf_file: UploadFile = File(...),
    ) -> RedirectResponse:
        prompt = prompt.strip()
        if not prompt:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="gemini",
                    success=False,
                    summary="プロンプトを入力してください",
                    detail=None,
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        if pdf_file is None:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="gemini",
                    success=False,
                    summary="PDF ファイルをアップロードしてください",
                    detail=None,
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        try:
            page_bytes = await pdf_file.read()
        except Exception as exc:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="gemini",
                    success=False,
                    summary="PDF ファイルの読み込みに失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        if not page_bytes:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="gemini",
                    success=False,
                    summary="PDF ファイルが空です",
                    detail="ファイル内容を確認してください。",
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
                    timestamp=datetime.now(timezone.utc),
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
                    timestamp=datetime.now(timezone.utc),
                    category="gemini",
                    success=False,
                    summary="数値パラメータの変換に失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        if temperature_value is None:
            temperature_value = DEFAULT_GEMINI_TEMPERATURE
        if top_p_value is None:
            top_p_value = DEFAULT_GEMINI_TOP_P

        gemini_client: GeminiClient = request.app.state.gemini_client
        repository = request.app.state.repository
        settings: Settings = request.app.state.settings
        effective_model = model or settings.gemini_model
        effective_mime_type = pdf_file.content_type or "application/pdf"
        filename = pdf_file.filename or "uploaded.pdf"
        request_snapshot = _build_admin_gemini_request_snapshot(
            prompt=prompt,
            filename=filename,
            mime_type=effective_mime_type,
            page_bytes=page_bytes,
            masters=masters_payload,
            model=effective_model,
            temperature=temperature_value,
            top_p=top_p_value,
            top_k=top_k_value,
            max_output_tokens=max_tokens_value,
        )
        try:
            result = gemini_client.generate(
                model=effective_model,
                prompt=prompt,
                page_bytes=page_bytes,
                mime_type=effective_mime_type,
                masters=masters_payload,
                temperature=temperature_value,
                top_p=top_p_value,
                top_k=top_k_value,
                max_output_tokens=max_tokens_value,
            )
        except Exception as exc:
            repository.record_gemini_log(
                source="admin",
                prompt=prompt,
                model=effective_model,
                mime_type=effective_mime_type,
                request=request_snapshot,
                success=False,
                response_text=None,
                meta=None,
                error=str(exc),
            )
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="gemini",
                    success=False,
                    summary="Gemini へのリクエストでエラーが発生しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        repository.record_gemini_log(
            source="admin",
            prompt=prompt,
            model=effective_model,
            mime_type=effective_mime_type,
            request=request_snapshot,
            success=True,
            response_text=result.text,
            meta=result.meta,
            error=None,
        )
        state.add_message(
            AdminMessage(
                timestamp=datetime.now(timezone.utc),
                category="gemini",
                success=True,
                summary="Gemini へのリクエストが完了しました",
                detail=f"モデル: {effective_model}",
            )
        )
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/admin/jobs/cancel")
    async def cancel_job(request: Request, job_id: str = Form(...)) -> RedirectResponse:
        repository = request.app.state.repository
        job = repository.get_job(job_id)
        if job is None:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="jobs",
                    success=False,
                    summary="ジョブが見つかりません",
                    detail=f"指定されたジョブ ID {job_id} は存在しません。",
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        try:
            status_value = JobStatus(job["status"])
        except ValueError:  # pragma: no cover - defensive guard
            status_value = JobStatus.ERROR

        redirect = RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        cancellable = {
            JobStatus.RECEIVED,
            JobStatus.ENQUEUED,
            JobStatus.PROCESSING,
        }
        if status_value not in cancellable:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="jobs",
                    success=False,
                    summary="この状態では中断できません",
                    detail=f"現在のステータス: {status_value.value}",
                )
            )
            return redirect

        LOGGER.info(
            "Cancellation requested from admin console",
            extra={"jobId": job_id, "orderId": job["order_id"], "status": status_value.value},
        )
        repository.update_job_status(job_id, JobStatus.CANCELLED, "Cancelled via admin console")
        state.request_job_cancellation(job_id)
        detail = "処理中のジョブの場合、完全に停止するまで時間がかかることがあります。"
        state.add_message(
            AdminMessage(
                timestamp=datetime.now(timezone.utc),
                category="jobs",
                success=True,
                summary=f"ジョブ {job_id} の中断要求を受け付けました",
                detail=detail,
            )
        )
        return redirect

    @app.post("/admin/webhook")
    async def send_webhook(
        request: Request,
        url: str = Form(...),
        payload: str = Form(...),
        token: str = Form(""),
    ) -> RedirectResponse:
        try:
            payload_dict = json.loads(payload)
            if not isinstance(payload_dict, dict):
                raise ValueError("payload must be a JSON object")
        except Exception as exc:
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="webhook",
                    success=False,
                    summary="JSON ペイロードの解析に失敗しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        dispatcher: WebhookDispatcher = request.app.state.webhook_dispatcher
        auth_token = token.strip() or None
        try:
            response = dispatcher.send(url, payload_dict, token=auth_token)
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
                    timestamp=datetime.now(timezone.utc),
                    url=url,
                    payload=payload,
                    success=False,
                    status_code=status_code,
                    response_text=response_text,
                    error=str(exc),
                    token=auth_token,
                )
            )
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
                    category="webhook",
                    success=False,
                    summary="Webhook の送信でエラーが発生しました",
                    detail=str(exc),
                )
            )
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)

        state.add_webhook_log(
            WebhookLogEntry(
                timestamp=datetime.now(timezone.utc),
                url=url,
                payload=json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":")),
                success=True,
                status_code=status_code,
                response_text=response_text,
                error=None,
                token=auth_token,
            )
        )
        state.add_message(
            AdminMessage(
                timestamp=datetime.now(timezone.utc),
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
                    timestamp=datetime.now(timezone.utc),
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
            env_snapshot = {field["env"]: os.getenv(field["env"]) for field in CONFIG_FIELDS}
            save_env_file(env_snapshot)
        except Exception as exc:
            _restore()
            get_settings.cache_clear()
            restored_settings = get_settings()
            request.app.state.settings = restored_settings
            state.add_message(
                AdminMessage(
                    timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
                category="settings",
                success=True,
                summary="環境変数を更新しました",
                detail="ワーカーとクライアントを再初期化しました",
            )
        )
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


__all__ = ["AdminState", "register_admin_routes"]
