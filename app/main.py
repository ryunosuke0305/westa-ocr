"""FastAPI application entry-point."""

from __future__ import annotations

import asyncio
import json
import queue
import secrets
import sqlite3
import string
from contextlib import ExitStack
from datetime import datetime
from typing import Dict

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .admin import AdminState, register_admin_routes
from .auth import verify_request
from .file_fetcher import FileFetcher
from .gemini import GeminiClient
from .logging_config import configure_logging, get_logger
from .models import ErrorResponse, JobDetail, JobRequest, JobResponse, JobStatus
from .repository import JobRepository
from .settings import ENV_FILE_PATH, Settings, get_settings, load_env_file
from .webhook import WebhookDispatcher
from .worker import JobWorker

LOGGER = get_logger(__name__)

INITIALIZATION_EXEMPT_PATHS = frozenset({"/", "/healthz"})


def _generate_job_id() -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"job_{now}_{suffix}"


def _error_response(code: str, message: str, status_code: int) -> Response:
    return Response(
        status_code=status_code,
        content=json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False),
        media_type="application/json",
    )


def create_application() -> FastAPI:
    app = FastAPI(title="westa-ocr relay", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.admin_state = AdminState()
    app.state.initialization_complete = None
    app.state.initialization_error = None

    @app.middleware("http")
    async def wait_for_initialization(request: Request, call_next):  # pragma: no cover - middleware
        if request.url.path not in INITIALIZATION_EXEMPT_PATHS:
            event = request.app.state.initialization_complete
            if event is None:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": "initializing"},
                )
            await event.wait()
            if request.app.state.initialization_error is not None:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": "initialization_failed"},
                )
        return await call_next(request)

    register_routes(app)
    register_admin_routes(app)
    register_events(app)

    return app


def register_routes(app: FastAPI) -> None:
    @app.get("/", tags=["health"])
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz", tags=["health"])
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/jobs",
        response_model=JobResponse,
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_409_CONFLICT: {"model": ErrorResponse},
        },
        tags=["jobs"],
    )
    async def create_job(
        payload: JobRequest,
        request: Request,
        _: None = Depends(verify_request),
    ) -> JobResponse | Response:
        repository: JobRepository = request.app.state.repository
        job_queue: queue.Queue[str] = request.app.state.job_queue
        admin_state: AdminState = request.app.state.admin_state
        settings: Settings = request.app.state.settings

        idempotency_key = payload.idempotency_key or payload.order_id
        request_snapshot = json.dumps(
            payload.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        existing = repository.find_job_by_idempotency(idempotency_key)
        if existing:
            status_value = existing["status"]
            try:
                job_status = JobStatus(status_value)
            except ValueError:  # pragma: no cover - defensive guard
                job_status = JobStatus.ERROR
            LOGGER.info(
                "Job already registered for idempotency key",
                extra={"jobId": existing["job_id"], "idempotencyKey": idempotency_key},
            )
            admin_state.add_relay_request_log(
                job_id=existing["job_id"],
                order_id=payload.order_id,
                idempotency_key=idempotency_key,
                status=job_status.value,
                payload=request_snapshot,
                message="既存のジョブを返却しました",
            )
            return JobResponse(
                job_id=existing["job_id"],
                correlation_id=existing["order_id"],
                status=job_status,
            )

        job_id = _generate_job_id()
        LOGGER.info("Registering new job", extra={"jobId": job_id, "orderId": payload.order_id})
        webhook_payload = payload.webhook.model_dump(mode="json")
        webhook_override_applied = False
        if settings.webhook_url:
            webhook_payload["url"] = settings.webhook_url
            webhook_override_applied = True
            LOGGER.info(
                "Overriding webhook URL from settings",
                extra={
                    "jobId": job_id,
                    "orderId": payload.order_id,
                    "webhookUrl": settings.webhook_url,
                },
            )

        try:
            repository.insert_job(
                job_id=job_id,
                order_id=payload.order_id,
                file_id=payload.file_id,
                prompt=payload.prompt,
                pattern=payload.pattern,
                masters=payload.masters.model_dump(mode="json", by_alias=True),
                webhook=webhook_payload,
                gemini=(
                    payload.gemini.model_dump(mode="json", by_alias=True, exclude_none=True)
                    if payload.gemini
                    else None
                ),
                options=(
                    payload.options.model_dump(mode="json", by_alias=True, exclude_none=True)
                    if payload.options
                    else None
                ),
                idempotency_key=idempotency_key,
            )
        except sqlite3.IntegrityError as exc:
            LOGGER.exception("Integrity error while inserting job", extra={"jobId": job_id})
            admin_state.add_relay_request_log(
                job_id=job_id,
                order_id=payload.order_id,
                idempotency_key=idempotency_key,
                status="ERROR",
                payload=request_snapshot,
                message=f"Integrity error: {exc}",
            )
            return _error_response("ALREADY_EXISTS", "Job already exists", status.HTTP_409_CONFLICT)

        repository.mark_enqueued(job_id)
        job_queue.put(job_id)
        message = "新規ジョブを登録しました"
        if webhook_override_applied:
            message += "（Webhook URL を設定値で上書きしました）"
        admin_state.add_relay_request_log(
            job_id=job_id,
            order_id=payload.order_id,
            idempotency_key=idempotency_key,
            status=JobStatus.RECEIVED.value,
            payload=request_snapshot,
            message=message,
        )
        return JobResponse(job_id=job_id, correlation_id=payload.order_id, status=JobStatus.RECEIVED)

    @app.get(
        "/jobs/{job_id}",
        response_model=JobDetail,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        },
        tags=["jobs"],
    )
    async def get_job(job_id: str, request: Request, _: None = Depends(verify_request)) -> JobDetail | Response:
        repository: JobRepository = request.app.state.repository
        detail = repository.get_job_detail(job_id)
        if detail is None:
            return _error_response("NOT_FOUND", "Job not found", status.HTTP_404_NOT_FOUND)
        return JobDetail(**detail)


def register_events(app: FastAPI) -> None:
    @app.on_event("startup")
    async def on_startup() -> None:  # pragma: no cover - lifecycle hook
        event = asyncio.Event()
        app.state.initialization_complete = event

        async def initialize() -> None:
            try:
                loaded = load_env_file()
                if loaded:
                    LOGGER.info(
                        "Loaded environment overrides from file",
                        extra={"path": str(ENV_FILE_PATH), "keyCount": len(loaded)},
                    )
                components, pending = await asyncio.to_thread(
                    _build_components, app.state.admin_state
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("Failed to initialize application", exc_info=exc)
                app.state.initialization_error = exc
            else:
                app.state.settings = components["settings"]
                app.state.repository = components["repository"]
                app.state.job_queue = components["job_queue"]
                app.state.file_fetcher = components["file_fetcher"]
                app.state.gemini_client = components["gemini_client"]
                app.state.webhook_dispatcher = components["webhook_dispatcher"]
                app.state.workers = components["workers"]
                for worker in app.state.workers:
                    worker.start()
                LOGGER.info(
                    "Startup complete",
                    extra={"pendingJobs": pending, "workerCount": len(app.state.workers)},
                )
            finally:
                event.set()

        asyncio.create_task(initialize())

    @app.on_event("shutdown")
    async def on_shutdown() -> None:  # pragma: no cover - lifecycle hook
        event = getattr(app.state, "initialization_complete", None)
        if event is not None:
            await event.wait()
        if getattr(app.state, "initialization_error", None) is not None:
            return

        workers: list[JobWorker] = getattr(app.state, "workers", [])
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=10)

        app.state.webhook_dispatcher.close()
        app.state.file_fetcher.close()
        app.state.gemini_client.close()
        app.state.repository.close()


def _build_components(admin_state: AdminState) -> tuple[dict[str, object], int]:
    settings = get_settings()
    configure_logging(settings.log_level)

    job_queue: "queue.Queue[str]" = queue.Queue()
    with ExitStack() as stack:
        repository = JobRepository(settings.sqlite_path)
        stack.callback(repository.close)

        file_fetcher = FileFetcher(
            settings.request_timeout,
            drive_service_account_json=settings.drive_service_account_json,
        )
        stack.callback(file_fetcher.close)

        gemini_client = GeminiClient(
            settings.gemini_api_key,
            default_model=settings.gemini_model,
            timeout=settings.request_timeout,
        )
        stack.callback(gemini_client.close)

        webhook_dispatcher = WebhookDispatcher(settings.webhook_timeout)
        stack.callback(webhook_dispatcher.close)

        workers = [
            JobWorker(
                repository=repository,
                job_queue=job_queue,
                file_fetcher=file_fetcher,
                gemini_client=gemini_client,
                webhook_dispatcher=webhook_dispatcher,
                idle_sleep=settings.worker_idle_sleep,
                page_concurrency=settings.worker_page_concurrency,
                admin_state=admin_state,
                worker_number=index + 1,
                name=f"JobWorker-{index + 1}",
            )
            for index in range(settings.worker_count)
        ]

        pending_jobs = repository.list_pending_jobs()
        for job_id in pending_jobs:
            repository.mark_enqueued(job_id)
            job_queue.put(job_id)

        stack.pop_all()

    return (
        {
            "settings": settings,
            "repository": repository,
            "job_queue": job_queue,
            "file_fetcher": file_fetcher,
            "gemini_client": gemini_client,
            "webhook_dispatcher": webhook_dispatcher,
            "workers": workers,
        },
        len(pending_jobs),
    )


app = create_application()
