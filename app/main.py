"""FastAPI application entry-point."""

from __future__ import annotations

import json
import queue
import sqlite3
import secrets
import string
from datetime import datetime
from typing import Dict

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .auth import verify_request
from .file_fetcher import FileFetcher
from .gemini import GeminiClient
from .logging_config import configure_logging, get_logger
from .models import ErrorResponse, JobDetail, JobRequest, JobResponse, JobStatus
from .repository import JobRepository
from .settings import get_settings
from .webhook import WebhookDispatcher
from .worker import JobWorker

LOGGER = get_logger(__name__)


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
    settings = get_settings()
    configure_logging(settings.log_level)

    repository = JobRepository(settings.sqlite_path)
    job_queue: queue.Queue[str] = queue.Queue()
    file_fetcher = FileFetcher(settings.request_timeout)
    gemini_client = GeminiClient(
        settings.gemini_api_key,
        default_model=settings.gemini_model,
        timeout=settings.request_timeout,
    )
    webhook_dispatcher = WebhookDispatcher(settings.webhook_timeout)
    worker = JobWorker(
        repository=repository,
        job_queue=job_queue,
        file_fetcher=file_fetcher,
        gemini_client=gemini_client,
        webhook_dispatcher=webhook_dispatcher,
        idle_sleep=settings.worker_idle_sleep,
    )

    app = FastAPI(title="westa-ocr relay", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.repository = repository
    app.state.job_queue = job_queue
    app.state.file_fetcher = file_fetcher
    app.state.gemini_client = gemini_client
    app.state.webhook_dispatcher = webhook_dispatcher
    app.state.worker = worker

    register_routes(app)
    register_events(app)

    return app


def register_routes(app: FastAPI) -> None:
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

        idempotency_key = payload.idempotency_key or payload.order_id
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
            return JobResponse(
                job_id=existing["job_id"],
                correlation_id=existing["order_id"],
                status=job_status,
            )

        job_id = _generate_job_id()
        LOGGER.info("Registering new job", extra={"jobId": job_id, "orderId": payload.order_id})
        try:
            repository.insert_job(
                job_id=job_id,
                order_id=payload.order_id,
                file_id=payload.file_id,
                prompt=payload.prompt,
                pattern=payload.pattern,
                masters=payload.masters.model_dump(by_alias=True),
                webhook=payload.webhook.model_dump(),
                gemini=payload.gemini.model_dump(by_alias=True, exclude_none=True) if payload.gemini else None,
                options=payload.options.model_dump(by_alias=True, exclude_none=True) if payload.options else None,
                idempotency_key=idempotency_key,
            )
        except sqlite3.IntegrityError as exc:
            LOGGER.exception("Integrity error while inserting job", extra={"jobId": job_id})
            return _error_response("ALREADY_EXISTS", "Job already exists", status.HTTP_409_CONFLICT)

        repository.mark_enqueued(job_id)
        job_queue.put(job_id)
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
        repository: JobRepository = app.state.repository
        job_queue: queue.Queue[str] = app.state.job_queue
        worker: JobWorker = app.state.worker
        pending = repository.list_pending_jobs()
        for job_id in pending:
            repository.mark_enqueued(job_id)
            job_queue.put(job_id)
        worker.start()
        LOGGER.info("Startup complete", extra={"pendingJobs": len(pending)})

    @app.on_event("shutdown")
    async def on_shutdown() -> None:  # pragma: no cover - lifecycle hook
        worker: JobWorker = app.state.worker
        worker.stop()
        worker.join(timeout=10)

        app.state.webhook_dispatcher.close()
        app.state.file_fetcher.close()
        app.state.gemini_client.close()
        app.state.repository.close()


app = create_application()
