# FastAPI アプリケーションのエントリーポイント。

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
from .worker import JobWorker, JobTask

LOGGER = get_logger(__name__)

INITIALIZATION_EXEMPT_PATHS = frozenset({"/", "/healthz"})


# タイムスタンプとランダムサフィックスを組み合わせたジョブ ID を生成する。
# 返り値:
#     str: 重複しにくいジョブ ID。
def _generate_job_id() -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"job_{now}_{suffix}"


# 共通形式のエラーレスポンスを組み立てて返す。
# 引数:
#     code (str): エラー識別コード。
#     message (str): クライアント向け説明メッセージ。
#     status_code (int): HTTP ステータスコード。
# 返り値:
#     Response: エラー内容を含む JSON レスポンス。
def _error_response(code: str, message: str, status_code: int) -> Response:
    return Response(
        status_code=status_code,
        content=json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False),
        media_type="application/json",
    )


# ルーティングやイベント登録を行った FastAPI アプリケーションを生成する。
# 返り値:
#     FastAPI: 初期設定済みのアプリケーションインスタンス。
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
    # 初期化が完了するまでヘルスチェック以外のリクエストを待機させるミドルウェア。
    # 引数:
    #     request (Request): 受信した HTTP リクエスト。
    #     call_next: 後続処理を呼び出す FastAPI のコールバック。
    # 返り値:
    #     Response: 初期化状況に応じた JSON レスポンスまたは本来の応答。
    async def wait_for_initialization(request: Request, call_next):  # pragma: no cover - middleware
        if request.url.path not in INITIALIZATION_EXEMPT_PATHS:
            event = request.app.state.initialization_complete
            if event is None:
                # 初期化タスクが未登録の場合は即座に 503 を返し、リトライを促す。
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": "initializing"},
                )
            await event.wait()
            if request.app.state.initialization_error is not None:
                # 初期化失敗時は復旧するまで全てのリクエストをエラーで返す。
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": "initialization_failed"},
                )
        return await call_next(request)

    register_routes(app)
    register_admin_routes(app)
    register_events(app)

    return app


# ヘルスチェックおよびジョブ管理のルートを FastAPI アプリに登録する。
# 引数:
#     app (FastAPI): ルートを追加する対象アプリケーション。
def register_routes(app: FastAPI) -> None:

    @app.get("/", tags=["health"])
    # 稼働状況を確認するための固定レスポンスを返す。
    # 返り値:
    #     Dict[str, str]: 常に {"status": "ok"}。
    async def health() -> Dict[str, str]:

        return {"status": "ok"}

    @app.get("/healthz", tags=["health"])
    # 監視用に分離したヘルスチェックエンドポイント。
    # 返り値:
    #     Dict[str, str]: 常に {"status": "ok"}。
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
    # OCR ジョブを登録しキューへ投入する。
    # 引数:
    #     payload (JobRequest): リクエストボディのジョブ情報。
    #     request (Request): アプリケーションステートへアクセスするためのリクエスト。
    #     _ (None): verify_request の戻り値。副作用のみが目的で未使用。
    # 返り値:
    #     JobResponse | Response: 新規ジョブ情報またはエラーレスポンス。
    async def create_job(
        payload: JobRequest,
        request: Request,
        _: None = Depends(verify_request),
    ) -> JobResponse | Response:
        repository: JobRepository = request.app.state.repository
        job_queue: queue.Queue[object] = request.app.state.job_queue
        admin_state: AdminState = request.app.state.admin_state
        settings: Settings = request.app.state.settings

        # Idempotency key はユーザーから明示されない場合でも order_id で補完する。
        idempotency_key = payload.idempotency_key or payload.order_id
        request_snapshot = json.dumps(
            payload.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        # 既存ジョブがある場合は重複登録を避け、最新の状態をそのまま返す。
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
        if settings.webhook_url and not webhook_payload.get("url"):
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
            # ここでは payload 全体を永続化し、リトライ時に参照できる入力データを保持する。
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

        # ここまで到達したらキューへ投入し、ワーカーにファイル取得とページ分割を委任する。
        repository.mark_enqueued(job_id)
        job_queue.put(JobTask(job_id=job_id))
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
    # ジョブの詳細情報を取得して返す。
    # 引数:
    #     job_id (str): 取得対象のジョブ ID。
    #     request (Request): リポジトリへアクセスするためのリクエスト。
    #     _ (None): verify_request の戻り値。副作用のみが目的で未使用。
    # 返り値:
    #     JobDetail | Response: 詳細情報または 404 レスポンス。
    async def get_job(job_id: str, request: Request, _: None = Depends(verify_request)) -> JobDetail | Response:
        repository: JobRepository = request.app.state.repository
        detail = repository.get_job_detail(job_id)
        if detail is None:
            return _error_response("NOT_FOUND", "Job not found", status.HTTP_404_NOT_FOUND)
        return JobDetail(**detail)


# 起動時と終了時のイベントハンドラーを FastAPI に登録する。
# 引数:
#     app (FastAPI): イベントを設定する対象アプリケーション。
def register_events(app: FastAPI) -> None:

    @app.on_event("startup")
    # アプリ起動時にバックグラウンド初期化を開始する。
    async def on_startup() -> None:  # pragma: no cover - lifecycle hook

        event = asyncio.Event()
        app.state.initialization_complete = event

        # 設定読み込みと依存コンポーネントの構築を行う初期化タスク。
        async def initialize() -> None:

            try:
                # .env ファイルが存在する場合は環境変数を上書きする。
                loaded = load_env_file()
                if loaded:
                    LOGGER.info(
                        "Loaded environment overrides from file",
                        extra={"path": str(ENV_FILE_PATH), "keyCount": len(loaded)},
                    )
                # ブロッキング I/O を伴う初期化はスレッドプールで実施する。
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
                # 成否にかかわらず初期化待機中のリクエストを解放する。
                event.set()

        asyncio.create_task(initialize())

    @app.on_event("shutdown")
    # アプリ終了時にバックグラウンドワーカーと外部クライアントを片付ける。
    async def on_shutdown() -> None:  # pragma: no cover - lifecycle hook

        event = getattr(app.state, "initialization_complete", None)
        if event is not None:
            await event.wait()
        if getattr(app.state, "initialization_error", None) is not None:
            return

        workers: list[JobWorker] = getattr(app.state, "workers", [])
        # ワーカーを停止し、終了まで待機する。待機時間は過剰なブロックを避けるために制限。
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=10)

        app.state.webhook_dispatcher.close()
        app.state.file_fetcher.close()
        app.state.gemini_client.close()
        app.state.repository.close()


# 設定読み込み、依存コンポーネント生成、保留ジョブ再投入をまとめて行う。
# 引数:
#     admin_state (AdminState): 管理画面の共有状態。
# 返り値:
#     tuple[dict[str, object], int]: コンポーネント群と保留ジョブ件数。
def _build_components(admin_state: AdminState) -> tuple[dict[str, object], int]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # メインスレッドとワーカー間で共有するタスクキュー。
    job_queue: "queue.Queue[object]" = queue.Queue()
    with ExitStack() as stack:
        repository = JobRepository(settings.sqlite_path)
        stack.callback(repository.close)

        file_fetcher = FileFetcher(
            settings.request_timeout,
            drive_service_account_json=settings.drive_service_account_json,
        )
        stack.callback(file_fetcher.close)

        # Gemini クライアントは API 設定をラップしており、HTTP タイムアウトなどを統一管理する。
        gemini_client = GeminiClient(
            settings.gemini_api_key,
            default_model=settings.gemini_model,
            timeout=settings.request_timeout,
        )
        stack.callback(gemini_client.close)

        webhook_dispatcher = WebhookDispatcher(settings.webhook_timeout)
        stack.callback(webhook_dispatcher.close)

        # 指定されたワーカー数だけスレッドを生成し、同じ依存を共有させる。
        workers = [
            JobWorker(
                repository=repository,
                job_queue=job_queue,
                file_fetcher=file_fetcher,
                gemini_client=gemini_client,
                webhook_dispatcher=webhook_dispatcher,
                idle_sleep=settings.worker_idle_sleep,
                page_concurrency=settings.worker_count,
                admin_state=admin_state,
                worker_number=index + 1,
                name=f"JobWorker-{index + 1}",
            )
            for index in range(settings.worker_count)
        ]

        pending_jobs = repository.list_pending_jobs()
        # アプリ起動前に残っていたジョブを再キューイングすることで、処理漏れを防ぐ。
        for job_id in pending_jobs:
            repository.mark_enqueued(job_id)
            job_queue.put(JobTask(job_id=job_id))

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
