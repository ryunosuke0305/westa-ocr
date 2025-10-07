# バックグラウンドワーカーの実装。

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING, Union

from .file_fetcher import FileFetcher
from .gemini import GeminiClient
from .logging_config import get_logger
from .models import JobStatus
from .pdf_utils import PagePayload, split_pdf
from .repository import JobRepository
from .webhook import WebhookDispatcher

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .admin import AdminState

LOGGER = get_logger(__name__)

CANCEL_REASON = "Cancelled via admin console"


# ページ処理用タスク生成が必要なジョブを表すデータクラス。
@dataclass(frozen=True)
class JobTask:
    job_id: str


# 1 ページの処理リクエストを表すデータクラス。
@dataclass(frozen=True)
class PageTask:
    job_id: str
    page: PagePayload


WorkerTask = Union[JobTask, PageTask]


# ページ処理の進行状況を共有するためのコンテキストを表すクラス。
class PageJobContext:

    # ページ単位の処理を調停するための共有状態を初期化する。
    # 引数:
    #     job_row: データベースから取得したジョブ行。
    #     masters (Dict[str, str]): マスター情報の辞書。
    #     gemini_config (Dict): Gemini 呼び出しに使用する設定値。
    #     target_model (str): 利用する Gemini モデル名。
    #     prompt (str): 生成に用いるプロンプト。
    #     total_pages (int): ジョブ全体のページ数。
    #     max_parallel (int): 同時実行するページ数の上限。
    def __init__(
        self,
        *,
        job_row,
        masters: Dict[str, str],
        gemini_config: Dict,
        target_model: str,
        prompt: str,
        total_pages: int,
        max_parallel: int,
    ) -> None:
        self.job_row = job_row
        self.masters = masters
        self.gemini_config = gemini_config
        self.target_model = target_model
        self.prompt = prompt
        self.total_pages = total_pages
        self.pending_pages = total_pages
        self.processed_pages = 0
        self.page_errors: list[Dict[str, str]] = []
        self.cancelled = False
        self.completed = False
        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(max(1, max_parallel))


_JOB_CONTEXTS: dict[str, PageJobContext] = {}
_JOB_CONTEXTS_LOCK = threading.Lock()


# データベース上のジョブを非同期に処理するワーカースレッド。
class JobWorker(threading.Thread):

    # ワーカーが利用する依存を受け取り、スレッドを初期化する。
    # 引数:
    #     repository (JobRepository): ジョブ情報へアクセスするリポジトリ。
    #     job_queue (queue.Queue[WorkerTask]): ジョブやページタスクを受け取る共有キュー。
    #     file_fetcher (FileFetcher): ソースファイルを取得するフェッチャー。
    #     gemini_client (GeminiClient): Gemini API へのクライアント。
    #     webhook_dispatcher (WebhookDispatcher): Webhook 送信を担当するディスパッチャー。
    #     idle_sleep (float): キューが空の際に待機する秒数。
    #     admin_state (AdminState | None): 管理画面の共有状態。
    #     worker_number (int): ワーカーの通し番号。
    #     name (str | None): スレッド名。
    #     page_concurrency (int): ページ処理の同時実行数。
    def __init__(
        self,
        *,
        repository: JobRepository,
        job_queue: "queue.Queue[WorkerTask]",
        file_fetcher: FileFetcher,
        gemini_client: GeminiClient,
        webhook_dispatcher: WebhookDispatcher,
        idle_sleep: float,
        admin_state: "AdminState | None" = None,
        worker_number: int,
        name: str | None = None,
        page_concurrency: int = 1,
    ) -> None:
        worker_name = name or f"JobWorker-{worker_number}"
        super().__init__(daemon=True, name=worker_name)
        self._repository = repository
        self._queue = job_queue
        self._file_fetcher = file_fetcher
        self._gemini = gemini_client
        self._webhook = webhook_dispatcher
        self._idle_sleep = idle_sleep
        self._admin_state = admin_state
        self._stop_event = threading.Event()
        self.worker_number = worker_number
        if page_concurrency < 1:
            raise ValueError("page_concurrency must be >= 1")
        self._page_concurrency = page_concurrency

    # ワーカースレッドに停止を要求する。
    def stop(self) -> None:
        self._stop_event.set()

    # キューからタスクを取り出して処理するメインループ。
    def run(self) -> None:  # pragma: no cover - threading logic
        LOGGER.info(
            "Worker thread started",
            extra={"workerName": self.name, "workerNumber": self.worker_number},
        )
        # キューにタスクが入るまで待機し、ジョブ単位またはページ単位で処理を進める。
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=self._idle_sleep)
            except queue.Empty:
                continue
            try:
                if isinstance(task, JobTask):
                    self._process_job(task.job_id)
                elif isinstance(task, PageTask):
                    self._process_page_task(task)
                else:
                    LOGGER.error("Unknown task received by worker", extra={"task": task})
            except Exception as exc:  # pragma: no cover - unexpected errors
                job_id = getattr(task, "job_id", None)
                LOGGER.exception(
                    "Unexpected error during job processing",
                    extra={"jobId": job_id, "workerName": self.name},
                )
                if job_id:
                    self._repository.update_job_status(job_id, JobStatus.ERROR, str(exc))
            finally:
                self._queue.task_done()
        LOGGER.info(
            "Worker thread stopped",
            extra={"workerName": self.name, "workerNumber": self.worker_number},
        )

    # ジョブ全体の初期処理を行いページタスクを生成する。
    # 引数:
    #     job_id (str): 処理対象のジョブ ID。
    def _process_job(self, job_id: str) -> None:
        job_row = self._repository.get_job(job_id)
        if job_row is None:
            LOGGER.warning("Job not found when processing", extra={"jobId": job_id})
            return

        # 進行中のジョブのみを処理し、完了済みやエラー状態は無視する。
        if job_row["status"] not in (
            JobStatus.RECEIVED.value,
            JobStatus.ENQUEUED.value,
            JobStatus.PROCESSING.value,
        ):
            LOGGER.info(
                "Skipping job in terminal state",
                extra={"jobId": job_id, "status": job_row["status"]},
            )
            return

        if self._is_cancellation_requested(job_id):
            LOGGER.info("Skipping job that was cancelled before processing", extra={"jobId": job_id})
            self._repository.update_job_status(job_id, JobStatus.CANCELLED, CANCEL_REASON)
            self._clear_cancellation_flag(job_id)
            return

        self._repository.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            worker_name=self.name,
        )
        LOGGER.info("Processing job", extra={"jobId": job_id, "orderId": job_row["order_id"]})
        masters = json.loads(job_row["masters_json"])
        gemini_config = json.loads(job_row["gemini_json"]) if job_row["gemini_json"] else {}
        options = json.loads(job_row["options_json"]) if job_row["options_json"] else {}
        split_mode = (options.get("splitMode") or "pdf").lower()

        try:
            file_bytes = self._file_fetcher.fetch(job_row["file_id"])
        except Exception as exc:
            self._handle_initial_failure(
                job_row,
                job_id,
                log_message="Failed to fetch source file",
                error_prefix="file fetch failed",
                exc=exc,
            )
            self._clear_cancellation_flag(job_id)
            return

        try:
            if split_mode != "pdf":
                raise NotImplementedError("Only splitMode=pdf is currently supported")
            pages = split_pdf(file_bytes)
        except Exception as exc:
            self._handle_initial_failure(
                job_row,
                job_id,
                log_message="Failed to split PDF",
                error_prefix="split failed",
                exc=exc,
            )
            self._clear_cancellation_flag(job_id)
            return

        total_pages = len(pages)
        target_model = gemini_config.get("model") or self._gemini.default_model

        if total_pages == 0:
            # 0 ページの PDF はその場で完了扱いにし、各種カウンタをリセットする。
            self._repository.update_job_counters(
                job_id,
                total_pages=0,
                processed_pages=0,
                skipped_pages=0,
            )
            self._repository.update_job_status(job_id, JobStatus.DONE, None)
            self._send_summary(
                job_row,
                total_pages=0,
                processed_pages=0,
                skipped_pages=0,
                errors=[],
                status=JobStatus.DONE,
            )
            self._clear_cancellation_flag(job_id)
            return

        if self._is_cancellation_requested(job_id):
            LOGGER.info(
                "Skipping job that was cancelled after split",
                extra={"jobId": job_id},
            )
            self._handle_job_cancellation(
                job_row,
                total_pages,
                processed_pages=0,
                page_errors=[],
            )
            self._clear_cancellation_flag(job_id)
            return

        max_parallel = min(self._page_concurrency, total_pages)
        context = PageJobContext(
            job_row=job_row,
            masters=masters,
            gemini_config=gemini_config,
            target_model=target_model,
            prompt=job_row["prompt"],
            total_pages=total_pages,
            max_parallel=max_parallel or 1,
        )

        with _JOB_CONTEXTS_LOCK:
            _JOB_CONTEXTS[job_id] = context

        for page in pages:
            # ページごとの処理は別タスクとして再キューイングし、同一ワーカーでも別スレッドでも処理可能にする。
            self._queue.put(PageTask(job_id=job_id, page=page))

    # ページタスクを処理し、結果を記録および通知する。
    # 引数:
    #     task (PageTask): キューから取得したページタスク。
    def _process_page_task(self, task: PageTask) -> None:
        context = self._get_job_context(task.job_id)
        if context is None:
            LOGGER.warning(
                "Received page task without active job context",
                extra={"jobId": task.job_id, "pageIndex": task.page.index},
            )
            return

        context.semaphore.acquire(timeout=None)

        try:
            if context.cancelled or self._is_cancellation_requested(task.job_id):
                self._complete_page(
                    context,
                    success=False,
                    error=None,
                    cancelled=True,
                )
                return

            request_snapshot = self._build_request_snapshot(
                context.job_row,
                task.page,
                context.masters,
                context.gemini_config,
                context.target_model,
            )

            try:
                result = self._generate_page(
                    page=task.page,
                    prompt=context.prompt,
                    masters=context.masters,
                    gemini_config=context.gemini_config,
                )
            except Exception as exc:
                self._repository.record_gemini_log(
                    source="worker",
                    worker_name=self.name,
                    prompt=context.prompt,
                    model=context.target_model,
                    mime_type=task.page.mime_type,
                    request=request_snapshot,
                    success=False,
                    response_text=None,
                    meta=None,
                    error=str(exc),
                )
                LOGGER.exception(
                    "Failed to process page",
                    extra={"jobId": task.job_id, "pageIndex": task.page.index},
                )
                self._repository.record_page_result(
                    task.job_id,
                    task.page.index,
                    status="ERROR",
                    is_non_order_page=False,
                    raw_text=None,
                    error=str(exc),
                    meta=None,
                )
                self._complete_page(
                    context,
                    success=False,
                    error={"pageIndex": task.page.index, "message": str(exc)},
                    cancelled=False,
                )
                return

            if context.cancelled or self._is_cancellation_requested(task.job_id):
                self._complete_page(
                    context,
                    success=False,
                    error=None,
                    cancelled=True,
                )
                return

            self._repository.record_gemini_log(
                source="worker",
                worker_name=self.name,
                prompt=context.prompt,
                model=context.target_model,
                mime_type=task.page.mime_type,
                request=request_snapshot,
                success=True,
                response_text=result.text,
                meta=result.meta,
                error=None,
            )
            self._repository.record_page_result(
                task.job_id,
                task.page.index,
                status="DONE",
                is_non_order_page=False,
                raw_text=result.text,
                error=None,
                meta=result.meta,
            )

            webhook_error = self._send_page_result(context.job_row, task.page.index, result)
            if webhook_error:
                self._complete_page(
                    context,
                    success=False,
                    error={"pageIndex": task.page.index, "message": webhook_error},
                    cancelled=False,
                )
            else:
                self._complete_page(
                    context,
                    success=True,
                    error=None,
                    cancelled=False,
                )
        finally:
            context.semaphore.release()

    # アクティブなジョブコンテキストを取得する。
    # 引数:
    #     job_id (str): 取得対象のジョブ ID。
    # 返り値:
    #     Optional[PageJobContext]: 見つかったコンテキスト、存在しない場合は None。
    def _get_job_context(self, job_id: str) -> Optional[PageJobContext]:
        with _JOB_CONTEXTS_LOCK:
            return _JOB_CONTEXTS.get(job_id)

    # ページ処理完了時にカウンターとエラー情報を更新する。
    # 引数:
    #     context (PageJobContext): 対象ジョブの共有状態。
    #     success (bool): ページ処理が成功したか。
    #     error (Optional[Dict[str, str]]): エラー情報。
    #     cancelled (bool): キャンセル扱いかどうか。
    def _complete_page(
        self,
        context: PageJobContext,
        *,
        success: bool,
        error: Optional[Dict[str, str]],
        cancelled: bool,
    ) -> None:
        job_id = context.job_row["job_id"]
        effective_cancelled = cancelled or context.cancelled or self._is_cancellation_requested(job_id)
        should_finalize = False
        with context.lock:
            if effective_cancelled:
                context.cancelled = True
            if success:
                context.processed_pages += 1
            elif error and not effective_cancelled:
                context.page_errors.append(error)
            context.pending_pages -= 1
            should_finalize = context.pending_pages <= 0

        if should_finalize:
            # 最後のページが完了したタイミングでサマリー送信や後処理をまとめて行う。
            self._finalize_job_context(context)

    # ページ処理完了後に集計とサマリー送信を行う。
    # 引数:
    #     context (PageJobContext): 対象ジョブの共有状態。
    def _finalize_job_context(self, context: PageJobContext) -> None:
        job_id = context.job_row["job_id"]
        with context.lock:
            if context.completed:
                return
            context.completed = True
            total_pages = context.total_pages
            processed_pages = context.processed_pages
            page_errors = list(context.page_errors)
            cancelled = context.cancelled or self._is_cancellation_requested(job_id)

        with _JOB_CONTEXTS_LOCK:
            _JOB_CONTEXTS.pop(job_id, None)

        if cancelled:
            LOGGER.info(
                "Cancellation detected before summary",
                extra={"jobId": job_id},
            )
            self._handle_job_cancellation(
                context.job_row,
                total_pages,
                processed_pages,
                page_errors,
            )
        else:
            skipped_pages = max(total_pages - processed_pages, 0)
            self._repository.update_job_counters(
                job_id,
                total_pages=total_pages,
                processed_pages=processed_pages,
                skipped_pages=skipped_pages,
            )

            if page_errors:
                self._repository.update_job_status(
                    job_id, JobStatus.ERROR, "; ".join(err["message"] for err in page_errors)
                )
                summary_status = JobStatus.ERROR
            else:
                self._repository.update_job_status(job_id, JobStatus.DONE, None)
                summary_status = JobStatus.DONE

            self._send_summary(
                context.job_row,
                total_pages=total_pages,
                processed_pages=processed_pages,
                skipped_pages=skipped_pages,
                errors=page_errors,
                status=summary_status,
            )

        self._clear_cancellation_flag(job_id)

    # ジョブにキャンセル要求が出ているかを判定する。
    # 引数:
    #     job_id (str): 確認対象のジョブ ID。
    # 返り値:
    #     bool: キャンセル要求がある場合は True。
    def _is_cancellation_requested(self, job_id: str) -> bool:
        if self._admin_state and self._admin_state.is_cancellation_requested(job_id):
            return True
        status = self._repository.get_job_status(job_id)
        return status == JobStatus.CANCELLED.value

    # キャンセル時の状態更新と通知をまとめて行う。
    # 引数:
    #     job_row: ジョブのデータベース行。
    #     total_pages (int): ジョブ全体のページ数。
    #     processed_pages (int): 完了したページ数。
    #     page_errors (list[Dict[str, str]]): 蓄積したページエラー一覧。
    def _handle_job_cancellation(
        self,
        job_row,
        total_pages: int,
        processed_pages: int,
        page_errors: list[Dict[str, str]],
    ) -> None:
        # キャンセル時はページエラー一覧に通知用メッセージを追加する。
        skipped_pages = max(total_pages - processed_pages, 0)
        errors = list(page_errors)
        errors.append({"message": "Cancelled via admin console"})
        self._repository.update_job_counters(
            job_row["job_id"],
            total_pages=total_pages,
            processed_pages=processed_pages,
            skipped_pages=skipped_pages,
        )
        self._repository.update_job_status(job_row["job_id"], JobStatus.CANCELLED, CANCEL_REASON)
        self._send_summary(
            job_row,
            total_pages=total_pages,
            processed_pages=processed_pages,
            skipped_pages=skipped_pages,
            errors=errors,
            status=JobStatus.CANCELLED,
        )

    # 管理画面上のキャンセルフラグをリセットする。
    # 引数:
    #     job_id (str): フラグを解除するジョブ ID。
    def _clear_cancellation_flag(self, job_id: str) -> None:
        if self._admin_state is not None:
            self._admin_state.clear_job_cancellation(job_id)

    # Gemini 呼び出し内容をログ用にスナップショット化する。
    # 引数:
    #     job_row: ジョブ情報を含むデータベース行。
    #     page (PagePayload): 処理対象のページデータ。
    #     masters (Dict[str, str]): マスター情報。
    #     gemini_config (Dict): Gemini への追加設定。
    #     target_model (str): 使用するモデル名。
    # 返り値:
    #     Dict: ログ保存用に正規化した呼び出し情報。
    def _build_request_snapshot(
        self,
        job_row,
        page: PagePayload,
        masters: Dict[str, str],
        gemini_config: Dict,
        target_model: str,
    ) -> Dict:
        return {
            "jobId": job_row["job_id"],
            "orderId": job_row["order_id"],
            "pageIndex": page.index,
            "prompt": job_row["prompt"],
            "promptLength": len(job_row["prompt"]),
            "masters": masters,
            "mastersKeys": sorted(masters.keys()),
            "input": {
                "mode": "pdf_page",
                "mimeType": page.mime_type,
                "sizeBytes": len(page.data),
            },
            "parameters": {
                "model": target_model,
                "temperature": gemini_config.get("temperature"),
                "topP": gemini_config.get("topP"),
                "topK": gemini_config.get("topK"),
                "maxOutputTokens": gemini_config.get("maxOutputTokens"),
            },
        }

    # Gemini API を呼び出して 1 ページ分のテキストを生成する。
    # 引数:
    #     page (PagePayload): OCR 対象のページ。
    #     prompt (str): Gemini へ渡すプロンプト。
    #     masters (Dict[str, str]): マスターデータ。
    #     gemini_config (Dict): 追加設定。
    # 返り値:
    #     任意: Gemini クライアントが返すレスポンス。
    def _generate_page(
        self,
        *,
        page: PagePayload,
        prompt: str,
        masters: Dict[str, str],
        gemini_config: Dict,
    ):
        return self._gemini.generate(
            model=gemini_config.get("model"),
            prompt=prompt,
            page_bytes=page.data,
            mime_type=page.mime_type,
            masters=masters,
            temperature=gemini_config.get("temperature"),
            top_p=gemini_config.get("topP"),
            top_k=gemini_config.get("topK"),
            max_output_tokens=gemini_config.get("maxOutputTokens"),
        )

    # ページ処理結果を Webhook へ送信する。
    # 引数:
    #     job_row: 対象ジョブのデータベース行。
    #     page_index (int): ページ番号。
    #     result: Gemini の生成結果。
    # 返り値:
    #     Optional[str]: 送信失敗時のエラーメッセージ。成功時は None。
    def _send_page_result(self, job_row, page_index: int, result) -> Optional[str]:
        token = job_row["webhook_token"] or None
        payload = {
            "event": "PAGE_RESULT",
            "jobId": job_row["job_id"],
            "orderId": job_row["order_id"],
            "pageIndex": page_index,
            "isNonOrderPage": False,
            "rawText": result.text,
            "meta": result.meta,
            "idempotencyKey": f"{job_row['order_id']}:{page_index}",
            "token": token,
        }
        try:
            response = self._webhook.send(job_row["webhook_url"], payload, token=token)
        except Exception as exc:
            LOGGER.exception(
                "Failed to dispatch page webhook",
                extra={"jobId": job_row["job_id"], "pageIndex": page_index},
            )
            response = getattr(exc, "response", None)
            status_code = response.status_code if response is not None else None
            response_text = response.text if response is not None else None
            self._log_relay_webhook(
                job_row,
                event="PAGE_RESULT",
                payload=payload,
                success=False,
                status_code=status_code,
                response_text=response_text,
                error=str(exc),
                token=token,
            )
            # 失敗情報をサマリーペイロードにも反映させるために記録する。
            self._repository.record_page_result(
                job_row["job_id"],
                page_index,
                status="ERROR",
                is_non_order_page=False,
                raw_text=result.text,
                error=f"webhook failed: {exc}",
                meta=result.meta,
            )
            return f"webhook failed: {exc}"
        self._log_relay_webhook(
            job_row,
            event="PAGE_RESULT",
            payload=payload,
            success=True,
            status_code=response.status_code,
            response_text=response.text,
            error=None,
            token=token,
        )
        return None

    # ジョブ全体のサマリーを Webhook へ送信する。
    # 引数:
    #     job_row: 対象ジョブのデータベース行。
    #     total_pages (int): 全ページ数。
    #     processed_pages (int): 完了したページ数。
    #     skipped_pages (int): スキップされたページ数。
    #     errors (list[Dict[str, str]]): 発生したエラー情報の一覧。
    #     status (Optional[JobStatus]): 最終ジョブステータス。
    def _send_summary(
        self,
        job_row,
        *,
        total_pages: int,
        processed_pages: int,
        skipped_pages: int,
        errors: list[Dict[str, str]],
        status: Optional[JobStatus] = None,
    ) -> None:
        token = job_row["webhook_token"] or None
        payload = {
            "event": "JOB_SUMMARY",
            "jobId": job_row["job_id"],
            "orderId": job_row["order_id"],
            "totalPages": total_pages,
            "processedPages": processed_pages,
            "skippedPages": skipped_pages,
            "errors": errors,
            "token": token,
        }
        if status is not None:
            payload["status"] = status.value
        try:
            response = self._webhook.send(job_row["webhook_url"], payload, token=token)
        except Exception as exc:
            LOGGER.exception("Failed to dispatch summary webhook", extra={"jobId": job_row["job_id"]})
            response = getattr(exc, "response", None)
            status_code = response.status_code if response is not None else None
            response_text = response.text if response is not None else None
            self._log_relay_webhook(
                job_row,
                event="JOB_SUMMARY",
                payload=payload,
                success=False,
                status_code=status_code,
                response_text=response_text,
                error=str(exc),
                token=token,
            )
            return
        self._log_relay_webhook(
            job_row,
            event="JOB_SUMMARY",
            payload=payload,
            success=True,
            status_code=response.status_code,
            response_text=response.text,
            error=None,
            token=token,
        )

    # Webhook 呼び出し結果を管理画面のログへ記録する。
    # 引数:
    #     job_row: 対象ジョブのデータベース行。
    #     event (str): ログ対象イベント名。
    #     payload (Dict): 送信したペイロード。
    #     success (bool): 送信が成功したかどうか。
    #     status_code (Optional[int]): 応答ステータスコード。
    #     response_text (Optional[str]): 応答本文。
    #     error (Optional[str]): エラー文字列。
    #     token (Optional[str]): Webhook 認証トークン。
    def _log_relay_webhook(
        self,
        job_row,
        *,
        event: str,
        payload: Dict,
        success: bool,
        status_code: Optional[int],
        response_text: Optional[str],
        error: Optional[str],
        token: Optional[str],
    ) -> None:
        if self._admin_state is None:
            return
        # 認証トークンは管理画面のログに残さないようにマスクする。
        payload_for_log = dict(payload)
        if payload_for_log.get("token"):
            payload_for_log["token"] = "***"
        payload_text = json.dumps(payload_for_log, ensure_ascii=False, indent=2, sort_keys=True)
        self._admin_state.add_relay_webhook_log(
            job_id=job_row["job_id"],
            order_id=job_row["order_id"],
            event=event,
            url=job_row["webhook_url"],
            payload=payload_text,
            success=success,
            status_code=status_code,
            response_text=response_text,
            error=error,
            token=token,
        )

    # ジョブの初期段階で発生した失敗を処理する。
    # 引数:
    #     job_row: 対象ジョブのデータベース行。
    #     job_id (str): ジョブ ID。
    #     log_message (str): ログ出力時の説明文。
    #     error_prefix (str): エラーメッセージの接頭辞。
    #     exc (Exception): 発生した例外。
    def _handle_initial_failure(
        self,
        job_row,
        job_id: str,
        *,
        log_message: str,
        error_prefix: str,
        exc: Exception,
    ) -> None:
        LOGGER.exception(log_message, extra={"jobId": job_id})
        # 初期段階で失敗した場合も状態とサマリーを更新し、再試行に備える。
        self._repository.update_job_status(job_id, JobStatus.ERROR, str(exc))
        self._repository.update_job_counters(
            job_id,
            total_pages=None,
            processed_pages=None,
            skipped_pages=None,
        )
        self._send_summary(
            job_row,
            total_pages=0,
            processed_pages=0,
            skipped_pages=0,
            errors=[{"message": f"{error_prefix}: {exc}"}],
            status=JobStatus.ERROR,
        )


__all__ = ["JobWorker", "JobTask"]
