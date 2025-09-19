"""Background worker implementation."""

from __future__ import annotations

import json
import queue
import threading
from typing import Dict, Optional, TYPE_CHECKING

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


class JobWorker(threading.Thread):
    """Threaded worker that processes jobs from the database queue."""

    def __init__(
        self,
        *,
        repository: JobRepository,
        job_queue: "queue.Queue[str]",
        file_fetcher: FileFetcher,
        gemini_client: GeminiClient,
        webhook_dispatcher: WebhookDispatcher,
        idle_sleep: float,
        admin_state: "AdminState | None" = None,
    ) -> None:
        super().__init__(daemon=True)
        self._repository = repository
        self._queue = job_queue
        self._file_fetcher = file_fetcher
        self._gemini = gemini_client
        self._webhook = webhook_dispatcher
        self._idle_sleep = idle_sleep
        self._admin_state = admin_state
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - threading logic
        LOGGER.info("Worker thread started")
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=self._idle_sleep)
            except queue.Empty:
                continue
            try:
                self._process_job(job_id)
            except Exception as exc:  # pragma: no cover - unexpected errors
                LOGGER.exception("Unexpected error during job processing", extra={"jobId": job_id})
                self._repository.update_job_status(job_id, JobStatus.ERROR, str(exc))
            finally:
                self._queue.task_done()
        LOGGER.info("Worker thread stopped")

    def _process_job(self, job_id: str) -> None:
        job_row = self._repository.get_job(job_id)
        if job_row is None:
            LOGGER.warning("Job not found when processing", extra={"jobId": job_id})
            return

        if job_row["status"] not in (JobStatus.RECEIVED.value, JobStatus.ENQUEUED.value, JobStatus.PROCESSING.value):
            LOGGER.info("Skipping job in terminal state", extra={"jobId": job_id, "status": job_row["status"]})
            return

        self._repository.update_job_status(job_id, JobStatus.PROCESSING)
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
            return

        pages: list[PagePayload]
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
            return

        total_pages = len(pages)
        processed_pages = 0
        page_errors: list[Dict[str, str]] = []

        for page in pages:
            target_model = gemini_config.get("model") or self._gemini.default_model
            request_snapshot = {
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
            try:
                result = self._gemini.generate(
                    model=gemini_config.get("model"),
                    prompt=job_row["prompt"],
                    page_bytes=page.data,
                    mime_type=page.mime_type,
                    masters=masters,
                    temperature=gemini_config.get("temperature"),
                    top_p=gemini_config.get("topP"),
                    top_k=gemini_config.get("topK"),
                    max_output_tokens=gemini_config.get("maxOutputTokens"),
                )
                self._repository.record_gemini_log(
                    source="worker",
                    prompt=job_row["prompt"],
                    model=target_model,
                    mime_type=page.mime_type,
                    request=request_snapshot,
                    success=True,
                    response_text=result.text,
                    meta=result.meta,
                    error=None,
                )
                self._repository.record_page_result(
                    job_id,
                    page.index,
                    status="DONE",
                    is_non_order_page=False,
                    raw_text=result.text,
                    error=None,
                    meta=result.meta,
                )
                webhook_error = self._send_page_result(job_row, page.index, result)
                if webhook_error:
                    page_errors.append({"pageIndex": page.index, "message": webhook_error})
                else:
                    processed_pages += 1
            except Exception as exc:
                self._repository.record_gemini_log(
                    source="worker",
                    prompt=job_row["prompt"],
                    model=target_model,
                    mime_type=page.mime_type,
                    request=request_snapshot,
                    success=False,
                    response_text=None,
                    meta=None,
                    error=str(exc),
                )
                LOGGER.exception("Failed to process page", extra={"jobId": job_id, "pageIndex": page.index})
                self._repository.record_page_result(
                    job_id,
                    page.index,
                    status="ERROR",
                    is_non_order_page=False,
                    raw_text=None,
                    error=str(exc),
                    meta=None,
                )
                page_errors.append({"pageIndex": page.index, "message": str(exc)})

        skipped_pages = max(total_pages - processed_pages, 0)

        self._repository.update_job_counters(
            job_id,
            total_pages=total_pages,
            processed_pages=processed_pages,
            skipped_pages=skipped_pages,
        )

        if page_errors:
            summary_status = JobStatus.ERROR
            self._repository.update_job_status(job_id, JobStatus.ERROR, "; ".join(err["message"] for err in page_errors))
        else:
            summary_status = JobStatus.DONE
            self._repository.update_job_status(job_id, JobStatus.DONE, None)

        self._send_summary(
            job_row,
            total_pages=total_pages,
            processed_pages=processed_pages,
            skipped_pages=skipped_pages,
            errors=page_errors,
            status=summary_status,
        )

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
            # Record the failure so that it surfaces in the summary payload.
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


__all__ = ["JobWorker"]
