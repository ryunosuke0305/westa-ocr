"""SQLite-backed persistence utilities."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .logging_config import get_logger
from .models import JobStatus

LOGGER = get_logger(__name__)


class JobRepository:
    """Data-access layer backed by SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = threading.RLock()
        self._conn = self._connect(db_path)
        self._initialise()

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            LOGGER.info("Creating new SQLite database", extra={"path": str(path)})
            path.touch()
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _initialise(self) -> None:
        with self._locked():
            self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                pattern TEXT,
                masters_json TEXT NOT NULL,
                webhook_url TEXT NOT NULL,
                webhook_token TEXT,
                gemini_json TEXT,
                options_json TEXT,
                idempotency_key TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                total_pages INTEGER,
                processed_pages INTEGER,
                skipped_pages INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS job_pages (
                job_id TEXT NOT NULL,
                page_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                is_non_order_page INTEGER NOT NULL DEFAULT 0,
                raw_text TEXT,
                error TEXT,
                meta_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (job_id, page_index),
                FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS gemini_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                source TEXT NOT NULL,
                prompt_preview TEXT NOT NULL,
                model TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                request_json TEXT,
                success INTEGER NOT NULL,
                response_text TEXT,
                meta_json TEXT,
                error TEXT
            );
            """
        )

    @contextmanager
    def _locked(self):
        with self._lock:
            yield

    def close(self) -> None:
        with self._locked():
            self._conn.close()

    # ------------------------------------------------------------------
    # Job persistence helpers
    # ------------------------------------------------------------------
    def insert_job(
        self,
        *,
        job_id: str,
        order_id: str,
        file_id: str,
        prompt: str,
        pattern: Optional[str],
        masters: Dict[str, str],
        webhook: Dict[str, str],
        gemini: Optional[Dict],
        options: Optional[Dict],
        idempotency_key: str,
    ) -> None:
        webhook_token = (webhook.get("token") or "").strip()
        if not webhook_token:
            raise ValueError("webhook token is required")

        payload = {
            "job_id": job_id,
            "order_id": order_id,
            "file_id": file_id,
            "prompt": prompt,
            "pattern": pattern,
            "masters_json": json.dumps(masters),
            "webhook_url": webhook["url"],
            "webhook_token": webhook_token,
            "gemini_json": json.dumps(gemini) if gemini else None,
            "options_json": json.dumps(options) if options else None,
            "idempotency_key": idempotency_key,
            "status": JobStatus.RECEIVED.value,
        }

        with self._locked():
            self._conn.execute(
                """
                INSERT INTO jobs (
                    job_id, order_id, file_id, prompt, pattern, masters_json,
                    webhook_url, webhook_token, gemini_json, options_json,
                    idempotency_key, status
                ) VALUES (:job_id, :order_id, :file_id, :prompt, :pattern, :masters_json,
                          :webhook_url, :webhook_token, :gemini_json, :options_json,
                          :idempotency_key, :status)
                """,
                payload,
            )

    def update_job_status(self, job_id: str, status: JobStatus, error: Optional[str] = None) -> None:
        with self._locked():
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       last_error = ?,
                       updated_at = datetime('now')
                 WHERE job_id = ?
                """,
                (status.value, error, job_id),
            )

    def update_job_counters(
        self,
        job_id: str,
        *,
        total_pages: Optional[int],
        processed_pages: Optional[int],
        skipped_pages: Optional[int],
    ) -> None:
        with self._locked():
            self._conn.execute(
                """
                UPDATE jobs
                   SET total_pages = ?,
                       processed_pages = ?,
                       skipped_pages = ?,
                       updated_at = datetime('now')
                 WHERE job_id = ?
                """,
                (total_pages, processed_pages, skipped_pages, job_id),
            )

    def record_page_result(
        self,
        job_id: str,
        page_index: int,
        *,
        status: str,
        is_non_order_page: bool,
        raw_text: Optional[str],
        error: Optional[str],
        meta: Optional[Dict],
    ) -> None:
        payload = {
            "job_id": job_id,
            "page_index": page_index,
            "status": status,
            "is_non_order_page": 1 if is_non_order_page else 0,
            "raw_text": raw_text,
            "error": error,
            "meta_json": json.dumps(meta) if meta else None,
        }
        with self._locked():
            self._conn.execute(
                """
                INSERT INTO job_pages (
                    job_id, page_index, status, is_non_order_page, raw_text, error, meta_json
                ) VALUES (:job_id, :page_index, :status, :is_non_order_page, :raw_text, :error, :meta_json)
                ON CONFLICT(job_id, page_index) DO UPDATE SET
                    status = excluded.status,
                    is_non_order_page = excluded.is_non_order_page,
                    raw_text = excluded.raw_text,
                    error = excluded.error,
                    meta_json = excluded.meta_json,
                    updated_at = datetime('now')
                """,
                payload,
            )

    def find_job_by_idempotency(self, key: str) -> Optional[sqlite3.Row]:
        with self._locked():
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        return row

    def get_job(self, job_id: str) -> Optional[sqlite3.Row]:
        with self._locked():
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return row

    def get_job_status(self, job_id: str) -> Optional[str]:
        with self._locked():
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return row["status"] if row else None

    def cancel_job(self, job_id: str, *, reason: Optional[str] = None) -> bool:
        cancellable = {
            JobStatus.RECEIVED.value,
            JobStatus.ENQUEUED.value,
            JobStatus.PROCESSING.value,
        }
        cancellation_reason = reason or "Cancelled via admin console"
        with self._locked():
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return False
            if row["status"] not in cancellable:
                return False
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       last_error = ?,
                       updated_at = datetime('now')
                 WHERE job_id = ?
                """,
                (JobStatus.CANCELLED.value, cancellation_reason, job_id),
            )
        return True

    def list_pending_jobs(self) -> List[str]:
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT job_id
                  FROM jobs
                 WHERE status IN (?, ?, ?)
                 ORDER BY created_at ASC
                """,
                (
                    JobStatus.RECEIVED.value,
                    JobStatus.ENQUEUED.value,
                    JobStatus.PROCESSING.value,
                ),
            ).fetchall()
        return [row["job_id"] for row in rows]

    def list_recent_jobs(self, limit: int = 20) -> List[Dict]:
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT job_id, order_id, status, created_at, updated_at,
                       total_pages, processed_pages, skipped_pages, last_error
                  FROM jobs
                 ORDER BY updated_at DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()

        jobs: List[Dict] = []
        for row in rows:
            created_at = row["created_at"]
            updated_at = row["updated_at"]
            try:
                created = datetime.fromisoformat(created_at)
            except ValueError:
                created = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            try:
                updated = datetime.fromisoformat(updated_at)
            except ValueError:
                updated = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
            jobs.append(
                {
                    "job_id": row["job_id"],
                    "order_id": row["order_id"],
                    "status": row["status"],
                    "created_at": created,
                    "updated_at": updated,
                    "total_pages": row["total_pages"],
                    "processed_pages": row["processed_pages"],
                    "skipped_pages": row["skipped_pages"],
                    "last_error": row["last_error"],
                }
            )
        return jobs

    def mark_enqueued(self, job_id: str) -> None:
        with self._locked():
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?,
                       updated_at = datetime('now')
                 WHERE job_id = ?
                """,
                (JobStatus.ENQUEUED.value, job_id),
            )

    def list_page_errors(self, job_id: str) -> List[Dict[str, str]]:
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT page_index, error
                  FROM job_pages
                 WHERE job_id = ? AND error IS NOT NULL
                """,
                (job_id,),
            ).fetchall()
        return [
            {"pageIndex": row["page_index"], "message": row["error"]}
            for row in rows
        ]

    def list_pages(self, job_id: str) -> List[Dict]:
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT page_index, status, is_non_order_page, raw_text, error, meta_json
                  FROM job_pages
                 WHERE job_id = ?
                 ORDER BY page_index ASC
                """,
                (job_id,),
            ).fetchall()
        pages: List[Dict] = []
        for row in rows:
            pages.append(
                {
                    "pageIndex": row["page_index"],
                    "status": row["status"],
                    "isNonOrderPage": bool(row["is_non_order_page"]),
                    "rawText": row["raw_text"],
                    "error": row["error"],
                    "meta": json.loads(row["meta_json"]) if row["meta_json"] else None,
                }
            )
        return pages

    def get_job_detail(self, job_id: str) -> Optional[Dict]:
        job = self.get_job(job_id)
        if job is None:
            return None
        masters = json.loads(job["masters_json"])
        detail = {
            "jobId": job["job_id"],
            "orderId": job["order_id"],
            "status": job["status"],
            "fileId": job["file_id"],
            "prompt": job["prompt"],
            "pattern": job["pattern"],
            "masters": masters,
            "webhookUrl": job["webhook_url"],
            "webhookToken": job["webhook_token"],
            "createdAt": datetime.fromisoformat(job["created_at"]),
            "updatedAt": datetime.fromisoformat(job["updated_at"]),
            "totalPages": job["total_pages"],
            "processedPages": job["processed_pages"],
            "skippedPages": job["skipped_pages"],
            "lastError": job["last_error"],
            "pages": self.list_pages(job_id),
        }
        return detail

    # ------------------------------------------------------------------
    # Gemini logging helpers
    # ------------------------------------------------------------------
    def record_gemini_log(
        self,
        *,
        source: str,
        prompt: str,
        model: str,
        mime_type: str,
        request: Optional[Dict],
        success: bool,
        response_text: Optional[str],
        meta: Optional[Dict],
        error: Optional[str],
    ) -> None:
        payload = {
            "source": source,
            "prompt_preview": (prompt or "")[:200].replace("\n", " "),
            "model": model,
            "mime_type": mime_type,
            "request_json": json.dumps(request, ensure_ascii=False) if request else None,
            "success": 1 if success else 0,
            "response_text": response_text,
            "meta_json": json.dumps(meta, ensure_ascii=False) if meta else None,
            "error": error,
        }
        with self._locked():
            self._conn.execute(
                """
                INSERT INTO gemini_logs (
                    source, prompt_preview, model, mime_type, request_json,
                    success, response_text, meta_json, error
                ) VALUES (
                    :source, :prompt_preview, :model, :mime_type, :request_json,
                    :success, :response_text, :meta_json, :error
                )
                """,
                payload,
            )

    def list_gemini_logs(self, limit: int = 10) -> List[Dict]:
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT id, created_at, source, prompt_preview, model, mime_type,
                       request_json, success, response_text, meta_json, error
                  FROM gemini_logs
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()

        logs: List[Dict] = []
        for row in rows:
            created_at = row["created_at"]
            try:
                timestamp = datetime.fromisoformat(created_at)
            except ValueError:
                timestamp = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            request_payload = json.loads(row["request_json"]) if row["request_json"] else None
            meta_payload = json.loads(row["meta_json"]) if row["meta_json"] else None
            logs.append(
                {
                    "id": row["id"],
                    "timestamp": timestamp,
                    "source": row["source"],
                    "prompt_preview": row["prompt_preview"],
                    "model": row["model"],
                    "mime_type": row["mime_type"],
                    "request": request_payload,
                    "success": bool(row["success"]),
                    "response_text": row["response_text"],
                    "meta": meta_payload,
                    "error": row["error"],
                }
            )
        return logs


__all__ = ["JobRepository"]
