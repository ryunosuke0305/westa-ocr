"""Flask application exposing the OCR API."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import load_config
from ocr_service import OCRService

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

config = load_config()
ocr_service = OCRService(config)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    job = _start_job()
    _append_job_log(job, "request", "info", "OCRリクエストを受信しました。")

    def fail(
        stage: str,
        message: str,
        status: HTTPStatus,
        extra: Optional[Dict[str, Any]] = None,
    ):
        _append_job_log(job, stage, "error", message, extra)
        _append_job_log(job, "response", "error", message)
        return _error_response(message, status, job)

    if "file" not in request.files:
        return fail("validation", "ファイルが含まれていません。", HTTPStatus.BAD_REQUEST)

    upload = request.files["file"]
    job.original_filename = upload.filename or ""
    _append_job_log(
        job,
        "validation",
        "info",
        f"アップロードされたファイル名: {job.original_filename or '(未設定)'}",
    )
    if upload.filename == "":
        return fail("validation", "ファイル名が空です。", HTTPStatus.BAD_REQUEST)

    file_bytes = upload.read()
    file_size = len(file_bytes)
    if file_size == 0:
        return fail(
            "validation",
            "ファイルが空です。",
            HTTPStatus.BAD_REQUEST,
            {"file_size": file_size},
        )

    if file_size > config.max_upload_size:
        return fail(
            "validation",
            "ファイルサイズが上限を超えています。",
            HTTPStatus.BAD_REQUEST,
            {"file_size": file_size, "max_size": config.max_upload_size},
        )

    _append_job_log(
        job,
        "validation",
        "info",
        f"ファイルサイズ: {file_size}バイト",
        {"file_size": file_size},
    )

    try:
        saved_path = _save_upload(file_bytes, upload.filename)
        LOGGER.info("Saved uploaded file to %s", saved_path)
        _append_job_log(job, "persist", "success", f"ファイルを保存しました: {saved_path}")
    except OSError as exc:
        LOGGER.exception("Failed to persist uploaded file")
        return fail(
            "persist",
            "ファイルの保存に失敗しました。",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"detail": str(exc)},
        )

    _append_job_log(job, "ocr", "info", "OCR処理を開始します。")
    try:
        results = ocr_service.extract(file_bytes, upload.filename)
        _append_job_log(job, "ocr", "success", f"{len(results)}件の結果を取得しました。")
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to run OCR")
        return fail(
            "ocr",
            "OCR処理中にエラーが発生しました。",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"detail": str(exc)},
        )

    response_path = _persist_response(job, results)
    _append_job_log(job, "store", "success", f"結果を保存しました: {response_path}")
    _append_job_log(job, "response", "success", "クライアントへ結果を返却します。")

    return jsonify({"results": results, "job_id": job.job_id})


@app.route("/api/prompt", methods=["GET"])
def get_prompt_template():
    """Return the currently configured Gemini prompt template."""
    prompt = ocr_service.get_prompt_template()
    return jsonify({"prompt": prompt})


@app.route("/api/prompt", methods=["POST"])
def update_prompt_template():
    """Update the Gemini prompt template and persist it to storage."""
    payload = request.get_json(silent=True) or {}
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        return _error_response("プロンプトが指定されていません。", HTTPStatus.BAD_REQUEST)
    if not prompt.strip():
        return _error_response(
            "プロンプトを空にすることはできません。", HTTPStatus.BAD_REQUEST
        )
    try:
        ocr_service.update_prompt_template(prompt)
    except OSError as exc:
        LOGGER.exception("Failed to persist prompt template")
        return _error_response(
            "プロンプトの保存に失敗しました。", HTTPStatus.INTERNAL_SERVER_ERROR
        )
    return jsonify({"message": "プロンプトを更新しました。"})


def _error_response(
    message: str, status: HTTPStatus, job: Optional["JobContext"] = None
):
    payload: Dict[str, str] = {"error": message}
    if job is not None:
        payload["job_id"] = job.job_id
    return jsonify(payload), status


def _save_upload(file_bytes: bytes, original_filename: str) -> Path:
    """Persist the uploaded file to the configured directory."""
    safe_name = secure_filename(original_filename)
    if not safe_name:
        suffix = Path(original_filename).suffix
        safe_name = f"uploaded-file{suffix}" if suffix else "uploaded-file"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    unique_name = f"{timestamp}-{uuid4().hex}-{safe_name}"
    destination = config.upload_dir / unique_name

    with open(destination, "wb") as file_obj:
        file_obj.write(file_bytes)

    return destination


@dataclass
class JobContext:
    job_id: str
    directory: Path
    original_filename: str = ""

    @property
    def log_path(self) -> Path:
        return self.directory / "events.jsonl"


def _start_job() -> JobContext:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_id = f"{timestamp}-{uuid4().hex}"
    job_dir = config.job_history_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return JobContext(job_id=job_id, directory=job_dir)


def _append_job_log(
    job: JobContext,
    stage: str,
    status: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id,
        "stage": stage,
        "status": status,
        "message": message,
    }
    if job.original_filename:
        entry["original_filename"] = job.original_filename
    if extra:
        entry.update(extra)

    with open(job.log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _persist_response(job: JobContext, results: List[Dict[str, Any]]) -> Path:
    payload: Dict[str, Any] = {
        "job_id": job.job_id,
        "original_filename": job.original_filename,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    destination = config.response_history_dir / f"{job.job_id}.json"
    with open(destination, "w", encoding="utf-8") as response_file:
        json.dump(payload, response_file, ensure_ascii=False, indent=2)
    return destination


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """Return the latest job history entries."""
    raw_limit = request.args.get("limit", "20")
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 100))

    try:
        jobs = _collect_jobs(limit)
    except OSError:
        LOGGER.exception("Failed to list job history")
        return _error_response(
            "ジョブ履歴の取得に失敗しました。", HTTPStatus.INTERNAL_SERVER_ERROR
        )
    return jsonify({"jobs": jobs})


@app.route("/api/jobs/<job_id>/logs", methods=["GET"])
def job_logs(job_id: str):
    """Return log entries for a specific job."""
    job_dir = config.job_history_dir / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        return _error_response("指定されたジョブは見つかりません。", HTTPStatus.NOT_FOUND)

    try:
        payload = _load_job_logs(job_id)
    except OSError:
        return _error_response(
            "ジョブログの読み込みに失敗しました。", HTTPStatus.INTERNAL_SERVER_ERROR
        )

    return jsonify(payload)


def _load_job_metadata(job_dir: Path) -> Dict[str, Any]:
    events_path = job_dir / "events.jsonl"
    created_at: Optional[str] = None
    original_filename = ""

    if events_path.is_file():
        try:
            with open(events_path, "r", encoding="utf-8") as events_file:
                for line in events_file:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning(
                            "Invalid JSON in job history for %s: %s",
                            job_dir.name,
                            line.strip(),
                        )
                        continue
                    created_at = created_at or entry.get("timestamp")
                    if not original_filename:
                        original_filename = entry.get("original_filename", "")
                    if created_at and original_filename:
                        break
        except OSError:
            LOGGER.exception("Failed to read job metadata for %s", job_dir)

    if created_at is None:
        created_at = datetime.fromtimestamp(
            job_dir.stat().st_mtime, timezone.utc
        ).isoformat()

    return {
        "job_id": job_dir.name,
        "created_at": created_at,
        "original_filename": original_filename,
    }


def _collect_jobs(limit: int) -> List[Dict[str, Any]]:
    job_dirs = [
        path
        for path in config.job_history_dir.iterdir()
        if path.is_dir()
    ]
    sorted_dirs = sorted(job_dirs, key=lambda path: path.name, reverse=True)
    jobs: List[Dict[str, Any]] = []
    for job_dir in sorted_dirs[:limit]:
        jobs.append(_load_job_metadata(job_dir))
    return jobs


def _load_job_logs(job_id: str) -> Dict[str, Any]:
    job_dir = config.job_history_dir / job_id
    events_path = job_dir / "events.jsonl"
    logs: List[Dict[str, Any]] = []
    original_filename = ""
    created_at: Optional[str] = None

    if events_path.is_file():
        try:
            with open(events_path, "r", encoding="utf-8") as events_file:
                for line in events_file:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning(
                            "Invalid JSON in job history for %s: %s",
                            job_id,
                            line.strip(),
                        )
                        continue
                    logs.append(entry)
                    created_at = created_at or entry.get("timestamp")
                    if not original_filename:
                        original_filename = entry.get("original_filename", "")
        except OSError:
            LOGGER.exception("Failed to read job log for %s", job_id)
            raise

    if created_at is None:
        created_at = datetime.fromtimestamp(
            job_dir.stat().st_mtime, timezone.utc
        ).isoformat()

    return {
        "job_id": job_id,
        "logs": logs,
        "original_filename": original_filename,
        "created_at": created_at,
    }


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    """Serve the bundled frontend assets."""
    if path == "api" or path.startswith("api/"):
        return _error_response("指定されたパスは存在しません。", HTTPStatus.NOT_FOUND)

    if not FRONTEND_DIR.exists():
        LOGGER.error("Frontend directory is missing: %%s", FRONTEND_DIR)
        return (
            "フロントエンドが利用できません。管理者に問い合わせてください。",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    safe_path = path or "index.html"
    candidate = FRONTEND_DIR / safe_path
    if candidate.is_dir():
        candidate = candidate / "index.html"
    if not candidate.exists():
        candidate = FRONTEND_DIR / "index.html"

    try:
        relative_path = candidate.relative_to(FRONTEND_DIR)
    except ValueError:
        return _error_response("指定されたパスは存在しません。", HTTPStatus.NOT_FOUND)
    return send_from_directory(FRONTEND_DIR, relative_path.as_posix())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
