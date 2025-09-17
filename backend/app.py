"""Flask application exposing the OCR API."""
from __future__ import annotations

import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

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
    if "file" not in request.files:
        return _error_response("ファイルが含まれていません。", HTTPStatus.BAD_REQUEST)

    upload = request.files["file"]
    if upload.filename == "":
        return _error_response("ファイル名が空です。", HTTPStatus.BAD_REQUEST)

    file_bytes = upload.read()
    if not file_bytes:
        return _error_response("ファイルが空です。", HTTPStatus.BAD_REQUEST)

    if len(file_bytes) > config.max_upload_size:
        return _error_response("ファイルサイズが上限を超えています。", HTTPStatus.BAD_REQUEST)

    try:
        results = ocr_service.extract(file_bytes, upload.filename)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to run OCR")
        return _error_response(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({"results": results})


def _error_response(message: str, status: HTTPStatus):
    payload: Dict[str, str] = {"error": message}
    return jsonify(payload), status


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
