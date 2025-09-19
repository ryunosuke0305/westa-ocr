import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.environ.setdefault("RELAY_TOKEN", "test-token")

import pytest

from app.models import JobStatus
from app.repository import JobRepository


def _create_repository(tmp_path: Path) -> JobRepository:
    repo = JobRepository(tmp_path / "relay.db")
    return repo


def test_insert_job_persists_webhook_token(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path)
    repository.insert_job(
        job_id="job_1",
        order_id="order-1",
        file_id="file-1",
        prompt="prompt",
        pattern=None,
        masters={"shipCsv": "a", "itemCsv": "b"},
        webhook={"url": "https://example.com", "token": "secret-token"},
        gemini=None,
        options=None,
        idempotency_key="order-1",
    )

    row = repository.get_job("job_1")
    assert row is not None
    assert row["webhook_token"] == "secret-token"

    detail = repository.get_job_detail("job_1")
    assert detail is not None
    assert detail["webhookToken"] == "secret-token"

    repository.close()


def test_insert_job_requires_webhook_token(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path)

    with pytest.raises(ValueError):
        repository.insert_job(
            job_id="job_2",
            order_id="order-2",
            file_id="file-2",
            prompt="prompt",
            pattern=None,
            masters={"shipCsv": "a", "itemCsv": "b"},
            webhook={"url": "https://example.com", "token": "   "},
            gemini=None,
            options=None,
            idempotency_key="order-2",
        )

    repository.close()


def test_repository_creates_database_file_if_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "custom.db"
    assert not db_path.exists()

    repository = JobRepository(db_path)

    assert db_path.exists()
    repository.close()


def test_record_and_list_gemini_logs(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path)
    repository.record_gemini_log(
        source="admin",
        prompt="first prompt",
        model="model-a",
        mime_type="text/plain",
        request={"prompt": "first prompt", "input": {"mode": "text"}},
        success=True,
        response_text="ok",
        meta={"tokens": 10},
        error=None,
    )
    repository.record_gemini_log(
        source="worker",
        prompt="second prompt",
        model="model-b",
        mime_type="application/pdf",
        request={"prompt": "second prompt", "input": {"mode": "pdf_page"}},
        success=False,
        response_text=None,
        meta=None,
        error="boom",
    )

    logs = repository.list_gemini_logs()
    assert len(logs) == 2
    assert logs[0]["source"] == "worker"
    assert logs[0]["success"] is False
    assert logs[0]["request"]["prompt"] == "second prompt"
    assert logs[1]["prompt_preview"].startswith("first prompt")

    repository.close()


def test_cancel_job_and_recent_jobs(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path)
    repository.insert_job(
        job_id="job_cancel",
        order_id="order-cancel",
        file_id="file-cancel",
        prompt="prompt",
        pattern=None,
        masters={"shipCsv": "a", "itemCsv": "b"},
        webhook={"url": "https://example.com", "token": "token"},
        gemini=None,
        options=None,
        idempotency_key="order-cancel",
    )
    repository.mark_enqueued("job_cancel")

    assert repository.cancel_job("job_cancel", reason="stop it") is True
    row = repository.get_job("job_cancel")
    assert row is not None
    assert row["status"] == JobStatus.CANCELLED.value
    assert row["last_error"] == "stop it"
    assert repository.get_job_status("job_cancel") == JobStatus.CANCELLED.value
    assert repository.cancel_job("job_cancel") is False

    repository.insert_job(
        job_id="job_second",
        order_id="order-2",
        file_id="file-2",
        prompt="prompt",
        pattern=None,
        masters={"shipCsv": "a", "itemCsv": "b"},
        webhook={"url": "https://example.com", "token": "token"},
        gemini=None,
        options=None,
        idempotency_key="order-2",
    )
    repository.update_job_status("job_second", JobStatus.PROCESSING)
    repository.update_job_status("job_second", JobStatus.DONE)

    jobs = repository.list_recent_jobs()
    assert jobs
    job_ids = {job["job_id"] for job in jobs[:2]}
    assert job_ids == {"job_second", "job_cancel"}
    assert isinstance(jobs[0]["updated_at"], datetime)
    repository.close()
