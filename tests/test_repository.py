import sys
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.environ.setdefault("RELAY_TOKEN", "test-token")

import pytest

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
