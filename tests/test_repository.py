import json
import os
import sqlite3
import sys
from pathlib import Path

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


def test_migration_renames_webhook_secret_column(tmp_path: Path) -> None:
    db_path = tmp_path / "relay.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            file_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            pattern TEXT,
            masters_json TEXT NOT NULL,
            webhook_url TEXT NOT NULL,
            webhook_secret TEXT NOT NULL,
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
        """
    )
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, order_id, file_id, prompt, pattern, masters_json,
            webhook_url, webhook_secret, gemini_json, options_json,
            idempotency_key, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job_legacy",
            "legacy-order",
            "legacy-file",
            "legacy-prompt",
            None,
            json.dumps({"shipCsv": "a", "itemCsv": "b"}),
            "https://example.com",
            "legacy-secret",
            None,
            None,
            "legacy-order",
            "RECEIVED",
        ),
    )
    conn.commit()
    conn.close()

    repository = JobRepository(db_path)
    row = repository.get_job("job_legacy")
    assert row is not None
    assert "webhook_token" in row.keys()
    assert row["webhook_token"] == "legacy-secret"

    repository.close()


def test_migration_recreates_table_when_column_rename_not_supported(tmp_path: Path) -> None:
    db_path = tmp_path / "relay.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            file_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            pattern TEXT,
            masters_json TEXT NOT NULL,
            webhook_url TEXT NOT NULL,
            webhook_secret TEXT NOT NULL,
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
        """
    )
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, order_id, file_id, prompt, pattern, masters_json,
            webhook_url, webhook_secret, gemini_json, options_json,
            idempotency_key, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job_legacy",
            "legacy-order",
            "legacy-file",
            "legacy-prompt",
            None,
            json.dumps({"shipCsv": "a", "itemCsv": "b"}),
            "https://example.com",
            "legacy-secret",
            None,
            None,
            "legacy-order",
            "RECEIVED",
        ),
    )
    conn.commit()
    conn.close()

    class RenameFailingConnection:
        def __init__(self, inner: sqlite3.Connection) -> None:
            object.__setattr__(self, "_inner", inner)

        def execute(self, sql: str, *args, **kwargs):  # type: ignore[override]
            if "ALTER TABLE jobs RENAME COLUMN webhook_secret TO webhook_token" in sql:
                raise sqlite3.OperationalError("near \"COLUMN\": syntax error")
            return self._inner.execute(sql, *args, **kwargs)

        def executescript(self, sql: str):  # type: ignore[override]
            return self._inner.executescript(sql)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __setattr__(self, name, value):
            setattr(self._inner, name, value)

    class RenameFailJobRepository(JobRepository):
        @staticmethod
        def _connect(path: Path) -> sqlite3.Connection:  # type: ignore[override]
            base_conn = JobRepository._connect(path)
            return RenameFailingConnection(base_conn)

    repository = RenameFailJobRepository(db_path)
    row = repository.get_job("job_legacy")
    assert row is not None
    assert row["webhook_token"] == "legacy-secret"

    columns = {
        column["name"]
        for column in repository._conn.execute("PRAGMA table_info(jobs)").fetchall()  # type: ignore[attr-defined]
    }
    assert "webhook_secret" not in columns

    repository.close()


def test_migration_removes_legacy_webhook_secret_column(tmp_path: Path) -> None:
    db_path = tmp_path / "relay.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            file_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            pattern TEXT,
            masters_json TEXT NOT NULL,
            webhook_url TEXT NOT NULL,
            webhook_secret TEXT NOT NULL,
            webhook_token TEXT NOT NULL,
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
        """
    )
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, order_id, file_id, prompt, pattern, masters_json,
            webhook_url, webhook_secret, webhook_token, gemini_json, options_json,
            idempotency_key, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job_both",
            "order-both",
            "file-both",
            "prompt",
            None,
            json.dumps({"shipCsv": "a", "itemCsv": "b"}),
            "https://example.com",
            "legacy-secret",
            "",
            None,
            None,
            "order-both",
            "RECEIVED",
        ),
    )
    conn.commit()
    conn.close()

    repository = JobRepository(db_path)
    row = repository.get_job("job_both")
    assert row is not None
    assert row["webhook_token"] == "legacy-secret"

    columns = {
        column["name"]
        for column in repository._conn.execute("PRAGMA table_info(jobs)").fetchall()  # type: ignore[attr-defined]
    }
    assert "webhook_secret" not in columns

    repository.close()


def test_migration_drops_not_null_constraint_from_webhook_token(tmp_path: Path) -> None:
    db_path = tmp_path / "relay.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            file_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            pattern TEXT,
            masters_json TEXT NOT NULL,
            webhook_url TEXT NOT NULL,
            webhook_token TEXT NOT NULL,
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
        """
    )
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, order_id, file_id, prompt, pattern, masters_json,
            webhook_url, webhook_token, gemini_json, options_json,
            idempotency_key, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job_token",
            "order-token",
            "file-token",
            "prompt",
            None,
            json.dumps({"shipCsv": "a", "itemCsv": "b"}),
            "https://example.com",
            "token-value",
            None,
            None,
            "order-token",
            "RECEIVED",
        ),
    )
    conn.commit()
    conn.close()

    repository = JobRepository(db_path)
    columns = {
        column["name"]: column
        for column in repository._conn.execute("PRAGMA table_info(jobs)").fetchall()  # type: ignore[attr-defined]
    }
    assert columns["webhook_token"]["notnull"] == 0

    repository.close()


def test_repository_creates_database_file_if_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "custom.db"
    assert not db_path.exists()

    repository = JobRepository(db_path)

    assert db_path.exists()
    repository.close()
