"""Pluggable durable storage backends for feedback records."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FeedbackRecord:
    """A single feedback record stored durably."""

    feedback_id: str
    prompt: str
    response: str
    is_correct: bool
    correction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackRecord":
        return cls(
            feedback_id=data["feedback_id"],
            prompt=data["prompt"],
            response=data["response"],
            is_correct=bool(data["is_correct"]),
            correction=data.get("correction"),
        )


class FeedbackStorage(ABC):
    """Abstract interface for pluggable feedback storage backends."""

    @abstractmethod
    def save(self, record: FeedbackRecord) -> None:
        """Persist a feedback record."""

    @abstractmethod
    def get(self, feedback_id: str) -> FeedbackRecord | None:
        """Retrieve a feedback record by ID, or None if not found."""


class FileFeedbackStorage(FeedbackStorage):
    """Store each feedback record as a JSON file under a directory."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, feedback_id: str) -> Path:
        # Keep filenames simple and filesystem-safe (ids are hex).
        return self._directory / f"{feedback_id}.json"

    def save(self, record: FeedbackRecord) -> None:
        path = self._path_for(record.feedback_id)
        payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False)
        with self._lock:
            path.write_text(payload, encoding="utf-8")

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        path = self._path_for(feedback_id)
        if not path.is_file():
            return None
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return FeedbackRecord.from_dict(data)


class SqliteFeedbackStorage(FeedbackStorage):
    """Store feedback records in a SQLite database file."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        feedback_id TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        is_correct INTEGER NOT NULL,
                        correction TEXT
                    )
                    """
                )
                conn.commit()

    def save(self, record: FeedbackRecord) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feedback
                        (feedback_id, prompt, response, is_correct, correction)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.feedback_id,
                        record.prompt,
                        record.response,
                        1 if record.is_correct else 0,
                        record.correction,
                    ),
                )
                conn.commit()

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT feedback_id, prompt, response, is_correct, correction "
                    "FROM feedback WHERE feedback_id = ?",
                    (feedback_id,),
                ).fetchone()
        if row is None:
            return None
        return FeedbackRecord(
            feedback_id=row["feedback_id"],
            prompt=row["prompt"],
            response=row["response"],
            is_correct=bool(row["is_correct"]),
            correction=row["correction"],
        )


def create_feedback_storage(
    backend: str | None = None,
    path: str | None = None,
) -> FeedbackStorage:
    """
    Factory for feedback storage backends.

    Environment variables (used when arguments are omitted):
      FEEDBACK_STORAGE_BACKEND: "sqlite" (default) or "file"
      FEEDBACK_STORAGE_PATH: path to SQLite file or JSON directory
        defaults to data/feedback.db (sqlite) or data/feedback (file)
    """
    backend = (backend or os.getenv("FEEDBACK_STORAGE_BACKEND", "sqlite")).lower()
    default_path = (
        "data/feedback.db" if backend == "sqlite" else "data/feedback"
    )
    path = path or os.getenv("FEEDBACK_STORAGE_PATH", default_path)

    if backend == "sqlite":
        return SqliteFeedbackStorage(path)
    if backend == "file":
        return FileFeedbackStorage(path)
    raise ValueError(
        f"Unknown FEEDBACK_STORAGE_BACKEND '{backend}'. "
        "Supported values: 'sqlite', 'file'."
    )
