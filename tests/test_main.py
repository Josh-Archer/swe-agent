import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.storage import (
    FileFeedbackStorage,
    FeedbackRecord,
    SqliteFeedbackStorage,
    create_feedback_storage,
)


# --- Storage unit tests ---

def test_sqlite_storage_save_and_get(tmp_path: Path):
    storage = SqliteFeedbackStorage(tmp_path / "feedback.db")
    record = FeedbackRecord(
        feedback_id="abc123",
        prompt="What is 2+2?",
        response="4",
        is_correct=True,
        correction=None,
    )
    storage.save(record)
    loaded = storage.get("abc123")
    assert loaded is not None
    assert loaded.feedback_id == "abc123"
    assert loaded.prompt == "What is 2+2?"
    assert loaded.response == "4"
    assert loaded.is_correct is True
    assert loaded.correction is None
    assert storage.get("missing") is None


def test_file_storage_save_and_get(tmp_path: Path):
    storage = FileFeedbackStorage(tmp_path / "feedback")
    record = FeedbackRecord(
        feedback_id="def456",
        prompt="Capital of France?",
        response="Lyon",
        is_correct=False,
        correction="Paris",
    )
    storage.save(record)
    loaded = storage.get("def456")
    assert loaded is not None
    assert loaded.feedback_id == "def456"
    assert loaded.is_correct is False
    assert loaded.correction == "Paris"
    assert storage.get("missing") is None


def test_create_feedback_storage_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = str(tmp_path / "custom.db")
    monkeypatch.setenv("FEEDBACK_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("FEEDBACK_STORAGE_PATH", db_path)
    storage = create_feedback_storage()
    assert isinstance(storage, SqliteFeedbackStorage)
    storage.save(
        FeedbackRecord(
            feedback_id="id1",
            prompt="p",
            response="r",
            is_correct=True,
        )
    )
    assert storage.get("id1") is not None


def test_create_feedback_storage_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dir_path = str(tmp_path / "json_feedback")
    monkeypatch.setenv("FEEDBACK_STORAGE_BACKEND", "file")
    monkeypatch.setenv("FEEDBACK_STORAGE_PATH", dir_path)
    storage = create_feedback_storage()
    assert isinstance(storage, FileFeedbackStorage)
    storage.save(
        FeedbackRecord(
            feedback_id="id2",
            prompt="p",
            response="r",
            is_correct=False,
            correction="fixed",
        )
    )
    assert storage.get("id2") is not None


def test_create_feedback_storage_unknown_backend():
    with pytest.raises(ValueError, match="Unknown FEEDBACK_STORAGE_BACKEND"):
        create_feedback_storage(backend="redis")


# --- API tests (isolated temp storage per module) ---

@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App TestClient backed by a fresh SQLite store in tmp_path."""
    db_path = str(tmp_path / "test_feedback.db")
    monkeypatch.setenv("FEEDBACK_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("FEEDBACK_STORAGE_PATH", db_path)

    # Re-import / rebind storage after env is set so the app uses temp DB.
    import src.main as main_module
    from src.storage import create_feedback_storage as factory

    main_module.feedback_storage = factory()
    return TestClient(main_module.app)


@pytest.fixture
def file_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App TestClient backed by a file (JSON) store in tmp_path."""
    dir_path = str(tmp_path / "feedback_files")
    monkeypatch.setenv("FEEDBACK_STORAGE_BACKEND", "file")
    monkeypatch.setenv("FEEDBACK_STORAGE_PATH", dir_path)

    import src.main as main_module
    from src.storage import create_feedback_storage as factory

    main_module.feedback_storage = factory()
    return TestClient(main_module.app)


def test_read_root(client: TestClient):
    """
    Test the health check endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feedback_endpoint_persists_and_retrieves(client: TestClient):
    """
    POST /api/feedback stores the record; GET /api/feedback/{id} returns it.
    """
    feedback_data = {
        "prompt": "What is the capital of France?",
        "response": "Paris",
        "is_correct": True,
    }
    response = client.post("/api/feedback", json=feedback_data)

    assert response.status_code == 200
    json_response = response.json()
    assert "message" in json_response
    assert "feedback_id" in json_response
    assert json_response["message"] == "Feedback received successfully. Thank you!"
    feedback_id = json_response["feedback_id"]
    assert feedback_id

    get_response = client.get(f"/api/feedback/{feedback_id}")
    assert get_response.status_code == 200
    detail = get_response.json()
    assert detail["feedback_id"] == feedback_id
    assert detail["prompt"] == feedback_data["prompt"]
    assert detail["response"] == feedback_data["response"]
    assert detail["is_correct"] is True
    assert detail["correction"] is None


def test_feedback_with_correction(client: TestClient):
    feedback_data = {
        "prompt": "What is 2+2?",
        "response": "5",
        "is_correct": False,
        "correction": "4",
    }
    response = client.post("/api/feedback", json=feedback_data)
    assert response.status_code == 200
    feedback_id = response.json()["feedback_id"]

    detail = client.get(f"/api/feedback/{feedback_id}").json()
    assert detail["is_correct"] is False
    assert detail["correction"] == "4"


def test_get_feedback_not_found(client: TestClient):
    response = client.get("/api/feedback/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "Feedback not found"


def test_feedback_file_backend_roundtrip(file_client: TestClient):
    """Same API flow works with the file storage backend."""
    feedback_data = {
        "prompt": "Hello?",
        "response": "Hi!",
        "is_correct": True,
    }
    response = file_client.post("/api/feedback", json=feedback_data)
    assert response.status_code == 200
    feedback_id = response.json()["feedback_id"]

    detail = file_client.get(f"/api/feedback/{feedback_id}").json()
    assert detail["prompt"] == "Hello?"
    assert detail["response"] == "Hi!"


# Note: Testing the /api/generate endpoint would require mocking the ollama library,
# as it's not guaranteed to be available in a CI/CD environment.
# For this initial setup, we will skip that test.
