import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Provide a TestClient backed by an isolated feedback store so tests
    do not touch the default data/ directory.
    """
    store_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_STORE_PATH", str(store_path))

    # Re-import after env override so FEEDBACK_STORE_PATH is set for this test.
    import importlib
    import src.main as main_module

    importlib.reload(main_module)
    main_module.FEEDBACK_STORE_PATH = Path(store_path)

    with TestClient(main_module.app) as test_client:
        yield test_client, main_module, store_path


def test_read_root(client):
    """
    Test the health check endpoint.
    """
    test_client, _, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feedback_endpoint(client):
    """
    Test the /api/feedback endpoint accepts valid feedback and returns an id.
    """
    test_client, _, _ = client
    feedback_data = {
        "prompt": "What is the capital of France?",
        "response": "Paris",
        "is_correct": True,
    }
    response = test_client.post("/api/feedback", json=feedback_data)

    assert response.status_code == 200
    json_response = response.json()
    assert "message" in json_response
    assert "feedback_id" in json_response
    assert json_response["message"] == "Feedback received successfully. Thank you!"


def test_feedback_is_persisted(client):
    """
    Feedback must be written to durable storage, not only printed.
    """
    test_client, main_module, store_path = client
    feedback_data = {
        "prompt": "Write a hello world function",
        "response": "def hello(): print('hi')",
        "is_correct": False,
        "correction": "def hello(): print('hello world')",
    }
    response = test_client.post("/api/feedback", json=feedback_data)
    assert response.status_code == 200
    feedback_id = response.json()["feedback_id"]

    assert store_path.exists(), "feedback store file should be created"
    stored = main_module.load_feedback(feedback_id)
    assert stored is not None, "feedback record must be loadable by id"
    assert stored["feedback_id"] == feedback_id
    assert stored["prompt"] == feedback_data["prompt"]
    assert stored["response"] == feedback_data["response"]
    assert stored["is_correct"] is False
    assert stored["correction"] == feedback_data["correction"]


def test_feedback_persistence_survives_reload(client):
    """
    Persisted feedback remains available after reloading the module (new process).
    """
    test_client, main_module, store_path = client
    feedback_data = {
        "prompt": "2+2?",
        "response": "4",
        "is_correct": True,
    }
    response = test_client.post("/api/feedback", json=feedback_data)
    feedback_id = response.json()["feedback_id"]

    # Simulate a new process reading the same store path.
    reloaded = main_module.load_feedback(feedback_id, path=store_path)
    assert reloaded is not None
    assert reloaded["prompt"] == "2+2?"
    assert reloaded["is_correct"] is True


# Note: Testing the /api/generate endpoint would require mocking the ollama library,
# as it's not guaranteed to be available in a CI/CD environment.
# For this initial setup, we will skip that test.
