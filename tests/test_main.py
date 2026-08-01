"""Tests for health, feedback, API authentication, and streaming generation."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.main as main


@pytest.fixture
def client():
    """App client with authentication disabled (API_KEY unset)."""
    original = main.API_KEY
    main.API_KEY = ""
    with TestClient(main.app) as c:
        yield c
    main.API_KEY = original


@pytest.fixture
def authed_client():
    """App client with a configured API key."""
    original = main.API_KEY
    main.API_KEY = "test-secret-key"
    with TestClient(main.app) as c:
        yield c
    main.API_KEY = original


def test_read_root(client):
    """Health check is always public."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feedback_endpoint(client):
    """Feedback accepts valid payloads when auth is disabled."""
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


# --- Authentication ---


def test_auth_missing_key_rejected(authed_client):
    """Protected endpoints require a key when API_KEY is set."""
    response = authed_client.post(
        "/api/feedback",
        json={
            "prompt": "p",
            "response": "r",
            "is_correct": True,
        },
    )
    assert response.status_code == 401
    assert "API key" in response.json()["detail"]


def test_auth_wrong_key_rejected(authed_client):
    response = authed_client.post(
        "/api/feedback",
        headers={"X-API-Key": "wrong"},
        json={
            "prompt": "p",
            "response": "r",
            "is_correct": True,
        },
    )
    assert response.status_code == 401


def test_auth_x_api_key_accepted(authed_client):
    response = authed_client.post(
        "/api/feedback",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "prompt": "p",
            "response": "r",
            "is_correct": True,
        },
    )
    assert response.status_code == 200


def test_auth_bearer_token_accepted(authed_client):
    response = authed_client.post(
        "/api/feedback",
        headers={"Authorization": "Bearer test-secret-key"},
        json={
            "prompt": "p",
            "response": "r",
            "is_correct": True,
        },
    )
    assert response.status_code == 200


def test_health_remains_public_when_auth_enabled(authed_client):
    response = authed_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_requires_auth_when_configured(authed_client):
    response = authed_client.post(
        "/api/generate",
        json={"prompt": "hello"},
    )
    assert response.status_code == 401


# --- Non-streaming generate ---


def test_generate_non_streaming(client):
    mock_response = {"message": {"content": "Hello from the model"}}
    with patch.object(main.ollama, "chat", return_value=mock_response) as mock_chat:
        response = client.post(
            "/api/generate",
            json={"prompt": "Hi", "model": "llama2", "stream": False},
        )
        assert response.status_code == 200
        assert response.json() == {"response": "Hello from the model"}
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "llama2"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]


def test_generate_ollama_error(client):
    with patch.object(main.ollama, "chat", side_effect=RuntimeError("ollama down")):
        response = client.post(
            "/api/generate",
            json={"prompt": "Hi"},
        )
        assert response.status_code == 500
        assert "ollama down" in response.json()["detail"]


# --- Streaming generate ---


def _fake_stream_chunks():
    yield {"message": {"content": "Hel"}, "done": False}
    yield {"message": {"content": "lo"}, "done": False}
    yield {"message": {"content": "!"}, "done": True}


def test_generate_streaming_sse(client):
    with patch.object(main.ollama, "chat", return_value=_fake_stream_chunks()) as mock_chat:
        response = client.post(
            "/api/generate",
            json={"prompt": "stream me", "stream": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        assert 'data: {"content": "Hel"}' in body
        assert 'data: {"content": "lo"}' in body
        assert 'data: {"content": "!"}' in body
        assert "event: done" in body
        assert 'data: {"done": true}' in body

        # stream=True must be forwarded to ollama
        assert mock_chat.call_args.kwargs.get("stream") is True


def test_generate_streaming_requires_auth(authed_client):
    response = authed_client.post(
        "/api/generate",
        json={"prompt": "stream me", "stream": True},
    )
    assert response.status_code == 401


def test_generate_streaming_with_auth(authed_client):
    with patch.object(main.ollama, "chat", return_value=_fake_stream_chunks()):
        response = authed_client.post(
            "/api/generate",
            headers={"X-API-Key": "test-secret-key"},
            json={"prompt": "stream me", "stream": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert 'data: {"content": "Hel"}' in response.text


def test_generate_streaming_error_event(client):
    def boom(**_kwargs):
        raise RuntimeError("stream failed")

    with patch.object(main.ollama, "chat", side_effect=boom):
        response = client.post(
            "/api/generate",
            json={"prompt": "x", "stream": True},
        )
        assert response.status_code == 200  # SSE opens; error is in-band
        assert "event: error" in response.text
        assert "stream failed" in response.text
