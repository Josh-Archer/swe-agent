from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from ollama import RequestError, ResponseError

from src.main import app

# The TestClient runs the app in-process and handles the event loop,
# so we can use standard `def` test functions.
client = TestClient(app)

GENERATE_PAYLOAD = {
    "prompt": "What is the capital of France?",
    "model": "llama2",
}


def test_read_root():
    """
    Test the health check endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feedback_endpoint():
    """
    Test the /api/feedback endpoint to ensure it accepts valid feedback.
    The TestClient handles the async nature of the endpoint.
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


def test_generate_success():
    """Successful Ollama chat returns 200 with response text."""
    mock_reply = {
        "message": {"role": "assistant", "content": "Paris"},
    }
    with patch("src.main.ollama.chat", return_value=mock_reply) as mock_chat:
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"response": "Paris"}
    mock_chat.assert_called_once()


def test_generate_connection_error_returns_503():
    """Connection failures map to 503 Service Unavailable."""
    with patch(
        "src.main.ollama.chat",
        side_effect=ConnectionError("Failed to connect to Ollama"),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_generate_httpx_connect_error_returns_503():
    """httpx connection errors also map to 503."""
    with patch(
        "src.main.ollama.chat",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_generate_timeout_returns_503():
    """Timeouts map to 503 Service Unavailable."""
    with patch(
        "src.main.ollama.chat",
        side_effect=httpx.TimeoutException("timed out"),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_generate_request_error_returns_400():
    """Invalid Ollama requests map to 400 Bad Request."""
    with patch(
        "src.main.ollama.chat",
        side_effect=RequestError("model is required"),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 400
    assert "invalid ollama request" in response.json()["detail"].lower()


def test_generate_model_not_found_returns_400():
    """Missing model (404 / not found message) maps to 400."""
    with patch(
        "src.main.ollama.chat",
        side_effect=ResponseError("model 'llama2' not found", status_code=404),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "not found" in detail
    assert "llama2" in detail


def test_generate_upstream_response_error_returns_502():
    """Other Ollama ResponseErrors map to 502 Bad Gateway."""
    with patch(
        "src.main.ollama.chat",
        side_effect=ResponseError("internal server error", status_code=500),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 502
    detail = response.json()["detail"].lower()
    assert "upstream" in detail
    assert "internal server error" in detail


def test_generate_unexpected_error_returns_500():
    """Unexpected exceptions still return 500 without leaking full internals only type."""
    with patch(
        "src.main.ollama.chat",
        side_effect=RuntimeError("boom"),
    ):
        response = client.post("/api/generate", json=GENERATE_PAYLOAD)

    assert response.status_code == 500
    assert "RuntimeError" in response.json()["detail"]
