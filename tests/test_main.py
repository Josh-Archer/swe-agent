from unittest.mock import patch

from fastapi.testclient import TestClient
from src.main import app

# The TestClient runs the app in-process and handles the event loop,
# so we can use standard `def` test functions.
client = TestClient(app)


def test_read_root():
    """
    Test the liveness / health check endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_model_is_pulled():
    """
    /ready returns 200 when ollama.show succeeds for the configured model.
    """
    with patch("src.main.ollama.show", return_value={"modelfile": "FROM llama2"}) as mock_show:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert "model" in body
        mock_show.assert_called_once()


def test_ready_when_model_is_missing():
    """
    /ready returns 503 when the model has not been pulled.
    """
    with patch("src.main.ollama.show", side_effect=Exception("model not found")):
        response = client.get("/ready")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "not available" in detail


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


# Note: Testing the /api/generate endpoint would require mocking the ollama library,
# as it's not guaranteed to be available in a CI/CD environment.
# For this initial setup, we will skip that test.
