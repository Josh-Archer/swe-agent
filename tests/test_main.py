import logging
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import main
from src.main import app
from src.storage import create_feedback_storage

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


def test_feedback_endpoint(tmp_path, monkeypatch):
    """
    Test the /api/feedback endpoint to ensure it accepts valid feedback.
    The TestClient handles the async nature of the endpoint.
    """
    storage_path = tmp_path / "feedback.db"
    monkeypatch.setattr(
        main,
        "feedback_storage",
        create_feedback_storage("sqlite", str(storage_path)),
    )
    feedback_data = {
        "prompt": "What is the capital of France?",
        "response": "Paris",
        "is_correct": False,
        "correction": "Lyon",
    }
    response = client.post("/api/feedback", json=feedback_data)

    assert response.status_code == 200
    json_response = response.json()
    assert "message" in json_response
    assert "feedback_id" in json_response
    assert json_response["message"] == (
        "Feedback received successfully. Thank you!"
    )
    record = create_feedback_storage("sqlite", str(storage_path)).get(
        json_response["feedback_id"]
    )
    assert record is not None
    assert record.prompt == feedback_data["prompt"]
    assert record.response == feedback_data["response"]
    assert record.is_correct is feedback_data["is_correct"]
    assert record.correction == feedback_data["correction"]


API_KEY = "test-secret-key"


def _auth_headers(scheme: str = "x-api-key", key: str = API_KEY) -> dict[str, str]:
    if scheme == "bearer":
        return {"Authorization": f"Bearer {key}"}
    return {"X-API-Key": key}


def test_resolve_and_generate_reject_missing_key(monkeypatch):
    """Protected routes reject callers that do not send a key."""
    monkeypatch.setenv("API_KEY", API_KEY)
    with patch("src.main.run_resolve_job") as mock_resolve, patch(
        "src.main.ollama.chat"
    ) as mock_chat:
        resolve = client.post("/api/resolve", json={"issue": "fix the tests"})
        generate = client.post("/api/generate", json={"prompt": "hello"})

    assert resolve.status_code == 401
    assert generate.status_code == 401
    assert resolve.json()["detail"] == "Invalid or missing API key"
    assert generate.json()["detail"] == "Invalid or missing API key"
    assert resolve.headers["www-authenticate"] == "Bearer"
    assert generate.headers["www-authenticate"] == "Bearer"
    mock_resolve.assert_not_called()
    mock_chat.assert_not_called()


def test_resolve_and_generate_reject_wrong_key(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    with patch("src.main.run_resolve_job") as mock_resolve, patch(
        "src.main.ollama.chat"
    ) as mock_chat:
        headers = _auth_headers(key="not-the-key")
        resolve = client.post(
            "/api/resolve", headers=headers, json={"issue": "fix the tests"}
        )
        generate = client.post(
            "/api/generate", headers=headers, json={"prompt": "hello"}
        )

    assert resolve.status_code == 401
    assert generate.status_code == 401
    mock_resolve.assert_not_called()
    mock_chat.assert_not_called()


def test_protected_routes_reject_when_api_key_unset(monkeypatch):
    """Fail closed: a LoadBalancer must not serve these routes anonymously."""
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("src.main.run_resolve_job") as mock_resolve, patch(
        "src.main.ollama.chat"
    ) as mock_chat:
        resolve = client.post(
            "/api/resolve",
            headers=_auth_headers(),
            json={"issue": "fix the tests"},
        )
        generate = client.post(
            "/api/generate",
            headers={"Authorization": "Bearer anything"},
            json={"prompt": "hello"},
        )

    assert resolve.status_code == 401
    assert generate.status_code == 401
    mock_resolve.assert_not_called()
    mock_chat.assert_not_called()


def test_blank_api_key_config_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY", "   \n")
    response = client.post(
        "/api/generate",
        headers=_auth_headers(key=" "),
        json={"prompt": "hello"},
    )
    assert response.status_code == 401


def test_query_string_api_key_is_not_accepted(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    response = client.post(
        f"/api/generate?api_key={API_KEY}",
        json={"prompt": "hello"},
    )
    assert response.status_code == 401


def test_malformed_body_without_key_is_unauthorized(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    response = client.post(
        "/api/resolve",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_x_api_key_rejected_when_bearer_would_match(monkeypatch):
    """A present X-API-Key is the credential, even if Bearer is correct."""
    monkeypatch.setenv("API_KEY", API_KEY)
    response = client.post(
        "/api/generate",
        headers={
            "X-API-Key": "wrong",
            "Authorization": f"Bearer {API_KEY}",
        },
        json={"prompt": "hello"},
    )
    assert response.status_code == 401


def test_basic_authorization_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    response = client.post(
        "/api/generate",
        headers={"Authorization": f"Basic {API_KEY}"},
        json={"prompt": "hello"},
    )
    assert response.status_code == 401


def test_generate_accepts_x_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", f"  {API_KEY}\n")
    mock_response = {"message": {"content": "Hello from the model"}}
    with patch("src.main.ollama.chat", return_value=mock_response) as mock_chat:
        response = client.post(
            "/api/generate",
            headers=_auth_headers(),
            json={"prompt": "Hi"},
        )

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the model"}
    mock_chat.assert_called_once()
    assert mock_chat.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Hi"}
    ]


def test_generate_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    mock_response = {"message": {"content": "bearer ok"}}
    with patch("src.main.ollama.chat", return_value=mock_response):
        response = client.post(
            "/api/generate",
            headers=_auth_headers(scheme="bearer"),
            json={"prompt": "Hi", "model": "llama2"},
        )

    assert response.status_code == 200
    assert response.json()["response"] == "bearer ok"


def test_resolve_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    record = {
        "job_id": "job-1",
        "status": "succeeded",
        "request": {"issue": "x"},
        "result": {"summary": "done"},
        "error": None,
    }
    with patch("src.main.run_resolve_job", return_value=record) as mock_resolve:
        response = client.post(
            "/api/resolve",
            headers={"Authorization": f"bearer {API_KEY}"},
            json={"issue": "x", "async_mode": False},
        )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert response.json()["status"] == "succeeded"
    mock_resolve.assert_called_once()


def test_public_routes_stay_open_when_api_key_is_set(monkeypatch):
    """Health, readiness, tools, jobs, and feedback do not require the key."""
    monkeypatch.setenv("API_KEY", API_KEY)
    assert client.get("/").status_code == 200
    with patch("src.main.ollama.show", return_value={"modelfile": "FROM llama2"}):
        assert client.get("/ready").status_code == 200
    assert client.get("/api/tools").status_code == 200
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/api/jobs/missing").status_code == 404
    feedback = client.post(
        "/api/feedback",
        json={"prompt": "p", "response": "r", "is_correct": True},
    )
    assert feedback.status_code == 200


def test_missing_api_key_is_logged_once(monkeypatch, caplog):
    monkeypatch.delenv("API_KEY", raising=False)
    main._missing_api_key_logged = False
    with caplog.at_level(logging.WARNING):
        client.post("/api/resolve", json={"issue": "x"})
        client.post("/api/generate", json={"prompt": "x"})
    warnings = [r for r in caplog.records if "API_KEY is not set" in r.message]
    assert len(warnings) == 1
    assert "/api/generate" in warnings[0].message
    assert "/api/resolve" in warnings[0].message
