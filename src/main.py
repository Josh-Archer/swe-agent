import json
import os
from collections.abc import Iterator
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import ollama

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# When API_KEY is unset/empty, the API is open (suitable for ClusterIP + NetworkPolicy).
# When set, clients must send X-API-Key or Authorization: Bearer <key>.
API_KEY = os.getenv("API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Initialize the FastAPI app
app = FastAPI(
    title="AI Agent API",
    description=(
        "An API for interacting with an AI agent and providing feedback. "
        "Supports streaming generation and optional API-key authentication."
    ),
    version="0.2.0",
)


# --- Auth ---
def _extract_api_key(
    x_api_key: str | None,
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    """Prefer X-API-Key; fall back to Bearer token."""
    if x_api_key:
        return x_api_key.strip()
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials.strip()
    return None


async def require_api_key(
    x_api_key: Annotated[str | None, Security(api_key_header)] = None,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(bearer_scheme)
    ] = None,
) -> None:
    """
    Enforce API key when API_KEY is configured.

    Network-policy-friendly: leave API_KEY unset and restrict access at the
    cluster network layer (ClusterIP + NetworkPolicy) instead.
    """
    if not API_KEY:
        return

    provided = _extract_api_key(x_api_key, credentials)
    if not provided or provided != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    stream: bool = Field(
        default=False,
        description=(
            "If true, respond with Server-Sent Events (text/event-stream) "
            "token chunks instead of a single JSON body."
        ),
    )


class GenerateResponse(BaseModel):
    response: str


class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    is_correct: bool
    correction: str | None = None


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str  # A unique ID for the feedback record


# --- Streaming helpers ---
def _sse_event(data: dict, event: str | None = None) -> str:
    """Format a single Server-Sent Event."""
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    lines.append("")  # blank line terminates the event
    return "\n".join(lines) + "\n"


def _stream_ollama_chat(model: str, prompt: str) -> Iterator[str]:
    """Yield SSE-framed chunks from Ollama's streaming chat API."""
    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield _sse_event({"content": content})
            if chunk.get("done"):
                yield _sse_event({"done": True}, event="done")
                return
        # Ensure clients always see a terminal event
        yield _sse_event({"done": True}, event="done")
    except Exception as exc:  # noqa: BLE001 — surface Ollama errors to client
        yield _sse_event({"error": str(exc)}, event="error")


# --- API Endpoints ---

@app.get("/", summary="Health Check", description="Check if the API is running.")
def read_root():
    """
    Root endpoint to check if the API is up and running.
    Always public (no API key) for liveness/readiness probes.
    """
    return {"status": "ok"}


@app.post(
    "/api/generate",
    response_model=None,
    summary="Generate Agent Response",
    dependencies=[Depends(require_api_key)],
)
async def generate(request: GenerateRequest):
    """
    Generates a response from the specified Ollama model based on the prompt.

    - **prompt**: The input text to the model.
    - **model**: The name of the Ollama model to use (e.g. 'llama2', 'codellama').
    - **stream**: When true, returns `text/event-stream` (SSE) with incremental
      `data: {"content": "..."}` events and a final `event: done`.
    """
    if request.stream:
        return StreamingResponse(
            _stream_ollama_chat(request.model, request.prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        ollama_response = ollama.chat(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
        )
        response_content = ollama_response["message"]["content"]
        return GenerateResponse(response=response_content)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    summary="Submit Feedback",
    dependencies=[Depends(require_api_key)],
)
async def feedback(request: FeedbackRequest):
    """
    Submits feedback on a generated response. This data can be used later
    for fine-tuning the model.

    - **prompt**: The original prompt.
    - **response**: The response generated by the agent.
    - **is_correct**: A boolean indicating if the response was correct.
    - **correction**: (Optional) The corrected version of the response.
    """
    feedback_id = os.urandom(16).hex()
    print(f"Received feedback ({feedback_id}):")
    print(f"  Prompt: {request.prompt}")
    print(f"  Response: {request.response}")
    print(f"  Correct: {request.is_correct}")
    if request.correction:
        print(f"  Correction: {request.correction}")

    return FeedbackResponse(
        message="Feedback received successfully. Thank you!",
        feedback_id=feedback_id,
    )
