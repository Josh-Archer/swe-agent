import logging
import os
from typing import NoReturn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import ollama
from ollama import RequestError, ResponseError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="An API for interacting with an AI agent and providing feedback.",
    version="0.1.0",
)

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    model: str = os.getenv("OLLAMA_MODEL", "llama2") # Default model from env or hardcoded

class GenerateResponse(BaseModel):
    response: str

class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    is_correct: bool
    correction: str | None = None

class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str # A unique ID for the feedback record


def _is_model_not_found(error: ResponseError) -> bool:
    """Return True when Ollama indicates the requested model is missing."""
    message = (error.error or str(error)).lower()
    if error.status_code == 404:
        return True
    return "not found" in message or ("model" in message and "pull" in message)


def _raise_mapped_ollama_error(exc: Exception, *, model: str) -> NoReturn:
    """
    Map Ollama/client failures to clear HTTP status codes and log structured detail.

    - 503: Ollama unreachable / connection / timeout
    - 400: bad client input (missing model, model not found, request errors)
    - 502: other upstream Ollama response failures
    - 500: unexpected internal errors
    """
    if isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.TimeoutException)):
        logger.error(
            "ollama_unavailable",
            extra={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model": model,
                "http_status": 503,
            },
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama service is unavailable. "
                "Check that Ollama is running and reachable."
            ),
        ) from exc

    if isinstance(exc, RequestError):
        logger.error(
            "ollama_request_error",
            extra={
                "error_type": "RequestError",
                "error": str(exc),
                "model": model,
                "http_status": 400,
            },
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ollama request: {exc.error or str(exc)}",
        ) from exc

    if isinstance(exc, ResponseError):
        if _is_model_not_found(exc):
            status = 400
            detail = (
                f"Model '{model}' was not found on the Ollama server. "
                f"Pull the model or choose an available one. "
                f"Upstream: {exc.error or str(exc)}"
            )
            log_event = "ollama_model_not_found"
        else:
            status = 502
            detail = (
                f"Ollama upstream error while generating with model '{model}': "
                f"{exc.error or str(exc)}"
            )
            log_event = "ollama_upstream_error"

        logger.error(
            log_event,
            extra={
                "error_type": "ResponseError",
                "error": str(exc),
                "upstream_status": exc.status_code,
                "model": model,
                "http_status": status,
            },
        )
        raise HTTPException(status_code=status, detail=detail) from exc

    logger.exception(
        "ollama_unexpected_error",
        extra={
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model": model,
            "http_status": 500,
        },
    )
    raise HTTPException(
        status_code=500,
        detail=f"Unexpected error generating response: {type(exc).__name__}",
    ) from exc


# --- API Endpoints ---

@app.get("/", summary="Health Check", description="Check if the API is running.")
def read_root():
    """
    Root endpoint to check if the API is up and running.
    """
    return {"status": "ok"}


@app.post("/api/generate", response_model=GenerateResponse, summary="Generate Agent Response")
async def generate(request: GenerateRequest):
    """
    Generates a response from the specified Ollama model based on the prompt.

    - **prompt**: The input text to the model.
    - **model**: The name of the Ollama model to use (e.g., 'llama2', 'codellama').
    """
    try:
        ollama_response = ollama.chat(
            model=request.model,
            messages=[{'role': 'user', 'content': request.prompt}]
        )
        response_content = ollama_response['message']['content']
        return GenerateResponse(response=response_content)
    except Exception as e:
        _raise_mapped_ollama_error(e, model=request.model)


@app.post("/api/feedback", response_model=FeedbackResponse, summary="Submit Feedback")
async def feedback(request: FeedbackRequest):
    """
    Submits feedback on a generated response. This data can be used later
    for fine-tuning the model.

    - **prompt**: The original prompt.
    - **response**: The response generated by the agent.
    - **is_correct**: A boolean indicating if the response was correct.
    - **correction**: (Optional) The corrected version of the response.
    """
    # For now, we'll just print the feedback.
    # In a real application, you would save this to a database or a file.
    feedback_id = os.urandom(16).hex()
    print(f"Received feedback ({feedback_id}):")
    print(f"  Prompt: {request.prompt}")
    print(f"  Response: {request.response}")
    print(f"  Correct: {request.is_correct}")
    if request.correction:
        print(f"  Correction: {request.correction}")

    # You would typically save this to a database or a message queue
    # for further processing.
    return FeedbackResponse(
        message="Feedback received successfully. Thank you!",
        feedback_id=feedback_id
    )
