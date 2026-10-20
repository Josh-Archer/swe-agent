import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import ollama
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.agent_loop import job_store, resume_queued_jobs, run_resolve_job
from src.storage import FeedbackRecord, create_feedback_storage
from src.tools import Tools, _default_sandbox_root, host_mount_allowed

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Paths that run the model or the tool-using agent. The Service is a
# LoadBalancer, so these reject unauthenticated callers even when API_KEY
# is unset (fail closed). Health probes stay public.
_PROTECTED_PATHS = frozenset(
    {"/api/resolve", "/api/generate", "/api/feedback"}
)
MAX_FEEDBACK_BODY_BYTES = 64 * 1024
_AUTH_FAILED = "Invalid or missing API key"

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key from the API_KEY environment variable.",
)
bearer_scheme = HTTPBearer(auto_error=False, description="Bearer API key.")

_missing_api_key_logged = False


def _configured_api_key() -> str:
    """Return the server API key, or empty when it is unset or blank."""
    return os.getenv("API_KEY", "").strip()


def _warn_missing_api_key() -> None:
    """Log once when the process has no API_KEY so operators can see why."""
    global _missing_api_key_logged
    if _missing_api_key_logged:
        return
    _missing_api_key_logged = True
    logger.warning(
        "API_KEY is not set; rejecting unauthenticated calls to %s",
        ", ".join(sorted(_PROTECTED_PATHS)),
    )


def _extract_api_key(
    x_api_key: str | None,
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    """Prefer a non-empty X-API-Key; otherwise a Bearer token."""
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials.strip()
        if token:
            return token
    return None


def _bearer_credentials(request: Request) -> HTTPAuthorizationCredentials | None:
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    if not token:
        return None
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def _provided_api_key(request: Request) -> str | None:
    return _extract_api_key(
        request.headers.get("x-api-key"),
        _bearer_credentials(request),
    )


def _api_keys_match(provided: str, expected: str) -> bool:
    """Constant-time compare that does not leak the key length."""
    provided_digest = hashlib.sha256(provided.encode("utf-8")).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(provided_digest, expected_digest)


def ensure_api_key(request: Request) -> None:
    """
    Reject the request unless it presents the configured API key.

    Accepted credentials (header only; query parameters are ignored):

    - ``X-API-Key: <API_KEY>``
    - ``Authorization: Bearer <API_KEY>``

    When ``API_KEY`` is unset or blank, every protected call is rejected so a
    public LoadBalancer cannot serve these routes anonymously.
    """
    expected = _configured_api_key()
    if not expected:
        _warn_missing_api_key()
    provided = _provided_api_key(request)
    if not expected or not provided or not _api_keys_match(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_AUTH_FAILED,
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Security(api_key_header)] = None,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(bearer_scheme)
    ] = None,
) -> None:
    """
    FastAPI dependency for protected routes.

    ``Security`` parameters publish ``X-API-Key`` and HTTP Bearer in OpenAPI.
    The decision uses the raw request so it matches the middleware.
    """
    del x_api_key, credentials
    ensure_api_key(request)


# Initialize the FastAPI app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Resume jobs persisted before a process restart."""
    resume_queued_jobs()
    yield


app = FastAPI(
    title="AI Agent API",
    description=(
        "An API for interacting with an AI agent, submitting feedback, "
        "and running a tool-using issue-resolution loop in a sandbox. "
        "POST /api/resolve, /api/generate, and /api/feedback require "
        "an API key."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

feedback_storage = create_feedback_storage()


@app.middleware("http")
async def reject_unauthenticated_protected_routes(request: Request, call_next):
    """
    Fail closed before the body is parsed or a job/model call starts.

    Route dependencies enforce the same check; this middleware also rejects
    malformed payloads that would otherwise surface as 422.
    """
    if request.url.path in _PROTECTED_PATHS:
        try:
            ensure_api_key(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=dict(exc.headers) if exc.headers else None,
            )
    if request.url.path == "/api/feedback":
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdecimal()
            and int(content_length) > MAX_FEEDBACK_BODY_BYTES
        ):
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds 64 KB limit"},
            )
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_FEEDBACK_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds 64 KB limit"},
                )
        # BaseHTTPMiddleware replays the cached request body to the endpoint.
        request._body = bytes(body)
    return await call_next(request)


# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    model: str = os.getenv("OLLAMA_MODEL", "llama2")  # Default model from env or hardcoded


class GenerateResponse(BaseModel):
    response: str


class FeedbackRequest(BaseModel):
    prompt: str = Field(..., max_length=16_384)
    response: str = Field(..., max_length=32_768)
    is_correct: bool
    correction: str | None = Field(default=None, max_length=16_384)


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str  # A unique ID for the feedback record


class ResolveRequest(BaseModel):
    """Request to resolve a software issue via the tool-using agent loop."""

    issue: str = Field(..., description="Issue description or acceptance criteria.")
    repo_url: str | None = Field(
        default=None,
        description="Optional git repository URL to clone into the sandbox.",
    )
    git_ref: str | None = Field(
        default=None,
        description="Optional git ref (branch/tag/commit) to checkout after clone.",
    )
    max_steps: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Maximum agent loop iterations.",
    )
    test_command: str | None = Field(
        default=None,
        description="Optional test command the agent should prefer (e.g. pytest).",
    )
    model: str | None = Field(
        default=None,
        description="Ollama model override for this job.",
    )
    async_mode: bool = Field(
        default=False,
        description=(
            "If true, return a job id immediately and run the agent in the background. "
            "Poll GET /api/jobs/{job_id} for status."
        ),
    )


class ResolveResponse(BaseModel):
    job_id: str
    status: str
    request: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ToolsInfoResponse(BaseModel):
    tools: list[dict[str, str]]
    sandbox_root: str
    host_mount_allowed: bool
    notes: list[str]


# --- API Endpoints ---

@app.get("/", summary="Health Check", description="Check if the API is running.")
def read_root():
    """
    Root endpoint to check if the API is up and running.
    """
    return {"status": "ok"}


@app.get(
    "/ready",
    summary="Readiness Check",
    description="Confirm the configured Ollama model is available.",
)
def ready():
    """
    Readiness probe: verify the configured Ollama model has been pulled.
    """
    model = os.getenv("OLLAMA_MODEL", "llama2")
    try:
        ollama.show(model)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model!r} is not available: {exc}",
        ) from exc
    return {"status": "ready", "model": model}


@app.get(
    "/api/tools",
    response_model=ToolsInfoResponse,
    summary="List agent tools and sandbox safety defaults",
)
def list_tools():
    """
    Describe the minimal tool interface and current sandbox configuration.

    Safety defaults: workspaces are created under a temp sandbox root;
    host filesystem mounts are disabled unless AGENT_ALLOW_HOST_MOUNT=true.
    """
    return ToolsInfoResponse(
        tools=Tools.list_tools(),
        sandbox_root=str(_default_sandbox_root()),
        host_mount_allowed=host_mount_allowed(),
        notes=[
            (
                "No host mount by default: workspaces live under AGENT_SANDBOX_ROOT "
                "(or a temp directory)."
            ),
            "File paths are confined to the workspace; traversal outside is rejected.",
            (
                "Commands run with workspace cwd and a hard timeout "
                "(AGENT_COMMAND_TIMEOUT, default 60s)."
            ),
            (
                "This is application-level isolation; deploy in containers without "
                "hostPath volumes for stronger boundaries."
            ),
            "Set AGENT_ALLOW_HOST_MOUNT=true only when intentionally binding a host path.",
        ],
    )


@app.post(
    "/api/resolve",
    response_model=ResolveResponse,
    summary="Resolve an issue with the tool-using agent loop",
    dependencies=[Depends(require_api_key)],
)
def resolve_issue(request: ResolveRequest):
    """
    Run (or enqueue) an issue-resolution job.

    The agent may checkout a repository into a sandbox, use tools
    (run_command, read_file, write_file), run tests, and produce a patch.

    - **async_mode=false** (default): runs inline and returns the finished job.
    - **async_mode=true**: returns immediately with status ``queued``/``running``;
      poll ``GET /api/jobs/{job_id}``.

    Requires ``X-API-Key`` or ``Authorization: Bearer`` matching ``API_KEY``.
    """
    try:
        record = run_resolve_job(
            issue=request.issue,
            repo_url=request.repo_url,
            git_ref=request.git_ref,
            max_steps=request.max_steps,
            test_command=request.test_command,
            model=request.model,
            async_mode=request.async_mode,
        )
        return ResolveResponse(
            job_id=record["job_id"],
            status=record["status"],
            request=record.get("request"),
            result=record.get("result"),
            error=record.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/api/jobs/{job_id}",
    response_model=ResolveResponse,
    summary="Get status of a resolve job",
)
def get_job(job_id: str):
    """Fetch a previously submitted issue-resolution job by id."""
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return ResolveResponse(
        job_id=record["job_id"],
        status=record["status"],
        request=record.get("request"),
        result=record.get("result"),
        error=record.get("error"),
    )


@app.get(
    "/api/jobs",
    summary="List resolve jobs",
)
def list_jobs():
    """List persisted resolve jobs."""
    return {"jobs": job_store.list_jobs()}


@app.post(
    "/api/generate",
    response_model=GenerateResponse,
    summary="Generate Agent Response",
    dependencies=[Depends(require_api_key)],
)
async def generate(request: GenerateRequest):
    """
    Generates a response from the specified Ollama model based on the prompt.

    - **prompt**: The input text to the model.
    - **model**: The name of the Ollama model to use (e.g., 'llama2', 'codellama').

    Requires ``X-API-Key`` or ``Authorization: Bearer`` matching ``API_KEY``.
    """
    try:
        ollama_response = ollama.chat(
            model=request.model,
            messages=[{'role': 'user', 'content': request.prompt}]
        )
        response_content = ollama_response['message']['content']
        return GenerateResponse(response=response_content)
    except Exception as e:  # noqa: BLE001
        # Broad exception for now, can be refined
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse, summary="Submit Feedback")
def feedback(request: FeedbackRequest):
    """
    Submits feedback on a generated response. This data can be used later
    for fine-tuning the model.

    - **prompt**: The original prompt.
    - **response**: The response generated by the agent.
    - **is_correct**: A boolean indicating if the response was correct.
    - **correction**: (Optional) The corrected version of the response.
    """
    feedback_id = os.urandom(16).hex()
    feedback_storage.save(
        FeedbackRecord(
            feedback_id=feedback_id,
            prompt=request.prompt,
            response=request.response,
            is_correct=request.is_correct,
            correction=request.correction,
        )
    )
    return FeedbackResponse(
        message="Feedback received successfully. Thank you!",
        feedback_id=feedback_id
    )
