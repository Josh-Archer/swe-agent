import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import ollama

from src.agent_loop import job_store, run_resolve_job
from src.tools import Tools, host_mount_allowed, _default_sandbox_root

# Load environment variables from .env file
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="AI Agent API",
    description=(
        "An API for interacting with an AI agent, submitting feedback, "
        "and running a tool-using issue-resolution loop in a sandbox."
    ),
    version="0.2.0",
)

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    model: str = os.getenv("OLLAMA_MODEL", "llama2")  # Default model from env or hardcoded


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
    tools = Tools()
    return ToolsInfoResponse(
        tools=tools.list_tools(),
        sandbox_root=str(_default_sandbox_root()),
        host_mount_allowed=host_mount_allowed(),
        notes=[
            "No host mount by default: workspaces live under AGENT_SANDBOX_ROOT "
            "(or a temp directory).",
            "File paths are confined to the workspace; traversal outside is rejected.",
            "Commands run with workspace cwd and a hard timeout "
            "(AGENT_COMMAND_TIMEOUT, default 60s).",
            "This is application-level isolation; deploy in containers without "
            "hostPath volumes for stronger boundaries.",
            "Set AGENT_ALLOW_HOST_MOUNT=true only when intentionally binding a host path.",
        ],
    )


@app.post(
    "/api/resolve",
    response_model=ResolveResponse,
    summary="Resolve an issue with the tool-using agent loop",
)
def resolve_issue(request: ResolveRequest):
    """
    Run (or enqueue) an issue-resolution job.

    The agent may checkout a repository into a sandbox, use tools
    (run_command, read_file, write_file), run tests, and produce a patch.

    - **async_mode=false** (default): runs inline and returns the finished job.
    - **async_mode=true**: returns immediately with status ``queued``/``running``;
      poll ``GET /api/jobs/{job_id}``.
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
    except Exception as e:  # noqa: BLE001
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
    """List in-memory resolve jobs (single-replica demo store)."""
    return {"jobs": job_store.list_jobs()}


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
        # Broad exception for now, can be refined
        raise HTTPException(status_code=500, detail=str(e))


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
