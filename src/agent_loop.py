"""
Tool-using agent loop for issue resolution (checkout → test → patch).

The loop:
1. Prepares a sandboxed workspace (optionally clones a repo).
2. Iteratively asks an LLM for the next action (tool call or finish).
3. Executes tools via :class:`src.tools.Tools`.
4. Returns a structured result including the produced git patch.

LLM calls are injectable so tests do not require a live Ollama instance.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.tools import Sandbox, ToolResult, Tools


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal protocol for chat completion used by the agent loop."""

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        ...


@dataclass
class AgentStep:
    """One iteration of the agent loop."""

    step: int
    thought: str | None = None
    tool: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "thought": self.thought,
            "tool": self.tool,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "finish_reason": self.finish_reason,
        }


@dataclass
class AgentResult:
    """Final outcome of a resolve job / agent run."""

    job_id: str
    status: str  # succeeded | failed | max_steps | error
    issue: str
    patch: str | None = None
    summary: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    workspace: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "issue": self.issue,
            "patch": self.patch,
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "workspace": self.workspace,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Default Ollama-backed LLM client
# ---------------------------------------------------------------------------

class OllamaLLMClient:
    """Thin wrapper around the ollama Python package."""

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model or os.getenv("OLLAMA_MODEL", "llama2")

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        import ollama  # local import so tests can avoid the dependency path

        response = ollama.chat(
            model=model or self.default_model,
            messages=messages,
        )
        return response["message"]["content"]


# ---------------------------------------------------------------------------
# System prompt / action parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a software engineering agent working inside an isolated sandbox.
You may use tools to inspect and modify the workspace, run tests, and produce a patch.

Available tools (respond with a single JSON object, no markdown fences):
{"action":"run_command","command":"<shell command>"}
{"action":"read_file","path":"<relative path>"}
{"action":"write_file","path":"<relative path>","content":"<file contents>"}
{"action":"finish","summary":"<what you did>","success":true|false}

Guidelines:
- Prefer small, targeted edits.
- After changes, run tests if a test command is known or discoverable.
- When done, call finish. A git diff will be collected automatically.
- Do not attempt to access paths outside the workspace.
- Output ONLY the JSON object for your next action.
"""


def parse_agent_action(text: str) -> dict[str, Any]:
    """
    Parse an LLM response into an action dict.

    Accepts raw JSON or JSON embedded in markdown fences / surrounding prose.
    """
    text = text.strip()
    # Direct JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Fenced block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, dict) and "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    # First {...} object in the text
    brace = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace:
        try:
            data = json.loads(brace.group(0))
            if isinstance(data, dict) and "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    return {
        "action": "finish",
        "summary": f"Unparseable agent response: {text[:200]}",
        "success": False,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Checkout → tool loop → patch.

    Parameters
    ----------
    llm:
        Object implementing ``chat(messages, model=None) -> str``.
    tools:
        Pre-configured Tools instance. If omitted a fresh sandbox is created.
    max_steps:
        Hard cap on tool/LLM iterations.
    test_command:
        Optional default test command suggested in the initial user message.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: Tools | None = None,
        *,
        max_steps: int = 12,
        test_command: str | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm or OllamaLLMClient()
        self.tools = tools or Tools()
        self.max_steps = max_steps
        self.test_command = test_command
        self.model = model

    # -- public API ---------------------------------------------------------

    def prepare_workspace(
        self,
        *,
        repo_url: str | None = None,
        git_ref: str | None = None,
    ) -> ToolResult:
        """
        Optionally clone a repository into the sandbox workspace.

        If ``repo_url`` is None the existing (empty or pre-seeded) workspace
        is used as-is — useful for local job mode / tests.
        """
        if not repo_url:
            return ToolResult(
                ok=True,
                tool="prepare_workspace",
                output=f"Using existing workspace at {self.tools.sandbox.workspace}",
                meta={"workspace": str(self.tools.sandbox.workspace)},
            )

        # Clone into workspace. Workspace must be empty for git clone .
        clone_cmd = f"git clone --depth 1 {repo_url} ."
        if git_ref:
            # clone then checkout ref (shallow clone of default branch first)
            result = self.tools.run_command(clone_cmd, timeout=120)
            if not result.ok:
                return result
            return self.tools.run_command(f"git checkout {git_ref}", timeout=60)

        return self.tools.run_command(clone_cmd, timeout=120)

    def run(
        self,
        issue: str,
        *,
        repo_url: str | None = None,
        git_ref: str | None = None,
        job_id: str | None = None,
    ) -> AgentResult:
        """Execute the full agent loop for an issue description."""
        job_id = job_id or uuid.uuid4().hex
        steps: list[AgentStep] = []

        prep = self.prepare_workspace(repo_url=repo_url, git_ref=git_ref)
        if not prep.ok:
            return AgentResult(
                job_id=job_id,
                status="error",
                issue=issue,
                error=prep.error or prep.output,
                workspace=str(self.tools.sandbox.workspace),
                steps=[
                    AgentStep(
                        step=0,
                        tool="prepare_workspace",
                        tool_args={"repo_url": repo_url, "git_ref": git_ref},
                        tool_result=prep.to_dict(),
                    )
                ],
            )

        # Ensure git is initialized so we can always produce a patch
        self._ensure_git_baseline()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_task_message(issue, repo_url=repo_url),
            },
        ]

        for step_num in range(1, self.max_steps + 1):
            try:
                raw = self.llm.chat(messages, model=self.model)
            except Exception as exc:  # noqa: BLE001 — surface LLM failures cleanly
                return AgentResult(
                    job_id=job_id,
                    status="error",
                    issue=issue,
                    error=f"LLM error: {exc}",
                    steps=steps,
                    workspace=str(self.tools.sandbox.workspace),
                )

            action = parse_agent_action(raw)
            action_name = str(action.get("action", "finish")).lower()

            if action_name == "finish":
                success = bool(action.get("success", True))
                summary = str(action.get("summary") or "Agent finished.")
                step = AgentStep(
                    step=step_num,
                    thought=raw[:500],
                    finish_reason=summary,
                )
                steps.append(step)
                patch = self._collect_patch()
                return AgentResult(
                    job_id=job_id,
                    status="succeeded" if success else "failed",
                    issue=issue,
                    patch=patch,
                    summary=summary,
                    steps=steps,
                    workspace=str(self.tools.sandbox.workspace),
                )

            # Map action → tool
            tool_name, tool_kwargs = self._action_to_tool(action)
            result = self.tools.invoke(tool_name, **tool_kwargs)
            step = AgentStep(
                step=step_num,
                thought=raw[:500],
                tool=tool_name,
                tool_args=tool_kwargs,
                tool_result=result.to_dict(),
            )
            steps.append(step)

            # Feed observation back to the model
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result ({tool_name}):\n"
                        f"{json.dumps(result.to_dict(), indent=2)}\n"
                        "Continue. Respond with the next JSON action only."
                    ),
                }
            )

        # Max steps reached — still collect whatever patch exists
        patch = self._collect_patch()
        return AgentResult(
            job_id=job_id,
            status="max_steps",
            issue=issue,
            patch=patch,
            summary=f"Stopped after {self.max_steps} steps without finish.",
            steps=steps,
            workspace=str(self.tools.sandbox.workspace),
        )

    # -- internals ----------------------------------------------------------

    def _build_task_message(
        self,
        issue: str,
        *,
        repo_url: str | None,
    ) -> str:
        parts = [
            "Resolve the following software issue in the sandbox workspace.",
            "",
            f"Issue:\n{issue}",
        ]
        if repo_url:
            parts.append(f"\nRepository: {repo_url}")
        if self.test_command:
            parts.append(f"\nSuggested test command: {self.test_command}")
        parts.append(
            "\nWorkspace root: "
            f"{self.tools.sandbox.workspace}\n"
            "Start by inspecting the codebase, then make changes and run tests."
        )
        return "\n".join(parts)

    def _action_to_tool(self, action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = str(action.get("action", "")).lower()
        if name == "run_command":
            kwargs: dict[str, Any] = {"command": action.get("command", "")}
            if "timeout" in action:
                kwargs["timeout"] = action["timeout"]
            return "run_command", kwargs
        if name == "read_file":
            return "read_file", {"path": action.get("path", "")}
        if name == "write_file":
            return "write_file", {
                "path": action.get("path", ""),
                "content": action.get("content", ""),
            }
        # Unknown → treat as no-op command so the loop can continue
        return "run_command", {
            "command": f"echo Unknown action: {name}",
        }

    def _ensure_git_baseline(self) -> None:
        """Initialize git and commit current state so diffs are meaningful."""
        ws = self.tools.sandbox.workspace
        git_dir = ws / ".git"
        if not git_dir.exists():
            self.tools.run_command("git init")
            self.tools.run_command('git config user.email "agent@localhost"')
            self.tools.run_command('git config user.name "SWE Agent"')
            # Best-effort add/commit; empty trees are fine
            self.tools.run_command("git add -A")
            self.tools.run_command(
                'git commit --allow-empty -m "agent baseline" --no-gpg-sign'
            )
        else:
            # Record baseline commit of current tree if dirty
            self.tools.run_command("git add -A")
            # Commit only if there is something to commit
            status = self.tools.run_command("git status --porcelain")
            if status.output.strip():
                self.tools.run_command(
                    'git commit -m "agent baseline" --no-gpg-sign'
                )

    def _collect_patch(self) -> str | None:
        """Stage changes and return a unified diff patch string."""
        self.tools.run_command("git add -A")
        diff = self.tools.run_command("git diff --no-color --cached")
        if diff.ok and diff.output.strip():
            return diff.output
        # Fall back to unstaged + working tree
        diff2 = self.tools.run_command("git diff --no-color HEAD")
        if diff2.ok and diff2.output.strip():
            return diff2.output
        return diff.output if diff.output.strip() else None


# ---------------------------------------------------------------------------
# In-memory job store (simple job mode)
# ---------------------------------------------------------------------------

class JobStore:
    """
    Minimal in-process job registry for issue-resolution runs.

    Suitable for single-replica demos. For production, replace with a
    durable queue / database.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "job_id": job_id,
            "status": "queued",
            "request": payload,
            "result": None,
            "error": None,
        }
        self._jobs[job_id] = record
        return record

    def set_running(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "running"

    def set_result(self, job_id: str, result: AgentResult) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = result.status
            self._jobs[job_id]["result"] = result.to_dict()

    def set_error(self, job_id: str, error: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "error"
            self._jobs[job_id]["error"] = error

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return list(self._jobs.values())


# Module-level store shared by the API
job_store = JobStore()


def run_resolve_job(
    issue: str,
    *,
    repo_url: str | None = None,
    git_ref: str | None = None,
    max_steps: int = 12,
    test_command: str | None = None,
    model: str | None = None,
    llm: LLMClient | None = None,
    tools: Tools | None = None,
    job_id: str | None = None,
    async_mode: bool = False,
    runner: Callable[[Callable[[], None]], None] | None = None,
) -> dict[str, Any]:
    """
    Create and (optionally asynchronously) execute a resolve job.

    Returns the job record immediately when ``async_mode`` is True;
    otherwise runs the agent loop inline and returns the finished record.
    """
    job_id = job_id or uuid.uuid4().hex
    payload = {
        "issue": issue,
        "repo_url": repo_url,
        "git_ref": git_ref,
        "max_steps": max_steps,
        "test_command": test_command,
        "model": model,
    }
    record = job_store.create(job_id, payload)

    def _execute() -> None:
        job_store.set_running(job_id)
        try:
            loop = AgentLoop(
                llm=llm,
                tools=tools,
                max_steps=max_steps,
                test_command=test_command,
                model=model,
            )
            result = loop.run(
                issue,
                repo_url=repo_url,
                git_ref=git_ref,
                job_id=job_id,
            )
            job_store.set_result(job_id, result)
        except Exception as exc:  # noqa: BLE001
            job_store.set_error(job_id, str(exc))

    if async_mode:
        if runner is not None:
            runner(_execute)
        else:
            import threading

            threading.Thread(target=_execute, daemon=True).start()
        return job_store.get(job_id)  # type: ignore[return-value]

    _execute()
    return job_store.get(job_id)  # type: ignore[return-value]
