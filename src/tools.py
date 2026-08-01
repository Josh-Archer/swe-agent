"""
Minimal tool interface for a sandboxed agent workspace.

Safety defaults
---------------
* Workspaces live under a dedicated sandbox root (temp dir by default).
* The host filesystem is NOT mounted into the sandbox by default.
* All file paths are resolved and confined to the workspace root
  (path traversal outside the workspace is rejected).
* Commands run with the workspace as cwd and a hard timeout.
* Set AGENT_SANDBOX_ROOT to override the sandbox root directory.
* Set AGENT_ALLOW_HOST_MOUNT=true only if you intentionally accept
  the risk of binding a host path (still confined by path checks when
  workspace_dir is set under the sandbox root).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration (env-driven safety defaults)
# ---------------------------------------------------------------------------

def _default_sandbox_root() -> Path:
    """Return the sandbox root. Never the host project root by default."""
    configured = os.getenv("AGENT_SANDBOX_ROOT")
    if configured:
        root = Path(configured).resolve()
    else:
        root = Path(tempfile.gettempdir()) / "swe-agent-sandboxes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def host_mount_allowed() -> bool:
    """Whether binding an arbitrary host path as workspace is allowed."""
    return os.getenv("AGENT_ALLOW_HOST_MOUNT", "false").lower() in {
        "1",
        "true",
        "yes",
    }


DEFAULT_COMMAND_TIMEOUT = int(os.getenv("AGENT_COMMAND_TIMEOUT", "60"))
MAX_OUTPUT_BYTES = int(os.getenv("AGENT_MAX_OUTPUT_BYTES", str(256 * 1024)))


# ---------------------------------------------------------------------------
# Tool result / errors
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Structured result from a tool invocation."""

    ok: bool
    tool: str
    output: str = ""
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "output": self.output,
            "error": self.error,
            "meta": self.meta,
        }


class SandboxError(Exception):
    """Raised when a sandbox safety check fails."""


# ---------------------------------------------------------------------------
# Sandbox workspace
# ---------------------------------------------------------------------------

class Sandbox:
    """
    Isolated workspace for tool execution.

    By default a fresh directory is created under the sandbox root
    (typically a temp path). Host paths are refused unless
    AGENT_ALLOW_HOST_MOUNT=true.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        *,
        create: bool = True,
        sandbox_root: Path | None = None,
    ) -> None:
        self.sandbox_root = (sandbox_root or _default_sandbox_root()).resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

        if workspace_dir is None:
            self.workspace = (self.sandbox_root / f"ws-{uuid.uuid4().hex[:12]}").resolve()
            if create:
                self.workspace.mkdir(parents=True, exist_ok=True)
        else:
            candidate = Path(workspace_dir).resolve()
            if not self._is_under_sandbox(candidate) and not host_mount_allowed():
                raise SandboxError(
                    "Host path as workspace is disabled by default "
                    "(set AGENT_ALLOW_HOST_MOUNT=true to override). "
                    f"path={candidate}"
                )
            self.workspace = candidate
            if create:
                self.workspace.mkdir(parents=True, exist_ok=True)

    def _is_under_sandbox(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.sandbox_root)
            return True
        except ValueError:
            return False

    def resolve_path(self, relative_or_absolute: str | Path) -> Path:
        """
        Resolve a path and ensure it stays inside the workspace.
        Relative paths are interpreted relative to the workspace root.
        """
        raw = Path(relative_or_absolute)
        if raw.is_absolute():
            target = raw.resolve()
        else:
            target = (self.workspace / raw).resolve()

        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise SandboxError(
                f"Path escapes workspace: {relative_or_absolute!r} -> {target}"
            ) from exc
        return target

    def cleanup(self) -> None:
        """Remove the workspace directory if it lives under the sandbox root."""
        if self.workspace.exists() and self._is_under_sandbox(self.workspace):
            shutil.rmtree(self.workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class Tools:
    """
    Minimal tool surface for an SWE-style agent loop:

    * run_command – execute a shell command in the sandbox workspace
    * read_file   – read a text file inside the workspace
    * write_file  – write/create a text file inside the workspace
    """

    def __init__(
        self,
        sandbox: Sandbox | None = None,
        *,
        command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ) -> None:
        self.sandbox = sandbox or Sandbox()
        self.command_timeout = command_timeout

    # -- run_command --------------------------------------------------------

    def run_command(
        self,
        command: str,
        *,
        timeout: int | None = None,
        shell: bool = True,
    ) -> ToolResult:
        """
        Run a command with cwd=workspace and a hard timeout.

        Notes:
        * This is not a full OS-level sandbox (no seccomp/namespaces).
          Isolation relies on workspace confinement + container deployment.
        * Prefer deploying this agent in k8s without hostPath mounts.
        """
        if not command or not command.strip():
            return ToolResult(
                ok=False,
                tool="run_command",
                error="Empty command",
            )

        timeout = timeout if timeout is not None else self.command_timeout
        try:
            completed = subprocess.run(
                command,
                shell=shell,
                cwd=str(self.sandbox.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._safe_env(),
            )
            stdout = _truncate(completed.stdout or "")
            stderr = _truncate(completed.stderr or "")
            combined = stdout
            if stderr:
                combined = f"{stdout}\n--- stderr ---\n{stderr}".strip()

            return ToolResult(
                ok=completed.returncode == 0,
                tool="run_command",
                output=combined,
                error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
                meta={
                    "returncode": completed.returncode,
                    "command": command,
                    "timeout": timeout,
                    "cwd": str(self.sandbox.workspace),
                },
            )
        except subprocess.TimeoutExpired as exc:
            partial = _truncate((exc.stdout or "") if isinstance(exc.stdout, str) else "")
            return ToolResult(
                ok=False,
                tool="run_command",
                output=partial,
                error=f"Command timed out after {timeout}s",
                meta={"command": command, "timeout": timeout},
            )
        except OSError as exc:
            return ToolResult(
                ok=False,
                tool="run_command",
                error=str(exc),
                meta={"command": command},
            )

    # -- read_file ----------------------------------------------------------

    def read_file(self, path: str) -> ToolResult:
        """Read a UTF-8 text file inside the workspace."""
        try:
            target = self.sandbox.resolve_path(path)
            if not target.exists():
                return ToolResult(
                    ok=False,
                    tool="read_file",
                    error=f"File not found: {path}",
                    meta={"path": str(target)},
                )
            if not target.is_file():
                return ToolResult(
                    ok=False,
                    tool="read_file",
                    error=f"Not a file: {path}",
                    meta={"path": str(target)},
                )
            content = target.read_text(encoding="utf-8")
            content = _truncate(content)
            return ToolResult(
                ok=True,
                tool="read_file",
                output=content,
                meta={"path": str(target), "bytes": target.stat().st_size},
            )
        except SandboxError as exc:
            return ToolResult(ok=False, tool="read_file", error=str(exc))
        except OSError as exc:
            return ToolResult(ok=False, tool="read_file", error=str(exc))

    # -- write_file ---------------------------------------------------------

    def write_file(self, path: str, content: str) -> ToolResult:
        """Create or overwrite a UTF-8 text file inside the workspace."""
        try:
            target = self.sandbox.resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(
                ok=True,
                tool="write_file",
                output=f"Wrote {len(content)} characters to {path}",
                meta={"path": str(target), "bytes": len(content.encode('utf-8'))},
            )
        except SandboxError as exc:
            return ToolResult(ok=False, tool="write_file", error=str(exc))
        except OSError as exc:
            return ToolResult(ok=False, tool="write_file", error=str(exc))

    # -- helpers ------------------------------------------------------------

    def list_tools(self) -> list[dict[str, str]]:
        """Describe available tools (for LLM system prompts / API docs)."""
        return [
            {
                "name": "run_command",
                "description": (
                    "Run a shell command in the sandbox workspace. "
                    "Args: command (str), optional timeout (int seconds)."
                ),
            },
            {
                "name": "read_file",
                "description": (
                    "Read a text file relative to the workspace. Args: path (str)."
                ),
            },
            {
                "name": "write_file",
                "description": (
                    "Write a text file relative to the workspace. "
                    "Args: path (str), content (str)."
                ),
            },
        ]

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """Dispatch a tool by name. Unknown tools return an error result."""
        if name == "run_command":
            return self.run_command(
                kwargs.get("command", ""),
                timeout=kwargs.get("timeout"),
            )
        if name == "read_file":
            return self.read_file(kwargs.get("path", ""))
        if name == "write_file":
            return self.write_file(
                kwargs.get("path", ""),
                kwargs.get("content", ""),
            )
        return ToolResult(ok=False, tool=name, error=f"Unknown tool: {name}")

    def git_diff(self) -> ToolResult:
        """Convenience: produce a unified patch of workspace changes."""
        # Ensure we are in a git repo; if not, still try (will error cleanly).
        return self.run_command("git diff --no-color && git diff --no-color --cached")

    @staticmethod
    def _safe_env() -> dict[str, str]:
        """Pass a minimal environment into child processes."""
        keep = (
            "PATH",
            "HOME",
            "USER",
            "USERNAME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "COMSPEC",
            "PATHEXT",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        )
        env = {k: v for k, v in os.environ.items() if k in keep}
        # Mark child as agent-run for observability
        env["SWE_AGENT_SANDBOX"] = "1"
        return env


def _truncate(text: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if len(text.encode("utf-8", errors="replace")) <= limit:
        return text
    # Truncate by characters approximately at byte limit
    encoded = text.encode("utf-8", errors="replace")[:limit]
    return encoded.decode("utf-8", errors="replace") + "\n...[truncated]..."
