"""Tests for the tool-using agent loop and resolve job mode."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agent_loop import (
    AgentLoop,
    JobStore,
    parse_agent_action,
    run_resolve_job,
    validate_git_arg,
)
from src.main import app
from src.tools import Sandbox, ToolResult, Tools


class ScriptedLLM:
    """Deterministic LLM that returns a fixed sequence of responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, model=None):
        if self.calls >= len(self.responses):
            return json.dumps(
                {"action": "finish", "summary": "out of scripted responses", "success": False}
            )
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


@pytest.fixture
def sandbox_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SANDBOX_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.delenv("AGENT_ALLOW_HOST_MOUNT", raising=False)
    return Tools(Sandbox())


def test_parse_agent_action_raw_json():
    action = parse_agent_action('{"action":"read_file","path":"a.py"}')
    assert action["action"] == "read_file"
    assert action["path"] == "a.py"


def test_parse_agent_action_fenced():
    text = 'Sure.\n```json\n{"action":"finish","summary":"done","success":true}\n```'
    action = parse_agent_action(text)
    assert action["action"] == "finish"
    assert action["success"] is True


def test_agent_loop_write_and_finish(sandbox_tools):
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "action": "write_file",
                    "path": "fix.py",
                    "content": "def add(a, b):\n    return a + b\n",
                }
            ),
            json.dumps(
                {
                    "action": "run_command",
                    "command": "python -c \"from fix import add; assert add(1,2)==3\"",
                }
            ),
            json.dumps(
                {
                    "action": "finish",
                    "summary": "Added add() helper and verified it.",
                    "success": True,
                }
            ),
        ]
    )
    loop = AgentLoop(llm=llm, tools=sandbox_tools, max_steps=8)
    result = loop.run("Add an add(a,b) function in fix.py")

    assert result.status == "succeeded"
    assert result.patch  # should have a git diff
    assert "fix.py" in (result.patch or "")
    assert any(s.tool == "write_file" for s in result.steps)
    # File actually exists
    assert (sandbox_tools.sandbox.workspace / "fix.py").exists()


def test_agent_loop_max_steps(sandbox_tools):
    # Never finishes
    llm = ScriptedLLM(
        [
            json.dumps({"action": "run_command", "command": "python -c \"print(1)\""}),
            json.dumps({"action": "run_command", "command": "python -c \"print(2)\""}),
        ]
    )
    loop = AgentLoop(llm=llm, tools=sandbox_tools, max_steps=2)
    result = loop.run("Do something forever")
    assert result.status == "max_steps"
    assert len(result.steps) == 2


def test_run_resolve_job_sync(sandbox_tools):
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "action": "write_file",
                    "path": "note.txt",
                    "content": "resolved",
                }
            ),
            json.dumps(
                {"action": "finish", "summary": "wrote note", "success": True}
            ),
        ]
    )
    record = run_resolve_job(
        "Write note.txt",
        llm=llm,
        tools=sandbox_tools,
        max_steps=5,
        async_mode=False,
    )
    assert record["status"] == "succeeded"
    assert record["result"]["patch"] is not None
    assert "note.txt" in record["result"]["patch"]


def test_api_tools_endpoint():
    client = TestClient(app)
    response = client.get("/api/tools")
    assert response.status_code == 200
    body = response.json()
    names = {t["name"] for t in body["tools"]}
    assert names == {"run_command", "read_file", "write_file"}
    assert body["host_mount_allowed"] is False
    assert any("host mount" in n.lower() for n in body["notes"])


def test_api_resolve_sync(sandbox_tools, monkeypatch):
    """Drive /api/resolve with a scripted LLM via monkeypatch."""
    from src import agent_loop as agent_loop_mod

    responses = [
        json.dumps(
            {
                "action": "write_file",
                "path": "answer.txt",
                "content": "42",
            }
        ),
        json.dumps(
            {"action": "finish", "summary": "wrote answer", "success": True}
        ),
    ]
    llm = ScriptedLLM(responses)

    original = agent_loop_mod.run_resolve_job

    def patched_run_resolve_job(*args, **kwargs):
        kwargs.setdefault("llm", llm)
        kwargs.setdefault("tools", sandbox_tools)
        return original(*args, **kwargs)

    monkeypatch.setattr(agent_loop_mod, "run_resolve_job", patched_run_resolve_job)
    # main imports the function by name — patch there too
    monkeypatch.setattr("src.main.run_resolve_job", patched_run_resolve_job)

    client = TestClient(app)
    response = client.post(
        "/api/resolve",
        json={"issue": "Write 42 to answer.txt", "max_steps": 5, "async_mode": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["summary"]
    assert body["job_id"]

    # Job is fetchable
    job = client.get(f"/api/jobs/{body['job_id']}")
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"


def test_api_job_not_found():
    client = TestClient(app)
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_job_store_lifecycle():
    store = JobStore()
    store.create("j1", {"issue": "x"})
    assert store.get("j1")["status"] == "queued"
    store.set_running("j1")
    assert store.get("j1")["status"] == "running"


def test_prepare_workspace_no_repo(sandbox_tools):
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace()
    assert result.ok
    assert "Using existing workspace" in result.output


def test_prepare_workspace_runs_git_without_shell(sandbox_tools, monkeypatch):
    calls: list[tuple[object, object]] = []

    def mock_run_command(cmd, *, timeout=None, shell=None):
        calls.append((cmd, shell))
        return ToolResult(ok=True, tool="run_command", output="ok")

    monkeypatch.setattr(sandbox_tools, "run_command", mock_run_command)
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(
        repo_url="https://github.com/example/repo.git",
        git_ref="feature-branch",
    )
    assert result.ok
    assert len(calls) == 1
    assert calls[0] == (
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "feature-branch",
            "--",
            "https://github.com/example/repo.git",
            ".",
        ],
        False,
    )


def test_prepare_workspace_with_ref_uses_branch_flag(sandbox_tools, monkeypatch):
    """Non-default refs must be shallow-cloned via --branch, not checkout after default tip."""
    calls: list[list[str]] = []

    def mock_run_command(cmd, *, timeout=None, shell=None):
        calls.append(list(cmd))
        return ToolResult(ok=True, tool="run_command", output="ok")

    monkeypatch.setattr(sandbox_tools, "run_command", mock_run_command)
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(
        repo_url="https://github.com/example/repo.git",
        git_ref="v1.2.3",
    )
    assert result.ok
    assert calls[0][calls[0].index("--branch") + 1] == "v1.2.3"
    assert "--" in calls[0]
    assert "checkout" not in calls[0]


def test_prepare_workspace_clone_without_ref_has_separator(
    sandbox_tools, monkeypatch
):
    calls: list[tuple[object, object]] = []

    def mock_run_command(cmd, *, timeout=None, shell=None):
        calls.append((cmd, shell))
        return ToolResult(ok=True, tool="run_command", output="ok")

    monkeypatch.setattr(sandbox_tools, "run_command", mock_run_command)
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(
        repo_url="https://github.com/example/repo.git"
    )
    assert result.ok
    assert len(calls) == 1
    assert calls[0] == (
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--",
            "https://github.com/example/repo.git",
            ".",
        ],
        False,
    )


@pytest.mark.parametrize(
    "bad_repo_url",
    [
        "--upload-pack=touch /tmp/x",
        "-cprotocol.ext.allow=always",
        "-f",
        " --upload-pack=touch /tmp/x",
        "",
        "   ",
        "https://github.com/example/repo.git\n",
        "https://github.com/example/repo.git\r\n",
        "https://github.com/example/repo.git\x00",
    ],
)
def test_prepare_workspace_rejects_option_like_repo_url(
    sandbox_tools, monkeypatch, bad_repo_url
):
    calls: list[tuple[object, object]] = []

    def mock_run_command(cmd, *, timeout=None, shell=None):
        calls.append((cmd, shell))
        return ToolResult(ok=True, tool="run_command", output="ok")

    monkeypatch.setattr(sandbox_tools, "run_command", mock_run_command)
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(repo_url=bad_repo_url)

    assert not result.ok
    assert result.error is not None
    assert len(calls) == 0


@pytest.mark.parametrize(
    "bad_git_ref",
    [
        "--orphan=x",
        "-f",
        "--output=/tmp/pwn",
        " -b foo",
        "",
        "   ",
        "feature-branch\n",
        "feature-branch\r\n",
        "feature-branch\x00",
    ],
)
def test_prepare_workspace_rejects_option_like_git_ref(
    sandbox_tools, monkeypatch, bad_git_ref
):
    calls: list[tuple[object, object]] = []

    def mock_run_command(cmd, *, timeout=None, shell=None):
        calls.append((cmd, shell))
        return ToolResult(ok=True, tool="run_command", output="ok")

    monkeypatch.setattr(sandbox_tools, "run_command", mock_run_command)
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)

    # Rejection with repo_url provided (must reject without running git)
    result = loop.prepare_workspace(
        repo_url="https://github.com/example/repo.git",
        git_ref=bad_git_ref,
    )
    assert not result.ok
    assert result.error is not None
    assert len(calls) == 0

    # Rejection without repo_url
    result_no_repo = loop.prepare_workspace(
        repo_url=None,
        git_ref=bad_git_ref,
    )
    assert not result_no_repo.ok
    assert result_no_repo.error is not None
    assert len(calls) == 0


def test_prepare_workspace_prevents_shell_injection(sandbox_tools, tmp_path):
    marker = tmp_path / "pwned.txt"
    bad_repo_url = f"https://invalid.example.com/repo.git; touch {marker}"
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(repo_url=bad_repo_url)
    assert not result.ok
    assert not marker.exists()


def test_prepare_workspace_prevents_option_injection(sandbox_tools, tmp_path):
    marker = tmp_path / "pwned.txt"
    bad_repo_url = f"--upload-pack=touch {marker}"
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(repo_url=bad_repo_url)
    assert not result.ok
    assert not marker.exists()


def test_prepare_workspace_prevents_ref_option_injection(
    sandbox_tools, tmp_path
):
    marker = tmp_path / "pwned_ref.txt"
    bad_ref = f"--output={marker}"
    loop = AgentLoop(llm=ScriptedLLM([]), tools=sandbox_tools)
    result = loop.prepare_workspace(
        repo_url="https://invalid.example.com/repo.git",
        git_ref=bad_ref,
    )
    assert not result.ok
    assert not marker.exists()


def test_validate_git_arg_valid():
    assert validate_git_arg(
        "https://github.com/example/repo.git", "repo_url"
    ) is None
    assert validate_git_arg("main", "git_ref") is None
    assert validate_git_arg("v1.2.3", "git_ref") is None
    assert validate_git_arg("feature/branch-1", "git_ref") is None


def test_validate_git_arg_invalid():
    assert "cannot start with '-'" in validate_git_arg("-f", "git_ref")
    assert "cannot start with '-'" in validate_git_arg("--orphan=x", "git_ref")
    assert "cannot start with '-'" in validate_git_arg(
        "--upload-pack=x", "repo_url"
    )
    assert "cannot start with '-'" in validate_git_arg(
        " --option", "repo_url"
    )
    assert "cannot be empty" in validate_git_arg("", "repo_url")
    assert "cannot be empty" in validate_git_arg("   ", "git_ref")
    assert "contains forbidden" in validate_git_arg("ref\n", "git_ref")
    assert "contains forbidden" in validate_git_arg("ref\r", "git_ref")
    assert "contains forbidden" in validate_git_arg("ref\x00", "git_ref")
