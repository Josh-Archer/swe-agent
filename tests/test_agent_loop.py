"""Tests for the tool-using agent loop and resolve job mode."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agent_loop import AgentLoop, JobStore, parse_agent_action, run_resolve_job
from src.main import app
from src.tools import Sandbox, Tools


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
