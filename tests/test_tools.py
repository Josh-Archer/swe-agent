"""Tests for the sandboxed tool interface."""

import os
from pathlib import Path

import pytest

from src.tools import Sandbox, SandboxError, Tools, host_mount_allowed


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """Tools instance with sandbox root under pytest tmp_path."""
    monkeypatch.setenv("AGENT_SANDBOX_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.delenv("AGENT_ALLOW_HOST_MOUNT", raising=False)
    return Tools(Sandbox())


def test_write_and_read_file(tools):
    w = tools.write_file("hello.txt", "hello world")
    assert w.ok
    r = tools.read_file("hello.txt")
    assert r.ok
    assert r.output == "hello world"


def test_read_missing_file(tools):
    r = tools.read_file("nope.txt")
    assert not r.ok
    assert "not found" in (r.error or "").lower()


def test_path_traversal_rejected(tools):
    r = tools.read_file("../outside.txt")
    assert not r.ok
    assert "escapes workspace" in (r.error or "").lower()

    w = tools.write_file("../../etc/passwd", "x")
    assert not w.ok
    assert "escapes workspace" in (w.error or "").lower()


def test_run_command_in_workspace(tools):
    tools.write_file("a.txt", "data")
    # Cross-platform: use python to list/print rather than shell builtins
    result = tools.run_command("python -c \"print(open('a.txt').read())\"")
    assert result.ok
    assert "data" in result.output


def test_run_command_timeout(tools, monkeypatch):
    monkeypatch.setenv("AGENT_COMMAND_TIMEOUT", "1")
    t = Tools(tools.sandbox, command_timeout=1)
    # Sleep longer than timeout
    result = t.run_command("python -c \"import time; time.sleep(30)\"", timeout=1)
    assert not result.ok
    assert "timed out" in (result.error or "").lower()


def test_run_command_nonzero_exit(tools):
    result = tools.run_command("python -c \"raise SystemExit(2)\"")
    assert not result.ok
    assert result.meta.get("returncode") == 2


def test_host_mount_rejected_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SANDBOX_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.delenv("AGENT_ALLOW_HOST_MOUNT", raising=False)
    assert host_mount_allowed() is False
    outside = tmp_path / "host-project"
    outside.mkdir()
    with pytest.raises(SandboxError, match="Host path"):
        Sandbox(workspace_dir=outside)


def test_host_mount_allowed_with_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SANDBOX_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("AGENT_ALLOW_HOST_MOUNT", "true")
    outside = tmp_path / "host-project"
    outside.mkdir()
    sb = Sandbox(workspace_dir=outside)
    assert sb.workspace == outside.resolve()


def test_invoke_dispatch(tools):
    result = tools.invoke("write_file", path="x.py", content="print(1)\n")
    assert result.ok
    result = tools.invoke("read_file", path="x.py")
    assert result.ok and "print(1)" in result.output
    result = tools.invoke("unknown_tool")
    assert not result.ok


def test_list_tools(tools):
    names = {t["name"] for t in tools.list_tools()}
    assert names == {"run_command", "read_file", "write_file"}


def test_workspace_under_sandbox_root(tools, tmp_path):
    root = Path(os.environ["AGENT_SANDBOX_ROOT"]).resolve()
    assert tools.sandbox.workspace.is_relative_to(root)
