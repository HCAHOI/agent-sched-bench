from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agents.code_agent import CodeAgent


def create_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import value\n\n\ndef test_value():\n    assert value == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def test_code_agent_parses_tool_calls() -> None:
    agent = CodeAgent(agent_id="code-1", api_base="http://localhost:8000/v1", model="mock")
    tool = agent._parse_tool_call("bash(pytest -q)")
    assert tool is not None
    assert tool.name == "bash"
    assert tool.args == "pytest -q"


def test_code_agent_executes_real_bash_tool(tmp_path: Path) -> None:
    repo = create_git_repo(tmp_path)
    agent = CodeAgent(agent_id="code-2", api_base="http://localhost:8000/v1", model="mock")
    agent._workspace_path = repo
    tool = agent._parse_tool_call("bash(pwd)")
    assert tool is not None
    output = asyncio.run(agent._execute_tool(tool, {"test_cmd": "pytest"}))
    assert str(repo) in output


def test_code_agent_applies_patch_and_runs_tests(tmp_path: Path) -> None:
    repo = create_git_repo(tmp_path)
    agent = CodeAgent(agent_id="code-3", api_base="http://localhost:8000/v1", model="mock")
    agent._workspace_path = repo
    patch = """diff --git a/app.py b/app.py
index 4f2d7d6..43dd47e 100644
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
    success, output = asyncio.run(
        agent._apply_and_test(
            patch,
            {
                "instance_id": "demo",
                "problem_statement": "fix value",
                "repo_path": str(repo),
                "test_cmd": "python3 -m pytest -q",
            },
        )
    )
    assert success is True
    assert "1 passed" in output


def test_code_agent_workspace_copy_is_isolated_and_cleanup_removes_temp_root(tmp_path: Path) -> None:
    source_repo = create_git_repo(tmp_path)
    agent = CodeAgent(agent_id="code-4", api_base="http://localhost:8000/v1", model="mock")
    workspace = agent._prepare_workspace(
        {
            "instance_id": "demo-copy",
            "problem_statement": "noop",
            "repo_path": str(source_repo),
            "test_cmd": "python3 -m pytest -q",
        }
    )
    (workspace / "app.py").write_text("value = 99\n", encoding="utf-8")
    assert (source_repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    temp_root = workspace.parent
    agent._cleanup_workspace()
    assert not temp_root.exists()


class StaticLLMHandler(BaseHTTPRequestHandler):
    responses = [
        {
            "id": "resp-1",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "No tool needed."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        {
            "id": "resp-2",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Still no tool needed."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        },
    ]
    call_count = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        program_id = payload.get("program_id")
        if program_id is None:
            program_id = (payload.get("extra_body") or {}).get("program_id")
        assert program_id == "code-5"
        response = self.responses[self.__class__.call_count]
        self.__class__.call_count += 1
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_code_agent_run_clears_trace_between_tasks(tmp_path: Path) -> None:
    source_repo = create_git_repo(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), StaticLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    agent = CodeAgent(agent_id="code-5", api_base=api_base, model="mock")
    task = {
        "instance_id": "task-1",
        "problem_statement": "noop",
        "repo_path": str(source_repo),
        "test_cmd": "python3 -m pytest -q",
    }
    try:
        result_one = asyncio.run(agent.run(task))
        assert result_one is False
        assert len(agent.trace) == 1
        assert agent.trace[0].llm_output == "No tool needed."

        task_two = {**task, "instance_id": "task-2"}
        result_two = asyncio.run(agent.run(task_two))
        assert result_two is False
        assert len(agent.trace) == 1
        assert agent.trace[0].llm_output == "Still no tool needed."
        assert agent.task_id == "task-2"
    finally:
        server.shutdown()
        thread.join(timeout=2)
