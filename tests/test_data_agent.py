from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents.data_agent import DataAgent


def create_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "demo.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, team TEXT)")
        conn.executemany(
            "INSERT INTO employees (name, team) VALUES (?, ?)",
            [("Alice", "ml"), ("Bob", "infra"), ("Cara", "ml")],
        )
        conn.commit()
    return db_path


def test_data_agent_schema_inspect_and_sql_execute(tmp_path: Path) -> None:
    db_path = create_sqlite_db(tmp_path)
    agent = DataAgent(agent_id="data-1", api_base="http://localhost:8000/v1", model="mock")
    schema = asyncio.run(agent._schema_inspect(db_path, "employees"))
    assert "name TEXT" in schema
    result = asyncio.run(agent._sql_execute(db_path, "SELECT COUNT(*) AS n FROM employees"))
    payload = json.loads(result)
    assert payload["rows"][0][0] == 3


def test_data_agent_rejects_mutating_sql_and_preserves_db(tmp_path: Path) -> None:
    db_path = create_sqlite_db(tmp_path)
    agent = DataAgent(agent_id="data-1b", api_base="http://localhost:8000/v1", model="mock")
    result = asyncio.run(agent._sql_execute(db_path, "DELETE FROM employees WHERE team = 'ml'"))
    assert result.startswith("ERROR:")
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    assert count == 3


def test_data_agent_final_sql_handles_unordered_null_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "nulls.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.executemany("INSERT INTO t (x) VALUES (?)", [(None,), (1,)])
        conn.commit()
    agent = DataAgent(agent_id="data-1c", api_base="http://localhost:8000/v1", model="mock")
    success, _ = asyncio.run(
        agent._evaluate_final_sql(
            "SELECT x FROM t",
            {
                "db_path": str(db_path),
                "gold_sql": "SELECT x AS alias_name FROM t",
            },
        )
    )
    assert success is True


class SQLLLMHandler(BaseHTTPRequestHandler):
    responses = [
        {
            "id": "resp-1",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "schema_inspect", "arguments": '{"table": "employees"}'}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        {
            "id": "resp-2",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "sql_execute", "arguments": '{"query": "SELECT COUNT(*) FROM missing_table"}'}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
        {
            "id": "resp-3",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "sql_execute", "arguments": "{\"query\": \"SELECT COUNT(*) AS n FROM employees WHERE team = 'ml'\"}"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
        {
            "id": "resp-4",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "```sql\nSELECT COUNT(*) AS c FROM employees WHERE team = 'ml'\n```",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
    ]
    call_count = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
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


def test_data_agent_run_with_local_backend(tmp_path: Path) -> None:
    db_path = create_sqlite_db(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), SQLLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    agent = DataAgent(agent_id="data-2", api_base=api_base, model="mock")
    try:
        success = asyncio.run(
            agent.run(
                {
                    "task_id": "q-1",
                    "question": "How many ml employees are there?",
                    "db_path": str(db_path),
                    "gold_sql": "SELECT COUNT(*) AS n FROM employees WHERE team = 'ml'",
                    "evidence": "employees.team stores the team name",
                }
            )
        )
        assert success is True
        assert len(agent.trace) == 4
        assert any(record.tool_name == "sql_execute" for record in agent.trace)
        assert agent.trace[-1].tool_name == "final_sql"
        assert agent.trace[-1].tool_success is True
        assert agent.trace[-1].tool_duration_ms is not None
    finally:
        server.shutdown()
        thread.join(timeout=2)
