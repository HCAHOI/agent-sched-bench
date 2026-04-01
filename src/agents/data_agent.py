from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import textwrap
import time
from pathlib import Path
from typing import Any

from agents.base import AgentBase
from agents.tool_calling import extract_sql_block


SYSTEM_PROMPT = """You are a data analyst. Given a natural language question
and a database schema, write SQL to answer the question.

If your SQL has errors, read the error message and try again.
When you are ready to finish, call sql_execute with your final answer query."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "schema_inspect",
            "description": "Show column names and types for a table. Pass '*' to list all tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name, or '*' to list all tables.",
                    }
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_execute",
            "description": "Execute a read-only SQL query on the database and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL SELECT/WITH/PRAGMA query to execute.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


class DataAgent(AgentBase):
    """NL2SQL agent for SQLite-backed BIRD-style tasks."""

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        max_steps: int = 20,
        sql_timeout_s: float = 30.0,
        max_tool_output_chars: int = 8000,
    ) -> None:
        super().__init__(agent_id=agent_id, api_base=api_base, model=model)
        self.max_steps = max_steps
        self.sql_timeout_s = sql_timeout_s
        self.max_tool_output_chars = max_tool_output_chars

    def _format_task(self, task: dict[str, Any]) -> str:
        evidence = task.get("evidence", "")
        return textwrap.dedent(
            f"""
            Question:
            {task['question']}

            Database path:
            {task['db_path']}

            Evidence:
            {evidence or "(none)"}
            """
        ).strip()

    def _schema_inspect_sync(self, db_path: Path, table_name: str) -> str:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            if table_name in {"*", "all", ""}:
                tables = cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                return "\n".join(row[0] for row in tables) or "(no tables)"
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table_name):
                return f"ERROR: Invalid table name: {table_name!r}"
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            if not columns:
                return f"ERROR: table not found: {table_name}"
            return "\n".join(
                f"{column[1]} {column[2]} NOTNULL={column[3]} PK={column[5]}"
                for column in columns
            )

    def _sql_execute_sync(self, db_path: Path, query: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        if not re.match(r"^\s*(SELECT|WITH|PRAGMA)\b", query, flags=re.IGNORECASE):
            raise sqlite3.OperationalError("Only read-only SELECT/WITH/PRAGMA statements are allowed")
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            rows = cursor.execute(query).fetchall()
            columns = [description[0] for description in (cursor.description or [])]
            return columns, rows

    async def _schema_inspect(self, db_path: Path, table_name: str) -> str:
        return await asyncio.to_thread(self._schema_inspect_sync, db_path, table_name)

    async def _sql_execute(self, db_path: Path, query: str) -> str:
        try:
            columns, rows = await asyncio.wait_for(
                asyncio.to_thread(self._sql_execute_sync, db_path, query),
                timeout=self.sql_timeout_s,
            )
        except asyncio.TimeoutError:
            return "ERROR: SQL execution timed out"
        except sqlite3.Error as exc:
            return f"ERROR: {exc}"
        return json.dumps({"columns": columns, "rows": rows}, ensure_ascii=True)

    async def _evaluate_final_sql(self, candidate_sql: str, task: dict[str, Any]) -> tuple[bool, str]:
        db_path = Path(task["db_path"]).resolve()
        candidate_result = await self._sql_execute(db_path, candidate_sql)
        if candidate_result.startswith("ERROR:"):
            return False, candidate_result
        gold_sql = task.get("gold_sql")
        if not gold_sql:
            return True, candidate_result
        gold_result = await self._sql_execute(db_path, gold_sql)
        if gold_result.startswith("ERROR:"):
            return False, f"ERROR: gold SQL failed: {gold_result}"
        candidate_payload = json.loads(candidate_result)
        gold_payload = json.loads(gold_result)
        candidate_rows = candidate_payload["rows"]
        gold_rows = gold_payload["rows"]
        if "order by" not in candidate_sql.lower() and "order by" not in gold_sql.lower():
            candidate_rows = sorted(
                candidate_rows,
                key=lambda row: json.dumps(row, ensure_ascii=True),
            )
            gold_rows = sorted(
                gold_rows,
                key=lambda row: json.dumps(row, ensure_ascii=True),
            )
        return candidate_rows == gold_rows, candidate_result

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_output_chars:
            return text
        half = self.max_tool_output_chars // 2
        return text[:half] + f"\n[... truncated {len(text) - self.max_tool_output_chars} chars ...]\n" + text[-half:]

    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = str(task.get("task_id", task.get("question", "")))
        self.task_success = False
        self.trace = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_task(task)},
        ]
        db_path = Path(task["db_path"]).resolve()

        for step_idx in range(self.max_steps):
            ts_start = time.time()
            llm_result = await self._call_llm(messages, tools=TOOLS)
            ts_end = time.time()
            record = self.build_step_record(
                step_idx=step_idx,
                phase="reasoning",
                llm_result=llm_result,
                ts_start=ts_start,
                ts_end=ts_end,
            )

            if not llm_result.tool_calls:
                # No tool call — check for final SQL in text content (fallback)
                final_sql = extract_sql_block(llm_result.content) if llm_result.content else None
                if final_sql is None:
                    self.trace.append(record)
                    break
                tool_started = time.monotonic()
                success, output = await self._evaluate_final_sql(final_sql, task)
                record.phase = "acting"
                record.tool_name = "final_sql"
                record.tool_args = final_sql
                record.tool_result = output
                record.tool_success = success
                record.tool_duration_ms = (time.monotonic() - tool_started) * 1000
                record.extra["evaluation_mode"] = "denotation"
                self.task_success = success
                self.trace.append(record)
                break

            tc = llm_result.tool_calls[0]
            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                args = {}

            record.phase = "acting"
            record.tool_name = tc.name
            record.tool_args = json.dumps(args)
            tool_started = time.monotonic()

            if tc.name == "schema_inspect":
                tool_output = await self._schema_inspect(db_path, args.get("table", "*"))
            elif tc.name == "sql_execute":
                query = args.get("query", "")
                tool_output = await self._sql_execute(db_path, query)
                # Check if this might be the final answer (last step heuristic)
            else:
                tool_output = f"ERROR: unknown tool {tc.name}"

            record.tool_duration_ms = (time.monotonic() - tool_started) * 1000
            record.tool_result = tool_output
            record.tool_success = not tool_output.startswith("ERROR:")
            self.trace.append(record)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": llm_result.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
            ]
            messages.append(assistant_msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": self._truncate(tool_output),
            })
        return bool(self.task_success)
