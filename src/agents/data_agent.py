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
from agents.tool_calling import ToolCall, extract_sql_block, parse_tool_call


SYSTEM_PROMPT = """You are a data analyst. Given a natural language question
and a database schema, write SQL to answer the question.

Available tools:
- schema_inspect(table): Show column names and types for a table
- sql_execute(query): Execute SQL on the database and return results

If your SQL has errors, read the error message and try again.
When you are ready to finish, return the final SQL in a ```sql``` block."""


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

    def _parse_tool_call(self, response: str) -> ToolCall | None:
        return parse_tool_call(response, {"schema_inspect", "sql_execute"})

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

    async def _execute_tool(self, tool_call: ToolCall, task: dict[str, Any]) -> str:
        db_path = Path(task["db_path"]).resolve()
        if tool_call.name == "schema_inspect":
            return await self._schema_inspect(db_path, tool_call.args)
        if tool_call.name == "sql_execute":
            return await self._sql_execute(db_path, tool_call.args)
        raise ValueError(f"Unsupported tool: {tool_call.name}")

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

    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = str(task.get("task_id", task.get("question", "")))
        self.task_success = False
        self.trace = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_task(task)},
        ]

        for step_idx in range(self.max_steps):
            ts_start = time.time()
            llm_result = await self._call_llm(messages)
            ts_end = time.time()
            record = self.build_step_record(
                step_idx=step_idx,
                phase="reasoning",
                llm_result=llm_result,
                ts_start=ts_start,
                ts_end=ts_end,
            )
            tool_call = self._parse_tool_call(llm_result.content)
            if tool_call is None:
                final_sql = extract_sql_block(llm_result.content)
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

            record.phase = "acting"
            record.tool_name = tool_call.name
            record.tool_args = tool_call.args
            tool_started = time.monotonic()
            tool_output = await self._execute_tool(tool_call, task)
            record.tool_duration_ms = (time.monotonic() - tool_started) * 1000
            record.tool_result = tool_output
            record.tool_success = not tool_output.startswith("ERROR:")
            self.trace.append(record)
            if len(tool_output) > self.max_tool_output_chars:
                half = self.max_tool_output_chars // 2
                tool_output = tool_output[:half] + f"\n[... truncated {len(tool_output) - self.max_tool_output_chars} chars ...]\n" + tool_output[-half:]
            messages.append({"role": "assistant", "content": llm_result.content})
            messages.append({"role": "user", "content": f"Tool output:\n{tool_output}"})
        return bool(self.task_success)
