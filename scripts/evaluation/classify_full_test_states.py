#!/usr/bin/env python3
"""Classify full-test state packets in independent Codex contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL = "gpt-5.6-sol"
SCHEMA = "full-test-agent-state-v1"
STATES = ["unavailable", "collection_blocked", "ready", "unknown"]
RECORD_KEYS = {"query_id", "state", "evidence_event_indices", "rationale"}
QUERY_KEYS = {"query_id", "current_command", "prior_events"}
PRIOR_EVENT_KEYS = {"event_index", "command", "exit_code", "result_excerpt"}
TOP_EVENT_TYPES = {"thread.started", "turn.started", "turn.completed", "item.started", "item.updated", "item.completed"}
NON_TOOL_ITEM_TYPES = {"reasoning", "agent_message"}
PROMPT = """Classify the single JSON query below using only its current_command and
causally earlier prior_events. Do not execute commands, call tools, inspect files,
use the network, or predict latency, CPU, memory, or disk.

States:
- unavailable: the selected runner or interpreter cannot start and no later
  successful remediation is visible.
- collection_blocked: the runner starts, but the latest suite evidence stops in
  import, collection, or setup and no later remediation is visible.
- ready: a prior suite reached test execution, or every named earlier runner or
  collection blocker has a later successful explicit remediation.
- unknown: evidence is absent, conflicting, or does not meet those rules.

Use unknown when readiness is merely plausible. A non-unknown state must cite
the minimum decisive event_index values from this query. Return only the JSON
record required by the output schema.

QUERY:
"""
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(RECORD_KEYS),
    "properties": {
        "query_id": {"type": "string"},
        "state": {"enum": STATES},
        "evidence_event_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "rationale": {"type": "string"},
    },
}


def _validate_query(query: Any) -> dict[str, Any]:
    if (
        not isinstance(query, dict)
        or set(query) != QUERY_KEYS
        or not isinstance(query.get("query_id"), str)
        or not isinstance(query.get("current_command"), str)
        or not isinstance(query.get("prior_events"), list)
    ):
        raise ValueError("packet query differs from the frozen schema")
    for index, event in enumerate(query["prior_events"]):
        if (
            not isinstance(event, dict)
            or set(event) != PRIOR_EVENT_KEYS
            or event.get("event_index") != index
            or not isinstance(event.get("command"), str)
            or not isinstance(event.get("result_excerpt"), str)
            or len(event["result_excerpt"]) > 4_100
            or (
                event.get("exit_code") is not None
                and (
                    not isinstance(event["exit_code"], int)
                    or isinstance(event["exit_code"], bool)
                )
            )
        ):
            raise ValueError(f"{query['query_id']}: prior event differs from schema")
    return query


def _is_tool_free_event(event: Any) -> bool:
    if not isinstance(event, dict) or event.get("type") not in TOP_EVENT_TYPES:
        return False
    if not str(event["type"]).startswith("item."):
        return True
    item = event.get("item")
    return isinstance(item, dict) and item.get("type") in NON_TOOL_ITEM_TYPES


def _validate_record(record: Any, query: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise ValueError(f"{query['query_id']}: response fields differ from schema")
    if record["query_id"] != query["query_id"] or record["state"] not in STATES:
        raise ValueError(f"{query['query_id']}: response identity or state is invalid")
    evidence = record["evidence_event_indices"]
    available = {event["event_index"] for event in query["prior_events"]}
    if (
        not isinstance(evidence, list)
        or len(set(evidence)) != len(evidence)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index not in available
            for index in evidence
        )
        or (record["state"] != "unknown" and not evidence)
        or not isinstance(record["rationale"], str)
        or not record["rationale"].strip()
        or len(record["rationale"]) > 240
    ):
        raise ValueError(f"{query['query_id']}: response evidence or rationale is invalid")
    return record


def _classify(query: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    query_id = query["query_id"]
    response_cache = cache_dir / f"{query_id}.json"
    with tempfile.TemporaryDirectory(prefix=f"{query_id}-") as directory:
        isolated = Path(directory)
        schema_path = isolated / "schema.json"
        response_path = isolated / "response.json"
        schema_path.write_text(json.dumps(OUTPUT_SCHEMA))
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(isolated),
                "--model",
                MODEL,
                "--config",
                'service_tier="fast"',
                "--config",
                'model_reasoning_effort="medium"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(response_path),
                "--json",
                "-",
            ],
            input=PROMPT + json.dumps(query, sort_keys=True),
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(f"{query_id}: codex failed: {completed.stderr[-2000:]}")
        events = [json.loads(line) for line in completed.stdout.splitlines() if line]
        if not events or any(not _is_tool_free_event(event) for event in events):
            raise ValueError(f"{query_id}: classifier attempted forbidden tool use")
        record = _validate_record(json.loads(response_path.read_text()), query)
        response_cache.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        (cache_dir / f"{query_id}.events.jsonl").write_text(completed.stdout)
        (cache_dir / f"{query_id}.stderr.txt").write_text(completed.stderr)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    payload = json.loads(args.packets.read_text())
    if (
        set(payload) != {"schema", "classification_schema", "model", "queries"}
        or payload["schema"] != "full-test-agent-state-packets-v1"
        or payload["classification_schema"] != SCHEMA
        or payload["model"] != MODEL
        or not isinstance(payload["queries"], list)
    ):
        raise ValueError("packet artifact differs from the frozen schema")
    queries = [_validate_query(query) for query in payload["queries"]]
    expected_ids = [f"Q{index:04d}" for index in range(1, len(queries) + 1)]
    if [query.get("query_id") for query in queries] != expected_ids:
        raise ValueError("packet query IDs are not unique and ordered")

    cache_dir = args.out.parent / f"{args.out.stem}.work"
    if args.out.exists() or cache_dir.exists():
        raise FileExistsError("classification output already exists; use a fresh path")
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for start in range(0, len(queries), args.jobs):
        batch = queries[start : start + args.jobs]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for record in pool.map(lambda query: _classify(query, cache_dir), batch):
                records[record["query_id"]] = record
        print(f"classified {min(start + args.jobs, len(queries))}/{len(queries)}", flush=True)
    codex_version = subprocess.run(
        ["codex", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    artifact = {
        "schema": SCHEMA,
        "model": MODEL,
        "inference_mode": "one_ephemeral_codex_process_per_query",
        "requested_service_tier": "fast",
        "reasoning_effort": "medium",
        "temperature": "unsupported_by_codex_provider",
        "parallel_jobs": args.jobs,
        "codex_version": codex_version,
        "packet_sha256": hashlib.sha256(
            json.dumps(queries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "classifications": [records[query_id] for query_id in expected_ids],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
