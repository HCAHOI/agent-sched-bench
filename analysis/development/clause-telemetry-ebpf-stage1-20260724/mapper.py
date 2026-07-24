#!/usr/bin/env python3
"""Answer-free mapping from successful exec events to static shell clauses."""

from __future__ import annotations

from collections import Counter
from pathlib import PurePath
from typing import Any


ALLOWED_PAYLOAD_KEYS = {"static_clauses", "exec_events", "process_records", "timing"}


def map_events(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    extra = set(payload) - ALLOWED_PAYLOAD_KEYS
    if extra:
        raise ValueError(f"mechanism payload contains forbidden fields: {sorted(extra)}")

    clauses = payload["static_clauses"]
    records = {
        (row["host_pid"], row["exec_seq"]): row
        for row in payload["process_records"]
    }
    used_nonloop: set[str] = set()
    occurrence_counts: Counter[str] = Counter()
    mapped = []

    for event in sorted(payload["exec_events"], key=lambda row: row["t_exec_ns"]):
        argv = event["argv"]
        if not argv:
            continue
        normalized = [PurePath(argv[0]).name, *argv[1:]]
        candidates = [
            clause
            for clause in clauses
            if clause["argv"] == normalized
            and (clause["in_loop"] or clause["clause_id"] not in used_nonloop)
        ]
        base = {
            "event": event,
            "process_record": records.get((event["host_pid"], event["exec_seq"])),
        }
        if len(candidates) != 1:
            mapped.append(
                {
                    **base,
                    "status": "unresolved",
                    "reason": (
                        "multiple_static_candidates"
                        if len(candidates) > 1
                        else "no_eligible_static_candidate"
                    ),
                    "candidate_clause_ids": sorted(
                        clause["clause_id"] for clause in candidates
                    ),
                }
            )
            continue

        clause = candidates[0]
        clause_id = clause["clause_id"]
        occurrence_index = occurrence_counts[clause_id]
        occurrence_counts[clause_id] += 1
        if not clause["in_loop"]:
            used_nonloop.add(clause_id)
        mapped.append(
            {
                **base,
                "status": "resolved",
                "clause_id": clause_id,
                "clause_index": clause["clause_index"],
                "occurrence_index": occurrence_index,
            }
        )

    return {
        "mapped": mapped,
        "resolved": [row for row in mapped if row["status"] == "resolved"],
        "unresolved": [row for row in mapped if row["status"] == "unresolved"],
    }


def self_check() -> None:
    clauses = [
        {
            "clause_id": "loop",
            "clause_index": 0,
            "argv": ["sleep", "0.02"],
            "in_loop": True,
        },
        {
            "clause_id": "same-a",
            "clause_index": 1,
            "argv": ["sleep", "0.01"],
            "in_loop": False,
        },
        {
            "clause_id": "same-b",
            "clause_index": 2,
            "argv": ["sleep", "0.01"],
            "in_loop": False,
        },
    ]
    events = [
        {"host_pid": 10 + index, "exec_seq": index, "t_exec_ns": index, "argv": argv}
        for index, argv in enumerate(
            [["sleep", "0.02"]] * 3 + [["sleep", "0.01"]] * 2
        )
    ]
    records = [
        {"host_pid": row["host_pid"], "exec_seq": row["exec_seq"]}
        for row in events
    ]
    result = map_events(
        {
            "static_clauses": clauses,
            "exec_events": events,
            "process_records": records,
            "timing": {},
        }
    )
    assert [row["occurrence_index"] for row in result["resolved"]] == [0, 1, 2]
    assert len(result["unresolved"]) == 2
    assert all(
        row["candidate_clause_ids"] == ["same-a", "same-b"]
        for row in result["unresolved"]
    )
    try:
        map_events(
            {
                "static_clauses": [],
                "exec_events": [],
                "process_records": [],
                "timing": {},
                "evaluator": {},
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("evaluator payload was not rejected")


if __name__ == "__main__":
    self_check()
    print("mapper self-check passed")
