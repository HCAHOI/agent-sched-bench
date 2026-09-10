"""Reconcile router records with terminal vLLM telemetry during and after replay."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any


def rows(path: Path, final: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if not final and data and not data.endswith(b"\n"):
        data = data.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in data else b""
    return [json.loads(line) for line in data.splitlines()]


def check(run: Path, *, final: bool = False) -> None:
    routing = rows(run / "routing.jsonl", final)
    if final:
        assert routing, "No routing records"
    arrivals = Counter(r["route_id"] for r in routing if r["event"] == "arrival")
    finishes = Counter(r["route_id"] for r in routing if r["event"] == "finish")
    dispatches = {r["route_id"]: r for r in routing if r["event"] == "dispatch"}
    assert all(n == 1 for n in arrivals.values()), "Duplicate arrivals"
    assert all(n == 1 for n in finishes.values()), "Duplicate finishes"
    if final:
        assert arrivals == finishes, "Unmatched arrival/finish records"
    telemetry = {}
    for i in range(2):
        terminal = rows(run / f"instance-{i}/vllm-request-telemetry.jsonl", final)
        counts = Counter(r["request_id"] for r in terminal)
        assert all(n == 1 for n in counts.values()), f"Duplicate terminal telemetry on instance {i}"
        telemetry[f"http://127.0.0.1:{8000+i}"] = {r["request_id"]: r for r in terminal}
    for finish in routing:
        if finish["event"] != "finish" or finish["outcome"] != "complete":
            continue  # Timeouts and replacement cancellations remain explicit failures/cancellations.
        if not final and time.time() - finish["timestamp_s"] < 10:
            continue
        dispatch = dispatches[finish["route_id"]]
        terminal = telemetry[dispatch["backend"]].get(dispatch["engine_request_id"])
        assert terminal is not None, f"Missing terminal telemetry: {dispatch}"
        for key in ("ttft_s", "queue_s", "prefill_s", "decode_s", "e2e_s", "preempted_wait_s"):
            assert math.isfinite(terminal[key]) and terminal[key] >= 0, (key, terminal)
        assert terminal["preemption_timing_complete"], terminal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    check(args.run, final=args.final)
