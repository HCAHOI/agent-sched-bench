"""Reconcile both PD legs and worker-side KV receive records with replay."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from scripts.evaluation.check_two_instance_request_metrics import check as check_decode, rows


def check(run: Path, num_layers: int, *, final: bool = False) -> None:
    check_decode(run, final=final)
    routing = rows(run / "routing.jsonl", final)
    decisions = {r["route_id"]: r for r in routing if r["event"] == "decision"}
    prefill = {r["request_id"]: r for r in rows(run / "instance-0/vllm-request-telemetry.jsonl", final)}
    decode = {r["request_id"]: r for r in rows(run / "instance-1/vllm-request-telemetry.jsonl", final)}
    prefill_finishes = {r["route_id"]: r for r in routing if r["event"] == "prefill_finish"}
    transfers = defaultdict(list)
    allocations = defaultdict(list)
    log = (run / "instance-1/vllm.log").read_text()
    if not final and not log.endswith("\n"):
        log = log.rpartition("\n")[0]
    for line in log.splitlines():
        for marker, records in (("PPD_KV_TRANSFER ", transfers), ("PPD_KV_ALLOCATION ", allocations)):
            if marker in line:
                record, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
                records[record["request_id"]].append(record)
    releases = defaultdict(list)
    prefill_log = run / "instance-0/vllm.log"
    if prefill_log.exists():
        log = prefill_log.read_text()
        if not final and not log.endswith("\n"):
            log = log.rpartition("\n")[0]
        for line in log.splitlines():
            if "PPD_KV_TRANSFER " in line:
                record, _ = json.JSONDecoder().raw_decode(line.split("PPD_KV_TRANSFER ", 1)[1])
                assert record["transport"] == "nixl_push", record
                transfers[record["request_id"]].append(record)
            if "PPD_KV_RELEASE " in line:
                record, _ = json.JSONDecoder().raw_decode(line.split("PPD_KV_RELEASE ", 1)[1])
                releases[record["request_id"]].append(record)
    for finish in routing:
        if finish["event"] not in {"finish", "prefill_finish"} or finish["outcome"] != "complete":
            continue
        if not final and time.time() - finish["timestamp_s"] < 10:
            continue
        decision = decisions[finish["route_id"]]
        request_id = decision["engine_request_id"]
        if finish["event"] == "finish":
            assert decode[request_id]["generation_tokens"] == decision["requested_output_tokens"], decision
        if decision["routing_mode"] == "local":
            assert request_id not in prefill and request_id not in transfers, decision
            continue
        assert finish["route_id"] in prefill_finishes, decision
        terminal = prefill.get(request_id)
        assert terminal is not None and terminal["generation_tokens"] == 1, (decision, terminal)
        assert terminal["preemption_timing_complete"], terminal
        for key in ("ttft_s", "queue_s", "prefill_s", "decode_s", "e2e_s", "preempted_wait_s"):
            assert math.isfinite(terminal[key]) and terminal[key] >= 0, (key, terminal)
        if finish["event"] == "finish":
            assert allocations[request_id], f"Missing KV allocation records: {decision}"
            if final and "kv_transfer_params" in prefill_finishes[finish["route_id"]]:
                assert any(r["reason"] in {"consumer_notification", "write_complete"} for r in releases[request_id]), \
                    f"Missing producer KV release: {decision}"
                assert not any(r["reason"] == "expired" for r in releases[request_id]), decision
            if any(a["external_tokens"] > 0 for a in allocations[request_id]):
                assert transfers[request_id], f"Missing worker KV receive records: {decision}"
            else:
                assert all(a["external_tokens"] == 0 for a in allocations[request_id]), decision
                assert not transfers[request_id], decision
            for transfer in transfers[request_id]:
                assert transfer["num_layers"] == num_layers and transfer.get("transferred_bytes", transfer.get("received_bytes", 0)) > 0, transfer
                span = transfer["transfer_duration_ms"] if transfer.get("transport") in {"nixl", "nixl_push"} else transfer["receive_and_inject_span_ms"]
                assert math.isfinite(span) and span >= 0, transfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    check(args.run, args.num_layers, final=args.final)
