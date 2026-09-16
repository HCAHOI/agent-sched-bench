"""Reconcile both PD legs and worker-side KV receive records with replay.

Also reports the decode engine's prefix-cache hit share at the end of a run. That is the
counter that says whether PPD's local turns reused the conversation's KV or recomputed it:
the 2026-09-07 PPD run scored 0.012 there and measured nothing the paper describes.
It is a report, not a gate - no threshold was pre-registered for it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

from scripts.evaluation.check_two_instance_request_metrics import check as check_decode, rows


def decode_cache_share(run: Path) -> dict | None:
    """Prefix-cache hits/queries on the decode engine, from its final metrics scrape."""
    metrics = run / "instance-1/vllm-metrics-final.prom"
    if not metrics.exists():
        return None
    text = metrics.read_text()
    counts = {}
    for name in ("prefix_cache_queries_total", "prefix_cache_hits_total"):
        found = re.findall(rf"^vllm:{name}\{{[^}}]*\}}\s+([\d.eE+]+)", text, re.M)
        if not found:
            return None
        counts[name] = sum(map(float, found))
    queries = counts["prefix_cache_queries_total"]
    mode = None
    config = run / "adapter-config.json"
    if config.exists():
        mode = json.loads(config.read_text()).get("mode")
    return {"mode": mode, "prefix_cache_queries": queries, "prefix_cache_hits": counts["prefix_cache_hits_total"],
            "cached_prompt_share": counts["prefix_cache_hits_total"] / queries if queries else None}


def check(run: Path, num_layers: int, *, final: bool = False) -> None:
    check_decode(run, final=final)
    if final:
        share = decode_cache_share(run)
        assert share is not None, "final audit needs instance-1/vllm-metrics-final.prom with prefix-cache counters"
        (run / "mechanism-check.json").write_text(json.dumps(share, indent=1))
        print(f"decode engine prefix-cache share {share['cached_prompt_share']} "
              f"({share['prefix_cache_hits']:.0f} of {share['prefix_cache_queries']:.0f} prompt tokens queried), "
              f"mode {share['mode']}; near zero means every local turn re-prefilled its context", flush=True)
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
