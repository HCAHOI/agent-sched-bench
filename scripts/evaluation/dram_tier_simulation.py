"""Trace-driven simulation of a capacity-limited context cache (HBM + DRAM tier) under different eviction policies.

Replays a finished run's own timeline: per original task, the LLM requests (arrival, finish, prompt tokens) and the
tool that follows each step. A task's context is resident after its request finishes; on the next arrival the context
is a hit if still resident (only the new tokens are prefilled) or a miss (the whole prompt is recomputed). Capacity is
HBM plus the DRAM tier in 32B KV tokens; when exceeded, contexts of tasks that are in a tool gap are evicted by policy:

  lru       evict the context idle the longest (LMCache's CPU tier today)
  oracle    evict the context whose next arrival is farthest away (Belady; the upper bound of any return-time policy)
  toolmed   predicted next arrival = finish + per-tool median tool time from the trace pool (excluding the run's tasks)
  slowflag  evict contexts whose last tool is in the >= 2 s bucket first (per-tool bucket from the pool), then LRU

Usage: dram_tier_simulation.py results/<run> --hbm-tokens 236496 --tier-gib 48 96 144 [--kv-bytes 262144] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOLS = [REPO / "traces/exports/swe-rebench-original-flat-644-20260904",
         REPO / "traces/exports/terminal-bench-original-flat-239-20260904"]


def steps_with_tools(trace: Path) -> list[dict]:
    steps: list[dict] = []
    for line in trace.open():
        if '"action"' not in line[:40]:
            continue
        r = json.loads(line)
        if r.get("type") != "action":
            continue
        if r["action_type"] == "llm_call":
            sg = (r["data"].get("shadow_generation") or {})
            steps.append({"request_id": sg.get("request_id"), "tool": None, "tool_s": None})
        elif r["action_type"] == "tool_exec" and steps and steps[-1]["tool"] is None:
            steps[-1]["tool"] = r["data"].get("tool_name")
            steps[-1]["tool_s"] = (r["ts_end"] - r["ts_start"]) if r.get("ts_end") and r.get("ts_start") else None
    return steps


def pool_tool_times(exclude: set[str]) -> dict[str, float]:
    """Per-tool median execution time (s) over the pool, excluding the run's own traces."""
    by_tool: dict[str, list[float]] = defaultdict(list)
    for d in POOLS:
        for m in map(json.loads, (d / "MANIFEST.jsonl").open()):
            if m["flattened_name"] in exclude:
                continue
            for s in steps_with_tools(d / m["flattened_name"]):
                if s["tool"] and s["tool_s"] is not None:
                    by_tool[s["tool"]].append(s["tool_s"])
    return {t: statistics.median(v) for t, v in by_tool.items() if len(v) >= 20}


def load_requests(run: Path) -> list[dict]:
    argv = json.loads((run / "replay-command.json").read_text())["argv"]
    manifest = __import__("yaml").safe_load(open(argv[argv.index("--manifest") + 1]))
    own = {Path(t["trace"]).name for t in manifest["traces"]}
    tel = {}
    for path in (run / "server").glob("instance-*/vllm-request-telemetry.jsonl"):
        for line in path.open():
            t = json.loads(line)
            tel[t["request_id"]] = t
    rows = [json.loads(line) for line in (run / "server/routing.jsonl").open()]
    arrival = {r["route_id"]: r["timestamp_s"] for r in rows if r["event"] == "arrival"}
    finish = {r["route_id"]: r["timestamp_s"] for r in rows if r["event"] == "finish"}
    reqs = []
    for r in rows:
        if r["event"] != "dispatch" or "__replacement-" in str(r.get("job_id")) or r["route_id"] not in finish:
            continue
        t = tel.get(r.get("engine_request_id"))
        if not t:
            continue
        reqs.append({"job": str(r["job_id"]), "request_id": r["engine_request_id"], "arrival": arrival.get(r["route_id"], r["timestamp_s"]),
                     "finish": finish[r["route_id"]], "prompt": t["prompt_tokens"]})
    # tool after each step, from the replay traces
    by_req = {q["request_id"]: q for q in reqs}
    for out_dir in (run / "output").glob("*"):
        if "__replacement-" in out_dir.name:
            continue
        trace = out_dir / "attempt_1/openclaw_host_replay.jsonl"
        if trace.exists():
            for s in steps_with_tools(trace):
                if s["request_id"] in by_req:
                    by_req[s["request_id"]]["tool"] = s["tool"] or "final"
    reqs.sort(key=lambda q: q["arrival"])
    return reqs, own


def simulate(reqs: list[dict], capacity: int, policy: str, tool_median: dict[str, float], slow_tools: set[str]) -> dict:
    by_job: dict[str, list[dict]] = defaultdict(list)
    for q in reqs:
        by_job[q["job"]].append(q)
    next_arrival = {}
    for job, qs in by_job.items():
        for i, q in enumerate(qs):
            next_arrival[q["request_id"]] = qs[i + 1]["arrival"] if i + 1 < len(qs) else float("inf")
    events = sorted([(q["arrival"], 0, q) for q in reqs] + [(q["finish"], 1, q) for q in reqs], key=lambda e: (e[0], e[1]))
    resident: dict[str, dict] = {}   # job -> {"tokens", "last": finish time, "next": predicted next arrival, "slow": bool}
    in_flight: set[str] = set()
    used = 0
    recompute = new_tokens = 0
    misses = hits = 0
    for t, kind, q in events:
        job = q["job"]
        if kind == 0:  # arrival
            prev = resident.get(job)
            if prev is not None:
                hits += 1; new_tokens += max(0, q["prompt"] - prev["tokens"]); used -= prev["tokens"]
            else:
                misses += 1; recompute += q["prompt"]
            resident[job] = {"tokens": q["prompt"], "last": t, "next": float("inf"), "slow": False}
            used += q["prompt"]; in_flight.add(job)
            while used > capacity:
                cands = [j for j in resident if j not in in_flight]
                if not cands:
                    break  # in-flight contexts cannot be evicted; capacity is exceeded until something finishes
                if policy == "lru":
                    victim = min(cands, key=lambda j: resident[j]["last"])
                elif policy == "oracle":
                    victim = max(cands, key=lambda j: resident[j]["next"])
                elif policy == "toolmed":
                    victim = max(cands, key=lambda j: resident[j]["pred"])
                else:  # slowflag
                    victim = min(cands, key=lambda j: (0 if resident[j]["slow"] else 1, resident[j]["last"]))
                used -= resident.pop(victim)["tokens"]
        else:  # finish: context stays resident, now in a tool gap
            in_flight.discard(job)
            r = resident.get(job)
            if r is not None:
                tool = q.get("tool") or "final"
                r["last"] = t; r["next"] = next_arrival[q["request_id"]]
                r["pred"] = t + tool_median.get(tool, statistics.median(tool_median.values()) if tool_median else 0.0)
                r["slow"] = tool in slow_tools
    total_prompt = sum(q["prompt"] for q in reqs)
    return {"policy": policy, "capacity_tokens": capacity, "requests": len(reqs), "miss_share": round(misses / len(reqs), 3),
            "recompute_tokens": recompute, "recompute_share_of_prompt_tokens": round(recompute / total_prompt, 3),
            "new_tokens": new_tokens}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--hbm-tokens", type=int, default=236496)
    p.add_argument("--tier-gib", type=float, nargs="+", default=[48, 96, 144])
    p.add_argument("--kv-bytes", type=int, default=262144, help="KV bytes per token (Qwen3-32B: 262,144)")
    p.add_argument("--slow-threshold-s", type=float, default=2.0)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    reqs, own = load_requests(a.run)
    tool_median = pool_tool_times(own)
    slow_tools = {t for t, m in tool_median.items() if m >= a.slow_threshold_s}
    results = []
    for gib in a.tier_gib:
        cap = a.hbm_tokens + int(gib * (1 << 30) / a.kv_bytes)
        for policy in ("lru", "oracle", "toolmed", "slowflag"):
            results.append({"tier_gib": gib, **simulate(reqs, cap, policy, tool_median, slow_tools)})
    summary = {"run": a.run.name, "requests": len(reqs), "jobs": len({q["job"] for q in reqs}), "hbm_tokens": a.hbm_tokens,
               "tool_median_s": {t: round(v, 2) for t, v in sorted(tool_median.items())}, "slow_tools": sorted(slow_tools),
               "results": results}
    print(f"{a.run.name}: {len(reqs)} requests, slow tools {sorted(slow_tools)}")
    print("| tier GiB | policy | miss share | recompute share of prompt tokens |")
    for r in results:
        print(f"| {r['tier_gib']} | {r['policy']} | {r['miss_share']} | {r['recompute_share_of_prompt_tokens']} |")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
