"""Per-instance load balance and task affinity from a run's routing log (any instance count).

Reads results/<run>/server/routing.jsonl (dispatch and finish rows from the least-requests
or DualMap proxy) and reports, over fixed windows between first dispatch and last finish:
prompt tokens and dispatches per instance, the max-over-mean imbalance of prompt tokens,
the share of windows above a threshold, mean in-flight requests per instance, and the
home-instance share (for each task, the fraction of its requests served by its modal instance).

Usage: instance_balance_summary.py results/<run> [--window-s 300] [--threshold 1.5] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def instance_of(row: dict) -> str:
    return str(row["instance"]) if "instance" in row else str(row["backend"]).rsplit(":", 1)[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--window-s", type=float, default=300.0)
    p.add_argument("--threshold", type=float, default=1.5, help="imbalance level counted as unbalanced")
    p.add_argument("--originals-only", action="store_true", help="ignore replacement-task requests")
    p.add_argument("--min-active-tasks", type=int, help="count a window only while at least this many tasks are unfinished "
                   "(default: the instance count), so the drain at the tail is not read as imbalance")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    rows = [json.loads(line) for line in (a.run / "server" / "routing.jsonl").open()]
    prompt_tokens: dict[str, int] = {}  # engine request id -> prompt tokens (least-requests rows carry no token count)
    for path in sorted((a.run / "server").glob("instance-*/vllm-request-telemetry.jsonl")):
        for line in path.open():
            t = json.loads(line)
            prompt_tokens[t["request_id"]] = t["prompt_tokens"]
    dispatch = {r["route_id"]: r for r in rows if r["event"] == "dispatch"}
    finish = {r["route_id"]: r for r in rows if r["event"] == "finish"}
    if a.originals_only:
        dispatch = {k: r for k, r in dispatch.items() if "__replacement-" not in str(r.get("job_id", ""))}
    instances = sorted({instance_of(r) for r in dispatch.values()})
    t0 = min(r["timestamp_s"] for r in dispatch.values())
    t1 = max(finish[k]["timestamp_s"] for k in dispatch if k in finish)
    n_windows = int((t1 - t0) // a.window_s) + 1

    tokens = [Counter() for _ in range(n_windows)]
    count = [Counter() for _ in range(n_windows)]
    busy = [Counter() for _ in range(n_windows)]  # request-seconds per instance
    for k, r in dispatch.items():
        w = int((r["timestamp_s"] - t0) // a.window_s)
        inst = instance_of(r)
        tokens[w][inst] += r.get("input_tokens") or prompt_tokens.get(r.get("engine_request_id"), 0)
        count[w][inst] += 1
        end = finish[k]["timestamp_s"] if k in finish else t1
        t = r["timestamp_s"]
        while t < end:
            wi = int((t - t0) // a.window_s)
            edge = min(end, t0 + (wi + 1) * a.window_s)
            busy[wi][inst] += edge - t
            t = edge

    task_span: dict[str, list[float]] = {}
    for k, r in dispatch.items():
        job = str(r.get("job_id"))
        end = finish[k]["timestamp_s"] if k in finish else t1
        span = task_span.setdefault(job, [r["timestamp_s"], end])
        span[0], span[1] = min(span[0], r["timestamp_s"]), max(span[1], end)
    min_active = a.min_active_tasks if a.min_active_tasks is not None else len(instances)
    windows = []
    for w in range(n_windows):
        ws, we = t0 + w * a.window_s, t0 + (w + 1) * a.window_s
        active_tasks = sum(1 for s0, s1 in task_span.values() if s0 < we and s1 > ws)
        tok = [tokens[w][i] for i in instances]
        mean = statistics.fmean(tok)
        imbalance = (max(tok) / mean) if mean > 0 and active_tasks >= min_active else None
        windows.append({"window": w, "start_s": w * a.window_s, "active_tasks": active_tasks,
                        "prompt_tokens": dict(zip(instances, tok)),
                        "dispatches": {i: count[w][i] for i in instances},
                        "mean_in_flight": {i: round(busy[w][i] / a.window_s, 2) for i in instances},
                        "imbalance": None if imbalance is None else round(imbalance, 3)})
    active = [x for x in windows if x["imbalance"] is not None]
    unbalanced = sum(1 for x in active if x["imbalance"] > a.threshold)

    per_job: dict[str, Counter] = defaultdict(Counter)
    for r in dispatch.values():
        per_job[str(r.get("job_id"))][instance_of(r)] += 1
    home_share = [c.most_common(1)[0][1] / sum(c.values()) for c in per_job.values() if sum(c.values()) > 1]

    summary = {"run": a.run.name, "instances": instances, "window_s": a.window_s, "threshold": a.threshold,
               "min_active_tasks": min_active, "windows_total": n_windows,
               "span_s": round(t1 - t0, 1), "windows": len(active), "unbalanced_windows": unbalanced,
               "unbalanced_share": round(unbalanced / len(active), 3) if active else None,
               "imbalance_mean": round(statistics.fmean(x["imbalance"] for x in active), 3) if active else None,
               "imbalance_p90": round(sorted(x["imbalance"] for x in active)[int(0.9 * (len(active) - 1))], 3) if active else None,
               "home_instance_share_mean": round(statistics.fmean(home_share), 3) if home_share else None,
               "tasks": len(per_job), "requests": len(dispatch), "per_window": windows}
    print(f"{summary['run']}: {len(instances)} instances, {len(active)} windows of {a.window_s:.0f}s; "
          f"imbalance mean {summary['imbalance_mean']} p90 {summary['imbalance_p90']}, "
          f"share > {a.threshold}: {summary['unbalanced_share']}; home-instance share {summary['home_instance_share_mean']}")
    for x in active:
        print(f"  w{x['window']:3d} active {x['active_tasks']:3d} tokens {x['prompt_tokens']} in-flight {x['mean_in_flight']} imbalance {x['imbalance']}")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
