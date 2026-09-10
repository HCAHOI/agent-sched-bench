"""Compare one mixed56 two-instance replay run against reference runs.

Reports the Milestone 2 checklist for every pair: completion, task JCT
quantiles and makespan, token-weighted engine TPOT, common-window
throughput (original plus background), cached-prompt share on original
inputs, the paired mean-JCT difference with a source-trajectory bootstrap,
and the worst tasks. Inputs are the artifacts every run directory carries:
output/throughput_summary.json, output/<task>/attempt_*/openclaw_host_replay.jsonl
and server/instance-*/vllm-request-telemetry.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

WORST_TASKS = 6


@dataclass
class Call:
    task: str
    original: bool
    t_end: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    request_id: str | None


@dataclass
class Run:
    name: str
    anchor_s: float
    wall_time_s: float
    tasks: dict[str, dict]  # run_instance_id -> summary row (originals only)
    calls: list[Call]
    telemetry: dict[str, dict]  # engine request_id -> terminal telemetry row

    @property
    def jct(self) -> dict[str, float]:
        return {k: v["ready_to_terminal_s"] for k, v in self.tasks.items()}


def load_run(run: Path) -> Run:
    summary = json.loads((run / "output" / "throughput_summary.json").read_text())
    tasks = {t["run_instance_id"]: t for t in summary["tasks"]}
    calls: list[Call] = []
    for task_dir in sorted((run / "output").iterdir()):
        if not task_dir.is_dir():
            continue
        original = "__replacement-" not in task_dir.name
        for log in sorted(task_dir.glob("attempt_*/openclaw_host_replay.jsonl")):
            with log.open() as fh:
                for line in fh:
                    if '"llm_call_end"' not in line:
                        continue
                    row = json.loads(line)
                    if row.get("event") != "llm_call_end":
                        continue
                    d = row["data"]
                    sg = d["shadow_generation"]  # API usage as served: prompt, cached prompt, request id
                    calls.append(Call(task_dir.name, original, row["ts"], sg["prompt_tokens"], d["completion_tokens"],
                                      sg["cached_prompt_tokens"] or 0, sg["request_id"]))  # null = engine reported none
    telemetry: dict[str, dict] = {}
    for path in sorted((run / "server").glob("instance-*/vllm-request-telemetry.jsonl")):
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                telemetry[row["request_id"]] = row
    return Run(run.name, summary["arrival_zero_wall_time_s"], summary["wall_time_s"], tasks, calls, telemetry)


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def run_stats(run: Run, window_s: float) -> dict:
    jct = list(run.jct.values())
    originals = [c for c in run.calls if c.original]
    matched = [run.telemetry[c.request_id] for c in originals if c.request_id in run.telemetry]
    decode_s = sum(t["decode_s"] for t in matched if t["generation_tokens"] > 1)
    decode_tokens = sum(t["generation_tokens"] - 1 for t in matched if t["generation_tokens"] > 1)
    in_window = [c for c in run.calls if c.t_end - run.anchor_s <= window_s]
    prompt = sum(c.prompt_tokens for c in originals)
    return {
        "completed": sum(t["success"] for t in run.tasks.values()),
        "tasks": len(run.tasks),
        "requests": len(originals),
        "engine_matched_requests": len(matched),
        "mean_jct_s": statistics.fmean(jct),
        "p95_jct_s": percentile(jct, 0.95),
        "max_jct_s": max(jct),
        "makespan_s": run.wall_time_s,
        "weighted_tpot_ms": 1000 * decode_s / decode_tokens if decode_tokens else None,
        "cached_prompt_share": sum(c.cached_tokens for c in originals) / prompt if prompt else None,
        "window": {
            "window_s": window_s,
            "calls": len(in_window),
            "original_calls": sum(c.original for c in in_window),
            "steps_per_min": 60 * len(in_window) / window_s,
            "output_tokens_per_s": sum(c.completion_tokens for c in in_window) / window_s,
        },
        "worst_tasks": sorted(run.jct.items(), key=lambda kv: -kv[1])[:WORST_TASKS],
    }


def paired_bootstrap(cand: Run, ref: Run, draws: int, seed: int) -> dict:
    """Mean JCT difference over tasks paired by run_instance_id; resample source trajectories."""
    common = sorted(set(cand.tasks) & set(ref.tasks))
    diff = {k: cand.jct[k] - ref.jct[k] for k in common}
    groups: dict[str, list[float]] = {}
    for k in common:
        groups.setdefault(cand.tasks[k]["source_agent_id"], []).append(diff[k])
    sources = sorted(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        picked = [groups[s] for s in rng.choices(sources, k=len(sources))]
        flat = [d for g in picked for d in g]
        means.append(statistics.fmean(flat))
    means.sort()
    return {
        "paired_tasks": len(common),
        "paired_mean_jct_difference_s": statistics.fmean(diff.values()),
        "bootstrap_95pct_s": [percentile(means, 0.025), percentile(means, 0.975)],
        "bootstrap": {"unit": f"{len(sources)} source trajectories, replicas grouped", "draws": draws, "seed": seed,
                      "limitation": "One physical run per policy; does not estimate host-repeat variation"},
    }


def compare(cand: Run, ref: Run, draws: int, seed: int) -> dict:
    window_s = min(max(cand.jct.values()), max(ref.jct.values()))
    return {
        "reference": ref.name,
        "common_window_s": window_s,
        "candidate_stats": run_stats(cand, window_s),
        "reference_stats": run_stats(ref, window_s),
        **paired_bootstrap(cand, ref, draws, seed),
    }


def render(result: dict) -> str:
    c, r = result["candidate_stats"], result["reference_stats"]
    rows = [
        ("completed tasks", f"{c['completed']}/{c['tasks']}", f"{r['completed']}/{r['tasks']}"),
        ("original requests", c["requests"], r["requests"]),
        ("mean JCT, min", c["mean_jct_s"] / 60, r["mean_jct_s"] / 60),
        ("P95 JCT, min", c["p95_jct_s"] / 60, r["p95_jct_s"] / 60),
        ("max JCT, min", c["max_jct_s"] / 60, r["max_jct_s"] / 60),
        ("makespan, min", c["makespan_s"] / 60, r["makespan_s"] / 60),
        ("engine TPOT, ms (token-weighted)", c["weighted_tpot_ms"], r["weighted_tpot_ms"]),
        ("cached prompt share", c["cached_prompt_share"], r["cached_prompt_share"]),
        (f"steps/min over {result['common_window_s'] / 60:.1f} min", c["window"]["steps_per_min"], r["window"]["steps_per_min"]),
        ("output tokens/s, same window", c["window"]["output_tokens_per_s"], r["window"]["output_tokens_per_s"]),
    ]

    def fmt(v: object) -> str:
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    out = [f"candidate vs {result['reference']}", f"{'metric':38} {'candidate':>12} {'reference':>12}"]
    out += [f"{name:38} {fmt(a):>12} {fmt(b):>12}" for name, a, b in rows]
    lo, hi = result["bootstrap_95pct_s"]
    out.append(f"paired mean JCT difference: {result['paired_mean_jct_difference_s']:+.1f} s, 95% [{lo:+.1f}, {hi:+.1f}] "
               f"({result['bootstrap']['unit']}, {result['bootstrap']['draws']} draws)")
    out.append("worst tasks (candidate | reference):")
    for (ta, ja), (tb, jb) in zip(c["worst_tasks"], r["worst_tasks"]):
        out.append(f"  {ta:45} {ja / 60:7.2f} | {tb:45} {jb / 60:7.2f}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--reference", type=Path, action="append", required=True, help="repeatable")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, help="write the JSON result here (e.g. results/<candidate>/comparison.json)")
    a = p.parse_args()
    cand = load_run(a.candidate)
    results = [compare(cand, load_run(ref), a.draws, a.seed) for ref in a.reference]
    print("\n\n".join(render(r) for r in results))
    if a.out:
        a.out.write_text(json.dumps({"candidate": cand.name, "comparisons": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
