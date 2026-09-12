"""Staged step-time estimates for the sandbox side, evaluated on a finished replay run.

For every LLM step of every original task, the sandbox scheduler could receive three estimates of the
time until the step completes, each using only what is known at that moment:

  at arrival   : remaining = hold + prefill + decode, with hold unknown (reported as the actual hold so the
                 error shown is the prefill+decode part only) -- this is the lower-bound stage
  at scheduling: the engine started the request (hold and queue over); remaining = prefill_hat + decode_hat(prior)
  at tool name : the first tool call's name is known; remaining = decode_hat(tool-conditioned)

prefill_hat = uncached prompt tokens x the calibrated prefill constant; decode_hat = expected output length
x a causal TPOT estimate (median over the previous 50 completed requests on the same instance). Output
length priors are per-tool medians measured on the trace pool excluding the run's own tasks.

Also reports, per step, whether the prompt was served from cache against the length of the tool gap that
preceded it (an LRU proxy for the DRAM-tier question: do misses fall on long gaps only?).

Usage: step_time_staged_estimates.py results/<run> --prefill-us-per-token 143 [--pool DIR ...] [--out JSON]
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


def llm_steps(trace: Path) -> list[dict]:
    """LLM steps in order with the tool called right after each (None for the final answer)."""
    steps: list[dict] = []
    for line in trace.open():
        if '"action"' not in line[:40]:
            continue
        r = json.loads(line)
        if r.get("type") != "action":
            continue
        if r["action_type"] == "llm_call":
            d = r["data"]
            sg = d.get("shadow_generation") or {}  # replay runs: prompt and cached tokens as served, engine request id
            steps.append({"ts_start": r["ts_start"], "ts_end": r["ts_end"], "action_id": r["action_id"],
                          "prompt": sg.get("prompt_tokens") or d.get("prompt_tokens") or 0,
                          "cached": sg.get("cached_prompt_tokens") or 0, "completion": d.get("completion_tokens") or 0,
                          "request_id": sg.get("request_id"), "tool": None, "tool_end": None})
        elif r["action_type"] == "tool_exec" and steps and steps[-1]["tool"] is None:
            steps[-1]["tool"] = r["data"].get("tool_name")
            steps[-1]["tool_end"] = r["ts_end"]
    return steps


def pool_priors(exclude: set[str]) -> dict[str, dict[str, float]]:
    """Per-tool output-length quantiles over the pool, excluding the run's own trace files."""
    by_tool: dict[str, list[int]] = defaultdict(list)
    for d in POOLS:
        for m in map(json.loads, (d / "MANIFEST.jsonl").open()):
            if m["flattened_name"] in exclude:
                continue
            for s in llm_steps(d / m["flattened_name"]):
                if s["completion"]:
                    by_tool[s["tool"] or "final"].append(s["completion"])
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    priors = {t: {"p10": q(v, 0.1), "p50": q(v, 0.5), "n": len(v)} for t, v in by_tool.items()}
    allv = [x for v in by_tool.values() for x in v]
    priors["*"] = {"p10": q(allv, 0.1), "p50": q(allv, 0.5), "n": len(allv)}
    return priors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--prefill-us-per-token", type=float, required=True, help="calibrated prefill constant")
    p.add_argument("--probe-predictions", type=Path,
                   help="predictions.jsonl keyed <job>/<action_id> from output_length_hidden_probe.py; adds a stage-2 variant "
                        "whose decode estimate uses the per-step predicted length instead of the pool prior")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    probe = {}
    if a.probe_predictions:
        probe = {json.loads(l)["sample_id"]: json.loads(l)["predicted_tokens"] for l in a.probe_predictions.open()}

    argv = json.loads((a.run / "replay-command.json").read_text())["argv"]
    manifest = __import__("yaml").safe_load(open(argv[argv.index("--manifest") + 1]))
    own = {Path(t["trace"]).name for t in manifest["traces"]}
    priors = pool_priors(own)

    tel = {}
    for path in sorted((a.run / "server").glob("instance-*/vllm-request-telemetry.jsonl")):
        for line in path.open():
            t = json.loads(line)
            tel[t["request_id"]] = t
    rows = [json.loads(l) for l in (a.run / "server/routing.jsonl").open()]
    arrival = {r["route_id"]: r["timestamp_s"] for r in rows if r["event"] == "arrival"}
    dispatch = {r["route_id"]: r for r in rows if r["event"] == "dispatch" and "__replacement-" not in str(r.get("job_id"))}
    finish = {r["route_id"]: r["timestamp_s"] for r in rows if r["event"] == "finish"}
    by_job: dict[str, list[dict]] = defaultdict(list)
    for k, d in dispatch.items():
        t = tel.get(d.get("engine_request_id"))
        if t and k in finish:
            by_job[str(d["job_id"])].append({"route": k, "arrival": arrival.get(k, d["timestamp_s"]), "dispatch": d["timestamp_s"],
                                             "finish": finish[k], "instance": d.get("instance", 0), **t})
    for v in by_job.values():
        v.sort(key=lambda x: x["dispatch"])

    # causal TPOT per instance: median over the previous 50 completed requests
    completed: dict[int, list[tuple[float, float]]] = defaultdict(list)  # instance -> (finish, tpot)
    for job in by_job.values():
        for x in job:
            if x["generation_tokens"] > 1:
                completed[x["instance"]].append((x["finish"], x["decode_s"] / (x["generation_tokens"] - 1)))
    for v in completed.values():
        v.sort()

    def tpot_hat(instance: int, before: float) -> float:
        hist = [tp for f, tp in completed[instance] if f < before][-50:]
        return statistics.median(hist) if len(hist) >= 5 else statistics.median(tp for _, tp in completed[instance])

    results = {"dispatch": [], "toolname": [], "lower_bound_ok": 0, "n": 0}
    gap_rows = []
    c = a.prefill_us_per_token * 1e-6
    by_request = {x["request_id"]: x for reqs in by_job.values() for x in reqs}
    waits = []
    for job, reqs in by_job.items():
        out_dir = next(iter((a.run / "output").glob(job)), None)
        trace = next(iter((out_dir / "attempt_1").glob("openclaw_host_replay.jsonl")), None) if out_dir else None
        prev_finish = None
        for s in (llm_steps(trace) if trace else []):
            x = by_request.get(s["request_id"])
            if x is None:
                continue
            uncached = max(0, s["prompt"] - s["cached"])
            prefill_hat = uncached * c
            scheduled = x["finish"] - x["inference_s"]  # queue over: the engine started prefill
            tp = tpot_hat(x["instance"], scheduled)
            actual_remaining = x["finish"] - scheduled
            tool = s["tool"] or "final"
            prior = priors.get(tool, priors["*"])
            est_sched = prefill_hat + priors["*"]["p50"] * tp
            est_tool = prefill_hat + prior["p50"] * tp
            lower = prefill_hat + priors["*"]["p10"] * tp
            results["n"] += 1
            results["lower_bound_ok"] += actual_remaining >= lower
            results.setdefault("prefill_only_ok", 0)
            results["prefill_only_ok"] += actual_remaining >= prefill_hat
            results.setdefault("remaining", []).append(actual_remaining)
            results["dispatch"].append((est_sched, actual_remaining))
            results["toolname"].append((est_tool, actual_remaining))
            if probe:
                key = f"{job}/{s['action_id']}"
                if key in probe:
                    results.setdefault("probe", []).append((prefill_hat + probe[key] * tp, actual_remaining))
            waits.append(scheduled - x["arrival"])  # stage 1: hold + queue, unknown to the sandbox side
            if prev_finish is not None:
                gap_rows.append((x["arrival"] - prev_finish, s["cached"] / s["prompt"] if s["prompt"] else 0.0))
            prev_finish = x["finish"]

    def summarize(pairs):
        errs = sorted(abs(e - t) for e, t in pairs)
        qerr = sorted(max(e, t, 0.1) / max(min(e, t), 0.1) for e, t in pairs)
        q = lambda xs, f: xs[int(f * (len(xs) - 1))]
        return {"n": len(pairs), "abs_err_p50_s": round(q(errs, .5), 2), "abs_err_p90_s": round(q(errs, .9), 2),
                "qerr_p50": round(q(qerr, .5), 2), "qerr_p90": round(q(qerr, .9), 2),
                "over_share": round(sum(e > t for e, t in pairs) / len(pairs), 3),
                "actual_p50_s": round(statistics.median(t for _, t in pairs), 2)}

    buckets = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 1e9)]
    gap_table = []
    for lo, hi in buckets:
        sel = [cs for g, cs in gap_rows if lo <= g < hi]
        if sel:
            gap_table.append({"gap_s": f"{lo}-{hi if hi < 1e9 else 'inf'}", "steps": len(sel),
                              "cached_share_mean": round(statistics.fmean(sel), 3),
                              "miss_share": round(sum(cs < 0.5 for cs in sel) / len(sel), 3)})
    wq = lambda f: round(sorted(waits)[int(f * (len(waits) - 1))], 2)
    rem = sorted(results["remaining"])
    summary = {"run": a.run.name, "steps": results["n"],
               "remaining_at_scheduling_s": {"p10": round(rem[int(.1 * (len(rem) - 1))], 2), "p50": round(rem[len(rem) // 2], 2)},
               "share_remaining_at_least": {f"{t}s": round(sum(x >= t for x in rem) / len(rem), 3) for t in (2.2, 3.0, 5.0)},
               "prefill_only_lower_bound_coverage": round(results["prefill_only_ok"] / results["n"], 4),
               "stage1_wait_hold_plus_queue_s": {"p50": wq(.5), "p90": wq(.9), "mean": round(statistics.fmean(waits), 2)},
               "lower_bound_coverage": round(results["lower_bound_ok"] / results["n"], 4),
               "at_scheduling": summarize(results["dispatch"]), "at_tool_name": summarize(results["toolname"]),
               **({"at_scheduling_probe": summarize(results["probe"])} if results.get("probe") else {}),
               "priors": {k: v for k, v in priors.items() if v["n"] >= 100}, "cache_by_gap": gap_table}
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
