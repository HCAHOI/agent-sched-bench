"""Summarize a PD/local length-profiling run per group and pair the two paths.

Reads results/<run>/server/length-profile/<group>.json and <group>-turns.jsonl
(client rows) and joins the engine telemetry of both workers
(server/instance-0 = P, server/instance-1 = D) by engine request id. Works for
the Poisson-arrival stages and the two-phase ``load`` stage. Turn-2 rows only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def load_telemetry(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                rows[row["request_id"]] = row
    return rows


def summarize(run: Path) -> list[dict]:
    root = run / "server" / "length-profile"
    p_tel = load_telemetry(run / "server" / "instance-0" / "vllm-request-telemetry.jsonl")
    d_tel = load_telemetry(run / "server" / "instance-1" / "vllm-request-telemetry.jsonl")
    out = []
    for meta_path in sorted(root.glob("P*.json")):
        meta = json.loads(meta_path.read_text())
        if not meta.get("success"):
            continue
        rows = [json.loads(line) for line in (root / (meta["key"] + "-turns.jsonl")).read_text().splitlines()]
        t2 = [r for r in rows if r["turn"] == 2 and r["success"]]
        cached = [r["usage"]["prompt_tokens_details"]["cached_tokens"] / r["prompt_tokens"] for r in t2]
        d_rows = [d_tel[r["request_id"]] for r in t2 if r["request_id"] in d_tel]
        p_rows = [p_tel[r["request_id"]] for r in t2 if r["request_id"] in p_tel]
        out.append({
            "key": meta["key"], "point": meta["point"], "path": meta["path"],
            "load": meta["conversation_count"], "rate": meta.get("conversation_rate"), "seed": meta["seed"],
            "huo": meta["target_huo"], "n": len(t2),
            "t2_burst_offset_s": meta.get("t2_burst_offset_s"),
            "ttft_mean_s": mean([r["ttft_s"] for r in t2]), "ttft_max_s": max(r["ttft_s"] for r in t2),
            "tpot_mean_ms": 1000 * mean([r["decode_tpot_s"] for r in t2]),
            "e2e_mean_s": mean([r["e2e_s"] for r in t2]), "e2e_max_s": max(r["e2e_s"] for r in t2),
            "client_cached_fraction_mean": mean(cached), "client_cached_fraction_min": min(cached),
            "d_queue_mean_s": mean([d["queue_s"] for d in d_rows]),
            "d_prefill_mean_s": mean([d["prefill_s"] for d in d_rows]),
            "d_preemptions": sum(d["preemption_count"] for d in d_rows),
            "p_queue_mean_s": mean([p["queue_s"] for p in p_rows]),
            "p_prefill_mean_s": mean([p["prefill_s"] for p in p_rows]),
            "workload_s": meta["workload_s"],
        })
    return out


def pair(groups: list[dict]) -> list[dict]:
    by = {(g["point"], g["load"], g["rate"], g["seed"], g["path"]): g for g in groups}
    pairs = []
    for (point, load, rate, seed, path), local in by.items():
        if path != "local" or (point, load, rate, seed, "pd") not in by:
            continue
        pd = by[point, load, rate, seed, "pd"]
        pairs.append({"point": point, "load": load, "rate": rate, "seed": seed, "huo": local["huo"],
                      "local_cached_fraction": local["client_cached_fraction_mean"],
                      "ttft_local_minus_pd_s": local["ttft_mean_s"] - pd["ttft_mean_s"],
                      "tpot_local_minus_pd_ms": local["tpot_mean_ms"] - pd["tpot_mean_ms"],
                      "e2e_local_minus_pd_s": local["e2e_mean_s"] - pd["e2e_mean_s"],
                      "local": {k: local[k] for k in ("ttft_mean_s", "tpot_mean_ms", "e2e_mean_s", "d_queue_mean_s", "d_prefill_mean_s")},
                      "pd": {k: pd[k] for k in ("ttft_mean_s", "tpot_mean_ms", "e2e_mean_s", "p_queue_mean_s", "p_prefill_mean_s", "d_queue_mean_s")}})
    return pairs


def fmt(v: float | None, spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, help="write {groups, pairs} JSON here")
    a = p.parse_args()
    groups = summarize(a.run)
    pairs = pair(groups)
    print(f"{'group':34} {'n':>3} {'cached':>6} {'TTFT':>7} {'maxTTFT':>8} {'TPOT ms':>8} {'E2E':>7} {'Dq':>6} {'Dpf':>6} {'Pq':>6} {'Ppf':>6}")
    for g in groups:
        print(f"{g['key']:34} {g['n']:3d} {fmt(g['client_cached_fraction_mean']):>6} {fmt(g['ttft_mean_s']):>7} "
              f"{fmt(g['ttft_max_s']):>8} {fmt(g['tpot_mean_ms']):>8} {fmt(g['e2e_mean_s']):>7} "
              f"{fmt(g['d_queue_mean_s']):>6} {fmt(g['d_prefill_mean_s']):>6} {fmt(g['p_queue_mean_s']):>6} {fmt(g['p_prefill_mean_s']):>6}")
    print("\nlocal minus PD (turn-2 means):")
    print(f"{'point':6} {'load':>4} {'rate':>5} {'local cached':>12} {'dTTFT s':>8} {'dTPOT ms':>9} {'dE2E s':>7}")
    for q in sorted(pairs, key=lambda q: (q["point"], q["load"], q["rate"] or 0, q["seed"])):
        print(f"{q['point']:6} {q['load']:4d} {fmt(q['rate'], 'g') if q['rate'] else '-':>5} {fmt(q['local_cached_fraction']):>12} "
              f"{q['ttft_local_minus_pd_s']:+8.2f} {q['tpot_local_minus_pd_ms']:+9.2f} {q['e2e_local_minus_pd_s']:+7.2f}")
    if a.out:
        a.out.write_text(json.dumps({"groups": groups, "pairs": pairs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
