#!/usr/bin/python3
"""Stage-2 synthetic validation on the P/W/M/K/B scope (local-cgroup harness).

Each case exercises one property the full collector must get right; we report
measured per-clause values, attribution coverage, event loss, and pass/fail.
Stage-1b stays byte-identical; this is self-contained (the frozen Stage-1b
docker image is unavailable on the host). Root required.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from trace_collect import clause_telemetry as C  # noqa: E402

_HERE = Path(__file__).resolve().parent
_W = str(_HERE / "workload")
_REPS = 2

# M: allocate+touch ~200 MB (on-CPU, sampled), then idle 0.8 s (off-CPU,
# unsampled) before exit — the perf blind-spot test against the live reference.
_M_PY = (
    "python3 -c \"b=bytearray(200*1024*1024)\n"
    "for i in range(0,len(b),4096): b[i]=1\n"
    "import time; time.sleep(0.8)\""
)

CASES = {
    # concurrent pipeline: two 2-thread burners -> each clause ~2 cores, run
    # concurrently (~4 cores total); tests per-clause descendant aggregation
    # AND that concurrent clauses do not cross-leak.
    "P": f"{_W} cpu-threads 2 1.5 | {_W} cpu-threads 2 1.5",
    # multi-exec-per-pid: sh child execs env -> nice -> workload on ONE pid;
    # only the terminal burner is eligible; earlier same-pid clauses must NOT
    # inherit its CPU peak.
    "W": f"env nice -n 0 {_W} cpu-threads 1 1.3",
    # allocate/touch then idle before exit.
    "M": _M_PY,
    # killed mid-burn after 1.3 s: exit boundary must fire on SIGKILL; peak is
    # available over the truncated (>=1 s) window.
    "K": f"timeout -s KILL 1.3 {_W} cpu-threads 1 5",
    # background child outliving a non-immediate parent exit; burner CPU must
    # attribute to the background clause, clipped to quota, no cross-leak.
    "B": f"( {_W} cpu-threads 1 1.3 & sleep 0.1 ); sleep 1.3",
}


def _burner_clauses(metrics: list[C.ClauseMetrics]) -> list[C.ClauseMetrics]:
    return [m for m in metrics if m.bin == "workload"]


def _check_P(metrics, run):
    burners = _burner_clauses(metrics)
    avail = [m for m in burners if m.peak_cpu_cores is not None]
    peaks = [round(m.peak_cpu_cores, 2) for m in avail]
    cross_leak = any(m.peak_cpu_cores and m.peak_cpu_cores > 3.0 for m in avail)
    cov = [m.provenance["attribution_coverage"] for m in burners]
    ok = (
        len(avail) == 2
        and all(1.5 <= p <= 2.6 for p in peaks)
        and not cross_leak
        and all(c == 1.0 for c in cov)
    )
    return ok, {
        "eligible_burner_clauses": len(avail),
        "per_clause_peak_cpu_cores": peaks,
        "cross_leak_gt3cores": cross_leak,
        "attribution_coverage": cov,
    }


def _check_W(metrics, run):
    # group clauses by pid; find the pid with the exec chain (>=3 clauses)
    by_pid: dict[int, list] = {}
    for m in metrics:
        by_pid.setdefault(m.host_pid, []).append(m)
    chain = max(by_pid.values(), key=len)
    bins = [m.bin for m in sorted(chain, key=lambda m: m.exec_seq)]
    terminal_avail = [m for m in chain if m.terminal and m.peak_cpu_cores is not None]
    nonterminal_avail = [
        m for m in chain if not m.terminal and m.peak_cpu_cores is not None
    ]
    ok = (
        len(chain) >= 3
        and len(terminal_avail) == 1
        and 0.7 <= (terminal_avail[0].peak_cpu_cores or 0) <= 1.4
        and len(nonterminal_avail) == 0  # no cross-copy to env/nice
    )
    return ok, {
        "same_pid_clause_bins": bins,
        "terminal_peak_cpu_cores": (
            round(terminal_avail[0].peak_cpu_cores, 2) if terminal_avail else None
        ),
        "nonterminal_clauses_with_peak": [m.bin for m in nonterminal_avail],
    }


def _check_M(metrics, run):
    py = [m for m in metrics if m.bin == "python3"]
    if not py:
        return False, {"error": "no python3 clause"}
    m = max(py, key=lambda x: (x.sampled_peak_rss_mb or 0))
    oracle_mb = run.oracle_peak_rss_kb / 1024.0
    sampled = m.sampled_peak_rss_mb
    rel = abs(sampled - oracle_mb) / oracle_mb if (sampled and oracle_mb) else None
    gap = m.provenance["rss"]["max_intersample_gap_frac"]
    ok = (
        sampled is not None
        and rel is not None
        and rel <= 0.15
        and m.provenance["rss"]["perf_rss_samples"] >= 2
    )
    return ok, {
        "sampled_peak_rss_mb": sampled,
        "live_oracle_peak_rss_mb": round(oracle_mb, 1),
        "relative_error": None if rel is None else round(rel, 3),
        "idle_gap_frac_of_window": gap,
        "boundary_rss_samples": m.provenance["rss"]["boundary_rss_samples"],
        "availability": m.sampled_peak_rss_reason,
        "hiwater_kept_separate": True,
    }


def _check_K(metrics, run):
    burners = _burner_clauses(metrics)
    if not burners:
        return False, {"error": "no burner clause"}
    m = burners[0]
    ok = (
        m.provenance["boundary_coverage"]["has_exit"] is True
        and m.exit_signal == 9  # SIGKILL captured on the exit boundary
        and m.peak_cpu_cores is not None
        and 0.7 <= m.peak_cpu_cores <= 1.4
    )
    return ok, {
        "has_exit_boundary": m.provenance["boundary_coverage"]["has_exit"],
        "exit_signal": m.exit_signal,
        "peak_cpu_cores": None if m.peak_cpu_cores is None else round(m.peak_cpu_cores, 2),
        "availability": m.peak_cpu_cores_reason,
        "raw_cpu_ns_cumulative_ms": round(m.cpu_ns_cumulative / 1e6, 1),
    }


def _check_B(metrics, run):
    burners = _burner_clauses(metrics)
    avail = [m for m in burners if m.peak_cpu_cores is not None]
    # the outer sh clauses must not carry the burner's ~1 core
    sh_clauses = [m for m in metrics if m.bin in {"sh", "dash", "sleep"}]
    sh_leak = any(m.peak_cpu_cores and m.peak_cpu_cores > 0.5 for m in sh_clauses)
    ok = (
        len(avail) == 1
        and 0.7 <= (avail[0].peak_cpu_cores or 0) <= 1.4
        and (avail[0].peak_cpu_cores or 0) <= run.quota_cores
        and not sh_leak
    )
    return ok, {
        "background_burner_peak_cpu_cores": (
            round(avail[0].peak_cpu_cores, 2) if avail else None
        ),
        "quota_cores": run.quota_cores,
        "parent_sh_cpu_leak": sh_leak,
        "attribution_coverage": [m.provenance["attribution_coverage"] for m in burners],
    }


CHECKS = {"P": _check_P, "W": _check_W, "M": _check_M, "K": _check_K, "B": _check_B}


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root: sudo python3 stage2_harness.py")
    if not Path(_W).exists():
        raise SystemExit("build workload: gcc -O2 -pthread -o workload workload.c")

    t0 = time.monotonic()
    report: dict[str, dict] = {}
    for case, command in CASES.items():
        reps = []
        for rep in range(_REPS):
            run = C.collect_case(command, f"{case}_{rep}", marker="")
            metrics, gaps = C.analyze(run)
            gap_count = len(gaps)
            ok, detail = CHECKS[case](metrics, run)
            reps.append(
                {
                    "pass": ok,
                    "detail": detail,
                    "clauses": len(metrics),
                    "coverage_gap_samples": gap_count,
                    "reserve_failures": run.reserve_failures,
                    "perf_sample_count": run.perf_sample_count,
                    "wall_s": round(run.wall_ns / 1e9, 3),
                    "cgroup_cpu_s": round(run.usage_usec / 1e6, 3),
                }
            )
        report[case] = {
            "pass": all(r["pass"] for r in reps),
            "command": command,
            "reps": reps,
        }

    result = {
        "kernel": os.uname().release,
        "harness": "local-cgroup (frozen Stage-1b docker image unavailable)",
        "reps_per_case": _REPS,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "all_pass": all(v["pass"] for v in report.values()),
        "cases": report,
    }
    (_HERE / "stage2-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
