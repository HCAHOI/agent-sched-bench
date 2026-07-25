#!/usr/bin/env python3
"""Docker smoke phase 2 (user venv): bridge clause metrics -> KB, report.

Loads the phase-1 dump, maps runtime exec images to static mvdan clauses,
aggregates each clause, seeds a ClauseResourceKB, predicts the command's
per-target flags, and reports mapping gaps, coverage/loss, bridge metrics, and
KB flags. Usage:  python3 docker_smoke_bridge.py --in FILE [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from tool_resource.clause_bridge import ExecImageRecord, bridge_command  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CLASSIFIER_TARGETS,
    ClauseResourceKB,
)


def _to_record(cm: dict) -> ExecImageRecord:
    return ExecImageRecord(
        host_pid=cm["host_pid"],
        exec_seq=cm["exec_seq"],
        t_exec_ns=cm["t_exec_ns"],
        t_end_ns=cm["t_end_ns"],
        bin=cm["bin"],
        argv=tuple(cm["argv"]),
        terminal=cm["terminal"],
        cpu_windows=tuple(tuple(w) for w in cm["cpu_windows"]),
        rss_bins=tuple(tuple(b) for b in cm["rss_bins"]),
        peak_cpu_cores=cm["peak_cpu_cores"],
        peak_cpu_reason=cm["peak_cpu_cores_reason"],
        sampled_peak_rss_mb=cm["sampled_peak_rss_mb"],
        sampled_rss_reason=cm["sampled_peak_rss_reason"],
        cpu_ns_cumulative=cm["cpu_ns_cumulative"],
        exit_signal=cm["exit_signal"],
        has_causal_end=cm["has_causal_end"],
        provenance=cm["provenance"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    dump = json.loads(args.inp.read_text())

    repo = "smoke-repo"
    command = dump["command"]
    images = [_to_record(cm) for cm in dump["clause_metrics"]]
    fork_parent = {int(k): v for k, v in dump["fork_parent"].items()}
    entry_pid = dump["entry_pid"]

    result = bridge_command(
        repo, command, images, entry_pid=entry_pid, fork_parent=fork_parent,
        loss_count=dump["reserve_failures"],
    )

    # Seed a KB from the bridged observations and predict the command's flags.
    kb_flags: dict = {}
    if result.observations:
        kb = ClauseResourceKB.fit_public(result.observations)
        pred = kb.predict_command(repo, command, 1e12)
        kb_flags = {
            t: {
                "flag": pred.targets[t].flag,
                "note": pred.targets[t].note,
            }
            for t in CLASSIFIER_TARGETS
        }

    report = {
        "command": command,
        "image": dump["image"],
        "docker_exit_code": dump["docker_exit_code"],
        "gate_armed_after_launch_ms": (
            round(dump["cgroup"]["armed_after_launch_ns"] / 1e6, 2)
            if dump["cgroup"].get("armed_after_launch_ns")
            else None
        ),
        "start_gate": "armed before release (attach race eliminated)",
        "loss": {
            "reserve_failures": dump["reserve_failures"],
            "perf_sample_count": dump["perf_sample_count"],
            "coverage_gap_samples_total": dump["coverage_gap_samples"],
            "coverage_gap_samples_structural_entry_shell": dump.get(
                "structural_gap_samples"
            ),
            "coverage_gap_samples_relevant": dump.get("relevant_gap_samples"),
            "relevant_gap_pids": dump.get("relevant_gap_pids"),
        },
        "mapping": {
            "static_clause_count": result.static_clause_count,
            "mapped_observations": len(result.bridged),
            "coverage_gaps": [
                {"kind": g.kind, "detail": g.detail} for g in result.coverage_gaps
            ],
            "unobserved_builtins": result.unobserved_builtins,
        },
        "bridge_metrics": [
            {
                "bin": bc.observation.bin,
                "argv": list(bc.observation.argv),
                "latency_ms": round(bc.observation.latency_ms, 1)
                if bc.observation.latency_ms is not None
                else None,
                "peak_cpu_cores": bc.observation.peak_cpu_cores,
                "sampled_peak_rss_mb": bc.observation.sampled_peak_rss_mb,
                "availability": bc.availability,
                "mapping_evidence": bc.mapping_evidence,
                "owned_exec_images": [list(k) for k in bc.owned_exec_images],
            }
            for bc in result.bridged
        ],
        "kb_command_flags": kb_flags,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
