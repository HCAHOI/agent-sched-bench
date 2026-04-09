from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_trace_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL trace file into a DataFrame."""
    return pd.read_json(path, lines=True)


def summarize_trace_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the basic system-level analysis metrics from a trace frame."""
    if "type" not in frame.columns:
        raise ValueError(
            "Trace DataFrame missing 'type' column. "
            "Was it produced by TraceLogger? (AgentBase.get_trace() does not include 'type')"
        )
    step_rows = frame[frame["type"] == "step"].copy()
    summary_rows = frame[frame["type"] == "summary"].copy()
    throughput_steps_per_min = 0.0
    if not step_rows.empty:
        duration_s = step_rows["ts_end"].max() - step_rows["ts_start"].min()
        if duration_s > 0:
            throughput_steps_per_min = len(step_rows) / duration_s * 60.0

    if not step_rows.empty and {"agent_id", "ts_start", "ts_end"}.issubset(
        step_rows.columns
    ):
        jct_series = (
            step_rows.groupby("agent_id")
            .apply(lambda group: group["ts_end"].max() - group["ts_start"].min())
            .astype(float)
        )
    else:
        jct_series = pd.Series(dtype=float)

    return {
        "n_iterations": int(len(step_rows)),
        "n_summaries": int(len(summary_rows)),
        "throughput_steps_per_min": throughput_steps_per_min,
        "avg_jct_s": float(jct_series.mean()) if not jct_series.empty else 0.0,
        "p90_jct_s": float(jct_series.quantile(0.9)) if not jct_series.empty else 0.0,
        "p95_jct_s": float(jct_series.quantile(0.95)) if not jct_series.empty else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse JSONL traces into a summary JSON file."
    )
    parser.add_argument("trace_file")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_trace_jsonl(Path(args.trace_file))
    summary = summarize_trace_frame(frame)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
