#!/usr/bin/env python3
"""Pool raw outer-fold decisions from an offline-probe clock run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.tool_latency_offline_probe import aggregate_offline_probe_cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cv_root", type=Path)
    parser.add_argument("--expected-fold-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate_offline_probe_cv(
        args.cv_root,
        expected_fold_count=args.expected_fold_count,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Pooled {result['sample_count']} samples from "
        f"{result['fold_count']} folds -> {args.output}"
    )


if __name__ == "__main__":
    main()
