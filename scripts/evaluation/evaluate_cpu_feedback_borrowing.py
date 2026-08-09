#!/usr/bin/env python3
"""Evaluate feedback-driven logical admission with CPU borrowing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_cpu_feedback_admission import (  # noqa: E402
    SPLIT,
    _require_committed_inputs,
    run,
)


VERSION = "cpu-feedback-borrowing-v1"
PROTOCOL = _ROOT / "analysis/development/cpu-feedback-borrowing-protocol.md"
WORK_CONSERVING_CPU = True


def _require_borrowing_inputs() -> None:
    _require_committed_inputs()
    paths = (Path(__file__).resolve(), PROTOCOL.resolve())
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *(str(path) for path in paths)],
        cwd=_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    _require_borrowing_inputs()
    result = run(work_conserving_cpu=WORK_CONSERVING_CPU)
    if result["schema"] != VERSION:
        raise ValueError("borrowing evaluator returned the wrong schema")
    result["inputs"] = {
        "split": str(SPLIT.resolve()),
        "protocol": str(PROTOCOL.resolve()),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
