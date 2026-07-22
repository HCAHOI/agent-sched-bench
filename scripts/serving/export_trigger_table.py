#!/usr/bin/env python
"""Export a deployment trigger table from certified-union decisions.

Reads a certified-union decisions JSONL (e.g. ``rho_0.94_decisions.jsonl`` from
the fresh-corpus certification artifacts), selects one ``--kv-cost-ms`` cell,
applies the median-of-per-fold-median fold policy, and writes a small JSON
deployment table {command-prefix group key -> trigger_ms} with a deadline
fallback and full provenance metadata.

The output is a DEPLOYMENT DEMO TABLE derived from eval artifacts -- it is NOT
itself a certified object (see spike/trigger_table.py). Every knob (kv cost,
deadline, prefix depth, cd handling, restore cost) is a flag so no
benchmark-specific value is baked in; they MUST match the offline fit config
(the fresh cert used max_prefix_depth=4, skip_leading_cd=False, guard_ms=0 so
deadline==kv_cost).

``--restore-cost-fraction`` (rho) is REQUIRED, has NO default, and is
validated two ways: it must match the fraction encoded in the decisions
filename (``rho_<fraction>_decisions...``), and it must be > 0.0. Scoring at
rho=0 never charges a misfire's swap-back cost, which manufactures early-fire
wins out of nothing (campaign finding F1) -- exactly the defect this script
must not reproduce. The measured system operating point is rho=0.94 (see the
rho directive in analysis/); a sub-operating-point fraction is out of scope
for new work without explicit human approval outside this script.

    python scripts/export_trigger_table.py \
        --decisions .../certified-union-loo-lcb/rho_0.94_decisions.jsonl \
        --kv-cost-ms 5000 --deadline-ms 5000 --restore-cost-fraction 0.94 \
        --output trigger_table_kv5000.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from spike.trigger_table import build_trigger_table, read_decisions_jsonl

# Established repo-wide naming convention for restore-cost-fraction-swept
# decisions files, e.g. rho_0.94_decisions.jsonl / rho_0.0_decisions.jsonl
# (see analysis/*/rho_*_decisions.jsonl). Used to catch a mismatched
# --restore-cost-fraction before it silently mislabels the exported table.
_RHO_FILENAME_RE = re.compile(r"rho_([0-9]+(?:\.[0-9]+)?)_decisions")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decisions", required=True, type=Path, help="decisions JSONL path")
    p.add_argument("--kv-cost-ms", required=True, type=float, help="kv cost cell to export")
    p.add_argument(
        "--deadline-ms",
        required=True,
        type=float,
        help="deadline fallback (validated against each row's deadline_trigger_ms)",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--trigger-field",
        default="certified_union_trigger_ms",
        help="decision field holding the trigger to deploy",
    )
    p.add_argument("--max-prefix-depth", type=int, default=4, help="MUST match the fit")
    p.add_argument("--skip-leading-cd", action="store_true", help="MUST match the fit")
    p.add_argument(
        "--restore-cost-fraction",
        required=True,
        type=float,
        help=(
            "restore-cost fraction (rho) the decisions were fit/scored at. "
            "REQUIRED, no default; must equal the fraction encoded in the "
            "decisions filename and be > 0.0 -- see the module docstring"
        ),
    )
    p.add_argument(
        "--fold-tolerance-ms",
        type=float,
        default=1.0,
        help="per-group fold agreement tolerance flagged in metadata",
    )
    return p


def validate_restore_cost_fraction_for_export(
    decisions_path: Path, restore_cost_fraction: float
) -> None:
    """Reject a mismatched or sub-operating-point --restore-cost-fraction.

    Raises fast rather than silently exporting a mislabeled or F1-style
    restore-free table -- see the module docstring.
    """
    if restore_cost_fraction <= 0.0:
        raise ValueError(
            f"--restore-cost-fraction {restore_cost_fraction} is not positive; "
            "rho=0.0 never charges a misfire's restore cost, which "
            "manufactures early-fire wins (campaign finding F1) -- forbidden "
            "here without explicit human approval outside this script. Use "
            "the measured operating point rho=0.94."
        )
    match = _RHO_FILENAME_RE.search(decisions_path.name)
    if match is None:
        raise ValueError(
            f"decisions file {decisions_path.name!r} does not follow the "
            "'rho_<fraction>_decisions...' naming convention, so "
            "--restore-cost-fraction cannot be validated against it; rename "
            "or point at the file this fraction was actually fit at"
        )
    filename_fraction = float(match.group(1))
    if not math.isclose(filename_fraction, restore_cost_fraction, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"--restore-cost-fraction {restore_cost_fraction} does not match "
            f"the decisions file name {decisions_path.name!r} (encodes "
            f"rho={filename_fraction}); exporting would mislabel provenance"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_restore_cost_fraction_for_export(args.decisions, args.restore_cost_fraction)
    decisions = read_decisions_jsonl(args.decisions)
    table = build_trigger_table(
        decisions,
        kv_cost_ms=args.kv_cost_ms,
        deadline_ms=args.deadline_ms,
        trigger_field=args.trigger_field,
        max_prefix_depth=args.max_prefix_depth,
        skip_leading_cd=args.skip_leading_cd,
        source_file=str(args.decisions),
        restore_cost_fraction=args.restore_cost_fraction,
        fold_tolerance_ms=args.fold_tolerance_ms,
    )
    args.output.write_text(json.dumps(table.to_json_obj(), indent=2))
    meta = table.metadata
    print(
        f"wrote {args.output}: {meta['group_key_count']} early-firing group keys "
        f"of {meta['candidate_group_count']} candidates "
        f"(kv={args.kv_cost_ms}ms, deadline={args.deadline_ms}ms, "
        f"{meta['cell_row_count']} rows in cell)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
