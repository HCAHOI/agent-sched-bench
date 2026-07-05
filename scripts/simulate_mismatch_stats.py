#!/usr/bin/env python3
"""Extract mismatch and forced-sync stats from a simulate trace JSONL.

Usage:
  .venv/bin/python scripts/simulate_mismatch_stats.py <simulate_trace.jsonl>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from trace_collect.mismatch import MismatchOracle
from trace_collect.output_normalize import normalize_tool_output


def _normalize_diff_signature(text: str, *, max_chars: int = 120) -> str:
    return normalize_tool_output(text[:max_chars])


def _output_diff_signature(snippet: str) -> str | None:
    """Return a normalized signature for the first source/replay diff pair."""
    source_line: str | None = None
    lines = snippet.splitlines()
    for line in lines:
        if line.startswith("- "):
            source_line = line[2:]
            continue
        if line.startswith("+ ") and source_line is not None:
            replay_line = line[2:]
            if source_line == "<missing>":
                return _normalize_diff_signature(f"+ {replay_line}")
            if replay_line == "<missing>":
                return _normalize_diff_signature(f"- {source_line}")
            return _normalize_diff_signature(f"{source_line} -> {replay_line}")
    if source_line is not None:
        return _normalize_diff_signature(source_line)
    first_line = next((line for line in lines if line), None)
    if first_line is None:
        return None
    return _normalize_diff_signature(first_line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract mismatch and forced-sync stats from a simulate trace JSONL."
    )
    parser.add_argument("simulate_trace", type=Path, help="Simulate trace JSONL")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print tiered semantic-oracle comparison against replay_outcome_match",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional CSV from scripts/label_divergences.py with human labels",
    )
    args = parser.parse_args()

    trace_path = args.simulate_trace
    if not trace_path.is_file():
        raise SystemExit(f"not a file: {trace_path}")

    tool_execs = []
    summaries = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec":
            tool_execs.append(record)
        elif record.get("type") == "summary":
            summaries.append(record)

    # --- Overall counts ---
    total = len(tool_execs)
    mismatches = [t for t in tool_execs if not (t.get("data") or {}).get("replay_outcome_match", True)]
    matched = total - len(mismatches)

    # --- Mismatch reasons ---
    reason_counts: Counter[str] = Counter()
    for t in mismatches:
        reason = (t.get("data") or {}).get("mismatch_reason", "unknown")
        reason_counts[reason] += 1

    # --- Normalized output signal ---
    non_exit_matching_reasons = {
        "command_exit_code_mismatch",
        "source_artifact_unavailable",
        "timeout_mismatch",
        "tool_success_mismatch",
    }
    normalized_exit_code_matching = []
    for t in tool_execs:
        data = t.get("data") or {}
        normalized_match = data.get("normalized_output_match")
        if not isinstance(normalized_match, bool):
            continue
        if data.get("mismatch_reason") in non_exit_matching_reasons:
            continue
        normalized_exit_code_matching.append(normalized_match)

    # --- CAS mode comparison stats ---
    cas_mode_mismatch_counts = []
    for t in tool_execs:
        data = t.get("data") or {}
        count = data.get("cas_mode_mismatch_count")
        if isinstance(count, int) and not isinstance(count, bool):
            cas_mode_mismatch_counts.append(count)
        verification = data.get("forced_sync_verification")
        if isinstance(verification, dict):
            count = verification.get("cas_mode_mismatch_count")
            if isinstance(count, int) and not isinstance(count, bool):
                cas_mode_mismatch_counts.append(count)

    # --- Forced sync stats ---
    fs_attempted = 0
    fs_success = 0
    fs_fallback = 0
    fs_missing = 0
    fs_reapplied_actions = 0
    fs_reapplied_total = 0
    fs_verified = 0
    fs_verification_failed = 0
    fs_verification_unavailable = 0
    fs_errors = Counter()
    for t in tool_execs:
        data = t.get("data") or {}
        if not data.get("forced_sync_attempted"):
            continue
        fs_attempted += 1
        if data.get("forced_sync_success"):
            fs_success += 1
        if data.get("forced_sync_fallback"):
            fs_fallback += 1
        reapplied_count = data.get("forced_sync_reapplied_action_count", 0)
        if isinstance(reapplied_count, int):
            fs_reapplied_total += reapplied_count
            if reapplied_count > 0:
                fs_reapplied_actions += 1
        verified = data.get("forced_sync_verified")
        if verified is True:
            fs_verified += 1
        elif verified is False:
            fs_verification_failed += 1
        elif "forced_sync_verified" in data:
            fs_verification_unavailable += 1
        status = data.get("forced_sync_status", "")
        if status == "checkpoint_missing":
            fs_missing += 1
        elif "failed" in status or "error" in str(data.get("forced_sync_error", "")).lower():
            fs_errors[status] += 1

    # --- Checkpoint coverage (from source traces, not simulate output) ---
    # The simulate trace does not copy checkpoint_after from source actions.
    # Read source traces from simulate metadata to get real coverage.
    cp_present = 0
    cp_skipped = 0
    cp_types: Counter[str] = Counter()
    source_trace_paths = _load_source_trace_paths(trace_path)
    if source_trace_paths:
        cp_present, cp_skipped, cp_types = _scan_source_checkpoint_coverage(
            source_trace_paths
        )
    else:
        # Fallback: scan sibling directories
        run_dir = trace_path.parent.parent  # simulate_output/../
        cp_present, cp_skipped, cp_types = _scan_source_checkpoint_coverage(
            _discover_source_traces(run_dir)
        )

    # --- Per-agent summary ---
    agent_stats: dict[str, dict] = {}
    for t in tool_execs:
        agent_id = t.get("agent_id", "unknown")
        if agent_id not in agent_stats:
            agent_stats[agent_id] = {
                "total": 0,
                "mismatches": 0,
                "fs_attempted": 0,
                "fs_success": 0,
                "fs_fallback": 0,
                "fs_verified": 0,
                "fs_reapplied": 0,
            }
        s = agent_stats[agent_id]
        s["total"] += 1
        data = t.get("data") or {}
        if not data.get("replay_outcome_match", True):
            s["mismatches"] += 1
        if data.get("forced_sync_attempted"):
            s["fs_attempted"] += 1
        if data.get("forced_sync_success"):
            s["fs_success"] += 1
        if data.get("forced_sync_fallback"):
            s["fs_fallback"] += 1
        if data.get("forced_sync_verified") is True:
            s["fs_verified"] += 1
        reapplied_count = data.get("forced_sync_reapplied_action_count", 0)
        if isinstance(reapplied_count, int):
            s["fs_reapplied"] += reapplied_count

    # --- Print report ---
    print("=" * 60)
    print("SIMULATE MISMATCH & FORCED-SYNC REPORT")
    print("=" * 60)
    print(f"Trace: {trace_path}")
    print(f"Tool exec actions: {total}")
    print(f"  Matched:         {matched} ({100*matched/total:.1f}%)" if total else "")
    print(f"  Mismatched:      {len(mismatches)} ({100*len(mismatches)/total:.1f}%)" if total else "")
    print()

    if reason_counts:
        print("Mismatch reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
        print()

    if mismatches:
        print("Output diff patterns (top 10):")
        diff_first_lines: Counter[str] = Counter()
        for t in tool_execs:
            data = t.get("data") or {}
            snippet = data.get("output_diff_snippet")
            if not snippet:
                continue
            sig = _output_diff_signature(snippet)
            if sig is not None:
                diff_first_lines[sig] += 1
        if diff_first_lines:
            for sig, count in diff_first_lines.most_common(10):
                print(f"  [{count}x] {sig}")
        else:
            print("  (no diff signatures found)")
        print()

    print("Normalized output:")
    normalized_total = len(normalized_exit_code_matching)
    if normalized_total:
        normalized_matched = sum(1 for value in normalized_exit_code_matching if value)
        print(
            "  Match rate among exit-code-matching actions: "
            f"{normalized_matched}/{normalized_total} = "
            f"{100*normalized_matched/normalized_total:.1f}%"
        )
    else:
        print("  (no exit-code-matching exec actions with normalized-output signal)")
    print()

    if cas_mode_mismatch_counts:
        mode_mismatch_actions = sum(1 for count in cas_mode_mismatch_counts if count)
        mode_mismatch_total = sum(cas_mode_mismatch_counts)
        print("CAS mode comparison:")
        print(f"  Compared records: {len(cas_mode_mismatch_counts)}")
        print(
            f"  Mode mismatches:  {mode_mismatch_total} "
            f"across {mode_mismatch_actions} actions"
        )
        print()

    print("Forced sync:")
    print(f"  Attempted:       {fs_attempted}")
    print(f"  Successful:      {fs_success}")
    print(f"  Fallback (C):    {fs_fallback}")
    print(f"  Verified:        {fs_verified}")
    print(f"  Verify failed:   {fs_verification_failed}")
    print(f"  Verify unknown:  {fs_verification_unavailable}")
    print(f"  Reapply events:  {fs_reapplied_actions}")
    print(f"  Reapplied acts:  {fs_reapplied_total}")
    print(f"  Checkpoint miss: {fs_missing}")
    if fs_errors:
        print("  Errors:")
        for status, count in fs_errors.most_common():
            print(f"    {status}: {count}")
    print()

    recovery_rate = (100 * fs_success / fs_attempted) if fs_attempted else 0
    fallback_rate = (100 * fs_fallback / fs_success) if fs_success else 0
    print(f"Recovery rate:     {fs_success}/{fs_attempted} = {recovery_rate:.1f}%")
    print(f"Fallback ratio:    {fs_fallback}/{fs_success} = {fallback_rate:.1f}% (of successful)")
    print()

    print("Checkpoint coverage (from source traces):")
    total_checkpointed = cp_present + cp_skipped
    if total_checkpointed:
        print(f"  Checkpoints created: {cp_present}")
        print(f"  Skipped (no Δ):      {cp_skipped}")
        print(f"  Coverage:            {cp_present}/{cp_present + cp_skipped} exec tools checkpointed ({100*cp_present/(cp_present+cp_skipped):.1f}%)")
    else:
        print("  (no source traces found)")
    for kind, count in cp_types.most_common():
        print(f"    {kind}: {count}")
    print()

    if len(agent_stats) > 1:
        print("Per-agent breakdown:")
        header = (
            f"  {'Agent':<40} {'Total':>6} {'Mismatch':>9} {'FS_OK':>6} "
            f"{'FS_FB':>6} {'FS_VER':>6} {'REAPPLY':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for agent_id, s in sorted(agent_stats.items()):
            short_id = agent_id[:38] + ".." if len(agent_id) > 40 else agent_id
            print(
                f"  {short_id:<40} {s['total']:>6} {s['mismatches']:>9} "
                f"{s['fs_success']:>6} {s['fs_fallback']:>6} "
                f"{s['fs_verified']:>6} {s['fs_reapplied']:>7}"
            )
        print()

    # --- Summary-level stats ---
    total_elapsed = 0.0
    total_unresolved = 0
    for summary in summaries:
        total_elapsed += summary.get("elapsed_s", 0)
        total_unresolved += summary.get("unresolved_mismatches", 0)
    print("Aggregate from summaries:")
    print(f"  Total wall time:  {total_elapsed:.1f}s")
    print(f"  Unresolved:       {total_unresolved}")

    if args.report:
        print()
        _print_oracle_report(tool_execs, labels_path=args.labels)


def _load_source_trace_paths(simulate_trace: Path) -> list[Path]:
    """Extract source trace paths from simulate trace metadata (first JSONL line)."""
    with simulate_trace.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "trace_metadata":
                continue
            entries = record.get("source_trace_entries", [])
            if entries:
                return [Path(e["source_trace"]) for e in entries if "source_trace" in e]
            # Older format: source_traces list of strings
            raw = record.get("source_traces", [])
            return [Path(p) for p in raw if isinstance(p, str)]
    return []


def _discover_source_traces(run_dir: Path) -> list[Path]:
    """Discover trace.jsonl files under run_dir/<instance>/attempt_*/."""
    traces = []
    for instance_dir in sorted(run_dir.iterdir()):
        if not instance_dir.is_dir():
            continue
        attempts = sorted(instance_dir.glob("attempt_*/trace.jsonl"))
        if attempts:
            traces.append(attempts[-1])  # latest attempt
    return traces


def _scan_source_checkpoint_coverage(
    source_paths: list[Path],
) -> tuple[int, int, Counter[str]]:
    """Scan source traces for checkpoint_after fields.

    Returns (created, skipped, kind_counts).
    """
    created = 0
    skipped = 0
    kind_counts: Counter[str] = Counter()
    for src in source_paths:
        if not src.is_file():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "action" or record.get("action_type") != "tool_exec":
                continue
            data = record.get("data") or {}
            cp = data.get("checkpoint_after")
            if isinstance(cp, dict):
                if cp.get("skipped"):
                    skipped += 1
                else:
                    kind = cp.get("kind", "unknown")
                    kind_counts[kind] += 1
                    created += 1
    return created, skipped, kind_counts


def _print_oracle_report(
    tool_execs: list[dict[str, Any]],
    *,
    labels_path: Path | None,
) -> None:
    oracle = MismatchOracle()
    compared: list[tuple[bool, bool]] = []
    exit_code_matched: list[tuple[bool, bool]] = []
    exit_code_mismatched: list[tuple[bool, bool]] = []
    raw_mismatch_breakdown: Counter[tuple[int, str]] = Counter()
    raw_mismatch_without_verdict = 0

    for record in tool_execs:
        data = _tool_data(record)
        raw_match = _optional_bool_field(data, "replay_outcome_match")
        if raw_match is None:
            continue
        semantic_match = oracle.semantic_match_from_action_data(data)
        if semantic_match is None:
            continue
        compared.append((raw_match, semantic_match))

        subset = _exit_code_subset(data)
        if subset == "matched":
            exit_code_matched.append((raw_match, semantic_match))
        elif subset == "mismatched":
            exit_code_mismatched.append((raw_match, semantic_match))

        if not raw_match:
            verdict = oracle.verdict_from_action_data(data)
            if verdict is None:
                raw_mismatch_without_verdict += 1
            else:
                raw_mismatch_breakdown[(verdict.tier, verdict.category)] += 1

    print("=" * 60)
    print("TIERED SEMANTIC ORACLE REPORT")
    print("=" * 60)
    print("Oracle vs raw match rate across all actions:")
    _print_match_comparison("all compared actions", compared)
    print()

    print("Oracle vs raw by exit-code subset:")
    _print_match_comparison("exit-code-matched", exit_code_matched)
    _print_match_comparison("exit-code-mismatched", exit_code_mismatched)
    print()

    print("Raw mismatches by oracle tier/category:")
    if raw_mismatch_breakdown:
        for tier in (1, 2, 3):
            tier_total = sum(
                count
                for (bucket_tier, _category), count in raw_mismatch_breakdown.items()
                if bucket_tier == tier
            )
            if not tier_total:
                continue
            print(f"  Tier {tier}: {tier_total}")
            for category in ("cosmetic", "content_divergent", "unclassified"):
                count = raw_mismatch_breakdown.get((tier, category), 0)
                if count:
                    print(f"    {category}: {count}")
    else:
        print("  (no raw mismatches with oracle verdicts)")
    if raw_mismatch_without_verdict:
        print(f"  Legacy fallback without oracle verdict: {raw_mismatch_without_verdict}")
    print()

    if labels_path is not None:
        _print_human_label_estimate(
            tool_execs,
            labels_path=labels_path,
            oracle=oracle,
        )
    else:
        print("Human-labeled FP/FN estimate: (no labels CSV provided)")


def _print_match_comparison(
    label: str,
    pairs: list[tuple[bool, bool]],
) -> None:
    total = len(pairs)
    if not total:
        print(f"  {label}: no actions")
        return
    raw_matches = sum(1 for raw_match, _semantic_match in pairs if raw_match)
    oracle_matches = sum(
        1 for _raw_match, semantic_match in pairs if semantic_match
    )
    agreements = sum(
        1 for raw_match, semantic_match in pairs if raw_match == semantic_match
    )
    print(
        f"  {label}: agreement {agreements}/{total} = "
        f"{100*agreements/total:.1f}%; raw match "
        f"{raw_matches}/{total} = {100*raw_matches/total:.1f}%; oracle match "
        f"{oracle_matches}/{total} = {100*oracle_matches/total:.1f}%"
    )


def _print_human_label_estimate(
    tool_execs: list[dict[str, Any]],
    *,
    labels_path: Path,
    oracle: MismatchOracle,
) -> None:
    if not labels_path.is_file():
        raise SystemExit(f"labels CSV is not a file: {labels_path}")

    labels = _load_human_labels(labels_path)
    records_by_source_action_id = _records_by_source_action_id(tool_execs)
    oracle_counts = Counter()
    raw_counts = Counter()
    matched_labels = 0
    for source_action_id, human_different in labels.items():
        record = records_by_source_action_id.get(source_action_id)
        if record is None:
            continue
        data = _tool_data(record)
        semantic_match = oracle.semantic_match_from_action_data(data)
        raw_match = _optional_bool_field(data, "replay_outcome_match")
        if semantic_match is None or raw_match is None:
            continue
        matched_labels += 1
        _add_confusion_counts(
            oracle_counts,
            predicted_different=not semantic_match,
            human_different=human_different,
        )
        _add_confusion_counts(
            raw_counts,
            predicted_different=not raw_match,
            human_different=human_different,
        )

    print("Human-labeled FP/FN estimate:")
    if not matched_labels:
        print("  (no filled human labels matched trace actions)")
        return
    _print_confusion_counts("oracle", oracle_counts, matched_labels)
    _print_confusion_counts("raw replay_outcome_match", raw_counts, matched_labels)


def _load_human_labels(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "source_action_id" not in (reader.fieldnames or []):
            raise ValueError("labels CSV must contain source_action_id")
        if "human_label_different" not in (reader.fieldnames or []):
            raise ValueError("labels CSV must contain human_label_different")
        for row in reader:
            source_action_id = row.get("source_action_id", "")
            human_different = _parse_human_label_bool(
                row.get("human_label_different", "")
            )
            if not source_action_id or human_different is None:
                continue
            if source_action_id in labels:
                raise ValueError(f"duplicate human label for {source_action_id}")
            labels[source_action_id] = human_different
    return labels


def _parse_human_label_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"true", "t", "yes", "y", "1", "different"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "same"}:
        return False
    raise ValueError(f"unknown human_label_different value: {value!r}")


def _add_confusion_counts(
    counts: Counter[str],
    *,
    predicted_different: bool,
    human_different: bool,
) -> None:
    if predicted_different and human_different:
        counts["tp"] += 1
    elif predicted_different and not human_different:
        counts["fp"] += 1
    elif not predicted_different and human_different:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def _print_confusion_counts(
    label: str,
    counts: Counter[str],
    total: int,
) -> None:
    fp = counts["fp"]
    fn = counts["fn"]
    print(
        f"  {label}: FP {fp}/{total} = {100*fp/total:.1f}%; "
        f"FN {fn}/{total} = {100*fn/total:.1f}% "
        f"(TP={counts['tp']}, TN={counts['tn']})"
    )


def _records_by_source_action_id(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        source_action_id = _source_action_id(record)
        if source_action_id in by_id:
            raise ValueError(f"duplicate source_action_id in trace: {source_action_id}")
        by_id[source_action_id] = record
    return by_id


def _source_action_id(record: dict[str, Any]) -> str:
    data = _tool_data(record)
    return str(data.get("source_action_id", record.get("action_id", "")))


def _exit_code_subset(data: dict[str, Any]) -> str | None:
    source_returncode = _optional_int_field(data, "source_returncode")
    replay_returncode = _optional_int_value(
        "replay_returncode",
        data.get("replay_returncode", data.get("returncode")),
    )
    if source_returncode is None or replay_returncode is None:
        return None
    return "matched" if source_returncode == replay_returncode else "mismatched"


def _tool_data(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"tool_exec record data must be a dict: {record!r}")
    return data


def _optional_bool_field(data: dict[str, Any], key: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean when present, got {value!r}")
    return value


def _optional_int_field(data: dict[str, Any], key: str) -> int | None:
    return _optional_int_value(key, data.get(key))


def _optional_int_value(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be int when present, got {value!r}")
    return value


if __name__ == "__main__":
    main()
