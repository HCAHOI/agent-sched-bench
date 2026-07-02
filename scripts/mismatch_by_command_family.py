#!/usr/bin/env python3
"""Break down replay mismatches by command family.

Usage:
  .venv/bin/python scripts/mismatch_by_command_family.py <simulate_run_dir> [...]
  .venv/bin/python scripts/mismatch_by_command_family.py <simulate_run_dir> --json <path>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Family classification table (module-level for auditability).
# Each entry: (family_name, list_of_patterns).
# Patterns are matched against the command string (after stripping sudo/env).
# Order matters: first matching family wins.
# ---------------------------------------------------------------------------
FAMILY_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    ("pip_install", [
        re.compile(r"\b(?:pip|pip3)\s+install\b"),
        re.compile(r"\bpython\s+-m\s+pip\s+install\b"),
    ]),
    ("apt", [
        re.compile(r"\b(?:apt|apt-get|dpkg)\b"),
    ]),
    ("package_other", [
        re.compile(r"\b(?:conda|npm|yarn)\s+(?:install|add)\b"),
        re.compile(r"\b(?:cargo|gem)\s+install\b"),
        re.compile(r"\bbrew\s+install\b"),
    ]),
    ("network_fetch", [
        re.compile(r"\b(?:curl|wget)\b"),
        re.compile(r"\bgit\s+(?:clone|fetch|pull|submodule\s+update)\b"),
    ]),
    ("test", [
        re.compile(r"\b(?:pytest|tox|unittest)\b"),
    ]),
    ("build", [
        re.compile(r"\b(?:make|cmake|gcc|clang)\b"),
        re.compile(r"g\+\+"),
        re.compile(r"\bpython\s+setup\.py\s+(?:build|install)\b"),
    ]),
    ("git_local", [
        re.compile(r"\bgit\b"),
    ]),
]

# ---------------------------------------------------------------------------
# Redirection check for readonly classification
# ---------------------------------------------------------------------------
_HAS_REDIRECT = re.compile(r">[>|]?")


def _classify_family(command: str) -> str:
    """Classify a shell command string into a family.

    Strips leading ``sudo``, ``env VAR=x``, and ``bash -c`` wrappers,
    then matches against FAMILY_PATTERNS. Falls back to "readonly" if the
    command looks like a pure read-only operation, or "other".
    """
    cmd = command.strip()
    # Strip leading sudo
    cmd = re.sub(r"^sudo\s+", "", cmd)
    # Strip leading env VAR=... (one or more)
    cmd = re.sub(r"^env\s+(?:\w+=\S+\s+)*", "", cmd)
    # Strip bash -c / sh -c wrappers
    cmd = re.sub(r"^(?:bash|sh)\s+-c\s+['\"]", "", cmd)
    # Strip trailing quote from -c
    if cmd.endswith("'") or cmd.endswith('"'):
        cmd = cmd[:-1]

    for family, patterns in FAMILY_PATTERNS:
        for p in patterns:
            if p.search(cmd):
                return family

    # Readonly: simple list/ls/grep/find/head/tail/sed -n/awk (no redirect '>')
    readonly_pattern = re.compile(
        r"^\s*(?:cat|ls|grep|find|head|tail|diff|echo|pwd|which|type|"
        r"sort|uniq|wc|stat|du|df|file|od|xxd|env|printenv|"
        r"sed\s+(?:-[^>]*\b)?n\b|awk)\b"
    )
    if readonly_pattern.match(cmd) and not _HAS_REDIRECT.search(cmd):
        return "readonly"

    return "other"


def _parse_command(tool_args_json: str) -> str | None:
    """Extract the shell command from tool_args JSON.

    Supports tool_args that are a JSON object with a ``command`` key
    (for single-command exec) or ``commands`` key (ignored — multi-command
    exec is not classified per-family). Returns None for non-exec tools.
    """
    try:
        parsed = json.loads(tool_args_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    # Nested exec: openclaw wraps exec in {"exec": {"command": "..."}}
    inner = parsed.get("exec")
    if isinstance(inner, dict):
        return _command_from_payload(inner)
    return _command_from_payload(parsed)


def _command_from_payload(payload: dict) -> str | None:
    """Return the ``command`` field, or None if absent / multi-command."""
    cmd = payload.get("command")
    if isinstance(cmd, str) and cmd:
        return cmd
    cmds = payload.get("commands")
    if isinstance(cmds, list) and cmds:
        # Multi-command: use first segment for classification
        first = str(cmds[0]) if cmds else None
        return first
    return None


# ---------------------------------------------------------------------------
# Record filtering mirrors scripts/simulate_mismatch_stats.py (which keeps the
# equivalent logic inline in its main()): tool_exec actions, mismatch defined
# by replay_outcome_match.
# ---------------------------------------------------------------------------
def _load_tool_execs(trace_path: Path) -> list[dict]:
    """Load tool_exec actions from a simulate trace JSONL."""
    tool_execs: list[dict] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec":
            tool_execs.append(record)
    return tool_execs


def _discover_traces(path: Path) -> list[Path]:
    """Resolve a CLI arg to simulate trace JSONLs.

    Simulate writes flat ``<output_dir>/<run_id>.jsonl`` files
    (src/trace_collect/simulator.py). Accept a JSONL file directly, or a
    directory scanned recursively for ``*.jsonl``.
    """
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break down replay mismatches by command family."
    )
    parser.add_argument(
        "simulate_paths", nargs="+", type=Path,
        help="Simulate trace JSONL files, or directories scanned for *.jsonl",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write machine-readable results")
    args = parser.parse_args()

    # Aggregate per-family data
    family_stats: dict[str, dict] = {}
    mismatched_commands: dict[str, list[str]] = {}

    traces: list[Path] = []
    for sim_path in args.simulate_paths:
        if not sim_path.exists():
            raise SystemExit(f"no such file or directory: {sim_path}")
        traces.extend(_discover_traces(sim_path))
    if not traces:
        raise SystemExit(
            f"no *.jsonl traces found under: {', '.join(str(p) for p in args.simulate_paths)}"
        )

    for trace_path in traces:
        tool_execs = _load_tool_execs(trace_path)
        for record in tool_execs:
            data = record.get("data") or {}
            # Only replay-compared actions (skipped/no-op tools lack the field)
            if "replay_outcome_match" not in data:
                continue
            command = _parse_command(data.get("tool_args", "{}"))
            if command is None:
                continue
            family = _classify_family(command)
            if family not in family_stats:
                family_stats[family] = {"total": 0, "mismatches": 0, "reasons": Counter()}
                mismatched_commands[family] = []
            family_stats[family]["total"] += 1

            if not data["replay_outcome_match"]:
                reason = data.get("mismatch_reason") or "unknown"
                family_stats[family]["mismatches"] += 1
                family_stats[family]["reasons"][reason] += 1
                if len(mismatched_commands[family]) < 3:
                    mismatched_commands[family].append(command[:120])

    # Print report
    print("=" * 60)
    print("MISMATCH BY COMMAND FAMILY REPORT")
    print("=" * 60)

    header = (
        f"  {'Family':<18} {'Actions':>8} {'Mismatch':>9} {'Rate':>7}  Mismatch reasons"
    )
    print(header)
    print("  " + "-" * len(header))

    for family in sorted(family_stats.keys()):
        s = family_stats[family]
        rate = f"{100 * s['mismatches'] / s['total']:.1f}%" if s["total"] else "-"
        reasons_str = ", ".join(
            f"{r}:{c}" for r, c in s["reasons"].most_common()
        )
        print(
            f"  {family:<18} {s['total']:>8} {s['mismatches']:>9} {rate:>7}  {reasons_str}"
        )

    print()
    for family in sorted(family_stats.keys()):
        examples = mismatched_commands[family]
        if examples:
            print(f"  {family} mismatched command examples:")
            for ex in examples:
                print(f"    {ex}")
            print()

    # --- JSON output ---
    if args.json:
        json_output: dict = {}
        for family, s in sorted(family_stats.items()):
            json_output[family] = {
                "total": s["total"],
                "mismatches": s["mismatches"],
                "mismatch_rate": round(
                    s["mismatches"] / s["total"], 4
                ) if s["total"] else 0.0,
                "reasons": dict(s["reasons"].most_common()),
                "example_mismatched_commands": mismatched_commands[family],
            }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(json_output, indent=2), encoding="utf-8"
        )
        print(f"Machine-readable results written to: {args.json}")


if __name__ == "__main__":
    main()
