"""Bridge Stage-2 exec-image telemetry to static mvdan-clause observations.

The Stage-2 collector's ``(host_pid, exec_seq)`` object is a runtime *exec-image
occurrence*, NOT a shell clause. One static mvdan clause may own a same-PID exec
chain AND forked/execed descendants:

    env nice -n 0 workload ...
      mvdan:   one clause, headed by ``env``
      runtime: same-PID exec chain env -> nice -> workload (+ descendants)

This bridge parses the ORIGINAL command with :func:`parse_command_clauses`,
maps runtime exec images to static clauses, and folds each static clause's owned
exec images into a single :class:`~tool_resource.runtime_kb.ClauseObservation`
keyed by the mvdan clause identity (``bin``, ordered ``argv``).

Two correctness properties this module guarantees:

**Time-aligned aggregation, never scalar max.** A static clause that owns
concurrent execed descendants must not have its metrics computed as the max of
per-image scalar peaks — that under-counts and can flip a heavy/light label.
Each exec image exports compact time-aligned profiles:

- ``cpu_windows``: ``(absolute_500ms_window_index, cpu_ns)`` contributions;
- ``rss_bins``: ``(absolute_20ms_bin_index, mm_identity, rss_mb)`` samples.

The bridge merges ALL owned images before reducing: for CPU it sums owned
``cpu_ns`` within each common wall window, divides by the window's actual span,
quota-clips, then takes the max window; for RSS it deduplicates identical ``mm``
per common aligned bin, sums distinct live ``mm`` RSS, then takes the max bin.
If a profile is missing/incompatible for an owned image, the target is returned
``unavailable`` — never a scalar-max fallback. Per-image scalar peaks are kept
only as diagnostics.

**Evidence-prioritized, ambiguity-preserving mapping.** Runtime exec order is
not semantic evidence of pipeline source order, so timestamps are never used to
break ties. A staged matcher assigns (1) unique exact normalized-argv matches,
then (2) unique wrapper-chain-subsequence / argv-prefix matches, then (3)
bin-only matches that are unique on both sides. Remaining ties over genuinely
distinct static identities become explicit ``ambiguous`` coverage gaps that do
not update the KB; ties over *identical* static identities map interchangeably
(the KB observation is the same either way).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from tool_resource.features import parse_command_clauses
from tool_resource.runtime_kb import ClauseObservation

# Kept in sync with the Stage-2 collector's windowing constants.
_WINDOW_NS = 500_000_000
_MIN_ELIGIBLE_SPAN_NS = 1_000_000_000  # resource_timeline: clause >= 1 s
_MIN_WINDOW_SPAN_NS = 100_000_000
_MAX_CAPTURED_ARGS = 8
_ARGV_CAPPED_FLAG = 1 << _MAX_CAPTURED_ARGS

_SHELL_BINS = frozenset({"sh", "dash", "bash", "ash", "zsh"})
_SHELL_LOOKUP_DIAGNOSTIC = re.compile(
    r"^(?:/[^:\n]+|(?:ba|da|a|z)?sh): "
    r"(?:(?:line )?\d+): "
    r"(?P<head>[A-Za-z0-9_./+@%-]+): "
    r"(?:(?:command )?not found)$"
)
_NOEXEC_BUILTINS = frozenset(
    {
        "cd",
        "export",
        "unset",
        "set",
        "true",
        "false",
        ":",
        "alias",
        "umask",
        "shift",
        "local",
        "read",
        "echo",
        "printf",
        "test",
        "[",
        "wait",
        "eval",
        "source",
        ".",
        "pwd",
        "exit",
        "return",
        "break",
        "continue",
        "trap",
    }
)
# `source` is a bash-ism: the ONLY _NOEXEC_BUILTINS member a real POSIX sh
# (dash/ash) can report "not found" for. Every other member is mandated or
# universally built in, so a "not found" diagnostic naming it can only be
# forged payload and must keep failing closed.
_DIALECT_DEPENDENT_BUILTINS = frozenset({"source"})


@dataclass(frozen=True)
class ExecImageRecord:
    """One runtime exec-image occurrence produced by the Stage-2 collector.

    ``cpu_windows`` / ``rss_bins`` are the time-aligned profiles the bridge
    merges; ``None`` means the profile is unavailable (forcing that target to be
    reported unavailable rather than scalar-max'd). ``peak_cpu_cores`` /
    ``sampled_peak_rss_mb`` are per-image scalars kept ONLY as diagnostics.
    """

    host_pid: int
    exec_seq: int
    t_exec_ns: int
    t_end_ns: int
    bin: str
    argv: tuple[str, ...]
    terminal: bool
    cpu_windows: tuple[tuple[int, int], ...] | None  # (abs_window_idx, cpu_ns)
    rss_bins: tuple[tuple[int, int, float], ...] | None  # (abs_bin, mm, rss_mb)
    peak_cpu_cores: float | None = None  # diagnostic only
    peak_cpu_reason: str = "ok"
    sampled_peak_rss_mb: float | None = None  # diagnostic only
    sampled_rss_reason: str = "ok"
    disk_read_bytes_total: int | None = None
    disk_write_bytes_total: int | None = None
    disk_cancelled_write_bytes_total: int | None = None
    disk_io_reason: str = "missing_disk_io"
    cpu_ns_cumulative: int = 0
    exit_signal: int | None = None
    normal_exit_status: int | None = None
    has_causal_end: bool = True  # real exit / next same-pid exec; else fail closed
    argv_capture_flags: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailedExecAttempt:
    """One execve/execveat syscall that returned an errno without a new image."""

    host_pid: int
    exec_seq: int
    ts_ns: int
    argv: tuple[str, ...]
    errno: int
    argv_capture_flags: int = 0


@dataclass(frozen=True)
class ShellCommandLookupFailure:
    """Source/replay-agreed shell command-not-found evidence."""

    executable_head: str
    command: str
    source_tool_call_id: str
    replay_tool_call_id: str
    source_exit_code: int
    replay_exit_code: int
    source_diagnostic: str
    replay_diagnostic: str
    source_channel: str
    replay_channel: str
    parser: str
    exit_code_semantics: str


@dataclass(frozen=True)
class SafetyGuardBlockEvidence:
    """Source/replay-agreed rejection before any shell process was started."""

    command: str
    source_command: str
    source_tool_call_id: str
    replay_tool_call_id: str
    source_result: str
    replay_result: str


@dataclass(frozen=True)
class MappingGap:
    kind: str  # unmatched_exec_image | unmatched_static_clause | ambiguous
    detail: str


@dataclass(frozen=True)
class BridgedClause:
    observation: ClauseObservation
    owned_pids: tuple[int, ...]
    owned_exec_images: tuple[tuple[int, int], ...]
    mapping_evidence: str
    disk_read_bytes_total: int | None
    disk_write_bytes_total: int | None
    disk_cancelled_write_bytes_total: int | None
    availability: dict[str, str]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class NoRuntimeExec:
    """A static clause resolved to explicit non-runtime evidence."""

    bin: str
    argv: tuple[str, ...]
    mapping_evidence: str
    attempts: tuple[FailedExecAttempt, ...] = ()
    command_lookup_failure: ShellCommandLookupFailure | None = None
    control_short_circuit: Mapping[str, Any] | None = None
    safety_guard_blocked: SafetyGuardBlockEvidence | None = None
    availability: dict[str, str] = field(
        default_factory=lambda: {
            "latency": "unknown:no_runtime_exec",
            "cpu": "unknown:no_runtime_exec",
            "memory": "unknown:no_runtime_exec",
            "disk_io": "unknown:no_runtime_exec",
        }
    )


@dataclass(frozen=True)
class BridgeResult:
    bridged: list[BridgedClause]
    no_runtime_exec: list[NoRuntimeExec]
    coverage_gaps: list[MappingGap]
    unobserved_builtins: list[str]
    static_clause_count: int

    @property
    def observations(self) -> list[ClauseObservation]:
        return [
            bc.observation
            for bc in self.bridged
            if not all(
                reason == "unknown:protocol_timeout"
                for reason in bc.availability.values()
            )
        ]


# --------------------------------------------------------------------------
# Time-aligned aggregation
# --------------------------------------------------------------------------


_MIN_RSS_SAMPLES = 2  # fail closed on insufficient merged RSS coverage


def _merge_cpu(
    owned: Sequence[ExecImageRecord], t_exec: int, t_end: int, quota: float | None
) -> tuple[float | None, str]:
    if any(i.cpu_windows is None for i in owned):
        return None, "missing_cpu_profile"
    # Quota must be present, finite, and positive; clipping is meaningless
    # otherwise. No inf fallback.
    if quota is None or not math.isfinite(quota) or quota <= 0.0:
        return None, "missing_or_inconsistent_quota"
    if (t_end - t_exec) < _MIN_ELIGIBLE_SPAN_NS:
        return None, "clause_shorter_than_1s_ineligible_for_peak"
    merged: dict[int, int] = {}
    for img in owned:
        for widx, cpu_ns in img.cpu_windows or ():
            if not isinstance(widx, int) or not isinstance(cpu_ns, int) or cpu_ns < 0:
                return None, "invalid_cpu_profile"
            merged[widx] = merged.get(widx, 0) + cpu_ns
    if not merged:
        return None, "insufficient_cpu_samples"
    peak: float | None = None
    for widx, cpu_ns in merged.items():
        win_start = widx * _WINDOW_NS
        lo = max(win_start, t_exec)
        hi = min(win_start + _WINDOW_NS, t_end)
        span = hi - lo
        if span < _MIN_WINDOW_SPAN_NS:
            continue
        rate = min(cpu_ns / span, quota)
        peak = rate if peak is None else max(peak, rate)
    if peak is None:  # every merged window was too short to time -> no 0/ok
        return None, "no_eligible_merged_window"
    if not math.isfinite(peak):
        return None, "non_finite_cpu"
    return peak, "ok"


def _merge_rss(owned: Sequence[ExecImageRecord]) -> tuple[float | None, str]:
    """Max over time of the summed RSS of concurrently-LIVE distinct mm.

    Each mm's RSS is held between its samples over its observed lifetime
    ``[min_bin, max_bin]``; at each aligned bin only mm whose lifetime spans that
    bin contribute. This prevents summing mm whose lifetimes do not overlap
    (sequential peaks in adjacent bins are never added into one figure).
    """

    if any(i.rss_bins is None for i in owned):
        return None, "missing_rss_profile"
    # mm -> {bin: rss_mb}
    per_mm: dict[int, dict[int, float]] = {}
    n_samples = 0
    for img in owned:
        for bidx, mm, rss_mb in img.rss_bins or ():
            if (
                not isinstance(bidx, int)
                or not isinstance(mm, int)
                or not math.isfinite(rss_mb)
                or rss_mb < 0.0
            ):
                return None, "invalid_rss_profile"
            slot = per_mm.setdefault(mm, {})
            slot[bidx] = max(slot.get(bidx, 0.0), rss_mb)
            n_samples += 1
    if n_samples < _MIN_RSS_SAMPLES:
        return None, "insufficient_rss_samples"
    all_bins = sorted({b for slots in per_mm.values() for b in slots})
    totals: dict[int, float] = dict.fromkeys(all_bins, 0.0)
    for slots in per_mm.values():
        sbins = sorted(slots)
        lo, hi = sbins[0], sbins[-1]  # this mm's observed lifetime
        held, j = 0.0, 0
        for b in all_bins:
            if b < lo or b > hi:
                continue  # mm not alive at bin b -> not summed
            while j < len(sbins) and sbins[j] <= b:
                held = slots[sbins[j]]
                j += 1
            totals[b] += held
    peak = max(totals.values())
    if not math.isfinite(peak):
        return None, "non_finite_rss"
    return peak, "ok"


def _merge_disk_io(
    owned: Sequence[ExecImageRecord],
) -> tuple[tuple[int, int, int] | None, str]:
    fields = (
        "disk_read_bytes_total",
        "disk_write_bytes_total",
        "disk_cancelled_write_bytes_total",
    )
    if any(getattr(image, field) is None for image in owned for field in fields):
        reasons = sorted(
            {
                image.disk_io_reason
                for image in owned
                if any(getattr(image, field) is None for field in fields)
            }
        )
        return None, "owned_image_unavailable:" + ",".join(reasons)
    values = tuple(
        sum(int(getattr(image, field)) for image in owned) for field in fields
    )
    if any(value < 0 for value in values):
        return None, "invalid_negative_disk_io"
    return values, "ok"


# --------------------------------------------------------------------------
# Evidence-prioritized staged matching
# --------------------------------------------------------------------------


def _norm(argv: Sequence[str]) -> tuple[str, ...]:
    """Basename ONLY the executable head; preserve path-valued arguments verbatim.

    ``/usr/bin/python /path/to/a.py`` -> ``("python", "/path/to/a.py")``. Basenaming
    arguments too would merge distinct commands like ``cat a/log`` and ``cat b/log``.
    """

    if not argv:
        return ()
    return (argv[0].rsplit("/", 1)[-1], *argv[1:])


def parse_shell_lookup_diagnostic(line: str) -> str | None:
    """Return the exact missing executable head from one anchored shell line."""

    match = _SHELL_LOOKUP_DIAGNOSTIC.fullmatch(line)
    return match.group("head") if match is not None else None


def _lookup_exit_semantics(
    static: Sequence[Mapping[str, Any]],
    executable_head: str,
    exit_code: int,
) -> str | None:
    if exit_code == 127:
        return "direct_command_not_found_127"
    if exit_code != 0:
        return None
    candidates = [
        index
        for index, clause in enumerate(static)
        if clause.get("argv") and clause["argv"][0] == executable_head
    ]
    if len(candidates) != 1:
        return None
    index = candidates[0]
    clause = static[index]
    if index + 1 >= len(static) or not clause.get("in_pipe"):
        return None
    next_clause = static[index + 1]
    if (
        not next_clause.get("in_pipe")
        or int(next_clause.get("pipeline_position", -1))
        != int(clause.get("pipeline_position", -1)) + 1
    ):
        return None
    return "nonfinal_pipeline_masked_0"


def shell_lookup_exit_semantics(
    command: str,
    executable_head: str,
    exit_code: int,
) -> str | None:
    """Classify the only accepted source/replay exit-code semantics."""

    parsed = parse_command_clauses(command)
    if parsed["parse_failed"]:
        return None
    return _lookup_exit_semantics(parsed["clauses"], executable_head, exit_code)


def _valid_lookup_failure(
    evidence: ShellCommandLookupFailure,
    command: str,
    static: Sequence[Mapping[str, Any]],
) -> bool:
    source_head = parse_shell_lookup_diagnostic(evidence.source_diagnostic)
    replay_head = parse_shell_lookup_diagnostic(evidence.replay_diagnostic)
    expected_semantics = _lookup_exit_semantics(
        static,
        evidence.executable_head,
        evidence.source_exit_code,
    )
    return (
        bool(evidence.source_tool_call_id)
        and bool(evidence.replay_tool_call_id)
        and evidence.command == command
        and evidence.source_exit_code == evidence.replay_exit_code
        and source_head == replay_head == evidence.executable_head
        and evidence.source_channel == "source_tool_result"
        and evidence.replay_channel in {"raw_stderr", "tool_result"}
        and evidence.parser == "anchored_shell_command_not_found_v1"
        and expected_semantics is not None
        and evidence.exit_code_semantics == expected_semantics
    )


@dataclass(frozen=True)
class _ControlState:
    status: int
    controller_clause_index: int
    controller_pid: int | None
    controller_exec_seq: int | None
    controller_evidence: str = "mapped_exec_image"
    edge_path: tuple[int, ...] = ()


def _resolve_control_short_circuits(
    static: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    assigned: Mapping[int, int],
    evidence: Mapping[int, str],
    chains: Mapping[int, Sequence[ExecImageRecord]],
    lookup_assigned: Mapping[int, ShellCommandLookupFailure],
    excluded: set[int],
) -> tuple[dict[int, Mapping[str, Any]], list[MappingGap]]:
    """Resolve parser-proven short-circuits from exact controller evidence."""

    leaf_states: dict[int, _ControlState] = {}
    for clause_index, pid in assigned.items():
        chain = chains[pid]
        terminal = chain[-1]
        if (
            evidence.get(clause_index) != "interchangeable_identical"
            and terminal.terminal
            and terminal.has_causal_end
            and terminal.exit_signal is None
            and terminal.normal_exit_status is not None
        ):
            leaf_states[clause_index] = _ControlState(
                status=terminal.normal_exit_status,
                controller_clause_index=clause_index,
                controller_pid=pid,
                controller_exec_seq=terminal.exec_seq,
            )
    for clause_index, failure in lookup_assigned.items():
        if failure.exit_code_semantics == "direct_command_not_found_127":
            leaf_states[clause_index] = _ControlState(
                status=127,
                controller_clause_index=clause_index,
                controller_pid=None,
                controller_exec_seq=None,
                controller_evidence="shell_command_lookup_failure_exact_head",
            )

    edge_states: dict[int, _ControlState] = {}
    resolved: dict[int, Mapping[str, Any]] = {}
    gaps: list[MappingGap] = []

    def operand_state(operand: Mapping[str, Any]) -> _ControlState | None:
        if operand["kind"] == "clause":
            if (
                operand["negated"]
                or operand["contains_pipeline"]
                or operand["contains_subshell"]
            ):
                return None
            return leaf_states.get(int(operand["index"]))
        if operand["kind"] == "edge":
            return edge_states.get(int(operand["index"]))
        return None

    for edge in edges:
        edge_id = int(edge["id"])
        lhs = operand_state(edge["lhs"])
        if lhs is None:
            rhs = operand_state(edge["rhs"])
            if rhs is not None:
                edge_states[edge_id] = replace(
                    rhs,
                    edge_path=(*rhs.edge_path, edge_id),
                )
            continue
        short_circuited = (edge["operator"] == "&&" and lhs.status != 0) or (
            edge["operator"] == "||" and lhs.status == 0
        )
        if not short_circuited:
            rhs = operand_state(edge["rhs"])
            if rhs is not None:
                edge_states[edge_id] = replace(
                    rhs,
                    edge_path=(*rhs.edge_path, edge_id),
                )
            continue

        rhs_indices = [int(index) for index in edge["rhs"]["clause_indices"]]
        if any(index in assigned or index in excluded for index in rhs_indices):
            gaps.append(
                MappingGap(
                    "control_flow_contradiction",
                    f"control edge {edge_id} {edge['operator']} short-circuits "
                    "an RHS clause with runtime or conflicting evidence",
                )
            )
        else:
            executable_rhs_indices = [
                index
                for index in rhs_indices
                if str(static[index]["bin"]) not in _NOEXEC_BUILTINS
            ]
            for index in executable_rhs_indices:
                if index in resolved:
                    continue
                resolved[index] = {
                    "parser": "mvdan.cc/sh/v3",
                    "control_edge_id": edge_id,
                    "control_edge_path": [*lhs.edge_path, edge_id],
                    "operator": edge["operator"],
                    "controller_clause_index": lhs.controller_clause_index,
                    "controller_bin": static[lhs.controller_clause_index]["bin"],
                    "controller_pid": lhs.controller_pid,
                    "controller_exec_seq": lhs.controller_exec_seq,
                    "controller_mapping_evidence": lhs.controller_evidence,
                    "controller_normal_exit_status": lhs.status,
                    "controlled_clause_index": index,
                    "controlled_rhs_clause_indices": rhs_indices,
                    "controlled_rhs_executable_clause_indices": (
                        executable_rhs_indices
                    ),
                    "controlled_rhs_subtree": dict(edge["rhs"]),
                    "source_replay_fidelity": ("exact_tool_result_and_exit_code"),
                }
        edge_states[edge_id] = _ControlState(
            status=lhs.status,
            controller_clause_index=lhs.controller_clause_index,
            controller_pid=lhs.controller_pid,
            controller_exec_seq=lhs.controller_exec_seq,
            controller_evidence=lhs.controller_evidence,
            edge_path=(*lhs.edge_path, edge_id),
        )
    return resolved, gaps


def _is_subsequence(sub: Sequence[str], seq: Sequence[str]) -> bool:
    it = iter(seq)
    return all(any(s == w for w in it) for s in sub)


def _evidence_tier(
    static_argv: tuple[str, ...], static_bin: str, chain: Sequence[ExecImageRecord]
) -> int | None:
    """1=exact, 2=truncated prefix, 3=wrapper/prefix, 4=bin, None=no match."""

    chain_bins = tuple(img.bin for img in chain)
    terminal_argv = _norm(chain[-1].argv)
    terminal_flags = chain[-1].argv_capture_flags
    argv_capped = any(img.argv_capture_flags & _ARGV_CAPPED_FLAG for img in chain)
    if argv_capped:
        return None
    if terminal_flags == 0 and terminal_argv == static_argv:
        return 1
    valid_word_flags = (1 << len(terminal_argv)) - 1
    truncated_flags = terminal_flags & valid_word_flags
    if (
        truncated_flags
        and terminal_flags == truncated_flags
        and len(terminal_argv) == len(static_argv)
        and all(
            static_word.startswith(runtime_word)
            if truncated_flags & (1 << index)
            else runtime_word == static_word
            for index, (runtime_word, static_word) in enumerate(
                zip(terminal_argv, static_argv, strict=True)
            )
        )
    ):
        return 2
    if len(chain_bins) >= 2 and _is_subsequence(chain_bins, static_argv):
        return 3
    if (
        len(terminal_argv) >= 2
        and len(terminal_argv) < len(static_argv)
        and static_argv[: len(terminal_argv)] == terminal_argv
    ):
        return 3
    if chain[0].bin == static_bin:
        return 4
    return None


def _components(
    statics: Sequence[int], chains: Sequence[int], tier: Mapping[tuple[int, int], int]
) -> list[tuple[list[int], list[int]]]:
    """Connected components of the static<->chain candidate bipartite graph."""

    adj: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for si in statics:
        adj.setdefault(("s", si), [])
    for pid in chains:
        adj.setdefault(("c", pid), [])
    for si, pid in tier:
        if si in statics and pid in chains:
            adj[("s", si)].append(("c", pid))
            adj[("c", pid)].append(("s", si))
    seen: set[tuple[str, int]] = set()
    comps: list[tuple[list[int], list[int]]] = []
    for node in adj:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        cs: list[int] = []
        cp: list[int] = []
        while stack:
            kind, ident = stack.pop()
            (cs if kind == "s" else cp).append(ident)
            for nb in adj[(kind, ident)]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if cs and cp:  # ignore isolated nodes (no candidates)
            comps.append((cs, cp))
    return comps


def _assign(
    statics: Mapping[int, tuple[str, tuple[str, ...]]],
    chains: Mapping[int, list[ExecImageRecord]],
) -> tuple[dict[int, int], dict[int, str], set[int]]:
    """Return (static_idx -> chain_pid, evidence, ambiguous static indices)."""

    tier: dict[tuple[int, int], int] = {}
    for si, (sbin, sargv) in statics.items():
        for pid, imgs in chains.items():
            t = _evidence_tier(sargv, sbin, imgs)
            if t is not None:
                tier[(si, pid)] = t

    assigned: dict[int, int] = {}
    used: set[int] = set()
    evidence: dict[int, str] = {}
    evidence_labels = {
        1: "tier1",
        2: "tier1_truncated_prefix",
        3: "tier2",
        4: "tier3",
    }
    for tv in (1, 2, 3, 4):
        changed = True
        while changed:
            changed = False
            for si in statics:
                if si in assigned:
                    continue
                opts = [
                    pid
                    for pid in chains
                    if pid not in used and tier.get((si, pid)) == tv
                ]
                if len(opts) != 1:
                    continue
                pid = opts[0]
                claimants = [
                    sj
                    for sj in statics
                    if sj not in assigned and tier.get((sj, pid)) == tv
                ]
                if len(claimants) == 1:
                    assigned[si] = pid
                    used.add(pid)
                    evidence[si] = evidence_labels[tv]
                    changed = True

    ambiguous: set[int] = set()
    rem_statics = [
        si
        for si in statics
        if si not in assigned
        and any(pid not in used and (si, pid) in tier for pid in chains)
    ]
    rem_chains = [pid for pid in chains if pid not in used]
    for cs, cp in _components(rem_statics, rem_chains, tier):
        identities = {statics[si] for si in cs}
        if len(identities) == 1 and len(cs) <= len(cp):
            for si, pid in zip(sorted(cs), sorted(cp), strict=False):
                assigned[si] = pid
                used.add(pid)
                evidence[si] = "interchangeable_identical"
        else:
            ambiguous.update(cs)
    return assigned, evidence, ambiguous


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def bridge_command(
    repo: str,
    command: str,
    exec_images: Sequence[ExecImageRecord],
    *,
    failed_exec_attempts: Sequence[FailedExecAttempt] = (),
    command_lookup_failure: ShellCommandLookupFailure | None = None,
    safety_guard_blocked: SafetyGuardBlockEvidence | None = None,
    allow_control_short_circuit: bool = False,
    entry_pid: int,
    fork_parent: Mapping[int, int],
    epoch_offset: float = 0.0,
    loss_count: int = 0,
    protocol_timeout: bool = False,
) -> BridgeResult:
    """Map exec images to static mvdan clauses and aggregate per clause.

    Fails closed: a ``parse_failed`` command or a run with ``loss_count > 0``
    (any collector loss makes the event stream untrustworthy) yields NO usable
    KB observations — the clauses become coverage gaps instead.
    """

    parsed = parse_command_clauses(command)
    static = parsed["clauses"]

    if parsed["parse_failed"] or loss_count > 0:
        reason = "parse_failed" if parsed["parse_failed"] else "nonzero_loss"
        return BridgeResult(
            bridged=[],
            no_runtime_exec=[],
            coverage_gaps=[
                MappingGap(reason, f"{reason}: run withheld from KB ({command!r})")
            ],
            unobserved_builtins=[],
            static_clause_count=len(static),
        )

    chains: dict[int, list[ExecImageRecord]] = {}
    for img in exec_images:
        chains.setdefault(img.host_pid, []).append(img)
    for chain in chains.values():
        chain.sort(key=lambda r: r.exec_seq)

    children: dict[int, list[int]] = {}
    for child, parent in fork_parent.items():
        children.setdefault(parent, []).append(child)

    statics = {si: (str(c["bin"]), _norm(c["argv"])) for si, c in enumerate(static)}

    def is_shell(pid: int) -> bool:
        if pid not in chains or not all(img.bin in _SHELL_BINS for img in chains[pid]):
            return False
        # Ignore orchestration shells, but preserve an explicitly requested
        # shell clause (for example ``bash installer.sh``). Bin-only evidence is
        # intentionally insufficient here: the outer ``sh -c <command>`` often
        # shares the same bin and must remain structural.
        return not any(
            _evidence_tier(static_argv, static_bin, chains[pid]) in {1, 2, 3}
            for static_bin, static_argv in statics.values()
        )

    def nearest_nonstructural_ancestor(pid: int) -> int | None:
        cur = fork_parent.get(pid)
        while cur is not None and cur != entry_pid:
            if cur in chains and not is_shell(cur):
                return cur
            cur = fork_parent.get(cur)
        return None

    first_level = {
        pid: chains[pid]
        for pid in chains
        if not is_shell(pid) and nearest_nonstructural_ancestor(pid) is None
    }
    assigned, evidence, ambiguous = _assign(statics, first_level)
    mapped_roots = set(assigned.values())
    failed_by_identity: dict[tuple[str, tuple[str, ...]], list[FailedExecAttempt]] = {}
    for attempt in failed_exec_attempts:
        normalized = _norm(attempt.argv)
        if normalized and attempt.argv_capture_flags == 0:
            failed_by_identity.setdefault((normalized[0], normalized), []).append(
                attempt
            )
    failed_static_candidates: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for si, identity in statics.items():
        if (
            si not in assigned
            and si not in ambiguous
            and identity[0] not in _NOEXEC_BUILTINS
        ):
            failed_static_candidates.setdefault(identity, []).append(si)
    failed_assigned = {
        indices[0]: tuple(failed_by_identity[identity])
        for identity, indices in failed_static_candidates.items()
        if len(indices) == 1 and identity in failed_by_identity
    }
    lookup_assigned: dict[int, ShellCommandLookupFailure] = {}
    if command_lookup_failure is not None and _valid_lookup_failure(
        command_lookup_failure,
        command,
        static,
    ):
        lookup_candidates = [
            si
            for si, clause in enumerate(static)
            if clause.get("argv")
            if si not in assigned
            and si not in ambiguous
            and si not in failed_assigned
            and clause["argv"][0] == command_lookup_failure.executable_head
            and str(clause["bin"])
            not in (_NOEXEC_BUILTINS - _DIALECT_DEPENDENT_BUILTINS)
        ]
        if len(lookup_candidates) == 1:
            lookup_assigned[lookup_candidates[0]] = command_lookup_failure
    guard_assigned: dict[int, SafetyGuardBlockEvidence] = {}
    if (
        safety_guard_blocked is not None
        and not exec_images
        and safety_guard_blocked.command == command
        and safety_guard_blocked.source_command == command
        and bool(safety_guard_blocked.source_tool_call_id)
        and bool(safety_guard_blocked.replay_tool_call_id)
        and safety_guard_blocked.source_result == safety_guard_blocked.replay_result
        and safety_guard_blocked.replay_result.startswith(
            "Error: Command blocked by safety guard ("
        )
    ):
        guard_assigned = {
            si: safety_guard_blocked
            for si, clause in enumerate(static)
            if str(clause["bin"]) not in _NOEXEC_BUILTINS
        }
    control_assigned, control_gaps = (
        _resolve_control_short_circuits(
            static,
            parsed["control_edges"],
            assigned,
            evidence,
            chains,
            lookup_assigned,
            {
                *ambiguous,
                *failed_assigned,
                *lookup_assigned,
                *guard_assigned,
            },
        )
        if allow_control_short_circuit
        else ({}, [])
    )

    bridged: list[BridgedClause] = []
    no_runtime_exec: list[NoRuntimeExec] = []
    gaps: list[MappingGap] = list(control_gaps)
    unobserved: list[str] = []
    owned_all: set[int] = set()

    for si, clause in enumerate(static):
        cbin = str(clause["bin"])
        if si in assigned:
            owned_pids = _owned_pids(assigned[si], children, mapped_roots)
            owned_images = [img for pid in owned_pids for img in chains.get(pid, [])]
            owned_all.update(owned_pids)  # consumed either way (obs or fail-closed)
            if any(not img.has_causal_end for img in owned_images):
                gaps.append(
                    MappingGap(
                        "no_causal_end",
                        f"clause {si} bin={cbin!r}: an owned exec image has no "
                        "real exit/causal end; withheld from KB",
                    )
                )
            else:
                bridged.append(
                    _aggregate(
                        repo,
                        clause,
                        owned_pids,
                        owned_images,
                        evidence[si],
                        epoch_offset,
                        protocol_timeout_terminated=(
                            protocol_timeout
                            and any(
                                image.exit_signal is not None for image in owned_images
                            )
                        ),
                    )
                )
        elif si in ambiguous:
            gaps.append(
                MappingGap(
                    "ambiguous",
                    f"clause {si} bin={cbin!r} argv={list(clause['argv'])} "
                    "has multiple equally-valid runtime chains",
                )
            )
        elif si in failed_assigned:
            no_runtime_exec.append(
                NoRuntimeExec(
                    bin=cbin,
                    argv=tuple(clause["argv"]),
                    mapping_evidence="failed_exec_exact",
                    attempts=failed_assigned[si],
                )
            )
        elif si in lookup_assigned:
            no_runtime_exec.append(
                NoRuntimeExec(
                    bin=cbin,
                    argv=tuple(clause["argv"]),
                    mapping_evidence="shell_command_lookup_failure_exact_head",
                    command_lookup_failure=lookup_assigned[si],
                )
            )
        elif si in guard_assigned:
            no_runtime_exec.append(
                NoRuntimeExec(
                    bin=cbin,
                    argv=tuple(clause["argv"]),
                    mapping_evidence="safety_guard_blocked_before_runtime",
                    safety_guard_blocked=guard_assigned[si],
                )
            )
        elif si in control_assigned:
            no_runtime_exec.append(
                NoRuntimeExec(
                    bin=cbin,
                    argv=tuple(clause["argv"]),
                    mapping_evidence="shell_control_short_circuit",
                    control_short_circuit=control_assigned[si],
                )
            )
        elif cbin in _NOEXEC_BUILTINS:
            unobserved.append(cbin)
        else:
            gaps.append(
                MappingGap(
                    "unmatched_static_clause",
                    f"clause {si} bin={cbin!r} argv={list(clause['argv'])}",
                )
            )

    used_pids = set(assigned.values())
    # A chain is "unmatched_exec_image" only if it had NO candidate static clause;
    # a chain that matched but lost to ambiguity is already covered by the
    # ambiguous static-clause gap and must not be double-reported.
    chains_with_candidate = {
        pid
        for pid, imgs in first_level.items()
        if any(
            _evidence_tier(sargv, sbin, imgs) is not None
            for sbin, sargv in statics.values()
        )
    }
    for pid in first_level:
        if (
            pid not in used_pids
            and pid not in owned_all
            and pid not in chains_with_candidate
        ):
            gaps.append(
                MappingGap(
                    "unmatched_exec_image",
                    f"first-level pid={pid} bin={chains[pid][0].bin!r} "
                    "matched no static clause",
                )
            )

    return BridgeResult(
        bridged=bridged,
        no_runtime_exec=no_runtime_exec,
        coverage_gaps=gaps,
        unobserved_builtins=unobserved,
        static_clause_count=len(static),
    )


def _owned_pids(
    root_pid: int, children: Mapping[int, Sequence[int]], mapped_roots: set[int]
) -> tuple[int, ...]:
    owned = [root_pid]
    frontier = [root_pid]
    while frontier:
        nxt: list[int] = []
        for pid in frontier:
            for child in children.get(pid, ()):
                if child in mapped_roots and child != root_pid:
                    continue  # child starts its own static clause
                if child not in owned:
                    owned.append(child)
                    nxt.append(child)
        frontier = nxt
    return tuple(owned)


def _aggregate(
    repo: str,
    clause: Mapping[str, Any],
    owned_pids: tuple[int, ...],
    owned_images: Sequence[ExecImageRecord],
    evidence: str,
    epoch_offset: float,
    *,
    protocol_timeout_terminated: bool = False,
) -> BridgedClause:
    t_exec = min(img.t_exec_ns for img in owned_images)
    t_end = max(img.t_end_ns for img in owned_images)
    # Quota must be present and consistent across owned images (a single run's
    # cgroup quota). Missing (<=0) or conflicting values -> CPU unavailable.
    quotas = {
        float(img.provenance["quota_cores"])
        for img in owned_images
        if isinstance(img.provenance.get("quota_cores"), (int, float))
        and math.isfinite(img.provenance["quota_cores"])
        and img.provenance["quota_cores"] > 0.0
    }
    quota = quotas.pop() if len(quotas) == 1 else None
    peak_cpu, cpu_reason = _merge_cpu(owned_images, t_exec, t_end, quota)
    peak_rss, rss_reason = _merge_rss(owned_images)
    disk_io, disk_io_reason = _merge_disk_io(owned_images)
    exit_signals = [i.exit_signal for i in owned_images if i.exit_signal]

    obs = ClauseObservation(
        repo=repo,
        bin=str(clause["bin"]),
        argv=tuple(clause["argv"]),
        ts_start=epoch_offset + t_exec / 1e9,
        ts_end=epoch_offset + t_end / 1e9,
        latency_ms=(None if protocol_timeout_terminated else (t_end - t_exec) / 1e6),
        peak_cpu_cores=peak_cpu,
        sampled_peak_rss_mb=peak_rss,
        cpu_ns_cumulative=sum(i.cpu_ns_cumulative for i in owned_images),
        in_loop=bool(clause.get("in_loop", False)),
        in_pipe=bool(clause.get("in_pipe", False)),
        in_subst=bool(clause.get("in_subst", False)),
        pipeline_position=int(clause.get("pipeline_position", -1)),
    )
    availability = (
        dict.fromkeys(
            ("latency", "cpu", "memory", "disk_io"),
            "unknown:protocol_timeout",
        )
        if protocol_timeout_terminated
        else {
            "latency": "ok",
            "cpu": "ok" if peak_cpu is not None else f"unknown:{cpu_reason}",
            "memory": "ok" if peak_rss is not None else f"unknown:{rss_reason}",
            "disk_io": ("ok" if disk_io is not None else f"unknown:{disk_io_reason}"),
        }
    )
    provenance = {
        "mapping_evidence": evidence,
        "owned_exec_image_count": len(owned_images),
        "boundary_coverage": {
            "has_exec": True,
            "has_exit": any(i.terminal for i in owned_images),
        },
        "exit_signal": exit_signals[0] if exit_signals else None,
        "exit_signals": exit_signals,
        "merged_cpu_reason": cpu_reason,
        "merged_rss_reason": rss_reason,
        "merged_disk_io_reason": disk_io_reason,
        "disk_io_reduction": "sum_disjoint_owned_exec_image_totals",
        "per_image_diagnostics": [
            {
                "host_pid": i.host_pid,
                "exec_seq": i.exec_seq,
                "bin": i.bin,
                "normal_exit_status": i.normal_exit_status,
                "exit_signal": i.exit_signal,
                "scalar_peak_cpu_cores": i.peak_cpu_cores,
                "scalar_sampled_peak_rss_mb": i.sampled_peak_rss_mb,
                "has_cpu_profile": i.cpu_windows is not None,
                "has_rss_profile": i.rss_bins is not None,
                "disk_io_reason": i.disk_io_reason,
                "disk_read_bytes_total": i.disk_read_bytes_total,
                "disk_write_bytes_total": i.disk_write_bytes_total,
                "disk_cancelled_write_bytes_total": (
                    i.disk_cancelled_write_bytes_total
                ),
                "disk_io_provenance": i.provenance.get("disk_io"),
                "sample_attribution": i.provenance.get("sample_attribution"),
                "identity_only_sample_count": i.provenance.get(
                    "identity_only_sample_count", 0
                ),
                "identity_only_samples": i.provenance.get("identity_only_samples", []),
            }
            for i in owned_images
        ],
    }
    return BridgedClause(
        observation=obs,
        owned_pids=owned_pids,
        owned_exec_images=tuple((i.host_pid, i.exec_seq) for i in owned_images),
        mapping_evidence=evidence,
        disk_read_bytes_total=disk_io[0] if disk_io is not None else None,
        disk_write_bytes_total=disk_io[1] if disk_io is not None else None,
        disk_cancelled_write_bytes_total=(disk_io[2] if disk_io is not None else None),
        availability=availability,
        provenance=provenance,
    )


__all__ = [
    "BridgeResult",
    "BridgedClause",
    "ExecImageRecord",
    "FailedExecAttempt",
    "MappingGap",
    "NoRuntimeExec",
    "ShellCommandLookupFailure",
    "bridge_command",
    "parse_shell_lookup_diagnostic",
    "shell_lookup_exit_semantics",
]
