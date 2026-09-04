"""Deployment trigger table: {command-prefix group key -> trigger_ms}.

This is the DEPLOY-SIDE consumer of the offline certification. The fresh-corpus
run certifies (H1) that the *certified-union* trigger beats a fixed deadline at
rho=0.94; its per-decision output (``rho_0.94_decisions.jsonl``) carries, for
each held-out tool call, the trigger the certified-union rule would fire at plus
the command-prefix group node it resolved to (``prior_group_key``) and the outer
fold it was scored under (``outer_fold``).

:func:`build_trigger_table` collapses those per-call, per-fold decisions into a
small static table mapping each command-prefix group key to a single deploy
trigger, and :func:`lookup_trigger` reproduces the production trie's
exact-then-prefix-backoff ORDER using the SAME production helper
(``tool_time.command.command_prefix_keys``) the offline fit used --
but NOT its per-call support gating; see :func:`lookup_trigger` for the
divergence this implies.

IMPORTANT LABELLING: the table produced here is a *deployment demo table derived
from eval artifacts* -- it is NOT itself a certified object. The certified claim
is H1 (certified-union vs deadline) on the held-out eval; a static per-group
table drops the per-row GBM-hazard component of the union (collapsed by median,
see the fold policy below), so its firing decisions only approximate the
certified rule. Use it to demonstrate the mechanism-in-the-loop, not to make a
headline number.

Fold policy (stated, honest, simplest): the decisions are leave-fold-out
cross-fitted, so one group key can carry a different trigger in each outer fold.
Per group we take the median trigger within each fold (this also collapses the
per-row hazard variation the static table cannot represent), then the median of
those per-fold medians is the deploy trigger; the per-group fold spread
(max-min of the per-fold medians) is recorded in the table so a reader can see
disagreement. A group whose final trigger is not strictly below the deadline
fires no earlier than the deadline anyway, so it is dropped -- lookups for it
fall through to the deadline fallback, which is exactly the never-fire
semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from tool_time.command import command_prefix_keys
from tool_time.policy import validate_restore_cost

_TABLE_LABEL = (
    "deployment demo table derived from eval artifacts -- not itself a certified "
    "object; the certified claim is H1 (certified-union vs deadline) on the eval"
)
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TriggerLookup:
    """Result of resolving one tool call against the deploy table."""

    trigger_ms: float
    group_key: str | None  # matched command-prefix node, or None for the fallback
    backoff_level: int  # prefix levels shed from the deepest computed key (0 = exact)
    source: str  # "group" (a prefix node hit) or "deadline" (fallback)


@dataclass(frozen=True)
class TriggerTable:
    """Loaded deploy table: group triggers + deadline fallback + config."""

    group_triggers: dict[str, float]
    deadline_ms: float
    kv_cost_ms: float
    max_prefix_depth: int
    skip_leading_cd: bool
    metadata: dict[str, Any]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "label": _TABLE_LABEL,
            "kv_cost_ms": self.kv_cost_ms,
            "deadline_ms": self.deadline_ms,
            "max_prefix_depth": self.max_prefix_depth,
            "skip_leading_cd": self.skip_leading_cd,
            "group_triggers": self.group_triggers,
            "metadata": self.metadata,
        }


def build_trigger_table(
    decisions: Iterable[dict[str, Any]],
    *,
    kv_cost_ms: float,
    deadline_ms: float,
    trigger_field: str = "certified_union_trigger_ms",
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    source_file: str | None = None,
    restore_cost_fraction: float = 0.0,
    fold_tolerance_ms: float = 1.0,
) -> TriggerTable:
    """Collapse certified-union decisions into a static deploy trigger table.

    Only rows in the ``kv_cost_ms`` cell that resolved to a command-prefix group
    node (``prior_source == 'prior_group'``) can carry an early trigger; tool-
    and global-level nodes always fall back to the deadline. Each row's deadline
    is validated against ``deadline_ms`` (fail fast on a mismatched cell). See
    the module docstring for the fold policy.
    """

    if kv_cost_ms <= 0.0 or deadline_ms <= 0.0:
        raise ValueError("kv_cost_ms and deadline_ms must be positive")
    if max_prefix_depth < 1:
        raise ValueError(f"max_prefix_depth must be >= 1, got {max_prefix_depth}")
    if fold_tolerance_ms < 0.0:
        raise ValueError("fold_tolerance_ms must be non-negative")
    validate_restore_cost(restore_cost_fraction, label="restore_cost_fraction")

    # group_key -> fold -> list of triggers
    by_group_fold: dict[str, dict[str, list[float]]] = {}
    cell_row_count = 0
    for index, row in enumerate(decisions):
        if float(row["kv_cost_ms"]) != float(kv_cost_ms):
            continue
        cell_row_count += 1
        row_deadline = float(row.get("deadline_trigger_ms", row["threshold_ms"]))
        if abs(row_deadline - deadline_ms) > 1e-6:
            raise ValueError(
                f"decision row {index} deadline {row_deadline} != --deadline-ms "
                f"{deadline_ms} for kv cell {kv_cost_ms}; wrong cell or deadline"
            )
        if row.get("prior_source") != "prior_group":
            continue  # tool/global nodes fall back to the deadline
        group_key = row["prior_group_key"]
        if not isinstance(group_key, str) or not group_key:
            raise ValueError(f"decision row {index} has a group source but no group key")
        fold = str(row["outer_fold"])
        trigger = float(row[trigger_field])
        by_group_fold.setdefault(group_key, {}).setdefault(fold, []).append(trigger)

    if cell_row_count == 0:
        raise ValueError(f"no decisions found for kv_cost_ms cell {kv_cost_ms}")

    group_triggers: dict[str, float] = {}
    fold_spread: dict[str, dict[str, Any]] = {}
    for group_key, folds in by_group_fold.items():
        per_fold = [median(triggers) for triggers in folds.values()]
        final_trigger = median(per_fold)
        spread = max(per_fold) - min(per_fold)
        if final_trigger < deadline_ms - 1e-9:
            group_triggers[group_key] = final_trigger
            fold_spread[group_key] = {
                "trigger_ms": final_trigger,
                "fold_count": len(per_fold),
                "fold_spread_ms": spread,
                "folds_agree": spread <= fold_tolerance_ms,
                "row_count": sum(len(t) for t in folds.values()),
            }

    metadata = {
        "source_file": source_file,
        "trigger_field": trigger_field,
        "restore_cost_fraction": restore_cost_fraction,
        "fold_policy": (
            "per group: median trigger within each outer fold, then median of "
            "per-fold medians; groups with final trigger >= deadline dropped "
            "(they fall back to the deadline)"
        ),
        "fold_tolerance_ms": fold_tolerance_ms,
        "cell_row_count": cell_row_count,
        "group_key_count": len(group_triggers),
        "candidate_group_count": len(by_group_fold),
        "per_group_fold_spread": fold_spread,
    }
    return TriggerTable(
        group_triggers=group_triggers,
        deadline_ms=deadline_ms,
        kv_cost_ms=kv_cost_ms,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        metadata=metadata,
    )


def load_trigger_table(path: str | Path) -> TriggerTable:
    """Load and validate a deploy table JSON; raise on a malformed file."""

    obj = json.loads(Path(path).read_text())
    required = (
        "kv_cost_ms",
        "deadline_ms",
        "max_prefix_depth",
        "skip_leading_cd",
        "group_triggers",
    )
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"trigger table {path} is missing fields: {missing}")
    group_triggers_raw = obj["group_triggers"]
    if not isinstance(group_triggers_raw, dict):
        raise ValueError(f"trigger table {path} group_triggers is not an object")
    deadline_ms = float(obj["deadline_ms"])
    group_triggers: dict[str, float] = {}
    for key, value in group_triggers_raw.items():
        trigger = float(value)
        if trigger < 0.0:
            raise ValueError(f"trigger table {path} key {key!r} has negative trigger")
        if trigger >= deadline_ms:
            raise ValueError(
                f"trigger table {path} key {key!r} trigger {trigger} >= deadline "
                f"{deadline_ms}; such a group must be dropped, not stored"
            )
        group_triggers[key] = trigger
    return TriggerTable(
        group_triggers=group_triggers,
        deadline_ms=deadline_ms,
        kv_cost_ms=float(obj["kv_cost_ms"]),
        max_prefix_depth=int(obj["max_prefix_depth"]),
        skip_leading_cd=bool(obj["skip_leading_cd"]),
        metadata=obj.get("metadata", {}),
    )


def lookup_trigger(table: TriggerTable, tool_name: str, command: str) -> TriggerLookup:
    """Resolve one tool call to a trigger via deploy-time deepest-present-node
    resolution (support gating applied at fit time, not here).

    Computes the command's prefix-key chain with the SAME production helper the
    offline fit used (``command_prefix_keys``), then tries keys deepest-first
    (exact node, then successively shorter prefixes) against the table, exactly
    the prefix-backoff ORDER the profiled predictor resolves nodes in. A miss at
    every level -- including an empty/unparseable command that yields no keys --
    returns the deadline fallback, which IS the certified policy's behaviour for
    an unknown or never-fire group.

    This does NOT fully mirror the production trie: production applies
    per-call support gating (``min_tool_history``/``min_profile_tasks``) so a
    deep node is only eligible when it has enough evidence, whereas this table
    only stores nodes that already survived that gating at fit time and are
    keyed by string alone. A deploy-time command can therefore match a deep key
    in the table that a *different* command populated (same prefix chain,
    different underlying evidence) where the production predictor, re-run on
    that exact call, might have backed off further. The divergence is bounded
    by fit-time gating and is a known simplification of the static table, not a
    bug -- it is why this artifact is a demo table, not a certified object.
    """

    chain = command_prefix_keys(
        tool_name,
        command,
        max_depth=table.max_prefix_depth,
        skip_leading_cd=table.skip_leading_cd,
    )
    deepest = len(chain) - 1
    for i in range(deepest, -1, -1):
        key = chain[i]
        if key in table.group_triggers:
            return TriggerLookup(
                trigger_ms=table.group_triggers[key],
                group_key=key,
                backoff_level=deepest - i,
                source="group",
            )
    return TriggerLookup(
        trigger_ms=table.deadline_ms,
        group_key=None,
        backoff_level=len(chain),
        source="deadline",
    )


def read_decisions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a certified-union decisions JSONL (one decision object per line)."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"decisions file {path} is empty")
    return rows


__all__ = [
    "TriggerLookup",
    "TriggerTable",
    "build_trigger_table",
    "load_trigger_table",
    "lookup_trigger",
    "read_decisions_jsonl",
]
