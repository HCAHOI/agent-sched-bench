"""Post-hoc delta computation over absolute sim_metrics snapshots.

Phase 2 of the trace-sim-vastai-pipeline plan. The simulator stores
absolute `PreemptionSnapshot` field values per `llm_call` action under
`TraceAction.data.sim_metrics.vllm_scheduler_snapshot` (per CLAUDE.md
"preserve all intermediate outputs"). This module computes derived
delta values at analysis time — separating the lossless collection
path from the lossy aggregation path.

Field type rules (sourced verbatim from
`.omc/plans/phase0-schemas.md` section (b) — DO NOT silently broaden
the whitelist):

- `num_preemptions_total`  → COUNTER (monotonic non-decreasing).
                             Delta = current - previous = preemptions
                             that occurred in this interval.
- `gpu_cache_usage_perc`   → GAUGE (instantaneous fraction in [0, 1]).
                             Delta is meaningless ("change in
                             utilization" is not utilization).
- `cpu_cache_usage_perc`   → GAUGE — same.
- `gpu_prefix_cache_hit_rate` → RATIO (hits / lookups).
                                Delta of a hit rate is not a hit rate.
- `cpu_prefix_cache_hit_rate` → RATIO — same.

Any attempt to compute a delta over a gauge or ratio field MUST raise
`TypeError` with a descriptive message naming the field. The
`_DELTA_VALID_FIELDS` whitelist is the single source of truth.
"""

from __future__ import annotations

from typing import Any

from harness.scheduler_hooks import PreemptionSnapshot


# Whitelist of PreemptionSnapshot fields where pairwise subtraction
# yields a meaningful value. Sourced from .omc/plans/phase0-schemas.md
# section (b). Adding fields here MUST be accompanied by a citation
# update in that doc.
_DELTA_VALID_FIELDS: frozenset[str] = frozenset({"num_preemptions_total"})


def _assert_field_is_counter(field_name: str) -> None:
    """Raise TypeError if `field_name` is not a counter-typed field.

    The check uses `dataclasses.fields(PreemptionSnapshot)` to ensure
    the field exists at all (catches typos), then checks the
    `_DELTA_VALID_FIELDS` whitelist for delta-eligibility.
    """
    import dataclasses

    valid_field_names = {f.name for f in dataclasses.fields(PreemptionSnapshot)}
    if field_name not in valid_field_names:
        raise ValueError(
            f"'{field_name}' is not a field of PreemptionSnapshot. "
            f"Valid fields: {sorted(valid_field_names)}"
        )
    if field_name not in _DELTA_VALID_FIELDS:
        raise TypeError(
            f"Cannot compute delta of gauge/ratio field '{field_name}'. "
            f"Only counter fields support delta semantics; the valid set is "
            f"{sorted(_DELTA_VALID_FIELDS)}. Use mean/max aggregation for "
            f"non-counter fields instead — see "
            f".omc/plans/phase0-schemas.md section (b)."
        )


def compute_preemption_delta(actions: list[dict[str, Any]]) -> list[int]:
    """Compute consecutive deltas of `num_preemptions_total` across actions.

    Args:
        actions: A list of TraceAction dicts (as parsed from a v5
            trace's JSONL records). Each action is expected to have
            `data.sim_metrics.vllm_scheduler_snapshot.num_preemptions_total`
            populated. Actions that are missing the snapshot or have
            None for the counter are skipped silently (the delta list
            shrinks accordingly).

    Returns:
        A list of integer deltas. For an input of length N with valid
        counters at every position, returns a list of length N-1
        whose i-th element is `actions[i+1].counter - actions[i].counter`.

    Raises:
        TypeError: if any caller attempts to broaden this function to
            non-counter fields without updating `_DELTA_VALID_FIELDS`
            (see `_assert_field_is_counter`).
    """
    # Self-check: this function operates on the counter field, so
    # asserting it's in the whitelist guards against accidental copies
    # of this function being repurposed for gauges/ratios.
    _assert_field_is_counter("num_preemptions_total")

    counters: list[int] = []
    for action in actions:
        snap = (
            action.get("data", {})
            .get("sim_metrics", {})
            .get("vllm_scheduler_snapshot", {})
        )
        value = snap.get("num_preemptions_total")
        if value is None:
            continue
        counters.append(int(value))

    if len(counters) < 2:
        return []

    return [counters[i + 1] - counters[i] for i in range(len(counters) - 1)]


def compute_field_delta(
    actions: list[dict[str, Any]], field_name: str
) -> list[int]:
    """Generic delta computation gated by `_DELTA_VALID_FIELDS`.

    Use this when adding support for additional counter fields in the
    future. Raises `TypeError` for any field not in the whitelist.
    """
    _assert_field_is_counter(field_name)

    counters: list[int] = []
    for action in actions:
        snap = (
            action.get("data", {})
            .get("sim_metrics", {})
            .get("vllm_scheduler_snapshot", {})
        )
        value = snap.get(field_name)
        if value is None:
            continue
        counters.append(int(value))

    if len(counters) < 2:
        return []

    return [counters[i + 1] - counters[i] for i in range(len(counters) - 1)]
