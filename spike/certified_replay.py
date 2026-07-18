"""Per-call accounting for the certified scenario (pure, CPU-testable).

Given a real replayed tool-call duration and the trigger the deploy table
resolves for its command group, decide whether the certified policy fires and
account the KV-resident-time it saves against two baselines computed in the same
pass: the deadline-only policy (trigger = deadline) and never-pause (never
fires). The utility functional is the SAME one the offline certification scores
with (``trace_collect.tool_latency_utility_clock.trigger_policy_utility_ms``,
``hidden_on_long - exposed - rho*restore``).

:func:`account_call` takes ``restore_cost_ms`` as an explicit argument (default
0.0, for unit tests exercising other behavior in isolation) -- it does NOT
default to the certified analysis's rho on its own. For the "same footing as
the certified analysis" claim to hold, the CALLER must pass
``restore_cost_ms = restore_cost_fraction * kv_cost_ms`` using the SAME
``restore_cost_fraction`` the decisions were fit/scored at (see
``spike/trigger_table.py``'s ``TriggerTable.metadata["restore_cost_fraction"]``
and ``spike/run_spike.py``'s ``run_certified``, which does this and refuses to
run at a 0.0 fraction -- scoring restore-free manufactures early-fire wins,
campaign finding F1).

These are DEMO / integration numbers: the utility deltas are accounting over
real replayed durations, they do not re-certify anything. Timing of the pause /
resume MECHANISM is measured live on the GPU; this module only does the
duration-driven accounting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from trace_collect.tool_latency_utility_clock import trigger_policy_utility_ms

from spike.trigger_table import TriggerLookup


@dataclass(frozen=True)
class CertifiedCallResult:
    """Accounting for one replayed call under the certified trigger."""

    sample_id: str
    task_id: str
    tool_name: str
    command: str
    duration_ms: float
    trigger_ms: float
    group_key: str | None
    backoff_level: int
    trigger_source: str
    fired: bool
    correct_fire: bool  # fired on a call long enough to hide KV (duration > deadline)
    misfire: bool  # fired on a call that ends at/under the deadline (pure swap cost)
    free_window_ms: float  # gross wall-time the blocks are freed if fired
    kv_saved_ms: float  # net utility under the certified trigger
    kv_saved_deadline_ms: float  # net utility under deadline-only
    kv_saved_never_ms: float  # net utility under never-pause (0.0 by definition)
    delta_vs_deadline_ms: float
    delta_vs_never_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def account_call(
    lookup: TriggerLookup,
    *,
    sample_id: str,
    task_id: str,
    tool_name: str,
    command: str,
    duration_ms: float,
    kv_cost_ms: float,
    deadline_ms: float,
    restore_cost_ms: float = 0.0,
) -> CertifiedCallResult:
    """Account one replayed call under the certified trigger vs both baselines."""

    if duration_ms < 0.0:
        raise ValueError(f"duration_ms must be >= 0, got {duration_ms}")
    trigger_ms = lookup.trigger_ms
    fired = duration_ms > trigger_ms
    correct_fire = fired and duration_ms > deadline_ms
    misfire = fired and duration_ms <= deadline_ms
    free_window_ms = max(0.0, duration_ms - trigger_ms) if fired else 0.0

    kv_saved_ms = trigger_policy_utility_ms(
        duration_ms,
        trigger_ms,
        threshold_ms=deadline_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    kv_saved_deadline_ms = trigger_policy_utility_ms(
        duration_ms,
        deadline_ms,
        threshold_ms=deadline_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    # Never-pause never fires, so its utility is zero by construction; the
    # deadline is a valid never-fire trigger only because guard>=0 makes it >=
    # every short call, but never-pause is defined independently as trigger=+inf.
    kv_saved_never_ms = 0.0
    return CertifiedCallResult(
        sample_id=sample_id,
        task_id=task_id,
        tool_name=tool_name,
        command=command,
        duration_ms=duration_ms,
        trigger_ms=trigger_ms,
        group_key=lookup.group_key,
        backoff_level=lookup.backoff_level,
        trigger_source=lookup.source,
        fired=fired,
        correct_fire=correct_fire,
        misfire=misfire,
        free_window_ms=free_window_ms,
        kv_saved_ms=kv_saved_ms,
        kv_saved_deadline_ms=kv_saved_deadline_ms,
        kv_saved_never_ms=kv_saved_never_ms,
        delta_vs_deadline_ms=kv_saved_ms - kv_saved_deadline_ms,
        delta_vs_never_ms=kv_saved_ms - kv_saved_never_ms,
    )


def aggregate_certified(results: Sequence[CertifiedCallResult]) -> dict[str, object]:
    """Aggregate per-call accounting into the run-level demo summary."""

    if not results:
        raise ValueError("no call results to aggregate")
    fired = [r for r in results if r.fired]
    return {
        "call_count": len(results),
        "fired_count": len(fired),
        "correct_fire_count": sum(r.correct_fire for r in results),
        "misfire_count": sum(r.misfire for r in results),
        "early_fire_count": sum(
            r.fired and r.trigger_source == "group" for r in results
        ),
        "total_free_window_ms": sum(r.free_window_ms for r in results),
        "kv_saved_ms": sum(r.kv_saved_ms for r in results),
        "kv_saved_deadline_ms": sum(r.kv_saved_deadline_ms for r in results),
        "kv_saved_never_ms": sum(r.kv_saved_never_ms for r in results),
        "delta_vs_deadline_ms": sum(r.delta_vs_deadline_ms for r in results),
        "delta_vs_never_ms": sum(r.delta_vs_never_ms for r in results),
        "group_hit_count": sum(r.trigger_source == "group" for r in results),
        "deadline_fallback_count": sum(
            r.trigger_source == "deadline" for r in results
        ),
    }


__all__ = ["CertifiedCallResult", "account_call", "aggregate_certified"]
