"""CLI -> EvictionPolicyConfig adapter.

Step 3 only consumes argparse flags (`--kv-policy`, `--kv-budget`); YAML
resolution is step 8. Keeping the surface narrow now means step 8 can swap
this for a richer loader without churning callers — the contract is
"argparse Namespace in, optional EvictionPolicyConfig out".
"""

from __future__ import annotations

import argparse
from typing import Any

from serving.kv_policies.base import EvictionPolicyConfig


def load_eviction_config(args: Any) -> EvictionPolicyConfig | None:
    """Build an `EvictionPolicyConfig` from CLI args, or None if disabled.

    Returns None when `args.kv_policy == "none"` so downstream callers can
    pattern-match on `if eviction_config is not None`. Validates that a
    non-none policy carries a positive budget; emits an argparse error so the
    failure message lands in the same channel as other CLI mistakes.

    `sink_size` / `recent_window` are forwarded for the `streaming` policy;
    `random` ignores them. The dataclass validates the streaming-specific
    relationship (`budget >= sink_size + recent_window`) at cache construction
    time, not here, so that yaml-driven configs (step 8) hit the same gate.
    """
    policy = getattr(args, "kv_policy", "none")
    if policy == "none":
        return None
    budget = getattr(args, "kv_budget", None)
    if budget is None or int(budget) <= 0:
        raise argparse.ArgumentTypeError(
            f"--kv-policy {policy!r} requires --kv-budget > 0 (got {budget!r})"
        )
    sink_size = getattr(args, "kv_sink_size", None)
    recent_window = getattr(args, "kv_recent_window", None)
    aggregate = getattr(args, "kv_aggregate", None)
    kwargs: dict[str, Any] = {"name": policy, "budget": int(budget)}
    if sink_size is not None:
        kwargs["sink_size"] = int(sink_size)
    if recent_window is not None:
        kwargs["recent_window"] = int(recent_window)
    if aggregate is not None:
        # `aggregate` is h2o-specific; for streaming/random the dataclass
        # default ("sum") is irrelevant. Forwarding unconditionally keeps the
        # adapter narrow.
        kwargs["aggregate"] = str(aggregate)
    return EvictionPolicyConfig(**kwargs)
