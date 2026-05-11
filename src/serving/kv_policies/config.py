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
    """
    policy = getattr(args, "kv_policy", "none")
    if policy == "none":
        return None
    budget = getattr(args, "kv_budget", None)
    if budget is None or int(budget) <= 0:
        raise argparse.ArgumentTypeError(
            f"--kv-policy {policy!r} requires --kv-budget > 0 (got {budget!r})"
        )
    return EvictionPolicyConfig(name=policy, budget=int(budget))
