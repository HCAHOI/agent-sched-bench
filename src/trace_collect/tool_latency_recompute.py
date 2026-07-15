"""Reload-vs-recompute restore-cost model for the free-memory action.

The utility clock's swap action frees a long-running call's KV during the
call and restores it when the call returns. There are two ways to restore:

* swap-in: copy the KV back from host memory. Bandwidth-bound, so its cost
  scales with the KV footprint, charged in the utility clock as
  ``restore_cost_fraction * kv_cost_ms`` (with the measured swap-in/swap-out
  ratio rho = 0.94 as the realistic fraction).
* recompute: re-run prefill over the call's resident context. Its cost
  scales with the number of context tokens resident when the call started.

Continuum and ThunderAgent both note this reload-vs-recompute choice has a
context-length crossover: short contexts favour recompute (cheap prefill),
long contexts favour swap (bounded by bandwidth). Because the swap cost is a
per-deployment constant in the campaign's swept ``kv_cost_ms`` grid while the
recompute cost grows with the call's own context, the crossover context is
``restore_cost_fraction * kv_cost_ms / recompute_rate_ms_per_token``: below
it recompute is cheaper, above it swap is cheaper. A restore-mechanism-aware
policy pays the cheaper of the two per call:

    effective_restore_ms = min(swap_restore_ms, recompute_restore_ms)

This module supplies only the cost primitives; the per-call context length is
recovered from real traces (see :mod:`trace_collect.tool_latency_context`) and
the choice is applied by re-scoring fitted decisions (see
:mod:`trace_collect.restore_cost_analysis`). ``recompute_rate_ms_per_token``
is a swept parameter here, exactly as rho was swept before it was measured;
its grid is a stand-in for a later measured H100 prefill-vs-context curve.

The rate model is deliberately linear in context length. Real prefill is
superlinear in sequence length (attention is ~O(c^2)); a linear rate
therefore under-charges recompute at long contexts, which biases the min()
toward recompute there. Context length uses the llm_call ``prompt_tokens``
(the KV resident when the call is dispatched) and ignores tokens generated
during the call, which under-charges recompute slightly further — same
(conservative-toward-recompute) direction. A measured prefill curve would
move the crossover to shorter contexts. The sweep brackets this uncertainty
rather than hiding it.
"""

from __future__ import annotations

import math

__all__ = [
    "recompute_restore_ms",
    "effective_min_restore_ms",
    "validate_recompute_rate",
]


def validate_recompute_rate(rate_ms_per_token: float) -> None:
    """Reject a non-finite or negative recompute rate."""

    if not math.isfinite(rate_ms_per_token) or rate_ms_per_token < 0.0:
        raise ValueError(
            "recompute_rate_ms_per_token must be finite and non-negative, got "
            f"{rate_ms_per_token}"
        )


def recompute_restore_ms(
    context_length_tokens: float,
    recompute_rate_ms_per_token: float,
) -> float:
    """Recompute (prefill) restore cost for a call with the given context.

    Linear in the number of resident context tokens; see the module docstring
    for the superlinearity caveat.
    """

    if not math.isfinite(context_length_tokens) or context_length_tokens < 0.0:
        raise ValueError(
            f"context_length_tokens must be finite and non-negative, got "
            f"{context_length_tokens}"
        )
    validate_recompute_rate(recompute_rate_ms_per_token)
    return recompute_rate_ms_per_token * context_length_tokens


def effective_min_restore_ms(
    swap_restore_ms: float,
    recompute_restore_ms_value: float,
) -> float:
    """Cheaper of the swap-in and recompute restore costs for one call."""

    for label, value in (
        ("swap_restore_ms", swap_restore_ms),
        ("recompute_restore_ms", recompute_restore_ms_value),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and non-negative, got {value}")
    return min(swap_restore_ms, recompute_restore_ms_value)
