"""Shared argparse helpers for latency evaluation scripts."""

from __future__ import annotations

import argparse
import os
from typing import Callable

from trace_collect.kv_profile_sweep import filter_profiles, load_profile
from trace_collect.tool_latency_bucket import bucket_edges_from_profile


def comma_separated_floats(
    label: str,
    *,
    require_nonnegative: bool = False,
) -> Callable[[str], list[float]]:
    """argparse type factory parsing comma-separated float lists."""

    def parse(value: str) -> list[float]:
        parsed: list[float] = []
        for raw in value.split(","):
            text = raw.strip()
            if not text:
                continue
            try:
                number = float(text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid {label} {text!r}") from exc
            if require_nonnegative and number < 0:
                raise argparse.ArgumentTypeError(
                    f"{label} values must be non-negative"
                )
            parsed.append(number)
        if not parsed:
            raise argparse.ArgumentTypeError(f"at least one {label} is required")
        return parsed

    return parse


def resolve_kv_costs(args: argparse.Namespace) -> list[float]:
    """KV costs from --kv-costs-ms or a filtered --kv-profile quantile."""

    if args.kv_costs_ms is not None:
        return args.kv_costs_ms
    profiles = filter_profiles(
        load_profile(args.kv_profile),
        engine=args.engine,
        direction=args.direction,
    )
    # With zero guard this is exactly the deduped profiled costs per entry.
    return bucket_edges_from_profile(profiles, quantile=args.quantile, guard_ms=0.0)


def resolve_worker_count(workers: int | None = None) -> int:
    """Return a positive worker count, leaving one CPU for the OS by default."""

    resolved = max(1, (os.cpu_count() or 1) - 1) if workers is None else workers
    if resolved < 1:
        raise ValueError(f"workers must be >= 1, got {resolved}")
    return resolved


__all__ = ["comma_separated_floats", "resolve_kv_costs", "resolve_worker_count"]
