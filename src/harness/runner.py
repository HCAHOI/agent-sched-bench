from __future__ import annotations

import random


def build_arrival_offsets(
    num_tasks: int,
    *,
    arrival_mode: str,
    arrival_rate_per_s: float | None = None,
    arrival_seed: int | None = None,
) -> list[float]:
    if arrival_mode == "closed_loop":
        return [0.0] * num_tasks
    if arrival_mode != "poisson":
        raise ValueError(f"Unsupported arrival_mode: {arrival_mode}")
    if arrival_rate_per_s is None or arrival_rate_per_s <= 0:
        raise ValueError("arrival_rate_per_s must be positive for poisson mode")

    rng = random.Random(arrival_seed)
    offsets: list[float] = []
    elapsed = 0.0
    for _ in range(num_tasks):
        offsets.append(elapsed)
        elapsed += rng.expovariate(arrival_rate_per_s)
    return offsets
