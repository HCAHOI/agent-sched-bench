from __future__ import annotations


def parse_prometheus_metric_values(
    metrics_payload: str,
    metric_names: dict[str, str],
    *,
    include_missing: bool,
) -> dict[str, float | None]:
    """Extract numeric metric values keyed by their caller-provided aliases."""
    values: dict[str, float | None] = {}
    if include_missing:
        values = {alias: None for alias in metric_names.values()}

    for line in metrics_payload.splitlines():
        if not line or line.startswith("#"):
            continue
        for metric_name, alias in metric_names.items():
            if line.startswith(metric_name):
                values[alias] = float(line.split()[-1])
                break
    return values
