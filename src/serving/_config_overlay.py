"""Shared CLI/YAML config-overlay helper for serving subsystems.

`kv_policies` and `sparse_attention` use the same resolution pattern: YAML
supplies a base map, explicit (non-default) CLI flags overlay it, and the
`--<flag> none` argparse default does not clobber a yaml-supplied `name`.
This module factors out the YAML loader, the overlay merge, and the
coercion + kwargs build so both subsystems share one implementation.

Subsystem-specific validation (budget floors, method-specific required
fields, mutual-exclusivity, etc.) stays in each subsystem's `config.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

Coercer = Callable[[Any], Any]


def load_yaml_config(
    path: Path, *, flag_label: str, allowed_fields: set[str]
) -> dict[str, Any]:
    """Load a flat YAML mapping and reject unknown keys.

    Args:
        path: YAML file path.
        flag_label: CLI flag name used in error messages (e.g. ``--kv-config``).
        allowed_fields: dataclass field names accepted in the mapping.

    Returns:
        The mapping (possibly empty). Raises ``argparse.ArgumentTypeError``
        on a missing file, non-mapping content, or unknown keys.
    """
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{flag_label} path does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError(
            f"{flag_label} {path}: expected a YAML mapping, "
            f"got {type(raw).__name__}"
        )
    unknown = set(raw.keys()) - allowed_fields
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{flag_label} {path}: unknown keys {sorted(unknown)}; "
            f"allowed = {sorted(allowed_fields)}"
        )
    return dict(raw)


def coerce_value(
    field_name: str, value: Any, *, coercers: Mapping[str, Coercer]
) -> Any:
    """Apply a per-field coercer; pass through None or fields without one."""
    coercer = coercers.get(field_name)
    if coercer is None or value is None:
        return value
    return coercer(value)


def merge_cli_over_yaml(
    *,
    base: dict[str, Any],
    args: Any,
    cli_to_field: Mapping[str, str],
    cli_defaults: Mapping[str, Any],
    yaml_present: bool,
    name_flag: str,
) -> tuple[dict[str, Any], set[str]]:
    """Merge CLI flags over a YAML base map with default-preserving semantics.

    For each ``cli_attr -> field_name`` entry:

    - An explicit (non-default) CLI value overrides YAML and is tracked as
      explicit.
    - A flag left at its argparse default fills a gap only if the YAML did
      not supply that field and the default is not ``None``.
    - The ``name_flag`` (e.g. ``kv_policy`` / ``sparse_attn``) is special:
      ``--<flag> none`` is the implicit default and only sets ``name`` when
      no YAML file was supplied.

    Returns the merged map and the set of fields explicitly set on the CLI.
    The explicit set is used by callers that need to distinguish yaml- vs
    CLI-supplied values for downstream defaults (e.g. streaming's
    ``recent_window`` fallback).
    """
    merged: dict[str, Any] = dict(base)
    explicit_cli_fields: set[str] = set()
    for cli_attr, field_name in cli_to_field.items():
        cli_value = getattr(args, cli_attr, cli_defaults.get(cli_attr))
        default_value = cli_defaults.get(cli_attr)
        cli_explicit = cli_value != default_value
        if cli_attr == name_flag:
            if cli_explicit:
                merged[field_name] = cli_value
                explicit_cli_fields.add(field_name)
            elif not yaml_present and field_name not in merged:
                merged[field_name] = cli_value
        else:
            if cli_explicit:
                merged[field_name] = cli_value
                explicit_cli_fields.add(field_name)
            elif field_name not in merged and default_value is not None:
                merged[field_name] = default_value
    return merged, explicit_cli_fields


def build_kwargs(
    merged: Mapping[str, Any],
    *,
    allowed_fields: set[str],
    coercers: Mapping[str, Coercer],
) -> dict[str, Any]:
    """Project merged values into dataclass kwargs, coercing each field."""
    kwargs: dict[str, Any] = {}
    for field_name in allowed_fields:
        if field_name not in merged:
            continue
        kwargs[field_name] = coerce_value(
            field_name, merged[field_name], coercers=coercers
        )
    return kwargs
