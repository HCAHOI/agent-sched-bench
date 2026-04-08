"""Trace discovery for the dynamic Gantt viewer backend."""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from demo.gantt_viewer.backend.schema import TraceDescriptor


REPO_ROOT = Path(__file__).resolve().parents[3]
_CLAUDE_CODE_TYPES = frozenset(
    {"queue-operation", "assistant", "user", "tool_use", "tool_result", "message"}
)
_CLAUDE_CODE_PREAMBLE_TYPES = frozenset({"file-history-snapshot", "system"})


@dataclass(slots=True)
class DiscoveryGroup:
    name: str
    paths: list[str]


@dataclass(slots=True)
class DiscoveryConfig:
    config_path: Path
    repo_root: Path
    groups: list[DiscoveryGroup]


@dataclass(slots=True)
class DiscoveryState:
    config: DiscoveryConfig
    descriptors: list[TraceDescriptor] = field(default_factory=list)
    descriptors_by_id: dict[str, TraceDescriptor] = field(default_factory=dict)

    @classmethod
    def from_config_path(
        cls,
        config_path: Path,
        *,
        repo_root: Path | None = None,
    ) -> "DiscoveryState":
        config = load_discovery_config(config_path, repo_root=repo_root)
        state = cls(config=config)
        state.reload()
        return state

    def reload(self) -> None:
        self.descriptors = discover_traces(self.config)
        self.descriptors_by_id = {descriptor.id: descriptor for descriptor in self.descriptors}


def load_discovery_config(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> DiscoveryConfig:
    """Load the YAML discovery config."""
    resolved_path = config_path.resolve()
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError(f"{resolved_path} must define a non-empty 'groups' list")

    groups: list[DiscoveryGroup] = []
    for idx, item in enumerate(groups_raw):
        if not isinstance(item, dict):
            raise ValueError(f"group #{idx} must be a mapping, got {type(item).__name__}")
        name = item.get("name")
        paths = item.get("paths")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"group #{idx} has invalid name: {name!r}")
        if not isinstance(paths, list) or not paths or not all(
            isinstance(path, str) and path.strip() for path in paths
        ):
            raise ValueError(f"group {name!r} must define a non-empty string 'paths' list")
        groups.append(DiscoveryGroup(name=name, paths=paths))

    return DiscoveryConfig(
        config_path=resolved_path,
        repo_root=(repo_root or REPO_ROOT).resolve(),
        groups=groups,
    )


def discover_traces(config: DiscoveryConfig) -> list[TraceDescriptor]:
    """Walk configured globs, sniff formats, and build stable descriptors."""
    descriptors: list[TraceDescriptor] = []
    seen_ids: set[str] = set()
    for group in config.groups:
        for raw_pattern in group.paths:
            for path in _expand_pattern(config, raw_pattern):
                descriptor = _build_descriptor(group, path)
                if descriptor.id in seen_ids:
                    raise ValueError(f"duplicate trace id discovered: {descriptor.id}")
                seen_ids.add(descriptor.id)
                descriptors.append(descriptor)
    descriptors.sort(key=lambda descriptor: descriptor.id)
    return descriptors


def sniff_format(path: Path) -> Literal["v5", "claude-code"]:
    """Sniff the trace format from the first recognized JSONL record."""
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"trace file not found: {resolved_path}")

    observed_types: list[str] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL record in {resolved_path}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"record in {resolved_path} is not a JSON object")

                record_type = record.get("type")
                observed_types.append(str(record_type))
                if record_type == "trace_metadata":
                    return "v5"
                if record_type in _CLAUDE_CODE_TYPES:
                    return "claude-code"
                if record_type in _CLAUDE_CODE_PREAMBLE_TYPES:
                    continue
                observed_keys = ",".join(sorted(record.keys()))
                raise ValueError(
                    f"unable to sniff trace format for {resolved_path}; "
                    f"type={record_type!r}, keys=[{observed_keys}]"
                )
    if not observed_types:
        raise ValueError(f"empty JSONL file: {resolved_path}")
    raise ValueError(
        f"unable to sniff trace format for {resolved_path}; "
        f"observed types={observed_types}"
    )


def _expand_pattern(config: DiscoveryConfig, raw_pattern: str) -> list[Path]:
    pattern_path = Path(raw_pattern)
    pattern = raw_pattern if pattern_path.is_absolute() else str(config.repo_root / raw_pattern)
    return [Path(match).resolve() for match in sorted(glob.glob(pattern))]


def _build_descriptor(group: DiscoveryGroup, path: Path) -> TraceDescriptor:
    stat = path.stat()
    return TraceDescriptor(
        id=f"{_group_slug(group.name)}-{_trace_slug(path)}",
        label=_trace_label(path),
        source_format=sniff_format(path),
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
    )


def _group_slug(name: str) -> str:
    for separator in ("—", "–", ":", "-"):
        if separator in name:
            head = name.split(separator, 1)[0].strip()
            if head:
                return _slugify(head, keep_underscore=False)
    first_token = name.split()[0] if name.split() else name
    return _slugify(first_token, keep_underscore=False)


def _trace_label(path: Path) -> str:
    if path.parent.name.startswith("attempt_") and path.parent.parent.name:
        return path.parent.parent.name
    return path.parent.name


def _trace_slug(path: Path) -> str:
    return _slugify(_trace_label(path), keep_underscore=True)


def _slugify(value: str, *, keep_underscore: bool) -> str:
    pattern = r"[^a-z0-9_-]+" if keep_underscore else r"[^a-z0-9]+"
    slug = re.sub(pattern, "-", value.lower()).strip("-")
    return slug or "trace"
