"""Canonical client-visible tool-resource profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from tool_resource.runtime_kb import LatencyBuckets

_BEHAVIORS = {"predict", "observe_predict", "observe_predict_learn"}
_UPDATE_POLICIES = {"frozen", "causal"}
_TELEMETRY_REQUIREMENTS = {"best_effort", "required_for_valid_evidence"}
_PROFILE_KEYS = {
    "endpoint",
    "behavior",
    "update_policy",
    "snapshot",
    "telemetry_requirement",
    "latency_bucket_edges_ms",
}


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    endpoint: str
    behavior: str
    update_policy: str
    snapshot: str
    telemetry_requirement: str
    latency_bucket_edges_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "endpoint",
            "behavior",
            "update_policy",
            "snapshot",
            "telemetry_requirement",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"tool-resource {name} must be a string")
        self.socket_path()
        if self.behavior not in _BEHAVIORS:
            raise ValueError(f"unsupported tool-resource behavior {self.behavior!r}")
        if self.update_policy not in _UPDATE_POLICIES:
            raise ValueError(
                f"unsupported tool-resource update policy {self.update_policy!r}"
            )
        if not self.snapshot:
            raise ValueError("tool-resource snapshot is required")
        if self.telemetry_requirement not in _TELEMETRY_REQUIREMENTS:
            raise ValueError(
                "unsupported tool-resource telemetry requirement "
                f"{self.telemetry_requirement!r}"
            )
        LatencyBuckets(self.latency_bucket_edges_ms)

    @classmethod
    def load(cls, path: str | Path) -> ResourceProfile:
        profile_path = Path(path)
        try:
            raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid tool-resource profile {profile_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("tool-resource profile must be an object")
        if set(raw) != {"tool_resource"} or not isinstance(
            raw["tool_resource"],
            dict,
        ):
            raise ValueError(
                "tool-resource profile must contain exactly one tool_resource object"
            )
        config = raw["tool_resource"]
        unknown = set(config) - _PROFILE_KEYS
        missing = _PROFILE_KEYS - set(config)
        if unknown or missing:
            raise ValueError(
                "tool-resource profile fields differ from the canonical schema: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        edges = config["latency_bucket_edges_ms"]
        if not isinstance(edges, list):
            raise ValueError("latency_bucket_edges_ms must be a list")
        return cls(
            endpoint=config["endpoint"],
            behavior=config["behavior"],
            update_policy=config["update_policy"],
            snapshot=config["snapshot"],
            telemetry_requirement=config["telemetry_requirement"],
            latency_bucket_edges_ms=tuple(edges),
        )

    def socket_path(self) -> Path:
        parsed = urlparse(self.endpoint)
        if (
            parsed.scheme != "unix"
            or parsed.netloc
            or parsed.params
            or parsed.query
            or parsed.fragment
            or not parsed.path
        ):
            raise ValueError("tool-resource endpoint must be unix:///absolute/path")
        path = Path(unquote(parsed.path))
        if "\x00" in str(path) or not path.is_absolute():
            raise ValueError("tool-resource Unix socket path must be absolute")
        return path

    def open_run_payload(self, *, run_id: str, workspace_scope: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "workspace_scope": workspace_scope,
            "snapshot": self.snapshot,
            "latency_bucket_edges_ms": list(self.latency_bucket_edges_ms),
            "update_policy": self.update_policy,
            "telemetry_requirement": self.telemetry_requirement,
            "behavior": self.behavior,
        }


__all__ = ["ResourceProfile"]
