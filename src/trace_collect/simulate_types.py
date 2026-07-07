from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.container_stats_sampler import ContainerResourceRecorder, ContainerStatsSampler


class SimulateError(Exception):
    """Raised when simulation encounters a fatal issue."""


@dataclass(frozen=True, slots=True)
class TraceManifestEntry:
    """One resolved trace entry from a simulate manifest."""

    index: int
    trace: Path
    task_source: Path
    docker_image: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayTaskStats:
    """Per-trace throughput accounting for a simulate run."""

    agent_id: str
    run_instance_id: str
    source_agent_id: str
    manifest_index: int
    label: str | None
    source_trace: str
    success: bool
    elapsed_s: float
    action_count: int
    llm_call_count: int
    tool_exec_count: int
    failed_action_count: int = 0


@dataclass(frozen=True, slots=True)
class LLMTimingConfig:
    """LLM duration model for cloud replay."""

    mode: str = "source_scaled"
    ttft_ms: float | None = None
    tpot_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SleepDrift:
    """Expected-vs-observed asyncio sleep timing for replay diagnostics."""

    phase: str
    expected_s: float
    actual_s: float

    @property
    def drift_s(self) -> float:
        return self.actual_s - self.expected_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "expected_s": round(self.expected_s, 6),
            "actual_s": round(self.actual_s, 6),
            "drift_s": round(self.drift_s, 6),
            "drift_ms": round(self.drift_s * 1000.0, 3),
        }


@dataclass(slots=True)
class LoadedTraceSession:
    """Resolved replay inputs for one source trace."""

    source_trace: Path
    task_source: Path
    task_instance_id: str
    source_action_agent_id: str
    run_instance_id: str
    manifest_index: int
    scaffold: str
    metadata: dict[str, Any] | None
    summary: dict[str, Any] | None
    task: dict[str, Any]
    actions: list[dict[str, Any]]
    iterations: dict[int, dict[str, Any]]
    docker_image_override: str | None = None
    label: str | None = None

    @property
    def agent_id(self) -> str:
        return self.run_instance_id


@dataclass(frozen=True, slots=True)
class WorkerTraceInput:
    """Picklable replay input for a subprocess worker."""

    source_trace: str
    task_source: str
    manifest_index: int
    docker_image_override: str | None
    label: str | None
    run_instance_id: str
    task_instance_id: str
    source_action_agent_id: str


@dataclass(frozen=True, slots=True)
class WorkerReplayResult:
    """One subprocess worker's replay outputs."""

    wave_index: int
    worker_index: int
    trace_file: str
    task_stats: list[ReplayTaskStats]
    task_output_dirs: dict[str, str]


@dataclass(slots=True)
class PreparedContainer:
    """Container prepared for trace replay."""

    container_id: str
    container_executable: str
    docker_image: str
    agent: Any | None  # ContainerAgent
    fixed_image: str | None = None
    python_runtime: str | None = None
    pythonpath: str | None = None
    workdir: str = "/testbed"
    cleanup_fixed_image: bool = True
    cleanup_callback: Callable[[], None] | None = None
    extra_agents: list[Any] = dataclasses.field(default_factory=list)


@dataclass(slots=True)
class PreparedTraceSession:
    """Container plus the loaded source-trace context."""

    loaded: LoadedTraceSession
    container: PreparedContainer | None = None
    container_resource_recorder: ContainerResourceRecorder | None = None
    sampler: ContainerStatsSampler | None = None
    task_output_dir: Path | None = None
    resources_written: bool = False
    resource_monitoring_enabled: bool = True
    memory_bandwidth_enabled: bool = True
    monitoring_policy: dict[str, object] | None = None
    runtime_artifact_root_map: dict[str, str] = dataclasses.field(default_factory=dict)
