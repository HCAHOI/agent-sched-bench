"""Optional EAR resource leases for Docker trace replay."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Literal

import ear
from ear.controller import AdmissionPolicy, ResourceController, ResourceLedger
from ear.core.config import load_policy_config
from ear.executor.docker import (
    DockerCgroupView,
    DockerResourceManager,
    resolve_docker_container_cgroup,
)
from ear.metrics.summary import build_lease_summary, write_lease_summary_json

EarMode = Literal["fixed", "elastic"]
_LEASE_EVENTS = "lease_events.jsonl"
_CONTROLLER_SUMMARY = "controller_summary.json"


@dataclass(frozen=True, slots=True)
class EarTaskReservation:
    """An admitted lease whose container has not been bound yet."""

    lease: Any
    cpuset: str
    cpu_cores: int
    memory_bytes: int
    oom_score_adj: int


@dataclass(slots=True)
class _ContainerLease:
    lease: Any
    cgroup: DockerCgroupView
    manager: DockerResourceManager | None
    monitor: Any | None


class EarReplayRuntime:
    """One shared EAR controller and ledger for a single-process replay run."""

    def __init__(
        self,
        *,
        policy_path: Path,
        mode: EarMode,
        concurrency: int,
        container_executable: str,
    ) -> None:
        if Path(container_executable).name != "docker":
            raise ValueError("EAR replay requires the Docker executable")
        self.policy_path = policy_path.resolve()
        self.policy = load_policy_config(self.policy_path)
        self.mode = mode
        self.concurrency = concurrency
        self.container_executable = container_executable
        controller_config = self.policy.controller
        if controller_config.total_cpus is None:
            raise ValueError("EAR policy controller.total_cpus must be explicit")
        if controller_config.total_memory_gb is None:
            raise ValueError("EAR policy controller.total_memory_gb must be explicit")
        self.total_cpus = controller_config.total_cpus
        self.total_memory_gb = controller_config.total_memory_gb
        self.quota_only_cpu = self.policy.docker.quota_only_cpu
        manager = self.policy.docker.resource_manager
        if mode == "elastic" and not manager.enabled:
            raise ValueError(
                "EAR elastic mode requires docker.resource_manager.enabled"
            )
        if mode == "fixed":
            self.initial_cpus = manager.reserved_cpu_cores
            self.initial_memory_gb = manager.reserved_memory_gb
        else:
            self.initial_cpus = manager.initial_reserved_cpu_cores
            self.initial_memory_gb = manager.initial_reserved_memory_gb
        if self.initial_cpus > self.total_cpus:
            raise ValueError("EAR initial CPU lease exceeds the shared pool")
        if self.initial_memory_gb > self.total_memory_gb:
            raise ValueError("EAR initial memory lease exceeds the shared pool")
        if mode == "elastic" and self.initial_cpus * concurrency > self.total_cpus:
            raise ValueError("EAR initial CPU leases exceed the shared pool")
        if (
            mode == "elastic"
            and self.initial_memory_gb * concurrency > self.total_memory_gb
        ):
            raise ValueError("EAR initial memory leases exceed the shared pool")

        self.ledger = ResourceLedger()
        self.controller = ResourceController(
            self.ledger,
            total_cpus=self.total_cpus,
            total_memory_gb=self.total_memory_gb,
            default_agent_quota=controller_config.default_agent_quota,
            admission_policy=AdmissionPolicy(controller_config.admission_policy),
        )
        self._clock_anchor = {
            "epoch_s": time.time(),
            "monotonic_s": time.monotonic(),
        }
        self._reservations: dict[str, EarTaskReservation] = {}
        self._handles: dict[str, _ContainerLease] = {}
        self._container_results: dict[str, dict[str, Any]] = {}
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._finalized = False

    def metadata(self) -> dict[str, Any]:
        policy_bytes = self.policy_path.read_bytes()
        return {
            "mode": self.mode,
            "policy": str(self.policy_path),
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "ear_version": _ear_version(),
            "ear_git_commit": _ear_git_commit(),
            "pool": {
                "cpu_cores": self.total_cpus,
                "memory_gb": self.total_memory_gb,
            },
            "initial_lease": {
                "cpu_cores": self.initial_cpus,
                "memory_gb": self.initial_memory_gb,
            },
            "clock_anchor": dict(self._clock_anchor),
            "artifacts": {
                "lease_events": _LEASE_EVENTS,
                "controller_summary": _CONTROLLER_SUMMARY,
            },
        }

    def reserve_task(self, agent_id: str) -> EarTaskReservation:
        try:
            lease = self.controller.acquire(
                self.initial_cpus,
                self.initial_memory_gb,
                agent_id,
                timeout_s=self.policy.docker.run_timeout_s,
            )
        except Exception as exc:
            self._record_error(f"{agent_id}: EAR lease acquisition failed: {exc}")
            raise
        reservation = EarTaskReservation(
            lease=lease,
            cpuset=(
                ",".join(str(cpu) for cpu in range(self.total_cpus))
                if self.quota_only_cpu
                else self.controller.cpuset_string(lease)
            ),
            cpu_cores=len(lease.cpus),
            memory_bytes=int(lease.memory_gb * 1024**3),
            oom_score_adj=self.policy.docker.oom_score_adj,
        )
        with self._lock:
            self._reservations[lease.lease_id] = reservation
        return reservation

    def cancel_reservation(self, reservation: EarTaskReservation) -> None:
        with self._lock:
            active = self._reservations.pop(reservation.lease.lease_id, None)
        if active is not None:
            self.controller.release(active.lease)

    def attach_task_container(
        self,
        reservation: EarTaskReservation,
        container_id: str,
    ) -> None:
        with self._lock:
            active = self._reservations.get(reservation.lease.lease_id)
        if active is not reservation:
            raise ValueError("EAR reservation is not active")

        lease = reservation.lease
        monitor: Any | None = None
        manager: DockerResourceManager | None = None
        try:
            lease = self.controller.bind_vm(lease, container_id)
            manager_config = self.policy.docker.resource_manager
            cgroup = DockerCgroupView(
                container_id=container_id,
                path=resolve_docker_container_cgroup(
                    container_id,
                    docker_executable=self.container_executable,
                    cgroup_root=Path(manager_config.cgroup_root),
                    timeout_s=self.policy.docker.run_timeout_s,
                ),
                docker_executable=self.container_executable,
                update_timeout_s=self.policy.docker.run_timeout_s,
            )
            if cgroup.read_cpu_limit() != reservation.cpu_cores:
                raise ValueError("Docker CPU limit does not match EAR reservation")
            if cgroup.read_memory_limit() != reservation.memory_bytes:
                raise ValueError("Docker memory limit does not match EAR reservation")
            manager = (
                DockerResourceManager(
                    config=manager_config,
                    cgroup=cgroup,
                    resource_controller=self.controller,
                    lease=lease,
                    quota_only_cpu=self.quota_only_cpu,
                )
                if self.mode == "elastic"
                else None
            )
            monitor = manager.begin_command() if manager is not None else None
            if monitor is not None:
                monitor.start()
        except Exception as exc:
            if monitor is not None:
                try:
                    monitor.finish()
                except OSError:
                    pass
            with self._lock:
                self._reservations.pop(reservation.lease.lease_id, None)
            self.controller.release(manager.lease if manager is not None else lease)
            self._record_error(
                f"{reservation.lease.agent_id}: EAR attach failed: {exc}"
            )
            raise

        with self._lock:
            self._reservations.pop(reservation.lease.lease_id, None)
            self._handles[container_id] = _ContainerLease(
                lease=lease,
                cgroup=cgroup,
                manager=manager,
                monitor=monitor,
            )

    def start_task_container(
        self,
        fixed_image: str,
        *,
        agent_id: str,
        start_fn: Callable[..., str],
        stop_fn: Callable[[str], Any],
        extra_args: list[str] | None = None,
        **start_kwargs: Any,
    ) -> str:
        reservation = self.reserve_task(agent_id)
        resource_args = [
            "--cpuset-cpus",
            reservation.cpuset,
            "--cpus",
            str(reservation.cpu_cores),
            "--memory",
            str(reservation.memory_bytes),
            "--memory-swap",
            str(reservation.memory_bytes),
            "--oom-score-adj",
            str(reservation.oom_score_adj),
        ]
        container_id: str | None = None
        try:
            container_id = start_fn(
                fixed_image,
                extra_args=[*(extra_args or []), *resource_args],
                **start_kwargs,
            )
            self.attach_task_container(reservation, container_id)
            return container_id
        except Exception as exc:
            cleanup_error: Exception | None = None
            if container_id is not None:
                try:
                    stop_fn(container_id)
                except Exception as stop_exc:  # noqa: BLE001
                    cleanup_error = stop_exc
                    self._record_error(
                        f"{agent_id}: container cleanup after attach failed: {stop_exc}"
                    )
            self.cancel_reservation(reservation)
            if cleanup_error is not None:
                raise RuntimeError(f"{exc}; cleanup failed: {cleanup_error}") from exc
            raise

    def stop_task_container(
        self,
        container_id: str,
        *,
        stop_fn: Callable[[str], Any],
    ) -> Any:
        with self._lock:
            handle = self._handles.get(container_id)
        if handle is None:
            self._record_error(f"{container_id}: missing EAR container lease")
            return stop_fn(container_id)

        dynamic: Any | None = None
        if handle.monitor is not None:
            try:
                dynamic = handle.monitor.finish()
            except OSError as exc:
                self._record_error(f"{container_id}: EAR monitor finish failed: {exc}")
            handle.monitor = None
        memory_events = self._read_memory_events(container_id, handle.cgroup.path)
        try:
            result = stop_fn(container_id)
        except Exception as exc:
            self._record_error(f"{container_id}: container stop failed: {exc}")
            raise

        lease = handle.manager.lease if handle.manager is not None else handle.lease
        self.controller.release(lease)
        container_result = {
            "agent_id": lease.agent_id,
            "lease_id": lease.lease_id,
            "memory_events": memory_events,
        }
        if dynamic is not None:
            dynamic_summary = _dynamic_summary(dynamic)
            container_result["dynamic"] = dynamic_summary
            for error in dynamic.errors:
                self._record_error(f"{container_id}: {error}")
        with self._lock:
            self._handles.pop(container_id, None)
            self._container_results[container_id] = container_result
        return result

    def write_artifacts(self, output_dir: Path) -> None:
        if self._finalized:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            active_container_ids = sorted(self._handles)
            active_reservation_ids = sorted(self._reservations)
        if active_container_ids:
            self._record_error(
                "unreleased EAR container leases: " + ", ".join(active_container_ids)
            )
        if active_reservation_ids:
            self._record_error(
                "unreleased EAR reservations: " + ", ".join(active_reservation_ids)
            )
        self.ledger.flush_to_jsonl(output_dir / _LEASE_EVENTS)
        try:
            summary = build_lease_summary(
                self.ledger.events(),
                [],
                total_memory_gb=self.total_memory_gb,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._record_error(f"controller summary failed: {exc}")
            summary = {"summary_error": str(exc)}
        with self._lock:
            errors = list(self._errors)
            container_results = dict(self._container_results)
        oom_kill_count = sum(
            int(result["memory_events"].get("oom_kill", 0))
            for result in container_results.values()
        )
        runtime_summary = {
            **self.metadata(),
            "status": (
                "valid"
                if not errors
                and not active_container_ids
                and not active_reservation_ids
                and oom_kill_count == 0
                else "invalid"
            ),
            "errors": errors,
            "active_container_ids": active_container_ids,
            "active_reservation_ids": active_reservation_ids,
            "oom_kill_count": oom_kill_count,
            "containers": container_results,
        }
        summary["ear_runtime"] = runtime_summary
        write_lease_summary_json(output_dir, summary)
        self._finalized = True

    @property
    def valid(self) -> bool:
        if not self._finalized:
            raise RuntimeError(
                "EAR runtime validity is unavailable before finalization"
            )
        with self._lock:
            return (
                not self._errors
                and not self._handles
                and not self._reservations
                and all(
                    int(result["memory_events"].get("oom_kill", 0)) == 0
                    for result in self._container_results.values()
                )
            )

    def _read_memory_events(
        self,
        container_id: str,
        cgroup_path: Path,
    ) -> dict[str, int]:
        try:
            events = {
                key: int(value)
                for line in (cgroup_path / "memory.events")
                .read_text(encoding="utf-8")
                .splitlines()
                if len(parts := line.split()) == 2
                for key, value in [parts]
            }
        except (OSError, ValueError) as exc:
            self._record_error(f"{container_id}: cannot read memory.events: {exc}")
            return {}
        return events

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)


def _dynamic_summary(dynamic: Any) -> dict[str, Any]:
    fields = (
        "wall_time_s",
        "active_core_seconds",
        "reserved_cpu_core_seconds",
        "active_memory_gb_seconds",
        "reserved_memory_gb_seconds",
        "cpu_grow_events",
        "cpu_shrink_events",
        "cpu_resize_rejected_events",
        "memory_grow_events",
        "memory_in_command_shrink_events",
        "memory_resize_rejected_events",
        "errors",
    )
    return {
        field: getattr(dynamic, field) for field in fields if hasattr(dynamic, field)
    }


def _ear_git_commit() -> str | None:
    try:
        direct_url = json.loads(
            distribution("elastic-agent-runtime").read_text("direct_url.json") or "{}"
        )
    except (PackageNotFoundError, json.JSONDecodeError):
        direct_url = {}
    commit_id = direct_url.get("vcs_info", {}).get("commit_id")
    if isinstance(commit_id, str) and commit_id:
        return commit_id

    root = Path(ear.__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit_id = result.stdout.strip()
    return commit_id if result.returncode == 0 and commit_id else None


def _ear_version() -> str | None:
    try:
        return version("elastic-agent-runtime")
    except PackageNotFoundError:
        return None
