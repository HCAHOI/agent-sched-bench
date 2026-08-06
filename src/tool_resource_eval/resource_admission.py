"""Deterministic command-level CPU/RSS admission replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import statistics


_EPSILON = 1e-9


@dataclass(frozen=True)
class AdmissionCommand:
    command_id: str
    duration_s: float
    cpu_cores: float
    rss_mb: float
    delay_after_s: float


@dataclass(frozen=True)
class AdmissionProgram:
    task_id: str
    initial_delay_s: float
    commands: tuple[AdmissionCommand, ...]
    tail_s: float


@dataclass
class _Session:
    program: AdmissionProgram
    seed_rank: int
    command_index: int = 0
    ready_s: float = 0.0
    running: bool = False
    completion_s: float | None = None


@dataclass(frozen=True)
class _Running:
    session_index: int
    command: AdmissionCommand
    end_s: float
    requested_cpu_cores: float
    requested_rss_mb: float
    modeled_cpu_cores: float
    modeled_rss_mb: float


def simulate_admission(
    programs: list[AdmissionProgram],
    *,
    cpu_capacity: float,
    rss_capacity_mb: float,
    fixed_high: bool,
    requested_reservations: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Replay FCFS-ready commands with work-conserving resource backfill."""

    if not programs or cpu_capacity <= 0.0 or rss_capacity_mb <= 0.0:
        raise ValueError("admission replay requires programs and positive capacities")
    if fixed_high and requested_reservations is not None:
        raise ValueError("fixed-high and explicit reservations are mutually exclusive")
    if any(not program.commands for program in programs):
        raise ValueError("every admission program must contain an exec command")
    sessions = [
        _Session(program, rank, ready_s=program.initial_delay_s)
        for rank, program in enumerate(programs)
    ]
    if any(
        value < 0.0
        for program in programs
        for value in (program.initial_delay_s, program.tail_s)
    ) or any(
        value < 0.0
        for program in programs
        for command in program.commands
        for value in (
            command.duration_s,
            command.cpu_cores,
            command.rss_mb,
            command.delay_after_s,
        )
    ):
        raise ValueError("admission program times and reservations must be non-negative")
    if any(
        command.duration_s <= 0.0
        for program in programs
        for command in program.commands
    ):
        raise ValueError("admission command duration must be positive")

    now_s = min(session.ready_s for session in sessions)
    running: list[_Running] = []
    used_cpu = used_rss = 0.0
    modeled_cpu = modeled_rss = 0.0
    queue_s = reserved_cpu_s = reserved_rss_mb_s = service_s = 0.0
    max_concurrent = 0
    overlapped: set[str] = set()
    capacity_violation = False
    exposure_events = 0
    exposure_commands: set[str] = set()
    max_modeled_cpu = max_modeled_rss = 0.0
    starts = 0

    while running or any(
        session.command_index < len(session.program.commands) for session in sessions
    ):
        completed = sorted(
            (item for item in running if item.end_s <= now_s + _EPSILON),
            key=lambda item: (item.end_s, item.session_index),
        )
        for item in completed:
            running.remove(item)
            used_cpu -= item.requested_cpu_cores
            used_rss -= item.requested_rss_mb
            modeled_cpu -= item.modeled_cpu_cores
            modeled_rss -= item.modeled_rss_mb
            session = sessions[item.session_index]
            session.running = False
            session.command_index += 1
            if session.command_index == len(session.program.commands):
                session.completion_s = item.end_s + session.program.tail_s
            else:
                session.ready_s = item.end_s + item.command.delay_after_s

        ready = sorted(
            (
                (index, session)
                for index, session in enumerate(sessions)
                if not session.running
                and session.command_index < len(session.program.commands)
                and session.ready_s <= now_s + _EPSILON
            ),
            key=lambda item: (
                item[1].ready_s,
                item[1].seed_rank,
                item[1].program.task_id,
            ),
        )
        for index, session in ready:
            command = session.program.commands[session.command_index]
            if fixed_high:
                cpu, rss = cpu_capacity, rss_capacity_mb
            elif requested_reservations is not None:
                try:
                    cpu, rss = requested_reservations[command.command_id]
                except KeyError as error:
                    raise ValueError(
                        f"command lacks an explicit reservation: {command.command_id}"
                    ) from error
            else:
                cpu, rss = command.cpu_cores, command.rss_mb
            if cpu > cpu_capacity + _EPSILON or rss > rss_capacity_mb + _EPSILON:
                raise ValueError(f"command reservation exceeds capacity: {command.command_id}")
            if used_cpu + cpu > cpu_capacity + _EPSILON or used_rss + rss > rss_capacity_mb + _EPSILON:
                continue
            if running:
                overlapped.add(command.command_id)
            queue_s += max(0.0, now_s - session.ready_s)
            service_s += command.duration_s
            reserved_cpu_s += cpu * command.duration_s
            reserved_rss_mb_s += rss * command.duration_s
            running.append(
                _Running(
                    index,
                    command,
                    now_s + command.duration_s,
                    cpu,
                    rss,
                    command.cpu_cores,
                    command.rss_mb,
                )
            )
            used_cpu += cpu
            used_rss += rss
            modeled_cpu += command.cpu_cores
            modeled_rss += command.rss_mb
            max_modeled_cpu = max(max_modeled_cpu, modeled_cpu)
            max_modeled_rss = max(max_modeled_rss, modeled_rss)
            if modeled_cpu > cpu_capacity + _EPSILON or modeled_rss > rss_capacity_mb + _EPSILON:
                exposure_events += 1
                exposure_commands.add(command.command_id)
            capacity_violation |= (
                used_cpu > cpu_capacity + _EPSILON
                or used_rss > rss_capacity_mb + _EPSILON
            )
            session.running = True
            starts += 1
            max_concurrent = max(max_concurrent, len(running))

        if not running and all(
            session.command_index == len(session.program.commands)
            for session in sessions
        ):
            break
        future = [item.end_s for item in running]
        future.extend(
            session.ready_s
            for session in sessions
            if not session.running
            and session.command_index < len(session.program.commands)
            and session.ready_s > now_s + _EPSILON
        )
        if not future:
            raise ValueError("ready command cannot fit the empty admission pool")
        now_s = min(future)

    completion = [session.completion_s for session in sessions]
    if any(value is None for value in completion):
        raise ValueError("admission replay ended before every task completed")
    if (
        abs(used_cpu) > _EPSILON
        or abs(used_rss) > _EPSILON
        or abs(modeled_cpu) > _EPSILON
        or abs(modeled_rss) > _EPSILON
    ):
        raise ValueError("admission replay leaked a reservation")
    completion_s = [float(value) for value in completion if value is not None]
    return {
        "command_count": starts,
        "total_command_service_s": service_s,
        "makespan_s": max(completion_s),
        "mean_task_completion_s": statistics.fmean(completion_s),
        "total_command_queue_s": queue_s,
        "reserved_cpu_core_s": reserved_cpu_s,
        "reserved_rss_mb_s": reserved_rss_mb_s,
        "max_concurrent_commands": max_concurrent,
        "overlapped_command_ids": sorted(overlapped),
        "capacity_violation": capacity_violation,
        "modeled_capacity_exposure_events": exposure_events,
        "modeled_capacity_exposure_command_ids": sorted(exposure_commands),
        "max_modeled_cpu_demand_cores": max_modeled_cpu,
        "max_modeled_rss_demand_mb": max_modeled_rss,
    }
