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


@dataclass
class _BurstRunning:
    session_index: int
    command: AdmissionCommand
    start_s: float
    floor_end_s: float
    requested_cpu_cores: float
    requested_rss_mb: float
    max_cpu_cores: float
    remaining_cpu_work_core_s: float


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


def _weighted_cpu_allocations(
    running: list[_BurstRunning], cpu_capacity: float
) -> dict[int, float]:
    active = {
        index
        for index, item in enumerate(running)
        if item.remaining_cpu_work_core_s > _EPSILON
    }
    allocations = {index: 0.0 for index in range(len(running))}
    remaining_capacity = cpu_capacity
    while active:
        total_weight = sum(running[index].requested_cpu_cores for index in active)
        scale = remaining_capacity / total_weight
        capped = {
            index
            for index in active
            if running[index].max_cpu_cores
            <= running[index].requested_cpu_cores * scale + _EPSILON
        }
        if not capped:
            for index in active:
                allocations[index] = running[index].requested_cpu_cores * scale
            break
        for index in capped:
            allocation = running[index].max_cpu_cores
            allocations[index] = allocation
            remaining_capacity -= allocation
        active -= capped
    return allocations


def simulate_burstable_admission(
    programs: list[AdmissionProgram],
    *,
    cpu_capacity: float,
    rss_capacity_mb: float,
    requested_reservations: Mapping[str, tuple[float, float]],
    cpu_work_core_s: Mapping[str, float],
    max_cpu_cores: Mapping[str, float],
) -> dict[str, object]:
    """Replay admission requests while runnable work may borrow idle CPU."""

    if not programs or cpu_capacity <= 0.0 or rss_capacity_mb <= 0.0:
        raise ValueError("burstable replay requires programs and positive capacities")
    if any(not program.commands for program in programs):
        raise ValueError("every admission program must contain an exec command")
    commands = {
        command.command_id: command
        for program in programs
        for command in program.commands
    }
    if len(commands) != sum(len(program.commands) for program in programs):
        raise ValueError("command identities must be unique")
    if set(requested_reservations) != set(commands) or set(max_cpu_cores) != set(
        commands
    ):
        raise ValueError("burstable replay requires complete request and demand maps")
    if not set(cpu_work_core_s) <= set(commands):
        raise ValueError("CPU work contains an unknown command")
    if any(
        value < 0.0
        for program in programs
        for value in (program.initial_delay_s, program.tail_s)
    ) or any(
        command.duration_s <= 0.0
        or command.cpu_cores < 0.0
        or command.rss_mb < 0.0
        or command.delay_after_s < 0.0
        for command in commands.values()
    ):
        raise ValueError("burstable replay requires positive durations and non-negative inputs")
    for command_id in commands:
        requested_cpu, requested_rss = requested_reservations[command_id]
        demand = max_cpu_cores[command_id]
        work = cpu_work_core_s.get(command_id, 0.0)
        values = (requested_cpu, requested_rss, demand, work)
        if any(value < 0.0 for value in values):
            raise ValueError("burstable requests, demand, and work must be non-negative")
        if (
            requested_cpu <= 0.0
            or requested_cpu > demand + _EPSILON
            or demand > cpu_capacity + _EPSILON
            or requested_rss > rss_capacity_mb + _EPSILON
        ):
            raise ValueError(f"invalid burstable request or demand: {command_id}")

    sessions = [
        _Session(program, rank, ready_s=program.initial_delay_s)
        for rank, program in enumerate(programs)
    ]
    now_s = min(session.ready_s for session in sessions)
    running: list[_BurstRunning] = []
    used_cpu = used_rss = 0.0
    queue_s = service_s = reserved_cpu_s = reserved_rss_mb_s = 0.0
    served_cpu_work = 0.0
    max_concurrent = max_modeled_cpu = max_modeled_rss = 0.0
    exposure_events = starts = 0
    exposure_commands: set[str] = set()
    overlapped: set[str] = set()
    contended: set[str] = set()

    while running or any(
        session.command_index < len(session.program.commands) for session in sessions
    ):
        completed = sorted(
            (
                item
                for item in running
                if item.remaining_cpu_work_core_s <= _EPSILON
                and item.floor_end_s <= now_s + _EPSILON
            ),
            key=lambda item: (item.floor_end_s, item.session_index),
        )
        for item in completed:
            running.remove(item)
            used_cpu -= item.requested_cpu_cores
            used_rss -= item.requested_rss_mb
            duration_s = now_s - item.start_s
            service_s += duration_s
            reserved_cpu_s += item.requested_cpu_cores * duration_s
            reserved_rss_mb_s += item.requested_rss_mb * duration_s
            if duration_s > item.command.duration_s + _EPSILON:
                contended.add(item.command.command_id)
            session = sessions[item.session_index]
            session.running = False
            session.command_index += 1
            if session.command_index == len(session.program.commands):
                session.completion_s = now_s + session.program.tail_s
            else:
                session.ready_s = now_s + item.command.delay_after_s

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
            requested_cpu, requested_rss = requested_reservations[command.command_id]
            if (
                used_cpu + requested_cpu > cpu_capacity + _EPSILON
                or used_rss + requested_rss > rss_capacity_mb + _EPSILON
            ):
                continue
            if running:
                overlapped.add(command.command_id)
            queue_s += max(0.0, now_s - session.ready_s)
            work = cpu_work_core_s.get(command.command_id, 0.0)
            running.append(
                _BurstRunning(
                    index,
                    command,
                    now_s,
                    now_s + command.duration_s,
                    requested_cpu,
                    requested_rss,
                    max_cpu_cores[command.command_id],
                    work,
                )
            )
            used_cpu += requested_cpu
            used_rss += requested_rss
            modeled_cpu = sum(
                item.max_cpu_cores
                for item in running
                if item.remaining_cpu_work_core_s > _EPSILON
            )
            modeled_rss = sum(item.command.rss_mb for item in running)
            max_modeled_cpu = max(max_modeled_cpu, modeled_cpu)
            max_modeled_rss = max(max_modeled_rss, modeled_rss)
            if modeled_cpu > cpu_capacity + _EPSILON or modeled_rss > rss_capacity_mb + _EPSILON:
                exposure_events += 1
                exposure_commands.add(command.command_id)
            session.running = True
            starts += 1
            max_concurrent = max(max_concurrent, len(running))

        if not running and all(
            session.command_index == len(session.program.commands)
            for session in sessions
        ):
            break

        allocations = _weighted_cpu_allocations(running, cpu_capacity)
        future = [
            now_s + item.remaining_cpu_work_core_s / allocations[index]
            for index, item in enumerate(running)
            if item.remaining_cpu_work_core_s > _EPSILON
        ]
        future.extend(
            item.floor_end_s
            for item in running
            if item.remaining_cpu_work_core_s <= _EPSILON
            and item.floor_end_s > now_s + _EPSILON
        )
        future.extend(
            session.ready_s
            for session in sessions
            if not session.running
            and session.command_index < len(session.program.commands)
            and session.ready_s > now_s + _EPSILON
        )
        if not future:
            raise ValueError("ready command cannot fit the empty burstable pool")
        next_s = min(future)
        elapsed_s = next_s - now_s
        if elapsed_s <= 0.0:
            raise ValueError("burstable replay failed to advance time")
        for index, item in enumerate(running):
            served = allocations[index] * elapsed_s
            if served > item.remaining_cpu_work_core_s:
                served = item.remaining_cpu_work_core_s
            item.remaining_cpu_work_core_s -= served
            served_cpu_work += served
        now_s = next_s

    completion = [session.completion_s for session in sessions]
    if any(value is None for value in completion):
        raise ValueError("burstable replay ended before every task completed")
    total_cpu_work = sum(cpu_work_core_s.values())
    if (
        abs(used_cpu) > _EPSILON
        or abs(used_rss) > _EPSILON
        or abs(served_cpu_work - total_cpu_work) > max(1e-7, total_cpu_work * 1e-12)
    ):
        raise ValueError("burstable replay leaked reservation or CPU work")
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
        "capacity_violation": False,
        "modeled_capacity_exposure_events": exposure_events,
        "modeled_capacity_exposure_command_ids": sorted(exposure_commands),
        "max_modeled_cpu_demand_cores": max_modeled_cpu,
        "max_modeled_rss_demand_mb": max_modeled_rss,
        "total_cpu_work_core_s": total_cpu_work,
        "served_cpu_work_core_s": served_cpu_work,
        "contended_command_ids": sorted(contended),
        "added_service_s": service_s
        - sum(command.duration_s for command in commands.values()),
    }
