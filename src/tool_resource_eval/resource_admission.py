"""Deterministic command-level CPU/RSS admission replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import statistics


_EPSILON = 1e-9
_MAX_CPU_SHARE_RATIO = 1e12


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
    start_index: int
    command: AdmissionCommand
    start_s: float
    floor_end_s: float
    requested_cpu_cores: float
    requested_rss_mb: float
    cpu_share_weight: float
    max_cpu_cores: float
    remaining_cpu_work_core_s: float


@dataclass
class _FeedbackRunning:
    session_index: int
    command: AdmissionCommand
    start_s: float
    requested_cpu_cores: float
    requested_rss_mb: float
    profile: tuple[tuple[float, float], ...] | None
    profile_index: int
    profile_remaining_s: float
    finish_s: float
    next_observation_s: float
    pending_page: float | None = None
    pending_at_s: float = math.inf
    observed_cpu_core_s: float = 0.0
    observed_throttling: bool = False


@dataclass
class _IdleRunning:
    session_index: int
    command: AdmissionCommand
    start_s: float
    profile: tuple[tuple[float, float], ...]
    requested_rss_mb: float
    profile_index: int = 0
    profile_remaining_s: float = 0.0
    speculative: bool = False


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


def simulate_idle_backfill(
    programs: list[AdmissionProgram],
    *,
    cpu_capacity: float,
    rss_capacity_mb: float,
    cpu_work_profiles: Mapping[str, tuple[tuple[float, float], ...]],
    speculative_eligible_command_ids: set[str],
    rss_reservations: Mapping[str, float] | None = None,
    selection: str,
) -> dict[str, object]:
    """Replay one normal command plus one strict-idle-priority backfill."""

    if not programs or cpu_capacity <= 0.0 or rss_capacity_mb <= 0.0:
        raise ValueError("idle backfill requires programs and positive capacities")
    if selection not in {"serial", "fcfs", "shortest"}:
        raise ValueError("idle backfill selection must be serial, fcfs, or shortest")
    if any(not program.commands for program in programs):
        raise ValueError("every idle-backfill program must contain an exec command")
    commands = {
        command.command_id: command
        for program in programs
        for command in program.commands
    }
    if len(commands) != sum(len(program.commands) for program in programs):
        raise ValueError("idle-backfill command identities must be unique")
    if set(cpu_work_profiles) != set(commands):
        raise ValueError("idle backfill requires complete CPU work profiles")
    if not speculative_eligible_command_ids <= set(commands):
        raise ValueError("idle backfill eligibility contains an unknown command")
    reservation_by_id = (
        {command_id: command.rss_mb for command_id, command in commands.items()}
        if rss_reservations is None
        else dict(rss_reservations)
    )
    if set(reservation_by_id) != set(commands) or any(
        rss < 0.0
        or rss > rss_capacity_mb + _EPSILON
        or not math.isfinite(rss)
        for rss in reservation_by_id.values()
    ):
        raise ValueError("idle backfill requires complete in-capacity RSS reservations")
    if any(
        value < 0.0 or not math.isfinite(value)
        for program in programs
        for value in (program.initial_delay_s, program.tail_s)
    ) or any(
        command.duration_s <= 0.0
        or command.cpu_cores < 0.0
        or command.rss_mb < 0.0
        or command.rss_mb > rss_capacity_mb + _EPSILON
        or command.delay_after_s < 0.0
        or not all(
            math.isfinite(value)
            for value in (
                command.duration_s,
                command.cpu_cores,
                command.rss_mb,
                command.delay_after_s,
            )
        )
        for command in commands.values()
    ):
        raise ValueError("idle backfill requires finite in-capacity inputs")
    for command_id, command in commands.items():
        profile = cpu_work_profiles[command_id]
        if not profile or any(
            dt <= 0.0
            or cpu < 0.0
            or not math.isfinite(dt)
            or not math.isfinite(cpu)
            or cpu > cpu_capacity * dt + _EPSILON
            for dt, cpu in profile
        ):
            raise ValueError(f"invalid CPU work profile: {command_id}")
        if not math.isclose(
            sum(dt for dt, _cpu in profile),
            command.duration_s,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError(f"CPU work profile duration differs: {command_id}")

    sessions = [
        _Session(program, rank, ready_s=program.initial_delay_s)
        for rank, program in enumerate(programs)
    ]
    now_s = min(session.ready_s for session in sessions)
    running: list[_IdleRunning] = []
    used_rss = modeled_rss = queue_s = reserved_rss_mb_s = served_cpu_work = 0.0
    speculative_cpu_work = 0.0
    normal_starts = speculative_starts = speculative_completions = promotions = 0
    max_concurrent = 0
    capacity_violation = physical_capacity_violation = False
    exposure_events = 0
    exposure_commands: set[str] = set()
    max_modeled_rss = 0.0
    service_by_command: dict[str, float] = {}
    start_by_command: dict[str, float] = {}
    speculative_start_ids: list[str] = []

    def session_key(index: int) -> tuple[float, int, str]:
        session = sessions[index]
        return session.ready_s, session.seed_rank, session.program.task_id

    def ready_sessions() -> list[tuple[int, _Session]]:
        return sorted(
            (
                (index, session)
                for index, session in enumerate(sessions)
                if not session.running
                and session.command_index < len(session.program.commands)
                and session.ready_s <= now_s + _EPSILON
            ),
            key=lambda item: session_key(item[0]),
        )

    def start(index: int, *, speculative: bool) -> None:
        nonlocal used_rss, modeled_rss, queue_s, normal_starts
        nonlocal speculative_starts, exposure_events, max_modeled_rss
        session = sessions[index]
        command = session.program.commands[session.command_index]
        profile = cpu_work_profiles[command.command_id]
        running.append(
            _IdleRunning(
                session_index=index,
                command=command,
                start_s=now_s,
                profile=profile,
                requested_rss_mb=reservation_by_id[command.command_id],
                profile_remaining_s=profile[0][0],
                speculative=speculative,
            )
        )
        session.running = True
        used_rss += reservation_by_id[command.command_id]
        modeled_rss += command.rss_mb
        max_modeled_rss = max(max_modeled_rss, modeled_rss)
        if modeled_rss > rss_capacity_mb + _EPSILON:
            exposure_events += 1
            exposure_commands.add(command.command_id)
        queue_s += max(0.0, now_s - session.ready_s)
        start_by_command[command.command_id] = now_s
        if speculative:
            speculative_starts += 1
            speculative_start_ids.append(command.command_id)
        else:
            normal_starts += 1

    while running or any(
        session.command_index < len(session.program.commands) for session in sessions
    ):
        completed = sorted(
            (item for item in running if item.profile_index >= len(item.profile)),
            key=lambda item: session_key(item.session_index),
        )
        for item in completed:
            running.remove(item)
            used_rss -= item.requested_rss_mb
            modeled_rss -= item.command.rss_mb
            service_by_command[item.command.command_id] = now_s - item.start_s
            speculative_completions += int(item.speculative)
            session = sessions[item.session_index]
            session.running = False
            session.command_index += 1
            if session.command_index == len(session.program.commands):
                session.completion_s = now_s + session.program.tail_s
            else:
                session.ready_s = now_s + item.command.delay_after_s

        normal = next((item for item in running if not item.speculative), None)
        speculative = next((item for item in running if item.speculative), None)
        if normal is None and speculative is not None:
            speculative.speculative = False
            promotions += 1
            normal = speculative
            speculative = None

        ready = ready_sessions()
        if normal is None and ready:
            index, _session = ready[0]
            start(index, speculative=False)
            normal = running[-1]
            ready = ready_sessions()

        if selection != "serial" and normal is not None and speculative is None:
            fitting = [
                item
                for item in ready
                if item[1].program.commands[item[1].command_index].command_id
                in speculative_eligible_command_ids
                and normal.requested_rss_mb
                + reservation_by_id[
                    item[1].program.commands[item[1].command_index].command_id
                ]
                <= rss_capacity_mb + _EPSILON
            ]
            if fitting:
                if selection == "shortest":
                    fitting.sort(
                        key=lambda item: (
                            item[1].program.commands[item[1].command_index].duration_s,
                            *session_key(item[0]),
                        )
                    )
                start(fitting[0][0], speculative=True)

        max_concurrent = max(max_concurrent, len(running))
        capacity_violation |= used_rss > rss_capacity_mb + _EPSILON
        if not running and all(
            session.command_index == len(session.program.commands)
            for session in sessions
        ):
            break

        normal = next((item for item in running if not item.speculative), None)
        speculative = next((item for item in running if item.speculative), None)
        if normal is None:
            future_ready = [
                session.ready_s
                for session in sessions
                if not session.running
                and session.command_index < len(session.program.commands)
                and session.ready_s > now_s + _EPSILON
            ]
            if not future_ready:
                raise ValueError("idle backfill ended without a runnable command")
            now_s = min(future_ready)
            continue

        normal_dt, normal_cpu = normal.profile[normal.profile_index]
        normal_demand = normal_cpu / normal_dt
        normal_allocation = min(normal_demand, cpu_capacity)
        speculative_demand = speculative_allocation = 0.0
        if speculative is not None:
            speculative_dt, speculative_cpu = speculative.profile[
                speculative.profile_index
            ]
            speculative_demand = speculative_cpu / speculative_dt
            speculative_allocation = min(
                speculative_demand, cpu_capacity - normal_allocation
            )
        physical_capacity_violation |= (
            normal_allocation + speculative_allocation
            > cpu_capacity + _EPSILON
        )

        future: list[float] = []
        for item, demand, allocation in (
            (normal, normal_demand, normal_allocation),
            (speculative, speculative_demand, speculative_allocation),
        ):
            if item is None:
                continue
            progress_rate = 1.0 if demand <= allocation else allocation / demand
            if progress_rate > _EPSILON:
                future.append(now_s + item.profile_remaining_s / progress_rate)
        future.extend(
            session.ready_s
            for session in sessions
            if not session.running
            and session.command_index < len(session.program.commands)
            and session.ready_s > now_s + _EPSILON
        )
        if not future:
            raise ValueError("idle backfill made no progress")
        next_s = min(value for value in future if value > now_s + _EPSILON)
        elapsed_s = next_s - now_s
        reserved_rss_mb_s += sum(
            item.requested_rss_mb * elapsed_s for item in running
        )
        for item, demand, allocation in (
            (normal, normal_demand, normal_allocation),
            (speculative, speculative_demand, speculative_allocation),
        ):
            if item is None:
                continue
            progress_rate = 1.0 if demand <= allocation else allocation / demand
            item.profile_remaining_s -= elapsed_s * progress_rate
            work = allocation * elapsed_s
            served_cpu_work += work
            if item.speculative:
                speculative_cpu_work += work
            if item.profile_remaining_s <= _EPSILON:
                item.profile_index += 1
                if item.profile_index < len(item.profile):
                    item.profile_remaining_s = item.profile[item.profile_index][0]
        now_s = next_s

    completion = [session.completion_s for session in sessions]
    if any(value is None for value in completion):
        raise ValueError("idle backfill ended before every task completed")
    if (
        abs(used_rss) > _EPSILON
        or abs(modeled_rss) > _EPSILON
        or set(service_by_command) != set(commands)
    ):
        raise ValueError("idle backfill leaked command state")
    total_cpu_work = sum(
        cpu for profile in cpu_work_profiles.values() for _dt, cpu in profile
    )
    if not math.isclose(
        served_cpu_work, total_cpu_work, rel_tol=1e-12, abs_tol=1e-7
    ):
        raise ValueError("idle backfill failed to conserve CPU work")
    completion_s = [float(value) for value in completion if value is not None]
    recorded_service_s = sum(command.duration_s for command in commands.values())
    service_s = sum(service_by_command.values())
    return {
        "command_count": len(commands),
        "recorded_command_service_s": recorded_service_s,
        "total_command_service_s": service_s,
        "added_service_s": service_s - recorded_service_s,
        "makespan_s": max(completion_s),
        "mean_task_completion_s": statistics.fmean(completion_s),
        "total_command_queue_s": queue_s,
        "reserved_rss_mb_s": reserved_rss_mb_s,
        "max_concurrent_commands": max_concurrent,
        "normal_starts": normal_starts,
        "speculative_starts": speculative_starts,
        "speculative_completions": speculative_completions,
        "promotions": promotions,
        "speculative_cpu_work_core_s": speculative_cpu_work,
        "total_cpu_work_core_s": total_cpu_work,
        "served_cpu_work_core_s": served_cpu_work,
        "capacity_violation": capacity_violation,
        "physical_capacity_violation": physical_capacity_violation,
        "modeled_capacity_exposure_events": exposure_events,
        "modeled_capacity_exposure_command_ids": sorted(exposure_commands),
        "max_modeled_rss_demand_mb": max_modeled_rss,
        "speculative_start_ids": speculative_start_ids,
        "service_s_by_command": dict(sorted(service_by_command.items())),
        "start_s_by_command": dict(sorted(start_by_command.items())),
    }


def simulate_feedback_admission(
    programs: list[AdmissionProgram],
    *,
    cpu_capacity: float,
    rss_capacity_mb: float,
    requested_reservations: Mapping[str, tuple[float, float]],
    cpu_work_profiles: Mapping[str, tuple[tuple[float, float], ...]],
    feedback: bool,
    sample_interval_s: float,
    update_delay_s: float,
    cpu_pages: tuple[float, ...],
    work_conserving_cpu: bool = False,
) -> dict[str, object]:
    """Replay CPU admission with causal per-command reservation feedback."""

    if (
        not programs
        or cpu_capacity <= 0.0
        or rss_capacity_mb <= 0.0
        or sample_interval_s <= 0.0
        or update_delay_s < 0.0
    ):
        raise ValueError("feedback admission requires positive capacities and timing")
    if (
        not cpu_pages
        or tuple(sorted(cpu_pages)) != cpu_pages
        or len(set(cpu_pages)) != len(cpu_pages)
        or cpu_pages[-1] > cpu_capacity + _EPSILON
        or any(page <= 0.0 or not math.isfinite(page) for page in cpu_pages)
    ):
        raise ValueError("feedback admission requires ordered positive CPU pages")
    if any(not program.commands for program in programs):
        raise ValueError("every admission program must contain an exec command")
    commands = {
        command.command_id: command
        for program in programs
        for command in program.commands
    }
    if len(commands) != sum(len(program.commands) for program in programs):
        raise ValueError("command identities must be unique")
    if set(requested_reservations) != set(commands):
        raise ValueError("feedback admission requires complete reservations")
    if not set(cpu_work_profiles) <= set(commands):
        raise ValueError("CPU work profile contains an unknown command")
    if work_conserving_cpu and set(cpu_work_profiles) != set(commands):
        raise ValueError("work-conserving CPU requires complete CPU work profiles")
    if any(
        value < 0.0 or not math.isfinite(value)
        for program in programs
        for value in (program.initial_delay_s, program.tail_s)
    ) or any(
        command.duration_s <= 0.0
        or command.cpu_cores < 0.0
        or command.rss_mb < 0.0
        or command.delay_after_s < 0.0
        or not all(
            math.isfinite(value)
            for value in (
                command.duration_s,
                command.cpu_cores,
                command.rss_mb,
                command.delay_after_s,
            )
        )
        for command in commands.values()
    ):
        raise ValueError("feedback admission requires finite non-negative inputs")
    for command_id, command in commands.items():
        requested_cpu, requested_rss = requested_reservations[command_id]
        if (
            requested_cpu not in cpu_pages
            or requested_rss < 0.0
            or requested_rss > rss_capacity_mb + _EPSILON
            or not math.isfinite(requested_rss)
        ):
            raise ValueError(f"invalid feedback reservation: {command_id}")
        profile = cpu_work_profiles.get(command_id)
        if profile is None:
            continue
        if not profile or any(
            dt <= 0.0
            or cpu < 0.0
            or not math.isfinite(dt)
            or not math.isfinite(cpu)
            or cpu > cpu_capacity * dt + _EPSILON
            for dt, cpu in profile
        ):
            raise ValueError(f"invalid CPU work profile: {command_id}")
        profile_duration = sum(dt for dt, _cpu in profile)
        if not math.isclose(
            profile_duration, command.duration_s, rel_tol=1e-9, abs_tol=1e-7
        ):
            raise ValueError(f"CPU work profile duration differs: {command_id}")

    sessions = [
        _Session(program, rank, ready_s=program.initial_delay_s)
        for rank, program in enumerate(programs)
    ]
    now_s = min(session.ready_s for session in sessions)
    running: list[_FeedbackRunning] = []
    used_cpu = used_rss = 0.0
    queue_s = reserved_cpu_s = reserved_rss_mb_s = 0.0
    served_cpu_work = 0.0
    feedback_updates = shrinks = expansions = denied_expansions = 0
    starts = max_concurrent = 0
    capacity_violation = physical_capacity_violation = False
    overlapped: set[str] = set()
    service_by_command: dict[str, float] = {}
    start_by_command: dict[str, float] = {}

    def running_key(item: _FeedbackRunning) -> tuple[float, int, str]:
        session = sessions[item.session_index]
        return session.ready_s, session.seed_rank, session.program.task_id

    while running or any(
        session.command_index < len(session.program.commands) for session in sessions
    ):
        completed = sorted(
            (
                item
                for item in running
                if (
                    item.profile is None
                    and item.finish_s <= now_s + _EPSILON
                )
                or (
                    item.profile is not None
                    and item.profile_index >= len(item.profile)
                )
            ),
            key=running_key,
        )
        for item in completed:
            running.remove(item)
            used_cpu -= item.requested_cpu_cores
            used_rss -= item.requested_rss_mb
            service_by_command[item.command.command_id] = now_s - item.start_s
            session = sessions[item.session_index]
            session.running = False
            session.command_index += 1
            if session.command_index == len(session.program.commands):
                session.completion_s = now_s + session.program.tail_s
            else:
                session.ready_s = now_s + item.command.delay_after_s

        due_updates = sorted(
            (
                item
                for item in running
                if item.pending_page is not None
                and item.pending_at_s <= now_s + _EPSILON
            ),
            key=running_key,
        )
        shrinking = [
            item
            for item in due_updates
            if float(item.pending_page) < item.requested_cpu_cores - _EPSILON
        ]
        expanding = [
            item
            for item in due_updates
            if float(item.pending_page) > item.requested_cpu_cores + _EPSILON
        ]
        unchanged = [
            item
            for item in due_updates
            if item not in shrinking and item not in expanding
        ]
        for item in shrinking:
            target = float(item.pending_page)
            used_cpu -= item.requested_cpu_cores - target
            item.requested_cpu_cores = target
            item.pending_page = None
            item.pending_at_s = math.inf
            shrinks += 1
        for item in expanding:
            target = float(item.pending_page)
            added = target - item.requested_cpu_cores
            if used_cpu + added <= cpu_capacity + _EPSILON:
                used_cpu += added
                item.requested_cpu_cores = target
                expansions += 1
            else:
                denied_expansions += 1
            item.pending_page = None
            item.pending_at_s = math.inf
        for item in unchanged:
            item.pending_page = None
            item.pending_at_s = math.inf

        for item in sorted(running, key=running_key):
            if item.next_observation_s > now_s + _EPSILON:
                continue
            target = (
                cpu_pages[-1]
                if item.observed_throttling
                else next(
                    page
                    for page in cpu_pages
                    if page
                    >= item.observed_cpu_core_s / sample_interval_s - _EPSILON
                )
            )
            item.pending_page = target
            item.pending_at_s = now_s + update_delay_s
            item.next_observation_s += sample_interval_s
            item.observed_cpu_core_s = 0.0
            item.observed_throttling = False
            feedback_updates += 1

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
            profile = cpu_work_profiles.get(command.command_id)
            if running:
                overlapped.add(command.command_id)
            queue_s += max(0.0, now_s - session.ready_s)
            start_by_command[command.command_id] = now_s
            running.append(
                _FeedbackRunning(
                    session_index=index,
                    command=command,
                    start_s=now_s,
                    requested_cpu_cores=requested_cpu,
                    requested_rss_mb=requested_rss,
                    profile=profile,
                    profile_index=0,
                    profile_remaining_s=(
                        profile[0][0] if profile is not None else 0.0
                    ),
                    finish_s=now_s + command.duration_s,
                    next_observation_s=(
                        now_s + sample_interval_s
                        if feedback and profile is not None
                        else math.inf
                    ),
                )
            )
            used_cpu += requested_cpu
            used_rss += requested_rss
            session.running = True
            starts += 1
            max_concurrent = max(max_concurrent, len(running))
            capacity_violation |= (
                used_cpu > cpu_capacity + _EPSILON
                or used_rss > rss_capacity_mb + _EPSILON
            )

        if not running and all(
            session.command_index == len(session.program.commands)
            for session in sessions
        ):
            break

        demands = []
        for item in running:
            if item.profile is not None:
                dt, cpu = item.profile[item.profile_index]
                demands.append(cpu / dt)
            else:
                demands.append(0.0)
        if work_conserving_cpu:
            allocations = {index: 0.0 for index in range(len(running))}
            active = {
                index
                for index, demand in enumerate(demands)
                if demand > _EPSILON
            }
            remaining_capacity = cpu_capacity
            while active:
                equal_share = remaining_capacity / len(active)
                capped = {
                    index
                    for index in active
                    if demands[index] <= equal_share + _EPSILON
                }
                if not capped:
                    for index in active:
                        allocations[index] = equal_share
                    break
                for index in capped:
                    allocations[index] = demands[index]
                    remaining_capacity -= demands[index]
                active -= capped
        else:
            allocations = {
                index: min(demand, item.requested_cpu_cores)
                for index, (item, demand) in enumerate(zip(running, demands, strict=True))
            }
        physical_capacity_violation |= (
            sum(allocations.values()) > cpu_capacity + _EPSILON
        )

        future: list[float] = []
        for index, item in enumerate(running):
            if item.profile is None:
                future.append(item.finish_s)
            else:
                rate = demands[index]
                served_rate = allocations[index]
                progress_rate = 1.0 if rate <= served_rate else served_rate / rate
                future.append(now_s + item.profile_remaining_s / progress_rate)
                if item.next_observation_s < math.inf:
                    future.append(item.next_observation_s)
                if item.pending_at_s < math.inf:
                    future.append(item.pending_at_s)
        future.extend(
            session.ready_s
            for session in sessions
            if not session.running
            and session.command_index < len(session.program.commands)
            and session.ready_s > now_s + _EPSILON
        )
        if not future:
            raise ValueError("ready command cannot fit the empty feedback pool")
        next_s = min(value for value in future if value > now_s + _EPSILON)
        elapsed_s = next_s - now_s
        reserved_cpu_s += sum(
            item.requested_cpu_cores * elapsed_s for item in running
        )
        reserved_rss_mb_s += sum(
            item.requested_rss_mb * elapsed_s for item in running
        )
        for index, item in enumerate(running):
            if item.profile is None:
                continue
            rate = demands[index]
            served_rate = allocations[index]
            progress_rate = 1.0 if rate <= served_rate else served_rate / rate
            item.profile_remaining_s -= elapsed_s * progress_rate
            served_cpu_work += served_rate * elapsed_s
            if item.next_observation_s < math.inf:
                item.observed_cpu_core_s += served_rate * elapsed_s
                item.observed_throttling |= (
                    elapsed_s > 0.0 and rate > served_rate + _EPSILON
                )
            if item.profile_remaining_s <= _EPSILON:
                item.profile_index += 1
                if item.profile_index < len(item.profile):
                    item.profile_remaining_s = item.profile[item.profile_index][0]
        now_s = next_s

    completion = [session.completion_s for session in sessions]
    if any(value is None for value in completion):
        raise ValueError("feedback admission ended before every task completed")
    if (
        abs(used_cpu) > _EPSILON
        or abs(used_rss) > _EPSILON
        or set(service_by_command) != set(commands)
        or set(start_by_command) != set(commands)
    ):
        raise ValueError("feedback admission leaked a reservation")
    completion_s = [float(value) for value in completion if value is not None]
    service_s = sum(service_by_command.values())
    recorded_service_s = sum(command.duration_s for command in commands.values())
    total_cpu_work = sum(
        cpu for profile in cpu_work_profiles.values() for _dt, cpu in profile
    )
    if not math.isclose(
        served_cpu_work, total_cpu_work, rel_tol=1e-12, abs_tol=1e-7
    ):
        raise ValueError("feedback admission failed to conserve CPU work")
    return {
        "command_count": starts,
        "total_command_service_s": service_s,
        "recorded_command_service_s": recorded_service_s,
        "added_service_s": service_s - recorded_service_s,
        "makespan_s": max(completion_s),
        "mean_task_completion_s": statistics.fmean(completion_s),
        "total_command_queue_s": queue_s,
        "reserved_cpu_core_s": reserved_cpu_s,
        "reserved_rss_mb_s": reserved_rss_mb_s,
        "max_concurrent_commands": max_concurrent,
        "overlapped_command_ids": sorted(overlapped),
        "capacity_violation": capacity_violation,
        "physical_capacity_violation": physical_capacity_violation,
        "total_cpu_work_core_s": total_cpu_work,
        "served_cpu_work_core_s": served_cpu_work,
        "feedback_updates": feedback_updates,
        "reservation_shrinks": shrinks,
        "reservation_expansions": expansions,
        "denied_expansions": denied_expansions,
        "service_s_by_command": dict(sorted(service_by_command.items())),
        "start_s_by_command": dict(sorted(start_by_command.items())),
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
        total_weight = sum(running[index].cpu_share_weight for index in active)
        scale = remaining_capacity / total_weight
        capped = {
            index
            for index in active
            if running[index].max_cpu_cores
            <= running[index].cpu_share_weight * scale + _EPSILON
        }
        if not capped:
            for index in active:
                allocations[index] = running[index].cpu_share_weight * scale
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
    cpu_share_weights: Mapping[str, float] | None = None,
    admission_priorities: Mapping[str, float] | None = None,
    age_admission_priorities: bool = False,
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
    if cpu_share_weights is not None and set(cpu_share_weights) != set(commands):
        raise ValueError("burstable replay requires complete CPU share weights")
    if admission_priorities is not None and set(admission_priorities) != set(commands):
        raise ValueError("burstable replay requires complete admission priorities")
    if age_admission_priorities and admission_priorities is None:
        raise ValueError("priority aging requires admission priorities")
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
        share_weight = (
            requested_cpu
            if cpu_share_weights is None
            else cpu_share_weights[command_id]
        )
        values = (requested_cpu, requested_rss, demand, work, share_weight)
        if any(value < 0.0 for value in values):
            raise ValueError("burstable requests, demand, and work must be non-negative")
        if (
            requested_cpu <= 0.0
            or requested_cpu > demand + _EPSILON
            or demand > cpu_capacity + _EPSILON
            or requested_rss > rss_capacity_mb + _EPSILON
            or not math.isfinite(share_weight)
            or share_weight <= 0.0
        ):
            raise ValueError(f"invalid burstable request or demand: {command_id}")
    raw_share_weights = {
        command_id: (
            requested_reservations[command_id][0]
            if cpu_share_weights is None
            else cpu_share_weights[command_id]
        )
        for command_id in commands
    }
    max_share_weight = max(raw_share_weights.values())
    normalized_share_weights = {
        command_id: value / max_share_weight
        for command_id, value in raw_share_weights.items()
    }
    if min(normalized_share_weights.values()) < 1.0 / _MAX_CPU_SHARE_RATIO:
        raise ValueError("CPU share weights exceed the supported dynamic range")
    if admission_priorities is not None and any(
        not math.isfinite(value) for value in admission_priorities.values()
    ):
        raise ValueError("admission priorities must be finite")

    sessions = [
        _Session(program, rank, ready_s=program.initial_delay_s)
        for rank, program in enumerate(programs)
    ]
    now_s = min(session.ready_s for session in sessions)
    running: list[_BurstRunning] = []
    used_cpu = used_rss = 0.0
    queue_s = 0.0
    service_terms: list[float | None] = []
    reserved_cpu_terms: list[float | None] = []
    reserved_rss_terms: list[float | None] = []
    served_cpu_work = 0.0
    service_by_command: dict[str, float] = {}
    start_by_command: dict[str, float] = {}
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
            if abs(duration_s - item.command.duration_s) <= _EPSILON:
                duration_s = item.command.duration_s
            service_terms[item.start_index] = duration_s
            service_by_command[item.command.command_id] = duration_s
            reserved_cpu_terms[item.start_index] = (
                item.requested_cpu_cores * duration_s
            )
            reserved_rss_terms[item.start_index] = (
                item.requested_rss_mb * duration_s
            )
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
                0.0
                if admission_priorities is None
                else (
                    admission_priorities[
                        item[1].program.commands[item[1].command_index].command_id
                    ]
                    + (item[1].ready_s if age_admission_priorities else 0.0)
                ),
                item[1].ready_s,
                item[1].seed_rank,
                item[1].program.task_id,
            ),
        )
        for index, session in ready:
            command = session.program.commands[session.command_index]
            requested_cpu, requested_rss = requested_reservations[command.command_id]
            share_weight = normalized_share_weights[command.command_id]
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
                    starts,
                    command,
                    now_s,
                    now_s + command.duration_s,
                    requested_cpu,
                    requested_rss,
                    share_weight,
                    max_cpu_cores[command.command_id],
                    work,
                )
            )
            service_terms.append(None)
            start_by_command[command.command_id] = now_s
            reserved_cpu_terms.append(None)
            reserved_rss_terms.append(None)
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
        or any(value is None for value in service_terms)
        or any(value is None for value in reserved_cpu_terms)
        or any(value is None for value in reserved_rss_terms)
        or set(service_by_command) != set(commands)
        or set(start_by_command) != set(commands)
    ):
        raise ValueError("burstable replay leaked reservation or CPU work")
    service_s = sum(float(value) for value in service_terms if value is not None)
    reserved_cpu_s = sum(
        float(value) for value in reserved_cpu_terms if value is not None
    )
    reserved_rss_mb_s = sum(
        float(value) for value in reserved_rss_terms if value is not None
    )
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
        "service_s_by_command": dict(sorted(service_by_command.items())),
        "start_s_by_command": dict(sorted(start_by_command.items())),
        "contended_command_ids": sorted(contended),
        "added_service_s": service_s
        - sum(command.duration_s for command in commands.values()),
    }
