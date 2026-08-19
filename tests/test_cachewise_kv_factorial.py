import numpy as np
import pytest

import tool_resource_eval.cachewise_kv_factorial as factorial
from tool_resource_eval.cachewise_kv_factorial import (
    CAPACITY_BLOCKS,
    Program,
    Session,
    Turn,
    _choose_session,
    _bootstrap_ratio,
    _make_room,
    _paper_sized,
    simulate,
)
from tool_resource_eval.cachewise_reproduction import (
    Gap,
    _histories,
    fit_clusters,
)


def _program(task_id: str, prompt_tokens: int) -> Program:
    return Program(task_id, (Turn(prompt_tokens, 16, 1.0, None),))


def test_prefix_scheduler_prefers_the_smallest_additional_allocation() -> None:
    older = Session(_program("older", 160), seed_rank=0, resident_blocks=0)
    reusable = Session(
        Program(
            "reusable",
            (
                Turn(64, 16, 1.0, None),
                Turn(160, 16, 1.0, None),
            ),
        ),
        seed_rank=1,
        turn_index=1,
        resident_blocks=5,
    )

    assert _choose_session([older, reusable], "fcfs") is older
    assert _choose_session([older, reusable], "prefix") is reusable


def test_simulator_preserves_a_resident_prefix_without_pressure() -> None:
    gap = Gap("task", 0.0, 1.0, "exec", "command")
    program = Program(
        "task",
        (
            Turn(16, 16, 0.1, gap),
            Turn(32, 16, 0.1, None),
        ),
    )

    result = simulate(
        [program],
        scheduler="fcfs",
        eviction="lru",
        global_history=np.asarray([1.0]),
        tool_history={"exec": np.asarray([1.0])},
        clusters={100: {}},
        label_cache={},
    )

    assert result["request_count"] == 2
    assert result["evicted_blocks"] == 0
    assert result["recomputed_prefix_blocks"] == 0


def test_c100_evicts_the_session_with_longer_predicted_reuse() -> None:
    fit = [
        Gap("fit", 0.0, duration, "exec", args)
        for args, duration in (("short", 1.0), ("short", 2.0), ("long", 9.0), ("long", 10.0))
    ]
    global_history, tool_history = _histories(fit)
    clusters, _ = fit_clusters(fit, cluster_counts=(100,))
    current = Session(_program("current", 16), seed_rank=0)

    def waiting(task_id: str, gap: Gap) -> Session:
        return Session(
            Program(
                task_id,
                (Turn(16, 16, 1.0, gap), Turn(32, 16, 1.0, None)),
            ),
            seed_rank=1,
            turn_index=1,
            arrival_s=gap.end,
            resident_blocks=5,
            last_access_s=0.0,
            gap_started_s=0.0,
        )

    short = waiting("short", Gap("short", 0.0, 1.0, "exec", "short"))
    long = waiting("long", Gap("long", 0.0, 9.0, "exec", "long"))

    _, evicted, events = _make_room(
        1,
        total_blocks=CAPACITY_BLOCKS,
        current=current,
        sessions=[current, short, long],
        now_s=0.5,
        eviction="c100",
        global_history=global_history,
        tool_history=tool_history,
        clusters=clusters,
        label_cache={},
    )

    assert (evicted, events) == (1, 1)
    assert long.resident_blocks == 4
    assert short.resident_blocks == 5


def test_c100_equal_predictions_use_lru_then_task_id() -> None:
    current = Session(_program("current", 16), seed_rank=0)
    first = Session(
        _program("a", 16), seed_rank=1, resident_blocks=1, last_access_s=0.0
    )
    second = Session(
        _program("b", 16), seed_rank=2, resident_blocks=1, last_access_s=0.0
    )
    recent = Session(
        _program("c", 16), seed_rank=3, resident_blocks=1, last_access_s=1.0
    )

    _make_room(
        1,
        total_blocks=CAPACITY_BLOCKS,
        current=current,
        sessions=[current, recent, second, first],
        now_s=0.0,
        eviction="c100",
        global_history=np.asarray([1.0]),
        tool_history={},
        clusters={100: {}},
        label_cache={},
    )

    assert first.resident_blocks == 0
    assert second.resident_blocks == 1
    assert recent.resident_blocks == 1


def test_custom_remaining_predictor_controls_the_victim() -> None:
    current = Session(_program("current", 16), seed_rank=0)
    short = Session(_program("short", 16), seed_rank=1, resident_blocks=1)
    long = Session(_program("long", 16), seed_rank=2, resident_blocks=1)

    _make_room(
        1,
        total_blocks=CAPACITY_BLOCKS,
        current=current,
        sessions=[current, short, long],
        now_s=0.0,
        eviction="predicted",
        global_history=np.asarray([1.0]),
        tool_history={},
        clusters={100: {}},
        label_cache={},
        remaining_predictor=lambda session, _now: {
            "short": 1.0,
            "long": 10.0,
        }[session.program.task_id],
    )

    assert short.resident_blocks == 1
    assert long.resident_blocks == 0


def test_block_belady_evicts_dead_suffix_then_latest_reuse() -> None:
    current = Session(_program("current", 16), seed_rank=0)
    near = Session(
        Program(
            "near",
            (Turn(64, 16, 1.0, None), Turn(32, 16, 1.0, None)),
        ),
        seed_rank=1,
        turn_index=1,
        resident_blocks=5,
    )
    far = Session(
        Program(
            "far",
            (Turn(64, 16, 1.0, None), Turn(80, 16, 1.0, None)),
        ),
        seed_rank=2,
        turn_index=1,
        resident_blocks=5,
    )

    _make_room(
        4,
        total_blocks=CAPACITY_BLOCKS,
        current=current,
        sessions=[current, near, far],
        now_s=0.0,
        eviction="belady",
        global_history=np.asarray([1.0]),
        tool_history={},
        clusters={100: {}},
        label_cache={},
        next_request_rank={("near", 1): 1, ("far", 1): 10},
    )

    assert near.resident_blocks == 2
    assert far.resident_blocks == 4


def test_suffix_arrival_uses_gap_arrival_after_dead_suffixes() -> None:
    current = Session(_program("current", 16), seed_rank=0)
    later_arrival = Session(
        Program(
            "later",
            (Turn(64, 16, 1.0, None), Turn(80, 16, 1.0, None)),
        ),
        seed_rank=1,
        turn_index=1,
        arrival_s=10.0,
        resident_blocks=5,
    )
    later_rank = Session(
        Program(
            "rank",
            (Turn(64, 16, 1.0, None), Turn(80, 16, 1.0, None)),
        ),
        seed_rank=2,
        turn_index=1,
        arrival_s=1.0,
        resident_blocks=5,
    )

    _make_room(
        1,
        total_blocks=CAPACITY_BLOCKS,
        current=current,
        sessions=[current, later_rank, later_arrival],
        now_s=0.0,
        eviction="suffix_arrival",
        global_history=np.asarray([1.0]),
        tool_history={},
        clusters={100: {}},
        label_cache={},
        next_request_rank={("later", 1): 1, ("rank", 1): 10},
    )

    assert later_arrival.resident_blocks == 4
    assert later_rank.resident_blocks == 5


def test_suffix_arrival_keeps_greedy_tie_break_for_arrived_sessions() -> None:
    current = Session(_program("current", 16), seed_rank=0)
    turns = (Turn(64, 16, 1.0, None), Turn(64, 16, 1.0, None))
    lru_victim = Session(
        Program("lru", turns),
        seed_rank=1,
        turn_index=1,
        arrival_s=8.0,
        resident_blocks=4,
        last_access_s=1.0,
    )
    later_arrival = Session(
        Program("later", turns),
        seed_rank=2,
        turn_index=1,
        arrival_s=9.0,
        resident_blocks=4,
        last_access_s=2.0,
    )

    _make_room(
        1,
        total_blocks=CAPACITY_BLOCKS,
        current=current,
        sessions=[current, lru_victim, later_arrival],
        now_s=10.0,
        eviction="suffix_arrival",
        global_history=np.asarray([1.0]),
        tool_history={},
        clusters={100: {}},
        label_cache={},
    )

    assert lru_victim.resident_blocks == 3
    assert later_arrival.resident_blocks == 4


def test_arrival_during_service_releases_suffix_before_growth(monkeypatch) -> None:
    monkeypatch.setattr(factorial, "CAPACITY_BLOCKS", 5)
    shrink = Program(
        "shrink",
        (
            Turn(32, 16, 0.1, Gap("shrink", 0.0, 0.1, "exec", "shrink")),
            Turn(16, 16, 0.1, None),
        ),
    )
    grow = Program("grow", (Turn(16, 32, 1.0, None),))

    result = simulate(
        [shrink, grow],
        scheduler="fcfs",
        eviction="lru",
        global_history=np.asarray([0.1]),
        tool_history={"exec": np.asarray([0.1])},
        clusters={100: {}},
        label_cache={},
    )

    assert result["evicted_blocks"] == 0


def test_partial_completion_requires_a_physical_block(monkeypatch) -> None:
    monkeypatch.setattr(factorial, "CAPACITY_BLOCKS", 1)

    with pytest.raises(ValueError, match="cannot fit"):
        simulate(
            [Program("partial", (Turn(16, 1, 0.1, None),))],
            scheduler="fcfs",
            eviction="lru",
            global_history=np.asarray([1.0]),
            tool_history={},
            clusters={100: {}},
            label_cache={},
        )


def test_paper_sized_gate_requires_c100_alone_to_help() -> None:
    assert _paper_sized(-1.0, -1.0, -1.0, 2.0) is True
    assert _paper_sized(-1.0, 1.0, -1.0, 2.0) is False


def test_ratio_bootstraps_paired_means_when_one_seed_has_zero_evictions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(factorial, "BOOTSTRAP_DRAWS", 100)

    result = _bootstrap_ratio([2.0, 4.0], [0.0, 2.0])

    assert result["ratio_of_means"] == 3.0
    assert result["zero_denominator_draws"] > 0
    assert result["ci95_paired_seed_bootstrap"][1] is None
