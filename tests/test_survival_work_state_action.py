from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from spike.multitenant import ToolSpan, TraceProgram, TraceTurn

from scripts.evaluation.evaluate_survival_work_state_action import (
    AptWork,
    CausalDurationMemory,
    CausalWorkState,
    DurationObservation,
    ReplayExec,
    _arm_gate,
    _assert_program_order,
    _program_context,
    command_work_key,
    duration_observations,
    parse_apt_work,
)


def test_cli_exposes_the_frozen_evaluator() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluation/evaluate_survival_work_state_action.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout


def test_apt_and_command_work_signatures_are_name_free_and_fail_closed() -> None:
    assert parse_apt_work(("apt-get", "update", "-qq")) == AptWork("update", ())
    assert parse_apt_work(
        ("apt", "install", "-y", "python3-zeta", "python3-alpha")
    ) == AptWork("install", ("python3-alpha", "python3-zeta"))
    assert parse_apt_work(("apt-get", "install", "--mystery", "alpha")) is None

    key = command_work_key(
        "apt-get update -qq && apt-get install -y beta alpha 2>&1 | tail -10"
    )
    assert key is not None
    assert key == (
        ("apt", AptWork("update", ())),
        ("apt", AptWork("install", ("alpha", "beta"))),
    )


def test_causal_state_uses_only_prior_completed_work() -> None:
    state = CausalWorkState()
    install = command_work_key("apt-get install -y alpha beta")
    pytest = command_work_key("python3 -m pytest tests/test_x.py -q")
    pip = command_work_key("python3 -m pip install numpy")
    assert install is not None and pytest is not None and pip is not None

    assert state.query(install) == (("apt", False, ("alpha", "beta")),)
    state.observe("apt-get update", "updated\nExit code: 0")
    assert state.query(install) == (("apt", True, ("alpha", "beta")),)
    state.observe("apt-get install -y alpha", "installed\nExit code: 0")
    assert state.query(install) == (("apt", True, ("beta",)),)

    assert state.query(pytest) == (("pytest", False),)
    state.observe("pytest tests/test_a.py", "one failed\nExit code: 1")
    assert state.query(pytest) == (("pytest", True),)

    assert state.query(pip)[0][1:] == (False, "unknown", ("numpy",))
    state.observe(
        "python3 -m pip install numpy",
        "Successfully installed numpy\nExit code: 0",
    )
    assert state.query(pip)[0][1:] == (True, "present", ())


def test_causal_state_rejects_ambiguous_shell_control() -> None:
    state = CausalWorkState()
    install = command_work_key("apt-get install -y alpha beta")
    assert install is not None

    state.observe("false && apt-get install -y alpha", "Exit code: 0")
    state.observe("apt-get install -y beta || true", "failed\nExit code: 0")
    assert state.query(install) == (("apt", False, ("alpha", "beta")),)


def test_duration_memory_prefers_settled_repo_history_and_excludes_public_repo() -> None:
    work = command_work_key("pytest tests/test_x.py")
    assert work is not None
    state = (("pytest", False),)
    public = [
        DurationObservation("repo-a", "pytest tests/test_x.py", work, state, 9000.0),
        DurationObservation("repo-b", "pytest tests/test_x.py", work, state, 7000.0),
        DurationObservation(
            "repo-b",
            "pytest tests/test_x.py::test_one",
            command_work_key("pytest tests/test_x.py::test_one"),
            state,
            100.0,
        ),
    ]
    memory = CausalDurationMemory(public)

    assert memory.query("repo-a", "pytest tests/test_x.py", work, state) == {
        "exact_command": (7000.0,),
        "work_signature": (7000.0,),
        "work_plus_state": (7000.0,),
    }
    memory.observe_task(
        [DurationObservation("repo-a", "pytest tests/test_x.py", work, state, 100.0)]
    )
    assert memory.query("repo-a", "pytest tests/test_x.py", work, state) == {
        "exact_command": (100.0,),
        "work_signature": (100.0,),
        "work_plus_state": (100.0,),
    }


def test_duration_observations_capture_pre_command_state() -> None:
    command = "python3 -m pip install numpy"
    rows = duration_observations(
        "repo-a",
        [
            ReplayExec(command, 9000.0, "Successfully installed numpy\nExit code: 0"),
            ReplayExec(command, 5.0, "Requirement already satisfied\nExit code: 0"),
        ],
    )
    assert rows[0].state is not None and rows[1].state is not None
    assert rows[0].state[0][1:] == (False, "unknown", ("numpy",))
    assert rows[1].state[0][1:] == (True, "present", ())


def test_frozen_order_and_gate_are_explicit() -> None:
    class Program:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id

    programs = [Program("a"), Program("b")]
    _assert_program_order(programs, ["a", "b"], expected_count=2)
    try:
        _assert_program_order(programs, ["b", "a"], expected_count=2)
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("permuted task order was accepted")

    checks = _arm_gate(
        {
            "released_gib_s_delta": 1.0,
            "critical_path_stall_ms_delta": 0.0,
            "changed_task_count": 20,
        }
    )
    assert checks == {
        "released_gib_s_strictly_higher": True,
        "critical_path_stall_not_higher": True,
        "changed_at_least_20_tasks": True,
    }


def test_program_context_does_not_leak_overlapping_sibling_state(
    tmp_path: Path,
) -> None:
    command = "python3 -m pip install numpy"
    trace = tmp_path / "trace.jsonl"
    actions = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": command}),
                "tool_result": "Successfully installed numpy\nExit code: 0",
            },
        }
        for _ in range(3)
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in actions))
    program = TraceProgram(
        "repo-1",
        str(trace),
        (
            TraceTurn(
                (),
                0,
                0,
                10.0,
                (
                    ToolSpan("exec", command, 0.0, 10.0),
                    ToolSpan("exec", command, 5.0, 6.0),
                ),
            ),
            TraceTurn(
                (),
                0,
                0,
                1.0,
                (ToolSpan("exec", command, 0.0, 1.0),),
            ),
        ),
    )

    context = _program_context(program)
    assert context.states[(0, 1)][0][1] is False
    assert context.states[(1, 0)][0][1] is True
