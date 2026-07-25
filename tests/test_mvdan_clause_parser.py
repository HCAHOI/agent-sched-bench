from __future__ import annotations

import pytest

from tool_resource.features import parse_command_clauses


_CLAUSE_KEYS = {
    "bin",
    "argv",
    "original",
    "span",
    "in_loop",
    "in_pipe",
    "in_subst",
    "pipeline_position",
}


def _operand(
    kind: str,
    index: int,
    clause_indices: list[int],
    span: tuple[int, int],
    *,
    negated: bool = False,
    pipeline: bool = False,
    subshell: bool = False,
) -> dict[str, object]:
    return {
        "kind": kind,
        "index": index,
        "clause_indices": clause_indices,
        "span": span,
        "negated": negated,
        "contains_pipeline": pipeline,
        "contains_subshell": subshell,
    }


@pytest.mark.parametrize(
    ("command", "bins", "spans"),
    [
        (
            "/usr/bin/printf '%s\\n' hello",
            ["printf"],
            [(0, 28)],
        ),
        (
            "echo one; printf two",
            ["echo", "printf"],
            [(0, 8), (10, 20)],
        ),
        (
            "(cd /tmp; pwd)",
            ["cd", "pwd"],
            [(1, 8), (10, 13)],
        ),
        (
            'echo "$(( $(date +%s) + 1))"',
            ["echo", "date"],
            [(0, 28), (12, 20)],
        ),
        (
            "printf '%s\\n' \"$(echo \"$(date +%s)\")\"",
            ["printf", "echo", "date"],
            [(0, 37), (17, 35), (25, 33)],
        ),
    ],
)
def test_contract_critical_bakeoff_cases(
    command: str,
    bins: list[str],
    spans: list[tuple[int, int]],
) -> None:
    parsed = parse_command_clauses(command)

    assert not parsed["parse_failed"]
    assert [clause["bin"] for clause in parsed["clauses"]] == bins
    assert [clause["span"] for clause in parsed["clauses"]] == spans
    for clause in parsed["clauses"]:
        assert set(clause) == _CLAUSE_KEYS
        start, end = clause["span"]
        assert clause["original"] == command[start:end]


def test_loops_and_pipeline_context() -> None:
    command = (
        "for item in a b; do echo \"$item\" | wc -c; done; "
        "for ((i=0; i<2; i++)); do printf '%s\\n' \"$i\"; done; "
        "while test -f flag; do sleep 1; done"
    )
    clauses = parse_command_clauses(command)["clauses"]
    by_bin = {clause["bin"]: clause for clause in clauses}

    assert [clause["bin"] for clause in clauses] == [
        "echo",
        "wc",
        "printf",
        "test",
        "sleep",
    ]
    assert all(clause["in_loop"] for clause in clauses)
    assert by_bin["echo"]["in_pipe"]
    assert by_bin["echo"]["pipeline_position"] == 0
    assert by_bin["wc"]["in_pipe"]
    assert by_bin["wc"]["pipeline_position"] == 1
    assert not by_bin["printf"]["in_pipe"]
    assert not by_bin["test"]["in_pipe"]


def test_nested_pipeline_and_command_substitution_context() -> None:
    command = 'echo "$(printf x | wc -c)" | tail -1'
    clauses = parse_command_clauses(command)["clauses"]
    by_bin = {clause["bin"]: clause for clause in clauses}

    assert [clause["bin"] for clause in clauses] == ["echo", "printf", "wc", "tail"]
    assert by_bin["echo"]["pipeline_position"] == 0
    assert by_bin["printf"]["in_subst"]
    assert by_bin["printf"]["pipeline_position"] == 0
    assert by_bin["wc"]["in_subst"]
    assert by_bin["wc"]["pipeline_position"] == 1
    assert by_bin["tail"]["pipeline_position"] == 1


def test_control_edges_preserve_mvdan_short_circuit_tree() -> None:
    parsed = parse_command_clauses("left && middle && right || fallback")

    assert [clause["bin"] for clause in parsed["clauses"]] == [
        "left",
        "middle",
        "right",
        "fallback",
    ]
    assert parsed["control_edges"] == [
        {
            "id": 0,
            "operator": "&&",
            "lhs": _operand("clause", 0, [0], (0, 4)),
            "rhs": _operand("clause", 1, [1], (8, 14)),
        },
        {
            "id": 1,
            "operator": "&&",
            "lhs": _operand("edge", 0, [0, 1], (0, 14)),
            "rhs": _operand("clause", 2, [2], (18, 23)),
        },
        {
            "id": 2,
            "operator": "||",
            "lhs": _operand("edge", 1, [0, 1, 2], (0, 23)),
            "rhs": _operand("clause", 3, [3], (27, 35)),
        },
    ]


@pytest.mark.parametrize(
    ("command", "operator", "rhs_span"),
    [
        ("false && (b | c)", "&&", (9, 16)),
        ("true || (b | c)", "||", (8, 15)),
    ],
)
def test_outer_control_edge_preserves_rhs_pipeline_subtree(
    command: str,
    operator: str,
    rhs_span: tuple[int, int],
) -> None:
    edge = parse_command_clauses(command)["control_edges"][0]

    assert edge == {
        "id": 0,
        "operator": operator,
        "lhs": _operand("clause", 0, [0], (0, len(command.split()[0]))),
        "rhs": _operand(
            "unsupported",
            -1,
            [1, 2],
            rhs_span,
            pipeline=True,
            subshell=True,
        ),
    }


@pytest.mark.parametrize("command", ["! left && right", "! left || right"])
def test_negated_control_operand_is_not_runtime_status_evidence(
    command: str,
) -> None:
    parsed = parse_command_clauses(command)

    assert parsed["control_edges"] == [
        {
            "id": 0,
            "operator": command.split()[2],
            "lhs": _operand(
                "unsupported",
                -1,
                [0],
                (0, 6),
                negated=True,
            ),
            "rhs": _operand("clause", 1, [1], (10, 15)),
        }
    ]


def test_quoted_and_multiple_heredoc_spans_exclude_bodies() -> None:
    commands_and_headers = [
        ("cat <<'EOF'\n$dollars stay literal\nEOF\n", "cat <<'EOF'"),
        ("cat <<A <<B\nfirst\nA\nsecond\nB\n", "cat <<A <<B"),
    ]

    for command, header in commands_and_headers:
        parsed = parse_command_clauses(command)
        assert not parsed["parse_failed"]
        assert len(parsed["clauses"]) == 1
        assert parsed["clauses"][0]["original"] == header
        assert parsed["clauses"][0]["span"] == (0, len(header))


def test_unicode_byte_spans_become_python_string_indices() -> None:
    command = "printf 'é'; echo \"雪$(date)\""
    clauses = parse_command_clauses(command)["clauses"]

    assert [clause["original"] for clause in clauses] == [
        "printf 'é'",
        'echo "雪$(date)"',
        "date",
    ]
    assert [clause["span"] for clause in clauses] == [
        (0, 10),
        (12, 27),
        (21, 25),
    ]
    assert clauses[0]["argv"] == ["printf", "é"]
    assert clauses[1]["argv"] == ["echo", "雪$(date)"]


def test_head_policy_and_non_executable_statements() -> None:
    command = (
        "X=1; >out; FOO=x /usr/bin/printf x; "
        "env VAR=x python x; timeout 5 python y; export PATH=/tmp"
    )
    clauses = parse_command_clauses(command)["clauses"]

    assert [clause["bin"] for clause in clauses] == [
        "printf",
        "env",
        "timeout",
        "export",
    ]
    assert clauses[0]["argv"] == ["/usr/bin/printf", "x"]
    assert clauses[1]["argv"] == ["env", "VAR=x", "python", "x"]
    assert clauses[2]["argv"] == ["timeout", "5", "python", "y"]


def test_escaped_and_empty_command_heads_preserve_logical_words() -> None:
    escaped = parse_command_clauses(
        r'\git status; foo\ bar x; echo "a\$b" "a\qb" "\雪"'
    )["clauses"]
    empty = parse_command_clauses("'' arg")["clauses"]

    assert [clause["bin"] for clause in escaped] == ["git", "foo bar", "echo"]
    assert escaped[0]["argv"] == ["git", "status"]
    assert escaped[1]["argv"] == ["foo bar", "x"]
    assert escaped[2]["argv"] == ["echo", "a$b", r"a\qb", "\\雪"]
    assert empty[0]["bin"] == ""
    assert empty[0]["argv"] == ["", "arg"]


def test_malformed_input_uses_shell_segment_fallback() -> None:
    command = "echo ok )"
    parsed = parse_command_clauses(command)

    assert parsed["parse_failed"]
    assert [clause["bin"] for clause in parsed["clauses"]] == ["echo"]
    assert parsed["clauses"][0]["original"] == command
    assert parsed["clauses"][0]["span"] == (0, len(command))
