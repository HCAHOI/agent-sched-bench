"""Static Bash clause parsing for prediction and Runtime KB identity.

The mvdan adapter is the parser of record; ``_shell_split`` provides the
degraded fallback used only when the adapter cannot parse a command.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from tool_resource._shell_split import shell_command_segments
from tool_resource.mvdan_client import (
    MvdanClientError,
    get_client as get_mvdan_client,
)


_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def parse_command_clauses(command: str) -> dict[str, Any]:
    """Return Bash clauses and causal context, falling back on malformed input.

    Mvdan byte offsets are converted to Python code-point indices. Leading
    assignments are excluded from argv, while executable wrappers remain the
    head. Statements without an executable head emit no clause.
    """

    response = get_mvdan_client().parse(command)
    if not response.get("ok"):
        return {
            "clauses": _fallback_clauses(command),
            "control_edges": [],
            "parse_failed": True,
        }
    raw_clauses = response.get("clauses")
    if not isinstance(raw_clauses, list):
        raise MvdanClientError("mvdan adapter response has no clause list")
    byte_to_character = _byte_to_character_offsets(command)
    clauses = [
        _clause_from_adapter(command, raw_clause, byte_to_character)
        for raw_clause in raw_clauses
    ]
    raw_control_edges = response.get("control_edges")
    if not isinstance(raw_control_edges, list):
        raise MvdanClientError("mvdan adapter response has no control-edge list")
    control_edges = [
        _control_edge_from_adapter(
            raw_edge,
            edge_id,
            len(clauses),
            byte_to_character,
        )
        for edge_id, raw_edge in enumerate(raw_control_edges)
    ]
    return {
        "clauses": clauses,
        "control_edges": control_edges,
        "parse_failed": False,
    }


def _byte_to_character_offsets(command: str) -> tuple[int, ...]:
    encoded = command.encode()
    offsets = [-1] * (len(encoded) + 1)
    byte_offset = 0
    for character_offset, character in enumerate(command):
        offsets[byte_offset] = character_offset
        byte_offset += len(character.encode())
    offsets[byte_offset] = len(command)
    return tuple(offsets)


def _clause_from_adapter(
    command: str,
    raw_clause: object,
    byte_to_character: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(raw_clause, dict):
        raise MvdanClientError("mvdan adapter returned a non-object clause")
    raw_span = raw_clause.get("span")
    argv = raw_clause.get("argv")
    if (
        not isinstance(raw_span, list)
        or len(raw_span) != 2
        or not all(isinstance(offset, int) for offset in raw_span)
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(argument, str) for argument in argv)
    ):
        raise MvdanClientError("mvdan adapter returned an invalid clause")
    byte_start, byte_end = raw_span
    if (
        byte_start < 0
        or byte_end < byte_start
        or byte_end >= len(byte_to_character)
        or byte_to_character[byte_start] < 0
        or byte_to_character[byte_end] < 0
    ):
        raise MvdanClientError(f"mvdan adapter returned invalid byte span {raw_span}")
    start = byte_to_character[byte_start]
    end = byte_to_character[byte_end]
    raw_words = raw_clause.get("words")
    if not isinstance(raw_words, list):
        raise MvdanClientError("mvdan adapter returned invalid word intents")
    structural_context = raw_clause.get("structural_context")
    if not isinstance(structural_context, list) or not all(
        isinstance(item, str) for item in structural_context
    ):
        raise MvdanClientError("mvdan adapter returned invalid structural context")

    def intent_span(raw: object) -> tuple[int, int]:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or not all(isinstance(offset, int) for offset in raw)
            or raw[0] < 0
            or raw[1] < raw[0]
            or raw[1] >= len(byte_to_character)
            or byte_to_character[raw[0]] < 0
            or byte_to_character[raw[1]] < 0
        ):
            raise MvdanClientError("mvdan adapter returned invalid word span")
        return byte_to_character[raw[0]], byte_to_character[raw[1]]

    word_intents: list[dict[str, Any]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict) or not isinstance(
            raw_word.get("components"), list
        ):
            raise MvdanClientError("mvdan adapter returned invalid word intent")
        components: list[dict[str, Any]] = []
        for raw_component in raw_word["components"]:
            if (
                not isinstance(raw_component, dict)
                or raw_component.get("kind")
                not in {
                    "literal",
                    "parameter",
                    "command_substitution",
                    "arithmetic_expansion",
                    "process_substitution",
                    "pathname_expansion",
                    "unsupported",
                }
                or not isinstance(raw_component.get("source"), str)
                or not isinstance(raw_component.get("quoted"), bool)
                or not isinstance(raw_component.get("escaped"), bool)
            ):
                raise MvdanClientError(
                    "mvdan adapter returned invalid word component"
                )
            components.append(
                {
                    "kind": raw_component["kind"],
                    "source": raw_component["source"],
                    "span": intent_span(raw_component.get("span")),
                    "quoted": raw_component["quoted"],
                    "escaped": raw_component["escaped"],
                }
            )
        if (
            not isinstance(raw_word.get("cooked"), str)
            or not isinstance(raw_word.get("source"), str)
            or not isinstance(raw_word.get("quoted"), bool)
            or not isinstance(raw_word.get("escaped"), bool)
        ):
            raise MvdanClientError("mvdan adapter returned invalid word intent")
        word_intents.append(
            {
                "cooked": raw_word["cooked"],
                "source": raw_word["source"],
                "span": intent_span(raw_word.get("span")),
                "quoted": raw_word["quoted"],
                "escaped": raw_word["escaped"],
                "components": components,
            }
        )
    if word_intents and [word["cooked"] for word in word_intents] != argv:
        raise MvdanClientError("mvdan adapter word intents disagree with argv")
    return {
        "bin": str(raw_clause["bin"]),
        "argv": argv,
        "original": command[start:end],
        "span": (start, end),
        "in_loop": bool(raw_clause["in_loop"]),
        "in_pipe": bool(raw_clause["in_pipe"]),
        "in_subst": bool(raw_clause["in_subst"]),
        "pipeline_position": int(raw_clause["pipeline_position"]),
        "structural_context": structural_context,
        "word_intents": word_intents,
    }


def _control_edge_from_adapter(
    raw_edge: object,
    edge_id: int,
    clause_count: int,
    byte_to_character: Sequence[int],
) -> dict[str, Any]:
    if (
        not isinstance(raw_edge, dict)
        or raw_edge.get("id") != edge_id
        or raw_edge.get("operator") not in {"&&", "||"}
    ):
        raise MvdanClientError("mvdan adapter returned an invalid control edge")

    def operand(name: str) -> dict[str, Any]:
        raw = raw_edge.get(name)
        if not isinstance(raw, dict):
            raise MvdanClientError("mvdan adapter returned an invalid control operand")
        kind, index, indices = raw.get("kind"), raw.get("index"), raw.get(
            "clause_indices"
        )
        raw_span = raw.get("span")
        negated = raw.get("negated")
        contains_pipeline = raw.get("contains_pipeline")
        contains_subshell = raw.get("contains_subshell")
        if (
            kind not in {"clause", "edge", "unsupported"}
            or not isinstance(index, int)
            or not isinstance(indices, list)
            or not all(
                isinstance(item, int) and 0 <= item < clause_count
                for item in indices
            )
            or (kind == "clause" and (index not in indices or len(indices) != 1))
            or (kind == "edge" and not 0 <= index < edge_id)
            or (kind == "unsupported" and index != -1)
            or not isinstance(raw_span, list)
            or len(raw_span) != 2
            or not all(isinstance(offset, int) for offset in raw_span)
            or raw_span[0] < 0
            or raw_span[1] < raw_span[0]
            or raw_span[1] >= len(byte_to_character)
            or byte_to_character[raw_span[0]] < 0
            or byte_to_character[raw_span[1]] < 0
            or not isinstance(negated, bool)
            or not isinstance(contains_pipeline, bool)
            or not isinstance(contains_subshell, bool)
        ):
            raise MvdanClientError("mvdan adapter returned an invalid control operand")
        return {
            "kind": kind,
            "index": index,
            "clause_indices": indices,
            "span": (
                byte_to_character[raw_span[0]],
                byte_to_character[raw_span[1]],
            ),
            "negated": negated,
            "contains_pipeline": contains_pipeline,
            "contains_subshell": contains_subshell,
        }

    return {
        "id": edge_id,
        "operator": raw_edge["operator"],
        "lhs": operand("lhs"),
        "rhs": operand("rhs"),
    }


def _fallback_clauses(command: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for segment in shell_command_segments(command):
        groups: list[list[str]] = [[]]
        for token in segment:
            if token in {"|", "|&", "&"}:
                if groups[-1]:
                    groups.append([])
            else:
                groups[-1].append(token)
        groups = [group for group in groups if group]
        for position, group in enumerate(groups):
            clause = _clause_from_words(group)
            if clause is None:
                continue
            clauses.append(
                {
                    **clause,
                    "original": command,
                    "span": (0, len(command)),
                    "in_loop": False,
                    "in_pipe": len(groups) > 1,
                    "in_subst": False,
                    "pipeline_position": position if len(groups) > 1 else -1,
                }
            )
    return clauses


def _clause_from_words(words: Sequence[str]) -> dict[str, Any] | None:
    start = next(
        (index for index, word in enumerate(words) if not _ENV_ASSIGNMENT.match(word)),
        None,
    )
    if start is None:
        return None
    argv = list(words[start:])
    return {"bin": argv[0].rsplit("/", 1)[-1], "argv": argv}



__all__ = ["parse_command_clauses"]
