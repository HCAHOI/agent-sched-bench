"""Bounded documentation-derived command semantics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from tool_resource.runtime_kb import _canonical_dynamic_value


MAX_SPEC_BYTES = 65_536
MAX_LITERAL_CHARS = 256
MAX_INVOCATIONS = 32
MAX_INVOCATION_TOKENS = 8
MAX_OPERATIONS = 32
MAX_ARGUMENTS = 128
MAX_FORMS = 8
MAX_RELATIONS = 256
MAX_POSITIONALS = 64
MAX_ARGV_TOKENS = 256
MAX_ARGV_LITERAL_BYTES = 4_096
MAX_FEATURES = 1_024
MAX_SCOPE_DEPTH = 16
ROLES = frozenset(
    {"work_item", "work_selector", "execution_policy", "output", "opaque"}
)
RELATION_KINDS = frozenset(
    {
        "unordered_collection",
        "fixed_value_equivalence",
        "scope_order",
        "requires",
        "excludes",
    }
)


@dataclass(frozen=True)
class Invocation:
    tokens: tuple[str, ...]
    operation: str


@dataclass(frozen=True)
class Argument:
    id: str
    forms: tuple[str, ...]
    arity: int
    role: str
    repeatable: bool


@dataclass(frozen=True)
class Positionals:
    id: str
    role: str
    min_items: int
    max_items: int


@dataclass(frozen=True)
class Operation:
    name: str
    arguments: tuple[Argument, ...]
    positionals: Positionals | None


@dataclass(frozen=True)
class Relation:
    kind: str
    operation: str
    argument: str
    other: str | None = None
    value: str | None = None
    delimiter: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    tool: str
    documented_version: str
    invocations: tuple[Invocation, ...]
    operations: tuple[Operation, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True)
class InterpretedArgv:
    scope: str
    features: frozenset[str]


def _object(value: Any, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("object keys differ from schema")
    return value


def _literal(value: Any, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_LITERAL_CHARS:
        raise ValueError("invalid bounded string")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("spec literals cannot contain whitespace or controls")
    if identifier and (
        not value[0].isalnum()
        or any(not (character.isalnum() or character in "._-") for character in value)
    ):
        raise ValueError("invalid identifier")
    return value


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError("integer outside schema bounds")
    return value


def _parse_argument(value: Any) -> Argument:
    row = _object(value, {"id", "forms", "arity", "role", "repeatable"})
    argument_id = _literal(row["id"], identifier=True)
    forms_value = row["forms"]
    if not isinstance(forms_value, list) or not 1 <= len(forms_value) <= MAX_FORMS:
        raise ValueError("argument forms outside bounds")
    forms = tuple(_literal(form) for form in forms_value)
    if len(set(forms)) != len(forms) or any(not form.startswith("-") or "=" in form for form in forms):
        raise ValueError("invalid option forms")
    arity = _integer(row["arity"], 0, 1)
    role = row["role"]
    if role not in ROLES:
        raise ValueError("unknown argument role")
    repeatable = row["repeatable"]
    if not isinstance(repeatable, bool):
        raise ValueError("repeatable must be boolean")
    return Argument(argument_id, forms, arity, role, repeatable)


def _parse_positionals(value: Any) -> Positionals | None:
    if value is None:
        return None
    row = _object(value, {"id", "role", "min_items", "max_items"})
    positional_id = _literal(row["id"], identifier=True)
    role = row["role"]
    if role not in ROLES:
        raise ValueError("unknown positional role")
    minimum = _integer(row["min_items"], 0, MAX_POSITIONALS)
    maximum = _integer(row["max_items"], 0, MAX_POSITIONALS)
    if minimum > maximum:
        raise ValueError("positional bounds are reversed")
    return Positionals(positional_id, role, minimum, maximum)


def _parse_relation(value: Any) -> Relation:
    if not isinstance(value, Mapping):
        raise ValueError("relation is not an object")
    kind = value.get("kind")
    if kind not in RELATION_KINDS:
        raise ValueError("unknown relation kind")
    common = {"kind", "operation", "argument"}
    expected = {
        "unordered_collection": common,
        "fixed_value_equivalence": common | {"other", "value"},
        "scope_order": common | {"delimiter"},
        "requires": common | {"other"},
        "excludes": common | {"other"},
    }[kind]
    row = _object(value, expected)
    return Relation(
        kind=kind,
        operation=_literal(row["operation"], identifier=True),
        argument=_literal(row["argument"], identifier=True),
        other=(
            _literal(row["other"], identifier=True) if "other" in row else None
        ),
        value=_literal(row["value"]) if "value" in row else None,
        delimiter=_literal(row["delimiter"]) if "delimiter" in row else None,
    )


def _has_cycle(edges: Mapping[str, str]) -> bool:
    for start in edges:
        seen: set[str] = set()
        node = start
        while node in edges:
            if node in seen:
                return True
            seen.add(node)
            node = edges[node]
    return False


def _validate_relations(spec: ToolSpec) -> None:
    operations = {operation.name: operation for operation in spec.operations}
    normalized: set[tuple[str, str, str, str | None, str | None, str | None]] = set()
    requires: set[tuple[str, str, str]] = set()
    excludes: set[tuple[str, str, str]] = set()
    fixed: dict[str, dict[str, str]] = defaultdict(dict)
    for relation in spec.relations:
        operation = operations.get(relation.operation)
        if operation is None:
            raise ValueError("relation references an unknown operation")
        arguments = {argument.id: argument for argument in operation.arguments}
        if operation.positionals is not None:
            arguments[operation.positionals.id] = operation.positionals
        if relation.argument not in arguments or (
            relation.other is not None and relation.other not in arguments
        ):
            raise ValueError("relation references an unknown argument")
        key = (
            relation.kind,
            relation.operation,
            relation.argument,
            relation.other,
            relation.value,
            relation.delimiter,
        )
        if key in normalized:
            raise ValueError("duplicate relation")
        normalized.add(key)
        if relation.kind == "unordered_collection" and (
            operation.positionals is None
            or relation.argument != operation.positionals.id
        ):
            raise ValueError("unordered_collection requires positionals")
        if relation.kind == "scope_order":
            subject = arguments[relation.argument]
            if (
                relation.delimiter is None
                or len(relation.delimiter) > 8
                or (
                    isinstance(subject, Argument)
                    and (subject.arity != 1 or subject.role not in {"work_item", "work_selector"})
                )
            ):
                raise ValueError("scope_order requires a scoped value")
        if relation.kind == "fixed_value_equivalence":
            source = arguments[relation.argument]
            target = arguments[relation.other]
            if (
                not isinstance(source, Argument)
                or not isinstance(target, Argument)
                or source.arity != 0
                or target.arity != 1
                or source.role != target.role
            ):
                raise ValueError("fixed equivalence requires option arguments")
            if relation.argument in fixed[relation.operation]:
                raise ValueError("argument has multiple fixed equivalents")
            fixed[relation.operation][relation.argument] = relation.other
        elif relation.kind == "requires":
            requires.add((relation.operation, relation.argument, relation.other))
        elif relation.kind == "excludes":
            left, right = sorted((relation.argument, relation.other))
            excludes.add((relation.operation, left, right))
    if any(
        (operation, *sorted((left, right))) in excludes
        for operation, left, right in requires
    ):
        raise ValueError("requires and excludes conflict")
    if any(_has_cycle(edges) for edges in fixed.values()):
        raise ValueError("fixed equivalence contains a cycle")


def _parse_tool_spec(value: Any) -> ToolSpec:
    if len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()) > MAX_SPEC_BYTES:
        raise ValueError("ToolSpec exceeds 64 KiB")
    row = _object(
        value,
        {"schema", "tool", "documented_version", "invocations", "operations", "relations"},
    )
    if row["schema"] != "tool-spec-v1":
        raise ValueError("unknown ToolSpec schema")
    tool = _literal(row["tool"], identifier=True)
    version = _literal(row["documented_version"])

    operation_values = row["operations"]
    if not isinstance(operation_values, list) or not 1 <= len(operation_values) <= MAX_OPERATIONS:
        raise ValueError("operations outside bounds")
    operations: list[Operation] = []
    for value_operation in operation_values:
        operation_row = _object(value_operation, {"name", "arguments", "positionals"})
        name = _literal(operation_row["name"], identifier=True)
        argument_values = operation_row["arguments"]
        if not isinstance(argument_values, list) or len(argument_values) > MAX_ARGUMENTS:
            raise ValueError("arguments outside bounds")
        arguments = tuple(_parse_argument(argument) for argument in argument_values)
        ids = [argument.id for argument in arguments]
        forms = [form for argument in arguments for form in argument.forms]
        positionals = _parse_positionals(operation_row["positionals"])
        if len(ids) != len(set(ids)) or len(forms) != len(set(forms)):
            raise ValueError("duplicate argument ID or option form")
        if positionals is not None and positionals.id in ids:
            raise ValueError("positional ID conflicts with an option")
        operations.append(Operation(name, arguments, positionals))
    operation_names = [operation.name for operation in operations]
    if len(operation_names) != len(set(operation_names)):
        raise ValueError("duplicate operation")

    invocation_values = row["invocations"]
    if not isinstance(invocation_values, list) or not 1 <= len(invocation_values) <= MAX_INVOCATIONS:
        raise ValueError("invocations outside bounds")
    invocations: list[Invocation] = []
    for value_invocation in invocation_values:
        invocation_row = _object(value_invocation, {"tokens", "operation"})
        token_values = invocation_row["tokens"]
        if not isinstance(token_values, list) or not 1 <= len(token_values) <= MAX_INVOCATION_TOKENS:
            raise ValueError("invocation tokens outside bounds")
        tokens = tuple(_literal(token) for token in token_values)
        operation = _literal(invocation_row["operation"], identifier=True)
        if operation not in set(operation_names):
            raise ValueError("invocation references an unknown operation")
        invocations.append(Invocation(tokens, operation))
    invocation_tokens = [invocation.tokens for invocation in invocations]
    if len(invocation_tokens) != len(set(invocation_tokens)):
        raise ValueError("duplicate invocation")

    relation_values = row["relations"]
    if not isinstance(relation_values, list) or len(relation_values) > MAX_RELATIONS:
        raise ValueError("relations outside bounds")
    spec = ToolSpec(
        tool,
        version,
        tuple(invocations),
        tuple(operations),
        tuple(_parse_relation(relation) for relation in relation_values),
    )
    _validate_relations(spec)
    return spec


def validate_tool_spec(value: Any) -> ToolSpec | None:
    """Return a compiled spec, or ``None`` for any invalid generator output."""

    try:
        return _parse_tool_spec(value)
    except (KeyError, TypeError, ValueError):
        return None


def _relation_maps(
    spec: ToolSpec, operation: str
) -> tuple[
    set[str],
    dict[str, tuple[str, str]],
    dict[str, str],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    unordered: set[str] = set()
    fixed: dict[str, tuple[str, str]] = {}
    scopes: dict[str, str] = {}
    requires: list[tuple[str, str]] = []
    excludes: list[tuple[str, str]] = []
    for relation in spec.relations:
        if relation.operation != operation:
            continue
        if relation.kind == "unordered_collection":
            unordered.add(relation.argument)
        elif relation.kind == "fixed_value_equivalence":
            fixed[relation.argument] = (relation.other, relation.value)
        elif relation.kind == "scope_order":
            scopes[relation.argument] = relation.delimiter
        elif relation.kind == "requires":
            requires.append((relation.argument, relation.other))
        elif relation.kind == "excludes":
            excludes.append((relation.argument, relation.other))
    def canonical(argument: str) -> str:
        return fixed.get(argument, (argument, ""))[0]

    return (
        unordered,
        fixed,
        scopes,
        tuple((canonical(left), canonical(right)) for left, right in requires),
        tuple((canonical(left), canonical(right)) for left, right in excludes),
    )


def _option_match(token: str, by_form: Mapping[str, Argument]) -> tuple[Argument, str | None] | None:
    direct = by_form.get(token)
    if direct is not None:
        return direct, None
    if token.startswith("--") and "=" in token:
        form, value = token.split("=", 1)
        argument = by_form.get(form)
        if argument is not None and argument.arity == 1 and value:
            return argument, value
        return None
    candidates = [
        (form, argument)
        for form, argument in by_form.items()
        if len(form) == 2
        and argument.arity == 1
        and token.startswith(form)
        and len(token) > 2
    ]
    if len(candidates) == 1:
        form, argument = candidates[0]
        return argument, token[len(form) :]
    return None


def _feature(role: str, argument: str, value: str | None, index: int | None = None) -> str:
    prefix = {
        "work_item": "work_item",
        "work_selector": "selector",
        "execution_policy": "policy",
        "output": "output",
        "opaque": "opaque",
    }[role]
    base = f"{prefix}:{argument}"
    if index is not None:
        base += f":{index}"
    if value is None or role == "output":
        return base
    rendered = _canonical_dynamic_value(value) if role == "opaque" else value
    return f"{base}={rendered}" if role == "execution_policy" else f"{base}:{rendered}"


def _scope_values(value: str, delimiter: str) -> tuple[str, ...] | None:
    parts = value.split(delimiter)
    if not 1 <= len(parts) <= MAX_SCOPE_DEPTH or any(not part for part in parts):
        return None
    prefixes = [parts[0]]
    for part in parts[1:]:
        prefixes.append(prefixes[-1] + delimiter + part)
    return tuple(prefixes)


def _valid_argv_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return len(value.encode()) <= MAX_ARGV_LITERAL_BYTES
    except UnicodeEncodeError:
        return False


def interpret_argv(
    spec: ToolSpec,
    bin_: str,
    argv: Sequence[str],
    observed_version: str,
) -> InterpretedArgv | None:
    """Interpret argv deterministically, abstaining on every unknown construct."""

    if (
        not isinstance(bin_, str)
        or not isinstance(observed_version, str)
        or observed_version != spec.documented_version
        or not argv
        or len(argv) > MAX_ARGV_TOKENS
        or any(not _valid_argv_token(value) for value in argv)
    ):
        return None
    if PurePosixPath(str(argv[0])).name != PurePosixPath(bin_).name:
        return None
    words = (PurePosixPath(str(argv[0])).name, *(str(value) for value in argv[1:]))
    matches = [
        invocation
        for invocation in spec.invocations
        if words[: len(invocation.tokens)] == invocation.tokens
    ]
    if not matches:
        return None
    invocation = max(matches, key=lambda item: len(item.tokens))
    operation = next(item for item in spec.operations if item.name == invocation.operation)
    by_form = {
        form: argument for argument in operation.arguments for form in argument.forms
    }
    by_id = {argument.id: argument for argument in operation.arguments}
    occurrences: dict[str, list[str | None]] = defaultdict(list)
    positional_values: list[str] = []
    operands_only = False
    tail = words[len(invocation.tokens) :]
    index = 0
    while index < len(tail):
        token = tail[index]
        if token == "--" and not operands_only:
            operands_only = True
            index += 1
            continue
        matched = None if operands_only or not token.startswith("-") else _option_match(token, by_form)
        if matched is not None:
            argument, inline_value = matched
            if occurrences[argument.id] and not argument.repeatable:
                return None
            value = inline_value
            if argument.arity == 1 and value is None:
                index += 1
                if index >= len(tail) or tail[index].startswith("-"):
                    return None
                value = tail[index]
            occurrences[argument.id].append(value)
        elif not operands_only and token.startswith("-"):
            return None
        else:
            positional_values.append(token)
        index += 1

    positionals = operation.positionals
    if positionals is None:
        if positional_values:
            return None
    elif not positionals.min_items <= len(positional_values) <= positionals.max_items:
        return None

    unordered, fixed, scopes, requires, excludes = _relation_maps(spec, operation.name)
    for source, (target, fixed_value) in fixed.items():
        if source not in occurrences:
            continue
        if target in occurrences and any(value != fixed_value for value in occurrences[target]):
            return None
        occurrences.pop(source)
        occurrences[target] = [fixed_value]

    present = set(occurrences)
    if positional_values and positionals is not None:
        present.add(positionals.id)
    if any(left in present and right not in present for left, right in requires):
        return None
    if any(left in present and right in present for left, right in excludes):
        return None

    features = {f"operation:{operation.name}"}
    for argument_id, values in occurrences.items():
        argument = by_id[argument_id]
        for value in values:
            scoped = (
                _scope_values(value, scopes[argument_id])
                if value is not None and argument_id in scopes
                else (value,)
            )
            if scoped is None:
                return None
            features.update(
                _feature(argument.role, argument_id, item) for item in scoped
            )
    if positionals is not None:
        delimiter = scopes.get(positionals.id)
        for ordinal, value in enumerate(positional_values):
            values = [value]
            if delimiter is not None:
                scoped = _scope_values(value, delimiter)
                if scoped is None:
                    return None
                values = list(scoped)
            feature_index = None if positionals.id in unordered else ordinal
            features.update(
                _feature(positionals.role, positionals.id, item, feature_index)
                for item in values
            )
    if len(features) > MAX_FEATURES:
        return None
    return InterpretedArgv(f"{spec.tool}:{operation.name}", frozenset(features))


def tool_spec_schema() -> dict[str, Any]:
    """Return the strict structured-output schema used by the one-shot compiler."""

    literal = {"type": "string", "minLength": 1, "maxLength": MAX_LITERAL_CHARS}
    identifier = dict(literal)
    argument = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "forms", "arity", "role", "repeatable"],
        "properties": {
            "id": identifier,
            "forms": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_FORMS,
                "items": literal,
            },
            "arity": {"type": "integer", "enum": [0, 1]},
            "role": {"type": "string", "enum": sorted(ROLES)},
            "repeatable": {"type": "boolean"},
        },
    }
    positional = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "role", "min_items", "max_items"],
        "properties": {
            "id": identifier,
            "role": {"type": "string", "enum": sorted(ROLES)},
            "min_items": {"type": "integer", "minimum": 0, "maximum": MAX_POSITIONALS},
            "max_items": {"type": "integer", "minimum": 0, "maximum": MAX_POSITIONALS},
        },
    }

    def relation(kind: str, extra: Mapping[str, Any]) -> dict[str, Any]:
        properties = {
            "kind": {"type": "string", "const": kind},
            "operation": identifier,
            "argument": identifier,
            **extra,
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "tool",
            "documented_version",
            "invocations",
            "operations",
            "relations",
        ],
        "properties": {
            "schema": {"type": "string", "const": "tool-spec-v1"},
            "tool": identifier,
            "documented_version": literal,
            "invocations": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_INVOCATIONS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["tokens", "operation"],
                    "properties": {
                        "tokens": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_INVOCATION_TOKENS,
                            "items": literal,
                        },
                        "operation": identifier,
                    },
                },
            },
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_OPERATIONS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "arguments", "positionals"],
                    "properties": {
                        "name": identifier,
                        "arguments": {
                            "type": "array",
                            "maxItems": MAX_ARGUMENTS,
                            "items": argument,
                        },
                        "positionals": {"anyOf": [{"type": "null"}, positional]},
                    },
                },
            },
            "relations": {
                "type": "array",
                "maxItems": MAX_RELATIONS,
                "items": {
                    "anyOf": [
                        relation("unordered_collection", {}),
                        relation("fixed_value_equivalence", {"other": identifier, "value": literal}),
                        relation(
                            "scope_order",
                            {
                                "delimiter": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 8,
                                }
                            },
                        ),
                        relation("requires", {"other": identifier}),
                        relation("excludes", {"other": identifier}),
                    ]
                },
            },
        },
    }
