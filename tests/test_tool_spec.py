from copy import deepcopy

from tool_resource.tool_spec import interpret_argv, tool_spec_schema, validate_tool_spec


def _pytest_value() -> dict:
    return {
        "schema": "tool-spec-v1",
        "tool": "pytest",
        "documented_version": "8.3.5",
        "invocations": [
            {"tokens": ["pytest"], "operation": "run"},
            {"tokens": ["python", "-m", "pytest"], "operation": "run"},
        ],
        "operations": [
            {
                "name": "run",
                "arguments": [
                    {
                        "id": "maxfail",
                        "forms": ["--maxfail"],
                        "arity": 1,
                        "role": "execution_policy",
                        "repeatable": False,
                    },
                    {
                        "id": "exitfirst",
                        "forms": ["-x", "--exitfirst"],
                        "arity": 0,
                        "role": "execution_policy",
                        "repeatable": False,
                    },
                    {
                        "id": "workers",
                        "forms": ["-n"],
                        "arity": 1,
                        "role": "execution_policy",
                        "repeatable": False,
                    },
                    {
                        "id": "dist",
                        "forms": ["--dist"],
                        "arity": 1,
                        "role": "execution_policy",
                        "repeatable": False,
                    },
                    {
                        "id": "collect",
                        "forms": ["--collect-only"],
                        "arity": 0,
                        "role": "execution_policy",
                        "repeatable": False,
                    },
                    {
                        "id": "junit",
                        "forms": ["--junitxml"],
                        "arity": 1,
                        "role": "output",
                        "repeatable": False,
                    },
                    {
                        "id": "plugin",
                        "forms": ["-p"],
                        "arity": 1,
                        "role": "opaque",
                        "repeatable": True,
                    },
                ],
                "positionals": {
                    "id": "targets",
                    "role": "work_item",
                    "min_items": 0,
                    "max_items": 32,
                },
            }
        ],
        "relations": [
            {
                "kind": "unordered_collection",
                "operation": "run",
                "argument": "targets",
            },
            {
                "kind": "fixed_value_equivalence",
                "operation": "run",
                "argument": "exitfirst",
                "other": "maxfail",
                "value": "1",
            },
            {
                "kind": "scope_order",
                "operation": "run",
                "argument": "targets",
                "delimiter": "::",
            },
            {
                "kind": "requires",
                "operation": "run",
                "argument": "dist",
                "other": "workers",
            },
            {
                "kind": "excludes",
                "operation": "run",
                "argument": "collect",
                "other": "workers",
            },
        ],
    }


def test_interpreter_canonicalizes_aliases_work_sets_and_scope() -> None:
    spec = validate_tool_spec(_pytest_value())
    assert spec is not None

    direct = interpret_argv(
        spec,
        "pytest",
        (
            "pytest",
            "-x",
            "--junitxml",
            "out.xml",
            "-p",
            "plugin_name",
            "b.py::test_b",
            "a.py::test_a",
        ),
        "8.3.5",
    )
    module = interpret_argv(
        spec,
        "python",
        (
            "python",
            "-m",
            "pytest",
            "--maxfail=1",
            "--junitxml=elsewhere.xml",
            "-pplugin_name",
            "a.py::test_a",
            "b.py::test_b",
        ),
        "8.3.5",
    )

    assert direct == module
    assert direct is not None
    assert direct.scope == "pytest:run"
    assert direct.features == frozenset(
        {
            "operation:run",
            "policy:maxfail=1",
            "output:junit",
            "opaque:plugin:<ARG>",
            "work_item:targets:a.py",
            "work_item:targets:a.py::test_a",
            "work_item:targets:b.py",
            "work_item:targets:b.py::test_b",
        }
    )


def test_interpreter_uses_longest_invocation_and_fails_closed() -> None:
    value = {
        "schema": "tool-spec-v1",
        "tool": "git",
        "documented_version": "2.39.2",
        "invocations": [
            {"tokens": ["git"], "operation": "default"},
            {"tokens": ["git", "diff"], "operation": "diff"},
        ],
        "operations": [
            {"name": "default", "arguments": [], "positionals": None},
            {"name": "diff", "arguments": [], "positionals": None},
        ],
        "relations": [],
    }
    spec = validate_tool_spec(value)
    assert spec is not None
    parsed = interpret_argv(spec, "git", ("/usr/bin/git", "diff"), "2.39.2")
    assert parsed is not None and parsed.scope == "git:diff"
    assert interpret_argv(spec, "git", ("git", "status"), "2.39.2") is None
    assert interpret_argv(spec, "git", ("git", "diff"), "2.40.0") is None

    pytest_spec = validate_tool_spec(_pytest_value())
    assert pytest_spec is not None
    assert (
        interpret_argv(
            pytest_spec,
            "pytest",
            ("pytest", "--unknown-plugin-option"),
            "8.3.5",
        )
        is None
    )
    assert interpret_argv(
        pytest_spec,
        "pytest",
        ("pytest", "--dist", "load"),
        "8.3.5",
    ) is None
    assert interpret_argv(
        pytest_spec,
        "pytest",
        ("pytest", "-n2", "--collect-only"),
        "8.3.5",
    ) is None
    assert interpret_argv(
        pytest_spec,
        "pytest",
        ("pytest", "--maxfail", "--collect-only"),
        "8.3.5",
    ) is None
    assert interpret_argv(
        pytest_spec,
        "pytest",
        ("pytest", 123),
        "8.3.5",
    ) is None

    canonical_exclusion = _pytest_value()
    canonical_exclusion["relations"].append(
        {
            "kind": "excludes",
            "operation": "run",
            "argument": "collect",
            "other": "maxfail",
        }
    )
    exclusion_spec = validate_tool_spec(canonical_exclusion)
    assert exclusion_spec is not None
    assert interpret_argv(
        exclusion_spec,
        "pytest",
        ("pytest", "-x", "--collect-only"),
        "8.3.5",
    ) is None


def test_validator_rejects_ambiguous_or_unbounded_specs() -> None:
    valid = _pytest_value()
    assert tool_spec_schema()["additionalProperties"] is False

    extra = deepcopy(valid)
    extra["regex"] = ".*"
    duplicate_form = deepcopy(valid)
    duplicate_form["operations"][0]["arguments"][1]["forms"] = ["-x", "--maxfail"]
    dangling = deepcopy(valid)
    dangling["relations"][0]["argument"] = "missing"
    conflict = deepcopy(valid)
    conflict["relations"].append(
        {
            "kind": "excludes",
            "operation": "run",
            "argument": "dist",
            "other": "workers",
        }
    )
    too_many = deepcopy(valid)
    too_many["invocations"] = [
        {"tokens": [f"pytest{index}"], "operation": "run"} for index in range(33)
    ]
    too_many_invocation_tokens = deepcopy(valid)
    too_many_invocation_tokens["invocations"][0]["tokens"] = [
        f"token{index}" for index in range(9)
    ]
    too_many_operations = deepcopy(valid)
    too_many_operations["operations"].extend(
        {"name": f"extra{index}", "arguments": [], "positionals": None}
        for index in range(32)
    )
    long_literal = deepcopy(valid)
    long_literal["tool"] = "x" * 257

    duplicate_invocation = deepcopy(valid)
    duplicate_invocation["invocations"].append(
        deepcopy(duplicate_invocation["invocations"][0])
    )
    too_many_arguments = deepcopy(valid)
    template_argument = too_many_arguments["operations"][0]["arguments"][0]
    too_many_arguments["operations"][0]["arguments"] = [
        {
            **template_argument,
            "id": f"arg{index}",
            "forms": [f"--arg{index}"],
        }
        for index in range(129)
    ]
    too_many_forms = deepcopy(valid)
    too_many_forms["operations"][0]["arguments"][0]["forms"] = [
        f"--form{index}" for index in range(9)
    ]
    too_many_relations = deepcopy(valid)
    too_many_relations["relations"] = [
        {
            "kind": "scope_order",
            "operation": "run",
            "argument": "targets",
            "delimiter": f"d{index}",
        }
        for index in range(257)
    ]
    too_many_positionals = deepcopy(valid)
    too_many_positionals["operations"][0]["positionals"]["max_items"] = 65

    oversized = deepcopy(valid)
    oversized["operations"][0]["arguments"] = [
        {
            "id": f"arg{index}",
            "forms": [
                f"--{index}-{form}-" + "x" * 80 for form in range(8)
            ],
            "arity": 0,
            "role": "opaque",
            "repeatable": False,
        }
        for index in range(128)
    ]

    option_scope = deepcopy(valid)
    option_scope["relations"].append(
        {
            "kind": "scope_order",
            "operation": "run",
            "argument": "maxfail",
            "delimiter": ".",
        }
    )
    cross_role_fixed = deepcopy(valid)
    cross_role_fixed["relations"].append(
        {
            "kind": "fixed_value_equivalence",
            "operation": "run",
            "argument": "plugin",
            "other": "maxfail",
            "value": "1",
        }
    )

    for invalid in (
        extra,
        duplicate_form,
        dangling,
        conflict,
        too_many,
        too_many_invocation_tokens,
        too_many_operations,
        long_literal,
        duplicate_invocation,
        too_many_arguments,
        too_many_forms,
        too_many_relations,
        too_many_positionals,
        oversized,
        option_scope,
        cross_role_fixed,
    ):
        assert validate_tool_spec(invalid) is None


def test_interpreter_bounds_scope_expansion() -> None:
    spec = validate_tool_spec(_pytest_value())
    assert spec is not None
    deeply_scoped = "::".join(["part"] * 17)
    assert interpret_argv(
        spec,
        "pytest",
        ("pytest", deeply_scoped),
        "8.3.5",
    ) is None
    assert interpret_argv(
        spec,
        "pytest",
        ("pytest", *(item for _ in range(128) for item in ("-p", "plugin"))),
        "8.3.5",
    ) is None
    assert interpret_argv(
        spec,
        "pytest",
        ("pytest", "\udcff"),
        "8.3.5",
    ) is None


def test_validator_rejects_fixed_equivalence_cycles() -> None:
    value = _pytest_value()
    value["operations"][0]["arguments"].extend(
        [
            {
                "id": name,
                "forms": [f"--{name}"],
                "arity": 1,
                "role": "execution_policy",
                "repeatable": False,
            }
            for name in ("left", "right")
        ]
    )
    value["relations"].extend(
        [
            {
                "kind": "fixed_value_equivalence",
                "operation": "run",
                "argument": "left",
                "other": "right",
                "value": "1",
            },
            {
                "kind": "fixed_value_equivalence",
                "operation": "run",
                "argument": "right",
                "other": "left",
                "value": "1",
            },
        ]
    )
    assert validate_tool_spec(value) is None
