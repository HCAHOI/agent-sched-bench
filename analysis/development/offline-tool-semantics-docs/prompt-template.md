You are compiling one deployed command-line tool's official `--help` text into
the supplied `tool-spec-v1` JSON schema. This is an offline documentation task.
You have no access to traces, repositories, task IDs, outcomes, resource labels,
or runtime tools. Return only one JSON object accepted by the schema.

Inputs:

- tool: `{{TOOL}}`
- exact documented version: `{{VERSION}}`
- official help text follows after `DOCUMENTATION`

Rules:

1. Use only facts explicitly supported by the help text or by the source command
   printed in its header. Do not guess undocumented options or subcommands.
2. An invocation is the literal executable prefix that selects an operation.
   Keep documented aliases as separate invocations for the same operation, such
   as a direct executable and `python -m ...`, or `git` plus a subcommand.
3. An argument represents one documented option. List all documented spellings
   in `forms`; `arity` is exactly 0 or 1; and `repeatable` is true only when the
   documentation permits repetition.
4. Assign the narrowest documented role:
   - `work_item`: requested files, packages, targets, revisions, or other units
     whose amount or identity changes the requested work;
   - `work_selector`: filters that select a subset of work;
   - `execution_policy`: concurrency, stopping, cache, isolation, or execution
     mode controls;
   - `output`: formatting, verbosity, display, or output destination controls;
   - `opaque`: documented options that do not safely fit the other roles.
5. Use one positional group per operation. Use the schema maximum when the help
   documents an unbounded list. Do not invent positionals for omitted syntax.
6. Add a relation only when the documentation supports it:
   - `unordered_collection`: order of repeated values or positionals is
     semantically irrelevant;
   - `fixed_value_equivalence`: one flag means another argument has one fixed
     value; `argument` is the alias, `other` is the canonical argument;
   - `scope_order`: a literal delimiter defines nested scopes, such as `::`;
   - `requires` and `excludes`: documented presence constraints.
7. Prefer the primary workload and execution surface when the schema's fixed
   bounds prevent exhaustive coverage. The help text's own common/primary list
   determines priority; no workload-specific priority exists.
8. Never emit regexes, code, shell snippets, predictions, thresholds, resource
   classes, scheduling actions, package-specific facts, repository names, or
   values learned from examples beyond documented syntax.
9. Set `documented_version` to the exact input version. Invalid or incomplete
   output will be marked unsupported; there is no repair call.

DOCUMENTATION

{{DOCUMENTATION}}
