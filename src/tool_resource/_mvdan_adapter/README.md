# Mvdan clause adapter

This JSONL service is the static parser behind
`tool_resource.clause_parser.parse_command_clauses`. It uses
`mvdan.cc/sh/v3` v3.13.1 and reports byte offsets; the Python client converts
them to Python string indices.

Build it with the bundled script, which needs no repository context:

```sh
src/tool_resource/_mvdan_adapter/build.sh
```

`tool_resource.mvdan_client.ensure_compatible_adapter()` runs this script
automatically when the cached binary is missing or stale, so the directory
stays self-contained: copy `src/tool_resource/` anywhere and the parser still
builds.

The script uses exactly Go 1.26.1. If that version is not on `PATH`, it
downloads the official Linux amd64 archive into
`${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/go-toolchain-1.26.1`.
The protocol-3 adapter binary is
`${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/mvdan-clause-adapter-protocol-3-mvdan-v3.13.1`.
The formal clause launcher validates its protocol, required capabilities, and
parser version before task setup, and atomically rebuilds a missing or stale
cache entry with the pinned script. No toolchain or binary is written into the
checkout.

Clause policy:

- Leading shell assignments are environment, not `argv`; `FOO=x cmd` has head
  `cmd`.
- Executable wrappers stay visible: `env FOO=x cmd` has head `env`, and
  `timeout 5 cmd` has head `timeout`.
- Assignment-only, redirect-only, and arithmetic-only statements have no
  executable head and emit no clause. Declaration builtins such as `export`
  are executable clauses.
- A heredoc clause span covers its command header through the last redirect
  delimiter, excluding every heredoc body.
- Each cooked `argv` word also carries its source span, quote/escape state, and
  literal, parameter, command-substitution, arithmetic, process-substitution,
  or pathname-expansion components. The cooked `argv` remains the prediction
  and Runtime KB identity.
- Each clause carries its position-free structural ancestor context. Repeated
  clauses may exchange runtime ownership only when that context proves their
  consumers equivalent.
