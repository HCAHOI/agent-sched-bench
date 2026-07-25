# Mvdan clause adapter

This JSONL service is the static parser behind
`tool_resource.features.parse_command_clauses`. It uses
`mvdan.cc/sh/v3` v3.13.1 and reports byte offsets; the Python client converts
them to Python string indices.

Build it from the repository root:

```sh
scripts/setup/build_mvdan_adapter.sh
```

The script uses exactly Go 1.26.1. If that version is not on `PATH`, it
downloads the official Linux amd64 archive into
`${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/go-toolchain-1.26.1`.
The adapter binary is
`${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/mvdan-clause-adapter-v3.13.1`.
No toolchain or binary is written into the checkout.

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
