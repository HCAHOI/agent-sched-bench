# Trace Tool Analysis Plan

## Scope

- Analyze the 10 Terminal-Bench traces listed in `demo/gantt_viewer/configs/terminal-bench-10.yaml`.
- Use real trace records only; do not synthesize or impute missing tool calls.
- Produce call-level and aggregate summaries for tool execution duration, output length, and tool type relationships.

## Metrics

- Tool call identity: task, trace path, iteration, action id, tool name.
- Timing: `ts_start`, `ts_end`, `duration_s`.
- Output size: character length, byte length, line count.
- Input size: argument character length where available.
- Outcome fields when present: `success`, missing/incomplete span flags.

## Artifacts

- `results/trace_tool_analysis/tool_calls.csv`
- `results/trace_tool_analysis/summary.json`
- `results/trace_tool_analysis/summary.md`
- Optional figures if the environment has plotting dependencies available.

## Checks

- Confirm all 10 configured traces are loaded.
- Report incomplete `tool_exec_start` events separately from completed `tool_exec` actions.
- Aggregate both per-call and per-task views.
