"""CLI entry point for trace collection and simulation.

Usage (collect):
    OPENROUTER_API_KEY=sk-xxx python -m trace_collect.cli \\
        --model qwen/qwen3.6-plus:free \\
        --max-steps 50 \\
        --sample 5

Usage (simulate):
    python -m trace_collect.cli simulate \\
        --source-trace traces/swebench/qwen-plus/.../task.jsonl \\
        --api-base http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct

Usage (import OpenClaw):
    python -m trace_collect.cli import-openclaw \\
        --results /path/to/nanobot/results.jsonl \\
        --model-name Qwen3.6-Plus

Usage (inspect):
    python -m trace_collect.cli inspect <trace.jsonl> overview
    python -m trace_collect.cli inspect <trace.jsonl> step 5 [--json]
    python -m trace_collect.cli inspect <trace.jsonl> search "pattern"

Usage (gantt server):
    python -m trace_collect.cli gantt-serve --config demo/gantt_viewer/configs/example.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


_PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "qwen/qwen3.6-plus:free",
    },
    "dashscope": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus-latest",
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
}


def parse_collect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SWE-Bench agent traces using an external LLM API.",
    )
    parser.add_argument(
        "--provider",
        choices=list(_PROVIDERS.keys()),
        default="openrouter",
        help="LLM provider preset (default: openrouter). Sets api-base, api-key env var, and default model.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (default: from --provider).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override API key (default: from provider's env var).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from --provider).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum agent steps per task.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds per bash command.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds per task overall.",
    )
    parser.add_argument(
        "--benchmark",
        default="swe-bench-verified",
        help=(
            "Benchmark slug (e.g. 'swe-bench-verified', 'swe-rebench'). "
            "Loads configs/benchmarks/<slug>.yaml and constructs the plugin."
        ),
    )
    parser.add_argument(
        "--task-source",
        default=None,
        help=(
            "Optional override for the tasks JSON file. Defaults to "
            "<benchmark.data_root>/tasks.json from the benchmark YAML."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N tasks (for testing).",
    )
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated list of instance IDs to run (e.g., 'django__django-12345,sympy__sympy-67890').",
    )
    parser.add_argument(
        "--scaffold",
        choices=["mini-swe-agent", "openclaw"],
        default="mini-swe-agent",
        help="Agent scaffold to use: mini-swe-agent (bash-only) or openclaw (structured tools).",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "MCP server configuration. Required when --scaffold=openclaw. "
            "Accepts a YAML path (e.g. configs/mcp/context7.yaml) OR the "
            "literal string 'none' for an affirmative MCP-less run. The "
            "trace header records the chosen value under "
            "metadata.run_config.mcp_config so analysis can distinguish "
            "explicit 'none' from a legacy MCP-less default. Phase 4 of "
            "trace-sim-vastai-pipeline."
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256_000,
        help="Sliding window token budget for context management.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume an interrupted run by providing its existing run ID.",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the official SWE-bench harness on generated predictions.",
    )
    parser.add_argument(
        "--harness-workers",
        type=int,
        default=1,
        help="Official harness max_workers; keep 1 for serial evaluation.",
    )
    parser.add_argument(
        "--harness-timeout",
        type=int,
        default=1800,
        help="Official harness timeout per task in seconds.",
    )
    parser.add_argument(
        "--harness-run-id",
        default=None,
        help="Optional explicit run id for the official harness.",
    )
    parser.add_argument(
        "--harness-report-dir",
        default=None,
        help="Directory where official harness logs/reports should be written.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def parse_simulate_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a trace with local model timing (TTFT/TPOT).",
    )
    parser.add_argument(
        "--source-trace",
        required=True,
        help="Path to the source API trace JSONL file.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="Local model OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local model name (e.g. Qwen/Qwen2.5-Coder-7B-Instruct).",
    )
    parser.add_argument(
        "--task-source",
        default="data/swebench_verified/tasks.json",
        help="Path to tasks JSON file.",
    )
    parser.add_argument(
        "--repos-root",
        default="data/swebench_repos",
        help="Path to pre-cloned repos directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="traces/simulate",
        help="Output directory for the simulate trace.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds per bash command.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=1200.0,
        help="Timeout for the entire simulation.",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help=(
            "vLLM Prometheus /metrics endpoint URL. When set, the simulator "
            "snapshots scheduler metrics (PreemptionSnapshot) per iteration "
            "and stores them under TraceAction.data.sim_metrics. When unset, "
            "the simulator records empty (all-None) snapshots — the explicit "
            "opt-out path used for local runs without a vLLM server. "
            "Phase 2 of trace-sim-vastai-pipeline."
        ),
    )
    parser.add_argument(
        "--warmup-skip-iterations",
        type=int,
        default=0,
        help=(
            "Tag the first N replay iterations with sim_metrics.warmup=true "
            "for analysis-time exclusion. Iterations are still measured at "
            "collection time; the flag controls analysis treatment only. "
            "Default 0 (no warmup tagging) per CLAUDE.md No Unjustified "
            "Complexity. Opt in only when an empirical probe shows "
            "first-iteration latency variance >20%% vs steady-state — see "
            ".omc/plans/phase1.5-design.md Q4 deferral. Phase 1.5.1 of "
            "trace-sim-vastai-pipeline."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def parse_import_openclaw_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import nanobot/OpenClaw traces into the benchmark run layout.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to nanobot results.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="traces",
        help="Root output directory for imported benchmark-compatible traces.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen3.6-Plus",
        help="Recorded model name stored in preds.json for imported traces.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run directory suffix.",
    )
    return parser.parse_args(argv)


def parse_import_claude_code_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Claude Code session JSONL to a v5 trace for the Gantt "
            "viewer. Post-hoc, read-only — no collection, no simulation. "
            "Rich Claude Code fields (cache tokens, thinking blocks, "
            "toolUseResult sidecar) are backfilled into additive data.* and "
            "metadata.run_config.* slots per the v5 extension convention."
        ),
    )
    parser.add_argument(
        "--session",
        required=True,
        help=(
            "Path to the Claude Code session JSONL "
            "(typically ~/.claude/projects/<slug>/<session-uuid>.jsonl)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="traces",
        help=(
            "Root output directory. Final file lands at "
            "<output-dir>/claude-code-import/<session-uuid>/<session-uuid>.jsonl."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional explicit run directory suffix (default: session uuid)."
        ),
    )
    parser.add_argument(
        "--no-sidechains",
        dest="include_sidechains",
        action="store_false",
        default=True,
        help=(
            "Skip folding <session-dir>/<session-uuid>/subagents/agent-*.jsonl "
            "into the output. Default: include them as distinct agent_id lanes."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    # Keyword detection: subcommand as first arg routes to the right parser.
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        args = parse_simulate_args(sys.argv[2:])
        _run_simulate(args)
    elif len(sys.argv) > 1 and sys.argv[1] == "import-openclaw":
        args = parse_import_openclaw_args(sys.argv[2:])
        _run_import_openclaw(args)
    elif len(sys.argv) > 1 and sys.argv[1] == "import-claude-code":
        args = parse_import_claude_code_args(sys.argv[2:])
        _run_import_claude_code(args)
    elif len(sys.argv) > 1 and sys.argv[1] == "inspect":
        _run_inspect(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "gantt-serve":
        _run_gantt_serve(sys.argv[2:])
    else:
        args = parse_collect_args()
        _run_collect(args)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_collect(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Phase 4 of trace-sim-vastai-pipeline: --mcp-config is MANDATORY for
    # openclaw runs. Driver 3 ("OpenClaw realism delta") is violated if a
    # forgotten flag silently produces an MCP-less openclaw trace, so the
    # CLI refuses to start. Opt-out is the affirmative literal "none".
    if args.scaffold == "openclaw" and args.mcp_config is None:
        print(
            "ERROR: MCP config is required for openclaw; pass "
            "--mcp-config configs/mcp/context7.yaml or --mcp-config none "
            "to acknowledge running without MCP",
            file=sys.stderr,
        )
        sys.exit(2)

    preset = _PROVIDERS[args.provider]
    api_base = args.api_base or preset["api_base"]
    api_key = args.api_key or os.environ.get(preset["env_key"])
    model = args.model or preset["default_model"]
    if not api_key:
        print(
            f"ERROR: Set {preset['env_key']} or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    from agents.benchmarks import get_benchmark_class
    from agents.benchmarks.base import BenchmarkConfig
    from trace_collect.collector import collect_traces

    benchmark_yaml = REPO_ROOT / "configs" / "benchmarks" / f"{args.benchmark}.yaml"
    if not benchmark_yaml.exists():
        print(f"ERROR: No benchmark config at {benchmark_yaml}", file=sys.stderr)
        sys.exit(1)
    config = BenchmarkConfig.from_yaml(benchmark_yaml)
    plugin_cls = get_benchmark_class(config.slug)
    benchmark = plugin_cls(config)

    run_dir = asyncio.run(
        collect_traces(
            api_base=api_base,
            api_key=api_key,
            model=model,
            benchmark=benchmark,
            task_source=args.task_source if args.task_source else None,
            max_steps=args.max_steps,
            command_timeout_s=args.command_timeout,
            task_timeout_s=args.task_timeout,
            sample=args.sample,
            instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
            scaffold=args.scaffold,
            run_id=args.run_id,
            max_context_tokens=args.max_context_tokens,
            evaluate=args.evaluate,
            harness_max_workers=args.harness_workers,
            harness_timeout=args.harness_timeout,
            harness_run_id=args.harness_run_id,
            harness_report_dir=args.harness_report_dir,
            mcp_config=args.mcp_config,
        )
    )
    print(f"Traces written to: {run_dir}/")
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        print(f"Results written to: {results_path}")
    predictions_path = run_dir / "preds.json"
    if predictions_path.exists():
        print(f"Predictions written to: {predictions_path}")


def _run_simulate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "VLLM_API_KEY", "EMPTY"
    )

    from trace_collect.simulator import simulate

    trace_file = asyncio.run(
        simulate(
            source_trace=Path(args.source_trace),
            task_source=Path(args.task_source),
            repos_root=Path(args.repos_root),
            output_dir=Path(args.output_dir),
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            command_timeout_s=args.command_timeout,
            task_timeout_s=args.task_timeout,
            metrics_url=args.metrics_url,
            warmup_skip_iterations=args.warmup_skip_iterations,
        )
    )
    print(f"Simulate trace written to: {trace_file}")


def _run_import_openclaw(args: argparse.Namespace) -> None:
    from trace_collect.openclaw_import import import_openclaw_run

    run_dir = import_openclaw_run(
        results_path=Path(args.results),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        run_id=args.run_id,
    )
    print(f"Imported OpenClaw traces to: {run_dir}/")
    print(f"Results written to: {run_dir / 'results.jsonl'}")
    print(f"Predictions written to: {run_dir / 'preds.json'}")


def _run_import_claude_code(args: argparse.Namespace) -> None:
    """Convert a Claude Code session JSONL into a v5 trace for the Gantt viewer."""
    from trace_collect.claude_code_import import import_claude_code_session

    trace_file = import_claude_code_session(
        session_path=Path(args.session),
        output_dir=Path(args.output_dir),
        include_sidechains=args.include_sidechains,
        run_id=args.run_id,
    )
    print(f"Claude Code trace written to: {trace_file}")
    print(
        "Start the dynamic Gantt viewer with: "
        "python -m trace_collect.cli gantt-serve --dev"
    )


def _run_inspect(argv: list[str]) -> None:
    import argparse as _argparse
    from trace_collect.trace_inspector import (
        TraceData,
        cmd_overview,
        cmd_step,
        cmd_messages,
        cmd_response,
        cmd_events,
        cmd_tools,
        cmd_search,
        cmd_timeline,
    )

    parser = _argparse.ArgumentParser(
        prog="python -m trace_collect.cli inspect",
        description="Inspect an OpenClaw / mini-swe-agent JSONL trace file.",
        epilog="""commands:
  overview   Summary stats: steps, tokens, tool counts, elapsed time
  step N     Full details of step N (0-indexed): LLM stats, tool call, result
  messages N Show messages_in (prompt list) for step N
  response N Show raw_response (LLM output) for step N
  events     List fine-grained events (SCHEDULING, SESSION, TOOL, LLM, ...)
  tools      Tool usage breakdown: name, count, total duration, success rate
  search P   Regex search through llm_output fields across all steps
  timeline   Concise per-step timeline with icons, relative timestamps, durations

examples:
  %(prog)s trace.jsonl overview
  %(prog)s trace.jsonl step 3 --full
  %(prog)s trace.jsonl messages 0 --role user
  %(prog)s trace.jsonl response 5 --truncate 500
  %(prog)s trace.jsonl events --category SCHEDULING
  %(prog)s trace.jsonl events --category TOOL --iteration 2
  %(prog)s trace.jsonl tools
  %(prog)s trace.jsonl search "def main"
  %(prog)s trace.jsonl overview --json
  %(prog)s trace.jsonl step 0 --agent django
  %(prog)s trace.jsonl timeline
  %(prog)s trace.jsonl timeline --agent django""",
        formatter_class=_argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace", help="Path to the JSONL trace file.")
    parser.add_argument(
        "command",
        choices=[
            "overview",
            "step",
            "messages",
            "response",
            "events",
            "tools",
            "search",
            "timeline",
        ],
        help="Inspection command (see above).",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Command argument: step index (for step/messages/response) or regex pattern (for search).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON for machine consumption.",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=2000,
        help="Truncate long fields to N chars (default: 2000, 0=no truncation).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable truncation (show complete content).",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Filter records by agent_id substring.",
    )
    parser.add_argument(
        "--role",
        default=None,
        help="Filter messages by role (system/user/assistant/tool). Used with 'messages' command.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter events by category (SCHEDULING/SESSION/CONTEXT/TOOL/LLM/MCP/MEMORY/SUBAGENT). Used with 'events' command.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Filter events by iteration number. Used with 'events' command.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Filter tool stats by step index. Used with 'tools' command.",
    )
    parsed = parser.parse_args(argv)

    truncate = 0 if parsed.full else parsed.truncate
    data = TraceData.load(Path(parsed.trace), agent_filter=parsed.agent)

    def _parse_step_idx(args: list[str]) -> int:
        """Parse step index from CLI args, returning 0 as default."""
        if not args:
            return 0
        try:
            return int(args[0])
        except ValueError:
            parser.error(f"step index must be an integer, got: {args[0]!r}")

    cmd = parsed.command
    if cmd == "overview":
        cmd_overview(data, as_json=parsed.as_json)
    elif cmd == "step":
        step_n = _parse_step_idx(parsed.args)
        cmd_step(data, step_n, truncate=truncate, as_json=parsed.as_json)
    elif cmd == "messages":
        step_n = _parse_step_idx(parsed.args)
        cmd_messages(
            data,
            step_n,
            role_filter=parsed.role,
            truncate=truncate,
            as_json=parsed.as_json,
        )
    elif cmd == "response":
        step_n = _parse_step_idx(parsed.args)
        cmd_response(data, step_n, truncate=truncate, as_json=parsed.as_json)
    elif cmd == "events":
        cmd_events(
            data,
            category=parsed.category,
            iteration=parsed.iteration,
            as_json=parsed.as_json,
        )
    elif cmd == "tools":
        cmd_tools(data, step_idx=parsed.step, as_json=parsed.as_json)
    elif cmd == "search":
        pattern = parsed.args[0] if parsed.args else ""
        cmd_search(data, pattern, truncate=truncate, as_json=parsed.as_json)
    elif cmd == "timeline":
        if parsed.as_json:
            print(json.dumps({"error": "timeline does not support --json output"}))
            return
        cmd_timeline(data)


def _run_gantt_serve(argv: list[str]) -> None:
    """Top-level dynamic Gantt viewer subcommand."""
    from demo.gantt_viewer.backend.dev import main as run_gantt_server

    run_gantt_server(argv)


if __name__ == "__main__":
    main()
