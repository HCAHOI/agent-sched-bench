"""CLI entry point for trace collection, import, replay, and viewer helpers."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from trace_collect.provider_presets import (
    provider_choices,
    resolve_provider_config,
)

def parse_collect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SWE-Bench agent traces using an external LLM API.",
    )
    parser.add_argument(
        "--provider",
        choices=provider_choices(),
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
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum agent iterations per task.",
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
        choices=["miniswe", "openclaw"],
        default="miniswe",
        help="Agent scaffold to use: miniswe (bash-only) or openclaw (structured tools).",
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
            "explicit 'none' from a legacy MCP-less default."
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256_000,
        help="Sliding window token budget for context management.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help=(
            "Optional prompt template override under configs/prompts/swe_rebench/. "
            "When omitted, uses the benchmark config default "
            "(e.g. swe-rebench -> cc_aligned)."
        ),
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=30.0,
        help="Abort per-task run if free disk falls below this threshold (GB).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume an interrupted run by passing its existing run directory path.",
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
            "opt-out path used for local runs without a vLLM server."
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
            "Default 0 (no warmup tagging). Opt in only when first-iteration "
            "latency variance is empirically >20%% vs steady-state."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)

def parse_import_claude_code_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Claude Code session JSONL to a canonical trace for the Gantt "
            "viewer. Post-hoc, read-only — no collection, no simulation. "
            "Rich Claude Code fields (cache tokens, thinking blocks, "
            "toolUseResult sidecar) are backfilled into additive data.* and "
            "metadata.run_config.* slots in the canonical trace schema."
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
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub == "simulate":
        _run_simulate(parse_simulate_args(sys.argv[2:]))
    elif sub == "import-claude-code":
        _run_import_claude_code(parse_import_claude_code_args(sys.argv[2:]))
    elif sub == "inspect":
        _run_inspect(sys.argv[2:])
    elif sub == "gantt-serve":
        from demo.gantt_viewer.backend.dev import main as run_gantt_server

        run_gantt_server(sys.argv[2:])
    else:
        _run_collect(parse_collect_args())

REPO_ROOT = Path(__file__).resolve().parents[2]

def _run_collect(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --mcp-config is MANDATORY for openclaw runs: a forgotten flag would
    # silently produce an MCP-less trace. Opt-out is the literal "none".
    if args.scaffold == "openclaw" and args.mcp_config is None:
        print(
            "ERROR: MCP config is required for openclaw; pass "
            "--mcp-config configs/mcp/context7.yaml or --mcp-config none "
            "to acknowledge running without MCP",
            file=sys.stderr,
        )
        sys.exit(2)

    provider_config = resolve_provider_config(
        provider=args.provider,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        environ=os.environ,
    )
    if not provider_config.api_key:
        print(
            f"ERROR: Set {provider_config.env_key} or pass --api-key.",
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
            scaffold=args.scaffold,
            provider_name=provider_config.name,
            api_base=provider_config.api_base,
            api_key=provider_config.api_key,
            model=provider_config.model,
            benchmark=benchmark,
            max_iterations=args.max_iterations,
            command_timeout_s=args.command_timeout,
            task_timeout_s=args.task_timeout,
            sample=args.sample,
            instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
            run_id=args.run_id,
            max_context_tokens=args.max_context_tokens,
            mcp_config=args.mcp_config,
            prompt_template=args.prompt_template,
            min_free_disk_gb=args.min_free_disk_gb,
        )
    )
    print(f"Traces written to: {run_dir}/")
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        print(f"Results written to: {results_path}")

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

def _run_import_claude_code(args: argparse.Namespace) -> None:
    """Convert a Claude Code session JSONL into a canonical trace for the Gantt viewer."""
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
        description="Inspect an OpenClaw / miniswe JSONL trace file.",
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

if __name__ == "__main__":
    main()
