"""CLI entry point for trace collection, simulation, and viewer helpers."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from llm_call import add_llm_config_arguments, resolve_llm_config
from llm_call.config import (
    nonnegative_float_arg,
    nonnegative_int_arg,
    positive_float_arg,
    positive_int_arg,
    top_p_arg,
)
from trace_collect.monitoring import MONITORING_CHOICES


def parse_collect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect benchmark agent traces using a cloud LLM API.",
    )
    add_llm_config_arguments(parser)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum agent iterations per task.",
    )
    parser.add_argument(
        "--temperature",
        type=nonnegative_float_arg,
        default=None,
        help=(
            "Optional agent sampling temperature. When omitted, the scaffold "
            "default is used."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=top_p_arg,
        default=None,
        help="Optional agent nucleus sampling top_p value.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int_arg,
        default=None,
        help="Optional agent top_k sampling value for compatible providers.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=positive_float_arg,
        default=None,
        help="Optional agent repetition penalty for compatible providers.",
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
        type=nonnegative_int_arg,
        default=None,
        help="Only run the first N tasks after filtering and skipping (for testing).",
    )
    parser.add_argument(
        "--skip",
        type=nonnegative_int_arg,
        default=0,
        help="Skip the first N tasks after --instance-ids filtering and before --sample.",
    )
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated list of instance IDs to run (e.g., 'django__django-12345,sympy__sympy-67890').",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int_arg,
        default=1,
        help="Maximum concurrent collect tasks.",
    )
    parser.add_argument(
        "--scaffold",
        choices=["openclaw"],
        default="openclaw",
        help="Agent scaffold to use.",
    )
    parser.add_argument(
        "--container",
        choices=["docker", "podman"],
        default=None,
        help="Container CLI executable for benchmark collection runtime.",
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
            "Optional prompt template override; resolved as "
            "configs/prompts/<benchmark_slug>/<name>.md (hyphens converted to underscores). "
            "When omitted, uses the benchmark config default "
            "(e.g. swe-rebench -> cc_aligned, terminal-bench -> default)."
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
        description="Replay cloud-model traces using source-trace timing.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "YAML simulate manifest. It may be a list of absolute trace paths "
            "or an object with defaults.task_source and traces entries."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["cloud_model"],
        default="cloud_model",
        help="Replay mode. Only cloud_model is supported on this branch.",
    )
    parser.add_argument(
        "--concurrency",
        default="1",
        help=(
            "Maximum active traces. Use a comma-separated list such as 1,2,4,8 "
            "to run a throughput sweep."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int_arg,
        default=1,
        help=(
            "Number of OS worker processes for cloud_model replay. Each worker "
            "runs an independent asyncio event loop to reduce sleep wake-up "
            "drift at high concurrency. Default 1 preserves the legacy path."
        ),
    )
    parser.add_argument(
        "--prep-concurrency",
        type=nonnegative_int_arg,
        default=0,
        help=(
            "System-wide concurrent container preparation limit shared across "
            "simulate workers. 0 preserves the default limit of 20."
        ),
    )
    parser.add_argument(
        "--resource-monitoring",
        choices=MONITORING_CHOICES,
        default="auto",
        help="Built-in simulate resource monitoring policy (default: auto).",
    )
    parser.add_argument(
        "--pmu-monitoring",
        choices=MONITORING_CHOICES,
        default="auto",
        help=(
            "PMU/cgroup memory-access monitoring policy. Auto enables it only "
            "for non-concurrent container replay."
        ),
    )
    parser.add_argument(
        "--memory-bandwidth-monitoring",
        choices=MONITORING_CHOICES,
        default="auto",
        help=(
            "Host memory-bandwidth monitoring policy. Auto enables it only for "
            "non-concurrent container replay."
        ),
    )
    parser.add_argument(
        "--task-source",
        default="data/swe-rebench/tasks.json",
        help="Path to tasks JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="traces/simulate",
        help="Output directory for the simulate trace.",
    )
    parser.add_argument(
        "--container",
        default=None,
        choices=["docker", "podman"],
        help="Container executable for container-mode trace replay.",
    )
    parser.add_argument(
        "--network-mode",
        default="host",
        help="Container network mode (default: host). Use 'none' for isolated replay.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=600.0,
        help=(
            "Fallback timeout in seconds for replayed shell commands when the "
            "source trace does not carry a tool-specific timeout."
        ),
    )
    parser.add_argument(
        "--warmup-skip-iterations",
        type=int,
        default=0,
        help=(
            "Tag the first N replay iterations with sim_metrics.warmup=true "
            "for analysis-time exclusion. Iterations are still replayed."
        ),
    )
    parser.add_argument(
        "--replay-speed",
        type=positive_float_arg,
        default=1.0,
        help=(
            "Wall-clock acceleration factor for source inter-action gaps and "
            "source-scaled action durations. Example: --replay-speed 50 "
            "replays source timing at 50x."
        ),
    )
    parser.add_argument(
        "--llm-timing",
        choices=["source-scaled", "ttft-tpot"],
        default="source-scaled",
        help=(
            "LLM replay duration model. source-scaled sleeps for source LLM "
            "duration divided by --replay-speed. ttft-tpot sleeps for "
            "--llm-ttft-ms + (completion_tokens - 1) * --llm-tpot-ms."
        ),
    )
    parser.add_argument(
        "--llm-ttft-ms",
        type=nonnegative_float_arg,
        default=None,
        help="Simulated TTFT in milliseconds when --llm-timing ttft-tpot.",
    )
    parser.add_argument(
        "--llm-tpot-ms",
        type=nonnegative_float_arg,
        default=None,
        help="Simulated TPOT in milliseconds when --llm-timing ttft-tpot.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)

def main() -> None:
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub == "simulate":
        _run_simulate(parse_simulate_args(sys.argv[2:]))
    elif sub == "gantt-serve":
        from demo.gantt_viewer.backend.dev import main as run_gantt_server

        run_gantt_server(sys.argv[2:])
    elif sub == "gantt-export":
        from demo.gantt_viewer.backend.static_export import (
            build_parser as build_gantt_export_parser,
            export_from_args,
        )

        result = export_from_args(build_gantt_export_parser().parse_args(sys.argv[2:]))
        print(json.dumps(result, indent=2, sort_keys=True))
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
    if args.mcp_config is None:
        print(
            "ERROR: MCP config is required for openclaw; pass "
            "--mcp-config configs/mcp/context7.yaml or --mcp-config none "
            "to acknowledge running without MCP",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        provider_config = resolve_llm_config(
            provider=args.provider,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            environ=os.environ,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
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
            container_executable=args.container,
            provider_name=provider_config.name,
            env_key=provider_config.env_key,
            api_base=provider_config.api_base,
            api_key=provider_config.api_key,
            model=provider_config.model,
            benchmark=benchmark,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            sample=args.sample,
            skip=args.skip,
            concurrency=args.concurrency,
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

def _parse_concurrency_values(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--concurrency must be a positive integer or comma-separated list")
    values: list[int] = []
    for part in parts:
        try:
            concurrency = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid --concurrency value: {part!r}") from exc
        if concurrency < 1:
            raise ValueError("--concurrency values must be >= 1")
        values.append(concurrency)
    return values


def _append_throughput_sweep_record(sweep_path: Path, trace_file: Path) -> None:
    summary_path = trace_file.with_name(f"{trace_file.stem}.throughput_summary.json")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with sweep_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _run_simulate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from trace_collect.simulator import simulate

    try:
        concurrency_values = _parse_concurrency_values(args.concurrency)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    simulate_kwargs = {
        "manifest": Path(args.manifest),
        "task_source": Path(args.task_source),
        "output_dir": Path(args.output_dir),
        "mode": args.mode,
        "container_executable": args.container,
        "network_mode": args.network_mode,
        "workers": args.workers,
        "prep_concurrency": args.prep_concurrency,
        "resource_monitoring": args.resource_monitoring,
        "pmu_monitoring": args.pmu_monitoring,
        "memory_bandwidth_monitoring": args.memory_bandwidth_monitoring,
        "command_timeout_s": args.command_timeout,
        "warmup_skip_iterations": args.warmup_skip_iterations,
        "replay_speed": args.replay_speed,
        "llm_timing_mode": args.llm_timing.replace("-", "_"),
        "llm_ttft_ms": args.llm_ttft_ms,
        "llm_tpot_ms": args.llm_tpot_ms,
        "structured_output": args.output_dir == "traces/simulate",
    }

    sweep_path = Path(args.output_dir) / "throughput_sweep.jsonl"
    if len(concurrency_values) > 1 and sweep_path.exists():
        sweep_path.unlink()
    for concurrency in concurrency_values:
        trace_file = asyncio.run(simulate(**simulate_kwargs, concurrency=concurrency))
        print(f"Simulate trace written to: {trace_file}")
        if len(concurrency_values) > 1:
            _append_throughput_sweep_record(sweep_path, trace_file)
    if len(concurrency_values) > 1:
        print(f"Throughput sweep written to: {sweep_path}")

if __name__ == "__main__":
    main()
