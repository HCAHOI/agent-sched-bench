"""CLI entry point for trace collection, import, replay, and viewer helpers."""

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
    positive_float_arg,
    positive_int_arg,
    top_p_arg,
)


def parse_collect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SWE-Bench agent traces using an external LLM API.",
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
        "--record-internals",
        action="store_true",
        help=(
            "Record per-call attention aggregates and MoE routing with a "
            "host-side HuggingFace SDPA backend. Forces OpenClaw model request "
            "concurrency to 1 for the run."
        ),
    )
    parser.add_argument(
        "--local-hf",
        action="store_true",
        help=(
            "Run the host-side HuggingFace backend, loading --model locally "
            "instead of calling the external --provider endpoint. With "
            "--kv-policy none, this uses plain full-KV DynamicCache plus the "
            "session prefix cache, avoiding KV-policy proxy overhead."
        ),
    )
    parser.add_argument(
        "--kv-policy",
        choices=[
            "none",
            "random",
            "streaming",
            "h2o",
            "metadata",
            "position_control",
            "null_eviction",
        ],
        default="none",
        help="KV cache eviction policy for the HF recording backend. See src/trace_collect/CLAUDE.md.",
    )
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=None,
        help="Per-layer KV budget (kept tokens). Required when --kv-policy != none.",
    )
    parser.add_argument(
        "--kv-sink-size",
        type=int,
        default=4,
        help=(
            "Sink-prefix length (head tokens kept) for `streaming` and `h2o`. "
            "Default 4 matches StreamingLLM's §3 ablation. Ignored by "
            "`random`."
        ),
    )
    parser.add_argument(
        "--kv-recent-window",
        type=int,
        default=256,
        help=(
            "Recent-window length (tail tokens kept) for `streaming` and "
            "`h2o`. Default 256. Ignored by `random`."
        ),
    )
    parser.add_argument(
        "--kv-aggregate",
        choices=["sum", "mean", "ema"],
        default="sum",
        help="H2O score aggregation: sum (default) | mean | ema (yaml-only decay). Ignored by random/streaming.",
    )
    parser.add_argument(
        "--kv-config",
        type=str,
        default=None,
        help="YAML under configs/kv_policies/ with EvictionPolicyConfig fields; CLI flags overlay yaml.",
    )
    parser.add_argument(
        "--kv-metadata-rung",
        choices=["rung1", "rung2", "rung3", "rung4"],
        default="rung4",
        help=(
            "Metadata-residency ablation rung. rung1=role/age/offset only; "
            "rung2 adds sink+recent reservation; rung3 adds recent tool-result "
            "reservation; rung4 adds tool error/exit-code reservation."
        ),
    )
    parser.add_argument(
        "--kv-position-control",
        choices=["random", "middle", "structured"],
        default="random",
        help=(
            "Control mode for --kv-policy position_control. Controls use no "
            "role or exit-code metadata and are later matched on Delta-pos."
        ),
    )
    parser.add_argument(
        "--kv-per-layer-table",
        action="store_true",
        help=(
            "Enable the pre-registered per-layer metadata table arm. Requires "
            "--kv-per-layer-table-path with a frozen P0 score table; global "
            "remains the default."
        ),
    )
    parser.add_argument(
        "--kv-per-layer-table-path",
        type=str,
        default=None,
        help=(
            "Path to the frozen P0 per-layer metadata score table used by "
            "--kv-per-layer-table. The table is required before any GPU "
            "campaign arm can be labeled per_layer_table=true."
        ),
    )
    parser.add_argument(
        "--kv-per-layer-budget",
        action="store_true",
        help=(
            "Enable the stretch per-layer budget arm. This is parsed for "
            "pre-registration but should not be used without a frozen P0 rule."
        ),
    )
    parser.add_argument(
        "--kv-record",
        choices=["on", "off"],
        default="on",
        help=(
            "Whether to write `kv_eviction.npz` recordings. Default `on` "
            "preserves the audit trail. `off` runs the policy but skips the "
            "per-call recorder allocation and npz write — used by the perf "
            "microbench to isolate eviction overhead from recording "
            "overhead. Meaningful only when --kv-policy != none."
        ),
    )
    parser.add_argument(
        "--sparse-attn",
        choices=[
            "none",
            "sliding",
            "streaming",
            "heavy_hitter",
            "block_topk",
            "quest",
            "metadata",
        ],
        default="none",
        help=(
            "Sparse attention method for the HF recording backend. Enforce mode is "
            "mutually exclusive with --kv-policy. Requires --record-internals. "
            "See src/trace_collect/CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--sparse-attn-sink-size",
        type=int,
        default=4,
        help=(
            "Sink-prefix length kept attended for `sliding`. Default 4. "
            "Ignored when --sparse-attn != sliding."
        ),
    )
    parser.add_argument(
        "--sparse-attn-recent-window",
        type=int,
        default=256,
        help=(
            "Recent-window length kept attended for `sliding`. Default 256. "
            "Ignored when --sparse-attn != sliding."
        ),
    )
    parser.add_argument(
        "--sparse-attn-config",
        type=str,
        default=None,
        help=(
            "Optional YAML file (e.g. configs/sparse_attention/sliding.yaml) "
            "carrying a flat map of SparseAttentionConfig fields. CLI flags "
            "overlay yaml using the same rules as --kv-config."
        ),
    )
    parser.add_argument(
        "--sparse-attn-record",
        choices=["on", "off"],
        default="on",
        help=(
            "Whether to write `sparse_attention.npz` recordings. Default "
            "`on`. `off` runs the method but skips the per-call recorder "
            "allocation and npz write. Meaningful only when --sparse-attn "
            "!= none."
        ),
    )
    parser.add_argument(
        "--sparse-attn-observe-only",
        action="store_true",
        help="Record sparse selection without enforcing it (compatible with --kv-policy).",
    )
    parser.add_argument(
        "--sparse-attn-budget",
        type=int,
        default=None,
        help=(
            "Token budget for dynamic sparse attention methods "
            "(`heavy_hitter`, `block_topk`, `quest`)."
        ),
    )
    parser.add_argument(
        "--sparse-attn-block-size",
        type=int,
        default=16,
        help="Block/page size for `block_topk` and `quest`. Default 16.",
    )
    parser.add_argument(
        "--sparse-attn-score-reduction",
        choices=["max", "mean", "vote"],
        default="max",
        help=(
            "How to reduce token scores into block/page scores. Default max. "
            "'vote' (block_topk only) ranks blocks by cross-head top-B votes."
        ),
    )
    parser.add_argument(
        "--sparse-attn-phase-scope",
        choices=["decode_only"],
        default="decode_only",
        help=(
            "Where dynamic methods enforce sparse masks. Only decode_only is "
            "currently supported; prefill remains dense causal."
        ),
    )
    parser.add_argument(
        "--sparse-attn-metadata-rung",
        choices=["rung1", "rung2", "rung3", "rung4"],
        default="rung4",
        help=(
            "Rung for the observe-only metadata sparse sidecar. Meaningful "
            "only with --sparse-attn metadata."
        ),
    )
    parser.add_argument(
        "--per-head-stats-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices or preset 'qwen3-coder-30b'/'default' for "
            "per-head head_span_* arrays. Requires --record-internals."
        ),
    )
    parser.add_argument(
        "--per-head-block-stats",
        action="store_true",
        help=(
            "Capture per-selected-block attention mean/std into block_span_* arrays "
            "(bucket axis = sink | selection rank 1..R_max | recent). "
            "Requires --record-internals, --sparse-attn block_topk, non-empty --per-head-stats-layers."
        ),
    )
    parser.add_argument(
        "--record-per-head-topk",
        action="store_true",
        help=(
            "Record per-head independent top-R block selections into per_head_topk_csr_* arrays. "
            "Requires --record-internals, --sparse-attn block_topk, non-empty --per-head-stats-layers."
        ),
    )
    parser.add_argument(
        "--per-head-topk-rank",
        type=int,
        default=64,
        help=(
            "Per-head rank cap R_ph for --record-per-head-topk. Default 64 "
            "(matches the typical middle-block budget). Lower it to cut storage."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Generation seed for the HF recording backend. Campaign scripts "
            "sweep this as the task x policy x beta x seed dimension."
        ),
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
        "--manifest",
        required=True,
        help=(
            "YAML simulate manifest. It may be a list of absolute trace paths "
            "or an object with defaults.task_source and traces entries."
        ),
    )
    add_llm_config_arguments(parser)
    parser.add_argument(
        "--mode",
        choices=["local_model", "cloud_model"],
        default="local_model",
        help=(
            "Simulation mode. local_model replays one trace through a local "
            "OpenAI-compatible model; cloud_model replays one or more traces "
            "using source-trace timing without issuing any LLM requests."
        ),
    )
    parser.add_argument(
        "--concurrency",
        default="1",
        help=(
            "Maximum active traces for cloud_model. Use a comma-separated list "
            "such as 1,2,4,8 to run a throughput sweep."
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
        "--replay-speed",
        type=float,
        default=1.0,
        help=(
            "Wall-clock acceleration factor for cloud_model replay. "
            "Example: --replay-speed 50 replays source timing at 50x."
        ),
    )
    parser.add_argument(
        "--gpu-tracking",
        choices=["on", "off"],
        default="off",
        help=(
            "Enable GPU memory tracking. When 'on', requires --metrics-url, "
            "--vllm-pid, and --vllm-startup-log. Forbidden in cloud_model mode."
        ),
    )
    parser.add_argument(
        "--gpu-sample-hz",
        type=float,
        default=10.0,
        help="GPU memory sampling rate in Hz (default: 10.0). Used only when --gpu-tracking on.",
    )
    parser.add_argument(
        "--vllm-pid",
        type=int,
        default=None,
        help="PID of the vLLM server process. Required when --gpu-tracking on.",
    )
    parser.add_argument(
        "--vllm-startup-log",
        type=Path,
        default=None,
        help=(
            "Path to vLLM startup stderr log. Required when --gpu-tracking on. "
            "Used to extract GPU baseline (weights MiB, KV cache MiB)."
        ),
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
    elif sub == "inspect":
        _run_inspect(sys.argv[2:])
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
    elif sub == "profile-gpu":
        from trace_collect.profile_gpu import main as run_profile_gpu

        sys.exit(run_profile_gpu(sys.argv[2:]))
    else:
        _run_collect(parse_collect_args())


REPO_ROOT = Path(__file__).resolve().parents[2]


_QWEN3_CODER_30B_LAYERS: tuple[int, ...] = (0, 6, 12, 18, 24, 30, 36, 47)
_PER_HEAD_STATS_PRESETS: dict[str, tuple[int, ...]] = {
    "qwen3-coder-30b": _QWEN3_CODER_30B_LAYERS,
    "default": _QWEN3_CODER_30B_LAYERS,
}


def _parse_per_head_stats_layers(value: str | None) -> tuple[int, ...]:
    """Resolve --per-head-stats-layers into a sorted tuple of layer indices.

    Accepts a preset token (case-insensitive) or a comma-separated integer
    list. Empty/None disables the feature (empty tuple). Out-of-range checks
    against the model's layer count happen later in the recording stack.
    """
    if value is None:
        return ()
    token = value.strip()
    if token == "":
        return ()
    preset = _PER_HEAD_STATS_PRESETS.get(token.lower())
    if preset is not None:
        return preset
    layers = tuple(int(part) for part in token.split(",") if part.strip() != "")
    if not layers:
        raise ValueError(f"--per-head-stats-layers parsed to no layers: {value!r}")
    if any(layer < 0 for layer in layers):
        raise ValueError("--per-head-stats-layers indices must be non-negative")
    return tuple(sorted(set(layers)))


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
    # KV eviction uses the HF backend because that is where HF Cache subclasses
    # can be injected. Attention-independent policies may run without recording
    # internal artifacts; attention-dependent policies are checked after config
    # resolution via the policy capability registry.
    sparse_attn_active = (
        args.sparse_attn != "none" or args.sparse_attn_config is not None
    )
    if sparse_attn_active and not args.record_internals:
        print(
            "ERROR: --sparse-attn / --sparse-attn-config requires "
            "--record-internals (sparse attention only applies to the HF "
            "recording backend).",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.per_head_stats_layers and not args.record_internals:
        print(
            "ERROR: --per-head-stats-layers requires --record-internals "
            "(per-head span stats only apply to the HF recording backend).",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        per_head_stats_layers = _parse_per_head_stats_layers(
            args.per_head_stats_layers
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    # Per-selected-block within-block stats require the block_topk method (its
    # selection ranking IS the bucket axis) AND recorded layers to aggregate
    # over. Validate eagerly (no silent fallback) using a torch-free sparse
    # config resolution so the misconfig surfaces before model load. The main
    # resolution + exclusivity check below re-derives the config for collection.
    if args.per_head_block_stats or args.record_per_head_topk:
        from serving.sparse_attention.config import load_sparse_attention_config

        block_sparse_config = load_sparse_attention_config(args)
        method_name = (
            block_sparse_config.name if block_sparse_config is not None else "none"
        )
        # Name the offending flag in the message so a user enabling either knob
        # without block_topk gets a specific, no-silent-fallback error.
        offending = (
            "--per-head-block-stats"
            if args.per_head_block_stats
            else "--record-per-head-topk"
        )
        if method_name != "block_topk":
            print(
                f"ERROR: {offending} requires --sparse-attn block_topk "
                f"(resolved method: {method_name!r}).",
                file=sys.stderr,
            )
            sys.exit(2)
        if not per_head_stats_layers:
            print(
                f"ERROR: {offending} requires a non-empty --per-head-stats-layers "
                "(which layers to record for).",
                file=sys.stderr,
            )
            sys.exit(2)
    if args.record_per_head_topk and args.per_head_topk_rank <= 0:
        print(
            "ERROR: --per-head-topk-rank must be > 0 "
            f"(got {args.per_head_topk_rank!r}).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.record_internals or args.local_hf:
        os.environ["NANOBOT_MAX_CONCURRENT_REQUESTS"] = "1"

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
    if not provider_config.api_key and not args.local_hf:
        print(
            f"ERROR: Set {provider_config.env_key} or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    from agents.benchmarks import get_benchmark_class
    from agents.benchmarks.base import BenchmarkConfig
    from serving.kv_policies import eviction_policy_requires_attention
    from serving.kv_policies.config import load_eviction_config
    from serving.sparse_attention.config import (
        load_sparse_attention_config,
        validate_attention_method_exclusivity,
    )
    from trace_collect.collector import collect_traces

    eviction_config = load_eviction_config(args)
    if eviction_config is not None:
        os.environ["NANOBOT_MAX_CONCURRENT_REQUESTS"] = "1"
        if (
            eviction_policy_requires_attention(eviction_config)
            and not args.record_internals
        ):
            print(
                "ERROR: the selected KV policy requires attention; pass "
                "--record-internals so AttentionBus can publish post-softmax scores.",
                file=sys.stderr,
            )
            sys.exit(2)
    try:
        sparse_attention_config = load_sparse_attention_config(args)
        validate_attention_method_exclusivity(eviction_config, sparse_attention_config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

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
            instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
            run_id=args.run_id,
            max_context_tokens=args.max_context_tokens,
            mcp_config=args.mcp_config,
            prompt_template=args.prompt_template,
            min_free_disk_gb=args.min_free_disk_gb,
            record_internals=args.record_internals,
            local_hf=args.local_hf,
            eviction_config=eviction_config,
            sparse_attention_config=sparse_attention_config,
            per_head_stats_layers=per_head_stats_layers,
            per_head_block_stats=args.per_head_block_stats,
            record_per_head_topk=args.record_per_head_topk,
            per_head_topk_rank=args.per_head_topk_rank,
            generation_seed=args.seed,
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

    from trace_collect.simulator import simulate, validate_gpu_tracking_args

    try:
        validate_gpu_tracking_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

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
        "command_timeout_s": args.command_timeout,
        "warmup_skip_iterations": args.warmup_skip_iterations,
        "replay_speed": args.replay_speed,
        "structured_output": args.output_dir == "traces/simulate",
    }

    if args.mode == "cloud_model":
        if args.metrics_url:
            print(
                "ERROR: cloud_model replay does not support --metrics-url.",
                file=sys.stderr,
            )
            sys.exit(2)
        sweep_path = Path(args.output_dir) / "throughput_sweep.jsonl"
        if len(concurrency_values) > 1 and sweep_path.exists():
            sweep_path.unlink()
        for concurrency in concurrency_values:
            trace_file = asyncio.run(
                simulate(**simulate_kwargs, concurrency=concurrency)
            )
            print(f"Simulate trace written to: {trace_file}")
            if len(concurrency_values) > 1:
                _append_throughput_sweep_record(sweep_path, trace_file)
        if len(concurrency_values) > 1:
            print(f"Throughput sweep written to: {sweep_path}")
        return

    if concurrency_values != [1]:
        print(
            "ERROR: local_model mode requires --concurrency 1.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.api_base:
        print(
            "ERROR: local_model simulate requires --api-base for the target "
            "OpenAI-compatible endpoint.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        llm_config = resolve_llm_config(
            provider=args.provider,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            environ=os.environ,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    if not llm_config.api_key:
        print(
            f"ERROR: Set {llm_config.env_key} or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    gpu_tracking_kwargs: dict = {}
    if args.gpu_tracking == "on":
        from harness.vllm_startup_parser import parse_startup_log_file

        gpu_baseline = parse_startup_log_file(args.vllm_startup_log)
        if gpu_baseline is None:
            print(
                f"ERROR: Failed to parse vLLM startup log at {args.vllm_startup_log}; "
                "check log file content and vLLM version (supported 0.5–0.7)",
                file=sys.stderr,
            )
            sys.exit(2)
        gpu_tracking_kwargs = {
            "gpu_baseline": gpu_baseline,
            "vllm_pid": args.vllm_pid,
            "gpu_sample_hz": args.gpu_sample_hz,
        }

    trace_file = asyncio.run(
        simulate(
            **simulate_kwargs,
            concurrency=1,
            api_base=llm_config.api_base,
            api_key=llm_config.api_key,
            model=llm_config.model,
            metrics_url=args.metrics_url,
            **gpu_tracking_kwargs,
        )
    )
    print(f"Simulate trace written to: {trace_file}")


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
        description="Inspect an OpenClaw JSONL trace file.",
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
