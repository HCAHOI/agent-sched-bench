#!/usr/bin/env python
"""W1 vLLM connector KV-offload spike driver (GPU box only).

Launches vLLM with the SelectiveOffloadConnector, drives N background "load"
requests plus one "agent" request, and mid-generation triggers offload of the
agent request's KV blocks to pinned host memory via the out-of-band control
file, waits a configurable tool duration, restores, and resumes. Measures and
reports (JSON): per-repetition offload/restore latency, bytes moved, effective
GB/s, and the co-running load requests' inter-token-latency (ITL) distribution
in windows WITH vs WITHOUT an offload event -- the interference measurement.

All timing/reporting math lives in ``spike.vllm_connector.core`` and is unit
tested on CPU. This driver only orchestrates vLLM and is exercised on the box.

Run (on the GPU box, after `uv pip install -e '.[serving-spike]'`):

    PYTHONPATH=src:. python spike/run_spike.py \
        --model meta-llama/Llama-3.1-8B --num-load 8 --repetitions 5 \
        --tool-duration-s 3.0 --output spike_note.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from spike.vllm_connector.control import OffloadControl
from spike.vllm_connector.core import RepetitionResult, SpikeReport, TransferTiming, env_info
from spike.vllm_connector.gpu import vllm_version

# A long, low-entropy prompt so the agent request holds many KV blocks worth
# offloading. Repetition is intentional -- we want KV bytes, not diversity.
_AGENT_PROMPT = "Summarize the following log in detail.\n" + ("audit line\n" * 512)
_LOAD_PROMPT = "Count slowly and explain each step.\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--num-load", type=int, default=8, help="co-running load requests")
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--tool-duration-s", type=float, default=3.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--control-dir", default=".spike_control")
    p.add_argument("--output", default="spike_note.json")
    p.add_argument(
        "--block-dim",
        type=int,
        default=0,
        help=(
            "Block-index dimension of the paged KV tensor vLLM hands to "
            "register_kv_caches; forwarded to the connector's CudaBlockTransfer. "
            "Default 0 matches the standard v1 layout -- verify on the box."
        ),
    )
    p.add_argument(
        "--transfer-mode",
        choices=["strided", "staged"],
        default="staged",
        help=(
            "KV transfer path. 'staged' (default): pinned staging buffer + "
            "batched async copies on a dedicated stream (the improved path). "
            "'strided': the original per-block loop with a full device sync "
            "(the W1 baseline) -- select it to A/B the improvement."
        ),
    )
    p.add_argument(
        "--staging-max-blocks",
        type=int,
        default=128,
        help=(
            "Staged mode only: pre-allocated staging buffer capacity in blocks. "
            "A transfer of more blocks than this fails fast. Size above the "
            "largest request's block count (W1 measured ~97)."
        ),
    )
    p.add_argument(
        "--chunk-bytes",
        type=int,
        default=0,
        help=(
            "Staged mode only: cap bytes moved per copy op as a rate limiter "
            "(0 = one copy, no chunking). Lets co-tenant work slip between "
            "chunks to shave the ITL tail at the offload moment."
        ),
    )
    return p


def _gpu_env_info() -> dict[str, object]:
    """Best-effort GPU/driver env info for the report (GPU box only).

    Every field is optional: torch/CUDA absence or a missing ``nvidia-smi``
    silently omits the corresponding keys rather than failing the run --
    this is provenance metadata, not something the spike depends on.
    """
    info: dict[str, object] = {}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,pcie.link.width.current,pcie.link.gen.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        first = out.splitlines()[0] if out else ""
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            info["driver_version"], info["pcie_link_width"], info["pcie_link_gen"] = parts[:3]
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def make_engine(args: argparse.Namespace, control_path: str, timing_path: str):
    """Construct the vLLM async engine wired to the offload connector."""
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    # Register our connector by import path so vLLM can instantiate it.
    KVConnectorFactory.register_connector(
        "SelectiveOffloadConnector",
        "spike.vllm_connector.gpu",
        "SelectiveOffloadConnector",
    )
    kv_cfg = KVTransferConfig(
        kv_connector="SelectiveOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "control_path": control_path,
            "timing_path": timing_path,
            "block_dim": args.block_dim,
            "transfer_mode": args.transfer_mode,
            "max_blocks": args.staging_max_blocks,
            "chunk_bytes": args.chunk_bytes,
        },
    )
    engine_args = AsyncEngineArgs(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_transfer_config=kv_cfg,
        seed=args.seed,
        enforce_eager=True,  # deterministic timing; skip cudagraph capture
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def _drain(engine, prompt: str, request_id: str, max_tokens: int, seed: int) -> list[float]:
    """Consume a request's stream, returning per-token arrival timestamps."""
    from vllm import SamplingParams

    # temperature=0.0 is deterministic greedy decoding -- `seed` does not
    # affect the generated tokens here. It is threaded through purely as
    # per-repetition provenance (see RepetitionResult docstring), so a rerun's
    # request ids/prompts line up for comparison, not because it reseeds
    # sampling.
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed)
    stamps: list[float] = []
    prev = 0
    async for out in engine.generate(prompt, params, request_id):
        n = len(out.outputs[0].token_ids)
        if n > prev:
            stamps.append(time.perf_counter())
            prev = n
    return stamps


def _itl_ms(stamps: list[float]) -> list[float]:
    return [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]


def _read_timings(timing_path: str, phase: str) -> list[dict]:
    if not Path(timing_path).exists():
        return []
    rows = [json.loads(line) for line in Path(timing_path).read_text().splitlines() if line]
    return [r for r in rows if r["phase"] == phase]


async def run(args: argparse.Namespace) -> SpikeReport:
    Path(args.control_dir).mkdir(parents=True, exist_ok=True)
    control_path = str(Path(args.control_dir) / "control.json")
    timing_path = str(Path(args.control_dir) / "timings.jsonl")
    Path(timing_path).unlink(missing_ok=True)

    control = OffloadControl(control_path)
    engine = make_engine(args, control_path, timing_path)
    rng = random.Random(args.seed)

    reps: list[RepetitionResult] = []
    itl_with: list[float] = []
    itl_without: list[float] = []

    for rep in range(args.repetitions):
        seed = rng.randint(0, 2**31 - 1)
        offload_event = rep % 2 == 0  # alternate offload / no-offload windows

        load_tasks = [
            asyncio.create_task(
                _drain(engine, _LOAD_PROMPT, f"load-{rep}-{i}", args.max_tokens, seed + i)
            )
            for i in range(args.num_load)
        ]
        agent_id = f"agent-{rep}"
        agent_task = asyncio.create_task(
            _drain(engine, _AGENT_PROMPT, agent_id, args.max_tokens, seed)
        )

        if offload_event:
            await asyncio.sleep(0.5)  # let the agent request accumulate KV
            control.request_offload(agent_id)
            await asyncio.sleep(args.tool_duration_s)  # the simulated tool call
            control.request_restore(agent_id)

        load_stamps = await asyncio.gather(*load_tasks)
        await agent_task
        control.clear()

        for stamps in load_stamps:
            (itl_with if offload_event else itl_without).extend(_itl_ms(stamps))

        if offload_event:
            off = _read_timings(timing_path, "offload")
            res = _read_timings(timing_path, "restore")
            if not off or not res:
                raise RuntimeError(
                    f"rep {rep}: connector recorded no transfer "
                    f"(offload={len(off)}, restore={len(res)}); "
                    "the seam did not fire -- see README risk section"
                )
            o, r = off[-1], res[-1]
            reps.append(
                RepetitionResult.from_timings(
                    repetition=rep,
                    seed=seed,
                    tool_duration_s=args.tool_duration_s,
                    offload=TransferTiming(o["bytes_moved"], o["milliseconds"], o["num_blocks"]),
                    restore=TransferTiming(r["bytes_moved"], r["milliseconds"], r["num_blocks"]),
                )
            )

    return SpikeReport(
        model=args.model,
        vllm_version=vllm_version(),
        kv_seam="KVConnectorBase_V1.build_connector_meta + worker start_load_kv",
        num_load_requests=args.num_load,
        repetitions=reps,
        itl_with_offload_ms=itl_with,
        itl_without_offload_ms=itl_without,
        env={**env_info(), **_gpu_env_info()},
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Any failure -- notably vLLM EngineDeadError surfacing through the awaited
    # request streams -- must exit nonzero. The engine can die mid-run and a
    # bare `asyncio.run(...)` had let a swallowed error still exit 0, marking a
    # broken run as a clean spike. Catch, report, and fail loudly instead.
    try:
        report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - top-level guard, re-report and exit
        print(f"spike run failed: {exc!r}", file=sys.stderr)
        return 1
    Path(args.output).write_text(report.to_json())
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
