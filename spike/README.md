# W1 spike — vLLM connector selective KV offload

**Goal (roadmap W1-2):** de-risk the P4 long pole — offload one real agent
request's KV blocks to host memory *mid tool-call* through vLLM's connector
layer, restore on completion, and **measure** the offload-path latency +
co-tenant interference. Output feeds the end-of-W2 go/descope decision on the
6-8 week P4 integration.

This package is standalone. It does **not** touch `src/trace_collect/` and adds
no CLI flags there. The base install stays GPU-free; vLLM/torch load only via
the `serving-spike` extra and are imported only in
`spike/vllm_connector/gpu.py` + `spike/run_spike.py`.

## The vLLM seam we target (and its limits)

**Pinned version: `vllm>=0.11,<0.12`** — 0.11.0 is the release that introduced
the v1 KV-connector interface `KVConnectorBase_V1` and the built-in
`OffloadingConnector`.

Entry points used (verified against `v0.11.0`):

- `vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1`
  — abstract base. Scheduler-process methods: `get_num_new_matched_tokens`,
  `update_state_after_alloc`, `build_connector_meta`. Worker-process methods:
  `register_kv_caches`, `start_load_kv`, `wait_for_layer_load`,
  `save_kv_layer`, `wait_for_save`.
  Source: <https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py>
- `vllm.config.KVTransferConfig` (`kv_connector`, `kv_role`,
  `kv_connector_extra_config`) + `KVConnectorFactory.register_connector(name,
  module_path, class_name)` to load a custom connector by import path.
  Docs: <https://docs.vllm.ai/en/latest/features/kv_offloading_usage/>
- Comparison point — the built-in `OffloadingConnector`
  (`kv_connector="OffloadingConnector"`, `CPUOffloadingSpec`): offloads
  *completed* blocks to CPU automatically as a prefix-cache tier. **This is
  NOT what P4 needs** — it is throughput-oriented prefix reuse, not
  externally-triggered per-request offload during a tool call. LMCache plugs in
  the same way (`kv_connector="LMCacheConnectorV1"`) and is likewise
  block-completion / prefix driven, not command driven.
  Blog: <https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html>

### Why a custom connector, and what it does NOT do (the honest finding)

**vLLM 0.11 exposes no public API to say "offload request R's resident KV now,
pause R, restore later" on external command.** Every offload-relevant hook
fires on the scheduler's own lifecycle (block completion, preemption under
memory pressure, prefix match) — the trigger is internal and the victim is
chosen by the scheduler, not a caller. `OffloadingConnectorScheduler` does
offload-on-preemption + reload-on-resume, but the trigger is GPU memory
pressure, not a tool-call signal.

The closest public seam, which this spike exercises:

1. `build_connector_meta(scheduler_output)` runs in the scheduler **every
   step**. Our `SelectiveOffloadConnector` reads an out-of-band control file
   (`OffloadControl`) there and, for the target request whose block ids it
   tracked in `update_state_after_alloc`, emits connector metadata directing
   the worker to copy those blocks.
2. `start_load_kv` runs in the **worker** (where the paged KV tensors live) and
   executes the copy GPU↔pinned-host via `CudaBlockTransfer`, timed with CUDA
   events. This is byte-for-byte the operation vLLM's own preemption-swap uses.

**Limit this spike deliberately measures around:** emitting offload metadata
*copies* the blocks; it does **not** by itself evict them from vLLM's block
pool or pause the request's generation. Making the offload actually *free* GPU
memory (the point of P4) additionally requires scheduler-level block eviction +
request pause/resume — the machinery `OffloadingConnectorScheduler` owns on the
preemption path. So this spike measures the **transfer path** (latency,
bandwidth, interference) — the long-pole systems risk — and flags the
eviction/pause + external-trigger wiring as net-new P4 work not covered by any
public API. That gap is the primary input to the W2 go/descope call.

**Layout assumption to verify on the box:** `CudaBlockTransfer` defaults to the
block index on dim 0 (`tensor[block_id]`), the standard v1 layout, but every
shape/index computation runs through `tensor.movedim(block_dim, 0)`, so a
different layout is `--block-dim N` on the driver (threaded through
`kv_connector_extra_config["block_dim"]` into the connector and
`CudaBlockTransfer`), not a code change. Confirm the real dim against the
running model's cache shape before trusting byte counts — `_copy` bounds
-checks every block id against that dim's size and raises rather than
corrupting an out-of-range block.

## Setup (GPU box, the moment it is rented)

```bash
bash scripts/setup/benchmark_server.sh          # base env (no GPU deps)
source .venv/bin/activate
uv pip install -e '.[serving-spike]'            # pulls vllm 0.11.x + torch/CUDA
```

## Run

```bash
PYTHONPATH=src:. python spike/run_spike.py \
    --model meta-llama/Llama-3.1-8B \
    --num-load 8 --repetitions 6 --tool-duration-s 3.0 \
    --seed 0 --output spike_note.json
```

`--num-load` co-running requests generate token load; one `agent-*` request is
offloaded mid-stream on even repetitions (odd reps are the no-offload control
window for the interference baseline). Latencies come back via a JSONL the
worker appends (`--control-dir`), since the connector runs in a separate
process from the driver.

## CPU tests (no GPU)

```bash
PYTHONPATH=src:. python -m pytest tests/test_vllm_connector_spike_logic.py -q
```

Covers control channel, timing math, ITL stats, fake backend, report schema —
everything the driver depends on except the CUDA copy and vLLM orchestration.

## Output schema (`spike_note.json`)

```jsonc
{
  "model": "...", "vllm_version": "0.11.x",
  "kv_seam": "KVConnectorBase_V1.build_connector_meta + worker start_load_kv",
  "num_load_requests": 8,
  "env": {"host": "...", "python": "...",
          "torch": "...", "gpu": "...", "cuda": "...",
          "driver_version": "...", "pcie_link_width": "...", "pcie_link_gen": "..."},
  "repetitions": [{"repetition": 0, "seed": ..., "tool_duration_s": 3.0,
                   "num_blocks": ..., "bytes_moved": ...,
                   "offload_ms": ..., "restore_ms": ...,
                   "offload_gbps": ..., "restore_gbps": ...}, ...],
  "itl_with_offload_ms": [...], "itl_without_offload_ms": [...],
  "summary": {"offload_ms": {"median":..,"p99":..},
              "restore_ms": {"median":..,"p99":..},
              "offload_gbps_median": .., "restore_gbps_median": ..,
              "bytes_moved": ..,
              "interference": {"with_offload": {...}, "without_offload": {...}}}
}
```

`host`/`python` come from `core.env_info()` (always present). `torch`/`gpu`/
`cuda`/`driver_version`/`pcie_link_width`/`pcie_link_gen` come from
`run_spike._gpu_env_info()` and are best-effort: each is present only if the
corresponding probe (torch import, `nvidia-smi`) succeeds on the box.

## W1 spike-note skeleton (fill from the run, name commit + box)

> **Commit:** `<sha>` · **Box:** `<host / GPU / PCIe gen·width from nvidia-smi>`
> · **vLLM:** `<version>` · **Model:** Llama-3.1-8B
>
> **Seam reached:** custom `KVConnectorBase_V1` — `build_connector_meta`
> (scheduler) → worker `start_load_kv` CUDA copy. Connector fired on N/M
> offload triggers under `--num-load=8` load. `[yes/no + evidence]`
>
> **Measured offload path** (`summary`): offload `<median>/<p99> ms`, restore
> `<median>/<p99> ms`, `<bytes_moved>` B/event, `<offload_gbps>/<restore_gbps>`
> GB/s effective. Cross-check vs raw-memcpy ceiling from
> `scripts/measure_kv_swap_cost.py` — connector overhead = `<Δ>`.
>
> **Interference:** load-request ITL p99 `<with>` vs `<without>` ms
> (`Δ = <x>×`). Relates to the Fig-7a exogeneity concern (offload perturbs
> co-tenant latency?): `[quantify]`.
>
> **W2 go/descope decision reads from:**
> - Did the connector seam fire externally at all? `[yes → seam exists]`
> - Is offload+restore latency ≪ tool-call duration (µs–ms vs seconds)?
>   `[yes → hiding cost is viable]`
> - **Gap to real P4:** block eviction + request pause/resume + trigger wiring
>   are NOT in any public 0.11 API — estimate `<weeks>` to build on top of the
>   proven transfer path. `[drives 6-8wk realistic? / descope to single-tenant
>   selective-offload validation per roadmap go/no-go]`

## Known risks to the 6-8 week P4 estimate

1. **No public external-offload trigger** (above): P4 must add scheduler-side
   eviction + pause/resume, not just a connector. Largest schedule risk.
2. **Connector fires only when the request is scheduled.** A request that is
   not being forward-passed may not hit `start_load_kv`; driving the copy on an
   idle/paused request may need the preemption path or a scheduler patch —
   verify on the box, this is exactly what the spike surfaces. Related liveness
   gap: a request can finish or get preempted out between the driver's trigger
   and the step that would execute it, and vLLM reassigns its freed block ids
   to other requests. `SelectiveOffloadConnector` prunes its `request_id ->
   block_ids` table from `scheduler_output.finished_req_ids` at the top of
   every `build_connector_meta` call and refuses (logs + no-ops) a directive
   for an untracked/finished target, rather than risking offload/restore
   against reassigned blocks — verify `finished_req_ids` is the field v0.11
   actually exposes on `SchedulerOutput` on the box; if the attribute name
   drifted, the guard degrades to "directive silently dropped" (still safe,
   but worth confirming the field resolves).
3. **Process split:** connector runs in scheduler/worker processes, hence the
   file-based control + JSONL timing channel instead of an in-process object.
   Metadata (`SelectiveOffloadMeta.directives`) crosses the same boundary via
   vLLM's own `bind_connector_metadata` / scheduler-output plumbing — confirm
   on the box that a step's metadata reliably reaches the worker that executes
   it before trusting a "0 directives fired" reading as "seam absent" rather
   than "delivery dropped."
4. **KV layout drift** across vLLM versions (`block_dim`, default 0); pinned
   `<0.12`.
