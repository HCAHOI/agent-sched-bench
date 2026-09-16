# agent-sched-bench — archived

**This repository is closed to new work as of 2026-09-16.** Its code was split
in two along the line the benchmark actually has: one side runs the agent and
executes its tools, the other serves its prompts. Work continues there.

| Repository | What it owns |
|---|---|
| [`agent-sched-bench-cpu`](https://github.com/HCAHOI/agent-sched-bench-cpu) (private) | The agent runtime: benchmark plugins, trace collection, task containers, tool execution, replay. Takes an LLM response, produces the next prompt. |
| [`agent-sched-bench-gpu`](https://github.com/HCAHOI/agent-sched-bench-gpu) (private) | The serving side: vLLM engines, the published scheduling baselines and their pinned forks, prefill/decode layouts, KV tiering, engine-pool simulation. Takes a prompt, produces a response. |
| [`agent-sched-bench-common`](https://github.com/HCAHOI/agent-sched-bench-common) (private) | `asb_common` — the contract both sides import: the serving wire format, the recorded-trace and workload formats, replay timing, and the two-machine handshake. No runtime dependencies. |

Each side can run alone. With no agent side attached, the serving side replays a
recorded trajectory and treats every tool call as a sleep it may accelerate; with
no serving side attached, the agent side treats LLM inference the same way. That
is why the tool-timing regime is a required field of the handshake rather than a
default — a run whose records do not say which regime it used cannot be compared
with any other run.

## What is still here

Two things, and nothing else worth cloning for:

- **`results/` — 159 GB, the only copy.** Every run directory named in Milestones
  1 through 4 lives here, in this working tree, on one host. It was never in Git
  and it did not move at the split, because it is too large to carry and no
  result receipt depends on having it. **Never delete or rewrite it.** The
  receipts in the two new repositories are the portable record; this tree is the
  provenance behind them. A result is locally recoverable only when its receipt
  also names a retained in-repository file, archive, or Git object.
- **The history up to `split-point-20260916`.** The tag marks the last commit
  before the split. Both new repositories start from copies of this tree, so
  anything deleted during the split is recoverable here by path and revision —
  for example the pre-split `analysis/ROADMAP.md` at `9a13b794`, which the CPU
  repository's `CLAIMS.md` still cites for the spent-corpora text.

The code in this tree is the pre-split arrangement. It is kept for provenance,
not for use: it has no successor commits, and every fix since 2026-09-16 landed
in one of the three repositories above.

## If you are looking for

| | Where it went |
|---|---|
| Benchmark plugins, `src/trace_collect/`, `src/agents/`, task containers | `agent-sched-bench-cpu`, `src/` |
| Workload manifests, task pools, run pre-registrations | `agent-sched-bench-cpu`, `analysis/development/` |
| The July 2026 KV-stopping lane and its closed questions | `agent-sched-bench-cpu`, `analysis/` |
| `scripts/evaluation/`, `scripts/baselines/`, `scripts/serving/` | `agent-sched-bench-gpu`, `scripts/` |
| Paper-baseline receipts, the single-instance L40S notes, serving measurements | `agent-sched-bench-gpu`, `analysis/` |
| The serving wire format and the replay handshake | `agent-sched-bench-common`, `asb_common/` |
| The milestone series and `PENDING.md` | Both new repositories, `millstone/`, identical in each |
| Raw run directories | Here, in `results/`, and nowhere else |
