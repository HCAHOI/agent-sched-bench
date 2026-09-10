# Next-turn output-length prediction: handoff

Last updated: 2026-09-04. This file is the current handoff for continuing the output-length prediction experiment. It records the active objective, frozen data, code and machine state, completed checks, known failures, and the next executable steps.

## Objective

For every agent LLM step, use only the full message/tool prefix visible before the next assistant turn to predict that turn's generated token count. Compare four published methods through one modular harness:

- SSJF-Reg;
- EGTP-static;
- OUTLETS-static;
- TIE.

The intended primary evaluation uses natural generations from the target model rather than the source trace's original completion length. That path is currently blocked by non-terminating Qwen3-4B generations. A separate exploratory branch now evaluates methods that can learn from the single recorded completion length for each trace prefix. Report q-error q50/q90/q95/q99 and mean, mean accuracy, MAE, and underprediction rate. Recorded lengths and oracle lengths remain analysis-only and must never become deployable features.

Target model: `Qwen/Qwen3-4B-Instruct-2507-FP8`. The current development run uses two RTX 5090 GPUs. The original L40S node has been destroyed.

## Frozen benchmark data

Source archives:

- `traces/exports/swe-rebench-original-flat-644-20260904.tar.zst`
- `traces/exports/terminal-bench-original-flat-239-20260904.tar.zst`

Source inventory:

- SWE-ReBench: 644 trajectories, 631 underlying tasks, 271 repositories, 28,415 LLM steps; 374 `qwen3.7-max` and 270 `gpt-5.6-sol` trajectories.
- Terminal-Bench: 239 trajectories/tasks, 3,354 LLM steps, all `gpt-5.6-sol`.
- Combined: 883 trajectories, 870 underlying task IDs, 31,769 LLM steps.

Sampling is fixed at up to five evenly spaced assistant-turn prefixes per trajectory: first, 25%, 50%, 75%, and last. Duplicate rounded positions are removed. Nine selected source actions with errors or zero output tokens are excluded explicitly.

Splits are grouped by underlying task so that two trajectories for the same task cannot cross splits. Seed 42 gives 70/10/20 train/validation/test splits. The exported dataset contains:

| Item | Count |
|---|---:|
| Samples | 4,371 |
| Sessions with retained samples | 866 |
| Trajectories | 883 |
| SWE-ReBench samples | 3,219 |
| Terminal-Bench samples | 1,152 |
| Train / validation / test samples | 3,066 / 436 / 869 |
| Train / validation / test sessions | 606 / 86 / 174 |

Every request includes the same ten OpenClaw tool definitions (`read_file`, `write_file`, `edit_file`, `list_dir`, `exec`, `web_search`, `web_fetch`, `message`, `spawn`, `sessions_yield`) and `tool_choice=auto`.

Remote dataset:

```text
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/dataset
```

The dataset is about 249 MB. Its authoritative metadata is `dataset.json`; prefixes are in `prefixes.jsonl`.

## Code and predictor repositories

Local repository:

```text
/home/chiyu/workspace/agent-sched-bench
branch: codex/cleanup-research-dead-code
commit: 0875daa0a7b95d10beb42ecd257c5c9d592684f3
```

Commit `0875daa` contains the original implementation. The local worktree now also contains reviewed, uncommitted support for using canonical recorded-trace labels with SSJF-Reg and EGTP-static, plus this handoff and locally copied result artifacts:

- `scripts/evaluation/output_length_benchmark.py`: export, natural-label collection, and evaluation;
- `scripts/evaluation/output_length_predictors.py`: modular predictor adapters;
- `configs/evaluation/output_length_openclaw_tools.json`: shared tool schema;
- `tests/test_output_length_benchmark.py`: focused contract tests.

The implementation was independently reviewed in stages. Important resolved issues include immutable resume protocol inputs, exact and even draw coverage, rejection of conflicting OpenAI options, task-grouped splits, isolated/commit-verified upstream adapters, full-prefix rendering for EGTP, and removal of ALPS because the published code leaks decode-time hidden state.

The code is copied to the remote node at `/workspace/agent-sched-bench`, but the remote copy has no Git metadata. Commit `0875daa` has not been pushed to GitHub. Therefore local Git is the source of truth; the remote node is only an execution copy.

Pinned upstream checkouts on the remote node:

| Method | Path | Commit |
|---|---|---|
| SSJF-Reg | `/workspace/upstreams/ssjf` | `4b866866a32626677a1841a3b92875b93b1f03ab` |
| EGTP | `/workspace/upstreams/l_p_bench` | `170a47893e1351ecc062186aafc686ab3f8be3d1` |
| TIE | `/workspace/upstreams/tie` | `ce6ddc4d7abe6a4a0821a03d463ad7af86113850` |
| OUTLETS | `/workspace/upstreams/outlets` | `4b53761496da49ad9829a9817cbfa3c7c9047c52` |

All four checkouts were clean at the last check. OUTLETS still requires a checkpoint/config produced by its official training pipeline; a ready official checkpoint has not yet been confirmed.

## Current compute and serving configuration

SSH:

```bash
ssh -p 28126 root@137.175.22.196
```

Node and environment:

- Vast.ai host `a886e2f6529f`;
- 2 x RTX 5090, 32,607 MiB each;
- driver 580.159.03, CUDA 12.8;
- 256 CPU threads, about 503 GiB RAM;
- about 120 GiB free disk at the last check;
- Python 3.12, PyTorch 2.9.0+cu128, Transformers 4.57.6, vLLM 0.11.2 in `/venv/main`;
- unprivileged container, so no Docker-in-Docker or kernel profiler;
- `/workspace` is ephemeral unless separately synchronized. Pull all irreplaceable results locally before recycling or destroying the node.

Model snapshot:

```text
/workspace/.hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/8591804019c8b22094c3b5b4454e0edc05dffc98
```

The model server is managed by Supervisor as `length_vllm`, configured in `/etc/supervisor/conf.d/length-vllm.conf`, and listens only on `127.0.0.1:8000`. The active serving parameters are:

```text
--served-model-name Qwen/Qwen3-4B-Instruct-2507-FP8
--data-parallel-size 2
--enable-prefix-caching
--gpu-memory-utilization 0.95
--max-model-len 163840
--max-num-seqs 10
--enable-auto-tool-choice
--tool-call-parser hermes
```

The 163,840-token service limit is the 32K revision: maximum measured prompt 127,126 plus `max_tokens=32,768`, with 3,946 tokens of margin. After restart, each data-parallel engine reported 184,208 KV-cache tokens and 1.12x maximum concurrency at the 163,840-token limit.

Target prompt-token distribution under the actual Qwen chat template and full tool schema:

| Percentile | Prompt tokens |
|---|---:|
| p50 | 13,058 |
| p75 | 23,866.5 |
| p90 | 36,428 |
| p95 | 45,668 |
| p99 | 79,254.4 |
| max | 127,126 |

Token statistics are saved at:

```text
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/prompt-token-stats.json
```

## Completed work and evidence

1. The cross-benchmark dataset was exported and its grouped split/counts were verified.
2. The full request tool schema was preserved and vLLM auto-tool parsing was enabled; without the two tool-parser flags vLLM returns HTTP 400.
3. The longest target-tokenized prefix was tested with 20 natural draws at the old 16K output cap. All 20 terminated naturally: 19 `stop`, one `tool_calls`, output lengths 13 to 1,243 tokens, and no OOM/truncation. This established that a 127,126-token prompt fits, but it did not test the longest natural output.
4. A formal 20-draw run began with a 16,384-token output cap and produced 660 valid labels in about ten minutes. The next generation hit the cap, so the labeler correctly failed rather than recording a censored target.
5. SSJF-Reg and EGTP-static were run on all 3,066/869 recorded-label train/test samples. Their source-label adapter changes passed focused tests and independent review.

Longest-prefix smoke artifacts:

```text
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/longest-prefix-smoke-dataset
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/longest-prefix-smoke-natural-labels
```

Failed 16K run artifacts:

```text
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/natural-labels-qwen3-4b-fp8-20draw
```

Failure sample:

```text
qwen3.7-max/AzureAD__microsoft-authentication-library-for-python-723/AzureAD__microsoft-authentication-library-for-python-723/llm_54
RuntimeError: natural label ... hit max_tokens=16384
```

The 660 existing labels remain useful only as a failed-run diagnostic. They belong to the 16K protocol and must not be resumed or combined with 32K labels.

## Recorded-completion exploratory results

Only SSJF-Reg and EGTP-static can consume one scalar completion length per prefix without inventing missing distributional labels. EGTP is explicitly a cross-model proxy here: local Qwen features predict lengths produced by `gpt-5.6-sol` and `qwen3.7-max`. TIE needs repeated draws to estimate a per-prefix distribution; official OUTLETS training needs the generating model's completion-side hidden states.

| Method | q50 | q90 | q95 | q99 | Mean accuracy | MAE | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| SSJF-Reg | 12.282 | 53.389 | 71.607 | 192.809 | 0.100 | 256.542 | Severe underfit; mean prediction 11.08 vs. truth 267.62 |
| EGTP-static, Qwen proxy | 2.423 | 4.960 | 7.105 | 12.612 | 0.465 | 223.207 | Collapsed to near-constant 262.898 because official `k=4` sees nearly invariant prompt headers |
| Train-set median constant | 1.854 | 4.473 | 5.466 | 15.262 | 0.546 | 186.974 | Stronger than both published-method runs on q50/q95/accuracy/MAE |

Remote outputs:

```text
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/ssjf-reg-source-labels-seed42
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/egtp-static-source-labels-seed42
```

Local copies, including predictions, protocols, trained weights, labels, and evaluations:

```text
analysis/results/output-length-source-labels-crossbench-20260904
```

## Natural-label amendment and outcome

After observing the 16K censoring failure, the user changed the natural-generation cap to 32,768 tokens. This is an explicit development-protocol amendment made with the 16K outcome visible, not a preregistered choice.

The 32K labeling protocol is otherwise unchanged:

```text
model: Qwen/Qwen3-4B-Instruct-2507-FP8
max_tokens: 32768
temperature: 0.7
top_p: 0.8
seed: 42
draws: 20
concurrency: 19
splits: train,validation,test
timeout: 3600 seconds
natural termination: required
```

The service restart and exact-prefix check are complete. The prior failing sample's draw 0 terminated naturally at 469 tokens with `finish_reason=tool_calls`. This proves the 32K request path works, but it also shows that a seed is not fully deterministic under concurrent data-parallel serving: the earlier run's nominally corresponding draw reached 16K.

The full 4,371 x 20 run started at approximately 2026-09-04 15:41:25 UTC under Supervisor program `length_full_32k`. It reproduced the same pathological sample after writing 660 labels and automatically failed at 15:56 UTC because the generation reached the new 32,768-token cap. vLLM recorded exactly one `finished_reason=length`; there was no OOM or server failure. Its diagnostic output is:

```text
/workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/natural-labels-qwen3-4b-fp8-20draw-max32k
```

This sample therefore has stochastic behavior: the isolated 32K probe ended after 469 tokens, while the formal warm draw generated all 32,768 tokens without natural termination. The 32K run is invalid as a complete natural-label dataset and must not be resumed as though all targets were uncensored. To resume the natural-label study:

1. Choose and declare a label contract for non-terminating/heavy-tail draws: censored survival target, explicit invalid-draw exclusion, or another bounded target. Do not silently raise the cap again.
2. Run the four predictor adapters only after that contract is fixed and the resulting labels have consistent coverage.

## Current limitations and open problems

- Natural output has a heavy tail: one draw exceeded 16K even though the longest-prefix smoke outputs were all below 1.3K. The 32K cap is still a censoring boundary, so any cap hit invalidates that label run.
- One warm request is performed per prefix before its other 19 draws run concurrently. This improves prefix reuse but means only one data-parallel engine is active during that warm draw. Do not change this behavior mid-protocol; optimize it only as a separately declared revision.
- The node is ephemeral and the formal dataset currently exists only remotely. The source trace archives and code exist locally, so it can be rebuilt, but losing completed labels would waste substantial GPU time.
- The current remote environment was created by installing vLLM 0.11.2 into the image's environment, which replaced its original PyTorch stack and cost about an hour. Do not repeat this on a replacement node: prefer a compatible vLLM image or inspect the preinstalled PyTorch/vLLM compatibility before installing.
- EGTP's published default uses only the first four target-tokenizer tokens; the adapter preserves the official extractor after model-faithful full-prefix rendering. Interpret a weak result as a property of the published method/configuration, not as proof that richer hidden-state predictors cannot work.
- TIE predicts a distribution/risk quantity rather than a native point token length. The adapter's point reduction must be reported explicitly alongside q-error; distribution metrics should remain separate.
- OUTLETS cannot be evaluated faithfully on recorded labels: its official training regenerates completions with the target model and supervises completion-side hidden states. A ready official checkpoint has also not been confirmed.
- GitHub does not yet contain commit `0875daa`; pushing is a separate authorized action.

## Useful commands

Inspect current services:

```bash
ssh -p 28126 root@137.175.22.196 'supervisorctl status; nvidia-smi'
```

Inspect server logs:

```bash
ssh -p 28126 root@137.175.22.196 'tail -100 /workspace/vllm-length.log; tail -100 /workspace/vllm-length.err'
```

Run the complete 32K label job after the exact failing-prefix check passes:

```bash
/venv/main/bin/python scripts/evaluation/output_length_benchmark.py label \
  --dataset-dir /workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/dataset \
  --output-dir /workspace/results/output-length-qwen3-4b-crossbench-5point-seed42-20260904/natural-labels-qwen3-4b-fp8-20draw-max32k \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-tokens 32768 --temperature 0.7 --top-p 0.8 --seed 42 \
  --draws 20 --concurrency 19 --splits train,validation,test --timeout-s 3600
```

All long-running commands on this node must be managed by Supervisor rather than a detached shell process.
