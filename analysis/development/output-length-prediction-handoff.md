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

### Session-history baselines and output decomposition (2026-09-11)

Meeting question (2026-09-07 §2): does a session's accumulated history of recorded output lengths predict the next output better than a per-request content model or a constant, and what part of the output does the tool-call format bound? `scripts/evaluation/output_length_history_baselines.py` answers both on the same 869 test samples, using only the session's earlier recorded lengths (causal); results in `history-baselines/`. Predictors fall back to the train median at a session's first step (175 of 869 samples). Buckets: <128, 128–512, >512 tokens; "long" = >512.

| Predictor (869 samples) | q50 | q90 | q95 | q99 | MAE | Bucket acc. | Long recall / precision |
|---|---:|---:|---:|---:|---:|---:|---|
| Train-median constant | 1.854 | 4.471 | 5.434 | 15.2 | 187.0 | 0.398 | 0 / 0 |
| Last recorded length | 1.877 | 7.259 | 10.843 | 19.296 | 227.9 | 0.409 | 0.236 / 0.317 |
| Running median | 1.750 | 5.847 | 8.793 | 14.281 | 190.6 | 0.440 | 0.073 / 0.421 |
| EWMA (α = 0.5) | 1.901 | 5.843 | 8.313 | 16.491 | 207.8 | 0.382 | 0.200 / 0.344 |
| Median shrunk to constant (pseudo-count 3) | 1.740 | 5.388 | 7.498 | 12.748 | 187.5 | 0.445 | 0.064 / 0.500 |

On the 694 samples that have history the picture is the same (shrunk median q50 1.713 vs constant 1.924; q90 6.27 vs 4.90). History buys a few percent at the median and a few points of bucket accuracy, and costs the tail: every history predictor has a worse q90/q95 than the constant. The only thing history adds that the constant cannot is some detection of long outputs, at 24% recall and 32% precision (last value). Meeting Assumption 1 (accumulated history helps) is not supported at a level any scheduler could use.

Decomposition of the recorded outputs (token split proportional to tiktoken o200k counts of the visible text and tool-call arguments; recorded completion tokens also include hidden reasoning):

| Quantity | Value |
|---|---:|
| Tool-call arguments, share of visible output tokens (mean / median) | 0.60 / 0.63 |
| Share of output-length variance carried by tool-call arguments | 0.91 |
| Share carried by free text | 0.17 |
| Visible share of recorded completion tokens, gpt-5.6-sol | 0.71 |
| Visible share of recorded completion tokens, qwen3.7-max | 0.43 |

Meeting Assumption 2 is inverted: the tool-call format does not bound the variable part. Free text is short and stable; the arguments (file contents for `write_file` and `edit_file`, long shell commands) carry 91% of the variance, and for the qwen agent 57% of every completion is hidden reasoning that no prefix feature observes. Output length is therefore content-determined at the argument level, which explains why both the content models and the history models sit near the constant. Decision: output length is not a signal worth carrying into a scheduler on this data; a per-task forecast, if any, has to come from context growth, not from generation length.

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

## Amendment 2026-09-12: corrected SSJF-Reg and EGTP-static runs (pre-registered 06:25 UTC, before any number)

The 2026-09-04 runs of both published methods were misapplied, in ways visible in their protocols:

- **SSJF-Reg never trained.** The released recipe regresses raw token counts (targets in the hundreds, variance about 2×10⁵) at learning rate 1e-5, then freezes the encoder after three epochs; train loss went 220K → 216K over six epochs and the mean prediction was 11 tokens against a truth of 268. Fix: `--target-transform log1p` (regress log(1 + tokens), invert at prediction time); everything else as released.
- **EGTP-static saw only the chat-template header.** The official extractor keeps the first *k* = 4 target-model tokens of the prompt. In LP_Bench's chat data those are the user's question; in an agent step they are `<|im_start|>system\n#`, identical for every sample, so the predictor collapsed to a constant. Fix: `--tail-tokens 256` projects the rendered prompt to its last 256 target-model tokens (the latest tool result and the previous assistant turn) before the extractor; run with *k* = 256 (the whole window) and with the official *k* = 4 on that window.

Dataset re-exported locally from the same archives with the same seed (`source_labels.jsonl` byte-identical, test sample ids identical to the frozen run). Runs on the GPU host (`/workspace/outlen/`, GPU 1, venv separate from the serving stack), seed 42, released hyper-parameters otherwise.

Expectation: if per-request content carries the signal, the corrected runs beat the train-median constant on q50 (< 1.85) and detect long outputs (recall > 0.3 at > 512 tokens). If they still sit at the constant, the 2026-09-11 decomposition (91% of variance in tool-call arguments, 43–71% of tokens visible) is the explanation and output length is closed as a scheduling signal on this data.

### Corrected runs: result (read 07:01 UTC, `analysis/results/output-length-source-labels-crossbench-20260904/corrected-20260912/`)

| Predictor (869 test samples) | q50 | q90 | q95 | q99 | Mean accuracy | MAE | Prediction spread |
|---|---:|---:|---:|---:|---:|---:|---|
| Train-median constant (frozen reference) | 1.854 | 4.473 | 5.466 | 15.262 | 0.546 | 187.0 | constant 152 |
| SSJF-Reg, log1p target | 2.025 | 4.376 | 5.331 | 13.392 | 0.533 | 190.4 | 173–176 (converged to a constant) |
| EGTP-static, last 256 tokens, k = 256 | 1.944 | 4.875 | 6.848 | 15.048 | 0.539 | 201.4 | 79–1,717, sd 162 |
| EGTP-static, last 256 tokens, official k = 4 | 2.044 | 5.834 | 8.061 | 15.314 | 0.507 | 224.8 | 92–1,764, sd 164 |

SSJF-Reg now trains (loss 11.4 → 0.82 in log space over six epochs) and its head settles at exp(mean log length): the BERT encoder finds nothing in the last 512 tokens of the canonical context beyond the mean. EGTP with the tail window is no longer constant, its predictions spread over an order of magnitude, but they land in the wrong places: every metric is worse than the constant. The pre-registered expectation (q50 < 1.85, long-output recall > 0.3) is not met by any corrected run.

Decision: output length is closed as a scheduling signal on this data, now with both published methods applied as their papers intend. The explanation stands from the decomposition above: 91% of the variance sits in tool-call arguments whose length is set by content the encoder does not see (file bodies being written), and 29–57% of recorded completions are hidden reasoning.

## Natural labels from Qwen3-30B-A3B-Instruct-2507-FP8 (2026-09-12)

Collected on the Pro 6000 host (vLLM 0.28, one GPU, 163,840-token context, 32,768-token cap, temperature 0, seed 42,
one draw, 16 prefixes in flight): `analysis/results/output-length-source-labels-crossbench-20260904/natural-labels-qwen3-30b-a3b-32k/`
(`labels.jsonl`, `protocol.json`, `rejected.jsonl`). 4,322 of 4,371 prefixes labeled in 2 h 15 min; 49 rejected
(20 hit the 32K cap, 29 exceeded the 30-minute timeout; 41 of the 49 are gpt-agent prefixes). Natural lengths: p10 33,
p50 67, p90 367, p99 1,494, max 9,745, mean 168 tokens; 3,586 end in a tool call, 736 in a stop. The recorded source
lengths on the same prefixes have p50 118, so the target model is terser than the source agents. The rejected samples
are excluded from the natural-label evaluations by a filtered prefix list (`dataset-nat/filter-note.json` on the host,
copied with the results); `dataset.json` is unchanged so the label protocol matches.

The label collector was changed on the way (commits c75699f1, bbe351df): prefixes are labeled concurrently, rows are
written as they finish, and censored or timed-out samples are recorded in `rejected.jsonl` instead of aborting the run.

### Corrected predictors on natural labels (read 14:53 UTC, `natural-20260912/`)

860 test samples (869 minus 9 rejected). Constant = train-split median of the natural labels (68 tokens).

| Predictor | q50 | q90 | q95 | MAE | Mean accuracy | Prediction spread |
|---|---:|---:|---:|---:|---:|---|
| Constant (natural train median) | 2.061 | 5.665 | 8.500 | 118.5 | 0.512 | 68 |
| SSJF-Reg, log1p target | 2.369 | 4.186 | 6.916 | 122.6 | 0.474 | near-constant |
| EGTP-static, last 256 tokens, k = 256 | 2.550 | 5.355 | 8.841 | 146.8 | 0.453 | 49–1,343, sd 143 |

With labels generated by the target model itself, the ranking is unchanged: neither published method beats the constant at the median or on MAE; SSJF only trims the tail (q90 4.2 vs 5.7). The cross-model-proxy caveat of the 2026-09-04 runs is therefore not what held the methods back. OUTLETS remains the one untested method; it needs its official code (not on any machine we have) or a reimplementation.

### Internal-state probe on the target model (2026-09-12 16:25 UTC, user's choice after OUTLETS's code proved unavailable)

`scripts/evaluation/output_length_hidden_probe.py`: the final-layer, last-token hidden state of Qwen3-30B-A3B-Instruct-2507-FP8
after prefill (vLLM 0.28 pooling server, LAST pooling, no activation, 2,048 dims; 4,371 prefixes in 33 min on one GPU),
then a 2-layer MLP (256 hidden, GELU, dropout 0.1) on log(1 + tokens), selected on the validation split. This is the
"shallow internal-state probe" the OUTLETS paper positions itself against, not OUTLETS (no draft backbone, no fused layers).

| Labels | Predictor | q50 | q90 | q95 | MAE | Mean accuracy | Bucket acc. (<128/128–512/>512) | Long recall / precision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| natural (30B-A3B), 860 | constant (68) | 2.061 | 5.665 | 8.500 | 118.5 | 0.512 | 0.676 | 0 / 0 |
| natural | SSJF-Reg log1p | 2.369 | 4.186 | 6.916 | 122.6 | 0.474 | — | — |
| natural | EGTP tail-256 k=256 | 2.550 | 5.355 | 8.841 | 146.8 | 0.453 | — | — |
| natural | **hidden-state probe** (seed 42) | **1.350** | **2.661** | **3.938** | **78.1** | **0.702** | **0.810** | 0.163 / 0.583 |
| natural | probe, seeds 1 / 2 / 3 | 1.357 / 1.357 / 1.350 | 2.82 / 2.80 / 2.65 | 3.80 / 3.74 / 3.63 | 82.8 / 80.7 / 79.5 | 0.695–0.699 | — | — |
| recorded (source agents), 869 | constant (152) | 1.854 | 4.473 | 5.466 | 187.0 | 0.546 | 0.398 | 0 / 0 |
| recorded | hidden-state probe (seed 42) | **1.385** | **3.035** | **4.313** | **142.8** | **0.674** | — | — |

First predictor in this study that beats the constant, and by a wide margin on every metric, stable across four seeds. It
works for the model's own generations (natural labels) and, less well but still clearly, for other agents' recorded
outputs from the same prefixes. The tail is still the weak spot: 16% recall of outputs above 512 tokens at 58% precision.
Reading against the earlier negatives: the length signal exists inside the target model's representation of the whole
context; external encoders on a 512-token window (SSJF) or 4–256 prompt tokens (EGTP) do not see it.

Tail variants (2026-09-12 17:07 UTC, pre-registered in `millstone/PENDING.md` §8b "Tail lane", results in
`probe-replay-pool64v4/variants/`): pinball 0.75 / 0.9, long-weighted MSE, a 3-bucket class head, and class head +
weighting raise long recall (0.16 → 0.47–0.74) only by losing precision (0.58 → 0.23–0.43) and q50 (1.35 → 1.41–1.97),
and every one of them makes the sandbox stage-2 estimate worse (p90 absolute error 25.8 s → 31–56 s). The head cannot
separate long from not-long from the prompt-side state; it can only shift everything up. Sampled labels (2026-09-13, 850 test prefixes × 4 draws at temperature 0.7, `natural-labels-sampled-test-d4/`): the
tail is reproducible, P(long in another draw | long in one) = 0.70, within-prefix share of log-length variance 12%;
the leave-one-draw-out ceiling is q50 1.10 with long recall 0.69, so the probe's tail gap (recall 0.14) is a method
limit. Continuation probe (`continuation-20260913/`, the same head on prefixes extended by the first k greedy tokens):
k = 16 and 64 add nothing about the tail (recall 0.09 / 0.05), k = 256 reaches 0.47 at precision 0.44; the
final-layer last-token state does not expose what the model will do until it is well under way. Multi-layer probe (`aux-layers-20260913/`, residual stream after layers 2 / 24 / 45 plus the final state, 8,192 dims,
via `scripts/evaluation/vllm_aux_layers_sitecustomize.py`): q50 1.36–1.41 over four seeds, long recall 0.14–0.30 at
precision 0.41–0.50, replay stage-2 p90 25.2 s; no better than the final layer alone. Conclusion of the line: the
prompt-side hidden state, at any depth, does not expose the tail to an MLP head; a point predictor for long outputs
would need completion-side supervision (OUTLETS proper, draft model), which is not a prompt-side estimate. Not started.

## Sandbox-side interface: three signals, not a point prediction (2026-09-12)

The collaborator restores each task's sandbox around the LLM step (restore p50 1.0 s, p90 1.8 s) and asked for a
step-time estimate. Measured on four finished replay runs with `scripts/evaluation/step_time_staged_estimates.py`
(1,951 steps each; `probe-replay-pool64v4/staged-*.json`; full tables in `millstone/PENDING.md` §10), the useful
interface is three events with a bound attached, not one predicted duration:

1. **`scheduled`** (the engine started prefill; hold and queue are the LLM side's own decisions, so they are sent, not
   predicted). Attached bound: uncached prompt tokens × the calibrated prefill constant. It is a true lower bound on
   the remaining time in 100% of steps. At 32B every step still has ≥ 1.8 s of engine time after this event, so a
   restore started here is never late; at 4B 76–82% do, the rest need the hold as slack.
2. **`reasoning closed`** (the model emitted its reasoning-end token; the tool name follows within a few tokens).
   Attached estimate: pool median visible tokens for that tool × causal TPOT. Reasoning is a third of the output
   tokens (median 0.36 on the recorded traces), so this lands after ~40% of the decode; the absolute error of the
   remainder halves (32B DualMap p50 5.1 → 2.5 s, p90 25.8 → 14.8 s) because less time is left, while the relative
   error is unchanged (q-err ≈ 1.84): what remains is the tool arguments, whose spread no prompt-side predictor
   resolved. At 32B, 85% of steps keep ≥ 1.8 s after this event (97% keep ≥ 1.0 s); at 4B about half.
3. **`finished`**.

For a thinking-mode target the second event can be announced ahead of time: a per-position probe on the thinking
model's own final-layer states (`hazard-think-20260913/`, P(reasoning closes within X tokens), user's hazard
formulation) reaches AUROC 0.88–0.94 with recall ≈ 0.8 at precision ≥ 0.78 for X = 256 past 25% of the output, and
AUROC ≥ 0.94 with recall ≥ 0.83 for X = 64; the end of the whole step stays unpredictable in the middle of the output
(recall 0.42–0.46) because the visible tool call after `</think>` carries the old tail.

Thinking-mode check (Qwen3-30B-A3B-Thinking-2507-FP8, one sampled draw per prefix at temperature 0.6, 64K cap, 4,301
labels; `thinking-20260913/`): reasoning is 91% of the output at the median (672 reasoning vs 44 visible tokens), so
event 2 arrives when the step is essentially finished; the remainder by tool name scores q50 1.57 (visible constant
3.09). A probe on the Instruct-model features predicts the thinking total at q50 1.71 (constant 2.27) and the
reasoning length at 1.67 with MAE 1,080 tokens: for a thinking target the prompt-side estimate should quote the
visible part with a wide reasoning interval, and the early signal that matters is event 1.

The point estimate at `scheduled` (prefill + probe-predicted length × TPOT) stays available as the planning value:
q-err p50 1.60 / p90 4.02 at 32B DualMap with the leak-free probe head (pool prior 1.83 / 4.44; tool name alone
1.70 / 3.86). The natural-label lane with a thinking-mode target (Qwen3-30B-A3B-Thinking-2507, PENDING §8b) turns
event 2 into a real token event and adds reasoning length as its own target; no other experiment is needed for the
interface itself.

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
