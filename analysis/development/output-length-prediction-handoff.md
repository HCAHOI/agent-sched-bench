# Next-turn output-length prediction: handoff

Last updated: 2026-09-14. This handoff records the dated output-length experiments. The current reasoning-end development results and protocol are under "Reasoning-end development amendment" below; older machine and launch sections describe their historical runs. The current literature synthesis, mechanism findings and Phase 2 TODOs are in [REASONING.md](../../REASONING.md).

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

The initial thinking-model probe (`hazard-think-20260913/`) predicts P(reasoning closes within X tokens). Its
progress-quartile scores included positions after reasoning had already closed and overstate advance-warning
performance. On the during-reasoning slice alone, the original 256-token head has recall 0.557 / precision 0.708;
its truncated-endpoint labels are corrected in the 2026-09-14 experiment below. Instruct has a near-end signal
inside tool calls (`hazard-instruct-20260913/evaluation-full.json`): total-within-64 pooled recall 0.820 / precision
0.839, falling to 0.530 / 0.558 on long outputs. These are offline feature-probe results, not an online timing guarantee.

The old thinking restore-policy comparison (`hazard-think-20260913/policy.json`, restore 1.8 s ≈ 60 tokens) reduces
conditional median waiting from about 30 to 11 s, but 19.2% of requests never trigger and ready-in-time falls from
94.5% to 74.3%. Thus "one extra point of lateness" alone was incomplete. For Instruct, even a first-token trigger is
late in 47.5% of requests under this conversion; the earlier `scheduled` event provides the relevant prefill slack.

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

### Measured tables behind the three signals (from the four replay runs; `probe-replay-pool64v4/staged-*.json`)

| Run | Stage-1 wait p50 / p90 (s) | Remaining at scheduling p10 / p50 (s) | Share ≥ 1.8 s (≥ 3 s) | Stage-2 q-err p50 / p90 | Stage-2 + probe | Stage-3 (tool name) |
|---|---|---|---:|---|---|---|
| 32B DualMap 96 GiB | 3.2 / 12.9 | 3.7 / 9.6 | 1.00 (0.94) | 1.83 / 4.44 | 1.60 / 4.02 | 1.70 / 3.86 |
| 32B FCFS sticky | 36.0 / 56.7 | 4.5 / 22.8 | 1.00 (0.95) | 1.87 / 4.90 | 1.66 / 4.28 | 1.73 / 4.13 |
| 4B FCFS full | 3.1 / 6.3 | 1.2 / 3.2 | 0.76 (0.53) | 1.82 / 4.47 | 1.61 / 4.07 | 1.70 / 3.82 |
| 4B DualMap capped | 1.7 / 6.2 | 1.4 / 3.6 | 0.82 (0.58) | 1.79 / 4.52 | 1.57 / 4.10 | 1.70 / 3.81 |

| Run | Reasoning share of output tokens p50 / mean | Remaining after `reasoning closed` p10 / p50 (s) | Share ≥ 1.0 / 1.8 s after it | After-alert abs error p50 / p90 (s) | q-err p50 / p90 |
|---|---|---|---|---|---|
| 32B DualMap 96 GiB | 0.36 / 0.41 | 1.6 / 5.3 | 0.97 / 0.85 | 2.5 / 14.8 (stage 2 probe: 5.1 / 25.8) | 1.84 / 4.49 |
| 32B FCFS sticky | same traces | 1.7 / 9.5 | 0.98 / 0.89 | 5.4 / 31.5 | 1.99 / 5.12 |
| 4B FCFS full | same traces | 0.5 / 1.8 | 0.66 / 0.49 | 0.8 / 4.9 | 1.83 / 4.39 |
| 4B DualMap capped | same traces | 0.6 / 2.0 | 0.68 / 0.53 | 0.9 / 5.4 | 1.84 / 4.37 |

The prefill-only lower bound holds in 100% of steps (a bound that adds the 10th-percentile output length fails
14%). Stage-1 wait is the LLM side's own decision and is sent as the `scheduled` event, not predicted. Restore
times are the collaborator's: p50 1.0 s, p90 1.8 s.

## State on 2026-09-14

Closed with a diagnosis: session-history predictors, SSJF-Reg, EGTP (lose to the constant); the shallow probe
(wins the median, not the tail); tail-oriented heads, fused layers, re-prediction after k tokens, OUTLETS-agent
(all within seed noise of the probe on the tail; the point-predictor line is closed as a prompt-side hidden-state
limit while the sampling ceiling shows the information exists). Kept: the three-signal interface with the prefill
bound; the thinking-tier hazard alert (`hazard-think-20260913/`, `hazard-instruct-20260913/`) as an optimisation for
long thinking steps; the long-step and slow-tool alerts as static rankings. Not started, by decision: OUTLETS proper
(draft model with completion-side supervision), distributional multi-draw retraining (B, stopped), whole-prompt
pooled features (C). The reasoning-end repair, short-history comparison, and full-observation follow-up are complete
below. Filling 1,183 existing completions removes the frozen single-trigger heads' missing alerts. Full-data refitting
then gives current/history test recall 41.77/47.96% at validation precision 70%, with achieved test precision
69.32/69.01%. History/single reduces offline mean waste by 4.35 s versus current/single, with similar lateness, but
only 36.30% of first alerts fall within the target 256-token window. The subsequent user-authorized exploration
keeps history/single as its reference and improves the validation-selected candidate to a remaining-length MLP
with four closing-token signals and EWMA 0.7: development-test within-256 44.83%, mean waste 24.64 s, lateness
6.25% (baseline 5.77%). Ridge, eight-position remaining/progress heads, and closing-signal-only heads fail the
validation dual constraint. Middle-layer correctness/cost preflight passes on eight real requests; the user
authorized full layer-24 extraction and evaluation. The first partial run was stopped and deleted at the user's
request. Disabling HTTP connection reuse passed the 128-request control and the restarted full 4,215-request run
with zero failures (~102 minutes, 8.9 GB). Full feature and endpoint audits pass. Layer 24 loses to the retained
final-layer candidate: validation within-256 26.97% versus 43.68%; development-test 33.41% versus 44.83%, with
more early alerts and greater waste. Stop layer expansion after this negative primary. Full OUTLETS remains conditional.
No new responses or online integration were run.

## Current limitations and open problems

### Reasoning-end development amendment (2026-09-14, before corrected-run results)

The user authorized two sequential comparisons using the existing thinking cache: repair the current MLP's training
target, then test short causal history. The old test results above were already visible; this is development on an
exposed split, not fresh confirmation. The older reasoning progress-quartile scores include positions after
`</think>`; during reasoning alone, the old 256-token head has recall 0.557 and precision 0.708. Its old labels also
substitute total output length when the closing token lies beyond the 2,048-token extraction window. Those scores
are not a valid reference for advance alerts until evaluated on corrected full-completion endpoints.

Entry point: `scripts/evaluation/reasoning_end_hazard.py`. Reconstruct full rendered completions with the original
tokenizer/template and check exact alignment with the cached truncated completion. Keep only states strictly before
the closing token (endpoint = tokens consumed including `</think>`), with equal per-request weight in normalization,
training BCE, and validation BCE. Decode-position precision/recall retain the old pooled-position metric; also report
request-balanced metrics. This clarification follows the plumbing smoke, before full-data corrected results: equal
training influence must not silently change the precision-70 comparison. Prefill states can train the head but cannot
trigger an in-decode alert. Retain original task-grouped splits. Real-data alignment found one truncated Unicode
character changed the last cached token; retain only the exactly matching prefix and record all tail mismatches.

Primary comparison: frozen old head recalibrated on corrected validation labels versus a corrected 256-wide MLP,
seed 42, 30 epochs, validation-BCE checkpoint selection, AdamW lr 0.001 / weight decay 0.01. Target: reasoning closes
within 256 tokens. Select the threshold maximizing pooled-position validation recall at precision >= 0.70; use
complete tied-score groups, or no alert if the target is unattainable. The next comparison adds log consumed tokens
and the current state's difference from the mean of up to three preceding cached states, with width 128 to match
parameter count. No feature extraction, new generations, or test-based hyperparameter choices in these two rounds.

For each model, compare one crossing and three consecutive crossings, with validation-selected thresholds for each.
Score first alerts per request, lead to reasoning and whole-output end, coverage including never-alert requests,
late fraction, and waste. For an explicit accounting fallback, never-alert requests restore at whole-output end,
incurring the entire restore delay. Keep the prior 1.8 s restore / 0.03 s per token conversion for comparability;
report token leads and label seconds as offline estimates, excluding unmeasured online predictor overhead. The
2,048-token cache still limits late observation; correcting labels does not invent missing states. Compare paired
results clustered by underlying task (2,000 bootstrap draws, seed 42, descriptive 95% intervals; about 50 draws per
tail, sufficient for the percentage-point comparison, not a fresh-confirmation gate). Interpret recall together
with achieved precision and first-alert costs.

Real-data alignment and a short training timing check precede formal runs. Run outputs are new directories under
`/workspace/outlen/results-reasoning-20260914/`; original caches, checkpoints and result files remain frozen.

#### Results (2026-09-14 07:49 UTC)

Local artifacts: `analysis/results/output-length-source-labels-crossbench-20260904/reasoning-end-20260914/`.
`comparison.json` contains all validation/test metrics, paired intervals and cache-coverage diagnosis; each model
directory contains frozen thresholds, scores, request-level alerts and training metadata. Full-endpoint alignment
retained all 4,215 cached requests (2,964 / 419 / 832), with one mismatched cache-tail token and no request exclusions.
There are 1,183 endpoints beyond the cache; 215 of 832 test requests have no cached position in the final 256 tokens.
The matched prefix mask did not remove a sampled state in that one Unicode case (the stride skipped that token).

Both new heads selected epoch 2 by validation BCE; current/history have 524,801 / 524,673 parameters. Training
uses 743,612 pre-end positions; decode evaluation uses the identical 198,680 test positions for every model. Total
run times, including feature loading and evaluation: legacy 8.9 s, corrected current 59.7 s, history 94.6 s.

| Model, single crossing | Validation precision / recall | Test precision / recall | Recall change versus corrected current |
|---|---|---|---|
| Frozen old head, recalibrated | 70.01% / 54.93% | 70.43% / 56.13% | +0.63 pp |
| Corrected current-state MLP | 70.00% / 54.50% | 69.92% / 55.50% | reference |
| Short-history MLP | 70.01% / 59.13% | 69.21% / 60.41% | +4.91 pp, paired 95% interval [4.37, 5.43] |

Repairing training alone does not improve recall (versus legacy: −0.63 pp, interval [−1.37, +0.11]). Short history
adds information at this operating point, with test precision 0.70 pp lower than the corrected current head; these
are achieved test precisions at validation-selected thresholds, not test-retuned precision-70 scores. Intervals
describe task variation at seed 42, not training-seed variation or fresh confirmation. The two added input types
(elapsed tokens and hidden-state change) were tested together, so the gain cannot be assigned to either alone.

All following denominators include every test request. "Within 256" means the *first* alert falls in the last
256 tokens before reasoning ends. Never-alert requests use the explicitly charged end-of-output fallback. Waste
means are over all requests, with zero waste when late/never. Seconds assume 0.03 s/token and 1.8 s restore.

| Model / trigger | First alert within 256 | Never alert | Late including never | Mean waste (s) | Mean stall with fallback (s) | Median lead to reasoning / output end, fired only (s) |
|---|---:|---:|---:|---:|---:|---|
| Old / single | 32.69% | 12.02% | 17.91% | 31.689 | 0.236 | 10.68 / 17.715 |
| Corrected / single | 31.25% | 11.66% | 17.43% | 32.719 | 0.229 | 10.80 / 18.990 |
| History / single | 33.41% | 13.94% | 19.71% | 28.122 | 0.271 | 10.05 / 17.490 |
| Old / three | 37.98% | 19.35% | 26.20% | 20.229 | 0.379 | 8.01 / 13.980 |
| Corrected / three | 36.66% | 19.35% | 26.20% | 21.037 | 0.379 | 8.28 / 14.670 |
| History / three | 36.78% | 18.51% | 25.48% | 22.376 | 0.365 | 8.49 / 15.000 |

History/single versus corrected/single cuts mean waste 4.60 s (interval [−7.43, −1.90]) but increases never-alerts
from 97 to 116 requests and late-including-never by 2.28 pp. Every single-trigger never-alert request, for all three
models, belongs to the 215 requests lacking the final-256 cache window. Fired-but-late counts are 48 for both new
heads; the net ready-in-time decrease comes entirely from the extra never-alerts. This is a measured limitation
of the truncated-observation policy, not proof that the history model would miss those requests if observed later.
Three consecutive crossings further reduce early waste but increase missed/late requests; they are not a default
improvement. History/three versus history/single: mean waste −5.75 s, late +5.77 pp, stall +0.094 s/request.

Decision: keep corrected labels/training and short history as a candidate; do not claim a finished restore-policy
improvement. The next evidence needed for that decision is the missing later states of the existing long reasoning
completions, not new responses. No extraction or `</think>` probability feature was launched in that initial
comparison. Its results do not establish whether a score-history feature or EOS-probability feature would help.

Verification: three focused regression tests and an independent bounded review passed. Review caught and fixed
the exact-60-token floating-point lateness boundary. Real-data preparation caught the truncated-Unicode token
mismatch; its prefix-mask fix was re-reviewed before model runs. No serving/evaluation baseline code was changed.

#### Full-observation follow-up (authorized 2026-09-14, before extended-feature verdicts)

The user authorized continuing on `ssh -p 35803 root@connect.singapore-a.gpuhub.com`. Primary question: how much of
the single-alert missingness is caused by the 2,048-token observation cutoff? Hold the three existing heads and
their previously selected validation thresholds fixed; add the missing observations and score the identical 832
test requests, including never alerts. This isolates observation coverage from retraining or recalibration. Compare
all prior first-alert measures and position precision/recall, with the same task-clustered uncertainty and offline
time conversion. Further fitting is conditional on a remaining predictor failure; no new responses are generated.

Supplement exactly the 1,183 existing requests whose full reasoning endpoint exceeds the cached window, across
the original train/validation/test splits. Existing `outlets_agent.py extract-hazard` reads the same completed
answers and tokenizer snapshot, with completion cap 65,536 (largest actual rendered completion 17,513), stride 4,
final-layer token embeddings, activation disabled. Model: Qwen3-30B-A3B-Thinking-2507-FP8 snapshot
`60d80c83c53c3b611c642dbb8c942b3f90c5948a`; vLLM 0.28.0, same bf16 KV, pooling conversion, max context 163,840,
max sequences 16, GPU memory fraction 0.90, GPU 0. Max required input 90,388 tokens. Extraction input is 18.63M
tokens versus 76.77M in the previous 109-minute extraction; initial estimate 25–35 min plus startup/validation,
new files about 6.4 GB against 25 GB free. Smoke the longest input and longest reasoning before launching the full
set. Reuse successful smoke feature files only under the identical protocol.

Merge only later positions into the original cache in memory, retaining every original sampled state exactly.
Require the supplement request-ID set, prompt length, reasoning endpoint, output endpoint and split to match the
frozen full-endpoint manifest; require a complete grid `[0, 1, 5, 9, ... < reasoning_end]` for every request.
Original caches, heads and previous results stay frozen. New run root: `/workspace/outlen/results-reasoning-full-20260914/`.
The changed loader and frozen-checkpoint/threshold branches passed four regression tests and independent review
before use; extraction code is unchanged. No `</think>` probability feature is added in this comparison.

Amendment after feature audit, before any full-observation prediction verdicts: all 1,183 requests were recovered
(13 connection failures succeeded on one identical-protocol retry), and all 4,215 position grids/endpoints pass.
However, 239 supplement requests have non-identical overlap states relative to the old cache; most states remain
close, but the worst per-position cosine is 0.521. A repeat of the worst validation request gave bit-identical
states across current full-batch, full-single, and truncated-single extraction, so extending the answer did not
change its earlier states in that control. Old/current server logs agree on model, dtype, attention/MoE backends,
and chunk budget; the historical drift's exact source remains unresolved. Original states remain fixed in the
primary analysis. Add one sensitivity comparison for current/history: use this extraction's entire state sequence
for the 1,183 supplemented requests (including overlap), with the same frozen heads/thresholds and all requests.
This checks whether the result depends on overlap drift or the history feature at the cache join; it is not a
replacement dataset or a new fit. Preserve both outcomes and the audit/repeat records.

Conditional fitting amendment (2026-09-14, after frozen full-observation verdicts): single-trigger never-alerts
became zero for all three heads. Full validation precision is only 63.56/64.18/66.43% for legacy/current/history,
and test first-alert-within-256 is 33.41/32.21/35.34%; thus coverage is repaired but the target operating point and
early alerts remain unresolved. First recalibrate all existing heads on full validation at the unchanged 70%
precision floor. Then refit current/history on all 1,359,019 train positions using the original 30 epochs, seed 42,
request-balanced normalization/BCE, validation-BCE checkpoint choice and parameter counts. No tuning sweep or new
features. Compare refit versus the same frozen architecture recalibrated on full validation, and history versus
current at that operating point; retain all first-alert costs. Estimated 3–5 minutes for these small heads. Existing
full-test results are now development-exposed; this is a dated amendment, not fresh confirmation.

#### Full-observation results (2026-09-14 08:56 UTC)

Local artifacts: `analysis/results/output-length-source-labels-crossbench-20260904/reasoning-full-20260914/`.
`comparison.json` contains the paired task intervals, coverage diagnosis and overlap sensitivity; model directories
contain thresholds, scores and all request-level alerts. The 6.38 GB feature supplement remains at the declared
remote root, with its index and full audit copied locally. All 1,183 requests succeeded, including 13 recovered
connection failures. Added 883,197 pre-end states: train/validation/test = 615,407 / 86,918 / 180,872. Every one of
4,215 requests has the full stride-4 pre-end grid; test has 379,552 decode positions and 45,006 positives.

**Coverage, with heads and thresholds frozen.** All three single-trigger heads now alert on all 832 test requests.
Current/history never-alert counts fall 97/116 → 0/0; late-including-never falls 17.43/19.71% → 5.77/6.01%.
All previously fired first alerts are unchanged. But only 8/97 and 16/116 recovered alerts fall within the final
256 tokens; their median reasoning leads are 1,147 and 841 tokens. Coverage explains the old missingness, while
premature first alerts remain a separate failure. Full validation precision falls to 64.18/66.43%, so those frozen
thresholds do not represent the intended precision-70 operating point.

**Numerical check.** The overlap drift described in the amendment remains unresolved at the kernel/source level.
Replacing all overlap states for supplemented requests changes only 2–5 first-alert locations per model/policy,
no never-alert/late/within-256 counts, and at most 0.086 s mean waste. Primary old-position probabilities differ
from their previous saved values by less than 5e-7; thresholds are identical. This supports the coverage conclusion
under the checked sensitivity, without asserting bitwise reproduction of every historical feature.

**Refit at the same validation rule.** Current/history select epochs 2/3 by request-balanced validation BCE, with
524,801/524,673 parameters and 1,359,019 train positions. Runs take 87.0/146.8 s including loading and evaluation.
Every row below selects its threshold on full validation only; test is not recalibrated.

| Model, single trigger | Validation precision / recall | Test precision / recall |
|---|---|---|
| Legacy head, full-validation recalibration | 70.03% / 40.46% | 69.59% / 39.89% |
| Previous current head, recalibrated | 70.00% / 39.60% | 69.87% / 39.52% |
| Previous history head, recalibrated | 70.01% / 45.13% | 69.73% / 44.54% |
| Current MLP, full-data refit | 70.01% / 41.91% | 69.32% / 41.77% |
| History MLP, full-data refit | 70.02% / 48.86% | 69.01% / 47.96% |

Full-data refitting increases current/history recall by 2.25/3.43 pp versus their recalibrated previous heads
(paired 95% intervals [1.79, 2.72] / [3.01, 3.87]); achieved test precision falls 0.55/0.72 pp. Refit history versus
refit current gains 6.19 pp recall [5.50, 6.85], with test precision 0.31 pp lower. These are task-variation intervals
from 2,000 paired task bootstrap draws across 174 tasks at one model seed, not independent confirmation.

| Full-data model / trigger | First alert within 256 | Never alert | Late including never | Mean waste (s) | Mean stall (s) | Median reasoning / output lead (s), fired only |
|---|---:|---:|---:|---:|---:|---|
| Current / single | 36.90% | 0/832 | 5.89% | 35.286 | 0.0208 | 10.665 / 18.075 |
| History / single | 36.30% | 1/832 | 5.77% | 30.932 | 0.0211 | 10.980 / 17.670 |
| Current / three | 44.83% | 16/832 | 9.50% | 23.659 | 0.0689 | 8.520 / 14.985 |
| History / three | 44.11% | 13/832 | 8.89% | 24.694 | 0.0614 | 8.580 / 15.240 |

History/single versus current/single cuts mean waste 4.35 s [−6.93, −1.81], with no clear change in lateness or
first-alert-within-256. Full refitting of history versus its old recalibrated version improves position recall but
does not clearly reduce waste (−0.12 s [−1.98, +1.52]); within-256 first alerts decrease 2.28 pp [−4.53, −0.12].
Thus better position recall does not establish better alert placement. History/three versus history/single cuts
waste another 6.24 s [−7.91, −4.65], but increases late requests by 3.125 pp [1.92, 4.43] and mean stall by 0.0403 s
[0.0258, 0.0569]. The three-position trigger uses its separately validation-selected threshold.

Decision: keep full endpoints/observations and history/single as the next development baseline; do not add a seed
sweep to rescue first-alert accuracy or default to three crossings. The remaining problem is premature first
alerts under repeated detection, not missing long-completion observations. Score-history or closing-token
probability remains an untested next feature, rather than a demonstrated fix. Seconds still assume 0.03 s/token
and 1.8 s restore, include no-alert fallback, and exclude online prediction/serving overhead. No actual sandbox
restore improvement has been measured. Four regression tests and the independent code review passed before
scientific use; the dedicated pooling server is stopped and all head runs are finished.

### Sequential first-alert exploration (2026-09-14)

The user authorized five ordered stages: fixed-head policies, log remaining-length targets, relative progress with
eight sampled states, direct closing-token signals (then selected middle layers if needed), and conditional full
OUTLETS. Existing test is development-exposed. Selection uses validation only: lateness and mean waste must not
exceed the original full-data history/single baseline on that same validation split; maximize within-256 first
alerts, then minimize waste and lateness. Keep no-alert requests and the original offline time assumptions.

Stage 1 fixes the existing head and compares single, three crossings, EWMA coefficients 0.1/0.3/0.7, and empirical
per-request maximum-early-score thresholds. Validation selects EWMA 0.7: 140/419 within-256, 18/419 late, 27.463 s
waste versus baseline 136/419, 18/419, 28.542 s. Development test gives 306/832 within-256, 49/832 late, 29.797 s
waste versus 302/832, 48/832, 30.932 s. Paired task-bootstrap deltas: within +0.48 pp [-0.84, 1.81], lateness
+0.12 pp [0.00, 0.37], waste -1.135 s [-2.207, -0.220]. Three crossings has no threshold satisfying both validation
constraints. This limited gain does not justify expanding the smoothing grid. Maxwise calibration is empirical;
validation tuning provides no conformal guarantee. Frozen artifacts: `reasoning-explore-20260914/policy/`.

Before stage 2/3 results: ridge uses history inputs, request-weighted squared error, an unpenalized intercept and
validation-selected alpha from {0.1, 1, 10, 100}. Remaining-length MLP keeps the 524,673-parameter history head.
Both predict train-min/max-scaled log(1 + remaining tokens). Eight-position models use a depthwise temporal
convolution and matched ~525K-parameter head; compare direct remaining length with sigmoid relative remaining
progress (SmoothL1), then causal prefix least-squares extrapolation with intercept one. These are target/representation
adaptations, not faithful DyCon or Fuel Gauge reproductions; stride-4 samples are not consecutive tokens. All MLPs
retain seed 42, 30 epochs, AdamW 0.001/0.01, batch 1024, request weighting and validation-loss checkpoint selection.
The same finite policy family is applied to each model; no test tuning or seed search. The full-data one-epoch
remaining-length plumbing check took 116 s, with ~1 s training; it is not a scientific result. Seven focused tests
and independent scoped reviews passed before formal stage 1 and stage 2/3 use.

Stage 2: ridge selects alpha 10 but has no feasible alert policy (minimum validation waste under the lateness cap
is 30.102 s). The history remaining-length MLP selects epoch 21 and single crossing at log-remaining <= 5.223106:
validation within/late/waste = 34.84% / 4.30% / 27.200 s. Development test = 39.78% / 5.77% / 28.086 s, with no
never alerts. Paired deltas versus original history/single: within +3.49 pp [1.09, 5.93], lateness 0.00 pp
[-0.36, 0.36], waste -2.846 s [-5.038, -0.965]. Auxiliary single-trigger precision/recall = 70.99% / 45.27%.
Retain this candidate: position recall alone would have missed the improved first-alert utility. Fits took
145.5 s (ridge) and 138.7 s (MLP), including loading. The eight-position remaining-length comparator has no feasible
validation policy (minimum waste 30.789 s), so do not expand its architecture; finish the predeclared progress
target comparison to distinguish target from representation effects.

Before stage 4 learned-head results: source inspection confirms Qwen3MoE final RMSNorm, token_embed with
use_activation=false and no sentence-transformer projector, untied BF16 lm_head excluded from FP8 quantization,
and unscaled logits. Project existing states without another norm; retain log P(</think>) and negative log rank,
plus differences from the preceding three cached positions. These are raw model logits before sampling filters,
not temperature/top-p generation probabilities, and inherit the historical cache drift. Compare a 32-wide binary
MLP on these four scalars alone with adding the four scalars to the strongest validation-selected model from
stages 1-3, retaining its target and width. Keep the same 30-epoch optimizer/selection protocol and finite policy
family. No intermediate layers are extracted at this stage. A 4,096-position projection check took 0.0415 s
(~98.7K positions/s); complete projection is ~20 s plus input loading, and compact signals need <40 MB. The eighth
focused test verifies projection probability/rank, alignment, and causal request-local deltas. Independent review
caught binary logits versus probability EWMA incompatibility; sigmoid score export was corrected and re-reviewed
before any stage 4 learned-head result.

Stage 3: eight-position remaining/progress select epochs 4/1, take 130.8/128.6 s, and both fail the validation
dual constraint. The progress model requires at least 53.378 s mean waste under the lateness cap. At its natural
256-token diagnostic threshold, validation has 44/419 never alerts and 15.75% late; 52/419 requests reach an
extrapolated zero remaining length while actually more than 256 tokens remain. This is inaccurate trajectory
extrapolation, not an observation-coverage failure. No parameter or seed expansion follows the negative result.

Stage 4: full LM-head projection took 43.17 s including loading (15.96 s projection, 1,929,234 cached positions).
Probability >=0.1 occurs at 93 validation positions, all in the final four tokens; none occur with 5-256 tokens
left. The signals-only head (193 parameters, epoch 29, 59.1 s) cannot meet the dual constraint (minimum validation
waste 55.338 s); its validation-P70 test precision/recall is 79.88% / 1.20%. Direct strong closing-token signals
are too late for this advance-warning objective, although the four continuous scalars can help a hidden-state head.

The remaining-length history head with four signals has 525,185 parameters, selects epoch 12, and takes 161.5 s.
Validation selects EWMA 0.7, threshold -5.3992874885. All choices below were frozen before test scoring:

| Candidate | Validation within / late / waste | Development-test within / late / waste | Test never |
|---|---|---|---:|
| Original history/single | 32.46% / 4.30% / 28.542 s | 36.30% / 5.77% / 30.932 s | 1/832 |
| Fixed head, selected EWMA 0.7 | 33.41% / 4.30% / 27.463 s | 36.78% / 5.89% / 29.797 s | 1/832 |
| Log-remaining MLP, selected single | 34.84% / 4.30% / 27.200 s | 39.78% / 5.77% / 28.086 s | 0/832 |
| Log-remaining + signals, selected EWMA 0.7 | 43.68% / 4.30% / 23.090 s | 44.83% / 6.25% / 24.639 s | 1/832 |

The last candidate's paired deltas versus original history/single are within +8.53 pp [6.13, 11.11], lateness
+0.48 pp [-0.003, 1.083], and waste -6.292 s [-8.434, -4.330]. Versus the remaining-length MLP, they are
+5.05 pp [3.09, 7.14], +0.48 pp [0.12, 0.97], and -3.447 s [-4.857, -2.207]. These compare complete selected
pipelines, including their validation-selected policy; they do not identify the contribution of each scalar or
separate training-seed variation. The candidate satisfies the prescribed validation constraint, but its four
additional late test requests prevent a claim of demonstrated no-lateness-regression. Do not retune on this test.
Its auxiliary validation-P70 test precision/recall is 69.60% / 49.32%. Median reasoning/output lead is 8.70/15.69 s
among fired requests; average fallback-inclusive stall is 0.02326 s. Seconds remain offline assumptions.

On the fixed test length groups, combined-head within/waste/late is 53.81% / 15.207 s / 7.95% for 604 requests
with reasoning <=2048, and 21.05% / 49.626 s / 1.75% for 228 longer requests. Thus important early-alert loss
remains on long reasoning. Keep the validation winner as a development candidate, not an independently confirmed
restore policy. Artifacts, checkpoints, score trajectories, protocols and task-paired bootstrap outputs are under
`analysis/results/output-length-source-labels-crossbench-20260904/reasoning-explore-20260914/`, mirrored from
`/workspace/outlen/results-reasoning-explore-20260914/`. Nine focused tests and the scoped independent reviews pass.

Middle-layer preflight: fix the next primary layer at residual stream after block 24 (of 48); block 36 is a
subsequent candidate only if the first comparison warrants it. The existing vLLM auxiliary-state wrapper now
supports auxiliary-only output to avoid storing another final-state copy. Its previously hidden forward signature
prevented vLLM from marking token dimensions dynamic and caused an 8192-versus-32 compiled-shape failure. Restoring
the signature with functools.wraps fixes the root cause; a dedicated regression and independent review pass.
Default concatenation remains available. The successful smoke retained torch.compile and CUDA graphs.

The longest input and longest reasoning both succeed (116,602 total input tokens, 27.12 s); layer width is 2048,
states are finite, complete stride-4 grids and full endpoints match, and the returned raw residual states differ
from the final-normalized cache. A seed-42 request from each of six prompt-length sextiles also passes, using
126,545 input tokens in 18.41 s. This is plumbing/cost evidence only. Scaling this small batch to the fixed 4,215
requests / 80,633,930 input tokens gives ~3.26 hours; prefix-cache reuse, batch scale and startup limit that estimate.
The 2,172,124 stored output positions require ~8.90 GB FP16 features, against ~19 GB free before extraction.

The full layer-24 extraction and evaluation authorized on 2026-09-14 are complete under `middle24/`. The comparison
uses the retained history remaining-length + scalar-signals setup, original labels, splits, seed, training and
validation policy rules. All 4,215 requests, 2,172,124 stored states and 1,929,234 pre-end states pass the finite-value,
2048-dimensional and exact-position audit. Existing endpoint preparation with cache-window 65536 retains all requests
without exclusions, tail mismatches or missing endpoints. Canonical request order was restored before joining the
unchanged final-head scalar signals; original labels and task splits match exactly. No new completions were generated.
The eight-request audits, timings and smoke server configuration are saved under `middle24-smoke/`. The formal
pooling service and extractor use the private Python Supervisor configuration in `middle24/`. At the user's
instruction, the first 183-record partial feature directory was deleted after three connection resets; it will
not be reused. Both extraction callers now disable persistent HTTP connections, cancel pending work and close the
client on a transport failure, then exit nonzero. Already-running workers must unwind before process exit; no
automatic retry is introduced. Eleven focused tests and independent review pass. A real 128-request old-client
control reproduced a reset on a reused connection while receiving response headers (33 new TCP connections for
128 requests). The fixed client passed all 128 requests using 128 fresh TCP connections, with finite, aligned
2048-dimensional features; elapsed time was 195.3 s versus 197.5 s for the old client. This identifies connection
reuse as the observed failure path; expiry timing itself was not established by packet capture. The diagnostic
saved no features. Its receipt is `middle24/http-transport-validation.json`. The full extractor restarted from
`0 cached, 4215 to extract` and finished with 4,215 successes and zero rejections in about 102 minutes. The pooling
service is stopped. Training took 176.1 s including loading and scoring, retained 525,185 parameters and selected
epoch 16 of 30 by validation loss. Policy selection chose EWMA 0.3 under the original baseline's validation constraints.

| Development comparison | Validation within-256 / late / waste | Test within-256 / late / waste |
|---|---|---|
| Original history/single baseline | 32.46% / 4.30% / 28.542 s | 36.30% / 5.77% / 30.932 s |
| Retained final-layer remaining + signals | 43.68% / 4.30% / 23.090 s | 44.83% / 6.25% / 24.639 s |
| Layer-24 remaining + same signals | 26.97% / 4.30% / 28.531 s | 33.41% / 6.49% / 28.093 s |

Layer 24 passes the baseline feasibility constraint but is dominated by the retained candidate on validation.
Against the retained candidate on development test, task-paired bootstrap gives within-256 -11.42 pp
[-14.06, -8.94], lateness +0.24 pp [-0.36, 0.96], and mean waste +3.454 s [2.169, 4.790]. Early first alerts
increase from 458 to 550, within-window alerts fall from 373 to 278, and never alerts increase from one to four.
For reasoning >2048, within-256 falls from 21.05% to 15.79%, with waste increasing from 49.626 to 57.289 s.

The failure is at the constrained decision point: layer 24's validation precision-70 threshold reaches 53.57%
position recall but gives 7.88% lateness, above the 4.30% cap. Meeting the cap shifts alerts earlier, yielding only
113/419 within-window alerts versus 183/419 for the retained candidate. Its auxiliary test precision/recall is
70.79% / 51.02% versus 69.60% / 49.32%; this modest position-level gain does not improve first-alert utility.
Keep the final-layer candidate and stop layer/seed/threshold expansion after the negative primary. This compares
the complete pipelines, including their separately validation-selected rules; it does not isolate layer effects
from all extraction numerics, and it does not establish that every intermediate layer is uninformative.
Frozen checkpoints, score trajectories, policies, full audits and paired diagnosis are in `middle24/` locally and
remotely; full feature arrays remain on the GPU host. Existing test remains development-exposed and all seconds
retain the 0.03 s/token and 1.8 s restore assumptions. Full OUTLETS and independent real-restore confirmation are
not launched; the long-reasoning loss remains open.

The user also authorized unused-weight cleanup. Removed only the Qwen3-4B-Instruct-2507-FP8 pretrained weight
blob (revision 8591804019c8b22094c3b5b4454e0edc05dffc98), releasing 5,190,053,264 bytes after checking process
references and blob sharing. Its tokenizer/configuration remain; historical 4B inference commands require
re-downloading the weights at that revision. Current Thinking, recent TPOT model weights, learned heads,
features and results are retained. Cleanup receipt is in `middle24/protocol.json`; free space rose to ~23 GiB.

### Reasoning-process diagnosis and final-FFN readout (2026-09-14)

Following the user's direction to investigate reasoning mechanisms, the next phase reuses existing answers and
the frozen final-layer predictor. No predictor is fitted and no new continuation is generated. All observations
below use development-exposed validation; they do not establish a forecasting improvement.

Text alignment covers all 419 validation requests. Of 183 within-window alerts, 83 come from reasoning of at most
256 tokens. Among the other 336 requests, only 100 alerts hit the window (29.76%); 236 are early. Twenty early
alerts occur at the first generated token, and 99/236 occur by token 256. The first-token cases still have the
entire input context available, but cannot reflect progress within the current reasoning. Paragraph delimiters
are descriptive boundaries, not verified cognitive steps.

A fixed, stratified 24-request sample was selected before reading its text. Unblinded qualitative inspection
finds both premature local decisions followed by repeated command/code elaboration and provisional solutions
that are subsequently revised. Correct alerts also accompany repeated intentions. These examples motivate
distinguishing action readiness, reasoning progress and natural termination; they do not justify keyword rules
or estimates of how common each reasoning pattern is. A later quality check joined these 24 completions to
their original input histories: some final recaps have supporting test output, while one completion invents
permission to stop after a full-suite timeout. Static checks on all 419 validation completions found 346
schema-valid tool calls, but an invalid notebook JSON payload and a shell quoting defect. Generated tools
were not executed, so this is not a task-success estimate. A separate exact-32-token-suffix control yields ten early/late pairs, mostly repeated
code or structured text. Closing-token log probability rises in only four pairs; this is not a clean test of
semantic convergence and does not support that claim.

The model diagnostic captures the final MoE block's pre-FFN residual, FFN update and native normalized output
at the same 24 requests' alert and at leads 256, 64 and 1 (96 existing-answer prefixes). True endpoints only
choose diagnostic observations. Removing the final FFN update, then applying the original RMSNorm and LM head,
tests its contribution to next-token readout. Reconstructing the intact output passes first: maximum relative
hidden-state error 0.0000243, minimum cosine 0.99999988, and maximum closing-token log-probability error zero.

| Diagnostic position | Median native P(close) | Median P(close), final FFN omitted | Requests where FFN raises log P(close) |
|---|---:|---:|---:|
| Frozen first alert | 1.77e-15 | 3.83e-11 | 0/24 |
| 256 tokens before close | 3.80e-15 | 8.58e-12 | 0/24 |
| 64 tokens before close | 2.58e-15 | 4.93e-11 | 0/24 |
| 1 token before close | 0.999978 | 0.992616 | 24/24 |

In this sample the final FFN suppresses immediate closing at the earlier observations and strengthens it just
before emission. Crucially, the residual-only readout already strongly favors closing at lead 1, so the final
FFN is not the sole source of that terminal signal. Sparse positions do not locate the transition or establish
that earlier representations lack useful information. This is whole-block readout ablation, not neuron
identification, a NEAT reproduction, or evidence about changed natural stopping time or answer quality.

The follow-up newline control uses 24 paired earlier/final prose prefixes from 22 tasks, at one/two trailing
newlines (96 reads, 52.6 s). Median closing probabilities are 0.997722/4.01e-15 for earlier prose and
0.999980/3.96e-14 for final prose. Reconstruction relative and closing-logP errors are zero. This shows strong
format dependence in the current model, not format-only behavior or a general reasoning clock: under the same
two-newline format, 18/24 final prefixes still have higher closing log probability. Saved validation reasoning
already ends in one newline in all 419 cases; raw generated token streams were not saved. These are controlled
prefix readouts, not natural continuation or cross-model evidence. Protocol, annotations and states are in
`reasoning-mechanism/phase2-boundaries/`; scoped independent review and real-model smoke passed.
Following the user's generalization concern, further component extraction along the newline branch is stopped.
Reflection-marker studies are conditional diagnostics; the user redirected the immediate priority to forecast
quality. A complete-history GRU comparison is now finished: identical cached inputs/target, seed 42, 30 epochs,
524,591 parameters vs MLP 525,185, plus an MLP with identical 8-request minibatches. Original MLP reproduction
has zero score difference on validation and development test. GRU reduces validation MSE by 19.9% but its selected
first-alert hit rate is 31.50% vs incumbent 43.68%; development test is 34.38% vs 44.83%, with waste 25.512 vs
24.639 s and late 5.89% vs 6.25%. Paired task within-256 difference vs incumbent is -10.46 pp [-13.13, -7.77].
Both heads cross threshold inside the target window on 418/419 validation requests, but GRU crosses too early
on 286 vs 236. It is not retained; stop this size/seed branch. Formal reproduction/control/GRU/policy runs took
953.9 s; 11 focused tests and independent review passed. Frozen outputs are in `reasoning-explore-20260914/gru-head/`.
See REASONING.md §4.6 for controls and precision-dependent inference checks. No attention-decomposition result or reflection predictor is claimed. Keep
the current forecasting candidate unchanged. Frozen protocol, annotations, paired states, runnable diagnostic
and 96 observations are under `reasoning-explore-20260914/reasoning-mechanism/`. Extraction took 41.9 s after
startup; the pooling service is stopped. Twelve focused tests, real-model reconstruction and independent review
of capture and evaluation logic pass.

### Gemma/DFlash first-boundary comparison (2026-09-14)

The fixed primary is complete: 32 task-distinct TPOT replay requests, seed 42, thinking enabled, Gemma4 26B-A4B
FP8 + existing DFlash, block size 16 / 15 speculative tokens, FP8 KV, concurrency 1. Intentional replay instructions
and full message histories are retained; no source replacement occurred. The real vLLM v2 capture reuses native
target copies before drafting and asynchronously records draft IDs. Captured and streamed tokens match, including
Gemma reasoning boundary 101 and tool stop 50; all effective prompt counts match the serving render endpoint.
Formal runtime 99.10 s, no request failures or capped primary outputs; independent review and focused tests pass.

For 25 schema-valid local sandbox tool requests, assuming 1.8 s restore, target vs first-draft rules give lateness
88% vs 80%, mean blocked 1.28112 vs 1.08412 s, idle 0.15746 vs 0.21755 s. Blocking reduction is 0.19700 s
(paired task 95% CI [0.06035, 0.37158]); increased idle fails the frozen gate. Of 14 advance alerts, only 6 correspond
to a boundary in the next 15 tokens (30–104 ms actual lead); all first candidate paths were rejected. Fifteen requests
emit the reasoning boundary immediately, leaving no draft lookahead opportunity. All 32 outcomes remain recorded:
27 tool_calls include one invalid timeout and one nonlocal message; five are final answers with unnecessary restores.
Across all 32, observable pre-finish idle lower bound rises 0.33287→0.54480 s at 1.8 s restore. No tools or restores were
executed. These are real generation event times plus explicit restore assumptions, not end-to-end sandbox results.

Keep the target-boundary control; stop expansion of the single-draft-marker rule. Next analyze whether sandbox is
needed and time until tool readiness using existing trajectories. See REASONING.md §4.7 for sensitivity, diagnosis
and current TODOs. Manifest, raw streams/events, configuration and summaries are mirrored under
`reasoning-explore-20260914/dflash-boundary/`; entrypoints are `dflash_boundary.py` and
`vllm_dflash_trace_sitecustomize.py`. The experiment server is stopped.

## Historical label-run notes

- Natural output has a heavy tail: one draw exceeded 16K even though the longest-prefix smoke outputs were all below 1.3K. The 32K cap is still a censoring boundary, so any cap hit invalidates that label run.
- One warm request is performed per prefix before its other 19 draws run concurrently. This improves prefix reuse but means only one data-parallel engine is active during that warm draw. Do not change this behavior mid-protocol; optimize it only as a separately declared revision.
- The node is ephemeral and the formal dataset currently exists only remotely. The source trace archives and code exist locally, so it can be rebuilt, but losing completed labels would waste substantial GPU time.
- The current remote environment was created by installing vLLM 0.11.2 into the image's environment, which replaced its original PyTorch stack and cost about an hour. Do not repeat this on a replacement node: prefer a compatible vLLM image or inspect the preinstalled PyTorch/vLLM compatibility before installing.
- EGTP's published default uses only the first four target-tokenizer tokens; the adapter preserves the official extractor after model-faithful full-prefix rendering. Interpret a weak result as a property of the published method/configuration, not as proof that richer hidden-state predictors cannot work.
- TIE predicts a distribution/risk quantity rather than a native point token length. The adapter's point reduction must be reported explicitly alongside q-error; distribution metrics should remain separate.
- OUTLETS cannot be evaluated faithfully on recorded labels: its official training regenerates completions with the target model and supervises completion-side hidden states. A ready official checkpoint has also not been confirmed.
- GitHub does not yet contain commit `0875daa`; pushing is a separate authorized action.

## Historical label-run commands

Inspect the original services:

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
