# Pending: experiment queue and open decisions

Current as of 2026-09-12 16:30 UTC. Rewritten, not appended: this file says
what is queued, why, and what each result decides. Records of finished work
live in the milestone files; this file only points at them.

## 0. What every run measures, and why always FCFS sticky and DualMap

The two policies are instruments, not the object of study. FCFS sticky is
the system as deployed with no help: each task pinned to one engine, vLLM's
own queue, no admission control, no DRAM tier. DualMap is the strongest
existing fix we can run: request-level admission at the proxy, a DRAM KV
tier, and migration. At any operating point (instances N, concurrency c =
number of tasks active at once in the closed loop, model size) two numbers
answer one question: the gap FCFS → DualMap says whether pressure is a
problem that an existing mechanism can fix; the residual DualMap → ideal
(engine time with no hold and no queue) says how much is left to research.

| Operating point | FCFS → DualMap gap | DualMap residual | Reading |
|---|---|---|---|
| 4B, full memory (R 0.5) | none | none | no problem |
| 4B, capped, c32–c64 (R 1.2–2.4) | large | small (hold 3–8 s of a 7 s step) | problem exists, solved by admission |
| 4B, N=8 | small (mean), large (makespan) | small | tail stranding only |
| 32B, full memory (R 1.3) | huge (57 → 30 min) | **large (hold 14 s of a 36 s step)** | problem exists, **not solved**: research goes here |

Chain 13 asks what the 32B residual is made of (DRAM-tier capacity, or the
admission logic) and whether the operating point itself is right (TP=2).

## 1. Running and queued on the GPU host (2× RTX Pro 6000)

All runs on `analysis/development/pool64-distinct-v4` (64 distinct traces).
Per-run analysis lands in `results/<run>/comparison.txt` and
`instance-balance.txt`.

| Order | Run | Purpose | Status |
|---|---|---|---|
| 1–4 | N=2 full and N=2 capped, FCFS sticky and DualMap | Step 2 reference and same-capacity control for N=8 | done (M4 §3) |
| 5–6 | N=8 capped (k=4 per GPU under MPS), FCFS sticky and DualMap | Step 2 decision point | done 19:01 UTC; verdict in M4 §3 |
| — | N=4 capped pair | optional middle point of the N axis | dropped 17:52 UTC, no result read; rerun only if N=8 is anomalous |
| 7–10 | N=2 capped at concurrency 48 and 64, FCFS sticky and DualMap | Step 3 pressure grid (below) | done 22:50 UTC; result in M4 §3 step 3 |
| 11–14 | Qwen3-32B-FP8: smoke, DualMap calibration, N=2 full FCFS sticky and DualMap at concurrency 32 | Model-size check (§5) | done 03:01 UTC; result in M4 §3.1 |
| 15 | 32B: DualMap with 96 GiB DRAM tier | §5 next run 1 | done 06:45 UTC; 23.35 vs 29.71 min, hold 14 → 9.9 s; M4 §3.1 |
| — | c16 floor, TP=2 smoke and run | §5 items 2–3, §7.1 | dropped 06:10 UTC (user): no decision depends on them; chain 13 is stopped after run 15 |

Chain 11 waits for the N=8 DualMap analysis, stops chain 10 before it
launches the N=4 pair, then runs the grid.

## 2. Step 3 pressure grid (pre-registered 17:56 UTC, before any grid number)

**Question.** Today's mechanism result (M4 §3): at R_avg 1.2 the winning
mechanism under KV pressure is proxy-side admission, not the DRAM tier. FCFS
sticky thrashes (cached share 0.96 → 0.56, prefill 7×, +46% JCT); DualMap
holds each request 2.7 s at the proxy regardless of pressure and recovers
nearly everything (9.40 vs 9.11 min at full). Does this hold at twice the
pressure? DualMap's hold is a fixed throttle; at higher pressure it either
throttles too little (thrash returns) or too much (engine idles).

**Configuration.** N=2 at 262,144 tokens per instance (524,288 total),
concurrency 48 (R_avg 1.77) and 64 (R_avg 2.36), FCFS sticky and DualMap,
same manifest, closed loop with replacement as in step 2. References: the
concurrency-32 capped pair.

**Expectation.** FCFS: cached share keeps falling (toward 0.3), JCT grows
faster than linearly in concurrency. DualMap: prediction is that it starts
to fail, its cached-token estimate degrades under high eviction rates and
hold time and in-engine queue rise together.

**Decision.** If DualMap at R 2.4 stays near the work-conserving bound (JCT
growth ≈ concurrency growth, cached share ≥ 0.9, in-engine queue < 1 s),
scheduling under KV pressure is closed on this workload and the direction in
§3 is not pursued. If it degrades, §3 becomes the method to build.

**Result (22:50 UTC, M4 §3 step 3).** DualMap held: 9.40 / 10.00 / 9.93 min
at c32 / c48 / c64, cached ≥ 0.90, queue 0.7 s; the hold adapts (2.7 → 8.1 s).
FCFS collapsed (cached 0.07, queue 23 s, 17.84 min). Scheduling under KV
pressure is closed on this workload; §3 is not pursued. One tail failure at
c48 (largest task held 23.6 min) did not recur at c64.

**Measured.** Per-request decomposition (proxy hold, in-engine queue,
prefill, decode) from `routing.jsonl` and `vllm-request-telemetry.jsonl`,
cached share, JCT with the paired bootstrap, LMCache retrieve counters.

## 3. Candidate direction: task-level working-set admission (literature checked 18:02 UTC)

Not a KV retention or prefetch mechanism (crowded: ThunderAgent, CacheWise,
SAGA, Continuum; and prefetch has nothing to recover here, a CPU-tier
retrieve costs 39 ms). Not dependent on tool-time or output-length prediction
(both measured as unreliable in this project).

Signal: an agent's context grows monotonically; prompt at step k+1 ≥ prompt
at step k plus output plus tool result, and the growth rate is measurable
from the task's own history. Decision: admit a task to an instance only if
the projected contexts of its resident tasks over the next few steps fit the
KV cache (multiprogramming-level control of the working-set model).

**Literature check (independent agent, sources read on arxiv pages):**
task-level admission on *current* KV occupancy is already published twice in
2026; only the forecast is unpublished.

| System | Unit | Admission signal | Forecasts context growth |
|---|---|---|---|
| ThunderAgent (arXiv 2602.13692) | LLM program | pause/restore on KV watermarks over Σ current context; pauses shortest programs first | no |
| KAIROS (arXiv 2604.16682) | agent | admit from pending set while Σ current context < 0.9 × capacity | no (objective is power; no scheduler baselines) |
| Continuum / CacheWise / SAGA | request | none (TTL or eviction by predicted tool duration) | no |
| Llumnix, Preble, Autellix, Parrot, Mooncake | request or program | none or request-level | no |
| ConServe (arXiv 2606.01839) | conversation | placement on observed occupancy; argues against prediction | no, deliberately |

Consequence for us: the published task-level admission is ThunderAgent,
which is already a baseline in this repo and lost to request-level
throttling (DualMap) by 2× on mean JCT on L40S with a 51.8 s mean outside-
engine wait and a 568-min worst task (Milestone 2). So "task-level" is not
by itself the right unit; a working-set controller would have to show that
growth-aware admission and victim choice fix ThunderAgent's starvation while
keeping DualMap's cache protection, against both as baselines. Thin margin;
the advisor's call.

Gate: §2 result. Closed 22:50 UTC for the 4B model (DualMap did not degrade at R 2.4). **Reopened 03:02 UTC at 32B**: DualMap's hold is 40% of step time there (M4 §3.1). Order of tests: DRAM-tier sizing first (§5 item 1), then this direction only if capacity does not explain the hold.

## 4. After the grid

- **DualMap holds at R 2.4 (this is what happened).** KV-pressure scheduling closed on this workload.
  Open question for the advisor: which problem next. Evidence in hand that
  points elsewhere: GPU-side time per step (7–14 s) is small against tool
  time (median 66 s per step), so cost per task (GPU-hours) rather than
  latency may be the deployment metric; and the tail behaviour of sticky
  placement (one GPU idle for the last 10–25 min of every run).
- **DualMap degrades.** Build the working-set admission controller: first an
  oracle version fed the true next-step contexts from the traces (upper bound,
  analysis only), then the online version, evaluated at the grid points
  against FCFS sticky and DualMap.
- **N=8 verdict (read 19:01 UTC, M4 §3).** Prong 1 met exactly at the
  threshold (1 of 4 windows, the drain window); prong 2 not met (DualMap
  gains 75 s, bound was 120 s). Multi-instance stays in scope by the letter
  of the rule. The open-loop pair is not run: the threshold is met, so it
  cannot change the verdict. The mean-JCT story is small; the makespan story
  (26 vs 38 min) is tail stranding under sticky placement.

## 5. Model-size check with Qwen3-32B-FP8 (pre-registered 18:07 UTC, before any 32B number)

**Question.** Every result so far is on a 4B model. A 32B model changes the
cost ratios that the mechanism claims rest on: KV per token 262,144 vs
147,456 bytes (1.8×), prefill compute about 8× per token, decode slower. Does
"FCFS thrashes, admission recovers" hold when recompute is 8× more expensive
and the natural pressure (no cap, 0.95 memory fraction, about 200K tokens per
instance) is R_avg ≈ 1.4–1.5?

**Configuration.** Qwen/Qwen3-32B-FP8, N=2 full memory, concurrency 32,
pool64-v4, engine context 65,536 via YaRN factor 2 over the model's native
40,960 (generation content is discarded by the replay; only KV and compute
matter). Smoke gates calibration; calibration gates the two runs.

**Expectation.** FCFS sticky: cached share well below the 4B capped run's
0.56, JCT at least 1.5× DualMap's. DualMap: cached ≥ 0.9, per-request hold a
few seconds; JCT gap to FCFS larger than the 4B capped gap (209 s per task)
because each recomputed token costs more.

**Decision.** If the pattern holds, the mechanism claim is model-size robust
and the 32B point becomes the realistic operating point for anything built
next. If DualMap also degrades here, the working-set admission direction
(§3) gets a second, independent motivation.

**Result (03:02 UTC, M4 §3.1).** Pattern held, much larger: FCFS 57.1 min
(cached 0.26, TPOT 156 ms), DualMap 29.7 min (cached 0.86), −1,642 s per
task. But DualMap's hold is now 14 s per request, about 40% of step time, so
KV-pressure scheduling is *not* closed at 32B. The DRAM tier supplied 2% of
tokens: 48 GiB per instance is only 192K tokens at 262 KB per token,
mis-sized for the model. 32B is the operating point from here.

**Next runs at 32B (launched 05:37 UTC as chain 13 after the host sat idle from 03:01; each about 75 min):**
1. DualMap with the CPU tier at 96 GiB per instance (§7.2 test): **done
   06:45 UTC**. Cached 0.95 as expected, hold 14.0 → 9.9 s (expectation was
   below 8 s), −382 s per task. Both explanations hold: capacity (the tier
   is full and still evicting) and admission logic (9.9 s of a 29 s step
   remain). §3 stays gated on the single-engine runs (§8a).
2. TP=2 single instance, FCFS (§7.1): launcher ready, smoke first.
3. A low-pressure 32B reference (concurrency 8, R 0.33) so the 32B pressure
   cost has a floor to be measured against.

## 6. Output-length prediction (meeting notes §2), CPU only, tonight

Data: `analysis/results/output-length-source-labels-crossbench-20260904`
(869 test samples from 174 sessions; sample id encodes model, task, step).
Frozen results there: SSJF-Reg q50 12.3, EGTP-static q50 2.42, train-median
constant q50 1.85.

1. **History-based calibration.** Predict step k's output length from the
   same session's previous recorded outputs only (last value, running median,
   EWMA, and each shrunk toward the train median). Question from the meeting:
   does accumulated history beat a per-request content model? Expectation:
   running median beats the constant on q50 and q90; last-value does not.
2. **Output decomposition.** Split each recorded output into tool-call
   arguments and free text; report each part's share of tokens and of
   variance. Question: does the tool-call format bound the predictable part?
3. **Action-relevant accuracy.** Report bucket accuracy (<128, 128–512,
   >512 tokens) and exceed-threshold hit rates alongside q-error, so the
   consumer of a prediction (admission or KV decision) can be sized.

**Done 18:13 UTC** (`scripts/evaluation/output_length_history_baselines.py`,
results in the handoff document, section "Session-history baselines"):
history predictors gain a few percent at the median (shrunk median q50 1.74
vs constant 1.85) and lose the tail (q90 5.4 vs 4.5); tool-call arguments
carry 91% of output-length variance; 57% of qwen completions are hidden
reasoning. Decision: output length is not a schedulable signal on this data.

## 7. Design-space levers at 32B (agreed 18:25 UTC; runs need the advisor's go after the smoke)

The advisor's direction: extend the design space (parallelism, memory
hierarchy, compute) rather than add schedulers. Two levers survive the
literature check; one was dropped.

**7.1 Parallelism: TP=2 single instance vs DP=2 two instances (Qwen3-32B).**
- Question: 32B fits one GPU, so the deployment choice is open. TP=2 gives one
  unified KV pool (about 400K tokens), 2× prefill speed per request, no sticky
  placement, no cross-instance imbalance, no migration. Does the multi-instance
  problem of Milestones 2–4 dissolve under TP at this scale?
- Expectation: agent steps are prefill-dominated with short outputs, so TP=2
  beats DP=2 sticky on mean JCT at concurrency 32; DP=2 wins aggregate
  throughput only at high batch.
- Decision: TP=2 winning makes "multi-instance" a configuration question,
  not a scheduling one, at this model size; DP=2 winning keeps the N axis.
- Needs: launcher single-instance TP mode (`--tensor-parallel-size 2`, one
  engine on both GPUs, proxy with one backend), smoke, then FCFS and DualMap
  at N=1/TP=2 full memory against chain 12's DP=2 runs. About 1 h code, 10 min
  smoke, 2 runs of about 1 h.

**7.2 Memory hierarchy at 32B: recompute vs DRAM retrieve vs migrate.**
- Question: at 4B a retrieve costs 39 ms and recompute is cheap, so the DRAM
  tier supplied 4% of tokens. At 32B recompute is 8× dearer and retrieve
  1.8×; does the balance tip toward storing rather than recomputing?
- Read from chain 12's DualMap run: the tier supplied 2% of prompt tokens with 66K evictions; at 262 KB per token 48 GiB is 192K tokens, below one instance's GPU cache. The balance did not tip because the tier is too small to hold anything; the 96 GiB run (§5 next runs, item 1) is the actual test.

**7.3 Dropped: n-gram speculative decoding for tool-call arguments.** The
observation that long outputs are copies of context is already exploited:
ToolSpec (arXiv 2604.13519, schema-FSM drafts plus retrieved historical
calls, up to 4.2×), AgentSpec (2608.24004), and the speculative
tool-execution line (2510.04371, 2603.18897, 2512.15834, 2607.25816).

## 8. Decision 06:10 UTC: multi-instance closed, platform moves to one engine per GPU

Multi-instance on this workload is an engineering note (allow migration in
the drain), not a research problem: steady-state imbalance 1.16–1.42 at
N=8, no steady window above 1.5, routing buys 75 s of mean JCT; at 32B
imbalance stays ≤ 1.22. The 32B result puts the problem inside one engine:
40% of each step waits for admission under KV pressure.

**Next platform.** Qwen3-32B-FP8, one engine per GPU, two independent
experiments in parallel (one per GPU), concurrency 16 per engine (R_avg
1.31, same pressure as the N=2 c32 runs). Needs: launcher single-GPU mode
(one instance, proxy with one backend; DualMap's proxy asserts two backends
and needs that relaxed), smoke, then FCFS and DualMap on one engine as the
two instruments of §0.

**Research question.** Of the 14 s per step that DualMap holds a request,
how much is necessary? First an oracle admission that knows every resident
task's next context size (analysis on the traces plus one run with the
oracle in the proxy) to bound the recoverable time; then a method only if
the bound is worth it (the 4B grid says the bound is near zero there; the
32B run says it is not).

**Server change.** The host will be reconfigured (single GPU class or
different count) and its data disk cleared. Saved before that: every run's
`server/` is pulled into `results/<run>/` (verified 06:15 UTC, only the
running 96 GiB run outstanding, pulled at its completion); host launch
logs, LMCache rebuild logs, setup and download logs and the manifests are in
`results/gpuhub-host-backup-20260912/`. Rebuildable and not saved: model
weights (37 GB, about 45 min to re-download), venvs and the LMCache sm_120
source build (about 30 min via `benchmark_server.sh --serving-host`), CUDA
JIT cache.

## 8a. Two lanes from 06:40 UTC (user decision): GPU 0 scheduling, GPU 1 output-length

**Scheduling lane, both GPUs from 07:05 UTC.** Qwen3-32B-FP8, one engine
per GPU (`--single-gpu`, smoke passed 06:51, 236,496 KV tokens), concurrency
16 on pool64-v4. Chain 14 runs FCFS sticky on GPU 0 (started 06:52) and is
stopped after it. Then in parallel: chain 15 on GPU 1, DualMap with a 96 GiB
DRAM tier under `--port-base 100` (new: shifted ports so two runs share the
host, DualMap single-backend smoke first); chain 16 on GPU 0, Continuum
task-sticky (its own smoke first). All three land by about 10:00 UTC and go
to M4 §3.1 as the single-engine baseline table.

**GPU 1, output-length lane: done 07:01 UTC.** Corrected SSJF-Reg (log1p
target) converges to a constant (q50 2.03); EGTP-static on the last 256
tokens spreads its predictions but is worse than the constant on every
metric (q50 1.94 / 2.04, q90 4.9 / 5.8). Neither meets the pre-registered
bar. Output length is closed as a scheduling signal with both methods applied
as intended (handoff document, "Corrected runs"). GPU 1 is free.

Dropped: c16 floor, TP=2 (no decision depends on them), repetition runs
(all decisive gaps are 30× any plausible run noise).

## 8b. Lanes from 11:00 UTC (user decision): GPU 1 OUTLETS, GPU 0 storage pool

The 32B scheduling line (chains 12–16) is stopped: it was my choice, not a
request, and it displaced the OUTLETS test the 30B model was meant for. Its
results stay in M4 §3.1 as a robustness note.

**GPU 1, OUTLETS (the user's purpose for a 30B model).** Natural labels done 14:37 UTC (4,322 labeled, 49 rejected); corrected SSJF/EGTP on natural labels lose to the constant (handoff). Official OUTLETS code unavailable; HF prefill on the FP8 MoE is 761 tok/s (30 h for the features), so the user chose the shallow probe (final-layer hidden state from a vLLM pooling server + MLP head), running since 15:05. **Result 16:25 UTC** (replay-step probe for §10 done 16:50 UTC; GPU 1 idle since)**:** the probe is the first predictor to beat the constant, by a wide margin (natural labels q50 1.35 vs 2.06, MAE 78 vs 119, bucket accuracy 0.81 vs 0.68; four seeds agree; recorded labels q50 1.39 vs 1.85). Tail still weak (long recall 0.16). Handoff document, "Internal-state probe". Target model
Qwen3-30B-A3B-Instruct-2507-FP8 with the EAGLE-3 draft
`lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex` (both
downloading to the host). Steps: regenerate natural completions for the
4,371 prefixes with the target model (32K cap), train the OUTLETS length
head on completion-side hidden states, evaluate on the 869 test samples
beside the constant, SSJF and EGTP. **Blocker:** the official OUTLETS code
(commit 4b53761, formerly at `/workspace/upstreams/outlets` on the old dev
node) is on neither this host nor the local machine, and no public
repository was found; the user is asked where it came from. Without it the
head is a reimplementation from the paper (MLP on fused hidden states of
layers 2, N/2, N−2), which changes the claim to "OUTLETS-style".

**Tail lane (GPU 1, user choice "2", pre-registered 17:08 UTC; `results/host-lanes/tail-20260912.sh`, log `/workspace/outlen/tail-20260912.log`).**
Question: is the probe's weak tail (long recall 0.16 on natural labels) an
objective problem (the MSE head regresses long outputs to the mean), an
information problem (the prompt's last-token state does not carry it), or an
inherent one (the target model itself does not reproduce long outputs between
draws)? Three stages, each answering one of those:

| Stage | Runs | Question | Reading rule (fixed before numbers) |
|---|---|---|---|
| A (minutes) | heads on the existing final-layer features: MSE (base), pinball τ 0.75 / 0.9, long-weighted MSE (×5), 3-bucket class head + MSE, class head + weighted | objective | **Primary, decision utility:** stage-2 estimate on the 32B DualMap 96 GiB replay (recorded-label heads with the 64 tasks excluded): a variant is adopted for the sandbox estimate if p90 absolute error ≤ 22 s (base 25.8 s) with median absolute error ≤ 5.5 s (base 5.1 s). Natural-test long recall / precision, bucket accuracy, q50 are reported as mechanism, not gates. |
| C (~1.5 h) | prefixes extended by the first k = 16 / 64 / 256 tokens of the greedy completion, features from the pooling server, same MSE head; base probe scored on the same subset | information | if long recall on the subset rises above 0.5 at some k, the tail is knowable once generation starts and the sandbox estimate should be updated at that token count; if it stays flat, the prompt-side state is not the limit |
| B (~4.5 h) | sampled labels at temperature 0.7: test × 4 draws, then all splits × 1 draw | inherent | between-draw agreement of the > 512 bucket on the test prefixes; if fewer than half of the prefixes that are long in one draw are long in another, no prompt-only predictor can reach high long recall under sampling, and the recall target is capped there |

**A result (17:07 UTC, `probe-replay-pool64v4/variants/`):** no variant
meets the primary rule; every tail variant makes the replay stage-2 estimate
worse (p90 absolute error 25.8 s base → 31.3–55.7 s; median 5.1 → 5.3–15.4 s).
On natural test labels they buy long recall (0.16 → 0.47 weighted, 0.56
pinball 0.75, 0.74 pinball 0.9) only by losing precision (0.58 → 0.36 / 0.34 /
0.23) and q50 (1.35 → 1.41 / 1.49 / 1.97): the head cannot separate long
from not-long from the prompt-side state, it can only shift everything up.
Diagnosis: not an objective problem; stages C and B decide between
information and inherent.

**Reorder 17:30 UTC** (user: "GPU 1 整个 lane 好像没什么好跑的了?"): the tail
lane was stopped during the k = 16 extraction (400 of 4,235 features cached)
and replaced by `results/host-lanes/lane2-20260912.sh` (log
`/workspace/outlen/lane2-20260912.log`) in decision order: **B first** (it
decides whether the tail is inherent, which also settles OUTLETS-faithful
yes/no), then the **thinking-mode labels** (below), then **C only if B shows
the long bucket is reproducible between draws**: P(long in another draw |
long in one draw) ≥ 0.5 on the 4-draw test labels (`results-tail/b-agreement.json`);
otherwise C is skipped, because a tail the target model itself does not
reproduce cannot be read from the prompt at any k. The reading rules above
are unchanged.

**B result (19:15 UTC, test split × 4 draws at temperature 0.7; 850 prefixes complete, 10 rows rejected at the
32K cap; `analysis/results/.../natural-labels-sampled-test-d4/`):** the tail is **not** inherent. P(long in another
draw | long in one draw) = **0.70** (rule ≥ 0.5 → C runs); 72 prefixes are long in some draw, 23 in all four; the
within-prefix share of log-length variance is 12%. Ceilings on the same 3,400 draw labels: the leave-one-draw-out
mean of the other draws reaches q50 1.10 / q90 1.99, long recall 0.69 at precision 0.79; even the greedy label
alone predicts sampled draws at q50 1.06. The probe trained on greedy labels sits at q50 1.34 / long recall 0.14 /
precision 0.67 (constant 2.03 / 0). Reading: the probe recovers the median but about a fifth of the tail that the
target model reproduces across draws; the gap to the ceiling is information/method, so predictor work (C now,
OUTLETS-faithful or a 32B probe, lane 3) is justified, and the sandbox interface's event design is a choice, not a
necessity forced by unpredictability.

**C result (04:14 UTC; `analysis/results/.../continuation-20260913/`; base probe scored on the same subsets):**
the rule "long recall above 0.5 at some k" is **not met**. After k = 16 tokens (842 test samples): q50 1.34 → 1.26,
long recall 0.16 → 0.09; k = 64 (428): 1.48 → 1.41, recall 0.05; k = 256 (157): 1.57 → 1.44, q90 4.9 → 2.9, recall
0.16 → 0.47 at precision 0.44. The first 64 generated tokens add almost nothing about the tail; only at 256 tokens,
when the output is already half-way to the 512 threshold, does the state start to show it. With B (the tail is
reproducible between draws), the information exists in the prefix but the final-layer last-token state does not
expose it: a representation limit of the shallow probe, the case for OUTLETS-style fused-layer or draft-model
features if a point predictor is ever needed. GPU 1 idle from 04:13 UTC.

Visible when this was written: one smoke of the class head + weighted variant
on the natural test split (long recall 0.465, precision 0.426, q50 1.44) had
been run as a plumbing check before the criteria were fixed; the primary rule
above is on the replay operating point, which had not been computed for any
variant. Results dir on the host `/workspace/outlen/results-tail`.

**Thinking-mode labels (GPU 1 after the tail lane, pre-registered 17:16 UTC; `results/host-lanes/thinking-20260912.sh`, log `/workspace/outlen/thinking-20260912.log`).**
The user's two-marker proposal (§10) measured with a target whose reasoning
close is a real token: Qwen3-30B-A3B-Thinking-2507-FP8 generates one draw per
prefix of dataset-nat (temperature 0.6 / top-p 0.95, Qwen's thinking-mode
setting; 64K cap so reasoning is uncensored; vLLM reasoning parser splits
`reasoning_content` from the visible message). Output
`/workspace/outlen/natural-labels-thinking-qwen3-30b-a3b-64k`. Questions and
reading rules, all CPU work on the labels plus the existing prompt-side
features: (1) reasoning share of the step's output tokens (report; §10 used a
0.36 median from recorded traces). (2) Remainder after the close, predicted
from the tool name (pool median visible tokens per tool): q-err p50 ≤ 1.4
means the alert carries information beyond the whole-output estimate (≈ 1.8);
otherwise the alert only shortens the horizon. (3) Reasoning length as its own
probe target (same features, MSE head): q50 compared with the whole-output
probe (1.35); if reasoning length is the less predictable part, the prompt-side
estimate should quote the visible part and treat reasoning as the interval.
Runs inside lane 2 after B (~22:00 UTC), ~3 h; rejects (censored at 64K, timeouts) are counted
and dropped as before.

**Result (labels 21:00–03:25 UTC, read 03:28; 4,301 labeled, 21 rejected;
`analysis/results/.../thinking-20260913/`).** (1) In thinking mode reasoning is
**91% of the output at the median** (mean 76%): 672 reasoning vs 44 visible
tokens per step (p90 5,328 vs 624), against the 36% estimated from the recorded
traces in §10. The close therefore arrives when the step is essentially over:
44 visible tokens ≈ 1.3 s at 30 ms TPOT. (2) Remainder after the close by tool
name: q50 1.57 / q90 4.19 (rule ≤ 1.4 not met), against 3.09 for the visible
constant and 2.27 for the whole-output constant: the alert carries information,
but the sandbox gains little from it because so little remains. (3) Probe on
the existing Instruct-model features (cross-model, a limitation) with thinking
targets: total q50 1.71 (constant 2.27), reasoning 1.67 (constant 2.79, MAE
1,080 tokens, q90 5.6), visible 1.64. Reasoning length is the less predictable
part in absolute terms, so a prompt-side estimate for a thinking model should
quote the visible part plus a wide reasoning interval. Interface reading: for a
thinking-mode target the useful early signal is `scheduled` with the prefill
bound; `reasoning closed` is a near-finish signal rather than a mid-step update.

**Overnight GPU 0 (pre-registered 17:08 UTC; `results/chain20-gpu0-32b-undersized-20260912.sh`, after chain 19).**
The undersized column of the 2×2 at 32B: FCFS + 24 GiB tier, then DualMap +
24 GiB, one engine c16 (GPU KV 236K tokens; working set 287K mean, 540K at the
p90 step; 24 GiB = 98K tokens covers the mean and not the peaks, the 4B
"12 GiB" ratio). Question: does "a tier smaller than the working set is pure
cost" (M4 §3.2 reading 1, 4B only so far) hold at 32B, and does admission
still rescue it (4B: −186 s)? Rule: FCFS+24 slower than plain FCFS c16 (3.8 h
run, `pool64v4-pro6000-qwen32b-gpu0-c16-fcfs-sticky-20260912-r1`) with a
tier hit share under 10% confirms it; DualMap+24 faster than plain FCFS
confirms the rescue. Expected finish (revised 17:45 UTC after FCFS+96 took 1.85 h instead of 3.5): DualMap+96 ~19:30, FCFS+24 by ~00:30 (5 h budget), DualMap+24 by ~04:00.

**Lane 4, GPU 1 (user ~04:20 UTC "为什么不取多几层呢？", go ~04:25; pre-registered 04:32 UTC, launched 04:32; `results/host-lanes/lane4-20260913.sh`, log `/workspace/outlen/lane4-20260913.log`).**
Multi-layer last-token features: the residual stream after layers 2 / 24 / 45 plus the final normed state of
Qwen3-30B-A3B-Instruct-2507-FP8 (48 layers; the EAGLE-3 / OUTLETS layer convention), 8,192 dims, from the pooling
server with the aux-hidden-state patch `scripts/evaluation/vllm_aux_layers_sitecustomize.py` (smoke 04:30 UTC: final
block matches the stored final-layer feature at cosine 0.988, eager kernels). Same MSE head, same splits, natural
labels (test tail metrics, seeds 42/1/2/3), recorded labels with the 64 replayed tasks excluded, replay predictions.
Question: is the tail information (B: reproducible; C: not in the final layer's last token) in other layers?
Rules: the multi-layer probe **beats the shallow probe** if on the natural test split long recall ≥ 0.35 at
precision ≥ 0.5 with q50 ≤ 1.35 (shallow: 0.16 / 0.58 / 1.35), and it is **adopted for the sandbox estimate** if the
replay stage-2 p90 absolute error ≤ 22 s with median ≤ 5.5 s (shallow: 25.8 / 5.1). Meeting the first rule keeps
the point-predictor line open; missing both closes it as a prompt-side hidden-state limit.

**Result (05:12 UTC; `analysis/results/.../aux-layers-20260913/`): both rules missed.** Natural test, seeds
42/1/2/3: q50 1.36 / 1.38 / 1.41 / 1.39 (shallow 1.35), q90 3.0–3.2 (shallow 2.66), long recall 0.16 / 0.30 / 0.14 /
0.23 at precision 0.41–0.50 (shallow 0.16 / 0.58). Replay stage-2 with the recorded-label multi-layer head: p90
absolute error 25.2 s, median 5.3 s (shallow 25.8 / 5.1; rule 22 / 5.5). The layers 2 / 24 / 45 add nothing the
final layer's last token does not already carry; the tail is not readable from the prompt-side hidden state at any
depth with an MLP head. Per the rule the point-predictor line closes as a prompt-side hidden-state limit: the
remaining lever would be completion-side supervision with a draft model (OUTLETS proper), which is not a prompt-side
estimate and is not queued. GPU 1 idle from 05:10 UTC.

**Agent-trace diagnostic before any OUTLETS build (user ~05:14 UTC "需要基于 agent trace 的性质调整方法吗"; CPU, natural labels, read 05:20 UTC).**
The long tail is a tool question first: edit_file and write_file are 11% of steps (470 of 4,322) but 70% of the
outputs above 512 tokens (39% + 31%; exec 15%, final answers 13%; read/list never). Within those tools the length is
not visible in the prompt: write_file targets were read earlier in only 10 of 146 steps; for edit_file the target
was read in 256 of 324 but its size explains nothing (corr of log length with log file size 0.11; with the largest
read output −0.05; with the number of earlier edits −0.21). Even a perfect tool oracle leaves q50 1.45 (edit) and
1.84 (write) on the per-tool median, so tool classification alone cannot reach the tail rule. What decides the size
of an edit is the model's plan, which B shows is consistent across samples but which no prompt-visible structure or
prompt-side hidden state (final or fused layers, lane 4) exposes. Adaptations that the data supports: a structured
target (tool × conditional length) and event-timed dynamic updates; the "context-visible size" feature is not
supported. OUTLETS proper remains the only untried reader of the prefix (attention over all positions with
completion-side supervision); the paper's own static gain over its MLP baseline is ~5% MAE.

**Fills for idle devices (user: "如果有设备空闲，但是我还没有回来，安排最有价值的实验进行填充"; pre-registered 18:45 UTC, both lanes launched 18:46 waiting on their predecessors).**

**OUTLETS-agent (user go ~05:25 UTC "可以，投吧"; pre-registered 05:30 UTC; ~1 day of work, GPU 1).**
The paper's method (arXiv 2609.01068: EAGLE-3-style draft decoder over the target's layer 2 / N/2 / N−2 states,
log-space remaining-length head supervised at every completion position, static estimate at t = 0) adapted to
agent steps as the diagnostic above supports: (a) backbone = the pretrained SpecForge EAGLE-3 draft for
Qwen3-30B-A3B-Instruct-2507 (`lmsys/SGLang-EAGLE3-…-SpecForge-Nex`; fc 6144→2048, one gated decoder layer), fine-tuned
with the length losses instead of joint training with the speculative-decoding loss (deviation 1); (b) the draft
attends over a window of the last 1,024 prompt positions plus up to 512 teacher-forced completion positions, with
the target's states computed on the **full** prompt (the paper truncates to 2,048 total; deviation 2); (c) structured
static target: tool-class logits and a per-tool log-length, prediction = expected log-length under the tool
distribution, next to the paper's scalar head; (d) dynamic remaining-length head at every completion position,
evaluated at the paper's MAE and at the event where the tool name has appeared. Training data: the 4,322 greedy
natural completions (target's own), splits as before; a second head on recorded labels with the 64 replayed tasks
excluded for the replay check. Seeds 42/1/2/3 for the static head. Features: per-token `fc(cat(h2, h24, h45))`
from the pooling server (`token_embed` task, aux patch v2), stored float16.
Rules (unchanged from lane 4): natural test long recall ≥ 0.35 at precision ≥ 0.5 with q50 ≤ 1.35 → beats the
shallow probe; replay stage-2 p90 absolute error ≤ 22 s with median ≤ 5.5 s → adopted for the sandbox estimate.
Dynamic: MAE at t of the paper's definition reported; the tool-name event estimate compared with the §10 stage-3
constant (q-err 1.70). Two ablations, frozen backbone (heads only) and scalar-only head, separate the attention
over the prompt from the structured target. Missing both rules closes the point-predictor line for good.

*Lane 3, GPU 1 after lane 2* — **cancelled by the user 02:35 UTC before it started** (GPU 1 stays idle after lane 2 until a stated need) (`results/host-lanes/lane3-20260912.sh`, log
`/workspace/outlen/lane3-20260912.log`): the output-length work moved onto the
platform model. Qwen3-32B-FP8 (YaRN ×4 for the 7% of prompts above 40K)
generates thinking-mode natural labels for dataset-nat (64K cap, temperature
0.6 / top-p 0.95, one draw; rejects dropped into `dataset-nat-32b`), a 32B
pooling server extracts final-layer features for the 4,322 prefixes and the
1,951 replay steps, and the probe is trained on (a) the 32B natural labels
and (b) recorded labels with the 64 replayed tasks excluded. Questions and
rules: (1) does the probe beat the constant on the platform model's own
labels: q50 at least 0.2 below the constant's (30B: 1.35 vs 2.06) says the
result transfers; (2) reasoning share of the 32B thinking output and the
after-close remainder by tool name (same rules as the thinking-mode lane);
(3) the sandbox stage-2 estimate on the 32B DualMap 96 GiB replay with the
32B probe against the 30B probe row (q-err 1.60 / 4.02, abs p90 25.8 s): the
platform probe is adopted if it is not worse. Expected 5–7 h (32B dense
decode of reasoning traces is the cost).

*Chain 21, GPU 0 after chain 20* (`results/chain21-gpu0-32b-c24-20260912.sh`,
log `results/chain21-gpu0-c24-20260912.log`): once the store is sized, does
admission's value grow with pressure? Concurrency 24 (mean working set 430K
tokens, above the 96 GiB tier alone and below tier + GPU; p90-step peaks 810K
exceed both): FCFS + 96 GiB, then DualMap + 96 GiB, 5 h budgets. Rule: the
paired DualMap − FCFS difference at c24 compared with the c16 difference
(chain 19). If it grows by more than the bootstrap interval, admission is
the pressure-side mechanism and the design is "store sized to the mean,
admit at the peaks"; if it stays within the interval or shrinks, the sized
store carries the pressure and admission is a fixed second-order term.
Throughput (steps/min) is reported beside JCT because c24 carries more load.
**05:38 UTC 2026-09-13: the host container restarted** (SSH closed, then refused for ~4 min; `/workspace` intact,
all processes gone). Chain 21's c24 FCFS+96 run died at 2.4 h (replay exit 143); chain 21 stopped. Rerun as
`results/chain22-gpu0-32b-c24-20260913.sh` (log `results/chain22-gpu0-c24-20260913.log`) from 05:43 UTC, same
runs named `…-20260913-r2`. The OUTLETS-agent lane (`results/host-lanes/lane5-20260913.sh`, log
`/workspace/outlen/lane5-20260913.log`) started on GPU 1 at 05:43 UTC.

**GPU 0, storage pool vs admission** (`results/chain17-gpu0-storage-20260912.sh`).
Resource picture from the 32B runs: memory is per context and lives through
decode and tool gaps; compute is only cold prefill; what agents need
separated is a compute pool and a history store, not prefill and decode.
First question: is DualMap's remaining wait (9.9 s of a 29 s step at 32B)
an admission necessity, or does a right-sized store make admission
unnecessary? Decomposition on Qwen3-4B capped to 262,144 tokens, one engine
on GPU 0, concurrency 16 (R_avg ≈ 1.2), tier sized below the GPU cache
(12 GiB ≈ 85K tokens, the 32B ratio) and far above it (48 GiB):

| Run | Storage | Admission | Question |
|---|---|---|---|
| FCFS sticky | none | none | pressure symptom |
| FCFS sticky + tier 12 GiB (new config, smoke first) | yes | none | storage alone |
| DualMap + tier 12 GiB | yes | yes | storage + admission |
| FCFS sticky + tier 48 GiB | oversized | none | does capacity alone close the gap |

**Result (15:00 UTC, M4 §3.2):** undersized tier = pure cost (+222 s, 0 tokens
served); admission with an undersized tier −186 s; right-sized tier without
admission **−353 s**, beating admission by 167 s per task. Chain 18 (15:47 UTC): DualMap +
48 GiB **16.91 min**, −192 s on top of the right-sized tier. Both mechanisms
matter; the store is the larger term, admission the second (M4 §3.2).
Confirmation at 32B on GPU 0 (`results/chain19-gpu0-32b-confirm-20260912.sh`): FCFS + 96 GiB landed 17:43 UTC,
**111.69 → 53.81 min** (−3,473 s per task; cached 0.18 → 0.92; M4 §3.2 32B table). DualMap + 96 GiB r1 aborted 19:16 UTC at 1,931 of 1,951 original requests: a replacement-stream (background load) task failed and `simulator.py` raises on a non-cycled replacement failure (`replacement load task failed`), which ends the measurement; not a scheduler failure, all 64 tasks had traces. Rerun (r2) prepended to chain 21 at 19:25; the simulator's abort-on-replacement-failure is a harness bug to fix when no replay is running. Chain 20: FCFS + 24 GiB landed 23:20 UTC, **113.67 min vs 111.69** plain (+119 s [+92, +145]; 0% of prompt tokens served, 170K evictions): undersized tier = pure cost confirmed at 32B. DualMap + 24 GiB landed 01:24 UTC, **57.83 min** (−3,232 s vs plain FCFS; hold 17.6 s, cached 0.85, tier 0%): admission alone recovers nearly the whole gain, 4 min/task behind the sized store alone (53.81). DualMap + 96 GiB r2 landed 03:10 UTC, **45.80 min** (−481 s [−615, −346] on top of the sized store; −3,954 s vs plain FCFS; hold 15.1 s). 32B 2×2 complete, same structure as 4B (M4 §3.2). Chain 21 c24 pair running from 03:11.

Decision: if FCFS + tier ≈ DualMap + tier, admission is not the mechanism
and the work is store sizing/placement/sharing; if DualMap stays ahead,
admission is necessary and the store is complementary; the 48 GiB row says
whether "more DRAM" alone is the answer. About 30 min per run; winner
confirmed once at 32B when GPU 1 frees.

## 9. Realism check: Qwen3.8-27B-FP8 (recorded 06:45 UTC, one measurement, not a platform change)

The current dense 30B-class model stays Qwen/Qwen3-32B-FP8 (Qwen3 generation,
64 layers × 8 KV heads × 128, 262 KB KV per token). Qwen/Qwen3.8-27B-FP8 is
newer but is a different kind of model: `qwen3_5` hybrid attention, 16 of 64
layers full attention (4 KV heads × 256), 48 linear-attention layers with a
constant recurrent state, native context 262,144, vLLM ≥ 0.17 required
(recipes.vllm.ai). Two consequences: (1) none of our policies run on it
without porting the Continuum vLLM 0.10.2 fork, DualMap patches and the
LMCache connector to a new vLLM, days of work plus re-running every
baseline; (2) KV per token is about 65 KB, a quarter of Qwen3-32B, so at the
same concurrency R_avg is about 0.3 on a 96 GB GPU and the KV pressure
measured on Qwen3-32B largely disappears, replaced by a per-sequence
recurrent state that must be saved and restored across tool gaps but cannot
be prefix-shared.

Planned measurement (needs the advisor's go, about 1.5 h): plain vLLM
0.28 (the PD venv on the host) serving Qwen3.8-27B-FP8, FCFS, one engine,
concurrency 16 on pool64-v4 traces, reading only KV usage, cached share and
per-request queue/prefill/decode. Question: is KV pressure still a problem
on a hybrid-attention model of this class? If not, the premise of the
KV-pressure line narrows to dense-attention deployments and the advisor
should know before more is built on it.

## 10. Sandbox-side interface: what the LLM side can promise (measured 08:00 UTC)

Context: a collaborator snapshots and restores each task's sandbox around the
LLM step (restore p50 1.0 s, p90 1.8 s; corrected by the user 17:30 UTC from
the earlier 2.2 / 3 s) and needs to know how long the step will take. `scripts/evaluation/step_time_staged_estimates.py` evaluates, on
finished runs, the estimate available at each moment: stage 1 (arrival: hold
and queue unknown), stage 2 (engine scheduled the request: remaining =
uncached tokens × prefill constant + prior output length × causal TPOT),
stage 3 (tool name known: tool-conditioned length prior). Priors from the
pool excluding the run's tasks.

| Run | Stage-1 wait p50 / p90 (s) | Remaining at scheduling p10 / p50 (s) | Share ≥ 1.8 s (≥ 3 s) | Stage-2 q-err p50 / p90 | Stage-2 + probe q-err p50 / p90 | Stage-3 q-err p50 / p90 |
|---|---|---|---:|---|---|---|
| 32B DualMap 96 GiB | 3.2 / 12.9 | 3.7 / 9.6 | 1.00 (0.94) | 1.83 / 4.44 | 1.60 / 4.02 | 1.70 / 3.86 |
| 32B FCFS sticky | 36.0 / 56.7 | 4.5 / 22.8 | 1.00 (0.95) | 1.87 / 4.90 | 1.66 / 4.28 | 1.73 / 4.13 |
| 4B FCFS full | 3.1 / 6.3 | 1.2 / 3.2 | 0.76 (0.53) | 1.82 / 4.47 | 1.61 / 4.07 | 1.70 / 3.82 |
| 4B DualMap capped | 1.7 / 6.2 | 1.4 / 3.6 | 0.82 (0.58) | 1.79 / 4.52 | 1.57 / 4.10 | 1.70 / 3.81 |

Reading. (1) The prefill-only lower bound holds 100% of the time; a bound
that adds the 10th-percentile output length fails 14%, so the safe promise
is "at least prefill". (2) Stage 2 is the useful event: at 32B every step has ≥ 1.8 s (the p90
restore) of engine time after scheduling, so a restore started at that event
is never late; at 4B 76–82% do (92–96% at the p50 restore of 1.0 s), and the
rest must start at arrival, using the hold as slack, or accept lateness.
(3) The tool name buys little (q-err 1.83 → 1.70): output length stays the
unpredictable part, as the output-length study concluded. (3b, 16:50 UTC,
user's choice "1") The hidden-state probe (§8b; Qwen3-30B-A3B final-layer
state of each replay step's prompt, head trained on recorded labels with the
64 replayed tasks removed from train/validation, 2,831 / 401 samples) replaces
the pool-prior output length at stage 2: q-err p50 1.83 → 1.60, p90 4.44 →
4.02, absolute error p90 30 → 26 s at 32B DualMap; median absolute error is
unchanged (4.8 → 5.1 s) and 62% of steps are over-predicted (mean prediction
214 tokens), which is the safe side for a restore deadline. A first pass
without the task exclusion scored 1.49 / 3.69: 46 of the 64 tasks were in the
head's training split, so that number is discarded (§3.2). Files:
`analysis/results/output-length-source-labels-crossbench-20260904/probe-replay-pool64v4/`. (4) Stage-1 wait
is large under FCFS (p50 36 s at 32B) and heavy-tailed under DualMap (mean
11.7 s, p50 3.2 s); it is our own decision, so it should be sent as an event
("scheduled"), not predicted.

**Two-marker proposal (user, 17:08 UTC): safe lower bound at arrival = queue estimate + prefill; alert when the reasoning closes.** Measured 17:13 UTC on the same four runs (`at_reasoning_end` in `probe-replay-pool64v4/staged-*.json`; reasoning tokens = recorded completion − tiktoken count of the visible text and tool arguments, alert time proportional inside the replay's decode; estimate after the alert = pool median visible tokens of the named tool × causal TPOT):

| Run | Reasoning share of output tokens p50 / mean | Remaining after alert p10 / p50 (s) | Share ≥ 1.0 / 1.8 s after alert | After-alert abs error p50 / p90 (s) | q-err p50 / p90 |
|---|---|---|---|---|---|
| 32B DualMap 96 GiB | 0.36 / 0.41 | 1.6 / 5.3 | 0.97 / 0.85 | 2.5 / 14.8 (stage 2 probe: 5.1 / 25.8) | 1.84 / 4.49 |
| 32B FCFS sticky | same traces | 1.7 / 9.5 | 0.98 / 0.89 | 5.4 / 31.5 (10.5 / 54.1) | 1.99 / 5.12 |
| 4B FCFS full | same traces | 0.5 / 1.8 | 0.66 / 0.49 | 0.8 / 4.9 (1.7 / 8.7) | 1.83 / 4.39 |
| 4B DualMap capped | same traces | 0.6 / 2.0 | 0.68 / 0.53 | 0.9 / 5.4 (1.9 / 9.3) | 1.84 / 4.37 |

Reading. (1) Reasoning is a third of the output tokens (median 0.36; qwen3.7-max steps 0.57, gpt-5.6-sol 0.29 by the visible share), so the alert lands after roughly 40% of the decode. (2) The absolute error halves after the alert (32B DualMap p50 5.1 → 2.5 s, p90 25.8 → 14.8 s) because less time remains, not because the remainder is more predictable: the q-error is unchanged (1.6–1.8 → 1.84), the remaining length is the tool arguments, and their spread (write_file vs read_file) is what the prompt-side probe already could not resolve. (3) At 32B the alert is early enough for a restore in 85% of steps (≥ 1.8 s, the p90 restore, after it), 97% at the p50 restore of 1.0 s; at 4B about half (49–53% at 1.8 s, 66–68% at 1.0 s). (4) The lower bound at arrival: queue/hold is our decision and should be sent as the "scheduled" event; prefill is the safe bound at that moment (100% coverage, above). So the interface is three signals, not two: scheduled (with prefill bound), reasoning closed (with tool name and the tool-conditioned remainder), and the finish. A thinking-mode target model (Qwen3-30B-A3B-Thinking-2507) would make the alert a real token event and give the reasoning length as a separate prediction target; queued on GPU 1 after the tail lane (§8b).

Reverse direction, tool-time prediction for us: with the 96 GiB tier the
prompt miss share is 3% and flat across tool-gap lengths (0–5 s: 3%, 5–15 s:
4%, longer: 0%), so return-time-aware eviction has at most 3% of steps to
gain at this operating point; under FCFS misses are 65% even at gaps under
5 s (capacity thrash, not age). No consumer for tool-time prediction on our
side right now.

## 10. Decisions waiting on the advisor

- Push branch `codex/cleanup-research-dead-code` (≈45 commits ahead, unpushed).
- Which problem to pursue if §2 closes KV-pressure scheduling.
- Whether a step 2 verdict carried by a single drain window keeps multi-instance as a research problem, or reduces it to tail placement (M4 §3 step 2 result).
