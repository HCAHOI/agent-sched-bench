# Pending: experiment queue and open decisions

Current as of 2026-09-12 15:05 UTC. Rewritten, not appended: this file says
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

**GPU 1, OUTLETS (the user's purpose for a 30B model).** Natural labels done 14:37 UTC (4,322 labeled, 49 rejected); corrected SSJF/EGTP on natural labels lose to the constant (handoff). Official OUTLETS code unavailable; HF prefill on the FP8 MoE is 761 tok/s (30 h for the features), so the user chose the shallow probe (final-layer hidden state from a vLLM pooling server + MLP head), running since 15:05. Target model
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
admission **−353 s**, beating admission by 167 s per task. Chain 18 runs the
missing cell (DualMap + 48 GiB) now.

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
LLM step (restore ≤ 3 s, p50 2.2 s) and needs to know how long the step will
take. `scripts/evaluation/step_time_staged_estimates.py` evaluates, on
finished runs, the estimate available at each moment: stage 1 (arrival: hold
and queue unknown), stage 2 (engine scheduled the request: remaining =
uncached tokens × prefill constant + prior output length × causal TPOT),
stage 3 (tool name known: tool-conditioned length prior). Priors from the
pool excluding the run's tasks.

| Run | Stage-1 wait p50 / p90 (s) | Remaining at scheduling p10 / p50 (s) | Share ≥ 3 s | Stage-2 q-err p50 / p90 | Stage-3 q-err p50 / p90 |
|---|---|---|---:|---|---|
| 32B DualMap 96 GiB | 3.2 / 12.9 | 3.7 / 9.6 | 0.94 | 1.83 / 4.44 | 1.70 / 3.86 |
| 32B FCFS sticky | 36.0 / 56.7 | 4.5 / 22.8 | 0.95 | 1.87 / 4.90 | 1.73 / 4.13 |
| 4B FCFS full | 3.1 / 6.3 | 1.2 / 3.2 | 0.53 | 1.82 / 4.47 | 1.70 / 3.82 |
| 4B DualMap capped | 1.7 / 6.2 | 1.4 / 3.6 | 0.58 | 1.79 / 4.52 | 1.70 / 3.81 |

Reading. (1) The prefill-only lower bound holds 100% of the time; a bound
that adds the 10th-percentile output length fails 14%, so the safe promise
is "at least prefill". (2) Stage 2 is the useful event: at 32B, 94–95% of
steps still have ≥ 3 s of engine time after scheduling, so a restore started
at that event is almost never late; at 4B only 53–58% do, and the restore
must start earlier (at arrival, using the hold as slack) or accept lateness.
(3) The tool name buys little (q-err 1.83 → 1.70): output length stays the
unpredictable part, as the output-length study concluded. (4) Stage-1 wait
is large under FCFS (p50 36 s at 32B) and heavy-tailed under DualMap (mean
11.7 s, p50 3.2 s); it is our own decision, so it should be sent as an event
("scheduled"), not predicted.

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
