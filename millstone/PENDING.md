# Pending: experiment queue and open decisions

Current as of 2026-09-11 18:08 UTC. Rewritten, not appended: this file says
what is queued, why, and what each result decides. Records of finished work
live in the milestone files; this file only points at them.

## 1. Running and queued on the GPU host (2× RTX Pro 6000)

All runs on `analysis/development/pool64-distinct-v4` (64 distinct traces).
Per-run analysis lands in `results/<run>/comparison.txt` and
`instance-balance.txt`.

| Order | Run | Purpose | Status |
|---|---|---|---|
| 1–4 | N=2 full and N=2 capped, FCFS sticky and DualMap | Step 2 reference and same-capacity control for N=8 | done (M4 §3) |
| 5–6 | N=8 capped (k=4 per GPU under MPS), FCFS sticky and DualMap | Step 2 decision point | running, ~19:15 UTC |
| — | N=4 capped pair | optional middle point of the N axis | dropped 17:52 UTC, no result read; rerun only if N=8 is anomalous |
| 7–10 | N=2 capped at concurrency 48 and 64, FCFS sticky and DualMap | Step 3 pressure grid (below) | queued, `results/chain11-grid-20260911.sh`, ~19:20–22:30 UTC |
| 11–14 | Qwen3-32B-FP8: smoke, DualMap calibration, N=2 full FCFS sticky and DualMap at concurrency 32 | Model-size check (§5) | queued, `results/chain12-qwen32b-20260911.sh`, ~22:30–01:30 UTC |

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

**Measured.** Per-request decomposition (proxy hold, in-engine queue,
prefill, decode) from `routing.jsonl` and `vllm-request-telemetry.jsonl`,
cached share, JCT with the paired bootstrap, LMCache retrieve counters.

## 3. Candidate direction: task-level working-set admission

Not a KV retention or prefetch mechanism (those are crowded: ThunderAgent,
CacheWise, SAGA, Continuum, Agentix, and prefetch has nothing to recover here:
a CPU-tier retrieve costs 39 ms). Not dependent on tool-time or output-length
prediction (both measured as unreliable in this project).

Signal: an agent's context grows monotonically; prompt at step k+1 ≥ prompt
at step k plus output plus tool result, and the growth rate is measurable
from the task's own history. Decision: admit a new task to an instance only
if the projected contexts of its resident tasks over the next few steps fit
the KV cache. This is the multiprogramming-level control of the working-set
model with agent sessions as processes and tool gaps as I/O waits. Expected
win: zero hold at low pressure (DualMap pays 19 s per task there), no thrash
at high pressure.

Before any code: literature check for task-level admission with context
growth forecasting (Llumnix, Preble, SAGA, Agentix, CacheWise, ThunderAgent,
Continuum, Autellix, Parrot, Mooncake). Result to be reported with sources,
not from memory.

Gate: §2 result. Only if DualMap degrades at R 2.4.

## 4. After the grid

- **DualMap holds at R 2.4.** KV-pressure scheduling closed on this workload.
  Open question for the advisor: which problem next. Evidence in hand that
  points elsewhere: GPU-side time per step (7–14 s) is small against tool
  time (median 66 s per step), so cost per task (GPU-hours) rather than
  latency may be the deployment metric; and the tail behaviour of sticky
  placement (one GPU idle for the last 10–25 min of every run).
- **DualMap degrades.** Build the working-set admission controller: first an
  oracle version fed the true next-step contexts from the traces (upper bound,
  analysis only), then the online version, evaluated at the grid points
  against FCFS sticky and DualMap.
- **N=8 verdict (tonight).** Apply M4 §3 rule. If imbalance is below the
  threshold, the closed loop may be masking arrival-driven imbalance; the
  open-loop N=8 pair (Poisson arrivals, mean gap 24 s) is the check before
  multi-instance is closed. If the threshold is met, the open-loop pair adds
  nothing to the rule and is not run.

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

Output: one table appended to the handoff document's results section, with
the script under `scripts/analysis/`. Decision: whether output length is a
signal worth carrying into any scheduler at all, or whether context growth
(§3) is the only usable per-task forecast.

## 7. Decisions waiting on the advisor

- Push branch `codex/cleanup-research-dead-code` (≈45 commits ahead, unpushed).
- Which problem to pursue if §2 closes KV-pressure scheduling.
