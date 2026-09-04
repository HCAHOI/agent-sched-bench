# P4 EVICTION DESIGN — pause/resume selective KV offload on vLLM 0.11.2
# Design study. All file:line refs are vLLM tag v0.11.2.
# Companion to spike/README.md and analysis/ROADMAP.md.

## Summary

vLLM v1's preemption and KV-connector rescheduling supply most of the
eviction/resume path. The required integration is a named external
force-preempt command plus a HOLD state so the victim is not immediately
rescheduled. Both are co-located with the scheduler-side connector and
reuse P/D-disaggregation machinery. The original implementation estimate
was ~70-100 net-new lines in one upstream scheduler file plus the connector,
or ~10.5 working days.

**Design verdict:** the connector plus minimal scheduler integration is
feasible. This estimate is not a live-system result; the current launch and
correctness gates are in `analysis/ROADMAP.md`.

## Root cause of the W1 gap (why the spike only copied, never evicted)

SPIKE_NOTE item #3 said eviction+pause is "net-new work, no public 0.11
API." That is true at the *connector* layer — a connector cannot free
blocks or change request status (base.py:432 forbids a connector even
mutating scheduler_output). But the SCHEDULER already does exactly
eviction+resume on its preemption path; the missing piece is not a new
subsystem, it is an *external, per-request trigger* into that path plus
a hold. The gap is a trigger/ownership gap, not a mechanism gap.

## Q1 — How v1 preemption works; can a specific request be force-preempted

Preemption lives inline in Scheduler.schedule(), in the running-request
loop, and fires only when block allocation fails:

- scheduler.py:277-286 — allocate_slots() returns None under memory
  pressure; that is the ONLY preemption trigger. Internal, not callable.
- Victim selection: FCFS pops the TAIL,
  `preempted_req = self.running.pop()` (scheduler.py:319). PRIORITY
  policy instead evicts the max-(priority, arrival_time) request
  (scheduler.py:291-294). Neither lets a caller name the victim.
- The eviction primitive itself (scheduler.py:321-332):
    self.kv_cache_manager.free(preempted_req)      # frees blocks (:321)
    self.encoder_cache_manager.free(preempted_req)             # (:322)
    preempted_req.status = RequestStatus.PREEMPTED             # (:323)
    preempted_req.num_computed_tokens = 0                      # (:324)
    preempted_req.num_preemptions += 1                         # (:325)
    self.waiting.prepend_request(preempted_req)               # (:331)
  kv_cache_manager.free() only calls coordinator.free(request_id)
  (kv_cache_manager.py:343) — it releases blocks to the pool and KEEPS
  the Request object in self.requests. So this frees GPU memory without
  destroying request state. This is the whole eviction we need.

- Mode: v1 has ONE preemption mode, RECOMPUTE. There is no v0-style SWAP
  copy; num_computed_tokens=0 (:324) is the RECOMPUTE signature and on
  resume the tokens are re-derived (recompute, prefix-cache hit, or —
  our lever — a connector load). No other mode exists to fight.

**Smallest hook:** a new scheduler method `force_preempt(req_id)` that
mirrors lines 321-324 for a named request (skip the waiting.prepend —
see Q3). ~15 LoC. A priority-poke alternative (bump the target to
lowest priority and induce pressure) is rejected: it only fires if
pressure already exists and preempts the max-priority-value request, not
necessarily ours — unreliable and indirect.

## Q2 — Does resume consult get_num_new_matched_tokens for a preempted req

YES, unconditionally, and this is the load-bearing finding. Because
preemption resets num_computed_tokens to 0 (:324), a PREEMPTED request
re-entering the waiting loop takes the same branch as a brand-new
request:

- scheduler.py:447 `if request.num_computed_tokens == 0:` — true for a
  preempted req.
- :449-451 local prefix-cache probe get_computed_blocks().
- :454-459 `self.connector.get_num_new_matched_tokens(request,
  num_new_local_computed_tokens)` — the connector IS consulted for the
  preempted request, receiving the count of tokens already matched
  locally and returning how many MORE it can supply externally.
- :472-474 num_computed_tokens = local + external.
- :551-559 allocate_slots(num_new_tokens + num_external, ...).
- :569-574 update_state_after_alloc() — connector records what to load.
- Sync path: request goes RUNNING; worker start_load_kv() fires for the
  allocated blocks (the proven W1 copy, now H2D on resume).
- Async path (load_kv_async=True): num_new_tokens=0 (:489), status set
  to WAITING_FOR_REMOTE_KVS (:586); parked until the worker's
  get_finished() reports the id in finished_recving (base.py:301-317 →
  scheduler.py:1472-1474) → _update_waiting_for_remote_kv() returns True
  (:1441-1455) → request resumes. This is the PD-disaggregation flow,
  reused verbatim.

**The one assumption to flag (reviewer-facing):** base.py:396-400
documents get_num_new_matched_tokens as matching "the largest prefix of
prompt-tokens." A paused agent request also has GENERATED tokens whose
KV we saved. Nothing in the scheduler enforces prompt-only —
scheduler.py:495 uses request.num_tokens (prompt+output) and the comment
at :492-494 explicitly says "to consider the resumed requests, which
have output tokens." Our connector keys on request_id (not a content
hash), so it reports the exact saved token count regardless of the
prompt/output split. This is within the mechanism but outside the
docstring's idiomatic use; document it and pin <0.12.

**Self-hit subtlety:** block_hashes are NOT cleared on preempt
(request.py:124-128, 166-167 only ever append). So get_computed_blocks
(:449) can find the request's own just-freed blocks still in the
prefix-cache pool and count them as local. Under LOW pressure a paused
request reloads for free via prefix cache and the connector reports 0
extra — harmless. Under REAL pressure (our target regime) those blocks
are overwritten, the local match shrinks, and the connector fills the
delta. Correct either way, but the connector MUST report only tokens
BEYOND num_new_local_computed_tokens (the arg at :457) or it double-
allocates.

## Q3 — "Pause": hold until tool-completion signal

Not expressible today: preemption immediately prepends the victim to the
waiting queue (:331), making it a first-class reschedule candidate the
very next step. There is no held/not-schedulable state — the agent
workload needs "stay evicted until the tool call returns," not "resume
ASAP."

But the scheduler already has skip machinery in the waiting loop:
WAITING_FOR_REMOTE_KVS (scheduler.py:404-415), WAITING_FOR_FSM
(:419-426), and max_loras (:430-441) all pop + prepend-to-skipped +
continue. A hold is one more skip predicate.

**Recommended encoding (hold set):** force_preempt(req_id) mirrors
:321-324 but, instead of waiting.prepend (:331), adds req_id to a new
`self.paused: set[str]` and leaves the Request in self.requests
(free() keeps it alive). The paused request is in NO queue, so the
scheduler never touches it — memory stays freed. resume_request(req_id)
pops it from `paused` and calls waiting.prepend_request(req), after
which the normal :447 → connector-load path restores it. ~20 LoC, zero
change to the waiting loop body. Preferred over reusing
WAITING_FOR_REMOTE_KVS, which would conflate tool-pause with async-KV
semantics and entangle the invalid-block recovery logic
(:1516-1570).

## The real subtlety — save-before-free ordering

Pause must SAVE R's KV host-side BEFORE freeing its blocks, or a co-
tenant's next forward pass overwrites them. vLLM already has the exact
"save async, free when done" protocol: request_finished() returns True
("I'm still saving, don't free") → worker saves → get_finished() reports
the id in finished_sending → scheduler frees at scheduler.py:1475-1478.
But request_finished only fires on a genuine finish (:1294), and we must
NOT finish a paused request. So pause is a 2-phase op:

  step N   : connector emits a SAVE directive for R (proven W1 path);
             scheduler marks R "pausing", R stays RUNNING/resident.
  step N+k : worker confirms save via a saved-req-ids set returned in
             KVConnectorOutput and consumed exactly like finished_recving
             (:1472); scheduler THEN frees + PREEMPTED + n_c_t=0 + adds
             to `paused`.

Simplest correct v1: a SYNCHRONOUS save (block one step, ~72 ms measured)
avoids the extra confirm channel entirely and is negligible against a
multi-second tool call. Upgrade to the async saved-set later if the
one-step stall shows in co-tenant ITL.

## Q4 — What breaks (and the disposition of each)

- num_computed_tokens bookkeeping: handled by the framework (reset :324,
  restored :610). Requirement: our saved token count == n_c_t at pause
  instant, exactly, or resume mis-slices. Add a fail-fast assert.
- Output-token / sequence continuity: SAFE. _output_token_ids /
  _all_token_ids live on the Request and are untouched by preempt
  (request.py:90-95, 109-110); num_tokens (:174) still returns
  prompt+output. Loading the SAVED KV is bit-identical to the pre-pause
  state — strictly MORE faithful than RECOMPUTE (no numerical drift).
- Sampler / RNG state: greedy (spike temperature=0) is stateless — N/A.
  Seeded RANDOM sampling loses the per-request generator offset when the
  request leaves and re-enters the worker batch; a known gap. Eval uses
  greedy; disable/annotate random sampling.
- logprobs: sample-logprobs are emitted at generation and need nothing
  retroactively — SAFE. prompt_logprobs sets skip_reading_prefix_cache
  (request.py:185-196 → kv_cache_manager.py:192 returns 0 local), but
  the connector at :454 is still consulted, so external resume still
  works; treat prompt_logprobs on a resumed request as an unsupported
  edge case for the agent workload.
- prefix-cache interaction: the self-hit above; correct if the connector
  reports delta-only.
- chunked prefill: COMPATIBLE. num_external is added whole to
  allocate_slots (:553) regardless of the chunk threshold, which only
  chunks NEW tokens (:497); async resume sets num_new_tokens=0. No
  conflict.
- spec decode: disable (roadmap permits). spec_token_ids churn on a
  paused request adds needless edge cases; cleaner off.
- Trigger liveness (request finishes between trigger and pause): the W1
  connector already prunes finished_req_ids each build_connector_meta
  (spike README risk 2); the scheduler force_preempt must likewise no-op
  on a finished/absent id.

## Q5 — Alternative seams if the hypothesis fails (it does not)

(a) sleep/wake_up (LLM.sleep()/wake_up): whole-ENGINE granularity, not
    per-request — pausing one agent would pause all co-tenants. Wrong
    semantics; usable only for a single-tenant descope demo. Reject as
    primary.
(b) KVCacheManager direct manipulation: calling kv_cache_manager.free()
    plus manual block bookkeeping outside the scheduler desyncs
    self.running / self.waiting / status and the worker's persistent
    input batch. The scheduler IS the correct layer; doing this below it
    is strictly higher risk. Reject.
(c) Fork-level scheduler patch: THIS is the recommendation (below). The
    hypothesis holds, so (a)/(b) are contingencies only.

## Recommended mechanism + exact integration points

A small fork patch to the scheduler, co-located with the scheduler-side
connector (both live in the EngineCore process — connector created at
scheduler.py:101), implementing pause/resume as force-preempt + hold +
connector-mediated save/load. It REUSES:
  - preemption free primitive           scheduler.py:321-324
  - n_c_t==0 resume gate                 scheduler.py:447
  - external-cache match path            scheduler.py:454-459, :569-574
  - async-load park + wake               scheduler.py:404-415, :1411-1455
  - worker finished_sending/recving loop scheduler.py:1472-1478
  - proven W1 copy in start_load_kv / save_kv_layer / build_connector_meta

## Minimal-patch surface (LoC estimate)

Upstream (fork) — meaningfully one file:
  1. v1/core/sched/scheduler.py:
     - self.paused: set[str] init (near :145 with the other kv sets)  ~1
     - force_preempt(req_id): mirror :321-324, add to paused           ~15
     - resume_request(req_id): paused -> waiting.prepend               ~5
     - saved-confirm consume (mirror finished_sending at :1475)        ~8
     - read external trigger at top of schedule() (:189)               ~5
  2. EngineCore RPC to expose pause/resume to the harness (~15) OR
     reuse the W1 control-file read inside schedule() (0 new IPC).
     Prefer control-file for the spike-continuous path.
  3. request.py: no change needed (track in scheduler's set).          0

Pure-connector (no upstream patch; extends the proven W1 module):
  - get_num_new_matched_tokens returns saved-token count by request_id  }
  - build_connector_meta emits SAVE directive on pause trigger          } ~35
  - update_state_after_alloc records resume-load blocks                 }
  (start_load_kv / save_kv_layer copies already proven in W1.)

Net-new total: ~70-100 LoC; upstream footprint ~1 file. Genuinely small
— the machinery already exists for P/D disaggregation.

## Risk list (ranked)

1. [Med-High] Save/free ordering race — free before save completes
   corrupts KV. Mitigation: synchronous save (v1) or saved-confirm gate.
   Isolated, testable.
2. [Med] Persistent-batch / model-runner state when a mid-generation
   request leaves and re-enters. Residual is LOW: normal preemption
   already produces scheduled_resumed_reqs (:598), so the runner path is
   exercised today — but VERIFY that loading SAVED KV yields logits
   bit-identical to the recompute path.
3. [Med] get_num_new_matched_tokens prompt-only API contract vs our
   full-sequence resume. Mitigation: key on request_id, pin <0.12,
   document. A future upstream tightening could break it.
4. [Med] Seeded random-sampling RNG offset lost on resume. Mitigation:
   greedy eval; document.
5. [Low] prefix-cache self-hit double count. Mitigation: delta-only
   reporting beyond num_new_local_computed_tokens.
6. [Low] chunked-prefill / spec-decode edges. Mitigation: spec decode
   off; chunked prefill already compatible.
7. [Low] Trigger liveness (finish between trigger and pause). Mitigation:
   reuse the W1 finished_req_ids pruning guard in force_preempt.

## Build-plan estimate (lane C)

  Scheduler pause/resume + paused set + trigger read      2.0 d
  Save-confirm gate (sync-save first, async later)        1.5 d
  Connector resume-by-request-id + pause save-directive   2.0 d
  Worker load-on-resume correctness (logit-identity test) 1.5 d
  Multi-tenant integration (pause frees mem -> admits a
    co-tenant -> resume restores) under real pressure     2.0 d
  Reviewer gate + Q4 edge hardening                       1.5 d
  ------------------------------------------------------------
  Total                                                  ~10.5 d (~2 wk)

The estimate is bounded to a roughly one-file, 70-100-line scheduler
integration over existing P/D-disaggregation machinery, not a from-scratch
swap subsystem. Live readiness is governed by the current roadmap gates.

## Trade-offs

| Option                         | Pros                        | Cons                         |
|--------------------------------|-----------------------------|------------------------------|
| A. Fork scheduler patch (rec.) | Correct layer; reuses       | Touches an upstream file;    |
|    force-preempt + hold set    | preempt+PD machinery;       | pinned to 0.11.x; must track |
|                                | ~70-100 LoC; multi-tenant   | vLLM API drift.              |
| B. Reuse WAITING_FOR_REMOTE_   | Zero new status; max reuse  | Conflates pause with async-  |
|    KVS as the hold state       | of the park/wake loop       | KV; entangles invalid-block  |
|                                |                             | recovery; murkier semantics. |
| C. KVCacheManager direct       | No scheduler edit           | Desyncs running/waiting/     |
|    free + manual bookkeeping   |                             | status + worker batch;       |
|                                |                             | higher risk than A.          |
| D. sleep/wake_up engine level  | ~0 new code                 | Whole-engine; pauses all     |
|                                |                             | tenants; single-tenant only. |

Recommendation: A, with hold-set encoding and synchronous save in v1.

## References (vLLM v0.11.2)

- vllm/v1/core/sched/scheduler.py:277-339  — preemption trigger + free
- vllm/v1/core/sched/scheduler.py:319,291-294 — FCFS/PRIORITY victim pick
- vllm/v1/core/sched/scheduler.py:321-324  — free+PREEMPTED+n_c_t=0 (evict)
- vllm/v1/core/sched/scheduler.py:447      — n_c_t==0 resume gate
- vllm/v1/core/sched/scheduler.py:454-459  — get_num_new_matched_tokens
- vllm/v1/core/sched/scheduler.py:472-474,492-495 — local+external, output
- vllm/v1/core/sched/scheduler.py:551-574  — allocate_slots + alloc update
- vllm/v1/core/sched/scheduler.py:404-415,586 — WAITING_FOR_REMOTE_KVS park
- vllm/v1/core/sched/scheduler.py:1411-1455 — _update_waiting_for_remote_kv
- vllm/v1/core/sched/scheduler.py:1472-1478 — finished_recving/sending loop
- vllm/v1/core/sched/scheduler.py:1247-1309 — finish_requests/_free_blocks
- vllm/distributed/.../v1/base.py:369-402   — get_num_new_matched_tokens API
- vllm/distributed/.../v1/base.py:396-400   — "prompt-tokens" assumption
- vllm/distributed/.../v1/base.py:450-469,301-317 — request_finished/get_finished
- vllm/v1/request.py:90-110,166-167,174     — output tokens survive preempt
- vllm/v1/request.py:223-243                — RequestStatus (PREEMPTED)
- vllm/v1/core/kv_cache_manager.py:335-343  — free() keeps request alive
- vllm/v1/core/kv_cache_manager.py:176-192  — get_computed_blocks/self-hit

## Implementation notes (lane C, appended 2026-07-18 — memo body unchanged)

Built per this memo. Files: `spike/vllm_connector/scheduler.py` (new),
`spike/vllm_connector/{control,core,gpu}.py`, `spike/run_spike.py`
(`--scenario pause`), `tests/test_vllm_connector_spike_logic.py`. CPU suite +
ruff green without vllm; GPU-verified next rental (checklist below).

**Deviation 1 (the only material one): NO fork patch.** The memo assumed a fork
of `v1/core/sched/scheduler.py`. v0.11.2 injects a custom scheduler BY CONFIG,
so `PausableScheduler(Scheduler)` lives in OUR repo, zero forked/vendored files.
Evidence: `SchedulerConfig.scheduler_cls: str | type[object] | None` ("Can be a
class directly or the path to a class of form 'mod.custom_class'"),
`get_scheduler_cls()`→`resolve_obj_by_qualname`; `EngineCore` does
`vllm_config.scheduler_config.get_scheduler_cls()(...)`; surfaced via
`EngineArgs.scheduler_cls`. Selected with
`scheduler_cls="spike.vllm_connector.scheduler.PausableScheduler"`. Strictly
cleaner than a fork and it rides the same `<0.12` pin. `force_preempt` still
mirrors the free primitive (:321-324) exactly, minus the `waiting.prepend`.

**Deviation 2 (minor): save-confirm channel is the timing file, not a new set.**
The v1 synchronous save is confirmed worker→scheduler by a `pause_saved` row the
worker appends to the timing JSONL it already writes; the scheduler polls it with
a line cursor (`SaveConfirmationReader`). Zero new IPC. Upgrade path unchanged:
the async saved-set in `KVConnectorOutput`, consumed like `finished_sending`.

**Resume recomputes a lag-bounded suffix (documented, not a bug).** The 2-phase
pause keeps the request RUNNING across the PAUSE step and however many steps the
synchronous save-confirm takes (memo "save-before-free") — this save-confirm lag
is USUALLY one step but is NOT bounded to exactly one token (chunked prefill /
batching can advance more than one token per step). Whatever tokens accrue
during the lag were never in the save, so on resume `num_computed_tokens` is
restored only to the pause-instant count; the scheduler recomputes the lag
suffix from the bit-identical saved prefix — correct, and preserves greedy
logit-identity. The fail-fast `assert_saved_covers_tokens` pins "saved == n_c_t
at pause instant." The upgrade path (async saved-set in `KVConnectorOutput`,
confirmed the same step the save is queued) shrinks this lag toward zero.

**Resume path uses the matched-tokens seam, not a directive-from-control.** On
RESUME the scheduler moves the held request to `waiting`; the stock :447 gate →
`get_num_new_matched_tokens` (delta-only, keyed on request_id) → `allocate_slots`
→ `update_state_after_alloc` (queues the resume-load) drives the H2D load.
Retained pause-saves use a dedicated pinned buffer per request (out of the
staging pool), freed after the resume-load or, if the request never needs one
(a finish-while-paused abort, or a free self-hit resume where the local prefix
cache already covers the save), by an explicit `release` directive.

**Reviewer fix: `update_state_after_alloc` receives the FULL prefix block set,
not just the delta.** v0.11.2 passes `new_computed_blocks + new_blocks`, so when
the save-confirm lag's suffix crosses a block boundary,
`len(new_block_ids) > saved_block_count` — the retained buffer only covers the
saved prefix. `resume_load_block_ids` (core.py) slices the resume-load to the
first `saved_block_count` blocks; the scheduler's own allocation owns the
recomputed suffix in the rest. A CPU regression test
(`test_resume_load_block_ids_slices_boundary_crossing_suffix`) exercises the
boundary-crossing case directly.

**GPU-session checklist (validate lane C):**
1. `uv pip install -e '.[serving-spike]'`; confirm `scheduler_cls` string
   resolves (engine boots with `PausableScheduler`, not the stock one).
2. `--scenario pause` single-tenant first: confirm `pause_events.jsonl` shows
   `pausing`→`freed`, `blocks_freed>0`, and generation continues post-resume.
3. **Logit-identity**: `identical: true` in the JSON (greedy). This is memo risk
   #2 — the GPU-only unknown: loading saved KV is bit-faithful, not just close.
   If false, inspect the lag-suffix recompute (above) and `--block-dim`.
4. Multi-tenant: under real memory pressure, confirm the freed blocks admit a
   co-tenant during the tool window, then resume restores the agent (memo build
   -plan row 5). Watch `pause_to_freed_ms` and `resume_to_first_token_ms`.
5. Confirm `finished_req_ids` prunes both the connector table and the scheduler's
   pause sets (liveness) — kill a load request mid-pause and check no crash.
6. `--transfer-mode staged` vs the sync retained save: the retained save is
   deliberately synchronous (strided-style) even in staged mode; verify it does
   not stall co-tenant ITL beyond the one-off budget.
7. RESUME-during-save-in-flight: fire RESUME immediately after PAUSE (before the
   `pause_saved` confirmation lands) and confirm the request still resumes —
   `PauseBook.defer_resume`/`pop_deferred_resume` should honor it right after
   `force_preempt`, not drop it (reviewer fix, was a hang-forever bug).
8. Retained-buffer leak check: finish a load request mid-pause and confirm its
   pinned buffer is released (`release` directive, `CudaBlockTransfer.
   free_retained`); separately, drive a low-pressure run where the resumed
   request's tokens are already prefix-cache-local (`num_external_tokens==0`)
   and confirm the buffer is released there too, not just on a real load.
