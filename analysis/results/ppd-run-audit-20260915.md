# Audit of our PD/PPD runs against the upstream design (2026-09-15)

Prompted by the advisor: check our launch configuration against the upstream's own scripts before spending money on
a rented pool. The upstream is pinned at PPD commit 28aaa63 (ICML 2026, "Not All Prefills Are Equal: Dynamic
Append-Prefill Routing for Disaggregated Multi-turn LLM Serving"); its checkout, launch scripts and proxy are on the
GPU host at `/workspace/.cache/agent-sched-bench/PPD-28aaa63.../`.

## What the upstream does

Four machine roles, not two: **P** (prefill only, KV producer), **D** (decode only, KV consumer), **pD**
(prefill-capable decode: append-prefill + decode locally, keeps a prefix cache), **R** (plain replica, no KV
transfer). Request flow in PPD mode: turn 1 goes P → pD over NCCL; **turn 2+ stays on the pD and is served from its
local prefix cache with no KV transfer at all**. The decision engine
(`ppd/optimizer/ppd_decision_engine.py::should_use_ppd`) chooses per request: turn 1 → always PD; appended input
< 512 tokens → PPD (local); otherwise a lookup table keyed by (context class, workload class, nearest of ten QPS
points), with extrapolation for "huge" contexts. Every instance in the shipped scripts runs with
`--enable-prefix-caching`, KV moves over `P2pNcclConnector` with `kv_role` `kv_producer` / `kv_consumer`, and the
proxy (`ppd/comprehensive_proxy.py`, 934 lines) registers instances by role over ZMQ and hashes conversations for
turn affinity. Twelve topologies ship, including 3P+1D, 2P+2D, 2P+1D+1pD, 1R+2P+1D, 2R+1P+1D and 4R, so P:D ratios
and mixed replica/disaggregated pools are upstream features. Evaluation in the repository is ShareGPT multi-turn.

## What our run actually did (`results/mixed56-vast-ppd-pcie-20260907-r1`)

| Measurement | Value |
|---|---|
| Cached prompt share over 7,968 original steps | **0.013** |
| Decode-side instance | 3,985 requests, mean prompt 25,320 tokens, mean prefill 3.49 s, mean TTFT 97.4 s |
| Prefill-side instance | 114 requests, mean prompt 1,194 tokens |
| Engine KV cache | 273,952 tokens per GPU |
| KV needed for residency | 32 agents × ≈ 25K tokens ≈ 800K, i.e. **2.9× the capacity** |
| Connector | `NixlPushConnector`, `kv_role: kv_both` on both instances |

The decode node re-prefilled the whole 25K-token prompt on essentially every turn. That is the opposite of the
mechanism the paper is about: PPD's premise is that turn 2+ finds its history in the decode node's prefix cache and
only appends. Our run measured the routing rule with a prefix cache that always missed — local full prefill and no
decode isolation, the worst of both paths. Two independent causes, either sufficient: (1) **capacity** — 32
concurrent agent contexts need 2.9× the KV the GPU had, so no policy could have kept them resident; (2)
**configuration** — two symmetric `kv_both` NIXL instances rather than the upstream's producer/consumer pair with a
distinct pD role and NCCL P2P transport.

The fixed-PD run has the mirror-image problem: its prefill worker had no cache hits either (101.7M prompt tokens
prefilled, P saturated for the whole run), so it re-prefilled every history on P every turn.

## What this invalidates

Milestone 3 §3 closes Frontier C on three numbers: fixed PD 71.5 min, public PPD 109.2 min, two-sided 75.0 min
against DualMap's 33.2. The PPD number does not measure PPD, and the closure sentence "prefill/decode disaggregation
is closed for this workload at 2 GPUs" rests on a regime where neither path could retain a single conversation.
**M3 §3 needs an amendment.** It is being edited by another session at the time of writing, so the amendment is
recorded here rather than applied there.

What follows for the open questions: the condition PPD requires is decode-side KV residency across turns. A 141 GB
GPU, or a DRAM offload tier of the kind Milestone 4 §3.3 sizes, is what creates that condition — so "a bigger GPU
removes the reason to want PD" (the pool-simulation reading in M3) is not established either; on our hardware the
mechanism was never given its premise. Any rented test must first show the decode side holding conversations
(cached prompt share well above 0.013) before its PD/PPD comparison means anything.
