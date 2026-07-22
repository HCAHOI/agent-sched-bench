# Current claims and evidence limits

The paper studies one object: a conditional residual-time prior for tool calls,
used to price KV-cache actions. Swap-out and pre-restore are two decision points
on the same call lifecycle, not independent modules.

## C1 — Priced stopping over certified latency priors

**Supported offline claim.** On the frozen Fresh-277 evaluation at the measured
`rho=0.94`, the certified-union swap trigger beats the fixed deadline by
`+66.7 s` at `kv=3500 ms` and `+150.6 s` at `kv=5000 ms`. Both certified cells
are task-broad rather than single-task wins. The authoritative artifact is
[`fresh-corpus-certification-20260717/findings.md`](fresh-corpus-certification-20260717/findings.md).

Pre-restore also passed its predeclared offline accounting gate under both
trigger sources. The robust clock yields `+156.2 s/277` at `kv=3500 ms`
(clustered 95% CI `[60.4, 256.6]`) and `+317.9 s/277` at `kv=5000 ms`
(`[181.0, 460.4]`). These numbers account for restore cost but cannot measure
live PCIe contention or serving interference. See
[`certification/rolling-survival-design-20260720.md`](certification/rolling-survival-design-20260720.md).

**Not yet supported.** There is no claim that the policy improves live
multi-tenant JCT, P99 TTFT, throughput, or GPU-memory efficiency. W5 has no
result. The harness remains in development and is blocked by a missing prefill
profile and an unresolved `rho=1.0` trigger-table versus `rho=0.94` runtime
contract.

## C2 — Decision utility, not estimator fit, decides what ships

**Supported claim.** Conditioning refinements must be judged by policy utility
at the operating point. Wrapper transparency is the concrete counterexample:
the apparent 21 ms fit improvement was confounded with a support gate, while the
joint policy was directionally harmful by `-59.6 s/277` at `kv=3500 ms`. The
gate therefore prevented an estimator-motivated regression.

Fresh-277 H2, command-prefix conditioning versus the tool-name baseline, is an
honest negative: all ten point estimates were positive, but the minimum
permutation p-value was `0.0037`, above the preregistered `0.0025` threshold.
The project may claim directional replication, not certification of that
estimator-class contrast.

The offline certificate authorizes only the initialized policy against its
specified fallback. It does not automatically certify later learned states.

## C3 — The elapsed-only re-check space is closed

**Supported claim.** With call survival as the only runtime observation, any
elapsed-only multi-check plan reduces to one precomputable stopping time. The
independent `k=2` dynamic program matched `k=1` on all 670 prior nodes across 10
KV cells, with maximum value gap `0.0 ms` at tolerance `1e-6`, and becomes
strictly worse when check overhead is priced.

Boundary identity is the measured flank: it changed 72.9% of decisions but had
paired log-score gain `-0.0137` nats with task-clustered 95% CI
`[-0.41, +0.20]`. It adds churn without measurable information.

This closure is limited to elapsed-only and observed-boundary enrichments under
the frozen functional. It does not resolve real multi-tenant pressure, which
Fresh-277 cannot represent.

## Adaptive deployment candidate

Completed-task publication is the sole retained adaptive candidate. In the
fixed-order, five-fold development replay, it improved on the warmup snapshot
by `+0.362 s/277` at `kv=3500 ms` and `+9.050 s/277` at `kv=5000 ms`. Per-call
publication added exactly zero realized utility beyond completed-task updates.
The evidence is
[`results/prequential-task-update-20260721/`](results/prequential-task-update-20260721/).

This screen is not an unopened-stream evaluation, a deployment certificate, or
evidence of order robustness. Same-repository conditioning is dropped, and the
same-trace B1 result remains a negative baseline rather than an active policy
direction.

## Explicit non-claims

- No live W5 or headline systems result exists.
- No later adaptive model state is certified by the initial offline certificate.
- No contention conclusion comes from Fresh-277; collection used a cloud model
  and had no shared KV cache.
- No per-call, same-repository, same-trace, wrapper-normalized, atom/segment, or
  boundary-conditioned policy is an active direction.
- The removed online-first replay is protocol-invalid for deployment and
  contributes no positive or negative paper result.
