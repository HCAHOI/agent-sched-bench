# Closed questions

Reopen an entry only with new data or an argument that addresses the recorded
failure. Git references below are intentional recovery points; they do not imply
that every removed artifact is archived.

## Adaptive update scope

| Direction | Decision and evidence | Surviving evidence |
|---|---|---|
| **Completed-task update** | **Retain as the sole development-only adaptive candidate.** Against the warmup snapshot, one fixed-order five-fold replay gained `+0.362 s/277` at `kv=3500 ms` and `+9.050 s/277` at `kv=5000 ms`. This is not order-robust, unopened-stream, or certified evidence. | [`results/prequential-task-update-20260721/`](results/prequential-task-update-20260721/) |
| **Per-call update** | **Drop.** It produced exactly `0.000 s` additional realized utility over completed-task publication at both headline costs. | Same retained prequential result. |
| **Same-repository history** | **Drop.** Delta versus the matched warmup baseline was `-9.760 s/277` at `kv=3500 ms` and `+7.819 s/277` at `kv=5000 ms`; only 7 tasks were affected, with 87 and 29 changed calls respectively. Support was sparse: 61/277 tasks. | [`results/same-repo-history-20260721/`](results/same-repo-history-20260721/) |
| **Same-trace / within-task B1** | **Negative baseline, not an active direction.** At the measured operating regime it was approximately `-398 s` near `rho=1`, while the deployment-appropriate Mode B remained positive. | [`serving/tool-time-rho-measurement-20260715/findings.md`](serving/tool-time-rho-measurement-20260715/findings.md). |
| **Online-first positive activation** | **Invalid protocol, closed without a scientific verdict.** It asked an uninitialized online evidence process to authorize a policy from zero deployment evidence. Deployment instead starts from a development-profiled, offline-authorized policy and permits harmful-only revocation. The replay contributes no positive or negative claim. | Historical replay at Git `897f1b9`; removed by `ab124ce`. |

## Estimator and policy enrichments

| Direction | Decision and evidence | Recovery or result |
|---|---|---|
| **Wrapper transparency / key normalization (WTN)** | **Closed.** The apparent 21 ms MAE gain was caused by a support gate rather than normalization. The joint certified replay was directionally harmful by `-59.6 s/277` at `kv=3500 ms`; no cell certified. Hardcoded `cd` skipping remains oracle/positive-control only. | Stage-1 and Stage-2 artifacts at Git `ab124ce^:analysis/wrapper-transparency-stage{1,2}-2026-07-19.*`. |
| **Boundary-conditioned re-checks** | **Closed.** Boundary identity changed 72.9% of decisions, but paired log-score gain was `-0.0137` nats with task-clustered 95% CI `[-0.41, +0.20]`. | Git `ab124ce^:analysis/boundary-evidence-stage1-2026-07-19.{md,json}`. |
| **Atom and additive segment models** | **Closed.** Naive atom MAE was 1022 versus 996 for the certified trie; argument-conditioned atoms reached 974 but not the 971 `cd`-normalized positive control. Heavy verbs almost always backed off to bare-verb nodes, and 46% of `exec` calls were pipelines/loops without well-defined per-segment durations. The additive segment-cost implementation had no independent surviving result that overcomes this premise failure. | Git `d8ed6b8^:analysis/segment-atom-study-2026-07-19.{md,json}`; implementation history `0fa23a6` and `a391969`. |
| **Stable-atom screening** | **Closed.** It fired on 29/4,824 held-out calls and reached thin-trie calls only 9 times (`0.19%` corpus mass). | Git `d8ed6b8^:analysis/stable-atom-overlap-2026-07-19.{md,json}`. |
| **More than one elapsed-only re-check** | **Closed by reduction and verification.** `k=2` matched `k=1` on 670 nodes × 10 KV cells, maximum value gap `0.0 ms` at tolerance `1e-6`, and loses once check overhead is priced. | [`certification/adjudication-k2-recheck-2026-07-20.md`](certification/adjudication-k2-recheck-2026-07-20.md). |
| **Per-call self-footprint pricing** | **Drop.** Headroom was `+8.75 s/277` at `kv=3500 ms`, simultaneous CI `[-85.00, 101.90]`; the upper bound is below the frozen `+156 s/277` pre-restore bar. This does not bound real multi-tenant pressure. | [`certification/pressure-headroom-2026-07-20.md`](certification/pressure-headroom-2026-07-20.md). |
| **Session-state candidate** | **Retired without a result.** It was implemented but never reviewed or run; deletion carries no positive or negative evidence. | Implementation recovery commit `55bd166`. |

## Corpus and evaluation limits

| Question | Decision and evidence | Reference |
|---|---|---|
| **Can Fresh-277 measure multi-tenant contention?** | **No.** Its apparent concurrency came from collection concurrency, cloud inference had no shared KV cache, and the overlapping tail was dominated by sub-millisecond calls. Contention claims require the live harness. | [`certification/pressure-headroom-design-20260720.md`](certification/pressure-headroom-design-20260720.md). |
| **Should Terminal-Bench traces be expanded?** | **Drop under the current budget and design.** The available corpus confounds benchmark with agent model, so buying more on that axis would not isolate workload effects. | Git `ab124ce^:analysis/tb-sizing-memo-20260720.{md,json}`. |
