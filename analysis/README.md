# analysis/ — INDEX (single entry point)

**Read this first.** One line per artifact; read only what your task needs.
Nothing has been moved — all paths referenced by handoffs/provenance are stable.

## ACTIVE (read for any new work)

| Doc | What |
|---|---|
| `HANDOFF-20260720.md` | **CURRENT STATE — read first.** Pre-restore SURVIVED both trigger gates (first new mechanism); C/B/WTN killed; k>1 re-check collapse confirmed; standing directives; lead decisions pending. Supersedes `HANDOFF-20260719.md`. |
| `ROADMAP-mlsys2027-20260717.md` | THE execution plan to MLSys 2027: thesis, keep/kill/modify, 13-week schedule w/ deliverables + go/no-go gates. Start here for "what next". |
| `HANDOFF-tool-time-campaign-20260716.md` | THE campaign handoff: problem, rho=0.94 discipline, architecture seam, 12-finding arc, thesis, code map. |
| `HANDOFF-fresh-cert-20260717.md` | In-flight fresh-corpus certification: state, how to finish/verify, gotchas. Note: run migrated to remote 216.81.248.101 (28c/78GB), fold-sharded (byte-identical, review-APPROVEd). |
| `fresh-corpus-preregistration-20260716.md` | LOCKED confirmatory protocol + amendments (N=277, concurrency=2). H1/H2 decision rules. |
| `related-work-synthesis-20260715.md` | Continuum + ThunderAgent vs us; conditioning spectrum; P1-P4 proposals. Basis for the MLSys 2027 direction. |
| `fresh-corpus-certification-20260717/` | The running confirmation's output root (`$B`). Verdict artifacts per handoff §3. |

## REFERENCE (read on demand)

| Doc | What |
|---|---|
| `tool-time-mechanism-analysis-20260715.md` | Why gates win: all divergence on exec; zero per-tool headroom; disjoint gates → union. |
| `tool-time-formulation-critique-20260714.md` | Running log/critique + full execution order of the campaign. |
| `tool-time-method-report-20260714.md` | Early method report (partly superseded by the campaign handoff). |

## RESULT DIRS (dev-exposed sensitivity record — dated, frozen; findings.md/summary.md inside each)

All on dev corpora (SWE-ReBench-100 / Terminal-Bench / ScienceAgentBench). Sensitivity
only — NEVER quote as certified. Finding numbers refer to campaign handoff §4.

- `tool-time-offline-probe-20260711/`, `tool-time-offline-gated-robust-20260711/`, `tool-time-threshold-sweep-100ms-20260711/` — earliest probe/gate iterations (superseded).
- `tool-time-offline-gated-robust-confirmation-20260712/`, `...-swe-rebench-100-20260713/` — frozen dev confirmation runs (pipeline template for the fresh run).
- `tool-time-dacd-20260714/` — DACD variant (rejected).
- `tool-time-restore-cost-sweep-20260714/` — F1: rho=0 manufactures success (finding 1).
- `tool-time-restore-cost-mode-b-20260714/` — Mode-B refit; +118s vs deadline (finding 2).
- `tool-time-within-task-baseline-20260714/`, `tool-time-within-task-gated-20260714/` — B1 within-task history (finding 3: can't self-certify at rho=0.94).
- `tool-time-transfer-terminal-bench-20260714/`, `tool-time-transfer-science-agent-bench-20260714/` — E2: priors don't transfer; gate bounds damage (finding 4).
- `tool-time-hazard-model-{full,cross-task,within-task}-20260714/`, `tool-time-hazard-model-gbm-full-20260715/` — learned hazard arc (finding 5: GBM wins low-rho, loses to trie at 0.94).
- `tool-time-hazard-ensemble-gbm-20260715/` — ensemble REFUTED (finding 6).
- `tool-time-gate-union-20260715/`, `tool-time-gate-union-terminal-bench-20260715/` — naive OR union (finding 8: wins SWE, fails TB).
- `tool-time-certified-union-{swe-rebench,terminal-bench,tb-lcb}-20260715/` — certified union (finding 9: the keeper).
- `tool-time-rho-measurement-20260715/` — rho=0.94 measured, H100 (finding 10).
- `tool-time-tool-name-baseline-{swe-rebench,terminal-bench}-20260715/`, `tool-time-frontier-terminal-bench-20260715/` — P1 Continuum-class baseline (finding 11).
- `tool-time-recompute-restore-swe-rebench-20260715/`, `tool-time-prefill-cost-20260716/` — P2 reload-vs-recompute (finding 12).
- `tool-time-gate-robustness-swe-rebench-20260716/` — Phase 0 gate calibration (percentile cert ~2.7x anticonservative → permutation cert).
- `tool-time-power-mde-swe-rebench-20260716/` — Phase 0b power/MDE (drove N=277 extension).

## Bulk data note

`rho_*_decisions.jsonl` files inside result dirs are multi-GB reproducible
intermediates — never read them into context; read `findings.md`/`summary.md`/
aggregate JSONs instead.
