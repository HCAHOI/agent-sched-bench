# analysis/ — INDEX

**Read `HANDOFF-20260720.md` first.** Everything else is here because
something still depends on it. Superseded plans and killed-lane design
docs were deleted (recoverable from git) — their outcomes live in
`CLOSED-QUESTIONS.md`, which is the file to read before proposing
anything that sounds like a good idea.

## Start here (4 files, in this order)

| Doc | What |
|---|---|
| `HANDOFF-20260720.md` | Current state: what is banked, what is running, what is parked, standing directives. |
| `PAPER-SKELETON-20260720.md` | The three contributions and the one object they share. Any new work must map to C1/C2/C3 or it does not run. |
| `CLOSED-QUESTIONS.md` | Settled questions with the reason each died. Do not re-open without new data. |
| `ROADMAP-mlsys2027-20260717.md` | 13-week schedule, go/no-go gates, W8 cut line. |

## Live specs (work in progress or pending a gate)

| Doc | Status |
|---|---|
| `rolling-survival-design-20260720.md` | Priced residual-time policies. A0 lemma CONFIRMED; A2 pre-restore SURVIVED both trigger gates. Remaining: GPU live validation. |
| `pressure-headroom-design-20260720.md` | Footprint pricing screen. UNPARKED 2026-07-20; ceiling blocker under repair, then re-review. Carries the dated amendment recording the mid-flight criterion change. Nothing citable until it clears. |
| `fresh-corpus-preregistration-20260716.md` | LOCKED confirmatory protocol (N=277). H1/H2 decision rules. |

## Results (frozen, citable)

| Artifact | Finding |
|---|---|
| `fresh-corpus-certification-20260717/` | **H1 CERTIFIED** (+66.7 s @kv3500, +150.6 s @kv5000 vs deadline, rho=0.94). H2 honest negative. The certified baseline everything else is measured against. |
| `prerestore-accounting-2026-07-20.md` + `-robust-` | **Pre-restore SURVIVED both trigger sources** (+156.2 / +317.9 s per 277 tasks). First new mechanism to clear its gates. Offline; GPU validation outstanding. |
| `prior-calibration-2026-07-20.md` | Priors are well-calibrated (all four quantiles cover nominal) and sharper than pooled (+18.0%) and tool-name (+15.4%) baselines. |
| `adjudication-k2-recheck-2026-07-20.md` | k>1 re-checks collapse to k=1 exactly (0.0 ms gap, 6700 cells). |
| `boundary-evidence-stage1-2026-07-19.md` | Boundary identity: 72.9% decision churn, zero information. |
| `stable-atom-overlap-2026-07-19.md` | Stable-atom screen is a one-atom trick (0.19% reachable mass). |
| `wrapper-transparency-stage1/stage2-2026-07-19.md` | Normalization: accuracy gain, utility regression. The accuracy≠utility evidence. |
| `segment-atom-study-2026-07-19.md` | Five-model atom comparison; why decomposition loses. |
| `tb-sizing-memo-20260720.md` | TB vs SWE opportunity mass; model confound documented. Basis for dropping TB. |

## Reference

| Doc | What |
|---|---|
| `HANDOFF-tool-time-campaign-20260716.md` | Original campaign handoff: problem framing, architecture seam, code map. |
| `related-work-synthesis-20260715.md` | Continuum + ThunderAgent positioning; the conditioning spectrum. |
| `tool-time-mechanism-analysis-20260715.md` | Why gates win: divergence concentrates on exec; disjoint gates → union. |

## Result directories (`tool-time-*/`)

Frozen provenance for the 12-finding dev-corpus arc (protocol.md,
results.md, review.md, per-fold decisions inside each). **Dev-exposed
sensitivity record — never quote as certified.** Kept for
reproducibility; not reading material. The certified results are in
`fresh-corpus-certification-20260717/`.
