# Fresh-corpus certification — FINDINGS (2026-07-17)

Pre-registered confirmatory run (`analysis/fresh-corpus-preregistration-20260716.md`)
on 277 never-seen SWE-ReBench tasks (seed42 skip150, disjoint from all dev
roots, 13,410 tool calls). Operating point rho = 0.94 by construction; paired
sign-flip permutation certificate, Bonferroni family tail 0.0025, 20,000 draws.
Executed on a 28-core remote box as two fold-sharded chains (byte-identity of
the sharded flow gate-APPROVEd); this run itself passed the mandatory
fresh-reviewer gate: **APPROVE, 0 blocking findings** (4 LOW cosmetic — see
"Follow-ups"). Verdict artifacts: `gate-robustness/gate_robustness_rho094.json`,
`frontier-p1/permutation_p1_rho094.json`. Case study:
`case-study-h1-h2-20260717.md`.

## H1 (flagship): certified-union trigger vs fixed deadline — **CERTIFIED**

| kv cost (ms) | perm. p+ | paired delta | note |
|---|---|---|---|
| **3500** | **0.00115** | **+66.7 s** | CERTIFIED (<= 0.0025) |
| **5000** | **0.00015** | **+150.6 s** | CERTIFIED |
| 4500 | 0.0094 | +106.2 s | strong, uncertified |
| 4000 | 0.0139 | +65.6 s | |
| (all 10 cells) | | positive | direction unanimous |

Effect concentration (pre-registered companion; official p reproduced to the
digit before computing): **both certified cells BROAD, not fragile** —
kv3500: 53 positive tasks, top-3 = 17.6% of positive mass, drop-top-3
diagnostic p stays 0.0005–0.0010; kv5000: 113 positive tasks, top-3 = 11.3%,
drop-top-3 p <= 0.0006. Both cells strengthen when the largest task is
removed. Repo-clustered re-resampling corroborates both cells.

Mechanism (exploratory): the win is task-broad but command-concentrated —
apt-get-update prefixes carry ~90% of kv5000's positive mass across many
tasks; kv3500 splits across apt-get and pip/python prefixes. 100% of policy
divergence is on `exec`, replicating the dev mechanism finding.

## H2 (P1, command-prefix trie vs tool-name/Continuum-class) — **NOT certified**

All 10 cells positive point estimates (+9.9 s @kv500 … +107.4 s @kv5000);
min p = 0.0037 @kv500 > 0.0025. Reported as the pre-registered honest
negative: **direction fully replicated, certification missed by dispersion,
not direction** (sign-flip z = 2.68 vs 2.81 needed; ~1.7 ms/task of uniform
lift short; per-task sd 6x mean at kv500).

Case-study diagnosis (exploratory, for the next iteration): the trie's edge
is broad-and-thin; a few deep LOW-SUPPORT `exec` prefixes bleed negative mass
(`apt-get update -qq &&` -19.2 s, `pip3 install` -8.4 s, `apt-get install -y`
-3.5 s @kv5000) while high-support prefixes carry the win (`apt-get update &&
apt-get` +131.0 s, 88 calls). Ranked mechanism fixes: (1) minimum-support
fallback (thin prefix nodes revert to the tool-name trigger); (2) cd-prefix
normalization; (3) support-gated depth. Guard-tuning: no supporting signal.
Notably, `apt-get update -qq &&` is H2's biggest negative but a large H1
positive — the union captures via the other gate exactly where the trie is
noisy (the disjoint-gates thesis, now visible on fresh data).

## Bottom line

The MECHANISM claim holds on never-seen data: fit per workload + certify each
component + deploy only what certifies + fall back to deadline otherwise
delivers a certified, broad, non-fragile win over the deadline at the
measured operating point (H1, 2 cells). The estimator-class contrast vs
tool-name conditioning (H2) is directionally replicated but uncertified at
n=277. Per `../ROADMAP-mlsys2027-20260717.md`: **GO** — this becomes the
paper's offline validity study; headline duty passes to the live-system eval.

## Deviations / provenance notes (all reviewer-assessed as non-invalidating)

- Run migrated mid-flight from the 8-core collection host (OOM at 5-way GBM
  fold concurrency) to a 28-core/78GB box; numpy pinned 2.5.1 to match local;
  manifest paths patched; protocol.md + git objects + pip added to the remote
  env (provenance-layer failures, 3 aggregate restarts, nothing written).
- Fraction rho=0.0 ran as Mode-B plumbing (refit anchor), ~70% of step-3
  wall-clock; flagged post-hoc — see rho directive addendum: future runs get
  a cheap trigger-equality anchor and 0.94-only downstream.
- Corpus dir is named `...n200` but holds 277 tasks (extension amendment);
  `task_ids.txt` (277) governs.

## Follow-ups (queued, W1)

1. Fix `analyze_gate_robustness.py` LOW-1/LOW-2 (stale rho=1.0 note string;
   dev-file default args) before quoting the JSON in a paper.
2. Cheap fraction-zero anchor (trigger-equality only) — saves ~2h+/run.
3. H2 mechanism fixes (min-support fallback first); validate free via
   TraceLab public-corpus replay — no new trace budget exists.
