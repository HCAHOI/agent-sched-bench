# H2 near-miss case study (EXPLORATORY / post-hoc — NOT a certification result)

**Discipline:** every number below is recomputed read-only from the frozen
`rho_0.94_decisions.jsonl`; the official verdict is NOT re-run or overturned.
Drop-task and MDE numbers are *instability diagnostics only*, never a re-verdict.
Reproduced the official min permutation p_positive @kv500 = **0.0037** to the digit
(pipeline call, seed 0, 20000 draws), so the contribution matrix below is faithful.

- n_tasks = 277, n_calls = 13410/cell (134100 total), rho = 0.94.
- Treatment = `offline_gated_robust_trigger_ms` (command-prefix trie);
  baseline = `offline_gated_tool_name_trigger_ms` (Continuum-class tool-name).
- **100% of the trie/tool-name divergence is on tool = `exec`.** Every non-exec
  call has identical triggers → zero contribution. So H2 is entirely a story about
  how `exec` commands get bucketed.

## Q1 — Effect concentration (per-task paired delta)

| cost | net total | #pos tasks | #neg tasks | top-1 | top-3 | top-10 | pos mass | neg mass |
|---|---|---|---|---|---|---|---|---|
| kv500  | +9.9s   | 71  | 70 | 9.1% | 23.1% | 51.3% | +15.7s | -5.8s |
| kv5000 | +107.4s | 112 | 20 | 4.9% | 12.9% | 26.2% | +239.8s | -132.4s |

Top-k = share of **positive** task mass held by the k largest positive tasks.
All 10 cells (top-3 pos share / #neg tasks): kv500 23%/70, kv1000 16%/65,
kv1500 28%/7, kv2000 45%/8, kv2500 25%/6, kv3000 24%/9, kv3500 18%/43,
kv4000 21%/7, kv4500 16%/10, kv5000 13%/20.

**Verdict: broad-thin-edge, not few-big-winners.** At kv500 the single biggest
task is only 9% of positive mass and *half the tasks (70/141 firing) are
net-negative*; the edge is a thin residual of many small ±. At kv5000 the top
task is 4.9% and mass spreads over 112 positive tasks — but note the winners
cluster in one *command family* (Q2), just distributed across many tasks.

## Q2 — Where trie and tool-name diverge (by command prefix `robust_group_key`)

### kv500 — top prefixes by |mass| (12 pos prefixes = +10.6s, 14 neg = -0.7s)
| prefix | #calls | Δmass |
|---|---|---|
| exec:pip3 install --break-system-packages | 58 | +4.51s |
| exec:bash -c | 59 | +1.27s |
| exec:python -c | 30 | +1.08s |
| exec:cd /testbed && pytest | 51 | +0.95s |
| exec:cd /testbed && python | 205 | +0.92s |
| exec:cd /testbed && pip3 | 350 | +0.86s |
| exec:ls /usr/bin/python* /usr/local/bin/python | 9 | **-0.42s** |
| exec:pip3 install | 44 | **-0.12s** |

### kv5000 — top prefixes by |mass| (4 pos = +138.4s, 3 neg = -31.0s)
| prefix | #calls | Δmass |
|---|---|---|
| exec:apt-get update && apt-get | 88 | **+130.98s** |
| exec:apt-get update -qq && | 83 | **-19.20s** |
| exec:pip3 install | 44 | **-8.35s** |
| exec:apt-get install -y | 26 | **-3.49s** |
| exec:pip3 install --break-system-packages | 48 | +3.16s |
| exec:apt-get install -y python3-pip | 3 | +1.99s |

**The whole high-kv win is essentially ONE prefix** (`apt-get update && apt-get`,
+131s = 95% of prefix-positive mass, 88 calls). Its *sibling* apt-get prefixes
(`apt-get update -qq &&`, `apt-get install -y`) and `pip3 install` carry large
NEGATIVE mass. So the trie helps the long, apt-heavy commands but a few deep,
low-support sibling prefixes actively HURT — that residual drag is what pins the
net down toward the tool-name baseline.

## Q3 — Why p misses 0.0025 @kv500 (diagnostic only, NOT a re-verdict)

Bonferroni one-sided family tail = 0.05/(2·10) = **0.0025**; official p@500 = 0.0037
(74/20001). MC floor = 5e-5; MC std-error near this p ≈ 4e-4, so 0.0037 sits
~3 MC-SE and only ~1 effective task above the boundary — a genuinely thin margin.

- Per-task deltas @kv500: mean +35.8ms but **sd 219.7ms** (6× the mean). A handful
  of high-variance tasks fatten the sign-flip null.
- Sign-flip z = total / sqrt(Σ delta²) = 9912 / 3698 = **2.68**; tail 0.0025 needs ≈2.81.
- Rough MDE at this dispersion: net would need ≈10390ms (**~1.05×** the observed
  +9912ms), i.e. ≈**+1.7 ms/task** of extra uniform lift. It is a hair short.
- Drop-top-k |contribution| tasks (independent seed-0 realization, k=0 lands at
  0.0024 within MC noise of the official 0.0037): k=0 →0.0024, k=1 →0.0059,
  k=2 →0.0105, k=3 →0.0030. Removing the single largest-variance task roughly
  doubles p — confirming (a) sign-flip instability from a few large-|Δ| tasks
  *and* (b) a thin margin across the rest. Not one dominant task (top-1 = 7% of
  Σ|contribution|, top-3 = 17%).

**Read:** near-miss = thin broad edge + heavy per-task variance sitting right on
the MC/boundary resolution limit. It failed by dispersion, not by direction.

## Q4 — Mechanism-improvement candidates (hypotheses for the NEXT corpus, ranked by evidence in this data)

1. **Minimum-support fallback to prune harmful deep prefixes** — *strongest.*
   kv5000 negatives are all deep/low-support siblings (`apt-get update -qq &&`
   -19.2s, `apt-get install -y` -3.49s, `pip3 install` -8.35s); kv500 `ls
   /usr/bin/python*` (9 calls) and `pip3 install` (44) are net-negative. Raising
   the per-group min task-support so thin nodes fall back to the tool-name trigger
   would drop negative mass while leaving the high-support winners
   (`apt-get update && apt-get`, 88 calls; `cd /testbed && pip3`, 350) untouched —
   directly attacks the -31s / -0.7s drag seen above.
2. **cd-skip / prefix normalization** — *moderate.* Many keys are
   `cd /testbed && X`; the wrapper fragments support across siblings and the small
   negatives `cd /testbed && git` (-50ms), `cd /testbed && timeout` (-35ms) live
   there. Stripping the leading `cd <dir> &&` before keying pools support onto the
   informative verb (python/pip3/pytest, all positive), plausibly flipping the
   small cd-negatives.
3. **Support-gated max_prefix_depth (shallower on thin nodes)** — *moderate.* The
   sign SPLIT among apt-get siblings (+131 vs -19/-3.5) is a classic over-deep-
   bucketing signature: one extra token over-fits. A shallower `apt-get` node
   would pool them, but risks diluting the +131 winner, so prefer gating depth by
   support rather than a global shallower cap. Same lever as #1, viewed as depth.
4. **Guard-threshold tightening** — *weakest / unsupported here.* `*_guard_normalized`
   fields are tiny (robust ~1e-4) and the failure mode is *wrong-prefix firing*,
   not *too-loose gating*; no computed signal favors a guard change. Deprioritize.

Candidates 1–3 all reduce to the same root: a few low-support `exec` prefixes are
noisier than their tool-name fallback and bleed negative mass. Killing that drag
is the highest-evidence lever for the next corpus.

---

## H1 concentration (certified_union vs deadline, rho=0.94) — EXPLORATORY / post-hoc fragility check
Pre-registered rule: certificate rides on <=3 tasks => FRAGILE. n=277 tasks.
Reproduced official permutation p+ (20000 draws, seed 0): kv3500=0.0011499 (official 0.0011499), kv5000=0.0001500 (official 0.00014999); deltas kv3500=66.7s kv5000=150.6s.

### Per-task concentration
| cell | net | #pos | #neg | top-1 | top-3 | top-10 | posmass | negmass | top3 net share |
|---|---|---|---|---|---|---|---|---|---|
| kv3500 (CERT) | +66.7s | 53 | 7 | 6.6% | 17.6% | 51.4% | +96.3s | -29.6s | 25% |
| kv4000 | +65.6s | 24 | 7 | 7.4% | 21.0% | 66.0% | +103.7s | -38.0s | 33% |
| kv4500 | +106.2s | 77 | 10 | 5.1% | 14.6% | 46.7% | +187.4s | -81.3s | 26% |
| kv5000 (CERT) | +150.6s | 113 | 20 | 4.3% | 11.3% | 24.7% | +272.5s | -121.9s | 21% |

### Drop-top-|contribution| sign-flip p (diagnostic only, NOT a re-verdict; independent MC realization)
| k dropped | p+@3500 | p+@5000 |
|---|---|---|
| 0 | 0.00105 | 0.00030 |
| 1 | 0.00050 | 0.00015 |
| 2 | 0.00055 | 0.00055 |
| 3 | 0.00100 | 0.00060 |
Top-3 positive tasks @kv3500: facelessuser__pymdown-extensions-2039 (+6.3s), matrix-org__synapse-5415 (+5.7s), mikedh__trimesh-1331 (+5.0s)
Top-3 positive tasks @kv5000: Duke-GCB__lando-173 (+11.8s), PennLINC__xcp_d-1073 (+9.6s), Luke-Poeppel__decitala-193 (+9.5s)

### Divergence (union fires early) by group key

kv=3500: divergent calls by tool: {'exec': 1136}
| group | #calls | Δmass |
|---|---|---|
| exec:apt-get update -qq && | 83 | +41.50s |
| exec:apt-get update && apt-get | 88 | +23.63s |
| exec:cd /testbed && pytest | 33 | -6.54s |
| exec:pip3 install --break-system-packages | 33 | +5.90s |
| exec:cd /testbed && python3 | 641 | +3.08s |
| exec:pip3 install | 44 | -2.31s |
| exec:apt-get install -y | 26 | +0.79s |
| exec:cd /testbed && python | 86 | +0.64s |
| exec:bash -c | 30 | -0.42s |
| exec:cd /testbed && timeout | 10 | +0.21s |
| exec:cd /testbed && apt-get | 6 | +0.21s |
| exec:cd /testbed && pip | 23 | +0.00s |
posmass=+76.0s (8 groups), negmass=-9.3s (3 groups)

kv=5000: divergent calls by tool: {'exec': 297}
| group | #calls | Δmass |
|---|---|---|
| exec:apt-get update && apt-get | 88 | +130.98s |
| exec:apt-get update -qq && | 83 | +15.76s |
| exec:pip3 install | 44 | -8.35s |
| exec:cd /testbed && apt-get | 3 | +5.69s |
| exec:apt-get install -y python3-pip | 4 | +4.61s |
| exec:apt-get install -y | 26 | -3.49s |
| exec:pip3 install --break-system-packages | 48 | +3.16s |
| exec:apt-get update | 1 | +2.28s |
posmass=+162.5s (6 groups), negmass=-11.8s (2 groups)

### Verdicts (pre-registered fragility rule: cert riding on <=3 tasks => FRAGILE)
- **kv3500: BROAD.** 53 positive tasks; top-3 = 17.6% of positive mass (25% of net). Dropping the top-1/2/3 |contribution| tasks leaves p+ at 0.0005-0.0010, all well under the 0.0025 tail.
- **kv5000: BROAD.** 113 positive tasks; top-3 = 11.3% of positive mass (21% of net). Drop-top-k p+ stays 0.00015-0.0006 for k=1..3.
- Context cells kv4000/kv4500 (uncertified) show the same broad shape (top-3 21%/15% of positive mass).
- Group-level caveat (exploratory): task-level breadth notwithstanding, the kv5000 positive mass is dominated by one command family (apt-get update prefixes, +147s of +162s divergent-positive mass, spread over ~171 calls in many tasks); kv3500 is more evenly split across apt-get and pip/python prefixes. Breadth is over tasks (the certified inference unit), not over command families.
- Divergence is 100% on tool=exec at both cells (1136 calls @3500, 297 @5000); sanity: group Δmass sums to the official net at both cells (66.7s, 150.6s).
