# WTN Stage-1: wrapper-transparency normalization

> **FINAL - complete corpus**
>
> EXPLORATORY. Durations replayed on our own hardware (our_hardware); segment_timeline v2. Generated 2026-07-19T18:53:58.

**Verdict: KILL**

## Step 0 - reachable-mass census (read before accuracy)

4026 multi-segment chains; 28 candidate verb classes. **Framing quantity -- chains with a wrapper BEYOND the leading segment (mass a leading-only stripper cannot reach): 541 (13.4% of chains, 6.9% of parent_total_ms mass).** Upper bound incl. leading cd (near-vacuous -- every multi-segment chain has a non-final segment): 4023 (99.9%).

| verb class | chains | tasks |
| --- | --- | --- |
| `cd` | 3488 | 277 |
| `echo` | 356 | 274 |
| `git` | 203 | 125 |
| `which` | 186 | 161 |
| `python3` | 148 | 121 |
| `ls` | 86 | 73 |
| `conda` | 22 | 22 |
| `python` | 16 | 13 |
| `cat` | 15 | 9 |
| `apt-get` | 15 | 15 |
| `pip3` | 12 | 8 |
| `source` | 11 | 11 |
| `find` | 10 | 3 |
| `wc` | 9 | 9 |
| `timeout` | 6 | 3 |
| `rm` | 5 | 4 |
| `pip` | 4 | 4 |
| `sed` | 3 | 2 |
| `grep` | 3 | 3 |
| `cp` | 3 | 1 |
| `EOF` | 2 | 1 |
| `mkdir` | 2 | 2 |
| `}` | 1 | 1 |
| `template` | 1 | 1 |
| `wget` | 1 | 1 |
| `printf` | 1 | 1 |
| `rmdir` | 1 | 1 |
| `mv` | 1 | 1 |

> Footnote (population mismatch): 3 multi-segment chains use newline-separated (or otherwise unsplit) commands that xtrace splits into segments but token-level candidate extraction sees as a single segment. They sit in the denominator with an empty candidate set; the beyond-leading framing quantity is unaffected (it requires a detected wrapper).

## Knob-matched grid (out-of-sample, task-grouped folds)

Target: chain `parent_total_ms`. Depth 4 fixed; cells differ only in skip mode and min_evidence. tail = P90+.

| cell | R^2 | MAE ms | tail MAE ms | count |
| --- | --- | --- | --- | --- |
| off@me1 | -0.000 | 996.111 | 9059.825 | 4026 |
| off@me5 | -0.001 | 995.108 | 9067.389 | 4026 |
| cd-only@me1 | -0.004 | 1020.260 | 8928.858 | 4026 |
| cd-only@me5 | 0.002 | 971.352 | 8904.810 | 4026 |
| WTN@me1 | -0.005 | 1020.735 | 8924.166 | 4026 |
| WTN@me5 | 0.001 | 972.804 | 8912.725 | 4026 |

### Pre-registered pairwise deltas (paired task-clustered bootstrap)

Statistic: task-weighted mean of per-chain (|y-WTN| - |y-other|); negative = WTN lower error.

| pair | mean delta ms | CI low | CI high | tasks |
| --- | --- | --- | --- | --- |
| WTN_vs_off@me1 | 25.459 | -78.623 | 185.026 | 277 |
| WTN_vs_cd-only@me1 | 0.302 | -2.694 | 2.749 | 277 |
| WTN_vs_off@me5 | -21.029 | -28.002 | -14.055 | 277 |
| WTN_vs_cd-only@me5 | 0.143 | -1.867 | 2.865 | 277 |

## Transparency screen (sole gate)

Primary tolerance 5% of fit-side MAE; min support 5 tasks AND 20 chains; 5 nested inner folds; evidence gate 1. Sensitivity tolerances: [0.02, 0.05, 0.1].

- fold 0: marginal=['cd', 'echo', 'git', 'ls', 'python3', 'which'] joint_ok=True applied=['cd', 'echo', 'git', 'ls', 'python3', 'which'] (fit-side MAE 784.198 ms)
- fold 1: marginal=['cd', 'echo', 'git', 'ls', 'python3', 'which'] joint_ok=True applied=['cd', 'echo', 'git', 'ls', 'python3', 'which'] (fit-side MAE 1019.608 ms)
- fold 2: marginal=['cd', 'echo', 'git', 'ls', 'python3', 'which'] joint_ok=True applied=['cd', 'echo', 'git', 'ls', 'python3', 'which'] (fit-side MAE 1095.525 ms)
- fold 3: marginal=['cd', 'echo', 'git', 'ls', 'python3', 'which'] joint_ok=True applied=['cd', 'echo', 'git', 'ls', 'python3', 'which'] (fit-side MAE 1078.112 ms)
- fold 4: marginal=['cd', 'conda', 'echo', 'git', 'ls', 'python3', 'which'] joint_ok=True applied=['cd', 'conda', 'echo', 'git', 'ls', 'python3', 'which'] (fit-side MAE 1164.923 ms)

## Kill readout

- **K1** (reproduce win, me=1): beats off (CI<0)=False, not worse than cd-only=True -> KILL
- **K2** (cd emergence positive control): candidate_present=True, per-fold transparent=[True, True, True, True, True] -> pass
- **K3** (mass-weighted stability): agreement 0.995 vs floor 90% (Jaccard diagnostic 0.943) -> pass

**Verdict: KILL**
