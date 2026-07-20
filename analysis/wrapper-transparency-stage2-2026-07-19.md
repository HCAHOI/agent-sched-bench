# WTN Stage-2: certified decision replay (joint policy vs cert)

> **FINAL - complete corpus**
>
> EXPLORATORY. Original-trace replay on our own hardware (our_hardware); collection swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277. Generated 2026-07-20T05:23:37 (git 0a4bc642e2eb67ea3b7a1f9232a1530bb1811323).

**Verdict: KILL**

- KILL: screen did not recover cd transparent in every fold (Stage-1 contradiction)
- KILL: no headline kv cell where P_new and P_0 diverge certified positive

Arms at rho=0.94: P_0 (min_tool_history=1, no normalization), P_new (min_tool_history=5, screen-learned normalization), oracle (cd-only strip, min_tool_history=5; reported, never certified). 277 tasks, 5 folds.

## Screen consistency (K2 on original durations)

cd is a candidate: True; cd transparent every fold: False.

| fold | degenerate | marginal transparent | applied transparent |
| --- | --- | --- | --- |
| 1 | False | ['apt-get', 'cat', 'cd', 'echo', 'find', 'git', 'ls', 'python3', 'which'] | ['apt-get', 'cat', 'cd', 'echo', 'find', 'git', 'ls', 'python3', 'which'] |
| 2 | False | ['apt-get', 'cat', 'cd', 'conda', 'echo', 'find', 'git', 'ls', 'python3', 'which'] | ['apt-get', 'cat', 'cd', 'conda', 'echo', 'find', 'git', 'ls', 'python3', 'which'] |
| 3 | False | ['apt-get', 'cat', 'conda', 'echo', 'find', 'git', 'ls', 'python3', 'which'] | ['apt-get', 'cat', 'conda', 'echo', 'find', 'git', 'ls', 'python3', 'which'] |
| 4 | False | ['apt-get', 'cd', 'conda', 'echo', 'find', 'git', 'ls', 'pip3', 'python3', 'which'] | ['apt-get', 'cd', 'conda', 'echo', 'find', 'git', 'ls', 'pip3', 'python3', 'which'] |
| 5 | False | ['apt-get', 'cd', 'conda', 'echo', 'find', 'git', 'ls', 'python3', 'which'] | ['apt-get', 'cd', 'conda', 'echo', 'find', 'git', 'ls', 'python3', 'which'] |

## Headline permutation cells (P_new vs P_0)

Positive = P_new beats P_0 (task-clustered sign-flip permutation, Bonferroni over 10 costs).

| kv cost | label | p(positive) | p(harmful) | paired delta ms | diverged samples |
| --- | --- | --- | --- | --- | --- |
| 3500 | inconclusive | 0.992 | 0.008 | -59617.605 | 1607 |
| 5000 | inconclusive | 0.928 | 0.072 | -63360.454 | 385 |

## Oracle row (cd-only + me5 vs P_0; reported, never certified)

| kv cost | label | paired delta ms | diverged samples |
| --- | --- | --- | --- |
| 3500 | inconclusive | -50868.347 | 1827 |
| 5000 | inconclusive | -25454.945 | 487 |
