# Segment atom-vs-chain variance study

> **FINAL - complete corpus**
>
> EXPLORATORY. Durations replayed on our own hardware (our_hardware); segment_timeline v2 telemetry. Generated 2026-07-18T21:23:01.

## Exclusion accounting (read first)

| quantity | value |
| --- | --- |
| trace files | 277 |
| tool_exec calls | 13410 |
| segment_timeline present | 8953 (66.8% coverage) |
| telemetry_absent | 0 |
| **pipe/loop excluded** | 4129 (46.1% of present) |
| duration-analysable | 4824 |

Reconciliation (raw_total vs sum-of-segments), gap ratio: median 0.126, p10 0.002, p90 0.612; negative-gap tail 0.0% (0 chains).

Chains: 4824 total, 4026 multi-segment across 277 tasks.

## Core decomposition (out-of-sample, task-grouped folds)

Target: chain `parent_total_ms`. tail = P90+ of true values.

| model | R^2 | MAE ms | tail MAE ms | tail R^2 | mid MAE ms |
| --- | --- | --- | --- | --- | --- |
| atom_identity | -0.003 | 1022.425 | 9276.296 | -0.066 | 104.316 |
| atom_plus_args | -0.003 | 1022.480 | 9273.202 | -0.066 | 104.721 |
| chain_prefix_cert | -0.000 | 996.111 | 9059.825 | -0.064 | 99.154 |
| chain_prefix_cdskip | 0.002 | 971.352 | 8904.810 | -0.061 | 88.883 |

## Atom-duration stability (lower CV = more stable across tasks)

| atom | tasks | median ms | CV across tasks |
| --- | --- | --- | --- |
| case | 11 | 0.008 | 0.107 |
| conda | 11 | 0.014 | 0.129 |
| source | 11 | 4.316 | 0.150 |
| __conda_activate | 11 | 0.014 | 0.161 |
| [ | 11 | 0.005 | 0.168 |
| __conda_hashr | 11 | 0.028 | 0.283 |
| head | 14 | 1.014 | 0.359 |
| rm | 11 | 1.586 | 0.363 |
| apt-get | 16 | 4991.777 | 0.379 |
| cd | 277 | 0.072 | 0.425 |
| wc | 20 | 1.812 | 0.433 |
| cat | 275 | 1.383 | 0.457 |
| ls | 127 | 1.914 | 0.646 |
| local | 11 | 91.630 | 0.653 |
| which | 167 | 1.488 | 0.656 |
| echo | 275 | 0.071 | 0.951 |
| find | 18 | 5.956 | 1.706 |
| hash | 11 | 0.023 | 1.773 |
| pytest | 13 | 1013.955 | 1.796 |
| git | 276 | 4.436 | 2.344 |
| __unparsed__ | 139 | 118.554 | 2.500 |
| python | 103 | 0.835 | 2.820 |
| pip3 | 56 | 0.635 | 2.931 |
| pip | 27 | 0.422 | 3.651 |
| grep | 76 | 2.291 | 4.276 |
| python3 | 236 | 121.700 | 5.505 |

## Within-chain variance (top families by chain count)

| family | chains | dominant atom | variance shares | chain CV/tasks |
| --- | --- | --- | --- | --- |
| `cd>>python3` | 1039 | python3 | 0.00, 1.00 | 4.293 |
| `cd>>git` | 665 | git | 0.00, 1.00 | 1.875 |
| `cd>>__unparsed__` | 521 | __unparsed__ | 0.00, 1.00 | 2.288 |
| `cd>>grep` | 288 | grep | 0.00, 1.00 | 2.493 |
| `echo>>cat` | 267 | cat | 0.01, 0.99 | 0.194 |
| `cd>>cat` | 161 | cat | 0.01, 0.99 | 0.226 |
| `cd>>python` | 128 | python | 0.00, 1.00 | 2.950 |
| `cd>>git>>cat` | 102 | git | 0.00, 0.98, 0.02 | 0.317 |
| `cd>>pip3` | 87 | pip3 | 0.00, 1.00 | 2.907 |
| `cd>>echo>>cat` | 70 | cat | 0.01, 0.00, 0.99 | 0.280 |
| `cd>>git>>git` | 57 | git | 0.00, 0.54, 0.46 | 1.348 |
| `which>>python3>>ls` | 41 | which | 0.49, 0.09, 0.42 | 0.262 |
| `which>>python3` | 34 | python3 | 0.01, 0.99 | 0.521 |
| `which>>python3>>python3` | 30 | python3 | 0.00, 0.00, 0.99 | 0.372 |
| `cd>>pytest` | 28 | pytest | 0.00, 1.00 | 2.500 |
| `cd>>__unparsed__>>python3` | 27 | python3 | 0.00, 0.00, 1.00 | 1.402 |
| `cd>>pip` | 25 | pip | 0.00, 1.00 | 4.015 |
| `cd>>__unparsed__>>__unparsed__` | 23 | __unparsed__ | 0.00, 0.00, 1.00 | 2.096 |
| `cd>>ls` | 21 | ls | 0.00, 1.00 | 0.112 |
