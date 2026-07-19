# Candidate B stable-atom overlap diagnostic

> **FINAL - complete corpus**
>
> EXPLORATORY. Durations replayed on our own hardware (our_hardware). Generated 2026-07-19T16:52:40.

**Verdict: KILL**

## Screen selection (cross-fitted)

- per-fold qualifying: [['apt-get', 'pytest'], ['apt-get'], ['apt-get'], ['apt-get'], ['apt-get']]
- intersection (all folds): ['apt-get']
- union (any fold): ['apt-get', 'pytest']
- mean pairwise Jaccard: 0.800
- fold-stable: False

## Overlap with the chain-prefix trie

| quantity | value |
| --- | --- |
| held-out calls | 4824 |
| divergent (stable atom fires) | 29 |
| overlap (trie fallback/thin AND fires) | 9 |
| strict fallback (source != prior_group) | 0 |
| overlap / divergent | 31.034% |
| overlap / total | 0.187% |

Firing atoms (divergent): {'apt-get': 18, 'pytest': 11}

Firing atoms (overlap): {'apt-get': 5, 'pytest': 4}

## Kill readout

KILL if overlap/divergent < 5.0% OR the atom selection is fold-unstable; SURVIVE otherwise.

- overlap below bar: False
- fold-stable: False
- **verdict: KILL**
