# Agent Comparison Report

## Summary

| Metric | mini-swe-agent | openclaw |
|--------|----------------|----------|
| Tasks | 3 | 3 |
| Solve rate | 33.3% | 0.0% |
| Patch rate | 33.3% | 0.0% |
| Avg steps | 34.3 | 49.7 |
| Avg elapsed (s) | 629.2 | 510.6 |
| Avg LLM time (ms) | 365109 | 503191 |
| Avg tool time (ms) | 10414 | 6678 |
| Avg tokens | 552296 | 1567719 |
| Avg LLM/total ratio | 0.97 | 0.99 |
| Avg tool diversity | 1.0 | 5.0 |
| Avg completion tokens/step | 877 | 380 |

## Tool Distribution

### mini-swe-agent
- bash: 103

### openclaw
- exec: 76
- read_file: 54
- edit_file: 9
- list_dir: 3
- web_search: 2
- write_file: 2
- web_fetch: 1

## Per-Task Comparison

| Task | mini-swe-agent steps | openclaw steps | mini-swe-agent tokens | openclaw tokens | mini-swe-agent patch | openclaw patch |
|------|---|---|---|---|---|---|
| django__django-11734 | 50 | 80 | 808483 | 3724037 | N | N |
| pytest-dev__pytest-7571 | 41 | 38 | 740033 | 525899 | Y | N |
| sympy__sympy-24443 | 12 | 31 | 108373 | 453220 | N | N |
