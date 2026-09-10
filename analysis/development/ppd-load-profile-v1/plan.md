# Controlled-load PD vs local profiling (stage `load`)

Purpose: measure turn-2 TTFT, decode TPOT and E2E for the PD path and the
local-prefill path when N turn-2 requests arrive together on an otherwise idle
system, for N in 8, 16, 32. The 2026-09-08 matrix submitted each turn 2 right
after its own turn 1 while other conversations' turn-1 prefills queued on P, so
it compared PD with a saturated, cold P against local with a warm, idle D and
never observed the online regime (uncached 8K–110K tokens, 8+ active requests
on D). Here all N histories are built first (turn 1 via PD, 32 output tokens),
the proxy is polled until nothing is outstanding, then all N turn-2 requests
are released at once. Cache state is not forced: it follows from N × history
against KV capacity and is read from D-side cached tokens and transfer sizes.

Points are the three most frequent long-history cells of mixed56 (from the
2026-09-08 plan; cell frequency over 1,207 deduplicated requests). Short
histories are settled by the matrix.

| Point | H: history | U: appended input | O: output | Cell frequency | Source task / turn |
|---|---:|---:|---:|---:|---|
| P17 | 22924 | 256 | 186 | 111 | PennyLaneAI__pennylane-5857 / 25 |
| P29 | 40524 | 1021 | 229 | 79 | tobymao__sqlglot-2521 / 41 |
| P35 | 75646 | 298 | 211 | 19 | PennyLaneAI__pennylane-4161 / 61 |

Schedule: 3 points × N ∈ {8, 16, 32} × {pd, local}, seed 42, 18 groups.
Same engines and settings as the matrix (vLLM 0.28.0, NIXL, 8 sequences,
2,048-token batch budget). Each group is preceded by a prefix-cache reset on
both engines. Per-request timeout 1,800 s. Estimated cost about 2.5 h,
dominated by the N=32 groups at P35.
