# Vendor Notes: Tongyi-DeepResearch

Vendored implementation of the Alibaba-NLP/DeepResearch ReAct scaffold,
pinned to support Ralplan R3 (see `docs/CURRENT_PLAN.md` lines 1-120).

## Upstream pin

- **Upstream**: https://github.com/Alibaba-NLP/DeepResearch
- **Pinned commit SHA**: `f72f75d8c3eb842f2bbbab096a12206ff66e270f`
- **Pinned date**: 2026-02-27 (upstream commit date, "fix bug")
- **License**: Apache-2.0 (full text at `vendor/NOTICE`)
- **Local clone used for vendoring**: `/private/tmp/deepresearch-recon/DeepResearch`

## Freeze protocol

Pin is **frozen** from Phase A through Phase F completion (R3 Pre-mortem #4).
Re-pin is allowed only for upstream blocker bugs during Phase C/D/E, and
requires re-running the patch-bucket audit and re-applying patches. Post-Phase-F
upstream bumps are handled as an independent follow-up PR, not within R3 scope.

## Vendored files

Only the 4 files required by R3 Principle #5 (enabled-tools-only scope) are
vendored. All other upstream files (`tool_scholar.py`, `tool_python.py`,
`tool_file.py`, `run_multi_react.py`, `run_react_infer.sh`, `eval_data/`,
`file_tools/`, …) are intentionally NOT vendored — add as Follow-up #2 if
future experiments require them.

| File | Upstream path | SHA256 | Status at Phase B |
|------|---------------|--------|-------------------|
| `vendor/react_agent.py` | `inference/react_agent.py` | `c9b7f64eb6b870c56ddc3d3e43fd54d6ffb2f75f97b174ac699b36871fe879eb` | verbatim, unpatched |
| `vendor/prompt.py`      | `inference/prompt.py`      | `6e4262c05e44a4104b349226e6d009aa442d536a011ae3abe2f48c3a4a222315` | verbatim, unpatched |
| `vendor/tool_search.py` | `inference/tool_search.py` | `b2cf78dd5d6766bcfa39971e4ed220efccdc6e16f1b3d43769a3ac74df1614f7` | verbatim, unpatched |
| `vendor/tool_visit.py`  | `inference/tool_visit.py`  | `9fea1ce1735d33a98320c021e457e664c7352a5a91185fafcf07957bfca5e6ab` | verbatim, unpatched |

Byte-for-byte identity verified against the upstream clone at pinned SHA
`f72f75d8c3eb842f2bbbab096a12206ff66e270f` on 2026-04-16 during Phase B.
Recompute with `shasum -a 256 vendor/*.py` to re-verify.

## Patch buckets (R3 Phase C scope — NOT applied at Phase A+B boundary)

All four buckets are at **zero LOC** at Phase A+B completion. Patches land in
Phase C (separate ralph run per user directive). Tracked here for audit.

| Bucket | Description | Actual LOC (+/-) | Applied in |
|--------|-------------|------------------|------------|
| A: trace-hook emit        | TraceAction emits at LLM + tool call sites | **0 / 0** | Adapter-side (zero vendor patch; adapter monkey-patches `vendor.OpenAI` and `vendor.TOOL_CLASS` with traced equivalents) |
| B: streaming shim         | `call_server` streaming + TTFT/TPOT capture | **0 / 0** | Adapter-side (internal to TracedStreamingOpenAI) |
| C: TOOL_CLASS+imports prune + dead code | Drop `FileParser/Scholar/PythonInterpreter` imports + registry entries; drop dead `python` / `parse_file` branches in `_run` + `custom_call_tool`; drop unused `import asyncio` | **7 / 44** (US-C1) | Phase C |
| D: package-import fix     | `from prompt import *` → `from .prompt import *` and analogous for `tool_search`, `tool_visit`, `tool_visit`'s `EXTRACTOR_PROMPT` | **4 / 1** (US-C1) | Phase C |

Total vendor patch footprint as of Phase C completion: **11 additions, 45 deletions** across 2 files (`react_agent.py`, `tool_visit.py`). No numerical LOC hard limit per user directive; recorded here for audit.

Note: `logical_turn_id` is **not** a patch bucket — per R3 Principle #2, it
is generated in the runner adapter (Phase D), not injected into vendor code.
Vendor source stays turn-semantics-agnostic.

### Adapter-side zero-patch strategy (Phase C design refinement)

R3 originally budgeted Bucket A + B as vendor patches. During Phase C kickoff
it was realized `vendor.OpenAI` and `vendor.TOOL_CLASS` are **module-level
attributes**, so the adapter (Phase D) can monkey-patch them with traced
equivalents. This eliminates ~70 LOC of vendor patch and moves trace logic
entirely into the adapter layer, maximizing fidelity preservation. Vendor
source only changes for the two concerns that cannot live in the adapter:
(C) runtime-unresolvable imports of unvendored tool files, and (D) package
layout mismatch (file-local imports).

### Known trace coverage gap: Visit tool's summarization LLM

`vendor/tool_visit.py` defines its own `Visit.call_server()` method that makes
a **separate LLM call** to a summarization model via the env-configured
`API_KEY` / `API_BASE` / `SUMMARY_MODEL_NAME` endpoint, using a raw
`openai.OpenAI()` client (not via our monkey-patched `TracedStreamingOpenAI`).
These extractor calls happen every time the Visit tool successfully fetches a
page through Jina Reader, to distill webpage content into evidence.

**Implication**: `summary.total_llm_ms` and `n_turns` under-report by the
wall-ms and call count of these extractor LLM calls. The Visit tool's total
wall time (including the sub-call) is correctly captured in the tool_exec
TraceAction's `duration_ms`, so the upstream-facing latency is right, but the
LLM-call breakdown inside the tool is opaque.

Why not trace it: the extractor is an implementation detail of the Visit tool,
not a ReAct turn, and tracing it would require a second vendor patch. For
scheduling analysis at the ReAct-turn granularity, this is acceptable (the
wall time is still attributed). If future research needs visibility into the
extractor latency distribution, the adapter can patch vendor_tool_visit's
inner `OpenAI` import the same way the main ReAct loop patches it.

## Import smoke (conda env `ML`, command `PYTHONPATH=src conda run -n ML python -c "import <module>"`)

| Module | Phase B (pre-patch) | Phase C (post-patch) |
|--------|---------------------|----------------------|
| `agents.tongyi_deepresearch`                    | OK | OK |
| `agents.tongyi_deepresearch.vendor`             | OK | OK |
| `agents.tongyi_deepresearch.vendor.prompt`      | OK | OK |
| `agents.tongyi_deepresearch.vendor.tool_search` | OK | OK |
| `agents.tongyi_deepresearch.vendor.react_agent` | FAIL (`ModuleNotFoundError: prompt`) | **OK** (after Bucket D) |
| `agents.tongyi_deepresearch.vendor.tool_visit`  | FAIL (`ModuleNotFoundError: prompt`) | **OK** (after Bucket D) |

### Qwen-agent dep status (resolved in Phase C)

`qwen-agent==0.0.34` is installed in conda env `ML` (newer than R3's original
`==0.0.26` pin; API-compatible based on successful vendor import). Pre-mortem
scenario #1 (dep conflict) did not fire. No install/pin adjustments needed
for downstream phases.

## Phase E smoke log (2026-04-16, UTC 09:45:06)

First end-to-end invocation of `TongyiDeepResearchRunner` against a real
cloud backend. Invocation:
`python scripts/smoke_tongyi_deepresearch.py --provider dashscope --model qwen-plus-latest --max-iterations 4`.
Synthetic task: "What is the capital of France?".

| Field | Value |
|---|---|
| provider | dashscope |
| model | qwen-plus-latest |
| api_base | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| exit_status | `completed` |
| vendor_termination | `answer` |
| n_turns | 1 |
| total_llm_ms | 2156.8 |
| total_tool_ms | 0.0 |
| total_tokens | 960 |
| transport_retry_count | 0 |
| llm_call actions | 1 (non-retry) |
| tool_exec actions | 0 |
| transport_retry spans | 0 |
| final_answer | `"Paris"` |

AC#4 invariants: every llm_call `TraceAction` has non-None `ttft_ms` + `tpot_ms`
on the DashScope streaming endpoint — **PASS**. Non-Tongyi model
(`qwen-plus-latest`, not the 30B-A3B) still produced valid `<think>...<answer>`
output on a trivial prompt, so the ReAct prompt format is not strictly Tongyi-
model-specific for simple cases (ACCURACY IS NOT A CONSTRAINT here — this is a
scheduling-trace validation).

## Phase I smoke log (2026-04-16, paid 1+1 via CLI)

Full `trace_collect.cli` pipeline against DashScope, 1 task each on
deep-research-bench + browsecomp, `--max-iterations 8`. Two bugs surfaced
and fixed during this phase:

1. **action_id counter reset per round**: vendor constructs a fresh `OpenAI`
   client inside every `call_server` invocation, so the per-instance counter
   started at 0 each round — every `llm_call` got `action_id="llm_1"`. Fix:
   runner now injects a shared `call_counter` + `retry_state` into the bound
   factory so IDs stay monotonic across vendor's per-round rebuilds.
2. **tool_exec not serialised to trace file**: vendor's `custom_call_tool`
   aliases `tool_args["params"] = tool_args` (self-reference), breaking
   `json.dumps` inside `log_trace_action`. Actions were captured in memory
   (so summary saw them) but never reached disk. Fix: tool wrapper strips
   the self-reference before recording `tool_args` in the TraceAction.
3. **Env-var name mismatch**: vendor reads `SERPER_KEY_ID` / `JINA_API_KEYS`,
   our repo convention is `SERPER_API_KEY` / `JINA_API_KEY`. Runner now
   aliases on demand at `run_task` entry.

### DRB: instance 51 (Japan elderly population + consumption, `qwen-plus-latest`)

| Field | Value |
|---|---|
| llm_call actions | 7 (`llm_1` .. `llm_7`) |
| tool_exec actions | 6 (5 search + 1 visit) |
| transport_retry_count | 0 |
| total_llm_ms | 74 515 |
| total_tool_ms | 55 579 (search ~4–9 s each; visit 22 s) |
| total_tokens | 68 883 |
| ttft_ms range | 740 – 3 244 |
| tpot_ms | ~22.0 (constant) |
| vendor_termination | `answer` |
| exit_status | `completed` |
| AC#2 React triplet | **PASS** (`<tool_call>` in LLM output + 6 tool_execs fired) |
| AC#4 ttft/tpot non-None | **PASS** (7/7) |

### browsecomp: instance 0 (named-person retrieval, `qwen-plus-latest`)

| Field | Value |
|---|---|
| llm_call actions | 1 (`llm_1`) |
| tool_exec actions | 0 (model answered directly from training memory) |
| transport_retry_count | 0 |
| total_llm_ms | 116 635 |
| total_tool_ms | 0 |
| total_tokens | 28 748 |
| ttft_ms | ~2 000 |
| tpot_ms | ~22 |
| vendor_termination | `answer` |
| exit_status | `completed` |
| AC#4 ttft/tpot non-None | **PASS** (1/1) |

Note: browsecomp task did not trigger tool use — the model's training memory
had the answer directly. Per R3 AC#2 "**at least one** produced trace contains
a complete ReAct triplet", the DRB trace satisfies this (browsecomp trace
does not, but the AC is over the set of Phase I traces, not per-task). The
browsecomp trace is still a valid scheduling-analysis artifact (one long
LLM turn with real TTFT/TPOT).

Total smoke cost: ~97k tokens on `qwen-plus-latest` ≈ **$0.14** (well under
R3's $2 paid-smoke budget).

## Deprecation / deletion tracker

- **`src/agents/research_agent/` deletion_deadline**: TBD — set to
  `<Phase I green date> + 3 calendar days` per R3 Principle #3 at Phase J.
- **Interim shim**: env-flag `OMCBENCH_ALLOW_DEPRECATED_SCAFFOLD=1` (introduced
  in Phase G wiring; removed in Phase F along with the old scaffold).
