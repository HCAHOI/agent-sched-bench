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

| Bucket | Description | Phase B LOC | Target phase |
|--------|-------------|------------:|--------------|
| A: trace-hook emit        | Insert TraceAction emits at `call_server` and each tool `call()` | 0 | Phase C |
| B: streaming shim         | `call_server` streaming + TTFT/TPOT capture | 0 | Phase C |
| C: TOOL_CLASS+import prune | Remove imports/registry entries for unvendored tools (`FileParser`, `Scholar`, `PythonInterpreter`) | 0 | Phase C |

Note: `logical_turn_id` is **not** a patch bucket — per R3 Principle #2, it
is generated in the runner adapter (Phase D), not injected into vendor code.
Vendor source stays turn-semantics-agnostic.

## Phase B import smoke (from Story US-B2, 2026-04-16 conda env `ML`)

Command: `PYTHONPATH=src conda run -n ML python -c "import <module>"` for each target.

| Module | Result | Notes |
|--------|--------|-------|
| `agents.tongyi_deepresearch` | OK | Top-level package discovered via `src` pythonpath |
| `agents.tongyi_deepresearch.vendor` | OK | Vendor submodule |
| `agents.tongyi_deepresearch.vendor.prompt` | OK | No external deps; standalone strings/constants |
| `agents.tongyi_deepresearch.vendor.tool_search` | OK | Imports resolve (incl. any stdlib/requests) |
| `agents.tongyi_deepresearch.vendor.react_agent` | **FAIL** | `ModuleNotFoundError: No module named 'prompt'` — file-local `from prompt import *` assumes CWD-style sibling resolution, breaks under package import |
| `agents.tongyi_deepresearch.vendor.tool_visit` | **FAIL** | Same `ModuleNotFoundError` — also uses `from prompt import *` |

### Phase C follow-up (new bucket discovered in Phase B)

The 2 failing files share a single root cause: upstream assumes execution
from the `inference/` working directory, so `from prompt import *` resolves
as a file-local sibling import. Under a Python package layout, this is
invalid. Phase C must patch these with package-relative imports
(`from .prompt import *` or explicit symbol imports).

Same issue likely applies to any `from tool_search import ...` /
`from tool_visit import ...` style imports — Phase C must audit all
cross-file imports in the 4 vendored files and convert to package-relative.

**New patch bucket (added to Phase C scope)**:

| Bucket | Description | Est. LOC |
|--------|-------------|---------:|
| D: package-import fix | Convert file-local imports (`from prompt import *`, `from tool_search import Search`, etc.) to package-relative (`from .prompt import *`, `from .tool_search import Search`) in `react_agent.py` and `tool_visit.py` | ~5-10 |

This is a tiny, mechanical patch. Does not affect upstream fidelity semantically
(symbols resolve to the same objects); it only adapts the import syntax to
match the vendor package layout. Patch diff will be recorded verbatim in
Phase C.

### Qwen-agent dep status: NOT YET CONFIRMED

Because `react_agent.py` fails at the `from prompt import *` line (very near
top of file), we did NOT reach its `from qwen_agent...` imports this pass.
Phase C should re-run import smoke after the package-import fix; if
`qwen-agent==0.0.26` is absent from conda env `ML`, that failure surfaces
then and becomes Pre-mortem #1 material.

## Deprecation / deletion tracker

- **`src/agents/research_agent/` deletion_deadline**: TBD — set to
  `<Phase I green date> + 3 calendar days` per R3 Principle #3 at Phase J.
- **Interim shim**: env-flag `OMCBENCH_ALLOW_DEPRECATED_SCAFFOLD=1` (introduced
  in Phase G wiring; removed in Phase F along with the old scaffold).
