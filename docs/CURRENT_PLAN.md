# Plan: Add SWE-rebench as a Second Benchmark (Benchmark Plugin Refactor)

**Branch:** `dev/swe-rebench-plugin` (do NOT commit to `main`)
**Date:** 2026-04-07
**Author:** Planner (ralplan consensus, REVISION 1 — addresses Architect + Critic verdict ITERATE)
**Status:** REVISION 1 — addresses 14 Critic must-fixes + 3 Architect top fixes + 3 bonus items
**Supersedes:** REVISION 0 of this plan (same path)

---

## 0. Verified by Planner (pre-revision grep checks)

The orchestrator asked me to verify three bonus items by direct codebase
inspection. Results:

| Check | Command | Result |
|---|---|---|
| `from_swebench_instance` callers in repo | `grep -rn "from_swebench_instance" src/ scripts/ tests/` | Definition: `src/agents/openclaw/eval/types.py:32`. Callers: `src/trace_collect/collector.py:609` and `scripts/run_nanobot_eval.py:71`. **No tests call it directly**, no other production callers. The "one remaining caller" claim in REVISION 0 was wrong by one — there are **two** call sites (collector + script). Both are touched in Phase 4 below. |
| `configs/workloads/code_agent.yaml` | `ls configs/workloads/code_agent.yaml` | **Exists**. Phase 5's plan to add a `benchmark:` field to it is valid. |
| `tests/test_trace_logger.py` | `ls tests/test_trace_logger.py` | **Exists**. REVISION 0 mislabeled it as `(new)`; it is `(extended)` in this revision. |

Additional facts I verified by reading source (used throughout the revision):

- `src/agents/openclaw/eval/types.py:27` — `EvalTask.fail_to_pass` is already
  `list[str]`. `from_swebench_instance()` (lines 31-64) parses the JSON string
  to a list at lines 36-52. **The `EvalTask` layer is already native-list.**
  REVISION 0's "convert rebench list → JSON string" was a list→string→list
  round trip.
- `src/agents/swebench_data.py:33-49` (`derive_test_cmd`) and `:59-67`
  (`_count_fail_to_pass`) — these are the **only two functions in the
  codebase** that require `FAIL_TO_PASS` to be a JSON string. They each have
  a 4-line `if isinstance(raw, str): json.loads(...)` branch and a fallback
  list branch already; the list branch was added defensively but is currently
  reached only via `_count_fail_to_pass`'s `len(raw)` else clause. Fixing
  these to always normalize to a list at the top of each function deletes
  the entire need for an O2 round trip.
- `src/trace_collect/collector.py:747-794` — `_normalize_openclaw_trace`
  hand-rolls `metadata = {...}` (lines 759-782) that does **NOT** include
  `trace_format_version`. Line 789: `if rec.get("type") == "trace_metadata":
  continue` — drops the original metadata record produced by
  `_session_runner.py:531-539` (which **does** correctly set
  `trace_format_version: 4`). Every openclaw trace produced through the
  collector today is missing `trace_format_version` at write time. This is
  a write-path bug, not just a missing field. **The fix is to route
  `_normalize_openclaw_trace` through `TraceLogger.log_metadata()`** (defined
  at `src/harness/trace_logger.py:35-48`, which stamps
  `trace_format_version: 4` automatically).
- `src/trace_collect/collector.py:274-280` — `collect_traces` currently has
  signature defaults `harness_dataset="princeton-nlp/SWE-bench_Verified"`,
  `harness_split="test"`, `harness_namespace="swebench"`. The same defaults
  appear at `collector.py:557-563` for `_collect_openclaw`. Both must be
  removed in Phase 4. The new signature is written down explicitly in
  Phase 4 below.
- Hardcoded dataset references in source (`grep -rn "princeton-nlp/SWE-bench_Verified|SWE-bench_Verified|\"swebench\"" src/`):
  - `src/agents/swebench_data.py:23`
  - `src/trace_collect/swebench_harness.py:17` (`importlib.util.find_spec("swebench")` — package check, not dataset; OK to keep)
  - `src/trace_collect/swebench_harness.py:62` (`namespace: str | None = "swebench"` default)
  - `src/trace_collect/collector.py:274` (signature default)
  - `src/trace_collect/collector.py:280` (signature default)
  - `src/trace_collect/collector.py:557` (`_collect_openclaw` signature default)
  - `src/trace_collect/collector.py:563` (`_collect_openclaw` signature default)
  - `src/trace_collect/cli.py:152` (CLI default)
  - `src/trace_collect/cli.py:184` (CLI default)
  Phase 7's grep MUST cover all 8 of these (Critic item 14).
- `tests/test_gantt_builder_parity.py` — already exists. It compares Python
  vs JS builder for the **same** input. It does **not** catch pre-refactor
  vs post-refactor drift. Phase 1 adds `tests/fixtures/legacy_gantt_payload.json`
  + a snapshot test (Critic item 13).

---

## 1. RALPLAN-DR Summary

### 1.1 Principles (invariants we refuse to break)

1. **P1 — Benchmark is a plugin, not a special case.** Every benchmark
   integration goes through the same protocol; there are no
   `if benchmark == "swe_bench"` branches in the collector, CLI, or runner.
   The collector only knows about plugin handles.
2. **P2 — Trace format is a contract.** The contract is enforced at write
   time: every trace_metadata record is produced by `TraceLogger.log_metadata()`
   (which stamps `trace_format_version`). We bump `v4 → v5` in this plan
   because (a) the openclaw write path is currently lying about v4 by
   producing a record that has no version stamp at all, and (b) we are
   adding required fields (`benchmark`, `benchmark_split`) whose presence
   we want to assert at read time. v5 gives us a clean reset; legacy v4
   traces are read-time-backfilled.
3. **P3 — Scaffolds stay benchmark-agnostic.** mini-swe-agent and openclaw
   see a task dict with the fields they already consume (`instance_id`,
   `repo`, `base_commit`, `problem_statement`, `test_cmd`); benchmark-specific
   quirks (schema normalization, Docker image selection, harness args) are
   absorbed by the plugin before the task reaches the scaffold.
4. **P4 — Output paths are YAML-configured per benchmark.** No defaults
   hardcoded in the collector. Every benchmark's `trace_root`, `data_root`,
   and `repos_root` lives in `configs/benchmarks/<slug>.yaml`. The collector
   reads `benchmark.config.trace_root` and writes there. There is no plugin
   code override of `build_run_dir()`; the function reads from config.
5. **P5 — No mocks, no cherry-picking, no benchmark-specific tuning, no
   undocumented selection filters.** Per CLAUDE.md, real filtered splits,
   real official harness, no dataset-specific magic. Specifically:
   `meta.is_lite` exclusion is **not** a default; it is a YAML knob
   `exclude_lite: bool` defaulting to **`false`**. If we ever default
   `exclude_lite: true` for an experiment, the rationale must be in the
   YAML's prose comment block — not in Python code.

### 1.2 Decision Drivers (ranked, top 3)

1. **D1 — Don't break the v5 trace consumer contract going forward.** We
   bump to v5 once, fix the openclaw write-path leak, and then the contract
   is enforced at write time forever after. Legacy v4 traces stay readable
   via read-time backfill.
2. **D2 — Make BFCL v4 (next target) cheap to add.** BFCL v4 is a
   function-calling benchmark, not a patch-application benchmark — the
   plugin abstraction must be loose enough that a non-SWE-shape task fits
   without another refactor round. The Benchmark protocol carries a
   `task_shape` discriminator and `build_runner` hook for exactly this.
3. **D3 — Decouple the collector from benchmark-specific kwargs.** The
   provable claim that validates Option C: `collect_traces`' signature
   loses every `harness_*` and dataset string parameter; it gains a single
   `benchmark: Benchmark` parameter. This is the litmus test for whether
   the abstraction landed. (This driver replaces REVISION 0's "BFCL cheap"
   headline because Phase 4's signature diff is concretely measurable;
   "BFCL cheap" is only provable in the future.)

### 1.3 Viable Options

We evaluated three architectural shapes against the drivers:

#### Option A — Pure plugin class hierarchy (Benchmark Protocol)

**File layout:**
```
src/agents/benchmarks/
    __init__.py          # registry: get_benchmark(slug) -> Benchmark
    base.py              # Benchmark Protocol / ABC
    swe_bench_verified.py
    swe_rebench.py
```

**Pros:**
- Full type safety; LSP surfaces contract violations at plugin authoring time.
- Easy to unit-test each plugin in isolation.
- BFCL v4 implements the same protocol with mostly-None optional fields.
- Registry dispatch is one line in the CLI.

**Cons:**
- Per-benchmark experiment knobs (paths, splits, max_steps) are hardcoded
  in Python class attributes. Changing them requires editing code or
  subclassing — not a YAML diff.
- Slightly more code than pure config.
- Protocol may leak SWE-shape assumptions (e.g., `repos_root`) that BFCL
  v4 won't use.

#### Option B — Pure config-driven benchmarks

**File layout:**
```
configs/benchmarks/
    swe-bench-verified.yaml
    swe-rebench.yaml
src/trace_collect/benchmark_loader.py    # dispatches on yaml['type']
```

**Pros:**
- Zero code per new benchmark *if* the shape matches an existing
  `type:` discriminator.
- Plain YAML diffs for experimenters.

**Cons:**
- Quirks that aren't pure data (SWE-rebench's explicit `docker_image`,
  custom `select_subset`) need code somewhere. We either re-introduce a
  `benchmark_type` switch in the loader (the exact coupling we want to
  remove) or paper over with `quirks:` flags (an `if quirks['foo']:` maze).
- BFCL v4 definitely needs new code.
- Hard to type-check; loader becomes a big dict-dispatch.

#### Option C — Hybrid: plugin class + YAML config binding (RECOMMENDED)

**File layout:**
```
src/agents/benchmarks/
    __init__.py          # REGISTRY: slug -> Benchmark class
    base.py              # Benchmark abstract class + BenchmarkConfig dataclass
    swe_bench_verified.py
    swe_rebench.py
configs/benchmarks/
    swe-bench-verified.yaml
    swe-rebench.yaml
```

YAML holds **experiment-tunable** settings (split, data_root, repos_root,
trace_root, default_max_steps, selection_n, selection_seed, docker_namespace,
exclude_lite). Python class holds **schema-and-quirks** logic (load,
normalize, derive_test_cmd, select_subset, build_harness_args, build_runner).

The registry dispatch:
```python
plugin_cls = get_benchmark_class("swe-rebench")
cfg = BenchmarkConfig.from_yaml("configs/benchmarks/swe-rebench.yaml")
plugin = plugin_cls(cfg)
tasks = plugin.load_tasks()
```

**Pros:**
- **Code for quirks, YAML for knobs** — the right boundary for research.
- Each plugin still has full Python power for schema divergence (Option A's
  win), but experimenters can change splits/paths without editing Python
  (Option B's win).
- BFCL v4 = one new Python file + one new YAML; collector and CLI don't
  change. Matches D2.
- **Validates D3 directly:** Phase 4's new `collect_traces` signature
  drops every benchmark-specific kwarg in favor of a single `benchmark`
  handle. The diff is the proof.

**Cons:**
- Two files per benchmark (Python + YAML), slightly more ceremony.
- Must define the YAML schema carefully upfront.

#### Selection

**We select Option C.** Option A doesn't give experimenters knobs without
editing code. Option B doesn't give us type safety for SWE-rebench's
schema quirks. Hybrid matches all three drivers.

**Why we are not stuck with only one viable option:** Option A and Option B
are both functional implementations of the same goal — they each have
documented pros that we are giving up by selecting C (Option A's smaller
file count, Option B's zero-code-per-benchmark for matching shapes). They
remain viable fallbacks if Phase 1 reveals the YAML schema is unworkable.

### 1.4 Risk Flagging — DELIBERATE MODE

This is a refactor touching the trace consumer contract, the collector
entry point, AND the openclaw write path. We treat it as **DELIBERATE**
(not SHORT) per the consensus protocol, which requires a pre-mortem and
expanded test plan below.

#### Pre-mortem: 5 ways this plan fails

**Scenario PM-1 — "Legacy v4 traces silently drop the `benchmark` field
and the Gantt viewer shows `unknown`."**
- *Cause:* We add `benchmark` to `log_metadata()` but forget to backfill it
  in `TraceData.load()`. Legacy traces parse with `metadata['benchmark'] =
  None`, and the Gantt UI displays "unknown".
- *Mitigation:* `TraceData.load()` (Phase 2) sets:
  ```python
  metadata.setdefault("trace_format_version", 4)  # legacy traces
  metadata.setdefault("benchmark", "swe-bench-verified")
  metadata.setdefault("benchmark_split", "test")
  ```
  Default is `swe-bench-verified` because that is the only benchmark that
  produced any pre-existing trace. The legacy regression fixture
  (`tests/fixtures/legacy_gantt_payload.json`, Phase 1) asserts byte-identical
  rendering pre/post refactor.
- *Detection trigger:* `tests/test_gantt_builder_parity.py` extended with
  the legacy fixture; CI fails if drift appears.

**Scenario PM-2 — "BFCL v4 doesn't fit the Benchmark protocol and we have
to refactor again in 3 weeks."**
- *Cause:* The Benchmark protocol assumed SWE-shaped tasks and the collector
  silently dispatches to `MiniSWECodeAgent` / `SWEBenchRunner` for every
  benchmark. BFCL v4 has no `repo`/`base_commit`/`test_cmd`; passing it
  through the SWE runners crashes mid-run.
- *Mitigation (strengthened in REVISION 1):* The Benchmark protocol marks
  `repo`, `base_commit`, `test_cmd`, `repos_root` as **optional**. The
  base class `Benchmark.build_runner()` does **NOT** silently default to
  the SWE runners. Instead:
  ```python
  def build_runner(self, *, scaffold: str, ...) -> Any:
      if self.task_shape == "swe_patch":
          # Subclass must explicitly opt in by overriding,
          # OR import the SWE runners explicitly here.
          raise NotImplementedError(
              f"{type(self).__name__} declares task_shape='swe_patch' "
              f"but does not override build_runner(). SWE-shape benchmarks "
              f"must explicitly return a MiniSWECodeAgent or SWEBenchRunner."
          )
      raise NotImplementedError(
          f"{type(self).__name__} has task_shape={self.task_shape!r} but "
          f"does not implement build_runner(). Non-SWE benchmarks must "
          f"provide their own runner."
      )
  ```
  Both `SWEBenchVerified` and `SWERebenchBenchmark` override `build_runner`
  explicitly. There is no implicit fallback. This is the exact failure mode
  PM-2 describes, fixed by making the protocol scream loudly instead of
  silently routing.
- *Detection trigger:* `tests/test_benchmark_protocol.py` instantiates a
  `MockFunctionCallBenchmark` (no override), calls `build_runner()`,
  asserts `NotImplementedError` is raised with the expected message.

**Scenario PM-3 — "SWE-rebench docker_image field breaks the openclaw
prepare_workspace() path and we silently fall back to a wrong image."**
- *Cause:* `prepare_workspace()` and the official harness derive a Docker
  image from `repo+commit` for SWE-Bench Verified. SWE-rebench ships its
  own `docker_image` URIs (`swerebench/sweb.eval.x86.*`). Without an
  explicit pin, the harness either picks the wrong image (silent wrong
  execution) or fails with "image not found" mid-run.
- *Mitigation:* SWE-rebench plugin's `normalize_task()` pins
  `task['image_name'] = raw['docker_image']`. `build_harness_args()` sets
  `namespace=None` (rebench images are fully qualified). `swebench_harness.py`
  asserts: if `namespace is None` and any prediction's task lacks
  `image_name`, raise before invoking the harness.
- *Detection trigger:* Phase 6 verification commands include
  `podman images | grep swerebench/sweb.eval` (Critic item 12) — implemented
  as an actual command in Phase 6's verification block, not just narrative.

**Scenario PM-4 — "`swebench_data.py` shim re-export breaks existing
imports because a symbol got renamed in the plugin port."**
- *Cause:* Phase 1 turns `swebench_data.py` into a thin re-export shim. If
  the plugin module renames `select_tool_intensive_tasks` to
  `select_subset` (its protocol method) but the shim forgets to alias it,
  every existing import (notebook, script, test) breaks.
- *Mitigation:* Phase 1 includes `tests/test_swebench_data_shim_import_parity.py`
  that imports the legacy module symbol-by-symbol and asserts every name
  the legacy module exported is still importable from
  `agents.swebench_data`. Concretely, the assert list is:
  ```python
  EXPECTED_LEGACY_SYMBOLS = {
      "load_swebench_verified",
      "select_tool_intensive_tasks",
      "derive_test_cmd",
      "_count_fail_to_pass",
      "download_and_save",
      "REPO_QUOTAS",
      "HEAVY_REPOS",
  }
  ```
  Any symbol that exists in the legacy module today must remain importable
  from the shim.
- *Detection trigger:* The new test fails immediately if a name is dropped.

**Scenario PM-5 — "Legacy openclaw traces (produced before this refactor)
lack `trace_format_version` at write time and the v5 read-time validator
rejects them as malformed."**
- *Cause:* P2 (the openclaw collector write-path leak) means every trace
  in `traces/swebench_verified/qwen-qwen3.6-plus-free/<old_ts>/` produced
  via the collector lacks `trace_format_version` entirely. After Phase 2's
  read-time backfill is added, those traces should default to
  `trace_format_version: 4`. If the read-time backfill is added but the
  v5 reader strict-validates `>= 5`, all legacy traces become unreadable.
- *Mitigation:* Phase 2's backfill explicitly handles the missing-version
  case BEFORE any version-strictness check:
  ```python
  metadata.setdefault("trace_format_version", 4)
  if metadata["trace_format_version"] not in (4, 5):
      raise ValueError(...)
  ```
  Phase 2 also adds a fixture `tests/fixtures/legacy_openclaw_no_version.jsonl`
  (a real one-line trace_metadata record with NO `trace_format_version`
  field, copied from an existing legacy file) and a test that loads it
  successfully and asserts it backfills to v4.
- *Detection trigger:* `tests/test_trace_inspector.py::test_legacy_openclaw_no_version_loads`.

#### Expanded Test Plan (DELIBERATE)

| Level | Scope | Test file | Assertion |
|---|---|---|---|
| Unit | Benchmark plugin registry | `tests/test_benchmark_registry.py` (new) | `get_benchmark_class("swe-rebench")` returns a class; unknown slug raises. |
| Unit | SWE-Bench Verified plugin parity | `tests/test_swe_bench_verified_plugin.py` (new) | Plugin `load_tasks()` output is identical (modulo ordering) to pre-refactor `load_swebench_verified()`. |
| Unit | `swebench_data` shim parity | `tests/test_swebench_data_shim_import_parity.py` (new — PM-4) | Every legacy symbol still importable from `agents.swebench_data`. |
| Unit | `derive_test_cmd` accepts list AND string | `tests/test_swebench_data.py` (extended) | `derive_test_cmd({"FAIL_TO_PASS": ["a", "b"]})` and `derive_test_cmd({"FAIL_TO_PASS": '["a","b"]'})` produce identical output. Same for `_count_fail_to_pass`. |
| Unit | SWE-rebench schema normalization | `tests/test_swe_rebench_plugin.py` (new) | Native-list `FAIL_TO_PASS` stays a native list; explicit `docker_image` is pinned to `image_name`; `exclude_lite=true` filters `meta.is_lite` rows; `exclude_lite=false` keeps them. |
| Unit | Trace metadata read-time backfill | `tests/test_trace_inspector.py` (extended) | Loading a pre-refactor v4 trace without `benchmark` defaults to `swe-bench-verified`. Loading a legacy openclaw trace with NO `trace_format_version` defaults to `4` (PM-5 fixture). |
| Unit | Gantt builder Python/JS parity | `tests/test_gantt_builder_parity.py` (extended) | Python and JS builders backfill `benchmark` identically for legacy traces. Now also reads `tests/fixtures/legacy_gantt_payload.json`. |
| Unit | Legacy Gantt payload regression snapshot | `tests/test_gantt_legacy_snapshot.py` (new — Critic item 13) | Post-refactor builder produces byte-identical payload to `tests/fixtures/legacy_gantt_payload.json` for the same input trace. |
| Integration | `log_metadata()` signature contract | `tests/test_trace_logger.py` (extended — Critic item OQ-5; not new) | `log_metadata(scaffold=..., benchmark=..., benchmark_split=...)` writes all fields and stamps `trace_format_version=5`. |
| Integration | Benchmark protocol contract | `tests/test_benchmark_protocol.py` (new) | Mock function-call benchmark satisfies the protocol without `repos_root`. Calling `build_runner()` on a no-override mock raises `NotImplementedError` with the explicit message (PM-2 mitigation). |
| Integration | Collector benchmark dispatch | `tests/test_collector_benchmark_dispatch.py` (new) | Passing `benchmark=swe-rebench-plugin-instance` routes through that plugin's `build_harness_args`; passing the verified plugin routes through its own. |
| Integration | Openclaw write path stamps version | `tests/test_collector_openclaw_metadata.py` (new) | After running `_normalize_openclaw_trace`, the destination JSONL's first record has `trace_format_version=5`, `benchmark`, `benchmark_split`, and `scaffold="openclaw"`. |
| E2E (smoke) | 2 mini-swe-agent tasks on SWE-rebench | Makefile `smoke-swe-rebench-miniswe` | Trace lands under `traces/swe-rebench/qwen-qwen3.6-plus-free/<ts>/`; trace_metadata has `benchmark: swe-rebench`, `benchmark_split: filtered`, `trace_format_version: 5`; Gantt HTML renders. |
| E2E (smoke) | 2 openclaw tasks on SWE-rebench | Makefile `smoke-swe-rebench-openclaw` | Same as above; `EvalTask.from_benchmark_instance()` exercised; `docker_image` pinned correctly; `podman images | grep swerebench/sweb.eval` returns ≥1 hit. |
| Regression | All existing tests | `make test` | Still passes at end of every phase. |
| Regression | Existing SWE-Bench Verified run | Makefile `smoke-swe-bench-verified` | E2E run on 1 Verified task behaves byte-identically to pre-refactor baseline. |
| Observability | Pre-commit review gate | `code-reviewer` agent (opus, fresh context) | Phase 7 — see review-gate spec there. |

---

## 2. Detailed Plan

**Phase order: 0 → 1 → 3 → 4 → 2 → 5 → 6 → 7.** Each phase is independently
mergeable, testable, and rollback-able. We do not merge to `main` until
Phase 7 review passes; the whole plan lives on `dev/swe-rebench-plugin`.

The order reasoning (Critic item 4): Phase 2 used to come right after
Phase 1, but Phase 2 writes a `benchmark` field whose value is sourced
from a `collect_traces(benchmark=...)` parameter that does not exist
until Phase 4. So Phase 2 must come AFTER Phase 4. Phase 3 (rebench
plugin) does not depend on Phase 2 or Phase 4, so it slots in early to
unblock parallel test authoring. Phase 4 contains the openclaw
write-path fix (the P2 leak) so that by the time Phase 2's read-time
backfill lands, both write and read sides are coherent.

### Phase 0 — Branch + data prep (no code changes)

**Files touched:** none (data download only)

**Actions:**
1. `git checkout -b dev/swe-rebench-plugin` from `main` (current: `main` at `8deb59d`).
2. `conda activate ML` (per CLAUDE.md).
3. Create `data/swe-rebench/` (empty) and `data/swe-rebench/repos/` (empty).
   Keep `data/swebench_verified/` untouched.
4. Download SWE-rebench filtered split once via a one-off Python invocation
   to verify HF access works. Do *not* check in any downloaded data.
5. Manually inspect 3 random rows to confirm:
   - `FAIL_TO_PASS` is `list`, not `str`
   - `docker_image` is a string URI
   - `install_config` is a dict
   - `meta.is_lite` is a bool
   Record findings in an internal note (not committed).

**Acceptance criteria:**
- `dev/swe-rebench-plugin` branch exists and matches `main`.
- `datasets.load_dataset("nebius/SWE-rebench", split="filtered")` succeeds
  and yields ≥ 6,500 rows.
- Sample row keys match expectations.

**Verification:**
```bash
git status && git rev-parse --abbrev-ref HEAD
python -c "from datasets import load_dataset; ds = load_dataset('nebius/SWE-rebench', split='filtered'); print(len(ds), list(ds[0].keys()))"
```

**Rollback:** `git checkout main && git branch -D dev/swe-rebench-plugin`.

---

### Phase 1 — Benchmark plugin base + SWE-Bench Verified port + list-shape fix + legacy fixture

**Files touched (new):**
- `src/agents/benchmarks/__init__.py`
- `src/agents/benchmarks/base.py`
- `src/agents/benchmarks/swe_bench_verified.py`
- `tests/test_benchmark_registry.py`
- `tests/test_swe_bench_verified_plugin.py`
- `tests/test_benchmark_protocol.py`
- `tests/test_swebench_data_shim_import_parity.py`  (PM-4)
- `tests/test_gantt_legacy_snapshot.py`              (Critic item 13)
- `tests/fixtures/legacy_gantt_payload.json`         (Critic item 13)

**Files touched (modified):**
- `src/agents/swebench_data.py`:
  - Becomes a thin shim re-exporting from `agents.benchmarks.swe_bench_verified`.
  - **Independently of the shim, fix `derive_test_cmd` (lines 33-49) and
    `_count_fail_to_pass` (lines 59-67) to accept `list | str` natively at
    the top of the function.** This is the Architect's must-fix #1 (flip
    O2 normalization direction). After this fix, every downstream consumer
    sees `FAIL_TO_PASS` as a Python list — no JSON-string round trip
    anywhere. The fix is ~6 LOC per function:
    ```python
    def derive_test_cmd(task: dict[str, Any]) -> str:
        raw = task.get("FAIL_TO_PASS", [])
        if isinstance(raw, str):
            try:
                test_ids = json.loads(raw)
            except json.JSONDecodeError:
                test_ids = [raw] if raw else []
        else:
            test_ids = list(raw)
        if not test_ids:
            return "python -m pytest --no-header -q"
        return f"python -m pytest {' '.join(test_ids)} -x --no-header -q"
    ```
    `_count_fail_to_pass` gets the same shape change.

**Design:**

- `base.py` defines:
  ```python
  @dataclass
  class BenchmarkConfig:
      slug: str
      display_name: str
      harness_dataset: str
      harness_split: str
      data_root: Path
      repos_root: Path | None
      trace_root: Path                # NEW (Architect must-fix #3): YAML-driven
      default_max_steps: int
      selection_n: int
      selection_seed: int
      docker_namespace: str | None
      exclude_lite: bool = False      # P5 (Critic item 5): explicit YAML knob
      extras: dict[str, Any] = field(default_factory=dict)

      @classmethod
      def from_yaml(cls, path: Path) -> "BenchmarkConfig": ...

  class Benchmark(ABC):
      slug: ClassVar[str]
      task_shape: ClassVar[Literal["swe_patch", "function_call"]] = "swe_patch"

      def __init__(self, config: BenchmarkConfig) -> None: ...

      @abstractmethod
      def load_tasks(self) -> list[dict[str, Any]]: ...

      @abstractmethod
      def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]: ...

      def derive_test_cmd(self, task: dict[str, Any]) -> str:
          # Default delegates to agents.swebench_data.derive_test_cmd
          # (which now accepts list OR str natively).
          ...

      def select_subset(self, tasks, n, seed) -> list[dict]:
          # Default: repo-stratified by FAIL_TO_PASS count, native-list aware.
          ...

      def build_harness_args(self, *, predictions_path, run_id, ...) -> dict: ...

      def image_name_for(self, task: dict) -> str | None:
          return task.get("image_name")

      def build_runner(self, *, scaffold: str, **_) -> Any:
          # PM-2 mitigation: NO silent dispatch. Subclasses MUST override.
          if self.task_shape == "swe_patch":
              raise NotImplementedError(
                  f"{type(self).__name__} declares task_shape='swe_patch' "
                  f"but does not override build_runner(). SWE-shape benchmarks "
                  f"must explicitly return a MiniSWECodeAgent or SWEBenchRunner."
              )
          raise NotImplementedError(
              f"{type(self).__name__} has task_shape={self.task_shape!r} "
              f"but does not implement build_runner()."
          )

      def build_run_dir(self, *, model: str, run_id: str | None = None) -> Path:
          # P4: reads trace_root from CONFIG, not class override.
          run_id = run_id or _new_run_id()
          return self.config.trace_root / model / run_id
  ```
- `__init__.py` exposes `REGISTRY: dict[str, type[Benchmark]]` and
  `get_benchmark_class(slug)`. Plugin modules register at import time.
- `swe_bench_verified.py` is a straight port of `swebench_data.py`.
  `REPO_QUOTAS` and `HEAVY_REPOS` become class attributes. **It explicitly
  overrides `build_runner()`** to return the existing `MiniSWECodeAgent` /
  `SWEBenchRunner` (no silent dispatch).

**Legacy regression fixture (Critic item 13):**

`tests/fixtures/legacy_gantt_payload.json` is captured **before** Phase 1's
shim refactor lands, by running:
```bash
PYTHONPATH=src python -c "
from trace_collect.gantt_data import build_gantt_payload
import json
payload = build_gantt_payload('traces/swebench_verified/qwen-qwen3.6-plus-free/<a known existing run>/django__django-11734.jsonl')
print(json.dumps(payload, indent=2, sort_keys=True))" > tests/fixtures/legacy_gantt_payload.json
```
The new `tests/test_gantt_legacy_snapshot.py` then loads the same input
trace, regenerates the payload, and asserts byte-equality with the
checked-in fixture. This catches any pre-refactor vs post-refactor drift
that the existing parity test (Python vs JS for the SAME input) cannot.

**Acceptance criteria:**
- All existing tests still pass.
- New tests pass:
  - `test_benchmark_registry.py::test_swe_bench_verified_registered`
  - `test_swe_bench_verified_plugin.py::test_load_tasks_matches_legacy`
  - `test_benchmark_protocol.py::test_mock_function_call_benchmark_satisfies_protocol`
  - `test_benchmark_protocol.py::test_default_build_runner_raises_not_implemented`
  - `test_swebench_data_shim_import_parity.py::test_all_legacy_symbols_importable`
  - `test_swebench_data.py::test_derive_test_cmd_accepts_list` (extended)
  - `test_swebench_data.py::test_derive_test_cmd_accepts_string` (extended)
  - `test_gantt_legacy_snapshot.py::test_legacy_payload_byte_identical`
- `agents.swebench_data` import surface unchanged (PM-4 test).

**Verification:**
```bash
conda activate ML
PYTHONPATH=src python -m pytest tests/ -x
PYTHONPATH=src python -c "from agents.swebench_data import load_swebench_verified, REPO_QUOTAS, derive_test_cmd, _count_fail_to_pass; print('shim ok')"
PYTHONPATH=src python -c "from agents.benchmarks import get_benchmark_class; print(get_benchmark_class('swe-bench-verified'))"
PYTHONPATH=src python -c "from agents.swebench_data import derive_test_cmd; print(derive_test_cmd({'FAIL_TO_PASS': ['tests/foo.py::test_a']}))"
```

**Rollback:** `git revert` the phase commit. The shim means no call sites
changed; reverting the new files is sufficient.

---

### Phase 3 — SWE-rebench plugin adapter

**Files touched (new):**
- `src/agents/benchmarks/swe_rebench.py`
- `tests/test_swe_rebench_plugin.py`

**Design:**

```python
class SWERebenchBenchmark(Benchmark):
    slug = "swe-rebench"
    task_shape = "swe_patch"

    def load_tasks(self):
        ds = load_dataset("nebius/SWE-rebench", split=self.config.harness_split)
        return [self.normalize_task(dict(row)) for row in ds]

    def normalize_task(self, raw):
        # NATIVE LIST end-to-end (Architect must-fix #1, O2 flipped).
        # FAIL_TO_PASS / PASS_TO_PASS / FAIL_TO_FAIL / PASS_TO_FAIL stay
        # as native Python lists. Phase 1 already fixed derive_test_cmd
        # and _count_fail_to_pass to accept lists, so no conversion needed.
        task = dict(raw)

        # Quirk: explicit docker_image -> image_name
        if raw.get("docker_image"):
            task["image_name"] = raw["docker_image"]

        # Derive test_cmd via base class (which delegates to the
        # list-aware derive_test_cmd in swebench_data).
        task["test_cmd"] = self.derive_test_cmd(task)
        return task

    def select_subset(self, tasks, n, seed):
        # Critic item 5: exclude_lite is a YAML knob, NOT a default.
        candidates = tasks
        if self.config.exclude_lite:
            candidates = [
                t for t in tasks
                if not (t.get("meta") or {}).get("is_lite", False)
            ]
        return super().select_subset(candidates, n, seed)

    def build_harness_args(self, **kwargs):
        args = super().build_harness_args(**kwargs)
        args["namespace"] = None    # rebench images are fully qualified
        return args

    def image_name_for(self, task):
        return task.get("image_name")

    def build_runner(self, *, scaffold, provider, workspace_base, **kwargs):
        # PM-2: explicit override; no implicit fallback.
        if scaffold == "openclaw":
            from agents.openclaw.eval.runner import SWEBenchRunner
            return SWEBenchRunner(
                provider=provider,
                workspace_base=workspace_base,
                benchmark=self,
                **kwargs,
            )
        from agents.miniswe import MiniSWECodeAgent
        return MiniSWECodeAgent(provider=provider, benchmark=self, **kwargs)
```

**Acceptance criteria:**
- `test_swe_rebench_plugin.py` covers:
  - `load_tasks()` returns >6,000 normalized tasks.
  - Every normalized task has `FAIL_TO_PASS` as a **native list** (not a JSON string).
  - Every task with `docker_image` has matching `image_name`.
  - `select_subset(n=32, seed=42)` with `exclude_lite=True` excludes all `meta.is_lite=True` tasks.
  - `select_subset(n=32, seed=42)` with `exclude_lite=False` (default) keeps them.
  - `build_harness_args(...)` returns `namespace=None`.
  - `derive_test_cmd` produces a valid pytest invocation for a normalized rebench task.
  - `build_runner(scaffold="openclaw", ...)` returns a `SWEBenchRunner` instance.

**Verification:**
```bash
conda activate ML
PYTHONPATH=src python -m pytest tests/test_swe_rebench_plugin.py -xvs
```

**Rollback:** Revert commit. No side effects outside the plugin file and its test.

---

### Phase 4 — Collector signature surgery + EvalTask + openclaw write-path fix

This phase is the load-bearing one. It is **atomic**: there is no in-between
half-state where the collector signature is partially changed. If anything
in this phase needs reverting, the whole commit is reverted as a single
unit. This is the Critic item 8 disposition (option B: explicitly atomic
+ "revert commit is the only rollback" note).

Optionally we may stage it as 4a/4b/4c for review readability, but the
commit lands as one. The sub-step labels below are review-aid only.

**Files touched (modified):**
- `src/agents/openclaw/eval/types.py`
- `src/agents/openclaw/eval/runner.py`
- `src/trace_collect/collector.py`  (signature + `_collect_openclaw` + `_normalize_openclaw_trace`)
- `scripts/run_nanobot_eval.py`

**Files touched (new):**
- `tests/test_collector_benchmark_dispatch.py`
- `tests/test_collector_openclaw_metadata.py`

**Sub-step 4a — `EvalTask` rename + alias:**

- `EvalTask.from_swebench_instance()` → `EvalTask.from_benchmark_instance(raw, workspace_base, benchmark: Benchmark | None = None)`.
- Keep `from_swebench_instance` as a deprecated alias that calls
  `from_benchmark_instance(raw, workspace_base, benchmark=None)` and emits
  `DeprecationWarning`.
- Inside `from_benchmark_instance`, if `benchmark` is provided, call
  `benchmark.normalize_task(raw)` first; otherwise use the existing
  raw-row schema handling.

**Sub-step 4b — `collect_traces` signature surgery + openclaw write-path fix:**

- The new signature, written down explicitly (Critic item 7):
  ```python
  async def collect_traces(
      *,
      api_base: str,
      api_key: str,
      model: str,
      benchmark: Benchmark,                     # NEW (replaces all harness_* and dataset string params)
      task_source: str | Path | None = None,    # if None, defaults to benchmark.config.data_root / "tasks.json"
      output_dir: str | Path | None = None,     # if None, defaults to benchmark.config.trace_root
      max_steps: int | None = None,             # if None, defaults to benchmark.config.default_max_steps
      command_timeout_s: float = 120.0,
      task_timeout_s: float = 1200.0,
      sample: int | None = None,
      instance_ids: list[str] | None = None,
      scaffold: str = "mini-swe-agent",
      run_id: str | None = None,
      max_context_tokens: int = 256_000,
      evaluate: bool = False,
      harness_max_workers: int = 1,
      harness_timeout: int = 1800,
      harness_run_id: str | None = None,
      harness_report_dir: str | Path | None = None,
  ) -> Path:
      ...
  ```
  **Removed parameters** (these all came from `benchmark.config` now):
  - `harness_dataset`        (was hardcoded `"princeton-nlp/SWE-bench_Verified"` at line 274)
  - `harness_split`          (was hardcoded `"test"` at line 275)
  - `harness_namespace`      (was hardcoded `"swebench"` at line 280)
  - `repos_root`             (was a positional arg; now read from `benchmark.config.repos_root`)

  The same signature surgery is applied to `_collect_openclaw` (currently
  at `collector.py:557-563` with the same hardcoded defaults).

  Inside `collect_traces`, defaults resolve as:
  ```python
  task_source = Path(task_source) if task_source else benchmark.config.data_root / "tasks.json"
  output_dir = Path(output_dir) if output_dir else benchmark.config.trace_root
  max_steps = max_steps if max_steps is not None else benchmark.config.default_max_steps
  ```

  `build_run_dir()` becomes:
  ```python
  def build_run_dir(benchmark: Benchmark, model: str, run_id: str) -> Path:
      return benchmark.build_run_dir(model=model, run_id=run_id)
  ```
  No more scraping `task_source` path heuristics.

- **Openclaw write-path fix (the P2 leak — folded into Phase 4 per Critic
  item 4):** rewrite `_normalize_openclaw_trace` (currently at
  `collector.py:747-794`) to route through `TraceLogger.log_metadata()`
  instead of hand-rolling a dict:
  ```python
  def _normalize_openclaw_trace(
      src: Path,
      dst: Path,
      *,
      model: str,
      api_base: str,
      max_steps: int,
      instance_id: str,
      benchmark: Benchmark,
  ) -> None:
      """Copy an OpenClaw trace to benchmark run-dir layout with metadata injection."""
      from harness.trace_logger import TraceLogger

      # Use TraceLogger so trace_format_version is stamped automatically.
      logger = TraceLogger(dst.parent, dst.stem)
      logger.log_metadata(
          scaffold="openclaw",
          model=model,
          api_base=api_base,
          max_steps=max_steps,
          instance_id=instance_id,
          mode="collect",
          benchmark=benchmark.config.slug,
          benchmark_split=benchmark.config.harness_split,
          scaffold_capabilities={
              "tools": ["bash", "file_read", "file_write", "file_edit",
                        "list_dir", "web_search", "web_fetch", "send_message"],
              "memory": True,
              "skills": True,
              "file_ops": "structured",
          },
      )

      # Copy body, dropping the original trace_metadata produced by
      # _session_runner.py:531 (which has a stale version stamp now).
      lines = src.read_text(encoding="utf-8").splitlines()
      with dst.open("a", encoding="utf-8") as f:
          for line in lines:
              if not line.strip():
                  continue
              rec = json.loads(line)
              if rec.get("type") == "trace_metadata":
                  continue
              if rec.get("type") == "step" and isinstance(rec.get("tool_args"), dict):
                  rec["tool_args"] = json.dumps(rec["tool_args"], ensure_ascii=False)
              f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  ```
  **`TraceLogger.log_metadata` is updated in this same phase to stamp
  `trace_format_version: 5` (the bump per O3) and to accept `benchmark`
  and `benchmark_split` as explicit kwargs in its signature for type
  documentation.** The mini-swe-agent path gets the same treatment: it
  already calls `log_metadata`, but the call site now passes `benchmark`
  and `benchmark_split` from `benchmark.config`.

  After this sub-step, **every new trace produced by the collector — both
  scaffolds — has `trace_format_version=5`, `benchmark`, and
  `benchmark_split` at write time**. The P2 leak is closed at the source.

- The `_session_runner.py:531-549` raw `metadata = {...}` block (the
  in-process openclaw trace) is also updated to `trace_format_version: 5`
  and gains `benchmark`, `benchmark_split`. (The collector currently drops
  this record and re-injects via `_normalize_openclaw_trace`, but we keep
  the in-process one consistent so single-process testing of openclaw
  produces v5 traces too.)

**Sub-step 4c — Plugin-routed runner dispatch:**

- `_collect_openclaw` (and the mini-swe-agent path) call
  `benchmark.build_runner(scaffold="openclaw", ...)` instead of importing
  `SWEBenchRunner` directly. The plugin's explicit override returns the
  right runner. PM-2 silent-fallback path is gone.
- `scripts/run_nanobot_eval.py:71` is updated from
  `EvalTask.from_swebench_instance(...)` to
  `EvalTask.from_benchmark_instance(data, ws_base, benchmark=...)`. The
  script gains a `--benchmark swe-bench-verified` flag (default) so
  invocations don't change.

**Acceptance criteria:**
- `EvalTask.from_swebench_instance()` still works (emits DeprecationWarning).
- `EvalTask.from_benchmark_instance()` is the new canonical entry.
- `collect_traces(..., benchmark=get_benchmark_class('swe-bench-verified')(cfg))`
  produces a run dir at `cfg.trace_root / model / run_id`. With the
  Verified YAML's `trace_root: traces/swebench_verified`, that resolves to
  `traces/swebench_verified/<model>/<run_id>/` — legacy path preserved.
- `collect_traces(..., benchmark=get_benchmark_class('swe-rebench')(cfg))`
  produces a run dir at `traces/swe-rebench/<model>/<run_id>/`.
- `collect_traces` signature has **no** `harness_dataset`, `harness_split`,
  `harness_namespace`, or `repos_root` parameters. (Litmus test for D3.)
- `_normalize_openclaw_trace` writes a trace_metadata record with
  `trace_format_version=5`, `benchmark`, and `benchmark_split`.
  `tests/test_collector_openclaw_metadata.py` asserts this on a synthetic
  source trace.
- `tests/test_collector_benchmark_dispatch.py` mocks both plugins and
  asserts dispatch routes through the plugin's `build_harness_args`.
- All existing tests pass.

**Verification:**
```bash
conda activate ML
PYTHONPATH=src python -m pytest tests/ -x
PYTHONPATH=src python -c "
import inspect
from trace_collect.collector import collect_traces
sig = inspect.signature(collect_traces)
forbidden = {'harness_dataset', 'harness_split', 'harness_namespace', 'repos_root'}
present = set(sig.parameters) & forbidden
assert not present, f'forbidden params still in signature: {present}'
print('signature ok')
"
PYTHONPATH=src python -c "
import warnings
from agents.openclaw.eval.types import EvalTask
from pathlib import Path
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    EvalTask.from_swebench_instance({'instance_id': 'x', 'problem_statement': 'y', 'repo': 'a/b', 'base_commit': 'c'}, Path('/tmp'))
    assert any(issubclass(x.category, DeprecationWarning) for x in w), 'no deprecation warning emitted'
print('deprecation ok')
"
```

**Rollback:** `git revert` the single phase commit. Atomic. **If
implementation fails mid-phase (e.g., tests red after the collector
signature change but before the openclaw write-path fix lands):**
`git reset --hard <start-of-phase-4-commit>` — do not leave half-states
committed, do not try to salvage partial sub-steps. Phase 4's coupling
(collector signature ↔ `EvalTask.from_benchmark_instance` ↔
`_normalize_openclaw_trace` ↔ `build_runner` dispatch) makes any
intermediate commit un-importable.

---

### Phase 2 — Read-time backfill + display label + JS parity (write-side already done in Phase 4)

By the time this phase runs, Phase 4 has already updated
`TraceLogger.log_metadata` to stamp `trace_format_version=5` and accept
`benchmark`/`benchmark_split` as explicit kwargs, AND updated
`_normalize_openclaw_trace` to use `TraceLogger`. So **the write side is
already coherent**. Phase 2's job is the **read side**: backfill legacy
traces, render the display label, and extend the parity test.

**Files touched (modified):**
- `src/trace_collect/trace_inspector.py` — `TraceData.load()` adds:
  ```python
  metadata.setdefault("trace_format_version", 4)
  if metadata["trace_format_version"] not in (4, 5):
      raise ValueError(
          f"Unsupported trace_format_version "
          f"{metadata['trace_format_version']!r} in {path}"
      )
  metadata.setdefault("benchmark", "swe-bench-verified")
  metadata.setdefault("benchmark_split", "test")
  ```
  Defaults chosen because every pre-v5 trace came from SWE-Bench Verified
  test split. Documented in the function docstring with the date and PR
  reference.
- `src/trace_collect/gantt_builder.js` — mirror the same backfill (with
  the same default values) so Python/JS parity holds. **Note:** this
  file has NO `trace_format_version` constant today (Architect-verified
  via grep); the only change here is adding read-time backfill logic
  mirroring `trace_inspector.py`. No version constant to bump.
- `src/trace_collect/gantt_data.py` — read `metadata["benchmark"]` and
  use the YAML `display_name` from the matching `BenchmarkConfig` if
  available; fall back to the slug. The header rendering uses the
  resolved display name.
- `tests/test_trace_inspector.py` — extend with backfill assertions; both
  existing `trace_format_version: 4` fixture sites (`:79`, `:322`) are
  audited — one stays at v4 to exercise the backfill path, the other is
  bumped to v5 to exercise the current-version path.
- `tests/test_gantt_smoke.py` — bump inline v4 fixture to v5.
- `tests/test_gantt_template_bugs.py` — bump inline v4 fixture to v5.
- `tests/test_gantt_data.py` — bump inline v4 fixtures to v5.
- `tests/test_openclaw_async_and_status.py` — **keep at v4** as an explicit
  legacy-backfill fixture; add a comment noting the fixture intentionally
  exercises the read-time backfill.
- `tests/test_simulator.py` — **keep at v4** as an explicit legacy-backfill
  fixture; same comment.

**Files touched (new):**
- `tests/fixtures/legacy_openclaw_no_version.jsonl`  (PM-5)
- (Tests for read-time backfill go into the existing
  `tests/test_trace_inspector.py` and `tests/test_gantt_builder_parity.py`,
  both extended.)

**Trace format contract addendum (see §4):** `trace_format_version` is
**bumped from 4 to 5** in this plan. The Architect's O3 disposition was
"bump", chosen by the orchestrator. v5 means: `trace_metadata` records
contain `benchmark` (str), `benchmark_split` (str), and
`trace_format_version: 5` at write time. v4 records (legacy traces on
disk) are read-time-backfilled. The reader accepts both. v3 and earlier
are rejected — which is no regression since we never produced v3.

**Acceptance criteria:**
- Loading a legacy trace with no `benchmark` field yields
  `metadata["benchmark"] == "swe-bench-verified"` and
  `metadata["benchmark_split"] == "test"`.
- Loading a legacy openclaw trace with **no** `trace_format_version`
  field (PM-5 fixture) loads successfully and backfills to `4`.
- New traces produced after this phase report `trace_format_version: 5`.
- `tests/test_gantt_builder_parity.py` still passes against the legacy
  fixture and the new v5 fixture.
- `tests/test_gantt_legacy_snapshot.py` (from Phase 1) still passes,
  asserting byte-identical Gantt payload pre/post refactor.
- Gantt viewer opened against an existing
  `traces/swebench_verified/**/*.jsonl` file renders with the literal
  display name **`SWE-Bench Verified`** in the header (Critic item 11:
  no "or the slug" ambiguity — assert the exact string from the YAML
  `display_name` field).
- All tests pass.

**Verification:**
```bash
conda activate ML
PYTHONPATH=src python -m pytest tests/ -x
PYTHONPATH=src python -m trace_collect.cli gantt \
  traces/swebench_verified/qwen-qwen3.6-plus-free/<existing_run>/django__django-11734.jsonl \
  --output /tmp/legacy_check.html
grep -c "SWE-Bench Verified" /tmp/legacy_check.html  # must be exactly 1 or more, no "swe-bench-verified" slug
test "$(grep -c '>swe-bench-verified<' /tmp/legacy_check.html)" -eq 0  # slug must NOT appear in display
```

**Rollback:** Revert this phase commit. Read-time backfill removal does
not affect any on-disk traces; legacy traces remain on disk untouched.

---

### Phase 5 — Config / CLI / scripts / Makefile generalization

**Files touched (new):**
- `configs/benchmarks/swe-bench-verified.yaml`:
  ```yaml
  slug: swe-bench-verified
  display_name: "SWE-Bench Verified"
  harness_dataset: princeton-nlp/SWE-bench_Verified
  harness_split: test
  data_root: data/swebench_verified        # legacy path, intentionally preserved
  repos_root: data/swebench_repos          # legacy path, intentionally preserved
  trace_root: traces/swebench_verified     # legacy path, intentionally preserved
  default_max_steps: 50
  selection_n: 32
  selection_seed: 42
  docker_namespace: swebench
  exclude_lite: false
  # NOTE: data_root / repos_root / trace_root use the legacy underscore form
  # to avoid migrating real on-disk experimental data. New benchmarks should
  # use the slug-with-hyphens form (see swe-rebench.yaml).
  ```
- `configs/benchmarks/swe-rebench.yaml`:
  ```yaml
  slug: swe-rebench
  display_name: "SWE-rebench (filtered)"
  harness_dataset: nebius/SWE-rebench
  harness_split: filtered
  data_root: data/swe-rebench
  repos_root: data/swe-rebench/repos
  trace_root: traces/swe-rebench
  default_max_steps: 50
  selection_n: 32
  selection_seed: 42
  docker_namespace: null    # rebench images are fully qualified (swerebench/...)
  exclude_lite: false       # P5: keep lite tasks by default
  ```
- `scripts/setup/swe_rebench_data.sh` — mirrors `swebench_data.sh` but
  downloads SWE-rebench filtered split to `data/swe-rebench/tasks.json`.

**Files touched (modified):**
- `src/trace_collect/cli.py`:
  - Add `--benchmark` flag (choices populated from registry).
  - Loads YAML from `configs/benchmarks/{slug}.yaml`, builds the plugin,
    constructs the `BenchmarkConfig`, passes it to `collect_traces` via
    `benchmark=`. Removes the per-flag `--harness-dataset`,
    `--harness-split`, `--harness-namespace` defaults (currently hardcoded
    at `cli.py:152` and `:184`). Individual CLI flags can still override
    YAML values.
  - Default `--benchmark` is `swe-bench-verified` so existing invocations
    keep working with zero changes.
- `configs/trace_collect/swebench.yaml` — add `benchmark: swe-bench-verified`.
- `configs/workloads/code_agent.yaml` — add `benchmark: swe-bench-verified`
  (verified to exist in §0).
- `scripts/setup/clone_repos.sh` — parameterize `TASKS_FILE` and
  `REPOS_ROOT` via `--tasks` and `--repos-root` flags.
- `Makefile`:
  - Keep `download-swebench-verified` as alias; add `download-swe-bench-verified` and `download-swe-rebench`.
  - Add `setup-swe-rebench-data`, `setup-swe-rebench-repos`,
    `smoke-swe-rebench-miniswe`, `smoke-swe-rebench-openclaw` targets.
  - `setup-swebench-repos` accepts `TASKS=...` override.

**Acceptance criteria:**
- `python -m trace_collect.cli --benchmark swe-bench-verified --sample 0` runs unchanged.
- `python -m trace_collect.cli --benchmark swe-rebench --sample 0` is a
  dry-run that loads the plugin, prints resolved config, exits without
  API calls.
- `make setup-swe-rebench-data` downloads filtered split to `data/swe-rebench/tasks.json`.
- All tests pass.

**Verification:**
```bash
conda activate ML
make setup-swe-rebench-data
PYTHONPATH=src python -m trace_collect.cli --benchmark swe-rebench --sample 0 --verbose
PYTHONPATH=src python -m trace_collect.cli --benchmark swe-bench-verified --sample 0 --verbose
make test
```

**Rollback:** Revert commit. Legacy `make` aliases keep working.

---

### Phase 6 — Smoke run + Gantt verification + Docker assertion

**Files touched:** none (runtime only)

**Actions:**
1. `make download-swe-rebench`.
2. `make setup-swe-rebench-repos` — clone only the 2 smoke task repos.
3. `make smoke-swe-rebench-miniswe` — 2 tasks via mini-swe-agent.
4. `make smoke-swe-rebench-openclaw` — same 2 tasks via openclaw.
5. Open the 4 traces in Gantt viewer.
6. Open one existing SWE-Bench Verified trace and confirm no regression.

**Acceptance criteria:**
- 4 new trace JSONL files exist under `traces/swe-rebench/.../`, NOT
  `traces/swebench_verified/`.
- Each new trace has `trace_format_version: 5`, `benchmark: swe-rebench`,
  `benchmark_split: filtered`.
- Gantt HTML for SWE-rebench renders with header label "SWE-rebench
  (filtered)" (the YAML `display_name`, exact match) and no JS console errors.
- Gantt HTML for legacy Verified traces still renders identically — the
  Phase 1 snapshot fixture test catches drift.
- `trace_inspector overview` on a new SWE-rebench trace shows non-empty
  steps and the new benchmark field.
- **Docker image assertion (PM-3 mitigation, Critic item 12):** for each
  smoke task, `podman images | grep swerebench/sweb.eval` returns ≥1 hit.

**Verification:**
```bash
conda activate ML
# Per-trace metadata validation
for f in traces/swe-rebench/*/*/*.jsonl; do
  head -1 "$f" | python -c "
import sys, json
m = json.loads(sys.stdin.read())
assert m['benchmark'] == 'swe-rebench', m
assert m['benchmark_split'] == 'filtered', m
assert m['trace_format_version'] == 5, m
print('ok', m['instance_id'])
"
done
# PM-3 / Critic item 12: actual docker image presence assertion
podman images | grep swerebench/sweb.eval || { echo 'FAIL: no swerebench docker images present after smoke run'; exit 1; }
# Display label assertion (Critic item 11 carryover for new traces)
PYTHONPATH=src python -m trace_collect.cli gantt traces/swe-rebench/*/*/*.jsonl --output /tmp/rebench_gantt.html
grep -c '>SWE-rebench (filtered)<' /tmp/rebench_gantt.html  # >= 1
```

**Rollback:** Delete `traces/swe-rebench/`. Legacy traces untouched.

---

### Phase 7 — Docs + pre-commit review gate (MANDATORY per CLAUDE.md)

**Files touched (modified):**
- `docs/EXPERIMENT_PLAN.md` — "Benchmarks" section listing both first-class
  benchmarks with pointers to `configs/benchmarks/`.
- `README.md` — "How to add a new benchmark" quick-reference.
- `CLAUDE.md` — directive: "New benchmarks MUST be added as
  `src/agents/benchmarks/<slug>.py` plugins with a matching YAML in
  `configs/benchmarks/`. Never hardcode dataset names in collector/CLI."
- `docs/benchmark_plugin_spec.md` — Benchmark protocol + schema-quirk reference.

**Actions (review gate spec, Critic item 9):**

1. **Spawn a fresh `code-reviewer` agent (model = `opus`) in a separate
   context** — explicitly NOT the same context window as any agent that
   touched this branch. This is the CLAUDE.md mandatory review gate
   ("Mandatory Review Gate for Vibe Coding"): the reviewer must have a
   fresh context to avoid the author-blind-spot problem documented in
   `CLAUDE.md`.
2. The review prompt MUST include verbatim:
   - "Strict research taste; no benchmark-specific tuning is acceptable per CLAUDE.md."
   - "Check for: hardcoded dataset names (run the grep below), hindsight
     leakage, unjustified magic numbers, trace format regressions, BFCL
     v4 extensibility, openclaw write-path coherence."
   - "Flag any 🔴/🟠 issues. The plan is not green until those are fixed.
     If only 🟡 minor issues remain, you may proceed but still log them."
3. **Iterate** until reviewer returns zero 🔴/🟠. Each iteration is its
   own commit; the previous reviewer context is discarded between rounds
   so the reviewer can never approve its own previous suggestions.
4. **Log** the audit trail to `docs/reviews/swe-rebench-plugin-review.md`
   with reviewer ID, model, prompt, findings, and resolutions.

**Acceptance criteria:**
- Reviewer returns clean (no 🔴/🟠).
- All docs updated.
- `dev/swe-rebench-plugin` is in a mergeable state.
- **User explicitly approves the merge** — we never self-approve per CLAUDE.md.

**Verification (expanded grep coverage, Critic item 14):**
```bash
conda activate ML
make test
make lint

# Critic item 14: hardcoded dataset / namespace strings.
# Whitelist the plugin file and its YAML; everything else is a fail.
WHITELIST='swe_bench_verified.py\|benchmarks/swe-bench-verified.yaml'

# Hardcoded dataset name #1
grep -rn "princeton-nlp/SWE-bench_Verified" src/ configs/ --include='*.py' --include='*.yaml' \
  | grep -v "$WHITELIST" \
  && { echo 'FAIL: princeton-nlp/SWE-bench_Verified leaked outside plugin'; exit 1; } || echo 'ok #1'

# Hardcoded dataset name #2 (sub-string)
grep -rn "SWE-bench_Verified" src/ configs/ --include='*.py' --include='*.yaml' \
  | grep -v "princeton-nlp/SWE-bench_Verified" \
  | grep -v "$WHITELIST" \
  && { echo 'FAIL: SWE-bench_Verified leaked outside plugin'; exit 1; } || echo 'ok #2'

# Harness namespace string default leak
grep -rn '"swebench"' src/ configs/ --include='*.py' --include='*.yaml' \
  | grep -v "$WHITELIST" \
  | grep -v "swebench_harness.py:.*importlib.util.find_spec" \
  && { echo 'FAIL: "swebench" namespace default leaked outside plugin'; exit 1; } || echo 'ok #3'
```

**Rollback:** Don't merge the branch. `dev/swe-rebench-plugin` remains as
an artifact for later revisit.

---

## 3. Directory Layout Proposal

### 3.1 Final target layout

```
src/agents/benchmarks/
    __init__.py              # REGISTRY and get_benchmark_class()
    base.py                  # Benchmark ABC, BenchmarkConfig dataclass
    swe_bench_verified.py    # slug: swe-bench-verified
    swe_rebench.py           # slug: swe-rebench
    # Future:
    # bfcl_v4.py             # slug: bfcl-v4   (task_shape="function_call")

src/agents/swebench_data.py  # THIN SHIM — re-exports from swe_bench_verified plugin

configs/benchmarks/
    swe-bench-verified.yaml
    swe-rebench.yaml

data/
    swebench_verified/       # LEGACY — preserved, not renamed
        tasks.json
    swebench_repos/          # LEGACY — preserved, not renamed
    swe-rebench/             # NEW
        tasks.json
        repos/

traces/
    swebench_verified/       # LEGACY — continues to receive new Verified runs (per YAML trace_root)
        <model>/<ts>/...
    swe-rebench/             # NEW — slug-scoped per YAML trace_root
        <model>/<ts>/...
```

### 3.2 Naming decision (Architect must-fix #3, Critic item 3)

Public identifiers (CLI flag value, YAML filename, registry key, trace
metadata `benchmark` field, new `traces/` directory) use **slug-with-hyphens**:
- `swe-bench-verified`
- `swe-rebench`

Python module filenames use **slug-with-underscores** (PEP 8):
`swe_bench_verified.py`, `swe_rebench.py`.

**Trace output directories are YAML-driven, with no plugin code override.**
The `BenchmarkConfig.trace_root` field is the single source of truth.
`Benchmark.build_run_dir()` reads `self.config.trace_root` directly. There
is **no** `build_run_dir()` override hook in the plugin classes; if a
plugin needs a different output root, it sets `trace_root` in its YAML.

For SWE-Bench Verified, `configs/benchmarks/swe-bench-verified.yaml` sets:
```yaml
trace_root: traces/swebench_verified    # legacy path, intentionally preserved
data_root: data/swebench_verified
repos_root: data/swebench_repos
```
For SWE-rebench:
```yaml
trace_root: traces/swe-rebench
data_root: data/swe-rebench
repos_root: data/swe-rebench/repos
```
We do **not rename** existing `data/swebench_verified/` or
`traces/swebench_verified/`. They predate the slug convention; renaming
them would migrate real on-disk experimental data. The slug/dir mismatch
is documented in a YAML comment block (see Phase 5 above).

This resolves **O1**: option (b) from REVISION 0 (YAML field), per the
Architect's ruling. The "or" ambiguity from REVISION 0 §3.2 is deleted.

### 3.3 Migration path for existing traces

**No on-disk migration.** Existing `traces/swebench_verified/` files are
preserved as-is. They lack `benchmark` and `benchmark_split` fields.
Phase 2's read-time backfill defaults them to `swe-bench-verified` /
`test`, which is correct (the Verified `test` split is the only thing
that ever produced a pre-refactor trace). Some openclaw legacy traces
also lack `trace_format_version`; the same backfill handles that case
(PM-5).

---

## 4. Trace Format Contract Addendum

### 4.1 Version bump v4 → v5

Per the orchestrator's O3 disposition: **we bump `trace_format_version`
from 4 to 5**.

Reasons:
1. The current openclaw write path (`collector.py:759-782`) is producing
   trace_metadata records that have **no** `trace_format_version` stamp
   at all. We're "lying about v4" — every openclaw trace produced via the
   collector since the last bump has been silently version-less. Fixing
   this requires touching the write path anyway, so we get a clean reset
   for free.
2. We are adding new fields (`benchmark`, `benchmark_split`) that we want
   to **assert** at read time for new traces. With a version bump, the
   reader can require those fields for v5 records and apply backfill only
   for v4 records.
3. Future readers can fast-path v5 (no backfill) and slow-path v4 (with
   backfill) without ambiguity.

The cost is small: add a read-time `setdefault` backfill in
`gantt_builder.js` (note: **no version constant exists in this file** —
Architect verified via grep; the change is purely adding the backfill
logic to mirror `trace_inspector.py`), update `trace_inspector`'s
version banner, and teach `tests/test_gantt_builder_parity.py` to
accept both v4 (legacy fixture) and v5 (new fixture). None of these
are large.

### 4.2 New fields

| Field | Type | Required at write time | Default for legacy v4 | Source of truth |
|---|---|---|---|---|
| `trace_format_version` | `int` | yes (5) | 4 (backfilled) | `TraceLogger.log_metadata` |
| `benchmark` | `str` | yes | `"swe-bench-verified"` | Plugin `slug` at collection time |
| `benchmark_split` | `str` | yes | `"test"` | `BenchmarkConfig.harness_split` |

### 4.3 JSON delta at v5

**Before (legacy v4 — note that openclaw v4 traces frequently lack the
version stamp due to the P2 leak):**
```json
{
  "type": "trace_metadata",
  "scaffold": "openclaw",
  "model": "qwen/qwen3.6-plus:free",
  "api_base": "https://openrouter.ai/api/v1",
  "max_steps": 80,
  "instance_id": "django__django-11734",
  "mode": "collect",
  "scaffold_capabilities": { "...": "..." }
}
```

**After (v5):**
```json
{
  "type": "trace_metadata",
  "scaffold": "openclaw",
  "trace_format_version": 5,
  "benchmark": "swe-rebench",
  "benchmark_split": "filtered",
  "model": "qwen/qwen3.6-plus:free",
  "api_base": "https://openrouter.ai/api/v1",
  "max_steps": 80,
  "instance_id": "nebius__foo-42",
  "mode": "collect",
  "scaffold_capabilities": { "...": "..." }
}
```

### 4.4 Why we bump (rather than stay additive at v4)

REVISION 0 argued for staying at v4 with read-time backfill. The
Architect/Critic feedback (combined with the openclaw write-path leak
discovery) flips this:

- The "no version bump" rationale only worked if the existing v4 traces
  were actually well-formed v4. They aren't — openclaw traces produced
  by the collector lack the version stamp entirely. Fixing this requires
  a write-path change anyway.
- Bumping gives the v5 reader the moral right to require new fields at
  write time, which is the only way to enforce the contract going forward.
- Legacy v4 traces (both well-formed and the version-less openclaw ones)
  are read-time-backfilled, so no on-disk migration is needed.
- Parity test cost: `gantt_builder.js` has no version constant (verified);
  the change is a backfill `if (!meta.trace_format_version) meta.trace_format_version = 4;`
  mirroring `trace_inspector.py`. Python side is a similar `setdefault`.
  `test_gantt_builder_parity.py` adds fixtures for both v4 and v5. Total:
  ~5-10 lines across 3 files, no large rewrite.

### 4.5 Parity test delta

`tests/test_gantt_builder_parity.py` is updated:
- Accepts both `trace_format_version: 4` and `trace_format_version: 5`
  as valid input.
- Adds a new fixture pair: one v4 trace (no `benchmark`) and one v5
  trace (with `benchmark`).
- Asserts Python and JS produce identical Gantt payloads for both.

`tests/test_gantt_legacy_snapshot.py` (Phase 1) additionally asserts
byte-equality with `tests/fixtures/legacy_gantt_payload.json` for the
pre-refactor input.

---

## 5. Out of Scope (Explicit)

The following are **intentionally** not part of this plan:

1. **BFCL v4 integration.** We make it cheap to add; we don't add it.
   The Benchmark protocol's `task_shape` discriminator and `build_runner`
   hook are the seam.
2. **Migrating existing `traces/swebench_verified/` to a new path.**
   Legacy traces stay put; new Verified runs continue to land there per
   YAML `trace_root`.
3. **Renaming `EvalTask.from_swebench_instance()` without backward-compat
   alias.** The old method name remains with a `DeprecationWarning`. Final
   removal is a separate future PR.
4. **Renaming `SWEBenchRunner` → `BenchmarkRunner`.** Same rationale as #3.
5. **Replacing `swebench_data.py`.** It becomes a shim that re-exports
   from the plugin; deletion is a later PR.
6. **Refactoring `mini-swe-agent` or `openclaw` core loops.** Those
   scaffolds continue to consume the Verified task dict shape; the plugin
   normalizes SWE-rebench to match (now natively as a list, no JSON
   round trip).
7. **Upgrading the official SWE-bench harness package.**
8. **Running a full SWE-rebench sweep (21k tasks).** Phase 6 smoke runs
   2 tasks per scaffold; full evaluation is a separate experimental PR.
9. **Rewriting legacy on-disk traces to v5.** They stay v4 forever (or
   un-versioned, in the openclaw legacy case), and the v5 reader handles
   them via backfill.

---

## 6. Open Questions — Closed

REVISION 1 closes all three open questions per the orchestrator's
disposition. None remain deferred to the next reviewer.

### O1 — Trace output directory ownership: CLOSED → YAML
**Decision:** Architect option (b). `trace_root` lives in
`configs/benchmarks/<slug>.yaml`. There is no `build_run_dir()` plugin
override. The slug/dir mismatch for legacy Verified is preserved by
setting `trace_root: traces/swebench_verified` in the Verified YAML, and
the rationale is documented in the YAML's comment block.
**Principle update:** P4 now reads "output paths are YAML-configured per
benchmark, no defaults hardcoded in the collector."

### O2 — `FAIL_TO_PASS` normalization direction: CLOSED → native list
**Decision:** Architect must-fix #1. **Native list end to end.** Phase 1
fixes `swebench_data.derive_test_cmd` and `_count_fail_to_pass` to accept
`list | str` natively. SWE-rebench's `normalize_task` does NOT convert
its native list to a JSON string. The downstream `EvalTask` was already
list-shaped, so this just deletes a round trip nobody needed.

### O3 — Bump `trace_format_version` to 5: CLOSED → YES, bump
**Decision:** Orchestrator. **Bump to v5.** The openclaw write-path leak
makes the "stay at v4" argument moot — the current openclaw v4 traces
are version-less anyway. Bumping gives a clean reset and lets the reader
assert new fields for v5 records. Cost is small (1-line constant updates
in JS, Python, and parity test). See §4 for the full delta.

---

## 7. Final Checklist Before Architect/Critic Re-review

- [x] REVISION 1 marker present
- [x] Branch name specified (`dev/swe-rebench-plugin`)
- [x] §0 verified-by-planner grep results recorded
- [x] Trace format contract addendum updated for v4→v5 bump (§4)
- [x] Output paths are YAML-configured per benchmark (§3.2, P4)
- [x] Hardcoded references enumerated (§0) and Phase 7 grep covers all 8
- [x] RALPLAN-DR: Principles (5 — P1..P5), Drivers (3, D3 rephrased to provable claim), Options (3 with selection)
- [x] DELIBERATE mode: pre-mortem (5 scenarios — added PM-4, PM-5), expanded test plan
- [x] ADR equivalent: Decision (Option C), Drivers, Alternatives considered, Why chosen, Consequences (in §5), Follow-ups (in §5)
- [x] Phase order: 0 → 1 → 3 → 4 → 2 → 5 → 6 → 7 (justified)
- [x] Phase 4 has the new `collect_traces` signature written down explicitly
- [x] Phase 4 contains the openclaw write-path fix (the P2 leak), routed through `TraceLogger.log_metadata`
- [x] Phase 4 marked atomic ("revert is the only rollback")
- [x] `Benchmark.build_runner()` default raises `NotImplementedError` (PM-2 strengthened)
- [x] `exclude_lite` is a YAML knob defaulting to `false` (P5, Critic item 5)
- [x] Phase 1 has the legacy Gantt snapshot fixture + test (Critic item 13)
- [x] Phase 1 has the swebench_data shim parity test (PM-4)
- [x] Phase 1 has the `derive_test_cmd` / `_count_fail_to_pass` list-shape fix
- [x] Phase 2 has PM-5 fixture (legacy openclaw trace with no version stamp)
- [x] Phase 2 display label asserts the exact YAML `display_name` string (Critic item 11)
- [x] Phase 6 has the actual `podman images | grep` assertion (Critic item 12)
- [x] Phase 7 reviewer agent specified: fresh `code-reviewer` (opus, separate context, citing CLAUDE.md mandatory review gate) (Critic item 9)
- [x] Phase 7 grep covers `princeton-nlp/SWE-bench_Verified`, `SWE-bench_Verified`, and `"swebench"` (Critic item 14)
- [x] CLAUDE.md rules respected (no mocks, real workloads, conda ML env, source-of-truth in this file, review gate before experiments)
- [x] Out of scope is explicit
- [x] Open questions O1/O2/O3 all CLOSED in §6
- [x] `tests/test_trace_logger.py` labeled as `(extended)`, not `(new)` (Critic item OQ-5)
- [x] Phase 5's `code_agent.yaml` reference verified to exist (§0, Critic item OQ-4)
- [x] Phase 4's `from_swebench_instance` callers verified by grep (§0, Critic item OQ-2)
