PYTHON ?= python3
UV ?= uv

.PHONY: help pull sync build verify-bootstrap verify-env1 verify-env2 verify-env3a verify-env3b verify-env3c verify-env4 verify-env5 test lint serve-vllm run-smoke smoke-code smoke-data smoke-research run-sweep collect-results setup-swebench-repos build-swebench-images download-swebench-verified download-swe-rebench setup-swe-rebench-repos setup-swe-rebench smoke-swe-rebench-miniswe smoke-swe-rebench-openclaw download-bfcl-v4 setup-bfcl-v4 smoke-bfcl-v4-openclaw gantt-viewer-install gantt-viewer-dev gantt-viewer-build gantt-viewer-test gantt-viewer-clean

help:
	@printf "Targets:\n"
	@printf "  pull              Fast-forward pull the current branch\n"
	@printf "  sync              Install dependencies with uv\n"
	@printf "  verify-bootstrap  Run BOOTSTRAP-0 verification\n"
	@printf "  verify-env1       Run ENV-1 static verification\n"
	@printf "  verify-env2       Run ENV-2 static verification\n"
	@printf "  verify-env3a      Run ENV-3a static verification\n"
	@printf "  verify-env3b      Run ENV-3b static verification\n"
	@printf "  verify-env3c      Run ENV-3c static verification\n"
	@printf "  verify-env4       Run ENV-4 static verification\n"
	@printf "  verify-env5       Run ENV-5 static verification\n"
	@printf "  test              Run the full test suite\n"
	@printf "  lint              Run ruff\n"
	@printf "  serve-vllm        Run the raw vLLM launcher\n"
	@printf "  run-smoke         Run the current infrastructure smoke suite\n"
	@printf "  smoke-code        Reserved for AGENT-2 smoke test\n"
	@printf "  smoke-data        Reserved for AGENT-3 smoke test\n"
	@printf "  smoke-research    Reserved for AGENT-4 smoke test\n"
	@printf "  run-sweep         Run the harness sweep when HARNESS-1 is available\n"
	@printf "  collect-results   Pull result artifacts back via rsync\n"
	@printf "  download-swebench-verified  Download & select 32 tasks from SWE-bench Verified\n"
	@printf "  setup-swebench-repos        Clone repos referenced by selected tasks\n"
	@printf "  build-swebench-images       Build Podman container images\n"
	@printf "  download-swe-rebench        Download SWE-rebench (nebius/SWE-rebench) filtered split\n"
	@printf "  setup-swe-rebench-repos     Clone repos referenced by SWE-rebench tasks\n"
	@printf "  setup-swe-rebench           Shortcut: download-swe-rebench + setup-swe-rebench-repos\n"
	@printf "  smoke-swe-rebench-miniswe   Run $(SMOKE_N) SWE-rebench tasks through mini-swe-agent\n"
	@printf "  smoke-swe-rebench-openclaw  Run $(SMOKE_N) SWE-rebench tasks through openclaw\n"
	@printf "  download-bfcl-v4            Download BFCL v4 JSONL data to data/bfcl-v4/\n"
	@printf "  setup-bfcl-v4               Alias for download-bfcl-v4 (BFCL has no git repos to clone)\n"
	@printf "  smoke-bfcl-v4-openclaw      Run $(SMOKE_N) BFCL v4 tasks through openclaw (mini-swe-agent is unsupported for function_call shape)\n"
	@printf "  gantt-viewer-install        Install frontend dependencies with npm\n"
	@printf "  gantt-viewer-dev            Launch the new dynamic Gantt viewer CLI scaffold\n"
	@printf "  gantt-viewer-build          Build the frontend bundle\n"
	@printf "  gantt-viewer-test           Run migrated Gantt payload tests\n"
	@printf "  gantt-viewer-clean          Remove viewer build/cache artifacts\n"

pull:
	./scripts/pull_repo.sh

sync:
	test -x .venv/bin/python || $(UV) venv .venv
	$(UV) pip install --python .venv/bin/python -e ".[dev]"

build: sync

verify-bootstrap:
	$(PYTHON) -m pytest tests/test_bootstrap.py

verify-env1:
	$(PYTHON) -m pytest tests/test_env1.py

verify-env2:
	$(PYTHON) -m pytest tests/test_env2.py

verify-env3a:
	$(PYTHON) -m pytest tests/test_env3a.py

verify-env3b:
	$(PYTHON) -m pytest tests/test_env3b.py

verify-env3c:
	$(PYTHON) -m pytest tests/test_env3c.py

verify-env4:
	$(PYTHON) -m pytest tests/test_env4.py

verify-env5:
	$(PYTHON) -m pytest tests/test_env5.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

serve-vllm:
	./scripts/serve_vllm.sh

run-smoke:
	./scripts/run_smoke.sh

smoke-code:
	$(PYTHON) -m pytest tests/test_code_agent.py

run-sweep:
	./scripts/run_sweep.sh

collect-results:
	./scripts/collect_results.sh

setup-swebench-repos:
	./scripts/setup/clone_repos.sh data/swebench_verified/tasks.json

build-swebench-images:
	./scripts/setup/build_images.sh

download-swebench-verified:
	./scripts/setup/swebench_data.sh

download-swe-rebench:
	./scripts/setup/swe_rebench_data.sh

setup-swe-rebench-repos:
	./scripts/setup/clone_repos.sh data/swe-rebench/tasks.json data/swe-rebench/repos

setup-swe-rebench: download-swe-rebench setup-swe-rebench-repos

# ── SWE-rebench smoke runs ─────────────────────────────────────────────
# Override the provider with PROVIDER=openrouter (default: dashscope per the
# user's Phase 6 choice). Override the sample size with SMOKE_N=<n>
# (default: 2 — minimum meaningful smoke coverage per the plan).
PROVIDER ?= dashscope
SMOKE_N ?= 2

smoke-swe-rebench-miniswe:
	PYTHONPATH=src $(PYTHON) -m trace_collect.cli \
	    --provider $(PROVIDER) \
	    --benchmark swe-rebench \
	    --scaffold mini-swe-agent \
	    --sample $(SMOKE_N) \
	    --verbose

smoke-swe-rebench-openclaw:
	PYTHONPATH=src $(PYTHON) -m trace_collect.cli \
	    --provider $(PROVIDER) \
	    --benchmark swe-rebench \
	    --scaffold openclaw \
	    --sample $(SMOKE_N) \
	    --verbose

# ── BFCL v4 (Berkeley Function-Calling Leaderboard v4) ─────────────────
# BFCL v4 has task_shape='function_call' — no git repos, no docker.
# Only the openclaw scaffold is supported (mini-swe-agent is bash-in-repo
# and cannot emit structured function calls; BFCLv4Benchmark.build_runner
# raises NotImplementedError for mini-swe-agent).
download-bfcl-v4:
	./scripts/setup/bfcl_v4_data.sh

setup-bfcl-v4: download-bfcl-v4

smoke-bfcl-v4-openclaw:
	PYTHONPATH=src $(PYTHON) -m trace_collect.cli \
	    --provider $(PROVIDER) \
	    --benchmark bfcl-v4 \
	    --scaffold openclaw \
	    --sample $(SMOKE_N)

collect-traces:
	PYTHONPATH=src $(PYTHON) -m trace_collect.cli $(ARGS)

gantt-viewer-install:
	cd demo/gantt_viewer/frontend && npm install

gantt-viewer-dev:
	PYTHONPATH=src:. $(PYTHON) -m trace_collect.cli gantt-serve --dev

gantt-viewer-build:
	cd demo/gantt_viewer/frontend && npm run build

gantt-viewer-test:
	PYTHONPATH=src:. $(PYTHON) -m pytest demo/gantt_viewer/tests -v

gantt-viewer-clean:
	rm -rf demo/gantt_viewer/frontend/dist demo/gantt_viewer/frontend/node_modules
	rm -rf ~/.cache/agent-sched-bench/gantt-cc-import
