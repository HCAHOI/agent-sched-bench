PYTHON ?= python3
UV ?= uv

.PHONY: help pull sync build verify-bootstrap verify-env1 verify-env2 verify-env3a verify-env3b verify-env3c verify-env4 verify-env5 test lint serve-vllm run-smoke smoke-code smoke-data smoke-research run-sweep collect-results setup-swebench-repos build-swebench-images download-swebench-verified

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

smoke-data:
	$(PYTHON) -m pytest tests/test_data_agent.py

smoke-research:
	$(PYTHON) -m pytest tests/integration_research_agent_live.py

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

collect-traces:
	PYTHONPATH=src $(PYTHON) -m trace_collect.cli $(ARGS)
