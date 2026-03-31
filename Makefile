PYTHON ?= python3
UV ?= uv

.PHONY: help pull sync verify-bootstrap verify-env1 verify-env2 verify-env3a verify-env3b test lint serve-vllm run-smoke smoke-code smoke-data smoke-research run-sweep collect-results

help:
	@printf "Targets:\n"
	@printf "  pull              Fast-forward pull the current branch\n"
	@printf "  sync              Install dependencies with uv\n"
	@printf "  verify-bootstrap  Run BOOTSTRAP-0 verification\n"
	@printf "  verify-env1       Run ENV-1 static verification\n"
	@printf "  verify-env2       Run ENV-2 static verification\n"
	@printf "  verify-env3a      Run ENV-3a static verification\n"
	@printf "  verify-env3b      Run ENV-3b static verification\n"
	@printf "  test              Run the full test suite\n"
	@printf "  lint              Run ruff\n"
	@printf "  serve-vllm        Stub until ENV-3a implements serving launch\n"
	@printf "  run-smoke         Stub until agent smoke checkpoints are implemented\n"
	@printf "  smoke-code        Reserved for AGENT-2 smoke test\n"
	@printf "  smoke-data        Reserved for AGENT-3 smoke test\n"
	@printf "  smoke-research    Reserved for AGENT-4 smoke test\n"
	@printf "  run-sweep         Reserved for HARNESS-1 end-to-end runs\n"
	@printf "  collect-results   Collect benchmark artifacts\n"

pull:
	git pull --ff-only

sync:
	test -x .venv/bin/python || $(UV) venv .venv
	$(UV) pip install --python .venv/bin/python -e ".[dev]"

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

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

serve-vllm:
	./scripts/serve_vllm.sh

run-smoke:
	./scripts/run_smoke.sh

smoke-code:
	@printf "AGENT-2 not implemented yet. Use after checkpoint approval.\n" >&2
	@exit 1

smoke-data:
	@printf "AGENT-3 not implemented yet. Use after checkpoint approval.\n" >&2
	@exit 1

smoke-research:
	@printf "AGENT-4 not implemented yet. Use after checkpoint approval.\n" >&2
	@exit 1

run-sweep:
	./scripts/run_sweep.sh

collect-results:
	./scripts/collect_results.sh
