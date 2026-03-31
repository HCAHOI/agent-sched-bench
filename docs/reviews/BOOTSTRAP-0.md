# BOOTSTRAP-0 Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d4328-87f9-7ee2-bcac-9d05fedc31f9`
Scope: repository bootstrap scaffold, configs, scripts, and bootstrap tests

## Initial Findings

1. `Makefile` sync path was not repo-local or idempotent.
2. Primary workload configs defaulted to reduced sample sizes.
3. `Makefile` targets diverged from the benchmark spec.
4. Placeholder scripts succeeded instead of failing closed.

## Fixes Applied

1. `make sync` now creates or reuses `.venv` and installs via
   `uv pip install --python .venv/bin/python -e ".[dev]"`.
2. Primary workload configs now default to full datasets, and dedicated
   `*_smoke.yaml` files hold reduced smoke-only subsets.
3. Added `pull` and `run-smoke` targets, and aligned `serve-vllm` with an
   explicit BOOTSTRAP-0 stub script.
4. Placeholder scripts now print guidance to stderr and exit non-zero until
   their checkpoint is implemented.
5. Expanded bootstrap tests to guard the Makefile contract and full-dataset
   defaults.

## Verification

- `python3 -m pytest tests/test_bootstrap.py`
- `python3 -m compileall src tests`
- `make -n sync`
- `make sync`
- `make verify-bootstrap`

## Final Verdict

Approved. No remaining material BOOTSTRAP-0 issues.
