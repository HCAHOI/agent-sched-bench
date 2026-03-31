# ENV-3b Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d4393-5d95-7d12-9f7c-3e3d66cb2c79`
Scope: Continuum launcher, Continuum serve script, shared health-check changes,
`tests/test_env3b.py`, and related config/docs/env updates

## Initial Findings

1. The Continuum launcher depended on a PATH-resolved `vllm` binary.
2. Continuum acceptance could false-positive without a Continuum-specific reuse
   signal.
3. Reproducibility still depended on a floating upstream branch.

## Fixes Applied

1. `serving.continuum_launcher` now resolves `vllm` from the same venv as
   `sys.executable`.
2. `serve_continuum.sh` now requires an immutable `CONTINUUM_REF` and enables a
   repeated `program_id` check with `--require-prefix-cache-hit`.
3. `serving.health_check` now performs true multi-turn repeated requests and
   validates a positive prefix-cache-hit-rate delta during the run.
4. `tests/test_env3b.py` now guards pinned-ref enforcement and the
   prefix-cache-hit acceptance path.

## Verification

- `make verify-env3b`
- `make verify-env3a`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py`
- `python3 -m compileall src scripts tests`

## Residual Caveat

The code checkpoint is approved, but real acceptance still requires a server run
with the actual Continuum runtime and the pinned upstream ref.

## Final Verdict

Approved. No remaining material code issues.
