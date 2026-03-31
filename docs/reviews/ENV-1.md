# ENV-1 Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d433b-2651-7c93-b5dd-189c9241cebc`
Scope: `scripts/setup_server.sh`, `scripts/report_server_env.py`,
`tests/test_env1.py`, and related README/Makefile updates

## Initial Findings

1. The bootstrap could report success on a non-target server.
2. The JSON audit report did not persist the repo-local venv and torch/CUDA
   runtime that was actually verified.
3. Version drift was insufficiently constrained or recorded.

## Fixes Applied

1. `setup_server.sh` now delegates pass/fail to `report_server_env.py` with
   explicit requirements for `A100-SXM-40GB`, at least `40 GiB` GPU memory,
   `CUDA >= 12.1`, and `torch.cuda.is_available()`.
2. The JSON report now records repo-local venv runtime metadata, torch version,
   CUDA runtime metadata, and the install configuration used during setup.
3. Default Python was pinned to `3.11.14`, and transient `uv.lock` output is no
   longer kept in the working tree.

## Verification

- `make verify-env1`
- `make verify-bootstrap`
- `python3 -m compileall scripts tests`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py`

## Residual Caveat

The code checkpoint is approved, but the real operational acceptance still
requires running the bootstrap on the approved Ubuntu+A100 server.

## Final Verdict

Approved. No remaining material code issues.
