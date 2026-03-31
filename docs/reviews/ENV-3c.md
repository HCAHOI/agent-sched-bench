# ENV-3c Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d439e-9b76-7350-8da7-d82d9f727629`
Scope: ThunderAgent launcher, ThunderAgent proxy checks, `tests/test_env3c.py`,
and related config/docs/env updates

## Initial Findings

1. Pinned-ref enforcement did not actually guarantee an immutable ref.
2. The ThunderAgent acceptance path could pass without proving both turns were
   tracked in the same live program.
3. The ThunderAgent config hardcoded the metrics URL to port 9000.

## Fixes Applied

1. `serve_thunderagent.sh` now requires `THUNDERAGENT_REF` to be either a full
   commit SHA or an existing tag after fetch.
2. The default `program_id` is unique per run, and
   `serving.thunderagent_check` now snapshots `/programs` before traffic and
   requires `step_count` to increase by at least 2 during the run.
3. `thunderagent.yaml` now takes `metrics_url` from `THUNDERAGENT_METRICS_URL`.

## Verification

- `make verify-env3c`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py tests/test_env3c.py`
- `python3 -m compileall src scripts tests`

## Residual Caveat

The code checkpoint is approved, but actual acceptance still requires an
approved-server run against the real ThunderAgent runtime and pinned upstream
ref.

## Final Verdict

Approved. No remaining material code issues.
