# Recording Campaign Ledgers

This directory is `active-ledger`, not a generic runtime config tree.

Campaign YAML files here preserve pre-registered research questions, exact CLI invocations, model choices, recording switches, and analysis caveats. They may not be consumed by a runner directly; for example, `agent_event_kv_step_level.yaml` states that `trace_collect.cli` has no `--campaign` flag and that each arm documents the exact command to run.

## Policy

- Keep campaign ledgers versioned with the code that can reproduce or analyze them.
- Do not collapse arm-specific settings into code defaults; the manifest is part of the reproducibility record.
- If a campaign is superseded, add a note pointing to the replacement instead of deleting the file.
- When moving a campaign ledger, preserve exact paths referenced inside the invocations or update those invocations in the same reviewed change.

## Related active config

Campaign ledgers often reference active runtime config under:

- `configs/sparse_attention/`
- `configs/kv_policies/`
- `configs/mcp/`
- `configs/benchmarks/`
