# OpenRouter Provider And Haiku Rerun

## Summary

- Centralize `trace_collect` provider presets so CLI defaults and scaffold runtime behavior share one source of truth.
- Make `miniswe` explicitly provider-aware so OpenRouter models do not get forced through the `openai/...` LiteLLM prefix.
- Re-run the 5 successful `swe-rebench` OpenClaw smoke tasks on OpenRouter model `anthropic/claude-haiku-4.5`.

## Work Items

1. Extract shared provider preset and resolution helpers under `src/trace_collect/`.
2. Thread `provider_name` through `trace_collect` into the MiniSWE host and task-container runtime paths.
3. Update `MiniSWECodeAgent` to derive the LiteLLM model identifier from provider metadata instead of hardcoding `openai/`.
4. Add focused tests for provider resolution and task-container request plumbing.
5. Run targeted tests for CLI/provider/collector runtime coverage.
6. Run the 5-task OpenClaw OpenRouter smoke with `--max-iterations 100`.

## Acceptance Checks

- `--provider openrouter` resolves API base and env var defaults without reading `.bashrc`.
- `miniswe` builds an OpenRouter-compatible LiteLLM model name for OpenRouter runs.
- The 5-task OpenClaw rerun emits canonical v5 traces under a fresh run directory.
- Normal attempt artifacts exist and disk cleanup remains bounded after the run.
