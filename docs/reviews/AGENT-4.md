# AGENT-4 Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d4403-bf58-7240-8089-d64e0bfa94fb`
Scope: `src/agents/research_agent.py`, `tests/test_research_agent.py`,
`tests/integration_research_agent_live.py`, and the `smoke-research` Make target

## Initial Findings

1. `reference_answer` leaked oracle information into the prompt.
2. The real DuckDuckGo search path lacked the headers and URL normalization
   needed for live use.
3. The smoke path used a fully local simulation instead of a live search/page
   flow.
4. The live smoke still had a hardcoded fallback URL that masked broken search
   output.

## Fixes Applied

1. `reference_answer` was removed from the runtime prompt path.
2. `ResearchAgent` now uses a DuckDuckGo-compatible User-Agent and normalizes /
   unwraps result URLs before `page_read()`.
3. The unit test remains local, but `smoke-research` now runs a separate live
   network smoke that hits the real DuckDuckGo HTML endpoint and real pages.
4. The live smoke now fails explicitly if no normalized `https://...` URL is
   present in the latest tool-output-derived user message.

## Verification

- `python3 -m pytest tests/test_research_agent.py`
- `make smoke-research`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py tests/test_env3c.py tests/test_env4.py tests/test_env5.py tests/test_agent_basic.py tests/test_code_agent.py tests/test_data_agent.py tests/test_research_agent.py -q`
- `python3 -m compileall src tests`

## Residual Caveat

The live smoke depends on DuckDuckGo HTML responses and real web pages, so it is
subject to external network/site variability.

## Final Verdict

Approved. No remaining material code issues.
